use crate::{Diagnostic, Error};

use itertools::Itertools;
use wgsl_parse::syntax::{ModulePath, PathOrigin, TranslationUnit};

use std::{
    borrow::Cow,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

/// Error produced by module resolution.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("file not found: `{0}` ({1})")]
    FileNotFound(PathBuf, String),
    #[error("module not found: `{0}` ({1})")]
    ModuleNotFound(ModulePath, String),
    #[error("{0}")]
    Error(#[from] Diagnostic<Error>),
}

type E = ResolveError;

/// A Resolver implements the module resolution algorithm: it returns a module contents
/// associated with a module path.
///
/// Typically implementations of [`Resolver`] only implement [`Resolver::resolve_source`].
///
/// Calls to `Resolver` functions must respect these preconditions:
/// * the import path must not be relative.
pub trait Resolver {
    /// Try to resolve a source file identified by a module path.
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError>;
    /// Try to resolve a source file identified by a module path.
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        let source = self.resolve_source(path)?;
        let wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_module_path(path.clone(), self.display_name(path))
                .with_source(source.to_string())
        })?;
        Ok(wesl)
    }
    /// Get the display name of the module path. Implementing this is optional.
    fn display_name(&self, _path: &ModulePath) -> Option<String> {
        None
    }
    /// Get the filesystem path of the module path. Implementing this is optional.
    /// Used by build scripts for dependency tracking.
    fn fs_path(&self, _path: &ModulePath) -> Option<PathBuf> {
        None
    }
}

impl<T: Resolver + ?Sized> Resolver for Box<T> {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        (**self).resolve_source(path)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        (**self).resolve_module(path)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        (**self).display_name(path)
    }
    fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
        (**self).fs_path(path)
    }
}

impl<T: Resolver> Resolver for &T {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        (**self).resolve_source(path)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        (**self).resolve_module(path)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        (**self).display_name(path)
    }
    fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
        (**self).fs_path(path)
    }
}

/// A resolver that never resolves anything.
///
/// Returns [`ResolveError::ModuleNotFound`] when calling [`Resolver::resolve_source`].
#[derive(Default, Clone, Debug)]
pub struct NoResolver;

impl Resolver for NoResolver {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        Err(E::ModuleNotFound(
            path.clone(),
            "no module resolver, imports are effectively disabled here".to_string(),
        ))
    }
}

/// A resolver that looks for files in the filesystem.
///
/// It simply translates module paths to file paths. This is the intended behavior.
#[derive(Default)]
pub struct FileResolver {
    base: PathBuf,
    extension: &'static str,
}

impl FileResolver {
    /// Create a new resolver.
    ///
    /// `base` is the root directory which absolute paths refer to.
    pub fn new(base: impl AsRef<Path>) -> Self {
        Self {
            base: base.as_ref().to_path_buf(),
            extension: "wesl",
        }
    }

    /// Look for files that ends with a different extension. Default: "wesl".
    pub fn set_extension(&mut self, extension: &'static str) {
        self.extension = extension;
    }

    fn file_path(&self, path: &ModulePath) -> Result<PathBuf, ResolveError> {
        if path.origin.is_package() {
            return Err(E::ModuleNotFound(
                path.clone(),
                "this is an external package import, not a file import. Use `package::` or `super::` for file imports."
                    .to_string(),
            ));
        }
        let mut fs_path = self.base.to_path_buf();
        fs_path.extend(&path.components);
        fs_path.set_extension(self.extension);
        if fs_path.exists() {
            Ok(fs_path)
        } else {
            fs_path.set_extension("wgsl");
            if fs_path.exists() {
                Ok(fs_path)
            } else {
                Err(E::FileNotFound(fs_path, "physical file".to_string()))
            }
        }
    }
}

impl Resolver for FileResolver {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        let fs_path = self.file_path(path)?;
        let source = fs::read_to_string(&fs_path)
            .map_err(|_| E::FileNotFound(fs_path, "physical file".to_string()))?;

        Ok(source.into())
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        self.file_path(path)
            .ok()
            .map(|fs_path| fs_path.display().to_string())
    }
    fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
        self.file_path(path).ok()
    }
}

/// A resolver that resolves in-memory modules added with [`Self::add_module`].
///
/// Use-cases are platforms that lack a filesystem (e.g. WASM), tests or
/// runtime-generated files.
#[derive(Default)]
pub struct VirtualResolver<'a> {
    files: HashMap<ModulePath, Cow<'a, str>>,
}

impl<'a> VirtualResolver<'a> {
    /// Create a new resolver.
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// Resolve imports of `path` with the given WESL string.
    pub fn add_module(&mut self, mut path: ModulePath, file: Cow<'a, str>) {
        path.origin = PathOrigin::Absolute; // we force absolute paths
        self.files.insert(path, file);
    }

    /// Get a module registered with [`Self::add_module`].
    pub fn get_module(&self, path: &ModulePath) -> Result<&str, ResolveError> {
        let source = self
            .files
            .get(path)
            .ok_or_else(|| E::ModuleNotFound(path.clone(), "virtual module".to_string()))?;
        Ok(source)
    }

    /// Iterate over all registered modules.
    pub fn modules(&self) -> impl Iterator<Item = (&ModulePath, &str)> {
        self.files.iter().map(|(res, file)| (res, &**file))
    }
}

impl Resolver for VirtualResolver<'_> {
    fn resolve_source<'b>(&'b self, path: &ModulePath) -> Result<Cow<'b, str>, ResolveError> {
        let source = self.get_module(path)?;
        Ok(source.into())
    }
}

// trait alias
pub trait ResolveFn: Fn(&mut TranslationUnit) -> Result<(), Error> {}
impl<T: Fn(&mut TranslationUnit) -> Result<(), Error>> ResolveFn for T {}

/// A WESL module preprocessor.
///
/// The preprocess function will be called each time the WESL compiler tries to load a
/// module.
pub struct Preprocessor<R: Resolver, F: ResolveFn> {
    pub resolver: R,
    pub preprocess: F,
}

impl<R: Resolver, F: ResolveFn> Preprocessor<R, F> {
    /// Create a new resolver that runs the preprocessing function before each call to
    /// [`Resolver::resolve_module`].
    pub fn new(resolver: R, preprocess: F) -> Self {
        Self {
            resolver,
            preprocess,
        }
    }
}

impl<R: Resolver, F: ResolveFn> Resolver for Preprocessor<R, F> {
    fn resolve_source<'b>(&'b self, path: &ModulePath) -> Result<Cow<'b, str>, ResolveError> {
        let res = self.resolver.resolve_source(path)?;
        Ok(res)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        let mut wesl = self.resolver.resolve_module(path)?;
        (self.preprocess)(&mut wesl).map_err(|e| {
            Diagnostic::from(e)
                .with_module_path(path.clone(), self.display_name(path))
                .with_source(self.resolve_source(path).unwrap().to_string())
        })?;
        Ok(wesl)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        self.resolver.display_name(path)
    }

    fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
        self.resolver.fs_path(path)
    }
}

/// A resolver that can dispatch imports to several sub-resolvers based on the import
/// path prefix.
///
/// Add sub-resolvers with [`Self::mount_resolver`].
///
/// This resolver is not thread-safe (not [`Send`] or [`Sync`]).
pub struct Router {
    mount_points: Vec<(ModulePath, Box<dyn Resolver>)>,
    fallback: Option<(ModulePath, Box<dyn Resolver>)>,
}

/// Dispatches resolution of a module path to sub-resolvers.
impl Router {
    /// Create a new resolver.
    pub fn new() -> Self {
        Self {
            mount_points: Vec::new(),
            fallback: None,
        }
    }

    /// Mount a resolver at a given path prefix. All imports that start with this prefix
    /// will be dispatched to that resolver with the suffix of the path.
    pub fn mount_resolver(&mut self, path: ModulePath, resolver: impl Resolver + 'static) {
        self.mount_points.push((path, Box::new(resolver)));
    }

    /// Mount a fallback resolver that is used when no other prefix match.
    pub fn mount_fallback_resolver(&mut self, resolver: impl Resolver + 'static) {
        self.fallback = Some((ModulePath::new_root(), Box::new(resolver)));
    }

    fn route(&self, path: &ModulePath) -> Result<(&dyn Resolver, ModulePath), ResolveError> {
        let (mount_path, resolver) = self
            .mount_points
            .iter()
            .filter(|(prefix, _)| path.starts_with(prefix))
            .max_by_key(|(prefix, _)| prefix.components.len())
            .or(self.fallback.as_ref())
            .ok_or_else(|| E::ModuleNotFound(path.clone(), "no mount point".to_string()))?;

        let components = path
            .components
            .iter()
            .skip(mount_path.components.len())
            .cloned()
            .collect_vec();
        let suffix = ModulePath::new(PathOrigin::Absolute, components);
        Ok((resolver, suffix))
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolver for Router {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        let (resolver, path) = self.route(path)?;
        resolver.resolve_source(&path)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        let (resolver, path) = self.route(path)?;
        resolver.resolve_module(&path)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        let (resolver, path) = self.route(path).ok()?;
        resolver.display_name(&path)
    }
    fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
        let (resolver, path) = self.route(path).ok()?;
        resolver.fs_path(&path)
    }
}

/// The type holding the source code of external packages.
///
/// You typically don't implement this, instead it is generated for you by [`crate::PkgBuilder`].
/// Crates containing shader packages export `const` instances of this type, which you can
/// then import and [add to your resolver][StandardResolver::add_package].
#[derive(Debug, PartialEq, Eq)]
pub struct CodegenPkg {
    pub crate_name: &'static str,
    pub root: &'static CodegenModule,
    pub dependencies: &'static [&'static CodegenPkg],
}

/// The type holding the source code of modules in external packages.
///
/// See [`CodegenPkg`].
#[derive(Debug, PartialEq, Eq)]
pub struct CodegenModule {
    pub name: &'static str,
    pub source: &'static str,
    pub submodules: &'static [&'static CodegenModule],
}

/// A resolver that only resolves module paths that refer to modules in external packages.
///
/// Register external packages with [`Self::add_package`].
pub struct PkgResolver {
    packages: Vec<&'static CodegenPkg>,
}

impl PkgResolver {
    /// Create a new resolver.
    pub fn new() -> Self {
        Self {
            packages: Vec::new(),
        }
    }

    /// Add a package to the resolver.
    pub fn add_package(&mut self, pkg: &'static CodegenPkg) {
        self.packages.push(pkg);
    }
}

impl Default for PkgResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolver for PkgResolver {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<std::borrow::Cow<'a, str>, E> {
        // This is a hack: when the package name contains `/`, it corresponds to a sub-dependency
        // of a package dependency. The name is created by the import resolution algorithm.
        // (see import.rs:join_paths)
        let pkg_path = match &path.origin {
            PathOrigin::Package(pkg) => pkg,
            _ => {
                return Err(E::ModuleNotFound(
                    path.clone(),
                    "resolver can only resolve package imports".to_string(),
                ));
            }
        };

        let pkg_parts = pkg_path.split('/').collect_vec();

        let root_pkg = pkg_parts
            .first()
            .and_then(|name| self.packages.iter().find(|p| p.root.name == *name))
            .ok_or_else(|| {
                E::ModuleNotFound(
                    path.clone(),
                    format!("dependency `{}` not found", pkg_parts.iter().format("/"),),
                )
            })?;

        let pkg = pkg_parts.iter().skip(1).try_fold(root_pkg, |dep, name| {
            dep.dependencies
                .iter()
                .find(|p| p.root.name == *name)
                .ok_or_else(|| {
                    E::ModuleNotFound(
                        path.clone(),
                        format!(
                            "dependency `{}` not found in package path `{}`",
                            name,
                            pkg_parts.iter().format("/"),
                        ),
                    )
                })
        })?;

        // TODO: the resolution algorithm is currently not spec-compliant.
        // https://github.com/wgsl-tooling-wg/wesl-spec/blob/imports-update/Imports.md
        let mut cur_mod = pkg.root;
        for comp in &path.components {
            if let Some(submod) = cur_mod.submodules.iter().find(|m| m.name == comp) {
                cur_mod = submod;
            } else {
                return Err(E::ModuleNotFound(
                    path.clone(),
                    format!("in module `{}`, no submodule named `{comp}`", cur_mod.name),
                ));
            }
        }
        Ok(cur_mod.source.into())
    }
}

/// The resolver that implements the WESL standard.
///
/// It resolves modules in external packages registered with [`Self::add_package`] and
/// modules in the local package with the filesystem.
pub struct StandardResolver {
    pkg: PkgResolver,
    files: FileResolver,
    constants: HashMap<String, f64>,
}

impl StandardResolver {
    /// Create a new resolver.
    ///
    /// `base` is the root directory which absolute paths refer to.
    pub fn new(base: impl AsRef<Path>) -> Self {
        Self {
            pkg: PkgResolver::new(),
            files: FileResolver::new(base),
            constants: HashMap::new(),
        }
    }

    /// Add an external package.
    pub fn add_package(&mut self, pkg: &'static CodegenPkg) {
        self.pkg.add_package(pkg)
    }

    /// Add a numeric constant.
    ///
    /// Numeric constants live WESL's special package named `constants`. This package is
    /// *virtual*, meaning it doesn't exist on the filesystem. Constants can be accessed
    /// by importing them: `import constants::MY_CONSTANT;`. All constants are of type
    /// AbstractFloat, which can be implicitly converted to all scalar types.
    pub fn add_constant(&mut self, name: impl ToString, value: f64) {
        self.constants.insert(name.to_string(), value);
    }

    fn generate_constant_module(&self) -> String {
        self.constants
            .iter()
            .map(|(name, value)| format!("const {name} = {value};"))
            .format("\n")
            .to_string()
    }
}

impl Resolver for StandardResolver {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        // a special case to handle the constants virtual module. For now, this module
        // is shared for all sub-dependencies.
        // TODO: in the future we'll change that.
        if let PathOrigin::Package(pkg_name) = &path.origin {
            if pkg_name == "constants" || pkg_name.ends_with("/constants") {
                return Ok(self.generate_constant_module().into());
            }
        }

        if path.origin.is_package() {
            self.pkg.resolve_source(path)
        } else {
            self.files.resolve_source(path)
        }
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        if path.origin.is_package() {
            self.pkg.display_name(path)
        } else {
            self.files.display_name(path)
        }
    }
    fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
        if path.origin.is_package() {
            self.pkg.fs_path(path)
        } else {
            self.files.fs_path(path)
        }
    }
}

pub fn emit_rerun_if_changed(modules: &[ModulePath], resolver: &impl Resolver) {
    for module in modules {
        if module.origin.is_package() {
            continue;
        }
        assert!(
            !module.origin.is_relative(),
            "the modules passed to emit_rerun_if_changed must be absolute"
        );
        println!("cargo::rerun-if-changed=build.rs");
        if let Some(mut path) = resolver.fs_path(module) {
            // Path::display is safe here because of the ModulePath naming restrictions
            println!("cargo::rerun-if-changed={}", path.display());

            // If it's a fallback path, we need to react to the higher priority path as well
            if path.extension().unwrap() == "wgsl" {
                path.set_extension("wesl");
                println!("cargo::rerun-if-changed={}", path.display());
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn router_resolver() {
        let mut r = Router::new();

        let mut v1 = VirtualResolver::new();
        v1.add_module("package".parse().unwrap(), "m1".into());
        v1.add_module("package::foo".parse().unwrap(), "m2".into());
        v1.add_module("package::bar".parse().unwrap(), "m3".into());
        r.mount_resolver("package".parse().unwrap(), v1);

        let mut v2 = VirtualResolver::new();
        v2.add_module("package".parse().unwrap(), "m4".into());
        v2.add_module("package::baz".parse().unwrap(), "m5".into());
        r.mount_resolver("package::bar".parse().unwrap(), v2);

        let mut v3 = VirtualResolver::new();
        v3.add_module("package::bar".parse().unwrap(), "m6".into());
        r.mount_fallback_resolver(v3);

        assert_eq!(r.resolve_source(&"package".parse().unwrap()).unwrap(), "m1");
        assert_eq!(
            r.resolve_source(&"package::foo".parse().unwrap()).unwrap(),
            "m2"
        );
        assert_eq!(
            r.resolve_source(&"package::bar".parse().unwrap()).unwrap(),
            "m4"
        );
        assert_eq!(
            r.resolve_source(&"package::bar::baz".parse().unwrap())
                .unwrap(),
            "m5"
        );
        assert_eq!(
            r.resolve_source(&"foo::bar".parse().unwrap()).unwrap(),
            "m6"
        );
    }
}
