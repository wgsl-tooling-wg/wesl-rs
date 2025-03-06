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
    /// Convert a source file into a syntax tree.
    fn source_to_module(
        &self,
        source: &str,
        path: &ModulePath,
    ) -> Result<TranslationUnit, ResolveError> {
        let wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_module_path(path.clone(), self.display_name(path))
                .with_source(source.to_string())
        })?;
        Ok(wesl)
    }
    /// Try to resolve a source file identified by a module path.
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        let source = self.resolve_source(path)?;
        let wesl = self.source_to_module(&source, path)?;
        Ok(wesl)
    }
    /// Get the display name of the module path. Implementing this is optional.
    fn display_name(&self, _path: &ModulePath) -> Option<String> {
        None
    }
}

impl<T: Resolver + ?Sized> Resolver for Box<T> {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        (**self).resolve_source(path)
    }
    fn source_to_module(
        &self,
        source: &str,
        path: &ModulePath,
    ) -> Result<TranslationUnit, ResolveError> {
        (**self).source_to_module(source, path)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        (**self).resolve_module(path)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        (**self).display_name(path)
    }
}

impl<T: Resolver> Resolver for &T {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        (**self).resolve_source(path)
    }
    fn source_to_module(
        &self,
        source: &str,
        path: &ModulePath,
    ) -> Result<TranslationUnit, ResolveError> {
        (**self).source_to_module(source, path)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        (**self).resolve_module(path)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        (**self).display_name(path)
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
    pub fn add_module(&mut self, path: impl Into<ModulePath>, file: Cow<'a, str>) {
        let mut path = path.into();
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
    fn source_to_module(
        &self,
        source: &str,
        path: &ModulePath,
    ) -> Result<TranslationUnit, ResolveError> {
        let mut wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_module_path(path.clone(), self.display_name(path))
                .with_source(source.to_string())
        })?;
        (self.preprocess)(&mut wesl).map_err(|e| {
            Diagnostic::from(e)
                .with_module_path(path.clone(), self.display_name(path))
                .with_source(source.to_string())
        })?;
        Ok(wesl)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        self.resolver.display_name(path)
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
    pub fn mount_resolver(
        &mut self,
        path: impl Into<ModulePath>,
        resolver: impl Resolver + 'static,
    ) {
        let path = path.into();
        let resolver: Box<dyn Resolver> = Box::new(resolver);
        if path.is_empty() {
            // when the path is empty, the resolver would match any path anyways.
            // (except external packages)
            self.fallback = Some((path, resolver));
        } else {
            self.mount_points.push((path, resolver));
        }
    }

    /// Mount a fallback resolver that is used when no other prefix match.
    pub fn mount_fallback_resolver(&mut self, resolver: impl Resolver + 'static) {
        self.mount_resolver("", resolver);
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
    fn source_to_module(
        &self,
        source: &str,
        path: &ModulePath,
    ) -> Result<TranslationUnit, ResolveError> {
        let (resolver, path) = self.route(path)?;
        resolver.source_to_module(source, &path)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        let (resolver, path) = self.route(path)?;
        resolver.resolve_module(&path)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        let (resolver, path) = self.route(path).ok()?;
        resolver.display_name(&path)
    }
}

/// The trait implemented by external packages.
///
/// You typically don't implement this, instead it is implemented for you by the
/// [`crate::PkgBuilder`].
pub trait PkgModule: Send + Sync {
    fn name(&self) -> &'static str;
    fn source(&self) -> &'static str;
    fn submodules(&self) -> &[&dyn PkgModule];
    fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
        self.submodules()
            .iter()
            .find(|sm| sm.name() == name)
            .copied()
    }
}

/// A resolver that only resolves module paths that refer to modules in external packages.
///
/// Register external packages with [`Self::add_package`].
pub struct PkgResolver {
    packages: Vec<&'static dyn PkgModule>,
}

impl PkgResolver {
    /// Create a new resolver.
    pub fn new() -> Self {
        Self {
            packages: Vec::new(),
        }
    }

    /// Add a package to the resolver.
    pub fn add_package(&mut self, pkg: &'static dyn PkgModule) {
        self.packages.push(pkg);
    }
}

impl Default for PkgResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolver for PkgResolver {
    fn resolve_source<'a>(
        &'a self,
        path: &ModulePath,
    ) -> Result<std::borrow::Cow<'a, str>, ResolveError> {
        for pkg in &self.packages {
            // TODO: the resolution algorithm is currently not spec-compliant.
            // https://github.com/wgsl-tooling-wg/wesl-spec/blob/imports-update/Imports.md
            if path.origin.is_package()
                && path.components.first().map(String::as_str) == Some(pkg.name())
            {
                let mut cur_mod = *pkg;
                for comp in path.components.iter().skip(1) {
                    if let Some(submod) = pkg.submodule(comp) {
                        cur_mod = submod;
                    } else {
                        return Err(E::ModuleNotFound(
                            path.clone(),
                            format!(
                                "in module `{}`, no submodule named `{comp}`",
                                cur_mod.name()
                            ),
                        ));
                    }
                }
                return Ok(cur_mod.source().into());
            }
        }
        Err(E::ModuleNotFound(
            path.clone(),
            "no package found".to_string(),
        ))
    }
}

/// The resolver that implements the WESL standard.
///
/// It resolves modules in external packages registered with [`Self::add_package`] and
/// modules in the local package with the filesystem.
pub struct StandardResolver {
    pkg: PkgResolver,
    files: FileResolver,
}

impl StandardResolver {
    /// Create a new resolver.
    ///
    /// `base` is the root directory which absolute paths refer to.
    pub fn new(base: impl AsRef<Path>) -> Self {
        Self {
            pkg: PkgResolver::new(),
            files: FileResolver::new(base),
        }
    }

    /// Add an external package.
    pub fn add_package(&mut self, pkg: &'static dyn PkgModule) {
        self.pkg.add_package(pkg)
    }
}

impl Resolver for StandardResolver {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
        if path.origin.is_package() {
            self.pkg.resolve_source(path)
        } else {
            self.files.resolve_source(path)
        }
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        if path.origin.is_package() {
            self.pkg.resolve_module(path)
        } else {
            self.files.resolve_module(path)
        }
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        if path.origin.is_package() {
            self.pkg.display_name(path)
        } else {
            self.files.display_name(path)
        }
    }
}
