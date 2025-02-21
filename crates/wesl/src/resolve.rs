use crate::{Diagnostic, Error, SyntaxUtil};

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
    #[error("invalid import path: `{0}` ({1})")]
    InvalidResource(ModulePath, String),
    #[error("file not found: `{0}` ({1})")]
    FileNotFound(PathBuf, String),
    #[error("module not found: `{0}` ({1})")]
    ModuleNotFound(ModulePath, String),
    #[error("{0}")]
    Error(#[from] Diagnostic<Error>),
}

type E = ResolveError;

/// A Resolver is responsible for turning an import path into a unique module path
/// ([`Resource`]) and providing the source file and syntax tree.
///
/// Typically implementations of [`Resolver`] only implement [`Resolver::resolve_source`].
///
/// Calls to `Resolver` functions must respect these preconditions:
/// * the resource must be canonical (absolute module path).
pub trait Resolver {
    /// Try to resolve a source file identified by a resource.
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, E>;
    /// Convert a source file into a syntax tree.
    fn source_to_module(&self, source: &str, path: &ModulePath) -> Result<TranslationUnit, E> {
        let mut wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_resource(path.clone(), self.display_name(path))
                .with_source(source.to_string())
        })?;
        wesl.retarget_idents(); // it's important to call that early on to have identifiers point at the right declaration.
        Ok(wesl)
    }
    /// Try to resolve a source file identified by a resource.
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, E> {
        let source = self.resolve_source(path)?;
        let wesl = self.source_to_module(&source, path)?;
        Ok(wesl)
    }
    /// Get the display name of the resource. Implementing this is optional.
    fn display_name(&self, _path: &ModulePath) -> Option<String> {
        None
    }
}

impl<T: Resolver + ?Sized> Resolver for Box<T> {
    fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, E> {
        (**self).resolve_source(path)
    }
    fn source_to_module(&self, source: &str, path: &ModulePath) -> Result<TranslationUnit, E> {
        (**self).source_to_module(source, path)
    }
    fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, E> {
        (**self).resolve_module(path)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        (**self).display_name(path)
    }
}

impl<T: Resolver> Resolver for &T {
    fn resolve_source<'a>(&'a self, resource: &ModulePath) -> Result<Cow<'a, str>, E> {
        (**self).resolve_source(resource)
    }
    fn source_to_module(&self, source: &str, resource: &ModulePath) -> Result<TranslationUnit, E> {
        (**self).source_to_module(source, resource)
    }
    fn resolve_module(&self, resource: &ModulePath) -> Result<TranslationUnit, E> {
        (**self).resolve_module(resource)
    }
    fn display_name(&self, resource: &ModulePath) -> Option<String> {
        (**self).display_name(resource)
    }
}

/// A resolver that never resolves anything.
///
/// Returns [`ResolveError::InvalidResource`] when calling [`Resolver::resolve_source`].
#[derive(Default, Clone, Debug)]
pub struct NoResolver;

impl Resolver for NoResolver {
    fn resolve_source<'a>(&'a self, resource: &ModulePath) -> Result<Cow<'a, str>, E> {
        Err(E::InvalidResource(
            resource.clone(),
            "no resolver".to_string(),
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

    fn file_path(&self, resource: &ModulePath) -> Result<PathBuf, E> {
        if resource.origin.is_package() {
            return Err(E::InvalidResource(
                resource.clone(),
                "this is an external package import, not a file import. Use `package::` or `super::` for file imports."
                    .to_string(),
            ));
        }
        let mut path = self.base.to_path_buf();
        path.extend(&resource.components);
        path.set_extension(self.extension);
        if path.exists() {
            Ok(path)
        } else {
            path.set_extension("wgsl");
            if path.exists() {
                Ok(path)
            } else {
                Err(E::FileNotFound(path, "physical file".to_string()))
            }
        }
    }
}

impl Resolver for FileResolver {
    fn resolve_source<'a>(&'a self, resource: &ModulePath) -> Result<Cow<'a, str>, E> {
        let path = self.file_path(resource)?;
        let source = fs::read_to_string(&path)
            .map_err(|_| E::FileNotFound(path, "physical file".to_string()))?;

        Ok(source.into())
    }
    fn display_name(&self, resource: &ModulePath) -> Option<String> {
        self.file_path(resource)
            .ok()
            .map(|path| path.display().to_string())
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
    pub fn add_module(&mut self, path: impl AsRef<Path>, file: Cow<'a, str>) {
        let mut path = ModulePath::from_path(path);
        path.origin = PathOrigin::Absolute; // we force absolute paths
        self.files.insert(path, file);
    }

    /// Get a module registered with [`Self::add_module`].
    pub fn get_module(&self, resource: &ModulePath) -> Result<&str, E> {
        let source = self
            .files
            .get(resource)
            .ok_or_else(|| E::ModuleNotFound(resource.clone(), "virtual module".to_string()))?;
        Ok(source)
    }

    /// Iterate over all registered modules.
    pub fn modules(&self) -> impl Iterator<Item = (&ModulePath, &str)> {
        self.files.iter().map(|(res, file)| (res, &**file))
    }
}

impl Resolver for VirtualResolver<'_> {
    fn resolve_source<'b>(&'b self, resource: &ModulePath) -> Result<Cow<'b, str>, E> {
        let source = self.get_module(resource)?;
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
    fn resolve_source<'b>(&'b self, resource: &ModulePath) -> Result<Cow<'b, str>, E> {
        let res = self.resolver.resolve_source(resource)?;
        Ok(res)
    }
    fn source_to_module(&self, source: &str, resource: &ModulePath) -> Result<TranslationUnit, E> {
        let mut wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_resource(resource.clone(), self.display_name(resource))
                .with_source(source.to_string())
        })?;
        wesl.retarget_idents(); // it's important to call that early on to have identifiers point at the right declaration.
        (self.preprocess)(&mut wesl).map_err(|e| {
            Diagnostic::from(e)
                .with_resource(resource.clone(), self.display_name(resource))
                .with_source(source.to_string())
        })?;
        Ok(wesl)
    }
    fn display_name(&self, resource: &ModulePath) -> Option<String> {
        self.resolver.display_name(resource)
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

/// Dispatches resolution of a resource to sub-resolvers.
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
    pub fn mount_resolver(&mut self, path: impl AsRef<Path>, resolver: impl Resolver + 'static) {
        let path = path.as_ref();
        let resource = ModulePath::from_path(path);
        let resolver: Box<dyn Resolver> = Box::new(resolver);
        if path == Path::new("") {
            self.fallback = Some((resource, resolver));
        } else {
            self.mount_points.push((resource, resolver));
        }
    }

    /// Mount a fallback resolver that is used when no other prefix match.
    pub fn mount_fallback_resolver(&mut self, resolver: impl Resolver + 'static) {
        self.mount_resolver("", resolver);
    }

    fn route(&self, resource: &ModulePath) -> Result<(&dyn Resolver, ModulePath), E> {
        let (mount_path, resolver) = self
            .mount_points
            .iter()
            .filter(|(prefix, _)| resource.starts_with(prefix))
            .max_by_key(|(prefix, _)| prefix.components.len())
            .or(self
                .fallback
                .as_ref()
                .take_if(|(prefix, _)| resource.starts_with(prefix)))
            .ok_or_else(|| E::InvalidResource(resource.clone(), "no mount point".to_string()))?;

        let components = resource
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
    fn resolve_source<'a>(&'a self, resource: &ModulePath) -> Result<Cow<'a, str>, E> {
        let (resolver, resource) = self.route(resource)?;
        resolver.resolve_source(&resource)
    }
    fn source_to_module(&self, source: &str, resource: &ModulePath) -> Result<TranslationUnit, E> {
        let (resolver, resource) = self.route(resource)?;
        resolver.source_to_module(source, &resource)
    }
    fn resolve_module(&self, resource: &ModulePath) -> Result<TranslationUnit, E> {
        let (resolver, resource) = self.route(resource)?;
        resolver.resolve_module(&resource)
    }
    fn display_name(&self, resource: &ModulePath) -> Option<String> {
        let (resolver, resource) = self.route(resource).ok()?;
        resolver.display_name(&resource)
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
    fn resolve_source<'a>(&'a self, resource: &ModulePath) -> Result<std::borrow::Cow<'a, str>, E> {
        for pkg in &self.packages {
            // TODO: the resolution algorithm is currently not spec-compliant.
            // https://github.com/wgsl-tooling-wg/wesl-spec/blob/imports-update/Imports.md
            if resource.origin.is_package()
                && resource.components.first().map(String::as_str) == Some(pkg.name())
            {
                let mut cur_mod = *pkg;
                for comp in resource.components.iter().skip(1) {
                    if let Some(submod) = pkg.submodule(comp) {
                        cur_mod = submod;
                    } else {
                        return Err(E::ModuleNotFound(
                            resource.clone(),
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
            resource.clone(),
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
    fn resolve_source<'a>(&'a self, resource: &ModulePath) -> Result<Cow<'a, str>, E> {
        if resource.origin.is_package() {
            self.pkg.resolve_source(resource)
        } else {
            self.files.resolve_source(resource)
        }
    }
    fn resolve_module(&self, resource: &ModulePath) -> Result<TranslationUnit, E> {
        if resource.origin.is_package() {
            self.pkg.resolve_module(resource)
        } else {
            self.files.resolve_module(resource)
        }
    }
    fn display_name(&self, resource: &ModulePath) -> Option<String> {
        if resource.origin.is_package() {
            self.pkg.display_name(resource)
        } else {
            self.files.display_name(resource)
        }
    }
}
