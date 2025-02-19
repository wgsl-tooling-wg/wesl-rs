use crate::{Diagnostic, Error, SyntaxUtil};

use itertools::Itertools;
use wgsl_parse::syntax::TranslationUnit;

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Display,
    fs,
    path::{Component, Path, PathBuf},
};

/// Error produced by module resolution.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("invalid import path: `{0}` ({1})")]
    InvalidResource(Resource, String),
    #[error("file not found: `{0}` ({1})")]
    FileNotFound(PathBuf, String),
    #[error("{0}")]
    Error(#[from] Diagnostic<Error>),
}

type E = ResolveError;

// TODO: rename ModulePath?
/// A resource uniquely identify an importable module (file).
///
/// Each module must be associated with a unique `Resource`, and a `Resource` must
/// identify a unique module.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Resource {
    // TODO: get rid or pathbuf. it is not adapted to current use-cases.
    path: PathBuf,
}

fn clean_path(path: impl AsRef<Path>) -> PathBuf {
    let mut res = PathBuf::new();
    for comp in path.as_ref().with_extension("").components() {
        match comp {
            Component::Prefix(_) => {}
            Component::RootDir => {
                res.push(comp);
            }
            Component::CurDir => {
                // can only 'start' with './'
                if res == Path::new("") {
                    res.push(comp)
                }
            }
            Component::ParentDir => {
                if !res.pop() {
                    res.push(comp);
                }
            }
            Component::Normal(_) => res.push(comp),
        }
    }
    res
}

impl Resource {
    /// Create a new resource from a module path.
    ///
    /// Precondition: the path components must be valid WGSL identifiers, or `..` or `.`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: clean_path(path),
        }
    }
    /// Represent this resource as a module path.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Add a path component to the resource.
    ///
    /// Precondition: the `item` must be a valid WGSL identifier.
    pub fn push(&mut self, item: &str) {
        self.path.push(item);
    }
    /// Get the parent resource, if this resource is not the root module.
    pub fn parent(&self) -> Option<Self> {
        self.path.parent().map(|p| Self {
            path: p.to_path_buf(),
        })
    }
    /// Get the first component of the module path.
    pub fn first(&self) -> Option<&str> {
        self.path.iter().next().map(|p| p.to_str().unwrap())
    }
    /// Get the last component of the module path.
    pub fn last(&self) -> Option<&str> {
        self.path.iter().last().map(|p| p.to_str().unwrap())
    }
    /// Append `suffix` to the module path.
    pub fn join(&self, suffix: impl AsRef<Path>) -> Self {
        let mut path = self.path.clone();
        path.push(suffix);
        Self {
            path: clean_path(path),
        }
    }
    /// Whether this resource points at an item in an external package.
    pub fn is_package(&self) -> bool {
        self.path.has_root()
    }
    /// Get the the module path inside of the package, if this resource refers
    /// to an item in an external package.
    pub fn package_local(&self) -> Option<Resource> {
        self.is_package().then(|| {
            let path = self.path.iter().skip(1).collect::<PathBuf>();
            Resource::new(path)
        })
    }
    /// Whether this resource is relative (corresponds to a module path
    /// starting with `super::`).
    pub fn is_relative(&self) -> bool {
        self.first() == Some(".")
    }
    /// Make this resource absolute by dropping the `super::` prefix.
    pub fn absolute(mut self) -> Resource {
        if self.is_relative() {
            self.path = PathBuf::from_iter(self.path.iter().skip(1));
            self
        } else {
            self
        }
    }
}

impl Display for Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn fmt_path<'a>(path: impl Iterator<Item = Component<'a>> + 'a) -> impl Display + 'a {
            path.filter_map(|seg| match seg {
                std::path::Component::Prefix(_) | std::path::Component::RootDir => None,
                std::path::Component::CurDir => Some("self"),
                std::path::Component::ParentDir => Some("super"),
                std::path::Component::Normal(str) => str.to_str(),
            })
            .format("::")
        }
        if self.path.has_root() {
            write!(f, "{}", fmt_path(self.path.components().skip(1)))
        } else if self.path.starts_with(".") || self.path.starts_with("..") {
            write!(f, "{}", fmt_path(self.path.components()))
        } else {
            write!(f, "package::{}", fmt_path(self.path.components()))
        }
    }
}

/// A Resolver is responsible for turning an import path into a unique module path
/// ([`Resource`]) and providing the source file and syntax tree.
///
/// Typically implementations of [`Resolver`] only implement [`Resolver::resolve_source`].
///
/// Calls to `Resolver` functions must respect these preconditions:
/// * the resource must be canonical (absolute module path).
pub trait Resolver {
    /// Try to resolve a source file identified by a resource.
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, E>;
    /// Convert a source file into a syntax tree.
    fn source_to_module(&self, source: &str, resource: &Resource) -> Result<TranslationUnit, E> {
        let mut wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_resource(resource.clone(), self.display_name(resource))
                .with_source(source.to_string())
        })?;
        wesl.retarget_idents(); // it's important to call that early on to have identifiers point at the right declaration.
        Ok(wesl)
    }
    /// Try to resolve a source file identified by a resource.
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, E> {
        let source = self.resolve_source(resource)?;
        let wesl = self.source_to_module(&source, resource)?;
        Ok(wesl)
    }
    /// Get the display name of the resource. Implementing this is optional.
    fn display_name(&self, _resource: &Resource) -> Option<String> {
        None
    }
}

impl<T: Resolver + ?Sized> Resolver for Box<T> {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, E> {
        (**self).resolve_source(resource)
    }
    fn source_to_module(&self, source: &str, resource: &Resource) -> Result<TranslationUnit, E> {
        (**self).source_to_module(source, resource)
    }
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, E> {
        (**self).resolve_module(resource)
    }
    fn display_name(&self, resource: &Resource) -> Option<String> {
        (**self).display_name(resource)
    }
}

impl<T: Resolver> Resolver for &T {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, E> {
        (**self).resolve_source(resource)
    }
    fn source_to_module(&self, source: &str, resource: &Resource) -> Result<TranslationUnit, E> {
        (**self).source_to_module(source, resource)
    }
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, E> {
        (**self).resolve_module(resource)
    }
    fn display_name(&self, resource: &Resource) -> Option<String> {
        (**self).display_name(resource)
    }
}

/// A resolver that never resolves anything.
///
/// Returns [`ResolveError::InvalidResource`] when calling [`Resolver::resolve_source`].
#[derive(Default, Clone, Debug)]
pub struct NoResolver;

impl Resolver for NoResolver {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, E> {
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

    fn file_path(&self, resource: &Resource) -> Result<PathBuf, E> {
        if resource.path().has_root() {
            return Err(E::InvalidResource(
                resource.clone(),
                "not a file".to_string(),
            ));
        }
        let mut path = self.base.to_path_buf();
        path.extend(resource.path());
        let has_extension = path.extension().is_some();
        if !has_extension {
            path.set_extension(self.extension);
        }
        if path.exists() {
            Ok(path)
        } else if !has_extension {
            path.set_extension("wgsl");
            if path.exists() {
                Ok(path)
            } else {
                Err(E::FileNotFound(path, "physical file".to_string()))
            }
        } else {
            Err(E::FileNotFound(path, "physical file".to_string()))
        }
    }
}

impl Resolver for FileResolver {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, E> {
        let path = self.file_path(resource)?;
        let source = fs::read_to_string(&path)
            .map_err(|_| E::FileNotFound(path, "physical file".to_string()))?;

        Ok(source.into())
    }
    fn display_name(&self, resource: &Resource) -> Option<String> {
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
    files: HashMap<Resource, Cow<'a, str>>,
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
        self.files.insert(Resource::new(path).absolute(), file);
    }

    /// Get a module registered with [`Self::add_module`].
    pub fn get_module(&self, resource: &Resource) -> Result<&str, E> {
        let source = self.files.get(resource).ok_or_else(|| {
            E::FileNotFound(resource.path.to_path_buf(), "virtual module".to_string())
        })?;
        Ok(source)
    }

    /// Iterate over all registered modules.
    pub fn modules(&self) -> impl Iterator<Item = (&Resource, &str)> {
        self.files.iter().map(|(res, file)| (res, &**file))
    }
}

impl Resolver for VirtualResolver<'_> {
    fn resolve_source<'b>(&'b self, resource: &Resource) -> Result<Cow<'b, str>, E> {
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
    fn resolve_source<'b>(&'b self, resource: &Resource) -> Result<Cow<'b, str>, E> {
        let res = self.resolver.resolve_source(resource)?;
        Ok(res)
    }
    fn source_to_module(&self, source: &str, resource: &Resource) -> Result<TranslationUnit, E> {
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
    fn display_name(&self, resource: &Resource) -> Option<String> {
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
    mount_points: Vec<(PathBuf, Box<dyn Resolver>)>,
    fallback: Option<(PathBuf, Box<dyn Resolver>)>,
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
        let path = path.as_ref().to_path_buf();
        let resolver: Box<dyn Resolver> = Box::new(resolver);
        if path.iter().count() == 0 {
            self.fallback = Some((path, resolver));
        } else {
            self.mount_points.push((path, resolver));
        }
    }

    /// Mount a fallback resolver that is used when no other prefix match.
    pub fn mount_fallback_resolver(&mut self, resolver: impl Resolver + 'static) {
        self.mount_resolver("", resolver);
    }

    fn route(&self, resource: &Resource) -> Result<(&dyn Resolver, Resource), E> {
        let (mount_path, resolver) = self
            .mount_points
            .iter()
            .filter(|(path, _)| resource.path().starts_with(path))
            .max_by_key(|(path, _)| path.iter().count())
            .or(self
                .fallback
                .as_ref()
                .take_if(|(path, _)| resource.path().starts_with(path)))
            .ok_or_else(|| E::InvalidResource(resource.clone(), "no mount point".to_string()))?;

        // SAFETY: we just checked that resource.path() starts with mount_path
        let suffix = resource.path().strip_prefix(mount_path).unwrap();
        let resource = Resource::new(suffix);
        Ok((resolver, resource))
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Resolver for Router {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, E> {
        let (resolver, resource) = self.route(resource)?;
        resolver.resolve_source(&resource)
    }
    fn source_to_module(&self, source: &str, resource: &Resource) -> Result<TranslationUnit, E> {
        let (resolver, resource) = self.route(resource)?;
        resolver.source_to_module(source, &resource)
    }
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, E> {
        let (resolver, resource) = self.route(resource)?;
        resolver.resolve_module(&resource)
    }
    fn display_name(&self, resource: &Resource) -> Option<String> {
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
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<std::borrow::Cow<'a, str>, E> {
        let path = resource.path();
        for pkg in &self.packages {
            // TODO: the resolution algorithm is currently not spec-compliant.
            // https://github.com/wgsl-tooling-wg/wesl-spec/blob/imports-update/Imports.md
            if resource.path().starts_with(pkg.name()) {
                let mut cur_mod = *pkg;
                for segment in path.iter().skip(1) {
                    let name = segment.to_str().ok_or_else(|| {
                        E::InvalidResource(resource.clone(), "invalid unicode".to_string())
                    })?;
                    if let Some(submod) = pkg.submodule(name) {
                        cur_mod = submod;
                    } else {
                        return Err(E::FileNotFound(
                            path.to_path_buf(),
                            format!("in package {}", pkg.name()),
                        ));
                    }
                }
                return Ok(cur_mod.source().into());
            }
        }
        Err(E::FileNotFound(
            resource.path().to_path_buf(),
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
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, E> {
        if let Some(res) = resource.package_local() {
            self.pkg.resolve_source(&res)
        } else {
            self.files.resolve_source(resource)
        }
    }
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, E> {
        if let Some(res) = resource.package_local() {
            self.pkg.resolve_module(&res)
        } else {
            self.files.resolve_module(resource)
        }
    }
    fn display_name(&self, resource: &Resource) -> Option<String> {
        if let Some(res) = resource.package_local() {
            self.pkg.display_name(&res)
        } else {
            self.files.display_name(resource)
        }
    }
}
