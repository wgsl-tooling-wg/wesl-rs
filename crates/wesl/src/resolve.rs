use crate::{Diagnostic, Error, SyntaxUtil};

use itertools::Itertools;
use wgsl_parse::syntax::TranslationUnit;

use std::{
    borrow::Cow,
    cell::RefCell,
    collections::HashMap,
    fmt::Display,
    fs,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("invalid import path: `{0}` ({1})")]
    InvalidResource(Resource, String),
    #[error("file not found: `{0}` ({1})")]
    FileNotFound(PathBuf, String),
    #[error("{0}")]
    Error(#[from] Diagnostic<Error>),
}

/// A resource uniquely identify an importable module (file).
///
/// Each module must be associated with a unique `Resource`, and a `Resource` must
/// identify a unique Module.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Resource {
    path: PathBuf,
}

impl Resource {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl From<PathBuf> for Resource {
    fn from(path: PathBuf) -> Self {
        Self { path }
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
        } else if self.path.starts_with(".") {
            write!(f, "{}", fmt_path(self.path.components()))
        } else {
            write!(f, "crate::{}", fmt_path(self.path.components()))
        }
    }
}

/// A Resolver is responsible for turning a import path into a unique resource identifer ([`Resource`]),
/// and providing the source file.
///
/// `Resolver` implementations must respect the following constraints:
/// * TODO
pub trait Resolver {
    /// Tries to resolve a source file identified by a resource.
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, ResolveError>;
    fn source_to_module(
        &self,
        source: &str,
        resource: &Resource,
    ) -> Result<TranslationUnit, ResolveError> {
        let mut wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_file(resource.clone())
                .with_source(source.to_string())
        })?;
        wesl.retarget_idents(); // it's important to call that early on to have identifiers point at the right declaration.
        Ok(wesl)
    }
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, ResolveError> {
        let source = self.resolve_source(resource)?;
        let wesl = self.source_to_module(&source, resource)?;
        Ok(wesl)
    }
}

impl<T: Resolver + ?Sized> Resolver for Box<T> {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, ResolveError> {
        (**self).resolve_source(resource)
    }
    fn source_to_module<'a>(
        &'a self,
        source: &str,
        resource: &Resource,
    ) -> Result<TranslationUnit, ResolveError> {
        (**self).source_to_module(source, resource)
    }
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, ResolveError> {
        (**self).resolve_module(resource)
    }
}

impl<T: Resolver> Resolver for &T {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, ResolveError> {
        (**self).resolve_source(resource)
    }
    fn source_to_module<'a>(
        &'a self,
        source: &str,
        resource: &Resource,
    ) -> Result<TranslationUnit, ResolveError> {
        (**self).source_to_module(source, resource)
    }
    fn resolve_module(&self, resource: &Resource) -> Result<TranslationUnit, ResolveError> {
        (**self).resolve_module(resource)
    }
}

/// A resolver that never resolves anything.
///
/// Returns [`ResolveError::FileNotFound`] when calling [`Resolver::resolve_source`].
#[derive(Default, Clone, Debug)]
pub struct NoResolver;

impl Resolver for NoResolver {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, ResolveError> {
        Err(ResolveError::InvalidResource(
            resource.clone(),
            "no resolver".to_string(),
        ))
    }
}

/// A resolver that remembers to avoid multiple fetches.
pub struct CacheResolver<'a> {
    resolver: Box<dyn Resolver + 'a>,
    cache: RefCell<HashMap<Resource, String>>,
}

impl<'a> CacheResolver<'a> {
    pub fn new(resolver: Box<dyn Resolver + 'a>) -> Self {
        Self {
            resolver,
            cache: Default::default(),
        }
    }
}

impl<'a> Resolver for CacheResolver<'a> {
    fn resolve_source<'b>(&'b self, resource: &Resource) -> Result<Cow<'b, str>, ResolveError> {
        let mut cache = self.cache.borrow_mut();

        let source = if let Some(source) = cache.get(resource) {
            source
        } else {
            let source = self.resolver.resolve_source(resource)?;
            cache.insert(resource.clone(), source.into_owned());
            cache.get(resource).unwrap()
        };

        Ok(source.clone().into())
    }
}

/// A resolver that looks for files in the filesystem.
///
/// This is the default resolver.
#[derive(Default)]
pub struct FileResolver {
    base: PathBuf,
    extension: &'static str,
}

impl FileResolver {
    /// `base` is the root directory to which absolute paths refer to.
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
}

impl Resolver for FileResolver {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, ResolveError> {
        let mut path = self.base.to_path_buf();
        path.extend(resource.path());
        if path.extension().is_none() {
            path.set_extension(self.extension);
        }

        let source = fs::read_to_string(&path)
            .map_err(|_| ResolveError::FileNotFound(path, "physical file".to_string()))?;

        Ok(source.into())
    }
}

/// A resolver that resolves WESL modules added with [`VirtualResolver::add_module`].
///
/// This can be used in platforms that lack a filesystem (e.g. WASM) or for
/// runtime-generated files.
#[derive(Default)]
pub struct VirtualResolver {
    files: HashMap<PathBuf, String>,
}

impl VirtualResolver {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// resolves imports in `path` with the given WESL string.
    pub fn add_module(&mut self, path: impl AsRef<Path>, file: String) {
        self.files.insert(path.as_ref().to_path_buf(), file);
    }

    pub fn get_module(&self, resource: &Resource) -> Result<&str, Error> {
        let path = resource.path();
        let source = self.files.get(path).ok_or_else(|| {
            ResolveError::FileNotFound(path.to_path_buf(), "virtual module".to_string())
        })?;
        Ok(source)
    }
}

impl Resolver for VirtualResolver {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, ResolveError> {
        let source = self
            .get_module(resource)
            .map_err(|e| Diagnostic::from(e).with_file(resource.clone()))?;
        Ok(source.into())
    }
}

// trait alias
pub trait ResolveFn: Fn(&mut TranslationUnit) -> Result<(), Error> {}
impl<T: Fn(&mut TranslationUnit) -> Result<(), Error>> ResolveFn for T {}

/// A WESL module preprocessor.
///
/// The preprocess function will be called each time the WESL compiler tries accesses a
/// module. The preprocessor can modify the module syntax tree at will.
///
/// The preprocess function *must* always return the same syntax tree for a given module
/// path.
pub struct Preprocessor<'a, F: ResolveFn> {
    pub resolver: Box<dyn Resolver + 'a>,
    pub preprocess: F,
}

impl<'a, F: ResolveFn> Preprocessor<'a, F> {
    pub fn new(resolver: Box<dyn Resolver + 'a>, preprocess: F) -> Self {
        Self {
            resolver,
            preprocess,
        }
    }
}

impl<'a, F: ResolveFn> Resolver for Preprocessor<'a, F> {
    fn resolve_source<'b>(&'b self, resource: &Resource) -> Result<Cow<'b, str>, ResolveError> {
        let res = self.resolver.resolve_source(resource)?;
        Ok(res)
    }
    fn source_to_module<'b>(
        &'b self,
        source: &str,
        resource: &Resource,
    ) -> Result<TranslationUnit, ResolveError> {
        let mut wesl: TranslationUnit = source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_file(resource.clone())
                .with_source(source.to_string())
        })?;
        wesl.retarget_idents(); // it's important to call that early on to have identifiers point at the right declaration.
        (self.preprocess)(&mut wesl).map_err(|e| {
            Diagnostic::from(e)
                .with_file(resource.clone())
                .with_source(source.to_string())
        })?;
        Ok(wesl)
    }
}

/// A router is a resolver that can dispatch imports to several sub-resolvers based on the
/// import path prefix.
///
/// Add sub-resolvers with [`Router::mount_resolver`].
pub struct Router {
    mount_points: Vec<(PathBuf, Box<dyn Resolver>)>,
    fallback: Option<(PathBuf, Box<dyn Resolver>)>,
}

/// Dispatches resolution of a resource to sub-resolvers.
impl Router {
    pub fn new() -> Self {
        Self {
            mount_points: Vec::new(),
            fallback: None,
        }
    }

    /// Mount a resolver at a given path prefix. All imports that start with this prefix
    /// will be dispatched to that resolver.
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
        self.mount_resolver(PathBuf::new(), resolver);
    }
}

impl Resolver for Router {
    fn resolve_source<'a>(&'a self, resource: &Resource) -> Result<Cow<'a, str>, ResolveError> {
        let (mount_path, resolver) = self
            .mount_points
            .iter()
            .filter(|(path, _)| resource.path().starts_with(path))
            .max_by_key(|(path, _)| path.iter().count())
            .or(self.fallback.as_ref())
            .ok_or_else(|| {
                ResolveError::InvalidResource(resource.clone(), "no mount point".to_string())
            })?;

        // SAFETY: we just checked that resource.path() starts with mount_path
        let suffix = resource.path().strip_prefix(mount_path).unwrap();
        let resource = Resource::from(suffix.to_path_buf());
        resolver.resolve_source(&resource)
        // .map_err(|e| match e {
        //     ResolveError::FileNotFound(msg) => {
        //         ResolveError::FileNotFound(format!("`{}`::{msg}", mount_path.display()))
        //     }
        //     ResolveError::Error(_) => e,
        // })
    }
}

pub trait PkgModule: Sync {
    fn name(&self) -> &'static str;
    fn source(&self) -> &'static str;
    fn submodules(&self) -> &[&dyn PkgModule];
    fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
        self.submodules()
            .iter()
            .find(|sm| sm.name() == name)
            .map(|sm| *sm)
    }
}

pub struct PkgResolver {
    packages: Vec<&'static dyn PkgModule>,
    fallback: Option<Box<dyn Resolver>>,
}

impl PkgResolver {
    pub fn new() -> Self {
        Self {
            packages: Vec::new(),
            fallback: None,
        }
    }

    pub fn add_package(&mut self, pkg: &'static dyn PkgModule) {
        self.packages.push(pkg);
    }

    pub fn mount_fallback_resolver(&mut self, fallback: impl Resolver + 'static) {
        self.fallback = Some(Box::new(fallback));
    }
}

impl Resolver for PkgResolver {
    fn resolve_source<'a>(
        &'a self,
        resource: &Resource,
    ) -> Result<std::borrow::Cow<'a, str>, ResolveError> {
        if resource.path().has_root() {
            let path = resource.path().strip_prefix("/").unwrap();
            for pkg in &self.packages {
                // TODO: the resolution algorithm is currently not spec-compliant.
                // https://github.com/wgsl-tooling-wg/wesl-spec/blob/imports-update/Imports.md
                if path.starts_with(pkg.name()) {
                    let mut cur_mod = *pkg;
                    for segment in path.iter().skip(1) {
                        let name = segment.to_str().ok_or_else(|| {
                            ResolveError::InvalidResource(
                                resource.clone(),
                                "invalid unicode".to_string(),
                            )
                        })?;
                        if let Some(submod) = pkg.submodule(name) {
                            cur_mod = submod;
                        } else {
                            return Err(ResolveError::FileNotFound(
                                path.to_path_buf(),
                                format!("in package {}", pkg.name()),
                            ));
                        }
                    }
                    return Ok(cur_mod.source().into());
                }
            }
            Err(ResolveError::FileNotFound(
                resource.path().to_path_buf(),
                "no package found".to_string(),
            ))
        } else {
            self.fallback
                .as_ref()
                .map(|fallback| fallback.resolve_source(resource))
                .unwrap_or_else(|| {
                    Err(ResolveError::FileNotFound(
                        resource.path().to_path_buf(),
                        "no package found".to_string(),
                    ))
                })
        }
    }
}
