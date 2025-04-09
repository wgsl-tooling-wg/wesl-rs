use std::{cell::RefCell, collections::HashMap};

use wgsl_parse::syntax::{TranslationUnit, TypeExpression};

use crate::{Mangler, ModulePath, ResolveError, Resolver};

/// A SourceMap is a lookup from compiled WGSL to source WESL. It translates a mangled
/// name into a module path and declaration name.
pub trait SourceMap {
    /// Get the module path and declaration name from a mangled name.
    fn get_decl(&self, decl: &str) -> Option<(&ModulePath, &str)>;
    /// Get a module contents.
    fn get_source(&self, path: &ModulePath) -> Option<&str>;
    /// Get a module display name.
    fn get_display_name(&self, path: &ModulePath) -> Option<&str>;
    /// Get the default module contents.
    fn get_default_source(&self) -> Option<&str> {
        None
    }
}

/// Basic implementation of [`SourceMap`].
#[derive(Clone, Debug, Default)]
pub struct BasicSourceMap {
    mappings: HashMap<String, (ModulePath, String)>,
    sources: HashMap<ModulePath, (Option<String>, String)>, // res -> (display_name, source)
    default_source: Option<String>,
}

impl BasicSourceMap {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn add_decl(&mut self, decl: String, path: ModulePath, item: String) {
        self.mappings.insert(decl, (path, item));
    }
    pub fn add_source(&mut self, file: ModulePath, name: Option<String>, source: String) {
        self.sources.insert(file, (name, source));
    }
    pub fn set_default_source(&mut self, source: String) {
        self.default_source = Some(source);
    }
}

impl SourceMap for BasicSourceMap {
    fn get_decl(&self, decl: &str) -> Option<(&ModulePath, &str)> {
        let (path, decl) = self.mappings.get(decl)?;
        Some((path, decl))
    }

    fn get_source(&self, path: &ModulePath) -> Option<&str> {
        self.sources.get(path).map(|(_, source)| source.as_str())
    }
    fn get_display_name(&self, path: &ModulePath) -> Option<&str> {
        self.sources.get(path).and_then(|(name, _)| name.as_deref())
    }
    fn get_default_source(&self) -> Option<&str> {
        self.default_source.as_deref()
    }
}

impl<T: SourceMap> SourceMap for Option<T> {
    fn get_decl(&self, decl: &str) -> Option<(&ModulePath, &str)> {
        self.as_ref().and_then(|map| map.get_decl(decl))
    }
    fn get_source(&self, path: &ModulePath) -> Option<&str> {
        self.as_ref().and_then(|map| map.get_source(path))
    }
    fn get_display_name(&self, path: &ModulePath) -> Option<&str> {
        self.as_ref().and_then(|map| map.get_display_name(path))
    }
    fn get_default_source(&self) -> Option<&str> {
        self.as_ref().and_then(|map| map.get_default_source())
    }
}

pub struct NoSourceMap;

impl SourceMap for NoSourceMap {
    fn get_decl(&self, _decl: &str) -> Option<(&ModulePath, &str)> {
        None
    }
    fn get_source(&self, _path: &ModulePath) -> Option<&str> {
        None
    }
    fn get_display_name(&self, _path: &ModulePath) -> Option<&str> {
        None
    }
    fn get_default_source(&self) -> Option<&str> {
        None
    }
}

/// Generate a SourceMap by keeping track of name mangling and file resolutions.
///
/// `SourceMapper` is a proxy that implements [`Mangler`] and [`Resolver`]. To record a
/// sourcemap, invoke the compiler with this instance as both the mangler and the
/// resolver.
pub struct SourceMapper<'a> {
    pub root: &'a ModulePath,
    pub resolver: &'a dyn Resolver,
    pub mangler: &'a dyn Mangler,
    pub sourcemap: RefCell<BasicSourceMap>,
}

impl<'a> SourceMapper<'a> {
    /// Create a new `SourceMapper` from a mangler and a resolver.
    pub fn new(root: &'a ModulePath, resolver: &'a dyn Resolver, mangler: &'a dyn Mangler) -> Self {
        Self {
            root,
            resolver,
            mangler,
            sourcemap: Default::default(),
        }
    }
    /// Consume this and return a [`BasicSourceMap`].
    pub fn finish(self) -> BasicSourceMap {
        let mut sourcemap = self.sourcemap.into_inner();
        if let Some(source) = sourcemap.get_source(self.root) {
            sourcemap.set_default_source(source.to_string());
        }
        sourcemap
    }
}

impl Resolver for SourceMapper<'_> {
    fn resolve_source<'a>(
        &'a self,
        path: &ModulePath,
    ) -> Result<std::borrow::Cow<'a, str>, ResolveError> {
        let res = self.resolver.resolve_source(path)?;
        let mut sourcemap = self.sourcemap.borrow_mut();
        sourcemap.add_source(
            path.clone(),
            self.resolver.display_name(path),
            res.clone().into(),
        );
        Ok(res)
    }
    fn display_name(&self, path: &ModulePath) -> Option<String> {
        self.resolver.display_name(path)
    }
}

impl Mangler for SourceMapper<'_> {
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        let res = self.mangler.mangle(path, item);
        let mut sourcemap = self.sourcemap.borrow_mut();
        sourcemap.add_decl(res.clone(), path.clone(), item.to_string());
        res
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        self.mangler.unmangle(mangled)
    }
    fn mangle_types(&self, item: &str, variant: u32, types: &[TypeExpression]) -> String {
        self.mangler.mangle_types(item, variant, types)
    }
}
