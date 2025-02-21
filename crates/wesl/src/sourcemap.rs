use std::{cell::RefCell, collections::HashMap};

use wgsl_parse::syntax::{TranslationUnit, TypeExpression};

use crate::{Mangler, ResolveError, Resolver, ModulePath};

/// A SourceMap is a lookup from compiled WGSL to source WESL. It translates a mangled
/// name into a module path and declaration name.
pub trait SourceMap {
    /// Get the module path and declaration name from a mangled name.
    fn get_decl(&self, decl: &str) -> Option<(&ModulePath, &str)>;
    /// Get a module contents.
    fn get_source(&self, resource: &ModulePath) -> Option<&str>;
    /// Get a module display name.
    fn get_display_name(&self, resource: &ModulePath) -> Option<&str>;
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
    pub fn add_decl(&mut self, decl: String, resource: ModulePath, item: String) {
        self.mappings.insert(decl, (resource, item));
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
        let (resource, decl) = self.mappings.get(decl)?;
        Some((resource, decl))
    }

    fn get_source(&self, resource: &ModulePath) -> Option<&str> {
        self.sources
            .get(resource)
            .map(|(_, source)| source.as_str())
    }
    fn get_display_name(&self, resource: &ModulePath) -> Option<&str> {
        self.sources
            .get(resource)
            .and_then(|(name, _)| name.as_deref())
    }
    fn get_default_source(&self) -> Option<&str> {
        self.default_source.as_deref()
    }
}

impl<T: SourceMap> SourceMap for Option<T> {
    fn get_decl(&self, decl: &str) -> Option<(&ModulePath, &str)> {
        self.as_ref().and_then(|map| map.get_decl(decl))
    }
    fn get_source(&self, resource: &ModulePath) -> Option<&str> {
        self.as_ref().and_then(|map| map.get_source(resource))
    }
    fn get_display_name(&self, resource: &ModulePath) -> Option<&str> {
        self.as_ref().and_then(|map| map.get_display_name(resource))
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
    fn get_source(&self, _resource: &ModulePath) -> Option<&str> {
        None
    }
    fn get_display_name(&self, _resource: &ModulePath) -> Option<&str> {
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
    pub resolver: &'a dyn Resolver,
    pub mangler: &'a dyn Mangler,
    pub sourcemap: RefCell<BasicSourceMap>,
}

impl<'a> SourceMapper<'a> {
    /// Create a new `SourceMapper` from a mangler and a resolver.
    pub fn new(resolver: &'a dyn Resolver, mangler: &'a dyn Mangler) -> Self {
        Self {
            resolver,
            mangler,
            sourcemap: Default::default(),
        }
    }
    /// Consume this and return a [`BasicSourceMap`].
    pub fn finish(self) -> BasicSourceMap {
        self.sourcemap.into_inner()
    }
}

impl<'a> Resolver for SourceMapper<'a> {
    fn resolve_source<'b>(
        &'b self,
        resource: &ModulePath,
    ) -> Result<std::borrow::Cow<'b, str>, ResolveError> {
        let res = self.resolver.resolve_source(resource)?;
        let mut sourcemap = self.sourcemap.borrow_mut();
        sourcemap.add_source(
            resource.clone(),
            self.resolver.display_name(resource),
            res.clone().into(),
        );
        Ok(res)
    }
    fn source_to_module(
        &self,
        source: &str,
        resource: &ModulePath,
    ) -> Result<TranslationUnit, ResolveError> {
        self.resolver.source_to_module(source, resource)
    }
    fn resolve_module(&self, resource: &ModulePath) -> Result<TranslationUnit, ResolveError> {
        self.resolver.resolve_module(resource)
    }
    fn display_name(&self, resource: &ModulePath) -> Option<String> {
        self.resolver.display_name(resource)
    }
}

impl<'a> Mangler for SourceMapper<'a> {
    fn mangle(&self, resource: &ModulePath, item: &str) -> String {
        let res = self.mangler.mangle(resource, item);
        let mut sourcemap = self.sourcemap.borrow_mut();
        sourcemap.add_decl(res.clone(), resource.clone(), item.to_string());
        res
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        self.mangler.unmangle(mangled)
    }
    fn mangle_types(&self, item: &str, variant: u32, types: &[TypeExpression]) -> String {
        self.mangler.mangle_types(item, variant, types)
    }
}
