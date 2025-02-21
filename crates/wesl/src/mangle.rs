use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

use itertools::Itertools;
use wgsl_parse::syntax::Expression;
use wgsl_parse::syntax::PathOrigin;
use wgsl_parse::syntax::TypeExpression;

use super::ModulePath;

/// A name mangler is responsible for renaming import-qualified identifiers into valid
/// WGSL identifiers.
///
/// `Mangler` implementations must respect the following constraints:
/// * A pair {resource, item} must be associated with a unique mangled name.
/// * A mangled name must be associated with a unique pair {resource, item} (or at least,
///   the risk of a collision must be negligible).
/// * The mangled name must be a valid WGSL identifier.
///
/// Calls to `Mangler::mangle` must respect these preconditions:
/// * the resource must be canonical (absolute module path).
/// * the item must be a valid WGSL identifier.
///
/// # WESL Reference
/// spec: [NameMangling.md](https://github.com/wgsl-tooling-wg/wesl-spec/blob/main/NameMangling.md)
pub trait Mangler {
    /// Turn an import path and item name into a mangled WGSL identifier. The resource
    /// must be the canonical (absolute) module path.
    fn mangle(&self, resource: &ModulePath, item: &str) -> String;
    /// Reverse the [`Mangler::mangle`] operation. Implementing this is optional.
    fn unmangle(&self, _mangled: &str) -> Option<(ModulePath, String)> {
        None
    }
    /// Used for generics. Is experimental. Implementing is optional.
    fn mangle_types(&self, item: &str, variant: u32, _types: &[TypeExpression]) -> String {
        format!("{item}_{variant}")
    }
}

impl<T: Mangler + ?Sized> Mangler for Box<T> {
    fn mangle(&self, resource: &ModulePath, item: &str) -> String {
        (**self).mangle(resource, item)
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        (**self).unmangle(mangled)
    }
    fn mangle_types(&self, item: &str, variant: u32, types: &[TypeExpression]) -> String {
        (**self).mangle_types(item, variant, types)
    }
}

impl<T: Mangler> Mangler for &T {
    fn mangle(&self, resource: &ModulePath, item: &str) -> String {
        (**self).mangle(resource, item)
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        (**self).unmangle(mangled)
    }
    fn mangle_types(&self, item: &str, variant: u32, types: &[TypeExpression]) -> String {
        (**self).mangle_types(item, variant, types)
    }
}

/// A mangler that hashes the resource identifier.
/// e.g. `foo::bar::baz item => item_32938483840293402930392`
#[derive(Default, Clone, Debug)]
pub struct HashMangler;

impl Mangler for HashMangler {
    fn mangle(&self, resource: &ModulePath, item: &str) -> String {
        let mut hasher = DefaultHasher::new();
        resource.hash(&mut hasher);
        item.hash(&mut hasher);
        let hash = hasher.finish();
        format!("{item}_{hash}")
    }
}

/// A mangler that replaces `::` with `_` and `_` with `__`.
/// e.g. `foo::bar_baz item => foo_bar__baz_item`
///
/// This is WESL's default mangler.
#[derive(Default, Clone, Debug)]
pub struct EscapeMangler;

impl Mangler for EscapeMangler {
    fn mangle(&self, resource: &ModulePath, item: &str) -> String {
        let origin = match resource.origin {
            PathOrigin::Absolute => "package_".to_string(),
            PathOrigin::Relative(0) => "self_".to_string(),
            PathOrigin::Relative(n) => format!("{}_", (1..n).map(|_| "super").format("_")),
            PathOrigin::Package => "".to_string(),
        };
        let path = resource
            .components
            .iter()
            .map(|p| p.replace('_', "__"))
            .format("_")
            .to_string();
        format!("{origin}{path}_{}", item.replace('_', "__"))
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        let mut parts = mangled.split('_').peekable();
        let mut origin = PathOrigin::Package;
        let mut components = Vec::new();

        while let Some(part) = parts.next() {
            if part == "package" {
                origin = PathOrigin::Absolute;
            } else {
                let mut part = part.to_string();
                while parts.peek() == Some(&"") {
                    part.push('_');
                    parts.next();
                    if let Some(next) = parts.next() {
                        part.push_str(next);
                    }
                }
                components.push(part);
            }
        }

        let item = components.pop()?;

        let resource = ModulePath::new(origin, components);
        Some((resource, item))
    }
}

/// A mangler that just returns the identifer as-is (no mangling).
/// e.g. `foo::bar::baz item => item`
///
/// Warning: this mangler is not spec-compliant. It can cause name collisions.
#[derive(Default, Clone, Debug)]
pub struct NoMangler;

impl Mangler for NoMangler {
    fn mangle(&self, _resource: &ModulePath, item: &str) -> String {
        item.to_string()
    }
}

/// A mangler that remembers and can unmangle.
pub struct CacheMangler<'a, T: Mangler> {
    cache: RefCell<HashMap<String, (ModulePath, String)>>,
    mangler: &'a T,
}

impl<'a, T: Mangler> CacheMangler<'a, T> {
    pub fn new(mangler: &'a T) -> Self {
        Self {
            cache: Default::default(),
            mangler,
        }
    }
}

impl<'a, T: Mangler> Mangler for CacheMangler<'a, T> {
    fn mangle(&self, resource: &ModulePath, item: &str) -> String {
        let res = self.mangler.mangle(resource, item);
        let mut cache = self.cache.borrow_mut();
        cache.insert(res.clone(), (resource.clone(), item.to_string()));
        res
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        {
            let cache = self.cache.borrow();
            if let Some(res) = cache.get(mangled).cloned() {
                return Some(res);
            }
        }

        self.mangler.unmangle(mangled)
    }
}

/// A mangler that uses cryptic unicode symbols that look like :, < and >
/// e.g. `foo::bar::baz array<f32,2> => foo::bar::baz::arrayᐸf32ˏ2ᐳ`
///
/// Very unlikely to collide unless using U+02D0 characters.
///
/// # Panics
/// if the TypeExpression is not normalized (i.e. contains only identifiers and literals)
#[derive(Default, Clone, Debug)]
pub struct UnicodeMangler;

impl UnicodeMangler {
    const LT: char = 'ᐸ'; // U+1438
    const GT: char = 'ᐳ'; // U+02CF
    const SEP: &'static str = "::"; // <-- these are NOT colons, they are U+02D0
    const TY_SEP: &'static str = "ˏ"; // U+02CF

    fn display_ty(ty: &TypeExpression) -> impl fmt::Display + '_ {
        format_args!(
            "{}{}",
            ty.ident,
            ty.template_args
                .iter()
                .format_with(Self::TY_SEP, |tplt, f| {
                    f(&format_args!(
                        "{}{}{}",
                        Self::LT,
                        tplt.iter().format_with(Self::TY_SEP, |tplt, f| {
                            match tplt.expression.node() {
                                Expression::Literal(lit) => f(lit),
                                Expression::TypeOrIdentifier(ty) => f(&Self::display_ty(ty)),
                                _ => panic!("only type names can be mangled"),
                            }
                        }),
                        Self::GT
                    ))
                })
        )
        .to_string()
    }
}

impl Mangler for UnicodeMangler {
    fn mangle(&self, resource: &ModulePath, item: &str) -> String {
        let sep = Self::SEP;
        let origin = match resource.origin {
            PathOrigin::Absolute => format!("package{sep}"),
            PathOrigin::Relative(0) => format!("self{sep}"),
            PathOrigin::Relative(n) => format!("{}{sep}", (1..n).map(|_| "super").format(sep)),
            PathOrigin::Package => "".to_string(),
        };
        let path = resource.components.iter().format(sep);
        format!("{origin}{path}{sep}{item}")
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        let mut components = mangled.split(Self::SEP).map(str::to_string).collect_vec();
        let mut origin = PathOrigin::Package;
        while let Some(comp) = components.first() {
            match comp.as_str() {
                "package" => {
                    components.remove(0);
                    origin = PathOrigin::Absolute;
                    break;
                }
                "self" => {
                    components.remove(0);
                    origin = PathOrigin::Relative(0);
                }
                "super" => {
                    components.remove(0);
                    origin = match origin {
                        PathOrigin::Relative(n) => PathOrigin::Relative(n + 1),
                        _ => PathOrigin::Relative(1),
                    };
                }
                _ => break,
            }
        }
        let name = components.pop()?;
        Some((ModulePath::new(origin, components), name))
    }
    fn mangle_types(&self, item: &str, _variant: u32, types: &[TypeExpression]) -> String {
        // these are NOT chevrons and comma!
        format!(
            "{item}{}{}{}",
            Self::LT,
            types
                .iter()
                .format_with(Self::TY_SEP, |ty, f| f(&Self::display_ty(ty))),
            Self::GT
        )
    }
}
