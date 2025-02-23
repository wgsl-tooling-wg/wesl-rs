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
/// * A pair {path, item} (aka. fully-qualifed name) must be associated with a unique mangled name.
/// * A mangled name must be associated with a unique pair {path, item} (or at least, the risk of
///   a collision must be negligible).
/// * The mangled name must be a valid WGSL identifier.
///
/// Calls to `Mangler::mangle` must respect these preconditions:
/// * the item must be a valid WGSL identifier.
///
/// # WESL Reference
/// spec: [NameMangling.md](https://github.com/wgsl-tooling-wg/wesl-spec/blob/main/NameMangling.md)
pub trait Mangler {
    /// Turn an import path and item name into a mangled WGSL identifier.
    fn mangle(&self, path: &ModulePath, item: &str) -> String;
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
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        (**self).mangle(path, item)
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        (**self).unmangle(mangled)
    }
    fn mangle_types(&self, item: &str, variant: u32, types: &[TypeExpression]) -> String {
        (**self).mangle_types(item, variant, types)
    }
}

impl<T: Mangler> Mangler for &T {
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        (**self).mangle(path, item)
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        (**self).unmangle(mangled)
    }
    fn mangle_types(&self, item: &str, variant: u32, types: &[TypeExpression]) -> String {
        (**self).mangle_types(item, variant, types)
    }
}

/// A mangler that hashes the module path.
/// e.g. `foo::bar::baz item => item_32938483840293402930392`
#[derive(Default, Clone, Debug)]
pub struct HashMangler;

impl Mangler for HashMangler {
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
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

impl EscapeMangler {
    pub fn escape_component(comp: &str) -> String {
        let underscores = comp.chars().filter(|c| *c == '_').count();
        if underscores > 0 {
            format!("_{underscores}{comp}")
        } else {
            comp.to_string()
        }
    }
}

impl Mangler for EscapeMangler {
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        let origin = match path.origin {
            PathOrigin::Absolute => "package_".to_string(),
            PathOrigin::Relative(0) => "self_".to_string(),
            PathOrigin::Relative(n) => format!("{}_", (1..n).map(|_| "super").format("_")),
            PathOrigin::Package => "".to_string(),
        };
        let path = path
            .components
            .iter()
            .map(|comp| Self::escape_component(comp))
            .format("_");
        format!("{origin}{path}_{}", Self::escape_component(item))
    }
    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        let mut parts = mangled.split('_').peekable();
        let mut origin = PathOrigin::Package;
        let mut components = Vec::new();

        fn extract_count(part: &str) -> Option<usize> {
            if !part.starts_with('_') {
                return None;
            }

            let digits = part
                .chars()
                .skip(1)
                .take_while(|c| c.is_ascii_digit())
                .count();

            if digits == 0 {
                return None;
            }

            part[1..digits].parse().ok()
        }

        while let Some(part) = parts.next() {
            match part {
                "package" => {
                    origin = PathOrigin::Absolute;
                }
                "super" => {
                    if let PathOrigin::Relative(n) = &mut origin {
                        *n += 1;
                    } else {
                        origin = PathOrigin::Relative(1)
                    }
                }
                "self" => origin = PathOrigin::Relative(0),
                _ => {
                    if let Some(n) = extract_count(part) {
                        let part = std::iter::once(part)
                            .chain((0..n).map(|_| parts.next().expect("invalid mangled string")))
                            .format("_")
                            .to_string();
                        components.push(part)
                    } else {
                        components.push(part.to_string())
                    }
                }
            }
        }

        let item = components.pop()?;

        let path = ModulePath::new(origin, components);
        Some((path, item))
    }
}

/// A mangler that just returns the identifer as-is (no mangling).
/// e.g. `foo::bar::baz item => item`
///
/// Warning: this mangler is not spec-compliant. It can cause name collisions.
#[derive(Default, Clone, Debug)]
pub struct NoMangler;

impl Mangler for NoMangler {
    fn mangle(&self, _path: &ModulePath, item: &str) -> String {
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

impl<T: Mangler> Mangler for CacheMangler<'_, T> {
    fn mangle(&self, paht: &ModulePath, item: &str) -> String {
        let res = self.mangler.mangle(paht, item);
        let mut cache = self.cache.borrow_mut();
        cache.insert(res.clone(), (paht.clone(), item.to_string()));
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
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        let sep = Self::SEP;
        let origin = match path.origin {
            PathOrigin::Absolute => format!("package{sep}"),
            PathOrigin::Relative(0) => format!("self{sep}"),
            PathOrigin::Relative(n) => format!("{}{sep}", (1..n).map(|_| "super").format(sep)),
            PathOrigin::Package => "".to_string(),
        };
        let path = path.components.iter().format(sep);
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
