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
/// * A pair {path, item} (aka. fully-qualified name) must be associated with a unique mangled name.
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

/// A mangler that replaces `::` with `_` and prefixes components with the number
/// of `_` they contain.
/// e.g. `foo::bar_baz item => foo__1bar_baz_item`
///
/// This is WESL's default mangler.
#[derive(Default, Clone, Debug)]
pub struct EscapeMangler;

impl EscapeMangler {
    pub fn escape_component(comp: &str) -> String {
        if comp.contains('/') {
            // This can exist only for dependencies of dependencies. see join_paths in import.rs.
            let underscores = comp.chars().filter(|c| *c == '/').count();
            format!(
                "_{underscores}import_{}",
                comp.split('/').map(Self::escape_component).format("")
            )
        } else {
            let underscores = comp.chars().filter(|c| *c == '_').count();
            if underscores > 0 {
                format!("_{underscores}{comp}")
            } else {
                comp.to_string()
            }
        }
    }
}

impl Mangler for EscapeMangler {
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        let origin = match &path.origin {
            PathOrigin::Absolute => "package".to_string(),
            PathOrigin::Relative(0) => "self".to_string(),
            PathOrigin::Relative(n) => format!("{}", (0..*n).map(|_| "super").format("_")),
            PathOrigin::Package(name) => Self::escape_component(name).to_string(),
        };

        let item = Self::escape_component(item);

        if path.components.is_empty() {
            format!("{origin}_{item}",)
        } else {
            let path = path
                .components
                .iter()
                .map(|comp| Self::escape_component(comp))
                .format("_");

            format!("{origin}_{path}_{item}",)
        }
    }

    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        let mut components = Vec::new();
        let mut parts = mangled.split('_').filter(|p| p != &"").peekable();

        fn extract_count(part: &str) -> Option<(usize, &str)> {
            let digits = part.chars().take_while(|c| c.is_ascii_digit()).count();

            if digits == 0 {
                return None;
            }

            let count = part[..digits].parse().ok()?;
            Some((count, &part[digits..]))
        }

        let origin = match parts.next() {
            Some("package") => PathOrigin::Absolute,
            Some("self") => PathOrigin::Relative(0),
            Some("super") => {
                let mut n = 1;
                while let Some(&"super") = parts.peek() {
                    n += 1;
                    parts.next().unwrap();
                }
                PathOrigin::Relative(n)
            }
            Some(name) => PathOrigin::Package(name.to_string()),
            None => return None,
        };

        while let Some(part) = parts.next() {
            if let Some((n, rem)) = extract_count(part) {
                let part = std::iter::once(rem)
                    .chain((0..n).map(|_| parts.next().expect("invalid mangled string")))
                    .format("_")
                    .to_string();
                components.push(part)
            } else {
                components.push(part.to_string())
            }
        }

        let item = components.pop()?;

        let path = ModulePath::new(origin, components);
        Some((path, item))
    }
}

#[cfg(test)]
#[test]
fn test_escape_mangler() {
    let paths = [
        vec!["bevy_pbr".to_string(), "lighting".to_string()],
        vec![],
        vec!["a".to_string(), "b_c_d".to_string()],
    ];
    let paths = paths.iter().flat_map(|p| {
        [
            ModulePath::new(PathOrigin::Absolute, p.clone()),
            ModulePath::new(PathOrigin::Package("pkg".to_string()), p.clone()),
            ModulePath::new(PathOrigin::Relative(0), p.clone()),
            ModulePath::new(PathOrigin::Relative(2), p.clone()),
        ]
    });
    let mangled = [
        "package__1bevy_pbr_lighting_item",
        "pkg__1bevy_pbr_lighting_item",
        "self__1bevy_pbr_lighting_item",
        "super_super__1bevy_pbr_lighting_item",
        "package_item",
        "pkg_item",
        "self_item",
        "super_super_item",
        "package_a__2b_c_d_item",
        "pkg_a__2b_c_d_item",
        "self_a__2b_c_d_item",
        "super_super_a__2b_c_d_item",
    ];

    for (p, m) in paths.zip(mangled) {
        println!("testing {p}::item -> {m}");
        assert_eq!(EscapeMangler.mangle(&p, "item"), m);
        assert_eq!(EscapeMangler.unmangle(m), Some((p, "item".to_string())));
    }
}

/// A mangler that just returns the identifier as-is (no mangling).
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
    fn mangle(&self, path: &ModulePath, item: &str) -> String {
        let res = self.mangler.mangle(path, item);
        let mut cache = self.cache.borrow_mut();
        cache.insert(res.clone(), (path.clone(), item.to_string()));
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
        let origin = match &path.origin {
            PathOrigin::Absolute => format!("package{sep}"),
            PathOrigin::Relative(0) => format!("self{sep}"),
            PathOrigin::Relative(n) => format!("{}{sep}", (1..*n).map(|_| "super").format(sep)),
            PathOrigin::Package(name) => format!("{name}{sep}"),
        };
        let path = path.components.iter().format(sep);
        format!("{origin}{path}{sep}{item}")
    }

    fn unmangle(&self, mangled: &str) -> Option<(ModulePath, String)> {
        let mut components = mangled.split(Self::SEP).map(str::to_string).collect_vec();
        let mut origin = PathOrigin::Absolute;
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
                    break;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mangle_textures3d() {
        let mangler = EscapeMangler;
        let module_path = ModulePath::new(PathOrigin::Absolute, vec![]);
        let mangled = mangler.mangle(&module_path, "textures_3d");
        assert_eq!("package___1textures_3d", mangled);
    }

    #[test]
    fn unmangle_textures3d() {
        let mangler = EscapeMangler;
        let unmangled = mangler.unmangle("package___1textures_3d").map(|x| x.1);
        assert_eq!(Some("textures_3d"), unmangled.as_deref());
    }

    #[test]
    #[should_panic]
    fn unmangle_invalid() {
        let mangler = EscapeMangler;
        mangler.unmangle("textures_3d");
    }
}
