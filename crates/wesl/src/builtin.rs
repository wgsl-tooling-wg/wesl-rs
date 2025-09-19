use std::{collections::HashMap, sync::LazyLock};

use wgsl_parse::syntax::Ident;

/// All built-in names as [`Ident`]s.
///
/// Using these idents allow better use-count tracking and referencing.
pub static BUILTIN_IDENTS: LazyLock<HashMap<&str, Ident>> = LazyLock::new(|| {
    macro_rules! ident {
        ($name:expr) => {
            ($name, Ident::new($name.to_string()))
        };
    }
    HashMap::from_iter(wgsl_types::idents::iter_builtin_idents().map(|id| ident!(id)))
});

/// Get a built-in WGSL name as [`Ident`].
///
/// Using these idents allow better use-count tracking and referencing.
pub fn builtin_ident(name: &str) -> Option<&'static Ident> {
    BUILTIN_IDENTS.get(name)
}
