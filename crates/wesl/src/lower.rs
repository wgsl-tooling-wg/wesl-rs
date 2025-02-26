use crate::{visit::Visit, Error, Exec};

use wgsl_parse::syntax::*;

/// Performs conversions on the final syntax tree to make it more compatible with naga,
/// catch errors early and perform optimizations.
pub fn lower(wesl: &mut TranslationUnit) -> Result<(), Error> {
    wesl.imports.clear();

    for attrs in Visit::<Attributes>::visit_mut(wesl) {
        attrs.retain(|attr| {
            !matches!(attr, 
            Attribute::Custom(CustomAttribute { name, .. }) if name == "generic")
        })
    }

    #[cfg(not(feature = "eval"))]
    {
        // these are redundant with eval::lower.
        remove_type_aliases(wesl);
        remove_global_consts(wesl);
    }
    #[cfg(feature = "eval")]
    {
        use crate::eval::{mark_functions_const, Context, Lower};
        use crate::Diagnostic;
        mark_functions_const(wesl);

        // we want to drop wesl2 at the end of the block for idents use_count
        {
            let wesl2 = wesl.clone();
            let mut ctx = Context::new(&wesl2);
            wesl.exec(&mut ctx)?; // populate the ctx with module-scope declarations
            wesl.lower(&mut ctx)
                .map_err(|e| Diagnostic::from(e).with_ctx(&ctx))?;
        }

        // remove `@const` attributes.
        for decl in &mut wesl.global_declarations {
            if let GlobalDeclaration::Function(decl) = decl {
                decl.attributes.retain(|attr| *attr != Attribute::Const);
            }
        }
    }
    Ok(())
}

/// Eliminate all type aliases.
/// Naga doesn't like this: `alias T = u32; vec<T>`
#[allow(unused)]
fn remove_type_aliases(wesl: &mut TranslationUnit) {
    let take_next_alias = |wesl: &mut TranslationUnit| {
        let index = wesl
            .global_declarations
            .iter()
            .position(|decl| matches!(decl, GlobalDeclaration::TypeAlias(_)));
        index.map(|index| {
            let decl = wesl.global_declarations.swap_remove(index);
            match decl {
                GlobalDeclaration::TypeAlias(alias) => alias,
                _ => unreachable!(),
            }
        })
    };

    while let Some(mut alias) = take_next_alias(wesl) {
        // we rename the alias and all references to its type expression,
        // and drop the alias declaration.
        alias.ident.rename(format!("{}", alias.ty));
    }
}

/// Eliminate all const-declarations.
///
/// Replace usages of the const-declaration with its expression.
///
/// # Panics
/// panics if the const-declaration is ill-formed, i.e. has no initializer.
#[allow(unused)]
fn remove_global_consts(wesl: &mut TranslationUnit) {
    let take_next_const = |wesl: &mut TranslationUnit| {
        let index = wesl.global_declarations.iter().position(|decl| {
            matches!(
                decl,
                GlobalDeclaration::Declaration(Declaration {
                    kind: DeclarationKind::Const,
                    ..
                })
            )
        });
        index.map(|index| {
            let decl = wesl.global_declarations.swap_remove(index);
            match decl {
                GlobalDeclaration::Declaration(d) => d,
                _ => unreachable!(),
            }
        })
    };

    while let Some(mut decl) = take_next_const(wesl) {
        // we rename the const and all references to its expression in parentheses,
        // and drop the const declaration.
        decl.ident
            .rename(format!("({})", decl.initializer.unwrap()));
    }
}
