use std::collections::HashSet;

use wesl_macros::query;
use wgsl_parse::syntax::{
    Expression, ExpressionNode, GlobalDeclaration, TranslationUnit, TypeExpression,
};

use crate::builtin::{BUILTIN_FUNCTIONS, BUILTIN_NAMES};
use crate::visit::Visit;
use crate::{Diagnostic, Error};

/// WESL or WGSL Validation error.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ValidateError {
    #[error("cannot find declaration of `{0}`")]
    UndefinedSymbol(String),
    #[error("incorrect number of arguments to `{0}`, expected `{1}`, got `{2}`")]
    ParamCount(String, usize, usize),
    #[error("`{0}` is not callable")]
    NotCallable(String),
    #[error("duplicate declaration of `{0}`")]
    Duplicate(String),
}

type E = ValidateError;

/// An identifier is linked to a declaration if:
/// * its use-count is greater than 1
/// * OR it is a built-in name
///
/// Note that this function could be simplified if we didn't care about the diagnostics metadata (declaration and expression)
fn check_defined_symbols(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    fn check_ty(ty: &TypeExpression) -> Result<(), Diagnostic<Error>> {
        if ty.ident.use_count() == 1 && !BUILTIN_NAMES.contains(&ty.ident.name().as_str()) {
            Err(E::UndefinedSymbol(ty.ident.to_string()).into())
        } else {
            for arg in ty.template_args.iter().flatten() {
                check_expr(&arg.expression)?;
            }
            Ok(())
        }
    }
    fn check_expr(expr: &ExpressionNode) -> Result<(), Diagnostic<Error>> {
        if let Expression::TypeOrIdentifier(ty) = expr.node() {
            check_ty(ty).map_err(|d| d.with_span(expr.span().clone()))
        } else if let Expression::FunctionCall(call) = expr.node() {
            check_ty(&call.ty).map_err(|d| d.with_span(expr.span().clone()))?;
            for expr in &call.arguments {
                check_expr(expr)?;
            }
            Ok(())
        } else {
            for expr in Visit::<ExpressionNode>::visit(expr.node()) {
                check_expr(expr)?;
            }
            Ok(())
        }
    }
    fn check_decl(decl: &GlobalDeclaration) -> Result<(), Diagnostic<Error>> {
        let decl_name = decl.ident().map(|ident| ident.name().to_string());
        for expr in Visit::<ExpressionNode>::visit(decl) {
            check_expr(expr).map_err(|mut d| {
                d.declaration = decl_name.clone();
                d
            })?;
        }

        // those are the attributes that don't have an expression as parent.
        // unfortunately the diagnostic won't have a span :(
        for ty in query!(decl.{
            GlobalDeclaration::Declaration.ty.[],
            GlobalDeclaration::TypeAlias.ty,
            GlobalDeclaration::Struct.members.[].ty,
            GlobalDeclaration::Function.{ parameters.[].ty, return_type.[] }
        }) {
            check_ty(ty).map_err(|mut d| {
                d.declaration = decl_name.clone();
                d
            })?;
        }
        Ok(())
    }

    for decl in &wesl.global_declarations {
        check_decl(decl)?;
    }
    Ok(())
}

fn check_function_calls(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    fn check_expr(expr: &Expression, wesl: &TranslationUnit) -> Result<(), E> {
        if let Expression::FunctionCall(call) = expr {
            let decl = wesl
                .global_declarations
                .iter()
                .find(|decl| decl.ident().is_some_and(|id| id == &call.ty.ident));

            match decl {
                Some(GlobalDeclaration::Function(decl)) => {
                    if call.arguments.len() != decl.parameters.len() {
                        return Err(E::ParamCount(
                            call.ty.ident.to_string(),
                            decl.parameters.len(),
                            call.arguments.len(),
                        ));
                    }
                }
                Some(GlobalDeclaration::Struct(decl)) => {
                    if call.arguments.len() != decl.members.len() && !call.arguments.is_empty() {
                        return Err(E::ParamCount(
                            call.ty.ident.to_string(),
                            decl.members.len(),
                            call.arguments.len(),
                        ));
                    }
                }
                Some(GlobalDeclaration::TypeAlias(_)) => {
                    // TODO: resolve type-alias
                }
                Some(_) => return Err(E::NotCallable(call.ty.ident.to_string())),
                None => {
                    if BUILTIN_FUNCTIONS
                        .iter()
                        .any(|name| name == &*call.ty.ident.name())
                    {
                        // TODO: check num args for builtin functions
                    } else {
                        // the ident is not a global declaration, it must be a local variable.
                        return Err(E::NotCallable(call.ty.ident.to_string()));
                    }
                }
            };
        }
        Ok(())
    }
    for decl in &wesl.global_declarations {
        for expr in Visit::<ExpressionNode>::visit(decl) {
            check_expr(expr, wesl).map_err(|e| {
                let mut err = Diagnostic::from(e);
                err.span = Some(expr.span().clone());
                err.declaration = decl.ident().map(|id| id.name().to_string());
                err
            })?;
        }
    }
    Ok(())
}

fn check_no_duplicate_decl(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    let mut unique = HashSet::new();
    for decl in &wesl.global_declarations {
        if let Some(id) = decl.ident() {
            if !unique.insert(id.to_string()) {
                return Err(
                    Diagnostic::from(E::Duplicate(id.to_string())).with_declaration(id.to_string())
                );
            }
        }
    }
    Ok(())
}

/// Validate a *resolved* WESL module. Must be called on module resolutions.
/// Resolved: has no imports, no qualified idents and no conditional translation.
/// Used idents must have use_count > 1.
pub(crate) fn validate_wesl(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    check_defined_symbols(wesl)?;
    check_no_duplicate_decl(wesl)?;
    Ok(())
}

/// Validate the final output (valid WGSL).
pub fn validate_wgsl(wgsl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    check_defined_symbols(wgsl)?;
    check_no_duplicate_decl(wgsl)?;
    check_function_calls(wgsl)?;
    Ok(())
}
