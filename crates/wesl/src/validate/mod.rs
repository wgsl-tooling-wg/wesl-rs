use std::collections::HashSet;

use wesl_macros::query;
use wgsl_parse::syntax::{
    Expression, ExpressionNode, FunctionCall, GlobalDeclaration, Ident, Statement, StatementNode,
    TranslationUnit, TypeExpression,
};

use crate::builtin::{BUILTIN_FUNCTIONS, BUILTIN_NAMES, RESERVED_WORDS};
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
    #[error("declaration of `{0}` is cyclic via `{1}`")]
    Cycle(String, String),
    #[error("use of reserved word `{0}`")]
    ReservedWord(String),
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
    fn check_call(call: &FunctionCall, ident: &Ident, wesl: &TranslationUnit) -> Result<(), E> {
        let decl = wesl
            .global_declarations
            .iter()
            .find(|decl| decl.ident().is_some_and(|id| id == ident));

        match decl {
            Some(GlobalDeclaration::Function(decl)) => {
                if call.arguments.len() != decl.parameters.len() {
                    return Err(E::ParamCount(
                        ident.to_string(),
                        decl.parameters.len(),
                        call.arguments.len(),
                    ));
                }
            }
            Some(GlobalDeclaration::Struct(decl)) => {
                if call.arguments.len() != decl.members.len() && !call.arguments.is_empty() {
                    return Err(E::ParamCount(
                        ident.to_string(),
                        decl.members.len(),
                        call.arguments.len(),
                    ));
                }
            }
            Some(GlobalDeclaration::TypeAlias(decl)) => {
                if decl.ty.template_args.is_some() {
                    return Err(E::NotCallable(ident.to_string()));
                } else {
                    check_call(call, &decl.ty.ident, wesl)?;
                }
            }
            Some(_) => return Err(E::NotCallable(ident.to_string())),
            None => {
                if BUILTIN_FUNCTIONS.iter().any(|name| name == &*ident.name()) {
                    // TODO: check num args for builtin functions
                } else {
                    // the ident is not a global declaration, it must be a local variable.
                    return Err(E::NotCallable(ident.to_string()));
                }
            }
        };
        Ok(())
    }
    fn check_expr(expr: &Expression, wesl: &TranslationUnit) -> Result<(), E> {
        if let Expression::FunctionCall(call) = expr {
            check_call(call, &call.ty.ident, wesl)?;
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

fn check_duplicate_decl(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
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

fn check_cycles(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    fn check_decl(
        id: &Ident,
        decl: &GlobalDeclaration,
        unique: &mut HashSet<Ident>,
        wesl: &TranslationUnit,
    ) -> Result<(), E> {
        for ty in Visit::<TypeExpression>::visit(decl) {
            if ty.ident == *id {
                return Err(E::Cycle(id.to_string(), decl.ident().unwrap().to_string()));
            } else if unique.insert(ty.ident.clone()) {
                if let Some(decl) = wesl
                    .global_declarations
                    .iter()
                    .find(|decl| decl.ident() == Some(&ty.ident))
                {
                    check_decl(id, decl, unique, wesl)?;
                }
            }
        }
        Ok(())
    }
    for decl in &wesl.global_declarations {
        if let Some(id) = decl.ident() {
            let mut unique = HashSet::new();
            check_decl(id, decl, &mut unique, wesl)
                .map_err(|e| Diagnostic::from(e).with_declaration(id.to_string()))?;
        }
    }
    Ok(())
}

fn check_reserved_words(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    fn check_ident(id: &Ident) -> Result<(), Diagnostic<Error>> {
        if RESERVED_WORDS.contains(&id.name().as_str()) {
            return Err(E::ReservedWord(id.to_string()).into());
        }
        Ok(())
    }
    fn check_stmt(stmt: &StatementNode) -> Result<(), Diagnostic<Error>> {
        if let Statement::Declaration(decl) = stmt.node() {
            check_ident(&decl.ident).map_err(|d| d.with_span(stmt.span().clone()))?;
        }
        for stmt in Visit::<StatementNode>::visit(stmt.node()) {
            check_stmt(stmt)?;
        }
        Ok(())
    }
    fn check_decl(decl: &GlobalDeclaration) -> Result<(), Diagnostic<Error>> {
        for id in query!(decl.{
            // GlobalDeclaration::Import.content
            GlobalDeclaration::Declaration.ident,
            GlobalDeclaration::TypeAlias.ident,
            GlobalDeclaration::Struct.{
                ident,
                members.[].ident,
            },
            GlobalDeclaration::Function.{
                ident,
                parameters.[].ident,
            }
        }) {
            check_ident(id)?;
        }

        if let GlobalDeclaration::Function(decl) = decl {
            for stmt in &decl.body.statements {
                check_stmt(stmt)?;
            }
        }

        Ok(())
    }

    for decl in &wesl.global_declarations {
        check_decl(decl).map_err(|mut d| {
            d.declaration = decl.ident().map(Ident::to_string);
            d
        })?;
    }
    Ok(())
}

/// Validate an intermediate WESL module.
///
/// This function only checks that a WESL module is valid on its own, without looking at
/// external modules (imports).
///
/// It currently does not validate a lot. It checks for:
/// * Reserved words: no declaration identifier is a reserved name.
/// * Defined declarations: all identifiers refer to a user declaration, import or
///   built-in name.
/// * Duplicate declarations: declarations in the same scope cannot have the same name.
/// * Cyclic declarations: no cycles are allowed in declarations.
pub fn validate_wesl(wesl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    check_reserved_words(wesl)?;
    check_defined_symbols(wesl)?;
    check_duplicate_decl(wesl)?;
    check_cycles(wesl)?;
    Ok(())
}

/// Validate the final output (valid WGSL).
///
/// It currently does not validate a lot. It checks for:
/// * Reserved words: no declaration identifier is a reserved name.
/// * Defined declarations: all identifiers refer to a user declaration or built-in name.
/// * Duplicate declarations: declarations in the same scope cannot have the same name.
/// * Cyclic declarations: no cycles are allowed in declarations.
/// * Function calls: call expressions must refer to a function or a type constructor.
///   Check the number of arguments but not their type.
pub fn validate_wgsl(wgsl: &TranslationUnit) -> Result<(), Diagnostic<Error>> {
    check_reserved_words(wgsl)?;
    check_defined_symbols(wgsl)?;
    check_duplicate_decl(wgsl)?;
    check_cycles(wgsl)?;
    check_function_calls(wgsl)?;
    Ok(())
}
