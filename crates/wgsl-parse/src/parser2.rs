use parser::{ParseEntryPoint, parse_entrypoint_with_capabilities};
use syntax::{AstNode, HasAttributes, HasName, ast};

use crate::{
    Error,
    span::{Span, Spanned},
    syntax::{
        Attribute, CompoundGlobalDeclaration, CompoundStatement, ConstAssert, Declaration,
        DeclarationKind, DiagnosticDirective, EnableDirective, Expression, Function,
        GlobalDeclaration, GlobalDirective, Ident, ImportStatement, RequiresDirective, Statement,
        Struct, StructMember, TranslationUnit, TypeAlias, TypeExpression,
    },
};

/// Parse a string into a syntax tree ([`TranslationUnit`]).
///
/// Identical to [`TranslationUnit::from_str`].
pub fn parse_str(source: &str) -> Result<TranslationUnit, Error> {
    let parsed = syntax::Parse::new(parse_entrypoint_with_capabilities(
        source,
        ParseEntryPoint::File,
        parser::Edition::Wesl2025Unstable,
        parser::Capabilities {
            shader_int64: true,
            early_depth_test: true,
        },
    ));

    if !parsed.errors().is_empty() {
        return Err(Error {
            error: crate::error::ErrorKind::InvalidToken,
            span: crate::span::Span::new(0..0),
        });
    }

    gen_translation_unit(&parsed.tree())
}

fn gen_translation_unit(ast: &ast::SourceFile) -> Result<TranslationUnit, Error> {
    Ok(TranslationUnit {
        imports: gen_imports(ast.items())?,
        global_directives: gen_directives(ast.directives())?,
        global_declarations: gen_global_declarations(ast.items())?,
    })
}

fn gen_attributes(
    item: &dyn HasAttributes,
) -> Result<Vec<Spanned<crate::syntax::Attribute>>, Error> {
    let Some(attributes) = item.attributes() else {
        return Ok(Vec::new());
    };
    attributes
        .map(|attribute| gen_attribute(&attribute))
        .collect()
}

fn gen_attribute(attribute: &ast::Attribute) -> Result<Spanned<Attribute>, Error> {
    fn one_expression(attribute: &ast::Attribute) -> Result<Spanned<Expression>, Error> {
        let mut arguments = attribute.arguments().expect_some(attribute)?.arguments();
        let expression = arguments.next().expect_some(attribute)?;
        arguments.next().expect_none(attribute)?;
        Ok(gen_expression(&expression)?)
    }

    let result = match attribute.kind() {
        ast::AttributeKind::Align => Attribute::Align(one_expression(attribute)?),
        ast::AttributeKind::Binding => Attribute::Binding(one_expression(attribute)?),
        ast::AttributeKind::BlendSrc => Attribute::BlendSrc(one_expression(attribute)?),
        ast::AttributeKind::Builtin => Attribute::Builtin(todo!()),
        ast::AttributeKind::Const => Attribute::Const,
        ast::AttributeKind::Diagnostic => Attribute::Diagnostic(todo!()),
        ast::AttributeKind::Group => Attribute::Group(one_expression(attribute)?),
        ast::AttributeKind::Id => Attribute::Id(one_expression(attribute)?),
        ast::AttributeKind::Interpolate => Attribute::Interpolate(todo!()),
        ast::AttributeKind::Invariant => Attribute::Invariant,
        ast::AttributeKind::Location => Attribute::Location(one_expression(attribute)?),
        ast::AttributeKind::MustUse => Attribute::MustUse,
        ast::AttributeKind::Size => Attribute::Size(one_expression(attribute)?),
        ast::AttributeKind::SubgroupSize => todo!(),
        ast::AttributeKind::WorkgroupSize => Attribute::WorkgroupSize(todo!()),
        ast::AttributeKind::Entrypoint(ast::EntrypointAttributeKind::Vertex) => Attribute::Vertex,
        ast::AttributeKind::Entrypoint(ast::EntrypointAttributeKind::Fragment) => {
            Attribute::Fragment
        }
        ast::AttributeKind::Entrypoint(ast::EntrypointAttributeKind::Compute) => Attribute::Compute,
        ast::AttributeKind::Conditional(ast::ConditionalAttributeKind::If) => {
            Attribute::If(one_expression(attribute)?)
        }
        ast::AttributeKind::Conditional(ast::ConditionalAttributeKind::Elif) => {
            Attribute::Elif(one_expression(attribute)?)
        }
        ast::AttributeKind::Conditional(ast::ConditionalAttributeKind::Else) => Attribute::Else,
        ast::AttributeKind::Other => todo!("also handle the experimental attributes"),
    };

    Ok(Spanned::new(result, attribute.span()))
}

fn gen_directives(
    directives: syntax::AstChildren<ast::Directive>,
) -> Result<Vec<GlobalDirective>, Error> {
    directives
        .map(|directive| gen_directive(&directive))
        .collect()
}

fn gen_directive(directive: &ast::Directive) -> Result<GlobalDirective, Error> {
    let result = match directive {
        ast::Directive::DiagnosticDirective(diagnostic_directive) => {
            GlobalDirective::Diagnostic(gen_diagnostic_directive(diagnostic_directive)?)
        }
        ast::Directive::EnableDirective(enable_directive) => {
            GlobalDirective::Enable(gen_enable_directive(enable_directive)?)
        }
        ast::Directive::RequiresDirective(requires_directive) => {
            GlobalDirective::Requires(gen_requires_directive(requires_directive)?)
        }
    };
    Ok(result)
}

fn gen_diagnostic_directive(
    diagnostic_directive: &ast::DiagnosticDirective,
) -> Result<DiagnosticDirective, Error> {
    todo!()
}

fn gen_enable_directive(enable_directive: &ast::EnableDirective) -> Result<EnableDirective, Error> {
    todo!()
}

fn gen_requires_directive(
    requires_directive: &ast::RequiresDirective,
) -> Result<RequiresDirective, Error> {
    todo!()
}

fn gen_imports(items: syntax::AstChildren<ast::Item>) -> Result<Vec<ImportStatement>, Error> {
    items
        .filter_map(|item| match item {
            ast::Item::ImportStatement(import_statement) => Some(import_statement),
            _ => None,
        })
        .map(|import_statement| gen_import_statement(&import_statement))
        .collect()
}

fn gen_import_statement(import_statement: &ast::ImportStatement) -> Result<ImportStatement, Error> {
    todo!()
}

fn gen_global_declarations(
    items: syntax::AstChildren<ast::Item>,
) -> Result<Vec<Spanned<GlobalDeclaration>>, Error> {
    items
        .filter(|item| match item {
            ast::Item::ImportStatement(_) => false,
            _ => true,
        })
        .map(|item| gen_global_declaration(&item))
        .collect()
}

/// # Panics
///
/// Panics if the given item is not a global declaration (an import statement).
fn gen_global_declaration(item: &ast::Item) -> Result<Spanned<GlobalDeclaration>, Error> {
    let declaration = match item {
        ast::Item::ImportStatement(_) => panic!("Not a global declaration"),
        ast::Item::FunctionDeclaration(function_declaration) => {
            GlobalDeclaration::Function(gen_function(function_declaration)?)
        }
        ast::Item::VariableDeclaration(variable_declaration) => {
            GlobalDeclaration::Declaration(gen_variable(variable_declaration)?)
        }
        ast::Item::ConstantDeclaration(constant_declaration) => {
            GlobalDeclaration::Declaration(gen_constant(constant_declaration)?)
        }
        ast::Item::OverrideDeclaration(override_declaration) => {
            GlobalDeclaration::Declaration(gen_override(override_declaration)?)
        }
        ast::Item::TypeAliasDeclaration(type_alias_declaration) => {
            GlobalDeclaration::TypeAlias(gen_type_alias(type_alias_declaration)?)
        }
        ast::Item::StructDeclaration(struct_declaration) => {
            GlobalDeclaration::Struct(gen_struct(struct_declaration)?)
        }
        ast::Item::AssertStatement(assert_statement) => {
            GlobalDeclaration::ConstAssert(gen_const_assert(assert_statement)?)
        }
        ast::Item::GlobalCompoundDeclaration(global_compound_declaration) => {
            GlobalDeclaration::Compound(gen_global_compound(global_compound_declaration)?)
        }
    };
    Ok(Spanned::new(declaration, item.span()))
}

fn gen_function(function_declaration: &ast::FunctionDeclaration) -> Result<Function, Error> {
    let return_attributes = function_declaration
        .return_type()
        .map(|rt| gen_attributes(&rt))
        .transpose()?
        .unwrap_or_default();
    let return_type = function_declaration
        .return_type()
        .map(|rt| gen_type_expression(&rt.r#type().expect_some(&rt)?))
        .transpose()?;
    Ok(Function {
        attributes: gen_attributes(function_declaration)?,
        visibility: todo!(),
        ident: gen_ident(function_declaration.name(), function_declaration)?,
        parameters: todo!(),
        return_attributes,
        return_type,
        body: gen_compound_statement(
            &function_declaration
                .body()
                .expect_some(function_declaration)?,
        )?,
    })
}

fn gen_variable(variable_declaration: &ast::VariableDeclaration) -> Result<Declaration, Error> {
    Ok(Declaration {
        attributes: gen_attributes(variable_declaration)?,
        visibility: todo!(),
        kind: DeclarationKind::Var(todo!()),
        ident: gen_ident(variable_declaration.name(), variable_declaration)?,
        ty: variable_declaration
            .r#type()
            .as_ref()
            .map(gen_type_expression)
            .transpose()?,
        initializer: variable_declaration
            .init()
            .as_ref()
            .map(gen_expression)
            .transpose()?,
    })
}

fn gen_constant(constant_declaration: &ast::ConstantDeclaration) -> Result<Declaration, Error> {
    Ok(Declaration {
        attributes: gen_attributes(constant_declaration)?,
        visibility: todo!(),
        kind: DeclarationKind::Const,
        ident: gen_ident(constant_declaration.name(), constant_declaration)?,
        ty: constant_declaration
            .r#type()
            .as_ref()
            .map(gen_type_expression)
            .transpose()?,
        initializer: constant_declaration
            .init()
            .as_ref()
            .map(gen_expression)
            .transpose()?,
    })
}
fn gen_override(override_declaration: &ast::OverrideDeclaration) -> Result<Declaration, Error> {
    Ok(Declaration {
        attributes: gen_attributes(override_declaration)?,
        visibility: todo!(),
        kind: DeclarationKind::Override,
        ident: gen_ident(override_declaration.name(), override_declaration)?,
        ty: override_declaration
            .r#type()
            .as_ref()
            .map(gen_type_expression)
            .transpose()?,
        initializer: override_declaration
            .init()
            .as_ref()
            .map(gen_expression)
            .transpose()?,
    })
}
fn gen_type_alias(type_alias_declaration: &ast::TypeAliasDeclaration) -> Result<TypeAlias, Error> {
    Ok(TypeAlias {
        attributes: gen_attributes(type_alias_declaration)?,
        visibility: todo!(),
        ident: gen_ident(type_alias_declaration.name(), type_alias_declaration)?,
        ty: gen_type_expression(
            &type_alias_declaration
                .type_declaration()
                .expect_some(type_alias_declaration)?,
        )?,
    })
}
fn gen_struct(struct_declaration: &ast::StructDeclaration) -> Result<Struct, Error> {
    let members = struct_declaration
        .body()
        .expect_some(struct_declaration)?
        .fields()
        .map(|field| gen_struct_member(&field))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Struct {
        attributes: gen_attributes(struct_declaration)?,
        visibility: todo!(),
        ident: gen_ident(struct_declaration.name(), struct_declaration)?,
        members,
    })
}

fn gen_struct_member(field: &ast::StructMember) -> Result<Spanned<StructMember>, Error> {
    todo!()
}

fn gen_const_assert(assert_statement: &ast::AssertStatement) -> Result<ConstAssert, Error> {
    Ok(ConstAssert {
        attributes: gen_attributes(assert_statement)?,
        expression: gen_expression(
            &assert_statement
                .expression()
                .expect_some(assert_statement)?,
        )?,
    })
}

fn gen_global_compound(
    global_compound_declaration: &ast::GlobalCompoundDeclaration,
) -> Result<CompoundGlobalDeclaration, Error> {
    todo!()
}

fn gen_compound_statement(statement: &ast::CompoundStatement) -> Result<CompoundStatement, Error> {
    let statements = statement
        .statements()
        .map(|statement| gen_statement(&statement))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CompoundStatement {
        attributes: gen_attributes(statement)?,
        statements,
    })
}

fn gen_statement(statement: &ast::Statement) -> Result<Spanned<Statement>, Error> {
    todo!()
}

fn gen_type_expression(ty: &ast::TypeSpecifier) -> Result<TypeExpression, Error> {
    todo!()
}

fn gen_expression(expression: &ast::Expression) -> Result<Spanned<Expression>, Error> {
    todo!()
}

fn gen_ident(name: Option<ast::Name>, node: &dyn AstNode) -> Result<Ident, Error> {
    name.expect_some(node)
        .map(|name| Ident::new(name.text().to_owned()))
}

trait WithSpan {
    fn span(&self) -> Span;
}
impl<T: ?Sized + AstNode> WithSpan for T {
    fn span(&self) -> Span {
        let range = self.syntax().text_range();
        let start = usize::from(range.start());
        let end = usize::from(range.end());
        Span { start, end }
    }
}

trait AstOption<T> {
    fn expect_some(self, node: &dyn AstNode) -> Result<T, Error>;
    fn expect_none(self, node: &dyn AstNode) -> Result<(), Error>;
}

impl<T> AstOption<T> for Option<T> {
    fn expect_some(self, node: &dyn AstNode) -> Result<T, Error> {
        match self {
            Some(value) => Ok(value),
            None => Err(Error {
                error: crate::error::ErrorKind::InvalidToken,
                span: node.span(),
            }),
        }
    }

    fn expect_none(self, node: &dyn AstNode) -> Result<(), Error> {
        match self {
            Some(_) => Err(Error {
                error: crate::error::ErrorKind::InvalidToken,
                span: node.span(),
            }),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_const() {
        let module = parse_str("const a = 3;").unwrap();
    }
}
