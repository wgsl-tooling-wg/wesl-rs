use std::iter::zip;

use crate::{
    eval::{ty_eval_ty, Context, Eval, EvalError, Exec, Ty, Type, ATTR_INTRINSIC},
    visit::Visit,
};
use wesl_macros::query_mut;
use wgsl_parse::{span::Spanned, syntax::*};

use super::{
    to_expr::ToExpr, with_scope, EvalStage, EvalTy, Instance, SyntaxUtil, EXPR_FALSE, EXPR_TRUE,
};

type E = EvalError;

// TODO: I am aware that it is not correct to make all implicit conversions explicit.
// I should fix that at some point, but meanwhile it fixes Naga not supporting automatic conversions.
pub fn make_explicit_conversions(wesl: &mut TranslationUnit, ctx: &mut Context) -> Result<(), E> {
    fn explicit_call(call: &mut FunctionCall, ctx: &mut Context) -> Result<(), E> {
        let decl = ctx.source.decl_function(&call.ty.ident.name());
        if let Some(decl) = decl {
            // we only do explicit conversions on user-defined functions,
            // because built-in functions have overloads for abstract types.
            if !decl.body.attributes.contains(&ATTR_INTRINSIC) {
                for (arg, param) in zip(&mut call.arguments, &decl.parameters) {
                    let ty = ty_eval_ty(&param.ty, ctx)?;
                    if ty.inner_ty().is_scalar() {
                        let ty = ty.to_expr(ctx)?.unwrap_type_or_identifier();
                        *arg.node_mut() = Expression::FunctionCall(FunctionCall {
                            ty,
                            arguments: vec![arg.clone()],
                        })
                    }
                }
            }
        }
        Ok(())
    }
    fn explicit_expr(expr: &mut Expression, ctx: &mut Context) -> Result<(), E> {
        if let Expression::FunctionCall(call) = expr {
            explicit_call(call, ctx)?;
        }
        for expr in Visit::<ExpressionNode>::visit_mut(expr) {
            explicit_expr(expr, ctx)?;
        }
        Ok(())
    }
    for expr in Visit::<ExpressionNode>::visit_mut(wesl) {
        explicit_expr(expr, ctx)?;
    }

    fn explicit_stat(stmt: &mut Statement, ret: &Type, ctx: &mut Context) -> Result<(), E> {
        if let Statement::Return(stmt) = stmt {
            if let Some(expr) = &mut stmt.expression {
                let ty = ret.to_expr(ctx)?.unwrap_type_or_identifier();
                *expr.node_mut() = Expression::FunctionCall(FunctionCall {
                    ty,
                    arguments: vec![expr.clone()],
                })
            }
        } else if let Statement::FunctionCall(stmt) = stmt {
            explicit_call(&mut stmt.call, ctx)?;
        }
        for stmt in Visit::<StatementNode>::visit_mut(stmt) {
            explicit_stat(stmt, ret, ctx)?;
        }
        Ok(())
    }
    for decl in query_mut!(wesl.global_declarations.[].GlobalDeclaration::Function) {
        if let Some(ret) = &decl.return_type {
            let ty = ty_eval_ty(ret, ctx)?;
            if ty.inner_ty().is_scalar() {
                for stmt in &mut decl.body.statements {
                    explicit_stat(stmt, &ty, ctx)?;
                }
            }
        }
    }
    Ok(())
}

pub trait Lower {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E>;
}

impl<T: Lower> Lower for Option<T> {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        if let Some(x) = self {
            x.lower(ctx)?;
        }
        Ok(())
    }
}

impl<T: Lower> Lower for Spanned<T> {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.node_mut()
            .lower(ctx)
            .inspect_err(|_| ctx.set_err_span_ctx(self.span()))?;
        Ok(())
    }
}

impl Lower for Expression {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        match self.eval_value(ctx) {
            Ok(inst) => *self = inst.to_expr(ctx)?,
            // `NotAccessible` is supposed to be the only possible error when evaluating valid code.
            Err(E::NotAccessible(_, EvalStage::Const)) => {
                ctx.err_span = None;
                match self {
                    Expression::Literal(_) => (),
                    Expression::Parenthesized(expr) => expr.expression.lower(ctx)?,
                    Expression::NamedComponent(expr) => expr.base.lower(ctx)?,
                    Expression::Indexing(expr) => {
                        expr.base.lower(ctx)?;
                        expr.index.lower(ctx)?;
                    }
                    Expression::Unary(expr) => expr.operand.lower(ctx)?,
                    Expression::Binary(expr) => {
                        expr.left.lower(ctx)?;
                        expr.right.lower(ctx)?;
                    }
                    Expression::FunctionCall(expr) => expr.lower(ctx)?,
                    Expression::TypeOrIdentifier(_) => {
                        if let Ok(expr) = self.eval_value(ctx).and_then(|inst| inst.to_expr(ctx)) {
                            *self = expr;
                        }
                    }
                }
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }
}

impl Lower for FunctionCall {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.ty = ctx.source.resolve_ty(&self.ty).clone();
        for arg in &mut self.arguments {
            arg.lower(ctx)?;
        }
        Ok(())
    }
}

impl Lower for TemplateArgs {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        if let Some(tplts) = self {
            for tplt in tplts {
                tplt.expression.lower(ctx)?;
            }
        }
        Ok(())
    }
}

impl Lower for TypeExpression {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        // types must be const-expressions
        let expr = ty_eval_ty(self, ctx)?.to_expr(ctx)?;
        *self = match expr {
            Expression::TypeOrIdentifier(ty) => ty,
            _ => unreachable!("eval_ty must return Literal"),
        };
        Ok(())
    }
}

impl Lower for Attributes {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        for attr in self {
            match attr {
                Attribute::Align(expr)
                | Attribute::Binding(expr)
                | Attribute::BlendSrc(expr)
                | Attribute::Group(expr)
                | Attribute::Id(expr)
                | Attribute::Location(expr)
                | Attribute::Size(expr) => {
                    expr.lower(ctx)?;
                }
                Attribute::WorkgroupSize(attr) => {
                    attr.x.lower(ctx)?;
                    attr.y.lower(ctx)?;
                    attr.z.lower(ctx)?;
                }
                Attribute::Custom(_) => {
                    // we ignore unknown attributes for now. We don't know how they are implemented.
                }
                _ => (),
            }
        }
        Ok(())
    }
}

impl Lower for Declaration {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.attributes.lower(ctx)?;
        self.ty.lower(ctx)?;
        self.initializer.lower(ctx)?;

        // TODO: this is copy-pasted 3 times now.
        let ty = match (&self.ty, &self.initializer) {
            (None, None) => return Err(E::UntypedDecl),
            (None, Some(init)) => {
                let ty = init.eval_ty(ctx)?;
                if self.kind.is_const() {
                    ty // only const declarations can be of abstract type.
                } else {
                    ty.concretize()
                }
            }
            (Some(ty), _) => ty_eval_ty(ty, ctx)?,
        };

        if ctx
            .scope
            .add(self.ident.to_string(), Instance::Deferred(ty))
        {
            Ok(())
        } else {
            return Err(E::DuplicateDecl(self.ident.to_string()));
        }
    }
}

impl Lower for TypeAlias {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.ty.lower(ctx)?;
        Ok(())
    }
}

impl Lower for Struct {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        for m in &mut self.members {
            m.attributes.lower(ctx)?;
            m.ty.lower(ctx)?;
        }
        Ok(())
    }
}

impl Lower for Function {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.attributes.contains(&Attribute::Const) && self.return_type.is_none() {
            self.body.statements.clear();
        }
        self.attributes.lower(ctx)?;
        for p in &mut self.parameters {
            p.attributes.lower(ctx)?;
            p.ty.lower(ctx)?;
        }
        self.return_attributes.lower(ctx)?;
        self.return_type.lower(ctx)?;

        with_scope!(ctx, {
            for p in &self.parameters {
                let inst = Instance::Deferred(ty_eval_ty(&p.ty, ctx)?);
                if !ctx.scope.add(p.ident.to_string(), inst) {
                    return Err(E::DuplicateDecl(p.ident.to_string()));
                }
            }
            compound_lower(&mut self.body, ctx, true)?;
            Ok(())
        })
    }
}

impl Lower for ConstAssert {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.exec(ctx).map(|_| ())
    }
}

impl Lower for Statement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        match self {
            Statement::Void => (),
            Statement::Compound(stmt) => {
                stmt.lower(ctx)?;
                if stmt.statements.is_empty() {
                    *self = Statement::Void;
                } else if let [stmt] = stmt.statements.as_slice() {
                    *self = stmt.node().clone();
                }
            }
            Statement::Assignment(stmt) => stmt.lower(ctx)?,
            Statement::Increment(stmt) => stmt.lower(ctx)?,
            Statement::Decrement(stmt) => stmt.lower(ctx)?,
            Statement::If(stmt) => {
                stmt.lower(ctx)?;

                // remove clauses evaluating to false
                stmt.else_if_clauses
                    .retain(|clause| *clause.expression != EXPR_FALSE);

                // remove subsequent clauses after a true
                if let Some(i) = stmt
                    .else_if_clauses
                    .iter()
                    .position(|clause| *clause.expression == EXPR_TRUE)
                {
                    stmt.else_if_clauses.resize_with(i + 1, || unreachable!());
                    stmt.else_clause = None;
                }

                macro_rules! assign_clause {
                    ($stmt:ident, $body:expr) => {
                        if $body.statements.is_empty() {
                            *$stmt = Statement::Void;
                        } else if let [s1] = $body.statements.as_slice() {
                            *$stmt = s1.node().clone();
                        } else {
                            *$stmt = Statement::Compound($body.clone())
                        }
                    };
                }

                // remove the whole statement if the first clause is true
                if *stmt.if_clause.expression == EXPR_TRUE {
                    assign_clause!(self, stmt.if_clause.body);
                } else if *stmt.if_clause.expression == EXPR_FALSE {
                    if let Some(clause) = stmt.else_if_clauses.first() {
                        if *clause.expression == EXPR_TRUE {
                            assign_clause!(self, clause.body);
                        }
                    } else if let Some(clause) = &stmt.else_clause {
                        assign_clause!(self, clause.body);
                    }
                }
            }
            Statement::Switch(stmt) => stmt.lower(ctx)?,
            Statement::Loop(stmt) => stmt.lower(ctx)?,
            Statement::For(stmt) => {
                stmt.lower(ctx)?;
                if stmt
                    .condition
                    .as_ref()
                    .is_some_and(|cond| **cond == EXPR_FALSE)
                {
                    *self = Statement::Void;
                }
            }
            Statement::While(stmt) => {
                stmt.lower(ctx)?;
                if *stmt.condition == EXPR_FALSE {
                    *self = Statement::Void;
                }
            }
            Statement::Break(_) => (),
            Statement::Continue(_) => (),
            Statement::Return(stmt) => stmt.lower(ctx)?,
            Statement::Discard(_) => (),
            Statement::FunctionCall(stmt) => {
                let decl = ctx.source.decl_function(&stmt.call.ty.ident.name());
                if let Some(decl) = decl {
                    if decl.attributes.contains(&Attribute::Const)
                        && !decl.attributes.contains(&Attribute::MustUse)
                    {
                        *self = Statement::Void; // a const function has no side-effects
                    } else {
                        stmt.lower(ctx)?
                    }
                } else {
                    stmt.lower(ctx)?
                }
            }
            Statement::ConstAssert(stmt) => stmt.exec(ctx).map(|_| ())?,
            Statement::Declaration(stmt) => {
                if stmt.kind == DeclarationKind::Const {
                    // eval and add it to the scope
                    stmt.exec(ctx)?;
                    *self = Statement::Void;
                } else {
                    stmt.lower(ctx)?;
                }
            }
        }
        Ok(())
    }
}

fn compound_lower(
    stmt: &mut CompoundStatement,
    ctx: &mut Context,
    transparent: bool,
) -> Result<(), E> {
    stmt.attributes.lower(ctx)?;
    with_scope!(ctx, {
        if transparent {
            ctx.scope.make_transparent();
        }
        for stmt in &mut stmt.statements {
            stmt.lower(ctx)?;
        }
        Ok(())
    })?;
    stmt.statements.retain(|stmt| match stmt.node() {
        Statement::Void => false,
        Statement::Compound(_) => true,
        Statement::Assignment(_) => true,
        Statement::Increment(_) => true,
        Statement::Decrement(_) => true,
        Statement::If(_) => true,
        Statement::Switch(_) => true,
        Statement::Loop(_) => true,
        Statement::For(_) => true,
        Statement::While(_) => true,
        Statement::Break(_) => true,
        Statement::Continue(_) => true,
        Statement::Return(_) => true,
        Statement::Discard(_) => true,
        Statement::FunctionCall(_) => true,
        Statement::ConstAssert(_) => false,
        Statement::Declaration(_) => true,
    });
    Ok(())
}

impl Lower for CompoundStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        compound_lower(self, ctx, false)
    }
}

impl Lower for AssignmentStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.lhs.lower(ctx)?;
        self.rhs.lower(ctx)?;
        Ok(())
    }
}

impl Lower for IncrementStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.expression.lower(ctx)?;
        Ok(())
    }
}

impl Lower for DecrementStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.expression.lower(ctx)?;
        Ok(())
    }
}

impl Lower for IfStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.attributes.lower(ctx)?;
        self.if_clause.expression.lower(ctx)?;
        self.if_clause.body.lower(ctx)?;
        for clause in &mut self.else_if_clauses {
            clause.expression.lower(ctx)?;
            clause.body.lower(ctx)?;
        }
        if let Some(clause) = &mut self.else_clause {
            clause.body.lower(ctx)?;
        }
        Ok(())
    }
}

impl Lower for SwitchStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.attributes.lower(ctx)?;
        self.expression.lower(ctx)?;
        self.body_attributes.lower(ctx)?;
        for clause in &mut self.clauses {
            for sel in &mut clause.case_selectors {
                if let CaseSelector::Expression(expr) = sel {
                    expr.lower(ctx)?;
                }
            }
            clause.body.lower(ctx)?;
        }
        Ok(())
    }
}

impl Lower for LoopStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.body.lower(ctx)?;
        if let Some(cont) = &mut self.continuing {
            cont.body.lower(ctx)?;
            if let Some(break_if) = &mut cont.break_if {
                break_if.expression.lower(ctx)?;
            }
        }
        Ok(())
    }
}

impl Lower for ForStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        // the initializer is in the same scope as the body.
        // https://github.com/gpuweb/gpuweb/issues/5024
        with_scope!(ctx, {
            self.initializer.lower(ctx)?;
            self.condition.lower(ctx)?;
            compound_lower(&mut self.body, ctx, true)?;
            Ok(())
        })
    }
}

impl Lower for WhileStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.condition.lower(ctx)?;
        self.body.lower(ctx)?;
        Ok(())
    }
}

impl Lower for ReturnStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.expression.lower(ctx)?;
        Ok(())
    }
}

impl Lower for FunctionCallStatement {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.call.lower(ctx)
    }
}

impl Lower for TranslationUnit {
    fn lower(&mut self, ctx: &mut Context) -> Result<(), E> {
        for decl in &mut self.global_declarations {
            match decl {
                GlobalDeclaration::Void => Ok(()),
                GlobalDeclaration::Declaration(decl) => {
                    if decl.kind == DeclarationKind::Const {
                        // eval and add it to the scope
                        decl.exec(ctx).map(|_| ())
                    } else {
                        decl.lower(ctx)
                    }
                }
                GlobalDeclaration::TypeAlias(decl) => decl.lower(ctx),
                GlobalDeclaration::Struct(decl) => decl.lower(ctx),
                GlobalDeclaration::Function(decl) => decl.lower(ctx),
                GlobalDeclaration::ConstAssert(decl) => decl.lower(ctx),
            }
            .inspect_err(|_| {
                decl.ident()
                    .inspect(|&ident| ctx.set_err_decl_ctx(ident.to_string()));
            })?;
        }
        self.global_declarations.retain(|decl| match decl {
            GlobalDeclaration::Void => false,
            GlobalDeclaration::Declaration(decl) => decl.kind != DeclarationKind::Const,
            GlobalDeclaration::TypeAlias(_) => false,
            GlobalDeclaration::Struct(_) => true,
            GlobalDeclaration::Function(_) => true,
            GlobalDeclaration::ConstAssert(_) => false,
        });
        Ok(())
    }
}
