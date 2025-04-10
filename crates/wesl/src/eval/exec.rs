use std::{fmt::Display, iter::zip};

use crate::eval::conv::Convert;

use super::{
    ATTR_INTRINSIC, AccessMode, Context, Eval, EvalError, EvalStage, EvalTy, Instance,
    LiteralInstance, RefInstance, ScopeKind, StructInstance, SyntaxUtil, Ty, Type,
    attrs::EvalAttrs, call_builtin, is_constructor_fn, ty_eval_ty,
};

use wgsl_parse::{span::Spanned, syntax::*};

type E = EvalError;

// reference: https://www.w3.org/TR/WGSL/#behaviors
#[derive(Clone, Debug, PartialEq)]
pub enum Flow {
    Next,
    Break,
    Continue,
    Return(Option<Instance>),
}

impl From<Instance> for Flow {
    fn from(inst: Instance) -> Self {
        Self::Return(Some(inst))
    }
}

macro_rules! with_stage {
    ($ctx:expr, $stage:expr, $body:tt) => {{
        let stage = $ctx.stage;
        $ctx.stage = $stage;
        #[allow(clippy::redundant_closure_call)]
        let body = (|| $body)();
        $ctx.stage = stage;
        body
    }};
}

macro_rules! with_scope {
    ($ctx:expr, $body:tt) => {{
        $ctx.scope.push();
        #[allow(clippy::redundant_closure_call)]
        let body = (|| $body)();
        $ctx.scope.pop();
        body
    }};
}

macro_rules! module_scope {
    ($ctx:expr, $body:tt) => {{
        assert!($ctx.scope.is_root());
        let kind = $ctx.kind;
        $ctx.kind = ScopeKind::Module;
        #[allow(clippy::redundant_closure_call)]
        let body = (|| $body)();
        $ctx.kind = kind;
        body
    }};
}
pub(super) use with_scope;
pub(super) use with_stage;

impl Display for Flow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Flow::Next => write!(f, "void"),
            Flow::Break => write!(f, "break"),
            Flow::Continue => write!(f, "continue"),
            Flow::Return(_) => write!(f, "return"),
        }
    }
}

pub trait Exec {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E>;
}

impl<T: Exec> Exec for Spanned<T> {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        self.node().exec(ctx).inspect_err(|_| {
            ctx.set_err_span_ctx(self.span());
        })
    }
}

impl Exec for TranslationUnit {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        module_scope!(ctx, {
            for decl in &ctx.source.global_declarations {
                let flow = decl.exec(ctx)?;
                match flow {
                    Flow::Next => (),
                    Flow::Break | Flow::Continue | Flow::Return(_) => {
                        decl.ident()
                            .inspect(|&ident| ctx.set_err_decl_ctx(ident.to_string()));
                        return Err(E::FlowInModule(flow));
                    }
                }
            }

            Ok(Flow::Next)
        })
    }
}

impl Exec for GlobalDeclaration {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        match self {
            GlobalDeclaration::Declaration(decl) => {
                if ctx.scope.contains(&decl.ident.name()) {
                    // because of module-scope hoisting, declarations may be executed out-of-order.
                    // TODO: check no duplicate declarations?
                    Ok(Flow::Next)
                } else {
                    decl.exec(ctx)
                }
            }
            GlobalDeclaration::ConstAssert(decl) => decl.exec(ctx),
            _ => Ok(Flow::Next),
        }
        .inspect_err(|_| {
            self.ident()
                .inspect(|&ident| ctx.set_err_decl_ctx(ident.to_string()));
        })
    }
}

impl Exec for Statement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        match self {
            Statement::Void => Ok(Flow::Next),
            Statement::Compound(s) => s.exec(ctx),
            Statement::Assignment(s) => s.exec(ctx),
            Statement::Increment(s) => s.exec(ctx),
            Statement::Decrement(s) => s.exec(ctx),
            Statement::If(s) => s.exec(ctx),
            Statement::Switch(s) => s.exec(ctx),
            Statement::Loop(s) => s.exec(ctx),
            Statement::For(s) => s.exec(ctx),
            Statement::While(s) => s.exec(ctx),
            Statement::Break(s) => s.exec(ctx),
            Statement::Continue(s) => s.exec(ctx),
            Statement::Return(s) => s.exec(ctx),
            Statement::Discard(s) => s.exec(ctx),
            Statement::FunctionCall(s) => s.exec(ctx),
            Statement::ConstAssert(s) => s.exec(ctx),
            Statement::Declaration(s) => s.exec(ctx),
        }
    }
}

impl Exec for CompoundStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        with_scope!(ctx, {
            for stmt in &self.statements {
                let flow = stmt.exec(ctx)?;
                match flow {
                    Flow::Next => (),
                    Flow::Break | Flow::Continue | Flow::Return(_) => {
                        return Ok(flow);
                    }
                }
            }

            Ok(Flow::Next)
        })
    }
}

// because some places in the grammar requires that no scope is created when executing the
// CompoundStatement, such as for loops with initializer or function invocations.
pub(crate) fn compound_exec_no_scope(
    stmt: &CompoundStatement,
    ctx: &mut Context,
) -> Result<Flow, E> {
    with_scope!(ctx, {
        ctx.scope.make_transparent();
        for stmt in &stmt.statements {
            let flow = stmt.exec(ctx)?;
            match flow {
                Flow::Next => (),
                Flow::Break | Flow::Continue | Flow::Return(_) => {
                    return Ok(flow);
                }
            }
        }
        Ok(Flow::Next)
    })
}

impl Exec for AssignmentStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        let lhs = self.lhs.eval(ctx)?;
        let ty = lhs.ty().concretize();

        if let Instance::Ref(mut r) = lhs {
            let rhs = self.rhs.eval_value(ctx)?;
            match self.operator {
                AssignmentOperator::Equal => {
                    let rhs = rhs
                        .convert_to(&ty)
                        .ok_or_else(|| E::AssignType(rhs.ty(), ty))?;
                    r.write(rhs)?;
                }
                AssignmentOperator::PlusEqual => {
                    let val = r.read()?.op_add(&rhs, ctx.stage)?;
                    r.write(val)?;
                }
                AssignmentOperator::MinusEqual => {
                    let val = r.read()?.op_sub(&rhs, ctx.stage)?;
                    r.write(val)?;
                }
                AssignmentOperator::TimesEqual => {
                    let val = r.read()?.op_mul(&rhs, ctx.stage)?;
                    r.write(val)?;
                }
                AssignmentOperator::DivisionEqual => {
                    let val = r.read()?.op_div(&rhs, ctx.stage)?;
                    r.write(val)?;
                }
                AssignmentOperator::ModuloEqual => {
                    let val = r.read()?.op_rem(&rhs, ctx.stage)?;
                    r.write(val)?;
                }
                AssignmentOperator::AndEqual => {
                    let val = r.read()?.op_bitand(&rhs)?;
                    r.write(val)?;
                }
                AssignmentOperator::OrEqual => {
                    let val = r.read()?.op_bitor(&rhs)?;
                    r.write(val)?;
                }
                AssignmentOperator::XorEqual => {
                    let val = r.read()?.op_bitxor(&rhs)?;
                    r.write(val)?;
                }
                AssignmentOperator::ShiftRightAssign => {
                    let val = r.read()?.op_shr(&rhs, ctx.stage)?;
                    r.write(val)?;
                }
                AssignmentOperator::ShiftLeftAssign => {
                    let val = r.read()?.op_shl(&rhs, ctx.stage)?;
                    r.write(val)?;
                }
            }
            Ok(Flow::Next)
        } else {
            Err(E::NotRef(lhs))
        }
    }
}

impl Exec for IncrementStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        let expr = self.expression.eval(ctx)?;
        if let Instance::Ref(mut r) = expr {
            let mut r = r.read_write()?;
            match &*r {
                Instance::Literal(LiteralInstance::I32(n)) => {
                    let val = n.checked_add(1).ok_or(E::IncrOverflow)?;
                    let _ = r.write(LiteralInstance::I32(val).into());
                    Ok(Flow::Next)
                }
                Instance::Literal(LiteralInstance::U32(n)) => {
                    let val = n.checked_add(1).ok_or(E::IncrOverflow)?;
                    let _ = r.write(LiteralInstance::U32(val).into());
                    Ok(Flow::Next)
                }
                i => Err(E::IncrType(i.ty())),
            }
        } else {
            Err(E::NotRef(expr))
        }
    }
}

impl Exec for DecrementStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        let expr = self.expression.eval(ctx)?;
        if let Instance::Ref(mut r) = expr {
            let mut r = r.read_write()?;
            match &*r {
                Instance::Literal(LiteralInstance::I32(n)) => {
                    let val = n.checked_sub(1).ok_or(E::DecrOverflow)?;
                    let _ = r.write(LiteralInstance::I32(val).into());
                    Ok(Flow::Next)
                }
                Instance::Literal(LiteralInstance::U32(n)) => {
                    let val = n.checked_sub(1).ok_or(E::DecrOverflow)?;
                    let _ = r.write(LiteralInstance::U32(val).into());
                    Ok(Flow::Next)
                }
                r => Err(E::DecrType(r.ty())),
            }
        } else {
            Err(E::NotRef(expr))
        }
    }
}

impl Exec for IfStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        {
            let expr = self.if_clause.expression.eval_value(ctx)?;
            let cond = match expr {
                Instance::Literal(LiteralInstance::Bool(b)) => Ok(b),
                _ => Err(E::Type(Type::Bool, expr.ty())),
            }?;

            if cond {
                let flow = self.if_clause.body.exec(ctx)?;
                return Ok(flow);
            }
        }

        for elif in &self.else_if_clauses {
            let expr = elif.expression.eval_value(ctx)?;
            let cond = match expr {
                Instance::Literal(LiteralInstance::Bool(b)) => Ok(b),
                _ => Err(E::Type(Type::Bool, expr.ty())),
            }?;
            if cond {
                let flow = elif.body.exec(ctx)?;
                return Ok(flow);
            }
        }

        if let Some(el) = &self.else_clause {
            let flow = el.body.exec(ctx)?;
            return Ok(flow);
        }

        Ok(Flow::Next)
    }
}

impl Exec for SwitchStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        let expr = self.expression.eval_value(ctx)?;
        let ty = expr.ty();

        for clause in &self.clauses {
            for selector in &clause.case_selectors {
                match selector {
                    CaseSelector::Default => return clause.body.exec(ctx),
                    CaseSelector::Expression(e) => {
                        let e = with_stage!(ctx, EvalStage::Const, { e.eval_value(ctx) })?;
                        let e = e
                            .convert_to(&ty)
                            .ok_or_else(|| E::Conversion(e.ty(), ty.clone()))?;
                        if e == expr {
                            return clause.body.exec(ctx);
                        }
                    }
                }
            }
        }

        Ok(Flow::Next)
    }
}

impl Exec for LoopStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        loop {
            let flow = self.body.exec(ctx)?;

            match flow {
                Flow::Next | Flow::Continue => {
                    if let Some(cont) = &self.continuing {
                        let flow = cont.exec(ctx)?;

                        match flow {
                            Flow::Next => (),
                            Flow::Break => return Ok(Flow::Next), // This must be a break-if, see impl Exec for ContinuingStatement
                            Flow::Continue => unreachable!("no continue in continuing"),
                            Flow::Return(_) => unreachable!("no return in continuing"),
                        }
                    }
                }
                Flow::Break => {
                    return Ok(Flow::Next);
                }
                Flow::Return(_) => {
                    return Ok(flow);
                }
            }
        }
    }
}

impl Exec for ContinuingStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        let flow = self.body.exec(ctx)?;
        match flow {
            Flow::Next => {
                if let Some(break_if) = &self.break_if {
                    let expr = break_if.expression.eval_value(ctx)?;
                    let cond = match expr {
                        Instance::Literal(LiteralInstance::Bool(b)) => Ok(b),
                        _ => Err(E::Type(Type::Bool, expr.ty())),
                    }?;
                    if cond {
                        Ok(Flow::Break)
                    } else {
                        Ok(Flow::Next)
                    }
                } else {
                    Ok(Flow::Next)
                }
            }
            Flow::Break | Flow::Continue | Flow::Return(_) => Err(E::FlowInContinuing(flow)),
        }
    }
}

impl Exec for ForStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        // the initializer is in the same scope as the body.
        // https://github.com/gpuweb/gpuweb/issues/5024
        with_scope!(ctx, {
            if let Some(init) = &self.initializer {
                let flow = init.exec(ctx)?;
                if flow != Flow::Next {
                    return Ok(flow);
                }
            }

            loop {
                let cond = self
                    .condition
                    .as_ref()
                    .map(|expr| {
                        let expr = expr.eval_value(ctx)?;
                        match expr {
                            Instance::Literal(LiteralInstance::Bool(b)) => Ok(b),
                            _ => Err(E::Type(Type::Bool, expr.ty())),
                        }
                    })
                    .unwrap_or(Ok(false))?;

                if !cond {
                    break;
                }

                // the body has to run in the same scope as the initializer.
                let flow = compound_exec_no_scope(&self.body, ctx)?;

                match flow {
                    Flow::Next | Flow::Continue => {
                        if let Some(updt) = &self.update {
                            updt.exec(ctx)?;
                        }
                    }
                    Flow::Break => {
                        break;
                    }
                    Flow::Return(_) => {
                        return Ok(flow);
                    }
                }
            }

            Ok(Flow::Next)
        })
    }
}

impl Exec for WhileStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        loop {
            let expr = self.condition.eval_value(ctx)?;
            let cond = match expr {
                Instance::Literal(LiteralInstance::Bool(b)) => Ok(b),
                _ => Err(E::Type(Type::Bool, expr.ty())),
            }?;

            if cond {
                let flow = self.body.exec(ctx)?;
                match flow {
                    Flow::Next | Flow::Continue => (),
                    Flow::Break => return Ok(Flow::Next),
                    Flow::Return(_) => return Ok(flow),
                }
            } else {
                return Ok(Flow::Next);
            }
        }
    }
}

impl Exec for BreakStatement {
    fn exec(&self, _ctx: &mut Context) -> Result<Flow, E> {
        Ok(Flow::Break)
    }
}

impl Exec for ContinueStatement {
    fn exec(&self, _ctx: &mut Context) -> Result<Flow, E> {
        Ok(Flow::Continue)
    }
}

impl Exec for ReturnStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        if let Some(e) = &self.expression {
            let inst = e.eval_value(ctx)?;
            Ok(Flow::Return(Some(inst)))
        } else {
            Ok(Flow::Return(None))
        }
    }
}

impl Exec for DiscardStatement {
    fn exec(&self, _ctx: &mut Context) -> Result<Flow, E> {
        Err(E::DiscardInConst)
    }
}

impl Exec for FunctionCallStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        let fn_name = self.call.ty.ident.to_string();

        let is_must_use = match ctx.source.decl(&fn_name) {
            Some(GlobalDeclaration::Function(decl)) => {
                decl.attributes.contains(&Attribute::MustUse)
            }
            Some(GlobalDeclaration::Struct(_)) => true,
            Some(_) => return Err(E::NotCallable(fn_name)),
            None => {
                if is_constructor_fn(&fn_name) {
                    true
                } else {
                    return Err(E::UnknownFunction(fn_name));
                }
            }
        };

        if is_must_use {
            return Err(E::MustUse(fn_name));
        }

        match self.call.eval(ctx) {
            Ok(_) => Ok(Flow::Next),
            Err(E::Void(_)) => Ok(Flow::Next),
            Err(e) => Err(e),
        }
    }
}

impl Exec for FunctionCall {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        let ty = ctx.source.resolve_ty(&self.ty);
        let fn_name = ty.ident.to_string();

        let args = self
            .arguments
            .iter()
            .map(|a| a.eval_value(ctx))
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(decl) = ctx.source.decl(&fn_name) {
            // function call
            if let GlobalDeclaration::Function(decl) = decl {
                if ctx.stage == EvalStage::Const && !decl.attributes.contains(&Attribute::Const) {
                    return Err(E::NotConst(decl.ident.to_string()));
                }

                if decl.body.attributes.contains(&ATTR_INTRINSIC) {
                    return call_builtin(ty, args, ctx).map(Into::into);
                }

                if self.arguments.len() != decl.parameters.len() {
                    return Err(E::ParamCount(
                        decl.ident.to_string(),
                        decl.parameters.len(),
                        self.arguments.len(),
                    ));
                }

                let ret_ty = decl
                    .return_type
                    .as_ref()
                    .map(|expr| ty_eval_ty(expr, ctx))
                    .transpose()?;

                let flow = with_scope!(ctx, {
                    let args = args
                        .iter()
                        .zip(&decl.parameters)
                        .map(|(arg, param)| {
                            let param_ty = ty_eval_ty(&param.ty, ctx)?;
                            arg.convert_to(&param_ty)
                                .ok_or_else(|| E::ParamType(param_ty.clone(), arg.ty()))
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .inspect_err(|_| ctx.set_err_decl_ctx(fn_name.clone()))?;

                    for (a, p) in zip(args, &decl.parameters) {
                        if !ctx.scope.add(p.ident.to_string(), a) {
                            return Err(E::DuplicateDecl(p.ident.to_string()));
                        }
                    }

                    // the arguments must be in the same scope as the function body.
                    let flow = compound_exec_no_scope(&decl.body, ctx)
                        .inspect_err(|_| ctx.set_err_decl_ctx(fn_name.clone()))?;

                    Ok(flow)
                })?;

                match (flow, ret_ty) {
                    (flow @ (Flow::Break | Flow::Continue), _) => Err(E::FlowInFunction(flow)),
                    (Flow::Next, Some(ret_ty)) => Err(E::NoReturn(fn_name, ret_ty)),
                    (Flow::Return(Some(inst)), Some(ret_ty)) => inst
                        .convert_to(&ret_ty)
                        .ok_or(E::ReturnType(inst.ty(), fn_name.clone(), ret_ty))
                        .map(Into::into)
                        .inspect_err(|_| ctx.set_err_decl_ctx(fn_name)),
                    (Flow::Return(Some(inst)), None) => {
                        Err(E::UnexpectedReturn(fn_name, inst.ty()))
                    }
                    (Flow::Return(None), Some(ret_ty)) => Err(E::NoReturn(fn_name, ret_ty)),
                    (Flow::Next | Flow::Return(None), None) => Ok(Flow::Return(None)),
                }
            }
            // struct constructor
            else if let GlobalDeclaration::Struct(decl) = decl {
                if args.len() == decl.members.len() {
                    let members = decl
                        .members
                        .iter()
                        .zip(args)
                        .map(|(member, inst)| {
                            let ty = ty_eval_ty(&member.ty, ctx)?;
                            let inst = inst
                                .convert_to(&ty)
                                .ok_or_else(|| E::ParamType(ty, inst.ty()))?;
                            Ok((member.ident.to_string(), inst))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Instance::Struct(StructInstance::new(fn_name, members)).into())
                } else if args.is_empty() {
                    Ok(Instance::Struct(StructInstance::zero_value(&fn_name, ctx)?).into())
                } else {
                    Err(E::ParamCount(fn_name, decl.members.len(), args.len()))
                }
            } else {
                Err(E::NotCallable(fn_name))
            }
        } else if is_constructor_fn(&fn_name) {
            call_builtin(ty, args, ctx).map(Into::into)
        } else {
            Err(E::UnknownFunction(fn_name))
        }
    }
}

impl Exec for ConstAssertStatement {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        with_stage!(ctx, EvalStage::Const, {
            let expr = self.expression.eval_value(ctx)?;
            let cond = match expr {
                Instance::Literal(LiteralInstance::Bool(b)) => Ok(b),
                _ => Err(E::Type(Type::Bool, expr.ty())),
            }?;

            if cond {
                Ok(Flow::Next)
            } else {
                Err(E::ConstAssertFailure(self.expression.clone()))
            }
        })
    }
}

// TODO: implement address space
impl Exec for Declaration {
    fn exec(&self, ctx: &mut Context) -> Result<Flow, E> {
        if ctx.scope.local_contains(&self.ident.name()) {
            return Err(E::DuplicateDecl(self.ident.to_string()));
        }

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

        let init = |ctx: &mut Context, stage: EvalStage| {
            self.initializer
                .as_ref()
                .map(|init| {
                    let inst = with_stage!(ctx, stage, { init.eval_value(ctx) })?;
                    inst.convert_to(&ty)
                        .ok_or_else(|| E::Conversion(inst.ty(), ty.clone()))
                })
                .transpose()
        };

        let inst = match (self.kind, ctx.kind) {
            (DeclarationKind::Const, _) => init(ctx, EvalStage::Const)?
                .ok_or_else(|| E::UninitConst(self.ident.to_string()))?,
            (DeclarationKind::Override, ScopeKind::Function) => return Err(E::OverrideInFn),
            (DeclarationKind::Let, ScopeKind::Function) => {
                init(ctx, ctx.stage)?.ok_or_else(|| E::UninitLet(self.ident.to_string()))?
            }
            (DeclarationKind::Var(space), ScopeKind::Function) => {
                if !matches!(space, Some(AddressSpace::Function) | None) {
                    return Err(E::ForbiddenDecl(self.kind, ctx.kind));
                }
                let inst = init(ctx, ctx.stage)?
                    .map(Ok)
                    .unwrap_or_else(|| Instance::zero_value(&ty, ctx))?;

                RefInstance::new(inst, AddressSpace::Function, AccessMode::ReadWrite).into()
            }
            (DeclarationKind::Override, ScopeKind::Module) => {
                if ctx.stage == EvalStage::Const {
                    Instance::Deferred(ty)
                } else if let Some(inst) = ctx.overridable(&self.ident.name()) {
                    inst.convert_to(&ty)
                        .ok_or_else(|| E::Conversion(inst.ty(), ty))?
                } else if let Some(inst) = init(ctx, EvalStage::Override)? {
                    inst
                } else {
                    return Err(E::UninitOverride(self.ident.to_string()));
                }
            }
            (DeclarationKind::Let, ScopeKind::Module) => return Err(E::LetInMod),
            (DeclarationKind::Var(addr_space), ScopeKind::Module) => {
                if ctx.stage == EvalStage::Const {
                    Instance::Deferred(ty)
                } else {
                    let addr_space = addr_space.unwrap_or(AddressSpace::Handle);

                    match addr_space {
                        AddressSpace::Function => {
                            return Err(E::ForbiddenDecl(self.kind, ctx.kind));
                        }
                        AddressSpace::Private => {
                            // the initializer for a private variable must be a const- or override-expression
                            let inst = if let Some(inst) = init(ctx, EvalStage::Override)? {
                                inst
                            } else {
                                Instance::zero_value(&ty, ctx)?
                            };

                            RefInstance::new(inst, AddressSpace::Private, AccessMode::ReadWrite)
                                .into()
                        }
                        AddressSpace::Uniform => {
                            if self.initializer.is_some() {
                                return Err(E::ForbiddenInitializer(addr_space));
                            }
                            let (group, binding) = self.attr_group_binding(ctx)?;
                            let inst = ctx
                                .resource(group, binding)
                                .ok_or(E::MissingResource(group, binding))?;
                            if inst.ty() != ty {
                                return Err(E::Type(ty, inst.ty()));
                            }
                            if !inst.space.is_uniform() {
                                return Err(E::AddressSpace(addr_space, inst.space));
                            }
                            if inst.access != AccessMode::Read {
                                return Err(E::AccessMode(AccessMode::Read, inst.access));
                            }
                            inst.clone().into()
                        }
                        AddressSpace::Storage(access_mode) => {
                            if self.initializer.is_some() {
                                return Err(E::ForbiddenInitializer(addr_space));
                            }
                            let Some(ty) = &self.ty else {
                                return Err(E::UntypedDecl);
                            };
                            let ty = ty_eval_ty(ty, ctx)?;
                            let (group, binding) = self.attr_group_binding(ctx)?;
                            let inst = ctx
                                .resource(group, binding)
                                .ok_or(E::MissingResource(group, binding))?;
                            if ty != inst.ty() {
                                return Err(E::Type(ty, inst.ty()));
                            }
                            if !inst.space.is_storage() {
                                return Err(E::AddressSpace(addr_space, inst.space));
                            }
                            let access_mode = access_mode.unwrap_or(AccessMode::Read);
                            if inst.access != access_mode {
                                return Err(E::AccessMode(access_mode, inst.access));
                            }
                            inst.clone().into()
                        }
                        AddressSpace::Workgroup => todo!("workgroup address space"),
                        AddressSpace::Handle => todo!("handle address space"),
                    }
                }
            }
        };

        if ctx.scope.add(self.ident.to_string(), inst) {
            Ok(Flow::Next)
        } else {
            Err(E::DuplicateDecl(self.ident.to_string()))
        }
    }
}
