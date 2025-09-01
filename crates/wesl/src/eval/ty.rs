use std::str::FromStr;

use crate::Eval;

use super::{
    ATTR_INTRINSIC, ArrayTemplate, AtomicTemplate, Context, Convert, EvalAttrs, EvalError, Exec,
    MatTemplate, PtrTemplate, SamplerType, ScopeKind, StructType, SyntaxUtil, TextureTemplate,
    TextureType, Ty, Type, VecTemplate, builtin_fn_type, check_swizzle, constructor_type,
    convert_ty, is_constructor_fn, with_stage,
};

type E = EvalError;

use wgsl_parse::{Decorated, span::Spanned, syntax::*};
use wgsl_types::{ShaderStage, enums::Enumerant, tplt::TpltParam, ty::StructMemberType};

pub fn eval_tplt_arg(tplt: &TemplateArg, ctx: &mut Context) -> Result<TpltParam, E> {
    with_stage!(ctx, ShaderStage::Const, {
        match tplt.expression.node() {
            Expression::TypeOrIdentifier(ty) if ty.template_args.is_some() => {
                ty_eval_ty(ty, ctx).map(TpltParam::Type)
            }
            Expression::TypeOrIdentifier(ty) => {
                if let Some(inst) = ctx.scope.get(&ty.ident.name()) {
                    Ok(TpltParam::Instance(inst.clone()))
                } else {
                    if ctx.kind == ScopeKind::Module {
                        // because of module-scope hoisting, declarations may be executed out-of-order.
                        if let Some(decl) = ctx.source.decl(&ty.ident.name()) {
                            decl.exec(ctx)?;
                            if let Some(inst) = ctx.scope.get(&ty.ident.name()) {
                                return Ok(TpltParam::Instance(inst.clone()));
                            }
                        }
                    }

                    ty_eval_ty(ty, ctx).map(TpltParam::Type).or_else(|e| {
                        Enumerant::from_str(&ty.ident.name())
                            .map(TpltParam::Enumerant)
                            .map_err(|()| e)
                    })
                }
            }
            e => e.eval_value(ctx).map(TpltParam::Instance),
        }
    })
}

pub trait EvalTy {
    /// Evaluate the type of an expression, without evaluating the expression itself (static analysis).
    /// We mean a 'real' expression, not a template expression resolving to a type. Refer to
    /// `ty_eval_ty` To evaluate the type of a type-expression.
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E>;
}

// impl<T: Ty> EvalTy for T {
//     fn eval_ty(&self, _ctx: &mut Context) -> Result<Type, E> {
//         Ok(self.ty())
//     }
// }

impl EvalTy for LiteralExpression {
    fn eval_ty(&self, _ctx: &mut Context) -> Result<Type, E> {
        match self {
            LiteralExpression::Bool(_) => Ok(Type::Bool),
            LiteralExpression::AbstractInt(_) => Ok(Type::AbstractInt),
            LiteralExpression::AbstractFloat(_) => Ok(Type::AbstractFloat),
            LiteralExpression::I32(_) => Ok(Type::I32),
            LiteralExpression::U32(_) => Ok(Type::U32),
            LiteralExpression::F32(_) => Ok(Type::F32),
            LiteralExpression::F16(_) => Ok(Type::F16),
            #[cfg(feature = "naga_ext")]
            LiteralExpression::I64(_) => Ok(Type::I64),
            #[cfg(feature = "naga_ext")]
            LiteralExpression::U64(_) => Ok(Type::U64),
            #[cfg(feature = "naga_ext")]
            LiteralExpression::F64(_) => Ok(Type::F64),
        }
    }
}

impl EvalTy for TypeExpression {
    /// Use only when the `TypeExpression` is an identifier (`Expression::TypeOrIdentifier`),
    /// NOT when it is a type-expression. For that, see [`ty_eval_ty`].
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        if self.template_args.is_some() {
            Err(E::UnexpectedTemplate(self.ident.to_string()))
        } else if let Some(inst) = ctx.scope.get(&self.ident.name()) {
            Ok(inst.ty())
        } else {
            if ctx.kind == ScopeKind::Module {
                // because of module-scope hoisting, declarations may be executed out-of-order.
                if let Some(decl) = ctx.source.decl(&self.ident.name()) {
                    decl.exec(ctx)?;
                    return if let Some(inst) = ctx.scope.get(&self.ident.name()) {
                        Ok(inst.ty())
                    } else {
                        Err(E::UnknownDecl(self.ident.to_string()))
                    };
                }
            }
            Err(E::UnknownDecl(self.ident.to_string()))
        }
    }
}

/// Evaluate the type of a type-expression. A type-expression only exists in templates, e.g.
/// `array<f32>`. Use [`TypeExpression::eval_ty`] if you want the type of an identifier reference.
pub fn ty_eval_ty(expr: &TypeExpression, ctx: &mut Context) -> Result<Type, E> {
    let ty = ctx.source.resolve_ty(expr);
    let name = ty.ident.name();
    let name = name.as_str();

    // structs and aliases are the only user-defined types. We resolved
    // aliases already. any user-defined declaration can shadow parent-scope
    // declarations and builtin declarations. even if that declaration is
    // not a type.
    if ctx.scope.contains(name) {
        return Err(E::NotType(name.to_string()));
    }
    // same as above
    if let Some(decl) = ctx.source.decl(name) {
        match decl {
            GlobalDeclaration::Struct(s) => {
                if ty.template_args.is_some() {
                    return Err(E::UnexpectedTemplate(name.to_string()));
                } else {
                    return s.eval_ty(ctx);
                }
            }
            _ => return Err(E::NotType(name.to_string())),
        }
    }

    // for now, only builtin types can be type-generators (have a template)
    if let Some(tplt) = &ty.template_args {
        let tplt = tplt
            .iter()
            .map(|arg| eval_tplt_arg(arg, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        match name {
            "array" => {
                let tplt = ArrayTemplate::parse(&tplt)?;
                Ok(tplt.ty())
            }
            #[cfg(feature = "naga_ext")]
            "binding_array" => {
                let tplt = super::BindingArrayTemplate::parse(&tplt)?;
                Ok(tplt.ty())
            }
            "vec2" | "vec3" | "vec4" => {
                let tplt = VecTemplate::parse(&tplt)?;
                let n = name.chars().nth(3).unwrap().to_digit(10).unwrap() as u8;
                Ok(tplt.ty(n))
            }
            "mat2x2" | "mat2x3" | "mat2x4" | "mat3x2" | "mat3x3" | "mat3x4" | "mat4x2"
            | "mat4x3" | "mat4x4" => {
                let tplt = MatTemplate::parse(&tplt)?;
                let c = name.chars().nth(3).unwrap().to_digit(10).unwrap() as u8;
                let r = name.chars().nth(5).unwrap().to_digit(10).unwrap() as u8;
                Ok(tplt.ty(c, r))
            }
            "ptr" => {
                let tplt = PtrTemplate::parse(&tplt)?;
                Ok(tplt.ty())
            }
            "atomic" => {
                let tplt = AtomicTemplate::parse(&tplt)?;
                Ok(tplt.ty())
            }
            name @ ("texture_1d"
            | "texture_2d"
            | "texture_2d_array"
            | "texture_3d"
            | "texture_cube"
            | "texture_cube_array"
            | "texture_multisampled_2d"
            | "texture_storage_1d"
            | "texture_storage_2d"
            | "texture_storage_2d_array"
            | "texture_storage_3d") => {
                let tplt = TextureTemplate::parse(name, &tplt)?;
                Ok(Type::Texture(tplt.ty()))
            }
            _ => Err(E::UnexpectedTemplate(name.to_string())),
        }
    }
    // builtin types without a template
    else {
        match name {
            "bool" => Ok(Type::Bool),
            "__AbstractInt" => Ok(Type::AbstractInt),
            "__AbstractFloat" => Ok(Type::AbstractFloat),
            "i32" => Ok(Type::I32),
            "u32" => Ok(Type::U32),
            "f32" => Ok(Type::F32),
            "f16" => Ok(Type::F16),
            "texture_depth_multisampled_2d" => Ok(Type::Texture(TextureType::DepthMultisampled2D)),
            "texture_external" => Ok(Type::Texture(TextureType::External)),
            "texture_depth_2d" => Ok(Type::Texture(TextureType::Depth2D)),
            "texture_depth_2d_array" => Ok(Type::Texture(TextureType::Depth2DArray)),
            "texture_depth_cube" => Ok(Type::Texture(TextureType::DepthCube)),
            "texture_depth_cube_array" => Ok(Type::Texture(TextureType::DepthCubeArray)),
            "sampler" => Ok(Type::Sampler(SamplerType::Sampler)),
            "sampler_comparison" => Ok(Type::Sampler(SamplerType::SamplerComparison)),
            _ => Err(E::UnknownType(name.to_string())),
        }
    }
}

impl<T: EvalTy> EvalTy for Spanned<T> {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        self.node().eval_ty(ctx).inspect_err(|_| {
            ctx.set_err_span_ctx(self.span());
        })
    }
}

impl EvalTy for ParenthesizedExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        self.expression.eval_ty(ctx)
    }
}

impl EvalTy for NamedComponentExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        match self.base.eval_ty(ctx)? {
            Type::Struct(s) => {
                let decl = ctx
                    .source
                    .decl_struct(&s.name)
                    .ok_or_else(|| E::UnknownStruct(s.name.clone()))?;
                let m = decl
                    .members
                    .iter()
                    .find(|m| *m.ident.name() == *self.component.name())
                    .ok_or_else(|| E::Component(Type::Struct(s), self.component.to_string()))?;
                ty_eval_ty(&m.ty, ctx)
            }
            Type::Vec(_, ty) => {
                let m = self.component.name().len();
                if !check_swizzle(&self.component.name()) {
                    Err(E::Swizzle(self.component.to_string()))
                } else if m == 1 {
                    Ok(*ty)
                } else {
                    Ok(Type::Vec(m as u8, ty))
                }
            }
            ty => Err(E::Component(ty, self.component.to_string())),
        }
    }
}

impl EvalTy for IndexingExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        let index_ty = self.index.eval_ty(ctx)?;
        if index_ty.is_integer() {
            match self.base.eval_ty(ctx)? {
                Type::Array(ty, _) => Ok(*ty),
                #[cfg(feature = "naga_ext")]
                Type::BindingArray(ty, _) => Ok(*ty),
                Type::Vec(_, ty) => Ok(*ty),
                Type::Mat(_, r, ty) => Ok(Type::Vec(r, ty)),
                ty => Err(E::NotIndexable(ty)),
            }
        } else {
            Err(E::Index(index_ty))
        }
    }
}

impl EvalTy for UnaryExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        let ty = self.operand.eval_ty(ctx)?;
        let inner = ty.inner_ty();
        if ty != inner
            && !ty.is_vec()
            && self.operator != UnaryOperator::AddressOf
            && self.operator != UnaryOperator::Indirection
        {
            return Err(E::Unary(self.operator, ty));
        }
        match self.operator {
            UnaryOperator::LogicalNegation if inner == Type::Bool => Ok(ty),
            UnaryOperator::Negation if inner.is_scalar() && !inner.is_u32() => Ok(ty),
            UnaryOperator::BitwiseComplement if inner.is_integer() => Ok(ty),
            UnaryOperator::AddressOf => Ok(Type::Ptr(
                AddressSpace::Function,
                Box::new(ty),
                AccessMode::ReadWrite,
            )), // TODO: we don't know the address space and access mode
            UnaryOperator::Indirection if ty.is_ptr() => Ok(inner),
            _ => Err(E::Unary(self.operator, ty)),
        }
    }
}

impl EvalTy for BinaryExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        type BinOp = BinaryOperator;
        let ty1 = self.left.eval_ty(ctx)?;
        let ty2 = self.right.eval_ty(ctx)?;

        // ty1 and ty2 must always have the same inner type, except for << and >> operators.
        let (inner, ty1, ty2) = if matches!(self.operator, BinOp::ShiftLeft | BinOp::ShiftRight) {
            (ty1.inner_ty(), ty1, ty2)
        } else {
            let inner = convert_ty(&ty1.inner_ty(), &ty2.inner_ty())
                .ok_or_else(|| E::Binary(self.operator, ty1.clone(), ty2.clone()))?
                .clone();
            let ty1 = ty1.convert_inner_to(&inner).unwrap();
            let ty2 = ty2.convert_inner_to(&inner).unwrap();
            (inner, ty1, ty2)
        };

        let is_num = (ty1.is_vec() || ty1.is_numeric())
            && (ty2.is_vec() || ty2.is_numeric())
            && inner.is_numeric();

        Ok(match self.operator {
            BinOp::ShortCircuitOr | BinOp::ShortCircuitAnd
                if (ty1.is_bool() || ty1.is_vec() && inner.is_bool()) && ty1 == ty2 =>
            {
                Type::Bool
            }
            BinOp::Addition
            | BinOp::Subtraction
            | BinOp::Multiplication
            | BinOp::Division
            | BinOp::Remainder
                if is_num && ty1 == ty2 =>
            {
                ty1
            }
            BinOp::Addition
            | BinOp::Subtraction
            | BinOp::Multiplication
            | BinOp::Division
            | BinOp::Remainder
                if is_num && ty1.is_vec() =>
            {
                ty1
            }
            BinOp::Addition
            | BinOp::Subtraction
            | BinOp::Multiplication
            | BinOp::Division
            | BinOp::Remainder
                if is_num && ty2.is_vec() =>
            {
                ty2
            }
            BinOp::Addition | BinOp::Subtraction if ty1.is_mat() && ty1 == ty2 => ty1,
            BinOp::Multiplication if ty1.is_mat() && ty2.is_float() => ty1,
            BinOp::Multiplication if ty1.is_float() && ty2.is_mat() => ty2,
            BinOp::Multiplication => match (ty1, ty2) {
                (Type::Mat(c, r, _), Type::Vec(n, _)) if n == c => Type::Vec(r, Box::new(inner)),
                (Type::Vec(n, _), Type::Mat(c, r, _)) if n == r => Type::Vec(c, Box::new(inner)),
                (Type::Mat(k1, r1, _), Type::Mat(c2, k2, _)) if k1 == k2 => {
                    Type::Mat(c2, r1, Box::new(inner))
                }
                (ty1, ty2) => return Err(E::Binary(self.operator, ty1, ty2)),
            },
            BinOp::Equality | BinOp::Inequality if ty1.is_scalar() => Type::Bool,
            BinOp::Equality | BinOp::Inequality if ty1.is_vec() && ty1 == ty2 => {
                Type::Vec(ty1.unwrap_vec().0, Box::new(Type::Bool))
            }
            BinOp::LessThan
            | BinOp::LessThanEqual
            | BinOp::GreaterThan
            | BinOp::GreaterThanEqual
                if ty1.is_numeric() =>
            {
                Type::Bool
            }
            BinOp::LessThan
            | BinOp::LessThanEqual
            | BinOp::GreaterThan
            | BinOp::GreaterThanEqual
                if ty1.is_vec() && ty1 == ty2 =>
            {
                Type::Vec(ty1.unwrap_vec().0, Box::new(Type::Bool))
            }
            BinOp::BitwiseOr | BinOp::BitwiseAnd
                if ty1.is_bool() || ty1.is_vec() && inner.is_bool() && ty1 == ty2 =>
            {
                ty1
            }
            BinOp::BitwiseOr | BinOp::BitwiseAnd | BinOp::BitwiseXor
                if ty1.is_integer() || ty1.is_vec() && inner.is_integer() && ty1 == ty2 =>
            {
                ty1
            }
            BinOp::ShiftLeft | BinOp::ShiftRight
                if ty1.is_integer() && ty2.is_convertible_to(&Type::U32)
                    || ty1.is_vec()
                        && ty2.is_vec()
                        && ty1.inner_ty().is_integer()
                        && ty2.inner_ty().is_convertible_to(&Type::U32) =>
            {
                ty1
            }
            _ => return Err(E::Binary(self.operator, ty1, ty2)),
        })
    }
}

impl EvalTy for Struct {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        let members = self
            .members
            .iter()
            .map(|m| {
                Ok(StructMemberType {
                    name: m.ident.to_string(),
                    ty: ty_eval_ty(&m.ty, ctx)?,
                    size: m.attr_size(ctx)?,
                    align: m.attr_align(ctx)?,
                })
            })
            .collect::<Result<_, E>>()?;

        Ok(Type::Struct(Box::new(StructType {
            name: self.ident.to_string(),
            members,
        })))
    }
}

impl EvalTy for FunctionCallExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        let ty = ctx.source.resolve_ty(&self.ty);
        let name = ty.ident.name();
        let tplt = ty
            .template_args
            .as_ref()
            .map(|tplt| {
                tplt.iter()
                    .map(|arg| eval_tplt_arg(arg, ctx))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let args = self
            .arguments
            .iter()
            .map(|arg| arg.eval_ty(ctx))
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(decl) = ctx.source.decl(&ty.ident.name()) {
            match decl {
                GlobalDeclaration::Struct(decl) => decl.eval_ty(ctx),
                GlobalDeclaration::Function(decl) => {
                    if decl.body.contains_attribute(&ATTR_INTRINSIC) {
                        builtin_fn_type(&name, tplt.as_deref(), &args)?
                            .ok_or_else(|| E::Void(decl.ident.to_string()))
                    } else {
                        // TODO: check argument types
                        let ty = decl
                            .return_type
                            .as_ref()
                            .ok_or_else(|| E::Void(decl.ident.to_string()))?;
                        ty_eval_ty(ty, ctx)
                    }
                }
                _ => Err(E::NotCallable(ty.to_string())),
            }
        } else if is_constructor_fn(&ty.ident.name()) {
            let res_ty = constructor_type(&name, tplt.as_deref(), &args)?;
            Ok(res_ty)
        } else {
            Err(E::UnknownFunction(ty.ident.to_string()))
        }
    }
}

impl EvalTy for Expression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        match self {
            Expression::Literal(expr) => expr.eval_ty(ctx),
            Expression::Parenthesized(expr) => expr.eval_ty(ctx),
            Expression::NamedComponent(expr) => expr.eval_ty(ctx),
            Expression::Indexing(expr) => expr.eval_ty(ctx),
            Expression::Unary(expr) => expr.eval_ty(ctx),
            Expression::Binary(expr) => expr.eval_ty(ctx),
            Expression::FunctionCall(expr) => expr.eval_ty(ctx),
            Expression::TypeOrIdentifier(expr) => expr.eval_ty(ctx),
        }
    }
}
