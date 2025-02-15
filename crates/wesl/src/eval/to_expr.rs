use super::{
    builtin_ident, ArrayInstance, BuiltinIdent, LiteralInstance, MatInstance, StructInstance,
    SyntaxUtil, TextureType, Ty, Type, VecInstance,
};
use crate::eval::{Context, EvalError, Instance};
use wgsl_parse::{span::Spanned, syntax::*};

type E = EvalError;

/// Convert and instance to an Expression.
pub trait ToExpr {
    fn to_expr(&self, ctx: &Context) -> Result<Expression, E>;
}

impl ToExpr for Instance {
    fn to_expr(&self, ctx: &Context) -> Result<Expression, E> {
        match self {
            Instance::Literal(inst) => inst.to_expr(ctx),
            Instance::Struct(inst) => inst.to_expr(ctx),
            Instance::Array(inst) => inst.to_expr(ctx),
            Instance::Vec(inst) => inst.to_expr(ctx),
            Instance::Mat(inst) => inst.to_expr(ctx),
            Instance::Deferred(_)
            | Instance::Atomic(_)
            | Instance::Ptr(_)
            | Instance::Ref(_)
            | Instance::Void => {
                // Err(E::NotConstructible(self.ty()))
                todo!()
            }
        }
    }
}

impl ToExpr for LiteralInstance {
    fn to_expr(&self, _ctx: &Context) -> Result<Expression, E> {
        Ok(match self {
            LiteralInstance::Bool(lit) => LiteralExpression::Bool(*lit),
            LiteralInstance::AbstractInt(lit) => LiteralExpression::AbstractInt(*lit),
            LiteralInstance::AbstractFloat(lit) => LiteralExpression::AbstractFloat(*lit),
            LiteralInstance::I32(lit) => LiteralExpression::I32(*lit),
            LiteralInstance::U32(lit) => LiteralExpression::U32(*lit),
            LiteralInstance::F32(lit) => LiteralExpression::F32(*lit),
            LiteralInstance::F16(lit) => LiteralExpression::F16(lit.to_f32()),
        }
        .into())
    }
}

impl ToExpr for StructInstance {
    fn to_expr(&self, ctx: &Context) -> Result<Expression, E> {
        let decl = ctx
            .source
            .decl_struct(self.name())
            .expect("struct declaration not found");
        Ok(Expression::FunctionCall(FunctionCall {
            ty: TypeExpression::new(Ident::new(self.name().to_string())),
            arguments: decl
                .members
                .iter()
                .map(|m| {
                    self.member(&*m.ident.name())
                        .expect("struct member not found")
                        .to_expr(ctx)
                        .map(Spanned::from)
                })
                .collect::<Result<Vec<_>, _>>()?,
        }))
    }
}

impl ToExpr for ArrayInstance {
    fn to_expr(&self, ctx: &Context) -> Result<Expression, E> {
        Ok(Expression::FunctionCall(FunctionCall {
            ty: TypeExpression::new(builtin_ident("array").unwrap().clone()),
            arguments: self
                .iter()
                .map(|c| c.to_expr(ctx).map(Spanned::from))
                .collect::<Result<Vec<_>, _>>()?,
        }))
    }
}

impl ToExpr for VecInstance {
    fn to_expr(&self, ctx: &Context) -> Result<Expression, E> {
        Ok(Expression::FunctionCall(FunctionCall {
            ty: TypeExpression::new(self.ty().builtin_ident().unwrap().clone()),
            arguments: self
                .iter()
                .map(|c| c.to_expr(ctx).map(Spanned::from))
                .collect::<Result<Vec<_>, _>>()?,
        }))
    }
}

impl ToExpr for MatInstance {
    fn to_expr(&self, ctx: &Context) -> Result<Expression, E> {
        Ok(Expression::FunctionCall(FunctionCall {
            ty: TypeExpression::new(self.ty().builtin_ident().unwrap().clone()),
            arguments: self
                .iter() // could also use iter_cols here to output matCxR(VecR(), ...)
                .map(|c| c.to_expr(ctx).map(Spanned::from))
                .collect::<Result<Vec<_>, _>>()?,
        }))
    }
}

impl ToExpr for Type {
    fn to_expr(&self, ctx: &Context) -> Result<Expression, E> {
        let ident = self.builtin_ident().cloned();
        match self {
            Type::Bool => Ok(TypeExpression::new(ident.unwrap())),
            Type::AbstractInt => Err(E::NotConstructible(Type::AbstractInt)),
            Type::AbstractFloat => Err(E::NotConstructible(Type::AbstractFloat)),
            Type::I32 => Ok(TypeExpression::new(ident.unwrap())),
            Type::U32 => Ok(TypeExpression::new(ident.unwrap())),
            Type::F32 => Ok(TypeExpression::new(ident.unwrap())),
            Type::F16 => Ok(TypeExpression::new(ident.unwrap())),
            Type::Struct(s) => Ok(TypeExpression::new(Ident::new(s.clone()))),
            Type::Array(Some(n), inner_ty) => {
                let mut ty = TypeExpression::new(ident.unwrap());
                ty.template_args = Some(vec![
                    TemplateArg {
                        expression: inner_ty.to_expr(ctx)?.into(),
                    },
                    TemplateArg {
                        expression: Expression::Literal(LiteralExpression::AbstractInt(*n as i64))
                            .into(),
                    },
                ]);
                Ok(ty)
            }
            Type::Array(None, inner_ty) => {
                let mut ty = TypeExpression::new(ident.unwrap());
                ty.template_args = Some(vec![TemplateArg {
                    expression: inner_ty.to_expr(ctx)?.into(),
                }]);
                Ok(ty)
            }
            Type::Vec(_, inner_ty) => {
                let mut ty = TypeExpression::new(ident.unwrap());
                ty.template_args = Some(vec![TemplateArg {
                    expression: inner_ty.to_expr(ctx)?.into(),
                }]);
                Ok(ty)
            }
            Type::Mat(_, _, inner_ty) => {
                let mut ty = TypeExpression::new(ident.unwrap());
                ty.template_args = Some(vec![TemplateArg {
                    expression: inner_ty.to_expr(ctx)?.into(),
                }]);
                Ok(ty)
            }
            Type::Atomic(inner_ty) => {
                let mut ty = TypeExpression::new(ident.unwrap());
                ty.template_args = Some(vec![TemplateArg {
                    expression: inner_ty.to_expr(ctx)?.into(),
                }]);
                Ok(ty)
            }
            Type::Ptr(space, inner_ty) => {
                let mut ty = TypeExpression::new(ident.unwrap());
                ty.template_args = Some(vec![
                    TemplateArg {
                        expression: space.to_expr(ctx)?.into(),
                    },
                    TemplateArg {
                        expression: inner_ty.to_expr(ctx)?.into(),
                    },
                ]);
                Ok(ty)
            }
            Type::Texture(tex) => {
                let mut ty = TypeExpression::new(ident.unwrap());
                ty.template_args = match tex {
                    TextureType::Sampled1D(sampled)
                    | TextureType::Sampled2D(sampled)
                    | TextureType::Sampled2DArray(sampled)
                    | TextureType::Sampled3D(sampled)
                    | TextureType::SampledCube(sampled)
                    | TextureType::SampledCubeArray(sampled)
                    | TextureType::Multisampled2D(sampled) => Some(vec![TemplateArg {
                        expression: Expression::TypeOrIdentifier(TypeExpression::new(
                            builtin_ident(&sampled.to_string()).unwrap().clone(),
                        ))
                        .into(),
                    }]),
                    TextureType::DepthMultisampled2D => None,
                    TextureType::External => None,
                    TextureType::Storage1D(texel, access)
                    | TextureType::Storage2D(texel, access)
                    | TextureType::Storage2DArray(texel, access)
                    | TextureType::Storage3D(texel, access) => Some(vec![
                        TemplateArg {
                            expression: Expression::TypeOrIdentifier(TypeExpression::new(
                                builtin_ident(&texel.to_string()).unwrap().clone(),
                            ))
                            .into(),
                        },
                        TemplateArg {
                            expression: Expression::TypeOrIdentifier(TypeExpression::new(
                                builtin_ident(&access.to_string()).unwrap().clone(),
                            ))
                            .into(),
                        },
                    ]),
                    TextureType::Depth2D => None,
                    TextureType::Depth2DArray => None,
                    TextureType::DepthCube => None,
                    TextureType::DepthCubeArray => None,
                };
                Ok(ty)
            }
            Type::Sampler(_) => Ok(TypeExpression::new(ident.unwrap())),
            Type::Void => Err(E::NotConstructible(Type::Void)),
        }
        .map(Into::into)
    }
}

impl ToExpr for AddressSpace {
    fn to_expr(&self, _ctx: &Context) -> Result<Expression, E> {
        Ok(Expression::TypeOrIdentifier(TypeExpression::from(
            Ident::new(self.to_string()),
        )))
    }
}
