use std::str::FromStr;

use super::{
    ArrayInstance, ArrayTemplate, AtomicInstance, AtomicTemplate, Context, EvalError, Instance,
    LiteralInstance, MatInstance, MatTemplate, PtrInstance, PtrTemplate, RefInstance,
    StructInstance, SyntaxUtil, TextureTemplate, VecInstance, VecTemplate,
};

type E = EvalError;

use derive_more::derive::{IsVariant, Unwrap};
use wgsl_parse::syntax::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum SampledType {
    I32,
    U32,
    F32,
}

impl FromStr for SampledType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i32" => Ok(Self::I32),
            "u32" => Ok(Self::U32),
            "f32" => Ok(Self::F32),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum SamplerType {
    Sampler,
    SamplerComparison,
}

impl FromStr for SamplerType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sampler" => Ok(Self::Sampler),
            "sampler_comparison" => Ok(Self::SamplerComparison),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum TexelFormat {
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    R32Uint,
    R32Sint,
    R32Float,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Bgra8Unorm,
}

impl FromStr for TexelFormat {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rgba8unorm" => Ok(Self::Rgba8Unorm),
            "rgba8snorm" => Ok(Self::Rgba8Snorm),
            "rgba8uint" => Ok(Self::Rgba8Uint),
            "rgba8sint" => Ok(Self::Rgba8Sint),
            "rgba16uint" => Ok(Self::Rgba16Uint),
            "rgba16sint" => Ok(Self::Rgba16Sint),
            "rgba16float" => Ok(Self::Rgba16Float),
            "r32uint" => Ok(Self::R32Uint),
            "r32sint" => Ok(Self::R32Sint),
            "r32float" => Ok(Self::R32Float),
            "rg32uint" => Ok(Self::Rg32Uint),
            "rg32sint" => Ok(Self::Rg32Sint),
            "rg32float" => Ok(Self::Rg32Float),
            "rgba32uint" => Ok(Self::Rgba32Uint),
            "rgba32sint" => Ok(Self::Rgba32Sint),
            "rgba32float" => Ok(Self::Rgba32Float),
            "bgra8unorm" => Ok(Self::Bgra8Unorm),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum TextureType {
    // sampled
    Sampled1D(SampledType),
    Sampled2D(SampledType),
    Sampled2DArray(SampledType),
    Sampled3D(SampledType),
    SampledCube(SampledType),
    SampledCubeArray(SampledType),
    // multisampled
    Multisampled2D(SampledType),
    DepthMultisampled2D,
    // external
    External,
    // storage
    Storage1D(TexelFormat, AccessMode),
    Storage2D(TexelFormat, AccessMode),
    Storage2DArray(TexelFormat, AccessMode),
    Storage3D(TexelFormat, AccessMode),
    // depth
    Depth2D,
    Depth2DArray,
    DepthCube,
    DepthCubeArray,
}

#[derive(Clone, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum Type {
    Bool,
    AbstractInt,
    AbstractFloat,
    I32,
    U32,
    F32,
    F16,
    Struct(String),
    // TODO: swap these two members
    Array(Option<usize>, Box<Type>),
    Vec(u8, Box<Type>),
    Mat(u8, u8, Box<Type>),
    Atomic(Box<Type>),
    Ptr(AddressSpace, Box<Type>),
    Texture(TextureType),
    Sampler(SamplerType),
    Void,
}

impl Type {
    /// reference: <https://www.w3.org/TR/WGSL/#scalar>
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            Type::Bool
                | Type::AbstractInt
                | Type::AbstractFloat
                | Type::I32
                | Type::U32
                | Type::F32
                | Type::F16
        )
    }

    /// reference: <https://www.w3.org/TR/WGSL/#numeric-scalar>
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Type::AbstractInt | Type::AbstractFloat | Type::I32 | Type::U32 | Type::F32 | Type::F16
        )
    }

    /// reference: <https://www.w3.org/TR/WGSL/#integer-scalar>
    pub fn is_integer(&self) -> bool {
        matches!(self, Type::AbstractInt | Type::I32 | Type::U32)
    }

    /// reference: <https://www.w3.org/TR/WGSL/#floating-point-types>
    pub fn is_float(&self) -> bool {
        matches!(self, Type::AbstractFloat | Type::F32 | Type::F16)
    }

    /// reference: <https://www.w3.org/TR/WGSL/#abstract-types>
    pub fn is_abstract(&self) -> bool {
        match self {
            Type::AbstractInt => true,
            Type::AbstractFloat => true,
            Type::Array(_, ty) | Type::Vec(_, ty) | Type::Mat(_, _, ty) => ty.is_abstract(),
            _ => false,
        }
    }

    pub fn is_concrete(&self) -> bool {
        !self.is_abstract()
    }

    /// reference: <https://www.w3.org/TR/WGSL/#storable-types>
    pub fn is_storable(&self) -> bool {
        self.is_concrete()
            && matches!(
                self,
                Type::Bool
                    | Type::I32
                    | Type::U32
                    | Type::F32
                    | Type::F16
                    | Type::Struct(_)
                    | Type::Array(_, _)
                    | Type::Vec(_, _)
                    | Type::Mat(_, _, _)
                    | Type::Atomic(_)
            )
    }
}

pub trait Ty {
    /// get the type of an instance.
    fn ty(&self) -> Type;

    /// get the inner type of an instance (not recursive).
    ///
    /// e.g. the inner type of `array<vec3<u32>>` is `vec3<u32>`.
    fn inner_ty(&self) -> Type {
        self.ty()
    }
}

impl Ty for Type {
    fn ty(&self) -> Type {
        self.clone()
    }

    fn inner_ty(&self) -> Type {
        match self {
            Type::Bool => self.clone(),
            Type::AbstractInt => self.clone(),
            Type::AbstractFloat => self.clone(),
            Type::I32 => self.clone(),
            Type::U32 => self.clone(),
            Type::F32 => self.clone(),
            Type::F16 => self.clone(),
            Type::Struct(_) => self.clone(),
            Type::Array(_, ty) => ty.ty(),
            Type::Vec(_, ty) => ty.ty(),
            Type::Mat(_, _, ty) => ty.inner_ty(),
            Type::Atomic(ty) => ty.ty(),
            Type::Ptr(_, ty) => ty.ty(),
            Type::Texture(_) => self.clone(),
            Type::Sampler(_) => self.clone(),
            Type::Void => self.clone(),
        }
    }
}

impl Ty for Instance {
    fn ty(&self) -> Type {
        match self {
            Instance::Literal(l) => l.ty(),
            Instance::Struct(s) => s.ty(),
            Instance::Array(a) => a.ty(),
            Instance::Vec(v) => v.ty(),
            Instance::Mat(m) => m.ty(),
            Instance::Ptr(p) => p.ty(),
            Instance::Ref(r) => r.ty(),
            Instance::Atomic(a) => a.ty(),
            Instance::Type(t) => t.ty(),
            Instance::Void => Type::Void,
        }
    }
    fn inner_ty(&self) -> Type {
        match self {
            Instance::Literal(l) => l.inner_ty(),
            Instance::Struct(s) => s.inner_ty(),
            Instance::Array(a) => a.inner_ty(),
            Instance::Vec(v) => v.inner_ty(),
            Instance::Mat(m) => m.inner_ty(),
            Instance::Ptr(p) => p.inner_ty(),
            Instance::Ref(r) => r.inner_ty(),
            Instance::Atomic(a) => a.inner_ty(),
            Instance::Type(t) => t.inner_ty(),
            Instance::Void => Type::Void,
        }
    }
}

impl Ty for LiteralInstance {
    fn ty(&self) -> Type {
        match self {
            LiteralInstance::Bool(_) => Type::Bool,
            LiteralInstance::AbstractInt(_) => Type::AbstractInt,
            LiteralInstance::AbstractFloat(_) => Type::AbstractFloat,
            LiteralInstance::I32(_) => Type::I32,
            LiteralInstance::U32(_) => Type::U32,
            LiteralInstance::F32(_) => Type::F32,
            LiteralInstance::F16(_) => Type::F16,
        }
    }
}

impl Ty for LiteralExpression {
    fn ty(&self) -> Type {
        match self {
            LiteralExpression::Bool(_) => Type::Bool,
            LiteralExpression::AbstractInt(_) => Type::AbstractInt,
            LiteralExpression::AbstractFloat(_) => Type::AbstractFloat,
            LiteralExpression::I32(_) => Type::I32,
            LiteralExpression::U32(_) => Type::U32,
            LiteralExpression::F32(_) => Type::F32,
            LiteralExpression::F16(_) => Type::F16,
        }
    }
}

impl Ty for StructInstance {
    fn ty(&self) -> Type {
        Type::Struct(self.name().to_string())
    }
}

impl Ty for ArrayInstance {
    fn ty(&self) -> Type {
        Type::Array(
            (!self.runtime_sized).then_some(self.n()),
            Box::new(self.inner_ty().clone()),
        )
    }
    fn inner_ty(&self) -> Type {
        self.get(0).unwrap().ty()
    }
}

impl Ty for VecInstance {
    fn ty(&self) -> Type {
        Type::Vec(self.n() as u8, Box::new(self.inner_ty()))
    }
    fn inner_ty(&self) -> Type {
        self.get(0).unwrap().ty()
    }
}

impl Ty for MatInstance {
    fn ty(&self) -> Type {
        Type::Mat(self.c() as u8, self.r() as u8, Box::new(self.inner_ty()))
    }
    fn inner_ty(&self) -> Type {
        self.get(0, 0).unwrap().ty()
    }
}

impl Ty for PtrInstance {
    fn ty(&self) -> Type {
        Type::Ptr(self.ptr.space, Box::new(self.ptr.ty.clone()))
    }
}

impl Ty for RefInstance {
    fn ty(&self) -> Type {
        self.ty.clone()
    }
}

impl Ty for AtomicInstance {
    fn ty(&self) -> Type {
        Type::Atomic(self.inner_ty().into())
    }
    fn inner_ty(&self) -> Type {
        self.inner().ty()
    }
}

pub trait EvalTy {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E>;
}

impl<T: Ty> EvalTy for T {
    fn eval_ty(&self, _ctx: &mut Context) -> Result<Type, E> {
        Ok(self.ty())
    }
}

impl EvalTy for TypeExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        let ty = ctx.source.resolve_ty(self);
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
                GlobalDeclaration::Struct(_) => {
                    if ty.template_args.is_some() {
                        return Err(E::UnexpectedTemplate(name.to_string()));
                    } else {
                        let ty = Type::Struct(name.to_string());
                        return Ok(ty);
                    }
                }
                _ => return Err(E::NotType(name.to_string())),
            }
        }

        // for now, only builtin types can be type-generators (have a template)
        if let Some(tplt) = &ty.template_args {
            match name {
                "array" => {
                    let tplt = ArrayTemplate::parse(tplt, ctx)?;
                    Ok(tplt.ty())
                }
                "vec2" | "vec3" | "vec4" => {
                    let tplt = VecTemplate::parse(tplt, ctx)?;
                    let n = name.chars().nth(3).unwrap().to_digit(10).unwrap() as u8;
                    Ok(tplt.ty(n))
                }
                "mat2x2" | "mat2x3" | "mat2x4" | "mat3x2" | "mat3x3" | "mat3x4" | "mat4x2"
                | "mat4x3" | "mat4x4" => {
                    let tplt = MatTemplate::parse(tplt, ctx)?;
                    let c = name.chars().nth(3).unwrap().to_digit(10).unwrap() as u8;
                    let r = name.chars().nth(5).unwrap().to_digit(10).unwrap() as u8;
                    Ok(tplt.ty(c, r))
                }
                "ptr" => {
                    let tplt = PtrTemplate::parse(tplt, ctx)?;
                    Ok(tplt.ty())
                }
                "atomic" => {
                    let tplt = AtomicTemplate::parse(tplt, ctx)?;
                    Ok(tplt.ty())
                }
                name @ ("texture_1d"
                | "texture_2d"
                | "texture_2d_array"
                | "texture_3d"
                | "texture_cube"
                | "texture_cube_array"
                | "texture_multisampled_2d"
                | "texture_depth_multisampled_2d"
                | "texture_storage_1d"
                | "texture_storage_2d"
                | "texture_storage_2d_array"
                | "texture_storage_3d") => {
                    let tplt = TextureTemplate::parse(name, tplt)?;
                    Ok(Type::Texture(tplt.ty()))
                }
                _ => Err(E::UnexpectedTemplate(name.to_string())),
            }
        }
        // builtin types without a template
        else {
            match name {
                "bool" => Ok(Type::Bool),
                "i32" => Ok(Type::I32),
                "u32" => Ok(Type::U32),
                "f32" => Ok(Type::F32),
                "f16" => Ok(Type::F16),
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
}

impl EvalTy for Expression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        match self {
            Expression::Literal(expr) => Ok(expr.ty()),
            Expression::Parenthesized(expr) => expr.expression.eval_ty(ctx),
            Expression::NamedComponent(expr) => match expr.base.eval_ty(ctx)? {
                Type::Struct(name) => {
                    let decl = ctx.source.decl_struct(&name).unwrap();
                    let mem = decl
                        .members
                        .iter()
                        .find(|m| *m.ident.name() == *expr.component.name())
                        .ok_or_else(|| {
                            E::Component(Type::Struct(name), expr.component.to_string())
                        })?;
                    mem.ty.eval_ty(ctx)
                }
                Type::Vec(_, ty) => {
                    // TODO: check valid swizzle
                    let m = expr.component.name().len();
                    if m == 1 {
                        Ok(*ty)
                    } else if m <= 4 {
                        Ok(Type::Vec(m as u8, ty))
                    } else {
                        Err(E::Swizzle(expr.component.to_string()))
                    }
                }
                ty => Err(E::Component(ty, expr.component.to_string())),
            },
            Expression::Indexing(expr) => {
                let index_ty = expr.index.eval_ty(ctx)?;
                if index_ty.is_integer() {
                    match expr.base.eval_ty(ctx)? {
                        Type::Array(_, ty) => Ok(*ty),
                        Type::Vec(_, ty) => Ok(*ty),
                        Type::Mat(c, _, ty) => Ok(Type::Vec(c, ty)),
                        ty => Err(E::NotIndexable(ty)),
                    }
                } else {
                    Err(E::Index(index_ty))
                }
            }
            Expression::Unary(expr) => {
                let ty = expr.operand.eval_ty(ctx)?;
                let inner = ty.inner_ty();
                if ty != inner
                    && !ty.is_vec()
                    && !expr.operator.is_address_of()
                    && !expr.operator.is_indirection()
                {
                    return Err(E::Unary(expr.operator, ty));
                }
                match expr.operator {
                    UnaryOperator::LogicalNegation if inner == Type::Bool => Ok(ty),
                    UnaryOperator::Negation if inner.is_scalar() && !inner.is_u_32() => Ok(ty),
                    UnaryOperator::BitwiseComplement if inner.is_integer() => Ok(ty),
                    UnaryOperator::AddressOf => Ok(Type::Ptr(AddressSpace::Function, Box::new(ty))), // TODO: we don't know the address space
                    UnaryOperator::Indirection if ty.is_ptr() => Ok(*ty.unwrap_ptr().1),
                    _ => Err(E::Unary(expr.operator, ty)),
                }
            }
            Expression::Binary(expr) => {
                let ty1 = expr.left.eval_ty(ctx)?;
                let ty2 = expr.right.eval_ty(ctx)?;
                let inner = ty1.inner_ty();
                if inner != ty2.inner_ty() {
                    return Err(E::Binary(expr.operator, ty1, ty2));
                }
                let is_numeric = (ty1.is_vec() || ty1.is_numeric())
                    && (ty2.is_vec() || ty2.is_numeric())
                    && inner.is_numeric();
                Ok(match expr.operator {
                    BinaryOperator::ShortCircuitOr | BinaryOperator::ShortCircuitAnd
                        if ty1.is_bool() && ty1 == ty2 =>
                    {
                        Type::Bool
                    }
                    BinaryOperator::Addition if is_numeric || ty1.is_mat() && ty1 == ty2 => ty1,
                    BinaryOperator::Subtraction if is_numeric || ty1.is_mat() && ty1 == ty2 => ty1,
                    BinaryOperator::Multiplication => match (ty1, ty2) {
                        (ty1, _) if is_numeric => ty1,
                        (ty1, ty2) if ty1.is_mat() && ty2.is_float() => ty1,
                        (ty1, ty2) if ty1.is_float() && ty2.is_mat() => ty2,
                        (Type::Mat(c, r, _), Type::Vec(n, _)) if n == c => {
                            Type::Vec(r, Box::new(inner))
                        }
                        (Type::Vec(n, _), Type::Mat(c, r, _)) if n == r => {
                            Type::Vec(c, Box::new(inner))
                        }
                        (Type::Mat(k1, r1, _), Type::Mat(c2, k2, _)) if k1 == k2 => {
                            Type::Mat(c2, r1, Box::new(inner))
                        }
                        (ty1, ty2) => return Err(E::Binary(expr.operator, ty1, ty2)),
                    },
                    BinaryOperator::Division | BinaryOperator::Remainder if is_numeric => ty1,
                    BinaryOperator::Equality | BinaryOperator::Inequality if ty1.is_scalar() => {
                        Type::Bool
                    }
                    BinaryOperator::Equality | BinaryOperator::Inequality
                        if ty1.is_vec() && ty1 == ty2 =>
                    {
                        Type::Vec(ty1.unwrap_vec().0, Box::new(Type::Bool))
                    }
                    BinaryOperator::LessThan
                    | BinaryOperator::LessThanEqual
                    | BinaryOperator::GreaterThan
                    | BinaryOperator::GreaterThanEqual
                        if ty1.is_numeric() =>
                    {
                        Type::Bool
                    }
                    BinaryOperator::LessThan
                    | BinaryOperator::LessThanEqual
                    | BinaryOperator::GreaterThan
                    | BinaryOperator::GreaterThanEqual
                        if ty1.is_vec() && ty1 == ty2 =>
                    {
                        Type::Vec(ty1.unwrap_vec().0, Box::new(Type::Bool))
                    }
                    BinaryOperator::BitwiseOr
                    | BinaryOperator::BitwiseAnd
                    | BinaryOperator::BitwiseXor
                    | BinaryOperator::ShiftLeft
                    | BinaryOperator::ShiftRight
                        if ty1.is_integer() || ty1.is_vec() && inner.is_integer() && ty2 == ty1 =>
                    {
                        ty1
                    }
                    _ => return Err(E::Binary(expr.operator, ty1, ty2)),
                })
            }
            Expression::FunctionCall(call) => {
                let decl = ctx.source.decl_function(&call.ty.ident.name()).unwrap();
                decl.return_type
                    .as_ref()
                    .map(|ty| ty.eval_ty(ctx))
                    .unwrap_or(Ok(Type::Void))
            }
            Expression::TypeOrIdentifier(ty) => ty.eval_ty(ctx),
        }
    }
}
