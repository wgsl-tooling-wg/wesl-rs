use std::str::FromStr;

use super::{
    ATTR_INTRINSIC, ArrayInstance, ArrayTemplate, AtomicInstance, AtomicTemplate, Context, Convert,
    EvalError, Exec, Instance, LiteralInstance, MatInstance, MatTemplate, PtrInstance, PtrTemplate,
    RefInstance, ScopeKind, StructInstance, SyntaxUtil, TextureTemplate, VecInstance, VecTemplate,
    builtin_fn_type, check_swizzle, constructor_type, convert_ty, is_constructor_fn,
};

type E = EvalError;

use derive_more::derive::{IsVariant, Unwrap};
use wgsl_parse::{span::Spanned, syntax::*};

#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum SampledType {
    I32,
    U32,
    F32,
}

impl From<SampledType> for Type {
    fn from(value: SampledType) -> Self {
        match value {
            SampledType::I32 => Type::I32,
            SampledType::U32 => Type::U32,
            SampledType::F32 => Type::F32,
        }
    }
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

impl TexelFormat {
    pub fn channel_type(&self) -> SampledType {
        match self {
            TexelFormat::Rgba8Unorm => SampledType::F32,
            TexelFormat::Rgba8Snorm => SampledType::F32,
            TexelFormat::Rgba8Uint => SampledType::U32,
            TexelFormat::Rgba8Sint => SampledType::I32,
            TexelFormat::Rgba16Uint => SampledType::U32,
            TexelFormat::Rgba16Sint => SampledType::I32,
            TexelFormat::Rgba16Float => SampledType::F32,
            TexelFormat::R32Uint => SampledType::U32,
            TexelFormat::R32Sint => SampledType::I32,
            TexelFormat::R32Float => SampledType::F32,
            TexelFormat::Rg32Uint => SampledType::U32,
            TexelFormat::Rg32Sint => SampledType::I32,
            TexelFormat::Rg32Float => SampledType::F32,
            TexelFormat::Rgba32Uint => SampledType::U32,
            TexelFormat::Rgba32Sint => SampledType::I32,
            TexelFormat::Rgba32Float => SampledType::F32,
            TexelFormat::Bgra8Unorm => SampledType::F32,
        }
    }

    pub fn num_channels(&self) -> u32 {
        match self {
            TexelFormat::Rgba8Unorm => 4,
            TexelFormat::Rgba8Snorm => 4,
            TexelFormat::Rgba8Uint => 4,
            TexelFormat::Rgba8Sint => 4,
            TexelFormat::Rgba16Uint => 4,
            TexelFormat::Rgba16Sint => 4,
            TexelFormat::Rgba16Float => 4,
            TexelFormat::R32Uint => 1,
            TexelFormat::R32Sint => 1,
            TexelFormat::R32Float => 1,
            TexelFormat::Rg32Uint => 2,
            TexelFormat::Rg32Sint => 2,
            TexelFormat::Rg32Float => 2,
            TexelFormat::Rgba32Uint => 4,
            TexelFormat::Rgba32Sint => 4,
            TexelFormat::Rgba32Float => 4,
            TexelFormat::Bgra8Unorm => 4,
        }
    }
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

#[derive(Clone, Debug, PartialEq, Eq, IsVariant)]
pub enum TextureDimensions {
    D1,
    D2,
    D3,
}

impl TextureType {
    pub fn dimensions(&self) -> TextureDimensions {
        match self {
            Self::Sampled1D(_) | Self::Storage1D(_, _) => TextureDimensions::D1,
            Self::Sampled2D(_)
            | Self::Sampled2DArray(_)
            | Self::SampledCube(_)
            | Self::SampledCubeArray(_)
            | Self::Multisampled2D(_)
            | Self::Depth2D
            | Self::Depth2DArray
            | Self::DepthCube
            | Self::DepthCubeArray
            | Self::DepthMultisampled2D
            | Self::Storage2D(_, _)
            | Self::Storage2DArray(_, _)
            | Self::External => TextureDimensions::D2,
            Self::Sampled3D(_) | Self::Storage3D(_, _) => TextureDimensions::D3,
        }
    }
    pub fn sampled_type(&self) -> Option<SampledType> {
        match self {
            TextureType::Sampled1D(st) => Some(*st),
            TextureType::Sampled2D(st) => Some(*st),
            TextureType::Sampled2DArray(st) => Some(*st),
            TextureType::Sampled3D(st) => Some(*st),
            TextureType::SampledCube(st) => Some(*st),
            TextureType::SampledCubeArray(st) => Some(*st),
            TextureType::Multisampled2D(_) => None,
            TextureType::DepthMultisampled2D => None,
            TextureType::External => None,
            TextureType::Storage1D(_, _) => None,
            TextureType::Storage2D(_, _) => None,
            TextureType::Storage2DArray(_, _) => None,
            TextureType::Storage3D(_, _) => None,
            TextureType::Depth2D => None,
            TextureType::Depth2DArray => None,
            TextureType::DepthCube => None,
            TextureType::DepthCubeArray => None,
        }
    }
    pub fn channel_type(&self) -> SampledType {
        match self {
            TextureType::Sampled1D(st) => *st,
            TextureType::Sampled2D(st) => *st,
            TextureType::Sampled2DArray(st) => *st,
            TextureType::Sampled3D(st) => *st,
            TextureType::SampledCube(st) => *st,
            TextureType::SampledCubeArray(st) => *st,
            TextureType::Multisampled2D(st) => *st,
            TextureType::DepthMultisampled2D => SampledType::F32,
            TextureType::External => SampledType::F32,
            TextureType::Storage1D(f, _) => f.channel_type(),
            TextureType::Storage2D(f, _) => f.channel_type(),
            TextureType::Storage2DArray(f, _) => f.channel_type(),
            TextureType::Storage3D(f, _) => f.channel_type(),
            TextureType::Depth2D => SampledType::F32,
            TextureType::Depth2DArray => SampledType::F32,
            TextureType::DepthCube => SampledType::F32,
            TextureType::DepthCubeArray => SampledType::F32,
        }
    }
    pub fn is_depth(&self) -> bool {
        matches!(
            self,
            TextureType::Depth2D
                | TextureType::Depth2DArray
                | TextureType::DepthCube
                | TextureType::DepthCubeArray
        )
    }
    pub fn is_storage(&self) -> bool {
        matches!(
            self,
            TextureType::Storage1D(_, _)
                | TextureType::Storage2D(_, _)
                | TextureType::Storage2DArray(_, _)
                | TextureType::Storage3D(_, _)
        )
    }
    pub fn is_sampled(&self) -> bool {
        matches!(
            self,
            TextureType::Sampled1D(_)
                | TextureType::Sampled2D(_)
                | TextureType::Sampled2DArray(_)
                | TextureType::Sampled3D(_)
                | TextureType::SampledCube(_)
                | TextureType::SampledCubeArray(_)
        )
    }
    pub fn is_multisampled(&self) -> bool {
        matches!(
            self,
            TextureType::Multisampled2D(_) | TextureType::DepthMultisampled2D
        )
    }
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
    Array(Box<Type>, Option<usize>),
    Vec(u8, Box<Type>),
    Mat(u8, u8, Box<Type>),
    Atomic(Box<Type>),
    Ptr(AddressSpace, Box<Type>),
    Texture(TextureType),
    Sampler(SamplerType),
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
            Type::Array(ty, _) | Type::Vec(_, ty) | Type::Mat(_, _, ty) => ty.is_abstract(),
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
            Type::Array(ty, _) => ty.ty(),
            Type::Vec(_, ty) => ty.ty(),
            Type::Mat(_, _, ty) => ty.ty(),
            Type::Atomic(ty) => ty.ty(),
            Type::Ptr(_, ty) => ty.ty(),
            Type::Texture(_) => self.clone(),
            Type::Sampler(_) => self.clone(),
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
            Instance::Deferred(t) => t.ty(),
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
            Instance::Deferred(t) => t.inner_ty(),
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
            Box::new(self.inner_ty().clone()),
            (!self.runtime_sized).then_some(self.n()),
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
    /// evaluate the type of an expression, without evaluating the expression itself (static analysis).
    /// we mean a 'real' expression, not a template expression resolving to a type. Refer to
    /// `ty_eval_ty` To evaluate the type of a type-expression.
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E>;
}

impl<T: Ty> EvalTy for T {
    fn eval_ty(&self, _ctx: &mut Context) -> Result<Type, E> {
        Ok(self.ty())
    }
}

impl EvalTy for TypeExpression {
    /// Use only when the `TypeExpression` is an identifier (`Expression::TypeOrIdentifer`),
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
            Type::Struct(name) => {
                let decl = ctx
                    .source
                    .decl_struct(&name)
                    .ok_or_else(|| E::UnknownStruct(name.clone()))?;
                let m = decl
                    .members
                    .iter()
                    .find(|m| *m.ident.name() == *self.component.name())
                    .ok_or_else(|| E::Component(Type::Struct(name), self.component.to_string()))?;
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
            && !self.operator.is_address_of()
            && !self.operator.is_indirection()
        {
            return Err(E::Unary(self.operator, ty));
        }
        match self.operator {
            UnaryOperator::LogicalNegation if inner == Type::Bool => Ok(ty),
            UnaryOperator::Negation if inner.is_scalar() && !inner.is_u_32() => Ok(ty),
            UnaryOperator::BitwiseComplement if inner.is_integer() => Ok(ty),
            UnaryOperator::AddressOf => Ok(Type::Ptr(AddressSpace::Function, Box::new(ty))), // TODO: we don't know the address space
            UnaryOperator::Indirection if ty.is_ptr() => Ok(*ty.unwrap_ptr().1),
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

impl EvalTy for FunctionCallExpression {
    fn eval_ty(&self, ctx: &mut Context) -> Result<Type, E> {
        let ty = ctx.source.resolve_ty(&self.ty);
        if let Some(decl) = ctx.source.decl(&ty.ident.name()) {
            match decl {
                GlobalDeclaration::Struct(decl) => {
                    // TODO: check member types
                    Ok(Type::Struct(decl.ident.to_string()))
                }
                GlobalDeclaration::Function(decl) => {
                    if decl.body.attributes.contains(&ATTR_INTRINSIC) {
                        let args = self
                            .arguments
                            .iter()
                            .map(|arg| arg.eval_ty(ctx))
                            .collect::<Result<Vec<_>, _>>()?;
                        builtin_fn_type(&self.ty, &args, ctx)?
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
            let args = self
                .arguments
                .iter()
                .map(|arg| arg.eval_ty(ctx))
                .collect::<Result<Vec<_>, _>>()?;
            constructor_type(ty, &args, ctx)
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
