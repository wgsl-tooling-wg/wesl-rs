use std::sync::LazyLock;

use half::prelude::*;
use num_traits::{FromPrimitive, One, ToBytes, ToPrimitive, Zero, real::Real};

use itertools::{Itertools, chain, izip};
use wesl_macros::{quote_expression, quote_module};
use wgsl_parse::syntax::*;

use crate::{
    builtin::builtin_ident,
    eval::{Context, Eval, convert_ty},
};

use super::{
    ArrayInstance, EvalError, EvalStage, Instance, LiteralInstance, MatInstance, RefInstance,
    SampledType, SamplerType, StructInstance, SyntaxUtil, TexelFormat, TextureType, Ty, Type,
    VecInstance,
    conv::{Convert, convert_all},
    convert, convert_all_inner_to, convert_all_to, convert_all_ty,
    ops::Compwise,
    ty_eval_ty,
};

type E = EvalError;

// TODO: when we have the wgsl! macro, we can refactor the consts.

pub static EXPR_TRUE: Expression = quote_expression!(true);
pub static EXPR_FALSE: Expression = quote_expression!(false);

pub trait BuiltinIdent {
    fn builtin_ident(&self) -> Option<&'static Ident>;
}

impl BuiltinIdent for Type {
    fn builtin_ident(&self) -> Option<&'static Ident> {
        match self {
            Type::Bool => builtin_ident("bool"),
            Type::AbstractInt => builtin_ident("__AbstractInt"),
            Type::AbstractFloat => builtin_ident("__AbstractFloat"),
            Type::I32 => builtin_ident("i32"),
            Type::U32 => builtin_ident("u32"),
            Type::F32 => builtin_ident("f32"),
            Type::F16 => builtin_ident("f16"),
            #[cfg(feature = "naga_ext")]
            Type::I64 => builtin_ident("i64"),
            #[cfg(feature = "naga_ext")]
            Type::U64 => builtin_ident("u64"),
            #[cfg(feature = "naga_ext")]
            Type::F64 => builtin_ident("f64"),
            Type::Struct(_) => None,
            Type::Array(_, _) => builtin_ident("array"),
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(_, _) => builtin_ident("binding_array"),
            Type::Vec(n, _) => match n {
                2 => builtin_ident("vec2"),
                3 => builtin_ident("vec3"),
                4 => builtin_ident("vec4"),
                _ => unreachable!("vec must be 2 3 or 4 components"),
            },
            Type::Mat(c, r, _) => match (c, r) {
                (2, 2) => builtin_ident("mat2x2"),
                (2, 3) => builtin_ident("mat2x3"),
                (2, 4) => builtin_ident("mat2x4"),
                (3, 2) => builtin_ident("mat3x2"),
                (3, 3) => builtin_ident("mat3x3"),
                (3, 4) => builtin_ident("mat3x4"),
                (4, 2) => builtin_ident("mat4x2"),
                (4, 3) => builtin_ident("mat4x3"),
                (4, 4) => builtin_ident("mat4x4"),
                _ => unreachable!("mat must be 2 3 or 4 components"),
            },
            Type::Atomic(_) => builtin_ident("atomic"),
            Type::Ptr(_, _) => builtin_ident("ptr"),
            Type::Texture(texture_type) => texture_type.builtin_ident(),
            Type::Sampler(sampler_type) => sampler_type.builtin_ident(),
        }
    }
}

impl BuiltinIdent for TextureType {
    fn builtin_ident(&self) -> Option<&'static Ident> {
        builtin_ident(match self {
            Self::Sampled1D(_) => "texture_1d",
            Self::Sampled2D(_) => "texture_2d",
            Self::Sampled2DArray(_) => "texture_2d_array",
            Self::Sampled3D(_) => "texture_3d",
            Self::SampledCube(_) => "texture_cube",
            Self::SampledCubeArray(_) => "texture_cube_array",
            Self::Multisampled2D(_) => "texture_multisampled_2d",
            Self::DepthMultisampled2D => "texture_depth_multisampled_2d",
            Self::External => "texture_external",
            Self::Storage1D(_, _) => "texture_storage_1d",
            Self::Storage2D(_, _) => "texture_storage_2d",
            Self::Storage2DArray(_, _) => "texture_storage_2d_array",
            Self::Storage3D(_, _) => "texture_storage_3d",
            Self::Depth2D => "texture_depth_2d",
            Self::Depth2DArray => "texture_depth_2d_array",
            Self::DepthCube => "texture_depth_cube",
            Self::DepthCubeArray => "texture_depth_cube_array",
        })
    }
}

impl BuiltinIdent for SamplerType {
    fn builtin_ident(&self) -> Option<&'static Ident> {
        match self {
            SamplerType::Sampler => builtin_ident("sampler"),
            SamplerType::SamplerComparison => builtin_ident("sampler_comparison"),
        }
    }
}

impl BuiltinIdent for SampledType {
    fn builtin_ident(&self) -> Option<&'static Ident> {
        match self {
            Self::I32 => builtin_ident("i32"),
            Self::U32 => builtin_ident("u32"),
            Self::F32 => builtin_ident("f32"),
        }
    }
}

impl BuiltinIdent for AddressSpace {
    fn builtin_ident(&self) -> Option<&'static Ident> {
        match self {
            Self::Function => builtin_ident("function"),
            Self::Private => builtin_ident("private"),
            Self::Workgroup => builtin_ident("workgroup"),
            Self::Uniform => builtin_ident("uniform"),
            Self::Storage(_) => builtin_ident("storage"),
            Self::Handle => None,
            #[cfg(feature = "naga_ext")]
            Self::PushConstant => builtin_ident("push_constant"),
        }
    }
}

impl BuiltinIdent for AccessMode {
    fn builtin_ident(&self) -> Option<&'static Ident> {
        match self {
            Self::Read => builtin_ident("read"),
            Self::Write => builtin_ident("write"),
            Self::ReadWrite => builtin_ident("read_write"),
        }
    }
}

pub static ATTR_INTRINSIC: LazyLock<Attribute> = LazyLock::new(|| {
    Attribute::Custom(CustomAttribute {
        ident: Ident::new("__intrinsic".to_string()),
        arguments: None,
    })
});

pub static PRELUDE: LazyLock<TranslationUnit> = LazyLock::new(|| {
    let intrinsic: &Attribute = &ATTR_INTRINSIC;
    let abstract_int = builtin_ident("__AbstractInt").unwrap();
    let abstract_float = builtin_ident("__AbstractFloat").unwrap();
    quote_module! {
        // The prelude contains all pre-declared aliases, built-in structs and functions in WGSL.
        // the @#intrinsic attribute indicates that a function definition is defined by the compiler.
        // it means that it is not representable with user code: has generics, or variadics.

        alias vec2i = vec2<i32>;
        alias vec3i = vec3<i32>;
        alias vec4i = vec4<i32>;
        alias vec2u = vec2<u32>;
        alias vec3u = vec3<u32>;
        alias vec4u = vec4<u32>;
        alias vec2f = vec2<f32>;
        alias vec3f = vec3<f32>;
        alias vec4f = vec4<f32>;
        // TODO: these are only enabled with the f16 extension
        alias vec2h = vec2<f16>;
        alias vec3h = vec3<f16>;
        alias vec4h = vec4<f16>;
        alias mat2x2f = mat2x2<f32>;
        alias mat2x3f = mat2x3<f32>;
        alias mat2x4f = mat2x4<f32>;
        alias mat3x2f = mat3x2<f32>;
        alias mat3x3f = mat3x3<f32>;
        alias mat3x4f = mat3x4<f32>;
        alias mat4x2f = mat4x2<f32>;
        alias mat4x3f = mat4x3<f32>;
        alias mat4x4f = mat4x4<f32>;
        // TODO: these are only enabled with the f16 extension
        alias mat2x2h = mat2x2<f16>;
        alias mat2x3h = mat2x3<f16>;
        alias mat2x4h = mat2x4<f16>;
        alias mat3x2h = mat3x2<f16>;
        alias mat3x3h = mat3x3<f16>;
        alias mat3x4h = mat3x4<f16>;
        alias mat4x2h = mat4x2<f16>;
        alias mat4x3h = mat4x3<f16>;
        alias mat4x4h = mat4x4<f16>;

        // internal declarations are prefixed with __, which is not representable in WGSL source
        // therefore it avoids name collisions. AbstractInt and AbstractFloat too.
        struct __frexp_result_f32 { fract: f32, exp: i32 }
        struct __frexp_result_f16 { fract: f16, exp: i32 }
        struct __frexp_result_abstract { fract: #abstract_float, exp: #abstract_int }
        struct __frexp_result_vec2_f32 { fract: vec2<f32>, exp: vec2<i32> }
        struct __frexp_result_vec3_f32 { fract: vec3<f32>, exp: vec3<i32> }
        struct __frexp_result_vec4_f32 { fract: vec4<f32>, exp: vec4<i32> }
        struct __frexp_result_vec2_f16 { fract: vec2<f16>, exp: vec2<i32> }
        struct __frexp_result_vec3_f16 { fract: vec3<f16>, exp: vec3<i32> }
        struct __frexp_result_vec4_f16 { fract: vec4<f16>, exp: vec4<i32> }
        struct __frexp_result_vec2_abstract { fract: vec2<#abstract_float>, exp: vec2<#abstract_int> }
        struct __frexp_result_vec3_abstract { fract: vec3<#abstract_float>, exp: vec3<#abstract_int> }
        struct __frexp_result_vec4_abstract { fract: vec4<#abstract_float>, exp: vec4<#abstract_int> }
        struct __modf_result_f32 { fract: f32, whole: f32 }
        struct __modf_result_f16 { fract: f16, whole: f16 }
        struct __modf_result_abstract { fract: #abstract_float, whole: #abstract_float }
        struct __modf_result_vec2_f32 { fract: vec2<f32>, whole: vec2<f32> }
        struct __modf_result_vec3_f32 { fract: vec3<f32>, whole: vec3<f32> }
        struct __modf_result_vec4_f32 { fract: vec4<f32>, whole: vec4<f32> }
        struct __modf_result_vec2_f16 { fract: vec2<f16>, whole: vec2<f16> }
        struct __modf_result_vec3_f16 { fract: vec3<f16>, whole: vec3<f16> }
        struct __modf_result_vec4_f16 { fract: vec4<f16>, whole: vec4<f16> }
        struct __modf_result_vec2_abstract { fract: vec2<#abstract_float>, whole: vec2<#abstract_float> }
        struct __modf_result_vec3_abstract { fract: vec3<#abstract_float>, whole: vec3<#abstract_float> }
        struct __modf_result_vec4_abstract { fract: vec4<#abstract_float>, whole: vec4<#abstract_float> }
        @generic(T) struct atomic_compare_exchange_result { old_value: T, exchanged: bool }

        // bitcast
        @const @must_use fn bitcast() @#intrinsic {}

        // logical
        @const @must_use fn all() @#intrinsic {}
        @const @must_use fn any() @#intrinsic {}
        @const @must_use fn select() @#intrinsic {}

        // array
        @const @must_use fn arrayLength() @#intrinsic {}

        // numeric
        @const @must_use fn abs() @#intrinsic {}
        @const @must_use fn acos() @#intrinsic {}
        @const @must_use fn acosh() @#intrinsic {}
        @const @must_use fn asin() @#intrinsic {}
        @const @must_use fn asinh() @#intrinsic {}
        @const @must_use fn atan() @#intrinsic {}
        @const @must_use fn atanh() @#intrinsic {}
        @const @must_use fn atan2() @#intrinsic {}
        @const @must_use fn ceil() @#intrinsic {}
        @const @must_use fn clamp() @#intrinsic {}
        @const @must_use fn cos() @#intrinsic {}
        @const @must_use fn cosh() @#intrinsic {}
        @const @must_use fn countLeadingZeros() @#intrinsic {}
        @const @must_use fn countOneBits() @#intrinsic {}
        @const @must_use fn countTrailingZeros() @#intrinsic {}
        @const @must_use fn cross() @#intrinsic {}
        @const @must_use fn degrees() @#intrinsic {}
        @const @must_use fn determinant() @#intrinsic {}
        @const @must_use fn distance() @#intrinsic {}
        @const @must_use fn dot() @#intrinsic {}
        @const @must_use fn dot4U8Packed() @#intrinsic {}
        @const @must_use fn dot4I8Packed() @#intrinsic {}
        @const @must_use fn exp() @#intrinsic {}
        @const @must_use fn exp2() @#intrinsic {}
        @const @must_use fn extractBits() @#intrinsic {}
        @const @must_use fn faceForward() @#intrinsic {}
        @const @must_use fn firstLeadingBit() @#intrinsic {}
        @const @must_use fn firstTrailingBit() @#intrinsic {}
        @const @must_use fn floor() @#intrinsic {}
        @const @must_use fn fma() @#intrinsic {}
        @const @must_use fn fract() @#intrinsic {}
        @const @must_use fn frexp() @#intrinsic {}
        @const @must_use fn insertBits() @#intrinsic {}
        @const @must_use fn inverseSqrt() @#intrinsic {}
        @const @must_use fn ldexp() @#intrinsic {}
        @const @must_use fn length() @#intrinsic {}
        @const @must_use fn log() @#intrinsic {}
        @const @must_use fn log2() @#intrinsic {}
        @const @must_use fn max() @#intrinsic {}
        @const @must_use fn min() @#intrinsic {}
        @const @must_use fn mix() @#intrinsic {}
        @const @must_use fn modf() @#intrinsic {}
        @const @must_use fn normalize() @#intrinsic {}
        @const @must_use fn pow() @#intrinsic {}
        @const @must_use fn quantizeToF16() @#intrinsic {}
        @const @must_use fn radians() @#intrinsic {}
        @const @must_use fn reflect() @#intrinsic {}
        @const @must_use fn refract() @#intrinsic {}
        @const @must_use fn reverseBits() @#intrinsic {}
        @const @must_use fn round() @#intrinsic {}
        @const @must_use fn saturate() @#intrinsic {}
        @const @must_use fn sign() @#intrinsic {}
        @const @must_use fn sin() @#intrinsic {}
        @const @must_use fn sinh() @#intrinsic {}
        @const @must_use fn smoothstep() @#intrinsic {}
        @const @must_use fn sqrt() @#intrinsic {}
        @const @must_use fn step() @#intrinsic {}
        @const @must_use fn tan() @#intrinsic {}
        @const @must_use fn tanh() @#intrinsic {}
        @const @must_use fn transpose() @#intrinsic {}
        @const @must_use fn trunc() @#intrinsic {}

        // derivative
        @must_use fn dpdx() @#intrinsic {}
        @must_use fn dpdxCoarse() @#intrinsic {}
        @must_use fn dpdxFine() @#intrinsic {}
        @must_use fn dpdy() @#intrinsic {}
        @must_use fn dpdyCoarse() @#intrinsic {}
        @must_use fn dpdyFine() @#intrinsic {}
        @must_use fn fwidth() @#intrinsic {}
        @must_use fn fwidthCoarse() @#intrinsic {}
        @must_use fn fwidthFine() @#intrinsic {}

        // texture
        @must_use fn textureDimensions() @#intrinsic {}
        @must_use fn textureGather() @#intrinsic {}
        @must_use fn textureGatherCompare() @#intrinsic {}
        @must_use fn textureLoad() @#intrinsic {}
        @must_use fn textureNumLayers() @#intrinsic {}
        @must_use fn textureNumLevels() @#intrinsic {}
        @must_use fn textureNumSamples() @#intrinsic {}
        @must_use fn textureSample() @#intrinsic {}
        @must_use fn textureSampleBias() @#intrinsic {}
        @must_use fn textureSampleCompare() @#intrinsic {}
        @must_use fn textureSampleCompareLevel() @#intrinsic {}
        @must_use fn textureSampleGrad() @#intrinsic {}
        @must_use fn textureSampleLevel() @#intrinsic {}
        @must_use fn textureSampleBaseClampToEdge() @#intrinsic {}
        fn textureStore() @#intrinsic {}

        // atomic
        fn atomicLoad() @#intrinsic {}
        fn atomicStore() @#intrinsic {}
        fn atomicAdd() @#intrinsic {}
        fn atomicSub() @#intrinsic {}
        fn atomicMax() @#intrinsic {}
        fn atomicMin() @#intrinsic {}
        fn atomicAnd() @#intrinsic {}
        fn atomicOr() @#intrinsic {}
        fn atomicXor() @#intrinsic {}
        fn atomicExchange() @#intrinsic {}
        fn atomicCompareExchangeWeak() @#intrinsic {}

        // packing
        @const @must_use fn pack4x8snorm() @#intrinsic { }
        @const @must_use fn pack4x8unorm() @#intrinsic {}
        @const @must_use fn pack4xI8() @#intrinsic {}
        @const @must_use fn pack4xU8() @#intrinsic {}
        @const @must_use fn pack4xI8Clamp() @#intrinsic {}
        @const @must_use fn pack4xU8Clamp() @#intrinsic {}
        @const @must_use fn pack2x16snorm() @#intrinsic {}
        @const @must_use fn pack2x16unorm() @#intrinsic {}
        @const @must_use fn pack2x16float() @#intrinsic {}
        @const @must_use fn unpack4x8snorm() @#intrinsic {}
        @const @must_use fn unpack4x8unorm() @#intrinsic {}
        @const @must_use fn unpack4xI8() @#intrinsic {}
        @const @must_use fn unpack4xU8() @#intrinsic {}
        @const @must_use fn unpack2x16snorm() @#intrinsic {}
        @const @must_use fn unpack2x16unorm() @#intrinsic {}
        @const @must_use fn unpack2x16float() @#intrinsic {}

        // synchronization
        fn storageBarrier() @#intrinsic {}
        fn textureBarrier() @#intrinsic {}
        fn workgroupBarrier() @#intrinsic {}
        @must_use fn workgroupUniformLoad() @#intrinsic {}

        // subgroup
        @must_use fn subgroupAdd() @#intrinsic {}
        @must_use fn subgroupExclusiveAdd() @#intrinsic {}
        @must_use fn subgroupInclusiveAdd() @#intrinsic {}
        @must_use fn subgroupAll() @#intrinsic {}
        @must_use fn subgroupAnd() @#intrinsic {}
        @must_use fn subgroupAny() @#intrinsic {}
        @must_use fn subgroupBallot() @#intrinsic {}
        @must_use fn subgroupBroadcast() @#intrinsic {}
        @must_use fn subgroupBroadcastFirst() @#intrinsic {}
        @must_use fn subgroupElect() @#intrinsic {}
        @must_use fn subgroupMax() @#intrinsic {}
        @must_use fn subgroupMin() @#intrinsic {}
        @must_use fn subgroupMul() @#intrinsic {}
        @must_use fn subgroupExclusiveMul() @#intrinsic {}
        @must_use fn subgroupInclusiveMul() @#intrinsic {}
        @must_use fn subgroupOr() @#intrinsic {}
        @must_use fn subgroupShuffle() @#intrinsic {}
        @must_use fn subgroupShuffleDown() @#intrinsic {}
        @must_use fn subgroupShuffleUp() @#intrinsic {}
        @must_use fn subgroupShuffleXor() @#intrinsic {}
        @must_use fn subgroupXor() @#intrinsic {}

        // quad
        @must_use fn quadBroadcast() @#intrinsic {}
        @must_use fn quadSwapDiagonal() @#intrinsic {}
        @must_use fn quadSwapX() @#intrinsic {}
        @must_use fn quadSwapY() @#intrinsic {}
    }
});

fn array_ctor_ty_t(tplt: ArrayTemplate, args: &[Type]) -> Result<Type, E> {
    if let Some(arg) = args
        .iter()
        .find(|arg| !arg.is_convertible_to(&tplt.inner_ty()))
    {
        Err(E::Conversion(arg.clone(), tplt.inner_ty()))
    } else {
        Ok(tplt.ty())
    }
}

fn array_ctor_ty(args: &[Type]) -> Result<Type, E> {
    let ty = convert_all_ty(args).ok_or(E::Builtin("array elements are incompatible"))?;
    Ok(Type::Array(Box::new(ty.clone()), Some(args.len())))
}

fn mat_ctor_ty_t(c: u8, r: u8, tplt: MatTemplate, args: &[Type]) -> Result<Type, E> {
    // overload 1: mat conversion constructor
    if let [ty @ Type::Mat(c2, r2, _)] = args {
        // note: this is an explicit conversion, not automatic conversion
        if *c2 != c || *r2 != r {
            return Err(E::Conversion(ty.clone(), tplt.ty(c, r)));
        }
    } else {
        if args.is_empty() {
            return Err(E::Builtin("matrix constructor expects arguments"));
        }
        let ty = convert_all_ty(args).ok_or(E::Builtin("matrix components are incompatible"))?;
        let ty = ty
            .convert_inner_to(tplt.inner_ty())
            .ok_or(E::Conversion(ty.inner_ty(), tplt.inner_ty().clone()))?;

        // overload 2: mat from column vectors
        if ty.is_vec() {
            if args.len() != c as usize {
                return Err(E::ParamCount(format!("mat{c}x{r}"), c as usize, args.len()));
            }
        }
        // overload 3: mat from float values
        else if ty.is_float() {
            let n = c as usize * r as usize;
            if args.len() != n {
                return Err(E::ParamCount(format!("mat{c}x{r}"), n, args.len()));
            }
        } else {
            return Err(E::Builtin(
                "matrix constructor expects float or vector of float arguments",
            ));
        }
    }

    Ok(tplt.ty(c, r))
}

fn mat_ctor_ty(c: u8, r: u8, args: &[Type]) -> Result<Type, E> {
    // overload 1: mat conversion constructor
    if let [ty @ Type::Mat(c2, r2, ty2)] = args {
        // note: this is an explicit conversion, not automatic conversion
        if *c2 != c || *r2 != r {
            return Err(E::Conversion(ty.clone(), Type::Mat(c, r, ty2.clone())));
        }
        Ok(ty.clone())
    } else {
        let ty = convert_all_ty(args).ok_or(E::Builtin("matrix components are incompatible"))?;
        let inner_ty = ty.inner_ty();

        if !inner_ty.is_float() && !inner_ty.is_abstract_int() {
            return Err(E::Builtin(
                "matrix constructor expects float or vector of float arguments",
            ));
        }

        // overload 2: mat from column vectors
        if ty.is_vec() {
            if args.len() != c as usize {
                return Err(E::ParamCount(format!("mat{c}x{r}"), c as usize, args.len()));
            }
        }
        // overload 3: mat from float values
        else if ty.is_float() || ty.is_abstract_int() {
            let n = c as usize * r as usize;
            if args.len() != n {
                return Err(E::ParamCount(format!("mat{c}x{r}"), n, args.len()));
            }
        } else {
            return Err(E::Builtin(
                "matrix constructor expects float or vector of float arguments",
            ));
        }

        Ok(Type::Mat(c, r, inner_ty.into()))
    }
}

fn vec_ctor_ty_t(n: u8, tplt: VecTemplate, args: &[Type]) -> Result<Type, E> {
    if let [arg] = args {
        // overload 1: vec init from single scalar value
        if arg.is_scalar() {
            if !arg.is_convertible_to(tplt.inner_ty()) {
                return Err(E::Conversion(arg.clone(), tplt.inner_ty().clone()));
            }
        }
        // overload 2: vec conversion constructor
        else if arg.is_vec() {
            // note: this is an explicit conversion, not automatic conversion
        } else {
            return Err(E::Conversion(arg.clone(), tplt.inner_ty().clone()));
        }
    }
    // overload 3: vec init from component values
    else {
        // flatten vecN args
        let n2 = args
            .iter()
            .try_fold(0, |acc, arg| match arg {
                ty if ty.is_scalar() => ty.is_convertible_to(tplt.inner_ty()).then_some(acc + 1),
                Type::Vec(n, ty) => ty.is_convertible_to(tplt.inner_ty()).then_some(acc + n),
                _ => None,
            })
            .ok_or(E::Builtin(
                "vector constructor expects scalar or vector arguments",
            ))?;
        if n2 != n {
            return Err(E::ParamCount(format!("vec{n}"), n as usize, args.len()));
        }
    }

    Ok(tplt.ty(n))
}

fn vec_ctor_ty(n: u8, args: &[Type]) -> Result<Type, E> {
    if let [arg] = args {
        // overload 1: vec init from single scalar value
        if arg.is_scalar() {
        }
        // overload 2: vec conversion constructor
        else if arg.is_vec() {
            // note: `vecN(e: vecN<S>) -> vecN<S>` is no-op
        } else {
            return Err(E::Builtin(
                "vector constructor expects scalar or vector arguments",
            ));
        }
        Ok(Type::Vec(n, arg.inner_ty().into()))
    }
    // overload 3: vec init from component values
    else if !args.is_empty() {
        // flatten vecN args
        let n2 = args
            .iter()
            .try_fold(0, |acc, arg| match arg {
                ty if ty.is_scalar() => Some(acc + 1),
                Type::Vec(n, _) => Some(acc + n),
                _ => None,
            })
            .ok_or(E::Builtin(
                "vector constructor expects scalar or vector arguments",
            ))?;
        if n2 != n {
            return Err(E::ParamCount(format!("vec{n}"), n as usize, args.len()));
        }

        let tys = args.iter().map(|arg| arg.inner_ty()).collect_vec();
        let ty = convert_all_ty(&tys).ok_or(E::Builtin("vector components are incompatible"))?;

        Ok(Type::Vec(n, ty.clone().into()))
    }
    // overload 3: zero-vec
    else {
        Ok(Type::Vec(n, Type::AbstractInt.into()))
    }
}

pub fn constructor_type(ty: &TypeExpression, args: &[Type], ctx: &mut Context) -> Result<Type, E> {
    match (ty.ident.name().as_str(), ty.template_args.as_deref(), args) {
        ("array", Some(t), []) => Ok(ArrayTemplate::parse(t, ctx)?.ty()),
        ("array", Some(t), _) => array_ctor_ty_t(ArrayTemplate::parse(t, ctx)?, args),
        ("array", None, _) => array_ctor_ty(args),
        ("bool", None, []) => Ok(Type::Bool),
        ("bool", None, [a]) if a.is_scalar() => Ok(Type::Bool),
        ("i32", None, []) => Ok(Type::I32),
        ("i32", None, [a]) if a.is_scalar() => Ok(Type::I32),
        ("u32", None, []) => Ok(Type::U32),
        ("u32", None, [a]) if a.is_scalar() => Ok(Type::U32),
        ("f32", None, []) => Ok(Type::F32),
        ("f32", None, [a]) if a.is_scalar() => Ok(Type::F32),
        ("f16", None, []) => Ok(Type::F16),
        ("f16", None, [a]) if a.is_scalar() => Ok(Type::F16),
        ("mat2x2", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(2, 2)),
        ("mat2x2", Some(t), _) => mat_ctor_ty_t(2, 2, MatTemplate::parse(t, ctx)?, args),
        ("mat2x2", None, _) => mat_ctor_ty(2, 2, args),
        ("mat2x3", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(2, 3)),
        ("mat2x3", Some(t), _) => mat_ctor_ty_t(2, 3, MatTemplate::parse(t, ctx)?, args),
        ("mat2x3", None, _) => mat_ctor_ty(2, 3, args),
        ("mat2x4", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(2, 4)),
        ("mat2x4", Some(t), _) => mat_ctor_ty_t(2, 4, MatTemplate::parse(t, ctx)?, args),
        ("mat2x4", None, _) => mat_ctor_ty(2, 4, args),
        ("mat3x2", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(3, 2)),
        ("mat3x2", Some(t), _) => mat_ctor_ty_t(3, 2, MatTemplate::parse(t, ctx)?, args),
        ("mat3x2", None, _) => mat_ctor_ty(3, 2, args),
        ("mat3x3", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(3, 3)),
        ("mat3x3", Some(t), _) => mat_ctor_ty_t(3, 3, MatTemplate::parse(t, ctx)?, args),
        ("mat3x3", None, _) => mat_ctor_ty(3, 3, args),
        ("mat3x4", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(3, 4)),
        ("mat3x4", Some(t), _) => mat_ctor_ty_t(3, 4, MatTemplate::parse(t, ctx)?, args),
        ("mat3x4", None, _) => mat_ctor_ty(3, 4, args),
        ("mat4x2", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(4, 2)),
        ("mat4x2", Some(t), _) => mat_ctor_ty_t(4, 2, MatTemplate::parse(t, ctx)?, args),
        ("mat4x2", None, _) => mat_ctor_ty(4, 2, args),
        ("mat4x3", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(4, 3)),
        ("mat4x3", Some(t), _) => mat_ctor_ty_t(4, 3, MatTemplate::parse(t, ctx)?, args),
        ("mat4x3", None, _) => mat_ctor_ty(4, 3, args),
        ("mat4x4", Some(t), []) => Ok(MatTemplate::parse(t, ctx)?.ty(4, 4)),
        ("mat4x4", Some(t), _) => mat_ctor_ty_t(4, 4, MatTemplate::parse(t, ctx)?, args),
        ("mat4x4", None, _) => mat_ctor_ty(4, 4, args),
        ("vec2", Some(t), []) => Ok(VecTemplate::parse(t, ctx)?.ty(2)),
        ("vec2", Some(t), _) => vec_ctor_ty_t(2, VecTemplate::parse(t, ctx)?, args),
        ("vec2", None, _) => vec_ctor_ty(2, args),
        ("vec3", Some(t), []) => Ok(VecTemplate::parse(t, ctx)?.ty(3)),
        ("vec3", Some(t), _) => vec_ctor_ty_t(3, VecTemplate::parse(t, ctx)?, args),
        ("vec3", None, _) => vec_ctor_ty(3, args),
        ("vec4", Some(t), []) => Ok(VecTemplate::parse(t, ctx)?.ty(4)),
        ("vec4", Some(t), _) => vec_ctor_ty_t(4, VecTemplate::parse(t, ctx)?, args),
        ("vec4", None, _) => vec_ctor_ty(4, args),
        _ => Err(E::Signature(ty.clone(), args.to_vec())),
    }
}

pub fn builtin_fn_type(
    ty: &TypeExpression,
    args: &[Type],
    ctx: &mut Context,
) -> Result<Option<Type>, E> {
    fn is_float(ty: &Type) -> bool {
        ty.is_float() || ty.is_vec() && ty.inner_ty().is_float()
    }
    fn is_numeric(ty: &Type) -> bool {
        ty.is_numeric() || ty.is_vec() && ty.inner_ty().is_numeric()
    }
    fn is_integer(ty: &Type) -> bool {
        ty.is_integer() || ty.is_vec() && ty.inner_ty().is_integer()
    }
    let err = || E::Signature(ty.clone(), args.to_vec());

    match (ty.ident.name().as_str(), ty.template_args.as_deref(), args) {
        // bitcast
        ("bitcast", Some(t), [_]) => Ok(Some(BitcastTemplate::parse(t, ctx)?.ty())),
        // logical
        ("all", None, [_]) | ("any", None, [_]) => Ok(Some(Type::Bool)),
        ("select", None, [a1, a2, a3]) if (a1.is_scalar() || a1.is_vec()) && a3.is_bool() => {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        ("select", None, [a1, a2, a3])
            if (a1.is_vec()) && a3.is_vec() && a3.inner_ty().is_bool() =>
        {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        // array
        ("arrayLength", None, [_]) => Ok(Some(Type::U32)),
        // numeric
        ("abs", None, [a]) if is_numeric(a) => Ok(Some(a.clone())),
        ("acos", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("acosh", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("asin", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("asinh", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("atan", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("atanh", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("atan2", None, [a1, a2]) if is_float(a1) => {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        ("ceil", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("clamp", None, [a1, _, _]) if is_numeric(a1) => {
            convert_all_ty(args).cloned().map(Some).ok_or_else(err)
        }
        ("cos", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("cosh", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("countLeadingZeros", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        ("countOneBits", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        ("countTrailingZeros", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        ("cross", None, [a1, a2]) if a1.is_vec() && a1.inner_ty().is_float() => {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        ("degrees", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("determinant", None, [a @ Type::Mat(c, r, _)]) if c == r => Ok(Some(a.clone())),
        ("distance", None, [a1, a2]) if is_float(a1) => convert_ty(a1, a2)
            .map(|ty| Some(ty.inner_ty()))
            .ok_or_else(err),
        ("dot", None, [a1, a2]) if a1.is_vec() && a1.inner_ty().is_numeric() => convert_ty(a1, a2)
            .map(|ty| Some(ty.inner_ty()))
            .ok_or_else(err),
        ("dot4U8Packed", None, [a1, a2])
            if a1.is_convertible_to(&Type::U32) && a2.is_convertible_to(&Type::U32) =>
        {
            Ok(Some(Type::U32))
        }
        ("dot4I8Packed", None, [a1, a2])
            if a1.is_convertible_to(&Type::U32) && a2.is_convertible_to(&Type::U32) =>
        {
            Ok(Some(Type::I32))
        }
        ("exp", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("exp2", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("extractBits", None, [a1, a2, a3])
            if is_integer(a1)
                && a2.is_convertible_to(&Type::U32)
                && a3.is_convertible_to(&Type::U32) =>
        {
            Ok(Some(a1.concretize()))
        }
        ("faceForward", None, [a1, _, _]) if a1.is_vec() && a1.inner_ty().is_float() => {
            convert_all_ty(args).cloned().map(Some).ok_or_else(err)
        }
        ("firstLeadingBit", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        ("firstTrailingBit", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        ("floor", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("fma", None, [a1, _, _]) if is_float(a1) => {
            convert_all_ty(args).cloned().map(Some).ok_or_else(err)
        }
        ("fract", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("frexp", None, [a]) if is_float(a) => Ok(Some(Type::Struct(
            frexp_struct_name(a).unwrap().to_string(),
        ))),
        ("insertBits", None, [a1, a2, a3, a4])
            if is_integer(a1)
                && a3.is_convertible_to(&Type::U32)
                && a4.is_convertible_to(&Type::U32) =>
        {
            convert_ty(a1, a2)
                .map(|ty| Some(ty.concretize()))
                .ok_or_else(err)
        }
        ("inverseSqrt", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("ldexp", None, [a1, a2])
            if (a1.is_vec()
                && a1.inner_ty().is_float()
                && a2.is_vec()
                && a2.inner_ty().concretize().is_i_32()
                || a1.is_float() && a2.concretize().is_i_32())
                && (a1.is_concrete() && a2.is_concrete()
                    || a1.is_abstract() && a2.is_abstract()) =>
        {
            Ok(Some(a1.clone()))
        }
        ("length", None, [a]) if is_float(a) => Ok(Some(a.inner_ty())),
        ("log", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("log2", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("max", None, [a1, a2]) if is_numeric(a1) => {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        ("min", None, [a1, a2]) if is_numeric(a1) => {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        ("mix", None, [Type::Vec(n1, ty1), Type::Vec(n2, ty2), a3])
            if n1 == n2 && a3.is_float() =>
        {
            convert_all_ty([ty1, ty2, a3])
                .map(|inner| Some(Type::Vec(*n1, inner.clone().into())))
                .ok_or_else(err)
        }
        ("mix", None, [a1, _, _]) if is_float(a1) => {
            convert_all_ty(args).cloned().map(Some).ok_or_else(err)
        }
        ("modf", None, [a]) if is_float(a) => {
            Ok(Some(Type::Struct(modf_struct_name(a).unwrap().to_string())))
        }
        ("normalize", None, [a @ Type::Vec(_, ty)]) if ty.is_float() => Ok(Some(a.clone())),
        ("pow", None, [a1, a2]) => convert_ty(a1, a2).cloned().map(Some).ok_or_else(err),
        ("quantizeToF16", None, [a])
            if a.concretize().is_f_32() || a.is_vec() && a.inner_ty().concretize().is_f_32() =>
        {
            Ok(Some(a.clone()))
        }
        ("radians", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("reflect", None, [a1, a2]) if a1.is_vec() && a1.inner_ty().is_float() => {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        ("refract", None, [Type::Vec(n1, ty1), Type::Vec(n2, ty2), a3])
            if n1 == n2 && a3.is_float() =>
        {
            convert_all_ty([ty1, ty2, a3])
                .map(|inner| Some(Type::Vec(*n1, inner.clone().into())))
                .ok_or_else(err)
        }
        ("reverseBits", None, [a]) if is_integer(a) => Ok(Some(a.clone())),
        ("round", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("saturate", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("sign", None, [a]) if is_numeric(a) && !a.inner_ty().is_u_32() => Ok(Some(a.clone())),
        ("sin", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("sinh", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("smoothstep", None, [a1, _, _]) if is_float(a1) => {
            convert_all_ty(args).cloned().map(Some).ok_or_else(err)
        }
        ("sqrt", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("step", None, [a1, a2]) if is_float(a1) => {
            convert_ty(a1, a2).cloned().map(Some).ok_or_else(err)
        }
        ("tan", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("tanh", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        ("transpose", None, [Type::Mat(c, r, ty)]) => Ok(Some(Type::Mat(*r, *c, ty.clone()))),
        ("trunc", None, [a]) if is_float(a) => Ok(Some(a.clone())),
        // derivative
        ("dpdx", None, [a]) if is_float(a) => Ok(Some(a.convert_inner_to(&Type::F32).unwrap())),
        ("dpdxCoarse", None, [a]) if is_float(a) => {
            Ok(Some(a.convert_inner_to(&Type::F32).unwrap()))
        }
        ("dpdxFine", None, [a]) if is_float(a) => Ok(Some(a.convert_inner_to(&Type::F32).unwrap())),
        ("dpdy", None, [a]) if is_float(a) => Ok(Some(a.convert_inner_to(&Type::F32).unwrap())),
        ("dpdyCoarse", None, [a]) if is_float(a) => {
            Ok(Some(a.convert_inner_to(&Type::F32).unwrap()))
        }
        ("dpdyFine", None, [a]) if is_float(a) => Ok(Some(a.convert_inner_to(&Type::F32).unwrap())),
        ("fwidth", None, [a]) if is_float(a) => Ok(Some(a.convert_inner_to(&Type::F32).unwrap())),
        ("fwidthCoarse", None, [a]) if is_float(a) => {
            Ok(Some(a.convert_inner_to(&Type::F32).unwrap()))
        }
        ("fwidthFine", None, [a]) if is_float(a) => {
            Ok(Some(a.convert_inner_to(&Type::F32).unwrap()))
        }
        // texture
        // TODO check arguments for texture functions
        // some of these are a bit more lenient. The goal here is just to get the
        // valid return type which is needed for type inference.
        ("textureDimensions", None, [Type::Texture(t)] | [Type::Texture(t), _])
            if t.dimensions().is_d_1() =>
        {
            Ok(Some(Type::U32))
        }
        ("textureDimensions", None, [Type::Texture(t)] | [Type::Texture(t), _])
            if t.dimensions().is_d_2() =>
        {
            Ok(Some(Type::Vec(2, Type::U32.into())))
        }
        ("textureDimensions", None, [Type::Texture(t)] | [Type::Texture(t), _])
            if t.dimensions().is_d_3() =>
        {
            Ok(Some(Type::Vec(3, Type::U32.into())))
        }
        ("textureGather", None, [_, Type::Texture(t), ..]) if t.is_sampled() => Ok(Some(
            Type::Vec(4, Box::new(t.sampled_type().unwrap().into())),
        )),
        ("textureGather", None, [Type::Texture(t), ..]) if t.is_depth() => {
            Ok(Some(Type::Vec(4, Type::F32.into())))
        }
        ("textureGatherCompare", None, [Type::Texture(t), ..]) if t.is_depth() => {
            Ok(Some(Type::Vec(4, Type::F32.into())))
        }
        ("textureLoad", None, [Type::Texture(TextureType::DepthMultisampled2D), ..]) => {
            Ok(Some(Type::F32))
        }
        ("textureLoad", None, [Type::Texture(t), ..]) if t.is_depth() => Ok(Some(Type::F32)),
        ("textureLoad", None, [Type::Texture(t), ..]) => {
            Ok(Some(Type::Vec(4, Box::new(t.channel_type().into()))))
        }
        ("textureNumLayers", None, [Type::Texture(t)])
            if t.is_sampled() || t.is_depth() || t.is_storage() =>
        {
            Ok(Some(Type::U32))
        }
        ("textureNumLevels", None, [Type::Texture(t)]) if t.is_sampled() || t.is_depth() => {
            Ok(Some(Type::U32))
        }
        ("textureNumSamples", None, [Type::Texture(t)]) if t.is_multisampled() => {
            Ok(Some(Type::U32))
        }
        ("textureSample", None, [Type::Texture(t), ..]) if t.is_sampled() => {
            Ok(Some(Type::Vec(4, Box::new(Type::F32))))
        }
        ("textureSample", None, [Type::Texture(t), ..]) if t.is_depth() => Ok(Some(Type::F32)),
        ("textureSampleBias", None, [Type::Texture(t), ..]) if t.is_sampled() => {
            Ok(Some(Type::Vec(4, Box::new(Type::F32))))
        }
        ("textureSampleCompare", None, [Type::Texture(t), ..]) if t.is_depth() => {
            Ok(Some(Type::F32))
        }
        ("textureSampleCompareLevel", None, [Type::Texture(t), ..]) if t.is_depth() => {
            Ok(Some(Type::F32))
        }
        ("textureSampleGrad", None, [Type::Texture(t), ..]) if t.is_sampled() => {
            Ok(Some(Type::Vec(4, Box::new(Type::F32))))
        }
        ("textureSampleLevel", None, [Type::Texture(t), ..]) if t.is_sampled() => {
            Ok(Some(Type::Vec(4, Box::new(Type::F32))))
        }
        ("textureSampleLevel", None, [Type::Texture(t), ..]) if t.is_depth() => Ok(Some(Type::F32)),
        ("textureSampleBaseClampToEdge", None, [Type::Texture(t), ..])
            if t.is_sampled_2_d() || t.is_external() =>
        {
            Ok(Some(Type::Vec(4, Box::new(Type::F32))))
        }
        ("textureStore", None, [Type::Texture(t), ..]) if t.is_storage() => Ok(None),
        // atomic
        // TODO check arguments for atomic functions
        ("atomicLoad", None, [Type::Ptr(_, t)]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicStore", None, [Type::Ptr(_, t)]) if matches!(**t, Type::Atomic(_)) => Ok(None),
        ("atomicAdd", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicSub", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicMax", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicMin", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicAnd", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicOr", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicXor", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicExchange", None, [Type::Ptr(_, t), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicCompareExchangeWeak", None, [Type::Ptr(_, t), _, _])
            if matches!(**t, Type::Atomic(_)) =>
        {
            Ok(Some(Type::Struct(
                "__atomic_compare_exchange_result".to_string(),
            )))
        }
        // packing
        ("pack4x8snorm", None, [a]) if a.is_convertible_to(&Type::Vec(4, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack4x8unorm", None, [a]) if a.is_convertible_to(&Type::Vec(4, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack4xI8", None, [a]) if a.is_convertible_to(&Type::Vec(4, Type::I32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack4xU8", None, [a]) if a.is_convertible_to(&Type::Vec(4, Type::U32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack4xI8Clamp", None, [a]) if a.is_convertible_to(&Type::Vec(2, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack4xU8Clamp", None, [a]) if a.is_convertible_to(&Type::Vec(2, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack2x16snorm", None, [a]) if a.is_convertible_to(&Type::Vec(2, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack2x16unorm", None, [a]) if a.is_convertible_to(&Type::Vec(2, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack2x16float", None, [a]) if a.is_convertible_to(&Type::Vec(2, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("unpack4x8snorm", None, [a]) if a.is_convertible_to(&Type::U32) => {
            Ok(Some(Type::Vec(4, Type::F32.into())))
        }
        ("unpack4x8unorm", None, [a]) if a.is_convertible_to(&Type::U32) => {
            Ok(Some(Type::Vec(4, Type::F32.into())))
        }
        ("unpack4xI8", None, [a]) if a.is_convertible_to(&Type::U32) => {
            Ok(Some(Type::Vec(4, Type::I32.into())))
        }
        ("unpack4xU8", None, [a]) if a.is_convertible_to(&Type::U32) => {
            Ok(Some(Type::Vec(4, Type::U32.into())))
        }
        ("unpack2x16snorm", None, [a]) if a.is_convertible_to(&Type::U32) => {
            Ok(Some(Type::Vec(2, Type::F32.into())))
        }
        ("unpack2x16unorm", None, [a]) if a.is_convertible_to(&Type::U32) => {
            Ok(Some(Type::Vec(2, Type::F32.into())))
        }
        ("unpack2x16float", None, [a]) if a.is_convertible_to(&Type::U32) => {
            Ok(Some(Type::Vec(2, Type::F32.into())))
        }
        // synchronization
        ("storageBarrier", None, []) => Ok(None),
        ("textureBarrier", None, []) => Ok(None),
        ("workgroupBarrier", None, []) => Ok(None),
        ("workgroupUniformLoad", None, [Type::Ptr(AddressSpace::Workgroup, t)]) => {
            Ok(Some(*t.clone()))
        }
        // subgroup
        ("subgroupAdd", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupExclusiveAdd", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupInclusiveAdd", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupAll", None, [Type::Bool]) => Ok(Some(Type::Bool)),
        ("subgroupAnd", None, [Type::Bool]) => Ok(Some(Type::Bool)),
        ("subgroupAny", None, [Type::Bool]) => Ok(Some(Type::Bool)),
        ("subgroupBallot", None, [Type::Bool]) => Ok(Some(Type::Vec(4, Type::U32.into()))),
        ("subgroupBroadcast", None, [a1, a2]) if is_numeric(a1) && a2.is_integer() => {
            Ok(Some(a1.concretize()))
        }
        ("subgroupBroadcastFirst", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupElect", None, []) => Ok(Some(Type::Bool)),
        ("subgroupMax", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupMin", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupMul", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupExclusiveMul", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupInclusiveMul", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupOr", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        ("subgroupShuffle", None, [a1, a2]) if is_numeric(a1) && a2.is_integer() => {
            Ok(Some(a1.concretize()))
        }
        ("subgroupShuffleDown", None, [a1, a2]) if is_numeric(a1) && a2.is_integer() => {
            Ok(Some(a1.concretize()))
        }
        ("subgroupShuffleUp", None, [a1, a2]) if is_numeric(a1) && a2.is_integer() => {
            Ok(Some(a1.concretize()))
        }
        ("subgroupShuffleXor", None, [a1, a2]) if is_numeric(a1) && a2.is_integer() => {
            Ok(Some(a1.concretize()))
        }
        ("subgroupXor", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        // quad
        ("quadBroadcast", None, [a1, a2]) if is_numeric(a1) && a2.is_integer() => {
            Ok(Some(a1.concretize()))
        }
        ("quadSwapDiagonal", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("quadSwapX", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("quadSwapY", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        _ => Err(err()),
    }
}

pub fn is_constructor_fn(name: &str) -> bool {
    matches!(
        name,
        "array"
            | "bool"
            | "i32"
            | "u32"
            | "f32"
            | "f16"
            | "mat2x2"
            | "mat2x3"
            | "mat2x4"
            | "mat3x2"
            | "mat3x3"
            | "mat3x4"
            | "mat4x2"
            | "mat4x3"
            | "mat4x4"
            | "vec2"
            | "vec3"
            | "vec4"
    )
}

pub fn call_builtin(
    ty: &TypeExpression,
    args: Vec<Instance>,
    ctx: &mut Context,
) -> Result<Instance, E> {
    match (
        ty.ident.name().as_str(),
        ty.template_args.as_deref(),
        args.as_slice(),
    ) {
        // constructors
        ("array", Some(t), []) => Instance::zero_value(&ArrayTemplate::parse(t, ctx)?.ty(), ctx),
        ("array", Some(t), a) => call_array_t(ArrayTemplate::parse(t, ctx)?, a),
        ("array", None, a) => call_array(a),
        ("bool", None, []) => Instance::zero_value(&Type::Bool, ctx),
        ("bool", None, [a1]) => call_bool_1(a1),
        ("i32", None, []) => Instance::zero_value(&Type::I32, ctx),
        ("i32", None, [a1]) => call_i32_1(a1),
        ("u32", None, []) => Instance::zero_value(&Type::U32, ctx),
        ("u32", None, [a1]) => call_u32_1(a1),
        ("f32", None, []) => Instance::zero_value(&Type::F32, ctx),
        ("f32", None, [a1]) => call_f32_1(a1, ctx.stage),
        ("f16", None, []) => Instance::zero_value(&Type::F16, ctx),
        ("f16", None, [a1]) => call_f16_1(a1, ctx.stage),
        ("mat2x2", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(2, 2), ctx),
        ("mat2x2", Some(t), a) => call_mat_t(2, 2, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat2x2", None, a) => call_mat(2, 2, a),
        ("mat2x3", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(2, 3), ctx),
        ("mat2x3", Some(t), a) => call_mat_t(2, 3, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat2x3", None, a) => call_mat(2, 3, a),
        ("mat2x4", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(2, 4), ctx),
        ("mat2x4", Some(t), a) => call_mat_t(2, 4, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat2x4", None, a) => call_mat(2, 4, a),
        ("mat3x2", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(3, 2), ctx),
        ("mat3x2", Some(t), a) => call_mat_t(3, 2, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat3x2", None, a) => call_mat(3, 2, a),
        ("mat3x3", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(3, 3), ctx),
        ("mat3x3", Some(t), a) => call_mat_t(3, 3, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat3x3", None, a) => call_mat(3, 3, a),
        ("mat3x4", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(3, 4), ctx),
        ("mat3x4", Some(t), a) => call_mat_t(3, 4, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat3x4", None, a) => call_mat(3, 4, a),
        ("mat4x2", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(4, 2), ctx),
        ("mat4x2", Some(t), a) => call_mat_t(4, 2, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat4x2", None, a) => call_mat(4, 2, a),
        ("mat4x3", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(4, 3), ctx),
        ("mat4x3", Some(t), a) => call_mat_t(4, 3, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat4x3", None, a) => call_mat(4, 3, a),
        ("mat4x4", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t, ctx)?.ty(4, 4), ctx),
        ("mat4x4", Some(t), a) => call_mat_t(4, 4, MatTemplate::parse(t, ctx)?, a, ctx.stage),
        ("mat4x4", None, a) => call_mat(4, 4, a),
        ("vec2", Some(t), []) => Instance::zero_value(&VecTemplate::parse(t, ctx)?.ty(2), ctx),
        ("vec2", Some(t), a) => call_vec_t(2, VecTemplate::parse(t, ctx)?, a, ctx.stage),
        ("vec2", None, a) => call_vec(2, a),
        ("vec3", Some(t), []) => Instance::zero_value(&VecTemplate::parse(t, ctx)?.ty(3), ctx),
        ("vec3", Some(t), a) => call_vec_t(3, VecTemplate::parse(t, ctx)?, a, ctx.stage),
        ("vec3", None, a) => call_vec(3, a),
        ("vec4", Some(t), []) => Instance::zero_value(&VecTemplate::parse(t, ctx)?.ty(4), ctx),
        ("vec4", Some(t), a) => call_vec_t(4, VecTemplate::parse(t, ctx)?, a, ctx.stage),
        ("vec4", None, a) => call_vec(4, a),
        // bitcast
        ("bitcast", Some(t), [a1]) => call_bitcast_t(BitcastTemplate::parse(t, ctx)?, a1),
        // logical
        ("all", None, [a]) => call_all(a),
        ("any", None, [a]) => call_any(a),
        ("select", None, [a1, a2, a3]) => call_select(a1, a2, a3),
        // array
        ("arrayLength", None, [a]) => call_arraylength(a),
        // numeric
        ("abs", None, [a]) => call_abs(a),
        ("acos", None, [a]) => call_acos(a),
        ("acosh", None, [a]) => call_acosh(a),
        ("asin", None, [a]) => call_asin(a),
        ("asinh", None, [a]) => call_asinh(a),
        ("atan", None, [a]) => call_atan(a),
        ("atanh", None, [a]) => call_atanh(a),
        ("atan2", None, [a1, a2]) => call_atan2(a1, a2),
        ("ceil", None, [a]) => call_ceil(a),
        ("clamp", None, [a1, a2, a3]) => call_clamp(a1, a2, a3),
        ("cos", None, [a]) => call_cos(a),
        ("cosh", None, [a]) => call_cosh(a),
        ("countLeadingZeros", None, [a]) => call_countleadingzeros(a),
        ("countOneBits", None, [a]) => call_countonebits(a),
        ("countTrailingZeros", None, [a]) => call_counttrailingzeros(a),
        ("cross", None, [a1, a2]) => call_cross(a1, a2, ctx.stage),
        ("degrees", None, [a]) => call_degrees(a),
        ("determinant", None, [a]) => call_determinant(a),
        ("distance", None, [a1, a2]) => call_distance(a1, a2, ctx.stage),
        ("dot", None, [a1, a2]) => call_dot(a1, a2, ctx.stage),
        ("dot4U8Packed", None, [a1, a2]) => call_dot4u8packed(a1, a2),
        ("dot4I8Packed", None, [a1, a2]) => call_dot4i8packed(a1, a2),
        ("exp", None, [a]) => call_exp(a),
        ("exp2", None, [a]) => call_exp2(a),
        ("extractBits", None, [a1, a2, a3]) => call_extractbits(a1, a2, a3),
        ("faceForward", None, [a1, a2, a3]) => call_faceforward(a1, a2, a3),
        ("firstLeadingBit", None, [a]) => call_firstleadingbit(a),
        ("firstTrailingBit", None, [a]) => call_firsttrailingbit(a),
        ("floor", None, [a]) => call_floor(a),
        ("fma", None, [a1, a2, a3]) => call_fma(a1, a2, a3),
        ("fract", None, [a]) => call_fract(a, ctx.stage),
        ("frexp", None, [a]) => call_frexp(a),
        ("insertBits", None, [a1, a2, a3, a4]) => call_insertbits(a1, a2, a3, a4),
        ("inverseSqrt", None, [a]) => call_inversesqrt(a),
        ("ldexp", None, [a1, a2]) => call_ldexp(a1, a2),
        ("length", None, [a]) => call_length(a),
        ("log", None, [a]) => call_log(a),
        ("log2", None, [a]) => call_log2(a),
        ("max", None, [a1, a2]) => call_max(a1, a2),
        ("min", None, [a1, a2]) => call_min(a1, a2),
        ("mix", None, [a1, a2, a3]) => call_mix(a1, a2, a3),
        ("modf", None, [a]) => call_modf(a),
        ("normalize", None, [a]) => call_normalize(a),
        ("pow", None, [a1, a2]) => call_pow(a1, a2),
        ("quantizeToF16", None, [a]) => call_quantizetof16(a),
        ("radians", None, [a]) => call_radians(a),
        ("reflect", None, [a1, a2]) => call_reflect(a1, a2),
        ("refract", None, [a1, a2, a3]) => call_refract(a1, a2, a3),
        ("reverseBits", None, [a]) => call_reversebits(a),
        ("round", None, [a]) => call_round(a),
        ("saturate", None, [a]) => call_saturate(a),
        ("sign", None, [a]) => call_sign(a),
        ("sin", None, [a]) => call_sin(a),
        ("sinh", None, [a]) => call_sinh(a),
        ("smoothstep", None, [a1, a2, a3]) => call_smoothstep(a1, a2, a3),
        ("sqrt", None, [a]) => call_sqrt(a),
        ("step", None, [a1, a2]) => call_step(a1, a2),
        ("tan", None, [a]) => call_tan(a),
        ("tanh", None, [a]) => call_tanh(a),
        ("transpose", None, [a]) => call_transpose(a),
        ("trunc", None, [a]) => call_trunc(a),
        // packing
        ("pack4x8snorm", None, [a]) => call_pack4x8snorm(a),
        ("pack4x8unorm", None, [a]) => call_pack4x8unorm(a),
        ("pack4xI8", None, [a]) => call_pack4xi8(a),
        ("pack4xU8", None, [a]) => call_pack4xu8(a),
        ("pack4xI8Clamp", None, [a]) => call_pack4xi8clamp(a),
        ("pack4xU8Clamp", None, [a]) => call_pack4xu8clamp(a),
        ("pack2x16snorm", None, [a]) => call_pack2x16snorm(a),
        ("pack2x16unorm", None, [a]) => call_pack2x16unorm(a),
        ("pack2x16float", None, [a]) => call_pack2x16float(a),
        ("unpack4x8snorm", None, [a]) => call_unpack4x8snorm(a),
        ("unpack4x8unorm", None, [a]) => call_unpack4x8unorm(a),
        ("unpack4xI8", None, [a]) => call_unpack4xi8(a),
        ("unpack4xU8", None, [a]) => call_unpack4xu8(a),
        ("unpack2x16snorm", None, [a]) => call_unpack2x16snorm(a),
        ("unpack2x16unorm", None, [a]) => call_unpack2x16unorm(a),
        ("unpack2x16float", None, [a]) => call_unpack2x16float(a),

        _ => Err(E::Signature(ty.clone(), args.iter().map(Ty::ty).collect())),
    }
}

// -----------
// ZERO VALUES
// -----------
// reference: <https://www.w3.org/TR/WGSL/#zero-value>

impl Instance {
    /// zero-value initialize an instance of a given type.
    pub fn zero_value(ty: &Type, ctx: &mut Context) -> Result<Self, E> {
        match ty {
            Type::Bool => Ok(LiteralInstance::Bool(false).into()),
            Type::AbstractInt => Ok(LiteralInstance::AbstractInt(0).into()),
            Type::AbstractFloat => Ok(LiteralInstance::AbstractFloat(0.0).into()),
            Type::I32 => Ok(LiteralInstance::I32(0).into()),
            Type::U32 => Ok(LiteralInstance::U32(0).into()),
            Type::F32 => Ok(LiteralInstance::F32(0.0).into()),
            Type::F16 => Ok(LiteralInstance::F16(f16::zero()).into()),
            #[cfg(feature = "naga_ext")]
            Type::I64 => Ok(LiteralInstance::I64(0).into()),
            #[cfg(feature = "naga_ext")]
            Type::U64 => Ok(LiteralInstance::U64(0).into()),
            #[cfg(feature = "naga_ext")]
            Type::F64 => Ok(LiteralInstance::F64(0.0).into()),
            Type::Struct(name) => StructInstance::zero_value(name, ctx).map(Into::into),
            Type::Array(a_ty, Some(n)) => ArrayInstance::zero_value(*n, a_ty, ctx).map(Into::into),
            Type::Array(_, None) => Err(E::NotConstructible(ty.clone())),
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(_, _) => Err(E::NotConstructible(ty.clone())),
            Type::Vec(n, v_ty) => VecInstance::zero_value(*n, v_ty).map(Into::into),
            Type::Mat(c, r, m_ty) => MatInstance::zero_value(*c, *r, m_ty).map(Into::into),
            Type::Atomic(_) | Type::Ptr(_, _) | Type::Texture(_) | Type::Sampler(_) => {
                Err(E::NotConstructible(ty.clone()))
            }
        }
    }
}

impl LiteralInstance {
    pub fn zero_value(ty: &Type) -> Result<Self, E> {
        match ty {
            Type::Bool => Ok(LiteralInstance::Bool(false)),
            Type::AbstractInt => Ok(LiteralInstance::AbstractInt(0)),
            Type::AbstractFloat => Ok(LiteralInstance::AbstractFloat(0.0)),
            Type::I32 => Ok(LiteralInstance::I32(0)),
            Type::U32 => Ok(LiteralInstance::U32(0)),
            Type::F32 => Ok(LiteralInstance::F32(0.0)),
            Type::F16 => Ok(LiteralInstance::F16(f16::zero())),
            #[cfg(feature = "naga_ext")]
            Type::I64 => Ok(LiteralInstance::I64(0)),
            #[cfg(feature = "naga_ext")]
            Type::U64 => Ok(LiteralInstance::U64(0)),
            #[cfg(feature = "naga_ext")]
            Type::F64 => Ok(LiteralInstance::F64(0.0)),
            _ => Err(E::NotScalar(ty.clone())),
        }
    }
}

impl StructInstance {
    /// zero-value initialize a struct instance.
    pub fn zero_value(name: &str, ctx: &mut Context) -> Result<Self, E> {
        let decl = ctx
            .source
            .decl_struct(name)
            .expect("struct declaration not found");

        let members = decl
            .members
            .iter()
            .map(|m| {
                let ty = ty_eval_ty(&m.ty, ctx)?;
                let val = Instance::zero_value(&ty, ctx)?;
                Ok((m.ident.to_string(), val))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(StructInstance::new(name.to_string(), members))
    }
}

impl ArrayInstance {
    /// zero-value initialize an array instance.
    pub fn zero_value(n: usize, ty: &Type, ctx: &mut Context) -> Result<Self, E> {
        let zero = Instance::zero_value(ty, ctx)?;
        let comps = (0..n).map(|_| zero.clone()).collect_vec();
        Ok(ArrayInstance::new(comps, false))
    }
}

impl VecInstance {
    /// zero-value initialize a vec instance.
    pub fn zero_value(n: u8, ty: &Type) -> Result<Self, E> {
        let zero = Instance::Literal(LiteralInstance::zero_value(ty)?);
        let comps = (0..n).map(|_| zero.clone()).collect_vec();
        Ok(VecInstance::new(comps))
    }
}

impl MatInstance {
    /// zero-value initialize a mat instance.
    pub fn zero_value(c: u8, r: u8, ty: &Type) -> Result<Self, E> {
        let zero = Instance::Literal(LiteralInstance::zero_value(ty)?);
        let zero_col = Instance::Vec(VecInstance::new((0..r).map(|_| zero.clone()).collect_vec()));
        let comps = (0..c).map(|_| zero_col.clone()).collect_vec();
        Ok(MatInstance::from_cols(comps))
    }
}

// ------------
// CONSTRUCTORS
// ------------
// reference: <https://www.w3.org/TR/WGSL/#constructor-builtin-function>

pub struct ArrayTemplate {
    n: Option<usize>,
    ty: Type,
}
impl ArrayTemplate {
    pub fn parse(tplt: &[TemplateArg], ctx: &mut Context) -> Result<ArrayTemplate, E> {
        let (t1, t2) = match tplt {
            [t1] => Ok((t1, None)),
            [t1, t2] => Ok((t1, Some(t2))),
            _ => Err(E::TemplateArgs("array")),
        }?;
        let ty = match t1.expression.node() {
            Expression::TypeOrIdentifier(ty) => ty_eval_ty(ty, ctx),
            _ => Err(E::TemplateArgs("array")),
        }?;
        if let Some(t2) = t2 {
            let n = t2.expression.eval_value(ctx)?;
            let n = match n {
                Instance::Literal(LiteralInstance::AbstractInt(n)) => (n > 0).then_some(n as usize),
                Instance::Literal(LiteralInstance::I32(n)) => (n > 0).then_some(n as usize),
                Instance::Literal(LiteralInstance::U32(n)) => (n > 0).then_some(n as usize),
                #[cfg(feature = "naga_ext")]
                Instance::Literal(LiteralInstance::I64(n)) => (n > 0).then_some(n as usize),
                #[cfg(feature = "naga_ext")]
                Instance::Literal(LiteralInstance::U64(n)) => (n > 0).then_some(n as usize),
                _ => None,
            }
            .ok_or(E::Builtin(
                "the array element count must evaluate to a `u32` or a `i32` greater than `0`",
            ))?;
            Ok(ArrayTemplate { n: Some(n), ty })
        } else {
            Ok(ArrayTemplate { n: None, ty })
        }
    }
    pub fn ty(&self) -> Type {
        Type::Array(Box::new(self.ty.clone()), self.n)
    }
    pub fn inner_ty(&self) -> Type {
        self.ty.clone()
    }
    pub fn n(&self) -> Option<usize> {
        self.n
    }
}

pub struct VecTemplate {
    ty: Type,
}
impl VecTemplate {
    pub fn parse(tplt: &[TemplateArg], ctx: &mut Context) -> Result<VecTemplate, E> {
        let ty = match tplt {
            [t1] => match t1.expression.node() {
                Expression::TypeOrIdentifier(ty) => ty_eval_ty(ty, ctx),
                _ => Err(E::TemplateArgs("vector")),
            },
            _ => Err(E::TemplateArgs("vector")),
        }?;
        if ty.is_scalar() && ty.is_concrete() {
            Ok(VecTemplate { ty })
        } else {
            Err(EvalError::Builtin("vector template type must be a scalar"))
        }
    }
    pub fn ty(&self, n: u8) -> Type {
        Type::Vec(n, self.ty.clone().into())
    }
    pub fn inner_ty(&self) -> &Type {
        &self.ty
    }
}

pub struct MatTemplate {
    ty: Type,
}

impl MatTemplate {
    pub fn parse(tplt: &[TemplateArg], ctx: &mut Context) -> Result<MatTemplate, E> {
        let ty = match tplt {
            [t1] => match t1.expression.node() {
                Expression::TypeOrIdentifier(ty) => ty_eval_ty(ty, ctx),
                _ => Err(E::TemplateArgs("matrix")),
            },
            _ => Err(E::TemplateArgs("matrix")),
        }?;
        if ty.is_f_32() || ty.is_f_16() {
            Ok(MatTemplate { ty })
        } else {
            Err(EvalError::Builtin(
                "matrix template type must be f32 or f16",
            ))
        }
    }

    pub fn ty(&self, c: u8, r: u8) -> Type {
        Type::Mat(c, r, self.ty.clone().into())
    }

    pub fn inner_ty(&self) -> &Type {
        &self.ty
    }
}

pub struct PtrTemplate {
    pub space: AddressSpace,
    pub ty: Type,
    pub access: AccessMode,
}
impl PtrTemplate {
    pub fn parse(tplt: &[TemplateArg], ctx: &mut Context) -> Result<PtrTemplate, E> {
        let mut it = tplt.iter().map(|t| t.expression.node());
        match (it.next(), it.next(), it.next(), it.next()) {
            (
                Some(Expression::TypeOrIdentifier(TypeExpression {
                    path: None,
                    ident: e1,
                    template_args: None,
                })),
                Some(Expression::TypeOrIdentifier(e2)),
                e3,
                None,
            ) => {
                let mut space = e1
                    .name()
                    .parse()
                    .map_err(|()| EvalError::Builtin("invalid pointer storage space"))?;
                let ty = ty_eval_ty(e2, ctx)?;
                if !ty.is_storable() {
                    return Err(EvalError::Builtin("pointer type must be storable"));
                }
                let access = if let Some(e3) = e3 {
                    match e3 {
                        Expression::TypeOrIdentifier(TypeExpression {
                            path: None,
                            ident,
                            template_args: None,
                        }) => Some(
                            ident
                                .name()
                                .parse()
                                .map_err(|()| EvalError::Builtin("invalid pointer access mode"))?,
                        ),
                        _ => Err(EvalError::Builtin("invalid pointer access mode"))?,
                    }
                } else {
                    None
                };
                // selecting the default access mode per address space.
                // reference: <https://www.w3.org/TR/WGSL/#address-space>
                let access = match (&mut space, access) {
                    (AddressSpace::Function, Some(access))
                    | (AddressSpace::Private, Some(access))
                    | (AddressSpace::Workgroup, Some(access)) => access,
                    (AddressSpace::Function, None)
                    | (AddressSpace::Private, None)
                    | (AddressSpace::Workgroup, None) => AccessMode::ReadWrite,
                    (AddressSpace::Uniform, Some(AccessMode::Read) | None) => AccessMode::Read,
                    (AddressSpace::Uniform, _) => {
                        return Err(EvalError::Builtin(
                            "pointer in uniform address space must have a `read` access mode",
                        ));
                    }
                    (AddressSpace::Storage(a1), Some(a2)) => {
                        *a1 = Some(a2);
                        a2
                    }
                    (AddressSpace::Storage(None), None) => AccessMode::Read,
                    (AddressSpace::Storage(_), _) => unreachable!(),
                    (AddressSpace::Handle, _) => {
                        unreachable!("handle address space cannot be spelled")
                    }
                    #[cfg(feature = "naga_ext")]
                    (AddressSpace::PushConstant, _) => {
                        todo!("push_constant")
                    }
                };
                Ok(PtrTemplate { space, ty, access })
            }
            _ => Err(E::TemplateArgs("pointer")),
        }
    }

    pub fn ty(&self) -> Type {
        Type::Ptr(self.space, self.ty.clone().into())
    }
}

pub struct AtomicTemplate {
    pub ty: Type,
}
impl AtomicTemplate {
    pub fn parse(tplt: &[TemplateArg], ctx: &mut Context) -> Result<AtomicTemplate, E> {
        let ty = match tplt {
            [t1] => match t1.expression.node() {
                Expression::TypeOrIdentifier(ty) => ty_eval_ty(ty, ctx),
                _ => Err(E::TemplateArgs("atomic")),
            },
            _ => Err(E::TemplateArgs("atomic")),
        }?;
        if ty.is_i_32() || ty.is_u_32() {
            Ok(AtomicTemplate { ty })
        } else {
            Err(EvalError::Builtin(
                "atomic template type must be i32 or u32",
            ))
        }
    }
    pub fn ty(&self) -> Type {
        Type::Atomic(self.ty.clone().into())
    }
    pub fn inner_ty(&self) -> Type {
        self.ty.clone()
    }
}

pub struct TextureTemplate {
    ty: TextureType,
}
impl TextureTemplate {
    pub fn parse(name: &str, tplt: &[TemplateArg]) -> Result<TextureTemplate, E> {
        let ty = match name {
            "texture_1d" => TextureType::Sampled1D(Self::sampled_type(tplt)?),
            "texture_2d" => TextureType::Sampled2D(Self::sampled_type(tplt)?),
            "texture_2d_array" => TextureType::Sampled2DArray(Self::sampled_type(tplt)?),
            "texture_3d" => TextureType::Sampled3D(Self::sampled_type(tplt)?),
            "texture_cube" => TextureType::SampledCube(Self::sampled_type(tplt)?),
            "texture_cube_array" => TextureType::SampledCubeArray(Self::sampled_type(tplt)?),
            "texture_multisampled_2d" => TextureType::Multisampled2D(Self::sampled_type(tplt)?),
            "texture_storage_1d" => {
                let (tex, acc) = Self::texel_access(tplt)?;
                TextureType::Storage1D(tex, acc)
            }
            "texture_storage_2d" => {
                let (tex, acc) = Self::texel_access(tplt)?;
                TextureType::Storage2D(tex, acc)
            }
            "texture_storage_2d_array" => {
                let (tex, acc) = Self::texel_access(tplt)?;
                TextureType::Storage2DArray(tex, acc)
            }
            "texture_storage_3d" => {
                let (tex, acc) = Self::texel_access(tplt)?;
                TextureType::Storage3D(tex, acc)
            }
            _ => return Err(E::Builtin("not a templated texture type")),
        };
        Ok(Self { ty })
    }
    fn sampled_type(tplt: &[TemplateArg]) -> Result<SampledType, E> {
        match tplt {
            [t1] => match t1.expression.node() {
                Expression::TypeOrIdentifier(ty) if ty.template_args.is_none() => {
                    ty.ident.name().parse().map_err(|()| {
                        EvalError::Builtin("invalid sampled type, expected `i32`, `u32` of `f32`")
                    })
                }
                _ => Err(EvalError::Builtin(
                    "invalid sampled type, expected `i32`, `u32` of `f32`",
                )),
            },
            _ => Err(EvalError::Builtin(
                "sampled texture types take a single template parameter",
            )),
        }
    }
    fn texel_access(tplt: &[TemplateArg]) -> Result<(TexelFormat, AccessMode), E> {
        match tplt {
            [t1, t2] => {
                let texel = match t1.expression.node() {
                    Expression::TypeOrIdentifier(ty) if ty.template_args.is_none() => ty
                        .ident
                        .name()
                        .parse()
                        .map_err(|()| EvalError::Builtin("invalid texel format")),
                    _ => Err(EvalError::Builtin("invalid texel format")),
                }?;
                let access = match t2.expression.node() {
                    Expression::TypeOrIdentifier(ty) if ty.template_args.is_none() => ty
                        .ident
                        .name()
                        .parse()
                        .map_err(|()| EvalError::Builtin("invalid access mode")),
                    _ => Err(EvalError::Builtin("invalid access mode")),
                }?;
                Ok((texel, access))
            }
            _ => Err(EvalError::Builtin(
                "storage texture types take two template parameters",
            )),
        }
    }
    pub fn ty(&self) -> TextureType {
        self.ty.clone()
    }
}

pub struct BitcastTemplate {
    ty: Type,
}
impl BitcastTemplate {
    pub fn parse(tplt: &[TemplateArg], ctx: &mut Context) -> Result<BitcastTemplate, E> {
        let ty = match tplt {
            [t1] => match t1.expression.node() {
                Expression::TypeOrIdentifier(ty) => ty_eval_ty(ty, ctx),
                _ => Err(E::TemplateArgs("bitcast")),
            },
            _ => Err(E::TemplateArgs("bitcast")),
        }?;
        if ty.is_numeric() || ty.is_vec() && ty.inner_ty().is_numeric() {
            Ok(BitcastTemplate { ty })
        } else {
            Err(EvalError::Builtin(
                "bitcast template type must be a numeric scalar or numeric vector",
            ))
        }
    }
    pub fn ty(&self) -> Type {
        self.ty.clone()
    }
    pub fn inner_ty(&self) -> Type {
        self.ty.inner_ty()
    }
}

fn call_array_t(tplt: ArrayTemplate, args: &[Instance]) -> Result<Instance, E> {
    let args = args
        .iter()
        .map(|a| {
            a.convert_to(&tplt.ty)
                .ok_or_else(|| E::ParamType(tplt.ty.clone(), a.ty()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if Some(args.len()) != tplt.n {
        return Err(E::ParamCount(
            "array".to_string(),
            tplt.n.unwrap_or_default(),
            args.len(),
        ));
    }

    Ok(ArrayInstance::new(args, false).into())
}
fn call_array(args: &[Instance]) -> Result<Instance, E> {
    let args = convert_all(args).ok_or(E::Builtin("array elements are incompatible"))?;

    if args.is_empty() {
        return Err(E::Builtin("array constructor expects at least 1 argument"));
    }

    Ok(ArrayInstance::new(args, false).into())
}

fn call_bool_1(a1: &Instance) -> Result<Instance, E> {
    match a1 {
        Instance::Literal(l) => {
            let zero = LiteralInstance::zero_value(&l.ty())?;
            Ok(LiteralInstance::Bool(*l != zero).into())
        }
        _ => Err(E::Builtin("bool constructor expects a scalar argument")),
    }
}

// TODO: check that "If T is a floating point type, e is converted to i32, rounding towards zero."
fn call_i32_1(a1: &Instance) -> Result<Instance, E> {
    match a1 {
        Instance::Literal(l) => {
            let val = match l {
                LiteralInstance::Bool(n) => Some(n.then_some(1).unwrap_or(0)),
                LiteralInstance::AbstractInt(n) => n.to_i32(), // identity if representable
                LiteralInstance::AbstractFloat(n) => Some(*n as i32), // rounding towards 0
                LiteralInstance::I32(n) => Some(*n),           // identity operation
                LiteralInstance::U32(n) => Some(*n as i32),    // reinterpretation of bits
                LiteralInstance::F32(n) => Some(*n as i32),    // rounding towards 0
                LiteralInstance::F16(n) => Some(f16::to_f32(*n) as i32), // rounding towards 0
                #[cfg(feature = "naga_ext")]
                LiteralInstance::I64(n) => n.to_i32(), // identity if representable
                #[cfg(feature = "naga_ext")]
                LiteralInstance::U64(n) => n.to_i32(), // identity if representable
                #[cfg(feature = "naga_ext")]
                LiteralInstance::F64(n) => Some(*n as i32), // rounding towards 0
            }
            .ok_or(E::ConvOverflow(*l, Type::I32))?;
            Ok(LiteralInstance::I32(val).into())
        }
        _ => Err(E::Builtin("i32 constructor expects a scalar argument")),
    }
}

fn call_u32_1(a1: &Instance) -> Result<Instance, E> {
    match a1 {
        Instance::Literal(l) => {
            let val = match l {
                LiteralInstance::Bool(n) => Some(n.then_some(1).unwrap_or(0)),
                LiteralInstance::AbstractInt(n) => n.to_u32(), // identity if representable
                LiteralInstance::AbstractFloat(n) => Some(*n as u32), // rounding towards 0
                LiteralInstance::I32(n) => Some(*n as u32),    // reinterpretation of bits
                LiteralInstance::U32(n) => Some(*n),           // identity operation
                LiteralInstance::F32(n) => Some(*n as u32),    // rounding towards 0
                LiteralInstance::F16(n) => Some(f16::to_f32(*n) as u32), // rounding towards 0
                #[cfg(feature = "naga_ext")]
                LiteralInstance::I64(n) => n.to_u32(), // identity if representable
                #[cfg(feature = "naga_ext")]
                LiteralInstance::U64(n) => n.to_u32(), // identity if representable
                #[cfg(feature = "naga_ext")]
                LiteralInstance::F64(n) => Some(*n as u32), // rounding towards 0
            }
            .ok_or(E::ConvOverflow(*l, Type::U32))?;
            Ok(LiteralInstance::U32(val).into())
        }
        _ => Err(E::Builtin("u32 constructor expects a scalar argument")),
    }
}

/// see [`LiteralInstance::convert_to`]
/// "If T is a numeric scalar (other than f32), e is converted to f32 (including invalid conversions)."
/// TODO: implicit conversions are incorrect, I think. I'm not sure if f32(too_big) is correct.
fn call_f32_1(a1: &Instance, _stage: EvalStage) -> Result<Instance, E> {
    match a1 {
        Instance::Literal(l) => {
            let val = match l {
                LiteralInstance::Bool(n) => Some(n.then_some(f32::one()).unwrap_or(f32::zero())),
                LiteralInstance::AbstractInt(n) => n.to_f32(), // implicit conversion
                LiteralInstance::AbstractFloat(n) => n.to_f32(), // implicit conversion
                LiteralInstance::I32(n) => Some(*n as f32),    // scalar to float (never overflows)
                LiteralInstance::U32(n) => Some(*n as f32),    // scalar to float (never overflows)
                LiteralInstance::F32(n) => Some(*n),           // identity operation
                LiteralInstance::F16(n) => Some(f16::to_f32(*n)), // exactly representable
                #[cfg(feature = "naga_ext")]
                LiteralInstance::I64(n) => n.to_f32(), // implicit conversion
                #[cfg(feature = "naga_ext")]
                LiteralInstance::U64(n) => n.to_f32(), // implicit conversion
                #[cfg(feature = "naga_ext")]
                LiteralInstance::F64(n) => n.to_f32(), // implicit conversion
            }
            .ok_or(E::ConvOverflow(*l, Type::F32))?;
            Ok(LiteralInstance::F32(val).into())
        }
        _ => Err(E::Builtin("f32 constructor expects a scalar argument")),
    }
}

/// see [`LiteralInstance::convert_to`]
/// "If T is a numeric scalar (other than f16), e is converted to f16 (including invalid conversions)."
fn call_f16_1(a1: &Instance, stage: EvalStage) -> Result<Instance, E> {
    match a1 {
        Instance::Literal(l) => {
            let val = match l {
                LiteralInstance::Bool(n) => Some(n.then_some(f16::one()).unwrap_or(f16::zero())),
                LiteralInstance::AbstractInt(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        let range = -65504..=65504;
                        range.contains(n).then_some(f16::from_f32(*n as f32))
                    } else {
                        Some(f16::from_f32(*n as f32))
                    }
                }
                LiteralInstance::AbstractFloat(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        let range = -65504.0..=65504.0;
                        range.contains(n).then_some(f16::from_f32(*n as f32))
                    } else {
                        Some(f16::from_f32(*n as f32))
                    }
                }
                LiteralInstance::I32(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        f16::from_i32(*n)
                    } else {
                        Some(f16::from_f32(*n as f32))
                    }
                }
                LiteralInstance::U32(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        f16::from_u32(*n)
                    } else {
                        Some(f16::from_f32(*n as f32))
                    }
                }
                LiteralInstance::F32(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        let range = -65504.0..=65504.0;
                        range.contains(n).then_some(f16::from_f32(*n))
                    } else {
                        Some(f16::from_f32(*n))
                    }
                }
                LiteralInstance::F16(n) => Some(*n), // identity operation
                #[cfg(feature = "naga_ext")]
                LiteralInstance::I64(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        let range = -65504..=65504;
                        range.contains(n).then_some(f16::from_f32(*n as f32))
                    } else {
                        Some(f16::from_f32(*n as f32))
                    }
                }
                #[cfg(feature = "naga_ext")]
                LiteralInstance::U64(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        f16::from_u64(*n)
                    } else {
                        Some(f16::from_f32(*n as f32))
                    }
                }
                #[cfg(feature = "naga_ext")]
                LiteralInstance::F64(n) => {
                    // scalar to float (can overflow)
                    if stage == EvalStage::Const {
                        let range = -65504.0..=65504.0;
                        range.contains(n).then_some(f16::from_f32(*n as f32))
                    } else {
                        Some(f16::from_f32(*n as f32))
                    }
                }
            }
            .ok_or(E::ConvOverflow(*l, Type::F16))?;
            Ok(LiteralInstance::F16(val).into())
        }
        _ => Err(E::Builtin("f16 constructor expects a scalar argument")),
    }
}

fn call_mat_t(
    c: usize,
    r: usize,
    tplt: MatTemplate,
    args: &[Instance],
    stage: EvalStage,
) -> Result<Instance, E> {
    // overload 1: mat conversion constructor
    if let [Instance::Mat(m)] = args {
        if m.c() != c || m.r() != r {
            return Err(E::Conversion(m.ty(), tplt.ty(c as u8, r as u8)));
        }

        let conv_fn = match tplt.inner_ty() {
            Type::F32 => call_f32_1,
            Type::F16 => call_f16_1,
            _ => return Err(E::Builtin("matrix type must be a f32 or f16")),
        };

        let comps = m
            .iter_cols()
            .map(|v| {
                v.unwrap_vec_ref()
                    .iter()
                    .map(|n| conv_fn(n, stage))
                    .collect::<Result<Vec<_>, _>>()
                    .map(|s| Instance::Vec(VecInstance::new(s)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(MatInstance::from_cols(comps).into())
    } else {
        let ty = args
            .first()
            .ok_or(E::Builtin("matrix constructor expects arguments"))?
            .ty();
        let ty = ty
            .convert_inner_to(tplt.inner_ty())
            .ok_or(E::Conversion(ty.inner_ty(), tplt.inner_ty().clone()))?;
        let args =
            convert_all_to(args, &ty).ok_or(E::Builtin("matrix components are incompatible"))?;

        // overload 2: mat from column vectors
        if ty.is_vec() {
            if args.len() != c {
                return Err(E::ParamCount(format!("mat{c}x{r}"), c, args.len()));
            }

            Ok(MatInstance::from_cols(args).into())
        }
        // overload 3: mat from float values
        else if ty.is_float() {
            if args.len() != c * r {
                return Err(E::ParamCount(format!("mat{c}x{r}"), c * r, args.len()));
            }

            let args = args
                .chunks(r)
                .map(|v| Instance::Vec(VecInstance::new(v.to_vec())))
                .collect_vec();

            Ok(MatInstance::from_cols(args).into())
        } else {
            return Err(E::Builtin(
                "matrix constructor expects float or vector of float arguments",
            ));
        }
    }
}

fn call_mat(c: usize, r: usize, args: &[Instance]) -> Result<Instance, E> {
    // overload 1: mat conversion constructor
    if let [Instance::Mat(m)] = args {
        if m.c() != c || m.r() != r {
            let ty2 = Type::Mat(c as u8, r as u8, m.inner_ty().into());
            return Err(E::Conversion(m.ty(), ty2));
        }
        // note: `matCxR(e: matCxR<S>) -> matCxR<S>` is no-op
        Ok(m.clone().into())
    } else {
        let tys = args.iter().map(|a| a.ty()).collect_vec();
        let ty = convert_all_ty(&tys).ok_or(E::Builtin("matrix components are incompatible"))?;
        let mut inner_ty = ty.inner_ty();

        if inner_ty.is_abstract_int() {
            // force conversion from AbstractInt to a float type
            inner_ty = Type::AbstractInt;
        } else if !inner_ty.is_float() {
            return Err(E::Builtin(
                "matrix constructor expects float or vector of float arguments",
            ));
        }

        let args = convert_all_inner_to(args, &inner_ty)
            .ok_or(E::Builtin("matrix components are incompatible"))?;

        // overload 2: mat from column vectors
        if ty.is_vec() {
            if args.len() != c {
                return Err(E::ParamCount(format!("mat{c}x{r}"), c, args.len()));
            }

            Ok(MatInstance::from_cols(args).into())
        }
        // overload 3: mat from float values
        else if ty.is_float() || ty.is_abstract_int() {
            if args.len() != c * r {
                return Err(E::ParamCount(format!("mat{c}x{r}"), c * r, args.len()));
            }
            let args = args
                .chunks(r)
                .map(|v| Instance::Vec(VecInstance::new(v.to_vec())))
                .collect_vec();

            Ok(MatInstance::from_cols(args).into())
        } else {
            return Err(E::Builtin(
                "matrix constructor expects float or vector of float arguments",
            ));
        }
    }
}

fn call_vec_t(
    n: usize,
    tplt: VecTemplate,
    args: &[Instance],
    stage: EvalStage,
) -> Result<Instance, E> {
    // overload 1: vec init from single scalar value
    if let [Instance::Literal(l)] = args {
        let val = l
            .convert_to(tplt.inner_ty())
            .map(Instance::Literal)
            .ok_or_else(|| E::ParamType(tplt.inner_ty().clone(), l.ty()))?;
        let comps = (0..n).map(|_| val.clone()).collect_vec();
        Ok(VecInstance::new(comps).into())
    }
    // overload 2: vec conversion constructor
    else if let [Instance::Vec(v)] = args {
        let ty = tplt.ty(n as u8);
        if v.n() != n {
            return Err(E::Conversion(v.ty(), ty));
        }

        let conv_fn = match ty.inner_ty() {
            Type::Bool => |n, _| call_bool_1(n),
            Type::I32 => |n, _| call_i32_1(n),
            Type::U32 => |n, _| call_u32_1(n),
            Type::F32 => |n, stage| call_f32_1(n, stage),
            Type::F16 => |n, stage| call_f16_1(n, stage),
            _ => return Err(E::Builtin("vector type must be a scalar")),
        };

        let comps = v
            .iter()
            .map(|n| conv_fn(n, stage))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(VecInstance::new(comps).into())
    }
    // overload 3: vec init from component values
    else {
        // flatten vecN args
        let args = args
            .iter()
            .flat_map(|a| -> Box<dyn Iterator<Item = &Instance>> {
                match a {
                    Instance::Vec(v) => Box::new(v.iter()),
                    _ => Box::new(std::iter::once(a)),
                }
            })
            .collect_vec();
        if args.len() != n {
            return Err(E::ParamCount(format!("vec{n}"), n, args.len()));
        }

        let comps = args
            .iter()
            .map(|a| {
                a.convert_inner_to(tplt.inner_ty())
                    .ok_or_else(|| E::ParamType(tplt.inner_ty().clone(), a.ty()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(VecInstance::new(comps).into())
    }
}

fn call_vec(n: usize, args: &[Instance]) -> Result<Instance, E> {
    // overload 1: vec init from single scalar value
    if let [Instance::Literal(l)] = args {
        let val = Instance::Literal(*l);
        let comps = (0..n).map(|_| val.clone()).collect_vec();
        Ok(VecInstance::new(comps).into())
    }
    // overload 2: vec conversion constructor
    else if let [Instance::Vec(v)] = args {
        if v.n() != n {
            let ty = v.ty();
            let ty2 = Type::Vec(n as u8, ty.inner_ty().into());
            return Err(E::Conversion(ty, ty2));
        }
        // note: `vecN(e: vecN<S>) -> vecN<S>` is no-op
        Ok(v.clone().into())
    }
    // overload 3: vec init from component values
    else if !args.is_empty() {
        // flatten vecN args
        let args = args
            .iter()
            .flat_map(|a| -> Box<dyn Iterator<Item = &Instance>> {
                match a {
                    Instance::Vec(v) => Box::new(v.iter()),
                    _ => Box::new(std::iter::once(a)),
                }
            })
            .cloned()
            .collect_vec();
        if args.len() != n {
            return Err(E::ParamCount(format!("vec{n}"), n, args.len()));
        }

        let comps = convert_all(&args).ok_or(E::Builtin("vector components are incompatible"))?;

        if !comps.first().unwrap(/* SAFETY: len() checked above */).ty().is_scalar() {
            return Err(E::Builtin("vec constructor expects scalar arguments"));
        }
        Ok(VecInstance::new(comps).into())
    }
    // overload 3: zero-vec
    else {
        VecInstance::zero_value(n as u8, &Type::AbstractInt).map(Into::into)
    }
}

// -------
// BITCAST
// -------
// reference: <https://www.w3.org/TR/WGSL/#bit-reinterp-builtin-functions>

fn call_bitcast_t(tplt: BitcastTemplate, e: &Instance) -> Result<Instance, E> {
    fn lit_bytes(l: &LiteralInstance, ty: &Type) -> Result<Vec<u8>, E> {
        match l {
            LiteralInstance::Bool(_) => Err(E::Builtin("bitcast argument cannot be bool")),
            LiteralInstance::AbstractInt(n) => {
                if ty == &Type::U32 {
                    n.to_u32()
                        .map(|n| n.to_le_bytes().to_vec())
                        .ok_or(E::ConvOverflow(*l, Type::U32))
                } else {
                    n.to_i32()
                        .map(|n| n.to_le_bytes().to_vec())
                        .ok_or(E::ConvOverflow(*l, Type::I32))
                }
            }
            LiteralInstance::AbstractFloat(n) => n
                .to_f32()
                .map(|n| n.to_le_bytes().to_vec())
                .ok_or(E::ConvOverflow(*l, Type::F32)),
            LiteralInstance::I32(n) => Ok(n.to_le_bytes().to_vec()),
            LiteralInstance::U32(n) => Ok(n.to_le_bytes().to_vec()),
            LiteralInstance::F32(n) => Ok(n.to_le_bytes().to_vec()),
            LiteralInstance::F16(n) => Ok(n.to_le_bytes().to_vec()),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(n) => Ok(n.to_le_bytes().to_vec()),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(n) => Ok(n.to_le_bytes().to_vec()),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(n) => Ok(n.to_le_bytes().to_vec()),
        }
    }

    fn vec_bytes(v: &VecInstance, ty: &Type) -> Result<Vec<u8>, E> {
        v.iter()
            .map(|n| lit_bytes(n.unwrap_literal_ref(), ty))
            .reduce(|n1, n2| Ok(chain(n1?, n2?).collect_vec()))
            .unwrap()
    }

    let ty = tplt.ty();
    let inner_ty = tplt.inner_ty();

    let bytes = match e {
        Instance::Literal(l) => lit_bytes(l, &inner_ty),
        Instance::Vec(v) => vec_bytes(v, &inner_ty),
        _ => Err(E::Builtin(
            "`bitcast` expects a numeric scalar or vector argument",
        )),
    }?;

    let size_err = E::Builtin("`bitcast` input and output types must have the same size");

    match ty {
        Type::I32 => {
            let n = i32::from_le_bytes(bytes.try_into().map_err(|_| size_err)?);
            Ok(LiteralInstance::I32(n).into())
        }
        Type::U32 => {
            let n = u32::from_le_bytes(bytes.try_into().map_err(|_| size_err)?);
            Ok(LiteralInstance::U32(n).into())
        }
        Type::F32 => {
            let n = f32::from_le_bytes(bytes.try_into().map_err(|_| size_err)?);
            Ok(LiteralInstance::F32(n).into())
        }
        Type::F16 => {
            let n = f16::from_le_bytes(bytes.try_into().map_err(|_| size_err)?);
            Ok(LiteralInstance::F16(n).into())
        }
        Type::Vec(n, ty) => {
            if *ty == Type::I32 && bytes.len() == 4 * (n as usize) {
                let v = bytes
                    .chunks(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .map(|n| LiteralInstance::from(n).into())
                    .collect_vec();
                Ok(VecInstance::new(v).into())
            } else if *ty == Type::U32 && bytes.len() == 4 * (n as usize) {
                let v = bytes
                    .chunks(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .map(|n| LiteralInstance::from(n).into())
                    .collect_vec();
                Ok(VecInstance::new(v).into())
            } else if *ty == Type::F32 && bytes.len() == 4 * (n as usize) {
                let v = bytes
                    .chunks(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .map(|n| LiteralInstance::from(n).into())
                    .collect_vec();
                Ok(VecInstance::new(v).into())
            } else if *ty == Type::F16 && bytes.len() == 2 * (n as usize) {
                let v = bytes
                    .chunks(2)
                    .map(|b| f16::from_le_bytes(b.try_into().unwrap()))
                    .map(|n| LiteralInstance::from(n).into())
                    .collect_vec();
                Ok(VecInstance::new(v).into())
            } else {
                Err(size_err)
            }
        }
        _ => unreachable!("invalid `bitcast` template"),
    }
}

// -------
// LOGICAL
// -------
// reference: <https://www.w3.org/TR/WGSL/#logical-builtin-functions>

fn call_all(e: &Instance) -> Result<Instance, E> {
    match e {
        Instance::Literal(LiteralInstance::Bool(_)) => Ok(e.clone()),
        Instance::Vec(v) if v.inner_ty() == Type::Bool => {
            let b = v.iter().all(|b| b.unwrap_literal_ref().unwrap_bool());
            Ok(LiteralInstance::Bool(b).into())
        }
        _ => Err(E::Builtin(
            "`all` expects a boolean or vector of boolean argument",
        )),
    }
}

fn call_any(e: &Instance) -> Result<Instance, E> {
    match e {
        Instance::Literal(LiteralInstance::Bool(_)) => Ok(e.clone()),
        Instance::Vec(v) if v.inner_ty() == Type::Bool => {
            let b = v.iter().any(|b| b.unwrap_literal_ref().unwrap_bool());
            Ok(LiteralInstance::Bool(b).into())
        }
        _ => Err(E::Builtin(
            "`any` expects a boolean or vector of boolean argument",
        )),
    }
}

fn call_select(f: &Instance, t: &Instance, cond: &Instance) -> Result<Instance, E> {
    let (f, t) = convert(f, t).ok_or(E::Builtin(
        "`select` 1st and 2nd arguments are incompatible",
    ))?;

    match cond {
        Instance::Literal(LiteralInstance::Bool(b)) => Ok(b.then_some(t).unwrap_or(f)),
        Instance::Vec(v) if v.inner_ty() == Type::Bool => match (f, t) {
            (Instance::Vec(v1), Instance::Vec(v2)) => {
                if v1.n() != v.n() {
                    Err(E::Builtin(
                        "`select` vector arguments must have the same number of components",
                    ))
                } else {
                    let v = izip!(v1, v2, v.iter())
                        .map(|(f, t, b)| {
                            if b.unwrap_literal_ref().unwrap_bool() {
                                t.to_owned() // BUG: is it a bug in rust_analyzer? it displays f as Instance and t as &Instance
                            } else {
                                f.to_owned()
                            }
                        })
                        .collect_vec();
                    Ok(VecInstance::new(v).into())
                }
            }
            _ => Err(E::Builtin(
                "`select` arguments must be vectors when the condition is a vector",
            )),
        },
        _ => Err(E::Builtin(
            "`select` 3rd argument must be a boolean or vector of boolean",
        )),
    }
}

// -----
// ARRAY
// -----
// reference: <https://www.w3.org/TR/WGSL/#array-builtin-functions>

fn call_arraylength(p: &Instance) -> Result<Instance, E> {
    let err = E::Builtin("`arrayLength` expects a pointer to array argument");
    let r = match p {
        Instance::Ptr(p) => RefInstance::from(p.clone()),
        _ => return Err(err),
    };
    let r = r.read()?;
    match &*r {
        Instance::Array(a) => Ok(LiteralInstance::U32(a.n() as u32).into()),
        _ => Err(err),
    }
}

// -------
// NUMERIC
// -------
// reference: <https://www.w3.org/TR/WGSL/#numeric-builtin-function>

macro_rules! impl_call_float_unary {
    ($name:literal, $e:ident, $n:ident => $expr:expr) => {{
        const ERR: E = E::Builtin(concat!(
            "`",
            $name,
            "` expects a float or vector of float argument"
        ));
        fn lit_fn(l: &LiteralInstance) -> Result<LiteralInstance, E> {
            match l {
                LiteralInstance::Bool(_) => Err(ERR),
                LiteralInstance::AbstractInt(_) => {
                    let $n = l
                        .convert_to(&Type::AbstractFloat)
                        .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?
                        .unwrap_abstract_float();
                    Ok(LiteralInstance::from($expr))
                }
                LiteralInstance::AbstractFloat($n) => Ok(LiteralInstance::from($expr)),
                LiteralInstance::I32(_) => Err(ERR),
                LiteralInstance::U32(_) => Err(ERR),
                LiteralInstance::F32($n) => Ok(LiteralInstance::from($expr)),
                LiteralInstance::F16($n) => Ok(LiteralInstance::from($expr)),
                #[cfg(feature = "naga_ext")]
                LiteralInstance::I64(_) => Err(ERR),
                #[cfg(feature = "naga_ext")]
                LiteralInstance::U64(_) => Err(ERR),
                #[cfg(feature = "naga_ext")]
                LiteralInstance::F64($n) => Ok(LiteralInstance::F64($expr)),
            }
        }
        match $e {
            Instance::Literal(l) => lit_fn(l).map(Into::into),
            Instance::Vec(v) => v.compwise_unary(lit_fn).map(Into::into),
            _ => Err(ERR),
        }
    }};
}

// TODO: checked_abs
fn call_abs(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`abs` expects a scalar or vector of scalar argument");
    fn lit_abs(l: &LiteralInstance) -> Result<LiteralInstance, E> {
        match l {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(n) => Ok(LiteralInstance::from(n.wrapping_abs())),
            LiteralInstance::AbstractFloat(n) => Ok(LiteralInstance::from(n.abs())),
            LiteralInstance::I32(n) => Ok(LiteralInstance::from(n.wrapping_abs())),
            LiteralInstance::U32(_) => Ok(*l),
            LiteralInstance::F32(n) => Ok(LiteralInstance::from(n.abs())),
            LiteralInstance::F16(n) => Ok(LiteralInstance::from(n.abs())),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(n) => Ok(LiteralInstance::I64(n.wrapping_abs())),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => Ok(*l),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(n) => Ok(LiteralInstance::F64(n.abs())),
        }
    }
    match e {
        Instance::Literal(l) => lit_abs(l).map(Into::into),
        Instance::Vec(v) => v.compwise_unary(lit_abs).map(Into::into),
        _ => Err(ERR),
    }
}

// NOTE: the function returns NaN as an `indeterminate value` if computed out of domain
fn call_acos(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("acos", e, n => n.acos())
}

// NOTE: the function returns NaN as an `indeterminate value` if computed out of domain
fn call_acosh(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("acosh", e, n => n.acosh())
}

// NOTE: the function returns NaN as an `indeterminate value` if computed out of domain
fn call_asin(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("asin", e, n => n.asin())
}

fn call_asinh(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("asinh", e, n => n.asinh())
}

fn call_atan(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("atan", e, n => n.atan())
}

// NOTE: the function returns NaN as an `indeterminate value` if computed out of domain
fn call_atanh(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("atanh", e, n => n.atanh())
}

fn call_atan2(y: &Instance, x: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`atan2` expects a float or vector of float argument");
    fn lit_atan2(y: &LiteralInstance, x: &LiteralInstance) -> Result<LiteralInstance, E> {
        match y {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(_) => {
                let y = y
                    .convert_to(&Type::AbstractFloat)
                    .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?;
                let x = x
                    .convert_to(&Type::AbstractFloat)
                    .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?;
                Ok(LiteralInstance::from(
                    y.unwrap_abstract_float().atan2(x.unwrap_abstract_float()),
                ))
            }
            LiteralInstance::AbstractFloat(y) => {
                Ok(LiteralInstance::from(y.atan2(x.unwrap_abstract_float())))
            }
            LiteralInstance::I32(_) => Err(ERR),
            LiteralInstance::U32(_) => Err(ERR),
            LiteralInstance::F32(y) => Ok(LiteralInstance::from(y.atan2(x.unwrap_f_32()))),
            LiteralInstance::F16(y) => Ok(LiteralInstance::from(y.atan2(x.unwrap_f_16()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(y) => Ok(LiteralInstance::F64(y.atan2(x.unwrap_f_64()))),
        }
    }
    let (y, x) = convert(y, x).ok_or(E::Builtin("`atan2` arguments are incompatible"))?;
    match (y, x) {
        (Instance::Literal(y), Instance::Literal(x)) => lit_atan2(&y, &x).map(Into::into),
        (Instance::Vec(y), Instance::Vec(x)) => y.compwise_binary(&x, lit_atan2).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_ceil(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("ceil", e, n => n.ceil())
}

fn call_clamp(e: &Instance, low: &Instance, high: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`clamp` arguments are incompatible");
    let tys = [e.ty(), low.ty(), high.ty()];
    let ty = convert_all_ty(&tys).ok_or(ERR)?;
    let e = e.convert_to(ty).ok_or(ERR)?;
    let low = low.convert_to(ty).ok_or(ERR)?;
    let high = high.convert_to(ty).ok_or(ERR)?;
    call_min(&call_max(&e, &low)?, &high)
}

// NOTE: the function returns NaN as an `indeterminate value` if computed out of domain
fn call_cos(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("cos", e, n => n.cos())
}

fn call_cosh(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("cosh", e, n => n.cosh())
}

fn call_countleadingzeros(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`countLeadingZeros` expects a float or vector of float argument");
    fn lit_leading_zeros(l: &LiteralInstance) -> Result<LiteralInstance, E> {
        match l {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(n) => {
                Ok(LiteralInstance::AbstractInt(n.leading_zeros() as i64))
            }
            LiteralInstance::AbstractFloat(_) => Err(ERR),
            LiteralInstance::I32(n) => Ok(LiteralInstance::I32(n.leading_zeros() as i32)),
            LiteralInstance::U32(n) => Ok(LiteralInstance::U32(n.leading_zeros())),
            LiteralInstance::F32(_) => Err(ERR),
            LiteralInstance::F16(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(n) => Ok(LiteralInstance::I64(n.leading_zeros() as i64)),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(n) => Ok(LiteralInstance::U64(n.leading_zeros() as u64)),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(_) => Err(ERR),
        }
    }
    match e {
        Instance::Literal(l) => lit_leading_zeros(l).map(Into::into),
        Instance::Vec(v) => v.compwise_unary(lit_leading_zeros).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_countonebits(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`countOneBits` expects a float or vector of float argument");
    fn lit_count_ones(l: &LiteralInstance) -> Result<LiteralInstance, E> {
        match l {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(n) => {
                Ok(LiteralInstance::AbstractInt(n.count_ones() as i64))
            }
            LiteralInstance::AbstractFloat(_) => Err(ERR),
            LiteralInstance::I32(n) => Ok(LiteralInstance::I32(n.count_ones() as i32)),
            LiteralInstance::U32(n) => Ok(LiteralInstance::U32(n.count_ones())),
            LiteralInstance::F32(_) => Err(ERR),
            LiteralInstance::F16(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(n) => Ok(LiteralInstance::I64(n.count_ones() as i64)),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(n) => Ok(LiteralInstance::U64(n.count_ones() as u64)),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(_) => Err(ERR),
        }
    }
    match e {
        Instance::Literal(l) => lit_count_ones(l).map(Into::into),
        Instance::Vec(v) => v.compwise_unary(lit_count_ones).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_counttrailingzeros(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`countTrailingZeros` expects a float or vector of float argument");
    fn lit_trailing_zeros(l: &LiteralInstance) -> Result<LiteralInstance, E> {
        match l {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(n) => {
                Ok(LiteralInstance::AbstractInt(n.trailing_zeros() as i64))
            }
            LiteralInstance::AbstractFloat(_) => Err(ERR),
            LiteralInstance::I32(n) => Ok(LiteralInstance::I32(n.trailing_zeros() as i32)),
            LiteralInstance::U32(n) => Ok(LiteralInstance::U32(n.trailing_zeros())),
            LiteralInstance::F32(_) => Err(ERR),
            LiteralInstance::F16(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(n) => Ok(LiteralInstance::I64(n.trailing_zeros() as i64)),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(n) => Ok(LiteralInstance::U64(n.trailing_zeros() as u64)),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(_) => Err(ERR),
        }
    }
    match e {
        Instance::Literal(l) => lit_trailing_zeros(l).map(Into::into),
        Instance::Vec(v) => v.compwise_unary(lit_trailing_zeros).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_cross(a: &Instance, b: &Instance, stage: EvalStage) -> Result<Instance, E> {
    let (a, b) = convert(a, b).ok_or(E::Builtin("`cross` arguments are incompatible"))?;
    match (a, b) {
        (Instance::Vec(a), Instance::Vec(b)) if a.n() == 3 => {
            let s1 = a[1]
                .op_mul(&b[2], stage)?
                .op_sub(&a[2].op_mul(&b[1], stage)?, stage)?;
            let s2 = a[2]
                .op_mul(&b[0], stage)?
                .op_sub(&a[0].op_mul(&b[2], stage)?, stage)?;
            let s3 = a[0]
                .op_mul(&b[1], stage)?
                .op_sub(&a[1].op_mul(&b[0], stage)?, stage)?;
            Ok(VecInstance::new(vec![s1, s2, s3]).into())
        }
        _ => Err(E::Builtin(
            "`cross` expects a 3-component vector of float arguments",
        )),
    }
}

fn call_degrees(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("degrees", e, n => n.to_degrees())
}

fn call_determinant(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("determinant".to_string()))
}

// NOTE: the function returns an error if computed out of domain
fn call_distance(e1: &Instance, e2: &Instance, stage: EvalStage) -> Result<Instance, E> {
    call_length(&e1.op_sub(e2, stage)?)
}

fn call_dot(e1: &Instance, e2: &Instance, stage: EvalStage) -> Result<Instance, E> {
    let (e1, e2) = convert(e1, e2).ok_or(E::Builtin("`dot` arguments are incompatible"))?;
    match (e1, e2) {
        (Instance::Vec(e1), Instance::Vec(e2)) => e1.dot(&e2, stage).map(Into::into),
        _ => Err(E::Builtin("`dot` expects vector arguments")),
    }
}

fn call_dot4u8packed(_a1: &Instance, _a2: &Instance) -> Result<Instance, E> {
    Err(E::Todo("dot4U8Packed".to_string()))
}

fn call_dot4i8packed(_a1: &Instance, _a2: &Instance) -> Result<Instance, E> {
    Err(E::Todo("dot4I8Packed".to_string()))
}

fn call_exp(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("exp", e, n => n.exp())
}

fn call_exp2(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("exp2", e, n => n.exp2())
}

fn call_extractbits(_a1: &Instance, _a2: &Instance, _a3: &Instance) -> Result<Instance, E> {
    Err(E::Todo("extractBits".to_string()))
}

fn call_faceforward(_a1: &Instance, _a2: &Instance, _a3: &Instance) -> Result<Instance, E> {
    Err(E::Todo("faceForward".to_string()))
}

fn call_firstleadingbit(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("firstLeadingBit".to_string()))
}

fn call_firsttrailingbit(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("firstTrailingBit".to_string()))
}

fn call_floor(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("floor", e, n => n.floor())
}

fn call_fma(_a1: &Instance, _a2: &Instance, _a3: &Instance) -> Result<Instance, E> {
    Err(E::Todo("fma".to_string()))
}

fn call_fract(e: &Instance, stage: EvalStage) -> Result<Instance, E> {
    e.op_sub(&call_floor(e)?, stage)
    // impl_call_float_unary!("fract", e, n => n.fract())
}

fn frexp_struct_name(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::AbstractFloat => Some("__frexp_result_abstract"),
        Type::F32 => Some("__frexp_result_f32"),
        Type::F16 => Some("__frexp_result_f16"),
        Type::Vec(n, ty) => match (n, &**ty) {
            (2, Type::AbstractFloat) => Some("__frexp_result_vec2_abstract"),
            (2, Type::F32) => Some("__frexp_result_vec2_f32"),
            (2, Type::F16) => Some("__frexp_result_vec2_f16"),
            (3, Type::AbstractFloat) => Some("__frexp_result_vec3_abstract"),
            (3, Type::F32) => Some("__frexp_result_vec3_f32"),
            (3, Type::F16) => Some("__frexp_result_vec3_f16"),
            (4, Type::AbstractFloat) => Some("__frexp_result_vec4_abstract"),
            (4, Type::F32) => Some("__frexp_result_vec4_f32"),
            (4, Type::F16) => Some("__frexp_result_vec4_f16"),
            _ => None,
        },
        _ => None,
    }
}

fn call_frexp(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`frexp` expects a float or vector of float argument");
    fn make_frexp_inst(name: &'static str, fract: Instance, exp: Instance) -> Instance {
        Instance::Struct(StructInstance::new(
            name.to_string(),
            vec![("fract".to_string(), fract), ("exp".to_string(), exp)],
        ))
    }
    // from: https://docs.rs/libm/latest/src/libm/math/frexp.rs.html#1-20
    fn frexp(x: f64) -> (f64, i32) {
        let mut y = x.to_bits();
        let ee = ((y >> 52) & 0x7ff) as i32;

        if ee == 0 {
            if x != 0.0 {
                let x1p64 = f64::from_bits(0x43f0000000000000);
                let (x, e) = frexp(x * x1p64);
                return (x, e - 64);
            }
            return (x, 0);
        } else if ee == 0x7ff {
            return (x, 0);
        }

        let e = ee - 0x3fe;
        y &= 0x800fffffffffffff;
        y |= 0x3fe0000000000000;
        (f64::from_bits(y), e)
    }
    match e {
        Instance::Literal(l) => match l {
            LiteralInstance::Bool(_) => todo!(),
            LiteralInstance::AbstractInt(_) => todo!(),
            LiteralInstance::AbstractFloat(n) => {
                let (fract, exp) = frexp(*n);
                Ok(make_frexp_inst(
                    "__frexp_result_abstract",
                    LiteralInstance::AbstractFloat(fract).into(),
                    LiteralInstance::AbstractInt(exp as i64).into(),
                ))
            }
            LiteralInstance::I32(_) => todo!(),
            LiteralInstance::U32(_) => todo!(),
            LiteralInstance::F32(n) => {
                let (fract, exp) = frexp(*n as f64);
                Ok(make_frexp_inst(
                    "__frexp_result_f32",
                    LiteralInstance::F32(fract as f32).into(),
                    LiteralInstance::I32(exp).into(),
                ))
            }
            LiteralInstance::F16(n) => {
                let (fract, exp) = frexp(n.to_f64().unwrap(/* SAFETY: f16 to f64 is lossless */));
                Ok(make_frexp_inst(
                    "__frexp_result_f16",
                    LiteralInstance::F16(f16::from_f64(fract)).into(),
                    LiteralInstance::I32(exp).into(),
                ))
            }
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(_) => todo!(),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => todo!(),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(n) => {
                let (fract, exp) = frexp(*n);
                Ok(make_frexp_inst(
                    "__frexp_result_f64",
                    LiteralInstance::F64(fract).into(),
                    LiteralInstance::I64(exp as i64).into(),
                ))
            }
        },
        Instance::Vec(v) => {
            let ty = v.inner_ty();
            let (fracts, exps): (Vec<_>, Vec<_>) = v
                .iter()
                .map(|l| match l.unwrap_literal_ref() {
                    LiteralInstance::AbstractFloat(n) => Ok(*n),
                    LiteralInstance::F32(n) => Ok(*n as f64),
                    LiteralInstance::F16(n) => {
                        Ok(n.to_f64().unwrap(/* SAFETY: f16 to f64 is lossless */))
                    }
                    _ => Err(ERR),
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(frexp)
                .unzip();
            let fracts = fracts
                .into_iter()
                .map(|n| match ty {
                    Type::AbstractFloat => LiteralInstance::AbstractFloat(n).into(),
                    Type::F32 => LiteralInstance::F32(n as f32).into(),
                    Type::F16 => LiteralInstance::F16(f16::from_f64(n)).into(),
                    _ => unreachable!("case handled above"),
                })
                .collect_vec();
            let exps = exps
                .into_iter()
                .map(|n| match ty {
                    Type::AbstractFloat => LiteralInstance::AbstractInt(n as i64).into(),
                    Type::F32 => LiteralInstance::I32(n).into(),
                    Type::F16 => LiteralInstance::I32(n).into(),
                    _ => unreachable!("case handled above"),
                })
                .collect_vec();
            let fract = VecInstance::new(fracts).into();
            let exp = VecInstance::new(exps).into();
            let name = frexp_struct_name(&v.ty()).ok_or(ERR)?;
            Ok(make_frexp_inst(name, fract, exp))
        }
        _ => Err(ERR),
    }
}

fn call_insertbits(
    _a1: &Instance,
    _a2: &Instance,
    _a3: &Instance,
    _a4: &Instance,
) -> Result<Instance, E> {
    Err(E::Todo("insertBits".to_string()))
}

// NOTE: the function returns NaN as an `indeterminate value` if computed out of domain
fn call_inversesqrt(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`inverseSqrt` expects a float or vector of float argument");
    fn lit_isqrt(l: &LiteralInstance) -> Result<LiteralInstance, E> {
        match l {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(_) => l
                .convert_to(&Type::AbstractFloat)
                .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))
                .map(|n| LiteralInstance::from(1.0 / n.unwrap_abstract_float().sqrt())),
            LiteralInstance::AbstractFloat(n) => Ok(LiteralInstance::from(1.0 / n.sqrt())),
            LiteralInstance::I32(_) => Err(ERR),
            LiteralInstance::U32(_) => Err(ERR),
            LiteralInstance::F32(n) => Ok(LiteralInstance::from(1.0 / n.sqrt())),
            LiteralInstance::F16(n) => Ok(LiteralInstance::from(f16::one() / n.sqrt())),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(n) => Ok(LiteralInstance::F64(1.0 / n.sqrt())),
        }
    }
    match e {
        Instance::Literal(l) => lit_isqrt(l).map(Into::into),
        Instance::Vec(v) => v.compwise_unary(lit_isqrt).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_ldexp(e1: &Instance, e2: &Instance) -> Result<Instance, E> {
    // from: https://docs.rs/libm/latest/src/libm/math/scalbn.rs.html#3-34
    fn scalbn(x: f64, mut n: i32) -> f64 {
        let x1p1023 = f64::from_bits(0x7fe0000000000000); // 0x1p1023 === 2 ^ 1023
        let x1p53 = f64::from_bits(0x4340000000000000); // 0x1p53 === 2 ^ 53
        let x1p_1022 = f64::from_bits(0x0010000000000000); // 0x1p-1022 === 2 ^ (-1022)

        let mut y = x;

        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                y *= x1p1023;
                n -= 1023;
                if n > 1023 {
                    n = 1023;
                }
            }
        } else if n < -1022 {
            /* make sure final n < -53 to avoid double
            rounding in the subnormal range */
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                y *= x1p_1022 * x1p53;
                n += 1022 - 53;
                if n < -1022 {
                    n = -1022;
                }
            }
        }
        y * f64::from_bits(((0x3ff + n) as u64) << 52)
    }
    fn ldexp_lit(l1: &LiteralInstance, l2: &LiteralInstance) -> Result<LiteralInstance, E> {
        match (l1, l2) {
            (LiteralInstance::AbstractInt(n1), LiteralInstance::AbstractInt(n2)) => Ok(
                LiteralInstance::AbstractFloat(scalbn(n1.to_f64().unwrap(), n2.to_i32().unwrap())),
            ),
            (LiteralInstance::AbstractFloat(n1), LiteralInstance::AbstractInt(n2)) => Ok(
                LiteralInstance::AbstractFloat(scalbn(*n1, n2.to_i32().unwrap())),
            ),
            (LiteralInstance::AbstractInt(n1), LiteralInstance::I32(n2)) => Ok(
                LiteralInstance::F32(scalbn(n1.to_f64().unwrap(), *n2) as f32),
            ),
            (LiteralInstance::AbstractFloat(n1), LiteralInstance::I32(n2)) => Ok(
                LiteralInstance::F32(scalbn(*n1, n2.to_i32().unwrap()) as f32),
            ),
            (LiteralInstance::F32(n1), LiteralInstance::AbstractInt(n2)) => Ok(
                LiteralInstance::F32(scalbn(n1.to_f64().unwrap(), n2.to_i32().unwrap()) as f32),
            ),
            (LiteralInstance::F32(n1), LiteralInstance::I32(n2)) => Ok(LiteralInstance::F32(
                scalbn(n1.to_f64().unwrap(), n2.to_i32().unwrap()) as f32,
            )),
            (LiteralInstance::F16(n1), LiteralInstance::AbstractInt(n2)) => {
                Ok(LiteralInstance::F16(f16::from_f64(scalbn(
                    n1.to_f64().unwrap(),
                    n2.to_i32().unwrap(),
                ))))
            }
            (LiteralInstance::F16(n1), LiteralInstance::I32(n2)) => Ok(LiteralInstance::F16(
                f16::from_f64(scalbn(n1.to_f64().unwrap(), *n2)),
            )),
            _ => Err(E::Builtin(
                "`ldexp` with scalar arguments expects a float and a i32 arguments",
            )),
        }
    }

    // TODO conversion errors
    match (e1, e2) {
        (Instance::Literal(l1), Instance::Literal(l2)) => ldexp_lit(l1, l2).map(Into::into),
        (Instance::Vec(v1), Instance::Vec(v2)) => v1.compwise_binary(v2, ldexp_lit).map(Into::into),
        _ => Err(E::Builtin(
            "`ldexp` expects two scalar or two vector arguments",
        )),
    }
}

fn call_length(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`length` expects a float or vector of float argument");
    match e {
        Instance::Literal(_) => call_abs(e),
        Instance::Vec(v) => call_sqrt(
            &v.op_mul(v, EvalStage::Exec)?
                .into_iter()
                .map(Ok)
                .reduce(|a, b| a?.op_add(&b?, EvalStage::Exec))
                .unwrap()?,
        ),
        _ => Err(ERR),
    }
}

fn call_log(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("log", e, n => n.ln())
}

fn call_log2(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("log2", e, n => n.log2())
}

fn call_max(e1: &Instance, e2: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`max` expects a scalar or vector of scalar argument");
    fn lit_max(e1: &LiteralInstance, e2: &LiteralInstance) -> Result<LiteralInstance, E> {
        match e1 {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(e1) => {
                Ok(LiteralInstance::from(*e1.max(&e2.unwrap_abstract_int())))
            }
            LiteralInstance::AbstractFloat(e1) => {
                Ok(LiteralInstance::from(e1.max(e2.unwrap_abstract_float())))
            }
            LiteralInstance::I32(e1) => Ok(LiteralInstance::from(*e1.max(&e2.unwrap_i_32()))),
            LiteralInstance::U32(e1) => Ok(LiteralInstance::from(*e1.max(&e2.unwrap_u_32()))),
            LiteralInstance::F32(e1) => Ok(LiteralInstance::from(e1.max(e2.unwrap_f_32()))),
            LiteralInstance::F16(e1) => Ok(LiteralInstance::from(e1.max(e2.unwrap_f_16()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(e1) => Ok(LiteralInstance::I64(*e1.max(&e2.unwrap_i_64()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(e1) => Ok(LiteralInstance::U64(*e1.max(&e2.unwrap_u_64()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(e1) => Ok(LiteralInstance::F64(e1.max(e2.unwrap_f_64()))),
        }
    }
    let (e1, e2) = convert(e1, e2).ok_or(E::Builtin("`max` arguments are incompatible"))?;
    match (e1, e2) {
        (Instance::Literal(e1), Instance::Literal(e2)) => lit_max(&e1, &e2).map(Into::into),
        (Instance::Vec(e1), Instance::Vec(e2)) => e1.compwise_binary(&e2, lit_max).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_min(e1: &Instance, e2: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`min` expects a scalar or vector of scalar argument");
    fn lit_min(e1: &LiteralInstance, e2: &LiteralInstance) -> Result<LiteralInstance, E> {
        match e1 {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(e1) => {
                Ok(LiteralInstance::from(*e1.min(&e2.unwrap_abstract_int())))
            }
            LiteralInstance::AbstractFloat(e1) => {
                Ok(LiteralInstance::from(e1.min(e2.unwrap_abstract_float())))
            }
            LiteralInstance::I32(e1) => Ok(LiteralInstance::from(*e1.min(&e2.unwrap_i_32()))),
            LiteralInstance::U32(e1) => Ok(LiteralInstance::from(*e1.min(&e2.unwrap_u_32()))),
            LiteralInstance::F32(e1) => Ok(LiteralInstance::from(e1.min(e2.unwrap_f_32()))),
            LiteralInstance::F16(e1) => Ok(LiteralInstance::from(e1.min(e2.unwrap_f_16()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(e1) => Ok(LiteralInstance::I64(*e1.max(&e2.unwrap_i_64()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(e1) => Ok(LiteralInstance::U64(*e1.max(&e2.unwrap_u_64()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(e1) => Ok(LiteralInstance::F64(e1.max(e2.unwrap_f_64()))),
        }
    }
    let (e1, e2) = convert(e1, e2).ok_or(E::Builtin("`min` arguments are incompatible"))?;
    match (e1, e2) {
        (Instance::Literal(e1), Instance::Literal(e2)) => lit_min(&e1, &e2).map(Into::into),
        (Instance::Vec(e1), Instance::Vec(e2)) => e1.compwise_binary(&e2, lit_min).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_mix(_a1: &Instance, _a2: &Instance, _a3: &Instance) -> Result<Instance, E> {
    Err(E::Todo("mix".to_string()))
}

fn modf_struct_name(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::AbstractFloat => Some("__modf_result_abstract"),
        Type::F32 => Some("__modf_result_f32"),
        Type::F16 => Some("__modf_result_f16"),
        Type::Vec(n, ty) => match (n, &**ty) {
            (2, Type::AbstractFloat) => Some("__modf_result_vec2_abstract"),
            (2, Type::F32) => Some("__modf_result_vec2_f32"),
            (2, Type::F16) => Some("__modf_result_vec2_f16"),
            (3, Type::AbstractFloat) => Some("__modf_result_vec3_abstract"),
            (3, Type::F32) => Some("__modf_result_vec3_f32"),
            (3, Type::F16) => Some("__modf_result_vec3_f16"),
            (4, Type::AbstractFloat) => Some("__modf_result_vec4_abstract"),
            (4, Type::F32) => Some("__modf_result_vec4_f32"),
            (4, Type::F16) => Some("__modf_result_vec4_f16"),
            _ => None,
        },
        _ => None,
    }
}

fn call_modf(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("modf".to_string()))
}

fn call_normalize(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("normalize".to_string()))
}

fn call_pow(e1: &Instance, e2: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`pow` expects a scalar or vector of scalar argument");
    fn lit_powf(e1: &LiteralInstance, e2: &LiteralInstance) -> Result<LiteralInstance, E> {
        match e1 {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(_) => {
                let e1 = e1
                    .convert_to(&Type::AbstractFloat)
                    .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?
                    .unwrap_abstract_float();
                let e2 = e2
                    .convert_to(&Type::AbstractFloat)
                    .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?
                    .unwrap_abstract_float();
                Ok(LiteralInstance::from(e1.powf(e2)))
            }
            LiteralInstance::AbstractFloat(e1) => {
                Ok(LiteralInstance::from(e1.powf(e2.unwrap_abstract_float())))
            }
            LiteralInstance::I32(_) => Err(ERR),
            LiteralInstance::U32(_) => Err(ERR),
            LiteralInstance::F32(e1) => Ok(LiteralInstance::from(e1.powf(e2.unwrap_f_32()))),
            LiteralInstance::F16(e1) => Ok(LiteralInstance::from(e1.powf(e2.unwrap_f_16()))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(e1) => Ok(LiteralInstance::F64(e1.powf(e2.unwrap_f_64()))),
        }
    }
    let (e1, e2) = convert(e1, e2).ok_or(E::Builtin("`pow` arguments are incompatible"))?;
    match (e1, e2) {
        (Instance::Literal(e1), Instance::Literal(e2)) => lit_powf(&e1, &e2).map(Into::into),
        (Instance::Vec(e1), Instance::Vec(e2)) => e1.compwise_binary(&e2, lit_powf).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_quantizetof16(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("quantizeToF16".to_string()))
}

fn call_radians(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("radians", e, n => n.to_radians())
}

fn call_reflect(_a1: &Instance, _a2: &Instance) -> Result<Instance, E> {
    Err(E::Todo("reflect".to_string()))
}

fn call_refract(_a1: &Instance, _a2: &Instance, _a3: &Instance) -> Result<Instance, E> {
    Err(E::Todo("refract".to_string()))
}

fn call_reversebits(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("reverseBits".to_string()))
}

fn call_round(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`round` expects a float or vector of float argument");
    fn lit_fn(l: &LiteralInstance) -> Result<LiteralInstance, E> {
        match l {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(_) => {
                let n = l
                    .convert_to(&Type::AbstractFloat)
                    .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?
                    .unwrap_abstract_float();
                Ok(LiteralInstance::from(n.round_ties_even()))
            }
            LiteralInstance::AbstractFloat(n) => Ok(LiteralInstance::from(n.round_ties_even())),
            LiteralInstance::I32(_) => Err(ERR),
            LiteralInstance::U32(_) => Err(ERR),
            LiteralInstance::F32(n) => Ok(LiteralInstance::from(n.round_ties_even())),
            LiteralInstance::F16(n) => Ok(LiteralInstance::from(f16::from_f32(
                f16::to_f32(*n).round_ties_even(),
            ))),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(n) => Ok(LiteralInstance::F64(n.round_ties_even())),
        }
    }
    match e {
        Instance::Literal(l) => lit_fn(l).map(Into::into),
        Instance::Vec(v) => v.compwise_unary(lit_fn).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_saturate(e: &Instance) -> Result<Instance, E> {
    match e {
        Instance::Literal(_) => {
            let zero = LiteralInstance::AbstractFloat(0.0);
            let one = LiteralInstance::AbstractFloat(1.0);
            call_clamp(e, &zero.into(), &one.into())
        }
        Instance::Vec(v) => {
            let n = v.n();
            let zero = Instance::from(LiteralInstance::AbstractFloat(0.0));
            let one = Instance::from(LiteralInstance::AbstractFloat(1.0));
            let zero = VecInstance::new((0..n).map(|_| zero.clone()).collect_vec());
            let one = VecInstance::new((0..n).map(|_| one.clone()).collect_vec());
            call_clamp(e, &zero.into(), &one.into())
        }
        _ => Err(E::Builtin(
            "`saturate` expects a float or vector of float argument",
        )),
    }
}

fn call_sign(e: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin(concat!(
        "`",
        "sign",
        "` expects a float or vector of float argument"
    ));
    fn lit_fn(l: &LiteralInstance) -> Result<LiteralInstance, E> {
        match l {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(n) => Ok(LiteralInstance::from(n.signum())),
            LiteralInstance::AbstractFloat(n) => Ok(LiteralInstance::from(if n.is_zero() {
                *n
            } else {
                n.signum()
            })),
            LiteralInstance::I32(n) => Ok(LiteralInstance::from(n.signum())),
            LiteralInstance::U32(n) => Ok(LiteralInstance::from(if n.is_zero() {
                *n
            } else {
                1
            })),
            LiteralInstance::F32(n) => Ok(LiteralInstance::from(if n.is_zero() {
                *n
            } else {
                n.signum()
            })),
            LiteralInstance::F16(n) => Ok(LiteralInstance::from(if n.is_zero() {
                *n
            } else {
                n.signum()
            })),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(n) => Ok(LiteralInstance::I64(n.signum())),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(n) => Ok(LiteralInstance::U64(if n.is_zero() {
                *n
            } else {
                1
            })),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(n) => Ok(LiteralInstance::F64(if n.is_zero() {
                *n
            } else {
                n.signum()
            })),
        }
    }
    match e {
        Instance::Literal(l) => lit_fn(l).map(Into::into),
        Instance::Vec(v) => v.compwise_unary(lit_fn).map(Into::into),
        _ => Err(ERR),
    }
}

fn call_sin(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("sin", e, n => n.sin())
}

fn call_sinh(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("sinh", e, n => n.sinh())
}

fn call_smoothstep(_low: &Instance, _high: &Instance, _x: &Instance) -> Result<Instance, E> {
    Err(E::Todo("smoothstep".to_string()))
}

fn call_sqrt(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("sqrt", e, n => n.sqrt())
}

fn call_step(edge: &Instance, x: &Instance) -> Result<Instance, E> {
    const ERR: E = E::Builtin("`step` expects a float or vector of float argument");
    fn lit_step(edge: &LiteralInstance, x: &LiteralInstance) -> Result<LiteralInstance, E> {
        match edge {
            LiteralInstance::Bool(_) => Err(ERR),
            LiteralInstance::AbstractInt(_) => {
                let edge = edge
                    .convert_to(&Type::AbstractFloat)
                    .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?
                    .unwrap_abstract_float();
                let x = x
                    .convert_to(&Type::AbstractFloat)
                    .ok_or(E::Conversion(Type::AbstractInt, Type::AbstractFloat))?
                    .unwrap_abstract_float();
                Ok(LiteralInstance::from(if edge <= x {
                    1.0
                } else {
                    0.0
                }))
            }
            LiteralInstance::AbstractFloat(edge) => Ok(LiteralInstance::from(
                if *edge <= x.unwrap_abstract_float() {
                    1.0
                } else {
                    0.0
                },
            )),
            LiteralInstance::I32(_) => Err(ERR),
            LiteralInstance::U32(_) => Err(ERR),
            LiteralInstance::F32(edge) => Ok(LiteralInstance::from(if *edge <= x.unwrap_f_32() {
                1.0
            } else {
                0.0
            })),
            LiteralInstance::F16(edge) => Ok(LiteralInstance::from(if *edge <= x.unwrap_f_16() {
                1.0
            } else {
                0.0
            })),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => Err(ERR),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(edge) => Ok(LiteralInstance::F64(if *edge <= x.unwrap_f_64() {
                1.0
            } else {
                0.0
            })),
        }
    }
    let (edge, x) = convert(edge, x).ok_or(E::Builtin("`step` arguments are incompatible"))?;
    match (edge, x) {
        (Instance::Literal(edge), Instance::Literal(x)) => lit_step(&edge, &x).map(Into::into),
        (Instance::Vec(edge), Instance::Vec(x)) => {
            edge.compwise_binary(&x, lit_step).map(Into::into)
        }
        _ => Err(ERR),
    }
}

fn call_tan(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("tan", e, n => n.tan())
}

fn call_tanh(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("tanh", e, n => n.tanh())
}

fn call_transpose(e: &Instance) -> Result<Instance, E> {
    match e {
        Instance::Mat(e) => Ok(e.transpose().into()),
        _ => Err(E::Builtin("`transpose` expects a matrix argument")),
    }
}

fn call_trunc(e: &Instance) -> Result<Instance, E> {
    impl_call_float_unary!("trunc", e, n => n.trunc())
}

// ------------
// DATA PACKING
// ------------
// reference: <https://www.w3.org/TR/WGSL/#pack-builtin-functions>

fn call_pack4x8snorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack4x8snorm".to_string()))
}

fn call_pack4x8unorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack4x8unorm".to_string()))
}

fn call_pack4xi8(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack4xI8".to_string()))
}

fn call_pack4xu8(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack4xU8".to_string()))
}

fn call_pack4xi8clamp(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack4xI8Clamp".to_string()))
}

fn call_pack4xu8clamp(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack4xU8Clamp".to_string()))
}

fn call_pack2x16snorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack2x16snorm".to_string()))
}

fn call_pack2x16unorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack2x16unorm".to_string()))
}

fn call_pack2x16float(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("pack2x16float".to_string()))
}

fn call_unpack4x8snorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("unpack4x8snorm".to_string()))
}

fn call_unpack4x8unorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("unpack4x8unorm".to_string()))
}

fn call_unpack4xi8(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("unpack4xI8".to_string()))
}

fn call_unpack4xu8(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("unpack4xU8".to_string()))
}

fn call_unpack2x16snorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("unpack2x16snorm".to_string()))
}

fn call_unpack2x16unorm(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("unpack2x16unorm".to_string()))
}

fn call_unpack2x16float(_a1: &Instance) -> Result<Instance, E> {
    Err(E::Todo("unpack2x16float".to_string()))
}

impl VecInstance {
    /// warning, this function does not check operand types
    pub fn dot(&self, rhs: &VecInstance, stage: EvalStage) -> Result<LiteralInstance, E> {
        self.compwise_binary(rhs, |a, b| a.op_mul(b, stage))?
            .into_iter()
            .map(|c| Ok(c.unwrap_literal()))
            .reduce(|a, b| a?.op_add(&b?, stage))
            .unwrap()
    }
}

impl MatInstance {
    /// warning, this function does not check operand types
    pub fn transpose(&self) -> MatInstance {
        let components = (0..self.r())
            .map(|j| {
                VecInstance::new(
                    (0..self.c())
                        .map(|i| self.get(i, j).unwrap().clone())
                        .collect_vec(),
                )
                .into()
            })
            .collect_vec();
        MatInstance::from_cols(components)
    }
}
