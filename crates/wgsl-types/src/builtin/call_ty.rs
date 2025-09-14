//! Type computations for built-in functions.

use crate::{
    CallSignature, Error,
    conv::{Convert, convert_all_ty, convert_ty},
    syntax::AddressSpace,
    tplt::{BitcastTemplate, TpltParam},
    ty::{StructMemberType, StructType, TextureDimensions, TextureType, Ty, Type},
};

type E = Error;

/// Compute the return type of calling a built-in function.
///
/// The arguments must be [loaded][Type::loaded].
///
/// Does not include constructor built-ins, see [`type_ctor`][super::type_ctor].
///
/// Some functions are still TODO, see [`call`][super::call] for the list of functions and statuses.
pub fn type_builtin_fn(
    name: &str,
    tplt: Option<&[TpltParam]>,
    args: &[Type],
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
    let err = || {
        E::Signature(CallSignature {
            name: name.to_string(),
            tplt: tplt.map(|t| t.to_vec()),
            args: args.to_vec(),
        })
    };

    match (name, tplt, args) {
        // bitcast
        ("bitcast", Some(t), [_]) => Ok(Some(BitcastTemplate::parse(t)?.ty().clone())),
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
        ("frexp", None, [a]) if is_float(a) => Ok(Some(frexp_struct_type(a).unwrap().into())),
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
                && a2.inner_ty().concretize().is_i32()
                || a1.is_float() && a2.concretize().is_i32())
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
        ("modf", None, [a]) if is_float(a) => Ok(Some(modf_struct_type(a).unwrap().into())),
        ("normalize", None, [a @ Type::Vec(_, ty)]) if ty.is_float() => Ok(Some(a.clone())),
        ("pow", None, [a1, a2]) => convert_ty(a1, a2).cloned().map(Some).ok_or_else(err),
        ("quantizeToF16", None, [a])
            if a.concretize().is_f32() || a.is_vec() && a.inner_ty().concretize().is_f32() =>
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
        ("sign", None, [a]) if is_numeric(a) && !a.inner_ty().is_u32() => Ok(Some(a.clone())),
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
            if t.dimensions() == TextureDimensions::D1 =>
        {
            Ok(Some(Type::U32))
        }
        ("textureDimensions", None, [Type::Texture(t)] | [Type::Texture(t), _])
            if t.dimensions() == TextureDimensions::D2 =>
        {
            Ok(Some(Type::Vec(2, Type::U32.into())))
        }
        ("textureDimensions", None, [Type::Texture(t)] | [Type::Texture(t), _])
            if t.dimensions() == TextureDimensions::D3 =>
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
        ("textureNumLayers", None, [Type::Texture(t)]) if t.is_arrayed() => Ok(Some(Type::U32)),
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
        (
            "textureSampleBaseClampToEdge",
            None,
            [
                Type::Texture(TextureType::Sampled2D(_) | TextureType::External),
                ..,
            ],
        ) => Ok(Some(Type::Vec(4, Box::new(Type::F32)))),
        ("textureStore", None, [Type::Texture(t), ..]) if t.is_storage() => Ok(None),
        // atomic
        // TODO check arguments for atomic functions
        ("atomicLoad", None, [Type::Ptr(_, t, _)]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicStore", None, [Type::Ptr(_, t, _)]) if matches!(**t, Type::Atomic(_)) => Ok(None),
        ("atomicAdd", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicSub", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicMax", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicMin", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicAnd", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicOr", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicXor", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicExchange", None, [Type::Ptr(_, t, _), _]) if matches!(**t, Type::Atomic(_)) => {
            Ok(Some(*t.clone().unwrap_atomic()))
        }
        ("atomicCompareExchangeWeak", None, [Type::Ptr(_, t, _), _, _])
            if matches!(**t, Type::Atomic(_)) =>
        {
            let ty = match &**t {
                Type::Atomic(ty) => &**ty,
                _ => unreachable!("type atomic matched above"),
            };
            Ok(Some(atomic_compare_exchange_struct_type(ty).into()))
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
        ("pack4xI8Clamp", None, [a]) if a.is_convertible_to(&Type::Vec(4, Type::F32.into())) => {
            Ok(Some(Type::U32))
        }
        ("pack4xU8Clamp", None, [a]) if a.is_convertible_to(&Type::Vec(4, Type::F32.into())) => {
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
        ("workgroupUniformLoad", None, [Type::Ptr(AddressSpace::Workgroup, t, _)]) => {
            Ok(Some(*t.clone()))
        }
        // subgroup
        ("subgroupAdd", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupExclusiveAdd", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupInclusiveAdd", None, [a]) if is_numeric(a) => Ok(Some(a.concretize())),
        ("subgroupAll", None, [Type::Bool]) => Ok(Some(Type::Bool)),
        ("subgroupAnd", None, [a]) if is_integer(a) => Ok(Some(a.concretize())),
        ("subgroupAny", None, [Type::Bool]) => Ok(Some(Type::Bool)),
        ("subgroupBallot", None, [Type::Bool]) => Ok(Some(Type::Vec(4, Type::U32.into()))),
        #[cfg(feature = "naga-ext")]
        ("subgroupBallot", None, []) => Ok(Some(Type::Vec(4, Type::U32.into()))),
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

// ---------------------
// BUILT-IN RETURN TYPES
// ---------------------

pub(crate) fn frexp_struct_name(ty: &Type) -> Option<&'static str> {
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

pub(crate) fn frexp_struct_type(ty: &Type) -> Option<StructType> {
    frexp_struct_name(ty).map(|name| {
        let exp_inner_ty = if ty.is_abstract() {
            Type::AbstractInt
        } else {
            Type::I32
        };
        let exp_ty = match ty {
            Type::Vec(n, _) => Type::Vec(*n, Box::new(exp_inner_ty)),
            _ => exp_inner_ty,
        };
        StructType {
            name: name.to_string(),
            members: vec![
                StructMemberType::new("fract".to_string(), ty.clone()),
                StructMemberType::new("exp".to_string(), exp_ty),
            ],
        }
    })
}

pub(crate) fn modf_struct_name(ty: &Type) -> Option<&'static str> {
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

pub(crate) fn atomic_compare_exchange_struct_type(ty: &Type) -> StructType {
    StructType {
        name: "__atomic_compare_exchange_result".to_string(),
        members: vec![
            StructMemberType::new("old_value".to_string(), ty.clone()),
            StructMemberType::new("exchanged".to_string(), Type::Bool),
        ],
    }
}

pub(crate) fn modf_struct_type(ty: &Type) -> Option<StructType> {
    modf_struct_name(ty).map(|name| StructType {
        name: name.to_string(),
        members: vec![
            StructMemberType::new("fract".to_string(), ty.clone()),
            StructMemberType::new("whole".to_string(), ty.clone()),
        ],
    })
}
