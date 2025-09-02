//! Implementations of built-in functions.
//!
//! This module implements the built-in functions and constructors.
//! Operators are implemented on the [`Instance`] type directly.
//! Some functions are still TODO.

pub mod call;

use half::prelude::*;
use num_traits::Zero;

use itertools::Itertools;

use crate::{
    CallSignature, Error, Instance, ShaderStage,
    conv::{Convert, convert_all_ty, convert_ty},
    enums::AddressSpace,
    inst::{ArrayInstance, LiteralInstance, MatInstance, StructInstance, VecInstance},
    ops::Compwise,
    tplt::{ArrayTemplate, BitcastTemplate, MatTemplate, TpltParam, VecTemplate},
    ty::{StructMemberType, StructType, TextureType, Ty, Type},
};

type E = Error;

// -----------------
// CONSTRUCTOR TYPES
// -----------------

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

/// Compute the return type of calling a built-in constructor function.
///
/// The arguments must be [loaded][Type::loaded].
pub fn constructor_type(name: &str, tplt: Option<&[TpltParam]>, args: &[Type]) -> Result<Type, E> {
    match (name, tplt, args) {
        ("array", Some(t), []) => Ok(ArrayTemplate::parse(t)?.ty()),
        ("array", Some(t), _) => array_ctor_ty_t(ArrayTemplate::parse(t)?, args),
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
        ("mat2x2", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(2, 2)),
        ("mat2x2", Some(t), _) => mat_ctor_ty_t(2, 2, MatTemplate::parse(t)?, args),
        ("mat2x2", None, _) => mat_ctor_ty(2, 2, args),
        ("mat2x3", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(2, 3)),
        ("mat2x3", Some(t), _) => mat_ctor_ty_t(2, 3, MatTemplate::parse(t)?, args),
        ("mat2x3", None, _) => mat_ctor_ty(2, 3, args),
        ("mat2x4", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(2, 4)),
        ("mat2x4", Some(t), _) => mat_ctor_ty_t(2, 4, MatTemplate::parse(t)?, args),
        ("mat2x4", None, _) => mat_ctor_ty(2, 4, args),
        ("mat3x2", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(3, 2)),
        ("mat3x2", Some(t), _) => mat_ctor_ty_t(3, 2, MatTemplate::parse(t)?, args),
        ("mat3x2", None, _) => mat_ctor_ty(3, 2, args),
        ("mat3x3", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(3, 3)),
        ("mat3x3", Some(t), _) => mat_ctor_ty_t(3, 3, MatTemplate::parse(t)?, args),
        ("mat3x3", None, _) => mat_ctor_ty(3, 3, args),
        ("mat3x4", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(3, 4)),
        ("mat3x4", Some(t), _) => mat_ctor_ty_t(3, 4, MatTemplate::parse(t)?, args),
        ("mat3x4", None, _) => mat_ctor_ty(3, 4, args),
        ("mat4x2", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(4, 2)),
        ("mat4x2", Some(t), _) => mat_ctor_ty_t(4, 2, MatTemplate::parse(t)?, args),
        ("mat4x2", None, _) => mat_ctor_ty(4, 2, args),
        ("mat4x3", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(4, 3)),
        ("mat4x3", Some(t), _) => mat_ctor_ty_t(4, 3, MatTemplate::parse(t)?, args),
        ("mat4x3", None, _) => mat_ctor_ty(4, 3, args),
        ("mat4x4", Some(t), []) => Ok(MatTemplate::parse(t)?.ty(4, 4)),
        ("mat4x4", Some(t), _) => mat_ctor_ty_t(4, 4, MatTemplate::parse(t)?, args),
        ("mat4x4", None, _) => mat_ctor_ty(4, 4, args),
        ("vec2", Some(t), []) => Ok(VecTemplate::parse(t)?.ty(2)),
        ("vec2", Some(t), _) => vec_ctor_ty_t(2, VecTemplate::parse(t)?, args),
        ("vec2", None, _) => vec_ctor_ty(2, args),
        ("vec3", Some(t), []) => Ok(VecTemplate::parse(t)?.ty(3)),
        ("vec3", Some(t), _) => vec_ctor_ty_t(3, VecTemplate::parse(t)?, args),
        ("vec3", None, _) => vec_ctor_ty(3, args),
        ("vec4", Some(t), []) => Ok(VecTemplate::parse(t)?.ty(4)),
        ("vec4", Some(t), _) => vec_ctor_ty_t(4, VecTemplate::parse(t)?, args),
        ("vec4", None, _) => vec_ctor_ty(4, args),
        _ => Err(E::Signature(CallSignature {
            name: name.to_string(),
            tplt: tplt.map(|t| t.to_vec()),
            args: args.to_vec(),
        })),
    }
}

// -----------------------
// BUILT-IN FUNCTION TYPES
// -----------------------

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

fn frexp_struct_type(ty: &Type) -> Option<StructType> {
    frexp_struct_name(ty).map(|name| {
        let exp_ty = if ty.is_abstract() {
            Type::AbstractInt
        } else {
            Type::I32
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

fn atomic_compare_exchange_struct_type(ty: &Type) -> StructType {
    StructType {
        name: "__atomic_compare_exchange_result".to_string(),
        members: vec![
            StructMemberType::new("old_value".to_string(), ty.clone()),
            StructMemberType::new("exchanged".to_string(), Type::Bool),
        ],
    }
}

fn modf_struct_type(ty: &Type) -> Option<StructType> {
    modf_struct_name(ty).map(|name| StructType {
        name: name.to_string(),
        members: vec![
            StructMemberType::new("fract".to_string(), ty.clone()),
            StructMemberType::new("whole".to_string(), ty.clone()),
        ],
    })
}

/// Compute the return type of calling a built-in function.
///
/// The arguments must be [loaded][Type::loaded].
///
/// Does not include constructor built-ins, see [`constructor_type`].
/// Some functions are still TODO, see [`call`] for the list of functions and statuses.
pub fn builtin_fn_type(
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
        ("bitcast", Some(t), [_]) => Ok(Some(BitcastTemplate::parse(t)?.ty())),
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
        ("modf", None, [a]) if is_float(a) => Ok(Some(modf_struct_type(a).unwrap().into())),
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

// -----------
// ZERO VALUES
// -----------
// reference: <https://www.w3.org/TR/WGSL/#zero-value>

impl Instance {
    /// Zero-value initialize an instance of a given type.
    pub fn zero_value(ty: &Type) -> Result<Self, E> {
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
            Type::Struct(s) => StructInstance::zero_value(s).map(Into::into),
            Type::Array(a_ty, Some(n)) => ArrayInstance::zero_value(*n, a_ty).map(Into::into),
            Type::Array(_, None) => Err(E::NotConstructible(ty.clone())),
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(_, _) => Err(E::NotConstructible(ty.clone())),
            Type::Vec(n, v_ty) => VecInstance::zero_value(*n, v_ty).map(Into::into),
            Type::Mat(c, r, m_ty) => MatInstance::zero_value(*c, *r, m_ty).map(Into::into),
            Type::Atomic(_)
            | Type::Ptr(_, _, _)
            | Type::Ref(_, _, _)
            | Type::Texture(_)
            | Type::Sampler(_) => Err(E::NotConstructible(ty.clone())),
        }
    }

    /// Apply the load rule.
    ///
    /// Reference: <https://www.w3.org/TR/WGSL/#load-rule>
    pub fn loaded(mut self) -> Result<Self, E> {
        while let Instance::Ref(r) = self {
            self = r.read()?.to_owned();
        }
        Ok(self)
    }
}

impl LiteralInstance {
    /// The zero-value constructor.
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
    /// Zero-value initialize a `struct` instance.
    pub fn zero_value(s: &StructType) -> Result<Self, E> {
        let members = s
            .members
            .iter()
            .map(|mem| {
                let val = Instance::zero_value(&mem.ty)?;
                Ok(val)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(StructInstance::new(s.clone(), members))
    }
}

impl ArrayInstance {
    /// Zero-value initialize an `array` instance.
    pub fn zero_value(n: usize, ty: &Type) -> Result<Self, E> {
        let zero = Instance::zero_value(ty)?;
        let comps = (0..n).map(|_| zero.clone()).collect_vec();
        Ok(ArrayInstance::new(comps, false))
    }
}

impl VecInstance {
    /// Zero-value initialize a `vec` instance.
    pub fn zero_value(n: u8, ty: &Type) -> Result<Self, E> {
        let zero = Instance::Literal(LiteralInstance::zero_value(ty)?);
        let comps = (0..n).map(|_| zero.clone()).collect_vec();
        Ok(VecInstance::new(comps))
    }
}

impl MatInstance {
    /// Zero-value initialize a `mat` instance.
    pub fn zero_value(c: u8, r: u8, ty: &Type) -> Result<Self, E> {
        let zero = Instance::Literal(LiteralInstance::zero_value(ty)?);
        let zero_col = Instance::Vec(VecInstance::new((0..r).map(|_| zero.clone()).collect_vec()));
        let comps = (0..c).map(|_| zero_col.clone()).collect_vec();
        Ok(MatInstance::from_cols(comps))
    }
}

impl VecInstance {
    /// Warning, this function does not check operand types
    pub fn dot(&self, rhs: &VecInstance, stage: ShaderStage) -> Result<LiteralInstance, E> {
        self.compwise_binary(rhs, |a, b| a.op_mul(b, stage))?
            .into_iter()
            .map(|c| Ok(c.unwrap_literal()))
            .reduce(|a, b| a?.op_add(&b?, stage))
            .unwrap()
    }
}

impl MatInstance {
    /// Warning, this function does not check operand types
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

/// Call a built-in function.
///
/// The arguments must be [loaded][Type::loaded].
///
/// Includes constructor built-ins.
/// Some functions are still TODO, see [`call`] for the list of functions and statuses.
pub fn call_builtin(
    name: &str,
    tplt: Option<&[TpltParam]>,
    args: &[Instance],
    stage: ShaderStage,
) -> Result<Option<Instance>, E> {
    match (name, tplt, args) {
        // constructors
        ("array", Some(t), []) => Instance::zero_value(&ArrayTemplate::parse(t)?.ty()),
        ("array", Some(t), a) => call::array_t(ArrayTemplate::parse(t)?, a),
        ("array", None, a) => call::array(a),
        ("bool", None, []) => Instance::zero_value(&Type::Bool),
        ("bool", None, [a1]) => call::bool(a1),
        ("i32", None, []) => Instance::zero_value(&Type::I32),
        ("i32", None, [a1]) => call::i32(a1),
        ("u32", None, []) => Instance::zero_value(&Type::U32),
        ("u32", None, [a1]) => call::u32(a1),
        ("f32", None, []) => Instance::zero_value(&Type::F32),
        ("f32", None, [a1]) => call::f32(a1, stage),
        ("f16", None, []) => Instance::zero_value(&Type::F16),
        ("f16", None, [a1]) => call::f16(a1, stage),
        ("mat2x2", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(2, 2)),
        ("mat2x2", Some(t), a) => call::mat_t(2, 2, MatTemplate::parse(t)?, a, stage),
        ("mat2x2", None, a) => call::mat(2, 2, a),
        ("mat2x3", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(2, 3)),
        ("mat2x3", Some(t), a) => call::mat_t(2, 3, MatTemplate::parse(t)?, a, stage),
        ("mat2x3", None, a) => call::mat(2, 3, a),
        ("mat2x4", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(2, 4)),
        ("mat2x4", Some(t), a) => call::mat_t(2, 4, MatTemplate::parse(t)?, a, stage),
        ("mat2x4", None, a) => call::mat(2, 4, a),
        ("mat3x2", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(3, 2)),
        ("mat3x2", Some(t), a) => call::mat_t(3, 2, MatTemplate::parse(t)?, a, stage),
        ("mat3x2", None, a) => call::mat(3, 2, a),
        ("mat3x3", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(3, 3)),
        ("mat3x3", Some(t), a) => call::mat_t(3, 3, MatTemplate::parse(t)?, a, stage),
        ("mat3x3", None, a) => call::mat(3, 3, a),
        ("mat3x4", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(3, 4)),
        ("mat3x4", Some(t), a) => call::mat_t(3, 4, MatTemplate::parse(t)?, a, stage),
        ("mat3x4", None, a) => call::mat(3, 4, a),
        ("mat4x2", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(4, 2)),
        ("mat4x2", Some(t), a) => call::mat_t(4, 2, MatTemplate::parse(t)?, a, stage),
        ("mat4x2", None, a) => call::mat(4, 2, a),
        ("mat4x3", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(4, 3)),
        ("mat4x3", Some(t), a) => call::mat_t(4, 3, MatTemplate::parse(t)?, a, stage),
        ("mat4x3", None, a) => call::mat(4, 3, a),
        ("mat4x4", Some(t), []) => Instance::zero_value(&MatTemplate::parse(t)?.ty(4, 4)),
        ("mat4x4", Some(t), a) => call::mat_t(4, 4, MatTemplate::parse(t)?, a, stage),
        ("mat4x4", None, a) => call::mat(4, 4, a),
        ("vec2", Some(t), []) => Instance::zero_value(&VecTemplate::parse(t)?.ty(2)),
        ("vec2", Some(t), a) => call::vec_t(2, VecTemplate::parse(t)?, a, stage),
        ("vec2", None, a) => call::vec(2, a),
        ("vec3", Some(t), []) => Instance::zero_value(&VecTemplate::parse(t)?.ty(3)),
        ("vec3", Some(t), a) => call::vec_t(3, VecTemplate::parse(t)?, a, stage),
        ("vec3", None, a) => call::vec(3, a),
        ("vec4", Some(t), []) => Instance::zero_value(&VecTemplate::parse(t)?.ty(4)),
        ("vec4", Some(t), a) => call::vec_t(4, VecTemplate::parse(t)?, a, stage),
        ("vec4", None, a) => call::vec(4, a),
        // bitcast
        ("bitcast", Some(t), [a1]) => call::bitcast_t(BitcastTemplate::parse(t)?, a1),
        // logical
        ("all", None, [a]) => call::all(a),
        ("any", None, [a]) => call::any(a),
        ("select", None, [a1, a2, a3]) => call::select(a1, a2, a3),
        // array
        ("arrayLength", None, [a]) => call::arrayLength(a),
        // numeric
        ("abs", None, [a]) => call::abs(a),
        ("acos", None, [a]) => call::acos(a),
        ("acosh", None, [a]) => call::acosh(a),
        ("asin", None, [a]) => call::asin(a),
        ("asinh", None, [a]) => call::asinh(a),
        ("atan", None, [a]) => call::atan(a),
        ("atanh", None, [a]) => call::atanh(a),
        ("atan2", None, [a1, a2]) => call::atan2(a1, a2),
        ("ceil", None, [a]) => call::ceil(a),
        ("clamp", None, [a1, a2, a3]) => call::clamp(a1, a2, a3),
        ("cos", None, [a]) => call::cos(a),
        ("cosh", None, [a]) => call::cosh(a),
        ("countLeadingZeros", None, [a]) => call::countLeadingZeros(a),
        ("countOneBits", None, [a]) => call::countOneBits(a),
        ("countTrailingZeros", None, [a]) => call::countTrailingZeros(a),
        ("cross", None, [a1, a2]) => call::cross(a1, a2, stage),
        ("degrees", None, [a]) => call::degrees(a),
        ("determinant", None, [a]) => call::determinant(a),
        ("distance", None, [a1, a2]) => call::distance(a1, a2, stage),
        ("dot", None, [a1, a2]) => call::dot(a1, a2, stage),
        ("dot4U8Packed", None, [a1, a2]) => call::dot4U8Packed(a1, a2),
        ("dot4I8Packed", None, [a1, a2]) => call::dot4I8Packed(a1, a2),
        ("exp", None, [a]) => call::exp(a),
        ("exp2", None, [a]) => call::exp2(a),
        ("extractBits", None, [a1, a2, a3]) => call::extractBits(a1, a2, a3),
        ("faceForward", None, [a1, a2, a3]) => call::faceForward(a1, a2, a3),
        ("firstLeadingBit", None, [a]) => call::firstLeadingBit(a),
        ("firstTrailingBit", None, [a]) => call::firstTrailingBit(a),
        ("floor", None, [a]) => call::floor(a),
        ("fma", None, [a1, a2, a3]) => call::fma(a1, a2, a3),
        ("fract", None, [a]) => call::fract(a, stage),
        ("frexp", None, [a]) => call::frexp(a),
        ("insertBits", None, [a1, a2, a3, a4]) => call::insertBits(a1, a2, a3, a4),
        ("inverseSqrt", None, [a]) => call::inverseSqrt(a),
        ("ldexp", None, [a1, a2]) => call::ldexp(a1, a2),
        ("length", None, [a]) => call::length(a),
        ("log", None, [a]) => call::log(a),
        ("log2", None, [a]) => call::log2(a),
        ("max", None, [a1, a2]) => call::max(a1, a2),
        ("min", None, [a1, a2]) => call::min(a1, a2),
        ("mix", None, [a1, a2, a3]) => call::mix(a1, a2, a3, stage),
        ("modf", None, [a]) => call::modf(a),
        ("normalize", None, [a]) => call::normalize(a, stage),
        ("pow", None, [a1, a2]) => call::pow(a1, a2),
        ("quantizeToF16", None, [a]) => call::quantizeToF16(a),
        ("radians", None, [a]) => call::radians(a),
        ("reflect", None, [a1, a2]) => call::reflect(a1, a2),
        ("refract", None, [a1, a2, a3]) => call::refract(a1, a2, a3),
        ("reverseBits", None, [a]) => call::reverseBits(a),
        ("round", None, [a]) => call::round(a),
        ("saturate", None, [a]) => call::saturate(a),
        ("sign", None, [a]) => call::sign(a),
        ("sin", None, [a]) => call::sin(a),
        ("sinh", None, [a]) => call::sinh(a),
        ("smoothstep", None, [a1, a2, a3]) => call::smoothstep(a1, a2, a3),
        ("sqrt", None, [a]) => call::sqrt(a),
        ("step", None, [a1, a2]) => call::step(a1, a2),
        ("tan", None, [a]) => call::tan(a),
        ("tanh", None, [a]) => call::tanh(a),
        ("transpose", None, [a]) => call::transpose(a),
        ("trunc", None, [a]) => call::trunc(a),
        // atomic
        ("atomicLoad", None, [a]) => call::atomicLoad(a),
        ("atomicStore", None, [a1, a2]) => {
            call::atomicStore(a1, a2)?;
            return Ok(None);
        }
        ("atomicSub", None, [a1, a2]) => call::atomicSub(a1, a2),
        ("atomicMax", None, [a1, a2]) => call::atomicMax(a1, a2),
        ("atomicMin", None, [a1, a2]) => call::atomicMin(a1, a2),
        ("atomicAnd", None, [a1, a2]) => call::atomicAnd(a1, a2),
        ("atomicOr", None, [a1, a2]) => call::atomicOr(a1, a2),
        ("atomicXor", None, [a1, a2]) => call::atomicXor(a1, a2),
        ("atomicExchange", None, [a1, a2]) => call::atomicExchange(a1, a2),
        ("atomicCompareExchangeWeak", None, [a1, a2]) => call::atomicCompareExchangeWeak(a1, a2),
        // packing
        ("pack4x8snorm", None, [a]) => call::pack4x8snorm(a),
        ("pack4x8unorm", None, [a]) => call::pack4x8unorm(a),
        ("pack4xI8", None, [a]) => call::pack4xI8(a),
        ("pack4xU8", None, [a]) => call::pack4xU8(a),
        ("pack4xI8Clamp", None, [a]) => call::pack4xI8Clamp(a),
        ("pack4xU8Clamp", None, [a]) => call::pack4xU8Clamp(a),
        ("pack2x16snorm", None, [a]) => call::pack2x16snorm(a),
        ("pack2x16unorm", None, [a]) => call::pack2x16unorm(a),
        ("pack2x16float", None, [a]) => call::pack2x16float(a),
        ("unpack4x8snorm", None, [a]) => call::unpack4x8snorm(a),
        ("unpack4x8unorm", None, [a]) => call::unpack4x8unorm(a),
        ("unpack4xI8", None, [a]) => call::unpack4xI8(a),
        ("unpack4xU8", None, [a]) => call::unpack4xU8(a),
        ("unpack2x16snorm", None, [a]) => call::unpack2x16snorm(a),
        ("unpack2x16unorm", None, [a]) => call::unpack2x16unorm(a),
        ("unpack2x16float", None, [a]) => call::unpack2x16float(a),
        // synchronization
        // barrier primitives are no-op on the cpu
        ("storageBarrier", None, []) => return Ok(None),
        ("textureBarrier", None, []) => return Ok(None),
        ("workgroupBarrier", None, []) => return Ok(None),
        _ => Err(E::Signature(CallSignature {
            name: name.to_string(),
            tplt: tplt.map(|tplt| tplt.to_vec()),
            args: args.iter().map(|a| a.ty()).collect_vec(),
        })),
    }
    .map(Option::Some)
}
