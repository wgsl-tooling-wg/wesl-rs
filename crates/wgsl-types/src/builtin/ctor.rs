//! Implementations of the constructor functions, including zero-value constructors.

use half::prelude::*;
use itertools::Itertools;
use num_traits::Zero;

use crate::{
    CallSignature, Error, ShaderStage,
    conv::{Convert, convert_all_ty},
    inst::{ArrayInstance, Instance, LiteralInstance, MatInstance, StructInstance, VecInstance},
    tplt::{ArrayTemplate, MatTemplate, TpltParam, VecTemplate},
    ty::{StructType, Ty, Type},
};

use super::Compwise;

type E = Error;

pub fn is_ctor_fn(name: &str) -> bool {
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
pub fn ctor_type(name: &str, tplt: Option<&[TpltParam]>, args: &[Type]) -> Result<Type, E> {
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
