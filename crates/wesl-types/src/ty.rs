//! WGSL [`Type`]s.

use std::str::FromStr;

use crate::{
    EvalError, Instance,
    enums::{AddressSpace, TextureType},
    inst::{
        ArrayInstance, AtomicInstance, LiteralInstance, MatInstance, PtrInstance, RefInstance,
        StructInstance, VecInstance,
    },
};

use derive_more::derive::{IsVariant, Unwrap};
use itertools::Itertools;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructType {
    pub name: String,
    pub members: Vec<(String, Type)>,
}

impl StructType {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn member(&self, name: &str) -> Option<&Type> {
        self.members
            .iter()
            .find_map(|(n, inst)| (n == name).then_some(inst))
    }
    pub fn member_mut(&mut self, name: &str) -> Option<&mut Type> {
        self.members
            .iter_mut()
            .find_map(|(n, inst)| (n == name).then_some(inst))
    }
    pub fn iter_members(&self) -> impl Iterator<Item = &(String, Type)> {
        self.members.iter()
    }
}

impl From<StructType> for Type {
    fn from(value: StructType) -> Self {
        Self::Struct(Box::new(value))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum SampledType {
    I32,
    U32,
    F32,
}

impl TryFrom<&Type> for SampledType {
    type Error = EvalError;

    fn try_from(value: &Type) -> Result<Self, Self::Error> {
        match value {
            Type::I32 => Ok(SampledType::I32),
            Type::U32 => Ok(SampledType::U32),
            Type::F32 => Ok(SampledType::F32),
            _ => Err(EvalError::SampledType(value.clone())),
        }
    }
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

#[derive(Clone, Debug, PartialEq, Eq, IsVariant, Unwrap)]
pub enum Type {
    Bool,
    AbstractInt,
    AbstractFloat,
    I32,
    U32,
    F32,
    F16,
    #[cfg(feature = "naga_ext")]
    I64,
    #[cfg(feature = "naga_ext")]
    U64,
    #[cfg(feature = "naga_ext")]
    F64,
    Struct(Box<StructType>),
    Array(Box<Type>, Option<usize>),
    #[cfg(feature = "naga_ext")]
    BindingArray(Box<Type>, Option<usize>),
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
            #[cfg(feature = "naga_ext")]
            Type::I64 => self.clone(),
            #[cfg(feature = "naga_ext")]
            Type::U64 => self.clone(),
            #[cfg(feature = "naga_ext")]
            Type::F64 => self.clone(),
            Type::Struct(_) => self.clone(),
            Type::Array(ty, _) => ty.ty(),
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(ty, _) => ty.ty(),
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
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(_) => Type::I64,
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(_) => Type::U64,
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(_) => Type::F64,
        }
    }
}

impl Ty for StructInstance {
    fn ty(&self) -> Type {
        Type::Struct(Box::new(StructType {
            name: self.name().to_string(),
            members: self
                .iter_members()
                .map(|(name, inst)| (name.to_string(), inst.ty()))
                .collect_vec(),
        }))
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
