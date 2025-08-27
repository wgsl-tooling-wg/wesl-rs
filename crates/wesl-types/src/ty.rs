use std::str::FromStr;

use crate::{AccessMode, AddressSpace};

use super::{
    ArrayInstance, AtomicInstance, EvalError, Instance, LiteralInstance, MatInstance, PtrInstance,
    RefInstance, StructInstance, VecInstance,
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
    #[cfg(feature = "naga_ext")]
    R8Unorm,
    #[cfg(feature = "naga_ext")]
    R8Snorm,
    #[cfg(feature = "naga_ext")]
    R8Uint,
    #[cfg(feature = "naga_ext")]
    R8Sint,
    #[cfg(feature = "naga_ext")]
    R16Unorm,
    #[cfg(feature = "naga_ext")]
    R16Snorm,
    #[cfg(feature = "naga_ext")]
    R16Uint,
    #[cfg(feature = "naga_ext")]
    R16Sint,
    #[cfg(feature = "naga_ext")]
    R16Float,
    #[cfg(feature = "naga_ext")]
    Rg8Unorm,
    #[cfg(feature = "naga_ext")]
    Rg8Snorm,
    #[cfg(feature = "naga_ext")]
    Rg8Uint,
    #[cfg(feature = "naga_ext")]
    Rg8Sint,
    #[cfg(feature = "naga_ext")]
    Rg16Unorm,
    #[cfg(feature = "naga_ext")]
    Rg16Snorm,
    #[cfg(feature = "naga_ext")]
    Rg16Uint,
    #[cfg(feature = "naga_ext")]
    Rg16Sint,
    #[cfg(feature = "naga_ext")]
    Rg16Float,
    #[cfg(feature = "naga_ext")]
    Rgb10a2Uint,
    #[cfg(feature = "naga_ext")]
    Rgb10a2Unorm,
    #[cfg(feature = "naga_ext")]
    Rg11b10Float,
    #[cfg(feature = "naga_ext")]
    R64Uint,
    #[cfg(feature = "naga_ext")]
    Rgba16Unorm,
    #[cfg(feature = "naga_ext")]
    Rgba16Snorm,
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
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Unorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Snorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Uint => SampledType::U32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Sint => SampledType::I32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Unorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Snorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Uint => SampledType::U32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Sint => SampledType::I32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Float => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Unorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Snorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Uint => SampledType::U32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Sint => SampledType::I32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Unorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Snorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Uint => SampledType::U32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Sint => SampledType::I32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Float => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgb10a2Uint => SampledType::U32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgb10a2Unorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg11b10Float => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R64Uint => SampledType::U32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgba16Unorm => SampledType::F32,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgba16Snorm => SampledType::F32,
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
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Unorm => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Snorm => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Uint => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Sint => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Unorm => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Snorm => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Uint => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Sint => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Float => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Unorm => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Snorm => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Uint => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Sint => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Unorm => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Snorm => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Uint => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Sint => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Float => 2,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgb10a2Uint => 4,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgb10a2Unorm => 4,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg11b10Float => 3,
            #[cfg(feature = "naga_ext")]
            TexelFormat::R64Uint => 1,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgba16Unorm => 4,
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgba16Snorm => 4,
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
            #[cfg(feature = "naga_ext")]
            "r8unorm" => Ok(Self::R8Unorm),
            #[cfg(feature = "naga_ext")]
            "r8snorm" => Ok(Self::R8Snorm),
            #[cfg(feature = "naga_ext")]
            "r8uint" => Ok(Self::R8Uint),
            #[cfg(feature = "naga_ext")]
            "r8sint" => Ok(Self::R8Sint),
            #[cfg(feature = "naga_ext")]
            "r16unorm" => Ok(Self::R16Unorm),
            #[cfg(feature = "naga_ext")]
            "r16snorm" => Ok(Self::R16Snorm),
            #[cfg(feature = "naga_ext")]
            "r16uint" => Ok(Self::R16Uint),
            #[cfg(feature = "naga_ext")]
            "r16sint" => Ok(Self::R16Sint),
            #[cfg(feature = "naga_ext")]
            "r16float" => Ok(Self::R16Float),
            #[cfg(feature = "naga_ext")]
            "rg8unorm" => Ok(Self::Rg8Unorm),
            #[cfg(feature = "naga_ext")]
            "rg8snorm" => Ok(Self::Rg8Snorm),
            #[cfg(feature = "naga_ext")]
            "rg8uint" => Ok(Self::Rg8Uint),
            #[cfg(feature = "naga_ext")]
            "rg8sint" => Ok(Self::Rg8Sint),
            #[cfg(feature = "naga_ext")]
            "rg16unorm" => Ok(Self::Rg16Unorm),
            #[cfg(feature = "naga_ext")]
            "rg16snorm" => Ok(Self::Rg16Snorm),
            #[cfg(feature = "naga_ext")]
            "rg16uint" => Ok(Self::Rg16Uint),
            #[cfg(feature = "naga_ext")]
            "rg16sint" => Ok(Self::Rg16Sint),
            #[cfg(feature = "naga_ext")]
            "rg16float" => Ok(Self::Rg16Float),
            #[cfg(feature = "naga_ext")]
            "rgb10a2uint" => Ok(Self::Rgb10a2Uint),
            #[cfg(feature = "naga_ext")]
            "rgb10a2unorm" => Ok(Self::Rgb10a2Unorm),
            #[cfg(feature = "naga_ext")]
            "rg11b10float" => Ok(Self::Rg11b10Float),
            #[cfg(feature = "naga_ext")]
            "r64uint" => Ok(Self::R64Uint),
            #[cfg(feature = "naga_ext")]
            "rgba16unorm" => Ok(Self::Rgba16Unorm),
            #[cfg(feature = "naga_ext")]
            "rgba16snorm" => Ok(Self::Rgba16Snorm),
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
