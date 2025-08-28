//! Built-in enumerations.

use std::str::FromStr;

use derive_more::{IsVariant, Unwrap};

use crate::ty::SampledType;

/// Memory access mode enumeration.
///
/// Reference: <https://www.w3.org/TR/WGSL/#access-mode>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

impl AccessMode {
    /// Is [`Self::Read`] or [`Self::ReadWrite`]
    pub fn is_read(&self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
    /// Is [`Self::Write`] or [`Self::ReadWrite`]
    pub fn is_write(&self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

impl FromStr for AccessMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "read_write" => Ok(Self::ReadWrite),
            _ => Err(()),
        }
    }
}

/// Address space enumeration.
///
/// Reference: <https://www.w3.org/TR/WGSL/#address-spaces>
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum AddressSpace {
    Function,
    Private,
    Workgroup,
    Uniform,
    Storage(Option<AccessMode>),
    Handle, // the handle address space cannot be spelled in WGSL.
    #[cfg(feature = "naga_ext")]
    PushConstant,
}

impl FromStr for AddressSpace {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "function" => Ok(Self::Function),
            "private" => Ok(Self::Private),
            "workgroup" => Ok(Self::Workgroup),
            "uniform" => Ok(Self::Uniform),
            "storage" => Ok(Self::Storage(None)),
            #[cfg(feature = "naga_ext")]
            "push_constant" => Ok(Self::PushConstant),
            // "WGSL predeclares an enumerant for each address space, except for the handle address space."
            // "handle" => Ok(Self::Handle),
            _ => Err(()),
        }
    }
}
/// Texel format enumeration.
///
/// Reference: <https://www.w3.org/TR/WGSL/#texel-format>
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
