use derive_more::IsVariant;
use std::{fmt::Display, str::FromStr};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "tokrepr")]
use tokrepr::TokRepr;

/// Address space enumeration.
///
/// Reference: <https://www.w3.org/TR/WGSL/#address-spaces>
#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum AddressSpace {
    Function,
    Private,
    Workgroup,
    Uniform,
    Storage,
    Handle, // the handle address space cannot be spelled in WGSL.
    #[cfg(feature = "naga_ext")]
    PushConstant,
}

/// Memory access mode enumeration.
///
/// Reference: <https://www.w3.org/TR/WGSL/#access-mode>
#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// Texel format enumeration.
///
/// Reference: <https://www.w3.org/TR/WGSL/#texel-format>
#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
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

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, IsVariant)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Off,
}

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum BuiltinValue {
    VertexIndex,
    InstanceIndex,
    ClipDistances, // requires WGSL extension clip_distances
    Position,
    FrontFacing,
    FragDepth,
    SampleIndex,
    SampleMask,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    SubgroupInvocationId, // requires WGSL extension subgroups
    SubgroupSize,         // requires WGSL extension subgroups
    #[cfg(feature = "naga_ext")]
    PrimitiveIndex,
    #[cfg(feature = "naga_ext")]
    ViewIndex,
}

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum InterpolationType {
    Perspective,
    Linear,
    Flat,
}

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum InterpolationSampling {
    Center,
    Centroid,
    Sample,
    First,
    Either,
}

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum UnaryOperator {
    /// `!`
    LogicalNegation,
    /// `-`
    Negation,
    /// `~`
    BitwiseComplement,
    /// `&`
    AddressOf,
    /// `*`
    Indirection,
}

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum BinaryOperator {
    /// `||`
    ShortCircuitOr,
    /// `&&`
    ShortCircuitAnd,
    /// `+`
    Addition,
    /// `-`
    Subtraction,
    /// `*`
    Multiplication,
    /// `/`
    Division,
    /// `%`
    Remainder,
    /// `==`
    Equality,
    /// `!=`
    Inequality,
    /// `<`
    LessThan,
    /// `<=`
    LessThanEqual,
    /// `>`
    GreaterThan,
    /// `>=`
    GreaterThanEqual,
    /// `|`
    BitwiseOr,
    /// `&`
    BitwiseAnd,
    /// `^`
    BitwiseXor,
    /// `<<`
    ShiftLeft,
    /// `>>`
    ShiftRight,
}

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum AssignmentOperator {
    /// `=`
    Equal,
    /// `+=`
    PlusEqual,
    /// `-=`
    MinusEqual,
    /// `*=`
    TimesEqual,
    /// `/=`
    DivisionEqual,
    /// `%=`
    ModuloEqual,
    /// `&=`
    AndEqual,
    /// `|=`
    OrEqual,
    /// `^=`
    XorEqual,
    /// `>>=`
    ShiftRightAssign,
    /// `<<=`
    ShiftLeftAssign,
}

#[cfg(feature = "naga_ext")]
#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum ConservativeDepth {
    GreaterEqual,
    LessEqual,
    Unchanged,
}

// ---------------
// implementations
// ---------------

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

#[cfg_attr(feature = "tokrepr", derive(TokRepr))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, IsVariant)]
pub enum SampledType {
    I32,
    U32,
    F32,
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

// -------------
// FromStr impls
// -------------

impl FromStr for AddressSpace {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "function" => Ok(Self::Function),
            "private" => Ok(Self::Private),
            "workgroup" => Ok(Self::Workgroup),
            "uniform" => Ok(Self::Uniform),
            "storage" => Ok(Self::Storage),
            #[cfg(feature = "naga_ext")]
            "push_constant" => Ok(Self::PushConstant),
            // "WGSL predeclares an enumerant for each address space, except for the handle address space."
            // "handle" => Ok(Self::Handle),
            _ => Err(()),
        }
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

impl FromStr for DiagnosticSeverity {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "error" => Ok(Self::Error),
            "warning" => Ok(Self::Warning),
            "info" => Ok(Self::Info),
            "off" => Ok(Self::Off),
            _ => Err(()),
        }
    }
}

impl FromStr for BuiltinValue {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vertex_index" => Ok(Self::VertexIndex),
            "instance_index" => Ok(Self::InstanceIndex),
            "clip_distances" => Ok(Self::ClipDistances),
            "position" => Ok(Self::Position),
            "front_facing" => Ok(Self::FrontFacing),
            "frag_depth" => Ok(Self::FragDepth),
            "sample_index" => Ok(Self::SampleIndex),
            "sample_mask" => Ok(Self::SampleMask),
            "local_invocation_id" => Ok(Self::LocalInvocationId),
            "local_invocation_index" => Ok(Self::LocalInvocationIndex),
            "global_invocation_id" => Ok(Self::GlobalInvocationId),
            "workgroup_id" => Ok(Self::WorkgroupId),
            "num_workgroups" => Ok(Self::NumWorkgroups),
            "subgroup_invocation_id" => Ok(Self::SubgroupInvocationId),
            "subgroup_size" => Ok(Self::SubgroupSize),
            #[cfg(feature = "naga_ext")]
            "primitive_index" => Ok(Self::PrimitiveIndex),
            #[cfg(feature = "naga_ext")]
            "view_index" => Ok(Self::ViewIndex),
            _ => Err(()),
        }
    }
}

impl FromStr for InterpolationType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "perspective" => Ok(Self::Perspective),
            "linear" => Ok(Self::Linear),
            "flat" => Ok(Self::Flat),
            _ => Err(()),
        }
    }
}

impl FromStr for InterpolationSampling {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "center" => Ok(Self::Center),
            "centroid" => Ok(Self::Centroid),
            "sample" => Ok(Self::Sample),
            "first" => Ok(Self::First),
            "either" => Ok(Self::Either),
            _ => Err(()),
        }
    }
}

#[cfg(feature = "naga_ext")]
impl FromStr for ConservativeDepth {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "greater_equal" => Ok(Self::GreaterEqual),
            "less_equal" => Ok(Self::LessEqual),
            "unchanged" => Ok(Self::Unchanged),
            _ => Err(()),
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

// -------------
// AsStr impls
// -------------

impl Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Function => write!(f, "function"),
            Self::Private => write!(f, "private"),
            Self::Workgroup => write!(f, "workgroup"),
            Self::Uniform => write!(f, "uniform"),
            Self::Storage => write!(f, "storage"),
            Self::Handle => write!(f, "handle"),
            #[cfg(feature = "naga_ext")]
            Self::PushConstant => write!(f, "push_constant"),
        }
    }
}

impl Display for AccessMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::ReadWrite => write!(f, "read_write"),
        }
    }
}

impl Display for TexelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TexelFormat::Rgba8Unorm => write!(f, "rgba8unorm"),
            TexelFormat::Rgba8Snorm => write!(f, "rgba8snorm"),
            TexelFormat::Rgba8Uint => write!(f, "rgba8uint"),
            TexelFormat::Rgba8Sint => write!(f, "rgba8sint"),
            TexelFormat::Rgba16Uint => write!(f, "rgba16uint"),
            TexelFormat::Rgba16Sint => write!(f, "rgba16sint"),
            TexelFormat::Rgba16Float => write!(f, "rgba16float"),
            TexelFormat::R32Uint => write!(f, "r32uint"),
            TexelFormat::R32Sint => write!(f, "r32sint"),
            TexelFormat::R32Float => write!(f, "r32float"),
            TexelFormat::Rg32Uint => write!(f, "rg32uint"),
            TexelFormat::Rg32Sint => write!(f, "rg32sint"),
            TexelFormat::Rg32Float => write!(f, "rg32float"),
            TexelFormat::Rgba32Uint => write!(f, "rgba32uint"),
            TexelFormat::Rgba32Sint => write!(f, "rgba32sint"),
            TexelFormat::Rgba32Float => write!(f, "rgba32float"),
            TexelFormat::Bgra8Unorm => write!(f, "bgra8unorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Unorm => write!(f, "r8unorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Snorm => write!(f, "r8snorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Uint => write!(f, "r8uint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R8Sint => write!(f, "r8sint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Unorm => write!(f, "r16unorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Snorm => write!(f, "r16snorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Uint => write!(f, "r16uint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Sint => write!(f, "r16sint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R16Float => write!(f, "r16float"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Unorm => write!(f, "rg8unorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Snorm => write!(f, "rg8snorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Uint => write!(f, "rg8uint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg8Sint => write!(f, "rg8sint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Unorm => write!(f, "rg16unorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Snorm => write!(f, "rg16snorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Uint => write!(f, "rg16uint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Sint => write!(f, "rg16sint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg16Float => write!(f, "rg16float"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgb10a2Uint => write!(f, "rgb10a2uint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgb10a2Unorm => write!(f, "rgb10a2unorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rg11b10Float => write!(f, "rg11b10float"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::R64Uint => write!(f, "r64uint"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgba16Unorm => write!(f, "rgba16unorm"),
            #[cfg(feature = "naga_ext")]
            TexelFormat::Rgba16Snorm => write!(f, "rgba16snorm"),
        }
    }
}

impl Display for BuiltinValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VertexIndex => write!(f, "vertex_index"),
            Self::InstanceIndex => write!(f, "instance_index"),
            Self::ClipDistances => write!(f, "clip_distances"),
            Self::Position => write!(f, "position"),
            Self::FrontFacing => write!(f, "front_facing"),
            Self::FragDepth => write!(f, "frag_depth"),
            Self::SampleIndex => write!(f, "sample_index"),
            Self::SampleMask => write!(f, "sample_mask"),
            Self::LocalInvocationId => write!(f, "local_invocation_id"),
            Self::LocalInvocationIndex => write!(f, "local_invocation_index"),
            Self::GlobalInvocationId => write!(f, "global_invocation_id"),
            Self::WorkgroupId => write!(f, "workgroup_id"),
            Self::NumWorkgroups => write!(f, "num_workgroups"),
            Self::SubgroupInvocationId => write!(f, "subgroup_invocation_id"),
            Self::SubgroupSize => write!(f, "subgroup_size"),
            #[cfg(feature = "naga_ext")]
            Self::PrimitiveIndex => write!(f, "primitive_index"),
            #[cfg(feature = "naga_ext")]
            Self::ViewIndex => write!(f, "view_index"),
        }
    }
}

impl Display for InterpolationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpolationType::Perspective => write!(f, "perspective"),
            InterpolationType::Linear => write!(f, "linear"),
            InterpolationType::Flat => write!(f, "flat"),
        }
    }
}

impl Display for InterpolationSampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Center => write!(f, "center"),
            Self::Centroid => write!(f, "centroid"),
            Self::Sample => write!(f, "sample"),
            Self::First => write!(f, "first"),
            Self::Either => write!(f, "either"),
        }
    }
}

impl Display for UnaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOperator::LogicalNegation => write!(f, "!"),
            UnaryOperator::Negation => write!(f, "-"),
            UnaryOperator::BitwiseComplement => write!(f, "~"),
            UnaryOperator::AddressOf => write!(f, "&"),
            UnaryOperator::Indirection => write!(f, "*"),
        }
    }
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::ShortCircuitOr => write!(f, "||"),
            BinaryOperator::ShortCircuitAnd => write!(f, "&&"),
            BinaryOperator::Addition => write!(f, "+"),
            BinaryOperator::Subtraction => write!(f, "-"),
            BinaryOperator::Multiplication => write!(f, "*"),
            BinaryOperator::Division => write!(f, "/"),
            BinaryOperator::Remainder => write!(f, "%"),
            BinaryOperator::Equality => write!(f, "=="),
            BinaryOperator::Inequality => write!(f, "!="),
            BinaryOperator::LessThan => write!(f, "<"),
            BinaryOperator::LessThanEqual => write!(f, "<="),
            BinaryOperator::GreaterThan => write!(f, ">"),
            BinaryOperator::GreaterThanEqual => write!(f, ">="),
            BinaryOperator::BitwiseOr => write!(f, "|"),
            BinaryOperator::BitwiseAnd => write!(f, "&"),
            BinaryOperator::BitwiseXor => write!(f, "^"),
            BinaryOperator::ShiftLeft => write!(f, "<<"),
            BinaryOperator::ShiftRight => write!(f, ">>"),
        }
    }
}

impl Display for AssignmentOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssignmentOperator::Equal => write!(f, "="),
            AssignmentOperator::PlusEqual => write!(f, "+="),
            AssignmentOperator::MinusEqual => write!(f, "-="),
            AssignmentOperator::TimesEqual => write!(f, "*="),
            AssignmentOperator::DivisionEqual => write!(f, "/="),
            AssignmentOperator::ModuloEqual => write!(f, "%="),
            AssignmentOperator::AndEqual => write!(f, "&="),
            AssignmentOperator::OrEqual => write!(f, "|="),
            AssignmentOperator::XorEqual => write!(f, "^="),
            AssignmentOperator::ShiftRightAssign => write!(f, ">>="),
            AssignmentOperator::ShiftLeftAssign => write!(f, "<<="),
        }
    }
}

#[cfg(feature = "naga_ext")]
impl Display for ConservativeDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GreaterEqual => write!(f, "Greater_equal"),
            Self::LessEqual => write!(f, "less_equal"),
            Self::Unchanged => write!(f, "unchanged"),
        }
    }
}

impl Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
            Self::Info => write!(f, "info"),
            Self::Off => write!(f, "off"),
        }
    }
}

impl Display for SampledType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampledType::I32 => write!(f, "i32"),
            SampledType::U32 => write!(f, "u32"),
            SampledType::F32 => write!(f, "f32"),
        }
    }
}
