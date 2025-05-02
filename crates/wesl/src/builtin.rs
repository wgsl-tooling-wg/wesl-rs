use std::{collections::HashMap, sync::LazyLock};

use itertools::chain;
use wgsl_parse::syntax::*;

/// Builtin WGSL types.
/// reference: https://www.w3.org/TR/WGSL/#predeclared-types
pub const BUILTIN_TYPE_NAMES: &[&str] = &[
    // abstract types cannot be spelled in user code.
    // they are prefixed with `__`, which is not a valid WGSL identifier prefix.
    "__AbstractInt",
    "__AbstractFloat",
    // plain types
    "bool",
    "f16",
    "f32",
    "i32",
    "sampler",
    "sampler_comparison",
    "texture_depth_2d",
    "texture_depth_2d_array",
    "texture_depth_cube",
    "texture_depth_cube_array",
    "texture_depth_multisampled_2d",
    "texture_external",
    "u32",
    // naga extensions
    #[cfg(feature = "naga_ext")]
    "i64",
    #[cfg(feature = "naga_ext")]
    "u64",
    #[cfg(feature = "naga_ext")]
    "f64",
    #[cfg(feature = "naga_ext")]
    "binding_array",
];

/// reference: https://www.w3.org/TR/WGSL/#predeclared-types
pub const BUILTIN_TYPE_GENERATOR_NAMES: &[&str] = &[
    "array",
    "atomic",
    "mat2x2",
    "mat2x3",
    "mat2x4",
    "mat3x2",
    "mat3x3",
    "mat3x4",
    "mat4x2",
    "mat4x3",
    "mat4x4",
    "ptr",
    "texture_1d",
    "texture_2d",
    "texture_2d_array",
    "texture_3d",
    "texture_cube",
    "texture_cube_array",
    "texture_multisampled_2d",
    "texture_storage_1d",
    "texture_storage_2d",
    "texture_storage_2d_array",
    "texture_storage_3d",
    "vec2",
    "vec3",
    "vec4",
];

/// Builtin struct types in WGSL.
pub const BUILTIN_STRUCT_NAMES: &[&str] = &[
    // function return types cannot be spelled in user code.
    // they are prefixed with `__`, which is not a valid WGSL identifier prefix.
    "__frexp_result_f32",
    "__frexp_result_f16",
    "__frexp_result_abstract",
    "__frexp_result_vec2_f32",
    "__frexp_result_vec3_f32",
    "__frexp_result_vec4_f32",
    "__frexp_result_vec2_f16",
    "__frexp_result_vec3_f16",
    "__frexp_result_vec4_f16",
    "__frexp_result_vec2_abstract",
    "__frexp_result_vec3_abstract",
    "__frexp_result_vec4_abstract",
    "__modf_result_f32",
    "__modf_result_f16",
    "__modf_result_abstract",
    "__modf_result_vec2_f32",
    "__modf_result_vec3_f32",
    "__modf_result_vec4_f32",
    "__modf_result_vec2_f16",
    "__modf_result_vec3_f16",
    "__modf_result_vec4_f16",
    "__modf_result_vec2_abstract",
    "__modf_result_vec3_abstract",
    "__modf_result_vec4_abstract",
    "__atomic_compare_exchange_result",
];

pub const BUILTIN_DECLARATION_NAMES: &[&str] = &[
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_NONE",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_FORCE_OPAQUE",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_FORCE_NO_OPAQUE",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_TERMINATE_ON_FIRST_HIT",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_CULL_BACK_FACING",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_CULL_FRONT_FACING",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_CULL_OPAQUE",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_CULL_NO_OPAQUE",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_SKIP_TRIANGLES",
    #[cfg(feature = "naga_ext")]
    "RAY_FLAG_SKIP_AABBS",
    #[cfg(feature = "naga_ext")]
    "RAY_QUERY_INTERSECTION_NONE",
    #[cfg(feature = "naga_ext")]
    "RAY_QUERY_INTERSECTION_TRIANGLE",
    #[cfg(feature = "naga_ext")]
    "RAY_QUERY_INTERSECTION_GENERATED",
    #[cfg(feature = "naga_ext")]
    "RAY_QUERY_INTERSECTION_AABB",
];

pub const BUILTIN_ALIAS_NAMES: &[&str] = &[
    "vec2i", "vec3i", "vec4i", "vec2u", "vec3u", "vec4u", "vec2f", "vec3f", "vec4f", "vec2h",
    "vec3h", "vec4h", "mat2x2f", "mat2x3f", "mat2x4f", "mat3x2f", "mat3x3f", "mat3x4f", "mat4x2f",
    "mat4x3f", "mat4x4f", "mat2x2h", "mat2x3h", "mat2x4h", "mat3x2h", "mat3x3h", "mat3x4h",
    "mat4x2h", "mat4x3h", "mat4x4h",
];

/// reference: https://www.w3.org/TR/WGSL/#predeclared-enumerants
/// These enumerants are not context-dependent names, and can therefore be shadowed.
pub const BUILTIN_ENUMERANT_NAMES: &[&str] = &[
    // : access mode
    "read",
    "write",
    "read_write",
    // : address space (enumerant)
    "function",
    "private",
    "workgroup",
    "uniform",
    "storage",
    #[cfg(feature = "naga_ext")]
    "push_constant",
    // : texel format
    "rgba8unorm",
    "rgba8snorm",
    "rgba8uint",
    "rgba8sint",
    "rgba16uint",
    "rgba16sint",
    "rgba16float",
    "r32uint",
    "r32sint",
    "r32float",
    "rg32uint",
    "rg32sint",
    "rg32float",
    "rgba32uint",
    "rgba32sint",
    "rgba32float",
    "bgra8unorm",
    #[cfg(feature = "naga_ext")]
    "r8unorm",
    #[cfg(feature = "naga_ext")]
    "r8snorm",
    #[cfg(feature = "naga_ext")]
    "r8uint",
    #[cfg(feature = "naga_ext")]
    "r8sint",
    #[cfg(feature = "naga_ext")]
    "r16unorm",
    #[cfg(feature = "naga_ext")]
    "r16snorm",
    #[cfg(feature = "naga_ext")]
    "r16uint",
    #[cfg(feature = "naga_ext")]
    "r16sint",
    #[cfg(feature = "naga_ext")]
    "r16float",
    #[cfg(feature = "naga_ext")]
    "rg8unorm",
    #[cfg(feature = "naga_ext")]
    "rg8snorm",
    #[cfg(feature = "naga_ext")]
    "rg8uint",
    #[cfg(feature = "naga_ext")]
    "rg8sint",
    #[cfg(feature = "naga_ext")]
    "rg16unorm",
    #[cfg(feature = "naga_ext")]
    "rg16snorm",
    #[cfg(feature = "naga_ext")]
    "rg16uint",
    #[cfg(feature = "naga_ext")]
    "rg16sint",
    #[cfg(feature = "naga_ext")]
    "rg16float",
    #[cfg(feature = "naga_ext")]
    "rgb10a2uint",
    #[cfg(feature = "naga_ext")]
    "rgb10a2unorm",
    #[cfg(feature = "naga_ext")]
    "rg11b10float",
    #[cfg(feature = "naga_ext")]
    "r64uint",
    #[cfg(feature = "naga_ext")]
    "rgba16unorm",
    #[cfg(feature = "naga_ext")]
    "rgba16snorm",
];

/// Builtin functions in WGSL, excluding builtin constructors.
/// Reference: https://www.w3.org/TR/WGSL/#builtin-functions
pub const BUILTIN_FUNCTION_NAMES: &[&str] = &[
    // : bitcast
    "bitcast",
    // : logical
    "all",
    "any",
    "select",
    "arrayLength",
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "atan2",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "countLeadingZeros",
    "countOneBits",
    "countTrailingZeros",
    "cross",
    "degrees",
    "determinant",
    "distance",
    "dot",
    "dot4U8Packed",
    "exp",
    "exp2",
    "extractBits",
    "faceForward",
    "firstLeadingBit",
    "firstTrailingBit",
    "floor",
    "fma",
    "fract",
    "frexp",
    "insertBits",
    "inverseSqrt",
    "ldexp",
    "length",
    "log",
    "log2",
    "max",
    "min",
    "mix",
    "modf",
    "normalize",
    "pow",
    "quantizeToF16",
    "radians",
    "reflect",
    "refract",
    "reverseBits",
    "round",
    "saturate",
    "sign",
    "sin",
    "sinh",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    // : derivative
    "dpdx",
    "dpdxCoarse",
    "dpdxFine",
    "dpdy",
    "dpdyCoarse",
    "dpdyFine",
    "fwidth",
    "fwidthCoarse",
    "fwidthFine",
    // : texture
    "textureDimensions",
    "textureGather",
    "textureGatherCompare",
    "textureLoad",
    "textureNumLayers",
    "textureNumLevels",
    "textureNumSamples",
    "textureSample",
    "textureSampleBias",
    "textureSampleCompare",
    "textureSampleCompareLevel",
    "textureSampleGrad",
    "textureSampleLevel",
    "textureSampleBaseClampToEdge",
    "textureStore",
    // : atomic
    "atomicLoad",
    "atomicStore",
    "atomicAdd",
    "atomicSub",
    "atomicMax",
    "atomicMin",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "atomicExchange",
    "atomicCompareExchangeWeak",
    // : packing
    "pack4x8snorm",
    "pack4x8unorm",
    "pack4xI8",
    "pack4xU8",
    "pack4xI8Clamp",
    "pack4xU8Clamp",
    "pack2x16snorm",
    "pack2x16unorm",
    "pack2x16float",
    "unpack4x8snorm",
    "unpack4x8unorm",
    "unpack4xI8",
    "unpack4xU8",
    "unpack2x16snorm",
    "unpack2x16unorm",
    "unpack2x16float",
    // : synchronization
    "storageBarrier",
    "textureBarrier",
    "workgroupBarrier",
    "workgroupUniformLoad",
    // : subgroup
    "subgroupAdd",
    "subgroupExclusiveAdd",
    "subgroupInclusiveAdd",
    "subgroupAll",
    "subgroupAnd",
    "subgroupAny",
    "subgroupBallot",
    "subgroupBroadcast",
    "subgroupBroadcastFirst",
    "subgroupElect",
    "subgroupMax",
    "subgroupMin",
    "subgroupMul",
    "subgroupExclusiveMul",
    "subgroupInclusiveMul",
    "subgroupOr",
    "subgroupShuffle",
    "subgroupShuffleDown",
    "subgroupShuffleUp",
    "subgroupShuffleXor",
    "subgroupXor",
    // : quad
    "quadBroadcast",
    "quadSwapDiagonal",
    "quadSwapX",
    "quadSwapY",
    // : ray queries
];

/// Builtin constructors (zero-value and value-constructors).
/// Reference: https://www.w3.org/TR/WGSL/#constructor-builtin-function
pub const BUILTIN_CONSTRUCTOR_NAMES: &[&str] = &[
    // constructor built-in functions
    "bool",
    "f16",
    "f32",
    "i32",
    "u32",
    #[cfg(feature = "naga_ext")]
    "i64",
    #[cfg(feature = "naga_ext")]
    "u64",
    #[cfg(feature = "naga_ext")]
    "f64",
    // type-generators
    "array",
    "mat2x2",
    "mat2x3",
    "mat2x4",
    "mat3x2",
    "mat3x3",
    "mat3x4",
    "mat4x2",
    "mat4x3",
    "mat4x4",
    "vec2",
    "vec3",
    "vec4",
    // predeclared aliases
    "vec2i",
    "vec3i",
    "vec4i",
    "vec2u",
    "vec3u",
    "vec4u",
    "vec2f",
    "vec3f",
    "vec4f",
    "vec2h",
    "vec3h",
    "vec4h",
    "mat2x2f",
    "mat2x3f",
    "mat2x4f",
    "mat3x2f",
    "mat3x3f",
    "mat3x4f",
    "mat4x2f",
    "mat4x3f",
    "mat4x4f",
    "mat2x2h",
    "mat2x3h",
    "mat2x4h",
    "mat3x2h",
    "mat3x3h",
    "mat3x4h",
    "mat4x2h",
    "mat4x3h",
    "mat4x4h",
];

/// All built-in names as [`Ident`]s.
///
/// Using these idents allow better use-count tracking and referencing.
pub static BUILTIN_IDENTS: LazyLock<HashMap<&str, Ident>> = LazyLock::new(|| {
    macro_rules! ident {
        ($name:expr) => {
            ($name, Ident::new($name.to_string()))
        };
    }
    HashMap::from_iter(chain!(
        BUILTIN_TYPE_NAMES.iter().map(|id| ident!(*id)),
        BUILTIN_TYPE_GENERATOR_NAMES.iter().map(|id| ident!(*id)),
        BUILTIN_STRUCT_NAMES.iter().map(|id| ident!(*id)),
        BUILTIN_DECLARATION_NAMES.iter().map(|id| ident!(*id)),
        BUILTIN_ALIAS_NAMES.iter().map(|id| ident!(*id)),
        BUILTIN_ENUMERANT_NAMES.iter().map(|id| ident!(*id)),
        BUILTIN_FUNCTION_NAMES.iter().map(|id| ident!(*id)),
        BUILTIN_CONSTRUCTOR_NAMES.iter().map(|id| ident!(*id)),
    ))
});

/// Get a built-in WGSL name as [`Ident`].
///
/// Using these idents allow better use-count tracking and referencing.
pub fn builtin_ident(name: &str) -> Option<&'static Ident> {
    BUILTIN_IDENTS.get(name)
}

/// Get the name of the type corresponding to a literal suffix.
pub fn litteral_suffix_type(suffix: &str) -> Option<&'static str> {
    match suffix {
        "i" => Some("i32"),
        "u" => Some("u32"),
        "f" => Some("f32"),
        "h" => Some("f16"),
        #[cfg(feature = "naga_ext")]
        "li" => Some("i64"),
        #[cfg(feature = "naga_ext")]
        "lu" => Some("u64"),
        #[cfg(feature = "naga_ext")]
        "lf" => Some("f64"),
        _ => None,
    }
}
