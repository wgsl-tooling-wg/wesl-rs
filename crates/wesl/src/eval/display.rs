use std::fmt::Display;

use itertools::Itertools;
use wgsl_parse::syntax::AddressSpace;

use crate::eval::Ty;

use super::{
    ArrayInstance, AtomicInstance, Instance, LiteralInstance, MatInstance, MemView, PtrInstance,
    RefInstance, SampledType, SamplerType, StructInstance, TexelFormat, TextureType, Type,
    VecInstance,
};

impl Display for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instance::Literal(inst) => write!(f, "{inst}"),
            Instance::Struct(inst) => write!(f, "{inst}"),
            Instance::Array(inst) => write!(f, "{inst}"),
            Instance::Vec(inst) => write!(f, "{inst}"),
            Instance::Mat(inst) => write!(f, "{inst}"),
            Instance::Ptr(inst) => write!(f, "{inst}"),
            Instance::Ref(inst) => write!(f, "{inst}"),
            Instance::Atomic(inst) => write!(f, "{inst}"),
            Instance::Deferred(ty) => write!(f, "__deferred<{ty}>"),
        }
    }
}

impl Display for LiteralInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiteralInstance::Bool(lit) => write!(f, "{lit}"),
            LiteralInstance::AbstractInt(lit) => write!(f, "{lit}"),
            LiteralInstance::AbstractFloat(lit) => write!(f, "{lit:?}"), // using the Debug formatter to print the trailing .0 in floats representing integers. because format!("{}", 3.0f32) == "3"
            LiteralInstance::I32(lit) => write!(f, "{lit}i"),
            LiteralInstance::U32(lit) => write!(f, "{lit}u"),
            LiteralInstance::F32(lit) => write!(f, "{lit}f"),
            LiteralInstance::F16(lit) => write!(f, "{lit}h"),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(lit) => write!(f, "{lit}li"),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(lit) => write!(f, "{lit}lu"),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(lit) => write!(f, "{lit}lf"),
        }
    }
}

impl Display for StructInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self.name();
        let comps = self
            .iter_members()
            .map(|(_, v)| format!("{v}"))
            .format(", ");
        write!(f, "{name}({comps})")
    }
}

impl Display for ArrayInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let comps = self.iter().format(", ");
        write!(f, "array({comps})")
    }
}

impl Display for VecInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.n();
        let comps = self.iter().format(", ");
        write!(f, "vec{n}({comps})")
    }
}

impl Display for MatInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.c();
        let r = self.r();
        let comps = (0..c).map(|i| self.col(i).unwrap()).format(", ");
        write!(f, "mat{c}x{r}({comps})")
    }
}

fn addr_space(space: AddressSpace) -> &'static str {
    match space {
        AddressSpace::Function => "function",
        AddressSpace::Private => "private",
        AddressSpace::Workgroup => "workgroup",
        AddressSpace::Uniform => "uniform",
        AddressSpace::Storage(_) => "storage",
        AddressSpace::Handle => "handle",
        #[cfg(feature = "naga_ext")]
        AddressSpace::PushConstant => "push_constant",
    }
}

impl Display for PtrInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let space = addr_space(self.ptr.space);
        let ty = &self.ptr.ty;
        let access = &self.ptr.access;
        let val = self.ptr.read().expect("invalid reference");
        write!(f, "ptr<{space}, {ty}, {access}>({val})")
    }
}

impl Display for RefInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let space = addr_space(self.space);
        let ty = &self.ty;
        let access = &self.access;
        let val = self.read().expect("invalid reference");
        write!(f, "ref<{space}, {ty}, {access}>({val})")
    }
}

impl Display for MemView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn fmt_view(f: &mut std::fmt::Formatter<'_>, view: &MemView) -> std::fmt::Result {
            match view {
                MemView::Whole => Ok(()),
                MemView::Member(m, v) => {
                    write!(f, ":{m}")?;
                    fmt_view(f, v)
                }
                MemView::Index(i, v) => {
                    write!(f, ":{i}")?;
                    fmt_view(f, v)
                }
            }
        }

        fmt_view(f, self)
    }
}

impl Display for AtomicInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = &self.inner_ty();
        let val = self.inner();
        write!(f, "atomic<{ty}>({val})")
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Bool => write!(f, "bool"),
            Type::AbstractInt => write!(f, "AbstractInt"),
            Type::AbstractFloat => write!(f, "AbstractFloat"),
            Type::I32 => write!(f, "i32"),
            Type::U32 => write!(f, "u32"),
            Type::F32 => write!(f, "f32"),
            Type::F16 => write!(f, "f16"),
            #[cfg(feature = "naga_ext")]
            Type::I64 => write!(f, "i64"),
            #[cfg(feature = "naga_ext")]
            Type::U64 => write!(f, "u64"),
            #[cfg(feature = "naga_ext")]
            Type::F64 => write!(f, "f64"),
            Type::Struct(name) => write!(f, "{name}"),
            Type::Array(ty, Some(n)) => write!(f, "array<{ty}, {n}>"),
            Type::Array(ty, None) => write!(f, "array<{ty}>"),
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(ty, Some(n)) => write!(f, "binding_array<{ty}, {n}>"),
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(ty, None) => write!(f, "binding_array<{ty}>"),
            Type::Vec(n, ty) => write!(f, "vec{n}<{ty}>"),
            Type::Mat(m, n, ty) => write!(f, "mat{m}x{n}<{ty}>"),
            Type::Atomic(ty) => write!(f, "atomic<{ty}>"),
            Type::Ptr(a_s, ty) => write!(f, "ptr<{a_s}, {ty}>"),
            Type::Texture(texture_type) => texture_type.fmt(f),
            Type::Sampler(sampler_type) => sampler_type.fmt(f),
        }
    }
}

impl Display for TextureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureType::Sampled1D(sampled) => write!(f, "texture_1d<{sampled}>"),
            TextureType::Sampled2D(sampled) => write!(f, "texture_2d<{sampled}>"),
            TextureType::Sampled2DArray(sampled) => write!(f, "texture_2d_array<{sampled}>"),
            TextureType::Sampled3D(sampled) => write!(f, "texture_3d<{sampled}>"),
            TextureType::SampledCube(sampled) => write!(f, "texture_cube<{sampled}>"),
            TextureType::SampledCubeArray(sampled) => write!(f, "texture_cube_array<{sampled}>"),
            TextureType::Multisampled2D(sampled) => write!(f, "texture_multisampled_2d<{sampled}>"),
            TextureType::DepthMultisampled2D => write!(f, "texture_depth_multisampled_2d"),
            TextureType::External => write!(f, "texture_external<>"),
            TextureType::Storage1D(texel, access) => {
                write!(f, "texture_storage_1d<{texel}, {access}>")
            }
            TextureType::Storage2D(texel, access) => {
                write!(f, "texture_storage_2d<{texel}, {access}>")
            }
            TextureType::Storage2DArray(texel, access) => {
                write!(f, "texture_storage_2d_array<{texel}, {access}>")
            }
            TextureType::Storage3D(texel, access) => {
                write!(f, "texture_storage_3d<{texel}, {access}>")
            }
            TextureType::Depth2D => write!(f, "texture_depth_2d"),
            TextureType::Depth2DArray => write!(f, "texture_depth_2d_array"),
            TextureType::DepthCube => write!(f, "texture_depth_cube"),
            TextureType::DepthCubeArray => write!(f, "texture_depth_cube_array"),
        }
    }
}

impl Display for SamplerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerType::Sampler => write!(f, "sampler"),
            SamplerType::SamplerComparison => write!(f, "sampler_comparison"),
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
