use half::f16;
use itertools::Itertools;

use crate::{
    inst::{
        ArrayInstance, AtomicInstance, Instance, LiteralInstance, MatInstance, StructInstance,
        VecInstance,
    },
    ty::{Ty, Type},
};

pub trait HostShareable: Ty + Sized {
    /// Returns the memory for host-shareable types
    /// Returns None if the type is not host-shareable
    fn to_buffer(&self) -> Option<Vec<u8>>;
}

impl HostShareable for Instance {
    fn to_buffer(&self) -> Option<Vec<u8>> {
        match self {
            Instance::Literal(l) => l.to_buffer(),
            Instance::Struct(s) => s.to_buffer(),
            Instance::Array(a) => a.to_buffer(),
            Instance::Vec(v) => v.to_buffer(),
            Instance::Mat(m) => m.to_buffer(),
            Instance::Ptr(_) => None,
            Instance::Ref(_) => None,
            Instance::Atomic(a) => a.inner().to_buffer(),
            Instance::Deferred(_) => None,
        }
    }
}

impl Instance {
    pub fn from_buffer(buf: &[u8], ty: &Type) -> Option<Self> {
        match ty {
            Type::Bool => None,
            Type::AbstractInt => None,
            Type::AbstractFloat => None,
            Type::I32 => buf
                .get(..4)?
                .try_into()
                .ok()
                .map(|buf| LiteralInstance::I32(i32::from_le_bytes(buf)).into()),
            Type::U32 => buf
                .get(..4)?
                .try_into()
                .ok()
                .map(|buf| LiteralInstance::U32(u32::from_le_bytes(buf)).into()),
            Type::F32 => buf
                .get(..4)?
                .try_into()
                .ok()
                .map(|buf| LiteralInstance::F32(f32::from_le_bytes(buf)).into()),
            Type::F16 => buf
                .get(..2)?
                .try_into()
                .ok()
                .map(|buf| LiteralInstance::F16(f16::from_le_bytes(buf)).into()),
            #[cfg(feature = "naga_ext")]
            Type::I64 => buf
                .get(..8)?
                .try_into()
                .ok()
                .map(|buf| LiteralInstance::I64(i64::from_le_bytes(buf)).into()),
            #[cfg(feature = "naga_ext")]
            Type::U64 => buf
                .get(..8)?
                .try_into()
                .ok()
                .map(|buf| LiteralInstance::U64(u64::from_le_bytes(buf)).into()),
            #[cfg(feature = "naga_ext")]
            Type::F64 => buf
                .get(..8)?
                .try_into()
                .ok()
                .map(|buf| LiteralInstance::F64(f64::from_le_bytes(buf)).into()),
            Type::Struct(s) => {
                let mut offset = 0;
                let members = s
                    .members
                    .iter()
                    .map(|(name, ty)| {
                        // handle the specific case of runtime-sized arrays.
                        // they can only be the last member of a struct.
                        let inst = if let Type::Array(_, None) = ty {
                            let buf = buf.get(offset as usize..)?;
                            Instance::from_buffer(buf, ty)?
                        } else {
                            // TODO: handle errors, check valid size...
                            // TODO: since refactor, Type::Struct doesn't know about size/align attrs
                            // let size = m.attr_size().ok().flatten().or_else(|| ty.size_of())?;
                            // let align = m.attr_align().ok().flatten().or_else(|| ty.align_of())?;
                            let size = ty.size_of()?;
                            let align = ty.align_of()?;
                            offset = round_up(align, offset);
                            let buf = buf.get(offset as usize..(offset + size) as usize)?;
                            offset += size;
                            Instance::from_buffer(buf, &ty)?
                        };
                        Some((name.to_string(), inst))
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(StructInstance::new(s.name.clone(), members).into())
            }
            Type::Array(ty, Some(n)) => {
                let mut offset = 0;
                let size = ty.size_of()?;
                let stride = round_up(ty.align_of()?, size);
                let mut comps = Vec::new();
                while comps.len() != *n {
                    let buf = buf.get(offset as usize..(offset + size) as usize)?;
                    offset += stride;
                    let inst = Instance::from_buffer(buf, ty)?;
                    comps.push(inst);
                }
                Some(ArrayInstance::new(comps, false).into())
            }
            Type::Array(ty, None) => {
                let mut offset = 0;
                let size = ty.size_of()?;
                let stride = round_up(ty.align_of()?, size);
                let n = buf.len() as u32 / stride;
                if n == 0 {
                    // arrays must not be empty
                    return None;
                }
                let comps = (0..n)
                    .map(|_| {
                        let buf = buf.get(offset as usize..(offset + size) as usize)?;
                        offset += stride;
                        Instance::from_buffer(buf, ty)
                    })
                    .collect::<Option<_>>()?;
                Some(ArrayInstance::new(comps, true).into())
            }
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(_, _) => None,
            Type::Vec(n, ty) => {
                let mut offset = 0;
                let size = ty.size_of()?;
                let comps = (0..*n)
                    .map(|_| {
                        let buf = buf.get(offset as usize..(offset + size) as usize)?;
                        offset += size;
                        Instance::from_buffer(buf, ty)
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(VecInstance::new(comps).into())
            }
            Type::Mat(c, r, ty) => {
                let mut offset = 0;
                let col_ty = Type::Vec(*r, ty.clone());
                let col_size = col_ty.size_of()?;
                let col_off = round_up(col_ty.align_of()?, col_size);
                let cols = (0..*c)
                    .map(|_| {
                        let buf = buf.get(offset as usize..(offset + col_size) as usize)?;
                        offset += col_off;
                        Instance::from_buffer(buf, &col_ty)
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(MatInstance::from_cols(cols).into())
            }
            Type::Atomic(ty) => {
                let buf = buf.get(..4)?.try_into().ok()?;
                let inst = match &**ty {
                    Type::I32 => LiteralInstance::I32(i32::from_le_bytes(buf)).into(),
                    Type::U32 => LiteralInstance::U32(u32::from_le_bytes(buf)).into(),
                    _ => unreachable!("atomic type must be u32 or i32"),
                };
                Some(AtomicInstance::new(inst).into())
            }
            Type::Ptr(_, _) | Type::Texture(_) | Type::Sampler(_) => None,
        }
    }
}

impl HostShareable for LiteralInstance {
    fn to_buffer(&self) -> Option<Vec<u8>> {
        match self {
            LiteralInstance::Bool(_) => None,
            LiteralInstance::AbstractInt(_) => None,
            LiteralInstance::AbstractFloat(_) => None,
            LiteralInstance::I32(n) => Some(n.to_le_bytes().to_vec()),
            LiteralInstance::U32(n) => Some(n.to_le_bytes().to_vec()),
            LiteralInstance::F32(n) => Some(n.to_le_bytes().to_vec()),
            LiteralInstance::F16(n) => Some(n.to_le_bytes().to_vec()),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::I64(n) => Some(n.to_le_bytes().to_vec()),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::U64(n) => Some(n.to_le_bytes().to_vec()),
            #[cfg(feature = "naga_ext")]
            LiteralInstance::F64(n) => Some(n.to_le_bytes().to_vec()),
        }
    }
}

// TODO: layout
impl HostShareable for StructInstance {
    fn to_buffer(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        for (i, (_, inst)) in self.iter_members().enumerate() {
            let ty = inst.ty();
            let len = buf.len() as u32;
            // TODO: since refactor, Type::Struct doesn't know about size/align attrs
            // let size = m.attr_size().ok().flatten().or_else(|| ty.min_size_of())?;
            let size = ty.min_size_of()?;

            // handle runtime-size arrays as last struct member
            let size = match inst {
                Instance::Array(a) if a.runtime_sized => {
                    (i == self.iter_members().count() - 1).then(|| a.n() as u32 * size)
                }
                _ => Some(size),
            }?;

            // TODO: since refactor, Type::Struct doesn't know about size/align attrs
            // let align = m.attr_align().ok().flatten().or_else(|| ty.align_of())?;
            let align = ty.align_of()?;
            let off = round_up(align, len);
            if off > len {
                buf.extend((len..off).map(|_| 0));
            }
            let mut bytes = inst.to_buffer()?;
            let bytes_len = bytes.len() as u32;
            if size > bytes_len {
                bytes.extend((bytes_len..size).map(|_| 0));
            }
            buf.extend(bytes);
        }
        Some(buf)
    }
}

impl HostShareable for ArrayInstance {
    fn to_buffer(&self) -> Option<Vec<u8>> {
        let mut buf = Vec::new();
        let ty = self.inner_ty();
        let size = ty.size_of()?;
        let stride = round_up(ty.align_of()?, size);
        for c in self.iter() {
            buf.extend(c.to_buffer()?);
            if stride > size {
                buf.extend((size..stride).map(|_| 0))
            }
        }
        Some(buf)
    }
}

impl HostShareable for VecInstance {
    fn to_buffer(&self) -> Option<Vec<u8>> {
        Some(
            self.iter()
                .flat_map(|v| v.to_buffer().unwrap(/* SAFETY: vector elements must be host-shareable */).into_iter())
                .collect_vec(),
        )
    }
}

impl HostShareable for MatInstance {
    fn to_buffer(&self) -> Option<Vec<u8>> {
        Some(
            self.iter_cols()
                .flat_map(|v| {
                    // SAFETY: vector elements must be host-shareable
                    let mut v_buf = v.to_buffer().unwrap();
                    let len = v_buf.len() as u32;
                    let align = v.ty().align_of().unwrap();
                    if len < align {
                        v_buf.extend((len..align).map(|_| 0));
                    }
                    v_buf.into_iter()
                })
                .collect_vec(),
        )
    }
}

pub fn round_up(align: u32, size: u32) -> u32 {
    size.div_ceil(align) * align
}

impl Type {
    pub fn size_of(&self) -> Option<u32> {
        match self {
            Type::Bool => Some(4),
            Type::AbstractInt => None,
            Type::AbstractFloat => None,
            Type::I32 => Some(4),
            Type::U32 => Some(4),
            Type::F32 => Some(4),
            Type::F16 => Some(2),
            #[cfg(feature = "naga_ext")]
            Type::I64 => Some(8),
            #[cfg(feature = "naga_ext")]
            Type::U64 => Some(8),
            #[cfg(feature = "naga_ext")]
            Type::F64 => Some(8),
            Type::Struct(s) => {
                let past_last_mem = s
                    .members
                    .iter()
                    .map(|(_, ty)| {
                        // TODO: handle errors, check valid size...
                        // TODO: since refactor, Type::Struct doesn't know about size/align attrs
                        // let size = m.attr_size().ok().flatten().or_else(|| ty.size_of())?;
                        // let align = m.attr_align().ok().flatten().or_else(|| ty.align_of())?;
                        let size = ty.size_of()?;
                        let align = ty.align_of()?;
                        Some((size, align))
                    })
                    .try_fold(0, |offset, mem| {
                        let (size, align) = mem?;
                        Some(round_up(align, offset) + size)
                    })?;
                Some(round_up(self.align_of()?, past_last_mem))
            }
            Type::Array(ty, Some(n)) => {
                let (size, align) = (ty.size_of()?, ty.align_of()?);
                Some(*n as u32 * round_up(align, size))
            }
            Type::Array(_, None) => None,
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(_, _) => None,
            Type::Vec(n, ty) => {
                let size = ty.size_of()?;
                Some(*n as u32 * size)
            }
            Type::Mat(c, r, ty) => {
                let align = Type::Vec(*r, ty.clone()).align_of()?;
                Some(*c as u32 * align)
            }
            Type::Atomic(_) => Some(4),
            Type::Ptr(_, _) | Type::Texture(_) | Type::Sampler(_) => None,
        }
    }

    pub fn min_size_of(&self) -> Option<u32> {
        match self {
            Type::Array(ty, None) => Some(round_up(ty.align_of()?, ty.size_of()?)),
            _ => self.size_of(),
        }
    }

    pub fn align_of(&self) -> Option<u32> {
        match self {
            Type::Bool => Some(4),
            Type::AbstractInt => None,
            Type::AbstractFloat => None,
            Type::I32 => Some(4),
            Type::U32 => Some(4),
            Type::F32 => Some(4),
            Type::F16 => Some(2),
            #[cfg(feature = "naga_ext")]
            Type::I64 => Some(8),
            #[cfg(feature = "naga_ext")]
            Type::U64 => Some(8),
            #[cfg(feature = "naga_ext")]
            Type::F64 => Some(8),
            Type::Struct(s) => s
                .members
                .iter()
                // TODO: since refactor, Type::Struct doesn't know about size/align attrs
                // .map(|(name, ty)| m.attr_align().ok().flatten().or_else(|| ty.align_of()))
                .map(|(_, ty)| ty.align_of())
                .try_fold(0, |a, b| Some(a.max(b?))),
            Type::Array(ty, _) => ty.align_of(),
            #[cfg(feature = "naga_ext")]
            Type::BindingArray(_, _) => None,
            Type::Vec(n, ty) => {
                if *n == 3 {
                    match **ty {
                        Type::I32 | Type::U32 | Type::F32 => Some(16),
                        Type::F16 => Some(8),
                        _ => None,
                    }
                } else {
                    self.size_of()
                }
            }
            Type::Mat(_, r, ty) => Type::Vec(*r, ty.clone()).align_of(),
            Type::Atomic(_) => Some(4),
            Type::Ptr(_, _) | Type::Texture(_) | Type::Sampler(_) => None,
        }
    }
}
