mod display;
mod error;
mod mem;
mod ops;

pub mod builtin;
pub mod conv;
pub mod enums;
pub mod idents;
pub mod inst;
pub mod ty;

pub use error::*;

use inst::Instance;
use ty::Type;

/// A single tempate parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum TpltParam {
    Type(Type),
    Instance(Instance),
    Enumerant(String),
}

/// Function call signature.
#[derive(Clone, Debug, PartialEq)]
pub struct CallSignature {
    name: String,
    tplt: Option<Vec<TpltParam>>,
    args: Vec<Type>,
}

/// Shader compilation stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    /// Shader module creation
    Const,
    /// Pipeline creation
    Override,
    /// Shader execution
    Exec,
}
