mod display;
mod error;
mod mem;

pub mod builtin;
pub mod conv;
pub mod enums;
pub mod idents;
pub mod inst;
pub mod tplt;
pub mod ty;

pub use error::Error;

use inst::Instance;
use tplt::TpltParam;
use ty::Type;

/// Function call signature.
#[derive(Clone, Debug, PartialEq)]
pub struct CallSignature {
    pub name: String,
    pub tplt: Option<Vec<TpltParam>>,
    pub args: Vec<Type>,
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
