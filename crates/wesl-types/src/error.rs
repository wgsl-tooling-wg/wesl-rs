use thiserror::Error;

use crate::{CallSignature, Type, inst::LiteralInstance};

/// Evaluation and Execution errors.
#[derive(Clone, Debug, Error)]
pub enum EvalError {
    #[error("not implemented: `{0}`")]
    Todo(String),

    // types & templates
    #[error("expected a scalar type, got `{0}`")]
    NotScalar(Type),
    #[error("invalid sampled type, expected `i32`, `u32` of `f32`, got `{0}`")]
    SampledType(Type),

    // references
    #[error("cannot write a `{0}` to a reference to `{1}`")]
    WriteRefType(Type, Type),
    #[error("attempt to write to a read-only reference")]
    NotWrite,
    #[error("attempt to read a write-only reference")]
    NotRead,
    #[error("reference is not read-write")]
    NotReadWrite,

    // conversions
    #[error("cannot convert from `{0}` to `{1}`")]
    Conversion(Type, Type),
    #[error("overflow while converting `{0}` to `{1}`")]
    ConvOverflow(LiteralInstance, Type),

    // indexing
    #[error("`{0}` has no component `{1}`")]
    Component(Type, String),
    #[error("`{0}` cannot be indexed")]
    NotIndexable(Type),
    #[error("index `{0}` is out-of-bounds for `{1}` of `{2}` components")]
    OutOfBounds(usize, Type, usize),

    // arithmetic
    #[error("cannot use unary operator `{0}` on type `{1}`")]
    Unary(&'static str, Type),
    #[error("cannot use binary operator `{0}` with operands `{1}` and `{2}`")]
    Binary(&'static str, Type, Type),
    #[error("cannot apply component-wise binary operation on operands `{0}` and `{1}`")]
    CompwiseBinary(Type, Type),
    #[error("attempt to add with overflow")]
    AddOverflow,
    #[error("attempt to subtract with overflow")]
    SubOverflow,
    #[error("attempt to multiply with overflow")]
    MulOverflow,
    #[error("attempt to divide by zero")]
    DivByZero,
    #[error("attempt to calculate the remainder with a divisor of zero")]
    RemZeroDiv,
    #[error("attempt to shift left by `{0}`, which would overflow `{1}`")]
    ShlOverflow(u32, LiteralInstance),
    #[error("attempt to shift right by `{0}`, which would overflow `{1}`")]
    ShrOverflow(u32, LiteralInstance),

    // functions
    #[error("invalid function call signature: `{0}`")]
    Signature(CallSignature),
    #[error("{0}")]
    Builtin(&'static str),
    #[error("invalid template arguments to `{0}`")]
    TemplateArgs(&'static str),
    #[error("incorrect number of arguments to `{0}`, expected `{1}`, got `{2}`")]
    ParamCount(String, usize, usize),
    #[error("invalid parameter type, expected `{0}`, got `{1}`")]
    ParamType(Type, Type),
}
