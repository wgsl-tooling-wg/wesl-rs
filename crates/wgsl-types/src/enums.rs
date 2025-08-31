//! Built-in enumerations.

use std::str::FromStr;

use derive_more::{From, IsVariant, Unwrap};

pub use wgsl_syntax::{AccessMode, AddressSpace, TexelFormat};

/// One of the predeclared enumerants.
///
/// Reference: <https://www.w3.org/TR/WGSL/#predeclared-enumerants>
#[derive(Clone, Copy, Debug, PartialEq, Eq, From, IsVariant, Unwrap)]
pub enum Enumerant {
    AccessMode(AccessMode),
    AddressSpace(AddressSpace),
    TexelFormat(TexelFormat),
}

impl FromStr for Enumerant {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        AccessMode::from_str(s)
            .map(Into::into)
            .or_else(|()| AddressSpace::from_str(s).map(Into::into))
            .or_else(|()| TexelFormat::from_str(s).map(Into::into))
    }
}
