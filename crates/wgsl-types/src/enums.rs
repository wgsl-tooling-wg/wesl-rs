//! Built-in enumerations.

use std::str::FromStr;

pub use wgsl_syntax::{AccessMode, AddressSpace, TexelFormat};

/// One of the predeclared enumerants.
///
/// Reference: <https://www.w3.org/TR/WGSL/#predeclared-enumerants>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enumerant {
    AccessMode(AccessMode),
    AddressSpace(AddressSpace),
    TexelFormat(TexelFormat),
}

impl FromStr for Enumerant {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        AccessMode::from_str(s)
            .map(Enumerant::AccessMode)
            .or_else(|()| AddressSpace::from_str(s).map(Enumerant::AddressSpace))
            .or_else(|()| TexelFormat::from_str(s).map(Enumerant::TexelFormat))
    }
}
