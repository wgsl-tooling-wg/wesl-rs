#![doc = include_str!("../README.md")]

pub mod error;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod syntax;

mod parser_support;
mod syntax_display;
mod syntax_impl;

#[cfg(feature = "reify")]
mod reify;

pub use error::Error;
pub use parser::{parse_str, recognize_str};
pub use syntax_impl::Decorated;
