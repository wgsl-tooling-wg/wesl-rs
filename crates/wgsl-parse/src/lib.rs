#![doc = include_str!("../README.md")]

pub mod error;
pub mod span;
pub mod syntax;

mod lexer;
mod parser;
mod parser_support;
mod syntax_display;
mod syntax_impl;

pub use error::Error;
pub use parser::{parse_str, recognize_str};
pub use syntax_impl::Decorated;
