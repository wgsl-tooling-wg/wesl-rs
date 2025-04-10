use std::str::FromStr;

use crate::{
    error::Error,
    lexer::Lexer,
    syntax::{Expression, GlobalDeclaration, GlobalDirective, Statement, TranslationUnit},
};

use lalrpop_util::lalrpop_mod;

lalrpop_mod!(
    #[allow(clippy::all)]
    wgsl
);
lalrpop_mod!(
    #[allow(clippy::all)]
    wgsl_recognize
);

/// Parse a string into a syntax tree ([`TranslationUnit`]).
///
/// Identical to [`TranslationUnit::from_str`].
pub fn parse_str(source: &str) -> Result<TranslationUnit, Error> {
    let lexer = Lexer::new(source);
    let parser = wgsl::TranslationUnitParser::new();
    parser.parse(lexer).map_err(Into::into)
}

/// Test whether a string represent a valid WGSL module ([`TranslationUnit`]).
///
/// Warning: it does not take WESL extensions into account.
pub fn recognize_str(source: &str) -> Result<(), Error> {
    let lexer = Lexer::new(source);
    let parser = wgsl_recognize::TranslationUnitParser::new();
    parser.parse(lexer).map_err(Into::into)
}

pub(crate) fn recognize_template_list(lexer: &mut Lexer) -> Result<(), Error> {
    let parser = wgsl::TryTemplateListParser::new();
    parser.parse(lexer).map(|_| ()).map_err(Into::into)
}

impl FromStr for TranslationUnit {
    type Err = Error;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let lexer = Lexer::new(source);
        let parser = wgsl::TranslationUnitParser::new();
        parser.parse(lexer).map_err(Into::into)
    }
}
impl FromStr for GlobalDirective {
    type Err = Error;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let lexer = Lexer::new(source);
        let parser = wgsl::GlobalDirectiveParser::new();
        parser.parse(lexer).map_err(Into::into)
    }
}
impl FromStr for GlobalDeclaration {
    type Err = Error;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let lexer = Lexer::new(source);
        let parser = wgsl::GlobalDeclParser::new();
        parser.parse(lexer).map_err(Into::into)
    }
}
impl FromStr for Statement {
    type Err = Error;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let lexer = Lexer::new(source);
        let parser = wgsl::StatementParser::new();
        parser.parse(lexer).map_err(Into::into)
    }
}
impl FromStr for Expression {
    type Err = Error;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let lexer = Lexer::new(source);
        let parser = wgsl::ExpressionParser::new();
        parser.parse(lexer).map_err(Into::into)
    }
}
#[cfg(feature = "imports")]
impl FromStr for crate::syntax::ImportStatement {
    type Err = Error;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let lexer = Lexer::new(source);
        let parser = wgsl::ImportStatementParser::new();
        parser.parse(lexer).map_err(Into::into)
    }
}
