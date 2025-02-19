use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};

use itertools::Itertools;
use thiserror::Error;

use crate::{lexer::Token, span::Span};

/// WGSL parse error kind.
#[derive(Error, Clone, Debug, PartialEq)]
pub enum ErrorKind {
    #[error("invalid token")]
    InvalidToken,
    #[error("unexpected token `{token}`, expected `{}`", .expected.iter().format(", "))]
    UnexpectedToken {
        token: String,
        expected: Vec<String>,
    },
    #[error("unexpected end of file, expected `{}`", .expected.iter().format(", "))]
    UnexpectedEof { expected: Vec<String> },
    #[error("extra token `{0}` at the end of the file")]
    ExtraToken(String),
    #[error("invalid diagnostic severity")]
    DiagnosticSeverity,
    #[error("invalid `{0}` attribute, {1}")]
    Attribute(&'static str, &'static str),
    #[error("invalid `var` template arguments, {0}")]
    VarTemplate(&'static str),
}

#[derive(Default, Clone, Debug, PartialEq)]
pub(crate) enum CustomLalrError {
    #[default]
    LexerError,
    DiagnosticSeverity,
    Attribute(&'static str, &'static str),
    VarTemplate(&'static str),
}

type LalrError = lalrpop_util::ParseError<usize, Token, (usize, CustomLalrError, usize)>;

/// WGSL parse error.
///
/// This error can be pretty-printed with the source snippet with [`Error::with_source`]
#[derive(Error, Clone, Debug, PartialEq)]
pub struct Error {
    pub error: ErrorKind,
    pub span: Span,
}

impl Error {
    /// Returns an [`ErrorWithSource`], a wrapper type that implements `Display` and prints
    /// a user-friendly error snippet.
    pub fn with_source(self, source: Cow<'_, str>) -> ErrorWithSource<'_> {
        ErrorWithSource::new(self, source)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "chars {:?}: {}", self.span.range(), self.error)
    }
}

impl From<LalrError> for Error {
    fn from(err: LalrError) -> Self {
        match err {
            LalrError::InvalidToken { location } => {
                let span = Span::new(location..location + 1);
                let error = ErrorKind::InvalidToken;
                Self { span, error }
            }
            LalrError::UnrecognizedEof { location, expected } => {
                let span = Span::new(location..location + 1);
                let error = ErrorKind::UnexpectedEof { expected };
                Self { span, error }
            }
            LalrError::UnrecognizedToken {
                token: (l, token, r),
                expected,
            } => {
                let span = Span::new(l..r);
                let error = ErrorKind::UnexpectedToken {
                    token: token.to_string(),
                    expected,
                };
                Self { span, error }
            }
            LalrError::ExtraToken {
                token: (l, token, r),
            } => {
                let span = Span::new(l..r);
                let error = ErrorKind::ExtraToken(token.to_string());
                Self { span, error }
            }
            LalrError::User {
                error: (l, error, r),
            } => {
                let span = Span::new(l..r);
                let error = match error {
                    CustomLalrError::DiagnosticSeverity => ErrorKind::DiagnosticSeverity,
                    CustomLalrError::LexerError => ErrorKind::InvalidToken,
                    CustomLalrError::Attribute(attr, expected) => {
                        ErrorKind::Attribute(attr, expected)
                    }
                    CustomLalrError::VarTemplate(reason) => ErrorKind::VarTemplate(reason),
                };
                Self { span, error }
            }
        }
    }
}

/// A wrapper type that implements `Display` and prints a user-friendly error snippet.
#[derive(Clone, Debug, PartialEq)]
pub struct ErrorWithSource<'s> {
    pub error: Error,
    pub source: Cow<'s, str>,
}

impl std::error::Error for ErrorWithSource<'_> {}

impl<'s> ErrorWithSource<'s> {
    pub fn new(error: Error, source: Cow<'s, str>) -> Self {
        Self { error, source }
    }
}

impl Display for ErrorWithSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use annotate_snippets::*;
        let text = format!("{}", self.error.error);

        let annot = Level::Info.span(self.error.span.range());
        let snip = Snippet::source(&self.source).fold(true).annotation(annot);
        let msg = Level::Error.title(&text).snippet(snip);

        let renderer = Renderer::styled();
        let rendered = renderer.render(msg);
        write!(f, "{rendered}")
    }
}
