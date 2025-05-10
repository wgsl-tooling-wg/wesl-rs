use proc_macro_error::abort;
use proc_macro2::{Ident, Literal, Punct, Spacing, Span, TokenStream};
use token_stream_flatten::{Delimiter, DelimiterKind, DelimiterPosition, FlattenRec, Token};
use wgsl_parse::{Reify, lexer::Token as WeslTok, parser::TranslationUnitParser};

struct Lexer {
    it: FlattenRec,
}

fn delim2tok(delim: &Delimiter) -> WeslTok {
    match (delim.kind(), delim.position()) {
        (DelimiterKind::Brace, DelimiterPosition::Open) => WeslTok::SymBraceLeft,
        (DelimiterKind::Brace, DelimiterPosition::Close) => WeslTok::SymBraceRight,
        (DelimiterKind::Bracket, DelimiterPosition::Open) => WeslTok::SymBracketLeft,
        (DelimiterKind::Bracket, DelimiterPosition::Close) => WeslTok::SymBracketRight,
        (DelimiterKind::Parenthesis, DelimiterPosition::Open) => WeslTok::SymParenLeft,
        (DelimiterKind::Parenthesis, DelimiterPosition::Close) => WeslTok::SymParenRight,
    }
}

fn ident2tok(ident: Ident) -> WeslTok {
    let repr = ident.to_string();
    match repr.as_str() {
        "alias" => WeslTok::KwAlias,
        "break" => WeslTok::KwBreak,
        "case" => WeslTok::KwCase,
        "const" => WeslTok::KwConst,
        "const_assert" => WeslTok::KwConstAssert,
        "continue" => WeslTok::KwContinue,
        "continuing" => WeslTok::KwContinuing,
        "default" => WeslTok::KwDefault,
        "diagnostic" => WeslTok::KwDiagnostic,
        "discard" => WeslTok::KwDiscard,
        "else" => WeslTok::KwElse,
        "enable" => WeslTok::KwEnable,
        "false" => WeslTok::KwFalse,
        "fn" => WeslTok::KwFn,
        "for" => WeslTok::KwFor,
        "if" => WeslTok::KwIf,
        "let" => WeslTok::KwLet,
        "loop" => WeslTok::KwLoop,
        "override" => WeslTok::KwOverride,
        "requires" => WeslTok::KwRequires,
        "return" => WeslTok::KwReturn,
        "struct" => WeslTok::KwStruct,
        "switch" => WeslTok::KwSwitch,
        "true" => WeslTok::KwTrue,
        "var" => WeslTok::KwVar,
        "while" => WeslTok::KwWhile,
        // #[cfg(feature = "imports")]
        "self" => WeslTok::KwSelf,
        // #[cfg(feature = "imports")]
        "super" => WeslTok::KwSuper,
        // #[cfg(feature = "imports")]
        "package" => WeslTok::KwPackage,
        // #[cfg(feature = "imports")]
        "as" => WeslTok::KwAs,
        // #[cfg(feature = "imports")]
        "import" => WeslTok::KwImport,
        _ => WeslTok::Ident(repr),
    }
}

fn lit2tok(lit: Literal) -> WeslTok {
    match syn::Lit::new(lit) {
        syn::Lit::Int(lit) => match lit.suffix() {
            "" => WeslTok::AbstractInt(
                lit.base10_parse::<i64>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            "i" => WeslTok::I32(
                lit.base10_parse::<i32>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            "u" => WeslTok::U32(
                lit.base10_parse::<u32>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            "f" => WeslTok::F32(
                lit.base10_parse::<f32>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            "h" => WeslTok::F16(
                // TODO validate that if fits in f16
                lit.base10_parse::<f32>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            _ => abort!(lit, "invalid literal suffix"),
        },
        syn::Lit::Float(lit) => match lit.suffix() {
            "" => WeslTok::AbstractFloat(
                lit.base10_parse::<f64>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            "f" => WeslTok::F32(
                lit.base10_parse::<f32>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            "h" => WeslTok::F16(
                // TODO validate that if fits in f16
                lit.base10_parse::<f32>()
                    .unwrap_or_else(|e| abort!(lit, "invalid literal: {}", e)),
            ),
            _ => abort!(lit, "invalid literal suffix"),
        },
        syn::Lit::Bool(lit) => match lit.value() {
            true => WeslTok::KwTrue,
            false => WeslTok::KwFalse,
        },
        lit => abort!(lit, "invalid WESL token"),
    }
}

fn punct2tok(punct: Punct, repr: &str) -> WeslTok {
    match repr {
        "&" => WeslTok::SymAnd,
        "&&" => WeslTok::SymAndAnd,
        "->" => WeslTok::SymArrow,
        "@" => WeslTok::SymAttr,
        "/" => WeslTok::SymForwardSlash,
        "!" => WeslTok::SymBang,
        "[" => WeslTok::SymBracketLeft,
        "]" => WeslTok::SymBracketRight,
        "{" => WeslTok::SymBraceLeft,
        "}" => WeslTok::SymBraceRight,
        ":" => WeslTok::SymColon,
        "," => WeslTok::SymComma,
        "=" => WeslTok::SymEqual,
        "==" => WeslTok::SymEqualEqual,
        "!=" => WeslTok::SymNotEqual,
        ">" => WeslTok::SymGreaterThan,
        ">=" => WeslTok::SymGreaterThanEqual,
        ">>" => WeslTok::SymShiftRight,
        "<" => WeslTok::SymLessThan,
        "<=" => WeslTok::SymLessThanEqual,
        "<<" => WeslTok::SymShiftLeft,
        "%" => WeslTok::SymModulo,
        "-" => WeslTok::SymMinus,
        "--" => WeslTok::SymMinusMinus,
        "." => WeslTok::SymPeriod,
        "+" => WeslTok::SymPlus,
        "++" => WeslTok::SymPlusPlus,
        "|" => WeslTok::SymOr,
        "||" => WeslTok::SymOrOr,
        "(" => WeslTok::SymParenLeft,
        ")" => WeslTok::SymParenRight,
        ";" => WeslTok::SymSemicolon,
        "*" => WeslTok::SymStar,
        "~" => WeslTok::SymTilde,
        "_" => WeslTok::SymUnderscore,
        "^" => WeslTok::SymXor,
        "+=" => WeslTok::SymPlusEqual,
        "-=" => WeslTok::SymMinusEqual,
        "*=" => WeslTok::SymTimesEqual,
        "/=" => WeslTok::SymDivisionEqual,
        "%=" => WeslTok::SymModuloEqual,
        "&=" => WeslTok::SymAndEqual,
        "|=" => WeslTok::SymOrEqual,
        "^=" => WeslTok::SymXorEqual,
        ">>=" => WeslTok::SymShiftRightAssign,
        "<<=" => WeslTok::SymShiftLeftAssign,
        // #[cfg(feature = "imports")]
        "::" => WeslTok::SymColonColon,
        _ => abort!(punct, "invalid WESL punctuation `{}`", repr),
    }
}

impl Lexer {
    fn new(input: TokenStream) -> Self {
        Self {
            it: input.into_iter().into(),
        }
    }

    fn next_tok(&mut self) -> Option<(Span, WeslTok)> {
        match self.it.next()? {
            Token::Delimiter(delim) => Some((delim.span(), delim2tok(&delim))),
            Token::Ident(id) => Some((id.span(), ident2tok(id))),
            Token::Literal(lit) => Some((lit.span(), lit2tok(lit))),
            Token::Punct(mut punct) => {
                let mut repr = punct.to_string();
                let span = punct.span();

                if repr == "#" {
                    match self.it.next()? {
                        Token::Ident(id) => Some((id.span(), WeslTok::Ident(format!("#{id}")))),
                        _ => None,
                    }
                } else {
                    while Spacing::Joint == punct.spacing() {
                        punct = match self.it.next().unwrap() {
                            Token::Punct(punct) => punct,
                            tok => abort!(tok.span(), "unreachable"),
                        };
                        repr.push(punct.as_char());
                    }
                    Some((span, punct2tok(punct, &repr)))
                }
            }
        }
    }
}

type Spanned<Tok, Loc, ParseError> = Result<(Loc, Tok, Loc), (Loc, ParseError, Loc)>;

impl Iterator for Lexer {
    type Item = Spanned<WeslTok, usize, wgsl_parse::error::ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (span, tok) = self.next_tok()?;
        let range = span.byte_range();
        Some(Ok((range.start, tok, range.end)))
    }
}

pub(crate) fn quote_wesl_impl(input: TokenStream) -> TokenStream {
    let lexer = Lexer::new(input);
    let parser = TranslationUnitParser::new();
    let syntax = parser
        .parse(lexer)
        .unwrap_or_else(|e| abort!(Span::call_site(), "{}", wgsl_parse::Error::from(e)));

    syntax.reify()
}
