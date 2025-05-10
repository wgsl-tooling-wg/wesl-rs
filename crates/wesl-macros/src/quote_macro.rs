use std::iter::Peekable;

use itertools::Itertools;
use proc_macro_error::abort;
use proc_macro2::{Ident, Literal, Punct, Spacing, Span, TokenStream};
use token_stream_flatten::{Delimiter, DelimiterKind, DelimiterPosition, FlattenRec, Token};
use wgsl_parse::{Reify, lexer::Token as WeslTok};

type NextToken = Option<(WeslTok, Span)>;

struct Lexer {
    token_stream: Peekable<FlattenRec>,
    next_token: NextToken,
    recognizing_template: bool,
    opened_templates: u32,
    extras: LexerState,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct LexerState {
    depth: i32,
    template_depths: Vec<i32>,
    lookahead: Option<WeslTok>,
}

fn maybe_template_end(lex: &mut Lexer, current: WeslTok, lookahead: Option<WeslTok>) -> WeslTok {
    if let Some(depth) = lex.extras.template_depths.last() {
        // if found a ">" on the same nesting level as the opening "<", it is a template end.
        if lex.extras.depth == *depth {
            lex.extras.template_depths.pop();
            // if lookahead is GreaterThan, we may have a second closing template.
            // note that >>= can never be (TemplateEnd, TemplateEnd, Equal).
            if let Some(depth) = lex.extras.template_depths.last() {
                if lex.extras.depth == *depth && lookahead == Some(WeslTok::SymGreaterThan) {
                    lex.extras.template_depths.pop();
                    lex.extras.lookahead = Some(WeslTok::TemplateArgsEnd);
                } else {
                    lex.extras.lookahead = lookahead;
                }
            } else {
                lex.extras.lookahead = lookahead;
            }
            return WeslTok::TemplateArgsEnd;
        }
    }

    current
}

// operators && and || have lower precedence than < and >.
// therefore, this is not a template: a < b || c > d
fn maybe_fail_template(lex: &mut Lexer) -> bool {
    if let Some(depth) = lex.extras.template_depths.last() {
        if lex.extras.depth == *depth {
            return false;
        }
    }
    true
}

fn incr_depth(lex: &mut Lexer) {
    lex.extras.depth += 1;
}

fn decr_depth(lex: &mut Lexer) {
    lex.extras.depth -= 1;
}

fn delim2tok(lex: &mut Lexer, delim: &Delimiter) -> WeslTok {
    match (delim.kind(), delim.position()) {
        (DelimiterKind::Brace, DelimiterPosition::Open) => WeslTok::SymBraceLeft,
        (DelimiterKind::Brace, DelimiterPosition::Close) => WeslTok::SymBraceRight,
        (DelimiterKind::Bracket, DelimiterPosition::Open) => {
            incr_depth(lex);
            WeslTok::SymBracketLeft
        }
        (DelimiterKind::Bracket, DelimiterPosition::Close) => {
            decr_depth(lex);
            WeslTok::SymBracketRight
        }
        (DelimiterKind::Parenthesis, DelimiterPosition::Open) => {
            incr_depth(lex);
            WeslTok::SymParenLeft
        }
        (DelimiterKind::Parenthesis, DelimiterPosition::Close) => {
            decr_depth(lex);
            WeslTok::SymParenRight
        }
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

fn punct2tok(lex: &mut Lexer, punct: Punct, repr: &str) -> WeslTok {
    match repr {
        "&" => WeslTok::SymAnd,
        "&&" => {
            if maybe_fail_template(lex) {
                WeslTok::SymAnd
            } else {
                abort!(punct, "invalid WESL punctuation `{}`", repr)
            }
        }
        "->" => WeslTok::SymArrow,
        "@" => WeslTok::SymAttr,
        "/" => WeslTok::SymForwardSlash,
        "!" => WeslTok::SymBang,
        "{" => WeslTok::SymBraceLeft,
        "}" => WeslTok::SymBraceRight,
        ":" => WeslTok::SymColon,
        "," => WeslTok::SymComma,
        "=" => WeslTok::SymEqual,
        "==" => WeslTok::SymEqualEqual,
        "!=" => WeslTok::SymNotEqual,
        ">" => maybe_template_end(lex, WeslTok::SymGreaterThan, None),
        ">=" => maybe_template_end(lex, WeslTok::SymGreaterThanEqual, Some(WeslTok::SymEqual)),
        ">>" => maybe_template_end(lex, WeslTok::SymShiftRight, Some(WeslTok::SymGreaterThan)),
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
        "||" => {
            if maybe_fail_template(lex) {
                WeslTok::SymOrOr
            } else {
                abort!(punct, "invalid WESL punctuation `{}`", repr)
            }
        }
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
        ">>=" => maybe_template_end(
            lex,
            WeslTok::SymShiftRightAssign,
            Some(WeslTok::SymGreaterThanEqual),
        ),
        "<<=" => WeslTok::SymShiftLeftAssign,
        // #[cfg(feature = "imports")]
        "::" => WeslTok::SymColonColon,
        _ => abort!(punct, "invalid WESL punctuation `{}`", repr),
    }
}

pub fn recognize_template_list(token_stream: Peekable<FlattenRec>) -> bool {
    let mut lexer = Lexer::new(
        token_stream,
        Some((WeslTok::TemplateArgsStart, Span::call_site())),
    );
    lexer.recognizing_template = true;
    lexer.opened_templates = 1;
    lexer.extras.template_depths.push(0);
    wgsl_parse::parser::recognize_template_list(lexer).is_ok()
}

impl Lexer {
    fn new(token_stream: Peekable<FlattenRec>, next_token: NextToken) -> Self {
        let mut lex = Self {
            token_stream,
            next_token,
            recognizing_template: false,
            opened_templates: 0,
            extras: Default::default(),
        };
        if lex.next_token.is_none() {
            lex.next_token = lex.token_stream.next().and_then(|tok| lex.tok2wesl(tok));
        }
        lex
    }

    fn take_two_tokens(&mut self) -> (NextToken, NextToken) {
        let tok1 = self.next_token.take();

        let lookahead = self.extras.lookahead.take();
        let tok2 = match lookahead {
            Some(tok) => {
                let (_, span) = tok1.as_ref().unwrap(); // safety: lookahead implies lexer looked at a `<` token
                Some((tok, span.clone()))
            }
            None => self.token_stream.next().and_then(|tok| self.tok2wesl(tok)),
        };

        (tok1, tok2)
    }

    fn tok2wesl(&mut self, tok: Token) -> NextToken {
        match tok {
            Token::Delimiter(delim) => Some((delim2tok(self, &delim), delim.span())),
            Token::Ident(id) => {
                let span = id.span();
                let tok = ident2tok(id);
                Some((tok, span))
            }
            Token::Literal(lit) => {
                let span = lit.span();
                let tok = lit2tok(lit);
                Some((tok, span))
            }
            Token::Punct(punct) => {
                let mut repr = punct.to_string();
                let span = punct.span();

                if repr == "#" {
                    match self.token_stream.next()? {
                        Token::Ident(id) => Some((WeslTok::Ident(format!("#{id}")), id.span())),
                        tok => abort!(tok.span(), "cannot escape token `{}`", tok),
                    }
                } else {
                    let mut join_punct = punct.spacing() == Spacing::Joint;
                    while join_punct {
                        match self.token_stream.peek().unwrap() {
                            Token::Punct(punct) => {
                                // TODO: this is not ideal, we should check if it forms a valid lit.
                                let chr = punct.as_char();
                                if ".;,#".chars().contains(&chr) {
                                    join_punct = false;
                                } else {
                                    repr.push(chr);
                                    join_punct = punct.spacing() == Spacing::Joint;
                                    self.token_stream.next();
                                }
                            }
                            tok => abort!(tok.span(), "unreachable"),
                        };
                    }
                    Some((punct2tok(self, punct, &repr), span))
                }
            }
        }
    }

    fn next_tok(&mut self) -> NextToken {
        let (cur, mut next) = self.take_two_tokens();

        let (cur_tok, cur_span) = match cur {
            Some((tok, span)) => (tok, span),
            None => return None,
        };

        if let Some((next_tok, _)) = &mut next {
            if (matches!(cur_tok, WeslTok::Ident(_)) || cur_tok.is_keyword())
                && *next_tok == WeslTok::SymLessThan
            {
                let input = self.token_stream.clone();
                if recognize_template_list(input) {
                    *next_tok = WeslTok::TemplateArgsStart;
                    let cur_depth = self.extras.depth;
                    self.extras.template_depths.push(cur_depth);
                    self.opened_templates += 1;
                }
            }
        }

        // if we finished recognition of a template
        if self.recognizing_template && cur_tok == WeslTok::TemplateArgsEnd {
            self.opened_templates -= 1;
            if self.opened_templates == 0 {
                next = None; // push eof after end of template
            }
        }

        self.next_token = next;
        Some((cur_tok, cur_span))
    }
}

type Spanned<Tok, Loc, ParseError> = Result<(Loc, Tok, Loc), (Loc, ParseError, Loc)>;

impl Iterator for Lexer {
    type Item = Spanned<WeslTok, usize, wgsl_parse::error::ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (tok, span) = self.next_tok()?;
        let range = span.byte_range();
        Some(Ok((range.start, tok, range.end)))
    }
}

impl wgsl_parse::lexer::TokenIterator for Lexer {}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuoteNodeKind {
    TranslationUnit,
    ImportStatement,
    GlobalDeclaration,
    Literal,
    GlobalDirective,
    Expression,
    Statement,
}

pub(crate) fn quote_impl(kind: QuoteNodeKind, input: TokenStream) -> TokenStream {
    use wgsl_parse::parser::*;
    let token_stream = FlattenRec::from(input.into_iter()).peekable();
    let lexer = Lexer::new(token_stream, None);

    macro_rules! parser_impl {
        ($parser:ident) => {{
            let parser = $parser::new();

            let syntax = parser
                .parse(lexer)
                .unwrap_or_else(|e| abort!(Span::call_site(), "{}", wgsl_parse::Error::from(e)));

            syntax.reify()
        }};
    }

    match kind {
        QuoteNodeKind::TranslationUnit => parser_impl!(TranslationUnitParser),
        QuoteNodeKind::ImportStatement => parser_impl!(ImportStatementParser),
        QuoteNodeKind::GlobalDeclaration => parser_impl!(GlobalDeclParser),
        QuoteNodeKind::Literal => parser_impl!(LiteralParser),
        QuoteNodeKind::GlobalDirective => parser_impl!(GlobalDirectiveParser),
        QuoteNodeKind::Expression => parser_impl!(ExpressionParser),
        QuoteNodeKind::Statement => parser_impl!(StatementParser),
    }
}
