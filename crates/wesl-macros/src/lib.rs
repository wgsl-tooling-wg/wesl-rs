use proc_macro_error::proc_macro_error;
use query_macro::{QueryInput, query_impl};
use quote_macro::{QuoteNodeKind, quote_impl};
use syn::parse_macro_input;

mod query_macro;
mod quote_macro;

#[proc_macro]
pub fn query(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as QueryInput);
    query_impl(input, false).into()
}

#[proc_macro]
pub fn query_mut(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as QueryInput);
    query_impl(input, true).into()
}

#[proc_macro_error]
#[proc_macro]
pub fn quote_module(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_impl(QuoteNodeKind::TranslationUnit, input.into()).into()
}
#[proc_macro_error]
#[proc_macro]
pub fn quote_import(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_impl(QuoteNodeKind::ImportStatement, input.into()).into()
}
#[proc_macro_error]
#[proc_macro]
pub fn quote_declaration(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_impl(QuoteNodeKind::GlobalDeclaration, input.into()).into()
}
#[proc_macro_error]
#[proc_macro]
pub fn quote_literal(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_impl(QuoteNodeKind::Literal, input.into()).into()
}
#[proc_macro_error]
#[proc_macro]
pub fn quote_directive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_impl(QuoteNodeKind::GlobalDirective, input.into()).into()
}
#[proc_macro_error]
#[proc_macro]
pub fn quote_expression(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_impl(QuoteNodeKind::Expression, input.into()).into()
}
#[proc_macro_error]
#[proc_macro]
pub fn quote_statement(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_impl(QuoteNodeKind::Statement, input.into()).into()
}
