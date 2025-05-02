use proc_macro_error::proc_macro_error;
use query_macro::{QueryInput, query_impl};
use syn::parse_macro_input;
use wesl_macro::quote_wesl_impl;

mod query_macro;
mod reify;
mod wesl_macro;

#[proc_macro]
pub fn query(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as QueryInput);
    query_impl(input, false)
}

#[proc_macro]
pub fn query_mut(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as QueryInput);
    query_impl(input, true)
}

#[proc_macro_error]
#[proc_macro]
pub fn quote_wesl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote_wesl_impl(input.into()).into()
}

#[proc_macro_derive(Reify)]
pub fn derive_reify(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    reify::reify_impl(input).into()
}
