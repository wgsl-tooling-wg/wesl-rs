//! Derive macro for tokrepr.
//!
//!

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(TokRepr)]
pub fn derive_tokrepr(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    tokrepr_impl(input).into()
}

pub(crate) fn tokrepr_impl(input: DeriveInput) -> TokenStream {
    let name = &input.ident;
    let tokrepr_path = quote! { tokrepr };
    let self_path = quote! {};

    let body = match &input.data {
        Data::Struct(data) => {
            let fields = match &data.fields {
                Fields::Named(fields) => {
                    let fields = fields.named.iter().map(|f| &f.ident).collect_vec();
                    quote! {
                        #(let #fields = #tokrepr_path::TokRepr::tok_repr(&self.#fields);)*

                        #tokrepr_path::quote::quote! {
                            #self_path #name {
                                #(#fields: # #fields,)*
                            }
                        }
                    }
                }
                _ => unimplemented!(),
            };
            fields
        }
        Data::Enum(data) => {
            let fields = data.variants.iter().map(|v| {
                let variant = &v.ident;
                if v.fields.is_empty() {
                    quote! {
                        Self::#variant => #tokrepr_path::quote::quote! { #self_path #name::#variant }
                    }
                } else {
                    let fields = (0..v.fields.len())
                        .into_iter()
                        .map(|n| format_ident!("f{n}"))
                        .collect_vec();

                    quote! {
                        Self::#variant(#(#fields),*) => {
                            #(let #fields = #tokrepr_path::TokRepr::tok_repr(#fields);)*
                            #tokrepr_path::quote::quote!{
                                #self_path #name::#variant(#(# #fields,)*)
                            }
                        }
                    }
                }
            });

            quote! {
                match self {
                    #(#fields,)*
                }
            }
        }
        _ => unimplemented!(),
    };

    quote! {
        impl #tokrepr_path::TokRepr for #name {
            fn tok_repr(&self) -> #tokrepr_path::proc_macro2::TokenStream {
                #body
            }
        }
    }
}
