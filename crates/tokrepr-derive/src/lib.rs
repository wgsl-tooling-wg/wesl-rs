//! Derive macro for [tokrepr].
//!
//! [tokrepr]: https://docs.rs/tokrepr

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
        Data::Struct(data) => match &data.fields {
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
            Fields::Unnamed(f) => {
                let fields = (0..f.unnamed.len())
                    .map(|n| format_ident!("f{n}"))
                    .collect_vec();

                quote! {
                    #(let #fields = #tokrepr_path::TokRepr::tok_repr(#fields);)*

                    #tokrepr_path::quote::quote!{
                        #self_path #name(#(# #fields,)*)
                    }
                }
            }
            Fields::Unit => {
                quote! {
                    #tokrepr_path::quote::quote! {
                        #self_path #name
                    }
                }
            }
        },
        Data::Enum(data) => {
            let fields = data.variants.iter().map(|v| {
                let variant = &v.ident;
                match &v.fields {
                    Fields::Named(_) => unimplemented!(),
                    Fields::Unnamed(f) => {
                        let fields = (0..f.unnamed.len())
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
                    },
                    Fields::Unit => {
                        quote! {
                            Self::#variant => #tokrepr_path::quote::quote! { #self_path #name::#variant }
                        }
                    },
                }
            });

            quote! {
                match self {
                    #(#fields,)*
                }
            }
        }
        Data::Union(_) => unimplemented!("tokrepr derive is not implemented for unions"),
    };

    quote! {
        impl #tokrepr_path::TokRepr for #name {
            fn tok_repr(&self) -> #tokrepr_path::proc_macro2::TokenStream {
                #body
            }
        }
    }
}
