use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(Reify)]
pub fn derive_reify(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    reify_impl(input).into()
}

pub(crate) fn reify_impl(input: DeriveInput) -> TokenStream {
    let name = &input.ident;
    let reify_path = quote! { reify };
    let self_path = quote! {};

    let body = match &input.data {
        Data::Struct(data) => {
            let fields = match &data.fields {
                Fields::Named(fields) => {
                    let fields = fields.named.iter().map(|f| &f.ident).collect_vec();
                    quote! {
                        #(let #fields = #reify_path::Reify::reify(&self.#fields);)*

                        #reify_path::quote::quote! {
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
                        Self::#variant => #reify_path::quote::quote! { #self_path #name::#variant }
                    }
                } else {
                    let fields = (0..v.fields.len())
                        .into_iter()
                        .map(|n| format_ident!("f{n}"))
                        .collect_vec();

                    quote! {
                        Self::#variant(#(#fields),*) => {
                            #(let #fields = #reify_path::Reify::reify(#fields);)*
                            #reify_path::quote::quote!{
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
        impl #reify_path::Reify for #name {
            fn reify(&self) -> #reify_path::proc_macro2::TokenStream {
                #body
            }
        }
    }
}
