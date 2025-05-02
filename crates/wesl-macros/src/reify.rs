use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

pub(crate) fn reify_impl(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    let body = match &input.data {
        Data::Struct(data) => {
            let fields = match &data.fields {
                Fields::Named(fields) => {
                    let fields = fields.named.iter().map(|f| &f.ident).collect_vec();
                    quote! {
                        #(let #fields = self.#fields.reify();)*

                        quote::quote! {
                            #name {
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
                        #name::#variant => quote::quote! { #name::#variant }
                    }
                } else {
                    let fields = (0..v.fields.len())
                        .into_iter()
                        .map(|n| format_ident!("f{n}"))
                        .collect_vec();

                    quote! {
                        #name::#variant(#(#fields),*) => {
                            #(let #fields = #fields.reify();)*
                            quote::quote!{
                                #name::#variant(#(# #fields,)*)
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
        impl crate::reify::Reify for #name {
            fn reify(&self) -> proc_macro2::TokenStream {
                #body
            }
        }
    }
}
