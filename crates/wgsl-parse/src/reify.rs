//! Turn an instance into a `TokenStream` that represents the instance.

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

use crate::{span::Spanned, syntax::Ident};

pub trait Reify {
    fn reify(&self) -> TokenStream;
}

impl<T: Reify> Reify for Vec<T> {
    fn reify(&self) -> TokenStream {
        let elements = self.iter().map(Reify::reify);
        quote! { vec![#(#elements),*] }
    }
}

impl<T: Reify> Reify for Option<T> {
    fn reify(&self) -> TokenStream {
        if let Some(val) = self {
            let val = val.reify();
            quote! { Some(#val) }
        } else {
            quote! { None }
        }
    }
}

impl<T: Reify> Reify for Spanned<T> {
    fn reify(&self) -> TokenStream {
        let node = self.node().reify();
        let span = self.span().reify();

        quote! {
            Spanned::new(#node, #span)
        }
    }
}

impl Reify for String {
    fn reify(&self) -> TokenStream {
        let value = self.as_str();

        quote! {
            #value.to_string()
        }
    }
}

impl Reify for Ident {
    fn reify(&self) -> TokenStream {
        let value = self.name().as_str();

        quote! {
            Ident::new(#value.to_string())
        }
    }
}

macro_rules! impl_with_totokens {
    ($name:ident) => {
        impl Reify for $name {
            fn reify(&self) -> TokenStream {
                self.to_token_stream()
            }
        }
    };
}

impl_with_totokens!(i32);
impl_with_totokens!(u32);
impl_with_totokens!(f32);
impl_with_totokens!(i64);
impl_with_totokens!(u64);
impl_with_totokens!(f64);
impl_with_totokens!(usize);
impl_with_totokens!(bool);
