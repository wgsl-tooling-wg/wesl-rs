//! Turn an instance into a `TokenStream` that represents the instance.

pub use ::proc_macro2;
pub use ::quote;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

#[cfg(feature = "derive")]
pub use ::reify_derive::Reify;

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

impl Reify for String {
    fn reify(&self) -> TokenStream {
        let value = self.as_str();

        quote! {
            #value.to_string()
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
