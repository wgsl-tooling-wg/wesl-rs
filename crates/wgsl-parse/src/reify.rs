//! Turn an instance into a `TokenStream` that represents the instance.

use reify::Reify;
use reify::proc_macro2::TokenStream;
use reify::quote::quote;

use crate::{span::Spanned, syntax::Ident};

impl<T: Reify> Reify for Spanned<T> {
    fn reify(&self) -> TokenStream {
        let node = self.node().reify();
        let span = self.span().reify();

        quote! {
            Spanned::new(#node, #span)
        }
    }
}

impl Reify for Ident {
    fn reify(&self) -> TokenStream {
        let value = self.name();
        let value = value.as_str();

        quote! {
            Ident::new(#value.to_string())
        }
    }
}
