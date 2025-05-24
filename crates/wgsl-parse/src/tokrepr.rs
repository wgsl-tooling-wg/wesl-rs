//! Turn an instance into a `TokenStream` that represents the instance.

use tokrepr::TokRepr;
use tokrepr::proc_macro2::TokenStream;
use tokrepr::quote::{format_ident, quote};

use crate::syntax::*;
use crate::{span::Spanned, syntax::Ident};

trait NamedNode {
    fn ident(&self) -> Option<&Ident>;
}

impl NamedNode for GlobalDeclaration {
    fn ident(&self) -> Option<&Ident> {
        self.ident()
    }
}

impl NamedNode for StructMember {
    fn ident(&self) -> Option<&Ident> {
        Some(&self.ident)
    }
}

impl NamedNode for Attribute {
    fn ident(&self) -> Option<&Ident> {
        if let Attribute::Custom(attr) = self {
            Some(&attr.ident)
        } else {
            None
        }
    }
}

impl NamedNode for Expression {
    fn ident(&self) -> Option<&Ident> {
        if let Expression::TypeOrIdentifier(ty) = self {
            Some(&ty.ident)
        } else {
            None
        }
    }
}

impl NamedNode for Statement {
    fn ident(&self) -> Option<&Ident> {
        None
    }
}

impl<T: NamedNode + TokRepr> TokRepr for Spanned<T> {
    fn tok_repr(&self) -> TokenStream {
        let node = self.node().tok_repr();
        let span = self.span().tok_repr();

        if let Some(ident) = self.ident() {
            let name = ident.name();
            if name.starts_with("#") {
                let ident = format_ident!("{}", name[1..]);

                return quote! {
                    Spanned::new(#ident.to_owned().into(), #span)
                };
            }
        }

        quote! {
            Spanned::new(#node, #span)
        }
    }
}

impl TokRepr for Ident {
    fn tok_repr(&self) -> TokenStream {
        let name = self.name();
        if name.starts_with("#") {
            let name = format_ident!("{}", name[1..]);
            quote! {
                Ident::from(#name.to_owned())
            }
        } else {
            let name = name.as_str();
            quote! {
                Ident::new(#name.to_string())
            }
        }
    }
}
