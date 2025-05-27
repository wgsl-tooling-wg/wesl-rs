//! Turn an instance into a Rust [`TokenStream`] that represents the instance.
//!
//! This crate provides the [`TokRepr`] trait which turns an instance into rust code that
//! when called, produce an expression that evaluates to the same instance. This can be
//! useful for code generation in procedural macros.
//!
//! ## Use-Case
//!
//! This trait is primarily meant to assist with the creation of procedural macros. It can
//! be useful if the macro needs to output an instance of a complex data structure.
//! Without if, writing the token stream manually, or with the `quote!` trait can be a
//! burden.
//!
//! ## How it works
//!
//! The trait is implemented for Rust primitives and most data types from the standard
//! library. It uses `proc_macro2` and `quote`.
//!
//! For user-defined types, the trait can be implemented manually or
//! automatically with the [derive macro](https://docs.rs/tokrepr-derive). See the macro
//! documentation for configuration options.
//!
//! ## Limitations
//!
//! This trait generates code. Like all procedural macros, the generated code is
//! unhygienic. Implementers of `TokRepr` need to pay attention to the scope of
//! identifiers. The derive macro assumes that the struct or enum is in scope at the macro
//! call site.
//!
//! ## See also
//!
//! The [`uneval`](https://docs.rs/uneval) crate provides a similar feature
//! but leverages `serde` for serialization, which has the advantage of being widely
//! implemented for data structure types. It does not provide a way to customize the
//! codegen of a type.

use std::{
    borrow::Cow,
    cell::{Cell, LazyCell, OnceCell, RefCell},
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque},
    ffi::{OsStr, OsString},
    ops::{Range, RangeFrom, RangeInclusive, RangeTo},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, RwLock},
};

#[cfg(feature = "rc")]
use std::{
    rc::{Rc, Weak as RcWeak},
    sync::{Arc, Weak as ArcWeak},
};

pub use ::proc_macro2;
pub use ::quote;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

#[cfg(feature = "derive")]
pub use ::tokrepr_derive::TokRepr;

pub trait TokRepr {
    fn tok_repr(&self) -> TokenStream;
}

macro_rules! impl_with_totokens {
    ($name:ident) => {
        impl TokRepr for $name {
            fn tok_repr(&self) -> TokenStream {
                self.to_token_stream()
            }
        }
    };
}

impl_with_totokens!(bool);
impl_with_totokens!(char);
impl_with_totokens!(str);
impl_with_totokens!(f32);
impl_with_totokens!(f64);
impl_with_totokens!(i8);
impl_with_totokens!(i16);
impl_with_totokens!(i32);
impl_with_totokens!(i64);
impl_with_totokens!(i128);
impl_with_totokens!(isize);
impl_with_totokens!(u8);
impl_with_totokens!(u16);
impl_with_totokens!(u32);
impl_with_totokens!(u64);
impl_with_totokens!(u128);
impl_with_totokens!(usize);

impl<T: TokRepr> TokRepr for &T {
    fn tok_repr(&self) -> TokenStream {
        (**self).tok_repr()
    }
}

impl<T: TokRepr> TokRepr for &mut T {
    fn tok_repr(&self) -> TokenStream {
        (**self).tok_repr()
    }
}

impl TokRepr for () {
    fn tok_repr(&self) -> TokenStream {
        quote! { () }
    }
}

impl<T: ToOwned<Owned: TokRepr> + TokRepr> TokRepr for Cow<'_, T> {
    fn tok_repr(&self) -> TokenStream {
        match self {
            Cow::Borrowed(val) => {
                let val = val.tok_repr();
                quote! { ::std::borrow::Cow::Borrowed(#val) }
            }
            Cow::Owned(val) => {
                let val = val.tok_repr();
                quote! { ::std::borrow::Cow::Owned(#val) }
            }
        }
    }
}

impl<T: TokRepr> TokRepr for Box<T> {
    fn tok_repr(&self) -> TokenStream {
        let val = (**self).tok_repr();
        quote! { ::std::boxed::Box::new(#val) }
    }
}

#[cfg(feature = "rc")]
impl<T: TokRepr> TokRepr for Rc<T> {
    fn tok_repr(&self) -> TokenStream {
        let val = (**self).tok_repr();
        quote! { ::std::rc::Rc::new(#val) }
    }
}

#[cfg(feature = "rc")]
impl<T: TokRepr> TokRepr for RcWeak<T> {
    fn tok_repr(&self) -> TokenStream {
        match self.upgrade() {
            Some(rc) => {
                let val = rc.tok_repr();
                quote! { ::std::rc::Rc::downgrade(&#val) }
            }
            None => quote! { ::std::rc::Weak::new() },
        }
    }
}

#[cfg(feature = "rc")]
impl<T: TokRepr> TokRepr for Arc<T> {
    fn tok_repr(&self) -> TokenStream {
        let val = (**self).tok_repr();
        quote! { ::std::sync::Arc::new(#val) }
    }
}

#[cfg(feature = "rc")]
impl<T: TokRepr> TokRepr for ArcWeak<T> {
    fn tok_repr(&self) -> TokenStream {
        match self.upgrade() {
            Some(rc) => {
                let val = rc.tok_repr();
                quote! { ::std::sync::Arc::downgrade(&#val) }
            }
            None => quote! { ::std::sync::Weak::new() },
        }
    }
}

impl<T: TokRepr> TokRepr for Mutex<T> {
    fn tok_repr(&self) -> TokenStream {
        match self.lock() {
            Ok(val) => {
                let val = val.tok_repr();
                quote! { ::std::sync::Mutex::new(#val) }
            }
            Err(err) => panic!("{err}"),
        }
    }
}

impl<T: Copy + TokRepr> TokRepr for OnceLock<T> {
    fn tok_repr(&self) -> TokenStream {
        match self.get() {
            Some(val) => {
                let val = val.tok_repr();
                quote! { ::std::sync::OnceLock::from(#val) }
            }
            None => {
                quote! { ::std::sync::OnceLock::new() }
            }
        }
    }
}

impl<T: Copy + TokRepr> TokRepr for RwLock<T> {
    fn tok_repr(&self) -> TokenStream {
        match self.read() {
            Ok(val) => {
                let val = val.tok_repr();
                quote! { ::std::sync::RwLock::new(#val) }
            }
            Err(err) => panic!("{err}"),
        }
    }
}

impl<T: Copy + TokRepr> TokRepr for Cell<T> {
    fn tok_repr(&self) -> TokenStream {
        let val = self.get().tok_repr();
        quote! { ::std::cell::Cell::new(#val) }
    }
}

impl<T: TokRepr> TokRepr for RefCell<T> {
    fn tok_repr(&self) -> TokenStream {
        let val = self.borrow().tok_repr();
        quote! { ::std::cell::RefCell::new(#val) }
    }
}

impl<T: TokRepr> TokRepr for OnceCell<T> {
    fn tok_repr(&self) -> TokenStream {
        match self.get() {
            Some(val) => {
                let val = val.tok_repr();
                quote! { ::std::cell::OnceCell::from(#val) }
            }
            None => quote! { ::std::cell::OnceCell::new() },
        }
    }
}

impl<T: TokRepr> TokRepr for LazyCell<T> {
    fn tok_repr(&self) -> TokenStream {
        let val = LazyCell::force(self).tok_repr();
        quote! { ::std::cell::LazyCell::new(|| #val) }
    }
}

impl<T: TokRepr> TokRepr for Option<T> {
    fn tok_repr(&self) -> TokenStream {
        if let Some(val) = self {
            let val = val.tok_repr();
            quote! { ::std::option::Option::Some(#val) }
        } else {
            quote! { ::std::option::Option::None }
        }
    }
}

impl<T: TokRepr, U: TokRepr> TokRepr for Result<T, U> {
    fn tok_repr(&self) -> TokenStream {
        match self {
            Ok(val) => {
                let val = val.tok_repr();
                quote! { ::std::result::Result::Ok(#val) }
            }
            Err(val) => {
                let val = val.tok_repr();
                quote! { ::std::result::Result::Err(#val) }
            }
        }
    }
}

impl<T: TokRepr, const N: usize> TokRepr for [T; N] {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { [#(#elements),*] }
    }
}

impl<T: TokRepr> TokRepr for [T] {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { [#(#elements),*] }
    }
}

macro_rules! impl_tuple {
    ($($name:ident),+) => {
        impl<$($name: TokRepr),+> TokRepr for ($($name,)+) {
            fn tok_repr(&self) -> TokenStream {
                #![allow(non_snake_case)]
                let ($($name,)+) = self;
                $( let $name = $name.tok_repr(); )+
                quote! { ($(#$name),+) }
            }
        }
    };
}

#[cfg_attr(docsrs, doc(fake_variadic))]
#[cfg_attr(docsrs, doc = "Implemented for tuples up to 12 items long.")]
impl<A: TokRepr> TokRepr for (A,) {
    fn tok_repr(&self) -> TokenStream {
        #![allow(non_snake_case)]
        let (A,) = self;
        let A = A.tok_repr();
        quote! { (#A) }
    }
}
impl_tuple! { A, B }
impl_tuple! { A, B, C }
impl_tuple! { A, B, C, D }
impl_tuple! { A, B, C, D, E }
impl_tuple! { A, B, C, D, E, F }
impl_tuple! { A, B, C, D, E, F, G }
impl_tuple! { A, B, C, D, E, F, G, H }
impl_tuple! { A, B, C, D, E, F, G, H, I }
impl_tuple! { A, B, C, D, E, F, G, H, I, J }
impl_tuple! { A, B, C, D, E, F, G, H, I, J, K }
impl_tuple! { A, B, C, D, E, F, G, H, I, J, K, L }

impl<T: TokRepr> TokRepr for Range<T> {
    fn tok_repr(&self) -> TokenStream {
        let start = self.start.tok_repr();
        let end = self.end.tok_repr();
        quote! { (#start..#end) }
    }
}

impl<T: TokRepr> TokRepr for RangeFrom<T> {
    fn tok_repr(&self) -> TokenStream {
        let start = self.start.tok_repr();
        quote! { (#start..) }
    }
}

impl<T: TokRepr> TokRepr for RangeTo<T> {
    fn tok_repr(&self) -> TokenStream {
        let end = self.end.tok_repr();
        quote! { (..#end) }
    }
}

impl<T: TokRepr> TokRepr for RangeInclusive<T> {
    fn tok_repr(&self) -> TokenStream {
        let start = self.start().tok_repr();
        let end = self.end().tok_repr();
        quote! { (#start..=#end) }
    }
}

impl TokRepr for OsStr {
    fn tok_repr(&self) -> TokenStream {
        match self.to_str() {
            Some(val) => quote! { ::std::ffi::OsStr::new(#val) },
            None => panic!("OsStr is not valid Unicode"),
        }
    }
}

impl TokRepr for OsString {
    fn tok_repr(&self) -> TokenStream {
        match self.to_str() {
            Some(val) => quote! { ::std::ffi::OsString::from(#val) },
            None => panic!("OsString is not valid Unicode"),
        }
    }
}

impl TokRepr for Path {
    fn tok_repr(&self) -> TokenStream {
        match self.to_str() {
            Some(val) => quote! { ::std::path::Path::new(#val) },
            None => panic!("Path is not valid Unicode"),
        }
    }
}

impl TokRepr for PathBuf {
    fn tok_repr(&self) -> TokenStream {
        match self.to_str() {
            Some(val) => quote! { ::std::path::PathBuf::new(#val) },
            None => panic!("PathBuf is not valid Unicode"),
        }
    }
}

impl TokRepr for String {
    fn tok_repr(&self) -> TokenStream {
        let value = self.as_str();

        quote! {
            ::std::primitive::str::to_string(#value)
        }
    }
}
// COLLECTIONS

impl<T: TokRepr> TokRepr for Vec<T> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { ::std::vec![#(#elements),*] }
    }
}

impl<T: TokRepr> TokRepr for VecDeque<T> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { ::std::collections::VecDeque::from([#(#elements),*]) }
    }
}

impl<T: TokRepr> TokRepr for LinkedList<T> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { ::std::collections::LinkedList::from([#(#elements),*]) }
    }
}

impl<T: TokRepr, U: TokRepr> TokRepr for HashMap<T, U> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(|(k, v)| {
            let (k, v) = (k.tok_repr(), v.tok_repr());
            quote! { (#k, #v) }
        });
        quote! { ::std::collections::HashMap::from([#(#elements),*]) }
    }
}

impl<T: TokRepr, U: TokRepr> TokRepr for BTreeMap<T, U> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(|(k, v)| {
            let (k, v) = (k.tok_repr(), v.tok_repr());
            quote! { (#k, #v) }
        });
        quote! { ::std::collections::BTreeMap::from([#(#elements),*]) }
    }
}

impl<T: TokRepr> TokRepr for HashSet<T> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { ::std::collections::HashSet::from([#(#elements),*]) }
    }
}

impl<T: TokRepr> TokRepr for BTreeSet<T> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { ::std::collections::BTreeSet::from([#(#elements),*]) }
    }
}

impl<T: TokRepr> TokRepr for BinaryHeap<T> {
    fn tok_repr(&self) -> TokenStream {
        let elements = self.iter().map(TokRepr::tok_repr);
        quote! { ::std::collections::BinaryHeap::from([#(#elements),*]) }
    }
}
