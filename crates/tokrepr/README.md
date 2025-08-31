# TokRepr

Turn an instance into a Rust [`TokenStream`] that represents the instance.

This crate provides the [`TokRepr`] trait which turns an instance into rust code that
when called, produce an expression that evaluates to the same instance. This can be
useful for code generation in procedural macros.

## Use-Case

This trait is primarily meant to assist with the creation of procedural macros. It can
be useful if the macro needs to output an instance of a complex data structure.
Without if, writing the token stream manually, or with the `quote!` trait can be a
burden.

## How it works

The trait is implemented for Rust primitives and most data types from the standard
library. It uses `proc_macro2` and `quote`.

For user-defined types, the trait can be implemented manually or
automatically with the [derive macro](https://docs.rs/tokrepr-derive). See the macro
documentation for configuration options.

## Limitations

This trait generates code. Like all procedural macros, the generated code is
unhygienic. Implementers of `TokRepr` need to pay attention to the scope of
identifiers. The derive macro assumes that the struct or enum is in scope at the macro
call site.

## See also

The [`uneval`](https://docs.rs/uneval) crate provides a similar feature
but leverages `serde` for serialization, which has the advantage of being widely
implemented for data structure types. It does not provide a way to customize the
codegen of a type.

