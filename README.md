# WESL-Rust

[![crates.io](https://img.shields.io/crates/v/wesl)][crates-io]
[![docs.rs](https://img.shields.io/docsrs/wesl)][docs-rs]

[crates-io]: https://crates.io/crates/wesl
[docs-rs]: https://docs.rs/wesl/
[spec]: https://github.com/wgsl-tooling-wg/wesl-spec
[discord]: https://discord.gg/Ng5FWmHuSv

`wesl-rs` implements the necessary tools to build complex WGSL shaders, like what [naga_oil](https://github.com/bevyengine/naga_oil) does for [Bevy](https://bevyengine.org/), but in a framework-agnostic way. Visit [wesl-lang.dev](https://wesl-lang.dev/) to learn more about WGSL shader tools and language extensions.

## Status

(*update: 2025-04*)

* WESL recently released its first [M1 release](https://github.com/wgsl-tooling-wg/wesl-spec/issues/54). It includes imports, Conditional Compilation and Packaging.
* Experimental support for WESL in Bevy was merged.

Currently implemented:

* [x] Imports & Modules
* [x] Conditional Compilation

Experimental:

* [x] Cargo Packages
* [x] Validation
* [x] Compile-time Evaluation and Execution
* [x] Polyfills

Probable future work:

* [ ] Namespaces
* [ ] Generics

## Usage

Read the [WESL for Rust tutorial](https://wesl-lang.dev/docs/Getting-Started-Rust).

This project can be used as a Rust library or as a standalone CLI, refer to the following crates documentation.

[![crates.io](https://img.shields.io/crates/v/wesl)](https://crates.io/crates/wesl)
[![docs.rs](https://img.shields.io/docsrs/wesl)](https://docs.rs/wesl)
**The crate `wesl`** is a WGSL compiler that implements the [WESL specification][spec].

[![crates.io](https://img.shields.io/crates/v/wesl)](https://crates.io/crates/wesl)
[![docs.rs](https://img.shields.io/docsrs/wesl)](https://docs.rs/wesl)
**The crate `wesl-cli`** is the command-line tool to run the compiler.

[![crates.io](https://img.shields.io/crates/v/wgsl-parse)](https://crates.io/crates/wgsl-parse)
[![docs.rs](https://img.shields.io/docsrs/wgsl-parse)](https://docs.rs/wgsl-parse)
**The crate `wgsl-parse`** is a WGSL-compliant syntax tree and parser, with optional syntax extensions from the [WESL specification][spec].

## Contributing

Contributions are welcome. Please join the [discord][discord] to get in touch with the community. Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting Pull Requests.

## License

Except where noted (below and/or in individual files), all code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option.

### Your contributions

Unless you explicitly state otherwise,
any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license,
shall be dual licensed as above,
without any additional terms or conditions.
