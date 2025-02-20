# WESL: A Community Standard for Enhanced WGSL

This is the crate for all your [WESL](https://github.com/wgsl-tooling-wg/wesl-spec)
needs.

See also the [standalone CLI](https://github.com/wgsl-tooling-wg/wesl-rs).

## Basic Usage

See [`Wesl`] for an overview of the high-level API.
```rust
# use wesl::{Wesl, VirtualResolver};
let compiler = Wesl::new("src/shaders");
# // just adding a virtual file here so the doctest runs without a filesystem
# let mut resolver = VirtualResolver::new();
# resolver.add_module("main", "fn my_fn() {}".into());
# let compiler = compiler.set_custom_resolver(resolver);

// compile a WESL file to a WGSL string
let wgsl_str = compiler
    .compile("main.wesl")
    .inspect_err(|e| eprintln!("WESL error: {e}")) // pretty errors with `display()`
    .unwrap()
    .to_string();
```

## Usage in [`build.rs`](https://doc.rust-lang.org/cargo/reference/build-scripts.html)

In your rust project you probably want to have your WESL code converted automatically
to a WGSL string at build-time, unless your WGSL code must be assembled at runtime.

Add this crate to your build dependencies in `Cargo.toml`:
```toml
[build-dependencies]
wesl = "0.1"
```

Create the `build.rs` file with the following content:
```ignore
# use wesl::{Wesl, FileResolver};
fn main() {
    Wesl::new("src/shaders")
        .build_artefact("main.wesl", "my_shader");
}
```

Include the compiled WGSL string in your code:
```ignore
let module = device.create_shader_module(ShaderModuleDescriptor {
    label: Some("my_shader"),
    source: ShaderSource::Wgsl(include_wesl!("my_shader")),
});
```

## Advanced Examples

Evaluate const-expressions.
```rust
# use wesl::{Wesl, VirtualResolver, eval_str};
// ...standalone expression
let wgsl_expr = eval_str("abs(3 - 5)").unwrap().to_string();
assert_eq!(wgsl_expr, "2");

// ...expression using declarations in a WESL file
let source = "const my_const = 4; @const fn my_fn(v: u32) -> u32 { return v * 10; }";
# let mut resolver = VirtualResolver::new();
# resolver.add_module("source", source.into());
# let compiler = Wesl::new_barebones().set_custom_resolver(resolver);
let wgsl_expr = compiler
    .compile("source").unwrap()
    .eval("my_fn(my_const) + 2").unwrap()
    .to_string();
assert_eq!(wgsl_expr, "42u");
```

## Features

| name     | description                                           | WESL Specification        |
|----------|-------------------------------------------------------|---------------------------|
| imports  | import statements and qualified identifiers with `::` | [in progress][imports]    |
| condcomp | conditional compilation with `@if` attributes         | [complete][cond-trans]    |
| generics | user-defined type-generators and generic functions    | [experimental][generics]  |
| package  | create shader libraries published to `crates.io`      | [experimental][packaging] |
| eval     | execute shader code on the CPU and `@const` attribute | not part of the spec      |

`imports` and `condcomp` are default features.

[cond-trans]: https://github.com/wgsl-tooling-wg/wesl-spec/blob/main/ConditionalTranslation.md
[imports]: https://github.com/wgsl-tooling-wg/wesl-spec/blob/main/Imports.md
[generics]: https://github.com/wgsl-tooling-wg/wesl-spec/blob/main/Generics.md
[packaging]: https://github.com/wgsl-tooling-wg/wesl-spec/blob/main/Packaging.md
