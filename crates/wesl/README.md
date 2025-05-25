# WESL: A Community Standard for Enhanced WGSL

This is the crate for all your [WESL][wesl] needs.

See also the [standalone CLI][cli].

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

```rust,ignore
# use wesl::{Wesl, FileResolver};
fn main() {
    Wesl::new("src/shaders")
        .build_artifact("main.wesl", "my_shader");
}
```

Include the compiled WGSL string in your code:

```rust,ignore
let module = device.create_shader_module(ShaderModuleDescriptor {
    label: Some("my_shader"),
    source: ShaderSource::Wgsl(include_wesl!("my_shader")),
});
```

## Write shaders inline with the [`quote_module`] macro

With the `quote_*!` macros one can write WGSL shaders directly in source code. This has a
few advantages:

* Like [`quote`](https://docs.rs/quote), it supports local variable injection. This can be
  used e.g. to customize a shader module at runtime.
* The module is parsed at build-time, and syntax errors will be reported at the right
  location in the macro invocation.
* WGSL and Rust have a similar syntax. Your Rust syntax highlighter will also highlight
  injected WGSL code.

```rust
# use wesl::quote_wgsl;
// this i64 value is computed at runtime and injected into the shader.
let num_iterations = 8i64;

// the following variable has type `TranslationUnit`.
let wgsl = quote_module! {
    @fragment
    fn fs_main(@location(0) in_color: vec4<f32>) -> @location(0) vec4<f32> {
        for (let i = 0; i < #num_iterations; i++) {
            // ...
        }
        return in_color;
    }
}
```

One can inject variables into the following places by prefixing the name with
a `#` symbol:
| Code location | Injected type | Indirectly supported injection type with `Into` |
|---------------|---------------|-------------------------------------------------|
| name of a global declaration | `GlobalDeclaration` | `Declaration` `TypeAlias` `Struct` `Function` `ConstAssert` |
| name of a struct member | `StructMember` |   |
| name of an attribute, after `@` | `Attribute` | `BuiltinValue` `InterpolateAttribute` `WorkgroupSizeAttribute` `TypeConstraint` `CustomAttribute` |
| type or identifier expression | `Expression` | `LiteralExpression` `ParenthesizedExpression` `NamedComponentExpression` `IndexingExpression` `UnaryExpression` `BinaryExpression` `FunctionCallExpression` `TypeOrIdentifierExpression` and transitively: `bool` `i64` (AbstractInt) `f64` (AbstractFloat) `i32` `u32` `f32` `Ident` |
| name of an attribute preceding and empty block statement | `Statement` | `CompoundStatement` `AssignmentStatement` `IncrementStatement` `DecrementStatement` `IfStatement` `SwitchStatement` `LoopStatement` `ForStatement` `WhileStatement` `BreakStatement` `ContinueStatement` `ReturnStatement` `DiscardStatement` `FunctionCallStatement` `ConstAssertStatement` `DeclarationStatement` |

```rust
# use wesl::syntax::*;
let inject_struct = Struct::new(Ident::new("mystruct".to_string()));
let inject_func = Function::new(Ident::new("myfunc".to_string()));
let inject_stmt = Statement::Void;
let wgsl = quote_module! {
    struct #inject_struct {}
    fn #inject_func() {}
    fn foo() {
        @#inject_stmt {}
    }
};
```

## Evaluating const-expressions

This is an advanced and experimental feature. `wesl-rs` supports evaluation and execution
of WESL code with the `eval` feature flag. Early evaluation (in particular of
const-expressions) helps developers to catch bugs early by improving the validation and
error reporting capabilities of WESL. Full evaluation of const-expressions can be enabled
with the `lower` compiler option.

Additionally, the `eval` feature adds support for user-defined `@const` attributes on
functions, which allows one to precompute data ahead of time, and ensure that code has no
runtime dependencies.

The eval/exec implementation is tested with the [WebGPU Conformance Test Suite][cts].

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

| name       | description                                           | Status/Specification      |
|------------|-------------------------------------------------------|---------------------------|
| `generics` | user-defined type-generators and generic functions    | [experimental][generics]  |
| `package`  | create shader libraries published to `crates.io`      | [experimental][packaging] |
| `eval`     | execute shader code on the CPU and `@const` attribute | experimental              |
| `naga_ext` | enable all Naga/WGPU extensions                       | experimental              |
| `serde`    | derive `Serialize` and `Deserialize` for syntax nodes |                           |

[wesl]: https://wesl-lang.dev
[cli]: https://crates.io/crates/wesl-cli
[generics]: https://github.com/k2d222/wesl-spec/blob/generics/Generics.md
[packaging]: https://github.com/wgsl-tooling-wg/wesl-spec/blob/main/Packaging.md
[cts]: https://github.com/k2d222/wesl-cts
