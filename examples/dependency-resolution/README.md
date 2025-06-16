# Dependency resolution example with `wesl-rs`

This example shows how WESL relies on Cargo to resolve dependencies using [SemVer] compatibility.

The current package depends on A and B, which both depend on different versions of C and D:

| A dependency | B dependency | Resolution             |
|--------------|--------------|------------------------|
| C v0.1.0     | C v0.1.1     | Unified v0.1.1         |
| D v0.1.0     | D v0.2.0     | Both v0.1.0 and v0.2.0 |

See Cargo resolution documentation:
<https://doc.rust-lang.org/cargo/reference/resolver.html>.

## Running this example

After `cargo run`, you should see `Patch c v0.1.0 was not used in the crate graph`. This shows that
Cargo unified `A/C` and `B/C` dependencies together, keeping only the latest compatible version.

Running the example with `eval` should not produce and error (`const_assert`s passed).

[SemVer]: https://semver.org/
