# wesl-consumer

This is an example crate that makes use of an external WESL package `random_wgsl` distributed via Cargo.

The final shader is assembled at build-time in the `build.rs` file. It is possible to do this at runtime. Move the `random_wgsl` crate from `[build-dependencies]` to `[dependencies]`, and copy-paste the code in `build.rs`. (replace `.build("main.wgsl")` with `.compile("main.wgsl").unwrap().to_string()`)
