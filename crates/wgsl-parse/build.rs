use std::env;

use lalrpop::Configuration;

fn main() {
    let default_features = [
        "CARGO_FEATURE_ATTRIBUTES",
        "CARGO_FEATURE_IMPORTS",
        "CARGO_FEATURE_NAGA_EXT",
    ];
    println!("cargo::warning=Info: Build file running.");

    if default_features.iter().all(|f| env::var(f).is_ok()) {
        println!("cargo::warning=Info: All syntax features are enabled, using prebuilt parser.");
    } else {
        println!("cargo::warning=Info: Rebuilding the parser.");
        println!("cargo::rerun-if-changed=build.rs");
        Configuration::new()
            .use_cargo_dir_conventions()
            .emit_rerun_directives(true)
            .process()
            .unwrap();
    }
}
