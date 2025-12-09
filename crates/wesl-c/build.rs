use std::env;
use std::path::PathBuf;

fn main() {
    let handles = ["WeslCompiler", "WeslTranslationUnit"];

    let mut builder = bindgen::Builder::default()
        .header("include/wesl.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_item("wesl.*")
        .allowlist_item("Wesl.*")
        .prepend_enum_name(false)
        .ignore_functions();

    for handle in handles {
        builder = builder
            .blocklist_item(handle)
            .raw_line(format!("type {handle} = crate::{handle};"));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
