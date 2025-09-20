fn main() {
    // run wesl at build-time
    #[cfg(feature = "build-time")]
    let source = wesl::include_wesl!("main");

    // run wesl at run-time
    #[cfg(not(feature = "build-time"))]
    let source = wesl::Wesl::new("src/shaders")
        .add_package(&random_wgsl::PACKAGE)
        .compile(&"package::main".parse().unwrap())
        .inspect_err(|e| {
            eprintln!("{e}");
            panic!();
        })
        .unwrap()
        .to_string();

    println!("{source}");

    use wesl::syntax::*;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // using the procedural macro
    let source = wesl::quote_module! {
        const timestamp = #timestamp;
    };
    println!("quoted: {source}")
}
