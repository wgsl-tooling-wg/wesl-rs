fn main() {
    // run wesl at build-time
    #[cfg(feature = "build-time")]
    let source = {
        use wesl::include_wesl;
        include_wesl!("main")
    };

    // run wesl at run-time
    #[cfg(not(feature = "build-time"))]
    let source = wesl::Wesl::new("src/shaders")
        .add_package(&random_wgsl::random::Mod)
        .compile("main")
        .inspect_err(|e| {
            eprintln!("{e}");
            panic!();
        })
        .unwrap()
        .to_string();

    println!("{source}");

    // using the procedural macro
    let source = {
        use wesl::quote_wesl;
        quote_wesl! {
            import bevy::pbr::{ foo };

            fn main() {
                return foo();
            }
        }
    };
    println!("quoted: {source}")
}
