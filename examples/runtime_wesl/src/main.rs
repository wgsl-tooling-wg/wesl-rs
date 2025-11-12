fn main() {
    let source = wesl::Wesl::new("src/shaders")
        .compile(&"package::main".parse().unwrap())
        .inspect_err(|e| {
            eprintln!("{e}");
            panic!();
        })
        .unwrap()
        .to_string();

    println!("{source}");
}
