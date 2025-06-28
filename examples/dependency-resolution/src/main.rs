use std::str::FromStr;

fn main() {
    let shader = wesl::Wesl::new("src")
        .set_options(wesl::CompileOptions {
            lazy: true,
            ..Default::default()
        })
        .add_package(&a::a::PACKAGE)
        .add_package(&b::b::PACKAGE)
        .compile(&"package::main".parse().unwrap())
        .map_err(|e| eprintln!("{e}"))
        .expect("compilation error")
        .to_string();

    println!("{shader}");
}
