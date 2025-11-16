fn main() {
    wesl::Wesl::new("src/shaders")
        .add_package(&random_wgsl::PACKAGE)
        .build_artifact(&"package::main".parse().unwrap(), "main");
}
