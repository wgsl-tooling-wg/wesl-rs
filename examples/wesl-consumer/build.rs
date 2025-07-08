fn main() {
    #[cfg(feature = "build-time")]
    wesl::Wesl::new("src/shaders")
        .add_package(&random_wgsl::random::PACKAGE)
        .build_artifact(&"package::main".parse().unwrap(), "main");
}
