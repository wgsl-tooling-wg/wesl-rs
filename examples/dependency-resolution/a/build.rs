fn main() {
    wesl::PkgBuilder::new("a")
        .add_package(&c::c::PACKAGE)
        .add_package(&d::d::PACKAGE)
        .scan_directory("src/main")
        .expect("failed to scan WESL files")
        .validate()
        .map_err(|e| eprintln!("{e}"))
        .expect("validation error")
        .build_artifact()
        .expect("failed to build artifact");
}
