use lalrpop::Configuration;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    Configuration::new()
        .use_cargo_dir_conventions()
        .emit_rerun_directives(true)
        .process()
        .unwrap();

    lelwel::build("src/lelwel/wesl.llw");
}
