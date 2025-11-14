fn main() {
    let source = include_str!(concat!(env!("OUT_DIR"), "/", "main", ".wgsl"));

    println!("{source}");
}
