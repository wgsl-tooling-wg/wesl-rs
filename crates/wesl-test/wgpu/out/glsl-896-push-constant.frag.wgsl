struct Immediates {
    example: f32,
}

var<immediate> i: Immediates;

fn main_1() {
    return;
}

@fragment
fn main() {
    main_1();
    return;
}
