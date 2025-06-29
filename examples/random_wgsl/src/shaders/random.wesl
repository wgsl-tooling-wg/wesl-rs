// this file will be the source of the top module "random"

// MIT License. © Stefan Gustavson, Munrocket
// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
fn fade2(t: vec2f) -> vec2f {
    return t * t * t * (t * (t * 6. - 15.) + 10.);
}
