import super::util;
import super::util::rotate_3u;

import random::noise::perlin::perlin_noise2;
import random::hash::rand11;

@compute
fn main(@builtin(global_invocation_id) index: vec3u) {
    // fully qualified
    let rotated_vec3f = super::util::rotate_3u(vec3f(1.0, 0.0, 0.0));
    // partially qualified
    let rotated_vec3f = util::rotate_3u(vec3f(1.0, 0.0, 0.0));
    // unqualified
    let rotated_vec3f = rotate_3u(vec3f(1.0, 0.0, 0.0));

    // external module
    let hash = rand11(18.0);
    let perlin_noise = perlin_noise2(vec2f(1.0, 2.0));
    let value_noise = random::noise::value::value_noise(2.0);
}
