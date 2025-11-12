import super::util;
import super::util::rotate_3u;

@compute
fn main(@builtin(global_invocation_id) index: vec3u) {
    let rotated_fully_qualified = super::util::rotate_3u(vec3f(1.0, 0.0, 0.0));
    let rotated_partially_qualified = util::rotate_3u(vec3f(1.0, 0.0, 0.0));
    let rotated_unqualified = rotate_3u(vec3f(1.0, 0.0, 0.0));

}
