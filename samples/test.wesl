struct Output {
    @size(4) value : f32
};
@group(0) @binding(0) var<storage, read_write> outputs : array<Output, 2>;
@compute @workgroup_size(1)
fn main() {
    outputs[0].value = vec2(4294967296.0, 8589934592.0)[0u] / 0x100000000;
    outputs[1].value = vec2(4294967296.0, 8589934592.0)[1u] / 0x100000000;
}
