import bevy::pbr::forward_io::VertexOutput;
import bevy::pbr::mesh_bindings::mesh;

struct Color {
    base_color: vec4<f32>,
}

@if(BINDLESS) @group(2) @binding(0) var<storage> material_color: binding_array<Color, 4>;
@else         @group(2) @binding(0) var<uniform> material_color: Color;
@if(BINDLESS) @group(2) @binding(1) var material_color_texture: binding_array<texture_2d<f32>, 4>;
@else         @group(2) @binding(1) var material_color_texture: texture_2d<f32>;
@if(BINDLESS) @group(2) @binding(2) var material_color_sampler: binding_array<sampler, 4>;
@else         @group(2) @binding(2) var material_color_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    @if(BINDLESS)
        let slot = mesh[in.instance_index].material_and_lightmap_bind_group_slot & 0xffffu;
    @if(BINDLESS)
        let base_color = material_color[slot].base_color;
    @else
        let base_color = material_color.base_color;

    @if(BINDLESS)
        return base_color * textureSampleLevel(
            material_color_texture[slot],
            material_color_sampler[slot],
            in.uv,
            0.0
        );
    @else
        return base_color * textureSampleLevel(
            material_color_texture,
            material_color_sampler,
            in.uv,
            0.0
        );
}
