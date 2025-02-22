struct VertexOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) world_tangent: vec4<f32>,
    @location(6) @interpolate(flat) instance_index: u32,
}

struct FragmentOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX {
    @location(0) color: vec4<f32>,
}

struct StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    attenuation_color: vec4<f32>,
    uv_transform: mat3x3<f32>,
    reflectance: vec3<f32>,
    perceptual_roughness: f32,
    metallic: f32,
    diffuse_transmission: f32,
    specular_transmission: f32,
    thickness: f32,
    ior: f32,
    attenuation_distance: f32,
    clearcoat: f32,
    clearcoat_perceptual_roughness: f32,
    anisotropy_strength: f32,
    anisotropy_rotation: vec2<f32>,
    flags: u32,
    alpha_cutoff: f32,
    parallax_depth_scale: f32,
    max_parallax_layer_count: f32,
    lightmap_exposure: f32,
    max_relief_mapping_search_steps: u32,
    deferred_lighting_pass_id: u32,
}

struct PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    material: StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX,
    diffuse_occlusion: vec3<f32>,
    specular_occlusion: f32,
    frag_coord: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    lightmap_light: vec3<f32>,
    clearcoat_N: vec3<f32>,
    anisotropy_strength: f32,
    anisotropy_T: vec3<f32>,
    anisotropy_B: vec3<f32>,
    is_orthographic: bool,
    flags: u32,
}

struct MeshX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX {
    world_from_local: mat3x4<f32>,
    previous_world_from_local: mat3x4<f32>,
    local_from_world_transpose_a: mat2x4<f32>,
    local_from_world_transpose_b: f32,
    flags: u32,
    lightmap_uv_rect: vec2<u32>,
    first_vertex_index: u32,
    current_skin_index: u32,
    previous_skin_index: u32,
    material_and_lightmap_bind_group_slot: u32,
    tag: u32,
}

struct ColorGradingX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU5TJMV3QX {
    balance: mat3x3<f32>,
    saturation: vec3<f32>,
    contrast: vec3<f32>,
    gamma: vec3<f32>,
    gain: vec3<f32>,
    lift: vec3<f32>,
    midtone_range: vec2<f32>,
    exposure: f32,
    hue: f32,
    post_saturation: f32,
}

struct ViewX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU5TJMV3QX {
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    exposure: f32,
    viewport: vec4<f32>,
    frustum: array<vec4<f32>, 6>,
    color_grading: ColorGradingX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU5TJMV3QX,
    mip_bias: f32,
}

struct DirectionalCascadeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    clip_from_world: mat4x4<f32>,
    texel_size: f32,
    far_bound: f32,
}

struct DirectionalLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    cascades: array<DirectionalCascadeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, 4>,
    color: vec4<f32>,
    direction_to_light: vec3<f32>,
    flags: u32,
    soft_shadow_size: f32,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
    num_cascades: u32,
    cascades_overlap_proportion: f32,
    depth_texture_base_index: u32,
    skip: u32,
}

struct LightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    directional_lights: array<DirectionalLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, 10>,
    ambient_color: vec4<f32>,
    cluster_dimensions: vec4<u32>,
    cluster_factors: vec4<f32>,
    n_directional_lights: u32,
    spot_light_shadowmap_offset: i32,
    environment_map_smallest_specular_mip_level: u32,
    environment_map_intensity: f32,
}

struct FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    base_color: vec4<f32>,
    directional_light_color: vec4<f32>,
    be: vec3<f32>,
    directional_light_exponent: f32,
    bi: vec3<f32>,
    mode: u32,
}

struct ClusterableObjectX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    light_custom_data: vec4<f32>,
    color_inverse_square_range: vec4<f32>,
    position_radius: vec4<f32>,
    flags: u32,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
    spot_light_tan_angle: f32,
    soft_shadow_size: f32,
    shadow_map_near_z: f32,
    texture_index: u32,
    pad: f32,
}

struct ClusterableObjectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    data: array<ClusterableObjectX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX>,
}

struct ClusterLightIndexListsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    data: array<u32>,
}

struct ClusterOffsetsAndCountsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    data: array<array<vec4<u32>, 2>>,
}

struct LightProbeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    light_from_world_transposed: mat3x4<f32>,
    cubemap_index: i32,
    intensity: f32,
    affects_lightmapped_mesh_diffuse: u32,
}

struct LightProbesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    reflection_probes: array<LightProbeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, 8>,
    irradiance_volumes: array<LightProbeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, 8>,
    reflection_probe_count: i32,
    irradiance_volume_count: i32,
    view_cubemap_index: i32,
    smallest_specular_mip_level_for_view: u32,
    intensity_for_view: f32,
    view_environment_map_affects_lightmapped_mesh_diffuse: u32,
}

struct ScreenSpaceReflectionsSettingsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    perceptual_roughness_threshold: f32,
    thickness: f32,
    linear_steps: u32,
    linear_march_exponent: f32,
    bisection_steps: u32,
    use_secant: u32,
}

struct EnvironmentMapUniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX {
    transform: mat4x4<f32>,
}

struct GlobalsX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DUZ3MN5RGC3DTX {
    time: f32,
    delta_time: f32,
    frame_count: u32,
}

struct LayerLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX {
    N: vec3<f32>,
    R: vec3<f32>,
    NdotV: f32,
    perceptual_roughness: f32,
    roughness: f32,
}

struct LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX {
    layers: array<LayerLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX, 2>,
    P: vec3<f32>,
    V: vec3<f32>,
    diffuse_color: vec3<f32>,
    F0_: vec3<f32>,
    F_ab: vec2<f32>,
    clearcoat_strength: f32,
}

struct DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX {
    H: vec3<f32>,
    NdotL: f32,
    NdotH: f32,
    LdotH: f32,
}

struct ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX {
    first_point_light_index_offset: u32,
    first_spot_light_index_offset: u32,
    first_reflection_probe_index_offset: u32,
    first_irradiance_volume_index_offset: u32,
    first_decal_offset: u32,
    last_clusterable_object_index_offset: u32,
}

struct LightProbeQueryResultX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX {
    texture_index: i32,
    intensity: f32,
    light_from_world: mat4x4<f32>,
    affects_lightmapped_mesh_diffuse: bool,
}

struct EnvironmentMapLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

struct EnvironmentMapRadiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    irradiance: vec3<f32>,
    radiance: vec3<f32>,
}

struct SampleBiasX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX {
    mip_bias: f32,
}

const STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 1u;
const STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 2u;
const STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 4u;
const STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 8u;
const STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 16u;
const STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 32u;
const STANDARD_MATERIAL_FLAGS_DEPTH_MAP_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 512u;
const STANDARD_MATERIAL_FLAGS_CLEARCOAT_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 16384u;
const STANDARD_MATERIAL_FLAGS_CLEARCOAT_ROUGHNESS_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 32768u;
const STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUEX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 0u;
const STANDARD_MATERIAL_FLAGS_TWO_COMPONENT_NORMAL_MAPX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 64u;
const STANDARD_MATERIAL_FLAGS_FLIP_NORMAL_MAP_YX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 128u;
const STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 3758096384u;
const PIX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX: f32 = 3.1415927f;
const POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 2u;
const LAYER_BASEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX: u32 = 0u;
const LAYER_CLEARCOATX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX: u32 = 1u;
const MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX: u32 = 536870912u;
const POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 1u;
const DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 1u;
const SPIRAL_OFFSET_0_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(-0.7071f, 0.7071f);
const SPIRAL_OFFSET_1_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(-0f, -0.875f);
const SPIRAL_OFFSET_2_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(0.5303f, 0.5303f);
const SPIRAL_OFFSET_3_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(-0.625f, -0f);
const SPIRAL_OFFSET_4_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(0.3536f, -0.3536f);
const SPIRAL_OFFSET_5_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(-0f, 0.375f);
const SPIRAL_OFFSET_6_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(-0.1768f, -0.1768f);
const SPIRAL_OFFSET_7_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX: vec2<f32> = vec2<f32>(0.125f, -0f);
const SPOT_SHADOW_TEXEL_SIZEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX: f32 = 0.013427734f;
const POINT_SHADOW_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX: f32 = 0.003f;
const POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX: f32 = 0.5f;
const PI_2X_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX: f32 = 6.2831855f;
const FRAC_PI_3X_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX: f32 = 1.0471976f;
const flip_zX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX: vec3<f32> = vec3<f32>(1f, 1f, -1f);

@group(1) @binding(0) 
var<storage> meshX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7MJUW4ZDJNZTXGX: array<MeshX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX>;
@group(0) @binding(0) 
var<uniform> viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: ViewX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU5TJMV3QX;
@group(0) @binding(1) 
var<uniform> lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: LightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(8) 
var<storage> clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: ClusterableObjectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(9) 
var<storage> clusterable_object_index_listsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: ClusterLightIndexListsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(10) 
var<storage> cluster_offsets_and_countsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: ClusterOffsetsAndCountsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(2) 
var point_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: texture_depth_cube_array;
@group(0) @binding(3) 
var point_shadow_textures_comparison_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: sampler_comparison;
@group(0) @binding(5) 
var directional_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: texture_depth_2d_array;
@group(0) @binding(6) 
var directional_shadow_textures_comparison_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: sampler_comparison;
@group(0) @binding(11) 
var<uniform> globalsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: GlobalsX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DUZ3MN5RGC3DTX;
@group(0) @binding(13) 
var<uniform> light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: LightProbesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(17) 
var diffuse_environment_mapsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: binding_array<texture_cube<f32>, 8>;
@group(0) @binding(18) 
var specular_environment_mapsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: binding_array<texture_cube<f32>, 8>;
@group(0) @binding(19) 
var environment_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: sampler;
@group(0) @binding(20) 
var<uniform> environment_map_uniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: EnvironmentMapUniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(2) @binding(0) 
var<uniform> materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;
@group(2) @binding(1) 
var base_color_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: texture_2d<f32>;
@group(2) @binding(2) 
var base_color_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: sampler;
@group(2) @binding(3) 
var emissive_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: texture_2d<f32>;
@group(2) @binding(4) 
var emissive_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: sampler;
@group(2) @binding(5) 
var metallic_roughness_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: texture_2d<f32>;
@group(2) @binding(6) 
var metallic_roughness_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: sampler;
@group(2) @binding(7) 
var occlusion_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: texture_2d<f32>;
@group(2) @binding(8) 
var occlusion_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: sampler;
@group(2) @binding(21) 
var clearcoat_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: texture_2d<f32>;
@group(2) @binding(22) 
var clearcoat_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: sampler;
@group(2) @binding(23) 
var clearcoat_roughness_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: texture_2d<f32>;
@group(2) @binding(24) 
var clearcoat_roughness_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: sampler;
@group(2) @binding(11) 
var depth_map_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: texture_2d<f32>;
@group(2) @binding(12) 
var depth_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX: sampler;

fn standard_material_newX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX() -> StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    var material: StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;

    material.base_color = vec4<f32>(1f, 1f, 1f, 1f);
    material.emissive = vec4<f32>(0f, 0f, 0f, 1f);
    material.perceptual_roughness = 0.5f;
    material.metallic = 0f;
    material.reflectance = vec3(0.5f);
    material.diffuse_transmission = 0f;
    material.specular_transmission = 0f;
    material.thickness = 0f;
    material.ior = 1.5f;
    material.attenuation_distance = 1f;
    material.attenuation_color = vec4<f32>(1f, 1f, 1f, 1f);
    material.clearcoat = 0f;
    material.clearcoat_perceptual_roughness = 0f;
    material.flags = STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUEX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;
    material.alpha_cutoff = 0.5f;
    material.parallax_depth_scale = 0.1f;
    material.max_parallax_layer_count = 16f;
    material.max_relief_mapping_search_steps = 5u;
    material.deferred_lighting_pass_id = 1u;
    material.uv_transform = mat3x3<f32>(vec3<f32>(1f, 0f, 0f), vec3<f32>(0f, 1f, 0f), vec3<f32>(0f, 0f, 1f));
    let _e66 = material;
    return _e66;
}

fn pbr_input_newX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX() -> PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    var pbr_input_1: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;

    let _e2 = standard_material_newX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX();
    pbr_input_1.material = _e2;
    pbr_input_1.diffuse_occlusion = vec3(1f);
    pbr_input_1.specular_occlusion = 1f;
    pbr_input_1.frag_coord = vec4<f32>(0f, 0f, 0f, 1f);
    pbr_input_1.world_position = vec4<f32>(0f, 0f, 0f, 1f);
    pbr_input_1.world_normal = vec3<f32>(0f, 0f, 1f);
    pbr_input_1.is_orthographic = false;
    pbr_input_1.N = vec3<f32>(0f, 0f, 1f);
    pbr_input_1.V = vec3<f32>(1f, 0f, 0f);
    pbr_input_1.clearcoat_N = vec3(0f);
    pbr_input_1.anisotropy_T = vec3(0f);
    pbr_input_1.anisotropy_B = vec3(0f);
    pbr_input_1.lightmap_light = vec3(0f);
    pbr_input_1.flags = 0u;
    let _e51 = pbr_input_1;
    return _e51;
}

fn F_ABX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptual_roughness: f32, NdotV: f32) -> vec2<f32> {
    const c0_ = vec4<f32>(-1f, -0.0275f, -0.572f, 0.022f);
    const c1_ = vec4<f32>(1f, 0.0425f, 1.04f, -0.04f);
    let r = ((perceptual_roughness * c0_) + c1_);
    let a004_ = ((min((r.x * r.x), exp2((-9.28f * NdotV))) * r.x) + r.y);
    return ((vec2<f32>(-1.04f, 1.04f) * a004_) + r.zw);
}

fn perceptualRoughnessToRoughnessX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptualRoughness: f32) -> f32 {
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089f, 1f);
    return (clampedPerceptualRoughness * clampedPerceptualRoughness);
}

fn getDistanceAttenuationX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(distanceSquare: f32, inverseRangeSquared: f32) -> f32 {
    let factor = (distanceSquare * inverseRangeSquared);
    let smoothFactor = saturate((1f - (factor * factor)));
    let attenuation = (smoothFactor * smoothFactor);
    return ((attenuation * 1f) / max(distanceSquare, 0.0001f));
}

fn compute_specular_layer_values_for_point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, layer: u32, V: vec3<f32>, light_to_frag: vec3<f32>, light_position_radius: f32) -> vec4<f32> {
    let R_1 = (*input).layers[layer].R;
    let a = (*input).layers[layer].roughness;
    let centerToRay = ((dot(light_to_frag, R_1) * R_1) - light_to_frag);
    let closestPoint = (light_to_frag + (centerToRay * saturate((light_position_radius * inverseSqrt(dot(centerToRay, centerToRay))))));
    let LspecLengthInverse = inverseSqrt(dot(closestPoint, closestPoint));
    let normalizationFactor = (a / saturate((a + ((light_position_radius * 0.5f) * LspecLengthInverse))));
    let intensity = (normalizationFactor * normalizationFactor);
    let L_1 = (closestPoint * LspecLengthInverse);
    return vec4<f32>(L_1, intensity);
}

fn derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(N: vec3<f32>, V_1: vec3<f32>, L: vec3<f32>) -> DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX {
    var input_1: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var H: vec3<f32>;

    H = normalize((L + V_1));
    let _e7 = H;
    input_1.H = _e7;
    input_1.NdotL = saturate(dot(N, L));
    let _e13 = H;
    input_1.NdotH = saturate(dot(N, _e13));
    let _e17 = H;
    input_1.LdotH = saturate(dot(L, _e17));
    let _e20 = input_1;
    return _e20;
}

fn D_GGXX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(roughness: f32, NdotH: f32, h: vec3<f32>) -> f32 {
    let oneMinusNdotHSquared = (1f - (NdotH * NdotH));
    let a_1 = (NdotH * roughness);
    let k = (roughness / (oneMinusNdotHSquared + (a_1 * a_1)));
    let d = ((k * k) * 0.31830987f);
    return d;
}

fn V_SmithGGXCorrelatedX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(roughness_1: f32, NdotV_1: f32, NdotL: f32) -> f32 {
    let a2_ = (roughness_1 * roughness_1);
    let lambdaV = (NdotL * sqrt((((NdotV_1 - (a2_ * NdotV_1)) * NdotV_1) + a2_)));
    let lambdaL = (NdotV_1 * sqrt((((NdotL - (a2_ * NdotL)) * NdotL) + a2_)));
    let v = (0.5f / (lambdaV + lambdaL));
    return v;
}

fn F_Schlick_vecX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(f0_: vec3<f32>, f90_: f32, VdotH: f32) -> vec3<f32> {
    return (f0_ + ((vec3(f90_) - f0_) * pow((1f - VdotH), 5f)));
}

fn fresnelX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(f0_1: vec3<f32>, LdotH: f32) -> vec3<f32> {
    let f90_2 = saturate(dot(f0_1, vec3(16.5f)));
    let _e6 = F_Schlick_vecX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(f0_1, f90_2, LdotH);
    return _e6;
}

fn specular_multiscatterX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_2: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, D: f32, V_2: f32, F: vec3<f32>, specular_intensity: f32) -> vec3<f32> {
    var Fr: vec3<f32>;

    let F0_1 = (*input_2).F0_;
    let F_ab_1 = (*input_2).F_ab;
    Fr = (((specular_intensity * D) * V_2) * F);
    let _e22 = Fr;
    Fr = (_e22 * (vec3(1f) + (F0_1 * ((1f / F_ab_1.x) - 1f))));
    let _e24 = Fr;
    return _e24;
}

fn specularX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_3: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, derived_input: ptr<function, DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, specular_intensity_1: f32) -> vec3<f32> {
    let roughness_3 = (*input_3).layers[0].roughness;
    let NdotV_3 = (*input_3).layers[0].NdotV;
    let F0_2 = (*input_3).F0_;
    let H_1 = (*derived_input).H;
    let NdotL_1 = (*derived_input).NdotL;
    let NdotH_1 = (*derived_input).NdotH;
    let LdotH_2 = (*derived_input).LdotH;
    let _e20 = D_GGXX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(roughness_3, NdotH_1, H_1);
    let _e21 = V_SmithGGXCorrelatedX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(roughness_3, NdotV_3, NdotL_1);
    let _e22 = fresnelX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(F0_2, LdotH_2);
    let _e24 = specular_multiscatterX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_3, _e20, _e21, _e22, specular_intensity_1);
    return _e24;
}

fn V_KelemenX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(LdotH_1: f32) -> f32 {
    return (0.25f / (LdotH_1 * LdotH_1));
}

fn F_SchlickX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(f0_2: f32, f90_1: f32, VdotH_1: f32) -> f32 {
    return (f0_2 + ((f90_1 - f0_2) * pow((1f - VdotH_1), 5f)));
}

fn specular_clearcoatX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_4: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, derived_input_1: ptr<function, DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, clearcoat_strength: f32, specular_intensity_2: f32) -> vec2<f32> {
    let roughness_4 = (*input_4).layers[1].roughness;
    let H_2 = (*derived_input_1).H;
    let NdotH_2 = (*derived_input_1).NdotH;
    let LdotH_3 = (*derived_input_1).LdotH;
    let _e12 = D_GGXX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(roughness_4, NdotH_2, H_2);
    let _e13 = V_KelemenX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(LdotH_3);
    let _e16 = F_SchlickX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(0.04f, 1f, LdotH_3);
    let Fc = (_e16 * clearcoat_strength);
    let Frc = (((specular_intensity_2 * _e12) * _e13) * Fc);
    return vec2<f32>(Fc, Frc);
}

fn Fd_BurleyX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_5: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, derived_input_2: ptr<function, DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>) -> f32 {
    let roughness_5 = (*input_5).layers[0].roughness;
    let NdotV_4 = (*input_5).layers[0].NdotV;
    let NdotL_2 = (*derived_input_2).NdotL;
    let LdotH_4 = (*derived_input_2).LdotH;
    let f90_3 = (0.5f + (((2f * roughness_5) * LdotH_4) * LdotH_4));
    let _e21 = F_SchlickX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(1f, f90_3, NdotL_2);
    let _e23 = F_SchlickX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(1f, f90_3, NdotV_4);
    return ((_e21 * _e23) * 0.31830987f);
}

fn point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id: u32, input_6: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, enable_diffuse: bool) -> vec3<f32> {
    var specular_derived_input: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var clearcoat_specular_derived_input: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var derived_input_3: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var diffuse: vec3<f32> = vec3(0f);
    var color: vec3<f32>;

    let diffuse_color_1 = (*input_6).diffuse_color;
    let P = (*input_6).P;
    let N_3 = (*input_6).layers[0].N;
    let V_5 = (*input_6).V;
    let light = (&clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[light_id]);
    let _e18 = (*light).position_radius;
    let light_to_frag_1 = (_e18.xyz - P);
    let L_2 = normalize(light_to_frag_1);
    let distance_square = dot(light_to_frag_1, light_to_frag_1);
    let _e25 = (*light).color_inverse_square_range.w;
    let _e26 = getDistanceAttenuationX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(distance_square, _e25);
    let _e29 = (*light).position_radius.w;
    let _e31 = compute_specular_layer_values_for_point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_6, LAYER_BASEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX, V_5, light_to_frag_1, _e29);
    let _e33 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(N_3, V_5, _e31.xyz);
    specular_derived_input = _e33;
    let specular_intensity_3 = _e31.w;
    let _e36 = specularX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_6, (&specular_derived_input), specular_intensity_3);
    let clearcoat_N = (*input_6).layers[1].N;
    let clearcoat_strength_1 = (*input_6).clearcoat_strength;
    let _e45 = (*light).position_radius.w;
    let _e47 = compute_specular_layer_values_for_point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_6, LAYER_CLEARCOATX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX, V_5, light_to_frag_1, _e45);
    let _e49 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(clearcoat_N, V_5, _e47.xyz);
    clearcoat_specular_derived_input = _e49;
    let clearcoat_specular_intensity = _e47.w;
    let _e52 = specular_clearcoatX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_6, (&clearcoat_specular_derived_input), clearcoat_strength_1, clearcoat_specular_intensity);
    let inv_Fc = (1f - _e52.x);
    let Frc_1 = _e52.y;
    let _e57 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(N_3, V_5, L_2);
    derived_input_3 = _e57;
    if enable_diffuse {
        let _e60 = Fd_BurleyX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_6, (&derived_input_3));
        diffuse = (diffuse_color_1 * _e60);
    }
    let _e63 = diffuse;
    color = (((_e63 + (_e36 * inv_Fc)) * inv_Fc) + vec3(Frc_1));
    let _e70 = color;
    let _e72 = (*light).color_inverse_square_range;
    let _e76 = derived_input_3.NdotL;
    return ((_e70 * _e72.xyz) * (_e26 * _e76));
}

fn spot_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id_1: u32, input_7: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, enable_diffuse_1: bool) -> vec3<f32> {
    var spot_dir: vec3<f32>;

    let _e3 = point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id_1, input_7, enable_diffuse_1);
    let light_1 = (&clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[light_id_1]);
    let _e9 = (*light_1).light_custom_data.x;
    let _e12 = (*light_1).light_custom_data.y;
    spot_dir = vec3<f32>(_e9, 0f, _e12);
    let _e18 = spot_dir.x;
    let _e20 = spot_dir.x;
    let _e26 = spot_dir.z;
    let _e28 = spot_dir.z;
    spot_dir.y = sqrt(max(0f, ((1f - (_e18 * _e20)) - (_e26 * _e28))));
    let _e34 = (*light_1).flags;
    if ((_e34 & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u) {
        let _e41 = spot_dir.y;
        spot_dir.y = -(_e41);
    }
    let _e44 = (*light_1).position_radius;
    let _e47 = (*input_7).P;
    let light_to_frag_2 = (_e44.xyz - _e47.xyz);
    let _e50 = spot_dir;
    let cd = dot(-(_e50), normalize(light_to_frag_2));
    let _e56 = (*light_1).light_custom_data.z;
    let _e60 = (*light_1).light_custom_data.w;
    let attenuation_1 = saturate(((cd * _e56) + _e60));
    let spot_attenuation = (attenuation_1 * attenuation_1);
    return (_e3 * spot_attenuation);
}

fn directional_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id_2: u32, input_8: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, enable_diffuse_2: bool) -> vec3<f32> {
    var derived_input_4: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var diffuse_1: vec3<f32> = vec3(0f);
    var derived_clearcoat_input: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var color_1: vec3<f32>;

    let diffuse_color_2 = (*input_8).diffuse_color;
    let NdotV_5 = (*input_8).layers[0].NdotV;
    let N_4 = (*input_8).layers[0].N;
    let V_6 = (*input_8).V;
    let roughness_6 = (*input_8).layers[0].roughness;
    let light_2 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[light_id_2]);
    let _e24 = (*light_2).direction_to_light;
    let L_3 = _e24.xyz;
    let _e26 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(N_4, V_6, L_3);
    derived_input_4 = _e26;
    if enable_diffuse_2 {
        let _e29 = Fd_BurleyX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_8, (&derived_input_4));
        diffuse_1 = (diffuse_color_2 * _e29);
    }
    let _e33 = specularX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_8, (&derived_input_4), 1f);
    let clearcoat_N_1 = (*input_8).layers[1].N;
    let clearcoat_strength_2 = (*input_8).clearcoat_strength;
    let _e40 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(clearcoat_N_1, V_6, L_3);
    derived_clearcoat_input = _e40;
    let _e43 = specular_clearcoatX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_8, (&derived_clearcoat_input), clearcoat_strength_2, 1f);
    let inv_Fc_1 = (1f - _e43.x);
    let Frc_2 = _e43.y;
    let _e48 = diffuse_1;
    let _e53 = derived_input_4.NdotL;
    let _e56 = derived_clearcoat_input.NdotL;
    color_1 = ((((_e48 + (_e33 * inv_Fc_1)) * inv_Fc_1) * _e53) + vec3((Frc_2 * _e56)));
    let _e61 = color_1;
    let _e63 = (*light_2).color;
    return (_e61 * _e63.xyz);
}

fn view_z_to_z_sliceX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(view_z: f32, is_orthographic: bool) -> u32 {
    var z_slice: u32 = 0u;

    if is_orthographic {
        let _e6 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_factors.z;
        let _e11 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_factors.w;
        z_slice = u32(floor(((view_z - _e6) * _e11)));
    } else {
        let _e21 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_factors.z;
        let _e26 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_factors.w;
        z_slice = u32((((log(-(view_z)) * _e21) - _e26) + 1f));
    }
    let _e31 = z_slice;
    let _e35 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_dimensions.z;
    return min(_e31, (_e35 - 1u));
}

fn fragment_cluster_indexX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(frag_coord: vec2<f32>, view_z_1: f32, is_orthographic_1: bool) -> u32 {
    let _e3 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.viewport;
    let _e8 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_factors;
    let xy = vec2<u32>(floor(((frag_coord - _e3.xy) * _e8.xy)));
    let _e15 = view_z_to_z_sliceX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(view_z_1, is_orthographic_1);
    let _e20 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_dimensions.x;
    let _e27 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_dimensions.z;
    let _e33 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_dimensions.w;
    return min(((((xy.y * _e20) + xy.x) * _e27) + _e15), (_e33 - 1u));
}

fn unpack_clusterable_object_index_rangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(cluster_index: u32) -> ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX {
    let offset_and_counts_a = cluster_offsets_and_countsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[cluster_index][0];
    let offset_and_counts_b = cluster_offsets_and_countsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[cluster_index][1];
    let point_light_offset = offset_and_counts_a.x;
    let spot_light_offset = (point_light_offset + offset_and_counts_a.y);
    let reflection_probe_offset = (spot_light_offset + offset_and_counts_a.z);
    let irradiance_volume_offset = (reflection_probe_offset + offset_and_counts_a.w);
    let decal_offset = (irradiance_volume_offset + offset_and_counts_b.x);
    let last_clusterable_offset = (decal_offset + offset_and_counts_b.y);
    return ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(point_light_offset, spot_light_offset, reflection_probe_offset, irradiance_volume_offset, decal_offset, last_clusterable_offset);
}

fn get_clusterable_object_idX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(index: u32) -> u32 {
    let _e4 = clusterable_object_index_listsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[index];
    return _e4;
}

fn cluster_debug_visualizationX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(input_color: vec4<f32>, view_z_2: f32, is_orthographic_2: bool, clusterable_object_index_ranges: ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX, cluster_index_1: u32) -> vec4<f32> {
    var output_color: vec4<f32>;

    output_color = input_color;
    let _e2 = output_color;
    return _e2;
}

fn orthonormalizeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(z_unnormalized: vec3<f32>, up: vec3<f32>) -> mat3x3<f32> {
    let z_basis = normalize(z_unnormalized);
    let x_basis_2 = normalize(cross(z_basis, up));
    let y_basis_2 = cross(z_basis, x_basis_2);
    return mat3x3<f32>(x_basis_2, y_basis_2, z_basis);
}

fn interleaved_gradient_noiseX_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX(pixel_coordinates: vec2<f32>, frame: u32) -> f32 {
    let xy_1 = (pixel_coordinates + vec2((5.588238f * f32((frame % 64u)))));
    return fract((52.982918f * fract(((0.06711056f * xy_1.x) + (0.00583715f * xy_1.y)))));
}

fn sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local: vec2<f32>, depth: f32, array_index: i32) -> f32 {
    let _e5 = textureSampleCompareLevel(directional_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, directional_shadow_textures_comparison_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, light_local, array_index, depth);
    return _e5;
}

fn sample_shadow_map_castano_thirteenX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_1: vec2<f32>, depth_1: f32, array_index_1: i32) -> f32 {
    var base_uv: vec2<f32>;
    var sum: f32 = 0f;

    let _e2 = textureDimensions(directional_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX);
    let shadow_map_size = vec2<f32>(_e2);
    let inv_shadow_map_size = (vec2(1f) / shadow_map_size);
    let uv_3 = (light_local_1 * shadow_map_size);
    base_uv = floor((uv_3 + vec2(0.5f)));
    let _e18 = base_uv.x;
    let s = ((uv_3.x + 0.5f) - _e18);
    let _e24 = base_uv.y;
    let t = ((uv_3.y + 0.5f) - _e24);
    let _e27 = base_uv;
    base_uv = (_e27 - vec2(0.5f));
    let _e30 = base_uv;
    base_uv = (_e30 * inv_shadow_map_size);
    let uw0_ = (4f - (3f * s));
    let uw2_ = (1f + (3f * s));
    let u0_ = (((3f - (2f * s)) / uw0_) - 2f);
    let u1_ = ((3f + s) / 7f);
    let u2_ = ((s / uw2_) + 2f);
    let vw0_ = (4f - (3f * t));
    let vw2_ = (1f + (3f * t));
    let v0_ = (((3f - (2f * t)) / vw0_) - 2f);
    let v1_ = ((3f + t) / 7f);
    let v2_ = ((t / vw2_) + 2f);
    let _e77 = base_uv;
    let _e83 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e77 + (vec2<f32>(u0_, v0_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e86 = sum;
    sum = (_e86 + ((uw0_ * vw0_) * _e83));
    let _e89 = base_uv;
    let _e93 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e89 + (vec2<f32>(u1_, v0_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e95 = sum;
    sum = (_e95 + ((7f * vw0_) * _e93));
    let _e98 = base_uv;
    let _e102 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e98 + (vec2<f32>(u2_, v0_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e104 = sum;
    sum = (_e104 + ((uw2_ * vw0_) * _e102));
    let _e107 = base_uv;
    let _e111 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e107 + (vec2<f32>(u0_, v1_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e113 = sum;
    sum = (_e113 + ((uw0_ * 7f) * _e111));
    let _e116 = base_uv;
    let _e120 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e116 + (vec2<f32>(u1_, v1_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e122 = sum;
    sum = (_e122 + ((7f * 7f) * _e120));
    let _e125 = base_uv;
    let _e129 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e125 + (vec2<f32>(u2_, v1_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e131 = sum;
    sum = (_e131 + ((uw2_ * 7f) * _e129));
    let _e134 = base_uv;
    let _e138 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e134 + (vec2<f32>(u0_, v2_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e140 = sum;
    sum = (_e140 + ((uw0_ * vw2_) * _e138));
    let _e143 = base_uv;
    let _e147 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e143 + (vec2<f32>(u1_, v2_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e149 = sum;
    sum = (_e149 + ((7f * vw2_) * _e147));
    let _e152 = base_uv;
    let _e156 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((_e152 + (vec2<f32>(u2_, v2_) * inv_shadow_map_size)), depth_1, array_index_1);
    let _e158 = sum;
    sum = (_e158 + ((uw2_ * vw2_) * _e156));
    let _e160 = sum;
    return (_e160 * 0.0069444445f);
}

fn sample_shadow_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_2: vec2<f32>, depth_2: f32, array_index_2: i32, texel_size: f32) -> f32 {
    let _e3 = sample_shadow_map_castano_thirteenX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_2, depth_2, array_index_2);
    return _e3;
}

fn search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_3: vec2<f32>, depth_3: f32, array_index_3: i32) -> vec2<f32> {
    return vec2(0f);
}

fn search_for_blockers_in_shadow_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_4: vec2<f32>, depth_4: f32, array_index_4: i32, texel_size_1: f32, search_size: f32) -> f32 {
    var sum_1: vec2<f32> = vec2(0f);

    let _e3 = textureDimensions(directional_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX);
    let shadow_map_size_1 = vec2<f32>(_e3);
    let uv_offset_scale = (vec2(search_size) / (texel_size_1 * shadow_map_size_1));
    let offset0_ = (vec2<f32>(0.125f, -0.375f) * uv_offset_scale);
    let offset1_ = (vec2<f32>(-0.125f, 0.375f) * uv_offset_scale);
    let offset2_ = (vec2<f32>(0.625f, 0.125f) * uv_offset_scale);
    let offset3_ = (vec2<f32>(-0.375f, -0.625f) * uv_offset_scale);
    let offset4_ = (vec2<f32>(-0.625f, 0.625f) * uv_offset_scale);
    let offset5_ = (vec2<f32>(-0.875f, -0.125f) * uv_offset_scale);
    let offset6_ = (vec2<f32>(0.375f, 0.875f) * uv_offset_scale);
    let offset7_ = (vec2<f32>(0.875f, -0.875f) * uv_offset_scale);
    let _e46 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset0_), depth_4, array_index_4);
    let _e48 = sum_1;
    sum_1 = (_e48 + _e46);
    let _e51 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset1_), depth_4, array_index_4);
    let _e52 = sum_1;
    sum_1 = (_e52 + _e51);
    let _e55 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset2_), depth_4, array_index_4);
    let _e56 = sum_1;
    sum_1 = (_e56 + _e55);
    let _e59 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset3_), depth_4, array_index_4);
    let _e60 = sum_1;
    sum_1 = (_e60 + _e59);
    let _e63 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset4_), depth_4, array_index_4);
    let _e64 = sum_1;
    sum_1 = (_e64 + _e63);
    let _e67 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset5_), depth_4, array_index_4);
    let _e68 = sum_1;
    sum_1 = (_e68 + _e67);
    let _e71 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset6_), depth_4, array_index_4);
    let _e72 = sum_1;
    sum_1 = (_e72 + _e71);
    let _e75 = search_for_blockers_in_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_4 + offset7_), depth_4, array_index_4);
    let _e76 = sum_1;
    sum_1 = (_e76 + _e75);
    let _e79 = sum_1.y;
    if (_e79 == 0f) {
        return 0f;
    }
    let _e84 = sum_1.x;
    let _e86 = sum_1.y;
    return (_e84 / _e86);
}

fn random_rotation_matrixX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(scale: vec2<f32>, temporal: bool) -> mat2x2<f32> {
    let _e4 = globalsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.frame_count;
    let _e7 = interleaved_gradient_noiseX_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX(scale, select(1u, _e4, temporal));
    let random_angle = (6.2831855f * _e7);
    let m = vec2<f32>(sin(random_angle), cos(random_angle));
    return mat2x2<f32>(vec2<f32>(m.y, -(m.x)), vec2<f32>(m.x, m.y));
}

fn mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(min1_: f32, max1_: f32, min2_: f32, max2_: f32, value: f32) -> f32 {
    return (min2_ + (((value - min1_) * (max2_ - min2_)) / (max1_ - min1_)));
}

fn calculate_uv_offset_scale_jimenez_fourteenX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(texel_size_2: f32, blur_size: f32) -> vec2<f32> {
    let _e1 = textureDimensions(directional_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX);
    let shadow_map_size_2 = vec2<f32>(_e1);
    let _e8 = mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(0.00390625f, 0.022949219f, 0.015f, 0.035f, texel_size_2);
    return (vec2((_e8 * blur_size)) / (texel_size_2 * shadow_map_size_2));
}

fn sample_shadow_map_jimenez_fourteenX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_5: vec2<f32>, depth_5: f32, array_index_5: i32, texel_size_3: f32, blur_size_1: f32, temporal_1: bool) -> f32 {
    var sum_2: f32 = 0f;

    let _e2 = textureDimensions(directional_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX);
    let shadow_map_size_3 = vec2<f32>(_e2);
    let _e7 = random_rotation_matrixX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 * shadow_map_size_3), temporal_1);
    let _e10 = calculate_uv_offset_scale_jimenez_fourteenX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(texel_size_3, blur_size_1);
    let sample_offset0_ = ((_e7 * SPIRAL_OFFSET_0_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let sample_offset1_ = ((_e7 * SPIRAL_OFFSET_1_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let sample_offset2_ = ((_e7 * SPIRAL_OFFSET_2_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let sample_offset3_ = ((_e7 * SPIRAL_OFFSET_3_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let sample_offset4_ = ((_e7 * SPIRAL_OFFSET_4_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let sample_offset5_ = ((_e7 * SPIRAL_OFFSET_5_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let sample_offset6_ = ((_e7 * SPIRAL_OFFSET_6_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let sample_offset7_ = ((_e7 * SPIRAL_OFFSET_7_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * _e10);
    let _e38 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset0_), depth_5, array_index_5);
    let _e40 = sum_2;
    sum_2 = (_e40 + _e38);
    let _e43 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset1_), depth_5, array_index_5);
    let _e44 = sum_2;
    sum_2 = (_e44 + _e43);
    let _e47 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset2_), depth_5, array_index_5);
    let _e48 = sum_2;
    sum_2 = (_e48 + _e47);
    let _e51 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset3_), depth_5, array_index_5);
    let _e52 = sum_2;
    sum_2 = (_e52 + _e51);
    let _e55 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset4_), depth_5, array_index_5);
    let _e56 = sum_2;
    sum_2 = (_e56 + _e55);
    let _e59 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset5_), depth_5, array_index_5);
    let _e60 = sum_2;
    sum_2 = (_e60 + _e59);
    let _e63 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset6_), depth_5, array_index_5);
    let _e64 = sum_2;
    sum_2 = (_e64 + _e63);
    let _e67 = sample_shadow_map_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((light_local_5 + sample_offset7_), depth_5, array_index_5);
    let _e68 = sum_2;
    sum_2 = (_e68 + _e67);
    let _e70 = sum_2;
    return (_e70 / 8f);
}

fn sample_shadow_map_pcssX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_6: vec2<f32>, depth_6: f32, array_index_6: i32, texel_size_4: f32, light_size: f32) -> f32 {
    let _e5 = search_for_blockers_in_shadow_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_6, depth_6, array_index_6, texel_size_4, light_size);
    let blur_size_2 = max((((_e5 - depth_6) * light_size) / depth_6), 0.5f);
    let _e12 = sample_shadow_map_jimenez_fourteenX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_6, depth_6, array_index_6, texel_size_4, blur_size_2, false);
    return _e12;
}

fn sample_shadow_cubemap_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_7: vec3<f32>, depth_7: f32, light_id_3: u32) -> f32 {
    let _e6 = textureSampleCompareLevel(point_shadow_texturesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, point_shadow_textures_comparison_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, light_local_7, i32(light_id_3), depth_7);
    return _e6;
}

fn sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(position: vec2<f32>, coeff: f32, x_basis: vec3<f32>, y_basis: vec3<f32>, light_local_8: vec3<f32>, depth_8: f32, light_id_4: u32) -> f32 {
    let _e12 = sample_shadow_cubemap_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(((light_local_8 + (position.x * x_basis)) + (position.y * y_basis)), depth_8, light_id_4);
    return (_e12 * coeff);
}

fn sample_shadow_cubemap_gaussianX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_9: vec3<f32>, depth_9: f32, scale_1: f32, distance_to_light: f32, light_id_5: u32) -> f32 {
    var up_1: vec3<f32> = vec3<f32>(0f, 1f, 0f);
    var sum_3: f32 = 0f;

    let _e5 = up_1;
    if (dot(_e5, normalize(light_local_9)) > 0.99f) {
        up_1 = vec3<f32>(1f, 0f, 0f);
    }
    let _e14 = up_1;
    let _e15 = orthonormalizeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(light_local_9, _e14);
    let basis = ((_e15 * scale_1) * distance_to_light);
    let _e28 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.125f, -0.375f), 0.157112f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e30 = sum_3;
    sum_3 = (_e30 + _e28);
    let _e38 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.125f, 0.375f), 0.157112f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e39 = sum_3;
    sum_3 = (_e39 + _e38);
    let _e47 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.625f, 0.125f), 0.138651f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e48 = sum_3;
    sum_3 = (_e48 + _e47);
    let _e56 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.375f, -0.625f), 0.130251f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e57 = sum_3;
    sum_3 = (_e57 + _e56);
    let _e65 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.625f, 0.625f), 0.114946f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e66 = sum_3;
    sum_3 = (_e66 + _e65);
    let _e74 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.875f, -0.125f), 0.114946f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e75 = sum_3;
    sum_3 = (_e75 + _e74);
    let _e83 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.375f, 0.875f), 0.107982f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e84 = sum_3;
    sum_3 = (_e84 + _e83);
    let _e92 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.875f, -0.875f), 0.079001f, basis[0], basis[1], light_local_9, depth_9, light_id_5);
    let _e93 = sum_3;
    sum_3 = (_e93 + _e92);
    let _e95 = sum_3;
    return _e95;
}

fn sample_shadow_cubemapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_10: vec3<f32>, distance_to_light_1: f32, depth_10: f32, light_id_6: u32) -> f32 {
    let _e5 = sample_shadow_cubemap_gaussianX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_10, depth_10, POINT_SHADOW_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX, distance_to_light_1, light_id_6);
    return _e5;
}

fn search_for_blockers_in_shadow_cubemap_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_11: vec3<f32>, depth_11: f32, light_id_7: u32) -> vec2<f32> {
    return vec2(0f);
}

fn search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(position_1: vec2<f32>, x_basis_1: vec3<f32>, y_basis_1: vec3<f32>, light_local_12: vec3<f32>, depth_12: f32, light_id_8: u32) -> vec2<f32> {
    let _e12 = search_for_blockers_in_shadow_cubemap_hardwareX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(((light_local_12 + (position_1.x * x_basis_1)) + (position_1.y * y_basis_1)), depth_12, light_id_8);
    return _e12;
}

fn search_for_blockers_in_shadow_cubemapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_13: vec3<f32>, depth_13: f32, scale_2: f32, distance_to_light_2: f32, light_id_9: u32) -> f32 {
    var up_2: vec3<f32> = vec3<f32>(0f, 1f, 0f);
    var sum_4: vec2<f32> = vec2(0f);

    let _e6 = up_2;
    if (dot(_e6, normalize(light_local_13)) > 0.99f) {
        up_2 = vec3<f32>(1f, 0f, 0f);
    }
    let _e15 = up_2;
    let _e16 = orthonormalizeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(light_local_13, _e15);
    let basis_1 = ((_e16 * scale_2) * distance_to_light_2);
    let _e28 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.125f, -0.375f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e30 = sum_4;
    sum_4 = (_e30 + _e28);
    let _e37 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.125f, 0.375f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e38 = sum_4;
    sum_4 = (_e38 + _e37);
    let _e45 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.625f, 0.125f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e46 = sum_4;
    sum_4 = (_e46 + _e45);
    let _e53 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.375f, -0.625f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e54 = sum_4;
    sum_4 = (_e54 + _e53);
    let _e61 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.625f, 0.625f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e62 = sum_4;
    sum_4 = (_e62 + _e61);
    let _e69 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(-0.875f, -0.125f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e70 = sum_4;
    sum_4 = (_e70 + _e69);
    let _e77 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.375f, 0.875f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e78 = sum_4;
    sum_4 = (_e78 + _e77);
    let _e85 = search_for_blockers_in_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2<f32>(0.875f, -0.875f), basis_1[0], basis_1[1], light_local_13, depth_13, light_id_9);
    let _e86 = sum_4;
    sum_4 = (_e86 + _e85);
    let _e89 = sum_4.y;
    if (_e89 == 0f) {
        return 0f;
    }
    let _e94 = sum_4.x;
    let _e96 = sum_4.y;
    return (_e94 / _e96);
}

fn sample_shadow_cubemap_jitteredX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_14: vec3<f32>, depth_14: f32, scale_3: f32, distance_to_light_3: f32, light_id_10: u32, temporal_2: bool) -> f32 {
    var up_3: vec3<f32> = vec3<f32>(0f, 1f, 0f);
    var sum_5: f32 = 0f;

    let _e5 = up_3;
    if (dot(_e5, normalize(light_local_14)) > 0.99f) {
        up_3 = vec3<f32>(1f, 0f, 0f);
    }
    let _e14 = up_3;
    let _e15 = orthonormalizeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(light_local_14, _e14);
    let basis_2 = ((_e15 * scale_3) * distance_to_light_3);
    let _e23 = random_rotation_matrixX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(vec2(1f), temporal_2);
    let sample_offset0_1 = ((_e23 * SPIRAL_OFFSET_0_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let sample_offset1_1 = ((_e23 * SPIRAL_OFFSET_1_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let sample_offset2_1 = ((_e23 * SPIRAL_OFFSET_2_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let sample_offset3_1 = ((_e23 * SPIRAL_OFFSET_3_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let sample_offset4_1 = ((_e23 * SPIRAL_OFFSET_4_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let sample_offset5_1 = ((_e23 * SPIRAL_OFFSET_5_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let sample_offset6_1 = ((_e23 * SPIRAL_OFFSET_6_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let sample_offset7_1 = ((_e23 * SPIRAL_OFFSET_7_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX) * POINT_SHADOW_TEMPORAL_OFFSET_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    let _e61 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset0_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e63 = sum_5;
    sum_5 = (_e63 + _e61);
    let _e68 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset1_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e69 = sum_5;
    sum_5 = (_e69 + _e68);
    let _e74 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset2_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e75 = sum_5;
    sum_5 = (_e75 + _e74);
    let _e80 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset3_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e81 = sum_5;
    sum_5 = (_e81 + _e80);
    let _e86 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset4_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e87 = sum_5;
    sum_5 = (_e87 + _e86);
    let _e92 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset5_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e93 = sum_5;
    sum_5 = (_e93 + _e92);
    let _e98 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset6_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e99 = sum_5;
    sum_5 = (_e99 + _e98);
    let _e104 = sample_shadow_cubemap_at_offsetX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(sample_offset7_1, 0.125f, basis_2[0], basis_2[1], light_local_14, depth_14, light_id_10);
    let _e105 = sum_5;
    sum_5 = (_e105 + _e104);
    let _e107 = sum_5;
    return _e107;
}

fn sample_shadow_cubemap_pcssX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_15: vec3<f32>, distance_to_light_4: f32, depth_15: f32, light_id_11: u32, light_size_1: f32) -> f32 {
    let _e5 = search_for_blockers_in_shadow_cubemapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_15, depth_15, light_size_1, distance_to_light_4, light_id_11);
    let blur_size_3 = max((((_e5 - depth_15) * light_size_1) / depth_15), 0.5f);
    let _e14 = sample_shadow_cubemap_jitteredX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(light_local_15, depth_15, (POINT_SHADOW_SCALEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX * blur_size_3), distance_to_light_4, light_id_11, false);
    return _e14;
}

fn hsv_to_rgbX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DUY3PNRXXEX3POBSXEYLUNFXW44YX(hsv: vec3<f32>) -> vec3<f32> {
    const n = vec3<f32>(5f, 3f, 1f);
    let k_1 = ((n + vec3((hsv.x / FRAC_PI_3X_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX))) % vec3(6f));
    return (vec3(hsv.z) - ((hsv.z * hsv.y) * max(vec3(0f), min(k_1, min((vec3(4f) - k_1), vec3(1f))))));
}

fn fetch_point_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_12: u32, frag_position: vec4<f32>, surface_normal: vec3<f32>) -> f32 {
    let light_3 = (&clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[light_id_12]);
    let _e6 = (*light_3).position_radius;
    let surface_to_light = (_e6.xyz - frag_position.xyz);
    let surface_to_light_abs = abs(surface_to_light);
    let distance_to_light_5 = max(surface_to_light_abs.x, max(surface_to_light_abs.y, surface_to_light_abs.z));
    let _e18 = (*light_3).shadow_normal_bias;
    let normal_offset = ((_e18 * distance_to_light_5) * surface_normal.xyz);
    let _e23 = (*light_3).shadow_depth_bias;
    let depth_offset = (_e23 * normalize(surface_to_light.xyz));
    let offset_position_1 = ((frag_position.xyz + normal_offset) + depth_offset);
    let _e32 = (*light_3).position_radius;
    let frag_ls = (offset_position_1.xyz - _e32.xyz);
    let abs_position_ls = abs(frag_ls);
    let major_axis_magnitude = max(abs_position_ls.x, max(abs_position_ls.y, abs_position_ls.z));
    let _e43 = (*light_3).light_custom_data;
    let _e47 = (*light_3).light_custom_data;
    let zw = ((-(major_axis_magnitude) * _e43.xy) + _e47.zw);
    let depth_16 = (zw.x / zw.y);
    let _e54 = (*light_3).soft_shadow_size;
    if (_e54 > 0f) {
        let _e60 = (*light_3).soft_shadow_size;
        let _e61 = sample_shadow_cubemap_pcssX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((frag_ls * flip_zX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX), distance_to_light_5, depth_16, light_id_12, _e60);
        return _e61;
    }
    let _e64 = sample_shadow_cubemapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX((frag_ls * flip_zX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX), distance_to_light_5, depth_16, light_id_12);
    return _e64;
}

fn fetch_spot_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_13: u32, frag_position_1: vec4<f32>, surface_normal_1: vec3<f32>, near_z: f32) -> f32 {
    var spot_dir_1: vec3<f32>;
    var sign: f32 = -1f;

    let light_4 = (&clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[light_id_13]);
    let _e7 = (*light_4).position_radius;
    let surface_to_light_1 = (_e7.xyz - frag_position_1.xyz);
    let _e13 = (*light_4).light_custom_data.x;
    let _e16 = (*light_4).light_custom_data.y;
    spot_dir_1 = vec3<f32>(_e13, 0f, _e16);
    let _e22 = spot_dir_1.x;
    let _e24 = spot_dir_1.x;
    let _e30 = spot_dir_1.z;
    let _e32 = spot_dir_1.z;
    spot_dir_1.y = sqrt(max(0f, ((1f - (_e22 * _e24)) - (_e30 * _e32))));
    let _e38 = (*light_4).flags;
    if ((_e38 & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u) {
        let _e45 = spot_dir_1.y;
        spot_dir_1.y = -(_e45);
    }
    let _e47 = spot_dir_1;
    let fwd = -(_e47);
    let distance_to_light_6 = dot(fwd, surface_to_light_1);
    let _e53 = (*light_4).shadow_depth_bias;
    let _e59 = (*light_4).shadow_normal_bias;
    let offset_position_2 = ((-(surface_to_light_1) + (_e53 * normalize(surface_to_light_1))) + ((surface_normal_1.xyz * _e59) * distance_to_light_6));
    if (fwd.z >= 0f) {
        sign = 1f;
    }
    let _e69 = sign;
    let a_2 = (-1f / (fwd.z + _e69));
    let b = ((fwd.x * fwd.y) * a_2);
    let _e77 = sign;
    let _e85 = sign;
    let _e87 = sign;
    let up_dir = vec3<f32>((1f + (((_e77 * fwd.x) * fwd.x) * a_2)), (_e85 * b), (-(_e87) * fwd.x));
    let _e93 = sign;
    let right_dir = vec3<f32>(-(b), (-(_e93) - ((fwd.y * fwd.y) * a_2)), fwd.y);
    let light_inv_rot = mat3x3<f32>(right_dir, up_dir, fwd);
    let projected_position = (offset_position_2 * light_inv_rot);
    let _e105 = (*light_4).spot_light_tan_angle;
    let f_div_minus_z = (1f / (_e105 * -(projected_position.z)));
    let shadow_xy_ndc = (projected_position.xy * f_div_minus_z);
    let shadow_uv = ((shadow_xy_ndc * vec2<f32>(0.5f, -0.5f)) + vec2<f32>(0.5f, 0.5f));
    let depth_17 = (near_z / -(projected_position.z));
    let _e128 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.spot_light_shadowmap_offset;
    let array_index_7 = (i32(light_id_13) + _e128);
    let _e131 = (*light_4).soft_shadow_size;
    if (_e131 > 0f) {
        let _e135 = (*light_4).soft_shadow_size;
        let _e137 = sample_shadow_map_pcssX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(shadow_uv, depth_17, array_index_7, SPOT_SHADOW_TEXEL_SIZEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX, _e135);
        return _e137;
    }
    let _e139 = sample_shadow_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(shadow_uv, depth_17, array_index_7, SPOT_SHADOW_TEXEL_SIZEX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX);
    return _e139;
}

fn get_cascade_indexX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_14: u32, view_z_3: f32) -> u32 {
    var i: u32 = 0u;

    let light_5 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[light_id_14]);
    loop {
        let _e6 = i;
        let _e8 = (*light_5).num_cascades;
        if (_e6 < _e8) {
        } else {
            break;
        }
        {
            let _e13 = i;
            let _e16 = (*light_5).cascades[_e13].far_bound;
            if (-(view_z_3) < _e16) {
                let _e18 = i;
                return _e18;
            }
        }
        continuing {
            let _e19 = i;
            i = (_e19 + 1u);
        }
    }
    let _e23 = (*light_5).num_cascades;
    return _e23;
}

fn world_to_directional_light_localX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_15: u32, cascade_index: u32, offset_position: vec4<f32>) -> vec4<f32> {
    let light_6 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[light_id_15]);
    let cascade = (&(*light_6).cascades[cascade_index]);
    let _e9 = (*cascade).clip_from_world;
    let offset_position_clip = (_e9 * offset_position);
    if (offset_position_clip.w <= 0f) {
        return vec4(0f);
    }
    let offset_position_ndc = (offset_position_clip.xyz / vec3(offset_position_clip.w));
    if ((any((offset_position_ndc.xy < vec2(-1f))) || (offset_position_ndc.z < 0f)) || any((offset_position_ndc > vec3(1f)))) {
        return vec4(0f);
    }
    const flip_correction = vec2<f32>(0.5f, -0.5f);
    let light_local_16 = ((offset_position_ndc.xy * flip_correction) + vec2<f32>(0.5f, 0.5f));
    let depth_18 = offset_position_ndc.z;
    return vec4<f32>(light_local_16, depth_18, 1f);
}

fn sample_directional_cascadeX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_16: u32, cascade_index_1: u32, frag_position_2: vec4<f32>, surface_normal_2: vec3<f32>) -> f32 {
    let light_7 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[light_id_16]);
    let cascade_1 = (&(*light_7).cascades[cascade_index_1]);
    let _e9 = (*light_7).shadow_normal_bias;
    let _e11 = (*cascade_1).texel_size;
    let normal_offset_1 = ((_e9 * _e11) * surface_normal_2.xyz);
    let _e16 = (*light_7).shadow_depth_bias;
    let _e18 = (*light_7).direction_to_light;
    let depth_offset_1 = (_e16 * _e18.xyz);
    let offset_position_3 = vec4<f32>(((frag_position_2.xyz + normal_offset_1) + depth_offset_1), frag_position_2.w);
    let _e27 = world_to_directional_light_localX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_16, cascade_index_1, offset_position_3);
    if (_e27.w == 0f) {
        return 1f;
    }
    let _e33 = (*light_7).depth_texture_base_index;
    let array_index_8 = i32((_e33 + cascade_index_1));
    let texel_size_5 = (*cascade_1).texel_size;
    let _e39 = (*light_7).soft_shadow_size;
    if (_e39 > 0f) {
        let _e45 = (*light_7).soft_shadow_size;
        let _e46 = sample_shadow_map_pcssX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(_e27.xy, _e27.z, array_index_8, texel_size_5, _e45);
        return _e46;
    }
    let _e49 = sample_shadow_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5PXGYLNOBWGS3THX(_e27.xy, _e27.z, array_index_8, texel_size_5);
    return _e49;
}

fn fetch_directional_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_17: u32, frag_position_3: vec4<f32>, surface_normal_3: vec3<f32>, view_z_4: f32) -> f32 {
    var shadow: f32;

    let light_8 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[light_id_17]);
    let _e5 = get_cascade_indexX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_17, view_z_4);
    let _e7 = (*light_8).num_cascades;
    if (_e5 >= _e7) {
        return 1f;
    }
    let _e12 = sample_directional_cascadeX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_17, _e5, frag_position_3, surface_normal_3);
    shadow = _e12;
    let next_cascade_index = (_e5 + 1u);
    let _e17 = (*light_8).num_cascades;
    if (next_cascade_index < _e17) {
        let this_far_bound = (*light_8).cascades[_e5].far_bound;
        let _e24 = (*light_8).cascades_overlap_proportion;
        let next_near_bound = ((1f - _e24) * this_far_bound);
        if (-(view_z_4) >= next_near_bound) {
            let _e30 = sample_directional_cascadeX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(light_id_17, next_cascade_index, frag_position_3, surface_normal_3);
            let _e31 = shadow;
            shadow = mix(_e31, _e30, ((-(view_z_4) - next_near_bound) / (this_far_bound - next_near_bound)));
        }
    }
    let _e37 = shadow;
    return _e37;
}

fn transpose_affine_matrixX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX(matrix: mat3x4<f32>) -> mat4x4<f32> {
    let matrix4x4_ = mat4x4<f32>(matrix[0], matrix[1], matrix[2], vec4<f32>(0f, 0f, 0f, 1f));
    return transpose(matrix4x4_);
}

fn query_light_probeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX(world_position: vec3<f32>, is_irradiance_volume: bool, clusterable_object_index_ranges_1: ptr<function, ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX>) -> LightProbeQueryResultX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX {
    var result: LightProbeQueryResultX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX;
    var start_offset: u32;
    var end_offset: u32;
    var light_probe_index_offset: u32;
    var light_probe: LightProbeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;

    result.texture_index = -1i;
    if is_irradiance_volume {
        let _e6 = (*clusterable_object_index_ranges_1).first_irradiance_volume_index_offset;
        start_offset = _e6;
        let _e9 = (*clusterable_object_index_ranges_1).first_decal_offset;
        end_offset = _e9;
    } else {
        let _e12 = (*clusterable_object_index_ranges_1).first_reflection_probe_index_offset;
        start_offset = _e12;
        let _e14 = (*clusterable_object_index_ranges_1).first_irradiance_volume_index_offset;
        end_offset = _e14;
    }
    let _e15 = start_offset;
    light_probe_index_offset = _e15;
    loop {
        let _e17 = light_probe_index_offset;
        let _e18 = end_offset;
        let _e21 = result.texture_index;
        if ((_e17 < _e18) && (_e21 < 0i)) {
        } else {
            break;
        }
        {
            let _e25 = light_probe_index_offset;
            let _e26 = get_clusterable_object_idX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e25);
            let light_probe_index = i32(_e26);
            if is_irradiance_volume {
                let _e31 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.irradiance_volumes[light_probe_index];
                light_probe = _e31;
            } else {
                let _e36 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.reflection_probes[light_probe_index];
                light_probe = _e36;
            }
            let _e38 = light_probe.light_from_world_transposed;
            let _e39 = transpose_affine_matrixX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX(_e38);
            let probe_space_pos = (_e39 * vec4<f32>(world_position, 1f)).xyz;
            if all((abs(probe_space_pos) <= vec3(0.5f))) {
                let _e52 = light_probe.cubemap_index;
                result.texture_index = _e52;
                let _e55 = light_probe.intensity;
                result.intensity = _e55;
                result.light_from_world = _e39;
                let _e59 = light_probe.affects_lightmapped_mesh_diffuse;
                result.affects_lightmapped_mesh_diffuse = (_e59 != 0u);
                break;
            }
        }
        continuing {
            let _e63 = light_probe_index_offset;
            light_probe_index_offset = (_e63 + 1u);
        }
    }
    let _e65 = result;
    return _e65;
}

fn radiance_sample_directionX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(N_1: vec3<f32>, R: vec3<f32>, roughness_2: f32) -> vec3<f32> {
    let smoothness = saturate((1f - roughness_2));
    let lerp_factor = (smoothness * (sqrt(smoothness) + roughness_2));
    return mix(N_1, R, lerp_factor);
}

fn compute_radiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(input_9: LayerLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX, clusterable_object_index_ranges_2: ptr<function, ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX>, world_position_1: vec3<f32>, found_diffuse_indirect: bool) -> EnvironmentMapRadiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    var radiances: EnvironmentMapRadiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX;
    var query_result: LightProbeQueryResultX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX;
    var enable_diffuse_3: bool;
    var irradiance_sample_dir: vec3<f32>;
    var radiance_sample_dir: vec3<f32>;

    let N_5 = input_9.N;
    let R_2 = input_9.R;
    let NdotV_6 = input_9.NdotV;
    let perceptual_roughness_3 = input_9.perceptual_roughness;
    let roughness_7 = input_9.roughness;
    let _e9 = query_light_probeX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUL5YHE33CMUX(world_position_1, false, clusterable_object_index_ranges_2);
    query_result = _e9;
    let _e12 = query_result.texture_index;
    if (_e12 < 0i) {
        let _e18 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_cubemap_index;
        query_result.texture_index = _e18;
        let _e22 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.intensity_for_view;
        query_result.intensity = _e22;
        let _e26 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_environment_map_affects_lightmapped_mesh_diffuse;
        query_result.affects_lightmapped_mesh_diffuse = (_e26 != 0u);
    }
    let _e30 = query_result.texture_index;
    if (_e30 < 0i) {
        radiances.irradiance = vec3(0f);
        radiances.radiance = vec3(0f);
        let _e40 = radiances;
        return _e40;
    }
    let _e43 = query_result.texture_index;
    let _e45 = textureNumLevels(specular_environment_mapsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX[_e43]);
    let radiance_level = (perceptual_roughness_3 * f32((_e45 - 1u)));
    enable_diffuse_3 = !(found_diffuse_indirect);
    let _e53 = enable_diffuse_3;
    if _e53 {
        irradiance_sample_dir = N_5;
        let _e57 = environment_map_uniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.transform;
        let _e58 = irradiance_sample_dir;
        irradiance_sample_dir = (_e57 * vec4<f32>(_e58, 1f)).xyz;
        let _e65 = irradiance_sample_dir.z;
        irradiance_sample_dir.z = -(_e65);
        let _e70 = query_result.texture_index;
        let _e72 = irradiance_sample_dir;
        let _e75 = textureSampleLevel(diffuse_environment_mapsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX[_e70], environment_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, _e72, 0f);
        let _e78 = query_result.intensity;
        radiances.irradiance = (_e75.xyz * _e78);
    }
    let _e80 = radiance_sample_directionX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(N_5, R_2, roughness_7);
    radiance_sample_dir = _e80;
    let _e84 = environment_map_uniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.transform;
    let _e85 = radiance_sample_dir;
    radiance_sample_dir = (_e84 * vec4<f32>(_e85, 1f)).xyz;
    let _e92 = radiance_sample_dir.z;
    radiance_sample_dir.z = -(_e92);
    let _e97 = query_result.texture_index;
    let _e100 = radiance_sample_dir;
    let _e101 = textureSampleLevel(specular_environment_mapsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX[_e97], environment_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, _e100, radiance_level);
    let _e104 = query_result.intensity;
    radiances.radiance = (_e101.xyz * _e104);
    let _e106 = radiances;
    return _e106;
}

fn environment_map_light_clearcoatX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(out_1: ptr<function, EnvironmentMapLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX>, input_10: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, clusterable_object_index_ranges_3: ptr<function, ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX>, found_diffuse_indirect_1: bool) {
    let world_position_5 = (*input_10).P;
    let clearcoat_NdotV = (*input_10).layers[1].NdotV;
    let clearcoat_strength_3 = (*input_10).clearcoat_strength;
    const clearcoat_F0_ = vec3(0.04f);
    let _e12 = F_Schlick_vecX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(clearcoat_F0_, 1f, clearcoat_NdotV);
    let Fc_1 = (_e12 * clearcoat_strength_3);
    let inv_Fc_2 = (vec3(1f) - Fc_1);
    let _e19 = (*input_10).layers[1];
    let _e22 = compute_radiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(_e19, clusterable_object_index_ranges_3, world_position_5, found_diffuse_indirect_1);
    let _e25 = (*out_1).diffuse;
    (*out_1).diffuse = (_e25 * inv_Fc_2);
    let _e29 = (*out_1).specular;
    (*out_1).specular = (((_e29 * inv_Fc_2) * inv_Fc_2) + (_e22.radiance * Fc_1));
    return;
}

fn environment_map_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(input_11: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, clusterable_object_index_ranges_4: ptr<function, ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX>, found_diffuse_indirect_2: bool) -> EnvironmentMapLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    var out_2: EnvironmentMapLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX;

    let roughness_8 = (*input_11).layers[0].roughness;
    let diffuse_color_3 = (*input_11).diffuse_color;
    let NdotV_7 = (*input_11).layers[0].NdotV;
    let F_ab_2 = (*input_11).F_ab;
    let F0_3 = (*input_11).F0_;
    let world_position_6 = (*input_11).P;
    let _e19 = (*input_11).layers[0];
    let _e22 = compute_radiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(_e19, clusterable_object_index_ranges_4, world_position_6, found_diffuse_indirect_2);
    if (all((_e22.irradiance == vec3(0f))) && all((_e22.radiance == vec3(0f)))) {
        out_2.diffuse = vec3(0f);
        out_2.specular = vec3(0f);
        let _e41 = out_2;
        return _e41;
    }
    let specular_occlusion_1 = saturate(dot(F0_3, vec3(16.5f)));
    let Fr_1 = (max(vec3((1f - roughness_8)), F0_3) - F0_3);
    let kS = (F0_3 + (Fr_1 * pow((1f - NdotV_7), 5f)));
    let Ess = (F_ab_2.x + F_ab_2.y);
    let FssEss = ((kS * Ess) * specular_occlusion_1);
    let Ems = (1f - Ess);
    let Favg = (F0_3 + ((vec3(1f) - F0_3) / vec3(21f)));
    let Fms = ((FssEss * Favg) / (vec3(1f) - (Ems * Favg)));
    let FmsEms = (Fms * Ems);
    let Edss = (vec3(1f) - (FssEss + FmsEms));
    let kD = (diffuse_color_3 * Edss);
    if !(found_diffuse_indirect_2) {
        out_2.diffuse = ((FmsEms + kD) * _e22.irradiance);
    } else {
        out_2.diffuse = vec3(0f);
    }
    out_2.specular = (FssEss * _e22.radiance);
    environment_map_light_clearcoatX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX((&out_2), input_11, clusterable_object_index_ranges_4, found_diffuse_indirect_2);
    let _e94 = out_2;
    return _e94;
}

fn EnvBRDFApproxX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(F0_: vec3<f32>, F_ab: vec2<f32>) -> vec3<f32> {
    return ((F0_ * F_ab.x) + vec3(F_ab.y));
}

fn ambient_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MFWWE2LFNZ2AX(world_position_2: vec4<f32>, world_normal: vec3<f32>, V_3: vec3<f32>, NdotV_2: f32, diffuse_color: vec3<f32>, specular_color: vec3<f32>, perceptual_roughness_1: f32, occlusion: vec3<f32>) -> vec3<f32> {
    let _e2 = F_ABX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(1f, NdotV_2);
    let _e4 = EnvBRDFApproxX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(diffuse_color, _e2);
    let _e6 = F_ABX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptual_roughness_1, NdotV_2);
    let _e8 = EnvBRDFApproxX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(specular_color, _e6);
    let specular_occlusion_2 = saturate(dot(specular_color, vec3(16.5f)));
    let _e18 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.ambient_color;
    return (((_e4 + (_e8 * specular_occlusion_2)) * _e18.xyz) * occlusion);
}

fn prepare_world_normalX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(world_normal_1: vec3<f32>, double_sided: bool, is_front_1: bool) -> vec3<f32> {
    var output: vec3<f32>;

    output = world_normal_1;
    let _e2 = output;
    return _e2;
}

fn calculate_tbn_mikktspaceX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(world_normal_2: vec3<f32>, world_tangent: vec4<f32>) -> mat3x3<f32> {
    var N_2: vec3<f32>;
    var T: vec3<f32>;
    var B: vec3<f32>;

    N_2 = world_normal_2;
    T = world_tangent.xyz;
    let _e6 = N_2;
    let _e7 = T;
    B = (world_tangent.w * cross(_e6, _e7));
    let _e11 = T;
    let _e12 = B;
    let _e13 = N_2;
    return mat3x3<f32>(_e11, _e12, _e13);
}

fn calculate_viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(world_position_3: vec4<f32>, is_orthographic_3: bool) -> vec3<f32> {
    var V_4: vec3<f32>;

    if is_orthographic_3 {
        let _e5 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_world[0][2];
        let _e10 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_world[1][2];
        let _e15 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_world[2][2];
        V_4 = normalize(vec3<f32>(_e5, _e10, _e15));
    } else {
        let _e22 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.world_position;
        V_4 = normalize((_e22.xyz - world_position_3.xyz));
    }
    let _e27 = V_4;
    return _e27;
}

fn sample_depth_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBQXEYLMNRQXQX3NMFYHA2LOM4X(uv: vec2<f32>, material_bind_group_slot: u32) -> f32 {
    let _e4 = textureSampleLevel(depth_map_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, depth_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, uv, 0f);
    return _e4.x;
}

fn parallaxed_uvX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBQXEYLMNRQXQX3NMFYHA2LOM4X(depth_scale: f32, max_layer_count: f32, max_steps: u32, original_uv: vec2<f32>, Vt: vec3<f32>, material_bind_group_slot_1: u32) -> vec2<f32> {
    var uv_1: vec2<f32>;
    var delta_uv: vec2<f32>;
    var current_layer_depth: f32 = 0f;
    var texture_depth: f32;
    var i_1: i32 = 0i;

    if (max_layer_count < 1f) {
        return original_uv;
    }
    uv_1 = original_uv;
    let view_steepness = abs(Vt.z);
    let layer_count = mix(max_layer_count, 1f, view_steepness);
    let layer_depth = (1f / layer_count);
    delta_uv = ((((depth_scale * layer_depth) * Vt.xy) * vec2<f32>(1f, -1f)) / vec2(view_steepness));
    let _e25 = uv_1;
    let _e27 = sample_depth_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBQXEYLMNRQXQX3NMFYHA2LOM4X(_e25, material_bind_group_slot_1);
    texture_depth = _e27;
    loop {
        let _e31 = texture_depth;
        let _e32 = current_layer_depth;
        let _e34 = i_1;
        if ((_e31 > _e32) && (_e34 <= i32(layer_count))) {
        } else {
            break;
        }
        {
            let _e38 = current_layer_depth;
            current_layer_depth = (_e38 + layer_depth);
            let _e40 = delta_uv;
            let _e41 = uv_1;
            uv_1 = (_e41 + _e40);
            let _e43 = uv_1;
            let _e44 = sample_depth_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBQXEYLMNRQXQX3NMFYHA2LOM4X(_e43, material_bind_group_slot_1);
            texture_depth = _e44;
        }
        continuing {
            let _e46 = i_1;
            i_1 = (_e46 + 1i);
        }
    }
    let _e48 = uv_1;
    let _e49 = delta_uv;
    let previous_uv = (_e48 - _e49);
    let _e51 = texture_depth;
    let _e52 = current_layer_depth;
    let next_depth = (_e51 - _e52);
    let _e54 = sample_depth_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBQXEYLMNRQXQX3NMFYHA2LOM4X(previous_uv, material_bind_group_slot_1);
    let _e55 = current_layer_depth;
    let previous_depth = ((_e54 - _e55) + layer_depth);
    let weight = (next_depth / (next_depth - previous_depth));
    let _e60 = uv_1;
    uv_1 = mix(_e60, previous_uv, weight);
    let _e63 = current_layer_depth;
    current_layer_depth = (_e63 + mix(next_depth, previous_depth, weight));
    let _e65 = uv_1;
    return _e65;
}

fn pbr_input_from_vertex_outputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOJQWO3LFNZ2AX(in_1: VertexOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX, is_front_2: bool, double_sided_1: bool) -> PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    var pbr_input_2: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;

    let _e0 = pbr_input_newX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX();
    pbr_input_2 = _e0;
    let _e8 = meshX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7MJUW4ZDJNZTXGX[in_1.instance_index].flags;
    pbr_input_2.flags = _e8;
    let _e14 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_view[3][3];
    pbr_input_2.is_orthographic = (_e14 == 1f);
    let _e20 = pbr_input_2.is_orthographic;
    let _e21 = calculate_viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(in_1.world_position, _e20);
    pbr_input_2.V = _e21;
    pbr_input_2.frag_coord = in_1.position;
    pbr_input_2.world_position = in_1.world_position;
    let _e30 = prepare_world_normalX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(in_1.world_normal, double_sided_1, is_front_2);
    pbr_input_2.world_normal = _e30;
    let _e33 = pbr_input_2.world_normal;
    pbr_input_2.N = normalize(_e33);
    let _e35 = pbr_input_2;
    return _e35;
}

fn pbr_input_from_standard_materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOJQWO3LFNZ2AX(in_2: VertexOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX, is_front_3: bool) -> PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    var pbr_input_3: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;
    var bias: SampleBiasX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX;
    var uv_2: vec2<f32>;
    var uv_b: vec2<f32>;
    var emissive: vec4<f32>;
    var metallic: f32;
    var perceptual_roughness_2: f32;
    var specular_transmission: f32;
    var thickness: f32;
    var diffuse_transmission: f32;
    var diffuse_occlusion: vec3<f32> = vec3(1f);
    var specular_occlusion: f32 = 1f;

    let _e7 = meshX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7MJUW4ZDJNZTXGX[in_2.instance_index].material_and_lightmap_bind_group_slot;
    let slot = (_e7 & 65535u);
    let flags = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.flags;
    let base_color_3 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.base_color;
    let deferred_lighting_pass_id = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.deferred_lighting_pass_id;
    let double_sided_2 = ((flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u);
    let _e24 = pbr_input_from_vertex_outputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOJQWO3LFNZ2AX(in_2, is_front_3, double_sided_2);
    pbr_input_3 = _e24;
    pbr_input_3.material.flags = flags;
    let _e30 = pbr_input_3.material.base_color;
    pbr_input_3.material.base_color = (_e30 * base_color_3);
    pbr_input_3.material.deferred_lighting_pass_id = deferred_lighting_pass_id;
    let _e35 = pbr_input_3.N;
    let _e37 = pbr_input_3.V;
    let NdotV_8 = max(dot(_e35, _e37), 0.0001f);
    let _e45 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.mip_bias;
    bias.mip_bias = _e45;
    let uv_transform = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.uv_transform;
    uv_2 = (uv_transform * vec3<f32>(in_2.uv, 1f)).xy;
    let _e55 = uv_2;
    uv_b = _e55;
    if ((flags & STANDARD_MATERIAL_FLAGS_DEPTH_MAP_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
        let V_7 = pbr_input_3.V;
        let _e65 = calculate_tbn_mikktspaceX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(in_2.world_normal, in_2.world_tangent);
        let T_1 = _e65[0];
        let B_1 = _e65[1];
        let N_6 = _e65[2];
        let Vt_1 = vec3<f32>(dot(V_7, T_1), dot(V_7, B_1), dot(V_7, N_6));
        let _e75 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.parallax_depth_scale;
        let _e78 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.max_parallax_layer_count;
        let _e81 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.max_relief_mapping_search_steps;
        let _e82 = uv_2;
        let _e84 = parallaxed_uvX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBQXEYLMNRQXQX3NMFYHA2LOM4X(_e75, _e78, _e81, _e82, -(Vt_1), slot);
        uv_2 = _e84;
        let _e85 = uv_2;
        uv_b = _e85;
    }
    if ((flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
        let _e94 = uv_2;
        let _e96 = bias.mip_bias;
        let _e97 = textureSampleBias(base_color_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, base_color_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, _e94, _e96);
        let _e98 = pbr_input_3.material.base_color;
        pbr_input_3.material.base_color = (_e98 * _e97);
    }
    pbr_input_3.material.flags = flags;
    if ((flags & STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) == 0u) {
        let _e110 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.ior;
        pbr_input_3.material.ior = _e110;
        let _e115 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.attenuation_color;
        pbr_input_3.material.attenuation_color = _e115;
        let _e120 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.attenuation_distance;
        pbr_input_3.material.attenuation_distance = _e120;
        let _e125 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.alpha_cutoff;
        pbr_input_3.material.alpha_cutoff = _e125;
        let _e130 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.reflectance;
        pbr_input_3.material.reflectance = _e130;
        let _e133 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.emissive;
        emissive = _e133;
        if ((flags & STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
            let _e139 = emissive;
            let _e143 = uv_2;
            let _e145 = bias.mip_bias;
            let _e146 = textureSampleBias(emissive_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, emissive_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, _e143, _e145);
            let _e150 = emissive.w;
            emissive = vec4<f32>((_e139.xyz * _e146.xyz), _e150);
        }
        let _e154 = emissive;
        pbr_input_3.material.emissive = _e154;
        let _e157 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.metallic;
        metallic = _e157;
        let _e161 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.perceptual_roughness;
        perceptual_roughness_2 = _e161;
        let _e163 = perceptual_roughness_2;
        let _e164 = perceptualRoughnessToRoughnessX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(_e163);
        if ((flags & STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
            let _e171 = uv_2;
            let _e173 = bias.mip_bias;
            let metallic_roughness = textureSampleBias(metallic_roughness_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, metallic_roughness_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, _e171, _e173);
            let _e176 = metallic;
            metallic = (_e176 * metallic_roughness.z);
            let _e179 = perceptual_roughness_2;
            perceptual_roughness_2 = (_e179 * metallic_roughness.y);
        }
        let _e183 = metallic;
        pbr_input_3.material.metallic = _e183;
        let _e186 = perceptual_roughness_2;
        pbr_input_3.material.perceptual_roughness = _e186;
        let _e191 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.clearcoat;
        pbr_input_3.material.clearcoat = _e191;
        if ((flags & STANDARD_MATERIAL_FLAGS_CLEARCOAT_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
            let _e200 = uv_2;
            let _e202 = bias.mip_bias;
            let _e203 = textureSampleBias(clearcoat_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, clearcoat_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, _e200, _e202);
            let _e205 = pbr_input_3.material.clearcoat;
            pbr_input_3.material.clearcoat = (_e205 * _e203.x);
        }
        let _e211 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.clearcoat_perceptual_roughness;
        pbr_input_3.material.clearcoat_perceptual_roughness = _e211;
        if ((flags & STANDARD_MATERIAL_FLAGS_CLEARCOAT_ROUGHNESS_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
            let _e220 = uv_2;
            let _e222 = bias.mip_bias;
            let _e223 = textureSampleBias(clearcoat_roughness_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, clearcoat_roughness_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, _e220, _e222);
            let _e225 = pbr_input_3.material.clearcoat_perceptual_roughness;
            pbr_input_3.material.clearcoat_perceptual_roughness = (_e225 * _e223.y);
        }
        let _e229 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.specular_transmission;
        specular_transmission = _e229;
        let _e233 = specular_transmission;
        pbr_input_3.material.specular_transmission = _e233;
        let _e236 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.thickness;
        thickness = _e236;
        let _e242 = meshX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7MJUW4ZDJNZTXGX[in_2.instance_index].world_from_local;
        let _e245 = pbr_input_3.N;
        let _e251 = thickness;
        thickness = (_e251 * length((transpose(_e242) * vec4<f32>(_e245, 0f)).xyz));
        let _e255 = thickness;
        pbr_input_3.material.thickness = _e255;
        let _e258 = materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX.diffuse_transmission;
        diffuse_transmission = _e258;
        let _e262 = diffuse_transmission;
        pbr_input_3.material.diffuse_transmission = _e262;
        if ((flags & STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
            let _e270 = uv_2;
            let _e272 = bias.mip_bias;
            let _e273 = textureSampleBias(occlusion_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, occlusion_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3CNFXGI2LOM5ZQX, _e270, _e272);
            let _e275 = diffuse_occlusion;
            diffuse_occlusion = (_e275 * _e273.x);
        }
        let _e278 = diffuse_occlusion;
        pbr_input_3.diffuse_occlusion = _e278;
        let _e281 = specular_occlusion;
        pbr_input_3.specular_occlusion = _e281;
        let _e284 = pbr_input_3.world_normal;
        pbr_input_3.N = normalize(_e284);
        let _e288 = pbr_input_3.N;
        pbr_input_3.clearcoat_N = _e288;
        let _e290 = pbr_input_3.world_normal;
        let _e292 = calculate_tbn_mikktspaceX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e290, in_2.world_tangent);
    }
    let _e293 = pbr_input_3;
    return _e293;
}

fn alpha_discardX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(material_1: StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX, output_color_1: vec4<f32>) -> vec4<f32> {
    var color_2: vec4<f32>;

    color_2 = output_color_1;
    let alpha_mode = (material_1.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX);
    if (alpha_mode == STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUEX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) {
        color_2.w = 1f;
    }
    let _e10 = color_2;
    return _e10;
}

fn calculate_diffuse_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(base_color: vec3<f32>, metallic_1: f32, specular_transmission_1: f32, diffuse_transmission_1: f32) -> vec3<f32> {
    return (((base_color * (1f - metallic_1)) * (1f - specular_transmission_1)) * (1f - diffuse_transmission_1));
}

fn calculate_F0X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(base_color_1: vec3<f32>, metallic_2: f32, reflectance: vec3<f32>) -> vec3<f32> {
    return ((((0.16f * reflectance) * reflectance) * (1f - metallic_2)) + (base_color_1 * metallic_2));
}

fn apply_pbr_lightingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(in_3: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) -> vec4<f32> {
    var output_color_2: vec4<f32>;
    var direct_light: vec3<f32> = vec3(0f);
    var transmitted_light: vec3<f32> = vec3(0f);
    var lighting_input: LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var clusterable_object_index_ranges_5: ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX;
    var i_2: u32;
    var shadow_1: f32;
    var i_3: u32;
    var shadow_2: f32;
    var i_4: u32 = 0u;
    var shadow_3: f32;
    var light_contrib: vec3<f32>;
    var indirect_light: vec3<f32> = vec3(0f);
    var found_diffuse_indirect_3: bool = false;
    var specular_transmitted_environment_light: vec3<f32> = vec3(0f);
    var emissive_light: vec3<f32>;

    output_color_2 = in_3.material.base_color;
    let emissive_1 = in_3.material.emissive;
    let metallic_3 = in_3.material.metallic;
    let perceptual_roughness_4 = in_3.material.perceptual_roughness;
    let _e14 = perceptualRoughnessToRoughnessX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptual_roughness_4);
    let ior = in_3.material.ior;
    let thickness_1 = in_3.material.thickness;
    let reflectance_1 = in_3.material.reflectance;
    let diffuse_transmission_2 = in_3.material.diffuse_transmission;
    let specular_transmission_2 = in_3.material.specular_transmission;
    let specular_transmissive_color = (specular_transmission_2 * in_3.material.base_color.xyz);
    let diffuse_occlusion_1 = in_3.diffuse_occlusion;
    let specular_occlusion_3 = in_3.specular_occlusion;
    let NdotV_9 = max(dot(in_3.N, in_3.V), 0.0001f);
    let R_3 = reflect(-(in_3.V), in_3.N);
    let clearcoat = in_3.material.clearcoat;
    let clearcoat_perceptual_roughness = in_3.material.clearcoat_perceptual_roughness;
    let _e44 = perceptualRoughnessToRoughnessX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(clearcoat_perceptual_roughness);
    let clearcoat_N_2 = in_3.clearcoat_N;
    let clearcoat_NdotV_1 = max(dot(clearcoat_N_2, in_3.V), 0.0001f);
    let clearcoat_R = reflect(-(in_3.V), clearcoat_N_2);
    let _e53 = output_color_2;
    let _e55 = calculate_diffuse_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e53.xyz, metallic_3, specular_transmission_2, diffuse_transmission_2);
    let _e56 = output_color_2;
    let diffuse_transmissive_color = (((_e56.xyz * (1f - metallic_3)) * (1f - specular_transmission_2)) * diffuse_transmission_2);
    let diffuse_transmissive_lobe_world_position = (in_3.world_position - (vec4<f32>(in_3.world_normal, 0f) * thickness_1));
    let _e71 = output_color_2;
    let _e73 = calculate_F0X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e71.xyz, metallic_3, reflectance_1);
    let _e74 = F_ABX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptual_roughness_4, NdotV_9);
    lighting_input.layers[0].NdotV = NdotV_9;
    lighting_input.layers[0].N = in_3.N;
    lighting_input.layers[0].R = R_3;
    lighting_input.layers[0].perceptual_roughness = perceptual_roughness_4;
    lighting_input.layers[0].roughness = _e14;
    lighting_input.P = in_3.world_position.xyz;
    lighting_input.V = in_3.V;
    lighting_input.diffuse_color = _e55;
    lighting_input.F0_ = _e73;
    lighting_input.F_ab = _e74;
    lighting_input.layers[1].NdotV = clearcoat_NdotV_1;
    lighting_input.layers[1].N = clearcoat_N_2;
    lighting_input.layers[1].R = clearcoat_R;
    lighting_input.layers[1].perceptual_roughness = clearcoat_perceptual_roughness;
    lighting_input.layers[1].roughness = _e44;
    lighting_input.clearcoat_strength = clearcoat;
    let _e120 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[0][2];
    let _e125 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[1][2];
    let _e130 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[2][2];
    let _e135 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[3][2];
    let view_z_5 = dot(vec4<f32>(_e120, _e125, _e130, _e135), in_3.world_position);
    let _e142 = fragment_cluster_indexX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(in_3.frag_coord.xy, view_z_5, in_3.is_orthographic);
    let _e143 = unpack_clusterable_object_index_rangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e142);
    clusterable_object_index_ranges_5 = _e143;
    let _e146 = clusterable_object_index_ranges_5.first_point_light_index_offset;
    i_2 = _e146;
    loop {
        let _e148 = i_2;
        let _e150 = clusterable_object_index_ranges_5.first_spot_light_index_offset;
        if (_e148 < _e150) {
        } else {
            break;
        }
        {
            let _e152 = i_2;
            let _e153 = get_clusterable_object_idX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e152);
            shadow_1 = 1f;
            let _e165 = clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[_e153].flags;
            if (((in_3.flags & MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX) != 0u) && ((_e165 & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u)) {
                let _e173 = fetch_point_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(_e153, in_3.world_position, in_3.world_normal);
                shadow_1 = _e173;
            }
            let _e175 = point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(_e153, (&lighting_input), true);
            let _e177 = shadow_1;
            let _e179 = direct_light;
            direct_light = (_e179 + (_e175 * _e177));
        }
        continuing {
            let _e181 = i_2;
            i_2 = (_e181 + 1u);
        }
    }
    let _e185 = clusterable_object_index_ranges_5.first_spot_light_index_offset;
    i_3 = _e185;
    loop {
        let _e187 = i_3;
        let _e189 = clusterable_object_index_ranges_5.first_reflection_probe_index_offset;
        if (_e187 < _e189) {
        } else {
            break;
        }
        {
            let _e191 = i_3;
            let _e192 = get_clusterable_object_idX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e191);
            shadow_2 = 1f;
            let _e204 = clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[_e192].flags;
            if (((in_3.flags & MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX) != 0u) && ((_e204 & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u)) {
                let _e216 = clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[_e192].shadow_map_near_z;
                let _e217 = fetch_spot_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(_e192, in_3.world_position, in_3.world_normal, _e216);
                shadow_2 = _e217;
            }
            let _e219 = spot_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(_e192, (&lighting_input), true);
            let _e220 = shadow_2;
            let _e222 = direct_light;
            direct_light = (_e222 + (_e219 * _e220));
        }
        continuing {
            let _e224 = i_3;
            i_3 = (_e224 + 1u);
        }
    }
    let n_directional_lights = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.n_directional_lights;
    loop {
        let _e231 = i_4;
        if (_e231 < n_directional_lights) {
        } else {
            break;
        }
        {
            let _e235 = i_4;
            let light_9 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[_e235]);
            let _e238 = (*light_9).skip;
            if (_e238 != 0u) {
                continue;
            }
            shadow_3 = 1f;
            let _e250 = i_4;
            let _e253 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[_e250].flags;
            if (((in_3.flags & MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX) != 0u) && ((_e253 & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u)) {
                let _e259 = i_4;
                let _e262 = fetch_directional_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(_e259, in_3.world_position, in_3.world_normal, view_z_5);
                shadow_3 = _e262;
            }
            let _e263 = i_4;
            let _e265 = directional_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(_e263, (&lighting_input), true);
            light_contrib = _e265;
            let _e267 = light_contrib;
            let _e268 = shadow_3;
            let _e270 = direct_light;
            direct_light = (_e270 + (_e267 * _e268));
        }
        continuing {
            let _e272 = i_4;
            i_4 = (_e272 + 1u);
        }
    }
    let _e276 = found_diffuse_indirect_3;
    let _e277 = environment_map_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX((&lighting_input), (&clusterable_object_index_ranges_5), _e276);
    if !(false) {
        let _e280 = found_diffuse_indirect_3;
        let _e281 = environment_map_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX((&lighting_input), (&clusterable_object_index_ranges_5), _e280);
        let _e288 = indirect_light;
        indirect_light = (_e288 + ((_e281.diffuse * diffuse_occlusion_1) + (_e281.specular * specular_occlusion_3)));
    }
    let _e293 = ambient_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MFWWE2LFNZ2AX(in_3.world_position, in_3.N, in_3.V, NdotV_9, _e55, _e73, perceptual_roughness_4, diffuse_occlusion_1);
    let _e294 = indirect_light;
    indirect_light = (_e294 + _e293);
    let _e298 = output_color_2.w;
    emissive_light = (emissive_1.xyz * _e298);
    let _e301 = emissive_light;
    emissive_light = (_e301 * (0.04f + (0.96f * pow((1f - clearcoat_NdotV_1), 5f))));
    let _e311 = emissive_light;
    let _e315 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.exposure;
    emissive_light = (_e311 * mix(1f, _e315, emissive_1.w));
    let _e322 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.exposure;
    let _e323 = transmitted_light;
    let _e324 = direct_light;
    let _e326 = indirect_light;
    let _e329 = emissive_light;
    let _e332 = output_color_2.w;
    output_color_2 = vec4<f32>(((_e322 * ((_e323 + _e324) + _e326)) + _e329), _e332);
    let _e334 = output_color_2;
    let _e336 = clusterable_object_index_ranges_5;
    let _e337 = cluster_debug_visualizationX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e334, view_z_5, in_3.is_orthographic, _e336, _e142);
    output_color_2 = _e337;
    let _e338 = output_color_2;
    return _e338;
}

fn main_pass_post_lighting_processingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(pbr_input_4: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX, input_color_1: vec4<f32>) -> vec4<f32> {
    var output_color_3: vec4<f32>;

    output_color_3 = input_color_1;
    let _e2 = output_color_3;
    return _e2;
}

fn apply_decal_base_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2MRSWGYLMHI5GG3DVON2GK4TFMQX(world_position_4: vec3<f32>, frag_coord_1: vec2<f32>, initial_base_color: vec4<f32>) -> vec4<f32> {
    var base_color_2: vec4<f32>;

    base_color_2 = initial_base_color;
    let _e2 = base_color_2;
    return _e2;
}

@fragment 
fn fragment(vertex_output: VertexOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX, @builtin(front_facing) is_front: bool) -> FragmentOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX {
    var in: VertexOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX;
    var pbr_input: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;
    var out: FragmentOutputX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXXE53BOJSF62LPX;

    in = vertex_output;
    let _e2 = in;
    let _e4 = pbr_input_from_standard_materialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOJQWO3LFNZ2AX(_e2, is_front);
    pbr_input = _e4;
    let _e9 = pbr_input.material;
    let _e12 = pbr_input.material.base_color;
    let _e13 = alpha_discardX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e9, _e12);
    pbr_input.material.base_color = _e13;
    let _e17 = in.world_position;
    let _e20 = in.position;
    let _e24 = pbr_input.material.base_color;
    let _e25 = apply_decal_base_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2MRSWGYLMHI5GG3DVON2GK4TFMQX(_e17.xyz, _e20.xy, _e24);
    pbr_input.material.base_color = _e25;
    let _e28 = pbr_input.material.flags;
    if ((_e28 & STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) == 0u) {
        let _e35 = pbr_input;
        let _e36 = apply_pbr_lightingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e35);
        out.color = _e36;
    } else {
        let _e40 = pbr_input.material.base_color;
        out.color = _e40;
    }
    let _e42 = pbr_input;
    let _e44 = out.color;
    let _e45 = main_pass_post_lighting_processingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e42, _e44);
    out.color = _e45;
    let _e46 = out;
    return _e46;
}
