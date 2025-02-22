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

struct LayerLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX {
    N: vec3<f32>,
    R: vec3<f32>,
    NdotV: f32,
    perceptual_roughness: f32,
    roughness: f32,
}

struct LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX {
    layers: array<LayerLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX, 1>,
    P: vec3<f32>,
    V: vec3<f32>,
    diffuse_color: vec3<f32>,
    F0_: vec3<f32>,
    F_ab: vec2<f32>,
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

struct EnvironmentMapLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

struct EnvironmentMapRadiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    irradiance: vec3<f32>,
    radiance: vec3<f32>,
}

struct PreviousViewUniformsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBZGK4DBONZV6YTJNZSGS3THOMX {
    view_from_world: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
}

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct PbrDeferredLightingDepthId {
    depth_id: u32,
}

const STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 32u;
const STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUEX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 0u;
const MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX: u32 = 536870912u;
const STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 256u;
const DEFERRED_MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX: u32 = 4u;
const DEFERRED_FLAGS_FOG_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX: u32 = 2u;
const DEFERRED_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX: u32 = 1u;
const U12MAXFX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX: f32 = 4095f;
const STANDARD_MATERIAL_FLAGS_TWO_COMPONENT_NORMAL_MAPX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 64u;
const STANDARD_MATERIAL_FLAGS_FLIP_NORMAL_MAP_YX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 128u;
const STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 3758096384u;
const PIX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX: f32 = 3.1415927f;
const POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 2u;
const LAYER_BASEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX: u32 = 0u;
const POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 1u;
const DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 1u;
const FOG_MODE_LINEARX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 1u;
const FOG_MODE_EXPONENTIALX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 2u;
const FOG_MODE_EXPONENTIAL_SQUAREDX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 3u;
const FOG_MODE_ATMOSPHERICX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX: u32 = 4u;
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
const RGB9E5_EXP_BIASX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX: i32 = 15i;
const RGB9E5_MANTISSA_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX: i32 = 9i;
const RGB9E5_MANTISSA_VALUESX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX: i32 = 512i;
const RGB9E5_EXPONENT_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX: u32 = 5u;
const RGB9E5_MANTISSA_BITSUX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX: u32 = 9u;

@group(0) @binding(31) 
var deferred_prepass_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: texture_2d<u32>;
@group(0) @binding(28) 
var depth_prepass_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: texture_depth_2d;
@group(0) @binding(30) 
var motion_vector_prepass_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: texture_2d<f32>;
@group(0) @binding(0) 
var<uniform> viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: ViewX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU5TJMV3QX;
@group(0) @binding(1) 
var<uniform> lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: LightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(8) 
var<storage> clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: ClusterableObjectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(12) 
var<uniform> fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
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
var diffuse_environment_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: texture_cube<f32>;
@group(0) @binding(18) 
var specular_environment_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: texture_cube<f32>;
@group(0) @binding(19) 
var environment_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: sampler;
@group(0) @binding(20) 
var<uniform> environment_map_uniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX: EnvironmentMapUniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX;
@group(0) @binding(26) 
var dt_lut_textureX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM5PWY5LUL5RGS3TENFXGO4YX: texture_3d<f32>;
@group(0) @binding(27) 
var dt_lut_samplerX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM5PWY5LUL5RGS3TENFXGO4YX: sampler;
@group(0) @binding(2) 
var<uniform> previous_view_uniformsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBZGK4DBONZV6YTJNZSGS3THOMX: PreviousViewUniformsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBZGK4DBONZV6YTJNZSGS3THOMX;
@group(1) @binding(0) 
var<uniform> depth_id: PbrDeferredLightingDepthId;

fn prepass_depthX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBZGK4DBONZV65LUNFWHGX(frag_coord_1: vec4<f32>, sample_index: u32) -> f32 {
    let _e5 = textureLoad(depth_prepass_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, vec2<i32>(frag_coord_1.xy), 0i);
    return _e5;
}

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

fn deferred_flags_from_mesh_material_flagsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(mesh_flags: u32, mat_flags: u32) -> u32 {
    var flags: u32 = 0u;

    let _e10 = flags;
    flags = (_e10 | (u32(((mesh_flags & MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX) != 0u)) * DEFERRED_MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX));
    let _e20 = flags;
    flags = (_e20 | (u32(((mat_flags & STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u)) * DEFERRED_FLAGS_FOG_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX));
    let _e29 = flags;
    flags = (_e29 | (u32(((mat_flags & STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u)) * DEFERRED_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX));
    let _e31 = flags;
    return _e31;
}

fn mesh_material_flags_from_deferred_flagsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(deferred_flags: u32) -> vec2<u32> {
    var mat_flags_1: u32 = 0u;
    var mesh_flags_1: u32 = 0u;

    let _e10 = mesh_flags_1;
    mesh_flags_1 = (_e10 | (u32(((deferred_flags & DEFERRED_MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX) != 0u)) * MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX));
    let _e20 = mat_flags_1;
    mat_flags_1 = (_e20 | (u32(((deferred_flags & DEFERRED_FLAGS_FOG_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX) != 0u)) * STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX));
    let _e29 = mat_flags_1;
    mat_flags_1 = (_e29 | (u32(((deferred_flags & DEFERRED_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX) != 0u)) * STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX));
    let _e31 = mesh_flags_1;
    let _e32 = mat_flags_1;
    return vec2<u32>(_e31, _e32);
}

fn pack_24bit_normal_and_flagsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(octahedral_normal: vec2<f32>, flags_1: u32) -> u32 {
    let unorm1_ = u32(((saturate(octahedral_normal.x) * U12MAXFX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX) + 0.5f));
    let unorm2_ = u32(((saturate(octahedral_normal.y) * U12MAXFX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX) + 0.5f));
    return (((unorm1_ & 4095u) | ((unorm2_ & 4095u) << 12u)) | ((flags_1 & 255u) << 24u));
}

fn unpack_24bit_normalX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(packed: u32) -> vec2<f32> {
    let unorm1_1 = (packed & 4095u);
    let unorm2_1 = ((packed >> 12u) & 4095u);
    return vec2<f32>((f32(unorm1_1) / U12MAXFX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX), (f32(unorm2_1) / U12MAXFX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX));
}

fn unpack_flagsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(packed_1: u32) -> u32 {
    return ((packed_1 >> 24u) & 255u);
}

fn unpack_unorm4x8_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(v: u32) -> vec4<f32> {
    return (vec4<f32>(f32((v & 255u)), f32(((v >> 8u) & 255u)), f32(((v >> 16u) & 255u)), f32(((v >> 24u) & 255u))) / vec4(255f));
}

fn pack_unorm4x8_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(values: vec4<f32>) -> u32 {
    let v_6 = vec4<u32>(((saturate(values) * 255f) + vec4(0.5f)));
    return ((((v_6.w << 24u) | (v_6.z << 16u)) | (v_6.y << 8u)) | v_6.x);
}

fn octahedral_encodeX_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX(v_1: vec3<f32>) -> vec2<f32> {
    var n: vec3<f32>;

    n = (v_1 / vec3(((abs(v_1.x) + abs(v_1.y)) + abs(v_1.z))));
    let _e12 = n;
    let _e22 = n;
    let octahedral_wrap = ((vec2(1f) - abs(_e12.yx)) * select(vec2(-1f), vec2(1f), (_e22.xy > vec2(0f))));
    let _e29 = n;
    let _e32 = n.z;
    let n_xy = select(octahedral_wrap, _e29.xy, (_e32 >= 0f));
    return ((n_xy * 0.5f) + vec2(0.5f));
}

fn octahedral_decode_signedX_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX(v_2: vec2<f32>) -> vec3<f32> {
    var n_1: vec3<f32>;

    n_1 = vec3<f32>(v_2.xy, ((1f - abs(v_2.x)) - abs(v_2.y)));
    let _e12 = n_1.z;
    let t = saturate(-(_e12));
    let _e18 = n_1;
    let w = select(vec2(t), vec2(-(t)), (_e18.xy >= vec2(0f)));
    let _e24 = n_1;
    let _e28 = n_1.z;
    n_1 = vec3<f32>((_e24.xy + w), _e28);
    let _e30 = n_1;
    return normalize(_e30);
}

fn octahedral_decodeX_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX(v_3: vec2<f32>) -> vec3<f32> {
    var n_2: vec3<f32>;

    let f = ((v_3 * 2f) - vec2(1f));
    let _e6 = octahedral_decode_signedX_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX(f);
    n_2 = _e6;
    let _e8 = n_2;
    return normalize(_e8);
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
    let v_7 = (0.5f / (lambdaV + lambdaL));
    return v_7;
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
    let LdotH_1 = (*derived_input).LdotH;
    let _e20 = D_GGXX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(roughness_3, NdotH_1, H_1);
    let _e21 = V_SmithGGXCorrelatedX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(roughness_3, NdotV_3, NdotL_1);
    let _e22 = fresnelX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(F0_2, LdotH_1);
    let _e24 = specular_multiscatterX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_3, _e20, _e21, _e22, specular_intensity_1);
    return _e24;
}

fn F_SchlickX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(f0_2: f32, f90_1: f32, VdotH_1: f32) -> f32 {
    return (f0_2 + ((f90_1 - f0_2) * pow((1f - VdotH_1), 5f)));
}

fn Fd_BurleyX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_4: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, derived_input_1: ptr<function, DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>) -> f32 {
    let roughness_4 = (*input_4).layers[0].roughness;
    let NdotV_4 = (*input_4).layers[0].NdotV;
    let NdotL_2 = (*derived_input_1).NdotL;
    let LdotH_2 = (*derived_input_1).LdotH;
    let f90_3 = (0.5f + (((2f * roughness_4) * LdotH_2) * LdotH_2));
    let _e21 = F_SchlickX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(1f, f90_3, NdotL_2);
    let _e23 = F_SchlickX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(1f, f90_3, NdotV_4);
    return ((_e21 * _e23) * 0.31830987f);
}

fn point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id: u32, input_5: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, enable_diffuse: bool) -> vec3<f32> {
    var specular_derived_input: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var derived_input_2: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var diffuse: vec3<f32> = vec3(0f);
    var color: vec3<f32>;

    let diffuse_color_1 = (*input_5).diffuse_color;
    let P = (*input_5).P;
    let N_2 = (*input_5).layers[0].N;
    let V_5 = (*input_5).V;
    let light = (&clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[light_id]);
    let _e18 = (*light).position_radius;
    let light_to_frag_1 = (_e18.xyz - P);
    let L_2 = normalize(light_to_frag_1);
    let distance_square = dot(light_to_frag_1, light_to_frag_1);
    let _e25 = (*light).color_inverse_square_range.w;
    let _e26 = getDistanceAttenuationX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(distance_square, _e25);
    let _e29 = (*light).position_radius.w;
    let _e31 = compute_specular_layer_values_for_point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_5, LAYER_BASEX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX, V_5, light_to_frag_1, _e29);
    let _e33 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(N_2, V_5, _e31.xyz);
    specular_derived_input = _e33;
    let specular_intensity_2 = _e31.w;
    let _e36 = specularX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_5, (&specular_derived_input), specular_intensity_2);
    let _e37 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(N_2, V_5, L_2);
    derived_input_2 = _e37;
    if enable_diffuse {
        let _e40 = Fd_BurleyX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_5, (&derived_input_2));
        diffuse = (diffuse_color_1 * _e40);
    }
    let _e43 = diffuse;
    color = (_e43 + _e36);
    let _e46 = color;
    let _e48 = (*light).color_inverse_square_range;
    let _e52 = derived_input_2.NdotL;
    return ((_e46 * _e48.xyz) * (_e26 * _e52));
}

fn spot_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id_1: u32, input_6: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, enable_diffuse_1: bool) -> vec3<f32> {
    var spot_dir: vec3<f32>;

    let _e3 = point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id_1, input_6, enable_diffuse_1);
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
    let _e47 = (*input_6).P;
    let light_to_frag_2 = (_e44.xyz - _e47.xyz);
    let _e50 = spot_dir;
    let cd = dot(-(_e50), normalize(light_to_frag_2));
    let _e56 = (*light_1).light_custom_data.z;
    let _e60 = (*light_1).light_custom_data.w;
    let attenuation_1 = saturate(((cd * _e56) + _e60));
    let spot_attenuation = (attenuation_1 * attenuation_1);
    return (_e3 * spot_attenuation);
}

fn directional_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(light_id_2: u32, input_7: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, enable_diffuse_2: bool) -> vec3<f32> {
    var derived_input_3: DerivedLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var diffuse_1: vec3<f32> = vec3(0f);
    var color_1: vec3<f32>;

    let diffuse_color_2 = (*input_7).diffuse_color;
    let NdotV_5 = (*input_7).layers[0].NdotV;
    let N_3 = (*input_7).layers[0].N;
    let V_6 = (*input_7).V;
    let roughness_5 = (*input_7).layers[0].roughness;
    let light_2 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[light_id_2]);
    let _e24 = (*light_2).direction_to_light;
    let L_3 = _e24.xyz;
    let _e26 = derive_lighting_inputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(N_3, V_6, L_3);
    derived_input_3 = _e26;
    if enable_diffuse_2 {
        let _e29 = Fd_BurleyX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_7, (&derived_input_3));
        diffuse_1 = (diffuse_color_2 * _e29);
    }
    let _e33 = specularX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(input_7, (&derived_input_3), 1f);
    let _e34 = diffuse_1;
    let _e37 = derived_input_3.NdotL;
    color_1 = ((_e34 + _e33) * _e37);
    let _e40 = color_1;
    let _e42 = (*light_2).color;
    return (_e40 * _e42.xyz);
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

fn fragment_cluster_indexX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(frag_coord_2: vec2<f32>, view_z_1: f32, is_orthographic_1: bool) -> u32 {
    let _e3 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.viewport;
    let _e8 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.cluster_factors;
    let xy = vec2<u32>(floor(((frag_coord_2 - _e3.xy) * _e8.xy)));
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
    var output_color_1: vec4<f32>;

    output_color_1 = input_color;
    let _e2 = output_color_1;
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
    let uv_2 = (light_local_1 * shadow_map_size);
    base_uv = floor((uv_2 + vec2(0.5f)));
    let _e18 = base_uv.x;
    let s = ((uv_2.x + 0.5f) - _e18);
    let _e24 = base_uv.y;
    let t_1 = ((uv_2.y + 0.5f) - _e24);
    let _e27 = base_uv;
    base_uv = (_e27 - vec2(0.5f));
    let _e30 = base_uv;
    base_uv = (_e30 * inv_shadow_map_size);
    let uw0_ = (4f - (3f * s));
    let uw2_ = (1f + (3f * s));
    let u0_ = (((3f - (2f * s)) / uw0_) - 2f);
    let u1_ = ((3f + s) / 7f);
    let u2_ = ((s / uw2_) + 2f);
    let vw0_ = (4f - (3f * t_1));
    let vw2_ = (1f + (3f * t_1));
    let v0_ = (((3f - (2f * t_1)) / vw0_) - 2f);
    let v1_ = ((3f + t_1) / 7f);
    let v2_ = ((t_1 / vw2_) + 2f);
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
    const n_3 = vec3<f32>(5f, 3f, 1f);
    let k_1 = ((n_3 + vec3((hsv.x / FRAC_PI_3X_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX))) % vec3(6f));
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

fn radiance_sample_directionX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(N_1: vec3<f32>, R: vec3<f32>, roughness_2: f32) -> vec3<f32> {
    let smoothness = saturate((1f - roughness_2));
    let lerp_factor = (smoothness * (sqrt(smoothness) + roughness_2));
    return mix(N_1, R, lerp_factor);
}

fn compute_radiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(input_8: LayerLightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX, clusterable_object_index_ranges_1: ptr<function, ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX>, world_position: vec3<f32>, found_diffuse_indirect: bool) -> EnvironmentMapRadiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    var radiances: EnvironmentMapRadiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX;
    var enable_diffuse_3: bool;
    var irradiance_sample_dir: vec3<f32>;
    var radiance_sample_dir: vec3<f32>;

    let N_4 = input_8.N;
    let R_2 = input_8.R;
    let NdotV_6 = input_8.NdotV;
    let perceptual_roughness_2 = input_8.perceptual_roughness;
    let roughness_6 = input_8.roughness;
    let _e8 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_cubemap_index;
    if (_e8 < 0i) {
        radiances.irradiance = vec3(0f);
        radiances.radiance = vec3(0f);
        let _e18 = radiances;
        return _e18;
    }
    let _e21 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.smallest_specular_mip_level_for_view;
    let radiance_level = (perceptual_roughness_2 * f32(_e21));
    let intensity_1 = light_probesX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.intensity_for_view;
    enable_diffuse_3 = !(found_diffuse_indirect);
    let _e30 = enable_diffuse_3;
    if _e30 {
        irradiance_sample_dir = N_4;
        let _e34 = environment_map_uniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.transform;
        let _e35 = irradiance_sample_dir;
        irradiance_sample_dir = (_e34 * vec4<f32>(_e35, 1f)).xyz;
        let _e42 = irradiance_sample_dir.z;
        irradiance_sample_dir.z = -(_e42);
        let _e45 = irradiance_sample_dir;
        let _e49 = textureSampleLevel(diffuse_environment_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, environment_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, _e45, 0f);
        radiances.irradiance = (_e49.xyz * intensity_1);
    }
    let _e52 = radiance_sample_directionX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(N_4, R_2, roughness_6);
    radiance_sample_dir = _e52;
    let _e56 = environment_map_uniformX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.transform;
    let _e57 = radiance_sample_dir;
    radiance_sample_dir = (_e56 * vec4<f32>(_e57, 1f)).xyz;
    let _e64 = radiance_sample_dir.z;
    radiance_sample_dir.z = -(_e64);
    let _e69 = radiance_sample_dir;
    let _e70 = textureSampleLevel(specular_environment_mapX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, environment_map_samplerX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, _e69, radiance_level);
    radiances.radiance = (_e70.xyz * intensity_1);
    let _e73 = radiances;
    return _e73;
}

fn environment_map_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(input_9: ptr<function, LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX>, clusterable_object_index_ranges_2: ptr<function, ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX>, found_diffuse_indirect_1: bool) -> EnvironmentMapLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX {
    var out: EnvironmentMapLightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX;

    let roughness_7 = (*input_9).layers[0].roughness;
    let diffuse_color_3 = (*input_9).diffuse_color;
    let NdotV_7 = (*input_9).layers[0].NdotV;
    let F_ab_2 = (*input_9).F_ab;
    let F0_3 = (*input_9).F0_;
    let world_position_3 = (*input_9).P;
    let _e19 = (*input_9).layers[0];
    let _e22 = compute_radiancesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX(_e19, clusterable_object_index_ranges_2, world_position_3, found_diffuse_indirect_1);
    if (all((_e22.irradiance == vec3(0f))) && all((_e22.radiance == vec3(0f)))) {
        out.diffuse = vec3(0f);
        out.specular = vec3(0f);
        let _e41 = out;
        return _e41;
    }
    let specular_occlusion = saturate(dot(F0_3, vec3(16.5f)));
    let Fr_1 = (max(vec3((1f - roughness_7)), F0_3) - F0_3);
    let kS = (F0_3 + (Fr_1 * pow((1f - NdotV_7), 5f)));
    let Ess = (F_ab_2.x + F_ab_2.y);
    let FssEss = ((kS * Ess) * specular_occlusion);
    let Ems = (1f - Ess);
    let Favg = (F0_3 + ((vec3(1f) - F0_3) / vec3(21f)));
    let Fms = ((FssEss * Favg) / (vec3(1f) - (Ems * Favg)));
    let FmsEms = (Fms * Ems);
    let Edss = (vec3(1f) - (FssEss + FmsEms));
    let kD = (diffuse_color_3 * Edss);
    if !(found_diffuse_indirect_1) {
        out.diffuse = ((FmsEms + kD) * _e22.irradiance);
    } else {
        out.diffuse = vec3(0f);
    }
    out.specular = (FssEss * _e22.radiance);
    let _e94 = out;
    return _e94;
}

fn EnvBRDFApproxX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(F0_: vec3<f32>, F_ab: vec2<f32>) -> vec3<f32> {
    return ((F0_ * F_ab.x) + vec3(F_ab.y));
}

fn ambient_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MFWWE2LFNZ2AX(world_position_1: vec4<f32>, world_normal: vec3<f32>, V_3: vec3<f32>, NdotV_2: f32, diffuse_color: vec3<f32>, specular_color: vec3<f32>, perceptual_roughness_1: f32, occlusion: vec3<f32>) -> vec3<f32> {
    let _e2 = F_ABX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(1f, NdotV_2);
    let _e4 = EnvBRDFApproxX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(diffuse_color, _e2);
    let _e6 = F_ABX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptual_roughness_1, NdotV_2);
    let _e8 = EnvBRDFApproxX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(specular_color, _e6);
    let specular_occlusion_1 = saturate(dot(specular_color, vec3(16.5f)));
    let _e18 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.ambient_color;
    return (((_e4 + (_e8 * specular_occlusion_1)) * _e18.xyz) * occlusion);
}

fn scattering_adjusted_fog_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params: FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, scattering: vec3<f32>) -> vec4<f32> {
    if (fog_params.directional_light_color.w > 0f) {
        return vec4<f32>((fog_params.base_color.xyz + ((scattering * fog_params.directional_light_color.xyz) * fog_params.directional_light_color.w)), fog_params.base_color.w);
    } else {
        return fog_params.base_color;
    }
}

fn linear_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_1: FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, input_color_1: vec4<f32>, distance: f32, scattering_1: vec3<f32>) -> vec4<f32> {
    var fog_color: vec4<f32>;

    let _e2 = scattering_adjusted_fog_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_1, scattering_1);
    fog_color = _e2;
    let start = fog_params_1.be.x;
    let end = fog_params_1.be.y;
    let _e18 = fog_color.w;
    fog_color.w = (_e18 * (1f - clamp(((end - distance) / (end - start)), 0f, 1f)));
    let _e22 = fog_color;
    let _e25 = fog_color.w;
    return vec4<f32>(mix(input_color_1.xyz, _e22.xyz, _e25), input_color_1.w);
}

fn exponential_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_2: FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, input_color_2: vec4<f32>, distance_1: f32, scattering_2: vec3<f32>) -> vec4<f32> {
    var fog_color_1: vec4<f32>;

    let _e2 = scattering_adjusted_fog_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_2, scattering_2);
    fog_color_1 = _e2;
    let density = fog_params_2.be.x;
    let _e14 = fog_color_1.w;
    fog_color_1.w = (_e14 * (1f - (1f / exp((distance_1 * density)))));
    let _e18 = fog_color_1;
    let _e21 = fog_color_1.w;
    return vec4<f32>(mix(input_color_2.xyz, _e18.xyz, _e21), input_color_2.w);
}

fn exponential_squared_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_3: FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, input_color_3: vec4<f32>, distance_2: f32, scattering_3: vec3<f32>) -> vec4<f32> {
    var fog_color_2: vec4<f32>;

    let _e2 = scattering_adjusted_fog_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_3, scattering_3);
    fog_color_2 = _e2;
    let distance_times_density = (distance_2 * fog_params_3.be.x);
    let _e15 = fog_color_2.w;
    fog_color_2.w = (_e15 * (1f - (1f / exp((distance_times_density * distance_times_density)))));
    let _e19 = fog_color_2;
    let _e22 = fog_color_2.w;
    return vec4<f32>(mix(input_color_3.xyz, _e19.xyz, _e22), input_color_3.w);
}

fn atmospheric_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_4: FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, input_color_4: vec4<f32>, distance_3: f32, scattering_4: vec3<f32>) -> vec4<f32> {
    var fog_color_3: vec4<f32>;

    let _e2 = scattering_adjusted_fog_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_4, scattering_4);
    fog_color_3 = _e2;
    let extinction_factor = (vec3(1f) - (vec3(1f) / exp((distance_3 * fog_params_4.be))));
    let inscattering_factor = (vec3(1f) - (vec3(1f) / exp((distance_3 * fog_params_4.bi))));
    let _e26 = fog_color_3.w;
    let _e32 = fog_color_3;
    let _e36 = fog_color_3.w;
    return vec4<f32>(((input_color_4.xyz * (vec3(1f) - (extinction_factor * _e26))) + ((_e32.xyz * inscattering_factor) * _e36)), input_color_4.w);
}

fn powsafeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(color_2: vec3<f32>, power: f32) -> vec3<f32> {
    return (pow(abs(color_2), vec3(power)) * sign(color_2));
}

fn screen_space_ditherX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(frag_coord_3: vec2<f32>) -> vec3<f32> {
    var dither: vec3<f32>;

    dither = vec3(dot(vec2<f32>(171f, 231f), frag_coord_3)).xxx;
    let _e8 = dither;
    dither = fract((_e8.xyz / vec3<f32>(103f, 71f, 97f)));
    let _e16 = dither;
    return ((_e16 - vec3(0.5f)) / vec3(255f));
}

fn sample_current_lutX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(p: vec3<f32>) -> vec3<f32> {
    let _e4 = textureSampleLevel(dt_lut_textureX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM5PWY5LUL5RGS3TENFXGO4YX, dt_lut_samplerX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM5PWY5LUL5RGS3TENFXGO4YX, p, 0f);
    return _e4.xyz;
}

fn sample_tony_mc_mapface_lutX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(stimulus: vec3<f32>) -> vec3<f32> {
    var uv: vec3<f32>;

    uv = (((stimulus / (stimulus + vec3(1f))) * 0.9791667f) + vec3(0.010416667f));
    let _e11 = uv;
    let _e13 = sample_current_lutX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(saturate(_e11));
    return _e13.xyz;
}

fn tonemapping_luminanceX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(v_4: vec3<f32>) -> f32 {
    return dot(v_4, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}

fn saturationX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(color_3: vec3<f32>, saturationAmount: f32) -> vec3<f32> {
    let _e1 = tonemapping_luminanceX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(color_3);
    return mix(vec3(_e1), color_3, vec3(saturationAmount));
}

fn tone_mappingX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(in_1: vec4<f32>, in_color_grading: ColorGradingX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU5TJMV3QX) -> vec4<f32> {
    var color_4: vec3<f32>;
    var color_grading: ColorGradingX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU5TJMV3QX;

    color_4 = max(in_1.xyz, vec3(0f));
    color_grading = in_color_grading;
    let _e8 = color_4;
    let _e12 = color_grading.exposure;
    let _e13 = powsafeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(vec3(2f), _e12);
    color_4 = (_e8 * _e13);
    let _e15 = color_4;
    let _e16 = sample_tony_mc_mapface_lutX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(_e15);
    color_4 = _e16;
    let _e17 = color_4;
    let _e19 = color_grading.post_saturation;
    let _e20 = saturationX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(_e17, _e19);
    color_4 = _e20;
    let _e21 = color_4;
    return vec4<f32>(_e21, in_1.w);
}

fn calculate_viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(world_position_2: vec4<f32>, is_orthographic_3: bool) -> vec3<f32> {
    var V_4: vec3<f32>;

    if is_orthographic_3 {
        let _e5 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_world[0][2];
        let _e10 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_world[1][2];
        let _e15 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_world[2][2];
        V_4 = normalize(vec3<f32>(_e5, _e10, _e15));
    } else {
        let _e22 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.world_position;
        V_4 = normalize((_e22.xyz - world_position_2.xyz));
    }
    let _e27 = V_4;
    return _e27;
}

fn calculate_diffuse_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(base_color: vec3<f32>, metallic: f32, specular_transmission: f32, diffuse_transmission: f32) -> vec3<f32> {
    return (((base_color * (1f - metallic)) * (1f - specular_transmission)) * (1f - diffuse_transmission));
}

fn floor_log2_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(x: f32) -> i32 {
    let f_1 = bitcast<u32>(x);
    let biasedexponent = ((f_1 & 2139095040u) >> 23u);
    return (i32(biasedexponent) - 127i);
}

fn vec3_to_rgb9e5_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(rgb_in: vec3<f32>) -> u32 {
    var exp_shared: i32;
    var denom: f32;

    let rgb = clamp(rgb_in, vec3(0f), vec3(65408f));
    let maxrgb = max(rgb.x, max(rgb.y, rgb.z));
    let _e11 = floor_log2_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(maxrgb);
    exp_shared = ((max(-16i, _e11) + 1i) + RGB9E5_EXP_BIASX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX);
    let _e19 = exp_shared;
    denom = exp2(f32(((_e19 - RGB9E5_EXP_BIASX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX) - RGB9E5_MANTISSA_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX)));
    let _e27 = denom;
    let maxm = i32(floor(((maxrgb / _e27) + 0.5f)));
    if (maxm == RGB9E5_MANTISSA_VALUESX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX) {
        let _e36 = denom;
        denom = (_e36 * 2f);
        let _e39 = exp_shared;
        exp_shared = (_e39 + 1i);
    }
    let _e41 = denom;
    let n_4 = vec3<u32>(floor(((rgb / vec3(_e41)) + vec3(0.5f))));
    let _e49 = exp_shared;
    return ((((u32(_e49) << 27u) | (n_4.z << 18u)) | (n_4.y << 9u)) | (n_4.x << 0u));
}

fn extract_bitsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(value_1: u32, offset: u32, bits: u32) -> u32 {
    let mask = ((1u << bits) - 1u);
    return ((value_1 >> offset) & mask);
}

fn rgb9e5_to_vec3_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(v_5: u32) -> vec3<f32> {
    let _e3 = extract_bitsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(v_5, 27u, RGB9E5_EXPONENT_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX);
    let exponent = ((i32(_e3) - RGB9E5_EXP_BIASX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX) - RGB9E5_MANTISSA_BITSX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX);
    let scale_4 = exp2(f32(exponent));
    let _e13 = extract_bitsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(v_5, 0u, RGB9E5_MANTISSA_BITSUX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX);
    let _e17 = extract_bitsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(v_5, 9u, RGB9E5_MANTISSA_BITSUX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX);
    let _e21 = extract_bitsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(v_5, 18u, RGB9E5_MANTISSA_BITSUX_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX);
    return (vec3<f32>(f32(_e13), f32(_e17), f32(_e21)) * scale_4);
}

fn position_ndc_to_worldX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(ndc_pos: vec3<f32>) -> vec3<f32> {
    let _e2 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.world_from_clip;
    let world_pos = (_e2 * vec4<f32>(ndc_pos, 1f));
    return (world_pos.xyz / vec3(world_pos.w));
}

fn frag_coord_to_uvX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(frag_coord_4: vec2<f32>) -> vec2<f32> {
    let _e3 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.viewport;
    let _e8 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.viewport;
    return ((frag_coord_4 - _e3.xy) / _e8.zw);
}

fn uv_to_ndcX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(uv_1: vec2<f32>) -> vec2<f32> {
    return ((uv_1 * vec2<f32>(2f, -2f)) + vec2<f32>(-1f, 1f));
}

fn frag_coord_to_ndcX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(frag_coord_5: vec4<f32>) -> vec3<f32> {
    let _e2 = frag_coord_to_uvX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(frag_coord_5.xy);
    let _e3 = uv_to_ndcX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(_e2);
    return vec3<f32>(_e3, frag_coord_5.z);
}

fn pbr_input_from_deferred_gbufferX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF6ZTVNZRXI2LPNZZQX(frag_coord_6: vec4<f32>, gbuffer: vec4<u32>) -> PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    var pbr: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;

    let _e0 = pbr_input_newX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX();
    pbr = _e0;
    let _e4 = unpack_flagsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(gbuffer.w);
    let _e5 = mesh_material_flags_from_deferred_flagsX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(_e4);
    pbr.flags = _e5.x;
    pbr.material.flags = _e5.y;
    let _e12 = unpack_unorm4x8_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(gbuffer.x);
    pbr.material.perceptual_roughness = _e12.w;
    let _e17 = rgb9e5_to_vec3_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OJTWEOLFGUX(gbuffer.y);
    let _e20 = pbr.material.flags;
    if ((_e20 & STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
        pbr.material.base_color = vec4<f32>(_e17, 1f);
        pbr.material.emissive = vec4<f32>(vec3(0f), 0f);
    } else {
        pbr.material.base_color = vec4<f32>(pow(_e12.xyz, vec3(2.2f)), 1f);
        pbr.material.emissive = vec4<f32>(_e17, 0f);
    }
    let _e48 = unpack_unorm4x8_X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(gbuffer.z);
    pbr.material.reflectance = vec3(_e48.x);
    pbr.material.metallic = _e48.y;
    pbr.diffuse_occlusion = vec3(_e48.z);
    let _e60 = unpack_24bit_normalX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF65DZOBSXGX(gbuffer.w);
    let _e61 = octahedral_decodeX_naga_oil_mod_XMJSXM6K7OBRHEOR2OV2GS3DTX(_e60);
    let _e63 = frag_coord_to_ndcX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(frag_coord_6);
    let _e64 = position_ndc_to_worldX_naga_oil_mod_XMJSXM6K7OBRHEOR2OZUWK527ORZGC3TTMZXXE3LBORUW63TTX(_e63);
    let world_position_4 = vec4<f32>(_e64, 1f);
    let _e71 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.clip_from_view[3][3];
    let is_orthographic_4 = (_e71 == 1f);
    let _e74 = calculate_viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(world_position_4, is_orthographic_4);
    pbr.frag_coord = frag_coord_6;
    pbr.world_normal = _e61;
    pbr.world_position = world_position_4;
    pbr.N = _e61;
    pbr.V = _e74;
    pbr.is_orthographic = is_orthographic_4;
    let _e81 = pbr;
    return _e81;
}

fn calculate_F0X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(base_color_1: vec3<f32>, metallic_1: f32, reflectance: vec3<f32>) -> vec3<f32> {
    return ((((0.16f * reflectance) * reflectance) * (1f - metallic_1)) + (base_color_1 * metallic_1));
}

fn apply_pbr_lightingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(in_2: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) -> vec4<f32> {
    var output_color_2: vec4<f32>;
    var direct_light: vec3<f32> = vec3(0f);
    var transmitted_light: vec3<f32> = vec3(0f);
    var lighting_input: LightingInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX;
    var clusterable_object_index_ranges_3: ClusterableObjectIndexRangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX;
    var i_1: u32;
    var shadow_1: f32;
    var i_2: u32;
    var shadow_2: f32;
    var i_3: u32 = 0u;
    var shadow_3: f32;
    var light_contrib: vec3<f32>;
    var indirect_light: vec3<f32> = vec3(0f);
    var found_diffuse_indirect_2: bool = false;
    var specular_transmitted_environment_light: vec3<f32> = vec3(0f);
    var emissive_light: vec3<f32>;

    output_color_2 = in_2.material.base_color;
    let emissive = in_2.material.emissive;
    let metallic_2 = in_2.material.metallic;
    let perceptual_roughness_3 = in_2.material.perceptual_roughness;
    let _e14 = perceptualRoughnessToRoughnessX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptual_roughness_3);
    let ior = in_2.material.ior;
    let thickness = in_2.material.thickness;
    let reflectance_1 = in_2.material.reflectance;
    let diffuse_transmission_1 = in_2.material.diffuse_transmission;
    let specular_transmission_1 = in_2.material.specular_transmission;
    let specular_transmissive_color = (specular_transmission_1 * in_2.material.base_color.xyz);
    let diffuse_occlusion = in_2.diffuse_occlusion;
    let specular_occlusion_2 = in_2.specular_occlusion;
    let NdotV_8 = max(dot(in_2.N, in_2.V), 0.0001f);
    let R_3 = reflect(-(in_2.V), in_2.N);
    let _e40 = output_color_2;
    let _e42 = calculate_diffuse_colorX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e40.xyz, metallic_2, specular_transmission_1, diffuse_transmission_1);
    let _e43 = output_color_2;
    let diffuse_transmissive_color = (((_e43.xyz * (1f - metallic_2)) * (1f - specular_transmission_1)) * diffuse_transmission_1);
    let diffuse_transmissive_lobe_world_position = (in_2.world_position - (vec4<f32>(in_2.world_normal, 0f) * thickness));
    let _e58 = output_color_2;
    let _e60 = calculate_F0X_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e58.xyz, metallic_2, reflectance_1);
    let _e61 = F_ABX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(perceptual_roughness_3, NdotV_8);
    lighting_input.layers[0].NdotV = NdotV_8;
    lighting_input.layers[0].N = in_2.N;
    lighting_input.layers[0].R = R_3;
    lighting_input.layers[0].perceptual_roughness = perceptual_roughness_3;
    lighting_input.layers[0].roughness = _e14;
    lighting_input.P = in_2.world_position.xyz;
    lighting_input.V = in_2.V;
    lighting_input.diffuse_color = _e42;
    lighting_input.F0_ = _e60;
    lighting_input.F_ab = _e61;
    let _e91 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[0][2];
    let _e96 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[1][2];
    let _e101 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[2][2];
    let _e106 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.view_from_world[3][2];
    let view_z_5 = dot(vec4<f32>(_e91, _e96, _e101, _e106), in_2.world_position);
    let _e113 = fragment_cluster_indexX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(in_2.frag_coord.xy, view_z_5, in_2.is_orthographic);
    let _e114 = unpack_clusterable_object_index_rangesX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e113);
    clusterable_object_index_ranges_3 = _e114;
    let _e117 = clusterable_object_index_ranges_3.first_point_light_index_offset;
    i_1 = _e117;
    loop {
        let _e119 = i_1;
        let _e121 = clusterable_object_index_ranges_3.first_spot_light_index_offset;
        if (_e119 < _e121) {
        } else {
            break;
        }
        {
            let _e123 = i_1;
            let _e124 = get_clusterable_object_idX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e123);
            shadow_1 = 1f;
            let _e136 = clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[_e124].flags;
            if (((in_2.flags & MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX) != 0u) && ((_e136 & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u)) {
                let _e144 = fetch_point_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(_e124, in_2.world_position, in_2.world_normal);
                shadow_1 = _e144;
            }
            let _e146 = point_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(_e124, (&lighting_input), true);
            let _e148 = shadow_1;
            let _e150 = direct_light;
            direct_light = (_e150 + (_e146 * _e148));
        }
        continuing {
            let _e152 = i_1;
            i_1 = (_e152 + 1u);
        }
    }
    let _e156 = clusterable_object_index_ranges_3.first_spot_light_index_offset;
    i_2 = _e156;
    loop {
        let _e158 = i_2;
        let _e160 = clusterable_object_index_ranges_3.first_reflection_probe_index_offset;
        if (_e158 < _e160) {
        } else {
            break;
        }
        {
            let _e162 = i_2;
            let _e163 = get_clusterable_object_idX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e162);
            shadow_2 = 1f;
            let _e175 = clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[_e163].flags;
            if (((in_2.flags & MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX) != 0u) && ((_e175 & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u)) {
                let _e187 = clusterable_objectsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.data[_e163].shadow_map_near_z;
                let _e188 = fetch_spot_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(_e163, in_2.world_position, in_2.world_normal, _e187);
                shadow_2 = _e188;
            }
            let _e190 = spot_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(_e163, (&lighting_input), true);
            let _e191 = shadow_2;
            let _e193 = direct_light;
            direct_light = (_e193 + (_e190 * _e191));
        }
        continuing {
            let _e195 = i_2;
            i_2 = (_e195 + 1u);
        }
    }
    let n_directional_lights = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.n_directional_lights;
    loop {
        let _e202 = i_3;
        if (_e202 < n_directional_lights) {
        } else {
            break;
        }
        {
            let _e206 = i_3;
            let light_9 = (&lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[_e206]);
            let _e209 = (*light_9).skip;
            if (_e209 != 0u) {
                continue;
            }
            shadow_3 = 1f;
            let _e221 = i_3;
            let _e224 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[_e221].flags;
            if (((in_2.flags & MESH_FLAGS_SHADOW_RECEIVER_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OR4XAZLTX) != 0u) && ((_e224 & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) != 0u)) {
                let _e230 = i_3;
                let _e233 = fetch_directional_shadowX_naga_oil_mod_XMJSXM6K7OBRHEOR2ONUGCZDPO5ZQX(_e230, in_2.world_position, in_2.world_normal, view_z_5);
                shadow_3 = _e233;
            }
            let _e234 = i_3;
            let _e236 = directional_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2NRUWO2DUNFXGOX(_e234, (&lighting_input), true);
            light_contrib = _e236;
            let _e238 = light_contrib;
            let _e239 = shadow_3;
            let _e241 = direct_light;
            direct_light = (_e241 + (_e238 * _e239));
        }
        continuing {
            let _e243 = i_3;
            i_3 = (_e243 + 1u);
        }
    }
    let _e247 = found_diffuse_indirect_2;
    let _e248 = environment_map_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX((&lighting_input), (&clusterable_object_index_ranges_3), _e247);
    if !(false) {
        let _e251 = found_diffuse_indirect_2;
        let _e252 = environment_map_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MVXHM2LSN5XG2ZLOORPW2YLQX((&lighting_input), (&clusterable_object_index_ranges_3), _e251);
        let _e259 = indirect_light;
        indirect_light = (_e259 + ((_e252.diffuse * diffuse_occlusion) + (_e252.specular * specular_occlusion_2)));
    }
    let _e264 = ambient_lightX_naga_oil_mod_XMJSXM6K7OBRHEOR2MFWWE2LFNZ2AX(in_2.world_position, in_2.N, in_2.V, NdotV_8, _e42, _e60, perceptual_roughness_3, diffuse_occlusion);
    let _e265 = indirect_light;
    indirect_light = (_e265 + _e264);
    let _e269 = output_color_2.w;
    emissive_light = (emissive.xyz * _e269);
    let _e272 = emissive_light;
    let _e276 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.exposure;
    emissive_light = (_e272 * mix(1f, _e276, emissive.w));
    let _e283 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.exposure;
    let _e284 = transmitted_light;
    let _e285 = direct_light;
    let _e287 = indirect_light;
    let _e290 = emissive_light;
    let _e293 = output_color_2.w;
    output_color_2 = vec4<f32>(((_e283 * ((_e284 + _e285) + _e287)) + _e290), _e293);
    let _e295 = output_color_2;
    let _e297 = clusterable_object_index_ranges_3;
    let _e298 = cluster_debug_visualizationX_naga_oil_mod_XMJSXM6K7OBRHEOR2MNWHK43UMVZGKZC7MZXXE53BOJSAX(_e295, view_z_5, in_2.is_orthographic, _e297, _e113);
    output_color_2 = _e298;
    let _e299 = output_color_2;
    return _e299;
}

fn apply_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(fog_params_5: FogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX, input_color_5: vec4<f32>, fragment_world_position: vec3<f32>, view_world_position: vec3<f32>) -> vec4<f32> {
    var scattering_5: vec3<f32> = vec3(0f);
    var i_4: u32 = 0u;

    let view_to_world = (fragment_world_position.xyz - view_world_position.xyz);
    let distance_4 = length(view_to_world);
    if (fog_params_5.directional_light_color.w > 0f) {
        let view_to_world_normalized = (view_to_world / vec3(distance_4));
        let n_directional_lights_1 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.n_directional_lights;
        loop {
            let _e20 = i_4;
            if (_e20 < n_directional_lights_1) {
            } else {
                break;
            }
            {
                let _e24 = i_4;
                let light_10 = lightsX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.directional_lights[_e24];
                let _e39 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.exposure;
                let _e41 = scattering_5;
                scattering_5 = (_e41 + ((pow(max(dot(view_to_world_normalized, light_10.direction_to_light), 0f), fog_params_5.directional_light_exponent) * light_10.color.xyz) * _e39));
            }
            continuing {
                let _e43 = i_4;
                i_4 = (_e43 + 1u);
            }
        }
    }
    if (fog_params_5.mode == FOG_MODE_LINEARX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) {
        let _e49 = scattering_5;
        let _e51 = linear_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_5, input_color_5, distance_4, _e49);
        return _e51;
    } else {
        if (fog_params_5.mode == FOG_MODE_EXPONENTIALX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) {
            let _e55 = scattering_5;
            let _e56 = exponential_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_5, input_color_5, distance_4, _e55);
            return _e56;
        } else {
            if (fog_params_5.mode == FOG_MODE_EXPONENTIAL_SQUAREDX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) {
                let _e60 = scattering_5;
                let _e61 = exponential_squared_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_5, input_color_5, distance_4, _e60);
                return _e61;
            } else {
                if (fog_params_5.mode == FOG_MODE_ATMOSPHERICX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527OR4XAZLTX) {
                    let _e65 = scattering_5;
                    let _e66 = atmospheric_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2MZXWOX(fog_params_5, input_color_5, distance_4, _e65);
                    return _e66;
                } else {
                    return input_color_5;
                }
            }
        }
    }
}

fn main_pass_post_lighting_processingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(pbr_input_2: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX, input_color_6: vec4<f32>) -> vec4<f32> {
    var output_color_3: vec4<f32>;
    var output_rgb: vec3<f32>;

    output_color_3 = input_color_6;
    if ((pbr_input_2.material.flags & STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) != 0u) {
        let _e10 = fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX;
        let _e11 = output_color_3;
        let _e16 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.world_position;
        let _e18 = apply_fogX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e10, _e11, pbr_input_2.world_position.xyz, _e16.xyz);
        output_color_3 = _e18;
    }
    let _e19 = output_color_3;
    let _e22 = viewX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX.color_grading;
    let _e23 = tone_mappingX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(_e19, _e22);
    output_color_3 = _e23;
    let _e24 = output_color_3;
    output_rgb = _e24.xyz;
    let _e27 = output_rgb;
    let _e29 = powsafeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(_e27, 0.45454547f);
    output_rgb = _e29;
    let _e32 = screen_space_ditherX_naga_oil_mod_XMJSXM6K7MNXXEZK7OBUXAZLMNFXGKOR2ORXW4ZLNMFYHA2LOM4X(pbr_input_2.frag_coord.xy);
    let _e33 = output_rgb;
    output_rgb = (_e33 + _e32);
    let _e35 = output_rgb;
    let _e37 = powsafeX_naga_oil_mod_XMJSXM6K7OJSW4ZDFOI5DU3LBORUHGX(_e35, 2.2f);
    output_rgb = _e37;
    let _e38 = output_rgb;
    let _e40 = output_color_3.w;
    output_color_3 = vec4<f32>(_e38, _e40);
    let _e42 = output_color_3;
    return _e42;
}

@vertex 
fn vertex(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    let uv_3 = (vec2<f32>(f32((vertex_index >> 1u)), f32((vertex_index & 1u))) * 2f);
    let _e20 = depth_id.depth_id;
    let clip_position = vec4<f32>(((uv_3 * vec2<f32>(2f, -2f)) + vec2<f32>(-1f, 1f)), (f32(_e20) / 255f), 1f);
    return FullscreenVertexOutput(clip_position, uv_3);
}

@fragment 
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var frag_coord: vec4<f32>;
    var pbr_input: PbrInputX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;
    var output_color: vec4<f32> = vec4(0f);

    frag_coord = vec4<f32>(in.position.xy, 0f, 0f);
    let _e9 = frag_coord;
    let deferred_data = textureLoad(deferred_prepass_textureX_naga_oil_mod_XMJSXM6K7OBRHEOR2NVSXG2C7OZUWK527MJUW4ZDJNZTXGX, vec2<i32>(_e9.xy), 0i);
    let _e18 = prepass_depthX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBZGK4DBONZV65LUNFWHGX(in.position, 0u);
    frag_coord.z = _e18;
    let _e19 = frag_coord;
    let _e20 = pbr_input_from_deferred_gbufferX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3EMVTGK4TSMVSF6ZTVNZRXI2LPNZZQX(_e19, deferred_data);
    pbr_input = _e20;
    let _e24 = pbr_input.material.flags;
    if ((_e24 & STANDARD_MATERIAL_FLAGS_UNLIT_BITX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX) == 0u) {
        let _e29 = pbr_input;
        let _e30 = apply_pbr_lightingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e29);
        output_color = _e30;
    } else {
        let _e34 = pbr_input.material.base_color;
        output_color = _e34;
    }
    let _e35 = pbr_input;
    let _e36 = output_color;
    let _e37 = main_pass_post_lighting_processingX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3GOVXGG5DJN5XHGX(_e35, _e36);
    output_color = _e37;
    let _e38 = output_color;
    return _e38;
}
