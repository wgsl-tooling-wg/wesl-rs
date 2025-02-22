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

const STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUEX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX: u32 = 0u;

fn standard_material_newX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX() -> StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX {
    var material: StandardMaterialX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;

    material.base_color = vec4<f32>(1f, 1f, 1f, 1f);
    material.emissive = vec4<f32>(0f, 0f, 0f, 1f);
    // material.perceptual_roughness = 0.5f;
    // material.metallic = 0f;
    // material.reflectance = vec3(0.5f);
    // material.diffuse_transmission = 0f;
    // material.specular_transmission = 0f;
    // material.thickness = 0f;
    // material.ior = 1.5f;
    // material.attenuation_distance = 1f;
    // material.attenuation_color = vec4<f32>(1f, 1f, 1f, 1f);
    // material.clearcoat = 0f;
    // material.clearcoat_perceptual_roughness = 0f;
    // material.flags = STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUEX_naga_oil_mod_XMJSXM6K7OBRHEOR2OBRHEX3UPFYGK4YX;
    // material.alpha_cutoff = 0.5f;
    // material.parallax_depth_scale = 0.1f;
    // material.max_parallax_layer_count = 16f;
    // material.max_relief_mapping_search_steps = 5u;
    // material.deferred_lighting_pass_id = 1u;
    // material.uv_transform = mat3x3<f32>(vec3<f32>(1f, 0f, 0f), vec3<f32>(0f, 1f, 0f), vec3<f32>(0f, 0f, 1f));
    let _e66 = material;
    return _e66;
}

