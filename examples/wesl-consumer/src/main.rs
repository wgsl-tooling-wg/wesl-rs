fn main() {
    // run wesl at build-time
    #[cfg(feature = "build-time")]
    let source = {
        use wesl::include_wesl;
        include_wesl!("main")
    };

    // run wesl at run-time
    #[cfg(not(feature = "build-time"))]
    let source = wesl::Wesl::new("src/shaders")
        .add_package(&random_wgsl::random::Mod)
        .compile("main")
        .inspect_err(|e| {
            eprintln!("{e}");
            panic!();
        })
        .unwrap()
        .to_string();

    println!("{source}");

    use wesl::syntax::*;

    // using the procedural macro
    let source = wesl::quote_wesl! {
        struct StandardMaterial{
            base_color: vec4f,
            emissive: vec4f,
            attenuation_color: vec4f,
            uv_transform: mat3x3f,
            reflectance: vec3f,
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
            anisotropy_rotation: vec2f,
            flags: u32,
            alpha_cutoff: f32,
            parallax_depth_scale: f32,
            max_parallax_layer_count: f32,
            lightmap_exposure: f32,
            max_relief_mapping_search_steps: u32,
            deferred_lighting_pass_id: u32,
        }

        const STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE: u32 = 0u;

        fn standard_material_new() -> StandardMaterial{
            var material: StandardMaterial;

            material.base_color = vec4f(1f, 1f, 1f, 1f);
            material.emissive = vec4f(0f, 0f, 0f, 1f);
            material.perceptual_roughness = 0.5f;
            material.metallic = 0f;
            material.reflectance = vec3(0.5f);
            material.diffuse_transmission = 0f;
            material.specular_transmission = 0f;
            material.thickness = 0f;
            material.ior = 1.5f;
            material.attenuation_distance = 1f;
            material.attenuation_color = vec4f(1f, 1f, 1f, 1f);
            material.clearcoat = 0f;
            material.clearcoat_perceptual_roughness = 0f;
            material.flags = STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE;
            material.alpha_cutoff = 0.5f;
            material.parallax_depth_scale = 0.1f;
            material.max_parallax_layer_count = 16f;
            material.max_relief_mapping_search_steps = 5u;
            material.deferred_lighting_pass_id = 1u;
            material.uv_transform = mat3x3f(vec3f(1f, 0f, 0f), vec3f(0f, 1f, 0f), vec3f(0f, 0f, 1f));
            let _e66 = material;
            return _e66;
        }
    };
    println!("quoted: {source}")
}
