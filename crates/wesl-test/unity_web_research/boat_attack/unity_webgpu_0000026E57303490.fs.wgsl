diagnostic(off, derivative_uniformity);

alias Arr = array<mat4x4<f32>, 4u>;

struct PGlobals {
  /* @offset(0) */
  x_WorldSpaceCameraPos : vec3<f32>,
  /* @offset(16) */
  x_ProjectionParams : vec4<f32>,
  /* @offset(32) */
  x_ZBufferParams : vec4<f32>,
  /* @offset(48) */
  unity_CameraToWorld : mat4x4<f32>,
  /* @offset(112) */
  unity_WorldToShadow : Arr,
  /* @offset(368) */
  x_LightShadowData : vec4<f32>,
  /* @offset(384) */
  unity_ShadowFadeCenterAndType : vec4<f32>,
  /* @offset(400) */
  x_LightPos : vec4<f32>,
  /* @offset(416) */
  x_LightColor : vec4<f32>,
  /* @offset(432) */
  unity_WorldToLight : mat4x4<f32>,
  /* @offset(496) */
  x_ShadowMapTexture_TexelSize : vec4<f32>,
}

var<private> u_xlat0 : vec4<f32>;

@group(1) @binding(0) var<uniform> x_17 : PGlobals;

var<private> vs_TEXCOORD1 : vec3<f32>;

var<private> u_xlat1 : vec3<f32>;

var<private> vs_TEXCOORD0 : vec4<f32>;

var<private> u_xlat27 : f32;

@group(0) @binding(0) var x_CameraDepthTexture : texture_2d<f32>;

@group(0) @binding(8) var sampler_CameraDepthTexture : sampler;

var<private> u_xlat2 : vec4<f32>;

var<private> u_xlat3 : vec4<f32>;

var<private> u_xlat9 : vec3<f32>;

var<private> u_xlat19 : f32;

var<private> u_xlat28 : f32;

@group(0) @binding(1) var x_LightTexture0 : texture_2d<f32>;

@group(0) @binding(10) var sampler_LightTexture0 : sampler;

var<private> u_xlatb20 : bool;

var<private> u_xlat20 : f32;

@group(0) @binding(2) var x_LightTextureB0 : texture_2d<f32>;

@group(0) @binding(9) var sampler_LightTextureB0 : sampler;

var<private> u_xlatb28 : bool;

var<private> u_xlat4 : vec4<f32>;

var<private> u_xlat5 : vec4<f32>;

var<private> u_xlat6 : vec4<f32>;

var<private> u_xlat22 : vec2<f32>;

var<private> u_xlat7 : vec4<f32>;

var<private> u_xlat8 : vec4<f32>;

@group(0) @binding(7) var x_ShadowMapTexture : texture_depth_2d;

@group(0) @binding(11) var sampler_ShadowMapTexture : sampler_comparison;

@group(0) @binding(3) var x_CameraGBufferTexture0 : texture_2d<f32>;

@group(0) @binding(13) var sampler_CameraGBufferTexture0 : sampler;

@group(0) @binding(4) var x_CameraGBufferTexture1 : texture_2d<f32>;

@group(0) @binding(14) var sampler_CameraGBufferTexture1 : sampler;

@group(0) @binding(5) var x_CameraGBufferTexture2 : texture_2d<f32>;

@group(0) @binding(15) var sampler_CameraGBufferTexture2 : sampler;

@group(0) @binding(6) var unity_NHxRoughness : texture_2d<f32>;

@group(0) @binding(12) var samplerunity_NHxRoughness : sampler;

var<private> SV_Target0 : vec4<f32>;

var<private> gl_FragCoord : vec4<f32>;

fn main_1() {
  var txVec0 : vec3<f32>;
  var txVec1 : vec3<f32>;
  var txVec2 : vec3<f32>;
  var txVec3 : vec3<f32>;
  var u_xlat_precise_vec4 : vec4<f32>;
  var u_xlat_precise_ivec4 : vec4<i32>;
  var u_xlat_precise_bvec4 : vec4<bool>;
  var u_xlat_precise_uvec4 : vec4<u32>;
  let x_23 : f32 = x_17.x_ProjectionParams.z;
  let x_28 : f32 = vs_TEXCOORD1.z;
  u_xlat0.x = (x_23 / x_28);
  let x_33 : vec4<f32> = u_xlat0;
  let x_35 : vec3<f32> = vs_TEXCOORD1;
  let x_36 : vec3<f32> = (vec3<f32>(x_33.x, x_33.x, x_33.x) * x_35);
  let x_37 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_36.x, x_36.y, x_36.z, x_37.w);
  let x_44 : vec4<f32> = vs_TEXCOORD0;
  let x_46 : vec4<f32> = vs_TEXCOORD0;
  let x_48 : vec2<f32> = (vec2<f32>(x_44.x, x_44.y) / vec2<f32>(x_46.w, x_46.w));
  let x_49 : vec3<f32> = u_xlat1;
  u_xlat1 = vec3<f32>(x_48.x, x_48.y, x_49.z);
  let x_62 : vec3<f32> = u_xlat1;
  let x_64 : vec4<f32> = textureSample(x_CameraDepthTexture, sampler_CameraDepthTexture, vec2<f32>(x_62.x, x_62.y));
  u_xlat27 = x_64.x;
  let x_68 : f32 = x_17.x_ZBufferParams.x;
  let x_69 : f32 = u_xlat27;
  let x_73 : f32 = x_17.x_ZBufferParams.y;
  u_xlat27 = ((x_68 * x_69) + x_73);
  let x_76 : f32 = u_xlat27;
  u_xlat27 = (1.0f / x_76);
  let x_79 : f32 = u_xlat27;
  let x_81 : vec4<f32> = u_xlat0;
  let x_83 : vec3<f32> = (vec3<f32>(x_79, x_79, x_79) * vec3<f32>(x_81.x, x_81.y, x_81.z));
  let x_84 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_83.x, x_83.y, x_83.z, x_84.w);
  let x_87 : vec4<f32> = u_xlat2;
  let x_92 : vec4<f32> = x_17.unity_CameraToWorld[1i];
  let x_94 : vec3<f32> = (vec3<f32>(x_87.y, x_87.y, x_87.y) * vec3<f32>(x_92.x, x_92.y, x_92.z));
  let x_95 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_94.x, x_94.y, x_94.z, x_95.w);
  let x_99 : vec4<f32> = x_17.unity_CameraToWorld[0i];
  let x_101 : vec4<f32> = u_xlat2;
  let x_104 : vec4<f32> = u_xlat3;
  let x_106 : vec3<f32> = ((vec3<f32>(x_99.x, x_99.y, x_99.z) * vec3<f32>(x_101.x, x_101.x, x_101.x)) + vec3<f32>(x_104.x, x_104.y, x_104.z));
  let x_107 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_106.x, x_106.y, x_107.z, x_106.z);
  let x_110 : vec4<f32> = x_17.unity_CameraToWorld[2i];
  let x_112 : vec4<f32> = u_xlat2;
  let x_115 : vec4<f32> = u_xlat2;
  let x_117 : vec3<f32> = ((vec3<f32>(x_110.x, x_110.y, x_110.z) * vec3<f32>(x_112.z, x_112.z, x_112.z)) + vec3<f32>(x_115.x, x_115.y, x_115.w));
  let x_118 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_117.x, x_117.y, x_118.z, x_117.z);
  let x_120 : vec4<f32> = u_xlat2;
  let x_123 : vec4<f32> = x_17.unity_CameraToWorld[3i];
  let x_125 : vec3<f32> = (vec3<f32>(x_120.x, x_120.y, x_120.w) + vec3<f32>(x_123.x, x_123.y, x_123.z));
  let x_126 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_125.x, x_125.y, x_126.z, x_125.z);
  let x_128 : vec4<f32> = u_xlat2;
  let x_132 : vec4<f32> = x_17.unity_ShadowFadeCenterAndType;
  let x_135 : vec3<f32> = (vec3<f32>(x_128.x, x_128.y, x_128.w) + -(vec3<f32>(x_132.x, x_132.y, x_132.z)));
  let x_136 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_135.x, x_135.y, x_135.z, x_136.w);
  let x_138 : vec4<f32> = u_xlat3;
  let x_140 : vec4<f32> = u_xlat3;
  u_xlat0.x = dot(vec3<f32>(x_138.x, x_138.y, x_138.z), vec3<f32>(x_140.x, x_140.y, x_140.z));
  let x_145 : f32 = u_xlat0.x;
  u_xlat0.x = sqrt(x_145);
  let x_149 : f32 = u_xlat0.z;
  let x_151 : f32 = u_xlat27;
  let x_154 : f32 = u_xlat0.x;
  u_xlat0.x = ((-(x_149) * x_151) + x_154);
  let x_159 : f32 = x_17.unity_ShadowFadeCenterAndType.w;
  let x_161 : f32 = u_xlat0.x;
  let x_164 : f32 = u_xlat2.z;
  u_xlat0.x = ((x_159 * x_161) + x_164);
  let x_168 : vec4<f32> = u_xlat2;
  let x_173 : vec4<f32> = x_17.x_LightPos;
  u_xlat9 = (-(vec3<f32>(x_168.x, x_168.y, x_168.w)) + vec3<f32>(x_173.x, x_173.y, x_173.z));
  let x_177 : vec3<f32> = u_xlat9;
  let x_178 : vec3<f32> = u_xlat9;
  u_xlat19 = dot(x_177, x_178);
  let x_181 : f32 = u_xlat19;
  u_xlat28 = inverseSqrt(x_181);
  let x_183 : vec3<f32> = u_xlat9;
  let x_184 : f32 = u_xlat28;
  u_xlat9 = (x_183 * vec3<f32>(x_184, x_184, x_184));
  let x_187 : vec4<f32> = u_xlat2;
  let x_191 : vec4<f32> = x_17.unity_WorldToLight[1i];
  let x_193 : vec3<f32> = (vec3<f32>(x_187.y, x_187.y, x_187.y) * vec3<f32>(x_191.x, x_191.y, x_191.w));
  let x_194 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_193.x, x_193.y, x_193.z, x_194.w);
  let x_197 : vec4<f32> = x_17.unity_WorldToLight[0i];
  let x_199 : vec4<f32> = u_xlat2;
  let x_202 : vec4<f32> = u_xlat3;
  let x_204 : vec3<f32> = ((vec3<f32>(x_197.x, x_197.y, x_197.w) * vec3<f32>(x_199.x, x_199.x, x_199.x)) + vec3<f32>(x_202.x, x_202.y, x_202.z));
  let x_205 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_204.x, x_204.y, x_204.z, x_205.w);
  let x_208 : vec4<f32> = x_17.unity_WorldToLight[2i];
  let x_210 : vec4<f32> = u_xlat2;
  let x_213 : vec4<f32> = u_xlat3;
  let x_215 : vec3<f32> = ((vec3<f32>(x_208.x, x_208.y, x_208.w) * vec3<f32>(x_210.w, x_210.w, x_210.w)) + vec3<f32>(x_213.x, x_213.y, x_213.z));
  let x_216 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_215.x, x_215.y, x_215.z, x_216.w);
  let x_218 : vec4<f32> = u_xlat3;
  let x_221 : vec4<f32> = x_17.unity_WorldToLight[3i];
  let x_223 : vec3<f32> = (vec3<f32>(x_218.x, x_218.y, x_218.z) + vec3<f32>(x_221.x, x_221.y, x_221.w));
  let x_224 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_223.x, x_223.y, x_223.z, x_224.w);
  let x_226 : vec4<f32> = u_xlat3;
  let x_228 : vec4<f32> = u_xlat3;
  let x_230 : vec2<f32> = (vec2<f32>(x_226.x, x_226.y) / vec2<f32>(x_228.z, x_228.z));
  let x_231 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_230.x, x_230.y, x_231.z, x_231.w);
  let x_238 : vec4<f32> = u_xlat3;
  let x_241 : vec4<f32> = textureSampleBias(x_LightTexture0, sampler_LightTexture0, vec2<f32>(x_238.x, x_238.y), -8.0f);
  u_xlat28 = x_241.w;
  let x_247 : f32 = u_xlat3.z;
  u_xlatb20 = (x_247 < 0.0f);
  let x_251 : bool = u_xlatb20;
  u_xlat20 = select(0.0f, 1.0f, x_251);
  let x_253 : f32 = u_xlat28;
  let x_254 : f32 = u_xlat20;
  u_xlat28 = (x_253 * x_254);
  let x_256 : f32 = u_xlat19;
  let x_258 : f32 = x_17.x_LightPos.w;
  u_xlat19 = (x_256 * x_258);
  let x_265 : f32 = u_xlat19;
  let x_267 : vec4<f32> = textureSample(x_LightTextureB0, sampler_LightTextureB0, vec2<f32>(x_265, x_265));
  u_xlat19 = x_267.x;
  let x_269 : f32 = u_xlat19;
  let x_270 : f32 = u_xlat28;
  u_xlat19 = (x_269 * x_270);
  let x_273 : f32 = u_xlat0.x;
  let x_276 : f32 = x_17.x_LightShadowData.z;
  let x_279 : f32 = x_17.x_LightShadowData.w;
  u_xlat0.x = ((x_273 * x_276) + x_279);
  let x_283 : f32 = u_xlat0.x;
  u_xlat0.x = clamp(x_283, 0.0f, 1.0f);
  let x_288 : f32 = u_xlat0.x;
  u_xlatb28 = (x_288 < 0.99000000953674316406f);
  let x_291 : bool = u_xlatb28;
  if (x_291) {
    let x_294 : vec4<f32> = u_xlat2;
    let x_298 : vec4<f32> = x_17.unity_WorldToShadow[0i][1i];
    u_xlat3 = (vec4<f32>(x_294.y, x_294.y, x_294.y, x_294.y) * x_298);
    let x_301 : vec4<f32> = x_17.unity_WorldToShadow[0i][0i];
    let x_302 : vec4<f32> = u_xlat2;
    let x_305 : vec4<f32> = u_xlat3;
    u_xlat3 = ((x_301 * vec4<f32>(x_302.x, x_302.x, x_302.x, x_302.x)) + x_305);
    let x_308 : vec4<f32> = x_17.unity_WorldToShadow[0i][2i];
    let x_309 : vec4<f32> = u_xlat2;
    let x_312 : vec4<f32> = u_xlat3;
    u_xlat3 = ((x_308 * vec4<f32>(x_309.w, x_309.w, x_309.w, x_309.w)) + x_312);
    let x_314 : vec4<f32> = u_xlat3;
    let x_316 : vec4<f32> = x_17.unity_WorldToShadow[0i][3i];
    u_xlat3 = (x_314 + x_316);
    let x_318 : vec4<f32> = u_xlat3;
    let x_320 : vec4<f32> = u_xlat3;
    let x_322 : vec3<f32> = (vec3<f32>(x_318.x, x_318.y, x_318.z) / vec3<f32>(x_320.w, x_320.w, x_320.w));
    let x_323 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_322.x, x_322.y, x_322.z, x_323.w);
    let x_326 : vec4<f32> = u_xlat3;
    let x_330 : vec4<f32> = x_17.x_ShadowMapTexture_TexelSize;
    let x_335 : vec2<f32> = ((vec2<f32>(x_326.x, x_326.y) * vec2<f32>(x_330.z, x_330.w)) + vec2<f32>(0.5f, 0.5f));
    let x_336 : vec4<f32> = u_xlat4;
    u_xlat4 = vec4<f32>(x_335.x, x_335.y, x_336.z, x_336.w);
    let x_338 : vec4<f32> = u_xlat4;
    let x_340 : vec2<f32> = floor(vec2<f32>(x_338.x, x_338.y));
    let x_341 : vec4<f32> = u_xlat4;
    u_xlat4 = vec4<f32>(x_340.x, x_340.y, x_341.z, x_341.w);
    let x_343 : vec4<f32> = u_xlat3;
    let x_346 : vec4<f32> = x_17.x_ShadowMapTexture_TexelSize;
    let x_349 : vec4<f32> = u_xlat4;
    let x_352 : vec2<f32> = ((vec2<f32>(x_343.x, x_343.y) * vec2<f32>(x_346.z, x_346.w)) + -(vec2<f32>(x_349.x, x_349.y)));
    let x_353 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_352.x, x_352.y, x_353.z, x_353.w);
    let x_356 : vec4<f32> = u_xlat3;
    u_xlat5 = (vec4<f32>(x_356.x, x_356.x, x_356.y, x_356.y) + vec4<f32>(0.5f, 1.0f, 0.5f, 1.0f));
    let x_361 : vec4<f32> = u_xlat5;
    let x_363 : vec4<f32> = u_xlat5;
    let x_365 : vec2<f32> = (vec2<f32>(x_361.x, x_361.z) * vec2<f32>(x_363.x, x_363.z));
    let x_366 : vec4<f32> = u_xlat6;
    u_xlat6 = vec4<f32>(x_365.x, x_366.y, x_366.z, x_365.y);
    let x_370 : vec4<f32> = u_xlat6;
    let x_373 : vec4<f32> = u_xlat3;
    u_xlat22 = ((vec2<f32>(x_370.x, x_370.w) * vec2<f32>(0.5f, 0.5f)) + -(vec2<f32>(x_373.x, x_373.y)));
    let x_377 : vec4<f32> = u_xlat3;
    let x_381 : vec2<f32> = (-(vec2<f32>(x_377.x, x_377.y)) + vec2<f32>(1.0f, 1.0f));
    let x_382 : vec4<f32> = u_xlat5;
    u_xlat5 = vec4<f32>(x_381.x, x_382.y, x_381.y, x_382.w);
    let x_385 : vec4<f32> = u_xlat3;
    let x_388 : vec2<f32> = min(vec2<f32>(x_385.x, x_385.y), vec2<f32>(0.0f, 0.0f));
    let x_389 : vec4<f32> = u_xlat7;
    u_xlat7 = vec4<f32>(x_388.x, x_388.y, x_389.z, x_389.w);
    let x_391 : vec4<f32> = u_xlat7;
    let x_394 : vec4<f32> = u_xlat7;
    let x_397 : vec4<f32> = u_xlat5;
    let x_399 : vec2<f32> = ((-(vec2<f32>(x_391.x, x_391.y)) * vec2<f32>(x_394.x, x_394.y)) + vec2<f32>(x_397.x, x_397.z));
    let x_400 : vec4<f32> = u_xlat5;
    u_xlat5 = vec4<f32>(x_399.x, x_400.y, x_399.y, x_400.w);
    let x_402 : vec4<f32> = u_xlat3;
    let x_404 : vec2<f32> = max(vec2<f32>(x_402.x, x_402.y), vec2<f32>(0.0f, 0.0f));
    let x_405 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_404.x, x_404.y, x_405.z, x_405.w);
    let x_407 : vec4<f32> = u_xlat3;
    let x_410 : vec4<f32> = u_xlat3;
    let x_413 : vec4<f32> = u_xlat5;
    let x_415 : vec2<f32> = ((-(vec2<f32>(x_407.x, x_407.y)) * vec2<f32>(x_410.x, x_410.y)) + vec2<f32>(x_413.y, x_413.w));
    let x_416 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_415.x, x_415.y, x_416.z, x_416.w);
    let x_419 : f32 = u_xlat22.x;
    u_xlat7.x = x_419;
    let x_422 : f32 = u_xlat5.x;
    u_xlat7.y = x_422;
    let x_425 : f32 = u_xlat3.x;
    u_xlat7.z = x_425;
    let x_428 : f32 = u_xlat6.x;
    u_xlat7.w = x_428;
    let x_430 : vec4<f32> = u_xlat7;
    u_xlat7 = (x_430 * vec4<f32>(0.44444000720977783203f, 0.44444000720977783203f, 0.44444000720977783203f, 0.22222000360488891602f));
    let x_436 : f32 = u_xlat22.y;
    u_xlat6.x = x_436;
    let x_439 : f32 = u_xlat5.z;
    u_xlat6.y = x_439;
    let x_442 : f32 = u_xlat3.y;
    u_xlat6.z = x_442;
    let x_444 : vec4<f32> = u_xlat6;
    u_xlat5 = (x_444 * vec4<f32>(0.44444000720977783203f, 0.44444000720977783203f, 0.44444000720977783203f, 0.22222000360488891602f));
    let x_446 : vec4<f32> = u_xlat7;
    let x_448 : vec4<f32> = u_xlat7;
    u_xlat6 = (vec4<f32>(x_446.y, x_446.w, x_446.y, x_446.w) + vec4<f32>(x_448.x, x_448.z, x_448.x, x_448.z));
    let x_452 : vec4<f32> = u_xlat5;
    let x_454 : vec4<f32> = u_xlat5;
    u_xlat8 = (vec4<f32>(x_452.y, x_452.y, x_452.w, x_452.w) + vec4<f32>(x_454.x, x_454.x, x_454.z, x_454.z));
    let x_457 : vec4<f32> = u_xlat7;
    let x_459 : vec4<f32> = u_xlat6;
    let x_461 : vec2<f32> = (vec2<f32>(x_457.y, x_457.w) / vec2<f32>(x_459.z, x_459.w));
    let x_462 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_461.x, x_461.y, x_462.z, x_462.w);
    let x_464 : vec4<f32> = u_xlat3;
    let x_468 : vec2<f32> = (vec2<f32>(x_464.x, x_464.y) + vec2<f32>(-1.5f, 0.5f));
    let x_469 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_468.x, x_468.y, x_469.z, x_469.w);
    let x_471 : vec4<f32> = u_xlat5;
    let x_473 : vec4<f32> = u_xlat8;
    u_xlat22 = (vec2<f32>(x_471.y, x_471.w) / vec2<f32>(x_473.y, x_473.w));
    let x_476 : vec2<f32> = u_xlat22;
    u_xlat22 = (x_476 + vec2<f32>(-1.5f, 0.5f));
    let x_478 : vec4<f32> = u_xlat3;
    let x_481 : vec4<f32> = x_17.x_ShadowMapTexture_TexelSize;
    let x_483 : vec2<f32> = (vec2<f32>(x_478.x, x_478.y) * vec2<f32>(x_481.x, x_481.x));
    let x_484 : vec4<f32> = u_xlat5;
    u_xlat5 = vec4<f32>(x_483.x, x_483.y, x_484.z, x_484.w);
    let x_486 : vec2<f32> = u_xlat22;
    let x_488 : vec4<f32> = x_17.x_ShadowMapTexture_TexelSize;
    let x_490 : vec2<f32> = (x_486 * vec2<f32>(x_488.y, x_488.y));
    let x_491 : vec4<f32> = u_xlat5;
    u_xlat5 = vec4<f32>(x_491.x, x_491.y, x_490.x, x_490.y);
    let x_493 : vec4<f32> = u_xlat6;
    let x_494 : vec4<f32> = u_xlat8;
    u_xlat6 = (x_493 * x_494);
    let x_496 : vec4<f32> = u_xlat4;
    let x_499 : vec4<f32> = x_17.x_ShadowMapTexture_TexelSize;
    let x_502 : vec4<f32> = u_xlat5;
    u_xlat7 = ((vec4<f32>(x_496.x, x_496.y, x_496.x, x_496.y) * vec4<f32>(x_499.x, x_499.y, x_499.x, x_499.y)) + vec4<f32>(x_502.x, x_502.z, x_502.y, x_502.z));
    let x_507 : vec4<f32> = u_xlat7;
    let x_508 : vec2<f32> = vec2<f32>(x_507.x, x_507.y);
    let x_510 : f32 = u_xlat3.z;
    txVec0 = vec3<f32>(x_508.x, x_508.y, x_510);
    let x_522 : vec3<f32> = txVec0;
    let x_524 : f32 = textureSampleCompareLevel(x_ShadowMapTexture, sampler_ShadowMapTexture, x_522.xy, x_522.z);
    u_xlat28 = x_524;
    let x_526 : vec4<f32> = u_xlat7;
    let x_527 : vec2<f32> = vec2<f32>(x_526.z, x_526.w);
    let x_529 : f32 = u_xlat3.z;
    txVec1 = vec3<f32>(x_527.x, x_527.y, x_529);
    let x_536 : vec3<f32> = txVec1;
    let x_538 : f32 = textureSampleCompareLevel(x_ShadowMapTexture, sampler_ShadowMapTexture, x_536.xy, x_536.z);
    u_xlat20 = x_538;
    let x_539 : f32 = u_xlat20;
    let x_541 : f32 = u_xlat6.y;
    u_xlat20 = (x_539 * x_541);
    let x_544 : f32 = u_xlat6.x;
    let x_545 : f32 = u_xlat28;
    let x_547 : f32 = u_xlat20;
    u_xlat28 = ((x_544 * x_545) + x_547);
    let x_549 : vec4<f32> = u_xlat4;
    let x_552 : vec4<f32> = x_17.x_ShadowMapTexture_TexelSize;
    let x_555 : vec4<f32> = u_xlat5;
    u_xlat4 = ((vec4<f32>(x_549.x, x_549.y, x_549.x, x_549.y) * vec4<f32>(x_552.x, x_552.y, x_552.x, x_552.y)) + vec4<f32>(x_555.x, x_555.w, x_555.y, x_555.w));
    let x_559 : vec4<f32> = u_xlat4;
    let x_560 : vec2<f32> = vec2<f32>(x_559.x, x_559.y);
    let x_562 : f32 = u_xlat3.z;
    txVec2 = vec3<f32>(x_560.x, x_560.y, x_562);
    let x_569 : vec3<f32> = txVec2;
    let x_571 : f32 = textureSampleCompareLevel(x_ShadowMapTexture, sampler_ShadowMapTexture, x_569.xy, x_569.z);
    u_xlat20 = x_571;
    let x_573 : f32 = u_xlat6.z;
    let x_574 : f32 = u_xlat20;
    let x_576 : f32 = u_xlat28;
    u_xlat28 = ((x_573 * x_574) + x_576);
    let x_579 : vec4<f32> = u_xlat4;
    let x_580 : vec2<f32> = vec2<f32>(x_579.z, x_579.w);
    let x_582 : f32 = u_xlat3.z;
    txVec3 = vec3<f32>(x_580.x, x_580.y, x_582);
    let x_589 : vec3<f32> = txVec3;
    let x_591 : f32 = textureSampleCompareLevel(x_ShadowMapTexture, sampler_ShadowMapTexture, x_589.xy, x_589.z);
    u_xlat20 = x_591;
    let x_593 : f32 = u_xlat6.w;
    let x_594 : f32 = u_xlat20;
    let x_596 : f32 = u_xlat28;
    u_xlat28 = ((x_593 * x_594) + x_596);
    let x_599 : f32 = x_17.x_LightShadowData.x;
    u_xlat20 = (-(x_599) + 1.0f);
    let x_602 : f32 = u_xlat28;
    let x_603 : f32 = u_xlat20;
    let x_606 : f32 = x_17.x_LightShadowData.x;
    u_xlat28 = ((x_602 * x_603) + x_606);
  } else {
    u_xlat28 = 1.0f;
  }
  let x_609 : f32 = u_xlat28;
  u_xlat20 = (-(x_609) + 1.0f);
  let x_613 : f32 = u_xlat0.x;
  let x_614 : f32 = u_xlat20;
  let x_616 : f32 = u_xlat28;
  u_xlat0.x = ((x_613 * x_614) + x_616);
  let x_620 : f32 = u_xlat0.x;
  let x_621 : f32 = u_xlat19;
  u_xlat0.x = (x_620 * x_621);
  let x_624 : vec4<f32> = u_xlat0;
  let x_628 : vec4<f32> = x_17.x_LightColor;
  let x_630 : vec3<f32> = (vec3<f32>(x_624.x, x_624.x, x_624.x) * vec3<f32>(x_628.x, x_628.y, x_628.z));
  let x_631 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_630.x, x_630.y, x_630.z, x_631.w);
  let x_638 : vec3<f32> = u_xlat1;
  let x_640 : vec4<f32> = textureSample(x_CameraGBufferTexture0, sampler_CameraGBufferTexture0, vec2<f32>(x_638.x, x_638.y));
  let x_641 : vec3<f32> = vec3<f32>(x_640.x, x_640.y, x_640.z);
  let x_642 : vec4<f32> = u_xlat4;
  u_xlat4 = vec4<f32>(x_641.x, x_641.y, x_641.z, x_642.w);
  let x_649 : vec3<f32> = u_xlat1;
  let x_651 : vec4<f32> = textureSample(x_CameraGBufferTexture1, sampler_CameraGBufferTexture1, vec2<f32>(x_649.x, x_649.y));
  u_xlat5 = x_651;
  let x_657 : vec3<f32> = u_xlat1;
  let x_659 : vec4<f32> = textureSample(x_CameraGBufferTexture2, sampler_CameraGBufferTexture2, vec2<f32>(x_657.x, x_657.y));
  u_xlat1 = vec3<f32>(x_659.x, x_659.y, x_659.z);
  let x_661 : vec3<f32> = u_xlat1;
  u_xlat1 = ((x_661 * vec3<f32>(2.0f, 2.0f, 2.0f)) + vec3<f32>(-1.0f, -1.0f, -1.0f));
  let x_668 : vec3<f32> = u_xlat1;
  let x_669 : vec3<f32> = u_xlat1;
  u_xlat0.x = dot(x_668, x_669);
  let x_673 : f32 = u_xlat0.x;
  u_xlat0.x = inverseSqrt(x_673);
  let x_676 : vec4<f32> = u_xlat0;
  let x_678 : vec3<f32> = u_xlat1;
  u_xlat1 = (vec3<f32>(x_676.x, x_676.x, x_676.x) * x_678);
  let x_680 : vec4<f32> = u_xlat2;
  let x_684 : vec3<f32> = x_17.x_WorldSpaceCameraPos;
  let x_686 : vec3<f32> = (vec3<f32>(x_680.x, x_680.y, x_680.w) + -(x_684));
  let x_687 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_686.x, x_686.y, x_686.z, x_687.w);
  let x_689 : vec4<f32> = u_xlat2;
  let x_691 : vec4<f32> = u_xlat2;
  u_xlat0.x = dot(vec3<f32>(x_689.x, x_689.y, x_689.z), vec3<f32>(x_691.x, x_691.y, x_691.z));
  let x_696 : f32 = u_xlat0.x;
  u_xlat0.x = inverseSqrt(x_696);
  let x_699 : vec4<f32> = u_xlat0;
  let x_701 : vec4<f32> = u_xlat2;
  let x_703 : vec3<f32> = (vec3<f32>(x_699.x, x_699.x, x_699.x) * vec3<f32>(x_701.x, x_701.y, x_701.z));
  let x_704 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_703.x, x_703.y, x_703.z, x_704.w);
  let x_706 : vec4<f32> = u_xlat2;
  let x_709 : vec3<f32> = u_xlat1;
  u_xlat0.x = dot(-(vec3<f32>(x_706.x, x_706.y, x_706.z)), x_709);
  let x_713 : f32 = u_xlat0.x;
  let x_715 : f32 = u_xlat0.x;
  u_xlat0.x = (x_713 + x_715);
  let x_718 : vec3<f32> = u_xlat1;
  let x_719 : vec4<f32> = u_xlat0;
  let x_723 : vec4<f32> = u_xlat2;
  let x_726 : vec3<f32> = ((x_718 * -(vec3<f32>(x_719.x, x_719.x, x_719.x))) + -(vec3<f32>(x_723.x, x_723.y, x_723.z)));
  let x_727 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_726.x, x_726.y, x_726.z, x_727.w);
  let x_729 : vec3<f32> = u_xlat1;
  let x_730 : vec3<f32> = u_xlat9;
  u_xlat0.x = dot(x_729, x_730);
  let x_734 : f32 = u_xlat0.x;
  u_xlat0.x = clamp(x_734, 0.0f, 1.0f);
  let x_737 : vec4<f32> = u_xlat2;
  let x_739 : vec3<f32> = u_xlat9;
  u_xlat9.x = dot(vec3<f32>(x_737.x, x_737.y, x_737.z), x_739);
  let x_743 : f32 = u_xlat9.x;
  let x_745 : f32 = u_xlat9.x;
  u_xlat9.x = (x_743 * x_745);
  let x_749 : f32 = u_xlat9.x;
  let x_751 : f32 = u_xlat9.x;
  u_xlat1.x = (x_749 * x_751);
  let x_755 : f32 = u_xlat5.w;
  u_xlat1.y = (-(x_755) + 1.0f);
  let x_764 : vec3<f32> = u_xlat1;
  let x_766 : vec4<f32> = textureSample(unity_NHxRoughness, samplerunity_NHxRoughness, vec2<f32>(x_764.x, x_764.y));
  u_xlat9.x = x_766.x;
  let x_770 : f32 = u_xlat9.x;
  u_xlat9.x = (x_770 * 16.0f);
  let x_774 : vec3<f32> = u_xlat9;
  let x_776 : vec4<f32> = u_xlat5;
  let x_779 : vec4<f32> = u_xlat4;
  u_xlat9 = ((vec3<f32>(x_774.x, x_774.x, x_774.x) * vec3<f32>(x_776.x, x_776.y, x_776.z)) + vec3<f32>(x_779.x, x_779.y, x_779.z));
  let x_782 : vec4<f32> = u_xlat0;
  let x_784 : vec4<f32> = u_xlat3;
  u_xlat1 = (vec3<f32>(x_782.x, x_782.x, x_782.x) * vec3<f32>(x_784.x, x_784.y, x_784.z));
  let x_787 : vec3<f32> = u_xlat9;
  let x_788 : vec3<f32> = u_xlat1;
  let x_789 : vec3<f32> = (x_787 * x_788);
  let x_790 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_789.x, x_789.y, x_789.z, x_790.w);
  u_xlat0.w = 1.0f;
  let x_795 : vec4<f32> = u_xlat0;
  SV_Target0 = exp2(-(x_795));
  return;
}

struct main_out {
  @location(0)
  SV_Target0_1 : vec4<f32>,
}

@fragment
fn main(@location(1) vs_TEXCOORD1_param : vec3<f32>, @location(0) vs_TEXCOORD0_param : vec4<f32>, @builtin(position) gl_FragCoord_param : vec4<f32>) -> main_out {
  vs_TEXCOORD1 = vs_TEXCOORD1_param;
  vs_TEXCOORD0 = vs_TEXCOORD0_param;
  gl_FragCoord = gl_FragCoord_param;
  main_1();
  return main_out(SV_Target0);
}


