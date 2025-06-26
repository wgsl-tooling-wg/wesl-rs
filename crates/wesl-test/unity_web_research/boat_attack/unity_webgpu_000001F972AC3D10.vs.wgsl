diagnostic(off, derivative_uniformity);

alias Arr = array<vec4<f32>, 8u>;

struct VGlobals {
  /* @offset(0) */
  x_WorldSpaceCameraPos : vec3<f32>,
  /* @offset(16) */
  x_ScreenParams : vec4<f32>,
  /* @offset(32) */
  unity_4LightPosX0 : vec4<f32>,
  /* @offset(48) */
  unity_4LightPosY0 : vec4<f32>,
  /* @offset(64) */
  unity_4LightPosZ0 : vec4<f32>,
  /* @offset(80) */
  unity_4LightAtten0 : vec4<f32>,
  /* @offset(96) */
  unity_LightColor : Arr,
  /* @offset(224) */
  unity_SHBr : vec4<f32>,
  /* @offset(240) */
  unity_SHBg : vec4<f32>,
  /* @offset(256) */
  unity_SHBb : vec4<f32>,
  /* @offset(272) */
  unity_SHC : vec4<f32>,
  /* @offset(288) */
  unity_ObjectToWorld : mat4x4<f32>,
  /* @offset(352) */
  unity_WorldToObject : mat4x4<f32>,
  /* @offset(416) */
  unity_WorldTransformParams : vec4<f32>,
  /* @offset(432) */
  glstate_matrix_projection : mat4x4<f32>,
  /* @offset(496) */
  unity_MatrixVP : mat4x4<f32>,
  /* @offset(560) */
  x_FaceDilate : f32,
  /* @offset(576) */
  x_EnvMatrix : mat4x4<f32>,
  /* @offset(640) */
  x_WeightNormal : f32,
  /* @offset(644) */
  x_WeightBold : f32,
  /* @offset(648) */
  x_ScaleRatioA : f32,
  /* @offset(652) */
  x_VertexOffsetX : f32,
  /* @offset(656) */
  x_VertexOffsetY : f32,
  /* @offset(660) */
  x_GradientScale : f32,
  /* @offset(664) */
  x_ScaleX : f32,
  /* @offset(668) */
  x_ScaleY : f32,
  /* @offset(672) */
  x_PerspectiveFilter : f32,
  /* @offset(676) */
  x_Sharpness : f32,
  /* @offset(688) */
  x_MainTex_ST : vec4<f32>,
  /* @offset(704) */
  x_FaceTex_ST : vec4<f32>,
  /* @offset(720) */
  x_OutlineTex_ST : vec4<f32>,
}

var<private> u_xlat0 : vec4<f32>;

var<private> in_POSITION0 : vec4<f32>;

@group(1) @binding(1) var<uniform> x_22 : VGlobals;

var<private> u_xlat1 : vec4<f32>;

var<private> u_xlat2 : vec4<f32>;

var<private> u_xlat3 : vec4<f32>;

var<private> vs_TEXCOORD8 : f32;

var<private> u_xlat15 : f32;

var<private> in_TEXCOORD1 : vec4<f32>;

var<private> vs_TEXCOORD0 : vec4<f32>;

var<private> vs_TEXCOORD1 : vec2<f32>;

var<private> in_TEXCOORD0 : vec4<f32>;

var<private> u_xlatb15 : bool;

var<private> u_xlat16 : f32;

var<private> vs_TEXCOORD5 : vec2<f32>;

var<private> u_xlat7 : vec3<f32>;

var<private> in_NORMAL0 : vec3<f32>;

var<private> u_xlati5 : i32;

var<private> u_xlati0 : i32;

var<private> in_TANGENT0 : vec4<f32>;

var<private> u_xlat4 : vec4<f32>;

var<private> vs_TEXCOORD2 : vec4<f32>;

var<private> vs_TEXCOORD3 : vec4<f32>;

var<private> vs_TEXCOORD4 : vec4<f32>;

var<private> vs_COLOR0 : vec4<f32>;

var<private> in_COLOR0 : vec4<f32>;

var<private> vs_TEXCOORD6 : vec3<f32>;

var<private> vs_TEXCOORD7 : vec3<f32>;

var<private> gl_Position : vec4<f32>;

fn main_1() {
  var u_xlat_precise_vec4 : vec4<f32>;
  var u_xlat_precise_ivec4 : vec4<i32>;
  var u_xlat_precise_bvec4 : vec4<bool>;
  var u_xlat_precise_uvec4 : vec4<u32>;
  let x_13 : vec4<f32> = in_POSITION0;
  let x_27 : f32 = x_22.x_VertexOffsetX;
  let x_30 : f32 = x_22.x_VertexOffsetY;
  let x_32 : vec2<f32> = (vec2<f32>(x_13.x, x_13.y) + vec2<f32>(x_27, x_30));
  let x_33 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_32.x, x_32.y, x_33.z, x_33.w);
  let x_36 : vec4<f32> = u_xlat0;
  let x_42 : vec4<f32> = x_22.unity_ObjectToWorld[1i];
  u_xlat1 = (vec4<f32>(x_36.y, x_36.y, x_36.y, x_36.y) * x_42);
  let x_46 : vec4<f32> = x_22.unity_ObjectToWorld[0i];
  let x_47 : vec4<f32> = u_xlat0;
  let x_50 : vec4<f32> = u_xlat1;
  u_xlat1 = ((x_46 * vec4<f32>(x_47.x, x_47.x, x_47.x, x_47.x)) + x_50);
  let x_54 : vec4<f32> = x_22.unity_ObjectToWorld[2i];
  let x_55 : vec4<f32> = in_POSITION0;
  let x_58 : vec4<f32> = u_xlat1;
  u_xlat1 = ((x_54 * vec4<f32>(x_55.z, x_55.z, x_55.z, x_55.z)) + x_58);
  let x_61 : vec4<f32> = u_xlat1;
  let x_64 : vec4<f32> = x_22.unity_ObjectToWorld[3i];
  u_xlat2 = (x_61 + x_64);
  let x_67 : vec4<f32> = x_22.unity_ObjectToWorld[3i];
  let x_69 : vec4<f32> = in_POSITION0;
  let x_72 : vec4<f32> = u_xlat1;
  let x_74 : vec3<f32> = ((vec3<f32>(x_67.x, x_67.y, x_67.z) * vec3<f32>(x_69.w, x_69.w, x_69.w)) + vec3<f32>(x_72.x, x_72.y, x_72.z));
  let x_75 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_74.x, x_74.y, x_74.z, x_75.w);
  let x_78 : vec4<f32> = u_xlat2;
  let x_82 : vec4<f32> = x_22.unity_MatrixVP[1i];
  u_xlat3 = (vec4<f32>(x_78.y, x_78.y, x_78.y, x_78.y) * x_82);
  let x_85 : vec4<f32> = x_22.unity_MatrixVP[0i];
  let x_86 : vec4<f32> = u_xlat2;
  let x_89 : vec4<f32> = u_xlat3;
  u_xlat3 = ((x_85 * vec4<f32>(x_86.x, x_86.x, x_86.x, x_86.x)) + x_89);
  let x_92 : vec4<f32> = x_22.unity_MatrixVP[2i];
  let x_93 : vec4<f32> = u_xlat2;
  let x_96 : vec4<f32> = u_xlat3;
  u_xlat3 = ((x_92 * vec4<f32>(x_93.z, x_93.z, x_93.z, x_93.z)) + x_96);
  let x_99 : vec4<f32> = x_22.unity_MatrixVP[3i];
  let x_100 : vec4<f32> = u_xlat2;
  let x_103 : vec4<f32> = u_xlat3;
  u_xlat3 = ((x_99 * vec4<f32>(x_100.w, x_100.w, x_100.w, x_100.w)) + x_103);
  let x_110 : vec4<f32> = u_xlat3;
  gl_Position = x_110;
  let x_118 : f32 = u_xlat3.z;
  vs_TEXCOORD8 = x_118;
  let x_124 : f32 = in_TEXCOORD1.x;
  u_xlat15 = (x_124 * 0.000244140625f);
  let x_127 : f32 = u_xlat15;
  u_xlat3.x = floor(x_127);
  let x_131 : f32 = u_xlat3.x;
  let x_136 : f32 = in_TEXCOORD1.x;
  u_xlat3.y = ((-(x_131) * 4096.0f) + x_136);
  let x_139 : vec4<f32> = u_xlat3;
  let x_143 : vec2<f32> = (vec2<f32>(x_139.x, x_139.y) * vec2<f32>(0.001953125f, 0.001953125f));
  let x_144 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_143.x, x_143.y, x_144.z, x_144.w);
  let x_147 : vec4<f32> = u_xlat3;
  let x_151 : vec4<f32> = x_22.x_FaceTex_ST;
  let x_155 : vec4<f32> = x_22.x_FaceTex_ST;
  let x_157 : vec2<f32> = ((vec2<f32>(x_147.x, x_147.y) * vec2<f32>(x_151.x, x_151.y)) + vec2<f32>(x_155.z, x_155.w));
  let x_158 : vec4<f32> = vs_TEXCOORD0;
  vs_TEXCOORD0 = vec4<f32>(x_158.x, x_158.y, x_157.x, x_157.y);
  let x_162 : vec4<f32> = u_xlat3;
  let x_166 : vec4<f32> = x_22.x_OutlineTex_ST;
  let x_170 : vec4<f32> = x_22.x_OutlineTex_ST;
  vs_TEXCOORD1 = ((vec2<f32>(x_162.x, x_162.y) * vec2<f32>(x_166.x, x_166.y)) + vec2<f32>(x_170.z, x_170.w));
  let x_174 : vec4<f32> = in_TEXCOORD0;
  let x_178 : vec4<f32> = x_22.x_MainTex_ST;
  let x_182 : vec4<f32> = x_22.x_MainTex_ST;
  let x_184 : vec2<f32> = ((vec2<f32>(x_174.x, x_174.y) * vec2<f32>(x_178.x, x_178.y)) + vec2<f32>(x_182.z, x_182.w));
  let x_185 : vec4<f32> = vs_TEXCOORD0;
  vs_TEXCOORD0 = vec4<f32>(x_184.x, x_184.y, x_185.z, x_185.w);
  let x_192 : f32 = in_TEXCOORD1.y;
  u_xlatb15 = (0.0f >= x_192);
  let x_194 : bool = u_xlatb15;
  u_xlat15 = select(0.0f, 1.0f, x_194);
  let x_200 : f32 = x_22.x_WeightNormal;
  let x_204 : f32 = x_22.x_WeightBold;
  u_xlat16 = (-(x_200) + x_204);
  let x_206 : f32 = u_xlat15;
  let x_207 : f32 = u_xlat16;
  let x_210 : f32 = x_22.x_WeightNormal;
  u_xlat15 = ((x_206 * x_207) + x_210);
  let x_212 : f32 = u_xlat15;
  let x_217 : f32 = x_22.x_FaceDilate;
  u_xlat15 = ((x_212 * 0.25f) + x_217);
  let x_219 : f32 = u_xlat15;
  let x_222 : f32 = x_22.x_ScaleRatioA;
  u_xlat15 = (x_219 * x_222);
  let x_225 : f32 = u_xlat15;
  vs_TEXCOORD5.x = (x_225 * 0.5f);
  let x_230 : f32 = u_xlat2.y;
  let x_233 : f32 = x_22.unity_MatrixVP[1i].w;
  u_xlat15 = (x_230 * x_233);
  let x_236 : f32 = x_22.unity_MatrixVP[0i].w;
  let x_238 : f32 = u_xlat2.x;
  let x_240 : f32 = u_xlat15;
  u_xlat15 = ((x_236 * x_238) + x_240);
  let x_243 : f32 = x_22.unity_MatrixVP[2i].w;
  let x_245 : f32 = u_xlat2.z;
  let x_247 : f32 = u_xlat15;
  u_xlat15 = ((x_243 * x_245) + x_247);
  let x_250 : f32 = x_22.unity_MatrixVP[3i].w;
  let x_252 : f32 = u_xlat2.w;
  let x_254 : f32 = u_xlat15;
  u_xlat15 = ((x_250 * x_252) + x_254);
  let x_257 : vec4<f32> = x_22.x_ScreenParams;
  let x_261 : vec4<f32> = x_22.glstate_matrix_projection[1i];
  let x_263 : vec2<f32> = (vec2<f32>(x_257.y, x_257.y) * vec2<f32>(x_261.x, x_261.y));
  let x_264 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_263.x, x_263.y, x_264.z, x_264.w);
  let x_267 : vec4<f32> = x_22.glstate_matrix_projection[0i];
  let x_270 : vec4<f32> = x_22.x_ScreenParams;
  let x_273 : vec4<f32> = u_xlat2;
  let x_275 : vec2<f32> = ((vec2<f32>(x_267.x, x_267.y) * vec2<f32>(x_270.x, x_270.x)) + vec2<f32>(x_273.x, x_273.y));
  let x_276 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_275.x, x_275.y, x_276.z, x_276.w);
  let x_278 : vec4<f32> = u_xlat2;
  let x_282 : f32 = x_22.x_ScaleX;
  let x_285 : f32 = x_22.x_ScaleY;
  let x_287 : vec2<f32> = (vec2<f32>(x_278.x, x_278.y) * vec2<f32>(x_282, x_285));
  let x_288 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_287.x, x_287.y, x_288.z, x_288.w);
  let x_290 : f32 = u_xlat15;
  let x_292 : vec4<f32> = u_xlat2;
  let x_294 : vec2<f32> = (vec2<f32>(x_290, x_290) / vec2<f32>(x_292.x, x_292.y));
  let x_295 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_294.x, x_294.y, x_295.z, x_295.w);
  let x_297 : vec4<f32> = u_xlat2;
  let x_299 : vec4<f32> = u_xlat2;
  u_xlat15 = dot(vec2<f32>(x_297.x, x_297.y), vec2<f32>(x_299.x, x_299.y));
  let x_302 : f32 = u_xlat15;
  u_xlat15 = inverseSqrt(x_302);
  let x_305 : f32 = in_TEXCOORD1.y;
  let x_309 : f32 = x_22.x_GradientScale;
  u_xlat16 = (abs(x_305) * x_309);
  let x_313 : f32 = x_22.x_Sharpness;
  u_xlat2.x = (x_313 + 1.0f);
  let x_316 : f32 = u_xlat16;
  let x_318 : f32 = u_xlat2.x;
  u_xlat16 = (x_316 * x_318);
  let x_320 : f32 = u_xlat15;
  let x_321 : f32 = u_xlat16;
  u_xlat2.x = (x_320 * x_321);
  let x_328 : f32 = x_22.x_PerspectiveFilter;
  u_xlat7.x = (-(x_328) + 1.0f);
  let x_333 : f32 = u_xlat7.x;
  let x_335 : f32 = u_xlat2.x;
  u_xlat2.x = (x_333 * x_335);
  let x_338 : f32 = u_xlat15;
  let x_339 : f32 = u_xlat16;
  let x_342 : f32 = u_xlat2.x;
  u_xlat15 = ((x_338 * x_339) + -(x_342));
  let x_347 : vec3<f32> = x_22.x_WorldSpaceCameraPos;
  let x_351 : vec4<f32> = x_22.unity_WorldToObject[1i];
  u_xlat7 = (vec3<f32>(x_347.y, x_347.y, x_347.y) * vec3<f32>(x_351.x, x_351.y, x_351.z));
  let x_355 : vec4<f32> = x_22.unity_WorldToObject[0i];
  let x_358 : vec3<f32> = x_22.x_WorldSpaceCameraPos;
  let x_361 : vec3<f32> = u_xlat7;
  u_xlat7 = ((vec3<f32>(x_355.x, x_355.y, x_355.z) * vec3<f32>(x_358.x, x_358.x, x_358.x)) + x_361);
  let x_364 : vec4<f32> = x_22.unity_WorldToObject[2i];
  let x_367 : vec3<f32> = x_22.x_WorldSpaceCameraPos;
  let x_370 : vec3<f32> = u_xlat7;
  u_xlat7 = ((vec3<f32>(x_364.x, x_364.y, x_364.z) * vec3<f32>(x_367.z, x_367.z, x_367.z)) + x_370);
  let x_372 : vec3<f32> = u_xlat7;
  let x_374 : vec4<f32> = x_22.unity_WorldToObject[3i];
  u_xlat7 = (x_372 + vec3<f32>(x_374.x, x_374.y, x_374.z));
  let x_378 : f32 = in_POSITION0.z;
  u_xlat0.z = x_378;
  let x_380 : vec4<f32> = u_xlat0;
  let x_383 : vec3<f32> = u_xlat7;
  let x_384 : vec3<f32> = (-(vec3<f32>(x_380.x, x_380.y, x_380.z)) + x_383);
  let x_385 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_384.x, x_384.y, x_384.z, x_385.w);
  let x_389 : vec3<f32> = in_NORMAL0;
  let x_390 : vec4<f32> = u_xlat0;
  u_xlat0.x = dot(x_389, vec3<f32>(x_390.x, x_390.y, x_390.z));
  let x_397 : f32 = u_xlat0.x;
  u_xlati5 = bitcast<i32>(select(0u, 4294967295u, (0.0f < x_397)));
  let x_404 : f32 = u_xlat0.x;
  u_xlati0 = bitcast<i32>(select(0u, 4294967295u, (x_404 < 0.0f)));
  let x_408 : i32 = u_xlati5;
  let x_410 : i32 = u_xlati0;
  u_xlati0 = (-(x_408) + x_410);
  let x_412 : i32 = u_xlati0;
  u_xlat0.x = f32(x_412);
  let x_415 : vec4<f32> = u_xlat0;
  let x_417 : vec3<f32> = in_NORMAL0;
  let x_418 : vec3<f32> = (vec3<f32>(x_415.x, x_415.x, x_415.x) * x_417);
  let x_419 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_418.x, x_418.y, x_418.z, x_419.w);
  let x_421 : vec4<f32> = u_xlat0;
  let x_424 : vec4<f32> = x_22.unity_WorldToObject[0i];
  u_xlat3.x = dot(vec3<f32>(x_421.x, x_421.y, x_421.z), vec3<f32>(x_424.x, x_424.y, x_424.z));
  let x_428 : vec4<f32> = u_xlat0;
  let x_431 : vec4<f32> = x_22.unity_WorldToObject[1i];
  u_xlat3.y = dot(vec3<f32>(x_428.x, x_428.y, x_428.z), vec3<f32>(x_431.x, x_431.y, x_431.z));
  let x_435 : vec4<f32> = u_xlat0;
  let x_438 : vec4<f32> = x_22.unity_WorldToObject[2i];
  u_xlat3.z = dot(vec3<f32>(x_435.x, x_435.y, x_435.z), vec3<f32>(x_438.x, x_438.y, x_438.z));
  let x_442 : vec4<f32> = u_xlat3;
  let x_444 : vec4<f32> = u_xlat3;
  u_xlat0.x = dot(vec3<f32>(x_442.x, x_442.y, x_442.z), vec3<f32>(x_444.x, x_444.y, x_444.z));
  let x_449 : f32 = u_xlat0.x;
  u_xlat0.x = inverseSqrt(x_449);
  let x_452 : vec4<f32> = u_xlat0;
  let x_454 : vec4<f32> = u_xlat3;
  u_xlat3 = (vec4<f32>(x_452.x, x_452.x, x_452.x, x_452.x) * vec4<f32>(x_454.x, x_454.y, x_454.z, x_454.z));
  let x_457 : vec4<f32> = u_xlat1;
  let x_461 : vec3<f32> = x_22.x_WorldSpaceCameraPos;
  let x_462 : vec3<f32> = (-(vec3<f32>(x_457.x, x_457.y, x_457.z)) + x_461);
  let x_463 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_462.x, x_462.y, x_462.z, x_463.w);
  let x_465 : vec4<f32> = u_xlat0;
  let x_467 : vec4<f32> = u_xlat0;
  u_xlat16 = dot(vec3<f32>(x_465.x, x_465.y, x_465.z), vec3<f32>(x_467.x, x_467.y, x_467.z));
  let x_470 : f32 = u_xlat16;
  u_xlat16 = inverseSqrt(x_470);
  let x_472 : vec4<f32> = u_xlat0;
  let x_474 : f32 = u_xlat16;
  u_xlat7 = (vec3<f32>(x_472.x, x_472.y, x_472.z) * vec3<f32>(x_474, x_474, x_474));
  let x_477 : vec4<f32> = u_xlat3;
  let x_479 : vec3<f32> = u_xlat7;
  u_xlat16 = dot(vec3<f32>(x_477.x, x_477.y, x_477.w), x_479);
  let x_481 : f32 = u_xlat16;
  let x_483 : f32 = u_xlat15;
  let x_486 : f32 = u_xlat2.x;
  vs_TEXCOORD5.y = ((abs(x_481) * x_483) + x_486);
  let x_490 : vec4<f32> = in_TANGENT0;
  let x_493 : vec4<f32> = x_22.unity_ObjectToWorld[1i];
  let x_495 : vec3<f32> = (vec3<f32>(x_490.y, x_490.y, x_490.y) * vec3<f32>(x_493.y, x_493.z, x_493.x));
  let x_496 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_495.x, x_495.y, x_495.z, x_496.w);
  let x_499 : vec4<f32> = x_22.unity_ObjectToWorld[0i];
  let x_501 : vec4<f32> = in_TANGENT0;
  let x_504 : vec4<f32> = u_xlat2;
  let x_506 : vec3<f32> = ((vec3<f32>(x_499.y, x_499.z, x_499.x) * vec3<f32>(x_501.x, x_501.x, x_501.x)) + vec3<f32>(x_504.x, x_504.y, x_504.z));
  let x_507 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_506.x, x_506.y, x_506.z, x_507.w);
  let x_510 : vec4<f32> = x_22.unity_ObjectToWorld[2i];
  let x_512 : vec4<f32> = in_TANGENT0;
  let x_515 : vec4<f32> = u_xlat2;
  let x_517 : vec3<f32> = ((vec3<f32>(x_510.y, x_510.z, x_510.x) * vec3<f32>(x_512.z, x_512.z, x_512.z)) + vec3<f32>(x_515.x, x_515.y, x_515.z));
  let x_518 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_517.x, x_517.y, x_517.z, x_518.w);
  let x_520 : vec4<f32> = u_xlat2;
  let x_522 : vec4<f32> = u_xlat2;
  u_xlat15 = dot(vec3<f32>(x_520.x, x_520.y, x_520.z), vec3<f32>(x_522.x, x_522.y, x_522.z));
  let x_525 : f32 = u_xlat15;
  u_xlat15 = inverseSqrt(x_525);
  let x_527 : f32 = u_xlat15;
  let x_529 : vec4<f32> = u_xlat2;
  let x_531 : vec3<f32> = (vec3<f32>(x_527, x_527, x_527) * vec3<f32>(x_529.x, x_529.y, x_529.z));
  let x_532 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_531.x, x_531.y, x_531.z, x_532.w);
  let x_535 : vec4<f32> = u_xlat2;
  let x_537 : vec4<f32> = u_xlat3;
  let x_539 : vec3<f32> = (vec3<f32>(x_535.x, x_535.y, x_535.z) * vec3<f32>(x_537.w, x_537.x, x_537.y));
  let x_540 : vec4<f32> = u_xlat4;
  u_xlat4 = vec4<f32>(x_539.x, x_539.y, x_539.z, x_540.w);
  let x_542 : vec4<f32> = u_xlat3;
  let x_544 : vec4<f32> = u_xlat2;
  let x_547 : vec4<f32> = u_xlat4;
  let x_550 : vec3<f32> = ((vec3<f32>(x_542.y, x_542.w, x_542.x) * vec3<f32>(x_544.y, x_544.z, x_544.x)) + -(vec3<f32>(x_547.x, x_547.y, x_547.z)));
  let x_551 : vec4<f32> = u_xlat4;
  u_xlat4 = vec4<f32>(x_550.x, x_550.y, x_550.z, x_551.w);
  let x_554 : f32 = in_TANGENT0.w;
  let x_557 : f32 = x_22.unity_WorldTransformParams.w;
  u_xlat15 = (x_554 * x_557);
  let x_559 : f32 = u_xlat15;
  let x_561 : vec4<f32> = u_xlat4;
  let x_563 : vec3<f32> = (vec3<f32>(x_559, x_559, x_559) * vec3<f32>(x_561.x, x_561.y, x_561.z));
  let x_564 : vec4<f32> = u_xlat4;
  u_xlat4 = vec4<f32>(x_563.x, x_563.y, x_563.z, x_564.w);
  let x_568 : f32 = u_xlat4.x;
  vs_TEXCOORD2.y = x_568;
  let x_571 : f32 = u_xlat3.x;
  vs_TEXCOORD2.z = x_571;
  let x_574 : f32 = u_xlat2.z;
  vs_TEXCOORD2.x = x_574;
  let x_577 : f32 = u_xlat1.x;
  vs_TEXCOORD2.w = x_577;
  let x_581 : f32 = u_xlat2.x;
  vs_TEXCOORD3.x = x_581;
  let x_585 : f32 = u_xlat2.y;
  vs_TEXCOORD4.x = x_585;
  let x_588 : f32 = u_xlat3.y;
  vs_TEXCOORD3.z = x_588;
  let x_591 : f32 = u_xlat4.y;
  vs_TEXCOORD3.y = x_591;
  let x_594 : f32 = u_xlat4.z;
  vs_TEXCOORD4.y = x_594;
  let x_597 : f32 = u_xlat1.y;
  vs_TEXCOORD3.w = x_597;
  let x_600 : f32 = u_xlat3.w;
  vs_TEXCOORD4.z = x_600;
  let x_603 : f32 = u_xlat1.z;
  vs_TEXCOORD4.w = x_603;
  let x_607 : vec4<f32> = in_COLOR0;
  vs_COLOR0 = x_607;
  let x_608 : vec4<f32> = u_xlat0;
  let x_612 : vec4<f32> = x_22.x_EnvMatrix[1i];
  let x_614 : vec3<f32> = (vec3<f32>(x_608.y, x_608.y, x_608.y) * vec3<f32>(x_612.x, x_612.y, x_612.z));
  let x_615 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_614.x, x_614.y, x_614.z, x_615.w);
  let x_618 : vec4<f32> = x_22.x_EnvMatrix[0i];
  let x_620 : vec4<f32> = u_xlat0;
  let x_623 : vec4<f32> = u_xlat2;
  let x_625 : vec3<f32> = ((vec3<f32>(x_618.x, x_618.y, x_618.z) * vec3<f32>(x_620.x, x_620.x, x_620.x)) + vec3<f32>(x_623.x, x_623.y, x_623.z));
  let x_626 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_625.x, x_625.y, x_626.z, x_625.z);
  let x_631 : vec4<f32> = x_22.x_EnvMatrix[2i];
  let x_633 : vec4<f32> = u_xlat0;
  let x_636 : vec4<f32> = u_xlat0;
  vs_TEXCOORD6 = ((vec3<f32>(x_631.x, x_631.y, x_631.z) * vec3<f32>(x_633.z, x_633.z, x_633.z)) + vec3<f32>(x_636.x, x_636.y, x_636.w));
  let x_639 : vec4<f32> = u_xlat1;
  let x_643 : vec4<f32> = x_22.unity_4LightPosY0;
  u_xlat0 = (-(vec4<f32>(x_639.y, x_639.y, x_639.y, x_639.y)) + x_643);
  let x_645 : vec4<f32> = u_xlat3;
  let x_647 : vec4<f32> = u_xlat0;
  u_xlat2 = (vec4<f32>(x_645.y, x_645.y, x_645.y, x_645.y) * x_647);
  let x_649 : vec4<f32> = u_xlat0;
  let x_650 : vec4<f32> = u_xlat0;
  u_xlat0 = (x_649 * x_650);
  let x_652 : vec4<f32> = u_xlat1;
  let x_656 : vec4<f32> = x_22.unity_4LightPosX0;
  u_xlat4 = (-(vec4<f32>(x_652.x, x_652.x, x_652.x, x_652.x)) + x_656);
  let x_658 : vec4<f32> = u_xlat1;
  let x_663 : vec4<f32> = x_22.unity_4LightPosZ0;
  u_xlat1 = (-(vec4<f32>(x_658.z, x_658.z, x_658.z, x_658.z)) + x_663);
  let x_665 : vec4<f32> = u_xlat4;
  let x_666 : vec4<f32> = u_xlat3;
  let x_669 : vec4<f32> = u_xlat2;
  u_xlat2 = ((x_665 * vec4<f32>(x_666.x, x_666.x, x_666.x, x_666.x)) + x_669);
  let x_671 : vec4<f32> = u_xlat4;
  let x_672 : vec4<f32> = u_xlat4;
  let x_674 : vec4<f32> = u_xlat0;
  u_xlat0 = ((x_671 * x_672) + x_674);
  let x_676 : vec4<f32> = u_xlat1;
  let x_677 : vec4<f32> = u_xlat1;
  let x_679 : vec4<f32> = u_xlat0;
  u_xlat0 = ((x_676 * x_677) + x_679);
  let x_681 : vec4<f32> = u_xlat1;
  let x_682 : vec4<f32> = u_xlat3;
  let x_685 : vec4<f32> = u_xlat2;
  u_xlat1 = ((x_681 * vec4<f32>(x_682.w, x_682.w, x_682.z, x_682.w)) + x_685);
  let x_687 : vec4<f32> = u_xlat0;
  u_xlat0 = max(x_687, vec4<f32>(0.00000099999999747524f, 0.00000099999999747524f, 0.00000099999999747524f, 0.00000099999999747524f));
  let x_691 : vec4<f32> = u_xlat0;
  u_xlat2 = inverseSqrt(x_691);
  let x_693 : vec4<f32> = u_xlat0;
  let x_696 : vec4<f32> = x_22.unity_4LightAtten0;
  u_xlat0 = ((x_693 * x_696) + vec4<f32>(1.0f, 1.0f, 1.0f, 1.0f));
  let x_700 : vec4<f32> = u_xlat0;
  u_xlat0 = (vec4<f32>(1.0f, 1.0f, 1.0f, 1.0f) / x_700);
  let x_702 : vec4<f32> = u_xlat1;
  let x_703 : vec4<f32> = u_xlat2;
  u_xlat1 = (x_702 * x_703);
  let x_705 : vec4<f32> = u_xlat1;
  u_xlat1 = max(x_705, vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f));
  let x_708 : vec4<f32> = u_xlat0;
  let x_709 : vec4<f32> = u_xlat1;
  u_xlat0 = (x_708 * x_709);
  let x_711 : vec4<f32> = u_xlat0;
  let x_715 : vec4<f32> = x_22.unity_LightColor[1i];
  let x_717 : vec3<f32> = (vec3<f32>(x_711.y, x_711.y, x_711.y) * vec3<f32>(x_715.x, x_715.y, x_715.z));
  let x_718 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_717.x, x_717.y, x_717.z, x_718.w);
  let x_721 : vec4<f32> = x_22.unity_LightColor[0i];
  let x_723 : vec4<f32> = u_xlat0;
  let x_726 : vec4<f32> = u_xlat1;
  let x_728 : vec3<f32> = ((vec3<f32>(x_721.x, x_721.y, x_721.z) * vec3<f32>(x_723.x, x_723.x, x_723.x)) + vec3<f32>(x_726.x, x_726.y, x_726.z));
  let x_729 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_728.x, x_728.y, x_728.z, x_729.w);
  let x_732 : vec4<f32> = x_22.unity_LightColor[2i];
  let x_734 : vec4<f32> = u_xlat0;
  let x_737 : vec4<f32> = u_xlat1;
  let x_739 : vec3<f32> = ((vec3<f32>(x_732.x, x_732.y, x_732.z) * vec3<f32>(x_734.z, x_734.z, x_734.z)) + vec3<f32>(x_737.x, x_737.y, x_737.z));
  let x_740 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_739.x, x_739.y, x_739.z, x_740.w);
  let x_743 : vec4<f32> = x_22.unity_LightColor[3i];
  let x_745 : vec4<f32> = u_xlat0;
  let x_748 : vec4<f32> = u_xlat0;
  let x_750 : vec3<f32> = ((vec3<f32>(x_743.x, x_743.y, x_743.z) * vec3<f32>(x_745.w, x_745.w, x_745.w)) + vec3<f32>(x_748.x, x_748.y, x_748.z));
  let x_751 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_750.x, x_750.y, x_750.z, x_751.w);
  let x_754 : f32 = u_xlat3.y;
  let x_756 : f32 = u_xlat3.y;
  u_xlat15 = (x_754 * x_756);
  let x_759 : f32 = u_xlat3.x;
  let x_761 : f32 = u_xlat3.x;
  let x_763 : f32 = u_xlat15;
  u_xlat15 = ((x_759 * x_761) + -(x_763));
  let x_766 : vec4<f32> = u_xlat3;
  let x_768 : vec4<f32> = u_xlat3;
  u_xlat1 = (vec4<f32>(x_766.y, x_766.w, x_766.z, x_766.x) * x_768);
  let x_772 : vec4<f32> = x_22.unity_SHBr;
  let x_773 : vec4<f32> = u_xlat1;
  u_xlat2.x = dot(x_772, x_773);
  let x_778 : vec4<f32> = x_22.unity_SHBg;
  let x_779 : vec4<f32> = u_xlat1;
  u_xlat2.y = dot(x_778, x_779);
  let x_784 : vec4<f32> = x_22.unity_SHBb;
  let x_785 : vec4<f32> = u_xlat1;
  u_xlat2.z = dot(x_784, x_785);
  let x_790 : vec4<f32> = x_22.unity_SHC;
  let x_792 : f32 = u_xlat15;
  let x_795 : vec4<f32> = u_xlat2;
  let x_797 : vec3<f32> = ((vec3<f32>(x_790.x, x_790.y, x_790.z) * vec3<f32>(x_792, x_792, x_792)) + vec3<f32>(x_795.x, x_795.y, x_795.z));
  let x_798 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_797.x, x_797.y, x_797.z, x_798.w);
  let x_801 : vec4<f32> = u_xlat0;
  let x_803 : vec4<f32> = u_xlat1;
  vs_TEXCOORD7 = (vec3<f32>(x_801.x, x_801.y, x_801.z) + vec3<f32>(x_803.x, x_803.y, x_803.z));
  return;
}

struct main_out {
  @builtin(position)
  gl_Position : vec4<f32>,
  @location(8)
  vs_TEXCOORD8_1 : f32,
  @location(0)
  vs_TEXCOORD0_1 : vec4<f32>,
  @location(1)
  vs_TEXCOORD1_1 : vec2<f32>,
  @location(2)
  vs_TEXCOORD5_1 : vec2<f32>,
  @location(3)
  vs_TEXCOORD2_1 : vec4<f32>,
  @location(4)
  vs_TEXCOORD3_1 : vec4<f32>,
  @location(5)
  vs_TEXCOORD4_1 : vec4<f32>,
  @location(6)
  vs_COLOR0_1 : vec4<f32>,
  @location(7)
  vs_TEXCOORD6_1 : vec3<f32>,
  @location(9)
  vs_TEXCOORD7_1 : vec3<f32>,
}

@vertex
fn main(@location(0) in_POSITION0_param : vec4<f32>, @location(4) in_TEXCOORD1_param : vec4<f32>, @location(3) in_TEXCOORD0_param : vec4<f32>, @location(2) in_NORMAL0_param : vec3<f32>, @location(1) in_TANGENT0_param : vec4<f32>, @location(5) in_COLOR0_param : vec4<f32>) -> main_out {
  in_POSITION0 = in_POSITION0_param;
  in_TEXCOORD1 = in_TEXCOORD1_param;
  in_TEXCOORD0 = in_TEXCOORD0_param;
  in_NORMAL0 = in_NORMAL0_param;
  in_TANGENT0 = in_TANGENT0_param;
  in_COLOR0 = in_COLOR0_param;
  main_1();
  return main_out(gl_Position, vs_TEXCOORD8, vs_TEXCOORD0, vs_TEXCOORD1, vs_TEXCOORD5, vs_TEXCOORD2, vs_TEXCOORD3, vs_TEXCOORD4, vs_COLOR0, vs_TEXCOORD6, vs_TEXCOORD7);
}


