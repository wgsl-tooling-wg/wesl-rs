diagnostic(off, derivative_uniformity);

struct strided_arr {
  @size(16)
  el : f32,
}

alias Arr = array<strided_arr, 4u>;

alias Arr_1 = array<strided_arr, 4u>;

alias Arr_2 = array<strided_arr, 4u>;

alias Arr_3 = array<vec4<f32>, 4u>;

struct PGlobals {
  /* @offset(0) */
  x_ManualTex2SRGB : i32,
  /* @offset(4) */
  x_SrcBlend : i32,
  @size(8)
  padding : u32,
  /* @offset(16) */
  x_CornerRadiuses : Arr_2,
  /* @offset(80) */
  x_BorderWidths : Arr_2,
  /* @offset(144) */
  x_Rect : Arr_2,
  /* @offset(208) */
  x_BorderColors : Arr_3,
}

var<private> u_xlat0 : vec4<f32>;

var<private> vs_TEXCOORD2 : vec4<f32>;

var<private> u_xlat1 : vec4<f32>;

@group(0) @binding(0) var x_MainTex : texture_2d<f32>;

@group(0) @binding(2) var sampler_MainTex : sampler;

var<private> vs_TEXCOORD0 : vec2<f32>;

var<private> u_xlat9 : vec3<f32>;

@group(1) @binding(0) var<uniform> x_75 : PGlobals;

var<private> u_xlat18 : f32;

var<private> u_xlatb9 : vec2<bool>;

var<private> u_xlati2 : vec3<i32>;

var<private> u_xlati27 : i32;

var<private> u_xlat3 : vec4<f32>;

var<private> u_xlat4 : vec4<f32>;

var<private> u_xlat5 : vec4<f32>;

var<private> u_xlat6 : vec4<f32>;

var<private> u_xlatb11 : vec3<bool>;

var<private> u_xlatb3 : vec2<bool>;

var<private> u_xlatb4 : vec2<bool>;

var<private> u_xlat11 : f32;

var<private> u_xlatb12 : bool;

var<private> u_xlatb21 : bool;

var<private> u_xlati30 : i32;

var<private> u_xlati13 : i32;

var<private> u_xlatb29 : bool;

var<private> u_xlatb18 : bool;

var<private> u_xlati29 : i32;

var<private> u_xlati18 : i32;

var<private> u_xlat7 : vec4<f32>;

var<private> u_xlat12 : vec3<f32>;

var<private> u_xlat30 : f32;

var<private> u_xlat13 : vec2<f32>;

var<private> u_xlatb20 : bool;

var<private> u_xlat8 : vec4<f32>;

var<private> vs_COLOR0 : vec4<f32>;

var<private> u_xlatb2 : vec4<bool>;

var<private> u_xlat28 : f32;

var<private> u_xlat2 : f32;

var<private> u_xlat27 : f32;

var<private> u_xlat20 : f32;

var<private> u_xlatb1 : bool;

var<private> u_xlatb27 : bool;

@group(0) @binding(1) var x_GUIClipTexture : texture_2d<f32>;

@group(0) @binding(3) var sampler_GUIClipTexture : sampler;

var<private> vs_TEXCOORD1 : vec2<f32>;

var<private> SV_Target0 : vec4<f32>;

var<private> gl_FragCoord : vec4<f32>;

fn main_1() {
  var x_83 : vec3<f32>;
  var x_155 : i32;
  var x_269 : vec4<f32>;
  var x_298 : vec2<bool>;
  var x_323 : vec2<bool>;
  var x_589 : i32;
  var x_604 : i32;
  var x_748 : f32;
  var x_757 : f32;
  var x_771 : f32;
  var x_782 : f32;
  var x_845 : vec3<f32>;
  var u_xlat_precise_vec4 : vec4<f32>;
  var u_xlat_precise_ivec4 : vec4<i32>;
  var u_xlat_precise_bvec4 : vec4<bool>;
  var u_xlat_precise_uvec4 : vec4<u32>;
  let x_16 : f32 = vs_TEXCOORD2.x;
  u_xlat0.x = dpdxCoarse(x_16);
  let x_22 : f32 = u_xlat0.x;
  u_xlat0.x = (1.0f / abs(x_22));
  let x_40 : vec2<f32> = vs_TEXCOORD0;
  let x_41 : vec4<f32> = textureSample(x_MainTex, sampler_MainTex, x_40);
  u_xlat1 = x_41;
  let x_45 : vec4<f32> = u_xlat1;
  u_xlat9 = max(vec3<f32>(x_45.x, x_45.y, x_45.z), vec3<f32>(0.0f, 0.0f, 0.0f));
  let x_50 : vec3<f32> = u_xlat9;
  u_xlat9 = log2(x_50);
  let x_52 : vec3<f32> = u_xlat9;
  u_xlat9 = (x_52 * vec3<f32>(0.4166666567325592041f, 0.4166666567325592041f, 0.4166666567325592041f));
  let x_56 : vec3<f32> = u_xlat9;
  u_xlat9 = exp2(x_56);
  let x_58 : vec3<f32> = u_xlat9;
  u_xlat9 = ((x_58 * vec3<f32>(1.05499994754791259766f, 1.05499994754791259766f, 1.05499994754791259766f)) + vec3<f32>(-0.05499999970197677612f, -0.05499999970197677612f, -0.05499999970197677612f));
  let x_65 : vec3<f32> = u_xlat9;
  u_xlat9 = max(x_65, vec3<f32>(0.0f, 0.0f, 0.0f));
  let x_79 : i32 = x_75.x_ManualTex2SRGB;
  if ((x_79 != 0i)) {
    let x_86 : vec3<f32> = u_xlat9;
    x_83 = x_86;
  } else {
    let x_88 : vec4<f32> = u_xlat1;
    x_83 = vec3<f32>(x_88.x, x_88.y, x_88.z);
  }
  let x_90 : vec3<f32> = x_83;
  let x_91 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_90.x, x_90.y, x_90.z, x_91.w);
  let x_94 : f32 = vs_TEXCOORD2.x;
  let x_98 : f32 = x_75.x_Rect[0i].el;
  u_xlat9.x = (x_94 + -(x_98));
  let x_104 : f32 = x_75.x_Rect[2i].el;
  let x_109 : f32 = u_xlat9.x;
  u_xlat9.x = ((-(x_104) * 0.5f) + x_109);
  let x_115 : f32 = vs_TEXCOORD2.y;
  let x_118 : f32 = x_75.x_Rect[1i].el;
  u_xlat18 = (x_115 + -(x_118));
  let x_123 : f32 = x_75.x_Rect[3i].el;
  let x_126 : f32 = u_xlat18;
  u_xlat9.y = ((-(x_123) * 0.5f) + x_126);
  let x_133 : vec3<f32> = u_xlat9;
  let x_136 : vec4<bool> = (vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f) >= vec4<f32>(x_133.x, x_133.y, x_133.x, x_133.x));
  u_xlatb9 = vec2<bool>(x_136.x, x_136.y);
  let x_143 : bool = u_xlatb9.y;
  u_xlati2 = select(vec3<i32>(3i, 2i, 8i), vec3<i32>(0i, 1i, 2i), vec3<bool>(x_143, x_143, x_143));
  let x_153 : bool = u_xlatb9.x;
  if (x_153) {
    let x_159 : i32 = u_xlati2.x;
    x_155 = x_159;
  } else {
    let x_162 : i32 = u_xlati2.y;
    x_155 = x_162;
  }
  let x_163 : i32 = x_155;
  u_xlati27 = x_163;
  let x_165 : bool = u_xlatb9.x;
  u_xlati2.x = select(4i, 1i, x_165);
  let x_170 : f32 = x_75.x_Rect[0i].el;
  let x_171 : i32 = u_xlati27;
  let x_173 : f32 = x_75.x_CornerRadiuses[x_171].el;
  u_xlat3.y = (x_170 + x_173);
  let x_178 : f32 = x_75.x_Rect[1i].el;
  let x_179 : i32 = u_xlati27;
  let x_181 : f32 = x_75.x_CornerRadiuses[x_179].el;
  u_xlat4.y = (x_178 + x_181);
  let x_185 : f32 = x_75.x_BorderWidths[0i].el;
  let x_187 : f32 = x_75.x_Rect[0i].el;
  u_xlat3.w = (x_185 + x_187);
  let x_192 : f32 = x_75.x_BorderWidths[1i].el;
  let x_194 : f32 = x_75.x_Rect[1i].el;
  u_xlat4.w = (x_192 + x_194);
  let x_199 : f32 = x_75.x_Rect[0i].el;
  let x_201 : f32 = x_75.x_Rect[2i].el;
  u_xlat5.x = (x_199 + x_201);
  let x_205 : f32 = u_xlat5.x;
  let x_206 : i32 = u_xlati27;
  let x_208 : f32 = x_75.x_CornerRadiuses[x_206].el;
  u_xlat5.y = (x_205 + -(x_208));
  let x_213 : f32 = u_xlat5.x;
  let x_215 : f32 = x_75.x_BorderWidths[2i].el;
  u_xlat5.w = (x_213 + -(x_215));
  let x_220 : f32 = x_75.x_Rect[0i].el;
  u_xlat3.x = x_220;
  let x_223 : f32 = x_75.x_BorderWidths[0i].el;
  u_xlat3.z = x_223;
  let x_227 : f32 = x_75.x_BorderWidths[2i].el;
  u_xlat5.z = x_227;
  let x_230 : bool = u_xlatb9.x;
  let x_231 : vec4<f32> = u_xlat3;
  let x_232 : vec4<f32> = u_xlat5;
  u_xlat5 = select(x_232, x_231, vec4<bool>(x_230, x_230, x_230, x_230));
  let x_237 : f32 = x_75.x_Rect[1i].el;
  let x_239 : f32 = x_75.x_Rect[3i].el;
  u_xlat6.x = (x_237 + x_239);
  let x_243 : f32 = u_xlat6.x;
  let x_244 : i32 = u_xlati27;
  let x_246 : f32 = x_75.x_CornerRadiuses[x_244].el;
  u_xlat6.y = (x_243 + -(x_246));
  let x_251 : f32 = u_xlat6.x;
  let x_253 : f32 = x_75.x_BorderWidths[3i].el;
  u_xlat6.w = (x_251 + -(x_253));
  let x_258 : f32 = x_75.x_Rect[1i].el;
  u_xlat4.x = x_258;
  let x_261 : f32 = x_75.x_BorderWidths[1i].el;
  u_xlat4.z = x_261;
  let x_264 : f32 = x_75.x_BorderWidths[3i].el;
  u_xlat6.z = x_264;
  let x_267 : bool = u_xlatb9.y;
  if (x_267) {
    let x_272 : vec4<f32> = u_xlat4;
    x_269 = vec4<f32>(x_272.y, x_272.x, x_272.z, x_272.w);
  } else {
    let x_275 : vec4<f32> = u_xlat6;
    x_269 = vec4<f32>(x_275.y, x_275.x, x_275.z, x_275.w);
  }
  let x_277 : vec4<f32> = x_269;
  u_xlat6 = x_277;
  let x_280 : vec4<f32> = u_xlat5;
  let x_282 : vec4<f32> = vs_TEXCOORD2;
  let x_284 : vec4<bool> = (vec4<f32>(x_280.y, x_280.y, x_280.w, x_280.w) >= vec4<f32>(x_282.x, x_282.x, x_282.x, x_282.x));
  let x_285 : vec2<bool> = vec2<bool>(x_284.x, x_284.z);
  let x_286 : vec3<bool> = u_xlatb11;
  u_xlatb11 = vec3<bool>(x_285.x, x_286.y, x_285.y);
  let x_289 : vec4<f32> = vs_TEXCOORD2;
  let x_291 : vec4<f32> = u_xlat5;
  let x_293 : vec4<bool> = (vec4<f32>(x_289.x, x_289.x, x_289.x, x_289.x) >= vec4<f32>(x_291.y, x_291.w, x_291.y, x_291.y));
  u_xlatb3 = vec2<bool>(x_293.x, x_293.y);
  let x_296 : bool = u_xlatb9.x;
  if (x_296) {
    let x_301 : vec3<bool> = u_xlatb11;
    x_298 = vec2<bool>(x_301.x, x_301.z);
  } else {
    let x_304 : vec2<bool> = u_xlatb3;
    x_298 = x_304;
  }
  let x_305 : vec2<bool> = x_298;
  let x_306 : vec3<bool> = u_xlatb11;
  u_xlatb11 = vec3<bool>(x_305.x, x_306.y, x_305.y);
  let x_308 : vec4<f32> = u_xlat6;
  let x_310 : vec4<f32> = vs_TEXCOORD2;
  let x_312 : vec4<bool> = (vec4<f32>(x_308.x, x_308.w, x_308.x, x_308.x) >= vec4<f32>(x_310.y, x_310.y, x_310.y, x_310.y));
  u_xlatb3 = vec2<bool>(x_312.x, x_312.y);
  let x_315 : vec4<f32> = vs_TEXCOORD2;
  let x_317 : vec4<f32> = u_xlat6;
  let x_319 : vec4<bool> = (vec4<f32>(x_315.y, x_315.y, x_315.y, x_315.y) >= vec4<f32>(x_317.x, x_317.w, x_317.x, x_317.x));
  u_xlatb4 = vec2<bool>(x_319.x, x_319.y);
  let x_322 : bool = u_xlatb9.y;
  if (x_322) {
    let x_326 : vec2<bool> = u_xlatb3;
    x_323 = x_326;
  } else {
    let x_328 : vec2<bool> = u_xlatb4;
    x_323 = x_328;
  }
  let x_329 : vec2<bool> = x_323;
  u_xlatb9 = x_329;
  let x_331 : bool = u_xlatb9.x;
  let x_333 : bool = u_xlatb11.x;
  u_xlatb9.x = (x_331 & x_333);
  let x_338 : f32 = x_75.x_BorderWidths[0i].el;
  let x_340 : f32 = x_75.x_BorderWidths[2i].el;
  u_xlat11 = (x_338 + x_340);
  let x_342 : f32 = u_xlat11;
  let x_345 : f32 = x_75.x_Rect[2i].el;
  u_xlat11 = (-(x_342) + x_345);
  let x_348 : f32 = x_75.x_BorderWidths[1i].el;
  let x_350 : f32 = x_75.x_BorderWidths[3i].el;
  u_xlat3.x = (x_348 + x_350);
  let x_354 : f32 = u_xlat3.x;
  let x_357 : f32 = x_75.x_Rect[3i].el;
  u_xlat3.x = (-(x_354) + x_357);
  let x_362 : f32 = vs_TEXCOORD2.x;
  let x_364 : f32 = u_xlat3.w;
  u_xlatb12 = (x_362 >= x_364);
  let x_366 : f32 = u_xlat11;
  let x_368 : f32 = u_xlat3.w;
  u_xlat11 = (x_366 + x_368);
  let x_370 : f32 = u_xlat11;
  let x_372 : f32 = vs_TEXCOORD2.x;
  u_xlatb11.x = (x_370 >= x_372);
  let x_376 : bool = u_xlatb11.x;
  let x_377 : bool = u_xlatb12;
  u_xlatb11.x = (x_376 & x_377);
  let x_381 : f32 = vs_TEXCOORD2.y;
  let x_383 : f32 = u_xlat4.w;
  u_xlatb12 = (x_381 >= x_383);
  let x_386 : bool = u_xlatb11.x;
  let x_387 : bool = u_xlatb12;
  u_xlatb11.x = (x_386 & x_387);
  let x_391 : f32 = u_xlat3.x;
  let x_393 : f32 = u_xlat4.w;
  u_xlat3.x = (x_391 + x_393);
  let x_397 : f32 = u_xlat3.x;
  let x_399 : f32 = vs_TEXCOORD2.y;
  u_xlatb3.x = (x_397 >= x_399);
  let x_403 : bool = u_xlatb11.x;
  let x_405 : bool = u_xlatb3.x;
  u_xlatb11.x = (x_403 & x_405);
  let x_408 : i32 = u_xlati27;
  let x_412 : vec4<bool> = (vec4<i32>(x_408, x_408, x_408, x_408) == vec4<i32>(0i, 2i, 0i, 0i));
  u_xlatb3 = vec2<bool>(x_412.x, x_412.y);
  let x_415 : bool = u_xlatb3.y;
  let x_417 : bool = u_xlatb3.x;
  u_xlatb3.x = (x_415 | x_417);
  let x_420 : i32 = u_xlati27;
  let x_422 : f32 = x_75.x_CornerRadiuses[x_420].el;
  u_xlatb12 = (0.0f < x_422);
  let x_425 : bool = u_xlatb9.x;
  let x_426 : bool = u_xlatb12;
  u_xlatb12 = (x_425 & x_426);
  let x_430 : f32 = u_xlat5.z;
  u_xlatb21 = (0.0f < x_430);
  let x_433 : bool = u_xlatb21;
  u_xlati30 = select(0i, 1i, x_433);
  let x_436 : f32 = u_xlat6.z;
  u_xlatb4.x = (0.0f < x_436);
  let x_441 : bool = u_xlatb4.x;
  u_xlati13 = select(0i, 1i, x_441);
  let x_444 : i32 = u_xlati2.z;
  let x_445 : i32 = u_xlati13;
  u_xlati13 = (x_444 * x_445);
  let x_447 : i32 = u_xlati30;
  let x_449 : i32 = u_xlati2.x;
  let x_451 : i32 = u_xlati13;
  u_xlati30 = ((x_447 * x_449) + x_451);
  let x_455 : bool = u_xlatb11.z;
  let x_456 : bool = u_xlatb21;
  u_xlatb29 = (x_455 & x_456);
  let x_460 : bool = u_xlatb9.y;
  let x_462 : bool = u_xlatb4.x;
  u_xlatb18 = (x_460 & x_462);
  let x_465 : bool = u_xlatb29;
  u_xlati29 = select(0i, 1i, x_465);
  let x_468 : bool = u_xlatb18;
  u_xlati18 = select(0i, 1i, x_468);
  let x_471 : i32 = u_xlati2.z;
  let x_472 : i32 = u_xlati18;
  u_xlati18 = (x_471 * x_472);
  let x_474 : i32 = u_xlati29;
  let x_476 : i32 = u_xlati2.x;
  let x_478 : i32 = u_xlati18;
  u_xlati18 = ((x_474 * x_476) + x_478);
  let x_481 : bool = u_xlatb11.x;
  let x_482 : i32 = u_xlati18;
  u_xlati18 = select(x_482, 0i, x_481);
  let x_484 : bool = u_xlatb12;
  let x_485 : i32 = u_xlati30;
  let x_486 : i32 = u_xlati18;
  u_xlati18 = select(x_486, x_485, x_484);
  let x_489 : i32 = u_xlati2.z;
  let x_491 : i32 = u_xlati2.x;
  u_xlati29 = (x_489 + x_491);
  let x_494 : bool = u_xlatb3.x;
  u_xlat3.x = select(-1.0f, 1.0f, x_494);
  let x_499 : vec4<f32> = u_xlat5;
  let x_500 : vec2<f32> = vec2<f32>(x_499.w, x_499.y);
  let x_501 : vec4<f32> = u_xlat7;
  u_xlat7 = vec4<f32>(x_500.x, x_501.y, x_500.y, x_501.w);
  let x_503 : vec4<f32> = u_xlat6;
  let x_504 : vec2<f32> = vec2<f32>(x_503.w, x_503.x);
  let x_505 : vec4<f32> = u_xlat7;
  u_xlat7 = vec4<f32>(x_505.x, x_504.x, x_505.z, x_504.y);
  let x_508 : f32 = u_xlat5.x;
  u_xlat6.x = x_508;
  let x_511 : vec4<f32> = u_xlat7;
  let x_514 : vec4<f32> = u_xlat6;
  let x_516 : vec2<f32> = (-(vec2<f32>(x_511.y, x_511.x)) + vec2<f32>(x_514.y, x_514.x));
  let x_517 : vec3<f32> = u_xlat12;
  u_xlat12 = vec3<f32>(x_516.x, x_517.y, x_516.y);
  let x_519 : vec3<f32> = u_xlat12;
  let x_521 : vec4<f32> = u_xlat3;
  let x_523 : vec2<f32> = (vec2<f32>(x_519.x, x_519.z) * vec2<f32>(x_521.x, x_521.x));
  let x_524 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_523.x, x_523.y, x_524.z, x_524.w);
  let x_527 : vec4<f32> = u_xlat3;
  let x_529 : vec4<f32> = u_xlat3;
  u_xlat30 = dot(vec2<f32>(x_527.x, x_527.y), vec2<f32>(x_529.x, x_529.y));
  let x_532 : f32 = u_xlat30;
  u_xlat30 = inverseSqrt(x_532);
  let x_534 : f32 = u_xlat30;
  let x_536 : vec4<f32> = u_xlat3;
  let x_538 : vec2<f32> = (vec2<f32>(x_534, x_534) * vec2<f32>(x_536.x, x_536.y));
  let x_539 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_538.x, x_538.y, x_539.z, x_539.w);
  let x_541 : vec4<f32> = u_xlat3;
  let x_546 : vec4<f32> = u_xlat6;
  let x_548 : vec2<f32> = ((vec2<f32>(x_541.x, x_541.y) * vec2<f32>(100.0f, 100.0f)) + vec2<f32>(x_546.y, x_546.x));
  let x_549 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_548.x, x_548.y, x_549.z, x_549.w);
  let x_553 : vec4<f32> = u_xlat3;
  let x_556 : vec4<f32> = u_xlat6;
  u_xlat13 = (-(vec2<f32>(x_553.y, x_553.x)) + vec2<f32>(x_556.x, x_556.y));
  let x_559 : vec4<f32> = u_xlat3;
  let x_562 : vec4<f32> = vs_TEXCOORD2;
  let x_564 : vec2<f32> = (-(vec2<f32>(x_559.x, x_559.y)) + vec2<f32>(x_562.y, x_562.x));
  let x_565 : vec4<f32> = u_xlat3;
  u_xlat3 = vec4<f32>(x_564.x, x_564.y, x_565.z, x_565.w);
  let x_568 : f32 = u_xlat3.y;
  let x_570 : f32 = u_xlat13.y;
  u_xlat12.x = (x_568 * x_570);
  let x_574 : f32 = u_xlat13.x;
  let x_576 : f32 = u_xlat3.x;
  let x_579 : f32 = u_xlat12.x;
  u_xlat3.x = ((x_574 * x_576) + -(x_579));
  let x_584 : f32 = u_xlat3.x;
  u_xlatb3.x = (x_584 >= 0.0f);
  let x_588 : bool = u_xlatb3.x;
  if (x_588) {
    let x_593 : i32 = u_xlati2.x;
    x_589 = x_593;
  } else {
    let x_596 : i32 = u_xlati2.z;
    x_589 = x_596;
  }
  let x_597 : i32 = x_589;
  u_xlati2.x = x_597;
  let x_600 : i32 = u_xlati18;
  let x_601 : i32 = u_xlati29;
  u_xlatb20 = (x_600 != x_601);
  let x_603 : bool = u_xlatb20;
  if (x_603) {
    let x_607 : i32 = u_xlati18;
    x_604 = x_607;
  } else {
    let x_610 : i32 = u_xlati2.x;
    x_604 = x_610;
  }
  let x_611 : i32 = x_604;
  u_xlati18 = x_611;
  let x_612 : i32 = u_xlati18;
  if ((x_612 == 0i)) {
    let x_617 : vec4<f32> = u_xlat1;
    let x_619 : vec4<f32> = vs_COLOR0;
    u_xlat8 = (x_617 * x_619);
  } else {
    let x_624 : i32 = u_xlati18;
    let x_627 : vec4<bool> = (vec4<i32>(x_624, x_624, x_624, x_624) == vec4<i32>(2i, 0i, 4i, 8i));
    let x_628 : vec3<bool> = vec3<bool>(x_627.x, x_627.z, x_627.w);
    let x_629 : vec4<bool> = u_xlatb2;
    u_xlatb2 = vec4<bool>(x_628.x, x_629.y, x_628.y, x_628.z);
    let x_632 : bool = u_xlatb2.w;
    u_xlati18 = select(0i, 3i, x_632);
    let x_635 : bool = u_xlatb2.z;
    let x_636 : i32 = u_xlati18;
    u_xlati18 = select(x_636, 2i, x_635);
    let x_639 : bool = u_xlatb2.x;
    let x_640 : i32 = u_xlati18;
    u_xlati18 = select(x_640, 1i, x_639);
    let x_642 : vec4<f32> = u_xlat1;
    let x_644 : i32 = u_xlati18;
    let x_647 : vec4<f32> = x_75.x_BorderColors[x_644];
    u_xlat8 = (x_642 * x_647);
  }
  let x_649 : bool = u_xlatb21;
  let x_651 : bool = u_xlatb4.x;
  u_xlatb18 = (x_649 | x_651);
  let x_653 : vec4<f32> = u_xlat7;
  let x_656 : vec4<f32> = vs_TEXCOORD2;
  let x_658 : vec2<f32> = (-(vec2<f32>(x_653.z, x_653.w)) + vec2<f32>(x_656.x, x_656.y));
  let x_659 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_658.x, x_658.y, x_659.z, x_659.w);
  let x_662 : vec4<f32> = u_xlat1;
  let x_664 : vec4<f32> = u_xlat1;
  u_xlat28 = dot(vec2<f32>(x_662.x, x_662.y), vec2<f32>(x_664.x, x_664.y));
  let x_667 : f32 = u_xlat28;
  u_xlat28 = sqrt(x_667);
  let x_669 : f32 = u_xlat28;
  let x_670 : i32 = u_xlati27;
  let x_672 : f32 = x_75.x_CornerRadiuses[x_670].el;
  u_xlat28 = (x_669 + -(x_672));
  let x_675 : f32 = u_xlat28;
  let x_677 : f32 = u_xlat0.x;
  u_xlat28 = ((x_675 * x_677) + 0.5f);
  let x_680 : f32 = u_xlat28;
  u_xlat28 = clamp(x_680, 0.0f, 1.0f);
  let x_682 : bool = u_xlatb18;
  let x_683 : f32 = u_xlat28;
  u_xlat28 = select(0.0f, x_683, x_682);
  let x_687 : f32 = u_xlat5.z;
  let x_689 : i32 = u_xlati27;
  let x_691 : f32 = x_75.x_CornerRadiuses[x_689].el;
  u_xlat2 = (-(x_687) + x_691);
  let x_695 : f32 = u_xlat6.z;
  let x_697 : i32 = u_xlati27;
  let x_699 : f32 = x_75.x_CornerRadiuses[x_697].el;
  u_xlat27 = (-(x_695) + x_699);
  let x_702 : f32 = u_xlat2;
  let x_703 : f32 = u_xlat27;
  u_xlat20 = (x_702 / x_703);
  let x_706 : f32 = u_xlat1.y;
  let x_707 : f32 = u_xlat20;
  u_xlat1.z = (x_706 * x_707);
  let x_710 : vec4<f32> = u_xlat1;
  let x_712 : vec4<f32> = u_xlat1;
  u_xlat1.x = dot(vec2<f32>(x_710.x, x_710.z), vec2<f32>(x_712.x, x_712.z));
  let x_717 : f32 = u_xlat1.x;
  u_xlat1.x = sqrt(x_717);
  let x_720 : f32 = u_xlat2;
  let x_723 : f32 = u_xlat1.x;
  u_xlat1.x = (-(x_720) + x_723);
  let x_727 : f32 = u_xlat1.x;
  let x_729 : f32 = u_xlat0.x;
  u_xlat0.x = ((x_727 * x_729) + 0.5f);
  let x_734 : f32 = u_xlat0.x;
  u_xlat0.x = clamp(x_734, 0.0f, 1.0f);
  let x_738 : f32 = u_xlat2;
  u_xlatb1 = (0.0f < x_738);
  let x_741 : f32 = u_xlat27;
  u_xlatb27 = (0.0f < x_741);
  let x_743 : bool = u_xlatb27;
  let x_744 : bool = u_xlatb1;
  u_xlatb27 = (x_743 & x_744);
  let x_746 : bool = u_xlatb27;
  if (x_746) {
    let x_752 : f32 = u_xlat0.x;
    x_748 = x_752;
  } else {
    x_748 = 1.0f;
  }
  let x_754 : f32 = x_748;
  u_xlat0.x = x_754;
  let x_756 : bool = u_xlatb18;
  if (x_756) {
    let x_761 : f32 = u_xlat0.x;
    x_757 = x_761;
  } else {
    x_757 = 0.0f;
  }
  let x_763 : f32 = x_757;
  u_xlat0.x = x_763;
  let x_765 : f32 = u_xlat28;
  u_xlatb18 = (x_765 == 0.0f);
  let x_767 : f32 = u_xlat28;
  u_xlat27 = (-(x_767) + 1.0f);
  let x_770 : bool = u_xlatb18;
  if (x_770) {
    let x_775 : f32 = u_xlat0.x;
    x_771 = x_775;
  } else {
    let x_777 : f32 = u_xlat27;
    x_771 = x_777;
  }
  let x_778 : f32 = x_771;
  u_xlat0.x = x_778;
  let x_781 : bool = u_xlatb9.x;
  if (x_781) {
    let x_786 : f32 = u_xlat0.x;
    x_782 = x_786;
  } else {
    x_782 = 1.0f;
  }
  let x_788 : f32 = x_782;
  u_xlat0.x = x_788;
  let x_791 : f32 = u_xlat0.x;
  let x_793 : f32 = u_xlat8.w;
  u_xlat18 = (x_791 * x_793);
  let x_796 : bool = u_xlatb11.x;
  u_xlat27 = select(1.0f, 0.0f, x_796);
  let x_799 : bool = u_xlatb9.x;
  let x_800 : f32 = u_xlat27;
  u_xlat9.x = select(x_800, 1.0f, x_799);
  let x_804 : f32 = u_xlat9.x;
  let x_805 : f32 = u_xlat18;
  u_xlat18 = (x_804 * x_805);
  let x_813 : vec2<f32> = vs_TEXCOORD1;
  let x_814 : vec4<f32> = textureSample(x_GUIClipTexture, sampler_GUIClipTexture, x_813);
  u_xlat27 = x_814.w;
  let x_818 : f32 = u_xlat27;
  let x_819 : f32 = u_xlat18;
  SV_Target0.w = (x_818 * x_819);
  let x_824 : i32 = x_75.x_SrcBlend;
  u_xlatb18 = (x_824 != 5i);
  let x_827 : f32 = u_xlat9.x;
  let x_829 : f32 = u_xlat0.x;
  u_xlat0.x = (x_827 * x_829);
  let x_832 : f32 = u_xlat27;
  let x_834 : f32 = u_xlat0.x;
  u_xlat0.x = (x_832 * x_834);
  let x_837 : vec4<f32> = u_xlat0;
  let x_839 : vec4<f32> = u_xlat8;
  let x_841 : vec3<f32> = (vec3<f32>(x_837.x, x_837.x, x_837.x) * vec3<f32>(x_839.x, x_839.y, x_839.z));
  let x_842 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_841.x, x_841.y, x_842.z, x_841.z);
  let x_844 : bool = u_xlatb18;
  if (x_844) {
    let x_848 : vec4<f32> = u_xlat0;
    x_845 = vec3<f32>(x_848.x, x_848.y, x_848.w);
  } else {
    let x_851 : vec4<f32> = u_xlat8;
    x_845 = vec3<f32>(x_851.x, x_851.y, x_851.z);
  }
  let x_853 : vec3<f32> = x_845;
  let x_854 : vec4<f32> = SV_Target0;
  SV_Target0 = vec4<f32>(x_853.x, x_853.y, x_853.z, x_854.w);
  return;
}

struct main_out {
  @location(0)
  SV_Target0_1 : vec4<f32>,
}

@fragment
fn main(@location(3) vs_TEXCOORD2_param : vec4<f32>, @location(1) vs_TEXCOORD0_param : vec2<f32>, @location(0) vs_COLOR0_param : vec4<f32>, @location(2) vs_TEXCOORD1_param : vec2<f32>, @builtin(position) gl_FragCoord_param : vec4<f32>) -> main_out {
  vs_TEXCOORD2 = vs_TEXCOORD2_param;
  vs_TEXCOORD0 = vs_TEXCOORD0_param;
  vs_COLOR0 = vs_COLOR0_param;
  vs_TEXCOORD1 = vs_TEXCOORD1_param;
  gl_FragCoord = gl_FragCoord_param;
  main_1();
  return main_out(SV_Target0);
}


