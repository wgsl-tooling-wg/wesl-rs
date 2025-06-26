diagnostic(off, derivative_uniformity);

alias Arr = array<vec4<f32>, 16u>;

struct VGlobals {
  /* @offset(0) */
  unity_ObjectToWorld : mat4x4<f32>,
  /* @offset(64) */
  unity_MatrixVP : mat4x4<f32>,
  /* @offset(128) */
  x_ShaderInfoTex_TexelSize : vec4<f32>,
  /* @offset(144) */
  x_TextureInfo : Arr,
}

var<private> u_xlat0 : vec4<f32>;

var<private> in_TEXCOORD2 : vec4<f32>;

var<private> u_xlat1 : vec4<f32>;

var<private> u_xlatb1 : vec4<bool>;

var<private> u_xlat2 : vec4<f32>;

var<private> u_xlat10 : vec3<f32>;

var<private> in_TEXCOORD1 : vec4<f32>;

var<private> u_xlat3 : vec4<f32>;

@group(1) @binding(1) var<uniform> x_131 : VGlobals;

var<private> u_xlat4 : vec4<f32>;

@group(0) @binding(8) var x_ShaderInfoTex : texture_2d<f32>;

@group(0) @binding(11) var sampler_ShaderInfoTex : sampler;

var<private> u_xlat5 : vec4<f32>;

var<private> in_TEXCOORD3 : vec4<f32>;

var<private> u_xlatb6 : vec4<bool>;

var<private> u_xlat7 : vec4<f32>;

var<private> u_xlat20 : vec2<f32>;

var<private> in_POSITION0 : vec4<f32>;

var<private> u_xlati1 : i32;

var<private> u_xlatb18 : bool;

var<private> u_xlat18 : f32;

var<private> u_xlatu18 : u32;

var<private> in_TEXCOORD7 : f32;

var<private> vs_TEXCOORD1 : vec4<f32>;

var<private> in_TEXCOORD5 : vec4<f32>;

var<private> u_xlatb2 : vec2<bool>;

var<private> u_xlat9 : vec3<f32>;

var<private> u_xlatu9 : u32;

var<private> vs_TEXCOORD0 : vec4<f32>;

var<private> in_TEXCOORD0 : vec2<f32>;

var<private> in_TEXCOORD4 : vec4<f32>;

var<private> u_xlat11 : vec3<f32>;

var<private> u_xlat8 : vec3<f32>;

var<private> vs_COLOR0 : vec4<f32>;

var<private> in_COLOR0 : vec4<f32>;

var<private> u_xlat16 : f32;

var<private> vs_TEXCOORD3 : vec2<f32>;

var<private> vs_TEXCOORD4 : vec4<f32>;

var<private> in_TEXCOORD6 : vec4<f32>;

var<private> gl_Position : vec4<f32>;

fn main_1() {
  var x_377 : f32;
  var x_440 : f32;
  var x_649 : f32;
  var x_661 : f32;
  var x_673 : f32;
  var x_728 : f32;
  var x_742 : vec2<f32>;
  var x_769 : f32;
  var u_xlat_precise_vec4 : vec4<f32>;
  var u_xlat_precise_ivec4 : vec4<i32>;
  var u_xlat_precise_bvec4 : vec4<bool>;
  var u_xlat_precise_uvec4 : vec4<u32>;
  let x_12 : vec4<f32> = in_TEXCOORD2;
  u_xlat0 = (vec4<f32>(x_12.x, x_12.z, x_12.w, x_12.y) * vec4<f32>(255.0f, 255.0f, 255.0f, 255.0f));
  let x_17 : vec4<f32> = u_xlat0;
  u_xlat0 = round(x_17);
  let x_20 : vec4<f32> = u_xlat0;
  u_xlat1 = (x_20 * vec4<f32>(32.0f, 32.0f, 32.0f, 32.0f));
  let x_28 : vec4<f32> = u_xlat1;
  let x_29 : vec4<f32> = u_xlat1;
  u_xlatb1 = (x_28 >= -(x_29));
  let x_37 : bool = u_xlatb1.x;
  u_xlat2.x = select(-32.0f, 32.0f, x_37);
  let x_43 : bool = u_xlatb1.x;
  u_xlat2.y = select(-0.03125f, 0.03125f, x_43);
  let x_50 : bool = u_xlatb1.y;
  u_xlat2.z = select(-32.0f, 32.0f, x_50);
  let x_55 : bool = u_xlatb1.y;
  u_xlat2.w = select(-0.03125f, 0.03125f, x_55);
  let x_60 : vec4<f32> = u_xlat0;
  let x_62 : vec4<f32> = u_xlat2;
  let x_64 : vec2<f32> = (vec2<f32>(x_60.x, x_60.y) * vec2<f32>(x_62.y, x_62.w));
  let x_65 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_64.x, x_64.y, x_65.z, x_65.w);
  let x_67 : vec4<f32> = u_xlat1;
  let x_69 : vec2<f32> = fract(vec2<f32>(x_67.x, x_67.y));
  let x_70 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_69.x, x_69.y, x_70.z, x_70.w);
  let x_75 : vec4<f32> = u_xlat1;
  let x_77 : vec4<f32> = u_xlat2;
  let x_79 : vec2<f32> = (vec2<f32>(x_75.x, x_75.y) * vec2<f32>(x_77.x, x_77.z));
  let x_80 : vec3<f32> = u_xlat10;
  u_xlat10 = vec3<f32>(x_79.x, x_80.y, x_79.y);
  let x_82 : vec4<f32> = u_xlat2;
  let x_85 : vec4<f32> = u_xlat1;
  let x_88 : vec4<f32> = u_xlat0;
  let x_90 : vec2<f32> = ((-(vec2<f32>(x_82.x, x_82.z)) * vec2<f32>(x_85.x, x_85.y)) + vec2<f32>(x_88.x, x_88.y));
  let x_91 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_90.x, x_90.y, x_91.z, x_91.w);
  let x_96 : f32 = in_TEXCOORD1.x;
  let x_100 : f32 = u_xlat10.x;
  u_xlat1.x = ((x_96 * 8160.0f) + x_100);
  let x_103 : vec4<f32> = u_xlat0;
  let x_107 : vec2<f32> = (vec2<f32>(x_103.x, x_103.y) * vec2<f32>(0.09375f, 0.03125f));
  let x_108 : vec4<f32> = u_xlat0;
  u_xlat0 = vec4<f32>(x_107.x, x_107.y, x_108.z, x_108.w);
  let x_111 : f32 = in_TEXCOORD1.y;
  let x_115 : f32 = u_xlat0.x;
  u_xlat1.y = ((x_111 * 2040.0f) + x_115);
  let x_119 : vec4<f32> = u_xlat1;
  u_xlat3 = (vec4<f32>(x_119.x, x_119.y, x_119.x, x_119.y) + vec4<f32>(0.5f, 0.5f, 0.5f, 1.5f));
  let x_125 : vec4<f32> = u_xlat3;
  let x_136 : vec4<f32> = x_131.x_ShaderInfoTex_TexelSize;
  u_xlat3 = (x_125 * vec4<f32>(x_136.x, x_136.y, x_136.x, x_136.y));
  let x_139 : vec4<f32> = u_xlat1;
  let x_143 : vec2<f32> = (vec2<f32>(x_139.x, x_139.y) + vec2<f32>(0.5f, 2.5f));
  let x_144 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_143.x, x_143.y, x_144.z, x_144.w);
  let x_146 : vec4<f32> = u_xlat1;
  let x_149 : vec4<f32> = x_131.x_ShaderInfoTex_TexelSize;
  let x_151 : vec2<f32> = (vec2<f32>(x_146.x, x_146.y) * vec2<f32>(x_149.x, x_149.y));
  let x_152 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_151.x, x_151.y, x_152.z, x_152.w);
  let x_165 : vec4<f32> = u_xlat3;
  let x_168 : vec4<f32> = textureSampleLevel(x_ShaderInfoTex, sampler_ShaderInfoTex, vec2<f32>(x_165.x, x_165.y), 0.0f);
  u_xlat4 = x_168;
  let x_172 : vec4<f32> = u_xlat3;
  let x_174 : vec4<f32> = textureSampleLevel(x_ShaderInfoTex, sampler_ShaderInfoTex, vec2<f32>(x_172.z, x_172.w), 0.0f);
  u_xlat3 = x_174;
  let x_179 : vec4<f32> = u_xlat1;
  let x_181 : vec4<f32> = textureSampleLevel(x_ShaderInfoTex, sampler_ShaderInfoTex, vec2<f32>(x_179.x, x_179.y), 0.0f);
  u_xlat5 = x_181;
  let x_184 : f32 = in_TEXCOORD3.x;
  u_xlat0.x = (x_184 * 255.0f);
  let x_188 : f32 = u_xlat0.x;
  u_xlat0.x = round(x_188);
  let x_192 : vec4<f32> = u_xlat0;
  u_xlatb6 = (vec4<f32>(x_192.x, x_192.x, x_192.x, x_192.x) == vec4<f32>(4.0f, 3.0f, 2.0f, 1.0f));
  let x_202 : bool = u_xlatb6.x;
  u_xlat7.x = select(0.0f, 1.0f, x_202);
  let x_206 : bool = u_xlatb6.y;
  u_xlat7.y = select(0.0f, 1.0f, x_206);
  let x_210 : bool = u_xlatb6.z;
  u_xlat7.z = select(0.0f, 1.0f, x_210);
  let x_214 : bool = u_xlatb6.w;
  u_xlat7.w = select(0.0f, 1.0f, x_214);
  let x_217 : vec4<f32> = u_xlat7;
  let x_219 : vec4<f32> = u_xlat7;
  let x_221 : vec2<f32> = (vec2<f32>(x_217.z, x_217.z) + vec2<f32>(x_219.w, x_219.y));
  let x_222 : vec4<f32> = u_xlat1;
  u_xlat1 = vec4<f32>(x_221.x, x_221.y, x_222.z, x_222.w);
  let x_225 : f32 = u_xlat7.y;
  let x_227 : f32 = u_xlat1.x;
  u_xlat0.x = (x_225 + x_227);
  let x_231 : f32 = u_xlat7.x;
  let x_233 : f32 = u_xlat0.x;
  u_xlat0.x = (x_231 + x_233);
  let x_237 : f32 = u_xlat0.x;
  u_xlat0.x = min(x_237, 1.0f);
  let x_241 : f32 = u_xlat0.x;
  u_xlat0.x = (-(x_241) + 1.0f);
  let x_247 : vec4<f32> = u_xlat4;
  let x_249 : vec4<f32> = in_POSITION0;
  u_xlat20.x = dot(x_247, x_249);
  let x_252 : vec4<f32> = u_xlat3;
  let x_253 : vec4<f32> = in_POSITION0;
  u_xlat20.y = dot(x_252, x_253);
  let x_256 : vec4<f32> = u_xlat5;
  let x_257 : vec4<f32> = in_POSITION0;
  u_xlat1.x = dot(x_256, x_257);
  let x_260 : vec2<f32> = u_xlat20;
  let x_265 : vec4<f32> = x_131.unity_ObjectToWorld[1i];
  u_xlat3 = (vec4<f32>(x_260.y, x_260.y, x_260.y, x_260.y) * x_265);
  let x_268 : vec4<f32> = x_131.unity_ObjectToWorld[0i];
  let x_269 : vec2<f32> = u_xlat20;
  let x_272 : vec4<f32> = u_xlat3;
  u_xlat3 = ((x_268 * vec4<f32>(x_269.x, x_269.x, x_269.x, x_269.x)) + x_272);
  let x_275 : vec4<f32> = x_131.unity_ObjectToWorld[2i];
  let x_276 : vec4<f32> = u_xlat1;
  let x_279 : vec4<f32> = u_xlat3;
  u_xlat3 = ((x_275 * vec4<f32>(x_276.x, x_276.x, x_276.x, x_276.x)) + x_279);
  let x_281 : vec4<f32> = u_xlat3;
  let x_284 : vec4<f32> = x_131.unity_ObjectToWorld[3i];
  u_xlat3 = (x_281 + x_284);
  let x_286 : vec4<f32> = u_xlat3;
  let x_289 : vec4<f32> = x_131.unity_MatrixVP[1i];
  u_xlat5 = (vec4<f32>(x_286.y, x_286.y, x_286.y, x_286.y) * x_289);
  let x_292 : vec4<f32> = x_131.unity_MatrixVP[0i];
  let x_293 : vec4<f32> = u_xlat3;
  let x_296 : vec4<f32> = u_xlat5;
  u_xlat5 = ((x_292 * vec4<f32>(x_293.x, x_293.x, x_293.x, x_293.x)) + x_296);
  let x_299 : vec4<f32> = x_131.unity_MatrixVP[2i];
  let x_300 : vec4<f32> = u_xlat3;
  let x_303 : vec4<f32> = u_xlat5;
  u_xlat5 = ((x_299 * vec4<f32>(x_300.z, x_300.z, x_300.z, x_300.z)) + x_303);
  let x_310 : vec4<f32> = x_131.unity_MatrixVP[3i];
  let x_311 : vec4<f32> = u_xlat3;
  let x_314 : vec4<f32> = u_xlat5;
  gl_Position = ((x_310 * vec4<f32>(x_311.w, x_311.w, x_311.w, x_311.w)) + x_314);
  u_xlat2.y = 0.0f;
  u_xlat3.x = 0.0f;
  u_xlat3.y = 0.0f;
  u_xlati1 = 0i;
  loop {
    if (true) {
    } else {
      break;
    }
    let x_331 : f32 = u_xlat3.y;
    u_xlatb18 = (x_331 >= 7.0f);
    u_xlati1 = 0i;
    let x_334 : bool = u_xlatb18;
    if (x_334) {
      break;
    }
    let x_340 : f32 = u_xlat3.y;
    let x_342 : f32 = u_xlat3.y;
    u_xlat18 = (x_340 + x_342);
    let x_346 : f32 = u_xlat18;
    u_xlatu18 = u32(x_346);
    let x_349 : f32 = in_TEXCOORD7;
    let x_350 : u32 = u_xlatu18;
    let x_354 : f32 = x_131.x_TextureInfo[bitcast<i32>(x_350)].x;
    u_xlatb18 = (x_349 == x_354);
    let x_356 : bool = u_xlatb18;
    if (x_356) {
      let x_360 : f32 = u_xlat3.y;
      u_xlat3.x = x_360;
      u_xlati1 = -1i;
      break;
    }
    let x_365 : f32 = u_xlat3.y;
    u_xlat2.x = (x_365 + 1.0f);
    let x_368 : vec4<f32> = u_xlat2;
    let x_369 : vec2<f32> = vec2<f32>(x_368.y, x_368.x);
    let x_370 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_369.x, x_369.y, x_370.z, x_370.w);
    let x_372 : bool = u_xlatb18;
    u_xlatb1.x = x_372;
  }
  let x_374 : i32 = u_xlati1;
  if ((x_374 != 0i)) {
    let x_381 : f32 = u_xlat3.x;
    x_377 = x_381;
  } else {
    x_377 = 7.0f;
  }
  let x_383 : f32 = x_377;
  u_xlat1.x = x_383;
  let x_386 : bool = u_xlatb6.w;
  u_xlat2.x = select(0.0f, 2.0f, x_386);
  let x_390 : bool = u_xlatb6.x;
  u_xlat2.y = select(0.0f, 4.0f, x_390);
  let x_394 : f32 = u_xlat0.x;
  let x_396 : f32 = u_xlat2.x;
  u_xlat0.x = (x_394 + x_396);
  let x_400 : f32 = u_xlat1.y;
  let x_403 : f32 = u_xlat0.x;
  u_xlat0.x = ((x_400 * 3.0f) + x_403);
  let x_408 : f32 = u_xlat2.y;
  let x_410 : f32 = u_xlat0.x;
  vs_TEXCOORD1.x = (x_408 + x_410);
  let x_415 : vec4<f32> = in_TEXCOORD5;
  u_xlat0.x = dot(vec2<f32>(x_415.x, x_415.y), vec2<f32>(65025.0f, 255.0f));
  let x_425 : vec4<f32> = in_TEXCOORD3;
  let x_427 : vec4<bool> = (vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f) < vec4<f32>(x_425.z, x_425.w, x_425.z, x_425.z));
  u_xlatb2 = vec2<bool>(x_427.x, x_427.y);
  let x_431 : bool = u_xlatb2.x;
  u_xlat9.x = select(0.0f, 1.0f, x_431);
  let x_435 : bool = u_xlatb2.x;
  u_xlat2.x = select(2.0f, 3.0f, x_435);
  let x_439 : bool = u_xlatb2.y;
  if (x_439) {
    let x_444 : f32 = u_xlat2.x;
    x_440 = x_444;
  } else {
    let x_447 : f32 = u_xlat9.x;
    x_440 = x_447;
  }
  let x_448 : f32 = x_440;
  vs_TEXCOORD1.w = x_448;
  let x_451 : bool = u_xlatb6.y;
  if (x_451) {
    let x_455 : f32 = u_xlat1.x;
    let x_457 : f32 = u_xlat1.x;
    u_xlat9.x = (x_455 + x_457);
    let x_462 : f32 = u_xlat9.x;
    u_xlatu9 = u32(x_462);
    let x_467 : vec2<f32> = in_TEXCOORD0;
    let x_468 : u32 = u_xlatu9;
    let x_471 : vec4<f32> = x_131.x_TextureInfo[bitcast<i32>(x_468)];
    let x_473 : vec2<f32> = (x_467 * vec2<f32>(x_471.y, x_471.z));
    let x_474 : vec4<f32> = vs_TEXCOORD0;
    vs_TEXCOORD0 = vec4<f32>(x_473.x, x_473.y, x_474.z, x_474.w);
  } else {
    let x_477 : vec2<f32> = in_TEXCOORD0;
    let x_478 : vec4<f32> = vs_TEXCOORD0;
    vs_TEXCOORD0 = vec4<f32>(x_477.x, x_477.y, x_478.z, x_478.w);
  }
  let x_482 : f32 = in_TEXCOORD4.x;
  let x_485 : f32 = u_xlat10.z;
  u_xlat3.x = ((x_482 * 8160.0f) + x_485);
  let x_489 : f32 = in_TEXCOORD4.y;
  let x_492 : f32 = u_xlat0.y;
  u_xlat3.y = ((x_489 * 2040.0f) + x_492);
  let x_495 : vec4<f32> = u_xlat3;
  let x_498 : vec2<f32> = (vec2<f32>(x_495.x, x_495.y) + vec2<f32>(0.5f, 0.5f));
  let x_499 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_498.x, x_499.y, x_498.y, x_499.w);
  let x_501 : vec4<f32> = u_xlat2;
  let x_504 : vec4<f32> = x_131.x_ShaderInfoTex_TexelSize;
  let x_506 : vec2<f32> = (vec2<f32>(x_501.x, x_501.z) * vec2<f32>(x_504.x, x_504.y));
  let x_507 : vec4<f32> = u_xlat2;
  u_xlat2 = vec4<f32>(x_506.x, x_507.y, x_506.y, x_507.w);
  let x_510 : bool = u_xlatb1.z;
  u_xlat3.x = select(-32.0f, 32.0f, x_510);
  let x_514 : bool = u_xlatb1.z;
  u_xlat3.y = select(-0.03125f, 0.03125f, x_514);
  let x_518 : bool = u_xlatb1.w;
  u_xlat3.z = select(-32.0f, 32.0f, x_518);
  let x_522 : bool = u_xlatb1.w;
  u_xlat3.w = select(-0.03125f, 0.03125f, x_522);
  let x_525 : vec4<f32> = u_xlat0;
  let x_527 : vec4<f32> = u_xlat3;
  u_xlat9 = (vec3<f32>(x_525.z, x_525.w, x_525.z) * vec3<f32>(x_527.y, x_527.w, x_527.y));
  let x_530 : vec3<f32> = u_xlat9;
  u_xlat9 = fract(x_530);
  let x_533 : vec3<f32> = u_xlat9;
  let x_535 : vec4<f32> = u_xlat3;
  let x_537 : vec2<f32> = (vec2<f32>(x_533.y, x_533.z) * vec2<f32>(x_535.z, x_535.x));
  let x_538 : vec3<f32> = u_xlat11;
  u_xlat11 = vec3<f32>(x_537.x, x_538.y, x_537.y);
  let x_541 : vec4<f32> = u_xlat3;
  let x_544 : vec3<f32> = u_xlat9;
  let x_546 : vec4<f32> = u_xlat0;
  u_xlat8 = ((-(vec3<f32>(x_541.x, x_541.z, x_541.x)) * x_544) + vec3<f32>(x_546.z, x_546.w, x_546.z));
  let x_549 : vec3<f32> = u_xlat8;
  u_xlat8 = (x_549 * vec3<f32>(0.03125f, 0.03125f, 0.125f));
  let x_554 : f32 = in_TEXCOORD4.z;
  let x_557 : f32 = u_xlat11.z;
  u_xlat5.x = ((x_554 * 8160.0f) + x_557);
  let x_560 : vec4<f32> = in_TEXCOORD4;
  let x_564 : vec3<f32> = u_xlat8;
  let x_566 : vec2<f32> = ((vec2<f32>(x_560.w, x_560.w) * vec2<f32>(2040.0f, 2040.0f)) + vec2<f32>(x_564.x, x_564.z));
  let x_567 : vec4<f32> = u_xlat5;
  u_xlat5 = vec4<f32>(x_567.x, x_566.x, x_566.y, x_567.w);
  let x_569 : vec4<f32> = u_xlat5;
  let x_571 : vec2<f32> = (vec2<f32>(x_569.x, x_569.y) + vec2<f32>(0.5f, 0.5f));
  let x_572 : vec3<f32> = u_xlat8;
  u_xlat8 = vec3<f32>(x_571.x, x_572.y, x_571.y);
  let x_574 : vec3<f32> = u_xlat8;
  let x_577 : vec4<f32> = x_131.x_ShaderInfoTex_TexelSize;
  let x_579 : vec2<f32> = (vec2<f32>(x_574.x, x_574.z) * vec2<f32>(x_577.x, x_577.y));
  let x_580 : vec3<f32> = u_xlat8;
  u_xlat8 = vec3<f32>(x_579.x, x_580.y, x_579.y);
  let x_583 : bool = u_xlatb2.y;
  if (x_583) {
    let x_589 : vec3<f32> = u_xlat8;
    let x_591 : vec4<f32> = textureSampleLevel(x_ShaderInfoTex, sampler_ShaderInfoTex, vec2<f32>(x_589.x, x_589.z), 0.0f);
    u_xlat7 = vec4<f32>(x_591.w, x_591.x, x_591.y, x_591.z);
    let x_594 : vec4<f32> = u_xlat7;
    let x_595 : vec3<f32> = vec3<f32>(x_594.y, x_594.z, x_594.w);
    let x_596 : vec4<f32> = vs_COLOR0;
    vs_COLOR0 = vec4<f32>(x_595.x, x_595.y, x_595.z, x_596.w);
  } else {
    let x_600 : vec4<f32> = in_COLOR0;
    u_xlat9 = ((vec3<f32>(x_600.x, x_600.y, x_600.z) * vec3<f32>(0.08497100323438644409f, 0.08497100323438644409f, 0.08497100323438644409f)) + vec3<f32>(-0.00016302900621667504f, -0.00016302900621667504f, -0.00016302900621667504f));
    let x_608 : vec4<f32> = in_COLOR0;
    let x_615 : vec3<f32> = ((vec3<f32>(x_608.x, x_608.y, x_608.z) * vec3<f32>(0.26588499546051025391f, 0.26588499546051025391f, 0.26588499546051025391f)) + vec3<f32>(0.73658400774002075195f, 0.73658400774002075195f, 0.73658400774002075195f));
    let x_616 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_615.x, x_616.y, x_615.y, x_615.z);
    let x_618 : vec4<f32> = in_COLOR0;
    let x_620 : vec4<f32> = u_xlat3;
    let x_625 : vec3<f32> = ((vec3<f32>(x_618.x, x_618.y, x_618.z) * vec3<f32>(x_620.x, x_620.z, x_620.w)) + vec3<f32>(-0.0098018404096364975f, -0.0098018404096364975f, -0.0098018404096364975f));
    let x_626 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_625.x, x_626.y, x_625.y, x_625.z);
    let x_628 : vec4<f32> = in_COLOR0;
    let x_630 : vec4<f32> = u_xlat3;
    let x_635 : vec3<f32> = ((vec3<f32>(x_628.x, x_628.y, x_628.z) * vec3<f32>(x_630.x, x_630.z, x_630.w)) + vec3<f32>(0.00319697009399533272f, 0.00319697009399533272f, 0.00319697009399533272f));
    let x_636 : vec4<f32> = u_xlat3;
    u_xlat3 = vec4<f32>(x_635.x, x_636.y, x_635.y, x_635.z);
    let x_638 : vec4<f32> = in_COLOR0;
    let x_642 : vec4<bool> = (vec4<f32>(x_638.x, x_638.y, x_638.z, x_638.x) < vec4<f32>(0.07254900038242340088f, 0.07254900038242340088f, 0.07254900038242340088f, 0.0f));
    let x_644 : vec3<bool> = vec3<bool>(x_642.x, x_642.y, x_642.z);
    let x_645 : vec4<bool> = u_xlatb6;
    u_xlatb6 = vec4<bool>(x_644.x, x_644.y, x_644.z, x_645.w);
    let x_648 : bool = u_xlatb6.x;
    if (x_648) {
      let x_653 : f32 = u_xlat9.x;
      x_649 = x_653;
    } else {
      let x_656 : f32 = u_xlat3.x;
      x_649 = x_656;
    }
    let x_657 : f32 = x_649;
    vs_COLOR0.x = x_657;
    let x_660 : bool = u_xlatb6.y;
    if (x_660) {
      let x_665 : f32 = u_xlat9.y;
      x_661 = x_665;
    } else {
      let x_668 : f32 = u_xlat3.z;
      x_661 = x_668;
    }
    let x_669 : f32 = x_661;
    vs_COLOR0.y = x_669;
    let x_672 : bool = u_xlatb6.z;
    if (x_672) {
      let x_677 : f32 = u_xlat9.z;
      x_673 = x_677;
    } else {
      let x_680 : f32 = u_xlat3.w;
      x_673 = x_680;
    }
    let x_681 : f32 = x_673;
    vs_COLOR0.z = x_681;
    let x_684 : f32 = in_COLOR0.w;
    u_xlat7.x = x_684;
  }
  let x_687 : f32 = in_TEXCOORD1.z;
  let x_690 : f32 = u_xlat11.x;
  u_xlat3.x = ((x_687 * 8160.0f) + x_690);
  let x_694 : f32 = in_TEXCOORD1.w;
  let x_697 : f32 = u_xlat8.y;
  u_xlat3.y = ((x_694 * 2040.0f) + x_697);
  let x_700 : vec4<f32> = u_xlat3;
  let x_702 : vec2<f32> = (vec2<f32>(x_700.x, x_700.y) + vec2<f32>(0.5f, 0.5f));
  let x_703 : vec3<f32> = u_xlat9;
  u_xlat9 = vec3<f32>(x_702.x, x_702.y, x_703.z);
  let x_705 : vec3<f32> = u_xlat9;
  let x_708 : vec4<f32> = x_131.x_ShaderInfoTex_TexelSize;
  let x_710 : vec2<f32> = (vec2<f32>(x_705.x, x_705.y) * vec2<f32>(x_708.x, x_708.y));
  let x_711 : vec3<f32> = u_xlat9;
  u_xlat9 = vec3<f32>(x_710.x, x_710.y, x_711.z);
  let x_717 : vec4<f32> = u_xlat2;
  let x_719 : vec4<f32> = textureSampleLevel(x_ShaderInfoTex, sampler_ShaderInfoTex, vec2<f32>(x_717.x, x_717.z), 0.0f);
  u_xlat16 = x_719.w;
  let x_721 : f32 = u_xlat16;
  let x_723 : f32 = u_xlat7.x;
  vs_COLOR0.w = (x_721 * x_723);
  let x_727 : bool = u_xlatb6.w;
  if (x_727) {
    let x_731 : f32 = u_xlat16;
    x_728 = x_731;
  } else {
    let x_734 : f32 = u_xlat0.x;
    x_728 = x_734;
  }
  let x_735 : f32 = x_728;
  vs_TEXCOORD1.z = x_735;
  let x_740 : bool = u_xlatb6.w;
  if (x_740) {
    let x_745 : vec4<f32> = u_xlat5;
    x_742 = vec2<f32>(x_745.x, x_745.z);
  } else {
    let x_748 : vec3<f32> = u_xlat8;
    x_742 = vec2<f32>(x_748.x, x_748.z);
  }
  let x_750 : vec2<f32> = x_742;
  vs_TEXCOORD3 = x_750;
  let x_754 : vec3<f32> = u_xlat9;
  let x_756 : vec4<f32> = textureSampleLevel(x_ShaderInfoTex, sampler_ShaderInfoTex, vec2<f32>(x_754.x, x_754.y), 0.0f);
  u_xlat0 = x_756;
  let x_757 : vec2<f32> = u_xlat20;
  let x_758 : vec4<f32> = u_xlat0;
  let x_761 : vec4<f32> = u_xlat0;
  let x_763 : vec2<f32> = ((x_757 * vec2<f32>(x_758.x, x_758.y)) + vec2<f32>(x_761.z, x_761.w));
  let x_764 : vec4<f32> = vs_TEXCOORD0;
  vs_TEXCOORD0 = vec4<f32>(x_764.x, x_764.y, x_763.x, x_763.y);
  let x_768 : bool = u_xlatb6.w;
  if (x_768) {
    let x_773 : f32 = in_TEXCOORD3.y;
    x_769 = x_773;
  } else {
    let x_777 : f32 = in_TEXCOORD6.x;
    x_769 = x_777;
  }
  let x_778 : f32 = x_769;
  vs_TEXCOORD4.x = x_778;
  let x_781 : f32 = u_xlat1.x;
  vs_TEXCOORD1.y = x_781;
  let x_783 : vec4<f32> = in_TEXCOORD6;
  let x_784 : vec3<f32> = vec3<f32>(x_783.y, x_783.z, x_783.w);
  let x_785 : vec4<f32> = vs_TEXCOORD4;
  vs_TEXCOORD4 = vec4<f32>(x_785.x, x_784.x, x_784.y, x_784.z);
  return;
}

struct main_out {
  @builtin(position)
  gl_Position : vec4<f32>,
  @location(2) @interpolate(flat)
  vs_TEXCOORD1_1 : vec4<f32>,
  @location(1)
  vs_TEXCOORD0_1 : vec4<f32>,
  @location(0)
  vs_COLOR0_1 : vec4<f32>,
  @location(3) @interpolate(flat)
  vs_TEXCOORD3_1 : vec2<f32>,
  @location(4)
  vs_TEXCOORD4_1 : vec4<f32>,
}

@vertex
fn main(@location(4) in_TEXCOORD2_param : vec4<f32>, @location(3) in_TEXCOORD1_param : vec4<f32>, @location(5) in_TEXCOORD3_param : vec4<f32>, @location(0) in_POSITION0_param : vec4<f32>, @location(9) in_TEXCOORD7_param : f32, @location(7) in_TEXCOORD5_param : vec4<f32>, @location(2) in_TEXCOORD0_param : vec2<f32>, @location(6) in_TEXCOORD4_param : vec4<f32>, @location(1) in_COLOR0_param : vec4<f32>, @location(8) in_TEXCOORD6_param : vec4<f32>) -> main_out {
  in_TEXCOORD2 = in_TEXCOORD2_param;
  in_TEXCOORD1 = in_TEXCOORD1_param;
  in_TEXCOORD3 = in_TEXCOORD3_param;
  in_POSITION0 = in_POSITION0_param;
  in_TEXCOORD7 = in_TEXCOORD7_param;
  in_TEXCOORD5 = in_TEXCOORD5_param;
  in_TEXCOORD0 = in_TEXCOORD0_param;
  in_TEXCOORD4 = in_TEXCOORD4_param;
  in_COLOR0 = in_COLOR0_param;
  in_TEXCOORD6 = in_TEXCOORD6_param;
  main_1();
  return main_out(gl_Position, vs_TEXCOORD1, vs_TEXCOORD0, vs_COLOR0, vs_TEXCOORD3, vs_TEXCOORD4);
}


