// This file contains calls to all constructor functions.

enable f16;

struct Student {
    grade: i32,
    gpa: f32,
    attendance: array<bool, 4>,
}

fn zero_value_constructors() {
    const bool_explicit = bool();
    const i32_explicit = i32();
    const u32_explicit = u32();
    const f32_explicit = f32();
    const f16_explicit = f16();

    const vec2_bool_explicit = vec2<bool>();
    const vec3_bool_explicit = vec3<bool>();
    const vec4_bool_explicit = vec4<bool>();
    const vec2_i32_explicit = vec2<i32>();
    const vec3_i32_explicit = vec3<i32>();
    const vec4_i32_explicit = vec4<i32>();
    const vec2_u32_explicit = vec2<u32>();
    const vec3_u32_explicit = vec3<u32>();
    const vec4_u32_explicit = vec4<u32>();
    const vec2_f32_explicit = vec2<f32>();
    const vec3_f32_explicit = vec3<f32>();
    const vec4_f32_explicit = vec4<f32>();
    const vec2_f16_explicit = vec2<f16>();
    const vec3_f16_explicit = vec3<f16>();
    const vec4_f16_explicit = vec4<f16>();

    const mat2x2_f32_explicit = mat2x2<f32>();
    const mat2x3_f32_explicit = mat2x3<f32>();
    const mat2x4_f32_explicit = mat2x4<f32>();
    const mat3x2_f32_explicit = mat3x2<f32>();
    const mat3x3_f32_explicit = mat3x3<f32>();
    const mat3x4_f32_explicit = mat3x4<f32>();
    const mat4x2_f32_explicit = mat4x2<f32>();
    const mat4x3_f32_explicit = mat4x3<f32>();
    const mat4x4_f32_explicit = mat4x4<f32>();
    const mat2x2_f16_explicit = mat2x2<f16>();
    const mat2x3_f16_explicit = mat2x3<f16>();
    const mat2x4_f16_explicit = mat2x4<f16>();
    const mat3x2_f16_explicit = mat3x2<f16>();
    const mat3x3_f16_explicit = mat3x3<f16>();
    const mat3x4_f16_explicit = mat3x4<f16>();
    const mat4x2_f16_explicit = mat4x2<f16>();
    const mat4x3_f16_explicit = mat4x3<f16>();
    const mat4x4_f16_explicit = mat4x4<f16>();

    const array_f32_explicit = array<f32, 3>();
    const array_struct_explicit = array<Student, 3>();
    const student_explicit = Student();

    var bool_implicit: bool;
    var i32_implicit: i32;
    var u32_implicit: u32;
    var f32_implicit: f32;
    var f16_implicit: f16;

    var vec2_bool_implicit: vec2<bool>;
    var vec3_bool_implicit: vec3<bool>;
    var vec4_bool_implicit: vec4<bool>;
    var vec2_i32_implicit: vec2<i32>;
    var vec3_i32_implicit: vec3<i32>;
    var vec4_i32_implicit: vec4<i32>;
    var vec2_u32_implicit: vec2<u32>;
    var vec3_u32_implicit: vec3<u32>;
    var vec4_u32_implicit: vec4<u32>;
    var vec2_f32_implicit: vec2<f32>;
    var vec3_f32_implicit: vec3<f32>;
    var vec4_f32_implicit: vec4<f32>;
    var vec2_f16_implicit: vec2<f16>;
    var vec3_f16_implicit: vec3<f16>;
    var vec4_f16_implicit: vec4<f16>;

    var mat2x2_f32_implicit: mat2x2<f32>;
    var mat2x3_f32_implicit: mat2x3<f32>;
    var mat2x4_f32_implicit: mat2x4<f32>;
    var mat3x2_f32_implicit: mat3x2<f32>;
    var mat3x3_f32_implicit: mat3x3<f32>;
    var mat3x4_f32_implicit: mat3x4<f32>;
    var mat4x2_f32_implicit: mat4x2<f32>;
    var mat4x3_f32_implicit: mat4x3<f32>;
    var mat4x4_f32_implicit: mat4x4<f32>;
    var mat2x2_f16_implicit: mat2x2<f16>;
    var mat2x3_f16_implicit: mat2x3<f16>;
    var mat2x4_f16_implicit: mat2x4<f16>;
    var mat3x2_f16_implicit: mat3x2<f16>;
    var mat3x3_f16_implicit: mat3x3<f16>;
    var mat3x4_f16_implicit: mat3x4<f16>;
    var mat4x2_f16_implicit: mat4x2<f16>;
    var mat4x3_f16_implicit: mat4x3<f16>;
    var mat4x4_f16_implicit: mat4x4<f16>;

    var array_f32_implicit: array<f32, 3>;
    var array_student_implicit: array<Student, 3>;
    var student_implicit: Student;
}

fn scalar_value_constructors() {
    const bool_from_AbstractFloat = bool(1.0);
    const i32_from_AbstractFloat = i32(1.0);
    const u32_from_AbstractFloat = u32(1.0);
    const f32_from_AbstractFloat = f32(1.0);
    const f16_from_AbstractFloat = f16(1.0);

    const bool_from_AbstractInt = bool(1);
    const i32_from_AbstractInt = i32(1);
    const u32_from_AbstractInt = u32(1);
    const f32_from_AbstractInt = f32(1);
    const f16_from_AbstractInt = f16(1);

    const bool_from_bool = bool(true);
    const i32_from_bool = i32(true);
    const u32_from_bool = u32(true);
    const f32_from_bool = f32(true);
    const f16_from_bool = f16(true);

    const bool_from_i32 = bool(1i);
    const i32_from_i32 = i32(1i);
    const u32_from_i32 = u32(1i);
    const f32_from_i32 = f32(1i);
    const f16_from_i32 = f16(1i);

    const bool_from_u32 = bool(1u);
    const i32_from_u32 = i32(1u);
    const u32_from_u32 = u32(1u);
    const f32_from_u32 = f32(1u);
    const f16_from_u32 = f16(1u);

    const bool_from_f32 = bool(1.f);
    const i32_from_f32 = i32(1.f);
    const u32_from_f32 = u32(1.f);
    const f32_from_f32 = f32(1.f);
    const f16_from_f32 = f16(1.f);

    const bool_from_f16 = bool(1.h);
    const i32_from_f16 = i32(1.h);
    const u32_from_f16 = u32(1.h);
    const f32_from_f16 = f32(1.h);
    const f16_from_f16 = f16(1.h);
}

fn vec_value_constructors() {
    const vec2_from_scalar = vec2(1.0);
    const vec3_from_scalar = vec3(1.0);
    const vec4_from_scalar = vec4(1.0);

    const vec2_from_vec2 = vec2(vec2(0.0, 1.0));
    const vec3_from_vec3 = vec3(vec3(0.0, 1.0, 2.0));
    const vec4_from_vec4 = vec4(vec4(0.0, 1.0, 2.0, 3.0));

    const vec2_from_scalars = vec2(0.0, 1.0);
    const vec3_from_scalars = vec3(0.0, 1.0, 2.0);
    const vec4_from_scalars = vec4(0.0, 1.0, 2.0, 3.0);

    const vec3_from_vec2_scalar = vec3(vec2(0.0, 1.0), 2.0);
    const vec3_from_scalar_vec2 = vec3(0.0, vec2(1.0, 2.0));

    const vec4_from_scalar_vec2_scalar = vec4(0.0, vec2(1.0, 2.0), 3.0);
    const vec4_from_scalars_vec2 = vec4(0.0, 1.0, vec2(2.0, 3.0));
    const vec4_from_vec2_vec2 = vec4(vec2(0.0, 1.0), vec2(2.0, 3.0));
    const vec4_from_vec2_scalars = vec4(vec2(0.0, 1.0), 2.0, 3.0);
    const vec4_from_vec3_scalar = vec4(vec3(0.0, 1.0, 2.0), 3.0);
    const vec4_from_scalar_vec3 = vec4(0.0, vec3(1.0, 2.0, 3.0));
}

fn mat_value_constructors() {
    const mat2x2_from_matrix = mat2x2(mat2x2<f32>());
    const mat2x3_from_matrix = mat2x3(mat2x3<f32>());
    const mat2x4_from_matrix = mat2x4(mat2x4<f32>());
    const mat3x2_from_matrix = mat3x2(mat3x2<f32>());
    const mat3x3_from_matrix = mat3x3(mat3x3<f32>());
    const mat3x4_from_matrix = mat3x4(mat3x4<f32>());
    const mat4x2_from_matrix = mat4x2(mat4x2<f32>());
    const mat4x3_from_matrix = mat4x3(mat4x3<f32>());
    const mat4x4_from_matrix = mat4x4(mat4x4<f32>());

    const mat2x2_from_vectors = mat2x2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    const mat2x3_from_vectors = mat2x3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    const mat2x4_from_vectors = mat2x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0));
    const mat3x2_from_vectors = mat3x2(vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 0.0));
    const mat3x3_from_vectors = mat3x3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
    const mat3x4_from_vectors = mat3x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0));
    const mat4x2_from_vectors = mat4x2(vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 0.0), vec2(0.0, 0.0));
    const mat4x3_from_vectors = mat4x3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.0));
    const mat4x4_from_vectors = mat4x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));

    const mat2x2_from_scalars = mat2x2(1.0, 0.0, 0.0, 1.0);
    const mat2x3_from_scalars = mat2x3(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    const mat2x4_from_scalars = mat2x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    const mat3x2_from_scalars = mat3x2(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    const mat3x3_from_scalars = mat3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    const mat3x4_from_scalars = mat3x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    const mat4x2_from_scalars = mat4x2(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    const mat4x3_from_scalars = mat4x3(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const mat4x4_from_scalars = mat4x4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
}

fn other_value_constructors() {
    const struct_from_members = Student(0, 0.0, array<bool, 4>(false, false, false, false));

    const array_from_scalars = array<i32, 3>(1, 2, 3);
    const array_from_structs = array<Student, 3>(Student(), Student(), Student());
    const array_inferred = array(1, 2, 3);
}
