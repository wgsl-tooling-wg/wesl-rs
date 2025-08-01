#ifndef WESL_H
#define WESL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// -- handles
typedef struct WeslCompiler WeslCompiler;

// -- enums
typedef enum WeslManglerKind {
    WESL_MANGLER_ESCAPE = 0,
    WESL_MANGLER_HASH = 1,
    WESL_MANGLER_NONE = 2
} WeslManglerKind;

typedef enum WeslBindingType {
    WESL_BINDING_UNIFORM = 0,
    WESL_BINDING_STORAGE = 1,
    WESL_BINDING_READ_ONLY_STORAGE = 2,
    WESL_BINDING_FILTERING = 3,
    WESL_BINDING_NON_FILTERING = 4,
    WESL_BINDING_COMPARISON = 5,
    WESL_BINDING_FLOAT = 6,
    WESL_BINDING_UNFILTERABLE_FLOAT = 7,
    WESL_BINDING_SINT = 8,
    WESL_BINDING_UINT = 9,
    WESL_BINDING_DEPTH = 10,
    WESL_BINDING_WRITE_ONLY = 11,
    WESL_BINDING_READ_WRITE = 12,
    WESL_BINDING_READ_ONLY = 13
} WeslBindingType;

// -- structs
typedef struct WeslBinding {
    unsigned int group;
    unsigned int binding;
    WeslBindingType kind;
    const uint8_t* data;
    size_t data_len;
} WeslBinding;

typedef struct WeslCompileOptions {
    WeslManglerKind mangler;
    bool sourcemap;
    bool imports;
    bool condcomp;
    bool generics;
    bool strip;
    bool lower;
    bool validate;
    bool naga;
    bool lazy;
    bool keep_root;
    bool mangle_root;
} WeslCompileOptions;

typedef struct WeslStringMap {
    const char* const* keys;
    const char* const* values;
    size_t len;
} WeslStringMap;

typedef struct WeslBoolMap {
    const char* const* keys;
    const bool* values;
    size_t len;
} WeslBoolMap;

typedef struct WeslStringArray {
    const char* const* items;
    size_t len;
} WeslStringArray;

typedef struct WeslBindingArray {
    const WeslBinding* items;
    size_t len;
} WeslBindingArray;

typedef struct WeslDiagnostic {
    const char* file;
    unsigned int span_start;
    unsigned int span_end;
    const char* title;
} WeslDiagnostic;

typedef struct WeslError {
    const char* source;
    const char* message;
    const WeslDiagnostic* diagnostics;
    size_t diagnostics_len;
} WeslError;

typedef struct WeslResult {
    bool success;
    const char* data;
    WeslError error;
} WeslResult;

typedef struct WeslExecOptions {
    WeslCompileOptions compile;
    const char* entrypoint;
    const WeslBindingArray* resources;
    const WeslStringMap* overrides;
} WeslExecOptions;

typedef struct WeslExecResult {
    bool success;
    const WeslBindingArray* resources;
    WeslError error;
} WeslExecResult;

// -- main API
WeslCompiler* wesl_create_compiler(void);
void wesl_destroy_compiler(WeslCompiler* compiler);

WeslResult wesl_compile(
    const WeslStringMap* files,
    const char* root,
    const WeslCompileOptions* options,
    const WeslStringArray* keep,
    const WeslBoolMap* features
);

WeslResult wesl_eval(
    const WeslStringMap* files,
    const char* root,
    const char* expression,
    const WeslCompileOptions* options,
    const WeslBoolMap* features
);

WeslExecResult wesl_exec(
    const WeslStringMap* files,
    const char* root,
    const char* entrypoint,
    const WeslCompileOptions* options,
    const WeslBindingArray* resources,
    const WeslStringMap* overrides,
    const WeslBoolMap* features
);

// -- memory
void wesl_free_string(const char* ptr);
void wesl_free_result(WeslResult* result);
void wesl_free_exec_result(WeslExecResult* result);

// -- utility

// note: results from this function must not be freed
const char* wesl_version(void);

#ifdef __cplusplus
}
#endif

#endif // WESL_H
