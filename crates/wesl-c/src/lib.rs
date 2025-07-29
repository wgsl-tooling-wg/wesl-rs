use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uint};
use std::ptr;

use wesl::{CompileResult, VirtualResolver, Wesl};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum WeslManglerKind {
    Escape = 0,
    Hash = 1,
    None = 2,
}

impl From<WeslManglerKind> for wesl::ManglerKind {
    fn from(value: WeslManglerKind) -> Self {
        match value {
            WeslManglerKind::Escape => wesl::ManglerKind::Escape,
            WeslManglerKind::Hash => wesl::ManglerKind::Hash,
            WeslManglerKind::None => wesl::ManglerKind::None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum WeslBindingType {
    Uniform = 0,
    Storage = 1,
    ReadOnlyStorage = 2,
    Filtering = 3,
    NonFiltering = 4,
    Comparison = 5,
    Float = 6,
    UnfilterableFloat = 7,
    Sint = 8,
    Uint = 9,
    Depth = 10,
    WriteOnly = 11,
    ReadWrite = 12,
    ReadOnly = 13,
}

#[repr(C)]
pub struct WeslBinding {
    pub group: c_uint,
    pub binding: c_uint,
    pub kind: WeslBindingType,
    pub data: *const u8,
    pub data_len: usize,
}

#[repr(C)]
pub struct WeslCompileOptions {
    pub mangler: WeslManglerKind,
    pub sourcemap: bool,
    pub imports: bool,
    pub condcomp: bool,
    pub generics: bool,
    pub strip: bool,
    pub lower: bool,
    pub validate: bool,
    pub naga: bool,
    pub lazy: bool,
    pub keep_root: bool,
    pub mangle_root: bool,
}

#[repr(C)]
pub struct WeslStringMap {
    pub keys: *const *const c_char,
    pub values: *const *const c_char,
    pub len: usize,
}

#[repr(C)]
pub struct WeslBoolMap {
    pub keys: *const *const c_char,
    pub values: *const bool,
    pub len: usize,
}

#[repr(C)]
pub struct WeslStringArray {
    pub items: *const *const c_char,
    pub len: usize,
}

#[repr(C)]
pub struct WeslBindingArray {
    pub items: *const WeslBinding,
    pub len: usize,
}

#[repr(C)]
pub struct WeslDiagnostic {
    pub file: *const c_char,
    pub span_start: c_uint,
    pub span_end: c_uint,
    pub title: *const c_char,
}

#[repr(C)]
pub struct WeslError {
    pub source: *const c_char,
    pub message: *const c_char,
    pub diagnostics: *const WeslDiagnostic,
    pub diagnostics_len: usize,
}

#[repr(C)]
pub struct WeslResult {
    pub success: bool,
    pub data: *const c_char,
    pub error: WeslError,
}

// -- handles

#[repr(C)]
pub struct WeslCompiler {
    compiler: Box<dyn std::any::Any + Send + Sync>,
}

#[repr(C)]
pub struct WeslCompileResult {
    result: CompileResult,
}

// -- helpers

unsafe fn string_map_to_hashmap(map: *const WeslStringMap) -> HashMap<String, String> {
    if map.is_null() {
        return HashMap::new();
    }

    unsafe {
        let map = &*map;
        let mut result = HashMap::new();

        for i in 0..map.len {
            let key_ptr = *map.keys.add(i);
            let value_ptr = *map.values.add(i);

            if !key_ptr.is_null() && !value_ptr.is_null() {
                let key = CStr::from_ptr(key_ptr).to_string_lossy().into_owned();
                let value = CStr::from_ptr(value_ptr).to_string_lossy().into_owned();
                result.insert(key, value);
            }
        }

        result
    }
}

unsafe fn bool_map_to_hashmap(map: *const WeslBoolMap) -> HashMap<String, bool> {
    if map.is_null() {
        return HashMap::new();
    }

    unsafe {
        let map = &*map;
        let mut result = HashMap::new();

        for i in 0..map.len {
            let key_ptr = *map.keys.add(i);
            let value = *map.values.add(i);

            if !key_ptr.is_null() {
                let key = CStr::from_ptr(key_ptr).to_string_lossy().into_owned();
                result.insert(key, value);
            }
        }

        result
    }
}

unsafe fn string_array_to_vec(array: *const WeslStringArray) -> Option<Vec<String>> {
    if array.is_null() {
        return None;
    }

    unsafe {
        let array = &*array;
        let mut result = Vec::new();

        for i in 0..array.len {
            let item_ptr = *array.items.add(i);
            if !item_ptr.is_null() {
                let item = CStr::from_ptr(item_ptr).to_string_lossy().into_owned();
                result.push(item);
            }
        }

        Some(result)
    }
}

fn create_c_string(s: &str) -> *const c_char {
    match CString::new(s) {
        Ok(c_str) => {
            let ptr = c_str.as_ptr();
            std::mem::forget(c_str);
            ptr
        }
        Err(_) => ptr::null(),
    }
}

fn wesl_error_to_c(e: wesl::Error) -> WeslError {
    let d = wesl::Diagnostic::from(e);

    let diagnostics = if let (Some(span), Some(res)) = (&d.detail.span, &d.detail.module_path) {
        let diag = WeslDiagnostic {
            file: create_c_string(&res.components.join("/")),
            span_start: span.start as u32,
            span_end: span.end as u32,
            title: create_c_string(&d.error.to_string()),
        };

        let boxed = Box::new(diag);
        let ptr = Box::into_raw(boxed);
        ptr
    } else {
        ptr::null()
    };

    WeslError {
        source: d
            .detail
            .output
            .as_ref()
            .map_or(ptr::null(), |s| create_c_string(s)),
        message: create_c_string(&d.to_string()),
        diagnostics,
        diagnostics_len: if diagnostics.is_null() {
            0
        } else {
            1
        },
    }
}

// -- main API

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_create_compiler() -> *mut WeslCompiler {
    let compiler = Wesl::new_barebones();
    let boxed = Box::new(WeslCompiler {
        compiler: Box::new(compiler),
    });
    Box::into_raw(boxed)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_destroy_compiler(compiler: *mut WeslCompiler) {
    if !compiler.is_null() {
        let _ = unsafe { Box::from_raw(compiler) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_compile(
    files: *const WeslStringMap,
    root: *const c_char,
    options: *const WeslCompileOptions,
    keep: *const WeslStringArray,
    features: *const WeslBoolMap,
) -> WeslResult {
    if files.is_null() || root.is_null() || options.is_null() {
        return WeslResult {
            success: false,
            data: ptr::null(),
            error: WeslError {
                source: ptr::null(),
                message: create_c_string("Invalid parameters"),
                diagnostics: ptr::null(),
                diagnostics_len: 0,
            },
        };
    }

    let files_map = unsafe { string_map_to_hashmap(files) };
    let root_str = unsafe { CStr::from_ptr(root).to_string_lossy() };
    let opts = unsafe { &*options };
    let keep_vec = unsafe { string_array_to_vec(keep) };
    let features_map = unsafe { bool_map_to_hashmap(features) };

    let root_path = match root_str.parse() {
        Ok(path) => path,
        Err(e) => {
            return WeslResult {
                success: false,
                data: ptr::null(),
                error: WeslError {
                    source: ptr::null(),
                    message: create_c_string(&format!("Invalid root path: {}", e)),
                    diagnostics: ptr::null(),
                    diagnostics_len: 0,
                },
            };
        }
    };

    let mut resolver = VirtualResolver::new();
    for (path, source) in files_map {
        if let Ok(module_path) = path.parse() {
            resolver.add_module(module_path, source.into());
        }
    }

    let mut compiler = Wesl::new_barebones().set_custom_resolver(resolver);
    let compiler = compiler
        .set_options(wesl::CompileOptions {
            imports: opts.imports,
            condcomp: opts.condcomp,
            generics: opts.generics,
            strip: opts.strip,
            lower: opts.lower,
            validate: opts.validate,
            lazy: opts.lazy,
            mangle_root: opts.mangle_root,
            keep: keep_vec,
            features: wesl::Features {
                default: wesl::Feature::Disable,
                flags: features_map
                    .into_iter()
                    .map(|(k, v)| (k, v.into()))
                    .collect(),
            },
            keep_root: opts.keep_root,
        })
        .use_sourcemap(opts.sourcemap)
        .set_mangler(opts.mangler.into());

    match compiler.compile(&root_path) {
        Ok(result) => {
            let output = result.to_string();
            WeslResult {
                success: true,
                data: create_c_string(&output),
                error: WeslError {
                    source: ptr::null(),
                    message: ptr::null(),
                    diagnostics: ptr::null(),
                    diagnostics_len: 0,
                },
            }
        }
        Err(e) => WeslResult {
            success: false,
            data: ptr::null(),
            error: wesl_error_to_c(e),
        },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_eval(
    files: *const WeslStringMap,
    root: *const c_char,
    expression: *const c_char,
    options: *const WeslCompileOptions,
    features: *const WeslBoolMap,
) -> WeslResult {
    if files.is_null() || root.is_null() || expression.is_null() || options.is_null() {
        return WeslResult {
            success: false,
            data: ptr::null(),
            error: WeslError {
                source: ptr::null(),
                message: create_c_string("Invalid parameters"),
                diagnostics: ptr::null(),
                diagnostics_len: 0,
            },
        };
    }

    let files_map = unsafe { string_map_to_hashmap(files) };
    let root_str = unsafe { CStr::from_ptr(root).to_string_lossy() };
    let expr_str = unsafe { CStr::from_ptr(expression).to_string_lossy() };
    let opts = unsafe { &*options };
    let features_map = unsafe { bool_map_to_hashmap(features) };

    let root_path = match root_str.parse() {
        Ok(path) => path,
        Err(e) => {
            return WeslResult {
                success: false,
                data: ptr::null(),
                error: WeslError {
                    source: ptr::null(),
                    message: create_c_string(&format!("Invalid root path: {}", e)),
                    diagnostics: ptr::null(),
                    diagnostics_len: 0,
                },
            };
        }
    };

    let mut resolver = VirtualResolver::new();
    for (path, source) in files_map {
        if let Ok(module_path) = path.parse() {
            resolver.add_module(module_path, source.into());
        }
    }

    let mut compiler = Wesl::new_barebones().set_custom_resolver(resolver);
    let compiler = compiler
        .set_options(wesl::CompileOptions {
            imports: opts.imports,
            condcomp: opts.condcomp,
            generics: opts.generics,
            strip: opts.strip,
            lower: opts.lower,
            validate: opts.validate,
            lazy: opts.lazy,
            mangle_root: opts.mangle_root,
            keep: None,
            features: wesl::Features {
                default: wesl::Feature::Disable,
                flags: features_map
                    .into_iter()
                    .map(|(k, v)| (k, v.into()))
                    .collect(),
            },
            keep_root: opts.keep_root,
        })
        .use_sourcemap(opts.sourcemap)
        .set_mangler(opts.mangler.into());

    match compiler.compile(&root_path) {
        Ok(result) => match result.eval(&expr_str) {
            Ok(eval_result) => WeslResult {
                success: true,
                data: create_c_string(&eval_result.inst.to_string()),
                error: WeslError {
                    source: ptr::null(),
                    message: ptr::null(),
                    diagnostics: ptr::null(),
                    diagnostics_len: 0,
                },
            },
            Err(e) => WeslResult {
                success: false,
                data: ptr::null(),
                error: wesl_error_to_c(e),
            },
        },
        Err(e) => WeslResult {
            success: false,
            data: ptr::null(),
            error: wesl_error_to_c(e),
        },
    }
}

// -- memory

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_free_string(ptr: *const c_char) {
    if !ptr.is_null() {
        let _ = unsafe { CString::from_raw(ptr as *mut c_char) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_free_result(result: *mut WeslResult) {
    if !result.is_null() {
        unsafe {
            let result = &mut *result;

            if !result.data.is_null() {
                wesl_free_string(result.data);
            }

            if !result.error.source.is_null() {
                wesl_free_string(result.error.source);
            }

            if !result.error.message.is_null() {
                wesl_free_string(result.error.message);
            }

            if !result.error.diagnostics.is_null() {
                let diag = &*result.error.diagnostics;
                if !diag.file.is_null() {
                    wesl_free_string(diag.file);
                }
                if !diag.title.is_null() {
                    wesl_free_string(diag.title);
                }
                let _ = Box::from_raw(result.error.diagnostics as *mut WeslDiagnostic);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_free_compile_result(result: *mut WeslCompileResult) {
    if !result.is_null() {
        let _ = unsafe { Box::from_raw(result) };
    }
}

// -- utility

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_version() -> *const c_char {
    create_c_string(env!("CARGO_PKG_VERSION"))
}
