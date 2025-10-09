#![doc = include_str!("../README.md")]
#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::os::raw::{c_char, c_uint, c_void};
use std::ptr;

use wesl::{ModulePath, ResolveError, VirtualResolver, Wesl};

#[cfg(feature = "eval")]
use wesl::{
    eval::{Eval, EvalAttrs, Inputs, Instance, RefInstance},
    syntax::{AccessMode, AddressSpace},
};

pub mod native {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct WeslCompiler {
    compiler: Wesl<wesl::NoResolver>,
}

pub struct WeslTranslationUnit {}

fn map_mangler_kind(value: native::WeslManglerKind) -> Option<wesl::ManglerKind> {
    match value {
        native::WESL_MANGLER_NONE => Some(wesl::ManglerKind::None),
        native::WESL_MANGLER_HASH => Some(wesl::ManglerKind::Hash),
        native::WESL_MANGLER_ESCAPE => Some(wesl::ManglerKind::Escape),
        _ => None,
    }
}

// -- helpers

unsafe fn string_map_to_hashmap(map: *const native::WeslStringMap) -> HashMap<String, String> {
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

unsafe fn bool_map_to_hashmap(map: *const native::WeslBoolMap) -> HashMap<String, bool> {
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

unsafe fn string_array_to_vec(array: *const native::WeslStringArray) -> Option<Vec<String>> {
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

fn wesl_error_to_c(e: wesl::Error) -> native::WeslError {
    let d = wesl::Diagnostic::from(e);

    let diagnostics = if let (Some(span), Some(res)) = (&d.detail.span, &d.detail.module_path) {
        let diag = native::WeslDiagnostic {
            file: create_c_string(&res.components.join("/")),
            span_start: span.start as u32,
            span_end: span.end as u32,
            title: create_c_string(&d.error.to_string()),
        };

        let boxed = Box::new(diag);

        Box::into_raw(boxed)
    } else {
        ptr::null()
    };

    native::WeslError {
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

#[cfg(feature = "eval")]
unsafe fn binding_array_to_vec(array: *const WeslBindingArray) -> Vec<WeslBinding> {
    if array.is_null() {
        return Vec::new();
    }

    unsafe {
        let array = &*array;
        let mut result = Vec::new();

        for i in 0..array.len {
            let binding = *array.items.add(i);
            result.push(binding);
        }

        result
    }
}

#[cfg(feature = "eval")]
fn parse_c_binding(
    b: &WeslBinding,
    wgsl: &wesl::syntax::TranslationUnit,
) -> Result<((u32, u32), RefInstance), wesl::Error> {
    let mut ctx = wesl::eval::Context::new(wgsl);

    let ty_expr = wgsl
        .global_declarations
        .iter()
        .find_map(|d| match d.node() {
            wesl::syntax::GlobalDeclaration::Declaration(d) => {
                let (group, binding) = d.attr_group_binding(&mut ctx).ok()?;
                if group == b.group && binding == b.binding {
                    d.ty.clone()
                } else {
                    None
                }
            }
            _ => None,
        })
        .ok_or_else(|| {
            wesl::Error::Custom(format!(
                "Resource @group({}) @binding({}) not found",
                b.group, b.binding
            ))
        })?;

    let ty = wesl::eval::ty_eval_ty(&ty_expr, &mut ctx)
        .map_err(|e| wesl::Error::Custom(format!("Failed to evaluate type: {e}")))?;

    let (storage, access) = match b.kind {
        WeslBindingType::Uniform => (AddressSpace::Uniform, AccessMode::Read),
        WeslBindingType::Storage => (AddressSpace::Storage, AccessMode::ReadWrite),
        WeslBindingType::ReadOnlyStorage => (AddressSpace::Storage, AccessMode::Read),
        _ => return Err(wesl::Error::Custom("Unsupported binding type".to_string())),
    };

    let data_slice = unsafe { std::slice::from_raw_parts(b.data, b.data_len) };
    let inst = Instance::from_buffer(data_slice, &ty).ok_or_else(|| {
        wesl::Error::Custom(format!(
            "Resource @group({}) @binding({}) ({} bytes) incompatible with type ({} bytes)",
            b.group,
            b.binding,
            b.data_len,
            ty.size_of().unwrap_or_default()
        ))
    })?;

    Ok((
        (b.group, b.binding),
        RefInstance::new(inst, storage, access),
    ))
}

#[cfg(feature = "eval")]
fn create_c_binding_array(bindings: Vec<WeslBinding>) -> *const WeslBindingArray {
    if bindings.is_empty() {
        return ptr::null();
    }

    let items = bindings.into_boxed_slice();
    let len = items.len();
    let items_ptr = Box::into_raw(items) as *const WeslBinding;

    let array = Box::new(WeslBindingArray {
        items: items_ptr,
        len,
    });
    Box::into_raw(array)
}

// -- main API

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_create_compiler() -> *mut WeslCompiler {
    let compiler = Wesl::new_barebones();
    Box::into_raw(Box::new(WeslCompiler { compiler }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_destroy_compiler(compiler: *mut WeslCompiler) {
    if !compiler.is_null() {
        let _ = unsafe { Box::from_raw(compiler) };
    }
}

fn error_from_str(s: &str) -> native::WeslError {
    native::WeslError {
        source: ptr::null(),
        message: create_c_string(s),
        diagnostics: ptr::null(),
        diagnostics_len: 0,
    }
}

fn result_from_str(s: &str) -> native::WeslResult {
    native::WeslResult {
        success: false,
        data: ptr::null(),
        error: error_from_str(s),
    }
}

fn result_invalid_parameters() -> native::WeslResult {
    result_from_str("Invalid parameters")
}

const NO_ERROR: native::WeslError = native::WeslError {
    source: ptr::null(),
    message: ptr::null(),
    diagnostics: ptr::null(),
    diagnostics_len: 0,
};

struct CustomResolver {
    pub options: native::WeslResolverOptions,
}

struct FreeGuard<T> {
    pub data: *const T,
    pub free_function: unsafe extern "C" fn(*const T, *mut c_void),
    pub free_userdata: *mut c_void,
}

impl<T> Drop for FreeGuard<T> {
    fn drop(&mut self) {
        unsafe {
            (self.free_function)(self.data, self.free_userdata);
        }
    }
}

impl<T> Deref for FreeGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.data }
    }
}

impl wesl::Resolver for CustomResolver {
    fn resolve_source<'a>(
        &'a self,
        path: &ModulePath,
    ) -> Result<std::borrow::Cow<'a, str>, ResolveError> {
        println!("RESOLVING: {path:?}, {path}");
        let path = path.to_string();

        let cstring = CString::new(path).expect("Module path contained nul bytes!");

        let result = unsafe {
            self.options.resolve_source.unwrap()(cstring.as_ptr(), self.options.userdata)
        };

        if result.is_null() {
            return Err(ResolveError::Error(
                wesl::Error::Custom("No value returned from resolver".into()).into(),
            ))
        }

        let result = unsafe {
            FreeGuard {
                data: &*result,
                free_function: self.options.resolve_source_free.unwrap(),
                free_userdata: self.options.userdata,
            }
        };

        if !result.success {
            return Err(ResolveError::Error(
                wesl::Error::Custom("Custom resolver failed".into()).into(),
            ))
        }

        let result_cstr = unsafe { CStr::from_ptr(result.source) };
        let result_str = result_cstr.to_str().map_err(|_| {
            ResolveError::Error(
                wesl::Error::Custom("Resolved source is not valid UTF-8".into()).into(),
            )
        })?;

        Ok(result_str.to_owned().into())
    }

    /*
    TODO: IMPLEMENT
    fn resolve_module(&self, path: &wesl::ModulePath) -> Result<wesl::syntax::TranslationUnit, wesl::ResolveError> {
        let source = self.resolve_source(path)?;
        let wesl: wesl::syntax::TranslationUnit = source.parse().map_err(|e| {
            wesl::Diagnostic::from(e)
                .with_module_path(path.clone(), self.display_name(path))
                .with_source(source.to_string())
        })?;
        Ok(wesl)
    }
    */

    // TODO: IMPLEMENT
    fn display_name(&self, _path: &ModulePath) -> Option<String> {
        None
    }

    fn fs_path(&self, _path: &ModulePath) -> Option<std::path::PathBuf> {
        None
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_compile(
    files: *const native::WeslStringMap,
    root: *const c_char,
    options: *const native::WeslCompileOptions,
    keep: *const native::WeslStringArray,
    features: *const native::WeslBoolMap,
) -> native::WeslResult {
    if root.is_null() || options.is_null() {
        return result_invalid_parameters();
    }

    let root_str = unsafe { CStr::from_ptr(root).to_string_lossy() };
    let opts = unsafe { &*options };
    let keep_vec = unsafe { string_array_to_vec(keep) };
    let features_map = unsafe { bool_map_to_hashmap(features) };

    let root_path = match root_str.parse() {
        Ok(path) => path,
        Err(e) => return result_from_str(&format!("Invalid root path: {e}")),
    };

    let resolver: Box<dyn wesl::Resolver> = match (files.is_null(), opts.resolver.is_null()) {
        (false, true) => {
            let files_map = unsafe { string_map_to_hashmap(files) };
            let mut resolver = VirtualResolver::new();
            for (path, source) in files_map {
                if let Ok(module_path) = path.parse() {
                    resolver.add_module(module_path, source.into());
                }
            }

            Box::new(resolver)
        }
        (true, false) => Box::new(CustomResolver {
            options: unsafe { *opts.resolver },
        }),
        (false, false) => {
            return result_from_str("Files and custom resolver cannot be specified at once");
        }
        _ => return result_from_str("Files or custom resolver must be specified"),
    };

    let Some(mangler) = map_mangler_kind(opts.mangler) else {
        return result_from_str("Invalid mangler kind specified");
    };

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
        .set_mangler(mangler);

    match compiler.compile(&root_path) {
        Ok(result) => {
            let output = result.to_string();
            native::WeslResult {
                success: true,
                data: create_c_string(&output),
                error: NO_ERROR,
            }
        }
        Err(e) => native::WeslResult {
            success: false,
            data: ptr::null(),
            error: wesl_error_to_c(e),
        },
    }
}

#[cfg(feature = "eval")]
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
                    message: create_c_string(&format!("Invalid root path: {e}")),
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

#[cfg(not(feature = "eval"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_eval(
    _files: *const native::WeslStringMap,
    _root: *const c_char,
    _expression: *const c_char,
    _options: *const native::WeslCompileOptions,
    _features: *const native::WeslBoolMap,
) -> native::WeslResult {
    result_from_str("wesl_eval requires the 'eval' feature to be enabled")
}

#[cfg(feature = "eval")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_exec(
    files: *const WeslStringMap,
    root: *const c_char,
    entrypoint: *const c_char,
    options: *const WeslCompileOptions,
    resources: *const WeslBindingArray,
    overrides: *const WeslStringMap,
    features: *const WeslBoolMap,
) -> WeslExecResult {
    if files.is_null() || root.is_null() || entrypoint.is_null() || options.is_null() {
        return WeslExecResult {
            success: false,
            resources: ptr::null(),
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
    let entrypoint_str = unsafe { CStr::from_ptr(entrypoint).to_string_lossy() };
    let opts = unsafe { &*options };
    let resources_vec = unsafe { binding_array_to_vec(resources) };
    let overrides_map = unsafe { string_map_to_hashmap(overrides) };
    let features_map = unsafe { bool_map_to_hashmap(features) };

    let root_path = match root_str.parse() {
        Ok(path) => path,
        Err(e) => {
            return WeslExecResult {
                success: false,
                resources: ptr::null(),
                error: WeslError {
                    source: ptr::null(),
                    message: create_c_string(&format!("Invalid root path: {e}")),
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
        Ok(result) => {
            // parse resources
            let parsed_resources: Result<HashMap<(u32, u32), RefInstance>, wesl::Error> =
                resources_vec
                    .iter()
                    .map(|r| parse_c_binding(r, &result.syntax))
                    .collect();

            let parsed_resources = match parsed_resources {
                Ok(resources) => resources,
                Err(e) => {
                    return WeslExecResult {
                        success: false,
                        resources: ptr::null(),
                        error: wesl_error_to_c(e),
                    };
                }
            };

            // parse overrides
            let parsed_overrides: Result<HashMap<String, Instance>, wesl::Error> = overrides_map
                .iter()
                .map(|(name, expr)| {
                    let mut ctx = wesl::eval::Context::new(&result.syntax);
                    let expr = expr.parse::<wesl::syntax::Expression>().map_err(|e| {
                        wesl::Error::Custom(format!("Failed to parse override expression: {e}"))
                    })?;
                    let inst = expr.eval_value(&mut ctx).map_err(|e| {
                        wesl::Error::Custom(format!("Failed to evaluate override: {e}"))
                    })?;
                    Ok((name.clone(), inst))
                })
                .collect();

            let parsed_overrides = match parsed_overrides {
                Ok(overrides) => overrides,
                Err(e) => {
                    return WeslExecResult {
                        success: false,
                        resources: ptr::null(),
                        error: wesl_error_to_c(e),
                    };
                }
            };

            // execute
            let inputs = Inputs::new_zero_initialized();
            match result.exec(&entrypoint_str, inputs, parsed_resources, parsed_overrides) {
                Ok(exec_result) => {
                    // convert resources back to C format
                    let output_resources: Vec<WeslBinding> = resources_vec
                        .iter()
                        .filter_map(|r| {
                            let resource = exec_result.resource(r.group, r.binding)?;
                            let inst = resource.read().ok()?.to_owned();
                            let mut new_binding = *r;
                            if let Some(buffer) = inst.to_buffer() {
                                let boxed_data = buffer.into_boxed_slice();
                                new_binding.data_len = boxed_data.len();
                                new_binding.data = Box::into_raw(boxed_data) as *const u8;
                            }
                            Some(new_binding)
                        })
                        .collect();

                    WeslExecResult {
                        success: true,
                        resources: create_c_binding_array(output_resources),
                        error: WeslError {
                            source: ptr::null(),
                            message: ptr::null(),
                            diagnostics: ptr::null(),
                            diagnostics_len: 0,
                        },
                    }
                }
                Err(e) => WeslExecResult {
                    success: false,
                    resources: ptr::null(),
                    error: wesl_error_to_c(e),
                },
            }
        }
        Err(e) => WeslExecResult {
            success: false,
            resources: ptr::null(),
            error: wesl_error_to_c(e),
        },
    }
}

#[cfg(not(feature = "eval"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_exec(
    _files: *const native::WeslStringMap,
    _root: *const c_char,
    _entrypoint: *const c_char,
    _options: *const native::WeslCompileOptions,
    _resources: *const native::WeslBindingArray,
    _overrides: *const native::WeslStringMap,
    _features: *const native::WeslBoolMap,
) -> native::WeslExecResult {
    native::WeslExecResult {
        success: false,
        resources: ptr::null(),
        error: error_from_str("wesl_exec requires the 'eval' feature to be enabled"),
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
pub unsafe extern "C" fn wesl_free_result(result: *mut native::WeslResult) {
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
                let _ = Box::from_raw(result.error.diagnostics as *mut native::WeslDiagnostic);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_free_exec_result(result: *mut native::WeslExecResult) {
    if !result.is_null() {
        unsafe {
            let result = &mut *result;

            if !result.resources.is_null() {
                let resources = &*result.resources;

                // free each binding
                for i in 0..resources.len {
                    let binding = *resources.items.add(i);
                    if !binding.data.is_null() {
                        let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                            binding.data as *mut u8,
                            binding.data_len,
                        ));
                    }
                }

                let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                    resources.items as *mut native::WeslBinding,
                    resources.len,
                ));

                let _ = Box::from_raw(result.resources as *mut native::WeslBindingArray);
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
                let _ = Box::from_raw(result.error.diagnostics as *mut native::WeslDiagnostic);
            }
        }
    }
}

// -- utility

// note: results from this function must not be freed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wesl_version() -> *const c_char {
    const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}
