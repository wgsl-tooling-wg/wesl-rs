//! JSON Tests from the `wesl-testsuite`
//! See schemas: https://github.com/wgsl-tooling-wg/wesl-testsuite/blob/main/src/TestSchema.ts
//!
//! These tests are run with `harness = false` in `Cargo.toml`, because they rely on the
//! `libtest_mimic` custom harness to generate tests at runtime based on the JSON files.

use std::{ffi::OsStr, path::PathBuf, process::Command, str::FromStr};

use wesl::{CompileOptions, EscapeMangler, NoMangler, VirtualResolver, syntax::*};
use wesl_test::schemas::*;

fn eprint_test(case: &Test) {
    eprintln!(
        "case: `{}`\n* desc: {}{}\n* kind: {}\n* expect: {}\n* skip: {}\n* issue: {}",
        case.name,
        case.desc,
        case.note
            .as_ref()
            .map(|n| format!(" (note: {n})"))
            .unwrap_or_default(),
        case.kind,
        case.expect,
        case.skip.unwrap_or(false),
        case.issue.as_deref().unwrap_or("<none>")
    );
}

fn eprint_parsing_test(case: &ParsingTest) {
    let expects = if case.fails {
        "Fail"
    } else {
        "Pass"
    };
    println!("case `{}`: expect {expects}", case.src);
}

fn eprint_wgsl_test(case: &WgslTestSrc) {
    println!(
        "case: `{}`{}",
        case.name,
        case.notes
            .as_ref()
            .map(|n| format!(" (note: {n})"))
            .unwrap_or_default(),
    );
}

fn main() {
    let mut tests: Vec<libtest_mimic::Trial> = Vec::new();

    tests.extend({
        let file = std::fs::read_to_string("spec-tests/lit-type-inference.json")
            .expect("failed to read test file");
        let json: Vec<Test> = serde_json::from_str(&file).expect("failed to parse json file");
        json.into_iter().map(|case| {
            let name = format!("lit-type-inference.json::{}", case.name);
            let ignored = case.skip.unwrap_or(false);
            libtest_mimic::Trial::test(name, move || {
                json_case(&case).inspect_err(|_| eprint_test(&case))
            })
            .with_ignored_flag(ignored)
        })
    });

    tests.extend({
        let file =
            std::fs::read_to_string("spec-tests/idents.json").expect("failed to read test file");
        let json: Vec<Test> = serde_json::from_str(&file).expect("failed to parse json file");
        json.into_iter().map(|case| {
            let name = format!("idents.json::{}", case.name);
            let ignored = case.skip.unwrap_or(false);
            libtest_mimic::Trial::test(name, move || {
                json_case(&case).inspect_err(|_| eprint_test(&case))
            })
            .with_ignored_flag(ignored)
        })
    });

    tests.extend({
        let file =
            std::fs::read_to_string("wesl-testsuite/src/test-cases-json/importSyntaxCases.json")
                .expect("failed to read test file");
        let json: Vec<ParsingTest> =
            serde_json::from_str(&file).expect("failed to parse json file");
        json.into_iter().map(|mut case| {
            case.normalize();
            let name = format!("importSyntaxCases.json::{}", case.src);
            libtest_mimic::Trial::test(name, move || {
                testsuite_syntax_case(&case).inspect_err(|_| eprint_parsing_test(&case))
            })
        })
    });

    tests.extend({
        let file = std::fs::read_to_string("wesl-testsuite/src/test-cases-json/importCases.json")
            .expect("failed to read test file");
        let json: Vec<WgslTestSrc> =
            serde_json::from_str(&file).expect("failed to parse json file");
        json.into_iter().map(|case| {
            let name = format!("importCases.json::{}", case.name);
            libtest_mimic::Trial::test(name, move || {
                testsuite_case(&case).inspect_err(|_| eprint_wgsl_test(&case))
            })
        })
    });

    tests.extend({
        let file = std::fs::read_to_string(
            "wesl-testsuite/src/test-cases-json/conditionalTranslationCases.json",
        )
        .expect("failed to read test file");
        let json: Vec<WgslTestSrc> =
            serde_json::from_str(&file).expect("failed to parse json file");
        json.into_iter().map(|case| {
            let name = format!("conditionalTranslationCases.json::{}", case.name);
            libtest_mimic::Trial::test(name, move || {
                testsuite_case(&case).inspect_err(|_| eprint_wgsl_test(&case))
            })
        })
    });

    tests.extend({
        let file = std::fs::read_to_string("wesl-testsuite/src/test-cases-json/bulkTests.json")
            .expect("failed to read test file");
        let json: Vec<WgslBulkTest> =
            serde_json::from_str(&file).expect("failed to parse json file");
        json.into_iter().flat_map(|bulk_case| {
            let name = format!("bulkTests.json::{}", bulk_case.name);
            let cwd = std::path::Path::new("wesl-testsuite");
            fetch_bulk_test(&bulk_case, cwd)
                .unwrap_or_else(|_| panic!("failed to fetch bulk test {name}"));

            assert!(
                bulk_case.exclude.is_none_or(|v| v.is_empty()),
                "Globs are not supported"
            );
            let base_dir = cwd.join(&bulk_case.base_dir);
            let include_paths: Vec<_> = bulk_case
                .include
                .map(|v| v.iter().map(|v| base_dir.join(v)).collect())
                .unwrap_or_else(|| {
                    std::fs::read_dir(&bulk_case.base_dir)
                        .unwrap_or_else(|_| panic!("missing dir `{}`", &bulk_case.base_dir))
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().extension() == Some(OsStr::new("wgsl")))
                        .map(|v| v.path())
                        .collect()
                });

            include_paths.into_iter().map(move |shader_path| {
                libtest_mimic::Trial::test(format!("{name}::{shader_path:?}"), move || {
                    validation_case(shader_path)
                })
            })
        })
    });

    tests.extend({
        let entries = std::fs::read_dir("bevy").expect("missing dir `bevy`");
        entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension() == Some(OsStr::new("wgsl")))
            .map(|e| {
                let name = format!("bevy::{:?}", e.file_name());
                libtest_mimic::Trial::test(name, move || bevy_case(e.path()))
            })
    });

    tests.extend({
        let in_entries = std::fs::read_dir("wgpu/in")
            .expect("missing dir `wgpu/in`")
            .map(|f| (f, "in"));
        let out_entries = std::fs::read_dir("wgpu/out")
            .expect("missing dir `wgpu/out`")
            .map(|f| (f, "out"));
        in_entries
            .chain(out_entries)
            .filter_map(|(e, d)| e.ok().map(|e| (e, d)))
            .filter(|(e, _)| e.path().extension() == Some(OsStr::new("wgsl")))
            .map(|(e, d)| {
                let name = format!("wgpu::{d}::{:?}", e.file_name());
                libtest_mimic::Trial::test(name, move || validation_case(e.path()))
            })
    });

    let args = libtest_mimic::Arguments::from_args();
    libtest_mimic::run(&args, tests).exit();
}

fn fetch_bulk_test(bulk_test: &WgslBulkTest, cwd: &std::path::Path) -> std::io::Result<()> {
    // Modeled after https://github.com/gfx-rs/wgpu/blob/c0a580d6f0343a725b3defa8be4fdf0a9691eaad/xtask/src/cts.rs
    let Some(WgslGitSrc { url, revision }) = &bulk_test.git else {
        return Ok(());
    };
    let base_dir = cwd.join(&bulk_test.base_dir);
    if std::fs::exists(&base_dir)? {
        // Do a git update
        let commit_exists = Command::new("git")
            .args(["cat-file", "commit", revision])
            .current_dir(&base_dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("failed to execute git cat-file")
            .wait()
            .expect("failed to wait on git")
            .success();

        if !commit_exists {
            let git_fetch = Command::new("git")
                .args(["fetch", "--quiet"])
                .current_dir(&base_dir)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .expect("failed to execute git fetch")
                .wait()
                .expect("failed to wait on git");
            if !git_fetch.success() {
                panic!("Git fetch failed");
            }
        }

        let git_checkout = Command::new("git")
            .args(["checkout", "--quiet", revision])
            .current_dir(&base_dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("failed to execute git checkout")
            .wait()
            .expect("failed to wait on git");

        if !git_checkout.success() {
            panic!("Git checkout failed");
        }
    } else {
        let git_clone = Command::new("git")
            .args([
                "clone",
                "--depth=1",
                url,
                "--revision",
                revision,
                &bulk_test.base_dir,
            ])
            .current_dir(cwd)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("failed to execute git clone")
            .wait()
            .expect("failed to wait on git");

        if !git_clone.success() {
            panic!("Git clone failed");
        }
    }

    Ok(())
}

fn json_case(case: &Test) -> Result<(), libtest_mimic::Failed> {
    match &case.kind {
        TestKind::Syntax { syntax } => {
            let res = match syntax {
                SyntaxKind::Declaration => case.code.parse::<TranslationUnit>().map(|_| ()),
                SyntaxKind::Statement => case.code.parse::<Statement>().map(|_| ()),
                SyntaxKind::Expression => case.code.parse::<Expression>().map(|_| ()),
            };
            match res {
                Ok(()) => {
                    if case.expect == Expectation::Fail {
                        return Err("expected Fail, got Pass".into());
                    }
                }
                Err(e) => {
                    if case.expect == Expectation::Pass {
                        return Err(format!("expected Pass, got Fail (`{e}`)").into());
                    }
                }
            }
        }
        TestKind::Eval { eval, result } => {
            let wesl = case.code.parse::<TranslationUnit>()?;
            let expr = eval.parse::<Expression>()?;
            let (eval_inst, _) = wesl::eval(&expr, &wesl);
            match result {
                Some(expect) => {
                    let expr = expect.parse::<Expression>()?;
                    let (expect_inst, _) = wesl::eval(&expr, &wesl);
                    let expect_inst = expect_inst?;
                    let eval_inst = eval_inst?;
                    if eval_inst != expect_inst {
                        return Err(format!("expected `{expect_inst}`, got `{eval_inst}`").into());
                    }
                }
                None => {
                    if let Ok(inst) = eval_inst {
                        return Err(format!("expected Fail, got Pass (`{inst}`)").into());
                    }
                }
            };
        }
        TestKind::Context => {
            panic!("TODO: context tests")
        }
    }

    Ok(())
}

fn testsuite_syntax_case(case: &ParsingTest) -> Result<(), libtest_mimic::Failed> {
    let parse = wgsl_parse::parse_str(&case.src);
    match parse {
        Ok(s) if case.fails => Err(format!("expected Fail, got Pass (`{s}`)").into()),
        Ok(s) => {
            let str1 = s.to_string();
            let str2 = wgsl_parse::parse_str(&str1)
                .map_err(|e| {
                    format!("failed to parse after stringification\nerror: `{e}`\nsource: `{str1}`")
                })?
                .to_string();
            if str1 == str2 {
                Ok(())
            } else {
                Err(format!("stringification is lossy\nbefore: `{str1}`\nafter: `{str2}`").into())
            }
        }
        Err(e) if !case.fails => Err(format!("expected Pass, got Fail (`{e}`)").into()),
        Err(_) => Ok(()),
    }
}
pub fn testsuite_case(case: &WgslTestSrc) -> Result<(), libtest_mimic::Failed> {
    let mut resolver = VirtualResolver::new();

    for (path, file) in &case.wesl_src {
        let path = ModulePath::new_root().join_path(&ModulePath::from_path(path));
        resolver.add_module(path, file.into());
    }

    let root_module = ModulePath::from_str("package::main")?;
    let compile_options = CompileOptions {
        keep_root: true,
        ..Default::default()
    };

    let mut case_wgsl =
        wesl::compile_sourcemap(&root_module, &resolver, &EscapeMangler, &compile_options)?;

    if let Some(expect_wgsl) = &case.underscore_wgsl {
        let mut expect_wgsl = wgsl_parse::parse_str(expect_wgsl)?;
        sort_decls(&mut case_wgsl.syntax);
        sort_decls(&mut expect_wgsl);
        assert_eq!(case_wgsl.to_string(), expect_wgsl.to_string());
    }

    Ok(())
}

pub fn validation_case(path: PathBuf) -> Result<(), libtest_mimic::Failed> {
    let input = std::fs::read_to_string(path).expect("failed to read test file");
    let mut resolver = VirtualResolver::new();
    let root = ModulePath::from_str("package::main")?;
    resolver.add_module(root.clone(), input.into());
    let options = CompileOptions {
        strip: false,
        lower: true,
        validate: true,
        ..Default::default()
    };
    wesl::compile_sourcemap(&root, &resolver, &NoMangler, &options)?;
    Ok(())
}

pub fn bevy_case(path: PathBuf) -> Result<(), libtest_mimic::Failed> {
    let base = path.parent().ok_or("file not found")?;
    let name = path
        .file_name()
        .ok_or("file not found")?
        .to_string_lossy()
        .to_string();
    let mut compiler = wesl::Wesl::new(base);
    compiler
        .add_package(&bevy_wgsl::bevy::PACKAGE)
        .add_constants([
            ("MAX_CASCADES_PER_LIGHT", 10.0),
            ("MAX_DIRECTIONAL_LIGHTS", 10.0),
            ("PER_OBJECT_BUFFER_BATCH_SIZE", 10.0),
            ("TONEMAPPING_LUT_TEXTURE_BINDING_INDEX", 10.0),
            ("TONEMAPPING_LUT_SAMPLER_BINDING_INDEX", 10.0),
        ])
        .set_options(CompileOptions {
            strip: false,
            lower: true,
            validate: true,
            lazy: false,
            ..Default::default()
        })
        .set_feature("MULTISAMPLED", true) // show_prepass needs it
        .set_feature("DEPTH_PREPASS", true) // show_prepass needs it
        .set_feature("NORMAL_PREPASS", true) // show_prepass needs it
        .set_feature("IRRADIANCE_VOLUMES_ARE_USABLE", true) // irradiance_volume_voxel_visualization needs it
        .set_feature("IRRADIANCE_VOLUMES_ARE_USABLE", true) // irradiance_volume_voxel_visualization needs it
        .set_feature("MOTION_VECTOR_PREPASS", true) // show_prepass needs it
        .set_feature("CLUSTERED_DECALS_ARE_USABLE", true) // custom_clustered_decal needs it
        .set_feature("VERTEX_UVS_A", true) // texture_binding_array needs it
        .set_feature("VERTEX_OUTPUT_INSTANCE_INDEX", true); // extended_material needs it
    if name == "water_material.wgsl" {
        compiler.set_feature("PREPASS_FRAGMENT", true); // water_material needs it
        compiler.set_feature("PREPASS_PIPELINE", true); // water_material needs it
        compiler.set_feature("NORMAL_PREPASS_OR_DEFERRED_PREPASS", true); // water_material needs it
    }
    compiler.compile(&ModulePath::new(PathOrigin::Absolute, vec![name]))?;
    Ok(())
}

fn sort_decls(wgsl: &mut TranslationUnit) {
    use std::cmp::Ordering;
    type Decl = GlobalDeclaration;
    wgsl.global_declarations
        .sort_unstable_by(|a, b| match (a.node(), b.node()) {
            (Decl::Void, Decl::Void) => Ordering::Equal,
            (Decl::Void, Decl::Declaration(_)) => Ordering::Less,
            (Decl::Void, Decl::Struct(_)) => Ordering::Less,
            (Decl::Void, Decl::TypeAlias(_)) => Ordering::Less,
            (Decl::Void, Decl::ConstAssert(_)) => Ordering::Less,
            (Decl::Void, Decl::Function(_)) => Ordering::Less,

            (Decl::Declaration(_), Decl::Void) => Ordering::Greater,
            (Decl::Declaration(d1), Decl::Declaration(d2)) => d1.ident.name().cmp(&d2.ident.name()),
            (Decl::Declaration(_), Decl::Struct(_)) => Ordering::Less,
            (Decl::Declaration(_), Decl::TypeAlias(_)) => Ordering::Less,
            (Decl::Declaration(_), Decl::ConstAssert(_)) => Ordering::Less,
            (Decl::Declaration(_), Decl::Function(_)) => Ordering::Less,

            (Decl::Struct(_), Decl::Void) => Ordering::Greater,
            (Decl::Struct(_), Decl::Declaration(_)) => Ordering::Greater,
            (Decl::Struct(d1), Decl::Struct(d2)) => d1.ident.name().cmp(&d2.ident.name()),
            (Decl::Struct(_), Decl::TypeAlias(_)) => Ordering::Less,
            (Decl::Struct(_), Decl::ConstAssert(_)) => Ordering::Less,
            (Decl::Struct(_), Decl::Function(_)) => Ordering::Less,

            (Decl::TypeAlias(_), Decl::Void) => Ordering::Greater,
            (Decl::TypeAlias(_), Decl::Declaration(_)) => Ordering::Greater,
            (Decl::TypeAlias(_), Decl::Struct(_)) => Ordering::Greater,
            (Decl::TypeAlias(d1), Decl::TypeAlias(d2)) => d1.ident.name().cmp(&d2.ident.name()),
            (Decl::TypeAlias(_), Decl::ConstAssert(_)) => Ordering::Less,
            (Decl::TypeAlias(_), Decl::Function(_)) => Ordering::Less,

            (Decl::ConstAssert(_), Decl::Void) => Ordering::Greater,
            (Decl::ConstAssert(_), Decl::Declaration(_)) => Ordering::Greater,
            (Decl::ConstAssert(_), Decl::Struct(_)) => Ordering::Greater,
            (Decl::ConstAssert(_), Decl::TypeAlias(_)) => Ordering::Greater,
            (Decl::ConstAssert(_), Decl::ConstAssert(_)) => Ordering::Equal,
            (Decl::ConstAssert(_), Decl::Function(_)) => Ordering::Less,

            (Decl::Function(_), Decl::Void) => Ordering::Greater,
            (Decl::Function(_), Decl::Declaration(_)) => Ordering::Greater,
            (Decl::Function(_), Decl::Struct(_)) => Ordering::Greater,
            (Decl::Function(_), Decl::TypeAlias(_)) => Ordering::Greater,
            (Decl::Function(_), Decl::ConstAssert(_)) => Ordering::Greater,
            (Decl::Function(d1), Decl::Function(d2)) => d1.ident.name().cmp(&d2.ident.name()),
        });
}
