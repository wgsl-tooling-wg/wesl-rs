//! JSON Tests from the `wesl-testsuite`
//! See schemas: https://github.com/wgsl-tooling-wg/wesl-testsuite/blob/main/src/TestSchema.ts
//!
//! These tests are run with `harness = false` in `Cargo.toml`, because they rely on the
//! `libtest_mimic` custom harness to generate tests at runtime based on the JSON files.

use std::ffi::OsStr;

use wesl::{CompileOptions, EscapeMangler, NoMangler, Resolver, VirtualResolver, syntax::*};
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
        let file = std::fs::read_to_string("spec-tests/idents.json")
            .expect("failed to read test file");
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

    // TODO: fix this test. See https://github.com/wgsl-tooling-wg/wesl-rs/issues/84
    // tests.extend({
    //     let file = std::fs::read_to_string("wesl-testsuite/src/test-cases-json/importCases.json")
    //         .expect("failed to read test file");
    //     let json: Vec<WgslTestSrc> =
    //         serde_json::from_str(&file).expect("failed to parse json file");
    //     json.into_iter().map(|case| {
    //         let name = format!("importCases.json::{}", case.name);
    //         libtest_mimic::Trial::test(name, move || {
    //             testsuite_case(&case).inspect_err(|_| eprint_wgsl_test(&case))
    //         })
    //     })
    // });

    tests.extend({
        let file = std::fs::read_to_string(
            "wesl-testsuite/src/test-cases-json/conditionalTranslationCases.json",
        )
        .expect("failed to read test file");
        let json: Vec<WgslTestSrc> =
            serde_json::from_str(&file).expect("failed to parse json file");
        json.into_iter().map(|case| {
            let name = format!("conditionalTranslationCases.json::{}", case.name);
            let ignored = case.name == "declaration shadowing"; // this test requires stripping enabled.
            libtest_mimic::Trial::test(name, move || {
                testsuite_case(&case).inspect_err(|_| eprint_wgsl_test(&case))
            })
            .with_ignored_flag(ignored)
        })
    });

    tests.extend({
        let entries = std::fs::read_dir("webgpu-samples").expect("missing dir `webgpu-samples`");
        entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension() == Some(OsStr::new("wgsl")))
            .map(|e| {
                let name = format!("webgpu-samples::{:?}", e.file_name());
                libtest_mimic::Trial::test(name, move || {
                    let case = std::fs::read_to_string(e.path()).expect("failed to read test file");
                    validation_case(&case)
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
                libtest_mimic::Trial::test(name, move || {
                    let case = std::fs::read_to_string(e.path()).expect("failed to read test file");
                    bevy_parse_case(&case)
                })
            })
    });

    tests.extend({
        [
            // these are the most distinct files from unity's `boat_attack` sample. There is not great
            // value in testing all of them, they don't differ by much.
            // https://github.com/wgsl-tooling-wg/wesl-js/issues/161
            "unity_webgpu_0000020A44565050.fs.wgsl",
            "unity_webgpu_000001D9D2114040.fs.wgsl",
            "unity_webgpu_0000014DFB395020.fs.wgsl",
            "unity_webgpu_0000017E9E2D81A0.vs.wgsl",
            "unity_webgpu_00000277907BA020.fs.wgsl",
            "unity_webgpu_000002778F64B030.vs.wgsl",
            "unity_webgpu_000001F972AC3D10.vs.wgsl",
            "unity_webgpu_0000026E57303490.fs.wgsl",
            "unity_webgpu_000001D9D2114040.fs.wgsl",
            "unity_webgpu_000001D9CDD5C6D0.vs.wgsl",
        ]
        .iter()
        .map(|file| {
            let name = format!("unity_web_research::{}", file);
            libtest_mimic::Trial::test(name, move || {
                let path = format!("unity_web_research/boat_attack/{}", file);
                let case = std::fs::read_to_string(&path).expect("failed to read test file");
                validation_case(&case)
            })
        })
    });

    let args = libtest_mimic::Arguments::from_args();
    libtest_mimic::run(&args, tests).exit();
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
    if case.fails && parse.is_ok() {
        Err("expected Fail, got Pass".into())
    } else if !case.fails && parse.is_err() {
        Err(format!("expected Pass, got Fail (`{}`)", parse.unwrap_err()).into())
    } else {
        Ok(())
    }
}
pub fn testsuite_case(case: &WgslTestSrc) -> Result<(), libtest_mimic::Failed> {
    let mut resolver = VirtualResolver::new();

    for (path, file) in &case.wesl_src {
        resolver.add_module(path, file.into());
    }

    let root_module = ModulePath::from_path("/main");
    let compile_options = CompileOptions {
        strip: false,
        ..Default::default()
    };

    let mut case_wgsl = wesl::compile(&root_module, &resolver, &EscapeMangler, &compile_options)?;

    if let Some(expect_wgsl) = &case.underscore_wgsl {
        let mut expect_wgsl = wgsl_parse::parse_str(expect_wgsl)?;
        sort_decls(&mut case_wgsl.syntax);
        sort_decls(&mut expect_wgsl);
        assert_eq!(case_wgsl.to_string(), expect_wgsl.to_string());
    }

    Ok(())
}

pub fn validation_case(input: &str) -> Result<(), libtest_mimic::Failed> {
    let mut resolver = VirtualResolver::new();
    let root = ModulePath::from_path("/main");
    resolver.add_module(root.clone(), input.into());
    let options = CompileOptions {
        imports: true,
        condcomp: true,
        generics: false,
        strip: false,
        lower: true,
        validate: true,
        ..Default::default()
    };
    wesl::compile(&root, &resolver, &NoMangler, &options)?;
    Ok(())
}

pub fn bevy_parse_case(input: &str) -> Result<(), libtest_mimic::Failed> {
    // TODO this is temporary, eventually we want to resolve bevy internal shaders.
    struct UniversalResolver<'a> {
        root: ModulePath,
        input: &'a str,
    }
    impl Resolver for UniversalResolver<'_> {
        fn resolve_source<'a>(
            &'a self,
            path: &ModulePath,
        ) -> Result<std::borrow::Cow<'a, str>, wesl::ResolveError> {
            if path == &self.root {
                Ok(self.input.into())
            } else {
                Ok("".into())
            }
        }
    }
    let resolver = UniversalResolver {
        root: ModulePath::from_path("/main"),
        input,
    };
    let options = CompileOptions {
        imports: false,
        condcomp: true,
        generics: false,
        strip: false,
        lower: false,
        validate: false,
        ..Default::default()
    };
    wesl::compile(&resolver.root, &resolver, &NoMangler, &options)?;
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
