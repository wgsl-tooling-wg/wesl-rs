#![cfg_attr(not(test), allow(dead_code, unused_imports))]
#![feature(custom_test_frameworks)]
#![test_runner(datatest::runner)]

use std::cmp::Ordering;

use regex::Regex;
use wesl::{
    CompileOptions, EscapeMangler, NoMangler, VirtualResolver,
    syntax::{Expression, GlobalDeclaration, ModulePath, Statement, TranslationUnit},
};

mod schemas;
use schemas::*;

#[datatest::files("webgpu-samples", { input in r"^.*\.wgsl$", })]
#[test]
fn webgpu_samples(input: &str) {
    parse_test(input);
}

#[datatest::files("unity_web_research", {
  // input in r"webgpu/wgsl/boat_attack/.*\.wgsl$",
  input in r"unity_webgpu_(000001AC1A5BA040|0000026E572CD040)\.[fv]s\.wgsl$",
})]
#[test]
fn unity_web_research(input: &str) {
    parse_test(input);
}

#[datatest::data("spec-tests/lit-type-inference.json")]
#[test]
fn spec_tests(case: Test) {
    json_case(case);
}

#[datatest::data("wesl-testsuite/src/test-cases-json/importSyntaxCases.json")]
#[test]
fn wesl_testsuite_import_syntax_cases(case: ParsingTest) {
    testsuite_test_syntax(case);
}

#[datatest::data("wesl-testsuite/src/test-cases-json/importCases.json")]
#[test]
fn wesl_testsuite_import_cases(case: WgslTestSrc) {
    testsuite_test(case);
}

#[datatest::data("wesl-testsuite/src/test-cases-json/conditionalTranslationCases.json")]
#[test]
fn wesl_testsuite_conditional_translation_cases(case: WgslTestSrc) {
    testsuite_test(case);
}

fn parse_test(input: &str) {
    let mut resolver = VirtualResolver::new();
    let root = ModulePath::from_path("/main");
    resolver.add_module(root.clone(), input.into());
    let options = CompileOptions {
        imports: true,
        condcomp: true,
        generics: false,
        strip: true,
        lower: true,
        validate: true,
        ..Default::default()
    };
    wesl::compile(&root, &resolver, &NoMangler, &options)
        .inspect_err(|err| eprintln!("[FAIL] {err}"))
        .expect("test failed");
}

fn json_case(case: Test) {
    println!(
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
        case.issue.unwrap_or("<none>".to_string())
    );

    if case.skip.unwrap_or(false) {
        return;
    }

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
                        eprintln!("[FAIL] expected Fail, got Pass");
                        panic!("test failed")
                    }
                }
                Err(e) => {
                    if case.expect == Expectation::Fail {
                        eprintln!("[FAIL] expected Pass, got Fail ({e})");
                        panic!("test failed")
                    }
                }
            }
        }
        TestKind::Eval { eval, result } => {
            let wesl = case
                .code
                .parse::<TranslationUnit>()
                .inspect_err(|e| eprintln!("[FAIL] parse `code`: {e}"))
                .expect("test failed");
            let expr = eval
                .parse::<Expression>()
                .inspect_err(|e| eprintln!("[FAIL] parse `eval`: {e}"))
                .expect("test failed");
            let (eval_inst, _) = wesl::eval(&expr, &wesl);
            match result {
                Some(expect) => {
                    let expr = expect
                        .parse::<Expression>()
                        .inspect_err(|e| eprintln!("[FAIL] parse `expect`: {e}"))
                        .expect("test failed");
                    let (expect_inst, _) = wesl::eval(&expr, &wesl);
                    let expect_inst = expect_inst
                        .inspect_err(|e| eprintln!("[FAIL] eval `expect`: {e}"))
                        .expect("test failed");
                    let eval_inst = eval_inst
                        .inspect_err(|e| eprintln!("[FAIL] eval `eval`: {e}"))
                        .expect("test failed");
                    if eval_inst != expect_inst {
                        eprintln!("[FAIL] expected `{expect_inst}`, got `{eval_inst}`");
                        panic!("test failed");
                    }
                }
                None => {
                    if let Ok(inst) = eval_inst {
                        eprintln!("[FAIL] expected Fail, got Pass (`{inst}`)",);
                        panic!("test failed");
                    }
                }
            };
        }
        TestKind::Context => {
            panic!("TODO: context tests")
        }
    }
}

fn sort_decls(wgsl: &mut TranslationUnit) {
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

fn testsuite_test(case: WgslTestSrc) {
    println!(
        "case: `{}`{}",
        case.name,
        case.notes
            .as_ref()
            .map(|n| format!(" (note: {n})"))
            .unwrap_or_default(),
    );

    let mut resolver = VirtualResolver::new();

    for (path, file) in case.wesl_src {
        resolver.add_module(path, file.into());
    }

    let root_module = ModulePath::from_path("/main");
    let compile_options = CompileOptions {
        strip: false,
        ..Default::default()
    };

    let mut case_wgsl = wesl::compile(&root_module, &resolver, &EscapeMangler, &compile_options)
        .inspect_err(|err| eprintln!("[FAIL] compile: {err}"))
        .expect("test failed");

    if let Some(expect_wgsl) = case.underscore_wgsl {
        let mut expect_wgsl = wgsl_parse::parse_str(&expect_wgsl)
            .inspect_err(|err| eprintln!("[FAIL] parse `expectedWgsl`: {err}"))
            .expect("parse error");
        sort_decls(&mut case_wgsl.syntax);
        sort_decls(&mut expect_wgsl);
        assert_eq!(
            normalize_wgsl(&case_wgsl.to_string()),
            normalize_wgsl(&expect_wgsl.to_string())
        )
    }
}

fn normalize_wgsl(wgsl: &str) -> String {
    let re = Regex::new(r"\s+").unwrap();
    re.replace_all(wgsl, " ").trim().to_string()
}

fn testsuite_test_syntax(case: ParsingTest) {
    let expects = if case.fails {
        "Fail"
    } else {
        "Pass"
    };
    println!("case `{}`: expect {expects}", normalize_wgsl(&case.src));

    let parse = wgsl_parse::parse_str(&case.src);
    if case.fails && parse.is_ok() {
        println!("[FAIL] parse success but expected failure");
        panic!("test failed");
    } else if !case.fails && parse.is_err() {
        println!(
            "[FAIL] parse failure but expected success\n* error: {}",
            parse.unwrap_err(),
        );
        panic!("test failed");
    }
}
