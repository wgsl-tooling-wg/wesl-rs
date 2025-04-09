#![cfg_attr(not(test), allow(dead_code, unused_imports))]
#![feature(custom_test_frameworks)]
#![test_runner(datatest::runner)]

use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{self, Display},
};

use regex::Regex;
use serde::Deserialize;
use wesl::{
    syntax::{Expression, GlobalDeclaration, ModulePath, Statement, TranslationUnit},
    CompileOptions, EscapeMangler, NoMangler, VirtualResolver,
};

fn parse_test(input: &str) {
    let mut resolver = VirtualResolver::new();
    let root = ModulePath::from_path("/main");
    resolver.add_module(root.clone(), input.into());
    let mut options = CompileOptions::default();
    options.imports = true;
    options.condcomp = true;
    options.generics = false;
    options.strip = true;
    options.lower = true;
    options.validate = true;
    wesl::compile(&root, &resolver, &NoMangler, &options)
        .inspect_err(|err| eprintln!("[FAIL] {err}"))
        .expect("test failed");
}

#[datatest::files("webgpu-samples", {
  input in r"^.*\.wgsl$",
})]
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

#[derive(PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum SyntaxKind {
    Declaration,
    Statement,
    Expression,
}

#[derive(PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "kind")]
enum TestKind {
    Syntax {
        syntax: SyntaxKind,
    },
    Eval {
        eval: String,
        result: Option<String>, // must be None when expect is Fail, must be Some when expect is Pass
    },
    Context,
}

impl Display for TestKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestKind::Syntax { .. } => f.write_str("Syntax"),
            TestKind::Eval { .. } => f.write_str("Eval"),
            TestKind::Context => f.write_str("Context"),
        }
    }
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Expectation {
    Pass,
    Fail,
}

impl From<bool> for Expectation {
    fn from(value: bool) -> Self {
        if value {
            Self::Pass
        } else {
            Self::Fail
        }
    }
}

impl Display for Expectation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expectation::Pass => f.write_str("Pass"),
            Expectation::Fail => f.write_str("Fail"),
        }
    }
}

#[derive(Deserialize)]
struct Test {
    name: String,
    desc: String,
    #[serde(flatten)]
    kind: TestKind,
    code: String,
    expect: Expectation,
    note: Option<String>,
}

impl fmt::Display for Test {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[datatest::data("spec-tests/lit-type-inference.json")]
#[test]
fn spec_tests(case: Test) {
    json_case(case);
}

fn json_case(case: Test) {
    println!(
        "case: `{}`\n* desc: {}{}\n* kind: {}\n* expect: {}",
        case.name,
        case.desc,
        case.note
            .as_ref()
            .map(|n| format!(" (note: {n})"))
            .unwrap_or_default(),
        case.kind,
        case.expect
    );

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

// see schemas: https://github.com/wgsl-tooling-wg/wesl-testsuite/blob/main/src/TestSchema.ts
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WgslTestSrc {
    name: String,
    wesl_src: HashMap<String, String>,
    #[serde(default)]
    notes: Option<String>,
    #[allow(unused)]
    #[serde(default)]
    expected_wgsl: Option<String>,
    #[serde(default)]
    underscore_wgsl: Option<String>,
}
#[derive(Deserialize)]
struct ParsingTest {
    src: String,
    #[serde(default)]
    fails: bool,
}

#[allow(unused)]
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BulkTest {
    name: String,
    base_dir: String,
    exclude: Option<Vec<String>>,
    include: Option<Vec<String>>,
    glob_include: Option<Vec<String>>,
}

impl fmt::Display for WgslTestSrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Display for ParsingTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`", normalize_wgsl(&self.src))
    }
}

#[datatest::data("wesl-testsuite/src/test-cases-json/importSyntaxCases.json")]
fn wesl_testsuite_import_syntax_cases(case: ParsingTest) {
    let expects = if case.fails { "Fail" } else { "Pass" };
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

fn normalize_wgsl(wgsl: &str) -> String {
    let re = Regex::new(r"\s+").unwrap();
    re.replace_all(wgsl, " ").trim().to_string()
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
    let mut compile_options = CompileOptions::default();
    compile_options.strip = false;

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

#[datatest::data("wesl-testsuite/src/test-cases-json/importCases.json")]
fn wesl_testsuite_import_cases(case: WgslTestSrc) {
    testsuite_test(case);
}

#[datatest::data("wesl-testsuite/src/test-cases-json/conditionalTranslationCases.json")]
fn wesl_testsuite_conditional_translation_cases(case: WgslTestSrc) {
    testsuite_test(case);
}
