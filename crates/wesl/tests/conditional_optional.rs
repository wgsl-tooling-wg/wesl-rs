use wesl::{Features, pass::condcomp, syntax::TranslationUnit};

#[test]
fn conditional_break_if() {
    let mut module: TranslationUnit = "fn enabled() { loop { continuing { @if(true) break if true; } } } fn disabled() { loop { continuing { @if(false) break if true; } } }".parse().unwrap();
    let expected: TranslationUnit = "fn enabled() { loop { continuing { break if true; } } } fn disabled() { loop { continuing { } } }".parse().unwrap();

    condcomp(&mut module, &Features::default()).unwrap();

    assert_eq!(module.to_string(), expected.to_string());
}

#[test]
fn conditional_continuing() {
    let mut module: TranslationUnit = "fn enabled() { loop { @if(true) continuing { } } } fn disabled() { loop { @if(false) continuing { } } }".parse().unwrap();
    let expected: TranslationUnit =
        "fn enabled() { loop { continuing { } } } fn disabled() { loop { } }"
            .parse()
            .unwrap();

    condcomp(&mut module, &Features::default()).unwrap();

    assert_eq!(module.to_string(), expected.to_string());
}

#[test]
fn conditional_for_initializer_and_update() {
    let mut module: TranslationUnit = "fn enabled() { for (@if(true) var index = 0; index < 4; @if(true) index++) { } } fn disabled() { for (@if(false) var index = 0; false; @if(false) index++) { } }".parse().unwrap();
    let expected: TranslationUnit = "fn enabled() { for (var index = 0; index < 4; index++) { } } fn disabled() { for (; false; ) { } }".parse().unwrap();

    condcomp(&mut module, &Features::default()).unwrap();

    assert_eq!(module.to_string(), expected.to_string());
}
