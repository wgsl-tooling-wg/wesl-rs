// cargo run -- eval "foo()" --file samples/eval_shadowing.wesl --no-strip --no-lower

var x = 0; // even if this is not a const declaration...
// const x = 0; // ...this should not be allowed, duplicate declaration

override s: array<S>; 
struct S { x: u32 }

@const
fn foo() -> i32 {
    return x;
}
