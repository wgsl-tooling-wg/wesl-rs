#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/wesl.h"

// -- helpers
WeslStringMap create_string_map(const char** keys, const char** values, size_t len) {
    WeslStringMap map = {
        .keys = keys,
        .values = values,
        .len = len
    };
    return map;
}

WeslBoolMap create_bool_map(const char** keys, const bool* values, size_t len) {
    WeslBoolMap map = {
        .keys = keys,
        .values = values,
        .len = len
    };
    return map;
}

int main() {
    // print version
    const char* version = wesl_version();
    printf("WESL version: %s\n", version);

    // add some modules
    const char* modules[] = {
        "package::main",
        "package::utils"
    };

    const char* sources[] = {
        "import package::utils::add;\nfn some_fn_in_main() -> i32 { let a = add(1, 2); return a; } ",
        "fn add(a: i32, b: i32) -> i32 { return a + b; } "
    };

    // file map
    WeslStringMap file_map = create_string_map(modules, sources, 2);

    // setup compile options
    WeslCompileOptions options = {
        .mangler = WESL_MANGLER_NONE,
        .sourcemap = true,
        .imports = true,
        .condcomp = true,
        .generics = true,
        .strip = false,
        .lower = true,
        .validate = true,
        .naga = true,
        .lazy = false,
        .keep_root = true,
        .mangle_root = false
    };

    // setup features
    const char* feature_keys[] = {"debug"};
    bool feature_values[] = {true};
    WeslBoolMap features = create_bool_map(feature_keys, feature_values, 1);

    // compile
    printf("calling wesl_compile...\n");
    WeslResult result = wesl_compile(
        &file_map,
        "package::main",
        &options,
        NULL,   // omit keep array
        &features
    );

    if (result.success) {
        printf("Compilation successful!\n");
        printf("Output:\n%s\n", result.data);

        // evaluate
        const char* eval_modules[] = {
            "package::source"
        };

        const char* eval_sources[] = {
            "const my_const = 4; @const fn my_fn(v: u32) -> u32 { return v * 10; }",
        };

        WeslStringMap eval_file_map = create_string_map(eval_modules, eval_sources, 1);

        printf("\ncalling wesl_eval...\n");
        WeslCompileOptions eval_options = {0};
        WeslResult eval_result = wesl_eval(
            &eval_file_map,
            "package::source",
            "my_fn(my_const) + 2",
            &eval_options,
            &features
        );

        if (eval_result.success) {
            printf("Evaluation successful!\n");
            printf("Result: %s (expected: 42u)\n", eval_result.data);
            assert(strcmp(eval_result.data, "42u") == 0 && "wesl_eval produced unexpected result");
        } else {
            printf("Evaluation failed!\n");
            printf("%s\n", eval_result.error.message);
            if (eval_result.error.diagnostics_len > 0) {
                printf("Diagnostic: %s at %s (%u:%u)\n",
                    eval_result.error.diagnostics[0].title,
                    eval_result.error.diagnostics[0].file,
                    eval_result.error.diagnostics[0].span_start,
                    eval_result.error.diagnostics[0].span_end
                );
            }
        }

        // cleanup
        wesl_free_result(&eval_result);
    } else {
        printf("Compilation failed!\n");
        printf("%s\n", result.error.message);
        if (result.error.diagnostics_len > 0) {
            printf("Diagnostic: %s at %s (%u:%u)\n",
                result.error.diagnostics[0].title,
                result.error.diagnostics[0].file,
                result.error.diagnostics[0].span_start,
                result.error.diagnostics[0].span_end
            );
        }
    }

    // cleanup
    wesl_free_result(&result);
    
    return 0;
}
