#!/bin/bash
set -e

# cargo build
RUSTFLAGS="--print=native-static-libs" cargo build --package wesl-c --features eval,generics --release

# set paths
WESL_LIB_PATH="../../../target/release"
INCLUDE_PATH="../include"

# build
clang \
    simple.c \
    -I"${INCLUDE_PATH}" \
    -L"${WESL_LIB_PATH}" \
    -lwesl_c \
    -o simple
