@echo off
setlocal enabledelayedexpansion

rem cargo build
rem NOTE: make sure you are using the GNU toolchain.
set RUSTFLAGS=--print=native-static-libs
cargo build --package wesl-c --features eval,generics --release

rem set paths
set WESL_LIB_PATH=../../../target/release
set INCLUDE_PATH=../include

rem build
clang ^
    simple.c ^
    -I%INCLUDE_PATH% ^
    -L%WESL_LIB_PATH% ^
    -lwesl_c ^
    -lkernel32 ^
    -lntdll ^
    -luserenv ^
    -lws2_32 ^
    -ldbghelp ^
    -o simple.exe

if %ERRORLEVEL% NEQ 0 (
    echo build failed
    exit /b 1
)
