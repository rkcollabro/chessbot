@echo off
setlocal
cd /d "%~dp0"

rem Usage: build_clang.bat [output dir, default clang-out]
rem chesscli built with clang-cl (LLVM) instead of MSVC: ThinLTO + profile-guided optimisation from the bench,
rem against the MSVC headers and libraries. Same source and search as build.bat; only the code generation differs.
set "LLVM=C:\Users\Randy\tools\clang+llvm-23.1.5-x86_64-pc-windows-msvc\bin"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set "OUT=%~1"
if "%OUT%"=="" set "OUT=clang-out"
if not exist "%OUT%" mkdir "%OUT%"
if not exist build\clangpgo mkdir build\clangpgo
del /q build\clangpgo\*.profraw build\clangpgo\*.profdata 2>nul
set SRC=src\board.cpp src\eval.cpp src\search.cpp src\tablebase.cpp src\book.cpp src\nnue.cpp src\largepages.cpp src\uci.cpp src\datagen.cpp src\cli.cpp
set CFLAGS=/nologo /std:c++17 /O2 /EHsc /MT /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /utf-8 /arch:AVX2 -march=native -flto=thin -w

rem 1. instrumented build, run from the repo root so it finds nnue.bin and tb\
"%LLVM%\clang-cl.exe" %CFLAGS% -fprofile-instr-generate="build\clangpgo\bench-%%p.profraw" %SRC% /Fe:chesscli-clang-instr.exe -fuse-ld=lld /link /STACK:8388608 || exit /b 1
.\chesscli-clang-instr.exe bench 400000 >nul || exit /b 1
"%LLVM%\llvm-profdata.exe" merge -o build\clangpgo\bench.profdata build\clangpgo\*.profraw || exit /b 1
rem 2. optimised build using the profile
"%LLVM%\clang-cl.exe" %CFLAGS% -fprofile-instr-use="build\clangpgo\bench.profdata" %SRC% /Fe:"%OUT%\chesscli.exe" -fuse-ld=lld /link /STACK:8388608 || exit /b 1
del /q chesscli-clang-instr.exe chesscli-clang-instr.lib chesscli-clang-instr.exp 2>nul
echo Build OK: %OUT%\chesscli.exe (clang PGO)
