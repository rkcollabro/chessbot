@echo off
setlocal
cd /d "%~dp0"

rem Usage: build.bat [chesscli output dir, default .] [nogui]

rem Locate the MSVC toolchain via vswhere and set up a 64-bit build environment.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo Could not find Visual Studio C++ build tools.
    exit /b 1
)
set "PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set "OUT=%~1"
if "%OUT%"=="" set "OUT=."
if not exist build\pgo mkdir build\pgo
if not exist "%OUT%" mkdir "%OUT%"
set CFLAGS=/nologo /std:c++17 /O2 /arch:AVX2 /EHsc /W3 /MT /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /utf-8
set CORE=src\board.cpp src\eval.cpp src\search.cpp src\tablebase.cpp src\book.cpp src\nnue.cpp

rem chesscli is built with profile-guided optimization: an instrumented build runs the bench, then
rem the linker lays out the code around the measured hot paths. Same source, same results, faster.
del /q build\pgo\*.pgc build\pgo\*.pgd chesscli!*.pgc 2>nul
cl %CFLAGS% /GL /Fobuild\pgo\ /Fe:chesscli-instr.exe %CORE% src\uci.cpp src\datagen.cpp src\cli.cpp /link /LTCG /GENPROFILE:PGD=build\pgo\chesscli.pgd /STACK:8388608 || exit /b 1
rem Runs from the repo root so it finds nnue.bin and tb\ (without the network it would profile the wrong eval).
.\chesscli-instr.exe bench 400000 >nul || exit /b 1
rem The profile lands next to the exe, named after the .pgd.
move /y chesscli!*.pgc build\pgo\ >nul || exit /b 1
link /nologo /LTCG /USEPROFILE:PGD=build\pgo\chesscli.pgd build\pgo\*.obj /OUT:"%OUT%\chesscli.exe" /STACK:8388608 || exit /b 1
del /q chesscli-instr.exe chesscli-instr.exp chesscli-instr.lib 2>nul

if /i "%~2"=="nogui" goto done
rc /nologo /fo build\resources.res src\resources.rc || exit /b 1
cl %CFLAGS% /Fobuild\ /Fe:chessbot.exe %CORE% src\gui.cpp build\resources.res /link /SUBSYSTEM:WINDOWS /STACK:8388608 user32.lib gdi32.lib gdiplus.lib ole32.lib || exit /b 1

:done
echo Build OK: %OUT%\chesscli.exe (PGO)
