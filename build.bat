@echo off
setlocal
cd /d "%~dp0"

rem Locate the MSVC toolchain via vswhere and set up a 64-bit build environment.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo Could not find Visual Studio C++ build tools.
    exit /b 1
)
set "PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

if not exist build mkdir build
set CFLAGS=/nologo /std:c++17 /O2 /EHsc /W3 /MT /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /utf-8
set CORE=src\board.cpp src\eval.cpp src\search.cpp src\tablebase.cpp src\book.cpp

cl %CFLAGS% /Fobuild\ /Fe:chesscli.exe %CORE% src\uci.cpp src\datagen.cpp src\cli.cpp /link /STACK:8388608 || exit /b 1
rc /nologo /fo build\resources.res src\resources.rc || exit /b 1
cl %CFLAGS% /Fobuild\ /Fe:chessbot.exe %CORE% src\gui.cpp build\resources.res /link /SUBSYSTEM:WINDOWS /STACK:8388608 user32.lib gdi32.lib gdiplus.lib ole32.lib || exit /b 1

echo Build OK: chessbot.exe (GUI), chesscli.exe (tests)
