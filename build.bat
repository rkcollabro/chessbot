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
set CORE=src\board.cpp src\eval.cpp src\search.cpp src\tablebase.cpp

cl %CFLAGS% /Fobuild\ /Fe:chesscli.exe %CORE% src\cli.cpp || exit /b 1
cl %CFLAGS% /Fobuild\ /Fe:chessbot.exe %CORE% src\gui.cpp /link /SUBSYSTEM:WINDOWS user32.lib gdi32.lib || exit /b 1

echo Build OK: chessbot.exe (GUI), chesscli.exe (tests)
