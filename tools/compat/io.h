// Linux stand-in for the Windows <io.h> used by older revisions of src/cli.cpp, so tools/ci_build.sh can build them.
#pragma once
#include <unistd.h>
#define _isatty isatty
#define _fileno fileno
