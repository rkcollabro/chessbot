#!/bin/sh
# Builds chesscli from any revision's source tree: tools/ci_build.sh <src dir> <output> [extra compiler flags]
# Compiles every .cpp except the Windows GUI, so it works for revisions with different file lists.
set -e
src=$1; out=$2; shift 2
g++ -O3 -march=native -std=c++17 -pthread -DNDEBUG -w "$@" $(ls "$src"/*.cpp | grep -v gui.cpp) -o "$out"
echo "built $out"
