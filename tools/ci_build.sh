#!/bin/sh
# Builds chesscli from any revision's source tree: tools/ci_build.sh <src dir> <output> [extra compiler flags]
# Compiles every .cpp except the Windows GUI, so it works for revisions with different file lists.
# When the output's directory already holds nnue.bin and tb/, and the revision has the bench command, the
# build is profile-guided: an instrumented binary runs the bench, then the compiler optimizes for the
# measured hot paths. Same source, same search, faster. Otherwise it is a plain build.
set -e
src=$1; out=$2; shift 2
case $out in */*) ;; *) out=./$out ;; esac  # the PGO step runs "$out": a bare name would be looked up in PATH
compat=$(cd "$(dirname "$0")" && pwd)/compat  # <io.h> for revisions that predate the portable cli.cpp
files=$(ls "$src"/*.cpp | grep -v gui.cpp)
build() { g++ -O3 -march=native -std=c++17 -pthread -DNDEBUG -w -idirafter "$compat" "$@" $files -o "$out"; }
dir=$(dirname "$out")
if [ -f "$dir/nnue.bin" ] && [ -d "$dir/tb" ] && grep -q '"bench"' "$src/cli.cpp" 2>/dev/null; then
  prof=$(mktemp -d)
  build "$@" -fprofile-generate="$prof"
  "$out" bench 400000 > /dev/null
  build "$@" -fprofile-use="$prof" -fprofile-partial-training -Wno-missing-profile
  rm -rf "$prof"
  echo "built $out (PGO)"
else
  build "$@"
  echo "built $out"
fi
