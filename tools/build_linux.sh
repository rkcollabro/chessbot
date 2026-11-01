#!/bin/sh
# Builds chesscli for Linux: tools/build_linux.sh <output> [source dir, default src]
set -e
out=$1; src=${2:-src}
g++ -O3 -march=native -std=c++17 -pthread -DNDEBUG -w \
    $src/board.cpp $src/eval.cpp $src/search.cpp $src/tablebase.cpp $src/book.cpp $src/nnue.cpp $src/uci.cpp $src/datagen.cpp \
    $src/cli.cpp -o "$out"
echo "built $out"
