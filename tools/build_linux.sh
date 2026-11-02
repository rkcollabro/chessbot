#!/bin/sh
# Builds chesscli for Linux: tools/build_linux.sh <output> [source dir, default src]
# Profile-guided when the output's directory has nnue.bin and tb/ (see ci_build.sh).
set -e
sh "$(dirname "$0")/ci_build.sh" "${2:-src}" "$1"
