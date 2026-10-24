#!/bin/bash
# Rebuilds unmix-stretch (a small command-line wrapper around Rubber Band 4.0.0, GPL-2.0-or-later).
# Only needed when upgrading Rubber Band; the compiled arm64 binary is committed next to this file.
# Source: https://breakfastquay.com/files/releases/rubberband-4.0.0.tar.bz2
set -euo pipefail
cd "$(dirname "$0")"
RB="${1:?usage: build.sh /path/to/rubberband-4.0.0}"
clang++ -std=c++17 -O2 -arch arm64 -DNDEBUG -DNO_THREADING -DUSE_BQRESAMPLER -I "$RB" -I "$RB/src" \
  unmix_stretch.cpp "$RB/single/RubberBandSingle.cpp" -framework Accelerate -o unmix-stretch
