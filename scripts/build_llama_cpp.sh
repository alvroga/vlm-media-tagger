#!/usr/bin/env bash
# Build the pinned llama.cpp `llama-server` (static, Metal) into
# vendor/llama_cpp/llama-server — the inference engine for MiniCPM-V 4.7,
# whose vision projector (minicpmv4_7) Ollama doesn't support yet.
#
#   scripts/build_llama_cpp.sh                 # clone + build the pinned commit
#   LLAMA_CPP_SRC=~/src/llama.cpp scripts/build_llama_cpp.sh   # reuse a local clone
#
# Static (BUILD_SHARED_LIBS=OFF) so the result is ONE self-contained binary:
# a shared build links @rpath dylibs whose rpath is the build dir, which breaks
# the moment it is copied into an app bundle. Metal shaders are embedded. The
# built-in web UI is off (its asset step fails from a temp dir; we only use the API),
# and so is OpenSSL (it links Homebrew dylibs other Macs lack; we only use localhost).
# Changing LLAMA_CPP_COMMIT is an engine upgrade: rerun the harness canary.
set -euo pipefail
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-b76b133}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/vendor/llama_cpp"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/llama-cpp-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [ -n "${LLAMA_CPP_SRC:-}" ]; then
  git clone -q "$LLAMA_CPP_SRC" "$WORK/src"
else
  git clone -q https://github.com/ggml-org/llama.cpp "$WORK/src"
fi
git -C "$WORK/src" checkout -q "$LLAMA_CPP_COMMIT"
FULL="$(git -C "$WORK/src" rev-parse HEAD)"

cmake -S "$WORK/src" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_UI_GZIP=OFF \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 >"$WORK/cmake.log"
cmake --build "$WORK/build" --target llama-server -j "$(sysctl -n hw.ncpu)" >"$WORK/build.log" || { tail -30 "$WORK/build.log"; exit 1; }

mkdir -p "$OUT"
cp "$WORK/build/bin/llama-server" "$OUT/llama-server"
cp "$WORK/src/LICENSE" "$OUT/LICENSE"
cat >"$OUT/VENDORED.md" <<MD
# Vendored llama.cpp llama-server

Built by scripts/build_llama_cpp.sh — static, Metal, Release.
Source: https://github.com/ggml-org/llama.cpp @ $FULL
Built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(sw_vers -productVersion) $(uname -m)
License: MIT (LICENSE). Not committed to git (binary); rebuild with the script.
MD
echo "built $OUT/llama-server ($(du -h "$OUT/llama-server" | cut -f1)) from $FULL"
otool -L "$OUT/llama-server" | tail -n +2
