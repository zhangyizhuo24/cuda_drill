#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

echo ""
echo "=== CUDA kernels ==="
"$BUILD_DIR/transpose_test" "$@"

echo ""
echo "=== PyTorch baseline ==="
python3 "$SCRIPT_DIR/bench_pytorch.py"
