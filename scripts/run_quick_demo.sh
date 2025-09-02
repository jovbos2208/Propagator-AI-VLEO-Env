#!/usr/bin/env bash
set -euo pipefail

# Quick end-to-end test: build and run a minimal env_demo scenario.
# Usage: ./scripts/run_quick_demo.sh

here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
build_dir="$repo_root/cpp/build"

echo "[INFO] Repo: $repo_root"

if [ -d "$repo_root/.git" ]; then
  echo "[INFO] Initializing submodules (if any)..."
  git -C "$repo_root" submodule update --init --recursive || true
fi

mkdir -p "$build_dir"
cd "$build_dir"

echo "[INFO] Configuring CMake (Release)..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "[INFO] Building..."
cmake --build . -j

echo "[INFO] Running env_demo (1 orbit, aligned attitude @300 km)..."
./env_demo \
  --integrator dp54 \
  --align-to-velocity \
  --orbits 1 \
  --alt-min 300000 --alt-max 300000 \
  --incl-min 0 --incl-max 0 \
  --eta-limit 0 \
  --thrust-min 0 --thrust-max 0 \
  --impulse-budget 0 \
  --target-alt 300000 \
  --log-csv quick.csv

echo "[OK] env_demo finished. CSV: $build_dir/quick.csv"

