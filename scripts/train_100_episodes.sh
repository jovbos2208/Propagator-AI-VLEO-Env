#!/usr/bin/env bash
set -euo pipefail

# Kick off a 100-episode PPO training run using the Python wrapper.
# Usage: ./scripts/train_100_episodes.sh

here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
build_dir="$repo_root/cpp/build"

echo "[INFO] Repo: $repo_root"

# Ensure C++ lib is built
if [ ! -f "$build_dir/libshuttlecock_env.so" ]; then
  echo "[INFO] Building C++ shared library..."
  mkdir -p "$build_dir"
  cd "$build_dir"
  cmake .. -DCMAKE_BUILD_TYPE=Release
  cmake --build . -j
fi

# Python deps hint
if ! python3 -c 'import numpy, gymnasium, stable_baselines3' >/dev/null 2>&1; then
  echo "[WARN] Python deps missing. Consider:"
  echo "       python3 -m venv .venv && source .venv/bin/activate && pip install -U pip -r python/requirements.txt"
fi

export PYTHONPATH="$repo_root/python:$PYTHONPATH"

echo "[INFO] Starting training: 100 episodes, control-dt=5s"
cd "$repo_root"
python3 train_shuttlecock.py \
  --episodes 100 \
  --control-dt 5 \
  --config cpp/configs/shuttlecock_250km.json \
  --save checkpoints/shuttlecock_ppo_v1

echo "[OK] Training finished. Model at checkpoints/shuttlecock_ppo_v1"

