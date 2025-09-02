#!/usr/bin/env bash
set -euo pipefail

# Quickstart: build C++ lib, set up venv, install deps, and launch training.
# Usage: scripts/quickstart.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "[1/4] Creating Python venv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[2/4] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[2.1/4] Installing Python requirements"
python -m pip install -r "${ROOT_DIR}/python/requirements.txt" || true

# Ensure torch is available for stable-baselines3 if not pulled as a dependency
python - <<'PY'
try:
    import torch  # noqa: F401
    print("Torch already installed")
except Exception:
    raise SystemExit(1)
PY
if [[ $? -ne 0 ]]; then
  echo "Torch not found; attempting CPU-only install..."
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch || true
fi

echo "[3/4] Building C++ shared library (libshuttlecock_env.so)"
mkdir -p "${ROOT_DIR}/cpp/build"
cmake -S "${ROOT_DIR}/cpp" -B "${ROOT_DIR}/cpp/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${ROOT_DIR}/cpp/build" -j

LIB_PATH="${ROOT_DIR}/cpp/build/libshuttlecock_env.so"
if [[ ! -f "${LIB_PATH}" ]]; then
  echo "Error: ${LIB_PATH} not found after build." >&2
  exit 2
fi

echo "[4/4] Starting training (2 envs, 4 episodes, 20 actions/orbit)"
cd "${ROOT_DIR}"
python3 train_shuttlecock.py \
  --config cpp/configs/shuttlecock_250km.json \
  --num-envs 2 \
  --episodes 4 \
  --per-orbit-steps 20 \
  --normalize \
  --n-steps 2048 \
  --batch-size 512

