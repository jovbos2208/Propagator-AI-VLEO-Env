#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train_sac --config src/rl_sat/configs/orbit_only.yaml --num-envs 4 --total-steps 100000 --normalize

