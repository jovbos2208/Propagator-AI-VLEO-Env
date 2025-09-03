#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train_ppo --config src/rl_sat/configs/attitude_only.yaml --num-envs 4 --total-steps 100000 --normalize

