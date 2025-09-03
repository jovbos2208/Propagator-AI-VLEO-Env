#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train_ppo --config src/rl_sat/configs/coupled.yaml --num-envs 8 --total-steps 200000 --normalize

