#!/usr/bin/env bash
set -euo pipefail

# Train PPO with Shuttlecock adaptor in angles+thrust mode
# 4 parallel environments, 10 orbits/episode, 1 action/orbit, ~50 episodes total

python -m src.training.train_ppo \
  --config src/rl_sat/configs/angles_thrust_10orbits_1step.yaml \
  --num-envs 4 \
  --total-steps 500 \
  --normalize \
  --tb-logdir runs/tb \
  --use-shuttlecock

