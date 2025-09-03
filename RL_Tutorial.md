RL Satellite Training Tutorial

This tutorial shows how to train reinforcement learning agents for VLEO satellite station keeping and attitude control using this repository. It covers setup, choosing the simulation and action interface, running parallel training, logging to TensorBoard, and evaluating results.

Prerequisites
- Python 3.10+ recommended
- Optional: Shuttlecock C++ VLEO aerodynamics simulator and Python bindings (python.shuttlecock_env) if you want physics-accurate angles control

1) Install
- Create and activate a virtual environment (recommended):
  - python -m venv .venv
  - source .venv/bin/activate
- Install dependencies:
  - pip install -r requirements.txt

2) Choose Simulation and Actions
You can train against either a dummy dynamics model (fast smoke-testing) or the Shuttlecock C++ simulator.

- Action types (set in YAML under env.action_type):
  - rtn_torque (default): 6-D action vector [thrust_rtn (3), torque_cmd (3)]
  - angles_only: 2-D [eta1, eta2] wing angles in radians
  - angles_thrust: 3-D [eta1, eta2, thrust_N] (thrust gated in eclipse if enabled)

- Simulation backend selection:
  - If action_type is angles_only or angles_thrust → the training/eval scripts auto-enable the Shuttlecock adaptor.
  - Otherwise, pass --use-shuttlecock to force C++ sim; omit to use DummyPropagator.

Configs
- Ready-to-use configs live under src/rl_sat/configs/:
  - default.yaml (rtn_torque)
  - attitude_only.yaml, orbit_only.yaml, coupled.yaml
  - angles_only.yaml, angles_thrust.yaml

3) Train
PPO example (parallel on 8 processes, with TensorBoard logging):
- python -m src.training.train_ppo \
    --config src/rl_sat/configs/coupled.yaml \
    --num-envs 8 \
    --total-steps 2000000 \
    --normalize \
    --tb-logdir runs/tb

SAC example (angles + thrust with Shuttlecock, parallel on 8 processes):
- python -m src.training.train_sac \
    --config src/rl_sat/configs/angles_thrust.yaml \
    --num-envs 8 \
    --total-steps 2000000 \
    --normalize \
    --tb-logdir runs/tb \
    --use-shuttlecock

Notes
- --num-envs > 1 uses SubprocVecEnv to run multiple environments in parallel (one per CPU core, up to your machine’s capacity).
- --normalize wraps the vectorized env with VecNormalize for observation/reward normalization.
- Models save to runs/ppo_rlsat.zip or runs/sac_rlsat.zip.
- For Shuttlecock, ensure python.shuttlecock_env is importable; otherwise stick to DummyPropagator or rtn_torque.

4) Monitor Training with TensorBoard
- Start TensorBoard and point it to the chosen log directory:
  - tensorboard --logdir runs/tb
- In scripts above, tb logs are enabled by passing --tb-logdir. PPO/SAC will write per-step training metrics under that folder.

5) Evaluate Policies and Collect Metrics
Run evaluation on the desired config + checkpoint:
- python -m src.training.eval_policy \
    --config src/rl_sat/configs/coupled.yaml \
    --ckpt runs/ppo_rlsat.zip \
    --episodes 10 \
    --outdir runs/eval

For angles-based control using Shuttlecock, evaluation auto-selects the C++ backend when action_type is angles_only/angles_thrust; otherwise pass --use-shuttlecock to force it.

Outputs
- JSON summary at runs/eval/eval_metrics.json containing episode returns, steps, and statistics such as mean/final/max of:
  - theta_deg, da_m, e, dr_rtn_m, dv_rtn_mps, cum_dv_mps

6) Key Configuration Options
- env.action_type: rtn_torque | angles_only | angles_thrust
- env.gate_thrust_in_eclipse: true|false to zero thrust during eclipse
- env.eta_limit_rad: max wing angle (rad)
- env.thrust_max_N: thrust cap for angles_thrust
- env.obs_scales: per-feature scaling for observation normalization
- reward_weights: weights for orbit, attitude, effort, and smoothness terms
- Termination thresholds:
  - env.min_perigee_km, env.max_att_err_deg, optional env.soc_min

7) Troubleshooting
- No module named rl_sat when running tests: ensure tests/conftest.py is present or run with PYTHONPATH=src.
- C++ sim import fails: ensure Shuttlecock bindings are built and python.shuttlecock_env is importable; otherwise use DummyPropagator or rtn_torque.
- TensorBoard shows no data: make sure you passed --tb-logdir to the training script and pointed TensorBoard to that directory.

8) Run Tests
- pytest -q
This runs sanity tests for env shapes, rewards, safety filter, and termination logic.

9) Quick Recipes
- Attitude-only warmup (no thrust):
  - python -m src.training.train_ppo --config src/rl_sat/configs/attitude_only.yaml --num-envs 4 --total-steps 300000 --tb-logdir runs/tb
- Angles-only control with C++ sim (wing steering only):
  - python -m src.training.train_ppo --config src/rl_sat/configs/angles_only.yaml --num-envs 4 --total-steps 300000 --tb-logdir runs/tb --use-shuttlecock
- Coupled orbit+attitude with RTN thrust and torques:
  - python -m src.training.train_sac --config src/rl_sat/configs/coupled.yaml --num-envs 8 --total-steps 2000000 --normalize --tb-logdir runs/tb

That’s it — you’re set to train, visualize, and evaluate RL agents for VLEO satellite control.

