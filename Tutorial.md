# Shuttlecock AI Training Tutorial (VLEO ~250 km)

This tutorial walks you through training an AI controller to keep the “shuttlecock” satellite stable around 250 km altitude, with efficient fuel (impulse) use. It uses the provided C++ propagator, aerodynamic black‑box, and the `env_demo` CLI as a reference runner.

## Build The Project

- Requirements: CMake ≥ 3.10, a C++17 compiler, gfortran (for NRLMSIS2.1), Eigen3
- Build:
  - `cd cpp/build`
  - `cmake .. -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build . -j`

This produces the tools `env_demo`, `propagate_demo`, and test/geometry tools.

## Environment Overview

- State: `r_ECI, v_ECI, q_BI, w_B`; atmosphere queried via NRLMSIS2.1 (lat/lon/alt from ECI→ECEF)
- Forces/Torques: gravity μ + J2–J4, aero F/T from black‑box, thrust along +X_B
- Integrators: fixed RK4, adaptive Dormand–Prince 5(4) (DP54)
- Rewards (time‑averaged over each control window):
  - Altitude tracking (negative): `-w_alt * |alt − target|`
  - Attitude tracking (negative): `-w_att_ao * (|AoA|+|AoS|) − w_att_roll * |roll_err|`
  - Spin penalty (negative): `-w_spin * (|ωx| + |ωy| + |ωz|)`
  - Effort penalty (negative): `-w_effort * (|eta1|+|eta2| + thrust/Tmax)`
  - Dwell bonus (positive): `w_dwell_alt * frac_alt_band + w_dwell_att * frac_att_band`
- Termination: reentry (< 80 km), escape (> 20×Re), or impulse budget exhausted (if >0). Rotation does not terminate (penalized instead).

## Recommended Training Setup (250 km)

Mission‑typical values and bands for stable training:

- Vehicle and control
  - Mass: 5 kg; Inertia: `diag(0.05, 0.05, 0.05) kg m²`
  - Wing deflections: `eta1, eta2 ∈ [−30°, +30°]`
  - Thrust: `[0, 0.05] N` along +X_B; total impulse budget per episode: `200 Ns`
- Target & bands
  - Target altitude: 250 km
  - Altitude band: ±500 m
  - AoA band: ±3°, AoS band: ±3°, Roll band: ±3°
- Rewards (weights)
  - `w_alt = 1e-6`, `w_att_ao = 1.0`, `w_att_roll = 1.0`, `w_spin = 0.1`, `w_effort = 1e-2`
  - `w_dwell_alt = 1.0`, `w_dwell_att = 1.0`
- Space weather (moderate; can randomize per episode)
  - `f107a ≈ 140`, `f107 ≈ 135`, `ap ≈ [10,8,8,8,8,8,8]`
- Initial conditions (randomized per episode)
  - Altitude: 245–255 km, Inclination: 0–98°; RAAN/ArgPeri/True anomaly uniform in [0, 2π)
  - Align attitude to velocity: enabled (AoA/AoS/Roll start near 0)
- Integrator & action cadence
  - DP54 (rtol=atol=1e-6; min_dt=1e-4; max_dt=10.0)
  - Action cadence: every 5 s (time‑based) or every 20 integrator substeps (substep‑based)
- Episodes and horizon
  - A sensible starting point: 1,000 episodes × 10 orbits per episode (adjust for compute)

## Example Config JSON

Place this file at `cpp/configs/shuttlecock_250km.json` (already added below).

```
{
  "integrator": "dp54",
  "episodes": 1000,
  "orbits": 10,
  "seed": 12345,
  "dt": 1.0,
  "alt_min": 245000,
  "alt_max": 255000,
  "incl_min_deg": 0,
  "incl_max_deg": 98,
  "target_altitude": 250000,
  "eta_limit_deg": 30,
  "thrust_min": 0.0,
  "thrust_max": 0.05,
  "impulse_budget": 200.0,
  "w_alt": 1e-6,
  "w_ao": 1.0,
  "w_roll": 1.0,
  "w_effort": 1e-2,
  "w_dwell_alt": 1.0,
  "w_dwell_att": 1.0,
  "alt_band": 500,
  "aoa_band_deg": 3.0,
  "aos_band_deg": 3.0,
  "roll_band_deg": 3.0,
  "omega_band_deg_s": 5.0,
  "f107a": 140,
  "f107": 135,
  "ap": [10,8,8,8,8,8,8]
}
```

## Running The Environment

Time‑based control (action update every 5 seconds):

```
./env_demo \
  --config cpp/configs/shuttlecock_250km.json \
  --align-to-velocity \
  --control-dt 5 \
  --log-csv train_log.csv \
  --debug-csv debug_steps.csv
```

Substep‑based control (action update every 20 integrator substeps):

```
./env_demo \
  --config cpp/configs/shuttlecock_250km.json \
  --align-to-velocity \
  --control-substeps 20 \
  --log-csv train_log.csv \
  --debug-csv debug_steps.csv
```

- `train_log.csv`: per action window (per 5 s or per N substeps): reward breakdown, dwell fractions, substeps used, remaining impulse, and final observation.
- `debug_steps.csv`: per integrator substep: `jd_utc, alt, lat, lon, ωx, ωy, ωz`.

## Hooking Up An AI (save/resume)

You have two practical integration paths for training and checkpointing.

1) Python RL (recommended)
- Wrap the C++ environment with a Python binding (e.g., pybind11) exposing reset/step for either time‑based or substep‑based control. Map observations/actions and rewards exactly as in `env_demo`.
- Train with a standard RL library (e.g., Stable‑Baselines3 PPO/SAC):

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Make your pybind11-wrapped env: step(action) -> (obs, reward, terminated, truncated, info)
# Observation: concatenate r,v,q,w, altitude, rho, AoA, AoS, roll (normalized)
# Action: [eta1, eta2, thrust_norm] with bounds [-1,1], scale to limits in the wrapper

env = make_shuttlecock_env(config_path="cpp/configs/shuttlecock_250km.json",
                           control_dt=5.0, align_to_velocity=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/")
model.learn(total_timesteps=20_000_000)          # train
model.save("checkpoints/shuttlecock_ppo_v1")     # save

# Resume training later
model = PPO.load("checkpoints/shuttlecock_ppo_v1", env=env)
model.learn(total_timesteps=10_000_000, reset_num_timesteps=False)
```

- Checkpointing: save every N environment steps or when evaluation reward improves. Keep the config JSON alongside the model (for reproducibility) and record RNG seeds.

2) Process‑level loop (no bindings)
- As a simpler baseline, you can run `env_demo` in short control windows from Python (subprocess), read back logs, and emit the next action. This is slower and intended only as a quick prototype.

## Checkpointing & Resuming Best Practices
- Save policy weights and optimizer state regularly (e.g., every 100k–500k steps) to `checkpoints/` with the config JSON copied to the same folder.
- Use deterministic seeding for episodes (`--seed` and/or per‑episode seeds) so experiments are reproducible.
- When resuming, set `reset_num_timesteps=False` (SB3) to continue training curves.
- Version your reward weights, bands, and action limits; changing them mid‑training effectively changes the task.

## Troubleshooting & Tips
- If the agent burns too much impulse: increase `w_effort`, reduce `thrust_max`, or enlarge bands slightly.
- If attitude drifts: increase `w_att_ao` / `w_att_roll` or shorten control cadence to 2–3 s.
- If learning is unstable at start: begin with larger bands (e.g., AoA/AoS ±5°, roll ±5°, alt ±1–2 km), then tighten (curriculum).
- Randomize space weather moderately across episodes to improve robustness.

## Files Added By This Tutorial
- `cpp/configs/shuttlecock_250km.json` — recommended config
- This `Tutorial.md` — end‑to‑end training guide

Happy training!

