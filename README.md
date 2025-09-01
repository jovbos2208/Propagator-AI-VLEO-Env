# Shuttlecock VLEO Propagator & AI Environment

A C++ project for simulating and controlling a “shuttlecock” satellite in Very Low Earth Orbit (VLEO). It combines:

- High‑fidelity 6‑DOF dynamics in ECI with gravity (μ + J2–J4).
- Aerodynamic force/torque from a geometry‑based black‑box (C++ implementation).
- Atmospheric density/temperature from NRLMSIS 2.1 (Fortran, wrapped for C++).
- An RL‑ready environment (orbit‑length or intra‑orbit control), CLI tools, C API, and a Python wrapper for training with Stable‑Baselines3.

See `FirstStart.md` for the fastest way to train a model for 100 episodes. See `Tutorial.md` for an in‑depth guide and recommended settings.

## Features

- Frames & Transforms: ECI↔ECEF (GMST), ECEF→Geodetic (WGS‑84), body/inertial rotations (quaternions).
- Forces & Torques:
  - Gravity: point‑mass + J2–J4 zonal harmonics.
  - Aerodynamics: black‑box C++ (vleo aerodynamics) returning total F/T in body frame.
  - Thrust: along +X body, min/max clamp and total impulse budget.
- Atmosphere: NRLMSIS 2.1 (density, temperature; optional species → mean molecular mass), latitude/longitude/altitude per step.
- Integrators: RK4 (fixed) and Dormand–Prince 5(4) (adaptive).
- RL Environment:
  - State: r, v, q, ω, altitude, ρ, AoA/AoS, roll; actions: eta1/eta2 (±), thrust (0..Tmax).
  - Rewards: altitude & attitude tracking, spin penalty, effort penalty, dwell bonuses.
  - Termination: reentry/escape (rotation is penalized, not terminating).
  - Intra‑orbit actions: fixed time (e.g., 5 s) or every N integrator substeps.
- Tooling: CLI demos, per‑step debug CSV (lat/lon/alt/ω), JSON config, C API, Python wrapper.

## Repository Layout

- `cpp/`
  - `include/`, `src/`: core env, dynamics, gravity, frames, aero adapter, NRLMSIS wrapper, C API.
  - `tools/`: `env_demo`, `propagate_demo` and geometry tools.
  - `configs/`: example JSON configs (e.g., `shuttlecock_250km.json`).
- `nrlmsis2.1_cpp/Fortran/`: NRLMSIS 2.1 Fortran sources (built as `msis21`).
- `python/`: ctypes wrapper (`shuttlecock_env.py`) and packaging stubs.
- `train_shuttlecock.py`: ready‑to‑run PPO training script.
- `FirstStart.md`: 100‑episode quick start.
- `Tutorial.md`: full training guide and recommendations.

## Build

Prerequisites: CMake ≥ 3.10, C++17 compiler, gfortran, Eigen3

```
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Targets:
- `env_demo`, `propagate_demo` (CLI)
- `libvleo-aerodynamics-core.a` (static lib)
- `libmsis21.so` (NRLMSIS 2.1, Fortran)
- `libshuttlecock_env.so` (C API for Python)

## Demos

- Propagator sanity demo:
```
./propagate_demo
```
- Environment demo (orbit by orbit or intra‑orbit):
```
# Orbit-by-orbit, DP54, align attitude, log
./env_demo --integrator dp54 --align-to-velocity --orbits 2 --log-csv run.csv

# Intra-orbit control every 5 s
./env_demo --integrator dp54 --align-to-velocity --control-dt 5 --log-csv run.csv --debug-csv debug_steps.csv

# Intra-orbit control every 20 substeps
./env_demo --integrator dp54 --align-to-velocity --control-substeps 20 --log-csv run.csv
```

Common flags:
- `--config cpp/configs/shuttlecock_250km.json` (recommended)
- `--alt-min/--alt-max`, `--incl-min/--incl-max` for initial orbit ranges
- `--eta-limit`, `--thrust-min/--thrust-max`, `--impulse-budget`
- `--w-alt/--w-ao/--w-roll/--w-effort/--w-dwell-alt/--w-dwell-att`, bands via `--alt-band`, `--aoa-band`, `--aos-band`, `--roll-band`
- `--align-to-velocity` to start with AoA/AoS/Roll near zero

## Configuration

JSON config files (flat keys) define integrator, initial ranges, target altitude, action limits, rewards, bands, and space weather. See `cpp/configs/shuttlecock_250km.json` and `Tutorial.md` for details.

## Training (Python)

1) Build the C++ shared lib (see Build).
2) Python setup:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip -r python/requirements.txt
export PYTHONPATH=$PWD/python:$PYTHONPATH
```
3) Run the included training script (PPO, 100 episodes by default):
```
python train_shuttlecock.py --episodes 100 --control-dt 5 --config cpp/configs/shuttlecock_250km.json --save checkpoints/shuttlecock_ppo_v1
```
4) Resume later:
```
python train_shuttlecock.py --episodes 100 --control-dt 5 --config cpp/configs/shuttlecock_250km.json --resume checkpoints/shuttlecock_ppo_v1 --save checkpoints/shuttlecock_ppo_v1
```

Programmatic access (ctypes):
- `python/shuttlecock_env.py` exposes `ShuttlecockEnvC` and a minimal `GymWrapper`.
- Control cadence:
  - Time‑based: `step_duration(eta1, eta2, thrust, seconds)`
  - Substeps: `step_substeps(eta1, eta2, thrust, N)`
- Logs: `--log-csv` aggregates per action window; `--debug-csv` writes per integrator substep (jd_utc, alt, lat, lon, ωx, ωy, ωz).

## Physics & Rewards (Summary)

- Gravity: μ + J2, J3, J4 (aligned to ECI z axis).
- Atmosphere: NRLMSIS 2.1 (density/temperature; winds set to zero here).
- Aerodynamics: black‑box geometry routine returning F/T in body frame; AoA/AoS computed from relative wind in body.
- Rewards: altitude error, AoA/AoS error, roll error, spin penalty (|ωx|+|ωy|+|ωz|), and a small control effort penalty. Dwell bonuses reward time spent inside altitude/attitude bands.
- Termination: only reentry (<80 km) or escape; rotation is penalized but not fatal.

## Notes

- For realistic operations in VLEO, consider randomizing space weather (`f107a`, `f107`, `ap`) across episodes.
- Start with `--align-to-velocity` to avoid immediate aero torques; tighten bands as policy improves.
- Use DP54 for adaptive accuracy; RK4 is available for fast prototyping.

## Acknowledgements

- NRLMSIS 2.1 (Naval Research Laboratory) — Fortran sources integrated via a small C shim.
- Eigen for linear algebra; tinyobjloader for geometry.

## License

This repository includes third‑party code (NRLMSIS, Eigen, tinyobjloader) under their respective licenses. See their files for details.

