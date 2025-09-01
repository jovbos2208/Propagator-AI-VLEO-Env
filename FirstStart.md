# First Start: Train a Shuttlecock AI (100 Episodes)

This quick‑start shows how to build, configure, and launch training for 100 episodes using Stable‑Baselines3 (PPO). It uses the C++ propagator, NRLMSIS2.1 atmosphere, and the shared C API + Python ctypes wrapper included in this repo.

## 1) Build the C++ Project

- Requirements: CMake ≥ 3.10, C++17 compiler, gfortran, Eigen3
- Build:
  - `cd cpp/build`
  - `cmake .. -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build . -j`

This builds:
- `libshuttlecock_env.so` (C API for Python)
- Tools: `env_demo`, `propagate_demo`

## 2) Recommended Config

A sensible training config for 250 km is provided:
- `cpp/configs/shuttlecock_250km.json`
  - Target 250 km, thrust 0..0.05 N, impulse budget 200 Ns/episode
  - Wing limits ±30°, reward weights and bands tuned for VLEO

## 3) Python Environment

- Create venv and install deps:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -U pip numpy gymnasium stable-baselines3`
- Add the Python wrapper to `PYTHONPATH` (from repo root):
  - `export PYTHONPATH=$PWD/python:$PYTHONPATH`

## 4) Minimal Training Script (train_shuttlecock.py)

Save the following as `train_shuttlecock.py` in the repo root.

```
import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from python.shuttlecock_env import ShuttlecockEnvC, GymWrapper

class ShuttleGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path, control_dt=5.0, align_to_velocity=True):
        super().__init__()
        self.envc = ShuttlecockEnvC(
            config_path=config_path.encode("utf-8"),
            align_to_velocity=align_to_velocity,
            integrator="dp54",
        )
        self.inner = GymWrapper(self.envc, control_dt=control_dt, use_substeps=False)
        # Observation: [r(3), v(3), q(4), w(3), alt, rho, AoA, AoS, roll] -> 18 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        # Actions in physical units: [eta1(rad), eta2(rad), thrust(N)]
        self.action_space = spaces.Box(
            low=np.array([-np.pi/6, -np.pi/6, 0.0], dtype=np.float32),
            high=np.array([+np.pi/6, +np.pi/6, 0.05], dtype=np.float32),
            dtype=np.float32,
        )
        self.steps_per_episode = None
        self.step_count = 0
        self.control_dt = control_dt

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = 0
        obs = self.inner.reset(seed=seed, jd0_utc=2451545.0)
        self.step_count = 0
        # ~10 orbits per episode, LEO orbit ~5400s, control_dt=5s => ~10800 steps/episode
        orbits_per_episode = 10
        self.steps_per_episode = int(orbits_per_episode * (5400.0 / self.control_dt))
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        self.step_count += 1
        if self.step_count >= self.steps_per_episode:
            truncated = True
        return obs, reward, terminated, truncated, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpp/configs/shuttlecock_250km.json")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--control-dt", type=float, default=5.0)
    ap.add_argument("--save", default="checkpoints/shuttlecock_ppo_v1")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    env = Monitor(ShuttleGym(args.config, control_dt=args.control_dt))
    steps_per_episode = int(10 * (5400.0 / args.control_dt))
    total_timesteps = steps_per_episode * args.episodes

    if args.resume:
        model = PPO.load(args.resume, env=env)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/")
        model.learn(total_timesteps=total_timesteps)

    model.save(args.save)
    print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
```

## 5) Launch a 100‑Episode Training Run

- Activate venv and set `PYTHONPATH`:
  - `source .venv/bin/activate`
  - `export PYTHONPATH=$PWD/python:$PYTHONPATH`
- Run training (100 episodes, action every 5 s):
  - `python train_shuttlecock.py --episodes 100 --control-dt 5 --config cpp/configs/shuttlecock_250km.json --save checkpoints/shuttlecock_ppo_v1`

## 6) Resume Training Later

- Continue training where you left off:
  - `python train_shuttlecock.py --episodes 100 --control-dt 5 --config cpp/configs/shuttlecock_250km.json --resume checkpoints/shuttlecock_ppo_v1 --save checkpoints/shuttlecock_ppo_v1`

Notes
- You can switch to substep‑based cadence (instead of 5 s) using the `GymWrapper` option `use_substeps=True, substeps=20`, but then adjust episode length accordingly.
- For deeper background, see `Tutorial.md` for reward details, recommended bands/weights, and CLI usage (`env_demo`).

