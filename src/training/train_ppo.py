from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Ensure 'src' is on sys.path when running as a module
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_sat.envs.satellite_env import RLSatEnv
from rl_sat.sim.interfaces import DummyPropagator
from rl_sat.sim.shuttlecock_adapter import ShuttlecockPropagator
from .callbacks import ProgressBarCallback, LogPreActionLLA, EpisodeEndTB


def make_env(
    cfg: dict,
    use_shuttle: bool,
    shuttle_cfg_path: str | None,
):
    def _thunk():
        # Ensure env-specific debug configuration is present
        cfg.setdefault("debug", {})
        cfg["debug"].update({
            "pre_action": True,
            "every": 1,
            "print": True,
        })
        if use_shuttle:
            prop = ShuttlecockPropagator(
                config_path=shuttle_cfg_path or "cpp/configs/shuttlecock_250km.json",
                control_dt=float(cfg["env"].get("dt", 1.0)),
                use_substeps=False,
                substeps=20,
                per_orbit_steps=int(cfg["env"].get("per_orbit_steps", 0)) or None,
                mass_kg=float(cfg["env"].get("mass_kg", 1.0)),
                thrust_max_N=float(cfg["env"].get("thrust_max_N", 0.05)),
                eta_limit_rad=float(cfg["env"].get("eta_limit_rad", 0.5235987756)),
                debug_print=True,
            )
        else:
            prop = DummyPropagator(dt=float(cfg["env"].get("dt", 1.0)))
        return RLSatEnv(prop, cfg)
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--total-steps", type=int, default=1_000_000, help="Total training timesteps across all envs")
    ap.add_argument("--episode-orbits", type=int, required=True, help="Number of orbits per episode")
    ap.add_argument("--steps-per-orbit", type=int, required=True, help="Number of actions per orbit")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply episode/orbit overrides
    cfg.setdefault("env", {})
    cfg["env"]["episode_orbits"] = int(args.episode_orbits)
    cfg["env"]["per_orbit_steps"] = int(args.steps_per_orbit)
    # Also enforce fixed episode length in steps to avoid relying on timing
    cfg["env"]["episode_steps"] = int(args.episode_orbits) * int(args.steps_per_orbit)

    action_type = str(cfg.get("env", {}).get("action_type", "rtn_torque")).lower()
    auto_shuttle = action_type in ("angles_only", "angles_thrust")
    use_shuttle = bool(auto_shuttle)

    # episode_steps is derived above; no CLI override

    if args.num_envs > 1:
        # Avoid CPU oversubscription: force 1 thread per process
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        env = SubprocVecEnv([
            make_env(
                cfg,
                use_shuttle,
                "cpp/configs/shuttlecock_250km.json",
            )
            for _ in range(args.num_envs)
        ], start_method="fork")
    else:
        env = DummyVecEnv([
            make_env(
                cfg,
                use_shuttle,
                "cpp/configs/shuttlecock_250km.json",
            )
        ])

    # Default TB logging to runs/tb
    tb_dir = "runs/tb"
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_dir)
    callbacks = []
    callbacks.append(ProgressBarCallback(total=int(args.total_steps), desc="PPO"))
    callbacks.append(EpisodeEndTB())
    callbacks.append(LogPreActionLLA(every=1))

    # Choose a sensible default training horizon if not specified elsewhere
    # Use a very large log_interval so default TB flushes happen via EpisodeEndTB
    learn_kwargs = dict(total_timesteps=int(args.total_steps), log_interval=10**9, tb_log_name="ppo_rlsat")
    model.learn(callback=callbacks if callbacks else None, **learn_kwargs)

    os.makedirs("runs", exist_ok=True)
    model.save("runs/ppo_rlsat.zip")


if __name__ == "__main__":
    main()
