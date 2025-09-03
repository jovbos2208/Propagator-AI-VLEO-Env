from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import yaml
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure 'src' is on sys.path when running as a module
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_sat.envs.satellite_env import RLSatEnv
from rl_sat.sim.interfaces import DummyPropagator
from rl_sat.sim.shuttlecock_adapter import ShuttlecockPropagator


def make_env(cfg: dict, use_shuttle: bool, shuttle_cfg_path: str | None):
    def _thunk():
        if use_shuttle:
            prop = ShuttlecockPropagator(
                config_path=shuttle_cfg_path or "cpp/configs/shuttlecock_250km.json",
                control_dt=float(cfg["env"].get("dt", 1.0)),
            )
        else:
            prop = DummyPropagator(dt=float(cfg["env"].get("dt", 1.0)))
        return RLSatEnv(prop, cfg)
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--outdir", default="runs/eval")
    ap.add_argument("--use-shuttlecock", action="store_true", help="Use Shuttlecock C++ env via adaptor")
    ap.add_argument("--shuttle-config", default="cpp/configs/shuttlecock_250km.json")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    action_type = str(cfg.get("env", {}).get("action_type", "rtn_torque")).lower()
    auto_shuttle = action_type in ("angles_only", "angles_thrust")
    use_shuttle = bool(args.use_shuttlecock or auto_shuttle)
    env = DummyVecEnv([make_env(cfg, use_shuttle, args.shuttle_config)])
    if args.algo == "ppo":
        model = PPO.load(args.ckpt, env=env)
    else:
        model = SAC.load(args.ckpt, env=env)

    os.makedirs(args.outdir, exist_ok=True)
    all_eps = []

    for ep in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        ep_r = 0.0
        ep_metrics = {"theta_deg": [], "da_m": [], "e": [], "dr_rtn_m": [], "dv_rtn_mps": [], "cum_dv_mps": []}
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            # VecEnv returns arrays/lists
            r = float(np.asarray(reward)[0])
            info = infos[0]
            ep_r += r
            steps += 1
            # Collect metrics if present
            for k in ep_metrics.keys():
                if k in info:
                    ep_metrics[k].append(float(info[k]))
        summary = {
            "episode": ep + 1,
            "return": ep_r,
            "steps": steps,
        }
        # Aggregate metrics
        for k, arr in ep_metrics.items():
            if arr:
                summary[f"mean_{k}"] = float(np.mean(arr))
                summary[f"final_{k}"] = float(arr[-1])
                summary[f"max_{k}"] = float(np.max(arr))
        all_eps.append(summary)
        print(f"Episode {ep+1}/{args.episodes}: return={ep_r:.3f}, steps={steps}")

    with open(os.path.join(args.outdir, "eval_metrics.json"), "w") as f:
        json.dump(all_eps, f, indent=2)
    print(f"Wrote metrics to {os.path.join(args.outdir, 'eval_metrics.json')}")


if __name__ == "__main__":
    main()
