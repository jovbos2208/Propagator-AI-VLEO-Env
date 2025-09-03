from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# Ensure 'src' is on sys.path when running as a module
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_sat.envs.satellite_env import RLSatEnv
from rl_sat.sim.interfaces import DummyPropagator
from rl_sat.sim.shuttlecock_adapter import ShuttlecockPropagator
from .callbacks import ProgressBarCallback, EvalEveryOrbitsTB


def make_env(cfg: dict, use_shuttle: bool, shuttle_cfg_path: str | None):
    def _thunk():
        if use_shuttle:
            prop = ShuttlecockPropagator(
                config_path=shuttle_cfg_path or "cpp/configs/shuttlecock_250km.json",
                control_dt=float(cfg["env"].get("dt", 1.0)),
                per_orbit_steps=int(cfg["env"].get("per_orbit_steps", 0)) or None,
            )
        else:
            prop = DummyPropagator(dt=float(cfg["env"].get("dt", 1.0)))
        return RLSatEnv(prop, cfg)
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--use-shuttlecock", action="store_true", help="Use Shuttlecock C++ env via adaptor")
    ap.add_argument("--shuttle-config", default="cpp/configs/shuttlecock_250km.json")
    ap.add_argument("--tb-logdir", default=None, help="TensorBoard log directory (enables TB logging)")
    ap.add_argument("--log-interval", type=int, default=1000, help="Training log interval (SB3)")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar during training")
    ap.add_argument("--mp-start-method", default="spawn", choices=["spawn", "fork", "forkserver"], help="Multiprocessing start method for SubprocVecEnv")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    action_type = str(cfg.get("env", {}).get("action_type", "rtn_torque")).lower()
    auto_shuttle = action_type in ("angles_only", "angles_thrust")
    use_shuttle = bool(args.use_shuttlecock or auto_shuttle)

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
        env = SubprocVecEnv([make_env(cfg, use_shuttle, args.shuttle_config) for _ in range(args.num_envs)], start_method=args.mp_start_method)
    else:
        env = DummyVecEnv([make_env(cfg, use_shuttle, args.shuttle_config)])

    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=args.tb_logdir if args.tb_logdir else None)
    callbacks = []
    if not args.no_progress:
        callbacks.append(ProgressBarCallback(total=args.total_steps, desc="SAC"))
    def make_eval_env():
        return DummyVecEnv([make_env(cfg, use_shuttle, args.shuttle_config)])
    callbacks.append(EvalEveryOrbitsTB(make_eval_env, cfg, orbits_interval=10))
    learn_kwargs = dict(total_timesteps=args.total_steps, log_interval=args.log_interval)
    if args.tb_logdir:
        learn_kwargs.update(tb_log_name="sac_rlsat")
    model.learn(callback=callbacks if callbacks else None, **learn_kwargs)

    os.makedirs("runs", exist_ok=True)
    model.save("runs/sac_rlsat.zip")


if __name__ == "__main__":
    main()
