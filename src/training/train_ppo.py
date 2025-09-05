from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

# Ensure 'src' is on sys.path when running as a module
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_sat.envs.satellite_env import RLSatEnv
from rl_sat.sim.interfaces import DummyPropagator
from rl_sat.sim.shuttlecock_adapter import ShuttlecockPropagator
from .callbacks import ProgressBarCallback, LogPreActionLLA, EpisodeEndTB, StepMetricsTB, EvalEveryOrbitsTB


def make_env(
    cfg: dict,
    use_shuttle: bool,
    shuttle_cfg_path: str | None,
):
    def _thunk():
        # Ensure env-specific debug configuration is present
        cfg.setdefault("debug", {})
        cfg["debug"].update({
            "pre_action": False,
            "every": 50,
            "print": False,
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
                tau_max=float(cfg["env"].get("tau_max", 0.01)),
                batt_wh=float(cfg["env"].get("batt_wh", 50.0)),
                power_gen_w=float(cfg["env"].get("power_gen_w", 5.0)),
                power_use_w=float(cfg["env"].get("power_use_w", 5.0)),
                debug_print=False,
            )
        else:
            prop = DummyPropagator(dt=float(cfg["env"].get("dt", 1.0)))
        return RLSatEnv(prop, cfg)
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-envs", type=int, default=-1)
    ap.add_argument("--total-steps", type=int, default=-1, help="Total training timesteps across all envs")
    ap.add_argument("--episode-orbits", type=int, default=-1, help="Number of orbits per episode (overrides config)")
    ap.add_argument("--steps-per-orbit", type=int, default=-1, help="Number of actions per orbit (overrides config)")
    ap.add_argument(
        "--update-every-episodes",
        type=int,
        default=-1,
        help="Episodes per env per PPO update (overrides config)",
    )
    ap.add_argument("--n-epochs", type=int, default=-1, help="PPO epochs per update (overrides config)")
    ap.add_argument("--learning-rate", type=float, default=-1.0, help="PPO optimizer learning rate (overrides config)")
    ap.add_argument("--normalize", action="store_true", help="Wrap env with VecNormalize (obs+reward)")
    ap.add_argument("--tb-logdir", default="runs/tb", help="TensorBoard log directory")
    ap.add_argument("--tb-train-every", type=int, default=10, help="Log per-step TB metrics every N steps")
    ap.add_argument("--tb-eval-orbits", type=int, default=10, help="Run periodic eval every N orbits")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Merge training params from config with CLI overrides
    cfg.setdefault("env", {})
    tcfg = cfg.get("training", {}) or {}
    # Read values with precedence: CLI override (>0) -> training section -> env defaults
    episode_orbits = int(args.episode_orbits) if int(args.episode_orbits) > 0 else int(tcfg.get("episode_orbits", cfg["env"].get("episode_orbits", 0) or 0))
    steps_per_orbit = int(args.steps_per_orbit) if int(args.steps_per_orbit) > 0 else int(tcfg.get("steps_per_orbit", cfg["env"].get("per_orbit_steps", 0) or 0))
    if episode_orbits <= 0 or steps_per_orbit <= 0:
        raise ValueError("episode_orbits and steps_per_orbit must be set either via CLI or training section in config")
    num_envs = int(args.num_envs) if int(args.num_envs) > 0 else int(tcfg.get("num_envs", 1))
    total_steps = int(args.total_steps) if int(args.total_steps) > 0 else int(tcfg.get("total_steps", 1_000_000))
    update_every_episodes = int(args.update_every_episodes) if int(args.update_every_episodes) > 0 else int(tcfg.get("update_every_episodes", 1))
    n_epochs = int(args.n_epochs) if int(args.n_epochs) > 0 else int(tcfg.get("n_epochs", 10))
    learning_rate = float(args.learning_rate) if float(args.learning_rate) > 0 else float(tcfg.get("learning_rate", 3e-4))
    tb_dir = str(args.tb_logdir or tcfg.get("tb_logdir", "runs/tb"))
    resume_from = tcfg.get("resume_from", None)
    reset_num_timesteps = bool(tcfg.get("reset_num_timesteps", False))

    # Apply to env config
    cfg["env"]["episode_orbits"] = episode_orbits
    cfg["env"]["per_orbit_steps"] = steps_per_orbit
    # Also enforce fixed episode length in steps
    cfg["env"]["episode_steps"] = episode_orbits * steps_per_orbit

    action_type = str(cfg.get("env", {}).get("action_type", "rtn_torque")).lower()
    auto_shuttle = action_type in ("angles_only", "angles_thrust")
    use_shuttle = bool(auto_shuttle)

    # episode_steps is derived above; no CLI override

    if num_envs > 1:
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
            for _ in range(num_envs)
        ], start_method="fork")
    else:
        env = DummyVecEnv([
            make_env(
                cfg,
                use_shuttle,
                "cpp/configs/shuttlecock_250km.json",
            )
        ])

    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Compute PPO rollout length from episodes and envs
    episode_steps = int(cfg["env"]["episode_steps"]) 
    episodes_per_update = max(1, int(update_every_episodes))
    n_steps = max(1, episode_steps * episodes_per_update)
    # Use full-batch updates by default (one batch per rollout)
    batch_size = max(1, n_steps * int(num_envs))

    # Default TB logging to runs/tb
    # TB directory from config (default runs/tb)
    if resume_from and os.path.exists(str(resume_from)):
        # Resume: load model with saved hyperparams
        model = PPO.load(str(resume_from), env=env, tensorboard_log=tb_dir)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_dir,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
        )
    callbacks = []
    callbacks.append(ProgressBarCallback(total=int(total_steps), desc="PPO"))
    callbacks.append(EpisodeEndTB())
    callbacks.append(LogPreActionLLA(every=max(1, int(episode_orbits))))
    callbacks.append(StepMetricsTB(every=max(1, int(args.tb_train_every))))
    callbacks.append(EvalEveryOrbitsTB(lambda: DummyVecEnv([
        make_env(
            cfg,
            use_shuttle,
            "cpp/configs/shuttlecock_250km.json",
        )
    ]), cfg, orbits_interval=max(1, int(args.tb_eval_orbits))))

    # Choose a sensible default training horizon if not specified elsewhere
    # Use a very large log_interval so default TB flushes happen via EpisodeEndTB
    learn_kwargs = dict(total_timesteps=int(total_steps), log_interval=10**9, tb_log_name="ppo_rlsat", reset_num_timesteps=bool(reset_num_timesteps))
    model.learn(callback=callbacks if callbacks else None, **learn_kwargs)

    os.makedirs("runs", exist_ok=True)
    model.save("runs/ppo_rlsat.zip")


if __name__ == "__main__":
    main()
