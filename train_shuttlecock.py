import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from python.shuttlecock_env import ShuttlecockEnvC, GymWrapper


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except Exception:
        return False


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
        # Observation: [r(3), v(3), q(4), w(3), alt, rho, AoA, AoS, roll] -> 18 dims (float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
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


class EnvDemoLikeCallback(BaseCallback):
    def __init__(self, csv_path: str | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._csv = None
        self._wrote_header = False

    def _on_training_start(self) -> None:
        if self.csv_path:
            import os
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            self._csv = open(self.csv_path, "w")
            self._csv.write(
                "step,R_total,R_alt,R_att,R_spin,R_effort,dwell_alt,dwell_att,rem_impulse,substeps,altitude,AoA,AoS,roll\n"
            )
            self._wrote_header = True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        if infos and len(infos) > 0:
            info = infos[0]
            r_total = float(rewards[0]) if rewards is not None else 0.0
            line = (
                f"step={self.num_timesteps} total={r_total:.6g} "
                f"[alt={info.get('R_alt',0):.6g}, att={info.get('R_att',0):.6g}, spin={info.get('R_spin',0):.6g}, "
                f"effort={info.get('R_effort',0):.6g}] dwell_alt={info.get('dwell_alt',0):.2%} dwell_att={info.get('dwell_att',0):.2%} "
                f"rem_impulse={info.get('remaining_impulse',0):.3g} Ns substeps={info.get('substeps',0)} "
                f"alt={info.get('altitude',0):.3f} AoA={info.get('AoA',0):.3g} AoS={info.get('AoS',0):.3g} roll={info.get('roll',0):.3g}"
            )
            if self.verbose:
                print(line)
            if self._csv:
                self._csv.write(
                    f"{self.num_timesteps},{r_total},{info.get('R_alt',0)},{info.get('R_att',0)},{info.get('R_spin',0)},{info.get('R_effort',0)},{info.get('dwell_alt',0)},{info.get('dwell_att',0)},{info.get('remaining_impulse',0)},{info.get('substeps',0)},{info.get('altitude',0)},{info.get('AoA',0)},{info.get('AoS',0)},{info.get('roll',0)}\n"
                )
        return True

    def _on_training_end(self) -> None:
        if self._csv:
            self._csv.close()


class TqdmProgressBarCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.pbar = None
        self._last_num = 0
        self._total = None

    def _on_training_start(self) -> None:
        try:
            from tqdm import tqdm
        except Exception:
            return  # Silently disable if tqdm not available
        total = self.locals.get("total_timesteps", None)
        # Fallback when not provided; estimate from arguments if needed
        self._total = int(total) if total is not None else None
        self._last_num = int(self.model.num_timesteps)
        if self._total is not None:
            self.pbar = tqdm(total=self._total, desc="Training", unit="steps")
        else:
            self.pbar = tqdm(desc="Training", unit="steps")

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True
        cur = int(self.model.num_timesteps)
        delta = cur - self._last_num
        if delta > 0:
            self.pbar.update(delta)
            self._last_num = cur
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            # Ensure bar completes to total
            if self._total is not None and self._last_num < self._total:
                self.pbar.update(self._total - self._last_num)
            self.pbar.close()


def make_env(config_path, control_dt, align_to_velocity, use_substeps, substeps, per_orbit_steps):
    def _thunk():
        env = ShuttleGym(config_path, control_dt=control_dt, align_to_velocity=align_to_velocity)
        env.inner.use_substeps = use_substeps
        env.inner.substeps = substeps
        env.inner.per_orbit_steps = per_orbit_steps
        return env
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpp/configs/shuttlecock_250km.json")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--control-dt", type=float, default=5.0)
    ap.add_argument("--save", default="checkpoints/shuttlecock_ppo_v1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--print-steps", action="store_true", help="Print per-step env_demo-like lines")
    ap.add_argument("--csv-log", default=None, help="Optional CSV log file for per-step lines")
    ap.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs")
    ap.add_argument("--use-substeps", action="store_true", help="Use substep-based stepping (DP54) instead of fixed duration")
    ap.add_argument("--substeps", type=int, default=20, help="Substeps per call when using --use-substeps")
    ap.add_argument("--normalize", action="store_true", help="Use VecNormalize for observations/rewards")
    ap.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps per env")
    ap.add_argument("--batch-size", type=int, default=256, help="PPO batch size")
    ap.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate")
    ap.add_argument("--per-orbit-steps", type=int, default=None, help="If set, the env will take this many actions per estimated orbit (uses duration stepping)")
    ap.add_argument("--progress", action="store_true", help="Show a tqdm progress bar (default on)")
    ap.add_argument("--no-progress", dest="progress", action="store_false", help="Disable the tqdm progress bar")
    ap.set_defaults(progress=True)
    args = ap.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    # Build vectorized envs
    if args.num_envs > 1:
        env_fns = [make_env(args.config, args.control_dt, True, args.use_substeps, args.substeps, args.per_orbit_steps) for _ in range(args.num_envs)]
        vec = SubprocVecEnv(env_fns)
    else:
        vec = DummyVecEnv([make_env(args.config, args.control_dt, True, args.use_substeps, args.substeps, args.per_orbit_steps)])
    env = VecMonitor(vec)
    steps_per_episode = int(10 * (5400.0 / args.control_dt))
    total_timesteps = steps_per_episode * args.episodes

    callbacks = [EnvDemoLikeCallback(csv_path=args.csv_log, verbose=1 if args.print_steps else 0)]
    if args.progress:
        callbacks.append(TqdmProgressBarCallback())
    callback = CallbackList(callbacks)
    # Optional normalization
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    tb_log = "runs/" if _tensorboard_available() else None
    if args.resume:
        model = PPO.load(args.resume, env=env)
        # If the loaded model was saved without TB, keep it disabled; else SB3 will handle it
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=tb_log,
            n_steps=max(64, args.n_steps // max(1, args.num_envs)),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(args.save)
    print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
