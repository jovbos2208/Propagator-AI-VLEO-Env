import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpp/configs/shuttlecock_250km.json")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--control-dt", type=float, default=5.0)
    ap.add_argument("--save", default="checkpoints/shuttlecock_ppo_v1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--print-steps", action="store_true", help="Print per-step env_demo-like lines")
    ap.add_argument("--csv-log", default=None, help="Optional CSV log file for per-step lines")
    args = ap.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    env = Monitor(ShuttleGym(args.config, control_dt=args.control_dt))
    steps_per_episode = int(10 * (5400.0 / args.control_dt))
    total_timesteps = steps_per_episode * args.episodes

    callback = EnvDemoLikeCallback(csv_path=args.csv_log, verbose=1 if args.print-steps else 0)
    if args.resume:
        model = PPO.load(args.resume, env=env)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/")
        model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(args.save)
    print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
