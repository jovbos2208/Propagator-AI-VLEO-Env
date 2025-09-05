from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm


class EarlyStopOnPlateau(BaseCallback):
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def _on_step(self) -> bool:
        # Placeholder: do nothing (requires eval pipeline to compute mean reward)
        return True


@dataclass
class ProgressBarCallback(BaseCallback):
    """Simple tqdm progress bar for SB3 training.

    If total is None, tries to read model._total_timesteps set by learn().
    """

    total: Optional[int] = None
    desc: str = "training"
    disable: bool = False

    def __post_init__(self):
        super().__init__()
        self._bar: Optional[tqdm] = None
        self._last: int = 0

    def _on_training_start(self) -> None:
        total = self.total
        if total is None:
            total = getattr(self.model, "_total_timesteps", None)
        try:
            total = int(total) if total is not None else None
        except Exception:
            total = None
        self._bar = tqdm(total=total, desc=self.desc, disable=self.disable)
        self._last = 0
        # Touch TensorBoard logger so event files are created immediately
        try:
            self.logger.record("progress/initialized", 1)
            # step 0 ensures TB event file exists from the start
            self.logger.dump(step=0)
        except Exception:
            pass

    def _on_step(self) -> bool:
        if self._bar is not None:
            cur = int(self.num_timesteps)
            delta = max(0, cur - self._last)
            if delta:
                self._bar.update(delta)
                self._last = cur
        return True

    def _on_training_end(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class EvalEveryOrbitsTB(BaseCallback):
    """Runs a short evaluation every K orbits and logs metrics, actions, and config to TensorBoard.

    It uses a provided factory to build a single-env DummyVecEnv for evaluation.
    Steps per orbit are inferred from cfg: per_orbit_steps if set, else approx period_s/dt.
    """

    def __init__(self, make_env_fn, cfg: dict, orbits_interval: int = 10, deterministic: bool = True):
        super().__init__()
        self.make_env_fn = make_env_fn
        self.cfg = cfg
        self.orbits_interval = int(orbits_interval)
        self.deterministic = bool(deterministic)
        self._last_eval_step = 0
        self._steps_per_orbit = None

    def _compute_steps_per_orbit(self):
        env_cfg = self.cfg.get("env", {})
        per_orbit = env_cfg.get("per_orbit_steps")
        if per_orbit and int(per_orbit) > 0:
            return int(per_orbit)
        dt = float(env_cfg.get("dt", 1.0))
        period = float(env_cfg.get("period_s", 0.0) or 0.0)
        if period <= 0:
            # rough default ~90 minutes
            period = 5400.0
        return max(1, int(round(period / max(dt, 1e-6))))

    def _on_training_start(self) -> None:
        self._steps_per_orbit = self._compute_steps_per_orbit()

    def _on_step(self) -> bool:
        steps_per_eval = self._steps_per_orbit * self.orbits_interval
        if self.num_timesteps - self._last_eval_step < steps_per_eval:
            return True
        self._last_eval_step = self.num_timesteps

        # Build eval env
        eval_env = self.make_env_fn()
        # Rollout exactly K orbits worth of steps
        target_steps = steps_per_eval
        obs = eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_r = 0.0
        action_norms = []
        action_abs_sum = None
        action_count = 0
        theta = []
        da = []
        ecc = []
        dr = []
        dv = []
        cumdv = []
        t_list = []
        steps = 0
        while steps < target_steps:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, infos = eval_env.step(action)
            r = float(np.asarray(reward)[0]) if hasattr(reward, "__len__") else float(reward)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            total_r += r
            steps += 1
            # Collect metrics from info
            theta.append(float(info.get("theta_deg", np.nan)))
            da.append(float(info.get("da_m", np.nan)))
            ecc.append(float(info.get("e", np.nan)))
            dr.append(float(info.get("dr_rtn_m", np.nan)))
            dv.append(float(info.get("dv_rtn_mps", np.nan)))
            cumdv.append(float(info.get("cum_dv_mps", np.nan)))
            t_list.append(float(info.get("t_s", np.nan)))
            # Action magnitudes
            a = np.asarray(action).reshape(-1)
            action_norms.append(float(np.linalg.norm(a)))
            aa = np.abs(a)
            if action_abs_sum is None:
                action_abs_sum = np.zeros_like(aa, dtype=float)
            action_abs_sum = action_abs_sum + aa
            action_count += 1

        # Aggregate and log
        def nanmean(x):
            arr = np.asarray(x, dtype=float)
            return float(np.nanmean(arr)) if arr.size else float("nan")

        self.logger.record("eval10/return", total_r)
        self.logger.record("eval10/steps", steps)
        self.logger.record("eval10/theta_deg_mean", nanmean(theta))
        self.logger.record("eval10/da_m_mean", nanmean(da))
        self.logger.record("eval10/e_mean", nanmean(ecc))
        self.logger.record("eval10/dr_rtn_m_mean", nanmean(dr))
        self.logger.record("eval10/dv_rtn_mps_mean", nanmean(dv))
        self.logger.record("eval10/cum_dv_mps_final", float(cumdv[-1]) if cumdv else float("nan"))
        self.logger.record("eval10/action_norm_mean", nanmean(action_norms))
        if action_abs_sum is not None and action_count > 0:
            action_abs_mean = action_abs_sum / float(action_count)
            for i, val in enumerate(action_abs_mean):
                self.logger.record(f"eval10/action_abs_mean_{i}", float(val))

        # Log config parameters (numeric) once per eval
        def _log_numeric_dict(prefix: str, d: dict):
            for k, v in d.items():
                try:
                    self.logger.record(f"{prefix}/{k}", float(v))
                except Exception:
                    # skip non-numeric
                    continue
        env_cfg = self.cfg.get("env", {})
        rw_cfg = self.cfg.get("reward_weights", {})
        _log_numeric_dict("config/env", env_cfg)
        _log_numeric_dict("config/reward_weights", rw_cfg)

        # Flush at the current training timestep
        self.logger.dump(self.num_timesteps)
        try:
            eval_env.close()
        except Exception:
            pass
        return True


class StepMetricsTB(BaseCallback):
    """Logs per-step training reward breakdown and actions to TensorBoard.

    - Records mean total reward across envs.
    - Records mean of reward components found in infos: r_alt, r_e, r_theta.
    - Records selected metrics from infos (means across envs) such as theta_deg, alt_m, target_alt_m,
      e, target_ecc, da_m, dr_rtn_m, dv_rtn_mps, cum_dv_mps, a_cmd_mps2, tau_cmd_nm.
    - Records action stats: mean L2 norm and per-component mean and mean absolute.

    Note: Logging every step can be heavy; use `every` to subsample.
    """

    def __init__(self, every: int = 1):
        super().__init__()
        self.every = max(1, int(every))

    def _on_step(self) -> bool:
        if (self.num_timesteps % self.every) != 0:
            return True
        rewards = self.locals.get("rewards", None)
        infos = self.locals.get("infos", None)
        actions = self.locals.get("actions", None)
        if rewards is not None:
            try:
                r = np.asarray(rewards).reshape(-1)
                self.logger.record("train/reward_mean", float(np.mean(r)))
            except Exception:
                pass
        # Reward components and metrics from infos
        if isinstance(infos, (list, tuple)) and infos:
            def mean_from_infos(key: str):
                vals = []
                for info in infos:
                    try:
                        if key in info:
                            vals.append(float(info[key]))
                    except Exception:
                        continue
                return (float(np.mean(vals)) if vals else None)

            # Reward parts
            for k in ("r_alt", "r_e", "r_theta"):
                m = mean_from_infos(k)
                if m is not None:
                    self.logger.record(f"train/{k}_mean", m)
            # Selected metrics helpful for evaluation
            for k in ("theta_deg", "alt_m", "target_alt_m", "e", "target_ecc", "da_m",
                      "dr_rtn_m", "dv_rtn_mps", "cum_dv_mps", "a_cmd_mps2", "tau_cmd_nm", "t_s"):
                m = mean_from_infos(k)
                if m is not None:
                    self.logger.record(f"train/{k}_mean", m)
        # Actions
        if actions is not None:
            try:
                a = np.asarray(actions)
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                norms = np.linalg.norm(a, axis=1)
                self.logger.record("train/action_l2_mean", float(np.mean(norms)))
                # per-component statistics
                comp_mean = np.mean(a, axis=0)
                comp_abs_mean = np.mean(np.abs(a), axis=0)
                for i, (m, am) in enumerate(zip(comp_mean, comp_abs_mean)):
                    self.logger.record(f"train/action_mean_{i}", float(m))
                    self.logger.record(f"train/action_abs_mean_{i}", float(am))
            except Exception:
                pass
        # Scaled/raw actions from infos (if provided by env)
        if isinstance(infos, (list, tuple)) and infos:
            # Helper to average values across infos
            def mean_from_infos(key: str):
                vals = []
                for info in infos:
                    try:
                        if key in info:
                            vals.append(float(info[key]))
                    except Exception:
                        continue
                return (float(np.mean(vals)) if vals else None)
            # Determine component indices by inspecting keys
            keys0 = list(infos[0].keys()) if isinstance(infos[0], dict) else []
            def _comp_indices(prefix: str):
                idx = []
                for k in keys0:
                    if k.startswith(prefix):
                        try:
                            idx.append(int(k[len(prefix):]))
                        except Exception:
                            continue
                return sorted(set(idx))
            for prefix, name in (("act_raw_", "act_raw"), ("act_scaled_", "act_scaled")):
                idxs = _comp_indices(prefix)
                for i in idxs:
                    m = mean_from_infos(f"{prefix}{i}")
                    if m is not None:
                        self.logger.record(f"train/{name}_mean_{i}", float(m))
        self.logger.dump(self.num_timesteps)
        return True


class LogPreActionLLA(BaseCallback):
    """Logs pre-action lat/lon/alt and time from each sub-env to TensorBoard.

    Requires envs to implement get_pre_debug() returning a dict with keys:
      lat_deg, lon_deg, alt_m, t_s (optional)
    """

    def __init__(self, every: int = 10):
        super().__init__()
        self.every = max(1, int(every))

    def _on_step(self) -> bool:
        if (self.num_timesteps % self.every) != 0:
            return True
        try:
            # env_method returns a list of results, one per env
            vals = self.training_env.env_method("get_pre_debug")
        except Exception:
            return True
        for i, d in enumerate(vals):
            if not isinstance(d, dict):
                continue
            if "lat_deg" in d:
                self.logger.record(f"pre/lat_deg_env{i}", float(d["lat_deg"]))
            if "lon_deg" in d:
                self.logger.record(f"pre/lon_deg_env{i}", float(d["lon_deg"]))
            if "alt_m" in d:
                self.logger.record(f"pre/alt_m_env{i}", float(d["alt_m"]))
            if "t_s" in d:
                self.logger.record(f"pre/t_s_env{i}", float(d["t_s"]))
        # Do not dump here; EpisodeEndTB will flush at episode boundaries
        return True


class EpisodeEndTB(BaseCallback):
    """Logs per-episode return and length (per env) to TensorBoard.

    Also logs select final-step metrics if present in info.
    """

    def __init__(self):
        super().__init__()
        self._ep_ret = None
        self._ep_len = None

    def _on_training_start(self) -> None:
        try:
            n = int(getattr(self.training_env, "num_envs", 1))
        except Exception:
            n = 1
        self._ep_ret = np.zeros(n, dtype=float)
        self._ep_len = np.zeros(n, dtype=int)

    def _on_step(self) -> bool:
        # Expect vectors
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if rewards is None or dones is None or infos is None:
            return True
        n = len(dones)
        # Ensure arrays sized
        if self._ep_ret is None or self._ep_ret.shape[0] != n:
            self._ep_ret = np.zeros(n, dtype=float)
            self._ep_len = np.zeros(n, dtype=int)
        self._ep_ret[:n] += np.asarray(rewards).reshape(-1)[:n]
        self._ep_len[:n] += 1
        any_done = False
        for i, done in enumerate(dones):
            if not bool(done):
                continue
            any_done = True
            info = infos[i] if i < len(infos) else {}
            # Scalars: return and length
            self.logger.record(f"episode/return_env{i}", float(self._ep_ret[i]))
            self.logger.record(f"episode/length_env{i}", int(self._ep_len[i]))
            # Truncation flag if available
            if isinstance(info, dict):
                if "t_s" in info:
                    self.logger.record(f"episode/t_s_env{i}", float(info["t_s"]))
                if "episode_steps_limit" in info:
                    self.logger.record(f"episode/steps_limit_env{i}", float(bool(info["episode_steps_limit"])) )
                # Selected final metrics
                for k in ("theta_deg", "da_m", "e", "dr_rtn_m", "dv_rtn_mps", "cum_dv_mps"):
                    if k in info:
                        self.logger.record(f"final/{k}_env{i}", float(info[k]))
            # Reset counters for env i
            self._ep_ret[i] = 0.0
            self._ep_len[i] = 0
        if any_done:
            self.logger.dump(self.num_timesteps)
        return True
