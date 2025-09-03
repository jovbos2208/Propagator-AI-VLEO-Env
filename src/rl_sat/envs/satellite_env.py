from __future__ import annotations

import gymnasium as gym
import numpy as np
from ..sim.interfaces import PropagatorInterface, PropagatorState, DummyPropagator
from ..sim import frames, rewards, termination, normalization
from .safety_filter import apply as apply_safety


class RLSatEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, propagator: PropagatorInterface | None, cfg: dict):
        self.propagator = propagator or DummyPropagator(dt=float(cfg.get("env", {}).get("dt", 1.0)))
        self.cfg = cfg
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(23,), dtype=np.float32)
        action_type = str(self.cfg.get("env", {}).get("action_type", "rtn_torque")).lower()
        if action_type == "angles_only":
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        elif action_type == "angles_thrust":
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.norm = normalization.RunningNorm(self.observation_space.shape[0])
        self.u_prev = np.zeros(self.action_space.shape[0], dtype=np.float32)
        # Reference state for error computation
        self._ref_r0 = None
        self._ref_v0 = None
        self._ref_a0 = None
        # Tracking for metrics
        self._last_ps: PropagatorState | None = None
        self._cum_dv: float = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_cfg = (options or {}).get("init_cfg", {})
        ps: PropagatorState = self.propagator.reset(seed or 0, init_cfg)
        # Store initial reference
        self._ref_r0 = ps.r_eci.copy()
        self._ref_v0 = ps.v_eci.copy()
        self._ref_a0 = self._semi_major_axis(ps.r_eci, ps.v_eci)
        # Estimate orbital period for time-limit episodes
        self._period_s = self._estimate_period(ps)
        # Store in cfg so termination can read it
        self.cfg.setdefault("env", {})["period_s"] = float(self._period_s)
        self._last_ps = ps
        self._cum_dv = 0.0
        obs = self._build_obs(ps)
        # Reset action history to current action dimension
        if self.u_prev.shape[0] != self.action_space.shape[0]:
            self.u_prev = np.zeros(self.action_space.shape[0], dtype=np.float32)
        else:
            self.u_prev[:] = 0.0
        return obs, {}

    def step(self, action):
        # Apply safety with knowledge of current state (for eclipse gating)
        u_scaled, sf_info = apply_safety(action, self.cfg, state=self._last_ps)
        # Step propagator depending on action type
        action_type = str(self.cfg.get("env", {}).get("action_type", "rtn_torque")).lower()
        if action_type in ("angles_only", "angles_thrust"):
            # Expect propagator to support direct angles. If not, raise.
            if hasattr(self.propagator, "step_angles"):
                eta1, eta2, thrust_N = float(u_scaled[0]), float(u_scaled[1]), float(u_scaled[2])
                ps = self.propagator.step_angles(eta1, eta2, thrust_N)
            else:
                raise NotImplementedError("Angles action type requires a propagator with step_angles(eta1, eta2, thrust_N)")
        else:
            ps = self.propagator.step(u_scaled[:3], u_scaled[3:])
        obs = self._build_obs(ps)
        # Metrics for logging/eval
        metrics = self._metrics(ps, u_scaled)
        info = {"sf": sf_info}
        info.update(metrics)
        # Reward and termination
        r, terms = rewards.compute(obs, self.u_prev, u_scaled, self.cfg)
        info.update(terms)
        terminated, truncated, fail_flag = termination.check(ps, obs, self.cfg)
        # Update trackers
        if action_type in ("angles_only", "angles_thrust"):
            # Use thrust_N and mass to estimate acceleration magnitude along body X ~ commanded accel
            # If mass unavailable, skip accumulation
            try:
                a_mag = float(u_scaled[2]) / max(float(ps.mass), 1e-6) if hasattr(ps, "mass") else 0.0
                self._cum_dv += a_mag * float(getattr(ps, "dt", 1.0))
            except Exception:
                pass
        else:
            self._cum_dv += float(np.linalg.norm(u_scaled[:3])) * float(getattr(ps, "dt", 1.0))
        self._last_ps = ps
        self.u_prev = u_scaled
        return obs, float(r), bool(terminated), bool(truncated), info

    def _build_obs(self, ps: PropagatorState):
        env_cfg = self.cfg.get("env", {})
        scales = env_cfg.get("obs_scales", {})
        dr_scale = float(scales.get("dr", 5000.0))
        dv_scale = float(scales.get("dv", 0.1))
        da_scale = float(scales.get("da", 100.0))
        e_scale = float(scales.get("e", 0.01))
        theta_deg_scale = float(scales.get("theta_deg", 20.0))
        omega_scale = float(scales.get("omega", 0.1))

        r = np.asarray(ps.r_eci, dtype=np.float64)
        v = np.asarray(ps.v_eci, dtype=np.float64)

        # RTN frame from current state
        R_eci_to_rtn = frames.eci_to_rtn(r, v)

        # Position/velocity errors vs. stored reference (initial state)
        if self._ref_r0 is None:
            self._ref_r0 = r.copy()
        if self._ref_v0 is None:
            self._ref_v0 = v.copy()
        dr_eci = r - self._ref_r0
        dv_eci = v - self._ref_v0
        dr_rtn = R_eci_to_rtn.T @ dr_eci
        dv_rtn = R_eci_to_rtn.T @ dv_eci

        # Orbital elements (semi-major axis and eccentricity magnitude)
        a_now = self._semi_major_axis(r, v)
        da = float(a_now - (self._ref_a0 if self._ref_a0 is not None else a_now))
        e_mag = float(self._eccentricity_mag(r, v))

        # Attitude target and error quaternion
        mode = str(env_cfg.get("mode", "coupled"))
        q_target = frames.target_quat(mode, r, v)
        q_err = frames.quat_err(q_target, np.asarray(ps.q_be, dtype=np.float64))
        q_err = q_err.astype(np.float64)
        # Attitude error angle (deg)
        w = np.clip(q_err[0], -1.0, 1.0)
        theta_rad = 2.0 * float(np.arccos(w))
        theta_deg = np.degrees(theta_rad)

        omega_b = np.asarray(ps.omega_b, dtype=np.float64)
        h_rw = np.asarray(ps.h_rw, dtype=np.float64)

        # Assemble feature vector (scaled)
        vec = np.zeros(23, dtype=np.float64)
        vec[0:3] = dr_rtn / (dr_scale if dr_scale > 0 else 1.0)
        vec[3:6] = dv_rtn / (dv_scale if dv_scale > 0 else 1.0)
        vec[6] = da / (da_scale if da_scale > 0 else 1.0)
        vec[7] = e_mag / (e_scale if e_scale > 0 else 1.0)
        vec[8:12] = q_err
        vec[12:15] = omega_b / (omega_scale if omega_scale > 0 else 1.0)
        vec[15:18] = h_rw
        vec[18] = float(ps.density)
        vec[19] = 1.0 if bool(ps.in_eclipse) else 0.0
        vec[20] = float(ps.mass)
        vec[21] = float(ps.soc)
        vec[22] = (theta_deg / (theta_deg_scale if theta_deg_scale > 0 else 1.0))

        return self.norm(vec.astype(np.float32))

    def _metrics(self, ps: PropagatorState, u_scaled: np.ndarray) -> dict:
        # Compute raw metrics for logging/evaluation
        r = np.asarray(ps.r_eci, dtype=np.float64)
        v = np.asarray(ps.v_eci, dtype=np.float64)
        R_eci_to_rtn = frames.eci_to_rtn(r, v)
        dr_eci = r - (self._ref_r0 if self._ref_r0 is not None else r)
        dv_eci = v - (self._ref_v0 if self._ref_v0 is not None else v)
        dr_rtn = R_eci_to_rtn.T @ dr_eci
        dv_rtn = R_eci_to_rtn.T @ dv_eci
        a_now = self._semi_major_axis(r, v)
        da = float(a_now - (self._ref_a0 if self._ref_a0 is not None else a_now))
        e_mag = float(self._eccentricity_mag(r, v))
        q_target = frames.target_quat(str(self.cfg.get("env", {}).get("mode", "coupled")), r, v)
        q_err = frames.quat_err(q_target, np.asarray(ps.q_be, dtype=np.float64))
        theta_deg = float(np.degrees(2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0))))
        a_cmd = float(np.linalg.norm(u_scaled[:3]))
        tau_cmd = float(np.linalg.norm(u_scaled[3:]))
        return {
            "dr_rtn_m": float(np.linalg.norm(dr_rtn)),
            "dv_rtn_mps": float(np.linalg.norm(dv_rtn)),
            "da_m": da,
            "e": e_mag,
            "theta_deg": theta_deg,
            "density": float(ps.density),
            "in_eclipse": bool(ps.in_eclipse),
            "mass": float(ps.mass),
            "soc": float(ps.soc),
            "a_cmd_mps2": a_cmd,
            "tau_cmd_nm": tau_cmd,
            "cum_dv_mps": float(self._cum_dv),
        }

    def _estimate_period(self, ps: PropagatorState) -> float:
        # Prefer adaptor-provided period if available
        try:
            envc = getattr(self.propagator, "envc", None)
            if envc is not None and hasattr(envc, "estimate_period_s"):
                T = float(envc.estimate_period_s())
                if np.isfinite(T) and T > 0:
                    return T
        except Exception:
            pass
        # Keplerian approximation
        mu = 3.986004418e14
        a = max(self._semi_major_axis(ps.r_eci, ps.v_eci), 1.0)
        return float(2.0 * np.pi * np.sqrt((a ** 3) / mu))

    @staticmethod
    def _semi_major_axis(r: np.ndarray, v: np.ndarray) -> float:
        mu = 3.986004418e14  # m^3/s^2 (Earth)
        rnorm = float(np.linalg.norm(r))
        v2 = float(np.dot(v, v))
        inv_a = 2.0 / max(rnorm, 1e-9) - v2 / mu
        if inv_a == 0:
            return 1e20
        return 1.0 / inv_a

    @staticmethod
    def _eccentricity_mag(r: np.ndarray, v: np.ndarray) -> float:
        mu = 3.986004418e14
        r = np.asarray(r, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        h = np.cross(r, v)
        e_vec = (np.cross(v, h) / mu) - (r / (np.linalg.norm(r) + 1e-12))
        return float(np.linalg.norm(e_vec))
