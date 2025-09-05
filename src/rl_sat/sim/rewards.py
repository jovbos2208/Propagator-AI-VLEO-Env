from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    w_a: float = 1.0
    w_e: float = 0.5
    w_dr: float = 0.2
    w_dv: float = 0.2
    w_theta: float = 1.2
    w_omega: float = 0.3
    w_h: float = 0.1
    w_dv_cmd: float = 0.02
    w_power: float = 0.0
    w_jerk: float = 0.01
    fail: float = -50.0
    goal: float = 5.0


def compute(obs: np.ndarray, u_prev: np.ndarray, u: np.ndarray, cfg: dict, ps=None):
    """Compute reward using raw, unscaled metrics for alt/ecc and nadir attitude.

    Only three terms are used:
      - Altitude: r_alt = -w_a * |alt_current_m - target_alt_m|
      - Eccentricity: r_e = -w_e * |e_current - e_target|
      - Attitude (nadir): r_theta = -w_theta * (1 - w^2) where q_err[0] = w

    Notes:
      - Requires access to current state `ps` to compute altitude (meters) and eccentricity.
      - If `ps` is None, eccentricity falls back to unscaled obs[7] using env.obs_scales['e'],
        and altitude falls back to semi-major-axis delta from obs[6] times obs_scales['da'].
    """
    env_cfg = cfg.get("env", {})
    rw = RewardConfig(**cfg.get("reward_weights", {}))

    # Attitude error quaternion already in observation
    q_err = obs[8:12]
    w = float(q_err[0])

    # Targets
    target_alt_m = env_cfg.get("target_alt_m", None)
    target_e = float(env_cfg.get("target_ecc", 0.0))

    # Compute current altitude and eccentricity from state if available
    alt_m = None
    e_now = None
    if ps is not None:
        try:
            r = np.asarray(getattr(ps, "r_eci"), dtype=np.float64)
            v = np.asarray(getattr(ps, "v_eci"), dtype=np.float64)
            rnorm = float(np.linalg.norm(r))
            R_E = 6371e3
            alt_m = rnorm - R_E
            # Eccentricity magnitude
            mu = 3.986004418e14
            v2 = float(np.dot(v, v))
            rv = float(np.dot(r, v))
            e_vec = ((v2 - mu / rnorm) * r - rv * v) / mu
            e_now = float(np.linalg.norm(e_vec))
        except Exception:
            alt_m = None
            e_now = None

    # Fallbacks using observation scales if state is unavailable
    if alt_m is None:
        da_s = float(obs[6])
        da_scale = float(env_cfg.get("obs_scales", {}).get("da", 100.0)) or 1.0
        alt_m = da_s * da_scale  # proxy (semi-major axis delta)
    if e_now is None:
        e_s = float(obs[7])
        e_scale = float(env_cfg.get("obs_scales", {}).get("e", 0.01)) or 1.0
        e_now = e_s * e_scale

    # Rewards (negative penalties)
    r_theta = -rw.w_theta * (1.0 - (w * w))
    r_e = -rw.w_e * abs(float(e_now) - float(target_e))
    if target_alt_m is not None:
        r_alt = -rw.w_a * abs(float(alt_m) - float(target_alt_m))
    else:
        r_alt = -rw.w_a * abs(float(alt_m))  # penalize deviation proxy if no explicit target

    rr = {
        "r_alt": r_alt,
        "r_e": r_e,
        "r_theta": r_theta,
    }
    r_total = float(sum(rr.values()))
    return r_total, rr
