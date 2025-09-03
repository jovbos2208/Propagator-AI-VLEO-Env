from __future__ import annotations

import numpy as np


def check(ps, obs: np.ndarray, cfg: dict):
    """Return (terminated, truncated, fail_flag) based on constraints.

    Uses both propagator state (for altitude) and observation (for attitude/dispersion).
    """
    env = cfg.get("env", {})
    # Altitude constraint (approximate perigee with instantaneous radius)
    r = getattr(ps, "r_eci", None)
    if r is not None:
        rnorm = float(np.linalg.norm(np.asarray(r)))
        R_E = 6371e3
        alt = rnorm - R_E
        min_alt_m = float(env.get("min_perigee_km", 180)) * 1000.0
        if alt < min_alt_m:
            return True, False, True

    # Attitude error (theta)
    theta_s = float(obs[22]) if obs.shape[0] > 22 else 0.0
    theta_deg_scale = float(env.get("obs_scales", {}).get("theta_deg", 20.0) or 1.0)
    theta_deg = theta_s * theta_deg_scale
    max_theta = float(env.get("max_att_err_deg", 1e9))
    if theta_deg > max_theta:
        return True, False, True

    # Dispersion in RTN position (dr)
    dr = obs[0:3]
    dr_scale = float(env.get("obs_scales", {}).get("dr", 5000.0) or 1.0)
    dr_m = float(np.linalg.norm(dr) * dr_scale)
    if dr_m > 50_000.0:  # 50 km window
        return True, False, True

    # SOC (only if configured)
    soc_min = cfg.get("env", {}).get("soc_min")
    if soc_min is not None:
        soc = float(obs[21]) if obs.shape[0] > 21 else 1.0
        if soc < float(soc_min):
            return True, False, True

    # Episode time limit in number of orbits (truncation)
    env = cfg.get("env", {})
    max_orbits = float(env.get("episode_orbits", 0.0) or 0.0)
    period_s = float(env.get("period_s", 0.0) or 0.0)
    t = float(getattr(ps, "t", 0.0) or 0.0)
    if max_orbits > 0.0 and period_s > 0.0:
        if t >= max_orbits * period_s:
            return False, True, False

    return False, False, False
