from __future__ import annotations

import numpy as np


def check(ps, obs: np.ndarray, cfg: dict):
    """Return (terminated, truncated, fail_flag).

    Policy: Only enforce altitude deviation from a target altitude; truncate by episode time limit.
    - Terminate (fail) if |alt - target_alt_m| > alt_tol_m (default 50 km).
    - Truncate (no fail) when episode_orbits * period_s is reached.
    """
    env = cfg.get("env", {})

    # Altitude deviation constraint relative to target altitude
    r = getattr(ps, "r_eci", None)
    target_alt_m = env.get("target_alt_m", None)
    alt_tol_m = float(env.get("alt_tol_m", 50_000.0))
    if r is not None and target_alt_m is not None:
        rnorm = float(np.linalg.norm(np.asarray(r)))
        R_E = 6371e3
        alt = rnorm - R_E
        if abs(float(alt) - float(target_alt_m)) > alt_tol_m:
            return True, False, True

    # Episode time limit in number of orbits (truncation)
    max_orbits = float(env.get("episode_orbits", 0.0) or 0.0)
    period_s = float(env.get("period_s", 0.0) or 0.0)
    t = float(getattr(ps, "t", 0.0) or 0.0)
    if max_orbits > 0.0 and period_s > 0.0 and t >= max_orbits * period_s:
        return False, True, False

    return False, False, False
