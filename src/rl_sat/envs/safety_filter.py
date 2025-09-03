import numpy as np


def apply(action, cfg, state=None):
    """Scale and constrain the action.

    - Clips input to [-1, 1]
    - Scales to physical limits (a_max, tau_max)
    - Optional eclipse thrust gating if env.gate_thrust_in_eclipse is True
    """
    env_cfg = cfg.get("env", {})
    mode = str(env_cfg.get("action_type", "rtn_torque")).lower()
    gate_eclipse = bool(env_cfg.get("gate_thrust_in_eclipse", False))
    a_in = np.asarray(action, dtype=np.float32).copy()
    a_in = np.clip(a_in, -1.0, 1.0)

    if mode == "rtn_torque":
        a_max = float(env_cfg.get("a_max", 1e-4))
        tau_max = float(env_cfg.get("tau_max", 0.01))
        a = a_in.copy()
        a[:3] *= a_max
        a[3:] *= tau_max
        gated = False
        if gate_eclipse and state is not None:
            try:
                if bool(getattr(state, "in_eclipse", False)):
                    a[:3] = 0.0
                    gated = True
            except Exception:
                pass
        info = {"clipped": True, "gated_eclipse": gated, "mode": mode}
        return a, info

    elif mode in ("angles_only", "angles_thrust"):
        eta_lim = float(env_cfg.get("eta_limit_rad", np.pi/6))
        thrust_max = float(env_cfg.get("thrust_max_N", 0.05))
        a = np.zeros(3, dtype=np.float32)
        # angles in [-eta_lim, +eta_lim]
        a[0] = a_in[0] * eta_lim
        a[1] = a_in[1] * eta_lim
        # thrust command (0..thrust_max) if provided
        thrust = 0.0
        if mode == "angles_thrust":
            frac = 0.5 * (a_in[2] + 1.0)  # map [-1,1] -> [0,1]
            thrust = float(frac * thrust_max)
        gated = False
        if gate_eclipse and state is not None:
            try:
                if bool(getattr(state, "in_eclipse", False)):
                    thrust = 0.0
                    gated = True
            except Exception:
                pass
        a[2] = thrust
        info = {"clipped": True, "gated_eclipse": gated, "mode": mode}
        return a, info

    else:
        # Fallback: pass through as rtn_torque scaling
        a_max = float(env_cfg.get("a_max", 1e-4))
        tau_max = float(env_cfg.get("tau_max", 0.01))
        a = a_in.copy()
        a[:3] *= a_max
        a[3:] *= tau_max
        info = {"clipped": True, "gated_eclipse": False, "mode": mode}
        return a, info
