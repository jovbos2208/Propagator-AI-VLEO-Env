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


def compute(obs: np.ndarray, u_prev: np.ndarray, u: np.ndarray, cfg: dict):
    """Compute reward from scaled observation and actions.

    Observation layout (scaled):
      0:3 dr_rtn/dr_scale, 3:6 dv_rtn/dv_scale, 6 da/da_scale, 7 e/e_scale,
      8:12 q_err (w,x,y,z), 12:15 omega/omega_scale, 15:18 h_rw,
      18 density, 19 eclipse_flag, 20 mass, 21 soc, 22 theta_deg/theta_scale
    """
    env_cfg = cfg.get("env", {})
    rw = RewardConfig(**cfg.get("reward_weights", {}))

    # Unpack scaled features
    dr = obs[0:3]
    dv = obs[3:6]
    da = float(obs[6])
    e_s = float(obs[7])
    q_err = obs[8:12]
    omega = obs[12:15]
    h = obs[15:18]
    theta_s = float(obs[22])

    # Core penalties (squared errors)
    r_dr = -rw.w_dr * float(np.dot(dr, dr))
    r_dv = -rw.w_dv * float(np.dot(dv, dv))
    r_a = -rw.w_a * (da * da)
    r_e = -rw.w_e * (e_s * e_s)
    r_theta = -rw.w_theta * (theta_s * theta_s)
    r_omega = -rw.w_omega * float(np.dot(omega, omega))
    r_h = -rw.w_h * float(np.dot(h, h))

    # Effort and smoothness terms depend on action type
    action_type = str(env_cfg.get("action_type", "rtn_torque")).lower()
    if action_type == "rtn_torque":
        a_max = float(env_cfg.get("a_max", 1e-4)) or 1.0
        tau_max = float(env_cfg.get("tau_max", 0.01)) or 1.0
        u_pad = u if u.shape[0] >= 6 else np.pad(u, (0, 6 - u.shape[0]))
        u_prev_pad = u_prev if u_prev.shape[0] >= 6 else np.pad(u_prev, (0, 6 - u_prev.shape[0]))
        a_cmd = float(np.linalg.norm(u_pad[:3])) / a_max
        tau_cmd = float(np.linalg.norm(u_pad[3:6])) / tau_max
        r_dv_cmd = -rw.w_dv_cmd * (a_cmd + 0.1 * tau_cmd)
        du = u_pad - u_prev_pad
        du_scaled = np.concatenate([du[:3] / a_max, du[3:6] / tau_max])
        r_jerk = -rw.w_jerk * float(np.linalg.norm(du_scaled))
    else:
        # angles_only or angles_thrust
        eta_lim = float(env_cfg.get("eta_limit_rad", np.pi/6)) or 1.0
        thrust_max = float(env_cfg.get("thrust_max_N", 0.05)) or 1.0
        # u layout: [eta1, eta2] or [eta1, eta2, thrust_N]
        eta = u[:2]
        thrust = float(u[2]) if u.shape[0] >= 3 else 0.0
        r_dv_cmd = -rw.w_dv_cmd * (abs(thrust) / thrust_max)
        # Jerk scaled per component
        du = u - u_prev
        # Build scale vector matching u length
        scales = np.array([eta_lim, eta_lim] + ([thrust_max] if u.shape[0] >= 3 else []), dtype=float)
        du_scaled = du / (scales + 1e-12)
        r_jerk = -rw.w_jerk * float(np.linalg.norm(du_scaled))

    rr = {
        "r_dr": r_dr,
        "r_dv": r_dv,
        "r_a": r_a,
        "r_e": r_e,
        "r_theta": r_theta,
        "r_omega": r_omega,
        "r_h": r_h,
        "r_dv_cmd": r_dv_cmd,
        "r_jerk": r_jerk,
    }
    rr["r_action"] = r_dv_cmd  # keep legacy key for tests
    r_total = float(sum(rr.values()))
    return r_total, rr
