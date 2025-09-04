from __future__ import annotations

import math
from typing import Dict, Any
import numpy as np

from .interfaces import PropagatorInterface, PropagatorState


def _quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Rotate inertial vector v into body using q (w,x,y,z): v_B = q * v * q_conj
    w, x, y, z = q_wxyz
    qv = np.array([0.0, *v], dtype=np.float64)
    q = np.array([w, x, y, z], dtype=np.float64)
    qc = np.array([w, -x, -y, -z], dtype=np.float64)
    def qmul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], dtype=np.float64)
    r = qmul(qmul(q, qv), qc)
    return r[1:]


def _eci_to_rtn(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    r = r_eci / (np.linalg.norm(r_eci) + 1e-12)
    h = np.cross(r_eci, v_eci)
    w = h / (np.linalg.norm(h) + 1e-12)
    t = np.cross(w, r)
    R = np.stack([r, t, w], axis=1)  # columns RTN
    return R


class ShuttlecockPropagator(PropagatorInterface):
    """Adaptor to the C++ shuttlecock env via python.shuttlecock_env.ShuttlecockEnvC.

    Maps actions (thrust in RTN [m/s^2], torque_cmd [Nm]) to [eta1, eta2, thrust_N].
    """

    def __init__(self,
                 config_path: str = "cpp/configs/shuttlecock_250km.json",
                 control_dt: float = 5.0,
                 use_substeps: bool = False,
                 substeps: int = 20,
                 per_orbit_steps: int | None = None,
                 mass_kg: float = 1.0,
                 thrust_max_N: float = 0.05,
                 eta_limit_rad: float = math.pi/6,
                 tau_max: float = 0.01,
                 debug_print: bool = False):
        from python.shuttlecock_env import ShuttlecockEnvC  # lazy import
        dbg_csv = b"runs/shuttle_debug.csv" if debug_print else None
        self.envc = ShuttlecockEnvC(config_path=config_path.encode("utf-8"), align_to_velocity=True, integrator="dp54", debug_csv=dbg_csv)
        self.control_dt = float(control_dt)
        self.use_substeps = bool(use_substeps)
        self.substeps = int(substeps)
        self.per_orbit_steps = per_orbit_steps
        self.mass_kg = float(mass_kg)
        self.thrust_max_N = float(thrust_max_N)
        self.eta_limit_rad = float(eta_limit_rad)
        self.tau_max = float(tau_max)
        self._last_state: PropagatorState | None = None
        self.debug_print = bool(debug_print)

    def set_mode(self, mode: str):
        # Not used by underlying env; kept for API compatibility
        pass

    def set_constraints(self, cfg: Dict[str, Any]):
        # Not used for now
        pass

    def reset(self, seed: int, init_cfg: Dict[str, Any]) -> PropagatorState:
        jd0 = float(init_cfg.get("jd0_utc", 2451545.0))
        self.jd0_utc = jd0
        obs = self.envc.reset_random(int(seed), jd0)
        ps = self._obs_to_state(obs, t=0.0, dt=self.control_dt)
        self._last_state = ps
        return ps

    def step(self, thrust_rtn: np.ndarray, torque_cmd: np.ndarray) -> PropagatorState:
        assert self._last_state is not None, "Call reset() before step()"
        # Determine dt
        if self.per_orbit_steps and self.per_orbit_steps > 0:
            T = max(1e-6, self.envc.estimate_period_s())
            dt = max(1e-3, T / float(self.per_orbit_steps))
        else:
            dt = self.control_dt

        # Map thrust_rtn [m/s^2] to body-X thrust [N]
        r = self._last_state.r_eci
        v = self._last_state.v_eci
        q = self._last_state.q_be
        R = _eci_to_rtn(r, v)  # columns are RTN axes in ECI
        a_eci = R @ np.asarray(thrust_rtn, dtype=np.float64)
        a_body = _quat_rotate(q, a_eci)  # inertial->body using q (body<-ECI)
        a_bx = float(a_body[0])
        thrust_N = float(np.clip(self.mass_kg * a_bx, 0.0, self.thrust_max_N))

        # Map torque_cmd (Nm) to wing angles [-eta_limit, +eta_limit]
        tau = np.asarray(torque_cmd, dtype=np.float64)
        # Scale by tau_max to [-1,1], then to angle limit
        s = np.clip(tau / (self.tau_max if self.tau_max > 0 else 1.0), -1.0, 1.0)
        eta1 = float(np.clip(s[0] * self.eta_limit_rad, -self.eta_limit_rad, self.eta_limit_rad))
        eta2 = float(np.clip(s[1] * self.eta_limit_rad, -self.eta_limit_rad, self.eta_limit_rad))

        # Step the shuttlecock env
        if self.debug_print:
            try:
                ps = self._last_state
                from . import frames as _f
                jd = getattr(self, "jd0_utc", 2451545.0) + float(ps.t) / 86400.0
                r_ecef = _f.eci_to_ecef(ps.r_eci, jd)
                lat, lon, alt = _f.ecef_to_geodetic(r_ecef)
                import os
                print(
                    f"[Shuttlecock pid={os.getpid()}] t={ps.t:.3f}s pre lat={np.degrees(lat):.5f} lon={np.degrees(lon):.5f} alt={alt:.1f}m "
                    f"| dt={dt:.3f}s eta1={eta1:.5f} eta2={eta2:.5f} thrust_N={thrust_N:.6f}"
                )
            except Exception:
                pass
        if self.use_substeps:
            sr = self.envc.step_substeps(eta1, eta2, thrust_N, self.substeps)
            dt_eff = max(1e-9, float(sr.substeps)) * dt  # approximate elapsed time
        else:
            sr = self.envc.step_duration(eta1, eta2, thrust_N, dt)
            dt_eff = dt

        ps = self._obs_to_state(sr.obs, t=self._last_state.t + dt_eff, dt=dt_eff)
        if self.debug_print:
            try:
                from . import frames as _f
                jd = getattr(self, "jd0_utc", 2451545.0) + float(ps.t) / 86400.0
                r_ecef = _f.eci_to_ecef(ps.r_eci, jd)
                lat, lon, alt = _f.ecef_to_geodetic(r_ecef)
                import os
                print(f"[Shuttlecock pid={os.getpid()}] post t={ps.t:.3f}s lat={np.degrees(lat):.5f} lon={np.degrees(lon):.5f} alt={alt:.1f}m")
            except Exception:
                pass
        self._last_state = ps
        return ps

    def step_angles(self, eta1: float, eta2: float, thrust_N: float) -> PropagatorState:
        """Directly apply wing angles and thrust to the underlying env for one control interval.

        Uses same dt logic as step().
        """
        assert self._last_state is not None, "Call reset() before step_angles()"
        # Determine dt
        if self.per_orbit_steps and self.per_orbit_steps > 0:
            T = max(1e-6, self.envc.estimate_period_s())
            dt = max(1e-3, T / float(self.per_orbit_steps))
        else:
            dt = self.control_dt

        eta1c = float(np.clip(eta1, -self.eta_limit_rad, self.eta_limit_rad))
        eta2c = float(np.clip(eta2, -self.eta_limit_rad, self.eta_limit_rad))
        thrust_c = float(np.clip(thrust_N, 0.0, self.thrust_max_N))

        if self.debug_print:
            try:
                ps0 = self._last_state
                from . import frames as _f
                jd = getattr(self, "jd0_utc", 2451545.0) + float(ps0.t) / 86400.0
                r_ecef = _f.eci_to_ecef(ps0.r_eci, jd)
                lat, lon, alt = _f.ecef_to_geodetic(r_ecef)
                import os
                print(
                    f"[Shuttlecock pid={os.getpid()}] t={ps0.t:.3f}s pre lat={np.degrees(lat):.5f} lon={np.degrees(lon):.5f} alt={alt:.1f}m "
                    f"| dt={dt:.3f}s angles=({eta1c:.5f},{eta2c:.5f}) thrust_N={thrust_c:.6f}"
                )
            except Exception:
                pass
        if self.use_substeps:
            sr = self.envc.step_substeps(eta1c, eta2c, thrust_c, self.substeps)
            dt_eff = max(1e-9, float(sr.substeps)) * dt
        else:
            sr = self.envc.step_duration(eta1c, eta2c, thrust_c, dt)
            dt_eff = dt

        ps = self._obs_to_state(sr.obs, t=self._last_state.t + dt_eff, dt=dt_eff)
        if self.debug_print:
            try:
                from . import frames as _f
                jd = getattr(self, "jd0_utc", 2451545.0) + float(ps.t) / 86400.0
                r_ecef = _f.eci_to_ecef(ps.r_eci, jd)
                lat, lon, alt = _f.ecef_to_geodetic(r_ecef)
                import os
                print(f"[Shuttlecock pid={os.getpid()}] post t={ps.t:.3f}s lat={np.degrees(lat):.5f} lon={np.degrees(lon):.5f} alt={alt:.1f}m")
            except Exception:
                pass
        self._last_state = ps
        return ps

    @staticmethod
    def _obs_to_state(obs, t: float, dt: float) -> PropagatorState:
        r = np.array([obs.r_eci[0], obs.r_eci[1], obs.r_eci[2]], dtype=np.float64)
        v = np.array([obs.v_eci[0], obs.v_eci[1], obs.v_eci[2]], dtype=np.float64)
        q = np.array([obs.q_BI[0], obs.q_BI[1], obs.q_BI[2], obs.q_BI[3]], dtype=np.float64)
        w = np.array([obs.w_B[0], obs.w_B[1], obs.w_B[2]], dtype=np.float64)
        h = np.zeros(3, dtype=np.float64)
        return PropagatorState(
            r_eci=r,
            v_eci=v,
            q_be=q,
            omega_b=w,
            h_rw=h,
            soc=1.0,
            mass=1.0,
            in_eclipse=False,
            density=float(obs.rho),
            t=float(t),
            dt=float(dt),
        )
