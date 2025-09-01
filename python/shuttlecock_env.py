import os
import ctypes as C
from ctypes import c_double, c_int, c_uint64, c_char_p, POINTER, Structure


class ControlsC(Structure):
    _fields_ = [
        ("eta1_rad", c_double),
        ("eta2_rad", c_double),
        ("thrust_N", c_double),
    ]


class ObsC(Structure):
    _fields_ = [
        ("r_eci", c_double * 3),
        ("v_eci", c_double * 3),
        ("q_BI", c_double * 4),
        ("w_B", c_double * 3),
        ("altitude_m", c_double),
        ("rho", c_double),
        ("aoa_rad", c_double),
        ("aos_rad", c_double),
        ("roll_rad", c_double),
    ]


class StepResultC(Structure):
    _fields_ = [
        ("obs", ObsC),
        ("R_total", c_double),
        ("R_alt", c_double),
        ("R_att", c_double),
        ("R_spin", c_double),
        ("R_effort", c_double),
        ("dwell_alt_frac", c_double),
        ("dwell_att_frac", c_double),
        ("remaining_impulse_Ns", c_double),
        ("substeps", c_int),
        ("done", c_int),
    ]


class EnvHandle(C.Structure):
    pass


def _load_lib(libpath=None):
    if libpath is None:
        # default to cpp/build
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.abspath(os.path.join(here, "..", "cpp", "build", "libshuttlecock_env.so"))
        libpath = cand
    return C.CDLL(libpath)


class ShuttlecockEnvC:
    def __init__(self, libpath=None, config_path=None, align_to_velocity=True, integrator="dp54", debug_csv=None):
        self.lib = _load_lib(libpath)
        # signatures
        self.lib.env_create.restype = C.POINTER(EnvHandle)
        self.lib.env_destroy.argtypes = [C.POINTER(EnvHandle)]
        self.lib.env_load_config.argtypes = [C.POINTER(EnvHandle), c_char_p]
        self.lib.env_set_align_to_velocity.argtypes = [C.POINTER(EnvHandle), c_int]
        self.lib.env_set_integrator.argtypes = [C.POINTER(EnvHandle), c_int]
        self.lib.env_set_debug_csv.argtypes = [C.POINTER(EnvHandle), c_char_p]
        self.lib.env_reset_random.argtypes = [C.POINTER(EnvHandle), c_uint64, c_double, C.POINTER(ObsC)]
        self.lib.env_step_duration.argtypes = [C.POINTER(EnvHandle), ControlsC, c_double, C.POINTER(StepResultC)]
        self.lib.env_step_substeps.argtypes = [C.POINTER(EnvHandle), ControlsC, c_int, C.POINTER(StepResultC)]

        self.h = self.lib.env_create()
        if config_path is not None:
            rc = self.lib.env_load_config(self.h, config_path.encode("utf-8"))
            if rc != 0:
                raise RuntimeError(f"Failed to load config: {config_path}")
        self.lib.env_set_align_to_velocity(self.h, 1 if align_to_velocity else 0)
        self.lib.env_set_integrator(self.h, 0 if integrator.lower()=="rk4" else 1)
        if debug_csv:
            self.lib.env_set_debug_csv(self.h, debug_csv.encode("utf-8"))

    def __del__(self):
        try:
            if getattr(self, "h", None):
                self.lib.env_destroy(self.h)
        except Exception:
            pass

    def reset_random(self, seed: int, jd0_utc: float):
        obs = ObsC()
        rc = self.lib.env_reset_random(self.h, seed, jd0_utc, C.byref(obs))
        if rc != 0:
            raise RuntimeError("env_reset_random failed")
        return obs

    def step_duration(self, eta1, eta2, thrust, duration_s):
        sr = StepResultC()
        u = ControlsC(eta1, eta2, thrust)
        rc = self.lib.env_step_duration(self.h, u, duration_s, C.byref(sr))
        if rc != 0:
            raise RuntimeError("env_step_duration failed")
        return sr

    def step_substeps(self, eta1, eta2, thrust, substeps):
        sr = StepResultC()
        u = ControlsC(eta1, eta2, thrust)
        rc = self.lib.env_step_substeps(self.h, u, substeps, C.byref(sr))
        if rc != 0:
            raise RuntimeError("env_step_substeps failed")
        return sr


# Example Gymnasium-style wrapper (minimal, single-agent, continuous actions)
class GymWrapper:
    def __init__(self, envc: ShuttlecockEnvC, control_dt=5.0, use_substeps=False, substeps=20):
        self.envc = envc
        self.control_dt = control_dt
        self.use_substeps = use_substeps
        self.substeps = substeps

    def reset(self, seed=0, jd0_utc=2451545.0):
        obs = self.envc.reset_random(seed, jd0_utc)
        return self._obs_to_np(obs)

    def step(self, action):
        eta1, eta2, thrust = float(action[0]), float(action[1]), float(action[2])
        if self.use_substeps:
            sr = self.envc.step_substeps(eta1, eta2, thrust, self.substeps)
        else:
            sr = self.envc.step_duration(eta1, eta2, thrust, self.control_dt)
        obs = self._obs_to_np(sr.obs)
        reward = sr.R_total
        terminated = bool(sr.done)
        truncated = False
        info = {
            "R_alt": sr.R_alt, "R_att": sr.R_att, "R_spin": sr.R_spin, "R_effort": sr.R_effort,
            "dwell_alt": sr.dwell_alt_frac, "dwell_att": sr.dwell_att_frac,
            "remaining_impulse": sr.remaining_impulse_Ns, "substeps": sr.substeps,
        }
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _obs_to_np(obs: ObsC):
        import numpy as np
        # Flatten a reasonable observation vector for RL
        return np.array([
            *obs.r_eci, *obs.v_eci, *obs.q_BI, *obs.w_B,
            obs.altitude_m, obs.rho, obs.aoa_rad, obs.aos_rad, obs.roll_rad
        ], dtype=np.float64)

