from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class PropagatorState:
    r_eci: np.ndarray
    v_eci: np.ndarray
    q_be: np.ndarray
    omega_b: np.ndarray
    h_rw: np.ndarray
    soc: float
    mass: float
    in_eclipse: bool
    density: float
    t: float
    dt: float


class PropagatorInterface:
    def reset(self, seed: int, init_cfg: Dict[str, Any]) -> PropagatorState:  # pragma: no cover
        raise NotImplementedError

    def step(self, thrust_rtn: np.ndarray, torque_cmd: np.ndarray) -> PropagatorState:  # pragma: no cover
        raise NotImplementedError

    def set_mode(self, mode: str):  # pragma: no cover
        pass

    def set_constraints(self, cfg: Dict[str, Any]):  # pragma: no cover
        pass


class DummyPropagator(PropagatorInterface):
    """Minimal placeholder propagator to enable basic training loop wiring.
    Models a point mass with simplistic kinematics in ECI and identity attitude.
    """

    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)
        self.t = 0.0
        self.state: PropagatorState | None = None

    def reset(self, seed: int, init_cfg: Dict[str, Any]) -> PropagatorState:
        rng = np.random.default_rng(seed)
        r = np.array([7.0e6, 0.0, 0.0], dtype=np.float64)
        v = np.array([0.0, 7.5e3, 0.0], dtype=np.float64)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        w = np.zeros(3, dtype=np.float64)
        h = np.zeros(3, dtype=np.float64)
        self.t = 0.0
        self.state = PropagatorState(r, v, q, w, h, soc=1.0, mass=100.0, in_eclipse=False, density=1e-11, t=self.t, dt=self.dt)
        return self.state

    def step(self, thrust_rtn: np.ndarray, torque_cmd: np.ndarray) -> PropagatorState:
        assert self.state is not None
        # Simple kinematics: r += v*dt; v += small accel from thrust along ECI x
        a_eci = np.zeros(3)
        a_eci[:3] = thrust_rtn[:3] if thrust_rtn is not None else 0.0
        r = self.state.r_eci + self.state.v_eci * self.dt
        v = self.state.v_eci + a_eci * self.dt
        q = self.state.q_be.copy()
        w = self.state.omega_b + (torque_cmd if torque_cmd is not None else 0.0) * 0.0
        h = self.state.h_rw
        self.t += self.dt
        self.state = PropagatorState(r, v, q, w, h, soc=self.state.soc, mass=self.state.mass,
                                     in_eclipse=False, density=self.state.density, t=self.t, dt=self.dt)
        return self.state

    # Optional support for angles mode so the env can run without Shuttlecock
    def step_angles(self, eta1: float, eta2: float, thrust_N: float) -> PropagatorState:
        assert self.state is not None
        # Interpret thrust as along +X body which aligns with ECI X here; scale by mass to accel
        mass = max(float(self.state.mass), 1e-6)
        a_eci = np.array([float(thrust_N) / mass, 0.0, 0.0], dtype=np.float64)
        r = self.state.r_eci + self.state.v_eci * self.dt
        v = self.state.v_eci + a_eci * self.dt
        q = self.state.q_be.copy()
        w = self.state.omega_b.copy()
        h = self.state.h_rw.copy()
        self.t += self.dt
        self.state = PropagatorState(r, v, q, w, h, soc=self.state.soc, mass=self.state.mass,
                                     in_eclipse=False, density=self.state.density, t=self.t, dt=self.dt)
        return self.state
