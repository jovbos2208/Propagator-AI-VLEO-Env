import numpy as np
from rl_sat.sim import termination
from rl_sat.sim.interfaces import PropagatorState


def test_termination_default_false():
    ps = object()  # placeholder
    obs = np.zeros(23, dtype=np.float32)
    cfg = {}
    ter, tru, fail = termination.check(ps, obs, cfg)
    assert ter is False and tru is False


def test_termination_low_altitude_triggers():
    R_E = 6371e3
    r = np.array([R_E + 150_000.0, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros(3)
    h = np.zeros(3)
    ps = PropagatorState(r, v, q, w, h, soc=1.0, mass=1.0, in_eclipse=False, density=0.0, t=0.0, dt=1.0)
    obs = np.zeros(23, dtype=np.float32)
    cfg = {"env": {"min_perigee_km": 180, "obs_scales": {"theta_deg": 20.0}}}
    ter, tru, fail = termination.check(ps, obs, cfg)
    assert ter is True and fail is True
