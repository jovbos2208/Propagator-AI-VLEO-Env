import numpy as np
from rl_sat.envs.safety_filter import apply


def test_safety_filter_scales_and_clips():
    cfg = {"env": {"a_max": 1e-4, "tau_max": 0.01}}
    a = np.array([2.0, -2.0, 0.5, 2.0, -2.0, 0.0], dtype=np.float32)
    u, info = apply(a, cfg)
    assert np.all(np.isfinite(u))
    assert np.all(u[:3] <= cfg["env"]["a_max"] + 1e-12)
    assert np.all(u[:3] >= -cfg["env"]["a_max"] - 1e-12)
    assert np.all(u[3:] <= cfg["env"]["tau_max"] + 1e-12)
    assert np.all(u[3:] >= -cfg["env"]["tau_max"] - 1e-12)

