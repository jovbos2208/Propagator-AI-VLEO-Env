import numpy as np
from rl_sat.sim import rewards


def test_reward_zero_obs_zero_action_near_zero():
    obs = np.zeros(23, dtype=np.float32)
    u_prev = np.zeros(6, dtype=np.float32)
    u = np.zeros(6, dtype=np.float32)
    cfg = {"reward_weights": {}}
    r, terms = rewards.compute(obs, u_prev, u, cfg)
    assert isinstance(r, float)
    # All terms should be zero
    assert abs(r) < 1e-9


def test_reward_worse_dr_more_negative():
    cfg = {"reward_weights": {"w_dr": 1.0}}
    base = np.zeros(23, dtype=np.float32)
    # small error
    obs1 = base.copy()
    obs1[0:3] = np.array([0.1, 0.0, 0.0])
    # larger error
    obs2 = base.copy()
    obs2[0:3] = np.array([0.5, 0.0, 0.0])
    u_prev = np.zeros(6, dtype=np.float32)
    u = np.zeros(6, dtype=np.float32)
    r1, _ = rewards.compute(obs1, u_prev, u, cfg)
    r2, _ = rewards.compute(obs2, u_prev, u, cfg)
    assert r2 < r1

