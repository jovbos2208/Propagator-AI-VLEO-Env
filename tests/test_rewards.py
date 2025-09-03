import numpy as np
from rl_sat.sim import rewards


def test_reward_compute_runs():
    obs = np.zeros(23, dtype=np.float32)
    u_prev = np.zeros(6, dtype=np.float32)
    u = np.ones(6, dtype=np.float32)
    cfg = {"reward_weights": {}}
    r, terms = rewards.compute(obs, u_prev, u, cfg)
    assert isinstance(r, float)
    assert "r_action" in terms

