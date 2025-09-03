import numpy as np
from rl_sat.envs.satellite_env import RLSatEnv
from rl_sat.sim.interfaces import DummyPropagator


def test_env_reset_step_shapes():
    cfg = {"env": {"dt": 0.5}}
    env = RLSatEnv(DummyPropagator(dt=0.5), cfg)
    obs, info = env.reset(seed=42)
    assert obs.shape == (23,)
    a = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(a)
    assert obs2.shape == (23,)
    assert isinstance(r, float)

