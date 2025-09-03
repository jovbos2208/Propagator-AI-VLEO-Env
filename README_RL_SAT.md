# RL Satellite Station-Keeping & Attitude Control — Implementation Plan

This plan describes **exact files, functions, and steps** to add a Gymnasium-compatible RL environment for **LEO orbit (200–400 km) altitude keeping** and **attitude control**, plus training & evaluation scripts. Hand this to your code assistant to implement exactly in your project.

---

## 0) High-level Goals

- Provide a **single Gymnasium environment** that simulates coupled **orbit** and **attitude** dynamics via your propagator.
- Expose a **clean API** for:
  - Observation/state building (RTN errors for orbit + quaternion error for attitude).
  - Action application (thruster / torque commands) with **safety filtering**.
  - Reward calculation (tracking + efficiency + constraints) with configurable weights.
- Include **curriculum modes** (attitude-only → orbit-only → coupled).
- Ship **ready-to-run PPO/SAC training** and **evaluation** scripts, plus unit tests.

---

## 1) Directory Layout

Create the following within your repository root (adjust `src/` if you use different layout):

```
.
├─ src/
│  ├─ rl_sat/
│  │  ├─ envs/
│  │  │  ├─ satellite_env.py
│  │  │  ├─ safety_filter.py
│  │  │  └─ __init__.py
│  │  ├─ sim/
│  │  │  ├─ interfaces.py           # adaptor interface to your propagator
│  │  │  ├─ frames.py               # RTN/LVLH transforms, helpers
│  │  │  ├─ rewards.py              # reward terms
│  │  │  ├─ termination.py          # termination & constraint checks
│  │  │  ├─ normalization.py        # running stats & scaling
│  │  │  └─ utils.py
│  │  ├─ configs/
│  │  │  ├─ default.yaml
│  │  │  ├─ attitude_only.yaml
│  │  │  ├─ orbit_only.yaml
│  │  │  └─ coupled.yaml
│  │  └─ __init__.py
│  ├─ training/
│  │  ├─ train_ppo.py
│  │  ├─ train_sac.py
│  │  ├─ eval_policy.py
│  │  └─ callbacks.py
│  └─ __init__.py
├─ tests/
│  ├─ test_env_shapes.py
│  ├─ test_rewards.py
│  ├─ test_safety_filter.py
│  └─ test_termination.py
├─ scripts/
│  ├─ run_attitude_only.sh
│  ├─ run_orbit_only.sh
│  └─ run_coupled.sh
├─ README_RL_SAT.md                 # (this file’s content)
└─ requirements.txt
```

---

## 2) Interfaces & Assumptions

### 2.1 Propagator adaptor (`src/rl_sat/sim/interfaces.py`)

Define a **thin adaptor** so the env is independent of the propagator implementation.

```python
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

@dataclass
class PropagatorState:
    # Absolute Cartesian state in ECI (m, m/s)
    r_eci: np.ndarray  # shape (3,)
    v_eci: np.ndarray  # shape (3,)

    # Attitude quaternion body->ECI (w, x, y, z) and body rates (rad/s)
    q_be: np.ndarray   # shape (4,)
    omega_b: np.ndarray  # shape (3,)

    # Actuator states
    h_rw: np.ndarray   # reaction wheel angular momenta, shape (3,)

    # Power/housekeeping (optional)
    soc: float         # 0..1
    mass: float        # kg

    # Environment
    in_eclipse: bool
    density: float     # kg/m^3
    t: float           # seconds since episode start
    dt: float          # last step dt
```

```python
class PropagatorInterface:
    def reset(self, seed: int, init_cfg: Dict[str, Any]) -> PropagatorState: ...
    def step(self, thrust_rtn: np.ndarray, torque_cmd: np.ndarray) -> PropagatorState: ...
    def set_mode(self, mode: str): ...  # "attitude_only" | "orbit_only" | "coupled"
    def set_constraints(self, cfg: Dict[str, Any]): ...
```

Your concrete adaptor should convert **actions** (RTN thrust, body torque) into the propagator’s native inputs.

### 2.2 Frames utilities (`src/rl_sat/sim/frames.py`)

- `eci_to_rtn(r_eci, v_eci)` → rotation matrix ECI→RTN
- `quat_err(q_body_to_target, q_body_to_eci)` → error quaternion body→target
- `target_quat(mode, r_eci, v_eci)` → target pointing (e.g., nadir/LVLH z-axis down)

**Target frames:**
- *Nadir-pointing LVLH:* body +Z to −R (toward Earth), +X along velocity, +Y completes RHS.
- Provide helpers to compute **reference trajectory** \(\mathcal{R}(t)\) or accept external reference (Δa/e targets = 0).

---

## 3) Observation Space (state)

### 3.1 Content (normalized; shapes in parentheses)

- `dr_rtn (3)` = position error in RTN [m] vs. reference
- `dv_rtn (3)` = velocity error in RTN [m/s]
- `da (1)` = semi-major-axis error [m]
- `e (1)`  = scalar eccentricity
- `q_err (4)` = quaternion error body→target
- `omega_b (3)` = body rates [rad/s]
- `h_rw (3)` = reaction wheel momenta [Nms]
- `density (1)` = atmospheric density proxy
- `in_eclipse (1)` = {0,1}
- `mass (1)` = propellant mass fraction (0..1 scaled)
- `soc (1)` = battery state of charge (0..1)

> Maintain **running mean/std** and clip to \([-5,5]\). Optionally stack last **K=2–4 frames** for temporal context.

### 3.2 Space (Gymnasium)

- `Box(low=-5, high=5, shape=(23,))` for default 23-dim state (adjust if you add/remove features).

---

## 4) Action Space

- `thrust_rtn (3)` = commanded specific acceleration in RTN \([m/s^2]\), clipped to `a_max`.
- `torque_cmd (3)` = body torque for reaction wheels \([Nm]\), clipped to `tau_max`.

Represented as `Box(low=-1, high=1, shape=(6,))` and scaled to physical limits inside `safety_filter.py`.

---

## 5) Rewards

Implement in `src/rl_sat/sim/rewards.py` with pure functions:

```
r = r_orbit(da, e, dr, dv) + r_attitude(theta, omega) + r_wheels(h) \
    + r_effort(dv_cmd, power_frac, jerk) + r_constraints(flags) + r_goal(flags)
```

**Orbit terms**
- `r_a = - w_a * (da/A_scale)^2`
- `r_e = - w_e * (e/e_scale)^2`
- `r_rtn = - w_dr * (||dr||/R_scale)^2 - w_dv * (||dv||/V_scale)^2`

**Attitude terms**
- `theta = 2*arccos(q_err.w)` (clamped to [0, π])
- `r_att = - w_theta*(theta/theta_scale)^2 - w_omega*(||omega||/Omega_scale)^2`
- `r_h = - w_h*(||h||/h_max)^2`

**Effort & smoothness**
- `r_dv_cmd = - w_dv_cmd * ||thrust_rtn||` (or power)
- `r_power = - w_power * (P_use/P_avail)` (optional)
- `r_jerk = - w_jerk * ||u_t - u_{t-1}||`

**Constraints & events**
- `terminal_fail = large negative` if perigee < min_alt, keep-out violated, SOC < min, etc.
- `goal_bonus = small positive` if thresholds held for G consecutive steps.

Provide a `RewardConfig` dataclass and unit-tested **scaling defaults**.

---

## 6) Termination & Constraints (`termination.py`)

- **Fail if**: perigee altitude `< 180 km`, attitude error `theta > 60°` for `> T_violate`, `soc < 0.1`, `||dr|| > 50 km`.
- **Success window** (optional): maintain `theta < 5°`, `||dr|| < 2 km`, `e < 0.005` for `T_goal`.
- `truncate` on max steps or numerical errors.

---

## 7) Safety Filter (`safety_filter.py`)

- Scale and **project** actions into feasible set:
  - `||thrust_rtn|| ≤ a_max`; zero thrust in eclipse if required.
  - `||torque_cmd|| ≤ tau_max`; wheel desaturation rules (optionally schedule magnetorquers).
- Optional **QP-based** projection with keep-out cones.

Provide `apply(action, state, cfg) -> scaled_action, info` and include flags for clipping events.

---

## 8) Environment (`envs/satellite_env.py`)

- Gymnasium `Env` with `reset(seed, options)` and `step(action)`.
- Internals:
  - Holds `PropagatorInterface` instance.
  - Builds observation via `frames.py` + `normalization.py`.
  - Applies `safety_filter` before calling propagator.
  - Computes reward via `rewards.py` and checks `termination.py`.
  - Supports modes: `"attitude_only"`, `"orbit_only"`, `"coupled"` (via config or `env.set_mode`).
- `info` dict fields: all raw metrics (`da`, `e`, `theta`, `omega_norm`, `dv_cmd`, `power_frac`, flags).

---

## 9) Configs (`configs/*.yaml`)

Example keys:
```yaml
env:
  mode: coupled  # attitude_only | orbit_only | coupled
  dt: 1.0
  frame_stack: 3
  obs_scales: {dr: 5000.0, dv: 0.1, da: 100.0, e: 0.01, theta_deg: 20.0, omega: 0.1}
  a_max: 1e-4          # m/s^2
  tau_max: 0.01        # Nm
  min_perigee_km: 180
  max_att_err_deg: 60
  goal_window: {theta_deg: 5, dr_m: 2000, e: 0.005, steps: 300}

reward_weights:
  w_a: 1.0; w_e: 0.5; w_dr: 0.2; w_dv: 0.2
  w_theta: 1.2; w_omega: 0.3; w_h: 0.1
  w_dv_cmd: 0.02; w_power: 0.0; w_jerk: 0.01
  fail: -50.0; goal: +5.0
```

---

## 10) Training Scripts (`src/training`)

- `train_ppo.py` and `train_sac.py` (Stable-Baselines3 or CleanRL):
  - Parse config path.
  - Register env `id="RLSat-v0"` via entry point.
  - Vectorized envs, `gamma=0.995–0.999`, normalize obs/reward.
  - Save checkpoints, TensorBoard logs, and final policy.

- `eval_policy.py`:
  - Load policy, run N episodes, save metrics and plots (θ, Δa, e, ΣΔv, wheel dumps).

- `callbacks.py`:
  - Early stopping if constraint-violations spike or reward plateaus.
  - Periodic evaluation on deterministic seeds.

Shell runners in `scripts/` call training with appropriate configs.

---

## 11) Tests (`tests/`)

- `test_env_shapes.py`: obs/action shapes, spaces, determinism on fixed seed.
- `test_rewards.py`: unit tests for each reward term with known inputs.
- `test_safety_filter.py`: clipping & projection behaviors.
- `test_termination.py`: triggers for perigee, attitude, SOC thresholds.

Run via `pytest -q` in CI.

---

## 12) Requirements

Minimal (adjust to your stack):

```
gymnasium>=0.29
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
stable-baselines3>=2.3.0
tensorboard>=2.15
pytest>=7.0
```

---

## 13) Implementation Checklist

- [ ] Create directories and placeholder modules as above.
- [ ] Implement `PropagatorInterface` adaptor and wire to your dynamics.
- [ ] Implement `frames.py`: ECI↔RTN transforms, target quaternion, error quaternion.
- [ ] Implement `rewards.py` with dataclasses + unit tests.
- [ ] Implement `termination.py` with thresholds and tests.
- [ ] Implement `safety_filter.py` with scaling & projection + tests.
- [ ] Build `satellite_env.py` (Gymnasium) and register `RLSat-v0`.
- [ ] Add YAML configs (attitude_only / orbit_only / coupled).
- [ ] Training scripts (PPO/SAC) + eval + callbacks.
- [ ] Scripts in `scripts/` to run experiments.
- [ ] Fill `requirements.txt`; ensure `pip install -r requirements.txt` works.
- [ ] CI: run lint + tests.

---

## 14) Acceptance Criteria

- `pytest` passes (≥ 4 tests).
- `train_ppo.py --config configs/attitude_only.yaml` reaches median pointing error `< 5°` within 2M steps.
- `train_sac.py --config configs/coupled.yaml` maintains `Δa` RMS `< 200 m` over 1-orbit eval and `ΣΔv` finite.
- `eval_policy.py` outputs JSON metrics and plots to `runs/<exp_id>/`.

---

## 15) Notes & Extensions

- Add **RNN policy** option (GRU) for partial observability (density memory).
- Optional **QP safety layer** and **wheel desaturation scheduling** via magnetorquers.
- If actions should be **Δv pulses** instead of continuous thrust, switch the thrust head to 3 discreet gates (prog/retro/cross) + duration param and adapt reward to encourage sparse burns.

---

## 16) Minimal Code Stubs (copy in place)

**`src/rl_sat/envs/__init__.py`**
```python
from .satellite_env import RLSatEnv
```

**`src/rl_sat/envs/satellite_env.py`**
```python
import gymnasium as gym
import numpy as np
from ..sim.interfaces import PropagatorInterface, PropagatorState
from ..sim import frames, rewards, termination, normalization
from .safety_filter import apply as apply_safety

class RLSatEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, propagator: PropagatorInterface, cfg: dict):
        self.propagator = propagator
        self.cfg = cfg
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(23,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.norm = normalization.RunningNorm(self.observation_space.shape[0])
        self.u_prev = np.zeros(6, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_cfg = (options or {}).get("init_cfg", {})
        ps: PropagatorState = self.propagator.reset(seed or 0, init_cfg)
        obs = self._build_obs(ps)
        self.u_prev[:] = 0.0
        return obs, {}

    def step(self, action):
        u_scaled, sf_info = apply_safety(action, self.cfg)
        ps = self.propagator.step(u_scaled[:3], u_scaled[3:])
        obs = self._build_obs(ps)
        info = {"sf": sf_info}
        r, terms = rewards.compute(obs, self.u_prev, u_scaled, self.cfg)
        info.update(terms)
        terminated, truncated, fail_flag = termination.check(ps, obs, self.cfg)
        self.u_prev = u_scaled
        return obs, float(r), bool(terminated), bool(truncated), info

    def _build_obs(self, ps: PropagatorState):
        # TODO: compute RTN errors vs. reference, quaternion error, etc.
        vec = np.zeros(23, dtype=np.float32)
        return self.norm(vec)
```

**`src/rl_sat/envs/safety_filter.py`**
```python
import numpy as np

def apply(action, cfg):
    a = np.asarray(action, dtype=np.float32).copy()
    a[:3] *= cfg["env"]["a_max"]
    a[3:] *= cfg["env"]["tau_max"]
    # clamp
    a[:3] = np.clip(a[:3], -cfg["env"]["a_max"], cfg["env"]["a_max"])
    a[3:] = np.clip(a[3:], -cfg["env"]["tau_max"], cfg["env"]["tau_max"])
    info = {"clipped": False}
    return a, info
```

**`src/rl_sat/sim/normalization.py`**
```python
import numpy as np

class RunningNorm:
    def __init__(self, dim, eps=1e-8):
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        self.count = eps

    def __call__(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def update(self, x):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count
```

---

## 17) Run Commands

```
# Install
pip install -r requirements.txt

# Train attitude-only
python -m src.training.train_ppo --config src/rl_sat/configs/attitude_only.yaml

# Train orbit-only
python -m src.training.train_sac --config src/rl_sat/configs/orbit_only.yaml

# Train coupled
python -m src.training.train_ppo --config src/rl_sat/configs/coupled.yaml

# Evaluate
python -m src.training.eval_policy --ckpt runs/coupled/best.zip --episodes 10
```

---

## 18) TODO Tags For Your Assistant

- `# TODO: frames.py` implement ECI↔RTN and LVLH target.
- `# TODO: rewards.compute` assemble terms & scales from cfg.
- `# TODO: termination.check` perigee, keep-out, SOC, etc.
- `# TODO: interfaces.PropagatorInterface` wire into your simulator.
- `# TODO: satellite_env._build_obs` compute and normalize full state.

---

**End of plan.** Good luck & clear skies 🚀
