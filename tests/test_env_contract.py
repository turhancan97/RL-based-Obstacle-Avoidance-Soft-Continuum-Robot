from __future__ import annotations

import numpy as np

from continuum_rl.env import ContinuumEnv


def test_canonical_observation_contract():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == (env.obs_size,)
    assert env.obs_size == 4 + (2 * env.num_obstacles)
    assert env.observation_space.contains(obs)


def test_legacy4d_observation_contract():
    env = ContinuumEnv(observation_mode="legacy4d", goal_type="fixed_goal")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    assert env.obs_size == 4
    assert env.observation_space.contains(obs)


def test_stop_7_branch_does_not_crash():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    env.reset(seed=0)
    env.kappa1 = env.kappa_max
    env.kappa2 = env.kappa_max
    env.kappa3 = env.kappa_max
    env.stop = 7

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    assert obs.shape == (env.obs_size,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
