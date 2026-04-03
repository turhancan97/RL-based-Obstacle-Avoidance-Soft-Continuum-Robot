from __future__ import annotations

import numpy as np
import pytest

from continuum_rl.env import ContinuumEnv


def test_canonical_observation_contract():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == (env.obs_size,)
    assert env.obs_size == 7 + (2 * env.num_obstacles)
    assert info["observation_schema"] == env.obs_schema
    assert np.isclose(obs[4], env.kappa1, atol=1e-8)
    assert np.isclose(obs[5], env.kappa2, atol=1e-8)
    assert np.isclose(obs[6], env.kappa3, atol=1e-8)
    assert env.observation_space.contains(obs)


def test_non_canonical_mode_rejected():
    with pytest.raises(ValueError, match="Only canonical mode is supported"):
        ContinuumEnv(observation_mode="legacy4d", goal_type="fixed_goal")


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


def test_reset_supports_initial_kappa_override():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    obs, _ = env.reset(seed=0, options={"initial_kappa": [1.0, 2.0, 3.0]})
    assert np.isclose(obs[4], 1.0, atol=1e-8)
    assert np.isclose(obs[5], 2.0, atol=1e-8)
    assert np.isclose(obs[6], 3.0, atol=1e-8)


def test_reset_supports_goal_xy_override():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    obs, _ = env.reset(seed=0, options={"goal_xy": [-0.2, 0.15]})
    assert np.isclose(obs[2], -0.2, atol=1e-8)
    assert np.isclose(obs[3], 0.15, atol=1e-8)


def test_reset_initial_kappa_override_validation():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"initial_kappa": [100.0, 0.0, 0.0]})
