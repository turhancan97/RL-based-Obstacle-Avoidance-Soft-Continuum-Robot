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


def test_body_collision_terminates_and_sets_info():
    env = ContinuumEnv(
        observation_mode="canonical",
        goal_type="fixed_goal",
        obstacles=[{"x": 0.0, "y": 0.3, "radius": 0.02}],
        collision_mode="body",
        safety_margin=0.005,
    )
    env.reset(seed=0, options={"initial_kappa": [0.0, 0.0, 0.0], "goal_xy": [-0.2, 0.1]})
    _, _, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    assert terminated is True
    assert truncated is False
    assert info["collided"] is True
    assert info["collision_mode"] == "body"
    assert info["collision_count_episode"] >= 1
    assert info["min_clearance"] <= 0.0


def test_tip_mode_is_less_strict_than_body_for_mid_body_obstacle():
    obstacle = [{"x": 0.0, "y": 0.15, "radius": 0.02}]
    env_body = ContinuumEnv(
        observation_mode="canonical",
        goal_type="fixed_goal",
        obstacles=obstacle,
        collision_mode="body",
        safety_margin=0.005,
    )
    env_tip = ContinuumEnv(
        observation_mode="canonical",
        goal_type="fixed_goal",
        obstacles=obstacle,
        collision_mode="tip",
        safety_margin=0.005,
    )
    reset_options = {"initial_kappa": [0.0, 0.0, 0.0], "goal_xy": [-0.2, 0.1]}
    env_body.reset(seed=0, options=reset_options)
    env_tip.reset(seed=0, options=reset_options)

    _, _, terminated_body, _, info_body = env_body.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    _, _, terminated_tip, _, info_tip = env_tip.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    assert terminated_body is True
    assert info_body["collided"] is True
    assert terminated_tip is False
    assert info_tip["collided"] is False


def test_obstacle_radius_override_and_default_radius_behavior():
    obstacle_no_radius = [{"x": 0.0, "y": 0.29}]
    obstacle_with_radius = [{"x": 0.0, "y": 0.29, "radius": 0.02}]

    env_default_radius = ContinuumEnv(
        observation_mode="canonical",
        goal_type="fixed_goal",
        obstacles=obstacle_no_radius,
        obstacle_radius_default=0.001,
        safety_margin=0.0,
        collision_mode="tip",
    )
    env_override_radius = ContinuumEnv(
        observation_mode="canonical",
        goal_type="fixed_goal",
        obstacles=obstacle_with_radius,
        obstacle_radius_default=0.001,
        safety_margin=0.0,
        collision_mode="tip",
    )
    reset_options = {"initial_kappa": [0.0, 0.0, 0.0], "goal_xy": [-0.2, 0.1]}
    env_default_radius.reset(seed=0, options=reset_options)
    env_override_radius.reset(seed=0, options=reset_options)

    _, _, terminated_default, _, info_default = env_default_radius.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    _, _, terminated_override, _, info_override = env_override_radius.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    assert terminated_default is False
    assert info_default["collided"] is False
    assert terminated_override is True
    assert info_override["collided"] is True


def test_collision_never_sets_truncated_when_it_terminates():
    env = ContinuumEnv(
        observation_mode="canonical",
        goal_type="fixed_goal",
        obstacles=[{"x": 0.0, "y": 0.3, "radius": 0.02}],
        collision_mode="body",
        max_episode_steps=1,
    )
    env.reset(seed=0, options={"initial_kappa": [0.0, 0.0, 0.0], "goal_xy": [-0.2, 0.1]})
    _, _, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    assert terminated is True
    assert truncated is False
    assert info["collided"] is True
