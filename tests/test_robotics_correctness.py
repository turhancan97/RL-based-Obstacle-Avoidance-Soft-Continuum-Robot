from __future__ import annotations

import types

import numpy as np

from continuum_rl.env import ContinuumEnv
from kinematics.forward_velocity_kinematics import three_section_planar_robot, trans_mat_cc


def test_three_section_planar_robot_zero_curvature_is_finite():
    T = three_section_planar_robot(0.0, 0.0, 0.0, [0.1, 0.1, 0.1])
    assert T.shape == (4, 4)
    assert np.all(np.isfinite(T))
    assert np.isclose(T[1, 3], 0.3, atol=1e-6)


def test_trans_mat_cc_near_zero_is_continuous():
    T_zero = trans_mat_cc(0.0, 0.1)
    T_eps = trans_mat_cc(1e-9, 0.1)
    assert np.all(np.isfinite(T_zero))
    assert np.all(np.isfinite(T_eps))
    assert np.allclose(T_zero[:, 12:14], T_eps[:, 12:14], atol=1e-6)


def test_step_with_zero_curvature_remains_finite():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    env.reset(seed=0)
    env.kappa1 = 0.0
    env.kappa2 = 0.0
    env.kappa3 = 0.0
    env._update_stop_mode()

    obs, reward, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_stop_mask_applies_to_kappa_updates():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    env.reset(seed=0)
    env.kappa1 = env.kappa_max
    env.kappa2 = 0.0
    env.kappa3 = 0.0
    env._update_stop_mode()
    assert env.stop == 1

    action = np.array([1.0, 0.5, -0.5], dtype=np.float32)
    before = (env.kappa1, env.kappa2, env.kappa3)
    env.step(action, reward_function="step_minus_weighted_euclidean")

    assert np.isclose(env.kappa1, before[0], atol=1e-8)
    assert np.isclose(env.kappa2, before[1] + (action[1] * env.dt), atol=1e-8)
    assert np.isclose(env.kappa3, before[2] + (action[2] * env.dt), atol=1e-8)


def test_reward_is_computed_from_post_action_state():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    obs, _ = env.reset(seed=0)
    env.kappa1, env.kappa2, env.kappa3 = 2.0, 1.5, 1.0
    env._update_stop_mode()
    T = three_section_planar_robot(env.kappa1, env.kappa2, env.kappa3, env.l)
    x, y = float(T[0, 3]), float(T[1, 3])
    env._full_state = env._compose_full_state(x, y, float(obs[2]), float(obs[3]))
    pre_xy = np.array([x, y], dtype=np.float64)
    captured: dict[str, tuple[float, float]] = {}
    original = env._compute_reward_cost

    def wrapped(self, x, y, goal_x, goal_y, reward_function):
        captured["xy"] = (x, y)
        return original(x, y, goal_x, goal_y, reward_function)

    env._compute_reward_cost = types.MethodType(wrapped, env)
    step_obs, _, _, _, _ = env.step(
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    assert "xy" in captured
    assert np.allclose(np.array(captured["xy"], dtype=np.float64), step_obs[:2], atol=1e-10)
    assert not np.allclose(np.array(captured["xy"], dtype=np.float64), pre_xy, atol=1e-10)


def test_reset_seed_is_deterministic():
    env_a = ContinuumEnv(observation_mode="canonical", goal_type="random_goal")
    env_b = ContinuumEnv(observation_mode="canonical", goal_type="random_goal")
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert np.allclose(obs_a, obs_b, atol=1e-7)


def test_max_episode_steps_sets_truncated_flag():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal", max_episode_steps=2)
    obs, _ = env.reset(seed=0)
    x, y = float(obs[0]), float(obs[1])
    env._full_state = env._compose_full_state(x, y, x + 0.1, y + 0.1)
    env.previous_error = 1.0
    env.error = 1.0
    env.initial_distance = 1.0

    _, _, terminated_1, truncated_1, _ = env.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )
    _, _, terminated_2, truncated_2, _ = env.step(
        np.zeros(3, dtype=np.float32),
        reward_function="step_minus_weighted_euclidean",
    )

    assert terminated_1 is False
    assert truncated_1 is False
    assert terminated_2 is False
    assert truncated_2 is True
    assert env.step_count == 2


def test_clearance_penalty_applies_to_all_reward_modes():
    reward_modes = [
        "step_error_comparison",
        "step_minus_euclidean_square",
        "step_minus_weighted_euclidean",
        "step_distance_based",
    ]
    for reward_mode in reward_modes:
        env_near = ContinuumEnv(
            observation_mode="canonical",
            goal_type="fixed_goal",
            obstacles=[{"x": 0.0, "y": 0.285, "radius": 0.002}],
            obstacle_radius_default=0.002,
            safety_margin=0.0,
            clearance_penalty_weight=1.0,
            collision_penalty=0.0,
            collision_mode="tip",
        )
        env_far = ContinuumEnv(
            observation_mode="canonical",
            goal_type="fixed_goal",
            obstacles=[{"x": 0.3, "y": -0.3, "radius": 0.002}],
            obstacle_radius_default=0.002,
            safety_margin=0.0,
            clearance_penalty_weight=1.0,
            collision_penalty=0.0,
            collision_mode="tip",
        )

        reset_options = {"initial_kappa": [0.0, 0.0, 0.0], "goal_xy": [-0.2, 0.1]}
        env_near.reset(seed=42, options=reset_options)
        env_far.reset(seed=42, options=reset_options)
        _, reward_near, _, _, _ = env_near.step(np.zeros(3, dtype=np.float32), reward_function=reward_mode)
        _, reward_far, _, _, _ = env_far.step(np.zeros(3, dtype=np.float32), reward_function=reward_mode)
        assert reward_near < reward_far, f"Expected stronger penalty near obstacle for mode={reward_mode}"
