from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def test_keras_import_is_safe():
    from Keras import DDPG as keras_ddpg

    assert hasattr(keras_ddpg, "train")
    assert hasattr(keras_ddpg, "evaluate_smoke")


def test_keras_soft_update_and_forward():
    from continuum_rl.env import ContinuumEnv
    from Keras import DDPG as keras_ddpg

    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    num_states = env.obs_size
    num_actions = env.action_space.shape[0]
    upper_bound = float(env.action_space.high[0])

    actor = keras_ddpg.get_actor(num_states, num_actions, upper_bound)
    target_actor = keras_ddpg.get_actor(num_states, num_actions, upper_bound)
    keras_ddpg.soft_update(target_actor, actor, tau=1e-3)

    obs, _ = env.reset(seed=0)
    state = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), 0)
    action = actor(state).numpy().reshape(-1)
    assert action.shape[0] == num_actions
    assert np.all(np.isfinite(action))
