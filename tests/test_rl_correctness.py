from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from continuum_rl.artifacts import write_metadata
from continuum_rl.env import ContinuumEnv
from Pytorch.ddpg_agent import OUNoise


def test_observation_schema_and_shape_include_kappa():
    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    obs, info = env.reset(seed=0)
    assert env.obs_schema == "canonical_v3"
    assert env.obs_size == 7 + (2 * env.num_obstacles)
    assert obs.shape == (env.obs_size,)
    assert np.isclose(obs[4], env.kappa1, atol=1e-8)
    assert np.isclose(obs[5], env.kappa2, atol=1e-8)
    assert np.isclose(obs[6], env.kappa3, atol=1e-8)
    assert info["observation_schema"] == "canonical_v3"


def test_pytorch_checkpoint_validation_requires_obs_schema_metadata(tmp_path):
    from Pytorch.ddpg import _expected_metadata, validate_checkpoint_compatibility

    env = ContinuumEnv(observation_mode="canonical", goal_type="fixed_goal")
    expected = _expected_metadata(
        env=env,
        goal_type="fixed_goal",
        reward_file="reward_step_minus_weighted_euclidean",
        reward_function="step_minus_weighted_euclidean",
    )

    artifact = tmp_path / "checkpoint_actor.pth"
    artifact.write_bytes(b"stub")
    # missing obs_schema on purpose
    write_metadata(
        artifact,
        {
            "framework": "pytorch",
            "state_dim": expected["state_dim"],
            "obstacle_count": expected["obstacle_count"],
            "goal_type": expected["goal_type"],
            "reward_function": expected["reward_function"],
        },
    )

    with pytest.raises(ValueError, match="canonical_v3"):
        validate_checkpoint_compatibility(artifact, expected)


def test_keras_replay_buffer_tracks_done_flags():
    pytest.importorskip("tensorflow")
    from Keras.DDPG import ReplayBuffer

    buffer = ReplayBuffer(num_states=5, num_actions=3, buffer_capacity=32, batch_size=8)
    for idx in range(10):
        state = np.ones(5, dtype=np.float32) * idx
        action = np.ones(3, dtype=np.float32)
        reward = float(idx)
        next_state = state + 1.0
        done = float(idx % 2)
        buffer.record((state, action, reward, next_state, done))

    sampled = buffer.sample()
    assert sampled is not None
    assert len(sampled) == 5
    done_batch = sampled[-1].numpy()
    assert done_batch.shape[1] == 1
    assert np.all((done_batch == 0.0) | (done_batch == 1.0))


def test_keras_train_uses_done_mask_in_td_target():
    pytest.importorskip("tensorflow")
    from Keras import DDPG as keras_ddpg

    source = inspect.getsource(keras_ddpg.train)
    assert "(1.0 - done_batch)" in source


def test_pytorch_train_uses_terminated_for_backup(monkeypatch):
    from Pytorch import ddpg as pytorch_ddpg

    captured_terminal_flags: list[bool] = []

    class DummyAgent:
        def __init__(self, state_size, action_size, random_seed):
            del state_size, action_size, random_seed

        def reset(self):
            return None

        def act(self, state, add_noise=True):
            del state, add_noise
            return np.zeros(3, dtype=np.float32)

        def step(self, state, action, reward, next_state, terminal):
            del state, action, reward, next_state
            captured_terminal_flags.append(bool(terminal))

    class DummyEnv:
        obs_size = 13
        num_obstacles = 3
        obs_schema = "canonical_v3"
        kappa1 = 0.0
        kappa2 = 0.0
        kappa3 = 0.0
        target_k1 = 0.0
        target_k2 = 0.0
        target_k3 = 0.0
        overshoot0 = 0
        overshoot1 = 0

        def reset(self, seed=None):
            del seed
            return np.zeros(self.obs_size, dtype=np.float32), {}

        def step(self, action, reward_function="step_minus_weighted_euclidean"):
            del action, reward_function
            obs = np.zeros(self.obs_size, dtype=np.float32)
            reward = 0.0
            terminated = False
            truncated = True
            return obs, reward, terminated, truncated, {}

    monkeypatch.setattr(pytorch_ddpg, "Agent", DummyAgent)
    monkeypatch.setattr(pytorch_ddpg, "_make_env", lambda goal_type, max_episode_steps=None: DummyEnv())
    monkeypatch.setattr(pytorch_ddpg, "_save_checkpoints", lambda **kwargs: None)

    scores = pytorch_ddpg.train(
        n_episodes=1,
        max_t=1,
        print_every=1,
        goal_type="fixed_goal",
        reward_function="step_minus_weighted_euclidean",
        reward_file="reward_step_minus_weighted_euclidean",
        output_base_dir=Path("."),
        seed=1,
        deterministic=False,
    )
    assert len(scores) == 1
    assert captured_terminal_flags == [False]


def test_pytorch_ou_noise_increments_are_near_zero_mean():
    np.random.seed(7)
    noise = OUNoise(size=3, seed=7)
    deltas = []
    prev = noise.state.copy()
    for _ in range(1000):
        curr = noise.sample()
        deltas.append(curr - prev)
        prev = curr.copy()
    mean_delta = np.mean(np.asarray(deltas), axis=0)
    assert np.all(np.abs(mean_delta) < 0.1)


def test_pytorch_actor_critic_architecture_is_128x128():
    from Pytorch.model import Actor, Critic

    actor = Actor(state_size=13, action_size=3, seed=0)
    critic = Critic(state_size=13, action_size=3, seed=0)

    assert actor.fc1.out_features == 128
    assert actor.fc2.out_features == 128
    assert actor.fc3.out_features == 3

    assert critic.fc1.in_features == 16  # state + action
    assert critic.fc1.out_features == 128
    assert critic.fc2.out_features == 128
    assert critic.fc3.out_features == 1


def test_keras_actor_critic_architecture_is_128x128():
    pytest.importorskip("tensorflow")
    from Keras.DDPG import get_actor, get_critic

    actor = get_actor(num_states=13, num_actions=3, upper_bound=1.0)
    critic = get_critic(num_states=13, num_actions=3)

    actor_dense = [layer.units for layer in actor.layers if hasattr(layer, "units")]
    critic_dense = [layer.units for layer in critic.layers if hasattr(layer, "units")]

    assert actor_dense == [128, 128, 3]
    assert critic_dense == [128, 128, 1]
