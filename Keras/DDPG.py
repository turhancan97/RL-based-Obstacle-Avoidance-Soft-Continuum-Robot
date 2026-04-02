"""Keras DDPG training/evaluation entrypoint (import-safe)."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import layers

from continuum_rl.artifacts import ARTIFACT_VERSION, ensure_dir, read_metadata, write_metadata
from continuum_rl.env import ContinuumEnv
from continuum_rl.gym_compat import unpack_step_output


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()


class OUActionNoise:
    """Ornstein-Uhlenbeck noise for exploration."""

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, num_states: int, num_actions: int, buffer_capacity: int = 100_000, batch_size: int = 64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states), dtype=np.float32)

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        if record_range < self.batch_size:
            return None
        batch_indices = np.random.choice(record_range, self.batch_size, replace=False)
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch


def get_actor(num_states: int, num_actions: int, upper_bound: float):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(128, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)
    outputs = outputs * upper_bound
    return tf.keras.Model(inputs, outputs)


def _build_actor_from_checkpoint_shapes(checkpoint_path: Path, upper_bound: float) -> tf.keras.Model:
    kernels: list[tuple[str, tuple[int, int]]] = []
    with h5py.File(checkpoint_path, "r") as f:
        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("kernel:0") and len(obj.shape) == 2:
                kernels.append((name, (int(obj.shape[0]), int(obj.shape[1]))))
        f.visititems(_collect)

    if not kernels:
        raise ValueError(f"No dense kernel tensors found in checkpoint '{checkpoint_path}'.")

    def _kernel_sort_key(item: tuple[str, tuple[int, int]]):
        name = item[0].split("/")[0]
        if "_" in name and name.split("_")[-1].isdigit():
            return int(name.split("_")[-1])
        if name == "dense":
            return 0
        return 10_000

    kernels.sort(key=_kernel_sort_key)
    shapes = [shape for _, shape in kernels]
    input_dim = shapes[0][0]

    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for idx, (_, out_dim) in enumerate(shapes):
        activation = "tanh" if idx == (len(shapes) - 1) else "relu"
        x = layers.Dense(out_dim, activation=activation)(x)
    outputs = x * upper_bound
    return tf.keras.Model(inputs, outputs)


def get_critic(num_states: int, num_actions: int):
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(256, activation="relu")(state_input)
    state_out = layers.Dense(256, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(256, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    return tf.keras.Model([state_input, action_input], outputs)


def soft_update(target_model: tf.keras.Model, source_model: tf.keras.Model, tau: float) -> None:
    """Keras 2/3-compatible target update (no SymbolicTensor.assign path)."""
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()
    new_weights = [source * tau + target * (1.0 - tau) for source, target in zip(source_weights, target_weights)]
    target_model.set_weights(new_weights)


def _policy_impl(state, noise_object, actor, lower_bound, upper_bound, add_noise=True):
    sampled_actions = tf.squeeze(actor(state))
    if add_noise:
        sampled_actions = sampled_actions.numpy() + noise_object()
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return np.asarray(legal_action, dtype=np.float32)


def _expected_metadata(env: ContinuumEnv, goal_type: str, reward_file: str, reward_function: str) -> dict[str, Any]:
    return {
        "framework": "keras",
        "artifact_version": ARTIFACT_VERSION,
        "state_dim": env.obs_size,
        "obstacle_count": env.num_obstacles,
        "goal_type": goal_type,
        "reward_function": reward_function,
        "reward_file": reward_file,
    }


def validate_checkpoint_compatibility(checkpoint_path: Path, expected: dict[str, Any]) -> None:
    checkpoint_path = _resolve_weights_path(checkpoint_path)
    metadata = read_metadata(checkpoint_path)
    if metadata:
        actual_state_dim = metadata.get("state_dim")
    else:
        with h5py.File(checkpoint_path, "r") as f:
            actual_state_dim = None
            for layer in f.keys():
                group = f[layer]
                if isinstance(group, h5py.Group) and "vars" in group and "0" in group["vars"]:
                    arr = group["vars"]["0"]
                    if len(arr.shape) == 2:
                        actual_state_dim = int(arr.shape[0])
                        break
            if actual_state_dim is None:
                for name, dataset in f.items():
                    if isinstance(dataset, h5py.Dataset) and len(dataset.shape) == 2:
                        actual_state_dim = int(dataset.shape[0])
                        break
            if actual_state_dim is None:
                def _finder(_name, obj):
                    nonlocal actual_state_dim
                    if actual_state_dim is not None:
                        return
                    if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2:
                        actual_state_dim = int(obj.shape[0])
                f.visititems(_finder)
        if actual_state_dim is None:
            raise ValueError(f"Unable to infer state_dim for checkpoint '{checkpoint_path}'.")

    if actual_state_dim != expected["state_dim"]:
        raise ValueError(
            f"Checkpoint incompatible: expected state_dim={expected['state_dim']}, actual={actual_state_dim}. "
            "Regenerate checkpoint in canonical mode."
        )


def _save_weights(
    actor_model: tf.keras.Model,
    critic_model: tf.keras.Model,
    target_actor: tf.keras.Model,
    target_critic: tf.keras.Model,
    env: ContinuumEnv,
    goal_type: str,
    reward_file: str,
    reward_function: str,
    ep_reward_list: list[float] | None = None,
    avg_reward_list: list[float] | None = None,
) -> None:
    metadata = _expected_metadata(env, goal_type, reward_file, reward_function)

    model_dir = ensure_dir(BASE_DIR / goal_type / reward_file / "model")
    actor_path = model_dir / "continuum_actor.weights.h5"
    critic_path = model_dir / "continuum_critic.weights.h5"
    target_actor_path = model_dir / "continuum_target_actor.weights.h5"
    target_critic_path = model_dir / "continuum_target_critic.weights.h5"

    actor_model.save_weights(actor_path)
    critic_model.save_weights(critic_path)
    target_actor.save_weights(target_actor_path)
    target_critic.save_weights(target_critic_path)
    for p in (actor_path, critic_path, target_actor_path, target_critic_path):
        write_metadata(p, metadata)

    rewards_dir = ensure_dir(BASE_DIR / goal_type / reward_file / "rewards")
    if ep_reward_list is not None:
        with (rewards_dir / "ep_reward_list.pickle").open("wb") as f:
            pickle.dump(ep_reward_list, f, pickle.HIGHEST_PROTOCOL)
    if avg_reward_list is not None:
        with (rewards_dir / "avg_reward_list.pickle").open("wb") as f:
            pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)


def train(
    total_episodes: int = 500,
    max_steps: int = 500,
    goal_type: str = "fixed_goal",
    reward_function: str = "step_minus_weighted_euclidean",
    reward_file: str = "reward_step_minus_weighted_euclidean",
) -> list[float]:
    start_time = time.time()
    env = ContinuumEnv(observation_mode="canonical", goal_type=goal_type)
    num_states = env.obs_size
    num_actions = env.action_space.shape[0]
    upper_bound = float(env.action_space.high[0])
    lower_bound = float(env.action_space.low[0])

    actor_model = get_actor(num_states, num_actions, upper_bound)
    critic_model = get_critic(num_states, num_actions)
    target_actor = get_actor(num_states, num_actions, upper_bound)
    target_critic = get_critic(num_states, num_actions)
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    critic_optimizer = tf.keras.optimizers.Adam(1e-3)
    actor_optimizer = tf.keras.optimizers.Adam(1e-4)
    gamma = 0.99
    tau = 1e-3

    @tf.function
    def update_batch(state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    buffer = ReplayBuffer(num_states=num_states, num_actions=num_actions, buffer_capacity=int(1e6), batch_size=256)
    ep_reward_list: list[float] = []
    avg_reward_list: list[float] = []
    counter = 0
    noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(0.3) * np.ones(num_actions))

    for ep in range(total_episodes):
        prev_state, _ = env.reset()
        episodic_reward = 0.0

        if ep % 50 == 0:
            print("Episode Number", ep)
            print("Initial Position is", prev_state[0:2])
            print("Target Position is", prev_state[2:4])

        for _ in range(max_steps):
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = _policy_impl(
                tf_prev_state,
                noise,
                actor_model,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                add_noise=True,
            )
            step_out = unpack_step_output(env.step(action, reward_function=reward_function))
            state = step_out.obs
            reward = step_out.reward
            done = step_out.terminated or step_out.truncated

            buffer.record((prev_state, action, reward, state))
            batch = buffer.sample()
            if batch is not None:
                update_batch(*batch)
                soft_update(target_actor, actor_model, tau)
                soft_update(target_critic, critic_model, tau)

            episodic_reward += reward
            prev_state = state
            if done:
                counter += 1
                break

        ep_reward_list.append(float(episodic_reward))
        avg_reward = float(np.mean(ep_reward_list[-250:]))
        avg_reward_list.append(avg_reward)
        print(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")

    print(f"{counter} times robot reached the target point in total {total_episodes} episodes")
    end_time = time.time() - start_time
    print("Total Overshoot 0: ", env.overshoot0)
    print("Total Overshoot 1: ", env.overshoot1)
    print(f"Total Elapsed Time is {int(end_time)/60} minutes")

    _save_weights(
        actor_model=actor_model,
        critic_model=critic_model,
        target_actor=target_actor,
        target_critic=target_critic,
        env=env,
        goal_type=goal_type,
        reward_file=reward_file,
        reward_function=reward_function,
        ep_reward_list=ep_reward_list,
        avg_reward_list=avg_reward_list,
    )
    return ep_reward_list


def evaluate_smoke(
    checkpoint_actor: Path,
    goal_type: str = "fixed_goal",
    reward_function: str = "step_minus_weighted_euclidean",
    max_steps: int = 20,
) -> float:
    env = ContinuumEnv(observation_mode="canonical", goal_type=goal_type)
    num_states = env.obs_size
    num_actions = env.action_space.shape[0]
    upper_bound = float(env.action_space.high[0])
    lower_bound = float(env.action_space.low[0])

    expected = _expected_metadata(env, goal_type, "manual", reward_function)
    validate_checkpoint_compatibility(checkpoint_actor, expected)
    resolved_actor = _resolve_weights_path(checkpoint_actor)
    metadata = read_metadata(resolved_actor)
    actor_model = (
        get_actor(num_states, num_actions, upper_bound)
        if metadata
        else _build_actor_from_checkpoint_shapes(resolved_actor, upper_bound)
    )
    actor_model.load_weights(resolved_actor)

    noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(0.3) * np.ones(num_actions))
    state, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = _policy_impl(
            tf_prev_state,
            noise,
            actor_model,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            add_noise=False,
        )
        step_out = unpack_step_output(env.step(action, reward_function=reward_function))
        state = step_out.obs
        total_reward += step_out.reward
        if step_out.terminated or step_out.truncated:
            break
    print(f"Keras smoke eval finished, total_reward={total_reward:.3f}")
    return total_reward


# Backward-compatible globals for demo scripts.
_preview_env = ContinuumEnv(observation_mode="canonical", goal_type=config.get("goal_type", "fixed_goal"))
num_states = _preview_env.obs_size
num_actions = _preview_env.action_space.shape[0]
upper_bound = float(_preview_env.action_space.high[0])
lower_bound = float(_preview_env.action_space.low[0])
actor_model = get_actor(num_states, num_actions, upper_bound)


def policy_for_demo(state, noise_object, add_noise=True):
    action = _policy_impl(
        state=state,
        noise_object=noise_object,
        actor=actor_model,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        add_noise=add_noise,
    )
    return [np.squeeze(action)]


# Preserve historical symbol name used by existing scripts.
policy = policy_for_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keras DDPG runner for continuum RL.")
    parser.add_argument("--mode", choices=["train", "eval-smoke"], default="train")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default=config.get("goal_type", "fixed_goal"))
    parser.add_argument("--reward-function", default=config.get("reward", {}).get("function", "step_minus_weighted_euclidean"))
    parser.add_argument("--reward-file", default=config.get("reward", {}).get("file", "reward_step_minus_weighted_euclidean"))
    parser.add_argument("--checkpoint-actor", type=Path, default=None)
    return parser.parse_args()


def _resolve_weights_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix == ".h5":
        alt = path.with_name(path.stem + ".weights.h5")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(
            total_episodes=args.episodes,
            max_steps=args.max_steps,
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            reward_file=args.reward_file,
        )
        return

    actor = args.checkpoint_actor or (
        BASE_DIR / args.goal_type / args.reward_file / "model" / "continuum_actor.h5"
    )
    evaluate_smoke(
        checkpoint_actor=actor,
        goal_type=args.goal_type,
        reward_function=args.reward_function,
        max_steps=min(args.max_steps, 100),
    )


if __name__ == "__main__":
    main()
