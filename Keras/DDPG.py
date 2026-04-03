"""Keras DDPG training/evaluation module."""

from __future__ import annotations

import argparse
import pickle
import random
import time
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from continuum_rl.artifacts import (
    ARTIFACT_VERSION,
    ensure_dir,
    read_metadata,
    validate_metadata,
    write_metadata,
)
from continuum_rl.env import ContinuumEnv
from continuum_rl.gym_compat import unpack_step_output


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GOAL_TYPE = "fixed_goal"
DEFAULT_REWARD_FUNCTION = "step_minus_weighted_euclidean"
DEFAULT_REWARD_FILE = "reward_step_minus_weighted_euclidean"
MODEL_ARCH = "ddpg_mlp_actor_128x128_critic_128x128_concat"

# Backward-compatible module-level configuration consumed by legacy demo scripts.
config: dict[str, Any] = {
    "goal_type": DEFAULT_GOAL_TYPE,
    "reward": {
        "function": DEFAULT_REWARD_FUNCTION,
        "file": DEFAULT_REWARD_FILE,
    },
}


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
        self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]
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
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices], dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


def get_actor(num_states: int, num_actions: int, upper_bound: float):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(128, activation="relu")(inputs)
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
    action_input = layers.Input(shape=(num_actions,))
    concat = layers.Concatenate()([state_input, action_input])
    out = layers.Dense(128, activation="relu")(concat)
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


def _resolve_output_base_dir(output_base_dir: Path | str | None) -> Path:
    if output_base_dir is None:
        return BASE_DIR
    return Path(output_base_dir)


def _expected_metadata(env: ContinuumEnv, goal_type: str, reward_file: str, reward_function: str) -> dict[str, Any]:
    return {
        "framework": "keras",
        "artifact_version": ARTIFACT_VERSION,
        "model_arch": MODEL_ARCH,
        "state_dim": env.obs_size,
        "obs_schema": env.obs_schema,
        "obstacle_count": env.num_obstacles,
        "goal_type": goal_type,
        "reward_function": reward_function,
        "reward_file": reward_file,
    }


def _resolve_weights_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix == ".h5":
        alt = path.with_name(path.stem + ".weights.h5")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def validate_checkpoint_compatibility(checkpoint_path: Path, expected: dict[str, Any]) -> None:
    checkpoint_path = _resolve_weights_path(checkpoint_path)
    try:
        validate_metadata(
            checkpoint_path,
            expected,
            strict_keys=("state_dim", "obs_schema", "model_arch", "obstacle_count", "goal_type", "reward_function"),
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc} Migration note: observation schema changed to canonical_v3 and "
            "the DDPG network architecture was standardized to 128x128; older checkpoints are intentionally incompatible."
        ) from exc


def _configure_runtime(seed: int | None, deterministic: bool) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def _save_weights(
    actor_model: tf.keras.Model,
    critic_model: tf.keras.Model,
    target_actor: tf.keras.Model,
    target_critic: tf.keras.Model,
    env: ContinuumEnv,
    goal_type: str,
    reward_file: str,
    reward_function: str,
    output_base_dir: Path,
    ep_reward_list: list[float] | None = None,
    avg_reward_list: list[float] | None = None,
) -> None:
    metadata = _expected_metadata(env, goal_type, reward_file, reward_function)

    model_dir = ensure_dir(output_base_dir / goal_type / reward_file / "model")
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

    rewards_dir = ensure_dir(output_base_dir / goal_type / reward_file / "rewards")
    if ep_reward_list is not None:
        with (rewards_dir / "ep_reward_list.pickle").open("wb") as f:
            pickle.dump(ep_reward_list, f, pickle.HIGHEST_PROTOCOL)
    if avg_reward_list is not None:
        with (rewards_dir / "avg_reward_list.pickle").open("wb") as f:
            pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)


def train(
    total_episodes: int = 500,
    max_steps: int = 500,
    goal_type: str = DEFAULT_GOAL_TYPE,
    reward_function: str = DEFAULT_REWARD_FUNCTION,
    reward_file: str = DEFAULT_REWARD_FILE,
    output_base_dir: Path | str | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    env_kwargs: dict[str, Any] | None = None,
) -> list[float]:
    _configure_runtime(seed=seed, deterministic=deterministic)
    start_time = time.time()
    kwargs = dict(env_kwargs or {})
    env = ContinuumEnv(
        observation_mode="canonical",
        goal_type=goal_type,
        max_episode_steps=max_steps,
        **kwargs,
    )
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
    grad_clip_norm = 1.0

    @tf.function
    def update_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * (1.0 - done_batch) * target_critic(
                [next_state_batch, target_actions],
                training=True,
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_pairs = [(g, v) for g, v in zip(critic_grad, critic_model.trainable_variables) if g is not None]
        if critic_pairs:
            critic_grads, critic_vars = zip(*critic_pairs)
            clipped_critic_grads, _ = tf.clip_by_global_norm(critic_grads, grad_clip_norm)
            critic_optimizer.apply_gradients(zip(clipped_critic_grads, critic_vars))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_pairs = [(g, v) for g, v in zip(actor_grad, actor_model.trainable_variables) if g is not None]
        if actor_pairs:
            actor_grads, actor_vars = zip(*actor_pairs)
            clipped_actor_grads, _ = tf.clip_by_global_norm(actor_grads, grad_clip_norm)
            actor_optimizer.apply_gradients(zip(clipped_actor_grads, actor_vars))

    buffer = ReplayBuffer(num_states=num_states, num_actions=num_actions, buffer_capacity=int(1e6), batch_size=256)
    ep_reward_list: list[float] = []
    avg_reward_list: list[float] = []
    success_counter = 0
    truncation_counter = 0
    noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(0.3) * np.ones(num_actions))

    for ep in range(total_episodes):
        episode_seed = None if seed is None else seed + ep
        prev_state, _ = env.reset(seed=episode_seed)
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
            episode_done = step_out.terminated or step_out.truncated
            terminal_for_backup = float(step_out.terminated)

            buffer.record((prev_state, action, reward, state, terminal_for_backup))
            batch = buffer.sample()
            if batch is not None:
                update_batch(*batch)
                soft_update(target_actor, actor_model, tau)
                soft_update(target_critic, critic_model, tau)

            episodic_reward += reward
            prev_state = state
            if episode_done:
                if step_out.terminated:
                    success_counter += 1
                elif step_out.truncated:
                    truncation_counter += 1
                break

        ep_reward_list.append(float(episodic_reward))
        avg_reward = float(np.mean(ep_reward_list[-250:]))
        avg_reward_list.append(avg_reward)
        print(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")

    print(f"{success_counter} episodes reached the target point in total {total_episodes} episodes")
    print(f"{truncation_counter} episodes ended by time-limit truncation")
    end_time = time.time() - start_time
    print("Total Overshoot 0: ", env.overshoot0)
    print("Total Overshoot 1: ", env.overshoot1)
    print(f"Total Elapsed Time is {int(end_time) / 60} minutes")

    _save_weights(
        actor_model=actor_model,
        critic_model=critic_model,
        target_actor=target_actor,
        target_critic=target_critic,
        env=env,
        goal_type=goal_type,
        reward_file=reward_file,
        reward_function=reward_function,
        output_base_dir=_resolve_output_base_dir(output_base_dir),
        ep_reward_list=ep_reward_list,
        avg_reward_list=avg_reward_list,
    )
    return ep_reward_list


def evaluate_smoke(
    checkpoint_actor: Path,
    goal_type: str = DEFAULT_GOAL_TYPE,
    reward_function: str = DEFAULT_REWARD_FUNCTION,
    max_steps: int = 20,
    seed: int | None = None,
    deterministic: bool = False,
    env_kwargs: dict[str, Any] | None = None,
) -> float:
    _configure_runtime(seed=seed, deterministic=deterministic)
    kwargs = dict(env_kwargs or {})
    env = ContinuumEnv(
        observation_mode="canonical",
        goal_type=goal_type,
        max_episode_steps=max_steps,
        **kwargs,
    )
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
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    success_counter = 0
    truncation_counter = 0
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
            if step_out.terminated:
                success_counter += 1
            elif step_out.truncated:
                truncation_counter += 1
            break

    print(
        "Keras smoke eval finished, "
        f"total_reward={total_reward:.3f}, successes={success_counter}, truncations={truncation_counter}"
    )
    return total_reward


# Backward-compatible globals for demo scripts.
num_states: int | None = None
num_actions: int | None = None
upper_bound: float | None = None
lower_bound: float | None = None
actor_model: tf.keras.Model | None = None


def _ensure_demo_policy_initialized() -> None:
    global num_states, num_actions, upper_bound, lower_bound, actor_model
    if actor_model is not None:
        return
    preview_env = ContinuumEnv(observation_mode="canonical", goal_type=config["goal_type"])
    num_states = preview_env.obs_size
    num_actions = preview_env.action_space.shape[0]
    upper_bound = float(preview_env.action_space.high[0])
    lower_bound = float(preview_env.action_space.low[0])
    actor_model = get_actor(num_states, num_actions, upper_bound)


def policy_for_demo(state, noise_object, add_noise=True):
    _ensure_demo_policy_initialized()
    assert actor_model is not None
    assert lower_bound is not None
    assert upper_bound is not None
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
    parser.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default=DEFAULT_GOAL_TYPE)
    parser.add_argument("--reward-function", default=DEFAULT_REWARD_FUNCTION)
    parser.add_argument("--reward-file", default=DEFAULT_REWARD_FILE)
    parser.add_argument("--checkpoint-actor", type=Path, default=None)
    parser.add_argument("--output-base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _warn_deprecated(mode: str) -> None:
    replacement = "continuum-rl task=keras_train" if mode == "train" else "continuum-rl task=keras_eval_smoke"
    message = (
        "DEPRECATION: `python -m Keras.DDPG` compatibility mode will be removed in the "
        "next release milestone. Use `" + replacement + " ...` instead."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    print(message)


def main() -> None:
    args = parse_args()
    _warn_deprecated(args.mode)

    if args.mode == "train":
        train(
            total_episodes=args.episodes,
            max_steps=args.max_steps,
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            reward_file=args.reward_file,
            output_base_dir=args.output_base_dir,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        return

    actor = args.checkpoint_actor or (
        args.output_base_dir / args.goal_type / args.reward_file / "model" / "continuum_actor.h5"
    )
    evaluate_smoke(
        checkpoint_actor=actor,
        goal_type=args.goal_type,
        reward_function=args.reward_function,
        max_steps=min(args.max_steps, 100),
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
