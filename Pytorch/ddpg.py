"""PyTorch DDPG training/evaluation module."""

from __future__ import annotations

import argparse
import pickle
import random
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from continuum_rl.artifacts import (
    ARTIFACT_VERSION,
    ensure_dir,
    validate_metadata,
    write_metadata,
)
from continuum_rl.env import ContinuumEnv
from continuum_rl.gym_compat import unpack_step_output

try:
    from .ddpg_agent import Agent
except ImportError:  # script mode
    from ddpg_agent import Agent


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


def _resolve_output_base_dir(output_base_dir: Path | str | None) -> Path:
    if output_base_dir is None:
        return BASE_DIR
    return Path(output_base_dir)


def _expected_metadata(
    env: ContinuumEnv,
    goal_type: str,
    reward_file: str,
    reward_function: str,
) -> dict[str, Any]:
    return {
        "framework": "pytorch",
        "artifact_version": ARTIFACT_VERSION,
        "model_arch": MODEL_ARCH,
        "state_dim": env.obs_size,
        "obs_schema": env.obs_schema,
        "obstacle_count": env.num_obstacles,
        "goal_type": goal_type,
        "reward_function": reward_function,
        "reward_file": reward_file,
    }


def validate_checkpoint_compatibility(checkpoint_path: Path, expected: dict[str, Any]) -> None:
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
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _make_env(
    goal_type: str,
    max_episode_steps: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> ContinuumEnv:
    kwargs = dict(env_kwargs or {})
    return ContinuumEnv(
        observation_mode="canonical",
        goal_type=goal_type,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def _save_checkpoints(
    agent: Agent,
    env: ContinuumEnv,
    goal_type: str,
    reward_file: str,
    reward_function: str,
    output_base_dir: Path,
    scores: list[float] | None = None,
    avg_reward_list: list[float] | None = None,
) -> None:
    model_dir = ensure_dir(output_base_dir / goal_type / reward_file / "model")
    actor_path = model_dir / "checkpoint_actor.pth"
    critic_path = model_dir / "checkpoint_critic.pth"
    torch.save(agent.actor_local.state_dict(), actor_path)
    torch.save(agent.critic_local.state_dict(), critic_path)

    metadata = _expected_metadata(env, goal_type, reward_file, reward_function)
    write_metadata(actor_path, metadata)
    write_metadata(critic_path, metadata)

    rewards_dir = ensure_dir(output_base_dir / goal_type / reward_file / "rewards")
    if scores is not None:
        with (rewards_dir / "scores.pickle").open("wb") as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
    if avg_reward_list is not None:
        with (rewards_dir / "avg_reward_list.pickle").open("wb") as f:
            pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)


def train(
    n_episodes: int = 300,
    max_t: int = 750,
    print_every: int = 25,
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
    env = _make_env(goal_type=goal_type, max_episode_steps=max_t, env_kwargs=env_kwargs)
    agent = Agent(state_size=env.obs_size, action_size=3, random_seed=10)

    scores_deque = deque(maxlen=print_every)
    scores: list[float] = []
    avg_reward_list: list[float] = []
    success_counter = 0
    truncation_counter = 0

    for i_episode in range(1, n_episodes + 1):
        episode_seed = None if seed is None else seed + i_episode
        state, _ = env.reset(seed=episode_seed)
        agent.reset()
        score = 0.0

        if i_episode % print_every == 0:
            print("\n")
            print("Initial Position is", state[0:2])
            print("===============================================================")
            print("Target Position is", state[2:4])
            print("===============================================================")
            print("Initial Kappas are ", [env.kappa1, env.kappa2, env.kappa3])
            print("===============================================================")
            print("Goal Kappas are ", [env.target_k1, env.target_k2, env.target_k3])
            print("===============================================================")

        for _ in range(max_t):
            action = agent.act(state)
            step_out = unpack_step_output(env.step(action, reward_function=reward_function))
            next_state = step_out.obs
            episode_done = step_out.terminated or step_out.truncated
            terminal_for_backup = bool(step_out.terminated)
            agent.step(state, action, step_out.reward, next_state, terminal_for_backup)
            state = next_state
            score += step_out.reward
            if episode_done:
                if step_out.terminated:
                    success_counter += 1
                elif step_out.truncated:
                    truncation_counter += 1
                break

        scores_deque.append(score)
        scores.append(score)
        avg_reward_list.append(float(np.mean(scores[-100:])))
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}", end="")

    print("\n")
    print(f"{success_counter} episodes reached the target point in total {n_episodes} episodes")
    print(f"{truncation_counter} episodes ended by time-limit truncation")
    end_time = time.time() - start_time
    print("Total Overshoot 0: ", env.overshoot0)
    print("Total Overshoot 1: ", env.overshoot1)
    print(f"Total Elapsed Time is {int(end_time) / 60} minutes")

    _save_checkpoints(
        agent=agent,
        env=env,
        goal_type=goal_type,
        reward_file=reward_file,
        reward_function=reward_function,
        output_base_dir=_resolve_output_base_dir(output_base_dir),
        scores=scores,
        avg_reward_list=avg_reward_list,
    )
    return scores


def evaluate_smoke(
    checkpoint_actor: Path,
    checkpoint_critic: Path,
    goal_type: str = DEFAULT_GOAL_TYPE,
    reward_function: str = DEFAULT_REWARD_FUNCTION,
    max_t: int = 20,
    seed: int | None = None,
    deterministic: bool = False,
    env_kwargs: dict[str, Any] | None = None,
) -> float:
    _configure_runtime(seed=seed, deterministic=deterministic)
    env = _make_env(goal_type=goal_type, max_episode_steps=max_t, env_kwargs=env_kwargs)
    expected = _expected_metadata(env, goal_type, "manual", reward_function)
    validate_checkpoint_compatibility(checkpoint_actor, expected)
    validate_checkpoint_compatibility(checkpoint_critic, expected)

    agent = Agent(state_size=env.obs_size, action_size=3, random_seed=10)
    agent.actor_local.load_state_dict(torch.load(checkpoint_actor, map_location=torch.device("cpu")))
    agent.critic_local.load_state_dict(torch.load(checkpoint_critic, map_location=torch.device("cpu")))

    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    success_counter = 0
    truncation_counter = 0
    for _ in range(max_t):
        action = agent.act(state, add_noise=False)
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
        "Smoke eval finished, "
        f"total_reward={total_reward:.3f}, successes={success_counter}, truncations={truncation_counter}"
    )
    return total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DDPG runner for continuum RL.")
    parser.add_argument("--mode", choices=["train", "eval-smoke"], default="train")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-t", type=int, default=750)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default=DEFAULT_GOAL_TYPE)
    parser.add_argument("--reward-function", default=DEFAULT_REWARD_FUNCTION)
    parser.add_argument("--reward-file", default=DEFAULT_REWARD_FILE)
    parser.add_argument("--checkpoint-actor", type=Path, default=None)
    parser.add_argument("--checkpoint-critic", type=Path, default=None)
    parser.add_argument("--output-base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _warn_deprecated(mode: str) -> None:
    replacement = (
        "continuum-rl task=pytorch_train"
        if mode == "train"
        else "continuum-rl task=pytorch_eval_smoke"
    )
    message = (
        "DEPRECATION: `python -m Pytorch.ddpg` compatibility mode will be removed in the "
        "next release milestone. Use `" + replacement + " ...` instead."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    print(message)


def main() -> None:
    args = parse_args()
    _warn_deprecated(args.mode)

    if args.mode == "train":
        train(
            n_episodes=args.episodes,
            max_t=args.max_t,
            print_every=args.print_every,
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            reward_file=args.reward_file,
            output_base_dir=args.output_base_dir,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        return

    actor = args.checkpoint_actor or (
        args.output_base_dir / args.goal_type / args.reward_file / "model" / "checkpoint_actor.pth"
    )
    critic = args.checkpoint_critic or (
        args.output_base_dir / args.goal_type / args.reward_file / "model" / "checkpoint_critic.pth"
    )
    evaluate_smoke(
        checkpoint_actor=actor,
        checkpoint_critic=critic,
        goal_type=args.goal_type,
        reward_function=args.reward_function,
        max_t=min(args.max_t, 100),
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
