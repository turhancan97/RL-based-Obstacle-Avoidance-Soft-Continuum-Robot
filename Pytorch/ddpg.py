"""PyTorch DDPG training/evaluation entrypoint (import-safe)."""

from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from continuum_rl.artifacts import ARTIFACT_VERSION, ensure_dir, read_metadata, write_metadata
from continuum_rl.env import ContinuumEnv
from continuum_rl.gym_compat import unpack_step_output

try:
    from .ddpg_agent import Agent
except ImportError:  # script mode
    from ddpg_agent import Agent


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()


def _expected_metadata(env: ContinuumEnv, goal_type: str, reward_file: str, reward_function: str) -> dict[str, Any]:
    return {
        "framework": "pytorch",
        "artifact_version": ARTIFACT_VERSION,
        "state_dim": env.obs_size,
        "obstacle_count": env.num_obstacles,
        "goal_type": goal_type,
        "reward_function": reward_function,
        "reward_file": reward_file,
    }


def _infer_state_dim_from_checkpoint(checkpoint_path: Path) -> int:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    weight = state_dict.get("fc1.weight")
    if weight is None:
        weight = state_dict.get("fcs1.weight")
    if weight is None:
        raise ValueError(f"Unable to infer state_dim from checkpoint '{checkpoint_path}'.")
    return int(weight.shape[1])


def validate_checkpoint_compatibility(checkpoint_path: Path, expected: dict[str, Any]) -> None:
    metadata = read_metadata(checkpoint_path)
    if metadata:
        expected_state_dim = expected["state_dim"]
        actual_state_dim = metadata.get("state_dim")
        if expected_state_dim != actual_state_dim:
            raise ValueError(
                f"Checkpoint incompatible: expected state_dim={expected_state_dim}, "
                f"actual={actual_state_dim}. Use --observation-mode legacy4d or a matching checkpoint."
            )
        return

    # Legacy checkpoint without metadata.
    actual_state_dim = _infer_state_dim_from_checkpoint(checkpoint_path)
    expected_state_dim = expected["state_dim"]
    if actual_state_dim != expected_state_dim:
        raise ValueError(
            f"Legacy checkpoint incompatible: expected state_dim={expected_state_dim}, actual={actual_state_dim}. "
            "Use --observation-mode legacy4d or regenerate v2 checkpoints."
        )


def _make_env(observation_mode: str, goal_type: str) -> ContinuumEnv:
    return ContinuumEnv(observation_mode=observation_mode, goal_type=goal_type)


def _save_checkpoints(
    agent: Agent,
    env: ContinuumEnv,
    goal_type: str,
    reward_file: str,
    reward_function: str,
    scores: list[float] | None = None,
    avg_reward_list: list[float] | None = None,
) -> None:
    # New canonical v2 output path.
    v2_model_dir = ensure_dir(BASE_DIR / "v2" / goal_type / reward_file / "model")
    actor_v2 = v2_model_dir / "checkpoint_actor.pth"
    critic_v2 = v2_model_dir / "checkpoint_critic.pth"
    torch.save(agent.actor_local.state_dict(), actor_v2)
    torch.save(agent.critic_local.state_dict(), critic_v2)

    metadata = _expected_metadata(env, goal_type, reward_file, reward_function)
    write_metadata(actor_v2, metadata)
    write_metadata(critic_v2, metadata)

    v2_rewards_dir = ensure_dir(BASE_DIR / "v2" / goal_type / reward_file / "rewards")
    if scores is not None:
        with (v2_rewards_dir / "scores.pickle").open("wb") as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
    if avg_reward_list is not None:
        with (v2_rewards_dir / "avg_reward_list.pickle").open("wb") as f:
            pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)

    # Legacy compatibility output path.
    legacy_model_dir = ensure_dir(BASE_DIR / "experiment")
    legacy_actor = legacy_model_dir / "checkpoint_actor.pth"
    legacy_critic = legacy_model_dir / "checkpoint_critic.pth"
    torch.save(agent.actor_local.state_dict(), legacy_actor)
    torch.save(agent.critic_local.state_dict(), legacy_critic)
    write_metadata(legacy_actor, metadata)
    write_metadata(legacy_critic, metadata)

    if scores is not None:
        with (legacy_model_dir / "scores.pickle").open("wb") as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
    if avg_reward_list is not None:
        with (legacy_model_dir / "avg_reward_list.pickle").open("wb") as f:
            pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)


def train(
    n_episodes: int = 300,
    max_t: int = 750,
    print_every: int = 25,
    observation_mode: str = "canonical",
    goal_type: str = "fixed_goal",
    reward_function: str = "step_minus_weighted_euclidean",
    reward_file: str = "reward_step_minus_weighted_euclidean",
) -> list[float]:
    start_time = time.time()
    env = _make_env(observation_mode=observation_mode, goal_type=goal_type)
    agent = Agent(state_size=env.obs_size, action_size=3, random_seed=10)

    scores_deque = deque(maxlen=print_every)
    scores: list[float] = []
    avg_reward_list: list[float] = []
    counter = 0

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
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
            next_state, reward, done = step_out.obs, step_out.reward, step_out.terminated or step_out.truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                counter += 1
                break

        scores_deque.append(score)
        scores.append(score)
        avg_reward_list.append(float(np.mean(scores[-100:])))
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}", end="")

    print("\n")
    print(f"{counter} times robot reached the target point in total {n_episodes} episodes")
    end_time = time.time() - start_time
    print("Total Overshoot 0: ", env.overshoot0)
    print("Total Overshoot 1: ", env.overshoot1)
    print(f"Total Elapsed Time is {int(end_time)/60} minutes")

    _save_checkpoints(
        agent=agent,
        env=env,
        goal_type=goal_type,
        reward_file=reward_file,
        reward_function=reward_function,
        scores=scores,
        avg_reward_list=avg_reward_list,
    )
    return scores


def evaluate_smoke(
    checkpoint_actor: Path,
    checkpoint_critic: Path,
    observation_mode: str = "canonical",
    goal_type: str = "fixed_goal",
    reward_function: str = "step_minus_weighted_euclidean",
    max_t: int = 20,
) -> float:
    env = _make_env(observation_mode=observation_mode, goal_type=goal_type)
    expected = _expected_metadata(env, goal_type, "manual", reward_function)
    validate_checkpoint_compatibility(checkpoint_actor, expected)
    validate_checkpoint_compatibility(checkpoint_critic, expected)

    agent = Agent(state_size=env.obs_size, action_size=3, random_seed=10)
    agent.actor_local.load_state_dict(torch.load(checkpoint_actor, map_location=torch.device("cpu")))
    agent.critic_local.load_state_dict(torch.load(checkpoint_critic, map_location=torch.device("cpu")))

    state, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_t):
        action = agent.act(state, add_noise=False)
        step_out = unpack_step_output(env.step(action, reward_function=reward_function))
        state = step_out.obs
        total_reward += step_out.reward
        if step_out.terminated or step_out.truncated:
            break
    print(f"Smoke eval finished, total_reward={total_reward:.3f}")
    return total_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DDPG runner for continuum RL.")
    parser.add_argument("--mode", choices=["train", "eval-smoke"], default="train")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-t", type=int, default=750)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--observation-mode", choices=["canonical", "legacy4d"], default="canonical")
    parser.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default=config.get("goal_type", "fixed_goal"))
    parser.add_argument("--reward-function", default=config.get("reward", {}).get("function", "step_minus_weighted_euclidean"))
    parser.add_argument("--reward-file", default=config.get("reward", {}).get("file", "reward_step_minus_weighted_euclidean"))
    parser.add_argument("--checkpoint-actor", type=Path, default=None)
    parser.add_argument("--checkpoint-critic", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(
            n_episodes=args.episodes,
            max_t=args.max_t,
            print_every=args.print_every,
            observation_mode=args.observation_mode,
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            reward_file=args.reward_file,
        )
        return

    actor = args.checkpoint_actor or (BASE_DIR / args.goal_type / args.reward_file / "model" / "checkpoint_actor.pth")
    critic = args.checkpoint_critic or (BASE_DIR / args.goal_type / args.reward_file / "model" / "checkpoint_critic.pth")
    evaluate_smoke(
        checkpoint_actor=actor,
        checkpoint_critic=critic,
        observation_mode=args.observation_mode,
        goal_type=args.goal_type,
        reward_function=args.reward_function,
        max_t=min(args.max_t, 100),
    )


if __name__ == "__main__":
    main()
