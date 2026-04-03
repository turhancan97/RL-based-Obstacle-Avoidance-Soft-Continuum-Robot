"""PyTorch DDPG training/evaluation module."""

from __future__ import annotations

import argparse
import pickle
import random
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from continuum_rl.artifacts import (
    ARTIFACT_VERSION,
    ensure_dir,
    metadata_path_for,
    validate_metadata,
    write_metadata,
)
from continuum_rl.env import ContinuumEnv
from continuum_rl.health import LossHealthMonitor
from continuum_rl.gym_compat import unpack_step_output
from continuum_rl.tracking import create_wandb_tracker

try:
    from .ddpg_agent import Agent
except ImportError:  # script mode
    from ddpg_agent import Agent


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_GOAL_TYPE = "fixed_goal"
DEFAULT_REWARD_FUNCTION = "step_minus_weighted_euclidean"
DEFAULT_REWARD_FILE = "reward_step_minus_weighted_euclidean"
DEFAULT_OUTPUT_BASE_DIR = REPO_ROOT / "runs" / "pytorch"
MODEL_ARCH = "ddpg_mlp_actor_128x128_critic_128x128_concat"

# Backward-compatible module-level configuration consumed by legacy demo scripts.
config: dict[str, Any] = {
    "goal_type": DEFAULT_GOAL_TYPE,
    "reward": {
        "function": DEFAULT_REWARD_FUNCTION,
        "file": DEFAULT_REWARD_FILE,
    },
}


def _resolve_repo_relative_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    preferred = (REPO_ROOT / p).resolve()
    if preferred.exists():
        return preferred
    # Backward-compatible fallback for accidental nested output dirs (e.g., Pytorch/Pytorch/...).
    fallback = (BASE_DIR / p).resolve()
    if fallback.exists():
        return fallback
    return preferred


def _resolve_output_base_dir(output_base_dir: Path | str | None) -> Path:
    if output_base_dir is None:
        return DEFAULT_OUTPUT_BASE_DIR
    return _resolve_repo_relative_path(output_base_dir)


def _resolve_reward_file(reward_function: str, reward_file: Optional[str]) -> str:
    if reward_file is not None and reward_file.strip() and reward_file.lower() != "auto":
        return reward_file
    return f"reward_{reward_function}"


def _seed_dir_name(seed: int | None) -> str:
    # Keep strict numeric seed directory names for paper-figure discovery.
    seed_id = 0 if seed is None else int(seed)
    return f"seed_{seed_id}"


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
    seed_dir: str,
    scores: list[float] | None = None,
    avg_reward_list: list[float] | None = None,
) -> dict[str, Path]:
    run_dir = output_base_dir / goal_type / reward_file / seed_dir
    model_dir = ensure_dir(run_dir / "model")
    actor_path = model_dir / "checkpoint_actor.pth"
    critic_path = model_dir / "checkpoint_critic.pth"
    torch.save(agent.actor_local.state_dict(), actor_path)
    torch.save(agent.critic_local.state_dict(), critic_path)

    metadata = _expected_metadata(env, goal_type, reward_file, reward_function)
    write_metadata(actor_path, metadata)
    write_metadata(critic_path, metadata)

    rewards_dir = ensure_dir(run_dir / "rewards")
    if scores is not None:
        with (rewards_dir / "scores.pickle").open("wb") as f:
            pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)
    if avg_reward_list is not None:
        with (rewards_dir / "avg_reward_list.pickle").open("wb") as f:
            pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)

    return {
        "actor": actor_path,
        "critic": critic_path,
        "actor_metadata": metadata_path_for(actor_path),
        "critic_metadata": metadata_path_for(critic_path),
    }


def _run_eval_rollout(
    agent: Agent,
    goal_type: str,
    reward_function: str,
    max_t: int,
    env_kwargs: dict[str, Any] | None,
    seed: int | None,
) -> dict[str, float]:
    if env_kwargs is None:
        eval_env = _make_env(goal_type=goal_type, max_episode_steps=max_t)
    else:
        eval_env = _make_env(goal_type=goal_type, max_episode_steps=max_t, env_kwargs=env_kwargs)
    state, _ = eval_env.reset(seed=seed)
    total_reward = 0.0
    episode_length = 0
    success = 0
    truncated = 0
    for t in range(max_t):
        action = agent.act(state, add_noise=False)
        step_out = unpack_step_output(eval_env.step(action, reward_function=reward_function))
        state = step_out.obs
        total_reward += step_out.reward
        episode_length = t + 1
        if step_out.terminated or step_out.truncated:
            if step_out.terminated:
                success = 1
            elif step_out.truncated:
                truncated = 1
            break

    return {
        "eval/mean_return": float(total_reward),
        "eval/success_rate": float(success),
        "eval/truncation_rate": float(truncated),
        "eval/mean_episode_length": float(episode_length),
    }


def _env_runtime_metadata(env: Any) -> dict[str, Any]:
    return {
        "obs_schema": getattr(env, "obs_schema", None),
        "state_dim": getattr(env, "obs_size", None),
        "env_dt": getattr(env, "dt", None),
        "env_delta_kappa": getattr(env, "delta_kappa", None),
        "env_l": list(getattr(env, "l", [])),
        "env_obstacles": list(getattr(env, "obstacles", [])),
    }


def train(
    n_episodes: int = 300,
    max_t: int = 750,
    print_every: int = 25,
    goal_type: str = DEFAULT_GOAL_TYPE,
    reward_function: str = DEFAULT_REWARD_FUNCTION,
    reward_file: Optional[str] = None,
    output_base_dir: Path | str | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    env_kwargs: dict[str, Any] | None = None,
    wandb_cfg: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    task_name: str = "pytorch_train",
    agent_seed: int = 10,
    buffer_size: int = 50_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    tau: float = 5e-4,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 0.5,
    noise_theta: float = 0.15,
    noise_sigma: float = 0.1,
) -> list[float]:
    _configure_runtime(seed=seed, deterministic=deterministic)
    start_time = time.time()
    resolved_reward_file = _resolve_reward_file(reward_function=reward_function, reward_file=reward_file)
    seed_dir = _seed_dir_name(seed)
    if env_kwargs is None:
        env = _make_env(goal_type=goal_type, max_episode_steps=max_t)
    else:
        env = _make_env(goal_type=goal_type, max_episode_steps=max_t, env_kwargs=env_kwargs)
    output_dir = _resolve_output_base_dir(output_base_dir)
    agent = Agent(
        state_size=env.obs_size,
        action_size=3,
        random_seed=agent_seed,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        lr_actor=actor_lr,
        lr_critic=critic_lr,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        noise_theta=noise_theta,
        noise_sigma=noise_sigma,
    )
    wandb_settings = dict(wandb_cfg or {})
    tracker_run_config = dict(run_config or {})
    tracker_run_config.setdefault("runtime_metadata", {})
    tracker_run_config["runtime_metadata"].update(
        {
            "framework": "pytorch",
            "model_arch": MODEL_ARCH,
            "goal_type": goal_type,
            "reward_function": reward_function,
            "reward_file": reward_file,
            "reward_file_resolved": resolved_reward_file,
            "agent_seed": agent_seed,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "noise_theta": noise_theta,
            "noise_sigma": noise_sigma,
            "health_alerts_enabled": bool(wandb_settings.get("health_alerts_enabled", True)),
            "health_ema_alpha": float(wandb_settings.get("health_ema_alpha", 0.01)),
            "health_warmup_steps": int(wandb_settings.get("health_warmup_steps", 2000)),
            "health_growth_factor": float(wandb_settings.get("health_growth_factor", 3.0)),
            "health_actor_loss_min_abs": float(wandb_settings.get("health_actor_loss_min_abs", 10.0)),
            "health_critic_loss_min_abs": float(wandb_settings.get("health_critic_loss_min_abs", 2.0)),
            "health_grad_norm_max": float(wandb_settings.get("health_grad_norm_max", 20.0)),
            **_env_runtime_metadata(env),
        }
    )
    tracker = create_wandb_tracker(
        wandb_cfg=wandb_cfg,
        run_config=tracker_run_config,
        context={
            "framework": "pytorch",
            "task_name": task_name,
            "goal_type": goal_type,
            "reward_id": resolved_reward_file,
            "seed": seed,
        },
    )
    eval_interval = max(1, int(wandb_settings.get("eval_interval_episodes", 50)))
    artifact_interval = max(1, int(wandb_settings.get("artifact_interval_episodes", 100)))
    upload_checkpoints = bool(wandb_settings.get("upload_checkpoints", True))
    health_monitor = LossHealthMonitor(
        enabled=bool(wandb_settings.get("health_alerts_enabled", True)),
        ema_alpha=float(wandb_settings.get("health_ema_alpha", 0.01)),
        warmup_steps=int(wandb_settings.get("health_warmup_steps", 2000)),
        growth_factor=float(wandb_settings.get("health_growth_factor", 3.0)),
        actor_loss_min_abs=float(wandb_settings.get("health_actor_loss_min_abs", 10.0)),
        critic_loss_min_abs=float(wandb_settings.get("health_critic_loss_min_abs", 2.0)),
        grad_norm_max=float(wandb_settings.get("health_grad_norm_max", 20.0)),
    )

    scores_deque = deque(maxlen=print_every)
    scores: list[float] = []
    avg_reward_list: list[float] = []
    success_counter = 0
    truncation_counter = 0
    global_step = 0

    try:
        for i_episode in range(1, n_episodes + 1):
            episode_seed = None if seed is None else seed + i_episode
            state, _ = env.reset(seed=episode_seed)
            agent.reset()
            score = 0.0
            episode_steps = 0
            episode_success = 0
            episode_truncated = 0

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

            for t in range(max_t):
                action = agent.act(state)
                step_out = unpack_step_output(env.step(action, reward_function=reward_function))
                global_step += 1
                next_state = step_out.obs
                episode_done = step_out.terminated or step_out.truncated
                terminal_for_backup = bool(step_out.terminated)
                learn_metrics = agent.step(state, action, step_out.reward, next_state, terminal_for_backup)
                state = next_state
                score += step_out.reward
                episode_steps = t + 1

                if tracker.active and learn_metrics is not None:
                    health_metrics, health_messages = health_monitor.update(
                        step=global_step,
                        actor_loss=float(learn_metrics["actor_loss"]),
                        critic_loss=float(learn_metrics["critic_loss"]),
                        actor_grad_norm=float(learn_metrics["actor_grad_norm"]),
                        critic_grad_norm=float(learn_metrics["critic_grad_norm"]),
                    )
                    for msg in health_messages:
                        print(f"WARNING: {msg}")
                    tracker.log_metrics(
                        {
                            "opt/actor_loss": learn_metrics["actor_loss"],
                            "opt/critic_loss": learn_metrics["critic_loss"],
                            "opt/actor_grad_norm": learn_metrics["actor_grad_norm"],
                            "opt/critic_grad_norm": learn_metrics["critic_grad_norm"],
                            "replay/size": learn_metrics["replay_size"],
                            **health_metrics,
                        },
                        step=global_step,
                    )

                if episode_done:
                    if step_out.terminated:
                        success_counter += 1
                        episode_success = 1
                    elif step_out.truncated:
                        truncation_counter += 1
                        episode_truncated = 1
                    break

            scores_deque.append(score)
            scores.append(score)
            avg_window = float(np.mean(scores[-100:]))
            avg_reward_list.append(avg_window)
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}", end="")

            if tracker.active:
                tracker.log_metrics(
                    {
                        "train/episode_reward": float(score),
                        "train/episode_length": float(episode_steps),
                        "train/success": float(episode_success),
                        "train/truncated": float(episode_truncated),
                        "train/episode_index": float(i_episode),
                        "train/avg_reward_window": avg_window,
                        "train/error_final": float(env.error),
                        "train/overshoot0_total": float(env.overshoot0),
                        "train/overshoot1_total": float(env.overshoot1),
                    },
                    step=global_step,
                )

                if i_episode % eval_interval == 0:
                    eval_seed = (seed if seed is not None else 12345) + i_episode
                    eval_metrics = _run_eval_rollout(
                        agent=agent,
                        goal_type=goal_type,
                        reward_function=reward_function,
                        max_t=max_t,
                        env_kwargs=env_kwargs,
                        seed=eval_seed,
                    )
                    eval_metrics["eval/episode_index"] = float(i_episode)
                    tracker.log_metrics(eval_metrics, step=global_step)

                if upload_checkpoints and i_episode % artifact_interval == 0:
                    saved_paths = _save_checkpoints(
                        agent=agent,
                        env=env,
                        goal_type=goal_type,
                        reward_file=resolved_reward_file,
                        reward_function=reward_function,
                        output_base_dir=output_dir,
                        seed_dir=seed_dir,
                    )
                    tracker.log_artifact_files(
                        name=f"pytorch-{goal_type}-{resolved_reward_file}",
                        artifact_type="model",
                        paths=list(saved_paths.values()),
                        metadata=_expected_metadata(env, goal_type, resolved_reward_file, reward_function),
                        aliases=["latest", f"episode-{i_episode}"],
                    )

        print("\n")
        print(f"{success_counter} episodes reached the target point in total {n_episodes} episodes")
        print(f"{truncation_counter} episodes ended by time-limit truncation")
        end_time = time.time() - start_time
        print("Total Overshoot 0: ", env.overshoot0)
        print("Total Overshoot 1: ", env.overshoot1)
        print(f"Total Elapsed Time is {int(end_time) / 60} minutes")

        final_paths = _save_checkpoints(
            agent=agent,
            env=env,
            goal_type=goal_type,
            reward_file=resolved_reward_file,
            reward_function=reward_function,
            output_base_dir=output_dir,
            seed_dir=seed_dir,
            scores=scores,
            avg_reward_list=avg_reward_list,
        )
        if tracker.active and upload_checkpoints:
            tracker.log_artifact_files(
                name=f"pytorch-{goal_type}-{resolved_reward_file}",
                artifact_type="model",
                paths=list(final_paths.values()),
                metadata=_expected_metadata(env, goal_type, resolved_reward_file, reward_function),
                aliases=["latest", "final"],
            )
        return scores
    finally:
        tracker.finish()


def evaluate_smoke(
    checkpoint_actor: Path,
    checkpoint_critic: Path,
    goal_type: str = DEFAULT_GOAL_TYPE,
    reward_function: str = DEFAULT_REWARD_FUNCTION,
    max_t: int = 20,
    seed: int | None = None,
    deterministic: bool = False,
    env_kwargs: dict[str, Any] | None = None,
    wandb_cfg: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    task_name: str = "pytorch_eval_smoke",
) -> float:
    _configure_runtime(seed=seed, deterministic=deterministic)
    checkpoint_actor = _resolve_repo_relative_path(checkpoint_actor)
    checkpoint_critic = _resolve_repo_relative_path(checkpoint_critic)
    if env_kwargs is None:
        env = _make_env(goal_type=goal_type, max_episode_steps=max_t)
    else:
        env = _make_env(goal_type=goal_type, max_episode_steps=max_t, env_kwargs=env_kwargs)
    tracker_run_config = dict(run_config or {})
    tracker_run_config.setdefault("runtime_metadata", {})
    tracker_run_config["runtime_metadata"].update(
        {
            "framework": "pytorch",
            "model_arch": MODEL_ARCH,
            "goal_type": goal_type,
            "reward_function": reward_function,
            **_env_runtime_metadata(env),
        }
    )
    tracker = create_wandb_tracker(
        wandb_cfg=wandb_cfg,
        run_config=tracker_run_config,
        context={
            "framework": "pytorch",
            "task_name": task_name,
            "goal_type": goal_type,
            "reward_id": reward_function,
            "seed": seed,
        },
    )
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
    try:
        if tracker.active:
            tracker.log_metrics(
                {
                    "eval/mean_return": float(total_reward),
                    "eval/success_rate": float(success_counter),
                    "eval/truncation_rate": float(truncation_counter),
                },
                step=1,
            )
        return total_reward
    finally:
        tracker.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DDPG runner for continuum RL.")
    parser.add_argument("--mode", choices=["train", "eval-smoke"], default="train")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-t", type=int, default=750)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default=DEFAULT_GOAL_TYPE)
    parser.add_argument("--reward-function", default=DEFAULT_REWARD_FUNCTION)
    parser.add_argument("--reward-file", default=None)
    parser.add_argument("--checkpoint-actor", type=Path, default=None)
    parser.add_argument("--checkpoint-critic", type=Path, default=None)
    parser.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
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
        args.output_base_dir
        / args.goal_type
        / _resolve_reward_file(args.reward_function, args.reward_file)
        / _seed_dir_name(args.seed)
        / "model"
        / "checkpoint_actor.pth"
    )
    critic = args.checkpoint_critic or (
        args.output_base_dir
        / args.goal_type
        / _resolve_reward_file(args.reward_function, args.reward_file)
        / _seed_dir_name(args.seed)
        / "model"
        / "checkpoint_critic.pth"
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
