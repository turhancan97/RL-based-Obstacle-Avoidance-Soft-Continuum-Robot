"""Deterministic checkpoint rollouts for visualization metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from continuum_rl.env import ContinuumEnv
from continuum_rl.gym_compat import unpack_step_output
from continuum_rl.visualization.data import RunRecord


PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class RolloutEpisode:
    positions: np.ndarray
    actions: np.ndarray
    kappas: np.ndarray
    rewards: np.ndarray
    clearances: np.ndarray
    terminated: bool
    truncated: bool
    total_return: float
    length: int
    min_clearance: float
    final_position: np.ndarray
    goal_position: np.ndarray
    action_saturation_rate: float
    action_smoothness: float


@dataclass
class RolloutSummary:
    run: RunRecord
    episodes: List[RolloutEpisode]
    obstacles: np.ndarray

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([1.0 if e.terminated else 0.0 for e in self.episodes]))

    @property
    def truncation_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([1.0 if e.truncated else 0.0 for e in self.episodes]))

    @property
    def mean_return(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.total_return for e in self.episodes]))

    @property
    def mean_min_clearance(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.min_clearance for e in self.episodes]))

    @property
    def mean_length(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.length for e in self.episodes]))


def _load_policy_pytorch(actor_checkpoint: Path, state_dim: int, action_dim: int = 3) -> PolicyFn:
    try:
        import torch
        from Pytorch.model import Actor
    except Exception as exc:  # pragma: no cover - dependency/env specific
        raise RuntimeError(f"Failed to import PyTorch policy runtime: {exc}") from exc

    model = Actor(state_size=state_dim, action_size=action_dim, seed=0)
    state_dict = torch.load(actor_checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    def _policy(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action = model(torch.from_numpy(obs).float()).cpu().numpy()
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    return _policy


def _load_policy_keras(actor_checkpoint: Path, state_dim: int, action_dim: int = 3) -> PolicyFn:
    try:
        import tensorflow as tf
        from Keras.DDPG import get_actor
    except Exception as exc:  # pragma: no cover - dependency/env specific
        raise RuntimeError(f"Failed to import Keras policy runtime: {exc}") from exc

    actor_model = get_actor(num_states=state_dim, num_actions=action_dim, upper_bound=1.0)
    actor_model.load_weights(actor_checkpoint)

    def _policy(obs: np.ndarray) -> np.ndarray:
        tf_obs = tf.expand_dims(tf.convert_to_tensor(obs), 0)
        action = tf.squeeze(actor_model(tf_obs)).numpy()
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    return _policy


def build_policy_for_run(run: RunRecord, state_dim: int) -> PolicyFn:
    if run.framework == "pytorch":
        return _load_policy_pytorch(run.actor_checkpoint_path, state_dim=state_dim, action_dim=3)
    if run.framework == "keras":
        return _load_policy_keras(run.actor_checkpoint_path, state_dim=state_dim, action_dim=3)
    raise ValueError(f"Unsupported framework={run.framework}")


def _compute_clearance(state: np.ndarray) -> float:
    tip_x, tip_y = float(state[0]), float(state[1])
    obstacle_xy = state[7:]
    if obstacle_xy.size < 2:
        return float("nan")
    distances: List[float] = []
    for idx in range(0, obstacle_xy.size, 2):
        ox = float(obstacle_xy[idx])
        oy = float(obstacle_xy[idx + 1])
        distances.append(float(np.hypot(tip_x - ox, tip_y - oy)))
    return float(np.min(distances))


def evaluate_run_rollouts(
    run: RunRecord,
    rollouts_per_seed: int,
    max_steps: int,
    reward_function: str,
    env_kwargs: Optional[Dict[str, Any]],
    seed_base: int,
) -> RolloutSummary:
    env = ContinuumEnv(
        observation_mode="canonical",
        goal_type=run.goal_type,  # type: ignore[arg-type]
        max_episode_steps=max_steps,
        **dict(env_kwargs or {}),
    )
    state_dim = int(env.obs_size)
    policy = build_policy_for_run(run=run, state_dim=state_dim)

    episodes: List[RolloutEpisode] = []
    obstacles = np.asarray([[float(o["x"]), float(o["y"])] for o in env.obstacles], dtype=np.float64)
    for rollout_idx in range(rollouts_per_seed):
        rollout_seed = int(seed_base + (run.seed * 10000) + rollout_idx)
        state, _ = env.reset(seed=rollout_seed)
        goal_position = np.asarray(state[2:4], dtype=np.float64)
        positions: List[np.ndarray] = [np.asarray(state[:2], dtype=np.float64)]
        actions: List[np.ndarray] = []
        kappas: List[np.ndarray] = [np.asarray(state[4:7], dtype=np.float64)]
        rewards: List[float] = []
        clearances: List[float] = [float(_compute_clearance(state))]
        terminated = False
        truncated = False

        for _ in range(max_steps):
            action = policy(state)
            step_out = unpack_step_output(env.step(action, reward_function=reward_function))
            state = step_out.obs
            actions.append(np.asarray(action, dtype=np.float64))
            positions.append(np.asarray(state[:2], dtype=np.float64))
            kappas.append(np.asarray(state[4:7], dtype=np.float64))
            rewards.append(float(step_out.reward))
            clearances.append(float(_compute_clearance(state)))
            if step_out.terminated or step_out.truncated:
                terminated = bool(step_out.terminated)
                truncated = bool(step_out.truncated)
                break

        actions_arr = np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 3), dtype=np.float64)
        diffs = np.diff(actions_arr, axis=0) if len(actions_arr) > 1 else np.zeros((0, 3), dtype=np.float64)
        saturation_rate = 0.0
        if actions_arr.size > 0:
            saturation_rate = float(np.mean(np.abs(actions_arr) >= 0.999))
        action_smoothness = float(np.mean(np.abs(diffs))) if diffs.size > 0 else 0.0

        episodes.append(
            RolloutEpisode(
                positions=np.asarray(positions, dtype=np.float64),
                actions=actions_arr,
                kappas=np.asarray(kappas, dtype=np.float64),
                rewards=np.asarray(rewards, dtype=np.float64),
                clearances=np.asarray(clearances, dtype=np.float64),
                terminated=terminated,
                truncated=truncated,
                total_return=float(np.sum(rewards)),
                length=len(rewards),
                min_clearance=float(np.nanmin(clearances)) if clearances else float("nan"),
                final_position=np.asarray(state[:2], dtype=np.float64),
                goal_position=goal_position,
                action_saturation_rate=saturation_rate,
                action_smoothness=action_smoothness,
            )
        )

    return RolloutSummary(run=run, episodes=episodes, obstacles=obstacles)
