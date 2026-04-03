"""Strict local run-layout discovery and training-curve loading."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


@dataclass(frozen=True)
class RunRecord:
    framework: str
    goal_type: str
    reward_id: str
    seed: int
    run_dir: Path
    rewards_dir: Path
    reward_series_path: Path
    avg_reward_series_path: Path
    actor_checkpoint_path: Path
    critic_checkpoint_path: Optional[Path]

    @property
    def reward_function(self) -> str:
        if self.reward_id.startswith("reward_"):
            return self.reward_id[len("reward_") :]
        return self.reward_id


@dataclass(frozen=True)
class SkippedRun:
    run_dir: Path
    reason: str


def _required_paths_for_framework(
    framework: str,
    run_dir: Path,
) -> Tuple[Path, Path, Path, Optional[Path]]:
    rewards_dir = run_dir / "rewards"
    model_dir = run_dir / "model"

    if framework == "pytorch":
        reward_series_path = rewards_dir / "scores.pickle"
        avg_reward_series_path = rewards_dir / "avg_reward_list.pickle"
        actor_checkpoint = model_dir / "checkpoint_actor.pth"
        critic_checkpoint = model_dir / "checkpoint_critic.pth"
        return reward_series_path, avg_reward_series_path, actor_checkpoint, critic_checkpoint

    if framework == "keras":
        reward_series_path = rewards_dir / "ep_reward_list.pickle"
        avg_reward_series_path = rewards_dir / "avg_reward_list.pickle"
        actor_h5 = model_dir / "continuum_actor.h5"
        actor_weights_h5 = model_dir / "continuum_actor.weights.h5"
        actor_checkpoint = actor_h5 if actor_h5.exists() else actor_weights_h5
        return reward_series_path, avg_reward_series_path, actor_checkpoint, None

    raise ValueError(f"Unsupported framework={framework}")


def discover_runs(
    runs_root: Path,
    include_goal_types: Sequence[str],
) -> Tuple[List[RunRecord], List[SkippedRun]]:
    """Discover valid run directories under strict layout:

    runs/<framework>/<goal>/<reward>/seed_<id>/
    """
    runs: List[RunRecord] = []
    skipped: List[SkippedRun] = []
    include_goal_set = set(include_goal_types)

    if not runs_root.exists():
        skipped.append(SkippedRun(run_dir=runs_root, reason="runs_root_missing"))
        return runs, skipped

    for framework_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        framework = framework_dir.name.lower()
        if framework not in {"pytorch", "keras"}:
            continue

        for goal_dir in sorted(p for p in framework_dir.iterdir() if p.is_dir()):
            goal_type = goal_dir.name
            if goal_type not in include_goal_set:
                continue

            for reward_dir in sorted(p for p in goal_dir.iterdir() if p.is_dir()):
                reward_id = reward_dir.name
                for seed_dir in sorted(p for p in reward_dir.iterdir() if p.is_dir()):
                    match = SEED_DIR_RE.match(seed_dir.name)
                    if match is None:
                        skipped.append(SkippedRun(run_dir=seed_dir, reason="seed_dir_name_invalid"))
                        continue
                    seed = int(match.group(1))
                    reward_series, avg_reward_series, actor_ckpt, critic_ckpt = _required_paths_for_framework(
                        framework=framework,
                        run_dir=seed_dir,
                    )
                    missing: List[str] = []
                    if not reward_series.exists():
                        missing.append(str(reward_series))
                    if not avg_reward_series.exists():
                        missing.append(str(avg_reward_series))
                    if not actor_ckpt.exists():
                        missing.append(str(actor_ckpt))
                    if critic_ckpt is not None and not critic_ckpt.exists():
                        missing.append(str(critic_ckpt))

                    if missing:
                        skipped.append(
                            SkippedRun(
                                run_dir=seed_dir,
                                reason=f"missing_required_files:{';'.join(missing)}",
                            )
                        )
                        continue

                    runs.append(
                        RunRecord(
                            framework=framework,
                            goal_type=goal_type,
                            reward_id=reward_id,
                            seed=seed,
                            run_dir=seed_dir,
                            rewards_dir=seed_dir / "rewards",
                            reward_series_path=reward_series,
                            avg_reward_series_path=avg_reward_series,
                            actor_checkpoint_path=actor_ckpt,
                            critic_checkpoint_path=critic_ckpt,
                        )
                    )

    return runs, skipped


def load_pickle_series(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = pickle.load(f)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def load_run_series(run: RunRecord) -> Dict[str, np.ndarray]:
    ep_reward = load_pickle_series(run.reward_series_path)
    avg_reward = load_pickle_series(run.avg_reward_series_path)
    return {
        "episode_reward": ep_reward,
        "avg_reward": avg_reward,
    }


def group_runs_by_key(runs: Iterable[RunRecord]) -> Dict[Tuple[str, str, str], List[RunRecord]]:
    grouped: Dict[Tuple[str, str, str], List[RunRecord]] = {}
    for run in runs:
        key = (run.framework, run.goal_type, run.reward_id)
        grouped.setdefault(key, []).append(run)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda item: item.seed)
    return grouped
