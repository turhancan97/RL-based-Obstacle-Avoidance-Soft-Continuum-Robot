"""Reward visualization helper for PyTorch experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from continuum_robot.utils import load_pickle_file, reward_log10_visualization, reward_visualization


BASE_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = BASE_DIR.parent
DEFAULT_GOAL_TYPE = "fixed_goal"
DEFAULT_REWARD_TYPE = "reward_step_minus_weighted_euclidean"


def _resolve_reward_dir(goal_type: str, reward_type: str, base_dir: Path) -> Path:
    exact = base_dir / goal_type / reward_type / "rewards"
    if exact.exists():
        return exact

    # Seed-aware layout: <base>/<goal>/<reward>/seed_<id>/rewards
    seed_parent = base_dir / goal_type / reward_type
    seed_dirs = sorted(
        p for p in seed_parent.glob("seed_*") if p.is_dir() and (p / "rewards").exists()
    )
    if seed_dirs:
        return seed_dirs[0] / "rewards"

    parent = base_dir / goal_type
    candidates = sorted(p for p in parent.glob(f"{reward_type}*") if (p / "rewards").exists())
    if candidates:
        return candidates[0] / "rewards"

    available = sorted([p.name for p in parent.iterdir() if p.is_dir()]) if parent.exists() else []
    raise FileNotFoundError(
        f"Could not resolve reward directory for goal_type='{goal_type}', reward_type='{reward_type}'. "
        f"Available under '{parent}': {available}"
    )


def run(
    goal_type: str = DEFAULT_GOAL_TYPE,
    reward_type: str = DEFAULT_REWARD_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    root = PYTORCH_DIR if base_dir is None else Path(base_dir)
    reward_dir = _resolve_reward_dir(goal_type, reward_type, root)
    avg_reward_list = load_pickle_file(str((reward_dir / "avg_reward_list").resolve()))
    ep_reward_list = load_pickle_file(str((reward_dir / "scores").resolve()))
    output_dir = reward_dir / "plots"

    reward_visualization(
        ep_reward_list,
        avg_reward_list,
        output_dir=output_dir,
        filename="reward_visualization.png",
    )
    reward_log10_visualization(
        ep_reward_list,
        avg_reward_list,
        output_dir=output_dir,
        filename="reward_log10_visualization.png",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch reward visualization runner.")
    parser.add_argument("--goal-type", default=DEFAULT_GOAL_TYPE)
    parser.add_argument("--reward-type", default=DEFAULT_REWARD_TYPE)
    parser.add_argument("--base-dir", type=Path, default=PYTORCH_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run(goal_type=args.goal_type, reward_type=args.reward_type, base_dir=args.base_dir)
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"[reward-vis] {exc}")
