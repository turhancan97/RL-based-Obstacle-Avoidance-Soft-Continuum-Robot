"""Reward visualization helper for Keras experiments."""

from __future__ import annotations

from pathlib import Path

import yaml

from continuum_robot.utils import load_pickle_file, reward_log10_visualization, reward_visualization


BASE_DIR = Path(__file__).resolve().parent
KERAS_DIR = BASE_DIR.parent


def _load_config() -> dict:
    with (BASE_DIR / "config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_reward_dir(goal_type: str, reward_type: str) -> Path:
    exact = KERAS_DIR / goal_type / reward_type / "rewards"
    if exact.exists():
        return exact

    parent = KERAS_DIR / goal_type
    candidates = sorted(
        p for p in parent.glob(f"{reward_type}*") if (p / "rewards").exists()
    )
    if candidates:
        return candidates[0] / "rewards"

    available = sorted([p.name for p in parent.iterdir() if p.is_dir()]) if parent.exists() else []
    raise FileNotFoundError(
        f"Could not resolve reward directory for goal_type='{goal_type}', reward_type='{reward_type}'. "
        f"Available under '{parent}': {available}"
    )


def main() -> None:
    config = _load_config()
    reward_dir = _resolve_reward_dir(config["goal_type"], config["reward_type"])
    avg_reward_list = load_pickle_file(str((reward_dir / "avg_reward_list").resolve()))
    ep_reward_list = load_pickle_file(str((reward_dir / "ep_reward_list").resolve()))
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


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"[reward-vis] {exc}")
