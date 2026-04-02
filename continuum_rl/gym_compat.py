"""Compatibility helpers for Gymnasium/Gym API differences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


def _import_gym_backend():
    try:
        import gymnasium as gym_backend  # type: ignore

        return gym_backend, "gymnasium"
    except ImportError:
        import gym as gym_backend  # type: ignore

        return gym_backend, "gym"


gym, GYM_BACKEND = _import_gym_backend()
spaces = gym.spaces


@dataclass(frozen=True)
class StepOutput:
    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict

    def legacy(self) -> Tuple[Any, float, bool, dict]:
        return self.obs, self.reward, self.terminated or self.truncated, self.info


def unpack_reset_output(reset_output: Any) -> Tuple[Any, dict]:
    """Return `(obs, info)` for both Gym and Gymnasium style outputs."""
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        return reset_output
    return reset_output, {}


def unpack_step_output(step_output: Any) -> StepOutput:
    """Return a normalized 5-tuple-style output object."""
    if isinstance(step_output, tuple) and len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        return StepOutput(obs, float(reward), bool(terminated), bool(truncated), dict(info))

    if isinstance(step_output, tuple) and len(step_output) == 4:
        obs, reward, done, info = step_output
        return StepOutput(obs, float(reward), bool(done), False, dict(info))

    raise ValueError(
        "Unsupported step output format. Expected 4-tuple (Gym) or 5-tuple (Gymnasium)."
    )
