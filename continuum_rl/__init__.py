"""Continuum RL package with env/runtime compatibility utilities."""

from __future__ import annotations

__all__ = ["ContinuumEnv", "continuumEnv"]


def __getattr__(name: str):
    if name in {"ContinuumEnv", "continuumEnv"}:
        from .env import ContinuumEnv, continuumEnv

        return ContinuumEnv if name == "ContinuumEnv" else continuumEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
