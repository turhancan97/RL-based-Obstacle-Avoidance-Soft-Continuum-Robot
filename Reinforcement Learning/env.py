"""Backward-compatible environment module.

This module keeps the historical import path while delegating to the
new `continuum_rl.env` implementation.
"""

from continuum_rl.env import ContinuumEnv, continuumEnv

__all__ = ["ContinuumEnv", "continuumEnv"]
