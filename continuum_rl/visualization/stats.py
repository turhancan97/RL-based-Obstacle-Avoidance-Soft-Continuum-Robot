"""Statistical helpers for paper figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ConfidenceBand:
    x: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray


def align_by_min_length(series_list: Sequence[np.ndarray]) -> np.ndarray:
    if not series_list:
        raise ValueError("series_list must not be empty.")
    min_len = min(len(s) for s in series_list)
    if min_len <= 0:
        raise ValueError("series_list contains empty arrays.")
    return np.vstack([np.asarray(s[:min_len], dtype=np.float64) for s in series_list])


def bootstrap_ci_mean(
    samples_2d: np.ndarray,
    ci_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples_2d.ndim != 2:
        raise ValueError("samples_2d must be a 2D array shaped [n_samples, n_points].")
    n_samples, _ = samples_2d.shape
    if n_samples == 0:
        raise ValueError("samples_2d has no rows.")
    rng = np.random.default_rng(seed)
    mean = samples_2d.mean(axis=0)
    idx = rng.integers(0, n_samples, size=(n_bootstrap, n_samples))
    boot_means = samples_2d[idx, :].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low = np.quantile(boot_means, alpha, axis=0)
    high = np.quantile(boot_means, 1.0 - alpha, axis=0)
    return mean, low, high


def confidence_band_for_series(
    series_list: Sequence[np.ndarray],
    x: np.ndarray,
    ci_level: float,
    n_bootstrap: int,
    seed: int,
) -> ConfidenceBand:
    matrix = align_by_min_length(series_list)
    x_cut = np.asarray(x[: matrix.shape[1]], dtype=np.float64)
    mean, low, high = bootstrap_ci_mean(
        matrix,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return ConfidenceBand(x=x_cut, mean=mean, low=low, high=high)


def summarize_scalar_with_ci(
    values: Iterable[float],
    ci_level: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 123,
) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must not be empty.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(arr.mean()),
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def exploratory_flag(n_seeds: int, min_seeds_for_claims: int) -> bool:
    return n_seeds < min_seeds_for_claims


def cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size < 2 or arr_b.size < 2:
        return float("nan")
    var_a = arr_a.var(ddof=1)
    var_b = arr_b.var(ddof=1)
    pooled_std = np.sqrt(((arr_a.size - 1) * var_a + (arr_b.size - 1) * var_b) / (arr_a.size + arr_b.size - 2))
    if pooled_std == 0:
        return 0.0
    return float((arr_a.mean() - arr_b.mean()) / pooled_std)
