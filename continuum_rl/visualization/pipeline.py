"""Pipeline entrypoint for conference-grade paper figures."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from continuum_rl.visualization.data import RunRecord, discover_runs, load_run_series
from continuum_rl.visualization.figures import (
    FigureContext,
    plot_ablation_summary,
    plot_control_diagnostics,
    plot_curvature_dynamics,
    plot_framework_comparison,
    plot_learning_curves,
    plot_obstacle_clearance_profiles,
    plot_safety_performance_pareto,
    plot_sample_efficiency,
    plot_stability_diagnostics,
    plot_success_truncation_timeline,
    plot_terminal_scatter,
    plot_trajectory_atlas,
    plot_workspace_occupancy_heatmap,
)
from continuum_rl.visualization.rollouts import RolloutSummary, evaluate_run_rollouts
from continuum_rl.visualization.style import apply_conference_theme


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _config_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _prepare_output_dir(output_dir: Path, clear_output_dir: bool) -> None:
    if clear_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_all_training_series(runs: Sequence[RunRecord]) -> Dict[RunRecord, Dict[str, Any]]:
    loaded: Dict[RunRecord, Dict[str, Any]] = {}
    for run in runs:
        loaded[run] = load_run_series(run)
    return loaded


def _run_rollout_suite(
    runs: Sequence[RunRecord],
    rollouts_per_seed: int,
    max_steps: int,
    reward_function_default: str,
    env_kwargs: Optional[Dict[str, Any]],
    bootstrap_seed: int,
) -> tuple[List[RolloutSummary], List[dict]]:
    summaries: List[RolloutSummary] = []
    skipped: List[dict] = []
    for run in runs:
        reward_function = run.reward_function or reward_function_default
        try:
            summary = evaluate_run_rollouts(
                run=run,
                rollouts_per_seed=rollouts_per_seed,
                max_steps=max_steps,
                reward_function=reward_function,
                env_kwargs=env_kwargs,
                seed_base=bootstrap_seed,
            )
            summaries.append(summary)
        except Exception as exc:
            skipped.append({"run_dir": str(run.run_dir), "reason": f"rollout_failed:{exc}"})
            print(f"[paper-figures] skipping rollout for {run.run_dir}: {exc}")
    return summaries, skipped


def run_paper_figures(
    runs_root: Path,
    output_dir: Path,
    image_format: str = "jpeg",
    show: bool = False,
    min_seeds_for_claims: int = 5,
    ci_method: str = "bootstrap",
    ci_level: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 123,
    rollouts_per_seed: int = 100,
    include_goal_types: Sequence[str] = ("fixed_goal", "random_goal"),
    max_steps: int = 750,
    reward_function: str = "step_minus_weighted_euclidean",
    clear_output_dir: bool = True,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    if ci_method != "bootstrap":
        raise ValueError(f"Unsupported ci_method={ci_method}; expected bootstrap.")
    if image_format.lower() not in {"jpeg", "jpg"}:
        raise ValueError(f"Unsupported format={image_format}; only jpeg is supported.")

    apply_conference_theme()
    _prepare_output_dir(output_dir=output_dir, clear_output_dir=clear_output_dir)

    runs, skipped_runs = discover_runs(
        runs_root=runs_root,
        include_goal_types=include_goal_types,
    )
    training_series = _load_all_training_series(runs)
    rollout_summaries, skipped_rollouts = _run_rollout_suite(
        runs=runs,
        rollouts_per_seed=rollouts_per_seed,
        max_steps=max_steps,
        reward_function_default=reward_function,
        env_kwargs=env_kwargs,
        bootstrap_seed=bootstrap_seed,
    )

    ctx = FigureContext(
        output_dir=output_dir,
        image_format=image_format,
        show=show,
        ci_level=ci_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        min_seeds_for_claims=min_seeds_for_claims,
        max_steps=max_steps,
    )

    figure_manifest: List[dict] = []
    plot_learning_curves(training_series=training_series, ctx=ctx, manifest=figure_manifest)
    plot_sample_efficiency(
        rollout_summaries=rollout_summaries,
        training_series=training_series,
        ctx=ctx,
        manifest=figure_manifest,
    )
    plot_success_truncation_timeline(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_safety_performance_pareto(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_framework_comparison(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_ablation_summary(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_trajectory_atlas(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_workspace_occupancy_heatmap(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_terminal_scatter(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_obstacle_clearance_profiles(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_control_diagnostics(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_curvature_dynamics(rollout_summaries=rollout_summaries, ctx=ctx, manifest=figure_manifest)
    plot_stability_diagnostics(runs=runs, ctx=ctx, manifest=figure_manifest)

    settings_payload: Dict[str, Any] = {
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "format": image_format,
        "show": show,
        "min_seeds_for_claims": min_seeds_for_claims,
        "ci_method": ci_method,
        "ci_level": ci_level,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "rollouts_per_seed": rollouts_per_seed,
        "include_goal_types": list(include_goal_types),
        "max_steps": max_steps,
        "reward_function": reward_function,
        "clear_output_dir": clear_output_dir,
    }
    manifest_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_hash": _config_hash(settings_payload),
        "settings": settings_payload,
        "source_runs": [str(run.run_dir) for run in runs],
        "skipped_runs": [{"run_dir": str(item.run_dir), "reason": item.reason} for item in skipped_runs],
        "skipped_rollouts": skipped_rollouts,
        "figures": figure_manifest,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return output_dir
