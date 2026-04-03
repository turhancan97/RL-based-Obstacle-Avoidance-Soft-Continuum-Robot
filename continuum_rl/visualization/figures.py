"""Paper-figure rendering for RL and robotics diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from continuum_rl.visualization.data import RunRecord
from continuum_rl.visualization.rollouts import RolloutSummary
from continuum_rl.visualization.stats import (
    cohen_d,
    confidence_band_for_series,
    exploratory_flag,
    summarize_scalar_with_ci,
)
from continuum_rl.visualization.style import framework_color_map, placeholder_figure, save_figure


@dataclass(frozen=True)
class FigureContext:
    output_dir: Path
    image_format: str
    show: bool
    ci_level: float
    bootstrap_samples: int
    bootstrap_seed: int
    min_seeds_for_claims: int
    max_steps: int


def _record_figure(
    manifest: List[dict],
    name: str,
    path: Path,
    sources: Sequence[str],
    exploratory: bool = False,
    notes: str = "",
) -> None:
    manifest.append(
        {
            "name": name,
            "path": str(path),
            "sources": list(sources),
            "exploratory": bool(exploratory),
            "notes": notes,
        }
    )


def _framework_colors() -> Dict[str, str]:
    cmap = framework_color_map()
    cmap.setdefault("pytorch", "#0072B2")
    cmap.setdefault("keras", "#D55E00")
    return cmap


def _group_rollout_summaries_by_framework(
    rollout_summaries: Sequence[RolloutSummary],
) -> Dict[str, List[RolloutSummary]]:
    grouped: Dict[str, List[RolloutSummary]] = {}
    for summary in rollout_summaries:
        grouped.setdefault(summary.run.framework, []).append(summary)
    return grouped


def _final_avg_reward(training_series: Mapping[RunRecord, Dict[str, np.ndarray]], run: RunRecord) -> float:
    series = training_series.get(run, {})
    arr = np.asarray(series.get("avg_reward", np.array([], dtype=np.float64)), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr[-1])


def _collect_stability_metrics(run: RunRecord) -> Optional[dict]:
    try:
        import pandas as pd
    except Exception:
        return None

    candidates = [
        run.run_dir / "metrics" / "opt_metrics.csv",
        run.run_dir / "opt_metrics.csv",
        run.run_dir / "metrics.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    lower_cols = {c.lower(): c for c in df.columns}

    def _pick(*aliases: str) -> Optional[np.ndarray]:
        for alias in aliases:
            key = alias.lower()
            if key in lower_cols:
                return np.asarray(df[lower_cols[key]].to_numpy(), dtype=np.float64)
        return None

    actor_loss = _pick("opt/actor_loss", "actor_loss", "opt_actor_loss")
    critic_loss = _pick("opt/critic_loss", "critic_loss", "opt_critic_loss")
    actor_grad = _pick("opt/actor_grad_norm", "actor_grad_norm", "opt_actor_grad_norm")
    critic_grad = _pick("opt/critic_grad_norm", "critic_grad_norm", "opt_critic_grad_norm")
    if actor_loss is None and critic_loss is None and actor_grad is None and critic_grad is None:
        return None
    step = _pick("step", "global_step", "_step")
    if step is None:
        max_len = max(
            len(arr)
            for arr in (actor_loss, critic_loss, actor_grad, critic_grad)
            if arr is not None
        )
        step = np.arange(1, max_len + 1, dtype=np.float64)
    return {
        "step": step,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "actor_grad": actor_grad,
        "critic_grad": critic_grad,
    }


def plot_learning_curves(
    training_series: Mapping[RunRecord, Dict[str, np.ndarray]],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    grouped: Dict[Tuple[str, str, str], List[RunRecord]] = {}
    for run in training_series:
        grouped.setdefault((run.framework, run.goal_type, run.reward_id), []).append(run)

    if not grouped:
        fig = placeholder_figure("Learning Curves", "No valid runs found in strict layout.")
        path = save_figure(fig, ctx.output_dir, "learning_curves", ctx.image_format, ctx.show)
        _record_figure(manifest, "learning_curves", path, sources=[], exploratory=True, notes="no_runs")
        return

    fig, ax = plt.subplots(1, 1)
    colors = _framework_colors()
    any_exploratory = False

    for key, runs in sorted(grouped.items()):
        framework, goal_type, reward_id = key
        runs_sorted = sorted(runs, key=lambda r: r.seed)
        series_list = [training_series[r]["avg_reward"] for r in runs_sorted if training_series[r]["avg_reward"].size > 0]
        if not series_list:
            continue
        min_len = min(len(s) for s in series_list)
        x = np.arange(1, min_len + 1, dtype=np.float64) * float(ctx.max_steps)
        band = confidence_band_for_series(
            series_list=series_list,
            x=x,
            ci_level=ctx.ci_level,
            n_bootstrap=ctx.bootstrap_samples,
            seed=ctx.bootstrap_seed,
        )
        is_exploratory = exploratory_flag(len(series_list), ctx.min_seeds_for_claims)
        any_exploratory = any_exploratory or is_exploratory
        label = f"{framework}|{goal_type}|{reward_id}|n={len(series_list)}"
        if is_exploratory:
            label += "|exploratory"
        color = colors.get(framework, "#444444")
        ax.plot(band.x, band.mean, label=label, linewidth=2.2, color=color)
        ax.fill_between(band.x, band.low, band.high, color=color, alpha=0.18)

    ax.set_title("Multi-Seed Learning Curves (Avg Episode Reward)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Episodic Reward")
    ax.legend(loc="best", fontsize=8)
    path = save_figure(fig, ctx.output_dir, "learning_curves", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "learning_curves",
        path,
        sources=[str(r.run_dir) for r in training_series.keys()],
        exploratory=any_exploratory,
        notes="bootstrap_ci",
    )


def plot_sample_efficiency(
    rollout_summaries: Sequence[RolloutSummary],
    training_series: Mapping[RunRecord, Dict[str, np.ndarray]],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    if not rollout_summaries:
        fig = placeholder_figure("Sample Efficiency", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "sample_efficiency", ctx.image_format, ctx.show)
        _record_figure(manifest, "sample_efficiency", path, sources=[], exploratory=True, notes="no_rollouts")
        return

    colors = _framework_colors()
    fig, ax = plt.subplots(1, 1)
    for summary in rollout_summaries:
        series = training_series.get(summary.run, {})
        n_episodes = int(len(series.get("avg_reward", np.array([]))))
        env_steps = max(n_episodes, 1) * int(ctx.max_steps)
        ax.scatter(
            env_steps,
            summary.success_rate,
            color=colors.get(summary.run.framework, "#666666"),
            alpha=0.8,
            label=summary.run.framework,
            s=36,
        )
    # deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    uniq: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        uniq.setdefault(label, handle)
    ax.legend(uniq.values(), uniq.keys(), loc="best")
    ax.set_title("Sample Efficiency")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Success Rate (Checkpoint Rollouts)")
    path = save_figure(fig, ctx.output_dir, "sample_efficiency", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "sample_efficiency",
        path,
        sources=[str(s.run.run_dir) for s in rollout_summaries],
    )


def plot_success_truncation_timeline(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    grouped = _group_rollout_summaries_by_framework(rollout_summaries)
    if not grouped:
        fig = placeholder_figure("Success/Truncation Timeline", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "success_truncation_timeline", ctx.image_format, ctx.show)
        _record_figure(manifest, "success_truncation_timeline", path, sources=[], exploratory=True, notes="no_rollouts")
        return

    colors = _framework_colors()
    fig, ax = plt.subplots(1, 1)
    for framework, summaries in sorted(grouped.items()):
        success_rows: List[np.ndarray] = []
        trunc_rows: List[np.ndarray] = []
        for summary in summaries:
            success_rows.append(np.asarray([1.0 if e.terminated else 0.0 for e in summary.episodes], dtype=np.float64))
            trunc_rows.append(np.asarray([1.0 if e.truncated else 0.0 for e in summary.episodes], dtype=np.float64))
        min_len = min(len(r) for r in success_rows) if success_rows else 0
        if min_len <= 0:
            continue
        success_mat = np.vstack([r[:min_len] for r in success_rows])
        trunc_mat = np.vstack([r[:min_len] for r in trunc_rows])
        x = np.arange(1, min_len + 1, dtype=np.float64)
        success_curve = np.cumsum(success_mat, axis=1) / x
        trunc_curve = np.cumsum(trunc_mat, axis=1) / x
        ax.plot(x, success_curve.mean(axis=0), color=colors.get(framework, "#444444"), linewidth=2.2, label=f"{framework} success")
        ax.plot(
            x,
            trunc_curve.mean(axis=0),
            color=colors.get(framework, "#444444"),
            linestyle="--",
            linewidth=1.8,
            label=f"{framework} truncation",
        )
    ax.set_title("Success/Truncation Timeline (Cumulative)")
    ax.set_xlabel("Rollout Index")
    ax.set_ylabel("Rate")
    ax.legend(loc="best")
    path = save_figure(fig, ctx.output_dir, "success_truncation_timeline", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "success_truncation_timeline",
        path,
        sources=[str(s.run.run_dir) for s in rollout_summaries],
    )


def plot_safety_performance_pareto(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    if not rollout_summaries:
        fig = placeholder_figure("Safety-Performance Pareto", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "safety_performance_pareto", ctx.image_format, ctx.show)
        _record_figure(manifest, "safety_performance_pareto", path, sources=[], exploratory=True, notes="no_rollouts")
        return
    colors = _framework_colors()
    fig, ax = plt.subplots(1, 1)
    for summary in rollout_summaries:
        ax.scatter(
            summary.mean_min_clearance,
            summary.success_rate,
            color=colors.get(summary.run.framework, "#555555"),
            s=48,
            alpha=0.8,
            label=summary.run.framework,
        )
    handles, labels = ax.get_legend_handles_labels()
    uniq: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        uniq.setdefault(label, handle)
    ax.legend(uniq.values(), uniq.keys(), loc="best")
    ax.set_title("Safety-Performance Pareto")
    ax.set_xlabel("Mean Min Obstacle Clearance")
    ax.set_ylabel("Success Rate")
    path = save_figure(fig, ctx.output_dir, "safety_performance_pareto", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "safety_performance_pareto",
        path,
        sources=[str(s.run.run_dir) for s in rollout_summaries],
    )


def plot_framework_comparison(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    grouped = _group_rollout_summaries_by_framework(rollout_summaries)
    if not grouped:
        fig = placeholder_figure("Framework Comparison", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "framework_comparison", ctx.image_format, ctx.show)
        _record_figure(manifest, "framework_comparison", path, sources=[], exploratory=True, notes="no_rollouts")
        return

    frameworks = sorted(grouped.keys())
    metrics = ["success_rate", "mean_return", "mean_min_clearance"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    colors = _framework_colors()

    for ax, metric in zip(axs, metrics):
        vals = []
        lows = []
        highs = []
        for fw in frameworks:
            series = [float(getattr(summary, metric)) for summary in grouped[fw]]
            mean, low, high = summarize_scalar_with_ci(
                values=series,
                ci_level=ctx.ci_level,
                n_bootstrap=ctx.bootstrap_samples,
                seed=ctx.bootstrap_seed,
            )
            vals.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        xpos = np.arange(len(frameworks))
        ax.bar(xpos, vals, color=[colors.get(fw, "#444444") for fw in frameworks], alpha=0.9)
        ax.errorbar(xpos, vals, yerr=[lows, highs], fmt="none", color="black", capsize=4)
        ax.set_xticks(xpos)
        ax.set_xticklabels(frameworks)
        ax.set_title(metric.replace("_", " ").title())
    fig.suptitle("Framework Comparison with Bootstrap CI")

    path = save_figure(fig, ctx.output_dir, "framework_comparison", ctx.image_format, ctx.show)
    notes = ""
    if "pytorch" in grouped and "keras" in grouped:
        d_val = cohen_d(
            [s.mean_return for s in grouped["pytorch"]],
            [s.mean_return for s in grouped["keras"]],
        )
        notes = f"cohen_d_mean_return={d_val:.4f}"
    _record_figure(
        manifest,
        "framework_comparison",
        path,
        sources=[str(s.run.run_dir) for s in rollout_summaries],
        notes=notes,
    )


def plot_ablation_summary(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    grouped: Dict[str, List[float]] = {}
    for summary in rollout_summaries:
        key = f"{summary.run.framework}|{summary.run.reward_id}"
        grouped.setdefault(key, []).append(summary.mean_return)
    if not grouped:
        fig = placeholder_figure("Ablation Summary", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "ablation_summary", ctx.image_format, ctx.show)
        _record_figure(manifest, "ablation_summary", path, sources=[], exploratory=True, notes="no_rollouts")
        return
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(grouped) * 1.4), 6))
    labels = sorted(grouped.keys())
    data = [grouped[label] for label in labels]
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:  # Matplotlib < 3.9 compatibility
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title("Ablation Summary (Mean Return by Framework/Reward)")
    ax.set_ylabel("Mean Return")
    ax.tick_params(axis="x", rotation=35)
    path = save_figure(fig, ctx.output_dir, "ablation_summary", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "ablation_summary",
        path,
        sources=[str(s.run.run_dir) for s in rollout_summaries],
    )


def plot_trajectory_atlas(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    if not rollout_summaries:
        fig = placeholder_figure("Trajectory Atlas", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "trajectory_atlas", ctx.image_format, ctx.show)
        _record_figure(manifest, "trajectory_atlas", path, sources=[], exploratory=True, notes="no_rollouts")
        return
    fig, ax = plt.subplots(1, 1)
    for summary in rollout_summaries:
        for episode in summary.episodes:
            if episode.positions.size == 0:
                continue
            color_val = np.clip((episode.total_return + 1000.0) / 2000.0, 0.0, 1.0)
            ax.plot(episode.positions[:, 0], episode.positions[:, 1], color=plt.cm.viridis(color_val), alpha=0.22)
    # obstacle markers from first summary
    obstacles = rollout_summaries[0].obstacles
    if obstacles.size > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], marker="x", color="black", s=40, label="obstacles")
    ax.set_title("Trajectory Atlas (Color by Return)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    path = save_figure(fig, ctx.output_dir, "trajectory_atlas", ctx.image_format, ctx.show)
    _record_figure(manifest, "trajectory_atlas", path, sources=[str(s.run.run_dir) for s in rollout_summaries])


def plot_workspace_occupancy_heatmap(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    points: List[np.ndarray] = []
    for summary in rollout_summaries:
        for episode in summary.episodes:
            if episode.positions.size > 0:
                points.append(episode.positions)
    if not points:
        fig = placeholder_figure("Workspace Occupancy", "No trajectory points available.")
        path = save_figure(fig, ctx.output_dir, "workspace_occupancy_heatmap", ctx.image_format, ctx.show)
        _record_figure(manifest, "workspace_occupancy_heatmap", path, sources=[], exploratory=True, notes="no_positions")
        return
    data = np.vstack(points)
    fig, ax = plt.subplots(1, 1)
    hist = ax.hist2d(data[:, 0], data[:, 1], bins=50, cmap="magma")
    fig.colorbar(hist[3], ax=ax, label="count")
    ax.set_title("Workspace Occupancy Heatmap")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    path = save_figure(fig, ctx.output_dir, "workspace_occupancy_heatmap", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "workspace_occupancy_heatmap",
        path,
        sources=[str(s.run.run_dir) for s in rollout_summaries],
    )


def plot_terminal_scatter(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    if not rollout_summaries:
        fig = placeholder_figure("Terminal Scatter", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "terminal_scatter", ctx.image_format, ctx.show)
        _record_figure(manifest, "terminal_scatter", path, sources=[], exploratory=True, notes="no_rollouts")
        return
    fig, ax = plt.subplots(1, 1)
    for summary in rollout_summaries:
        finals = np.vstack([ep.final_position for ep in summary.episodes if ep.final_position.size > 0])
        goals = np.vstack([ep.goal_position for ep in summary.episodes if ep.goal_position.size > 0])
        if finals.size > 0:
            ax.scatter(finals[:, 0], finals[:, 1], s=12, alpha=0.55, label=f"{summary.run.framework}-final")
        if goals.size > 0:
            ax.scatter(goals[:, 0], goals[:, 1], s=16, marker="+", alpha=0.55, label=f"{summary.run.framework}-goal")
    handles, labels = ax.get_legend_handles_labels()
    uniq: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        uniq.setdefault(label, handle)
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")
    ax.set_title("Terminal State Scatter vs Goal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    path = save_figure(fig, ctx.output_dir, "terminal_scatter", ctx.image_format, ctx.show)
    _record_figure(manifest, "terminal_scatter", path, sources=[str(s.run.run_dir) for s in rollout_summaries])


def plot_obstacle_clearance_profiles(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    grouped = _group_rollout_summaries_by_framework(rollout_summaries)
    if not grouped:
        fig = placeholder_figure("Obstacle Clearance Profiles", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "obstacle_clearance_profiles", ctx.image_format, ctx.show)
        _record_figure(manifest, "obstacle_clearance_profiles", path, sources=[], exploratory=True, notes="no_rollouts")
        return
    colors = _framework_colors()
    fig, ax = plt.subplots(1, 1)
    for framework, summaries in sorted(grouped.items()):
        clearance_series: List[np.ndarray] = []
        for summary in summaries:
            clearance_series.extend([ep.clearances for ep in summary.episodes if ep.clearances.size > 0])
        if not clearance_series:
            continue
        min_len = min(len(s) for s in clearance_series)
        mat = np.vstack([s[:min_len] for s in clearance_series])
        x = np.arange(1, min_len + 1, dtype=np.float64)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        color = colors.get(framework, "#444444")
        ax.plot(x, mean, color=color, linewidth=2.2, label=f"{framework}")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
    ax.axhline(0.025, color="black", linestyle="--", linewidth=1.2, label="collision_threshold")
    ax.set_title("Obstacle Clearance Profile")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to nearest obstacle")
    ax.legend(loc="best")
    path = save_figure(fig, ctx.output_dir, "obstacle_clearance_profiles", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "obstacle_clearance_profiles",
        path,
        sources=[str(s.run.run_dir) for s in rollout_summaries],
    )


def plot_control_diagnostics(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    grouped = _group_rollout_summaries_by_framework(rollout_summaries)
    if not grouped:
        fig = placeholder_figure("Control Diagnostics", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "control_diagnostics", ctx.image_format, ctx.show)
        _record_figure(manifest, "control_diagnostics", path, sources=[], exploratory=True, notes="no_rollouts")
        return

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    colors = _framework_colors()
    for framework, summaries in sorted(grouped.items()):
        action_mag: List[float] = []
        action_delta: List[float] = []
        saturation_rates: List[float] = []
        for summary in summaries:
            for ep in summary.episodes:
                if ep.actions.size > 0:
                    action_mag.extend(np.linalg.norm(ep.actions, axis=1).tolist())
                    if len(ep.actions) > 1:
                        action_delta.extend(np.linalg.norm(np.diff(ep.actions, axis=0), axis=1).tolist())
                saturation_rates.append(ep.action_saturation_rate)
        color = colors.get(framework, "#444444")
        if action_mag:
            axs[0].hist(action_mag, bins=35, alpha=0.45, color=color, label=framework)
        if action_delta:
            axs[1].hist(action_delta, bins=35, alpha=0.45, color=color, label=framework)
        if saturation_rates:
            axs[2].hist(saturation_rates, bins=30, alpha=0.45, color=color, label=framework)
    axs[0].set_title("Action Magnitude")
    axs[1].set_title("Action Delta Magnitude")
    axs[2].set_title("Saturation Rate")
    for ax in axs:
        ax.legend(loc="best")
    path = save_figure(fig, ctx.output_dir, "control_diagnostics", ctx.image_format, ctx.show)
    _record_figure(manifest, "control_diagnostics", path, sources=[str(s.run.run_dir) for s in rollout_summaries])


def plot_curvature_dynamics(
    rollout_summaries: Sequence[RolloutSummary],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    grouped = _group_rollout_summaries_by_framework(rollout_summaries)
    if not grouped:
        fig = placeholder_figure("Curvature Dynamics", "No rollout summaries available.")
        path = save_figure(fig, ctx.output_dir, "curvature_dynamics", ctx.image_format, ctx.show)
        _record_figure(manifest, "curvature_dynamics", path, sources=[], exploratory=True, notes="no_rollouts")
        return
    colors = _framework_colors()
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for framework, summaries in sorted(grouped.items()):
        kappa_series = {0: [], 1: [], 2: []}
        for summary in summaries:
            for ep in summary.episodes:
                if ep.kappas.size > 0:
                    for i in (0, 1, 2):
                        kappa_series[i].append(ep.kappas[:, i])
        for i in (0, 1, 2):
            if not kappa_series[i]:
                continue
            min_len = min(len(s) for s in kappa_series[i])
            mat = np.vstack([s[:min_len] for s in kappa_series[i]])
            x = np.arange(1, min_len + 1, dtype=np.float64)
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            color = colors.get(framework, "#444444")
            axs[i].plot(x, mean, color=color, linewidth=2.0, label=framework)
            axs[i].fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
            axs[i].set_ylabel(f"kappa{i+1}")
            axs[i].legend(loc="best")
    axs[-1].set_xlabel("Step")
    fig.suptitle("Curvature Dynamics")
    path = save_figure(fig, ctx.output_dir, "curvature_dynamics", ctx.image_format, ctx.show)
    _record_figure(manifest, "curvature_dynamics", path, sources=[str(s.run.run_dir) for s in rollout_summaries])


def plot_stability_diagnostics(
    runs: Sequence[RunRecord],
    ctx: FigureContext,
    manifest: List[dict],
) -> None:
    stability_rows: List[dict] = []
    for run in runs:
        metrics = _collect_stability_metrics(run)
        if metrics is not None:
            stability_rows.append({"run": run, "metrics": metrics})

    if not stability_rows:
        fig = placeholder_figure(
            "Stability Diagnostics",
            "No opt metric CSV found.\nExpected one of:\nmetrics/opt_metrics.csv, opt_metrics.csv, metrics.csv",
        )
        path = save_figure(fig, ctx.output_dir, "stability_diagnostics", ctx.image_format, ctx.show)
        _record_figure(
            manifest,
            "stability_diagnostics",
            path,
            sources=[str(r.run_dir) for r in runs],
            exploratory=True,
            notes="no_opt_metric_csv",
        )
        return

    colors = _framework_colors()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for row in stability_rows:
        run = row["run"]
        m = row["metrics"]
        step = np.asarray(m["step"], dtype=np.float64)
        color = colors.get(run.framework, "#444444")
        if m["actor_loss"] is not None:
            axs[0, 0].plot(step[: len(m["actor_loss"])], m["actor_loss"], color=color, alpha=0.3)
        if m["critic_loss"] is not None:
            axs[0, 1].plot(step[: len(m["critic_loss"])], m["critic_loss"], color=color, alpha=0.3)
        if m["actor_grad"] is not None:
            axs[1, 0].plot(step[: len(m["actor_grad"])], m["actor_grad"], color=color, alpha=0.3)
        if m["critic_grad"] is not None:
            axs[1, 1].plot(step[: len(m["critic_grad"])], m["critic_grad"], color=color, alpha=0.3)

    axs[0, 0].set_title("Actor Loss")
    axs[0, 1].set_title("Critic Loss")
    axs[1, 0].set_title("Actor Grad Norm")
    axs[1, 1].set_title("Critic Grad Norm")
    for ax in axs.flat:
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")

    # outlier annotations for critic loss
    critic_vals = []
    for row in stability_rows:
        c = row["metrics"]["critic_loss"]
        if c is not None:
            critic_vals.extend(np.asarray(c, dtype=np.float64).tolist())
    if critic_vals:
        arr = np.asarray(critic_vals, dtype=np.float64)
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        outlier_count = int(np.sum(arr > threshold))
        axs[0, 1].text(
            0.99,
            0.97,
            f"outliers>{threshold:.3f}: {outlier_count}",
            ha="right",
            va="top",
            transform=axs[0, 1].transAxes,
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    path = save_figure(fig, ctx.output_dir, "stability_diagnostics", ctx.image_format, ctx.show)
    _record_figure(
        manifest,
        "stability_diagnostics",
        path,
        sources=[str(row["run"].run_dir) for row in stability_rows],
    )
