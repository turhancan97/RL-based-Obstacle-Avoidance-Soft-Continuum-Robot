from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from continuum_rl.visualization.data import RunRecord, discover_runs
from continuum_rl.visualization.rollouts import RolloutEpisode, RolloutSummary
from continuum_rl.visualization.stats import bootstrap_ci_mean


def _write_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def _create_run_tree(root: Path, framework: str, seed: int) -> Path:
    run_dir = root / framework / "fixed_goal" / "reward_step_minus_weighted_euclidean" / f"seed_{seed}"
    rewards_dir = run_dir / "rewards"
    model_dir = run_dir / "model"
    rewards_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    series = np.linspace(-10.0, 5.0, 20).tolist()
    if framework == "pytorch":
        _write_pickle(rewards_dir / "scores.pickle", series)
        _write_pickle(rewards_dir / "avg_reward_list.pickle", np.linspace(-12.0, 4.0, 20).tolist())
        (model_dir / "checkpoint_actor.pth").write_bytes(b"pt-actor")
        (model_dir / "checkpoint_critic.pth").write_bytes(b"pt-critic")
    else:
        _write_pickle(rewards_dir / "ep_reward_list.pickle", series)
        _write_pickle(rewards_dir / "avg_reward_list.pickle", np.linspace(-11.0, 3.0, 20).tolist())
        (model_dir / "continuum_actor.weights.h5").write_bytes(b"keras-actor")
    return run_dir


def _synthetic_rollout_summary(run: RunRecord) -> RolloutSummary:
    episodes = []
    for idx in range(3):
        positions = np.stack([np.linspace(0.0, 0.2, 10), np.linspace(0.0, 0.1, 10)], axis=1)
        actions = np.ones((9, 3), dtype=np.float64) * (0.1 + idx * 0.01)
        kappas = np.ones((10, 3), dtype=np.float64) * (idx + 1)
        rewards = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
        clearances = np.linspace(0.08, 0.03, 10, dtype=np.float64)
        episodes.append(
            RolloutEpisode(
                positions=positions,
                actions=actions,
                kappas=kappas,
                rewards=rewards,
                clearances=clearances,
                terminated=True,
                truncated=False,
                total_return=float(np.sum(rewards)),
                length=len(rewards),
                min_clearance=float(np.min(clearances)),
                final_position=positions[-1],
                goal_position=np.array([0.2, 0.1], dtype=np.float64),
                action_saturation_rate=0.0,
                action_smoothness=0.0,
            )
        )
    obstacles = np.array([[-0.16, 0.22], [-0.22, 0.02], [-0.16, 0.08]], dtype=np.float64)
    return RolloutSummary(run=run, episodes=episodes, obstacles=obstacles)


def test_discover_runs_strict_layout_and_missing_files(tmp_path: Path):
    _create_run_tree(tmp_path, framework="pytorch", seed=1)
    broken = tmp_path / "keras" / "fixed_goal" / "reward_step_minus_weighted_euclidean" / "seed_2"
    (broken / "rewards").mkdir(parents=True, exist_ok=True)
    # Intentionally leave out required model file for keras.
    _write_pickle(broken / "rewards" / "ep_reward_list.pickle", [1, 2, 3])
    _write_pickle(broken / "rewards" / "avg_reward_list.pickle", [1, 2, 3])

    runs, skipped = discover_runs(tmp_path, include_goal_types=("fixed_goal", "random_goal"))
    assert len(runs) == 1
    assert runs[0].framework == "pytorch"
    assert any("missing_required_files" in item.reason for item in skipped)


def test_bootstrap_ci_mean_shape():
    arr = np.vstack([np.linspace(0, 1, 20), np.linspace(0.2, 1.2, 20), np.linspace(-0.1, 0.9, 20)])
    mean, low, high = bootstrap_ci_mean(arr, ci_level=0.95, n_bootstrap=200, seed=7)
    assert mean.shape == (20,)
    assert low.shape == (20,)
    assert high.shape == (20,)
    assert np.all(low <= mean)
    assert np.all(mean <= high)


def test_run_paper_figures_end_to_end_with_mocked_rollouts(tmp_path: Path, monkeypatch):
    from continuum_rl.visualization import pipeline

    _create_run_tree(tmp_path / "runs", framework="pytorch", seed=1)
    _create_run_tree(tmp_path / "runs", framework="keras", seed=1)

    def _fake_eval(**kwargs):
        run = kwargs["run"]
        return _synthetic_rollout_summary(run)

    monkeypatch.setattr(pipeline, "evaluate_run_rollouts", _fake_eval)

    out_dir = pipeline.run_paper_figures(
        runs_root=tmp_path / "runs",
        output_dir=tmp_path / "figures" / "paper" / "latest",
        image_format="jpeg",
        show=False,
        min_seeds_for_claims=1,
        ci_method="bootstrap",
        ci_level=0.95,
        bootstrap_samples=100,
        bootstrap_seed=11,
        rollouts_per_seed=3,
        include_goal_types=("fixed_goal",),
        max_steps=50,
        reward_function="step_minus_weighted_euclidean",
        clear_output_dir=True,
        env_kwargs=None,
    )
    expected_files = [
        "learning_curves.jpeg",
        "sample_efficiency.jpeg",
        "trajectory_atlas.jpeg",
        "framework_comparison.jpeg",
        "stability_diagnostics.jpeg",
        "manifest.json",
    ]
    for filename in expected_files:
        assert (out_dir / filename).exists(), filename

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "figures" in manifest
    assert len(manifest["figures"]) >= 10
    assert len(manifest["source_runs"]) == 2
