from __future__ import annotations

import threading

import numpy as np
import pytest

from continuum_rl.gradio_demo import (
    GradioRuntimeConfig,
    _normalize_obstacles,
    persist_result,
    run_simulation,
)


def _manual_cfg() -> GradioRuntimeConfig:
    return GradioRuntimeConfig(
        framework="pytorch",
        control_mode="manual",
        goal_type="fixed_goal",
        reward_function="step_minus_weighted_euclidean",
        checkpoint_actor="does_not_matter_for_manual_mode.pth",
        checkpoint_critic=None,
        device="cpu",
        max_steps=10,
        seed=3,
        initial_kappa=(1.0, 2.0, 3.0),
        fixed_goal_xy=(-0.2, 0.15),
        manual_action=(0.0, 0.0, 0.0),
        output_dir="visualizations/gradio",
        save_outputs=False,
        save_animation=False,
        animation_format="gif",
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        single_run_lock=True,
        show_progress=False,
        env_dt=0.05,
        env_delta_kappa=0.001,
        env_l=(0.1, 0.1, 0.1),
        env_obstacles=(
            {"x": -0.16, "y": 0.22},
            {"x": -0.22, "y": 0.02},
            {"x": -0.16, "y": 0.08},
        ),
    )


def test_manual_mode_runs_without_checkpoint():
    cfg = _manual_cfg()
    result = run_simulation(cfg)
    assert result.control_mode == "manual"
    assert result.steps <= cfg.max_steps
    assert result.positions.shape[1] == 2
    assert result.kappas.shape[1] == 3


def test_initial_kappa_override_applies_to_first_state():
    cfg = _manual_cfg()
    init_override = (4.0, 5.0, 6.0)
    result = run_simulation(cfg, initial_kappa_override=init_override)
    assert np.allclose(result.kappas[0], np.asarray(init_override, dtype=np.float64), atol=1e-8)
    assert result.initial_kappa == init_override


def test_invalid_kappa_raises():
    cfg = _manual_cfg()
    with pytest.raises(ValueError):
        run_simulation(cfg, initial_kappa_override=(50.0, 0.0, 0.0))


def test_cancel_event_stops_rollout_early():
    cfg = _manual_cfg()
    cancel = threading.Event()
    cancel.set()
    result = run_simulation(cfg, cancel_event=cancel)
    assert result.cancelled is True
    assert result.steps == 0


def test_persist_result_writes_expected_files(tmp_path):
    cfg = _manual_cfg()
    result = run_simulation(cfg, initial_kappa_override=(1.0, 2.0, 3.0))
    saved = persist_result(result, output_dir=tmp_path, save_animation=False, animation_format="gif")
    assert saved.run_dir is not None
    assert saved.workspace_fig_path is not None
    assert saved.diagnostics_fig_path is not None


def test_normalize_obstacles_accepts_dict_of_columns():
    obstacles = _normalize_obstacles({"x": [-0.1, -0.2], "y": [0.1, 0.2]})
    assert len(obstacles) == 2
    assert obstacles[0]["x"] == -0.1
    assert obstacles[1]["y"] == 0.2


def test_normalize_obstacles_accepts_numpy_table():
    raw = np.array([[-0.1, 0.1], [-0.2, 0.2]], dtype=np.float64)
    obstacles = _normalize_obstacles(raw)
    assert len(obstacles) == 2
    assert obstacles[0]["x"] == -0.1
    assert obstacles[1]["y"] == 0.2
