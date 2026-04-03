from __future__ import annotations

import pytest

from continuum_rl.hydra_app import compose_config


TASKS = [
    "pytorch_train",
    "pytorch_eval_smoke",
    "keras_train",
    "keras_eval_smoke",
    "pytorch_reward_vis",
    "keras_reward_vis",
]


def test_hydra_compose_all_tasks():
    for task in TASKS:
        cfg = compose_config([f"task={task}"])
        assert cfg.task_name == task
        assert cfg.observation_mode == "canonical"
        assert cfg.env.delta_kappa == 0.001
        assert list(cfg.env.l) == [0.1, 0.1, 0.1]
        assert cfg.env.dt == 5e-2
        assert len(cfg.env.obstacles) == 3


def test_hydra_unknown_key_rejected():
    with pytest.raises(Exception):
        compose_config(["unknown_key=1"])


def test_hydra_override_smoke():
    cfg = compose_config(
        [
            "task=pytorch_train",
            "task.goal_type=random_goal",
            "task.episodes=3",
            "task.max_t=4",
            "task.output_base_dir=/tmp/pytorch_out",
            "task.seed=42",
            "task.deterministic=true",
        ]
    )
    assert cfg.task.goal_type == "random_goal"
    assert cfg.task.episodes == 3
    assert cfg.task.max_t == 4
    assert cfg.task.output_base_dir == "/tmp/pytorch_out"
    assert cfg.task.seed == 42
    assert cfg.task.deterministic is True


def test_hydra_env_overrides_smoke():
    cfg = compose_config(
        [
            "task=keras_eval_smoke",
            "env.delta_kappa=0.002",
            "env.dt=0.01",
            "env.l=[0.12,0.11,0.09]",
            "env.obstacles=[{x:-0.1,y:0.2},{x:-0.2,y:0.1}]",
        ]
    )
    assert cfg.env.delta_kappa == 0.002
    assert cfg.env.dt == 0.01
    assert list(cfg.env.l) == [0.12, 0.11, 0.09]
    assert len(cfg.env.obstacles) == 2
    assert cfg.env.obstacles[0].x == -0.1
    assert cfg.env.obstacles[0].y == 0.2


def test_observation_mode_is_canonical_only():
    with pytest.raises(Exception):
        compose_config(["observation_mode=legacy4d"])
