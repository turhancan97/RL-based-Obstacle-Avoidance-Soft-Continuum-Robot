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
        ]
    )
    assert cfg.task.goal_type == "random_goal"
    assert cfg.task.episodes == 3
    assert cfg.task.max_t == 4
    assert cfg.task.output_base_dir == "/tmp/pytorch_out"


def test_observation_mode_is_canonical_only():
    with pytest.raises(Exception):
        compose_config(["observation_mode=legacy4d"])
