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
    "paper_figures",
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
        assert isinstance(cfg.wandb.enabled, bool)
        assert cfg.wandb.mode in {"offline", "online"}
        if task in {"pytorch_train", "keras_train"}:
            assert cfg.task.reward_file == f"reward_{cfg.task.reward_function}"
            assert cfg.task.gamma == 0.99
            assert cfg.task.tau == 0.0005
        if task == "paper_figures":
            assert cfg.task.format == "jpeg"
            assert cfg.task.ci_method == "bootstrap"
            assert cfg.task.show is False


def test_hydra_paper_figures_override_smoke():
    cfg = compose_config(
        [
            "task=paper_figures",
            "task.runs_root=/tmp/runs",
            "task.output_dir=/tmp/figures",
            "task.rollouts_per_seed=20",
            "task.include_goal_types=[fixed_goal]",
            "task.bootstrap_samples=500",
        ]
    )
    assert cfg.task_name == "paper_figures"
    assert cfg.task.runs_root == "/tmp/runs"
    assert cfg.task.output_dir == "/tmp/figures"
    assert cfg.task.rollouts_per_seed == 20
    assert list(cfg.task.include_goal_types) == ["fixed_goal"]
    assert cfg.task.bootstrap_samples == 500


def test_hydra_paper_figures_invalid_format_rejected():
    with pytest.raises(Exception):
        compose_config(["task=paper_figures", "task.format=png"])


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
    assert cfg.task.batch_size == 64


def test_hydra_train_hyperparameter_overrides_smoke():
    cfg = compose_config(
        [
            "task=keras_train",
            "task.batch_size=128",
            "task.gamma=0.95",
            "task.tau=0.005",
            "task.actor_lr=0.0002",
            "task.critic_lr=0.0008",
            "task.noise_std=0.25",
        ]
    )
    assert cfg.task.batch_size == 128
    assert cfg.task.gamma == 0.95
    assert cfg.task.tau == 0.005
    assert cfg.task.actor_lr == 0.0002
    assert cfg.task.critic_lr == 0.0008
    assert cfg.task.noise_std == 0.25


def test_hydra_wandb_overrides_smoke():
    cfg = compose_config(
        [
            "task=keras_train",
            "wandb.enabled=true",
            "wandb.mode=online",
            "wandb.project=test-project",
            "wandb.eval_interval_episodes=5",
            "wandb.artifact_interval_episodes=10",
            "wandb.upload_checkpoints=false",
        ]
    )
    assert cfg.wandb.enabled is True
    assert cfg.wandb.mode == "online"
    assert cfg.wandb.project == "test-project"
    assert cfg.wandb.eval_interval_episodes == 5
    assert cfg.wandb.artifact_interval_episodes == 10
    assert cfg.wandb.upload_checkpoints is False


def test_hydra_reward_file_auto_derives_from_reward_function():
    cfg = compose_config(
        [
            "task=pytorch_train",
            "task.reward_function=step_distance_based",
        ]
    )
    assert cfg.task.reward_file == "reward_step_distance_based"


def test_hydra_reward_file_explicit_override_is_preserved():
    cfg = compose_config(
        [
            "task=keras_train",
            "task.reward_function=step_distance_based",
            "task.reward_file=my_custom_reward_dir",
        ]
    )
    assert cfg.task.reward_file == "my_custom_reward_dir"


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


def test_wandb_invalid_mode_rejected():
    with pytest.raises(Exception):
        compose_config(["wandb.mode=disabled"])
