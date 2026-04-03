from __future__ import annotations

from continuum_rl import cli


def test_cli_parses_pytorch_train():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "pytorch-train",
            "--episodes",
            "2",
            "--max-t",
            "3",
        ]
    )
    assert args.command == "pytorch-train"
    assert args.episodes == 2
    assert args.max_t == 3


def test_cli_parses_keras_eval():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "keras-eval-smoke",
            "--checkpoint-actor",
            "Keras/fixed_goal/reward_step_minus_euclidean_square/model/continuum_actor.h5",
        ]
    )
    assert args.command == "keras-eval-smoke"
    assert args.checkpoint_actor.endswith(".h5")


def test_cli_parses_paper_figures():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "paper-figures",
            "--runs-root",
            "runs",
            "--output-dir",
            "figures/paper/latest",
            "--rollouts-per-seed",
            "25",
        ]
    )
    assert args.command == "paper-figures"
    assert args.runs_root == "runs"
    assert args.output_dir == "figures/paper/latest"
    assert args.rollouts_per_seed == 25


def test_cli_parses_gradio_demo():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "gradio-demo",
            "--framework",
            "keras",
            "--control-mode",
            "manual",
            "--max-steps",
            "123",
            "--initial-kappa",
            "1.0,2.0,3.0",
        ]
    )
    assert args.command == "gradio-demo"
    assert args.framework == "keras"
    assert args.control_mode == "manual"
    assert args.max_steps == 123
    assert args.initial_kappa == "1.0,2.0,3.0"


def test_legacy_cli_routes_to_hydra(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _fake_run_with_overrides(overrides: list[str]) -> None:
        captured["overrides"] = overrides

    monkeypatch.setattr(cli, "run_with_overrides", _fake_run_with_overrides)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "pytorch-train",
            "--episodes",
            "5",
            "--max-t",
            "6",
            "--goal-type",
            "random_goal",
            "--seed",
            "123",
            "--deterministic",
        ],
    )

    cli.main()
    overrides = captured["overrides"]
    assert "task=pytorch_train" in overrides
    assert "task.episodes=5" in overrides
    assert "task.max_t=6" in overrides
    assert "task.goal_type=random_goal" in overrides
    assert "task.seed=123" in overrides
    assert "task.deterministic=true" in overrides
    assert not any(item.startswith("task.reward_file=") for item in overrides)


def test_legacy_cli_wandb_routes_to_hydra(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _fake_run_with_overrides(overrides: list[str]) -> None:
        captured["overrides"] = overrides

    monkeypatch.setattr(cli, "run_with_overrides", _fake_run_with_overrides)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "keras-train",
            "--episodes",
            "2",
            "--max-steps",
            "3",
            "--wandb-enabled",
            "--wandb-mode",
            "offline",
            "--wandb-project",
            "demo",
            "--wandb-eval-interval-episodes",
            "7",
            "--wandb-artifact-interval-episodes",
            "9",
            "--wandb-no-upload-checkpoints",
        ],
    )

    cli.main()
    overrides = captured["overrides"]
    assert "task=keras_train" in overrides
    assert "wandb.enabled=true" in overrides
    assert "wandb.mode=offline" in overrides
    assert "wandb.project=demo" in overrides
    assert "wandb.eval_interval_episodes=7" in overrides
    assert "wandb.artifact_interval_episodes=9" in overrides
    assert "wandb.upload_checkpoints=false" in overrides


def test_legacy_cli_explicit_reward_file_override(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _fake_run_with_overrides(overrides: list[str]) -> None:
        captured["overrides"] = overrides

    monkeypatch.setattr(cli, "run_with_overrides", _fake_run_with_overrides)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "pytorch-train",
            "--reward-function",
            "step_distance_based",
            "--reward-file",
            "my_reward_dir",
        ],
    )

    cli.main()
    overrides = captured["overrides"]
    assert "task.reward_function=step_distance_based" in overrides
    assert "task.reward_file=my_reward_dir" in overrides


def test_legacy_cli_paper_figures_routes_to_hydra(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _fake_run_with_overrides(overrides: list[str]) -> None:
        captured["overrides"] = overrides

    monkeypatch.setattr(cli, "run_with_overrides", _fake_run_with_overrides)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "paper-figures",
            "--runs-root",
            "runs",
            "--output-dir",
            "figures/paper/latest",
            "--goal-types",
            "fixed_goal,random_goal",
            "--show",
        ],
    )

    cli.main()
    overrides = captured["overrides"]
    assert "task=paper_figures" in overrides
    assert "task.runs_root=runs" in overrides
    assert "task.output_dir=figures/paper/latest" in overrides
    assert "task.include_goal_types=[fixed_goal,random_goal]" in overrides
    assert "task.show=true" in overrides


def test_legacy_cli_gradio_demo_routes_to_hydra(monkeypatch):
    captured: dict[str, list[str]] = {}

    def _fake_run_with_overrides(overrides: list[str]) -> None:
        captured["overrides"] = overrides

    monkeypatch.setattr(cli, "run_with_overrides", _fake_run_with_overrides)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "gradio-demo",
            "--framework",
            "pytorch",
            "--control-mode",
            "manual",
            "--goal-type",
            "fixed_goal",
            "--initial-kappa",
            "1.0,2.0,3.0",
            "--manual-action",
            "0.1,0.0,-0.1",
            "--env-dt",
            "0.04",
            "--no-share",
        ],
    )

    cli.main()
    overrides = captured["overrides"]
    assert "task=gradio_demo" in overrides
    assert "task.framework=pytorch" in overrides
    assert "task.control_mode=manual" in overrides
    assert "task.initial_kappa=[1.0,2.0,3.0]" in overrides
    assert "task.manual_action=[0.1,0.0,-0.1]" in overrides
    assert "env.dt=0.04" in overrides
    assert "task.share=false" in overrides
