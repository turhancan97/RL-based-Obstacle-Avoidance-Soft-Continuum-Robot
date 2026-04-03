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
