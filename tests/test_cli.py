from __future__ import annotations

from continuum_rl.cli import build_parser


def test_cli_parses_pytorch_train():
    parser = build_parser()
    args = parser.parse_args(
        [
            "pytorch-train",
            "--episodes",
            "2",
            "--max-t",
            "3",
            "--observation-mode",
            "legacy4d",
        ]
    )
    assert args.command == "pytorch-train"
    assert args.episodes == 2
    assert args.max_t == 3
    assert args.observation_mode == "legacy4d"


def test_cli_parses_keras_eval():
    parser = build_parser()
    args = parser.parse_args(
        [
            "keras-eval-smoke",
            "--checkpoint-actor",
            "Keras/fixed_goal/reward_step_minus_euclidean_square/model/continuum_actor.h5",
        ]
    )
    assert args.command == "keras-eval-smoke"
    assert args.checkpoint_actor.endswith(".h5")
