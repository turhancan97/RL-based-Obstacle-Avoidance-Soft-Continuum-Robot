"""Unified CLI for continuum RL train/eval/visualization flows."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuum RL unified runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pytorch_train = subparsers.add_parser("pytorch-train", help="Train PyTorch DDPG.")
    pytorch_train.add_argument("--episodes", type=int, default=300)
    pytorch_train.add_argument("--max-t", type=int, default=750)
    pytorch_train.add_argument("--print-every", type=int, default=25)
    pytorch_train.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_train.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    pytorch_train.add_argument("--reward-file", default="reward_step_minus_weighted_euclidean")

    pytorch_eval = subparsers.add_parser("pytorch-eval-smoke", help="Run PyTorch smoke evaluation.")
    pytorch_eval.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_eval.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    pytorch_eval.add_argument("--max-t", type=int, default=20)
    pytorch_eval.add_argument("--checkpoint-actor", required=True)
    pytorch_eval.add_argument("--checkpoint-critic", required=True)

    keras_train = subparsers.add_parser("keras-train", help="Train Keras DDPG.")
    keras_train.add_argument("--episodes", type=int, default=500)
    keras_train.add_argument("--max-steps", type=int, default=500)
    keras_train.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_train.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    keras_train.add_argument("--reward-file", default="reward_step_minus_weighted_euclidean")

    keras_eval = subparsers.add_parser("keras-eval-smoke", help="Run Keras smoke evaluation.")
    keras_eval.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_eval.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    keras_eval.add_argument("--max-steps", type=int, default=20)
    keras_eval.add_argument("--checkpoint-actor", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pytorch-train":
        from Pytorch.ddpg import train

        train(
            n_episodes=args.episodes,
            max_t=args.max_t,
            print_every=args.print_every,
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            reward_file=args.reward_file,
        )
        return

    if args.command == "pytorch-eval-smoke":
        from pathlib import Path
        from Pytorch.ddpg import evaluate_smoke

        evaluate_smoke(
            checkpoint_actor=Path(args.checkpoint_actor),
            checkpoint_critic=Path(args.checkpoint_critic),
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            max_t=args.max_t,
        )
        return

    if args.command == "keras-train":
        from Keras.DDPG import train

        train(
            total_episodes=args.episodes,
            max_steps=args.max_steps,
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            reward_file=args.reward_file,
        )
        return

    if args.command == "keras-eval-smoke":
        from pathlib import Path
        from Keras.DDPG import evaluate_smoke

        evaluate_smoke(
            checkpoint_actor=Path(args.checkpoint_actor),
            goal_type=args.goal_type,
            reward_function=args.reward_function,
            max_steps=args.max_steps,
        )
        return

    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
