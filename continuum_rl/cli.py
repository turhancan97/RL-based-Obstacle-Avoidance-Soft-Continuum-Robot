"""Compatibility CLI wrapper that forwards legacy commands to Hydra."""

from __future__ import annotations

import argparse
import warnings

from .hydra_app import run_with_overrides


DEPRECATION_WINDOW = "next release milestone"


def _emit_deprecation(replacement: str) -> None:
    message = (
        "DEPRECATION: legacy CLI compatibility mode will be removed in the "
        f"{DEPRECATION_WINDOW}. Use `{replacement}` instead."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    print(message)


def _to_hydra_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy continuum RL command wrapper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pytorch_train = subparsers.add_parser("pytorch-train", help="Train PyTorch DDPG.")
    pytorch_train.add_argument("--episodes", type=int, default=300)
    pytorch_train.add_argument("--max-t", type=int, default=750)
    pytorch_train.add_argument("--print-every", type=int, default=25)
    pytorch_train.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_train.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    pytorch_train.add_argument("--reward-file", default="reward_step_minus_weighted_euclidean")
    pytorch_train.add_argument("--output-base-dir", default="Pytorch")
    pytorch_train.add_argument("--seed", type=int, default=None)
    pytorch_train.add_argument("--deterministic", action="store_true")

    pytorch_eval = subparsers.add_parser("pytorch-eval-smoke", help="Run PyTorch smoke evaluation.")
    pytorch_eval.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_eval.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    pytorch_eval.add_argument("--max-t", type=int, default=20)
    pytorch_eval.add_argument("--checkpoint-actor", required=True)
    pytorch_eval.add_argument("--checkpoint-critic", required=True)
    pytorch_eval.add_argument("--seed", type=int, default=None)
    pytorch_eval.add_argument("--deterministic", action="store_true")

    keras_train = subparsers.add_parser("keras-train", help="Train Keras DDPG.")
    keras_train.add_argument("--episodes", type=int, default=500)
    keras_train.add_argument("--max-steps", type=int, default=500)
    keras_train.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_train.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    keras_train.add_argument("--reward-file", default="reward_step_minus_weighted_euclidean")
    keras_train.add_argument("--output-base-dir", default="Keras")
    keras_train.add_argument("--seed", type=int, default=None)
    keras_train.add_argument("--deterministic", action="store_true")

    keras_eval = subparsers.add_parser("keras-eval-smoke", help="Run Keras smoke evaluation.")
    keras_eval.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_eval.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    keras_eval.add_argument("--max-steps", type=int, default=20)
    keras_eval.add_argument("--checkpoint-actor", required=True)
    keras_eval.add_argument("--seed", type=int, default=None)
    keras_eval.add_argument("--deterministic", action="store_true")

    pytorch_vis = subparsers.add_parser("pytorch-reward-vis", help="Generate PyTorch reward visualizations.")
    pytorch_vis.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_vis.add_argument("--reward-type", default="reward_step_minus_weighted_euclidean")
    pytorch_vis.add_argument("--base-dir", default="Pytorch")

    keras_vis = subparsers.add_parser("keras-reward-vis", help="Generate Keras reward visualizations.")
    keras_vis.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_vis.add_argument("--reward-type", default="reward_step_minus_weighted_euclidean")
    keras_vis.add_argument("--base-dir", default="Keras")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pytorch-train":
        replacement = (
            "continuum-rl task=pytorch_train "
            f"task.episodes={args.episodes} "
            f"task.max_t={args.max_t}"
        )
        _emit_deprecation(replacement)
        run_with_overrides(
            [
                "task=pytorch_train",
                f"task.episodes={args.episodes}",
                f"task.max_t={args.max_t}",
                f"task.print_every={args.print_every}",
                f"task.goal_type={args.goal_type}",
                f"task.reward_function={args.reward_function}",
                f"task.reward_file={args.reward_file}",
                f"task.output_base_dir={args.output_base_dir}",
                f"task.seed={_to_hydra_scalar(args.seed)}",
                f"task.deterministic={_to_hydra_scalar(args.deterministic)}",
            ]
        )
        return

    if args.command == "pytorch-eval-smoke":
        replacement = "continuum-rl task=pytorch_eval_smoke"
        _emit_deprecation(replacement)
        run_with_overrides(
            [
                "task=pytorch_eval_smoke",
                f"task.goal_type={args.goal_type}",
                f"task.reward_function={args.reward_function}",
                f"task.max_t={args.max_t}",
                f"task.checkpoint_actor={args.checkpoint_actor}",
                f"task.checkpoint_critic={args.checkpoint_critic}",
                f"task.seed={_to_hydra_scalar(args.seed)}",
                f"task.deterministic={_to_hydra_scalar(args.deterministic)}",
            ]
        )
        return

    if args.command == "keras-train":
        replacement = (
            "continuum-rl task=keras_train "
            f"task.episodes={args.episodes} "
            f"task.max_steps={args.max_steps}"
        )
        _emit_deprecation(replacement)
        run_with_overrides(
            [
                "task=keras_train",
                f"task.episodes={args.episodes}",
                f"task.max_steps={args.max_steps}",
                f"task.goal_type={args.goal_type}",
                f"task.reward_function={args.reward_function}",
                f"task.reward_file={args.reward_file}",
                f"task.output_base_dir={args.output_base_dir}",
                f"task.seed={_to_hydra_scalar(args.seed)}",
                f"task.deterministic={_to_hydra_scalar(args.deterministic)}",
            ]
        )
        return

    if args.command == "keras-eval-smoke":
        replacement = "continuum-rl task=keras_eval_smoke"
        _emit_deprecation(replacement)
        run_with_overrides(
            [
                "task=keras_eval_smoke",
                f"task.goal_type={args.goal_type}",
                f"task.reward_function={args.reward_function}",
                f"task.max_steps={args.max_steps}",
                f"task.checkpoint_actor={args.checkpoint_actor}",
                f"task.seed={_to_hydra_scalar(args.seed)}",
                f"task.deterministic={_to_hydra_scalar(args.deterministic)}",
            ]
        )
        return

    if args.command == "pytorch-reward-vis":
        replacement = "continuum-rl task=pytorch_reward_vis"
        _emit_deprecation(replacement)
        run_with_overrides(
            [
                "task=pytorch_reward_vis",
                f"task.goal_type={args.goal_type}",
                f"task.reward_type={args.reward_type}",
                f"task.base_dir={args.base_dir}",
            ]
        )
        return

    if args.command == "keras-reward-vis":
        replacement = "continuum-rl task=keras_reward_vis"
        _emit_deprecation(replacement)
        run_with_overrides(
            [
                "task=keras_reward_vis",
                f"task.goal_type={args.goal_type}",
                f"task.reward_type={args.reward_type}",
                f"task.base_dir={args.base_dir}",
            ]
        )
        return

    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
