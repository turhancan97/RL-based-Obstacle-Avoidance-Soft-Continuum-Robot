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


def _add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true")
    parser.add_argument("--wandb-disabled", dest="wandb_enabled", action="store_false")
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name-template", default=None)
    parser.add_argument("--wandb-group-by-experiment", dest="wandb_group_by_experiment", action="store_true")
    parser.add_argument("--wandb-no-group-by-experiment", dest="wandb_group_by_experiment", action="store_false")
    parser.set_defaults(wandb_group_by_experiment=None)
    parser.add_argument("--wandb-log-system-metrics", dest="wandb_log_system_metrics", action="store_true")
    parser.add_argument("--wandb-no-log-system-metrics", dest="wandb_log_system_metrics", action="store_false")
    parser.set_defaults(wandb_log_system_metrics=None)
    parser.add_argument("--wandb-eval-interval-episodes", type=int, default=None)
    parser.add_argument("--wandb-artifact-interval-episodes", type=int, default=None)
    parser.add_argument("--wandb-upload-checkpoints", dest="wandb_upload_checkpoints", action="store_true")
    parser.add_argument("--wandb-no-upload-checkpoints", dest="wandb_upload_checkpoints", action="store_false")
    parser.set_defaults(wandb_upload_checkpoints=None)
    parser.add_argument("--wandb-fail-open-on-init-error", dest="wandb_fail_open_on_init_error", action="store_true")
    parser.add_argument(
        "--wandb-no-fail-open-on-init-error",
        dest="wandb_fail_open_on_init_error",
        action="store_false",
    )
    parser.set_defaults(wandb_fail_open_on_init_error=None)


def _wandb_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    mapping = [
        ("wandb_enabled", "wandb.enabled"),
        ("wandb_mode", "wandb.mode"),
        ("wandb_project", "wandb.project"),
        ("wandb_entity", "wandb.entity"),
        ("wandb_run_name_template", "wandb.run_name_template"),
        ("wandb_group_by_experiment", "wandb.group_by_experiment"),
        ("wandb_log_system_metrics", "wandb.log_system_metrics"),
        ("wandb_eval_interval_episodes", "wandb.eval_interval_episodes"),
        ("wandb_artifact_interval_episodes", "wandb.artifact_interval_episodes"),
        ("wandb_upload_checkpoints", "wandb.upload_checkpoints"),
        ("wandb_fail_open_on_init_error", "wandb.fail_open_on_init_error"),
    ]
    for attr_name, hydra_key in mapping:
        value = getattr(args, attr_name, None)
        if value is not None:
            overrides.append(f"{hydra_key}={_to_hydra_scalar(value)}")
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy continuum RL command wrapper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pytorch_train = subparsers.add_parser("pytorch-train", help="Train PyTorch DDPG.")
    pytorch_train.add_argument("--episodes", type=int, default=300)
    pytorch_train.add_argument("--max-t", type=int, default=750)
    pytorch_train.add_argument("--print-every", type=int, default=25)
    pytorch_train.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_train.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    pytorch_train.add_argument("--reward-file", default=None)
    pytorch_train.add_argument("--output-base-dir", default="Pytorch")
    pytorch_train.add_argument("--seed", type=int, default=None)
    pytorch_train.add_argument("--deterministic", action="store_true")
    _add_wandb_args(pytorch_train)

    pytorch_eval = subparsers.add_parser("pytorch-eval-smoke", help="Run PyTorch smoke evaluation.")
    pytorch_eval.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_eval.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    pytorch_eval.add_argument("--max-t", type=int, default=20)
    pytorch_eval.add_argument("--checkpoint-actor", required=True)
    pytorch_eval.add_argument("--checkpoint-critic", required=True)
    pytorch_eval.add_argument("--seed", type=int, default=None)
    pytorch_eval.add_argument("--deterministic", action="store_true")
    _add_wandb_args(pytorch_eval)

    keras_train = subparsers.add_parser("keras-train", help="Train Keras DDPG.")
    keras_train.add_argument("--episodes", type=int, default=500)
    keras_train.add_argument("--max-steps", type=int, default=500)
    keras_train.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_train.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    keras_train.add_argument("--reward-file", default=None)
    keras_train.add_argument("--output-base-dir", default="Keras")
    keras_train.add_argument("--seed", type=int, default=None)
    keras_train.add_argument("--deterministic", action="store_true")
    _add_wandb_args(keras_train)

    keras_eval = subparsers.add_parser("keras-eval-smoke", help="Run Keras smoke evaluation.")
    keras_eval.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_eval.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    keras_eval.add_argument("--max-steps", type=int, default=20)
    keras_eval.add_argument("--checkpoint-actor", required=True)
    keras_eval.add_argument("--seed", type=int, default=None)
    keras_eval.add_argument("--deterministic", action="store_true")
    _add_wandb_args(keras_eval)

    pytorch_vis = subparsers.add_parser("pytorch-reward-vis", help="Generate PyTorch reward visualizations.")
    pytorch_vis.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    pytorch_vis.add_argument("--reward-type", default="reward_step_minus_weighted_euclidean")
    pytorch_vis.add_argument("--base-dir", default="Pytorch")

    keras_vis = subparsers.add_parser("keras-reward-vis", help="Generate Keras reward visualizations.")
    keras_vis.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    keras_vis.add_argument("--reward-type", default="reward_step_minus_weighted_euclidean")
    keras_vis.add_argument("--base-dir", default="Keras")

    gradio_demo = subparsers.add_parser("gradio-demo", help="Launch interactive Gradio robot demo.")
    gradio_demo.add_argument("--framework", choices=["pytorch", "keras"], default="pytorch")
    gradio_demo.add_argument("--control-mode", choices=["policy", "manual"], default="policy")
    gradio_demo.add_argument("--goal-type", choices=["fixed_goal", "random_goal"], default="fixed_goal")
    gradio_demo.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    gradio_demo.add_argument("--checkpoint-actor", default=None)
    gradio_demo.add_argument("--checkpoint-critic", default=None)
    gradio_demo.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    gradio_demo.add_argument("--max-steps", type=int, default=300)
    gradio_demo.add_argument("--seed", type=int, default=0)
    gradio_demo.add_argument("--initial-kappa", default="0.0,0.0,0.0")
    gradio_demo.add_argument("--manual-action", default="0.0,0.0,0.0")
    gradio_demo.add_argument("--fixed-goal-xy", default="-0.2,0.15")
    gradio_demo.add_argument("--output-dir", default="visualizations/gradio")
    gradio_demo.add_argument("--save-outputs", dest="save_outputs", action="store_true")
    gradio_demo.add_argument("--no-save-outputs", dest="save_outputs", action="store_false")
    gradio_demo.set_defaults(save_outputs=True)
    gradio_demo.add_argument("--save-animation", dest="save_animation", action="store_true")
    gradio_demo.add_argument("--no-save-animation", dest="save_animation", action="store_false")
    gradio_demo.set_defaults(save_animation=True)
    gradio_demo.add_argument("--animation-format", choices=["gif", "mp4"], default="gif")
    gradio_demo.add_argument("--share", dest="share", action="store_true")
    gradio_demo.add_argument("--no-share", dest="share", action="store_false")
    gradio_demo.set_defaults(share=False)
    gradio_demo.add_argument("--server-name", default="127.0.0.1")
    gradio_demo.add_argument("--server-port", type=int, default=7860)
    gradio_demo.add_argument("--single-run-lock", dest="single_run_lock", action="store_true")
    gradio_demo.add_argument("--no-single-run-lock", dest="single_run_lock", action="store_false")
    gradio_demo.set_defaults(single_run_lock=True)
    gradio_demo.add_argument("--show-progress", dest="show_progress", action="store_true")
    gradio_demo.add_argument("--no-show-progress", dest="show_progress", action="store_false")
    gradio_demo.set_defaults(show_progress=True)
    gradio_demo.add_argument("--env-dt", type=float, default=None)
    gradio_demo.add_argument("--env-delta-kappa", type=float, default=None)
    gradio_demo.add_argument("--env-l", default=None)
    gradio_demo.add_argument("--env-obstacles", default=None)

    paper_figs = subparsers.add_parser("paper-figures", help="Generate conference-grade paper figures.")
    paper_figs.add_argument("--runs-root", default="runs")
    paper_figs.add_argument("--output-dir", default="figures/paper/latest")
    paper_figs.add_argument("--rollouts-per-seed", type=int, default=100)
    paper_figs.add_argument("--max-steps", type=int, default=750)
    paper_figs.add_argument("--goal-types", default="fixed_goal,random_goal")
    paper_figs.add_argument("--reward-function", default="step_minus_weighted_euclidean")
    paper_figs.add_argument("--show", action="store_true")

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
            (
                [
                "task=pytorch_train",
                f"task.episodes={args.episodes}",
                f"task.max_t={args.max_t}",
                f"task.print_every={args.print_every}",
                f"task.goal_type={args.goal_type}",
                f"task.reward_function={args.reward_function}",
                f"task.output_base_dir={args.output_base_dir}",
                f"task.seed={_to_hydra_scalar(args.seed)}",
                f"task.deterministic={_to_hydra_scalar(args.deterministic)}",
                *_wandb_overrides(args),
                ]
                + ([f"task.reward_file={args.reward_file}"] if args.reward_file is not None else [])
            )
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
                *_wandb_overrides(args),
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
            (
                [
                "task=keras_train",
                f"task.episodes={args.episodes}",
                f"task.max_steps={args.max_steps}",
                f"task.goal_type={args.goal_type}",
                f"task.reward_function={args.reward_function}",
                f"task.output_base_dir={args.output_base_dir}",
                f"task.seed={_to_hydra_scalar(args.seed)}",
                f"task.deterministic={_to_hydra_scalar(args.deterministic)}",
                *_wandb_overrides(args),
                ]
                + ([f"task.reward_file={args.reward_file}"] if args.reward_file is not None else [])
            )
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
                *_wandb_overrides(args),
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

    if args.command == "paper-figures":
        replacement = "continuum-rl task=paper_figures"
        _emit_deprecation(replacement)
        goal_types = [item.strip() for item in args.goal_types.split(",") if item.strip()]
        goal_types_override = ",".join(goal_types)
        run_with_overrides(
            [
                "task=paper_figures",
                f"task.runs_root={args.runs_root}",
                f"task.output_dir={args.output_dir}",
                f"task.rollouts_per_seed={args.rollouts_per_seed}",
                f"task.max_steps={args.max_steps}",
                f"task.include_goal_types=[{goal_types_override}]",
                f"task.reward_function={args.reward_function}",
                f"task.show={_to_hydra_scalar(args.show)}",
            ]
        )
        return

    if args.command == "gradio-demo":
        replacement = "continuum-rl task=gradio_demo"
        _emit_deprecation(replacement)
        overrides = [
            "task=gradio_demo",
            f"task.framework={args.framework}",
            f"task.control_mode={args.control_mode}",
            f"task.goal_type={args.goal_type}",
            f"task.reward_function={args.reward_function}",
            f"task.device={args.device}",
            f"task.max_steps={args.max_steps}",
            f"task.seed={args.seed}",
            f"task.initial_kappa=[{args.initial_kappa}]",
            f"task.manual_action=[{args.manual_action}]",
            f"task.fixed_goal_xy=[{args.fixed_goal_xy}]",
            f"task.output_dir={args.output_dir}",
            f"task.save_outputs={_to_hydra_scalar(args.save_outputs)}",
            f"task.save_animation={_to_hydra_scalar(args.save_animation)}",
            f"task.animation_format={args.animation_format}",
            f"task.share={_to_hydra_scalar(args.share)}",
            f"task.server_name={args.server_name}",
            f"task.server_port={args.server_port}",
            f"task.single_run_lock={_to_hydra_scalar(args.single_run_lock)}",
            f"task.show_progress={_to_hydra_scalar(args.show_progress)}",
        ]
        if args.checkpoint_actor is not None:
            overrides.append(f"task.checkpoint_actor={args.checkpoint_actor}")
        if args.checkpoint_critic is not None:
            overrides.append(f"task.checkpoint_critic={args.checkpoint_critic}")
        if args.env_dt is not None:
            overrides.append(f"env.dt={args.env_dt}")
        if args.env_delta_kappa is not None:
            overrides.append(f"env.delta_kappa={args.env_delta_kappa}")
        if args.env_l is not None:
            overrides.append(f"env.l=[{args.env_l}]")
        if args.env_obstacles is not None:
            overrides.append(f"env.obstacles={args.env_obstacles}")
        run_with_overrides(overrides)
        return

    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
