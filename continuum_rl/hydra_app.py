"""Hydra-first dispatcher for continuum RL tasks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from .hydra_config import AppConfig, EnvRuntimeConfig, validate_and_convert


CONF_DIR = Path(__file__).resolve().parent / "conf"


def _env_kwargs(env_cfg: EnvRuntimeConfig) -> dict:
    return {
        "obstacles": [{"x": obs.x, "y": obs.y} for obs in env_cfg.obstacles],
        "delta_kappa": env_cfg.delta_kappa,
        "l": list(env_cfg.l),
        "dt": env_cfg.dt,
    }


def run_task(cfg: AppConfig) -> None:
    if cfg.observation_mode != "canonical":
        raise ValueError(
            f"Unsupported observation_mode={cfg.observation_mode}. Only canonical mode is supported."
        )

    if cfg.task_name == "pytorch_train":
        from Pytorch.ddpg import train as pytorch_train

        task = cfg.task
        run_config = asdict(cfg)
        pytorch_train(
            n_episodes=task.episodes,
            max_t=task.max_t,
            print_every=task.print_every,
            goal_type=task.goal_type,
            reward_function=task.reward_function,
            reward_file=task.reward_file,
            output_base_dir=Path(task.output_base_dir),
            seed=task.seed,
            deterministic=task.deterministic,
            env_kwargs=_env_kwargs(cfg.env),
            wandb_cfg=asdict(cfg.wandb),
            run_config=run_config,
            task_name=cfg.task_name,
            agent_seed=task.agent_seed,
            buffer_size=task.buffer_size,
            batch_size=task.batch_size,
            gamma=task.gamma,
            tau=task.tau,
            actor_lr=task.actor_lr,
            critic_lr=task.critic_lr,
            weight_decay=task.weight_decay,
            grad_clip_norm=task.grad_clip_norm,
            noise_theta=task.noise_theta,
            noise_sigma=task.noise_sigma,
        )
        return

    if cfg.task_name == "pytorch_eval_smoke":
        from Pytorch.ddpg import evaluate_smoke as pytorch_eval_smoke

        task = cfg.task
        run_config = asdict(cfg)
        pytorch_eval_smoke(
            checkpoint_actor=Path(task.checkpoint_actor),
            checkpoint_critic=Path(task.checkpoint_critic),
            goal_type=task.goal_type,
            reward_function=task.reward_function,
            max_t=task.max_t,
            seed=task.seed,
            deterministic=task.deterministic,
            env_kwargs=_env_kwargs(cfg.env),
            wandb_cfg=asdict(cfg.wandb),
            run_config=run_config,
            task_name=cfg.task_name,
        )
        return

    if cfg.task_name == "keras_train":
        from Keras.DDPG import train as keras_train

        task = cfg.task
        run_config = asdict(cfg)
        keras_train(
            total_episodes=task.episodes,
            max_steps=task.max_steps,
            goal_type=task.goal_type,
            reward_function=task.reward_function,
            reward_file=task.reward_file,
            output_base_dir=Path(task.output_base_dir),
            seed=task.seed,
            deterministic=task.deterministic,
            env_kwargs=_env_kwargs(cfg.env),
            wandb_cfg=asdict(cfg.wandb),
            run_config=run_config,
            task_name=cfg.task_name,
            buffer_capacity=task.buffer_capacity,
            batch_size=task.batch_size,
            gamma=task.gamma,
            tau=task.tau,
            actor_lr=task.actor_lr,
            critic_lr=task.critic_lr,
            grad_clip_norm=task.grad_clip_norm,
            noise_std=task.noise_std,
            noise_theta=task.noise_theta,
            noise_dt=task.noise_dt,
        )
        return

    if cfg.task_name == "keras_eval_smoke":
        from Keras.DDPG import evaluate_smoke as keras_eval_smoke

        task = cfg.task
        run_config = asdict(cfg)
        keras_eval_smoke(
            checkpoint_actor=Path(task.checkpoint_actor),
            goal_type=task.goal_type,
            reward_function=task.reward_function,
            max_steps=task.max_steps,
            seed=task.seed,
            deterministic=task.deterministic,
            env_kwargs=_env_kwargs(cfg.env),
            wandb_cfg=asdict(cfg.wandb),
            run_config=run_config,
            task_name=cfg.task_name,
        )
        return

    if cfg.task_name == "pytorch_reward_vis":
        from Pytorch.reward_visualization.reward_vis import run as pytorch_reward_vis

        task = cfg.task
        out_dir = pytorch_reward_vis(
            goal_type=task.goal_type,
            reward_type=task.reward_type,
            base_dir=Path(task.base_dir),
        )
        print(f"Saved plots to: {out_dir}")
        return

    if cfg.task_name == "keras_reward_vis":
        from Keras.reward_visualization.reward_vis import run as keras_reward_vis

        task = cfg.task
        out_dir = keras_reward_vis(
            goal_type=task.goal_type,
            reward_type=task.reward_type,
            base_dir=Path(task.base_dir),
        )
        print(f"Saved plots to: {out_dir}")
        return

    raise ValueError(f"Unsupported task={cfg.task_name}")


def compose_config(overrides: Sequence[str] | None = None) -> AppConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    return validate_and_convert(cfg)


def run_with_overrides(overrides: Sequence[str] | None = None) -> None:
    run_task(compose_config(overrides=overrides))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    run_task(validate_and_convert(cfg))


def main() -> None:
    _hydra_main()


if __name__ == "__main__":
    main()
