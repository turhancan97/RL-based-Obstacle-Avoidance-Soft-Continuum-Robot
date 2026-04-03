"""Structured Hydra configuration for continuum RL tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf


GoalType = str
ObservationMode = str
TaskName = str


@dataclass(frozen=True)
class ObstacleConfig:
    x: float
    y: float


@dataclass
class EnvRuntimeConfig:
    delta_kappa: float = 0.001
    l: tuple[float, float, float] = (0.1, 0.1, 0.1)  # noqa: E741
    dt: float = 5e-2
    obstacles: tuple[ObstacleConfig, ...] = field(
        default_factory=lambda: (
            ObstacleConfig(x=-0.16, y=0.22),
            ObstacleConfig(x=-0.22, y=0.02),
            ObstacleConfig(x=-0.16, y=0.08),
        )
    )


@dataclass
class WandbConfig:
    enabled: bool = False
    mode: str = "offline"
    project: str = "continuum-rl-obstacle-avoidance"
    entity: Optional[str] = None
    group_by_experiment: bool = True
    run_name_template: str = "{framework}-{task_name}-{goal_type}-{reward_id}-seed{seed}"
    log_system_metrics: bool = True
    eval_interval_episodes: int = 50
    artifact_interval_episodes: int = 100
    upload_checkpoints: bool = True
    fail_open_on_init_error: bool = True
    health_alerts_enabled: bool = True
    health_ema_alpha: float = 0.01
    health_warmup_steps: int = 2_000
    health_growth_factor: float = 3.0
    health_actor_loss_min_abs: float = 10.0
    health_critic_loss_min_abs: float = 2.0
    health_grad_norm_max: float = 20.0


@dataclass
class AppBaseConfig:
    observation_mode: ObservationMode = "canonical"
    env: EnvRuntimeConfig = field(default_factory=EnvRuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class PytorchTrainConfig:
    name: str = "pytorch_train"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    reward_file: Optional[str] = None
    episodes: int = 300
    max_t: int = 750
    print_every: int = 25
    output_base_dir: str = "runs/pytorch"
    seed: Optional[int] = 0
    deterministic: bool = False
    agent_seed: int = 10
    buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 5e-4
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.5
    noise_theta: float = 0.15
    noise_sigma: float = 0.1


@dataclass
class PytorchEvalSmokeConfig:
    name: str = "pytorch_eval_smoke"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    max_t: int = 20
    checkpoint_actor: str = (
        "runs/pytorch/fixed_goal/reward_step_minus_weighted_euclidean/seed_0/model/checkpoint_actor.pth"
    )
    checkpoint_critic: str = (
        "runs/pytorch/fixed_goal/reward_step_minus_weighted_euclidean/seed_0/model/checkpoint_critic.pth"
    )
    seed: Optional[int] = 0
    deterministic: bool = False


@dataclass
class KerasTrainConfig:
    name: str = "keras_train"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    reward_file: Optional[str] = None
    episodes: int = 300
    max_steps: int = 750
    output_base_dir: str = "runs/keras"
    seed: Optional[int] = 0
    deterministic: bool = False
    buffer_capacity: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 5e-4
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    grad_clip_norm: float = 0.5
    noise_std: float = 0.1
    noise_theta: float = 0.15
    noise_dt: float = 1.0


@dataclass
class KerasEvalSmokeConfig:
    name: str = "keras_eval_smoke"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    max_steps: int = 20
    checkpoint_actor: str = (
        "runs/keras/fixed_goal/reward_step_minus_weighted_euclidean/seed_0/model/continuum_actor.h5"
    )
    seed: Optional[int] = 0
    deterministic: bool = False


@dataclass
class PytorchRewardVisConfig:
    name: str = "pytorch_reward_vis"
    goal_type: GoalType = "fixed_goal"
    reward_type: str = "reward_step_minus_weighted_euclidean"
    base_dir: str = "runs/pytorch"


@dataclass
class KerasRewardVisConfig:
    name: str = "keras_reward_vis"
    goal_type: GoalType = "fixed_goal"
    reward_type: str = "reward_step_minus_weighted_euclidean"
    base_dir: str = "runs/keras"


@dataclass
class PaperFiguresConfig:
    name: str = "paper_figures"
    runs_root: str = "runs"
    output_dir: str = "figures/paper/latest"
    format: str = "jpeg"
    show: bool = False
    min_seeds_for_claims: int = 5
    ci_method: str = "bootstrap"
    ci_level: float = 0.95
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 123
    rollouts_per_seed: int = 100
    include_goal_types: tuple[str, ...] = ("fixed_goal", "random_goal")
    max_steps: int = 750
    reward_function: str = "step_minus_weighted_euclidean"
    clear_output_dir: bool = True


TaskConfig = Union[
    PytorchTrainConfig,
    PytorchEvalSmokeConfig,
    KerasTrainConfig,
    KerasEvalSmokeConfig,
    PytorchRewardVisConfig,
    KerasRewardVisConfig,
    PaperFiguresConfig,
]


@dataclass
class AppConfig:
    observation_mode: ObservationMode
    env: EnvRuntimeConfig
    wandb: WandbConfig
    task_name: TaskName
    task: TaskConfig


_TASK_SCHEMAS: dict[str, type] = {
    "pytorch_train": PytorchTrainConfig,
    "pytorch_eval_smoke": PytorchEvalSmokeConfig,
    "keras_train": KerasTrainConfig,
    "keras_eval_smoke": KerasEvalSmokeConfig,
    "pytorch_reward_vis": PytorchRewardVisConfig,
    "keras_reward_vis": KerasRewardVisConfig,
    "paper_figures": PaperFiguresConfig,
}


def validate_and_convert(cfg: DictConfig) -> AppConfig:
    """Merge runtime cfg onto structured schema, rejecting unknown keys."""
    base_structured = OmegaConf.structured(AppBaseConfig)
    base_merged = OmegaConf.merge(
        base_structured,
        {
            "observation_mode": cfg.get("observation_mode"),
            "env": cfg.get("env"),
            "wandb": cfg.get("wandb"),
        },
    )
    observation_mode = base_merged.observation_mode
    if observation_mode != "canonical":
        raise ValueError(f"Unsupported observation_mode={observation_mode}. Only canonical mode is supported.")
    env_cfg: EnvRuntimeConfig = OmegaConf.to_object(base_merged.env)
    if env_cfg.delta_kappa <= 0:
        raise ValueError("env.delta_kappa must be greater than 0.")
    if env_cfg.dt <= 0:
        raise ValueError("env.dt must be greater than 0.")
    if len(env_cfg.l) != 3:
        raise ValueError("env.l must contain exactly 3 segment lengths.")
    wandb_cfg: WandbConfig = OmegaConf.to_object(base_merged.wandb)
    if wandb_cfg.mode not in {"offline", "online"}:
        raise ValueError("wandb.mode must be either 'offline' or 'online'.")
    if wandb_cfg.eval_interval_episodes <= 0:
        raise ValueError("wandb.eval_interval_episodes must be greater than 0.")
    if wandb_cfg.artifact_interval_episodes <= 0:
        raise ValueError("wandb.artifact_interval_episodes must be greater than 0.")
    if not (0 < wandb_cfg.health_ema_alpha <= 1):
        raise ValueError("wandb.health_ema_alpha must be in (0, 1].")
    if wandb_cfg.health_warmup_steps < 0:
        raise ValueError("wandb.health_warmup_steps must be >= 0.")
    if wandb_cfg.health_growth_factor <= 0:
        raise ValueError("wandb.health_growth_factor must be > 0.")
    if wandb_cfg.health_actor_loss_min_abs <= 0:
        raise ValueError("wandb.health_actor_loss_min_abs must be > 0.")
    if wandb_cfg.health_critic_loss_min_abs <= 0:
        raise ValueError("wandb.health_critic_loss_min_abs must be > 0.")
    if wandb_cfg.health_grad_norm_max <= 0:
        raise ValueError("wandb.health_grad_norm_max must be > 0.")
    if not wandb_cfg.project:
        raise ValueError("wandb.project must be a non-empty string.")

    task_name = cfg.task.name
    schema = _TASK_SCHEMAS.get(task_name)
    if schema is None:
        raise ValueError(f"Unsupported task={task_name}")

    task_structured = OmegaConf.structured(schema)
    task_merged = OmegaConf.merge(task_structured, cfg.task)
    task_obj = OmegaConf.to_object(task_merged)
    if hasattr(task_obj, "goal_type") and task_obj.goal_type not in {"fixed_goal", "random_goal"}:
        raise ValueError(f"Unsupported goal_type={task_obj.goal_type}")
    if hasattr(task_obj, "reward_function") and hasattr(task_obj, "reward_file"):
        reward_file = getattr(task_obj, "reward_file", None)
        if reward_file is None or str(reward_file).strip() == "" or str(reward_file).lower() == "auto":
            setattr(task_obj, "reward_file", f"reward_{task_obj.reward_function}")
    if hasattr(task_obj, "gamma") and not (0 < task_obj.gamma <= 1):
        raise ValueError("task.gamma must be in (0, 1].")
    if hasattr(task_obj, "tau") and not (0 < task_obj.tau <= 1):
        raise ValueError("task.tau must be in (0, 1].")
    for positive_key in (
        "buffer_size",
        "buffer_capacity",
        "batch_size",
        "actor_lr",
        "critic_lr",
        "grad_clip_norm",
        "noise_sigma",
        "noise_std",
        "noise_theta",
        "noise_dt",
    ):
        if hasattr(task_obj, positive_key):
            value = getattr(task_obj, positive_key)
            if value <= 0:
                raise ValueError(f"task.{positive_key} must be greater than 0.")
    if hasattr(task_obj, "weight_decay") and task_obj.weight_decay < 0:
        raise ValueError("task.weight_decay must be >= 0.")
    if task_name == "paper_figures":
        if task_obj.format.lower() not in {"jpeg", "jpg"}:
            raise ValueError("task.format must be jpeg (or jpg alias).")
        if task_obj.ci_method != "bootstrap":
            raise ValueError("task.ci_method must be bootstrap.")
        if not (0 < task_obj.ci_level < 1):
            raise ValueError("task.ci_level must be in (0, 1).")
        if task_obj.min_seeds_for_claims <= 0:
            raise ValueError("task.min_seeds_for_claims must be > 0.")
        if task_obj.bootstrap_samples <= 0:
            raise ValueError("task.bootstrap_samples must be > 0.")
        if task_obj.rollouts_per_seed <= 0:
            raise ValueError("task.rollouts_per_seed must be > 0.")
        if task_obj.max_steps <= 0:
            raise ValueError("task.max_steps must be > 0.")
        allowed_goal_types = {"fixed_goal", "random_goal"}
        if not task_obj.include_goal_types:
            raise ValueError("task.include_goal_types must not be empty.")
        invalid_goal_types = [g for g in task_obj.include_goal_types if g not in allowed_goal_types]
        if invalid_goal_types:
            raise ValueError(
                "task.include_goal_types contains invalid values: "
                f"{invalid_goal_types}. Allowed: {sorted(allowed_goal_types)}"
            )

    return AppConfig(
        observation_mode=observation_mode,
        env=env_cfg,
        wandb=wandb_cfg,
        task_name=task_name,
        task=task_obj,
    )
