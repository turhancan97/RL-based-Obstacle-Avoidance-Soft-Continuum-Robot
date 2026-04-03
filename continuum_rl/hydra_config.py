"""Structured Hydra configuration for continuum RL tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf


GoalType = str
ObservationMode = str
TaskName = str


@dataclass
class AppBaseConfig:
    observation_mode: ObservationMode = "canonical"
    env: "EnvRuntimeConfig" = field(default_factory=lambda: EnvRuntimeConfig())


@dataclass(frozen=True)
class ObstacleConfig:
    x: float
    y: float


@dataclass
class EnvRuntimeConfig:
    delta_kappa: float = 0.001
    l: tuple[float, float, float] = (0.1, 0.1, 0.1)
    dt: float = 5e-2
    obstacles: tuple[ObstacleConfig, ...] = field(
        default_factory=lambda: (
            ObstacleConfig(x=-0.16, y=0.22),
            ObstacleConfig(x=-0.22, y=0.02),
            ObstacleConfig(x=-0.16, y=0.08),
        )
    )


@dataclass
class PytorchTrainConfig:
    name: str = "pytorch_train"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    reward_file: str = "reward_step_minus_weighted_euclidean"
    episodes: int = 300
    max_t: int = 750
    print_every: int = 25
    output_base_dir: str = "Pytorch"
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class PytorchEvalSmokeConfig:
    name: str = "pytorch_eval_smoke"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    max_t: int = 20
    checkpoint_actor: str = (
        "Pytorch/fixed_goal/reward_step_minus_weighted_euclidean/model/checkpoint_actor.pth"
    )
    checkpoint_critic: str = (
        "Pytorch/fixed_goal/reward_step_minus_weighted_euclidean/model/checkpoint_critic.pth"
    )
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class KerasTrainConfig:
    name: str = "keras_train"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    reward_file: str = "reward_step_minus_weighted_euclidean"
    episodes: int = 500
    max_steps: int = 500
    output_base_dir: str = "Keras"
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class KerasEvalSmokeConfig:
    name: str = "keras_eval_smoke"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    max_steps: int = 20
    checkpoint_actor: str = (
        "Keras/fixed_goal/reward_step_minus_weighted_euclidean/model/continuum_actor.h5"
    )
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class PytorchRewardVisConfig:
    name: str = "pytorch_reward_vis"
    goal_type: GoalType = "fixed_goal"
    reward_type: str = "reward_step_minus_weighted_euclidean"
    base_dir: str = "Pytorch"


@dataclass
class KerasRewardVisConfig:
    name: str = "keras_reward_vis"
    goal_type: GoalType = "fixed_goal"
    reward_type: str = "reward_step_minus_weighted_euclidean"
    base_dir: str = "Keras"


TaskConfig = Union[
    PytorchTrainConfig,
    PytorchEvalSmokeConfig,
    KerasTrainConfig,
    KerasEvalSmokeConfig,
    PytorchRewardVisConfig,
    KerasRewardVisConfig,
]


@dataclass
class AppConfig:
    observation_mode: ObservationMode
    env: EnvRuntimeConfig
    task_name: TaskName
    task: TaskConfig


_TASK_SCHEMAS: dict[str, type] = {
    "pytorch_train": PytorchTrainConfig,
    "pytorch_eval_smoke": PytorchEvalSmokeConfig,
    "keras_train": KerasTrainConfig,
    "keras_eval_smoke": KerasEvalSmokeConfig,
    "pytorch_reward_vis": PytorchRewardVisConfig,
    "keras_reward_vis": KerasRewardVisConfig,
}


def validate_and_convert(cfg: DictConfig) -> AppConfig:
    """Merge runtime cfg onto structured schema, rejecting unknown keys."""
    base_structured = OmegaConf.structured(AppBaseConfig)
    base_merged = OmegaConf.merge(
        base_structured,
        {
            "observation_mode": cfg.get("observation_mode"),
            "env": cfg.get("env"),
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

    task_name = cfg.task.name
    schema = _TASK_SCHEMAS.get(task_name)
    if schema is None:
        raise ValueError(f"Unsupported task={task_name}")

    task_structured = OmegaConf.structured(schema)
    task_merged = OmegaConf.merge(task_structured, cfg.task)
    task_obj = OmegaConf.to_object(task_merged)
    if hasattr(task_obj, "goal_type") and task_obj.goal_type not in {"fixed_goal", "random_goal"}:
        raise ValueError(f"Unsupported goal_type={task_obj.goal_type}")

    return AppConfig(
        observation_mode=observation_mode,
        env=env_cfg,
        task_name=task_name,
        task=task_obj,
    )
