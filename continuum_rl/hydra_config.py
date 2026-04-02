"""Structured Hydra configuration for continuum RL tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from omegaconf import DictConfig, OmegaConf


GoalType = str
ObservationMode = str
TaskName = str


@dataclass
class AppBaseConfig:
    observation_mode: ObservationMode = "canonical"


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


@dataclass
class KerasTrainConfig:
    name: str = "keras_train"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    reward_file: str = "reward_step_minus_weighted_euclidean"
    episodes: int = 500
    max_steps: int = 500
    output_base_dir: str = "Keras"


@dataclass
class KerasEvalSmokeConfig:
    name: str = "keras_eval_smoke"
    goal_type: GoalType = "fixed_goal"
    reward_function: str = "step_minus_weighted_euclidean"
    max_steps: int = 20
    checkpoint_actor: str = (
        "Keras/fixed_goal/reward_step_minus_weighted_euclidean/model/continuum_actor.h5"
    )


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
    base_merged = OmegaConf.merge(base_structured, {"observation_mode": cfg.get("observation_mode")})
    observation_mode = base_merged.observation_mode
    if observation_mode != "canonical":
        raise ValueError(f"Unsupported observation_mode={observation_mode}. Only canonical mode is supported.")

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
        task_name=task_name,
        task=task_obj,
    )
