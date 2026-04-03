"""Continuum robot reinforcement-learning environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from kinematics.forward_velocity_kinematics import (
    coupletransformations,
    jacobian_matrix,
    three_section_planar_robot,
    trans_mat_cc,
)

from .gym_compat import gym, spaces
from .spaces import AmorphousSpace

ObservationMode = Literal["canonical"]
GoalType = Literal["fixed_goal", "random_goal"]
OBS_SCHEMA_CANONICAL_V3 = "canonical_v3"


DEFAULT_OBSTACLES = (
    {"x": -0.16, "y": 0.22},
    {"x": -0.22, "y": 0.02},
    {"x": -0.16, "y": 0.08},
)


@dataclass(frozen=True)
class EnvConfig:
    observation_mode: ObservationMode = "canonical"
    goal_type: GoalType = "fixed_goal"
    fixed_goal_kappa: tuple[float, float, float] = (6.2, 6.2, 6.2)
    random_goal_range: tuple[float, float] = (-4.0, 16.0)
    delta_kappa: float = 0.001
    l: tuple[float, float, float] = (0.1, 0.1, 0.1)  # noqa: E741
    dt: float = 5e-2
    max_episode_steps: int | None = None


class ContinuumEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    obs_schema = OBS_SCHEMA_CANONICAL_V3

    def __init__(
        self,
        obstacles: Iterable[dict[str, float]] | None = None,
        observation_mode: ObservationMode = "canonical",
        goal_type: GoalType = "fixed_goal",
        fixed_goal_kappa: tuple[float, float, float] = (6.2, 6.2, 6.2),
        random_goal_range: tuple[float, float] = (-4.0, 16.0),
        delta_kappa: float = 0.001,
        l: Sequence[float] = (0.1, 0.1, 0.1),  # noqa: E741
        dt: float = 5e-2,
        max_episode_steps: int | None = None,
    ):
        if observation_mode != "canonical":
            raise ValueError(
                f"Unsupported observation_mode={observation_mode}. "
                "Only canonical mode is supported."
            )
        if goal_type not in {"fixed_goal", "random_goal"}:
            raise ValueError(f"Unsupported goal_type={goal_type}.")
        if delta_kappa <= 0:
            raise ValueError("delta_kappa must be greater than 0.")
        if dt <= 0:
            raise ValueError("dt must be greater than 0.")
        if len(l) != 3:
            raise ValueError("l must contain exactly 3 segment lengths.")
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer or None.")
        normalized_l = tuple(float(segment_length) for segment_length in l)

        self.config = EnvConfig(
            observation_mode=observation_mode,
            goal_type=goal_type,
            fixed_goal_kappa=fixed_goal_kappa,
            random_goal_range=random_goal_range,
            delta_kappa=float(delta_kappa),
            l=normalized_l,
            dt=float(dt),
            max_episode_steps=max_episode_steps,
        )

        self.delta_kappa = self.config.delta_kappa
        self.kappa_dot_max = 1.0
        self.kappa_max = 16.0
        self.kappa_min = -4.0

        self.l = list(self.config.l)
        self.dt = self.config.dt
        self.J = np.zeros((2, 3), dtype=np.float32)
        self.error = 0.0
        self.previous_error = 0.0
        self.start_kappa = [0.0, 0.0, 0.0]
        self.time = 0.0
        self.overshoot0 = 0
        self.overshoot1 = 0
        self.stop = 0

        self.obstacles = [dict(o) for o in (obstacles if obstacles is not None else DEFAULT_OBSTACLES)]
        self.num_obstacles = len(self.obstacles)
        self.workspace = AmorphousSpace()

        self.obs_size = 7 + (2 * self.num_obstacles)
        low = np.full((self.obs_size,), -0.5, dtype=np.float32)
        high = np.full((self.obs_size,), 0.5, dtype=np.float32)
        low[4:7] = self.kappa_min
        high[4:7] = self.kappa_max
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.kappa_dot_max,
            high=self.kappa_dot_max,
            shape=(3,),
            dtype=np.float32,
        )

        self.position_dic = {
            "Section1": {"x": [], "y": []},
            "Section2": {"x": [], "y": []},
            "Section3": {"x": [], "y": []},
        }
        for i in range(self.num_obstacles):
            self.position_dic[f"Obs{i+1}"] = {"x": [], "y": []}

        self._full_state = np.zeros(7 + 2 * self.num_obstacles, dtype=np.float32)
        self.last_u = np.zeros(3, dtype=np.float32)
        self.initial_distance: float | None = None
        self.step_count = 0

    def _select_observation(self, full_state: np.ndarray) -> np.ndarray:
        return full_state.astype(np.float32)

    def _compose_full_state(
        self,
        x: float,
        y: float,
        goal_x: float,
        goal_y: float,
        kappa1: float | None = None,
        kappa2: float | None = None,
        kappa3: float | None = None,
    ) -> np.ndarray:
        k1 = self.kappa1 if kappa1 is None else kappa1
        k2 = self.kappa2 if kappa2 is None else kappa2
        k3 = self.kappa3 if kappa3 is None else kappa3
        state = [x, y, goal_x, goal_y, k1, k2, k3]
        for obstacle in self.obstacles:
            state.extend([obstacle["x"], obstacle["y"]])
        return np.array(state, dtype=np.float32)

    def _compute_reward_cost(
        self,
        x: float,
        y: float,
        goal_x: float,
        goal_y: float,
        reward_function: str,
    ) -> float:
        if reward_function == "step_error_comparison":
            self.error = math.sqrt(((goal_x - x) ** 2) + ((goal_y - y) ** 2))
            if self.error < self.previous_error:
                costs = 1.0
            elif self.error == self.previous_error:
                costs = -0.5
            else:
                costs = -1.0
            return costs

        if reward_function == "step_minus_euclidean_square":
            self.error = ((goal_x - x) ** 2) + ((goal_y - y) ** 2)
            return self.error

        if reward_function == "step_minus_weighted_euclidean":
            self.error = math.sqrt(((goal_x - x) ** 2) + ((goal_y - y) ** 2))
            obstacle_penalties = 0.0
            for obstacle in self.obstacles:
                obstacle_dist = math.sqrt(((obstacle["x"] - x) ** 2) + ((obstacle["y"] - y) ** 2))
                if obstacle_dist <= 0.025:
                    obstacle_penalties += (1 - (obstacle_dist / 0.025)) * 2

            if self.initial_distance is None:
                self.initial_distance = max(self.error, 1e-6)

            progress = (self.previous_error - self.error) / self.initial_distance
            costs = self.error / self.initial_distance
            costs -= 2.0 * progress
            costs += 0.1 * self.dt
            if self.error <= 0.02:
                costs -= (0.02 - self.error) * 5
            costs += obstacle_penalties
            return costs

        if reward_function == "step_distance_based":
            self.error = math.sqrt(((goal_x - x) ** 2) + ((goal_y - y) ** 2))
            if self.error == self.previous_error:
                return -100.0
            if self.error <= 0.025:
                return 200.0
            if self.error <= 0.05:
                return 150.0
            if self.error <= 0.1:
                return 100.0
            return 1000.0 * (self.previous_error - self.error)

        raise ValueError(f"Unknown reward function: {reward_function}")

    def _ensure_finite_array(self, values: Sequence[float] | np.ndarray, label: str, context: str) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(
                f"Non-finite {label} detected during {context}. "
                f"values={arr.tolist()}, stop={self.stop}, "
                f"kappa=({self.kappa1}, {self.kappa2}, {self.kappa3})"
            )

    def _apply_stop_mask(self, action: np.ndarray) -> np.ndarray:
        control = action.copy()
        if self.stop == 1:
            control = np.array([0.0, action[1], action[2]], dtype=np.float32)
        elif self.stop == 2:
            control = np.array([action[0], 0.0, action[2]], dtype=np.float32)
        elif self.stop == 3:
            control = np.array([action[0], action[1], 0.0], dtype=np.float32)
        elif self.stop == 4:
            control = np.array([0.0, 0.0, action[2]], dtype=np.float32)
        elif self.stop == 5:
            control = np.array([0.0, action[1], 0.0], dtype=np.float32)
        elif self.stop == 6:
            control = np.array([action[0], 0.0, 0.0], dtype=np.float32)
        elif self.stop == 7:
            control = np.zeros(3, dtype=np.float32)
        return control

    def _compute_termination_reward(self, costs: float, reward_function: str) -> tuple[bool, float]:
        if reward_function == "step_minus_euclidean_square":
            terminated = bool(math.sqrt(max(costs, 0.0)) <= 0.005)
            reward = -costs
        else:
            terminated = bool(self.error <= 0.005)
            reward = -costs if reward_function == "step_minus_weighted_euclidean" else costs
        return terminated, float(reward)

    def _update_stop_mode(self):
        self.stop = 0
        k1 = self.kappa1 <= self.kappa_min or self.kappa1 >= self.kappa_max
        k2 = self.kappa2 <= self.kappa_min or self.kappa2 >= self.kappa_max
        k3 = self.kappa3 <= self.kappa_min or self.kappa3 >= self.kappa_max

        if k1:
            self.stop = 1
        elif k2:
            self.stop = 2
        elif k3:
            self.stop = 3

        if k1 and k2:
            self.stop = 4
        elif k1 and k3:
            self.stop = 5
        elif k2 and k3:
            self.stop = 6

        if k1 and k2 and k3:
            self.stop = 7

    def step(self, action: Sequence[float], reward_function: str = "step_minus_euclidean_square"):
        step_context = f"step(reward_function={reward_function})"
        x, y, goal_x, goal_y = self._full_state[:4]
        u = np.clip(np.asarray(action, dtype=np.float32), -self.kappa_dot_max, self.kappa_dot_max)
        self._ensure_finite_array(u, "action", f"{step_context}:action preprocessing")

        self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
        self._ensure_finite_array(self.J, "jacobian", f"{step_context}:jacobian")
        control = self._apply_stop_mask(u)
        self._ensure_finite_array(control, "masked control", f"{step_context}:stop mask")

        x_vel = self.J @ control
        self._ensure_finite_array(x_vel, "tip velocity", f"{step_context}:velocity")
        new_x = float(x + (x_vel[0] * self.dt))
        new_y = float(y + (x_vel[1] * self.dt))
        self._ensure_finite_array([new_x, new_y], "next tip position", f"{step_context}:integration")

        self.kappa1 = float(np.clip(self.kappa1 + (control[0] * self.dt), self.kappa_min, self.kappa_max))
        self.kappa2 = float(np.clip(self.kappa2 + (control[1] * self.dt), self.kappa_min, self.kappa_max))
        self.kappa3 = float(np.clip(self.kappa3 + (control[2] * self.dt), self.kappa_min, self.kappa_max))
        self._update_stop_mode()

        if not self.workspace.contains([new_x, new_y]):
            self.overshoot0 += 1
            new_x, new_y = self.workspace.clip([new_x, new_y]).astype(np.float32).tolist()
            self._ensure_finite_array(
                [new_x, new_y],
                "clipped tip position",
                f"{step_context}:workspace clip",
            )

        if self.workspace.contains([goal_x, goal_y]):
            new_goal_x, new_goal_y = goal_x, goal_y
        else:
            self.overshoot1 += 1
            new_goal_x, new_goal_y = self.workspace.clip([goal_x, goal_y]).astype(np.float32).tolist()
        self._ensure_finite_array([new_goal_x, new_goal_y], "goal position", f"{step_context}:goal handling")

        self._full_state = self._compose_full_state(new_x, new_y, new_goal_x, new_goal_y)
        costs = self._compute_reward_cost(
            new_x,
            new_y,
            new_goal_x,
            new_goal_y,
            reward_function=reward_function,
        )
        terminated, reward = self._compute_termination_reward(costs=costs, reward_function=reward_function)
        self.previous_error = self.error
        self.last_u = control
        self.step_count += 1

        obs = self._select_observation(self._full_state)
        truncated = bool(
            self.config.max_episode_steps is not None
            and self.step_count >= self.config.max_episode_steps
        )
        info = {
            "error": float(self.error),
            "reward_function": reward_function,
            "observation_mode": self.config.observation_mode,
            "observation_schema": self.obs_schema,
            "goal_type": self.config.goal_type,
            "step_count": self.step_count,
        }
        return obs, float(reward), terminated, truncated, info

    def _sample_goal(self) -> tuple[float, float, float, float, float]:
        if self.config.goal_type == "fixed_goal":
            target_k1, target_k2, target_k3 = self.config.fixed_goal_kappa
        else:
            low, high = self.config.random_goal_range
            target_k1 = float(self.np_random.uniform(low=low, high=high))
            target_k2 = float(self.np_random.uniform(low=low, high=high))
            target_k3 = float(self.np_random.uniform(low=low, high=high))

        T3_target = three_section_planar_robot(target_k1, target_k2, target_k3, self.l)
        self._ensure_finite_array(T3_target, "target tip transform", "goal sampling")
        goal_x, goal_y = np.array([T3_target[0, 3], T3_target[1, 3]], dtype=np.float32)
        if not self.workspace.contains([goal_x, goal_y]):
            self.overshoot1 += 1
            goal_x, goal_y = self.workspace.clip([goal_x, goal_y]).astype(np.float32)
        return float(target_k1), float(target_k2), float(target_k3), float(goal_x), float(goal_y)

    def _parse_initial_kappa_override(self, options: dict | None) -> tuple[float, float, float] | None:
        if not options or "initial_kappa" not in options:
            return None
        raw = options["initial_kappa"]
        if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != 3:
            raise ValueError("reset options.initial_kappa must be a sequence of length 3.")
        values = [float(v) for v in raw]
        self._ensure_finite_array(values, "initial_kappa", "reset options parsing")
        for idx, val in enumerate(values):
            if not (self.kappa_min <= val <= self.kappa_max):
                raise ValueError(
                    f"reset options.initial_kappa[{idx}]={val} is out of bounds "
                    f"[{self.kappa_min}, {self.kappa_max}]."
                )
        return float(values[0]), float(values[1]), float(values[2])

    def _parse_goal_xy_override(self, options: dict | None) -> tuple[float, float] | None:
        if not options or "goal_xy" not in options:
            return None
        raw = options["goal_xy"]
        if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != 2:
            raise ValueError("reset options.goal_xy must be a sequence of length 2.")
        values = [float(v) for v in raw]
        self._ensure_finite_array(values, "goal_xy", "reset options parsing")
        goal = np.asarray(values, dtype=np.float32)
        if not self.workspace.contains(goal):
            self.overshoot1 += 1
            goal = self.workspace.clip(goal).astype(np.float32)
        return float(goal[0]), float(goal[1])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.workspace.set_rng(self.np_random)
        self.initial_distance = None
        self.step_count = 0

        initial_kappa_override = self._parse_initial_kappa_override(options)
        if initial_kappa_override is None:
            self.kappa1 = float(self.np_random.uniform(low=self.kappa_min, high=self.kappa_max))
            self.kappa2 = float(self.np_random.uniform(low=self.kappa_min, high=self.kappa_max))
            self.kappa3 = float(self.np_random.uniform(low=self.kappa_min, high=self.kappa_max))
        else:
            self.kappa1, self.kappa2, self.kappa3 = initial_kappa_override
        self._update_stop_mode()

        T3_cc = three_section_planar_robot(self.kappa1, self.kappa2, self.kappa3, self.l)
        self._ensure_finite_array(T3_cc, "tip transform", "reset")
        x, y = np.array([T3_cc[0, 3], T3_cc[1, 3]], dtype=np.float32)
        self._ensure_finite_array([x, y], "tip position", "reset")
        if not self.workspace.contains([x, y]):
            x, y = self.workspace.clip([x, y]).astype(np.float32)
            self.overshoot0 += 1

        goal_override = self._parse_goal_xy_override(options)
        if goal_override is None:
            self.target_k1, self.target_k2, self.target_k3, goal_x, goal_y = self._sample_goal()
        else:
            self.target_k1, self.target_k2, self.target_k3 = self.config.fixed_goal_kappa
            goal_x, goal_y = goal_override

        self._full_state = self._compose_full_state(float(x), float(y), goal_x, goal_y)
        self.error = math.sqrt(((goal_x - x) ** 2) + ((goal_y - y) ** 2))
        self.previous_error = self.error
        self.last_u = np.zeros(3, dtype=np.float32)

        obs = self._select_observation(self._full_state)
        info = {
            "observation_mode": self.config.observation_mode,
            "observation_schema": self.obs_schema,
            "goal_type": self.config.goal_type,
        }
        return obs, info

    @property
    def state(self) -> np.ndarray:
        return self._select_observation(self._full_state)

    def _full_state_view(self) -> np.ndarray:
        return self._full_state.astype(np.float32)

    def render_calculate(self):
        T1_cc = trans_mat_cc(self.kappa1, self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc) - 1, :], (4, 4), order="F")
        T2 = trans_mat_cc(self.kappa2, self.l[1])
        T2_cc = coupletransformations(T2, T1_tip)
        T2_tip = np.reshape(T2_cc[len(T2_cc) - 1, :], (4, 4), order="F")
        T3 = trans_mat_cc(self.kappa3, self.l[2])
        T3_cc = coupletransformations(T3, T2_tip)

        self.position_dic["Section1"]["x"].append(T1_cc[:, 12])
        self.position_dic["Section1"]["y"].append(T1_cc[:, 13])
        self.position_dic["Section2"]["x"].append(T2_cc[:, 12])
        self.position_dic["Section2"]["y"].append(T2_cc[:, 13])
        self.position_dic["Section3"]["x"].append(T3_cc[:, 12])
        self.position_dic["Section3"]["y"].append(T3_cc[:, 13])

        for i in range(self.num_obstacles):
            obs_idx = 7 + i * 2
            self.position_dic[f"Obs{i+1}"]["x"].append(self._full_state[obs_idx])
            self.position_dic[f"Obs{i+1}"]["y"].append(self._full_state[obs_idx + 1])

    def render_init(self):
        self.fig = plt.figure()
        self.fig.set_dpi(75)
        self.ax = plt.axes()

    def render_update(self, frame_idx: int):
        self.ax.cla()
        self.ax.plot([-0.025, 0.025], [0, 0], "black", linewidth=5)
        self.ax.plot(
            self.position_dic["Section1"]["x"][frame_idx],
            self.position_dic["Section1"]["y"][frame_idx],
            "b",
            linewidth=3,
        )
        self.ax.plot(
            self.position_dic["Section2"]["x"][frame_idx],
            self.position_dic["Section2"]["y"][frame_idx],
            "r",
            linewidth=3,
        )
        self.ax.plot(
            self.position_dic["Section3"]["x"][frame_idx],
            self.position_dic["Section3"]["y"][frame_idx],
            "g",
            linewidth=3,
        )
        self.ax.scatter(
            self.position_dic["Section3"]["x"][frame_idx][-1],
            self.position_dic["Section3"]["y"][frame_idx][-1],
            linewidths=5,
            color="black",
        )

        for i in range(self.num_obstacles):
            self.ax.scatter(
                self.position_dic[f"Obs{i+1}"]["x"][frame_idx],
                self.position_dic[f"Obs{i+1}"]["y"][frame_idx],
                linewidths=5,
                color="red",
            )

        self.ax.scatter(self._full_state[2], self._full_state[3], 100, marker="x", linewidths=2, color="red")
        self.ax.set_title(f"The time elapsed in the simulation is {round(self.time, 2)} seconds.")
        self.ax.set_xlabel("X - Position [m]")
        self.ax.set_ylabel("Y - Position [m]")
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_ylim([-0.4, 0.4])

    def render(self):
        return FuncAnimation(
            fig=self.fig,
            func=self.render_update,
            frames=np.shape(self.position_dic["Section1"]["x"])[0],
            interval=1,
        )

    def visualization(
        self,
        x_pos,
        y_pos,
        output_dir: str | Path = "visualizations/env",
        filename: str = "continuum_env_visualization.png",
        save: bool = True,
    ):
        T1_cc = trans_mat_cc(self.start_kappa[0], self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc) - 1, :], (4, 4), order="F")
        T2 = trans_mat_cc(self.start_kappa[1], self.l[1])
        T2_cc = coupletransformations(T2, T1_tip)
        T2_tip = np.reshape(T2_cc[len(T2_cc) - 1, :], (4, 4), order="F")
        T3 = trans_mat_cc(self.start_kappa[2], self.l[2])
        T3_cc = coupletransformations(T3, T2_tip)

        plt.plot([-0.025, 0.025], [0, 0], "black", linewidth=5)
        plt.plot(T1_cc[:, 12], T1_cc[:, 13], "b", linewidth=3)
        plt.plot(T2_cc[:, 12], T2_cc[:, 13], "r", linewidth=3)
        plt.plot(T3_cc[:, 12], T3_cc[:, 13], "g", linewidth=3)
        plt.scatter(T3_cc[-1, 12], T3_cc[-1, 13], linewidths=5, color="orange", label="Initial Point")

        T1_cc = trans_mat_cc(self.kappa1, self.l[0])
        T1_tip = np.reshape(T1_cc[len(T1_cc) - 1, :], (4, 4), order="F")
        T2 = trans_mat_cc(self.kappa2, self.l[1])
        T2_cc = coupletransformations(T2, T1_tip)
        T2_tip = np.reshape(T2_cc[len(T2_cc) - 1, :], (4, 4), order="F")
        T3 = trans_mat_cc(self.kappa3, self.l[2])
        T3_cc = coupletransformations(T3, T2_tip)

        plt.plot(T1_cc[:, 12], T1_cc[:, 13], "b", linewidth=3)
        plt.plot(T2_cc[:, 12], T2_cc[:, 13], "r", linewidth=3)
        plt.plot(T3_cc[:, 12], T3_cc[:, 13], "g", linewidth=3)
        plt.scatter(T3_cc[-1, 12], T3_cc[-1, 13], linewidths=5, color="black")
        plt.scatter(
            self._full_state[2],
            self._full_state[3],
            100,
            marker="x",
            linewidths=4,
            color="red",
            label="Target Point",
        )

        for i, obstacle in enumerate(self.obstacles):
            plt.scatter(
                obstacle["x"],
                obstacle["y"],
                100,
                marker="x",
                linewidths=4,
                color="black",
                label=f"Obstacle Point {i+1}",
            )

        plt.scatter(x_pos, y_pos, 25, linewidths=0.03, color="blue", alpha=0.2)
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(which="major", linewidth=0.7)
        plt.grid(which="minor", linewidth=0.5)
        plt.minorticks_on()
        if save:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.gcf().savefig(output_dir / filename, dpi=300, bbox_inches="tight")


# Backward-compatible name alias.
continuumEnv = ContinuumEnv
