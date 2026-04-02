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

ObservationMode = Literal["canonical", "legacy4d"]
GoalType = Literal["fixed_goal", "random_goal"]


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


class ContinuumEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        obstacles: Iterable[dict[str, float]] | None = None,
        observation_mode: ObservationMode = "canonical",
        goal_type: GoalType = "fixed_goal",
        fixed_goal_kappa: tuple[float, float, float] = (6.2, 6.2, 6.2),
        random_goal_range: tuple[float, float] = (-4.0, 16.0),
    ):
        if observation_mode not in {"canonical", "legacy4d"}:
            raise ValueError(f"Unsupported observation_mode={observation_mode}.")
        if goal_type not in {"fixed_goal", "random_goal"}:
            raise ValueError(f"Unsupported goal_type={goal_type}.")

        self.config = EnvConfig(
            observation_mode=observation_mode,
            goal_type=goal_type,
            fixed_goal_kappa=fixed_goal_kappa,
            random_goal_range=random_goal_range,
        )

        self.delta_kappa = 0.001
        self.kappa_dot_max = 1.0
        self.kappa_max = 16.0
        self.kappa_min = -4.0

        self.l = [0.1, 0.1, 0.1]
        self.dt = 5e-2
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

        self.obs_size = 4 + (2 * self.num_obstacles) if self.config.observation_mode == "canonical" else 4
        obs_bound = 0.5
        self.observation_space = spaces.Box(
            low=-obs_bound,
            high=obs_bound,
            shape=(self.obs_size,),
            dtype=np.float32,
        )
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

        self._full_state = np.zeros(4 + 2 * self.num_obstacles, dtype=np.float32)
        self.last_u = np.zeros(3, dtype=np.float32)
        self.initial_distance: float | None = None

    def _select_observation(self, full_state: np.ndarray) -> np.ndarray:
        if self.config.observation_mode == "legacy4d":
            return full_state[:4].astype(np.float32)
        return full_state.astype(np.float32)

    def _compose_full_state(self, x: float, y: float, goal_x: float, goal_y: float) -> np.ndarray:
        state = [x, y, goal_x, goal_y]
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
        x, y, goal_x, goal_y = self._full_state[:4]
        u = np.clip(np.asarray(action, dtype=np.float32), -self.kappa_dot_max, self.kappa_dot_max)
        costs = self._compute_reward_cost(x, y, goal_x, goal_y, reward_function=reward_function)

        if reward_function == "step_minus_euclidean_square":
            terminated = bool(math.sqrt(max(costs, 0.0)) <= 0.005)
            reward = -costs
        else:
            terminated = bool(self.error <= 0.005)
            reward = -costs if reward_function == "step_minus_weighted_euclidean" else costs

        self.J = jacobian_matrix(self.delta_kappa, self.kappa1, self.kappa2, self.kappa3, self.l)
        control = u.copy()
        if self.stop == 1:
            control = np.array([0.0, u[1], u[2]], dtype=np.float32)
        elif self.stop == 2:
            control = np.array([u[0], 0.0, u[2]], dtype=np.float32)
        elif self.stop == 3:
            control = np.array([u[0], u[1], 0.0], dtype=np.float32)
        elif self.stop == 4:
            control = np.array([0.0, 0.0, u[2]], dtype=np.float32)
        elif self.stop == 5:
            control = np.array([0.0, u[1], 0.0], dtype=np.float32)
        elif self.stop == 6:
            control = np.array([u[0], 0.0, 0.0], dtype=np.float32)
        elif self.stop == 7:
            control = np.zeros(3, dtype=np.float32)

        x_vel = self.J @ control
        new_x = float(x + (x_vel[0] * self.dt))
        new_y = float(y + (x_vel[1] * self.dt))

        self.kappa1 = float(np.clip(self.kappa1 + (u[0] * self.dt), self.kappa_min, self.kappa_max))
        self.kappa2 = float(np.clip(self.kappa2 + (u[1] * self.dt), self.kappa_min, self.kappa_max))
        self.kappa3 = float(np.clip(self.kappa3 + (u[2] * self.dt), self.kappa_min, self.kappa_max))
        self._update_stop_mode()

        if not self.workspace.contains([new_x, new_y]):
            self.overshoot0 += 1
            new_x, new_y = self.workspace.clip([new_x, new_y]).astype(np.float32).tolist()

        if self.workspace.contains([goal_x, goal_y]):
            new_goal_x, new_goal_y = goal_x, goal_y
        else:
            self.overshoot1 += 1
            new_goal_x, new_goal_y = self.workspace.clip([goal_x, goal_y]).astype(np.float32).tolist()

        self._full_state = self._compose_full_state(new_x, new_y, new_goal_x, new_goal_y)
        self.previous_error = self.error
        self.last_u = u

        obs = self._select_observation(self._full_state)
        truncated = False
        info = {
            "error": float(self.error),
            "reward_function": reward_function,
            "observation_mode": self.config.observation_mode,
            "goal_type": self.config.goal_type,
        }
        return obs, float(reward), terminated, truncated, info

    def _sample_goal(self) -> tuple[float, float, float, float, float]:
        if self.config.goal_type == "fixed_goal":
            target_k1, target_k2, target_k3 = self.config.fixed_goal_kappa
        else:
            low, high = self.config.random_goal_range
            target_k1 = float(np.random.uniform(low=low, high=high))
            target_k2 = float(np.random.uniform(low=low, high=high))
            target_k3 = float(np.random.uniform(low=low, high=high))

        T3_target = three_section_planar_robot(target_k1, target_k2, target_k3, self.l)
        goal_x, goal_y = np.array([T3_target[0, 3], T3_target[1, 3]], dtype=np.float32)
        if not self.workspace.contains([goal_x, goal_y]):
            self.overshoot1 += 1
            goal_x, goal_y = self.workspace.clip([goal_x, goal_y]).astype(np.float32)
        return float(target_k1), float(target_k2), float(target_k3), float(goal_x), float(goal_y)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options  # reserved for future use
        self.initial_distance = None

        self.kappa1 = float(np.random.uniform(low=-4, high=16))
        self.kappa2 = float(np.random.uniform(low=-4, high=16))
        self.kappa3 = float(np.random.uniform(low=-4, high=16))
        self._update_stop_mode()

        T3_cc = three_section_planar_robot(self.kappa1, self.kappa2, self.kappa3, self.l)
        x, y = np.array([T3_cc[0, 3], T3_cc[1, 3]], dtype=np.float32)
        if not self.workspace.contains([x, y]):
            x, y = self.workspace.clip([x, y]).astype(np.float32)
            self.overshoot0 += 1

        self.target_k1, self.target_k2, self.target_k3, goal_x, goal_y = self._sample_goal()

        self._full_state = self._compose_full_state(float(x), float(y), goal_x, goal_y)
        self.error = math.sqrt(((goal_x - x) ** 2) + ((goal_y - y) ** 2))
        self.previous_error = self.error
        self.last_u = np.zeros(3, dtype=np.float32)

        obs = self._select_observation(self._full_state)
        info = {"observation_mode": self.config.observation_mode, "goal_type": self.config.goal_type}
        return obs, info

    def reset_legacy(self):
        obs, _ = self.reset()
        return obs

    def step_legacy(self, action: Sequence[float], reward_function: str = "step_minus_euclidean_square"):
        obs, reward, terminated, truncated, info = self.step(action, reward_function=reward_function)
        done = terminated or truncated
        return obs, reward, done, info

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
            obs_idx = 4 + i * 2
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
