"""Hydra-dispatched Gradio demo for interactive continuum robot simulation."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from continuum_rl.env import ContinuumEnv, DEFAULT_OBSTACLES
from continuum_rl.gym_compat import unpack_step_output
from kinematics.forward_velocity_kinematics import coupletransformations, trans_mat_cc


PolicyFn = Callable[[np.ndarray], np.ndarray]


_RUN_LOCK = threading.Lock()
_CANCEL_EVENT = threading.Event()

DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "initial_kappa": (0.0, 0.0, 0.0),
        "goal_xy": (-0.2, 0.15),
        "obstacles": tuple((o["x"], o["y"]) for o in DEFAULT_OBSTACLES),
    },
    "narrow_passage": {
        "initial_kappa": (2.0, 2.0, 2.0),
        "goal_xy": (-0.2, 0.16),
        "obstacles": ((-0.18, 0.12), (-0.2, 0.06), (-0.15, 0.18)),
    },
    "left_arc": {
        "initial_kappa": (6.0, 2.0, -1.0),
        "goal_xy": (-0.24, 0.2),
        "obstacles": ((-0.16, 0.22), (-0.22, 0.02), (-0.16, 0.08)),
    },
}


@dataclass(frozen=True)
class GradioRuntimeConfig:
    framework: str
    control_mode: str
    goal_type: str
    reward_function: str
    checkpoint_actor: str
    checkpoint_critic: Optional[str]
    device: str
    max_steps: int
    seed: int
    initial_kappa: tuple[float, float, float]
    fixed_goal_xy: tuple[float, float]
    manual_action: tuple[float, float, float]
    output_dir: str
    save_outputs: bool
    save_animation: bool
    animation_format: str
    share: bool
    server_name: str
    server_port: int
    single_run_lock: bool
    show_progress: bool
    env_dt: float
    env_delta_kappa: float
    env_l: tuple[float, float, float]
    env_obstacles: tuple[dict[str, float], ...]


@dataclass
class SimulationResult:
    status: str
    framework: str
    control_mode: str
    device_used: str
    goal_type: str
    reward_function: str
    steps: int
    terminated: bool
    truncated: bool
    cancelled: bool
    total_reward: float
    final_error: float
    initial_kappa: tuple[float, float, float]
    final_kappa: tuple[float, float, float]
    segment_lengths: tuple[float, float, float]
    goal_xy: tuple[float, float]
    obstacles: tuple[tuple[float, float], ...]
    positions: np.ndarray
    kappas: np.ndarray
    rewards: np.ndarray
    errors: np.ndarray
    actions: np.ndarray
    workspace_fig_path: Optional[str] = None
    diagnostics_fig_path: Optional[str] = None
    animation_path: Optional[str] = None
    run_dir: Optional[str] = None

    def summary_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "framework": self.framework,
            "control_mode": self.control_mode,
            "device_used": self.device_used,
            "goal_type": self.goal_type,
            "reward_function": self.reward_function,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "cancelled": self.cancelled,
            "total_reward": self.total_reward,
            "final_error": self.final_error,
            "initial_kappa": list(self.initial_kappa),
            "final_kappa": list(self.final_kappa),
            "goal_xy": list(self.goal_xy),
            "segment_lengths": list(self.segment_lengths),
            "obstacles": [list(o) for o in self.obstacles],
            "run_dir": self.run_dir,
            "workspace_fig_path": self.workspace_fig_path,
            "diagnostics_fig_path": self.diagnostics_fig_path,
            "animation_path": self.animation_path,
        }
        return payload


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent.parent / p).resolve()


def _validate_triplet(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values.")
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr.tolist()}")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _validate_goal(goal_xy: Sequence[float]) -> tuple[float, float]:
    if len(goal_xy) != 2:
        raise ValueError("goal_xy must contain exactly 2 values.")
    arr = np.asarray(goal_xy, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"goal_xy contains non-finite values: {arr.tolist()}")
    return float(arr[0]), float(arr[1])


def _normalize_obstacles(raw_obstacles: Any) -> tuple[dict[str, float], ...]:
    if raw_obstacles is None:
        return tuple(dict(o) for o in DEFAULT_OBSTACLES)

    parsed: list[dict[str, float]] = []
    # Gradio Dataframe may return a pandas DataFrame.
    if hasattr(raw_obstacles, "to_dict"):
        try:
            raw_obstacles = raw_obstacles.to_dict(orient="records")
        except TypeError:
            raw_obstacles = raw_obstacles.to_dict()

    # Gradio may return numpy-backed tables.
    if isinstance(raw_obstacles, np.ndarray):
        raw_obstacles = raw_obstacles.tolist()

    if isinstance(raw_obstacles, str):
        try:
            raw_obstacles = json.loads(raw_obstacles)
        except json.JSONDecodeError as exc:
            raise ValueError("obstacles must be valid JSON list or table rows.") from exc

    if isinstance(raw_obstacles, dict):
        # Support dict-of-columns shape, e.g. {"x":[...], "y":[...]}.
        if "x" in raw_obstacles and "y" in raw_obstacles and (
            isinstance(raw_obstacles["x"], (list, tuple, np.ndarray))
            or isinstance(raw_obstacles["y"], (list, tuple, np.ndarray))
        ):
            x_vals = list(raw_obstacles["x"])
            y_vals = list(raw_obstacles["y"])
            if len(x_vals) != len(y_vals):
                raise ValueError("obstacles column arrays must have equal length.")
            raw_obstacles = [{"x": x_val, "y": y_val} for x_val, y_val in zip(x_vals, y_vals)]
        else:
            raw_obstacles = [raw_obstacles]

    if not isinstance(raw_obstacles, (list, tuple)):
        raise ValueError("obstacles must be a list/tuple.")

    for row in raw_obstacles:
        if row is None:
            continue
        if isinstance(row, dict):
            if "x" not in row or "y" not in row:
                raise ValueError("Each obstacle dict must include 'x' and 'y'.")
            if row["x"] in (None, "") and row["y"] in (None, ""):
                continue
            x_val = float(row["x"])
            y_val = float(row["y"])
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            if row[0] in (None, "") and row[1] in (None, ""):
                continue
            x_val = float(row[0])
            y_val = float(row[1])
        else:
            raise ValueError(f"Invalid obstacle row: {row}")
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            raise ValueError("Obstacle coordinates must be finite.")
        parsed.append({"x": x_val, "y": y_val})

    if not parsed:
        raise ValueError("At least one obstacle must be provided.")
    return tuple(parsed)


def _validate_runtime_config(cfg: GradioRuntimeConfig) -> None:
    if cfg.framework not in {"pytorch", "keras"}:
        raise ValueError("framework must be one of: pytorch, keras.")
    if cfg.control_mode not in {"policy", "manual"}:
        raise ValueError("control_mode must be one of: policy, manual.")
    if cfg.goal_type not in {"fixed_goal", "random_goal"}:
        raise ValueError("goal_type must be one of: fixed_goal, random_goal.")
    if cfg.device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be one of: auto, cpu, gpu.")
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be > 0.")
    if cfg.env_dt <= 0 or cfg.env_delta_kappa <= 0:
        raise ValueError("env_dt and env_delta_kappa must be > 0.")
    _validate_triplet(cfg.initial_kappa, "initial_kappa")
    _validate_triplet(cfg.manual_action, "manual_action")
    _validate_triplet(cfg.env_l, "env_l")
    _validate_goal(cfg.fixed_goal_xy)
    _normalize_obstacles(cfg.env_obstacles)
    if cfg.animation_format not in {"gif", "mp4"}:
        raise ValueError("animation_format must be gif or mp4.")
    if cfg.server_port <= 0:
        raise ValueError("server_port must be > 0.")


def _torch_policy(
    actor_checkpoint: Path,
    env: ContinuumEnv,
    reward_function: str,
    goal_type: str,
    device_pref: str,
) -> tuple[PolicyFn, str]:
    import torch

    from Pytorch.ddpg import validate_checkpoint_compatibility
    from Pytorch.model import Actor

    expected = {
        "state_dim": env.obs_size,
        "obstacle_count": env.num_obstacles,
        "goal_type": goal_type,
        "reward_function": reward_function,
    }
    validate_checkpoint_compatibility(actor_checkpoint, expected)

    if device_pref == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            raise RuntimeError("Requested device='gpu' but CUDA is unavailable.")
    elif device_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Actor(state_size=env.obs_size, action_size=3, seed=0).to(device)
    state_dict = torch.load(actor_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    def _policy(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_obs = torch.from_numpy(obs).float().to(device)
            action = model(tensor_obs).detach().cpu().numpy()
        return np.clip(action.astype(np.float32), -1.0, 1.0)

    return _policy, str(device)


def _keras_policy(
    actor_checkpoint: Path,
    env: ContinuumEnv,
    reward_function: str,
    goal_type: str,
    device_pref: str,
) -> tuple[PolicyFn, str]:
    import tensorflow as tf

    from Keras.DDPG import get_actor, validate_checkpoint_compatibility

    expected = {
        "state_dim": env.obs_size,
        "obstacle_count": env.num_obstacles,
        "goal_type": goal_type,
        "reward_function": reward_function,
    }
    validate_checkpoint_compatibility(actor_checkpoint, expected)

    if not actor_checkpoint.exists() and actor_checkpoint.suffix == ".h5":
        alt = actor_checkpoint.with_name(actor_checkpoint.stem + ".weights.h5")
        if alt.exists():
            actor_checkpoint = alt
    if not actor_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {actor_checkpoint}")

    if device_pref == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    actor_model = get_actor(env.obs_size, env.action_space.shape[0], 1.0)
    actor_model.load_weights(actor_checkpoint)
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    device_used = "gpu" if (device_pref != "cpu" and gpu_count > 0) else "cpu"

    def _policy(obs: np.ndarray) -> np.ndarray:
        tf_obs = tf.expand_dims(tf.convert_to_tensor(obs), 0)
        action = tf.squeeze(actor_model(tf_obs)).numpy().astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    return _policy, device_used


def _manual_policy(action_triplet: tuple[float, float, float]) -> PolicyFn:
    action = np.asarray(action_triplet, dtype=np.float32)
    action = np.clip(action, -1.0, 1.0)

    def _policy(_obs: np.ndarray) -> np.ndarray:
        return action.copy()

    return _policy


def _compute_sections(
    kappa: Sequence[float],
    segment_lengths: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T1_cc = trans_mat_cc(float(kappa[0]), float(segment_lengths[0]))
    T1_tip = np.reshape(T1_cc[len(T1_cc) - 1, :], (4, 4), order="F")
    T2 = trans_mat_cc(float(kappa[1]), float(segment_lengths[1]))
    T2_cc = coupletransformations(T2, T1_tip)
    T2_tip = np.reshape(T2_cc[len(T2_cc) - 1, :], (4, 4), order="F")
    T3 = trans_mat_cc(float(kappa[2]), float(segment_lengths[2]))
    T3_cc = coupletransformations(T3, T2_tip)
    return T1_cc, T2_cc, T3_cc


def _workspace_figure(result: SimulationResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([-0.025, 0.025], [0.0, 0.0], "black", linewidth=4)

    T1_i, T2_i, T3_i = _compute_sections(result.initial_kappa, result.segment_lengths)
    ax.plot(T1_i[:, 12], T1_i[:, 13], color="#7aa2f7", alpha=0.35, linewidth=2)
    ax.plot(T2_i[:, 12], T2_i[:, 13], color="#7aa2f7", alpha=0.35, linewidth=2)
    ax.plot(T3_i[:, 12], T3_i[:, 13], color="#7aa2f7", alpha=0.35, linewidth=2)
    ax.scatter(T3_i[-1, 12], T3_i[-1, 13], color="#f59e0b", s=60, label="initial tip")

    T1_f, T2_f, T3_f = _compute_sections(result.final_kappa, result.segment_lengths)
    ax.plot(T1_f[:, 12], T1_f[:, 13], color="#2563eb", linewidth=3)
    ax.plot(T2_f[:, 12], T2_f[:, 13], color="#2563eb", linewidth=3)
    ax.plot(T3_f[:, 12], T3_f[:, 13], color="#2563eb", linewidth=3)
    ax.scatter(T3_f[-1, 12], T3_f[-1, 13], color="black", s=60, label="final tip")

    if result.positions.size > 0:
        ax.plot(result.positions[:, 0], result.positions[:, 1], color="#1d4ed8", alpha=0.45, linewidth=2, label="trajectory")

    if result.obstacles:
        obs = np.asarray(result.obstacles, dtype=np.float64)
        ax.scatter(obs[:, 0], obs[:, 1], marker="x", color="red", s=80, linewidths=2.0, label="obstacles")

    ax.scatter(result.goal_xy[0], result.goal_xy[1], marker="x", color="darkred", s=120, linewidths=2.5, label="goal")
    ax.set_title("Continuum Robot Workspace")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.4, 0.4])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def _diagnostics_figure(result: SimulationResult) -> plt.Figure:
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    steps = np.arange(len(result.rewards))
    if len(steps) == 0:
        axs[0].set_title("No rollout steps available.")
        fig.tight_layout()
        return fig

    axs[0].plot(steps, result.kappas[1:, 0], label="kappa1", linewidth=2)
    axs[0].plot(steps, result.kappas[1:, 1], label="kappa2", linewidth=2)
    axs[0].plot(steps, result.kappas[1:, 2], label="kappa3", linewidth=2)
    axs[0].set_ylabel("Curvature")
    axs[0].legend(loc="upper right")
    axs[0].grid(alpha=0.25)

    axs[1].plot(steps, result.errors, color="#d97706", linewidth=2)
    axs[1].set_ylabel("Distance To Goal")
    axs[1].grid(alpha=0.25)

    axs[2].plot(steps, result.rewards, color="#7c3aed", linewidth=2)
    axs[2].set_ylabel("Reward")
    axs[2].set_xlabel("Step")
    axs[2].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _save_animation(result: SimulationResult, output_path: Path, fmt: str) -> Path:
    if result.kappas.shape[0] == 0:
        raise RuntimeError("Cannot save animation without kappa trajectory.")

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    obs = np.asarray(result.obstacles, dtype=np.float64) if result.obstacles else np.zeros((0, 2), dtype=np.float64)

    def _update(frame_idx: int):
        ax.cla()
        ax.plot([-0.025, 0.025], [0.0, 0.0], "black", linewidth=4)
        kappa = result.kappas[min(frame_idx, result.kappas.shape[0] - 1)]
        T1, T2, T3 = _compute_sections(kappa, result.segment_lengths)
        ax.plot(T1[:, 12], T1[:, 13], color="#2563eb", linewidth=3)
        ax.plot(T2[:, 12], T2[:, 13], color="#2563eb", linewidth=3)
        ax.plot(T3[:, 12], T3[:, 13], color="#2563eb", linewidth=3)
        if frame_idx > 0 and result.positions.size > 0:
            traj = result.positions[: frame_idx + 1]
            ax.plot(traj[:, 0], traj[:, 1], color="#1d4ed8", alpha=0.45, linewidth=2)
        if obs.size > 0:
            ax.scatter(obs[:, 0], obs[:, 1], marker="x", color="red", s=80, linewidths=2.0)
        ax.scatter(result.goal_xy[0], result.goal_xy[1], marker="x", color="darkred", s=120, linewidths=2.5)
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        ax.set_title(f"Rollout step {frame_idx}")
        ax.grid(alpha=0.25)

    anim = FuncAnimation(fig, _update, frames=max(result.kappas.shape[0], 1), interval=40)
    if fmt == "mp4":
        anim.save(output_path, dpi=120)
    else:
        anim.save(output_path, dpi=120, writer="pillow")
    plt.close(fig)
    return output_path


def run_simulation(
    cfg: GradioRuntimeConfig,
    *,
    obstacles_override: Any = None,
    initial_kappa_override: Optional[Sequence[float]] = None,
    goal_xy_override: Optional[Sequence[float]] = None,
    manual_action_override: Optional[Sequence[float]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> SimulationResult:
    _validate_runtime_config(cfg)
    obstacles = _normalize_obstacles(obstacles_override if obstacles_override is not None else cfg.env_obstacles)
    initial_kappa = _validate_triplet(
        initial_kappa_override if initial_kappa_override is not None else cfg.initial_kappa,
        "initial_kappa",
    )
    goal_xy = _validate_goal(goal_xy_override if goal_xy_override is not None else cfg.fixed_goal_xy)
    manual_action = _validate_triplet(
        manual_action_override if manual_action_override is not None else cfg.manual_action,
        "manual_action",
    )

    env = ContinuumEnv(
        observation_mode="canonical",
        goal_type=cfg.goal_type,  # type: ignore[arg-type]
        max_episode_steps=cfg.max_steps,
        obstacles=obstacles,
        dt=cfg.env_dt,
        delta_kappa=cfg.env_delta_kappa,
        l=cfg.env_l,
    )

    if cfg.control_mode == "policy":
        actor_path = _resolve_path(cfg.checkpoint_actor)
        if cfg.framework == "pytorch":
            policy, device_used = _torch_policy(
                actor_checkpoint=actor_path,
                env=env,
                reward_function=cfg.reward_function,
                goal_type=cfg.goal_type,
                device_pref=cfg.device,
            )
        else:
            policy, device_used = _keras_policy(
                actor_checkpoint=actor_path,
                env=env,
                reward_function=cfg.reward_function,
                goal_type=cfg.goal_type,
                device_pref=cfg.device,
            )
    else:
        policy = _manual_policy(manual_action)
        device_used = "cpu"

    reset_options: dict[str, Any] = {"initial_kappa": list(initial_kappa)}
    if cfg.goal_type == "fixed_goal":
        reset_options["goal_xy"] = list(goal_xy)
    state, _ = env.reset(seed=int(cfg.seed), options=reset_options)

    positions = [np.asarray(state[:2], dtype=np.float64)]
    kappas = [np.asarray(state[4:7], dtype=np.float64)]
    rewards: list[float] = []
    errors: list[float] = []
    actions: list[np.ndarray] = []
    terminated = False
    truncated = False
    cancelled = False
    total_reward = 0.0

    cancel = cancel_event or _CANCEL_EVENT
    for step_idx in range(cfg.max_steps):
        if cancel.is_set():
            cancelled = True
            break
        action = policy(np.asarray(state, dtype=np.float32))
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        step_out = unpack_step_output(env.step(action, reward_function=cfg.reward_function))
        state = step_out.obs
        actions.append(action.astype(np.float64))
        positions.append(np.asarray(state[:2], dtype=np.float64))
        kappas.append(np.asarray(state[4:7], dtype=np.float64))
        rewards.append(float(step_out.reward))
        errors.append(float(step_out.info.get("error", np.nan)))
        total_reward += float(step_out.reward)

        if progress_cb is not None:
            progress_cb(step_idx + 1, cfg.max_steps)

        if step_out.terminated or step_out.truncated:
            terminated = bool(step_out.terminated)
            truncated = bool(step_out.truncated)
            break

    final_kappa = tuple(float(v) for v in kappas[-1]) if kappas else initial_kappa
    status = "cancelled" if cancelled else "completed"
    if terminated:
        status = "terminated_success"
    elif truncated:
        status = "truncated"

    result = SimulationResult(
        status=status,
        framework=cfg.framework,
        control_mode=cfg.control_mode,
        device_used=device_used,
        goal_type=cfg.goal_type,
        reward_function=cfg.reward_function,
        steps=len(rewards),
        terminated=terminated,
        truncated=truncated,
        cancelled=cancelled,
        total_reward=float(total_reward),
        final_error=float(errors[-1]) if errors else float("nan"),
        initial_kappa=initial_kappa,
        final_kappa=final_kappa,
        segment_lengths=tuple(float(v) for v in cfg.env_l),
        goal_xy=(float(state[2]), float(state[3])),
        obstacles=tuple((float(o["x"]), float(o["y"])) for o in obstacles),
        positions=np.asarray(positions, dtype=np.float64),
        kappas=np.asarray(kappas, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        errors=np.asarray(errors, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 3), dtype=np.float64),
    )
    return result


def persist_result(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    save_animation: bool,
    animation_format: str,
) -> SimulationResult:
    output_root = _resolve_path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"gradio_demo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    workspace_fig = _workspace_figure(result)
    workspace_path = run_dir / "workspace.jpeg"
    workspace_fig.savefig(workspace_path, dpi=220, bbox_inches="tight")
    plt.close(workspace_fig)

    diagnostics_fig = _diagnostics_figure(result)
    diagnostics_path = run_dir / "diagnostics.jpeg"
    diagnostics_fig.savefig(diagnostics_path, dpi=220, bbox_inches="tight")
    plt.close(diagnostics_fig)

    animation_path: Optional[Path] = None
    if save_animation:
        animation_path = run_dir / f"rollout.{animation_format}"
        _save_animation(result, animation_path, animation_format)

    result.workspace_fig_path = str(workspace_path)
    result.diagnostics_fig_path = str(diagnostics_path)
    result.animation_path = str(animation_path) if animation_path is not None else None
    result.run_dir = str(run_dir)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(result.summary_dict(), indent=2), encoding="utf-8")
    return result


def _build_runtime_config(task: Any, env_cfg: Any) -> GradioRuntimeConfig:
    return GradioRuntimeConfig(
        framework=str(task.framework),
        control_mode=str(task.control_mode),
        goal_type=str(task.goal_type),
        reward_function=str(task.reward_function),
        checkpoint_actor=str(task.checkpoint_actor),
        checkpoint_critic=None if task.checkpoint_critic is None else str(task.checkpoint_critic),
        device=str(task.device),
        max_steps=int(task.max_steps),
        seed=int(task.seed),
        initial_kappa=tuple(float(v) for v in task.initial_kappa),
        fixed_goal_xy=tuple(float(v) for v in task.fixed_goal_xy),
        manual_action=tuple(float(v) for v in task.manual_action),
        output_dir=str(task.output_dir),
        save_outputs=bool(task.save_outputs),
        save_animation=bool(task.save_animation),
        animation_format=str(task.animation_format),
        share=bool(task.share),
        server_name=str(task.server_name),
        server_port=int(task.server_port),
        single_run_lock=bool(task.single_run_lock),
        show_progress=bool(task.show_progress),
        env_dt=float(env_cfg.dt),
        env_delta_kappa=float(env_cfg.delta_kappa),
        env_l=tuple(float(v) for v in env_cfg.l),
        env_obstacles=tuple({"x": float(o.x), "y": float(o.y)} for o in env_cfg.obstacles),
    )


def request_cancel() -> str:
    _CANCEL_EVENT.set()
    return "Cancellation requested."


def launch_gradio_demo(task: Any, env_cfg: Any) -> None:
    cfg = _build_runtime_config(task=task, env_cfg=env_cfg)
    _validate_runtime_config(cfg)

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Install UI extras with `pip install -e .[ui]`."
        ) from exc

    default_obstacles = [[o["x"], o["y"]] for o in cfg.env_obstacles]

    def _apply_preset(name: str):
        preset = DEFAULT_PRESETS.get(name, DEFAULT_PRESETS["default"])
        return (
            float(preset["initial_kappa"][0]),
            float(preset["initial_kappa"][1]),
            float(preset["initial_kappa"][2]),
            float(preset["goal_xy"][0]),
            float(preset["goal_xy"][1]),
            [[float(x), float(y)] for x, y in preset["obstacles"]],
        )

    def _run_ui(
        framework: str,
        control_mode: str,
        goal_type: str,
        reward_function: str,
        checkpoint_actor: str,
        checkpoint_critic: str,
        device: str,
        max_steps: int,
        seed: int,
        k1: float,
        k2: float,
        k3: float,
        goal_x: float,
        goal_y: float,
        m1: float,
        m2: float,
        m3: float,
        obstacles_table: Any,
        save_outputs: bool,
        save_animation: bool,
        animation_format: str,
        output_dir: str,
        single_run_lock: bool,
        show_progress: bool,
        progress=gr.Progress(),
    ):
        if single_run_lock and not _RUN_LOCK.acquire(blocking=False):
            return (
                "Another simulation is currently running. Wait or cancel it.",
                {"status": "busy"},
                None,
                None,
                "",
            )

        _CANCEL_EVENT.clear()
        try:
            runtime = GradioRuntimeConfig(
                framework=framework,
                control_mode=control_mode,
                goal_type=goal_type,
                reward_function=reward_function,
                checkpoint_actor=checkpoint_actor,
                checkpoint_critic=checkpoint_critic or None,
                device=device,
                max_steps=int(max_steps),
                seed=int(seed),
                initial_kappa=(float(k1), float(k2), float(k3)),
                fixed_goal_xy=(float(goal_x), float(goal_y)),
                manual_action=(float(m1), float(m2), float(m3)),
                output_dir=output_dir,
                save_outputs=bool(save_outputs),
                save_animation=bool(save_animation),
                animation_format=animation_format,
                share=cfg.share,
                server_name=cfg.server_name,
                server_port=cfg.server_port,
                single_run_lock=bool(single_run_lock),
                show_progress=bool(show_progress),
                env_dt=cfg.env_dt,
                env_delta_kappa=cfg.env_delta_kappa,
                env_l=cfg.env_l,
                env_obstacles=cfg.env_obstacles,
            )
            _validate_runtime_config(runtime)

            def _progress_cb(done: int, total: int) -> None:
                if runtime.show_progress:
                    progress(float(done) / float(max(total, 1)), desc=f"Running step {done}/{total}")

            result = run_simulation(
                cfg=runtime,
                obstacles_override=obstacles_table,
                progress_cb=_progress_cb,
                cancel_event=_CANCEL_EVENT,
            )
            workspace_fig = _workspace_figure(result)
            diagnostics_fig = _diagnostics_figure(result)

            artifacts_path = ""
            if runtime.save_outputs:
                result = persist_result(
                    result=result,
                    output_dir=runtime.output_dir,
                    save_animation=runtime.save_animation,
                    animation_format=runtime.animation_format,
                )
                artifacts_path = result.run_dir or ""

            status = (
                f"status={result.status}, framework={result.framework}, control={result.control_mode}, "
                f"device={result.device_used}, steps={result.steps}, reward={result.total_reward:.4f}, "
                f"error={result.final_error:.6f}"
            )
            return status, result.summary_dict(), workspace_fig, diagnostics_fig, artifacts_path
        except Exception as exc:
            return f"Error: {exc}", {"status": "error", "message": str(exc)}, None, None, ""
        finally:
            if single_run_lock and _RUN_LOCK.locked():
                _RUN_LOCK.release()

    with gr.Blocks(title="Continuum Robot Gradio Demo") as demo:
        gr.Markdown("## Continuum Robot Interactive Demo")
        gr.Markdown(
            "Set initial kappa values, goal and obstacles, then run policy or manual control rollout. "
            "Use Cancel to stop long runs."
        )

        with gr.Row():
            preset_name = gr.Dropdown(
                choices=list(DEFAULT_PRESETS.keys()),
                value="default",
                label="Scene Preset",
            )
            apply_preset_btn = gr.Button("Apply Preset")

        with gr.Row():
            framework = gr.Dropdown(["pytorch", "keras"], value=cfg.framework, label="Framework")
            control_mode = gr.Dropdown(["policy", "manual"], value=cfg.control_mode, label="Control Mode")
            goal_type = gr.Dropdown(["fixed_goal", "random_goal"], value=cfg.goal_type, label="Goal Type")
            device = gr.Dropdown(["auto", "cpu", "gpu"], value=cfg.device, label="Device")

        with gr.Row():
            checkpoint_actor = gr.Textbox(value=cfg.checkpoint_actor, label="Actor Checkpoint")
            checkpoint_critic = gr.Textbox(
                value="" if cfg.checkpoint_critic is None else cfg.checkpoint_critic,
                label="Critic Checkpoint (PyTorch policy mode)",
            )

        with gr.Row():
            reward_function = gr.Dropdown(
                [
                    "step_error_comparison",
                    "step_minus_euclidean_square",
                    "step_minus_weighted_euclidean",
                    "step_distance_based",
                ],
                value=cfg.reward_function,
                label="Reward Function",
            )
            max_steps = gr.Slider(minimum=1, maximum=2000, value=cfg.max_steps, step=1, label="Max Steps")
            seed = gr.Number(value=cfg.seed, precision=0, label="Seed")

        with gr.Row():
            kappa1 = gr.Number(value=cfg.initial_kappa[0], label="Initial kappa1")
            kappa2 = gr.Number(value=cfg.initial_kappa[1], label="Initial kappa2")
            kappa3 = gr.Number(value=cfg.initial_kappa[2], label="Initial kappa3")

        with gr.Row():
            goal_x = gr.Number(value=cfg.fixed_goal_xy[0], label="Goal X (fixed mode)")
            goal_y = gr.Number(value=cfg.fixed_goal_xy[1], label="Goal Y (fixed mode)")

        with gr.Row():
            manual1 = gr.Slider(-1.0, 1.0, value=cfg.manual_action[0], step=0.01, label="Manual action 1")
            manual2 = gr.Slider(-1.0, 1.0, value=cfg.manual_action[1], step=0.01, label="Manual action 2")
            manual3 = gr.Slider(-1.0, 1.0, value=cfg.manual_action[2], step=0.01, label="Manual action 3")

        obstacles_df = gr.Dataframe(
            headers=["x", "y"],
            datatype=["number", "number"],
            value=default_obstacles,
            row_count=(len(default_obstacles), "dynamic"),
            col_count=(2, "fixed"),
            label="Obstacles",
        )

        with gr.Row():
            save_outputs = gr.Checkbox(value=cfg.save_outputs, label="Save Outputs")
            save_animation = gr.Checkbox(value=cfg.save_animation, label="Save Animation")
            animation_format = gr.Dropdown(["gif", "mp4"], value=cfg.animation_format, label="Animation Format")
            single_run_lock = gr.Checkbox(value=cfg.single_run_lock, label="Single Run Lock")
            show_progress = gr.Checkbox(value=cfg.show_progress, label="Show Progress")
            output_dir = gr.Textbox(value=cfg.output_dir, label="Output Directory")

        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            cancel_btn = gr.Button("Cancel")

        status_box = gr.Textbox(label="Status")
        summary_json = gr.JSON(label="Run Summary")
        workspace_plot = gr.Plot(label="Workspace Plot", format="png")
        diagnostics_plot = gr.Plot(label="Diagnostics Plot", format="png")
        artifact_path = gr.Textbox(label="Artifacts Directory")

        apply_preset_btn.click(
            _apply_preset,
            inputs=[preset_name],
            outputs=[kappa1, kappa2, kappa3, goal_x, goal_y, obstacles_df],
        )
        cancel_btn.click(lambda: request_cancel(), outputs=[status_box])

        start_btn.click(
            _run_ui,
            inputs=[
                framework,
                control_mode,
                goal_type,
                reward_function,
                checkpoint_actor,
                checkpoint_critic,
                device,
                max_steps,
                seed,
                kappa1,
                kappa2,
                kappa3,
                goal_x,
                goal_y,
                manual1,
                manual2,
                manual3,
                obstacles_df,
                save_outputs,
                save_animation,
                animation_format,
                output_dir,
                single_run_lock,
                show_progress,
            ],
            outputs=[status_box, summary_json, workspace_plot, diagnostics_plot, artifact_path],
        )

    demo.queue()
    demo.launch(
        server_name=cfg.server_name,
        server_port=cfg.server_port,
        share=cfg.share,
    )


__all__ = [
    "DEFAULT_PRESETS",
    "GradioRuntimeConfig",
    "SimulationResult",
    "launch_gradio_demo",
    "persist_result",
    "request_cancel",
    "run_simulation",
]
