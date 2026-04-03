"""Shared experiment tracking helpers with graceful W&B fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _import_wandb() -> Any:
    import wandb  # type: ignore

    return wandb


def _sanitize_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(v) for v in value]
    return str(value)


@dataclass
class WandbTracker:
    enabled: bool
    _wandb: Any = None
    _run: Any = None

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._run is not None and self._wandb is not None)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.active:
            return
        payload = _sanitize_payload(dict(metrics))
        try:
            if step is None:
                self._wandb.log(payload)
            else:
                self._wandb.log(payload, step=step)
        except Exception as exc:  # pragma: no cover - best-effort instrumentation
            print(f"W&B warning: failed to log metrics ({exc}).")

    def log_artifact_files(
        self,
        name: str,
        artifact_type: str,
        paths: Sequence[Path],
        metadata: Mapping[str, Any] | None = None,
        aliases: Sequence[str] | None = None,
    ) -> None:
        if not self.active:
            return
        existing_paths = [Path(p) for p in paths if Path(p).exists()]
        if not existing_paths:
            return

        try:
            artifact = self._wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=_sanitize_payload(dict(metadata or {})),
            )
            for p in existing_paths:
                artifact.add_file(str(p))
            self._wandb.log_artifact(artifact, aliases=list(aliases or []))
        except Exception as exc:  # pragma: no cover - best-effort instrumentation
            print(f"W&B warning: failed to upload artifact ({exc}).")

    def finish(self) -> None:
        if not self.active:
            return
        try:
            self._wandb.finish()
        except Exception as exc:  # pragma: no cover - best-effort instrumentation
            print(f"W&B warning: failed to finish run cleanly ({exc}).")


def create_wandb_tracker(
    wandb_cfg: Mapping[str, Any] | None,
    run_config: Mapping[str, Any] | None,
    context: Mapping[str, Any],
) -> WandbTracker:
    cfg = dict(wandb_cfg or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return WandbTracker(enabled=False)

    fail_open = bool(cfg.get("fail_open_on_init_error", True))
    mode = str(cfg.get("mode", "offline"))
    project = str(cfg.get("project", "continuum-rl-obstacle-avoidance"))
    entity = cfg.get("entity")
    group_by_experiment = bool(cfg.get("group_by_experiment", True))
    run_name_template = str(cfg.get("run_name_template", "{framework}-{task_name}-{seed}"))
    log_system_metrics = bool(cfg.get("log_system_metrics", True))

    if mode not in {"offline", "online"}:
        raise ValueError(f"Unsupported wandb.mode={mode}. Expected 'offline' or 'online'.")

    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("W&B warning: WANDB_API_KEY is not set. Falling back to offline mode.")
        mode = "offline"

    framework = str(context.get("framework", "unknown"))
    task_name = str(context.get("task_name", "unknown"))
    goal_type = str(context.get("goal_type", "unknown"))
    reward_id = str(context.get("reward_id", "unknown"))
    seed = context.get("seed", "none")

    group = None
    if group_by_experiment:
        group = f"{framework}-{goal_type}-{reward_id}"

    try:
        run_name = run_name_template.format(
            framework=framework,
            task_name=task_name,
            goal_type=goal_type,
            reward_id=reward_id,
            seed=seed,
        )
    except Exception:
        run_name = f"{framework}-{task_name}-{goal_type}-{reward_id}-seed{seed}"

    try:
        wandb = _import_wandb()
        settings = None
        if not log_system_metrics:
            try:
                settings = wandb.Settings(_disable_stats=True)
            except Exception:
                settings = None

        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            config=_sanitize_payload(dict(run_config or {})),
            group=group,
            name=run_name,
            settings=settings,
            reinit=True,
        )
        if run is None:
            raise RuntimeError("wandb.init returned None")
        return WandbTracker(enabled=True, _wandb=wandb, _run=run)
    except Exception as exc:
        if fail_open:
            print(f"W&B warning: initialization failed ({exc}). Continuing without W&B.")
            return WandbTracker(enabled=False)
        raise
