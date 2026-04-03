"""Lightweight training health monitor for RL loss/gradient drift alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LossHealthMonitor:
    enabled: bool = True
    ema_alpha: float = 0.01
    warmup_steps: int = 2_000
    growth_factor: float = 3.0
    actor_loss_min_abs: float = 10.0
    critic_loss_min_abs: float = 2.0
    grad_norm_max: float = 20.0

    def __post_init__(self) -> None:
        self._ema: Dict[str, float] = {}
        self._baseline: Dict[str, float] = {}
        self._emitted: set[str] = set()

    def _update_ema(self, key: str, value: float) -> float:
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = float(value)
        else:
            self._ema[key] = (1.0 - self.ema_alpha) * prev + self.ema_alpha * float(value)
        return self._ema[key]

    def update(
        self,
        *,
        step: int,
        actor_loss: float,
        critic_loss: float,
        actor_grad_norm: float,
        critic_grad_norm: float,
    ) -> Tuple[Dict[str, float], List[str]]:
        if not self.enabled:
            return {}, []

        actor_loss_ema = self._update_ema("actor_loss", actor_loss)
        critic_loss_ema = self._update_ema("critic_loss", critic_loss)
        actor_grad_ema = self._update_ema("actor_grad_norm", actor_grad_norm)
        critic_grad_ema = self._update_ema("critic_grad_norm", critic_grad_norm)

        # Record initial EMA as the baseline reference for drift checks.
        self._baseline.setdefault("actor_loss", actor_loss_ema)
        self._baseline.setdefault("critic_loss", critic_loss_ema)

        alert_actor_loss = 0
        alert_critic_loss = 0
        if step >= self.warmup_steps:
            actor_threshold = max(
                self.actor_loss_min_abs,
                self._baseline["actor_loss"] * self.growth_factor,
            )
            critic_threshold = max(
                self.critic_loss_min_abs,
                self._baseline["critic_loss"] * self.growth_factor,
            )
            alert_actor_loss = int(actor_loss_ema > actor_threshold)
            alert_critic_loss = int(critic_loss_ema > critic_threshold)

        alert_actor_grad = int(actor_grad_ema > self.grad_norm_max)
        alert_critic_grad = int(critic_grad_ema > self.grad_norm_max)

        metrics = {
            "health/actor_loss_ema": float(actor_loss_ema),
            "health/critic_loss_ema": float(critic_loss_ema),
            "health/actor_grad_norm_ema": float(actor_grad_ema),
            "health/critic_grad_norm_ema": float(critic_grad_ema),
            "health/alert_actor_loss_drift": float(alert_actor_loss),
            "health/alert_critic_loss_drift": float(alert_critic_loss),
            "health/alert_actor_grad_explosion": float(alert_actor_grad),
            "health/alert_critic_grad_explosion": float(alert_critic_grad),
        }

        new_messages: List[str] = []
        alert_map = {
            "actor_loss_drift": alert_actor_loss,
            "critic_loss_drift": alert_critic_loss,
            "actor_grad_explosion": alert_actor_grad,
            "critic_grad_explosion": alert_critic_grad,
        }
        for key, state in alert_map.items():
            if state and key not in self._emitted:
                self._emitted.add(key)
                new_messages.append(
                    f"Training health alert at step={step}: {key} "
                    f"(actor_loss_ema={actor_loss_ema:.4f}, "
                    f"critic_loss_ema={critic_loss_ema:.4f}, "
                    f"actor_grad_norm_ema={actor_grad_ema:.4f}, "
                    f"critic_grad_norm_ema={critic_grad_ema:.4f})"
                )
        return metrics, new_messages
