from __future__ import annotations

import pytest

from continuum_rl import tracking


def test_wandb_missing_package_fail_open(monkeypatch):
    def _raise_import_error():
        raise ModuleNotFoundError("wandb")

    monkeypatch.setattr(tracking, "_import_wandb", _raise_import_error)
    tracker = tracking.create_wandb_tracker(
        wandb_cfg={"enabled": True, "fail_open_on_init_error": True},
        run_config={"a": 1},
        context={"framework": "pytorch", "task_name": "pytorch_train"},
    )
    assert tracker.active is False


def test_wandb_missing_package_fail_closed(monkeypatch):
    def _raise_import_error():
        raise ModuleNotFoundError("wandb")

    monkeypatch.setattr(tracking, "_import_wandb", _raise_import_error)
    with pytest.raises(ModuleNotFoundError):
        tracking.create_wandb_tracker(
            wandb_cfg={"enabled": True, "fail_open_on_init_error": False},
            run_config={"a": 1},
            context={"framework": "pytorch", "task_name": "pytorch_train"},
        )


def test_wandb_online_without_api_key_falls_back_offline(monkeypatch):
    class _FakeRun:
        pass

    class _FakeWandb:
        def __init__(self):
            self.init_kwargs = {}

        class Settings:
            def __init__(self, _disable_stats=False):
                self.disable = _disable_stats

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return _FakeRun()

        def finish(self):
            return None

    fake = _FakeWandb()
    monkeypatch.setattr(tracking, "_import_wandb", lambda: fake)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tracker_obj = tracking.create_wandb_tracker(
        wandb_cfg={"enabled": True, "mode": "online"},
        run_config={},
        context={"framework": "keras", "task_name": "keras_train"},
    )
    assert tracker_obj.active is True
    assert fake.init_kwargs["mode"] == "offline"


def test_wandb_log_artifact_noop_when_disabled():
    tracker_obj = tracking.WandbTracker(enabled=False)
    tracker_obj.log_metrics({"train/episode_reward": 1.0}, step=1)
    tracker_obj.log_artifact_files(name="x", artifact_type="model", paths=[])
    tracker_obj.finish()
    assert tracker_obj.active is False
