from __future__ import annotations

import json


def test_pytorch_run_manifest_writer_creates_json_and_txt(tmp_path):
    from Pytorch.ddpg import _write_run_manifest_files

    payload = {
        "framework": "pytorch",
        "resolved": {"delta_kappa": 0.001, "max_t": 10},
    }
    paths = _write_run_manifest_files(tmp_path, payload)
    assert paths["json"].exists()
    assert paths["txt"].exists()
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["framework"] == "pytorch"
    assert loaded["resolved"]["max_t"] == 10


def test_keras_run_manifest_writer_creates_json_and_txt(tmp_path):
    import pytest

    pytest.importorskip("tensorflow")
    from Keras.DDPG import _write_run_manifest_files

    payload = {
        "framework": "keras",
        "resolved": {"delta_kappa": 0.001, "max_steps": 10},
    }
    paths = _write_run_manifest_files(tmp_path, payload)
    assert paths["json"].exists()
    assert paths["txt"].exists()
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["framework"] == "keras"
    assert loaded["resolved"]["max_steps"] == 10
