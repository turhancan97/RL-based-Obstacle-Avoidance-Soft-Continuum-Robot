from __future__ import annotations

from pathlib import Path


def test_keras_resolves_relative_output_dir_to_repo_root():
    from Keras import DDPG as keras_ddpg

    resolved = keras_ddpg._resolve_output_base_dir("Keras")
    expected = (Path(keras_ddpg.__file__).resolve().parents[1] / "Keras").resolve()
    assert resolved == expected


def test_pytorch_resolves_relative_output_dir_to_repo_root():
    from Pytorch import ddpg as pytorch_ddpg

    resolved = pytorch_ddpg._resolve_output_base_dir("Pytorch")
    expected = (Path(pytorch_ddpg.__file__).resolve().parents[1] / "Pytorch").resolve()
    assert resolved == expected
