from __future__ import annotations

from continuum_rl.artifacts import metadata_path_for, read_metadata, write_metadata


def test_metadata_roundtrip(tmp_path):
    artifact = tmp_path / "checkpoint_actor.pth"
    artifact.write_bytes(b"stub")
    metadata = {
        "framework": "pytorch",
        "state_dim": 10,
        "obstacle_count": 3,
        "goal_type": "fixed_goal",
        "reward_function": "step_minus_weighted_euclidean",
    }
    sidecar = write_metadata(artifact, metadata)
    assert sidecar == metadata_path_for(artifact)
    loaded = read_metadata(artifact)
    assert loaded["state_dim"] == 10
    assert loaded["framework"] == "pytorch"
