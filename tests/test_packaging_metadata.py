from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def test_pyproject_contains_required_contract():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    project = data["project"]
    assert project["requires-python"] == ">=3.9,<3.10"

    scripts = project["scripts"]
    assert scripts["continuum-rl"] == "continuum_rl.hydra_app:main"
    assert scripts["continuum-rl-compat"] == "continuum_rl.cli:main"

    deps = set(project["dependencies"])
    assert "hydra-core==1.3.2" in deps
    assert "omegaconf==2.3.0" in deps

    extras = project["optional-dependencies"]
    assert "wandb==0.16.6" in set(extras["wandb"])
