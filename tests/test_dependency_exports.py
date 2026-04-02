from __future__ import annotations

from pathlib import Path

from scripts.export_dependency_files import render_environment_yml, render_requirements


def test_requirements_matches_pyproject_export():
    pyproject = Path("pyproject.toml")
    expected = render_requirements(pyproject)
    actual = Path("requirements.txt").read_text(encoding="utf-8")
    assert actual == expected


def test_environment_matches_pyproject_export():
    pyproject = Path("pyproject.toml")
    expected = render_environment_yml(pyproject)
    actual = Path("environment.yml").read_text(encoding="utf-8")
    assert actual == expected
