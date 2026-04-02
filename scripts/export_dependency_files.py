"""Generate compatibility dependency files from pyproject.toml."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _load_pyproject(pyproject_path: Path) -> dict:
    with pyproject_path.open("rb") as f:
        return tomllib.load(f)


def _get_pinned_dependencies(pyproject_data: dict) -> list[str]:
    deps = pyproject_data.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError("[project].dependencies must be a list.")
    for dep in deps:
        if "==" not in dep:
            raise ValueError(f"Dependency must be exact-pinned with '==': {dep}")
    return deps


def render_requirements(pyproject_path: Path) -> str:
    pyproject_data = _load_pyproject(pyproject_path)
    deps = _get_pinned_dependencies(pyproject_data)
    return "\n".join(deps) + "\n"


def render_environment_yml(pyproject_path: Path) -> str:
    pyproject_data = _load_pyproject(pyproject_path)
    deps = _get_pinned_dependencies(pyproject_data)
    exports = pyproject_data.get("tool", {}).get("continuum_rl", {}).get("exports", {})
    env_name = exports.get("conda_env_name", "continuum-rl")
    python_pin = exports.get("conda_python", "3.9.7")

    lines = [
        f"name: {env_name}",
        "dependencies:",
        f"  - python={python_pin}",
        "  - pip",
        "  - pip:",
    ]
    lines.extend(f"    - {dep}" for dep in deps)
    return "\n".join(lines) + "\n"


def export_files(pyproject_path: Path, requirements_path: Path, environment_path: Path) -> None:
    requirements_path.write_text(render_requirements(pyproject_path), encoding="utf-8")
    environment_path.write_text(render_environment_yml(pyproject_path), encoding="utf-8")


def _check_no_drift(pyproject_path: Path, requirements_path: Path, environment_path: Path) -> bool:
    expected_requirements = render_requirements(pyproject_path)
    expected_environment = render_environment_yml(pyproject_path)
    current_requirements = requirements_path.read_text(encoding="utf-8")
    current_environment = environment_path.read_text(encoding="utf-8")
    return expected_requirements == current_requirements and expected_environment == current_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dependency files from pyproject.toml")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--environment", type=Path, default=Path("environment.yml"))
    parser.add_argument("--check", action="store_true", help="Check for drift without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        if _check_no_drift(args.pyproject, args.requirements, args.environment):
            print("Dependency exports are up to date.")
            return
        raise SystemExit("Dependency export drift detected. Run scripts/export_dependency_files.py")

    export_files(args.pyproject, args.requirements, args.environment)
    print("Updated requirements.txt and environment.yml from pyproject.toml")


if __name__ == "__main__":
    main()
