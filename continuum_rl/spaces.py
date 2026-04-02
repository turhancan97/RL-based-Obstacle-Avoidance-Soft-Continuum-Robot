"""Custom task-space geometry utilities used by the continuum robot env."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .gym_compat import spaces


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RL_DIR = REPO_ROOT / "Reinforcement Learning"
DEFAULT_CIRCLES_PATH = LEGACY_RL_DIR / "circles.txt"
DEFAULT_POLYGON_POINTS_PATH = LEGACY_RL_DIR / "task_space.npy"


def load_circles(circles_path: Path | None = None) -> list[dict[str, np.ndarray | float]]:
    path = circles_path or DEFAULT_CIRCLES_PATH
    circles: list[dict[str, np.ndarray | float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            x, y, r = line.strip().split(",")
            circles.append({"center": np.array([float(x), float(y)], dtype=np.float32), "radius": float(r)})
    return circles


class AmorphousSpace(spaces.Space):
    """2D workspace represented as the union of circles."""

    def __init__(self, circles: Iterable[dict[str, np.ndarray | float]] | None = None):
        self.circles = list(circles) if circles is not None else load_circles()
        super().__init__(shape=(2,), dtype=np.float32)

    def sample(self):
        circle = self.circles[np.random.randint(len(self.circles))]
        center = np.asarray(circle["center"], dtype=np.float32)
        radius = float(circle["radius"])
        angle = np.random.uniform(low=0.0, high=2.0 * np.pi)
        distance = np.random.uniform(low=0.0, high=radius)
        x = center[0] + distance * np.cos(angle)
        y = center[1] + distance * np.sin(angle)
        return np.array([x, y], dtype=np.float32)

    def contains(self, point: Sequence[float]) -> bool:  # type: ignore[override]
        arr = np.asarray(point, dtype=np.float32)
        if arr.shape != (2,):
            return False
        for circle in self.circles:
            center = np.asarray(circle["center"], dtype=np.float32)
            radius = float(circle["radius"])
            if np.linalg.norm(arr - center) <= radius:
                return True
        return False

    def clip(self, point: Sequence[float]) -> np.ndarray:
        arr = np.asarray(point, dtype=np.float32)
        if arr.shape != (2,):
            raise ValueError(f"Expected point with shape (2,), got {arr.shape}.")
        if self.contains(arr):
            return arr

        min_distance = float("inf")
        nearest_point: np.ndarray | None = None
        for circle in self.circles:
            center = np.asarray(circle["center"], dtype=np.float32)
            radius = float(circle["radius"])
            direction = arr - center
            norm = np.linalg.norm(direction)
            if norm == 0:
                candidate = center + np.array([radius, 0.0], dtype=np.float32)
            else:
                candidate = center + radius * direction / norm
            dist = np.linalg.norm(arr - candidate)
            if dist < min_distance:
                min_distance = dist
                nearest_point = candidate

        if nearest_point is None:
            raise RuntimeError("Unable to clip point to AmorphousSpace boundary.")
        return nearest_point.astype(np.float32)


class PolygonSpace(spaces.Space):
    """Convex-hull polygon workspace loaded from sample points."""

    def __init__(self, points_path: Path | None = None):
        from scipy.spatial import ConvexHull
        import matplotlib.path as mplt_path

        path = points_path or DEFAULT_POLYGON_POINTS_PATH
        self.points = np.load(path)
        self.hull = ConvexHull(self.points)
        self.polygon = mplt_path.Path(self.points[self.hull.vertices])
        self.bounding_box = self._calculate_bounding_box()
        super().__init__(shape=(2,), dtype=np.float32)

    def _calculate_bounding_box(self) -> tuple[float, float, float, float]:
        min_x, min_y = np.min(self.points, axis=0)
        max_x, max_y = np.max(self.points, axis=0)
        return float(min_x), float(min_y), float(max_x), float(max_y)

    def sample(self):
        min_x, min_y, max_x, max_y = self.bounding_box
        while True:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if self.contains((x, y)):
                return np.array([x, y], dtype=np.float32)

    def contains(self, point: Sequence[float]) -> bool:  # type: ignore[override]
        return bool(self.polygon.contains_point(point))

    def clip(self, point: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(point, dtype=np.float32), self.bounding_box[:2], self.bounding_box[2:])
