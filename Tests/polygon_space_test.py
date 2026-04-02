"""Legacy demo script for PolygonSpace sampling/plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from continuum_rl.spaces import PolygonSpace


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    polygon_space = PolygonSpace()
    sampled_points = np.array([polygon_space.sample() for _ in range(100)])

    plt.figure(figsize=(8, 6))
    for simplex in polygon_space.hull.simplices:
        plt.plot(polygon_space.points[simplex, 0], polygon_space.points[simplex, 1], "k-")

    plt.plot(polygon_space.points[:, 0], polygon_space.points[:, 1], "o", markersize=5, label="Polygon Vertices")
    plt.scatter(sampled_points[:, 0], sampled_points[:, 1], alpha=0.5, color="red", label="Sampled Points")
    plt.fill(
        polygon_space.points[polygon_space.hull.vertices, 0],
        polygon_space.points[polygon_space.hull.vertices, 1],
        alpha=0.3,
    )
    plt.title("Polygon Space and Sampled Points")
    plt.legend()
    plt.savefig(output_dir / "polygon_space.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
