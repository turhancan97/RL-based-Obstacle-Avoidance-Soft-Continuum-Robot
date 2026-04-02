"""Legacy demo script for AmorphousSpace sampling/plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from continuum_rl.spaces import AmorphousSpace


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    space = AmorphousSpace()
    print("Shape of the State: ", space.shape[0] * 2)
    point = space.sample()
    print("Sample Point is ", point)
    print("The point is within the bounds of the space: ", space.contains(point))
    clipped_point = space.clip(point)
    print("Clip the point to the bounds of the space: ", clipped_point)

    fig, ax = plt.subplots()
    for circle in space.circles:
        ax.add_artist(plt.Circle(circle["center"], circle["radius"], fill=False))
    ax.scatter(clipped_point[0], clipped_point[1])
    ax.set_title("Space for the Continuum Robot Consisting of Several Circular Shapes")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(-0.3, 0.2)
    ax.set_ylim(-0.15, 0.3)
    plt.savefig(output_dir / "amorphous_space.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
