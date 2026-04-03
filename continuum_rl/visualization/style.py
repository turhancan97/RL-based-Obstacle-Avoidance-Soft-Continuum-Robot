"""Matplotlib style/theme for conference-ready figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FigureTheme:
    figsize: tuple = (10, 6)
    dpi: int = 300
    format: str = "jpeg"
    facecolor: str = "#ffffff"
    grid_color: str = "#d8d8d8"
    palette: tuple = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def apply_conference_theme() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "figure.constrained_layout.use": True,
            "axes.grid": True,
            "grid.alpha": 0.55,
            "grid.color": "#d8d8d8",
            "axes.facecolor": "#ffffff",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
        }
    )


def framework_color_map() -> Dict[str, str]:
    return {
        "pytorch": "#0072B2",
        "keras": "#D55E00",
    }


def save_figure(fig, output_dir: Path, filename_stem: str, image_format: str, show: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = "jpeg" if image_format.lower() in {"jpg", "jpeg"} else image_format.lower()
    out_path = output_dir / f"{filename_stem}.{fmt}"
    fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def placeholder_figure(title: str, text: str):
    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11)
    return fig
