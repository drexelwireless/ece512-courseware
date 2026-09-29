"""Hexagonal cell drawing (Python port of ``drawCell.m``).

Cells are flat-topped hexagons: the corners sit at angles 0, 60, ..., 300
degrees from the center, and ``radius`` is the center-to-corner distance R.
Adjacent cell centers are therefore sqrt(3)*R apart.
"""

import numpy as np
import matplotlib.pyplot as plt


def hex_vertices(center, radius):
    """Return the 7 corners (closed polygon) of the hexagon as complex numbers."""
    return center + radius * np.exp(1j * np.pi * np.arange(0, 13, 2) / 6)


def draw_cell(ax, center, radius, label=None, **plot_kwargs):
    """Draw one hexagonal cell (and optional label at its center) on ``ax``.

    Extra keyword arguments are passed to ``ax.plot`` (e.g. ``color='r'``).
    """
    plot_kwargs.setdefault("color", "k")
    plot_kwargs.setdefault("linewidth", 1)
    v = hex_vertices(center, radius)
    ax.plot(v.real, v.imag, **plot_kwargs)
    if label is not None:
        ax.text(center.real, center.imag, label, ha="center", va="center")


def fill_cell(ax, center, radius, color, alpha=0.35):
    """Shade one hexagonal cell."""
    v = hex_vertices(center, radius)
    ax.fill(v.real, v.imag, color=color, alpha=alpha, linewidth=0)


def cell_axes(ax=None, figsize=(6, 6)):
    """Return square, equal-aspect axes suitable for drawing cells."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax
