"""Cell clusters, frequency reuse, and serving-cell selection.

Public course module (used by the Week 1 lecture notebook and HW1).

Geometry (flat-topped hexagons of radius R, see ``hexgrid``):
  hex-lattice basis   a1 = sqrt(3) R e^{j30deg},  a2 = sqrt(3) R e^{j90deg}
  "move i cells, turn 60 deg counter-clockwise, move j cells" reaches the
  nearest co-channel cell  u1 = i a1 + j a2;  the other five first-tier
  co-channel cells are u1 rotated by multiples of 60 deg.
"""

import numpy as np

from .hexgrid import draw_cell, fill_cell

_ROT60 = np.exp(1j * np.pi / 3)


def lattice_basis(radius):
    """The two hex-lattice basis vectors (complex) for cell radius R."""
    a1 = np.sqrt(3) * radius * np.exp(1j * np.pi / 6)
    return a1, a1 * _ROT60


def cluster_size(i, j):
    """Number of cells per cluster, N = i^2 + i j + j^2."""
    return i * i + i * j + j * j


def cochannel_offsets(i, j, radius):
    """Offsets from any cell to its six first-tier co-channel cells."""
    a1, a2 = lattice_basis(radius)
    u1 = i * a1 + j * a2
    return u1 * _ROT60 ** np.arange(6)


def reuse_distance(i, j, radius):
    """Co-channel reuse distance D = R sqrt(3N)."""
    return radius * np.sqrt(3 * cluster_size(i, j))


def cluster_centers(center, i, j, radius):
    """Centers of the N cells in the cluster around ``center`` (``center`` first).

    Each cell of the hex lattice belongs to exactly one of N channel groups
    (cosets of the co-channel lattice); the cluster takes the representative
    of each group closest to the cluster center.
    """
    n_cells = cluster_size(i, j)
    if n_cells == 0:
        raise ValueError("i and j cannot both be zero")
    a1, a2 = lattice_basis(radius)
    # Co-channel lattice basis in (a1, a2) integer coordinates:
    # u1 = (i, j), u2 = u1 rotated 60 deg = (-j, i + j); det = N.
    basis = np.array([[i, -j], [j, i + j]], dtype=float)
    inv = np.linalg.inv(basis)
    span = i + j + 1
    best = {}
    for m in range(-span, span + 1):
        for n in range(-span, span + 1):
            coset = inv @ np.array([m, n])
            key = tuple(np.round((coset - np.floor(coset + 1e-9)) * n_cells).astype(int) % n_cells)
            offset = m * a1 + n * a2
            rank = (round(abs(offset), 6), round(np.angle(offset) % (2 * np.pi), 6))
            if key not in best or rank < best[key][0]:
                best[key] = (rank, offset)
    offsets = sorted((v for v in best.values()), key=lambda v: v[0])
    return center + np.array([off for _, off in offsets])


def draw_cluster(ax, center, i, j, radius, labels=True, colors=None):
    """Draw the cluster around ``center``; return its N cell centers."""
    centers = cluster_centers(center, i, j, radius)
    for k, c in enumerate(centers):
        if colors is not None:
            fill_cell(ax, c, radius, colors[k % len(colors)])
        draw_cell(ax, c, radius, label=_group_label(k) if labels else None)
    return centers


def first_tier_cochannel(center, i, j, radius):
    """Centers of the six first-tier co-channel cells of the cell at ``center``."""
    return center + cochannel_offsets(i, j, radius)


def find_serving_cell(mobile_location, cell_centers):
    """Index and center of the closest base station (circular-coverage rule)."""
    cell_centers = np.asarray(cell_centers)
    k = int(np.argmin(np.abs(cell_centers - mobile_location)))
    return k, cell_centers[k]


def _group_label(k):
    return chr(ord("A") + k) if k < 26 else str(k)
