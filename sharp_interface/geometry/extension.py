"""Transfer of interface data (defined at edge crossings) to the grid and back.

`InterfaceMap` projects every grid node onto its closest interface point  p = x - phi n  and evaluates
crossing data there by inverse-distance weighting over the nearest crossings. Used for (a) extending the
interface speed to all nodes for level-set advection, and (b) filling nodes that change phase with
(Dirichlet value + one-sided gradient correction).
"""
import numpy as np
from scipy.spatial import cKDTree

from . import levelset as ls


def crossing_values(field, cs):
    """Linear interpolation of a node field to the crossings."""
    ia, ja = cs.i, cs.j
    ib = ia + (cs.axis == 0)
    jb = ja + (cs.axis == 1)
    return (1 - cs.theta) * field[ia, ja] + cs.theta * field[ib, jb]


def crossing_normals(phi, cs, dx):
    nx, ny = ls.normal(phi, dx)
    cx, cy = crossing_values(nx, cs), crossing_values(ny, cs)
    r = np.hypot(cx, cy) + 1e-300
    return cx / r, cy / r


class InterfaceMap:
    def __init__(self, grid, phi, cs, k=4):
        self.grid, self.cs = grid, cs
        X, Y = grid.mesh()
        nx, ny = ls.normal(phi, grid.dx)
        self.px, self.py = X - phi * nx, Y - phi * ny
        self.tree = cKDTree(np.column_stack([cs.x, cs.y]))
        kk = min(k, cs.n)
        d, ix = self.tree.query(np.column_stack([self.px.ravel(), self.py.ravel()]), k=kk)
        d, ix = d.reshape(grid.shape + (-1,)), ix.reshape(grid.shape + (-1,))
        w = 1.0 / (d**2 + (1e-3 * grid.dx) ** 2)
        self.w = w / w.sum(axis=-1, keepdims=True)
        self.ix = ix

    def project(self, vals):
        """vals: (K,) crossing data -> (nx, ny) grid field (value at the closest interface point)."""
        return np.sum(self.w * vals[self.ix], axis=-1)
