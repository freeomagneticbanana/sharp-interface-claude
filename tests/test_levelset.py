import numpy as np
import pytest

from sharp_interface import Grid2D
from sharp_interface.geometry import levelset as ls


def disc(grid, r0, cx=0.5, cy=0.5):
    X, Y = grid.mesh()
    return np.hypot(X - cx, Y - cy) - r0


def test_grid_shapes():
    g = Grid2D.square(16)
    X, Y = g.mesh()
    assert X.shape == (16, 16) and np.isclose(X[1, 0] - X[0, 0], g.dx)


def test_curvature_second_order():
    errs = []
    for n in (64, 128, 256):
        g = Grid2D.square(n)
        phi = disc(g, 0.25)
        k = ls.curvature(phi, g.dx)
        band = np.abs(phi) < 3 * g.dx
        X, Y = g.mesh()
        exact = 1.0 / np.hypot(X - 0.5, Y - 0.5)  # curvature of the level set through each node
        errs.append(np.max(np.abs(k[band] - exact[band])))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.8), (errs, orders)


def test_advect_growing_disc():
    g = Grid2D.square(128)
    phi = disc(g, 0.15)
    v = np.full(g.shape, 0.2)
    t, T = 0.0, 0.5
    while t < T - 1e-12:
        dt = min(ls.max_stable_dt(v, g.dx), T - t)
        phi = ls.advect(phi, v, dt, g.dx)
        t += dt
    r = np.sqrt(np.sum(phi < 0) * g.dx**2 / np.pi)
    assert abs(r - (0.15 + 0.2 * T)) < 2 * g.dx


def test_reinit_preserves_front_and_restores_distance():
    g = Grid2D.square(128)
    exact = disc(g, 0.25)
    phi0 = exact * (1.0 + 4.0 * np.exp(-((g.mesh()[0] - 0.3) ** 2) / 0.05))  # distorted gradient
    phi = ls.reinitialize(phi0, g.dx, iters=60)
    band = np.abs(exact) < 4 * g.dx
    assert np.max(np.abs(phi[band] - exact[band])) < 0.5 * g.dx
    gx, gy = ls.central_gradient(phi, g.dx)
    assert np.max(np.abs(np.hypot(gx, gy)[band] - 1.0)) < 0.15


def test_crossings_on_circle():
    g = Grid2D.square(64)
    phi = disc(g, 0.25)
    c = ls.crossings(phi, g)
    r = np.hypot(c["x"] - 0.5, c["y"] - 0.5)
    assert np.max(np.abs(r - 0.25)) < 0.1 * g.dx**1  # linear interpolation, O(dx^2) in practice
    assert len(r) > 100
