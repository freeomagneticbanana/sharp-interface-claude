"""Radial growth of a solid disc in a supersaturated binary liquid (D_s = 0, gamma = 0), quarter domain.

Exact: R = 2 lam sqrt(D t), c_l = c_inf + (c* - c_inf) E1(r^2 / 4Dt) / E1(lam^2),
lam^2 e^{lam^2} E1(lam^2) = (c* - c_inf) / (c* - c_sG).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import exp1

from planar import C_INF, T_ISO, phases, tie_line
from sharp_interface import Grid2D
from sharp_interface.interface.anisotropy import Kinetics, SurfaceEnergy
from sharp_interface.solver import AlloySolver, Material


def lam_exact():
    c_star, c_sg = tie_line()
    om = (c_star - C_INF) / (c_star - c_sg)
    return brentq(lambda l: l**2 * np.exp(l**2) * exp1(l**2) - om, 1e-4, 5.0)


def build(n, L=1.5, t0=0.01, gamma=0.0, eps4=0.0, beta=0.0):
    lam = lam_exact()
    c_star, c_sg = tie_line()
    dx = L / n
    g = Grid2D(n, n, dx, 0.5 * dx, 0.5 * dx)
    X, Y = g.mesh()
    r = np.hypot(X, Y)
    R0 = 2 * lam * np.sqrt(t0)
    phi = r - R0
    c_l = C_INF + (c_star - C_INF) * exp1(np.maximum(r, 1e-12) ** 2 / (4 * t0)) / exp1(lam**2)
    c_l = np.minimum(c_l, c_star)  # inside the disc (only used for extrapolated values)
    c_s = np.full(g.shape, c_sg)
    s, l = phases()
    mat = Material(s, l, [[0.0]], [[1.0]], SurfaceEnergy(gamma, eps4), Kinetics(beta))
    sol = AlloySolver(g, phi, c_s[None], c_l[None], mat, lambda x, y, t: np.full_like(x, T_ISO),
                      E_ref=0.01, t0=t0)
    return sol, lam


def radius(sol):
    """Equivalent radius from the solid area of the quarter domain (cell fractions from phi)."""
    dx = sol.grid.dx
    f = np.clip(0.5 - sol.phi / dx, 0, 1)
    return np.sqrt(4 * f.sum() * dx**2 / np.pi)
