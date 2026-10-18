"""Shared set-up and exact similarity solution for the isothermal planar binary front."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc, erfcx

from sharp_interface import Grid2D
from sharp_interface.interface.anisotropy import Kinetics, SurfaceEnergy
from sharp_interface.solver import AlloySolver, Material
from sharp_interface.thermo import equilibrium as eq
from sharp_interface.thermo.free_energy import QuadraticPhase

TM, T_ISO, S = 1000.0, 990.0, 0.002
C_INF = 0.10


def phases():
    s = QuadraticPhase(lambda T: S * (T - TM), [0.0], [[4.0]], [0.0])
    l = QuadraticPhase(0.0, [0.0], [[2.0]], [0.05])
    return s, l


def tie_line():
    s, l = phases()
    c_star = eq.liquidus_composition(s, l, T_ISO, [0.1])[0]
    c_sg = eq.solid_composition(s, l, np.array([[c_star]]), T_ISO)[0, 0]
    return c_star, c_sg


def exact(D_l, D_s=0.0, c_s0=None):
    """Similarity solution: front X = 2 lam sqrt(D_l t). lam solves the interface mass balance
    lam sqrt(pi) (c* - c_sG) = (c* - c_inf)/erfcx(lam) + (c_sG - c_s0) sqrt(D_s/D_l) e^{-lam^2 r^2}/erfc(-lam r),
    r = sqrt(D_l/D_s); c_s0 is the far-field (initial) solid composition when D_s > 0."""
    c_star, c_sg = tie_line()

    def f(lam):
        val = lam * np.sqrt(np.pi) * (c_star - c_sg) - (c_star - C_INF) / erfcx(lam)
        if D_s > 0:
            r = np.sqrt(D_l / D_s)
            val -= (c_sg - c_s0) * np.sqrt(D_s / D_l) * np.exp(-(lam * r) ** 2) / erfc(-lam * r)
        return val

    return brentq(f, 1e-6, 3.0), c_star, c_sg


def initial_fields(grid, t0, D_l, D_s, lam, c_star, c_sg, c_s0):
    X, Y = grid.mesh()
    s0 = 2 * lam * np.sqrt(D_l * t0)
    phi = X - s0
    c_l = C_INF + (c_star - C_INF) * erfc(X / (2 * np.sqrt(D_l * t0))) / erfc(lam)
    if D_s > 0:
        c_s = c_s0 + (c_sg - c_s0) * erfc(-X / (2 * np.sqrt(D_s * t0))) / erfc(-lam * np.sqrt(D_l / D_s))
    else:
        c_s = np.full(grid.shape, c_sg)
    return phi, c_s[None], c_l[None], s0


def build_solver(nx, D_s=0.0, c_s0=None, Lx=2.0, ny=1, t0=0.01):
    dx = Lx / nx
    grid = Grid2D(nx, ny, dx, 0.5 * dx, 0.5 * dx)
    D_l = 1.0
    lam, c_star, c_sg = exact(D_l, D_s, c_s0)
    phi, c_s, c_l, s0 = initial_fields(grid, t0, D_l, D_s, lam, c_star, c_sg, c_s0)
    s, l = phases()
    mat = Material(s, l, [[D_s]], [[D_l]], SurfaceEnergy(0.0), Kinetics(0.0))
    sol = AlloySolver(grid, phi, c_s, c_l, mat, lambda x, y, t: np.full_like(x, T_ISO),
                      V_ref=1.0, E_ref=0.01, t0=t0)
    return sol, lam, D_l


def front_error(sol, lam, D_l):
    return sol.front_position_x() - 2 * lam * np.sqrt(D_l * sol.t)


def mass_error(sol, lam, D_l, D_s=0.0, c_s0=None, L=None):
    """Relative error of the total solute (cell-fraction quadrature with the numerical front) against the
    exact solute content of the domain at time sol.t."""
    from scipy.integrate import quad
    g, t = sol.grid, sol.t
    c_star, c_sg = tie_line()
    L = g.nx * g.dx if L is None else L
    X = sol.front_position_x(0)
    fs = np.clip((X - (g.x - g.dx / 2)) / g.dx, 0, 1)
    num = np.sum(fs * sol.c_s[0, :, 0] + (1 - fs) * sol.c_l[0, :, 0]) * g.dx
    Xe = 2 * lam * np.sqrt(D_l * t)
    cl = lambda x: C_INF + (c_star - C_INF) * erfc(x / (2 * np.sqrt(D_l * t))) / erfc(lam)
    if D_s > 0:
        r = np.sqrt(D_l / D_s)
        cs = lambda x: c_s0 + (c_sg - c_s0) * erfc(-x / (2 * np.sqrt(D_s * t))) / erfc(-lam * r)
        ms = quad(cs, 0.0, Xe, limit=200)[0]
    else:
        ms = None
    return num, ms, quad(cl, Xe, L, limit=200)[0]
