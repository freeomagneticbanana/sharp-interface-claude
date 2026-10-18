"""M4: isothermal planar front in a ternary alloy (m = 2) with unequal liquid diffusivities.

Exact solution: X = 2 lam sqrt(D_1 t); liquid c_i = c_inf_i + (c*_i - c_inf_i) erfc(x / 2 sqrt(D_i t)) / erfc(lam_i),
lam_i = lam sqrt(D_1 / D_i); unknowns (lam, mu*_1, mu*_2) from omega_l = omega_s and two mass balances.
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erfc, erfcx

from sharp_interface import Grid2D
from sharp_interface.interface.anisotropy import Kinetics, SurfaceEnergy
from sharp_interface.solver import AlloySolver, Material
from sharp_interface.thermo.free_energy import QuadraticPhase

TM, T = 1000.0, 990.0
D = np.array([1.0, 0.4])
C_INF = np.array([0.08, 0.05])
Hs = np.array([[4.0, 0.5], [0.5, 3.0]])
Hl = np.array([[2.0, 0.3], [0.3, 1.5]])
solid = QuadraticPhase(lambda T: 0.002 * (T - TM), [0.0, 0.0], Hs, [0.0, 0.0])
liquid = QuadraticPhase(0.0, [0.0, 0.0], Hl, [0.05, 0.03])


def exact():
    def F(z):
        lam, mu = z[0], z[1:].reshape(1, 2)
        cl, cs = liquid.c_of_mu(mu, T)[0], solid.c_of_mu(mu, T)[0]
        lam_i = lam * np.sqrt(D[0] / D)
        bal = lam * np.sqrt(np.pi) * np.sqrt(D[0] / D) * (cl - cs) - (cl - C_INF) / erfcx(lam_i)
        return [liquid.omega(mu, T)[0] - solid.omega(mu, T)[0], *bal]
    z = fsolve(F, [0.5, 0.2, 0.2], xtol=1e-14)
    assert np.max(np.abs(F(z))) < 1e-10
    lam, mu = z[0], z[1:].reshape(1, 2)
    return lam, liquid.c_of_mu(mu, T)[0], solid.c_of_mu(mu, T)[0]


def run(nx, Lx=2.0, t0=0.01, t_end=0.05):
    lam, cl, cs = exact()
    dx = Lx / nx
    g = Grid2D(nx, 1, dx, 0.5 * dx, 0.5 * dx)
    X, _ = g.mesh()
    lam_i = lam * np.sqrt(D[0] / D)
    c_l = np.stack([C_INF[i] + (cl[i] - C_INF[i]) * erfc(X / (2 * np.sqrt(D[i] * t0))) / erfc(lam_i[i])
                    for i in range(2)])
    c_s = np.stack([np.full(g.shape, cs[i]) for i in range(2)])
    phi = X - 2 * lam * np.sqrt(D[0] * t0)
    mat = Material(solid, liquid, np.zeros((2, 2)), np.diag(D), SurfaceEnergy(0.0), Kinetics(0.0))
    sol = AlloySolver(g, phi, c_s, c_l, mat, lambda x, y, t: np.full_like(x, T), E_ref=0.01, t0=t0)
    sol.run(t_end, 0.32 / nx)
    return sol, lam, cl, cs


def test_ternary_front_and_tie_line():
    errs, tie = [], []
    for nx in (160, 320):
        sol, lam, cl, cs = run(nx)
        X = 2 * lam * np.sqrt(D[0] * sol.t)
        errs.append(abs(sol.front_position_x(0) - X) / X)
        tie.append(max(np.max(np.abs(sol.last["c_l"][0] / cl - 1)), np.max(np.abs(sol.last["c_s"][0] / cs - 1))))
    assert errs[-1] < 0.03 and np.log2(errs[0] / errs[1]) > 0.7, errs
    assert tie[-1] < 0.01 and np.log2(tie[0] / tie[1]) > 0.7, tie
