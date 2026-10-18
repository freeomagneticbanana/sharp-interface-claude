"""M4: isothermal planar binary front against the exact similarity solution (frozen temperature)."""
import numpy as np
import pytest

from planar import build_solver, front_error, mass_error, exact

T_END = 0.05


def run(nx, D_s=0.0, c_s0=None):
    sol, lam, D_l = build_solver(nx, D_s=D_s, c_s0=c_s0)
    sol.run(T_END, 0.32 / nx)  # dt proportional to dx
    return sol, lam, D_l


@pytest.mark.parametrize("D_s,c_s0", [(0.0, None), (0.3, 0.0)])
def test_front_position_converges(D_s, c_s0):
    errs = []
    for nx in (160, 320):
        sol, lam, D_l = run(nx, D_s, c_s0)
        X = 2 * lam * np.sqrt(D_l * sol.t)
        errs.append(abs(front_error(sol, lam, D_l)) / X)
    assert errs[-1] < 0.025, errs
    assert np.log2(errs[0] / errs[1]) > 0.7, errs  # first order (backward Euler + lagged geometry)


def test_solid_diffusion_changes_growth_rate():
    """The front shift caused by solid diffusion matches the exact shift (discretisation errors cancel
    between the two runs at equal resolution)."""
    sol0, lam0, _ = run(320)
    sol1, lam1, _ = run(320, 0.3, 0.0)
    t = sol0.t
    d_num = sol1.front_position_x(0) - sol0.front_position_x(0)
    d_exact = 2 * (lam1 - lam0) * np.sqrt(t)
    assert abs(d_num - d_exact) < 0.15 * abs(d_exact), (d_num, d_exact)


def test_solute_conservation_first_order():
    errs = []
    for nx in (160, 320, 640):
        sol, lam, D_l = run(nx)
        num, _, ml = mass_error(sol, lam, D_l)
        X = 2 * lam * np.sqrt(D_l * sol.t)
        from planar import tie_line
        exact_total = tie_line()[1] * X + ml
        errs.append(abs(num - exact_total) / exact_total)
    assert errs[-1] < 1e-3 and errs[0] > errs[1] > errs[2], errs
