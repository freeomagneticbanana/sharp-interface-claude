import numpy as np
import pytest

from sharp_interface.interface.anisotropy import SurfaceEnergy, Kinetics
from sharp_interface.thermo.free_energy import QuadraticPhase
from sharp_interface.thermo import equilibrium as eq

rng = np.random.default_rng(0)


def random_phase(m, shift=0.0):
    A = rng.normal(size=(m, m))
    H = A @ A.T + m * np.eye(m)
    return QuadraticPhase(g0=lambda T: -1e-3 * (T - 1000.0) + shift, a=lambda T: 1e-3 * (T - 900.0) * np.ones(m),
                          H=H, c0=0.2 * np.ones(m) + shift)


def test_binary_tie_line_matches_analytic():
    Hs, Hl, cs0, cl0, a_s, a_l = 4.0, 2.0, 0.1, 0.3, 0.5, 0.2
    s = QuadraticPhase(0.0, [a_s], [[Hs]], [cs0])
    l = QuadraticPhase(0.0, [a_l], [[Hl]], [cl0])
    c_l = np.array([[0.25]])
    mu = a_l + Hl * (0.25 - cl0)
    cs_expected = cs0 + (mu - a_s) / Hs
    assert np.allclose(eq.solid_composition(s, l, c_l, 1000.0), cs_expected)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_common_tangent_and_omega_derivative(m):
    s, l = random_phase(m), random_phase(m, 0.05)
    T = 950.0
    c_l = rng.uniform(0.1, 0.3, size=(5, m))
    c_s = eq.solid_composition(s, l, c_l, T)
    assert np.allclose(s.mu(c_s, T), l.mu(c_l, T), atol=1e-12)
    # d omega / d mu_i = -c_i
    mu = l.mu(c_l, T)
    h = 1e-6
    for i in range(m):
        d = np.zeros(m); d[i] = h
        num = (l.omega(mu + d, T) - l.omega(mu - d, T)) / (2 * h)
        assert np.allclose(num, -c_l[:, i], atol=1e-7)
    assert s.is_stable(T) and l.is_stable(T)


def test_driving_force_vanishes_on_liquidus_and_changes_sign():
    s = QuadraticPhase(lambda T: 0.0, lambda T: [0.5], [[4.0]], [0.1])
    l = QuadraticPhase(lambda T: 0.02 * (T - 1000.0) * 0 + 0.0 - 1e-3 * (T - 1000.0), lambda T: [0.2], [[2.0]], [0.3])
    T = 980.0
    cL = eq.liquidus_composition(s, l, T, [0.25])
    assert abs(eq.driving_force(s, l, cL.reshape(1, -1), T)[0]) < 1e-11
    f_hi = eq.driving_force(s, l, (cL + 0.02).reshape(1, -1), T)[0]
    f_lo = eq.driving_force(s, l, (cL - 0.02).reshape(1, -1), T)[0]
    assert f_hi * f_lo < 0


def test_stiffness_matches_finite_difference():
    se = SurfaceEnergy(gamma_m=0.3, eps=0.02, nfold=4, theta0=0.2, gamma_T=-1e-3, T_ref=1000.0)
    th, T, h = np.linspace(0, 2 * np.pi, 11), 990.0, 1e-4
    num = se.gamma(th, T) + (se.gamma(th + h, T) - 2 * se.gamma(th, T) + se.gamma(th - h, T)) / h**2
    assert np.allclose(num, se.stiffness(th, T), atol=1e-4)
    assert np.isclose(se.gamma0(990.0), 0.3 + 1e-2)


def test_kinetics_fourfold():
    k = Kinetics(beta0=2.0, eps=0.1)
    assert np.isclose(k.beta(0.0), 2.2) and np.isclose(k.beta(np.pi / 4), 1.8)
