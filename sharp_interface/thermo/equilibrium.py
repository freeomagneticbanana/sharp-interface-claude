"""Interface thermodynamics from two phase free energies.

Local equilibrium at a flat interface: mu_s = mu_l (m conditions) and omega_s = omega_l.
At a curved, moving interface:  omega_l - omega_s = gamma_stiff * kappa + beta * V_n   (>0 = solid favoured).
"""
import numpy as np


def solid_composition(solid, liquid, c_l, T):
    """Solid composition in common-chemical-potential equilibrium with liquid composition c_l at T."""
    mu = liquid.mu(c_l, T)
    return solid.c_of_mu(mu, T)


def driving_force(solid, liquid, c_l, T):
    """omega_l - omega_s evaluated at the common mu set by the liquid composition. > 0: solid is favoured."""
    mu = liquid.mu(c_l, T)
    return liquid.omega(mu, T) - solid.omega(mu, T)


def liquidus_composition(solid, liquid, T, c_guess, tol=1e-13, maxit=60):
    """Liquid composition on the flat-interface coexistence curve at T (driving force = 0), by Newton with
    a finite-difference Jacobian. m components -> m unknowns need m equations; for m = 1 this is the
    liquidus. For m > 1 it fixes only the surface of coexistence, so extra constraints (e.g. along a
    fixed line in composition space) are the caller's concern: here the first component is solved and
    the others are held at c_guess."""
    c = np.array(c_guess, float).reshape(1, -1)
    for _ in range(maxit):
        f = driving_force(solid, liquid, c, T)[0]
        if abs(f) < tol:
            break
        h = 1e-7
        cp = c.copy(); cp[0, 0] += h
        df = (driving_force(solid, liquid, cp, T)[0] - f) / h
        c[0, 0] -= f / df
    return c[0]
