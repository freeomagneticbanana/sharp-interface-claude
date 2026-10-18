"""Local interface conditions at the crossings (everything except the bulk fluxes).

Unknowns at each crossing: liquid-side interface composition c_l (m) and normal speed V (+ interface
temperature T when the temperature evolves). Residuals:
    mass:  V (c_l - c_s) + D_l dn c_l - D_s dn c_s = 0      (m equations)    [Rankine-Hugoniot, n: s -> l]
    GT  :  (omega_l - omega_s)(mu, T) - stiffness(theta,T) kappa - beta(theta) V = 0   (1 equation)
with c_s from  mu_s(c_s) = mu_l(c_l)  (common diffusion potentials).
"""
import numpy as np

from ..thermo import equilibrium as eq


def interface_state(solid, liquid, c_l, T):
    c_s = eq.solid_composition(solid, liquid, c_l, T)
    dOmega = eq.driving_force(solid, liquid, c_l, T)
    return c_s, dOmega


def residuals(c_l, V, T, c_s, dOmega, dn_cl, dn_cs, D_l, D_s, stiff, kappa, beta):
    mass = V[:, None] * (c_l - c_s) + dn_cl @ D_l.T - dn_cs @ D_s.T
    gt = dOmega - stiff * kappa - beta * V
    return mass, gt
