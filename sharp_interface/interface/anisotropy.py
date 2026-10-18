"""Surface energy anisotropy and kinetic coefficient.

gamma(theta, T) = gamma0(T) (1 + eps cos(nfold (theta - theta0))),  gamma0(T) = gamma_m + gamma_T (T - T_ref)
stiffness = gamma + gamma'' = gamma0 (1 + eps (1 - nfold^2) cos(nfold (theta - theta0)))
theta is the angle of the interface normal.
Note: the variation of gamma with *position through T* adds a term  (dgamma/dT) d_n T  to the capillary
driving force. It is not included here (documented model simplification).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceEnergy:
    gamma_m: float
    eps: float = 0.0
    nfold: int = 4
    theta0: float = 0.0
    gamma_T: float = 0.0
    T_ref: float = 0.0

    def gamma0(self, T):
        return self.gamma_m + self.gamma_T * (np.asarray(T) - self.T_ref)

    def gamma(self, theta, T):
        return self.gamma0(T) * (1 + self.eps * np.cos(self.nfold * (theta - self.theta0)))

    def stiffness(self, theta, T):
        return self.gamma0(T) * (1 + self.eps * (1 - self.nfold**2) * np.cos(self.nfold * (theta - self.theta0)))


@dataclass
class Kinetics:
    """Linear kinetic resistance beta(theta) = beta0 (1 + eps_k cos(nfold (theta - theta0)))."""
    beta0: float = 0.0
    eps: float = 0.0
    nfold: int = 4
    theta0: float = 0.0

    def beta(self, theta):
        return self.beta0 * (1 + self.eps * np.cos(self.nfold * (theta - self.theta0)))
