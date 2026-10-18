"""Phase free energies g(c, T) per unit volume for N components (m = N-1 independent mole fractions).

Convention: arrays of compositions have shape (npts, m); temperatures broadcast to (npts,).
mu_i = dg/dc_i (diffusion potentials of the independent components);
grand potential  omega(mu, T) = g - mu . c  evaluated at c(mu, T).
"""
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

Num = Union[float, np.ndarray, Callable]


def _ev(x, T):
    """Evaluate a constant or a callable of T."""
    return x(T) if callable(x) else x


@dataclass
class QuadraticPhase:
    """g = g0(T) + a(T).(c - c0) + 1/2 (c - c0)^T H(T) (c - c0).  a, H, c0, g0 may be callables of T."""
    g0: Num
    a: Num
    H: Num
    c0: Num

    def _parts(self, T):
        T = np.asarray(T, float)
        g0 = np.asarray(_ev(self.g0, T), float)
        a = np.atleast_1d(np.asarray(_ev(self.a, T), float))
        H = np.atleast_2d(np.asarray(_ev(self.H, T), float))
        c0 = np.atleast_1d(np.asarray(_ev(self.c0, T), float))
        return g0, a, H, c0

    @property
    def m(self):
        return np.atleast_2d(np.asarray(_ev(self.H, 300.0), float)).shape[0]

    @staticmethod
    def _vec(a, T, m):
        """Broadcast a (m,) or (npts, m) quantity to (npts, m)."""
        a = np.asarray(a)
        return a if a.ndim == 2 else np.broadcast_to(a, (np.size(T) or 1, m))

    def g(self, c, T):
        g0, a, H, c0 = self._parts(T)
        d = np.atleast_2d(c) - c0
        return g0 + np.einsum("...i,...i->...", d, a * np.ones_like(d)) + 0.5 * np.einsum("...i,...ij,...j->...", d, H * np.ones(d.shape[:-1] + H.shape[-2:]) if H.ndim == 2 else H, d)

    def mu(self, c, T):
        _, a, H, c0 = self._parts(T)
        d = np.atleast_2d(c) - c0
        return a + np.einsum("ij,...j->...i", H, d) if H.ndim == 2 else a + np.einsum("...ij,...j->...i", H, d)

    def c_of_mu(self, mu, T):
        _, a, H, c0 = self._parts(T)
        rhs = np.atleast_2d(mu) - a
        if H.ndim == 2:
            return c0 + np.linalg.solve(H, rhs.T).T
        return c0 + np.linalg.solve(H, rhs[..., None])[..., 0]

    def omega(self, mu, T):
        c = self.c_of_mu(mu, T)
        return self.g(c, T) - np.einsum("...i,...i->...", np.atleast_2d(mu), c)

    def is_stable(self, T):
        """Hessian positive definite (necessary for a stable single-phase free energy)."""
        H = np.atleast_2d(np.asarray(_ev(self.H, T), float))
        return bool(np.all(np.linalg.eigvalsh(0.5 * (H + H.T)) > 0))
