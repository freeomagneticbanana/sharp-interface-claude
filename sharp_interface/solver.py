"""Sharp-interface N-component alloy solidification, frozen-temperature mode.

Per time step (geometry lagged by one step):
  1. crossings, normals, curvature, interface temperature from phi^n;
  2. bulk operators (Dirichlet ghost stencils) for solid and liquid, factorised once;
  3. Newton-Krylov solve for the interface unknowns (c_l, V) at every crossing, each residual evaluation
     doing two implicit bulk solves and one-sided normal derivatives;
  4. V extended to the grid, phi advected + reinitialised;
  5. nodes that changed phase are filled by extrapolating the donor phase from the old interface.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import newton_krylov

from .geometry import levelset as ls
from .geometry.extension import InterfaceMap, crossing_normals, crossing_values
from .interface import conditions
from .interface.anisotropy import Kinetics, SurfaceEnergy
from .operators.irregular_laplacian import CrossingSet, PhaseOperator
from .physics.composition import PhaseDiffusion
from .thermo.free_energy import QuadraticPhase


@dataclass
class Material:
    solid: QuadraticPhase
    liquid: QuadraticPhase
    D_s: np.ndarray
    D_l: np.ndarray
    surface: SurfaceEnergy
    kinetics: Kinetics


class AlloySolver:
    """c_s, c_l: arrays (m, nx, ny). temperature(x, y, t) -> T (prescribed). V_ref / E_ref set residual scales."""

    def __init__(self, grid, phi, c_s, c_l, material: Material, temperature: Callable,
                 V_ref=1.0, E_ref=1.0, cfl=0.4, reinit_iters=8, t0=0.0):
        self.grid, self.mat, self.temperature = grid, material, temperature
        phi = self._snap(phi.copy(), grid.dx)
        self.phi, self.c_s, self.c_l = phi, np.array(c_s, float), np.array(c_l, float)
        self.m = self.c_s.shape[0]
        self.D_s = np.atleast_2d(np.asarray(material.D_s, float))
        self.D_l = np.atleast_2d(np.asarray(material.D_l, float))
        self.V_ref, self.E_ref, self.cfl, self.reinit_iters = V_ref, E_ref, cfl, reinit_iters
        self.t = t0
        self.last = None  # interface data of the most recent step

    @staticmethod
    def _snap(phi, dx, delta=1e-3):
        """Keep nodes at least delta*dx from the interface (matches the stencil clamp; avoids sliver
        crossings created by round-off when the front passes almost exactly through a node)."""
        small = np.abs(phi) < delta * dx
        phi[small] = np.where(phi[small] < 0, -delta * dx, delta * dx)
        return phi

    # ------------------------------------------------------------------ one step
    def step(self, dt, newton_tol=1e-9):
        g, mat, m = self.grid, self.mat, self.m
        dx, phi = g.dx, self.phi
        cs = CrossingSet.from_phi(phi, g)
        K = cs.n
        nxc, nyc = crossing_normals(phi, cs, dx)
        kappa = crossing_values(ls.curvature(phi, dx), cs)
        theta = np.arctan2(nyc, nxc)
        T_G = self.temperature(cs.x, cs.y, self.t + dt)
        stiff = mat.surface.stiffness(theta, T_G)
        beta = mat.kinetics.beta(theta)

        solid_mask, liquid_mask = phi < 0, phi > 0
        opS = PhaseOperator(g, phi, solid_mask, cs)
        opL = PhaseOperator(g, phi, liquid_mask, cs)
        bulkS, bulkL = PhaseDiffusion(opS, self.D_s, dt), PhaseDiffusion(opL, self.D_l, dt)
        cS_old, cL_old = self.c_s[:, solid_mask], self.c_l[:, liquid_mask]
        cache = {}

        def evaluate(U):
            c_l = U[:K * m].reshape(K, m)
            V = U[K * m:] * self.V_ref
            c_s_G, dOm = conditions.interface_state(mat.solid, mat.liquid, c_l, T_G)
            cS_new = bulkS.solve(cS_old, c_s_G)
            cL_new = bulkL.solve(cL_old, c_l)
            dn_cs = bulkS.normal_derivative(cS_new, c_s_G, nxc, nyc)
            dn_cl = bulkL.normal_derivative(cL_new, c_l, nxc, nyc)
            mass, gt = conditions.residuals(c_l, V, T_G, c_s_G, dOm, dn_cl, dn_cs,
                                            self.D_l, self.D_s, stiff, kappa, beta)
            cache.update(c_s_G=c_s_G, c_l=c_l, V=V, cS_new=cS_new, cL_new=cL_new, dn_cs=dn_cs, dn_cl=dn_cl)
            return np.concatenate([(mass / self.V_ref).ravel(), gt / self.E_ref])

        # initial guess: liquid-side node values at the crossings, V from a previous step if sizes match
        ia, ja = cs.i, cs.j
        ib, jb = ia + (cs.axis == 0), ja + (cs.axis == 1)
        liq_first = liquid_mask[ia, ja]
        ci = np.where(liq_first, ia, ib), np.where(liq_first, ja, jb)
        c_l0 = np.stack([self.c_l[k][ci] for k in range(m)], axis=1)
        V0 = np.zeros(K)
        if self.last is not None and self.last["V"].shape == (K,):
            V0 = self.last["V"]
        U0 = np.concatenate([c_l0.ravel(), V0 / self.V_ref])
        U = newton_krylov(evaluate, U0, f_tol=newton_tol, method="lgmres", maxiter=60)
        evaluate(U)  # refresh cache at the converged point
        V = cache["V"]

        # ---- geometry update
        imap = InterfaceMap(g, phi, cs)
        Vg = imap.project(V)
        vmax = np.max(np.abs(Vg))
        nsub = max(1, int(np.ceil(vmax * dt / (self.cfl * dx))))
        phi_new = phi
        for _ in range(nsub):
            phi_new = ls.advect(phi_new, Vg, dt / nsub, dx)
        phi_new = ls.reinitialize(phi_new, dx, iters=self.reinit_iters)
        phi_new = self._snap(phi_new, dx)

        # ---- fields: solved values inside each (old) phase, extrapolated fill for new nodes
        c_s_new, c_l_new = self.c_s.copy(), self.c_l.copy()
        c_s_new[:, solid_mask] = cache["cS_new"]
        c_l_new[:, liquid_mask] = cache["cL_new"]
        new_solid = liquid_mask & (phi_new < 0)
        new_liquid = solid_mask & (phi_new > 0)
        for k in range(m):
            fill_s = imap.project(cache["c_s_G"][:, k]) + imap.project(cache["dn_cs"][:, k]) * phi
            fill_l = imap.project(cache["c_l"][:, k]) + imap.project(cache["dn_cl"][:, k]) * phi
            c_s_new[k][new_solid] = fill_s[new_solid]
            c_l_new[k][new_liquid] = fill_l[new_liquid]
        self.last = dict(V=V, cs=cs, c_l=cache["c_l"], c_s=cache["c_s_G"], T=T_G, kappa=kappa, theta=theta,
                         imap=imap, phi_old=phi)
        self.phi, self.c_s, self.c_l = phi_new, c_s_new, c_l_new
        self.t += dt
        return V

    def run(self, t_end, dt, callback=None):
        while self.t < t_end - 1e-14 * max(1.0, t_end):
            self.step(min(dt, t_end - self.t))
            if callback is not None:
                callback(self)

    # ------------------------------------------------------------------ diagnostics
    def solute_inventory(self):
        """Total solute per component with sub-cell weights: each node owns a cell; the part of that cell on the
        other side of the interface is filled with the interface (Dirichlet) value of the other phase."""
        g, phi, dx = self.grid, self.phi, self.grid.dx
        if self.last is None:
            f_s = np.where(phi < 0, 1.0, 0.0)
            return np.array([(f_s * self.c_s[k] + (1 - f_s) * self.c_l[k]).sum() * dx**2 for k in range(self.m)])
        imap = InterfaceMap(g, phi, CrossingSet.from_phi(phi, g))  # geometry of the *current* fields
        # Dirichlet values were solved on the previous interface; project them onto the current one
        inv = np.zeros(self.m)
        cGs = [self.last["imap"].project(self.last["c_s"][:, k]) for k in range(self.m)]
        cGl = [self.last["imap"].project(self.last["c_l"][:, k]) for k in range(self.m)]
        own = np.minimum(1.0, 0.5 + np.abs(phi) / dx)
        for k in range(self.m):
            solid_node = phi < 0
            val = np.where(solid_node, own * self.c_s[k] + (1 - own) * cGl[k],
                           own * self.c_l[k] + (1 - own) * cGs[k])
            inv[k] = val.sum() * dx**2
        return inv

    def front_position_x(self, j=None):
        """x-position of the interface along row j (planar tests)."""
        j = self.grid.ny // 2 if j is None else j
        row = self.phi[:, j]
        i = np.nonzero(row[:-1] * row[1:] < 0)[0]
        i = i[0]
        th = row[i] / (row[i] - row[i + 1])
        return self.grid.x[i] + th * self.grid.dx
