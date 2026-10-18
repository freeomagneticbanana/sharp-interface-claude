"""Sharp Laplacian on an irregular (level-set) domain, Gibou-Fedkiw style.

For one phase (a boolean node mask `inside`), unknowns live on nodes in the phase. Where a stencil leg
crosses the interface, the leg ends at the crossing point (distance h < dx from the node) where a
Dirichlet value uG[k] is imposed, and the second derivative uses the unequal-spacing 3-point formula

    u'' = 2 [ u_l / (h_l (h_l + h_r)) - u_0 / (h_l h_r) + u_r / (h_r (h_l + h_r)) ].

At the domain boundary a zero-normal-gradient reflection (ghost value = node value) is used, which is
exact to second order for the cell-centred grids made by Grid2D.square.

Result:  (L u)_nodes = A @ u_nodes + Bg @ uG        (unit diffusivity; scale by D outside)
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..geometry.levelset import crossings


@dataclass
class CrossingSet:
    """Edge crossings of phi = 0, with lookup tables from grid edges to crossing ids."""
    x: np.ndarray
    y: np.ndarray
    i: np.ndarray
    j: np.ndarray
    axis: np.ndarray
    theta: np.ndarray
    cid0: np.ndarray  # (nx-1, ny): crossing id on x-edges, -1 if none
    cid1: np.ndarray  # (nx, ny-1): crossing id on y-edges, -1 if none

    @property
    def n(self):
        return len(self.x)

    @classmethod
    def from_phi(cls, phi, grid):
        c = crossings(phi, grid)
        cid0 = -np.ones((grid.nx - 1, grid.ny), dtype=int)
        cid1 = -np.ones((grid.nx, grid.ny - 1), dtype=int)
        ids = np.arange(len(c["x"]))
        m0 = c["axis"] == 0
        cid0[c["i"][m0], c["j"][m0]] = ids[m0]
        cid1[c["i"][~m0], c["j"][~m0]] = ids[~m0]
        return cls(c["x"], c["y"], c["i"], c["j"], c["axis"], c["theta"], cid0, cid1)


class PhaseOperator:
    def __init__(self, grid, phi, inside, cs: CrossingSet = None):
        self.grid, self.inside = grid, inside
        self.cs = cs if cs is not None else CrossingSet.from_phi(phi, grid)
        self.n_unknown = int(inside.sum())
        self.idx = -np.ones(grid.shape, dtype=int)
        self.idx[inside] = np.arange(self.n_unknown)
        self._assemble()

    # ---- assembly -------------------------------------------------------------------------
    def _assemble(self):
        g, cs, ins, idx = self.grid, self.cs, self.inside, self.idx
        dx = g.dx
        rows, cols, vals = [], [], []
        brow, bcol, bval = [], [], []
        diag = np.zeros(self.n_unknown)
        nx, ny = g.shape
        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        for ax in (0, 1):
            # neighbour index arrays for left (-1) and right (+1) legs along this axis
            def leg(sign):
                if ax == 0:
                    ni, nj = I + sign, J
                    out = (ni < 0) | (ni >= nx)
                else:
                    ni, nj = I, J + sign
                    out = (nj < 0) | (nj >= ny)
                nic, njc = np.clip(ni, 0, nx - 1), np.clip(nj, 0, ny - 1)
                nb_in = ins[nic, njc] & ~out
                # crossing on the edge between (I,J) and (ni,nj)
                if ax == 0:
                    ei = np.where(sign > 0, I, I - 1)
                    ec = cs.cid0[np.clip(ei, 0, nx - 2), J]
                    ec = np.where((ei < 0) | (ei > nx - 2), -1, ec)
                    node_is_first = sign > 0
                else:
                    ej = np.where(sign > 0, J, J - 1)
                    ec = cs.cid1[I, np.clip(ej, 0, ny - 2)]
                    ec = np.where((ej < 0) | (ej > ny - 2), -1, ec)
                    node_is_first = sign > 0
                th = np.where(ec >= 0, cs.theta[np.clip(ec, 0, None)] if cs.n else 0.0, 0.5)
                h = np.where(ec >= 0, (th if node_is_first else 1 - th) * dx, dx)
                kind = np.where(out, 2, np.where(ec >= 0, 1, 0))  # 0 node, 1 crossing, 2 reflection
                return nic, njc, kind, h, ec, nb_in

            li, lj, lk, hl, lc, _ = leg(-1)
            ri, rj, rk, hr, rc, _ = leg(+1)
            hl = np.maximum(hl, 1e-3 * dx)
            hr = np.maximum(hr, 1e-3 * dx)
            cl = 2.0 / (hl * (hl + hr))
            cr = 2.0 / (hr * (hl + hr))
            c0 = -2.0 / (hl * hr)
            m = ins
            r = idx[m]
            diag_add = c0[m].copy()
            for (ni, nj, kind, coef, ec) in ((li, lj, lk, cl, lc), (ri, rj, rk, cr, rc)):
                k = kind[m]
                cf = coef[m]
                node = k == 0
                rows.append(r[node]); cols.append(idx[ni[m], nj[m]][node]); vals.append(cf[node])
                refl = k == 2
                diag_add[refl] += cf[refl]
                cr_ = k == 1
                brow.append(r[cr_]); bcol.append(ec[m][cr_]); bval.append(cf[cr_])
            np.add.at(diag, r, diag_add)
        rows.append(np.arange(self.n_unknown)); cols.append(np.arange(self.n_unknown)); vals.append(diag)
        n = self.n_unknown
        self.A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self.Bg = sp.csr_matrix((np.concatenate(bval), (np.concatenate(brow), np.concatenate(bcol))),
                                shape=(n, cs.n))

    # ---- helpers --------------------------------------------------------------------------
    def to_nodes(self, field):
        return field[self.inside]

    def to_grid(self, u_nodes, fill=0.0):
        out = np.full(self.grid.shape, fill, dtype=float)
        out[self.inside] = u_nodes
        return out

    # ---- normal derivative at the crossings ----------------------------------------------
    def normal_derivative(self, u, uG, nx_, ny_):
        """d/dn u at every crossing from the phase side (n = (nx_, ny_) given at the crossings).
        u: full-grid array (values used only where inside); uG: Dirichlet values at crossings."""
        g, cs, ins = self.grid, self.cs, self.inside
        dx = g.dx
        nxg, nyg = g.shape
        th = cs.theta
        ax = cs.axis
        i, j = cs.i, cs.j
        di = np.where(ax == 0, 1, 0)
        dj = 1 - di
        i2, j2 = i + di, j + dj  # second endpoint
        first_in = ins[i, j]
        # node1: the endpoint inside the phase; t points from crossing into the phase
        n1i = np.where(first_in, i, i2); n1j = np.where(first_in, j, j2)
        sg = np.where(first_in, -1.0, 1.0)  # t = sg * e_axis
        # Nodes 2 and 3 (>= dx from the interface) are used, not node 1: when the crossing sits very
        # close to node 1 (h1 << dx) its O(dx^2) error would be divided by h1.
        def step(k):
            return n1i + k * (sg * di).astype(int), n1j + k * (sg * dj).astype(int)

        def valid(ni, nj):
            inb = (ni >= 0) & (ni < nxg) & (nj >= 0) & (nj < nyg)
            nic, njc = np.clip(ni, 0, nxg - 1), np.clip(nj, 0, nyg - 1)
            return inb & ins[nic, njc], nic, njc

        h1 = np.where(first_in, th, 1 - th) * dx
        n2i, n2j = step(1)
        n3i, n3j = step(2)
        ok2, n2ic, n2jc = valid(n2i, n2j)
        ok3, n3ic, n3jc = valid(n3i, n3j)
        ok3 &= ok2
        u1 = u[n1i, n1j]
        u2 = np.where(ok2, u[n2ic, n2jc], 0.0)
        u3 = np.where(ok3, u[n3ic, n3jc], 0.0)
        # quadratic through (0,uG), (h1+dx,u2), (h1+2dx,u3)
        a_, b_ = h1 + dx, h1 + 2 * dx
        d3 = (-(a_ + b_) / (a_ * b_)) * uG + (b_ / (a_ * (b_ - a_))) * u2 + (-a_ / (b_ * (b_ - a_))) * u3
        # fall back: quadratic through (0,uG),(h1,u1),(h1+dx,u2), then two-point
        s1, s2 = h1, h1 + dx
        d2 = (-(s1 + s2) / (s1 * s2)) * uG + (s2 / (s1 * (s2 - s1))) * u1 + (-s1 / (s2 * (s2 - s1))) * u2
        d1 = (u1 - uG) / h1
        dt_ = np.where(ok3, d3, np.where(ok2, d2, d1))  # derivative along t
        d_ax = sg * dt_  # derivative along the crossing's grid axis

        # transverse derivative: finite differences at nodes, extrapolated linearly to Gamma
        def tdir(ni, nj):
            # second-order: central, else 3-point one-sided; first-order / zero only in thin slivers
            p1, p1i, p1j = valid(ni + dj, nj + di)
            p2, p2i, p2j = valid(ni + 2 * dj, nj + 2 * di)
            m1, m1i, m1j = valid(ni - dj, nj - di)
            m2, m2i, m2j = valid(ni - 2 * dj, nj - 2 * di)
            u0, up, up2, um, um2 = u[ni, nj], u[p1i, p1j], u[p2i, p2j], u[m1i, m1j], u[m2i, m2j]
            return np.where(
                p1 & m1, (up - um) / (2 * dx),
                np.where(p1 & p2, (-3 * u0 + 4 * up - up2) / (2 * dx),
                         np.where(m1 & m2, (3 * u0 - 4 * um + um2) / (2 * dx),
                                  np.where(p1, (up - u0) / dx, np.where(m1, (u0 - um) / dx, 0.0)))))
        g1 = tdir(n1i, n1j)
        g2 = np.where(ok2, tdir(n2ic, n2jc), g1)
        g3 = np.where(ok3, tdir(n3ic, n3jc), g2)
        d_tr = np.where(ok3, g2 + (g2 - g3) * ((h1 + dx) / dx),
                        np.where(ok2, g1 + (g1 - g2) * (h1 / dx), g1))
        ux = np.where(ax == 0, d_ax, d_tr)
        uy = np.where(ax == 1, d_ax, d_tr)
        return nx_ * ux + ny_ * uy
