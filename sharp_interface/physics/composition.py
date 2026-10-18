"""Implicit (backward-Euler) diffusion of m coupled components in one phase on the sharp-interface domain.

dc/dt = div(D grad c),  D an (m x m) interdiffusion matrix (constant per phase here).  Stacked unknowns
are component-major, giving the block system   (I - dt D (x) A) c^{n+1} = c^n + dt (D (x) Bg) uG.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class PhaseDiffusion:
    def __init__(self, op, D, dt):
        self.op, self.dt = op, dt
        self.D = np.atleast_2d(np.asarray(D, float))
        self.m = self.D.shape[0]
        n = op.n_unknown
        I = sp.identity(self.m * n, format="csc")
        self.DB = sp.kron(sp.csr_matrix(self.D), op.Bg, format="csr")
        self.lu = spla.splu((I - dt * sp.kron(sp.csr_matrix(self.D), op.A)).tocsc())

    def solve(self, c_old_nodes, uG):
        """c_old_nodes: (m, n); uG: (K, m) interface Dirichlet values -> (m, n) new nodal values."""
        rhs = c_old_nodes.reshape(-1) + self.dt * (self.DB @ uG.T.reshape(-1))
        return self.lu.solve(rhs).reshape(self.m, -1)

    def normal_derivative(self, c_nodes, uG, nxc, nyc):
        """d c_i / dn at the crossings, (K, m)."""
        out = np.empty((uG.shape[0], self.m))
        for i in range(self.m):
            out[:, i] = self.op.normal_derivative(self.op.to_grid(c_nodes[i]), uG[:, i], nxc, nyc)
        return out
