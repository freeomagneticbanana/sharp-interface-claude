"""Level-set geometry: phi < 0 solid, phi > 0 liquid, n = grad(phi)/|grad(phi)| (solid -> liquid).

Interface motion with normal speed V_n (> 0 = growth of the solid) is  phi_t + V_n |grad phi| = 0.
Edge (Neumann, zero-gradient) padding is used at the domain boundary.
"""
import numpy as np


def _pad(a):
    return np.pad(a, 1, mode="edge")


def central_gradient(phi, dx):
    p = _pad(phi)
    gx = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
    gy = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    return gx, gy


def normal(phi, dx, eps=1e-14):
    gx, gy = central_gradient(phi, dx)
    g = np.sqrt(gx**2 + gy**2) + eps
    return gx / g, gy / g


def curvature(phi, dx, eps=1e-14):
    """kappa = div(grad phi / |grad phi|) from the analytic form with central differences.
    For a solid disc (phi = r - R) kappa = +1/R."""
    p = _pad(phi)
    c = p[1:-1, 1:-1]
    px = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
    py = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    pxx = (p[2:, 1:-1] - 2 * c + p[:-2, 1:-1]) / dx**2
    pyy = (p[1:-1, 2:] - 2 * c + p[1:-1, :-2]) / dx**2
    pxy = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) / (4 * dx**2)
    num = pxx * py**2 - 2 * px * py * pxy + pyy * px**2
    return num / ((px**2 + py**2) ** 1.5 + eps)


def _godunov_gradnorm(phi, sign, dx):
    """|grad phi| with Godunov upwinding for motion with characteristic direction sign*grad(phi)."""
    p = _pad(phi)
    c = p[1:-1, 1:-1]
    a = (c - p[:-2, 1:-1]) / dx  # backward x
    b = (p[2:, 1:-1] - c) / dx  # forward x
    cc = (c - p[1:-1, :-2]) / dx
    d = (p[1:-1, 2:] - c) / dx
    pos = sign > 0
    ax = np.where(pos, np.maximum(np.maximum(a, 0) ** 2, np.minimum(b, 0) ** 2),
                  np.maximum(np.minimum(a, 0) ** 2, np.maximum(b, 0) ** 2))
    ay = np.where(pos, np.maximum(np.maximum(cc, 0) ** 2, np.minimum(d, 0) ** 2),
                  np.maximum(np.minimum(cc, 0) ** 2, np.maximum(d, 0) ** 2))
    return np.sqrt(ax + ay)


def advect(phi, vn, dt, dx):
    """One Heun (RK2) step of phi_t + vn |grad phi| = 0. vn may be negative (melting).
    Upwinding follows the sign of vn."""
    def rhs(f):
        up = _godunov_gradnorm(f, np.where(vn >= 0, 1.0, -1.0), dx)
        return -vn * up
    k1 = rhs(phi)
    k2 = rhs(phi + dt * k1)
    return phi + 0.5 * dt * (k1 + k2)


def max_stable_dt(vn, dx, cfl=0.4):
    vmax = np.max(np.abs(vn))
    return np.inf if vmax == 0 else cfl * dx / vmax


def reinitialize(phi0, dx, iters=20, cfl=0.3):
    """Russo-Smereka reinitialisation: pseudo-time relaxation toward |grad phi| = 1 with a sub-cell
    fix at nodes next to the interface so the zero level set does not move."""
    p0 = _pad(phi0)
    c0 = p0[1:-1, 1:-1]
    sgn = np.sign(c0)
    sgn[sgn == 0] = 1.0
    nb = [p0[2:, 1:-1], p0[:-2, 1:-1], p0[1:-1, 2:], p0[1:-1, :-2]]
    near = np.zeros(c0.shape, bool)
    for q in nb:
        near |= (q * c0 < 0)
    near |= (c0 == 0)
    gx0, gy0 = central_gradient(phi0, dx)
    # distance estimate from the largest one-sided slope, robust where the central gradient vanishes
    s = np.maximum.reduce([np.abs(c0 - q) for q in nb]) / dx
    s = np.maximum(s, 1e-8)
    gnorm = np.maximum(s, np.sqrt(gx0**2 + gy0**2))
    dist = np.abs(c0) / gnorm
    phi = phi0.copy()
    dt = cfl * dx
    for _ in range(iters):
        g = np.where(sgn > 0, _godunov_gradnorm(phi, np.ones_like(phi), dx),
                     _godunov_gradnorm(phi, -np.ones_like(phi), dx))
        upd = -sgn * (g - 1.0) * dt
        # interface nodes: relax toward the fixed distance estimate
        sub = -sgn * (np.abs(phi) - dist) / dx * dt
        phi = phi + np.where(near, sub, upd)
    return phi


def crossings(phi, grid):
    """Edge crossings of the zero level set. Returns arrays (x, y, i, j, axis, theta) where the edge
    runs from node (i, j) to (i+1, j) for axis 0 or (i, j+1) for axis 1, theta in [0,1] is the
    fractional distance from the first node (linear interpolation of phi)."""
    out = {"x": [], "y": [], "i": [], "j": [], "axis": [], "theta": []}
    X, Y = grid.mesh()
    for axis in (0, 1):
        a = phi[:-1, :] if axis == 0 else phi[:, :-1]
        b = phi[1:, :] if axis == 0 else phi[:, 1:]
        ii, jj = np.nonzero(a * b < 0)
        th = a[ii, jj] / (a[ii, jj] - b[ii, jj])
        x = X[ii, jj] + (th * grid.dx if axis == 0 else 0.0)
        y = Y[ii, jj] + (th * grid.dx if axis == 1 else 0.0)
        out["x"].append(x); out["y"].append(y); out["i"].append(ii); out["j"].append(jj)
        out["axis"].append(np.full(ii.shape, axis)); out["theta"].append(th)
    return {k: np.concatenate(v) for k, v in out.items()}
