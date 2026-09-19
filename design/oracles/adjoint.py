"""Reference model for differentiable stepping (docs/12).

A two-particle spring step in 2D: E(x; k) = sum m|x - x~|^2/(2h^2) + k/2 (|d| - L)^2.
Everything is analytic, so the implicit derivative dx*/dk can be formed with
the exact Hessian or with per-stencil PSD projection, and compared with
finite differences of the converged step.
"""

from __future__ import annotations

import numpy as np


class SpringStep:
    def __init__(self, x_tilde, m, h, L):
        self.xt, self.m, self.h, self.L = np.asarray(x_tilde, float), m, h, L

    def grad(self, x, k):
        x = x.reshape(2, 2)
        d = x[0] - x[1]
        r = np.linalg.norm(d)
        f = k * (r - self.L) * d / r
        g = self.m * (x - self.xt) / self.h ** 2
        g[0] += f
        g[1] -= f
        return g.ravel()

    def spring_hessian(self, x, k):
        """4x4 Hessian of the spring stencil (exact)."""
        x = x.reshape(2, 2)
        d = x[0] - x[1]
        r = np.linalg.norm(d)
        n = d / r
        Hs = k * (np.outer(n, n) + (1 - self.L / r) * (np.eye(2) - np.outer(n, n)))
        return np.block([[Hs, -Hs], [-Hs, Hs]])

    def hessian(self, x, k, project=False):
        Hs = self.spring_hessian(x, k)
        if project:
            w, V = np.linalg.eigh(Hs)
            Hs = V @ np.diag(np.maximum(w, 0)) @ V.T
        return self.m / self.h ** 2 * np.eye(4) + Hs

    def dgrad_dk(self, x):
        x = x.reshape(2, 2)
        d = x[0] - x[1]
        r = np.linalg.norm(d)
        f = (r - self.L) * d / r
        return np.concatenate([f, -f])

    def solve(self, k, tol=1e-13, x0=None, max_iter=200):
        """Newton with the exact Hessian; returns (x*, residual norm)."""
        x = self.xt.ravel().copy() if x0 is None else x0.copy()
        for _ in range(max_iter):
            g = self.grad(x, k)
            if np.linalg.norm(g) < tol:
                break
            H = self.hessian(x, k)
            w = np.linalg.eigvalsh(H)
            if w.min() <= 0:
                H = H + (1e-9 - w.min()) * np.eye(4)
            x = x - np.linalg.solve(H, g)
        return x, np.linalg.norm(self.grad(x, k))

    def implicit_dx_dk(self, x, k, project=False):
        return -np.linalg.solve(self.hessian(x, k, project), self.dgrad_dk(x))

    def implicit_dx_dxt(self, x, k, project=False):
        """dx*/dx~ (4x4): the step's dependence on the previous state, which an
        adjoint through time multiplies step after step. d grad/d x~ = -m/h^2 I."""
        return np.linalg.solve(self.hessian(x, k, project), self.m / self.h ** 2 * np.eye(4))

    def fd_dx_dxt(self, k, eps=1e-7):
        J = np.zeros((4, 4))
        base = self.xt.copy()
        for j in range(4):
            for sgn in (1, -1):
                xt = base.ravel().copy()
                xt[j] += sgn * eps
                self.xt = xt.reshape(2, 2)
                J[:, j] += sgn * self.solve(k)[0] / (2 * eps)
        self.xt = base
        return J


def fiber_stress(eps, E, sy, H):
    """1D elastoplastic fiber under monotonic loading: the exact forward law."""
    ey = sy / E
    return np.where(eps <= ey, E * eps, sy + E * H / (E + H) * (eps - ey))


def fiber_tangent_smoothed(eps, E, sy, H, width):
    """Backward-only derivative: the one-sided tangents blended over `width`
    around first yield. The forward law above is not touched."""
    ey = sy / E
    from scipy.special import expit
    w = expit((eps - ey) / width)
    return (1 - w) * E + w * E * H / (E + H)


def compressed_example():
    """Two masses driven together: at x* the spring is compressed, so its
    transverse curvature k(1 - L/r) is negative while the step is stable."""
    xt = np.array([[-0.3, 0.02], [0.3, -0.02]])
    return SpringStep(xt, m=1.0, h=0.05, L=1.0), 300.0
