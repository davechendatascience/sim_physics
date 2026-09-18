"""IPC log-barrier and smoothed Coulomb friction, as specified in docs/02 §2.

Scalar functions only; the integrator assembles them.
"""

from __future__ import annotations

import numpy as np


def barrier(d: float, dhat: float, kappa: float) -> float:
    """b(d) = -kappa (d - dhat)^2 ln(d/dhat) on (0, dhat), 0 beyond, +inf at contact."""
    if d <= 0.0:
        return np.inf
    if d >= dhat:
        return 0.0
    return -kappa * (d - dhat) ** 2 * np.log(d / dhat)


def barrier_grad(d: float, dhat: float, kappa: float) -> float:
    """db/dd. Negative on (0, dhat): the force -db/dd pushes surfaces apart."""
    if d <= 0.0 or d >= dhat:
        return 0.0
    return -kappa * (2.0 * (d - dhat) * np.log(d / dhat) + (d - dhat) ** 2 / d)


def f0(y: float, eps: float) -> float:
    """C1 smoothing of |u| used by the IPC friction potential (eps = eps_v * h)."""
    if y >= eps:
        return y
    return -y ** 3 / (3 * eps ** 2) + y ** 2 / eps + eps / 3


def f1_over_y(y: float, eps: float) -> float:
    """f0'(y)/y, finite at y = 0, so the friction gradient is smooth at rest."""
    if y >= eps:
        return 1.0 / y
    return -y / eps ** 2 + 2.0 / eps


def f1(y: float, eps: float) -> float:
    """f0'(y): friction magnitude as a fraction of mu*lambda; lies in [0, 1]."""
    return 1.0 if y >= eps else -y ** 2 / eps ** 2 + 2 * y / eps


def pressure_dependent_mu(p: float, mu0: float, p0: float, n: float) -> float:
    """mu = mu0 (p/p0)^(n-1) for elastomer pads (docs/02 §2)."""
    return mu0 * (p / p0) ** (n - 1.0)
