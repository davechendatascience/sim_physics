"""Reference model: an inextensible elastic ring pinched between two flat plates
(docs/11). Solved by quadrature and root-finding only, with no closed forms, so
the closed forms in docs/11 can be checked against it.

Units: R = ring radius, EI = bending stiffness per unit length. The load P is
per unit length, and lam^2 = P / (2 EI).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _arc(kA, lam, weight):
    """Integral over theta in (0, pi/2) of weight(theta) / theta'(theta)."""
    # theta = u^2 removes the 1/sqrt(sin theta) endpoint singularity when kA = 0
    f = lambda u: 2 * u * weight(u * u) / np.sqrt(kA ** 2 + 2 * lam ** 2 * np.sin(u * u))
    return quad(f, 0, np.sqrt(np.pi / 2), limit=200, epsabs=0, epsrel=1e-13)[0]


def flat_onset_lam(R):
    """lam at which the contact curvature reaches zero (line -> flat contact)."""
    return _arc(0.0, 1.0, lambda t: 1.0) / (np.pi * R / 2)


def state(P, R, EI):
    """Quarter-ring state under pad load P per length.

    Returns dict with gap (pad separation), contact curvature kA, flat contact
    length c (per pad, full width), side curvature kB, quarter arc length.
    """
    lam = np.sqrt(P / (2 * EI))
    L = np.pi * R / 2
    lam0 = flat_onset_lam(R)
    if lam < lam0:                                   # phase 1: line contact
        kA = 1 / R if lam == 0 else brentq(lambda k: _arc(k, lam, lambda t: 1.0) - L, 1e-12, 10 / R)
        height = _arc(kA, lam, np.sin)
        arc = _arc(kA, lam, lambda t: 1.0)
        c = 0.0
    else:                                            # phase 2: flat contact
        kA = 0.0
        free = _arc(0.0, lam, lambda t: 1.0)
        height = _arc(0.0, lam, np.sin)
        c_half = L - free
        arc = c_half + free
        c = 2 * c_half
    kB = np.sqrt(kA ** 2 + 2 * lam ** 2)
    return {"gap": 2 * height, "kA": kA, "c": c, "kB": kB, "arc": arc, "lam": lam}


def force_for_gap(gap, R, EI):
    """Invert gap(P) by bracketing (the gap decreases monotonically with P)."""
    hi = 2 * EI / R ** 2
    while state(hi, R, EI)["gap"] > gap:
        hi *= 2
    return brentq(lambda P: state(P, R, EI)["gap"] - gap, 0.0, hi, xtol=1e-14 * hi)
