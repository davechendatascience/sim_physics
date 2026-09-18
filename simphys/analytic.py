"""Fast analytic tier: a can pinched between long flat pads (docs/11).

The cross-section is an inextensible elastic ring (plane strain). Flat contact
is exact in closed form through the lemniscate constant; line contact needs
one scalar root per load, solved here with fixed Gauss-Legendre quadrature so
the whole curve is vectorised. Valid for long pads, elastic, open can, up to
first yield (docs/11 §2). Independent of design/oracles/elastica.py, which it
is tested against.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma

VARPI = gamma(0.25) ** 2 / (2 * np.sqrt(2 * np.pi))          # lemniscate constant
_U, _W = np.polynomial.legendre.leggauss(96)
_UMAX = np.sqrt(np.pi / 2)
_U = 0.5 * _UMAX * (_U + 1)                                   # theta = u^2 on (0, pi/2)
_W = 0.5 * _UMAX * _W


def _integrals(kA, lam):
    """(arc length, height) of a quarter ring for arrays kA, lam."""
    th = _U ** 2
    d = np.sqrt(kA[..., None] ** 2 + 2 * lam[..., None] ** 2 * np.sin(th))
    j = 2 * _U * _W
    return np.sum(j / d, -1), np.sum(j * np.sin(th) / d, -1)


class LongPadSqueeze:
    """Force, gap, contact and first yield for a ring of radius R, wall t."""

    def __init__(self, E, nu, sigma_y, t, R):
        self.R = R
        self.EI = E / (1 - nu ** 2) * t ** 3 / 12                # per unit length
        s1y = sigma_y / np.sqrt(1 - nu + nu ** 2)                 # first-yield fiber stress (axial strain 0)
        self.dk_yield = 2 * s1y * (1 - nu ** 2) / (E * t)
        self.gap_flat = np.pi ** 2 * R / VARPI ** 2
        self.gap_yield = 2 * np.pi / (VARPI * (self.dk_yield + 1 / R))
        self.force_yield = self.EI * (self.dk_yield + 1 / R) ** 2

    def force(self, gap):
        """Pad force per unit length at pad separation `gap` (array or scalar)."""
        g = np.atleast_1d(np.asarray(gap, float))
        P = np.empty_like(g)
        flat = g <= self.gap_flat
        P[flat] = 4 * np.pi ** 2 * self.EI / (VARPI ** 2 * g[flat] ** 2)
        if (~flat).any():
            P[~flat] = self.EI / self.R ** 2 * _branch()(g[~flat] / self.R)
        return P if np.ndim(gap) else float(P[0])

    def slope(self, gap):
        """dP/dg: exact (-2P/g) in flat contact, from the fitted branch in line contact."""
        g = np.atleast_1d(np.asarray(gap, float))
        d = np.empty_like(g)
        flat = g <= self.gap_flat
        d[flat] = -2 * self.force(g[flat]) / g[flat]
        if (~flat).any():
            d[~flat] = self.EI / self.R ** 3 * _branch().deriv()(g[~flat] / self.R)
        return d if np.ndim(gap) else float(d[0])

    def contact_length(self, gap):
        """Flat contact length per pad (zero during line contact)."""
        return np.maximum(0.0, np.pi * self.R - VARPI ** 2 * np.asarray(gap, float) / np.pi)

    def altered(self, gap):
        """True where the pinch has passed first yield (a permanent change)."""
        return np.asarray(gap, float) < self.gap_yield

    def _line_contact_force_exact(self, g):
        # bisection on lam for every gap at once; kA found by bisection on arc length
        R, L = self.R, np.pi * self.R / 2
        lam0 = np.sqrt(2) * np.pi / (VARPI * self.gap_flat)       # lam where flat contact begins
        lo, hi = np.zeros_like(g), np.full_like(g, lam0)
        for _ in range(60):
            lam = 0.5 * (lo + hi)
            k_lo, k_hi = np.zeros_like(g), np.full_like(g, 1 / R)
            for _ in range(60):
                k = 0.5 * (k_lo + k_hi)
                long = _integrals(k, lam)[0] > L                    # arc too long: curvature too small
                k_lo, k_hi = np.where(long, k, k_lo), np.where(long, k_hi, k)
            gap = 2 * _integrals(0.5 * (k_lo + k_hi), lam)[1]
            wide = gap > g                                        # not squeezed enough: more load
            lo, hi = np.where(wide, lam, lo), np.where(wide, hi, lam)
        lam = 0.5 * (lo + hi)
        return 2 * self.EI * lam ** 2


_BRANCH = None


def _branch():
    """Degree-12 Chebyshev fit of the dimensionless line-contact branch
    P R^2/(E'I) against g/R on [g0/R, 2] (docs/11 §1). Material-free, so it is
    built once per process from the exact solver, in about a tenth of a second."""
    global _BRANCH
    if _BRANCH is None:
        unit = LongPadSqueeze.__new__(LongPadSqueeze)
        unit.R, unit.EI = 1.0, 1.0
        unit.gap_flat = np.pi ** 2 / VARPI ** 2
        n = 80
        x = np.cos(np.pi * (np.arange(n) + 0.5) / n) * 0.5 * (2 - unit.gap_flat) + 0.5 * (2 + unit.gap_flat)
        P = unit._line_contact_force_exact(x)
        _BRANCH = np.polynomial.chebyshev.Chebyshev.fit(x, P, 12, domain=[unit.gap_flat, 2.0])
    return _BRANCH


def can_wall():
    """The AA3004-H19 can body of docs/05."""
    return LongPadSqueeze(E=69e9, nu=0.33, sigma_y=285e6, t=1e-4, R=0.033)


def plot_curve(path, model=None):
    """Save the force-gap curve (per metre of can) as a PNG. CPU, milliseconds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = model or can_wall()
    g = np.linspace(m.gap_yield, 2 * m.R * 0.999, 400)       # elastic range only: the tier ends at first yield
    P = m.force(g)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(g * 1e3, P, color="#2c7fb8", lw=2, label="elastic ring (closed form + line-contact solve)")
    ax.axvspan(0.8 * m.gap_yield * 1e3, m.gap_yield * 1e3, color="#d95f0e", alpha=0.12)
    ax.text(m.gap_yield * 1e3, P.max() * 0.55, "permanent dent\n(beyond this tier) ", color="#d95f0e",
            fontsize=8, ha="left")
    ax.plot([m.gap_yield * 1e3], [m.force_yield], "o", color="#d95f0e")
    ax.annotate(f"first yield: {m.force_yield:.1f} N/m at {m.gap_yield * 1e3:.1f} mm",
                (m.gap_yield * 1e3, m.force_yield), textcoords="offset points", xytext=(-10, -18),
                fontsize=8, color="#d95f0e", ha="right")
    ax.axvline(m.gap_flat * 1e3, color="0.6", lw=1, ls=":")
    ax.text(m.gap_flat * 1e3, P.max() * 0.9, " flat contact begins", color="0.4", fontsize=8)
    ax.set_xlim(2 * m.R * 1e3, 0.8 * m.gap_yield * 1e3)
    ax.set_xlabel("pad gap (mm)")
    ax.set_ylabel("pad force per metre of can (N/m)")
    ax.set_title("Can pinched between long pads (docs/11)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
