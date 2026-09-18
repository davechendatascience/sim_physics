"""Constitutive models from docs/03 §2, in their smallest faithful form.

- Compressible Neo-Hookean hyperelasticity (finite strain).
- J2 plasticity with isotropic linear + Voce hardening (small strain, radial
  return). The finite-strain F = F_e F_p version reduces to this for small
  strains; the law tests target the return map's thermodynamics.
- Hill48 anisotropic yield from Lankford r-values.
- Isothermal ideal gas in a closed volume (docs/03 §5).
"""

from __future__ import annotations

import numpy as np

R_GAS = 8.314462618


def lame(E: float, nu: float) -> tuple[float, float]:
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lam


# --- Neo-Hookean ---------------------------------------------------------

def nh_energy(F: np.ndarray, mu: float, lam: float) -> float:
    J = np.linalg.det(F)
    if J <= 0:
        return np.inf
    lnJ = np.log(J)
    return 0.5 * mu * (np.trace(F.T @ F) - 3) - mu * lnJ + 0.5 * lam * lnJ ** 2


def nh_pk1(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    J = np.linalg.det(F)
    FinvT = np.linalg.inv(F).T
    return mu * (F - FinvT) + lam * np.log(J) * FinvT


def cauchy(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    return P @ F.T / np.linalg.det(F)


def nh_tangent(F: np.ndarray, mu: float, lam: float, h: float = 1e-6) -> np.ndarray:
    """dP/dF as a 9x9 matrix (central differences of the analytic stress)."""
    A = np.zeros((9, 9))
    for k in range(9):
        dF = np.zeros(9)
        dF[k] = h
        A[:, k] = (nh_pk1(F + dF.reshape(3, 3), mu, lam) - nh_pk1(F - dF.reshape(3, 3), mu, lam)).ravel() / (2 * h)
    return A


def acoustic_min_eig(F: np.ndarray, mu: float, lam: float, n_dirs: int = 64, rng=None) -> float:
    """min over directions of the acoustic tensor's smallest eigenvalue (ellipticity)."""
    rng = rng or np.random.default_rng(0)
    A = nh_tangent(F, mu, lam).reshape(3, 3, 3, 3)
    worst = np.inf
    for _ in range(n_dirs):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        Q = np.einsum("iJkL,J,L->ik", A, n, n)
        worst = min(worst, float(np.linalg.eigvalsh(0.5 * (Q + Q.T)).min()))
    return worst


# --- J2 plasticity (small strain) ---------------------------------------

class J2:
    """Radial return with hardening sigma_y(a) = s0 + H a + Q (1 - exp(-b a))."""

    def __init__(self, E, nu, sigma_y0, H=0.0, Q=0.0, b=1.0):
        self.mu, self.lam = lame(E, nu)
        self.K = self.lam + 2 * self.mu / 3
        self.s0, self.H, self.Q, self.b = sigma_y0, H, Q, b
        self.eps_p = np.zeros((3, 3))
        self.alpha = 0.0

    def sigma_y(self, a):
        return self.s0 + self.H * a + self.Q * (1 - np.exp(-self.b * a))

    def stress_trial(self, eps):
        ee = eps - self.eps_p
        return self.lam * np.trace(ee) * np.eye(3) + 2 * self.mu * ee

    def yield_fn(self, sig, a=None):
        s = sig - np.trace(sig) / 3 * np.eye(3)
        return np.sqrt(1.5) * np.linalg.norm(s) - self.sigma_y(self.alpha if a is None else a)

    def update(self, eps):
        """Strain-driven update. Returns (stress, plastic strain increment)."""
        sig = self.stress_trial(eps)
        f = self.yield_fn(sig)
        if f <= 0:
            return sig, np.zeros((3, 3))
        s = sig - np.trace(sig) / 3 * np.eye(3)
        q = np.sqrt(1.5) * np.linalg.norm(s)
        nflow = 1.5 * s / q
        dg = 0.0
        for _ in range(50):   # Newton on the consistency condition (Voce is nonlinear)
            r = q - 3 * self.mu * dg - self.sigma_y(self.alpha + dg)
            dr = -3 * self.mu - self.H - self.Q * self.b * np.exp(-self.b * (self.alpha + dg))
            step = r / dr
            dg -= step
            if abs(step) < 1e-14 * max(1.0, dg):
                break
        deps_p = dg * nflow
        self.eps_p = self.eps_p + deps_p
        self.alpha += dg
        return self.stress_trial(eps), deps_p


# --- Hill48 --------------------------------------------------------------

def hill48_from_r(r0: float, r45: float, r90: float) -> dict[str, float]:
    """Hill48 coefficients (normalised to the rolling-direction yield stress)."""
    return {
        "F": r0 / (r90 * (1 + r0)),
        "G": 1 / (1 + r0),
        "H": r0 / (1 + r0),
        "N": (r0 + r90) * (1 + 2 * r45) / (2 * r90 * (1 + r0)),
        "L": 1.5, "M": 1.5,   # out-of-plane shear: isotropic default for sheet
    }


def hill48_matrix(c: dict[str, float]) -> np.ndarray:
    """Quadratic form on (s11, s22, s33, s23, s31, s12): f^2 = s^T A s."""
    F, G, H, L, M, N = (c[k] for k in "FGHLMN")
    A = np.zeros((6, 6))
    A[:3, :3] = [[G + H, -H, -G], [-H, F + H, -F], [-G, -F, F + G]]
    A[3, 3], A[4, 4], A[5, 5] = 2 * L, 2 * M, 2 * N
    return A


# --- Enclosed ideal gas (isothermal) --------------------------------------

def gas_energy(V: float, n_mol: float, T: float) -> float:
    """Helmholtz free energy of an isothermal ideal gas, up to a constant."""
    return -n_mol * R_GAS * T * np.log(V)


def gas_pressure(V: float, n_mol: float, T: float) -> float:
    return n_mol * R_GAS * T / V
