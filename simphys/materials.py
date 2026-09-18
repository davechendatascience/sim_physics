"""Materials and the shell plasticity return map (docs/03 §2, docs/09 §2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    rho: float                       # kg/m^3
    E: float = 1e9                   # Pa
    nu: float = 0.3
    sigma_y: float = np.inf          # Pa; inf = elastic
    H: float = 0.0                   # linear isotropic hardening modulus, Pa
    friction: float = 0.5            # Coulomb coefficient (pairs use the geometric mean)

    @property
    def lame(self):
        mu = self.E / (2 * (1 + self.nu))
        return mu, self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


ALUMINUM_CAN = Material("AA3004-H19 can body", rho=2720.0, E=69e9, nu=0.33, sigma_y=285e6,
                        H=0.0, friction=0.8)
STEEL = Material("steel", rho=7850.0, E=210e9, nu=0.3, friction=0.5)
WOOD = Material("wood", rho=600.0, E=10e9, nu=0.3, friction=0.5)
SILICONE = Material("silicone pad", rho=1100.0, E=1e6, nu=0.45, friction=0.8)
GROUND = Material("ground", rho=2000.0, E=30e9, nu=0.2, friction=0.6)


def plane_stress_return(E, nu, sy, H, eps, eps_p, alpha, iters=80):
    """Vectorised plane-stress J2 return map.

    eps, eps_p: (..., 2, 2) total and plastic strain; alpha: (...,) equivalent
    plastic strain. The trial stress's (s11+s22), (s11-s22) and s12 parts shrink
    by 1/(1 + lam E / (2(1 - nu))) and 1/(1 + 3 G lam); lam is found by
    bisection on the consistency condition. Returns (sigma, eps_p, alpha).
    """
    G = E / (2 * (1 + nu))
    ee = eps - eps_p
    tr = ee[..., 0, 0] + ee[..., 1, 1]
    c = E / (1 - nu ** 2)
    s11 = c * (ee[..., 0, 0] + nu * ee[..., 1, 1])
    s22 = c * (ee[..., 1, 1] + nu * ee[..., 0, 0])
    s12 = 2 * G * ee[..., 0, 1]
    sp, sm = s11 + s22, s11 - s22
    kp, km = E / (2 * (1 - nu)), 3 * G

    def vm(lam):
        a, b, d = sp / (1 + lam * kp), sm / (1 + lam * km), s12 / (1 + lam * km)
        return np.sqrt(a * a / 4 + 3 * b * b / 4 + 3 * d * d)

    yield0 = sy + H * alpha
    plastic = vm(0.0) > yield0
    if not np.any(plastic):
        sig = np.stack([np.stack([s11, s12], -1), np.stack([s12, s22], -1)], -2)
        return sig, eps_p, alpha
    lo = np.zeros_like(tr)
    hi = np.where(plastic, 1.0 / min(kp, km), 0.0)
    for _ in range(100):                     # grow the bracket until it holds the root
        v = vm(hi)
        grow = plastic & (v - (sy + H * (alpha + hi * v)) > 0)
        if not grow.any():
            break
        hi = np.where(grow, 2 * hi, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = vm(mid)
        over = v - (sy + H * (alpha + mid * v)) > 0
        lo = np.where(plastic & over, mid, lo)
        hi = np.where(plastic & ~over, mid, hi)
    lam = np.where(plastic, hi, 0.0)          # hi side: never outside the yield surface
    a, b, d = sp / (1 + lam * kp), sm / (1 + lam * km), s12 / (1 + lam * km)
    n11, n22 = (a + b) / 2, (a - b) / 2
    sig = np.stack([np.stack([n11, d], -1), np.stack([d, n22], -1)], -2)
    e11 = (n11 - nu * n22) / E
    e22 = (n22 - nu * n11) / E
    e12 = d / (2 * G)
    elastic = np.stack([np.stack([e11, e12], -1), np.stack([e12, e22], -1)], -2)
    return sig, eps - elastic, alpha + lam * vm(lam)
