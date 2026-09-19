"""Reference models for the 3D simulator's pieces (docs/09 §2).

- Through-thickness integration rules and a plane-stress J2 fiber section.
- 3D affine body: kinetic energy and orthogonality potential.
- Point-triangle and edge-edge distances with gradients; conservative CCD bound.
- Enclosed-volume isothermal gas.
"""

from __future__ import annotations

import numpy as np

# --- through-thickness integration on xi in [-1, 1] (weights sum to 2) ----------

def rule(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "simpson9":            # composite Simpson, 4 panels, split at the mid-surface
        return np.linspace(-1, 1, 9), np.array([1, 4, 2, 4, 2, 4, 2, 4, 1]) * (0.25 / 3)
    if name == "lobatto5":
        return (np.array([-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1]),
                np.array([0.1, 49 / 90, 32 / 45, 49 / 90, 0.1]))
    if name == "lobatto7":
        return (np.array([-1, -0.830223896278567, -0.468848793470714, 0,
                          0.468848793470714, 0.830223896278567, 1]),
                np.array([1 / 21, 0.276826047361566, 0.431745381209863, 0.487619047619048,
                          0.431745381209863, 0.276826047361566, 1 / 21]))
    raise ValueError(name)


# --- plane-stress J2 --------------------------------------------------------

def plane_stress(E, nu, eps):
    """sigma for 2x2 strain tensors (..., 2, 2)."""
    tr = eps[..., 0, 0] + eps[..., 1, 1]
    return E / (1 - nu ** 2) * ((1 - nu) * eps + nu * tr[..., None, None] * np.eye(2))


def von_mises_ps(sig):
    s11, s22, s12 = sig[..., 0, 0], sig[..., 1, 1], sig[..., 0, 1]
    return np.sqrt(np.maximum(s11 ** 2 - s11 * s22 + s22 ** 2 + 3 * s12 ** 2, 0.0))


def return_map_ps(E, nu, sy0, H, eps, eps_p, alpha):
    """Plane-stress J2 radial return (Simo & Taylor spectral form), vectorised.

    Returns (sigma, eps_p_new, alpha_new). The trial stress is split into
    (s11+s22), (s11-s22) and s12, which scale by 1/(1 + lam E/(2(1-nu))) and
    1/(1 + 3 G lam) respectively; lam solves the consistency condition by
    bisection, which is robust and vectorises.
    """
    G = E / (2 * (1 + nu))
    trial = plane_stress(E, nu, eps - eps_p)
    sp = trial[..., 0, 0] + trial[..., 1, 1]
    sm = trial[..., 0, 0] - trial[..., 1, 1]
    s12 = trial[..., 0, 1]

    def state(lam):
        a = sp / (1 + lam * E / (2 * (1 - nu)))
        b = sm / (1 + 3 * G * lam)
        c = s12 / (1 + 3 * G * lam)
        vm = np.sqrt(a ** 2 / 4 + 3 * b ** 2 / 4 + 3 * c ** 2)
        return a, b, c, vm

    vm_tr = state(0.0)[3]
    plastic = vm_tr > sy0 + H * alpha
    lo = np.zeros_like(vm_tr)
    hi = np.where(plastic, 1.0 / min(E / (2 * (1 - nu)), 3 * G), 0.0)
    for _ in range(200):                       # expand until the upper bound is feasible
        vm = state(hi)[3]
        bad = plastic & (vm - (sy0 + H * (alpha + hi * vm)) > 0)
        if not bad.any():
            break
        hi = np.where(bad, hi * 4, hi)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        vm = state(mid)[3]
        over = vm - (sy0 + H * (alpha + mid * vm)) > 0
        lo = np.where(plastic & over, mid, lo)
        hi = np.where(plastic & ~over, mid, hi)
    lam = np.where(plastic, 0.5 * (lo + hi), 0.0)
    a, b, c, vm = state(lam)
    sig = np.zeros_like(trial)
    sig[..., 0, 0], sig[..., 1, 1] = (a + b) / 2, (a - b) / 2
    sig[..., 0, 1] = sig[..., 1, 0] = c
    # elastic strain from the returned stress; the rest is plastic
    ee = np.zeros_like(sig)
    ee[..., 0, 0] = (sig[..., 0, 0] - nu * sig[..., 1, 1]) / E
    ee[..., 1, 1] = (sig[..., 1, 1] - nu * sig[..., 0, 0]) / E
    ee[..., 0, 1] = ee[..., 1, 0] = sig[..., 0, 1] / (2 * G)
    return sig, eps - ee, alpha + lam * vm


class FiberSection:
    """Per unit area of shell: fibers at z = xi t/2, plane-stress J2."""

    def __init__(self, E, nu, sigma_y, t, H=0.0, rule_name="simpson9"):
        self.E, self.nu, self.sy, self.H, self.t = E, nu, sigma_y, H, t
        xi, w = rule(rule_name)
        self.z, self.w = xi * t / 2, w * t / 2
        n = len(self.z)
        self.eps_p = np.zeros((n, 2, 2))
        self.alpha = np.zeros(n)

    def strains(self, eps_m, kappa):
        return eps_m[None] + self.z[:, None, None] * kappa[None]

    def stored_energy(self, eps_m, kappa):
        ee = self.strains(eps_m, kappa) - self.eps_p
        sig = plane_stress(self.E, self.nu, ee)
        return 0.5 * float(np.sum(self.w * np.einsum("gij,gij->g", sig, ee)))

    def update(self, eps_m, kappa):
        """Return map; returns (N 2x2, M 2x2, dissipated energy per area)."""
        eps = self.strains(eps_m, kappa)
        sig, eps_p_new, alpha_new = return_map_ps(self.E, self.nu, self.sy, self.H, eps,
                                                  self.eps_p, self.alpha)
        diss = float(np.sum(self.w * np.einsum("gij,gij->g", sig, eps_p_new - self.eps_p)))
        self.eps_p, self.alpha = eps_p_new, alpha_new
        N = np.einsum("g,gij->ij", self.w, sig)
        M = np.einsum("g,g,gij->ij", self.w, self.z, sig)
        return N, M, diss

    def max_yield_violation(self, eps_m, kappa):
        sig = plane_stress(self.E, self.nu, self.strains(eps_m, kappa) - self.eps_p)
        return float(np.max(von_mises_ps(sig) - (self.sy + self.H * self.alpha)))


# --- 3D affine body ----------------------------------------------------------

def box_mass_properties(size, rho):
    """Mass and second moment J = int rho X X^T dV of a centred box."""
    a, b, c = size
    m = rho * a * b * c
    return m, m / 12 * np.diag([a * a, b * b, c * c])


def affine_kinetic(m, J, pdot, Adot):
    return 0.5 * m * pdot @ pdot + 0.5 * np.trace(Adot @ J @ Adot.T)


def orthogonality_energy(A, kappa, V):
    """The affine-body potential of the first design (kept for reference, docs/02 §1)."""
    D = A.T @ A - np.eye(3)
    return kappa * V * np.sum(D * D)


def rotation_exp(w):
    """exp([w]x) by Rodrigues' formula, re-orthonormalised to machine precision."""
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        R = np.eye(3) + K
    else:
        R = np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


# --- distances -----------------------------------------------------------------

def point_triangle(p, t0, t1, t2):
    """Distance and gradient w.r.t. (p, t0, t1, t2) for the closest point on the triangle."""
    best = None
    e1, e2 = t1 - t0, t2 - t0
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n)
    # interior projection
    q = p - np.dot(p - t0, n) * n
    M = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
    b1, b2 = np.linalg.solve(M, [e1 @ (q - t0), e2 @ (q - t0)])
    cands = []
    if b1 >= 0 and b2 >= 0 and b1 + b2 <= 1:
        cands.append(np.array([1 - b1 - b2, b1, b2]))
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        a, c = (t0, t1, t2)[i], (t0, t1, t2)[j]
        s = np.clip(np.dot(p - a, c - a) / np.dot(c - a, c - a), 0, 1)
        beta = np.zeros(3)
        beta[i], beta[j] = 1 - s, s
        cands.append(beta)
    for beta in cands:
        c = beta[0] * t0 + beta[1] * t1 + beta[2] * t2
        d = np.linalg.norm(p - c)
        if best is None or d < best[0]:
            best = (d, beta, c)
    d, beta, c = best
    nr = (p - c) / d
    return d, np.concatenate([nr, -beta[0] * nr, -beta[1] * nr, -beta[2] * nr])


def edge_edge(a0, a1, b0, b1):
    """Distance between segments and gradient w.r.t. (a0, a1, b0, b1)."""
    da, db, r = a1 - a0, b1 - b0, a0 - b0
    A, B, C, D, E = da @ da, da @ db, db @ db, da @ r, db @ r
    den = A * C - B * B
    s = np.clip((B * E - C * D) / den, 0, 1) if den > 1e-14 * A * C else 0.0
    u = (B * s + E) / C
    if u < 0 or u > 1:
        u = np.clip(u, 0, 1)
        s = np.clip((B * u - D) / A, 0, 1)
    ca, cb = a0 + s * da, b0 + u * db
    d = np.linalg.norm(ca - cb)
    nr = (ca - cb) / d
    return d, np.concatenate([(1 - s) * nr, s * nr, -(1 - u) * nr, -u * nr])


def ccd_bound(d0, disp_a, disp_b, offset=0.0, safety=0.9):
    """Conservative step fraction: every point of a linearly moving primitive moves
    at most as far as its farthest vertex, so the distance falls at most by the
    sum of the two primitives' largest vertex displacements."""
    rate = max(np.linalg.norm(disp_a, axis=1)) + max(np.linalg.norm(disp_b, axis=1))
    return 1.0 if rate == 0 else min(1.0, safety * (d0 - offset) / rate)


# --- enclosed gas ------------------------------------------------------------

def mesh_volume(x, faces):
    a, b, c = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c)))) / 6


def gas_energy(x, faces, nRT):
    return -nRT * np.log(mesh_volume(x, faces))


def gas_force(x, faces, nRT):
    """-dPsi/dx = p dV/dx: the outward pressure force on each vertex."""
    p = nRT / mesh_volume(x, faces)
    f = np.zeros_like(x)
    a, b, c = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
    np.add.at(f, faces[:, 0], np.cross(b, c) / 6)
    np.add.at(f, faces[:, 1], np.cross(c, a) / 6)
    np.add.at(f, faces[:, 2], np.cross(a, b) / 6)
    return p * f


def ee_mollifier(a0, a1, b0, b1, a0r, a1r, b0r, b1r):
    """IPC edge-edge mollifier (docs/02 §2): (2 - c/e) c/e for c < e, else 1,
    with c = |ea x eb|^2 and e = 1e-3 |ea_rest|^2 |eb_rest|^2."""
    c = np.sum(np.cross(a1 - a0, b1 - b0) ** 2)
    e = 1e-3 * np.sum((a1r - a0r) ** 2) * np.sum((b1r - b0r) ** 2)
    return (2 - c / e) * c / e if c < e else 1.0
