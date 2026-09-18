"""Incremental-potential time step (docs/01 §3) for small particle systems.

A deliberately tiny, dense, CPU-only reference: disc-shaped particles in 2D,
springs, IPC barrier between particles and against planes, lagged smoothed
friction, gravity, and constant external forces. Each step minimises

    E(x) = sum_i m_i |x_i - x~_i|^2 / (2 h^2) + springs + barriers + friction
         + sum of anchor penalties

with Newton + backtracking and a CCD bound so no step crosses a contact.
It exists to test the *design* against physical laws, not to be fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contact import barrier, barrier_grad, f0, f1_over_y


@dataclass
class Plane:
    point: np.ndarray            # a point on the plane
    normal: np.ndarray           # unit normal pointing into the free side
    mu: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    owner: int | None = None     # if set, the plane is a flat pad carried by this particle;
                                 # `point` is then relative to the owner's position


@dataclass
class System:
    x: np.ndarray                # (n, 2) positions
    v: np.ndarray                # (n, 2) velocities
    m: np.ndarray                # (n,) masses
    r: np.ndarray                # (n,) radii
    springs: list[tuple[int, int, float, float]] = field(default_factory=list)  # (i, j, k, rest)
    planes: list[Plane] = field(default_factory=list)
    pair_mu: float = 0.0         # friction between particles
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    f_ext: np.ndarray | None = None          # (n, 2) constant external forces
    anchors: list[tuple[int, int, float, float]] = field(default_factory=list)  # (i, axis, value, k)
    dhat: float = 1e-3
    kappa: float = 1e3
    eps_v: float = 1e-3          # friction smoothing velocity

    @property
    def n(self) -> int:
        return len(self.m)


class Step:
    """One implicit step from (x_n, v_n); holds lagged friction data."""

    def __init__(self, s: System, h: float):
        self.s, self.h = s, h
        self.xn = s.x.copy()
        fext = s.f_ext if s.f_ext is not None else np.zeros_like(s.x)
        self.xt = s.x + h * s.v + h * h * (s.gravity[None, :] + fext / s.m[:, None])
        self.lag: list[tuple] = []
        self.path: list[np.ndarray] = []   # every configuration the solver visits

    # --- contact geometry -------------------------------------------------
    def _plane_d(self, x, i, p: Plane):
        origin = p.point + (x[p.owner] if p.owner is not None else 0.0)
        return float(np.dot(x[i] - origin, p.normal)) - self.s.r[i]

    def _pair_d(self, x, i, j):
        return float(np.linalg.norm(x[i] - x[j])) - self.s.r[i] - self.s.r[j]

    def contacts(self, x):
        s = self.s
        out = []
        for i in range(s.n):
            for k, p in enumerate(s.planes):
                if p.owner == i:
                    continue
                d = self._plane_d(x, i, p)
                if d < s.dhat:
                    out.append(("plane", i, k, d))
            for j in range(i + 1, s.n):
                d = self._pair_d(x, i, j)
                if d < s.dhat:
                    out.append(("pair", i, j, d))
        return out

    def update_lag(self, x):
        """Freeze normal force magnitude and tangent for friction (IPC lagging)."""
        s = self.s
        self.lag = []
        for kind, i, j, d in self.contacts(x):
            lam = -barrier_grad(d, s.dhat, s.kappa)
            if kind == "plane":
                p = s.planes[j]
                nrm = p.normal
                mu = p.mu
            else:
                nrm = (x[i] - x[j]) / np.linalg.norm(x[i] - x[j])
                mu = s.pair_mu
            if mu > 0 and lam > 0:
                tan = np.array([-nrm[1], nrm[0]])
                self.lag.append((kind, i, j, mu * lam, tan))

    # --- energy and gradient ---------------------------------------------
    def energy_grad(self, x):
        s, h = self.s, self.h
        e = 0.5 * np.sum(s.m[:, None] * (x - self.xt) ** 2) / h ** 2
        g = s.m[:, None] * (x - self.xt) / h ** 2
        for i, j, k, rest in s.springs:
            dx = x[i] - x[j]
            L = np.linalg.norm(dx)
            e += 0.5 * k * (L - rest) ** 2
            gi = k * (L - rest) * dx / L
            g[i] += gi
            g[j] -= gi
        for i, axis, val, k in s.anchors:
            e += 0.5 * k * (x[i, axis] - val) ** 2
            g[i, axis] += k * (x[i, axis] - val)
        for kind, i, j, d in self.contacts(x):
            b = barrier(d, s.dhat, s.kappa)
            if not np.isfinite(b):
                return np.inf, None
            e += b
            db = barrier_grad(d, s.dhat, s.kappa)
            if kind == "plane":
                g[i] += db * s.planes[j].normal
                if s.planes[j].owner is not None:
                    g[s.planes[j].owner] -= db * s.planes[j].normal
            else:
                nrm = (x[i] - x[j]) / np.linalg.norm(x[i] - x[j])
                g[i] += db * nrm
                g[j] -= db * nrm
        eps = s.eps_v * h
        for kind, i, j, mulam, tan in self.lag:
            o = j if kind == "pair" else s.planes[j].owner
            u = _slip(x, self.xn, i, o, tan, h, None if kind == "pair" else s.planes[j])
            e += mulam * f0(abs(u), eps)
            gu = mulam * f1_over_y(abs(u), eps) * u * tan
            g[i] += gu
            if o is not None:
                g[o] -= gu
        return e, g

    def ccd_bound(self, x, dx):
        """Largest alpha in (0, 1] keeping every contact distance positive."""
        s = self.s
        alpha = 1.0
        for i in range(s.n):
            for p in s.planes:
                if p.owner == i:
                    continue
                d0 = self._plane_d(x, i, p)
                rate = float(np.dot(dx[i] - (dx[p.owner] if p.owner is not None else 0.0), p.normal))
                if rate < 0:
                    alpha = min(alpha, 0.9 * d0 / -rate)
            for j in range(i + 1, s.n):
                p0 = x[i] - x[j]
                dp = dx[i] - dx[j]
                R = s.r[i] + s.r[j]
                a, b, c = dp @ dp, 2 * p0 @ dp, p0 @ p0 - R * R
                if a > 0:
                    disc = b * b - 4 * a * c
                    if disc > 0:
                        t = (-b - np.sqrt(disc)) / (2 * a)
                        if t > 0:
                            alpha = min(alpha, 0.9 * t)
        return alpha

    def solve(self, tol=1e-9, max_iter=200, lag_updates=3):
        x = self.xn.copy()
        self.path = [x.copy()]
        scale = max(1.0, float(np.max(self.s.m))) / self.h ** 2
        for _ in range(lag_updates):
            self.update_lag(x)
            for _ in range(max_iter):
                e, g = self.energy_grad(x)
                if np.linalg.norm(g) / scale < tol:
                    break
                H = self._hessian(x)
                gf = g.ravel()
                try:
                    # Saddle-free Newton: |eigenvalues|, floored at the inertia
                    # term m/h^2 that any PSD projection would keep. Clamping a
                    # negative curvature to ~0 instead (barrier curvature is
                    # negative tangentially) produces huge spurious steps.
                    w, V = np.linalg.eigh(H)
                    w = np.maximum(np.abs(w), float(np.min(self.s.m)) / self.h ** 2)
                    dx = -(V @ ((V.T @ gf) / w)).reshape(x.shape)
                except np.linalg.LinAlgError:
                    dx = -g / scale
                alpha = self.ccd_bound(x, dx)
                while alpha > 1e-12:
                    e_new, _ = self.energy_grad(x + alpha * dx)
                    if e_new <= e + 1e-4 * alpha * float(gf @ dx.ravel()):
                        break
                    alpha *= 0.5
                x = x + alpha * dx
                self.path.append(x.copy())
        return x

    def _hessian(self, x):
        n = x.size
        H = np.zeros((n, n))
        step = 1e-7 * max(1.0, float(np.max(np.abs(x))))
        flat = x.ravel()
        for k in range(n):
            xp, xm = flat.copy(), flat.copy()
            xp[k] += step
            xm[k] -= step
            _, gp = self.energy_grad(xp.reshape(x.shape))
            _, gm = self.energy_grad(xm.reshape(x.shape))
            if gp is None or gm is None:
                H[k, k] = 1.0
                continue
            H[:, k] = (gp - gm).ravel() / (2 * step)
        return 0.5 * (H + H.T)


def _slip(x, xn, i, o, tan, h, plane):
    """Tangential slip of particle i relative to its contact partner over the step."""
    rel = x[i] - xn[i]
    if o is not None:
        rel = rel - (x[o] - xn[o])
    elif plane is not None:
        rel = rel - h * plane.velocity
    return float(np.dot(rel, tan))


def step(s: System, h: float) -> dict:
    """Advance the system in place; return per-step diagnostics."""
    st = Step(s, h)
    x_new = st.solve()
    friction_work = 0.0   # work done by friction during the step, using the lag it solved with
    eps = s.eps_v * h
    for kind, i, j, mulam, tan in st.lag:
        o = j if kind == "pair" else s.planes[j].owner
        u = _slip(x_new, s.x, i, o, tan, h, None if kind == "pair" else s.planes[j])
        # work done *by* friction on the slip displacement
        friction_work -= mulam * f1_over_y(abs(u), eps) * u * u
    s.v = (x_new - s.x) / h
    s.x = x_new
    return {"friction_work": friction_work, "path": st.path,
            "min_gap": min((d for *_, d in st.contacts(x_new)), default=np.inf)}


def contact_forces(s: System) -> dict[tuple, np.ndarray]:
    """Barrier forces on each particle of each active pair at the current state."""
    out = {}
    st = Step(s, 1.0)
    for kind, i, j, d in st.contacts(s.x):
        if kind != "pair":
            continue
        nrm = (s.x[i] - s.x[j]) / np.linalg.norm(s.x[i] - s.x[j])
        f = -barrier_grad(d, s.dhat, s.kappa) * nrm
        out[(i, j)] = (f, -f)
    return out


def kinetic(s: System) -> float:
    return 0.5 * float(np.sum(s.m[:, None] * s.v ** 2))


def spring_energy(s: System) -> float:
    return sum(0.5 * k * (np.linalg.norm(s.x[i] - s.x[j]) - rest) ** 2 for i, j, k, rest in s.springs)


def momentum(s: System) -> np.ndarray:
    return np.sum(s.m[:, None] * s.v, axis=0)


def angular_momentum(s: System) -> float:
    return float(np.sum(s.m * (s.x[:, 0] * s.v[:, 1] - s.x[:, 1] * s.v[:, 0])))
