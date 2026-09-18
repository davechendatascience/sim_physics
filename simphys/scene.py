"""Scene and incremental-potential time stepping (docs/01 §3, docs/02 §1, docs/09).

Each step minimises

    E = sum_deformable m |x - x~|^2 / (2 h^2) + sum_rigid [ m |p - p~|^2 / (2 h^2)
        + tr((Q - Q~) J (Q - Q~)^T) / (2 h^2) ] - gravity work + shells + solids
        + gas + drives + contact barrier + lagged friction

with Newton's method on increments: 3 per free deformable vertex and (dp, w)
per rigid body, retracted as p += dp, Q <- exp([w]) Q so rotations stay exact.
Per-stencil Hessians are projected to PSD, the step is bounded by CCD, and a
backtracking line search runs on E, which is +inf for any interpenetrating
state. Plasticity is staggered: a return map runs after the step converges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from . import collision, kernels
from .bodies import Body, Deformable, Rigid, Shell, Solid
from .materials import plane_stress_return

P_ATM = 101325.0


@dataclass
class Drive:
    """Actuator on a rigid body: per-axis springs toward target(t), a force(t) on
    its position, and a spring holding its orientation at the identity."""
    body: str
    stiffness: tuple = (0.0, 0.0, 0.0)
    target: Callable[[float], np.ndarray] | None = None
    force: Callable[[float], np.ndarray] | None = None
    rot_stiffness: float = 0.0


@dataclass
class Gas:
    """Isothermal gas sealed inside a closed shell, at gauge pressure p_gauge at rest."""
    body: str
    p_gauge: float
    nRT: float = field(init=False, default=0.0)


def psd(H):
    """Project a batch of symmetric matrices onto the PSD cone."""
    w, V = np.linalg.eigh(0.5 * (H + np.swapaxes(H, 1, 2)))
    return np.einsum("nij,nj,nkj->nik", V, np.maximum(w, 0.0), V)


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rotation_exp(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


class Scene:
    def __init__(self, bodies: list[Body], dt=1e-2, gravity=(0.0, 0.0, -9.81), dhat=1e-3,
                 kappa=1e3, eps_v=1e-3, drives=(), gases=(), newton_tol=1e-5, max_newton=60,
                 friction_iters=2):
        self.bodies = bodies
        self.by_name = {b.name: b for b in bodies}
        self.dt, self.gravity = dt, np.asarray(gravity, float)
        self.dhat, self.kappa, self.eps_v = dhat, kappa, eps_v
        self.drives, self.gases = list(drives), list(gases)
        self.newton_tol, self.max_newton, self.friction_iters = newton_tol, max_newton, friction_iters
        self.t = 0.0
        self.stats = {"newton": [], "drive_force": {d.body: [] for d in self.drives}}

        nd = nv = 0
        for b in bodies:
            b.dof0, b.vert0 = nd, nv
            nd += b.n_dofs
            nv += b.n_verts
        self.nd, self.nv = nd, nv
        self.deformables = [b for b in bodies if isinstance(b, Deformable)]
        self.rigids = [b for b in bodies if isinstance(b, Rigid) and not b.fixed]

        free = np.ones(nd, bool)
        for b in self.deformables:
            fv = np.where(b.fixed_verts)[0]
            for c in range(3):
                free[b.dof0 + 3 * fv + c] = False
        self.free = free

        self.faces = np.concatenate([b.faces + b.vert0 for b in bodies])
        self.edges = np.concatenate([b.edges + b.vert0 for b in bodies])
        self.vert_body = np.concatenate([np.full(b.n_verts, i) for i, b in enumerate(bodies)])
        self.face_body = np.concatenate([np.full(len(b.faces), i) for i, b in enumerate(bodies)])
        self.edge_body = np.concatenate([np.full(len(b.edges), i) for i, b in enumerate(bodies)])
        self.half_t = np.concatenate([b.half_thickness() for b in bodies])
        nb = len(bodies)
        static = np.array([isinstance(b, Rigid) and b.fixed for b in bodies])
        self.pair_ok = ~np.eye(nb, dtype=bool) & ~(static[:, None] & static[None, :])
        self.mu_pair = np.sqrt(np.outer([b.material.friction for b in bodies],
                                        [b.material.friction for b in bodies]))
        x = self.x()
        for g in self.gases:
            b = self.by_name[g.body]
            g.nRT = (P_ATM + g.p_gauge) * self._volume(x, b)
        self.lag = {}

    # --- state -------------------------------------------------------------
    def state(self):
        """A snapshot of every body's configuration."""
        return {b.name: (b.x.copy(),) if isinstance(b, Deformable) else (b.p.copy(), b.Q.copy())
                for b in self.bodies}

    def load(self, st):
        for b in self.bodies:
            if isinstance(b, Deformable):
                b.x = st[b.name][0].copy()
            else:
                b.p, b.Q = st[b.name][0].copy(), st[b.name][1].copy()

    def x(self, st=None):
        """All vertex positions (nv, 3) of a state (default: the current one)."""
        out = []
        for b in self.bodies:
            if isinstance(b, Deformable):
                out.append(st[b.name][0] if st else b.x)
            else:
                p, Q = st[b.name] if st else (b.p, b.Q)
                out.append(b.X @ Q.T + p)
        return np.concatenate(out)

    def body_verts(self, body, st=None):
        return self.x(st)[body.vert0:body.vert0 + body.n_verts]

    def retract(self, st, delta, alpha):
        new = {}
        for b in self.bodies:
            if isinstance(b, Deformable):
                new[b.name] = (st[b.name][0] + alpha * delta[b.dof0:b.dof0 + b.n_dofs].reshape(-1, 3),)
            elif b.fixed:
                new[b.name] = st[b.name]
            else:
                p, Q = st[b.name]
                d = delta[b.dof0:b.dof0 + 6]
                new[b.name] = (p + alpha * d[:3], rotation_exp(alpha * d[3:]) @ Q)
        return new

    def jacobian(self, st):
        """Sparse J with dx = J delta at state st."""
        rows, cols, vals = [], [], []
        for b in self.bodies:
            if isinstance(b, Deformable):
                n = 3 * b.n_verts
                rows.append(3 * b.vert0 + np.arange(n))
                cols.append(b.dof0 + np.arange(n))
                vals.append(np.ones(n))
            elif not b.fixed:
                p, Q = st[b.name]
                r = b.X @ Q.T                                  # world offsets from the centre of mass
                v = b.vert0 + np.arange(b.n_verts)
                for i in range(3):
                    rows.append(3 * v + i)
                    cols.append(np.full(b.n_verts, b.dof0 + i))
                    vals.append(np.ones(b.n_verts))
                # dx = w x r  ->  d x_i / d w_j = -[r]x_ij
                for i in range(3):
                    for j in range(3):
                        coef = -hat_col(r, i, j)
                        rows.append(3 * v + i)
                        cols.append(np.full(b.n_verts, b.dof0 + 3 + j))
                        vals.append(coef)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(3 * self.nv, self.nd))

    # --- geometry helpers ------------------------------------------------------
    def _volume(self, x, body):
        return float(kernels.call(kernels.vol_E, x[body.faces + body.vert0]).sum())

    def _pairs(self, x, extra=0.0):
        """Contact pairs within the barrier's range (+extra), with their offsets."""
        cand = collision.candidates(x, self.faces, self.edges, self.vert_body, self.face_body,
                                    self.edge_body, self.pair_ok,
                                    self.dhat + 2 * self.half_t.max() + extra)
        out = {}
        for kind in ("pt", "ee"):
            st = collision.stencil(kind, cand[kind], self.faces, self.edges)
            if len(st) == 0:
                out[kind] = (st, np.zeros(0))
                continue
            k = collision.split(kind)
            off = self.half_t[st[:, :k]].max(1) + self.half_t[st[:, k:]].max(1)
            d = collision.distance(kind, x[st])
            near = d - off < self.dhat + extra
            out[kind] = (st[near], off[near])
        return out

    def _update_lag(self, x, x_start):
        """Freeze normal force, closest-point weights and tangent basis (IPC lagging)."""
        self.lag = {}
        eps = self.eps_v * self.dt
        for kind, (st, off) in self._pairs(x).items():
            if len(st) == 0:
                continue
            g = np.clip(collision.distance(kind, x[st]) - off, 1e-300, self.dhat)
            lam = self.kappa * (2 * (g - self.dhat) * np.log(g / self.dhat) + (g - self.dhat) ** 2 / g)
            k = collision.split(kind)
            mu = self.mu_pair[self.vert_body[st[:, 0]], self.vert_body[st[:, k]]]
            w, n = collision.closest_weights(kind, x[st])
            keep = lam * mu > 0
            if keep.any():
                self.lag[kind] = (st[keep], x_start[st[keep]], w[keep], collision.tangent_basis(n[keep]),
                                  (mu * lam)[keep], np.full(keep.sum(), eps))

    # --- energy ----------------------------------------------------------------
    def energy(self, st, targets, derivs=True):
        """E at state st; with derivs, also the gradient and PSD Hessian w.r.t. increments."""
        h = self.dt
        x = self.x(st)
        E = 0.0
        g = np.zeros(self.nd) if derivs else None
        hq_r, hq_c, hq_v = [], [], []
        gx = np.zeros(3 * self.nv) if derivs else None
        hx_r, hx_c, hx_v = [], [], []

        def add_q(idx, G, H):
            g[idx] += G
            hq_r.append(np.repeat(idx, len(idx)))
            hq_c.append(np.tile(idx, len(idx)))
            hq_v.append(H.ravel())

        def add_x(verts, G, H):
            dofs = (3 * verts[:, :, None] + np.arange(3)).reshape(len(verts), -1)
            np.add.at(gx, dofs.ravel(), G.reshape(len(verts), -1).ravel())
            k = dofs.shape[1]
            hx_r.append(np.repeat(dofs, k, axis=1).ravel())
            hx_c.append(np.tile(dofs, (1, k)).ravel())
            hx_v.append(psd(H.reshape(len(verts), k, k)).ravel())

        def run(kern, args, verts):
            nonlocal E
            e = kernels.call(kern[0], *args)
            if not np.all(np.isfinite(e)):
                return False
            E += float(e.sum())
            if derivs:
                add_x(verts, kernels.call(kern[1], *args), kernels.call(kern[2], *args))
            return True

        # inertia and gravity
        for b in self.deformables:
            xb = st[b.name][0]
            m = b.nodal_mass[:, None]
            d = xb - targets[b.name]
            E += 0.5 * float(np.sum(m * d * d)) / h ** 2 - float(np.sum(m * xb @ self.gravity))
            if derivs:
                idx = b.dof0 + np.arange(b.n_dofs)
                g[idx] += (m * d / h ** 2 - m * self.gravity).ravel()
                hq_r.append(idx), hq_c.append(idx), hq_v.append(np.repeat(b.nodal_mass, 3) / h ** 2)
        for b in self.rigids:
            p, Q = st[b.name]
            p_t, Q_t = targets[b.name]
            D = Q - Q_t
            E += (0.5 * b.mass * np.sum((p - p_t) ** 2) + 0.5 * np.trace(D @ b.J @ D.T)) / h ** 2 \
                - b.mass * p @ self.gravity
            k_rot = 0.0
            for dr in self.drives:
                if dr.body == b.name:
                    k = np.asarray(dr.stiffness, float)
                    tgt = dr.target(self.t + h) if dr.target else np.zeros(3)
                    f = dr.force(self.t + h) if dr.force else np.zeros(3)
                    E += 0.5 * k @ (p - tgt) ** 2 - f @ p + 0.5 * dr.rot_stiffness * np.sum((Q - np.eye(3)) ** 2)
                    k_rot += dr.rot_stiffness
                    if derivs:
                        add_q(np.arange(b.dof0, b.dof0 + 3), k * (p - tgt) - f, np.diag(k))
            if derivs:
                ip = np.arange(b.dof0, b.dof0 + 3)
                add_q(ip, b.mass * (p - p_t) / h ** 2 - b.mass * self.gravity, np.eye(3) * b.mass / h ** 2)
                args = (np.zeros((1, 3)), Q[None], Q_t[None], b.J[None], np.array([h]), np.array([k_rot]))
                Gr = kernels.call(kernels.rot_G, *args)[0]
                Hr = psd(kernels.call(kernels.rot_H, *args))[0]
                add_q(np.arange(b.dof0 + 3, b.dof0 + 6), Gr, Hr)

        # elasticity
        for b in self.deformables:
            if isinstance(b, Shell):
                stn = b.stencil + b.vert0
                nF = len(b.faces)
                m = b.material
                run((kernels.shell_E, kernels.shell_G, kernels.shell_H),
                    (x[stn], b.has_nb, b.Ainv, b.abar, b.IIbar, b.area, b.thickness,
                     np.full(nF, m.E), np.full(nF, m.nu), b.eps_p), stn)
            elif isinstance(b, Solid):
                stn = b.tets + b.vert0
                mu, lam = b.material.lame
                nT = len(b.tets)
                if not run((kernels.tet_E, kernels.tet_G, kernels.tet_H),
                           (x[stn], b.Dm_inv, b.vol, np.full(nT, mu), np.full(nT, lam)), stn):
                    return np.inf, None, None

        # enclosed gas
        for gas in self.gases:
            b = self.by_name[gas.body]
            fv = b.faces + b.vert0
            V = float(kernels.call(kernels.vol_E, x[fv]).sum())
            if V <= 0:
                return np.inf, None, None
            E += -gas.nRT * np.log(V) + P_ATM * V
            if derivs:
                coef = -gas.nRT / V + P_ATM
                add_x(fv, coef * kernels.call(kernels.vol_G, x[fv]), coef * kernels.call(kernels.vol_H, x[fv]))

        # contact barrier and friction
        for kind, (stn, off) in self._pairs(x).items():
            if len(stn):
                n = len(stn)
                kern = (kernels.pt_E, kernels.pt_G, kernels.pt_H) if kind == "pt" else \
                       (kernels.ee_E, kernels.ee_G, kernels.ee_H)
                if not run(kern, (x[stn], off, np.full(n, self.dhat), np.full(n, self.kappa)), stn):
                    return np.inf, None, None
        for kind, (stn, xp, w, T, mulam, eps) in self.lag.items():
            run((kernels.fr_E, kernels.fr_G, kernels.fr_H), (x[stn], xp, w, T, mulam, eps), stn)

        if not derivs:
            return E, None, None
        J = self.jacobian(st)
        g += J.T @ gx
        H = sp.csr_matrix((np.concatenate(hq_v), (np.concatenate(hq_r), np.concatenate(hq_c))),
                          shape=(self.nd, self.nd)) if hq_r else sp.csr_matrix((self.nd, self.nd))
        if hx_r:
            Hx = sp.csr_matrix((np.concatenate(hx_v), (np.concatenate(hx_r), np.concatenate(hx_c))),
                               shape=(3 * self.nv, 3 * self.nv))
            H = H + J.T @ Hx @ J
        # curvature of the rotation map: sum_v g_v . d2x_v/dw2 (projected to PSD)
        for b in self.rigids:
            p, Q = st[b.name]
            r = b.X @ Q.T
            gv = gx.reshape(-1, 3)[b.vert0:b.vert0 + b.n_verts]
            G2 = 0.5 * (gv.T @ r + r.T @ gv) - np.sum(gv * r) * np.eye(3)
            G2 = psd(G2[None])[0]
            idx = np.arange(b.dof0 + 3, b.dof0 + 6)
            H = H + sp.csr_matrix((G2.ravel(), (np.repeat(idx, 3), np.tile(idx, 3))), shape=(self.nd, self.nd))
        return E, g, H.tocsr()

    # --- stepping --------------------------------------------------------------
    def _ccd(self, st, delta):
        """Largest safe fraction of the step, allowing for rigid vertices moving on arcs."""
        x0 = self.x(st)
        dx = self.x(self.retract(st, delta, 1.0)) - x0
        margin = np.zeros(self.nv)
        for b in self.rigids:
            w = np.linalg.norm(delta[b.dof0 + 3:b.dof0 + 6])
            margin[b.vert0:b.vert0 + b.n_verts] = np.linalg.norm(b.X, axis=1) * w ** 2 / 8
        reach = 2 * (np.linalg.norm(dx, axis=1).max() + margin.max())
        alpha = 1.0
        for kind, (stn, off) in self._pairs(x0, extra=reach).items():
            if len(stn):
                k = collision.split(kind)
                off2 = off + margin[stn[:, :k]].max(1) + margin[stn[:, k:]].max(1)
                alpha = min(alpha, float(collision.ccd(kind, x0[stn], dx[stn], off2).min()))
        return alpha

    def step(self):
        h = self.dt
        start = self.state()
        x_start = self.x(start)
        targets = {}
        for b in self.deformables:
            targets[b.name] = b.x + h * b.v
        for b in self.rigids:
            targets[b.name] = (b.p + h * b.vp, b.Q + h * b.Qdot)
        st = start
        iters = 0
        for _ in range(self.friction_iters):
            self._update_lag(self.x(st), x_start)
            for _ in range(self.max_newton):
                E, g, H = self.energy(st, targets)
                f = self.free
                delta = np.zeros(self.nd)
                delta[f] = solve_spd(H[f][:, f], -g[f])
                iters += 1
                move = np.abs(self.x(self.retract(st, delta, 1.0)) - self.x(st)).max()
                if move / h < self.newton_tol:
                    break
                alpha = self._ccd(st, delta)
                slope = g @ delta
                while alpha > 1e-12:
                    trial = self.retract(st, delta, alpha)
                    E_new, _, _ = self.energy(trial, targets, derivs=False)
                    if E_new <= E + 1e-4 * alpha * slope:
                        break
                    alpha *= 0.5
                st = self.retract(st, delta, alpha)
        # velocities, then commit
        for b in self.deformables:
            b.v = (st[b.name][0] - b.x) / h
        for b in self.rigids:
            p, Q = st[b.name]
            U, _, Vt = np.linalg.svd(Q)                     # guard against round-off drift
            Q = U @ Vt
            st[b.name] = (p, Q)
            b.vp = (p - b.p) / h
            b.Qdot = (Q - b.Q) / h
        self.load(st)
        self.t += h
        for d in self.drives:                             # force each actuator applies to its body
            b = self.by_name[d.body]
            k = np.asarray(d.stiffness, float)
            tgt = d.target(self.t) if d.target else np.zeros(3)
            f = d.force(self.t) if d.force else np.zeros(3)
            self.stats["drive_force"][d.body].append(-k * (b.p - tgt) + f)
        self._plasticity()
        self.stats["newton"].append(iters)
        return iters

    def _plasticity(self):
        """Staggered return map on every shell fiber (docs/09 §2)."""
        x = self.x()
        for b in self.deformables:
            if not isinstance(b, Shell) or not np.isfinite(b.material.sigma_y):
                continue
            stn = b.stencil + b.vert0
            Em, K = kernels.call(kernels.shell_strains_batched, x[stn], b.has_nb, b.Ainv, b.abar, b.IIbar)
            z = kernels.XI[None, :] * b.thickness[:, None] / 2
            eps = Em[:, None] + z[:, :, None, None] * K[:, None]
            m = b.material
            _, b.eps_p, b.alpha = plane_stress_return(m.E, m.nu, m.sigma_y, m.H, eps, b.eps_p, b.alpha)

    def run(self, steps, callback=None):
        for i in range(steps):
            self.step()
            if callback:
                callback(self, i)
        return self


def solve_spd(A, b):
    """Solve with a symmetric-positive-definite system: minimum-degree ordering on
    A + A^T and no pivoting, which cuts fill ~2x and time ~5x versus the default."""
    lu = splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options=dict(SymmetricMode=True))
    return lu.solve(b)


def hat_col(r, i, j):
    """Entry (i, j) of [r]x for each row of r (vectorised)."""
    if i == j:
        return np.zeros(len(r))
    k = 3 - i - j
    sign = 1.0 if (i, j) in ((2, 1), (0, 2), (1, 0)) else -1.0
    return sign * r[:, k]
