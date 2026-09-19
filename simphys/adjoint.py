"""Implicit derivatives of the 3D solver through time (docs/12).

Usage:
    tape = adjoint.record(scene)          # before stepping
    scene.run(n)
    grads = adjoint.gradient(scene, tape, dJ, params=[("jelly", "E")], initial_velocity=True)

`dJ` is the gradient of the objective with respect to the final state, in the
solver's increment coordinates (3 per deformable vertex, (dp, dw) per rigid
body); `vertex_objective` converts a vertex-space gradient. The forward run is
never changed: recording only stores what each step already computed.

Scope (docs/12 §7): no derivatives through plastic flow (refused), friction's
lagged data held fixed within each step (exact for frictionless contact).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.sparse.linalg import spsolve

from . import kernels
from .bodies import Shell
from .scene import hat


def record(scene):
    scene.recorder = []
    return scene.recorder


def vertex_objective(scene, gx, st=None):
    """Map a vertex-space gradient (nv, 3) at state st to increment coordinates."""
    st = st or scene.state()
    return scene.jacobian(st).T @ np.asarray(gx).ravel()


def _check_no_plastic_flow(scene, tape):
    shells = [b for b in scene.deformables if isinstance(b, Shell)]
    for n, rec in enumerate(tape):
        after = tape[n + 1]["alpha_start"] if n + 1 < len(tape) else [b.alpha for b in shells]
        for a0, a1 in zip(rec["alpha_start"], after):
            if np.any(a1 > a0):
                raise NotImplementedError(
                    f"plastic flow occurred in recorded step {n}; derivatives through the return map "
                    "are not implemented yet (docs/12 §7), so no gradient is returned")


def _dg_dparam(scene, rec, body, attr, rel):
    b = scene.by_name[body]
    base = b.material
    v = getattr(base, attr)
    d = rel * abs(v)
    out = []
    for sign in (1, -1):
        b.material = dataclasses.replace(base, **{attr: v + sign * d})
        out.append(scene.energy(rec["final"], rec["targets"], derivs=True, project=False)[1])
    b.material = base
    return (out[0] - out[1]) / (2 * d)


def _rot_tangent(M, Q):
    """Components k of sum(M * ([e_k] Q)): a matrix gradient mapped to rotation increments."""
    return np.array([np.sum(M * (hat(e) @ Q)) for e in np.eye(3)])


def gradient(scene, tape, dJ, params=(), initial_velocity=False, fd_rel=1e-6, residual_warn=1e-6):
    if not tape:
        raise ValueError("empty tape: call record(scene) before stepping")
    _check_no_plastic_flow(scene, tape)
    worst = max(r["residual"] for r in tape)
    if worst > residual_warn:
        warnings.warn(f"largest step residual {worst:.2e}: derivatives assume converged steps (docs/12 §2.2); "
                      "tighten scene.newton_tol", stacklevel=2)
    h, f = scene.dt, scene.free
    N = len(tape)
    adj = {N: np.asarray(dJ, float).copy()}
    out = {p: 0.0 for p in params}
    v0 = None
    t_saved, lag_saved = scene.t, scene.lag
    try:
        for n in reversed(range(N)):
            rec = tape[n]
            a = adj.pop(n + 1)
            adj.setdefault(n, np.zeros(scene.nd))
            if n >= 1:
                adj.setdefault(n - 1, np.zeros(scene.nd))
            scene.t, scene.lag = rec["t"], rec["lag"]
            _, _, H = scene.energy(rec["final"], rec["targets"], derivs=True, project=False)
            lam = np.zeros(scene.nd)
            lam[f] = spsolve(H[f][:, f].tocsc(), a[f])            # exact Hessian (docs/12 §2.1)
            for p in params:
                out[p] += -lam @ _dg_dparam(scene, rec, *p, fd_rel)
            # inertial targets: x~_n = 2 x_n - x_{n-1} (n >= 1), x~_0 = x_0 + h v_0
            mu = np.zeros(scene.nd)                               # dJ/dx~ in increment coordinates at x_n
            prev = np.zeros(scene.nd)                             # contribution to x_{n-1} (n >= 1)
            vel = np.zeros(scene.nd)                              # dJ/dv_0 (n == 0)
            for b in scene.deformables:
                sl = slice(b.dof0, b.dof0 + b.n_dofs)
                mu[sl] = np.repeat(b.nodal_mass, 3) / h ** 2 * lam[sl]
            for b in scene.rigids:
                i = b.dof0
                p_mu = b.mass / h ** 2 * lam[i:i + 3]
                Q = rec["final"][b.name][1]
                Qt = rec["targets"][b.name][1]
                k_rot = sum(d.rot_stiffness for d in scene.drives if d.body == b.name)
                dQt = -np.asarray(kernels.rot_mixed(Qt, lam[i + 3:i + 6], Q, b.J, h, k_rot))   # dJ/dQ~
                Qn = rec["start"][b.name][1]
                mu[i:i + 3] = p_mu
                if n >= 1:
                    Qp = tape[n - 1]["start"][b.name][1]
                    mu[i + 3:i + 6] = _rot_tangent(dQt, Qn)
                    prev[i + 3:i + 6] = -_rot_tangent(dQt, Qp)
                else:
                    mu[i + 3:i + 6] = _rot_tangent(dQt, Qn)
                    vel[i + 3:i + 6] = h * _rot_tangent(dQt, Qn)
                if n >= 1:
                    prev[i:i + 3] = -p_mu
                else:
                    vel[i:i + 3] = h * p_mu
            for b in scene.deformables:
                sl = slice(b.dof0, b.dof0 + b.n_dofs)
                if n >= 1:
                    prev[sl] = -mu[sl]
                else:
                    vel[sl] = h * mu[sl]
            if n >= 1:
                adj[n] += 2 * mu
                adj[n - 1] += prev
            else:
                adj[n] += mu
                v0 = vel
    finally:
        scene.t, scene.lag = t_saved, lag_saved
    out["x0"] = adj[0]
    if initial_velocity:
        out["v0"] = v0
    return out
