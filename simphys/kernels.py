"""Per-stencil energies in JAX. Each energy is written once; gradients and
Hessians come from JAX, batched with vmap and compiled with jit.

Stencils are small fixed-size vertex sets (a shell face and its three
neighbours, a tet, a contact pair). The solver assembles their gradients and
PSD-projected Hessians into the global system.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
# persistent compilation cache: compile once per machine, not once per run (docs/10 §2)
from pathlib import Path as _Path
jax.config.update("jax_compilation_cache_dir", str(_Path(__file__).resolve().parents[1] / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

# composite Simpson through the thickness, split at the mid-surface (docs/09 §2)
XI = np.linspace(-1.0, 1.0, 9)
W_XI = np.array([1, 4, 2, 4, 2, 4, 2, 4, 1]) * (0.25 / 3)       # sums to 2


def _normalize(v):
    return v / jnp.sqrt(jnp.sum(v * v))


# --- thin shell: discrete Kirchhoff-Love with mid-edge normals --------------

def shell_forms(x6, has_nb):
    """First and second fundamental forms (covariant, in the face's edge basis)."""
    x0, x1, x2 = x6[0], x6[1], x6[2]
    e1, e2 = x1 - x0, x2 - x0
    a = jnp.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
    n = _normalize(jnp.cross(e1, e2))
    verts = (x0, x1, x2)
    ms = []
    for k in range(3):
        va, vb, o = verts[(k + 1) % 3], verts[(k + 2) % 3], x6[3 + k]
        nn = jnp.cross(va - vb, o - vb)                   # neighbour face (vb, va, o)
        nn = nn / jnp.sqrt(jnp.sum(nn * nn) + 1e-300)
        # blend rather than jnp.where: the unselected branch would still be
        # differentiated, and a NaN there poisons the gradient. On a boundary
        # edge the placeholder neighbour is weighted out, leaving m = n.
        ms.append(_normalize(n + has_nb[k] * nn))
    d1, d2 = ms[1] - ms[0], ms[2] - ms[0]
    II = 2.0 * jnp.array([[d1 @ e1, d1 @ e2], [d2 @ e1, d2 @ e2]])
    return a, 0.5 * (II + II.T)


def shell_strains(x6, has_nb, Ainv, abar, IIbar):
    """Membrane strain and curvature change as Cartesian tensors in the rest frame."""
    a, II = shell_forms(x6, has_nb)
    Em = 0.5 * Ainv.T @ (a - abar) @ Ainv
    K = Ainv.T @ (II - IIbar) @ Ainv
    return Em, K


def shell_energy(x6, has_nb, Ainv, abar, IIbar, area, t, E, nu, eps_p):
    Em, K = shell_strains(x6, has_nb, Ainv, abar, IIbar)
    z = jnp.asarray(XI) * t / 2
    w = jnp.asarray(W_XI) * t / 2
    ee = Em[None] + z[:, None, None] * K[None] - eps_p            # (9, 2, 2) elastic strain
    tr = ee[:, 0, 0] + ee[:, 1, 1]
    psi = 0.5 * E / (1 - nu ** 2) * ((1 - nu) * jnp.sum(ee * ee, axis=(1, 2)) + nu * tr * tr)
    return area * jnp.sum(w * psi)


# --- solid: Neo-Hookean linear tetrahedron ----------------------------------

def tet_energy(x4, Dm_inv, vol, mu, lam):
    Ds = jnp.stack([x4[1] - x4[0], x4[2] - x4[0], x4[3] - x4[0]], axis=1)
    F = Ds @ Dm_inv
    J = jnp.linalg.det(F)
    safe = jnp.where(J > 0, J, 1.0)
    lnJ = jnp.log(safe)
    psi = 0.5 * mu * (jnp.sum(F * F) - 3) - mu * lnJ + 0.5 * lam * lnJ ** 2
    return jnp.where(J > 0, vol * psi, jnp.inf)


# --- contact ----------------------------------------------------------------

def _seg_dist2(p, a, b):
    ab = b - a
    s = jnp.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
    r = p - (a + s * ab)
    return r @ r


def pt_dist2(x4):
    """Squared point-triangle distance (min over interior projection and edges)."""
    p, t0, t1, t2 = x4[0], x4[1], x4[2], x4[3]
    e1, e2 = t1 - t0, t2 - t0
    n = jnp.cross(e1, e2)
    nn = n @ n
    # barycentric of the projection
    w = p - t0
    d11, d12, d22 = e1 @ e1, e1 @ e2, e2 @ e2
    det = d11 * d22 - d12 * d12
    b1 = (d22 * (w @ e1) - d12 * (w @ e2)) / det
    b2 = (d11 * (w @ e2) - d12 * (w @ e1)) / det
    inside = (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
    plane = (w @ n) ** 2 / nn
    edge = jnp.minimum(jnp.minimum(_seg_dist2(p, t0, t1), _seg_dist2(p, t1, t2)), _seg_dist2(p, t2, t0))
    return jnp.where(inside, plane, edge)


def ee_dist2(x4):
    """Squared distance between segments a0-a1 and b0-b1."""
    a0, a1, b0, b1 = x4[0], x4[1], x4[2], x4[3]
    da, db, r = a1 - a0, b1 - b0, a0 - b0
    A, B, C, D, E = da @ da, da @ db, db @ db, da @ r, db @ r
    den = A * C - B * B
    parallel = den <= 1e-12 * A * C
    s = jnp.where(parallel, 0.0, jnp.clip((B * E - C * D) / jnp.where(parallel, 1.0, den), 0.0, 1.0))
    u = (B * s + E) / C
    s = jnp.where((u < 0) | (u > 1), jnp.clip((B * jnp.clip(u, 0.0, 1.0) - D) / A, 0.0, 1.0), s)
    u = jnp.clip(u, 0.0, 1.0)
    # endpoint candidates cover the clamped cases exactly
    d_main = jnp.sum((a0 + s * da - (b0 + u * db)) ** 2)
    ends = jnp.minimum(jnp.minimum(_seg_dist2(a0, b0, b1), _seg_dist2(a1, b0, b1)),
                       jnp.minimum(_seg_dist2(b0, a0, a1), _seg_dist2(b1, a0, a1)))
    return jnp.minimum(d_main, ends)


def barrier(d, dhat, kappa):
    """IPC log barrier on the gap d (docs/02 §2); +inf at or below contact."""
    safe = jnp.clip(d, 1e-300, dhat)
    b = -kappa * (safe - dhat) ** 2 * jnp.log(safe / dhat)
    return jnp.where(d <= 0, jnp.inf, jnp.where(d < dhat, b, 0.0))


def pt_barrier(x4, offset, dhat, kappa):
    return barrier(jnp.sqrt(pt_dist2(x4)) - offset, dhat, kappa)


def ee_mollifier(x4, xr4):
    """C1 weight that fades the edge-edge barrier to zero as the edges become
    parallel (IPC; docs/02 §2). xr4 are rest positions of the same vertices."""
    c = jnp.sum(jnp.cross(x4[1] - x4[0], x4[3] - x4[2]) ** 2)
    e = 1e-3 * jnp.sum((xr4[1] - xr4[0]) ** 2) * jnp.sum((xr4[3] - xr4[2]) ** 2)
    r = c / e
    return jnp.where(r < 1.0, (2.0 - r) * r, 1.0)


def ee_barrier(x4, xr4, offset, dhat, kappa):
    return ee_mollifier(x4, xr4) * barrier(jnp.sqrt(ee_dist2(x4)) - offset, dhat, kappa)


def f0(y, eps):
    return jnp.where(y >= eps, y, -y ** 3 / (3 * eps ** 2) + y ** 2 / eps + eps / 3)


def friction_energy(x4, x4_prev, weights, T, mulam, eps):
    """Lagged smoothed Coulomb friction: mu*lambda*f0(|u|), with u the tangential
    relative displacement of the two closest points over the step (docs/02 §2)."""
    rel = jnp.sum(weights[:, None] * (x4 - x4_prev), axis=0)
    u = T.T @ rel
    y = jnp.sqrt(u @ u + 1e-30)
    return mulam * f0(y, eps)


# --- enclosed gas: per-face volume contribution --------------------------------

def face_volume(x3):
    return x3[0] @ jnp.cross(x3[1], x3[2]) / 6.0


# --- rigid body rotation (derivatives taken at w = 0) -------------------------

def _hat(w):
    return jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rigid_rotational(w, Q, Qt, J, h, k_rot):
    """Rotational inertia + orientation spring as a function of the rotation
    increment w, with Q(w) = exp([w]) Q. Only derivatives at w = 0 are used, so
    exp is expanded to second order (exact for the gradient and Hessian there)."""
    K = _hat(w)
    Qw = (jnp.eye(3) + K + 0.5 * K @ K) @ Q
    D = Qw - Qt
    return 0.5 * jnp.trace(D @ J @ D.T) / h ** 2 + 0.5 * k_rot * jnp.sum((Qw - jnp.eye(3)) ** 2)


def _psd(H):
    w, V = jnp.linalg.eigh(0.5 * (H + H.T))
    return (V * jnp.maximum(w, 0.0)) @ V.T


def _batched(f):
    """(energy, gradient, Hessian, fused) batched kernels. The fused kernel
    returns energy, gradient and PSD-projected Hessian in one compiled call."""
    def fused(*a):
        n = a[0].size
        return f(*a), jax.grad(f)(*a), _psd(jax.hessian(f)(*a).reshape(n, n))
    return (jax.jit(jax.vmap(f)), jax.jit(jax.vmap(jax.grad(f))), jax.jit(jax.vmap(jax.hessian(f))),
            jax.jit(jax.vmap(fused)))


shell_E, shell_G, shell_H, shell_F = _batched(shell_energy)
tet_E, tet_G, tet_H, tet_F = _batched(tet_energy)
pt_E, pt_G, pt_H, pt_F = _batched(pt_barrier)
ee_E, ee_G, ee_H, ee_F = _batched(ee_barrier)
fr_E, fr_G, fr_H, fr_F = _batched(friction_energy)
vol_E, vol_G, vol_H, vol_F = _batched(face_volume)
rot_E, rot_G, rot_H, rot_F = _batched(rigid_rotational)
def _rot_mixed(Qt, lam, Q, J, h, k_rot):
    """d/dQ~ of lam . (d/dw rigid_rotational at w = 0): the adjoint's link from a
    rigid body's inertial target back to its previous rotations (docs/12 §7)."""
    g = jax.grad(rigid_rotational)(jnp.zeros(3), Q, Qt, J, h, k_rot)
    return lam @ g


rot_mixed = jax.jit(jax.grad(_rot_mixed))
shell_forms_batched = jax.jit(jax.vmap(shell_forms))
shell_strains_batched = jax.jit(jax.vmap(shell_strains))


def call(fn, *args):
    """Call a batched kernel, padding the batch to a power of two so jit
    compiles once per size bucket rather than once per distinct batch size."""
    n = len(args[0])
    if n == 0:
        return None
    m = max(16, 1 << (n - 1).bit_length())
    padded = [np.concatenate([a, np.repeat(a[:1], m - n, axis=0)]) if m > n else a
              for a in (np.asarray(x) for x in args)]
    out = fn(*padded)
    if isinstance(out, tuple):
        return tuple(np.asarray(o)[:n] for o in out)
    return np.asarray(out)[:n]
