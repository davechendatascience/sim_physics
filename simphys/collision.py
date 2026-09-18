"""Collision geometry in NumPy: broad phase, closest points, and CCD.

CCD is Lipschitz-sampled. Along a linear step every point of a primitive
moves at most as far as its farthest vertex, so the gap is Lipschitz in the
step fraction with constant L = (max vertex motion on side A) + (side B).
Sampling at spacing 1/N, the gap between two samples g_i, g_j cannot drop
below (g_i + g_j - L/N) / 2. The step is cut at the first interval whose
lower bound falls under a fraction of the starting gap. Sliding contacts,
whose gap barely changes, pass the whole step; approaching ones are cut.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def seg_closest(p, a, b):
    ab = b - a
    s = np.clip(_dot(p - a, ab) / _dot(ab, ab), 0.0, 1.0)
    r = p - (a + s[..., None] * ab)
    return _dot(r, r), s


def pt_closest(p, t0, t1, t2):
    """Squared distance and barycentric weights (b0, b1, b2) of the closest point."""
    e1, e2, w = t1 - t0, t2 - t0, p - t0
    d11, d12, d22 = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2)
    det = d11 * d22 - d12 * d12
    b1 = (d22 * _dot(w, e1) - d12 * _dot(w, e2)) / det
    b2 = (d11 * _dot(w, e2) - d12 * _dot(w, e1)) / det
    inside = (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
    n = np.cross(e1, e2)
    plane = _dot(w, n) ** 2 / _dot(n, n)
    cands = [seg_closest(p, t0, t1), seg_closest(p, t1, t2), seg_closest(p, t2, t0)]
    d2 = np.stack([c[0] for c in cands], -1)
    k = np.argmin(d2, -1)
    s = np.stack([c[1] for c in cands], -1)[np.arange(len(k)), k] if d2.ndim > 1 else None
    bary = np.zeros(d2.shape[:-1] + (3,))
    idx = np.arange(len(k))
    for kk, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        sel = (k == kk) & ~inside
        bary[idx[sel], i] = 1 - s[sel]
        bary[idx[sel], j] = s[sel]
    bary[inside] = np.stack([1 - b1 - b2, b1, b2], -1)[inside]
    dist2 = np.where(inside, plane, d2.min(-1))
    return dist2, bary


def ee_closest(a0, a1, b0, b1):
    """Squared distance and segment parameters (s on a, u on b) of the closest points."""
    da, db, r = a1 - a0, b1 - b0, a0 - b0
    A, B, C, D, E = _dot(da, da), _dot(da, db), _dot(db, db), _dot(da, r), _dot(db, r)
    den = A * C - B * B
    parallel = den <= 1e-12 * A * C
    s = np.where(parallel, 0.0, np.clip((B * E - C * D) / np.where(parallel, 1.0, den), 0, 1))
    u = np.clip((B * s + E) / C, 0, 1)
    # the interior candidate plus the four endpoint-to-segment candidates cover
    # every clamped case exactly; take the closest
    d_ab0, u0 = seg_closest(a0, b0, b1)
    d_ab1, u1 = seg_closest(a1, b0, b1)
    d_ba0, s0 = seg_closest(b0, a0, a1)
    d_ba1, s1 = seg_closest(b1, a0, a1)
    diff = a0 + s[:, None] * da - (b0 + u[:, None] * db)
    d2 = np.stack([_dot(diff, diff), d_ab0, d_ab1, d_ba0, d_ba1], 1)
    S = np.stack([s, np.zeros_like(s), np.ones_like(s), s0, s1], 1)
    U = np.stack([u, u0, u1, np.zeros_like(u), np.ones_like(u)], 1)
    k = d2.argmin(1)
    i = np.arange(len(k))
    return d2[i, k], S[i, k], U[i, k]


def distance(kind, x4):
    """Unsigned distance for pairs; x4 is (m, 4, 3)."""
    if kind == "pt":
        return np.sqrt(pt_closest(x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3])[0])
    return np.sqrt(ee_closest(x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3])[0])


def candidates(x, faces, edges, vert_body, face_body, edge_body, pair_ok, radius):
    """Point-triangle and edge-edge pairs whose primitives may come within
    `radius`, between bodies allowed by pair_ok[body_a, body_b]."""
    out = {}
    # point-triangle: query the vertex tree once per face, each with its own radius
    tri = x[faces]
    cen = tri.mean(1)
    rad_f = np.linalg.norm(tri - cen[:, None], axis=2).max(1)
    hits = cKDTree(x).query_ball_point(cen, r=rad_f + radius)
    pf = np.repeat(np.arange(len(faces)), [len(h) for h in hits])
    pv = np.fromiter(itertools.chain.from_iterable(hits), dtype=int, count=len(pf))
    keep = pair_ok[vert_body[pv], face_body[pf]]
    out["pt"] = np.stack([pv[keep], pf[keep]], 1) if keep.any() else np.zeros((0, 2), int)

    # edge-edge: small edges by a midpoint tree; the few long edges (e.g. a
    # gripper pad's) by brute-force bounding-box overlap against all edges
    seg = x[edges]
    mid = seg.mean(1)
    half = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) / 2
    big = half > 4 * np.median(half)
    small = np.where(~big)[0]
    found = []
    if len(small) > 1:
        p = cKDTree(mid[small]).query_pairs(r=radius + 2 * half[small].max(), output_type="ndarray")
        if len(p):
            found.append(small[p])
    lo, hi = seg.min(1) - radius / 2, seg.max(1) + radius / 2
    for i in np.where(big)[0]:
        overlap = np.all((lo <= hi[i]) & (hi >= lo[i]), axis=1)
        overlap[i] = False
        j = np.where(overlap & (~big | (np.arange(len(edges)) > i)))[0]
        if len(j):
            found.append(np.stack([np.full(len(j), i), j], 1))
    pairs = np.concatenate(found) if found else np.zeros((0, 2), int)
    keep = pair_ok[edge_body[pairs[:, 0]], edge_body[pairs[:, 1]]] if len(pairs) else np.zeros(0, bool)
    out["ee"] = pairs[keep] if keep.any() else np.zeros((0, 2), int)
    return out


def stencil(kind, pairs, faces, edges):
    """Global vertex indices (m, 4) of each pair's stencil."""
    if kind == "pt":
        return np.concatenate([pairs[:, :1], faces[pairs[:, 1]]], 1)
    return np.concatenate([edges[pairs[:, 0]], edges[pairs[:, 1]]], 1)


def split(kind):
    """How many of the 4 stencil vertices belong to side A."""
    return 1 if kind == "pt" else 2


def ccd(kind, x4, dx4, offset, keep_fraction=0.1, n_max=65536, chunk_points=2_000_000):
    """Largest safe step fraction for each pair (see module docstring)."""
    m = len(x4)
    if m == 0:
        return np.ones(0)
    k = split(kind)
    g0 = distance(kind, x4) - offset
    motion = np.linalg.norm(dx4, axis=2)
    L = motion[:, :k].max(1) + motion[:, k:].max(1)
    floor = keep_fraction * g0
    alpha = np.ones(m)
    moving = L > 0
    need = np.where(moving, np.ceil(2 * L / np.maximum(g0 - floor, 1e-300)), 1)
    N_all = np.minimum(2 ** np.ceil(np.log2(np.maximum(need, 1))), n_max).astype(int)
    for N in np.unique(N_all[moving]):
        group = np.where(moving & (N_all == N))[0]
        t = np.linspace(0, 1, N + 1)
        per = max(1, chunk_points // (N + 1))                   # pairs per chunk: bounded memory
        for c0 in range(0, len(group), per):
            sel = group[c0:c0 + per]
            xs = x4[sel][:, None] + t[None, :, None, None] * dx4[sel][:, None]
            g = distance(kind, xs.reshape(-1, 4, 3)).reshape(len(sel), N + 1) - offset[sel][:, None]
            lower = (g[:, :-1] + g[:, 1:] - L[sel][:, None] / N) / 2
            bad = lower <= floor[sel][:, None]
            first = np.where(bad.any(1), bad.argmax(1), N)      # first failing interval
            a_s = t[first]                                      # start of that interval (safe)
            a_c = (1 - keep_fraction) * g0[sel] / L[sel]        # one-shot conservative bound
            alpha[sel] = np.clip(np.maximum(a_s, a_c), 0, 1)
    return alpha


def tangent_basis(n):
    """Orthonormal (m, 3, 2) bases perpendicular to unit normals n (m, 3)."""
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2], axis=2)


def closest_weights(kind, x4):
    """Weights w (m, 4) so that sum_i w_i x_i = (closest point on A) - (closest on B),
    and the unit normal from B to A."""
    if kind == "pt":
        _, bary = pt_closest(x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3])
        w = np.concatenate([np.ones((len(x4), 1)), -bary], 1)
    else:
        _, s, u = ee_closest(x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3])
        w = np.stack([1 - s, s, -(1 - u), -u], 1)
    r = np.einsum("mi,mij->mj", w, x4)
    n = r / np.linalg.norm(r, axis=1, keepdims=True)
    return w, n
