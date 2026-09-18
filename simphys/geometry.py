"""Mesh generators and mesh utilities. All meshes are outward-oriented triangle
surfaces (counter-clockwise seen from outside) in metres."""

from __future__ import annotations

import numpy as np


def box(size, center=(0.0, 0.0, 0.0), n=1):
    """Closed box surface subdivided n times per edge."""
    sx, sy, sz = np.asarray(size, float) / 2
    verts, faces, index = [], [], {}

    def vid(p):
        key = tuple(np.round(p, 12))
        if key not in index:
            index[key] = len(verts)
            verts.append(p)
        return index[key]

    # each face: origin corner, two in-plane axes (u, v) with u x v = outward normal
    for axis in range(3):
        for sign in (-1, 1):
            u_ax, v_ax = [(1, 2), (2, 0), (0, 1)][axis]
            if sign < 0:
                u_ax, v_ax = v_ax, u_ax
            half = np.array([sx, sy, sz])
            origin = np.zeros(3)
            origin[axis] = sign * half[axis]
            origin[u_ax], origin[v_ax] = -half[u_ax], -half[v_ax]
            du, dv = np.zeros(3), np.zeros(3)
            du[u_ax], dv[v_ax] = 2 * half[u_ax] / n, 2 * half[v_ax] / n
            for i in range(n):
                for j in range(n):
                    a = vid(origin + i * du + j * dv)
                    b = vid(origin + (i + 1) * du + j * dv)
                    c = vid(origin + (i + 1) * du + (j + 1) * dv)
                    d = vid(origin + i * du + (j + 1) * dv)
                    faces += [[a, b, c], [a, c, d]]
    return np.array(verts) + np.asarray(center, float), np.array(faces)


def cylinder(radius, height, n_theta, n_z, capped=True, center=(0.0, 0.0, 0.0)):
    """Cylindrical shell along z. Returns verts, faces, and a face label array:
    0 = side wall, 1 = bottom cap, 2 = top cap. Caps are triangle fans."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(-height / 2, height / 2, n_z + 1)
    verts = [[radius * np.cos(t), radius * np.sin(t), z] for z in zs for t in th]
    faces, label = [], []
    for k in range(n_z):
        for j in range(n_theta):
            a, b = k * n_theta + j, k * n_theta + (j + 1) % n_theta
            c, d = a + n_theta, b + n_theta
            # alternate the diagonal so the mesh has no preferred shear direction
            if (j + k) % 2 == 0:
                faces += [[a, b, d], [a, d, c]]
            else:
                faces += [[a, b, c], [b, d, c]]
            label += [0, 0]
    if capped:
        bot = len(verts)
        verts.append([0, 0, -height / 2])
        top = len(verts)
        verts.append([0, 0, height / 2])
        last = n_z * n_theta
        for j in range(n_theta):
            faces.append([bot, (j + 1) % n_theta, j])
            label.append(1)
            faces.append([top, last + j, last + (j + 1) % n_theta])
            label.append(2)
    return np.array(verts, float) + np.asarray(center, float), np.array(faces), np.array(label)


def tet_box(size, n, center=(0.0, 0.0, 0.0)):
    """Box of n^3 cubes, 6 tets each (Kuhn split: conforming across cubes)."""
    s = np.asarray(size, float)
    g = np.stack(np.meshgrid(*[np.linspace(-0.5, 0.5, n + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    verts = g * s + np.asarray(center, float)
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
    cube = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    kuhn = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a, b, d in cube]
                tets += [[c[p] for p in t] for t in kuhn]
    tets = np.array(tets)
    # orient every tet positively
    x = verts[tets]
    vol = np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    return verts, tets


def tet_surface(tets):
    """Boundary triangles of a tet mesh, outward-oriented."""
    count = {}
    for t in tets:
        for f in ((t[0], t[2], t[1]), (t[0], t[1], t[3]), (t[0], t[3], t[2]), (t[1], t[2], t[3])):
            key = tuple(sorted(f))
            count.setdefault(key, []).append(f)
    return np.array([fs[0] for fs in count.values() if len(fs) == 1])


def edges_of(faces):
    """Unique undirected edges of a triangle mesh."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def opposite_vertices(faces):
    """For each face and each local vertex k, the vertex opposite edge k in the
    neighbouring face (-1 on a boundary edge). Edge k joins local vertices k+1, k+2."""
    owner = {}
    for f, (a, b, c) in enumerate(faces):
        for k, (u, v) in enumerate(((b, c), (c, a), (a, b))):
            owner[(u, v)] = (f, k)
    opp = -np.ones((len(faces), 3), dtype=int)
    for f, (a, b, c) in enumerate(faces):
        for k, (u, v) in enumerate(((b, c), (c, a), (a, b))):
            nb = owner.get((v, u))
            if nb is not None:
                g, kk = nb
                opp[f, k] = faces[g][kk]
    return opp


def mass_properties(verts, faces, rho):
    """Mass, centre of mass and second moment J = int rho X X^T dV (about the
    centre of mass) of a closed, outward-oriented surface."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))           # 6 x signed tet volume
    V = vol6.sum() / 6
    com = (vol6[:, None] * (a + b + c)).sum(0) / (24 * V)
    # second moment of each origin-tet: vol/20 * (sum_i x_i x_i^T + (sum x)(sum x)^T)
    s = a + b + c
    J = np.einsum("i,ij,ik->jk", vol6 / 120, s, s) + sum(
        np.einsum("i,ij,ik->jk", vol6 / 120, p, p) for p in (a, b, c))
    J = rho * J - rho * V * np.outer(com, com)
    return rho * V, com, J, V
