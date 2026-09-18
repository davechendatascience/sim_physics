"""Bodies: rigid, thin shell, and solid (docs/02 §1, docs/09 §2).

Each body holds its own state. Deformables store vertex positions; rigid
bodies store a position p and a rotation Q, with vertices x = Q X + p. The
solver works on increments: 3 per free deformable vertex, (dp, w) per rigid
body.
"""

from __future__ import annotations

import numpy as np

from . import geometry, kernels
from .materials import Material


class Body:
    kind = "body"

    def __init__(self, name: str, verts, faces, material: Material, fixed=False, color=None):
        self.name = name
        self.faces = np.asarray(faces, dtype=int)          # surface triangles (contact + render)
        self.material = material
        self.fixed = fixed
        self.color = color
        self.rest = np.asarray(verts, float)
        self.edges = geometry.edges_of(self.faces)

    # set by the scene
    dof0 = 0
    vert0 = 0

    @property
    def n_verts(self):
        return len(self.rest)

    def half_thickness(self):
        return np.zeros(self.n_verts)


class Rigid(Body):
    """Rigid body with an exact rotation: x = Q X + p (docs/02 §1).

    Newton works on increments (dp, w) and applies Q <- exp([w]) Q, so Q is a
    rotation at every iterate. Rotational inertia uses J = int rho X X^T dV.
    """
    kind = "rigid"

    def __init__(self, name, verts, faces, material, fixed=False, color=None):
        super().__init__(name, verts, faces, material, fixed, color)
        self.mass, com, self.J, self.volume = geometry.mass_properties(self.rest, self.faces, material.rho)
        self.X = self.rest - com                            # body-frame vertices
        self.com0 = com
        self.n_dofs = 0 if fixed else 6
        self.reset()

    def reset(self):
        self.p = self.com0.copy()
        self.Q = np.eye(3)
        self.vp = np.zeros(3)
        self.Qdot = np.zeros((3, 3))

    def verts(self, p=None, Q=None):
        p = self.p if p is None else p
        Q = self.Q if Q is None else Q
        return self.X @ Q.T + p

    @property
    def omega(self):
        """World angular velocity from Qdot = [w] Q."""
        W = self.Qdot @ self.Q.T
        return np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) / 2


class Deformable(Body):
    """Bodies whose DOFs are their vertex positions."""

    def __init__(self, name, verts, faces, material, fixed_verts=None, color=None):
        super().__init__(name, verts, faces, material, False, color)
        self.fixed_verts = np.zeros(self.n_verts, bool) if fixed_verts is None else np.asarray(fixed_verts, bool)
        self.n_dofs = 3 * self.n_verts
        self.reset()

    def reset(self):
        self.x = self.rest.copy()
        self.v = np.zeros_like(self.rest)

    def verts(self):
        return self.x


class Shell(Deformable):
    """Discrete Kirchhoff-Love shell with plane-stress J2 through the thickness."""
    kind = "shell"

    def __init__(self, name, verts, faces, material, thickness, fixed_verts=None, color=None):
        super().__init__(name, verts, faces, material, fixed_verts, color)
        nF = len(self.faces)
        self.thickness = np.broadcast_to(np.asarray(thickness, float), (nF,)).copy()
        opp = geometry.opposite_vertices(self.faces)
        self.has_nb = (opp >= 0).astype(float)
        # boundary placeholder: the face's own vertex k (weighted out in the kernel)
        self.stencil = np.concatenate([self.faces, np.where(opp >= 0, opp, self.faces)], axis=1)
        x = self.rest[self.faces]
        e1, e2 = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
        self.area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        # rest orthonormal frame per face: Abar = [[e1.t1, e2.t1], [0, e2.t2]]
        t1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        e2p = e2 - np.einsum("ij,ij->i", e2, t1)[:, None] * t1
        t2 = e2p / np.linalg.norm(e2p, axis=1, keepdims=True)
        Abar = np.zeros((nF, 2, 2))
        Abar[:, 0, 0] = np.einsum("ij,ij->i", e1, t1)
        Abar[:, 0, 1] = np.einsum("ij,ij->i", e2, t1)
        Abar[:, 1, 1] = np.einsum("ij,ij->i", e2, t2)
        self.Ainv = np.linalg.inv(Abar)
        self.frame = np.stack([t1, t2], axis=2)             # (nF, 3, 2) rest in-plane axes
        self.abar, self.IIbar = kernels.call(kernels.shell_forms_batched, self.rest[self.stencil], self.has_nb)
        self.eps_p = np.zeros((nF, 9, 2, 2))
        self.alpha = np.zeros((nF, 9))
        nodal = np.zeros(self.n_verts)
        np.add.at(nodal, self.faces.ravel(), np.repeat(material.rho * self.thickness * self.area / 3, 3))
        self.nodal_mass = nodal

    def half_thickness(self):
        h = np.zeros(self.n_verts)
        np.maximum.at(h, self.faces.ravel(), np.repeat(self.thickness / 2, 3))
        return h

    def face_plastic_strain(self):
        """Largest equivalent plastic strain through the thickness, per face."""
        return self.alpha.max(axis=1)


class Solid(Deformable):
    """Linear tetrahedra with Neo-Hookean energy."""
    kind = "solid"

    def __init__(self, name, verts, tets, material, fixed_verts=None, color=None):
        tets = np.asarray(tets, dtype=int)
        super().__init__(name, verts, geometry.tet_surface(tets), material, fixed_verts, color)
        self.tets = tets
        x = self.rest[tets]
        Dm = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
        self.Dm_inv = np.linalg.inv(Dm)
        self.vol = np.linalg.det(Dm) / 6
        nodal = np.zeros(self.n_verts)
        np.add.at(nodal, tets.ravel(), np.repeat(material.rho * self.vol / 4, 4))
        self.nodal_mass = nodal
