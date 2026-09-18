"""Small scenes shared by the simulator tests."""

from __future__ import annotations

import numpy as np

from simphys import geometry as G
from simphys.bodies import Rigid, Shell
from simphys.materials import ALUMINUM_CAN, GROUND, WOOD
from simphys.scene import Scene


def ground(size=0.6):
    v, f = G.box((size, size, 0.05), center=(0, 0, -0.025))
    return Rigid("ground", v, f, GROUND, fixed=True)


def box(name, size=(0.1, 0.1, 0.1), center=(0, 0, 0), material=WOOD):
    v, f = G.box(size, center=center)
    return Rigid(name, v, f, material)


def total_energy(scene):
    """Kinetic + gravitational potential + barrier energy (rigid-only scenes)."""
    from simphys import collision, kernels
    E = 0.0
    for b in scene.rigids:
        E += 0.5 * b.mass * b.vp @ b.vp + 0.5 * np.trace(b.Qdot @ b.J @ b.Qdot.T) - b.mass * b.p @ scene.gravity
    x = scene.x()
    for kind, (st, off) in scene._pairs(x).items():
        if len(st):
            g = collision.distance(kind, x[st]) - off
            g = np.clip(g, 1e-300, scene.dhat)
            E += float(np.sum(-scene.kappa * (g - scene.dhat) ** 2 * np.log(g / scene.dhat)))
    return E


def min_gap(scene):
    from simphys import collision
    x = scene.x()
    gaps = [collision.distance(k, x[st]) - off for k, (st, off) in scene._pairs(x).items() if len(st)]
    return float(min(g.min() for g in gaps)) if gaps else np.inf
