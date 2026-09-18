"""Built-in scenes. Each builder returns (scene, info) where info carries what
the scene measures and how long to run it."""

from __future__ import annotations

import numpy as np

from . import geometry as G
from .bodies import Rigid, Shell, Solid
from .materials import ALUMINUM_CAN, GELATIN, GROUND, SILICONE, STEEL, WOOD, Material
from .scene import Drive, Gas, Scene

CAN_R, CAN_H, CAN_T = 0.033, 0.122, 1.0e-4
PAD = Material("gripper pad (rigid)", rho=1200.0, E=3e9, nu=0.35, friction=0.8)


def box_drop(n_boxes=3, seed=0):
    """Wooden boxes dropped with a spin onto the ground: rigid contact and friction."""
    rng = np.random.default_rng(seed)
    gv, gf = G.box((0.4, 0.4, 0.05), center=(0, 0, -0.025))
    bodies = [Rigid("ground", gv, gf, GROUND, fixed=True, color="#8a8a8a")]
    for i in range(n_boxes):
        size = rng.uniform(0.06, 0.12, 3)
        v, f = G.box(size)
        ang = rng.uniform(-0.6, 0.6, 3)
        cx, cy, cz = np.cos(ang)
        sx, sy, sz = np.sin(ang)
        R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
             @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
        pos = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.12 + 0.15 * i])
        bodies.append(Rigid(f"box{i}", v @ R.T + pos, f, WOOD, color=["#c0392b", "#2980b9", "#27ae60", "#8e44ad"][i % 4]))
    scene = Scene(bodies, dt=1 / 100, dhat=1e-3, kappa=1e4, eps_v=1e-3)
    return scene, {"steps": 120, "view": (20, -60)}


def _pads(length, gap, width, thickness=0.01):
    """Two flat rigid pads facing the can along x, just outside its surface."""
    out = []
    for side, name in ((-1, "pad_left"), (1, "pad_right")):
        v, f = G.box((thickness, width, length))
        center = np.array([side * (CAN_R + CAN_T / 2 + gap + thickness / 2), 0.0, 0.0])
        out.append(Rigid(name, v + center, f, PAD, color="#34495e"))
    return out


def can_mesh(graded_mesh=True, n_theta=48, n_z=24):
    """Can surface. Graded: 7.5 deg / 5 mm within +-30 deg of the pads (+-x) and
    +-25 mm of mid-height, coarsening to ~22.5 deg / 15 mm elsewhere. Measured
    against the uniform mesh it is 3x smaller but its pad forces differ by
    10-30%, so it is not the default (docs/10 §1)."""
    if not graded_mesh:
        return G.cylinder(CAN_R, CAN_H, n_theta, n_z, capped=True)
    d = np.radians
    th = G.graded([d(-30), d(30), d(150), d(210), d(330)], [d(7.5), d(22.5), d(7.5), d(22.5)])[:-1] % (2 * np.pi)
    zz = G.graded([-CAN_H / 2, -0.025, 0.025, CAN_H / 2], [0.015, 0.005, 0.015])
    return G.cylinder(CAN_R, CAN_H, 0, 0, capped=True, theta=np.sort(th), z=zz)


def can_squeeze(depth=4e-3, speed=0.016, hold=0.1, pad_width=0.015, pad_length=0.02, sealed=False,
                p_gauge=2.5e5, n_theta=48, n_z=24, dt=5e-3, graded_mesh=False):
    """A soda can squeezed by two short pads (docs/09 §5).

    The can stands on a table (its bottom end is held fixed). Each pad follows a
    prescribed path at `speed` to `depth` into the wall, holds, and retracts;
    a stiff spring ties it to the path, and the spring force is the measured
    reaction. Open or sealed at p_gauge. After retraction, the ledger says
    whether the can was permanently altered.
    """
    v, f, label = can_mesh(graded_mesh, n_theta, n_z)
    thickness = np.where(label == 0, CAN_T, 2.5e-4)          # ends are thicker (~0.25 mm)
    base = np.isclose(v[:, 2], -CAN_H / 2)
    can = Shell("can", v, f, ALUMINUM_CAN, thickness, fixed_verts=base, color="#bdc3c7")
    pads = _pads(pad_length, gap=5e-6, width=pad_width)
    t_in = depth / speed
    total = 2 * t_in + hold

    def travel(t):
        if t < t_in:
            return speed * t
        if t < t_in + hold:
            return depth
        return max(0.0, depth - speed * (t - t_in - hold))

    drives = []
    for p, side in zip(pads, (-1, 1)):
        c = p.com0.copy()
        drives.append(Drive(p.name, stiffness=(1e6, 1e4, 1e4),
                            target=(lambda t, c=c, side=side: c - side * np.array([travel(t), 0.0, 0.0])),
                            rot_stiffness=1e3))
    gases = [Gas("can", p_gauge)] if sealed else []
    # 1 um per step is ample resolution for a dent verdict; the strict default is
    # kept for the validation scenes, where it measurably matters (docs/09 §4)
    scene = Scene([can] + pads, dt=dt, dhat=1e-5, kappa=1e3, eps_v=1e-4,
                  drives=drives, gases=gases, newton_tol=2e-4)
    return scene, {"steps": int(round(total / dt)), "travel": travel, "view": (15, -70)}


def can_squeeze_long(force_per_length=1.0, n_theta=48, n_z=12, length=0.12, steps=80, dt=5e-3):
    """Validation: an open can squeezed along its whole length by long pads is a
    ring in plane strain; its diametral compliance must match docs/09 §4."""
    v, f, _ = G.cylinder(CAN_R, length, n_theta, n_z, capped=False)
    can = Shell("can", v, f, ALUMINUM_CAN, CAN_T, color="#bdc3c7")
    pads = _pads(length * 1.2, gap=5e-6, width=0.03)
    F = force_per_length * length
    drives = []
    for p, s in zip(pads, (1, -1)):
        c = p.com0.copy()
        drives.append(Drive(p.name, stiffness=(0.0, 1e4, 1e4), target=(lambda t, c=c: c),
                            force=(lambda t, s=s: np.array([s * F * min(1.0, t / 0.2), 0, 0])),
                            rot_stiffness=1e3))
    scene = Scene([can] + pads, dt=dt, gravity=(0, 0, 0), dhat=1e-5, kappa=1e2, eps_v=1e-4,
                  drives=drives)
    return scene, {"steps": steps, "view": (25, -60)}


def jelly_drop(height=0.25, tilt_deg=12.0, dt=2.5e-3):
    """A gelatin block dropped tilted onto the ground: it squashes, wobbles and bounces.

    The time step resolves the impact: contact lasts pi*sqrt(m/k) ~ 45 ms for
    this block, so h = 2.5 ms gives ~18 steps per contact. At h = 10 ms the
    implicit solver's numerical damping would remove almost all of the bounce.
    """
    gv, gf = G.box((0.7, 0.7, 0.05), center=(0, 0, -0.025))
    tv, tets = G.tet_box((0.1, 0.1, 0.1), 5)
    a = np.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    tv = tv @ Rx.T + np.array([0, 0, height])     # tilt about one axis: it lands on an edge and rocks
    jelly = Solid("jelly", tv, tets, GELATIN, color="#e74c3c")
    scene = Scene([Rigid("ground", gv, gf, GROUND, fixed=True, color="#8a8a8a"), jelly],
                  dt=dt, dhat=1e-3, kappa=1e3, newton_tol=1e-4)
    return scene, {"steps": int(round(0.9 / dt)), "view": (12, -60)}


SCENES = {
    "box_drop": (box_drop, "Rigid boxes dropped with spin onto the ground (contact + friction)"),
    "jelly_drop": (jelly_drop, "Gelatin block dropped tilted: squashes, wobbles, bounces (Neo-Hookean)"),
    "can_squeeze": (can_squeeze, "Open soda can squeezed by two short pads, then released (plasticity)"),
    "can_squeeze_sealed": (lambda: can_squeeze(sealed=True), "Sealed can (2.5 bar) squeezed the same way"),
    "can_squeeze_long": (can_squeeze_long, "Validation: long pads on a long can vs ring theory"),
}
