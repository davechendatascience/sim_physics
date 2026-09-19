"""CMP-sim.adjoint: implicit derivatives through time (docs/12).

Every gradient is checked against central finite differences of the complete
forward simulation, re-run from scratch.
"""

import dataclasses

import numpy as np
import pytest

from simphys import adjoint, geometry as G
from simphys.bodies import Rigid, Shell, Solid
from simphys.materials import ALUMINUM_CAN, GELATIN, GROUND, WOOD
from simphys.scene import Scene, hat

TOL = 1e-8          # reliably reached with contact (1e-10 stalls at the iteration cap)


def converged(scene):
    """Finite differences are only a reference if every forward step converged."""
    assert max(scene.stats["newton"]) < scene.max_newton * scene.friction_iters, scene.stats["newton"]
    return scene


def _ground():
    v, f = G.box((0.4, 0.4, 0.05), center=(0, 0, -0.025))
    return Rigid("ground", v, f, GROUND, fixed=True)


def jelly_scene(E=5e4, v0=(0.0, 0.0, -0.8)):
    # general position: no vertex or edge of the block coincides with the ground
    # mesh's diagonal edge (a degenerate contact where the forward map is not smooth)
    tv, tets = G.tet_box((0.05, 0.05, 0.05), 2)
    a = np.radians(7.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    tv = tv @ Rz.T + np.array([0.0071, -0.0043, 0.0265])
    jelly = Solid("jelly", tv, tets, dataclasses.replace(GELATIN, E=E))
    jelly.v[:] = v0
    s = Scene([_ground(), jelly], dt=5e-3, dhat=1e-3, kappa=1e4, newton_tol=TOL, max_newton=300)
    s.mu_pair[:] = 0.0                                   # frictionless: exact scope of v1
    return s, jelly


def com_z(body):
    return float(body.nodal_mass @ body.x[:, 2] / body.nodal_mass.sum())


def com_z_grad(scene, body):
    gx = np.zeros((scene.nv, 3))
    gx[body.vert0:body.vert0 + body.n_verts, 2] = body.nodal_mass / body.nodal_mass.sum()
    return adjoint.vertex_objective(scene, gx)


STEPS = 8


def test_soft_block_landing_gradients_match_finite_differences():
    s, jelly = jelly_scene()
    tape = adjoint.record(s)
    converged(s.run(STEPS))
    g = adjoint.gradient(s, tape, com_z_grad(s, jelly), params=[("jelly", "E")], initial_velocity=True)

    def J(E=5e4, v0=(0.0, 0.0, -0.8)):
        s2, j2 = jelly_scene(E, v0)
        converged(s2.run(STEPS))
        return com_z(j2)

    dE = 100.0
    fd_E = (J(E=5e4 + dE) - J(E=5e4 - dE)) / (2 * dE)
    assert g[("jelly", "E")] == pytest.approx(fd_E, rel=2e-3)
    gv = g["v0"][jelly.dof0:jelly.dof0 + jelly.n_dofs].reshape(-1, 3).sum(0)
    base = np.array([0.0, 0.0, -0.8])
    for i in range(3):                                   # each component of the initial velocity
        e = np.zeros(3); e[i] = 1e-3
        fd = (J(v0=base + e) - J(v0=base - e)) / 2e-3
        assert gv[i] == pytest.approx(fd, rel=2e-3, abs=1e-8)


def box_scene(vp=(0.1, 0.0, -0.5), w=(2.0, -1.0, 3.0)):
    v, f = G.box((0.04, 0.03, 0.02), center=(0, 0, 0.02))
    box = Rigid("box", v, f, WOOD)
    box.vp[:] = vp
    box.Qdot = hat(np.asarray(w, float)) @ box.Q
    s = Scene([_ground(), box], dt=5e-3, dhat=1e-3, kappa=1e4, newton_tol=TOL, max_newton=300)
    s.mu_pair[:] = 0.0
    return s, box


def test_spinning_rigid_box_gradients_match_finite_differences():
    s, box = box_scene()
    tape = adjoint.record(s)
    s.run(STEPS)
    k = 0                                                # objective: height of one corner

    def corner(scene, b):
        return scene.body_verts(b)[k, 2]

    gx = np.zeros((s.nv, 3))
    gx[box.vert0 + k, 2] = 1.0
    g = adjoint.gradient(s, tape, adjoint.vertex_objective(s, gx), initial_velocity=True)
    gv = g["v0"][box.dof0:box.dof0 + 6]
    eps = 1e-3
    converged(s)
    for i in range(3):                                   # translational initial velocity
        dv = np.zeros(3); dv[i] = eps
        sp, bp = box_scene(vp=np.array([0.1, 0.0, -0.5]) + dv); converged(sp.run(STEPS))
        sm, bm = box_scene(vp=np.array([0.1, 0.0, -0.5]) - dv); converged(sm.run(STEPS))
        assert gv[i] == pytest.approx((corner(sp, bp) - corner(sm, bm)) / (2 * eps), rel=1e-3, abs=1e-7)
    for i in range(3):                                   # angular initial velocity
        dw = np.zeros(3); dw[i] = eps
        sp, bp = box_scene(w=np.array([2.0, -1.0, 3.0]) + dw); converged(sp.run(STEPS))
        sm, bm = box_scene(w=np.array([2.0, -1.0, 3.0]) - dw); converged(sm.run(STEPS))
        assert gv[3 + i] == pytest.approx((corner(sp, bp) - corner(sm, bm)) / (2 * eps), rel=1e-3, abs=1e-7)


def shell_scene(E=69e9, sigma_y=285e6):
    v, f, _ = G.cylinder(0.033, 0.03, 16, 3, capped=False)
    mat = dataclasses.replace(ALUMINUM_CAN, E=E, sigma_y=sigma_y)
    can = Shell("can", v, f, mat, 1e-4, fixed_verts=np.isclose(v[:, 2], -0.015))
    pv, pf = G.box((0.01, 0.02, 0.02), center=(0.033 + 5e-5 + 5e-6 + 0.005, 0, 0))
    pad = Rigid("pad", pv, pf, WOOD)
    pad.vp[:] = [-0.2, 0, 0]
    s = Scene([can, pad], dt=5e-3, gravity=(0, 0, 0), dhat=1e-5, kappa=1e5, newton_tol=TOL, max_newton=300)
    s.mu_pair[:] = 0.0
    return s, can


def test_shell_pressed_by_a_pad_gradient_wrt_modulus():
    s, can = shell_scene()
    tape = adjoint.record(s)
    converged(s.run(4))
    mid = np.abs(can.rest[:, 2]) < 0.006                                   # middle rings (not the clamped base)
    i = int(np.where(mid)[0][np.argmax(can.rest[mid, 0])])                 # wall point facing the pad
    gx = np.zeros((s.nv, 3)); gx[can.vert0 + i, 0] = 1.0
    g = adjoint.gradient(s, tape, adjoint.vertex_objective(s, gx), params=[("can", "E")])
    dE = 69e9 * 1e-3
    sp, cp = shell_scene(E=69e9 + dE); converged(sp.run(4))
    sm, cm = shell_scene(E=69e9 - dE); converged(sm.run(4))
    fd = (cp.x[i, 0] - cm.x[i, 0]) / (2 * dE)
    assert abs(fd) > 0
    assert g[("can", "E")] == pytest.approx(fd, rel=1e-3)


def test_plastic_flow_is_refused_not_silently_differentiated():
    s, can = shell_scene(sigma_y=1e3)                    # yields on the first contact
    tape = adjoint.record(s)
    s.run(4)
    assert can.alpha.max() > 0
    with pytest.raises(NotImplementedError):
        adjoint.gradient(s, tape, np.zeros(s.nd), params=[("can", "E")])


def test_recording_does_not_change_the_forward_run():
    a, ja = jelly_scene()
    b, jb = jelly_scene()
    adjoint.record(b)
    a.run(STEPS)
    b.run(STEPS)
    assert np.array_equal(ja.x, jb.x) and np.array_equal(ja.v, jb.v)
