"""CMP-sim.contact: IPC barrier, CCD and friction (docs/02 §2, docs/09 §2)."""

import numpy as np
import pytest

from simphys import geometry as G
from simphys.bodies import Rigid, Shell
from simphys.materials import ALUMINUM_CAN, GROUND, WOOD
from simphys.scene import Scene
from simphys.scenes import box_drop
from tests.sim.helpers import box, ground, min_gap


def test_tumbling_boxes_never_interpenetrate():
    s, _ = box_drop()
    for _ in range(80):
        s.step()
        assert min_gap(s) > 0


def _incline(theta_deg, steps=40):
    """A box on flat ground under tilted gravity is a box on an incline."""
    th = np.radians(theta_deg)
    g = ground()
    b = box("b", center=(0.0, 0.0, 0.0502))
    s = Scene([g, b], dt=0.01, gravity=(9.81 * np.sin(th), 0.0, -9.81 * np.cos(th)), dhat=1e-3, kappa=1e5)
    mu = float(s.mu_pair[0, 1])
    for _ in range(steps):
        s.step()
    return b.vp[0], mu, th, s


def test_block_sticks_below_the_friction_angle():
    v, mu, th, s = _incline(15)
    assert np.tan(th) < mu
    assert abs(v) < 2 * s.eps_v                       # creep bounded by the friction smoothing


def test_block_slides_at_the_kinetic_rate_above_the_friction_angle():
    v, mu, th, s = _incline(40, steps=30)
    assert np.tan(th) > mu
    expected = 9.81 * (np.sin(th) - mu * np.cos(th)) * s.t
    assert v == pytest.approx(expected, rel=0.03)


def test_shell_thickness_is_respected_by_contact():
    """A pad pressed on a 0.1 mm wall stops at half the thickness from its mid-surface."""
    v, f, _ = G.cylinder(0.033, 0.04, 24, 4, capped=False)
    can = Shell("can", v, f, ALUMINUM_CAN, 1e-4)
    pv, pf = G.box((0.01, 0.02, 0.02), center=(0.033 + 5e-5 + 5e-6 + 0.005, 0, 0))
    pad = Rigid("pad", pv, pf, WOOD)
    s = Scene([can, pad], dt=0.005, gravity=(0, 0, 0), dhat=1e-5, kappa=1e3)
    pad.vp[:] = [-0.05, 0, 0]
    gaps = []
    for _ in range(10):
        s.step()
        gaps.append((pad.p[0] - 0.005) - can.x[:, 0].max())    # pad face to wall mid-surface
    assert min(gaps) > 5e-5                             # never closer than t/2 to the mid-surface
    assert min(gaps) < 5e-5 + 1e-5                      # and it did reach contact (within d-hat)


def test_fast_box_does_not_tunnel_through_a_thin_plate():
    pv, pf = G.box((0.005, 0.3, 0.3), center=(0.0, 0.0, 0.0))
    plate = Rigid("plate", pv, pf, GROUND, fixed=True)
    b = box("b", size=(0.04, 0.04, 0.04), center=(-0.1, 0.0, 0.0))
    s = Scene([plate, b], dt=0.01, gravity=(0, 0, 0), dhat=1e-3, kappa=1e4)
    b.vp[:] = [30.0, 0.0, 0.0]                           # 0.3 m per step: 60x the plate thickness
    for _ in range(10):
        s.step()
        assert b.p[0] < 0.0 and min_gap(s) > 0
