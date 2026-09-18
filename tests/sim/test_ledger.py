"""CMP-sim.ledger: the irreversible-state verdict (docs/03 §4)."""

import numpy as np

from simphys import geometry as G, ledger
from simphys.bodies import Shell
from simphys.materials import ALUMINUM_CAN
from simphys.scene import Scene
from simphys.scenes import can_squeeze_long


def _can():
    v, f, _ = G.cylinder(0.033, 0.06, 24, 6, capped=False)
    can = Shell("can", v, f, ALUMINUM_CAN, 1e-4)
    return Scene([can], dt=0.01, gravity=(0, 0, 0)), can


def test_rigid_motion_is_not_an_alteration():
    s, can = _can()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    can.x = can.rest @ R.T + np.array([0.2, -0.1, 0.05])
    r = ledger.shell_report(s, "can")
    assert r["max_shape_change_mm"] < 1e-9
    assert not r["altered_by_shape"] and not r["altered_by_plastic_strain"]


def test_plastic_strain_above_threshold_is_an_alteration():
    s, can = _can()
    can.alpha[5, 0] = 3e-3
    r = ledger.shell_report(s, "can")
    assert r["altered_by_plastic_strain"] and r["plastic_area_mm2"] > 0


def test_residual_dent_is_an_alteration():
    s, can = _can()
    x = can.rest.copy()
    x[0, :2] *= 0.99                                   # a 0.33 mm inward dent at one node
    can.x = x
    assert ledger.shell_report(s, "can")["altered_by_shape"]


def test_light_squeeze_and_release_leaves_the_can_unaltered():
    s, _ = can_squeeze_long(force_per_length=1.0, n_theta=32, n_z=6)
    pads = [d for d in s.drives]
    for d in pads:
        f0 = d.force
        d.force = (lambda t, f0=f0: f0(t) * max(0.0, 1 - max(0.0, t - 0.3) / 0.2))   # release after 0.3 s
    for _ in range(140):
        s.step()
    r = ledger.shell_report(s, "can")
    assert r["max_plastic_strain"] == 0.0
    assert not r["altered_by_shape"]
