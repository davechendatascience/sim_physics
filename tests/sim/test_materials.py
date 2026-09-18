"""CMP-sim.materials: shell plasticity and enclosed gas (docs/03 §2, §5, docs/09 §2)."""

import numpy as np
import pytest

from design.oracles.sim3d import return_map_ps
from simphys import geometry as G
from simphys.bodies import Shell
from simphys.materials import ALUMINUM_CAN, plane_stress_return
from simphys.scene import Gas, Scene


def test_return_map_agrees_with_the_design_reference_model():
    """Differential test: the simulator's vectorised return map vs design/oracles/sim3d."""
    rng = np.random.default_rng(1)
    m = ALUMINUM_CAN
    for H in (0.0, 1e9):
        eps = rng.normal(scale=6e-3, size=(500, 2, 2))
        eps = 0.5 * (eps + np.swapaxes(eps, 1, 2))
        eps_p = rng.normal(scale=1e-3, size=(500, 2, 2))
        eps_p = 0.5 * (eps_p + np.swapaxes(eps_p, 1, 2))
        alpha = np.abs(rng.normal(scale=1e-3, size=500))
        s1, p1, a1 = plane_stress_return(m.E, m.nu, m.sigma_y, H, eps, eps_p, alpha)
        s2, p2, a2 = return_map_ps(m.E, m.nu, m.sigma_y, H, eps, eps_p, alpha)
        assert np.allclose(s1, s2, rtol=1e-6, atol=1e-6 * m.sigma_y)
        assert np.allclose(a1, a2, rtol=1e-6, atol=1e-12)


def test_return_map_is_the_identity_below_yield():
    m = ALUMINUM_CAN
    eps = np.full((10, 2, 2), 1e-4)
    sig, eps_p, alpha = plane_stress_return(m.E, m.nu, m.sigma_y, 0.0, eps, np.zeros_like(eps), np.zeros(10))
    assert np.all(eps_p == 0) and np.all(alpha == 0)


def test_sealed_can_swells_by_the_thin_wall_hoop_strain():
    """p R / (t E) (1 - nu/2) radial strain at mid-height of a closed thin cylinder."""
    R, H, t, p = 0.033, 0.122, 1e-4, 2.5e5
    v, f, label = G.cylinder(R, H, 40, 30, capped=True)
    can = Shell("can", v, f, ALUMINUM_CAN, np.where(label == 0, t, 2.5e-4))
    s = Scene([can], dt=2e-3, gravity=(0, 0, 0), gases=[Gas("can", p)])
    mid = (np.abs(v[:, 2]) < 0.01) & (np.hypot(v[:, 0], v[:, 1]) > R * 0.99)
    r0 = np.hypot(v[mid, 0], v[mid, 1]).mean()
    for _ in range(40):
        s.step()
    r = np.hypot(can.x[mid, 0], can.x[mid, 1]).mean()
    expected = p * R / (t * ALUMINUM_CAN.E) * (1 - ALUMINUM_CAN.nu / 2) * R
    assert r - r0 == pytest.approx(expected, rel=0.05)
