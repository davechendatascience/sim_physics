"""CMP-sim.solver: the incremental-potential step (docs/01 §3)."""

import numpy as np
import pytest

from simphys.scene import Scene
from tests.sim.helpers import box, ground, total_energy


def test_free_fall_matches_implicit_euler_exactly():
    """With only gravity, implicit Euler gives v_n = n h g and x_n = x_0 + h^2 g n(n+1)/2."""
    b = box("b")
    s = Scene([b], dt=0.01)
    z0 = b.p[2]
    for n in range(1, 31):
        s.step()
        assert b.vp[2] == pytest.approx(-9.81 * 0.01 * n, rel=1e-9)
    assert b.p[2] == pytest.approx(z0 - 9.81 * 0.01 ** 2 * 30 * 31 / 2, rel=1e-9)


def test_linear_momentum_is_conserved_through_a_frictional_collision():
    a = box("a", center=(-0.12, 0.01, 0.0))
    c = box("c", size=(0.08, 0.12, 0.1), center=(0.12, 0.0, 0.02))
    s = Scene([a, c], dt=0.01, gravity=(0, 0, 0), dhat=1e-3, kappa=1e4)
    a.vp[:] = [1.5, 0.0, 0.1]
    c.vp[:] = [-0.5, 0.2, 0.0]
    p0 = a.mass * a.vp + c.mass * c.vp
    touched = False
    for _ in range(40):
        s.step()
        p = a.mass * a.vp + c.mass * c.vp
        assert np.allclose(p, p0, atol=1e-6 * np.linalg.norm(p0))
        touched |= c.vp[0] > -0.4
    assert touched, "the boxes never collided"


def test_energy_never_increases_on_frictionless_ground():
    g = ground()
    b = box("b", center=(0.0, 0.0, 0.15))
    g.material.__class__  # frictionless via the pair table below
    s = Scene([g, b], dt=0.01, dhat=1e-3, kappa=1e4)
    s.mu_pair[:] = 0.0
    b.vp[:] = [0.3, 0.0, 0.0]
    E = total_energy(s)
    for _ in range(80):
        s.step()
        E_new = total_energy(s)
        assert E_new <= E + 1e-6
        E = E_new


def test_newton_converges_in_a_handful_of_iterations_for_a_resting_contact():
    g = ground()
    b = box("b", center=(0.0, 0.0, 0.0502))
    s = Scene([g, b], dt=0.01, dhat=1e-3, kappa=1e4)
    its = [s.step() for _ in range(30)]
    assert np.mean(its[-10:]) <= 6
