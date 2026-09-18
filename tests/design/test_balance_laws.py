"""Balance laws for the time-stepping and friction models (docs/01 §3, docs/02 §2-3).

Each test runs the reference model in design/oracles on a scenario with a
known physical answer. The discretisation is allowed to differ from the
continuous law only in the way the design states: linear momentum exactly,
angular momentum and energy to first order in h, energy never gained.
"""

import numpy as np
import pytest

from design.oracles.contact import barrier, f1
from design.oracles.integrator import (Plane, Step, System, angular_momentum, kinetic, momentum,
                                       spring_energy, step)
from design.quantities import _grasp_slide_speed
from tests.design.conftest import law

G = 9.81


def collision(offset_v=np.zeros(2)):
    """Off-centre collision of two discs, with friction, no gravity."""
    s = System(x=np.array([[-0.12, 0.02], [0.12, 0.0]]),
               v=np.array([[2.0, 0.0], [-1.0, 0.3]]) + offset_v,
               m=np.array([1.0, 2.0]), r=np.array([0.05, 0.05]), pair_mu=0.5, kappa=1e3, dhat=1e-3)
    return s


def rotating_dimer():
    return System(x=np.array([[-0.5, 0.0], [0.5, 0.0]]), v=np.array([[0.0, -1.0], [0.0, 1.0]]),
                  m=np.array([1.0, 1.0]), r=np.array([0.01, 0.01]), springs=[(0, 1, 200.0, 1.0)])


def total_energy(s):
    e = kinetic(s) + spring_energy(s) - float(np.sum(s.m * (s.x @ s.gravity)))
    st = Step(s, 1.0)
    for kind, i, j, d in st.contacts(s.x):
        e += barrier(d, s.dhat, s.kappa)
    return e


@law("MOD-ip-step", "linear_momentum", "01-architecture.md")
def test_linear_momentum_exact():
    s = collision()
    p0 = momentum(s)
    for _ in range(40):
        step(s, 0.005)
        assert np.allclose(momentum(s), p0, rtol=0, atol=1e-7 * np.linalg.norm(p0))


@law("MOD-ip-step", "newton_third", "01-architecture.md")
def test_contact_forces_equal_and_opposite():
    s = collision()
    ang = np.radians(20)                                   # oblique, inside the barrier's range
    s.x = np.array([[0.0, 0.0], [0.1005 * np.cos(ang), 0.1005 * np.sin(ang)]])
    st = Step(s, 1.0)
    h = 1e-9

    def contact_energy(x):
        return sum(barrier(d, s.dhat, s.kappa) for *_, d in st.contacts(x))

    grad = np.zeros_like(s.x)
    for i in range(2):
        for k in range(2):
            xp, xm = s.x.copy(), s.x.copy()
            xp[i, k] += h
            xm[i, k] -= h
            grad[i, k] = (contact_energy(xp) - contact_energy(xm)) / (2 * h)
    assert np.linalg.norm(grad[0]) > 0
    assert np.allclose(grad[0], -grad[1], rtol=1e-5)


@law("MOD-ip-step", "energy_nonincrease", "01-architecture.md")
def test_energy_never_increases():
    """Springy dimer bouncing on frictionless ground under gravity."""
    s = System(x=np.array([[0.0, 0.3], [0.2, 0.4]]), v=np.array([[0.5, 0.0], [0.0, 0.0]]),
               m=np.array([1.0, 1.0]), r=np.array([0.05, 0.05]), springs=[(0, 1, 500.0, 0.22)],
               planes=[Plane(np.zeros(2), np.array([0.0, 1.0]))], gravity=np.array([0.0, -G]))
    e = total_energy(s)
    for _ in range(150):
        step(s, 0.005)
        e_new = total_energy(s)
        assert e_new <= e + 1e-9 * max(1.0, abs(e))
        e = e_new


def _drift(make, quantity, h, T=1.0):
    s = make()
    q0 = quantity(s)
    for _ in range(int(round(T / h))):
        step(s, h)
    return abs(quantity(s) - q0)


@law("MOD-ip-step", "energy_first_order", "01-architecture.md")
def test_energy_error_is_first_order():
    e = lambda s: kinetic(s) + spring_energy(s)
    coarse, fine = _drift(rotating_dimer, e, 0.01), _drift(rotating_dimer, e, 0.005)
    assert coarse > 0 and 0.35 < fine / coarse < 0.65


@law("MOD-ip-step", "angular_momentum_first_order", "01-architecture.md")
def test_angular_momentum_error_is_first_order():
    coarse, fine = _drift(rotating_dimer, angular_momentum, 0.01), _drift(rotating_dimer, angular_momentum, 0.005)
    assert coarse > 0 and 0.35 < fine / coarse < 0.65


@law("MOD-ip-step", "galilean_invariance", "01-architecture.md")
def test_galilean_invariance():
    a, b = collision(), collision(np.array([3.0, -2.0]))
    for _ in range(40):
        step(a, 0.005)
        step(b, 0.005)
    rel_a, rel_b = a.x[1] - a.x[0], b.x[1] - b.x[0]
    assert np.allclose(rel_a, rel_b, atol=1e-7)
    assert np.allclose(a.v[1] - a.v[0], b.v[1] - b.v[0], atol=1e-6)


@law("MOD-ip-step", "non_penetration", "01-architecture.md")
def test_no_penetration_or_tunnelling_at_large_steps():
    """50 m/s at h = 10 ms moves 10 radii per step; the barrier + CCD must still stop it.

    Tunnelling means the motion between two states passes through an overlap,
    so the whole path the solver takes (piecewise linear through its Newton
    iterates, which is what IPC certifies) is checked, not just end states.
    The balls may legitimately squirt sideways around each other when rammed
    against the wall: a line of discs under compression is unstable.
    """
    s = System(x=np.array([[-0.5, 0.0], [0.0, 0.0]]), v=np.array([[50.0, 0.0], [0.0, 0.0]]),
               m=np.array([1.0, 1.0]), r=np.array([0.05, 0.05]),
               planes=[Plane(np.array([0.3, 0.0]), np.array([-1.0, 0.0]))], kappa=1e4, dhat=1e-3)
    for _ in range(30):
        info = step(s, 0.01)
        assert info["min_gap"] > 0
        for a, b in zip(info["path"], info["path"][1:]):
            for t in np.linspace(0, 1, 201):
                x = a + t * (b - a)
                assert np.linalg.norm(x[0] - x[1]) > 0.1
                assert np.all(x[:, 0] < 0.3 - 0.05)


def incline(theta, mu, T=0.5, h=0.01):
    n = np.array([-np.sin(theta), np.cos(theta)])
    s = System(x=np.array([n * (0.05 + 2e-4)]), v=np.zeros((1, 2)), m=np.array([1.0]),
               r=np.array([0.05]), planes=[Plane(np.zeros(2), n, mu)], gravity=np.array([0.0, -G]))
    t = -np.array([np.cos(theta), np.sin(theta)])        # down-slope
    works = [step(s, h)["friction_work"] for _ in range(int(round(T / h)))]
    return float(s.v[0] @ t), works, s


@law("MOD-friction", "coulomb_bound", "02-rigid-body-and-contact.md")
def test_friction_never_exceeds_coulomb_bound():
    eps = 1e-5
    y = np.linspace(0, 10 * eps, 1001)
    ratio = np.array([f1(v, eps) for v in y])            # |friction| / (mu * lambda)
    assert ratio.min() >= 0 and ratio.max() <= 1 + 1e-12


@law("MOD-friction", "stick_below_limit", "02-rigid-body-and-contact.md")
def test_block_sticks_when_slope_below_friction_angle():
    speed, _, s = incline(np.radians(30), mu=0.8)
    assert abs(speed) <= s.eps_v                           # creep bounded by the smoothing velocity


@law("MOD-friction", "kinetic_slide_rate", "02-rigid-body-and-contact.md")
def test_sliding_block_matches_kinetic_friction():
    th, mu, T = np.radians(30), 0.3, 0.5
    speed, _, _ = incline(th, mu, T)
    assert speed == pytest.approx(G * (np.sin(th) - mu * np.cos(th)) * T, rel=0.01)


@law("MOD-friction", "dissipation_nonneg", "02-rigid-body-and-contact.md")
def test_friction_only_removes_energy():
    _, works, _ = incline(np.radians(30), mu=0.3)
    assert max(works) <= 1e-12 and sum(works) < 0


@law("MOD-friction", "grasp_threshold", "02-rigid-body-and-contact.md")
def test_flat_pad_grasp_holds_above_and_slips_below_fmin():
    held, _ = _grasp_slide_speed(1.05, round_fingertips=False)
    slid, predicted = _grasp_slide_speed(0.95, round_fingertips=False)
    assert abs(held) < 2e-3
    assert slid == pytest.approx(predicted, rel=0.02)
