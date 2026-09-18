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


# --- 3D simulator pieces (docs/09 §2) -------------------------------------------

from design.oracles import sim3d as S            # noqa: E402

RNG3 = np.random.default_rng(11)


def rot3(w, t):
    """Rotation matrix exp(t [w]x) (Rodrigues)."""
    th = np.linalg.norm(w) * t
    k = w / np.linalg.norm(w)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


@law("MOD-rigid-body", "rigid_kinetic_energy", "02-rigid-body-and-contact.md")
def test_matrix_kinetic_energy_equals_rigid():
    m, J = S.box_mass_properties((0.3, 0.1, 0.05), 2700.0)
    I_body = np.trace(J) * np.eye(3) - J                    # inertia tensor from the second moment
    for _ in range(20):
        w = RNG3.normal(size=3)
        R0 = rot3(RNG3.normal(size=3), 1.0)
        Adot = (rot3(w, 1e-6) @ R0 - rot3(w, -1e-6) @ R0) / 2e-6   # d/dt exp(t[w]) R0 at t = 0
        rigid = 0.5 * w @ (R0 @ I_body @ R0.T) @ w
        assert S.affine_kinetic(m, J, np.zeros(3), Adot @ R0.T @ R0) == pytest.approx(rigid, rel=1e-6)


@law("MOD-rigid-body", "rotation_stays_orthogonal", "02-rigid-body-and-contact.md")
def test_rotation_increments_keep_the_body_rigid():
    """Q <- exp([w]) Q, applied many times with large increments, never drifts off SO(3)."""
    Q = np.eye(3)
    for _ in range(10000):
        Q = S.rotation_exp(RNG3.normal(scale=0.5, size=3)) @ Q
    assert np.abs(Q.T @ Q - np.eye(3)).max() < 1e-12
    assert np.linalg.det(Q) == pytest.approx(1.0, abs=1e-12)


def _fd_grad(fn, pts, h=1e-7):
    g = np.zeros(pts.size)
    flat = pts.ravel()
    for k in range(flat.size):
        p, m = flat.copy(), flat.copy()
        p[k] += h
        m[k] -= h
        g[k] = (fn(*p.reshape(pts.shape))[0] - fn(*m.reshape(pts.shape))[0]) / (2 * h)
    return g


@law("MOD-contact-distance", "gradient_consistency", "09-simulator.md")
def test_contact_distance_gradients_match_finite_differences():
    for _ in range(30):
        pts = RNG3.normal(size=(4, 3))
        for fn in (S.point_triangle, S.edge_edge):
            _, g = fn(*pts)
            assert np.allclose(g, _fd_grad(fn, pts), atol=1e-5)


@law("MOD-contact-distance", "distance_lipschitz", "09-simulator.md")
def test_contact_distance_is_continuous_across_regions():
    """Distance changes no faster than the vertices move: no jumps between closest-feature regions."""
    for _ in range(200):
        pts = RNG3.normal(size=(4, 3))
        delta = 1e-4 * RNG3.normal(size=(4, 3))
        for fn in (S.point_triangle, S.edge_edge):
            d0, d1 = fn(*pts)[0], fn(*(pts + delta))[0]
            assert abs(d1 - d0) <= np.linalg.norm(delta, axis=1).sum() + 1e-12


@law("MOD-contact-distance", "ccd_conservative", "09-simulator.md")
def test_ccd_bound_never_allows_contact():
    for _ in range(200):
        pts = RNG3.normal(size=(4, 3))
        disp = 2.0 * RNG3.normal(size=(4, 3))
        for fn, split in ((S.point_triangle, 1), (S.edge_edge, 2)):
            alpha = S.ccd_bound(fn(*pts)[0], disp[:split], disp[split:])
            for t in np.linspace(0, alpha, 200):
                assert fn(*(pts + t * disp))[0] > 0


@law("MOD-gas-3d", "net_force_zero", "09-simulator.md")
def test_pressurised_vessel_exerts_no_net_force_on_itself():
    from tests.design.test_admissibility import uv_sphere
    x, faces = uv_sphere()
    x = x * 0.03 + RNG3.normal(scale=2e-3, size=x.shape)
    f = S.gas_force(x, faces, 50.0)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-9 * np.abs(f).max())


# --- Ring pinched between long flat pads (docs/11) --------------------------------

from design import quantities as Qn             # noqa: E402
from design.oracles import elastica as EL        # noqa: E402

RING = dict(E=69e9, nu=0.33, sy=285e6, t=1e-4, R=0.033)
EPI = RING["E"] / (1 - RING["nu"] ** 2) * RING["t"] ** 3 / 12


@law("MOD-ring-elastica", "closed_forms_match_quadrature", "11-analytic-squeeze.md")
def test_flat_contact_closed_forms_match_the_quadrature_model():
    R = RING["R"]
    g0 = Qn.squeeze_flat_onset_gap(**RING)
    assert EL.state(EL.flat_onset_lam(R) ** 2 * 2 * EPI, R, EPI)["gap"] == pytest.approx(g0, rel=1e-8)
    for g in np.linspace(0.02, 0.99 * g0, 6):
        P = EL.force_for_gap(g, R, EPI)
        assert P == pytest.approx(Qn.squeeze_flat_force(g, RING["E"], RING["nu"], RING["t"]), rel=1e-7)
        assert EL.state(P, R, EPI)["c"] == pytest.approx(Qn.squeeze_flat_contact_length(g, R), rel=1e-6, abs=1e-12)
    gy = Qn.squeeze_first_yield_gap(**RING)
    st = EL.state(EL.force_for_gap(gy, R, EPI), R, EPI)
    assert st["kB"] - 1 / R == pytest.approx(yield_dk := Qn.yield_curvature_change(RING["E"], RING["nu"], RING["sy"], RING["t"]), rel=1e-7)
    assert EL.force_for_gap(gy, R, EPI) == pytest.approx(Qn.squeeze_first_yield_force(**RING), rel=1e-7)


@law("MOD-ring-elastica", "small_load_limit", "11-analytic-squeeze.md")
def test_small_loads_reproduce_linear_ring_compliance():
    R = RING["R"]
    P = 1e-3        # N/m: delta/R ~ 2.5e-5, so the first-order nonlinear correction is below the tolerance
    delta = 2 * R - EL.state(P, R, EPI)["gap"]
    assert delta == pytest.approx(Qn.ring_compliance(RING["E"], RING["nu"], 0.0, RING["t"], R) * P, rel=1e-4)


@law("MOD-ring-elastica", "phase_continuity", "11-analytic-squeeze.md")
def test_line_contact_joins_flat_contact_continuously():
    R = RING["R"]
    P0 = 2 * EPI * EL.flat_onset_lam(R) ** 2
    below, above = EL.state(P0 * (1 - 1e-7), R, EPI), EL.state(P0 * (1 + 1e-7), R, EPI)
    assert below["gap"] == pytest.approx(above["gap"], rel=1e-6)
    assert below["kA"] < 1e-2 / R and above["c"] < 1e-6 * R


@law("MOD-ring-elastica", "inextensibility", "11-analytic-squeeze.md")
def test_quarter_arc_length_is_conserved():
    R = RING["R"]
    for P in (0.0, 1.0, 10.0, 30.0, 80.0):
        assert EL.state(P, R, EPI)["arc"] == pytest.approx(np.pi * R / 2, rel=1e-8)
