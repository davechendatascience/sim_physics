"""CMP-sim.bodies: rigid bodies, thin shells and solids (docs/02 §1, docs/09 §2, §4)."""

import numpy as np
import pytest

from design.quantities import ring_compliance
from simphys import geometry as G, kernels
from simphys.bodies import Rigid, Shell, Solid
from simphys.materials import ALUMINUM_CAN, SILICONE
from simphys.scene import Scene
from simphys.scenes import can_squeeze_long
from tests.sim.helpers import box


def test_spinning_body_stays_exactly_rigid():
    b = box("b", size=(0.2, 0.1, 0.05))
    s = Scene([b], dt=0.01, gravity=(0, 0, 0))
    b.Qdot = np.array([[0, -3.0, 1.0], [3.0, 0, -2.0], [-1.0, 2.0, 0]])     # [w] with w = (2, 1, 3) rad/s
    for _ in range(200):
        s.step()
    assert np.abs(b.Q.T @ b.Q - np.eye(3)).max() < 1e-12
    assert np.linalg.det(b.Q) == pytest.approx(1.0, abs=1e-12)


def test_rigid_mass_properties_match_closed_form():
    v, f = G.box((0.3, 0.1, 0.05))
    m, com, J, V = G.mass_properties(v, f, 1000.0)
    assert V == pytest.approx(0.3 * 0.1 * 0.05)
    assert np.allclose(com, 0, atol=1e-12)
    assert np.allclose(J, m / 12 * np.diag([0.09, 0.01, 0.0025]))


def test_shell_bending_energy_of_a_rolled_strip_matches_plate_theory():
    """A flat strip bent isometrically onto a cylinder stores D/2 kappa^2 per unit area."""
    L, W, rho = 0.02, 0.01, 0.05
    nx, ny = 40, 20
    xs, ys = np.meshgrid(np.linspace(0, L, nx + 1), np.linspace(0, W, ny + 1), indexing="ij")
    flat = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    idx = lambda i, j: i * (ny + 1) + j
    faces = [f for i in range(nx) for j in range(ny)
             for f in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])]
    sh = Shell("strip", flat, np.array(faces), ALUMINUM_CAN, 1e-4)
    th = flat[:, 0] / rho
    rolled = np.stack([rho * np.sin(th), flat[:, 1], rho * (1 - np.cos(th))], 1)
    m = ALUMINUM_CAN
    nF = len(sh.faces)
    e = kernels.call(kernels.shell_E, rolled[sh.stencil], sh.has_nb, sh.Ainv, sh.abar, sh.IIbar,
                     sh.area, sh.thickness, np.full(nF, m.E), np.full(nF, m.nu), sh.eps_p)
    interior = sh.has_nb.min(1) > 0                              # boundary faces lack a neighbour normal
    D = m.E * 1e-4 ** 3 / (12 * (1 - m.nu ** 2))
    expected = 0.5 * D / rho ** 2 * sh.area[interior].sum()
    assert e[interior].sum() == pytest.approx(expected, rel=0.03)


def test_shell_energy_is_frame_indifferent():
    v, f, _ = G.cylinder(0.033, 0.04, 16, 3, capped=False)
    sh = Shell("c", v, f, ALUMINUM_CAN, 1e-4)
    rng = np.random.default_rng(0)
    x = v + 1e-3 * rng.normal(size=v.shape)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    m, nF = ALUMINUM_CAN, len(sh.faces)
    E = lambda y: kernels.call(kernels.shell_E, y[sh.stencil], sh.has_nb, sh.Ainv, sh.abar, sh.IIbar,
                                sh.area, sh.thickness, np.full(nF, m.E), np.full(nF, m.nu), sh.eps_p).sum()
    assert E(x @ Q.T + np.array([0.3, -1.0, 2.0])) == pytest.approx(E(x), rel=1e-9)


def _long_cylinder_ratio(n_theta):
    s, _ = can_squeeze_long(force_per_length=1.0, n_theta=n_theta, n_z=6)
    can = s.by_name["can"]
    w0 = np.ptp(can.x[:, 0])
    for _ in range(140):
        s.step()
    return (w0 - np.ptp(can.x[:, 0])) / ring_compliance(69e9, 0.33, 0.0, 1e-4, 0.033)


def test_long_can_matches_ring_compliance_and_converges():
    """docs/09 §4: a long can between long pads is a ring in plane strain."""
    coarse, fine = _long_cylinder_ratio(32), _long_cylinder_ratio(48)
    assert 1.0 < fine < coarse < 1.08                      # stiff-side error that shrinks with refinement


def test_solid_at_rest_stays_at_rest():
    v, tets = G.tet_box((0.1, 0.1, 0.1), 3)
    jelly = Solid("jelly", v, tets, SILICONE)
    s = Scene([jelly], dt=0.01, gravity=(0, 0, 0))
    for _ in range(5):
        s.step()
    assert np.abs(jelly.x - v).max() < 1e-12
