"""Thermodynamic and mechanical admissibility of the constitutive and contact models.

Classical continuum mechanics requires: energy that is frame-indifferent,
stress that derives from it, a stress-free reference, agreement with linear
elasticity at small strain, infinite energy at zero volume, ellipticity,
non-negative plastic dissipation, and yield-surface convexity.
"""

import numpy as np
import pytest

from design.oracles import constitutive as C
from design.oracles.contact import barrier, barrier_grad, pressure_dependent_mu
from tests.design.conftest import law

RNG = np.random.default_rng(7)
MU, LAM = C.lame(1.0e6, 0.45)          # silicone-like pad
AL = dict(E=69e9, nu=0.33, sigma_y0=285e6, H=1e9, Q=25e6, b=20.0)


def rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def deformation(rng, spread=0.3):
    while True:
        F = np.eye(3) + spread * rng.normal(size=(3, 3))
        if 0.5 < np.linalg.det(F) < 2.0:
            return F


# --- IPC barrier ---------------------------------------------------------

DHAT, KAPPA = 1e-3, 1e3


@law("MOD-barrier", "repulsive_only", "02-rigid-body-and-contact.md")
def test_barrier_force_is_repulsive_only():
    d = np.linspace(1e-9, 2 * DHAT, 5000)
    assert all(barrier_grad(x, DHAT, KAPPA) <= 0 for x in d)       # force -db/dd >= 0: no adhesion
    assert all(barrier(x, DHAT, KAPPA) >= 0 for x in d)


@law("MOD-barrier", "infinite_at_contact", "02-rigid-body-and-contact.md")
def test_barrier_blocks_contact():
    assert barrier(0.0, DHAT, KAPPA) == np.inf
    values = [barrier(DHAT * 10.0 ** -k, DHAT, KAPPA) for k in range(1, 12)]
    assert all(b2 > b1 for b1, b2 in zip(values, values[1:]))


@law("MOD-barrier", "smooth_activation", "02-rigid-body-and-contact.md")
def test_barrier_switches_on_smoothly():
    for d in (DHAT * (1 - 1e-6), DHAT, DHAT * (1 + 1e-6)):
        assert abs(barrier(d, DHAT, KAPPA)) < 1e-12
        assert abs(barrier_grad(d, DHAT, KAPPA)) < 1e-6


# --- Neo-Hookean -----------------------------------------------------------

@law("MOD-neo-hookean", "objectivity", "03-deformables-and-materials.md")
def test_neo_hookean_is_frame_indifferent():
    for _ in range(50):
        F, Q = deformation(RNG), rotation(RNG)
        assert C.nh_energy(Q @ F, MU, LAM) == pytest.approx(C.nh_energy(F, MU, LAM), rel=1e-10, abs=1e-6)
        assert np.allclose(C.nh_pk1(Q @ F, MU, LAM), Q @ C.nh_pk1(F, MU, LAM), atol=1e-6 * MU)


@law("MOD-neo-hookean", "stress_free_reference", "03-deformables-and-materials.md")
def test_neo_hookean_reference_is_stress_free():
    assert C.nh_energy(np.eye(3), MU, LAM) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(C.nh_pk1(np.eye(3), MU, LAM), 0.0, atol=1e-9)


@law("MOD-neo-hookean", "potential_consistency", "03-deformables-and-materials.md")
def test_neo_hookean_stress_derives_from_energy():
    h = 1e-7
    for _ in range(10):
        F = deformation(RNG)
        fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                dF = np.zeros((3, 3))
                dF[i, j] = h
                fd[i, j] = (C.nh_energy(F + dF, MU, LAM) - C.nh_energy(F - dF, MU, LAM)) / (2 * h)
        assert np.allclose(fd, C.nh_pk1(F, MU, LAM), rtol=1e-5, atol=1e-5 * MU)


@law("MOD-neo-hookean", "linear_limit", "03-deformables-and-materials.md")
def test_neo_hookean_matches_hooke_at_small_strain():
    for _ in range(10):
        grad_u = 1e-6 * RNG.normal(size=(3, 3))
        eps = 0.5 * (grad_u + grad_u.T)
        hooke = LAM * np.trace(eps) * np.eye(3) + 2 * MU * eps
        P = C.nh_pk1(np.eye(3) + grad_u, MU, LAM)
        assert np.allclose(P, hooke, rtol=1e-4, atol=1e-4 * np.abs(hooke).max())


@law("MOD-neo-hookean", "growth", "03-deformables-and-materials.md")
def test_neo_hookean_energy_blows_up_at_zero_volume():
    values = [C.nh_energy(np.diag([1.0, 1.0, 10.0 ** -k]), MU, LAM) for k in range(1, 10)]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert C.nh_energy(np.diag([1.0, 1.0, -0.1]), MU, LAM) == np.inf     # inversion is not admissible


@law("MOD-neo-hookean", "ellipticity", "03-deformables-and-materials.md")
def test_neo_hookean_is_elliptic_in_its_working_range():
    """Principal stretches 0.6-1.6: pad compression up to 40%, stretch up to 60%."""
    for _ in range(20):
        stretches = RNG.uniform(0.6, 1.6, size=3)
        F = rotation(RNG) @ np.diag(stretches) @ rotation(RNG)
        assert C.acoustic_min_eig(F, MU, LAM, rng=RNG) > 0


@law("MOD-neo-hookean", "cauchy_symmetry", "03-deformables-and-materials.md")
def test_neo_hookean_cauchy_stress_is_symmetric():
    for _ in range(20):
        F = deformation(RNG)
        sigma = C.cauchy(C.nh_pk1(F, MU, LAM), F)
        assert np.allclose(sigma, sigma.T, atol=1e-8 * MU)


@law("MOD-neo-hookean", "closed_cycle_work", "03-deformables-and-materials.md")
def test_neo_hookean_returns_all_work_on_a_closed_path():
    A, B = 0.2 * RNG.normal(size=(3, 3)), 0.2 * RNG.normal(size=(3, 3))
    t = np.linspace(0, 2 * np.pi, 20001)
    Fs = [np.eye(3) + A * np.sin(s) + B * (1 - np.cos(s)) for s in t]
    Ps = [C.nh_pk1(F, MU, LAM) for F in Fs]
    work = sum(0.5 * np.sum((P0 + P1) * (F1 - F0)) for P0, P1, F0, F1 in zip(Ps, Ps[1:], Fs, Fs[1:]))
    peak = max(C.nh_energy(F, MU, LAM) for F in Fs)
    assert abs(work) < 1e-6 * peak


# --- J2 plasticity ---------------------------------------------------------

def strain_path(amplitude, n=400, rng=RNG):
    """A random closed strain loop: out along a random direction and back."""
    d = rng.normal(size=(3, 3))
    d = 0.5 * (d + d.T)
    d /= np.linalg.norm(d)
    s = np.concatenate([np.linspace(0, 1, n), np.linspace(1, -1, 2 * n), np.linspace(-1, 0, n)])
    return [amplitude * k * d for k in s]


def run(model, path):
    out = []
    for eps in path:
        sig, deps_p = model.update(eps)
        out.append((eps, sig, deps_p))
    return out


def cycle_work(history):
    return sum(0.5 * np.sum((s0 + s1) * (e1 - e0))
               for (e0, s0, _), (e1, s1, _) in zip(history, history[1:]))


@law("MOD-j2", "yield_consistency", "03-deformables-and-materials.md")
def test_j2_stress_stays_on_or_inside_yield_surface():
    for _ in range(5):
        m = C.J2(**AL)
        for eps, sig, _ in run(m, strain_path(0.02)):
            assert m.yield_fn(sig) <= 1e-6 * AL["sigma_y0"]


@law("MOD-j2", "dissipation_nonneg", "03-deformables-and-materials.md")
def test_j2_plastic_dissipation_is_non_negative():
    m = C.J2(**AL)
    dissipated = [float(np.sum(sig * deps_p)) for _, sig, deps_p in run(m, strain_path(0.02))]
    assert min(dissipated) >= -1e-9 * AL["sigma_y0"] and sum(dissipated) > 0


@law("MOD-j2", "plastic_incompressibility", "03-deformables-and-materials.md")
def test_j2_plastic_flow_preserves_volume():
    m = C.J2(**AL)
    for _, _, deps_p in run(m, strain_path(0.02)):
        assert abs(np.trace(deps_p)) < 1e-12


@law("MOD-j2", "drucker_ilyushin", "03-deformables-and-materials.md")
def test_j2_closed_strain_cycle_absorbs_work():
    for _ in range(5):
        assert cycle_work(run(C.J2(**AL), strain_path(0.02))) > 0


@law("MOD-j2", "elastic_cycle_no_dissipation", "03-deformables-and-materials.md")
def test_j2_elastic_cycle_is_lossless():
    m = C.J2(**AL)
    history = run(m, strain_path(1e-4))            # well below yield strain (~4e-3)
    assert np.allclose(m.eps_p, 0.0)
    assert abs(cycle_work(history)) < 1e-9 * AL["sigma_y0"] * 1e-4


@law("MOD-j2", "shear_limit", "03-deformables-and-materials.md")
def test_j2_pure_shear_saturates_at_von_mises_limit():
    m = C.J2(E=69e9, nu=0.33, sigma_y0=285e6)        # perfect plasticity
    for g in np.linspace(0, 0.02, 200):
        eps = np.zeros((3, 3))
        eps[0, 1] = eps[1, 0] = g / 2
        sig, _ = m.update(eps)
    assert sig[0, 1] == pytest.approx(285e6 / np.sqrt(3), rel=1e-9)


# --- Hill48 ----------------------------------------------------------------

def deviatoric_basis():
    """Orthonormal basis of deviatoric stress in (s11, s22, s33, s23, s31, s12) coordinates."""
    B = np.zeros((6, 5))
    B[:3, 0] = np.array([1, -1, 0]) / np.sqrt(2)
    B[:3, 1] = np.array([1, 1, -2]) / np.sqrt(6)
    B[3, 2] = B[4, 3] = B[5, 4] = 1.0
    return B


@law("MOD-hill48", "convexity", "03-deformables-and-materials.md")
def test_hill48_from_can_r_values_is_convex():
    """r-values from the docs/03 material record: r0=0.6, r45=0.8, r90=0.7."""
    A = C.hill48_matrix(C.hill48_from_r(0.6, 0.8, 0.7))
    B = deviatoric_basis()
    assert np.linalg.eigvalsh(B.T @ A @ B).min() > 0          # strictly convex on deviatoric stress
    assert np.linalg.eigvalsh(A).min() > -1e-12               # never negative (pressure-insensitive)


@law("MOD-hill48", "von_mises_limit", "03-deformables-and-materials.md")
def test_hill48_reduces_to_von_mises_when_isotropic():
    A = C.hill48_matrix(C.hill48_from_r(1.0, 1.0, 1.0))
    for _ in range(20):
        s = RNG.normal(size=6)
        s11, s22, s33, s23, s31, s12 = s
        vm2 = 0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s23 ** 2 + s31 ** 2 + s12 ** 2)
        assert s @ A @ s == pytest.approx(vm2, rel=1e-12)


# --- Enclosed gas ----------------------------------------------------------

N_MOL, T_GAS = 0.02, 293.0


@law("MOD-gas", "potential_consistency", "03-deformables-and-materials.md")
def test_gas_pressure_is_minus_energy_derivative():
    for V in np.linspace(3e-4, 4e-4, 11):
        dV = 1e-10
        dpsi = (C.gas_energy(V + dV, N_MOL, T_GAS) - C.gas_energy(V - dV, N_MOL, T_GAS)) / (2 * dV)
        assert -dpsi == pytest.approx(C.gas_pressure(V, N_MOL, T_GAS), rel=1e-6)


@law("MOD-gas", "closed_cycle_work", "03-deformables-and-materials.md")
def test_gas_returns_all_work_on_a_closed_volume_cycle():
    t = np.linspace(0, 2 * np.pi, 20001)
    V = 3.5e-4 * (1 + 0.05 * np.sin(t))
    p = C.gas_pressure(V, N_MOL, T_GAS)
    work = np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(V))
    assert abs(work) < 1e-6 * np.max(p) * 3.5e-4


# --- Pressure-dependent pad friction --------------------------------------

@law("MOD-pressure-friction", "friction_force_monotone", "02-rigid-body-and-contact.md")
def test_pad_friction_force_grows_with_pressure():
    """mu falls with pressure, but the friction force mu*p must still rise, or more grip would hold less."""
    p = np.linspace(1e3, 1e6, 500)
    for n in (0.8, 0.9, 0.95):
        mu = pressure_dependent_mu(p, 0.9, 1e5, n)
        assert np.all(np.diff(mu) < 0)
        assert np.all(np.diff(mu * p) > 0)


# --- Shell fiber section (docs/09 §2): plane-stress J2 through the thickness --

from design.oracles import sim3d as S           # noqa: E402

CAN = dict(E=69e9, nu=0.33, sigma_y=285e6, t=1e-4)


def cylindrical_bending(sec, k):
    """Curvature about one axis with the other in-plane strain held at zero (long cylinder)."""
    return sec.update(np.zeros((2, 2)), np.diag([k, 0.0]))


@law("MOD-fiber-section", "elastic_bending_stiffness", "09-simulator.md")
def test_section_elastic_bending_is_exact():
    sec = S.FiberSection(**CAN)
    k = 1e-3
    _, M, _ = cylindrical_bending(sec, k)
    Ep = CAN["E"] / (1 - CAN["nu"] ** 2)
    assert M[0, 0] == pytest.approx(Ep * CAN["t"] ** 3 / 12 * k, rel=1e-12)


@law("MOD-fiber-section", "first_yield_at_surface", "09-simulator.md")
def test_section_first_yield_is_at_the_surface_fiber():
    s1y = CAN["sigma_y"] / np.sqrt(1 - CAN["nu"] + CAN["nu"] ** 2)
    k_y = s1y * (1 - CAN["nu"] ** 2) / CAN["E"] / (CAN["t"] / 2)
    below, above = S.FiberSection(**CAN), S.FiberSection(**CAN)
    cylindrical_bending(below, 0.999 * k_y)
    cylindrical_bending(above, 1.001 * k_y)
    assert np.all(below.alpha == 0)
    assert above.alpha[0] > 0 and above.alpha[-1] > 0 and np.all(above.alpha[1:-1] == 0)


@law("MOD-fiber-section", "plastic_moment_limit", "09-simulator.md")
def test_section_moment_saturates_at_fully_plastic_value():
    sec = S.FiberSection(**CAN)
    for k in np.linspace(0, 8000.0, 400):     # ~100x the first-yield curvature (83 /m)
        _, M, _ = cylindrical_bending(sec, k)
    Mp = 2 * CAN["sigma_y"] / np.sqrt(3) * CAN["t"] ** 2 / 4
    assert M[0, 0] == pytest.approx(Mp, rel=0.005)


def _random_path(seed, n=300):
    rng = np.random.default_rng(seed)
    for P in np.cumsum(rng.normal(scale=20.0, size=(n, 2, 2)), axis=0):   # curvature, 1/m
        kappa = 0.5 * (P + P.T)                                          # wanders well past yield
        yield 1e-5 * kappa, kappa


@law("MOD-fiber-section", "dissipation_nonneg", "09-simulator.md")
def test_section_dissipation_is_non_negative():
    sec = S.FiberSection(**CAN)
    total = 0.0
    for eps_m, kappa in _random_path(3):
        _, _, diss = sec.update(eps_m, kappa)
        assert diss >= -1e-12
        total += diss
    assert total > 0


@law("MOD-fiber-section", "yield_consistency", "09-simulator.md")
def test_section_stress_stays_inside_yield_surface():
    sec = S.FiberSection(**CAN)
    for eps_m, kappa in _random_path(4):
        sec.update(eps_m, kappa)
        assert sec.max_yield_violation(eps_m, kappa) <= 1e-6 * CAN["sigma_y"]


@law("MOD-staggered-plasticity", "return_map_lowers_energy", "09-simulator.md")
def test_return_map_never_raises_stored_energy():
    sec = S.FiberSection(**CAN)
    for eps_m, kappa in _random_path(5, 200):
        before = sec.stored_energy(eps_m, kappa)
        sec.update(eps_m, kappa)
        assert sec.stored_energy(eps_m, kappa) <= before * (1 + 1e-12) + 1e-18


def uv_sphere(n=12):
    """A closed, outward-oriented triangulated unit sphere."""
    th = np.linspace(0, np.pi, n + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    m = len(ph)
    verts = [[0, 0, 1]] + [[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
                           for a in th for b in ph] + [[0, 0, -1]]
    faces = [[0, 1 + j, 1 + (j + 1) % m] for j in range(m)]
    for i in range(len(th) - 1):
        for j in range(m):
            a, b = 1 + i * m + j, 1 + i * m + (j + 1) % m
            faces += [[a, a + m, b], [b, a + m, b + m]]
    last, base = len(verts) - 1, 1 + (len(th) - 1) * m
    faces += [[last, base + (j + 1) % m, base + j] for j in range(m)]
    return np.array(verts, float), np.array(faces)


@law("MOD-gas-3d", "potential_consistency", "09-simulator.md")
def test_gas_force_is_minus_energy_gradient():
    x, faces = uv_sphere()
    x = x * 0.03 + RNG.normal(scale=1e-3, size=x.shape)
    assert S.mesh_volume(x, faces) > 0
    nRT, h = 50.0, 1e-8
    f = S.gas_force(x, faces, nRT)
    for v in RNG.choice(len(x), 5, replace=False):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[v, k] += h
            xm[v, k] -= h
            fd = -(S.gas_energy(xp, faces, nRT) - S.gas_energy(xm, faces, nRT)) / (2 * h)
            assert fd == pytest.approx(f[v, k], rel=1e-5, abs=1e-6 * np.abs(f).max())
