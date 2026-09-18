"""Recompute the numbers and regime ratios stated in the design.

Inputs given as [lo, hi] are parameter ranges; functions with a `_range`
suffix return the (min, max) of the result over the box of input ranges.
"""

from __future__ import annotations

import itertools

import numpy as np

G = 9.81
R_GAS = 8.314462618


def _corners(inputs: dict):
    """Evaluate over every corner of the input box (monotone formulas)."""
    keys = list(inputs)
    axes = [v if isinstance(v, list) else [v] for v in inputs.values()]
    for combo in itertools.product(*axes):
        yield dict(zip(keys, combo))


def _span(fn, inputs):
    vals = [fn(**c) for c in _corners(inputs)]
    return min(vals), max(vals)


# --- point quantities ----------------------------------------------------

def hoop_stress(p, R, t):
    return p * R / t


def bending_length(R, t, nu):
    return np.sqrt(R * t) / (3 * (1 - nu ** 2)) ** 0.25


def dent_scale(sy, t, R):
    return sy * t ** 1.5 * R ** 0.5


def fmin(m, a, mu, g=G):
    return m * (g + a) / (2 * mu)


def ratio(num, den):
    return num / den


def full_can_mass(volume, rho_liquid, m_empty):
    return _span(lambda volume, rho_liquid, m_empty: volume * rho_liquid + m_empty,
                 {"volume": volume, "rho_liquid": rho_liquid, "m_empty": m_empty})


def wave_transit(E, rho, L):
    return L / np.sqrt(E / rho)


def grain_count(t, grain):
    return _span(lambda t, grain: t / grain, {"t": t, "grain": grain})


def _rate_shift(m_rate, rate_lo, rate_hi):
    return m_rate * np.log(rate_hi / rate_lo)


def rate_shift(m_rate, rate_lo, rate_hi):
    """Worst-case relative flow-stress change over the strain-rate span."""
    return _span(_rate_shift, {"m_rate": m_rate, "rate_lo": rate_lo, "rate_hi": rate_hi})[1]


def _z(p_abs, T, B):
    """Compressibility factor from the second virial coefficient."""
    return 1 + B * p_abs / (R_GAS * T)


def co2_z(p_abs, T, B):
    lo, hi = _span(_z, {"p_abs": p_abs, "T": T, "B": B})
    return 0.5 * (lo + hi)


def tilt_deg(a, g=G):
    return float(np.degrees(np.arctan(a / g)))


def _hertz_line(F, width, E_pad, nu_pad, R):
    """Contact half-width / R for a soft flat pad on a rigid cylinder (line contact)."""
    E_star = E_pad / (1 - nu_pad ** 2)
    P = F / width
    a = np.sqrt(4 * P * R / (np.pi * E_star))
    return a / R


def hertz_line_ratio(**kw):
    return _hertz_line(**kw)


# --- ranges for regime checks ---------------------------------------------

def ratio_range(num, den):
    return _span(lambda num, den: num / den, {"num": num, "den": den})


def rate_shift_range(m_rate, rate_lo, rate_hi):
    return _span(_rate_shift, {"m_rate": m_rate, "rate_lo": rate_lo, "rate_hi": rate_hi})


def co2_deviation_range(p_abs, T, B):
    return _span(lambda p_abs, T, B: abs(1 - _z(p_abs, T, B)), {"p_abs": p_abs, "T": T, "B": B})


def hertz_sphere_ratio_range(F, R, E1, nu1, E2, nu2):
    def f(F, R, E1, nu1, E2, nu2):
        E_star = 1 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)
        a = (3 * F * R / (4 * E_star)) ** (1 / 3)
        return a / R
    return _span(f, {"F": F, "R": R, "E1": E1, "nu1": nu1, "E2": E2, "nu2": nu2})


def hertz_line_ratio_range(**kw):
    return _span(_hertz_line, kw)


def pad_layer_ratio_range(F, width, E_pad, nu_pad, R, h_pad):
    """Pad thickness / contact half-width. Hertz's half-space needs this >> 1 (about 10)."""
    def f(F, width, E_pad, nu_pad, R, h_pad):
        return h_pad / (_hertz_line(F, width, E_pad, nu_pad, R) * R)
    return _span(f, {"F": F, "width": width, "E_pad": E_pad, "nu_pad": nu_pad, "R": R, "h_pad": h_pad})


# --- measured on the reference model ----------------------------------------

def _grasp_slide_speed(k, round_fingertips, m=0.38, mu=0.8, T=0.3, h=0.005):
    """Can held by two fingers pushed with k * F_min each; returns slide speed after T."""
    from design.oracles.integrator import Plane, System, step

    F = k * m * G / (2 * mu)
    r, gap = 0.033, 1e-4
    if round_fingertips:
        rf = 0.01
        x = np.array([[0, 0], [-(r + rf + gap), 0], [r + rf + gap, 0]], float)
        radii, planes, pair_mu = [r, rf, rf], [], mu
    else:
        off = 0.02
        x = np.array([[0, 0], [-(r + gap + off), 0], [r + gap + off, 0]], float)
        radii, pair_mu = [r, 0.001, 0.001], 0.0
        planes = [Plane(np.array([off, 0.0]), np.array([1.0, 0.0]), mu, owner=1),
                  Plane(np.array([-off, 0.0]), np.array([-1.0, 0.0]), mu, owner=2)]
    s = System(x=x, v=np.zeros((3, 2)), m=np.array([m, 0.2, 0.2]), r=np.array(radii),
               planes=planes, pair_mu=pair_mu,
               f_ext=np.array([[0, 0], [F, 0], [-F, 0]], float),
               anchors=[(1, 1, 0.0, 1e7), (2, 1, 0.0, 1e7)], kappa=1e3, dhat=1e-3)
    for _ in range(40):              # squeeze first, then lift: grip settles before load
        step(s, h)
    s.v[:] = 0
    s.gravity = np.array([0.0, -G])
    for _ in range(int(T / h)):
        step(s, h)
    predicted = max(0.0, G - 2 * mu * F / m) * T
    return -s.v[0, 1], predicted


def oracle_round_fingertip_ratio(k):
    got, predicted = _grasp_slide_speed(k, round_fingertips=True)
    return got / predicted


def oracle_flat_pad_ratio(k):
    got, predicted = _grasp_slide_speed(k, round_fingertips=False)
    return got / predicted


# --- long-cylinder limit of the 3D shell (docs/09 §4), per unit length ----------

def plane_strain_modulus(E, nu):
    return E / (1 - nu ** 2)


def plane_strain_yield(sy):
    """Fully plastic in-plane fiber stress with the axial strain held at zero."""
    return 2 * sy / np.sqrt(3)


def first_yield_stress(sy, nu):
    """Elastic fiber stress at first yield with the axial strain held at zero
    (sigma_22 = nu sigma_11 while elastic)."""
    return sy / np.sqrt(1 - nu + nu ** 2)


def _ring(E, nu, t):
    return E / (1 - nu ** 2), t ** 3 / 12


def ring_compliance(E, nu, sy, t, R):
    Ep, I = _ring(E, nu, t)
    return (np.pi / 4 - 2 / np.pi) * R ** 3 / (Ep * I)


def ring_first_yield(E, nu, sy, t, R):
    return np.pi * (first_yield_stress(sy, nu) * t ** 2 / 6) / R


def ring_collapse(E, nu, sy, t, R):
    return 4 * (plane_strain_yield(sy) * t ** 2 / 4) / R


def ring_buckling_pressure(E, nu, sy, t, R):
    Ep, I = _ring(E, nu, t)
    return 3 * Ep * I / R ** 3


def ring_compliance_pressurized(E, nu, sy, t, R, p, n_max=20000):
    """Diametral compliance of an inextensible ring under internal pressure: a sum
    over even Fourier modes n, each stiffened by the follower pressure load."""
    Ep, I = _ring(E, nu, t)
    beta = p * R ** 3 / (Ep * I)
    n = np.arange(2, n_max + 1, 2)
    return 4 / np.pi * R ** 3 / (Ep * I) * np.sum(1 / ((n ** 2 - 1) * ((n ** 2 - 1) + beta)))


def ring_pressure_stiffening(E, nu, sy, t, R, p):
    return ring_compliance_pressurized(E, nu, sy, t, R, 0.0) / ring_compliance_pressurized(E, nu, sy, t, R, p)


def ring_pressure_ratio(E, nu, sy, t, R, p):
    return p / ring_buckling_pressure(E, nu, sy, t, R)


def rule_plastic_moment_deficit(rule_name):
    """1 - (quadrature of |xi|) / (exact integral of |xi| over [-1, 1], which is 1)."""
    from design.oracles.sim3d import rule
    xi, w = rule(rule_name)
    return 1 - float(np.sum(w * np.abs(xi)))


def ring_deflection_ratio(F, E, nu, t, R):
    """Small-deflection diametral deflection / R at line load F (N/m)."""
    return ring_compliance(E, nu, 0.0, t, R) * F / R


def ring_deflection_ratio_range(F, E, nu, t, R):
    return _span(ring_deflection_ratio, {"F": F, "E": E, "nu": nu, "t": t, "R": R})


def yield_curvature_change(E, nu, sy, t):
    """Curvature change that brings the surface fiber to first yield in
    cylindrical bending (axial strain held at zero)."""
    eps_y = first_yield_stress(sy, nu) * (1 - nu ** 2) / E
    return 2 * eps_y / t


def flat_plate_gap_at_yield(E, nu, sy, t, R):
    """Plate gap at which the ends of a ring flattened between plates (taken as
    semicircles of diameter = gap) reach first yield."""
    return 2 / (yield_curvature_change(E, nu, sy, t) + 1 / R)


def small_deflection_load(E, nu, t, R, frac):
    """Line load giving a diametral deflection of frac * R."""
    return frac * R / ring_compliance(E, nu, 0.0, t, R)
