"""CMP-sim.analytic: the long-pad squeeze tier (docs/11)."""

import time

import numpy as np
import pytest

from design.oracles import elastica as EL
from simphys.analytic import can_wall, plot_curve

M = can_wall()


def test_force_matches_the_quadrature_reference_in_both_phases():
    for g in (0.064, 0.055, 0.049, 0.045, 0.030, 0.022):     # line contact, then flat contact
        assert M.force(g) == pytest.approx(EL.force_for_gap(g, M.R, M.EI), rel=1e-6)


def test_first_yield_and_flat_onset_match_the_design():
    assert M.gap_yield == pytest.approx(0.0211, rel=0.003)
    assert M.force_yield == pytest.approx(83.4, rel=0.002)
    assert M.gap_flat == pytest.approx(0.0474, rel=0.002)
    assert M.altered(0.020) and not M.altered(0.025)


def test_force_rises_monotonically_as_the_pads_close():
    g = np.linspace(0.021, 0.0659, 300)
    assert np.all(np.diff(M.force(g)) < 0)


def test_a_whole_curve_is_fast():
    g = np.linspace(0.022, 0.0659, 200)
    M.force(g)
    t0 = time.perf_counter()
    M.force(g)
    assert time.perf_counter() - t0 < 1.0


def test_curve_renders_to_png(tmp_path):
    from PIL import Image
    path = plot_curve(tmp_path / "curve.png")
    with Image.open(path) as im:
        assert im.size[0] > 400


def test_slope_matches_finite_differences_and_is_continuous():
    for g in (0.060, 0.050, 0.040, 0.025):
        h = 1e-7
        fd = (M.force(g + h) - M.force(g - h)) / (2 * h)
        assert M.slope(g) == pytest.approx(fd, rel=1e-5)
    g0 = M.gap_flat
    assert M.slope(g0 * (1 + 1e-9)) == pytest.approx(M.slope(g0 * (1 - 1e-9)), rel=1e-3)


def test_single_query_is_microseconds():
    M.force(0.055)
    t0 = time.perf_counter()
    for _ in range(1000):
        M.force(0.055)
    assert (time.perf_counter() - t0) / 1000 < 1e-4


def _perimeter(curve):
    return np.linalg.norm(np.diff(np.vstack([curve, curve[:1]]), axis=0), axis=1).sum()


@pytest.mark.parametrize("gap", [0.062, 0.050, 0.040, 0.025])
def test_profile_keeps_perimeter_and_touches_both_pads(gap):
    c = M.profile(gap)
    assert _perimeter(c) == pytest.approx(2 * np.pi * M.R, rel=2e-4)
    assert c[:, 0].min() == pytest.approx(-gap / 2, abs=2e-6 * M.R)
    assert c[:, 0].max() == pytest.approx(gap / 2, abs=2e-6 * M.R)


def test_undeformed_limit_is_the_circle():
    c = M.profile(2 * M.R * (1 - 1e-6))
    r = np.linalg.norm(c, axis=1)
    assert np.abs(r - M.R).max() < 1e-4 * M.R
