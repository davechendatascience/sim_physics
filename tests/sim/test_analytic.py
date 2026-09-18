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
