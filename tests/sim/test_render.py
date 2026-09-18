"""CMP-sim.render: headless and windowed rendering (docs/09 §3)."""

import os

import pytest
from PIL import Image

from simphys.render import Renderer
from tests.sim.helpers import box, ground
from simphys.scene import Scene


def test_headless_writes_an_animated_gif_and_a_png(tmp_path):
    s = Scene([ground(), box("b", center=(0, 0, 0.1))], dt=0.01)
    r = Renderer(s, "headless", title="test")
    for _ in range(4):
        s.step()
        r.draw()
    gif, png = r.save(tmp_path / "t.gif")
    r.close()
    with Image.open(gif) as im:
        assert im.n_frames == 4
    with Image.open(png) as im:
        assert im.size[0] > 100 and im.size[1] > 100


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Renderer(Scene([box("b")], dt=0.01), "opengl")


@pytest.mark.skipif(os.environ.get("SIMPHYS_TEST_WINDOW") != "1",
                    reason="opens a window; set SIMPHYS_TEST_WINDOW=1 to run")
def test_window_mode_draws_live():
    s = Scene([ground(), box("b", center=(0, 0, 0.1))], dt=0.01)
    r = Renderer(s, "window")
    for _ in range(3):
        s.step()
        r.draw()
    assert r.frames == []                                # nothing buffered: drawn to the screen
    r.plt.close(r.fig)
