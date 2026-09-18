"""CMP-sim end to end: the command line runs every built-in scene headless."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from simphys.scenes import SCENES

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.run([sys.executable, "-m", "simphys", *args], cwd=ROOT, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, timeout=600)


def test_list_names_every_scene():
    out = run("list")
    assert out.returncode == 0
    for name in SCENES:
        assert name in out.stdout


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_scene_runs_headless_and_writes_outputs(scene, tmp_path):
    gif = tmp_path / f"{scene}.gif"
    out = run("run", scene, "--headless", "--steps", "3", "--every", "1", "--out", str(gif))
    assert out.returncode == 0, out.stderr[-2000:]
    assert gif.exists() and gif.with_suffix(".png").exists()
    summary = json.loads(gif.with_suffix(".json").read_text())
    assert summary["steps"] == 3


def test_unknown_scene_fails_cleanly():
    out = run("run", "no_such_scene", "--headless")
    assert out.returncode == 2 and "unknown scene" in out.stderr
