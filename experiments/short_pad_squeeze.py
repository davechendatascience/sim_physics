"""Offline short-pad squeeze runs (brief rev. 3 §7.2-7.3 validation data).

Runs the 3D can squeeze (docs/09 §5) with a given pad size and depth, saving
snapshots for (a) testing the mirror-isometry readout against the 3D shape
and (b) seeding a reduced-order model later.

    python experiments/short_pad_squeeze.py NAME PAD_WIDTH_MM PAD_LENGTH_MM DEPTH_MM
Writes out/runs/NAME.npz (snapshots) and out/runs/NAME.log (progress).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simphys import ledger                      # noqa: E402
from simphys.scenes import can_squeeze          # noqa: E402


def main(name, width_mm, length_mm, depth_mm, every=2):
    out = Path("out/runs")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"{name}.log", "w", buffering=1)
    s, info = can_squeeze(depth=depth_mm * 1e-3, pad_width=width_mm * 1e-3, pad_length=length_mm * 1e-3)
    can = s.by_name["can"]
    snaps = {k: [] for k in ("t", "travel", "force_left", "force_right", "x", "alpha", "newton")}
    t0 = time.time()
    for i in range(info["steps"]):
        t1 = time.time()
        n = s.step()
        fl = s.stats["drive_force"]["pad_left"][-1]
        fr = s.stats["drive_force"]["pad_right"][-1]
        log.write(f"step {i + 1:4d}/{info['steps']}  t {s.t:.3f}  travel {info['travel'](s.t) * 1e3:5.2f} mm  "
                  f"force {abs(fr[0]):8.3f} N  alpha_max {can.alpha.max():.2e}  newton {n:3d}  {time.time() - t1:5.1f}s\n")
        if i % every == 0 or i == info["steps"] - 1:
            snaps["t"].append(s.t)
            snaps["travel"].append(info["travel"](s.t))
            snaps["force_left"].append(fl)
            snaps["force_right"].append(fr)
            snaps["x"].append(can.x.copy())
            snaps["alpha"].append(can.face_plastic_strain().copy())
            snaps["newton"].append(n)
    rep = ledger.shell_report(s, "can")
    log.write(f"done in {time.time() - t0:.0f}s; ledger {rep}\n")
    np.savez_compressed(out / f"{name}.npz", rest=can.rest, faces=can.faces,
                        pad_width=width_mm * 1e-3, pad_length=length_mm * 1e-3, depth=depth_mm * 1e-3,
                        **{k: np.array(v) for k, v in snaps.items()})
    log.close()


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
