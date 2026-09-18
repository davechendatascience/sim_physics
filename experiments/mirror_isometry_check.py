"""Test the mirror-isometry readout (brief rev. 3 §7.2) against 3D squeeze runs.

For each saved snapshot (loading phase only) with pad travel d, the mirror
model says: the wall beyond the pad plane x = R - d (right pad) is reflected
back across it, the rest is undeformed, and a ridge of width ~sqrt(R t)
carries the bending. Two checks:

1. Shape: in the pad footprint, outside the ridge, compare the simulated
   radial position of the wall with the reflected one, after removing the
   global ovalization of the can (measured at 90 degrees from the pads).
2. Force law: Pogorelov's U ~ d^(3/2) implies F ~ d^(1/2); fit the log-log
   slope of force against travel.

    python experiments/mirror_isometry_check.py out/runs/A_15x20_d6.npz ...
"""

from __future__ import annotations

import sys

import numpy as np

R, T = 0.033, 1e-4
RIDGE = np.sqrt(R * T)


def check(path):
    d = np.load(path)
    rest, xs, travel = d["rest"], d["x"], d["travel"]
    F = np.abs(d["force_right"][:, 0])
    w, L = float(d["pad_width"]), float(d["pad_length"])
    loading = np.r_[True, np.diff(travel) > 0] & (travel > 0)
    side = (rest[:, 0] > 0) & (np.abs(rest[:, 2]) < 0.05)
    # vertices under the right pad's footprint, away from the ridge at its edge
    under = side & (np.abs(rest[:, 1]) < w / 2 - 2 * RIDGE) & (np.abs(rest[:, 2]) < L / 2 - 2 * RIDGE)
    rows = []
    for x, dep in zip(xs[loading], travel[loading]):
        if dep < 5 * RIDGE * 0.1:
            continue
        plane = R - dep
        # global ovalization: how far the whole wall has moved in, measured beside (not under) the pad
        beside = side & (np.abs(rest[:, 1]) > w / 2 + 3 * RIDGE) & (np.abs(rest[:, 1]) < w / 2 + 6 * RIDGE) \
            & (np.abs(rest[:, 2]) < L / 2)
        shift = np.median(rest[beside, 0] - x[beside, 0]) if beside.any() else 0.0
        beyond = under & (rest[:, 0] - shift > plane)
        if beyond.sum() < 3:
            continue
        mirror_x = 2 * plane - (rest[beyond, 0] - shift)
        err = x[beyond, 0] - mirror_x
        rows.append((dep, np.sqrt(np.mean(err ** 2)), np.abs(err).max(), shift, beyond.sum()))
    rows = np.array(rows)
    sel = (travel > 0.5e-3) & loading & (F > 0)
    slope = np.polyfit(np.log(travel[sel]), np.log(F[sel]), 1)[0] if sel.sum() > 3 else np.nan
    print(f"\n{path}: pad {w * 1e3:.0f}x{L * 1e3:.0f} mm, max force {F.max():.2f} N, "
          f"plastic strain at end {d['alpha'][-1].max():.2e}, log-log slope F~d^{slope:.2f} (Pogorelov: 0.50)")
    if len(rows):
        for dep, rms, mx, shift, n in rows[:: max(1, len(rows) // 8)]:
            print(f"  travel {dep * 1e3:5.2f} mm: mirror-shape error rms {rms * 1e3:6.3f} mm, max {mx * 1e3:6.3f} mm "
                  f"(rel. to travel {rms / dep:5.1%}); global inward shift {shift * 1e3:5.2f} mm; {n} vertices")
    else:
        print("  no vertices beyond the pad plane in the footprint yet")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
