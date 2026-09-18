"""Irreversible-state ledger (docs/03 §4): has anything permanent changed?

Two verdicts, as in the design: equivalent plastic strain (the cheap per-step
proxy) and residual shape after unloading (the ground truth, measured after
rigid alignment so that a can that merely moved is not "altered").
"""

from __future__ import annotations

import numpy as np

from .bodies import Shell

PLASTIC_STRAIN_THRESHOLD = 2e-3      # docs/03 §4: "dent likely visible/tactile"
RESIDUAL_THRESHOLD = 1e-4            # docs/03 §4: 0.1 mm


def rigid_align(x, rest):
    """Best rigid fit of x onto rest (Kabsch); returns x expressed in rest's frame."""
    cx, cr = x.mean(0), rest.mean(0)
    U, _, Vt = np.linalg.svd((x - cx).T @ (rest - cr))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return (x - cx) @ R + cr


def shell_report(scene, name):
    b = scene.by_name[name]
    assert isinstance(b, Shell)
    alpha = b.face_plastic_strain()
    x = rigid_align(scene.body_verts(b), b.rest)
    residual = np.linalg.norm(x - b.rest, axis=1)
    return {
        "body": name,
        "max_plastic_strain": float(alpha.max()),
        "plastic_area_mm2": float(b.area[alpha > 0].sum() * 1e6),
        "max_shape_change_mm": float(residual.max() * 1e3),
        "altered_by_plastic_strain": bool(alpha.max() > PLASTIC_STRAIN_THRESHOLD),
        "altered_by_shape": bool(residual.max() > RESIDUAL_THRESHOLD),
    }


def report(scene):
    return [shell_report(scene, b.name) for b in scene.bodies if isinstance(b, Shell)]
