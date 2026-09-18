"""Every number stated in the docs follows from its stated inputs."""

import pytest

from design import quantities
from tests.design.conftest import CLAIMS, ids

QUANTITIES = CLAIMS["quantities"]


@pytest.mark.parametrize("q", QUANTITIES, ids=ids(QUANTITIES))
def test_stated_number_recomputes(q, tag):
    tag(q["id"], q["doc"], "stated_number")
    value = getattr(quantities, q["fn"])(**q["inputs"])
    if "stated_range" in q:
        lo, hi = q["stated_range"]
        got_lo, got_hi = value if isinstance(value, tuple) else (value, value)
        # the stated range must describe the computed one: overlap, and the
        # midpoints within 10% of the stated span
        assert got_lo <= hi and got_hi >= lo, f"computed {value} disjoint from stated {q['stated_range']}"
        assert abs((got_lo + got_hi) / 2 - (lo + hi) / 2) <= 0.1 * max(hi - lo, abs(hi)), \
            f"computed {value} centred away from stated {q['stated_range']}"
    else:
        value = value[1] if isinstance(value, tuple) else value
        assert value == pytest.approx(q["stated"], rel=q["rel_tol"]), \
            f"computed {value:.6g}, doc states {q['stated']:.6g}"
