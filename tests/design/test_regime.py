"""Every model is used inside the regime where classical physics holds.

Status over the stated parameter range:
  holds     - the whole range is on the valid side of `limit`
  marginal  - part of the range crosses `limit` but not `hard_limit`
  violated  - the whole range is past `limit`, or any part past `hard_limit`
A marginal regime passes only with a documented mitigation; a violated one
passes only if the design explicitly excludes that use.
"""

import pytest

from design import quantities
from tests.design.conftest import CLAIMS, doc_text, ids, normalize

REGIMES = CLAIMS["regimes"]


def status(lo, hi, r):
    limit, sense = r["limit"], r["sense"]
    hard = r.get("hard_limit")
    if sense == "max":          # valid while value <= limit
        if hi <= limit:
            return "holds"
        if lo > limit or (hard is not None and hi > hard):
            return "violated"
    else:                       # valid while value >= limit
        if lo >= limit:
            return "holds"
        if hi < limit or (hard is not None and lo < hard):
            return "violated"
    return "marginal"


@pytest.mark.parametrize("r", REGIMES, ids=ids(REGIMES))
def test_regime_of_validity(r, tag):
    tag(r["id"], r["doc"], "regime_of_validity")
    lo, hi = getattr(quantities, r["fn"])(**r["inputs"])
    st = status(lo, hi, r)
    text = doc_text(r["doc"])
    detail = f"range [{lo:.3g}, {hi:.3g}] vs limit {r['limit']} ({r['sense']}) -> {st}"
    if r.get("expect") == "violated":
        assert st == "violated", f"expected an excluded regime, got {detail}"
        assert normalize(r["exclusion"]) in text, "design uses a violated regime without excluding it"
    elif st == "marginal":
        assert "mitigation" in r and normalize(r["mitigation"]) in text, \
            f"marginal regime without a documented mitigation: {detail}"
    else:
        assert st == "holds", f"classical model used outside its regime: {detail}"
