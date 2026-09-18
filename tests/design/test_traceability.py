"""The docs and the claims registry say the same thing, and nothing is untested."""

import re

import pytest

from tests.design.conftest import CLAIMS, DOCS, ROOT, doc_text, ids, normalize

QUOTED = [(kind, e) for kind in ("equations", "quantities", "regimes", "empirical", "models")
          for e in CLAIMS[kind]]


@pytest.mark.parametrize("kind,entry", QUOTED, ids=[f"{k}:{e['id']}" for k, e in QUOTED])
def test_claim_is_quoted_in_its_doc(kind, entry, tag):
    tag(f"TRACE:{entry['id']}", entry["doc"], "quote_present")
    assert normalize(entry["quote"]) in doc_text(entry["doc"]), \
        f"{entry['id']}: quote not found in docs/{entry['doc']}"


EMPIRICAL = CLAIMS["empirical"]


@pytest.mark.parametrize("e", EMPIRICAL, ids=ids(EMPIRICAL))
def test_empirical_claim_has_validation_plan(e, tag):
    tag(f"PLAN:{e['id']}", e["doc"], "empirical_has_plan")
    assert normalize(e["plan"]) in doc_text(e["plan_doc"])


def _tagged_lines():
    out = []
    for path in sorted(DOCS.glob("*.md")):
        if path.name == "08-design-validation.md":      # describes the tags, uses none
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.search(r"\[(calibrate|measure)[^\]]*\]", line) and "Each one marked" not in line:
                out.append((path.name, normalize(line)))
    return out


def test_every_empirical_tag_is_registered(tag):
    tag("TRACE:empirical-tags-registered", "05-case-study-soda-can.md", "empirical_tagged")
    registered = [(e["doc"], normalize(e["quote"])) for e in EMPIRICAL]
    missing = [(doc, line) for doc, line in _tagged_lines()
               if not any(doc == d and q in line for d, q in registered)]
    assert not missing, f"untracked empirical values: {missing}"


SHARED = CLAIMS["shared_params"]


@pytest.mark.parametrize("p", SHARED, ids=ids(SHARED))
def test_shared_parameter_agrees_across_docs(p, tag):
    tag(p["id"], ",".join(m["doc"] for m in p["mentions"]), "cross_doc_consistency")
    lo, hi = -float("inf"), float("inf")
    for m in p["mentions"]:
        assert normalize(m["quote"]) in doc_text(m["doc"]), f"{m['quote']!r} not in {m['doc']}"
        a, b = m.get("range", [m.get("value")] * 2)
        lo, hi = max(lo, a * 0.99), min(hi, b * 1.01)
    assert lo <= hi, f"{p['id']}: docs disagree"


def test_every_model_law_pair_has_a_test(tag):
    tag("TRACE:model-law-coverage", "08-design-validation.md", "law_coverage")
    source = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests" / "design").glob("test_*.py"))
    found = set(re.findall(r'law\("([^"]+)",\s*"([^"]+)"', source))
    required = {(m["id"], l) for m in CLAIMS["models"] for l in m["laws"]}
    assert not required - found, f"model laws with no test: {sorted(required - found)}"
    assert not found - required, f"tests for laws not registered: {sorted(found - required)}"


LINKS = [(p.name, m) for p in sorted(DOCS.glob("*.md"))
         for m in re.findall(r"\]\(([^)#]+\.md)", p.read_text(encoding="utf-8"))]


@pytest.mark.parametrize("doc,target", LINKS, ids=[f"{d}->{t}" for d, t in LINKS])
def test_doc_links_resolve(doc, target, tag):
    tag(f"LINK:{doc}->{target}", doc, "link_resolves")
    assert (DOCS / target).exists()
