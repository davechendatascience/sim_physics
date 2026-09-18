"""Shared helpers for the design-validation checks.

Each test case is one design claim. `tag` records the claim id, source doc and
law as JUnit properties; tools/design_trials.py turns them into trial
conditions for the component-belief server.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from design.registry import load_claims

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CLAIMS = load_claims()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("`", "")).strip()


def doc_text(name: str) -> str:
    return normalize((DOCS / name).read_text(encoding="utf-8"))


def ids(entries):
    return [e["id"] for e in entries]


@pytest.fixture
def tag(record_property):
    def _tag(claim: str, doc: str = "", law: str = ""):
        record_property("claim", claim)
        record_property("doc", doc)
        record_property("law", law)
    return _tag


def law(model: str, law_id: str, doc: str):
    """Mark a law test; also parsed by the traceability coverage check."""
    return pytest.mark.law(model=model, law=law_id, doc=doc)


@pytest.fixture(autouse=True)
def _tag_law(request, record_property):
    marker = request.node.get_closest_marker("law")
    if marker:
        record_property("claim", f"{marker.kwargs['model']}:{marker.kwargs['law']}")
        record_property("doc", marker.kwargs["doc"])
        record_property("law", marker.kwargs["law"])


def pytest_configure(config):
    config.addinivalue_line("markers", "law(model, law, doc): design law check")
