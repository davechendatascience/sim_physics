"""Adapter: run pytest, emit one trial per test case to $OUT.

Adapted from Theoretically_Driven_LLM_Planning/tools/pytest_trials.py. Adds the
per-case properties recorded by tests/design (claim, doc, law) as trial
conditions, so the belief server can identify a trial by the design claim it
checks and not count a deterministic re-run as new evidence.

Usage (from a belief.yaml `run:` line):
    python tools/design_trials.py $OUT -- tests/design/test_dimensional.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def project_python() -> str:
    """The project venv's interpreter if it exists, else the current one, so a
    belief run uses the dependencies in requirements.txt whichever `python`
    the server happened to launch this script with."""
    root = Path(__file__).resolve().parents[1]
    for cand in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return sys.executable


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: design_trials.py <out.json> [-- <pytest args>]", file=sys.stderr)
        return 2

    out_path = Path(argv[0])
    pytest_args = argv[2:] if len(argv) > 1 and argv[1] == "--" else argv[1:]

    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        completed = subprocess.run(
            [project_python(), "-m", "pytest", f"--junit-xml={junit}",
             "-o", "junit_family=xunit1", "-p", "no:cacheprovider", *pytest_args],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        trials = _parse(junit) if junit.exists() else []

    # No parseable cases: emit an empty list rather than a synthetic pass.
    out_path.write_text(json.dumps({"trials": trials}, indent=2), encoding="utf-8")
    return completed.returncode


def _parse(junit: Path) -> list[dict]:
    try:
        root = ET.parse(junit).getroot()
    except ET.ParseError:
        return []

    trials: list[dict] = []
    for case in root.iter("testcase"):
        failed = any(case.find(tag) is not None for tag in ("failure", "error"))
        skipped = case.find("skipped") is not None
        name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
        conditions = {"case": name}
        for prop in case.iter("property"):
            conditions[prop.get("name")] = prop.get("value")
        conditions.setdefault("claim", name)
        trials.append({
            "metrics": {"passed": not failed and not skipped},
            "conditions": conditions,
            "outcome": "not_applicable" if skipped else ("fail" if failed else "pass"),
        })
    return trials


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
