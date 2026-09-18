"""Load design/claims.yaml with YAML 1.2 float rules (PyYAML's 1.1 rules read 80.0e6 as a string)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CLAIMS_PATH = Path(__file__).with_name("claims.yaml")


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$|^[-+]?\d+\.\d*$"),
    list("-+0123456789."),
)


def load_claims(path: Path = CLAIMS_PATH) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
