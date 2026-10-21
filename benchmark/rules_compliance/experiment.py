"""Shared constants for the fixed Top20 rule-recall experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / ".openviking/experiments/rules-compliance-20261016"
DEFAULT_OUTPUT = ROOT / ".openviking/experiments/rules-memory-vs-compiled-20261019"
MANIFEST = DATASET / "repositories/manifest.json"
POLICY = DATASET / "sources/open_source_compliance_rules.md"
GROUND_TRUTH = ROOT / "benchmark/rules_compliance/ground_truth.json"
PORTS = {"main": 2233, "rules": 2333}
RULE_IDS = tuple(f"R{index:02d}" for index in range(1, 41))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
