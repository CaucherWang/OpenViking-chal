import importlib
import json
from pathlib import Path

import pytest

ARCHIVE = "/sessions/policy/history/"
ARCHIVE_URI = f"viking://user/default{ARCHIVE}archive_001/messages.jsonl"


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmark/rules_compliance"))
    return importlib.import_module("benchmark.rules_compliance.evaluate")


def test_no_matches_and_failed_tools_do_not_recall_query_ids(evaluator):
    tools = [
        {"tool_name": "openviking_grep", "result": "No matches found for pattern: 'R01|R40'"},
        {
            "tool_name": "openviking_multi_read",
            "execute_success": False,
            "result": "Error reading R02",
        },
    ]
    assert evaluator.tool_diagnostics(tools, ARCHIVE) == (set(), 0, set(), False)


def test_listing_archive_is_discovery_without_rule_recall(evaluator):
    tools = [{"tool_name": "openviking_list", "result": ARCHIVE_URI}]
    assert evaluator.tool_diagnostics(tools, ARCHIVE) == (set(), 0, set(), True)


def test_archive_read_counts_only_returned_ids(evaluator):
    tools = [
        {
            "tool_name": "openviking_multi_read",
            "resolved_args": {"uris": [ARCHIVE_URI]},
            "result": "R01: preserve notices\n[... truncated ...]\nR40: isolated implementation",
        }
    ]
    exposed, calls, archive_ids, discovered = evaluator.tool_diagnostics(tools, ARCHIVE)
    assert exposed == archive_ids == {"R01", "R40"}
    assert calls == 0
    assert discovered


def test_reading_exported_archive_adds_rule_ids(evaluator):
    exported_path = "policy_export/policy/history/archive_001/messages.jsonl"
    tools = [
        {
            "tool_name": "openviking_export",
            "resolved_args": {"uri": ARCHIVE_URI, "dest": "policy_export"},
            "result": f"Exported 1 file:\n- {exported_path}  (12000 bytes)",
        },
        {
            "tool_name": "exec",
            "resolved_args": {"command": f"read {exported_path}"},
            "result": "R17: publish library modifications",
        },
        {
            "tool_name": "read_file",
            "resolved_args": {"path": exported_path},
            "result": "R18: retain file license",
        },
        {
            "tool_name": "exec",
            "resolved_args": {"command": "inspect repository"},
            "result": "R01 is an unrelated repository label",
        },
    ]
    exposed, _, archive_ids, _ = evaluator.tool_diagnostics(tools, ARCHIVE)
    assert exposed == archive_ids == {"R17", "R18"}


def test_rule_search_is_counted_and_exposes_rules(evaluator):
    tools = [{"tool_name": "openviking_rule_search", "result": "R01: include LICENSE"}]
    assert evaluator.tool_diagnostics(tools) == ({"R01"}, 1, set(), False)


def test_archive_recall_survives_empty_memory_inventory(evaluator, monkeypatch, tmp_path):
    truth = tmp_path / "truth.json"
    manifest = tmp_path / "manifest.json"
    evaluator.write_json(
        truth,
        {"repositories": [{"name": "repo", "applicable_rule_ids": ["R01", "R02", "R03"]}]},
    )
    evaluator.write_json(manifest, {"repositories": [{"name": "repo", "full_name": "org/repo"}]})
    monkeypatch.setattr(evaluator, "GROUND_TRUTH", truth)
    monkeypatch.setattr(evaluator, "MANIFEST", manifest)
    evaluator.write_json(tmp_path / "runtime/main_memory_inventory.json", {"rule_ids": []})
    evaluator.write_json(
        tmp_path / "runtime/offline_stage.json", {"main": {"session_id": "policy"}}
    )
    archive_path = (
        tmp_path
        / "runtime/main/storage/viking/default/user/default/sessions/policy/history"
        / "archive_001/messages.jsonl"
    )
    archive_path.parent.mkdir(parents=True)
    archive_path.write_text(
        json.dumps({"role": "user", "parts": [{"text": "R01 R02 R03 R40"}]}) + "\n"
    )
    evaluator.write_json(
        tmp_path / "primary/repo/main/result.json",
        {
            "completed": True,
            "submission": {"items": [{"rule_id": "R01"}]},
            "tools_used": [
                {
                    "tool_name": "openviking_multi_read",
                    "resolved_args": {"uris": [ARCHIVE_URI]},
                    "result": "R01: license\nR02: consistency\nR40: isolation",
                }
            ],
            "wall_seconds": 1.0,
        },
    )
    summary = evaluator.evaluate(tmp_path, "primary", False, ["main"], "corrected")
    arm = summary["arms"]["main"]
    assert arm["offline_stored_rule_ids"] == []
    assert arm["offline_archive_rule_ids"] == ["R01", "R02", "R03", "R40"]
    assert arm["micro_recall"] == pytest.approx(2 / 3)
    assert arm["micro_precision"] == pytest.approx(2 / 3)
    assert arm["false_negative_unavailable_offline"] == 0
    assert arm["false_negative_available_not_exposed"] == 1
    assert arm["session_archive_access_true_positive"] == 2
    assert arm["submission_diagnostics"]["micro_recall"] == pytest.approx(1 / 3)
    row = evaluator.read_json(tmp_path / "corrected/runs.json")[0]
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["expected_exposed_not_submitted"] == ["R02"]
    assert not (tmp_path / "evaluation").exists()
