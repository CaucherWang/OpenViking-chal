"""Evaluate rules exposed to the Agent, including retrieved Session archives."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from experiment import DEFAULT_OUTPUT, GROUND_TRUTH, MANIFEST, RULE_IDS, read_json, write_json

RULE_PATTERN = re.compile(r"\bR(?:0[1-9]|[1-3][0-9]|40)\b")


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def rule_ids_from(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(RULE_PATTERN.findall(value))
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        ids: set[str] = set()
        for item in value:
            ids.update(rule_ids_from(item))
        return ids
    return set()


def tool_diagnostics(
    tools: list[dict[str, Any]],
    archive_marker: str | None = None,
) -> tuple[set[str], int, set[str], bool]:
    exposed: set[str] = set()
    archive_exposed: set[str] = set()
    exported_archive_paths: set[str] = set()
    archive_discovered = False
    rule_search_calls = 0
    for tool in tools:
        name = str(tool.get("tool_name") or tool.get("name") or "")
        if name == "openviking_rule_search":
            rule_search_calls += 1
        if tool.get("execute_success") is False:
            continue
        result = tool.get("result")
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if text.lstrip().startswith(("No matches found", "Error:", "Tool execution failed")):
            continue
        arguments = json.dumps(
            tool.get("resolved_args") or tool.get("args") or {}, ensure_ascii=False
        )
        from_archive = bool(archive_marker and archive_marker in text)
        targets_archive = bool(archive_marker and archive_marker in arguments)
        if name.startswith("openviking_") and (from_archive or targets_archive):
            archive_discovered = True
        if name == "openviking_export" and targets_archive:
            exported_archive_paths.update(
                re.findall(r"^- (.+?)\s+\(\d+ bytes\)", text, flags=re.MULTILINE)
            )
        local_archive_read = name in {"exec", "read_file"} and (
            from_archive or any(path in arguments for path in exported_archive_paths)
        )
        if not name.startswith("openviking_") and not local_archive_read:
            continue
        ids = rule_ids_from(result)
        exposed.update(ids)
        if from_archive or targets_archive or local_archive_read:
            archive_exposed.update(ids)
    return exposed, rule_search_calls, archive_exposed, archive_discovered


def archive_inventory(output: Path, session_id: str | None) -> dict[str, Any]:
    """Inspect archived user messages separately from extracted long-term Memory."""
    paths = (
        sorted(
            (output / "runtime/main/storage").glob(
                f"viking/*/user/*/sessions/{session_id}/history/*/messages.jsonl"
            )
        )
        if session_id
        else []
    )
    ids: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            message = json.loads(line)
            if message.get("role") == "user":
                ids.update(rule_ids_from(message.get("parts")))
    return {
        "session_id": session_id,
        "files": [str(path.relative_to(output)) for path in paths],
        "rule_ids": sorted(ids),
    }


def evaluate(
    output: Path,
    phase: str,
    allow_partial: bool,
    selected_arms: list[str],
    evaluation_subdir: str = "evaluation",
) -> dict[str, Any]:
    truth_document = read_json(GROUND_TRUTH)
    truth = {
        item["name"]: set(item["applicable_rule_ids"]) for item in truth_document["repositories"]
    }
    repositories = read_json(MANIFEST)["repositories"]
    inventory_paths = {
        "main": output / "runtime/main_memory_inventory.json",
        "rules": output / "runtime/rules_inventory.json",
    }
    stored_rules = {arm: set(read_json(inventory_paths[arm])["rule_ids"]) for arm in selected_arms}
    offline_stage = read_json(output / "runtime/offline_stage.json")
    main_session_id = (offline_stage.get("main") or {}).get("session_id")
    archive_marker = f"/sessions/{main_session_id}/history/" if main_session_id else None
    main_archive = archive_inventory(output, main_session_id) if "main" in selected_arms else {}
    archive_ids = set(main_archive.get("rule_ids") or [])
    offline_available = {
        arm: ids | (archive_ids if arm == "main" else set()) for arm, ids in stored_rules.items()
    }
    rows: list[dict[str, Any]] = []

    for repo in repositories:
        expected = truth[repo["name"]]
        for arm in selected_arms:
            run_dir = output / phase / repo["name"] / arm
            result_path = run_dir / "result.json"
            if not result_path.exists():
                if allow_partial:
                    continue
                raise FileNotFoundError(result_path)
            result = read_json(result_path)
            items = (result.get("submission") or {}).get("items") or []
            predicted = {item["rule_id"] for item in items if item.get("rule_id") in set(RULE_IDS)}
            recalled = rule_ids_from(result.get("relevant_memories"))
            tools = result.get("tools_used") or []
            tool_exposed, rule_search_calls, archive_exposed, archive_discovered = tool_diagnostics(
                tools, archive_marker if arm == "main" else None
            )
            raw_exposed = recalled | tool_exposed
            exposed = raw_exposed & set(RULE_IDS)
            true_positive = expected & exposed
            false_positive = exposed - expected
            false_negative = expected - exposed
            submitted_true_positive = expected & predicted
            submitted_false_positive = predicted - expected
            submitted_false_negative = expected - predicted
            unavailable = false_negative - offline_available[arm]
            available_not_exposed = false_negative & offline_available[arm]
            exposed_not_submitted = (expected & exposed) - predicted
            rows.append(
                {
                    "repository": repo["full_name"],
                    "name": repo["name"],
                    "arm": arm,
                    "completed": bool(result.get("completed")),
                    "expected": sorted(expected),
                    "predicted": sorted(predicted),
                    "true_positive": sorted(true_positive),
                    "false_positive": sorted(false_positive),
                    "false_negative": sorted(false_negative),
                    "recall": divide(len(true_positive), len(expected)),
                    "precision": divide(len(true_positive), len(exposed)),
                    "f1": divide(
                        2 * len(true_positive),
                        2 * len(true_positive) + len(false_positive) + len(false_negative),
                    ),
                    "offline_available_rule_ids": sorted(offline_available[arm]),
                    "offline_stored_rule_ids": sorted(stored_rules[arm]),
                    "offline_archive_rule_ids": sorted(archive_ids) if arm == "main" else [],
                    "auto_recalled_rule_ids": sorted(recalled),
                    "tool_exposed_rule_ids": sorted(tool_exposed),
                    "raw_exposed_rule_ids": sorted(raw_exposed),
                    "exposed_rule_ids": sorted(exposed),
                    "false_negative_unavailable_offline": sorted(unavailable),
                    "false_negative_available_not_exposed": sorted(available_not_exposed),
                    "expected_exposed_not_submitted": sorted(exposed_not_submitted),
                    "submission_true_positive": sorted(submitted_true_positive),
                    "submission_false_positive": sorted(submitted_false_positive),
                    "submission_false_negative": sorted(submitted_false_negative),
                    "submission_recall": divide(len(submitted_true_positive), len(expected)),
                    "submission_precision": divide(
                        len(submitted_true_positive),
                        len(predicted),
                    ),
                    "submission_f1": divide(
                        2 * len(submitted_true_positive),
                        2 * len(submitted_true_positive)
                        + len(submitted_false_positive)
                        + len(submitted_false_negative),
                    ),
                    "rule_search_calls": rule_search_calls,
                    "auto_memory_recalled": bool(result.get("relevant_memories")),
                    "session_archive_discovered": archive_discovered,
                    "session_archive_accessed": bool(archive_exposed),
                    "session_archive_exposed_rule_ids": sorted(archive_exposed),
                    "session_archive_true_positive": sorted(expected & archive_exposed),
                    "wall_seconds": result.get("wall_seconds"),
                    "usage": result.get("usage"),
                }
            )

    arms: dict[str, Any] = {}
    for arm in selected_arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        tp = sum(len(row["true_positive"]) for row in arm_rows)
        fp = sum(len(row["false_positive"]) for row in arm_rows)
        fn = sum(len(row["false_negative"]) for row in arm_rows)
        submitted_tp = sum(len(row["submission_true_positive"]) for row in arm_rows)
        submitted_fp = sum(len(row["submission_false_positive"]) for row in arm_rows)
        submitted_fn = sum(len(row["submission_false_negative"]) for row in arm_rows)
        token_values = [
            int((row.get("usage") or {}).get("total_tokens") or 0)
            for row in arm_rows
            if (row.get("usage") or {}).get("total_tokens") is not None
        ]
        archive_rows = [row for row in arm_rows if row["session_archive_accessed"]]
        no_archive_rows = [row for row in arm_rows if not row["session_archive_accessed"]]
        arms[arm] = {
            "runs": len(arm_rows),
            "completed": sum(row["completed"] for row in arm_rows),
            "recall_basis": "rules_returned_to_agent",
            "offline_available_rule_ids": sorted(offline_available[arm]),
            "offline_missing_rule_ids": sorted(set(RULE_IDS) - offline_available[arm]),
            "offline_stored_rule_ids": sorted(stored_rules[arm]),
            "offline_archive_rule_ids": sorted(archive_ids) if arm == "main" else [],
            "macro_recall": mean(row["recall"] for row in arm_rows) if arm_rows else 0.0,
            "micro_recall": divide(tp, tp + fn),
            "micro_precision": divide(tp, tp + fp),
            "micro_f1": divide(2 * tp, 2 * tp + fp + fn),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "false_negative_unavailable_offline": sum(
                len(row["false_negative_unavailable_offline"]) for row in arm_rows
            ),
            "false_negative_available_not_exposed": sum(
                len(row["false_negative_available_not_exposed"]) for row in arm_rows
            ),
            "submission_diagnostics": {
                "micro_recall": divide(submitted_tp, submitted_tp + submitted_fn),
                "micro_precision": divide(submitted_tp, submitted_tp + submitted_fp),
                "micro_f1": divide(
                    2 * submitted_tp,
                    2 * submitted_tp + submitted_fp + submitted_fn,
                ),
                "true_positive": submitted_tp,
                "false_positive": submitted_fp,
                "false_negative": submitted_fn,
                "expected_exposed_not_submitted": sum(
                    len(row["expected_exposed_not_submitted"]) for row in arm_rows
                ),
            },
            "rule_search_calls": sum(row["rule_search_calls"] for row in arm_rows),
            "auto_memory_recall_runs": sum(row["auto_memory_recalled"] for row in arm_rows),
            "session_archive_discovery_runs": sum(
                row["session_archive_discovered"] for row in arm_rows
            ),
            "session_archive_access_runs": len(archive_rows),
            "session_archive_access_true_positive": sum(
                len(row["session_archive_true_positive"]) for row in archive_rows
            ),
            "session_archive_access_expected": sum(len(row["expected"]) for row in archive_rows),
            "no_session_archive_access_true_positive": sum(
                len(row["true_positive"]) for row in no_archive_rows
            ),
            "no_session_archive_access_expected": sum(
                len(row["expected"]) for row in no_archive_rows
            ),
            "mean_total_tokens": mean(token_values) if token_values else 0.0,
            "mean_wall_seconds": mean(
                row["wall_seconds"] for row in arm_rows if row["wall_seconds"] is not None
            )
            if arm_rows
            else 0.0,
        }

    paired: list[dict[str, Any]] = []
    for repo in repositories:
        matches = {row["arm"]: row for row in rows if row["name"] == repo["name"]}
        if set(matches) == {"main", "rules"}:
            paired.append(
                {
                    "repository": repo["full_name"],
                    "main_recall": matches["main"]["recall"],
                    "rules_recall": matches["rules"]["recall"],
                    "delta": matches["rules"]["recall"] - matches["main"]["recall"],
                }
            )

    summary = {
        "schema": "rules-recall-evaluation/v3",
        "phase": phase,
        "recall_basis": "rules_returned_to_agent",
        "selected_arms": selected_arms,
        "ground_truth_rule_assignments": sum(len(ids) for ids in truth.values()),
        "arms": arms,
        "paired": {
            "repositories": len(paired),
            "mean_recall_delta": mean(item["delta"] for item in paired) if paired else 0.0,
            "rules_better": sum(item["delta"] > 0 for item in paired),
            "equal": sum(item["delta"] == 0 for item in paired),
            "rules_worse": sum(item["delta"] < 0 for item in paired),
        },
    }
    destination = output / evaluation_subdir
    if main_archive:
        write_json(destination / "main_archive_inventory.json", main_archive)
    write_json(destination / "runs.json", rows)
    write_json(destination / "pairs.json", paired)
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", default="primary")
    parser.add_argument("--evaluation-subdir", default="evaluation")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("main", "rules"), default=["main", "rules"])
    arguments = parser.parse_args()
    evaluate(
        arguments.output.resolve(),
        arguments.phase,
        arguments.allow_partial,
        list(dict.fromkeys(arguments.arms)),
        arguments.evaluation_subdir,
    )
