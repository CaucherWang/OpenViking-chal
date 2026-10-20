"""Evaluate final Rule ID recall against the frozen manual trigger labels."""

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
    return set(RULE_PATTERN.findall(json.dumps(value, ensure_ascii=False)))


def tool_diagnostics(tools: list[dict[str, Any]]) -> tuple[set[str], int]:
    exposed: set[str] = set()
    rule_search_calls = 0
    for tool in tools:
        name = str(tool.get("tool_name") or tool.get("name") or "")
        if not name.startswith("openviking_"):
            continue
        if name == "openviking_rule_search":
            rule_search_calls += 1
        exposed.update(rule_ids_from(tool.get("result")))
    return exposed, rule_search_calls


def tools_contain(tools: list[dict[str, Any]], marker: str) -> bool:
    return any(marker in str(tool.get("result") or "") for tool in tools)


def evaluate(
    output: Path,
    phase: str,
    allow_partial: bool,
    selected_arms: list[str],
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
    offline_available = {
        arm: set(read_json(inventory_paths[arm])["rule_ids"]) for arm in selected_arms
    }
    offline_stage = read_json(output / "runtime/offline_stage.json")
    main_session_id = (offline_stage.get("main") or {}).get("session_id")
    archive_marker = f"/sessions/{main_session_id}/history/" if main_session_id else None
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
            tool_exposed, rule_search_calls = tool_diagnostics(tools)
            raw_exposed = recalled | tool_exposed
            exposed = raw_exposed & offline_available[arm]
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
                    "precision": divide(len(true_positive), len(predicted)),
                    "f1": divide(
                        2 * len(true_positive),
                        2 * len(true_positive) + len(false_positive) + len(false_negative),
                    ),
                    "offline_available_rule_ids": sorted(offline_available[arm]),
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
                    "rule_search_calls": rule_search_calls,
                    "auto_memory_recalled": bool(result.get("relevant_memories")),
                    "session_archive_accessed": bool(
                        arm == "main" and archive_marker and tools_contain(tools, archive_marker)
                    ),
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
            "session_archive_access_runs": len(archive_rows),
            "session_archive_access_true_positive": sum(
                len(row["true_positive"]) for row in archive_rows
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
        "schema": "rules-recall-evaluation/v2",
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
    destination = output / "evaluation"
    write_json(destination / "runs.json", rows)
    write_json(destination / "pairs.json", paired)
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", default="primary")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("main", "rules"), default=["main", "rules"])
    arguments = parser.parse_args()
    evaluate(
        arguments.output.resolve(),
        arguments.phase,
        arguments.allow_partial,
        list(dict.fromkeys(arguments.arms)),
    )
