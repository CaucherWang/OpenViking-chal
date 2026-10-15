#!/usr/bin/env python3
"""Classify official PersonaMem-v2 Seed errors by pipeline stage.

The classification is evidence-based and intentionally conservative:

1. Evidence absent from all persisted memories: storage failure.
2. Evidence present in persisted memories but absent from retrieved memories:
   retrieval failure.
3. Evidence present in retrieved memories: answer-model failure.

Because benchmark preferences are natural-language annotations rather than
literal memory spans, evidence is considered present when the preference or
its most relevant user message has at least the configured token coverage, or
when the normalized source message appears verbatim.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "benchmark" / "aml" / "data" / "results"
DEFAULT_SEARCH_CSV = RESULTS_DIR / "personamem_v2_search_official_mcq_32k_full.csv"
DEFAULT_FULL_HISTORY_CSV = RESULTS_DIR / "personamem_v2_full_history_mcq_32k.csv"
DEFAULT_STORAGE_ROOT = REPOSITORY_ROOT / ".openviking" / "data" / "viking" / "default" / "user"
DEFAULT_OUTPUT_CSV = (
    RESULTS_DIR / "personamem_v2_search_official_mcq_32k_error_classification.csv"
)
DEFAULT_OUTPUT_SUMMARY = (
    RESULTS_DIR / "personamem_v2_search_official_mcq_32k_error_classification_summary.json"
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
WHITESPACE_RE = re.compile(r"\s+")
STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "don't",
    "each",
    "even",
    "feel",
    "feels",
    "felt",
    "for",
    "from",
    "get",
    "gets",
    "getting",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "like",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "not",
    "now",
    "of",
    "on",
    "once",
    "one",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "really",
    "said",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "up",
    "very",
    "want",
    "wants",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

CLASS_STORAGE = "memory_not_stored"
CLASS_RETRIEVAL = "memory_not_retrieved"
CLASS_MODEL = "retrieved_but_not_used"
CLASS_LABELS = {
    CLASS_STORAGE: "记忆没存下来",
    CLASS_RETRIEVAL: "记忆已存但未召回",
    CLASS_MODEL: "已召回但模型没用好",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-csv", type=Path, default=DEFAULT_SEARCH_CSV)
    parser.add_argument("--full-history-csv", type=Path, default=DEFAULT_FULL_HISTORY_CSV)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--coverage-threshold", type=float, default=0.6)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.lower()).strip()


def normalize_token(token: str) -> str:
    """Apply lightweight English morphology normalization without dependencies."""
    if token.endswith("'s") and len(token) > 3:
        token = token[:-2]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> set[str]:
    return {
        normalize_token(token)
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS
    }


def token_coverage(target: set[str], candidate: set[str]) -> float:
    return len(target & candidate) / len(target) if target else 0.0


def parse_source_messages(raw_snippet: str) -> list[str]:
    try:
        messages = json.loads(raw_snippet)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(messages, list):
        return []
    return [
        str(item.get("content") or "")
        for item in messages
        if isinstance(item, dict)
        and str(item.get("content") or "").strip()
    ]


def select_target_source(row: dict[str, str]) -> str:
    messages = parse_source_messages(row.get("related_conversation_snippet", ""))
    if not messages:
        return ""

    pref_type = row.get("pref_type", "").lower()
    if "forget" in pref_type:
        forget_messages = [message for message in messages if "forget" in message.lower()]
        if forget_messages:
            messages = forget_messages

    preference_tokens = content_tokens(row.get("preference", ""))
    return max(
        messages,
        key=lambda message: (
            token_coverage(preference_tokens, content_tokens(message)),
            -len(content_tokens(message)),
        ),
    )


def memory_uri(path: Path, user_dir: Path) -> str:
    return f"viking://user/{user_dir.name}/{path.relative_to(user_dir).as_posix()}"


def load_storage(storage_root: Path) -> dict[str, list[dict[str, str]]]:
    storage: dict[str, list[dict[str, str]]] = {}
    for user_dir in sorted(storage_root.glob("personamem-v2-32k-official-persona*")):
        memories_dir = user_dir / "memories"
        memories = []
        for path in memories_dir.rglob("*.md"):
            if path.name.startswith("."):
                continue
            memories.append(
                {
                    "id": memory_uri(path, user_dir),
                    "content": path.read_text(encoding="utf-8"),
                }
            )
        storage[user_dir.name] = memories
    return storage


def parse_retrieved_memories(raw_value: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        {"id": str(item.get("id") or ""), "content": str(item.get("content") or "")}
        for item in parsed
        if isinstance(item, dict)
    ]


def score_memories(
    memories: Iterable[dict[str, str]],
    *,
    preference: str,
    target_source: str,
    threshold: float,
) -> dict[str, Any]:
    preference_tokens = content_tokens(preference)
    source_tokens = content_tokens(target_source)
    normalized_source = normalize_text(target_source)
    best: dict[str, Any] = {
        "found": False,
        "preference_coverage": 0.0,
        "source_coverage": 0.0,
        "exact_source": False,
        "best_uri": "",
    }
    best_rank = (False, 0.0, 0.0)

    for memory in memories:
        content = memory.get("content", "")
        tokens = content_tokens(content)
        preference_coverage = token_coverage(preference_tokens, tokens)
        source_coverage = token_coverage(source_tokens, tokens)
        exact_source = bool(
            normalized_source and normalized_source in normalize_text(content)
        )
        found = (
            exact_source
            or preference_coverage >= threshold
            or source_coverage >= threshold
        )
        rank = (exact_source, max(preference_coverage, source_coverage), source_coverage)
        if rank > best_rank:
            best_rank = rank
            best = {
                "found": found,
                "preference_coverage": preference_coverage,
                "source_coverage": source_coverage,
                "exact_source": exact_source,
                "best_uri": memory.get("id", ""),
            }

    return best


def parse_bool(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def load_full_history_correctness(path: Path) -> list[bool | None]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            parse_bool(row.get("is_correct_mcq_32k", ""))
            for row in csv.DictReader(file)
        ]


def rounded_percent(part: int, total: int) -> float:
    return round(100.0 * part / total, 3) if total else 0.0


def breakdown(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped[str(record[field])][str(record["classification"])] += 1
    result = {}
    for value, counts in sorted(grouped.items()):
        total = sum(counts.values())
        result[value] = {
            "total_errors": total,
            **{
                classification: {
                    "count": counts[classification],
                    "percent": rounded_percent(counts[classification], total),
                }
                for classification in CLASS_LABELS
            },
        }
    return result


def threshold_sensitivity(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for threshold in (0.5, 0.6, 0.7):
        counts: Counter[str] = Counter()
        for record in records:
            storage_found = bool(record["storage_exact_source"]) or max(
                float(record["storage_preference_coverage"]),
                float(record["storage_source_coverage"]),
            ) >= threshold
            retrieved_found = bool(record["retrieved_exact_source"]) or max(
                float(record["retrieved_preference_coverage"]),
                float(record["retrieved_source_coverage"]),
            ) >= threshold
            if not storage_found:
                counts[CLASS_STORAGE] += 1
            elif not retrieved_found:
                counts[CLASS_RETRIEVAL] += 1
            else:
                counts[CLASS_MODEL] += 1
        result[str(threshold)] = {
            classification: {
                "count": counts[classification],
                "percent": rounded_percent(counts[classification], len(records)),
            }
            for classification in CLASS_LABELS
        }
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.coverage_threshold <= 1.0:
        raise ValueError("--coverage-threshold must be in (0, 1]")
    csv.field_size_limit(sys.maxsize)

    storage = load_storage(args.storage_root)
    full_history_correctness = load_full_history_correctness(args.full_history_csv)
    records: list[dict[str, Any]] = []
    total_rows = 0
    scored_rows = 0
    correct_rows = 0
    evaluation_errors = 0

    with args.search_csv.open(newline="", encoding="utf-8") as file:
        for row_index, row in enumerate(csv.DictReader(file), start=1):
            total_rows += 1
            correctness = parse_bool(row.get("is_correct_mcq_32k", ""))
            if correctness is None:
                evaluation_errors += 1
                continue
            scored_rows += 1
            if correctness:
                correct_rows += 1
                continue

            target_source = select_target_source(row)
            user_id = row.get("openviking_user_id", "")
            storage_score = score_memories(
                storage.get(user_id, []),
                preference=row.get("preference", ""),
                target_source=target_source,
                threshold=args.coverage_threshold,
            )
            retrieved_score = score_memories(
                parse_retrieved_memories(row.get("retrieved_memories_mcq_32k", "")),
                preference=row.get("preference", ""),
                target_source=target_source,
                threshold=args.coverage_threshold,
            )
            if not storage_score["found"]:
                classification = CLASS_STORAGE
            elif not retrieved_score["found"]:
                classification = CLASS_RETRIEVAL
            else:
                classification = CLASS_MODEL

            records.append(
                {
                    "row_index": row_index,
                    "persona_id": row.get("persona_id", ""),
                    "pref_type": row.get("pref_type", ""),
                    "who": row.get("who", ""),
                    "conversation_scenario": row.get("conversation_scenario", ""),
                    "classification": classification,
                    "classification_zh": CLASS_LABELS[classification],
                    "full_history_is_correct": (
                        full_history_correctness[row_index - 1]
                        if row_index <= len(full_history_correctness)
                        else None
                    ),
                    "storage_evidence_found": storage_score["found"],
                    "storage_preference_coverage": round(
                        storage_score["preference_coverage"], 6
                    ),
                    "storage_source_coverage": round(
                        storage_score["source_coverage"], 6
                    ),
                    "storage_exact_source": storage_score["exact_source"],
                    "storage_best_uri": storage_score["best_uri"],
                    "retrieved_evidence_found": retrieved_score["found"],
                    "retrieved_preference_coverage": round(
                        retrieved_score["preference_coverage"], 6
                    ),
                    "retrieved_source_coverage": round(
                        retrieved_score["source_coverage"], 6
                    ),
                    "retrieved_exact_source": retrieved_score["exact_source"],
                    "retrieved_best_uri": retrieved_score["best_uri"],
                    "retrieved_count": int(row.get("retrieved_count_mcq_32k") or 0),
                }
            )

    if len(full_history_correctness) != total_rows:
        raise ValueError(
            "Search and full-history CSV row counts differ: "
            f"{total_rows} != {len(full_history_correctness)}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    class_counts = Counter(record["classification"] for record in records)
    retrieval_candidates = [
        record for record in records if record["classification"] == CLASS_RETRIEVAL
    ]
    full_history_by_class = {}
    for classification in CLASS_LABELS:
        values = [
            record["full_history_is_correct"]
            for record in records
            if record["classification"] == classification
            and record["full_history_is_correct"] is not None
        ]
        correct = sum(value is True for value in values)
        full_history_by_class[classification] = {
            "scored": len(values),
            "correct": correct,
            "accuracy_percent": rounded_percent(correct, len(values)),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "coverage_threshold": args.coverage_threshold,
            "evidence_rule": (
                "normalized source exact match OR preference token coverage >= threshold "
                "OR selected source-message token coverage >= threshold"
            ),
            "token_normalization": (
                "lowercase, English stopword removal, and lightweight normalization of "
                "possessives, plurals, and common verb suffixes"
            ),
            "target_source_rule": (
                "most preference-related source message (shorter message wins ties); "
                "for ask_to_forget, prefer a message containing 'forget'"
            ),
            "classification_labels_zh": CLASS_LABELS,
        },
        "inputs": {
            "search_csv": str(args.search_csv.relative_to(REPOSITORY_ROOT)),
            "full_history_csv": str(args.full_history_csv.relative_to(REPOSITORY_ROOT)),
            "storage_root": str(args.storage_root.relative_to(REPOSITORY_ROOT)),
            "official_persona_storage_count": len(storage),
            "persisted_memory_file_count": sum(len(value) for value in storage.values()),
        },
        "evaluation": {
            "total_rows": total_rows,
            "scored_rows": scored_rows,
            "correct_rows": correct_rows,
            "wrong_rows": len(records),
            "evaluation_error_rows": evaluation_errors,
            "accuracy_percent": rounded_percent(correct_rows, scored_rows),
        },
        "classification": {
            classification: {
                "label_zh": CLASS_LABELS[classification],
                "count": class_counts[classification],
                "percent": rounded_percent(class_counts[classification], len(records)),
            }
            for classification in CLASS_LABELS
        },
        "evidence_checks": {
            "exact_source_hits_in_storage": sum(
                bool(record["storage_exact_source"]) for record in records
            ),
            "exact_source_hits_in_retrieval": sum(
                bool(record["retrieved_exact_source"]) for record in records
            ),
            "retrieval_candidates_with_exact_source_in_storage": sum(
                bool(record["storage_exact_source"])
                for record in retrieval_candidates
            ),
            "retrieval_candidates_with_same_best_uri": sum(
                record["storage_best_uri"] == record["retrieved_best_uri"]
                for record in retrieval_candidates
            ),
            "median_retrieved_count_for_wrong_rows": statistics.median(
                record["retrieved_count"] for record in records
            ),
        },
        "full_history_accuracy_by_class": full_history_by_class,
        "threshold_sensitivity": threshold_sensitivity(records),
        "breakdown_by_pref_type": breakdown(records, "pref_type"),
        "breakdown_by_who": breakdown(records, "who"),
        "breakdown_by_conversation_scenario": breakdown(
            records, "conversation_scenario"
        ),
    }
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
