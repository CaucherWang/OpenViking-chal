#!/usr/bin/env python3
"""Build bilingual Excel workbooks for PersonaMem-v2 search errors.

The three worksheets separate failures into:

1. target evidence was not persisted;
2. target evidence was persisted but not retrieved;
3. target evidence was retrieved but the answer model still chose incorrectly.

Translations and per-row diagnoses are generated with GPT-5.5 and cached in a
SQLite database so an interrupted run can resume without repeating completed
requests.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from analyze_official_seed_errors import (
    CLASS_MODEL,
    CLASS_RETRIEVAL,
    CLASS_STORAGE,
    content_tokens,
    normalize_text,
    parse_bool,
    parse_retrieved_memories,
    select_target_source,
    token_coverage,
)
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "benchmark" / "aml" / "data" / "results"
STORAGE_ROOT = REPOSITORY_ROOT / ".openviking" / "data" / "viking" / "default" / "user"
OV_CONFIG = REPOSITORY_ROOT / ".openviking" / "ov.conf"

GPT_5_5_MODEL = "gpt-5.5-2026-04-24"
GPT_4O_MINI_MODEL = "gpt-4o-mini-2024-07-18"
ES_AZURE_ENDPOINT = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
ES_AZURE_API_VERSION = "2024-02-01"
EXCEL_CELL_LIMIT = 32_767
SAFE_CELL_LIMIT = 30_000
TRANSLATION_SOURCE_LIMIT = 8_000
ANALYSIS_FIELD_LIMIT = 1_500
EVIDENCE_EXCERPT_LIMIT = 1_200

SHEET_NAMES = {
    CLASS_STORAGE: "记忆没存储",
    CLASS_RETRIEVAL: "记忆没召回",
    CLASS_MODEL: "已召回但答错",
}

EXPERIMENTS = {
    "gpt55": {
        "label": "255k-Max-Token-official / GPT-5.5",
        "csv": RESULTS_DIR / "personamem_v2_search_255k_official_gpt5_5_topk100.csv",
        "user_prefix": "personamem-v2-32k-255k-Max-Token-official-persona",
        "output": RESULTS_DIR / "personamem_v2_255k_official_gpt5_5_error_analysis.xlsx",
    },
    "seed": {
        "label": "official / Seed",
        "csv": RESULTS_DIR / "personamem_v2_search_official_mcq_32k_full.csv",
        "user_prefix": "personamem-v2-32k-official-persona",
        "output": RESULTS_DIR / "personamem_v2_official_seed_error_analysis.xlsx",
    },
}

CURATED_SOURCE_FIELDS = {
    "persona_id",
    "user_query",
    "correct_answer",
    "incorrect_answers",
    "preference",
    "related_conversation_snippet",
    "openviking_user_id",
    "evaluation_error",
    "search_query_mcq_32k",
    "retrieval_query_mcq_32k",
    "retrieved_memories_mcq_32k",
    "retrieved_count_mcq_32k",
    "option_mapping_32k",
    "correct_mcq_option_32k",
    "model_response_mcq_32k",
    "predicted_answer_mcq_32k",
    "is_correct_mcq_32k",
}


@dataclass(frozen=True)
class EvidenceScore:
    found: bool
    preference_coverage: float
    source_coverage: float
    exact_source: bool
    best_uri: str
    best_content: str


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    csv_path: Path
    user_prefix: str
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["both", *EXPERIMENTS],
        default="both",
        help="Workbook set to generate.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=RESULTS_DIR / "personamem_v2_error_workbook_ai_cache.sqlite3",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.6)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument(
        "--translation-model",
        choices=["gpt-4o-mini", "gpt-5.5"],
        default="gpt-4o-mini",
        help="Model used for Chinese translation.",
    )
    parser.add_argument(
        "--diagnosis-model",
        choices=["gpt-4o-mini", "gpt-5.5"],
        default="gpt-5.5",
        help="Model used for per-row error explanations.",
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Write deterministic diagnoses and leave Chinese translations blank.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many wrong rows per experiment (for smoke tests).",
    )
    args = parser.parse_args()
    if not 0 < args.coverage_threshold <= 1:
        parser.error("--coverage-threshold must be in (0, 1]")
    if args.parallel < 1 or args.retry_attempts < 1:
        parser.error("--parallel and --retry-attempts must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate(text: Any, limit: int = SAFE_CELL_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    note = f"\n\n[内容过长，已截断；原始字符数：{len(value)}]"
    return value[: limit - len(note)] + note


def translation_source(text: str) -> str:
    return truncate(text, TRANSLATION_SOURCE_LIMIT)


def parse_user_query(raw_value: str) -> str:
    try:
        parsed = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return str(raw_value).strip("\"'")
    if isinstance(parsed, dict):
        return str(parsed.get("content", parsed.get("text", "")) or "")
    return str(parsed)


def parse_option_mapping(raw_value: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}


def format_options(mapping: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in mapping.items())


def correct_option(mapping: dict[str, str], answer: str) -> str:
    return next((key for key, value in mapping.items() if value == answer), "")


def memory_uri(path: Path, user_dir: Path) -> str:
    return f"viking://user/{user_dir.name}/{path.relative_to(user_dir).as_posix()}"


class StorageLoader:
    def __init__(self, root: Path, expected_prefix: str) -> None:
        self.root = root
        self.expected_prefix = expected_prefix
        self.cache: dict[str, list[dict[str, str]]] = {}

    def get(self, user_id: str) -> list[dict[str, str]]:
        if user_id in self.cache:
            return self.cache[user_id]
        if not user_id.startswith(self.expected_prefix):
            raise ValueError(
                f"unexpected user namespace {user_id!r}; expected {self.expected_prefix!r}"
            )
        user_dir = self.root / user_id
        if not user_dir.is_dir():
            self.cache[user_id] = []
            return []
        memories = []
        for path in sorted((user_dir / "memories").rglob("*.md")):
            if path.name.startswith("."):
                continue
            memories.append(
                {
                    "id": memory_uri(path, user_dir),
                    "content": path.read_text(encoding="utf-8"),
                }
            )
        self.cache[user_id] = memories
        return memories


def score_memories(
    memories: Iterable[dict[str, str]],
    *,
    preference: str,
    target_source: str,
    threshold: float,
) -> EvidenceScore:
    preference_tokens = content_tokens(preference)
    source_tokens = content_tokens(target_source)
    normalized_source = normalize_text(target_source)
    best_rank: tuple[bool, float, float] | None = None
    best: EvidenceScore | None = None

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
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = EvidenceScore(
                found=found,
                preference_coverage=preference_coverage,
                source_coverage=source_coverage,
                exact_source=exact_source,
                best_uri=memory.get("id", ""),
                best_content=content,
            )

    return best or EvidenceScore(False, 0.0, 0.0, False, "", "")


def uri_directory(uri: str) -> str:
    return uri.rsplit("/", 1)[0] if "/" in uri else uri


def unique_join(values: Iterable[str]) -> str:
    return "\n".join(dict.fromkeys(value for value in values if value))


def evidence_excerpt(
    text: str,
    *,
    preference: str,
    target_source: str,
    limit: int = EVIDENCE_EXCERPT_LIMIT,
) -> str:
    """Select the most target-relevant lines/windows from a long memory."""
    if len(text) <= limit:
        return text
    target_tokens = content_tokens(f"{preference}\n{target_source}")
    segments: list[tuple[int, str]] = []
    for line_index, line in enumerate(re.split(r"\n+", text)):
        line = line.strip()
        if not line:
            continue
        for offset in range(0, len(line), 800):
            segment = line[offset : offset + 800]
            segments.append((line_index * 10_000 + offset, segment))

    ranked = sorted(
        segments,
        key=lambda item: (
            token_coverage(target_tokens, content_tokens(item[1])),
            len(target_tokens & content_tokens(item[1])),
        ),
        reverse=True,
    )
    selected: list[tuple[int, str]] = []
    used_chars = 0
    for position, segment in ranked:
        if used_chars + len(segment) + 6 > limit:
            continue
        selected.append((position, segment))
        used_chars += len(segment) + 6
        if len(selected) >= 5:
            break
    if not selected:
        return truncate(text, limit)
    selected.sort()
    return "\n\n[...]\n\n".join(segment for _, segment in selected)


def classify_experiment(
    experiment: Experiment,
    *,
    threshold: float,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    csv.field_size_limit(sys.maxsize)
    storage = StorageLoader(STORAGE_ROOT, experiment.user_prefix)
    records: list[dict[str, Any]] = []
    counts = Counter()

    with experiment.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        for row_index, row in enumerate(reader, start=1):
            correctness = parse_bool(row.get("is_correct_mcq_32k", ""))
            if correctness is None:
                counts["evaluation_errors"] += 1
                continue
            counts["scored"] += 1
            if correctness:
                counts["correct"] += 1
                continue
            if limit is not None and len(records) >= limit:
                break

            preference = row.get("preference", "")
            target_source = select_target_source(row)
            persisted = storage.get(row.get("openviking_user_id", ""))
            retrieved = parse_retrieved_memories(
                row.get("retrieved_memories_mcq_32k", "")
            )
            storage_score = score_memories(
                persisted,
                preference=preference,
                target_source=target_source,
                threshold=threshold,
            )
            retrieved_score = score_memories(
                retrieved,
                preference=preference,
                target_source=target_source,
                threshold=threshold,
            )

            if not storage_score.found:
                classification = CLASS_STORAGE
            elif not retrieved_score.found:
                classification = CLASS_RETRIEVAL
            else:
                classification = CLASS_MODEL

            mapping = parse_option_mapping(row.get("option_mapping_32k", ""))
            answer = row.get("correct_answer", "")
            retrieved_uris = [item["id"] for item in retrieved]
            source_excerpt = evidence_excerpt(
                target_source,
                preference=preference,
                target_source=target_source,
            )
            storage_excerpt = evidence_excerpt(
                storage_score.best_content,
                preference=preference,
                target_source=target_source,
            )
            retrieved_excerpt = evidence_excerpt(
                retrieved_score.best_content,
                preference=preference,
                target_source=target_source,
            )
            records.append(
                {
                    "experiment": experiment.label,
                    "source_csv": str(experiment.csv_path.relative_to(REPOSITORY_ROOT)),
                    "source_row": row_index + 1,
                    "benchmark_row_index": row_index,
                    "classification": classification,
                    "persona_id": row.get("persona_id", ""),
                    "user_query": parse_user_query(row.get("user_query", "")),
                    "preference": preference,
                    "target_source": source_excerpt,
                    "storage_content": storage_excerpt,
                    "storage_uri": storage_score.best_uri,
                    "retrieved_content": retrieved_excerpt,
                    "retrieved_uri": retrieved_score.best_uri,
                    "retrieved_directory": uri_directory(retrieved_score.best_uri),
                    "all_retrieved_uris": unique_join(retrieved_uris),
                    "all_retrieved_directories": unique_join(
                        uri_directory(uri) for uri in retrieved_uris
                    ),
                    "retrieved_count": len(retrieved),
                    "options": format_options(mapping),
                    "option_mapping": mapping,
                    "model_response": row.get("model_response_mcq_32k", ""),
                    "predicted_option": row.get("predicted_answer_mcq_32k", ""),
                    "correct_option": (
                        row.get("correct_mcq_option_32k", "")
                        or correct_option(mapping, answer)
                    ),
                    "correct_answer": answer,
                    "storage_preference_coverage": storage_score.preference_coverage,
                    "storage_source_coverage": storage_score.source_coverage,
                    "storage_exact_source": storage_score.exact_source,
                    "retrieved_preference_coverage": (
                        retrieved_score.preference_coverage
                    ),
                    "retrieved_source_coverage": retrieved_score.source_coverage,
                    "retrieved_exact_source": retrieved_score.exact_source,
                    "raw": row,
                }
            )
            counts[classification] += 1

    counts["wrong"] = len(records)
    return records, source_fields, dict(counts)


class AICache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnoses (
                cache_key TEXT PRIMARY KEY,
                diagnosis TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def translations(self) -> dict[str, str]:
        return dict(
            self.connection.execute(
                "SELECT cache_key, translated_text FROM translations"
            )
        )

    def diagnoses(self) -> dict[str, str]:
        return dict(
            self.connection.execute("SELECT cache_key, diagnosis FROM diagnoses")
        )

    def save_translations(self, values: dict[str, tuple[str, str]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO translations
                (cache_key, source_text, translated_text, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (key, source, translated, now)
                for key, (source, translated) in values.items()
            ],
        )
        self.connection.commit()

    def save_diagnoses(self, values: dict[str, str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO diagnoses (cache_key, diagnosis, updated_at)
            VALUES (?, ?, ?)
            """,
            [(key, diagnosis, now) for key, diagnosis in values.items()],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def load_gpt_key(model: str) -> str:
    config_key = "gpt_4o_mini_key" if model == GPT_4O_MINI_MODEL else "gpt_5_5_key"
    env_name = "GPT_4O_MINI_API_KEY" if model == GPT_4O_MINI_MODEL else "GPT_5_5_API_KEY"
    env_key = os.getenv(env_name, "")
    if env_key:
        return env_key
    if not OV_CONFIG.is_file():
        return ""
    raw = json.loads(OV_CONFIG.read_text(encoding="utf-8"))
    vlm = raw.get("vlm", {}) if isinstance(raw, dict) else {}
    return str(vlm.get(config_key, "")) if isinstance(vlm, dict) else ""


def extract_json_items(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text, strict=False)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            raise
        parsed = json.loads(text[start : end + 1], strict=False)
    if isinstance(parsed, dict):
        parsed = parsed.get("items", [])
    if not isinstance(parsed, list):
        raise TypeError("model response is not a JSON array")
    return [item for item in parsed if isinstance(item, dict)]


async def model_json_request(
    client: Any,
    semaphore: asyncio.Semaphore,
    *,
    system_prompt: str,
    payload: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    retry_attempts: int,
) -> list[dict[str, Any]]:
    async with semaphore:
        for attempt in range(retry_attempts):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    max_tokens=max_tokens,
                    stream=False,
                )
                content = str(response.choices[0].message.content or "")
                return extract_json_items(content)
            except Exception as exc:
                if attempt + 1 >= retry_attempts:
                    raise RuntimeError(
                        f"{model} request failed after {retry_attempts} attempts"
                    ) from exc
                await asyncio.sleep(min(30.0, 1.5 * 2**attempt))
    return []


def make_translation_batches(
    items: dict[str, str],
    *,
    max_chars: int = 24_000,
    max_items: int = 40,
) -> list[list[dict[str, str]]]:
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_chars = 0
    for key, text in items.items():
        item = {"id": key, "text": text}
        item_chars = len(text)
        if current and (
            len(current) >= max_items or current_chars + item_chars > max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


async def fill_translations(
    client: Any,
    cache: AICache,
    records: list[dict[str, Any]],
    *,
    model: str,
    parallel: int,
    retry_attempts: int,
) -> dict[str, str]:
    fields = (
        "user_query",
        "preference",
        "target_source",
        "storage_content",
        "retrieved_content",
        "options",
        "model_response",
        "correct_answer",
    )
    sources: dict[str, str] = {}
    for record in records:
        for field in fields:
            source = translation_source(record[field])
            if source:
                sources.setdefault(sha256_text(source), source)

    translations = cache.translations()
    missing = {key: value for key, value in sources.items() if key not in translations}
    batches = make_translation_batches(missing)
    if not batches:
        return translations

    print(
        f"Translating {len(missing)} unique texts in {len(batches)} batches...",
        flush=True,
    )
    semaphore = asyncio.Semaphore(parallel)
    system_prompt = (
        "You are a precise English-to-Simplified-Chinese translator. Translate every "
        "item faithfully and completely. Preserve markdown, option letters, names, dates, "
        "URIs, and quoted strings. Do not summarize or explain. Return only a JSON array "
        'with objects shaped exactly as {"id":"same id","text":"Chinese translation"}.'
    )

    async def run(batch: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
        expected = {item["id"]: item["text"] for item in batch}
        by_id: dict[str, str] = {}
        remaining = batch
        for _ in range(retry_attempts):
            output = await model_json_request(
                client,
                semaphore,
                system_prompt=system_prompt,
                payload=remaining,
                model=model,
                max_tokens=16_000,
                retry_attempts=retry_attempts,
            )
            by_id.update(
                {
                    str(item.get("id", "")): str(item.get("text", "")).strip()
                    for item in output
                    if item.get("id") and item.get("text")
                }
            )
            remaining = [item for item in batch if item["id"] not in by_id]
            if not remaining:
                break
        if remaining:
            raise ValueError(f"translation response omitted {len(remaining)} items")
        return {key: (expected[key], by_id[key]) for key in expected}

    completed = 0
    tasks = [asyncio.create_task(run(batch)) for batch in batches]
    try:
        for task in asyncio.as_completed(tasks):
            values = await task
            cache.save_translations(values)
            translations.update({key: value[1] for key, value in values.items()})
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Translation batches: {completed}/{len(tasks)}", flush=True)
    except Exception:
        for task in tasks:
            task.cancel()
        raise
    return translations


def record_diagnosis_key(record: dict[str, Any]) -> str:
    relevant = {
        key: record.get(key, "")
        for key in (
            "experiment",
            "source_row",
            "classification",
            "user_query",
            "preference",
            "target_source",
            "storage_content",
            "retrieved_content",
            "options",
            "model_response",
            "predicted_option",
            "correct_option",
            "correct_answer",
            "diagnosis_model",
        )
    }
    return sha256_text(json.dumps(relevant, ensure_ascii=False, sort_keys=True))


def diagnosis_payload(record: dict[str, Any]) -> dict[str, Any]:
    def short(field: str) -> str:
        return truncate(record[field], ANALYSIS_FIELD_LIMIT)

    return {
        "id": record_diagnosis_key(record),
        "failure_stage": SHEET_NAMES[record["classification"]],
        "query": short("user_query"),
        "target_preference": short("preference"),
        "source_evidence": short("target_source"),
        "best_persisted_memory": short("storage_content"),
        "best_retrieved_memory": short("retrieved_content"),
        "storage_preference_coverage": round(
            record["storage_preference_coverage"], 3
        ),
        "storage_source_coverage": round(record["storage_source_coverage"], 3),
        "retrieved_preference_coverage": round(
            record["retrieved_preference_coverage"], 3
        ),
        "retrieved_source_coverage": round(record["retrieved_source_coverage"], 3),
        "options": short("options"),
        "model_response": short("model_response"),
        "predicted_option": record["predicted_option"],
        "correct_option": record["correct_option"],
        "correct_answer": short("correct_answer"),
    }


def deterministic_diagnosis(record: dict[str, Any]) -> str:
    storage_coverage = max(
        record["storage_preference_coverage"], record["storage_source_coverage"]
    )
    retrieval_coverage = max(
        record["retrieved_preference_coverage"],
        record["retrieved_source_coverage"],
    )
    if record["classification"] == CLASS_STORAGE:
        return (
            f"持久化记忆中未找到足以支持目标偏好“{truncate(record['preference'], 100)}”"
            f"的证据；最佳候选覆盖率仅为 {storage_coverage:.1%}，且没有原始证据的精确"
            "匹配，因此错误发生在记忆抽取/存储阶段。"
        )
    if record["classification"] == CLASS_RETRIEVAL:
        return (
            f"目标证据已经存储在 {record['storage_uri'] or '持久化记忆'}，但本次召回的"
            f"最佳证据覆盖率仅为 {retrieval_coverage:.1%}，未达到判定阈值；回答模型"
            "没有获得决定正确选项所需的目标偏好，因此错误发生在召回阶段。"
        )
    return (
        f"召回结果已包含目标证据，但模型选择了 {record['predicted_option'] or '未知'}，"
        f"正确选项是 {record['correct_option'] or '未知'}。模型没有正确利用已召回偏好，"
        "或在推理时错误地采信了干扰选项。"
    )


async def fill_diagnoses(
    client: Any,
    cache: AICache,
    records: list[dict[str, Any]],
    *,
    model: str,
    parallel: int,
    retry_attempts: int,
) -> dict[str, str]:
    diagnoses = cache.diagnoses()
    deterministic_by_id = {
        record_diagnosis_key(record): deterministic_diagnosis(record)
        for record in records
    }
    payloads = [
        diagnosis_payload(record)
        for record in records
        if record_diagnosis_key(record) not in diagnoses
    ]
    batches = [payloads[index : index + 12] for index in range(0, len(payloads), 12)]
    if not batches:
        return diagnoses

    print(f"Diagnosing {len(payloads)} rows in {len(batches)} batches...", flush=True)
    semaphore = asyncio.Semaphore(parallel)
    system_prompt = (
        "你是记忆检索系统的误差分析员。输入已经由确定性规则分成“记忆没存储”、"
        "“记忆没召回”或“已召回但答错”。请严格接受给定阶段，不要改分类。逐条比较"
        "目标偏好、原始证据、持久化/召回证据、选项和模型回答，用简体中文写 2-4 句"
        "具体原因。必须指出该行缺失、未召回或被误用的具体信息；对模型答错还要说明"
        "错误选项为何是干扰项、正确选项依赖什么证据。不要泛泛而谈。只返回 JSON 数组，"
        '每项格式为 {"id":"原 id","reason":"中文原因"}。'
    )

    async def run(batch: list[dict[str, Any]]) -> dict[str, str]:
        expected = {str(item["id"]) for item in batch}
        by_id: dict[str, str] = {}
        remaining = batch
        for _ in range(retry_attempts):
            output = await model_json_request(
                client,
                semaphore,
                system_prompt=system_prompt,
                payload=remaining,
                model=model,
                max_tokens=8_000,
                retry_attempts=retry_attempts,
            )
            by_id.update(
                {
                    str(item.get("id", "")): str(item.get("reason", "")).strip()
                    for item in output
                    if item.get("id") and item.get("reason")
                }
            )
            remaining = [item for item in batch if str(item["id"]) not in by_id]
            if not remaining:
                break
        if remaining:
            by_id.update(
                {
                    str(item["id"]): deterministic_by_id[str(item["id"])]
                    for item in remaining
                }
            )
        return {key: by_id[key] for key in expected}

    completed = 0
    tasks = [asyncio.create_task(run(batch)) for batch in batches]
    try:
        for task in asyncio.as_completed(tasks):
            values = await task
            cache.save_diagnoses(values)
            diagnoses.update(values)
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Diagnosis batches: {completed}/{len(tasks)}", flush=True)
    except Exception:
        for task in tasks:
            task.cancel()
        raise
    return diagnoses


def translated(translations: dict[str, str], text: str) -> str:
    source = translation_source(text)
    return translations.get(sha256_text(source), "") if source else ""


def workbook_columns(source_fields: list[str]) -> list[tuple[str, int]]:
    curated = [
        ("实验组", 28),
        ("源 CSV", 45),
        ("源 CSV 行号", 12),
        ("基准数据行号", 12),
        ("Persona ID", 12),
        ("错误分类", 18),
        ("错误 Query（原文）", 48),
        ("错误 Query（中文）", 48),
        ("目标相关记忆/偏好（原文）", 42),
        ("目标相关记忆/偏好（中文）", 42),
        ("原对话证据片段（原文）", 55),
        ("原对话证据片段（中文）", 55),
        ("持久化最佳证据片段（原文）", 60),
        ("持久化最佳证据片段（中文）", 60),
        ("持久化证据 URI", 55),
        ("召回的最佳证据片段（原文）", 60),
        ("召回的最佳证据片段（中文）", 60),
        ("召回证据 URI", 55),
        ("从哪个目录召回", 55),
        ("全部召回 URI", 65),
        ("全部召回目录", 65),
        ("召回数量", 10),
        ("题目选项（原文）", 55),
        ("题目选项（中文）", 55),
        ("模型回答（原文）", 55),
        ("模型回答（中文）", 55),
        ("模型选择", 10),
        ("正确选项", 10),
        ("正确答案（原文）", 55),
        ("正确答案（中文）", 55),
        ("具体错误原因（中文）", 65),
        ("存储-偏好覆盖率", 14),
        ("存储-原证据覆盖率", 16),
        ("存储-原证据精确匹配", 18),
        ("召回-偏好覆盖率", 14),
        ("召回-原证据覆盖率", 16),
        ("召回-原证据精确匹配", 18),
    ]
    metadata = [
        (f"其余信息:{field}", 24)
        for field in source_fields
        if field not in CURATED_SOURCE_FIELDS
    ]
    return curated + metadata


def workbook_row(
    record: dict[str, Any],
    translations: dict[str, str],
    diagnoses: dict[str, str],
    source_fields: list[str],
) -> list[Any]:
    values: list[Any] = [
        record["experiment"],
        record["source_csv"],
        record["source_row"],
        record["benchmark_row_index"],
        record["persona_id"],
        SHEET_NAMES[record["classification"]],
        record["user_query"],
        translated(translations, record["user_query"]),
        record["preference"],
        translated(translations, record["preference"]),
        record["target_source"],
        translated(translations, record["target_source"]),
        record["storage_content"],
        translated(translations, record["storage_content"]),
        record["storage_uri"],
        record["retrieved_content"],
        translated(translations, record["retrieved_content"]),
        record["retrieved_uri"],
        record["retrieved_directory"],
        record["all_retrieved_uris"],
        record["all_retrieved_directories"],
        record["retrieved_count"],
        record["options"],
        translated(translations, record["options"]),
        record["model_response"],
        translated(translations, record["model_response"]),
        record["predicted_option"],
        record["correct_option"],
        record["correct_answer"],
        translated(translations, record["correct_answer"]),
        diagnoses.get(
            record_diagnosis_key(record), deterministic_diagnosis(record)
        ),
        record["storage_preference_coverage"],
        record["storage_source_coverage"],
        record["storage_exact_source"],
        record["retrieved_preference_coverage"],
        record["retrieved_source_coverage"],
        record["retrieved_exact_source"],
    ]
    values.extend(
        record["raw"].get(field, "")
        for field in source_fields
        if field not in CURATED_SOURCE_FIELDS
    )
    return [truncate(value) if isinstance(value, str) else value for value in values]


def write_workbook(
    experiment: Experiment,
    records: list[dict[str, Any]],
    source_fields: list[str],
    translations: dict[str, str],
    diagnoses: dict[str, str],
) -> None:
    workbook = Workbook(write_only=True)
    workbook.properties.creator = "OpenViking PersonaMem-v2 analysis"
    workbook.properties.title = f"{experiment.label} error analysis"
    columns = workbook_columns(source_fields)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    header_alignment = Alignment(vertical="center", wrap_text=True)
    body_alignment = Alignment(vertical="top", wrap_text=True)

    for classification, sheet_name in SHEET_NAMES.items():
        sheet = workbook.create_sheet(sheet_name)
        sheet.freeze_panes = "A2"
        class_records = [
            record for record in records if record["classification"] == classification
        ]
        sheet.auto_filter.ref = (
            f"A1:{get_column_letter(len(columns))}{len(class_records) + 1}"
        )
        for index, (_, width) in enumerate(columns, start=1):
            sheet.column_dimensions[get_column_letter(index)].width = width

        header = []
        for title, _ in columns:
            cell = WriteOnlyCell(sheet, value=title)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment
            header.append(cell)
        sheet.append(header)

        for record in class_records:
            cells = []
            for value in workbook_row(record, translations, diagnoses, source_fields):
                cell = WriteOnlyCell(sheet, value=value)
                cell.alignment = body_alignment
                cells.append(cell)
            sheet.append(cells)

    experiment.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = experiment.output_path.with_suffix(".tmp.xlsx")
    workbook.save(temporary)
    temporary.replace(experiment.output_path)


def verify_workbook(path: Path, expected: Counter[str], expected_columns: int) -> None:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames != list(SHEET_NAMES.values()):
            raise ValueError(f"unexpected sheet names in {path}: {workbook.sheetnames}")
        for classification, sheet_name in SHEET_NAMES.items():
            sheet = workbook[sheet_name]
            rows = sheet.iter_rows()
            header_row = next(rows, ())
            headers = [cell.value for cell in header_row]
            actual_rows = sum(1 for _ in rows)
            if actual_rows != expected[classification]:
                raise ValueError(
                    f"{sheet_name} row mismatch: {actual_rows} != "
                    f"{expected[classification]}"
                )
            if len(headers) != expected_columns:
                raise ValueError(
                    f"{sheet_name} column mismatch: {len(headers)} != "
                    f"{expected_columns}"
                )
            if len(headers) != len(set(headers)):
                raise ValueError(f"duplicate headers in {sheet_name}")
    finally:
        workbook.close()


async def run(args: argparse.Namespace) -> None:
    selected_keys = list(EXPERIMENTS) if args.experiment == "both" else [args.experiment]
    experiments = [
        Experiment(
            key=key,
            label=str(EXPERIMENTS[key]["label"]),
            csv_path=Path(EXPERIMENTS[key]["csv"]),
            user_prefix=str(EXPERIMENTS[key]["user_prefix"]),
            output_path=Path(EXPERIMENTS[key]["output"]),
        )
        for key in selected_keys
    ]

    prepared: list[
        tuple[Experiment, list[dict[str, Any]], list[str], dict[str, int]]
    ] = []
    all_records: list[dict[str, Any]] = []
    for experiment in experiments:
        started = time.monotonic()
        records, source_fields, counts = classify_experiment(
            experiment,
            threshold=args.coverage_threshold,
            limit=args.limit,
        )
        prepared.append((experiment, records, source_fields, counts))
        all_records.extend(records)
        print(
            f"Classified {experiment.label}: {json.dumps(counts, ensure_ascii=False)} "
            f"in {time.monotonic() - started:.1f}s",
            flush=True,
        )

    translations: dict[str, str] = {}
    diagnoses: dict[str, str] = {}
    cache = AICache(args.cache)
    clients: list[Any] = []
    try:
        if not args.skip_ai:
            translation_model = (
                GPT_4O_MINI_MODEL
                if args.translation_model == "gpt-4o-mini"
                else GPT_5_5_MODEL
            )
            diagnosis_model = (
                GPT_4O_MINI_MODEL
                if args.diagnosis_model == "gpt-4o-mini"
                else GPT_5_5_MODEL
            )
            for record in all_records:
                record["diagnosis_model"] = diagnosis_model

            translation_key = load_gpt_key(translation_model)
            if not translation_key:
                raise RuntimeError(
                    f"{args.translation_model} API key missing; configure it in the "
                    "environment or .openviking/ov.conf"
                )
            from openai import AsyncAzureOpenAI

            translation_client = AsyncAzureOpenAI(
                api_key=translation_key,
                api_version=ES_AZURE_API_VERSION,
                azure_endpoint=ES_AZURE_ENDPOINT,
                default_headers={
                    "X-TT-LOGID": f"personamem-error-workbook-{uuid.uuid4().hex}"
                },
                timeout=args.timeout,
                max_retries=0,
            )
            clients.append(translation_client)
            translations = await fill_translations(
                translation_client,
                cache,
                all_records,
                model=translation_model,
                parallel=args.parallel,
                retry_attempts=args.retry_attempts,
            )

            diagnosis_key = load_gpt_key(diagnosis_model)
            if not diagnosis_key:
                raise RuntimeError(
                    f"{args.diagnosis_model} API key missing; configure it in the "
                    "environment or .openviking/ov.conf"
                )
            diagnosis_client = AsyncAzureOpenAI(
                api_key=diagnosis_key,
                api_version=ES_AZURE_API_VERSION,
                azure_endpoint=ES_AZURE_ENDPOINT,
                default_headers={
                    "X-TT-LOGID": f"personamem-error-workbook-{uuid.uuid4().hex}"
                },
                timeout=args.timeout,
                max_retries=0,
            )
            clients.append(diagnosis_client)
            diagnoses = await fill_diagnoses(
                diagnosis_client,
                cache,
                all_records,
                model=diagnosis_model,
                parallel=args.parallel,
                retry_attempts=args.retry_attempts,
            )

        for experiment, records, source_fields, counts in prepared:
            print(f"Writing {experiment.output_path.name}...", flush=True)
            write_workbook(
                experiment,
                records,
                source_fields,
                translations,
                diagnoses,
            )
            expected = Counter(record["classification"] for record in records)
            verify_workbook(
                experiment.output_path,
                expected,
                len(workbook_columns(source_fields)),
            )
            print(
                json.dumps(
                    {
                        "output": str(experiment.output_path),
                        "bytes": experiment.output_path.stat().st_size,
                        "sheets": {
                            SHEET_NAMES[key]: expected[key] for key in SHEET_NAMES
                        },
                        "evaluation_errors_excluded": counts.get(
                            "evaluation_errors", 0
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        for client in clients:
            await client.close()
        cache.close()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
