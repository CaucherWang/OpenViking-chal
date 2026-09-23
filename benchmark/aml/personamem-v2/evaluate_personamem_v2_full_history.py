#!/usr/bin/env python3
"""Evaluate PersonaMem-v2 with the complete chat history and no memory system.

For every benchmark row, this baseline sends the original history messages,
the official user query, and (in MCQ mode) the official option instruction to
the answer model. It does not ingest, search, or retrieve memories.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
AML_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = AML_ROOT.parents[1]
DEFAULT_DATASET_ROOT = AML_ROOT / "data" / "personamem_v2"
DEFAULT_BENCHMARK_FILE = DEFAULT_DATASET_ROOT / "benchmark" / "text" / "benchmark.csv"
DEFAULT_OV_CONFIG = REPOSITORY_ROOT / ".openviking" / "ov.conf"
RECALL_INSTRUCTION = (
    " Please recall my related preferences from our conversation history "
    "to give personalized responses."
)
SUMMARY_CATEGORIES = (
    "persona_id",
    "topic_query",
    "topic_preference",
    "conversation_scenario",
    "pref_type",
    "who",
    "updated",
    "sensitive_info",
    "distance_from_related_snippet_to_query_32k",
    "distance_from_related_snippet_to_query_128k",
    "num_persona_relevant_tokens_128k",
)


def parse_user_query(raw_value: str) -> dict[str, str]:
    """Parse the official user_query field and append its recall instruction."""
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            parsed = {"role": "user", "content": str(raw_value).strip("\"'")}

    if isinstance(parsed, dict):
        role = str(parsed.get("role") or "user")
        content = str(parsed.get("content", parsed.get("text", "")) or "")
    else:
        role = "user"
        content = str(parsed)
    if content and RECALL_INSTRUCTION.strip() not in content:
        content += RECALL_INSTRUCTION
    return {"role": role, "content": content}


def parse_incorrect_answers(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(value) for value in parsed]


def create_mcq_options(
    correct_answer: str,
    incorrect_answers: list[str],
    seed: int | None = None,
) -> tuple[str, dict[str, str]]:
    """Match the official PersonaMem-v2 option shuffle and instruction."""
    options = [correct_answer] + incorrect_answers
    random.Random(seed).shuffle(options)

    option_mapping: dict[str, str] = {}
    option_parts = []
    for index, option in enumerate(options):
        letter = chr(65 + index)
        option_mapping[letter] = option
        option_parts.append(f"{letter}. {option}")

    instruction = (
        "Please choose the best answer from the following options:\n\n"
        + "\n".join(option_parts)
        + "\n\nThink step by step about which answer best fits the user's query and conversation "
        "context. Provide your reasoning first, then give your final answer as "
        "'Final Answer: [Letter]'"
    )
    return instruction, option_mapping


def extract_final_answer(response: str) -> str:
    """Match the official final-answer extraction patterns."""
    if not response:
        return ""
    patterns = (
        r"\$\\boxed\{([A-Z])\}\$",
        r"\\boxed\{([A-Z])\}",
        r"Final Answer:\s*([A-Z])",
        r"final answer:\s*([A-Z])",
        r"Answer:\s*([A-Z])",
        r"answer:\s*([A-Z])",
        r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"final answer is\s*([A-Z])",
        r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"the answer is\s*([A-Z])",
        r"\b([A-Z])\.\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


def check_mcq_correctness(
    predicted_answer: str,
    correct_answer: str,
    option_mapping: dict[str, str],
) -> bool:
    if not predicted_answer or not option_mapping:
        return False
    return option_mapping.get(predicted_answer.upper(), "") == correct_answer


def read_benchmark_rows(
    path: Path,
    *,
    limit: int | None,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"benchmark has no CSV header: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if limit is not None:
        rows = rows[:limit]
    return fieldnames, rows


def resolve_history_path(
    link: str,
    *,
    dataset_root: Path,
    benchmark_file: Path,
) -> Path:
    candidate = Path(link).expanduser()
    candidates = (
        candidate,
        dataset_root / candidate,
        benchmark_file.parent / candidate,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()

    matches = list((dataset_root / "data").rglob(candidate.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(f"history link is ambiguous under {dataset_root}: {link}")
    raise FileNotFoundError(f"history file not found: {link}")


class HistoryLoader:
    """Small LRU cache for histories shared by adjacent benchmark rows."""

    def __init__(self, max_entries: int = 2) -> None:
        self.max_entries = max_entries
        self.cache: OrderedDict[Path, list[dict[str, str]]] = OrderedDict()

    def load(self, path: Path) -> list[dict[str, str]]:
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path)
            return cached

        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            messages = raw.get("conversations") or raw.get("chat_history")
            if messages is None:
                messages = next(
                    (value for value in raw.values() if isinstance(value, list)),
                    [],
                )
        else:
            messages = raw
        if not isinstance(messages, list):
            raise TypeError(f"history messages are not a list: {path}")

        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"history message {index} is not an object: {path}")
            role = message.get("role")
            content = message.get("content")
            if not role or content is None:
                raise ValueError(f"history message {index} lacks role or content: {path}")
            normalized.append({"role": str(role), "content": str(content)})
        if not normalized:
            raise ValueError(f"history contains no messages: {path}")

        self.cache[path] = normalized
        self.cache.move_to_end(path)
        while len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
        return normalized


def full_history_messages(
    history: list[dict[str, str]],
    user_query: dict[str, str],
    *,
    mcq_instruction: str | None = None,
) -> list[dict[str, str]]:
    """Build the same message layout as the official full-history evaluator."""
    messages = list(history)
    messages.append(user_query)
    if mcq_instruction is not None:
        messages.append({"role": "system", "content": mcq_instruction})
    return messages


def load_answer_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    vlm = raw.get("vlm") if isinstance(raw, dict) else None
    if not isinstance(vlm, dict):
        return {}
    return {
        "model": str(vlm.get("model") or ""),
        "api_key": str(vlm.get("api_key") or ""),
        "api_base": str(vlm.get("api_base") or ""),
    }


def create_answer_client(args: argparse.Namespace) -> Any:
    from volcenginesdkarkruntime import AsyncArk

    return AsyncArk(
        base_url=args.answer_base_url,
        api_key=args.answer_api_key,
        timeout=args.timeout,
        max_retries=0,
    )


def response_output_text(response: Any) -> str:
    output_text = [
        str(content.text)
        for item in response.output
        if item.type == "message"
        for content in item.content
        if content.type == "output_text" and content.text
    ]
    answer = "\n".join(output_text).strip()
    if not answer:
        raise TypeError("answer model returned no output_text")
    return answer


async def answer_with_model(
    client: Any,
    *,
    args: argparse.Namespace,
    messages: list[dict[str, str]],
) -> str:
    for attempt in range(1, args.answer_retry_attempts + 1):
        try:
            response = await client.responses.create(
                model=args.answer_model,
                input=messages,
            )
            return response_output_text(response)
        except Exception as exc:
            print(
                f"[answer attempt {attempt}/{args.answer_retry_attempts}] "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt == args.answer_retry_attempts:
                raise
            await asyncio.sleep(args.retry_delay * (2 ** (attempt - 1)))
    raise AssertionError("answer retry loop exited unexpectedly")


def mode_columns(eval_mode: str, size: str) -> list[str]:
    columns = [
        f"history_file_{size}",
        f"history_message_count_{size}",
        "evaluation_error",
    ]
    if eval_mode in {"mcq", "both"}:
        columns.extend(
            [
                f"input_message_count_mcq_{size}",
                f"option_mapping_{size}",
                f"correct_mcq_option_{size}",
                f"model_response_mcq_{size}",
                f"predicted_answer_mcq_{size}",
                f"is_correct_mcq_{size}",
            ]
        )
    if eval_mode in {"generative", "both"}:
        columns.extend(
            [
                f"input_message_count_openended_{size}",
                f"model_response_openended_{size}",
                f"is_correct_openended_{size}",
            ]
        )
    return columns


async def evaluate_row(
    row: dict[str, str],
    *,
    args: argparse.Namespace,
    answer_client: Any,
    history_loader: HistoryLoader,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    output: dict[str, Any] = dict(row)
    output.update(dict.fromkeys(mode_columns(args.eval_mode, args.size), ""))
    persona_id = str(row.get("persona_id") or "")
    user_query = parse_user_query(str(row.get("user_query") or ""))
    if not user_query["content"]:
        output["evaluation_error"] = "empty user query"
        return output

    try:
        async with semaphore:
            history_link = str(row.get(f"chat_history_{args.size}_link") or "")
            if not history_link:
                raise KeyError(f"missing chat_history_{args.size}_link")
            history_path = resolve_history_path(
                history_link,
                dataset_root=args.dataset_root,
                benchmark_file=args.benchmark_file,
            )
            history = history_loader.load(history_path)
            output[f"history_file_{args.size}"] = str(history_path)
            output[f"history_message_count_{args.size}"] = len(history)

            if args.eval_mode in {"mcq", "both"}:
                row_seed = hash(f"{persona_id}_{user_query['content']}") % 2**32
                instruction, option_mapping = create_mcq_options(
                    str(row.get("correct_answer") or ""),
                    parse_incorrect_answers(str(row.get("incorrect_answers") or "")),
                    seed=row_seed,
                )
                correct_option = next(
                    (
                        letter
                        for letter, answer in option_mapping.items()
                        if answer == row.get("correct_answer")
                    ),
                    "N/A",
                )
                messages = full_history_messages(
                    history,
                    user_query,
                    mcq_instruction=instruction,
                )
                output[f"input_message_count_mcq_{args.size}"] = len(messages)
                output[f"option_mapping_{args.size}"] = json.dumps(
                    option_mapping,
                    ensure_ascii=False,
                )
                output[f"correct_mcq_option_{args.size}"] = correct_option
                response = await answer_with_model(
                    answer_client,
                    args=args,
                    messages=messages,
                )
                predicted = extract_final_answer(response)
                output[f"model_response_mcq_{args.size}"] = response
                output[f"predicted_answer_mcq_{args.size}"] = predicted
                output[f"is_correct_mcq_{args.size}"] = str(
                    check_mcq_correctness(
                        predicted,
                        str(row.get("correct_answer") or ""),
                        option_mapping,
                    )
                )

            if args.eval_mode in {"generative", "both"}:
                messages = full_history_messages(history, user_query)
                output[f"input_message_count_openended_{args.size}"] = len(messages)
                output[f"model_response_openended_{args.size}"] = await answer_with_model(
                    answer_client,
                    args=args,
                    messages=messages,
                )
                # The official inference pass leaves open-ended correctness blank.
                output[f"is_correct_openended_{args.size}"] = ""
    except Exception as exc:
        output["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        print(
            json.dumps(
                {
                    "error": output["evaluation_error"],
                    "persona_id": persona_id,
                    "history_size": args.size,
                    "query": user_query["content"][:300],
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    return output


def category_label(category: str, value: str) -> str:
    if "distance_from_related_snippet_to_query" not in category:
        return value
    try:
        distance = int(value)
    except (TypeError, ValueError):
        return value
    start = (distance // 1024) * 1024
    return f"{start}-{start + 1023}"


def build_summary(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    correctness_column = f"is_correct_mcq_{args.size}"
    scored = [row for row in rows if row.get(correctness_column) in {"True", "False"}]
    correct = sum(row[correctness_column] == "True" for row in scored)
    summary: dict[str, Any] = {
        "evaluation_type": "full_history_no_memory",
        "benchmark_file": str(args.benchmark_file),
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "size": args.size,
        "eval_mode": args.eval_mode,
        "answer_model": args.answer_model,
        "rows": len(rows),
        "errors": sum(bool(row.get("evaluation_error")) for row in rows),
        "mcq_scored": len(scored),
        "mcq_correct": correct,
        "mcq_accuracy": correct / len(scored) if scored else None,
    }
    if not scored:
        return summary

    breakdown: dict[str, Any] = {}
    for category in SUMMARY_CATEGORIES:
        stats: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in scored:
            label = category_label(category, str(row.get(category, "")))
            stats[label]["total"] += 1
            if row[correctness_column] == "True":
                stats[label]["correct"] += 1
        breakdown[category] = {
            label: {
                **counts,
                "accuracy": counts["correct"] / counts["total"],
            }
            for label, counts in sorted(stats.items())
        }
    summary["category_breakdown"] = breakdown
    return summary


def report_progress(
    current: int,
    total: int,
    errors: int,
    *,
    started_at: float,
) -> None:
    current_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_seconds = int(time.monotonic() - started_at)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if sys.stderr.isatty():
        width = 30
        completed = width * current // max(total, 1)
        bar = "#" * completed + "-" * (width - completed)
        print(
            f"\r{current_time} [{bar}] {current}/{total} rows | "
            f"errors {errors} | elapsed {elapsed}",
            end="\n" if current == total else "",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "time": current_time,
                    "elapsed": elapsed,
                    "elapsed_seconds": elapsed_seconds,
                    "processed": current,
                    "total": total,
                    "errors": errors,
                }
            ),
            flush=True,
        )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.monotonic()
    source_fields, rows = read_benchmark_rows(
        args.benchmark_file,
        limit=args.limit,
    )
    if not rows:
        raise ValueError("benchmark contains no query rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = source_fields + [
        column for column in mode_columns(args.eval_mode, args.size) if column not in source_fields
    ]
    semaphore = asyncio.Semaphore(args.parallel)
    history_loader = HistoryLoader()
    completed_rows: list[tuple[int, dict[str, Any]]] = []
    errors = 0
    answer_client = create_answer_client(args)
    try:

        async def evaluate_indexed_row(
            query_index: int,
            row: dict[str, str],
        ) -> tuple[int, dict[str, Any]]:
            output = await evaluate_row(
                row,
                args=args,
                answer_client=answer_client,
                history_loader=history_loader,
                semaphore=semaphore,
            )
            return query_index, output

        history_column = f"chat_history_{args.size}_link"
        indexed_rows = sorted(
            enumerate(rows),
            key=lambda item: (item[1].get(history_column, ""), item[0]),
        )
        tasks = [
            asyncio.create_task(evaluate_indexed_row(index, row)) for index, row in indexed_rows
        ]
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            for current, task in enumerate(asyncio.as_completed(tasks), start=1):
                query_index, output = await task
                writer.writerow(output)
                handle.flush()
                completed_rows.append((query_index, output))
                errors += bool(output.get("evaluation_error"))
                report_progress(
                    current,
                    len(tasks),
                    errors,
                    started_at=started_at,
                )
    finally:
        await answer_client.close()

    ordered_rows = [output for _, output in sorted(completed_rows, key=lambda item: item[0])]
    sorted_output = args.output.with_name(f".{args.output.name}.sorted.tmp")
    with sorted_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(ordered_rows)
    sorted_output.replace(args.output)

    summary = build_summary(ordered_rows, args=args)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def default_output_path(size: str, eval_mode: str) -> Path:
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    return (
        AML_ROOT
        / "data"
        / "results"
        / f"personamem_v2_full_history_{eval_mode}_{size}_{timestamp}.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--size", choices=["32k", "128k"], default="32k")
    parser.add_argument(
        "--eval-mode",
        choices=["mcq", "generative", "both"],
        default="mcq",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--answer-retry-attempts",
        type=int,
        default=5,
        help="Maximum attempts for each answer-model request.",
    )
    parser.add_argument("--retry-delay", type=float, default=0.5)
    parser.add_argument("--ov-config", type=Path, default=DEFAULT_OV_CONFIG)
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-base-url")
    parser.add_argument("--answer-api-key")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for name in ("parallel", "answer_retry_attempts"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.timeout <= 0 or args.retry_delay < 0:
        parser.error("--timeout must be positive and --retry-delay must be non-negative")
    if not args.benchmark_file.is_file():
        parser.error(f"benchmark file does not exist: {args.benchmark_file}")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")

    answer_config = load_answer_config(args.ov_config)
    args.answer_model = (
        args.answer_model or os.getenv("DOUBAO_VLM_MODEL") or answer_config.get("model", "")
    )
    args.answer_base_url = (
        args.answer_base_url or os.getenv("ARK_API_BASE") or answer_config.get("api_base", "")
    )
    args.answer_api_key = (
        args.answer_api_key or os.getenv("ARK_API_KEY") or answer_config.get("api_key", "")
    )
    missing = [
        name
        for name, value in (
            ("answer model", args.answer_model),
            ("answer base URL", args.answer_base_url),
            ("answer API key", args.answer_api_key),
        )
        if not value
    ]
    if missing:
        parser.error(
            f"missing {', '.join(missing)}; configure .openviking/ov.conf "
            "or pass the corresponding --answer-* option"
        )

    args.output = args.output or default_output_path(args.size, args.eval_mode)
    args.summary_output = args.summary_output or args.output.with_name(
        args.output.stem + "_summary.json"
    )
    return args


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
