#!/usr/bin/env python3
"""Evaluate PersonaMem-v2 with the complete chat history and no memory system.

For every benchmark row, this baseline sends the original history messages,
the official user query, and (in MCQ mode) the official option instruction to
the answer model. It does not ingest, search, or retrieve memories.

Use ``--retry-errors --output <existing.csv>`` after a completed evaluation to
rerun failed rows and atomically update the CSV and summary.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__:
    from .utils import (
        AML_ROOT,
        DEFAULT_BENCHMARK_FILE,
        DEFAULT_DATASET_ROOT,
        DEFAULT_OV_CONFIG,
        ES_AZURE_API_VERSION,
        ES_AZURE_ENDPOINT,
        GPT_4O_MINI_MODEL,
        GPT_5_5_MODEL,
        answer_with_model,
        build_mcq_summary,
        check_mcq_correctness,
        create_answer_client,
        create_mcq_options,
        extract_final_answer,
        load_answer_config,
        parse_incorrect_answers,
        parse_user_query,
        read_benchmark_rows,
        report_progress,
        stable_row_seed,
    )
else:
    from utils import (
        AML_ROOT,
        DEFAULT_BENCHMARK_FILE,
        DEFAULT_DATASET_ROOT,
        DEFAULT_OV_CONFIG,
        ES_AZURE_API_VERSION,
        ES_AZURE_ENDPOINT,
        GPT_4O_MINI_MODEL,
        GPT_5_5_MODEL,
        answer_with_model,
        build_mcq_summary,
        check_mcq_correctness,
        create_answer_client,
        create_mcq_options,
        extract_final_answer,
        load_answer_config,
        parse_incorrect_answers,
        parse_user_query,
        read_benchmark_rows,
        report_progress,
        stable_row_seed,
    )


def read_result_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"result CSV has no header: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    incomplete = [
        index + 2
        for index, row in enumerate(rows)
        if None in row or any(value is None for value in row.values())
    ]
    if incomplete:
        preview = ", ".join(str(line) for line in incomplete[:5])
        raise ValueError(
            f"result CSV contains incomplete rows at line(s) {preview}; "
            "wait for the original evaluation to finish"
        )
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
                row_seed = stable_row_seed(persona_id, user_query["content"])
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


def build_summary(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "evaluation_type": "full_history_no_memory",
        "benchmark_file": str(args.benchmark_file),
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "size": args.size,
        "eval_mode": args.eval_mode,
        "answer_backend": args.answer_backend,
        "answer_model": args.answer_model,
        "rows": len(rows),
        "errors": sum(bool(row.get("evaluation_error")) for row in rows),
    }
    summary.update(build_mcq_summary(rows, args.size))
    return summary


def validate_retry_source(
    result_fields: list[str],
    result_rows: list[dict[str, str]],
    benchmark_rows: list[dict[str, str]],
    *,
    args: argparse.Namespace,
) -> None:
    required_fields = set(mode_columns(args.eval_mode, args.size))
    missing_fields = sorted(required_fields.difference(result_fields))
    if missing_fields:
        raise ValueError(
            "result CSV is incompatible with the requested size/eval mode; "
            f"missing columns: {', '.join(missing_fields)}"
        )
    has_mcq = f"model_response_mcq_{args.size}" in result_fields
    has_openended = f"model_response_openended_{args.size}" in result_fields
    detected_mode = (
        "both"
        if has_mcq and has_openended
        else "mcq"
        if has_mcq
        else "generative"
        if has_openended
        else ""
    )
    if detected_mode != args.eval_mode:
        raise ValueError(
            f"result CSV eval mode is {detected_mode or 'unknown'}, but "
            f"--eval-mode is {args.eval_mode}"
        )
    if len(result_rows) != len(benchmark_rows):
        raise ValueError(
            f"result CSV has {len(result_rows)} rows but benchmark has "
            f"{len(benchmark_rows)}; wait for the original evaluation to finish"
        )

    identity_fields = (
        "persona_id",
        f"chat_history_{args.size}_link",
        "user_query",
        "correct_answer",
    )
    for index, (result_row, benchmark_row) in enumerate(
        zip(result_rows, benchmark_rows, strict=True),
        start=1,
    ):
        if any(
            result_row.get(field, "") != benchmark_row.get(field, "") for field in identity_fields
        ):
            raise ValueError(
                f"result CSV row {index} does not match benchmark order; "
                "wait for the original evaluation to finish or use its finalized CSV"
            )


async def retry_error_rows(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.monotonic()
    result_fields, result_rows = read_result_rows(args.output)
    _, benchmark_rows = read_benchmark_rows(args.benchmark_file, limit=None)
    validate_retry_source(
        result_fields,
        result_rows,
        benchmark_rows,
        args=args,
    )

    error_indexes = [
        index for index, row in enumerate(result_rows) if row.get("evaluation_error", "").strip()
    ]
    selected_indexes = error_indexes[: args.limit]
    print(
        json.dumps(
            {
                "retry_errors": "started",
                "output": str(args.output),
                "rows": len(result_rows),
                "existing_errors": len(error_indexes),
                "selected_for_retry": len(selected_indexes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    replacements: dict[int, dict[str, Any]] = {}
    retry_failures = 0
    if selected_indexes:
        semaphore = asyncio.Semaphore(args.parallel)
        history_loader = HistoryLoader()
        answer_client = create_answer_client(args)
        try:

            async def evaluate_indexed_row(
                row_index: int,
            ) -> tuple[int, dict[str, Any]]:
                output = await evaluate_row(
                    result_rows[row_index],
                    args=args,
                    answer_client=answer_client,
                    history_loader=history_loader,
                    semaphore=semaphore,
                )
                return row_index, output

            history_column = f"chat_history_{args.size}_link"
            ordered_indexes = sorted(
                selected_indexes,
                key=lambda index: (
                    result_rows[index].get(history_column, ""),
                    index,
                ),
            )
            tasks = [asyncio.create_task(evaluate_indexed_row(index)) for index in ordered_indexes]
            for current, task in enumerate(asyncio.as_completed(tasks), start=1):
                row_index, output = await task
                replacements[row_index] = output
                retry_failures += bool(output.get("evaluation_error"))
                report_progress(
                    current,
                    len(tasks),
                    retry_failures,
                    started_at=started_at,
                )
        finally:
            await answer_client.close()

    merged_rows = [replacements.get(index, row) for index, row in enumerate(result_rows)]
    retry_output = args.output.with_name(f".{args.output.name}.retry.tmp")
    with retry_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(merged_rows)
    retry_output.replace(args.output)

    summary = build_summary(merged_rows, args=args)
    summary["retry"] = {
        "existing_errors": len(error_indexes),
        "attempted": len(selected_indexes),
        "recovered": len(selected_indexes) - retry_failures,
        "failed_again": retry_failures,
        "remaining_errors": sum(
            bool(row.get("evaluation_error", "").strip()) for row in merged_rows
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.retry_errors:
        return await retry_error_rows(args)

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
    parser.add_argument(
        "--limit",
        type=int,
        help=("Maximum benchmark rows, or maximum failed rows in --retry-errors mode."),
    )
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
    parser.add_argument(
        "--answer-backend",
        choices=["seed", "gpt-4o-mini", "gpt-5.5"],
        default=os.getenv("PERSONAMEM_ANSWER_BACKEND", "seed"),
    )
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-base-url")
    parser.add_argument("--answer-api-key")
    parser.add_argument("--answer-log-id")
    parser.add_argument("--answer-max-tokens", type=int, default=500)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Retry rows whose evaluation_error is non-empty in the existing "
            "--output CSV, then atomically update that CSV and its summary. "
            "Run only after the original evaluation has finished."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for name in ("parallel", "answer_retry_attempts", "answer_max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.timeout <= 0 or args.retry_delay < 0:
        parser.error("--timeout must be positive and --retry-delay must be non-negative")
    if not args.benchmark_file.is_file():
        parser.error(f"benchmark file does not exist: {args.benchmark_file}")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if args.retry_errors:
        if args.output is None:
            parser.error("--retry-errors requires an explicit --output CSV")
        if not args.output.is_file():
            parser.error(f"retry result CSV does not exist: {args.output}")

    answer_config = load_answer_config(args.ov_config)
    if args.answer_backend == "seed":
        args.answer_model = (
            args.answer_model or os.getenv("DOUBAO_VLM_MODEL") or answer_config.get("model", "")
        )
        args.answer_base_url = (
            args.answer_base_url or os.getenv("ARK_API_BASE") or answer_config.get("api_base", "")
        )
        args.answer_api_key = (
            args.answer_api_key
            or os.getenv("ARK_API_KEY")
            or answer_config.get("893_api_key", "")
            or answer_config.get("api_key", "")
        )
        args.answer_api_version = None
        args.answer_log_id = None
    elif args.answer_backend == "gpt-4o-mini":
        args.answer_model = GPT_4O_MINI_MODEL
        args.answer_base_url = ES_AZURE_ENDPOINT
        args.answer_api_key = (
            args.answer_api_key
            or answer_config.get("gpt_4o_mini_key", "")
            or os.getenv("GPT_4O_MINI_API_KEY", "")
        )
        args.answer_api_version = ES_AZURE_API_VERSION
        args.answer_log_id = (
            args.answer_log_id
            or os.getenv("ES_MODEL_LOG_ID")
            or f"personamem-v2-{uuid.uuid4().hex}"
        )
    else:
        args.answer_model = GPT_5_5_MODEL
        args.answer_base_url = ES_AZURE_ENDPOINT
        args.answer_api_key = (
            args.answer_api_key
            or answer_config.get("gpt_5_5_key", "")
            or os.getenv("GPT_5_5_API_KEY", "")
        )
        args.answer_api_version = ES_AZURE_API_VERSION
        args.answer_log_id = (
            args.answer_log_id
            or os.getenv("ES_MODEL_LOG_ID")
            or f"personamem-v2-{uuid.uuid4().hex}"
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
        if args.answer_backend == "seed":
            source = "configure .openviking/ov.conf"
        elif args.answer_backend == "gpt-4o-mini":
            source = "configure vlm.gpt_4o_mini_key in .openviking/ov.conf"
        else:
            source = "set GPT_5_5_API_KEY or configure vlm.gpt_5_5_key"
        parser.error(
            f"missing {', '.join(missing)}; {source} or pass the corresponding --answer-* option"
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
