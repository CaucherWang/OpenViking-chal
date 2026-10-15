#!/usr/bin/env python3
"""Evaluate pre-ingested PersonaMem-v2 histories with OpenViking search.

The default run evaluates every query row in the official CSV in MCQ mode.
Use ``--limit 1`` for one query, or ``--retrieval-only`` to inspect Search
output without calling the answer model.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

if __package__:
    from .utils import (
        AML_ROOT,
        DEFAULT_BENCHMARK_FILE,
        DEFAULT_OV_CONFIG,
        ES_AZURE_API_VERSION,
        ES_AZURE_ENDPOINT,
        EXPERIMENT_NAME_RE,
        GPT_4O_MINI_MODEL,
        GPT_5_5_MODEL,
        AnswerTokenUsage,
        answer_with_model,
        build_mcq_summary,
        check_mcq_correctness,
        create_answer_client,
        create_mcq_options,
        extract_final_answer,
        load_answer_config,
        parse_incorrect_answers,
        parse_user_query,
        post_json,
        read_benchmark_rows,
        report_progress,
        request_answer,
        stable_row_seed,
    )
else:
    from utils import (
        AML_ROOT,
        DEFAULT_BENCHMARK_FILE,
        DEFAULT_OV_CONFIG,
        ES_AZURE_API_VERSION,
        ES_AZURE_ENDPOINT,
        EXPERIMENT_NAME_RE,
        GPT_4O_MINI_MODEL,
        GPT_5_5_MODEL,
        AnswerTokenUsage,
        answer_with_model,
        build_mcq_summary,
        check_mcq_correctness,
        create_answer_client,
        create_mcq_options,
        extract_final_answer,
        load_answer_config,
        parse_incorrect_answers,
        parse_user_query,
        post_json,
        read_benchmark_rows,
        report_progress,
        request_answer,
        stable_row_seed,
    )


@dataclass
class PerformanceMetrics:
    answer_request_attempts: int = 0
    answer_requests_completed: int = 0
    answer_requests_with_token_usage: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def record_answer_request(self) -> None:
        self.answer_request_attempts += 1

    def record_answer_usage(self, usage: AnswerTokenUsage | None) -> None:
        self.answer_requests_completed += 1
        if usage is None:
            return
        self.answer_requests_with_token_usage += 1
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

    def summary(self, *, queries_completed: int, elapsed_seconds: float) -> dict[str, Any]:
        usage_requests = self.answer_requests_with_token_usage
        return {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "queries_completed": queries_completed,
            "benchmark_queries_per_minute": (
                round(queries_completed * 60 / elapsed_seconds, 3) if elapsed_seconds > 0 else None
            ),
            "answer_requests_completed": self.answer_requests_completed,
            "answer_requests_with_token_usage": usage_requests,
            "answer_model_request_attempts": self.answer_request_attempts,
            "answer_model_requests_per_minute": (
                round(self.answer_request_attempts * 60 / elapsed_seconds, 3)
                if elapsed_seconds > 0
                else None
            ),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "average_prompt_tokens_per_request": (
                round(self.total_prompt_tokens / usage_requests, 3) if usage_requests else None
            ),
            "average_completion_tokens_per_request": (
                round(self.total_completion_tokens / usage_requests, 3) if usage_requests else None
            ),
        }


def answer_messages(
    query: str,
    memories: list[dict[str, Any]],
    *,
    mcq_instruction: str | None = None,
) -> list[dict[str, str]]:
    """Replace the official full chat history with retrieved memory contents."""
    messages = [
        {"role": "system", "content": str(memory["content"])}
        for memory in memories
        if memory.get("content")
    ]
    messages.append({"role": "user", "content": query})
    if mcq_instruction is not None:
        messages.append({"role": "system", "content": mcq_instruction})
    return messages


def persona_key(raw_persona_id: str) -> str:
    value = str(raw_persona_id).strip()
    return value if value.startswith("persona") else f"persona{value}"


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
        return list(reader.fieldnames), list(reader)


async def probe_answer_client(client: Any, args: argparse.Namespace) -> None:
    """Fail fast if the configured answer model cannot serve a minimal request."""
    probe_timeout = min(args.timeout, 30.0)
    print(
        json.dumps(
            {
                "answer_client_probe": "started",
                "model": args.answer_model,
                "timeout_seconds": probe_timeout,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )
    try:
        answer = await asyncio.wait_for(
            request_answer(
                client,
                args=args,
                messages=[{"role": "user", "content": "Hi, who are you?"}],
            ),
            timeout=probe_timeout,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"answer client probe timed out after {probe_timeout:g} seconds"
        ) from exc

    print(
        json.dumps(
            {
                "answer_client_probe": "succeeded",
                "model": args.answer_model,
                "response": answer,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


async def search_http(
    client: httpx.AsyncClient,
    *,
    args: argparse.Namespace,
    user_id: str,
    search_prompt: str,
) -> list[dict[str, Any]]:
    headers = {}
    if args.aml_api_key:
        headers["Authorization"] = f"Bearer {args.aml_api_key}"
    payload: dict[str, Any] = {
        "query": search_prompt,
        "user_id": user_id,
        "top_k": args.top_k,
    }
    result = await post_json(
        client,
        args.adapter_url.rstrip("/") + "/search",
        payload,
        headers,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
    )
    memories = result.get("data")
    if not isinstance(memories, list) or any(not isinstance(item, dict) for item in memories):
        raise TypeError("expected /search response data to be a list of objects")
    return memories


def memory_data(hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Convert OpenVikingBackend.find hits to the AML memory result shape."""
    memories = []
    for hit in hits:
        content = hit.get("content") or hit.get("abstract")
        uri = hit.get("uri")
        if isinstance(uri, str) and uri and isinstance(content, str) and content:
            memories.append({"id": uri, "content": content})
        if len(memories) == limit:
            break
    return memories


def mode_columns(eval_mode: str, size: str) -> list[str]:
    columns = ["openviking_user_id", "evaluation_error"]
    if eval_mode in {"mcq", "both"}:
        columns.extend(
            [
                f"search_query_mcq_{size}",
                f"retrieval_query_mcq_{size}",
                f"retrieved_memories_mcq_{size}",
                f"retrieved_count_mcq_{size}",
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
                f"search_query_openended_{size}",
                f"retrieval_query_openended_{size}",
                f"retrieved_memories_openended_{size}",
                f"retrieved_count_openended_{size}",
                f"model_response_openended_{size}",
                f"is_correct_openended_{size}",
            ]
        )
    return columns


async def evaluate_row(
    row: dict[str, str],
    *,
    args: argparse.Namespace,
    client: httpx.AsyncClient,
    answer_client: Any,
    backend: Any,
    semaphore: asyncio.Semaphore,
    metrics: PerformanceMetrics,
) -> dict[str, Any]:
    output: dict[str, Any] = dict(row)
    output.update(dict.fromkeys(mode_columns(args.eval_mode, args.size), ""))
    persona_id = str(row.get("persona_id") or "")
    user_id = f"{args.user_id_prefix}-{persona_key(persona_id)}"
    output["openviking_user_id"] = user_id
    query = parse_user_query(str(row.get("user_query") or ""))["content"]
    if not query:
        output["evaluation_error"] = "empty user query"
        print(
            json.dumps(
                {
                    "error": output["evaluation_error"],
                    "persona_id": persona_id,
                    "user_id": user_id,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return output

    async def run_search(search_prompt: str) -> list[dict[str, Any]]:
        if args.search_mode == "http":
            return await search_http(
                client,
                args=args,
                user_id=user_id,
                search_prompt=search_prompt,
            )
        hits = await backend.find(
            user_id=user_id,
            query=search_prompt,
            limit=args.top_k,
        )
        return memory_data(hits, args.top_k)

    try:
        async with semaphore:
            if args.eval_mode in {"mcq", "both"}:
                row_seed = stable_row_seed(persona_id, query)
                instruction, option_mapping = create_mcq_options(
                    str(row.get("correct_answer") or ""),
                    parse_incorrect_answers(str(row.get("incorrect_answers") or "")),
                    seed=row_seed,
                )
                memories = await run_search(query)
                correct_option = next(
                    (
                        letter
                        for letter, answer in option_mapping.items()
                        if answer == row.get("correct_answer")
                    ),
                    "N/A",
                )
                output[f"search_query_mcq_{args.size}"] = query
                output[f"retrieval_query_mcq_{args.size}"] = query
                output[f"retrieved_memories_mcq_{args.size}"] = json.dumps(
                    memories, ensure_ascii=False
                )
                output[f"retrieved_count_mcq_{args.size}"] = len(memories)
                output[f"option_mapping_{args.size}"] = json.dumps(
                    option_mapping, ensure_ascii=False
                )
                output[f"correct_mcq_option_{args.size}"] = correct_option

                if not args.retrieval_only:
                    response = await answer_with_model(
                        answer_client,
                        args=args,
                        messages=answer_messages(
                            query,
                            memories,
                            mcq_instruction=instruction,
                        ),
                        usage_callback=metrics.record_answer_usage,
                        request_callback=metrics.record_answer_request,
                    )
                    predicted = extract_final_answer(response)
                    is_correct = check_mcq_correctness(
                        predicted,
                        str(row.get("correct_answer") or ""),
                        option_mapping,
                    )
                    output[f"model_response_mcq_{args.size}"] = response
                    output[f"predicted_answer_mcq_{args.size}"] = predicted
                    output[f"is_correct_mcq_{args.size}"] = str(is_correct)

            if args.eval_mode in {"generative", "both"}:
                memories = await run_search(query)
                output[f"search_query_openended_{args.size}"] = query
                output[f"retrieval_query_openended_{args.size}"] = query
                output[f"retrieved_memories_openended_{args.size}"] = json.dumps(
                    memories, ensure_ascii=False
                )
                output[f"retrieved_count_openended_{args.size}"] = len(memories)
                if not args.retrieval_only:
                    output[f"model_response_openended_{args.size}"] = await answer_with_model(
                        answer_client,
                        args=args,
                        messages=answer_messages(query, memories),
                        usage_callback=metrics.record_answer_usage,
                        request_callback=metrics.record_answer_request,
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
                    "user_id": user_id,
                    "query": query[:300],
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
    metrics: PerformanceMetrics | None = None,
    queries_completed: int | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "benchmark_file": str(args.benchmark_file),
        "output": str(args.output),
        "size": args.size,
        "eval_mode": args.eval_mode,
        "search_mode": args.search_mode,
        "experiment_name": args.experiment_name,
        "user_id_prefix": args.user_id_prefix,
        "answer_backend": args.answer_backend,
        "answer_model": args.answer_model,
        "top_k": args.top_k,
        "rows": len(rows),
        "errors": sum(bool(row.get("evaluation_error")) for row in rows),
    }
    summary.update(build_mcq_summary(rows, args.size))
    if metrics is not None and elapsed_seconds is not None:
        summary["performance"] = metrics.summary(
            queries_completed=len(rows) if queries_completed is None else queries_completed,
            elapsed_seconds=elapsed_seconds,
        )
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
        expected_user_id = f"{args.user_id_prefix}-{persona_key(result_row.get('persona_id', ''))}"
        if result_row.get("openviking_user_id", "") != expected_user_id:
            raise ValueError(
                f"result CSV row {index} belongs to a different experiment; "
                f"expected openviking_user_id {expected_user_id!r}"
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

    backend = None
    if args.search_mode == "direct":
        from benchmark.aml.server import AMLSettings, OpenVikingBackend

        backend = OpenVikingBackend(
            AMLSettings(
                openviking_url=args.openviking_url,
                openviking_api_key=args.openviking_api_key or None,
                openviking_account=args.openviking_account or None,
                request_timeout_seconds=args.timeout,
            )
        )

    replacements: dict[int, dict[str, Any]] = {}
    retry_failures = 0
    metrics = PerformanceMetrics()
    if selected_indexes:
        timeout = httpx.Timeout(args.timeout)
        semaphore = asyncio.Semaphore(args.parallel)
        answer_client = create_answer_client(args)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:

                async def evaluate_indexed_row(
                    row_index: int,
                ) -> tuple[int, dict[str, Any]]:
                    output = await evaluate_row(
                        result_rows[row_index],
                        args=args,
                        client=client,
                        answer_client=answer_client,
                        backend=backend,
                        semaphore=semaphore,
                        metrics=metrics,
                    )
                    return row_index, output

                tasks = [
                    asyncio.create_task(evaluate_indexed_row(index)) for index in selected_indexes
                ]
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

    elapsed_seconds = time.monotonic() - started_at
    summary = build_summary(
        merged_rows,
        args=args,
        metrics=metrics,
        queries_completed=len(selected_indexes),
        elapsed_seconds=elapsed_seconds,
    )
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

    backend = None
    if args.search_mode == "direct":
        from benchmark.aml.server import AMLSettings, OpenVikingBackend

        backend = OpenVikingBackend(
            AMLSettings(
                openviking_url=args.openviking_url,
                openviking_api_key=args.openviking_api_key or None,
                openviking_account=args.openviking_account or None,
                request_timeout_seconds=args.timeout,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = source_fields + [
        column for column in mode_columns(args.eval_mode, args.size) if column not in source_fields
    ]
    timeout = httpx.Timeout(args.timeout)
    semaphore = asyncio.Semaphore(args.parallel)
    completed_rows: list[tuple[int, dict[str, Any]]] = []
    errors = 0
    metrics = PerformanceMetrics()
    answer_client = None if args.retrieval_only else create_answer_client(args)
    try:
        # if answer_client is not None:
        #     await probe_answer_client(answer_client, args)
        async with httpx.AsyncClient(timeout=timeout) as client:

            async def evaluate_indexed_row(
                query_index: int,
                row: dict[str, str],
            ) -> tuple[int, dict[str, Any]]:
                output = await evaluate_row(
                    row,
                    args=args,
                    client=client,
                    answer_client=answer_client,
                    backend=backend,
                    semaphore=semaphore,
                    metrics=metrics,
                )
                return query_index, output

            tasks = [
                asyncio.create_task(
                    evaluate_indexed_row(
                        query_index,
                        row,
                    )
                )
                for query_index, row in enumerate(rows)
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
        if answer_client is not None:
            await answer_client.close()

    ordered_rows = [output for _, output in sorted(completed_rows, key=lambda item: item[0])]
    sorted_output = args.output.with_name(f".{args.output.name}.sorted.tmp")
    with sorted_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(ordered_rows)
    sorted_output.replace(args.output)

    elapsed_seconds = time.monotonic() - started_at
    summary = build_summary(
        ordered_rows,
        args=args,
        metrics=metrics,
        queries_completed=len(ordered_rows),
        elapsed_seconds=elapsed_seconds,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def default_output_path(size: str, eval_mode: str) -> Path:
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    return (
        AML_ROOT / "data" / "results" / f"personamem_v2_search_{eval_mode}_{size}_{timestamp}.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument("--size", choices=["32k", "128k"], default="32k")
    parser.add_argument("--eval-mode", choices=["mcq", "generative", "both"], default="mcq")
    parser.add_argument("--search-mode", choices=["http", "direct"], default="direct")
    parser.add_argument("--adapter-url", default="http://127.0.0.1:8088")
    parser.add_argument("--aml-api-key", default=os.getenv("AML_API_KEY", ""))
    parser.add_argument(
        "--openviking-url",
        default=os.getenv("OPENVIKING_URL", "http://127.0.0.1:1933"),
    )
    parser.add_argument("--openviking-api-key", default=os.getenv("OPENVIKING_API_KEY", ""))
    parser.add_argument("--openviking-account", default=os.getenv("OPENVIKING_ACCOUNT", ""))
    parser.add_argument(
        "--user-id-prefix",
        help="Defaults to personamem-v2-<size>, matching the ingest script convention.",
    )
    parser.add_argument(
        "--experiment-name",
        help="Read memories from the matching ingest experiment user-ID namespace.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum benchmark rows, or maximum failed rows in --retry-errors mode.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument(
        "--answer-retry-attempts",
        type=int,
        default=5,
        help="Maximum attempts for each downstream answer-model request.",
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
    parser.add_argument("--answer-max-tokens", type=int, default=5000)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run Search and save memories without calling the answer model.",
    )
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
    for name in (
        "top_k",
        "parallel",
        "retry_attempts",
        "answer_retry_attempts",
        "answer_max_tokens",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.timeout <= 0 or args.retry_delay < 0:
        parser.error("--timeout must be positive and --retry-delay must be non-negative")
    if not args.benchmark_file.is_file():
        parser.error(f"benchmark file does not exist: {args.benchmark_file}")
    if args.retry_errors:
        if args.output is None:
            parser.error("--retry-errors requires an explicit --output CSV")
        if not args.output.is_file():
            parser.error(f"retry result CSV does not exist: {args.output}")
        if args.retrieval_only:
            parser.error("--retry-errors cannot be combined with --retrieval-only")
    args.user_id_prefix = args.user_id_prefix or f"personamem-v2-{args.size}"
    if args.experiment_name:
        if not EXPERIMENT_NAME_RE.fullmatch(args.experiment_name):
            parser.error(
                "--experiment-name must start with an alphanumeric character and contain "
                "only letters, digits, underscores, dots, or hyphens"
            )
        args.user_id_prefix = f"{args.user_id_prefix}-{args.experiment_name}"

    answer_config = load_answer_config(args.ov_config)
    if args.answer_backend == "seed":
        args.answer_model = (
            args.answer_model or os.getenv("DOUBAO_VLM_MODEL") or answer_config.get("model", "")
        )
        args.answer_base_url = (
            args.answer_base_url or os.getenv("ARK_API_BASE") or answer_config.get("api_base", "")
        )
        args.answer_api_key = (
            args.answer_api_key or os.getenv("ARK_API_KEY") or answer_config.get("api_key", "")
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
    if not args.retrieval_only:
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
            parser.error(f"missing {', '.join(missing)}; {source} or use --retrieval-only")

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
