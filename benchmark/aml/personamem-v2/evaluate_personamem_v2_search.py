#!/usr/bin/env python3
"""Evaluate pre-ingested PersonaMem-v2 histories with OpenViking search.

The default run evaluates every query row in the official CSV in MCQ mode.
Use ``--limit 1`` for one query, or ``--retrieval-only`` to inspect Search
output without calling the answer model.
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "data" / "personamem_v2"
DEFAULT_BENCHMARK_FILE = DEFAULT_DATASET_ROOT / "benchmark" / "text" / "benchmark.csv"
DEFAULT_OV_CONFIG = REPOSITORY_ROOT / ".openviking" / "ov.conf"
EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
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


def parse_user_query(raw_value: str) -> str:
    """Parse the official CSV user_query field and append its recall instruction."""
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            parsed = {"content": str(raw_value).strip("\"'")}

    if isinstance(parsed, dict):
        query = str(parsed.get("content", parsed.get("text", "")) or "")
    else:
        query = str(parsed)
    if query and RECALL_INSTRUCTION.strip() not in query:
        query += RECALL_INSTRUCTION
    return query


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
    if seed is not None:
        random.seed(seed)

    options = [correct_answer] + incorrect_answers
    random.shuffle(options)

    option_mapping: dict[str, str] = {}
    option_parts = []
    for index, option in enumerate(options):
        letter = chr(65 + index)
        option_mapping[letter] = option
        option_parts.append(f"{letter}. {option}")

    mcq_instruction = (
        "Please choose the best answer from the following options:\n\n"
        + "\n".join(option_parts)
        + "\n\nThink step by step about which answer best fits the user's query and conversation "
        "context. Provide your reasoning first, then give your final answer as "
        "'Final Answer: [Letter]'"
    )
    return mcq_instruction, option_mapping


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
    """Create the official Ark async client used by the Responses API."""
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


async def probe_answer_client(client: Any, args: argparse.Namespace) -> None:
    """Fail fast if the configured Ark model cannot serve a minimal request."""
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
        response = await asyncio.wait_for(
            client.responses.create(
                model=args.answer_model,
                input="Hi, who are you?",
                # thinking={"type": "disabled"},
                # max_output_tokens=16,
                # store=False,
                # timeout=probe_timeout,
            ),
            timeout=probe_timeout,
        )
        answer = response_output_text(response)
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


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    retry_attempts: int,
    retry_delay: float,
) -> dict[str, Any]:
    for attempt in range(retry_attempts):
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise TypeError(f"expected JSON object from {url}")
            return result
        except (httpx.RequestError, httpx.HTTPStatusError):
            if attempt + 1 == retry_attempts:
                raise
            await asyncio.sleep(retry_delay * (2**attempt))
    raise AssertionError("retry loop exited unexpectedly")


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
) -> dict[str, Any]:
    output: dict[str, Any] = dict(row)
    output.update(dict.fromkeys(mode_columns(args.eval_mode, args.size), ""))
    persona_id = str(row.get("persona_id") or "")
    user_id = f"{args.user_id_prefix}-{persona_key(persona_id)}"
    output["openviking_user_id"] = user_id
    query = parse_user_query(str(row.get("user_query") or ""))
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
                row_seed = hash(f"{persona_id}_{query}") % 2**32
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
    scored = [
        row for row in rows if row.get(correctness_column) in {"True", "False"}
    ]
    correct = sum(row[correctness_column] == "True" for row in scored)
    summary: dict[str, Any] = {
        "benchmark_file": str(args.benchmark_file),
        "output": str(args.output),
        "size": args.size,
        "eval_mode": args.eval_mode,
        "search_mode": args.search_mode,
        "experiment_name": args.experiment_name,
        "user_id_prefix": args.user_id_prefix,
        "top_k": args.top_k,
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
        stats: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )
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
        column
        for column in mode_columns(args.eval_mode, args.size)
        if column not in source_fields
    ]
    timeout = httpx.Timeout(args.timeout)
    semaphore = asyncio.Semaphore(args.parallel)
    completed_rows: list[tuple[int, dict[str, Any]]] = []
    errors = 0
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

    ordered_rows = [
        output
        for _, output in sorted(completed_rows, key=lambda item: item[0])
    ]
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
        Path(__file__).resolve().parent
        / "data"
        / "results"
        / f"personamem_v2_search_{eval_mode}_{size}_{timestamp}.csv"
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=10)
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
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-base-url")
    parser.add_argument("--answer-api-key")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run Search and save memories without calling the answer model.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for name in ("top_k", "parallel", "retry_attempts", "answer_retry_attempts"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.timeout <= 0 or args.retry_delay < 0:
        parser.error("--timeout must be positive and --retry-delay must be non-negative")
    if not args.benchmark_file.is_file():
        parser.error(f"benchmark file does not exist: {args.benchmark_file}")
    args.user_id_prefix = args.user_id_prefix or f"personamem-v2-{args.size}"
    if args.experiment_name:
        if not EXPERIMENT_NAME_RE.fullmatch(args.experiment_name):
            parser.error(
                "--experiment-name must start with an alphanumeric character and contain "
                "only letters, digits, underscores, dots, or hyphens"
            )
        args.user_id_prefix = f"{args.user_id_prefix}-{args.experiment_name}"

    answer_config = load_answer_config(args.ov_config)
    args.answer_model = (
        args.answer_model
        or os.getenv("DOUBAO_VLM_MODEL")
        or answer_config.get("model", "")
    )
    args.answer_base_url = (
        args.answer_base_url
        or os.getenv("ARK_API_BASE")
        or answer_config.get("api_base", "")
    )
    args.answer_api_key = (
        args.answer_api_key
        or os.getenv("ARK_API_KEY")
        or answer_config.get("api_key", "")
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
            parser.error(
                f"missing {', '.join(missing)}; configure .openviking/ov.conf "
                "or use --retrieval-only"
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
