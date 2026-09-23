#!/usr/bin/env python3
"""Ingest PersonaMem-v2 chat history files through HTTP or direct backend calls."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_HISTORY_DIR = (
    Path(__file__).resolve().parent / "data" / "personamem_v2" / "data" / "chat_history_32k"
)
DEFAULT_BENCHMARK_DIR = (
    Path(__file__).resolve().parent / "data" / "personamem_v2" / "benchmark" / "text"
)
HTTP_MAX_ADD_MESSAGES = 20
DIRECT_MAX_ADD_MESSAGES = 100
PERSONA_RE = re.compile(r"(persona\d+)")
EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_HISTORY_PREFIX = "250913"
PERSONA_SPLITS = ("train", "val", "benchmark")
COMMIT_MODES = ("after-each-batch", "once-per-persona")


def timestamp_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number > 10_000_000_000 else number * 1000
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def format_timestamp(timestamp: int) -> str:
    return (
        datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def message_from_raw(raw: dict[str, Any]) -> dict[str, Any] | None:
    content = str(raw.get("content", raw.get("text", "")) or "").strip()
    if not content:
        return None

    role = str(raw.get("role") or "user").lower()
    if role == "system":
        role = "user"
        content = "[System instruction]\n" + content
    if role not in {"user", "assistant"}:
        role = "user"

    message: dict[str, Any] = {"role": role, "content": content}
    timestamp = timestamp_ms(raw.get("timestamp", raw.get("created_at")))
    if timestamp is not None:
        message["timestamp"] = timestamp
    return message


def persona_id_from_path(path: Path) -> str:
    match = PERSONA_RE.search(path.stem)
    if match is None:
        raise ValueError(f"cannot infer persona id from file name: {path}")
    return match.group(1)


def normalize_persona_id(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("persona"):
        text = text.removeprefix("persona")
    if not text.isdigit():
        return None
    return f"persona{int(text)}"


def read_query_personas(
    benchmark_dir: Path,
    splits: list[str],
) -> tuple[set[str], dict[str, dict[str, int]]]:
    persona_ids: set[str] = set()
    split_summary: dict[str, dict[str, int]] = {}
    for split in splits:
        path = benchmark_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"PersonaMem-v2 query CSV not found: {path}")

        rows = 0
        query_rows = 0
        split_personas: set[str] = set()
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                if not str(row.get("user_query") or "").strip():
                    continue
                persona_id = normalize_persona_id(row.get("persona_id"))
                if persona_id is None:
                    continue
                query_rows += 1
                split_personas.add(persona_id)

        persona_ids.update(split_personas)
        split_summary[split] = {
            "rows": rows,
            "query_rows": query_rows,
            "personas": len(split_personas),
        }
    return persona_ids, split_summary


def read_history(path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    persona_id = persona_id_from_path(path)

    if isinstance(raw, dict):
        metadata = raw.get("metadata")
        if isinstance(metadata, dict) and metadata.get("persona_id") is not None:
            persona_id = f"persona{metadata['persona_id']}"
        raw_messages = raw.get("chat_history") or raw.get("conversations")
        if raw_messages is None:
            raw_messages = next((value for value in raw.values() if isinstance(value, list)), [])
    else:
        raw_messages = raw

    if not isinstance(raw_messages, list):
        raise ValueError(f"chat history is not a list: {path}")

    messages = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        message = message_from_raw(item)
        if message is not None:
            messages.append(message)
    if not messages:
        raise ValueError(f"chat history has no valid messages: {path}")
    return persona_id, messages


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise TypeError(f"expected JSON object from {url}")
    return result


def direct_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        item = {"role": message["role"], "content": message["content"]}
        timestamp = message.get("timestamp")
        if isinstance(timestamp, int):
            item["created_at"] = format_timestamp(timestamp)
        result.append(item)
    return result


async def ingest_history_file_http(
    client: httpx.AsyncClient,
    *,
    adapter_url: str,
    headers: dict[str, str],
    user_id_prefix: str,
    path: Path,
    commit_once: bool,
) -> dict[str, Any]:
    persona_id, messages = read_history(path)
    user_id = f"{user_id_prefix}-{persona_id}"
    session_id = f"history-{persona_id}"
    add_requests = 0

    for chunk_index in range(0, len(messages), HTTP_MAX_ADD_MESSAGES):
        chunk = messages[chunk_index : chunk_index + HTTP_MAX_ADD_MESSAGES]
        await post_json(
            client,
            adapter_url.rstrip("/") + "/add",
            {
                "request_id": f"{user_id}:chunk-{chunk_index // HTTP_MAX_ADD_MESSAGES}",
                "messages": chunk,
                "user_id": user_id,
                "session_id": session_id,
                "commit": not commit_once,
            },
            headers,
        )
        add_requests += 1

    if commit_once:
        await post_json(
            client,
            adapter_url.rstrip("/") + "/commit",
            {
                "user_id": user_id,
                "session_id": session_id,
            },
            headers,
        )

    return {
        "file": str(path),
        "persona_id": persona_id,
        "user_id": user_id,
        "messages": len(messages),
        "add_requests": add_requests,
        "commit_requests": 1 if commit_once else add_requests,
    }


async def ingest_history_file_direct(
    backend: Any,
    *,
    user_id_prefix: str,
    path: Path,
    commit_once: bool,
) -> dict[str, Any]:
    persona_id, messages = read_history(path)
    user_id = f"{user_id_prefix}-{persona_id}"
    session_id = f"history-{persona_id}"
    add_requests = 0

    for chunk_index in range(0, len(messages), DIRECT_MAX_ADD_MESSAGES):
        chunk = messages[chunk_index : chunk_index + DIRECT_MAX_ADD_MESSAGES]
        add = backend.add_messages if commit_once else backend.add_and_commit
        await add(user_id=user_id, session_id=session_id, messages=direct_messages(chunk))
        add_requests += 1

    if commit_once:
        await backend.commit_session(user_id=user_id, session_id=session_id)

    return {
        "file": str(path),
        "persona_id": persona_id,
        "user_id": user_id,
        "messages": len(messages),
        "add_requests": add_requests,
        "commit_requests": 1 if commit_once else add_requests,
    }


def sorted_history_files(history_dir: Path, history_prefix: str) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        match = PERSONA_RE.search(path.stem)
        if match is None:
            return (10**12, path.name)
        return (int(match.group(1).removeprefix("persona")), path.name)

    pattern = f"chat_history_{history_prefix}_*.json"
    return sorted(history_dir.glob(pattern), key=sort_key)


def format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def finish_summary(summary: dict[str, Any], started_at: float) -> None:
    elapsed_seconds = time.monotonic() - started_at
    ingested_personas = summary["inserted_personas"] + summary["overwritten_personas"]
    average_seconds = (
        summary["ingest_elapsed_seconds"] / ingested_personas if ingested_personas else None
    )
    summary["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    summary["elapsed"] = format_elapsed(elapsed_seconds)
    summary["elapsed_seconds"] = round(elapsed_seconds, 3)
    summary["ingest_elapsed_seconds"] = round(summary["ingest_elapsed_seconds"], 3)
    summary["average_persona_seconds"] = (
        round(average_seconds, 3) if average_seconds is not None else None
    )


def group_persona_results(
    persona_results: list[dict[str, str]],
) -> dict[str, list[str]]:
    grouped = {
        "inserted": [],
        "overwritten": [],
        "skipped": [],
        "failed": [],
    }
    for result in persona_results:
        grouped[result["status"]].append(result["persona_id"])
    return grouped


def report_active_persona(persona_id: str, action: str) -> None:
    if sys.stderr.isatty():
        print(
            f"\r\033[2K{action} {persona_id} ...",
            end="",
            file=sys.stderr,
            flush=True,
        )


def report_progress(
    current: int,
    total: int,
    summary: dict[str, Any],
    persona_results: list[dict[str, str]],
    *,
    progress_every: int,
    started_at: float,
) -> bool:
    if progress_every <= 0 or (current % progress_every != 0 and current != total):
        return False

    elapsed_seconds = time.monotonic() - started_at
    elapsed = format_elapsed(elapsed_seconds)
    current_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    grouped_personas = group_persona_results(persona_results)
    ingested_personas = summary["inserted_personas"] + summary["overwritten_personas"]
    average_seconds = (
        summary["ingest_elapsed_seconds"] / ingested_personas if ingested_personas else 0.0
    )
    if sys.stderr.isatty():
        width = 30
        completed = width * current // total
        bar = "#" * completed + "-" * (width - completed)
        percent = current * 100 / total
        batch_summary = "; ".join(
            f"{status}({len(persona_ids)}): {','.join(persona_ids) or '-'}"
            for status, persona_ids in grouped_personas.items()
        )
        print(
            f"\r\033[2K[{bar}] {percent:6.2f}% "
            f"personas {current}/{total} | "
            f"inserted {summary['inserted_personas']} | "
            f"overwritten {summary['overwritten_personas']} | "
            f"skipped {summary['skipped_personas']} | "
            f"failed {summary['failed_personas']} | "
            f"now {current_time} | "
            f"elapsed {elapsed} | "
            f"avg/persona {average_seconds:.2f}s | "
            f"batch [{batch_summary}]",
            end="\n",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "processed": current,
                    "total": total,
                    "inserted": summary["inserted_personas"],
                    "overwritten": summary["overwritten_personas"],
                    "skipped": summary["skipped_personas"],
                    "failed": summary["failed_personas"],
                    "messages": summary["messages"],
                    "add_requests": summary["add_requests"],
                    "commit_requests": summary["commit_requests"],
                    "current_time": current_time,
                    "elapsed": elapsed,
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "average_persona_seconds": round(average_seconds, 3),
                    "batch_personas": grouped_personas,
                },
                ensure_ascii=False,
            )
        )
    return True


async def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.monotonic()
    commit_once = args.commit_mode == "once-per-persona"
    user_id_prefix = (
        f"{args.user_id_prefix}-{args.experiment_name}"
        if args.experiment_name
        else args.user_id_prefix
    )
    all_files = sorted_history_files(args.history_dir, args.history_prefix)
    files_by_persona = {persona_id_from_path(path): path for path in all_files}
    split_summary: dict[str, dict[str, int]] = {}
    missing_history_personas: list[str] = []

    if args.persona_ids:
        selected_personas = set(args.persona_ids)
        selection_source = "manual"
        missing_history_personas = sorted(
            selected_personas - files_by_persona.keys(),
            key=lambda value: int(value.removeprefix("persona")),
        )
        files = [path for path in all_files if persona_id_from_path(path) in selected_personas]
    elif args.persona_splits:
        selection_source = "splits"
        selected_personas, split_summary = read_query_personas(
            args.benchmark_dir,
            args.persona_splits,
        )
        missing_history_personas = sorted(
            selected_personas - files_by_persona.keys(),
            key=lambda value: int(value.removeprefix("persona")),
        )
        files = [path for path in all_files if persona_id_from_path(path) in selected_personas]
    else:
        selection_source = "all"
        selected_personas = set(files_by_persona)
        files = all_files

    if args.limit is not None:
        files = files[: args.limit]
    if not files and not args.persona_ids:
        raise FileNotFoundError(
            f"no PersonaMem-v2 history JSON files with prefix "
            f"{args.history_prefix!r} found in {args.history_dir}"
        )

    summary: dict[str, Any] = {
        "ingest_mode": args.ingest_mode,
        "concurrency": args.concurrency,
        "retry_attempts": args.retry_attempts,
        "retry_delay_seconds": args.retry_delay_seconds,
        "existing_policy": args.existing_policy,
        "experiment_name": args.experiment_name,
        "user_id_prefix": user_id_prefix,
        "commit_mode": args.commit_mode,
        "history_prefix": args.history_prefix,
        "selection_source": selection_source,
        "requested_persona_ids": args.persona_ids or [],
        "persona_splits": args.persona_splits if selection_source == "splits" else [],
        "split_summary": split_summary,
        "selected_personas": len(selected_personas),
        "query_personas": len(selected_personas) if selection_source == "splits" else None,
        "history_files": len(all_files),
        "scheduled_personas": len(files),
        "missing_history_count": len(missing_history_personas),
        "missing_history_personas": missing_history_personas,
        "processed_personas": 0,
        "existing_personas": 0,
        "inserted_personas": 0,
        "overwritten_personas": 0,
        "skipped_personas": 0,
        "failed_personas": 0,
        "messages": 0,
        "add_requests": 0,
        "commit_requests": 0,
        "ingest_elapsed_seconds": 0.0,
        "failures": [],
    }
    if args.dry_run:
        chunk_size = (
            DIRECT_MAX_ADD_MESSAGES if args.ingest_mode == "direct" else HTTP_MAX_ADD_MESSAGES
        )
        for path in files:
            try:
                _, messages = read_history(path)
                summary["messages"] += len(messages)
                add_requests = (len(messages) + chunk_size - 1) // chunk_size
                summary["add_requests"] += add_requests
                summary["commit_requests"] += 1 if commit_once else add_requests
            except Exception as exc:
                summary["failed_personas"] += 1
                summary["failures"].append(
                    {"file": str(path), "error": f"{type(exc).__name__}: {exc}"}
                )
            summary["processed_personas"] += 1
        summary["dry_run"] = True
        finish_summary(summary, started_at)
        return summary

    from benchmark.aml.server import AMLSettings, OpenVikingBackend

    settings = AMLSettings(
        openviking_url=args.openviking_url,
        openviking_api_key=args.openviking_api_key or None,
        openviking_account=args.openviking_account or None,
        request_timeout_seconds=args.timeout,
        extraction_timeout_seconds=args.extraction_timeout,
        task_poll_seconds=args.task_poll_seconds,
        retry_attempts=args.retry_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    backend = OpenVikingBackend(settings)
    headers = {}
    if args.aml_api_key:
        headers["Authorization"] = f"Bearer {args.aml_api_key}"

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        pending_persona_results: list[dict[str, str]] = []

        async def ingest_persona(path: Path) -> dict[str, Any]:
            persona_id = persona_id_from_path(path)
            user_id = f"{user_id_prefix}-{persona_id}"
            session_id = f"history-{persona_id}"
            outcome: dict[str, Any] = {
                "persona_id": persona_id,
                "user_id": user_id,
                "file": str(path),
                "status": "failed",
                "exists": False,
                "messages": 0,
                "add_requests": 0,
                "commit_requests": 0,
                "ingest_elapsed_seconds": 0.0,
            }
            async with semaphore:
                try:
                    if args.progress_every and args.concurrency == 1:
                        report_active_persona(persona_id, "checking")
                    exists = await backend.has_user_data(
                        user_id=user_id,
                        session_id=session_id,
                    )
                    outcome["exists"] = exists
                    if exists and args.existing_policy == "skip":
                        outcome["status"] = "skipped"
                        return outcome

                    persona_started_at = time.monotonic()
                    if args.progress_every and args.concurrency == 1:
                        report_active_persona(persona_id, "overwriting" if exists else "inserting")
                    if exists:
                        await backend.clear_user_data(user_id=user_id)
                    if args.ingest_mode == "direct":
                        result = await ingest_history_file_direct(
                            backend,
                            user_id_prefix=user_id_prefix,
                            path=path,
                            commit_once=commit_once,
                        )
                    else:
                        result = await ingest_history_file_http(
                            client,
                            adapter_url=args.adapter_url,
                            headers=headers,
                            user_id_prefix=user_id_prefix,
                            path=path,
                            commit_once=commit_once,
                        )
                    outcome.update(
                        {
                            "status": "overwritten" if exists else "inserted",
                            "messages": int(result["messages"]),
                            "add_requests": int(result["add_requests"]),
                            "commit_requests": int(result["commit_requests"]),
                            "ingest_elapsed_seconds": time.monotonic() - persona_started_at,
                        }
                    )
                except Exception as exc:
                    outcome["error"] = f"{type(exc).__name__}: {exc}"
                return outcome

        tasks = [asyncio.create_task(ingest_persona(path)) for path in files]
        for index, completed_task in enumerate(asyncio.as_completed(tasks), start=1):
            outcome = await completed_task
            persona_status = str(outcome["status"])
            if outcome["exists"]:
                summary["existing_personas"] += 1
            if persona_status == "failed":
                summary["failed_personas"] += 1
                summary["failures"].append(
                    {
                        "persona_id": outcome["persona_id"],
                        "user_id": outcome["user_id"],
                        "file": outcome["file"],
                        "error": outcome["error"],
                    }
                )
            else:
                summary[f"{persona_status}_personas"] += 1
                summary["messages"] += outcome["messages"]
                summary["add_requests"] += outcome["add_requests"]
                summary["commit_requests"] += outcome["commit_requests"]
                summary["ingest_elapsed_seconds"] += outcome["ingest_elapsed_seconds"]
            summary["processed_personas"] += 1
            pending_persona_results.append(
                {
                    "persona_id": str(outcome["persona_id"]),
                    "status": persona_status,
                }
            )
            reported = report_progress(
                index,
                len(files),
                summary,
                pending_persona_results,
                progress_every=args.progress_every,
                started_at=started_at,
            )
            if reported:
                pending_persona_results.clear()
        finish_summary(summary, started_at)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--history-prefix", default=DEFAULT_HISTORY_PREFIX)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--persona-ids",
        nargs="+",
        help=(
            "Only ingest these persona IDs (for example: persona12 34,persona56). "
            "Takes precedence over --persona-splits."
        ),
    )
    parser.add_argument(
        "--persona-splits",
        nargs="+",
        choices=PERSONA_SPLITS,
        default=["train"],
        help=(
            "Only ingest personas referenced by non-empty user_query rows in the "
            "selected train, val, or benchmark CSV files."
        ),
    )
    parser.add_argument(
        "--existing-policy",
        choices=["skip", "overwrite"],
        default="skip",
        help="Skip existing persona users or delete their data before re-ingesting.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--ingest-mode",
        choices=["http", "direct"],
        default="direct",
        help="http posts to the AML adapter /add endpoint; direct calls OpenVikingBackend.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of personas to ingest concurrently.",
    )
    parser.add_argument("--adapter-url", default="http://127.0.0.1:8088")
    parser.add_argument("--aml-api-key", default=os.getenv("AML_API_KEY", ""))
    parser.add_argument(
        "--openviking-url",
        default=os.getenv("OPENVIKING_URL", "http://127.0.0.1:1933"),
    )
    parser.add_argument("--openviking-api-key", default=os.getenv("OPENVIKING_API_KEY", ""))
    parser.add_argument("--openviking-account", default=os.getenv("OPENVIKING_ACCOUNT", ""))
    parser.add_argument("--user-id-prefix", default="personamem-v2-32k")
    parser.add_argument(
        "--experiment-name",
        help="Isolate this experiment under user IDs containing the provided name.",
    )
    parser.add_argument(
        "--commit-mode",
        choices=COMMIT_MODES,
        default="after-each-batch",
        help=(
            "Commit after every add batch, or add all messages for a persona "
            "before committing its session once."
        ),
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--extraction-timeout", type=float, default=900.0)
    parser.add_argument("--task-poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=8,
        help="Maximum attempts for retryable OpenViking requests.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=0.5,
        help="Initial retry delay; subsequent delays use exponential backoff.",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.retry_attempts < 1:
        parser.error("--retry-attempts must be positive")
    if not math.isfinite(args.retry_delay_seconds) or args.retry_delay_seconds <= 0:
        parser.error("--retry-delay-seconds must be a finite number greater than zero")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.persona_ids:
        normalized_persona_ids: set[str] = set()
        invalid_persona_ids: list[str] = []
        for raw_value in args.persona_ids:
            for value in raw_value.split(","):
                persona_id = normalize_persona_id(value)
                if persona_id is None:
                    invalid_persona_ids.append(value)
                else:
                    normalized_persona_ids.add(persona_id)
        if invalid_persona_ids:
            parser.error(
                "--persona-ids contains invalid values: "
                + ", ".join(repr(value) for value in invalid_persona_ids)
            )
        args.persona_ids = sorted(
            normalized_persona_ids,
            key=lambda value: int(value.removeprefix("persona")),
        )
    if args.experiment_name and not EXPERIMENT_NAME_RE.fullmatch(args.experiment_name):
        parser.error(
            "--experiment-name must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens"
        )

    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
