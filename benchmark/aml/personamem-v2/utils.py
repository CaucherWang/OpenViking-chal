from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
AML_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = AML_ROOT.parents[1]
DEFAULT_DATASET_ROOT = AML_ROOT / "data" / "personamem_v2"
DEFAULT_BENCHMARK_FILE = DEFAULT_DATASET_ROOT / "benchmark" / "text" / "benchmark.csv"
DEFAULT_OV_CONFIG = REPOSITORY_ROOT / ".openviking" / "ov.conf"
GPT_4O_MINI_MODEL = "gpt-4o-mini-2024-07-18"
GPT_5_5_MODEL = "gpt-5.5-2026-04-24"
ES_AZURE_ENDPOINT = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
ES_AZURE_API_VERSION = "2024-02-01"
ES_CHAT_COMPLETION_BACKENDS = frozenset({"gpt-4o-mini", "gpt-5.5"})
EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SENSITIVE_CONTENT_ERROR_CODE = "OutputTextSensitiveContentDetected"
SENSITIVE_CONTENT_RESPONSE = "The response was blocked because sensitive content was detected."
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


@dataclass(frozen=True)
class AnswerTokenUsage:
    prompt_tokens: int
    completion_tokens: int


def response_token_usage(response: Any) -> AnswerTokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    return AnswerTokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
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


def stable_row_seed(persona_id: str, query: str) -> int:
    """Return a reproducible 32-bit seed for one benchmark row."""
    digest = hashlib.sha256(f"{persona_id}_{query}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def extract_final_answer(response: str) -> str:
    """Extract the final MCQ letter from supported model answer formats."""
    if not response:
        return ""
    formatting = r"(?:(?:\*\*|__)\s*)*"
    patterns = (
        rf"{formatting}Final\s+Answer{formatting}:\s*{formatting}"
        rf"\[\s*{formatting}([A-Z]){formatting}\]\s*{formatting}",
        rf"{formatting}Final\s+Answer{formatting}:\s*{formatting}([A-Z])\b",
        rf"{formatting}Answer{formatting}:\s*{formatting}"
        rf"\[\s*{formatting}([A-Z]){formatting}\]\s*{formatting}",
        r"\$\\boxed\{([A-Z])\}\$",
        r"\\boxed\{([A-Z])\}",
        rf"{formatting}Answer{formatting}:\s*{formatting}([A-Z])\b",
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
        "893_api_key": str(vlm.get("893_api_key") or ""),
        "gpt_4o_mini_key": str(vlm.get("gpt_4o_mini_key") or ""),
        "gpt_5_5_key": str(vlm.get("gpt_5_5_key") or ""),
        "api_base": str(vlm.get("api_base") or ""),
    }


def create_answer_client(args: argparse.Namespace) -> Any:
    """Create the async client selected by the answer backend."""
    if args.answer_backend in ES_CHAT_COMPLETION_BACKENDS:
        from openai import AsyncAzureOpenAI

        return AsyncAzureOpenAI(
            api_key=args.answer_api_key,
            api_version=args.answer_api_version,
            azure_endpoint=args.answer_base_url,
            default_headers={"X-TT-LOGID": args.answer_log_id},
            timeout=args.timeout,
            max_retries=0,
        )

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


def chat_completion_output_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise TypeError("answer model returned no choices")
    answer = str(choices[0].message.content or "").strip()
    if not answer:
        raise TypeError("answer model returned empty message content")
    return answer


async def request_answer(
    client: Any,
    *,
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    usage_callback: Callable[[AnswerTokenUsage | None], None] | None = None,
) -> str:
    if args.answer_backend in ES_CHAT_COMPLETION_BACKENDS:
        azure_messages = [
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
            for message in messages
        ]
        response = await client.chat.completions.create(
            model=args.answer_model,
            messages=azure_messages,
            max_tokens=args.answer_max_tokens,
            stream=False,
        )
        if usage_callback is not None:
            usage_callback(response_token_usage(response))
        return chat_completion_output_text(response)

    response = await client.responses.create(
        model=args.answer_model,
        input=messages,
    )
    if usage_callback is not None:
        usage_callback(response_token_usage(response))
    return response_output_text(response)


async def answer_with_model(
    client: Any,
    *,
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    usage_callback: Callable[[AnswerTokenUsage | None], None] | None = None,
    request_callback: Callable[[], None] | None = None,
) -> str:
    for attempt in range(1, args.answer_retry_attempts + 1):
        if request_callback is not None:
            request_callback()
        try:
            return await request_answer(
                client,
                args=args,
                messages=messages,
                usage_callback=usage_callback,
            )
        except Exception as exc:
            if SENSITIVE_CONTENT_ERROR_CODE in str(exc):
                return SENSITIVE_CONTENT_RESPONSE
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


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    retry_attempts: int = 1,
    retry_delay: float = 0.0,
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


def category_label(category: str, value: str) -> str:
    if "distance_from_related_snippet_to_query" not in category:
        return value
    try:
        distance = int(value)
    except (TypeError, ValueError):
        return value
    start = (distance // 1024) * 1024
    return f"{start}-{start + 1023}"


def build_mcq_summary(rows: list[dict[str, Any]], size: str) -> dict[str, Any]:
    correctness_column = f"is_correct_mcq_{size}"
    scored = [row for row in rows if row.get(correctness_column) in {"True", "False"}]
    correct = sum(row[correctness_column] == "True" for row in scored)
    summary: dict[str, Any] = {
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


def format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def report_progress(
    current: int,
    total: int,
    errors: int,
    *,
    started_at: float,
) -> None:
    current_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_seconds = int(time.monotonic() - started_at)
    elapsed = format_elapsed(elapsed_seconds)
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
