# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Rule-specific Jev System One judgments."""

import json
import time
from dataclasses import dataclass
from typing import Dict, List

import httpx

from openviking.models.rerank.jev_rerank import (
    MODE_NOUL,
    JevRerankClient,
    NoulRerankMode,
)
from openviking_cli.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuleJevInput:
    name: str
    content: str
    applies_when: str
    exception: str


@dataclass(frozen=True)
class RuleJevScore:
    applies_when: float
    exception: float


def _validate_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"Invalid Rule Jev probability: {value!r}")
    return float(value)


class RuleJevClient:
    """Jev client for Rule applicability, independent of document reranking."""

    def __init__(self, config) -> None:
        api_base = (config.api_base or "https://api.typesafe.ai").rstrip("/")
        self.api_url = (
            api_base if api_base.endswith("/v1/systemone") else f"{api_base}/v1/systemone"
        )
        default_model = "typesafe-ai/jev" if "ai-gateway.vercel.sh" in api_base else "jev-latest"
        self.model_name = config.model or default_model
        self.log_payloads = config.log_payloads
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                **dict(config.extra_headers or {}),
            },
            timeout=config.timeout,
        )

    @classmethod
    def from_config(cls, config) -> "RuleJevClient | None":
        if not config or not config.is_available():
            return None
        if config._effective_provider() != "jev" or config.mode != MODE_NOUL:
            raise ValueError("Rule Jev judgments require provider='jev' and mode='noul'")
        return cls(config)

    @staticmethod
    def _questions(rules: List[RuleJevInput]) -> Dict[str, dict]:
        questions: Dict[str, dict] = {}
        for index, rule in enumerate(rules):
            if rule.applies_when:
                questions[f"applies_when_{index}"] = {
                    "type": MODE_NOUL,
                    "instructions": {
                        "rule_index": index,
                        "question": (
                            "Does rules[rule_index] apply to task_context? Judge only "
                            "applicability, not whether the Rule is satisfied or violated."
                        ),
                    },
                    "criteria": {
                        "true": rule.applies_when,
                        "false": "The task context does not establish the applies_when criterion.",
                    },
                }
            if rule.exception:
                questions[f"exception_{index}"] = {
                    "type": MODE_NOUL,
                    "instructions": {
                        "rule_index": index,
                        "question": (
                            "Does an exception to rules[rule_index] apply to task_context?"
                        ),
                    },
                    "criteria": {
                        "true": rule.exception,
                        "false": "The task context does not establish the exception criterion.",
                    },
                }
        return questions

    @staticmethod
    def _answer(
        answers: dict,
        key: str,
        *,
        expected: bool,
        default: float,
    ) -> float:
        if not expected:
            return default
        if key not in answers:
            raise ValueError(f"Missing Rule Jev answer: {key}")
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != MODE_NOUL:
            raise ValueError(f"Missing or malformed Rule Jev answer: {key}")
        return _validate_probability(answer.get("noul"))

    def judge_rules(
        self,
        task_context: str,
        rules: List[RuleJevInput],
    ) -> List[RuleJevScore] | None:
        if not rules:
            return []
        body = {
            "model": self.model_name,
            "state": {
                "task_context": task_context,
                "rules": [{"name": rule.name, "content": rule.content} for rule in rules],
            },
            "questions": self._questions(rules),
        }
        try:
            if self.log_payloads:
                logger.warning(
                    "[RuleJev] Request rules=%s payload=%s",
                    len(rules),
                    json.dumps(body, ensure_ascii=False),
                )
            started = time.monotonic()
            response = self._client.post(self.api_url, json=body)
            duration_seconds = time.monotonic() - started
            response.raise_for_status()
            data = response.json()
            if self.log_payloads:
                logger.warning(
                    "[RuleJev] Response status=%s duration_ms=%.1f payload=%s",
                    response.status_code,
                    duration_seconds * 1000,
                    json.dumps(data, ensure_ascii=False),
                )
            answers = data.get("answers")
            if not isinstance(answers, dict):
                logger.warning("[RuleJev] Response has no answers object")
                return None
            return [
                RuleJevScore(
                    applies_when=self._answer(
                        answers,
                        f"applies_when_{index}",
                        expected=bool(rule.applies_when),
                        default=1.0,
                    ),
                    exception=self._answer(
                        answers,
                        f"exception_{index}",
                        expected=bool(rule.exception),
                        default=0.0,
                    ),
                )
                for index, rule in enumerate(rules)
            ]
        except httpx.HTTPStatusError as error:
            logger.error(
                "[RuleJev] API error: %s %s",
                error.response.status_code,
                error.response.text,
            )
            return None
        except Exception as error:
            logger.error("[RuleJev] Judgment failed: %s", error)
            return None

    def close(self) -> None:
        self._client.close()


class _RuleNoulMode(NoulRerankMode):
    def __init__(
        self,
        *,
        question: str,
        true_criterion: str,
        false_criterion: str,
    ) -> None:
        self._question = question
        self._true_criterion = true_criterion
        self._false_criterion = false_criterion

    def build_questions(self, documents: List[str]) -> Dict[str, dict]:
        return {
            f"relevance_{index}": {
                "type": MODE_NOUL,
                "instructions": {
                    "candidate_index": index,
                    "question": self._question,
                },
                "criteria": {
                    "true": self._true_criterion,
                    "false": self._false_criterion,
                },
            }
            for index in range(len(documents))
        }


def _create_rule_jev_client(config) -> JevRerankClient | None:
    client = JevRerankClient.from_config(config)
    if client is None:
        return None
    if client.mode != MODE_NOUL:
        client.close()
        raise ValueError("Rule Jev judgments require mode='noul'")
    extra_headers = dict(getattr(config, "extra_headers", None) or {})
    if extra_headers:
        client._client.headers.update(extra_headers)  # noqa: SLF001 - isolated adapter hook
    return client


def create_rule_jev_judge(
    config,
    *,
    question: str,
    true_criterion: str,
    false_criterion: str,
) -> JevRerankClient | None:
    """Build an isolated Jev client with a Rule-specific Noul question."""
    client = _create_rule_jev_client(config)
    if client is None:
        return None

    client._strategy = _RuleNoulMode(  # noqa: SLF001 - isolated adapter hook
        question=question,
        true_criterion=true_criterion,
        false_criterion=false_criterion,
    )
    return client


__all__ = [
    "RuleJevClient",
    "RuleJevInput",
    "RuleJevScore",
    "create_rule_jev_judge",
]
