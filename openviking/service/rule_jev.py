# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Rule-only Jev Noul questions without changing the shared rerank client."""

from typing import Dict, List

from openviking.models.rerank.jev_rerank import (
    MODE_NOUL,
    JevRerankClient,
    NoulRerankMode,
)


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


def create_rule_jev_judge(
    config,
    *,
    question: str,
    true_criterion: str,
    false_criterion: str,
) -> JevRerankClient | None:
    """Build an isolated Jev client with a Rule-specific Noul question."""
    client = JevRerankClient.from_config(config)
    if client is None:
        return None
    if client.mode != MODE_NOUL:
        client.close()
        raise ValueError("Rule Jev judgments require mode='noul'")

    client._strategy = _RuleNoulMode(  # noqa: SLF001 - isolated adapter hook
        question=question,
        true_criterion=true_criterion,
        false_criterion=false_criterion,
    )
    extra_headers = dict(getattr(config, "extra_headers", None) or {})
    if extra_headers:
        client._client.headers.update(extra_headers)  # noqa: SLF001 - isolated adapter hook
    return client


__all__ = ["create_rule_jev_judge"]
