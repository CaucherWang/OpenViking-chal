# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared primitives for Jev-compatible System One endpoints."""

from typing import Dict, Tuple

MODE_NOUL = "noul"
MODE_CHOICE = "choice"
DEFAULT_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"


def resolve_jev_target(config) -> Tuple[str, str, Dict[str, str]]:
    """Resolve a System One endpoint, model, and headers from supported configs."""
    if hasattr(config, "resolved_api_base"):
        return (
            config.resolved_api_base(),
            config.resolved_model(),
            config.resolved_extra_headers(),
        )

    api_base = config.api_base or DEFAULT_SYSTEM_ONE_URL
    default_model = "typesafe-ai/jev" if "ai-gateway.vercel.sh" in api_base else "jev-latest"
    return api_base, config.model or default_model, dict(config.extra_headers or {})


def validate_jev_probability(value: object, *, label: str = "Jev probability") -> float:
    """Validate and normalize one probability returned by System One."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {label}: {value!r}")
    return float(value)
