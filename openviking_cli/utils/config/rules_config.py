# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Model configuration for rule compilation and applicability matching."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .rerank_config import RerankConfig
from .vlm_config import VLMConfig


class RulesConfig(BaseModel):
    """Models dedicated to the rules module."""

    deleted_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days a rule must remain deleted before permanent cleanup",
    )
    deleted_cleanup_interval_seconds: int = Field(
        default=2_592_000,
        ge=60,
        description="Interval between deleted-rule cleanup scans",
    )
    deleted_cleanup_enabled: bool = Field(
        default=True,
        description="Whether to periodically clean up rules deleted past retention",
    )
    compiler_context_window: int = Field(
        default=0,
        ge=0,
        description=(
            "Compiler model context window in tokens; 0 auto-detects it from the model "
            "configuration and falls back to a conservative default"
        ),
    )
    jev_rule_matcher: Optional[RerankConfig] = Field(
        default=None,
        description="Optional Jev Noul model for independent rule applicability scoring",
    )
    case_irrelevance_threshold: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description=(
            "Skip a recalled Case only when Jev's probability that it is completely "
            "unrelated to the current context meets this threshold"
        ),
    )
    rule_matcher: Optional[VLMConfig] = Field(
        default=None,
        description="Optional VLM used when Jev matching is unavailable",
    )
    compiler: Optional[VLMConfig] = Field(
        default=None,
        description="Optional VLM used to compile source material into rules",
    )

    @model_validator(mode="after")
    def validate_jev_rule_matcher(self) -> "RulesConfig":
        if 0 < self.compiler_context_window < 2048:
            raise ValueError("rules.compiler_context_window must be 0 or at least 2048")
        config = self.jev_rule_matcher
        if config is None or not config.is_available():
            return self
        if config._effective_provider() != "jev":
            raise ValueError("rules.jev_rule_matcher must use provider 'jev'")
        if config.mode != "noul":
            raise ValueError("rules.jev_rule_matcher must use mode 'noul'")
        return self

    @staticmethod
    def _configured_vlm(config: Optional[VLMConfig]) -> Optional[VLMConfig]:
        if config is not None and config._has_any_config():
            return config
        return None

    def get_rule_matcher(self) -> Optional[VLMConfig]:
        """Return the dedicated rule matcher, treating an empty block as unset."""
        return self._configured_vlm(self.rule_matcher)

    def get_compiler(self) -> Optional[VLMConfig]:
        """Return the dedicated compiler, treating an empty block as unset."""
        return self._configured_vlm(self.compiler)
