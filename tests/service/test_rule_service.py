import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest
import yaml

from openviking.server.identity import RequestContext, Role
from openviking.service.rule_resource_ingest import AddResourceRequest
from openviking.service.rule_service import (
    CASE_SCHEMA,
    RULE_SCHEMA,
    RuleCaseRef,
    RuleCompilePlan,
    RuleCompileRequest,
    RuleMutation,
    RuleRecord,
    RuleSearchRequest,
    RuleService,
    StoredCase,
    StoredRule,
    _build_material_batches,
    _render_markdown,
    _render_rules,
)
from openviking.service.task_tracker import TaskRecord, TaskStatus
from openviking.utils.token_estimation import estimate_text_tokens
from openviking_cli.exceptions import (
    InvalidArgumentError,
    NotFoundError,
    ProcessingError,
    UnavailableError,
    VLMFailedError,
)
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.rerank_config import RerankConfig
from openviking_cli.utils.config.rules_config import RulesConfig
from openviking_cli.utils.config.vlm_config import VLMConfig


def _ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("acme", "alice"), role=Role.USER)


def _record(name: str = "confirm-delete") -> RuleRecord:
    return RuleRecord(
        name=name,
        description="Confirm before deleting files.",
        content="Ask for confirmation before deleting files.",
        applies_when="when a task requires deleting files",
        does_not_apply_when="when the user already explicitly confirmed this deletion",
        body="# Confirm deletion\n\nNever delete files before confirmation.",
    )


def test_rule_record_is_an_implicit_user_constraint():
    record = _record()
    dumped = record.model_dump()

    assert record.confidence == "user"
    assert {"type", "domain", "scope", "scope_anchor"}.isdisjoint(dumped)
    assert record.content == "Ask for confirmation before deleting files."
    assert "rule_text" not in dumped
    assert RuleRecord.model_validate({**dumped, "confidence": "high"}).confidence == "high"


def test_compile_prompt_does_not_expose_memory_preference_fields():
    removed_comparison_field = "_".join(("nearest", "existing"))
    prompt = RuleService._render_compile_prompt(
        instruction="Always confirm deletion.",
        source_text="",
        case_data=[],
        rule_data=[],
    )

    assert '"confidence": "user"' in prompt
    assert '"content": "complete actionable constraint"' in prompt
    assert '"rule_text":' not in prompt
    assert '"type": "preference' not in prompt
    assert '"domain":' not in prompt
    assert '"scope":' not in prompt
    assert "Preference-like statements" in prompt
    assert "REPLACE" in prompt
    assert "Infer the single semantic target" in prompt
    assert "use UNCERTAIN instead of" in prompt
    assert "SUPERSEDE" not in prompt
    assert "SPLIT" not in prompt
    assert '"children":' not in prompt
    assert '"applicability_evidence":' in prompt
    assert f'"{removed_comparison_field}":' not in prompt
    assert "internally compare it" in prompt
    assert '"carried_over":' in prompt


def test_rule_mutations_are_flat_and_replace_has_one_target():
    removed_comparison_field = "_".join(("nearest", "existing"))
    replacement = _record("replacement")

    mutation = RuleMutation(
        operation="REPLACE",
        target_ids=["old-rule"],
        record=replacement,
    )

    assert mutation.operation == "REPLACE"
    assert "children" not in mutation.model_dump()
    assert RuleMutation(operation="UNCERTAIN").operation == "UNCERTAIN"
    legacy_uncertain_operation = "_".join(("NEEDS", "USER"))
    with pytest.raises(ValueError, match="unsupported rule operation"):
        RuleMutation(operation=legacy_uncertain_operation)
    with pytest.raises(ValueError, match="unsupported rule operation"):
        RuleMutation(operation="SUPERSEDE", target_ids=["old-rule"], record=replacement)
    with pytest.raises(ValueError, match="unsupported rule operation"):
        RuleMutation(operation="SPLIT")
    with pytest.raises(ValueError, match="exactly one target_id"):
        RuleMutation(
            operation="REPLACE",
            target_ids=["old-rule", "other-rule"],
            record=replacement,
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RuleMutation.model_validate(
            {
                "operation": "NOOP",
                "target_ids": ["old-rule"],
                "children": [],
            }
        )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RuleMutation.model_validate(
            {
                "operation": "UNCERTAIN",
                removed_comparison_field: None,
            }
        )


@pytest.mark.asyncio
async def test_load_rule_accepts_legacy_rule_text_frontmatter():
    uri = "viking://user/alice/memories/rules/items/legacy.md"
    metadata = {
        "schema": RULE_SCHEMA,
        "atomic_id": "legacy",
        "revision": 1,
        "status": "active",
        "case_id": "file-operations",
        "source_uris": [],
        "evidence": ["confirm deletion"],
        "superseded_by": [],
        "updated_at": "",
        "name": "legacy",
        "description": "Legacy deletion constraint.",
        "domain": "tools",
        "condition": "",
        "confidence": "high",
        "rule_text": "Confirm before deleting files.",
        "applies_when": "when deleting files",
        "does_not_apply_when": "",
    }
    markdown = f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n\nLegacy body.\n"
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(read=AsyncMock(return_value=markdown)),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )

    rule = await service._load_rule(uri, _ctx())

    assert rule.record.content == "Confirm before deleting files."
    assert rule.record.confidence == "high"


@pytest.mark.asyncio
async def test_load_rule_maps_legacy_superseded_status_to_deleted():
    uri = "viking://user/alice/memories/rules/items/legacy.md"
    metadata = {
        "schema": RULE_SCHEMA,
        "atomic_id": "legacy",
        "revision": 2,
        "status": "superseded",
        "case_id": "file-operations",
        "source_uris": [],
        "evidence": ["replace old rule"],
        "superseded_by": ["replacement"],
        "updated_at": "2026-09-01T00:00:00+00:00",
        **_record("legacy").model_dump(exclude={"body"}),
    }
    markdown = f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n\nLegacy body.\n"
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(read=AsyncMock(return_value=markdown)),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )

    rule = await service._load_rule(uri, _ctx())

    assert rule.status == "deleted"
    assert rule.deleted_at == metadata["updated_at"]


def _stored_rule(
    atomic_id: str,
    *,
    case_id: str = "file-operations",
    status: Literal["active", "archived", "deleted"] = "active",
    deleted_at: str | None = None,
) -> StoredRule:
    record = _record(atomic_id)
    uri = f"viking://user/alice/memories/rules/items/{atomic_id}.md"
    return StoredRule(
        schema=RULE_SCHEMA,
        atomic_id=atomic_id,
        revision=1,
        status=status,
        case_id=case_id,
        source_uris=[],
        evidence=["delete files"],
        deleted_at=deleted_at,
        record=record,
        uri=uri,
        content=record.body,
    )


class _FS:
    def __init__(self):
        self.batch = None

    async def batch_write(self, **kwargs):
        self.batch = kwargs
        return {
            "created": [operation["uri"] for operation in kwargs["operations"]],
            "updated": [],
            "unchanged": [],
        }


class _CleanupFS:
    def __init__(self, documents: dict[str, str]):
        self.documents = documents
        self.removed: list[str] = []

    async def glob(self, pattern, ctx, uri):
        assert pattern == "*.md"
        return {"matches": sorted(path for path in self.documents if path.startswith(f"{uri}/"))}

    async def read(self, uri, ctx):
        if uri not in self.documents:
            raise NotFoundError(uri, "file")
        return self.documents[uri]

    async def batch_write(self, **kwargs):
        for operation in kwargs["operations"]:
            self.documents[operation["uri"]] = operation["content"]
        return {"created": [], "updated": list(self.documents), "unchanged": []}

    async def rm(self, uri, ctx, wait):
        assert wait is True
        self.removed.append(uri)
        self.documents.pop(uri, None)


@pytest.mark.asyncio
async def test_rule_models_prefer_dedicated_vlm_configs():
    compiler = VLMConfig(provider="litellm", model="rule-compiler")
    matcher = VLMConfig(provider="litellm", model="rule-matcher")
    resolver = SimpleNamespace(get_vlm=AsyncMock())
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=resolver,
        rules_config=RulesConfig(compiler=compiler, rule_matcher=matcher),
    )

    assert await service._compiler_vlm(_ctx()) is compiler
    assert await service._rule_matcher_vlm(_ctx()) is matcher
    resolver.get_vlm.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_rule_model_configs_use_account_default_vlm():
    default_vlm = object()
    resolver = SimpleNamespace(get_vlm=AsyncMock(return_value=default_vlm))
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=resolver,
        rules_config=RulesConfig(compiler=VLMConfig(), rule_matcher=VLMConfig()),
    )

    assert await service._compiler_vlm(_ctx()) is default_vlm
    assert await service._rule_matcher_vlm(_ctx()) is default_vlm
    assert resolver.get_vlm.await_count == 2


@pytest.mark.parametrize(
    "rules_config,vlm,expected",
    [
        (
            RulesConfig(compiler_context_window=32768),
            SimpleNamespace(model="custom", max_tokens=4096),
            27034,
        ),
        (
            RulesConfig(),
            SimpleNamespace(
                model="custom",
                max_tokens=2048,
                extra_request_body={"num_ctx": 12000},
            ),
            9352,
        ),
        (
            RulesConfig(),
            SimpleNamespace(model="gpt-4o-mini", max_tokens=None),
            105216,
        ),
        (
            RulesConfig(),
            SimpleNamespace(model="unknown-custom", max_tokens=None),
            11469,
        ),
    ],
)
def test_compiler_prompt_limit_uses_context_metadata(rules_config, vlm, expected):
    service = RuleService()
    service._rules_config = rules_config

    assert service._compiler_prompt_token_limit(vlm) == expected


@pytest.mark.asyncio
async def test_jev_rule_matcher_has_priority_over_vlm(monkeypatch):
    class _JevClient:
        def __init__(self):
            self.closed = False

        def rerank_batch(self, query, documents):
            assert query == "delete generated files"
            assert documents == ["rule a", "rule b"]
            return [0.9, 0.2]

        def close(self):
            self.closed = True

    client = _JevClient()
    judge_config = {}
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock()),
        rules_config=RulesConfig(
            jev_rule_matcher=RerankConfig(provider="jev", api_key="key", mode="noul"),
            rule_matcher=VLMConfig(provider="litellm", model="rule-matcher"),
        ),
    )
    vlm_score = AsyncMock(side_effect=AssertionError("VLM matcher must not run"))
    monkeypatch.setattr(service, "_score_rules_with_vlm", vlm_score)
    monkeypatch.setattr(
        "openviking.service.rule_service.create_rule_jev_judge",
        lambda config, **kwargs: judge_config.update(kwargs) or client,
    )

    scores, backend, warning = await service._score_rules(
        "delete generated files",
        [SimpleNamespace(content="rule a"), SimpleNamespace(content="rule b")],
        _ctx(),
    )

    assert scores == [0.9, 0.2]
    assert backend == "jev_noul"
    assert warning is None
    assert client.closed is True
    assert "Judge applicability only" in judge_config["question"]
    assert "remains applicable when its requirement is satisfied" in judge_config["true_criterion"]
    assert "must not be treated as non-applicability" in judge_config["false_criterion"]
    vlm_score.assert_not_awaited()


@pytest.mark.asyncio
async def test_jev_rule_matcher_failure_falls_back_to_vlm(monkeypatch):
    class _FailingJevClient:
        def __init__(self):
            self.closed = False

        def rerank_batch(self, query, documents):
            raise RuntimeError("Jev unavailable")

        def close(self):
            self.closed = True

    client = _FailingJevClient()
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock()),
        rules_config=RulesConfig(
            jev_rule_matcher=RerankConfig(provider="jev", api_key="key", mode="noul"),
            rule_matcher=VLMConfig(provider="litellm", model="rule-matcher"),
        ),
    )
    vlm_score = AsyncMock(return_value=[0.7])
    monkeypatch.setattr(service, "_score_rules_with_vlm", vlm_score)
    monkeypatch.setattr(
        "openviking.service.rule_service.create_rule_jev_judge",
        lambda config, **kwargs: client,
    )

    scores, backend, warning = await service._score_rules(
        "delete generated files",
        [SimpleNamespace(content="rule a")],
        _ctx(),
    )

    assert scores == [0.7]
    assert backend == "vlm"
    assert warning == "Jev Noul was unavailable; VLM fallback was used."
    assert client.closed is True
    vlm_score.assert_awaited_once()


@pytest.mark.asyncio
async def test_vlm_rule_matcher_batches_without_omitting_rules(monkeypatch):
    service = RuleService()
    rules = [_stored_rule(f"rule-{index}") for index in range(45)]
    score_batch = AsyncMock(side_effect=lambda _task, batch, _ctx: [0.9] * len(batch))
    monkeypatch.setattr(service, "_score_rule_batch_with_vlm", score_batch)

    scores = await service._score_rules_with_vlm("validate rules", rules, _ctx())

    assert scores == [0.9] * len(rules)
    assert [len(call.args[1]) for call in score_batch.await_args_list] == [20, 20, 5]


@pytest.mark.asyncio
async def test_vlm_rule_matcher_judges_applicability_not_compliance(monkeypatch):
    vlm = SimpleNamespace(
        get_completion_async=AsyncMock(
            return_value='{"scores": [{"index": 0, "probability": 0.9}]}'
        )
    )
    service = RuleService()
    monkeypatch.setattr(service, "_rule_matcher_vlm", AsyncMock(return_value=vlm))

    scores = await service._score_rule_batch_with_vlm(
        "The requirement is satisfied and no violation was found.",
        [_stored_rule("confirm-delete")],
        _ctx(),
    )

    assert scores == [0.9]
    prompt = vlm.get_completion_async.await_args.kwargs["prompt"]
    assert "Evaluate only whether each saved Rule applies" in prompt
    assert "Do not judge whether the object passes or fails the Rule" in prompt
    assert "no violation is found" in prompt
    assert "remains applicable" in prompt


@pytest.mark.asyncio
async def test_jev_case_filter_skips_only_high_confidence_completely_unrelated(monkeypatch):
    class _JevClient:
        def __init__(self):
            self.closed = False
            self.call = None

        def rerank_batch(self, query, documents):
            self.call = (query, documents)
            return [0.95, 0.4, 0.89]

        def close(self):
            self.closed = True

    client = _JevClient()
    judge_config = {}
    service = RuleService()
    service._rules_config = RulesConfig(
        jev_rule_matcher=RerankConfig(provider="jev", api_key="key", mode="noul"),
        case_irrelevance_threshold=0.9,
    )
    monkeypatch.setattr(
        "openviking.service.rule_service.create_rule_jev_judge",
        lambda config, **kwargs: judge_config.update(kwargs) or client,
    )
    cases = [
        StoredCase(
            schema=CASE_SCHEMA,
            case_id=case_id,
            name=case_id,
            description=f"Scenario for {case_id}",
            rule_uris=[],
            uri=f"viking://user/alice/memories/rules/cases/{case_id}.md",
            content=f"# {case_id}\n\nScenario for {case_id}",
        )
        for case_id in ("unrelated", "relevant", "uncertain")
    ]

    passed, report = await service._filter_unrelated_cases_with_jev("current task", cases)

    assert [case.case_id for case in passed] == ["relevant", "uncertain"]
    assert report["skipped_case_ids"] == ["unrelated"]
    assert report["passed_case_ids"] == ["relevant", "uncertain"]
    assert report["irrelevance_threshold"] == 0.9
    assert client.call is not None
    assert "completely unrelated" in judge_config["question"]
    assert "any uncertainty" in judge_config["false_criterion"]
    assert client.closed is True


@pytest.mark.asyncio
async def test_jev_case_filter_passes_all_cases_when_judgment_fails(monkeypatch):
    class _FailingJevClient:
        def rerank_batch(self, query, documents):
            return None

        def close(self):
            pass

    service = RuleService()
    service._rules_config = RulesConfig(
        jev_rule_matcher=RerankConfig(provider="jev", api_key="key", mode="noul"),
    )
    monkeypatch.setattr(
        "openviking.service.rule_service.create_rule_jev_judge",
        lambda config, **kwargs: _FailingJevClient(),
    )
    cases = [
        StoredCase(
            schema=CASE_SCHEMA,
            case_id="uncertain",
            name="Uncertain",
            description="Possibly relevant.",
            rule_uris=[],
            uri="viking://user/alice/memories/rules/cases/uncertain.md",
            content="# Uncertain\n\nPossibly relevant.",
        )
    ]

    passed, report = await service._filter_unrelated_cases_with_jev("current task", cases)

    assert passed == cases
    assert report["skipped_case_ids"] == []
    assert report["decisions"][0]["reason"] == "jev_unavailable_or_invalid"
    assert "warning" in report


@pytest.mark.asyncio
async def test_apply_new_rule_writes_markdown_item_and_case():
    fs = _FS()
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    plan = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="NEW",
                record=_record(),
                case=RuleCaseRef(
                    case_id="destructive-file-operations",
                    name="Destructive file operations",
                    description="Rules for operations that remove or overwrite files.",
                ),
                evidence_spans=["confirm before deleting files"],
            )
        ]
    )

    result = await service._apply_plan(
        plan,
        existing_cases=[],
        existing_rules=[],
        source_uris=["viking://user/alice/resources/rules"],
        ctx=_ctx(),
    )

    assert len(result["written_uris"]) == 2
    assert fs.batch["root_uri"] == "viking://user/alice/memories/rules"
    by_uri = {operation["uri"]: operation["content"] for operation in fs.batch["operations"]}
    rule_uri = "viking://user/alice/memories/rules/items/confirm-delete.md"
    case_uri = "viking://user/alice/memories/rules/cases/destructive-file-operations.md"
    rule_frontmatter = yaml.safe_load(by_uri[rule_uri].split("---", 2)[1])
    case_frontmatter = yaml.safe_load(by_uri[case_uri].split("---", 2)[1])
    assert rule_frontmatter["schema"] == RULE_SCHEMA
    assert rule_frontmatter["status"] == "active"
    assert rule_frontmatter["source_uris"] == ["viking://user/alice/resources/rules"]
    assert rule_frontmatter["confidence"] == "user"
    assert rule_frontmatter["content"] == "Ask for confirmation before deleting files."
    assert "rule_text" not in rule_frontmatter
    assert {"type", "domain", "scope", "scope_anchor"}.isdisjoint(rule_frontmatter)
    assert case_frontmatter["schema"] == CASE_SCHEMA
    assert case_frontmatter["rule_uris"] == [rule_uri]


@pytest.mark.asyncio
async def test_replace_deletes_old_rule_and_creates_new_rule():
    fs = _FS()
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    old = _stored_rule("old-rule")
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id=old.case_id,
        name="File operations",
        description="Rules for file operations.",
        rule_uris=[old.uri],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )
    plan = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="REPLACE",
                target_ids=[old.atomic_id],
                record=_record("replacement"),
                evidence_spans=["replace old rule"],
            )
        ]
    )

    await service._apply_plan(
        plan,
        existing_cases=[case],
        existing_rules=[old],
        source_uris=[],
        ctx=_ctx(),
    )

    by_uri = {operation["uri"]: operation["content"] for operation in fs.batch["operations"]}
    old_metadata = yaml.safe_load(by_uri[old.uri].split("---", 2)[1])
    replacement_uri = "viking://user/alice/memories/rules/items/replacement.md"
    replacement_metadata = yaml.safe_load(by_uri[replacement_uri].split("---", 2)[1])
    assert old_metadata["status"] == "deleted"
    assert old_metadata["deleted_at"]
    assert replacement_metadata["status"] == "active"
    assert "superseded_by" not in old_metadata


@pytest.mark.asyncio
async def test_update_reactivates_archived_rule():
    fs = _FS()
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    archived = _stored_rule("archived-rule", status="archived")
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id=archived.case_id,
        name="File operations",
        description="Rules for file operations.",
        rule_uris=[archived.uri],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )

    await service._apply_plan(
        RuleCompilePlan(
            mutations=[
                RuleMutation(
                    operation="UPDATE",
                    target_ids=[archived.atomic_id],
                    record=_record("archived-rule"),
                    evidence_spans=["reassert archived rule"],
                )
            ]
        ),
        existing_cases=[case],
        existing_rules=[archived],
        source_uris=[],
        ctx=_ctx(),
    )

    by_uri = {operation["uri"]: operation["content"] for operation in fs.batch["operations"]}
    metadata = yaml.safe_load(by_uri[archived.uri].split("---", 2)[1])
    assert metadata["status"] == "active"
    assert metadata["revision"] == archived.revision + 1


@pytest.mark.asyncio
async def test_delete_is_idempotent_and_restore_resets_deleted_at():
    service = RuleService()
    service.set_dependencies(
        fs=_FS(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    deleted_at = "2026-09-01T00:00:00+00:00"
    deleted = _stored_rule("old-rule", status="deleted", deleted_at=deleted_at)
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id=deleted.case_id,
        name="File operations",
        description="Rules for file operations.",
        rule_uris=[deleted.uri],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )

    unchanged = await service._apply_plan(
        RuleCompilePlan(
            mutations=[
                RuleMutation(
                    operation="DELETE",
                    target_ids=[deleted.atomic_id],
                    evidence_spans=["delete old rule"],
                )
            ]
        ),
        existing_cases=[case],
        existing_rules=[deleted],
        source_uris=[],
        ctx=_ctx(),
        persist=False,
    )
    assert unchanged["_rules"][0].deleted_at == deleted_at
    assert unchanged["written_uris"] == []

    restored = await service._apply_plan(
        RuleCompilePlan(
            mutations=[
                RuleMutation(
                    operation="RESTORE",
                    target_ids=[deleted.atomic_id],
                    evidence_spans=["restore old rule"],
                )
            ]
        ),
        existing_cases=[case],
        existing_rules=[deleted],
        source_uris=[],
        ctx=_ctx(),
        persist=False,
    )
    restored_rule = next(rule for rule in restored["_rules"] if rule.atomic_id == "old-rule")
    assert restored_rule.status == "active"
    assert restored_rule.deleted_at is None


@pytest.mark.asyncio
async def test_cleanup_removes_only_rules_deleted_for_full_retention_period():
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    old = _stored_rule(
        "old-deleted",
        status="deleted",
        deleted_at=(now - timedelta(days=30)).isoformat(),
    )
    recent = _stored_rule(
        "recent-deleted",
        status="deleted",
        deleted_at=(now - timedelta(days=29)).isoformat(),
    )
    archived = _stored_rule("archived", status="archived")
    active = _stored_rule("active")
    rules = [old, recent, archived, active]

    def rule_markdown(rule: StoredRule) -> str:
        metadata = {
            "schema": RULE_SCHEMA,
            "atomic_id": rule.atomic_id,
            "revision": rule.revision,
            "status": rule.status,
            "case_id": rule.case_id,
            "source_uris": rule.source_uris,
            "evidence": rule.evidence,
            "updated_at": now.isoformat(),
            **rule.record.model_dump(exclude={"body"}),
        }
        if rule.deleted_at:
            metadata["deleted_at"] = rule.deleted_at
        return _render_markdown(metadata, rule.record.body)

    case_uri = "viking://user/alice/memories/rules/cases/file-operations.md"
    case_content = _render_markdown(
        {
            "schema": CASE_SCHEMA,
            "case_id": "file-operations",
            "name": "File operations",
            "description": "Rules for file operations.",
            "rule_uris": [rule.uri for rule in rules],
            "updated_at": now.isoformat(),
        },
        "# File operations\n\n## Active rules",
    )
    documents = {rule.uri: rule_markdown(rule) for rule in rules}
    documents[case_uri] = case_content
    fs = _CleanupFS(documents)
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=RulesConfig(deleted_retention_days=30),
    )

    result = await service.cleanup_deleted_rules(ctx=_ctx(), now=now)

    assert result["deleted"] == [old.uri]
    assert result["failed"] == []
    assert fs.removed == [old.uri]
    case_metadata = yaml.safe_load(fs.documents[case_uri].split("---", 2)[1])
    assert case_metadata["rule_uris"] == [recent.uri, archived.uri, active.uri]
    assert recent.uri in fs.documents
    assert archived.uri in fs.documents
    assert active.uri in fs.documents


@pytest.mark.asyncio
async def test_apply_update_rejects_rule_outside_recalled_cases():
    service = RuleService()
    service.set_dependencies(
        fs=_FS(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    plan = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="UPDATE",
                target_ids=["not-recalled"],
                record=_record(),
            )
        ]
    )

    with pytest.raises(InvalidArgumentError, match="unknown rule"):
        await service._apply_plan(
            plan,
            existing_cases=[],
            existing_rules=[],
            source_uris=[],
            ctx=_ctx(),
        )


@pytest.mark.asyncio
async def test_search_returns_only_rules_above_independent_probability_threshold(
    monkeypatch,
):
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id="file-operations",
        name="File operations",
        description="Rules for editing files.",
        rule_uris=["viking://user/alice/memories/rules/items/a.md"],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )
    rules = [
        _stored_rule("a").model_copy(
            update={
                "case_id": case.case_id,
                "uri": case.rule_uris[0],
                "content": "complete rule a",
            }
        ),
        _stored_rule("b").model_copy(
            update={
                "case_id": case.case_id,
                "revision": 2,
                "content": "complete rule b",
            }
        ),
    ]
    recall_cases = AsyncMock(return_value=[case])
    load_rules = AsyncMock(return_value=rules)
    monkeypatch.setattr(service, "_recall_cases", recall_cases)
    monkeypatch.setattr(service, "_load_rules_for_cases", load_rules)
    monkeypatch.setattr(
        service,
        "_score_rules",
        AsyncMock(return_value=([0.91, 0.4], "jev_noul", None)),
    )

    result = await service.search_rules(
        RuleSearchRequest(
            task_context="delete a generated file",
            rule_probability_threshold=0.7,
        ),
        ctx=_ctx(),
    )

    assert result["judge"] == "jev_noul"
    assert [rule["atomic_id"] for rule in result["rules"]] == ["a"]
    assert result["rules"][0]["content"] == "complete rule a"
    assert "[a@1]" in result["rendered"]
    assert "Ask for confirmation before deleting files." in result["rendered"]
    assert "Rules are behavioral constraints" in result["rendered"]
    assert result["delivery"] == {
        "complete": True,
        "rule_count": 1,
        "rendered_chars": len(result["rendered"]),
    }
    recall_cases.assert_awaited_once()
    assert recall_cases.await_args.kwargs["limit"] is None
    load_rules.assert_awaited_once_with([case], _ctx(), strict=True)


@pytest.mark.asyncio
async def test_search_renders_all_rules_without_persistence_metadata(monkeypatch):
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    rules = []
    for atomic_id in ("first", "second", "third"):
        content = f"Apply {atomic_id}: " + atomic_id * 120
        record = _record(atomic_id).model_copy(
            update={
                "content": content,
                "body": (f'Guidance for {atomic_id}.\n\n<!-- MEMORY_FIELDS\n{{"version": 1}}\n-->'),
            }
        )
        rules.append(
            _stored_rule(atomic_id).model_copy(
                update={
                    "record": record,
                    "content": f"---\nschema: {RULE_SCHEMA}\n---\n\n{record.body}",
                }
            )
        )
    monkeypatch.setattr(
        service,
        "_find_relevant_rules",
        AsyncMock(
            return_value=SimpleNamespace(
                cases=[],
                selected=[(rule, 0.9) for rule in rules],
                judge="jev_noul",
                warning=None,
                case_selection=None,
            )
        ),
    )

    first_page = await service.search_rules(
        RuleSearchRequest(task_context="apply all relevant rules"),
        ctx=_ctx(),
    )

    assert first_page["delivery"]["complete"] is True
    assert first_page["delivery"]["rule_count"] == len(rules)
    assert "MEMORY_FIELDS" not in first_page["rendered"]
    assert f"schema: {RULE_SCHEMA}" not in first_page["rendered"]
    for rule in rules:
        assert rule.record.content in first_page["rendered"]


@pytest.mark.asyncio
async def test_two_stage_search_scores_all_rules_from_recalled_cases_without_rule_limit(
    monkeypatch,
):
    service = RuleService()
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id="all-rules",
        name="All rules",
        description="Every active Rule is validated.",
        rule_uris=[],
        uri="viking://user/alice/memories/rules/cases/all-rules.md",
        content="case",
    )
    rules = [_stored_rule(f"rule-{index}", case_id=case.case_id) for index in range(125)]
    recall_cases = AsyncMock(return_value=[case])
    load_rules = AsyncMock(return_value=rules)
    monkeypatch.setattr(service, "_recall_cases", recall_cases)
    monkeypatch.setattr(service, "_load_rules_for_cases", load_rules)
    monkeypatch.setattr(
        service, "_score_rules", AsyncMock(return_value=([0.9] * len(rules), "vlm", None))
    )

    matches = await service._find_relevant_rules(
        "validate every applicable rule",
        ctx=_ctx(),
        rule_probability_threshold=0.65,
        allowed_statuses=frozenset({"active"}),
        strict_rule_loading=True,
        filter_cases_with_jev=True,
    )

    assert len(matches.selected) == len(rules)
    assert [rule.atomic_id for rule, _ in matches.selected] == [rule.atomic_id for rule in rules]
    recall_cases.assert_awaited_once()
    load_rules.assert_awaited_once_with([case], _ctx(), strict=True)


@pytest.mark.asyncio
async def test_runtime_case_recall_searches_all_cases_before_thresholding(monkeypatch):
    case_uris = [f"viking://user/alice/memories/rules/cases/case-{index}.md" for index in range(7)]
    fs = SimpleNamespace(
        stat=AsyncMock(),
        glob=AsyncMock(return_value={"matches": case_uris}),
    )
    search = SimpleNamespace(
        find=AsyncMock(
            return_value={
                "memories": [{"uri": uri} for uri in case_uris],
                "resources": [],
                "skills": [],
            }
        )
    )
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=search,
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    cases = [
        StoredCase(
            schema=CASE_SCHEMA,
            case_id=f"case-{index}",
            name=f"Case {index}",
            description=f"Scenario {index}",
            rule_uris=[],
            uri=uri,
            content="case",
        )
        for index, uri in enumerate(case_uris)
    ]
    monkeypatch.setattr(service, "_load_case", AsyncMock(side_effect=cases))

    recalled = await service._recall_cases(
        "current task",
        ctx=_ctx(),
        limit=None,
        score_threshold=0.1,
    )

    assert recalled == cases
    assert search.find.await_args.kwargs["limit"] == len(case_uris)


@pytest.mark.asyncio
async def test_recalled_case_rule_loading_fails_closed_on_unreadable_rule(monkeypatch):
    first_uri = "viking://user/alice/memories/rules/items/first.md"
    broken_uri = "viking://user/alice/memories/rules/items/broken.md"
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    monkeypatch.setattr(
        service,
        "_load_rule",
        AsyncMock(side_effect=[_stored_rule("first"), ValueError("invalid Rule document")]),
    )
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id="file-operations",
        name="File operations",
        description="Rules for file operations.",
        rule_uris=[first_uri, broken_uri],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )

    with pytest.raises(ProcessingError, match="recalled Case references an unreadable Rule"):
        await service._load_rules_for_cases([case], _ctx(), strict=True)


def test_rule_renderer_keeps_an_oversized_rule_complete():
    content = "Never partially render this constraint. " * 80
    record = _record("oversized").model_copy(update={"content": content})
    rule = _stored_rule("oversized").model_copy(update={"record": record})

    rendered = _render_rules([(rule, 0.99)])

    assert content.strip() in rendered
    assert len(rendered) > 1_000


@pytest.mark.asyncio
async def test_compile_uses_search_pipeline_to_select_existing_rules(monkeypatch):
    class _Tracker:
        update_stage = AsyncMock()

    fs = _FS()
    compiler = SimpleNamespace()
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=compiler)),
        rules_config=None,
    )
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id="file-operations",
        name="File operations",
        description="Rules for editing files.",
        rule_uris=[],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )
    relevant = _stored_rule("relevant")
    archived = _stored_rule("archived", status="archived")
    deleted = _stored_rule("deleted", status="deleted")
    unrelated = _stored_rule("unrelated")
    recall_cases = AsyncMock(return_value=[case])
    score_rules = AsyncMock(return_value=([0.9, 0.8, 0.2], "jev_noul", None))
    build_plan = AsyncMock(
        return_value=RuleCompilePlan(
            mutations=[
                RuleMutation(
                    operation="UNCERTAIN",
                    evidence_spans=["delete files"],
                    reason="ambiguous",
                )
            ]
        )
    )
    monkeypatch.setattr(
        "openviking.service.rule_service.get_task_tracker",
        lambda: _Tracker(),
    )
    monkeypatch.setattr(service, "_ensure_rule_directories", AsyncMock())
    monkeypatch.setattr(service, "_collect_sources", AsyncMock(return_value=("", [])))
    monkeypatch.setattr(service, "_recall_cases", recall_cases)
    monkeypatch.setattr(
        service,
        "_load_rules_for_cases",
        AsyncMock(return_value=[relevant, archived, deleted, unrelated]),
    )
    monkeypatch.setattr(service, "_score_rules", score_rules)
    monkeypatch.setattr(service, "_build_compile_plan", build_plan)
    monkeypatch.setattr(service, "_compiler_prompt_token_limit", lambda _: 8_000)

    await service._compile(
        RuleCompileRequest(instruction="delete files"),
        ctx=_ctx(),
        task_id="rul_match",
    )

    assert recall_cases.await_args.kwargs["limit"] == 5
    assert recall_cases.await_args.kwargs["score_threshold"] == 0.1
    assert score_rules.await_args.args[0] == "delete files"
    assert score_rules.await_args.args[1] == [relevant, archived, unrelated]
    assert [rule.atomic_id for rule in build_plan.await_args.kwargs["rules"]] == [
        "relevant",
        "archived",
    ]


@pytest.mark.asyncio
async def test_search_raises_when_jev_and_vlm_are_both_unavailable(monkeypatch):
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    monkeypatch.setattr(
        service,
        "_score_rules_with_vlm",
        AsyncMock(side_effect=RuntimeError("VLM unavailable")),
    )

    with pytest.raises(
        UnavailableError,
        match="rule applicability cannot be determined safely",
    ):
        await service._score_rules(
            "delete a generated file",
            [SimpleNamespace()],
            _ctx(),
        )


def test_compile_request_requires_instruction_or_source():
    with pytest.raises(ValueError, match="instruction or sources"):
        RuleCompileRequest()


def test_compile_request_reuses_complete_add_resource_model():
    request = RuleCompileRequest(
        sources=[
            {
                "path": "space:home",
                "add_type": "feishu",
                "to": "viking://resources/team-rules",
                "args": {"feishu_access_token": "token"},
                "strict": True,
                "processing_mode": "vectors_only",
                "tags": ["rules"],
            },
            {"temp_file_id": "upload_rules"},
            {"resource_uri": "viking://resources/existing-rules"},
        ]
    )

    assert isinstance(request.sources[0], AddResourceRequest)
    assert request.sources[0].add_type == "feishu"
    assert request.sources[0].args == {"feishu_access_token": "token"}
    assert isinstance(request.sources[1], AddResourceRequest)
    assert request.sources[1].temp_file_id == "upload_rules"
    assert request.sources[2].resource_uri == "viking://resources/existing-rules"


@pytest.mark.asyncio
async def test_collect_sources_uses_shared_add_resource_ingestion(monkeypatch):
    ingest = AsyncMock(return_value={"root_uri": "viking://user/alice/resources/team-rules"})
    monkeypatch.setattr(
        "openviking.service.rule_service.ingest_add_resource_request",
        ingest,
    )
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(add_resource=AsyncMock()),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
        config=SimpleNamespace(),
    )
    monkeypatch.setattr(service, "_read_source", AsyncMock(return_value="parsed rules"))
    request = RuleCompileRequest(
        sources=[
            {
                "path": "https://example.com/team-rules.md",
                "args": {"header": "value"},
            }
        ]
    )

    text, uris = await service._collect_sources(request, _ctx())

    assert text == "parsed rules"
    assert uris == ["viking://user/alice/resources/team-rules"]
    add_request = ingest.await_args.args[0]
    assert isinstance(add_request, AddResourceRequest)
    assert add_request.args == {"header": "value"}
    assert add_request.instruction == ""
    assert ingest.await_args.kwargs["force_wait"] is True
    assert ingest.await_args.kwargs["force_internal_task"] is True


@pytest.mark.asyncio
async def test_collect_sources_waits_for_declared_connector(monkeypatch):
    ingest = AsyncMock(
        return_value={
            "status": "accepted",
            "task_id": "connector-task",
            "resource_id": "viking://user/alice/resources/team-rules",
        }
    )
    tracker = SimpleNamespace(
        wait=AsyncMock(
            return_value=TaskRecord(
                task_id="connector-task",
                task_type="connector_import",
                status=TaskStatus.COMPLETED,
                resource_id="viking://user/alice/resources/team-rules",
            )
        )
    )
    monkeypatch.setattr(
        "openviking.service.rule_service.ingest_add_resource_request",
        ingest,
    )
    monkeypatch.setattr("openviking.service.rule_service.get_task_tracker", lambda: tracker)
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(add_resource=AsyncMock()),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
        config=SimpleNamespace(),
    )
    monkeypatch.setattr(service, "_read_source", AsyncMock(return_value="parsed rules"))
    request = RuleCompileRequest(
        sources=[
            {
                "path": "space:home",
                "add_type": "feishu",
                "to": "viking://resources/team-rules",
                "args": {"feishu_access_token": "token"},
                "instruction": "Summarize this source",
            }
        ]
    )

    text, uris = await service._collect_sources(request, _ctx())

    assert text == "parsed rules"
    assert uris == ["viking://user/alice/resources/team-rules"]
    assert ingest.await_args.args[0].instruction == ""
    assert ingest.await_args.kwargs["force_wait"] is False
    assert ingest.await_args.kwargs["force_internal_task"] is True
    tracker.wait.assert_awaited_once_with(
        "connector-task",
        account_id="acme",
        user_id="alice",
        timeout=None,
    )


@pytest.mark.asyncio
async def test_collect_sources_surfaces_declared_connector_failure(monkeypatch):
    monkeypatch.setattr(
        "openviking.service.rule_service.ingest_add_resource_request",
        AsyncMock(
            return_value={
                "status": "accepted",
                "task_id": "connector-task",
                "resource_id": "viking://user/alice/resources/team-rules",
            }
        ),
    )
    tracker = SimpleNamespace(
        wait=AsyncMock(
            return_value=TaskRecord(
                task_id="connector-task",
                task_type="connector_import",
                status=TaskStatus.FAILED,
                error="source unavailable",
            )
        )
    )
    monkeypatch.setattr("openviking.service.rule_service.get_task_tracker", lambda: tracker)
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
        config=SimpleNamespace(),
    )
    request = RuleCompileRequest(
        sources=[
            {
                "path": "space:home",
                "add_type": "feishu",
                "to": "viking://resources/team-rules",
            }
        ]
    )

    with pytest.raises(ProcessingError, match="source unavailable"):
        await service._collect_sources(request, _ctx())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        {"path": "https://example.com/screenshot.PNG?download=1"},
        {"temp_file_id": "upload_123", "source_name": "meeting.mp3"},
    ],
)
async def test_collect_sources_rejects_obvious_multimodal_inputs_before_ingestion(
    monkeypatch,
    source,
):
    ingest = AsyncMock()
    monkeypatch.setattr(
        "openviking.service.rule_service.ingest_add_resource_request",
        ingest,
    )
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
        config=SimpleNamespace(),
    )

    with pytest.raises(InvalidArgumentError, match="multimodal"):
        await service._collect_sources(RuleCompileRequest(sources=[source]), _ctx())

    ingest.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "https://example.com/rules.zip",
        "https://github.com/acme/rules.git",
    ],
)
async def test_collect_sources_allows_archive_and_git_inputs(monkeypatch, path):
    ingest = AsyncMock(return_value={"root_uri": "viking://user/alice/resources/rules"})
    monkeypatch.setattr(
        "openviking.service.rule_service.ingest_add_resource_request",
        ingest,
    )
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
        config=SimpleNamespace(),
    )
    monkeypatch.setattr(service, "_read_source", AsyncMock(return_value="parsed rules"))

    text, _ = await service._collect_sources(
        RuleCompileRequest(sources=[{"path": path}]),
        _ctx(),
    )

    assert text == "parsed rules"
    assert ingest.await_args.kwargs["force_wait"] is True


@pytest.mark.asyncio
async def test_collect_sources_rejects_binary_resource_reference():
    binary_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    fs = SimpleNamespace(
        stat=AsyncMock(return_value={"isDir": False}),
        read_visible=AsyncMock(side_effect=binary_error),
    )
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )

    with pytest.raises(InvalidArgumentError, match="readable text"):
        await service._collect_sources(
            RuleCompileRequest(sources=[{"resource_uri": "viking://resources/screenshot.png"}]),
            _ctx(),
        )


@pytest.mark.asyncio
async def test_collect_sources_validates_sources_after_text_budget_is_exhausted(monkeypatch):
    service = RuleService()
    service.set_dependencies(
        fs=SimpleNamespace(),
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )
    read_source = AsyncMock(side_effect=["x" * 128_000, ""])
    monkeypatch.setattr(service, "_read_source", read_source)

    with pytest.raises(InvalidArgumentError, match="readable text"):
        await service._collect_sources(
            RuleCompileRequest(
                sources=[
                    {"resource_uri": "viking://resources/large.md"},
                    {"resource_uri": "viking://resources/image.png"},
                ]
            ),
            _ctx(),
        )

    assert read_source.await_args_list[1].args[2] == 1


@pytest.mark.asyncio
async def test_read_source_keeps_text_from_mixed_directory():
    root = "viking://resources/mixed"
    text_uri = f"{root}/rules.md"
    image_uri = f"{root}/screenshot.png"

    async def stat(uri, *_args, **_kwargs):
        return {"isDir": uri == root}

    async def read_visible(uri, *_args, **_kwargs):
        if uri == image_uri:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return "Always confirm destructive operations."

    fs = SimpleNamespace(
        stat=AsyncMock(side_effect=stat),
        ls=AsyncMock(return_value=SimpleNamespace(entries=[image_uri, text_uri])),
        read_visible=AsyncMock(side_effect=read_visible),
    )
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(),
        rules_config=None,
    )

    text = await service._read_source(root, _ctx(), 10_000)

    assert text_uri in text
    assert "Always confirm destructive operations." in text
    assert image_uri not in text


def test_compile_plan_rejects_evidence_not_present_in_input():
    plan = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="NEW",
                record=_record(),
                case=RuleCaseRef(
                    name="File operations",
                    description="Rules for editing files.",
                ),
                evidence_spans=["invented instruction"],
            )
        ]
    )

    with pytest.raises(VLMFailedError, match="evidence not found"):
        RuleService._validate_plan_evidence(
            plan,
            material="Confirm before deleting files.",
        )


def test_compile_plan_requires_update_preservation_proof():
    old = _stored_rule("old-rule")
    plan = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="UPDATE",
                target_ids=[old.atomic_id],
                record=_record("updated"),
                evidence_spans=["confirm before deleting files"],
                applicability_evidence=f"existing:{old.atomic_id}",
            )
        ]
    )

    with pytest.raises(VLMFailedError, match="requires carried_over"):
        RuleService._validate_compile_plan(
            plan,
            material="confirm before deleting files",
            existing_rules=[old],
        )


def test_compile_plan_allows_archived_rule_update_and_replace():
    archived = _stored_rule("archived-rule", status="archived")
    update = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="UPDATE",
                target_ids=[archived.atomic_id],
                record=_record("archived-rule"),
                evidence_spans=["reassert archived rule"],
                applicability_evidence=f"existing:{archived.atomic_id}",
                carried_over={
                    "existing_requirements": [
                        "Ask for confirmation before deleting files.",
                    ]
                },
            )
        ]
    )
    replacement = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="REPLACE",
                target_ids=[archived.atomic_id],
                record=_record("replacement"),
                evidence_spans=["replace archived rule"],
                applicability_evidence="replace archived rule",
            )
        ]
    )

    RuleService._validate_compile_plan(
        update,
        material="reassert archived rule",
        existing_rules=[archived],
    )
    RuleService._validate_compile_plan(
        replacement,
        material="replace archived rule",
        existing_rules=[archived],
    )


def test_compile_plan_rejects_unsupported_applicability():
    plan = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="NEW",
                record=_record(),
                case=RuleCaseRef(
                    name="File operations",
                    description="Rules for editing files.",
                ),
                evidence_spans=["Always confirm deletion."],
                applicability_evidence="when deleting production files",
            )
        ]
    )

    with pytest.raises(VLMFailedError, match="applicability_evidence"):
        RuleService._validate_compile_plan(
            plan,
            material="Always confirm deletion.",
            existing_rules=[],
        )


@pytest.mark.asyncio
async def test_compile_plan_repairs_semantically_invalid_response():
    old = _stored_rule("old-rule")
    first = {
        "mutations": [
            {
                "operation": "NEW",
                "record": _record("duplicate").model_dump(),
                "case": {
                    "name": "File operations",
                    "description": "Rules for editing files.",
                },
                "evidence_spans": ["confirm before deleting files"],
                "applicability_evidence": "",
            }
        ]
    }
    repaired = {
        "mutations": [
            {
                "operation": "NOOP",
                "target_ids": [old.atomic_id],
                "evidence_spans": ["confirm before deleting files"],
                "reason": "The active rule already entails the instruction.",
            }
        ]
    }
    compiler = SimpleNamespace(
        get_completion_async=AsyncMock(side_effect=[json.dumps(first), json.dumps(repaired)])
    )
    service = RuleService()

    plan = await service._build_compile_plan(
        instruction="confirm before deleting files",
        source_text="",
        cases=[],
        rules=[old],
        ctx=_ctx(),
        vlm=compiler,
        prompt_token_limit=20_000,
    )

    assert plan.mutations[0].operation == "NOOP"
    assert compiler.get_completion_async.await_count == 2
    repair_prompt = compiler.get_completion_async.await_args_list[1].kwargs["prompt"]
    assert "failed deterministic validation" in repair_prompt
    assert "applicability_evidence" in repair_prompt


@pytest.mark.asyncio
async def test_compile_plan_rejects_oversized_prompt_without_trimming_state():
    service = RuleService()
    compiler = SimpleNamespace(get_completion_async=AsyncMock())
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id="file-operations",
        name="File operations",
        description="Rules for editing files.",
        rule_uris=[],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )
    rule = _stored_rule("existing-rule")
    full_prompt = service._render_compile_prompt(
        instruction="confirm before deleting files",
        source_text="",
        case_data=[
            {
                "case_id": case.case_id,
                "name": case.name,
                "description": case.description,
            }
        ],
        rule_data=[
            {
                **rule.record.model_dump(),
                "atomic_id": rule.atomic_id,
                "revision": rule.revision,
                "status": rule.status,
                "case_id": rule.case_id,
            }
        ],
    )

    with pytest.raises(InvalidArgumentError, match="refusing to omit existing cases or rules"):
        await service._build_compile_plan(
            instruction="confirm before deleting files",
            source_text="",
            cases=[case],
            rules=[rule],
            ctx=_ctx(),
            vlm=compiler,
            prompt_token_limit=estimate_text_tokens(full_prompt) - 1,
        )

    compiler.get_completion_async.assert_not_awaited()


def test_compile_repair_prompt_rejects_overflow_without_truncation():
    response = "previous response " * 500
    validation_error = "validation error " * 200
    full_repair = RuleService._render_compile_repair_prompt(
        "base prompt",
        response=response,
        validation_error=validation_error,
        prompt_token_limit=None,
    )

    assert response in full_repair
    assert validation_error in full_repair
    with pytest.raises(
        InvalidArgumentError,
        match="refusing to truncate validation diagnostics or compiler state",
    ):
        RuleService._render_compile_repair_prompt(
            "base prompt",
            response=response,
            validation_error=validation_error,
            prompt_token_limit=estimate_text_tokens(full_repair) - 1,
        )


def test_compile_material_batches_preserve_all_text_within_budget():
    instruction = ("First rule paragraph.\n\n" * 80) + ("必须先确认。\n" * 80)
    source = ("## Source\nSecond rule paragraph.\n" * 100) + "final rule"

    batches = _build_material_batches(instruction, source, max_tokens=300)

    assert len(batches) > 2
    assert "".join(batch_instruction for batch_instruction, _ in batches) == instruction
    assert "".join(batch_source for _, batch_source in batches) == source
    assert all(
        estimate_text_tokens(batch_instruction) + estimate_text_tokens(batch_source) <= 300
        for batch_instruction, batch_source in batches
    )


@pytest.mark.asyncio
async def test_compile_batches_sequentially_and_writes_once(monkeypatch):
    class _Tracker:
        update_stage = AsyncMock()

    fs = _FS()
    compiler = SimpleNamespace(get_completion_async=AsyncMock())
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=compiler)),
        rules_config=None,
    )
    monkeypatch.setattr(
        "openviking.service.rule_service.get_task_tracker",
        lambda: _Tracker(),
    )
    monkeypatch.setattr(service, "_ensure_rule_directories", AsyncMock())
    instruction = "A policy paragraph requiring confirmation.\n\n" * 400
    monkeypatch.setattr(
        service,
        "_collect_sources",
        AsyncMock(return_value=("", [])),
    )
    monkeypatch.setattr(service, "_recall_cases", AsyncMock(return_value=[]))
    monkeypatch.setattr(service, "_load_rules_for_cases", AsyncMock(return_value=[]))
    prompt_token_limit = (
        estimate_text_tokens(
            service._render_compile_prompt(
                instruction="",
                source_text="",
                case_data=[],
                rule_data=[],
            )
        )
        + 600
    )
    monkeypatch.setattr(
        service,
        "_compiler_prompt_token_limit",
        lambda _: prompt_token_limit,
    )

    calls = 0

    async def compile_plan(*, prompt, thinking):
        nonlocal calls
        assert thinking is True
        assert estimate_text_tokens(prompt) <= prompt_token_limit
        if calls == 0:
            response = {
                "mutations": [
                    {
                        "operation": "NEW",
                        "record": _record().model_dump(),
                        "case": {
                            "name": "File operations",
                            "description": "Rules for editing files.",
                        },
                        "evidence_spans": ["A policy paragraph"],
                        "applicability_evidence": "A policy paragraph",
                    }
                ],
                "reason": "created",
            }
        else:
            assert '"atomic_id": "confirm-delete"' in prompt
            response = {
                "mutations": [
                    {
                        "operation": "NOOP",
                        "target_ids": ["confirm-delete"],
                        "evidence_spans": ["A policy paragraph"],
                    }
                ],
                "reason": "already covered",
            }
        calls += 1
        return json.dumps(response)

    compiler.get_completion_async.side_effect = compile_plan

    result = await service._compile(
        RuleCompileRequest(instruction=instruction),
        ctx=_ctx(),
        task_id="rul_batch",
    )

    assert result["batch_count"] == calls
    assert calls > 1
    assert result["operations"][0]["operation"] == "NEW"
    assert all(item["operation"] == "NOOP" for item in result["operations"][1:])
    assert len(result["written_uris"]) == 2
    assert fs.batch is not None
    assert len(fs.batch["operations"]) == 2


@pytest.mark.asyncio
async def test_compile_does_not_carry_deleted_rules_into_later_batches(monkeypatch):
    class _Tracker:
        update_stage = AsyncMock()

    fs = _FS()
    compiler = SimpleNamespace()
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=compiler)),
        rules_config=None,
    )
    archived = _stored_rule("archived-rule", status="archived")
    case = StoredCase(
        schema=CASE_SCHEMA,
        case_id=archived.case_id,
        name="File operations",
        description="Rules for file operations.",
        rule_uris=[archived.uri],
        uri="viking://user/alice/memories/rules/cases/file-operations.md",
        content="case",
    )
    monkeypatch.setattr(
        "openviking.service.rule_service.get_task_tracker",
        lambda: _Tracker(),
    )
    monkeypatch.setattr(service, "_ensure_rule_directories", AsyncMock())
    monkeypatch.setattr(service, "_collect_sources", AsyncMock(return_value=("", [])))
    monkeypatch.setattr(service, "_compiler_prompt_token_limit", lambda _: 8_000)
    monkeypatch.setattr(
        "openviking.service.rule_service._build_material_batches",
        lambda *args, **kwargs: [("replace archived rule", ""), ("second batch", "")],
    )
    monkeypatch.setattr(
        service,
        "_find_relevant_rules",
        AsyncMock(
            side_effect=[
                SimpleNamespace(cases=[case], selected=[(archived, 0.9)]),
                SimpleNamespace(cases=[], selected=[]),
            ]
        ),
    )
    supplied_rules: list[list[tuple[str, str]]] = []

    async def build_plan(**kwargs):
        supplied_rules.append([(rule.atomic_id, rule.status) for rule in kwargs["rules"]])
        if len(supplied_rules) == 1:
            return RuleCompilePlan(
                mutations=[
                    RuleMutation(
                        operation="REPLACE",
                        target_ids=[archived.atomic_id],
                        record=_record("replacement"),
                        evidence_spans=["replace archived rule"],
                    )
                ]
            )
        return RuleCompilePlan(
            mutations=[
                RuleMutation(
                    operation="NOOP",
                    target_ids=["replacement"],
                    evidence_spans=["second batch"],
                )
            ]
        )

    monkeypatch.setattr(service, "_build_compile_plan", build_plan)

    await service._compile(
        RuleCompileRequest(instruction="replace archived rule"),
        ctx=_ctx(),
        task_id="rul_status_filter",
    )

    assert supplied_rules == [
        [("archived-rule", "archived")],
        [("replacement", "active")],
    ]


@pytest.mark.asyncio
async def test_compile_does_not_write_when_a_later_batch_fails(monkeypatch):
    class _Tracker:
        update_stage = AsyncMock()

    fs = _FS()
    compiler = SimpleNamespace()
    service = RuleService()
    service.set_dependencies(
        fs=fs,
        resources=SimpleNamespace(),
        search=SimpleNamespace(),
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=compiler)),
        rules_config=None,
    )
    monkeypatch.setattr(
        "openviking.service.rule_service.get_task_tracker",
        lambda: _Tracker(),
    )
    monkeypatch.setattr(service, "_ensure_rule_directories", AsyncMock())
    monkeypatch.setattr(
        service,
        "_collect_sources",
        AsyncMock(return_value=("", [])),
    )
    monkeypatch.setattr(service, "_recall_cases", AsyncMock(return_value=[]))
    monkeypatch.setattr(service, "_load_rules_for_cases", AsyncMock(return_value=[]))
    prompt_token_limit = (
        estimate_text_tokens(
            service._render_compile_prompt(
                instruction="",
                source_text="",
                case_data=[],
                rule_data=[],
            )
        )
        + 600
    )
    monkeypatch.setattr(
        service,
        "_compiler_prompt_token_limit",
        lambda _: prompt_token_limit,
    )
    first_plan = RuleCompilePlan(
        mutations=[
            RuleMutation(
                operation="NEW",
                record=_record(),
                case=RuleCaseRef(
                    name="File operations",
                    description="Rules for editing files.",
                ),
                evidence_spans=["A policy paragraph"],
            )
        ]
    )
    monkeypatch.setattr(
        service,
        "_build_compile_plan",
        AsyncMock(side_effect=[first_plan, VLMFailedError("second batch failed")]),
    )

    with pytest.raises(VLMFailedError, match="second batch failed"):
        await service._compile(
            RuleCompileRequest(instruction="A policy paragraph requiring confirmation.\n\n" * 400),
            ctx=_ctx(),
            task_id="rul_batch_failure",
        )

    assert fs.batch is None
