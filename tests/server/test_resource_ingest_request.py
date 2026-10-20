from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.rule_resource_ingest import (
    AddResourceRequest,
    ingest_add_resource_request,
)
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.asyncio
async def test_shared_ingest_forwards_complete_request_with_rule_overrides():
    resources = SimpleNamespace(
        add_resource=AsyncMock(
            return_value={"root_uri": "viking://user/alice/resources/team-rules"}
        )
    )
    request = AddResourceRequest(
        path="space:home",
        add_type="feishu",
        to="viking://resources/team-rules",
        args={"feishu_access_token": "token"},
        strict=True,
        wait=False,
        internal_task=False,
        processing_mode="vectors_only",
        tags=["rules"],
        tag_mode="append",
    )
    ctx = RequestContext(
        user=UserIdentifier("acme", "alice"),
        role=Role.USER,
    )

    result = await ingest_add_resource_request(
        request,
        ctx,
        resources=resources,
        config=None,
        force_wait=True,
        force_internal_task=True,
    )

    assert result["root_uri"].endswith("/team-rules")
    kwargs = resources.add_resource.await_args.kwargs
    assert kwargs["path"] == "space:home"
    assert kwargs["add_type"] == "feishu"
    assert kwargs["to"] == "viking://resources/team-rules"
    assert kwargs["args"] == {"feishu_access_token": "token"}
    assert kwargs["strict"] is True
    assert kwargs["processing_mode"] == "vectors_only"
    assert kwargs["tags"] == ["rules"]
    assert kwargs["tag_mode"] == "append"
    assert kwargs["wait"] is True
    assert kwargs["internal_task"] is True
