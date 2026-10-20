from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

import openviking.server.routers.rules as rules_router
from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.asyncio
async def test_rule_compile_and_search_http_contract(monkeypatch):
    ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.USER)
    task = SimpleNamespace(
        to_dict=lambda: {
            "task_id": "rul_123",
            "task_type": "rule_compile",
            "status": "pending",
        }
    )
    rules = SimpleNamespace(
        create_compile=AsyncMock(return_value=task),
        search_rules=AsyncMock(
            return_value={
                "rules": [
                    {
                        "atomic_id": "confirm-delete",
                        "content": "# Complete rule",
                        "probability": 0.91,
                    }
                ],
                "judge": "jev_noul",
            }
        ),
    )
    monkeypatch.setattr(
        rules_router,
        "get_service",
        lambda: SimpleNamespace(rules=rules),
    )
    app = FastAPI()
    app.include_router(rules_router.router)
    app.dependency_overrides[get_request_context] = lambda: ctx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        compile_response = await client.post(
            "/api/v1/rules/compile",
            json={"instruction": "Confirm before deleting files."},
        )
        search_response = await client.post(
            "/api/v1/rules/search",
            json={"task_context": "Delete a generated file."},
        )

    assert compile_response.status_code == 202
    assert compile_response.json()["result"]["task_id"] == "rul_123"
    assert search_response.status_code == 200
    assert search_response.json()["result"]["rules"][0]["atomic_id"] == "confirm-delete"
    rules.create_compile.assert_awaited_once()
    rules.search_rules.assert_awaited_once()
