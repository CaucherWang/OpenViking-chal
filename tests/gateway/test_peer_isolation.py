"""Trial isolation through real gateway HTTP, persisted capture and OV HTTP calls."""

import asyncio
import copy

import pytest
from aiohttp import web

from openviking_gateway.client import VikingError
from openviking_gateway.isolation import peer_identity, response_key, scoped_viking
from openviking_gateway.models import Policy
from openviking_gateway.storage import digest


def trial_headers(key, peer):
    return {
        "Authorization": "Bearer " + key["key"],
        "X-OpenViking-Actor-Peer": peer,
        "X-OpenViking-Memory-Isolation": "trial",
        "X-OpenViking-Session": "same-client-session",
    }


@pytest.mark.parametrize("peer", ["", "../b", "a/b", "__self"])
def test_invalid_peer_cannot_enable_isolation(peer):
    with pytest.raises(Exception) as error:
        peer_identity({
            "x-openviking-actor-peer": peer, "x-openviking-memory-isolation": "trial",
        })
    assert error.value.status_code == 400


async def test_concurrent_trials_recall_and_capture_their_own_peer(running_gateway):
    app, client, _, key, seen, writes = running_gateway
    requests = []
    policies = []
    delivered = asyncio.Event()

    async def ov(request):
        if request.path == "/api/v1/search/search":
            body = await request.json()
            peer = request.headers.get("X-OpenViking-Actor-Peer")
            requests.append((peer, body))
            root = f"viking://user/alice/peers/{peer}/memories/events/own.md"
            await asyncio.sleep(0.005)
            return web.json_response({
                "status": "ok",
                "result": {"entries": [{"uri": root}], "rendered": peer + "-memory"},
            })
        if request.path == "/api/v1/sessions":
            policies.append(await request.json())
        if request.path.endswith("/messages/batch"):
            writes.append(await request.json())
            if {m["peer_id"] for w in writes for m in w["messages"]} == {"trial-a", "trial-b"}:
                delivered.set()
            return web.json_response({"status": "ok", "result": {"pending_tokens": 20}})
        return None

    app.state.test_backend["handler"] = ov
    body = {
        "model": "model", "store": False,
        "messages": [{"role": "user", "content": "Recall deployment memory"}],
    }
    replies = await asyncio.gather(*(
        client.post("/v1/chat/completions", headers=trial_headers(key, peer), json=body)
        for peer in ("trial-a", "trial-b")
    ))
    assert all(reply.status_code == 200 for reply in replies)
    sent = [raw.decode() for path, raw, _ in seen if path.endswith("/chat/completions")]
    assert any("trial-a-memory" in raw and "trial-b-memory" not in raw for raw in sent)
    assert any("trial-b-memory" in raw and "trial-a-memory" not in raw for raw in sent)
    assert {peer for peer, _ in requests} == {"trial-a", "trial-b"}
    assert all(body["peer_scope"] == "actor" and body["context_type"] == ["memory"]
               and "target_uri" not in body for _, body in requests)
    # Parent memories and other peers never enter the model provider's headers.
    assert all(not name.lower().startswith("x-openviking-")
               for _, _, headers in seen for name in headers)
    with app.state.store.connect() as conn:
        rows = list(conn.execute("SELECT scope,session FROM capture"))
    assert len(rows) == 2 and len({scope for scope, _ in rows}) == 2
    for scope, session in rows:
        state = (await app.state.store.capture.get(scope, session)).value
        assert state["peer_id"] in {"trial-a", "trial-b"}
        assert state["memory_isolation"] == "trial"
    # Confirm a completed turn, deliver it and check extraction configuration.
    continued = copy.deepcopy(body)
    continued["messages"] += [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "Continue the deployment"},
    ]
    for peer in ("trial-a", "trial-b"):
        reply = await client.post(
            "/v1/chat/completions", headers=trial_headers(key, peer), json=continued,
        )
        assert reply.status_code == 200
    await asyncio.wait_for(delivered.wait(), timeout=5)
    assert policies and all(
        p["memory_policy"]["self"]["enabled"] is False
        and p["memory_policy"]["peer"]["enabled"] is True for p in policies
    )
    assert writes and {m["peer_id"] for w in writes for m in w["messages"]} == {
        "trial-a", "trial-b",
    }


async def test_dirty_parent_blocks_before_model_call(running_gateway):
    app, client, _, key, seen, _ = running_gateway

    async def ov(request):
        if request.path == "/api/v1/fs/ls" and request.query.get("uri") == (
            "viking://user/alice/memories"
        ):
            return web.json_response({
                "status": "ok", "result": [{"name": "profile.md", "isDir": False}],
            })
        if request.path == "/api/v1/content/read":
            return web.json_response({"status": "ok", "result": "old Trial memory"})
        return None

    app.state.test_backend["handler"] = ov
    reply = await client.post(
        "/v1/chat/completions", headers=trial_headers(key, "trial-a"),
        json={"model": "model", "messages": [{"role": "user", "content": "deploy"}]},
    )
    assert reply.status_code == 409
    assert reply.json()["reason"] == "parent_memory_not_empty"
    assert seen == []


async def test_dev_isolation_uses_clean_parent_and_peer_profile(running_gateway):
    app, client, _, key, seen, _ = running_gateway
    app.state.test_backend["auth_mode"] = "dev"
    app.state.viking.identities.clear()
    paths = []

    async def ov(request):
        if request.path.startswith("/api/"):
            paths.append((request.path, request.query.get("uri"), dict(request.headers)))
        if request.path == "/api/v1/search/search":
            peer = request.headers["X-OpenViking-Actor-Peer"]
            root = f"viking://user/benchmark-trials-v1/peers/{peer}/memories/events/x.md"
            return web.json_response({
                "status": "ok", "result": {"entries": [{"uri": root}], "rendered": "peer memory"},
            })
        return None

    app.state.test_backend["handler"] = ov
    reply = await client.post(
        "/v1/chat/completions", headers=trial_headers(key, "trial-a"),
        json={"model": "model", "messages": [{"role": "user", "content": "deploy memory"}]},
    )
    assert reply.status_code == 200, reply.text
    assert reply.headers["x-openviking-memory-user"] == "benchmark-trials-v1"
    uris = [uri for _, uri, _ in paths if uri]
    assert "viking://user/benchmark-trials-v1/memories" in uris
    assert any("/peers/trial-a/memories/profile.md" in uri for uri in uris)
    assert all("viking://user/default/memories" not in uri for uri in uris)
    scoped_calls = [headers for path, _, headers in paths if path == "/api/v1/search/search"]
    assert scoped_calls[0]["X-OpenViking-User"] == "benchmark-trials-v1"
    assert seen


async def test_unavailable_isolation_does_not_degrade(running_gateway):
    app, client, _, key, seen, _ = running_gateway

    async def broken(*args):
        raise OSError("storage unavailable")

    app.state.kernel.prepare = broken
    reply = await client.post(
        "/v1/chat/completions", headers=trial_headers(key, "trial-a"),
        json={"model": "model", "messages": [{"role": "user", "content": "deploy memory"}]},
    )
    assert reply.status_code == 503
    assert seen == []


def test_explicit_tools_cannot_read_other_peers_or_write_parent(credential):
    from openviking_gateway.client import VikingClient

    original = VikingClient(None, "http://ov", "0")
    scoped = scoped_viking(
        original, {**credential, "peer_id": "a", "memory_isolation": "trial"}, ("session-a",),
    )
    root = "viking://user/alice/peers/a"
    assert scoped.peer_tool_params({"name": "find", "arguments": {"query": "x"}})[
        "arguments"
    ]["target_uri"] == root
    search = scoped.peer_tool_params({
        "name": "search", "arguments": {"query": "x", "mode": "context", "peer_scope": "all"},
    })["arguments"]
    assert search["peer_scope"] == "actor" and search["context_type"] == ["memory"]
    assert "target_uri" not in search
    for name, args in (
        ("read", {"uris": ["viking://user/alice/peers/b/memories/x.md"]}),
        ("write", {"uri": "viking://user/alice/memories/profile.md"}),
        ("write", {"uri": root + "/../b/memories/x"}),
        ("search", {"query": "x", "session_id": "session-b"}),
        ("add_resource", {"path": "https://example.com"}),
    ):
        with pytest.raises(VikingError):
            scoped.peer_tool_params({"name": name, "arguments": args})
    assert original.peer_id is None
    assert response_key(credential, "r") == digest(credential["id"] + "r")
    assert response_key({**credential, "peer_id": "a"}, "r") != response_key(
        {**credential, "peer_id": "b"}, "r",
    )


async def test_client_uses_valid_ov_api_shapes_and_checks_resume_policy(credential):
    from openviking.server.routers.search import SearchRequest
    from openviking.server.routers.sessions import CommitRequest, CreateSessionRequest
    from openviking_gateway.client import VikingClient

    scoped = scoped_viking(
        VikingClient(None, "http://ov", "0"),
        {**credential, "peer_id": "a", "memory_isolation": "trial"},
    )
    bodies = {}

    async def request(method, path, key, body=None, timeout=30, peer=None):
        bodies[path] = body
        if path == "/health":
            return {
                "version": "0.5.0",
                "role": "user",
                "account_id": "tenant",
                "user_id": "alice",
            }
        if path == "/api/v1/search/search":
            SearchRequest.model_validate(body)
            return {"entries": [], "rendered": ""}
        if path == "/api/v1/sessions":
            CreateSessionRequest.model_validate(body)
            return {}
        if path.endswith("/commit"):
            CommitRequest.model_validate(body)
            assert "memory_policy" not in body
            return {}
        return {}

    scoped.request = request
    await scoped.recall("key", "deploy", Policy(), [], 100)
    await scoped.create_session("key", "s", Policy())
    await scoped.commit("key", "s")
    persisted_policy = bodies["/api/v1/sessions"]["memory_policy"]

    async def conflict(method, path, key, body=None, timeout=30, peer=None):
        if method == "POST":
            raise VikingError("openviking_http_409", 409)
        return {"memory_policy": persisted_policy}

    scoped.request = conflict
    await scoped.create_session("key", "s", Policy())
    persisted_policy["self"]["enabled"] = True
    with pytest.raises(VikingError, match="trial_session_policy_mismatch"):
        await scoped.create_session("key", "s", Policy())
