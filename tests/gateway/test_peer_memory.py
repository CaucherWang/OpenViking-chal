"""Peer-only profiles use the regular Gateway request and session APIs."""

import asyncio

import pytest
from aiohttp import web

from openviking_gateway.client import VikingError
from openviking_gateway.peer_memory import memory_root, tool_arguments
from openviking_gateway.proxy import context_override

PARENT = "viking://user/alice/memories"


def peer_headers(key, peer):
    return {
        "Authorization": "Bearer " + key["key"],
        "X-OpenViking-Actor-Peer": peer,
        "X-OpenViking-Session": f"session-{peer}",
        "X-OpenViking-Context": '{"recall_peer_scope":"actor","peer_allow_user_memory":false}',
    }


async def test_peer_only_profile_keeps_read_and_write_in_the_peer(running_gateway):
    app, client, _, key, seen, writes = running_gateway
    recalls, profiles, policies, session_peers = {}, set(), {}, {}

    async def backend(request):
        peer = request.headers.get("X-OpenViking-Actor-Peer")
        if request.path == "/api/v1/search/search":
            recalls[peer] = await request.json()
            uri = memory_root("alice", peer) + "/events/own.md"
            return web.json_response(
                {
                    "result": {"entries": [{"uri": uri}], "rendered": f"memory from {peer}"},
                }
            )
        if request.path == "/api/v1/content/read":
            profiles.add(request.query["uri"])
        if request.path == "/api/v1/sessions":
            body = await request.json()
            policies[body["session_id"]] = body["memory_policy"]
        if request.path.endswith("/messages/batch"):
            session = request.path.split("/")[4]
            session_peers.setdefault(session, set()).update(
                message["peer_id"]
                for message in (await request.json())["messages"]
                if message["role"] == "user"
            )
        return None

    app.state.test_backend["handler"] = backend
    replies = await asyncio.gather(
        *(
            client.post(
                "/v1/chat/completions",
                headers=peer_headers(key, peer),
                json={
                    "model": "model",
                    "messages": [{"role": "user", "content": f"Input marker for {peer}"}],
                },
            )
            for peer in ("trial-a", "trial-b")
        )
    )
    assert all(reply.status_code == 200 for reply in replies)
    assert set(recalls) == {"trial-a", "trial-b"}
    for peer, recall in recalls.items():
        assert recall["peer_scope"] == "actor"
        assert recall["context_type"] == ["memory"]
        assert PARENT in recall["exclude_uris"]
        assert memory_root("alice", peer) + "/profile.md" in profiles
    assert all(not uri.startswith(PARENT) for uri in profiles)
    payloads = [raw for path, raw, _ in seen if path.endswith("/chat/completions")]
    assert len(payloads) == 2
    for peer, other in (("trial-a", "trial-b"), ("trial-b", "trial-a")):
        [payload] = [raw for raw in payloads if f"memory from {peer}".encode() in raw]
        assert f"memory from {other}".encode() not in payload
        assert f"Input marker for {peer}".encode() in payload
        assert f"Input marker for {other}".encode() not in payload
    assert len(policies) == 2
    assert session_peers.keys() == policies.keys()
    assert {frozenset(peers) for peers in session_peers.values()} == {
        frozenset({"trial-a"}),
        frozenset({"trial-b"}),
    }
    for policy in policies.values():
        assert policy["self"]["enabled"] is False
        assert policy["peer"]["enabled"] is True
        assert policy["working_memory"]["enabled"] is False
    assert {
        message["peer_id"]
        for batch in writes
        for message in batch["messages"]
        if message["role"] == "user"
    } == {"trial-a", "trial-b"}


async def test_unscoped_recall_reply_is_discarded(running_gateway):
    app, client, _, key, seen, _ = running_gateway

    async def backend(request):
        if request.path == "/api/v1/search/search":
            return web.json_response(
                {
                    "result": {
                        "entries": [{"uri": PARENT + "/events/old.md"}],
                        "rendered": "unexpected shared memory",
                    },
                }
            )

    app.state.test_backend["handler"] = backend
    reply = await client.post(
        "/v1/chat/completions",
        headers=peer_headers(key, "trial-a"),
        json={"model": "model", "messages": [{"role": "user", "content": "Recall deployment"}]},
    )
    assert reply.status_code == 200
    assert seen and all(b"unexpected shared memory" not in raw for _, raw, _ in seen)


@pytest.mark.parametrize("name", ["read", "write"])
def test_explicit_tools_reject_user_memory(name):
    root = memory_root("alice", "trial-a")
    with pytest.raises(VikingError, match="tool_outside_peer_memory"):
        tool_arguments(root, name, {"uri": PARENT + "/profile.md"})
    assert tool_arguments(root, name, {"uri": root + "/notes.md"})["uri"] == root + "/notes.md"


def test_profile_restriction_cannot_be_relaxed_by_client():
    policy, _ = context_override(
        {"peer_allow_user_memory": False}, '{"peer_allow_user_memory":true}'
    )
    assert policy["peer_allow_user_memory"] is False


async def test_peer_only_requests_disable_stored_provider_responses(running_gateway):
    app, client, _, key, seen, _ = running_gateway
    headers = peer_headers(key, "trial-a")
    reply = await client.post(
        "/v1/responses",
        headers=headers,
        json={"model": "model", "input": "Hello", "store": True},
    )
    assert reply.status_code == 200
    assert b'"store":false' in seen[-1][1]
    for peer in ("trial-a", "trial-b"):
        response = await client.get("/v1/responses/resp-1", headers=peer_headers(key, peer))
        assert response.status_code == 404
    stateful = await client.post(
        "/v1/responses",
        headers=headers,
        json={"model": "model", "input": "Continue", "previous_response_id": "resp-1"},
    )
    assert stateful.status_code == 503
    compact = await client.post(
        "/v1/responses/compact",
        headers=headers,
        json={"model": "model", "input": "Continue", "previous_response_id": "resp-1"},
    )
    assert compact.status_code == 400
    referenced = await client.post(
        "/v1/responses",
        headers=headers,
        json={"model": "model", "input": [{"type": "item_reference", "id": "item-trip"}]},
    )
    assert referenced.status_code == 400
    assert len(seen) == 1


async def test_restricted_session_does_not_replay_user_memory(running_gateway):
    app, client, _, key, seen, _ = running_gateway

    async def backend(request):
        if request.path == "/api/v1/search/search":
            body = await request.json()
            if PARENT in body["exclude_uris"]:
                return web.json_response({"result": {"entries": [], "rendered": ""}})
            return web.json_response(
                {
                    "result": {
                        "entries": [{"uri": PARENT + "/events/shared.md"}],
                        "rendered": "cached shared memory",
                    },
                }
            )

    app.state.test_backend["handler"] = backend
    headers = peer_headers(key, "trial-a")
    body = {"model": "model", "messages": [{"role": "user", "content": "Recall deployment"}]}
    normal = dict(headers)
    del normal["X-OpenViking-Context"]
    assert (await client.post("/v1/chat/completions", headers=normal, json=body)).status_code == 200
    assert (
        await client.post("/v1/chat/completions", headers=headers, json=body)
    ).status_code == 200
    sent = [raw for path, raw, _ in seen if path.endswith("/chat/completions")]
    assert len(sent) == 2
    assert b"cached shared memory" in sent[0]
    assert b"cached shared memory" not in sent[1]
