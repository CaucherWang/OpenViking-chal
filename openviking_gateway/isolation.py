"""Request-local peer identities and strict benchmark memory isolation."""

import copy
from urllib.parse import urlencode

from fastapi import HTTPException

from .client import VikingError
from .storage import digest

ACTOR_HEADER = "x-openviking-actor-peer"
ISOLATION_HEADER = "x-openviking-memory-isolation"


def response_key(credential, response_id):
    key = credential["id"]
    if credential.get("peer_id"):
        key += "\0peer\0" + credential["peer_id"]
        key += "\0" + (credential.get("memory_isolation") or "")
    return digest(key + response_id)


def peer_identity(headers):
    peer = headers.get(ACTOR_HEADER)
    mode = headers.get(ISOLATION_HEADER)
    if mode not in (None, "trial"):
        raise HTTPException(400, "Unsupported OpenViking memory isolation mode")
    if peer is not None:
        # Keep this standalone gateway independent of server/service imports.
        from openviking.core.peer_id import normalize_peer_id

        try:
            peer = normalize_peer_id(peer)
        except ValueError:
            raise HTTPException(400, "Invalid OpenViking actor peer") from None
        if not peer or peer == "__self":
            raise HTTPException(400, "Invalid OpenViking actor peer")
    if mode and not peer:
        raise HTTPException(400, "Trial memory isolation requires an actor peer")
    return peer, mode == "trial"


def scoped_viking(viking, identity, sessions=()):
    """Copy the HTTP adapter, never mutate a client shared by concurrent Trials."""
    if not identity.get("peer_id"):
        return viking
    scoped = copy.copy(viking)
    scoped.peer_id = identity["peer_id"]
    scoped.isolated_peer = identity.get("memory_isolation") == "trial"
    scoped.peer_root = (
        f"viking://user/{identity['user_id']}/peers/{identity['peer_id']}"
    )
    if identity.get("dev_memory_namespace"):
        scoped.dev_memory_user = identity["user_id"]
    scoped.peer_sessions = tuple(
        f"viking://user/{identity['user_id']}/sessions/{session}" for session in sessions
    )
    return scoped


async def require_empty_parent(viking, credential):
    """Check the entire parent memory tree, including hidden files and pagination."""
    root = f"viking://user/{credential['user_id']}/memories"
    pending = [root]
    visited = set()
    while pending:
        uri = pending.pop()
        if uri in visited or len(visited) >= 10000:
            raise VikingError("parent_memory_check_failed")
        visited.add(uri)
        offset = 0
        while True:
            path = "/api/v1/fs/ls?" + urlencode(
                {
                    "uri": uri, "output": "original", "show_all_hidden": "true",
                    "node_limit": 1000, "offset": offset,
                }
            )
            try:
                entries = await viking.request(
                    "GET", path, credential["openviking_key"], timeout=5
                )
            except VikingError as error:
                if error.status == 404:
                    break
                raise
            if not isinstance(entries, list):
                raise VikingError("parent_memory_check_failed")
            for entry in entries:
                name = entry.get("name") if isinstance(entry, dict) else None
                if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
                    raise VikingError("parent_memory_check_failed")
                child = uri + "/" + name
                if entry.get("isDir"):
                    pending.append(child)
                else:
                    content = await viking.read_content(credential["openviking_key"], child)
                    if content is not None and (
                        not isinstance(content, str) or content.strip()
                    ):
                        raise HTTPException(
                            409,
                            {
                                "reason": "parent_memory_not_empty",
                                "detail": "Trial isolation requires an empty parent user memory. "
                                "Use a clean benchmark user; existing memories were preserved.",
                            },
                        )
            if len(entries) < 1000:
                break
            offset += len(entries)
            if offset > 10000:
                raise VikingError("parent_memory_check_failed")
