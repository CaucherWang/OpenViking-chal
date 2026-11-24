# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Scope Gateway memory access to the active peer when the profile requires it."""

from .client import VikingError

PEER_MEMORY_TOOLS = frozenset({"find", "read", "ls", "list", "write"})


def memory_root(user, peer):
    return f"viking://user/{user}/peers/{peer}/memories"


def contains(root, uri):
    return (
        isinstance(uri, str)
        and (uri.rstrip("/") == root or uri.startswith(root + "/"))
        and not any(char in uri for char in "%\\?#")
        and not {".", ".."}.intersection(uri.split("/"))
    )


def tool_arguments(root, name, args):
    """Only tools with a bound URI are available under a peer-only memory profile."""
    if name not in PEER_MEMORY_TOOLS:
        raise VikingError("tool_outside_peer_memory", 403)
    for field in ("uri", "uris", "target_uri"):
        if field not in args:
            continue
        value = args[field]
        values = value if isinstance(value, list) else [value]
        if not values or any(not contains(root, uri) for uri in values):
            raise VikingError("tool_outside_peer_memory", 403)
    if name == "find":
        return {**args, "target_uri": args.get("target_uri") or root, "context_type": ["memory"]}
    if name in {"ls", "list"}:
        return {**args, "uri": args.get("uri") or root}
    if not (args.get("uri") or (name == "read" and args.get("uris"))):
        raise VikingError("tool_outside_peer_memory", 403)
    return args
