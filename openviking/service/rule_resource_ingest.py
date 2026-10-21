# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Rule-only adapter for importing compile sources through ResourceService."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openviking.core.path_variables import resolve_path_variables
from openviking.core.uri_validation import validate_content_target_uri
from openviking.resource.processing_mode import DEFAULT_PROCESSING_MODE, ProcessingMode
from openviking.server.identity import RequestContext
from openviking.server.local_input_guard import require_remote_resource_source
from openviking.server.temp_upload_store import TempUploadStore
from openviking.storage.acl import AclSpec
from openviking.telemetry import TelemetryRequest
from openviking_cli.exceptions import InvalidArgumentError


class AddResourceRequest(BaseModel):
    """Resource source accepted inside a rule compile request."""

    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    temp_file_id: Optional[str] = None
    add_type: Optional[str] = None
    to: Optional[str] = None
    parent: Optional[str] = None
    create_parent: bool = False
    reason: str = ""
    instruction: str = ""
    wait: bool = False
    strict: bool = False
    internal_task: bool = False
    source_name: Optional[str] = None
    ignore_dirs: Optional[str] = None
    include: Optional[str] = None
    exclude: Optional[str] = None
    directly_upload_media: bool = True
    preserve_structure: Optional[bool] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    telemetry: TelemetryRequest = False
    watch_interval: float = 0
    is_active: bool = True
    processing_mode: ProcessingMode = DEFAULT_PROCESSING_MODE
    tags: Optional[list[str]] = None
    tag_mode: str = "replace"
    acl: AclSpec | None = None

    @model_validator(mode="after")
    def check_path_or_temp_file_id(self):
        if not self.path and not self.temp_file_id:
            raise ValueError("Either 'path' or 'temp_file_id' must be provided")
        return self

    @model_validator(mode="after")
    def check_add_type(self):
        if self.add_type is not None:
            self.add_type = self.add_type.strip() or None
        if self.add_type and self.temp_file_id:
            raise ValueError("'add_type' cannot be combined with 'temp_file_id'")
        if self.add_type and not self.path:
            raise ValueError("'add_type' requires 'path'")
        if self.add_type and self.parent:
            raise ValueError("'add_type' cannot be combined with 'parent'")
        if self.add_type and not self.to:
            raise ValueError("'add_type' requires an exact 'to' target")
        return self

    @model_validator(mode="after")
    def check_paused_watch(self):
        has_target = bool((self.to or "").strip() or (self.parent or "").strip())
        if self.is_active is False and (self.watch_interval <= 0 or not has_target):
            raise ValueError(
                "is_active=false requires watch_interval > 0 and either 'to' or 'parent'"
            )
        return self


async def ingest_add_resource_request(
    request: AddResourceRequest,
    ctx: RequestContext,
    *,
    resources: Any,
    config: Any,
    force_wait: bool,
    force_internal_task: bool = True,
) -> dict[str, Any]:
    """Import one rule source without changing the existing Resource API path."""
    to_uri = resolve_path_variables(request.to).strip() if request.to else ""
    if to_uri:
        to_uri = validate_content_target_uri(to_uri, ctx, kind="resource", field_name="to")
    parent_uri = resolve_path_variables(request.parent).strip() if request.parent else ""
    if parent_uri:
        parent_uri = validate_content_target_uri(
            parent_uri,
            ctx,
            kind="resource",
            field_name="parent",
        )

    path = request.path
    allow_local_path_resolution = False
    original_filename = None
    resolved = None
    shared_source_ref = None
    if request.temp_file_id:
        if request.watch_interval > 0:
            raise InvalidArgumentError(
                "watch_interval > 0 is not supported for uploaded content: an "
                "upload is a static snapshot, so the watch would re-process "
                "stale content forever. Watch a URL / "
                "sitemap / RSS source instead, or re-add the resource when the "
                "source changes."
            )
        server_config = config
        if not hasattr(server_config, "temp_upload"):
            from openviking.server.dependencies import get_server_config

            server_config = get_server_config()
        if server_config is None:
            raise RuntimeError("Server configuration is unavailable for temporary upload")
        store = TempUploadStore.build(server_config)
        shared_source_ref = await store.resolve_shared_reference(request.temp_file_id, ctx)
        if shared_source_ref is not None:
            path = shared_source_ref.original_filename or request.temp_file_id
            original_filename = shared_source_ref.original_filename or None
            allow_local_path_resolution = True
        else:
            resolved = await store.resolve_for_consume(request.temp_file_id, ctx)
            path = resolved.local_path
            original_filename = resolved.original_filename
            allow_local_path_resolution = True
    elif path is not None:
        path = require_remote_resource_source(
            path,
            declared_connector_add_type=request.add_type,
        )
    if path is None:
        raise InvalidArgumentError("Either 'path' or 'temp_file_id' must be provided.")

    source_name = request.source_name or original_filename
    kwargs = {
        "strict": request.strict,
        "source_name": source_name,
        "ignore_dirs": request.ignore_dirs,
        "include": request.include,
        "exclude": request.exclude,
        "directly_upload_media": request.directly_upload_media,
        "watch_interval": request.watch_interval,
        "processing_mode": request.processing_mode,
    }
    if "create_parent" in request.model_fields_set:
        kwargs["create_parent"] = request.create_parent
    if request.temp_file_id and request.watch_interval <= 0:
        kwargs["temp_file_id"] = request.temp_file_id
    if request.preserve_structure is not None:
        kwargs["preserve_structure"] = request.preserve_structure

    try:
        return await resources.add_resource(
            path=path,
            ctx=ctx,
            add_type=request.add_type,
            to=to_uri,
            parent=parent_uri,
            reason=request.reason,
            instruction=request.instruction,
            wait=force_wait,
            acl=request.acl,
            tags=request.tags,
            tag_mode=request.tag_mode,
            allow_local_path_resolution=allow_local_path_resolution,
            enforce_public_remote_targets=True,
            internal_task=force_internal_task,
            is_active=request.is_active,
            args=request.args,
            shared_source=shared_source_ref,
            **kwargs,
        )
    finally:
        if resolved:
            await resolved.cleanup()


__all__ = ["AddResourceRequest", "ingest_add_resource_request"]
