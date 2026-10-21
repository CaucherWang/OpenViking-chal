---
name: add-rules
description: Compile durable constraints into OpenViking Rules with rules/compile. Use when the user asks to add rules or import a policy. Do not use for preferences, facts, or rule search.
---

# Add Rules

## Goal

Use `POST /api/v1/rules/compile` to compile user-authorized durable constraints into
OpenViking Rules, then verify the task result and saved Rules.

## Workflow

1. Identify the user's rule text or policy sources and the intended OpenViking connection.
   Use the same instance and identity for upload, compilation, task polling, and verification.
2. Build a request with top-level `instruction` and `sources`; at least one must be non-empty.
   Preserve the source's obligations, applicability conditions, exceptions, thresholds,
   and identifiers without inventing policy.
3. Prepare each source using the appropriate form:
   - Existing Resource: `{"resource_uri": "viking://resources/team-policy"}`.
   - Remote text source: `{"path": "https://example.org/policy.md"}`.
   - Local file or directory: use the Python SDK to upload it, or upload through
     `POST /api/v1/resources/temp_upload` and submit the returned `temp_file_id`.
     Do not send a client-local filesystem path directly to the compile HTTP endpoint.
4. Submit `POST /api/v1/rules/compile`, or call
   `client.rule_compile(instruction, options={"sources": sources})` with the Python SDK.
   SDK sources may also be local path strings, URLs, or Resource URI strings.
   Save the returned `task_id`; HTTP 202 means accepted, not completed.
5. Poll `GET /api/v1/tasks/{task_id}` or `client.get_task(task_id)` until completed,
   failed, or cancelled. Query the existing task rather than resubmitting the request.
6. Inspect `result.operations`, `written_uris`, and `write_result`. Distinguish saved
   Rules from NOOP, UNCERTAIN, and REJECT results; completed does not guarantee new Rules.
7. Read representative saved Rules and their Scenarios using `client.read_raw(uri)` or
   `GET /api/v1/content/read?uri=...&raw=true`. Check the policy text and evidence,
   and verify `rules/<scenario_id>/abstract.md` holds Scenario metadata while Rule
   files sit directly beside it as `<atomic_id>.md`.
   Report the task ID, actual changes, representative URIs, and unresolved decisions.

## Boundaries

- Compile durable constraints only; do not promote preferences, facts, or one-task advice.
- Prefer faithful, non-empty `applies_when` conditions supported by the source.
  Empty `applies_when` means applicable in every situation except `exception`, not unknown.
- Request lifecycle changes only when the user explicitly authorizes them.
  Use PAUSE to suspend and ARCHIVE to mark for retention-based cleanup.
  Paused Rules can be updated and reactivated; archived Rules are not compiler candidates.
- Use client connection configuration, not service-side model credentials. Do not switch
  instances or expose credentials to resolve a failed request.
- The API accepts at most 16 sources and 64000 instruction characters; source collection
  reads at most 128 files and 128000 characters. Split oversized material into self-contained
  policies with their conditions and exceptions rather than silently omitting content.
- Rule compilation requires readable text. Do not submit image, audio, or video sources.
- If a request fails or its status is unknown, check the task and saved files before retrying.
  Previously imported Resources are not rolled back when later Rule compilation fails.
- Do not treat an immediate empty search result as proof that writing failed; semantic
  indexing may still be pending.
