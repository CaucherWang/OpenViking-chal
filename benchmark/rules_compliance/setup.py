"""Start isolated services, ingest one policy file, and compile the Rules arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from experiment import DEFAULT_OUTPUT, POLICY, PORTS, ROOT, RULE_IDS, write_json

sys.path.insert(0, str(ROOT / "sdk/python"))

from openviking_sdk import SyncHTTPClient

RULE_PATTERN = re.compile(r"\bR(?:0[1-9]|[1-3][0-9]|40)\b")


def api(client: httpx.Client, method: str, path: str, trace: Path, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    try:
        body = response.json()
    except ValueError:
        body = {"text": response.text}
    with trace.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "method": method,
                    "path": path,
                    "request": kwargs.get("json") or kwargs.get("params"),
                    "status_code": response.status_code,
                    "response": body,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    response.raise_for_status()
    if isinstance(body, dict) and body.get("status") == "error":
        raise RuntimeError(str(body))
    return body.get("result", body) if isinstance(body, dict) else body


def start(output: Path, arms: list[str]) -> None:
    local = json.loads((ROOT / ".openviking/ov.conf").read_text(encoding="utf-8"))
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    base = subprocess.check_output(["git", "rev-parse", "main"], cwd=ROOT, text=True).strip()
    main_checkout = output / "main_checkout"
    if "main" in arms:
        if not (main_checkout / ".git").exists():
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(main_checkout), base],
                cwd=ROOT,
                check=True,
            )
        elif (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=main_checkout, text=True
            ).strip()
            != base
        ):
            raise RuntimeError("Existing main_checkout does not match the frozen main revision")

    services_path = runtime / "services.json"
    services = json.loads(services_path.read_text()) if services_path.exists() else {}
    for arm in arms:
        port = PORTS[arm]
        with socket.socket() as probe:
            occupied = probe.connect_ex(("127.0.0.1", port)) == 0
        if occupied:
            if arm not in services:
                raise RuntimeError(f"Port {port} is occupied by an unrelated service")
            continue

        config = {key: local[key] for key in ("vlm", "embedding", "rerank") if key in local}
        config.update(
            storage={"workspace": str(runtime / arm / "storage")},
            server={"host": "127.0.0.1", "port": port, "auth_mode": "dev"},
            log={"level": "INFO", "output": "stderr"},
            bot={"enabled": False},
            gateway={"enabled": False},
        )
        if arm == "rules":
            config["rules"] = dict(local.get("rules", {}))
            config["rules"].update(
                compiler_context_window=32768,
                archived_cleanup_enabled=False,
            )
        config_path = runtime / f"{arm}.conf"
        write_json(config_path, config)
        config_path.chmod(0o600)

        cwd = ROOT if arm == "rules" else main_checkout
        if arm == "main":
            for source in (ROOT / "openviking").rglob("*.so"):
                target = cwd / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.symlink_to(source)
        log = (runtime / f"{arm}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "openviking.server.bootstrap",
                "--config",
                str(config_path),
                "--port",
                str(port),
            ],
            cwd=cwd,
            env={
                **os.environ,
                "OPENVIKING_CONFIG_FILE": str(config_path),
                "PYTHONPATH": str(cwd),
            },
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        services[arm] = {"pid": proc.pid, "port": port, "cwd": str(cwd), "base_sha": base}
    write_json(services_path, services)
    print(json.dumps(services, indent=2))


def wait_ready(port: int, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"OpenViking on port {port} did not become ready")


def wait_task(
    client: httpx.Client,
    task_id: str,
    trace: Path,
    state_path: Path,
    *,
    timeout: int = 7200,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = api(
            client,
            "GET",
            f"/api/v1/tasks/{task_id}",
            trace,
            params={"include_events": "true"},
        )
        write_json(state_path, state)
        if state["status"] in {"completed", "failed", "cancelled"}:
            if state["status"] != "completed":
                raise RuntimeError(
                    f"Task {task_id} ended as {state['status']}: {state.get('error')}"
                )
            return state
        time.sleep(10)
    raise TimeoutError(f"Task {task_id} did not finish within {timeout} seconds")


def write_inventory(
    client: httpx.Client,
    uri: str,
    trace: Path,
    destination: Path,
) -> dict[str, Any]:
    try:
        result = api(
            client,
            "POST",
            "/api/v1/search/grep",
            trace,
            json={
                "uri": uri,
                "pattern": RULE_PATTERN.pattern,
                "node_limit": 1000,
                "level_limit": 20,
                "before_context": 1,
                "after_context": 1,
            },
        )
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 404:
            raise
        result = {
            "matches": [],
            "count": 0,
            "match_count": 0,
            "files_scanned": 0,
            "missing_uri": True,
        }
    retained = sorted(set(RULE_PATTERN.findall(json.dumps(result, ensure_ascii=False))))
    inventory = {
        "uri": uri,
        "rule_ids": retained,
        "missing_rule_ids": sorted(set(RULE_IDS) - set(retained)),
        "grep_result": result,
    }
    write_json(destination, inventory)
    return inventory


def offline(output: Path, arms: list[str]) -> None:
    if not POLICY.is_file():
        raise FileNotFoundError(POLICY)
    for arm in arms:
        wait_ready(PORTS[arm])

    runtime = output / "runtime"
    policy_sha256 = hashlib.sha256(POLICY.read_bytes()).hexdigest()
    offline_stage: dict[str, Any] = {
        "status": "completed",
        "policy_sha256": policy_sha256,
        "arms": arms,
    }

    if "main" in arms:
        policy_text = POLICY.read_text(encoding="utf-8")
        main_trace = runtime / "main.offline.jsonl"
        main_client = httpx.Client(base_url=f"http://127.0.0.1:{PORTS['main']}", timeout=1200)
        commit_state_path = runtime / "main_commit_state.json"
        if (
            commit_state_path.exists()
            and json.loads(commit_state_path.read_text())["status"] == "completed"
        ):
            session = json.loads((runtime / "main_session.json").read_text())
            commit_task = json.loads((runtime / "main_commit_task.json").read_text())
            commit_state = json.loads(commit_state_path.read_text())
        else:
            session = api(
                main_client,
                "POST",
                "/api/v1/sessions",
                main_trace,
                json={
                    "session_id": f"rules-memory-{policy_sha256[:16]}",
                    "auto_commit_policy": None,
                },
            )
            write_json(runtime / "main_session.json", session)
            message = api(
                main_client,
                "POST",
                f"/api/v1/sessions/{session['session_id']}/messages",
                main_trace,
                json={"role": "user", "peer_id": "user", "content": policy_text},
            )
            write_json(runtime / "main_message.json", message)
            commit_task = api(
                main_client,
                "POST",
                f"/api/v1/sessions/{session['session_id']}/commit",
                main_trace,
                json={},
            )
            write_json(runtime / "main_commit_task.json", commit_task)
            commit_state = wait_task(
                main_client,
                commit_task["task_id"],
                main_trace,
                commit_state_path,
            )
        main_inventory = write_inventory(
            main_client,
            "viking://~/peers/user/memories",
            main_trace,
            runtime / "main_memory_inventory.json",
        )
        main_client.close()
        offline_stage["main"] = {
            "pipeline": "session user message + session.commit",
            "session_id": session["session_id"],
            "commit_task_id": commit_task["task_id"],
            "commit_result": commit_state.get("result"),
            "retained_rule_ids": main_inventory["rule_ids"],
            "missing_rule_ids": main_inventory["missing_rule_ids"],
        }

    if "rules" in arms:
        rules_trace = runtime / "rules.offline.jsonl"
        rules_url = f"http://127.0.0.1:{PORTS['rules']}"
        rules_client = httpx.Client(
            base_url=rules_url,
            timeout=1200,
        )
        rules_sdk = SyncHTTPClient(url=rules_url, timeout=1200)
        rules_sdk.initialize()
        try:
            compile_task = rules_sdk.rule_compile(
                (
                    "将文档中的 R01 到 R40 编译为可检索的独立强约束。每条源规则必须独立"
                    "编译，不得合并、遗漏或补充文档外义务；在 Rule 的 name、description 或"
                    "content 中原样保留对应 Rxx 标识，并保留触发条件、核查方法、合规要求和"
                    "违规证据。不要归档、删除或停用规则。"
                ),
                options={"sources": [str(POLICY)]},
            )
        finally:
            rules_sdk.close()
        write_json(runtime / "compile_task.json", compile_task)
        compile_state = wait_task(
            rules_client,
            compile_task["task_id"],
            rules_trace,
            runtime / "compile_state.json",
        )
        compile_result = compile_state.get("result") or {}
        rules_root = (
            compile_result.get("rules_root")
            or compile_state.get("resource_id")
            or "viking://~/memories/rules"
        )
        rules_inventory = write_inventory(
            rules_client,
            rules_root.rstrip("/"),
            rules_trace,
            runtime / "rules_inventory.json",
        )
        rules_client.close()
        offline_stage["rules"] = {
            "pipeline": "rule_compile(local source)",
            "compile_task_id": compile_task["task_id"],
            "compile_result": compile_result,
            "retained_rule_ids": rules_inventory["rule_ids"],
            "missing_rule_ids": rules_inventory["missing_rule_ids"],
        }

    write_json(runtime / "offline_stage.json", offline_stage)
    print("Offline preparation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["start", "offline"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", nargs="+", choices=PORTS, default=list(PORTS))
    arguments = parser.parse_args()
    {"start": start, "offline": offline}[arguments.step](
        arguments.output.resolve(),
        list(dict.fromkeys(arguments.arms)),
    )
