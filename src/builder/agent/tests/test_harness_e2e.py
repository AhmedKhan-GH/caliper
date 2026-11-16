"""End to end through the real C++ ACP client.

`caliper-acp-harness` (the acp module's headless tool, which drives
`caliper_builder::acp::AcpClient`) runs the sidecar as its agent, with a fake
key passed by `--env` and the scripted Claude API transport, against a fake
Caliper API. The test reads the harness's transcript. ctest passes the
harness's path in CALIPER_ACP_HARNESS; without it the test is skipped.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (AGENT_DIR, COMMAND_TABLE, FAKE_KEY, FAKE_TOKEN, FakeCaliper, message, read_requests, text,
                      thinking, tool_use, write_script)

HARNESS = os.environ.get("CALIPER_ACP_HARNESS", "")


@pytest.mark.skipif(not HARNESS or not Path(HARNESS).is_file(),
                    reason="CALIPER_ACP_HARNESS does not name a built caliper-acp-harness")
def test_a_turn_through_the_cpp_client(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake = FakeCaliper(tmp_path)
    steps = [
        message(thinking("Start with the catalog."), text("I will look at the catalog first."),
                tool_use("toolu_list", "catalog_list", {"kind": "table"}), stop="tool_use",
                usage={"input_tokens": 30, "output_tokens": 10, "cache_creation_input_tokens": 9000}),
        message(text("Adding the panel and saving."),
                tool_use("toolu_panel", "panel_create", {"id": "loss", "primitive": "series",
                                                         "bind": {"y": "metrics.loss"}}),
                tool_use("toolu_save", "applet_save", {"id": "demo"}), stop="tool_use",
                usage={"input_tokens": 20, "output_tokens": 15, "cache_read_input_tokens": 9000}),
        message(tool_use("toolu_capture", "capture_frame", {}), stop="tool_use"),
        message(text("The loss panel is in the spec and the applet is saved.")),
    ]
    # The anthropic route: the real Anthropic model behind the stubbed transport.
    script, record = write_script(tmp_path, steps, route="anthropic")
    agent = " ".join(shlex.quote(a) for a in [sys.executable, "-P", "-m", "caliper_agent",
                                               "--command-table", str(COMMAND_TABLE), "--api-file", str(fake.api_file)])
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env["PYTHONPATH"] = str(AGENT_DIR)
    try:
        proc = subprocess.run(
            [HARNESS, "--agent", agent, "--cwd", str(workspace),
             "--env", f"ANTHROPIC_API_KEY={FAKE_KEY}", "--env", f"CALIPER_AGENT_TEST_SCRIPT={script}",
             "--env", "PYTHONPATH", "--prompt", "Plot the loss and save the applet.",
             "--permission", "allow", "--timeout", "90"],
            capture_output=True, text=True, env=env, timeout=120)
    finally:
        fake.close()

    assert FAKE_KEY not in proc.stdout and FAKE_KEY not in proc.stderr
    assert FAKE_TOKEN not in proc.stdout and FAKE_TOKEN not in proc.stderr
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    by_type: dict[str, list[dict]] = {}
    for line in lines:
        by_type.setdefault(line["type"], []).append(line)

    assert by_type["harness"][0]["env_names"] == ["ANTHROPIC_API_KEY", "CALIPER_AGENT_TEST_SCRIPT", "PYTHONPATH"]
    init = by_type["initialized"][0]
    assert init["protocol_version"] == 1 and init["agent"]["name"] == "caliper-agent"
    assert by_type["session"][0]["session_id"].startswith("byok-")
    assert any(t["text"] == "Start with the catalog." for t in by_type["agent_thought_chunk"])
    said = "".join(m["text"] for m in by_type["agent_message_chunk"])
    assert said.startswith("I will look at the catalog first.")
    assert said.endswith("The loss panel is in the spec and the applet is saved.")

    started = [t["tool_call"]["id"] for t in by_type["tool_call"]]
    assert started == ["toolu_list", "toolu_panel", "toolu_save", "toolu_capture"]
    final = {}
    for update in by_type["tool_call_update"]:
        call = update["tool_call"]
        if call.get("status") in ("completed", "failed"):
            final[call["id"]] = call
    assert {k: v["status"] for k, v in final.items()} == {
        "toolu_list": "completed", "toolu_panel": "completed", "toolu_save": "completed",
        "toolu_capture": "completed"}
    panel_diff = [c for c in final["toolu_panel"]["content"] if c["type"] == "diff"]
    assert panel_diff and panel_diff[0]["path"] == str(workspace / "applets/demo/demo.spec.toml")
    assert 'id = "loss"' in panel_diff[0]["new_text"]
    assert any(c.get("content_type") == "image" for c in final["toolu_capture"]["content"])

    asked = [p["tool_call"]["id"] for p in by_type["permission_request"]]
    assert asked == ["toolu_panel", "toolu_save"]
    # The panel's permission request shows the change before it is made.
    preview = [c for c in by_type["permission_request"][0]["tool_call"]["content"] if c["type"] == "diff"]
    assert preview and preview[0]["new_text"] == panel_diff[0]["new_text"]
    assert all(a["option_kind"] == "allow_once" for a in by_type["permission_answer"])

    result = by_type["result"][0]
    assert result["ok"] is True and result["stop_reason"] == "end_turn"

    commands = [(c["path"], c["origin"], bool((c["args"] or {}).get("dry_run"))) for c in fake.calls
                if c["path"].startswith("/v0/commands/")
                and c["path"] not in ("/v0/commands/spec.get", "/v0/commands/review.status")]
    assert commands == [("/v0/commands/catalog.list", "agent", False), ("/v0/commands/panel.create", "agent", True),
                        ("/v0/commands/panel.create", "agent", False), ("/v0/commands/applet.save", "agent", False)]
    assert fake.calls[-1]["path"] == "/v0/capture.png"

    requests = read_requests(record)
    assert len(requests) == 4 and {r["host"] for r in requests} == {"api.anthropic.com"}
    assert all(r["has_credential"] for r in requests)
    assert FAKE_KEY not in record.read_text()
