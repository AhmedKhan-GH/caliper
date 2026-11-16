"""The provider conformance suite (caliper-agent-conformance) in its default modes.

The suite runs every task of conformance/tasks.json on its fixture workspace,
through the real session, context bundle and ACP client. Here it runs against
the scripted model with each task's passing answer, which must pass every
task, and with each failing answer, which must fail every task, so that the
suite's checks are tested themselves; and against the fake OpenAI-compatible
server, whose model gives the passing answers, after the probe. A hosted
provider is refused without --real. ctest passes the runner's path in
CALIPER_AGENT_CONFORMANCE; without it the tests are skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import AGENT_DIR, FAKE_KEY
from fake_openai import FakeOpenAI, png_colour

RUNNER = os.environ.get("CALIPER_AGENT_CONFORMANCE", "")
TASKS = json.loads((AGENT_DIR / "conformance" / "tasks.json").read_text())["tasks"]

pytestmark = pytest.mark.skipif(not RUNNER or not Path(RUNNER).is_file(),
                                reason="CALIPER_AGENT_CONFORMANCE does not name a built caliper-agent-conformance")


def run_suite(tmp_path: Path, *args: str, env_extra: dict | None = None) -> tuple[int, dict | None, str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CALIPER_MODEL", "OPENAI_"))}
    env.update(env_extra or {})
    run = subprocess.run([RUNNER, "--python", sys.executable, "--out", str(tmp_path / "out"),
                          "--work", str(tmp_path / "work"), *args], capture_output=True, text=True, env=env, timeout=600)
    files = sorted((tmp_path / "out").glob("conformance-*.json")) if (tmp_path / "out").is_dir() else []
    doc = json.loads(files[0].read_text()) if files else None
    md = files[0].with_suffix(".md").read_text() if files else ""
    return run.returncode, doc, md, run.stderr


def test_every_passing_answer_passes(tmp_path):
    code, doc, md, err = run_suite(tmp_path, "--scripted", "pass")
    assert code == 0, err
    assert [t["outcome"] for t in doc["tasks"]] == ["pass"] * len(TASKS), err
    assert doc["verdict"] == "passes" and doc["passed"] == doc["applicable"] == len(TASKS)
    assert doc["mode"] == "scripted-pass" and doc["date"] and doc["sidecar"].startswith("caliper-agent ")
    by_id = {t["id"]: t for t in doc["tasks"]}
    assert by_id[10]["runs"][0]["handoff"]["kind"] == "loader"
    assert by_id[1]["runs"][0]["tool_calls"] == ["panel_patch"]
    assert "| 1 | Rename a panel | pass | 1 of 1 |" in md and "# Conformance run" in md
    assert FAKE_KEY not in json.dumps(doc) and FAKE_KEY not in md


def test_every_failing_answer_fails(tmp_path):
    code, doc, md, err = run_suite(tmp_path, "--scripted", "fail")
    assert code == 0, err
    assert [t["outcome"] for t in doc["tasks"]] == ["fail"] * len(TASKS)
    assert doc["verdict"] == "fails" and doc["passed"] == 0
    reasons = {t["id"]: t["runs"][0]["reason"] for t in doc["tasks"]}
    assert "Training curve" in reasons[1] and "rdbu" in reasons[2] and "x_label" in reasons[3]
    assert "52" in reasons[5] and "loss" in reasons[7] and "reference" in reasons[8]
    assert "handed" in reasons[10] and "panel.curves" in reasons[12]


def test_the_vision_task_needs_vision(tmp_path):
    code, doc, md, err = run_suite(tmp_path, "--scripted", "pass", "--assume-vision", "no", "--tasks", "1,11")
    assert code == 0, err
    assert {t["id"]: t["outcome"] for t in doc["tasks"]} == {1: "pass", 11: "not applicable"}
    assert doc["applicable"] == 1 and doc["verdict"] == "passes"


def _responder(body: dict) -> dict:
    """The fake model: the passing answer of the task whose request is in the prompt."""
    messages = body["messages"]
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    content = messages[last_user]["content"]
    parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
    text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
    images = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    task = next(t for t in TASKS if t["request"] in text)
    colour = png_colour(images[0]) if images else "unseen"
    answer = json.loads(json.dumps(task["pass"]).replace("$colour", colour or "unseen"))
    if any(m["role"] == "tool" for m in messages[last_user:]) or not answer["calls"]:
        return {"text": answer["text"] or "Done."}
    return {"tool_calls": [{"id": f"call_{n}", "name": c["name"], "arguments": json.dumps(c["input"])}
                           for n, c in enumerate(answer["calls"])]}


def test_the_suite_against_the_fake_server(tmp_path):
    server = FakeOpenAI(models=[{"id": "fake-model", "context_length": 32768}], responder=_responder)
    try:
        code, doc, md, err = run_suite(tmp_path, "--provider", "openai-compatible", "--base-url", server.base_url,
                                       "--model", "fake-model", "--runs", "1", "--server", "fake server")
    finally:
        server.close()
    assert code == 0, err
    assert [t["outcome"] for t in doc["tasks"]] == ["pass"] * len(TASKS), err
    assert doc["mode"] == "offline" and doc["provider"] == "openai-compatible" and doc["model"] == "fake-model"
    assert doc["base_url_host"] == "127.0.0.1" and doc["server"] == "fake server"
    caps = doc["capabilities"]
    assert caps["tools"]["source"] == "probe" and caps["vision"]["value"] is True
    assert caps["context_length"]["value"] == 32768
    # The probe's two requests, then each task's requests, every one on loopback.
    assert any("record_number" in json.dumps(r.get("tools", [])) for r in server.requests)


def test_a_model_without_tools_cannot_drive_caliper(tmp_path):
    server = FakeOpenAI(refuse_tools=True, models=[{"id": "fake-model"}])
    try:
        code, doc, md, err = run_suite(tmp_path, "--provider", "openai-compatible", "--base-url", server.base_url,
                                       "--model", "fake-model", "--runs", "1")
    finally:
        server.close()
    assert code == 0, err
    assert doc["verdict"] == "cannot drive Caliper" and {t["outcome"] for t in doc["tasks"]} == {"not run"}


def test_a_hosted_provider_needs_real(tmp_path):
    for args in (["--preset", "openai", "--model", "m"], ["--provider", "claude"],
                 ["--provider", "openai-compatible", "--base-url", "https://models.example.com/v1", "--model", "m"]):
        code, doc, md, err = run_suite(tmp_path, *args)
        assert code == 2 and "--real" in err and doc is None, args
    code, _, _, err = run_suite(tmp_path, "--scripted", "maybe")
    assert code == 2
