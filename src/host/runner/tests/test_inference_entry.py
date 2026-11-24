"""CPU subprocess checks for the private inference entry contract."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from inference_probe import prepare


@pytest.fixture(scope="module")
def installation(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("inference-entry")
    source = Path(__file__).resolve().parents[4] / "python"
    prepare(workspace, source, editable=True)
    python = workspace / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return workspace, python


def invoke(installation, overrides):
    workspace, python = installation
    env = {key: value for key, value in os.environ.items() if not key.startswith("CALIPER_RUN_")}
    env.update(CALIPER_AGENT_OFFLINE="1", PYTHONPATH=str(workspace / "shadow-path"))
    env.update(overrides)
    return subprocess.run(
        [str(python), "-E", "-P", "-u", "-m", "caliper.interp._entry",
         "--model=fixture", "--dictionary=fixture"],
        cwd=workspace, env=env, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("overrides", [{}, {"CALIPER_RUN_PROMPT": ""}])
def test_missing_prompt_refuses_before_hosted_entry(installation, overrides):
    result = invoke(installation, overrides)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "the native runner did not supply a prompt" in result.stderr


def test_fork_environment_preserves_parameter_precedence(installation):
    prompt = "original prompt"
    params = {"prompt": "fork parameter override"}
    interventions = [{"point": "layers.0.mlp", "op": "ablate"}]
    result = invoke(installation, {
        "CALIPER_RUN_PROMPT": prompt,
        "CALIPER_RUN_PARAMS": json.dumps(params),
        "CALIPER_RUN_PARENT": "parent-run",
        "CALIPER_RUN_RESUME": "checkpoint-3.json",
        "CALIPER_RUN_INTERVENTIONS": json.dumps(interventions),
    })
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["prompt"] == prompt
    assert observed["params"] == params
    assert observed["parent"] == "parent-run"
    assert observed["resume"] == "checkpoint-3.json"
    assert observed["interventions"] == interventions
    assert observed["prompt_removed"] is True
    for name in ("original_argv", "argv"):
        assert prompt not in json.dumps(observed[name])
        assert "--prompt" not in json.dumps(observed[name])
