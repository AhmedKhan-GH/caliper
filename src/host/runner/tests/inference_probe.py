"""Bounded native launcher fixture with no model or network dependencies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import time
import venv


def prepare(workspace, source, editable):
    """Install the real entry helper and a harmless hosted stand-in."""
    workspace = Path(workspace)
    environment = workspace / ".venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    site = Path(sysconfig.get_path("purelib", vars={"base": str(environment),
                                                  "platbase": str(environment)}))
    site.mkdir(parents=True, exist_ok=True)
    root = workspace / "installed-source" if editable else site
    package = root / "caliper" / "interp"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    shutil.copyfile(Path(source) / "caliper" / "interp" / "_entry.py", package / "_entry.py")
    shutil.copyfile(__file__, package / "_hosted.py")
    if editable:
        (site / "caliper-fixture.pth").write_text(str(root) + "\n")
    for location in (workspace, workspace / "shadow-path"):
        shadow = location / "caliper"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text('raise RuntimeError("shadow package executed")\n')
    # Ordinary scripts must retain their workspace and PYTHONPATH import behavior.
    (workspace / "shadow-path" / "script_dependency.py").write_text('VALUE = "script path retained"\n')
    (workspace / "script_probe.py").write_text(
        'import json, os, sys\nimport script_dependency\n'
        'print(json.dumps({"argv": sys.argv[1:], "value": script_dependency.VALUE, '
        '"prompt_removed": "CALIPER_RUN_PROMPT" not in os.environ}))\n')


def main(argv=None):
    """Observe the entry contract without model imports or an inference run."""
    parser = argparse.ArgumentParser()
    for name in ("model", "dictionary", "prompt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--replication-profile")
    parser.add_argument("--private-output-directory")
    values = vars(parser.parse_args(argv))
    # The fixture records transport identity without prompt text in output.log.
    values["prompt_sha256"] = hashlib.sha256(values.pop("prompt").encode("utf-8")).hexdigest()
    params = json.loads(os.environ.get("CALIPER_RUN_PARAMS", "{}"))
    if "prompt" in params:
        params["prompt_sha256"] = hashlib.sha256(params.pop("prompt").encode("utf-8")).hexdigest()
    values.update(original_argv=sys.orig_argv, argv=sys.argv,
                  prompt_removed="CALIPER_RUN_PROMPT" not in os.environ,
                  params=params,
                  parent=os.environ.get("CALIPER_RUN_PARENT"),
                  resume=os.environ.get("CALIPER_RUN_RESUME"),
                  interventions=json.loads(os.environ.get("CALIPER_RUN_INTERVENTIONS", "[]")),
                  executable=sys.executable)
    print(json.dumps(values), flush=True)
    # A kept-checkpoint fork needs a live parent. The native test owns this file.
    release = os.environ.get("CALIPER_TEST_INFERENCE_HOLD")
    deadline = time.monotonic() + 60
    while release and Path(release).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("the native test did not release its inference fixture")
        time.sleep(0.01)


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2], sys.argv[3] == "editable")
