"""Bounded native launcher fixture with no model or network dependencies."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
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
    values = vars(parser.parse_args(argv))
    values.update(original_argv=sys.orig_argv, argv=sys.argv,
                  prompt_removed="CALIPER_RUN_PROMPT" not in os.environ,
                  params=json.loads(os.environ.get("CALIPER_RUN_PARAMS", "{}")),
                  parent=os.environ.get("CALIPER_RUN_PARENT"),
                  resume=os.environ.get("CALIPER_RUN_RESUME"),
                  interventions=json.loads(os.environ.get("CALIPER_RUN_INTERVENTIONS", "[]")),
                  executable=sys.executable)
    print(json.dumps(values), flush=True)


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2], sys.argv[3] == "editable")
