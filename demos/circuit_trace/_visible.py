"""Browser assets and the owned app lifetime for an interactive demo."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from _mcp import Failure
from _spec import toml_value

ASSETS = ("caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm")


class StartupFailure(Failure):
    """Retain ownership when startup fails after the visible app starts."""

    def __init__(self, session, error):
        super().__init__(str(error))
        self.session = session


def configure_assets(env):
    # The app uses WEB_DIR. The parity harness uses WASM_DIR.
    directory = env.get("CALIPER_WEB_DIR") or env.get("CALIPER_WASM_DIR")
    if directory:
        env["CALIPER_WEB_DIR"] = str(Path(directory).resolve())


def check_assets(session):
    """Check the actual viewer routes without recording their access token."""
    settings = json.loads((session.out / "data" / "api.json").read_text())
    url = urlsplit(settings["spec_url"])
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.path != "/v0/spec":
        raise Failure("The owned app did not publish a local spec URL.")
    evidence = {}
    for name in ASSETS:
        route = urlunsplit((url.scheme, url.netloc, url.path + "/" + name, url.query, ""))
        try:
            with urlopen(route, timeout=15) as response:
                body = response.read()
                status = response.status
        except OSError:
            raise Failure(f"The browser asset {name} is unavailable.") from None
        if status != 200 or not body or (name.endswith(".wasm") and body[:4] != b"\x00asm"):
            raise Failure(f"The browser asset {name} is invalid.")
        evidence[name] = {"status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    return evidence


def show_error(session, error):
    """Keep the existing graph and put the error in the instrument header."""
    text = str(error) or type(error).__name__
    (session.out / "demo-error.json").write_text(json.dumps({"error": text}, indent=2) + "\n")
    try:
        spec = deepcopy(session.mcp.body("spec.get")["json"])
        if not spec.get("applet"):
            spec.update(caliper_spec=1, applet={"id": "circuit-trace", "title": "Circuit Trace"})
        panels = [p for p in spec.get("panel", []) if p["id"] != "demo_error"]
        panels.append({"id": "demo_error", "primitive": "text", "title": "Demo Error", "options": {"text": text}})
        spec["panel"] = panels
        def replace_status(node):
            found = False
            if isinstance(node, dict):
                if node.get("panel") in ("clamp_status", "demo_error"):
                    node["panel"] = "demo_error"
                    return True
                for key in ("children", "tabs"):
                    for i, child in enumerate(node.get(key, [])):
                        if isinstance(child, str) and child in ("clamp_status", "demo_error"):
                            node[key][i] = "demo_error"
                            found = True
                        else:
                            found = replace_status(child) or found
            return found
        if not replace_status(spec.get("layout")):
            spec["layout"] = {"panel": "demo_error"}
        session.mcp.body("spec.set_text", text="\n".join(
            toml_value(key) + " = " + toml_value(value) for key, value in spec.items()) + "\n")
    except Exception as exc:
        # A broken command connection must not close the app either.
        (session.out / "demo-error-display.json").write_text(json.dumps({"error": str(exc)}) + "\n")


def hold_after_error(session, error):
    show_error(session, error)
    bridge = getattr(session, "bridge", None)
    if bridge is not None:
        try:
            bridge.keep_alive()
        except Exception as exc:
            show_error(session, exc)
    # Even a failed bridge leaves the app open until its owner closes it.
    while session.live():
        try:
            session.app.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
