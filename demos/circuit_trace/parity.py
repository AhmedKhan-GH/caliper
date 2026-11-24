"""Graph parity with actual MCP resource values and the existing piece harness."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from _checks import check
from _mcp import Failure


def compare(app, app_ids, web, web_ids):
    """Use the established two-level interior and one-percent edge tolerances."""
    def interior(ids):
        padded = np.pad(ids, ((1, 1), (1, 1), (0, 0)), mode="edge")
        same = np.ones(ids.shape[:2], dtype=bool)
        for dy in range(3):
            for dx in range(3):
                same &= (padded[dy:dy + ids.shape[0], dx:dx + ids.shape[1]] == ids).all(axis=2)
        return same
    a, b = app.astype(np.int16), web.astype(np.int16)
    diff = np.abs(a - b).max(axis=2)
    inside = interior(app_ids) & interior(web_ids)
    def neighbor_match(left, right):
        padded = np.pad(right, ((1, 1), (1, 1), (0, 0)), mode="edge")
        match = np.zeros(left.shape[:2], dtype=bool)
        for dy in range(3):
            for dx in range(3):
                match |= np.abs(left - padded[dy:dy + left.shape[0], dx:dx + left.shape[1]]).max(axis=2) <= 2
        return match
    edge_bad = ~inside & (diff > 2) & ~neighbor_match(a, b) & ~neighbor_match(b, a)
    return {"interior_over": int(((diff > 2) & inside).sum()),
            "interior_id_mismatch": int(((app_ids != web_ids).any(axis=2) & inside).sum()),
            "maximum_interior_difference": int(diff[inside].max()) if inside.any() else None,
            "edge_share": float(edge_bad.mean())}


def run_parity(session, out):
    from run import ROOT, executable, save
    out.mkdir()
    snapshot = {"selected_node": session.mcp.body("state.get", ids=["selected_node"])["states"][0]["value"], "resources": {}}
    step = None
    for name in ("nodes", "curves", "labels", "strings"):
        reply = session.read("attribution", name)
        if step is None:
            step = reply["step"]
        if reply["step"] != step:
            raise Failure("The graph parity resources span different steps.")
        total = reply["count"]
        rows = list(reply["elements"])
        while len(rows) < total:
            page = session.mcp.body("program.read", id="attribution", resource=name, step=step, first=len(rows), count=1024)
            if not page["elements"] or page["step"] != step:
                raise Failure("The graph parity resource page is incomplete.")
            rows.extend(page["elements"])
        snapshot["resources"][name] = {"count": total, "elements": rows}
    save(out / "resources.json", snapshot)
    native = executable("CALIPER_CIRCUIT_PARITY_EXE", "cmake-build-release/caliper_circuit_parity")
    with (out / "native.log").open("w") as log:
        subprocess.run([str(native), str(out / "resources.json"), str(out)], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    env = dict(os.environ, CI="1")
    with (out / "browser.log").open("w") as log:
        subprocess.run(["node", str(Path(__file__).with_name("browser.mjs")), str(out)], env=env,
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    app = np.fromfile(out / "native.rgba", dtype=np.uint8).reshape(512, 512, 4)
    web = np.fromfile(out / "browser.rgba", dtype=np.uint8).reshape(512, 512, 4)
    a_ids = np.fromfile(out / "native.id", dtype="<u4").reshape(512, 512, 2)
    w_ids = np.fromfile(out / "browser.id", dtype="<u4").reshape(512, 512, 2)
    metrics = compare(app, a_ids, web, w_ids)
    save(out / "comparison.json", metrics)
    Image.fromarray(app).save(out / "native.png")
    Image.fromarray(web).save(out / "browser.png")
    return check("fixture_parity", metrics["interior_over"] == 0 and metrics["interior_id_mismatch"] == 0
                 and metrics["edge_share"] <= .01, metrics,
                 "interior color <=2 levels, exact interior IDs, edge difference <=1%")
