"""Check that the Caliper app and the browser draw the same scene the same way.

The script serves one 3-D scene that holds every primitive of `caliper.scene` through
`caliper.serve()`: points with values, radii and labels; segments, a strip and index-pair
edges; arrows; wire and filled boxes by class with labels; a lit mesh, a mesh coloured by
values and a textured quad; a pinhole camera with a colour image and a 2-D sketch on its
image plane; a camera with a depth image and its cloud; a camera with a segmentation image;
the class table; and entity transforms. Eight groups sit on a ring around a lit sphere at
the origin, so that the fitted camera looks at the sphere.

Two viewers attach without a window: the Caliper app with `--hidden`, and the viewer page in
headless Chromium through Playwright, drawn by Caliper's WebAssembly views. A scene view fits
its camera once, at the first frame that has bounds, and a viewer that attaches late may draw
that frame before the whole scene has arrived. So each viewer frames the scene again with a
double click once everything has arrived. For each, the script then saves one capture, prints
the fitted camera (its target and distance), clicks the centre of the picture and prints the
entity that the pick reports to the script. It exits with 1 when a value is not finite, a
viewer reports nothing, or the two viewers disagree.

    python tests/acceptance/scene_check.py <output folder> [--app <caliper>] [--adapter metal|swiftshader]

It needs PyTorch, the Caliper app (by default `cmake-build-release/caliper`, or
`CALIPER_TEST_CALIPER_EXE`), the WebAssembly views in `python/caliper/_web/`
(`scripts/build_wasm.py`), Node.js, and Playwright with its Chromium headless shell in
`web/` or `web/wasm/test` (`npm ci` there). Run it from the repository root with
`PYTHONPATH=python`. It downloads nothing and takes well under a minute."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch

import caliper

ROOT = Path(__file__).resolve().parents[2]
SCENE = "check"
RING = 3.0           # the groups sit on a circle of this radius around the sphere
SPHERE = 0.9         # the radius of the sphere at the origin, which holds the fitted target
TOLERANCE = 1e-3     # the relative difference allowed between the two fitted cameras


def ring_pose(k: int, z: float = 0.0, outward: bool = False):
    """The 4 x 4 pose of ring slot k of 8: a translation, and for a camera a rotation that
    makes its forward axis (z of the rdf convention) look away from the sphere, y down."""
    a = k * math.pi / 4
    c, s = math.cos(a), math.sin(a)
    m = torch.eye(4)
    m[:3, 3] = torch.tensor([RING * c, RING * s, z])
    if outward:
        # Columns: right (s, -c, 0), down (0, 0, -1), forward (c, s, 0).
        m[:3, :3] = torch.tensor([[s, 0.0, c], [-c, 0.0, s], [0.0, -1.0, 0.0]])
    return m


def build(sc) -> None:
    """Logs every part of the check scene, all static, so that a viewer that attaches late
    receives the whole scene."""
    S = dict(static=True)
    sc.classes({1: "car", 2: ("tree", (60, 180, 75)), 3: "sign", 4: "road"})

    # The centre: a lit sphere at the origin.
    n_lat, n_lon = 16, 24
    th = torch.linspace(0, math.pi, n_lat + 1)
    ph = torch.linspace(0, 2 * math.pi, n_lon + 1)[:-1]
    T, P = torch.meshgrid(th, ph, indexing="ij")
    normals = torch.stack([torch.sin(T) * torch.cos(P), torch.sin(T) * torch.sin(P), torch.cos(T)], -1).reshape(-1, 3)
    tri = []
    for i in range(n_lat):
        for j in range(n_lon):
            p0, p1 = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            tri += [[p0, p0 + n_lon, p1 + n_lon], [p0, p1 + n_lon, p1]]
    sc.mesh("center", normals * SPHERE, torch.tensor(tri, dtype=torch.int32), normals=normals,
            colors=(200, 200, 210), **S)

    # Slot 0: points by value with growing radii, and three labelled points.
    a = torch.linspace(0, 2 * math.pi, 240)
    ring = torch.stack([torch.cos(a) * 0.5, torch.sin(a) * 0.5, 0.15 * torch.sin(3 * a)], -1)
    sc.points("points/ring", ring, values=a, cmap="viridis", radii=torch.linspace(0.015, 0.05, 240), **S)
    sc.points("points/named", torch.tensor([[0.0, 0.0, 0.3], [0.25, 0.0, 0.2], [-0.25, 0.0, 0.2]]),
              colors=(255, 80, 80), radii=0.06, labels=["origin", "east", "west"], **S)
    sc.transform("points", matrix=ring_pose(0), **S)

    # Slot 1: segments, a strip coloured by value, and the index-pair edges of a small graph.
    seg = torch.tensor([[-0.5, -0.3, -0.3], [0.5, -0.3, -0.3], [-0.5, 0.0, -0.3], [0.5, 0.0, 0.3],
                        [-0.5, 0.3, 0.3], [0.5, 0.3, -0.3]])
    sc.lines("lines/segments", seg, colors=(240, 200, 60), width=3.0, **S)
    t = torch.linspace(0, 4 * math.pi, 160)
    helix = torch.stack([0.3 * torch.cos(t), 0.3 * torch.sin(t), t / (4 * math.pi) - 0.5], -1)
    sc.lines("lines/strip", helix, strip_lengths=[160], values=t, cmap="magma", width=2.5, **S)
    k = torch.arange(8, dtype=torch.float32)
    nodes = torch.stack([0.45 * torch.cos(k * math.pi / 4), 0.45 * torch.sin(k * math.pi / 4), torch.full((8,), 0.5)], -1)
    edges = torch.tensor([[i, (i + 1) % 8] for i in range(8)] + [[i, (i + 3) % 8] for i in range(8)], dtype=torch.int32)
    sc.lines("lines/edges", nodes, indices=edges, colors=(120, 200, 255), width=2.0, **S)
    sc.transform("lines", matrix=ring_pose(1), **S)

    # Slot 2: arrows of a small vector field.
    g = torch.linspace(-0.5, 0.5, 5)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    o = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(25)], -1)
    v = torch.stack([-o[:, 1], o[:, 0], torch.full((25,), 0.25)], -1) * 0.5
    sc.arrows("arrows", o, v, values=v.norm(dim=1), cmap="viridis", width=2.0, **S)
    sc.transform("arrows", matrix=ring_pose(2), **S)

    # Slot 3: wire boxes by class with labels, and filled boxes in their own colours, one rotated.
    c = torch.tensor([[-0.35, 0.0, 0.0], [0.1, 0.0, 0.0], [0.45, 0.0, 0.1]])
    h = torch.tensor([[0.15, 0.2, 0.2], [0.12, 0.12, 0.15], [0.1, 0.1, 0.3]])
    sc.boxes("boxes/wire", c, h, class_ids=torch.tensor([1, 3, 2]), labels=["car 1", "sign", "tree 4"], width=2.0, **S)
    half = 0.3
    sc.boxes("boxes/filled", torch.tensor([[-0.2, -0.45, 0.0], [0.3, -0.45, 0.1]]),
             torch.tensor([[0.12, 0.12, 0.12], [0.15, 0.1, 0.2]]),
             rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, math.sin(half), math.cos(half)]]),
             colors=torch.tensor([[230, 120, 40], [80, 120, 240]], dtype=torch.uint8), fill=True, **S)
    sc.transform("boxes", matrix=ring_pose(3), **S)

    # Slot 4: a surface coloured by values and a textured quad.
    N = 21
    gg = torch.linspace(-0.4, 0.4, N)
    Y, X = torch.meshgrid(gg, gg, indexing="ij")
    Z = 0.15 * torch.sin(6 * X) * torch.cos(6 * Y)
    surf = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    i = torch.arange(N - 1)
    q = (i[:, None] * N + i[None, :]).reshape(-1)
    stri = torch.stack([torch.stack([q, q + 1, q + N + 1], -1), torch.stack([q, q + N + 1, q + N], -1)], 1).reshape(-1, 3)
    sc.mesh("mesh/surface", surf, stri.to(torch.int32), values=Z.reshape(-1), cmap="viridis", **S)
    tex = torch.zeros(32, 32, 3, dtype=torch.uint8)
    ty, tx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    tex[..., 0] = (((tx // 4 + ty // 4) % 2) * 200 + 40).to(torch.uint8)
    tex[..., 1] = (tx * 7).to(torch.uint8)
    tex[..., 2] = (ty * 7).to(torch.uint8)
    quad = torch.tensor([[-0.3, -0.5, -0.3], [0.3, -0.5, -0.3], [0.3, -0.5, 0.3], [-0.3, -0.5, 0.3]])
    sc.mesh("mesh/textured", quad, torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32),
            uvs=torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]), texture=tex, **S)
    sc.transform("mesh", matrix=ring_pose(4), **S)

    # Slot 5: a camera with a colour image and a 2-D sketch on its image plane.
    W, H = 64, 48
    sc.camera("rig", width=W, height=H, fov=45.0, plane_distance=0.5, **S)
    sc.transform("rig", matrix=ring_pose(5, outward=True), **S)
    iy, ix = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    img = torch.stack([(ix * 4).clamp(0, 255), (iy * 5).clamp(0, 255), torch.full_like(ix, 160)], -1).to(torch.uint8)
    sc.image("rig/image", img, opacity=0.85, **S)
    sc.boxes("rig/sketch", corners=torch.tensor([[10.0, 8.0, 30.0, 26.0]]), colors=(255, 255, 0), width=2.0, **S)
    sc.points("rig/keypoints", torch.tensor([[40.0, 30.0], [48.0, 20.0], [52.0, 36.0]]), colors=(255, 0, 255),
              radii=1.5, **S)

    # Slot 6: a camera with a depth image and its back-projected cloud.
    DW, DH = 40, 30
    sc.camera("depthcam", width=DW, height=DH, fov=50.0, plane_distance=0.3, **S)
    sc.transform("depthcam", matrix=ring_pose(6, outward=True), **S)
    dy, dx = torch.meshgrid(torch.arange(DH, dtype=torch.float32), torch.arange(DW, dtype=torch.float32), indexing="ij")
    sc.depth("depthcam/depth", 0.8 + 0.2 * torch.sin(dx / 6) * torch.cos(dy / 5), vmin=0.5, vmax=1.1, **S)

    # Slot 7: a camera with a segmentation image through the class table.
    sc.camera("segcam", width=32, height=24, fov=40.0, plane_distance=0.4, **S)
    sc.transform("segcam", matrix=ring_pose(7, outward=True), **S)
    sy, sx = torch.meshgrid(torch.arange(24), torch.arange(32), indexing="ij")
    labels = torch.where(sy > 16, 4, torch.where(sx < 10, 1, torch.where(sx < 20, 2, 3))).to(torch.uint8)
    sc.segmentation("segcam/labels", labels, opacity=0.8, **S)


def finite(*xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def picks(server, got: list, settle: float = 1.0) -> list:
    """Every pick event on the check scene so far, after the events have settled."""
    end = time.monotonic() + settle
    while time.monotonic() < end:
        got.extend(e for e in server.events() if e.get("view") == SCENE)
        time.sleep(0.05)
    return [e for e in got if e.get("type") == "pick"]


def orbit_of(line: str) -> tuple[list, float]:
    """The orbit's target and distance from a `where` line of the app's UI driver."""
    words = line.split()
    i = words.index("target")
    return [float(w) for w in words[i + 1:i + 4]], float(words[words.index("distance") + 1])


def app_side(exe: str, server, out: Path, events: list) -> dict:
    """The hidden app: a double click frames the scene again, the UI driver's `where` gives
    the fitted camera, Caliper API gives a capture, and one click picks at the centre."""
    work = out / "app-data"
    shutil.rmtree(work, ignore_errors=True)
    (work / "data").mkdir(parents=True)
    (work / "user").mkdir()
    # Two clicks in a row are a double click, which fits the camera to what the view holds.
    ui = (f"where {SCENE}; wait 120; pick {SCENE} 0.5 0.5; pick {SCENE} 0.5 0.5; wait 60; where {SCENE}; "
          f"wait 240; pick {SCENE} 0.5 0.5; wait 60; where {SCENE}; wait 6000")
    env = dict(os.environ, CALIPER_DATA_DIR=str(work / "data"), CALIPER_USER_APPLETS_DIR=str(work / "user"),
               CALIPER_TEST_UI=ui, CALIPER_WINDOW_SIZE="1280x800")
    p = subprocess.Popen([exe, "--hidden", "--url", server.stream_url, "--api", "--api-port", "0"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    lines: list[str] = []
    threading.Thread(target=lambda: [lines.append(ln.rstrip("\n")) for ln in p.stderr], daemon=True).start()

    def where(n: int, timeout: float = 90.0) -> str:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            got = [ln for ln in lines if ln.startswith(f"[ui] where {SCENE} ")]
            if len(got) >= n:
                return got[n - 1]
            if p.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError(f"the app did not report its scene view ({n}):\n" + "\n".join(lines[-30:]))

    result: dict = {"side": "app"}
    try:
        result["target"], result["distance"] = orbit_of(where(2))
        before = len(picks(server, events))      # the double click's picks, before the last click
        if orbit_of(where(3)) != (result["target"], result["distance"]):
            raise RuntimeError("the app's camera moved after its fit: " + where(3))
        api = json.loads((work / "data" / "api.json").read_text())
        url = f"http://127.0.0.1:{api['port']}/v0/capture.png?token={api['token']}"
        with urllib.request.urlopen(url, timeout=20) as r:
            (out / "app.png").write_bytes(r.read())
        result["capture"] = str(out / "app.png")
        after = picks(server, events)
        result["pick"] = after[-1]["detail"] if len(after) > before else None
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(work, ignore_errors=True)
    return result


def browser_side(server, out: Path, adapter: str, events: list) -> dict:
    """The viewer page in headless Chromium: a double click frames the scene again, the
    module's geometry gives the fitted camera, then a capture of the scene's area and one
    click at the centre of the picture."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is not on PATH")
    driver = Path(__file__).with_name("scene_check_browser.mjs")
    cmd = [node, str(driver), server.url, SCENE, str(out / "browser.png"), adapter]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timer = threading.Timer(180.0, p.kill)
    timer.start()
    seen, before = [], 0
    for line in p.stdout:
        seen.append(line.rstrip("\n"))
        if line.startswith("REFIT"):
            # The double click's picks have landed; the driver clicks once more when told.
            before = len(picks(server, events))
            p.stdin.write("go\n")
            p.stdin.flush()
        if line.startswith(("RESULT ", "FAIL")):
            break
    p.stdin.close()
    code = p.wait()
    timer.cancel()
    report = next((ln[len("RESULT "):] for ln in seen if ln.startswith("RESULT ")), None)
    if code != 0 or report is None:
        raise RuntimeError(f"the browser driver failed ({code}):\n" + "\n".join(seen)[-3000:] + "\n" + p.stderr.read()[-2000:])
    r = json.loads(report)
    orbit = r["geometry"]["orbit"]
    after = picks(server, events)
    return {"side": "browser", "target": orbit[:3], "distance": orbit[3], "capture": str(out / "browser.png"),
            "pick": after[-1]["detail"] if len(after) > before else None, "adapter": r.get("adapter", "")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="the folder for the captures")
    ap.add_argument("--app", default=os.environ.get("CALIPER_TEST_CALIPER_EXE") or str(ROOT / "cmake-build-release" / "caliper"))
    ap.add_argument("--adapter", choices=("metal", "swiftshader"), default="metal")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not Path(args.app).is_file():
        print(f"scene_check: no Caliper app at {args.app} (give --app)", file=sys.stderr)
        return 1
    if not (Path(caliper.__file__).parent / "_web" / "caliper-views.wasm").is_file():
        print("scene_check: the WebAssembly views are not built (scripts/build_wasm.py)", file=sys.stderr)
        return 1
    os.environ["CALIPER_SESSION"] = "0"
    started = time.monotonic()
    server = caliper.serve(port=0)
    events: list = []
    try:
        build(caliper.scene(SCENE, up="+z", units="m"))
        sides = [app_side(args.app, server, out, events), browser_side(server, out, args.adapter, events)]
    finally:
        server.close()
    problems = []
    for s in sides:
        tgt, dist = s.get("target") or [], s.get("distance")
        pick = s.get("pick") or {}
        entity = pick.get("entity")
        print(f"{s['side']}: capture {s.get('capture')}")
        print(f"{s['side']}: fitted camera target {tgt} distance {dist}")
        print(f"{s['side']}: pick at the centre {entity!r} ({pick.get('primitive')}, item {pick.get('index')})")
        if len(tgt) != 3 or not finite(*tgt, dist) or not dist > 0:
            problems.append(f"{s['side']}: the fitted camera is not finite")
        if not entity:
            problems.append(f"{s['side']}: the pick at the centre reported no entity")
    app, web = sides
    if all(len(s.get("target") or []) == 3 and finite(s.get("distance")) for s in sides):
        scale = max(1.0, abs(app["distance"]))
        if any(abs(a - b) > TOLERANCE * scale for a, b in zip(app["target"], web["target"])):
            problems.append(f"the fitted targets differ: {app['target']} and {web['target']}")
        if abs(app["distance"] - web["distance"]) > TOLERANCE * scale:
            problems.append(f"the fitted distances differ: {app['distance']} and {web['distance']}")
    a_pick, w_pick = app.get("pick") or {}, web.get("pick") or {}
    if (a_pick.get("entity"), a_pick.get("primitive")) != (w_pick.get("entity"), w_pick.get("primitive")):
        problems.append(f"the picks differ: {a_pick.get('entity')!r} and {w_pick.get('entity')!r}")
    (out / "result.json").write_text(json.dumps({"sides": sides, "problems": problems}, indent=2))
    print(f"took {time.monotonic() - started:.1f} s")
    for p in problems:
        print("FAIL", p)
    print("the app and the browser agree" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
