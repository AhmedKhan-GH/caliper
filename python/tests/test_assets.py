"""The packaged browser assets (caliper._assets) and scripts/build_web.py.

Runs without torch or the native extensions."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from caliper import _assets

REPO = Path(__file__).resolve().parents[2]
PACKAGE_WEB = REPO / "python" / "caliper" / "_web"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "web_placeholder"
BUILD_WEB = REPO / "scripts" / "build_web.py"
PLACEHOLDER_MARK = b"CALIPER PLACEHOLDER ASSET"


def is_placeholder(path: Path) -> bool:
    return PLACEHOLDER_MARK in path.read_bytes()[:400]


requires_npm = pytest.mark.skipif(shutil.which("npm") is None, reason="npm is not on PATH")


def test_assets_are_found_in_the_source_tree(monkeypatch):
    monkeypatch.delenv("CALIPER_WEB_DIR", raising=False)
    assert _assets.asset_dir() == PACKAGE_WEB.resolve()
    _assets.check_assets()
    names = _assets.asset_names()
    for name in _assets.REQUIRED:
        assert name in names
        assert _assets.asset_path(name) == (PACKAGE_WEB / name).resolve()
        assert _assets.read_asset(name) == (PACKAGE_WEB / name).read_bytes()
    # One default export (the anywidget module), written either way a bundler writes it,
    # in the widget of a view and in the widget of the controls panel.
    for module in ("caliper-widget.js", "caliper-controls-widget.js"):
        widget = _assets.read_asset(module)
        assert len(re.findall(rb"export default|export\s*\{[^}]*\bas default\b", widget)) == 1, module
    assert b"caliper-view" in _assets.read_asset("index.html")


def test_the_package_ships_the_wasm_views(monkeypatch):
    # Optional build outputs: a checkout has them only after build_wasm.py or
    # fetch_wasm.py. CI, which builds them first, requires them.
    monkeypatch.delenv("CALIPER_WEB_DIR", raising=False)
    if _assets.missing_views() and os.environ.get("CALIPER_REQUIRE_WEB_VIEWS") != "1":
        pytest.skip(f"the WebAssembly views are not in {PACKAGE_WEB}; build them with "
                    f"`{_assets.WASM_BUILD_COMMAND}` or download them with `{_assets.WASM_FETCH_COMMAND}`")
    names = _assets.asset_names()
    for name in _assets.WASM_VIEWS:
        assert name in names
    assert _assets.read_asset("caliper-views.wasm")[:4] == b"\0asm"
    loader = _assets.read_asset("caliper-views-loader.js")
    assert b"export async function mountCaliperViews" in loader
    assert b"export const CALIPER_VIEWS_KINDS" in loader
    assert b"createCaliperViews" in _assets.read_asset("caliper-views.js")


def placeholder_web(root: Path) -> Path:
    web = root / "web"
    web.mkdir()
    for name in _assets.REQUIRED:
        (web / name).write_text("// placeholder")
    return web


def test_the_wasm_views_are_optional(tmp_path, monkeypatch):
    import io
    web = placeholder_web(tmp_path)
    monkeypatch.setenv("CALIPER_WEB_DIR", str(web))
    monkeypatch.setattr(_assets, "_noted", False)
    _assets.check_assets()                                   # required ones only
    assert _assets.missing_views() == list(_assets.WASM_VIEWS)
    out = io.StringIO()
    assert _assets.note_missing_views(out) is True
    assert _assets.note_missing_views(out) is False          # once per process
    line = out.getvalue()
    assert line.count("\n") == 1 and str(web) in line
    assert _assets.WASM_BUILD_COMMAND in line and _assets.WASM_FETCH_COMMAND in line
    (web / "caliper-views.js").write_text("x")
    assert _assets.missing_views() == ["caliper-views-loader.js", "caliper-views.wasm"]
    for name in _assets.WASM_VIEWS:
        (web / name).write_text("x")
    assert _assets.missing_views() == []
    monkeypatch.setattr(_assets, "_noted", False)
    assert _assets.note_missing_views(io.StringIO()) is False


def test_serve_names_the_two_commands_once_when_the_views_are_missing(tmp_path):
    web = placeholder_web(tmp_path)
    code = ("import caliper\n"
            "s = caliper.serve(); caliper.serve(); s.close()\n"
            "s = caliper.serve(); s.close()\n")
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), CALIPER_WEB_DIR=str(web), CALIPER_SESSION="0")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    notes = [l for l in r.stderr.splitlines() if "WebAssembly views" in l]
    assert len(notes) == 1, r.stderr
    assert _assets.WASM_BUILD_COMMAND in notes[0] and _assets.WASM_FETCH_COMMAND in notes[0]
    for name in _assets.WASM_VIEWS:
        (web / name).write_text("x")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert "WebAssembly views" not in r.stderr


def test_assets_are_found_with_pythonpath(tmp_path):
    # Imported from the source tree (PYTHONPATH=python) in a fresh interpreter,
    # from an unrelated working directory.
    code = ("import caliper._assets as a; a.check_assets(); "
            "print(a.asset_dir()); print(','.join(a.asset_names()))")
    env = {**os.environ, "PYTHONPATH": str(REPO / "python")}
    env.pop("CALIPER_WEB_DIR", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    assert Path(out[0]) == PACKAGE_WEB.resolve()
    assert set(_assets.REQUIRED) <= set(out[1].split(","))


def test_content_types():
    assert _assets.content_type("index.html") == "text/html; charset=utf-8"
    assert _assets.content_type("caliper-view.js") == "text/javascript; charset=utf-8"
    assert _assets.content_type("fonts/IBMPlexSansCondensed-Regular.woff2") == "font/woff2"
    assert _assets.content_type("fonts/x.TTF") == "font/ttf"
    assert _assets.content_type("viewer.wasm") == "application/wasm"
    assert _assets.content_type("blob") == "application/octet-stream"


@pytest.mark.parametrize("name", [
    "", "..", "../__init__.py", "a/../../__init__.py", "/etc/passwd", "..\\__init__.py",
    "C:/x.js", ".hidden", "fonts/.x", "./index.html", "fonts//x.woff2", "x\0.js",
])
def test_path_traversal_is_refused(name):
    with pytest.raises(ValueError):
        _assets.asset_path(name)
    with pytest.raises(ValueError):
        _assets.read_asset(name)


def test_symlink_out_of_the_folder_is_refused(tmp_path, monkeypatch):
    web = tmp_path / "web"
    web.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    (web / "index.html").write_text("<!doctype html>")
    (web / "leak.txt").symlink_to(secret)
    monkeypatch.setenv("CALIPER_WEB_DIR", str(web))
    assert _assets.read_asset("index.html") == b"<!doctype html>"
    with pytest.raises(ValueError):
        _assets.asset_path("leak.txt")


@pytest.mark.parametrize("url, name", [
    ("/", "index.html"),
    ("/?token=abc", "index.html"),
    ("/caliper-view.js", "caliper-view.js"),
    ("/caliper-widget.js?v=3#x", "caliper-widget.js"),
    ("/fonts/Plex%20Sans.woff2", "fonts/Plex Sans.woff2"),
    ("/fonts/", "fonts/index.html"),
])
def test_name_for_url_path(url, name):
    assert _assets.name_for_url_path(url) == name


@pytest.mark.parametrize("url", [
    "/../x", "/%2e%2e/__init__.py", "/fonts/%2E%2E/%2E%2E/x", "/%2Fetc%2Fpasswd",
    "/..%5C__init__.py", "//etc/passwd", "relative.js", "/%ff.js", "/.git/config",
])
def test_unsafe_url_paths_are_refused(url):
    with pytest.raises(ValueError):
        _assets.name_for_url_path(url)


def test_missing_asset_names_the_build_command(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIPER_WEB_DIR", str(tmp_path))
    with pytest.raises(_assets.AssetMissing, match="scripts/build_web.py") as e:
        _assets.read_asset("caliper-widget.js")
    assert isinstance(e.value, FileNotFoundError)
    with pytest.raises(_assets.AssetMissing, match="caliper-view.js"):
        _assets.check_assets()
    assert _assets.asset_names() == []


def run_build_web(*args, cwd=None, wasm=None):
    # Without a WebAssembly build unless one is given: a checkout's own
    # web/wasm/build must not change what these tests see.
    wasm_args = ["--wasm-build", wasm if wasm is not None else REPO / "web" / "wasm" / "no-such-build"]
    return subprocess.run([sys.executable, str(BUILD_WEB), *map(str, args), *map(str, wasm_args)], cwd=cwd,
                          capture_output=True, text=True)


def fake_wasm_build(root: Path) -> Path:
    build = root / "wasm-build"
    build.mkdir()
    for name in _assets.WASM_VIEWS:
        (build / name).write_bytes(b"\0asm\1\0\0\0" if name.endswith(".wasm") else f"// {name}\n".encode())
    (build / "demo.html").write_text("<!doctype html>")   # not one of the three: not copied
    return build


@requires_npm
def test_build_web_copies_and_checks(tmp_path):
    web = tmp_path / "web"
    shutil.copytree(FIXTURE, web, ignore=shutil.ignore_patterns("dist", "node_modules"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.js").write_text("old")

    r = run_build_web("--web", web, "--out", out)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir()) == sorted(_assets.REQUIRED)
    for name in _assets.REQUIRED:
        assert (out / name).read_bytes() == (web / "dist" / name).read_bytes()
        assert is_placeholder(out / name)

    r = run_build_web("--web", web, "--out", out, "--check", "--no-install")
    assert r.returncode == 0, r.stderr

    (out / "caliper-view.js").write_bytes(b"// edited\n")
    (out / "extra.woff2").write_bytes(b"\0")
    (out / "index.html").write_bytes((web / "dist" / "index.html").read_bytes().replace(b"\n", b"\r\n"))
    r = run_build_web("--web", web, "--out", out, "--check", "--no-install")
    assert r.returncode == 1
    assert "differs from the build: caliper-view.js" in r.stderr
    assert "not in the build: extra.woff2" in r.stderr
    assert "index.html" not in r.stderr          # line endings alone are no difference
    assert (out / "caliper-view.js").read_bytes() == b"// edited\n"   # --check changes nothing


@requires_npm
def test_build_web_refuses_a_folder_with_python_files(tmp_path):
    web = tmp_path / "web"
    shutil.copytree(FIXTURE, web, ignore=shutil.ignore_patterns("dist", "node_modules"))
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "keep.py").write_text("")
    r = run_build_web("--web", web, "--out", out, "--no-install")
    assert r.returncode == 2
    assert (out / "keep.py").exists()


@requires_npm
def test_checked_in_placeholders_match_their_fixture(tmp_path):
    # Until the orchestrator checks in the real build of web/, the package ships
    # the placeholder build of the fixture; it must be current.
    if not all(is_placeholder(PACKAGE_WEB / n) for n in _assets.REQUIRED):
        pytest.skip("the checked-in assets are the real build of web/")
    web = tmp_path / "web"
    shutil.copytree(FIXTURE, web, ignore=shutil.ignore_patterns("dist", "node_modules"))
    r = run_build_web("--web", web, "--check")
    assert r.returncode == 0, r.stderr


def test_wasm_views_content_types():
    assert _assets.content_type("caliper-views-loader.js") == "text/javascript; charset=utf-8"
    assert _assets.content_type("caliper-views.js") == "text/javascript; charset=utf-8"
    assert _assets.content_type("caliper-views.wasm") == "application/wasm"


@requires_npm
def test_build_web_copies_the_wasm_views_from_their_build(tmp_path):
    web = tmp_path / "web"
    shutil.copytree(FIXTURE, web, ignore=shutil.ignore_patterns("dist", "node_modules"))
    out = tmp_path / "out"
    wasm = fake_wasm_build(tmp_path)
    r = run_build_web("--web", web, "--out", out, wasm=wasm)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir()) == sorted(_assets.REQUIRED + _assets.WASM_VIEWS)
    for name in _assets.WASM_VIEWS:
        assert (out / name).read_bytes() == (wasm / name).read_bytes()
    r = run_build_web("--web", web, "--out", out, "--check", "--no-install", wasm=wasm)
    assert r.returncode == 0, r.stderr
    (out / "caliper-views.wasm").write_bytes(b"\0asm\2\0\0\0")
    r = run_build_web("--web", web, "--out", out, "--check", "--no-install", wasm=wasm)
    assert r.returncode == 1
    assert "differs from the build: caliper-views.wasm" in r.stderr


@requires_npm
def test_build_web_keeps_the_checked_in_wasm_views_without_a_build(tmp_path):
    web = tmp_path / "web"
    shutil.copytree(FIXTURE, web, ignore=shutil.ignore_patterns("dist", "node_modules"))
    out = tmp_path / "out"
    out.mkdir()
    for name in _assets.WASM_VIEWS:
        (out / name).write_bytes(b"checked in")
    r = run_build_web("--web", web, "--out", out)
    assert r.returncode == 0, r.stderr
    assert "the checked-in views are kept" in r.stdout
    for name in _assets.WASM_VIEWS:
        assert (out / name).read_bytes() == b"checked in"
    # Nothing to compare them with: --check compares the rest only.
    r = run_build_web("--web", web, "--out", out, "--check", "--no-install")
    assert r.returncode == 0, r.stderr
