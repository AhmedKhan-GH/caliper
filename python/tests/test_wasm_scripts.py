"""scripts/build_wasm.py and scripts/fetch_wasm.py, the two ways the browser's
WebAssembly views reach python/caliper/_web/ in a checkout.

Runs without torch, the native extensions, Emscripten or the network:
build_wasm.py runs against stand-ins for emcc, emcmake, cmake and ninja, and
fetch_wasm.py against a stand-in for the GitHub API on 127.0.0.1."""
import io
import json
import os
import re
import socketserver
import stat
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
BUILD_WASM = SCRIPTS / "build_wasm.py"
FETCH_WASM = SCRIPTS / "fetch_wasm.py"
sys.path.insert(0, str(SCRIPTS))
import wasm_views  # noqa: E402

PIN = wasm_views.pinned_emscripten()
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tools are shell scripts")


def good_files(tag: str = "a") -> dict[str, bytes]:
    return {"caliper-views-loader.js": f"export async function mountCaliperViews() {{}} // {tag}\n".encode(),
            "caliper-views.js": f"export default function createCaliperViews() {{}} // {tag}\n".encode(),
            "caliper-views.wasm": b"\0asm\1\0\0\0" + tag.encode()}


def test_the_pin_is_one_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", PIN)
    assert wasm_views.PIN_FILE.read_text().endswith("\n")


def test_problems_names_what_is_wrong():
    assert wasm_views.problems(good_files()) == []
    bad = good_files()
    bad["caliper-views.wasm"] = b"<html>"
    bad["caliper-views.js"] = b"export default 1"
    del bad["caliper-views-loader.js"]
    bad["extra.txt"] = b"x"
    got = wasm_views.problems(bad)
    assert "caliper-views-loader.js is missing" in got
    assert "caliper-views.wasm is not a WebAssembly module" in got
    assert "caliper-views.js is not the module's glue (no createCaliperViews)" in got
    assert "extra.txt is not one of the three files" in got


def test_install_replaces_each_file_whole(tmp_path):
    out = tmp_path / "web"
    out.mkdir()
    (out / "caliper-view.js").write_text("kept")
    wasm_views.install(good_files("1"), [out])
    wasm_views.install(good_files("2"), [out])
    assert sorted(p.name for p in out.iterdir()) == sorted([*wasm_views.FILES, "caliper-view.js"])
    for name, data in good_files("2").items():
        assert (out / name).read_bytes() == data
        if sys.platform != "win32":                       # readable by all, as a new file is
            assert (out / name).stat().st_mode & 0o044 == 0o044


# ---- build_wasm.py ---------------------------------------------------------------

def tools(root: Path, version: str) -> Path:
    """Stand-ins: emcc prints a version, emcmake runs its arguments, cmake
    writes the three files into the build folder on --build, ninja exists."""
    bin_ = root / "bin"
    bin_.mkdir()
    loader = good_files("built")["caliper-views-loader.js"].decode()
    glue = good_files("built")["caliper-views.js"].decode()
    scripts = {
        "emcc": f'echo "emcc (Emscripten gcc/clang-like replacement + linker emulating GNU ld) {version} (abc)"\n',
        "emcmake": 'exec "$@"\n',
        "ninja": "exit 0\n",
        "cmake": f"""
if [ "$1" = "--build" ]; then
  printf '%s' '{loader}' > "$2/caliper-views-loader.js"
  printf '%s' '{glue}' > "$2/caliper-views.js"
  printf '\\000asm\\001\\000\\000\\000built' > "$2/caliper-views.wasm"
  exit 0
fi
while [ $# -gt 0 ]; do if [ "$1" = "-B" ]; then mkdir -p "$2"; fi; shift; done
echo "$PWD" > /dev/null
""",
    }
    for name, body in scripts.items():
        p = bin_ / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return bin_


def run_build_wasm(tmp_path: Path, path: str, *args: str, emsdk: str | None = None):
    env = {k: v for k, v in os.environ.items() if k not in ("EMSDK", "EM_CONFIG")}
    env["PATH"] = path
    if emsdk is not None:
        env["EMSDK"] = emsdk
    return subprocess.run([sys.executable, str(BUILD_WASM), "--build-dir", str(tmp_path / "build"), *args],
                          capture_output=True, text=True, env=env)


@posix_only
def test_build_wasm_without_emscripten_says_what_to_do(tmp_path):
    r = run_build_wasm(tmp_path, "/usr/bin:/bin", "--out", str(tmp_path / "out"))
    assert r.returncode == 2
    assert "emcc is not on PATH" in r.stderr
    assert PIN in r.stderr and "scripts/fetch_wasm.py" in r.stderr
    assert not (tmp_path / "out").exists()


@posix_only
def test_build_wasm_refuses_another_emscripten(tmp_path):
    bin_ = tools(tmp_path, "1.2.3")
    r = run_build_wasm(tmp_path, f"{bin_}:/usr/bin:/bin", "--out", str(tmp_path / "out"))
    assert r.returncode == 2
    assert f"emcc is Emscripten 1.2.3, and web/wasm is built with {PIN}" in r.stderr
    assert f"emsdk install {PIN}" in r.stderr
    assert not (tmp_path / "out").exists()
    r = run_build_wasm(tmp_path, f"{bin_}:/usr/bin:/bin", "--out", str(tmp_path / "out"), "--any-version")
    assert r.returncode == 0, r.stderr
    assert "building with it anyway" in r.stderr


@posix_only
def test_build_wasm_builds_and_copies(tmp_path):
    bin_ = tools(tmp_path, PIN)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    r = run_build_wasm(tmp_path, f"{bin_}:/usr/bin:/bin", "--out", str(out1), "--out", str(out2))
    assert r.returncode == 0, r.stderr
    assert f"Emscripten {PIN}" in r.stdout
    for out in (out1, out2):
        assert sorted(p.name for p in out.iterdir()) == sorted(wasm_views.FILES)
        assert (out / "caliper-views.wasm").read_bytes().startswith(b"\0asm")
    r = run_build_wasm(tmp_path, f"{bin_}:/usr/bin:/bin", "--no-copy")
    assert r.returncode == 0 and "copied" not in r.stdout


@posix_only
def test_build_wasm_takes_emscripten_from_emsdk(tmp_path):
    emsdk = tmp_path / "emsdk"
    (emsdk / "upstream").mkdir(parents=True)
    tools(tmp_path, PIN).rename(emsdk / "upstream" / "emscripten")
    r = run_build_wasm(tmp_path, "/usr/bin:/bin", "--no-copy", emsdk=str(emsdk))
    assert "emcc is not on PATH" not in r.stderr
    # The stand-in cmake and ninja came with the stand-in emsdk here.
    assert r.returncode == 0, r.stderr
    r = run_build_wasm(tmp_path, "/usr/bin:/bin", "--no-copy", emsdk=str(tmp_path / "nowhere"))
    assert r.returncode == 2 and "holds no activated Emscripten" in r.stderr


# ---- fetch_wasm.py -------------------------------------------------------------

def zipped(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


class FakeGitHub:
    """The GitHub API as fetch_wasm.py uses it, on 127.0.0.1. Artifact
    downloads need a token, as on GitHub; release assets do not."""

    def __init__(self):
        self.artifacts: list[dict] = []
        self.blobs: dict[str, bytes] = {}
        self.release: dict | None = None
        self.seen: list[tuple[str, str | None]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body=b"", ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.seen.append((self.path, self.headers.get("Authorization")))
                assert self.headers.get("User-Agent") == "caliper-fetch-wasm"
                path = self.path.split("?", 1)[0]
                if path == "/repos/o/r/actions/artifacts":
                    return self.send(200, json.dumps({"artifacts": fake.artifacts}).encode())
                if path.startswith("/zip/"):
                    if not self.headers.get("Authorization"):
                        return self.send(401, b'{"message": "Requires authentication"}')
                    self.send_response(302)                  # GitHub sends the archive from another host
                    self.send_header("Location", f"{fake.base}/blob/{path[5:]}")
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
                if path.startswith("/blob/") or path.startswith("/dl/"):
                    return self.send(200, fake.blobs[path.rsplit("/", 1)[1]], "application/zip")
                if path == "/repos/o/r/releases/latest" and fake.release:
                    return self.send(200, json.dumps(fake.release).encode())
                return self.send(404, b'{"message": "Not Found"}')

        class LoopbackServer(ThreadingHTTPServer):
            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                self.server_name = "127.0.0.1"
                self.server_port = self.server_address[1]

        self.server = LoopbackServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add_artifact(self, sha: str, files: dict[str, bytes], expired: bool = False, created: str = "2026-11-15"):
        n = str(len(self.artifacts) + 1)
        self.blobs[n] = zipped(files)
        self.artifacts.append({"id": int(n), "name": "caliper-views", "expired": expired, "created_at": created,
                               "archive_download_url": f"{self.base}/zip/{n}",
                               "workflow_run": {"id": 100 + int(n), "head_sha": sha}})

    def add_release(self, tag: str, files: dict[str, bytes]):
        self.blobs["caliper-views.zip"] = zipped(files)
        self.release = {"tag_name": tag, "assets": [
            {"name": "caliper-views.zip", "browser_download_url": f"{self.base}/dl/caliper-views.zip"}]}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def github():
    g = FakeGitHub()
    yield g
    g.close()


def git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@localhost",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@localhost")
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A repository with three commits: the module's sources, a change to the
    docs only, and a change to the module's sources."""
    repo = tmp_path / "repo"
    (repo / "web" / "wasm").mkdir(parents=True)
    (repo / "docs").mkdir()
    git(repo.parent, "init", "-q", str(repo))
    (repo / "web" / "wasm" / "main.cpp").write_text("1")
    (repo / "docs" / "a.md").write_text("1")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")
    (repo / "docs" / "a.md").write_text("2")
    git(repo, "commit", "-q", "-am", "docs only")
    (repo / "web" / "wasm" / "main.cpp").write_text("2")
    git(repo, "commit", "-q", "-am", "sources")
    return repo


def run_fetch(github: FakeGitHub, checkout: Path, out: Path, *args: str, token: str | None = None):
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
    if token:
        env["GITHUB_TOKEN"] = token
    return subprocess.run([sys.executable, str(FETCH_WASM), "--api", github.base, "--repo", "o/r",
                           "--checkout", str(checkout), "--out", str(out), *args],
                          capture_output=True, text=True, env=env, timeout=60)


def installed(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in out.iterdir()} if out.is_dir() else {}


def test_fetch_takes_ci_build_of_this_commit_with_a_token(github, checkout, tmp_path):
    head = git(checkout, "rev-parse", "HEAD")
    github.add_artifact("0" * 40, good_files("other"), created="2026-11-16")
    github.add_artifact(head, good_files("head"))
    github.add_release("v1", good_files("release"))
    out = tmp_path / "out"
    r = run_fetch(github, checkout, out, token="sekrit")
    assert r.returncode == 0, r.stderr
    assert installed(out) == good_files("head")
    assert f"commit {head[:12]}" in r.stdout
    # The token went to the API only, never to the host the archive came from.
    auth = {p.split("?")[0]: a for p, a in github.seen}
    assert auth["/repos/o/r/actions/artifacts"] == "Bearer sekrit"
    assert auth["/zip/2"] == "Bearer sekrit"
    assert auth["/blob/2"] is None
    assert "/repos/o/r/releases/latest" not in auth


def test_fetch_takes_an_earlier_build_with_the_same_sources(github, checkout, tmp_path):
    first, docs, head = git(checkout, "rev-list", "--reverse", "HEAD").split()
    github.add_artifact(first, good_files("first"))
    out = tmp_path / "out"
    r = run_fetch(github, checkout, out, "--commit", docs, token="t")
    assert r.returncode == 0, r.stderr
    assert installed(out) == good_files("first")
    # The module's sources changed after the build: it is not this commit's.
    r = run_fetch(github, checkout, tmp_path / "out2", "--commit", head, token="t")
    assert r.returncode == 1
    assert "CI has no build" in r.stdout and "no build of the views was found" in r.stderr
    assert installed(tmp_path / "out2") == {}


def test_fetch_without_a_token_takes_the_release_and_says_so(github, checkout, tmp_path):
    github.add_artifact(git(checkout, "rev-parse", "HEAD"), good_files("head"))
    github.add_release("v1", good_files("release"))
    out = tmp_path / "out"
    r = run_fetch(github, checkout, out)
    assert r.returncode == 0, r.stderr
    assert installed(out) == good_files("release")
    assert "set GITHUB_TOKEN" in r.stderr
    assert "from release v1, which may be older than this checkout" in r.stderr
    assert all(a is None for _, a in github.seen)


def test_fetch_release_only(github, checkout, tmp_path):
    github.add_artifact(git(checkout, "rev-parse", "HEAD"), good_files("head"))
    github.add_release("v1", good_files("release"))
    r = run_fetch(github, checkout, tmp_path / "out", "--release", token="t")
    assert r.returncode == 0, r.stderr
    assert installed(tmp_path / "out") == good_files("release")
    assert not any(p.startswith("/repos/o/r/actions") for p, _ in github.seen)


def test_fetch_before_the_first_ci_run(github, checkout, tmp_path):
    r = run_fetch(github, checkout, tmp_path / "out", token="t")
    assert r.returncode == 1
    assert "has no release yet" in r.stdout
    assert "no build of the views was found" in r.stderr
    assert "scripts/build_wasm.py" in r.stderr and PIN in r.stderr
    assert installed(tmp_path / "out") == {}


def test_fetch_refuses_a_build_that_is_not_the_module(github, checkout, tmp_path):
    bad = good_files("bad")
    bad["caliper-views.wasm"] = b"<!doctype html>"
    github.add_artifact(git(checkout, "rev-parse", "HEAD"), bad)
    out = tmp_path / "out"
    out.mkdir()
    r = run_fetch(github, checkout, out, token="t")
    assert r.returncode == 1
    assert "caliper-views.wasm is not a WebAssembly module" in r.stderr
    assert installed(out) == {}


def test_fetch_offline_fails_with_a_clear_message(checkout, tmp_path):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                       # nothing listens there now
    r = subprocess.run([sys.executable, str(FETCH_WASM), "--api", f"http://127.0.0.1:{port}", "--repo", "o/r",
                        "--checkout", str(checkout), "--out", str(tmp_path / "out")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert f"cannot reach 127.0.0.1:{port}" in r.stderr
    assert "scripts/build_wasm.py" in r.stderr
    assert installed(tmp_path / "out") == {}



def test_the_workflow_runs_when_a_source_of_the_module_changes():
    # fetch_wasm.py takes an earlier commit's build when none of INPUTS changed
    # since, which is right only if the Browser views workflow runs whenever
    # one of them changes.
    yaml = pytest.importorskip("yaml")
    flow = yaml.safe_load((REPO / ".github" / "workflows" / "web.yml").read_text())
    on = flow.get("on") or flow.get(True)              # YAML 1.1 reads the key `on` as True

    def covers(pattern: str, source: str) -> bool:
        return pattern in (source, f"{source}/**") or (
            pattern.endswith("/**") and f"{source}/".startswith(pattern[:-2]))

    for event in ("push", "pull_request"):
        paths = on[event]["paths"]
        for source in wasm_views.INPUTS:
            assert any(covers(p, source) for p in paths), (event, source)
        assert ".github/workflows/web.yml" in paths
