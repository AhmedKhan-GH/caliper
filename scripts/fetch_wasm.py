#!/usr/bin/env python3
"""Download CI's build of Caliper's views compiled to WebAssembly into the Python package.

    python scripts/fetch_wasm.py                    # CI's build for this commit, else the latest release's
    python scripts/fetch_wasm.py --release          # the latest release's
    python scripts/fetch_wasm.py --release v0.2.0   # that release's
    python scripts/fetch_wasm.py --commit <sha>     # CI's build for another commit

It uses only the standard library and needs no Emscripten. It looks in this
order and takes the first build it finds:

1. The artifact `caliper-views` of the Browser views workflow for this
   checkout's commit, or for the newest earlier commit whose sources of the
   module (web/wasm, src, include, sdk/include, assets/fonts and the ImGui and
   ImPlot submodules) are the same. GitHub lists artifacts to anyone but hands
   them out only to signed-in clients, so this step needs a token in
   GITHUB_TOKEN or GH_TOKEN, one that can read the repository's Actions.
2. The asset `caliper-views.zip` of the latest release, or of the release
   --release names. It needs no token, and it may be older than this checkout.

The token is sent only to the GitHub API and only when one of the two
variables is set; nothing else about the user or the machine is sent. The
three files are checked (the module must be WebAssembly and the loader and
the glue must be the module's) and copied, as `scripts/build_wasm.py` copies
them, into python/caliper/_web/ and into the caliper package this Python has
installed, if any, or into the folders --out names. The repository is the one
`origin` points to on GitHub, else AhmedKhan-GH/caliper; --repo names
another.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wasm_views  # noqa: E402

API = "https://api.github.com"
DEFAULT_REPO = "AhmedKhan-GH/caliper"
ARTIFACT = "caliper-views"
RELEASE_ASSET = "caliper-views.zip"
# A generic client name: no user, host or address goes into a request.
USER_AGENT = "caliper-fetch-wasm"
TIMEOUT_S = 30
MAX_DOWNLOAD_BYTES = 3 * wasm_views.MAX_FILE_BYTES


class Fail(Exception):
    pass


class NotFound(Exception):
    pass


def token_from_env() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def fetch(url: str, token: str | None = None, accept: str = "application/vnd.github+json") -> bytes:
    """GET url. The token goes only on this request, never on a redirect,
    which for a download leads to another host."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept,
                                               "X-GitHub-Api-Version": "2022-11-28"})
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    host = urllib.parse.urlsplit(url).netloc
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = r.read(MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise NotFound(url) from None
        detail = ""
        if e.code in (401, 403):
            body = e.read(2000).decode("utf-8", "replace")
            detail = " (GitHub's rate limit; try again later or set GITHUB_TOKEN)" \
                if "rate limit" in body.lower() else " (the token is missing or cannot read this)"
        raise Fail(f"{host} answered {e.code} for {urllib.parse.urlsplit(url).path}{detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise Fail(f"cannot reach {host} ({reason}). Check the network, or build the views with "
                   f"`python scripts/build_wasm.py` (needs Emscripten {wasm_views.pinned_emscripten()})") from None
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise Fail(f"the download from {host} is larger than {MAX_DOWNLOAD_BYTES >> 20} MB")
    return data


def fetch_json(url: str, token: str | None = None):
    try:
        return json.loads(fetch(url, token).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Fail(f"{url} did not answer with JSON") from None


def git(checkout: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True)


def repo_from_origin(checkout: Path) -> str | None:
    r = git(checkout, "config", "--get", "remote.origin.url")
    m = re.search(r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$", r.stdout.strip())
    return m.group(1) if r.returncode == 0 and m else None


def same_sources(checkout: Path, built: str, commit: str) -> bool:
    """Whether a build of commit `built` is a build of `commit`: `built` is
    `commit` or an earlier commit, and no source of the module changed between."""
    if built == commit:
        return True
    if git(checkout, "cat-file", "-e", f"{built}^{{commit}}").returncode != 0:
        return False
    if git(checkout, "merge-base", "--is-ancestor", built, commit).returncode != 0:
        return False
    return git(checkout, "diff", "--quiet", built, commit, "--", *wasm_views.INPUTS).returncode == 0


def files_from_zip(data: bytes, where: str) -> dict[str, bytes]:
    """The three files at the top of a zip archive, checked."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise Fail(f"{where} is not a zip archive") from None
    files = {}
    with z:
        for info in z.infolist():
            if info.filename in wasm_views.FILES and not info.is_dir():
                if info.file_size > wasm_views.MAX_FILE_BYTES:
                    raise Fail(f"{info.filename} in {where} is larger than {wasm_views.MAX_FILE_BYTES >> 20} MB")
                files[info.filename] = z.read(info)
    bad = wasm_views.problems(files)
    if bad:
        raise Fail(f"{where} does not hold a usable build: " + "; ".join(bad))
    return files


def from_ci(api: str, repo: str, checkout: Path, commit: str, token: str | None) -> tuple[dict, str] | None:
    """CI's build for commit, as (files, where it came from), or None."""
    listing = fetch_json(f"{api}/repos/{repo}/actions/artifacts?name={ARTIFACT}&per_page=100", token)
    artifacts = [a for a in listing.get("artifacts", [])
                 if a.get("name") == ARTIFACT and not a.get("expired")]
    artifacts.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    chosen = next((a for a in artifacts if (a.get("workflow_run") or {}).get("head_sha") == commit), None)
    if chosen is None:
        chosen = next((a for a in artifacts
                       if same_sources(checkout, (a.get("workflow_run") or {}).get("head_sha") or "", commit)),
                      None)
    if chosen is None:
        print(f"fetch_wasm: CI has no build for {commit[:12]} or an earlier commit with the same sources")
        return None
    run = chosen.get("workflow_run") or {}
    built = run.get("head_sha") or ""
    where = f"CI run https://github.com/{repo}/actions/runs/{run.get('id')} (commit {built[:12]})"
    if not token:
        print(f"fetch_wasm: {where} built the views, but GitHub hands out CI builds only to signed-in "
              "clients; set GITHUB_TOKEN (or GH_TOKEN) to a token that can read the repository's "
              "Actions to download it", file=sys.stderr)
        return None
    data = fetch(chosen["archive_download_url"], token)
    return files_from_zip(data, where), where


def from_release(api: str, repo: str, tag: str | None) -> tuple[dict, str] | None:
    path = f"releases/tags/{urllib.parse.quote(tag)}" if tag else "releases/latest"
    try:
        release = fetch_json(f"{api}/repos/{repo}/{path}")
    except NotFound:
        print(f"fetch_wasm: {repo} has {'no release ' + tag if tag else 'no release yet'}")
        return None
    name = release.get("tag_name") or tag or "?"
    asset = next((a for a in release.get("assets", []) if a.get("name") == RELEASE_ASSET), None)
    if asset is None:
        print(f"fetch_wasm: release {name} has no {RELEASE_ASSET}")
        return None
    where = f"release {name}"
    return files_from_zip(fetch(asset["browser_download_url"], accept="application/octet-stream"), where), where


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--release", nargs="?", const="", default=None, metavar="TAG",
                       help="take the asset of this release, or of the latest one, and skip CI's builds")
    which.add_argument("--commit", help="take CI's build for this commit only (default: this checkout's "
                                        "commit, then the latest release)")
    ap.add_argument("--repo", help=f"OWNER/NAME on GitHub (default: origin's, else {DEFAULT_REPO})")
    ap.add_argument("--out", type=Path, action="append",
                    help="a folder to copy the three files into, instead of python/caliper/_web/ "
                         "and the installed package (may be given more than once)")
    ap.add_argument("--checkout", type=Path, default=wasm_views.ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--api", default=API, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    checkout = args.checkout.resolve()
    api = args.api.rstrip("/")
    repo = args.repo or repo_from_origin(checkout) or DEFAULT_REPO
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        print(f"fetch_wasm: --repo must be OWNER/NAME, not {repo!r}", file=sys.stderr)
        return 2
    token = token_from_env()

    commit = args.commit
    if commit is None and args.release is None:
        r = git(checkout, "rev-parse", "HEAD")
        commit = r.stdout.strip() if r.returncode == 0 else None
        if commit is None:
            print("fetch_wasm: this is not a git checkout, so only a release's build can be used")
    elif commit is not None:
        r = git(checkout, "rev-parse", "--verify", f"{commit}^{{commit}}")
        commit = r.stdout.strip() if r.returncode == 0 else commit

    print(f"fetch_wasm: looking in {repo} on GitHub")
    try:
        found = None
        if commit and args.release is None:
            found = from_ci(api, repo, checkout, commit, token)
        if found is None and args.commit is None:
            found = from_release(api, repo, args.release or None)
            if found is not None and commit:
                print(f"fetch_wasm: the views come from {found[1]}, which may be older than this "
                      f"checkout ({commit[:12]}); `python scripts/build_wasm.py` builds this "
                      "checkout's own", file=sys.stderr)
    except Fail as e:
        print(f"fetch_wasm: {e}", file=sys.stderr)
        return 1
    if found is None:
        print("fetch_wasm: no build of the views was found. CI makes one for every change to their "
              "sources on main and in pull requests; until one exists, build them with "
              f"`python scripts/build_wasm.py` (needs Emscripten {wasm_views.pinned_emscripten()})",
              file=sys.stderr)
        return 1
    files, where = found
    folders = wasm_views.destinations(args.out)
    wasm_views.install(files, folders)
    for folder in folders:
        print(f"fetch_wasm: copied the views from {where} into {wasm_views.describe(folder)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
