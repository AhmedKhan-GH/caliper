#!/usr/bin/env python3
"""Generate or verify docs/reference/packages-and-dependencies.md from repository manifests.

Reads the five dependency sources of truth:
  - pyproject.toml
  - .gitmodules (and Git index / submodule commits under third_party/)
  - cmake/Dependencies.cmake
  - web/package.json
  - web/wasm/emscripten-version.txt

In --check mode (the default), exits with status 1 when any source cannot be
parsed, when a required pin is missing or unresolved, or when
docs/reference/packages-and-dependencies.md drifts from the generated output.

Usage:
  generate_packages_reference.py [--root <repo_root>] [--check | --write]
"""

from __future__ import annotations

import argparse
import configparser
import html
import json
import pathlib
import re
import subprocess
import sys
import tomllib

PYPROJECT_REL = "pyproject.toml"
GITMODULES_REL = ".gitmodules"
CMAKE_DEPS_REL = "cmake/Dependencies.cmake"
WEB_PACKAGE_REL = "web/package.json"
EMSCRIPTEN_VERSION_REL = "web/wasm/emscripten-version.txt"
OUTPUT_REL = "docs/reference/packages-and-dependencies.md"

SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[.-][A-Za-z0-9]+)*$")
SUBMODULE_SECTION_RE = re.compile(r'^submodule\s+"([^"]+)"$')
STAGE_GITLINK_RE = re.compile(r"^160000\s+([0-9a-f]{40})\s+\d+\t(.+)$")
TREE_GITLINK_RE = re.compile(r"^160000\s+commit\s+([0-9a-f]{40})\t(.+)$")
CMAKE_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
CMAKE_SET_RE = re.compile(
    r'^\s*set\(\s*([A-Za-z0-9_]+)\s+(?:"([^"]*)"|([^\s()"]+))(?:\s+CACHE\b.*)?\s*\)\s*$',
    re.IGNORECASE,
)
FETCH_START_RE = re.compile(r"\bFetchContent_Declare\s*\(", re.IGNORECASE)
TOKEN_RE = re.compile(r'"([^"]*)"|([^\s()]+)')

PEP508_URL_RE = re.compile(r"^([A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?)\s*@\s*(\S.*)$")
PEP508_SPEC_RE = re.compile(r"^([A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?)\s*([=<>!~].+)$")
PEP508_BARE_RE = re.compile(r"^([A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?)$")

EXTRA_ORDER = ("torch", "examples", "notebook", "interp", "agent", "test", "docs")

EXTRA_INFO = {
    "torch": (
        "Users",
        "PyTorch (`torch`) for tensor views and model hooks.",
        "Install when a script streams PyTorch tensors and does not already install PyTorch.",
    ),
    "examples": (
        "Users",
        "PyTorch (`torch`) and `torchvision` for the shipped training examples.",
        "Install when you run `examples/mnist/train.py` or any other shipped example.",
    ),
    "notebook": (
        "Users",
        "`anywidget` interactive notebook widgets.",
        "Install when you display Caliper views inside JupyterLab or notebook cells.",
    ),
    "interp": (
        "Users",
        "PyTorch, Hugging Face libraries, `transformer-lens`, `sae-lens`, `nnsight`, and `circuit-tracer`.",
        "Install when you run mechanistic interpretability captures, dictionary queries, or circuit traces.",
    ),
    "agent": (
        "Users",
        "`pydantic-ai-slim[anthropic]`, `anthropic`, and `openai` for the `caliper-agent` CLI.",
        "Install when you run the Caliper side chat agent outside a source checkout.",
    ),
    "test": (
        "**Contributors only**",
        "`pytest`, schema validators, `pyarrow`, agent test packages, and interpretability test packages.",
        "Install in a contributor virtual environment to run the repository test suites.",
    ),
    "docs": (
        "**Contributors only**",
        "Pinned MkDocs, Material theme, `mkdocstrings`, `ruff`, and `sternbild-docs`.",
        "Install directly into a separate documentation virtual environment to build and check the docs site.",
    ),
}

SUBMODULE_ROLES = {
    "third_party/glfw": "Window creation, input events, and graphics surface setup for `libcaliper`.",
    "third_party/glm": "Header-only vector and matrix mathematics for 3-D scenes and cameras.",
    "third_party/imgui": "Immediate-mode UI library (part of the ABI-pinned UI stack).",
    "third_party/implot": "2-D plotting library for ImGui (part of the ABI-pinned UI stack).",
    "third_party/ImGuiFileDialog": "File open and save dialog widget for ImGui (part of the ABI-pinned UI stack).",
    "third_party/implot3d": "3-D plotting library for ImGui (part of the ABI-pinned UI stack).",
    "third_party/imgui-node-editor": "Node-graph editor widget for ImGui.",
    "third_party/ImTerm": "Header-only terminal and console widget for ImGui.",
    "third_party/ImGuiColorTextEdit": "Syntax-highlighted text editor widget for ImGui.",
    "third_party/duckdb": "Embedded analytical SQL database with static Parquet, core functions, and optional JSON extensions.",
    "third_party/slang": "SystemVerilog compiler library, optional and gated by `CALIPER_WITH_SLANG`.",
}

FETCHCONTENT_META = {
    "tomlplusplus": (
        "All builds (host, `src/builder`, `web/wasm`)",
        "TOML parser for instrument manifests, piece manifests, and spec documents.",
    ),
    "nlohmann_json": (
        "`CALIPER_WITH_BUILDER=ON` and `web/wasm`",
        "JSON parser and serializer for specs, commands, MCP messages, and WebAssembly program metadata.",
    ),
    "doctest": (
        "C++ tests (`BUILD_TESTS=ON`) and `caliper-sdk`",
        "Header-only C++ unit test framework.",
    ),
    "sternbild_tokens": (
        "All builds (native host and `web/wasm`)",
        "Sternbild design tokens, theme color palette, and house font files.",
    ),
    "zlib": (
        "Windows only (`WIN32`)",
        "Static compression library for Windows instruments (macOS uses the system library).",
    ),
    "curl": (
        "Windows only (`WIN32`)",
        "Static HTTP client with Windows Schannel TLS for instruments (macOS uses the system library).",
    ),
    "vulkan-headers": (
        "Windows only (`WIN32`)",
        "Khronos Vulkan C API headers for the Windows Vulkan renderer.",
    ),
    "volk": (
        "Windows only (`WIN32`)",
        "Vulkan entry-point meta-loader on Windows.",
    ),
    "glslang": (
        "Windows only (`WIN32`)",
        "GLSL-to-SPIR-V shader compiler for the Windows Vulkan backend.",
    ),
}

FETCHCONTENT_KEYWORDS = frozenset(
    {
        "GIT_REPOSITORY",
        "GIT_TAG",
        "GIT_SHALLOW",
        "GIT_PROGRESS",
        "GIT_SUBMODULES",
        "GIT_SUBMODULES_RECURSE",
        "URL",
        "URL_HASH",
        "SOURCE_SUBDIR",
        "DOWNLOAD_EXTRACT_TIMESTAMP",
        "FIND_PACKAGE_ARGS",
        "OVERRIDE_FIND_PACKAGE",
        "SYSTEM",
    }
)

BRANCH_DEPENDENT_CMAKE_VARS = frozenset(
    {
        "PYTORCH_URL",
        "PYTORCH_VARIANT",
        "PYTORCH_VARIANT_NAME",
        "PYTORCH_BUILD_TYPE",
        "PYTORCH_PLATFORM",
        "LIBTORCH_DIR",
        "BUILD_EXTENSIONS",
    }
)


def md_cell(text: str) -> str:
    escaped = html.escape(" ".join(str(text).split()), quote=False)
    return escaped.replace("|", "\\|")


def md_code(text: str) -> str:
    cleaned = " ".join(str(text).split()).replace("|", "\\|")
    if "`" in cleaned:
        return "`` %s ``" % cleaned
    return "`%s`" % cleaned


def split_pep508(req: str) -> tuple[str, str]:
    raw = req.strip()
    m = PEP508_URL_RE.match(raw)
    if m:
        return m.group(1), m.group(2).strip()
    m = PEP508_SPEC_RE.match(raw)
    if m:
        return m.group(1), m.group(2).strip()
    m = PEP508_BARE_RE.match(raw)
    if m:
        return m.group(1), "Unpinned"
    return "", ""


def parse_pyproject(root: pathlib.Path) -> tuple[dict, list[str]]:
    path = root / PYPROJECT_REL
    if not path.is_file():
        return {}, ["%s: file does not exist" % PYPROJECT_REL]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {}, ["%s: invalid TOML: %s" % (PYPROJECT_REL, exc)]

    problems: list[str] = []
    project = data.get("project")
    if not isinstance(project, dict):
        return {}, ["%s: missing [project] table" % PYPROJECT_REL]

    name = project.get("name")
    version = project.get("version")
    requires_python = project.get("requires-python")
    dependencies = project.get("dependencies")
    scripts = project.get("scripts") or {}
    opt_deps = project.get("optional-dependencies")

    if not isinstance(name, str) or not name.strip():
        problems.append("%s: missing project.name" % PYPROJECT_REL)
    if not isinstance(version, str) or not version.strip():
        problems.append("%s: missing project.version" % PYPROJECT_REL)
    if not isinstance(requires_python, str) or not requires_python.strip():
        problems.append("%s: missing project.requires-python" % PYPROJECT_REL)
    if not isinstance(dependencies, list) or not dependencies:
        problems.append("%s: missing or empty project.dependencies" % PYPROJECT_REL)
    if not isinstance(scripts, dict):
        problems.append("%s: project.scripts must be a table" % PYPROJECT_REL)
    if not isinstance(opt_deps, dict) or not opt_deps:
        problems.append("%s: missing or empty [project.optional-dependencies]" % PYPROJECT_REL)

    parsed_base: list[tuple[str, str]] = []
    if isinstance(dependencies, list):
        for req in dependencies:
            if not isinstance(req, str) or not req.strip():
                problems.append("%s: invalid entry in project.dependencies" % PYPROJECT_REL)
                continue
            pkg, spec = split_pep508(req)
            if not pkg:
                problems.append("%s: unparseable dependency requirement %r" % (PYPROJECT_REL, req))
            else:
                parsed_base.append((pkg, spec))

    parsed_extras: dict[str, list[tuple[str, str]]] = {}
    if isinstance(opt_deps, dict):
        for extra_name, reqs in opt_deps.items():
            if not isinstance(reqs, list) or not reqs:
                problems.append("%s: optional dependency extra %r is empty" % (PYPROJECT_REL, extra_name))
                continue
            items: list[tuple[str, str]] = []
            for req in reqs:
                if not isinstance(req, str) or not req.strip():
                    problems.append("%s: invalid entry in extra %r" % (PYPROJECT_REL, extra_name))
                    continue
                pkg, spec = split_pep508(req)
                if not pkg:
                    problems.append(
                        "%s: unparseable requirement %r in extra %r" % (PYPROJECT_REL, req, extra_name)
                    )
                else:
                    items.append((pkg, spec))
            parsed_extras[str(extra_name)] = items

    parsed_scripts: dict[str, str] = {}
    if isinstance(scripts, dict):
        for cmd, target in scripts.items():
            if not isinstance(target, str) or not target.strip():
                problems.append("%s: invalid entry for script %r" % (PYPROJECT_REL, cmd))
            else:
                parsed_scripts[str(cmd)] = target.strip()

    parsed_build: list[tuple[str, str]] = []
    build_sys = data.get("build-system")
    if isinstance(build_sys, dict) and "requires" in build_sys:
        build_reqs = build_sys.get("requires")
        if not isinstance(build_reqs, list):
            problems.append("%s: build-system.requires must be a list" % PYPROJECT_REL)
        else:
            for req in build_reqs:
                if not isinstance(req, str) or not req.strip():
                    problems.append("%s: invalid entry in build-system.requires" % PYPROJECT_REL)
                    continue
                pkg, spec = split_pep508(req)
                if not pkg:
                    problems.append("%s: unparseable build requirement %r" % (PYPROJECT_REL, req))
                else:
                    parsed_build.append((pkg, spec))

    return {
        "name": str(name or "").strip(),
        "version": str(version or "").strip(),
        "requires_python": str(requires_python or "").strip(),
        "build_requires": parsed_build,
        "dependencies": parsed_base,
        "scripts": parsed_scripts,
        "extras": parsed_extras,
    }, problems


def _git_stage_gitlinks(root: pathlib.Path) -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--stage"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}
    if proc.returncode != 0:
        return {}
    commits: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        m = STAGE_GITLINK_RE.match(line)
        if m:
            commits[m.group(2)] = m.group(1)
    return commits


def _git_ls_tree_commit(root: pathlib.Path, submod_path: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "HEAD", "--", submod_path],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        m = TREE_GITLINK_RE.match(line)
        if m and m.group(2) == submod_path:
            return m.group(1)
    return None


def _read_gitdir_head(git_dir: pathlib.Path) -> str | None:
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text(encoding="utf-8").strip()
    if SHA40_RE.fullmatch(head):
        return head
    if head.startswith("ref:"):
        ref_name = head.split(":", 1)[1].strip()
        ref_file = git_dir / ref_name
        if ref_file.is_file():
            sha = ref_file.read_text(encoding="utf-8").strip()
            if SHA40_RE.fullmatch(sha):
                return sha
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] == ref_name and SHA40_RE.fullmatch(parts[0]):
                    return parts[0]
    return None


def _resolve_local_submodule_head(submod_dir: pathlib.Path) -> str | None:
    dot_git = submod_dir / ".git"
    if not dot_git.exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(submod_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            sha = proc.stdout.strip()
            if SHA40_RE.fullmatch(sha):
                return sha
    except OSError:
        pass
    if dot_git.is_dir():
        return _read_gitdir_head(dot_git)
    if dot_git.is_file():
        first = dot_git.read_text(encoding="utf-8").strip()
        if first.startswith("gitdir:"):
            target = first.split(":", 1)[1].strip()
            git_dir = (submod_dir / target).resolve()
            return _read_gitdir_head(git_dir)
    return None


def parse_gitmodules(
    root: pathlib.Path,
    submodule_commits: dict[str, str] | None = None,
) -> tuple[list[dict], list[str]]:
    path = root / GITMODULES_REL
    if not path.is_file():
        return [], ["%s: file does not exist" % GITMODULES_REL]

    cfg = configparser.ConfigParser(interpolation=None)
    try:
        cfg.read_string(path.read_text(encoding="utf-8"))
    except configparser.Error as exc:
        return [], ["%s: invalid gitmodules syntax: %s" % (GITMODULES_REL, exc)]

    stage_commits = {} if submodule_commits is not None else _git_stage_gitlinks(root)
    problems: list[str] = []
    submodules: list[dict] = []
    declared_paths: set[str] = set()

    for section in cfg.sections():
        m = SUBMODULE_SECTION_RE.match(section.strip())
        if not m:
            continue
        sub_path = cfg[section].get("path", "").strip()
        url = cfg[section].get("url", "").strip()
        branch = cfg[section].get("branch", "").strip()
        if not sub_path:
            problems.append("%s: section [%s] missing path" % (GITMODULES_REL, section))
            continue
        if not url:
            problems.append("%s: submodule %s missing url" % (GITMODULES_REL, sub_path))
            continue
        declared_paths.add(sub_path)

        commit: str | None = None
        if submodule_commits is not None and sub_path in submodule_commits:
            cand = submodule_commits[sub_path].strip()
            if SHA40_RE.fullmatch(cand):
                commit = cand
            else:
                problems.append(
                    "%s: invalid commit SHA %r for submodule %s" % (GITMODULES_REL, cand, sub_path)
                )
        else:
            commit = stage_commits.get(sub_path)
            if not commit:
                commit = _git_ls_tree_commit(root, sub_path)
            if not commit:
                commit = _resolve_local_submodule_head(root / sub_path)

        if not commit:
            problems.append(
                "%s: cannot resolve 40-char pinned commit for submodule %s" % (GITMODULES_REL, sub_path)
            )
            continue

        submodules.append(
            {
                "path": sub_path,
                "url": url,
                "branch": branch,
                "commit": commit,
                "role": SUBMODULE_ROLES.get(sub_path, "Third-party C++ Git submodule."),
            }
        )

    if not declared_paths:
        problems.append("%s: no submodules parsed" % GITMODULES_REL)

    if submodule_commits is None and stage_commits:
        undeclared = sorted(
            p for p in stage_commits if p.startswith("third_party/") and p not in declared_paths
        )
        if undeclared:
            problems.append(
                "%s: Git index has third_party gitlink(s) missing from .gitmodules: %s"
                % (GITMODULES_REL, ", ".join(undeclared))
            )

    return submodules, problems


def _strip_cmake_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        out_chars: list[str] = []
        in_quotes = False
        escaped = False
        for ch in line:
            if escaped:
                out_chars.append(ch)
                escaped = False
                continue
            if ch == "\\" and in_quotes:
                out_chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_quotes = not in_quotes
                out_chars.append(ch)
                continue
            if ch == "#" and not in_quotes:
                break
            out_chars.append(ch)
        cleaned_lines.append("".join(out_chars))
    return "\n".join(cleaned_lines)


def _expand_cmake_vars(value: str, variables: dict[str, str]) -> str:
    return CMAKE_VAR_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), value)


def _extract_url_pin(url: str) -> str | None:
    m_tag = re.search(r"/archive/refs/tags/([^/]+?)(?:\.tar\.(?:gz|xz|bz2)|\.zip)$", url)
    if m_tag:
        return m_tag.group(1)
    m_rel = re.search(r"/releases/download/([^/]+)/([^/]+)$", url)
    if m_rel:
        rel_tag, filename = m_rel.group(1), m_rel.group(2)
        m_fver = re.search(r"[-_]v?(\d+(?:\.\d+)+)\.(?:tar\.(?:gz|xz|bz2)|zip)$", filename)
        if m_fver:
            ver = m_fver.group(1)
            if rel_tag in (ver, "v" + ver):
                return rel_tag
            return "%s (tag `%s`)" % (ver, rel_tag)
        return rel_tag
    m_gen = re.search(r"[-_]v?(\d+(?:\.\d+)+)\.(?:tar\.(?:gz|xz|bz2)|zip)$", url)
    if m_gen:
        return m_gen.group(1)
    return None


def _classify_libtorch_url(url: str) -> str:
    if "libtorch-win-shared-with-deps-debug-" in url:
        return "Windows (Debug)"
    if "libtorch-win-shared-with-deps-" in url:
        return "Windows (Release)"
    if "libtorch-macos-arm64-" in url:
        return "macOS ARM64 (MPS)"
    if "libtorch-cxx11-abi-shared-with-deps-" in url and "${PYTORCH_VARIANT}" in url:
        return "Linux (CUDA)"
    if "libtorch-cxx11-abi-shared-with-deps-" in url and ("%2Bcpu" in url or "+cpu" in url):
        return "Linux (CPU)"
    return "LibTorch archive"


def _iter_fetchcontent_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    pos = 0
    while True:
        m = FETCH_START_RE.search(text, pos)
        if not m:
            break
        start = m.end()
        depth = 1
        idx = start
        in_quotes = False
        while idx < len(text) and depth > 0:
            ch = text[idx]
            if ch == '"':
                in_quotes = not in_quotes
            elif not in_quotes:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            idx += 1
        if depth == 0:
            blocks.append(text[start : idx - 1])
        pos = idx
    return blocks


def parse_cmake_dependencies(root: pathlib.Path) -> tuple[dict, list[str]]:
    path = root / CMAKE_DEPS_REL
    if not path.is_file():
        return {}, ["%s: file does not exist" % CMAKE_DEPS_REL]

    raw = path.read_text(encoding="utf-8")
    cleaned = _strip_cmake_comments(raw)
    problems: list[str] = []

    variables: dict[str, str] = {}
    pytorch_versions: list[str] = []
    pytorch_variants: list[str] = []
    pytorch_urls_raw: list[str] = []

    for line in cleaned.splitlines():
        m = CMAKE_SET_RE.match(line)
        if not m:
            continue
        var = m.group(1)
        val = m.group(2) if m.group(2) is not None else m.group(3)
        if var == "PYTORCH_VERSION":
            if val:
                pytorch_versions.append(val)
            continue
        if var == "PYTORCH_VARIANT":
            if val and val not in pytorch_variants:
                pytorch_variants.append(val)
            continue
        if var == "PYTORCH_URL":
            if val:
                pytorch_urls_raw.append(val)
            continue
        if var not in BRANCH_DEPENDENT_CMAKE_VARS:
            variables[var] = val

    pytorch_version = ""
    if not pytorch_versions:
        problems.append("%s: PYTORCH_VERSION not found" % CMAKE_DEPS_REL)
    elif len(set(pytorch_versions)) > 1:
        problems.append(
            "%s: conflicting PYTORCH_VERSION values: %s"
            % (CMAKE_DEPS_REL, ", ".join(sorted(set(pytorch_versions))))
        )
    else:
        pytorch_version = pytorch_versions[0]
        variables["PYTORCH_VERSION"] = pytorch_version

    libtorch_archives: list[dict] = []
    for raw_url in pytorch_urls_raw:
        expanded = _expand_cmake_vars(raw_url, variables)
        unresolved = [v for v in CMAKE_VAR_RE.findall(expanded) if v != "PYTORCH_VARIANT"]
        if unresolved:
            problems.append(
                "%s: unresolved variable(s) %s in PYTORCH_URL %r"
                % (CMAKE_DEPS_REL, ", ".join(unresolved), raw_url)
            )
            continue
        libtorch_archives.append(
            {
                "platform": _classify_libtorch_url(expanded),
                "url": expanded,
            }
        )

    if not libtorch_archives:
        problems.append("%s: no PYTORCH_URL archive entries parsed" % CMAKE_DEPS_REL)

    fetch_items: list[dict] = []
    for block in _iter_fetchcontent_blocks(cleaned):
        tokens = [t[0] if t[0] != "" else t[1] for t in TOKEN_RE.findall(block)]
        if not tokens:
            continue
        name = tokens[0]
        kv: dict[str, str] = {}
        idx = 1
        while idx < len(tokens):
            tok = tokens[idx]
            if tok in FETCHCONTENT_KEYWORDS and idx + 1 < len(tokens):
                kv[tok] = tokens[idx + 1]
                idx += 2
            else:
                idx += 1

        expanded_kv: dict[str, str] = {}
        has_unresolved = False
        for k, v in kv.items():
            ev = _expand_cmake_vars(v, variables)
            if CMAKE_VAR_RE.search(ev):
                problems.append(
                    "%s: unresolved CMake variable in FetchContent_Declare(%s) %s=%r"
                    % (CMAKE_DEPS_REL, name, k, v)
                )
                has_unresolved = True
            expanded_kv[k] = ev
        if has_unresolved:
            continue

        git_repo = expanded_kv.get("GIT_REPOSITORY", "")
        git_tag = expanded_kv.get("GIT_TAG", "")
        url = expanded_kv.get("URL", "")
        url_hash = expanded_kv.get("URL_HASH", "")
        source_subdir = expanded_kv.get("SOURCE_SUBDIR", "")

        pin_md = ""
        source_url = ""

        if git_repo:
            source_url = git_repo
            if not git_tag:
                problems.append("%s: FetchContent_Declare(%s) missing GIT_TAG" % (CMAKE_DEPS_REL, name))
                continue
            if name == "sternbild_tokens":
                tag = variables.get("STERNBILD_TOKENS_TAG", "")
                commit = variables.get("STERNBILD_TOKENS_COMMIT", "")
                if not tag or not SHA40_RE.fullmatch(commit):
                    problems.append(
                        "%s: sternbild_tokens requires STERNBILD_TOKENS_TAG and 40-char STERNBILD_TOKENS_COMMIT"
                        % CMAKE_DEPS_REL
                    )
                    continue
                pin_md = "%s (commit %s)" % (md_code(tag), md_code(commit))
            else:
                pin_md = md_code(git_tag)
        elif url:
            source_url = url
            extracted = _extract_url_pin(url)
            if not extracted:
                problems.append(
                    "%s: cannot parse version or tag pin from URL in FetchContent_Declare(%s): %s"
                    % (CMAKE_DEPS_REL, name, url)
                )
                continue
            if "(tag `" in extracted:
                ver_part, tag_rest = extracted.split(" (tag ", 1)
                pin_md = "%s (tag %s" % (md_code(ver_part), tag_rest)
            else:
                pin_md = md_code(extracted)
            if url_hash:
                pin_md = "%s (%s)" % (pin_md, md_code(url_hash))
        else:
            problems.append(
                "%s: FetchContent_Declare(%s) has neither GIT_REPOSITORY nor URL" % (CMAKE_DEPS_REL, name)
            )
            continue

        scope, role = FETCHCONTENT_META.get(
            name,
            ("CMake `FetchContent`", "External C++ library declared in `cmake/Dependencies.cmake`."),
        )
        fetch_items.append(
            {
                "name": name,
                "pin_md": pin_md,
                "source_url": source_url,
                "source_subdir": source_subdir,
                "scope": scope,
                "role": role,
            }
        )

    if not fetch_items:
        problems.append("%s: no FetchContent_Declare entries parsed" % CMAKE_DEPS_REL)

    return {
        "fetch_content": fetch_items,
        "pytorch_version": pytorch_version,
        "pytorch_variants": pytorch_variants,
        "libtorch_archives": libtorch_archives,
    }, problems


def parse_web_package(root: pathlib.Path) -> tuple[dict, list[str]]:
    path = root / WEB_PACKAGE_REL
    if not path.is_file():
        return {}, ["%s: file does not exist" % WEB_PACKAGE_REL]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, ["%s: invalid JSON: %s" % (WEB_PACKAGE_REL, exc)]

    problems: list[str] = []
    if not isinstance(data, dict):
        return {}, ["%s: expected top-level JSON object" % WEB_PACKAGE_REL]

    name = data.get("name")
    version = data.get("version")
    deps = data.get("dependencies") or {}
    dev_deps = data.get("devDependencies")

    if not isinstance(name, str) or not name.strip():
        problems.append("%s: missing package name" % WEB_PACKAGE_REL)
    if not isinstance(version, str) or not version.strip():
        problems.append("%s: missing package version" % WEB_PACKAGE_REL)
    if not isinstance(deps, dict):
        problems.append("%s: dependencies must be an object" % WEB_PACKAGE_REL)
    if not isinstance(dev_deps, dict) or not dev_deps:
        problems.append("%s: missing or empty devDependencies" % WEB_PACKAGE_REL)

    clean_deps: dict[str, str] = {}
    if isinstance(deps, dict):
        for k, v in deps.items():
            if not isinstance(v, str) or not v.strip():
                problems.append("%s: missing version for dependency %r" % (WEB_PACKAGE_REL, k))
            else:
                clean_deps[str(k)] = v.strip()

    clean_dev: dict[str, str] = {}
    if isinstance(dev_deps, dict):
        for k, v in dev_deps.items():
            if not isinstance(v, str) or not v.strip():
                problems.append("%s: missing version for devDependency %r" % (WEB_PACKAGE_REL, k))
            else:
                clean_dev[str(k)] = v.strip()

    return {
        "name": str(name or "").strip(),
        "version": str(version or "").strip(),
        "dependencies": clean_deps,
        "dev_dependencies": clean_dev,
    }, problems


def parse_emscripten_version(root: pathlib.Path) -> tuple[str, list[str]]:
    path = root / EMSCRIPTEN_VERSION_REL
    if not path.is_file():
        return "", ["%s: file does not exist" % EMSCRIPTEN_VERSION_REL]
    ver = path.read_text(encoding="utf-8").strip()
    if not ver or not VERSION_RE.fullmatch(ver):
        return "", ["%s: invalid Emscripten version %r" % (EMSCRIPTEN_VERSION_REL, ver)]
    return ver, []


def load_all(
    root: pathlib.Path,
    submodule_commits: dict[str, str] | None = None,
) -> tuple[dict, list[str]]:
    problems: list[str] = []
    py_data, p1 = parse_pyproject(root)
    submodules, p2 = parse_gitmodules(root, submodule_commits=submodule_commits)
    cmake_data, p3 = parse_cmake_dependencies(root)
    web_data, p4 = parse_web_package(root)
    emsdk_ver, p5 = parse_emscripten_version(root)
    problems.extend(p1)
    problems.extend(p2)
    problems.extend(p3)
    problems.extend(p4)
    problems.extend(p5)
    return {
        "python": py_data,
        "submodules": submodules,
        "cmake": cmake_data,
        "web": web_data,
        "emscripten_version": emsdk_ver,
    }, problems


def _ordered_extras(extras: dict[str, list[tuple[str, str]]]) -> list[str]:
    ordered = [e for e in EXTRA_ORDER if e in extras]
    ordered.extend(sorted(e for e in extras if e not in EXTRA_ORDER))
    return ordered


def render_markdown(parsed: dict) -> str:
    py = parsed["python"]
    submodules = parsed["submodules"]
    cmake = parsed["cmake"]
    web = parsed["web"]
    emsdk_ver = parsed["emscripten_version"]

    py_name = py["name"]
    py_ver = py["version"]
    py_req = py["requires_python"]
    web_name = web["name"]
    web_ver = web["version"]
    ordered_extras = _ordered_extras(py["extras"])

    scripts_entries = ", ".join(
        "%s (%s)" % (md_code(cmd), md_code(target)) for cmd, target in sorted(py["scripts"].items())
    )
    if scripts_entries:
        scripts_sentence = (
            "The `[project.scripts]` table exposes %s, which runs with the `%s[agent]` extra."
            % (scripts_entries, py_name)
        )
        script_names = ", ".join(md_code(cmd) for cmd in sorted(py["scripts"]))
        py_interfaces = (
            "`caliper._native`, `caliper._live`, `caliper._netio`, `caliper._producer`, `caliper-viewer`, %s"
            % script_names
        )
    else:
        scripts_sentence = "The package declares no console entry points in `[project.scripts]`."
        py_interfaces = (
            "`caliper._native`, `caliper._live`, `caliper._netio`, `caliper._producer`, `caliper-viewer`"
        )

    lines: list[str] = [
        "<!-- Generated by scripts/generate_packages_reference.py. Do not edit by hand. -->",
        "",
        "# Packages and Dependencies",
        "",
        "This page lists every package that Caliper ships and every external dependency with its version or commit pin.",
        "The script `scripts/generate_packages_reference.py` generates this page from `pyproject.toml`, `.gitmodules`, `cmake/Dependencies.cmake`, `web/package.json`, and `web/wasm/emscripten-version.txt`.",
        "Continuous integration runs `python scripts/generate_packages_reference.py --check` and fails when this page drifts from those five sources.",
        "Run `python3 scripts/generate_packages_reference.py --write` after any edit to those five files.",
        "",
        "## Shipped Packages",
        "",
        "Caliper builds five deliverables from this repository for desktop users, Python developers, browser pages, native hosts, and instrument authors.",
        "The **Caliper Application** builds the native `caliper` desktop host and the `caliper-mcp` server from the C++ core.",
        "The `caliper` executable opens the desktop workspace, runs [Caliper CLI](cli.md) subcommands, and opens instruments.",
        "The `caliper-mcp` executable ships beside `caliper` and exposes the tools in [MCP Command Reference](mcp-commands.md) over stdio.",
        "",
        "The **Python Package** (%s %s, Python %s) ships Caliper LIB and the `caliper_agent` package from `pyproject.toml`."
        % (md_code(py_name), md_code(py_ver), md_code(py_req)),
        "The wheel bundles four native extensions (`caliper._native`, `caliper._live`, `caliper._netio`, and `caliper._producer`), the `caliper-viewer` executable, and the WebAssembly views.",
        scripts_sentence,
        "See [Caliper LIB API](../python/caliper-lib-api.md) for Python view calls and [Viewer](../python/viewer.md) for `caliper-viewer` options.",
        "",
        "The **Browser Module** (%s %s from `web/package.json`) supplies `<caliper-view>`, `<caliper-controls>`, and `<caliper-replay>` custom elements."
        % (md_code(web_name), md_code(web_ver)),
        "When a browser provides WebGPU, those elements render with `caliper-views.wasm`, compiled with Emscripten %s from `web/wasm/emscripten-version.txt`."
        % md_code(emsdk_ver),
        "**Caliper Embed** (`caliper-embed`) installs the shared library (`libcaliper_embed.dylib` on macOS and `caliper_embed.dll` on Windows) behind the C interface in `include/caliper/embed.h`.",
        "**Instrument SDK** (`caliper-sdk`) installs `sdk/include/caliper/abi.h`, `sdk/include/caliper/caliper.hpp`, `caliper_add_instrument()`, the fixture host, and the pinned UI stack for out-of-tree C++ instruments.",
        "See [Browser Elements](../web/browser-elements.md), [Program Embedding](../web/program-embedding.md), and [SDK Package](../instruments/sdk-package.md) for integration details.",
        "",
        "### Shipped Package Summary",
        "",
        "Each deliverable below maps to its package name, version or pin, and primary entry point.",
        "",
        "| Deliverable | Package or Artifact | Version or Pin | Primary Interface |",
        "|---|---|---|---|",
        "| Caliper Application | `caliper`, `caliper-mcp` | Repository C++ core | Desktop UI, [Caliper CLI](cli.md), [MCP Command Reference](mcp-commands.md) |",
        "| Python Package | %s | %s (Python %s) | %s |"
        % (md_code(py_name), md_code(py_ver), md_code(py_req), py_interfaces),
        "| Browser Module | %s, `caliper-views.wasm` | %s (Emscripten %s) | `<caliper-view>`, `<caliper-controls>`, `<caliper-replay>` |"
        % (md_code(web_name), md_code(web_ver), md_code(emsdk_ver)),
        "| Caliper Embed | `caliper-embed` (`libcaliper_embed.dylib`, `caliper_embed.dll`) | CMake package `caliper::embed` | `include/caliper/embed.h` |",
        "| Instrument SDK | `caliper-sdk` | CMake package `caliper::sdk` | `sdk/include/caliper/abi.h`, `sdk/include/caliper/caliper.hpp`, `caliper_add_instrument()`, pinned UI stack |",
        "",
        "### Python Package Extras",
        "",
        "The `[project.optional-dependencies]` table in `pyproject.toml` splits optional Python packages by workflow.",
        "User extras add PyTorch, the shipped examples, notebooks, interpretability tooling, or the side chat agent.",
        "The `[test]` and `[docs]` extras are for contributors only and never ship as user runtime requirements.",
        "",
        "| Extra | Audience | What It Adds | When to Install |",
        "|---|---|---|---|",
    ]

    for extra in ordered_extras:
        audience, adds, when = EXTRA_INFO.get(
            extra,
            (
                "Users",
                "Optional packages in `[project.optional-dependencies].%s`." % extra,
                "Install when your workflow uses the `%s` extra." % extra,
            ),
        )
        lines.append(
            "| %s | %s | %s | %s |"
            % (md_code("%s[%s]" % (py_name, extra)), audience, adds, when)
        )
    lines.append("")

    lines.extend(
        [
            "## External Dependencies",
            "",
            "Caliper declares every external dependency across five source files in this repository.",
            "The sections below list each dependency, its exact version or commit pin, and its scope.",
            "",
            "### Python Dependencies",
            "",
            "The root `pyproject.toml` file declares Python requirements for %s %s (Python %s), wheel build tools, and every optional extra."
            % (md_code(py_name), md_code(py_ver), md_code(py_req)),
            "Base runtime dependencies install with the package, while optional groups install through `%s[<extra>]`."
            % py_name,
            "The `[test]` and `[docs]` groups are for contributors only.",
            "",
            "| Group | Package | Version or Pin |",
            "|---|---|---|",
        ]
    )

    for pkg, spec in py["build_requires"]:
        pin_cell = "Unpinned" if spec == "Unpinned" else md_code(spec)
        lines.append("| Build backend (`[build-system]`) | %s | %s |" % (md_code(pkg), pin_cell))
    for pkg, spec in py["dependencies"]:
        pin_cell = "Unpinned" if spec == "Unpinned" else md_code(spec)
        lines.append("| Base runtime | %s | %s |" % (md_code(pkg), pin_cell))
    for extra in ordered_extras:
        if extra in ("test", "docs"):
            group_label = "%s (contributors only)" % md_code("[%s]" % extra)
        else:
            group_label = md_code("[%s]" % extra)
        for pkg, spec in py["extras"][extra]:
            pin_cell = "Unpinned" if spec == "Unpinned" else md_code(spec)
            lines.append("| %s | %s | %s |" % (group_label, md_code(pkg), pin_cell))
    lines.append("")

    lines.extend(
        [
            "### C++ Submodule Dependencies",
            "",
            "Caliper tracks third-party C++ source trees as Git submodules in `.gitmodules` under `third_party/`.",
            "Each submodule pins an exact 40-character Git commit in the repository index.",
            "The UI stack submodules (`imgui`, `implot`, `implot3d`, and `ImGuiFileDialog`) form part of the frozen instrument ABI epoch.",
            "",
            "| Submodule | Pinned Commit | Repository | Role |",
            "|---|---|---|---|",
        ]
    )

    for sub in submodules:
        pin_str = md_code(sub["commit"])
        if sub["branch"]:
            pin_str = "%s (branch %s)" % (pin_str, md_code(sub["branch"]))
        lines.append(
            "| %s | %s | %s | %s |"
            % (md_code(sub["path"]), pin_str, md_code(sub["url"]), sub["role"])
        )
    lines.append("")

    lines.extend(
        [
            "### C++ Downloaded Dependencies",
            "",
            "The CMake file `cmake/Dependencies.cmake` is the single source of truth for every `FetchContent_Declare` package and for pre-built LibTorch archives.",
            "Standalone sub-builds (`src/builder`, `python/netio`, and `web/wasm`) include `cmake/Dependencies.cmake` with `CALIPER_DEPENDENCIES_DECLARATIONS_ONLY=ON` to share the same pins.",
            "Python wheel builds set `CALIPER_WITH_BUILDER=OFF` and `CALIPER_WITH_TORCH=OFF`, thus they download neither `nlohmann_json` nor LibTorch.",
            "",
            "| Dependency | Pin | Scope | Source | Role |",
            "|---|---|---|---|---|",
        ]
    )

    for item in cmake["fetch_content"]:
        src_cell = md_code(item["source_url"])
        if item["source_subdir"]:
            src_cell = "%s (%s)" % (src_cell, md_code(item["source_subdir"]))
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (md_code(item["name"]), item["pin_md"], item["scope"], src_cell, item["role"])
        )
    lines.append("")

    variants_list = ", ".join(md_code(v) for v in cmake["pytorch_variants"])
    if variants_list:
        variants_sentence = "Supported LibTorch variants are %s." % variants_list
    else:
        variants_sentence = "The platform branch selects the matching CPU or CUDA archive."

    lines.extend(
        [
            "When `CALIPER_WITH_TORCH=ON`, `cmake/Dependencies.cmake` downloads pre-built LibTorch %s archives for instrument and torch test targets."
            % md_code(cmake["pytorch_version"]),
            "`libcaliper`, `caliper`, `caliper-embed`, and the Python wheel never link LibTorch.",
            variants_sentence,
            "",
            "| Platform | Version | Scope | Archive URL |",
            "|---|---|---|---|",
        ]
    )
    for arc in cmake["libtorch_archives"]:
        lines.append(
            "| %s | %s | `CALIPER_WITH_TORCH=ON` | %s |"
            % (md_cell(arc["platform"]), md_code(cmake["pytorch_version"]), md_code(arc["url"]))
        )
    lines.append("")

    if web["dependencies"]:
        web_intro = (
            "The browser package %s (%s) in `web/package.json` declares runtime packages in `dependencies` and build tools in `devDependencies`."
            % (md_code(web_name), md_code(web_ver))
        )
    else:
        web_intro = (
            "The browser package %s (%s) in `web/package.json` ships with zero runtime packages and pins its build and test tools in `devDependencies`."
            % (md_code(web_name), md_code(web_ver))
        )

    lines.extend(
        [
            "### Browser and WebAssembly Dependencies",
            "",
            web_intro,
            "The WebAssembly views in `web/wasm` compile with Emscripten %s, pinned in `web/wasm/emscripten-version.txt`."
            % md_code(emsdk_ver),
            "",
            "| Package or Toolchain | Scope | Version | Source |",
            "|---|---|---|---|",
            "| `emscripten` | WebAssembly toolchain | %s | `web/wasm/emscripten-version.txt` |"
            % md_code(emsdk_ver),
        ]
    )

    for pkg, ver in web["dependencies"].items():
        lines.append("| %s | Runtime dependency | %s | `web/package.json` |" % (md_code(pkg), md_code(ver)))
    for pkg, ver in web["dev_dependencies"].items():
        lines.append(
            "| %s | Development dependency | %s | `web/package.json` |" % (md_code(pkg), md_code(ver))
        )
    lines.append("")

    return "\n".join(lines)


def run(
    root: pathlib.Path,
    write: bool,
    submodule_commits: dict[str, str] | None = None,
) -> tuple[list[str], str]:
    parsed, problems = load_all(root, submodule_commits=submodule_commits)
    if problems:
        return problems, ""
    rendered = render_markdown(parsed)
    out_path = root / OUTPUT_REL
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        return [], rendered
    if not out_path.is_file():
        return [
            "%s: generated file does not exist (run scripts/generate_packages_reference.py --write)"
            % OUTPUT_REL
        ], rendered
    if out_path.read_text(encoding="utf-8") != rendered:
        return [
            "%s: out of date (run scripts/generate_packages_reference.py --write)" % OUTPUT_REL
        ], rendered
    return [], rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify docs/reference/packages-and-dependencies.md is up to date",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="write docs/reference/packages-and-dependencies.md",
    )
    args = parser.parse_args(argv)

    problems, _ = run(args.root.resolve(), write=args.write)
    if problems:
        for p in problems:
            print(p)
        print("%d problem(s) in packages and dependencies reference" % len(problems))
        return 1
    if args.write:
        print("Wrote %s" % OUTPUT_REL)
    else:
        print("Packages and dependencies reference complete and up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
