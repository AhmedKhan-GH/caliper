#!/usr/bin/env python3
"""Fail when documentation names a broken link, stale path, or unknown command or symbol.

Scans docs/**/*.md (excluding mkdocs.yml exclude_docs), CONTRIBUTING.md, and
README.md outside fenced code blocks and fails when a page contains:

  1. a broken relative Markdown link or missing #anchor (including generated
     roadmap and C API reference anchors);
  2. an inline-code repository file or directory path that no longer exists;
  3. an inline-code command or refusal code `<group>.<name>` in a command-table
     group that does not exist in command_table.json or the codebase;
  4. an inline-code `caliper <subcommand>` CLI invocation whose subcommand is
     not parsed by src/host/host_cli.h;
  5. an inline-code `caliper.<symbol>`, `caliper.live.<symbol>`, or
     `caliper.interp.<symbol>` Python symbol that does not exist.

Usage:
  check_doc_references.py [--root <repo_root>]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import pathlib
import re
import sys

import yaml
from markdown.extensions.toc import slugify

HOOK_REL = "docs-theme/hooks/c_reference.py"
COMMAND_TABLE_REL = "src/builder/commands/command_table.json"
HOST_CLI_REL = "src/host/host_cli.h"

FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
ATX_RE = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
EXPLICIT_ANCHOR_RE = re.compile(r"\{#([A-Za-z0-9_.:-]+)\}")
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

REPO_PREFIXES = (
    "src/",
    "include/",
    "sdk/",
    "examples/",
    "python/",
    "web/",
    "tools/",
    "tests/",
    "scripts/",
    "roadmap/",
    "candidates/",
    "cmake/",
    "docs/",
    "docs-theme/",
    "site/",
    "design/",
    ".github/",
)
ROOT_FILES = frozenset(
    {
        "CMakeLists.txt",
        "CMakePresets.json",
        "pyproject.toml",
        "mkdocs.yml",
        "README.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "init-submodules.sh",
        "init-submodules.bat",
    }
)

# Documented installed-package, build-artifact, or workspace example paths.
EXEMPT_PATH_PREFIXES = (
    "include/caliper-embed",
    "include/caliper-testing",
    "third_party/libtorch",
    "tools/list",
    "tools/call",
)
FILE_EXTS = frozenset(
    {
        "c", "cc", "cpp", "cu", "h", "hpp", "mm", "py", "ts", "js", "mjs", "json",
        "toml", "yml", "yaml", "md", "wgsl", "glsl", "spv", "ptx", "comp", "vert",
        "frag", "html", "css", "svg", "png", "woff", "woff2", "bin", "npy", "npz",
        "parquet", "csv", "txt", "sh", "bat", "cmake", "rs", "lock", "in", "inc",
    }
)
DOCUMENTED_DOTTED_FIELDS = frozenset(
    {
        "data.cursor",
        "data.code",
        "data.v1",
        "panel.annotation",
        "panel.entity",
        "instrument.spec",
    }
)
NATIVE_PYTHON_ATTRS = frozenset(
    {
        "__version__",
        "__all__",
        "HAVE_VIEWS",
        "HAVE_LIVE_VIEWS",
        "VIEW_FLAG_RAW",
        "VIEW_FLAG_WAIT_GPU",
        "EMBED_API_VERSION",
        "HEATMAP_IMAGE_VIA_LIVE_APPLY",
        "_netio",
        "_native",
        "_live",
        "_producer",
        "live",
        "interp",
        "loader_host",
    }
)


def load_c_hook(root: pathlib.Path):
    hook_path = root / HOOK_REL
    if not hook_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("c_reference_chk", hook_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("c_reference_chk", module)
    spec.loader.exec_module(module)
    return module


class _MkDocsSafeLoader(yaml.SafeLoader):
    pass


_MkDocsSafeLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def excluded_docs(root: pathlib.Path) -> set[str]:
    mkdocs_path = root / "mkdocs.yml"
    if not mkdocs_path.is_file():
        return set()
    data = yaml.load(mkdocs_path.read_text(encoding="utf-8"), Loader=_MkDocsSafeLoader) or {}
    raw = data.get("exclude_docs") or ""
    return {
        "docs/" + line.strip().lstrip("/")
        for line in str(raw).splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def outside_fences(text: str) -> list[tuple[int, str]]:
    """Return [(line_no, line)] outside frontmatter and fenced code blocks."""
    lines = text.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() in ("---", "..."):
                start = idx + 1
                break
    out: list[tuple[int, str]] = []
    fence = None
    for idx in range(start, len(lines)):
        line = lines[idx]
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0] * len(m.group(1))
            if fence is None:
                fence = marker[0]
            elif line.lstrip().startswith(fence * 3):
                fence = None
            continue
        if fence is None:
            out.append((idx + 1, line))
    return out


def strip_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub("", line)


def heading_text(raw: str) -> str:
    text = re.sub(r"(?<!`)<[A-Za-z/][^<>`]*>(?!`)", "", raw)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.strip()


def expand_c_api(root: pathlib.Path, hook, text: str, header_cache: dict) -> str:
    if hook is None or "<!-- c-api " not in text:
        return text

    def repl(match):
        raw_line = match.group(0)
        directives = hook.find_directives(raw_line)
        if not directives:
            return raw_line
        d = directives[0]
        try:
            if d.header not in header_cache:
                header_cache[d.header] = hook.parse_header(str(root / d.header))
            return hook.render(header_cache[d.header], d, set())
        except Exception:
            return raw_line

    return hook.DIRECTIVE.sub(repl, text)


def page_anchors(root: pathlib.Path, hook, header_cache: dict, path: pathlib.Path) -> set[str] | None:
    if not path.is_file():
        return None
    raw = expand_c_api(root, hook, path.read_text(encoding="utf-8"), header_cache)
    anchors: set[str] = set()
    seen: set[str] = set()
    for _, line in outside_fences(raw):
        for exp in EXPLICIT_ANCHOR_RE.finditer(line):
            anchors.add(exp.group(1))
        hm = ATX_RE.match(line)
        if not hm:
            continue
        htext = heading_text(hm.group(2))
        slug = base = slugify(htext, "-")
        suffix = 0
        while slug in seen:
            suffix += 1
            slug = "%s_%d" % (base, suffix)
        seen.add(slug)
        anchors.add(slug)
    return anchors


def virtual_docs_pages(root: pathlib.Path) -> set[str]:
    virt = {"docs/contents.md", "docs/roadmap/index.md"}
    rm_dir = root / "roadmap"
    if rm_dir.is_dir():
        for p in rm_dir.glob("*.yml"):
            virt.add("docs/roadmap/%s.md" % p.stem)
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            for feat in data.get("features") or []:
                if isinstance(feat, dict) and "id" in feat:
                    virt.add("docs/roadmap/%s.md" % str(feat["id"]).lower())
    cand_dir = root / "candidates"
    if cand_dir.is_dir():
        virt.add("docs/roadmap/candidates/index.md")
        for p in cand_dir.glob("*.yml"):
            virt.add("docs/roadmap/candidates/%s.md" % p.stem)
    return virt


def load_known_commands_and_codes(root: pathlib.Path) -> tuple[set[str], set[str]]:
    ct_path = root / COMMAND_TABLE_REL
    groups: set[str] = set()
    known: set[str] = set(DOCUMENTED_DOTTED_FIELDS)
    if ct_path.is_file():
        data = json.loads(ct_path.read_text(encoding="utf-8"))
        for cmd in data.get("commands") or []:
            name = cmd["name"]
            known.add(name)
            groups.add(name.split(".", 1)[0])
    dotted_str_re = re.compile(r'"([a-z][a-z0-9_]*\.[a-z][a-z0-9_.]*)"|\'([a-z][a-z0-9_]*\.[a-z][a-z0-9_.]*)\'')
    for folder in ("src", "include", "sdk", "python", "web/src"):
        base = root / folder
        if not base.is_dir():
            continue
        for fpath in base.rglob("*"):
            if not fpath.is_file() or fpath.suffix.lstrip(".") not in {"h", "hpp", "cpp", "mm", "py", "ts", "json"}:
                continue
            if "__pycache__" in fpath.parts or "node_modules" in fpath.parts:
                continue
            try:
                txt = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in dotted_str_re.finditer(txt):
                tok = m.group(1) or m.group(2)
                if tok.split(".", 1)[0] in groups:
                    known.add(tok)
    return groups, known


def load_cli_subcommands(root: pathlib.Path) -> set[str]:
    cli_path = root / HOST_CLI_REL
    if not cli_path.is_file():
        return set()
    txt = cli_path.read_text(encoding="utf-8")
    return set(re.findall(r'\barg\s*==\s*"([a-z][a-z0-9_-]*)"', txt))


def ast_exports(py_file: pathlib.Path) -> set[str]:
    if not py_file.is_file():
        return set()
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                    if t.id == "__all__" and isinstance(node, ast.Assign):
                        try:
                            names.update(ast.literal_eval(node.value))
                        except Exception:
                            pass
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            try:
                names.update(ast.literal_eval(node.value))
            except Exception:
                pass
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.If):
            for sub in node.body:
                if isinstance(sub, ast.ImportFrom):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name)
    return names


def load_python_symbols(root: pathlib.Path) -> tuple[dict[str, set[str]], set[str]]:
    pkg = root / "python/caliper"
    top = set(NATIVE_PYTHON_ATTRS) | ast_exports(pkg / "__init__.py")
    modules: dict[str, set[str]] = {}
    if pkg.is_dir():
        for py in pkg.glob("*.py"):
            if py.name != "__init__.py":
                top.add(py.stem)
                modules["caliper." + py.stem] = ast_exports(py)
        interp_dir = pkg / "interp"
        if interp_dir.is_dir():
            modules["caliper.interp"] = ast_exports(interp_dir / "__init__.py")
            for py in interp_dir.glob("*.py"):
                if py.name != "__init__.py":
                    modules["caliper.interp"].add(py.stem)
                    modules["caliper.interp." + py.stem] = ast_exports(py)

    service_ids: set[str] = set()
    svc_dir = root / "sdk/include/caliper/services"
    if svc_dir.is_dir():
        for h in svc_dir.glob("*.h"):
            for m in re.finditer(r'"(caliper\.([a-z0-9_]+)\.[a-z0-9_.]+)"', h.read_text(encoding="utf-8")):
                service_ids.add(m.group(1))
                service_ids.add("caliper." + m.group(2))
    if pkg.is_dir():
        for py in pkg.rglob("*.py"):
            for m in re.finditer(r'["\'](caliper\.[a-z][a-z0-9_.]*)["\']', py.read_text(encoding="utf-8")):
                service_ids.add(m.group(1))
    modules["caliper"] = top
    return modules, service_ids


def check_repo_path(root: pathlib.Path, token: str) -> str | None:
    cand = token.strip()
    if " " in cand or "\t" in cand:
        return None
    cand = re.split(r"[:#]", cand, maxsplit=1)[0]
    if not (cand in ROOT_FILES or cand.startswith(REPO_PREFIXES)):
        return None
    if any(ch in cand for ch in ("*", "?", "<", ">", "{", "}", "$", "|")):
        return None
    if "..." in cand:
        return None
    if any(cand.startswith(p) for p in EXEMPT_PATH_PREFIXES):
        return None
    if "/target/" in cand or "/.venv" in cand or cand.endswith("-captures"):
        return None
    if not (root / cand).exists():
        return "stale path `%s` does not exist" % cand
    return None


def check(root: pathlib.Path) -> list[str]:
    root = root.resolve()
    problems: list[str] = []
    excluded = excluded_docs(root)
    hook = load_c_hook(root)
    header_cache: dict = {}
    anchor_cache: dict[pathlib.Path, set[str] | None] = {}
    virt_pages = virtual_docs_pages(root)
    cmd_groups, known_cmds = load_known_commands_and_codes(root)
    cli_subcmds = load_cli_subcommands(root)
    py_modules, service_ids = load_python_symbols(root)

    md_files: list[pathlib.Path] = []
    if (root / "docs").is_dir():
        for p in sorted((root / "docs").rglob("*.md")):
            rel = p.relative_to(root).as_posix()
            if rel not in excluded:
                md_files.append(p)
    for root_doc in ("CONTRIBUTING.md", "README.md"):
        p = root / root_doc
        if p.is_file():
            md_files.append(p)

    def get_anchors(path: pathlib.Path) -> set[str] | None:
        if path not in anchor_cache:
            anchor_cache[path] = page_anchors(root, hook, header_cache, path)
        return anchor_cache[path]

    dotted_re = re.compile(r"^([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)$")
    cli_re = re.compile(r"^caliper\s+([a-z][a-z0-9_-]*)(?:\s|$)")
    py_re = re.compile(r"^(caliper(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?:\(|$)")

    for md_path in md_files:
        rel_file = md_path.relative_to(root).as_posix()
        lines = outside_fences(md_path.read_text(encoding="utf-8"))

        for line_no, line in lines:
            # 1. Check relative Markdown links outside inline code.
            prose_line = strip_inline_code(line)
            for lm in LINK_RE.finditer(prose_line):
                target = lm.group(1).strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "caliper://")):
                    continue
                path_part, _, anchor = target.partition("#")
                if not path_part:
                    target_path = md_path
                else:
                    target_path = (md_path.parent / path_part).resolve()
                try:
                    target_rel = target_path.relative_to(root).as_posix()
                except ValueError:
                    problems.append("%s:%d: link `%s` leaves repository root" % (rel_file, line_no, target))
                    continue
                if target_rel in virt_pages:
                    continue
                if not target_path.exists():
                    problems.append("%s:%d: broken link `%s` (%s does not exist)" % (rel_file, line_no, target, target_rel))
                    continue
                if anchor and target_path.suffix == ".md":
                    ancs = get_anchors(target_path)
                    if ancs is not None and anchor not in ancs:
                        problems.append(
                            "%s:%d: broken anchor `#%s` on `%s`" % (rel_file, line_no, anchor, target_rel)
                        )

            # 2-5. Check inline code spans.
            for cm in INLINE_CODE_RE.finditer(line):
                code = cm.group(1).strip()
                if not code:
                    continue

                # 2. Repository file/folder paths.
                path_err = check_repo_path(root, code)
                if path_err:
                    problems.append("%s:%d: %s" % (rel_file, line_no, path_err))
                    continue

                # 3. CLI subcommand `caliper <subcmd>`.
                m_cli = cli_re.match(code)
                if m_cli and cli_subcmds:
                    sub = m_cli.group(1)
                    if sub not in cli_subcmds:
                        problems.append(
                            "%s:%d: unknown CLI subcommand `caliper %s`" % (rel_file, line_no, sub)
                        )
                    continue

                # 4. Python `caliper.*` symbols (excluding C service IDs and spec/TOML keys).
                m_py = py_re.match(code)
                if m_py:
                    sym = m_py.group(1)
                    if sym in service_ids or sym.endswith((".toml", ".json", ".h", ".hpp", ".cpp", ".py")):
                        continue
                    parts = sym.split(".")
                    if len(parts) == 2:
                        attr = parts[1]
                        if attr not in py_modules.get("caliper", set()):
                            problems.append("%s:%d: unknown Python symbol `%s`" % (rel_file, line_no, sym))
                    elif len(parts) >= 3:
                        mod2 = ".".join(parts[:2])
                        attr2 = parts[2]
                        if mod2 in py_modules and attr2 not in py_modules[mod2]:
                            problems.append("%s:%d: unknown Python symbol `%s`" % (rel_file, line_no, sym))
                    continue

                # 5. Command-layer `<group>.<verb>` names.
                m_dot = dotted_re.match(code)
                if m_dot:
                    grp, rhs = m_dot.group(1), m_dot.group(2)
                    if rhs in FILE_EXTS:
                        continue
                    if grp in cmd_groups and code not in known_cmds:
                        problems.append(
                            "%s:%d: unknown command or code `%s`" % (rel_file, line_no, code)
                        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    if not (args.root / "docs").is_dir():
        print("error: %s is not the repository root" % args.root, file=sys.stderr)
        return 2
    problems = check(args.root)
    for p in problems:
        print(p)
    if problems:
        print("%d stale or broken reference(s) found" % len(problems))
        return 1
    print("Documentation references verified: all links, paths, commands, CLI subcommands, and symbols exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
