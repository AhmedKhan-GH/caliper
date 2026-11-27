#!/usr/bin/env python3
"""Fail when the API reference of the docs misses a public symbol.

The reference pages are generated from the source: mkdocstrings renders each
`::: caliper.<name>` directive from its docstring, and the MkDocs hook
docs-theme/hooks/c_reference.py renders each `<!-- c-api ... -->` placeholder
from the doc comments of a C header. This check reads the same sources
statically (no import of caliper, no MkDocs build) and fails when:

  1. a name in `caliper.__all__`, `caliper.live.__all__` or
     `caliper.interp.__all__` has no docstring, or no page has a `:::`
     directive for it;
  2. a function pointer of a struct in sdk/include/caliper/services/, or a
     function of include/caliper/embed.h, live.h or live_views.h, has no
     /** */ comment, or no placeholder on any page renders it;
  3. a placeholder names a header that does not parse or a pattern that
     matches nothing (the build would fail on it too).

A constant's docstring is the string literal right after its assignment, as
mkdocstrings reads it. A name that __init__.py imports from a private module
is looked up in that module, also when the import sits under
`if TYPE_CHECKING:` because the package imports the name on first use.

Usage: check_api_reference.py [--root <repository root>] [--list]
  --list prints every symbol checked and the page that renders it.
Exit status: 0 when the reference is complete, 1 when anything is missing.
"""

import argparse
import ast
import fnmatch
import importlib.util
import pathlib
import re
import sys

PACKAGE = "python/caliper"
SERVICES = "sdk/include/caliper/services"
EMBED_HEADERS = ("include/caliper/embed.h", "include/caliper/live.h", "include/caliper/live_views.h")
HOOK = "docs-theme/hooks/c_reference.py"

DIRECTIVE = re.compile(r"^:::[ \t]+([A-Za-z_][\w.]*)[ \t]*$", re.M)
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def load_hook(root):
    """The c_reference hook module, which holds the header parser and the placeholder rules."""
    spec = importlib.util.spec_from_file_location("c_reference", root / HOOK)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("c_reference", module)
    spec.loader.exec_module(module)
    return module


def outside_fences(text):
    """The text of a Markdown page with its fenced code blocks blanked out."""
    out, fence = [], None
    for line in text.splitlines(keepends=True):
        m = FENCE.match(line)
        if m and (fence is None or line.strip().startswith(fence)):
            fence = m.group(1)[0] * 3 if fence is None else None
            out.append("\n")
        else:
            out.append("\n" if fence else line)
    return "".join(out)


def pages(root):
    for path in sorted((root / "docs").rglob("*.md")):
        yield path.relative_to(root).as_posix(), outside_fences(path.read_text(encoding="utf-8"))


# --- Python ----------------------------------------------------------------------------------------------------------

def module_all(tree):
    """The names of a module's __all__, from `__all__ = [...]` and `__all__ += [...]` at module level."""
    names = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                names.extend(ast.literal_eval(node.value))
    return names


def docstrings(tree):
    """Maps each module-level function, class and assigned name to its docstring ("" when it has none), and each
    `from .module import name` to ("import", module, original name)."""
    found = {}
    body = tree.body
    for i, node in enumerate(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = ast.get_docstring(node) or ""
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            nxt = body[i + 1] if i + 1 < len(body) else None
            doc = ""
            if (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant)
                    and isinstance(nxt.value.value, str)):
                doc = nxt.value.value.strip()
            for t in targets:
                if isinstance(t, ast.Name):
                    found.setdefault(t.id, doc)
        elif isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                found.setdefault(alias.asname or alias.name, ("import", node.module, alias.name))
        elif isinstance(node, ast.If) and is_type_checking(node.test):
            # A name imported on first use (a module __getattr__) is named here for the docs and type checkers.
            for sub in node.body:
                if isinstance(sub, ast.ImportFrom) and sub.level == 1 and sub.module:
                    for alias in sub.names:
                        found.setdefault(alias.asname or alias.name, ("import", sub.module, alias.name))
    return found


def is_type_checking(test):
    """True for the condition `TYPE_CHECKING` or `typing.TYPE_CHECKING`."""
    return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def python_symbols(root):
    """Yields (public name, docstring) for caliper.__all__, caliper.live.__all__ and caliper.interp.__all__."""
    package = root / PACKAGE
    trees = {}

    def tree_of(module):
        if module not in trees:
            trees[module] = ast.parse((package / ("%s.py" % module)).read_text(encoding="utf-8"))
        return trees[module]

    def resolve(module, name, depth=0):
        entry = docstrings(tree_of(module)).get(name)
        if isinstance(entry, tuple) and depth < 4:
            pkg = module.rsplit("/", 1)[0] if "/" in module else ""
            target = "%s/%s" % (pkg, entry[1]) if pkg else entry[1]
            return resolve(target, entry[2], depth + 1)
        return entry

    for module, prefix in (("__init__", "caliper"), ("live", "caliper.live"), ("interp/__init__", "caliper.interp")):
        for name in module_all(tree_of(module)):
            yield "%s.%s" % (prefix, name), resolve(module, name)


def python_problems(root, page_texts, listing):
    rendered = {}
    for rel, text in page_texts:
        for m in DIRECTIVE.finditer(text):
            rendered.setdefault(m.group(1), rel)
    problems = []
    for name, doc in python_symbols(root):
        if doc is None:
            problems.append("%s: not defined where caliper imports it" % name)
            continue
        if not doc.strip():
            problems.append("%s: no docstring" % name)
        if name not in rendered:
            problems.append("%s: no page has a '::: %s' directive" % (name, name))
        elif listing is not None:
            listing.append("%s  %s" % (name, rendered[name]))
    return problems


# --- C ---------------------------------------------------------------------------------------------------------------

def c_problems(root, page_texts, listing):
    hook = load_hook(root)
    problems = []
    covered = {}                    # (header, symbol) -> page
    headers = {}

    def header(rel):
        if rel not in headers:
            headers[rel] = hook.parse_header(str(root / rel))
        return headers[rel]

    for rel, text in page_texts:
        for d in hook.find_directives(text):
            try:
                rows = hook.select(header(d.header), d)
            except hook.HeaderError as e:
                problems.append("%s: %s" % (rel, e))
                continue
            for row in rows:
                if d.kind == "calls":
                    owner, item = row
                    key = item.name if owner is None else "%s.%s" % (owner, item.names[0])
                    covered.setdefault((d.header, key), rel)
                elif d.kind == "types":
                    covered.setdefault((d.header, row.name), rel)
                    for member in row.members:
                        if getattr(member, "is_function", False):
                            covered.setdefault((d.header, "%s.%s" % (row.name, member.names[0])), rel)

    required = []                   # (header, symbol, doc)
    for path in sorted((root / SERVICES).glob("*.h")):
        rel = path.relative_to(root).as_posix()
        try:
            h = header(rel)
        except hook.HeaderError as e:
            problems.append(str(e))
            continue
        for d in h.of("struct"):
            for member in d.members:
                if member.is_function:
                    required.append((rel, "%s.%s" % (d.name, member.names[0]), member.doc))
    for rel in EMBED_HEADERS:
        try:
            h = header(rel)
        except hook.HeaderError as e:
            problems.append(str(e))
            continue
        for d in h.of("function"):
            required.append((rel, d.name, d.doc))

    for rel, name, doc in required:
        if not doc.strip():
            problems.append("%s: %s has no /** */ comment" % (rel, name))
        page = covered.get((rel, name))
        if page is None:
            problems.append("%s: %s is in no generated table (no <!-- c-api --> placeholder renders it)"
                            % (rel, name))
        elif listing is not None:
            listing.append("%s  %s  %s" % (rel, name, page))
    return problems


def check(root, listing=None):
    page_texts = list(pages(root))
    return python_problems(root, page_texts, listing) + c_problems(root, page_texts, listing)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent,
                        help="repository root (default: the parent of scripts/)")
    parser.add_argument("--list", action="store_true", help="print every symbol checked and its page")
    args = parser.parse_args(argv)
    if not (args.root / "docs").is_dir() or not (args.root / PACKAGE).is_dir():
        print("error: %s is not the repository root" % args.root, file=sys.stderr)
        return 2
    listing = [] if args.list else None
    problems = check(args.root, listing)
    for line in listing or ():
        print(line)
    for p in problems:
        print(p)
    if problems:
        print("%d problem(s) in the API reference" % len(problems))
        return 1
    print("API reference complete: every public symbol has a docstring or comment and a page")
    return 0


if __name__ == "__main__":
    sys.exit(main())
