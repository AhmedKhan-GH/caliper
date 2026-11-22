"""Fills the C API reference of the docs from the headers at build time.

A page asks for a generated table with a placeholder line of its own:

    <!-- c-api calls sdk/include/caliper/services/metrics_v1.h -->
    <!-- c-api types include/caliper/embed.h CaliperViewDesc CaliperViewInfo -->
    <!-- c-api types include/caliper/embed.h CaliperViewInfo level=0 -->
    <!-- c-api ids include/caliper/embed.h CALIPER_EMBED_VERSION* -->

The first word picks the table, the second names the header from the repository root, and the words after it are
optional fnmatch patterns that select symbols (all of that kind when there are none):

    calls  one table of the functions and the function pointers of the structs: Call, Signature, Description. A
           pattern matches a function name, a struct name, a member name or "Struct.member". A table whose rows
           come from more than one struct names each entry as "Struct.member".
    types  each struct, enum, typedef and opaque type, with its comment, under a heading of its own
           (level=N sets the level, 3 by default, and level=0 leaves the heading out), then a table: Field, Type,
           Description for a struct, Constant, Value, Description for an enum.
    ids    one table of the #define constants and the constants of anonymous enums: Name, Value, Description.

The descriptions are the /** ... */ comments right above each declaration, member and enum constant; other
comments are not part of the reference. The parser knows the conventions of these headers and no more: C
declarations ended by ';', include guards, '#ifdef __cplusplus' blocks (skipped) and object-like or function-like
macros. A declaration it cannot read, a header that does not exist and a pattern that matches nothing stop the
build, so the reference cannot silently fall out of step with the headers. scripts/check_api_reference.py imports
this module to check that every service table entry and every embed, live and live_views function has a comment
and appears in a generated table.
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field

DIRECTIVE = re.compile(r"^<!-- c-api (calls|types|ids) (\S+)((?: \S+)*?) -->[ \t]*$", re.M)
KINDS = ("calls", "types", "ids")

_DOC = "\x01%d\x02"
_DOC_RE = re.compile(r"\x01(\d+)\x02")
_MACRO_RE = re.compile(r"\x03(\d+)\x04")
_BARRIER = "\x05"
_IDENT = re.compile(r"[A-Za-z_]\w*")


class HeaderError(ValueError):
    """A header the parser cannot read, or a placeholder that does not fit its header."""


@dataclass
class Member:
    """A struct field (one or more declarators of one type) or a function pointer."""
    names: list
    type: str
    doc: str
    ret: str = ""
    params: str = ""

    @property
    def is_function(self) -> bool:
        return bool(self.ret)


@dataclass
class Constant:
    name: str
    value: str
    doc: str


@dataclass
class Decl:
    kind: str                        # function, struct, enum, callback, alias, opaque, macro, constant
    name: str
    doc: str
    ret: str = ""                    # function, callback
    params: str = ""                 # function, callback, function-like macro
    value: str = ""                  # macro, constant: the value; alias: the type it names
    members: list = field(default_factory=list)       # struct: Member, enum: Constant
    conditional: bool = False        # macro defined inside a platform #if


@dataclass
class Header:
    path: str
    decls: list

    def of(self, *kinds):
        return [d for d in self.decls if d.kind in kinds]


# --- parsing ---------------------------------------------------------------------------------------------------------

def _doc_text(raw: str) -> str:
    """The text of a /** ... */ comment: the leading '*' of each line removed, paragraphs kept apart by a blank
    line."""
    body = raw[3:-2]
    lines = []
    for line in body.split("\n"):
        s = line.strip()
        if s.startswith("*"):
            s = s[1:]
            if s.startswith(" "):
                s = s[1:]
        lines.append(s.rstrip())
    paras, cur = [], []
    for s in lines:
        if s:
            cur.append(s)
        elif cur:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras)


def _strip_comments(text: str):
    """Returns the code with every comment removed and each /** ... */ comment replaced by a marker, and the list of
    doc comment texts."""
    out, docs, i, n = [], [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise HeaderError("unterminated comment")
            raw = text[i:end + 2]
            if raw.startswith("/**") and len(raw) > 4 and raw[3] in " \t\n":
                out.append(_DOC % len(docs))
                docs.append(_doc_text(raw))
            else:
                out.append(" " + "\n" * raw.count("\n"))
            i = end + 2
        elif c == "/" and text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out), docs


def _preprocess(code: str):
    """Handles the preprocessor lines. Returns the code with each #define replaced by a macro marker, every other
    directive by a barrier (a doc comment above it belongs to nothing), the lines of '#ifdef __cplusplus' blocks
    dropped, and the list of macros as (name, params or None, value, conditional)."""
    lines = code.split("\n")
    joined, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("#"):
            while line.endswith("\\") and i + 1 < len(lines):
                i += 1
                line = line[:-1] + " " + lines[i].strip()
        joined.append(line)
        i += 1

    macros, out = [], []
    stack = []          # per open #if: [skipping, platform conditional, guard name]
    for idx, line in enumerate(joined):
        s = line.strip()
        skipping = any(f[0] for f in stack)
        if not s.startswith("#"):
            out.append("" if skipping else line)
            continue
        m = re.match(r"#\s*(\w+)\s*(.*)$", s)
        directive, rest = (m.group(1), m.group(2).strip()) if m else ("", "")
        if directive in ("if", "ifdef", "ifndef"):
            cplusplus = "__cplusplus" in rest
            guard = None
            if directive == "ifndef":
                nxt = next((x.strip() for x in joined[idx + 1:] if x.strip()), "")
                if re.fullmatch(r"#\s*define\s+%s" % re.escape(rest), nxt):
                    guard = rest
            stack.append([cplusplus, not cplusplus and guard is None, guard])
            out.append(_BARRIER)
        elif directive in ("else", "elif"):
            if not stack:
                raise HeaderError("#%s without #if" % directive)
            top = stack[-1]
            if "__cplusplus" in rest or top[0]:
                top[0] = directive == "else" and not top[0]
            out.append(_BARRIER)
        elif directive == "endif":
            if not stack:
                raise HeaderError("#endif without #if")
            stack.pop()
            out.append(_BARRIER)
        elif directive == "define" and not skipping:
            dm = re.match(r"([A-Za-z_]\w*)(\([^)]*\))?\s*(.*)$", rest)
            if not dm:
                raise HeaderError("cannot read #define %s" % rest)
            name, params, value = dm.group(1), dm.group(2), dm.group(3).strip()
            if any(f[2] == name for f in stack):
                out.append(_BARRIER)            # the include guard itself
                continue
            conditional = any(f[1] for f in stack)
            out.append("\x03%d\x04" % len(macros))
            macros.append((name, params, re.sub(r"\s+", " ", value), conditional))
        else:
            out.append(_BARRIER)                # #pragma, #include, #error, #undef, a skipped #define
    if stack:
        raise HeaderError("#if without #endif")
    return "\n".join(out), macros


def _split_top(text: str, sep: str):
    """Splits text at sep outside (), [] and {}."""
    parts, depth, cur = [], 0, []
    for c in text:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return parts


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+,", ",", s)
    return s


def _lead_doc(chunk: str, docs):
    """Splits off the doc markers at the start of a chunk. Returns (doc of the last marker or "", rest)."""
    doc = ""
    rest = chunk
    while True:
        m = re.match(r"\s*\x01(\d+)\x02", rest)
        if not m:
            break
        doc = docs[int(m.group(1))]
        rest = rest[m.end():]
    if _DOC_RE.search(rest):
        raise HeaderError("a doc comment inside a declaration: %s" % _norm(_DOC_RE.sub("", rest))[:80])
    return doc, rest


_FUNC_PTR = re.compile(r"^(.*?)\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\((.*)\)$", re.S)
_FUNC = re.compile(r"^(.*?)\b([A-Za-z_]\w*)\s*\((.*)\)$", re.S)


def _members(body: str, docs, where: str):
    members = []
    chunks = _split_top(body, ";")
    for chunk in chunks:
        doc, code = _lead_doc(chunk, docs)
        code = code.strip()
        if not code:
            continue
        m = _FUNC_PTR.match(code)
        if m:
            ret, name, params = _norm(m.group(1)), m.group(2), _norm(m.group(3))
            members.append(Member([name], "%s (*)(%s)" % (ret, params), doc, ret or "void", params))
            continue
        if "{" in code or "(" in code:
            raise HeaderError("cannot read the member '%s' of %s" % (_norm(code)[:80], where))
        declarators = _split_top(code, ",")
        first = _norm(declarators[0])
        fm = re.match(r"^(.*?)([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$", first)
        if not fm or not fm.group(1).strip():
            raise HeaderError("cannot read the member '%s' of %s" % (first, where))
        base = fm.group(1).strip()
        names, dims = [fm.group(2)], [fm.group(3).replace(" ", "")]
        for d in declarators[1:]:
            dm = re.match(r"^\s*([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$", d)
            if not dm:
                raise HeaderError("cannot read the member '%s' of %s" % (_norm(d), where))
            names.append(dm.group(1))
            dims.append(dm.group(2).replace(" ", ""))
        if len(set(dims)) == 1:
            members.append(Member(names, _norm(base) + dims[0], doc))
        else:
            for nm, dm in zip(names, dims):
                members.append(Member([nm], _norm(base) + dm, doc))
    return members


def _enum_constants(body: str, docs, where: str):
    consts, last = [], None
    for chunk in _split_top(body, ","):
        doc, code = _lead_doc(chunk, docs)
        code = _norm(code)
        if not code:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*(?:=\s*(.+))?$", code)
        if not m:
            raise HeaderError("cannot read the constant '%s' of %s" % (code, where))
        value = m.group(2)
        if value is None:
            if last is None:
                value = "0"
            elif re.fullmatch(r"-?\d+", last):
                value = str(int(last) + 1)
            else:
                value = "%s + 1" % last
        consts.append(Constant(m.group(1), value, doc))
        last = value
    return consts


def _declaration(text: str, doc: str, docs, path: str):
    """Classifies one top-level declaration (the text before its ';'). Returns a list of Decl."""
    code = text.strip()
    flat = _norm(_DOC_RE.sub(" ", code))
    if flat.startswith("static_assert") or flat.startswith("_Static_assert"):
        return []
    m = re.match(r"^(typedef\s+)?(struct|enum)\s*([A-Za-z_]\w*)?\s*\{(.*)\}\s*([A-Za-z_]\w*)?$", code, re.S)
    if m:
        typedef, what, tag, body, alias = m.groups()
        name = alias or tag
        if what == "struct":
            if not name:
                raise HeaderError("%s: an anonymous struct" % path)
            return [Decl("struct", name, doc, members=_members(body, docs, name))]
        consts = _enum_constants(body, docs, name or "an anonymous enum")
        if name:
            return [Decl("enum", name, doc, members=consts)]
        if doc and len(consts) > 0 and not consts[0].doc:
            consts[0].doc = doc
        return [Decl("constant", c.name, c.doc, value=c.value) for c in consts]
    if _DOC_RE.search(code):
        raise HeaderError("%s: a doc comment inside '%s'" % (path, flat[:80]))
    m = re.match(r"^typedef\s+struct\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)$", flat)
    if m:
        return [Decl("opaque", m.group(2), doc)]
    m = re.match(r"^typedef\s+(.*)$", flat)
    if m:
        fm = _FUNC_PTR.match(m.group(1))
        if fm:
            return [Decl("callback", fm.group(2), doc, ret=_norm(fm.group(1)), params=_norm(fm.group(3)))]
        am = re.match(r"^([A-Za-z_][\w ]*?[\w*])\s*\b([A-Za-z_]\w*)$", m.group(1))
        if not am or "(" in m.group(1):
            raise HeaderError("%s: cannot read '%s'" % (path, flat[:80]))
        return [Decl("alias", am.group(2), doc, value=_norm(am.group(1)))]
    if re.fullmatch(r"struct\s+[A-Za-z_]\w*", flat):
        return []                               # a forward declaration of a type another header defines
    fm = _FUNC.match(flat)
    if fm and "(" not in fm.group(1):
        return [Decl("function", fm.group(2), doc, ret=_norm(fm.group(1)), params=_norm(fm.group(3)))]
    raise HeaderError("%s: cannot read '%s'" % (path, flat[:80]))


def parse_header(path: str) -> Header:
    """Parses one header into its declarations, in the order of the file."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise HeaderError("cannot read %s: %s" % (path, e)) from None
    try:
        code, docs = _strip_comments(text)
        code, macros = _preprocess(code)
    except HeaderError as e:
        raise HeaderError("%s: %s" % (path, e)) from None

    decls, buf, pending, start_doc, depth = [], [], "", None, 0
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        m = _DOC_RE.match(code, i) if c == "\x01" else None
        if m:
            if "".join(buf).strip():
                buf.append(m.group(0))          # a member's comment, read with the body
            else:
                pending = docs[int(m.group(1))]
            i = m.end()
            continue
        m = _MACRO_RE.match(code, i) if c == "\x03" else None
        if m:
            if "".join(buf).strip():
                raise HeaderError("%s: a #define inside a declaration" % path)
            name, params, value, conditional = macros[int(m.group(1))]
            if params is not None or value:
                decls.append(Decl("macro", name, pending, params=params or "", value=value,
                                  conditional=conditional))
            pending = ""
            i = m.end()
            continue
        if c == _BARRIER:
            if not "".join(buf).strip():
                pending = ""
            i += 1
            continue
        if not buf or not "".join(buf).strip():
            if not c.isspace():
                start_doc = pending
                pending = ""
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == ";" and depth == 0:
            decls.extend(_declaration("".join(buf), start_doc or "", docs, path))
            buf, start_doc = [], None
        else:
            buf.append(c)
        i += 1
    rest = _norm(_DOC_RE.sub(" ", "".join(buf)))
    if rest:
        raise HeaderError("%s: text after the last declaration: '%s'" % (path, rest[:80]))
    return Header(path, decls)


# --- selection --------------------------------------------------------------------------------------------------------

@dataclass
class Directive:
    kind: str
    header: str
    patterns: list
    level: int
    text: str


def find_directives(markdown: str):
    """Every placeholder of a page, outside fenced code blocks."""
    found = []
    fenced = _fenced_ranges(markdown)
    for m in DIRECTIVE.finditer(markdown):
        if any(a <= m.start() < b for a, b in fenced):
            continue
        words = m.group(3).split()
        level, patterns = 3, []
        for w in words:
            lm = re.fullmatch(r"level=(\d)", w)
            if lm:
                level = int(lm.group(1))
            else:
                patterns.append(w)
        found.append(Directive(m.group(1), m.group(2), patterns, level, m.group(0)))
    return found


def _fenced_ranges(markdown: str):
    ranges, start, fence = [], None, None
    pos = 0
    for line in markdown.splitlines(keepends=True):
        m = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if m:
            if fence is None:
                fence, start = m.group(1), pos
            elif line.strip().startswith(fence[0] * len(fence)) and line.strip().strip(fence[0]) == "":
                ranges.append((start, pos + len(line)))
                fence = None
        pos += len(line)
    if fence is not None:
        ranges.append((start, pos))
    return ranges


def _matches(names, patterns):
    return not patterns or any(fnmatch.fnmatchcase(n, p) for n in names for p in patterns)


def select(header: Header, directive: Directive):
    """The rows a placeholder asks for. calls: (struct name or None, Member or Decl); types: Decl; ids: Decl. Raises
    HeaderError when a pattern matches nothing."""
    pats = directive.patterns
    if directive.kind == "calls":
        rows = []
        for d in header.decls:
            if d.kind == "function" and _matches([d.name], pats):
                rows.append((None, d))
            elif d.kind == "struct":
                for mem in d.members:
                    if mem.is_function and _matches([d.name, mem.names[0], "%s.%s" % (d.name, mem.names[0])], pats):
                        rows.append((d.name, mem))
        names = [r[1].name if r[0] is None else r[1].names[0] for r in rows]
        names += [r[0] for r in rows if r[0]]
        names += ["%s.%s" % (r[0], r[1].names[0]) for r in rows if r[0]]
    elif directive.kind == "types":
        rows = [d for d in header.of("struct", "enum", "callback", "alias", "opaque") if _matches([d.name], pats)]
        names = [d.name for d in rows]
    else:
        rows = [d for d in header.of("macro", "constant") if not d.conditional and _matches([d.name], pats)]
        names = [d.name for d in rows]
    for p in pats:
        if not any(fnmatch.fnmatchcase(n, p) for n in names):
            raise HeaderError("%s: '%s' in '%s' matches no %s" % (header.path, p, directive.text, directive.kind))
    if not rows:
        raise HeaderError("%s: '%s' selects nothing" % (header.path, directive.text))
    return rows


# --- rendering --------------------------------------------------------------------------------------------------------

def _code(text: str) -> str:
    ticks = "``" if "`" in text else "`"
    pad = " " if ticks == "``" else ""
    return "%s%s%s%s%s" % (ticks, pad, text, pad, ticks)


def _cell(text: str) -> str:
    """A comment as one table cell: one line, pipes outside code spans escaped."""
    text = re.sub(r"\s*\n\s*", " ", text.strip())
    parts = re.split(r"(`+[^`]*`+)", text)
    return "".join(p if p.startswith("`") else p.replace("|", "\\|") for p in parts)


def _anchor(name: str, seen: set) -> str:
    if name in seen:
        return ""
    seen.add(name)
    return '<span id="%s"></span>' % name


def render(header: Header, directive: Directive, seen: set) -> str:
    """The Markdown that replaces a placeholder. seen holds the anchors already on the page."""
    rows = select(header, directive)
    if directive.kind == "calls":
        many = len({r[0] for r in rows if r[0]}) > 1
        out = ["| Call | Signature | Description |", "|---|---|---|"]
        for owner, item in rows:
            if owner is None:
                name, label = item.name, item.name
                sig = "%s (%s)" % (item.ret, item.params)
            else:
                name = "%s.%s" % (owner, item.names[0])
                label = name if many else item.names[0]
                sig = "%s (%s)" % (item.ret, item.params)
            out.append("| %s%s | %s | %s |" % (_anchor(name, seen), _code(label), _code(sig), _cell(item.doc)))
        return "\n".join(out)
    if directive.kind == "ids":
        out = ["| Name | Value | Description |", "|---|---|---|"]
        for d in rows:
            label = d.name + (d.params if d.kind == "macro" else "")
            out.append("| %s%s | %s | %s |" % (_anchor(d.name, seen), _code(label), _code(d.value), _cell(d.doc)))
        return "\n".join(out)
    blocks = []
    for d in rows:
        part = []
        if directive.level:
            part.append("%s %s" % ("#" * directive.level, _code(d.name)))
        if d.doc:
            part.append(d.doc)
        if d.kind == "struct":
            table = ["| Field | Type | Description |", "|---|---|---|"]
            for mem in d.members:
                label = ", ".join(_code(nm) for nm in mem.names)
                table.append("| %s%s | %s | %s |" % (_anchor("%s.%s" % (d.name, mem.names[0]), seen), label,
                                                    _code(mem.type), _cell(mem.doc)))
            part.append("\n".join(table))
        elif d.kind == "enum":
            table = ["| Constant | Value | Description |", "|---|---|---|"]
            for c in d.members:
                table.append("| %s%s | %s | %s |" % (_anchor(c.name, seen), _code(c.name), _code(c.value),
                                                    _cell(c.doc)))
            part.append("\n".join(table))
        elif d.kind == "callback":
            part.append("```c\ntypedef %s (*%s)(%s);\n```" % (d.ret, d.name, d.params))
        elif d.kind == "alias":
            part.append("```c\ntypedef %s %s;\n```" % (d.value, d.name))
        blocks.append("\n\n".join(part))
    return "\n\n".join(blocks)


# --- MkDocs hook ------------------------------------------------------------------------------------------------------

_cache = {}


def _root(config) -> str:
    return os.path.dirname(os.path.abspath(config.config_file_path))


def header_for(root: str, rel: str) -> Header:
    path = os.path.normpath(os.path.join(root, rel))
    if path not in _cache:
        _cache[path] = parse_header(path)
        _cache[path].path = rel
    return _cache[path]


def on_pre_build(config, **kwargs):
    _cache.clear()


def on_page_markdown(markdown, page, config, files, **kwargs):
    directives = find_directives(markdown)
    if not directives:
        return markdown
    root = _root(config)
    seen = set()
    for d in directives:
        try:
            table = render(header_for(root, d.header), d, seen)
        except HeaderError as e:
            raise HeaderError("%s: %s" % (page.file.src_uri, e)) from None
        markdown = markdown.replace(d.text, table, 1)
    return markdown
