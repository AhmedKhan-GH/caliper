#!/usr/bin/env python3
"""Fail when a docs heading breaks the docs title rule.

Every H1, H2 and H3 (ATX `#`, `##`, `###`) in every .md file under docs/ is
checked. YAML front matter and fenced code blocks (``` or ~~~, any length)
are skipped, and so are H4 and deeper. Inline Markdown is stripped before a
heading is judged: links keep their text, emphasis markers, a trailing {#id}
attribute list and known HTML tags (a, b, br, em, i, kbd, span, strong, ...)
are dropped. A code span in backticks is one identifier and always passes.

A heading fails when:

  1. its first word (after an optional numbering prefix such as "1." or
     "10.") is The, A, An or Your, in any case;
  2. it contains a colon outside code spans;
  3. it ends with a question mark;
  4. it is not in Title Case: every word starts with an upper-case letter or
     a digit, except the minor words below, which are lower case when they
     are not the first word. The first word must not be a lower-case minor
     word. In a hyphenated word each part follows the same rule, and a minor
     word as a later part may be lower case ("Up-to-Date", "In-Tree").

Identifiers pass in their own case: tokens containing . _ / \\ = ( ) < >,
tokens starting with a dash, tokens with an upper-case letter after the first
character (PyTorch, macOS, WebGPU), and the names in IDENTIFIER_WORDS. Prefer
writing a real lower-case identifier in backticks to extending that list.
Punctuation such as , ( ) & + - em and en dashes, quotes and ? is not a word.

Usage: check_doc_headings.py [--root <repository root>] [--list]
  --list prints every heading checked (path:line: level text), for debugging.
Exit status: 0 when every heading passes, 1 when any fails, 2 on bad input.
"""

import argparse
import pathlib
import re
import sys

MINOR_WORDS = frozenset(
    "a an the and but or nor for so yet as at by in of on per to up via vs "
    "with from into onto over than".split()
)

# Lower-case identifiers that appear in headings without backticks.
IDENTIFIER_WORDS = frozenset(
    "macos ios npm pip uv git ctest cmake caliper arm64 x86".split()
)

ARTICLE_WORDS = {"the", "a", "an"}

HTML_TAGS = (
    "a abbr b br code div em i img kbd mark p s small span strong sub sup u"
).split()

HEADING = re.compile(r"^ {0,3}(#{1,3})[ \t]+(.*?)[ \t]*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
NUMBERING = re.compile(r"^\d+(\.\d+)*[.)]?$")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
PLACEHOLDER = re.compile(r"\x00(\d+)\x00")
IDENTIFIER_CHARS = re.compile(r"[._/\\=()<>]")
HTML_TAG = re.compile(r"</?(?:%s)(?:\s[^>]*)?/?>" % "|".join(HTML_TAGS), re.I)


def strip_inline(text):
    """Returns (text, spans): code spans replaced by \\x00N\\x00 markers, then
    links, emphasis, attribute lists and HTML tags removed."""
    spans = []

    def keep(match):
        spans.append(match.group(2))
        return "\x00%d\x00" % (len(spans) - 1)

    text = CODE_SPAN.sub(keep, text)
    text = re.sub(r"[ \t]*\{[^{}]*\}[ \t]*$", "", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = HTML_TAG.sub("", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"(?<!\w)__+(?=\S)(.+?)(?<=\S)__+(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)", r"\1", text)
    return text.strip(), spans


def clean_token(token):
    """Strips surrounding punctuation from one whitespace-separated token."""
    token = token.lstrip("([\"'“‘")
    token = token.rstrip(",;.)]\"'”’?!")
    return token


def words_of(text):
    """Returns the words of stripped heading text, numbering prefix removed."""
    words = []
    for token in text.split():
        word = clean_token(token)
        if not word or not any(c.isalnum() or c == "\x00" for c in word):
            continue
        words.append(word)
    if len(words) > 1 and NUMBERING.match(words[0]):
        words = words[1:]
    return words


def part_fails(part, first):
    """True when one word (or hyphen part) breaks the Title Case rule."""
    if not part or "\x00" in part:
        return False
    if not part[0].isalpha():
        return False
    if part[0].isupper():
        return False
    if any(c.isupper() for c in part[1:]):
        return False
    if IDENTIFIER_CHARS.search(part):
        return False
    if part.lower() in IDENTIFIER_WORDS:
        return False
    if re.fullmatch(r"v\d+([._]\d+)*", part):
        return False
    if part in MINOR_WORDS and not first:
        return False
    return True


def word_failure(word, first_word):
    """Returns the offending text when a word is not Title Case, else None."""
    if word.startswith("-") or "\x00" in word:
        return None
    if IDENTIFIER_CHARS.search(word):
        return None
    parts = word.split("-")
    for index, part in enumerate(parts):
        if part_fails(part, first_word and index == 0):
            return word
    return None


def judge(heading):
    """Returns the failure reason for one heading's raw text, or None."""
    text, spans = strip_inline(heading)
    words = words_of(text)
    if not words:
        return None
    first = words[0].lower()
    if first in ARTICLE_WORDS:
        return "starts with an article"
    if first == "your":
        return "starts with Your"
    if ":" in PLACEHOLDER.sub("", text):
        return "contains a colon"
    if text.rstrip().endswith("?"):
        return "question"
    for index, word in enumerate(words):
        bad = word_failure(word, index == 0)
        if bad:
            return "not Title Case (%s)" % bad
    return None


def headings(path):
    """Yields (line number, level, raw text) for the checked headings of a file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() in ("---", "..."):
                start = index + 1
                break
    fence = None
    for index in range(start, len(lines)):
        line = lines[index]
        match = FENCE.match(line)
        if fence:
            if (match and match.group(1)[0] == fence[0]
                    and len(match.group(1)) >= len(fence) and not match.group(2).strip()):
                fence = None
            continue
        if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
            fence = match.group(1)
            continue
        found = HEADING.match(line)
        if found:
            text = re.sub(r"[ \t]+#+$", "", found.group(2))
            yield index + 1, len(found.group(1)), text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent)
    parser.add_argument("--list", action="store_true",
                        help="print every heading checked")
    args = parser.parse_args(argv)
    docs = args.root / "docs"
    if not docs.is_dir():
        print("error: %s is not a directory" % docs, file=sys.stderr)
        return 2
    checked = failed = 0
    for path in sorted(docs.rglob("*.md")):
        rel = path.relative_to(args.root).as_posix()
        for number, level, text in headings(path):
            checked += 1
            if args.list:
                print("%s:%d: H%d %s" % (rel, number, level, text))
            reason = judge(text)
            if reason:
                failed += 1
                print("%s:%d: %s: %s" % (rel, number, text, reason))
    print("%d of %d docs headings break the title rule" % (failed, checked))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
