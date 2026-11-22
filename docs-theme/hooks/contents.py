"""Fills docs/contents.md with the full docs tree, built from the site navigation at build time.

The header logo links to that page (extra.homepage in mkdocs.yml), so every page of the navigation is one
click away. Pages outside the navigation, such as the generated roadmap feature pages, are not listed.
"""
import re

from mkdocs.structure.nav import Page, Section

PAGE = "contents.md"
MARK = "<!-- docs-tree -->"
_tree = ""


def _title(page):
    if page.title:
        return page.title
    m = re.search(r"^# (.+)$", page.file.content_string, re.M)
    return m.group(1).strip() if m else page.file.src_uri


def _items(items, depth, out):
    for item in items:
        pad = "    " * depth
        if isinstance(item, Section):
            out.append("%s- **%s**" % (pad, item.title))
            _items(item.children, depth + 1, out)
        elif isinstance(item, Page) and item.file.src_uri != PAGE:
            out.append("%s- [%s](%s)" % (pad, _title(item), item.file.src_uri))


def on_nav(nav, config, files):
    global _tree
    out = []
    for item in nav.items:
        if isinstance(item, Section):
            out += ["", "## " + item.title, ""]
            _items(item.children, 0, out)
        elif isinstance(item, Page) and item.file.src_uri != PAGE:
            out += ["", "## " + _title(item), "", "- [%s](%s)" % (_title(item), item.file.src_uri)]
    _tree = "\n".join(out) + "\n"
    return nav


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == PAGE:
        return markdown.replace(MARK, _tree)
    return markdown
