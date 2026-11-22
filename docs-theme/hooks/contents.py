"""Fills docs/contents.md with the full docs tree, built from the site navigation at build time.

The tree is HTML: each tab and section is a toggle that opens and closes its branch, each page is a leaf link,
and connector lines show the nesting. The header logo links to this page (extra.homepage in
mkdocs.yml). Pages outside the navigation, such as the generated roadmap feature pages, are not listed.
"""
import html
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


def _href(page):
    # The Contents page is served at contents/, so every other page is one level up.
    return "../" + page.url


def _leaf(page):
    return '<li class="dt-page"><a href="%s">%s</a></li>' % (html.escape(_href(page)), html.escape(_title(page)))


def _node(title, children, depth):
    body = "".join(_item(child, depth + 1) for child in children)
    return ('<li class="dt-section is-open"><button type="button" class="dt-toggle">%s</button><ul>%s</ul></li>'
            % (html.escape(title), body))


def _item(item, depth):
    if isinstance(item, Section):
        return _node(item.title, item.children, depth)
    if isinstance(item, Page) and item.file.src_uri != PAGE:
        return _leaf(item)
    return ""


def on_nav(nav, config, files):
    global _tree
    roots = "".join(_item(item, 0) for item in nav.items)
    buttons = ('<div class="dt-actions">'
               '<button type="button" class="dt-btn" data-all="open">Expand All</button>'
               '<button type="button" class="dt-btn" data-all="close">Collapse All</button></div>')
    script = ("<script>(function () {"
              "var tree = document.currentScript.parentElement;"
              "tree.addEventListener('click', function (e) {"
              "var t = e.target.closest('.dt-toggle');"
              "if (t) { t.parentElement.classList.toggle('is-open'); return; }"
              "var b = e.target.closest('[data-all]');"
              "if (b) tree.querySelectorAll('.dt-section').forEach(function (s) {"
              "s.classList.toggle('is-open', b.getAttribute('data-all') === 'open'); });"
              "}); })();</script>")
    _tree = '<div class="docs-tree">%s<ul class="dt-root">%s</ul>%s</div>' % (buttons, roots, script)
    return nav


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == PAGE:
        return markdown.replace(MARK, _tree)
    return markdown
