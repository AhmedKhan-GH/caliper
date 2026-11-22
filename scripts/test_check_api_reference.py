"""Tests for scripts/check_api_reference.py and the header parser of docs-theme/hooks/c_reference.py."""

import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import check_api_reference as chk  # noqa: E402

hook = chk.load_hook(ROOT)

INIT = '''\
"""caliper."""
from ._consts import LIMIT

__all__ = ["show", "LIMIT"]
__all__ += ["Thing"]


def show(t):
    """Show a tensor."""


class Thing:
    """A thing."""
'''

CONSTS = '''\
LIMIT = 4
"""The limit."""
'''

LIVE = '''\
__all__ = ["list_sessions"]


def list_sessions():
    """List the sessions."""
'''

SERVICE = '''\
#pragma once
/* A service. */
#include <stdint.h>

/** The id of the service. */
#define CALIPER_THING_V1 "caliper.thing.v1"

#ifdef __cplusplus
extern "C" {
#endif

/** The table of the service. */
typedef struct CaliperThingV1 {
    /** `sizeof(CaliperThingV1)`. */
    uint32_t struct_size;
    /** Open a thing and return its id. */
    uint64_t (*open)(const char* name);
    /** Close a thing. */
    void (*close)(uint64_t id);
} CaliperThingV1;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperThingV1) == 24, "frozen");
#endif
'''

EMBED = '''\
#ifndef CALIPER_EMBED_H
#define CALIPER_EMBED_H
/** The opaque core. */
typedef struct CaliperCore CaliperCore;
/** Create a core. */
CaliperCore* caliper_core_create(const char* dir);
#endif
'''

LIVE_H = '''\
#ifndef CALIPER_LIVE_H
#define CALIPER_LIVE_H
/** Attach to a session. */
int caliper_core_live_attach(CaliperCore* core, const char* session);
#endif
'''

LIVE_VIEWS_H = '''\
#ifndef CALIPER_LIVE_VIEWS_H
#define CALIPER_LIVE_VIEWS_H
/** Count the views. */
int caliper_core_live_view_count(CaliperCore* core);
#endif
'''

PAGE = '''\
# Things

::: caliper.show

::: caliper.LIMIT

::: caliper.Thing

::: caliper.live.list_sessions

## Calls

<!-- c-api calls sdk/include/caliper/services/thing_v1.h -->

<!-- c-api calls include/caliper/embed.h -->

<!-- c-api calls include/caliper/live.h caliper_core_live_* -->

<!-- c-api calls include/caliper/live_views.h -->
'''


class Repo:
    """A minimal repository with every source the check reads."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = {
            "python/caliper/__init__.py": INIT,
            "python/caliper/_consts.py": CONSTS,
            "python/caliper/live.py": LIVE,
            "sdk/include/caliper/services/thing_v1.h": SERVICE,
            "include/caliper/embed.h": EMBED,
            "include/caliper/live.h": LIVE_H,
            "include/caliper/live_views.h": LIVE_VIEWS_H,
            "docs/things.md": PAGE,
        }
        for rel, text in files.items():
            self.write(rel, text)
        (self.root / "docs-theme/hooks").mkdir(parents=True)
        shutil.copy(ROOT / chk.HOOK, self.root / chk.HOOK)

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def edit(self, rel, old, new):
        path = self.root / rel
        text = path.read_text(encoding="utf-8")
        assert old in text, old
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def problems(self):
        return chk.check(self.root)

    def close(self):
        self.tmp.cleanup()


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()

    def tearDown(self):
        self.repo.close()

    def assertProblem(self, fragment):
        problems = self.repo.problems()
        self.assertTrue(any(fragment in p for p in problems), "no problem with %r in %r" % (fragment, problems))

    def test_complete_reference_passes(self):
        self.assertEqual(self.repo.problems(), [])

    def test_listing_names_every_symbol(self):
        listing = []
        chk.check(self.repo.root, listing)
        text = "\n".join(listing)
        for name in ("caliper.show", "caliper.LIMIT", "caliper.Thing", "caliper.live.list_sessions",
                     "CaliperThingV1.open", "CaliperThingV1.close", "caliper_core_create",
                     "caliper_core_live_attach", "caliper_core_live_view_count"):
            self.assertIn(name, text)

    def test_name_imported_on_first_use_is_found(self):
        self.repo.write("python/caliper/_lazy.py", 'class Lazy:\n    """A thing imported on first use."""\n')
        self.repo.edit("python/caliper/__init__.py", '__all__ += ["Thing"]\n',
                       '__all__ += ["Thing", "Lazy"]\n\nfrom typing import TYPE_CHECKING\n\n'
                       'if TYPE_CHECKING:\n    from ._lazy import Lazy\n')
        self.assertProblem("caliper.Lazy: no page")
        self.repo.edit("docs/things.md", "::: caliper.Thing\n", "::: caliper.Thing\n\n::: caliper.Lazy\n")
        self.assertEqual(self.repo.problems(), [])

    def test_missing_docstring_fails(self):
        self.repo.edit("python/caliper/__init__.py", '    """Show a tensor."""\n', "    pass\n")
        self.assertProblem("caliper.show: no docstring")

    def test_missing_constant_docstring_fails(self):
        self.repo.edit("python/caliper/_consts.py", '"""The limit."""\n', "")
        self.assertProblem("caliper.LIMIT: no docstring")

    def test_symbol_on_no_page_fails(self):
        self.repo.edit("docs/things.md", "::: caliper.Thing\n", "")
        self.assertProblem("caliper.Thing: no page")

    def test_live_symbol_on_no_page_fails(self):
        self.repo.edit("docs/things.md", "::: caliper.live.list_sessions\n", "")
        self.assertProblem("caliper.live.list_sessions: no page")

    def test_directive_in_code_fence_does_not_count(self):
        self.repo.edit("docs/things.md", "::: caliper.Thing\n", "```text\n::: caliper.Thing\n```\n")
        self.assertProblem("caliper.Thing: no page")

    def test_function_pointer_without_comment_fails(self):
        self.repo.edit("sdk/include/caliper/services/thing_v1.h", "    /** Close a thing. */\n", "")
        self.assertProblem("CaliperThingV1.close has no /** */ comment")

    def test_plain_comment_is_not_a_doc_comment(self):
        self.repo.edit("sdk/include/caliper/services/thing_v1.h", "/** Close a thing. */", "/* Close a thing. */")
        self.assertProblem("CaliperThingV1.close has no /** */ comment")

    def test_function_pointer_in_no_table_fails(self):
        self.repo.edit("docs/things.md", "<!-- c-api calls sdk/include/caliper/services/thing_v1.h -->",
                       "<!-- c-api calls sdk/include/caliper/services/thing_v1.h open -->")
        self.assertProblem("CaliperThingV1.close is in no generated table")

    def test_function_pointer_in_a_types_table_counts(self):
        self.repo.edit("docs/things.md", "<!-- c-api calls sdk/include/caliper/services/thing_v1.h -->",
                       "<!-- c-api types sdk/include/caliper/services/thing_v1.h CaliperThingV1 -->")
        self.assertEqual(self.repo.problems(), [])

    def test_embed_function_in_no_table_fails(self):
        self.repo.edit("docs/things.md", "<!-- c-api calls include/caliper/embed.h -->\n", "")
        self.assertProblem("caliper_core_create is in no generated table")

    def test_live_function_without_comment_fails(self):
        self.repo.edit("include/caliper/live_views.h", "/** Count the views. */\n", "")
        self.assertProblem("caliper_core_live_view_count has no /** */ comment")

    def test_pattern_that_matches_nothing_fails(self):
        self.repo.edit("docs/things.md", "caliper_core_live_* -->", "caliper_core_live_* caliper_nothing -->")
        self.assertProblem("'caliper_nothing'")

    def test_missing_header_fails(self):
        self.repo.edit("docs/things.md", "include/caliper/live_views.h -->", "include/caliper/missing.h -->")
        self.assertProblem("cannot read")


def parse(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "h.h"
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return hook.parse_header(str(path))


class ParserTest(unittest.TestCase):
    def test_guards_and_cplusplus_blocks_are_skipped(self):
        h = parse(SERVICE)
        self.assertEqual([(d.kind, d.name) for d in h.decls],
                         [("macro", "CALIPER_THING_V1"), ("struct", "CaliperThingV1")])
        self.assertEqual(h.decls[0].doc, "The id of the service.")

    def test_members_keep_their_comments(self):
        table = parse(SERVICE).decls[1]
        self.assertEqual([m.names[0] for m in table.members], ["struct_size", "open", "close"])
        self.assertEqual(table.members[1].ret, "uint64_t")
        self.assertEqual(table.members[1].params, "const char* name")
        self.assertEqual(table.members[1].doc, "Open a thing and return its id.")

    def test_multi_line_comment_keeps_paragraphs(self):
        h = parse('''\
            /**
             * First paragraph,
             * one sentence.
             *
             * Second paragraph.
             */
            void f(void);
            ''')
        self.assertEqual(h.decls[0].doc, "First paragraph, one sentence.\n\nSecond paragraph.")

    def test_typedef_and_struct_of_one_name_are_one_type(self):
        h = parse('''\
            typedef struct Ctl Ctl;
            /** The control. */
            struct Ctl {
                /** Size. */
                uint32_t struct_size;
            };
            ''')
        self.assertEqual([(d.kind, d.name, d.doc) for d in h.decls], [("struct", "Ctl", "The control.")])

    def test_enum_values_count_on(self):
        h = parse('''\
            /** Kinds. */
            typedef enum Kind {
                /** None. */
                KIND_NONE,
                /** One. */
                KIND_ONE,
                /** Ten. */
                KIND_TEN = 10,
                /** Eleven. */
                KIND_ELEVEN
            } Kind;
            ''')
        self.assertEqual([(c.name, c.value) for c in h.decls[0].members],
                         [("KIND_NONE", "0"), ("KIND_ONE", "1"), ("KIND_TEN", "10"), ("KIND_ELEVEN", "11")])

    def test_anonymous_enum_gives_constants(self):
        h = parse('''\
            enum {
                /** Bit 0. */
                FLAG_A = 1 << 0,
                /** Bit 1. */
                FLAG_B = 1 << 1
            };
            ''')
        self.assertEqual([(d.kind, d.name, d.value) for d in h.decls],
                         [("constant", "FLAG_A", "1 << 0"), ("constant", "FLAG_B", "1 << 1")])

    def test_fields_arrays_and_declarator_lists(self):
        h = parse('''\
            typedef struct S {
                /** Shape. */
                int64_t shape[8];
                /** Range. */
                float vmin, vmax;
                /** Names. */
                const char* const* names;
            } S;
            ''')
        members = h.decls[0].members
        self.assertEqual([(m.names, m.type) for m in members],
                         [(["shape"], "int64_t[8]"), (["vmin", "vmax"], "float"), (["names"], "const char* const*")])

    def test_platform_macro_is_conditional(self):
        h = parse('''\
            #if defined(_WIN32)
              #define EXPORT __declspec(dllexport)
            #else
              #define EXPORT
            #endif
            ''')
        self.assertTrue(all(d.conditional for d in h.decls))

    def test_unreadable_declaration_stops(self):
        with self.assertRaises(hook.HeaderError):
            parse("int (*table[4])(void) = 0;\n")

    def test_render_escapes_pipes_outside_code(self):
        h = parse('''\
            /** Either a | b, or `a | b`. */
            int f(void);
            ''')
        d = hook.Directive("calls", "h.h", [], 3, "")
        row = hook.render(h, d, set()).splitlines()[2]
        self.assertIn("a \\| b", row)
        self.assertIn("`a | b`", row)


if __name__ == "__main__":
    unittest.main()
