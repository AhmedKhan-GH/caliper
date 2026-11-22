#pragma once
// ============================================================================
// spec_page_layout.h: every rectangle of the app's spec page under its strips,
// from one function, for each combination of build mode, the spec text strip
// and the side chat (design/builder-pane.md, The regions; the placement
// sketch of design/builder-overview.md). It is computed without ImGui so
// that tests check it at any window size.
//
//   - Outside build mode the canvas takes the work area, and the side chat,
//     when open, the right column that layout_chat_column() gives it.
//   - In build mode the builder pane's regions (builder_pane_layout.h): the
//     left column with Sources and Primitives, the canvas, the text strip
//     under it while it is open, and the right column with Properties in its
//     upper part. With the chat open, the chat takes the column's lower part
//     and the columns take the chat's shares and minimums
//     (ChatColumnInput, ChatLayoutMetrics), so opening Build beside an open
//     chat keeps the chat's width.
//
// The work area is what the host's strips leave: the page bar, a stream's
// run bar, the status bar, the jobs tray and the approval cards' strip, which
// keep their places.
// ============================================================================
#include "builder/builder_pane_layout.h"

namespace caliper_host {

struct SpecPageLayoutInput {
    SpecRect area;              // the work area under the page's strips
    bool build = false;
    bool text_open = false;     // the spec text strip (build mode only)
    bool chat = false;          // the side chat
    float text_scale = 1.0f;
};

// The page's regions. Outside build mode only `canvas`, `chat` and
// `chat_rect` are set.
BuilderPaneLayout layout_spec_page(const SpecPageLayoutInput& in);

// The builder pane's layout rule for the spec page (BuilderPane::set_layout_rule).
BuilderPaneLayout spec_page_layout_rule(const BuilderPaneLayoutInput& in, float text_scale);

}  // namespace caliper_host
