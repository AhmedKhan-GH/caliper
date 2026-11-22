// spec_page_layout.cpp: the spec page's rectangles (spec_page_layout.h).
#include "builder/spec_page_layout.h"

#include "builder/side_chat.h"

#include <algorithm>
#include <cmath>

namespace caliper_host {

BuilderPaneLayout layout_spec_page(const SpecPageLayoutInput& in) {
    const ChatLayoutMetrics cm = ChatLayoutMetrics::for_text_scale(in.text_scale);
    if (!in.build) {
        BuilderPaneLayout out;
        out.canvas = in.area;
        if (in.chat) {
            ChatColumnInput ci;
            ci.window = {in.area.x, in.area.y, in.area.w, in.area.h};
            const ChatColumnLayout c = layout_chat_column(ci, cm);
            out.canvas = {c.canvas.x, c.canvas.y, c.canvas.w, c.canvas.h};
            out.chat = true;
            out.chat_rect = {c.chat.x, c.chat.y, c.chat.w, c.chat.h};
        }
        return out;
    }
    BuilderPaneMetrics m = BuilderPaneMetrics::for_text_scale(in.text_scale);
    if (in.chat) {
        // The chat's column, as it is outside build mode, holds Properties
        // above it; the left column takes the chat layout's share for it.
        const ChatColumnInput ci;
        m.right_share = ci.share;
        m.right_min = std::max(m.right_min, cm.min_width);
        m.left_share = ci.left_share;
        m.left_min = std::max(m.left_min, cm.min_left);
        m.chat_share = 1.0f - ci.upper_share;
        m.properties_min_h = std::max(m.properties_min_h, cm.min_upper);
    }
    BuilderPaneLayoutInput bi;
    bi.area = in.area;
    bi.build = true;
    bi.text_open = in.text_open;
    bi.chat = in.chat;
    return compute_builder_pane_layout(bi, m);
}

BuilderPaneLayout spec_page_layout_rule(const BuilderPaneLayoutInput& in, float text_scale) {
    SpecPageLayoutInput p;
    p.area = in.area;
    p.build = in.build;
    p.text_open = in.text_open;
    p.chat = in.chat;
    p.text_scale = text_scale;
    return layout_spec_page(p);
}

}  // namespace caliper_host
