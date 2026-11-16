// The side chat's layout as plain rectangles, without ImGui: the right column
// beside the canvas at 2560 x 1440 and 1440 x 900, its minimums growing with
// the text scale, the upper part giving way, and the pane's rows with what
// collapses first as it narrows.
#include "builder/side_chat.h"

#include <doctest/doctest.h>

#include <cmath>

using namespace caliper_host;

namespace {

// Text widths without fonts: 8 px a character for labels, 7 for mono and
// tags, at text scale 1.
ChatMeasure fake_measure(float scale = 1.0f) {
    return [scale](const std::string& t, ChatFont f) {
        return (float)t.size() * (f == ChatFont::label ? 8.0f : 7.0f) * scale;
    };
}

// The work area the page's strips leave: the page bar (28) and the status bar (22).
WidgetRect work_area(float w, float h) { return {0, 28, w, h - 28 - 22}; }

void check_no_overlap(const std::vector<WidgetRect>& rs) {
    for (size_t i = 0; i < rs.size(); ++i)
        for (size_t j = i + 1; j < rs.size(); ++j) {
            INFO(i << " and " << j);
            CHECK_FALSE(rs[i].overlaps(rs[j]));
        }
}

ChatPaneInput pane_input(const WidgetRect& pane) {
    ChatPaneInput in;
    in.pane = pane;
    in.agent = "Stub agent";
    in.version = "1";
    in.mode = "default";
    in.tag = "RUNNING";
    in.header_actions = {ChatAction::restart};
    in.faults = 2;
    in.picker = "Stub agent";
    in.input_action = ChatAction::send;
    return in;
}

void check_pane(const ChatPaneLayout& lay, const WidgetRect& pane, const ChatLayoutMetrics& m) {
    CHECK(pane.contains(lay.header));
    CHECK(pane.contains(lay.conversation));
    CHECK(pane.contains(lay.input_row));
    CHECK(lay.header.y == pane.y);
    CHECK(lay.header.h == m.header);
    CHECK(lay.conversation.y == lay.header.bottom());
    CHECK(lay.input_row.bottom() == doctest::Approx(pane.bottom() - m.pad));
    CHECK(lay.conversation.bottom() <= lay.input_row.y);
    CHECK(lay.input_row.contains(lay.picker));
    CHECK(lay.input_row.contains(lay.input));
    CHECK(lay.input_row.contains(lay.send));
    check_no_overlap({lay.picker, lay.input, lay.send});
    std::vector<WidgetRect> head = {lay.title};
    for (const WidgetRect* r : {&lay.agent, &lay.version, &lay.mode, &lay.tag, &lay.faults})
        if (!r->empty()) head.push_back(*r);
    for (const WidgetRect& b : lay.header_buttons) head.push_back(b);
    for (const WidgetRect& r : head) CHECK(lay.header.contains(r));
    check_no_overlap(head);
    check_no_overlap({lay.header, lay.conversation, lay.input_row});
}

}  // namespace

TEST_CASE("side chat layout: the right column at 2560 x 1440 and 1440 x 900") {
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(1.0f);
    for (const auto& size : {std::pair<float, float>{2560, 1440}, std::pair<float, float>{1440, 900}}) {
        INFO(size.first << "x" << size.second);
        ChatColumnInput in;
        in.window = work_area(size.first, size.second);
        const ChatColumnLayout col = layout_chat_column(in, m);
        // The column is a share of the window; the canvas keeps the rest and stays the largest region.
        CHECK(col.column.w == std::round(size.first * 0.28f));
        CHECK(col.column.w >= m.min_width);
        CHECK(col.canvas.w + col.column.w == size.first);
        CHECK(col.canvas.w > col.column.w);
        CHECK(col.column.x == col.canvas.right());
        CHECK(col.column.h == in.window.h);
        CHECK(col.column.y == in.window.y);
        // Upper part closed: the chat takes the whole column.
        CHECK(col.upper.empty());
        CHECK(col.chat.x == col.column.x);
        CHECK(col.chat.w == col.column.w);
        CHECK(col.chat.h == col.column.h);
        // Upper part open: Selection and Properties above, the chat below.
        in.upper_open = true;
        const ChatColumnLayout open = layout_chat_column(in, m);
        CHECK(open.upper.h == std::round(in.window.h * 0.45f));
        CHECK(open.chat.y == open.upper.bottom());
        CHECK(open.chat.bottom() == in.window.bottom());
        CHECK(open.column.contains(open.upper));
        CHECK(open.column.contains(open.chat));
        CHECK(open.chat.h >= m.min_height);

        // The pane inside the chat's rectangle, at both heights.
        for (const ChatColumnLayout* c : {&col, &open}) {
            const ChatPaneLayout lay = layout_chat_pane(pane_input(c->chat), m, fake_measure());
            check_pane(lay, c->chat, m);
            CHECK(lay.conversation.h > c->chat.h * 0.6f);
            CHECK(lay.picker_label);
            CHECK(lay.keys);
            CHECK_FALSE(lay.agent.empty());
            CHECK_FALSE(lay.version.empty());
        }
    }
}

TEST_CASE("side chat layout: the minimums grow with the text scale, and the canvas gives way last") {
    const ChatLayoutMetrics m1 = ChatLayoutMetrics::for_text_scale(1.0f);
    const ChatLayoutMetrics m2 = ChatLayoutMetrics::for_text_scale(2.0f);
    CHECK(m2.min_width == 2 * m1.min_width);
    CHECK(m2.min_height == 2 * m1.min_height);
    CHECK(m2.header == 2 * m1.header);
    ChatColumnInput in;
    in.window = work_area(1440, 900);
    // At scale 2 the share (403 px) is below the minimum (600 px).
    CHECK(layout_chat_column(in, m2).column.w == m2.min_width);
    // A wide share in a narrow window: the share gives way, down to the
    // column's minimum, so that the canvas keeps its own.
    in.window = work_area(900, 700);
    in.share = 0.5f;
    ChatColumnLayout c = layout_chat_column(in, m1);
    CHECK(c.column.w == 900 - m1.min_canvas);
    CHECK(c.canvas.w == m1.min_canvas);
    // Narrower than both minimums: the column keeps its minimum, the canvas the rest.
    in.window = work_area(640, 700);
    c = layout_chat_column(in, m1);
    CHECK(c.column.w == m1.min_width);
    CHECK(c.canvas.w == 640 - m1.min_width);
    // A short window with the upper part open: the upper part gives way to keep the chat's minimum.
    in.window = work_area(1440, 420);
    in.upper_open = true;
    c = layout_chat_column(in, m1);
    CHECK(c.upper.empty());
    CHECK(c.chat.h == in.window.h);
    in.window = work_area(1440, 430);
    c = layout_chat_column(in, m1);
    CHECK(c.chat.h == m1.min_height);
    CHECK(c.upper.h == in.window.h - m1.min_height);
}

TEST_CASE("side chat layout: what collapses first as the pane narrows") {
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(1.0f);
    ChatPaneInput in = pane_input({0, 0, 560, 600});
    in.version = "0.107.0";
    in.agent = "Claude Code";
    in.picker = "Claude Code";
    ChatPaneLayout lay = layout_chat_pane(in, m, fake_measure());
    CHECK_FALSE(lay.agent.empty());
    CHECK_FALSE(lay.version.empty());
    CHECK(lay.picker_label);
    CHECK(lay.keys);
    check_pane(lay, in.pane, m);

    // Narrower: the version goes first, then the agent's name in the header.
    bool saw_version_only_gone = false, saw_agent_gone = false, saw_picker_gone = false, saw_keys_gone = false;
    for (float w = 560; w >= 200; w -= 4) {
        in.pane = {0, 0, w, 600};
        lay = layout_chat_pane(in, m, fake_measure());
        INFO("width " << w);
        check_pane(lay, in.pane, m);
        if (w == 560) CHECK_FALSE(lay.mode.empty());
        if (!lay.mode.empty()) CHECK_FALSE(lay.version.empty());   // the mode goes before the version
        if (lay.version.empty() && !lay.agent.empty()) saw_version_only_gone = true;
        if (lay.agent.empty()) {
            saw_agent_gone = true;
            CHECK(lay.version.empty());
        }
        if (!lay.picker_label) {
            saw_picker_gone = true;
            CHECK(saw_agent_gone);   // the header's name goes before the picker's
        }
        if (!lay.keys) {
            saw_keys_gone = true;
            CHECK_FALSE(lay.picker_label);   // the picker's name goes before the keys
        }
        CHECK(lay.input.w > 0);
    }
    CHECK(saw_version_only_gone);
    CHECK(saw_agent_gone);
    CHECK(saw_picker_gone);
    CHECK(saw_keys_gone);
}

TEST_CASE("side chat layout: a longer input grows the row up to its maximum, and the draft's chips fold") {
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(1.0f);
    ChatPaneInput in = pane_input({1000, 100, 400, 700});
    const ChatPaneLayout one = layout_chat_pane(in, m, fake_measure());
    in.input_lines = 3;
    const ChatPaneLayout three = layout_chat_pane(in, m, fake_measure());
    CHECK(three.input_row.h == m.row + 2 * m.line);
    CHECK(three.input_row.bottom() == one.input_row.bottom());
    CHECK(three.conversation.h == one.conversation.h - 2 * m.line);
    in.input_lines = 40;
    CHECK(layout_chat_pane(in, m, fake_measure()).input_row.h == m.row + (float)(m.max_input_lines - 1) * m.line);
    // The send button sits on the bottom line, beside the picker.
    CHECK(three.send.y == three.picker.y);
    CHECK(three.send.bottom() <= three.input_row.bottom());

    in.input_lines = 1;
    in.chips = {"context", "spec curves", "source 1", "render", "conversation", "preview diff", "ids"};
    const ChatPaneLayout chips = layout_chat_pane(in, m, fake_measure());
    CHECK_FALSE(chips.chips.empty());
    CHECK(chips.chips.bottom() <= chips.input_row.y);
    CHECK(chips.conversation.bottom() <= chips.chips.y);
    CHECK(chips.chips_shown > 0);
    CHECK(chips.chips_shown < (int)in.chips.size());
    CHECK(chips.chips.contains(chips.clear));
    for (const WidgetRect& r : chips.chip_rects) {
        CHECK(chips.chips.contains(r));
        CHECK(r.right() <= chips.clear.x);
    }
    // A wide pane shows them all.
    in.pane = {0, 0, 1400, 700};
    CHECK(layout_chat_pane(in, m, fake_measure()).chips_shown == (int)in.chips.size());
}

TEST_CASE("side chat layout: the builder's left column keeps its share free beside the chat") {
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(1.0f);
    for (const auto& size : {std::pair<float, float>{2560, 1440}, std::pair<float, float>{1440, 900}}) {
        INFO(size.first << "x" << size.second);
        ChatColumnInput in;
        in.window = work_area(size.first, size.second);
        in.left_open = true;
        const ChatColumnLayout c = layout_chat_column(in, m);
        CHECK(c.left.x == in.window.x);
        CHECK(c.left.w == std::max(m.min_left, std::round(size.first * 0.18f)));
        CHECK(c.canvas.x == c.left.right());
        CHECK(c.canvas.right() == c.column.x);
        CHECK(c.left.w + c.canvas.w + c.column.w == size.first);
        CHECK(c.canvas.w >= m.min_canvas);
        check_no_overlap({c.left, c.canvas, c.column});
        // Closed, it takes nothing.
        in.left_open = false;
        CHECK(layout_chat_column(in, m).left.empty());
        CHECK(layout_chat_column(in, m).canvas.x == in.window.x);
    }
    // A narrow window: the right column gives way to its minimum first, then the left one.
    ChatColumnInput in;
    in.window = work_area(1050, 700);
    in.left_open = true;
    in.left_share = 0.3f;   // 315 points wanted
    const ChatColumnLayout c = layout_chat_column(in, m);
    CHECK(c.column.w == m.min_width);
    CHECK(c.left.w == 1050 - m.min_width - m.min_canvas);
    CHECK(c.canvas.w == m.min_canvas);
}

TEST_CASE("side chat layout: the page with the chat open and closed, under the strips, at two sizes") {
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(1.0f);
    const float page_bar = 28, status_bar = 22, approval_strip = 120;
    for (const auto& size : {std::pair<float, float>{2560, 1440}, std::pair<float, float>{1440, 900}}) {
        INFO(size.first << "x" << size.second);
        const WidgetRect window{0, 0, size.first, size.second};
        for (const bool left : {false, true}) {
            INFO("left " << left);
            const ChatColumnLayout open = layout_chat_page(window, page_bar, status_bar + approval_strip, true, left, m);
            const ChatColumnLayout closed = layout_chat_page(window, page_bar, status_bar + approval_strip, false, left, m);
            // The strips keep their places: the page starts under the page bar and ends above the strip.
            for (const ChatColumnLayout* c : {&open, &closed}) {
                CHECK(c->canvas.y == page_bar);
                CHECK(c->canvas.bottom() == size.second - status_bar - approval_strip);
            }
            // Open: the chat at the right, the canvas beside it, nothing overlaps.
            CHECK(open.chat.right() == size.first);
            CHECK(open.chat.w == std::max(m.min_width, std::round(size.first * 0.28f)));
            CHECK(open.canvas.right() == open.chat.x);
            CHECK(open.chat.y == page_bar);
            CHECK(open.chat.bottom() == size.second - status_bar - approval_strip);
            check_no_overlap({open.left, open.canvas, open.chat});
            // Closed: no column, and the canvas takes its width back.
            CHECK(closed.chat.empty());
            CHECK(closed.column.empty());
            CHECK(closed.canvas.right() == size.first);
            CHECK(closed.canvas.w == open.canvas.w + open.chat.w);
            CHECK(closed.left.w == open.left.w);
        }
    }
}
