// side_chat_layout.cpp: pure-rectangle column, page, and pane layout
// computation for the side chat (side_chat.h), without ImGui.
#include "builder/side_chat.h"
#include "builder/side_chat_internal.h"
#include "builder/widget_surface.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace caliper_host {

using side_chat_detail::upper;

// ---- the layout ---------------------------------------------------------------------------

ChatLayoutMetrics ChatLayoutMetrics::for_text_scale(float s) {
    s = std::max(0.5f, s);
    ChatLayoutMetrics m;
    auto r = [s](float v) { return std::round(v * s); };
    m.header = r(26);
    m.row = r(30);
    m.line = r(18);
    m.chip = r(22);
    m.pad = r(8);
    m.gap = r(6);
    m.button = r(22);
    m.button_pad = r(8);
    m.key_gap = r(6);
    m.picker_mark = r(14);
    m.min_input = r(120);
    m.min_width = r(300);
    m.min_height = r(220);
    m.min_canvas = r(480);
    m.min_upper = r(160);
    m.min_left = r(220);
    return m;
}

ChatColumnLayout layout_chat_column(const ChatColumnInput& in, const ChatLayoutMetrics& m) {
    ChatColumnLayout out;
    const WidgetRect& w = in.window;
    float col = std::max(m.min_width, std::round(w.w * std::clamp(in.share, 0.05f, 0.9f)));
    float left = in.left_open ? std::max(m.min_left, std::round(w.w * std::clamp(in.left_share, 0.05f, 0.9f))) : 0.0f;
    if (w.w - col - left < m.min_canvas) col = std::max(m.min_width, w.w - left - m.min_canvas);
    if (in.left_open && w.w - col - left < m.min_canvas) left = std::max(m.min_left, w.w - col - m.min_canvas);
    col = std::min(col, w.w);
    left = std::min(left, std::max(0.0f, w.w - col));
    if (in.left_open) out.left = {w.x, w.y, left, w.h};
    out.canvas = {w.x + left, w.y, w.w - col - left, w.h};
    out.column = {w.x + w.w - col, w.y, col, w.h};
    out.chat = out.column;
    if (in.upper_open) {
        float upper = std::round(w.h * std::clamp(in.upper_share, 0.0f, 1.0f));
        float chat = w.h - upper;
        if (chat < m.min_height) {
            chat = std::min(m.min_height, w.h);
            upper = w.h - chat;
        }
        if (upper >= m.min_upper) {
            out.upper = {out.column.x, w.y, col, upper};
            out.chat = {out.column.x, w.y + upper, col, chat};
        }
    }
    return out;
}

ChatColumnLayout layout_chat_page(const WidgetRect& window, float top_strips, float bottom_strips, bool chat_open,
                                  bool left_open, const ChatLayoutMetrics& m) {
    ChatColumnInput in;
    in.window = {window.x, window.y + top_strips, window.w, std::max(0.0f, window.h - top_strips - bottom_strips)};
    in.left_open = left_open;
    ChatColumnLayout out = layout_chat_column(in, m);
    if (!chat_open) {
        out.canvas.w += out.column.w;
        out.column = out.upper = out.chat = WidgetRect{};
    }
    return out;
}

namespace {

float button_width(ChatAction a, bool keys, const ChatLayoutMetrics& m, const ChatMeasure& measure) {
    float w = 2.0f * m.button_pad + measure(chat_action_label(a), ChatFont::label);
    const char* key = chat_action_key(a);
    if (keys && key[0]) w += measure(key, ChatFont::key) + m.key_gap;
    return std::ceil(w);
}

}  // namespace

ChatPaneLayout layout_chat_pane(const ChatPaneInput& in, const ChatLayoutMetrics& m, const ChatMeasure& measure) {
    ChatPaneLayout out;
    const WidgetRect& P = in.pane;
    out.header = {P.x, P.y, P.w, std::min(m.header, P.h)};
    const int lines = std::clamp(in.input_lines, 1, std::max(1, m.max_input_lines));
    const float row_h = m.row + (float)(lines - 1) * m.line;

    // The input row and its collapses: the picker's name, then the keys.
    auto lay_input = [&](bool picker_label, bool keys) {
        out.input_row = {P.x + m.pad, P.bottom() - m.pad - row_h, std::max(0.0f, P.w - 2.0f * m.pad), row_h};
        const float by = std::round(out.input_row.bottom() - (m.row + m.button) * 0.5f);
        const float pw = 2.0f * m.button_pad + m.picker_mark + (picker_label ? measure(in.picker, ChatFont::label) : 0.0f);
        out.picker = {out.input_row.x + m.gap * 0.5f, by, std::ceil(pw), m.button};
        const float sw = button_width(in.input_action, keys, m, measure);
        out.send = {out.input_row.right() - m.gap * 0.5f - sw, by, sw, m.button};
        const float ix = out.picker.right() + m.gap;
        out.input = {ix, out.input_row.y, std::max(0.0f, out.send.x - m.gap - ix), row_h};
        out.picker_label = picker_label;
        out.keys = keys;
    };
    lay_input(true, true);
    if (out.input.w < m.min_input) lay_input(false, true);
    if (out.input.w < m.min_input) lay_input(false, false);

    // The draft's chips above the input row, with Clear at the right.
    if (!in.chips.empty()) {
        out.chips = {P.x + m.pad, out.input_row.y - m.gap - m.chip, std::max(0.0f, P.w - 2.0f * m.pad), m.chip};
        const float cw = std::ceil(2.0f * m.button_pad + measure(chat_action_label(ChatAction::clear), ChatFont::label));
        out.clear = {out.chips.right() - cw, out.chips.y, cw, m.chip};
        const float fold = std::ceil(measure("+" + std::to_string(in.chips.size()), ChatFont::label)) + m.gap;
        float x = out.chips.x;
        for (size_t i = 0; i < in.chips.size(); ++i) {
            const float w = std::ceil(2.0f * m.button_pad + measure(in.chips[i], ChatFont::label));
            const bool last = i + 1 == in.chips.size();
            const float limit = out.clear.x - m.gap - (last ? 0.0f : fold);
            if (x + w > limit) break;
            out.chip_rects.push_back({x, out.chips.y, w, m.chip});
            x += w + m.gap;
        }
        out.chips_shown = (int)out.chip_rects.size();
    }

    // The header: the title tag, the agent and its version at the left; the
    // buttons, the faults and the state's tag at the right.
    const WidgetRect& H = out.header;
    const float bh = std::min(m.button, H.h);
    const float by = std::round(H.y + (H.h - bh) * 0.5f);
    const float tw = std::ceil(measure(upper(in.title), ChatFont::tag));
    out.title = {H.x + m.pad, H.y, tw, H.h};
    float x = out.title.right() + 2.0f * m.gap;
    // Caliper's mode keeps its place after the title: the agent's details give way first.
    if (!in.permission.empty()) {
        const float pw = std::ceil(measure(in.permission, ChatFont::label) + 2.0f * m.button_pad + m.picker_mark);
        out.permission = {x, by, pw, bh};
        x += pw + 2.0f * m.gap;
    }
    float right = H.right() - m.pad;
    for (size_t i = in.header_actions.size(); i-- > 0;) {
        const float w = button_width(in.header_actions[i], out.keys, m, measure);
        out.header_buttons.insert(out.header_buttons.begin(), WidgetRect{right - w, by, w, bh});
        right -= w + m.gap;
    }
    // The faults chip, then the state's tag, give way when the header is too narrow for them.
    const float fw = in.faults > 0 ? std::ceil(measure("FAULTS " + std::to_string(in.faults), ChatFont::tag) + 2.0f * m.button_pad) : 0.0f;
    const float tgw = in.tag.empty() ? 0.0f : std::ceil(measure(in.tag, ChatFont::tag) + m.picker_mark);
    const bool tag_fits = tgw > 0 && right - tgw >= x;
    const bool faults_fit = fw > 0 && right - fw - (tag_fits ? tgw + m.gap : 0.0f) >= x;
    if (faults_fit) {
        out.faults = {right - fw, by, fw, bh};
        right -= fw + m.gap;
    }
    if (tag_fits && right - tgw >= x) {
        out.tag = {right - tgw, H.y, tgw, H.h};
        right -= tgw + m.gap;
    }
    const float aw = in.agent.empty() ? 0.0f : std::ceil(measure(in.agent, ChatFont::label));
    const float vw = in.version.empty() ? 0.0f : std::ceil(measure(in.version, ChatFont::mono));
    const float mw = in.mode.empty() ? 0.0f : std::ceil(measure(in.mode, ChatFont::mono));
    // The GPU note: a framed noun after the agent's name, which gives way
    // after the version and before the name.
    const float nw = in.note.empty() ? 0.0f : std::ceil(measure(in.note, ChatFont::tag) + 2.0f * m.button_pad);
    if (aw > 0 && x + aw <= right) {
        out.agent = {x, H.y, aw, H.h};
        x += aw + m.gap;
        const bool note_fits = nw > 0 && x + nw <= right;
        if (note_fits) {
            out.note = {x, by, nw, bh};
            x += nw + m.gap;
        }
        if ((nw == 0 || note_fits) && vw > 0 && x + vw <= right) {
            out.version = {x, H.y, vw, H.h};
            x += vw + 2.0f * m.gap;
            if (mw > 0 && x + mw <= right) out.mode = {x, H.y, mw, H.h};
        }
    }

    float top = H.bottom();
    if (in.allow_all) {
        out.banner = {P.x, top, P.w, m.row};
        const float ew = std::ceil(measure("End", ChatFont::label) + 2.0f * m.button_pad);
        out.banner_end = {P.right() - m.pad - ew, std::round(top + (m.row - m.button) * 0.5f), ew, m.button};
        top = out.banner.bottom();
    }
    const float bottom = in.chips.empty() ? out.input_row.y - m.pad : out.chips.y - m.gap;
    out.conversation = {P.x, top, P.w, std::max(0.0f, bottom - top)};
    return out;
}

ChatPaneInput chat_pane_input(const SideChat& chat, const WidgetRect& pane) {
    ChatPaneInput in;
    in.pane = pane;
    in.agent = chat.agent_title();
    in.version = chat.agent_info().version;
    in.mode = chat.mode();
    in.permission = chat.permission_word();
    in.allow_all = in.permission == "Allow all" || !chat.permission_note().empty();
    in.tag = chat.tag();
    in.note = chat.gpu_note();
    in.header_actions = chat.header_actions();
    in.faults = (int)chat.faults().size();
    in.picker = chat.agent() ? chat.agent_title() : std::string("Agent");
    in.input_lines = 1 + (int)std::count(chat.text().begin(), chat.text().end(), '\n');
    in.chips = chat.draft().chips;
    in.input_action = chat.input_action();
    return in;
}

}  // namespace caliper_host
