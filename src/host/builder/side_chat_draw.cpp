// side_chat_draw.cpp: SideChat::draw() and its ImGui rendering helpers in the
// house style (side_chat.h).
#include "builder/side_chat.h"
#include "builder/side_chat_internal.h"
#include "builder/chat_markdown.h"
#include "builder/widget_surface.h"

#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"

#include <caliper_builder/commands/command.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace caliper_host {

using side_chat_detail::split_lines;
using side_chat_detail::trimmed;
using side_chat_detail::upper;

// ---- the ImGui side -----------------------------------------------------------------------

namespace {

constexpr int kMaxCompactLines = 8;           // a permission card's compact lines before "+N"
constexpr int kMaxDiffLines = 16;             // a permission card's diff lines before "+N"

ImVec2 v2(float x, float y) { return ImVec2(x, y); }

ImVec2 extent(ImFont* f, float size, const std::string& s, float wrap = 0.0f) {
    if (!f) f = ImGui::GetFont();
    return f->CalcTextSizeA(size, FLT_MAX, wrap, s.c_str(), s.c_str() + s.size());
}

std::string fit(ImFont* f, float size, const std::string& s, float w) {
    if (extent(f, size, s).x <= w) return s;
    static const std::string kDots = "\xE2\x80\xA6";
    if (w <= 0.0f || extent(f, size, kDots).x > w) return {};
    size_t lo = 0, hi = s.size();
    while (lo < hi) {   // the longest prefix that fits with the dots
        size_t mid = (lo + hi + 1) / 2;
        while (mid > 0 && ((unsigned char)s[mid] & 0xC0) == 0x80) --mid;
        if (mid <= lo) break;
        if (extent(f, size, s.substr(0, mid) + kDots).x <= w) lo = mid;
        else hi = mid - 1;
    }
    while (lo > 0 && ((unsigned char)s[lo] & 0xC0) == 0x80) --lo;
    return s.substr(0, lo) + kDots;
}

void draw_str(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 c, const std::string& s, float wrap = 0.0f) {
    dl->AddText(f ? f : ImGui::GetFont(), size, ImVec2(std::round(p.x), std::round(p.y)), c, s.c_str(), s.c_str() + s.size(),
                wrap);
}

std::string first_line(const std::string& s) {
    const std::string t = trimmed(s);
    const size_t nl = t.find('\n');
    return nl == std::string::npos ? t : t.substr(0, nl);
}

int resize_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* s = static_cast<std::string*>(data->UserData);
        s->resize((size_t)data->BufTextLen);
        data->Buf = const_cast<char*>(s->c_str());
    }
    return 0;
}

// The house's buttons: an outlined key button on a square plate, the primary in the text colour.
bool house_button(const char* id, ImRect b, const std::string& label, const char* key, bool primary, bool keys,
                  const SpecLook& L, bool enabled = true) {
    const float s = L.text_scale;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::SetCursorScreenPos(b.Min);
    const bool clicked = ImGui::InvisibleButton(id, b.GetSize()) && enabled;
    const bool hovered = enabled && ImGui::IsItemHovered(), held = enabled && ImGui::IsItemActive();
    ImU32 outline = primary ? L.text : L.rule, fill = 0, key_ink = primary ? L.accent : L.secondary, ink = L.text;
    if (!enabled) {
        outline = L.rule_faint;
        key_ink = ink = L.faint;
    } else if (held && hovered) {
        fill = L.text;
        outline = fill;
        key_ink = ink = L.deep;
    } else if (hovered) {
        fill = L.plate_hover;
        key_ink = L.accent;
    }
    if (fill) dl->AddRectFilled(b.Min, b.Max, fill);
    dl->AddRect(b.Min, b.Max, outline, 0.0f, 0, 1.0f);
    const float ui_h = extent(L.ui, L.ui_size, "Ag").y, mono_h = extent(L.mono, L.mono_size, "Ag").y;
    float x = b.Min.x + std::round(8.0f * s);
    if (keys && key && key[0]) {
        draw_str(dl, L.mono, L.mono_size, v2(x, b.Min.y + (b.GetHeight() - mono_h) * 0.5f), key_ink, key);
        x += extent(L.mono, L.mono_size, key).x + std::round(6.0f * s);
    }
    draw_str(dl, L.ui, L.ui_size, v2(x, b.Min.y + (b.GetHeight() - ui_h) * 0.5f), ink,
         fit(L.ui, L.ui_size, label, b.Max.x - x - std::round(4.0f * s)));
    return clicked;
}

// A small tracked uppercase tag; returns its width.
float draw_tag(ImDrawList* dl, const SpecLook& L, ImVec2 p, float line_h, ImU32 c, const std::string& t) {
    const float tracking = std::round(L.small_size * 0.08f);
    const float h = extent(L.ui, L.small_size, "Ag").y;
    sternbild::text_tracked(dl, L.ui, L.small_size, v2(std::round(p.x), std::round(p.y + (line_h - h) * 0.5f)), c, t.c_str(),
                            tracking);
    return sternbild::text_tracked_width(L.ui, L.small_size, t.c_str(), tracking);
}

// The mark of a tool call's or a plan entry's status.
void status_mark(ImDrawList* dl, const SpecLook& L, ImVec2 c, const std::string& status) {
    const float half = std::round(3.5f * L.text_scale);
    if (status == "in_progress") sternbild::diamond(dl, c, half, L.accent, true);
    else if (status == "completed") sternbild::diamond(dl, c, half, L.secondary, true);
    else if (status == "failed" || status == "cancelled") {
        sternbild::diamond(dl, c, half, L.faint, false);
        dl->AddRectFilled(v2(c.x - half - 1.0f, std::round(c.y)), v2(c.x + half + 1.0f, std::round(c.y) + 1.0f), L.faint);
    } else sternbild::diamond(dl, c, half, L.faint, false);
}

}  // namespace

void SideChat::draw(const ImRect& pane_rect, const SpecLook& L) {
    const float s = L.text_scale;
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(s);
    const float tracking = std::round(L.small_size * 0.08f);
    const ChatMeasure measure = [&L, tracking](const std::string& t, ChatFont font) {
        switch (font) {
            case ChatFont::label: return extent(L.ui, L.ui_size, t).x;
            case ChatFont::key:
            case ChatFont::mono: return extent(L.mono, L.mono_size, t).x;
            case ChatFont::tag: return sternbild::text_tracked_width(L.ui, L.small_size, t.c_str(), tracking);
        }
        return 0.0f;
    };
    const WidgetRect P{pane_rect.Min.x, pane_rect.Min.y, pane_rect.GetWidth(), pane_rect.GetHeight()};
    if (P.empty()) return;
    const ChatPaneInput in = chat_pane_input(*this, P);
    const ChatPaneLayout lay = layout_chat_pane(in, m, measure);
    auto rr = [](const WidgetRect& r) { return ImRect(v2(r.x, r.y), v2(r.right(), r.bottom())); };
    const float ui_h = extent(L.ui, L.ui_size, "Ag").y;
    const float small_h = extent(L.ui, L.small_size, "Ag").y;
    const float mono_h = extent(L.mono, L.mono_size, "Ag").y;

    ImGui::PushID("##side.chat");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(pane_rect.Min, pane_rect.Max, true);
    dl->AddRectFilled(pane_rect.Min, pane_rect.Max, L.ground);
    dl->AddRectFilled(pane_rect.Min, v2(pane_rect.Min.x + 1.0f, pane_rect.Max.y), L.rule);

    // ---- the header
    {
        const ImRect h = rr(lay.header);
        dl->AddRectFilled(h.Min, h.Max, L.plate);
        dl->AddRectFilled(v2(h.Min.x, h.Max.y - 1.0f), h.Max, L.rule);
        draw_tag(dl, L, v2(lay.title.x, lay.title.y), lay.title.h, L.text, upper(in.title));
        if (!lay.permission.empty()) {
            // Caliper's mode for the agent, framed; bronze while it lets
            // commands run without a card. A click opens its menu.
            const ImRect pr = rr(lay.permission);
            ImGui::SetCursorScreenPos(pr.Min);
            if (ImGui::InvisibleButton("##chat.permission", pr.GetSize())) mode_menu_open_ = true;
            const bool hov = ImGui::IsItemHovered();
            const bool loose = in.permission == "Allow all" || in.permission == "Accept edits";
            if (hov) dl->AddRectFilled(pr.Min, pr.Max, L.plate_hover);
            dl->AddRect(pr.Min, pr.Max, in.permission == "Allow all" ? L.accent : L.rule, 0.0f, 0, 1.0f);
            draw_str(dl, L.ui, L.ui_size, v2(pr.Min.x + m.button_pad, pr.Min.y + (pr.GetHeight() - ui_h) * 0.5f),
                     loose ? L.accent : L.text, in.permission);
            // The menu's mark, ▾ as the picker's.
            const float t = std::round(3.0f * s);
            const ImVec2 c = v2(std::round(pr.Max.x - m.button_pad * 0.5f - t), std::round(pr.GetCenter().y));
            dl->AddTriangleFilled(v2(c.x - t, c.y - t * 0.5f), v2(c.x + t, c.y - t * 0.5f), v2(c.x, c.y + t * 0.6f), L.faint);
        }
        if (!lay.agent.empty())
            draw_str(dl, L.ui, L.ui_size, v2(lay.agent.x, lay.agent.y + (lay.agent.h - ui_h) * 0.5f), L.secondary, in.agent);
        if (!lay.note.empty()) {   // the GPU note, framed in bronze as the widget agent's settings show it
            const ImRect nr = rr(lay.note);
            dl->AddRect(nr.Min, nr.Max, L.accent, 0.0f, 0, 1.0f);
            const ImVec2 e = extent(L.ui, L.small_size, in.note);
            draw_str(dl, L.ui, L.small_size,
                     v2(std::round(nr.Min.x + (nr.GetWidth() - e.x) * 0.5f), std::round(nr.Min.y + (nr.GetHeight() - e.y) * 0.5f)),
                     L.accent, in.note);
        }
        if (!lay.version.empty())
            draw_str(dl, L.mono, L.mono_size, v2(lay.version.x, lay.version.y + (lay.version.h - mono_h) * 0.5f), L.faint,
                 in.version);
        if (!lay.mode.empty())   // the agent's mode, in bronze when it skips the agent's permission prompts
            draw_str(dl, L.mono, L.mono_size, v2(lay.mode.x, lay.mode.y + (lay.mode.h - mono_h) * 0.5f),
                     mode_skips_permission() ? L.accent : L.faint, in.mode);
        if (!lay.tag.empty()) {
            const bool live = state_ == ChatState::running && !stalled_;
            const bool alarm = state_ == ChatState::exited || state_ == ChatState::failed ||
                               state_ == ChatState::needs_login || stalled_;
            float x = lay.tag.x;
            const float half = std::round(3.5f * s);
            if (live || state_ == ChatState::starting) {
                sternbild::diamond(dl, v2(x + half, lay.tag.y + lay.tag.h * 0.5f), half, L.accent, live);
            }
            x += m.picker_mark;
            draw_tag(dl, L, v2(x, lay.tag.y), lay.tag.h, alarm ? L.accent : L.secondary, in.tag);
        }
        if (!lay.faults.empty()) {
            const ImRect f = rr(lay.faults);
            ImGui::SetCursorScreenPos(f.Min);
            if (ImGui::InvisibleButton("##chat.faults", f.GetSize())) diagnostics_open_ = !diagnostics_open_;
            const bool hov = ImGui::IsItemHovered();
            if (hov) dl->AddRectFilled(f.Min, f.Max, L.plate_hover);
            dl->AddRect(f.Min, f.Max, diagnostics_open_ ? L.accent : L.rule, 0.0f, 0, 1.0f);
            draw_tag(dl, L, v2(f.Min.x + m.button_pad, f.Min.y), f.GetHeight(), L.secondary, "FAULTS " + std::to_string(in.faults));
        }
        for (size_t i = 0; i < lay.header_buttons.size() && i < in.header_actions.size(); ++i) {
            const ChatAction a = in.header_actions[i];
            const std::string id = std::string("##chat.header.") + chat_action_label(a);
            if (house_button(id.c_str(), rr(lay.header_buttons[i]), chat_action_label(a), chat_action_key(a), false,
                             lay.keys, L))
                act(a);
        }
    }

    // ---- Allow all's banner: it stays while the session runs without cards
    if (!lay.banner.empty()) {
        const ImRect b = rr(lay.banner);
        dl->AddRectFilled(b.Min, b.Max, L.deep);
        dl->AddRectFilled(v2(b.Min.x, b.Min.y), v2(b.Min.x + std::round(3.0f * s), b.Max.y), L.accent);
        dl->AddRectFilled(v2(b.Min.x, b.Max.y - 1.0f), b.Max, L.accent);
        const float x = b.Min.x + m.pad + draw_tag(dl, L, v2(b.Min.x + m.pad, b.Min.y), b.GetHeight(), L.accent, "ALLOW ALL") +
                        2.0f * m.gap;
        const std::string note = permission_note();
        const std::string what = fit(L.ui, L.ui_size, note.empty() ? std::string("Saved, on until you end it") : note,
                                     lay.banner_end.x - m.gap - x);
        draw_str(dl, L.ui, L.ui_size, v2(x, b.Min.y + (b.GetHeight() - ui_h) * 0.5f), L.secondary, what);
        if (house_button("##chat.allow_all.end", rr(lay.banner_end), "End", "", false, false, L)) set_allow_all(false);
    }

    // ---- the conversation
    const ImRect conv = rr(lay.conversation);
    if (conv.GetHeight() > 1.0f && conv.GetWidth() > 1.0f) {
        ImGui::SetCursorScreenPos(conv.Min);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, v2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, 0);
        ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, 0);
        if (ImGui::BeginChild("##chat.conversation", conv.GetSize(), ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
            ImDrawList* cdl = ImGui::GetWindowDrawList();
            const bool at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
            const float pad = m.pad + std::round(2.0f * s);
            const float x_left = ImGui::GetCursorScreenPos().x + pad;
            const float w_full = std::max(40.0f, ImGui::GetContentRegionAvail().x - 2.0f * pad);
            float x0 = x_left, w = w_full;   // a group's members are drawn indented
            const float line = std::round(22.0f * s);
            const float gap = std::round(8.0f * s);
            // Bold is the text face drawn twice, the second a little to the right.
            const float bold_dx = std::max(0.5f, std::round(1.2f * s) * 0.5f);
            // The agent's text as chat_markdown.h lays it out: code on the
            // plate colour in the mono face, bullets and numbers in the
            // secondary colour.
            auto draw_markdown = [&](ImVec2 p, const MdLayout& md) {
                const float code_line = mono_h + std::round(3.0f * s);
                for (const MdPlate& pl : md.plates)
                    cdl->AddRectFilled(v2(p.x + pl.x, p.y + pl.y), v2(p.x + pl.x + pl.w, p.y + pl.y + pl.h), L.plate);
                for (const MdRun& r : md.runs) {
                    const float x = p.x + r.x, y = p.y + r.y;
                    if (r.style == MdStyle::code) {
                        const float lh = r.block ? code_line : ui_h;
                        if (!r.block)
                            cdl->AddRectFilled(v2(x - 1.0f, y + 1.0f), v2(x + r.w + 1.0f, y + ui_h - 1.0f), L.plate);
                        draw_str(cdl, L.mono, L.mono_size, v2(x, y + std::round((lh - mono_h) * 0.5f)), L.text, r.text);
                        continue;
                    }
                    const ImU32 c = r.marker ? L.secondary : L.text;
                    draw_str(cdl, L.ui, L.ui_size, v2(x, y), c, r.text);
                    if (r.style == MdStyle::bold) draw_str(cdl, L.ui, L.ui_size, v2(x + bold_dx, y), c, r.text);
                }
            };
            ImGui::Dummy(v2(1.0f, gap));
            // One item: draws with its top-left and returns its height.
            auto place = [&](float h, const std::function<void(ImVec2)>& fn) {
                const ImVec2 p = v2(x0, ImGui::GetCursorScreenPos().y);
                if (ImGui::IsRectVisible(v2(x0, p.y), v2(x0 + w, p.y + h))) fn(p);
                ImGui::SetCursorScreenPos(v2(x0 - pad, p.y));
                ImGui::Dummy(v2(w, h));
                ImGui::Dummy(v2(1.0f, gap));
            };

            if (state_ == ChatState::no_agent) {
                // No agent found: the agents as nouns, and one action.
                place(line, [&](ImVec2 p) { draw_tag(cdl, L, p, line, L.faint, "AGENTS"); });
                for (const ChatAgent& a : agents_) {
                    if (a.custom) continue;
                    // The name and whether it was found, as the picker shows them; never a command line.
                    place(line, [&](ImVec2 p) {
                        draw_str(cdl, L.ui, L.ui_size, v2(p.x, p.y + (line - ui_h) * 0.5f), a.found ? L.text : L.secondary, a.title);
                        const char* word = a.found ? "FOUND" : "NOT FOUND";
                        const float nw = sternbild::text_tracked_width(L.ui, L.small_size, word, tracking);
                        const float nx = p.x + std::max(std::round(140.0f * s), extent(L.ui, L.ui_size, a.title).x + gap);
                        if (nx + nw <= p.x + w) draw_tag(cdl, L, v2(nx, p.y), line, L.faint, word);
                    });
                }
                const float bw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.ui_size, chat_action_label(ChatAction::custom)).x +
                                           std::round(8.0f * s));
                const ImVec2 p = v2(x0, ImGui::GetCursorScreenPos().y + gap);
                if (house_button("##chat.custom.open", ImRect(p, v2(p.x + bw, p.y + m.button)),
                                 chat_action_label(ChatAction::custom), "", true, true, L))
                    custom_open_ = true;
                ImGui::SetCursorScreenPos(v2(x0 - pad, p.y));
                ImGui::Dummy(v2(w, m.button + gap));
            }

            auto draw_item = [&](size_t idx) {
                ChatItem& it = items_[idx];
                ImGui::PushID((int)idx);
                switch (it.kind) {
                    case ChatItem::Kind::person: {
                        const float tw = w - 2.0f * m.pad;
                        const float th = it.text.empty() ? 0.0f : extent(L.ui, L.ui_size, it.text, tw).y;
                        const float ch = it.chips.empty() ? 0.0f : m.chip + (it.text.empty() ? 0.0f : m.gap);
                        place(m.pad * 2.0f + th + ch, [&](ImVec2 p) {
                            const float h = m.pad * 2.0f + th + ch;
                            cdl->AddRectFilled(p, v2(p.x + w, p.y + h), L.plate);
                            if (!it.text.empty()) draw_str(cdl, L.ui, L.ui_size, v2(p.x + m.pad, p.y + m.pad), L.text, it.text, tw);
                            float cx = p.x + m.pad;
                            const float cy = p.y + m.pad + th + (it.text.empty() ? 0.0f : m.gap);
                            for (size_t c = 0; c < it.chips.size(); ++c) {
                                const float cw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.small_size, it.chips[c]).x);
                                if (cx + cw > p.x + w - m.pad) {
                                    draw_str(cdl, L.ui, L.small_size, v2(cx, cy + (m.chip - small_h) * 0.5f), L.faint,
                                         "+" + std::to_string(it.chips.size() - c));
                                    break;
                                }
                                cdl->AddRect(v2(cx, cy), v2(cx + cw, cy + m.chip), L.rule, 0.0f, 0, 1.0f);
                                draw_str(cdl, L.ui, L.small_size, v2(cx + m.button_pad, cy + (m.chip - small_h) * 0.5f), L.secondary,
                                     it.chips[c]);
                                cx += cw + m.gap;
                            }
                        });
                        break;
                    }
                    case ChatItem::Kind::agent: {
                        // The agent's text with its Markdown rendered (chat_markdown.h);
                        // the item keeps the text as the agent sent it.
                        MdDocument& md = markdown_[it.serial];
                        // An agent's text only grows, so its size and its end say
                        // whether it changed, without comparing it whole each frame.
                        const size_t tail = std::min<size_t>(64, it.text.size());
                        if (md.text().size() != it.text.size() ||
                            md.text().compare(md.text().size() - tail, tail, it.text, it.text.size() - tail, tail) != 0)
                            md.set_text(it.text);
                        MdMetrics mm;
                        mm.line = ui_h;
                        mm.code_line = mono_h + std::round(3.0f * s);
                        mm.gap = std::round(6.0f * s);
                        mm.indent = std::round(14.0f * s);
                        mm.marker = std::round(14.0f * s);
                        mm.code_pad = std::round(6.0f * s);
                        mm.bold_extra = bold_dx;
                        const MdMeasure measure = [&](const std::string& t, MdFont f) {
                            return f == MdFont::mono ? extent(L.mono, L.mono_size, t).x : extent(L.ui, L.ui_size, t).x;
                        };
                        const MdLayout& lay = md.layout(w, mm, measure);
                        place(lay.height, [&](ImVec2 p) { draw_markdown(p, lay); });
                        break;
                    }
                    case ChatItem::Kind::thought: {
                        const float bx = sternbild::text_tracked_width(L.ui, L.small_size, "THOUGHT", tracking) + gap;
                        const float th = it.expanded ? extent(L.ui, L.small_size, trimmed(it.text), w).y : 0.0f;
                        place(line + th, [&](ImVec2 p) {
                            ImGui::SetCursorScreenPos(p);
                            if (ImGui::InvisibleButton("##thought", v2(w, line))) it.expanded = !it.expanded;
                            const bool hov = ImGui::IsItemHovered();
                            draw_tag(cdl, L, p, line, hov ? L.secondary : L.faint, "THOUGHT");
                            if (!it.expanded)
                                draw_str(cdl, L.ui, L.small_size, v2(p.x + bx, p.y + (line - small_h) * 0.5f), L.faint,
                                     fit(L.ui, L.small_size, first_line(it.text), w - bx));
                            else
                                draw_str(cdl, L.ui, L.small_size, v2(p.x, p.y + line), L.faint, trimmed(it.text), w);
                        });
                        break;
                    }
                    case ChatItem::Kind::tool: {
                        place(line, [&](ImVec2 p) {
                            const float half = std::round(3.5f * s);
                            status_mark(cdl, L, v2(p.x + half + 1.0f, p.y + line * 0.5f), it.tool.status);
                            float x = p.x + 2.0f * half + gap;
                            const std::string word = chat_tool_status_word(it.tool.status);
                            const float ww = extent(L.mono, L.mono_size, word).x;
                            const bool show_word = w > std::round(220.0f * s);
                            const float right = p.x + w - (show_word ? ww + gap : 0.0f);
                            if (!it.command.empty()) {
                                const std::string c = fit(L.mono, L.mono_size, it.command, right - x);
                                draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.text, c);
                                x += extent(L.mono, L.mono_size, c).x + gap;
                            }
                            // The title, unless it only names the tool the command already shows.
                            const bool redundant = !it.command.empty() &&
                                                   it.tool.title.find(caliper_builder::command_tool_name(it.command)) != std::string::npos &&
                                                   it.tool.title.size() <= caliper_builder::command_tool_name(it.command).size() + 32;
                            if (!redundant && !it.tool.title.empty() && right - x > 20.0f)
                                draw_str(cdl, L.ui, L.small_size, v2(x, p.y + (line - small_h) * 0.5f),
                                     it.command.empty() ? L.secondary : L.faint, fit(L.ui, L.small_size, it.tool.title, right - x));
                            if (show_word)
                                draw_str(cdl, L.mono, L.mono_size, v2(p.x + w - ww, p.y + (line - mono_h) * 0.5f),
                                     it.tool.status == "in_progress" ? L.secondary : L.faint, word);
                        });
                        break;
                    }
                    case ChatItem::Kind::plan: {
                        float h = line;
                        for (const auto& e : it.plan) h += std::max(line, extent(L.ui, L.small_size, e.content, w - line).y);
                        place(h, [&](ImVec2 p) {
                            draw_tag(cdl, L, p, line, L.faint, "PLAN");
                            float y = p.y + line;
                            for (const auto& e : it.plan) {
                                const float eh = std::max(line, extent(L.ui, L.small_size, e.content, w - line).y);
                                status_mark(cdl, L, v2(p.x + std::round(4.5f * s), y + line * 0.5f), e.status);
                                const ImU32 c = e.status == "completed" ? L.faint : e.status == "in_progress" ? L.text : L.secondary;
                                draw_str(cdl, L.ui, L.small_size, v2(p.x + line, y + (line - small_h) * 0.5f), c, e.content, w - line);
                                y += eh;
                            }
                        });
                        break;
                    }
                    case ChatItem::Kind::permission: {
                        // The tag and the command or the tool's title; while open, the
                        // file, the compact diff as the inline surface and the approval
                        // cards draw it (diff_card.h) or the unified diff on demand and
                        // for a file that is not a spec, and the agent's options.
                        const bool open = it.answer.empty();
                        const std::string label = chat_permission_label(it);
                        const bool compact = !it.compact.empty() && !it.expanded;
                        std::string full;
                        int full_lines = 0;
                        if (open && !compact && !it.diff.empty())
                            for (const std::string& l : split_lines(it.diff)) {
                                if (l.rfind("---", 0) == 0 || l.rfind("+++", 0) == 0) continue;
                                if (full_lines < kMaxDiffLines) full += l + "\n";
                                ++full_lines;
                            }
                        const int total = compact ? (int)it.compact.size() : full_lines;
                        const int cap = compact ? kMaxCompactLines : kMaxDiffLines;
                        const int shown = std::min(total, cap);
                        const bool folded = total > shown;
                        const float dline = std::round(18.0f * s);
                        float h = line;
                        if (open) {
                            if (!it.diff_path.empty()) h += dline;
                            h += dline * (float)(shown + (folded ? 1 : 0));
                            h += m.gap + m.button + m.pad;
                        }
                        place(h + (open ? m.pad : 0.0f), [&](ImVec2 p) {
                            const ImVec2 q = v2(p.x + w, p.y + h + (open ? m.pad : 0.0f));
                            if (open) {
                                cdl->AddRectFilled(p, q, L.deep);
                                cdl->AddRect(p, q, L.accent, 0.0f, 0, 1.0f);
                            }
                            const float ix = p.x + (open ? m.pad : 0.0f);
                            const float iw = w - (open ? 2.0f * m.pad : 0.0f);
                            float y = p.y + (open ? m.pad * 0.5f : 0.0f);
                            const float tw = draw_tag(cdl, L, v2(ix, y), line, open ? L.secondary : L.faint, "PERMISSION");
                            const float x = ix + tw + gap;
                            if (!it.command.empty())
                                draw_str(cdl, L.mono, L.mono_size, v2(x, y + (line - mono_h) * 0.5f), open ? L.text : L.secondary,
                                         fit(L.mono, L.mono_size, label, ix + iw - x));
                            else
                                draw_str(cdl, L.ui, L.ui_size, v2(x, y + (line - ui_h) * 0.5f), open ? L.text : L.secondary,
                                         fit(L.ui, L.ui_size, label, ix + iw - x));
                            if (!open) {
                                const float ax = p.x + w - extent(L.ui, L.small_size, it.answer).x;
                                draw_str(cdl, L.ui, L.small_size, v2(ax, y + (line - small_h) * 0.5f), L.faint, it.answer);
                                return;
                            }
                            y += line;
                            if (!it.diff_path.empty()) {
                                draw_str(cdl, L.mono, L.mono_size, v2(ix, y + (dline - mono_h) * 0.5f), L.faint,
                                         fit(L.mono, L.mono_size, it.diff_path, iw));
                                y += dline;
                            }
                            const float dh = dline * (float)(shown + (folded ? 1 : 0));
                            if (compact) {
                                draw_compact_diff(cdl, L, ImRect(v2(ix, y), v2(ix + iw, y + dh)), it.compact, shown, folded);
                            } else if (shown > 0) {
                                cdl->PushClipRect(v2(ix, y), v2(ix + iw, y + dh), true);
                                ImGui::SetCursorScreenPos(v2(ix, y));
                                draw_unified_diff(L, full, dline);   // diff_card.h
                                cdl->PopClipRect();
                                if (folded)
                                    draw_str(cdl, L.mono, L.mono_size, v2(ix, y + dline * (float)shown + (dline - mono_h) * 0.5f), L.faint,
                                             "+" + std::to_string(total - shown));
                            }
                            y += dh + m.gap;
                            float bx = ix;
                            bool first_allow = true;
                            for (size_t o = 0; o < it.options.size(); ++o) {
                                const auto& opt = it.options[o];
                                const std::string name = opt.name.empty() ? opt.id : opt.name;
                                const float bw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.ui_size, name).x + std::round(8.0f * s));
                                if (bx + bw > ix + iw) break;
                                const bool primary = first_allow && opt.kind.rfind("allow", 0) == 0;
                                if (primary) first_allow = false;
                                const std::string bid = "##perm." + std::to_string(o);
                                if (house_button(bid.c_str(), ImRect(v2(bx, y), v2(bx + bw, y + m.button)), name, "", primary,
                                                 false, L))
                                    answer_permission(it.permission, opt.id);
                                bx += bw + m.gap;
                            }
                            // Diff: the full unified diff in place of the compact one.
                            if (!it.compact.empty()) {
                                const float dw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.ui_size, "Diff").x + std::round(8.0f * s));
                                const ImRect db(v2(ix + iw - dw, y), v2(ix + iw, y + m.button));
                                if (db.Min.x >= bx) {
                                    if (house_button("##perm.diff", db, "Diff", "", false, false, L)) it.expanded = !it.expanded;
                                    if (it.expanded) cdl->AddRect(db.Min, db.Max, L.accent, 0.0f, 0, 1.0f);
                                }
                            }
                        });
                        break;
                    }
                    case ChatItem::Kind::approval: {
                        // One row that points at the approval card in the strip.
                        place(line, [&](ImVec2 p) {
                            const bool waiting = it.answer.empty();
                            ImGui::SetCursorScreenPos(p);
                            if (ImGui::InvisibleButton("##approval", v2(w, line)) && hooks_.show_approval) hooks_.show_approval(it.approval);
                            const bool hov = ImGui::IsItemHovered();
                            if (hov && waiting) cdl->AddRectFilled(p, v2(p.x + w, p.y + line), L.plate_hover);
                            const float tw = draw_tag(cdl, L, p, line, waiting ? L.accent : L.faint, "APPROVAL");
                            const float x = p.x + tw + gap;
                            const std::string word = waiting ? "waiting" : it.answer;
                            const float ww = extent(L.mono, L.mono_size, word).x;
                            draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), waiting ? L.text : L.secondary,
                                     fit(L.mono, L.mono_size, it.command, p.x + w - ww - gap - x));
                            draw_str(cdl, L.mono, L.mono_size, v2(p.x + w - ww, p.y + (line - mono_h) * 0.5f),
                                     waiting ? L.secondary : L.faint, word);
                        });
                        break;
                    }
                    case ChatItem::Kind::stop: {
                        place(line, [&](ImVec2 p) {
                            float x = p.x + draw_tag(cdl, L, p, line, it.text == "FAILED" ? L.accent : L.faint, it.text) + gap;
                            if (!it.detail.empty()) {
                                const std::string d = fit(L.mono, L.mono_size, it.detail, p.x + w - x);
                                draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.faint, d);
                                x += extent(L.mono, L.mono_size, d).x + gap;
                            }
                            if (x < p.x + w) cdl->AddRectFilled(v2(x, std::round(p.y + line * 0.5f)), v2(p.x + w, std::round(p.y + line * 0.5f) + 1.0f), L.rule_faint);
                        });
                        break;
                    }
                    case ChatItem::Kind::applied: {
                        // A command that ran without a card: what allowed it, and Undo
                        // while it is the newest step.
                        place(line, [&](ImVec2 p) {
                            const bool undone = !it.answer.empty();
                            const bool can = !undone && hooks_.can_undo && hooks_.can_undo(it.seq);
                            const float tw = draw_tag(cdl, L, p, line, undone ? L.faint : L.accent, "APPLIED");
                            float x = p.x + tw + gap;
                            const float uw = std::ceil(2.0f * m.button_pad + extent(L.ui, L.ui_size, "Undo").x);
                            const float right = p.x + w - (undone ? extent(L.mono, L.mono_size, "undone").x : uw) - gap;
                            const std::string cmd = fit(L.mono, L.mono_size, it.command, right - x);
                            draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), undone ? L.secondary : L.text, cmd);
                            x += extent(L.mono, L.mono_size, cmd).x + gap;
                            if (x < right)
                                draw_str(cdl, L.ui, L.small_size, v2(x, p.y + (line - small_h) * 0.5f), L.faint,
                                         fit(L.ui, L.small_size, it.text, right - x));
                            if (undone) {
                                const float ww = extent(L.mono, L.mono_size, "undone").x;
                                draw_str(cdl, L.mono, L.mono_size, v2(p.x + w - ww, p.y + (line - mono_h) * 0.5f), L.faint, "undone");
                            } else if (house_button("##applied.undo", ImRect(v2(p.x + w - uw, p.y + (line - m.button) * 0.5f),
                                                                            v2(p.x + w, p.y + (line + m.button) * 0.5f)),
                                                    "Undo", "", false, false, L, can)) {
                                undo_applied(it.serial);
                            }
                        });
                        break;
                    }
                    case ChatItem::Kind::notice: {
                        // The tag, the error or the exit status, and the lines: the
                        // agent's stderr tail or its login methods. Restart is in the header.
                        const bool login = it.text == "LOGIN";
                        const float lh0 = std::round(18.0f * s);
                        auto mono_line = [&](const std::string& l) { return it.text == "EXITED" || (login && l.rfind("$ ", 0) == 0); };
                        float h = line;
                        auto line_h = [&](const std::string& l) {
                            if (it.text == "EXITED") return lh0;
                            return std::max(lh0, mono_line(l) ? extent(L.mono, L.mono_size, l, w).y : extent(L.ui, L.small_size, l, w).y);
                        };
                        for (const std::string& l : it.lines) h += line_h(l);
                        place(h, [&](ImVec2 p) {
                            const float x = p.x + draw_tag(cdl, L, p, line, L.accent, it.text) + gap;
                            if (!it.detail.empty())
                                draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.faint,
                                         fit(L.mono, L.mono_size, it.detail, p.x + w - x));
                            float y = p.y + line;
                            for (const std::string& l : it.lines) {
                                const bool mono = mono_line(l);
                                const float lh = line_h(l);
                                if (mono && !login)   // the agent's last stderr line, elided to one line
                                    draw_str(cdl, L.mono, L.mono_size, v2(p.x, y + (lh0 - mono_h) * 0.5f), L.faint, fit(L.mono, L.mono_size, l, w));
                                else if (mono) draw_str(cdl, L.mono, L.mono_size, v2(p.x, y + (lh0 - mono_h) * 0.5f), L.text, l, w);
                                else draw_str(cdl, L.ui, L.small_size, v2(p.x, y + (lh0 - small_h) * 0.5f), L.secondary, l, w);
                                y += lh;
                            }
                        });
                        break;
                    }
                }
                ImGui::PopID();
            };
            // The rows: the items, with runs of one command folded into a group.
            for (const ChatRow& row : chat_rows(items_, group_open_)) {
                if (row.kind != ChatRow::Kind::group) {
                    draw_item(row.item);
                    continue;
                }
                ImGui::PushID(row.key.c_str());
                place(line, [&](ImVec2 p) {
                    ImGui::SetCursorScreenPos(p);
                    if (ImGui::InvisibleButton("##group", v2(w, line))) group_open_[row.key] = !row.expanded;
                    if (ImGui::IsItemHovered()) cdl->AddRectFilled(p, v2(p.x + w, p.y + line), L.plate_hover);
                    const float half = std::round(3.5f * s);
                    status_mark(cdl, L, v2(p.x + half + 1.0f, p.y + line * 0.5f), row.mark);
                    float x = p.x + 2.0f * half + gap;
                    // The fold's mark: right when closed, down when open.
                    const float t = std::round(3.0f * s);
                    const ImVec2 c = v2(std::round(x + t), std::round(p.y + line * 0.5f));
                    if (row.expanded) cdl->AddTriangleFilled(v2(c.x - t, c.y - t * 0.6f), v2(c.x + t, c.y - t * 0.6f), v2(c.x, c.y + t * 0.7f), L.faint);
                    else cdl->AddTriangleFilled(v2(c.x - t * 0.6f, c.y - t), v2(c.x - t * 0.6f, c.y + t), v2(c.x + t * 0.7f, c.y), L.faint);
                    x += 2.0f * t + gap;
                    const std::string summary = chat_group_summary(row);
                    const float sw = extent(L.mono, L.mono_size, summary).x;
                    const float right = p.x + w - sw - gap;
                    ImFont* f = row.caliper ? L.mono : L.ui;
                    const float fs = row.caliper ? L.mono_size : L.small_size;
                    const float fh = row.caliper ? mono_h : small_h;
                    const std::string name = fit(f, fs, row.command, right - x);
                    draw_str(cdl, f, fs, v2(x, p.y + (line - fh) * 0.5f), row.caliper ? L.text : L.secondary, name);
                    x += extent(f, fs, name).x;
                    if (row.count > 1 && x < right) {
                        const std::string times = " \xC3\x97" + std::to_string(row.count);   // ×
                        draw_str(cdl, L.mono, L.mono_size, v2(x, p.y + (line - mono_h) * 0.5f), L.faint, fit(L.mono, L.mono_size, times, right - x));
                    }
                    draw_str(cdl, L.mono, L.mono_size, v2(p.x + w - sw, p.y + (line - mono_h) * 0.5f),
                             row.alert ? L.accent : row.mark == "in_progress" ? L.secondary : L.faint, summary);
                });
                if (row.expanded) {
                    x0 = x_left + std::round(16.0f * s);
                    w = w_full - std::round(16.0f * s);
                    for (size_t mi : row.members) draw_item(mi);
                    x0 = x_left;
                    w = w_full;
                }
                ImGui::PopID();
            }
            if (follow_ || items_.size() != drawn_items_) {
                if (at_bottom || follow_) ImGui::SetScrollHereY(1.0f);
            }
            drawn_items_ = items_.size();
            follow_ = at_bottom;
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);
    }

    // ---- the draft's chips
    if (!lay.chips.empty()) {
        for (int i = 0; i < lay.chips_shown; ++i) {
            const ImRect c = rr(lay.chip_rects[(size_t)i]);
            dl->AddRect(c.Min, c.Max, L.rule, 0.0f, 0, 1.0f);
            draw_str(dl, L.ui, L.ui_size, v2(c.Min.x + m.button_pad, c.Min.y + (c.GetHeight() - ui_h) * 0.5f), L.secondary,
                 fit(L.ui, L.ui_size, in.chips[(size_t)i], c.GetWidth() - 2.0f * m.button_pad));
        }
        if (lay.chips_shown < (int)in.chips.size()) {
            const float x = lay.chip_rects.empty() ? lay.chips.x : lay.chip_rects.back().right() + m.gap;
            draw_str(dl, L.ui, L.ui_size, v2(x, lay.chips.y + (lay.chips.h - ui_h) * 0.5f), L.faint,
                 "+" + std::to_string(in.chips.size() - (size_t)lay.chips_shown));
        }
        if (house_button("##chat.clear", rr(lay.clear), chat_action_label(ChatAction::clear), "", false, false, L))
            clear_draft();
    }

    // ---- the input row
    {
        const ImRect row = rr(lay.input_row);
        const bool usable = state_ != ChatState::no_agent;
        dl->AddRectFilled(row.Min, row.Max, L.deep);
        const ImGuiID input_id = ImGui::GetID("##chat.input");
        const bool typing = ImGui::GetActiveID() == input_id;
        dl->AddRect(row.Min, row.Max, typing ? L.accent : L.rule, 0.0f, 0, 1.0f);

        // The picker: the agent's name and its ▾ mark.
        const ImRect pk = rr(lay.picker);
        ImGui::SetCursorScreenPos(pk.Min);
        if (ImGui::InvisibleButton("##chat.picker", pk.GetSize())) picker_open_ = true;
        if (ImGui::IsItemHovered()) dl->AddRectFilled(pk.Min, pk.Max, L.plate_hover);
        dl->AddRect(pk.Min, pk.Max, L.rule, 0.0f, 0, 1.0f);
        float px = pk.Min.x + m.button_pad;
        if (lay.picker_label) {
            draw_str(dl, L.ui, L.ui_size, v2(px, pk.Min.y + (pk.GetHeight() - ui_h) * 0.5f), usable ? L.text : L.faint, in.picker);
            px += extent(L.ui, L.ui_size, in.picker).x + std::round(4.0f * s);
        }
        {
            const float t = std::round(3.5f * s);
            const ImVec2 c = v2(std::round(px + t + 1.0f), std::round(pk.GetCenter().y));
            dl->AddTriangleFilled(v2(c.x - t, c.y - t * 0.5f), v2(c.x + t, c.y - t * 0.5f), v2(c.x, c.y + t * 0.6f), L.secondary);
        }

        // The input: one line, more with ⇧⏎ or ⌘⏎ (up to the layout's maximum).
        const ImRect inp = rr(lay.input);
        if (!usable) ImGui::BeginDisabled();
        ImGui::SetCursorScreenPos(v2(inp.Min.x, inp.Min.y + std::round((m.row - ui_h) * 0.5f) - 2.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, 0);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, 0);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, 0);
        ImGui::PushStyleColor(ImGuiCol_Text, L.text);
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, sternbild::with_alpha(L.accent, 90));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, v2(0, 2));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
        ImGui::PushFont(L.ui, L.ui_size);
        // Shift-Tab while the input has the keyboard cycles Caliper's mode,
        // and the keyboard stays in the input.
        if (input_active_ && ImGui::IsKeyChordPressed(ImGuiMod_Shift | ImGuiKey_Tab)) {
            cycle_permission_mode();
            focus_input_ = true;
        }
        if (focus_input_ && usable) {
            ImGui::SetKeyboardFocusHere();
            focus_input_ = false;
        }
        if (text_.capacity() < 64) text_.reserve(64);
        const float ih = std::max(ui_h + 4.0f, inp.GetHeight() - std::round((m.row - ui_h) * 0.5f) * 2.0f + 4.0f);
        const bool entered = ImGui::InputTextMultiline(
            "##chat.input", const_cast<char*>(text_.c_str()), text_.capacity() + 1, v2(inp.GetWidth(), ih),
            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine |
                ImGuiInputTextFlags_CallbackResize | ImGuiInputTextFlags_NoHorizontalScroll,
            resize_callback, &text_);
        input_active_ = ImGui::IsItemActive();
        // While the input has the keyboard, Tab is the chat's (Shift-Tab
        // above), not ImGui's tabbing to another widget.
        if (input_active_) ImGui::SetKeyOwner(ImGuiKey_Tab, ImGui::GetItemID());
        ImGui::PopFont();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(5);
        if (!usable) ImGui::EndDisabled();
        if (entered && usable) {
            send();
            focus_input_ = true;
        }

        const ChatAction a = in.input_action;
        bool enabled = usable;
        if (a == ChatAction::send) enabled = usable && (!trimmed(text_).empty() || !draft_.blocks.empty()) &&
                                             state_ != ChatState::starting;
        if (house_button("##chat.send", rr(lay.send), chat_action_label(a), chat_action_key(a), a == ChatAction::send,
                         lay.keys, L, enabled))
            act(a);
    }
    dl->PopClipRect();

    // ---- Caliper's mode menu: the three modes, and Allow all under a rule
    if (mode_menu_open_) {
        ImGui::OpenPopup("##chat.modes");
        mode_menu_open_ = false;
    }
    ImGui::SetNextWindowPos(v2(lay.permission.x, lay.permission.bottom() + 2.0f), ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(m.pad, m.pad));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, L.deep);
    ImGui::PushStyleColor(ImGuiCol_Border, L.rule);
    if (ImGui::BeginPopup("##chat.modes")) {
        ImGui::PushFont(L.ui, L.ui_size);
        const float cw = std::round(240.0f * s), lh = std::round(22.0f * s);
        const std::string now = permission_word();
        const char* names[] = {"ask", "accept_edits", "read_only"};
        const char* words[] = {"Ask", "Accept edits", "Read only"};
        for (int i = 0; i < 3; ++i) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            if (ImGui::Selectable((std::string("##mode.") + names[i]).c_str(), now == words[i], 0, v2(cw, lh)) &&
                hooks_.set_permission_mode)
                hooks_.set_permission_mode(names[i]);
            draw_str(ImGui::GetWindowDrawList(), L.ui, L.ui_size, v2(p.x + std::round(4.0f * s), p.y + (lh - ui_h) * 0.5f),
                     now == words[i] ? L.accent : L.text, words[i]);
        }
        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
        const ImVec2 r = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddRectFilled(r, v2(r.x + cw, r.y + 1.0f), L.rule_faint);
        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const bool on = now == "Allow all";
            if (ImGui::Selectable("##mode.allow_all", on, 0, v2(cw, lh))) set_allow_all(!on);
            draw_str(ImGui::GetWindowDrawList(), L.ui, L.ui_size, v2(p.x + std::round(4.0f * s), p.y + (lh - ui_h) * 0.5f),
                     L.accent, on ? "End Allow all" : "Allow all");
        }
        ImGui::PopFont();
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();

    // ---- the picker's list: the agents found, the others in the faint tone, and the custom command
    if (picker_open_) {
        ImGui::OpenPopup("##chat.agents");
        picker_open_ = false;
    }
    const ImRect pk = rr(lay.picker);
    ImGui::SetNextWindowPos(v2(pk.Min.x, pk.Min.y - 2.0f), ImGuiCond_Always, v2(0.0f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(m.pad, m.pad));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, L.deep);
    ImGui::PushStyleColor(ImGuiCol_Border, L.rule);
    if (ImGui::BeginPopup("##chat.agents")) {
        ImGui::PushFont(L.ui, L.ui_size);
        const float cw = std::round(260.0f * s);
        for (const ChatAgent& ag : agents_) {
            ImGui::PushID(ag.id.c_str());
            const bool chosen = ag.found && ag.id == options_.settings.agent;
            const ImVec2 p = ImGui::GetCursorScreenPos();
            if (!ag.found) ImGui::BeginDisabled();
            if (ImGui::Selectable("##agent", chosen, 0, v2(cw, std::round(22.0f * s)))) choose_agent(ag.id);
            if (!ag.found) ImGui::EndDisabled();
            ImDrawList* pdl = ImGui::GetWindowDrawList();
            const float lh = std::round(22.0f * s);
            draw_str(pdl, L.ui, L.ui_size, v2(p.x + std::round(4.0f * s), p.y + (lh - ui_h) * 0.5f),
                 ag.found ? (chosen ? L.accent : L.text) : L.faint, ag.title);
            if (!ag.found) {   // the name and whether it was found; never its command line
                const float nw = sternbild::text_tracked_width(L.ui, L.small_size, "NOT FOUND", tracking);
                draw_tag(pdl, L, v2(p.x + cw - nw - std::round(4.0f * s), p.y), lh, L.faint, "NOT FOUND");
            }
            ImGui::PopID();
        }
        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
        ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetCursorScreenPos(), v2(ImGui::GetCursorScreenPos().x + cw, ImGui::GetCursorScreenPos().y + 1.0f), L.rule_faint);
        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
        if (ImGui::Selectable(chat_action_label(ChatAction::custom), false, 0, v2(cw, std::round(22.0f * s)))) custom_open_ = true;
        // Local model: the chosen agent's mark, a box filled in bronze while it
        // is set; the popup stays open so that the mark shows.
        {
            const bool can = agent() != nullptr, on = local_model();
            const float lh = std::round(22.0f * s);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            if (!can) ImGui::BeginDisabled();
            if (ImGui::Selectable("##chat.local_model", false, ImGuiSelectableFlags_NoAutoClosePopups, v2(cw, lh)))
                set_local_model(!on);
            if (!can) ImGui::EndDisabled();
            ImDrawList* pdl = ImGui::GetWindowDrawList();
            const float side = std::round(11.0f * s);
            const ImVec2 b(p.x + std::round(4.0f * s), std::round(p.y + (lh - side) * 0.5f));
            pdl->AddRect(b, v2(b.x + side, b.y + side), can ? L.secondary : L.faint, 0.0f, 0, 1.0f);
            if (on) {
                const float in = std::round(3.0f * s);
                pdl->AddRectFilled(v2(b.x + in, b.y + in), v2(b.x + side - in, b.y + side - in), L.accent);
            }
            draw_str(pdl, L.ui, L.ui_size, v2(b.x + side + std::round(8.0f * s), p.y + (lh - ui_h) * 0.5f),
                     !can ? L.faint : on ? L.accent : L.text, "Local model");
        }
        ImGui::PopFont();
        ImGui::EndPopup();
    }

    // ---- the custom command: a name and one command line
    if (custom_open_) {
        ImGui::OpenPopup("##chat.custom");
        if (custom_line_.empty()) custom_line_ = options_.settings.custom_command;
        if (custom_name_.empty()) custom_name_ = options_.settings.custom_name;
        custom_open_ = false;
    }
    ImGui::SetNextWindowPos(v2(pane_rect.Min.x + m.pad, lay.input_row.y - m.gap), ImGuiCond_Always, v2(0.0f, 1.0f));
    if (ImGui::BeginPopup("##chat.custom")) {
        ImGui::PushFont(L.ui, L.ui_size);
        const float fw = std::max(std::round(200.0f * s), P.w - 4.0f * m.pad);
        sternbild::label("Name");
        ImGui::PushItemWidth(fw);
        if (custom_name_.capacity() < 32) custom_name_.reserve(32);
        ImGui::InputText("##custom.name", const_cast<char*>(custom_name_.c_str()), custom_name_.capacity() + 1,
                         ImGuiInputTextFlags_CallbackResize, resize_callback, &custom_name_);
        sternbild::label("Command");
        if (custom_line_.capacity() < 64) custom_line_.reserve(64);
        const bool done = ImGui::InputText("##custom.line", const_cast<char*>(custom_line_.c_str()), custom_line_.capacity() + 1,
                                           ImGuiInputTextFlags_CallbackResize | ImGuiInputTextFlags_EnterReturnsTrue,
                                           resize_callback, &custom_line_);
        ImGui::PopItemWidth();
        if (!custom_error_.ok()) ImGui::TextColored(sternbild::to_vec4(L.faint), "%s", custom_error_.code.c_str());
        if (sternbild::primary_button("Save") || done) {
            custom_error_ = set_custom_agent(custom_name_, custom_line_);
            if (custom_error_.ok()) ImGui::CloseCurrentPopup();
        }
        ImGui::PopFont();
        ImGui::EndPopup();
    }

    // ---- the diagnostics: the protocol faults, out of the conversation
    if (diagnostics_open_ && !faults_.empty()) {
        ImGui::SetNextWindowPos(v2(lay.faults.right(), lay.header.bottom() + 2.0f), ImGuiCond_Always, v2(1.0f, 0.0f));
        ImGui::SetNextWindowSize(v2(std::min(P.w - 2.0f * m.pad, std::round(420.0f * s)), 0.0f));
        if (ImGui::Begin("##chat.diagnostics", &diagnostics_open_,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::PushFont(L.mono, L.mono_size);
            const size_t from = faults_.size() > 12 ? faults_.size() - 12 : 0;
            for (size_t i = from; i < faults_.size(); ++i) {
                ImGui::TextColored(sternbild::to_vec4(L.secondary), "%s", faults_[i].code.c_str());
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(sternbild::to_vec4(L.faint), "%s", faults_[i].message.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::PopFont();
        }
        ImGui::End();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace caliper_host
