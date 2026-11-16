// builder_pane_draw.cpp: the builder pane's ImGui side (builder_pane.h), on
// the frame thread in the house style: the Sources and Primitives palettes,
// Properties, the spec text strip and the canvas's overlays, with the mouse
// and the keys turned into the pane's gestures.
#include "builder/builder_pane.h"

#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"

#include <caliper_builder/session/spec_edit.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>

namespace caliper_host {

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using nlohmann::json;
using spec::Primitive;

namespace {

ImVec2 v2(float x, float y) { return ImVec2(x, y); }
ImRect rr(const SpecRect& r) { return ImRect(v2(r.x, r.y), v2(r.right(), r.bottom())); }
SpecRect sr(const ImRect& r) { return {r.Min.x, r.Min.y, r.GetWidth(), r.GetHeight()}; }

ImVec2 extent(ImFont* f, float size, const std::string& s) {
    if (!f) f = ImGui::GetFont();
    return f->CalcTextSizeA(size, FLT_MAX, 0.0f, s.c_str(), s.c_str() + s.size());
}

std::string fit(ImFont* f, float size, const std::string& s, float w) {
    if (extent(f, size, s).x <= w) return s;
    const std::string dots = "\xE2\x80\xA6";
    if (extent(f, size, dots).x > w) return {};
    std::string t = s;
    while (!t.empty()) {
        t.pop_back();
        while (!t.empty() && ((unsigned char)t.back() & 0xC0) == 0x80) t.pop_back();
        if (extent(f, size, t + dots).x <= w) return t + dots;
    }
    return dots;
}

void put(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 c, const std::string& s) {
    if (s.empty()) return;
    dl->AddText(f ? f : ImGui::GetFont(), size, v2(std::round(p.x), std::round(p.y)), c, s.c_str(), s.c_str() + s.size());
}

std::string upper(std::string s) {
    for (char& c : s)
        if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    return s;
}

int grow(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        auto* s = static_cast<std::string*>(data->UserData);
        s->resize((size_t)data->BufTextLen);
        data->Buf = s->data();
    }
    return 0;
}

std::string fmt_number(double v) {
    char buf[64];
    if (std::fabs(v - std::round(v)) < 1e-12 && std::fabs(v) < 1e15) std::snprintf(buf, sizeof buf, "%.0f", v);
    else std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}

bool parse_number(const std::string& s, double* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

bool parse_int(const std::string& s, int64_t* out) {
    double v = 0;
    if (!parse_number(s, &v) || std::fabs(v - std::round(v)) > 1e-9) return false;
    *out = (int64_t)std::llround(v);
    return true;
}

}  // namespace

// ---- the ImGui side's state ---------------------------------------------------------------

struct BuilderPane::Draw {
    SpecLook look;
    BuilderPaneMetrics m;
    std::map<std::string, SpecRect> items;
    std::map<std::string, std::string> bufs;   // text fields' buffers while they are edited
    // A press in a palette or on a panel's strip, which becomes a drag once
    // the pointer moves.
    DragPayload pressed;
    ImVec2 press_at{0, 0};
    // A divider being dragged: its split's path ("" is the root).
    bool dividing = false;
    std::string divider;
    double divider_ratio = -1.0;
    // Drop plans for the hit under the pointer, by key.
    std::map<std::string, PanePlan> plans;
    std::string plans_text;
    std::string popup;   // the binding row whose choices are open
    float strip_px = 0.0f;
    bool canvas_hovered = false;

    void item(const std::string& name, const ImRect& r) { items[name] = sr(r); }
    float ui_h() const { return extent(look.ui, look.ui_size, "Ag").y; }
    float mono_h() const { return extent(look.mono, look.mono_size, "Ag").y; }
    float small_h() const { return extent(look.ui, look.small_size, "Ag").y; }
};

BuilderPane::BuilderPane(std::unique_ptr<BuilderPort> port, Options options)
    : port_(std::move(port)), options_(std::move(options)) {
    prefix_ = options_.group_prefix;
    if (prefix_.empty()) {
        std::random_device rd;
        char buf[32];
        std::snprintf(buf, sizeof buf, "builder-%08x", (unsigned)rd());
        prefix_ = buf;
    }
    update();
}

BuilderPane::~BuilderPane() = default;

bool BuilderPane::item_rect(const std::string& name, SpecRect* out) const {
    if (!draw_) return false;
    const auto it = draw_->items.find(name);
    if (it == draw_->items.end()) return false;
    if (out) *out = it->second;
    return true;
}

std::vector<std::string> BuilderPane::item_names() const {
    std::vector<std::string> out;
    if (draw_)
        for (const auto& kv : draw_->items) out.push_back(kv.first);
    return out;
}

BuilderPaneLayout BuilderPane::layout(const ImRect& area, float text_scale) const {
    BuilderPaneLayoutInput in;
    in.area = sr(area);
    in.build = build_;
    in.text_open = text_open_;
    in.chat = chat_open_ && chat_draw_;
    if (layout_rule_) return layout_rule_(in, text_scale);
    return compute_builder_pane_layout(in, BuilderPaneMetrics::for_text_scale(text_scale));
}

void BuilderPane::set_chat(bool open, std::function<void(const SpecRect&)> draw) {
    chat_open_ = open;
    chat_draw_ = std::move(draw);
}

// ---- small widgets in the house style -----------------------------------------------------

namespace {

// A section's header strip: the title in tracked capitals on the plate, a
// subtitle in secondary, closed by a rule.
void header_strip(const SpecLook& L, ImDrawList* dl, const ImRect& r, const std::string& title,
                  const std::string& subtitle) {
    dl->AddRectFilled(r.Min, r.Max, L.plate);
    dl->AddRectFilled(v2(r.Min.x, r.Max.y - 1.0f), r.Max, L.rule);
    const float tracking = std::round(L.small_size * 0.08f);
    const float th = extent(L.ui, L.small_size, "Ag").y;
    const float pad = std::round(8.0f * L.text_scale);
    const std::string t = upper(title);
    sternbild::text_tracked(dl, L.ui, L.small_size, v2(r.Min.x + pad, std::round(r.Min.y + (r.GetHeight() - th) * 0.5f)),
                            L.text, t.c_str(), tracking);
    if (!subtitle.empty()) {
        const float x = r.Min.x + pad + sternbild::text_tracked_width(L.ui, L.small_size, t.c_str(), tracking) +
                        std::round(10.0f * L.text_scale);
        put(dl, L.ui, L.small_size, v2(x, r.Min.y + (r.GetHeight() - th) * 0.5f), L.secondary,
             fit(L.ui, L.small_size, subtitle, r.Max.x - x - pad));
    }
}

// A label in tracked capitals, for the groups of a list and the sections
// of Properties.
void caps(const SpecLook& L, ImDrawList* dl, ImVec2 p, ImU32 c, const std::string& s) {
    const std::string t = upper(s);
    sternbild::text_tracked(dl, L.ui, L.small_size, v2(std::round(p.x), std::round(p.y)), c, t.c_str(),
                            std::round(L.small_size * 0.08f));
}

// A key button drawn at `r`: the key in mono, then the action, a 1 px
// outline; bronze outline when `on`. True when clicked.
bool key_button_at(const SpecLook& L, const char* id, const ImRect& r, const std::string& key, const std::string& label,
                   bool enabled, bool on, const std::string& tip = {}) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::SetCursorScreenPos(r.Min);
    const bool clicked = ImGui::InvisibleButton(id, v2(std::max(1.0f, r.GetWidth()), std::max(1.0f, r.GetHeight())));
    const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
    const bool held = ImGui::IsItemActive();
    ImU32 outline = on ? L.accent : enabled ? L.rule : L.rule_faint, fill = 0;
    ImU32 key_ink = enabled ? L.secondary : L.faint, ink = enabled ? L.text : L.faint;
    if (enabled && held && hovered) {
        fill = L.text;
        key_ink = ink = L.deep;
    } else if (enabled && hovered) {
        fill = L.plate_hover;
        key_ink = L.accent;
    }
    if (fill) dl->AddRectFilled(r.Min, r.Max, fill);
    dl->AddRect(r.Min, r.Max, outline, 0.0f, 0, 1.0f);
    const float s = L.text_scale, mh = extent(L.mono, L.mono_size, "Ag").y, uh = extent(L.ui, L.ui_size, "Ag").y;
    float x = r.Min.x + std::round(7.0f * s);
    if (!key.empty()) {
        put(dl, L.mono, L.mono_size, v2(x, r.Min.y + (r.GetHeight() - mh) * 0.5f), key_ink, key);
        x += extent(L.mono, L.mono_size, key).x + (label.empty() ? 0.0f : std::round(6.0f * s));
    }
    put(dl, L.ui, L.ui_size, v2(x, r.Min.y + (r.GetHeight() - uh) * 0.5f), ink, label);
    if (hovered && !tip.empty()) ImGui::SetTooltip("%s", tip.c_str());
    return clicked && enabled;
}

float key_button_width(const SpecLook& L, const std::string& key, const std::string& label) {
    const float s = L.text_scale;
    float w = 2.0f * std::round(7.0f * s);
    if (!key.empty()) w += extent(L.mono, L.mono_size, key).x;
    if (!key.empty() && !label.empty()) w += std::round(6.0f * s);
    w += extent(L.ui, L.ui_size, label).x;
    return std::round(w);
}

// A ruled text field: no box, a 1 px rule under the text, parchment while
// it is edited. Its buffer follows `current` while it is not edited.
struct Field {
    bool changed = false, activated = false, deactivated = false, entered = false, active = false;
    std::string value;
};

Field text_field(std::map<std::string, std::string>& bufs, const SpecLook& L, const std::string& id, const ImRect& r,
                 const std::string& current, const std::string& hint, bool mono, bool multiline = false,
                 ImGuiInputTextFlags extra = 0) {
    Field out;
    ImFont* font = mono ? L.mono : L.ui;
    const float size = mono ? L.mono_size : L.ui_size;
    const ImGuiID gid = ImGui::GetID(id.c_str());
    std::string& buf = bufs[id];
    const bool was_active = ImGui::GetActiveID() == gid;
    if (!was_active) buf = current;
    if (buf.capacity() < 32) buf.reserve(32);
    const float th = extent(font, size, "Ag").y;
    ImGui::PushStyleColor(ImGuiCol_FrameBg, 0);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, 0);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, 0);
    ImGui::PushStyleColor(ImGuiCol_Text, L.text);
    // A placeholder (an unset value, or the data's extent for an unset
    // range) is dimmer than any value, so it never reads as one.
    ImGui::PushStyleColor(ImGuiCol_TextDisabled, sternbild::with_alpha(L.faint, 110));
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, sternbild::with_alpha(L.accent, 90));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, v2(0.0f, 1.0f));
    ImGui::PushFont(font, size);
    bool changed = false;
    if (multiline) {
        ImGui::SetCursorScreenPos(r.Min);
        changed = ImGui::InputTextMultiline(id.c_str(), buf.data(), buf.capacity() + 1, r.GetSize(),
                                            ImGuiInputTextFlags_CallbackResize | extra, grow, &buf);
    } else {
        ImGui::SetCursorScreenPos(v2(r.Min.x, std::round(r.Min.y + (r.GetHeight() - th) * 0.5f) - 1.0f));
        ImGui::PushItemWidth(std::max(1.0f, r.GetWidth()));
        changed = ImGui::InputTextWithHint(id.c_str(), hint.c_str(), buf.data(), buf.capacity() + 1,
                                           ImGuiInputTextFlags_CallbackResize | extra, grow, &buf);
        ImGui::PopItemWidth();
    }
    out.activated = ImGui::IsItemActivated();
    out.deactivated = ImGui::IsItemDeactivated();
    out.active = ImGui::IsItemActive();
    out.entered = out.deactivated && ImGui::IsKeyPressed(ImGuiKey_Enter, false);
    ImGui::PopFont();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(6);
    if (!multiline) {
        const float y = std::floor(r.Max.y) - 1.0f;
        sternbild::rule_h(ImGui::GetWindowDrawList(), std::floor(r.Min.x), std::floor(r.Max.x), y,
                          out.active ? L.text : L.rule);
    }
    out.changed = changed;
    out.value = buf;
    return out;
}

// A flag: a square of the text's height, filled while on.
bool flag_box(const SpecLook& L, const char* id, const ImRect& r, bool on) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float side = std::round(11.0f * L.text_scale);
    const ImVec2 a(std::round(r.Min.x), std::round(r.Min.y + (r.GetHeight() - side) * 0.5f));
    const ImRect box(a, v2(a.x + side, a.y + side));
    ImGui::SetCursorScreenPos(v2(r.Min.x, r.Min.y));
    const bool clicked = ImGui::InvisibleButton(id, v2(std::max(1.0f, side + std::round(6.0f * L.text_scale)), r.GetHeight()));
    const bool hovered = ImGui::IsItemHovered();
    if (hovered) dl->AddRectFilled(box.Min, box.Max, L.plate_hover);
    dl->AddRect(box.Min, box.Max, hovered ? L.text : L.secondary, 0.0f, 0, 1.0f);
    if (on) {
        const float in = std::round(3.0f * L.text_scale);
        dl->AddRectFilled(v2(box.Min.x + in, box.Min.y + in), v2(box.Max.x - in, box.Max.y - in), L.text);
    }
    return clicked;
}

}  // namespace

// ---- the page bar and the keys ------------------------------------------------------------

namespace {

// The key a chord shows: ⌘ on macOS, Ctrl elsewhere.
std::string chord(const char* key, bool shift = false) {
#ifdef __APPLE__
    return std::string(shift ? "\xE2\x87\xA7" : "") + "\xE2\x8C\x98" + key;
#else
    return std::string(shift ? "Shift " : "") + "Ctrl " + key;
#endif
}

struct BarItem {
    const char* name;
    std::string key;
    const char* label;
};

std::vector<BarItem> bar_items(bool build) {
    std::vector<BarItem> items = {{"bar:build", chord("B"), "Build"}};
    if (build) {
        items.push_back({"bar:text", chord("T"), "Text"});
        items.push_back({"bar:undo", chord("Z"), "Undo"});
        items.push_back({"bar:redo", chord("Z", true), "Redo"});
        items.push_back({"bar:save", chord("S"), "Save"});
    }
    return items;
}

// A key button's width as sternbild::key_button lays it out, measured with
// the current context's faces.
float bar_button_width(const BarItem& it) {
    const ImGuiStyle& st = ImGui::GetStyle();
    return ImGui::CalcTextSize(it.key.c_str()).x + ImGui::CalcTextSize(it.label).x + 2.0f * st.FramePadding.x + 14.0f;
}

}  // namespace

float BuilderPane::page_bar_width() const {
    float w = 0.0f;
    for (const auto& it : bar_items(build_)) w += bar_button_width(it) + 6.0f;
    return w;
}

void BuilderPane::page_bar(float max_width) {
    if (!draw_) draw_ = std::make_unique<Draw>();
    for (auto it = draw_->items.begin(); it != draw_->items.end();) {
        if (it->first.rfind("bar:", 0) == 0) it = draw_->items.erase(it);
        else ++it;
    }
    // Where the bar is too narrow for all of them, only Build stays; the
    // others keep their keys.
    std::vector<BarItem> items = bar_items(build_);
    if (max_width > 0.0f && page_bar_width() > max_width) items.resize(1);
    for (size_t i = 0; i < items.size(); ++i) {
        const BarItem& it = items[i];
        if (i) ImGui::SameLine(0.0f, 6.0f);
        bool enabled = true, on = false;
        if (std::string(it.name) == "bar:build") on = build_;
        else if (std::string(it.name) == "bar:text") on = text_open_;
        else if (std::string(it.name) == "bar:undo") enabled = can_undo();
        else if (std::string(it.name) == "bar:redo") enabled = can_redo();
        else if (std::string(it.name) == "bar:save") enabled = open_.has_value();
        const bool clicked = sternbild::key_button(it.key.c_str(), it.label, enabled);
        const ImRect r(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        draw_->item(it.name, r);
        if (on) ImGui::GetWindowDrawList()->AddRect(r.Min, r.Max, sternbild::col::bronze, 0.0f, 0, 1.0f);
        if (!clicked) continue;
        const std::string name = it.name;
        if (name == "bar:build") set_build_mode(!build_);
        else if (name == "bar:text") text_open_ = !text_open_;
        else if (name == "bar:undo") undo();
        else if (name == "bar:redo") redo();
        else if (name == "bar:save") save();
    }
    // The last refusal or failure: its code, with the message in its tooltip.
    if (build_ && error_ && error_.code != "builder.nothing") {
        ImGui::SameLine(0.0f, 14.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
        ImGui::TextUnformatted(error_.code.c_str());
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", error_.message.c_str());
    }
}

bool BuilderPane::keys(bool escape) {
    const ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_B)) set_build_mode(!build_);
    if (!build_) return false;
    const bool typing = io.WantTextInput;
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_T)) text_open_ = !text_open_;
    if (!typing && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z)) redo();
    else if (!typing && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Z)) undo();
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S)) save();
    if (!typing && !focus_panel_.empty() &&
        (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false)))
        remove_focused();
    if (!escape) return false;
    if (drag_.kind != DragPayload::Kind::none || (draw_ && draw_->pressed.kind != DragPayload::Kind::none)) {
        drag_ = {};
        hit_ = {};
        if (draw_) draw_->pressed = {};
        return true;
    }
    if (editing_) {
        set_editing(false);
        return true;
    }
    if (!focus_source_.empty()) {
        focus_source_.clear();
        return true;
    }
    return false;   // the page gives up a panel's focus, then goes Home
}

// ---- the canvas -----------------------------------------------------------------------------

namespace {

// The rectangle where a plan's panel lands: the layout of the spec it
// makes, laid out on the canvas.
bool landing_rect(const PanePlan& plan, const SpecRect& canvas, const BuilderPaneMetrics&, float scale, SpecRect* out) {
    if (!plan.after || plan.placed.empty()) return false;
    const auto root = cb::effective_layout(*plan.after);
    if (!root) return false;
    const SpecLayout l = compute_spec_layout(*root, *plan.after, canvas, SpecLayoutMetrics::for_text_scale(scale));
    const SpecLayoutPanel* p = l.find(plan.placed);
    if (!p) return false;
    *out = p->rect;
    return true;
}

std::string payload_label(const BuilderPane& pane, const DragPayload& d) {
    switch (d.kind) {
        case DragPayload::Kind::source: {
            const PaneSource* r = pane.find_source(d.source);
            return r ? r->name : d.source;
        }
        case DragPayload::Kind::column: {
            const PaneSource* r = pane.find_source(d.source);
            return (r ? r->name : d.source) + "." + d.column;
        }
        case DragPayload::Kind::primitive: return pane_primitive_noun(d.primitive);
        case DragPayload::Kind::panel: return d.panel;
        case DragPayload::Kind::none: break;
    }
    return {};
}

std::string plan_key(const DragPayload& d, const DropHit& h) {
    return std::to_string((int)d.kind) + "|" + d.source + "|" + d.column + "|" + spec::to_string(d.primitive) + "|" +
           d.panel + "|" + std::to_string((int)h.zone) + "|" + h.panel + "|" + h.where + "|" + std::to_string(h.slot);
}

}  // namespace

void BuilderPane::draw(const ImRect& area, SpecRuntime& runtime, const SpecLook& look, std::string* focus) {
    if (focus && *focus != synced_focus_) focus_panel(*focus);   // the page changed it
    draw_body(area, runtime, look);
    if (focus) *focus = synced_focus_ = focus_panel_;
}

void BuilderPane::draw_body(const ImRect& area, SpecRuntime& runtime, const SpecLook& look) {
    if (!draw_) draw_ = std::make_unique<Draw>();
    Draw& d = *draw_;
    d.look = look;
    d.m = BuilderPaneMetrics::for_text_scale(look.text_scale);
    for (auto it = d.items.begin(); it != d.items.end();) {
        if (it->first.rfind("bar:", 0) == 0) ++it;   // page_bar()'s, drawn before
        else it = d.items.erase(it);
    }
    update();
    if (open_ && open_->text != d.plans_text) {
        d.plans.clear();
        d.plans_text = open_->text;
    }
    if (open_) runtime.set_spec(open_->spec);
    // An empty canvas says how to start in build mode (canvas_hint()).
    runtime.set_empty_hint(canvas_hint());
    if (!build_) {
        runtime.draw(area, look);
        return;
    }
    const BuilderPaneLayout L = layout(area, look.text_scale);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (const auto& r : L.rules) dl->AddRectFilled(v2(r.x, r.y), v2(r.right(), r.bottom()), look.rule);
    const SpecRect canvas = L.canvas;
    const float strip = spec_panel_strip_height(look, true, 1000.0f);
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;

    // ---- the canvas: the runtime's own drawing.
    runtime.draw(rr(canvas), look);
    const SpecLayout& lay = runtime.layout();
    for (const auto& p : lay.panels) {
        d.item("panel:" + p.id, rr(p.rect));
        if (!p.tab) d.item("header:" + p.id, ImRect(v2(p.rect.x, p.rect.y), v2(p.rect.right(), p.rect.y + strip)));
    }
    for (const auto& s : lay.splits)
        if (s.children == 2) d.item("divider:" + s.path, rr(s.divider));
    d.item("canvas", rr(canvas));
    // Focus follows a panel that went away.
    if (!focus_panel_.empty() && !lay.find(focus_panel_)) {
        if (!open_ || !open_->spec.find_panel(focus_panel_)) focus_panel_.clear();
    }

    // ---- the canvas's input: focus, a strip's drag, a divider's drag.
    const bool over_canvas = spec_rect_contains(canvas, mouse.x, mouse.y) &&
                             ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                             !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
    const SpecLayoutSplit* grip = over_canvas && !d.dividing && drag_.kind == DragPayload::Kind::none
                                      ? hit_divider(lay, mouse.x, mouse.y, d.m)
                                      : nullptr;
    const std::string header = over_canvas && !grip ? hit_header(lay, mouse.x, mouse.y, strip) : std::string();
    if (grip) ImGui::SetMouseCursor(grip->horizontal ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
    else if (!header.empty() && drag_.kind == DragPayload::Kind::none) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    if (over_canvas && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (grip) {
            d.dividing = true;
            d.divider = grip->path;
            d.divider_ratio = grip->ratio;
            begin_gesture("ratio." + grip->path);
        } else {
            const std::string p = hit_panel(lay, mouse.x, mouse.y);
            focus_panel(p);
            if (!header.empty()) {
                d.pressed = {};
                d.pressed.kind = DragPayload::Kind::panel;
                d.pressed.panel = header;
                d.press_at = mouse;
            }
        }
    }
    if (d.dividing) {
        const SpecLayoutSplit* s = nullptr;
        for (const auto& x : lay.splits)
            if (x.path == d.divider) s = &x;
        if (!s || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            d.dividing = false;
            d.divider.clear();
            end_gesture();
        } else {
            ImGui::SetMouseCursor(s->horizontal ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
            const double r = divider_ratio(*s, mouse.x, mouse.y);
            if (std::fabs(r - d.divider_ratio) > 1e-9) {
                d.divider_ratio = r;
                set_ratio(d.divider, r);
            }
        }
    }

    // ---- a press becomes a drag once the pointer moves.
    if (d.pressed.kind != DragPayload::Kind::none) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && drag_.kind == DragPayload::Kind::none) {
            d.pressed = {};
        } else if (drag_.kind == DragPayload::Kind::none) {
            const float dx = mouse.x - d.press_at.x, dy = mouse.y - d.press_at.y;
            const float threshold = std::round(6.0f * look.text_scale);
            if (dx * dx + dy * dy > threshold * threshold) drag_ = d.pressed;
        }
    }
    if (drag_.kind != DragPayload::Kind::none) {
        const bool source = drag_.kind == DragPayload::Kind::source || drag_.kind == DragPayload::Kind::column;
        const DropHit probe = hit_drop(lay, canvas, mouse.x, mouse.y, 0, strip, d.m);
        const int slots = source && (probe.zone == DropZone::centre || probe.zone == DropZone::edge || probe.zone == DropZone::slot)
                              ? (int)drop_slots(probe.panel).size()
                              : 0;
        hit_ = hit_drop(lay, canvas, mouse.x, mouse.y, slots, strip, d.m);
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const DragPayload payload = drag_;
            const DropHit hit = hit_;
            drag_ = {};
            hit_ = {};
            d.pressed = {};
            if (hit.zone != DropZone::none) drop(payload, hit);
        }
    }

    // ---- the canvas's overlays, above the panels: the strip under the
    // pointer, the divider, where a drop lands.
    ImGui::SetCursorScreenPos(v2(canvas.x, canvas.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(0, 0));
    if (ImGui::BeginChild("##builder.overlay", v2(std::max(1.0f, canvas.w), std::max(1.0f, canvas.h)), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
                              ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoSavedSettings)) {
        ImDrawList* ol = ImGui::GetWindowDrawList();
        const float s = look.text_scale;
        // The focused panel's frame.
        if (const SpecLayoutPanel* fp = focus_panel_.empty() ? nullptr : lay.find(focus_panel_))
            ol->AddRect(v2(fp->rect.x, fp->rect.y), v2(fp->rect.right(), fp->rect.bottom()), look.accent, 0.0f, 0, 1.0f);
        // A strip that can be dragged, under the pointer.
        if (!header.empty() && drag_.kind == DragPayload::Kind::none && !d.dividing) {
            if (const SpecLayoutPanel* hp = lay.find(header))
                ol->AddRectFilled(v2(hp->rect.x, hp->rect.y + strip - 1.0f), v2(hp->rect.right(), hp->rect.y + strip),
                                  look.secondary);
        }
        // The divider under the pointer or being dragged, with its ratio.
        const SpecLayoutSplit* shown = grip;
        if (d.dividing)
            for (const auto& x : lay.splits)
                if (x.path == d.divider) shown = &x;
        if (shown) {
            // The divider, three pixels wide while it is held or under the
            // pointer, and the ratio beside the pointer on the first child's
            // side, clear of the strips.
            const SpecRect g = shown->divider;
            const float grow = 1.0f;
            ol->AddRectFilled(v2(g.x - (shown->horizontal ? grow : 0.0f), g.y - (shown->horizontal ? 0.0f : grow)),
                              v2(g.right() + (shown->horizontal ? grow : 0.0f), g.bottom() + (shown->horizontal ? 0.0f : grow)),
                              look.accent);
            if (d.dividing) {
                char buf[16];
                std::snprintf(buf, sizeof buf, "%.2f", d.divider_ratio);
                const ImVec2 ts = extent(look.mono, look.mono_size, buf);
                const float pad = std::round(5.0f * s);
                const ImVec2 p = shown->horizontal ? v2(g.x - pad * 3.0f - ts.x, std::round(mouse.y - ts.y * 0.5f))
                                                   : v2(std::round(mouse.x + pad * 3.0f), g.y - pad * 3.0f - ts.y);
                const ImRect box(v2(p.x - pad, p.y - pad * 0.5f), v2(p.x + ts.x + pad, p.y + ts.y + pad * 0.5f));
                ol->AddRectFilled(box.Min, box.Max, look.deep);
                ol->AddRect(box.Min, box.Max, look.accent, 0.0f, 0, 1.0f);
                put(ol, look.mono, look.mono_size, p, look.text, buf);
                d.item("ratio", box);
            }
        }
        // Where a drop lands.
        if (drag_.kind != DragPayload::Kind::none && hit_.zone != DropZone::none) {
            const bool source = drag_.kind == DragPayload::Kind::source || drag_.kind == DragPayload::Kind::column;
            const std::string k = plan_key(drag_, hit_);
            auto it = d.plans.find(k);
            if (it == d.plans.end()) it = d.plans.emplace(k, plan_drop(drag_, hit_)).first;
            const PanePlan& plan = it->second;
            // A source over a panel: its slots as rows, the ones it fits in
            // parchment, the hovered one outlined.
            if (source && !hit_.panel.empty() && (hit_.zone == DropZone::slot || hit_.zone == DropZone::centre ||
                                                   hit_.zone == DropZone::edge)) {
                const SpecLayoutPanel* hp = lay.find(hit_.panel);
                const std::vector<PaneSlot> slots = drop_slots(hit_.panel);
                if (hp) {
                    const std::vector<SpecRect> rows = drop_slot_rows(hp->rect, hp->tab ? 0.0f : strip, (int)slots.size(), d.m);
                    for (size_t i = 0; i < rows.size(); ++i) {
                        DropHit sh = hit_;
                        sh.zone = DropZone::slot;
                        sh.slot = (int)i;
                        const std::string sk = plan_key(drag_, sh);
                        auto si = d.plans.find(sk);
                        if (si == d.plans.end()) si = d.plans.emplace(sk, plan_drop(drag_, sh)).first;
                        const bool fits = !si->second.refusal;
                        const bool on = hit_.zone == DropZone::slot && hit_.slot == (int)i;
                        const ImRect r = rr(rows[i]);
                        d.item("slot:" + std::string(slots[i].name), r);
                        ol->AddRectFilled(r.Min, r.Max, on && fits ? look.plate_hover : look.deep);
                        ol->AddRect(r.Min, r.Max, on ? (fits ? look.accent : look.secondary) : look.rule, 0.0f, 0, 1.0f);
                        const float pad = std::round(10.0f * s);
                        const float ty = r.Min.y + (r.GetHeight() - d.mono_h()) * 0.5f;
                        put(ol, look.mono, look.mono_size, v2(r.Min.x + pad, ty), fits ? look.text : look.faint, slots[i].name);
                        if (slots[i].required) {
                            const float w = extent(look.mono, look.mono_size, slots[i].name).x;
                            put(ol, look.mono, look.mono_size, v2(r.Min.x + pad + w + 2.0f, ty), look.faint, "*");
                        }
                    }
                }
            }
            SpecRect land;
            if (hit_.zone == DropZone::slot) {
                // The slot row is the target; nothing moves.
            } else if (landing_rect(plan, canvas, d.m, look.text_scale, &land)) {
                const ImRect r = rr(land);
                if (!plan.refusal) ol->AddRectFilled(r.Min, r.Max, sternbild::with_alpha(look.accent, 34));
                ol->AddRect(v2(r.Min.x + 1.0f, r.Min.y + 1.0f), v2(r.Max.x - 1.0f, r.Max.y - 1.0f),
                            plan.refusal ? look.secondary : look.accent, 0.0f, 0, 2.0f);
                d.item("landing", r);
            } else if (hit_.zone == DropZone::empty) {
                const ImRect r = rr(canvas);
                ol->AddRect(v2(r.Min.x + 1.0f, r.Min.y + 1.0f), v2(r.Max.x - 1.0f, r.Max.y - 1.0f),
                            plan.refusal ? look.secondary : look.accent, 0.0f, 0, 2.0f);
                d.item("landing", r);
            }
            if (plan.refusal && !plan.refusal.message.empty() && hit_.zone != DropZone::slot)
                ImGui::SetTooltip("%s", plan.refusal.message.c_str());
            if (hit_.zone == DropZone::slot) {
                DropHit sh = hit_;
                auto si = d.plans.find(plan_key(drag_, sh));
                if (si != d.plans.end() && si->second.refusal) ImGui::SetTooltip("%s", si->second.refusal.message.c_str());
            }
        }
        // An empty canvas shows the runtime's own state, "No panels", with
        // how to start under it (canvas_hint()).
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();

    // ---- the columns and the strip.
    if (L.left) {
        draw_sources(L.sources);
        draw_primitives(L.primitives);
    }
    if (L.right) draw_properties(L.properties, runtime);
    if (L.chat && chat_draw_) {
        d.item("chat", rr(L.chat_rect));
        chat_draw_(L.chat_rect);
    }
    if (L.strip) draw_text_strip(L.strip_rect);

    // ---- what is dragged, by the pointer.
    if (drag_.kind != DragPayload::Kind::none) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();
        const std::string label = payload_label(*this, drag_);
        const float s = look.text_scale, pad = std::round(6.0f * s);
        const ImVec2 ts = extent(look.mono, look.mono_size, label);
        const ImVec2 p(std::round(mouse.x + 14.0f * s), std::round(mouse.y + 10.0f * s));
        fg->AddRectFilled(p, v2(p.x + ts.x + 2.0f * pad, p.y + ts.y + pad), look.plate);
        fg->AddRect(p, v2(p.x + ts.x + 2.0f * pad, p.y + ts.y + pad), look.accent, 0.0f, 0, 1.0f);
        put(fg, look.mono, look.mono_size, v2(p.x + pad, p.y + pad * 0.5f), look.text, label);
    }
}

// ---- Sources ------------------------------------------------------------------------------

void BuilderPane::draw_sources(const SpecRect& rect) {
    Draw& d = *draw_;
    const SpecLook& L = d.look;
    const float s = L.text_scale;
    ImGui::SetCursorScreenPos(v2(rect.x, rect.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, v2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, L.ground);
    if (ImGui::BeginChild("##builder.sources", v2(rect.w, rect.h), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImRect head(v2(rect.x, rect.y), v2(rect.right(), rect.y + d.m.header));
        int count = 0;
        for (const auto& r : rows_)
            if (r.group != PaneSource::Group::spec) ++count;
        header_strip(L, dl, head, "Sources", std::to_string(count));
        // Add: the selection as a panel of its default primitive.
        {
            const std::string key = "\xE2\x8F\x8E", label = "Add";
            const float w = key_button_width(L, key, label), h = std::round(d.m.header - 6.0f * s);
            const ImRect b(v2(head.Max.x - std::round(6.0f * s) - w, head.Min.y + std::round(3.0f * s)),
                           v2(head.Max.x - std::round(6.0f * s), head.Min.y + std::round(3.0f * s) + h));
            const PanePlan* plan = nullptr;
            if (!selected_.empty()) {
                // Kept with the primitives' plans until the spec or the selection change.
                const std::string k = "add|" + selected_ + "|" + selected_column_ + "|" + focus_panel_;
                auto it = primitive_plans_.find(k);
                if (it == primitive_plans_.end())
                    it = primitive_plans_.emplace(k, plan_new_panel(selected_, selected_column_, std::nullopt, DropHit{})).first;
                plan = &it->second;
            }
            const bool ok = plan && !plan->refusal;
            if (key_button_at(L, "##sources.add", b, key, label, ok, false, plan && plan->refusal ? plan->refusal.message : std::string()))
                add_selected();
            d.item("add:sources", b);
        }
        // The filter: a ruled line.
        const float pad = d.m.pad;
        const ImRect fr(v2(rect.x + pad, head.Max.y + std::round(2.0f * s)), v2(rect.right() - pad, head.Max.y + d.m.row + std::round(2.0f * s)));
        const Field f = text_field(d.bufs, L, "##sources.filter", fr, filter_, "Filter", false);
        if (f.changed) set_filter(f.value);
        d.item("filter", fr);
        // The rows, scrolling.
        const float list_y = fr.Max.y + std::round(4.0f * s);
        ImGui::SetCursorScreenPos(v2(rect.x, list_y));
        if (ImGui::BeginChild("##sources.list", v2(rect.w, std::max(1.0f, rect.bottom() - list_y)), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoBackground)) {
            ImDrawList* ldl = ImGui::GetWindowDrawList();
            const float row = d.m.row;
            const float w = ImGui::GetContentRegionAvail().x;
            // Made again only when the rows, the filter, an expansion or Show
            // all changed; a click below changes them after the loop.
            const std::vector<PaneListItem>& items = list_items();
            int last_group = -1;
            const ImGuiIO& io = ImGui::GetIO();
            bool want_add = false;   // run after the rows: an add rebuilds them
            std::string want_toggle;  // a run opened or closed, after the rows
            bool want_more = false;   // Show all or Show fewer, after the rows
            // The files the last drop refused: the file's name and the loader
            // store's code, its reason in the tooltip, until the next drop.
            if (!refused_.empty()) {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                caps(L, ldl, v2(p.x + pad, p.y + std::round(row * 0.45f) - 1.0f), L.faint, "Refused");
                ImGui::Dummy(v2(w, std::round(row * 1.15f)));
                for (size_t i = 0; i < refused_.size(); ++i) {
                    const Refused& f = refused_[i];
                    const ImVec2 rp = ImGui::GetCursorScreenPos();
                    const ImRect rr_(rp, v2(rp.x + w, rp.y + row));
                    ImGui::PushID((int)i);
                    ImGui::InvisibleButton("##refused", rr_.GetSize());
                    const bool hovered = ImGui::IsItemHovered();
                    ImGui::PopID();
                    d.item("refused:" + std::filesystem::path(f.path).filename().string(), rr_);
                    if (hovered) ldl->AddRectFilled(rr_.Min, rr_.Max, L.plate_hover);
                    std::string code = f.code;
                    if (code.rfind("loader.", 0) == 0) code = code.substr(7);
                    const float cw = std::min(extent(L.mono, L.mono_size, code).x, w * 0.5f);
                    const float x = rr_.Min.x + pad + std::round(18.0f * s);
                    put(ldl, L.ui, L.ui_size, v2(x, rr_.Min.y + (row - d.ui_h()) * 0.5f), L.secondary,
                        fit(L.ui, L.ui_size, std::filesystem::path(f.path).filename().string(),
                            rr_.Max.x - pad - cw - std::round(10.0f * s) - x));
                    const std::string shown = fit(L.mono, L.mono_size, code, cw + 1.0f);
                    put(ldl, L.mono, L.mono_size,
                        v2(rr_.Max.x - pad - extent(L.mono, L.mono_size, shown).x, rr_.Min.y + (row - d.mono_h()) * 0.5f),
                        L.faint, shown);
                    if (hovered && !f.message.empty()) ImGui::SetTooltip("%s", f.message.c_str());
                }
            }
            for (const PaneListItem& item : items) {
                const PaneSource::Group group =
                    item.kind == PaneListItem::Kind::source ? item.source->group : PaneSource::Group::metric;
                if ((int)group != last_group) {
                    last_group = (int)group;
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    caps(L, ldl, v2(p.x + pad, p.y + std::round(row * 0.45f) - 1.0f), L.faint, pane_group_label(group));
                    ImGui::Dummy(v2(w, std::round(row * 1.15f)));
                }
                if (item.kind != PaneListItem::Kind::source) {
                    // A run of metrics, which opens to them, or Show all.
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    const ImRect rowr(p, v2(p.x + w, p.y + row));
                    if (!ImGui::IsRectVisible(rowr.Min, rowr.Max)) {   // scrolled away: its space only
                        ImGui::Dummy(rowr.GetSize());
                        continue;
                    }
                    const bool run = item.kind == PaneListItem::Kind::run;
                    const std::string key = run ? "run:" + std::to_string(item.run) : std::string("runs:all");
                    ImGui::PushID(key.c_str());
                    const bool clicked = ImGui::InvisibleButton("##run", rowr.GetSize());
                    const bool hovered = ImGui::IsItemHovered();
                    ImGui::PopID();
                    d.item(key, rowr);
                    if (hovered) ldl->AddRectFilled(rowr.Min, rowr.Max, L.plate_hover);
                    float x = rowr.Min.x + pad;
                    if (run) {
                        const float a = std::round(4.0f * s);
                        const ImVec2 c(x + a, rowr.GetCenter().y);
                        if (item.open) ldl->AddTriangleFilled(v2(c.x - a, c.y - a * 0.6f), v2(c.x + a, c.y - a * 0.6f), v2(c.x, c.y + a * 0.7f), L.secondary);
                        else ldl->AddTriangleFilled(v2(c.x - a * 0.6f, c.y - a), v2(c.x - a * 0.6f, c.y + a), v2(c.x + a * 0.7f, c.y), L.secondary);
                    }
                    x += std::round(18.0f * s);
                    // A run: its id and its number of metrics; Show all: the runs in all.
                    const std::string right = run ? std::to_string(item.metrics) + (item.metrics == 1 ? " metric" : " metrics") +
                                                        "  #" + std::to_string(item.run)
                                                  : std::to_string(item.metrics) + " runs";
                    const float right_w = std::min(extent(L.mono, L.mono_size, right).x, w * 0.55f);
                    const float right_end = rowr.Max.x - pad;
                    put(ldl, L.ui, L.ui_size, v2(x, rowr.Min.y + (row - d.ui_h()) * 0.5f), run ? L.text : L.secondary,
                        fit(L.ui, L.ui_size, item.name, right_end - right_w - std::round(10.0f * s) - x));
                    const std::string sh = fit(L.mono, L.mono_size, right, right_w + 1.0f);
                    put(ldl, L.mono, L.mono_size,
                        v2(right_end - extent(L.mono, L.mono_size, sh).x, rowr.Min.y + (row - d.mono_h()) * 0.5f), L.faint, sh);
                    if (hovered && run && !item.experiment.empty()) ImGui::SetTooltip("%s", item.experiment.c_str());
                    if (clicked) {
                        if (run) want_toggle = key;
                        else want_more = true;
                    }
                    continue;
                }
                const PaneSource* r = item.source;
                const bool tabular = r->shape.tabular && !r->shape.columns.empty();
                const bool open = tabular && expanded(r->key);
                const bool sel = selected_ == r->key && selected_column_.empty();
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const ImRect rowr(p, v2(p.x + w, p.y + row));
                // A closed row scrolled out of sight takes its space only, so
                // that hundreds of rows cost little.
                if (!open && !ImGui::IsRectVisible(rowr.Min, rowr.Max) &&
                    !(widget_.height && widget_.height("source:" + r->key, w) > 0.0f)) {
                    ImGui::Dummy(rowr.GetSize());
                    continue;
                }
                // A file just dropped offers ⌘K at its row's right end. Its
                // target is submitted before the row's, so that it takes the
                // pointer there and the row everywhere else.
                const bool cmdk = widget_.open_file && !fresh_loader(r->key).empty();
                ImRect cmdk_r;
                bool cmdk_clicked = false;
                if (cmdk) {
                    const float kw = key_button_width(L, "\xE2\x8C\x98K", "");
                    cmdk_r = ImRect(v2(rowr.Max.x - pad - kw, rowr.Min.y + std::round(2.0f * s)),
                                    v2(rowr.Max.x - pad, rowr.Max.y - std::round(2.0f * s)));
                    ImGui::SetCursorScreenPos(cmdk_r.Min);
                    ImGui::PushID(r->key.c_str());
                    cmdk_clicked = ImGui::InvisibleButton("##cmdk.source", cmdk_r.GetSize());
                    ImGui::PopID();
                    d.item("cmdk:source:" + r->key, cmdk_r);
                    ImGui::SetCursorScreenPos(p);
                }
                ImGui::PushID(r->key.c_str());
                ImGui::InvisibleButton("##row", rowr.GetSize());
                const bool hovered = ImGui::IsItemHovered();
                if (ImGui::IsItemActivated()) {
                    // A press on the disclosure opens the columns; on the row, it selects and may drag.
                    const float arrow_w = std::round(18.0f * s) + pad + (item.nested ? std::round(12.0f * s) : 0.0f);
                    if (tabular && io.MousePos.x < rowr.Min.x + arrow_w) {
                        toggle_expanded(r->key);
                    } else {
                        select(r->key);
                        d.pressed = {};
                        d.pressed.kind = DragPayload::Kind::source;
                        d.pressed.source = r->key;
                        d.press_at = io.MousePos;
                    }
                }
                if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) want_add = true;
                ImGui::PopID();
                d.item("source:" + r->key, rowr);
                if (sel) ldl->AddRectFilled(rowr.Min, rowr.Max, L.plate);
                else if (hovered) ldl->AddRectFilled(rowr.Min, rowr.Max, L.plate_hover);
                if (sel) ldl->AddRectFilled(rowr.Min, v2(rowr.Min.x + 2.0f, rowr.Max.y), L.accent);
                // A run's metric is drawn under its run, by its tag.
                float x = rowr.Min.x + pad + (item.nested ? std::round(12.0f * s) : 0.0f);
                if (tabular) {
                    // ▸ closed, ▾ open.
                    const float a = std::round(4.0f * s);
                    const ImVec2 c(x + a, rowr.GetCenter().y);
                    if (open) ldl->AddTriangleFilled(v2(c.x - a, c.y - a * 0.6f), v2(c.x + a, c.y - a * 0.6f), v2(c.x, c.y + a * 0.7f), L.secondary);
                    else ldl->AddTriangleFilled(v2(c.x - a * 0.6f, c.y - a), v2(c.x - a * 0.6f, c.y + a), v2(c.x + a * 0.7f, c.y), L.secondary);
                }
                x += std::round(18.0f * s);
                float right_end = rowr.Max.x - pad;
                if (cmdk) {
                    // Drawn over the row's fill; its target is the one above.
                    const ImVec2 back = ImGui::GetCursorScreenPos();
                    ImGui::PushID(r->key.c_str());
                    widget_key("##cmdk.source.draw", cmdk_r, true);
                    ImGui::PopID();
                    ImGui::SetCursorScreenPos(back);
                    right_end = cmdk_r.Min.x - std::round(6.0f * s);
                    if (cmdk_clicked) {
                        select(r->key);
                        open_file_prompt();
                    }
                }
                const std::string shape = r->shape_text;
                const float shape_w = std::min(extent(L.mono, L.mono_size, shape).x, w * 0.55f);
                const float name_w = right_end - shape_w - std::round(10.0f * s) - x;
                const ImU32 ink = sel ? L.accent : L.text;
                put(ldl, L.ui, L.ui_size, v2(x, rowr.Min.y + (row - d.ui_h()) * 0.5f), ink,
                    fit(L.ui, L.ui_size, item.nested ? r->tag : r->name, name_w));
                const std::string sh = fit(L.mono, L.mono_size, shape, shape_w + 1.0f);
                put(ldl, L.mono, L.mono_size,
                     v2(right_end - extent(L.mono, L.mono_size, sh).x, rowr.Min.y + (row - d.mono_h()) * 0.5f),
                     L.faint, sh);
                if (open) {
                    for (const auto& c : r->shape.columns) {
                        const ImVec2 cp = ImGui::GetCursorScreenPos();
                        const ImRect cr(cp, v2(cp.x + w, cp.y + row));
                        const bool csel = selected_ == r->key && selected_column_ == c.name;
                        ImGui::PushID((r->key + "." + c.name).c_str());
                        ImGui::InvisibleButton("##col", cr.GetSize());
                        const bool ch = ImGui::IsItemHovered();
                        if (ImGui::IsItemActivated()) {
                            select(r->key, c.name);
                            d.pressed = {};
                            d.pressed.kind = DragPayload::Kind::column;
                            d.pressed.source = r->key;
                            d.pressed.column = c.name;
                            d.press_at = io.MousePos;
                        }
                        if (ch && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) want_add = true;
                        ImGui::PopID();
                        d.item("column:" + r->key + "." + c.name, cr);
                        if (csel) ldl->AddRectFilled(cr.Min, cr.Max, L.plate);
                        else if (ch) ldl->AddRectFilled(cr.Min, cr.Max, L.plate_hover);
                        if (csel) ldl->AddRectFilled(cr.Min, v2(cr.Min.x + 2.0f, cr.Max.y), L.accent);
                        const float cx = x + std::round(14.0f * s);
                        const float tw = extent(L.mono, L.mono_size, c.type).x;
                        put(ldl, L.mono, L.mono_size, v2(cx, cr.Min.y + (row - d.mono_h()) * 0.5f), csel ? L.accent : L.secondary,
                             fit(L.mono, L.mono_size, c.name, cr.Max.x - pad - tw - std::round(10.0f * s) - cx));
                        put(ldl, L.mono, L.mono_size, v2(cr.Max.x - pad - tw, cr.Min.y + (row - d.mono_h()) * 0.5f), L.faint, c.type);
                    }
                }
                // The inline prompt on this file, under its row.
                widget_under("source:" + r->key, rowr.Min.x, w);
            }
            if (items.empty()) {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                put(ldl, L.ui, L.ui_size, v2(p.x + pad, p.y + std::round(6.0f * s)), L.faint,
                     filter_.empty() ? "No sources" : "No match");
                ImGui::Dummy(v2(w, row));
            }
            // ⏎ with the list under the pointer adds the selection.
            if (ImGui::IsWindowHovered() && !ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Enter, false))
                want_add = true;
            if (want_add) add_selected();
            if (!want_toggle.empty()) toggle_expanded(want_toggle);
            if (want_more) set_show_all_runs(!show_all_runs());
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// ---- Primitives ---------------------------------------------------------------------------

void BuilderPane::draw_primitives(const SpecRect& rect) {
    Draw& d = *draw_;
    const SpecLook& L = d.look;
    const float s = L.text_scale;
    ImGui::SetCursorScreenPos(v2(rect.x, rect.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, v2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, L.ground);
    if (ImGui::BeginChild("##builder.primitives", v2(rect.w, rect.h), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImRect head(v2(rect.x, rect.y), v2(rect.right(), rect.y + d.m.header));
        std::string sub;
        if (const PaneSource* r = find_source(selected_))
            sub = selected_column_.empty() ? r->name : r->name + "." + selected_column_;
        header_strip(L, dl, head, "Primitives", sub);
        const float pad = d.m.pad, row = d.m.row;
        float y = head.Max.y + std::round(pad * 0.5f);
        const ImGuiIO& io = ImGui::GetIO();
        std::optional<Primitive> want;   // added after the rows
        for (Primitive p : pane_primitives()) {
            if (y + row > rect.bottom() + 0.5f) break;
            const ImRect r(v2(rect.x, y), v2(rect.right(), y + row));
            const PanePlan& plan = primitive_plan(p);
            const bool fits = !plan.refusal;
            ImGui::SetCursorScreenPos(r.Min);
            ImGui::PushID(spec::to_string(p));
            const bool clicked = ImGui::InvisibleButton("##prim", r.GetSize());
            const bool hovered = ImGui::IsItemHovered();
            if (ImGui::IsItemActivated()) {
                d.pressed = {};
                d.pressed.kind = DragPayload::Kind::primitive;
                d.pressed.primitive = p;
                d.press_at = io.MousePos;
            }
            ImGui::PopID();
            d.item(std::string("primitive:") + spec::to_string(p), r);
            if (hovered && fits) dl->AddRectFilled(r.Min, r.Max, L.plate_hover);
            const std::string noun = pane_primitive_noun(p);
            put(dl, L.ui, L.ui_size, v2(r.Min.x + pad + std::round(18.0f * s), r.Min.y + (row - d.ui_h()) * 0.5f),
                 fits ? L.text : L.faint, noun);
            if (hovered && !fits && !plan.refusal.message.empty()) ImGui::SetTooltip("%s", plan.refusal.message.c_str());
            if (clicked && drag_.kind == DragPayload::Kind::none) {
                const float dx = io.MousePos.x - d.press_at.x, dy = io.MousePos.y - d.press_at.y;
                if (dx * dx + dy * dy < 36.0f * s * s) want = p;
            }
            y += row;
        }
        if (want) add_primitive(*want);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// ---- Properties ---------------------------------------------------------------------------

// Properties' rows: a label column and a value column.
struct BuilderPane::Rows {
    const SpecLook& L;
    BuilderPaneMetrics m;
    float x0, x1, label_w;
    ImDrawList* dl;

    ImRect next(float h = 0.0f) {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float rh = h > 0.0f ? h : m.row + std::round(4.0f * L.text_scale);
        ImGui::Dummy(v2(x1 - x0, rh));
        return ImRect(v2(x0, p.y), v2(x1, p.y + rh));
    }
    void section(const std::string& title) {
        ImGui::Dummy(v2(1.0f, std::round(6.0f * L.text_scale)));
        const ImRect r = next(m.row);
        caps(L, dl, v2(r.Min.x + m.pad, r.Min.y + (r.GetHeight() - extent(L.ui, L.small_size, "Ag").y) * 0.5f), L.secondary, title);
        dl->AddRectFilled(v2(r.Min.x + m.pad, r.Max.y - 1.0f), v2(r.Max.x - m.pad, r.Max.y), L.rule_faint);
    }
    ImRect label(const ImRect& r, const std::string& name, bool required = false, ImU32 ink = 0) {
        const float th = extent(L.ui, L.ui_size, "Ag").y;
        const std::string shown = fit(L.ui, L.ui_size, name, label_w - m.pad * 2.0f);
        put(dl, L.ui, L.ui_size, v2(r.Min.x + m.pad, r.Min.y + (r.GetHeight() - th) * 0.5f), ink ? ink : L.secondary, shown);
        if (required) {
            const float w = extent(L.ui, L.ui_size, shown).x;
            put(dl, L.ui, L.ui_size, v2(r.Min.x + m.pad + w + 2.0f, r.Min.y + (r.GetHeight() - th) * 0.5f), L.faint, "*");
        }
        return ImRect(v2(r.Min.x + label_w, r.Min.y), v2(r.Max.x - m.pad, r.Max.y));
    }
};

// ---- ⌘K ------------------------------------------------------------------------------------

bool BuilderPane::widget_key(const std::string& id, const ImRect& r, bool shown) {
    if (!shown) return false;
    const SpecLook& L = draw_->look;
    return key_button_at(L, id.c_str(), r, "\xE2\x8C\x98K", "", true, false);
}

void BuilderPane::widget_under(const std::string& anchor, float x, float w) {
    if (!widget_.height || !widget_.draw) return;
    const float h = widget_.height(anchor, w);
    if (h <= 0.0f) return;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(v2(w, h));
    widget_.draw(anchor, SpecRect{x, p.y, w, h});
    ImGui::SetCursorScreenPos(v2(p.x, p.y + h));
    ImGui::Dummy(v2(0.0f, 0.0f));
}

// A property row's ⌘K: at the right end of its label, shown while the row is
// under the pointer or its prompt is open; then the prompt under the row. The
// property is the panel's, or with `source` given the spec source's.
void BuilderPane::property_key(Rows& rows, const ImRect& r, const std::string& panel, const std::string& path,
                               const std::string& source) {
    if (!widget_.open_property) return;
    const std::string anchor = "prop:" + path;
    const SpecLook& L = draw_->look;
    const float kw = key_button_width(L, "\xE2\x8C\x98K", "");
    const float rh = std::min(r.GetHeight(), draw_->m.row);
    const ImRect kr(v2(r.Min.x + rows.label_w - kw - std::round(4.0f * L.text_scale), r.Min.y + std::round(2.0f * L.text_scale)),
                    v2(r.Min.x + rows.label_w - std::round(4.0f * L.text_scale), r.Min.y + rh - std::round(2.0f * L.text_scale)));
    draw_->item("cmdk:" + anchor, kr);
    const bool open = widget_.height && widget_.height(anchor, r.GetWidth()) > 0.0f;
    const bool shown = open || ImGui::IsMouseHoveringRect(r.Min, r.Max, false);
    const ImVec2 back = ImGui::GetCursorScreenPos();
    ImGui::PushID(anchor.c_str());
    // A plate under the key, so that the label's end does not show through.
    if (shown) rows.dl->AddRectFilled(v2(kr.Min.x - 2.0f, r.Min.y), v2(kr.Max.x + 2.0f, r.Min.y + rh), L.ground);
    if (widget_key("##cmdk.prop", kr, shown)) widget_.open_property(panel, source, path);
    ImGui::PopID();
    ImGui::SetCursorScreenPos(back);
    widget_under(anchor, r.Min.x, r.GetWidth());
}

void BuilderPane::draw_properties(const SpecRect& rect, SpecRuntime& runtime) {
    Draw& d = *draw_;
    const SpecLook& L = d.look;
    const float s = L.text_scale;
    ImGui::SetCursorScreenPos(v2(rect.x, rect.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, v2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, L.ground);
    if (ImGui::BeginChild("##builder.properties", v2(rect.w, rect.h), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        // The tab strip of the column's upper part: Properties, with the
        // focused item's name at its right in mono.
        const ImRect bar(v2(rect.x, rect.y), v2(rect.right(), rect.y + d.m.header));
        dl->AddRectFilled(bar.Min, bar.Max, L.ground);
        dl->AddRectFilled(v2(bar.Min.x, bar.Max.y - 1.0f), bar.Max, L.rule);
        {
            const std::string tab = "Properties";
            const float pad = std::round(12.0f * s);
            const ImVec2 ts = extent(L.ui, L.ui_size, tab);
            const ImRect t(bar.Min, v2(bar.Min.x + ts.x + 2.0f * pad, bar.Max.y));
            dl->AddRectFilled(t.Min, v2(t.Max.x, t.Max.y), L.plate);
            dl->AddRectFilled(t.Min, v2(t.Max.x, t.Min.y + 1.0f), L.accent);
            dl->AddRectFilled(v2(t.Max.x, t.Min.y), v2(t.Max.x + 1.0f, t.Max.y), L.rule);
            put(dl, L.ui, L.ui_size, v2(t.Min.x + pad, t.Min.y + (bar.GetHeight() - ts.y) * 0.5f), L.text, tab);
            std::string what = !focus_panel_.empty() ? focus_panel_ : !focus_source_.empty() ? source_id() : open_ ? open_->spec.applet.id : "";
            const float right = bar.Max.x - std::round(8.0f * s);
            const std::string w = fit(L.mono, L.mono_size, what, right - t.Max.x - pad);
            put(dl, L.mono, L.mono_size, v2(right - extent(L.mono, L.mono_size, w).x, bar.Min.y + (bar.GetHeight() - d.mono_h()) * 0.5f),
                 L.faint, w);
        }
        ImGui::SetCursorScreenPos(v2(rect.x, bar.Max.y));
        if (ImGui::BeginChild("##properties.body", v2(rect.w, std::max(1.0f, rect.h - d.m.header)), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoBackground)) {
            Rows rows{L, d.m, rect.x, rect.right() - (ImGui::GetScrollMaxY() > 0.0f ? ImGui::GetStyle().ScrollbarSize : 0.0f),
                      std::round(std::min(rect.w * 0.36f, 8.0f * d.m.line)), ImGui::GetWindowDrawList()};
            if (const spec::Panel* p = panel()) draw_panel_properties(rows, *p, runtime);
            else if (!focus_source_.empty()) draw_source_properties(rows);
            else draw_spec_properties(rows);
            ImGui::Dummy(v2(1.0f, d.m.pad));
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void BuilderPane::draw_panel_properties(Rows& rows, const spec::Panel& panel_in, SpecRuntime& runtime) {
    Draw& d = *draw_;
    // A copy: a gesture below reads the open spec again, which frees the
    // panel it was drawn from.
    const spec::Panel p = panel_in;
    const SpecLook& L = d.look;
    const float s = L.text_scale;
    const std::string pid = p.id;
    rows.section("Panel");
    {   // Id: a rename on Enter or when the field is left.
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Id");
        const Field f = text_field(d.bufs, L, "##prop.id." + pid, v, pid, "", true, false, ImGuiInputTextFlags_AutoSelectAll);
        if ((f.entered || f.deactivated) && f.value != pid && !f.value.empty()) set_panel_id(f.value);
        d.item("prop:id", v);
    }
    {   // Primitive: a choice among the seven.
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Primitive");
        ImGui::SetCursorScreenPos(v.Min);
        const bool open_it = ImGui::InvisibleButton("##prop.primitive", v.GetSize());
        const bool hovered = ImGui::IsItemHovered();
        d.item("prop:primitive", v);
        put(rows.dl, L.ui, L.ui_size, v2(v.Min.x, v.Min.y + (v.GetHeight() - d.ui_h()) * 0.5f), hovered ? L.accent : L.text,
             pane_primitive_noun(p.primitive));
        sternbild::rule_h(rows.dl, std::floor(v.Min.x), std::floor(v.Max.x), std::floor(v.Max.y) - 1.0f, L.rule);
        if (open_it) ImGui::OpenPopup("##prop.primitive.popup");
        if (ImGui::BeginPopup("##prop.primitive.popup")) {
            for (Primitive q : pane_primitives()) {
                ImGui::PushID(spec::to_string(q));
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float w = std::max(v.GetWidth(), std::round(160.0f * s));
                const bool click = ImGui::InvisibleButton("##q", v2(w, d.m.row));
                const ImRect qr(at, v2(at.x + w, at.y + d.m.row));
                d.item(std::string("choice:primitive:") + spec::to_string(q), qr);
                if (ImGui::IsItemHovered()) ImGui::GetWindowDrawList()->AddRectFilled(qr.Min, qr.Max, L.plate_hover);
                put(ImGui::GetWindowDrawList(), L.ui, L.ui_size, v2(at.x + d.m.pad, at.y + (d.m.row - d.ui_h()) * 0.5f),
                     q == p.primitive ? L.accent : L.text, pane_primitive_noun(q));
                if (click) {
                    set_primitive(q);
                    ImGui::CloseCurrentPopup();
                }
                ImGui::PopID();
            }
            ImGui::EndPopup();
        }
    }
    {   // Title: typed, one undo step per edit of the field.
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Title");
        const std::string shown = p.title.value_or("");
        const Field f = text_field(d.bufs, L, "##prop.title." + pid, v, shown, spec_panel_title(open_->spec, p), false);
        if (f.activated) begin_gesture("title." + pid);
        if (f.changed) set_title(f.value);
        if (f.deactivated) end_gesture();
        d.item("prop:title", v);
        property_key(rows, r, pid, "title");
    }

    // ---- bindings: one row per slot, required slots marked.
    rows.section("Bindings");
    for (const PaneBindingRow& b : binding_rows()) {
        const std::string name = b.slot + ":" + std::to_string(b.index);
        const ImRect r = rows.next();
        const bool first = !b.list || b.index == 0;
        const ImRect v = rows.label(r, first ? b.slot : std::string(), first && b.required);
        const float xw = std::round(18.0f * s);
        const bool removable = !b.binding.empty();
        const ImRect field(v.Min, v2(v.Max.x - (removable ? xw + std::round(4.0f * s) : 0.0f), v.Max.y));
        ImGui::SetCursorScreenPos(field.Min);
        ImGui::PushID(("bind." + name).c_str());
        const bool open_it = ImGui::InvisibleButton("##bind", v2(std::max(1.0f, field.GetWidth()), field.GetHeight()));
        const bool hovered = ImGui::IsItemHovered();
        d.item("bind:" + name, field);
        const std::string shown = b.add ? std::string("+") : b.binding.empty() ? std::string("none") : b.binding;
        put(rows.dl, L.mono, L.mono_size, v2(field.Min.x, field.Min.y + (field.GetHeight() - d.mono_h()) * 0.5f),
             b.binding.empty() ? L.faint : hovered ? L.accent : L.text, fit(L.mono, L.mono_size, shown, field.GetWidth()));
        sternbild::rule_h(rows.dl, std::floor(field.Min.x), std::floor(field.Max.x), std::floor(field.Max.y) - 1.0f,
                          hovered ? L.secondary : L.rule);
        if (open_it) {
            ImGui::OpenPopup("##bind.popup");
            d.popup = name;
        }
        if (removable) {
            const ImRect xr(v2(v.Max.x - xw, v.Min.y), v.Max);
            ImGui::SetCursorScreenPos(xr.Min);
            const bool rm = ImGui::InvisibleButton("##bind.x", xr.GetSize());
            const bool xh = ImGui::IsItemHovered();
            d.item("bind.remove:" + name, xr);
            const std::vector<PaneBindingChoice> choices = binding_choices(b.slot, b.index);
            bool can = false;
            std::string why;
            for (const auto& c : choices)
                if (c.binding.empty()) {
                    can = c.fits;
                    why = c.message;
                }
            const ImVec2 c = xr.GetCenter();
            const float a = std::round(3.5f * s);
            const ImU32 ink = !can ? L.rule : xh ? L.text : L.faint;
            rows.dl->AddLine(v2(c.x - a, c.y - a), v2(c.x + a, c.y + a), ink, 1.0f);
            rows.dl->AddLine(v2(c.x - a, c.y + a), v2(c.x + a, c.y - a), ink, 1.0f);
            if (xh && !can && !why.empty()) ImGui::SetTooltip("%s", why.c_str());
            if (rm && can) remove_binding(b.slot, b.index);
        }
        if (ImGui::BeginPopup("##bind.popup")) {
            const std::vector<PaneBindingChoice> choices = binding_choices(b.slot, b.index);
            const float w = std::max(field.GetWidth(), std::round(220.0f * s));
            std::string last_source;
            for (const auto& c : choices) {
                if (c.binding.empty()) continue;
                ImGui::PushID(c.binding.c_str());
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const bool click = ImGui::InvisibleButton("##c", v2(w, d.m.row));
                const bool ch = ImGui::IsItemHovered();
                const ImRect cr(at, v2(at.x + w, at.y + d.m.row));
                d.item("choice:" + c.binding, cr);
                ImDrawList* pdl = ImGui::GetWindowDrawList();
                if (ch && c.fits) pdl->AddRectFilled(cr.Min, cr.Max, L.plate_hover);
                const float indent = c.column.empty() ? 0.0f : std::round(14.0f * s);
                const std::string label = c.column.empty() ? c.source : c.column;
                const ImU32 ink = !c.fits ? L.faint : c.binding == b.binding ? L.accent : L.text;
                put(pdl, L.mono, L.mono_size, v2(at.x + d.m.pad + indent, at.y + (d.m.row - d.mono_h()) * 0.5f), ink, label);
                if (ch && !c.fits && !c.message.empty()) ImGui::SetTooltip("%s", c.message.c_str());
                if (click && c.fits) {
                    set_binding(b.slot, b.index, c.binding);
                    ImGui::CloseCurrentPopup();
                }
                ImGui::PopID();
            }
            if (choices.empty()) put(ImGui::GetWindowDrawList(), L.ui, L.ui_size, ImGui::GetCursorScreenPos(), L.faint, "No sources");
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }

    // ---- options: exactly those the primitive takes.
    const std::vector<PaneOption>& opts = pane_options(p.primitive);
    if (!opts.empty()) rows.section("Options");
    const PreparedPanel* prepared = runtime.prepared(pid);
    for (const PaneOption& o : opts) {
        const std::string oname = o.name;
        const float option_top = ImGui::GetCursorScreenPos().y;
        const json current = [&]() -> json {
            const json all = spec::to_json(p).value("options", json::object());
            return all.contains(oname) ? all[oname] : json(nullptr);
        }();
        switch (o.kind) {
            case PaneOptionKind::text: {
                const ImRect r = rows.next();
                const ImRect v = rows.label(r, o.label);
                const Field f = text_field(d.bufs, L, "##opt." + oname + "." + pid, v,
                                           current.is_string() ? current.get<std::string>() : std::string(), "", false);
                if (f.activated) begin_gesture("option." + oname + "." + pid);
                if (f.changed) set_option(oname, f.value.empty() ? json(nullptr) : json(f.value));
                if (f.deactivated) end_gesture();
                d.item("option:" + oname, v);
                break;
            }
            case PaneOptionKind::longtext: {
                const ImRect r = rows.next(std::round(5.0f * d.m.row));
                const ImRect v = rows.label(r, o.label);
                rows.dl->AddRectFilled(v.Min, v.Max, L.deep);
                rows.dl->AddRect(v.Min, v.Max, L.rule, 0.0f, 0, 1.0f);
                const ImRect in(v2(v.Min.x + 4.0f * s, v.Min.y + 2.0f * s), v2(v.Max.x - 4.0f * s, v.Max.y - 2.0f * s));
                const Field f = text_field(d.bufs, L, "##opt." + oname + "." + pid, in,
                                           current.is_string() ? current.get<std::string>() : std::string(), "", false, true);
                if (f.activated) begin_gesture("option." + oname + "." + pid);
                if (f.changed) set_option(oname, f.value.empty() ? json(nullptr) : json(f.value));
                if (f.deactivated) end_gesture();
                d.item("option:" + oname, v);
                break;
            }
            case PaneOptionKind::flag: {
                const ImRect r = rows.next();
                const ImRect v = rows.label(r, o.label);
                const bool on = current.is_boolean() && current.get<bool>();
                if (flag_box(L, ("##opt." + oname).c_str(), v, on)) set_option(oname, on ? json(nullptr) : json(true));
                d.item("option:" + oname, ImRect(v.Min, v2(v.Min.x + std::round(17.0f * s), v.Max.y)));
                break;
            }
            case PaneOptionKind::colormap: {
                const ImRect r = rows.next();
                const ImRect v = rows.label(r, o.label);
                const std::string now = current.is_string() ? current.get<std::string>() : std::string();
                float x = v.Min.x;
                const auto& maps = pane_colormaps();
                const float gap = std::round(4.0f * s);
                const float w = std::floor((v.GetWidth() - gap * (float)(maps.size() - 1)) / (float)maps.size());
                for (const auto& name : maps) {
                    const ImRect c(v2(x, v.Min.y + std::round(3.0f * s)), v2(x + w, v.Max.y - std::round(3.0f * s)));
                    ImGui::SetCursorScreenPos(c.Min);
                    ImGui::PushID(name.c_str());
                    const bool click = ImGui::InvisibleButton("##cm", c.GetSize());
                    const bool ch = ImGui::IsItemHovered();
                    ImGui::PopID();
                    d.item("colormap:" + name, c);
                    // A ramp of the colormap under its name.
                    const uint32_t* lut = spec_colormap_lut(spec_colormap(name, 0));
                    const float ramp_h = std::round(3.0f * s);
                    const int steps = 24;
                    for (int i = 0; i < steps; ++i) {
                        const float x0 = c.Min.x + (c.GetWidth() * (float)i) / (float)steps;
                        const float x1 = c.Min.x + (c.GetWidth() * (float)(i + 1)) / (float)steps;
                        const uint32_t px = lut ? lut[(i * 255) / (steps - 1)] : 0xff808080u;
                        rows.dl->AddRectFilled(v2(std::floor(x0), c.Max.y - ramp_h), v2(std::ceil(x1), c.Max.y), px | 0xff000000u);
                    }
                    const bool on = name == now;
                    if (ch) rows.dl->AddRectFilled(c.Min, v2(c.Max.x, c.Max.y - ramp_h), L.plate_hover);
                    if (on) rows.dl->AddRect(v2(c.Min.x - 1.0f, c.Min.y - 1.0f), v2(c.Max.x + 1.0f, c.Max.y + 1.0f), L.accent, 0.0f, 0, 1.0f);
                    const std::string label = fit(L.mono, L.mono_size, name, c.GetWidth() - 4.0f);
                    const ImVec2 ts = extent(L.mono, L.mono_size, label);
                    put(rows.dl, L.mono, L.mono_size, v2(c.Min.x + (c.GetWidth() - ts.x) * 0.5f, c.Min.y + (c.GetHeight() - ramp_h - ts.y) * 0.5f),
                         on ? L.text : L.secondary, label);
                    if (click) set_option(oname, on ? json(nullptr) : json(name));
                    x += w + gap;
                }
                break;
            }
            case PaneOptionKind::range: {
                const ImRect r = rows.next();
                const ImRect v = rows.label(r, o.label);
                // The data's extent as the placeholder of each end.
                double lo = 0.0, hi = 1.0;
                bool data_known = false;
                if (prepared && prepared->state.ready()) {
                    if (p.primitive == Primitive::heatmap || p.primitive == Primitive::image) {
                        lo = prepared->grid.vmin;
                        hi = prepared->grid.vmax;
                        data_known = true;
                    } else if (p.primitive == Primitive::histogram) {
                        lo = prepared->histogram.lo;
                        hi = prepared->histogram.hi;
                        data_known = true;
                    } else if (p.primitive == Primitive::series) {
                        bool any = false;
                        for (const auto& line : prepared->series.lines) {
                            const auto& vals = oname == "x_range" ? line.x : line.y;
                            for (double x : vals) {
                                if (!any) lo = hi = x, any = true;
                                lo = std::min(lo, x);
                                hi = std::max(hi, x);
                            }
                        }
                        data_known = any;
                    }
                }
                const bool set = current.is_array() && current.size() == 2;
                const double a = set ? current[0].get<double>() : lo, b = set ? current[1].get<double>() : hi;
                const float gap = std::round(10.0f * s);
                const float half = std::floor((v.GetWidth() - gap) * 0.5f);
                const ImRect ra(v.Min, v2(v.Min.x + half, v.Max.y)), rb(v2(v.Min.x + half + gap, v.Min.y), v.Max);
                const Field fa = text_field(d.bufs, L, "##opt." + oname + ".min." + pid, ra, set ? fmt_number(a) : std::string(),
                                            data_known ? fmt_number(lo) : "min", true, false, ImGuiInputTextFlags_AutoSelectAll);
                if (fa.activated) begin_gesture("option." + oname + "." + pid);
                double na = 0;
                if (fa.changed) {
                    if (fa.value.empty() && !set) {
                    } else if (fa.value.empty()) set_option(oname, nullptr);
                    else if (parse_number(fa.value, &na) && na < b) set_option(oname, json::array({na, b}));
                }
                if (fa.deactivated) end_gesture();
                const Field fb = text_field(d.bufs, L, "##opt." + oname + ".max." + pid, rb, set ? fmt_number(b) : std::string(),
                                            data_known ? fmt_number(hi) : "max", true, false, ImGuiInputTextFlags_AutoSelectAll);
                if (fb.activated) begin_gesture("option." + oname + "." + pid);
                double nb = 0;
                if (fb.changed) {
                    if (fb.value.empty() && !set) {
                    } else if (fb.value.empty()) set_option(oname, nullptr);
                    else if (parse_number(fb.value, &nb) && a < nb) set_option(oname, json::array({a, nb}));
                }
                if (fb.deactivated) end_gesture();
                d.item("option:" + oname + ":min", ra);
                d.item("option:" + oname + ":max", rb);
                break;
            }
            case PaneOptionKind::bins: {
                const ImRect r = rows.next();
                const ImRect v = rows.label(r, o.label);
                const Field f = text_field(d.bufs, L, "##opt.bins." + pid, v,
                                           current.is_number() ? fmt_number(current.get<double>()) : std::string(), "64", true, false,
                                           ImGuiInputTextFlags_AutoSelectAll);
                if (f.activated) begin_gesture("option.bins." + pid);
                int64_t n = 0;
                if (f.changed) {
                    if (f.value.empty()) set_option(oname, nullptr);
                    else if (parse_int(f.value, &n) && n >= 1 && n <= 65536) set_option(oname, n);
                }
                if (f.deactivated) end_gesture();
                d.item("option:bins", v);
                break;
            }
            case PaneOptionKind::columns: {
                // One row per column of the bound table, on when shown.
                std::vector<std::string> all;
                const auto rb = p.bind.find("rows");
                if (rb != p.bind.end() && !rb->second.bindings.empty()) {
                    const auto sh = spec_shapes_.find(rb->second.bindings.front());
                    if (sh != spec_shapes_.end())
                        for (const auto& c : sh->second.columns) all.push_back(c.name);
                }
                std::vector<std::string> shown;
                if (current.is_array())
                    for (const auto& c : current) shown.push_back(c.get<std::string>());
                for (size_t i = 0; i < all.size(); ++i) {
                    const ImRect r = rows.next();
                    const ImRect v = rows.label(r, i == 0 ? o.label : "");
                    const bool on = shown.empty() || std::find(shown.begin(), shown.end(), all[i]) != shown.end();
                    if (flag_box(L, ("##col." + all[i]).c_str(), v, on)) {
                        std::vector<std::string> next;
                        for (const auto& c : all) {
                            const bool was = shown.empty() || std::find(shown.begin(), shown.end(), c) != shown.end();
                            if (c == all[i] ? !on : was) next.push_back(c);
                        }
                        if (next.size() == all.size()) set_option(oname, nullptr);
                        else if (!next.empty()) set_option(oname, next);
                    }
                    d.item("option:columns:" + all[i], ImRect(v.Min, v2(v.Min.x + std::round(17.0f * s), v.Max.y)));
                    put(rows.dl, L.mono, L.mono_size, v2(v.Min.x + std::round(22.0f * s), v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f),
                         on ? L.text : L.faint, all[i]);
                }
                break;
            }
        }
        // ⌘K beside the option, and the inline prompt on it under its row.
        const float option_bottom = ImGui::GetCursorScreenPos().y;
        property_key(rows, ImRect(v2(rows.x0, option_top), v2(rows.x1, option_bottom)), pid, "options." + oname);
    }

    // ---- the footer: Remove.
    ImGui::Dummy(v2(1.0f, std::round(10.0f * s)));
    const ImRect r = rows.next(d.m.row + std::round(6.0f * s));
    const std::string key = "\xE2\x8C\xAB", label = "Remove";
    const ImRect b(v2(r.Min.x + d.m.pad, r.Min.y + std::round(2.0f * s)),
                   v2(r.Min.x + d.m.pad + key_button_width(L, key, label), r.Max.y - std::round(2.0f * s)));
    if (key_button_at(L, "##panel.remove", b, key, label, true, false)) remove_focused();
    d.item("prop:remove", b);
}

void BuilderPane::draw_source_properties(Rows& rows) {
    Draw& d = *draw_;
    const SpecLook& L = d.look;
    const float s = L.text_scale;
    // Copies: a gesture below reads the open spec and the rows again.
    std::optional<spec::Source> src_copy;
    if (const spec::Source* x = spec_source()) src_copy = *x;
    const spec::Source* src = src_copy ? &*src_copy : nullptr;
    std::optional<PaneSource> row_copy;
    if (const PaneSource* x = find_source(focus_source_)) row_copy = *x;
    const PaneSource* row = row_copy ? &*row_copy : nullptr;
    rows.section("Source");
    {
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Id");
        const std::string id = source_id();
        const Field f = text_field(d.bufs, L, "##src.id." + focus_source_, v, id, "", true, false, ImGuiInputTextFlags_AutoSelectAll);
        if ((f.entered || f.deactivated) && f.value != id && !f.value.empty()) set_source_id(f.value);
        d.item("prop:source.id", v);
    }
    const std::string reads = src ? (src->catalog ? *src->catalog : src->from ? "from " + *src->from : std::string("sql"))
                                  : row ? row->catalog : std::string();
    {
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, src && src->from ? "From" : "Catalog");
        put(rows.dl, L.mono, L.mono_size, v2(v.Min.x, v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f), L.text,
             fit(L.mono, L.mono_size, reads, v.GetWidth()));
        // ⌘K beside what a spec source reads: its catalog, from or SQL.
        if (src) property_key(rows, r, "", src->catalog ? "catalog" : src->from ? "from" : "sql", src->id);
    }
    const std::vector<PaneShape> shapes = transform_shapes();
    {
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Shape");
        const std::string sh = shapes.empty() ? std::string() : pane_shape_text(shapes.front());
        put(rows.dl, L.mono, L.mono_size, v2(v.Min.x, v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f), L.text,
             fit(L.mono, L.mono_size, sh, v.GetWidth()));
    }
    const bool tabular = !shapes.empty() && shapes.front().tabular;
    if (tabular) {
        rows.section("Columns");
        for (const auto& c : shapes.front().columns) {
            const ImRect r = rows.next(d.m.row);
            const ImRect v = rows.label(r, c.name, false, L.text);
            put(rows.dl, L.mono, L.mono_size, v2(v.Min.x, v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f), L.faint, c.type);
        }
        return;
    }
    rows.section("Transforms");
    const float transforms_top = ImGui::GetCursorScreenPos().y;
    const size_t n = src ? src->transforms.size() : 0;
    for (size_t i = 0; i < n; ++i) {
        const spec::Transform& t = src->transforms[i];
        const std::string base = "transform:" + std::to_string(i);
        // The step's head: its number and op, the shape after it at the right, ×.
        const ImRect r = rows.next(d.m.row);
        {
            char num[8];
            std::snprintf(num, sizeof num, "%zu", i + 1);
            put(rows.dl, L.mono, L.mono_size, v2(r.Min.x + d.m.pad, r.Min.y + (r.GetHeight() - d.mono_h()) * 0.5f), L.faint, num);
            put(rows.dl, L.ui, L.ui_size, v2(r.Min.x + d.m.pad + std::round(18.0f * s), r.Min.y + (r.GetHeight() - d.ui_h()) * 0.5f),
                 L.text, spec::to_string(t.op));
            const float xw = std::round(18.0f * s);
            const std::string after = i + 1 < shapes.size() ? pane_shape_text(shapes[i + 1]) : std::string();
            const std::string shown = "\xE2\x86\x92 " + (after.empty() ? std::string("?") : after);
            put(rows.dl, L.mono, L.mono_size,
                 v2(r.Max.x - d.m.pad - xw - std::round(6.0f * s) - extent(L.mono, L.mono_size, shown).x,
                    r.Min.y + (r.GetHeight() - d.mono_h()) * 0.5f),
                 after.empty() ? L.faint : L.secondary, shown);
            const ImRect xr(v2(r.Max.x - d.m.pad - xw, r.Min.y), v2(r.Max.x - d.m.pad, r.Max.y));
            ImGui::SetCursorScreenPos(xr.Min);
            ImGui::PushID(base.c_str());
            const bool rm = ImGui::InvisibleButton("##x", xr.GetSize());
            const bool xh = ImGui::IsItemHovered();
            ImGui::PopID();
            d.item(base + ":remove", xr);
            const ImVec2 c = xr.GetCenter();
            const float a = std::round(3.5f * s);
            rows.dl->AddLine(v2(c.x - a, c.y - a), v2(c.x + a, c.y + a), xh ? L.text : L.faint, 1.0f);
            rows.dl->AddLine(v2(c.x - a, c.y + a), v2(c.x + a, c.y - a), xh ? L.text : L.faint, 1.0f);
            if (rm) remove_transform(i);
        }
        // Its fields.
        json tj = {{"op", spec::to_string(t.op)}};
        // A number field; `right` is drawn at the row's right end, in the
        // accent colour when `marked` (a slice's index range, or the note that
        // an index was set to the end of its range).
        auto field_int = [&](const char* key, const std::optional<int64_t>& value, bool required,
                             const std::string& right = {}, bool marked = false) {
            const ImRect fr = rows.next(d.m.row);
            const ImRect v = rows.label(fr, std::string("  ") + key);
            const std::string id = "##" + base + "." + key + "." + src->id;
            ImRect fv = v;
            if (!right.empty()) {
                const std::string shown = fit(L.mono, L.mono_size, right, v.GetWidth() * 0.6f);
                const float rw = extent(L.mono, L.mono_size, shown).x;
                put(rows.dl, L.mono, L.mono_size, v2(v.Max.x - rw, v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f),
                    marked ? L.accent : L.faint, shown);
                fv = ImRect(v.Min, v2(std::max(v.Min.x + 8.0f, v.Max.x - rw - std::round(10.0f * s)), v.Max.y));
            }
            const Field f = text_field(d.bufs, L, id, fv, value ? std::to_string(*value) : std::string(), required ? "" : "none", true,
                                       false, ImGuiInputTextFlags_AutoSelectAll);
            d.item(base + ":" + key, v);
            if (f.activated) begin_gesture(base + "." + src->id);
            if (f.deactivated) end_gesture();
            if (!f.changed) return;
            int64_t x = 0;
            // A slice's index stays within its axis's range (set_slice_index()).
            if (t.op == spec::TransformOp::slice && std::string(key) == "index" && parse_int(f.value, &x)) {
                set_slice_index(i, x);
                return;
            }
            json next = json::parse(spec::to_json(*src)["transform"][i].dump());
            if (f.value.empty() && !required) next.erase(key);
            else if (parse_int(f.value, &x)) next[key] = x;
            else return;
            // A slice takes index or a range, never both.
            if (t.op == spec::TransformOp::slice && std::string(key) == "index" && next.contains("index")) {
                next.erase("start"); next.erase("stop"); next.erase("step");
            }
            if (t.op == spec::TransformOp::slice && std::string(key) != "index" && std::string(key) != "axis" && next.contains(key))
                next.erase("index");
            set_transform(i, next);
        };
        auto field_choice = [&](const char* key, const std::vector<std::string>& options, const std::string& now) {
            // The choices as words, on as many lines as they need.
            const float gap = std::round(2.0f * s), pad_w = std::round(10.0f * s);
            const float avail = rows.x1 - rows.m.pad - (rows.x0 + rows.label_w);
            int lines = 1;
            {
                float x = 0.0f;
                for (const auto& o : options) {
                    const float w = extent(L.mono, L.mono_size, o).x + pad_w;
                    if (x > 0.0f && x + w > avail) {
                        ++lines;
                        x = 0.0f;
                    }
                    x += w + gap;
                }
            }
            const ImRect fr = rows.next(d.m.row * (float)lines);
            const ImRect v = rows.label(ImRect(fr.Min, v2(fr.Max.x, fr.Min.y + d.m.row)), std::string("  ") + key);
            float x = v.Min.x, y = v.Min.y;
            for (const auto& o : options) {
                const float w = extent(L.mono, L.mono_size, o).x + pad_w;
                if (x > v.Min.x && x + w > v.Min.x + avail) {
                    x = v.Min.x;
                    y += d.m.row;
                }
                const ImRect c(v2(x, y + std::round(3.0f * s)), v2(x + w, y + d.m.row - std::round(3.0f * s)));
                ImGui::SetCursorScreenPos(c.Min);
                ImGui::PushID((base + key + o).c_str());
                const bool click = ImGui::InvisibleButton("##ch", c.GetSize());
                const bool ch = ImGui::IsItemHovered();
                ImGui::PopID();
                d.item(base + ":" + key + ":" + o, c);
                const bool on = o == now;
                if (on) rows.dl->AddRectFilled(c.Min, c.Max, L.plate);
                else if (ch) rows.dl->AddRectFilled(c.Min, c.Max, L.plate_hover);
                if (on) rows.dl->AddRectFilled(v2(c.Min.x, c.Max.y - 1.0f), c.Max, L.accent);
                put(rows.dl, L.mono, L.mono_size, v2(c.Min.x + std::round(5.0f * s), c.Min.y + (c.GetHeight() - d.mono_h()) * 0.5f),
                    on ? L.text : L.secondary, o);
                if (click && !on) {
                    json next = spec::to_json(*src)["transform"][i];
                    next[key] = o;
                    set_transform(i, next);
                }
                x += w + gap;
            }
        };
        // A slice's or a reduce's axis: every axis of the step's input, with
        // its index range and its name where the source carries one, one
        // choice each; the number alone when the input's shape is not known.
        auto field_axis = [&](const std::optional<int64_t>& now) {
            const std::vector<std::string> axes = transform_axes(i);
            if (axes.empty()) {
                field_int("axis", now, true);
                return;
            }
            const int64_t rank = (int64_t)axes.size();
            // Every axis the step uses is highlighted: a reduce's whole list.
            const std::vector<int64_t> chosen = transform_axes_chosen(i);
            const float gap = std::round(2.0f * s), pad_w = std::round(10.0f * s);
            const float avail = rows.x1 - rows.m.pad - (rows.x0 + rows.label_w);
            int lines = 1;
            {
                float x = 0.0f;
                for (const auto& o : axes) {
                    const float w = extent(L.mono, L.mono_size, o).x + pad_w;
                    if (x > 0.0f && x + w > avail) {
                        ++lines;
                        x = 0.0f;
                    }
                    x += w + gap;
                }
            }
            const ImRect fr = rows.next(d.m.row * (float)lines);
            const ImRect v = rows.label(ImRect(fr.Min, v2(fr.Max.x, fr.Min.y + d.m.row)), "  axis");
            d.item(base + ":axis", ImRect(v.Min, v2(v.Max.x, fr.Max.y)));
            float x = v.Min.x, y = v.Min.y;
            for (int64_t k = 0; k < rank; ++k) {
                const std::string& o = axes[(size_t)k];
                const float w = std::min(extent(L.mono, L.mono_size, o).x + pad_w, avail);
                if (x > v.Min.x && x + w > v.Min.x + avail) {
                    x = v.Min.x;
                    y += d.m.row;
                }
                const ImRect c(v2(x, y + std::round(3.0f * s)), v2(x + w, y + d.m.row - std::round(3.0f * s)));
                ImGui::SetCursorScreenPos(c.Min);
                ImGui::PushID((base + ":axis:" + std::to_string(k)).c_str());
                const bool click = ImGui::InvisibleButton("##ax", c.GetSize());
                const bool ch = ImGui::IsItemHovered();
                ImGui::PopID();
                d.item(base + ":axis:" + std::to_string(k), c);
                const bool on = std::find(chosen.begin(), chosen.end(), k) != chosen.end();
                if (on) rows.dl->AddRectFilled(c.Min, c.Max, L.plate);
                else if (ch) rows.dl->AddRectFilled(c.Min, c.Max, L.plate_hover);
                if (on) rows.dl->AddRectFilled(v2(c.Min.x, c.Max.y - 1.0f), c.Max, L.accent);
                put(rows.dl, L.mono, L.mono_size, v2(c.Min.x + std::round(5.0f * s), c.Min.y + (c.GetHeight() - d.mono_h()) * 0.5f),
                    on ? L.text : L.secondary, fit(L.mono, L.mono_size, o, c.GetWidth() - std::round(5.0f * s)));
                // A click on a reduce's chosen axis takes it out of the list.
                if (click && (!on || t.op == spec::TransformOp::reduce)) set_transform_axis(i, k);
                x += w + gap;
            }
        };
        const std::optional<int64_t> axis = t.axis.empty() ? std::nullopt : std::optional<int64_t>(t.axis.front());
        switch (t.op) {
            case spec::TransformOp::slice: {
                field_axis(axis);
                const bool noted = !transform_note().empty() && transform_note_step() == i;
                if (t.index || !(t.start || t.stop || t.step))
                    field_int("index", t.index, false, noted ? transform_note() : transform_index_range(i), noted);
                if (!t.index) {
                    field_int("start", t.start, false);
                    field_int("stop", t.stop, false);
                    field_int("step", t.step, false);
                }
                break;
            }
            case spec::TransformOp::reduce:
                field_axis(axis);
                field_choice("fn", {"mean", "sum", "min", "max", "std", "var", "norm", "argmax", "argmin", "entropy"},
                             t.fn.value_or("mean"));
                break;
            case spec::TransformOp::normalise:
                field_choice("mode", {"minmax", "zscore", "softmax", "l2"}, t.mode.value_or("minmax"));
                field_int("axis", axis, false);
                break;
        }
        (void)tj;
    }
    // Add: a step of each op.
    {
        ImGui::Dummy(v2(1.0f, std::round(4.0f * s)));
        const ImRect r = rows.next(d.m.row + std::round(6.0f * s));
        const ImRect v = rows.label(r, "Add");
        float x = v.Min.x;
        const struct { spec::TransformOp op; const char* label; } ops[] = {
            {spec::TransformOp::slice, "Slice"}, {spec::TransformOp::reduce, "Reduce"}, {spec::TransformOp::normalise, "Normalise"}};
        for (const auto& o : ops) {
            const float w = key_button_width(L, "", o.label);
            if (x + w > v.Max.x + 1.0f) break;
            const ImRect b(v2(x, r.Min.y + std::round(3.0f * s)), v2(x + w, r.Max.y - std::round(3.0f * s)));
            if (key_button_at(L, (std::string("##add.") + o.label).c_str(), b, "", o.label, true, false))
                add_transform(o.op);
            d.item(std::string("transform:add:") + spec::to_string(o.op), b);
            x += w + std::round(6.0f * s);
        }
    }
    // ⌘K beside a spec source's transforms, as one property, and the prompt
    // under them.
    if (src) property_key(rows, ImRect(v2(rows.x0, transforms_top), v2(rows.x1, ImGui::GetCursorScreenPos().y)), "",
                          "transform", src->id);
    if (src) {
        ImGui::Dummy(v2(1.0f, std::round(10.0f * s)));
        const ImRect r = rows.next(d.m.row + std::round(6.0f * s));
        const std::string key = "\xE2\x8C\xAB", label = "Remove";
        const ImRect b(v2(r.Min.x + d.m.pad, r.Min.y + std::round(2.0f * s)),
                       v2(r.Min.x + d.m.pad + key_button_width(L, key, label), r.Max.y - std::round(2.0f * s)));
        if (key_button_at(L, "##source.remove", b, key, label, true, false)) remove_focused();
        d.item("prop:source.remove", b);
    }
}

void BuilderPane::draw_spec_properties(Rows& rows) {
    Draw& d = *draw_;
    const SpecLook& L = d.look;
    if (!open_) return;
    const spec::Spec s = open_->spec;   // a copy: the fields below change the open spec
    const std::optional<std::string> path = open_->path;
    rows.section("Spec");
    {   // The applet's id: a rename of the whole spec on Enter or when left.
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Id");
        const Field f = text_field(d.bufs, L, "##spec.id", v, s.applet.id, "", true, false, ImGuiInputTextFlags_AutoSelectAll);
        if ((f.entered || f.deactivated) && f.value != s.applet.id && !f.value.empty()) set_spec_id(f.value);
        d.item("prop:spec.id", v);
    }
    {
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Title");
        const Field f = text_field(d.bufs, L, "##spec.title", v, s.applet.title.value_or(""), s.applet.id, false);
        if (f.activated) begin_gesture("spec.title");
        if (f.changed) set_spec_title(f.value);
        if (f.deactivated) end_gesture();
        d.item("prop:spec.title", v);
    }
    {
        const ImRect r = rows.next(std::round(4.0f * d.m.row));
        const ImRect v = rows.label(r, "Description");
        rows.dl->AddRectFilled(v.Min, v.Max, L.deep);
        rows.dl->AddRect(v.Min, v.Max, L.rule, 0.0f, 0, 1.0f);
        const float in = std::round(4.0f * L.text_scale);
        const Field f = text_field(d.bufs, L, "##spec.description", ImRect(v2(v.Min.x + in, v.Min.y + 2.0f), v2(v.Max.x - in, v.Max.y - 2.0f)),
                                   s.applet.description.value_or(""), "", false, true);
        if (f.activated) begin_gesture("spec.description");
        if (f.changed) set_spec_description(f.value);
        if (f.deactivated) end_gesture();
        d.item("prop:spec.description", v);
    }
    {
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Panels");
        put(rows.dl, L.mono, L.mono_size, v2(v.Min.x, v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f), L.text,
             std::to_string(s.panels.size()));
    }
    {
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "Sources");
        put(rows.dl, L.mono, L.mono_size, v2(v.Min.x, v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f), L.text,
             std::to_string(s.sources.size()));
    }
    {
        const ImRect r = rows.next();
        const ImRect v = rows.label(r, "File");
        put(rows.dl, L.mono, L.mono_size, v2(v.Min.x, v.Min.y + (v.GetHeight() - d.mono_h()) * 0.5f),
             path ? L.text : L.faint, fit(L.mono, L.mono_size, path.value_or("unsaved"), v.GetWidth()));
    }
}

// ---- the spec text --------------------------------------------------------------------------

void BuilderPane::draw_text_strip(const SpecRect& rect) {
    Draw& d = *draw_;
    const SpecLook& L = d.look;
    const float s = L.text_scale;
    ImGui::SetCursorScreenPos(v2(rect.x, rect.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, v2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, v2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, L.deep);
    if (ImGui::BeginChild("##builder.text", v2(rect.w, rect.h), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImRect head(v2(rect.x, rect.y), v2(rect.right(), rect.y + d.m.header));
        std::string sub;
        if (open_) sub = open_->path.value_or(open_->spec.applet.id + ".spec.toml");
        if (editing_ && draft_pending_)
            sub = std::to_string(diagnostics_.size()) + (diagnostics_.size() == 1 ? " error" : " errors");
        header_strip(L, dl, head, "Spec text", sub);
        {
            const std::string key = editing_ ? "esc" : "\xE2\x8C\x98" "E";
            const std::string label = editing_ ? "Done" : "Edit";
            const float w = key_button_width(L, key, label), h = std::round(d.m.header - 6.0f * s);
            const ImRect b(v2(head.Max.x - std::round(6.0f * s) - w, head.Min.y + std::round(3.0f * s)),
                           v2(head.Max.x - std::round(6.0f * s), head.Min.y + std::round(3.0f * s) + h));
            if (key_button_at(L, "##text.edit", b, key, label, open_.has_value(), editing_)) set_editing(!editing_);
            d.item("text:edit", b);
        }
        if (!editing_ && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_E)) set_editing(true);
        const float gutter = std::round(extent(L.mono, L.mono_size, "000").x + 18.0f * s);
        const ImRect body(v2(rect.x, head.Max.y), v2(rect.right(), rect.bottom()));
        const float line_h = std::round(extent(L.mono, L.mono_size, "Ag").y + 2.0f * s);
        std::map<int, const spec::Diagnostic*> marks;
        for (const auto& g : diagnostics_)
            if (g.line > 0 && !marks.count(g.line)) marks[g.line] = &g;
        d.item("text:body", body);
        float scroll_y = 0.0f;
        const float top_pad = std::round(4.0f * s);
        if (editing_) {
            // The draft in an editor; the gutter's numbers follow its scroll.
            const ImRect ed(v2(body.Min.x + gutter, body.Min.y + top_pad), v2(body.Max.x - std::round(4.0f * s), body.Max.y));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, v2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, v2(0.0f, 2.0f * s));
            const Field f = text_field(d.bufs, L, "##text.draft", ed, draft_, "", true, true);
            ImGui::PopStyleVar(2);
            if (f.activated) begin_gesture("text");
            if (f.changed) set_draft(f.value);
            if (f.deactivated) end_gesture();
            // The editor's own window, for its scroll.
            const ImGuiID id = ImGui::GetID("##text.draft");
            char name[512];
            std::snprintf(name, sizeof name, "%s/%s_%08X", ImGui::GetCurrentWindow()->Name, "##text.draft", id);
            if (ImGuiWindow* w = ImGui::FindWindowByName(name)) scroll_y = w->Scroll.y;
            const float eh = ImGui::GetFontSize();
            (void)eh;
            const float lh = extent(L.mono, L.mono_size, "Ag").y + 2.0f * s;
            ImGui::PushClipRect(body.Min, body.Max, true);
            int lines = 1;
            for (char c : draft_)
                if (c == '\n') ++lines;
            for (int i = 0; i < lines; ++i) {
                const float y = ed.Min.y + (float)i * lh - scroll_y;
                if (y + lh < body.Min.y || y > body.Max.y) continue;
                const auto mk = marks.find(i + 1);
                char num[16];
                std::snprintf(num, sizeof num, "%d", i + 1);
                const float nw = extent(L.mono, L.mono_size, num).x;
                if (mk != marks.end()) {
                    dl->AddRectFilled(v2(body.Min.x, y), v2(body.Min.x + gutter - std::round(6.0f * s), y + lh), L.plate_hover);
                    dl->AddRectFilled(v2(body.Min.x, y), v2(body.Min.x + 2.0f, y + lh), L.text);
                    if (ImGui::IsMouseHoveringRect(v2(body.Min.x, y), v2(body.Max.x, y + lh)))
                        ImGui::SetTooltip("%s  %s", mk->second->code.c_str(), mk->second->message.c_str());
                    // The code at the line's right end.
                    const ImVec2 cs = extent(L.mono, L.mono_size, mk->second->code);
                    put(dl, L.mono, L.mono_size, v2(body.Max.x - std::round(12.0f * s) - cs.x, y), L.secondary, mk->second->code);
                }
                put(dl, L.mono, L.mono_size, v2(body.Min.x + gutter - std::round(12.0f * s) - nw, y),
                     mk != marks.end() ? L.text : L.faint, num);
            }
            ImGui::PopClipRect();
        } else {
            // The open spec's canonical text, read-only, scrolling.
            ImGui::SetCursorScreenPos(body.Min);
            if (ImGui::BeginChild("##text.view", body.GetSize(), ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                ImDrawList* vdl = ImGui::GetWindowDrawList();
                const std::string& t = text();
                std::vector<std::pair<size_t, size_t>> lines;
                size_t at = 0;
                while (at <= t.size()) {
                    size_t end = t.find('\n', at);
                    if (end == std::string::npos) end = t.size();
                    if (at == t.size() && end == t.size() && !lines.empty()) break;
                    lines.emplace_back(at, end);
                    at = end + 1;
                }
                ImGui::Dummy(v2(1.0f, top_pad));
                ImGuiListClipper clip;
                clip.Begin((int)lines.size(), line_h);
                while (clip.Step()) {
                    for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
                        const ImVec2 p = ImGui::GetCursorScreenPos();
                        char num[16];
                        std::snprintf(num, sizeof num, "%d", i + 1);
                        const float nw = extent(L.mono, L.mono_size, num).x;
                        put(vdl, L.mono, L.mono_size, v2(p.x + gutter - std::round(12.0f * s) - nw, p.y), L.faint, num);
                        const std::string line = t.substr(lines[(size_t)i].first, lines[(size_t)i].second - lines[(size_t)i].first);
                        // Keys in secondary, values in parchment, table heads in parchment.
                        const size_t eq = line.find(" = ");
                        const float x = p.x + gutter;
                        if (!line.empty() && line.find_first_not_of(' ') != std::string::npos &&
                            line[line.find_first_not_of(' ')] == '[') {
                            put(vdl, L.mono, L.mono_size, v2(x, p.y), L.text, line);
                        } else if (eq != std::string::npos) {
                            const std::string k = line.substr(0, eq + 3);
                            put(vdl, L.mono, L.mono_size, v2(x, p.y), L.secondary, k);
                            put(vdl, L.mono, L.mono_size, v2(x + extent(L.mono, L.mono_size, k).x, p.y), L.text, line.substr(eq + 3));
                        } else {
                            put(vdl, L.mono, L.mono_size, v2(x, p.y), L.text, line);
                        }
                        ImGui::Dummy(v2(std::max(1.0f, gutter + extent(L.mono, L.mono_size, line).x), line_h));
                    }
                }
                // The last change comes into view: a line below the view ends
                // it with two lines after, one above starts it.
                if (text_follow_ >= 0) {
                    const float y = top_pad + (float)text_follow_ * line_h;
                    const float top = ImGui::GetScrollY(), view = ImGui::GetWindowHeight();
                    if (y < top) ImGui::SetScrollY(std::max(0.0f, y - line_h));
                    else if (y + line_h > top + view) ImGui::SetScrollY(y + 3.0f * line_h - view);
                    text_follow_ = -1;
                }
            }
            ImGui::EndChild();
        }
        dl->AddRectFilled(v2(body.Min.x + gutter - std::round(6.0f * s), body.Min.y), v2(body.Min.x + gutter - std::round(5.0f * s), body.Max.y),
                          L.rule_faint);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

}  // namespace caliper_host
