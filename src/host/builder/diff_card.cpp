// diff_card.cpp: the compact diff of a change to a spec and its drawing
// (diff_card.h), shared by the inline surface and the approval cards.
#include "builder/diff_card.h"

#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_theme.h"

#include <caliper_builder/spec/format.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

namespace caliper_host {

using nlohmann::json;
namespace spec = caliper_builder::spec;

// ---- the compact diff -------------------------------------------------------------------

namespace {

// A value as the card shows it: JSON, with whole numbers without a
// fraction and a space after each comma.
std::string show_value(const json& v) {
    if (v.is_number_float()) {
        const double d = v.get<double>();
        char buf[32];
        if (std::isfinite(d) && std::fabs(d) < 1e15 && d == std::floor(d)) std::snprintf(buf, sizeof buf, "%.0f", d);
        else std::snprintf(buf, sizeof buf, "%.6g", d);
        return buf;
    }
    if (v.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + show_value(v[i]);
        return s + "]";
    }
    if (v.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            s += (first ? "" : ", ") + json(it.key()).dump() + ": " + show_value(it.value());
            first = false;
        }
        return s + "}";
    }
    return v.dump();
}

std::string compact_value(const json& v, size_t max_value) {
    std::string s = show_value(v);
    if (s.size() > max_value && max_value > 3) {
        // Cut on a character boundary of UTF-8.
        size_t cut = max_value - 1;
        while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) --cut;
        s = s.substr(0, cut) + "\xE2\x80\xA6";   // …
    }
    return s;
}

// Flattens objects to dotted keys; arrays and scalars are leaves.
void flatten(const json& v, const std::string& prefix, std::map<std::string, json>* out) {
    if (v.is_object() && !v.empty()) {
        for (auto it = v.begin(); it != v.end(); ++it)
            flatten(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(), out);
    } else {
        (*out)[prefix] = v;
    }
}

void diff_objects(const json& before, const json& after, const std::string& label, bool strip_options, size_t max_value,
                  std::vector<WidgetDiffLine>* out) {
    std::map<std::string, json> a, b;
    flatten(before, "", &a);
    flatten(after, "", &b);
    std::set<std::string> keys;
    for (const auto& kv : a) keys.insert(kv.first);
    for (const auto& kv : b) keys.insert(kv.first);
    for (const std::string& k : keys) {
        auto ia = a.find(k), ib = b.find(k);
        if (ia != a.end() && ib != b.end() && ia->second == ib->second) continue;
        std::string key = k;
        if (strip_options && key.rfind("options.", 0) == 0) key = key.substr(8);
        if (!label.empty()) key = label + (key.empty() ? "" : "." + key);
        WidgetDiffLine line;
        line.key = key;
        if (ia == a.end()) {
            line.kind = WidgetDiffLine::Kind::added;
            line.after = compact_value(ib->second, max_value);
        } else if (ib == b.end()) {
            line.kind = WidgetDiffLine::Kind::removed;
            line.before = compact_value(ia->second, max_value);
        } else {
            line.before = compact_value(ia->second, max_value);
            line.after = compact_value(ib->second, max_value);
        }
        out->push_back(std::move(line));
    }
}

}  // namespace

std::vector<WidgetDiffLine> widget_compact_diff(const spec::Spec& before, const spec::Spec& after,
                                                const std::string& panel, size_t max_value) {
    std::vector<WidgetDiffLine> out;
    // The target panel first, its keys relative to it.
    const spec::Panel* pa = panel.empty() ? nullptr : before.find_panel(panel);
    const spec::Panel* pb = panel.empty() ? nullptr : after.find_panel(panel);
    if (pa && pb) {
        json ja = spec::to_json(*pa), jb = spec::to_json(*pb);
        ja.erase("id");
        jb.erase("id");
        diff_objects(ja, jb, "", true, max_value, &out);
    } else if (pa && !pb) {
        out.push_back({WidgetDiffLine::Kind::removed, "panel " + panel, panel, ""});
    }
    // The other panels and the sources: added, removed, or their changed keys.
    auto items = [&](const char* what, auto list_a, auto list_b) {
        std::map<std::string, json> a, b;
        for (const auto* x : list_a) a[x->id] = spec::to_json(*x);
        for (const auto* x : list_b) b[x->id] = spec::to_json(*x);
        for (const auto& kv : b)
            if (!a.count(kv.first)) out.push_back({WidgetDiffLine::Kind::added, std::string(what) + " " + kv.first, "", kv.first});
        for (const auto& kv : a) {
            auto it = b.find(kv.first);
            if (it == b.end()) {
                out.push_back({WidgetDiffLine::Kind::removed, std::string(what) + " " + kv.first, kv.first, ""});
            } else if (it->second != kv.second) {
                json x = kv.second, y = it->second;
                x.erase("id");
                y.erase("id");
                diff_objects(x, y, std::string(what) + " " + kv.first, false, max_value, &out);
            }
        }
    };
    std::vector<const spec::Panel*> panels_a, panels_b;
    for (const auto& p : before.panels)
        if (p.id != panel) panels_a.push_back(&p);
    for (const auto& p : after.panels)
        if (p.id != panel || !pa) panels_b.push_back(&p);
    if (!pa)
        for (auto it = panels_a.begin(); it != panels_a.end();) it = (*it)->id == panel ? panels_a.erase(it) : it + 1;
    items("panel", panels_a, panels_b);
    std::vector<const spec::Source*> sources_a, sources_b;
    for (const auto& s : before.sources) sources_a.push_back(&s);
    for (const auto& s : after.sources) sources_b.push_back(&s);
    items("source", sources_a, sources_b);
    const json la = before.layout ? spec::to_json(*before.layout) : json();
    const json lb = after.layout ? spec::to_json(*after.layout) : json();
    if (la != lb) out.push_back({WidgetDiffLine::Kind::changed, "layout", "", ""});
    return out;
}


std::vector<WidgetDiffLine> widget_item_lines(const json& item, size_t max_value) {
    std::map<std::string, json> keys;
    flatten(item, "", &keys);
    std::vector<WidgetDiffLine> out;
    for (const auto& kv : keys) {
        if (kv.first == "id" || kv.first.empty() || kv.second.is_null()) continue;
        if (kv.second.is_object() && kv.second.empty()) continue;
        WidgetDiffLine line;
        line.kind = WidgetDiffLine::Kind::added;
        line.key = kv.first;
        line.after = compact_value(kv.second, max_value);
        out.push_back(std::move(line));
    }
    // What the item is and its name first, then its keys in order.
    auto rank = [](const std::string& k) { return k == "primitive" ? 0 : k == "title" ? 1 : 2; };
    std::stable_sort(out.begin(), out.end(),
                     [&](const WidgetDiffLine& a, const WidgetDiffLine& b) { return rank(a.key) < rank(b.key); });
    return out;
}

// ---- text -------------------------------------------------------------------------------

ImVec2 card_text_extent(ImFont* f, float size, const std::string& s) {
    if (!f) f = ImGui::GetFont();
    return f->CalcTextSizeA(size, FLT_MAX, 0.0f, s.c_str(), s.c_str() + s.size());
}

std::string card_fit_text(ImFont* f, float size, const std::string& s, float w) {
    if (card_text_extent(f, size, s).x <= w) return s;
    static const std::string kDots = "\xE2\x80\xA6";
    if (card_text_extent(f, size, kDots).x > w) return {};
    size_t n = s.size();
    while (n > 0) {
        --n;
        while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) --n;
        const std::string t = s.substr(0, n) + kDots;
        if (card_text_extent(f, size, t).x <= w) return t;
    }
    return kDots;
}

void card_draw_text(ImDrawList* dl, ImFont* f, float size, ImVec2 p, ImU32 c, const std::string& s) {
    dl->AddText(f ? f : ImGui::GetFont(), size, ImVec2(std::round(p.x), std::round(p.y)), c, s.c_str(),
                s.c_str() + s.size());
}

// ---- drawing ----------------------------------------------------------------------------

void draw_compact_diff(ImDrawList* dl, const SpecLook& L, const ImRect& d, const std::vector<WidgetDiffLine>& lines,
                       int shown, bool folded) {
    const float s = L.text_scale;
    const int rows = shown + (folded ? 1 : 0);
    if (rows <= 0 || d.GetWidth() <= 0.0f || d.GetHeight() <= 0.0f) return;
    const float lh = d.GetHeight() / (float)rows;
    const float mono_h = card_text_extent(L.mono, L.mono_size, "Ag").y;
    // The key, the old value struck through in faint, the arrow, the new
    // value in parchment.
    float key_w = 0.0f;
    for (int i = 0; i < shown && i < (int)lines.size(); ++i)
        key_w = std::max(key_w, card_text_extent(L.mono, L.mono_size, lines[(size_t)i].key).x);
    key_w = std::min(key_w, d.GetWidth() * 0.4f) + std::round(16.0f * s);
    for (int i = 0; i < shown && i < (int)lines.size(); ++i) {
        const WidgetDiffLine& line = lines[(size_t)i];
        const float y = d.Min.y + lh * (float)i, ty = y + (lh - mono_h) * 0.5f;
        float x = d.Min.x;
        card_draw_text(dl, L.mono, L.mono_size, ImVec2(x, ty), L.secondary,
                       card_fit_text(L.mono, L.mono_size, line.key, key_w - 8.0f * s));
        x += key_w;
        const std::string arrow = "\xE2\x86\x92";   // →
        // A whole panel or source added or removed ("panel rows"), or the
        // layout changed: a word for what happened, no values.
        auto ends_with_id = [&](const std::string& id) {
            return !id.empty() && line.key.size() > id.size() + 1 &&
                   line.key.compare(line.key.size() - id.size() - 1, std::string::npos, " " + id) == 0;
        };
        if ((line.kind == WidgetDiffLine::Kind::added && ends_with_id(line.after)) ||
            (line.kind == WidgetDiffLine::Kind::removed && ends_with_id(line.before)) ||
            (line.kind == WidgetDiffLine::Kind::changed && line.before.empty() && line.after.empty())) {
            const char* word = line.kind == WidgetDiffLine::Kind::added   ? "added"
                               : line.kind == WidgetDiffLine::Kind::removed ? "removed"
                                                                            : "changed";
            card_draw_text(dl, L.mono, L.mono_size, ImVec2(x, ty),
                           line.kind == WidgetDiffLine::Kind::added ? L.text : L.faint, word);
            continue;
        }
        if (line.kind != WidgetDiffLine::Kind::added && !line.before.empty()) {
            const std::string b = card_fit_text(L.mono, L.mono_size, line.before, (d.Max.x - x) * 0.45f);
            const float bw = card_text_extent(L.mono, L.mono_size, b).x;
            card_draw_text(dl, L.mono, L.mono_size, ImVec2(x, ty), L.faint, b);
            dl->AddRectFilled(ImVec2(x, std::round(ty + mono_h * 0.55f)),
                              ImVec2(x + bw, std::round(ty + mono_h * 0.55f) + 1.0f), L.faint);
            x += bw + std::round(10.0f * s);
        }
        if (line.kind == WidgetDiffLine::Kind::changed && line.before.empty() && line.after.empty()) continue;
        card_draw_text(dl, L.mono, L.mono_size, ImVec2(x, ty), L.faint, arrow);
        x += card_text_extent(L.mono, L.mono_size, arrow).x + std::round(10.0f * s);
        if (line.kind == WidgetDiffLine::Kind::removed)
            card_draw_text(dl, L.mono, L.mono_size, ImVec2(x, ty), L.faint, "unset");
        else
            card_draw_text(dl, L.mono, L.mono_size, ImVec2(x, ty), L.text,
                           card_fit_text(L.mono, L.mono_size, line.after, d.Max.x - x));
    }
    if (folded) {
        const int rest = (int)lines.size() - shown;
        const float y = d.Min.y + lh * (float)shown;
        card_draw_text(dl, L.mono, L.mono_size, ImVec2(d.Min.x, y + (lh - mono_h) * 0.5f), L.faint,
                       "+" + std::to_string(rest));
    }
}

void draw_unified_diff(const SpecLook& L, const std::string& diff, float lh) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float mono_h = card_text_extent(L.mono, L.mono_size, "Ag").y;
    const float x = ImGui::GetCursorScreenPos().x;
    size_t start = 0;
    while (start < diff.size()) {
        const size_t end = diff.find('\n', start);
        const std::string line = diff.substr(start, end == std::string::npos ? std::string::npos : end - start);
        start = end == std::string::npos ? diff.size() : end + 1;
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const bool added = !line.empty() && line[0] == '+' && line.rfind("+++", 0) != 0;
        const bool removed = !line.empty() && line[0] == '-' && line.rfind("---", 0) != 0;
        const ImU32 c = added ? L.text : removed ? L.faint : sternbild::with_alpha(L.faint, 150);
        card_draw_text(dl, L.mono, L.mono_size, ImVec2(x, p.y + (lh - mono_h) * 0.5f), c, line);
        if (removed) {
            const float lw = card_text_extent(L.mono, L.mono_size, line).x;
            dl->AddRectFilled(ImVec2(x + card_text_extent(L.mono, L.mono_size, "-").x, std::round(p.y + lh * 0.5f)),
                              ImVec2(x + lw, std::round(p.y + lh * 0.5f) + 1.0f), L.faint);
        }
        ImGui::Dummy(ImVec2(1.0f, lh));
    }
}

}  // namespace caliper_host
