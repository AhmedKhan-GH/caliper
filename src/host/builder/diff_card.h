#pragma once
// ============================================================================
// diff_card.h: the compact diff of a change to a spec and its drawing in the
// house style, shared by the widget agent's inline surface (widget_surface.h,
// design/widget-agent.md, The inline prompt and the preview) and the approval
// cards (approval_cards.h, design/acp-integration.md, Approvals and diffs for
// every agent), so that a change reads the same wherever the person decides
// on it.
//
// The compact diff has one line for each changed key: the key, the old value
// struck through in the faint tone, an arrow and the new value in parchment,
// since colour belongs to data. Past the lines that fit, the rest fold into
// "+N". The full diff is the unified diff of the canonical TOML, drawn line
// by line with added lines in parchment and removed lines faint and struck
// through.
//
// Built only with CALIPER_WITH_BUILDER. The drawing runs on the frame thread
// inside an ImGui frame; widget_compact_diff() needs no window.
// ============================================================================
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <imgui.h>
#include <imgui_internal.h>   // ImRect

#include <cstddef>
#include <string>
#include <vector>

namespace caliper_host {

struct SpecLook;   // spec_runtime/spec_runtime.h

// One changed key between two versions of a spec: a key of the target panel
// relative to it (options without their "options." prefix: y_log, title,
// bind.y), or of another item ("panel rows", "source metrics.catalog",
// "layout").
struct WidgetDiffLine {
    enum class Kind { changed, added, removed };
    Kind kind = Kind::changed;
    std::string key;
    std::string before;   // the old value, compact; empty when added
    std::string after;    // the new value, compact; empty when removed
};

// The changed keys of `panel` and of the rest of the spec, the panel's first.
// Values are compact JSON, cut at `max_value` characters. With `panel` empty
// or in neither spec, each changed key carries its item's name
// ("panel curves.options.y_log").
std::vector<WidgetDiffLine> widget_compact_diff(const caliper_builder::spec::Spec& before,
                                                const caliper_builder::spec::Spec& after, const std::string& panel,
                                                size_t max_value = 40);

// The keys of one new item, such as the panel of panel.create, as added
// lines: every key of its canonical JSON but its id, objects flattened to
// dotted keys (primitive, title, bind.rows, options.x_label).
std::vector<WidgetDiffLine> widget_item_lines(const nlohmann::json& item, size_t max_value = 40);

// Text in a card's faces: its extent, the text cut with an ellipsis to fit
// `width` (empty when not even the ellipsis fits), and the text drawn on whole
// pixels. A null font is ImGui's current one.
ImVec2 card_text_extent(ImFont* font, float size, const std::string& text);
std::string card_fit_text(ImFont* font, float size, const std::string& text, float width);
void card_draw_text(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, ImU32 colour, const std::string& text);

// The compact diff in `area`: the first `shown` lines, each area.height /
// (shown + folded) tall, then "+N" for the rest when `folded`.
void draw_compact_diff(ImDrawList* dl, const SpecLook& look, const ImRect& area,
                       const std::vector<WidgetDiffLine>& lines, int shown, bool folded);

// The unified diff `diff`, one line of `line_height` each, at the cursor of
// the current window, which is a child that scrolls past what fits; each
// line advances the cursor.
void draw_unified_diff(const SpecLook& look, const std::string& diff, float line_height);

}  // namespace caliper_host
