// The inline surface's layout as plain rectangles, without ImGui: at a
// comfortable panel and at a small one, at two text scales, every part
// inside the panel and the band, no two parts on each other, and what
// collapses first as the panel shrinks.
#include "builder/widget_surface.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

using namespace caliper_host;

namespace {

// Fixed advances: 7 px a character in the UI face and in mono, 8 in tags.
WidgetMeasure measure(float scale = 1.0f) {
    return [scale](const std::string& s, WidgetFont f) {
        size_t n = 0;
        for (unsigned char c : s) n += (c & 0xC0) != 0x80;   // code points
        return (float)n * (f == WidgetFont::tag ? 8.0f : 7.0f) * scale;
    };
}

WidgetLayoutInput preview_input(WidgetRect panel, int diff_lines = 1) {
    WidgetLayoutInput in;
    in.panel = panel;
    in.strip = 24;
    in.state = WidgetState::preview;
    in.status = "Claude Haiku 5.5 \xC2\xB7 low";
    in.reply = true;
    in.diff_lines = diff_lines;
    in.card_actions = {WidgetAction::accept, WidgetAction::reject, WidgetAction::refine, WidgetAction::diff};
    in.row_actions = {WidgetAction::ask_in_chat};
    return in;
}

void check_inside(const WidgetLayout& L, const WidgetLayoutInput& in) {
    const WidgetRect band{in.panel.x, in.panel.y + in.strip, in.panel.w, L.band};
    CHECK(L.band <= in.panel.h - in.strip);
    std::vector<WidgetRect> parts = {L.prompt, L.mark, L.input, L.status, L.card, L.tag, L.reply, L.diff, L.actions};
    for (const auto& b : L.row_buttons) parts.push_back(b);
    for (const auto& b : L.buttons) parts.push_back(b);
    for (const auto& r : parts) {
        if (r.empty()) continue;
        INFO(r.x << "," << r.y << " " << r.w << "x" << r.h);
        CHECK(in.panel.contains(r));
        CHECK(band.contains(r));
    }
    // The row's parts sit side by side, and the buttons never touch.
    std::vector<WidgetRect> row = {L.mark, L.input, L.status};
    for (const auto& b : L.row_buttons) row.push_back(b);
    for (size_t i = 0; i < row.size(); ++i)
        for (size_t j = i + 1; j < row.size(); ++j) CHECK_FALSE(row[i].overlaps(row[j]));
    for (size_t i = 0; i < L.buttons.size(); ++i)
        for (size_t j = i + 1; j < L.buttons.size(); ++j) CHECK_FALSE(L.buttons[i].overlaps(L.buttons[j]));
    // The card's rows stack under the prompt row.
    if (!L.card.empty()) CHECK(L.card.y >= L.prompt.bottom());
    if (!L.reply.empty()) CHECK(L.reply.y >= L.card.y);
    if (!L.diff.empty() && !L.reply.empty()) CHECK(L.diff.y >= L.reply.bottom());
    if (!L.actions.empty() && !L.diff.empty()) CHECK(L.actions.y >= L.diff.bottom());
}

}  // namespace

TEST_CASE("widget layout: a comfortable panel holds the row, the reply, the diff and the actions") {
    const auto m = WidgetLayoutMetrics::for_text_scale(1.0f);
    // The dashboard's curves panel at 1440 x 900.
    const WidgetLayoutInput in = preview_input({0, 0, 720, 584}, 3);
    const WidgetLayout L = layout_widget_surface(in, m, measure());
    check_inside(L, in);
    CHECK_FALSE(L.content_hidden);
    CHECK(L.keys);
    CHECK_FALSE(L.status.empty());
    CHECK_FALSE(L.reply.empty());
    CHECK(L.diff_shown == 3);
    CHECK_FALSE(L.diff_folded);
    REQUIRE(L.buttons.size() == 4);
    REQUIRE(L.row_buttons.size() == 1);
    CHECK(L.row_buttons[0].right() <= L.prompt.right());
    // Diff sits at the action row's right end, the other three from the left.
    CHECK(L.buttons[3].right() == doctest::Approx(L.actions.right()));
    CHECK(L.buttons[0].x == doctest::Approx(L.actions.x));
    // The band leaves the content most of the panel.
    CHECK(L.band < 0.3f * in.panel.h);
    CHECK(in.panel.h - in.strip - L.band >= m.min_content);
    // The prompt alone takes one row.
    WidgetLayoutInput prompt = in;
    prompt.state = WidgetState::prompt;
    prompt.reply = false;
    prompt.diff_lines = 0;
    prompt.card_actions.clear();
    const WidgetLayout P = layout_widget_surface(prompt, m, measure());
    CHECK(P.card.empty());
    CHECK(P.band == doctest::Approx(m.row + 2 * m.gap));
    check_inside(P, prompt);
    // A long diff folds into "+N" past the compact lines.
    const WidgetLayout F = layout_widget_surface(preview_input({0, 0, 720, 584}, 20), m, measure());
    CHECK(F.diff_shown == m.max_diff_lines);
    CHECK(F.diff_folded);
}

TEST_CASE("widget layout: a small panel collapses the reply, then the diff, then the status and the keys") {
    const auto m = WidgetLayoutMetrics::for_text_scale(1.0f);
    // Short: the reply goes first, then the diff lines past the first.
    WidgetLayoutInput in = preview_input({100, 50, 260, 200}, 4);
    WidgetLayout L = layout_widget_surface(in, m, measure());
    check_inside(L, in);
    CHECK(L.reply.empty());
    CHECK(L.diff_shown == 1);
    CHECK(L.diff_folded);
    CHECK_FALSE(L.content_hidden);
    CHECK(in.panel.h - in.strip - L.band >= m.min_content);
    // Narrow: the model label goes, then the keys of the buttons.
    CHECK(L.status.empty());
    CHECK_FALSE(L.keys);
    CHECK(L.input.w >= m.min_input);
    REQUIRE(L.buttons.size() == 4);
    for (const auto& b : L.buttons) CHECK(b.right() <= L.actions.right() + 0.5f);

    // At the minimum width everything still has a place.
    in = preview_input({0, 0, m.min_width, 240}, 1);
    L = layout_widget_surface(in, m, measure());
    check_inside(L, in);
    CHECK_FALSE(L.content_hidden);
    CHECK(L.buttons.size() == 4);

    // Below the minimum height the band takes the panel and the content goes.
    in = preview_input({0, 0, 300, 120}, 2);
    L = layout_widget_surface(in, m, measure());
    CHECK(L.content_hidden);
    CHECK(L.band == doctest::Approx(in.panel.h - in.strip));
    check_inside(L, in);
}

TEST_CASE("widget layout: the rows grow with the text scale") {
    const auto m1 = WidgetLayoutMetrics::for_text_scale(1.0f);
    const auto m2 = WidgetLayoutMetrics::for_text_scale(1.5f);
    CHECK(m2.row > m1.row);
    CHECK(m2.min_content > m1.min_content);
    const WidgetLayoutInput in = preview_input({0, 0, 1280, 935}, 2);
    const WidgetLayout a = layout_widget_surface(in, m1, measure(1.0f));
    const WidgetLayout b = layout_widget_surface(in, m2, measure(1.5f));
    check_inside(a, in);
    check_inside(b, in);
    CHECK(b.band > a.band);
    CHECK(b.prompt.h == doctest::Approx(m2.row));
}
