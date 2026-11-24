// The tests of label placement: no overlap, the priority order and the same placement for the same input
// (design/instrument-interface.md, Label Placement).
#include <doctest/doctest.h>

#include "spec_runtime/label_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using namespace caliper_host;

namespace {

LabelRect rect_of(const PlacedLabel& p, const LabelBox& b) { return {p.x, p.y, p.x + b.width, p.y + b.height}; }

bool meet(const LabelRect& a, const LabelRect& b) { return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1; }

// No two shown labels overlap, none overlaps a marker, and each lies in the panel.
void check_clear(const std::vector<LabelBox>& boxes, const std::vector<PlacedLabel>& placed, const LabelRect& panel,
                 const std::vector<LabelRect>& markers = {}) {
    REQUIRE(placed.size() == boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        if (!placed[i].shown) continue;
        const LabelRect a = rect_of(placed[i], boxes[i]);
        CHECK(a.x0 >= panel.x0);
        CHECK(a.y0 >= panel.y0);
        CHECK(a.x1 <= panel.x1);
        CHECK(a.y1 <= panel.y1);
        for (const LabelRect& m : markers) CHECK_FALSE(meet(a, m));
        for (std::size_t j = i + 1; j < boxes.size(); ++j)
            if (placed[j].shown) CHECK_FALSE(meet(a, rect_of(placed[j], boxes[j])));
    }
}

// A small fixed generator, so that the cases are the same on every machine.
struct Lcg {
    std::uint32_t s;
    float next(float lo, float hi) {
        s = s * 1664525u + 1013904223u;
        return lo + (hi - lo) * (float)(s >> 8) / (float)(1u << 24);
    }
};

}  // namespace

TEST_CASE("three labels at one anchor no longer overlap") {
    // The case that drew three labels on top of each other: three items
    // whose anchors fall within two pixels.
    const LabelRect panel{0, 0, 400, 300};
    const std::vector<LabelBox> boxes = {{200, 150, 60, 15, 0}, {201, 151, 70, 15, 1}, {202, 150, 50, 15, 2}};
    const std::vector<PlacedLabel> placed = place_labels(boxes, panel, {});
    for (const PlacedLabel& p : placed) CHECK(p.shown);
    check_clear(boxes, placed, panel);
}

TEST_CASE("the first free position is right above the anchor, then the others in order") {
    const LabelRect panel{0, 0, 400, 300};
    const std::vector<PlacedLabel> one = place_labels({{100, 100, 40, 10, 0}}, panel, {});
    REQUIRE(one.size() == 1);
    CHECK(one[0].shown);
    CHECK_FALSE(one[0].leader);
    CHECK(one[0].x > 100);              // to the right
    CHECK(one[0].y + 10 < 100);         // above
    // Against the right edge of the panel, the label goes to the left.
    const std::vector<PlacedLabel> edge = place_labels({{395, 100, 40, 10, 0}}, panel, {});
    CHECK(edge[0].shown);
    CHECK(edge[0].x + 40 <= 395);
}

TEST_CASE("a crowded anchor gets a leader line, then only its anchor") {
    const LabelRect panel{0, 0, 400, 300};
    std::vector<LabelBox> boxes;
    for (int i = 0; i < 40; ++i) boxes.push_back({200, 150, 30, 10, i});
    const std::vector<PlacedLabel> placed = place_labels(boxes, panel, {});
    check_clear(boxes, placed, panel);
    int near = 0, leader = 0, hidden = 0;
    for (const PlacedLabel& p : placed) {
        if (!p.shown) ++hidden;
        else if (p.leader) ++leader;
        else ++near;
    }
    CHECK(near >= 2);
    CHECK(leader >= 2);
    CHECK(hidden > 0);
    CHECK(near + leader + hidden == 40);
    // The near labels are the first in priority, the leaders come after them.
    CHECK_FALSE(placed[0].leader);
    CHECK(placed[0].shown);
}

TEST_CASE("priority decides who keeps the near position, not the input order") {
    const LabelRect panel{0, 0, 400, 300};
    // Two labels that want the same spot; the second has the higher priority.
    const std::vector<LabelBox> boxes = {{100, 100, 40, 10, 5}, {100, 100, 40, 10, 1}};
    const std::vector<PlacedLabel> placed = place_labels(boxes, panel, {});
    const std::vector<PlacedLabel> alone = place_labels({boxes[1]}, panel, {});
    CHECK(placed[1].x == alone[0].x);
    CHECK(placed[1].y == alone[0].y);
    CHECK((placed[0].x != placed[1].x || placed[0].y != placed[1].y));
    // Equal priorities keep the input order.
    const std::vector<LabelBox> equal = {{100, 100, 40, 10, 0}, {100, 100, 40, 10, 0}};
    const std::vector<PlacedLabel> eq = place_labels(equal, panel, {});
    CHECK(eq[0].x == alone[0].x);
    CHECK(eq[0].y == alone[0].y);
}

TEST_CASE("markers and pinned labels are obstacles") {
    const LabelRect panel{0, 0, 400, 300};
    const LabelRect ring{95, 75, 160, 100};   // covers the space right above the anchor
    const std::vector<LabelBox> boxes = {{100, 100, 40, 10, 0}};
    const std::vector<PlacedLabel> placed = place_labels(boxes, panel, {ring});
    CHECK(placed[0].shown);
    check_clear(boxes, placed, panel, {ring});

    // A pinned label stays at its anchor even over a marker, and the next
    // label goes around it.
    const std::vector<LabelBox> pinned = {{110, 80, 60, 12, 0, true}, {100, 100, 40, 10, 1}};
    const std::vector<PlacedLabel> pp = place_labels(pinned, panel, {ring});
    CHECK(pp[0].shown);
    CHECK(pp[0].x == 110);
    CHECK(pp[0].y == 80);
    CHECK(pp[1].shown);
    CHECK_FALSE(meet(rect_of(pp[0], pinned[0]), rect_of(pp[1], pinned[1])));
}

TEST_CASE("a label larger than the panel, a label off the panel and bad numbers are not shown") {
    const LabelRect panel{0, 0, 100, 50};
    const std::vector<LabelBox> boxes = {{50, 25, 200, 10, 0}, {50, 25, 10, 10, 0}, {NAN, 10, 10, 10, 0},
                                         {500, 500, 10, 10, 0}, {10, 10, -1, 10, 0}};
    const std::vector<PlacedLabel> placed = place_labels(boxes, panel, {});
    CHECK_FALSE(placed[0].shown);
    CHECK(placed[1].shown);
    CHECK_FALSE(placed[2].shown);
    CHECK_FALSE(placed[3].shown);
    CHECK_FALSE(placed[4].shown);
    // A label not shown keeps its anchor as its position, for the hover.
    CHECK(placed[0].x == 50);
    CHECK(placed[0].y == 25);
}

TEST_CASE("many random labels never overlap, and the same input gives the same placement") {
    const LabelRect panel{0, 0, 2560, 1440};
    for (std::uint32_t seed : {1u, 7u, 42u}) {
        Lcg g{seed};
        std::vector<LabelBox> boxes;
        for (int i = 0; i < 300; ++i)
            boxes.push_back({g.next(0, 2560), g.next(0, 1440), g.next(20, 160), 15, (int)g.next(0, 4)});
        const std::vector<PlacedLabel> a = place_labels(boxes, panel, {});
        const std::vector<PlacedLabel> b = place_labels(boxes, panel, {});
        check_clear(boxes, a, panel);
        REQUIRE(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i].x == b[i].x);
            CHECK(a[i].y == b[i].y);
            CHECK(a[i].shown == b[i].shown);
            CHECK(a[i].leader == b[i].leader);
        }
        int shown = 0;
        for (const PlacedLabel& p : a) shown += p.shown;
        CHECK(shown > 250);
    }
}

TEST_CASE("a panel places at most 512 labels, the first in priority") {
    const LabelRect panel{0, 0, 100000, 100000};
    std::vector<LabelBox> boxes;
    for (int i = 0; i < 600; ++i) boxes.push_back({(float)(i % 30) * 300 + 50, (float)(i / 30) * 300 + 50, 40, 10, 600 - i});
    const std::vector<PlacedLabel> placed = place_labels(boxes, panel, {});
    int shown = 0;
    for (const PlacedLabel& p : placed) shown += p.shown;
    CHECK(shown == (int)kMaxLabels);
    // The lowest priorities are the last boxes, and those are shown.
    CHECK(placed[599].shown);
    CHECK_FALSE(placed[0].shown);
}

TEST_CASE("pinned slots retain the priority winner at coincident anchors") {
    const LabelRect panel{20, 20, 220, 120};
    std::vector<LabelBox> boxes = {{120, 60, 40, 10, 0, true}, {120, 60, 40, 10, 1, true}};
    std::size_t winner = 0;
    SUBCASE("the first label has priority") {}
    SUBCASE("the second label has priority") {
        boxes[1].priority = -1;
        winner = 1;
    }
    SUBCASE("equal priorities retain input order") { boxes[1].priority = 0; }

    const auto slots = pinned_label_slots(boxes, panel);
    REQUIRE(slots.size() == boxes.size());
    CHECK(slots[winner].x0 == panel.x0);
    CHECK(slots[winner].x1 == panel.x1);
    CHECK(slots[1 - winner].x0 == boxes[0].anchor_x);
    CHECK(slots[1 - winner].x1 == boxes[0].anchor_x);

    // Each alignment keeps text for the winner and no text for the loser.
    const auto bytes = [](std::string_view text) { return (float)text.size(); };
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const float left = boxes[i].anchor_x - slots[i].x0;
        const float right = slots[i].x1 - boxes[i].anchor_x;
        for (float available : {right, left, 2 * std::min(left, right)}) {
            const auto text = fit_with_ellipsis("coincident text", available - 6, bytes);
            if (i == winner) CHECK(text == "coincident text");
            else CHECK(text.empty());
        }
    }
}

TEST_CASE("pinned slots discard only conflicts with retained priority winners") {
    const LabelRect panel{20, 20, 220, 120};
    // The middle row overlaps both others, but the outer rows do not overlap.
    const std::vector<LabelBox> boxes = {
        {120, 48, 40, 10, 1, true}, {120, 56, 40, 10, 2, true}, {120, 40, 40, 10, 0, true}};
    const auto slots = pinned_label_slots(boxes, panel);
    REQUIRE(slots.size() == boxes.size());
    CHECK(slots[0].x0 == slots[0].x1);
    CHECK(slots[1].x0 == panel.x0);
    CHECK(slots[1].x1 == panel.x1);
    CHECK(slots[2].x0 == panel.x0);
    CHECK(slots[2].x1 == panel.x1);
}

TEST_CASE("pinned slots retain separate rows and ignore unpinned anchors") {
    const LabelRect panel{20, 20, 220, 120};
    const std::vector<LabelBox> boxes = {
        {120, 40, 40, 10, 1, true}, {120, 50, 40, 10, 2, true}, {120, 40, 40, 10, 0, false}};
    const auto slots = pinned_label_slots(boxes, panel);
    REQUIRE(slots.size() == boxes.size());
    for (const auto& slot : slots) {
        CHECK(slot.x0 == panel.x0);
        CHECK(slot.y0 == panel.y0);
        CHECK(slot.x1 == panel.x1);
        CHECK(slot.y1 == panel.y1);
    }
}

TEST_CASE("pinned slots preserve midpoint fits around coincident anchors") {
    const LabelRect panel{20, 20, 220, 120};
    const std::vector<LabelBox> boxes = {{60, 60, 100, 10, 3, true}, {120, 60, 100, 10, 2, true},
                                          {180, 60, 100, 10, 1, true}, {120, 60, 100, 10, 0, true}};
    const auto slots = pinned_label_slots(boxes, panel);
    const auto again = pinned_label_slots(boxes, panel);
    REQUIRE(slots.size() == boxes.size());
    REQUIRE(again.size() == slots.size());
    CHECK(slots[0].x0 == 20);
    CHECK(slots[0].x1 == 90);
    CHECK(slots[1].x0 == 120);
    CHECK(slots[1].x1 == 120);
    CHECK(slots[2].x0 == 150);
    CHECK(slots[2].x1 == 220);
    CHECK(slots[3].x0 == 90);
    CHECK(slots[3].x1 == 150);
    const auto bytes = [](std::string_view text) { return (float)text.size(); };
    const std::string full(100, 'x');
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        CHECK(slots[i].x0 == again[i].x0);
        CHECK(slots[i].x1 == again[i].x1);
        CHECK(slots[i].x0 >= panel.x0);
        CHECK(slots[i].x1 <= panel.x1);
        const float available = 2 * std::min(boxes[i].anchor_x - slots[i].x0, slots[i].x1 - boxes[i].anchor_x);
        const auto text = fit_with_ellipsis(full, available - 6, bytes);
        if (i == 1) {
            CHECK(text.empty());
            continue;
        }
        REQUIRE_FALSE(text.empty());
        const float width = bytes(text) + 4;
        const LabelRect rect{boxes[i].anchor_x - width / 2, 55, boxes[i].anchor_x + width / 2, 65};
        CHECK(rect.x0 >= slots[i].x0);
        CHECK(rect.x1 <= slots[i].x1);
        for (std::size_t j = 0; j < boxes.size(); ++j)
            if (j != i && j != 1) CHECK_FALSE(meet(rect, {slots[j].x0, 55, slots[j].x1, 65}));
    }
}

TEST_CASE("a cut text keeps the longest prefix that fits, ends on a whole character and on no space") {
    // One unit of width for each byte; the ellipsis is three bytes.
    const auto bytes = [](std::string_view t) { return (float)t.size(); };
    const std::string ell = kTextEllipsis;
    CHECK(fit_with_ellipsis("short", 5, bytes) == "short");
    CHECK(fit_with_ellipsis("abcdefghij", 8, bytes) == "abcde" + ell);
    CHECK(fit_with_ellipsis("abcd efghij", 8, bytes) == "abcd" + ell);   // no space before the ellipsis
    // "\xC3\xA9" is one character of two bytes, which a cut never splits.
    CHECK(fit_with_ellipsis("ab\xC3\xA9xyzw", 6, bytes) == "ab" + ell);
    CHECK(fit_with_ellipsis("ab\xC3\xA9xyzw", 7, bytes) == "ab\xC3\xA9" + ell);
    CHECK(fit_with_ellipsis("abcdef", 3, bytes) == ell);   // only the ellipsis fits
    CHECK(fit_with_ellipsis("   abcdef", 5, bytes) == ell);
}

TEST_CASE("a cut text refuses a width that is negative or not a number, and a width where no ellipsis fits") {
    const auto bytes = [](std::string_view t) { return (float)t.size(); };
    CHECK(fit_with_ellipsis("abcdef", 2, bytes).empty());
    CHECK(fit_with_ellipsis("abcdef", -1, bytes).empty());
    CHECK(fit_with_ellipsis("abcdef", std::nanf(""), bytes).empty());
    CHECK(fit_with_ellipsis("abcdef", 10, nullptr).empty());
    CHECK(fit_with_ellipsis("", 0, bytes).empty());
}

TEST_CASE("a cut text of 1,024 bytes takes few measures and gives what a search from the end gives") {
    std::string text;
    Lcg g{7};
    for (int i = 0; i < 1024; ++i) text.push_back(g.next(0, 1) < 0.15f ? ' ' : (char)('a' + (int)g.next(0, 26)));
    int measures = 0;
    // A proportional font: a space is narrow and the letters differ.
    const auto width = [&](std::string_view t) {
        ++measures;
        float w = 0;
        for (char c : t) w += c == ' ' ? 3.0f : 5.0f + (float)(c % 4);
        return w;
    };
    for (float limit : {40.0f, 333.0f, 2000.0f, 5000.0f}) {
        measures = 0;
        const std::string got = fit_with_ellipsis(text, limit, width);
        CHECK(measures <= 14);
        // The longest prefix that fits, found from the end as before.
        std::string want;
        for (std::size_t n = text.size(); n > 0 && want.empty(); --n) {
            if (text[n - 1] == ' ') continue;
            const std::string t = text.substr(0, n) + kTextEllipsis;
            if (width(t) <= limit) want = t;
        }
        if (want.empty()) want = kTextEllipsis;
        CHECK(got == want);
        CHECK(width(got) <= limit);
    }
}
