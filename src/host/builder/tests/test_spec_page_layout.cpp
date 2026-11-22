// The spec page's rectangles from one function (spec_page_layout.h), at
// 2560x1440 and 1440x900 points, for every combination of build mode, the
// spec text strip and the side chat: each region inside the work area the
// host's strips leave, on whole pixels, none overlapping another; the canvas
// the largest; the chat's column as wide in build mode as outside it, with
// Properties above it; and outside build mode the chat's own column.
#include "builder/side_chat.h"
#include "builder/spec_page_layout.h"

#include <doctest/doctest.h>

#include <cmath>
#include <vector>

using namespace caliper_host;

namespace {

bool inside(const SpecRect& outer, const SpecRect& r) {
    return r.x >= outer.x && r.y >= outer.y && r.right() <= outer.right() && r.bottom() <= outer.bottom();
}

bool apart(const SpecRect& a, const SpecRect& b) {
    return a.right() <= b.x || b.right() <= a.x || a.bottom() <= b.y || b.bottom() <= a.y;
}

bool whole(const SpecRect& r) {
    return r.x == std::round(r.x) && r.y == std::round(r.y) && r.w == std::round(r.w) && r.h == std::round(r.h);
}

double area_of(const SpecRect& r) { return (double)r.w * r.h; }

}  // namespace

TEST_CASE("spec page layout: every combination of Build, Text and Chat at two sizes") {
    for (const auto& size : {std::pair<float, float>{2560, 1440}, {1440, 900}}) {
        // The page bar (28) above, the status bar (22) and an approval
        // strip (96) below: the work area they leave.
        const SpecRect area{0.0f, 28.0f, size.first, size.second - 28.0f - 22.0f - 96.0f};
        for (int mask = 0; mask < 8; ++mask) {
            SpecPageLayoutInput in;
            in.area = area;
            in.build = mask & 1;
            in.text_open = mask & 2;
            in.chat = mask & 4;
            CAPTURE(size.first);
            CAPTURE(in.build);
            CAPTURE(in.text_open);
            CAPTURE(in.chat);
            const BuilderPaneLayout l = layout_spec_page(in);
            std::vector<SpecRect> regions = {l.canvas};
            if (l.left) regions.insert(regions.end(), {l.sources, l.primitives});
            if (l.right) regions.push_back(l.properties);
            if (l.strip) regions.push_back(l.strip_rect);
            if (l.chat) regions.push_back(l.chat_rect);
            for (const auto& r : regions) {
                CHECK(inside(area, r));
                CHECK(whole(r));
                CHECK(r.w > 0.0f);
                CHECK(r.h > 0.0f);
            }
            for (size_t i = 0; i < regions.size(); ++i)
                for (size_t j = i + 1; j < regions.size(); ++j) CHECK(apart(regions[i], regions[j]));
            for (const auto& r : regions) CHECK(area_of(l.canvas) >= area_of(r));
            CHECK(l.chat == in.chat);

            // The chat's column outside build mode, which ChatPane::draw_column() takes.
            ChatColumnInput ci;
            ci.window = {area.x, area.y, area.w, area.h};
            const ChatColumnLayout c = layout_chat_column(ci, ChatLayoutMetrics::for_text_scale(1.0f));
            if (!in.build) {
                CHECK_FALSE(l.left);
                CHECK_FALSE(l.right);
                CHECK_FALSE(l.strip);   // the text strip is build mode's
                if (in.chat) {
                    CHECK(l.chat_rect.x == c.column.x);
                    CHECK(l.chat_rect.w == c.column.w);
                    CHECK(l.canvas.w + l.chat_rect.w == area.w);
                } else {
                    CHECK(l.canvas.w == area.w);
                    CHECK(l.canvas.h == area.h);
                }
                continue;
            }
            REQUIRE(l.left);
            REQUIRE(l.right);
            CHECK(l.strip == in.text_open);
            if (in.text_open) {
                CHECK(l.strip_rect.y == l.canvas.bottom() + 1.0f);
                CHECK(l.strip_rect.w == l.canvas.w);
            }
            if (in.chat) {
                // The chat keeps its width, under Properties in the right column.
                CHECK(l.right_column.w == c.column.w);
                CHECK(l.right_column.right() == area.right());
                CHECK(l.chat_rect.x == l.properties.x);
                CHECK(l.chat_rect.w == l.properties.w);
                CHECK(l.chat_rect.y == l.properties.bottom() + 1.0f);
                CHECK(l.chat_rect.bottom() == area.bottom());
                CHECK(l.chat_rect.h >= ChatLayoutMetrics::for_text_scale(1.0f).min_height);
            } else {
                CHECK(l.properties.h == area.h);
            }
        }
    }
}

TEST_CASE("spec page layout: the builder pane's rule is the page's") {
    BuilderPaneLayoutInput in;
    in.area = {0, 28, 1440, 850};
    in.build = true;
    in.chat = true;
    SpecPageLayoutInput p;
    p.area = in.area;
    p.build = true;
    p.chat = true;
    const BuilderPaneLayout a = spec_page_layout_rule(in, 1.0f), b = layout_spec_page(p);
    CHECK(a.canvas.x == b.canvas.x);
    CHECK(a.canvas.w == b.canvas.w);
    CHECK(a.chat_rect.y == b.chat_rect.y);
}
