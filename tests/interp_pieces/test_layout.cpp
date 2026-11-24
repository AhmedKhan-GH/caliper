#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "spec_runtime/label_layout.h"

using namespace caliper_host;

TEST_CASE("pinned token slots fit eight rows without adjacent text overlap") {
    for (const float width : {512.0f, 1440.0f}) {
        const LabelRect panel{0, 0, width, 320};
        std::vector<LabelBox> boxes;
        for (int row = 0; row < 8; ++row)
            for (int col = 0; col < 25; ++col)
                boxes.push_back({width * (col + 0.5f) / 25, 20.0f + row * 40, 300, 18, 0, true});
        const auto slots = pinned_label_slots(boxes, panel);
        REQUIRE(slots.size() == 200);
        for (std::size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            CHECK(slot.x0 >= panel.x0);
            CHECK(slot.x1 <= panel.x1);
            CHECK(slot.x1 > slot.x0);
            const float available = slot.x1 - slot.x0 - 6;
            const auto text = fit_with_ellipsis("A Long Token Label", available,
                [](std::string_view s) { return float(s.size()) * 4; });
            CHECK_FALSE(text.empty());
            CHECK(text.size() * 4 <= available);
            if (i % 25) CHECK(slots[i - 1].x1 <= slot.x0 + 0.001f);
        }
    }
}

TEST_CASE("pinned slots separate close rows and retain the first coincident anchor") {
    const LabelRect panel{0, 0, 200, 100};
    const std::vector<LabelBox> boxes = {
        {40, 25, 150, 20, 0, true}, {100, 30, 150, 20, 0, true},
        {40, 60, 150, 20, 0, true}, {100, 60, 150, 20, 0, true},
        {100, 60, 150, 20, 0, true}};
    const auto slots = pinned_label_slots(boxes, panel);
    CHECK(slots[0].x1 == 70);
    CHECK(slots[1].x0 == 70);
    CHECK(slots[2].x1 == 70);
    CHECK(slots[3].x0 == 70);
    CHECK(slots[3].x1 == panel.x1);
    CHECK(slots[4].x0 == slots[4].x1);
}
