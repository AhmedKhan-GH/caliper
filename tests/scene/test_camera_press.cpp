// The camera press of a scene's picture (src/host/live/camera_drag.h), headless and with no GPU,
// over a view that behaves as the scene view does: an orbit or a pan moves the camera and marks it
// moved, a frame frames the view again while it is not moved, and camera.set moves it.
#include "live/camera_drag.h"

#include <doctest/doctest.h>

#include <cmath>
#include <vector>

using caliper_live_views::camera_press_begin;
using caliper_live_views::camera_press_change;
using caliper_live_views::camera_press_frame;
using caliper_live_views::CameraPress;

namespace {

// A camera of the fake view: an orbit's azimuth, its target along x and its distance.
struct FakeState {
    double azimuth = 0.5, target = 0.0, distance = 3.0;
    bool moved = false;
    bool operator==(const FakeState& o) const {
        return azimuth == o.azimuth && target == o.target && distance == o.distance && moved == o.moved;
    }
    bool operator!=(const FakeState& o) const { return !(*this == o); }
};

struct FakeView {
    using State = FakeState;
    FakeState now;
    bool pan = false;
    static bool same(const State& a, const State& b) { return a == b; }
    State state() const { return now; }
    void restore(const State& s) { now = s; }
    void move(float dx, float dy) {   // 0.01 rad or 0.01 units a point, as the scene view's orbit
        (void)dy;
        if (pan) now.target -= 0.01 * double(dx);
        else now.azimuth += 0.01 * double(dx);
        now.moved = true;
    }
    // What the scene view does without the press: a frame that frames new data while not moved, a
    // fit at the angles it has, a zoom, camera.set.
    void reframe(double target) {
        if (!now.moved) now.target = target;
    }
    void fit() { now.target = 0.0; }
    void zoom(double k) {
        now.distance *= k;
        now.moved = true;
    }
    void set(double azimuth) {
        now.azimuth = azimuth;
        now.moved = true;
    }
};

// One frame of the press with the pointer at (x, y).
bool frame(CameraPress<FakeState>& p, FakeView& v, float x, float y, bool released = false) {
    return camera_press_frame(p, v, true, x, y, released);
}

}  // namespace

TEST_CASE("camera press: a drag of a point a frame moves the camera on its first frame of motion, by every point") {
    for (const bool pan : {false, true}) {
        CAPTURE(pan);
        FakeView v;
        v.pan = pan;
        CameraPress<FakeState> p;
        camera_press_begin(p, v, 100.0f, 50.0f);
        frame(p, v, 100.0f, 50.0f);
        const FakeState press = v.now;
        std::vector<FakeState> after;
        for (int k = 1; k <= 30; ++k) {
            frame(p, v, 100.0f + float(k), 50.0f);
            after.push_back(v.now);
        }
        CHECK(after[0] != press);   // the first point moved the camera
        for (std::size_t k = 1; k < after.size(); ++k) CHECK(after[k] != after[k - 1]);
        CHECK_FALSE(frame(p, v, 130.0f, 50.0f, true));   // a drag
        CHECK(v.now == after.back());
        if (pan) CHECK(v.now.target == doctest::Approx(press.target - 0.30).epsilon(1e-12));
        else CHECK(v.now.azimuth == doctest::Approx(press.azimuth + 0.30).epsilon(1e-12));
        CHECK(v.now.moved);
    }
}

TEST_CASE("camera press: a click puts the camera and moved() back exactly, after a wobble or none") {
    for (const float wobble : {0.0f, 1.0f, 2.0f}) {
        CAPTURE(wobble);
        FakeView v;
        CameraPress<FakeState> p;
        const FakeState before = v.now;
        camera_press_begin(p, v, 10.0f, 10.0f);
        frame(p, v, 10.0f, 10.0f);
        frame(p, v, 10.0f + wobble, 10.0f);
        CHECK((wobble == 0.0f) == (v.now == before));
        CHECK(frame(p, v, 10.0f + wobble, 10.0f, true));   // a click
        CHECK(v.now == before);
        CHECK_FALSE(v.now.moved);
    }
}

TEST_CASE("camera press: a drag out to 30 points and back within one is a drag") {
    FakeView v;
    CameraPress<FakeState> p;
    const FakeState before = v.now;
    camera_press_begin(p, v, 0.0f, 0.0f);
    for (int k = 1; k <= 30; ++k) frame(p, v, float(k), 0.0f);
    for (int k = 29; k >= 1; --k) frame(p, v, float(k), 0.0f);
    CHECK_FALSE(frame(p, v, 1.0f, 0.0f, true));
    CHECK(v.now.azimuth == doctest::Approx(before.azimuth + 0.01).epsilon(1e-12));
    CHECK(v.now.moved);
}

TEST_CASE("camera press: a resting finger's sub-point jitter leaves no drift over many clicks") {
    FakeView v;
    const FakeState before = v.now;
    const float jitter[][2] = {{0.25f, 0.0f}, {0.5f, -0.25f}, {0.75f, 0.25f}, {-0.5f, 0.5f}, {1.0f, 1.25f}};
    for (int n = 0; n < 20; ++n) {
        CameraPress<FakeState> p;
        camera_press_begin(p, v, 40.0f, 40.0f);
        bool wobbled = false;
        for (int k = 0; k < 15; ++k) {
            frame(p, v, 40.0f + jitter[k % 5][0], 40.0f + jitter[k % 5][1]);
            wobbled = wobbled || v.now != before;
        }
        CHECK(wobbled);
        CHECK(frame(p, v, 40.25f, 40.0f, true));
        CHECK(v.now == before);
    }
}

TEST_CASE("camera press: a change from outside during a click stays, the wobble goes, and moved() is as before") {
    SUBCASE("a reframe by new data before the first motion, and the view frames new data after the click") {
        FakeView v;
        CameraPress<FakeState> p;
        camera_press_begin(p, v, 0.0f, 0.0f);
        frame(p, v, 0.0f, 0.0f);
        v.reframe(5.0);   // the frame after the press frames new bounds
        frame(p, v, 1.0f, 0.0f);
        v.reframe(6.0);   // moved by the wobble: no frame frames until the release
        CHECK(frame(p, v, 1.0f, 0.0f, true));
        CHECK(v.now.target == 5.0);
        CHECK(v.now.azimuth == FakeState{}.azimuth);
        CHECK_FALSE(v.now.moved);
        v.reframe(6.0);   // the view still follows its data
        CHECK(v.now.target == 6.0);
    }
    SUBCASE("camera.set after the wobble") {
        FakeView v;
        CameraPress<FakeState> p;
        camera_press_begin(p, v, 0.0f, 0.0f);
        frame(p, v, 1.0f, 0.0f);
        v.set(2.0);
        frame(p, v, 1.5f, 0.0f);
        CHECK(frame(p, v, 1.5f, 0.0f, true));
        CHECK(v.now.azimuth == 2.0);   // the wobble after it goes
        CHECK(v.now.moved);   // as camera.set leaves it
    }
    SUBCASE("camera.set after the wobble, with no motion after it") {
        FakeView v;
        CameraPress<FakeState> p;
        camera_press_begin(p, v, 0.0f, 0.0f);
        frame(p, v, 1.0f, 0.0f);
        v.set(2.0);
        CHECK(frame(p, v, 1.0f, 0.0f, true));
        CHECK(v.now.azimuth == 2.0);
        CHECK(v.now.moved);
    }
    SUBCASE("the view's own fit and zoom during the click apply to the camera of the press") {
        FakeView v;
        v.now.target = 3.0;
        CameraPress<FakeState> p;
        camera_press_begin(p, v, 0.0f, 0.0f);
        frame(p, v, 1.0f, 0.0f);
        camera_press_change(p, v, [&] { v.fit(); });
        CHECK(v.now.azimuth == doctest::Approx(FakeState{}.azimuth + 0.01).epsilon(1e-12));   // the wobble stays on top
        frame(p, v, 2.0f, 0.0f);
        camera_press_change(p, v, [&] { v.zoom(0.5); });
        CHECK(frame(p, v, 2.0f, 0.0f, true));
        FakeState want;
        want.target = 0.0;
        want.distance = 1.5;
        want.moved = true;   // as the zoom leaves it
        CHECK(v.now == want);
    }
    SUBCASE("a fit during a drag stays with the drag") {
        FakeView v;
        CameraPress<FakeState> p;
        camera_press_begin(p, v, 0.0f, 0.0f);
        for (int k = 1; k <= 10; ++k) frame(p, v, float(k), 0.0f);
        v.now.target = 3.0;
        camera_press_change(p, v, [&] { v.fit(); });
        CHECK_FALSE(frame(p, v, 10.0f, 0.0f, true));
        CHECK(v.now.target == 0.0);
        CHECK(v.now.azimuth == doctest::Approx(FakeState{}.azimuth + 0.10).epsilon(1e-12));
    }
}

TEST_CASE("camera press: a press that keeps its view, as during the 3-D box's own animation, puts nothing back") {
    FakeView v;
    CameraPress<FakeState> p;
    const FakeState before = v.now;
    camera_press_begin(p, v, 0.0f, 0.0f);
    p.keep = true;
    frame(p, v, 1.0f, 0.0f);
    CHECK(frame(p, v, 1.0f, 0.0f, true));   // a click, which keeps what it did
    CHECK(v.now != before);
}

TEST_CASE("camera press: the click tolerance is 2 points from the press, at every frame of the press") {
    CHECK(caliper_live_views::camera_press_is_click(0.0f));
    CHECK(caliper_live_views::camera_press_is_click(4.0f));
    CHECK_FALSE(caliper_live_views::camera_press_is_click(4.01f));
    FakeView v;
    CameraPress<FakeState> p;
    camera_press_begin(p, v, 0.0f, 0.0f);
    frame(p, v, 1.0f, 1.0f);
    frame(p, v, 2.0f, 0.0f);
    CHECK(p.click());
    frame(p, v, 2.0f, 0.5f);   // past 2 points once
    frame(p, v, 0.0f, 0.0f);
    CHECK_FALSE(p.click());
    CHECK_FALSE(frame(p, v, 0.0f, 0.0f, true));
}
