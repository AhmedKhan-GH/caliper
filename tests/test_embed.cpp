// caliper_embed_tests — the embed C ABI (caliper/embed.h) battery (R4 L2a).
//
// Drives the ABI the way an embedder would: create an OFFSCREEN core, load the
// hello fixture applet, pump frames, read the composited pixels back, and shut
// down clean — plus the refusal gates (second live core, frame/event before a
// canvas, double attach) that must fail HONESTLY, never crash.
//
// One CaliperCore per process (v0), so every case pairs create with shutdown.
// The pixel case self-skips when no Metal device is present (offscreen attach
// fails) so a headless CI run stays green, mirroring the gfx suite.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <caliper/embed.h>

// White-box internals for the §7 host-axis byte-compare (last case): the embed
// core wires its renderer into the process bridge at attach, so we drive the
// SAME public tensor_bridge.v1 service and read the texture back through the
// renderer, comparing to the shared CPU reference the gfx rows use.
#include <caliper/tensor.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/metrics_v1.h>   // v1.1 get_service: the Compass case
#include <caliper/services/jobs_v1.h>      // v1.1 get_service: cross-thread producer
#include <caliper/services/log_v1.h>       // v1.1 log routing
#include "host_services.h"
#include "metrics_store.h"          // caliper_host::host_metrics_store() (reader)
#include "tensor_bridge.h"          // caliper_host::expand_u8_to_rgba8 (shared ref)
#include "renderer/host_renderer.h"
#include "app_paths.h"              // caliper::app_data_path (data_dir routing)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <io.h>
#  define caliper_dup    _dup
#  define caliper_dup2   _dup2
#  define caliper_close  _close
#  define caliper_fileno _fileno
#  define setenv(k, v, overwrite) _putenv_s(k, v)
#  define unsetenv(k) _putenv_s(k, "")
#else
#  include <unistd.h>
#  define caliper_dup    dup
#  define caliper_dup2   dup2
#  define caliper_close  close
#  define caliper_fileno fileno
#endif

namespace {

// Capture C-level stderr for a scope, then restore fd 2. The caliper.log.v1
// sink writes applet log lines ("hello.on_init" / "hello.on_cleanup") to stderr
// via fprintf in v0 (embedding.md caveat), so this is how a host-side test
// observes an applet's lifecycle hooks firing through the ABI. Keep the capture
// window tight around the load calls so a failing CHECK's text still reaches the
// real console.
class StderrTap {
public:
    StderrTap() {
        path_ = (std::filesystem::temp_directory_path() /
                 "embed_stderr_tap.txt").string();
        std::fflush(stderr);
        saved_fd_ = caliper_dup(caliper_fileno(stderr));
        std::freopen(path_.c_str(), "w", stderr);
    }
    std::string drain() {
        std::fflush(stderr);
        if (saved_fd_ != -1) {
            caliper_dup2(saved_fd_, caliper_fileno(stderr));
            caliper_close(saved_fd_);
            saved_fd_ = -1;
        }
        std::ifstream in(path_);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    ~StderrTap() { drain(); }
private:
    std::string path_;
    int saved_fd_ = -1;
};

CaliperCoreDesc base_desc() {
    CaliperCoreDesc d{};
    d.struct_size = sizeof(CaliperCoreDesc);
    d.renderer = CALIPER_RENDERER_DEFAULT;
    d.applets_dir = CALIPER_TEST_APPLETS_DIR;   // build/applets: hello lives here
    return d;
}

CaliperCanvasDesc offscreen_desc(int w, int h) {
    CaliperCanvasDesc c{};
    c.struct_size = sizeof(CaliperCanvasDesc);
    c.mode = CALIPER_CANVAS_OFFSCREEN;
    c.width = w;
    c.height = h;
    c.content_scale = 1.0f;
    return c;
}

// A frame is "drawn" if some pixel is markedly brighter than the ~(13,13,20)
// clear — i.e. the applet's dark-theme window / white text / plot actually
// rasterized. The clear never exceeds ~20 in any channel.
int bright_pixels(const std::vector<uint8_t>& px) {
    int n = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        uint8_t m = px[i];
        if (px[i + 1] > m) m = px[i + 1];
        if (px[i + 2] > m) m = px[i + 2];
        if (m > 80) ++n;
    }
    return n;
}

}  // namespace

TEST_CASE("embed/gate: one core per process; create-destroy cycles twice") {
    CaliperCoreDesc d = base_desc();

    CaliperCore* a = caliper_core_create(&d);
    REQUIRE(a != nullptr);

    // A second live core is refused with an honest NULL (not a crash).
    CaliperCore* b = caliper_core_create(&d);
    CHECK(b == nullptr);

    caliper_core_shutdown(a);

    // The lock cleared on shutdown, so a fresh create succeeds — twice in one
    // process (the ImGui context + service registry cycle cleanly).
    CaliperCore* c = caliper_core_create(&d);
    REQUIRE(c != nullptr);
    caliper_core_shutdown(c);
}

TEST_CASE("embed/gate: frame + event before a canvas refuse without crashing") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    // No canvas yet: frame is a no-op that records an honest error.
    caliper_core_frame(core);
    CHECK(std::string(caliper_core_last_error(core)).find("canvas") !=
          std::string::npos);

    // Event before a canvas: silently ignored, no crash.
    CaliperInputEvent ev{};
    ev.struct_size = sizeof ev;
    ev.type = CALIPER_EVENT_MOUSE_MOVE;
    ev.x = 10.0f; ev.y = 10.0f;
    caliper_core_event(core, &ev);

    caliper_core_shutdown(core);
}

TEST_CASE("embed/offscreen: load hello, pump, read non-blank pixels, clean shutdown") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    const int W = 256, H = 256;
    CaliperCanvasDesc c = offscreen_desc(W, H);
    if (!caliper_core_attach_canvas(core, nullptr, &c)) {
        MESSAGE("no embeddable GPU (offscreen attach failed) — skipping: "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    // Double attach is refused (one canvas per core in v0).
    CHECK(caliper_core_attach_canvas(core, nullptr, &c) == 0);

    // With a canvas attached, an unknown manifest id is an honest refusal
    // (this is where the unknown-id path is genuinely reached — before a canvas
    // the W1 gate below refuses first).
    CHECK(caliper_core_load_applet(core, "dev.caliper.does-not-exist") == 0);

    REQUIRE(caliper_core_load_applet(core, "dev.caliper.hello") == 1);

    // Pump a few frames: window position settles (FirstUseEver) and the font
    // atlas uploads on the first composite.
    for (int i = 0; i < 4; ++i) caliper_core_frame(core);

    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    REQUIRE(caliper_core_read_pixels(core, px.data(), W * 4) == 1);

    // The applet actually rasterized onto the canvas (not just the clear).
    CHECK(bright_pixels(px) > 50);

    caliper_core_unload_applet(core);
    caliper_core_shutdown(core);
}

TEST_CASE("embed/gate: load before a canvas refuses (W1); read_pixels needs a canvas") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    // W1: loading an applet before a canvas is attached is an honest refusal
    // (the applet's launch + first frame touch the renderer's ImGui backend,
    // which canvas_init wires). No canvas here — even the fixture id refuses,
    // and the reason names the canvas so the embedder knows the fix.
    CHECK(caliper_core_load_applet(core, "dev.caliper.hello") == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("canvas") !=
          std::string::npos);

    std::vector<uint8_t> px(16, 0);
    CHECK(caliper_core_read_pixels(core, px.data(), 4) == 0);   // no canvas

    caliper_core_shutdown(core);
}

// ---------------------------------------------------------------------------
// Reload / swap / failed-load semantics (final-review fix pass). load_applet is
// teardown-first: any active applet is torn down (workers joined, THEN instance)
// BEFORE the new launch. These three lock the documented consequences. All need
// a live offscreen canvas, so they self-skip headless like the pixel case.
// ---------------------------------------------------------------------------
namespace {
int         g_fault_count = 0;
std::string g_last_fault_id;
void record_fault(void*, const char* id, const char*) {
    ++g_fault_count;
    g_last_fault_id = id ? id : "";
}
}  // namespace

TEST_CASE("embed/reload: same-id load is a clean restart (on_cleanup then on_init)") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    CaliperCanvasDesc c = offscreen_desc(128, 128);
    if (!caliper_core_attach_canvas(core, nullptr, &c)) {
        MESSAGE("no embeddable GPU — skipping same-id restart case: "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    REQUIRE(caliper_core_load_applet(core, "dev.caliper.hello") == 1);
    caliper_core_frame(core);

    // Reload the SAME id: the running instance is torn down (on_cleanup) and a
    // fresh one launched (on_init), in that order — a restart, not a no-op or a
    // carry-over.
    std::string log;
    {
        StderrTap tap;
        REQUIRE(caliper_core_load_applet(core, "dev.caliper.hello") == 1);
        log = tap.drain();
    }
    auto cleanup_at = log.find("hello.on_cleanup");
    auto init_at    = log.find("hello.on_init");
    CHECK(cleanup_at != std::string::npos);
    CHECK(init_at    != std::string::npos);
    CHECK(cleanup_at < init_at);   // teardown-first: cleanup precedes reinit

    caliper_core_frame(core);      // the restarted applet still renders
    caliper_core_unload_applet(core);
    caliper_core_shutdown(core);
}

TEST_CASE("embed/reload: load a different id after active swaps applets") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    CaliperCanvasDesc c = offscreen_desc(256, 256);
    if (!caliper_core_attach_canvas(core, nullptr, &c)) {
        MESSAGE("no embeddable GPU — skipping swap case: "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    REQUIRE(caliper_core_load_applet(core, "dev.caliper.hello") == 1);
    caliper_core_frame(core);

    // Swap to a DIFFERENT applet: hello is torn down (on_cleanup) and sine-scope
    // launched. sine-scope schedules init jobs — under the previous cancel-
    // after-launch order those were killed at birth; teardown-first lets them
    // survive.
    std::string log;
    {
        StderrTap tap;
        REQUIRE(caliper_core_load_applet(core, "dev.example.sine-scope") == 1);
        log = tap.drain();
    }
    CHECK(log.find("hello.on_cleanup") != std::string::npos);   // old torn down

    for (int i = 0; i < 8; ++i) caliper_core_frame(core);        // sine runs
    std::vector<uint8_t> px((size_t)256 * 256 * 4, 0);
    REQUIRE(caliper_core_read_pixels(core, px.data(), 256 * 4) == 1);
    CHECK(bright_pixels(px) > 50);                               // sine rasterized

    caliper_core_unload_applet(core);
    caliper_core_shutdown(core);
}

TEST_CASE("embed/reload: a failed load after an active applet leaves NO applet") {
    g_fault_count = 0;
    g_last_fault_id.clear();

    CaliperCoreDesc d = base_desc();
    d.crash_fn = &record_fault;
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    CaliperCanvasDesc c = offscreen_desc(128, 128);
    if (!caliper_core_attach_canvas(core, nullptr, &c)) {
        MESSAGE("no embeddable GPU — skipping failed-load case: "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    REQUIRE(caliper_core_load_applet(core, "dev.caliper.hello") == 1);
    caliper_core_frame(core);

    // Force the NEXT launch to fail cleanly (initialize() returns false). Per
    // teardown-first semantics the active hello is destroyed BEFORE that launch,
    // so the refusal leaves NO applet — not the old one, not a phantom index.
    setenv("CALIPER_HELLO_INIT_FAIL", "1", 1);
    CHECK(caliper_core_load_applet(core, "dev.caliper.hello") == 0);
    unsetenv("CALIPER_HELLO_INIT_FAIL");

    // No applet loaded: a frame draws only the clear (no bright pixels) and
    // fires NO spurious fault callback (the previous order left a dangling
    // active index that frame() would have surfaced as a crash).
    caliper_core_frame(core);
    std::vector<uint8_t> px((size_t)128 * 128 * 4, 0);
    REQUIRE(caliper_core_read_pixels(core, px.data(), 128 * 4) == 1);
    CHECK(bright_pixels(px) < 10);
    CHECK(g_fault_count == 0);

    // The core is still fully usable — a fresh load of a DIFFERENT applet
    // succeeds and renders. (The init-failed hello entry is itself now marked
    // Failed by the loader and can't be relaunched — existing loader policy,
    // orthogonal to the embed teardown-order fix under test here.)
    REQUIRE(caliper_core_load_applet(core, "dev.example.sine-scope") == 1);
    for (int i = 0; i < 8; ++i) caliper_core_frame(core);
    REQUIRE(caliper_core_read_pixels(core, px.data(), 128 * 4) == 1);
    CHECK(bright_pixels(px) > 50);

    caliper_core_unload_applet(core);
    caliper_core_shutdown(core);
}

// ---------------------------------------------------------------------------
// §7 host-axis byte-compare. The design's ideal is "the gfx matrix produces the
// SAME bytes under `caliper` and `embed_host`". The exe's on-screen swapchain is
// not in-process readable, and the embed offscreen canvas composites ImGui draw
// data (not a raw geometry-texture readback), so a literal end-to-end compare of
// the two composites is out of this task's honest reach.
//
// What IS the rendering seam §7 protects — and what both hosts genuinely share —
// is the tensor->texture BRIDGE upload. Here we drive it through the renderer the
// EMBED CORE created and wired (services_set_renderer at attach), via the SAME
// public caliper.tensor_bridge.v1 service an applet uses, and byte-compare the
// readback against caliper_host::expand_u8_to_rgba8 — the IDENTICAL CPU reference
// the gfx rows (mat_u8_direct) assert against with their standalone renderer. If
// both harnesses reduce to the same bytes vs the same reference, the seam is
// byte-stable across hosts. DELTA from §7's literal ideal: bridge-upload layer,
// not the final windowed composite (stated in the L2b report).
TEST_CASE("embed/§7 host-axis: bridge upload under the embed core is byte-exact "
          "vs the shared CPU reference") {
    using namespace caliper_host;

    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    const int W = 4, H = 3;
    CaliperCanvasDesc c = offscreen_desc(W, H);
    if (!caliper_core_attach_canvas(core, nullptr, &c)) {
        MESSAGE("no embeddable GPU (offscreen attach failed) — skipping §7 case: "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    // The embed core's renderer is now bound to the process bridge. Reach both
    // through the same surfaces an applet (service) / the gfx harness (renderer)
    // would use.
    auto* bridge =
        (const CaliperTensorBridgeV1*)services_get(CALIPER_TENSOR_BRIDGE_V1);
    REQUIRE(bridge != nullptr);
    HostRenderer* r = services_renderer();
    REQUIRE(r != nullptr);

    // A deterministic (H,W,4) u8 tensor — a fixed, contiguous pattern.
    std::vector<uint8_t> src((size_t)H * W * 4);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i * 7 + 3);

    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = src.data();
    t.dtype = CALIPER_DT_U8;
    t.ndim = 3;
    t.shape[0] = H; t.shape[1] = W; t.shape[2] = 4;
    t.strides[0] = W * 4; t.strides[1] = 4; t.strides[2] = 1;
    t.device = CALIPER_DEV_CPU;

    CaliperTextureId id = bridge->texture_from_tensor(&t, 0);
    REQUIRE(id != 0);

    std::vector<uint8_t> got = r->debug_readback_rgba8(id, W, H);
    std::vector<uint8_t> ref((size_t)W * H * 4);
    expand_u8_to_rgba8(src.data(), W, H, 4, ref.data());

    REQUIRE(got.size() == ref.size());
    CHECK(got == ref);   // byte-exact: the embed-core-wired seam == the CPU ref

    bridge->release_texture(id);
    caliper_core_shutdown(core);
}

// ===========================================================================
// v1.1 — get_service for hosts (the P2/P3 "consumer" surface). These cases are
// GPU-FREE (metrics/jobs need no renderer), so they run on headless CI too.
// ===========================================================================

TEST_CASE("embed/get_service: known ids vend the process table; unknown/null refuse") {
    // NULL core -> NULL (no create needed; the contract is checkable up front).
    CHECK(caliper_core_get_service(nullptr, CALIPER_METRICS_V1) == nullptr);

    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    // No canvas attached: metrics/jobs are frame-thread-independent and vend now.
    const void* metrics = caliper_core_get_service(core, CALIPER_METRICS_V1);
    const void* jobs    = caliper_core_get_service(core, CALIPER_JOBS_V1);
    CHECK(metrics != nullptr);
    CHECK(jobs != nullptr);

    // The SAME process-static table an applet gets via CaliperHost.get_service.
    bool same_as_applet_table = (metrics == caliper_host::services_get(CALIPER_METRICS_V1));
    CHECK(same_as_applet_table);

    // Unknown id and NULL id -> NULL, never UB.
    CHECK(caliper_core_get_service(core, "does.not.exist.v1") == nullptr);
    CHECK(caliper_core_get_service(core, nullptr) == nullptr);

    caliper_core_shutdown(core);
}

// The load-bearing Compass case: a non-frame (host UI) thread queries metrics.v1
// rows in a bounded loop WHILE a jobs.v1 worker thread streams them in — the wx
// UI reading a live run. Asserts monotonic visibility (row count never shrinks)
// and no torn reads (every visible (step,value) pair is coherent). Deterministic:
// the reader drains until all N rows are visible (the writer always produces N),
// bounded by a guard — no sleep, no is_running race.
namespace {
struct ScalarStream { const CaliperMetricsV1* m; uint64_t run; int n; };
void stream_scalars(void* user, const CaliperJobControl*) {
    ScalarStream* s = (ScalarStream*)user;
    for (int i = 0; i < s->n; ++i)
        s->m->scalar(s->run, "loss", (int64_t)i, (double)i);
}
}  // namespace

TEST_CASE("embed/get_service: the Compass case — a host thread reads metrics.v1 "
          "rows while a jobs.v1 worker streams them") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    auto* metrics = (const CaliperMetricsV1*)caliper_core_get_service(core, CALIPER_METRICS_V1);
    auto* jobs    = (const CaliperJobsV1*)caliper_core_get_service(core, CALIPER_JOBS_V1);
    REQUIRE(metrics != nullptr);
    REQUIRE(jobs != nullptr);

    uint64_t run = metrics->begin_run("compass", "stream");
    if (run == 0) {   // metrics store failed to open on this box: no rows to read
        MESSAGE("metrics store not open — skipping the Compass streaming case");
        caliper_core_shutdown(core);
        return;
    }

    const int N = 200;
    ScalarStream s{metrics, run, N};
    uint64_t jid = jobs->submit("stream-scalars", &stream_scalars, &s);
    REQUIRE(jid != 0);

    // Reader loop on THIS (non-frame) thread, racing the worker's writes.
    caliper_host::MetricsStore& store = caliper_host::host_metrics_store();
    size_t prev = 0;
    bool shrank = false;
    bool torn = false;
    int guard = 0;
    std::vector<std::pair<int64_t, double>> rows;
    do {
        rows = store.scalars(run, "loss");
        if (rows.size() < prev) shrank = true;
        prev = rows.size();
        for (size_t i = 0; i < rows.size(); ++i) {
            bool step_ok  = (rows[i].first  == (int64_t)i);
            bool value_ok = (rows[i].second == (double)i);
            if (!step_ok || !value_ok) torn = true;
        }
        ++guard;
    } while (rows.size() < (size_t)N && guard < 1000000);

    metrics->end_run(run);

    CHECK(shrank == false);                 // monotonic visibility under the race
    CHECK(torn == false);                   // no torn / partial rows
    CHECK(rows.size() == (size_t)N);        // all streamed rows became visible

    caliper_core_shutdown(core);            // joins the worker (cancel_all_and_join)
}
