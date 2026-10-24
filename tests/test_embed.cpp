// caliper_embed_tests — the embed C ABI (caliper/embed.h) battery (R4 L2a).
//
// Drives the ABI the way an embedder would: create an OFFSCREEN core, load the
// hello fixture applet, pump frames, read the composited pixels back, and shut
// down clean — plus the refusal gates (second live core, frame/event before a
// canvas, double attach) that must fail HONESTLY, never crash.
//
// One CaliperCore per process (v0), so every case pairs create with shutdown.
// The pixel cases skip when no Metal device is present (offscreen attach
// fails) — counted and reported as Skipped, a failure under
// CALIPER_REQUIRE_GPU=1 (gpu_skip.h), mirroring the gfx suite.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "gpu_skip.h"

#include <caliper/embed.h>

// White-box internals for the §7 host-axis byte-compare (last case): the embed
// core wires its renderer into the process bridge at attach, so we drive the
// SAME public tensor_bridge.v1 service and read the texture back through the
// renderer, comparing to the shared CPU reference the gfx rows use.
#include <caliper/tensor.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/metrics_v1.h>   // v1.1 get_service: the Compass case
#include <caliper/services/metrics_v1_1.h> // C0b: the metrics READ surface (Arrow)
#include <caliper/services/jobs_v1.h>      // v1.1 get_service: cross-thread producer
#include <caliper/services/log_v1.h>       // v1.1 log routing
#include <caliper/arrow_c.h>               // C0b: drain metrics.v1_1 Arrow streams
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
#include <mutex>
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
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (offscreen attach failed): "
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
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (same-id restart case): "
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

#ifndef CALIPER_TESTS_NO_TORCH_APPLETS
// Needs dev.example.sine-scope, which links libtorch — compiled out of the
// torch-free build (tests/CMakeLists.txt defines CALIPER_TESTS_NO_TORCH_APPLETS).
TEST_CASE("embed/reload: load a different id after active swaps applets") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    CaliperCanvasDesc c = offscreen_desc(256, 256);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (swap case): "
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
#endif  // CALIPER_TESTS_NO_TORCH_APPLETS

TEST_CASE("embed/reload: a failed load after an active applet leaves NO applet") {
    g_fault_count = 0;
    g_last_fault_id.clear();

    CaliperCoreDesc d = base_desc();
    d.crash_fn = &record_fault;
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    CaliperCanvasDesc c = offscreen_desc(128, 128);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (failed-load case): "
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

#ifndef CALIPER_TESTS_NO_TORCH_APPLETS
    // The core is still fully usable — a fresh load of a DIFFERENT applet
    // succeeds and renders. (The init-failed hello entry is itself now marked
    // Failed by the loader and can't be relaunched — existing loader policy,
    // orthogonal to the embed teardown-order fix under test here.) sine-scope
    // links libtorch, so this half is compiled out of the torch-free build.
    REQUIRE(caliper_core_load_applet(core, "dev.example.sine-scope") == 1);
    for (int i = 0; i < 8; ++i) caliper_core_frame(core);
    REQUIRE(caliper_core_read_pixels(core, px.data(), 128 * 4) == 1);
    CHECK(bright_pixels(px) > 50);
#endif

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
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (offscreen attach failed, §7 case): "
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

// ===========================================================================
// C0b — the metrics READ surface across the ABI (caliper.metrics.v1_1). The
// PURE-ABI Compass path: a host reads runs/scalars through get_service +
// metrics.v1_1.query (Arrow C stream), NOT the host-private MetricsStore. This
// is what an out-of-tree consumer (Compass) can actually reach. GPU-free.
// ===========================================================================
namespace {
// Drain a metrics.v1_1 (step BIGINT, value DOUBLE) Arrow stream into a vector.
// Returns false if the stream schema/rows can't be read (never throws here).
bool drain_step_value(ArrowArrayStream* st,
                      std::vector<std::pair<int64_t, double>>& out) {
    ArrowSchema schema = {};
    if (st->get_schema(st, &schema) != 0) { st->release(st); return false; }
    if (schema.release) schema.release(&schema);
    for (;;) {
        ArrowArray array = {};
        if (st->get_next(st, &array) != 0) { st->release(st); return false; }
        if (!array.release) break;                      // end of stream (spec)
        const ArrowArray* cs = array.children[0];
        const ArrowArray* cv = array.children[1];
        const int64_t* sp = (const int64_t*)cs->buffers[1];
        const double*  vp = (const double*)cv->buffers[1];
        for (int64_t i = 0; i < array.length; i++)
            out.emplace_back(sp[cs->offset + i], vp[cv->offset + i]);
        array.release(&array);
    }
    st->release(st);
    return true;
}
}  // namespace

TEST_CASE("embed/metrics.v1_1: a host reads runs+scalars via the ABI (Arrow) "
          "while a jobs.v1 worker streams them; non-SELECT refused") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    auto* metrics = (const CaliperMetricsV1*)caliper_core_get_service(core, CALIPER_METRICS_V1);
    auto* reader  = (const CaliperMetricsV1_1*)caliper_core_get_service(core, CALIPER_METRICS_V1_1);
    auto* jobs    = (const CaliperJobsV1*)caliper_core_get_service(core, CALIPER_JOBS_V1);
    REQUIRE(metrics != nullptr);
    REQUIRE(reader != nullptr);
    REQUIRE(jobs != nullptr);

    // v1_1 vends the SAME process-static table an applet would get.
    bool same_as_applet_table =
        ((const void*)reader == caliper_host::services_get(CALIPER_METRICS_V1_1));
    CHECK(same_as_applet_table);

    uint64_t run = metrics->begin_run("compass", "arrow-stream");
    if (run == 0) {   // metrics store failed to open on this box: nothing to read
        MESSAGE("metrics store not open — skipping the metrics.v1_1 Arrow case");
        caliper_core_shutdown(core);
        return;
    }

    const int N = 200;
    ScalarStream s{metrics, run, N};
    uint64_t jid = jobs->submit("stream-scalars", &stream_scalars, &s);
    REQUIRE(jid != 0);

    // Host-thread reader loop, purely through the ABI: query() a fresh Arrow
    // stream each pass and drain it, racing the worker's writes. Bounded/
    // deterministic (the writer always produces N) — no sleep, no is_running.
    const std::string sql =
        "SELECT step, value FROM scalars WHERE run = " + std::to_string(run) +
        " AND tag = 'loss' ORDER BY step";
    size_t prev = 0; bool shrank = false, torn = false; int guard = 0;
    std::vector<std::pair<int64_t, double>> rows;
    do {
        rows.clear();
        ArrowArrayStream st = {};
        REQUIRE(reader->query(sql.c_str(), &st) == true);
        REQUIRE(drain_step_value(&st, rows) == true);
        if (rows.size() < prev) shrank = true;
        prev = rows.size();
        for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i].first != (int64_t)i || rows[i].second != (double)i) torn = true;
        ++guard;
    } while (rows.size() < (size_t)N && guard < 1000000);

    metrics->end_run(run);

    CHECK(shrank == false);              // monotonic visibility under the race
    CHECK(torn == false);                // no torn / partial rows
    CHECK(rows.size() == (size_t)N);     // all streamed rows became visible

    // List runs through the ABI too (the runs-browser query).
    {
        ArrowArrayStream st = {};
        REQUIRE(reader->query("SELECT COUNT(*) FROM runs", &st) == true);
        ArrowSchema sc = {}; REQUIRE(st.get_schema(&st, &sc) == 0);
        if (sc.release) sc.release(&sc);
        ArrowArray a = {}; REQUIRE(st.get_next(&st, &a) == 0);
        REQUIRE(a.release != nullptr);
        int64_t n_runs = ((const int64_t*)a.children[0]->buffers[1])[a.children[0]->offset];
        CHECK(n_runs >= 1);
        a.release(&a);
        ArrowArray end = {}; st.get_next(&st, &end);   // drain to end
        st.release(&st);
    }

    // Read-only ENFORCED across the ABI: a write is refused, the stream is never
    // filled, last_error explains, and the store is untouched (run still there).
    {
        ArrowArrayStream st = {};
        CHECK(reader->query("DROP TABLE runs", &st) == false);
        CHECK(st.release == nullptr);
        CHECK(std::string(reader->last_error()).empty() == false);

        ArrowArrayStream st2 = {};
        REQUIRE(reader->query("SELECT COUNT(*) FROM runs", &st2) == true);
        st2.release(&st2);   // the runs table survived the refused DROP
    }

    caliper_core_shutdown(core);         // joins the worker (cancel_all_and_join)
}

// ===========================================================================
// v1.1 — data_dir routing (§3.3). A non-NULL data_dir roots the stores under
// the embedder's directory; NULL keeps the OS default byte-for-byte.
// ===========================================================================
TEST_CASE("embed/data_dir: a per-core data_dir roots the stores under it; "
          "NULL keeps the default") {
    namespace fs = std::filesystem;
    // A temp root distinct from the default app-data. Cleaned up front so a
    // stale prior run cannot mask the "landed under root" assertion (portable:
    // no getpid, the Windows box reruns this battery).
    fs::path root = fs::temp_directory_path() / "caliper_embed_datadir_test";
    std::error_code ec;
    fs::remove_all(root, ec);

    // --- data_dir set: the metrics store must land UNDER root ---
    CaliperCoreDesc d = base_desc();
    std::string root_str = root.string();
    d.data_dir = root_str.c_str();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    // Drive a metrics write so the store is exercised (opened at create).
    auto* metrics = (const CaliperMetricsV1*)caliper_core_get_service(core, CALIPER_METRICS_V1);
    REQUIRE(metrics != nullptr);
    uint64_t run = metrics->begin_run("proj", "r0");
    metrics->scalar(run, "loss", 0, 1.0);
    metrics->end_run(run);

    bool db_under_root = fs::exists(root / "metrics.duckdb");
    caliper_core_shutdown(core);
    CHECK(db_under_root);   // routed under the embedder's directory, not default

    // --- data_dir NULL: the default root is used, and the override did NOT leak ---
    CaliperCoreDesc d2 = base_desc();
    d2.data_dir = nullptr;
    CaliperCore* core2 = caliper_core_create(&d2);
    REQUIRE(core2 != nullptr);

    // After a shut-down override, path resolution is back to the OS default:
    // the resolved metrics path must NOT sit under the temp root.
    std::string def_path = caliper::app_data_path("metrics.duckdb");
    bool default_restored = (def_path.find(root_str) == std::string::npos);
    CHECK(default_restored);

    caliper_core_shutdown(core2);
    fs::remove_all(root, ec);
}

// ===========================================================================
// v1.1 — log.v1 routing (§3.3). An installed log_fn captures applet log lines
// (tagged "[applet] ") that were stderr-only in v0; with NO log_fn the caliper
// exe path stays stderr, unchanged.
// ===========================================================================
namespace {
std::mutex  g_log_mu;
std::string g_log_capture;
void capture_log_fn(void*, int /*level*/, const char* msg) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log_capture += (msg ? msg : "");
    g_log_capture += "\n";
}
std::string drain_log_capture() {
    std::lock_guard<std::mutex> lk(g_log_mu);
    return g_log_capture;
}
void reset_log_capture() {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_log_capture.clear();
}
}  // namespace

TEST_CASE("embed/log: with log_fn installed, a log.v1 line reaches the sink "
          "tagged, NOT raw stderr") {
    reset_log_capture();
    CaliperCoreDesc d = base_desc();
    d.log_fn = &capture_log_fn;
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    auto* log = (const CaliperLogV1*)caliper_core_get_service(core, CALIPER_LOG_V1);
    REQUIRE(log != nullptr);

    // Drive log.v1 directly (GPU-free): the applet-facing service an applet uses.
    std::string err;
    {
        StderrTap tap;
        log->log(CALIPER_LOG_INFO, "compass-log-probe");
        err = tap.drain();
    }
    std::string cap = drain_log_capture();
    bool reached_sink       = cap.find("compass-log-probe") != std::string::npos;
    bool tagged_applet      = cap.find("[applet] compass-log-probe") != std::string::npos;
    bool absent_from_stderr = err.find("compass-log-probe") == std::string::npos;
    CHECK(reached_sink);
    CHECK(tagged_applet);
    CHECK(absent_from_stderr);

    caliper_core_shutdown(core);
}

TEST_CASE("embed/log: without log_fn, log.v1 still writes stderr (exe path unchanged)") {
    CaliperCoreDesc d = base_desc();   // no log_fn: the caliper exe configuration
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    auto* log = (const CaliperLogV1*)caliper_core_get_service(core, CALIPER_LOG_V1);
    REQUIRE(log != nullptr);

    std::string err;
    {
        StderrTap tap;
        log->log(CALIPER_LOG_WARN, "compass-stderr-probe");
        err = tap.drain();
    }
    bool on_stderr = err.find("compass-stderr-probe") != std::string::npos;
    CHECK(on_stderr);   // the built-in stderr writer, exactly as v0 / the exe

    caliper_core_shutdown(core);
}

TEST_CASE("embed/log: a live applet's log.v1 line routes to the embedder sink, "
          "not stderr") {
    reset_log_capture();
    CaliperCoreDesc d = base_desc();
    d.log_fn = &capture_log_fn;
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    CaliperCanvasDesc c = offscreen_desc(128, 128);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (applet-log routing case): "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    // hello.on_init logs "hello.on_init" through caliper.log.v1 (host.log_info).
    std::string err;
    {
        StderrTap tap;
        int loaded = caliper_core_load_applet(core, "dev.caliper.hello");
        REQUIRE(loaded == 1);
        caliper_core_frame(core);
        err = tap.drain();
    }
    std::string cap = drain_log_capture();
    bool applet_line_at_sink = cap.find("hello.on_init") != std::string::npos;
    bool applet_line_off_stderr = err.find("hello.on_init") == std::string::npos;
    CHECK(applet_line_at_sink);       // reached the embedder's log pane
    CHECK(applet_line_off_stderr);    // no longer leaks to raw stderr

    caliper_core_unload_applet(core);
    caliper_core_shutdown(core);
}

// ===========================================================================
// v1.2 — built-in tensor views (caliper_core_view_*). The Python entry point's
// caliper.show: the core draws a host tensor itself. Byte-exact against the
// SAME CPU reference the bridge/gfx rows use (map_f32_to_rgba8 + colormap_lut),
// at two levels — the view's texture (view_readback) and the composited canvas
// (RAW view + read_pixels). The METAL cases build real MTLBuffers from this
// plain C++ TU through the ObjC runtime (dlsym — no ObjC compile, no extra link)
// and prove the zero-copy imported path at a NONZERO byte offset. All GPU cases
// skip headless like the pixel cases above (gpu_skip.h: counted, Skipped).
// ===========================================================================
#ifdef __APPLE__
#  include <dlfcn.h>
#endif

namespace {

#ifdef __APPLE__
// Minimal Metal through the ObjC runtime C API, resolved at run time from the
// already-loaded Metal + libobjc images (libcaliper's Metal backend links them).
// objc_msgSend is cast to each exact non-variadic signature (arm64 ABI).
struct MetalRT {
    void* (*sel)(const char*) = nullptr;
    void* msg = nullptr;
    void* device = nullptr;

    MetalRT() {
        auto create = (void* (*)())dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice");
        sel = (void* (*)(const char*))dlsym(RTLD_DEFAULT, "sel_registerName");
        msg = dlsym(RTLD_DEFAULT, "objc_msgSend");
        // CALIPER_TEST_FORCE_NO_GPU=1 reports no device, like the canvas gates.
        if (create && sel && msg && !caliper_test::force_no_gpu())
            device = create();   // +1 (create rule)
    }
    ~MetalRT() { release(device); }
    bool ok() const { return device != nullptr; }

    // [device newBufferWithBytes:len:options:MTLResourceStorageModeShared] (+1)
    void* new_buffer(const void* bytes, size_t len) {
        using Fn = void* (*)(void*, void*, const void*, unsigned long, unsigned long);
        return ((Fn)msg)(device, sel("newBufferWithBytes:length:options:"),
                         bytes, (unsigned long)len, 0ul);
    }
    void* contents(void* buf) {
        using Fn = void* (*)(void*, void*);
        return ((Fn)msg)(buf, sel("contents"));
    }
    void release(void* obj) {
        if (!obj || !msg) return;
        using Fn = void (*)(void*, void*);
        ((Fn)msg)(obj, sel("release"));
    }
};
#endif

CaliperTensor make_tensor(void* data, CaliperDType dt, std::vector<int64_t> shape,
                          CaliperDeviceKind dev) {
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = data;
    t.dtype = dt;
    t.ndim = (int32_t)shape.size();
    for (int i = 0; i < t.ndim; ++i) t.shape[i] = shape[i];
    int64_t st = 1;
    for (int i = t.ndim - 1; i >= 0; --i) { t.strides[i] = st; st *= t.shape[i]; }
    t.device = dev;
    return t;
}

CaliperViewDesc make_view(const char* name, const CaliperTensor* t, uint32_t flags,
                          int32_t cm = 0, float vmin = 0.0f, float vmax = 1.0f,
                          uint64_t byte_offset = 0) {
    CaliperViewDesc d{};
    d.struct_size = sizeof d;
    d.name = name;
    d.tensor = t;
    d.byte_offset = byte_offset;
    d.colormap = cm;
    d.vmin = vmin;
    d.vmax = vmax;
    d.flags = flags;
    return d;
}

CaliperViewInfo view_info(CaliperCore* core, const char* name, int* ok = nullptr) {
    CaliperViewInfo vi{};
    vi.struct_size = sizeof vi;
    const int r = caliper_core_view_info(core, name, &vi);
    if (ok) *ok = r;
    return vi;
}

std::vector<float> ramp_f32(int W, int H, bool reversed = false) {
    std::vector<float> v((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float r = (float)x / (float)(W - 1);
            v[(size_t)y * W + x] = reversed ? 1.0f - r : r;
        }
    return v;
}

std::vector<uint8_t> map_ref(const std::vector<float>& v, int W, int H, int32_t cm,
                             float vmin, float vmax) {
    std::vector<uint8_t> out((size_t)W * H * 4);
    caliper_host::map_f32_to_rgba8(v.data(), W, H, caliper_host::colormap_lut(cm),
                                   vmin, vmax, out.data());
    return out;
}

int max_abs_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return 256;
    int m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const int d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
        if (d > m) m = d;
    }
    return m;
}

std::vector<uint8_t> view_readback(CaliperCore* core, const char* name, int W, int H) {
    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    if (!caliper_core_view_readback(core, name, px.data(), px.size())) px.clear();
    return px;
}

// Pack f32 values after `junk` bytes of 0xAB filler, as one byte blob.
std::vector<uint8_t> blob_at(size_t junk, const void* data, size_t bytes) {
    std::vector<uint8_t> b(junk + bytes, 0xAB);
    std::memcpy(b.data() + junk, data, bytes);
    return b;
}

}  // namespace

TEST_CASE("embed/views: colormap_lut ABI == the bridge's internal LUTs") {
    for (int32_t cm = 0; cm < 3; ++cm) {
        uint32_t lut[256] = {};
        REQUIRE(caliper_core_colormap_lut(cm, lut) == 1);
        CHECK(std::memcmp(lut, caliper_host::colormap_lut(cm), sizeof lut) == 0);
        CHECK((lut[0] >> 24) == 0xffu);   // opaque
    }
    uint32_t lut[256];
    CHECK(caliper_core_colormap_lut(3, lut) == 0);
    CHECK(caliper_core_colormap_lut(-1, lut) == 0);
}

TEST_CASE("embed/views: RAW f32 ramp (CPU) — texture and canvas byte-exact vs the "
          "CPU colormap reference") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);
    const int W = 64, H = 48;
    CaliperCanvasDesc c = offscreen_desc(W, H);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (RAW ramp case): "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    std::vector<float> ramp = ramp_f32(W, H);
    CaliperTensor t = make_tensor(ramp.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperViewDesc vd = make_view("ramp", &t, CALIPER_VIEW_FLAG_RAW, 0, 0.0f, 1.0f);
    REQUIRE_MESSAGE(caliper_core_view_tensor(core, &vd) == 1,
                    caliper_core_last_error(core));
    CHECK(caliper_core_view_count(core) == 1);

    int ok = 0;
    CaliperViewInfo vi = view_info(core, "ramp", &ok);
    REQUIRE(ok == 1);
    CHECK(std::string(vi.path) == "cpu-staged");
    CHECK(vi.imported == 0);
    CHECK(vi.width == W);
    CHECK(vi.height == H);
    CHECK(vi.device == CALIPER_DEV_CPU);
    CHECK(vi.updates == 1);
    CHECK(vi.creates == 1);

    const std::vector<uint8_t> ref = map_ref(ramp, W, H, 0, 0.0f, 1.0f);

    // Texture level: byte-exact.
    CHECK(view_readback(core, "ramp", W, H) == ref);

    // Canvas level: the RAW view composites 1:1 at (0,0), nearest-sampled.
    caliper_core_frame(core);
    caliper_core_frame(core);
    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    REQUIRE(caliper_core_read_pixels(core, px.data(), W * 4) == 1);
    const int diff = max_abs_diff(px, ref);
    MESSAGE("RAW ramp canvas vs reference: max abs diff = " << diff);
    CHECK(diff == 0);

    caliper_core_shutdown(core);
}

#ifdef __APPLE__
TEST_CASE("embed/views: METAL f32 at a nonzero byte_offset is zero-copy "
          "(compute-imported); re-import, mapping change, in-place rewrite") {
    MetalRT mt;
    if (!mt.ok()) CALIPER_SKIP_NO_GPU("no Metal device");
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);
    const int W = 64, H = 48;
    CaliperCanvasDesc c = offscreen_desc(W, H);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (METAL view case): "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }
    const size_t bytes = (size_t)W * H * sizeof(float);

    // Buffer 1: 64 bytes of junk, then the ramp at byte_offset 64.
    std::vector<float> ramp = ramp_f32(W, H);
    std::vector<uint8_t> blob1 = blob_at(64, ramp.data(), bytes);
    void* buf1 = mt.new_buffer(blob1.data(), blob1.size());
    REQUIRE(buf1 != nullptr);

    CaliperTensor t1 = make_tensor(buf1, CALIPER_DT_F32, {H, W}, CALIPER_DEV_METAL);
    CaliperViewDesc vd = make_view("act", &t1,
                                   CALIPER_VIEW_FLAG_RAW | CALIPER_VIEW_FLAG_WAIT_GPU,
                                   0, 0.0f, 1.0f, /*byte_offset=*/64);
    REQUIRE_MESSAGE(caliper_core_view_tensor(core, &vd) == 1,
                    caliper_core_last_error(core));
    CaliperViewInfo vi = view_info(core, "act");
    CHECK(std::string(vi.path) == "compute-imported");
    CHECK(vi.imported == 1);
    CHECK(vi.device == CALIPER_DEV_METAL);
    CHECK(vi.width == W);
    CHECK(vi.height == H);
    CHECK(vi.updates == 1);
    CHECK(vi.creates == 1);
    MESSAGE("METAL f32 64x48 view_tensor (create + WAIT_GPU): " << vi.last_update_ms << " ms");
    CHECK(view_readback(core, "act", W, H) == map_ref(ramp, W, H, 0, 0.0f, 1.0f));

    // The composite from the device path matches too.
    caliper_core_frame(core);
    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    REQUIRE(caliper_core_read_pixels(core, px.data(), W * 4) == 1);
    CHECK(max_abs_diff(px, map_ref(ramp, W, H, 0, 0.0f, 1.0f)) == 0);

    // Buffer 2 (a DIFFERENT MTLBuffer -> re-import): reversed ramp at offset 128.
    std::vector<float> rev = ramp_f32(W, H, /*reversed=*/true);
    std::vector<uint8_t> blob2 = blob_at(128, rev.data(), bytes);
    void* buf2 = mt.new_buffer(blob2.data(), blob2.size());
    REQUIRE(buf2 != nullptr);
    CaliperTensor t2 = make_tensor(buf2, CALIPER_DT_F32, {H, W}, CALIPER_DEV_METAL);
    vd.tensor = &t2;
    vd.byte_offset = 128;
    REQUIRE_MESSAGE(caliper_core_view_tensor(core, &vd) == 1,
                    caliper_core_last_error(core));
    vi = view_info(core, "act");
    CHECK(std::string(vi.path) == "compute-imported");
    CHECK(vi.imported == 1);
    CHECK(vi.updates == 2);
    CHECK(vi.creates == 1);            // same shape: texture kept
    CHECK(view_readback(core, "act", W, H) == map_ref(rev, W, H, 0, 0.0f, 1.0f));

    // Mapping change (vmin/vmax) on the live view: re-pinned, NOT recreated.
    vd.vmin = 0.25f;
    vd.vmax = 0.75f;
    REQUIRE_MESSAGE(caliper_core_view_tensor(core, &vd) == 1,
                    caliper_core_last_error(core));
    vi = view_info(core, "act");
    CHECK(vi.updates == 3);
    CHECK(vi.creates == 1);
    CHECK(vi.width == W);
    CHECK(vi.height == H);
    CHECK(view_readback(core, "act", W, H) == map_ref(rev, W, H, 0, 0.25f, 0.75f));

    // Colormap change (magma), same texture.
    vd.colormap = 1;
    REQUIRE(caliper_core_view_tensor(core, &vd) == 1);
    vi = view_info(core, "act");
    CHECK(vi.creates == 1);
    CHECK(view_readback(core, "act", W, H) == map_ref(rev, W, H, 1, 0.25f, 0.75f));

    // In-place rewrite of the SAME buffer (cached import): the view reads the
    // buffer at update time, so the new contents show.
    std::vector<float> fwd = ramp_f32(W, H);
    std::memcpy((uint8_t*)mt.contents(buf2) + 128, fwd.data(), bytes);
    REQUIRE(caliper_core_view_tensor(core, &vd) == 1);
    vi = view_info(core, "act");
    CHECK(vi.updates == 5);
    CHECK(view_readback(core, "act", W, H) == map_ref(fwd, W, H, 1, 0.25f, 0.75f));

    // Without WAIT_GPU the call returns before the GPU read; the readback
    // (same-queue) still observes it. Report both timings for the record.
    const double waited_ms = vi.last_update_ms;
    vd.flags = CALIPER_VIEW_FLAG_RAW;
    REQUIRE(caliper_core_view_tensor(core, &vd) == 1);
    vi = view_info(core, "act");
    MESSAGE("METAL f32 64x48 update: WAIT_GPU " << waited_ms << " ms, no-wait "
            << vi.last_update_ms << " ms");
    CHECK(view_readback(core, "act", W, H) == map_ref(fwd, W, H, 1, 0.25f, 0.75f));

    // Shape change -> recreate (creates == 2), still zero-copy.
    CaliperTensor t3 = make_tensor(buf2, CALIPER_DT_F32, {H / 2, W}, CALIPER_DEV_METAL);
    vd.tensor = &t3;
    REQUIRE(caliper_core_view_tensor(core, &vd) == 1);
    vi = view_info(core, "act");
    CHECK(vi.creates == 2);
    CHECK(vi.height == H / 2);
    CHECK(std::string(vi.path) == "compute-imported");

    caliper_core_shutdown(core);   // releases the view's texture + import
    mt.release(buf1);
    mt.release(buf2);
}

TEST_CASE("embed/views: METAL (H,W,4) u8 image takes blit-imported; readback == bytes") {
    MetalRT mt;
    if (!mt.ok()) CALIPER_SKIP_NO_GPU("no Metal device");
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);
    CaliperCanvasDesc c = offscreen_desc(64, 64);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (METAL image case): "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }
    const int W = 40, H = 24;
    std::vector<uint8_t> img((size_t)W * H * 4);
    for (size_t i = 0; i < img.size(); ++i) img[i] = (uint8_t)(i * 7 + 3);
    std::vector<uint8_t> blob = blob_at(256, img.data(), img.size());
    void* buf = mt.new_buffer(blob.data(), blob.size());
    REQUIRE(buf != nullptr);

    CaliperTensor t = make_tensor(buf, CALIPER_DT_U8, {H, W, 4}, CALIPER_DEV_METAL);
    CaliperViewDesc vd = make_view("image", &t, CALIPER_VIEW_FLAG_WAIT_GPU, 0, 0, 1, 256);
    REQUIRE_MESSAGE(caliper_core_view_tensor(core, &vd) == 1,
                    caliper_core_last_error(core));
    CaliperViewInfo vi = view_info(core, "image");
    CHECK(std::string(vi.path) == "blit-imported");
    CHECK(vi.imported == 1);
    CHECK(vi.width == W);
    CHECK(vi.height == H);
    CHECK(view_readback(core, "image", W, H) == img);

    // A windowed (non-RAW) view composites without fault.
    for (int i = 0; i < 2; ++i) caliper_core_frame(core);

    caliper_core_shutdown(core);
    mt.release(buf);
}
#endif  // __APPLE__

#ifdef __APPLE__
TEST_CASE("embed/views: measured cost at 1024x1024 f32 (CPU-staged vs METAL zero-copy)") {
    MetalRT mt;
    if (!mt.ok()) CALIPER_SKIP_NO_GPU("no Metal device");
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);
    CaliperCanvasDesc c = offscreen_desc(64, 64);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (1024x1024 cost case): "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }
    const int N = 1024;
    std::vector<float> v((size_t)N * N);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float)(i % 977) / 976.0f;
    void* buf = mt.new_buffer(v.data(), v.size() * sizeof(float));
    REQUIRE(buf != nullptr);
    CaliperTensor tc = make_tensor(v.data(), CALIPER_DT_F32, {N, N}, CALIPER_DEV_CPU);
    CaliperTensor tm = make_tensor(buf, CALIPER_DT_F32, {N, N}, CALIPER_DEV_METAL);
    CaliperViewDesc dc = make_view("cpu", &tc, 0);
    CaliperViewDesc dm = make_view("metal", &tm, CALIPER_VIEW_FLAG_WAIT_GPU);
    REQUIRE(caliper_core_view_tensor(core, &dc) == 1);   // creates
    REQUIRE(caliper_core_view_tensor(core, &dm) == 1);
    double cpu_ms = 0, metal_ms = 0;
    const int reps = 20;
    for (int i = 0; i < reps; ++i) {
        REQUIRE(caliper_core_view_tensor(core, &dc) == 1);
        cpu_ms += view_info(core, "cpu").last_update_ms;
        REQUIRE(caliper_core_view_tensor(core, &dm) == 1);
        metal_ms += view_info(core, "metal").last_update_ms;
    }
    CaliperViewInfo vi = view_info(core, "metal");
    CHECK(std::string(vi.path) == "compute-imported");
    CHECK(vi.imported == 1);
    MESSAGE("1024x1024 f32 update, mean of " << reps << ": cpu-staged "
            << cpu_ms / reps << " ms, metal compute-imported + WAIT_GPU "
            << metal_ms / reps << " ms");
    CHECK(view_readback(core, "metal", N, N) == view_readback(core, "cpu", N, N));
    caliper_core_shutdown(core);
    mt.release(buf);
}
#endif

TEST_CASE("embed/views: windowed view draws (chrome + image) next to nothing else") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);
    CaliperCanvasDesc c = offscreen_desc(256, 256);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (windowed view case): "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }
    // Nothing loaded, nothing shown: only the dark clear.
    caliper_core_frame(core);
    std::vector<uint8_t> px((size_t)256 * 256 * 4, 0);
    REQUIRE(caliper_core_read_pixels(core, px.data(), 256 * 4) == 1);
    CHECK(bright_pixels(px) < 10);

    // A 16x16 ramp in a normal window (magnified, nearest): bright yellow end.
    std::vector<float> ramp = ramp_f32(16, 16);
    CaliperTensor t = make_tensor(ramp.data(), CALIPER_DT_F32, {16, 16}, CALIPER_DEV_CPU);
    CaliperViewDesc vd = make_view("weights", &t, 0);
    REQUIRE(caliper_core_view_tensor(core, &vd) == 1);
    for (int i = 0; i < 3; ++i) caliper_core_frame(core);
    REQUIRE(caliper_core_read_pixels(core, px.data(), 256 * 4) == 1);
    CHECK(bright_pixels(px) > 200);

    // Removing it returns the canvas to the clear.
    CHECK(caliper_core_view_remove(core, "weights") == 1);
    CHECK(caliper_core_view_count(core) == 0);
    caliper_core_frame(core);
    REQUIRE(caliper_core_read_pixels(core, px.data(), 256 * 4) == 1);
    CHECK(bright_pixels(px) < 10);

    caliper_core_shutdown(core);
}

TEST_CASE("embed/views: gates refuse honestly; shutdown with live views is clean") {
    CaliperCoreDesc d = base_desc();
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);

    std::vector<float> f(8 * 8, 0.5f);
    CaliperTensor tf = make_tensor(f.data(), CALIPER_DT_F32, {8, 8}, CALIPER_DEV_CPU);
    CaliperViewDesc vd = make_view("v", &tf, 0);

    // Before a canvas: refused, reason names the canvas.
    CHECK(caliper_core_view_tensor(core, &vd) == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("canvas") != std::string::npos);
    CHECK(caliper_core_view_count(core) == 0);

    // Unknown names.
    CaliperViewInfo vi{};
    vi.struct_size = sizeof vi;
    CHECK(caliper_core_view_info(core, "nope", &vi) == 0);
    CHECK(caliper_core_view_remove(core, "nope") == 0);
    uint8_t one[4];
    CHECK(caliper_core_view_readback(core, "nope", one, sizeof one) == 0);
    // Null core is safe everywhere.
    CHECK(caliper_core_view_tensor(nullptr, &vd) == 0);
    CHECK(caliper_core_view_count(nullptr) == 0);

    CaliperCanvasDesc c = offscreen_desc(64, 64);
    if (caliper_test::force_no_gpu() ||
        !caliper_core_attach_canvas(core, nullptr, &c)) {
        CALIPER_NOTE_NO_GPU("no embeddable GPU (the gates case's canvas half): "
                << caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return;
    }

    // Short struct_size.
    CaliperViewDesc shortd = vd;
    shortd.struct_size = sizeof(CaliperViewDesc) - 4;
    CHECK(caliper_core_view_tensor(core, &shortd) == 0);
    // Empty name.
    CaliperViewDesc noname = vd;
    noname.name = "";
    CHECK(caliper_core_view_tensor(core, &noname) == 0);
    // Wrong rank/dtype.
    std::vector<uint8_t> u(8 * 8 * 4, 0);
    CaliperTensor t2u8 = make_tensor(u.data(), CALIPER_DT_U8, {8, 32}, CALIPER_DEV_CPU);
    vd.tensor = &t2u8;
    CHECK(caliper_core_view_tensor(core, &vd) == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("want a 2-D f32") != std::string::npos);
    // Non-contiguous.
    CaliperTensor tnc = tf;
    tnc.strides[0] = 16;
    vd.tensor = &tnc;
    CHECK(caliper_core_view_tensor(core, &vd) == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("contiguous") != std::string::npos);
    // Bad colormap.
    vd.tensor = &tf;
    vd.colormap = 7;
    CHECK(caliper_core_view_tensor(core, &vd) == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("colormap") != std::string::npos);
    vd.colormap = 0;
    CHECK(caliper_core_view_count(core) == 0);   // no failed call left a view

#ifdef __APPLE__
    MetalRT mt;
    if (mt.ok()) {
        std::vector<uint8_t> rgb(8 * 8 * 3, 9);
        void* buf = mt.new_buffer(rgb.data(), rgb.size());
        REQUIRE(buf != nullptr);
        // (H,W,3) u8 on METAL: refused, the reason says RGBA.
        CaliperTensor t3 = make_tensor(buf, CALIPER_DT_U8, {8, 8, 3}, CALIPER_DEV_METAL);
        vd.tensor = &t3;
        CHECK(caliper_core_view_tensor(core, &vd) == 0);
        CHECK(std::string(caliper_core_last_error(core)).find("RGBA") != std::string::npos);
        // Misaligned METAL byte_offset.
        CaliperTensor tm = make_tensor(buf, CALIPER_DT_F32, {4, 4}, CALIPER_DEV_METAL);
        vd.tensor = &tm;
        vd.byte_offset = 2;
        CHECK(caliper_core_view_tensor(core, &vd) == 0);
        CHECK(std::string(caliper_core_last_error(core)).find("multiple of 4") != std::string::npos);
        // Window past the buffer end: the import refuses — and nothing falls
        // back to a host copy, no view is left behind.
        vd.byte_offset = 4096;
        CHECK(caliper_core_view_tensor(core, &vd) == 0);
        CHECK(std::string(caliper_core_last_error(core)).find("import refused") != std::string::npos);
        CHECK(caliper_core_view_count(core) == 0);
        vd.byte_offset = 0;

        // A live METAL view held across shutdown.
        CHECK(caliper_core_view_tensor(core, &vd) == 1);
        // A failed update of an EXISTING view keeps it (and its pixels).
        vd.byte_offset = 4096;
        CHECK(caliper_core_view_tensor(core, &vd) == 0);
        CHECK(caliper_core_view_count(core) == 1);
        vd.byte_offset = 0;
        caliper_core_frame(core);
        // Shut down with the view live, then drop our buffer.
        vd.name = "cpu-too";
        vd.tensor = &tf;
        CHECK(caliper_core_view_tensor(core, &vd) == 1);
        CHECK(caliper_core_view_count(core) == 2);
        caliper_core_frame(core);
        caliper_core_shutdown(core);
        mt.release(buf);
        core = nullptr;
    }
#endif
    if (core) {
        vd.name = "live";
        vd.tensor = &tf;
        CHECK(caliper_core_view_tensor(core, &vd) == 1);
        caliper_core_frame(core);
        caliper_core_shutdown(core);   // live view: released before the renderer
    }

    // A fresh core after a shutdown-with-views works and starts empty.
    CaliperCore* again = caliper_core_create(&d);
    REQUIRE(again != nullptr);
    CHECK(caliper_core_view_count(again) == 0);
    CaliperCanvasDesc c2 = offscreen_desc(32, 32);
    if (caliper_core_attach_canvas(again, nullptr, &c2)) {
        vd.name = "after";
        vd.tensor = &tf;
        vd.flags = CALIPER_VIEW_FLAG_RAW;
        CHECK(caliper_core_view_tensor(again, &vd) == 1);
        caliper_core_frame(again);
    }
    caliper_core_shutdown(again);
}

CALIPER_GPU_TEST_MAIN()
