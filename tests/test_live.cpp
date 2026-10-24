// caliper_live_tests: the live-streaming protocol (caliper/live.h) end to end
// in ONE process. The client API creates a session and views exactly as the
// training process does; an OFFSCREEN embed core attaches to that session as
// the viewer does; the texture readback is byte-exact against the shared CPU
// colormap reference (map_f32_to_rgba8 + colormap_lut, as test_embed.cpp).
// A forked child maps a segment by name (plain POSIX, no Metal) to prove a
// second process sees, and can write, the same pages.
//
// Compiled as ObjC++ with ARC (tests/CMakeLists.txt). GPU cases skip when the
// offscreen attach fails (headless CI): counted and reported as Skipped, a
// failure under CALIPER_REQUIRE_GPU=1 (gpu_skip.h), like the embed suite.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "gpu_skip.h"

#import <Metal/Metal.h>

#include <caliper/embed.h>
#include <caliper/live.h>
#include <caliper/tensor.h>

#include "tensor_bridge.h"   // caliper_host::map_f32_to_rgba8 / colormap_lut (CPU reference)
#include "live/live_shm.h"     // session-file helpers (fake a dead session's file)
#include "live/live_source.h"  // the viewer-side Source (claim words, deterministic)

#include <caliper/live_views.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// Every session file this binary writes goes under a private data root (never
// the user's ~/Library/Application Support/Caliper/live).
struct DataRoot {
    std::string dir;
    DataRoot() {
        char t[] = "/tmp/clp-live-tests-XXXXXX";
        dir = mkdtemp(t) ? t : "/tmp";
        setenv("CALIPER_DATA_DIR", dir.c_str(), 1);
    }
    ~DataRoot() {
        std::error_code ec;
        if (dir != "/tmp") std::filesystem::remove_all(dir, ec);
    }
} g_root;

std::string test_session(const char* tag) {
    return std::string("clpt") + tag + "-" + std::to_string((int)getpid());
}

CaliperLiveClient* open_client(const std::string& session) {
    CaliperLiveClientDesc d{};
    d.struct_size = sizeof d;
    d.session = session.c_str();
    return caliper_live_client_open(&d);
}

CaliperLiveViewDesc heatmap_desc(const char* name, int H, int W) {
    CaliperLiveViewDesc v{};
    v.struct_size = sizeof v;
    v.name = name;
    v.kind = CALIPER_LIVE_KIND_HEATMAP;
    v.dtype = CALIPER_DT_F32;
    v.ndim = 2;
    v.shape[0] = H;
    v.shape[1] = W;
    return v;
}

CaliperLiveSegmentInfo seg_info(CaliperLiveSegment* s) {
    CaliperLiveSegmentInfo i{};
    i.struct_size = sizeof i;
    REQUIRE(caliper_live_segment_info(s, &i) == 1);
    return i;
}

CaliperLivePublish pub(int32_t cm, uint32_t flags, float vmin, float vmax) {
    CaliperLivePublish p{};
    p.struct_size = sizeof p;
    p.stamp_ns = caliper_live_now_ns();
    p.colormap = cm;
    p.flags = flags;
    p.vmin = vmin;
    p.vmax = vmax;
    return p;
}

std::vector<float> ramp_f32(int W, int H, float scale = 1.0f, bool reversed = false) {
    std::vector<float> v((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float r = (float)x / (float)(W - 1);
            v[(size_t)y * W + x] = (reversed ? 1.0f - r : r) * scale;
        }
    return v;
}

std::vector<uint8_t> map_ref(const std::vector<float>& v, int W, int H, int32_t cm, float lo, float hi) {
    std::vector<uint8_t> out((size_t)W * H * 4);
    caliper_host::map_f32_to_rgba8(v.data(), W, H, caliper_host::colormap_lut(cm), lo, hi, out.data());
    return out;
}

std::vector<uint8_t> readback(CaliperCore* core, const char* name, int W, int H) {
    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    if (!caliper_core_view_readback(core, name, px.data(), px.size())) px.clear();
    return px;
}

CaliperViewInfo view_info(CaliperCore* core, const char* name) {
    CaliperViewInfo vi{};
    vi.struct_size = sizeof vi;
    REQUIRE(caliper_core_view_info(core, name, &vi) == 1);
    return vi;
}

// Poll until `want` views updated (the previous update's GPU read must retire
// first; the probe never blocks, so a fresh update can take a frame or two).
int poll_until(CaliperCore* core, int want, int max_tries = 200) {
    for (int i = 0; i < max_tries; ++i) {
        const int n = caliper_core_live_poll(core);
        if (n >= want) return n;
        caliper_core_frame(core);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 0;
}

// An offscreen core with its canvas attached, or null with *why set (no
// embeddable GPU, or CALIPER_TEST_FORCE_NO_GPU=1): the case then skips.
CaliperCore* offscreen_core(int W, int H, std::string* why) {
    CaliperCoreDesc d{};
    d.struct_size = sizeof d;
    d.renderer = CALIPER_RENDERER_DEFAULT;
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);
    CaliperCanvasDesc c{};
    c.struct_size = sizeof c;
    c.mode = CALIPER_CANVAS_OFFSCREEN;
    c.width = W;
    c.height = H;
    c.content_scale = 1.0f;
    if (caliper_test::force_no_gpu() || !caliper_core_attach_canvas(core, nullptr, &c)) {
        *why = caliper_core_last_error(core);
        caliper_core_shutdown(core);
        return nullptr;
    }
    return core;
}

}  // namespace

TEST_CASE("live/gates: client refusals are honest") {
    CHECK(caliper_live_client_open(nullptr) == nullptr);
    CaliperLiveClientDesc shortd{};
    shortd.struct_size = sizeof shortd - 1;
    CHECK(caliper_live_client_open(&shortd) == nullptr);
    CHECK(std::string(caliper_live_last_error()).find("short") != std::string::npos);

    CaliperLiveClientDesc bad{};
    bad.struct_size = sizeof bad;
    bad.session = "bad/name";
    CHECK(caliper_live_client_open(&bad) == nullptr);
    bad.session = "a-session-name-longer-than-twenty";
    CHECK(caliper_live_client_open(&bad) == nullptr);

    const std::string s = test_session("g");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    CHECK(std::string(caliper_live_client_session(c)) == s);
    // A second client cannot take a live session.
    CHECK(open_client(s) == nullptr);
    CHECK(std::string(caliper_live_last_error()).find("in use") != std::string::npos);

    CaliperLiveViewDesc v = heatmap_desc("w", 4, 4);
    v.struct_size = sizeof v - 1;
    CHECK(caliper_live_client_add_view(c, &v) == nullptr);
    v = heatmap_desc("w", 4, 4);
    v.dtype = CALIPER_DT_U8;
    CHECK(caliper_live_client_add_view(c, &v) == nullptr);   // heatmap must be f32
    v = heatmap_desc("pts", 4, 4);
    v.kind = CALIPER_LIVE_KIND_POINTS;
    CHECK(caliper_live_client_add_view(c, &v) == nullptr);   // points are (N,3), not (4,4)
    CHECK(std::string(caliper_live_last_error()).find("(N,3)") != std::string::npos);
    v = heatmap_desc("ser", 1, 1);
    v.kind = CALIPER_LIVE_KIND_SERIES;
    v.ndim = 1;
    v.shape[0] = 128;
    v.slots = 3;
    CHECK(caliper_live_client_add_view(c, &v) == nullptr);   // a series has ONE slot
    CHECK(std::string(caliper_live_last_error()).find("1 slot") != std::string::npos);
    v = heatmap_desc("aux", 4, 4);
    v.aux_count = 16;
    CHECK(caliper_live_client_add_view(c, &v) == nullptr);   // only points carry aux
    v = heatmap_desc("pts", 1, 1);
    v.kind = CALIPER_LIVE_KIND_POINTS;
    v.shape[0] = 8; v.shape[1] = 3;
    v.aux_count = 7;
    CHECK(caliper_live_client_add_view(c, &v) == nullptr);   // aux_count must be 0 or N
    CaliperLiveViewDesc img{};
    img.struct_size = sizeof img;
    img.name = "rgb";
    img.kind = CALIPER_LIVE_KIND_IMAGE;
    img.dtype = CALIPER_DT_U8;
    img.ndim = 3;
    img.shape[0] = 4; img.shape[1] = 4; img.shape[2] = 3;
    CHECK(caliper_live_client_add_view(c, &img) == nullptr); // RGBA only

    v = heatmap_desc("w", 4, 4);
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    CHECK(caliper_live_client_add_view(c, &v) == nullptr);   // duplicate name
    CaliperLivePublish p = pub(0, 0, 0.f, 1.f);
    CHECK(caliper_live_segment_publish(seg, 0, &p) == 0);    // not acquired
    caliper_live_client_close(c);
    // The name is gone after close: a fresh client may take it again.
    CaliperLiveClient* c2 = open_client(s);
    CHECK(c2 != nullptr);
    caliper_live_client_close(c2);
}

TEST_CASE("live/slots: acquire never hands out the published or the viewer's slot") {
    const std::string s = test_session("s");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    CaliperLiveViewDesc v = heatmap_desc("w", 8, 8);
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    CaliperLiveSegmentInfo info = seg_info(seg);
    auto* h = (CaliperLiveSegmentHeader*)info.host_ptr;
    CHECK(info.slot_count == 3);
    CHECK(h->magic == CALIPER_LIVE_SEG_MAGIC);
    CHECK(info.published_slot == CALIPER_LIVE_SLOT_NONE);
    CHECK((uintptr_t)info.host_ptr % (uintptr_t)getpagesize() == 0);
    for (int i = 0; i < 3; ++i) CHECK(info.slot_offset[i] % (uint64_t)getpagesize() == 0);

    int busy = 0;
    const int a = caliper_live_segment_acquire(seg, &busy);
    CHECK(busy == 0);
    CaliperLivePublish p = pub(0, 0, 0.f, 1.f);
    REQUIRE(caliper_live_segment_publish(seg, a, &p) == 1);
    CHECK(seg_info(seg).published_slot == (uint32_t)a);
    CHECK(seg_info(seg).seq == 1);

    // Simulate a viewer reading slot a while the client publishes b.
    __atomic_store_n(&h->viewer_reading_slot, (uint32_t)a, __ATOMIC_SEQ_CST);
    const int b = caliper_live_segment_acquire(seg, &busy);
    CHECK(b != a);
    REQUIRE(caliper_live_segment_publish(seg, b, &p) == 1);
    // P = b, R = a: the only free slot is the third one.
    const int third = 3 - a - b;
    CHECK(caliper_live_segment_acquire(seg, &busy) == third);
    CHECK(busy == 0);
    // It is now in flight (acquired, unpublished): no slot is left, so the
    // client must drop new snapshots instead of overwriting the pending one.
    for (int k = 0; k < 3; ++k) {
        CHECK(caliper_live_segment_acquire(seg, &busy) == -1);
        CHECK(busy == 1);
    }
    CHECK(seg_info(seg).dropped_client == 3);
    REQUIRE(caliper_live_segment_publish(seg, third, &p) == 1);
    CHECK(seg_info(seg).seq == 3);
    // P = third, R = a -> b is free again.
    CHECK(caliper_live_segment_acquire(seg, &busy) == b);
    // P == R frees two slots: with b in flight, the remaining one is handed out.
    __atomic_store_n(&h->viewer_reading_slot, (uint32_t)third, __ATOMIC_SEQ_CST);
    CHECK(caliper_live_segment_acquire(seg, &busy) == a);
    CHECK(caliper_live_segment_acquire(seg, &busy) == -1);
    caliper_live_segment_abandon(seg, a);
    caliper_live_segment_abandon(seg, b);
    CHECK(caliper_live_client_remove_view(c, seg) == 1);
    caliper_live_client_close(c);
}

TEST_CASE("live/shm: a second process sees and writes the same pages; GPU writes land too") {
    const std::string s = test_session("x");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    const int W = 64, H = 48;
    CaliperLiveViewDesc v = heatmap_desc("w", H, W);
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    CaliperLiveSegmentInfo info = seg_info(seg);
    std::vector<float> ramp = ramp_f32(W, H);
    auto* base = (uint8_t*)info.host_ptr;
    std::memcpy(base + info.slot_offset[1], ramp.data(), ramp.size() * 4);
    const std::string segname = info.segment_name;
    const uint64_t seglen = info.segment_bytes;
    const uint64_t off1 = info.slot_offset[1], st1 = info.stats_offset[1];

    // The child: POSIX only (no ObjC/Metal after fork).
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        int fd = shm_open(segname.c_str(), O_RDWR, 0);
        if (fd < 0) _exit(10);
        void* p = mmap(nullptr, (size_t)seglen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) _exit(11);
        const bool same = std::memcmp((uint8_t*)p + off1, ramp.data(), ramp.size() * 4) == 0;
        const uint32_t marker = 0xC0FFEE11u;
        std::memcpy((uint8_t*)p + st1 + 8, &marker, 4);
        _exit(same ? 0 : 12);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);   // child saw the ramp bytes
    uint32_t marker = 0;
    std::memcpy(&marker, base + st1 + 8, 4);
    CHECK(marker == 0xC0FFEE11u);      // and its write reached us

    // A GPU write through the client's MTLBuffer lands in the same pages.
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)info.mtl_buffer;
    if (buf == nil || caliper_test::force_no_gpu()) {
        CALIPER_NOTE_NO_GPU("no Metal device (the GPU-write half)");
    } else {
        CHECK((uint8_t*)buf.contents == base);
        id<MTLCommandQueue> q = [buf.device newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl fillBuffer:buf range:NSMakeRange((NSUInteger)info.slot_offset[2], 4096) value:0x5A];
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        bool all = true;
        for (int i = 0; i < 4096; ++i) all = all && base[info.slot_offset[2] + i] == 0x5A;
        CHECK(all);
    }
    caliper_live_client_close(c);
}

TEST_CASE("live/viewer: offscreen core attaches, heatmap byte-exact, second publish updates in place") {
    const int W = 64, H = 48;
    std::string no_gpu;
    CaliperCore* core = offscreen_core(W, H, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);

    // Viewer-side gates first.
    CHECK(caliper_core_live_attach(core, "no-such-sess") == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("no live session") != std::string::npos);
    CHECK(caliper_core_live_attach(core, "bad/name") == 0);
    CHECK(caliper_core_live_poll(core) == -1);
    CaliperLiveStatus shortst{};
    shortst.struct_size = sizeof shortst - 1;
    CHECK(caliper_core_live_status(core, &shortst, 0) == 0);

    const std::string s = test_session("v");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    CaliperLiveViewDesc v = heatmap_desc("ramp", H, W);
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    CaliperLiveSegmentInfo info = seg_info(seg);
    auto* base = (uint8_t*)info.host_ptr;

    // Publish 1: ramp in [0,1], fixed range, written through the CPU side.
    std::vector<float> ramp = ramp_f32(W, H);
    int sup = 0;   // busy flag
    const int s1 = caliper_live_segment_acquire(seg, &sup);
    std::memcpy(base + info.slot_offset[s1], ramp.data(), ramp.size() * 4);
    CaliperLivePublish p1 = pub(0, 0, 0.0f, 1.0f);
    REQUIRE(caliper_live_segment_publish(seg, s1, &p1) == 1);

    REQUIRE_MESSAGE(caliper_core_live_attach(core, s.c_str()) == 1, caliper_core_last_error(core));
    CHECK(caliper_core_live_attach(core, s.c_str()) == 0);   // already attached
    REQUIRE(poll_until(core, 1) == 1);
    caliper_core_frame(core);
    CHECK(caliper_core_view_count(core) == 1);
    CaliperViewInfo vi = view_info(core, "ramp");
    CHECK(std::string(vi.path) == "compute-imported");
    CHECK(vi.imported == 1);
    CHECK(vi.device == CALIPER_DEV_METAL);
    CHECK(vi.width == W);
    CHECK(vi.height == H);
    CHECK(vi.updates == 1);
    CHECK(vi.creates == 1);
    CHECK(readback(core, "ramp", W, H) == map_ref(ramp, W, H, 0, 0.0f, 1.0f));
    // The viewer claimed the published slot; the client can no longer get it.
    CHECK(seg_info(seg).viewer_reading_slot == (uint32_t)s1);
    CHECK(seg_info(seg).viewer_seq == 1);

    // Publish 2: reversed ramp scaled by 10, AUTOSCALED: the stats region holds
    // what the client's GPU aminmax would write.
    std::vector<float> rev = ramp_f32(W, H, 10.0f, /*reversed=*/true);
    const int s2 = caliper_live_segment_acquire(seg, &sup);
    CHECK(s2 != s1);
    std::memcpy(base + info.slot_offset[s2], rev.data(), rev.size() * 4);
    float stats[2] = {0.0f, 10.0f};
    std::memcpy(base + info.stats_offset[s2], stats, sizeof stats);
    CaliperLivePublish p2 = pub(1, CALIPER_LIVE_SNAP_AUTO_VMIN | CALIPER_LIVE_SNAP_AUTO_VMAX, 0.f, 0.f);
    REQUIRE(caliper_live_segment_publish(seg, s2, &p2) == 1);
    REQUIRE(poll_until(core, 1) == 1);
    vi = view_info(core, "ramp");
    CHECK(vi.updates == 2);
    CHECK(vi.creates == 1);   // in place: same texture
    CHECK(std::string(vi.path) == "compute-imported");
    CHECK(readback(core, "ramp", W, H) == map_ref(rev, W, H, 1, 0.0f, 10.0f));
    CHECK(seg_info(seg).viewer_reading_slot == (uint32_t)s2);

    // Nothing new published -> no update.
    for (int i = 0; i < 5; ++i) { CHECK(caliper_core_live_poll(core) == 0); caliper_core_frame(core); }

    // Three quick publishes between two polls: the viewer draws only the newest
    // and counts the two it skipped.
    for (int k = 0; k < 3; ++k) {
        const int sk = caliper_live_segment_acquire(seg, &sup);
        std::vector<float> fill((size_t)W * H, 0.25f * (float)(k + 1));
        std::memcpy(base + info.slot_offset[sk], fill.data(), fill.size() * 4);
        CaliperLivePublish pk = pub(0, 0, 0.0f, 1.0f);
        REQUIRE(caliper_live_segment_publish(seg, sk, &pk) == 1);
    }
    REQUIRE(poll_until(core, 1) == 1);
    std::vector<float> last((size_t)W * H, 0.75f);
    CHECK(readback(core, "ramp", W, H) == map_ref(last, W, H, 0, 0.0f, 1.0f));

    CaliperLiveStatus st{};
    st.struct_size = sizeof st;
    REQUIRE(caliper_core_live_status(core, &st, 0) == 1);
    CHECK(st.attached == 1);
    CHECK(st.client_alive == 1);
    CHECK(st.client_pid == (int32_t)getpid());
    CHECK(st.views == 1);
    CHECK(st.updates == 3);
    CHECK(st.skipped == 2);
    CHECK(st.latency_samples == 3);
    CHECK(st.latency_ms_median >= 0.0);
    CaliperLiveViewStatus vs{};
    vs.struct_size = sizeof vs;
    REQUIRE(caliper_core_live_view_status(core, 0, &vs) == 1);
    CHECK(std::string(vs.name) == "ramp");
    CHECK(vs.seq == 5);
    CHECK(std::string(vs.error).empty());

    // Remove the view on the client: the viewer drops its embed view.
    REQUIRE(caliper_live_client_remove_view(c, seg) == 1);
    caliper_core_live_poll(core);
    CHECK(caliper_core_view_count(core) == 0);
    REQUIRE(caliper_core_live_status(core, &st, 0) == 1);
    CHECK(st.views == 0);

    // Client close: the viewer notices and stays up.
    caliper_live_client_close(c);
    caliper_core_live_poll(core);
    caliper_core_frame(core);
    REQUIRE(caliper_core_live_status(core, &st, 0) == 1);
    CHECK(st.client_alive == 0);
    caliper_core_live_detach(core);
    CHECK(caliper_core_live_poll(core) == -1);
    caliper_core_shutdown(core);
}

TEST_CASE("live/viewer: (H,W,4) u8 image is blit-imported; shutdown with a live attach is clean") {
    const int W = 32, H = 16;
    std::string no_gpu;
    CaliperCore* core = offscreen_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    const std::string s = test_session("i");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    CaliperLiveViewDesc v{};
    v.struct_size = sizeof v;
    v.name = "img";
    v.kind = CALIPER_LIVE_KIND_IMAGE;
    v.dtype = CALIPER_DT_U8;
    v.ndim = 3;
    v.shape[0] = H; v.shape[1] = W; v.shape[2] = 4;
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    CaliperLiveSegmentInfo info = seg_info(seg);
    std::vector<uint8_t> px((size_t)W * H * 4);
    for (size_t i = 0; i < px.size(); ++i) px[i] = (uint8_t)(i * 7 + 3);
    int sup = 0;
    const int sl = caliper_live_segment_acquire(seg, &sup);
    std::memcpy((uint8_t*)info.host_ptr + info.slot_offset[sl], px.data(), px.size());
    CaliperLivePublish p = pub(0, 0, 0.f, 1.f);
    REQUIRE(caliper_live_segment_publish(seg, sl, &p) == 1);

    REQUIRE(caliper_core_live_attach(core, s.c_str()) == 1);
    REQUIRE(poll_until(core, 1) == 1);
    CaliperViewInfo vi = view_info(core, "img");
    CHECK(std::string(vi.path) == "blit-imported");
    CHECK(vi.imported == 1);
    CHECK(readback(core, "img", W, H) == px);
    caliper_core_frame(core);
    // Shutdown while attached: caliper_live_shutdown releases everything first.
    caliper_core_shutdown(core);
    caliper_live_client_close(c);
}

// ===========================================================================
// v2: session files, heartbeat, starting sessions, points / series / histogram
// ===========================================================================
namespace {

bool shm_exists(const std::string& name) {
    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) return false;
    close(fd);
    return true;
}

void make_shm(const std::string& name, uint64_t len) {
    const int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
    REQUIRE(fd >= 0);
    REQUIRE(ftruncate(fd, (off_t)len) == 0);
    close(fd);
}

std::string live_dir() { return g_root.dir + "/live"; }

// Sessions listed under the test root, keyed by token.
std::vector<CaliperLiveSessionInfo> list_all() {
    const int n = caliper_live_list_sessions(nullptr, 0);
    REQUIRE(n >= 0);
    std::vector<CaliperLiveSessionInfo> v((size_t)n + 4);
    for (auto& e : v) e.struct_size = sizeof e;
    const int m = caliper_live_list_sessions(v.data(), (int)v.size());
    REQUIRE(m >= 0);
    v.resize((size_t)std::min<int>(m, (int)v.size()));
    return v;
}

const CaliperLiveSessionInfo* find_token(const std::vector<CaliperLiveSessionInfo>& v, const std::string& tok) {
    for (const auto& e : v)
        if (tok == e.session) return &e;
    return nullptr;
}

int32_t dead_pid() {
    const pid_t p = fork();
    REQUIRE(p >= 0);
    if (p == 0) _exit(0);
    int st = 0;
    waitpid(p, &st, 0);   // reaped: the pid is gone
    return (int32_t)p;
}

// A mapping of a session's control segment, as a viewer would see it.
CaliperLiveControl* map_ctl(const std::string& session) {
    const int fd = shm_open(("/" + session + "-ctl").c_str(), O_RDWR, 0);
    REQUIRE(fd >= 0);
    void* p = mmap(nullptr, (size_t)caliper_live::ctl_bytes(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    REQUIRE(p != MAP_FAILED);
    return (CaliperLiveControl*)p;
}
void unmap_ctl(CaliperLiveControl* c) { munmap(c, (size_t)caliper_live::ctl_bytes()); }

CaliperLiveStatus core_status(CaliperCore* core) {
    CaliperLiveStatus st{};
    st.struct_size = sizeof st;
    REQUIRE(caliper_core_live_status(core, &st, 0) == 1);
    return st;
}

CaliperLiveViewStatus view_status(CaliperCore* core, int i) {
    CaliperLiveViewStatus vs{};
    vs.struct_size = sizeof vs;
    REQUIRE(caliper_core_live_view_status(core, i, &vs) == 1);
    return vs;
}

CaliperLivePublish pub2(int32_t cm, uint32_t flags, float vmin, float vmax, uint64_t series_total,
                        int32_t dims, int32_t labels, int32_t stats_count) {
    CaliperLivePublish p = pub(cm, flags, vmin, vmax);
    p.series_total = series_total;
    p.points_dims = dims;
    p.label_count = labels;
    p.stats_count = stats_count;
    return p;
}

}  // namespace

TEST_CASE("live/sessions: the session file is written, listed, rewritten per view and removed at close") {
    const std::string s = test_session("f");
    CaliperLiveClientDesc d{};
    d.struct_size = sizeof d;
    d.session = s.c_str();
    d.script = "unit_train.py";
    CaliperLiveClient* c = caliper_live_client_open(&d);
    REQUIRE(c != nullptr);
    const std::string file = caliper_live_client_session_file(c);
    CHECK(file == live_dir() + "/" + std::to_string((int)getpid()) + ".json");
    struct stat st{};
    REQUIRE(stat(file.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);
    caliper_live::SessionFile f;
    REQUIRE(caliper_live::read_session_file(file, &f));
    CHECK(f.protocol == (int32_t)CALIPER_LIVE_PROTOCOL_VERSION);
    CHECK(f.pid == (int32_t)getpid());
    CHECK(f.proc_start_us == caliper_live::proc_start_us((int32_t)getpid()));
    CHECK(f.proc_start_us != 0);
    CHECK(f.session == s);
    CHECK(f.script == "unit_train.py");
    CHECK(f.start_unix_ms > 0);
    CHECK(f.views.empty());
    REQUIRE(f.segments.size() == 1);
    CHECK(f.segments[0] == "/" + s + "-ctl");
    CHECK(caliper_live_client_attached(c) == 0);   // nobody polled yet

    CaliperLiveViewDesc v = heatmap_desc("fc1.weight", 8, 8);
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    REQUIRE(caliper_live::read_session_file(file, &f));
    REQUIRE(f.views.size() == 1);
    CHECK(f.views[0] == "fc1.weight");
    REQUIRE(f.segments.size() == 2);
    CHECK(f.segments[1] == std::string(seg_info(seg).segment_name));

    auto all = list_all();
    const CaliperLiveSessionInfo* e = find_token(all, s);
    REQUIRE(e != nullptr);
    CHECK(e->pid == (int32_t)getpid());
    CHECK(e->alive == 1);
    CHECK(e->compatible == 1);
    CHECK(e->view_count == 1);
    CHECK(e->viewer_pid == 0);
    CHECK(std::string(e->script) == "unit_train.py");
    CHECK(std::string(e->path) == file);
    CHECK(std::string(caliper_live_data_root()) == g_root.dir);

    // A second concurrent client of this process gets its own file.
    const std::string s2 = test_session("f2");
    CaliperLiveClient* c2 = open_client(s2);
    REQUIRE(c2 != nullptr);
    CHECK(std::string(caliper_live_client_session_file(c2)) ==
          live_dir() + "/" + std::to_string((int)getpid()) + "-" + s2 + ".json");
    CHECK(list_all().size() == 2);
    caliper_live_client_close(c2);

    REQUIRE(caliper_live_client_remove_view(c, seg) == 1);
    REQUIRE(caliper_live::read_session_file(file, &f));
    CHECK(f.views.empty());
    caliper_live_client_close(c);
    CHECK(access(file.c_str(), F_OK) != 0);
    CHECK(find_token(list_all(), s) == nullptr);
    CHECK_FALSE(shm_exists("/" + s + "-ctl"));

    // NO_SESSION_FILE: shm only.
    d.flags = CALIPER_LIVE_CLIENT_NO_SESSION_FILE;
    c = caliper_live_client_open(&d);
    REQUIRE(c != nullptr);
    CHECK(std::string(caliper_live_client_session_file(c)).empty());
    CHECK(find_token(list_all(), s) == nullptr);
    caliper_live_client_close(c);

    // Listing gates.
    CHECK(caliper_live_list_sessions(nullptr, 1) == -1);
    CaliperLiveSessionInfo shorti{};
    shorti.struct_size = sizeof shorti - 1;
    CHECK(caliper_live_list_sessions(&shorti, 1) == -1);
    CHECK(caliper_live_list_sessions_in("/nonexistent/caliper-root", nullptr, 0) == 0);
}

TEST_CASE("live/sessions: the lister reclaims a dead or reused pid's session, never a live one's names") {
    // A SIGKILLed trainer: its file and shm names stay behind.
    const int32_t dead = dead_pid();
    const std::string tok = test_session("d");
    make_shm("/" + tok + "-ctl", 4096);
    make_shm("/" + tok + "-v1", 4096);
    const std::string foreign = "/clptz-" + std::to_string((int)getpid());   // outside its prefix
    make_shm(foreign, 4096);
    caliper_live::SessionFile f;
    f.protocol = (int32_t)CALIPER_LIVE_PROTOCOL_VERSION;
    f.pid = dead;
    f.proc_start_us = 1;
    f.session = tok;
    f.script = "crashed.py";
    f.start_unix_ms = caliper_live::unix_ms();
    f.views = {"w"};
    f.segments = {"/" + tok + "-ctl", "/" + tok + "-v1", foreign};
    REQUIRE(caliper_live::ensure_dir(live_dir()));
    const std::string dead_file = live_dir() + "/" + std::to_string(dead) + ".json";
    REQUIRE(caliper_live::write_session_file(dead_file, f, nullptr));

    // A reused pid: our own (alive) pid, but another process start time.
    const std::string tok2 = test_session("r");
    make_shm("/" + tok2 + "-ctl", 4096);
    caliper_live::SessionFile g = f;
    g.pid = (int32_t)getpid();
    g.proc_start_us = caliper_live::proc_start_us(g.pid) + 12345;
    g.session = tok2;
    g.segments = {"/" + tok2 + "-ctl"};
    const std::string reused_file = live_dir() + "/" + std::to_string((int)getpid()) + "-reused.json";
    REQUIRE(caliper_live::write_session_file(reused_file, g, nullptr));
    // An interrupted atomic write of a dead writer.
    const std::string tmp_file = live_dir() + "/x.json.tmp." + std::to_string(dead);
    REQUIRE(caliper_live::write_session_file(tmp_file, f, nullptr));

    auto all = list_all();
    CHECK(find_token(all, tok) == nullptr);
    CHECK(find_token(all, tok2) == nullptr);
    CHECK(access(dead_file.c_str(), F_OK) != 0);
    CHECK(access(reused_file.c_str(), F_OK) != 0);
    CHECK(access(tmp_file.c_str(), F_OK) != 0);
    CHECK_FALSE(shm_exists("/" + tok + "-ctl"));
    CHECK_FALSE(shm_exists("/" + tok + "-v1"));
    CHECK_FALSE(shm_exists("/" + tok2 + "-ctl"));
    CHECK(shm_exists(foreign));   // not its name to unlink
    shm_unlink(foreign.c_str());

    // A stale file naming a token a LIVE client has since re-opened: only the
    // stale file goes; the live session keeps its names and stays listed.
    const std::string tok3 = test_session("l");
    CaliperLiveClient* c = open_client(tok3);
    REQUIRE(c != nullptr);
    caliper_live::SessionFile h = f;
    h.pid = dead_pid();
    h.session = tok3;
    h.segments = {"/" + tok3 + "-ctl"};
    const std::string stale3 = live_dir() + "/" + std::to_string(h.pid) + ".json";
    REQUIRE(caliper_live::write_session_file(stale3, h, nullptr));
    all = list_all();
    CHECK(access(stale3.c_str(), F_OK) != 0);
    CHECK(shm_exists("/" + tok3 + "-ctl"));
    const CaliperLiveSessionInfo* e = find_token(all, tok3);
    REQUIRE(e != nullptr);
    CHECK(e->pid == (int32_t)getpid());
    CHECK(e->alive == 1);
    caliper_live_client_close(c);
}

TEST_CASE("live/heartbeat: attached() follows the viewer's polls; staleness and detach clear it") {
    std::string no_gpu;
    CaliperCore* core = offscreen_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    const std::string s = test_session("h");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    CHECK(caliper_live_client_attached(nullptr) == 0);
    CHECK(caliper_live_client_attached(c) == 0);   // no heartbeat yet

    REQUIRE(caliper_core_live_attach(core, s.c_str()) == 1);
    CHECK(caliper_live_client_attached(c) == 0);   // attach alone writes no heartbeat
    CHECK(caliper_core_live_poll(core) == 0);
    CHECK(caliper_live_client_attached(c) == 1);
    int32_t vpid = 0;
    uint64_t age = 0;
    REQUIRE(caliper_live_client_viewer(c, &vpid, &age) == 1);
    CHECK(vpid == (int32_t)getpid());
    CHECK(age < CALIPER_LIVE_ATTACH_WINDOW_NS);

    // Stale: a heartbeat older than the window means "not attached".
    CaliperLiveControl* ctl = map_ctl(s);
    __atomic_store_n(&ctl->viewer_heartbeat_ns, caliper_live_now_ns() - 1500000000ull, __ATOMIC_RELEASE);
    CHECK(caliper_live_client_attached(c) == 0);
    caliper_core_live_poll(core);
    CHECK(caliper_live_client_attached(c) == 1);

    // Its cost (one atomic load + a clock read).
    const auto t0 = std::chrono::steady_clock::now();
    int sum = 0;
    for (int i = 0; i < 1000000; ++i) sum += caliper_live_client_attached(c);
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 1e6;
    MESSAGE("caliper_live_client_attached: " << ns << " ns per call");
    CHECK(sum == 1000000);
    CHECK(ns < 1000.0);

    // Detach clears the heartbeat at once.
    caliper_core_live_detach(core);
    CHECK(caliper_live_client_attached(c) == 0);
    CHECK(__atomic_load_n(&ctl->viewer_pid, __ATOMIC_ACQUIRE) == 0);

    // One viewer per session: another live process with a fresh heartbeat
    // holds it; a stale one does not.
    __atomic_store_n(&ctl->viewer_pid, (int32_t)getppid(), __ATOMIC_RELEASE);
    __atomic_store_n(&ctl->viewer_heartbeat_ns, caliper_live_now_ns(), __ATOMIC_RELEASE);
    CHECK(caliper_core_live_attach(core, s.c_str()) == 0);
    CHECK(std::string(caliper_core_last_error(core)).find("one viewer per session") != std::string::npos);
    __atomic_store_n(&ctl->viewer_heartbeat_ns, caliper_live_now_ns() - 1500000000ull, __ATOMIC_RELEASE);
    CHECK(caliper_core_live_attach(core, s.c_str()) == 1);
    caliper_core_live_poll(core);
    CHECK(__atomic_load_n(&ctl->viewer_pid, __ATOMIC_ACQUIRE) == (int32_t)getpid());

    // The client goes: the viewer keeps running and says so.
    caliper_live_client_close(c);
    caliper_core_live_poll(core);
    CHECK(core_status(core).client_alive == 0);
    CHECK(core_status(core).attached == 1);
    unmap_ctl(ctl);
    caliper_core_shutdown(core);
}

TEST_CASE("live/attach: tolerant of a starting session; attach by pid; picker lists") {
    const int W = 16, H = 8;
    std::string no_gpu;
    CaliperCore* core = offscreen_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);

    // (a) The control segment exists but is not initialised yet (magic 0).
    const std::string s = test_session("a");
    make_shm("/" + s + "-ctl", caliper_live::ctl_bytes());
    REQUIRE(caliper_core_live_attach(core, s.c_str()) == 1);
    CaliperLiveStatus st = core_status(core);
    CHECK(st.attached == 1);
    CHECK(st.pending == 1);
    CHECK(caliper_core_live_poll(core) == 0);
    // The client opens it (reclaiming the uninitialised name) and publishes.
    CaliperLiveClientDesc d{};
    d.struct_size = sizeof d;
    d.session = s.c_str();
    d.script = "starting.py";
    CaliperLiveClient* c = caliper_live_client_open(&d);
    REQUIRE(c != nullptr);
    CaliperLiveViewDesc v = heatmap_desc("w", H, W);
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    std::vector<float> ramp = ramp_f32(W, H);
    int busy = 0;
    const int sl = caliper_live_segment_acquire(seg, &busy);
    std::memcpy((uint8_t*)seg_info(seg).host_ptr + seg_info(seg).slot_offset[sl], ramp.data(), ramp.size() * 4);
    CaliperLivePublish p = pub(0, 0, 0.f, 1.f);
    REQUIRE(caliper_live_segment_publish(seg, sl, &p) == 1);
    REQUIRE(poll_until(core, 1) == 1);
    st = core_status(core);
    CHECK(st.pending == 0);
    CHECK(std::string(st.script) == "starting.py");
    CHECK(std::string(st.session) == s);
    CHECK(st.client_pid == (int32_t)getpid());
    CHECK(readback(core, "w", W, H) == map_ref(ramp, W, H, 0, 0.0f, 1.0f));
    CHECK(std::string(view_status(core, 0).apply) == "live_views");
    caliper_core_live_detach(core);
    caliper_live_client_close(c);

    // (b) Only a session file names the token so far (no control segment).
    const std::string s2 = test_session("b");
    caliper_live::SessionFile f;
    f.protocol = (int32_t)CALIPER_LIVE_PROTOCOL_VERSION;
    f.pid = (int32_t)getpid();
    f.proc_start_us = caliper_live::proc_start_us(f.pid);
    f.session = s2;
    f.script = "early.py";
    f.segments = {"/" + s2 + "-ctl"};
    REQUIRE(caliper_live::ensure_dir(live_dir()));
    const std::string early = live_dir() + "/" + std::to_string((int)getpid()) + "-early.json";
    REQUIRE(caliper_live::write_session_file(early, f, nullptr));
    REQUIRE(caliper_core_live_attach(core, s2.c_str()) == 1);
    CHECK(core_status(core).pending == 1);
    CHECK(std::string(core_status(core).script) == "early.py");
    unlink(early.c_str());
    CaliperLiveClient* c2 = open_client(s2);
    REQUIRE(c2 != nullptr);
    for (int i = 0; i < 100 && core_status(core).pending; ++i) {
        caliper_core_live_poll(core);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(core_status(core).pending == 0);
    CHECK(caliper_live_client_attached(c2) == 1);
    caliper_core_live_detach(core);

    // (c) Unknown token: refused. (d) Attach by pid finds our session file.
    CHECK(caliper_core_live_attach(core, "clpt-unknown") == 0);
    CHECK(caliper_core_live_attach_pid(core, dead_pid(), nullptr) == 0);
    REQUIRE(caliper_core_live_attach_pid(core, (int32_t)getpid(), nullptr) == 1);
    caliper_core_live_poll(core);
    CHECK(std::string(core_status(core).session) == s2);
    caliper_core_live_detach(core);

    // (e) The picker lists sessions while not attached; poll stays -1.
    REQUIRE(caliper_core_live_picker(core, 1, nullptr) == 1);
    CHECK(caliper_core_live_poll(core) == -1);
    st = core_status(core);
    CHECK(st.attached == 0);
    CHECK(st.picker == 1);
    CHECK(st.sessions_listed == 1);
    REQUIRE(caliper_core_live_attach(core, s2.c_str()) == 1);
    CHECK(caliper_core_live_poll(core) == 0);
    caliper_core_live_detach(core);   // the API detach ends the picker too
    CHECK(caliper_core_live_poll(core) == -1);
    CHECK(core_status(core).picker == 0);
    caliper_live_client_close(c2);
    caliper_core_shutdown(core);
}

TEST_CASE("live/points: (N,3) + labels through a segment; two claims until the superseded read retired") {
    std::string no_gpu;
    CaliperCore* core = offscreen_core(128, 128, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    const std::string s = test_session("p");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    const int N = 1 << 18;   // enough points that a frame stays on the GPU for a while
    CaliperLiveViewDesc v{};
    v.struct_size = sizeof v;
    v.name = "pts";
    v.kind = CALIPER_LIVE_KIND_POINTS;
    v.dtype = CALIPER_DT_F32;
    v.ndim = 2;
    v.shape[0] = N;
    v.shape[1] = 3;
    v.aux_count = N;
    CaliperLiveSegment* seg = caliper_live_client_add_view(c, &v);
    REQUIRE(seg != nullptr);
    CaliperLiveSegmentInfo info = seg_info(seg);
    CHECK(info.slot_count == 3);
    CHECK(info.kind == CALIPER_LIVE_KIND_POINTS);
    CHECK(info.aux_bytes == (uint64_t)N * 4);
    for (int i = 0; i < 3; ++i) {
        CHECK(info.aux_offset[i] >= info.slot_offset[i] + info.payload_bytes);
        CHECK(info.stats_offset[i] >= info.aux_offset[i] + info.aux_bytes);
        CHECK(info.slot_offset[i] % (uint64_t)getpagesize() == 0);
    }
    auto* base = (uint8_t*)info.host_ptr;
    auto write_slot = [&](int sl, float scale) {
        auto* pos = (float*)(base + info.slot_offset[sl]);
        auto* lab = (float*)(base + info.aux_offset[sl]);
        for (int i = 0; i < N; ++i) {
            const float t = (float)i / (float)N * 6.2831853f;
            pos[i * 3 + 0] = std::cos(t) * scale;
            pos[i * 3 + 1] = std::sin(t) * scale;
            pos[i * 3 + 2] = 0.0f;
            lab[i] = (float)(i % 4);
        }
        float* st = (float*)(base + info.stats_offset[sl]);
        const float bbox[8] = {0, 0, -scale, scale, -scale, scale, 0, 0};
        std::memcpy(st, bbox, sizeof bbox);
    };
    REQUIRE(caliper_core_live_attach(core, s.c_str()) == 1);
    int busy = 0;
    const int a = caliper_live_segment_acquire(seg, &busy);
    int a_slot = a;
    write_slot(a, 1.0f);
    CaliperLivePublish p = pub2(0, 0, 0.f, 1.f, 0, 2, 4, 8);
    REQUIRE(caliper_live_segment_publish(seg, a, &p) == 1);
    REQUIRE(poll_until(core, 1) == 1);
    CaliperLiveViewStatus vs = view_status(core, 0);
    if (std::string(vs.error).size()) MESSAGE("apply error: " << vs.error);
    REQUIRE(std::string(vs.error).empty());
    CHECK(vs.kind == CALIPER_LIVE_KIND_POINTS);
    CHECK(vs.points_dims == 2);
    CHECK(vs.label_count == 4);
    CHECK(vs.stats_count == 8);
    CHECK(vs.slot == a);
    CHECK(std::string(vs.apply) == "live_views");
    CaliperLiveViewInfo li{};
    li.struct_size = sizeof li;
    REQUIRE(caliper_core_live_view_info(core, "pts", &li) == 1);
    CHECK(li.count == N);
    CHECK(li.imported == 1);
    CHECK(seg_info(seg).viewer_reading_slot == (uint32_t)a);   // current: drawn every frame
    CHECK(seg_info(seg).viewer_prev_slot == CALIPER_LIVE_SLOT_NONE);

    // Client rule with two claims (deterministic, as a viewer would leave the
    // words): current b, superseded a, published b -> only the third is free,
    // and once it is in flight the client must drop.
    {
        auto* h = (CaliperLiveSegmentHeader*)base;
        const uint32_t R0 = __atomic_load_n(&h->viewer_reading_slot, __ATOMIC_SEQ_CST);
        const int b = caliper_live_segment_acquire(seg, &busy);
        REQUIRE(b >= 0);
        CHECK(b != a);
        write_slot(b, 0.5f);
        REQUIRE(caliper_live_segment_publish(seg, b, &p) == 1);
        // (simulate: the viewer claimed b and still protects a)
        __atomic_store_n(&h->viewer_prev_slot, (uint32_t)a, __ATOMIC_SEQ_CST);
        __atomic_store_n(&h->viewer_reading_slot, (uint32_t)b, __ATOMIC_SEQ_CST);
        const int third = 3 - a - b;
        CHECK(caliper_live_segment_acquire(seg, &busy) == third);
        CHECK(caliper_live_segment_acquire(seg, &busy) == -1);
        CHECK(busy == 1);
        caliper_live_segment_abandon(seg, third);
        uint32_t r = 0, pr = 0;
        REQUIRE(caliper_live_segment_claims(seg, &r, &pr) == 1);
        CHECK(r == (uint32_t)b);
        CHECK(pr == (uint32_t)a);
        // back to what the viewer really holds (it has not seen b yet)
        __atomic_store_n(&h->viewer_prev_slot, CALIPER_LIVE_SLOT_NONE, __ATOMIC_SEQ_CST);
        __atomic_store_n(&h->viewer_reading_slot, R0, __ATOMIC_SEQ_CST);
    }

    // The viewer half, deterministic: a second Source over the same session
    // (the claim words are what live_core.cpp drives). take() keeps the shown
    // slot as "previous" while claiming the newest; until release_prev() (the
    // view layer's read_done) the client must avoid both.
    {
        caliper_core_live_detach(core);   // one reader at a time
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        std::string err;
        caliper_live::OpenFail why;
        auto src = caliper_live::open_shm_source(s, (__bridge void*)dev, &err, &why);
        REQUIRE_MESSAGE(src != nullptr, err);
        std::vector<caliper_live::SourceView> views;
        REQUIRE(src->sync(&views));
        REQUIRE(views.size() == 1);
        const uint64_t key = views[0].key;
        caliper_live::Snapshot snap;
        REQUIRE(src->take(key, 0, &snap));   // claims the published one (b of the block above)
        const uint32_t cur = snap.slot;
        CHECK(snap.points_dims == 2);
        CHECK(snap.label_count == 4);
        CHECK(snap.aux_count == N);
        CHECK(snap.stats_count == 8);
        CHECK(snap.stats[3] == 0.5f);        // bbox ymax of the scale-0.5 ring
        const auto* pos = (const float*)((const uint8_t*)snap.host_base + snap.byte_offset);
        const auto* lab = (const float*)((const uint8_t*)snap.host_base + snap.aux_offset);
        CHECK(pos[0] == 0.5f);
        CHECK(lab[5] == 1.0f);
        src->release_prev(key);
        // the client publishes the next one into the free slot
        const int nxt = caliper_live_segment_acquire(seg, &busy);
        REQUIRE(nxt >= 0);
        CHECK(nxt != (int)cur);
        write_slot(nxt, 0.75f);
        REQUIRE(caliper_live_segment_publish(seg, nxt, &p) == 1);
        REQUIRE(src->take(key, snap.seq, &snap));
        CHECK(snap.slot == (uint32_t)nxt);
        uint32_t r = 0, pr = 0;
        caliper_live_segment_claims(seg, &r, &pr);
        CHECK(r == (uint32_t)nxt);            // current
        CHECK(pr == cur);                     // superseded, still protected
        const int third = 3 - (int)cur - nxt;
        CHECK(caliper_live_segment_acquire(seg, &busy) == third);
        CHECK(caliper_live_segment_acquire(seg, &busy) == -1);   // nothing else is free
        caliper_live_segment_abandon(seg, third);
        src->release_prev(key);               // read_done: the superseded read retired
        caliper_live_segment_claims(seg, &r, &pr);
        CHECK(pr == CALIPER_LIVE_SLOT_NONE);
        CHECK(caliper_live_segment_acquire(seg, &busy) >= 0);     // two slots free again
        caliper_live_segment_abandon(seg, 0);
        caliper_live_segment_abandon(seg, 1);
        caliper_live_segment_abandon(seg, 2);
        src.reset();                          // detaching releases both claims
        caliper_live_segment_claims(seg, &r, &pr);
        CHECK(r == CALIPER_LIVE_SLOT_NONE);
        CHECK(pr == CALIPER_LIVE_SLOT_NONE);
        // republish b's content as the newest for the end-to-end part
        const int again = caliper_live_segment_acquire(seg, &busy);
        write_slot(again, 0.5f);
        REQUIRE(caliper_live_segment_publish(seg, again, &p) == 1);
        REQUIRE(caliper_core_live_attach(core, s.c_str()) == 1);
        REQUIRE(poll_until(core, 1) == 1);
        a_slot = (int)seg_info(seg).viewer_reading_slot;
        const int b2 = caliper_live_segment_acquire(seg, &busy);
        write_slot(b2, 0.5f);
        REQUIRE(caliper_live_segment_publish(seg, b2, &p) == 1);
    }

    // End to end: draw a (a frame on the GPU), then take b right away. While
    // the frame that drew a may still run, the viewer holds BOTH slots; the
    // client's only free slot is the third. read_done -> a is released.
    caliper_core_frame(core);
    caliper_core_frame(core);
    REQUIRE(caliper_core_live_poll(core) == 1);   // takes b (published above)
    const CaliperLiveSegmentInfo after = seg_info(seg);
    const int b = (int)after.viewer_reading_slot;
    CHECK(b != a_slot);
    CHECK(view_status(core, 0).slot == b);
    if (after.viewer_prev_slot != CALIPER_LIVE_SLOT_NONE) {
        CHECK(after.viewer_prev_slot == (uint32_t)a_slot);
        const int third = 3 - a_slot - b;
        CHECK(caliper_live_segment_acquire(seg, &busy) == third);
        caliper_live_segment_abandon(seg, third);
        MESSAGE("two claims observed: current " << b << ", superseded " << a_slot << " (frame still in flight)");
    } else {
        MESSAGE("the frame that drew slot " << a_slot << " had already retired: released in the same poll");
    }
    for (int i = 0; i < 200 && seg_info(seg).viewer_prev_slot != CALIPER_LIVE_SLOT_NONE; ++i) {
        caliper_core_frame(core);
        caliper_core_live_poll(core);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(seg_info(seg).viewer_prev_slot == CALIPER_LIVE_SLOT_NONE);   // read_done released it
    CHECK(seg_info(seg).viewer_reading_slot == (uint32_t)b);           // the current one stays
    CHECK(caliper_core_live_view_read_done(core, "pts") == 1);
    caliper_core_frame(core);
    REQUIRE(caliper_core_live_view_info(core, "pts", &li) == 1);
    REQUIRE(li.width > 0);
    std::vector<uint8_t> px((size_t)li.width * li.height * 4);
    REQUIRE(caliper_core_live_view_readback(core, "pts", px.data(), px.size()) == 1);
    size_t lit = 0;
    for (size_t i = 0; i < px.size(); i += 4) lit += (px[i] | px[i + 1] | px[i + 2]) != 0;
    CHECK(lit > 0);   // the ring of points was drawn

    caliper_core_shutdown(core);
    caliper_live_client_close(c);
}

TEST_CASE("live/series+histogram: one never-claimed slot copied at apply; histogram range from stats") {
    std::string no_gpu;
    CaliperCore* core = offscreen_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    const std::string s = test_session("q");
    CaliperLiveClient* c = open_client(s);
    REQUIRE(c != nullptr);
    const int M = 16;
    CaliperLiveViewDesc v{};
    v.struct_size = sizeof v;
    v.name = "loss";
    v.kind = CALIPER_LIVE_KIND_SERIES;
    v.dtype = CALIPER_DT_F32;
    v.ndim = 1;
    v.shape[0] = M;
    CaliperLiveSegment* ser = caliper_live_client_add_view(c, &v);
    REQUIRE(ser != nullptr);
    CaliperLiveSegmentInfo si = seg_info(ser);
    CHECK(si.slot_count == 1);
    int busy = 0;
    CHECK(caliper_live_segment_acquire(ser, &busy) == 0);   // always slot 0
    CHECK(caliper_live_segment_acquire(ser, &busy) == 0);
    auto* ring = (float*)((uint8_t*)si.host_ptr + si.slot_offset[0]);
    // 20 values appended: the ring holds 4..19 at index i % 16.
    uint64_t total = 0;
    for (int i = 0; i < 20; ++i) ring[total++ % M] = 100.0f + (float)i;
    CaliperLivePublish p = pub2(0, 0, 0.f, 1.f, total, 0, 0, 2);
    REQUIRE(caliper_live_segment_publish_slot(ser, 0, &p) == 1);

    v.name = "hist";
    v.kind = CALIPER_LIVE_KIND_HISTOGRAM;
    v.shape[0] = 8;
    CaliperLiveSegment* hs = caliper_live_client_add_view(c, &v);
    REQUIRE(hs != nullptr);
    CaliperLiveSegmentInfo hi = seg_info(hs);
    const int hsl = caliper_live_segment_acquire(hs, &busy);
    auto* bins = (float*)((uint8_t*)hi.host_ptr + hi.slot_offset[hsl]);
    for (int i = 0; i < 8; ++i) bins[i] = (float)(i * i);
    float range[2] = {-2.0f, 2.0f};
    std::memcpy((uint8_t*)hi.host_ptr + hi.stats_offset[hsl], range, sizeof range);
    p = pub2(0, 0, 0.f, 1.f, 0, 0, 0, 2);
    REQUIRE(caliper_live_segment_publish(hs, hsl, &p) == 1);

    REQUIRE(caliper_core_live_attach(core, s.c_str()) == 1);
    REQUIRE(poll_until(core, 2) == 2);
    std::vector<float> got(M, -1.0f);
    size_t count = 0;
    REQUIRE(caliper_core_live_view_values(core, "loss", got.data(), got.size(), &count) == 1);
    REQUIRE(count == (size_t)M);
    for (int i = 0; i < M; ++i) CHECK(got[(size_t)i] == 104.0f + (float)i);   // oldest first
    CHECK(seg_info(ser).viewer_reading_slot == CALIPER_LIVE_SLOT_NONE);       // never claimed
    CHECK(seg_info(ser).viewer_prev_slot == CALIPER_LIVE_SLOT_NONE);
    const int si_idx = std::string(view_status(core, 0).name) == "loss" ? 0 : 1;
    CHECK(view_status(core, si_idx).series_total == 20);

    // Append 3 more while the viewer looks: only the new total matters.
    for (int i = 20; i < 23; ++i) ring[total++ % M] = 100.0f + (float)i;
    p = pub2(0, 0, 0.f, 1.f, total, 0, 0, 2);
    REQUIRE(caliper_live_segment_publish_slot(ser, 0, &p) == 1);
    REQUIRE(poll_until(core, 1) == 1);
    REQUIRE(caliper_core_live_view_values(core, "loss", got.data(), got.size(), &count) == 1);
    for (int i = 0; i < M; ++i) CHECK(got[(size_t)i] == 107.0f + (float)i);

    std::vector<float> hb(8, -1.0f);
    REQUIRE(caliper_core_live_view_values(core, "hist", hb.data(), hb.size(), &count) == 1);
    REQUIRE(count == 8);
    for (int i = 0; i < 8; ++i) CHECK(hb[(size_t)i] == (float)(i * i));
    CHECK(caliper_core_live_view_read_done(core, "hist") == 1);   // copied at apply

    // The script ends: the viewer says so and keeps what it received.
    caliper_live_client_close(c);
    for (int i = 0; i < 5; ++i) { caliper_core_live_poll(core); caliper_core_frame(core); }
    CHECK(core_status(core).client_alive == 0);
    CHECK(core_status(core).views == 2);
    CHECK(caliper_core_live_view_count(core) == 2);
    REQUIRE(caliper_core_live_view_values(core, "loss", got.data(), got.size(), &count) == 1);
    CHECK(got[0] == 107.0f);
    caliper_core_shutdown(core);
}

CALIPER_GPU_TEST_MAIN()
