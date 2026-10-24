// caliper-viewer: the standalone live viewer process. An AppKit window hosting
// the embed core, which lists the live sessions of training processes
// (caliper/live.h session files), attaches to one and draws every view.
// A minimal host: the session list, attach/detach and the status window are
// libcaliper's (caliper_core_live_picker / _attach / the status window's
// Detach button), so the Caliper app's "Attach to process" can reuse them.
//
//   caliper-viewer [--session <token>] [--data-dir <root>] [--exit-after <s>]
//                  [--stats] [--latency-log <file>] [--size <w>x<h>]
//                  [--headless --dump-dir <dir>] [--probe-file <file>]
//
// Without --session the window shows the session list (refreshed every
// second) with an Attach button per session; with it, it attaches at once
// (tolerating a session that is still starting). Detach (status window)
// returns to the list. The title reads "caliper: <script> (<pid>)", plus
// "session ended" once the training process is gone (its last frames stay).
//
// The window is driven by a 60 Hz NSTimer in the common run-loop modes (so it
// keeps ticking through live resize and window drags): each tick polls the
// session (claims the newest completed snapshot of every view, writes the
// heartbeat) and renders one frame. Nothing here waits on the training
// process: the only cross-process traffic is lock-free reads/writes of shared
// sync words, so the window stays live whatever the training loop does.
//
// --headless: no window, an offscreen canvas (CI, tests). With --dump-dir, every
// time a view's seq changes its pixels (texture kinds: RGBA8 readback) or values
// (series/histogram: f32) are written to <dir>/<name>.rgba|.f32 and then
// <dir>/<name>.json ({seq, kind, width, height, count, path, imported}).
//
// Signals (automation): SIGTERM/SIGINT close the window gracefully (final stats
// line, core shutdown). SIGUSR1 posts synthetic input through this process's
// own AppKit event queue: with --probe-file, one line per action ("move x y",
// "click x y", in points from the content view's top-left; read at signal
// time); without it, a move+click on the live status window's title bar.
// SIGUSR2 posts a move to the window centre.
//
// --stats prints one line to stderr every 2 s: fps, per-view updates,
// step-to-screen latency median/p95/max over that window, skipped / client-
// dropped counters, each view's renderer path, and this process's RSS and
// physical footprint (for the no-host-copy evidence).
#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

#include <caliper/embed.h>
#include <caliper/live.h>
#include <caliper/live_views.h>

#include "live/live_activity.h"

#include <mach/mach.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cerrno>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static CaliperCore* g_core = nullptr;

struct Options {
    std::string session;
    std::string data_root;     // --data-dir, else the default resolved before the core exists
    double exit_after = 0.0;
    bool stats = false;
    std::string latency_log;   // --latency-log <file>: every sample, "t_ns lat_ms" per line
    int width = 1280, height = 800;
    bool headless = false;
    std::string dump_dir;
    std::string probe_file;
};
static Options g_opt;

static void viewer_log(void*, int level, const char* msg) {
    static const char* kTag[] = {"debug", "info", "warn", "error"};
    if (level < 1) return;
    std::fprintf(stderr, "[caliper-viewer] %s: %s\n", kTag[(level >= 0 && level <= 3) ? level : 1], msg);
}

static void rss_mb(double* rss, double* footprint) {
    mach_task_basic_info_data_t bi{};
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    *rss = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&bi, &n) == KERN_SUCCESS
               ? (double)bi.resident_size / 1048576.0 : -1.0;
    task_vm_info_data_t vi{};
    n = TASK_VM_INFO_COUNT;
    *footprint = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n) == KERN_SUCCESS
                     ? (double)vi.phys_footprint / 1048576.0 : -1.0;
}

static unsigned long long g_input_events = 0;   // reported in the stats line

static void send(CaliperInputEvent ev) {
    if (!g_core) return;
    ++g_input_events;
    ev.struct_size = sizeof ev;
    caliper_core_event(g_core, &ev);
}

// ---------------------------------------------------------------------------
// shared by both modes
// ---------------------------------------------------------------------------
static CaliperLiveStatus live_status(int reset = 0) {
    CaliperLiveStatus st{};
    st.struct_size = sizeof st;
    caliper_core_live_status(g_core, &st, reset);
    return st;
}

// Renderer path / imported of a view: the typed live view first, the v1.2
// view of the same name otherwise.
static void view_path(const char* name, std::string* path, int* imported) {
    CaliperLiveViewInfo li{};
    li.struct_size = sizeof li;
    if (caliper_core_live_view_info(g_core, name, &li) == 1 && li.path) {
        *path = li.path;
        *imported = li.imported;
        return;
    }
    CaliperViewInfo vi{};
    vi.struct_size = sizeof vi;
    if (caliper_core_view_info(g_core, name, &vi) == 1) {
        *path = vi.path;
        *imported = vi.imported;
        return;
    }
    *path = "-";
    *imported = 0;
}

static void print_stats(double t) {
    // Per-view counters first: the overall status call below resets the window.
    CaliperLiveStatus st = live_status(0);
    if (!st.attached) {
        std::fprintf(stderr, "[caliper-viewer] stats t=%.1f attached=0 picker=%d sessions=%d\n", t, st.picker,
                     st.sessions_listed);
        return;
    }
    std::string per;
    for (int i = 0; i < st.views; ++i) {
        CaliperLiveViewStatus vs{};
        vs.struct_size = sizeof vs;
        if (!caliper_core_live_view_status(g_core, i, &vs)) continue;
        std::string path = "-";
        int imported = 0;
        if (vs.apply && vs.apply[0]) view_path(vs.name, &path, &imported);   // applied at least once
        char buf[512];
        std::snprintf(buf, sizeof buf, " %s:upd=%llu,seq=%llu,skipped=%llu,dropped=%llu,path=%s,imported=%d%s%s",
                      vs.name, (unsigned long long)vs.window_updates, (unsigned long long)vs.seq,
                      (unsigned long long)vs.skipped, (unsigned long long)vs.client_dropped, path.c_str(),
                      imported, vs.error[0] ? ",error=" : "", vs.error);
        per += buf;
    }
    if (!g_opt.latency_log.empty()) {
        const size_t n = caliper_core_live_window_latencies(g_core, nullptr, nullptr, 0);
        std::vector<uint64_t> ts(n);
        std::vector<double> ms(n);
        caliper_core_live_window_latencies(g_core, ts.data(), ms.data(), n);
        if (FILE* f = std::fopen(g_opt.latency_log.c_str(), "a")) {
            for (size_t i = 0; i < n; ++i) std::fprintf(f, "%llu %.4f\n", (unsigned long long)ts[i], ms[i]);
            std::fclose(f);
        }
    }
    st = live_status(1);
    double rss = 0, fp = 0;
    rss_mb(&rss, &fp);
    std::fprintf(stderr,
                 "[caliper-viewer] stats t=%.1f fps=%.1f frames=%llu updates=%llu lat_ms_median=%.3f "
                 "lat_ms_p95=%.3f lat_ms_max=%.3f lat_n=%llu skipped=%llu client_dropped=%llu "
                 "errors=%llu client=%s rss_mb=%.1f footprint_mb=%.1f input_events=%llu views=%d pending=%d%s\n",
                 t, st.fps, (unsigned long long)st.window_frames, (unsigned long long)st.window_updates,
                 st.latency_ms_median, st.latency_ms_p95, st.latency_ms_max,
                 (unsigned long long)st.latency_samples, (unsigned long long)st.skipped,
                 (unsigned long long)st.client_dropped, (unsigned long long)st.errors,
                 st.client_alive ? "alive" : "exited", rss, fp, g_input_events, st.views, st.pending, per.c_str());
}

// --dump-dir: write each view whose seq changed since the last dump.
static std::map<std::string, uint64_t> g_dumped;

static bool write_atomic(const std::string& path, const void* data, size_t n) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data, 1, n, f) == n;
    std::fclose(f);
    return ok && std::rename(tmp.c_str(), path.c_str()) == 0;
}

static void dump_views() {
    if (g_opt.dump_dir.empty()) return;
    const CaliperLiveStatus st = live_status(0);
    if (!st.attached) return;
    for (int i = 0; i < st.views; ++i) {
        CaliperLiveViewStatus vs{};
        vs.struct_size = sizeof vs;
        if (!caliper_core_live_view_status(g_core, i, &vs) || vs.seq == 0) continue;
        const std::string name = vs.name;
        if (g_dumped[name] == vs.seq) continue;
        CaliperLiveViewInfo li{};
        li.struct_size = sizeof li;
        if (caliper_core_live_view_info(g_core, name.c_str(), &li) != 1) continue;
        const std::string base = g_opt.dump_dir + "/" + name;
        bool ok = false;
        const char* ext = "";
        if (vs.kind == CALIPER_LIVE_KIND_SERIES || vs.kind == CALIPER_LIVE_KIND_HISTOGRAM) {
            size_t count = 0;
            caliper_core_live_view_values(g_core, name.c_str(), nullptr, 0, &count);
            std::vector<float> v(count);
            ok = caliper_core_live_view_values(g_core, name.c_str(), v.data(), v.size(), &count) == 1 &&
                 write_atomic(base + ".f32", v.data(), count * sizeof(float));
            ext = "f32";
        } else if (li.width > 0 && li.height > 0) {
            std::vector<uint8_t> px((size_t)li.width * li.height * 4);
            ok = caliper_core_live_view_readback(g_core, name.c_str(), px.data(), px.size()) == 1 &&
                 write_atomic(base + ".rgba", px.data(), px.size());
            ext = "rgba";
        }
        if (!ok) continue;
        std::string path;
        int imported = 0;
        view_path(name.c_str(), &path, &imported);
        char js[512];
        const int n = std::snprintf(js, sizeof js,
                                    "{\"seq\": %llu, \"kind\": %u, \"width\": %d, \"height\": %d, \"count\": %lld, "
                                    "\"path\": \"%s\", \"imported\": %d, \"file\": \"%s.%s\", \"apply\": \"%s\"}\n",
                                    (unsigned long long)vs.seq, vs.kind, li.width, li.height, (long long)li.count,
                                    path.c_str(), imported, name.c_str(), ext, vs.apply);
        if (write_atomic(base + ".json", js, (size_t)n)) g_dumped[name] = vs.seq;
    }
}

// Exit 2 before any window when --session names nothing: no control segment
// and no session file for that token.
static bool session_known(const std::string& token) {
    const std::string ctl = "/" + token + "-ctl";
    const int fd = shm_open(ctl.c_str(), O_RDONLY, 0);
    if (fd >= 0) { close(fd); return true; }
    const int n = caliper_live_list_sessions_in(g_opt.data_root.c_str(), nullptr, 0);
    if (n <= 0) return false;
    std::vector<CaliperLiveSessionInfo> v((size_t)n + 4);
    for (auto& e : v) e.struct_size = sizeof e;
    const int m = caliper_live_list_sessions_in(g_opt.data_root.c_str(), v.data(), (int)v.size());
    for (int i = 0; i < m && i < (int)v.size(); ++i)
        if (token == v[(size_t)i].session) return true;
    return false;
}

// The viewer's private data root: /tmp/caliper-viewer-<pid>-XXXXXX. A viewer
// that was SIGKILLed never removed its own, so each start removes the roots of
// viewers that are gone (pid in the name, not running).
static std::string make_private_root() {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/tmp", ec)) {
        const std::string n = e.path().filename().string();
        static const std::string kPre = "caliper-viewer-";
        if (n.compare(0, kPre.size(), kPre) != 0) continue;
        const int pid = std::atoi(n.c_str() + kPre.size());
        if (pid <= 0 || pid == (int)getpid()) continue;
        if (kill(pid, 0) == 0 || errno == EPERM) continue;
        std::error_code rm;
        std::filesystem::remove_all(e.path(), rm);
    }
    std::string t = "/tmp/caliper-viewer-" + std::to_string((int)getpid()) + "-XXXXXX";
    std::vector<char> buf(t.begin(), t.end());
    buf.push_back('\0');
    return mkdtemp(buf.data()) ? std::string(buf.data()) : std::string();
}

static CaliperCore* make_core(const std::string& data_dir) {
    CaliperCoreDesc desc{};
    desc.struct_size = sizeof desc;
    desc.renderer = CALIPER_RENDERER_DEFAULT;
    desc.data_dir = data_dir.empty() ? nullptr : data_dir.c_str();
    desc.log_fn = &viewer_log;
    return caliper_core_create(&desc);
}

// ---------------------------------------------------------------------------
// window mode
// ---------------------------------------------------------------------------
static void mouse_xy(NSView* v, NSEvent* e, float* x, float* y) {
    NSPoint p = [v convertPoint:e.locationInWindow fromView:nil];
    CGFloat s = v.window.backingScaleFactor;
    *x = (float)(p.x * s);
    *y = (float)((v.bounds.size.height - p.y) * s);
}

// The content view: the core replaces its layer with a CAMetalLayer at attach;
// it only translates Cocoa input into CaliperInputEvent (as embed_host does).
@interface CanvasView : NSView
@end

@implementation CanvasView
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)acceptsFirstMouse:(NSEvent*)e { return YES; }
- (BOOL)wantsUpdateLayer { return YES; }
- (void)updateTrackingAreas {
    for (NSTrackingArea* a in self.trackingAreas) [self removeTrackingArea:a];
    NSTrackingArea* ta = [[NSTrackingArea alloc]
        initWithRect:self.bounds
             options:NSTrackingMouseMoved | NSTrackingActiveAlways | NSTrackingInVisibleRect
               owner:self userInfo:nil];
    [self addTrackingArea:ta];
}
- (void)mouseMoved:(NSEvent*)e { [self move:e]; }
- (void)mouseDragged:(NSEvent*)e { [self move:e]; }
- (void)rightMouseDragged:(NSEvent*)e { [self move:e]; }
- (void)otherMouseDragged:(NSEvent*)e { [self move:e]; }
- (void)move:(NSEvent*)e {
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_MOVE;
    mouse_xy(self, e, &ev.x, &ev.y);
    send(ev);
}
- (void)button:(int)b down:(int)d event:(NSEvent*)e {
    [self move:e];
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_BUTTON;
    ev.button = b; ev.down = d;
    send(ev);
}
- (void)mouseDown:(NSEvent*)e { [self button:0 down:1 event:e]; }
- (void)mouseUp:(NSEvent*)e { [self button:0 down:0 event:e]; }
- (void)rightMouseDown:(NSEvent*)e { [self button:1 down:1 event:e]; }
- (void)rightMouseUp:(NSEvent*)e { [self button:1 down:0 event:e]; }
- (void)otherMouseDown:(NSEvent*)e { [self button:2 down:1 event:e]; }
- (void)otherMouseUp:(NSEvent*)e { [self button:2 down:0 event:e]; }
- (void)scrollWheel:(NSEvent*)e {
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_SCROLL;
    ev.dx = (float)e.scrollingDeltaX * 0.1f;
    ev.dy = (float)e.scrollingDeltaY * 0.1f;
    send(ev);
}
- (void)keyDown:(NSEvent*)e {
    for (NSUInteger i = 0; i < e.characters.length; ++i) {
        CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_TEXT;
        ev.codepoint = [e.characters characterAtIndex:i];
        send(ev);
    }
}
@end

@interface ViewerDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, strong) NSWindow* window;
@property(nonatomic, strong) CanvasView* view;
@property(nonatomic, strong) NSTimer* pump;
@property(nonatomic, assign) void* activity;   // caliper_live::begin_viewer_activity
@property(nonatomic, assign) double t0;
@property(nonatomic, assign) double lastStats;
@end

@implementation ViewerDelegate {
    std::string _dataDir;
    std::string _title;
    dispatch_source_t _sig[4];
}

// Post a synthetic mouse event at logical (x, y) from the content view's TOP-
// left, through NSApp's event queue: the path a real mouse event takes once
// AppKit has turned the HID event into an NSEvent (sendEvent -> window -> view
// -> caliper_core_event -> ImGui on the next frame).
- (void)postMouse:(NSEventType)type x:(CGFloat)x y:(CGFloat)y {
    NSSize sz = self.view.bounds.size;
    NSPoint p = NSMakePoint(x, sz.height - y);   // window coords (content view == window)
    NSEvent* e = [NSEvent mouseEventWithType:type location:p modifierFlags:0
                                   timestamp:[NSProcessInfo processInfo].systemUptime
                                windowNumber:self.window.windowNumber context:nil
                                 eventNumber:0 clickCount:(type == NSEventTypeMouseMoved ? 0 : 1)
                                    pressure:(type == NSEventTypeLeftMouseDown ? 1.0f : 0.0f)];
    [NSApp postEvent:e atStart:NO];
}

// SIGUSR1: the actions of --probe-file, else a click on the live status
// window's title bar (bottom-right corner, 520 x 250 pt, see live_core.cpp).
// SIGUSR2: move to the window centre.
- (void)probe:(int)which {
    NSSize sz = self.view.bounds.size;
    if (which == 1 && !g_opt.probe_file.empty()) {
        std::ifstream in(g_opt.probe_file);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string op;
            double x = 0, y = 0;
            if (!(ls >> op >> x >> y)) continue;
            [self postMouse:NSEventTypeMouseMoved x:x y:y];
            if (op == "click") {
                [self postMouse:NSEventTypeLeftMouseDown x:x y:y];
                [self postMouse:NSEventTypeLeftMouseUp x:x y:y];
            }
            std::fprintf(stderr, "[caliper-viewer] input-probe: %s at (%.0f, %.0f) pt\n", op.c_str(), x, y);
        }
        return;
    }
    if (which == 1) {
        const CGFloat x = sz.width - 8 - 260, y = sz.height - 8 - 250 + 9;   // status title bar
        [self postMouse:NSEventTypeMouseMoved x:x y:y];
        [self postMouse:NSEventTypeLeftMouseDown x:x y:y];
        [self postMouse:NSEventTypeLeftMouseUp x:x y:y];
        std::fprintf(stderr, "[caliper-viewer] input-probe: posted move+click at (%.0f, %.0f) pt (status title bar), "
                     "input_events so far %llu\n", x, y, g_input_events);
    } else {
        [self postMouse:NSEventTypeMouseMoved x:sz.width * 0.5 y:sz.height * 0.5];
        std::fprintf(stderr, "[caliper-viewer] input-probe: posted move to the window centre, input_events so far %llu\n",
                     g_input_events);
    }
}

- (void)installSignals {
    const int sigs[4] = {SIGTERM, SIGINT, SIGUSR1, SIGUSR2};
    for (int i = 0; i < 4; ++i) {
        signal(sigs[i], SIG_IGN);
        _sig[i] = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, (uintptr_t)sigs[i], 0,
                                         dispatch_get_main_queue());
        const int sg = sigs[i];
        __weak ViewerDelegate* weakSelf = self;
        dispatch_source_set_event_handler(_sig[i], ^{
            ViewerDelegate* me = weakSelf;
            if (!me) return;
            if (sg == SIGUSR1) [me probe:1];
            else if (sg == SIGUSR2) [me probe:2];
            else [me.window performClose:nil];
        });
        dispatch_resume(_sig[i]);
    }
}

- (void)pushResizeAndScale {
    if (!g_core) return;
    CGFloat s = self.window.backingScaleFactor;
    NSSize pts = self.view.bounds.size;
    CaliperInputEvent sc{}; sc.type = CALIPER_EVENT_CONTENT_SCALE; sc.scale = (float)s;
    send(sc);
    CaliperInputEvent rz{}; rz.type = CALIPER_EVENT_RESIZE;
    rz.width = (int)(pts.width * s); rz.height = (int)(pts.height * s);
    send(rz);
}
- (void)windowDidResize:(NSNotification*)n { [self pushResizeAndScale]; }
- (void)windowDidChangeBackingProperties:(NSNotification*)n { [self pushResizeAndScale]; }
- (void)windowDidBecomeKey:(NSNotification*)n {
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_FOCUS; ev.focused = 1; send(ev);
}
- (void)windowDidResignKey:(NSNotification*)n {
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_FOCUS; ev.focused = 0; send(ev);
}

// "caliper: <script> (<pid>)" while attached (+ "session ended"), the list
// title otherwise. Set only when it changes.
- (void)updateTitle {
    const CaliperLiveStatus st = live_status(0);
    char b[256];
    if (st.attached) {
        std::snprintf(b, sizeof b, "caliper: %s (%d)%s", st.script && st.script[0] ? st.script : st.session,
                      st.client_pid, !st.client_alive ? " - session ended" : (st.pending ? " - starting" : ""));
    } else {
        std::snprintf(b, sizeof b, "caliper: live sessions");
    }
    if (_title != b) {
        _title = b;
        [self.window setTitle:[NSString stringWithUTF8String:b]];
    }
}

- (void)tick:(NSTimer*)t {
    if (!g_core) return;
    @autoreleasepool {
        caliper_core_live_poll(g_core);
        caliper_core_frame(g_core);
        [self updateTitle];
        dump_views();
    }
    if (g_opt.stats) {
        const double now = CACurrentMediaTime();
        if (now - self.lastStats >= 2.0) {
            self.lastStats = now;
            print_stats(now - self.t0);
        }
    }
}

- (void)applicationDidFinishLaunching:(NSNotification*)n {
    const CGFloat W = g_opt.width, H = g_opt.height;
    self.window = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, W, H)
                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                            NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
                    backing:NSBackingStoreBuffered defer:NO];
    [self.window setTitle:@"caliper"];
    [self.window setDelegate:self];
    [self.window setAcceptsMouseMovedEvents:YES];
    [self.window setReleasedWhenClosed:NO];
    self.view = [[CanvasView alloc] initWithFrame:NSMakeRect(0, 0, W, H)];
    [self.window setContentView:self.view];
    [self.window makeFirstResponder:self.view];
    [self.window center];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    // A hidden or occluded window must keep its 60 Hz tick (live_activity.h:
    // App Nap would let the heartbeat go stale and the client would stop
    // snapshotting while the user is still attached). The Caliper app's live
    // page holds the same activity.
    self.activity = caliper_live::begin_viewer_activity("caliper live session viewer");

    // A private data root: the viewer persists nothing, and the default root
    // would share DuckDB stores with a running Caliper app. (The session
    // files' root was resolved before this override exists: g_opt.data_root.)
    _dataDir = make_private_root();

    g_core = make_core(_dataDir);
    if (!g_core) { std::fprintf(stderr, "[caliper-viewer] core create failed\n"); std::exit(3); }

    const CGFloat scale = self.window.backingScaleFactor;
    CaliperCanvasDesc canvas{};
    canvas.struct_size = sizeof canvas;
    canvas.mode = CALIPER_CANVAS_WINDOW;
    canvas.width = (int)(W * scale);
    canvas.height = (int)(H * scale);
    canvas.content_scale = (float)scale;
    if (!caliper_core_attach_canvas(g_core, (__bridge void*)self.view, &canvas)) {
        std::fprintf(stderr, "[caliper-viewer] attach_canvas failed: %s\n", caliper_core_last_error(g_core));
        std::exit(3);
    }
    // The session list is always there to come back to (Detach returns to it).
    caliper_core_live_picker(g_core, 1, g_opt.data_root.c_str());
    if (!g_opt.session.empty() && !caliper_core_live_attach(g_core, g_opt.session.c_str())) {
        std::fprintf(stderr, "[caliper-viewer] %s\n", caliper_core_last_error(g_core));
        caliper_core_shutdown(g_core);
        g_core = nullptr;
        std::exit(2);
    }
    [self updateTitle];
    // The CGWindowID, for screenshots / automation (screencapture -l <id>).
    std::fprintf(stderr, "[caliper-viewer] ready pid=%d window=%ld session=%s data_root=%s\n", (int)getpid(),
                 (long)self.window.windowNumber, g_opt.session.empty() ? "-" : g_opt.session.c_str(),
                 g_opt.data_root.c_str());

    [self installSignals];
    self.t0 = self.lastStats = CACurrentMediaTime();
    self.pump = [NSTimer timerWithTimeInterval:1.0 / 60.0 target:self selector:@selector(tick:)
                                      userInfo:nil repeats:YES];
    [[NSRunLoop currentRunLoop] addTimer:self.pump forMode:NSRunLoopCommonModes];
    if (g_opt.exit_after > 0)
        [self.window performSelector:@selector(performClose:) withObject:nil afterDelay:g_opt.exit_after];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)a { return YES; }

- (void)windowWillClose:(NSNotification*)n {
    [self.pump invalidate];
    self.pump = nil;
    if (g_core) {
        if (g_opt.stats) print_stats(CACurrentMediaTime() - self.t0);
        caliper_core_shutdown(g_core);   // detaches the live session (unmaps, never unlinks)
        g_core = nullptr;
    }
    if (!_dataDir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(_dataDir, ec);
    }
    caliper_live::end_viewer_activity(self.activity);
    self.activity = nullptr;
}
@end

// ---------------------------------------------------------------------------
// headless mode
// ---------------------------------------------------------------------------
static volatile sig_atomic_t g_stop = 0;
static void on_stop(int) { g_stop = 1; }

static int run_headless() {
    signal(SIGTERM, on_stop);
    signal(SIGINT, on_stop);
    const std::string dir = make_private_root();
    g_core = make_core(dir);
    if (!g_core) { std::fprintf(stderr, "[caliper-viewer] core create failed\n"); return 3; }
    CaliperCanvasDesc canvas{};
    canvas.struct_size = sizeof canvas;
    canvas.mode = CALIPER_CANVAS_OFFSCREEN;
    canvas.width = g_opt.width;
    canvas.height = g_opt.height;
    canvas.content_scale = 1.0f;
    int rc = 0;
    if (!caliper_core_attach_canvas(g_core, nullptr, &canvas)) {
        std::fprintf(stderr, "[caliper-viewer] no GPU for an offscreen canvas: %s\n", caliper_core_last_error(g_core));
        rc = 3;
    } else if (!caliper_core_live_attach(g_core, g_opt.session.c_str())) {
        std::fprintf(stderr, "[caliper-viewer] %s\n", caliper_core_last_error(g_core));
        rc = 2;
    } else {
        std::fprintf(stderr, "[caliper-viewer] ready pid=%d headless session=%s\n", (int)getpid(),
                     g_opt.session.c_str());
        const double t0 = CACurrentMediaTime();
        double last_stats = t0;
        while (!g_stop) {
            @autoreleasepool {
                caliper_core_live_poll(g_core);
                caliper_core_frame(g_core);
                dump_views();
            }
            const double now = CACurrentMediaTime();
            if (g_opt.stats && now - last_stats >= 2.0) {
                last_stats = now;
                print_stats(now - t0);
            }
            if (g_opt.exit_after > 0 && now - t0 >= g_opt.exit_after) break;
            usleep(16000);
        }
        if (g_opt.stats) print_stats(CACurrentMediaTime() - t0);
    }
    caliper_core_shutdown(g_core);
    g_core = nullptr;
    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    return rc;
}

static void usage() {
    std::fprintf(stderr, "usage: caliper-viewer [--session <token>] [--data-dir <root>] [--exit-after <seconds>] "
                         "[--stats] [--size <w>x<h>] [--latency-log <file>] [--headless --dump-dir <dir>] "
                         "[--probe-file <file>]\n");
}

int main(int argc, const char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (!std::strcmp(a, "--session")) {
            const char* v = next(); if (!v) { usage(); return 64; } g_opt.session = v;
        } else if (!std::strcmp(a, "--data-dir")) {
            const char* v = next(); if (!v) { usage(); return 64; } g_opt.data_root = v;
        } else if (!std::strcmp(a, "--exit-after")) {
            const char* v = next(); if (!v) { usage(); return 64; } g_opt.exit_after = std::atof(v);
        } else if (!std::strcmp(a, "--stats")) {
            g_opt.stats = true;
        } else if (!std::strcmp(a, "--latency-log")) {
            const char* v = next(); if (!v) { usage(); return 64; } g_opt.latency_log = v; g_opt.stats = true;
        } else if (!std::strcmp(a, "--size")) {
            const char* v = next();
            if (!v || std::sscanf(v, "%dx%d", &g_opt.width, &g_opt.height) != 2) { usage(); return 64; }
        } else if (!std::strcmp(a, "--headless")) {
            g_opt.headless = true;
        } else if (!std::strcmp(a, "--dump-dir")) {
            const char* v = next(); if (!v) { usage(); return 64; } g_opt.dump_dir = v;
        } else if (!std::strcmp(a, "--probe-file")) {
            const char* v = next(); if (!v) { usage(); return 64; } g_opt.probe_file = v;
        } else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
            usage(); return 0;
        } else {
            std::fprintf(stderr, "caliper-viewer: unknown argument '%s'\n", a);
            usage();
            return 64;
        }
    }
    // Resolve the session-file root now: the core's private data_dir override
    // (below) would otherwise redirect caliper::app_data_dir().
    if (g_opt.data_root.empty()) g_opt.data_root = caliper_live_data_root();
    setenv("CALIPER_DATA_DIR", g_opt.data_root.c_str(), 1);   // every later default lookup agrees
    if (g_opt.headless && g_opt.session.empty()) {
        std::fprintf(stderr, "caliper-viewer: --headless needs --session\n");
        return 64;
    }
    if (!g_opt.session.empty() && !session_known(g_opt.session)) {
        // Fail before any window appears when there is no such session.
        std::fprintf(stderr, "[caliper-viewer] no live session '%s' (no control segment, no session file under %s/live)\n",
                     g_opt.session.c_str(), g_opt.data_root.c_str());
        return 2;
    }
    if (g_opt.headless) return run_headless();
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        ViewerDelegate* d = [ViewerDelegate new];
        [NSApp setDelegate:d];
        [NSApp run];
    }
    return 0;
}
