// ===========================================================================
// examples/embed_host — Win32 (HWND) sibling of main.mm. Same five embed calls,
// a bare Win32 message loop instead of AppKit. Renderer DEFAULT resolves to
// Vulkan on Windows.
//
// STATUS: TRANSCRIPTION — NOT yet run on Windows hardware. It compiles only on
// _WIN32 (CMake gates it); the next Windows pass is its first live driver. No
// hardware/byte-exact claim is made here. The macOS half (main.mm) is the
// run-proven one; this mirrors its structure so the port is mechanical.
// ===========================================================================
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

#include <caliper/embed.h>

#include <cstdio>
#include <cstdlib>

static CaliperCore* g_core = nullptr;
static float        g_scale = 1.0f;   // physical px per logical px (DPI/96)

static void embed_log(void*, int level, const char* msg) {
    static const char* kTag[] = {"debug", "info", "warn", "error"};
    std::fprintf(stderr, "[embed-host] %s: %s\n",
                 kTag[(level >= 0 && level <= 3) ? level : 1], msg);
}
static void embed_crash(void*, const char* id, const char* fault) {
    std::fprintf(stderr, "[embed-host] applet '%s' faulted and was quarantined "
                         "(the host lives on): %s\n", id, fault);
}
static void send(CaliperInputEvent ev) { ev.struct_size = sizeof ev; caliper_core_event(g_core, &ev); }

// STEP 4 (input half): translate Win32 messages into CaliperInputEvent. Mouse
// coords arrive in physical px already (client area), so no scaling needed.
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_MOUSEMOVE: {
            CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_MOVE;
            ev.x = (float)GET_X_LPARAM(lp); ev.y = (float)GET_Y_LPARAM(lp);
            send(ev); return 0;
        }
        case WM_LBUTTONDOWN: case WM_LBUTTONUP:
        case WM_RBUTTONDOWN: case WM_RBUTTONUP:
        case WM_MBUTTONDOWN: case WM_MBUTTONUP: {
            CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_BUTTON;
            ev.button = (msg == WM_LBUTTONDOWN || msg == WM_LBUTTONUP) ? 0
                      : (msg == WM_RBUTTONDOWN || msg == WM_RBUTTONUP) ? 1 : 2;
            ev.down = (msg == WM_LBUTTONDOWN || msg == WM_RBUTTONDOWN || msg == WM_MBUTTONDOWN);
            send(ev); return 0;
        }
        case WM_MOUSEWHEEL: {
            CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_SCROLL;
            ev.dy = (float)GET_WHEEL_DELTA_WPARAM(wp) / (float)WHEEL_DELTA;
            send(ev); return 0;
        }
        case WM_CHAR: {
            CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_TEXT;
            ev.codepoint = (unsigned)wp; send(ev); return 0;
        }
        case WM_SIZE: {
            CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_RESIZE;
            ev.width = LOWORD(lp); ev.height = HIWORD(lp);
            send(ev); return 0;
        }
        case WM_DPICHANGED: {
            g_scale = (float)LOWORD(wp) / 96.0f;
            CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_CONTENT_SCALE;
            ev.scale = g_scale; send(ev); return 0;
        }
        case WM_SETFOCUS: case WM_KILLFOCUS: {
            CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_FOCUS;
            ev.focused = (msg == WM_SETFOCUS); send(ev); return 0;
        }
        case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

int main(int argc, char** argv) {
    const char* applet_id = (argc > 1) ? argv[1] : "dev.caliper.instance-scope";
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    WNDCLASSA wc{}; wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "CaliperEmbedHost"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassA(&wc);
    HWND hwnd = CreateWindowA(wc.lpszClassName, "caliper embed_host",
                              WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                              CW_USEDEFAULT, CW_USEDEFAULT, 1280, 800,
                              nullptr, nullptr, wc.hInstance, nullptr);
    g_scale = (float)GetDpiForWindow(hwnd) / 96.0f;

    RECT rc; GetClientRect(hwnd, &rc);

    // STEP 1: create the core (renderer DEFAULT -> Vulkan on Windows).
    CaliperCoreDesc desc{}; desc.struct_size = sizeof desc;
    desc.renderer = CALIPER_RENDERER_DEFAULT;
    desc.applets_dir = std::getenv("CALIPER_EMBED_APPLETS");
    desc.log_fn = &embed_log; desc.crash_fn = &embed_crash;
    g_core = caliper_core_create(&desc);
    if (!g_core) { std::fprintf(stderr, "[embed-host] core create failed\n"); return 1; }

    // STEP 2: attach the HWND as the canvas (physical px).
    CaliperCanvasDesc canvas{}; canvas.struct_size = sizeof canvas;
    canvas.mode = CALIPER_CANVAS_WINDOW;
    canvas.width = rc.right - rc.left; canvas.height = rc.bottom - rc.top;
    canvas.content_scale = g_scale;
    if (!caliper_core_attach_canvas(g_core, (void*)hwnd, &canvas)) {
        std::fprintf(stderr, "[embed-host] attach_canvas failed: %s\n",
                     caliper_core_last_error(g_core));
        return 1;
    }

    // STEP 3: load one applet by manifest id.
    if (!caliper_core_load_applet(g_core, applet_id)) {
        std::fprintf(stderr, "[embed-host] load_applet '%s' failed: %s\n",
                     applet_id, caliper_core_last_error(g_core));
        return 1;
    }

    // STEP 4: pump from OUR message loop — one frame per iteration, no vsync
    // wait inside the core.
    MSG m{};
    for (;;) {
        while (PeekMessage(&m, nullptr, 0, 0, PM_REMOVE)) {
            if (m.message == WM_QUIT) goto done;
            TranslateMessage(&m); DispatchMessage(&m);
        }
        caliper_core_frame(g_core);
    }
done:
    // STEP 5: tear down.
    caliper_core_shutdown(g_core);
    g_core = nullptr;
    return 0;
}
#endif  // _WIN32
