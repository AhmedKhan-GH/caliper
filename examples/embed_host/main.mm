// ===========================================================================
// examples/embed_host — a SECOND embedder of libcaliper, on a bare AppKit
// window (no GLFW, no ImGui link, no renderer type in sight). The whole point:
// prove the embed C ABI (caliper/embed.h) from OUTSIDE the caliper exe's
// assumptions. Any toolkit that owns a native view can host the applet canvas;
// here that toolkit is raw Cocoa.
//
// EMBEDDING IS FIVE CALLS (grep for STEP 1..5 below):
//   1. caliper_core_create      — spin up the core (renderer + services + loader)
//   2. caliper_core_attach_canvas — hand it the NSView it should paint
//   3. caliper_core_load_applet  — launch one applet by manifest id
//   4. caliper_core_frame        — pump ONE frame from OUR event loop (the core
//                                  never owns the loop — that is the library/host
//                                  line), plus caliper_core_event to feed input
//   5. caliper_core_shutdown     — tear it all down on window close
//
// Everything else in this file is ordinary Cocoa boilerplate — window, view,
// timer, event translation — deliberately sparse so the five calls stand out.
//
// Run:  ./embed_host [manifest-id]        (default: dev.caliper.instance-scope)
//   CALIPER_EMBED_EXIT_AFTER=<seconds>   auto-close for a headless run-proof.
// ===========================================================================
#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

#include <caliper/embed.h>

#include <cstdio>
#include <cstdlib>

// The embedded core, shared by the view (input) and the frame timer (pump).
static CaliperCore* g_core = nullptr;

// --- The core's OWN diagnostics sink -------------------------------------
// This carries renderer pick, applet load/refusal, and crash text — the lines
// the CORE emits. It is NOT the applet log service: caliper.log.v1 is a separate
// process-global stderr sink in v0, so an applet's own provenance lines (e.g.
// the zero-copy line) still print themselves and do NOT arrive here.
static void embed_log(void*, int level, const char* msg) {
    static const char* kTag[] = {"debug", "info", "warn", "error"};
    std::fprintf(stderr, "[embed-host] %s: %s\n",
                 kTag[(level >= 0 && level <= 3) ? level : 1], msg);
}
static void embed_crash(void*, const char* applet_id, const char* fault) {
    std::fprintf(stderr, "[embed-host] applet '%s' faulted and was quarantined "
                         "(the host lives on): %s\n", applet_id, fault);
}

// Physical-pixel, top-left-origin mouse position from a Cocoa event. The core
// divides by content_scale to recover logical points for ImGui; we owe it
// physical px. (NSView is bottom-left origin, so we flip Y.)
static void mouse_xy(NSView* v, NSEvent* e, float* x, float* y) {
    NSPoint p = [v convertPoint:e.locationInWindow fromView:nil];
    CGFloat s = v.window.backingScaleFactor;
    *x = (float)(p.x * s);
    *y = (float)((v.bounds.size.height - p.y) * s);
}

static void send(CaliperInputEvent ev) {
    ev.struct_size = sizeof ev;
    caliper_core_event(g_core, &ev);
}

// ---------------------------------------------------------------------------
// The content view: a plain NSView whose backing layer the core replaces with
// its CAMetalLayer at attach time. All it does itself is translate Cocoa input
// into toolkit-neutral CaliperInputEvent and forward it (STEP 4, input half).
// ---------------------------------------------------------------------------
@interface CanvasView : NSView
@end

@implementation CanvasView
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)wantsUpdateLayer { return YES; }

- (void)updateTrackingAreas {
    for (NSTrackingArea* a in self.trackingAreas) [self removeTrackingArea:a];
    NSTrackingArea* ta = [[NSTrackingArea alloc]
        initWithRect:self.bounds
             options:NSTrackingMouseMoved | NSTrackingActiveInKeyWindow |
                     NSTrackingInVisibleRect
               owner:self userInfo:nil];
    [self addTrackingArea:ta];
}

- (void)mouseMoved:(NSEvent*)e     { [self move:e]; }
- (void)mouseDragged:(NSEvent*)e   { [self move:e]; }
- (void)rightMouseDragged:(NSEvent*)e { [self move:e]; }
- (void)otherMouseDragged:(NSEvent*)e { [self move:e]; }
- (void)move:(NSEvent*)e {
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_MOVE;
    mouse_xy(self, e, &ev.x, &ev.y);
    send(ev);
}

- (void)button:(int)b down:(int)d event:(NSEvent*)e {
    [self move:e];   // ImGui wants a fresh position with the click
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_BUTTON;
    ev.button = b; ev.down = d;
    send(ev);
}
- (void)mouseDown:(NSEvent*)e      { [self button:0 down:1 event:e]; }
- (void)mouseUp:(NSEvent*)e        { [self button:0 down:0 event:e]; }
- (void)rightMouseDown:(NSEvent*)e { [self button:1 down:1 event:e]; }
- (void)rightMouseUp:(NSEvent*)e   { [self button:1 down:0 event:e]; }
- (void)otherMouseDown:(NSEvent*)e { [self button:2 down:1 event:e]; }
- (void)otherMouseUp:(NSEvent*)e   { [self button:2 down:0 event:e]; }

- (void)scrollWheel:(NSEvent*)e {
    CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_MOUSE_SCROLL;
    ev.dx = (float)e.scrollingDeltaX * 0.1f;
    ev.dy = (float)e.scrollingDeltaY * 0.1f;
    send(ev);
}

// Text only: NSEvent keycode -> ImGuiKey is a big lookup table this tutorial
// leaves out. Applet camera control is mouse-driven; text covers ImGui fields.
- (void)keyDown:(NSEvent*)e {
    for (NSUInteger i = 0; i < e.characters.length; ++i) {
        CaliperInputEvent ev{}; ev.type = CALIPER_EVENT_TEXT;
        ev.codepoint = [e.characters characterAtIndex:i];
        send(ev);
    }
}
@end

// ---------------------------------------------------------------------------
// App + window delegate: owns lifetime and forwards resize / scale / focus.
// ---------------------------------------------------------------------------
@interface HostDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, strong) NSWindow* window;
@property(nonatomic, strong) CanvasView* view;
@property(nonatomic, strong) NSTimer* pump;
@property(nonatomic, assign) const char* appletId;
@end

@implementation HostDelegate

- (void)pushResizeAndScale {
    CGFloat s = self.window.backingScaleFactor;
    NSSize pts = self.view.bounds.size;
    CaliperInputEvent sc{}; sc.struct_size = sizeof sc;
    sc.type = CALIPER_EVENT_CONTENT_SCALE; sc.scale = (float)s;
    caliper_core_event(g_core, &sc);
    CaliperInputEvent rz{}; rz.struct_size = sizeof rz;
    rz.type = CALIPER_EVENT_RESIZE;
    rz.width = (int)(pts.width * s); rz.height = (int)(pts.height * s);
    caliper_core_event(g_core, &rz);
}

- (void)windowDidResize:(NSNotification*)n              { [self pushResizeAndScale]; }
- (void)windowDidChangeBackingProperties:(NSNotification*)n { [self pushResizeAndScale]; }
- (void)windowDidBecomeKey:(NSNotification*)n {
    CaliperInputEvent ev{}; ev.struct_size = sizeof ev;
    ev.type = CALIPER_EVENT_FOCUS; ev.focused = 1;
    caliper_core_event(g_core, &ev);
}
- (void)windowDidResignKey:(NSNotification*)n {
    CaliperInputEvent ev{}; ev.struct_size = sizeof ev;
    ev.type = CALIPER_EVENT_FOCUS; ev.focused = 0;
    caliper_core_event(g_core, &ev);
}

// STEP 4: pump exactly one frame from OUR timer. The core does one frame and
// returns — no sleep, no vsync wait; the loop is ours.
- (void)tick:(NSTimer*)t { caliper_core_frame(g_core); }

- (void)applicationDidFinishLaunching:(NSNotification*)n {
    const CGFloat W = 1280, H = 800;
    self.window = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, W, H)
                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                            NSWindowStyleMaskResizable
                    backing:NSBackingStoreBuffered defer:NO];
    [self.window setTitle:@"caliper embed_host"];
    [self.window setDelegate:self];
    [self.window setAcceptsMouseMovedEvents:YES];

    self.view = [[CanvasView alloc] initWithFrame:NSMakeRect(0, 0, W, H)];
    [self.window setContentView:self.view];
    [self.window makeFirstResponder:self.view];
    [self.window center];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];

    CGFloat scale = self.window.backingScaleFactor;

    // STEP 1: create the core. Renderer DEFAULT resolves to Metal here; applets
    // are discovered from applets_dir (the build's applets tree, passed in main).
    CaliperCoreDesc desc{};
    desc.struct_size = sizeof desc;
    desc.renderer    = CALIPER_RENDERER_DEFAULT;
    desc.applets_dir = std::getenv("CALIPER_EMBED_APPLETS");  // may be NULL
    desc.log_fn      = &embed_log;
    desc.crash_fn    = &embed_crash;
    g_core = caliper_core_create(&desc);
    if (!g_core) { std::fprintf(stderr, "[embed-host] core create failed\n"); [NSApp terminate:nil]; return; }

    // STEP 2: attach OUR NSView as the canvas. Size is PHYSICAL pixels; the core
    // paints a CAMetalLayer into the view and honors content_scale for HiDPI.
    CaliperCanvasDesc canvas{};
    canvas.struct_size   = sizeof canvas;
    canvas.mode          = CALIPER_CANVAS_WINDOW;
    canvas.width         = (int)(W * scale);
    canvas.height        = (int)(H * scale);
    canvas.content_scale = (float)scale;
    if (!caliper_core_attach_canvas(g_core, (__bridge void*)self.view, &canvas)) {
        std::fprintf(stderr, "[embed-host] attach_canvas failed: %s\n",
                     caliper_core_last_error(g_core));
        [NSApp terminate:nil]; return;
    }

    // STEP 3: load one applet by manifest id (argv[1], default set in main).
    if (!caliper_core_load_applet(g_core, self.appletId)) {
        std::fprintf(stderr, "[embed-host] load_applet '%s' failed: %s\n",
                     self.appletId, caliper_core_last_error(g_core));
        [NSApp terminate:nil]; return;
    }

    // Drive STEP 4 from a 60 Hz timer added to the common run-loop modes so it
    // keeps firing through live window resize.
    self.pump = [NSTimer timerWithTimeInterval:1.0 / 60.0 target:self
                                      selector:@selector(tick:) userInfo:nil repeats:YES];
    [[NSRunLoop currentRunLoop] addTimer:self.pump forMode:NSRunLoopCommonModes];

    // Headless run-proof hook: auto-close after N seconds (no user needed).
    if (const char* s = std::getenv("CALIPER_EMBED_EXIT_AFTER")) {
        [self.window performSelector:@selector(performClose:) withObject:nil
                          afterDelay:atof(s)];
    }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)a { return YES; }

- (void)windowWillClose:(NSNotification*)n {
    [self.pump invalidate]; self.pump = nil;
    // STEP 5: tear down the core (unload applet, join jobs, drop renderer +
    // ImGui) — the exact reverse of create.
    caliper_core_shutdown(g_core);
    g_core = nullptr;
}
@end

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        HostDelegate* d = [HostDelegate new];
        d.appletId = (argc > 1) ? argv[1] : "dev.caliper.instance-scope";
        [NSApp setDelegate:d];
        [NSApp run];
    }
    return 0;
}
