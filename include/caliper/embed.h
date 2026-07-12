/* ===========================================================================
 * caliper/embed.h — the embed C ABI (libcaliper / Compass R4, L2a)
 *
 * This is the SEAM that makes "embeddable" true: the small C ABI a host binary
 * (Compass's wx chrome, examples/embed_host, the caliper exe eventually) uses
 * to run the applet canvas — ImGui + HostRenderer + bridge + geometry — inside
 * a view it owns, WITHOUT linking ImGui, torch, or any renderer type.
 *
 * WHO CALLS THIS: embedders (hosts). NOT applets. The applet-facing ABI is
 * <caliper/abi.h> + the sugar; this header lives on a SEPARATE include root
 * (top-level include/, a PUBLIC include dir of the libcaliper target only) so
 * an applet — which links caliper::sdk (sdk/include) and never libcaliper —
 * physically cannot #include it. An applet embedding a core would be a category
 * error; the include topology forbids it.
 *
 * C, not C++ (mirrors D1, the applet contract): a host built years apart from
 * libcaliper must still embed it. C++ sugar for host authors can ship later.
 *
 * ---------------------------------------------------------------------------
 * LIFECYCLE & THREADING CONSTRAINTS (design §4.3 — verbatim, an embedder that
 * ignores these gets crashes the core cannot prevent):
 *
 *  - THE CORE NEVER OWNS THE EVENT LOOP. caliper_core_frame() does exactly ONE
 *    frame and returns: no polling, no sleeping, no vsync wait. The embedder
 *    calls it from ITS loop (wx idle/timer, a GLFW loop, a CVDisplayLink, ...).
 *    That is the whole difference between a library and a host.
 *
 *  - INPUT CROSSES AS DATA, NOT TOOLKIT TYPES. The embedder translates its own
 *    GLFW/AppKit/wx events into CaliperInputEvent; the core feeds ImGuiIO. No
 *    GLFWwindow, NSEvent, or wxEvent ever appears here.
 *
 *  - ONE ImGui CONTEXT PER CANVAS, owned by the core. The embedder never
 *    touches ImGui state; the allocator handoff stays internal.
 *
 *  - ONE CaliperCore PER PROCESS in v0 (the one-libtorch-per-process policy,
 *    D5, already binds the process). caliper_core_create refuses a second live
 *    core with a NULL return; shut the first down first.
 *
 *  - CRASH CONTAINMENT. An applet fault is caught by the core's existing crash
 *    guard, surfaced through CaliperCoreDesc.crash_fn, and the applet is
 *    quarantined — the embedder is NOT taken down with it.
 *
 *  - FRAME-THREAD DISCIPLINE carries over: call frame()/event()/attach/read
 *    from ONE thread (the UI thread). Applet torch work runs on jobs threads
 *    and draws from snapshots; that contract is unchanged and internal.
 * ===========================================================================*/
#ifndef CALIPER_EMBED_H
#define CALIPER_EMBED_H

#include <stddef.h>   /* size_t   */
#include <stdint.h>   /* uint32_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped when this ABI grows a field. The struct_size FIRST-member on every
 * struct is the compatibility gate, and the rule is exact: the core requires
 * caller->struct_size >= the core's own sizeof(...) and REFUSES otherwise (a
 * caller built against an OLDER, smaller header is rejected, not silently
 * misread). Fields are append-only; when the ABI grows, CALIPER_EMBED_API_VERSION
 * bumps and the new fields land after the old ones, so a caller compiled against
 * a header at least as new as the core passes the size gate and every field the
 * core reads is present. Newer-caller / older-core is out of scope in v0 (one
 * libcaliper per process, built together). */
#define CALIPER_EMBED_API_VERSION 1

/* Opaque handle. One live instance per process in v0 (see header note). */
typedef struct CaliperCore CaliperCore;

/* Which HostRenderer backend the core embeds. GL is intentionally ABSENT: its
 * context ownership is GLFW-coupled chrome (D13, the frozen fallback), never an
 * embed target. DEFAULT resolves to the platform backend (Metal on Apple,
 * Vulkan on Windows) honoring CALIPER_RENDERER. A core whose resolved backend
 * cannot embed refuses at attach_canvas ("embed requires Metal or Vulkan"). */
typedef enum CaliperRenderer {
    CALIPER_RENDERER_DEFAULT = 0,
    CALIPER_RENDERER_METAL   = 1,
    CALIPER_RENDERER_VULKAN  = 2
} CaliperRenderer;

/* Core diagnostics sink (renderer pick, refusals, crash text). NOT the applet
 * log service (caliper.log.v1 stays stderr in v0). NULL -> stderr. */
typedef void (*CaliperLogFn)(void* userdata, int level, const char* message);

/* Applet-fault callback (§4.3). Fired AFTER the faulting applet is quarantined
 * and torn down; the core keeps running. applet_id/fault are valid only for the
 * duration of the call. NULL -> the fault is logged and swallowed. */
typedef void (*CaliperCrashFn)(void* userdata, const char* applet_id,
                               const char* fault);

typedef struct CaliperCoreDesc {
    size_t          struct_size;   /* = sizeof(CaliperCoreDesc); FIRST member. */
    CaliperRenderer renderer;      /* backend to embed.                        */
    const char*     data_dir;      /* IGNORED in v0 (reserved): the process
                                    * app-data path is always used. Threading a
                                    * per-core data root is a host_services
                                    * signature change deferred past R4 (the
                                    * registry is process-global; see report).  */
    const char*     applets_dir;   /* extra applet scan dir; NULL -> default
                                    * discovery (app-data/applets + exe-side). */
    CaliperLogFn    log_fn;        /* NULL -> stderr.                          */
    CaliperCrashFn  crash_fn;      /* NULL -> log-and-swallow.                 */
    void*           userdata;      /* passed back to log_fn / crash_fn.        */
} CaliperCoreDesc;

typedef enum CaliperCanvasMode {
    CALIPER_CANVAS_WINDOW    = 0,  /* native_view is an NSView* / HWND.        */
    CALIPER_CANVAS_OFFSCREEN = 1   /* no view; render to a texture, read back. */
} CaliperCanvasMode;

typedef struct CaliperCanvasDesc {
    size_t            struct_size; /* = sizeof(CaliperCanvasDesc); FIRST.      */
    CaliperCanvasMode mode;
    int               width;       /* physical pixels.                        */
    int               height;      /* physical pixels.                        */
    float             content_scale; /* DPI scale (1.0 = 1x); <=0 -> 1.0.     */
} CaliperCanvasDesc;

typedef enum CaliperEventType {
    CALIPER_EVENT_MOUSE_MOVE    = 0, /* uses x, y (physical px)               */
    CALIPER_EVENT_MOUSE_BUTTON  = 1, /* uses button, down                     */
    CALIPER_EVENT_MOUSE_SCROLL  = 2, /* uses dx, dy                           */
    CALIPER_EVENT_KEY           = 3, /* uses key (== ImGuiKey), down, mods    */
    CALIPER_EVENT_TEXT          = 4, /* uses codepoint                        */
    CALIPER_EVENT_RESIZE        = 5, /* uses width, height (physical px)      */
    CALIPER_EVENT_CONTENT_SCALE = 6, /* uses scale                            */
    CALIPER_EVENT_FOCUS         = 7  /* uses focused (0/1)                    */
} CaliperEventType;

/* Bit flags for CaliperInputEvent.mods (a KEY event's modifier state). */
enum {
    CALIPER_MOD_CTRL  = 1 << 0,
    CALIPER_MOD_SHIFT = 1 << 1,
    CALIPER_MOD_ALT   = 1 << 2,
    CALIPER_MOD_SUPER = 1 << 3
};

/* One toolkit-neutral input event. Only the fields named in the CaliperEventType
 * comment above are read for a given type; leave the rest zero. */
typedef struct CaliperInputEvent {
    size_t           struct_size;  /* = sizeof(CaliperInputEvent); FIRST.     */
    CaliperEventType type;
    float            x, y;         /* mouse position (physical px)            */
    float            dx, dy;       /* scroll delta                            */
    int              button;       /* 0=left, 1=right, 2=middle               */
    int              down;         /* 0/1 for button/key press state          */
    int              key;          /* CaliperKey == ImGuiKey value            */
    int              mods;         /* CALIPER_MOD_* bitset                     */
    unsigned int     codepoint;    /* UTF-32 for CALIPER_EVENT_TEXT           */
    int              width, height;/* CALIPER_EVENT_RESIZE (physical px)      */
    float            scale;        /* CALIPER_EVENT_CONTENT_SCALE             */
    int              focused;      /* CALIPER_EVENT_FOCUS                     */
} CaliperInputEvent;

/* --- Lifecycle ---------------------------------------------------------- */

/* Create the core (ImGui context + renderer + service registry + loader, in
 * the L1-proven order). Returns NULL and logs on: a second live core, an
 * unsupported renderer for this OS, or renderer init failure. */
CaliperCore* caliper_core_create(const CaliperCoreDesc* desc);

/* Tear down: unload the applet, join jobs, close stores, drop the renderer and
 * ImGui context — the exact reverse order of create (crash-order load-bearing,
 * see the impl). Safe on NULL. Clears the one-core-per-process lock. */
void caliper_core_shutdown(CaliperCore* core);

/* Attach the applet canvas. native_view is an NSView* / HWND for CANVAS_WINDOW,
 * ignored (pass NULL) for CANVAS_OFFSCREEN. Returns 1 on success, 0 on refusal
 * (backend can't embed, canvas already attached, bad size) — a 0 leaves the
 * core usable and sets last_error. v0: one canvas per core. */
int  caliper_core_attach_canvas(CaliperCore* core, void* native_view,
                                const CaliperCanvasDesc* desc);

/* Pump exactly ONE frame: clear the canvas, run the loaded applet's draw under
 * the crash guard, composite + present/store. No-op (sets last_error) if no
 * canvas is attached. Never blocks on the event loop. */
void caliper_core_frame(CaliperCore* core);

/* Feed one input event into the core's ImGuiIO. No-op before a canvas exists. */
void caliper_core_event(CaliperCore* core, const CaliperInputEvent* event);

/* --- Applet control (reuses the loader's manifest discovery) ------------- */

/* Load + launch the applet whose manifest id matches (e.g. "dev.caliper.hello").
 * Returns 1 on success, 0 if unknown/refused/failed (last_error set). A canvas
 * must be attached FIRST — an applet's launch/first frame touches the renderer's
 * ImGui backend, so load before attach_canvas is an honest refusal, not a crash.
 *
 * TEARDOWN-FIRST SEMANTICS. Any currently-loaded applet is torn down FIRST
 * (workers joined, then the instance), and only THEN is the new one launched —
 * so init jobs the new applet schedules are never cancelled by the old one's
 * teardown. Two consequences follow, both intentional:
 *   - Reloading the SAME id is a clean RESTART: the running instance is torn
 *     down (on_cleanup) and a fresh one launched (on_init) — no state carries
 *     over.
 *   - A FAILED launch leaves NO applet loaded, not the previous one: once the
 *     old applet is torn down it is gone even if the new launch is refused.
 * An UNKNOWN id is the one exception — it refuses up front (0, last_error set)
 * WITHOUT disturbing a running applet, so a typo cannot kill the live session. */
int  caliper_core_load_applet(CaliperCore* core, const char* manifest_id);

/* Tear down the loaded applet (jobs joined first, then instance). No-op if none
 * is loaded. Call between frames. */
void caliper_core_unload_applet(CaliperCore* core);

/* --- Offscreen readback (the automatable + byte-compare surface, §7) ----- */

/* Copy the LAST composited frame's pixels to buf as tightly-packed RGBA8,
 * `stride` bytes per row (>= width*4). Returns 1 on success, 0 if the canvas is
 * not offscreen, buf is NULL, or stride is too small. */
int  caliper_core_read_pixels(CaliperCore* core, void* buf, int stride);

/* Human-readable reason for the most recent refusal (empty string if none).
 * Valid until the next core call. Never NULL for a non-NULL core. */
const char* caliper_core_last_error(CaliperCore* core);

/* --- Service consumption (v1.1) — the host becomes a CONSUMER, not just a
 * picture-in-picture embedder ------------------------------------------------
 *
 * Returns the SAME service table an applet receives via CaliperHost.get_service
 * for `id` (the applets' own vocabulary: "caliper.metrics.v1", "caliper.jobs.v1",
 * "caliper.artifacts.v1", "caliper.data.v1", ...). Cast the result to the matching
 * Caliper<Name>V1 struct from the caliper/services headers and call its thunks
 * as an applet would — no renderer/torch/ImGui type crosses the seam (D3: the
 * interchange is C ABI + CaliperTensor + Arrow C streams). C++ hosts may use the
 * caliper.hpp sugar.
 *
 * Returns NULL for an unknown id or a NULL core. The pointer is a process-static
 * table (embed.h's pointer-validity guarantee): valid from create until
 * caliper_core_shutdown, after which — like every other call here — the
 * CaliperCore* is dead and must not be used.
 *
 * THREADING CONTRACT (§3.2 — VERIFIED against the implementations, not assumed;
 * an embedder that ignores it gets data races the core cannot prevent). Two
 * classes:
 *
 *  - ANY-THREAD services — call from any thread, including a host UI thread that
 *    is NOT the caliper_core_frame() thread, concurrently with an applet's
 *    worker writes:
 *      * caliper.metrics.v1   — MetricsStore holds ONE DuckDB connection under
 *                               ONE mutex; every writer AND reader (runs/
 *                               scalars/histograms) takes it, so host-thread
 *                               reads racing applet-thread writes serialize
 *                               (verified: metrics_store.cpp — lock_guard on
 *                               every method).
 *      * caliper.artifacts.v1 — same one-connection-one-mutex model; put/
 *                               path_of/exists/by_run all lock (artifact_store
 *                               .cpp). path_of returns a thread_local buffer,
 *                               valid until the next artifacts call ON THAT
 *                               THREAD.
 *      * caliper.data.v1      — one mutex over query/register/open_dataset
 *                               (data_store.cpp); last_error() is thread-local
 *                               (each thread sees ITS last failing call).
 *      * caliper.jobs.v1      — cross-thread BY DESIGN (submit/cancel/is_running
 *                               /progress over the process JobSystem).
 *      * caliper.device.v1    — reads an immutable negotiated-at-startup record.
 *      * caliper.log.v1       — reentrant; callable from worker threads (routes
 *                               to log_fn when installed, else stderr — below).
 *
 *  - FRAME-THREAD-ONLY services — call ONLY from the thread that calls
 *    caliper_core_frame(); they touch the renderer / the single ImGui context:
 *      * caliper.tensor_bridge.v1 / v1.1 / v1.2  (GPU upload, draw-adjacent)
 *      * caliper.geometry.v1 ... v1.3            (GPU draw)
 *      * caliper.ui.v1                           (the ImGui/ImPlot contexts —
 *                                                 meaningless to a host anyway)
 *
 * P3 caveat (D5, one torch per process, never the host's): a host pushes
 * PARAMETERS; an APPLET's worker produces device tensors. The bridge's host-side
 * use is CPU-staged uploads on the frame thread only. */
const void* caliper_core_get_service(CaliperCore* core, const char* id);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_EMBED_H */
