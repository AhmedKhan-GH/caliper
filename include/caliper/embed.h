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
/* v1.2 views: CaliperTensor (the SDK's DLPack-aligned interchange type, pure C;
 * sdk/include rides libcaliper's PUBLIC caliper::sdk link). */
#include <caliper/tensor.h>

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

/* ---------------------------------------------------------------------------
 * v1.1 — the "consumer" pass (docs/superpowers/specs/2026-07-12-compass-
 * consumer-design.md §3). ALL additive; every v1 struct is byte-stable and no
 * struct grew a member, so CALIPER_EMBED_API_VERSION stays 1 (the size gate is
 * about struct fields; v1.1's additions are a new free function plus semantics
 * activated on EXISTING reserved fields). A v1 caller links and runs unchanged.
 *
 * HONESTY REGISTER (what v1.1 changed against the v0 gaps):
 *   - get_service           ADDED  — caliper_core_get_service: a host consumes
 *                                     the applets' service tables (P2/P3).
 *   - data_dir gap          CLOSED — CaliperCoreDesc.data_dir now roots the
 *                                     stores (was reserved/ignored in v0).
 *   - log-singleton gap     CLOSED — applet caliper.log.v1 lines route to
 *                                     CaliperLogFn (was stderr-only in v0).
 *   - one-core-per-process  REMAINS — create still refuses a second live core.
 *   - one-canvas-per-core   REMAINS — attach_canvas still refuses a second.
 * ------------------------------------------------------------------------- */

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

/* Diagnostics sink. NULL -> stderr. Receives BOTH streams, distinguishable by a
 * prefix on the message (v1.1):
 *   - CORE diagnostics (renderer pick, refusals, crash text) arrive UNtagged.
 *   - APPLET caliper.log.v1 lines arrive tagged "[applet] " (v1.1 routes the
 *     applet log service here when this callback is set — it was stderr-only in
 *     v0; a host surfaces them in a native Log pane, not stderr).
 * `level` is the CaliperLogLevel value (0=DEBUG..3=ERROR). `message` is valid
 * only for the call. Called from the frame thread for core lines, and from any
 * thread for applet log lines (log.v1 is worker-callable) — a host sink that
 * touches shared UI state must marshal to its UI thread itself. */
typedef void (*CaliperLogFn)(void* userdata, int level, const char* message);

/* Applet-fault callback (§4.3). Fired AFTER the faulting applet is quarantined
 * and torn down; the core keeps running. applet_id/fault are valid only for the
 * duration of the call. NULL -> the fault is logged and swallowed. */
typedef void (*CaliperCrashFn)(void* userdata, const char* applet_id,
                               const char* fault);

typedef struct CaliperCoreDesc {
    size_t          struct_size;   /* = sizeof(CaliperCoreDesc); FIRST member. */
    CaliperRenderer renderer;      /* backend to embed.                        */
    const char*     data_dir;      /* v1.1: per-core app-data root. NON-NULL
                                    * routes metrics/artifacts/applet-data under
                                    * this directory (created if missing) — a
                                    * document app's per-project root. NULL keeps
                                    * the OS default byte-for-byte (the caliper
                                    * exe passes NULL). Valid for one live core;
                                    * restored on shutdown (one core per process).
                                    * Also holds the ImGui window layout
                                    * (<data_dir>/imgui.ini); with NULL no layout
                                    * is persisted and nothing is written to the
                                    * process's working directory.               */
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

/* ---------------------------------------------------------------------------
 * v1.2 — built-in tensor views (the Python entry point's caliper.show). Additive.
 * ---------------------------------------------------------------------------
 * A host hands the core a CaliperTensor and the CORE draws it: a texture made
 * through the process tensor bridge, plus an ImGui window titled with the view's
 * name, composited in caliper_core_frame AFTER the loaded applet's draw (so views
 * work with no applet loaded and alongside one). No ImGui, renderer or texture
 * type crosses the seam — the caller only names views and passes tensors. Same
 * name = update the existing view in place. Views are drawn in creation order.
 *
 * WHAT IS ACCEPTED (validated here, refused with last_error otherwise; the
 * Python layer pre-converts everything else into one of these two shapes):
 *   - 2-D f32 (H,W): colormapped through `colormap` over [vmin, vmax]
 *     (CaliperColormap: 0 viridis, 1 magma, 2 rdbu; idx = clamp((v-vmin)/
 *     (vmax-vmin),0,1)*255+0.5, NaN -> idx 0 — the bridge's frozen rule).
 *   - 3-D u8 (H,W,4) RGBA: shown as-is. C must be 4 (no RGB/gray here).
 *   Row-major contiguous (strides in ELEMENTS), each dim in 1..16384, device
 *   CPU or METAL. For a CPU tensor the bytes are read at data + byte_offset.
 *
 * TWO DEVICE PATHS, NO SILENT FALLBACK:
 *   - CPU: CPU-staged upload (path "cpu-staged"). The bytes are copied before
 *     return; the caller may free/rewrite them immediately.
 *   - METAL: ZERO-COPY. tensor->data is an in-process id<MTLBuffer> (DLPack's
 *     kDLMetal `data`: the storage's MTLBuffer object pointer) and byte_offset is
 *     DLPack's byte_offset (must be a multiple of 4). The view imports the buffer
 *     (a strong retain — tensor_bridge.v1_2 import_allocation, cached per view
 *     and re-imported when the buffer pointer changes) and the renderer reads it
 *     IN PLACE at byte_offset on the GPU (path "compute-imported" for f32,
 *     "blit-imported" for u8; info.imported = 1). If the device path refuses
 *     (wrong GPU, buffer shorter than byte_offset + extent, no import cap), the
 *     call FAILS — the core never quietly downloads a METAL tensor to the host.
 *   tensor->stream is passed through (a non-NULL MTLCommandQueue* GPU-orders the
 *   read after that queue's committed work, bridge v1.1 D24); NULL is the normal
 *   case for a caller that synchronized first.
 *
 * SYNC / MEMORY-STABILITY CONTRACT (from tensor_bridge_v1_2.h, verbatim in
 * spirit): the GPU reads [byte_offset, byte_offset + extent) of the buffer AFTER
 * this call returns, unless WAIT_GPU is set. So:
 *   - BEFORE the call the producer must have FINISHED writing those bytes (the
 *     Python layer snapshots on torch's stream and polls a torch.mps.Event until
 *     the snapshot completed, never a synchronize; device-side ordering only
 *     covers producer work committed BEFORE the call).
 *   - WITHOUT CALIPER_VIEW_FLAG_WAIT_GPU the bytes must not be rewritten until
 *     the next update of the same view (or the next caliper_core_frame).
 *   - WITH CALIPER_VIEW_FLAG_WAIT_GPU the call blocks until the renderer's read
 *     has completed, so on return the caller may rewrite or free the buffer at
 *     once. The wait is measured: info.last_update_ms includes it.
 *   The texture is the view's own copy of the pixels; the source buffer is only
 *   read during the update. A view keeps its last MTLBuffer RETAINED (the import)
 *   until it is updated from a different buffer or removed.
 *
 * MAPPING CHANGES: a different vmin/vmax/colormap on an existing f32 view re-pins
 * the mapping and re-maps the new tensor into the SAME texture (info.creates does
 * not move). A different shape/dtype recreates the texture (creates += 1).
 *
 * THREADING: FRAME-THREAD ONLY — every function below except
 * caliper_core_colormap_lut must be called from the thread that calls
 * caliper_core_frame(), between frames (same class as caliper.tensor_bridge.*).
 * Views need an attached canvas (the renderer's device exists only after
 * attach_canvas); view_tensor before attach is an honest refusal. Views are
 * destroyed (textures and imports released) by caliper_core_shutdown, before the
 * renderer is torn down.
 * ------------------------------------------------------------------------- */
#define CALIPER_EMBED_HAVE_VIEWS 1

enum {
    CALIPER_VIEW_FLAG_RAW      = 1 << 0,  /* draw the texture 1:1 at canvas (0,0), no window chrome/padding (tests) */
    CALIPER_VIEW_FLAG_WAIT_GPU = 1 << 1   /* block until the device update completed (safe handoff; measured cost) */
};

typedef struct CaliperViewDesc {
    size_t               struct_size;   /* = sizeof(CaliperViewDesc) */
    const char*          name;          /* view id; same name = update in place */
    const CaliperTensor* tensor;        /* 2-D f32 (H,W) or 3-D u8 (H,W,4); CPU or METAL */
    uint64_t             byte_offset;   /* METAL only: byte offset into tensor->data's MTLBuffer (DLPack byte_offset) */
    int32_t              colormap;      /* CaliperColormap for 2-D f32 (0 viridis, 1 magma, 2 rdbu) */
    float                vmin, vmax;    /* mapping range for 2-D f32 */
    uint32_t             flags;         /* CALIPER_VIEW_FLAG_* */
} CaliperViewDesc;

typedef struct CaliperViewInfo {
    size_t      struct_size;
    const char* path;          /* renderer path of the LAST update: "compute-imported", "blit-imported", "compute", "blit", "cpu-staged"; valid until the next core call */
    int32_t     width, height; /* texture size in texels */
    int32_t     device;        /* CaliperDeviceKind of the last tensor */
    uint64_t    updates;       /* number of successful updates */
    double      last_update_ms;/* CPU wall time of the last view_tensor call (includes the GPU wait when WAIT_GPU) */
    int32_t     imported;      /* 1 if the last update read an imported MTLBuffer in place (zero-copy), else 0 */
    uint64_t    creates;       /* texture (re)creations: 1 after the first show, +1 per shape/dtype change; a mapping change does not bump it */
} CaliperViewInfo;

/* Frame-thread only (same thread as caliper_core_frame). Creates or updates the
 * named view. Returns 1 on success, 0 with last_error set. A failed FIRST show
 * leaves no view; a failed update of an existing view keeps its previous pixels. */
int caliper_core_view_tensor(CaliperCore* core, const CaliperViewDesc* desc);
int caliper_core_view_remove(CaliperCore* core, const char* name);    /* 1 if removed */
int caliper_core_view_count(CaliperCore* core);
int caliper_core_view_info(CaliperCore* core, const char* name, CaliperViewInfo* out); /* 1 on success */
/* Test/measurement surface: copy the view's TEXTURE (not the canvas) to buf as tightly
 * packed RGBA8 (width*height*4 bytes). 1 on success. */
int caliper_core_view_readback(CaliperCore* core, const char* name, void* buf, size_t buf_bytes);
/* The built-in 256-entry RGBA8 LUT for a colormap (CPU reference for tests). 1 on success.
 * Each entry packs r | g<<8 | b<<16 | a<<24. Needs no core; callable from any thread. */
int caliper_core_colormap_lut(int32_t colormap, uint32_t out256[256]);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_EMBED_H */
