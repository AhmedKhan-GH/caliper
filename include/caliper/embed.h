/* ===========================================================================
 * caliper/embed.h - the embed C ABI of libcaliper
 *
 * The small C ABI that a host binary uses to run the applet canvas inside a view it owns, without
 * linking ImGui, torch or any renderer type. Embedders call it, applets never do. The header sits
 * on its own include root, so an applet that links caliper::sdk cannot include it.
 * The docs page "Embedding API" explains ownership, threading and limits.
 * ===========================================================================*/
#ifndef CALIPER_EMBED_H
#define CALIPER_EMBED_H

#include <stddef.h>   /* size_t   */
#include <stdint.h>   /* uint32_t */
/* v1.2 views: CaliperTensor, the SDK interchange type (pure C). */
#include <caliper/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The major version of this ABI, which is the major part of `CALIPER_EMBED_VERSION`.
 *
 * It bumps when the ABI grows a field. Every struct starts with `struct_size`, which is the
 * compatibility gate. The core requires `struct_size` to be at least its own `sizeof` for the struct
 * and refuses a smaller one, so a caller built against an older header is rejected and never
 * misread. Fields are append-only and land after the old ones. A caller built against a header at
 * least as new as the core passes the gate and finds every field the core reads. A newer caller
 * with an older core is not supported.
 */
#define CALIPER_EMBED_API_VERSION 1

/* v1.1 added `caliper_core_get_service`, the `data_dir` root and the routing of applet log lines.
 * It changed no struct, so `CALIPER_EMBED_API_VERSION` stays 1. */

/** The opaque handle of one core. A process holds one live core at a time. */
typedef struct CaliperCore CaliperCore;

/**
 * The HostRenderer backend that the core embeds.
 *
 * There is no fallback renderer. `caliper_core_create` refuses when `CALIPER_RENDERER=gl` asks for
 * the removed OpenGL renderer, and `caliper_core_attach_canvas` refuses when the backend cannot
 * start, with a last error that names the renderer and why it could not start.
 */
typedef enum CaliperRenderer {
    /** The platform backend, Metal on Apple and Vulkan on Windows. */
    CALIPER_RENDERER_DEFAULT = 0,
    /** The Metal backend. */
    CALIPER_RENDERER_METAL   = 1,
    /** The Vulkan backend. */
    CALIPER_RENDERER_VULKAN  = 2
} CaliperRenderer;

/**
 * The diagnostics sink of a core.
 *
 * The callback receives two streams, which differ by a prefix on the message. Core diagnostics
 * such as the renderer choice, refusals and crash text arrive untagged. Applet `caliper.log.v1`
 * lines arrive tagged with `[applet] `. `level` is a `CaliperLogLevel` value, from 0 for debug to
 * 3 for error. `message` is valid only during the call. The core calls the sink from the frame
 * thread for its own lines and from any thread for applet lines, so a sink that touches shared
 * UI state must marshal to its UI thread. With a `NULL` sink, both streams go to stderr.
 */
typedef void (*CaliperLogFn)(void* userdata, int level, const char* message);

/**
 * The applet-fault callback of a core.
 *
 * The core fires it after it has quarantined and torn down the faulting applet, and the core keeps
 * running. `applet_id` and `fault` are valid only during the call. With a `NULL` callback, the core
 * logs the fault and swallows it.
 */
typedef void (*CaliperCrashFn)(void* userdata, const char* applet_id,
                               const char* fault);

/**
 * The settings of a core, which `caliper_core_create` takes.
 *
 * Set `struct_size` to `sizeof(CaliperCoreDesc)` and zero every field that you do not use.
 */
typedef struct CaliperCoreDesc {
    /** `sizeof(CaliperCoreDesc)`, as the caller built it. It must be the first member. */
    size_t          struct_size;
    /** The backend to embed. */
    CaliperRenderer renderer;
    /**
     * The per-core app-data root, or `NULL` for the default of the operating system. A directory
     * (created if missing) holds the metrics, artifacts and applet data stores of this core and
     * the window layout in `imgui.ini`. With `NULL`, the core keeps no layout and writes nothing to
     * the working directory. The setting lasts for the one live core and ends at shutdown.
     */
    const char*     data_dir;
    /** An extra applet search folder, or `NULL` for the default discovery. */
    const char*     applets_dir;
    /** The diagnostics sink, or `NULL` for stderr. */
    CaliperLogFn    log_fn;
    /** The applet-fault callback, or `NULL` to log and swallow faults. */
    CaliperCrashFn  crash_fn;
    /** The pointer that the core passes back to `log_fn` and `crash_fn`. */
    void*           userdata;
} CaliperCoreDesc;

/** The kind of canvas that `caliper_core_attach_canvas` creates. */
typedef enum CaliperCanvasMode {
    /** The canvas paints into a native view. `native_view` is an `NSView*` or an `HWND`. */
    CALIPER_CANVAS_WINDOW    = 0,
    /**
     * The canvas has no view. The core renders to a texture, which `caliper_core_read_pixels` reads
     * back.
     */
    CALIPER_CANVAS_OFFSCREEN = 1
} CaliperCanvasMode;

/** The settings of the canvas, which `caliper_core_attach_canvas` takes. */
typedef struct CaliperCanvasDesc {
    /** `sizeof(CaliperCanvasDesc)`, as the caller built it. It must be the first member. */
    size_t            struct_size;
    /** Whether the canvas paints into a native view or renders offscreen. */
    CaliperCanvasMode mode;
    /** The width of the canvas in physical pixels. */
    int               width;
    /** The height of the canvas in physical pixels. */
    int               height;
    /** The display scale, where 1.0 is one pixel per point. A value of 0 or less means 1.0. */
    float             content_scale;
} CaliperCanvasDesc;

/**
 * The type of an input event.
 *
 * Each constant names the fields of `CaliperInputEvent` that the core reads for that type.
 */
typedef enum CaliperEventType {
    /** The mouse moved. The core reads `x` and `y`, in physical pixels. */
    CALIPER_EVENT_MOUSE_MOVE    = 0,
    /** A mouse button changed. The core reads `button` and `down`. */
    CALIPER_EVENT_MOUSE_BUTTON  = 1,
    /** The mouse wheel or a trackpad scrolled. The core reads `dx` and `dy`. */
    CALIPER_EVENT_MOUSE_SCROLL  = 2,
    /** A key changed. The core reads `key`, which is an `ImGuiKey` value, and `down` and `mods`. */
    CALIPER_EVENT_KEY           = 3,
    /** A character was typed. The core reads `codepoint`. */
    CALIPER_EVENT_TEXT          = 4,
    /** The canvas changed size. The core reads `width` and `height`, in physical pixels. */
    CALIPER_EVENT_RESIZE        = 5,
    /** The display scale changed. The core reads `scale`. */
    CALIPER_EVENT_CONTENT_SCALE = 6,
    /** The canvas gained or lost focus. The core reads `focused`, which is 0 or 1. */
    CALIPER_EVENT_FOCUS         = 7
} CaliperEventType;

enum {
    /** Bit flag in `CaliperInputEvent.mods` for the Control key. */
    CALIPER_MOD_CTRL  = 1 << 0,
    /** Bit flag in `CaliperInputEvent.mods` for the Shift key. */
    CALIPER_MOD_SHIFT = 1 << 1,
    /** Bit flag in `CaliperInputEvent.mods` for the Alt key. */
    CALIPER_MOD_ALT   = 1 << 2,
    /** Bit flag in `CaliperInputEvent.mods` for the Super key. */
    CALIPER_MOD_SUPER = 1 << 3
};

/**
 * One toolkit-neutral input event.
 *
 * The core reads only the fields that the comment of the `CaliperEventType` constant names. Leave
 * the other fields zero.
 */
typedef struct CaliperInputEvent {
    /** `sizeof(CaliperInputEvent)`, as the caller built it. It must be the first member. */
    size_t           struct_size;
    /** The type of the event. */
    CaliperEventType type;
    /** The mouse position in physical pixels. */
    float            x, y;
    /** The scroll delta. */
    float            dx, dy;
    /** The mouse button: 0 for left, 1 for right and 2 for middle. */
    int              button;
    /** The press state of a button or key event: 1 for down and 0 for up. */
    int              down;
    /** The key, as an `ImGuiKey` value. */
    int              key;
    /** The modifier state of a key event, as a bit set of the `CALIPER_MOD_*` flags. */
    int              mods;
    /** The typed character as a UTF-32 code point. */
    unsigned int     codepoint;
    /** The new size of the canvas in physical pixels. */
    int              width, height;
    /** The new display scale. */
    float            scale;
    /** 1 when the canvas has focus, otherwise 0. */
    int              focused;
} CaliperInputEvent;

/* --- Lifecycle ---------------------------------------------------------- */

/**
 * Create the core, which holds the ImGui context, the renderer, the service registry and the loader.
 * Return `NULL` and log the reason for a second live core, a renderer that this system does not
 * support or `CALIPER_RENDERER=gl`. The renderer starts at `caliper_core_attach_canvas`.
 */
CaliperCore* caliper_core_create(const CaliperCoreDesc* desc);

/**
 * Tear down the core in the reverse order of creation: unload the applet, join the jobs, close the
 * stores and drop the renderer and the ImGui context. Safe on `NULL`. The call also ends the
 * one-core-per-process limit, and the `CaliperCore*` is dead afterward.
 */
void caliper_core_shutdown(CaliperCore* core);

/**
 * Attach the one canvas of the core. `native_view` is an `NSView*` or an `HWND` for
 * `CALIPER_CANVAS_WINDOW`, and `NULL` for `CALIPER_CANVAS_OFFSCREEN`. Return 1 on success, or 0 when
 * the core refuses because the renderer could not start, a canvas is attached already or the size
 * is invalid. A refusal leaves the core usable and sets the last error. When the renderer could not
 * start, the last error is "attach_canvas: the Metal renderer could not start (<why>), and there is
 * no fallback renderer until the WebGPU fallback through Dawn lands", with Vulkan on Windows.
 */
int  caliper_core_attach_canvas(CaliperCore* core, void* native_view,
                                const CaliperCanvasDesc* desc);

/**
 * Run exactly one frame: clear the canvas, draw the loaded applet under the crash guard, then
 * composite and present or store the result. The call never polls, sleeps or waits for vsync.
 * Without an attached canvas it does nothing except set the last error.
 */
void caliper_core_frame(CaliperCore* core);

/** Feed one input event into the core. The call does nothing before a canvas exists. */
void caliper_core_event(CaliperCore* core, const CaliperInputEvent* event);

/* --- Applet control ------------------------------------------------------ */

/**
 * Load and launch the applet whose manifest id is `manifest_id`, for example
 * `dev.example.my-applet`. Return 1 on success, or 0 when the id is unknown or the launch is refused
 * or fails, with the last error set. Attach a canvas first, because the first frame of an applet
 * touches the ImGui backend of the renderer. The core tears down a loaded applet first, so
 * reloading the same id restarts the applet with no carried state, and a failed launch leaves no
 * applet loaded. An unknown id is the one exception: the call refuses it at once and leaves a
 * running applet alone.
 */
int  caliper_core_load_applet(CaliperCore* core, const char* manifest_id);

/**
 * Tear down the loaded applet by joining its jobs and then dropping the instance. The call does
 * nothing when no applet is loaded. Call it between frames.
 */
void caliper_core_unload_applet(CaliperCore* core);

/* --- Offscreen readback -------------------------------------------------- */

/**
 * Copy the last composited frame to `buf` as tightly packed RGBA8 with `stride` bytes per row,
 * where `stride` is at least the width times 4. Return 1 on success, or 0 when the canvas is not
 * offscreen, `buf` is `NULL` or `stride` is too small.
 */
int  caliper_core_read_pixels(CaliperCore* core, void* buf, int stride);

/**
 * Return the reason for the most recent refusal, or an empty string when there is none. The string
 * is never `NULL` for a live core and stays valid until the next call on that core.
 */
const char* caliper_core_last_error(CaliperCore* core);

/* --- Service consumption (v1.1) ------------------------------------------ */

/**
 * Return the service table that an applet receives from `CaliperHost.get_service` for `id`, such as
 * `caliper.metrics.v1` or `caliper.jobs.v1`, or `NULL` for an unknown id or a `NULL` core. Cast the
 * result to the matching struct of the `caliper/services` headers and call its function pointers
 * as an applet does. The table is static for the process and valid from creation until
 * `caliper_core_shutdown`. The metrics, artifacts, data, jobs, device and log services may run on
 * any thread, while the tensor bridge, geometry and UI services are frame-thread only.
 */
const void* caliper_core_get_service(CaliperCore* core, const char* id);

/* --- Built-in tensor views (v1.2) ----------------------------------------
 * A host hands the core a CaliperTensor and the core draws it in an ImGui window that carries the
 * name of the view. Views need an attached canvas. Every call below except
 * `caliper_core_colormap_lut` runs on the frame thread between frames. */

/** Defined as 1 when the header declares the view calls, so that a host can test for them. */
#define CALIPER_EMBED_HAVE_VIEWS 1

enum {
    /** Bit flag to draw the texture 1:1 at the canvas origin with no window chrome, for tests. */
    CALIPER_VIEW_FLAG_RAW      = 1 << 0,
    /**
     * Bit flag to block until the device update completes, which makes the handoff safe and adds a
     * measured cost.
     */
    CALIPER_VIEW_FLAG_WAIT_GPU = 1 << 1
};

/**
 * One tensor view to create or update, which `caliper_core_view_tensor` takes.
 *
 * The core accepts two layouts and refuses any other with a reason in the last error. A 2-D f32
 * tensor of shape `(H,W)` is colormapped over `[vmin, vmax]`. A 3-D u8 tensor of shape `(H,W,4)` is
 * RGBA and shows as is. The tensor must be row-major contiguous, with strides in elements and each
 * dimension in 1 to 16384, on device CPU or METAL.
 *
 * For a CPU tensor, the core reads the bytes at `data + byte_offset` and copies them before the
 * call returns. For a METAL tensor, `data` is an in-process `id<MTLBuffer>` and the renderer reads
 * it in place without a copy. The call fails when the device path refuses, and the core never
 * downloads a METAL tensor to the host. The GPU reads the buffer after the call returns unless
 * `CALIPER_VIEW_FLAG_WAIT_GPU` is set, so the producer must have finished writing before the call.
 */
typedef struct CaliperViewDesc {
    /** `sizeof(CaliperViewDesc)`, as the caller built it. */
    size_t               struct_size;
    /** The id and window title of the view. The same name updates the existing view in place. */
    const char*          name;
    /** The tensor to draw, either 2-D f32 `(H,W)` or 3-D u8 `(H,W,4)`, on device CPU or METAL. */
    const CaliperTensor* tensor;
    /**
     * For METAL only, the byte offset into the `MTLBuffer` of `tensor->data`, as in DLPack. It must
     * be a multiple of 4.
     */
    uint64_t             byte_offset;
    /** The `CaliperColormap` for a 2-D f32 tensor: 0 for viridis, 1 for magma and 2 for rdbu. */
    int32_t              colormap;
    /** The mapping range of a 2-D f32 tensor. Both ends must be finite. */
    float                vmin, vmax;
    /** A bit set of the `CALIPER_VIEW_FLAG_*` flags. */
    uint32_t             flags;
} CaliperViewDesc;

/** The measured state of one view, which `caliper_core_view_info` fills. */
typedef struct CaliperViewInfo {
    /** `sizeof(CaliperViewInfo)`, as the caller built it. */
    size_t      struct_size;
    /**
     * The renderer path of the last update: `compute-imported`, `blit-imported`, `compute`, `blit`
     * or `cpu-staged`. It stays valid until the next core call.
     */
    const char* path;
    /** The size of the texture in texels. */
    int32_t     width, height;
    /** The `CaliperDeviceKind` of the last tensor. */
    int32_t     device;
    /** The number of successful updates. */
    uint64_t    updates;
    /**
     * The CPU wall time of the last `caliper_core_view_tensor` call in milliseconds, including the
     * GPU wait when `CALIPER_VIEW_FLAG_WAIT_GPU` was set.
     */
    double      last_update_ms;
    /**
     * 1 when the last update read an imported `MTLBuffer` in place, which is the zero-copy case,
     * otherwise 0.
     */
    int32_t     imported;
    /**
     * The number of texture creations: 1 after the first show and 1 more for each shape or dtype
     * change. A mapping change does not count.
     */
    uint64_t    creates;
} CaliperViewInfo;

/**
 * Create the view named `desc->name`, or update it when it exists. Return 1 on success, or 0 with
 * the last error set. A failed first show leaves no view, and a failed update keeps the previous
 * pixels of the view. A different `vmin`, `vmax` or colormap on an existing f32 view maps the new
 * tensor into the same texture, while a different shape or dtype recreates the texture.
 */
int caliper_core_view_tensor(CaliperCore* core, const CaliperViewDesc* desc);
/**
 * Remove the view named `name` and release its texture and import. Return 1 if the view existed and
 * was removed, otherwise 0.
 */
int caliper_core_view_remove(CaliperCore* core, const char* name);
/** Return the number of views. */
int caliper_core_view_count(CaliperCore* core);
/**
 * Fill `out` with the measured state of the view named `name`. Return 1 on success, otherwise 0.
 */
int caliper_core_view_info(CaliperCore* core, const char* name, CaliperViewInfo* out);
/**
 * Copy the texture of the view named `name`, not the canvas, to `buf` as tightly packed RGBA8 of
 * `width*height*4` bytes, for tests and measurement. `buf_bytes` is the size of `buf`. Return 1 on
 * success, otherwise 0.
 */
int caliper_core_view_readback(CaliperCore* core, const char* name, void* buf, size_t buf_bytes);
/**
 * Copy the built-in 256-entry RGBA8 table of `colormap` to `out256`, as a CPU reference for tests.
 * Each entry packs `r | g<<8 | b<<16 | a<<24`. Return 1 on success, otherwise 0. The call needs no
 * core and may run on any thread.
 */
int caliper_core_colormap_lut(int32_t colormap, uint32_t out256[256]);

/* --- Version (v1.4) ------------------------------------------------------- */

/**
 * Build a version number from its parts, in the encoding `(major << 16) | (minor << 8) | patch`. The
 * macro is an integer constant expression, so `#if` can test it.
 */
#define CALIPER_EMBED_MAKE_VERSION(major, minor, patch) \
    (((major) << 16) | ((minor) << 8) | (patch))
/**
 * The major version of this header, which equals `CALIPER_EMBED_API_VERSION`. It bumps when a
 * struct grows.
 */
#define CALIPER_EMBED_VERSION_MAJOR CALIPER_EMBED_API_VERSION
/**
 * The minor version of this header. It counts the additive sections: v1.1 services, v1.2 tensor
 * views, v1.3 typed live views and v1.4 the version call.
 */
#define CALIPER_EMBED_VERSION_MINOR 4
/** The patch version of this header. It counts library fixes that change no declaration. */
#define CALIPER_EMBED_VERSION_PATCH 0
/**
 * The version of this header, built with `CALIPER_EMBED_MAKE_VERSION` from the major, minor and
 * patch parts.
 */
#define CALIPER_EMBED_VERSION                                   \
    CALIPER_EMBED_MAKE_VERSION(CALIPER_EMBED_VERSION_MAJOR,     \
                               CALIPER_EMBED_VERSION_MINOR,     \
                               CALIPER_EMBED_VERSION_PATCH)

/**
 * Return the version of the library that the process loaded, in the encoding of
 * `CALIPER_EMBED_VERSION`. An embedder runs on a library of the same major whose minor is at least
 * `CALIPER_EMBED_VERSION_MINOR`. The call needs no core and may run on any thread.
 */
uint32_t caliper_embed_api_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_EMBED_H */
