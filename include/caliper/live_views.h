/* ===========================================================================
 * caliper/live_views.h - embed ABI v1.3, typed live views. Additive to embed.h.
 *
 * One view layer serves two transports. A snapshot reaches the core either from the in-process
 * window of a training script or from the shared-memory session of another process. Both end in
 * caliper_core_live_apply(), so the drawing code never knows where a snapshot came from.
 * The docs page "Embedding API" explains the claim rule and the threading of these calls.
 * ===========================================================================*/
#ifndef CALIPER_LIVE_VIEWS_H
#define CALIPER_LIVE_VIEWS_H

#include <caliper/embed.h>
#include <caliper/live.h>     /* CaliperLiveKind */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Defined as 1 when the header declares the typed live view calls, so that a host can test for
 * them.
 */
#define CALIPER_EMBED_HAVE_LIVE_VIEWS 1

enum {
    /**
     * Bit flag for heatmap and image views to draw the texture 1:1 at the canvas origin with no
     * window chrome, for tests.
     */
    CALIPER_LIVE_APPLY_RAW       = 1u << 0,
    /** Bit flag to take `vmin` from `stats[0]`. */
    CALIPER_LIVE_APPLY_AUTO_VMIN = 1u << 1,
    /** Bit flag to take `vmax` from `stats[1]`. */
    CALIPER_LIVE_APPLY_AUTO_VMAX = 1u << 2,
    /** Bit flag to block until the GPU read completed, for tests only. */
    CALIPER_LIVE_APPLY_WAIT_GPU  = 1u << 3,
    /**
     * Bit flag that marks `stamp_ns` as mapped from the clock of the producer, as for a network
     * source. The latency is then the delay above the fastest recent message and excludes a constant
     * transit time, and the view labels it "rel lat".
     */
    CALIPER_LIVE_APPLY_STAMP_RELATIVE = 1u << 4,
    /** Bit flag that marks `step` of the snapshot as valid. Without it the snapshot has no step. */
    CALIPER_LIVE_APPLY_HAS_STEP = 1u << 5
};

/**
 * One snapshot of a typed live view, which `caliper_core_live_apply` takes.
 *
 * The `kind` selects the payload, and the values of `CaliperLiveKind` are shared with
 * `caliper/live.h`. A `HEATMAP` payload is `data` of shape `(H,W)` in f32, colormapped over
 * `[vmin, vmax]`. The `AUTO` flags take the ends from `stats[0]` and `stats[1]`, which hold the
 * minimum and maximum of the producer. A non-finite automatic range keeps the last finite one, and
 * a flat range widens by 0.5 on each side. The core draws a heatmap through the tensor view of the
 * same name, so `caliper_core_view_info`, `caliper_core_view_readback`, `caliper_core_view_count`
 * and `caliper_core_view_remove` keep working for it. Its renderer path is `compute-imported` for
 * a METAL payload and `cpu-staged` for a CPU payload.
 *
 * An `IMAGE` payload is `data` of shape `(H,W,4)` in u8 RGBA, drawn like the image view of embed.h
 * with the path `blit-imported`. A `POINTS` payload is `data` of shape `(N,3)` in f32 positions.
 * A 2-D caller pads z with 0 and sets `points_dims` to 2. The optional `aux` of shape `(N,)` in
 * f32 carries one value per point and lives on the same device as `data`. With `label_count`
 * above 0, `aux` holds categorical labels from 0 to `label_count - 1`, drawn with a fixed palette
 * of 16 colors where label L takes color L mod 16. This is exact for integer labels up to 255, and
 * larger labels clamp to 255. With `label_count` 0, `aux` is colormapped over `[vmin, vmax]`.
 * Without `aux`, the points have one flat color. `stats[2..7]` may carry the bounding box (xmin,
 * xmax, ymin, ymax, zmin, zmax) for the automatic fit when `stats_count` is at least 8. Without
 * it, the fit scans the CPU-visible payload.
 *
 * For a METAL points payload, the GPU redraws the view straight from the payload buffer, which is
 * zero-copy with the path `points-imported`. It redraws whenever the payload, the camera or the
 * window size changes, so the current payload stays claimed while it is current. A CPU payload is
 * staged into a buffer that the view owns at apply, with the path `points-cpu-staged`, and the
 * caller may free it on return. Points views need the Metal renderer.
 *
 * A `SERIES` payload is `data` of shape `(M,)` in f32, a ring of capacity M. The values with index
 * i from `total - count` up to but excluding `total` live at index i mod M, where
 * `count = min(total, M)` and `total` is `series_total`. The viewer copies the window that it
 * draws. A value that the producer overwrites while the viewer copies the oldest sample is a
 * one-sample glitch at the old end, by design, because the processes share no lock. A `HISTOGRAM`
 * payload is `data` of shape `(B,)` in f32 bin counts over `[stats[0], stats[1]]`. Without finite
 * stats, the bins are numbered from 0. The core copies the payload at apply, so it is
 * reclaimable at once.
 *
 * The device of `data` is CPU or METAL. A METAL payload is an in-process `id<MTLBuffer>` that the
 * core reads at `data_offset`, which is the DLPack byte offset plus the slot offset. A CPU payload
 * is the bytes at `data->data + data_offset`. Every payload that a producer hands over is
 * shared-storage memory that the CPU can read too. The view layer reads series and histogram
 * values and hover values from that CPU side. It refuses a METAL payload whose contents the CPU
 * cannot see, such as a private torch buffer, for series and histogram views. For a heatmap, such
 * a payload still draws but loses its hover values. A heatmap hover reads the slot after its claim
 * ended, so it may show a value of a newer snapshot of the same view. This is a display aid only.
 * The core copies a CPU payload at apply for hover, because the caller may free its bytes on
 * return. An f32 payload of any kind and every METAL payload need a `data_offset` that is a
 * multiple of 4.
 */
typedef struct CaliperLiveSnapshot {
    /** `sizeof(CaliperLiveSnapshot)`, as the caller built it. */
    size_t               struct_size;
    /** The id of the view. The same name updates the existing view in place. */
    const char*          name;
    /** The `CaliperLiveKind` of the payload. */
    int32_t              kind;
    /** The payload tensor. Its shape and type depend on the kind. */
    const CaliperTensor* data;
    /** The byte offset into `data->data`. */
    uint64_t             data_offset;
    /**
     * For `POINTS` only, a tensor of shape `(N,)` in f32 with one value or label per point, or
     * `NULL`.
     */
    const CaliperTensor* aux;
    /** For `POINTS` only, the byte offset into `aux->data`. */
    uint64_t             aux_offset;
    /** CPU-readable producer statistics, valid for the call, or `NULL`. */
    const float*         stats;
    /** The number of floats in `stats`. */
    int32_t              stats_count;
    /** The `CaliperColormap` of a heatmap or of a points view with continuous values. */
    int32_t              colormap;
    /** The fixed ends of the mapping range. The `AUTO` flags override them. */
    float                vmin, vmax;
    /** A bit set of the `CALIPER_LIVE_APPLY_*` flags. */
    uint32_t             flags;
    /** For `SERIES` only, the number of values appended so far. */
    uint64_t             series_total;
    /** For `POINTS` only, 2 for a view with pan and zoom or 3 for an orbit camera. */
    int32_t              points_dims;
    /**
     * For `POINTS` only, the number of categorical classes in `aux`, or 0 for continuous values.
     */
    int32_t              label_count;
    /** The publish counter of the producer for this view. */
    uint64_t             seq;
    /**
     * The time of the snapshot in the clock of the viewer, which is `CLOCK_UPTIME_RAW` on Apple.
     * A producer on the same machine stamps with that clock, and a network source maps the stamps
     * of the producer into it. The value 0 means unknown.
     */
    uint64_t             stamp_ns;
    /**
     * The step stamp of the snapshot, valid only with `CALIPER_LIVE_APPLY_HAS_STEP` in `flags`.
     * `caliper_core_live_view_info` reports it as `step`.
     */
    int64_t              step;
} CaliperLiveSnapshot;

/** The measured state of one typed live view, which `caliper_core_live_view_info` fills. */
typedef struct CaliperLiveViewInfo {
    /** `sizeof(CaliperLiveViewInfo)`, as the caller built it. */
    size_t      struct_size;
    /** The `CaliperLiveKind` of the view. */
    int32_t     kind;
    /**
     * The renderer path of the last apply. A heatmap or image view reports the path of the tensor
     * view, a points view reports `points-imported` or `points-cpu-staged`, and a series or
     * histogram view reports `cpu-read`. The string stays valid until the next core call.
     */
    const char* path;
    /**
     * The texture size of a texture-backed kind, or 0 for the others. For a points view it is the
     * render target, which is 0 until the first frame draws it.
     */
    int32_t     width, height;
    /** The number of successful applies, not counting dropped ones. */
    uint64_t    updates;
    /**
     * The number of texture creations. For a heatmap or image view it counts the texture
     * creations of the tensor view, which a shape or dtype change causes. For a points view it counts
     * the render target creations, at the first frame and at each window resize. For a series or
     * histogram view it is 1.
     */
    uint64_t    creates;
    /** The number of applies that the core ignored while paused. */
    uint64_t    dropped;
    /** The `seq` of the last applied snapshot. */
    uint64_t    seq;
    /** The `stamp_ns` of the last applied snapshot. */
    uint64_t    stamp_ns;
    /** The CPU wall time of the last apply in milliseconds. */
    double      last_apply_ms;
    /**
     * The apply time minus `stamp_ns` for the last apply in milliseconds, or 0 when the stamp is
     * unknown.
     */
    double      latency_ms;
    /**
     * 1 when the GPU reads the last payload `MTLBuffer` in place, as for a METAL heatmap, image or
     * points view, or 0 for a CPU copy such as a series or histogram.
     */
    int32_t     imported;
    /** 1 when the GPU read of the previously applied payload has retired, otherwise 0. */
    int32_t     read_done;
    /** For `SERIES`, the values held. For `HISTOGRAM`, the bins. For `POINTS`, N. Otherwise 0. */
    int64_t     count;
    /** The step stamp of the last applied snapshot, or INT64_MIN when it had none. */
    int64_t     step;
} CaliperLiveViewInfo;

/**
 * Create the typed view named `snap->name`, or update it when it exists. Return 1 on success, or 0
 * with the last error set. The producer applies a snapshot only after the GPU work that wrote it
 * completed, and never rewrites a slot that the core may still read. An apply with the same name
 * and another kind rebuilds the view and restarts its counters, and a failed rebuild keeps the old
 * view. A switch between heatmap and image stays in place. A points, series or histogram apply
 * refuses a name that a plain tensor view owns.
 */
int  caliper_core_live_apply(CaliperCore* core, const CaliperLiveSnapshot* snap);
/**
 * Remove the typed view named `name`. Return 1 if the view existed and was removed, otherwise 0.
 */
int  caliper_core_live_view_remove(CaliperCore* core, const char* name);
/** Return the number of typed views. */
int  caliper_core_live_view_count(CaliperCore* core);
/**
 * Fill `out` with the measured state of the view named `name`. Return 1 on success, otherwise 0.
 */
int  caliper_core_live_view_info(CaliperCore* core, const char* name, CaliperLiveViewInfo* out);
/**
 * Probe without blocking whether a transport may rewrite the slots of the view named `name`.
 * Return 1 when it may, and never wait. The rule depends on the kind. For a heatmap or image view,
 * the result is 1 once the device update of every apply so far has completed, so the last payload
 * is free too. For a points view, the current payload stays claimed while it is current, because
 * every redraw reads it. The result is then 1 once the last frame that drew every superseded
 * payload has retired, which is also the case right after the first apply. A transport keeps the
 * current slot and frees the previous ones when the result is 1. A series or histogram view copies
 * at apply and claims nothing, and an unknown view reports 1.
 */
int  caliper_core_live_view_read_done(CaliperCore* core, const char* name);
/**
 * Copy the texture of a texture-backed view (heatmap, image or points) named `name` to `buf` as
 * tightly packed RGBA8 of `width*height*4` bytes. `buf_bytes` is the size of `buf`. Return 1 on
 * success, otherwise 0.
 */
int  caliper_core_live_view_readback(CaliperCore* core, const char* name, void* buf, size_t buf_bytes);
/**
 * Copy the values that a series or histogram view currently holds into `out`, oldest first for a
 * series and as bin counts for a histogram. The call copies at most `cap` values and sets `*count`
 * to the number available, so `out` may be `NULL` with `cap` 0 to count. Return 1 on success, and
 * 0 for other kinds.
 */
int  caliper_core_live_view_values(CaliperCore* core, const char* name, float* out, size_t cap, size_t* count);
/**
 * Freeze or resume the display of every typed view of this core. One flag serves the whole core.
 * While it is set, each apply still runs every gate and then the core accepts and drops the
 * snapshot, which adds to `dropped`. A first apply while paused creates an empty view. Claims keep
 * progressing, so training is never slowed. Each typed window shows a Pause and Resume button for
 * this flag.
 */
void caliper_core_live_set_paused(CaliperCore* core, int paused);
/** Return 1 when the display is paused, otherwise 0. */
int  caliper_core_live_paused(CaliperCore* core);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_LIVE_VIEWS_H */
