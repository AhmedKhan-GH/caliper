/* ===========================================================================
 * caliper/live_views.h — embed ABI v1.3: typed live views. Additive to embed.h.
 *
 * ONE view layer, TWO transports. A training script's show/points/series/
 * histogram calls produce GPU snapshots; a snapshot reaches the core either
 * from the in-process window's render thread (caliper.window(): the slots are
 * caliper-made shared MTLBuffers torch writes into) or from the shared-memory
 * session of another process (the Caliper app / caliper-viewer attaches:
 * src/host/live, caliper/live.h). Both end in caliper_core_live_apply(), so the
 * drawing code never knows where a snapshot came from.
 *
 * KINDS AND PAYLOADS (CaliperLiveKind values are shared with caliper/live.h):
 *   HEATMAP    data (H,W) f32, colormapped over [vmin,vmax]; AUTO flags take the
 *              ends from stats[0..1] (the producer's GPU aminmax; a non-finite
 *              AUTO range keeps the last finite one, a flat one widens by
 *              +-0.5). Drawn through the embed v1.2 TensorView of the SAME name
 *              (caliper_core_view_info/readback/count/remove keep working for
 *              it), path "compute-imported" for a METAL payload, "cpu-staged"
 *              for CPU. RAW / WAIT_GPU map to CALIPER_VIEW_FLAG_RAW / _WAIT_GPU.
 *   IMAGE      data (H,W,4) u8 RGBA, as the v1.2 image view ("blit-imported").
 *   POINTS     data (N,3) f32 positions (2-D callers pad z = 0 and set
 *              points_dims = 2); aux (N,) f32 per-point value or NULL (flat).
 *              label_count > 0: aux holds categorical labels 0..label_count-1,
 *              drawn with a fixed 16-colour palette (label L -> colour L mod 16,
 *              exact for integer labels 0..255; larger labels clamp to 255's);
 *              0: aux is colormapped through `colormap` over [vmin,vmax] (AUTO
 *              flags read stats[0..1]).
 *              stats[2..7] may carry the bbox (xmin,xmax,ymin,ymax,zmin,zmax)
 *              for auto-fit (stats_count >= 8); without it the fit scans the
 *              CPU-visible payload. aux lives on the same device as data.
 *              METAL: the GPU redraws the view straight from the payload buffer
 *              (path "points-imported", zero-copy) whenever the payload, the
 *              camera or the window size changes, so the CURRENT payload stays
 *              CLAIMED for as long as it is current (see read_done). CPU: the
 *              bytes are staged into a view-owned buffer at apply
 *              ("points-cpu-staged"; free on return). Needs the Metal renderer.
 *   SERIES     data (M,) f32 ring of capacity M; values i in [total - count,
 *              total) live at index i mod M, count = min(total, M), total =
 *              series_total. The viewer copies the window it draws. Only the
 *              newest `total` is trusted; a value the producer overwrites while
 *              the viewer copies the oldest sample is a one-sample glitch at
 *              the old end, by design (no lock between processes).
 *   HISTOGRAM  data (B,) f32 bin counts over [stats[0], stats[1]] (lo, hi;
 *              missing/non-finite stats: bin indices [0, B)). Copied at apply;
 *              the payload is reclaimable at once.
 * aux/aux_offset/label_count/points_dims are read for POINTS only, series_total
 * for SERIES only. f32 payloads (all kinds) and every METAL payload need a
 * data_offset that is a multiple of 4.
 *
 * PAYLOAD MEMORY. data->device is CPU or METAL. METAL: data->data is an in-
 * process id<MTLBuffer> read at data_offset (DLPack byte_offset + slot offset);
 * CPU: bytes at data->data + data_offset. Every payload a producer hands over is
 * SHARED-storage memory the CPU can read too ([buffer contents] for METAL):
 * the shm segments and the window transport's caliper-made buffers both are.
 * The view layer reads series/histogram values and hover values from that
 * CPU side; a METAL payload whose contents are not CPU-visible (a private
 * torch buffer) is refused for SERIES/HISTOGRAM and loses hover values for
 * HEATMAP (the colormap kernel still runs; nothing is copied to the host).
 * Heatmap hover reads the slot after its claim ended, so it may show a value of
 * a newer snapshot of the same view (display aid only; CPU payloads: no hover).
 *
 * SYNC. The producer applies a snapshot only after the GPU work that wrote it
 * completed (torch.mps.Event polled, never waited on) and never rewrites a
 * slot that the core may still read. The core's GPU reads are never waited on
 * either: read_done() polls the renderer's completion handles, so a transport
 * can release slots without blocking the frame thread. THE CLAIM RULE PER KIND
 * (read_done() == 1 means the slots below are free to rewrite):
 *   HEATMAP/IMAGE  each apply's colormap/blit reads the payload once into the
 *                  view's texture. read_done = 1 once the device update of
 *                  EVERY apply so far completed: the LAST payload is free too.
 *                  CPU payloads are copied at apply (always 1).
 *   POINTS         the current payload is re-read on every redraw and stays
 *                  claimed while current, whatever read_done says. read_done
 *                  = 1 once the last frame that drew every SUPERSEDED payload
 *                  retired (their imports are released then). So: keep the
 *                  current slot, free the previous ones when read_done is 1.
 *                  (1 right after the first apply: nothing superseded yet.)
 *   SERIES/HISTOGRAM  copied at apply: always 1, nothing is ever claimed.
 * WAIT_GPU (tests): heatmap/image block until the update completed; points
 * block until the superseded payloads' last frames retired (so read_done is 1
 * on return); series/histogram ignore it. RAW applies to heatmap/image only.
 *
 * PAUSE is ONE flag per core and freezes the display only: apply() still runs
 * every gate, then accepts and drops the snapshot (info.dropped += 1; a first
 * apply while paused creates an empty view) so claims progress and training is
 * never slowed. Each typed window shows a Pause/Resume button for that flag.
 *
 * KIND CHANGE: an apply with the same name and another kind rebuilds the view
 * (counters restart; a failed rebuild keeps the old view); heatmap <-> image
 * stays in place (the v1.2 view recreates its texture). A POINTS/SERIES/
 * HISTOGRAM apply refuses a name owned by a plain v1.2 view (caliper.show).
 *
 * THREADING: frame-thread only (same thread as caliper_core_frame), between
 * frames, after attach_canvas. Views are destroyed by caliper_core_shutdown.
 * ------------------------------------------------------------------------- */
#ifndef CALIPER_LIVE_VIEWS_H
#define CALIPER_LIVE_VIEWS_H

#include <caliper/embed.h>
#include <caliper/live.h>     /* CaliperLiveKind */

#ifdef __cplusplus
extern "C" {
#endif

#define CALIPER_EMBED_HAVE_LIVE_VIEWS 1

enum {
    CALIPER_LIVE_APPLY_RAW       = 1u << 0,  /* heatmap/image: draw 1:1 at canvas (0,0), no chrome (tests) */
    CALIPER_LIVE_APPLY_AUTO_VMIN = 1u << 1,  /* vmin = stats[0] */
    CALIPER_LIVE_APPLY_AUTO_VMAX = 1u << 2,  /* vmax = stats[1] */
    CALIPER_LIVE_APPLY_WAIT_GPU  = 1u << 3   /* block until the GPU read completed (tests only) */
};

typedef struct CaliperLiveSnapshot {
    size_t               struct_size;   /* = sizeof(CaliperLiveSnapshot) */
    const char*          name;          /* view id; same name = update in place */
    int32_t              kind;          /* CaliperLiveKind */
    const CaliperTensor* data;          /* payload, see the kind table */
    uint64_t             data_offset;   /* byte offset into data->data */
    const CaliperTensor* aux;           /* POINTS: (N,) f32 or NULL */
    uint64_t             aux_offset;
    const float*         stats;         /* CPU-readable producer stats, valid for the call; may be NULL */
    int32_t              stats_count;   /* floats in stats */
    int32_t              colormap;      /* CaliperColormap */
    float                vmin, vmax;    /* fixed ends (AUTO flags override) */
    uint32_t             flags;         /* CALIPER_LIVE_APPLY_* */
    uint64_t             series_total;  /* SERIES: values appended so far */
    int32_t              points_dims;   /* POINTS: 2 (pan/zoom) or 3 (orbit) */
    int32_t              label_count;   /* POINTS: categorical classes in aux, 0 = continuous */
    uint64_t             seq;           /* producer's publish counter for this view */
    uint64_t             stamp_ns;      /* producer CLOCK_UPTIME_RAW ns at snapshot time (0 = unknown) */
} CaliperLiveSnapshot;

typedef struct CaliperLiveViewInfo {
    size_t      struct_size;
    int32_t     kind;
    const char* path;           /* renderer path of the last apply (heatmap/image: as v1.2; points: "points-imported" / "points-cpu-staged"; series/histogram: "cpu-read"); valid until the next core call */
    int32_t     width, height;  /* texture size for texture-backed kinds (points: the view's render target, 0 until the first frame drew it), else 0 */
    uint64_t    updates;        /* successful applies (not counting dropped ones) */
    uint64_t    creates;        /* heatmap/image: v1.2 texture (re)creations (shape/dtype changes); points: render-target (re)creations (first frame, window resizes); series/histogram: 1 */
    uint64_t    dropped;        /* applies ignored while paused */
    uint64_t    seq;            /* seq of the last applied snapshot */
    uint64_t    stamp_ns;       /* stamp of the last applied snapshot */
    double      last_apply_ms;  /* CPU wall time of the last apply */
    double      latency_ms;     /* (apply time - stamp_ns) of the last apply, 0 when stamp unknown */
    int32_t     imported;       /* 1 if the GPU reads the last payload's MTLBuffer in place (heatmap/image/points METAL); series/histogram: 0 (CPU copy) */
    int32_t     read_done;      /* 1 if the GPU read of the previously applied payload has retired */
    int64_t     count;          /* SERIES: values held; HISTOGRAM: bins; POINTS: N; else 0 */
} CaliperLiveViewInfo;

/* Create or update the named typed view. 1 on success, 0 with last_error. */
int  caliper_core_live_apply(CaliperCore* core, const CaliperLiveSnapshot* snap);
int  caliper_core_live_view_remove(CaliperCore* core, const char* name);
int  caliper_core_live_view_count(CaliperCore* core);
int  caliper_core_live_view_info(CaliperCore* core, const char* name, CaliperLiveViewInfo* out);
/* Non-blocking claim probe (the per-kind rule is in the SYNC section above):
 * heatmap/image: every payload applied so far has been read (the last too);
 * points: every SUPERSEDED payload's last frame retired (the current payload
 * stays claimed while current); 1 for an unknown view and for kinds that copy
 * at apply. Never waits. */
int  caliper_core_live_view_read_done(CaliperCore* core, const char* name);
/* Texture-backed kinds (heatmap, image, points): the view texture as tightly
 * packed RGBA8 (width*height*4 bytes). 1 on success. */
int  caliper_core_live_view_readback(CaliperCore* core, const char* name, void* buf, size_t buf_bytes);
/* SERIES / HISTOGRAM: copy the values the view currently holds (series: oldest
 * first; histogram: bin counts) into out[0..min(cap, available)), *count =
 * values available (out may be NULL with cap 0 to count). 0 for other kinds. */
int  caliper_core_live_view_values(CaliperCore* core, const char* name, float* out, size_t cap, size_t* count);
/* Freeze the display of every typed view of this core (applies are validated,
 * accepted and dropped). 1 = paused. */
void caliper_core_live_set_paused(CaliperCore* core, int paused);
int  caliper_core_live_paused(CaliperCore* core);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* CALIPER_LIVE_VIEWS_H */
