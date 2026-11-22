// live_catalog.h: the views of the attached live session or stream as sources
// of the builder's catalog (design/catalog.md, the kind `live`), and their
// newest snapshots for the session's resolve (caliper_builder::
// CatalogDataSource). Host-internal; built only with CALIPER_WITH_BUILDER.
//
// One source per view: the id is `live:<view name>`, the title the name, the
// origin {session, view, view_type}, `wire` true, and once a snapshot has
// been applied the dtype and shape of the newest one, its size in `bytes`,
// `rate_hz` (new snapshots per second over the last 5 s, from two on) and
// `updated_at` (when the newest one arrived here). The payload, by kind:
//
//   heatmap     f32 [H, W]; a tiled heatmap (wire protocol v3, "Payload
//               axes") is cut back into its cells, f32 [cells, rows, cols],
//               so axis 0 is the tile dim (a layer, say) and the mosaic's
//               NaN fill is dropped
//   image       u8 [H, W, 4]
//   points      f32 [N, 3] (positions; the aux values are not given)
//   series      f32 [M], the values the view holds, oldest first
//   histogram   f32 [B], the bin counts
//
// A view that is announced but has drawn nothing yet has a null dtype and
// shape and answers catalog.no_data and resolve.unavailable.
//
// THE HAND-OVER. update() runs on the frame thread once per frame, after the
// live poll (Attachment::poll()). The live core has then applied the newest
// completed snapshot of each view, and that snapshot's slot stays claimed
// until the next poll takes a newer one (live_source.h, the claim rules), so
// update() reads it in place without a lock and without ever making the
// producer wait. It copies a view's payload into a new owned TensorData when
// a newer snapshot was applied and the view is wanted (a reader asked for it
// within `wanted_ns`), or when the last copy is older than `idle_copy_ns`, or
// when there is none; a payload past `max_view_bytes` is not copied. Readers
// on any thread get the newest copy under a short mutex and share it, never
// changed afterwards; a copy is at most one frame older than the view while
// the view is wanted. The copy costs one memcpy of the payload on the frame
// thread (16 MiB for a 2048 x 2048 heatmap) once per new snapshot of a
// wanted view.
//
// generation() is the copy's number (0 before the first copy): it grows by one
// with every copy, across all views. revision() grows when a view appears or
// goes, when its kind, dtype or shape changes (for a series, which grows with
// its data, only its rank), and on attach and detach, not when snapshots
// arrive. After a detach (update with no attachment) every
// view is gone: describe() finds nothing and read() answers
// resolve.unavailable. A session that ended keeps its views and their last
// copies, as the live page keeps its last frames.
//
// Thread-safe (update() from one thread at a time); no function throws.
#pragma once

#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/loaders/tensor.h>
#include <caliper_builder/session/resolve.h>

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct CaliperCore;
namespace caliper_builder {
class EventSink;
class StepHistory;
}  // namespace caliper_builder
namespace caliper_live { class Attachment; }

namespace caliper_host {

class LiveCatalog : public caliper_builder::CatalogProvider, public caliper_builder::CatalogDataSource {
public:
    struct Options {
        std::uint64_t max_view_bytes = std::uint64_t(64) << 20;      // a larger payload is not copied
        std::uint64_t wanted_ns = 2000000000ull;                     // a read keeps a view fresh this long
        std::uint64_t idle_copy_ns = 1000000000ull;                  // an unwanted view is copied this often
        std::uint64_t rate_window_ns = 5000000000ull;                // rate_hz is measured over this window
        std::function<std::uint64_t()> clock;                        // monotonic ns; empty: the steady clock
        // The kept steps of the open spec (design/instrument-interface.md,
        // Step History): each snapshot of a view whose steps the history
        // keeps is copied and appended at its step. Null: nothing is kept.
        caliper_builder::StepHistory* history = nullptr;
        // The run's step now, for a snapshot that carries no step stamp;
        // empty or nullopt: such a snapshot is not kept.
        std::function<std::optional<std::int64_t>()> run_step;
        // Where a `cursor` event goes when a new step moves a held cursor
        // that fell out of every source (the change "clamped").
        caliper_builder::EventSink* events = nullptr;
    };
    LiveCatalog();
    explicit LiveCatalog(Options options);

    // Frame thread, once per frame after the live poll: the views of the
    // attachment, or none when it is not attached.
    void update(const caliper_live::Attachment& attachment);
    // The same for any core whose live views to read (null: detached), with
    // the session's label for the origin.
    void update(CaliperCore* core, const std::string& session);

    // CatalogProvider
    caliper_builder::CatalogListResult list(const caliper_builder::CatalogFilter& filter) override;
    caliper_builder::CatalogDescribeResult describe(const std::string& id) override;
    caliper_builder::CatalogSampleResult sample(const std::string& id,
                                                const caliper_builder::CatalogSampleRequest& request) override;
    std::uint64_t revision() const override;

    // CatalogDataSource
    std::uint64_t generation(const std::string& id) override;
    caliper_builder::ServedData read(const std::string& id, const caliper_builder::ServedRequest& request) override;

    // Copies made since construction (for tests and measurements).
    std::uint64_t copies() const;

    // The step of the newest snapshot of view `id` (a catalog id or a view
    // name): its step stamp, else the run's step when it arrived; nullopt
    // before the first snapshot or when neither was known.
    std::optional<std::int64_t> step(const std::string& id) const;
    // The names of a live view's dims, one per dim of its shape, from the
    // stream's payload axes; empty for an id that is not a view, a view
    // without a payload yet, or one whose axes do not match its shape.
    std::vector<std::string> dim_names(const std::string& id) const;

    // What update() learned of one view: for tests, a stand-in producer
    // hands the same to apply() without a core.
    struct ViewFrame {
        std::string name;
        std::uint32_t kind = 0;             // CaliperLiveKind
        std::uint64_t updates = 0;          // snapshots applied so far
        bool has_payload = false;           // a snapshot was applied
        caliper_builder::DType dtype = caliper_builder::DType::f32;
        std::vector<std::int64_t> shape;    // of the payload as given (after untiling)
        const std::uint8_t* data = nullptr; // the payload as the view holds it; null: not readable on the CPU
        std::uint64_t payload_bytes = 0;    // bytes at `data`
        // A tiled heatmap: the mosaic's cells, rows and columns of cells, and
        // the rows and columns of one cell; cells == 0 when not tiled.
        std::int64_t cells = 0, tile_rows = 0, tile_cols = 0, cell_h = 0, cell_w = 0;
        // The names of the payload's dims, one per dim of `shape`, from the
        // stream's payload axes (wire protocol v3); empty when it gives none.
        std::vector<std::string> dims;
        // The step stamp of the newest snapshot (the `step` timeline of wire
        // protocol 3.2); nullopt when the producer sent none.
        std::optional<std::int64_t> step;
        // Why the view updates less often than it asks: the script's shared
        // cadence of its snapshots thinned it (cadence_note()); empty otherwise.
        std::string note;
    };
    // The frame-thread step that update() ends with: `views` is the full list,
    // `attached` false means detached.
    void apply(bool attached, const std::string& session, const std::vector<ViewFrame>& views);

private:
    struct View {
        std::string name;
        std::uint32_t kind = 0;
        bool has_payload = false;
        caliper_builder::DType dtype = caliper_builder::DType::f32;
        std::vector<std::int64_t> shape;
        std::vector<std::string> dims;   // ViewFrame::dims
        std::uint64_t bytes = 0;
        std::optional<double> rate_hz;
        std::optional<std::string> updated_at;
        caliper_builder::TensorData copy;   // the newest copy; invalid before the first
        std::uint64_t generation = 0;       // of the copy
        bool copy_too_large = false;
        bool readable = true;               // the payload is readable on the CPU
        std::uint64_t wanted_until = 0;     // written by readers
        // frame thread only
        std::uint64_t updates_seen = 0;
        std::uint64_t copied_updates = 0;
        std::uint64_t copied_at = 0;
        std::deque<std::uint64_t> arrivals;
        std::optional<std::int64_t> step;   // of the newest snapshot
        std::string note;                   // ViewFrame::note, as a warning of the entry
    };
    std::uint64_t now() const;
    caliper_builder::CatalogEntry entry_of(const View& v) const;
    // Appends the copies of the views whose steps the history keeps, at
    // their steps, outside the catalog's lock.
    void keep_steps(const std::vector<std::pair<const ViewFrame*, caliper_builder::TensorData>>& made,
                    const std::map<std::string, std::int64_t>& steps);
    bool find_copy(const std::string& id, View* out, std::string* why);

    Options options_;
    mutable std::mutex mutex_;
    std::map<std::string, View> views_;   // by name
    bool attached_ = false;
    std::string session_;
    std::uint64_t revision_ = 1;
    std::uint64_t next_generation_ = 1;
    std::uint64_t copies_ = 0;
};

}  // namespace caliper_host
