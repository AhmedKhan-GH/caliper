#include "catalog_providers/live_catalog.h"

#include "catalog_providers/catalog_util.h"

#include <caliper/live.h>
#include <caliper/live_views.h>
#include <caliper_builder/catalog/tensor_sample.h>

#include "embed/embed_core.h"
#include "live/live_attach.h"
#include "live/live_views_state.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>

namespace caliper_host {

using namespace caliper_builder;

namespace {

const char* view_type_of(std::uint32_t kind) {
    switch (kind) {
        case CALIPER_LIVE_KIND_HEATMAP: return "heatmap";
        case CALIPER_LIVE_KIND_IMAGE: return "image";
        case CALIPER_LIVE_KIND_POINTS: return "points";
        case CALIPER_LIVE_KIND_SERIES: return "series";
        case CALIPER_LIVE_KIND_HISTOGRAM: return "histogram";
        default: return nullptr;
    }
}

std::uint64_t shape_bytes(DType dtype, const std::vector<std::int64_t>& shape) {
    std::uint64_t numel = 0, nbytes = 0;
    return check_shape(shape, dtype, &numel, &nbytes).ok() ? nbytes : 0;
}

CatalogDtype catalog_dtype_of(DType dtype) {
    return catalog_dtype_from_name(dtype_name(dtype)).value_or(CatalogDtype::f32);
}

// The id's view name, or nothing for an id of another kind.
std::optional<std::string> view_name(const std::string& id) {
    const CatalogIdParse parsed = parse_catalog_id(id);
    if (parsed.error || parsed.kind != CatalogKind::live) return std::nullopt;
    return parsed.name;
}

// An owned copy of a view's payload, with a tiled heatmap cut into its cells.
TensorResult copy_payload(const LiveCatalog::ViewFrame& f) {
    TensorResult out = TensorData::allocate(f.dtype, f.shape);
    if (out.error) return out;
    std::uint8_t* dst = out.tensor.mutable_data();
    const std::uint64_t n = out.tensor.nbytes();
    if (!dst || n == 0) return out;
    if (f.cells == 0) {
        if (f.payload_bytes < n) {
            out.error = Error("resolve.unavailable", "The live view's payload is smaller than its shape.");
            return out;
        }
        std::memcpy(dst, f.data, n);
        return out;
    }
    // A tiled heatmap: cell c sits at row c / tile_cols and column c % tile_cols
    // of the mosaic, each cell_h x cell_w f32 values.
    const std::int64_t mosaic_w = f.tile_cols * f.cell_w;
    if (f.payload_bytes < static_cast<std::uint64_t>(f.tile_rows * f.cell_h * mosaic_w) * 4u) {
        out.error = Error("resolve.unavailable", "The live view's mosaic is smaller than its tiles.");
        return out;
    }
    const float* src = reinterpret_cast<const float*>(f.data);
    float* d = reinterpret_cast<float*>(dst);
    for (std::int64_t c = 0; c < f.cells; ++c) {
        const std::int64_t r0 = (c / f.tile_cols) * f.cell_h;
        const std::int64_t c0 = (c % f.tile_cols) * f.cell_w;
        for (std::int64_t y = 0; y < f.cell_h; ++y) {
            std::memcpy(d, src + (r0 + y) * mosaic_w + c0, static_cast<std::size_t>(f.cell_w) * 4u);
            d += f.cell_w;
        }
    }
    return out;
}

LiveCatalog::ViewFrame frame_shape_of(CaliperCore* core, const caliper_live_views::View& v);

// What the core's live views hold for one view, for apply(): its payload and
// shape, and the names of its dims where the stream's payload axes give one
// axis per dim of that shape, with the same sizes.
LiveCatalog::ViewFrame frame_of(CaliperCore* core, const caliper_live_views::View& v) {
    LiveCatalog::ViewFrame f = frame_shape_of(core, v);
    const auto& axes = v.map.axes;
    if (f.has_payload && !axes.empty() && axes.size() == f.shape.size()) {
        bool named = false, sized = true;
        for (size_t i = 0; i < axes.size(); ++i) {
            named = named || !axes[i].dim.empty();
            sized = sized && axes[i].size == f.shape[i];
        }
        if (named && sized)
            for (const auto& a : axes) f.dims.push_back(a.dim);
    }
    return f;
}

LiveCatalog::ViewFrame frame_shape_of(CaliperCore* core, const caliper_live_views::View& v) {
    LiveCatalog::ViewFrame f;
    f.name = v.name;
    f.kind = static_cast<std::uint32_t>(v.kind);
    f.updates = v.updates;
    if (v.updates == 0) return f;
    switch (v.kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
        case CALIPER_LIVE_KIND_IMAGE: {
            CaliperLiveViewInfo info{};
            info.struct_size = sizeof info;
            if (!caliper_core_live_view_info(core, v.name.c_str(), &info) || info.width <= 0 || info.height <= 0)
                return f;
            const std::int64_t h = info.height, w = info.width;
            f.has_payload = true;
            f.data = v.tex_cpu;
            f.payload_bytes = static_cast<std::uint64_t>(h * w) * 4u;
            if (v.kind == CALIPER_LIVE_KIND_IMAGE) {
                f.dtype = DType::u8;
                f.shape = {h, w, 4};
                return f;
            }
            f.dtype = DType::f32;
            f.shape = {h, w};
            const caliper_live::PayloadMap& m = v.map;
            if (m.tile_rows > 0 && m.tile_cols > 0 && m.axes.size() == 3) {
                const std::int64_t cells = m.axes[0].size, ch = m.axes[1].size, cw = m.axes[2].size;
                if (cells > 0 && ch > 0 && cw > 0 && cells <= m.tile_rows * m.tile_cols && m.tile_rows * ch == h &&
                    m.tile_cols * cw == w) {
                    f.cells = cells;
                    f.tile_rows = m.tile_rows;
                    f.tile_cols = m.tile_cols;
                    f.cell_h = ch;
                    f.cell_w = cw;
                    f.shape = {cells, ch, cw};
                }
            }
            return f;
        }
        case CALIPER_LIVE_KIND_POINTS: {
            f.has_payload = v.have_cur;
            f.dtype = DType::f32;
            f.shape = {static_cast<std::int64_t>(v.cur.n), 3};
            f.data = v.have_cur ? v.cur.pos_cpu : nullptr;
            f.payload_bytes = v.cur.n * 12u;
            return f;
        }
        case CALIPER_LIVE_KIND_SERIES:
        case CALIPER_LIVE_KIND_HISTOGRAM: {
            f.has_payload = true;
            f.dtype = DType::f32;
            f.shape = {static_cast<std::int64_t>(v.values.size())};
            f.data = reinterpret_cast<const std::uint8_t*>(v.values.data());
            f.payload_bytes = v.values.size() * 4u;
            return f;
        }
        default: return f;
    }
}

}  // namespace

LiveCatalog::LiveCatalog() : LiveCatalog(Options{}) {}
LiveCatalog::LiveCatalog(Options options) : options_(std::move(options)) {}

std::uint64_t LiveCatalog::now() const {
    if (options_.clock) return options_.clock();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

void LiveCatalog::update(const caliper_live::Attachment& attachment) {
    update(attachment.attached() ? attachment.core() : nullptr, attachment.status().session);
}

void LiveCatalog::update(CaliperCore* core, const std::string& session) {
    if (!core) {
        apply(false, {}, {});
        return;
    }
    std::vector<ViewFrame> frames;
    std::set<std::string> seen;
    if (core->live_views) {
        for (const auto& v : core->live_views->views) {
            if (!v || !seen.insert(v->name).second) continue;
            frames.push_back(frame_of(core, *v));
        }
    }
    // Views the attached session announced that have drawn nothing yet.
    for (int i = 0;; ++i) {
        CaliperLiveViewStatus st{};
        st.struct_size = sizeof st;
        if (!caliper_core_live_view_status(core, i, &st)) break;
        const std::string name = st.name ? st.name : "";
        if (name.empty() || !seen.insert(name).second) continue;
        ViewFrame f;
        f.name = name;
        f.kind = st.kind;
        frames.push_back(std::move(f));
    }
    apply(true, session, frames);
}

void LiveCatalog::apply(bool attached, const std::string& session, const std::vector<ViewFrame>& views) {
    const std::uint64_t t = now();
    std::vector<const ViewFrame*> to_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!attached) {
            if (attached_ || !views_.empty()) ++revision_;
            views_.clear();
            attached_ = false;
            session_.clear();
            return;
        }
        if (!attached_) ++revision_;
        attached_ = true;
        session_ = session;
        std::set<std::string> names;
        for (const ViewFrame& f : views) names.insert(f.name);
        for (auto it = views_.begin(); it != views_.end();) {
            if (!names.count(it->first)) {
                it = views_.erase(it);
                ++revision_;
            } else {
                ++it;
            }
        }
        for (const ViewFrame& f : views) {
            auto [it, fresh] = views_.try_emplace(f.name);
            View& v = it->second;
            if (fresh) {
                v.name = f.name;
                ++revision_;
            }
            // A series grows with its data, so only its rank is schema.
            const bool same_shape = f.kind == CALIPER_LIVE_KIND_SERIES ? v.shape.size() == f.shape.size()
                                                                        : v.shape == f.shape;
            if (!fresh && (v.kind != f.kind || v.has_payload != f.has_payload || v.dtype != f.dtype || !same_shape))
                ++revision_;
            v.kind = f.kind;
            v.has_payload = f.has_payload;
            v.dtype = f.dtype;
            v.shape = f.shape;
            v.dims = f.dims.size() == f.shape.size() ? f.dims : std::vector<std::string>();
            v.bytes = f.has_payload ? shape_bytes(f.dtype, f.shape) : 0;
            v.readable = f.data != nullptr;
            if (f.updates != v.updates_seen) {
                v.updates_seen = f.updates;
                if (f.has_payload) {
                    v.arrivals.push_back(t);
                    v.updated_at = catalog_detail::utc_text_ms(std::chrono::system_clock::now());
                }
            }
            while (!v.arrivals.empty() && t - v.arrivals.front() > options_.rate_window_ns) v.arrivals.pop_front();
            v.rate_hz.reset();
            if (v.arrivals.size() >= 2 && v.arrivals.back() > v.arrivals.front()) {
                const double hz = static_cast<double>(v.arrivals.size() - 1) * 1e9 /
                                  static_cast<double>(v.arrivals.back() - v.arrivals.front());
                if (hz > 0 && hz <= CatalogLimits::max_rate_hz) v.rate_hz = hz;
            }
            if (!f.has_payload || !f.data || v.copied_updates == f.updates) continue;
            const bool wanted = t < v.wanted_until;
            if (!wanted && v.generation != 0 && t - v.copied_at < options_.idle_copy_ns) continue;
            v.copy_too_large = v.bytes > options_.max_view_bytes;
            if (v.copy_too_large) continue;
            to_copy.push_back(&f);
        }
    }
    // The copies, outside the lock: the payloads stay claimed until the next poll.
    std::vector<std::pair<const ViewFrame*, TensorData>> made;
    for (const ViewFrame* f : to_copy) {
        TensorResult r = copy_payload(*f);
        if (r.error) continue;
        made.emplace_back(f, std::move(r.tensor));
    }
    if (made.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [f, tensor] : made) {
        const auto it = views_.find(f->name);
        if (it == views_.end()) continue;
        View& v = it->second;
        v.copy = std::move(tensor);
        v.generation = next_generation_++;
        v.copied_updates = f->updates;
        v.copied_at = t;
        ++copies_;
    }
}

CatalogEntry LiveCatalog::entry_of(const View& v) const {
    CatalogEntry e;
    e.kind = CatalogKind::live;
    e.name = v.name;
    e.id = make_catalog_id(e.kind, v.name);
    e.title = v.name;
    e.origin.session = catalog_detail::origin_value(session_);
    e.origin.view = catalog_detail::origin_value(v.name);
    if (const char* type = view_type_of(v.kind)) e.origin.view_type = type;
    if (v.has_payload) {
        e.dtype = catalog_dtype_of(v.dtype);
        e.shape = v.shape;
        e.bytes = v.bytes;
    }
    e.rate_hz = v.rate_hz;
    e.wire = true;
    e.updated_at = v.updated_at;
    return e;
}

CatalogListResult LiveCatalog::list(const CatalogFilter& filter) {
    if (filter.kind && *filter.kind != CatalogKind::live) return {};
    std::vector<CatalogEntry> all;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : views_)   // by name, so by id
            if (catalog_detail::is_catalog_name(kv.first)) all.push_back(entry_of(kv.second));
    }
    return catalog_detail::filter_entries(all, filter);
}

CatalogDescribeResult LiveCatalog::describe(const std::string& id) {
    CatalogDescribeResult out;
    const std::optional<std::string> name = view_name(id);
    if (!name) return out;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = views_.find(*name);
    if (it != views_.end()) out.entry = entry_of(it->second);
    return out;
}

// The view of `id` with its newest copy, marked as wanted; false with *why
// (a resolve.unavailable message) when there is no copy to give.
bool LiveCatalog::find_copy(const std::string& id, View* out, std::string* why) {
    const std::optional<std::string> name = view_name(id);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!attached_) {
        *why = "No live session is attached.";
        return false;
    }
    const auto it = name ? views_.find(*name) : views_.end();
    if (it == views_.end()) {
        *why = "The live session has no view " + (name ? *name : id) + ".";
        return false;
    }
    View& v = it->second;
    v.wanted_until = now() + options_.wanted_ns;
    if (!v.copy.valid()) {
        if (!v.has_payload) *why = "The live view " + v.name + " has published nothing yet.";
        else if (v.copy_too_large)
            *why = "The live view " + v.name + "'s snapshot of " + std::to_string(v.bytes) +
                   " bytes is past the copy limit of " + std::to_string(options_.max_view_bytes) + " bytes.";
        else if (!v.readable) *why = "The live view " + v.name + "'s snapshot is not readable on the CPU.";
        else *why = "The live view " + v.name + "'s snapshot is copied at the next frame.";
        return false;
    }
    *out = v;
    return true;
}

CatalogSampleResult LiveCatalog::sample(const std::string& id, const CatalogSampleRequest& request) {
    CatalogSampleResult out;
    View v;
    std::string why;
    if (!find_copy(id, &v, &why)) {
        bool known = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::optional<std::string> name = view_name(id);
            known = attached_ && name && views_.count(*name);
        }
        out.error = Error(known ? "catalog.no_data" : "catalog.not_found", why, "id");
        return out;
    }
    const TensorData& t = v.copy;
    TensorSampleResult s = sample_tensor(catalog_dtype_of(t.dtype()), t.shape(), t.data(),
                                         static_cast<std::size_t>(t.nbytes()), request.start, request.count);
    if (s.error) {
        out.error = std::move(s.error);
        return out;
    }
    out.sample.id = id;
    out.sample.kind = CatalogKind::live;
    out.sample.dtype = catalog_dtype_of(t.dtype());
    out.sample.shape = t.shape();
    out.sample.stats = std::move(s.stats);
    out.sample.slice = std::move(s.slice);
    return out;
}

std::uint64_t LiveCatalog::revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return revision_;
}

std::uint64_t LiveCatalog::generation(const std::string& id) {
    const std::optional<std::string> name = view_name(id);
    if (!name) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = views_.find(*name);
    if (it == views_.end()) return 0;
    it->second.wanted_until = now() + options_.wanted_ns;
    return it->second.generation;
}

ServedData LiveCatalog::read(const std::string& id, const ServedRequest& request) {
    ServedData out;
    View v;
    std::string why;
    if (!find_copy(id, &v, &why)) {
        out.error = Error("resolve.unavailable", why);
        return out;
    }
    if (v.copy.nbytes() > request.max_bytes) {
        out.error = Error("resolve.too_large", "The live view " + v.name + "'s snapshot of " +
                                                   std::to_string(v.copy.nbytes()) + " bytes is past the limit of " +
                                                   std::to_string(request.max_bytes) + " bytes.");
        return out;
    }
    out.tensor = v.copy;
    out.generation = v.generation;
    return out;
}

std::uint64_t LiveCatalog::copies() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return copies_;
}

std::vector<std::string> LiveCatalog::dim_names(const std::string& id) const {
    const std::optional<std::string> name = view_name(id);
    if (!name) return {};
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = views_.find(*name);
    if (it == views_.end() || !it->second.has_payload) return {};
    return it->second.dims;
}

}  // namespace caliper_host
