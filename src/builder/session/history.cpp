// The kept steps of the open spec and the time cursor
// (caliper_builder/session/history.h, design/instrument-interface.md,
// section Time Transport).
//
// Each source that keeps steps has a ring: a map from step to the value
// after the source's history point, at most `depth` values, oldest first.
// The budget shares kHistoryBudgetBytes among the rings by bytes: no ring
// keeps more than kHistorySourceBytes, then the largest rings shorten first
// (water filling), and a ring whose value size is not known yet counts as
// empty.
#include <caliper_builder/session/history.h>

#include <caliper_builder/loaders/tensor_ops.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <mutex>
#include <set>

namespace caliper_builder {

namespace {

using json = nlohmann::json;

bool is_live_catalog(const std::optional<std::string>& catalog) {
    return catalog && catalog->rfind("live:", 0) == 0;
}

// The sources from the root down to `source`, through `from`; empty when the
// chain has a missing source or a cycle.
std::vector<const spec::Source*> chain_path(const spec::Spec& s, const spec::Source& source) {
    std::vector<const spec::Source*> path;
    std::set<std::string> seen;
    const spec::Source* at = &source;
    while (at) {
        if (!seen.insert(at->id).second) return {};
        path.push_back(at);
        if (!at->from) break;
        const spec::Source* next = nullptr;
        for (const spec::Source& c : s.sources)
            if (c.id == *at->from) next = &c;
        if (!next) return {};
        at = next;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

bool is_shipped(spec::TransformOp op) {
    return op == spec::TransformOp::slice || op == spec::TransformOp::reduce || op == spec::TransformOp::normalise;
}

// What one source keeps, from the spec.
struct RingPlan {
    std::string catalog;          // the root's catalog id
    std::int64_t requested = 0;   // the depth before the budget
    std::size_t point = 0;        // the chain steps applied to the kept value
    json prefix = json::array();  // their records, which name the chain for a reset
    bool declared = false;        // the source states `history`, so it keeps the steps of a series too
};

std::optional<RingPlan> plan_of(const spec::Spec& s, const spec::Source& source) {
    const std::vector<const spec::Source*> path = chain_path(s, source);
    if (path.empty() || !is_live_catalog(path.front()->catalog)) return std::nullopt;
    RingPlan plan;
    plan.catalog = *path.front()->catalog;
    plan.point = history_point(s, source);
    plan.requested = history_depth(s, source);
    plan.declared = source.history.has_value();
    if (plan.requested <= 0) return std::nullopt;
    std::size_t k = 0;
    for (const spec::Source* p : path) {
        json records = spec::to_json(*p).value("transform", json::array());
        for (auto& r : records) {
            if (k++ >= plan.point) break;
            if (r.is_object()) r.erase("x");
            plan.prefix.push_back(std::move(r));
        }
    }
    return plan;
}

// Water filling over bytes: the depth of each ring so that all fit the
// budget, the largest rings shortened first.
std::vector<std::int64_t> budget_depths(const std::vector<std::int64_t>& requested,
                                        const std::vector<std::uint64_t>& step_bytes, std::uint64_t budget) {
    const std::size_t n = requested.size();
    std::vector<std::int64_t> depth(requested);
    long double total = 0;
    for (std::size_t i = 0; i < n; ++i) total += (long double)requested[i] * step_bytes[i];
    if (total <= budget) return depth;
    // Find the cap C on each ring's bytes with sum(min(want_i, C)) == budget.
    std::vector<long double> want(n);
    for (std::size_t i = 0; i < n; ++i) want[i] = (long double)requested[i] * step_bytes[i];
    std::vector<long double> sorted(want);
    std::sort(sorted.begin(), sorted.end());
    long double left = budget;
    long double cap = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const long double share = left / (long double)(n - i);
        if (sorted[i] <= share) {
            left -= sorted[i];
            continue;
        }
        cap = share;
        break;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (step_bytes[i] == 0 || want[i] <= cap) continue;
        depth[i] = (std::int64_t)std::floor(cap / (long double)step_bytes[i]);
    }
    return depth;
}

// Why a ring keeps `depth` of its `requested` steps: the source, the size of
// one step, the share of one source and of all, and the step that helps.
std::string budget_note(const std::string& id, std::int64_t depth, std::int64_t requested, std::uint64_t step_bytes) {
    auto mib = [](std::uint64_t b) {
        char text[32];
        std::snprintf(text, sizeof text, "%.1f MiB", double(b) / double(1 << 20));
        return std::string(text);
    };
    return "The source " + id + " keeps " + std::to_string(depth) + " of its " + std::to_string(requested) +
           " steps, because one step holds " + mib(step_bytes) + ", one source keeps at most " +
           mib(kHistorySourceBytes) + " and all sources share " + mib(kHistoryBudgetBytes) +
           ". A reduce step in the source before its kept value, such as a mean over an axis, keeps more steps.";
}

// The live registry of the catalog ids each history keeps, so that the live
// catalog copies each of their snapshots (kept_catalog_ids()).
struct CatalogIds {
    std::mutex mutex;
    std::vector<std::string> ids;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::map<const StepHistory*, std::shared_ptr<CatalogIds>>& registry() {
    static std::map<const StepHistory*, std::shared_ptr<CatalogIds>> r;
    return r;
}

}  // namespace

std::int64_t history_depth(const spec::Spec& s, const spec::Source& source) {
    const std::vector<const spec::Source*> path = chain_path(s, source);
    if (path.empty() || !is_live_catalog(path.front()->catalog)) return 0;
    // A chain that holds a window reads kept steps itself and keeps none.
    for (const spec::Source* p : path)
        for (const spec::Transform& t : p->transforms)
            if (t.op == spec::TransformOp::window) return 0;
    if (source.history) return std::max<std::int64_t>(0, *source.history);
    if (source.from) return 0;
    // The first step of a live root that is an operator runs in the script and
    // gives geometry, whose steps this work does not keep.
    if (!source.transforms.empty() && !is_shipped(source.transforms.front().op)) return 0;
    return kDefaultLiveDepth;
}

std::size_t history_point(const spec::Spec& s, const spec::Source& source) {
    std::size_t point = 0;
    for (const spec::Source* p : chain_path(s, source))
        for (const spec::Transform& t : p->transforms) {
            if (!is_shipped(t.op) || !t.references.empty()) return point;
            ++point;
        }
    return point;
}

std::vector<std::string> kept_catalog_ids(const StepHistory& history) {
    std::shared_ptr<CatalogIds> ids;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        const auto it = registry().find(&history);
        if (it == registry().end()) return {};
        ids = it->second;
    }
    std::lock_guard<std::mutex> lock(ids->mutex);
    return ids->ids;
}

struct StepHistory::Impl {
    struct Ring {
        RingPlan plan;
        std::vector<TensorTransform> steps;   // the prefix, parsed
        std::int64_t depth = 0;               // after the budget
        std::map<std::int64_t, TensorData> values;
        std::uint64_t bytes = 0;
        std::uint64_t step_bytes = 0;         // of one value, once one arrived
        std::string note;                     // why the budget shortened it
        std::vector<std::int64_t> shape;
        DType dtype = DType::f32;
    };

    mutable std::mutex mutex;
    std::map<std::string, Ring> rings;        // by source id
    CursorState cursor;
    std::optional<double> spec_speed;         // the transport's speed at the last configure
    double play = 0;                          // the fraction of a step that play has moved
    std::shared_ptr<CatalogIds> ids = std::make_shared<CatalogIds>();

    void publish_ids() {
        std::set<std::string> set;
        for (const auto& [id, r] : rings) set.insert(r.plan.catalog);
        std::lock_guard<std::mutex> lock(ids->mutex);
        ids->ids.assign(set.begin(), set.end());
    }

    void trim(Ring& r) {
        while ((std::int64_t)r.values.size() > r.depth && !r.values.empty()) {
            r.bytes -= r.values.begin()->second.nbytes();
            r.values.erase(r.values.begin());
        }
    }

    void apply_budget() {
        std::vector<std::int64_t> requested;
        std::vector<std::uint64_t> step_bytes;
        for (const auto& [id, r] : rings) {
            // One source keeps at most its share, so that one raw activation
            // never takes the whole budget from the others.
            std::int64_t want = r.plan.requested;
            if (r.step_bytes > 0)
                want = std::min<std::int64_t>(want, std::int64_t(kHistorySourceBytes / r.step_bytes));
            requested.push_back(want);
            step_bytes.push_back(r.step_bytes);
        }
        const std::vector<std::int64_t> depth = budget_depths(requested, step_bytes, kHistoryBudgetBytes);
        std::size_t i = 0;
        for (auto& [id, r] : rings) {
            r.depth = depth[i++];
            r.note = r.depth < r.plan.requested ? budget_note(id, r.depth, r.plan.requested, r.step_bytes) : "";
            trim(r);
        }
    }

    std::vector<std::int64_t> steps_locked() const {
        std::set<std::int64_t> all;
        for (const auto& [id, r] : rings)
            for (const auto& [step, v] : r.values) all.insert(step);
        return {all.begin(), all.end()};
    }

    // Updates `first` and clamps a held cursor that fell out of every source.
    void bounds() {
        std::optional<std::int64_t> first;
        for (const auto& [id, r] : rings)
            if (!r.values.empty()) first = first ? std::min(*first, r.values.begin()->first) : r.values.begin()->first;
        cursor.first = first.value_or(cursor.latest);
        if (cursor.live) {
            cursor.step = cursor.latest;
        } else if (cursor.step < cursor.first) {
            cursor.step = cursor.first;
        }
    }

    // The kept step nearest `step`, the older one on a tie.
    std::int64_t nearest(std::int64_t step) const {
        const std::vector<std::int64_t> all = steps_locked();
        if (all.empty()) return step;
        const auto hi = std::lower_bound(all.begin(), all.end(), step);
        if (hi == all.end()) return all.back();
        if (*hi == step || hi == all.begin()) return *hi;
        const std::int64_t below = *(hi - 1);
        return (step - below) <= (*hi - step) ? below : *hi;
    }
};

StepHistory::StepHistory() : impl_(std::make_unique<Impl>()) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[this] = impl_->ids;
}

StepHistory::~StepHistory() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(this);
}

void StepHistory::configure(const spec::Spec& spec) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::map<std::string, Impl::Ring> next;
    for (const spec::Source& source : spec.sources) {
        std::optional<RingPlan> plan = plan_of(spec, source);
        if (!plan) continue;
        Impl::Ring ring;
        const auto old = impl_->rings.find(source.id);
        if (old != impl_->rings.end() && old->second.plan.catalog == plan->catalog &&
            old->second.plan.prefix == plan->prefix) {
            ring = std::move(old->second);   // the same chain keeps its steps
        } else {
            for (const json& record : plan->prefix) {
                TensorTransform t;
                if (transform_from_json(record, &t)) break;
                ring.steps.push_back(t);
            }
        }
        ring.plan = std::move(*plan);
        next[source.id] = std::move(ring);
    }
    impl_->rings = std::move(next);
    impl_->apply_budget();
    // The transport's speed applies when the spec opens or the option changes.
    std::optional<double> speed;
    for (const spec::Control& c : spec.controls)
        if (c.kind == spec::ControlKind::transport) speed = c.options.speed.value_or(10.0);
    if (speed && speed != impl_->spec_speed) impl_->cursor.speed = std::clamp(*speed, 0.1, 1000.0);
    impl_->spec_speed = speed;
    // A spec whose sources keep nothing any more starts live.
    bool any = false;
    for (const auto& [id, r] : impl_->rings) any = any || !r.values.empty();
    if (!any) {
        impl_->cursor.live = true;
        impl_->cursor.playing = false;
    }
    impl_->bounds();
    impl_->publish_ids();
}

void StepHistory::clear() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto& [id, r] : impl_->rings) {
        r.values.clear();
        r.bytes = 0;
    }
    impl_->cursor.step = impl_->cursor.first = impl_->cursor.latest = 0;
    impl_->cursor.live = true;
    impl_->cursor.playing = false;
    impl_->play = 0;
}

void StepHistory::append(const std::string& catalog_id, std::int64_t step, const TensorData& value) {
    append_to(catalog_id, step, value, false);
}

void StepHistory::append_series(const std::string& catalog_id, std::int64_t step, double value) {
    TensorResult one = TensorData::allocate(DType::f32, {});
    if (one.error) return;
    *reinterpret_cast<float*>(one.tensor.mutable_data()) = static_cast<float>(value);
    append_to(catalog_id, step, one.tensor, true);
}

void StepHistory::append_to(const std::string& catalog_id, std::int64_t step, const TensorData& value,
                            bool declared_only) {
    if (!value.valid()) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    bool wanted = false;
    bool resized = false;
    for (auto& [id, r] : impl_->rings) {
        if (r.plan.catalog != catalog_id || (declared_only && !r.plan.declared)) continue;
        wanted = true;
        TensorData kept = value;
        if (!r.steps.empty()) {
            TensorResult t = apply_transforms(value, r.steps);
            if (t.error) continue;   // a value this chain cannot reduce is not kept
            kept = std::move(t.tensor);
        }
        // One ring holds values of one shape: a new shape starts it over.
        if (!r.values.empty() && (kept.shape() != r.shape || kept.dtype() != r.dtype)) {
            r.values.clear();
            r.bytes = 0;
        }
        r.shape = kept.shape();
        r.dtype = kept.dtype();
        if (r.step_bytes != kept.nbytes()) {
            r.step_bytes = kept.nbytes();
            resized = true;
        }
        const auto old = r.values.find(step);
        if (old != r.values.end()) r.bytes -= old->second.nbytes();
        r.bytes += kept.nbytes();
        r.values[step] = std::move(kept);
        impl_->trim(r);
    }
    if (!wanted) return;
    if (resized) impl_->apply_budget();
    impl_->cursor.latest = std::max(impl_->cursor.latest, step);
    impl_->bounds();
}

std::optional<std::pair<std::int64_t, TensorData>> StepHistory::at(const std::string& source,
                                                                    std::int64_t step) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto it = impl_->rings.find(source);
    if (it == impl_->rings.end() || it->second.values.empty()) return std::nullopt;
    const auto& values = it->second.values;
    auto v = values.upper_bound(step);
    if (v == values.begin()) return std::nullopt;
    --v;
    return std::make_pair(v->first, v->second);
}

std::vector<std::pair<std::int64_t, TensorData>> StepHistory::window(const std::string& source, std::int64_t lo,
                                                                     std::int64_t hi) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<std::pair<std::int64_t, TensorData>> out;
    const auto it = impl_->rings.find(source);
    if (it == impl_->rings.end() || lo > hi) return out;
    const auto& values = it->second.values;
    for (auto v = values.lower_bound(lo); v != values.end() && v->first <= hi; ++v) out.emplace_back(v->first, v->second);
    return out;
}

std::vector<KeptSource> StepHistory::kept() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<KeptSource> out;
    for (const auto& [id, r] : impl_->rings) {
        KeptSource k;
        k.id = id;
        if (!r.values.empty()) {
            k.first = r.values.begin()->first;
            k.last = r.values.rbegin()->first;
        }
        k.steps = (std::int64_t)r.values.size();
        k.requested = r.plan.requested;
        k.bytes = r.bytes;
        k.note = r.note;
        out.push_back(std::move(k));
    }
    return out;
}

std::vector<std::int64_t> StepHistory::kept_steps() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->steps_locked();
}

CursorState StepHistory::cursor() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->cursor;
}

CursorState StepHistory::set_cursor(const CursorState& wanted) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    CursorState& c = impl_->cursor;
    c.speed = std::clamp(std::isfinite(wanted.speed) ? wanted.speed : c.speed, 0.1, 1000.0);
    impl_->play = 0;
    if (wanted.live) {
        c.live = true;
        c.playing = false;
        c.step = c.latest;
    } else {
        c.live = false;
        c.playing = wanted.playing;
        c.step = impl_->nearest(wanted.step);
    }
    impl_->bounds();
    return c;
}

CursorState StepHistory::step_cursor(std::int64_t count) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    CursorState& c = impl_->cursor;
    const std::vector<std::int64_t> all = impl_->steps_locked();
    if (all.empty() || count == 0) return c;
    const std::int64_t n = (std::int64_t)all.size();
    std::int64_t target = 0;
    if (count < 0) {
        const std::int64_t lo = std::lower_bound(all.begin(), all.end(), c.step) - all.begin();
        target = lo + count;
    } else {
        const std::int64_t up = std::upper_bound(all.begin(), all.end(), c.step) - all.begin();
        target = up + count - 1;
    }
    target = std::clamp<std::int64_t>(target, 0, n - 1);
    c.live = false;
    c.playing = false;
    impl_->play = 0;
    c.step = all[(std::size_t)target];
    impl_->bounds();
    return c;
}

void StepHistory::tick(double seconds) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    CursorState& c = impl_->cursor;
    if (!c.playing || c.live || !(seconds > 0)) return;
    impl_->play += seconds * c.speed;
    const double whole = std::floor(impl_->play);
    if (whole < 1) return;
    impl_->play -= whole;
    const double next = (double)c.step + whole;
    if (next >= (double)c.latest) {
        c.live = true;
        c.playing = false;
        impl_->play = 0;
        c.step = c.latest;
        return;
    }
    c.step = (std::int64_t)next;
}

nlohmann::json cursor_json(const CursorState& c) {
    return {{"step", c.step},       {"first", c.first},     {"latest", c.latest},
            {"live", c.live},       {"playing", c.playing}, {"speed", c.speed}};
}

}  // namespace caliper_builder
