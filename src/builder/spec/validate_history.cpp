// The validator's checks of kept steps: a source's `history`, the `window`
// step and the transport control (design/instrument-interface.md, Time
// Transport). The codes are spec.history_not_allowed, spec.no_history,
// spec.reference_in_history, spec.out_of_range, spec.bad_transform and
// spec.too_many.
#include "validate_additions.h"

#include <set>

namespace caliper_builder::spec::detail {

namespace {

constexpr std::int64_t kMaxHistory = 65536;

const Source* find_source(const Spec& s, const std::string& id) {
    for (const Source& c : s.sources)
        if (c.id == id) return &c;
    return nullptr;
}

// The sources from the root down to `source`; empty for a missing source or
// a cycle, which the validator reports elsewhere.
std::vector<const Source*> chain_path(const Spec& s, const Source& source) {
    std::vector<const Source*> path;
    std::set<std::string> seen;
    const Source* at = &source;
    while (at) {
        if (!seen.insert(at->id).second) return {};
        path.insert(path.begin(), at);
        if (!at->from) break;
        at = find_source(s, *at->from);
        if (!at) return {};
    }
    return path;
}

bool live_root(const std::vector<const Source*>& path) {
    return !path.empty() && path.front()->catalog && path.front()->catalog->rfind("live:", 0) == 0;
}

bool has_window(const std::vector<const Source*>& path) {
    for (const Source* p : path)
        for (const Transform& t : p->transforms)
            if (t.op == TransformOp::window) return true;
    return false;
}

bool is_shipped(TransformOp op) {
    return op == TransformOp::slice || op == TransformOp::reduce || op == TransformOp::normalise;
}

// Whether `source` keeps steps, by the rules of history_depth() in the
// session library (resolve.h).
bool keeps_steps(const Spec& s, const Source& source) {
    const std::vector<const Source*> path = chain_path(s, source);
    if (!live_root(path) || has_window(path)) return false;
    if (source.history) return *source.history > 0;
    if (source.from) return false;
    return source.transforms.empty() || is_shipped(source.transforms.front().op);
}

}  // namespace

void validate_history(const AdditionsCheck& c) {
    const Spec& s = c.spec;
    for (std::size_t i = 0; i < s.sources.size(); ++i) {
        const Source& src = s.sources[i];
        const std::string path = index_path("source", i);
        const std::vector<const Source*> chain = chain_path(s, src);
        if (src.history) {
            if (*src.history < 0 || *src.history > kMaxHistory)
                c.sink.error("spec.out_of_range", "history must be an integer from 0 to 65,536.", path + ".history");
            else if (!chain.empty() && !live_root(chain))
                c.sink.error("spec.history_not_allowed",
                             "The source " + src.id +
                                 " does not come from a live tensor, so the app keeps no steps of it; history "
                                 "applies to a source whose root is a live: catalog source.",
                             path + ".history");
            else if (has_window(chain))
                c.sink.error("spec.history_not_allowed",
                             "The source " + src.id +
                                 " reads kept steps through a window, so it keeps no steps of its own.",
                             path + ".history");
        }
        for (std::size_t j = 0; j < src.transforms.size(); ++j) {
            const Transform& t = src.transforms[j];
            if (t.op != TransformOp::window) continue;
            const std::string tp = index_path(path + ".transform", j);
            if (j != 0) {
                c.sink.error("spec.bad_transform", "A window is the first step of a source.", tp);
                continue;
            }
            if (!src.from) {
                c.sink.error("spec.bad_transform",
                             "A window stacks the kept steps of the source that from names, and " + src.id +
                                 " has no from.",
                             tp);
                continue;
            }
            const bool ends = t.start || t.stop || t.references.count("start") || t.references.count("stop");
            if ((t.steps || t.references.count("steps")) && ends)
                c.sink.error("spec.bad_transform", "A window takes steps, or start and stop, but not both.", tp);
            // The range of steps, start and stop: validate.cpp, through integer_places().
            if (t.start && t.stop && *t.start > *t.stop)
                c.sink.error("spec.out_of_range", "start must not be after stop.", tp + ".start");
            const Source* from = find_source(s, *src.from);
            if (!from) continue;   // the validator reports the unknown source
            const std::vector<const Source*> kept = chain_path(s, *from);
            bool reads_reference = false;
            for (const Source* p : kept)
                for (const Transform& k : p->transforms) reads_reference = reads_reference || !k.references.empty();
            if (reads_reference)
                c.sink.error("spec.reference_in_history",
                             "The window reads the kept steps of " + from->id +
                                 ", whose own steps read a reference; the app keeps values before such a step, so a "
                                 "window cannot stack them. Move the referenced step after the window.",
                             tp);
            else if (!kept.empty() && !keeps_steps(s, *from))
                c.sink.error("spec.no_history",
                             "The source " + from->id +
                                 " keeps no steps, so the window has nothing to stack. Give it history, or read a "
                                 "live tensor source without from.",
                             tp);
        }
    }
    bool transport = false;
    for (std::size_t i = 0; i < s.controls.size(); ++i) {
        const Control& ctl = s.controls[i];
        const ControlOptions& o = ctl.options;
        const std::string path = index_path("control", i);
        if (ctl.kind == ControlKind::transport) {
            if (transport)
                c.sink.error("spec.too_many", "A spec has at most one transport control.", path);
            transport = true;
        }
        if (o.speed && ctl.kind != ControlKind::transport)
            c.sink.error("spec.unknown_option", "speed is an option of a transport control.", path + ".options.speed");
        else if (o.speed && !(*o.speed >= 0.1 && *o.speed <= 1000.0))
            c.sink.error("spec.out_of_range", "speed must be from 0.1 to 1,000 steps a second.",
                         path + ".options.speed");
        if (o.keys && ctl.kind != ControlKind::transport)
            c.sink.error("spec.unknown_option", "keys is an option of a transport control.", path + ".options.keys");
    }
}

bool apply_window_shape(const Transform& t, SourceInfo& info, const std::string& path, Sink& sink) {
    (void)t;
    (void)path;
    (void)sink;
    // The window stacks W kept values of the source before it: (W, ...).
    if (info.shape) {
        if (info.shape->size() >= static_cast<std::size_t>(limits::kMaxRank)) {
            info.shape.reset();
            return true;
        }
        info.shape->insert(info.shape->begin(), -1);
    }
    return true;
}

}  // namespace caliper_builder::spec::detail
