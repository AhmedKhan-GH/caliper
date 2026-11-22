// The chain hooks of binding references (chain_hooks.h,
// design/instrument-interface.md, Re-Evaluation).
//
// resolve.cpp calls substitute_references() for each step that holds a
// reference, and adds references_key() to the cache key of the chain. The key
// holds the values that the chain's steps read now, so a chain that no change
// touched hits the cache, and a chain evaluates again only when a value it
// reads changed: at most once a frame, because the app resolves a panel once a
// frame, and with the last value of a drag.
//
// A reference to the cursor reads the history's cursor, which is current on
// every thread; every other reference reads the session's values.
#include <caliper_builder/session/chain_hooks.h>

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/values.h>

#include <cmath>

namespace caliper_builder {

namespace {

using nlohmann::json;

// The value that `ref` reads in `ctx`, or its default.
json read(const spec::Reference& ref, const ChainContext& ctx) {
    if (ref.kind == spec::Reference::Kind::state && ref.target == spec::kCursorState && ctx.history) {
        const CursorState c = ctx.history->cursor();
        const std::string field = ref.field.value_or("step");
        if (field == "step") return c.step;
        if (field == "first") return c.first;
        if (field == "latest") return c.latest;
        if (field == "live") return c.live;
        if (field == "playing") return c.playing;
        if (field == "speed") return c.speed;
        return ref.fallback;
    }
    if (ctx.values) return ctx.values->value(ref);
    return ref.fallback;
}

std::string describe(const spec::Reference& ref) {
    std::string out = std::string(spec::reference_kind_name(ref.kind)) + " " + ref.target;
    if (ref.field) out += "." + *ref.field;
    return out;
}

}  // namespace

Error substitute_references(const spec::Transform& step, const ChainContext& ctx, spec::Transform* out) {
    *out = step;
    out->references.clear();
    for (const auto& [key, ref] : step.references) {
        const json v = read(ref, ctx);
        if (v.is_null())
            return Error("resolve.waiting", "Waiting for input: " + key + " reads the " + describe(ref) + ".", key);
        if (!v.is_number() || !std::isfinite(v.get<double>()))
            return Error("resolve.waiting",
                         "Waiting for input: " + key + " reads the " + describe(ref) + ", which is not a number now.",
                         key);
        // An integer place takes the nearest whole number of a slider's value.
        const std::int64_t n = std::llround(v.get<double>());
        if (key == "index") out->index = n;
        else if (key == "start") out->start = n;
        else if (key == "stop") out->stop = n;
        else if (key == "step") out->step = n;
        else if (key == "steps") out->steps = n;
    }
    return {};
}

std::string references_key(const std::vector<const spec::Source*>& chain, const ChainContext& ctx) {
    json parts = json::array();
    for (const spec::Source* s : chain) {
        if (!s) continue;
        for (std::size_t j = 0; j < s->transforms.size(); ++j)
            for (const auto& [key, ref] : s->transforms[j].references)
                parts.push_back(json::array({s->id, j, key, read(ref, ctx)}));
    }
    return parts.empty() ? std::string() : "\nreferences " + parts.dump();
}

}  // namespace caliper_builder
