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
#include <cstdint>
#include <limits>
#include <optional>

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

// The whole number that `v` gives, when it is one exactly: an integer that
// fits a signed 64-bit value, or a finite double of a magnitude of at most
// 2^53 - 1, rounded to the nearest whole number as an integer place takes a
// slider's value. Nothing for any other value.
static std::optional<std::int64_t> whole_number(const nlohmann::json& v) {
    if (v.is_number_unsigned()) {
        const std::uint64_t u = v.get<std::uint64_t>();
        if (u > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return std::nullopt;
        return static_cast<std::int64_t>(u);
    }
    if (v.is_number_integer()) return v.get<std::int64_t>();
    const double d = v.get<double>();
    if (!std::isfinite(d) || std::fabs(d) > static_cast<double>(spec::limits::kMaxSafeInteger)) return std::nullopt;
    return std::llround(d);
}

Error substitute_references(const spec::Transform& step, const ChainContext& ctx, spec::Transform* out) {
    *out = step;
    out->references.clear();
    for (const auto& [key, ref] : step.references) {
        // Every value a reference substitutes passes this check: the place
        // and its range come from the table that the validator applies to a
        // literal there (spec.h, integer_places()).
        const spec::IntegerPlace* place = spec::integer_place(step.op, key);
        if (!place)
            return Error("resolve.reference_not_allowed",
                         key + " of a " + std::string(spec::to_string(step.op)) + " step takes a literal only.", key);
        const json v = read(ref, ctx);
        if (v.is_null())
            return Error("resolve.waiting", "Waiting for input: " + key + " reads the " + describe(ref) + ".", key);
        if (!v.is_number() || (v.is_number_float() && !std::isfinite(v.get<double>())))
            return Error("resolve.waiting",
                         "Waiting for input: " + key + " reads the " + describe(ref) + ", which is not a number now.",
                         key);
        const std::optional<std::int64_t> n = whole_number(v);
        if (!n || *n < place->min || *n > place->max)
            return Error("resolve.out_of_range",
                         key + " reads the " + describe(ref) + ", which is " + v.dump() + " now; " + key +
                             " is a whole number from " + std::to_string(place->min) + " to " +
                             std::to_string(place->max) + ".",
                         key);
        out->*(place->field) = *n;
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
