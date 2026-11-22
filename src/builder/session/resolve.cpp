#include <caliper_builder/session/resolve.h>

#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/loaders/tensor_ops.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/loader_catalog.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
#include <exception>
#include <map>
#include <set>
#include <variant>

#include "util.h"

namespace caliper_builder {

namespace {

using nlohmann::json;

struct Binding {
    std::string source;
    std::string column;  // empty: the whole source
};

Binding split_binding(const std::string& b) {
    const auto dot = b.find('.');
    if (dot == std::string::npos) return {b, {}};
    return {b.substr(0, dot), b.substr(dot + 1)};
}

enum class RootKind { tensor, table, sql, served_tensor, served_table, unavailable };

struct Chain {
    std::vector<const spec::Source*> path;  // the root first, the bound source last
    RootKind kind = RootKind::unavailable;
    CatalogKind catalog_kind = CatalogKind::table;  // of a catalog root
    // The catalog id that a served root reads: the root's own, or for a live
    // tensor whose first step is an operator that runs in the script, the
    // live view of the operator's tensor output (design/instrument-interface.md,
    // Operator Programs).
    std::string catalog;
    Error error;
};

bool is_served(RootKind k) { return k == RootKind::served_tensor || k == RootKind::served_table; }

// Follows the chain of `from` from source `id` to its root and classifies the
// root. A served root (live, feed, metric) needs a registered data source.
Chain chain_of(const spec::Spec& spec, const std::string& id, const DataSources* sources) {
    Chain c;
    const spec::Source* cur = spec.find_source(id);
    if (!cur) {
        c.error = Error("resolve.unknown_source", "The spec has no source " + id + ".", id);
        return c;
    }
    std::vector<const spec::Source*> rev{cur};
    while (cur->from) {
        if (rev.size() > spec::limits::kSources) {
            c.error = Error("resolve.cycle", "The chain of from that starts at " + id + " does not end.", id);
            return c;
        }
        const spec::Source* next = spec.find_source(*cur->from);
        if (!next) {
            c.error = Error("resolve.unknown_source", "The spec has no source " + *cur->from + ".", cur->id);
            return c;
        }
        rev.push_back(next);
        cur = next;
    }
    c.path.assign(rev.rbegin(), rev.rend());
    const spec::Source& root = *c.path.front();
    if (root.sql) {
        c.kind = RootKind::sql;
        return c;
    }
    if (!root.catalog) {
        c.error = Error("resolve.unknown_source", "The source " + root.id + " names no data.", root.id);
        return c;
    }
    const CatalogIdParse parsed = parse_catalog_id(*root.catalog);
    if (parsed.error) {
        c.error = Error("resolve.unknown_source", "The catalog id " + *root.catalog + " is not valid.", root.id);
        return c;
    }
    c.catalog_kind = parsed.kind;
    c.catalog = *root.catalog;
    if (parsed.kind == CatalogKind::live && operator_runs_in_script(spec, root, 0))
        c.catalog = "live:program.operator." + root.id + ".0";
    switch (parsed.kind) {
        case CatalogKind::tensor: c.kind = RootKind::tensor; return c;
        case CatalogKind::table: c.kind = RootKind::table; return c;
        case CatalogKind::live:
        case CatalogKind::run: c.kind = RootKind::served_tensor; break;
        case CatalogKind::feed:
        case CatalogKind::metric: c.kind = RootKind::served_table; break;
    }
    if (!sources || !sources->get(parsed.kind)) {
        c.kind = RootKind::unavailable;
        c.error = Error("resolve.unavailable",
                        "The source " + root.id + " is a " + catalog_kind_name(parsed.kind) +
                            " source, which the app serves, and no data source for that kind is registered.",
                        root.id);
    }
    return c;
}

// The text that names a chain of transforms over its root: the root's catalog
// id and the transforms of every source down the chain, in order.
std::string chain_key(const Chain& chain, const ChainContext& ctx) {
    json parts = json::array();
    for (const spec::Source* s : chain.path) {
        json records = spec::to_json(*s).value("transform", json::array());
        for (auto& r : records)
            if (r.is_object()) r.erase("x");
        parts.push_back(std::move(records));
    }
    // The values that the chain's references read (chain_hooks.h), so that a
    // chain that no change touched hits the cache.
    return chain.catalog + "\n" + parts.dump() + references_key(chain.path, ctx);
}

// One step of the mix of generations (splitmix64), so that the panel's
// generation changes when any one of its sources changes.
std::uint64_t mix_generation(std::uint64_t h, std::uint64_t g) {
    std::uint64_t z = h + 0x9E3779B97F4A7C15ull + g;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// One step of a chain: its source and its index in the source's transforms,
// from the root's first step to the bound source's last.
struct ChainStep {
    const spec::Source* source = nullptr;
    std::size_t index = 0;
    spec::TransformOp op() const { return source->transforms[index].op; }
};

std::vector<ChainStep> chain_steps(const Chain& chain) {
    std::vector<ChainStep> out;
    for (const spec::Source* s : chain.path)
        for (std::size_t k = 0; k < s->transforms.size(); ++k) out.push_back({s, k});
    return out;
}

bool is_operator(spec::TransformOp op) { return op == spec::TransformOp::surface || op == spec::TransformOp::path; }

// Whether a chain reads kept steps: a window, or a source that keeps steps,
// whose value at a held cursor comes from the history.
bool reads_kept_steps(const spec::Spec& s, const Chain& chain) {
    for (const spec::Source* p : chain.path) {
        if (history_depth(s, *p) > 0) return true;
        for (const spec::Transform& t : p->transforms)
            if (t.op == spec::TransformOp::window) return true;
    }
    return false;
}

// The binding that a slot reference reads now: a text state whose value is
// a source id (design/instrument-interface.md, State Readers, Retarget).
Error reference_binding(const spec::Reference& ref, const ChainContext* ctx, std::string* out) {
    json v = ctx && ctx->values ? ctx->values->value(ref) : ref.fallback;
    if (v.is_null()) v = ref.fallback;
    if (v.is_string() && !v.get<std::string>().empty()) {
        *out = v.get<std::string>();
        return {};
    }
    return Error("resolve.waiting", "The slot reads " + ref.target + ", which has no source id yet.", ref.target);
}

// Each binding of a panel's slots in slot order, list slots in their order,
// with a slot reference read now; a reference without a value is left out
// and reported in `error`.
struct BoundSlot {
    std::string slot;
    std::size_t index = 0;
    std::string binding;
};

std::vector<BoundSlot> panel_bindings(const spec::Panel& panel, const ChainContext* ctx, Error* error) {
    std::vector<BoundSlot> out;
    for (const auto& [slot, value] : panel.bind) {
        if (value.bindings.empty() && value.reference) {
            std::string b;
            if (Error err = reference_binding(*value.reference, ctx, &b)) {
                if (error && !*error) *error = err;
                continue;
            }
            out.push_back({slot, 0, b});
            continue;
        }
        for (std::size_t i = 0; i < value.bindings.size(); ++i) out.push_back({slot, i, value.bindings[i]});
    }
    return out;
}

// Each binding of the entities of a scene panel, in entity order and then
// slot order, as its first element the entity's index.
std::vector<std::pair<std::size_t, BoundSlot>> entity_bindings(const spec::Panel& panel, const ChainContext* ctx,
                                                                std::map<std::size_t, Error>* errors) {
    std::vector<std::pair<std::size_t, BoundSlot>> out;
    for (std::size_t i = 0; i < panel.entities.size(); ++i) {
        const spec::Entity& e = panel.entities[i];
        if (e.source) {
            out.push_back({i, {std::string(), 0, *e.source}});
            continue;
        }
        for (const auto& [slot, value] : e.slots) {
            std::string b;
            if (const std::string* literal = std::get_if<std::string>(&value)) {
                b = *literal;
            } else if (Error err = reference_binding(std::get<spec::Reference>(value), ctx, &b)) {
                if (errors && !errors->count(i)) (*errors)[i] = err;
                continue;
            }
            out.push_back({i, {slot, 0, b}});
        }
    }
    return out;
}

// While the cursor is held, a panel that reads kept steps mixes in the
// cursor's step, so that its generation changes when the cursor moves.
std::uint64_t mix_cursor(std::uint64_t generation, bool reads_history, const ChainContext* ctx) {
    if (!reads_history || !ctx || ctx->cursor_live) return generation;
    return mix_generation(mix_generation(generation, 0xC0A5u), static_cast<std::uint64_t>(ctx->cursor_step));
}

// The references of an entity's options, read from their tables.
void option_references(const json& options, std::vector<spec::Reference>* out) {
    if (!options.is_object()) return;
    for (auto it = options.begin(); it != options.end(); ++it)
        if (spec::looks_like_reference(it.value()))
            if (std::optional<spec::Reference> r = spec::parse_reference(it.value())) out->push_back(*r);
}

// A number that changes when any reference that the panel reads changes:
// its slots, its options, its entities' slots and options, and the steps of
// the chains of `bindings`. 0 without a value store or without references,
// so that a panel that reads none keeps the generation of its sources.
std::uint64_t reference_versions(const spec::Spec& s, const spec::Panel& panel,
                                 const std::vector<std::string>& bindings, const ChainContext* ctx,
                                 const DataSources* sources) {
    if (!ctx || !ctx->values) return 0;
    std::vector<spec::Reference> refs;
    for (const auto& [slot, value] : panel.bind)
        if (value.reference) refs.push_back(*value.reference);
    for (const auto& [key, ref] : panel.options.references) refs.push_back(ref);
    for (const spec::Entity& e : panel.entities) {
        for (const auto& [slot, value] : e.slots)
            if (const spec::Reference* r = std::get_if<spec::Reference>(&value)) refs.push_back(*r);
        option_references(e.options, &refs);
    }
    std::set<std::string> seen;
    for (const std::string& b : bindings) {
        const Chain c = chain_of(s, split_binding(b).source, sources);
        for (const spec::Source* p : c.path)
            if (seen.insert(p->id).second)
                for (const spec::Transform& t : p->transforms)
                    for (const auto& [key, ref] : t.references) refs.push_back(ref);
    }
    if (refs.empty()) return 0;
    std::uint64_t h = 0x5EF5u;
    for (const spec::Reference& r : refs) h = mix_generation(h, ctx->values->version(r));
    return h;
}

// A panel's generation from its served sources' mix, the cursor and the
// versions of the references it reads; 0 when it reads neither.
std::uint64_t panel_mix(bool served, std::uint64_t generation, bool history, const ChainContext* ctx,
                        std::uint64_t references) {
    if (!served && references == 0) return 0;
    const std::uint64_t g = served ? mix_cursor(generation, history, ctx) : 0;
    return references ? mix_generation(g, references) : g;
}

// The value of a tensor chain: the tensor, the step of each row after a
// window, the geometry arrays after an operator, and what the cursor chose.
struct ChainValue {
    TensorData tensor;
    TensorData steps;
    GeometryArrays geometry;
    std::optional<std::int64_t> step;
    bool live_only = false;
};

class PanelResolver {
public:
    PanelResolver(const spec::Spec& s, LoaderStore& store, const ResolveLimits& limits, DataSources* sources,
                  const ChainContext* chain)
        : spec_(s), store_(store), limits_(limits), sources_(sources), chain_(chain ? *chain : ChainContext{}),
          chain_given_(chain != nullptr) {}

    PanelData run(std::string_view panel_id) {
        PanelData out;
        out.panel = std::string(panel_id);
        const spec::Panel* panel = spec_.find_panel(panel_id);
        if (!panel) {
            out.error = Error("panel.not_found", "The open spec has no panel " + out.panel + ".", out.panel);
            return out;
        }
        out.primitive = panel->primitive;
        const ChainContext* ctx = chain_given_ ? &chain_ : nullptr;
        // A scene panel draws a live scene of the attached script, which the
        // app's live views hold, and the entities that it builds from the
        // spec's sources (design/instrument-interface.md, Scene Entities).
        if (panel->primitive == spec::Primitive::scene) {
            resolve_entities(*panel, out);
            return out;
        }
        Error reference_error;
        const std::vector<BoundSlot> bound = panel_bindings(*panel, ctx, &reference_error);
        if (reference_error) {
            out.error = reference_error;
            if (sources_) out.generation = panel_generation(spec_, panel_id, *sources_, ctx);
            return out;
        }
        // First, what each tabular root must give, so that each is read once.
        for (const BoundSlot& b : bound)
            if (Error err = note_binding(*panel, b.slot, b.binding)) {
                out.error = err;
                if (sources_) out.generation = panel_generation(spec_, panel_id, *sources_, ctx);
                return out;
            }
        bool served = false;
        bool history = false;
        std::uint64_t generation = 0;
        for (const BoundSlot& b : bound) {
            SlotData data;
            data.slot = b.slot;
            data.index = b.index;
            data.binding = b.binding;
            const Binding parts = split_binding(data.binding);
            data.source = parts.source;
            data.column = parts.column;
            bool slot_served = false;
            if (Error err = fill(*panel, data, &slot_served)) {
                out.error = std::move(err);
                out.slots.clear();
                if (sources_) out.generation = panel_generation(spec_, panel_id, *sources_, ctx);
                return out;
            }
            history = history || reads_kept_steps(spec_, chain_of(data.source));
            if (slot_served) {
                served = true;
                generation = mix_generation(generation, data.generation);
            }
            out.slots.push_back(std::move(data));
        }
        std::vector<std::string> bindings;
        for (const BoundSlot& b : bound) bindings.push_back(b.binding);
        out.generation = panel_mix(served, generation, history, ctx,
                                   reference_versions(spec_, *panel, bindings, ctx, sources_));
        return out;
    }

private:
    Chain chain_of(const std::string& id) const { return ::caliper_builder::chain_of(spec_, id, sources_); }

    // Notes what a tabular root must give for one binding.
    Error note_binding(const spec::Panel& panel, const std::string& slot, const std::string& binding) {
        const Binding parts = split_binding(binding);
        Chain chain = chain_of(parts.source);
        if (chain.error) return chain.error;
        if (chain.kind != RootKind::table && chain.kind != RootKind::sql && chain.kind != RootKind::served_table)
            return {};
        const std::string root = chain.path.front()->id;
        if (!parts.column.empty()) {
            needed_[root].insert(parts.column);
        } else if (slot == "rows" && panel.options.columns) {
            for (const auto& c : *panel.options.columns) needed_[root].insert(c);
        } else {
            whole_.insert(root);
        }
        return {};
    }

    // The arrays of each entity (design/instrument-interface.md, Array
    // Resolution): a source id gives one slot for each geometry array of the
    // source, a table gives one slot for each binding. An entity's error
    // does not stop the panel or the other entities.
    void resolve_entities(const spec::Panel& panel, PanelData& out) {
        const ChainContext* ctx = chain_given_ ? &chain_ : nullptr;
        std::map<std::size_t, Error> errors;
        const auto bound = entity_bindings(panel, ctx, &errors);
        for (const auto& [i, b] : bound)
            if (!errors.count(i))
                if (Error err = note_binding(panel, b.slot, b.binding)) errors[i] = err;
        out.entities.resize(panel.entities.size());
        bool served = false;
        bool history = false;
        std::uint64_t generation = 0;
        for (std::size_t i = 0; i < panel.entities.size(); ++i) out.entities[i].path = panel.entities[i].path;
        for (const auto& [i, b] : bound) {
            EntityData& e = out.entities[i];
            if (errors.count(i)) continue;
            SlotData data;
            data.slot = b.slot;
            data.binding = b.binding;
            const Binding parts = split_binding(b.binding);
            data.source = parts.source;
            data.column = parts.column;
            bool slot_served = false;
            if (Error err = fill(panel, data, &slot_served)) {
                errors[i] = err;
                continue;
            }
            history = history || reads_kept_steps(spec_, chain_of(data.source));
            if (slot_served) {
                served = true;
                generation = mix_generation(generation, data.generation);
                e.generation = mix_generation(e.generation, data.generation);
            }
            if (!b.slot.empty()) {
                e.slots.push_back(std::move(data));
                continue;
            }
            // A source id: each geometry array binds the slot of its name.
            if (data.geometry.empty()) {
                errors[i] = Error("entity.shape",
                                  "The source " + data.source +
                                      " gives a tensor, not geometry; end it with surface or path, or bind the "
                                      "entity's slots in a table.",
                                  panel.entities[i].path);
                continue;
            }
            for (const auto& [name, array] : data.geometry) {
                SlotData one;
                one.slot = name;
                one.binding = data.source + "." + name;
                one.source = data.source;
                one.column = name;
                one.kind = SlotData::Kind::tensor;
                one.tensor = array;
                one.steps = data.steps;
                one.step = data.step;
                one.live_only = data.live_only;
                one.generation = data.generation;
                e.slots.push_back(std::move(one));
            }
        }
        for (auto& [i, err] : errors) {
            out.entities[i].slots.clear();
            out.entities[i].error = err;
            if (out.entities[i].error.where.empty()) out.entities[i].error.where = panel.entities[i].path;
        }
        limit_panel_arrays(panel, out);
        std::vector<std::string> bindings;
        for (const auto& [i, b] : bound) bindings.push_back(b.binding);
        out.generation = panel_mix(served, generation, history, ctx,
                                   reference_versions(spec_, panel, bindings, ctx, sources_));
    }

    // The arrays of one panel (design/instrument-interface.md, Entity
    // Limits), counted as the float32 values that the app converts them to,
    // in entity order, before the app stages any of them. An entity whose
    // arrays would take the panel past the limit has the error entity.limit,
    // which names it, and keeps no arrays; an entity after it that fits
    // still resolves.
    static void limit_panel_arrays(const spec::Panel& panel, PanelData& out) {
        constexpr std::uint64_t kLimit = static_cast<std::uint64_t>(spec::limits::kPanelArrayBytes);
        std::uint64_t total = 0;
        for (EntityData& e : out.entities) {
            if (e.error) continue;
            std::uint64_t values = 0;
            for (const SlotData& s : e.slots) {
                values += s.kind == SlotData::Kind::tensor ? s.tensor.numel() : s.values.size();
                values += s.steps.numel();
                for (const auto& [name, array] : s.geometry) values += array.numel();
            }
            const std::uint64_t bytes = values > kLimit ? kLimit + 1 : values * 4;
            if (bytes > kLimit - total) {
                e.slots.clear();
                e.error = Error("entity.limit",
                                "The arrays of the panel " + panel.id + " pass the limit of 64 MiB at the entity " +
                                    e.path + ", which adds " + std::to_string(bytes) + " bytes to " +
                                    std::to_string(total) + ".",
                                e.path);
                continue;
            }
            total += bytes;
        }
    }

    Error fill(const spec::Panel& panel, SlotData& data, bool* served) {
        Chain chain = chain_of(data.source);
        if (chain.error) return chain.error;
        *served = is_served(chain.kind);
        if (chain.kind == RootKind::tensor || chain.kind == RootKind::served_tensor) {
            ChainValue v;
            if (Error err = value_of(chain, &v, &data.generation)) return err;
            data.kind = SlotData::Kind::tensor;
            data.step = v.step;
            data.live_only = v.live_only;
            data.steps = v.steps;
            data.geometry = v.geometry;
            if (!data.column.empty()) {
                // A geometry array by name, and the columns of a window: the
                // step of each row and the stacked value.
                const auto array = v.geometry.find(data.column);
                if (array != v.geometry.end()) {
                    data.tensor = array->second;
                    return {};
                }
                if (v.steps.valid() && data.column == "step") {
                    data.tensor = v.steps;
                    return {};
                }
                if (v.steps.valid() && data.column == "value") {
                    data.tensor = std::move(v.tensor);
                    return {};
                }
                return Error("resolve.not_tabular", "The source " + data.source + " is a tensor and has no columns.",
                             data.binding);
            }
            data.tensor = std::move(v.tensor);
            return {};
        }
        for (const spec::Source* s : chain.path)
            if (!s->transforms.empty())
                return Error("resolve.not_a_tensor", "The source " + s->id + " is tabular, so it takes no transforms.",
                             s->id);
        const QueryResult* q = nullptr;
        if (Error err = table_of(chain, &q)) return err;
        // While the cursor is held, a table with a step column shows its rows
        // up to the cursor (design/instrument-interface.md, Latest at the
        // Cursor: a series shows the points up to the cursor).
        QueryResult held;
        if (chain_.history && !chain_.cursor_live) {
            std::optional<std::size_t> step_column;
            for (std::size_t c = 0; c < q->columns.size(); ++c)
                if (q->columns[c].name == "step") step_column = c;
            if (step_column) {
                held.columns = q->columns;
                held.truncated = q->truncated;
                held.rows = json::array();
                for (const auto& row : q->rows) {
                    const json cell = row.is_array() && *step_column < row.size() ? row[*step_column] : json();
                    if (cell.is_number() && cell.get<double>() > static_cast<double>(chain_.cursor_step)) continue;
                    held.rows.push_back(row);
                }
                q = &held;
                data.step = chain_.cursor_step;
            }
        }
        auto column_index = [&](const std::string& name, std::size_t* out) -> Error {
            for (std::size_t c = 0; c < q->columns.size(); ++c)
                if (q->columns[c].name == name) {
                    *out = c;
                    return {};
                }
            return Error("resolve.unknown_column", "The source " + data.source + " has no column " + name + ".",
                         data.binding);
        };
        data.truncated = q->truncated || truncated_.count(chain.path.front()->id) > 0;
        if (*served) data.generation = table_generations_[chain.path.front()->id];
        if (!data.column.empty()) {
            std::size_t c = 0;
            if (Error err = column_index(data.column, &c)) return err;
            data.kind = SlotData::Kind::column;
            data.columns = {q->columns[c]};
            data.values = json::array();
            for (const auto& row : q->rows) data.values.push_back(row.is_array() && c < row.size() ? row[c] : json());
            return {};
        }
        data.kind = SlotData::Kind::table;
        std::vector<std::size_t> picked;
        if (data.slot == "rows" && panel.options.columns) {
            for (const auto& name : *panel.options.columns) {
                std::size_t c = 0;
                if (Error err = column_index(name, &c)) return err;
                picked.push_back(c);
            }
        } else {
            for (std::size_t c = 0; c < q->columns.size(); ++c) picked.push_back(c);
        }
        for (std::size_t c : picked) data.columns.push_back(q->columns[c]);
        data.values = json::array();
        for (const auto& row : q->rows) {
            json kept = json::array();
            for (std::size_t c : picked) kept.push_back(row.is_array() && c < row.size() ? row[c] : json());
            data.values.push_back(std::move(kept));
        }
        return {};
    }

    std::uint64_t served_generation(const Chain& chain) const {
        if (!sources_ || chain.kind != RootKind::served_tensor) return 0;
        return sources_->generation(chain.catalog);
    }

    // The tensor of a served root (a live view) through its shipped steps,
    // from the cache when the chain was computed from the generation its
    // source has now.
    Error served_tensor_of(const Chain& chain, const std::vector<ChainStep>& steps, ChainValue* out,
                           std::uint64_t* generation) {
        const spec::Source& root = *chain.path.front();
        std::shared_ptr<CatalogDataSource> source = sources_->get(chain.catalog_kind);
        if (!source)
            return Error("resolve.unavailable", "No data source is registered for the source " + root.id + ".", root.id);
        bool cached_form = true;
        for (const ChainStep& s : steps) cached_form = cached_form && !is_operator(s.op());
        const std::string key = cached_form ? chain_key(chain, chain_) : std::string();
        const std::uint64_t now = source->generation(chain.catalog);
        if (cached_form && now != 0 && sources_->find_cached(key, now, &out->tensor)) {
            *generation = now;
            return {};
        }
        ServedRequest request;
        request.max_rows = limits_.max_rows;
        request.max_cells = limits_.max_cells;
        request.max_bytes = limits_.max_tensor_bytes;
        ServedData data = source->read(chain.catalog, request);
        if (data.error) {
            Error err = data.error;
            if (err.where.empty()) err.where = root.id;
            return err;
        }
        if (data.tabular || !data.tensor.valid())
            return Error("resolve.not_a_tensor", "The served source " + chain.catalog + " gave no tensor.", root.id);
        if (data.tensor.nbytes() > limits_.max_tensor_bytes)
            return Error("resolve.too_large",
                         "The source " + *root.catalog + " gave a tensor of " + std::to_string(data.tensor.nbytes()) +
                             " bytes, past the limit of " + std::to_string(limits_.max_tensor_bytes) + ".",
                         root.id);
        out->tensor = std::move(data.tensor);
        if (Error err = run_steps(steps, 0, *out)) return err;
        if (cached_form) sources_->store_cached(key, data.generation, out->tensor);
        *generation = data.generation;
        return {};
    }

    // The newest value of a chain: its root's newest data through every step.
    Error live_value(const Chain& chain, const std::vector<ChainStep>& steps, ChainValue* out,
                     std::uint64_t* generation) {
        if (chain.kind == RootKind::served_tensor) return served_tensor_of(chain, steps, out, generation);
        const spec::Source& root = *chain.path.front();
        TensorResult loaded = store_.load_tensor(*root.catalog);
        if (loaded.error) return loaded.error;
        out->tensor = std::move(loaded.tensor);
        *generation = 0;
        return run_steps(steps, 0, *out);
    }

    // The value of a tensor chain (design/instrument-interface.md, Latest at
    // the Cursor): after the chain's last window, its stack of kept steps
    // through the steps after it; while the cursor is held, the newest kept
    // value at or before the cursor of the nearest source that keeps steps,
    // through the steps after its history point; else the newest value.
    Error value_of(const Chain& chain, ChainValue* out, std::uint64_t* generation) {
        const std::string bound = chain.path.back()->id;
        const auto cached = values_.find(bound);
        if (cached != values_.end()) {
            *out = cached->second.first;
            *generation = cached->second.second;
            return {};
        }
        const std::vector<ChainStep> steps = chain_steps(chain);
        std::optional<std::size_t> window;
        for (std::size_t i = 0; i < steps.size(); ++i)
            if (steps[i].op() == spec::TransformOp::window) window = i;
        ChainValue v;
        std::uint64_t g = 0;
        if (window) {
            const ChainStep& w = steps[*window];
            spec::Transform literal;
            if (Error err = literal_step(w, &literal)) return err;
            if (Error err = apply_window(literal, *w.source, chain_, &v.tensor, &v.steps)) {
                if (err.where.empty()) err.where = step_where(w);
                return err;
            }
            if (Error err = run_steps(steps, *window + 1, v)) return err;
            g = served_generation(chain);
        } else if (chain_.history && !chain_.cursor_live && chain.kind == RootKind::served_tensor) {
            const spec::Source* keeper = nullptr;
            for (auto it = chain.path.rbegin(); it != chain.path.rend() && !keeper; ++it)
                if (history_depth(spec_, **it) > 0) keeper = *it;
            std::optional<std::pair<std::int64_t, TensorData>> kept;
            if (keeper) kept = chain_.history->at(keeper->id, chain_.cursor_step);
            if (kept) {
                v.tensor = std::move(kept->second);
                v.step = kept->first;
                if (Error err = run_steps(steps, history_point(spec_, *keeper), v)) return err;
                g = served_generation(chain);
            } else {
                if (Error err = live_value(chain, steps, &v, &g)) return err;
                v.live_only = true;
            }
        } else {
            if (Error err = live_value(chain, steps, &v, &g)) return err;
        }
        values_[bound] = {v, g};
        *generation = g;
        *out = std::move(v);
        return {};
    }

    static std::string step_where(const ChainStep& s) {
        return "source " + s.source->id + ", transform[" + std::to_string(s.index) + "]";
    }

    // The step with each reference replaced by the literal it reads now
    // (chain_hooks.h).
    Error literal_step(const ChainStep& s, spec::Transform* out) const {
        const spec::Transform& t = s.source->transforms[s.index];
        if (t.references.empty()) {
            *out = t;
            return {};
        }
        if (Error err = substitute_references(t, chain_, out)) {
            err.where = step_where(s);
            return err;
        }
        return {};
    }

    // Applies the chain's steps from `from` to the end, refusing a result
    // past max_tensor_bytes before it is made. An operator gives geometry,
    // and no shipped step may follow it.
    Error run_steps(const std::vector<ChainStep>& steps, std::size_t from, ChainValue& v) {
        for (std::size_t i = from; i < steps.size(); ++i) {
            const ChainStep& s = steps[i];
            const std::string where = step_where(s);
            spec::Transform literal;
            if (Error err = literal_step(s, &literal)) return err;
            const spec::TransformOp op = literal.op;
            if (op == spec::TransformOp::window)
                return Error("spec.bad_transform", "A window is the first step of a source.", where);
            if (!v.geometry.empty())
                return Error("spec.bad_transform", "A step after an operator has no tensor to read.", where);
            if (is_operator(op)) {
                if (Error err = apply_operator(literal, v.tensor, v.steps.valid() ? &v.steps : nullptr, &v.geometry)) {
                    if (err.where.empty()) err.where = where;
                    return err;
                }
                continue;
            }
            spec::Source one;
            one.id = s.source->id;
            one.transforms = {literal};
            json record = spec::to_json(one).value("transform", json::array()).at(0);
            record.erase("x");
            TensorTransform tt;
            if (Error err = transform_from_json(record, &tt)) {
                err.where = where;
                return err;
            }
            std::vector<int64_t> shape;
            DType dtype = DType::f32;
            if (Error err = transform_result_shape(v.tensor.shape(), v.tensor.dtype(), tt, &shape, &dtype)) {
                err.where = where;
                return err;
            }
            uint64_t numel = 0;
            uint64_t nbytes = 0;
            if (Error err = check_shape(shape, dtype, &numel, &nbytes)) {
                err.where = where;
                return err;
            }
            if (nbytes > limits_.max_tensor_bytes)
                return Error("resolve.too_large",
                             "The transform would make a tensor of " + std::to_string(nbytes) +
                                 " bytes, past the limit of " + std::to_string(limits_.max_tensor_bytes) + ".",
                             where);
            TensorData rows = v.steps.valid() ? rows_after(tt, v.tensor.rank(), v.steps) : TensorData();
            TensorResult r = apply_transform(v.tensor, tt);
            if (r.error) {
                r.error.where = where;
                return r.error;
            }
            v.tensor = std::move(r.tensor);
            v.steps = std::move(rows);
        }
        return {};
    }

    // The steps of a window's rows after one shipped step: kept while the
    // stacked axis 0 stays, sliced with it, and dropped when the step
    // removes it.
    static TensorData rows_after(const TensorTransform& tt, int rank, const TensorData& rows) {
        auto axis0 = [&](int64_t a) { return (a < 0 ? a + rank : a) == 0; };
        bool touches = false;
        for (int64_t a : tt.axes) touches = touches || axis0(a);
        if (tt.op == TransformOp::normalise || !touches) return rows;
        if (tt.op == TransformOp::reduce) return TensorData();
        if (tt.index) return TensorData();
        TensorTransform along = tt;
        along.axes = {0};
        TensorResult r = apply_transform(rows, along);
        return r.error ? TensorData() : std::move(r.tensor);
    }
    Error table_of(const Chain& chain, const QueryResult** out) {
        const spec::Source& root = *chain.path.front();
        const auto cached = tables_.find(root.id);
        if (cached != tables_.end()) {
            *out = &cached->second;
            return {};
        }
        QueryResult q;
        const std::uint64_t max_rows = std::max<std::uint64_t>(1, std::min<std::uint64_t>(limits_.max_rows, kMaxQueryRows));
        if (chain.kind == RootKind::served_table) {
            std::shared_ptr<CatalogDataSource> source = sources_->get(chain.catalog_kind);
            if (!source)
                return Error("resolve.unavailable", "No data source is registered for the source " + root.id + ".",
                             root.id);
            ServedRequest request;
            request.max_rows = std::max<std::uint64_t>(1, limits_.max_rows);
            request.max_cells = std::max<std::uint64_t>(1, limits_.max_cells);
            request.max_bytes = limits_.max_tensor_bytes;
            ServedData data = source->read(*root.catalog, request);
            if (data.error) {
                Error err = data.error;
                if (err.where.empty()) err.where = root.id;
                return err;
            }
            if (!data.tabular || !data.rows.is_array())
                return Error("resolve.not_tabular", "The served source " + *root.catalog + " gave no table.", root.id);
            q.columns = std::move(data.columns);
            q.rows = std::move(data.rows);
            q.truncated = data.truncated;
            table_generations_[root.id] = data.generation;
        } else if (chain.kind == RootKind::table) {
            LoaderStore::SourceGet got = store_.describe(*root.catalog);
            if (got.error)
                return Error("resolve.unknown_source",
                             "The catalog has no table " + *root.catalog + "; import the file or run its loader.",
                             root.id);
            if (got.source.kind != "table")
                return Error("resolve.not_tabular", "The catalog source " + *root.catalog + " is not a table.", root.id);
            std::string select;
            std::size_t ncols = got.source.columns ? got.source.columns->size() : 1;
            if (whole_.count(root.id) || !needed_.count(root.id)) {
                select = "*";
            } else {
                for (const auto& c : needed_[root.id]) {
                    if (!select.empty()) select += ", ";
                    select += quote_sql_identifier(c);
                }
                ncols = needed_[root.id].size();
            }
            const std::uint64_t rows = std::max<std::uint64_t>(
                1, std::min<std::uint64_t>(max_rows, limits_.max_cells / std::max<std::size_t>(1, ncols)));
            q = store_.query("SELECT " + select + " FROM " + quote_sql_identifier(got.source.name), rows,
                             limits_.query_timeout_ms);
        } else {
            q = store_.query(*root.sql, max_rows, limits_.query_timeout_ms);
        }
        if (q.error) {
            Error err = q.error;
            if (err.code.rfind("loader.", 0) == 0 || err.code.rfind("query.", 0) == 0) {
                err.message = "The query of the source " + root.id + " failed: " + err.message;
                err.code = err.code == "query.timeout" ? "resolve.timeout" : "resolve.query_failed";
            }
            err.where = root.id;
            return err;
        }
        if (!q.columns.empty()) {
            const std::uint64_t max_rows_by_cells = std::max<std::uint64_t>(1, limits_.max_cells / q.columns.size());
            if (q.rows.size() > max_rows_by_cells) {
                json kept = json::array();
                for (std::size_t r = 0; r < max_rows_by_cells; ++r) kept.push_back(std::move(q.rows[r]));
                q.rows = std::move(kept);
                truncated_.insert(root.id);
            }
        }
        session_detail::fix_nonfinite(q);
        auto [it, inserted] = tables_.emplace(root.id, std::move(q));
        *out = &it->second;
        return {};
    }

    const spec::Spec& spec_;
    LoaderStore& store_;
    const ResolveLimits& limits_;
    DataSources* sources_ = nullptr;
    ChainContext chain_;
    bool chain_given_ = false;   // a caller's context, not the default one
    std::map<std::string, std::set<std::string>> needed_;  // by tabular root
    std::set<std::string> whole_;                          // tabular roots read whole
    std::set<std::string> truncated_;                      // cut at max_cells here
    std::map<std::string, std::pair<ChainValue, std::uint64_t>> values_;  // by bound source, with its generation
    std::map<std::string, QueryResult> tables_;            // by tabular root
    std::map<std::string, std::uint64_t> table_generations_;  // by served tabular root
};

}  // namespace

// ---- DataSources --------------------------------------------------------------

DataSources::DataSources() = default;
DataSources::DataSources(Options options) : options_(options) {}

bool DataSources::set(CatalogKind kind, std::shared_ptr<CatalogDataSource> source) {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (kind) {
        case CatalogKind::live: live_ = std::move(source); break;
        case CatalogKind::feed: feed_ = std::move(source); break;
        case CatalogKind::metric: metric_ = std::move(source); break;
        case CatalogKind::run: run_ = std::move(source); break;
        default: return false;
    }
    cache_.clear();
    uses_.clear();
    cache_bytes_ = 0;
    return true;
}

std::shared_ptr<CatalogDataSource> DataSources::get(CatalogKind kind) const {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (kind) {
        case CatalogKind::live: return live_;
        case CatalogKind::feed: return feed_;
        case CatalogKind::metric: return metric_;
        case CatalogKind::run: return run_;
        default: return nullptr;
    }
}

std::uint64_t DataSources::generation(std::string_view catalog_id) const {
    const CatalogIdParse parsed = parse_catalog_id(catalog_id);
    if (parsed.error) return 0;
    std::shared_ptr<CatalogDataSource> source = get(parsed.kind);
    if (!source) return 0;
    try {
        return source->generation(std::string(catalog_id));
    } catch (...) {
        return 0;
    }
}

DataSources::CacheStats DataSources::cache_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats out;
    out.computed = computed_;
    out.reused = reused_;
    out.entries = cache_.size();
    out.bytes = cache_bytes_;
    return out;
}

bool DataSources::find_cached(const std::string& key, std::uint64_t generation, TensorData* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cache_.find(key);
    if (it == cache_.end() || it->second.generation != generation) return false;
    uses_.splice(uses_.begin(), uses_, it->second.use);
    *out = it->second.tensor;
    ++reused_;
    return true;
}

void DataSources::store_cached(const std::string& key, std::uint64_t generation, const TensorData& tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++computed_;
    const auto old = cache_.find(key);
    if (old != cache_.end()) {
        cache_bytes_ -= old->second.tensor.nbytes();
        uses_.erase(old->second.use);
        cache_.erase(old);
    }
    if (generation == 0 || options_.max_cache_entries == 0 || tensor.nbytes() > options_.max_cache_bytes) return;
    while (!uses_.empty() &&
           (cache_.size() >= options_.max_cache_entries || cache_bytes_ + tensor.nbytes() > options_.max_cache_bytes)) {
        const auto victim = cache_.find(uses_.back());
        cache_bytes_ -= victim->second.tensor.nbytes();
        cache_.erase(victim);
        uses_.pop_back();
    }
    uses_.push_front(key);
    cache_[key] = Cached{generation, tensor, uses_.begin()};
    cache_bytes_ += tensor.nbytes();
}

std::uint64_t panel_generation(const spec::Spec& s, std::string_view panel_id, const DataSources& sources,
                               const ChainContext* chain) {
    const spec::Panel* panel = s.find_panel(panel_id);
    if (!panel) return 0;
    std::vector<std::string> bindings;
    if (panel->primitive == spec::Primitive::scene) {
        for (const auto& [i, b] : entity_bindings(*panel, chain, nullptr)) {
            (void)i;
            bindings.push_back(b.binding);
        }
    } else {
        for (const BoundSlot& b : panel_bindings(*panel, chain, nullptr)) bindings.push_back(b.binding);
    }
    bool served = false;
    bool history = false;
    std::uint64_t generation = 0;
    for (const std::string& b : bindings) {
        const Chain c = chain_of(s, split_binding(b).source, &sources);
        if (c.path.empty()) continue;
        history = history || reads_kept_steps(s, c);
        if (!is_served(c.kind)) continue;
        served = true;
        generation = mix_generation(generation, sources.generation(c.catalog));
    }
    return panel_mix(served, generation, history, chain, reference_versions(s, *panel, bindings, chain, &sources));
}

spec::Panel panel_with_values(const spec::Panel& panel, const ChainContext* chain) {
    spec::Panel out = panel;
    auto read = [&](const spec::Reference& r) {
        json v = chain && chain->values ? chain->values->value(r) : r.fallback;
        return v.is_null() ? r.fallback : v;
    };
    for (const auto& [key, ref] : panel.options.references) {
        const json v = read(ref);
        if (key == "colormap") {
            if (v.is_string()) out.options.colormap = v.get<std::string>();
            continue;
        }
        if (!v.is_array() || v.size() != 2 || !v[0].is_number() || !v[1].is_number()) continue;
        const std::array<double, 2> pair{v[0].get<double>(), v[1].get<double>()};
        if (key == "range") out.options.range = pair;
        else if (key == "x_range") out.options.x_range = pair;
        else if (key == "y_range") out.options.y_range = pair;
    }
    for (spec::Entity& e : out.entities) {
        if (!e.options.is_object()) continue;
        for (auto it = e.options.begin(); it != e.options.end(); ++it) {
            if (!spec::looks_like_reference(it.value())) continue;
            const std::optional<spec::Reference> r = spec::parse_reference(it.value());
            if (!r || !chain || !chain->values) continue;
            const json v = chain->values->value(*r);
            if (!v.is_null()) it.value()["default"] = v;
        }
    }
    return out;
}

const char* slot_kind_name(SlotData::Kind kind) {
    switch (kind) {
        case SlotData::Kind::tensor: return "tensor";
        case SlotData::Kind::column: return "column";
        case SlotData::Kind::table: return "table";
    }
    return "tensor";
}

PanelData resolve_panel(const spec::Spec& s, std::string_view panel_id, LoaderStore& store,
                        const ResolveLimits& limits, DataSources* sources, const ChainContext* chain) {
    try {
        PanelResolver r(s, store, limits, sources, chain);
        return r.run(panel_id);
    } catch (const std::exception& e) {
        PanelData out;
        out.panel = std::string(panel_id);
        out.error = Error("resolve.failed", std::string("The panel's data could not be resolved: ") + e.what());
        return out;
    }
}

json panel_data_summary(const PanelData& data, std::size_t max_values) {
    json slots = json::array();
    for (const SlotData& s : data.slots) {
        json one{{"slot", s.slot},
                 {"index", s.index},
                 {"binding", s.binding},
                 {"kind", slot_kind_name(s.kind)},
                 {"truncated", s.truncated}};
        json values = json::array();
        if (s.kind == SlotData::Kind::tensor) {
            one["dtype"] = dtype_name(s.tensor.dtype());
            one["shape"] = s.tensor.shape();
            for (uint64_t i = 0; i < s.tensor.numel() && values.size() < max_values; ++i)
                values.push_back(tensor_element_json(s.tensor, i));
        } else {
            json cols = json::array();
            for (const auto& c : s.columns) cols.push_back(c.to_json());
            one["columns"] = std::move(cols);
            one["rows"] = s.values.size();
            for (const auto& v : s.values) {
                if (values.size() >= max_values) break;
                values.push_back(v);
            }
        }
        one["values"] = std::move(values);
        slots.push_back(std::move(one));
    }
    json out{{"panel", data.panel}, {"primitive", spec::to_string(data.primitive)}, {"slots", std::move(slots)}};
    if (data.error) out["error"] = {{"code", data.error.code}, {"message", data.error.message}, {"where", data.error.where}};
    return out;
}

}  // namespace caliper_builder
