// The current values of the inputs that references read
// (caliper_builder/session/values.h). Not built yet: the store holds no
// values, every reference reads its default, and the setters answer
// command.not_implemented.
#include <caliper_builder/session/values.h>

#include <caliper_builder/session/history.h>

namespace caliper_builder {

struct ValueStore::Impl {};

ValueStore::ValueStore() : impl_(std::make_unique<Impl>()) {}
ValueStore::~ValueStore() = default;

void ValueStore::reset(const spec::Spec& spec) { (void)spec; }

nlohmann::json ValueStore::value(const spec::Reference& ref) const { return ref.fallback; }

std::uint64_t ValueStore::version(const spec::Reference& ref) const {
    (void)ref;
    return 0;
}

Error ValueStore::set_state(const std::string& id, const nlohmann::json& value, const std::string& writer,
                            CommandOrigin origin, EventSink* events) {
    (void)id;
    (void)value;
    (void)writer;
    (void)origin;
    (void)events;
    return Error("command.not_implemented", "State variables are not built yet.", "state");
}

Error ValueStore::set_control(const std::string& id, const nlohmann::json& value, CommandOrigin origin,
                              EventSink* events) {
    (void)id;
    (void)value;
    (void)origin;
    (void)events;
    return Error("command.not_implemented", "Control values are not built yet.", "control");
}

nlohmann::json ValueStore::control_value(const std::string& id) const {
    (void)id;
    return nullptr;
}

void ValueStore::record_pick(const std::string& panel, const PickFields& fields, CommandOrigin origin,
                             EventSink* events) {
    (void)panel;
    (void)fields;
    (void)origin;
    (void)events;
}

std::optional<PickFields> ValueStore::last_pick(const std::string& panel) const {
    (void)panel;
    return std::nullopt;
}

void ValueStore::clear_pick(const std::string& panel) { (void)panel; }

void ValueStore::set_cursor(const CursorState& cursor) { (void)cursor; }

nlohmann::json ValueStore::describe_states(const spec::Spec& spec) const {
    (void)spec;
    return nlohmann::json::array();
}

nlohmann::json pick_fields_json(const PickFields& fields) {
    nlohmann::json out = {{"entity", fields.entity}};
    if (fields.index) out["index"] = *fields.index;
    if (fields.row_id) out["row_id"] = *fields.row_id;
    if (fields.row) out["row"] = *fields.row;
    if (fields.column) out["column"] = *fields.column;
    if (fields.position) {
        out["x"] = (*fields.position)[0];
        out["y"] = (*fields.position)[1];
        out["z"] = (*fields.position)[2];
    }
    if (fields.value) out["value"] = *fields.value;
    if (fields.step) out["step"] = *fields.step;
    if (fields.annotation) out["annotation"] = *fields.annotation;
    return out;
}

void DependencyIndex::rebuild(const spec::Spec& spec) {
    (void)spec;
    readers_.clear();
}

std::vector<ReaderId> DependencyIndex::readers_of(const InputKey& input) const {
    auto it = readers_.find(input);
    return it == readers_.end() ? std::vector<ReaderId>{} : it->second;
}

std::vector<spec::Diagnostic> DependencyIndex::cycles() const { return {}; }

}  // namespace caliper_builder
