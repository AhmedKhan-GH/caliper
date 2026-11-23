#pragma once
// caliper_builder: the current values of the inputs that binding references
// read (design/instrument-interface.md, sections Binding References,
// Re-Evaluation and Shared State).
//
// One ValueStore per session holds, for the open spec, the value of each
// control bound to `value` or to a state, each state variable, the last pick
// of each panel and the cursor. A change of an input bumps its version and
// marks its readers in the DependencyIndex; each marked reader evaluates
// again at the next frame, at most once. The spec never saves these values.
//
// Threads: a ValueStore is shared by the command thread and the frame
// thread, and every member function takes its own lock.
//
// The session creates both classes (session.h, Session::values()), and
// values.cpp defines them.
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/spec/spec.h>

namespace caliper_builder {

struct CursorState;  // history.h

// The fields of one pick (design/instrument-interface.md, Pick Fields). A
// field that does not apply to the draw kind is empty.
struct PickFields {
    std::string entity;                                // the entity's path
    std::optional<std::int64_t> index;                 // points, lines, live entities
    std::optional<std::int64_t> row_id;
    std::optional<std::int64_t> row, column;           // surface: the cell before pooling
    std::optional<std::array<double, 3>> position;     // x, y, z in data units
    std::optional<double> value;                       // the color value at the pick
    std::optional<std::int64_t> step;                  // after a window
    std::optional<std::string> annotation;             // a pick on a label or a marker
    // A pick of a program's item (design/programmable-graphics.md, Picking): the program, the
    // element of its item map, the map's fields by name, and its first four integer fields as
    // tensor indices, -1 where there are fewer.
    std::optional<std::string> program;
    std::optional<std::int64_t> item;
    std::vector<std::pair<std::string, double>> program_fields;
    std::optional<std::array<std::int64_t, 4>> tensor_index;
};
// The fields as the `pick` event and selection.get give them; absent fields
// are left out.
nlohmann::json pick_fields_json(const PickFields& fields);

// One input of the dependency index: a control's value, a state variable, a
// panel's last pick, the cursor, or the generation of a source.
struct InputKey {
    enum class Kind { control, state, selection, cursor, source };
    Kind kind = Kind::state;
    std::string id;
    bool operator<(const InputKey& o) const { return kind != o.kind ? kind < o.kind : id < o.id; }
    bool operator==(const InputKey& o) const { return kind == o.kind && id == o.id; }
};

// One reader of references: a source step, a panel's options or slot, an
// entity, an annotation, a bound text, or the params of a program.
// `path` is its document path ("source[2].transform[0]", "panel[1].options").
struct ReaderId {
    std::string path;
    bool operator<(const ReaderId& o) const { return path < o.path; }
    bool operator==(const ReaderId& o) const { return path == o.path; }
};

class ValueStore {
public:
    ValueStore();
    ~ValueStore();
    ValueStore(const ValueStore&) = delete;
    ValueStore& operator=(const ValueStore&) = delete;

    // Starts over for a newly opened or edited spec: each state variable at
    // its initial value, each control at its start value, no picks. A value
    // of an input that the new spec still has with the same type stays.
    void reset(const spec::Spec& spec);

    // The value that `ref` reads now: the input's value or field, else its
    // default, else null.
    nlohmann::json value(const spec::Reference& ref) const;
    // A number that grows on each change of the input that `ref` reads.
    std::uint64_t version(const spec::Reference& ref) const;

    // state.set and the writers of a state variable. Refuses the cursor
    // (state.read_only), an unknown id (state.not_found) and a value that
    // does not fit (state.bad_value). Pushes one `state` event to `events`
    // when the value changed.
    Error set_state(const std::string& id, const nlohmann::json& value, const std::string& writer,
                    CommandOrigin origin, EventSink* events);
    // The value of a control bound to `value` or to a state (control.set and
    // a person's edit).
    Error set_control(const std::string& id, const nlohmann::json& value, CommandOrigin origin, EventSink* events);
    nlohmann::json control_value(const std::string& id) const;

    // Records the last pick of a panel and runs the panel's `select` write.
    void record_pick(const std::string& panel, const PickFields& fields, CommandOrigin origin, EventSink* events);
    std::optional<PickFields> last_pick(const std::string& panel) const;
    void clear_pick(const std::string& panel);

    // The cursor as the history keeps it (history.h); state.set refuses the
    // id "cursor", and only the history writes it.
    void set_cursor(const CursorState& cursor);

    // Each state variable with its value, its default, its writers and its
    // readers, for state.get.
    nlohmann::json describe_states(const spec::Spec& spec) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The readers of each input, built after each spec change.
class DependencyIndex {
public:
    void rebuild(const spec::Spec& spec);
    std::vector<ReaderId> readers_of(const InputKey& input) const;
    // spec.reference_cycle: each cycle through a control's `follow` write.
    std::vector<spec::Diagnostic> cycles() const;

private:
    std::map<InputKey, std::vector<ReaderId>> readers_;
};

}  // namespace caliper_builder
