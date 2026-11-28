#pragma once
// caliper_builder: the commands of design/instrument-interface.md (section
// Commands), as functions that the session binds.
//
// The edits (state.create, state.patch, state.remove, entity.set,
// entity.remove, annotation.set, annotation.remove and camera.save) are
// pure functions of a spec and the arguments, like apply_control_edit();
// apply_spec_edit() hands them over, and they follow the rules of the panel
// edits. The session gives camera.save the argument `orbit`, which it reads
// from the app with SessionHost::camera_command("camera.get") before the
// edit runs, so that a replay of the journal makes the same edit.
//
// The other commands change or read session state (values.h, history.h).
// camera.get and camera.set need the app's view and go to the app through
// SessionHost::camera_command().
//
// A command that is not built yet answers command.not_implemented. Each
// function is defined in the file named beside it. Nothing here throws.
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include <caliper_builder/commands/registry.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>

namespace caliper_builder {

// The error of a command that is not built yet: command.not_implemented.
Error not_implemented(std::string_view name);

// state_commands.cpp: state.create, state.patch and state.remove.
SpecEdit apply_state_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                          const spec::SourceResolver& resolver);
// entity_commands.cpp: entity.set and entity.remove.
SpecEdit apply_entity_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                           const spec::SourceResolver& resolver);
// camera_commands.cpp: camera.save, with `orbit` from the app.
SpecEdit apply_camera_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                           const spec::SourceResolver& resolver);
// annotation_commands.cpp: annotation.set and annotation.remove.
SpecEdit apply_annotation_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                               const spec::SourceResolver& resolver);

// What the session commands read and change. `spec` is a copy of the open
// spec taken under the document's lock; it is empty when no spec is open.
struct InteractiveContext {
    const spec::Spec& spec;
    bool spec_open = false;
    ValueStore& values;
    StepHistory& history;
    EventSink* events = nullptr;
    // selection.set resolves the panel to find the picked item's fields.
    LoaderStore* store = nullptr;
    DataSources* sources = nullptr;
    ResolveLimits limits;
    // The workspace folder, where selection.set finds a program's workspace piece.
    std::string workspace;
};

// state_commands.cpp: state.set, state.get and control.set.
CommandOutput state_command(const CommandContext& ctx, InteractiveContext& ic);
// selection_commands.cpp: selection.set and selection.get.
CommandOutput selection_command(const CommandContext& ctx, InteractiveContext& ic);
// cursor_commands.cpp: cursor.get, cursor.set and cursor.step.
CommandOutput cursor_command(const CommandContext& ctx, InteractiveContext& ic);

}  // namespace caliper_builder
