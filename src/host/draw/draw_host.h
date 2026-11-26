#pragma once
// caliper_draw: the host of the draw halves of a spec's programs in the app's
// scene views (design/programmable-graphics.md, Program Instances, Render
// Graph, Draw State and Safety of Draws).
//
// A scene view (live/live_scene.h) owns one DrawRuntime for each program that
// its panel draws, on its own backend, and asks a DrawFeed for the programs
// and for each frame's values beside its own camera, time and target: the
// resource versions at the cursor, the packed params, the selection, the
// cursor and the epoch. The view reports each change of a program's state
// back to the feed.
//
// DrawHost is the app's feed for every scene panel: it takes the programs of
// a spec with a pinned piece and a panel, resolves each pin to its piece
// through the app's piece store, checks that the programs of one panel agree
// on the camera, reads the resource versions of the step history and the
// values of params' references through functions that the app gives, and
// pushes a `program` event with half "draw" for each change of a program's
// state or warning: compiled, warned, refused or stopped, with the code, the
// message, the pass and its entry. status() gives the `draw` object of
// program.status.
//
// Threads: every function on the frame thread, but status(), which any
// thread may call.
#include "draw/draw_runtime.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/spec/spec.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace caliper_draw {

// One program that a scene panel draws: its piece (manifest and files), its instance, the pick
// id that the panel gives out above its entities' ids, and a revision that changes when the
// program or its piece changes, so that the view installs it again.
struct PanelProgram {
    caliper_builder::spec::Piece piece;
    Instance instance;
    std::uint32_t pick_id = 0;
    std::uint64_t revision = 0;
    // A refusal before any runtime: an unresolved pin or a camera conflict. The view does not
    // install the program and reports this verdict as its state.
    Verdict refused;
};

// What a scene view draws of programs, and where it reports them.
class DrawFeed {
public:
    virtual ~DrawFeed() = default;
    // Changes whenever programs() changes.
    virtual std::uint64_t revision() const = 0;
    // The programs of the panel, in draw order.
    virtual std::vector<PanelProgram> programs() const = 0;
    // Before each frame of one program: gives the runtime the resource versions at the cursor
    // (DrawRuntime::set_resource) and fills the values that the view does not own (params,
    // selection, cursor, latest step, live, cursor_moved and epoch).
    virtual void frame(DrawRuntime& runtime, FrameInputs* in) = 0;
    // The program's state changed (installed, refused, running, stopped).
    virtual void status(const DrawRuntime& runtime) = 0;
    // The checks that the view's runtimes run before they make anything.
    virtual DrawChecks checks() const { return DrawChecks::pieces(); }
};

// The `draw` object of program.status for one program.
nlohmann::json draw_status_json(const DrawStatus& status);
// The `program` event of a change of a program's draw half: change "compiled", "warned", "refused"
// or "stopped", with half "draw", and the pass, its entry, the code and the message.
caliper_builder::Event draw_event(const std::string& program, const std::string& panel, const DrawStatus& status,
                                  const char* change);

// The app's host of draw halves.
class DrawHost {
public:
    // Resolves a program's pin to its piece (library, workspace or the spec's inline pieces);
    // a refusal is the program's state, with piece.missing or piece.hash_mismatch.
    using Resolver = std::function<Verdict(const caliper_builder::spec::Spec& spec,
                                           const caliper_builder::spec::Program& program,
                                           caliper_builder::spec::Piece* piece)>;
    // The newest kept version at or before `cursor` of a program's published resource, and the
    // epoch of the attachment that sent it; false when there is none. The bytes stay valid until
    // the next call. The version's serial tells a version that replaced another at the same
    // step, which the host gives again.
    using Resources = std::function<bool(const std::string& program, const std::string& resource,
                                         std::int64_t cursor, ResourceVersion* out, std::uint64_t* epoch)>;
    // The value of a params' reference (a control, a selection or a state), as the session's
    // values give it; null gives each reference the parameter's default.
    using Evaluate = std::function<nlohmann::json(const caliper_builder::spec::Reference&)>;
    // The cursor of the step history: its step, the newest step, and whether it is live.
    struct Cursor {
        std::int64_t step = -1;
        std::int64_t latest = -1;
        bool live = true;
        std::uint64_t moves = 0;   // counts moves by a command, a key or the transport
    };

    DrawHost();
    ~DrawHost();

    void set_resolver(Resolver resolver);
    void set_resources(Resources resources);
    void set_evaluate(Evaluate evaluate);
    void set_events(caliper_builder::EventSink* sink);
    void set_cursor(const Cursor& cursor);
    // The linked selection that every program reads (caliper_selection).
    void set_selection(std::vector<Selected> selection);

    // Takes the programs of the spec that pin a piece and name a panel. A program whose pin does
    // not resolve, or whose camera differs from another program's of the same panel, is refused
    // (program.camera_conflict) and pushes its event.
    void reconcile(const caliper_builder::spec::Spec& spec);
    // The feed of one scene panel; it lives as long as the host.
    DrawFeed* feed(const std::string& panel);
    // The last state of each program's draw half, by program id.
    std::map<std::string, DrawStatus> status() const;

private:
    class Panel;
    struct Record {
        std::string program;
        std::string panel;
        PanelProgram draw;
        caliper_builder::spec::Program spec;
    };
    void report(const std::string& program, const std::string& panel, const DrawStatus& status);
    std::vector<std::uint8_t> pack_params(const Record& r) const;

    Resolver resolver_;
    Resources resources_;
    Evaluate evaluate_;
    caliper_builder::EventSink* events_ = nullptr;
    Cursor cursor_;
    std::vector<Selected> selection_;
    std::vector<Record> records_;
    std::map<std::string, std::unique_ptr<Panel>> panels_;
    std::uint64_t revision_ = 0;
    std::uint32_t next_pick_ = 1u << 24;   // program pick ids, above the entities' ids
    mutable std::mutex mutex_;
    std::map<std::string, DrawStatus> status_;
    std::map<std::string, DrawState> reported_;
    std::map<std::string, std::string> reported_warning_;
};

}  // namespace caliper_draw
