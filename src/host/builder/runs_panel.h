#pragma once
// ============================================================================
// runs_panel.h: the runs panel of Hosted Runs (design/hosted-runs.md), where
// the person starts, forks and stops the managed runs of the workspace in the
// app, without an agent and without MCP.
//
// What it shows: a popup over the page, a share of the window, with the runs
// of the model runner (ModelRunner::list()) as a table, one row each: Id,
// State, Script, Parent, Step and Detail (the exit code, "stopping" or why
// the run failed), with Fork and Stop on each managed run that runs. Under the
// table, the form of a start: Script (a field, and the workspace's Python
// scripts as choices), Arguments, Parameters ("lr=0.01 momentum=0.9") and Id,
// which holds a free id from the script's name; Fork on a row turns the form
// into the fork of that run, with its parameters to change.
//
// The review card. A start and a fork go through the same command path as an
// agent's: the commands run.start and run.fork, with the origin ui, through the
// session's queue, so that the session checks the script, the id and the
// parameters and the journal and the Activity pane see them. The session
// parks a start or a fork on its card for every origin (design/hosted-runs.md,
// Review Card), and the panel shows that one card, with the session's text
// (caliper_builder::run_card_text()) and Approve and Cancel, which decide it
// through the session's queue. The approval tray shows the same card, and
// either decides it, so that the person confirms a start once. Stop sends
// run.stop at once, as the person's own click.
//
// Three parts, as the other panes: the state (RunsPanel, with an action for
// each click so that tests drive it without a window), the layout
// (layout_runs_panel(), plain rectangles from measured text) and the ImGui
// side (draw()). The panel never waits for a command: frame() reads the
// answer of the one command in flight when the session's queue has run it.
//
// Threads: frame thread only. Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include <caliper_builder/session/runner_commands.h>

#include <nlohmann/json.hpp>

#include <functional>
#include <future>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {
class Session;
}

namespace caliper_host {

struct SpecLook;

// A rectangle of the layout, in the window's coordinates.
struct RunsRect {
    float x = 0, y = 0, w = 0, h = 0;
    float right() const { return x + w; }
    float bottom() const { return y + h; }
};

// The text width of a string in the panel's font.
using RunsMeasure = std::function<float(const std::string& text)>;

// The two fields read every text as the review card writes it
// (caliper_builder::run_card_text()), so that the card shows the typed text
// as it runs: a bare word is itself, a backslash too, and double quotes keep
// spaces, commas and '=' inside one word and read the escapes of JSON (\",
// \\, \n, \t, \uXXXX and the others).
//
// The Parameters field: pairs name=value, separated by spaces or commas, split
// at the first '=' outside quotes. A value without quotes is a number, true
// or false, or else a string; a value in quotes is a string. The pairs come
// out in the command table's form, [{"name", "value"}], at most kRunMaxParams
// of them. False with *problem for a pair without '=', an empty or long name
// (more than 63 bytes), a name given twice, too many pairs, an open quote or
// an escape that the card does not write.
bool parse_run_params(const std::string& text, nlohmann::json* pairs, std::string* problem);

// The Arguments field: split at spaces, with quotes and escapes as above.
// False with *problem for more than kRunMaxArgs arguments, an open quote or an
// escape that the card does not write.
bool split_run_args(const std::string& text, std::vector<std::string>* args, std::string* problem);

// Parameters as the card shows them, which the Parameters field of a fork
// holds and parse_run_params() reads back: "lr=0.05 mode=adam", from an object
// of values by name or from the command table's pairs.
using caliper_builder::run_params_text;

// A free run id for a script, as the model runner chooses one: the file name
// of the script in a-z, 0-9 and '-' (at most 32 bytes, else "run"), then '-'
// and the smallest number from 1 that no run in `runs` has and that has no
// folder runs/<id> in the workspace.
std::string suggest_run_id(const std::string& script, const std::vector<caliper_builder::RunEntry>& runs,
                           const std::string& workspace);

// The Python scripts of the workspace, relative to it with '/', sorted: the
// .py files at most four folders deep, without the folders runs, .venv,
// __pycache__, node_modules and those whose name starts with '.'; at most 200.
std::vector<std::string> workspace_scripts(const std::string& workspace);

// The sizes of the layout at text scale 1; for_text_scale() grows them.
struct RunsPanelMetrics {
    float row = 28;          // a row of the table, the form or the buttons
    float line = 20;         // a line of the card text
    float pad = 16;          // inside the window
    float gap = 8;           // between rows and buttons
    float button_pad = 10;   // inside a button, left and right
    float label = 120;       // the label column of the form
    float min_w = 640;
    float share_w = 0.62f;   // of the work area's width; the height is the content's
    float top_share = 0.12f; // the window's top, as a share of the work area's height
    float margin = 24;       // between the window and the work area's edges
    static RunsPanelMetrics for_text_scale(float s);
};

// What the layout needs to know of the state.
struct RunsPanelLayoutInput {
    RunsRect area;                        // the work area the popup sits in
    int runs = 0;                         // rows of the table
    bool fork = false;                    // the form is a fork: no Script, no Arguments
    std::vector<std::string> scripts;     // the script choices, as their buttons read
    std::string card;                     // the card text; empty while no card is open
    bool problem = false;                 // a line for the last problem
};

// The panel as plain rectangles.
struct RunsPanelLayout {
    RunsRect window, title, close;
    RunsRect header;                      // the table's column titles
    static constexpr int kColumns = 6;    // Id, State, Script, Parent, Step, Detail
    float column_x[kColumns] = {};        // the left edge of each column
    float column_w[kColumns] = {};
    struct Row {
        RunsRect row, fork, stop;
    };
    std::vector<Row> rows;                // the rows that fit
    int rows_folded = 0;                  // the runs that do not fit, "+N"
    RunsRect folded;
    RunsRect empty;                       // "No runs", when there are none
    RunsRect form_title, cancel_fork;     // "Start" or "Fork <run>", and Cancel for a fork
    RunsRect script_label, script;
    std::vector<RunsRect> scripts;        // the script choices that fit on one row
    int scripts_folded = 0;
    RunsRect args_label, args;
    RunsRect params_label, params;
    RunsRect id_label, id, go;            // go: Start or Fork, which opens the card
    RunsRect card;                        // the card's frame
    std::vector<std::string> card_lines;  // its text, wrapped
    RunsRect approve, cancel;
    RunsRect problem;
};

RunsPanelLayout layout_runs_panel(const RunsPanelLayoutInput& in, const RunsPanelMetrics& m, const RunsMeasure& measure);

class RunsPanel {
public:
    RunsPanel() = default;
    RunsPanel(const RunsPanel&) = delete;
    RunsPanel& operator=(const RunsPanel&) = delete;

    bool is_open() const { return open_; }
    // Opening reads the runs and the workspace's scripts again.
    void set_open(bool open);
    // ⌘⇧R (Ctrl+Shift+R elsewhere) opens and closes the panel; esc closes it.
    void read_keys();

    // Once per frame, before draw(): the app's session (null until the
    // workspace is open), its model runner (null without one) and whether a
    // long command holds the session's registry. Reads the runs while the
    // panel is open, the session's card of the command in flight while the
    // registry is free, and the answer of that command.
    void frame(caliper_builder::Session* session, caliper_builder::ModelRunner* runner, bool busy = false);
    // The popup in the work area (x, y, w, h), while open.
    void draw(float x, float y, float w, float h, const SpecLook& look);

    // ---- the actions of the clicks, for draw() and for tests
    void set_script(const std::string& script);    // also suggests a new id
    void set_args(const std::string& text) { args_ = text; }
    void set_params(const std::string& text) { params_ = text; }
    void set_id(const std::string& id);             // the person's own id: no new suggestion
    // Fork on a row: the form becomes the fork of `run`, with its parameters.
    void begin_fork(const std::string& run);
    void cancel_fork();
    // Start or Fork under the form: checks the fields and sends run.start or
    // run.fork with the origin ui; the session's card shows at the next
    // frame() after the queue ran it. False with problem() when a field is
    // wrong, and false while a command is in flight.
    bool review();
    // Approve and Cancel on the card: approve or deny the session's card
    // through its queue. False when no card waits.
    bool approve();
    bool cancel_review();
    // Stop on a row: sends run.stop with the origin ui.
    bool stop(const std::string& run);

    // ---- what it shows
    const std::vector<caliper_builder::RunEntry>& runs() const { return runs_; }
    const std::vector<std::string>& scripts() const { return scripts_; }
    const std::string& script() const { return script_; }
    const std::string& args() const { return args_; }
    const std::string& params() const { return params_; }
    const std::string& id() const { return id_; }
    const std::string& fork_of() const { return fork_of_; }   // "" for a start
    const std::string& card() const { return card_; }         // "" while no card waits
    const std::string& problem() const { return problem_; }
    bool busy() const { return pending_.valid(); }            // a command in flight
    // The id of the last run the panel started or forked, once its command
    // answered.
    const std::string& last_run() const { return last_run_; }
    // Whether Fork and Stop apply to a run.
    static bool can_fork(const caliper_builder::RunEntry& e);
    static bool can_stop(const caliper_builder::RunEntry& e);
    // Every label, value and line the panel draws now, for tests and shots.
    std::vector<std::string> visible_text() const;
    RunsPanelLayoutInput layout_input(const RunsRect& area) const;

private:
    void refresh();
    bool send(const char* name, nlohmann::json args);
    void read_card();
    void forget_card();
    bool decide(bool approve);

    bool open_ = false;
    bool focus_ = false;
    caliper_builder::Session* session_ = nullptr;
    caliper_builder::ModelRunner* runner_ = nullptr;
    std::string workspace_;   // of the scripts read last
    std::vector<caliper_builder::RunEntry> runs_;
    std::vector<std::string> scripts_;
    std::string script_, args_, params_, id_;
    std::string fork_of_;
    std::string card_;                 // the text of the session's card of the call in flight
    std::string card_approval_;        // its approval id
    std::string decided_approval_;     // approved or denied, waiting for the queue's drain
    std::string card_script_;          // the parent's script of a fork, for its card
    std::string sent_id_;              // the invocation id of the call in flight
    unsigned long sent_count_ = 0;
    std::string problem_;
    std::string last_run_;
    std::future<caliper_builder::CommandResult> pending_;
    std::string pending_name_;
    bool id_edited_ = false;   // the person typed an id: no new suggestion
    // The ImGui fields' edit buffers.
    std::string script_edit_, args_edit_, params_edit_, id_edit_;
};

}  // namespace caliper_host
