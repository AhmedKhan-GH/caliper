#pragma once
// The command journal (docs/plans/command-layer.md, The journal): append-only
// JSON Lines at <workspace>/journal/commands.jsonl, one event per line:
//   {"seq", "ts", "origin", "name", "args", "ok", "group", "event"}
// with "error" (the code) when ok is false and "seqs" (the seq of every do
// event of the step) on undo and redo events.
//
// CommandJournal::append() gives each event the next seq, which continues
// after the journal is reopened and across rotation; redacts the arguments
// (a string over 64 KiB, and an array or object whose JSON text stays over
// 64 KiB after its members were redacted, become {"$redacted": {"sha256",
// "bytes"}}; the value of a key named like a secret becomes {"$redacted":
// "secret"}); writes one line and flushes it before it returns. When the file
// would pass max_bytes it is renamed commands.1.jsonl, older files shift up to
// commands.<keep>.jsonl and the oldest is deleted. Opening a file whose last
// line was cut short first ends that line, so the next line is whole.
//
// read_journal() reads the rotated files, oldest first, then the current one;
// it refuses lines over 1 MiB, skips lines that are not valid events and
// reports each with its file and line, and reports a seq that does not
// increase or that skips. replay_journal() works out which do events are in
// effect after every undo and redo, and applies those in seq order.
//
// CommandJournal is thread-safe. Paths are UTF-8. No function here throws.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace caliper_builder {

struct JournalOptions {
    std::uint64_t max_bytes = std::uint64_t(16) << 20;  // rotate before the file passes this
    int keep = 4;                                       // rotated files kept
    std::size_t redact_over = std::size_t(64) << 10;    // 64 KiB
    bool include_read_only = false;                     // the registry also writes read-only commands
};

struct JournalEvent {
    std::uint64_t seq = 0;
    std::string ts;  // UTC ISO 8601 with milliseconds
    CommandOrigin origin = CommandOrigin::ui;
    std::string name;
    nlohmann::json args = nlohmann::json::object();
    bool ok = true;
    std::string group;          // empty: null
    std::string event = "do";   // "do", "undo" or "redo"
    std::string error;          // the error code when ok is false
    std::vector<std::uint64_t> seqs;  // undo and redo events
};

nlohmann::json journal_event_to_json(const JournalEvent& event);

struct JournalEventParse {
    JournalEvent event;
    Error error;
};
JournalEventParse journal_event_from_json(const nlohmann::json& value);

// The redaction of section "What is redacted". A value nested too deeply to
// measure becomes {"$redacted": "malformed"}.
nlohmann::json redact_journal_args(const nlohmann::json& args, std::size_t limit = std::size_t(64) << 10);

// True when `args` holds a redacted value anywhere.
bool journal_args_redacted(const nlohmann::json& args);

// "2026-11-15T12:00:00.125Z".
std::string command_utc_timestamp(std::chrono::system_clock::time_point time);

// <dir>/<stem>.<n><ext>: "journal/commands.2.jsonl" for n = 2.
std::string journal_rotated_path(const std::string& path, int n);

class CommandJournal {
public:
    struct Open {
        std::unique_ptr<CommandJournal> journal;
        Error error;
    };
    // Opens (and creates, with its folder) the journal at `path`.
    static Open open(const std::string& path, JournalOptions options = {});

    ~CommandJournal();
    CommandJournal(const CommandJournal&) = delete;
    CommandJournal& operator=(const CommandJournal&) = delete;

    struct Append {
        std::uint64_t seq = 0;
        Error error;
    };
    // Sets the event's seq (and its ts when empty), redacts its arguments,
    // writes the line and flushes it. On failure no seq is used up.
    Append append(JournalEvent event);

    std::uint64_t last_seq() const;
    const JournalOptions& options() const { return options_; }
    const std::string& path() const { return path_; }
    void close();

private:
    CommandJournal(std::string path, JournalOptions options);
    Error rotate();

    mutable std::mutex mutex_;
    std::string path_;
    JournalOptions options_;
    std::FILE* file_ = nullptr;
    std::uint64_t size_ = 0;
    std::uint64_t last_seq_ = 0;
};

struct JournalReadOptions {
    std::size_t max_line = std::size_t(1) << 20;
    bool include_rotated = true;
};

struct JournalRead {
    std::vector<JournalEvent> events;   // in seq order
    std::vector<std::string> problems;  // "commands.jsonl:12: ..." for each line skipped or out of order
    Error error;                        // only when no file could be read
};

JournalRead read_journal(const std::string& path, JournalReadOptions options = {});

struct JournalReplay {
    std::vector<std::uint64_t> applied;  // the seqs applied, in order
    Error error;
};

// Applies the do events in effect, in seq order. Stops with journal.incomplete
// when the first seq is not 1, journal.unreplayable at an event that left the
// history inconsistent, journal.redacted at an event in effect whose arguments
// were redacted, or the apply function's own error, with `where` naming the seq.
JournalReplay replay_journal(const std::vector<JournalEvent>& events,
                             const std::function<Error(const JournalEvent&)>& apply);

}  // namespace caliper_builder
