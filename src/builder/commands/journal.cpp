#include <caliper_builder/commands/journal.h>
#include <caliper_builder/commands/schema.h>
#include <caliper_builder/commands/sha256.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <system_error>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;
namespace fs = std::filesystem;

std::string dump(const json& v) { return v.dump(-1, ' ', false, json::error_handler_t::replace); }

fs::path to_path(const std::string& utf8) {
#if defined(__cpp_char8_t)
    return fs::path(std::u8string(utf8.begin(), utf8.end()));
#else
    return fs::u8path(utf8);
#endif
}

std::string from_path(const fs::path& p) {
#if defined(__cpp_char8_t)
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
#else
    return p.u8string();
#endif
}

std::FILE* open_file(const fs::path& p, const char* mode) {
#ifdef _WIN32
    std::wstring wmode(mode, mode + std::strlen(mode));
    return _wfopen(p.c_str(), wmode.c_str());
#else
    return std::fopen(p.c_str(), mode);
#endif
}

bool secret_key(const std::string& key) {
    std::string k = key;
    std::transform(k.begin(), k.end(), k.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const char* exact : {"token", "secret", "password", "api_key", "apikey", "authorization"})
        if (k == exact) return true;
    for (const char* suffix : {"_token", "_secret", "_password", "_api_key"}) {
        const std::size_t n = std::strlen(suffix);
        if (k.size() > n && k.compare(k.size() - n, n, suffix) == 0) return true;
    }
    return false;
}

json marker(const std::string& sha, std::size_t bytes) {
    return json{{"$redacted", {{"sha256", sha}, {"bytes", bytes}}}};
}

// Redacts `v` bottom-up into `out` and returns the size of `out` as JSON text.
std::size_t redact(const json& v, json& out, std::size_t limit) {
    if (v.is_string()) {
        const std::string& s = v.get_ref<const std::string&>();
        if (s.size() > limit) {
            out = marker(command_sha256_hex(s), s.size());
            return dump(out).size();
        }
        out = v;
        return dump(out).size();
    }
    if (v.is_object()) {
        out = json::object();
        std::size_t size = 2;
        bool first = true;
        for (const auto& item : v.items()) {
            json child;
            std::size_t child_size;
            if (secret_key(item.key())) {
                child = json{{"$redacted", "secret"}};
                child_size = dump(child).size();
            } else {
                child_size = redact(item.value(), child, limit);
            }
            size += (first ? 0 : 1) + dump(json(item.key())).size() + 1 + child_size;
            first = false;
            out[item.key()] = std::move(child);
        }
        if (size > limit) {
            const std::string text = dump(v);
            out = marker(command_sha256_hex(text), text.size());
            return dump(out).size();
        }
        return size;
    }
    if (v.is_array()) {
        out = json::array();
        std::size_t size = 2;
        bool first = true;
        for (const auto& item : v) {
            json child;
            size += (first ? 0 : 1) + redact(item, child, limit);
            first = false;
            out.push_back(std::move(child));
        }
        if (size > limit) {
            const std::string text = dump(v);
            out = marker(command_sha256_hex(text), text.size());
            return dump(out).size();
        }
        return size;
    }
    out = v;
    return dump(out).size();
}

// The seq of the last valid event line in the file's last 2 MiB, or 0.
std::uint64_t last_seq_in(const fs::path& p) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return 0;
    const auto size = fs::file_size(p, ec);
    if (ec || size == 0) return 0;
    std::FILE* f = open_file(p, "rb");
    if (!f) return 0;
    const std::uint64_t window = std::min<std::uint64_t>(size, std::uint64_t(2) << 20);
    std::string tail(static_cast<std::size_t>(window), '\0');
    bool read_ok = std::fseek(f, static_cast<long>(size - window), SEEK_SET) == 0 &&
                   std::fread(&tail[0], 1, tail.size(), f) == tail.size();
    std::fclose(f);
    if (!read_ok) return 0;
    std::size_t end = tail.size();
    while (end > 0) {
        const std::size_t start = tail.rfind('\n', end - 1);
        const std::size_t from = start == std::string::npos ? 0 : start + 1;
        if (end > from) {
            const json j = json::parse(tail.begin() + static_cast<std::ptrdiff_t>(from),
                                       tail.begin() + static_cast<std::ptrdiff_t>(end), nullptr, false);
            if (j.is_object() && j.contains("seq") && j["seq"].is_number_unsigned()) return j["seq"].get<std::uint64_t>();
        }
        if (start == std::string::npos) break;
        end = start;
    }
    return 0;
}

bool ends_with_newline(const fs::path& p) {
    std::error_code ec;
    const auto size = fs::file_size(p, ec);
    if (ec || size == 0) return true;
    std::FILE* f = open_file(p, "rb");
    if (!f) return true;
    int c = EOF;
    if (std::fseek(f, -1, SEEK_END) == 0) c = std::fgetc(f);
    std::fclose(f);
    return c == '\n';
}

}  // namespace

std::string command_utc_timestamp(std::chrono::system_clock::time_point time) {
    using namespace std::chrono;
    const auto ms = duration_cast<milliseconds>(time.time_since_epoch()).count();
    auto secs = static_cast<std::time_t>(ms / 1000);
    int millis = static_cast<int>(ms % 1000);
    if (millis < 0) {
        millis += 1000;
        secs -= 1;
    }
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &secs);
#else
    gmtime_r(&secs, &tm);
#endif
    char buf[40];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, millis);
    return buf;
}

std::string journal_rotated_path(const std::string& path, int n) {
    const fs::path p = to_path(path);
    fs::path name = p.stem();
    name += "." + std::to_string(n);
    name += p.extension();
    return from_path(p.parent_path() / name);
}

json redact_journal_args(const json& args, std::size_t limit) {
    if (measure_command_value(args)) return json{{"$redacted", "malformed"}};
    json out;
    redact(args, out, limit);
    return out;
}

bool journal_args_redacted(const json& args) {
    std::vector<const json*> stack{&args};
    std::size_t visited = 0;
    while (!stack.empty()) {
        const json* v = stack.back();
        stack.pop_back();
        if (++visited > CommandSchemaLimits::max_value_nodes) return true;  // cannot tell: treat as redacted
        if (v->is_object() && v->contains("$redacted")) return true;
        if (v->is_structured())
            for (const auto& child : *v) stack.push_back(&child);
    }
    return false;
}

json journal_event_to_json(const JournalEvent& e) {
    json out = {{"seq", e.seq},
                {"ts", e.ts},
                {"origin", command_origin_name(e.origin)},
                {"name", e.name},
                {"args", e.args},
                {"ok", e.ok},
                {"group", e.group.empty() ? json(nullptr) : json(e.group)},
                {"event", e.event}};
    if (!e.ok) out["error"] = e.error;
    if (e.event != "do") out["seqs"] = e.seqs;
    if (!e.allowed_by.empty()) out["allowed_by"] = e.allowed_by;
    return out;
}

JournalEventParse journal_event_from_json(const json& v) {
    JournalEventParse out;
    auto fail = [&](std::string message, std::string where) {
        out.error = Error("journal.bad_event", std::move(message), std::move(where));
        return out;
    };
    if (!v.is_object()) return fail("An event is a JSON object.", "");
    for (const auto& item : v.items()) {
        const std::string& k = item.key();
        if (k != "seq" && k != "ts" && k != "origin" && k != "name" && k != "args" && k != "ok" && k != "group" &&
            k != "event" && k != "error" && k != "seqs" && k != "allowed_by")
            return fail("An event has no field of this name.", k);
    }
    for (const char* k : {"seq", "ts", "origin", "name", "args", "ok", "group", "event"})
        if (!v.contains(k)) return fail("An event has every field.", k);
    JournalEvent& e = out.event;
    if (!v["seq"].is_number_unsigned() || v["seq"].get<std::uint64_t>() == 0)
        return fail("seq is a positive integer.", "seq");
    e.seq = v["seq"].get<std::uint64_t>();
    if (!v["ts"].is_string()) return fail("ts is a string.", "ts");
    e.ts = v["ts"].get<std::string>();
    if (!v["origin"].is_string()) return fail("origin is a string.", "origin");
    const auto origin = command_origin_from_name(v["origin"].get_ref<const std::string&>());
    if (!origin) return fail("origin is ui, builder, agent, mcp or api.", "origin");
    e.origin = *origin;
    if (!v["name"].is_string()) return fail("name is a string.", "name");
    e.name = v["name"].get<std::string>();
    if (!v["args"].is_object()) return fail("args is an object.", "args");
    if (measure_command_value(v["args"])) return fail("args is nested too deeply or too large.", "args");
    e.args = v["args"];
    if (!v["ok"].is_boolean()) return fail("ok is true or false.", "ok");
    e.ok = v["ok"].get<bool>();
    if (!v["group"].is_null()) {
        if (!v["group"].is_string()) return fail("group is a string or null.", "group");
        e.group = v["group"].get<std::string>();
    }
    if (!v["event"].is_string()) return fail("event is do, undo or redo.", "event");
    e.event = v["event"].get<std::string>();
    if (e.event != "do" && e.event != "undo" && e.event != "redo") return fail("event is do, undo or redo.", "event");
    if (v.contains("error")) {
        if (!v["error"].is_string()) return fail("error is a code.", "error");
        e.error = v["error"].get<std::string>();
    }
    if (v.contains("allowed_by")) {
        if (!v["allowed_by"].is_string() || e.event != "do")
            return fail("allowed_by is a string on a do event.", "allowed_by");
        e.allowed_by = v["allowed_by"].get<std::string>();
    }
    if (v.contains("seqs")) {
        if (e.event == "do") return fail("Only an undo or redo event lists seqs.", "seqs");
        if (!v["seqs"].is_array()) return fail("seqs is a list of seq numbers.", "seqs");
        for (const auto& s : v["seqs"]) {
            if (!s.is_number_unsigned()) return fail("seqs is a list of seq numbers.", "seqs");
            e.seqs.push_back(s.get<std::uint64_t>());
        }
    }
    return out;
}

CommandJournal::CommandJournal(std::string path, JournalOptions options)
    : path_(std::move(path)), options_(options) {}

CommandJournal::~CommandJournal() { close(); }

CommandJournal::Open CommandJournal::open(const std::string& path, JournalOptions options) {
    Open out;
    if (path.empty()) {
        out.error = Error("journal.open_failed", "The journal needs a path.", "");
        return out;
    }
    if (options.keep < 0) options.keep = 0;
    if (options.max_bytes < 1024) options.max_bytes = 1024;
    if (options.redact_over < 64) options.redact_over = 64;
    const fs::path p = to_path(path);
    std::error_code ec;
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    if (ec) {
        out.error = Error("journal.open_failed", "The journal's folder could not be made: " + ec.message(),
                          from_path(p.filename()));
        return out;
    }
    if (fs::exists(p, ec) && !fs::is_regular_file(p, ec)) {
        out.error = Error("journal.open_failed", "The journal's path is not a file.", from_path(p.filename()));
        return out;
    }
    std::unique_ptr<CommandJournal> j(new CommandJournal(path, options));
    j->last_seq_ = last_seq_in(p);
    if (j->last_seq_ == 0 && options.keep > 0) j->last_seq_ = last_seq_in(to_path(journal_rotated_path(path, 1)));
    const bool end_line = !ends_with_newline(p);
    j->file_ = open_file(p, "ab");
    if (!j->file_) {
        out.error = Error("journal.open_failed", "The journal could not be opened for writing.", from_path(p.filename()));
        return out;
    }
    if (end_line) {
        std::fputc('\n', j->file_);
        std::fflush(j->file_);
    }
    j->size_ = fs::file_size(p, ec);
    if (ec) j->size_ = 0;
    out.journal = std::move(j);
    return out;
}

void CommandJournal::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

std::uint64_t CommandJournal::last_seq() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_seq_;
}

Error CommandJournal::rotate() {
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
    const fs::path p = to_path(path_);
    std::error_code ec;
    Error result;
    if (options_.keep > 0) {
        fs::remove(to_path(journal_rotated_path(path_, options_.keep)), ec);
        for (int k = options_.keep - 1; k >= 1; --k) {
            const fs::path from = to_path(journal_rotated_path(path_, k));
            if (fs::exists(from, ec)) fs::rename(from, to_path(journal_rotated_path(path_, k + 1)), ec);
        }
        fs::rename(p, to_path(journal_rotated_path(path_, 1)), ec);
    } else {
        fs::remove(p, ec);
    }
    if (ec) result = Error("journal.rotate_failed", "The journal could not be rotated: " + ec.message(), "");
    file_ = open_file(p, "ab");
    size_ = fs::file_size(p, ec);
    if (ec) size_ = 0;
    if (!file_) result = Error("journal.write_failed", "The journal could not be reopened after rotation.", "");
    return result;
}

CommandJournal::Append CommandJournal::append(JournalEvent event) {
    Append out;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_) {
        out.error = Error("journal.closed", "The journal is closed.", "");
        return out;
    }
    event.seq = last_seq_ + 1;
    if (event.ts.empty()) event.ts = command_utc_timestamp(std::chrono::system_clock::now());
    event.args = redact_journal_args(event.args, options_.redact_over);
    const std::string line = dump(journal_event_to_json(event)) + "\n";
    if (size_ > 0 && size_ + line.size() > options_.max_bytes) {
        Error err = rotate();
        if (!file_) {
            out.error = std::move(err);
            return out;
        }
    }
    const std::size_t written = std::fwrite(line.data(), 1, line.size(), file_);
    const bool flushed = std::fflush(file_) == 0;
    if (written != line.size() || !flushed) {
        out.error = Error("journal.write_failed", "The journal line could not be written.", "");
        return out;
    }
    size_ += line.size();
    last_seq_ = event.seq;
    out.seq = event.seq;
    if (options_.on_append) {
        try {
            options_.on_append(event);
        } catch (...) {
            // An observer that throws loses its event; the line is written.
        }
    }
    return out;
}

JournalRead read_journal(const std::string& path, JournalReadOptions options) {
    JournalRead out;
    std::vector<fs::path> files;
    std::error_code ec;
    if (options.include_rotated) {
        for (int n = 1; n <= 1000; ++n) {
            const fs::path r = to_path(journal_rotated_path(path, n));
            if (!fs::is_regular_file(r, ec)) break;
            files.push_back(r);
        }
        std::reverse(files.begin(), files.end());
    }
    const fs::path current = to_path(path);
    if (fs::is_regular_file(current, ec)) files.push_back(current);
    if (files.empty()) {
        out.error = Error("journal.not_found", "No journal file was found.", from_path(current.filename()));
        return out;
    }

    std::uint64_t last = 0;
    for (const fs::path& file : files) {
        const std::string name = from_path(file.filename());
        std::FILE* f = open_file(file, "rb");
        if (!f) {
            out.problems.push_back(name + ": the file could not be read.");
            continue;
        }
        std::size_t lineno = 1;
        std::string line;
        bool skipping = false;
        auto handle = [&](bool complete) {
            if (skipping) {
                out.problems.push_back(name + ":" + std::to_string(lineno) + ": the line is longer than " +
                                       std::to_string(options.max_line) + " bytes.");
                return;
            }
            if (line.empty()) return;
            const json j = json::parse(line, nullptr, false);
            if (j.is_discarded()) {
                out.problems.push_back(name + ":" + std::to_string(lineno) +
                                       (complete ? ": the line is not JSON." : ": the last line was cut short."));
                return;
            }
            const auto parsed = journal_event_from_json(j);
            if (parsed.error) {
                out.problems.push_back(name + ":" + std::to_string(lineno) + ": " + parsed.error.message +
                                       (parsed.error.where.empty() ? "" : " (" + parsed.error.where + ")"));
                return;
            }
            const std::uint64_t seq = parsed.event.seq;
            if (seq <= last) {
                out.problems.push_back(name + ":" + std::to_string(lineno) + ": seq " + std::to_string(seq) +
                                       " does not increase; the event is skipped.");
                return;
            }
            if (last != 0 && seq != last + 1)
                out.problems.push_back(name + ":" + std::to_string(lineno) + ": seq jumps from " +
                                       std::to_string(last) + " to " + std::to_string(seq) + ".");
            last = seq;
            out.events.push_back(parsed.event);
        };
        char buf[1 << 16];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
            for (std::size_t i = 0; i < n; ++i) {
                const char c = buf[i];
                if (c == '\n') {
                    handle(true);
                    line.clear();
                    skipping = false;
                    ++lineno;
                } else if (!skipping) {
                    line.push_back(c);
                    if (line.size() > options.max_line) {
                        skipping = true;
                        line.clear();
                        line.shrink_to_fit();
                    }
                }
            }
        }
        if (skipping || !line.empty()) handle(false);
        std::fclose(f);
    }
    return out;
}

JournalReplay replay_journal(const std::vector<JournalEvent>& events,
                             const std::function<Error(const JournalEvent&)>& apply) {
    JournalReplay out;
    if (events.empty()) return out;
    if (events.front().seq != 1) {
        out.error = Error("journal.incomplete",
                          "The journal starts at seq " + std::to_string(events.front().seq) +
                              ", so the events before it are missing and the state cannot be rebuilt.",
                          "seq " + std::to_string(events.front().seq));
        return out;
    }
    std::map<std::uint64_t, bool> active;
    for (const auto& e : events) {
        if (!e.ok && e.error == "history.inconsistent") {
            out.error = Error("journal.unreplayable",
                              "The history was left inconsistent here, so the state after it is unknown.",
                              "seq " + std::to_string(e.seq));
            return out;
        }
        if (!e.ok) continue;
        if (e.event == "do") {
            active[e.seq] = true;
        } else {
            const bool on = e.event == "redo";
            for (const auto s : e.seqs) {
                const auto it = active.find(s);
                if (it != active.end()) it->second = on;
            }
        }
    }
    for (const auto& e : events) {
        if (e.event != "do" || !e.ok) continue;
        const auto it = active.find(e.seq);
        if (it == active.end() || !it->second) continue;
        if (journal_args_redacted(e.args)) {
            out.error = Error("journal.redacted",
                              "The arguments of " + e.name + " were redacted, so it cannot be applied again.",
                              "seq " + std::to_string(e.seq));
            return out;
        }
        Error err;
        try {
            err = apply(e);
        } catch (const std::exception& ex) {
            err = Error("journal.apply_failed", ex.what());
        } catch (...) {
            err = Error("journal.apply_failed", "The apply function threw.");
        }
        if (err) {
            if (err.where.empty()) err.where = "seq " + std::to_string(e.seq);
            out.error = std::move(err);
            return out;
        }
        out.applied.push_back(e.seq);
    }
    return out;
}

}  // namespace caliper_builder
