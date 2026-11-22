// live_programs.cpp: ProgramSync and the ProgramLink of an Attachment (live_programs.h).
#include "live/live_programs.h"

#include "embed/embed_core.h"
#include "live/live_attach.h"

#include <algorithm>
#include <utility>

namespace caliper_live {

namespace wire = caliper_net::wire;
using caliper_net::Json;

// ---- AttachmentProgramLink ------------------------------------------------------------------

NetSource* AttachmentProgramLink::source() const {
    CaliperCore* core = attachment_.core();
    return core ? dynamic_cast<NetSource*>(caliper_live_stream(core)) : nullptr;
}

bool AttachmentProgramLink::ready() {
    NetSource* s = source();
    return s && s->programs_ready(nullptr);
}

uint64_t AttachmentProgramLink::epoch() {
    NetSource* s = source();
    return s ? s->program_epoch() : 0;
}

uint32_t AttachmentProgramLink::send(wire::Program msg) {
    NetSource* s = source();
    return s ? s->send_program(std::move(msg)) : 0;
}

std::vector<ProgramAnswer> AttachmentProgramLink::take() {
    NetSource* s = source();
    return s ? s->take_program_answers() : std::vector<ProgramAnswer>{};
}

// ---- ProgramSync ------------------------------------------------------------------------------

namespace {

std::string str_of(const Json& o, const char* key) {
    const Json* j = o.is_object() ? o.find(key) : nullptr;
    return j && j->is_string() ? j->as_string() : std::string();
}

uint32_t u32_of(const Json& o, const char* key) {
    uint64_t v = 0;
    const Json* j = o.is_object() ? o.find(key) : nullptr;
    return j && j->get_u64(&v) && v <= 0xffffffffull ? static_cast<uint32_t>(v) : 0;
}

}  // namespace

ProgramSync::Entry* ProgramSync::find(const std::string& id) {
    for (Entry& e : entries_)
        if (e.rec.want.id == id) return &e;
    return nullptr;
}

void ProgramSync::want(std::vector<WantedProgram> programs) {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<Entry> next;
    next.reserve(programs.size());
    for (WantedProgram& w : programs) {
        Entry e;
        if (Entry* old = find(w.id)) {
            e = *old;
            if (old->rec.want.sha256 != w.sha256 || old->rec.want.declaration != w.declaration ||
                old->rec.want.scene != w.scene) {
                e.sent_sha.clear();   // changed: send it again
                e.sends = 0;
                e.pending = 0;
            }
        }
        e.rec.want = std::move(w);
        next.push_back(std::move(e));
    }
    for (const Entry& old : entries_) {
        const bool kept = std::any_of(next.begin(), next.end(),
                                      [&](const Entry& e) { return e.rec.want.id == old.rec.want.id; });
        if (!kept && (old.rec.host == "script" || !old.sent_sha.empty()) &&
            std::find(removing_.begin(), removing_.end(), old.rec.want.id) == removing_.end())
            removing_.push_back(old.rec.want.id);
    }
    entries_ = std::move(next);
    for (Entry& e : entries_) {
        // A program the app refuses itself never goes out; a removal of the
        // same id that was queued before is dropped once it is wanted again.
        removing_.erase(std::remove(removing_.begin(), removing_.end(), e.rec.want.id), removing_.end());
        if (!e.rec.want.refused_code.empty()) {
            if (e.rec.host == "script" || !e.sent_sha.empty()) removing_.push_back(e.rec.want.id);
            e.sent_sha.clear();
            e.pending = 0;
            e.rec.host = "none";
            e.rec.item = Json();
            e.rec.state = "refused";
            e.rec.error_code = e.rec.want.refused_code;
            e.rec.error_message = e.rec.want.refused_message;
            e.rec.line = e.rec.column = 0;
            note_change(e);
        }
    }
}

void ProgramSync::note_change(Entry& e) {
    const std::string sha = str_of(e.rec.item, "sha256");
    const std::string key = e.rec.state + "/" + e.rec.error_code + "/" + sha;
    if (key == e.seen) return;
    const std::string before = e.seen_state, before_sha = e.seen_sha;
    e.seen = key;
    e.seen_state = e.rec.state;
    e.seen_sha = sha;
    const std::string& st = e.rec.state;
    const bool ok = st == "compiled" || st == "running";
    const bool was_ok = before == "compiled" || before == "running";
    if (st == "refused" || st == "stopped") changes_.push_back({e.rec.want.id, st});
    else if (ok && (!was_ok || sha != before_sha)) changes_.push_back({e.rec.want.id, "compiled"});
}

void ProgramSync::apply_item(const Json& item) {
    const std::string id = str_of(item, "id");
    Entry* e = find(id);
    if (!e) {
        if (std::find(removing_.begin(), removing_.end(), id) == removing_.end()) removing_.push_back(id);
        return;
    }
    if (!e->rec.want.refused_code.empty()) return;
    e->rec.item = item;
    e->rec.host = "script";
    // The script runs this source already (after a reconnect, say): it is
    // not sent again, so a stopped program stays stopped.
    if (str_of(item, "sha256") == e->rec.want.sha256) e->sent_sha = e->rec.want.sha256;
    e->rec.state = str_of(item, "state");
    if (e->rec.state.empty()) e->rec.state = "waiting";
    const Json* err = item.find("error");
    if (err && err->is_object()) {
        e->rec.error_code = str_of(*err, "code");
        e->rec.error_message = str_of(*err, "message");
        e->rec.line = u32_of(*err, "line");
        e->rec.column = u32_of(*err, "column");
    } else {
        e->rec.error_code.clear();
        e->rec.error_message.clear();
        e->rec.line = e->rec.column = 0;
    }
    note_change(*e);
}

void ProgramSync::poll(ProgramLink& link) {
    const bool ready = link.ready();
    std::vector<ProgramAnswer> answers = link.take();
    std::lock_guard<std::mutex> g(mu_);
    if (!ready) {
        // No script takes programs: each wanted program waits, and is sent
        // again once one does.
        if (was_ready_) {
            for (Entry& e : entries_) {
                e.sent_sha.clear();
                e.sends = 0;
                e.pending = 0;
                if (!e.rec.want.refused_code.empty()) continue;
                e.rec.host = "none";
                e.rec.state = "waiting";
                e.rec.item = Json();
                e.seen.clear();
                e.seen_state.clear();
                e.seen_sha.clear();
            }
            removing_.clear();
        }
        was_ready_ = false;
        return;
    }
    was_ready_ = true;
    const uint64_t epoch = link.epoch();
    if (epoch != epoch_) {
        epoch_ = epoch;
        for (Entry& e : entries_) {
            e.sent_sha.clear();
            e.sends = 0;
            e.pending = 0;
        }
    }

    for (ProgramAnswer& a : answers) {
        if (!a.error_code.empty()) {
            // A refused PROGRAM: the program it set shows the refusal.
            for (Entry& e : entries_) {
                if (!a.request_id || e.pending != *a.request_id) continue;
                e.pending = 0;
                e.rec.state = "refused";
                e.rec.error_code = "program.unavailable";
                e.rec.error_message = "The stream refused the program (" + a.error_code + ")" +
                                      (a.error_message.empty() ? std::string(".") : ": " + a.error_message);
                e.rec.line = e.rec.column = 0;
                note_change(e);
            }
            continue;
        }
        if (!a.programs.is_array()) continue;
        std::vector<std::string> reported;
        for (const Json& item : a.programs.items()) {
            reported.push_back(str_of(item, "id"));
            apply_item(item);
        }
        for (Entry& e : entries_) {
            if (a.request_id && e.pending == *a.request_id) e.pending = 0;
            // A program the script no longer holds (it removed its programs
            // while no viewer watched) goes out again, a few times at most.
            const bool held = std::find(reported.begin(), reported.end(), e.rec.want.id) != reported.end();
            if (!held && e.pending == 0 && !e.sent_sha.empty() && e.rec.host == "script") {
                e.sent_sha.clear();
                e.rec.host = "none";
                e.rec.state = "waiting";
                e.rec.item = Json();
            }
        }
    }

    for (Entry& e : entries_) {
        if (!e.rec.want.refused_code.empty() || e.sent_sha == e.rec.want.sha256) continue;
        if (e.sends >= kMaxSends) continue;
        wire::Program m;
        m.op = CALIPER_NET_PROGRAM_SET;
        m.id = e.rec.want.id;
        m.declaration = e.rec.want.declaration;
        m.wgsl = e.rec.want.wgsl;
        m.sha256 = e.rec.want.sha256;
        if (!e.rec.want.scene.empty()) m.scene = e.rec.want.scene;
        const uint32_t rid = link.send(std::move(m));
        if (rid == 0) break;   // the queue is full: the next frame tries again
        e.pending = rid;
        e.sent_sha = e.rec.want.sha256;
        ++e.sends;
    }
    while (!removing_.empty()) {
        wire::Program m;
        m.op = CALIPER_NET_PROGRAM_REMOVE;
        m.id = removing_.front();
        if (link.send(std::move(m)) == 0) break;
        removing_.erase(removing_.begin());
    }
}

std::vector<ProgramRecord> ProgramSync::records() const {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<ProgramRecord> out;
    out.reserve(entries_.size());
    for (const Entry& e : entries_) out.push_back(e.rec);
    return out;
}

std::vector<ProgramChange> ProgramSync::take_changes() {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<ProgramChange> out;
    out.swap(changes_);
    return out;
}

}  // namespace caliper_live
