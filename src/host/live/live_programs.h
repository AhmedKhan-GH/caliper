// live_programs.h: the app's side of GPU programs (design/gpu-programs.md,
// Wire Path, Attachment and Program Status); host-internal.
//
// ProgramSync keeps the programs of the attached script equal to the programs
// that the app wants, by id and hash. Once per frame it sends PROGRAM set for
// a new or changed program and remove for a removed one, sends each program
// again after an attachment (a new program_epoch() of the stream), removes a
// program that the script reports and the app does not want, and turns the
// script's PROGRAM_STATE into one record for each wanted program and into the
// changes that the `program` event reports (compiled, refused, stopped). It
// holds no builder type, so libcaliper keeps no dependency on the builder, and
// the tests drive it through a fake ProgramLink.
//
// AppProgramHost (only with CALIPER_WITH_BUILDER) is the builder session's
// ProgramHost over a ProgramSync: reconcile() makes the wanted programs from
// the open spec, status() gives the status items of program.status, and pump()
// runs the sync once per frame and pushes the program events.
//
// An instance of a piece (design/programmable-graphics.md, Program
// Instances) goes out as the instance form of PROGRAM set, with its sections
// resolved, only to a script that takes pieces (capability "resources"); the
// app's host also moves each RESOURCE version the stream received into the
// session's step history.
//
// Threads: want(), poll() and take_changes() run on the frame thread;
// records() may run on any thread.
#pragma once
#include "live/live_net_source.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace caliper_live {

class Attachment;

// One program that the app wants the script to run.
struct WantedProgram {
    std::string id;
    std::string kind;                  // "geometry", "pixels", "tensor", or "operator" for a built-in operator
    std::string panel;                 // geometry: the target scene panel's id; "" otherwise
    std::string scene;                 // geometry: that panel's scene name; "" otherwise
    std::string wgsl;
    std::string sha256;                // of wgsl, lowercase hex
    caliper_net::Json declaration;     // the declaration without wgsl and without params
    // The values of the uniform `params` by name, null when the program has
    // none. A change of the values alone goes out as PROGRAM op params, and a
    // set carries the values of its time in its declaration.
    caliper_net::Json params;
    // Set when the app refuses the program before it sends it: the record
    // shows it as refused and nothing goes out.
    std::string refused_code, refused_message;
    // An instance of a piece, when `piece` is an object: the pin {name,
    // version, hash}, the manifest, the bindings, the segments of `common`,
    // and the sections with the imports resolved. `sha256` then holds the 64
    // hex digits of the piece's hash, and `declaration`, `wgsl` and `scene`
    // stay empty.
    caliper_net::Json piece, manifest, bind, segments;
    std::string common, derive, draw;
    bool instance() const { return piece.is_object(); }
};

// Where a ProgramSync sends and reads: the attached stream in the app, a fake
// in the tests.
class ProgramLink {
public:
    virtual ~ProgramLink() = default;
    // Whether a PROGRAM may go out now.
    virtual bool ready() = 0;
    // Grows with each attachment after which the script may hold none of the
    // programs sent before.
    virtual uint64_t epoch() = 0;
    // Sends `msg` (its request_id is set by the link); the request id, or 0
    // when it did not go out.
    virtual uint32_t send(caliper_net::wire::Program msg) = 0;
    // The script's answers since the last call, oldest first.
    virtual std::vector<ProgramAnswer> take() = 0;
    // Whether the script takes instances of pieces (capability "resources").
    virtual bool takes_pieces() { return false; }
    // The RESOURCE versions the stream received since the last call.
    virtual std::vector<ReceivedResource> take_resources() { return {}; }
};

// The ProgramLink of an Attachment's network stream; not ready while the
// attachment holds no network stream.
class AttachmentProgramLink final : public ProgramLink {
public:
    explicit AttachmentProgramLink(const Attachment& attachment) : attachment_(attachment) {}
    bool ready() override;
    uint64_t epoch() override;
    uint32_t send(caliper_net::wire::Program msg) override;
    std::vector<ProgramAnswer> take() override;
    bool takes_pieces() override;
    std::vector<ReceivedResource> take_resources() override;

private:
    NetSource* source() const;
    const Attachment& attachment_;
};

// One wanted program as the app shows it.
struct ProgramRecord {
    WantedProgram want;
    std::string host = "none";         // "script" once the script reported the program
    std::string state = "waiting";     // the status item's state, or waiting
    caliper_net::Json item;            // the script's latest status item; null before it
    std::string error_code;            // the last error: the item's, a refusal's or the app's
    std::string error_message;
    uint32_t line = 0, column = 0;
};

// A change of a program that its `program` event reports.
struct ProgramChange {
    std::string id;
    std::string change;                // "compiled", "refused" or "stopped"
};

class ProgramSync {
public:
    // The sends of one program in one epoch with the same hash, at most: a
    // script that keeps dropping a program does not make the app resend it
    // without end.
    static constexpr int kMaxSends = 3;

    // The programs to run, in the spec's order. Frame thread.
    void want(std::vector<WantedProgram> programs);
    // Sends what differs, reads the answers and updates the records. Frame
    // thread, once per frame; it never waits.
    void poll(ProgramLink& link);
    // One record for each wanted program, in their order. Any thread.
    std::vector<ProgramRecord> records() const;
    // The changes since the last call, oldest first. Frame thread.
    std::vector<ProgramChange> take_changes();

private:
    struct Entry {
        ProgramRecord rec;
        std::string sent_sha;          // the hash sent in this epoch; "" when none
        int sends = 0;                 // sets of sent_sha in this epoch
        uint32_t pending = 0;          // the request id of the last set
        std::string seen;              // "<state>/<error code>/<sha256>" of the last change
        std::string seen_state, seen_sha;
        bool params_dirty = false;     // the values changed after the set of this epoch
    };
    void apply_item(const caliper_net::Json& item);
    void note_change(Entry& e);
    Entry* find(const std::string& id);

    mutable std::mutex mu_;            // guards entries_ for records()
    std::vector<Entry> entries_;
    std::vector<std::string> removing_;   // ids to remove from the script
    std::vector<ProgramChange> changes_;
    uint64_t epoch_ = 0;
    bool was_ready_ = false;
};

// ---- ProgramSync, inline so that a test with the builder and without libcaliper links it ------------------------------------------------------------------------------

namespace program_sync_detail {

inline std::string str_of(const caliper_net::Json& o, const char* key) {
    const caliper_net::Json* j = o.is_object() ? o.find(key) : nullptr;
    return j && j->is_string() ? j->as_string() : std::string();
}

inline uint32_t u32_of(const caliper_net::Json& o, const char* key) {
    uint64_t v = 0;
    const caliper_net::Json* j = o.is_object() ? o.find(key) : nullptr;
    return j && j->get_u64(&v) && v <= 0xffffffffull ? static_cast<uint32_t>(v) : 0;
}

}  // namespace program_sync_detail

inline ProgramSync::Entry* ProgramSync::find(const std::string& id) {
    for (Entry& e : entries_)
        if (e.rec.want.id == id) return &e;
    return nullptr;
}

inline void ProgramSync::want(std::vector<WantedProgram> programs) {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<Entry> next;
    next.reserve(programs.size());
    for (WantedProgram& w : programs) {
        Entry e;
        if (Entry* old = find(w.id)) {
            e = *old;
            if (old->rec.want.sha256 != w.sha256 || old->rec.want.declaration != w.declaration ||
                old->rec.want.scene != w.scene || old->rec.want.piece != w.piece || old->rec.want.bind != w.bind ||
                old->rec.want.panel != w.panel || old->rec.want.manifest != w.manifest) {
                e.sent_sha.clear();   // changed: send it again
                e.sends = 0;
                e.pending = 0;
                e.params_dirty = false;
            } else if (old->rec.want.params != w.params && !e.sent_sha.empty()) {
                e.params_dirty = true;   // only the values: PROGRAM op params, no new set
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
            e.rec.item = caliper_net::Json();
            e.rec.state = "refused";
            e.rec.error_code = e.rec.want.refused_code;
            e.rec.error_message = e.rec.want.refused_message;
            e.rec.line = e.rec.column = 0;
            note_change(e);
        }
    }
}

inline void ProgramSync::note_change(Entry& e) {
    const std::string sha = program_sync_detail::str_of(e.rec.item, "sha256");
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

inline void ProgramSync::apply_item(const caliper_net::Json& item) {
    const std::string id = program_sync_detail::str_of(item, "id");
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
    if (program_sync_detail::str_of(item, "sha256") == e->rec.want.sha256) e->sent_sha = e->rec.want.sha256;
    e->rec.state = program_sync_detail::str_of(item, "state");
    if (e->rec.state.empty()) e->rec.state = "waiting";
    const caliper_net::Json* err = item.find("error");
    if (err && err->is_object()) {
        e->rec.error_code = program_sync_detail::str_of(*err, "code");
        e->rec.error_message = program_sync_detail::str_of(*err, "message");
        e->rec.line = program_sync_detail::u32_of(*err, "line");
        e->rec.column = program_sync_detail::u32_of(*err, "column");
    } else {
        e->rec.error_code.clear();
        e->rec.error_message.clear();
        e->rec.line = e->rec.column = 0;
    }
    note_change(*e);
}

inline void ProgramSync::poll(ProgramLink& link) {
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
                e.rec.item = caliper_net::Json();
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
        for (const caliper_net::Json& item : a.programs.items()) {
            reported.push_back(program_sync_detail::str_of(item, "id"));
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
                e.rec.item = caliper_net::Json();
            }
        }
    }

    for (Entry& e : entries_) {
        if (!e.rec.want.refused_code.empty() || e.sent_sha == e.rec.want.sha256) continue;
        if (e.sends >= kMaxSends) continue;
        const WantedProgram& w = e.rec.want;
        caliper_net::wire::Program m;
        m.op = CALIPER_NET_PROGRAM_SET;
        m.id = w.id;
        if (w.instance()) {
            // An instance goes only to a script that takes pieces; until then it shows why.
            if (!link.takes_pieces()) {
                if (e.rec.error_code != "program.resources_unavailable") {
                    e.rec.host = "none";
                    e.rec.state = "refused";
                    e.rec.item = caliper_net::Json();
                    e.rec.error_code = "program.resources_unavailable";
                    e.rec.error_message = "The attached script takes no pieces: it and the server must announce the "
                                          "capability resources, which Caliper LIB does under "
                                          "caliper.watch(model, optimizer).";
                    e.rec.line = e.rec.column = 0;
                    note_change(e);
                }
                continue;
            }
            m.piece = w.piece;
            m.manifest = w.manifest;
            if (w.bind.is_object() && w.bind.size() > 0) m.bind = w.bind;
            if (w.params.is_object()) m.params = w.params;
            if (!w.panel.empty()) m.panel = w.panel;
            if (w.segments.is_array()) m.segments = w.segments;
            if (!w.common.empty()) m.common = w.common;
            if (!w.derive.empty()) m.derive = w.derive;
            if (!w.draw.empty()) m.draw = w.draw;
        } else {
            m.declaration = w.declaration;
            if (w.params.is_object()) m.declaration.set("params", w.params);
            m.wgsl = w.wgsl;
            m.sha256 = w.sha256;
            if (!w.scene.empty()) m.scene = w.scene;
        }
        const uint32_t rid = link.send(std::move(m));
        if (rid == 0) break;   // the queue is full: the next frame tries again
        e.pending = rid;
        e.sent_sha = e.rec.want.sha256;
        e.params_dirty = false;   // the set carries the values of now
        ++e.sends;
    }
    // New values of a program that the script holds: they apply at its next pass.
    for (Entry& e : entries_) {
        if (!e.params_dirty || e.sent_sha != e.rec.want.sha256 || !e.rec.want.params.is_object()) continue;
        caliper_net::wire::Program m;
        m.op = CALIPER_NET_PROGRAM_PARAMS;
        m.id = e.rec.want.id;
        m.params = e.rec.want.params;
        if (link.send(std::move(m)) == 0) break;
        e.params_dirty = false;
    }
    while (!removing_.empty()) {
        caliper_net::wire::Program m;
        m.op = CALIPER_NET_PROGRAM_REMOVE;
        m.id = removing_.front();
        if (link.send(std::move(m)) == 0) break;
        removing_.erase(removing_.begin());
    }
}

inline std::vector<ProgramRecord> ProgramSync::records() const {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<ProgramRecord> out;
    out.reserve(entries_.size());
    for (const Entry& e : entries_) out.push_back(e.rec);
    return out;
}

inline std::vector<ProgramChange> ProgramSync::take_changes() {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<ProgramChange> out;
    out.swap(changes_);
    return out;
}

}  // namespace caliper_live

#if defined(CALIPER_WITH_BUILDER) && CALIPER_WITH_BUILDER
// ---- the builder session's ProgramHost (the caliper executable only) --------------------------
#include "sha256.h"

#include <caliper_builder/operators/operators.h>
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/chain_hooks.h>
#include <caliper_builder/session/program_host.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/instrument.h>
#include <caliper_builder/spec/program.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <optional>

namespace caliper_live {

class AppProgramHost final : public caliper_builder::ProgramHost {
public:
    // The programs of `spec`, each with its declaration, its target scene and
    // its hash, and the program of each operator that runs in the script
    // (design/instrument-interface.md, Operator Programs); the next pump()
    // sends what differs. Frame thread.
    void reconcile(const caliper_builder::spec::Spec& spec) override {
        spec_ = spec;
        operators_.clear();
        for (const auto& source : spec_.sources)
            if (caliper_builder::operator_runs_in_script(spec_, source, 0)) {
                Operator op;
                op.source = source;
                op.source.catalog = root_catalog(spec_, source);
                operators_.push_back(std::move(op));
            }
        rebuild();
    }

    // How many values of a reference the declared range of their param refused.
    std::uint64_t refused_params() const { return refused_params_; }

    // Where the pieces of the library and the workspace come from: the piece
    // that a pin names, or nothing when its source lacks it. An inline piece
    // comes from the open spec. Without a lookup, a program pinned to a
    // library or workspace piece is refused with piece.missing. Frame thread.
    using PieceLookup =
        std::function<std::optional<caliper_builder::spec::Piece>(const caliper_builder::spec::PieceRef&)>;
    void set_pieces(PieceLookup lookup) {
        pieces_ = std::move(lookup);
        rebuild();
    }

    // Moves each RESOURCE version that `link` received into `history`, with
    // the `keep` and the element's fields of its resource in the piece of its
    // program; the history keeps only the versions of the open spec's
    // instances. Frame thread.
    void deliver_resources(ProgramLink& link, caliper_builder::StepHistory& history) {
        for (ReceivedResource& r : link.take_resources()) {
            ++resources_received_;
            std::uint32_t keep = 0;
            std::vector<caliper_builder::ResourceField> fields;
            if (auto p = resource_info_.find(r.program); p != resource_info_.end())
                if (auto q = p->second.find(r.resource); q != p->second.end()) {
                    keep = q->second.keep;
                    fields = q->second.fields;
                }
            caliper_builder::ResourceVersion v;
            v.step = r.step;
            v.count = r.count;
            v.stride = r.stride;
            v.width = r.width;
            v.height = r.height;
            v.bytes = std::move(r.bytes);
            history.append_resource(r.program, r.resource, keep, std::move(fields), std::move(v));
        }
    }
    // The RESOURCE versions moved into the history so far.
    std::uint64_t resources_received() const { return resources_received_; }

    std::vector<caliper_builder::ProgramStatus> status() const override {
        std::vector<caliper_builder::ProgramStatus> out;
        for (const ProgramRecord& r : sync_.records()) out.push_back(to_status(r));
        return out;
    }

    // What a reference reads now (ValueStore::value() in the app).
    using Evaluate = std::function<nlohmann::json(const caliper_builder::spec::Reference&)>;

    // Evaluates the references of each program's params with `evaluate` and,
    // when `resolver` is given, asks it for the shape of each operator's
    // input. A changed value goes out at the next pump() as PROGRAM op params
    // (applied at the next pass, with no new hash), and a changed shape sends
    // the operator again. Returns whether anything changed. Frame thread.
    bool refresh(const Evaluate& evaluate, const caliper_builder::spec::SourceResolver* resolver) {
        bool changed = false;
        if (evaluate)
            for (auto& [id, params] : params_) {
                (void)id;
                for (auto& [name, slot] : params) {
                    (void)name;
                    if (!caliper_builder::spec::is_reference(slot.place)) continue;
                    const nlohmann::json v = evaluate(std::get<caliper_builder::spec::Reference>(slot.place));
                    if (!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>() == slot.value) continue;
                    // A value outside the declared range of the field in
                    // struct Params is refused: the program keeps the last
                    // value it took, and nothing goes to the script.
                    if (slot.type && !caliper_builder::programs::param_in_range(*slot.type, v.get<double>())) {
                        ++refused_params_;
                        continue;
                    }
                    slot.value = v.get<double>();
                    changed = true;
                }
            }
        if (resolver && *resolver)
            for (Operator& op : operators_) {
                std::vector<std::int64_t> shape;
                if (op.source.catalog)
                    if (auto r = (*resolver)(*op.source.catalog))
                        if (r->shape) shape = *r->shape;
                if (shape != op.shape) {
                    op.shape = std::move(shape);
                    changed = true;
                }
            }
        if (changed) rebuild();
        return changed;
    }

    // Once per frame with the open session: refresh() with the session's
    // values, and with its catalog every kShapeFrames frames, then the sync
    // and the program events. Frame thread.
    void pump(ProgramLink& link, caliper_builder::Session* session) {
        if (session) {
            const caliper_builder::ValueStore& values = session->values();
            const Evaluate evaluate = [&values](const caliper_builder::spec::Reference& r) { return values.value(r); };
            if (!operators_.empty() && frames_++ % kShapeFrames == 0) {
                const caliper_builder::spec::SourceResolver resolver = session->source_resolver();
                refresh(evaluate, &resolver);
            } else if (!params_.empty()) {
                refresh(evaluate, nullptr);
            }
        }
        pump(link, session ? &session->events() : nullptr);
        if (session) deliver_resources(link, session->history());
    }

    // Once per frame: runs the sync through `link` and pushes one `program`
    // event for each change into `sink` (none when null). Frame thread.
    void pump(ProgramLink& link, caliper_builder::EventSink* sink) {
        sync_.poll(link);
        std::vector<ProgramChange> changes = sync_.take_changes();
        if (!sink || changes.empty()) return;
        const std::vector<ProgramRecord> recs = sync_.records();
        for (const ProgramChange& c : changes)
            for (const ProgramRecord& r : recs)
                if (r.want.id == c.id) caliper_builder::push_program_event(*sink, to_status(r), c.change);
    }

    // The wanted program of `p`: its declaration without the source, the
    // scene of a geometry output's panel, the SHA-256 of the source, and its
    // params with `values` in place of the references (each reference's
    // default, else 0, when `values` has none).
    static WantedProgram wanted_program(const caliper_builder::spec::Spec& spec,
                                        const caliper_builder::spec::Program& p,
                                        const std::map<std::string, double>* values = nullptr) {
        namespace sp = caliper_builder::spec;
        WantedProgram w;
        w.id = p.id;
        w.wgsl = p.wgsl;
        w.sha256 = caliper_host::sha256_hex(p.wgsl.data(), p.wgsl.size());
        const bool geometry = p.output.kind == sp::OutputKind::geometry;
        const bool tensor = p.output.kind == sp::OutputKind::tensor;
        w.kind = p.builtin ? "operator" : sp::output_kind_name(p.output.kind);
        nlohmann::json out = {{"kind", sp::output_kind_name(p.output.kind)}};
        if (geometry) {
            w.panel = p.output.panel;
            for (const auto& panel : spec.panels) {
                if (panel.id != p.output.panel || panel.primitive != sp::Primitive::scene) continue;
                auto it = panel.bind.find(sp::kSceneSlot);
                if (it != panel.bind.end() && !it->second.bindings.empty())
                    if (auto name = sp::live_scene_name(it->second.bindings.front())) w.scene = *name;
            }
            out["panel"] = p.output.panel;
            out["entity"] = p.output.entity.value_or("programs/" + p.id);
            out["part"] = p.output.part == sp::GeometryPart::lines ? "lines" : "points";
            out["max_count"] = p.output.max_count;
            nlohmann::json arrays = nlohmann::json::array();
            for (sp::GeometryArray a : p.output.arrays)
                arrays.push_back(a == sp::GeometryArray::positions ? "positions"
                                 : a == sp::GeometryArray::colors  ? "colors"
                                 : a == sp::GeometryArray::values  ? "values"
                                                                   : "radii");
            out["arrays"] = arrays;
            nlohmann::json style = nlohmann::json::object();
            if (p.output.style.colormap) style["colormap"] = *p.output.style.colormap;
            if (p.output.style.vmin) style["vmin"] = *p.output.style.vmin;
            if (p.output.style.vmax) style["vmax"] = *p.output.style.vmax;
            if (!style.empty()) out["style"] = style;
        } else if (tensor) {
            out["shape"] = p.output.shape;
        } else {
            out["width"] = p.output.width;
            out["height"] = p.output.height;
            out["format"] = "rgba8unorm";
        }
        nlohmann::json d = {{"output", out}};
        if (p.title) d["title"] = *p.title;
        if (p.entry) d["entry"] = *p.entry;
        nlohmann::json inputs = nlohmann::json::array();
        for (const auto& in : p.inputs) inputs.push_back({{"var", in.var}, {"source", in.source}});
        d["inputs"] = inputs;
        nlohmann::json limits = nlohmann::json::object();
        if (p.limits.time_ms) limits["time_ms"] = *p.limits.time_ms;
        if (p.limits.memory_mb) limits["memory_mb"] = *p.limits.memory_mb;
        if (p.limits.every) limits["every"] = *p.limits.every;
        if (!limits.empty()) d["limits"] = limits;
        if (p.workgroups) d["workgroups"] = *p.workgroups;
        if (p.builtin) d["builtin"] = *p.builtin;
        const std::string text = d.dump();
        std::string err;
        caliper_net::Json::parse(text.data(), text.size(), &w.declaration, &err);
        if (!p.params.empty()) {
            nlohmann::json params = nlohmann::json::object();
            for (const auto& [name, place] : p.params) {
                double v = 0;
                if (!sp::is_reference(place)) {
                    v = std::get<double>(place);
                } else if (values && values->count(name)) {
                    v = values->at(name);
                } else {
                    const nlohmann::json& fallback = std::get<sp::Reference>(place).fallback;
                    if (fallback.is_number()) v = fallback.get<double>();
                }
                params[name] = v;
            }
            const std::string ptext = params.dump();
            caliper_net::Json::parse(ptext.data(), ptext.size(), &w.params, &err);
        }
        if (p.id.size() > CALIPER_NET_PROGRAM_ID_MAX) {
            w.refused_code = "program.bad_target";
            w.refused_message = "A program id is at most " + std::to_string(CALIPER_NET_PROGRAM_ID_MAX) +
                                " bytes on the wire, so that its view program.<id> fits a view name" +
                                (p.builtin ? "; give the operator's source a shorter id." : ".");
        } else if (geometry && w.scene.empty()) {
            w.refused_code = "program.bad_target";
            w.refused_message = "The panel " + p.output.panel + " is not a scene panel bound to a live scene.";
        }
        return w;
    }

    // The wanted program of an instance of a piece: the pin, the manifest as
    // canonical JSON, the bindings, the params with `values` in place of the
    // references, the panel, and the sections with the imports resolved:
    // `common` holds the lib modules and common.wgsl, and `segments` their
    // lines. A missing piece, another hash than the pin's, or imports that do
    // not resolve refuse it before anything goes out.
    static WantedProgram wanted_instance(const caliper_builder::spec::Program& p,
                                         const caliper_builder::spec::Piece* piece,
                                         const std::map<std::string, double>& values) {
        namespace sp = caliper_builder::spec;
        namespace pg = caliper_builder::programs;
        WantedProgram w;
        w.id = p.id;
        w.kind = "piece";
        w.panel = p.panel.value_or("");
        const sp::PieceRef& ref = *p.piece;
        const std::string hex_prefix = "sha256:";
        w.sha256 = ref.hash.compare(0, hex_prefix.size(), hex_prefix) == 0 ? ref.hash.substr(hex_prefix.size()) : ref.hash;
        w.piece = caliper_net::Json::object();
        w.piece.set("name", ref.name);
        w.piece.set("version", ref.version);
        w.piece.set("hash", ref.hash);
        auto refuse = [&](const std::string& code, const std::string& message) {
            w.refused_code = code;
            w.refused_message = message;
            return w;
        };
        const std::string pin = ref.name + "@" + std::to_string(ref.version);
        if (p.id.size() > CALIPER_NET_PROGRAM_ID_MAX)
            return refuse("program.bad_target", "A program id is at most " + std::to_string(CALIPER_NET_PROGRAM_ID_MAX) +
                                                    " bytes on the wire.");
        if (!piece)
            return refuse(pg::kPieceMissing, "The piece " + pin + " that the program " + p.id + " pins is not in " +
                                                 sp::piece_from_name(ref.from) + ".");
        // The app's check of the hash is the one that counts: the script gets
        // the resolved text without the author's files, thus it checks only the
        // hash's form. The hash is computed here from the text that goes out,
        // never taken from the piece's own `hash`, which an inline piece states.
        const std::string hash = pg::piece_hash(*piece);
        if (hash != ref.hash)
            return refuse(pg::kPieceHashMismatch, "The piece " + pin + " has the hash " + hash +
                                                      ", and the program " + p.id + " pins " + ref.hash + ".");
        const pg::Resolved resolved = pg::resolve_imports(*piece);
        if (!resolved.verdict.ok()) return refuse(resolved.verdict.code, resolved.verdict.message);
        auto file = [&](const char* name) {
            auto it = resolved.files.find(name);
            return it == resolved.files.end() ? std::string() : it->second;
        };
        auto lines = [](const std::string& t) {
            std::uint32_t n = (std::uint32_t)std::count(t.begin(), t.end(), '\n');
            return t.empty() || t.back() == '\n' ? n : n + 1;
        };
        const std::string common_file = file("common.wgsl");
        w.common = resolved.lib_text + common_file;
        w.segments = caliper_net::Json::array();
        for (const pg::Segment& seg : resolved.lib_segments) {
            caliper_net::Json j = caliper_net::Json::object();
            j.set("part", seg.part);
            j.set("first_line", seg.first_line);
            j.set("lines", seg.lines);
            w.segments.push(std::move(j));
        }
        if (!common_file.empty()) {
            caliper_net::Json j = caliper_net::Json::object();
            j.set("part", "common.wgsl");
            j.set("first_line", lines(resolved.lib_text) + 1);
            j.set("lines", lines(common_file));
            w.segments.push(std::move(j));
        }
        if (piece->derive) w.derive = file("derive.wgsl");
        if (piece->draw) w.draw = file("draw.wgsl");
        if (w.derive.empty() && w.draw.empty())
            return refuse(pg::kPieceBadManifest, "The piece " + pin + " has neither a derive half nor a draw half.");
        std::string err;
        const std::string manifest = pg::canonical_manifest(*piece);
        if (manifest.empty() || !caliper_net::Json::parse(manifest.data(), manifest.size(), &w.manifest, &err))
            return refuse(pg::kProgramNotBuilt, "The manifest of the piece " + pin + " has no canonical form yet.");
        w.bind = caliper_net::Json::object();
        for (const auto& [input, binding] : p.bind) w.bind.set(input, binding);
        if (!values.empty()) {
            w.params = caliper_net::Json::object();
            for (const auto& [name, v] : values) w.params.set(name, v);
        }
        return w;
    }

    // The status item of program.status of `r`.
    static caliper_builder::ProgramStatus to_status(const ProgramRecord& r) {
        caliper_builder::ProgramStatus s;
        s.id = r.want.id;
        s.kind = r.want.kind;
        s.host = r.host;
        s.state = r.state;
        s.panel = r.want.panel;
        const caliper_net::Json& it = r.item;
        auto u32 = [&](const char* k) -> std::optional<std::uint32_t> {
            uint64_t v = 0;
            const caliper_net::Json* j = it.is_object() ? it.find(k) : nullptr;
            if (j && j->get_u64(&v) && v <= 0xffffffffull) return static_cast<std::uint32_t>(v);
            return std::nullopt;
        };
        auto u64 = [&](const char* k) -> std::uint64_t {
            uint64_t v = 0;
            const caliper_net::Json* j = it.is_object() ? it.find(k) : nullptr;
            return j && j->get_u64(&v) ? v : 0;
        };
        auto num = [&](const char* k) -> std::optional<double> {
            const caliper_net::Json* j = it.is_object() ? it.find(k) : nullptr;
            if (j && j->is_number()) return j->as_double();
            return std::nullopt;
        };
        s.every = u32("every");
        s.every_min = u32("every_min");
        s.last_pass_ms = num("last_pass_ms");
        s.last_cost_ms = num("last_cost_ms");
        s.step_ms = num("step_ms");
        s.overhead_share = num("overhead_share");
        s.passes_per_s = num("passes_per_s");
        if (const caliper_net::Json* j = it.is_object() ? it.find("last_step") : nullptr) {
            int64_t v = 0;
            if (j->get_i64(&v)) s.last_step = v;
        }
        s.runs = u64("runs");
        s.memory_bytes = u64("memory_bytes");
        s.memory_limit_bytes = u64("memory_limit_bytes");
        if (const caliper_net::Json* j = it.is_object() ? it.find("sha256") : nullptr)
            if (j->is_string() && !j->as_string().empty()) s.sha256 = j->as_string();
        if (!r.error_code.empty())
            s.error = caliper_builder::ProgramError{r.error_code, r.error_message, r.line, r.column};
        // The script runs an instance's derive half; its events name that half.
        if (r.want.kind == "piece" && !r.want.derive.empty()) s.half = "derive";
        return s;
    }

    // The catalog id of the root of `source`'s chain: its own catalog, or the
    // catalog of the first source up its `from` links that has one.
    static std::optional<std::string> root_catalog(const caliper_builder::spec::Spec& spec,
                                                   const caliper_builder::spec::Source& source) {
        const caliper_builder::spec::Source* s = &source;
        for (std::size_t depth = 0; s && depth <= spec.sources.size(); ++depth) {
            if (s->catalog) return s->catalog;
            if (!s->from) return std::nullopt;
            const caliper_builder::spec::Source* up = nullptr;
            for (const auto& c : spec.sources)
                if (c.id == *s->from) up = &c;
            s = up;
        }
        return std::nullopt;
    }

private:
    // The frames between two reads of the operators' input shapes.
    static constexpr std::uint64_t kShapeFrames = 30;

    // An operator that runs in the script: its source with the root's
    // catalog, and the input shape the catalog gave last (empty before).
    struct Operator {
        caliper_builder::spec::Source source;
        std::vector<std::int64_t> shape;
    };
    // One param of a program: its place in the spec, its value now, and its
    // type in the program's struct Params when the WGSL declares it.
    struct ParamSlot {
        caliper_builder::spec::ValueOr<double> place;
        double value = 0;
        std::optional<caliper_builder::programs::ParamType> type;
    };

    // The keep and the element's fields of one published resource.
    struct ResourceInfo {
        std::uint32_t keep = 0;
        std::vector<caliper_builder::ResourceField> fields;
    };

    // The piece that `ref` pins: an inline piece of the open spec, or one of
    // the lookup's.
    std::optional<caliper_builder::spec::Piece> find_piece(const caliper_builder::spec::PieceRef& ref) const {
        if (ref.from == caliper_builder::spec::PieceFrom::inline_piece) {
            for (const auto& piece : spec_.pieces)
                if (piece.name == ref.name && piece.version == ref.version) return piece;
            return std::nullopt;
        }
        return pieces_ ? pieces_(ref) : std::nullopt;
    }

    // The keep and fields of each published resource of `piece`, for the history.
    static std::map<std::string, ResourceInfo> resource_infos(const caliper_builder::spec::Piece& piece) {
        namespace sp = caliper_builder::spec;
        std::map<std::string, ResourceInfo> out;
        for (const sp::Resource& r : piece.resources) {
            if (!r.publish) continue;
            ResourceInfo info;
            info.keep = r.keep.value_or(0);
            if (r.kind == sp::ResourceKind::texture) {
                info.fields.push_back({"texel", "u32", 0});
            } else {
                caliper_builder::programs::ElementLayout layout;
                if (caliper_builder::programs::element_layout(r.layout, &layout).ok())
                    for (const auto& f : layout.fields) info.fields.push_back({f.name, f.type, f.offset});
            }
            out[r.id] = std::move(info);
        }
        return out;
    }

    // The wanted programs from the spec, the operators and the params' values.
    void rebuild() {
        namespace sp = caliper_builder::spec;
        std::vector<WantedProgram> wanted;
        std::map<std::string, std::map<std::string, ParamSlot>> next;
        resource_info_.clear();
        for (const auto& p : spec_.programs) {
            std::map<std::string, double> values;
            std::vector<caliper_builder::programs::ParamField> fields;
            std::optional<sp::Piece> piece;
            if (p.piece) {
                piece = find_piece(*p.piece);
                if (piece) {
                    resource_info_[p.id] = resource_infos(*piece);
                    for (const sp::PieceParam& pp : piece->params)
                        fields.push_back({pp.name, static_cast<caliper_builder::programs::ParamType>(pp.type)});
                }
            } else {
                caliper_builder::programs::params_fields(p.wgsl, &fields, nullptr);
            }
            for (const auto& [name, place] : p.params) {
                ParamSlot slot{place, 0, std::nullopt};
                for (const auto& f : fields)
                    if (f.name == name) slot.type = f.type;
                if (!sp::is_reference(place)) {
                    slot.value = std::get<double>(place);
                } else if (auto old = params_.find(p.id); old != params_.end() && old->second.count(name)) {
                    slot.value = old->second.at(name).value;   // the last evaluated value
                } else if (const auto& d = std::get<sp::Reference>(place).fallback;
                           d.is_number() && (!slot.type || caliper_builder::programs::param_in_range(*slot.type, d.get<double>()))) {
                    slot.value = d.get<double>();
                }
                values[name] = slot.value;
                next[p.id][name] = slot;
            }
            if (p.piece) wanted.push_back(wanted_instance(p, piece ? &*piece : nullptr, values));
            else wanted.push_back(wanted_program(spec_, p, &values));
        }
        for (const Operator& op : operators_) {
            const sp::Program p = caliper_builder::operators::operator_program(op.source, 0, op.shape);
            // Before the catalog gives the input's shape, the operator waits
            // and program.status reports it with host none.
            if (op.shape.empty()) continue;
            WantedProgram w = wanted_program(spec_, p);
            if (p.output.shape.empty()) {
                w.refused_code = "spec.operator_shape";
                w.refused_message = "The operator " + std::string(sp::to_string(op.source.transforms[0].op)) +
                                    " takes a tensor of rank 2, and " + op.source.catalog.value_or("its input") +
                                    " has rank " + std::to_string(op.shape.size()) + ".";
            }
            wanted.push_back(std::move(w));
        }
        params_ = std::move(next);
        sync_.want(std::move(wanted));
    }

    ProgramSync sync_;
    caliper_builder::spec::Spec spec_;
    std::vector<Operator> operators_;
    std::map<std::string, std::map<std::string, ParamSlot>> params_;   // by program id, then name
    std::uint64_t frames_ = 0;
    std::uint64_t refused_params_ = 0;   // values refused for the declared range of their field
    PieceLookup pieces_;                 // the library's and the workspace's pieces
    std::map<std::string, std::map<std::string, ResourceInfo>> resource_info_;   // by program id, then resource id
    std::uint64_t resources_received_ = 0;
};

}  // namespace caliper_live
#endif
