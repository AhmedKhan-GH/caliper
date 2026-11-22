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
// Threads: want(), poll() and take_changes() run on the frame thread;
// records() may run on any thread.
#pragma once
#include "live/live_net_source.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace caliper_live {

class Attachment;

// One program that the app wants the script to run.
struct WantedProgram {
    std::string id;
    std::string kind;                  // "geometry" or "pixels"
    std::string panel;                 // geometry: the target scene panel's id; "" otherwise
    std::string scene;                 // geometry: that panel's scene name; "" otherwise
    std::string wgsl;
    std::string sha256;                // of wgsl, lowercase hex
    caliper_net::Json declaration;     // the declaration without wgsl
    // Set when the app refuses the program before it sends it: the record
    // shows it as refused and nothing goes out.
    std::string refused_code, refused_message;
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

}  // namespace caliper_live

#if defined(CALIPER_WITH_BUILDER) && CALIPER_WITH_BUILDER
// ---- the builder session's ProgramHost (the caliper executable only) --------------------------
#include "sha256.h"

#include <caliper_builder/session/program_host.h>
#include <caliper_builder/spec/instrument.h>
#include <caliper_builder/spec/program.h>

#include <nlohmann/json.hpp>

namespace caliper_live {

class AppProgramHost final : public caliper_builder::ProgramHost {
public:
    // The programs of `spec`, each with its declaration, its target scene and
    // its hash; the next pump() sends what differs. Frame thread.
    void reconcile(const caliper_builder::spec::Spec& spec) override {
        std::vector<WantedProgram> wanted;
        for (const auto& p : spec.programs) wanted.push_back(wanted_program(spec, p));
        sync_.want(std::move(wanted));
    }

    std::vector<caliper_builder::ProgramStatus> status() const override {
        std::vector<caliper_builder::ProgramStatus> out;
        for (const ProgramRecord& r : sync_.records()) out.push_back(to_status(r));
        return out;
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
    // scene of a geometry output's panel, and the SHA-256 of the source.
    static WantedProgram wanted_program(const caliper_builder::spec::Spec& spec,
                                        const caliper_builder::spec::Program& p) {
        namespace sp = caliper_builder::spec;
        WantedProgram w;
        w.id = p.id;
        w.wgsl = p.wgsl;
        w.sha256 = caliper_host::sha256_hex(p.wgsl.data(), p.wgsl.size());
        const bool geometry = p.output.kind == sp::OutputKind::geometry;
        w.kind = geometry ? "geometry" : "pixels";
        nlohmann::json out = {{"kind", w.kind}};
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
        const std::string text = d.dump();
        std::string err;
        caliper_net::Json::parse(text.data(), text.size(), &w.declaration, &err);
        if (p.id.size() > CALIPER_NET_PROGRAM_ID_MAX) {
            w.refused_code = "program.bad_target";
            w.refused_message = "A program id is at most " + std::to_string(CALIPER_NET_PROGRAM_ID_MAX) +
                                " bytes on the wire, so that its view program.<id> fits a view name.";
        } else if (geometry && w.scene.empty()) {
            w.refused_code = "program.bad_target";
            w.refused_message = "The panel " + p.output.panel + " is not a scene panel bound to a live scene.";
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
        return s;
    }

private:
    ProgramSync sync_;
};

}  // namespace caliper_live
#endif
