// The app's host of the draw halves of a spec's programs (draw_host.h).
#include "draw/draw_host.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>

#include <algorithm>
#include <cstring>

namespace caliper_draw {

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;

namespace {

const char* state_name(DrawState s) {
    switch (s) {
        case DrawState::waiting: return "waiting";
        case DrawState::compiled: return "compiled";
        case DrawState::running: return "running";
        case DrawState::refused: return "refused";
        case DrawState::stopped: return "stopped";
        case DrawState::review: return "review";
    }
    return "waiting";
}

Verdict refusal(const char* code, std::string message, std::string where = {}) {
    Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.where = std::move(where);
    return v;
}

// Whether the program `p` of `spec` may draw its resolved `piece`: always in the person's own
// spec, and in a foreign spec only when provenance.reviewed holds program:<id> with the 64 hex
// digits of the hash of the piece that would compile, whatever its source (library, workspace
// or inline). Else program.needs_review.
Verdict review(const cs::Spec& spec, const cs::Program& p, const cs::Piece& piece) {
    if (!spec.provenance.foreign) return {};
    const std::string hash = cp::piece_hash(piece);
    const std::string hex = hash.substr(hash.find(':') + 1);
    const cp::Verdict v = cp::check_provenance(spec.provenance, cs::program_target(p), hex);
    if (v.ok()) return {};
    return refusal(cp::kProgramNeedsReview, v.message, "piece");
}

// "program:<id>/<resource>" as its two names; false for another form.
bool program_binding(const std::string& bind, std::string* program, std::string* resource) {
    static const std::string prefix = "program:";
    if (bind.compare(0, prefix.size(), prefix) != 0) return false;
    const auto slash = bind.find('/', prefix.size());
    if (slash == std::string::npos) return false;
    *program = bind.substr(prefix.size(), slash - prefix.size());
    *resource = bind.substr(slash + 1);
    return !program->empty() && !resource->empty();
}

}  // namespace

nlohmann::json draw_status_json(const DrawStatus& s) {
    nlohmann::json j = {{"state", state_name(s.state)},
                        {"frame_ms", s.frame_ms < 0 ? nlohmann::json(nullptr) : nlohmann::json(s.frame_ms)},
                        {"frame_limit_ms", s.frame_limit_ms},
                        {"memory_bytes", s.memory_bytes},
                        {"memory_limit_bytes", s.memory_limit_bytes},
                        {"resource_step", s.resource_step < 0 ? nlohmann::json(nullptr) : nlohmann::json(s.resource_step)},
                        {"resource_steps", s.resource_steps},
                        {"params", s.params},
                        {"frames", s.frames},
                        {"error", nullptr}};
    if (!s.error.ok()) {
        j["error"] = {{"code", s.error.code}, {"message", s.error.message}, {"where", s.error.where},
                      {"line", s.error.line}, {"column", s.error.column}};
    }
    return j;
}

caliper_builder::Event draw_event(const std::string& program, const std::string& panel, const DrawStatus& s,
                                  const char* change) {
    caliper_builder::Event e;
    e.kind = caliper_builder::EventKind::program;
    e.panel = panel;
    e.origin = "ui";
    e.fields = {{"program", program}, {"change", change}, {"host", "viewer"}, {"half", "draw"},
                {"pass", s.pass >= 0 ? nlohmann::json(s.pass) : nlohmann::json(nullptr)}};
    if (!s.error.ok()) {
        e.fields["code"] = s.error.code;
        e.fields["message"] = s.error.message;
        e.fields["where"] = s.error.where;
        e.fields["line"] = s.error.line;
        e.fields["column"] = s.error.column;
        if (!s.entry.empty()) e.fields["entry"] = s.entry;
    }
    return e;
}

// ---- the feed of one panel ------------------------------------------------------------------

class DrawHost::Panel final : public DrawFeed {
public:
    Panel(DrawHost& host, std::string id) : host_(host), id_(std::move(id)) {}

    std::uint64_t revision() const override { return host_.revision_; }

    std::vector<PanelProgram> programs() const override {
        std::vector<PanelProgram> out;
        for (const Record& r : host_.records_)
            if (r.panel == id_) out.push_back(r.draw);
        return out;
    }

    void frame(DrawRuntime& rt, FrameInputs* in) override {
        const Record* r = nullptr;
        for (const Record& rec : host_.records_)
            if (rec.program == rt.instance().program) r = &rec;
        if (r == nullptr) return;
        Given& given = given_[r->program];
        std::uint64_t epoch = 0;
        // Each published resource of the piece, then each buffer input from its program.
        auto give = [&](const std::string& program, const std::string& resource, const std::string& as) {
            if (!host_.resources_) return;
            ResourceVersion v;
            std::uint64_t e = 0;
            if (!host_.resources_(program, resource, host_.cursor_.step, &v, &e)) return;
            epoch = std::max(epoch, e);
            // A runtime installed again, in this view or another, holds no version yet, and a
            // version that replaced another at the same step has another serial.
            Given::Version& last = given.versions[as];
            if (rt.resource_step(as) == v.step && last.epoch == e && last.serial == v.serial) return;
            v.resource = as;
            if (rt.set_resource(v).ok()) last = {e, v.serial};
        };
        for (const cs::Resource& res : r->draw.piece.resources)
            if (res.publish) give(r->program, res.id, res.id);
        for (const cs::PieceInput& input : r->draw.piece.inputs) {
            if (input.kind != cs::InputKind::buffer) continue;
            const auto b = r->draw.instance.bind.find(input.name);
            std::string program, resource;
            if (b != r->draw.instance.bind.end() && program_binding(b->second, &program, &resource))
                give(program, resource, input.name);
        }
        in->params = host_.pack_params(*r);
        in->selection = host_.selection_;
        in->cursor_step = host_.cursor_.step;
        in->latest_step = host_.cursor_.latest;
        in->live = host_.cursor_.live;
        in->cursor_moved = given.moves != host_.cursor_.moves;
        given.moves = host_.cursor_.moves;
        in->epoch = epoch;
    }

    void status(const DrawRuntime& rt) override { host_.report(rt.instance().program, id_, rt.status()); }

private:
    struct Given {
        struct Version {
            std::uint64_t epoch = 0, serial = 0;
        };
        std::map<std::string, Version> versions;   // resource: the epoch and serial of the version given
        std::uint64_t moves = 0;
    };
    DrawHost& host_;
    std::string id_;
    std::map<std::string, Given> given_;
};

// ---- the host -------------------------------------------------------------------------------

DrawHost::DrawHost() = default;
DrawHost::~DrawHost() = default;

void DrawHost::set_resolver(Resolver resolver) { resolver_ = std::move(resolver); }
void DrawHost::set_resources(Resources resources) { resources_ = std::move(resources); }
void DrawHost::set_evaluate(Evaluate evaluate) { evaluate_ = std::move(evaluate); }
void DrawHost::set_events(caliper_builder::EventSink* sink) { events_ = sink; }
void DrawHost::set_cursor(const Cursor& cursor) { cursor_ = cursor; }
void DrawHost::set_selection(std::vector<Selected> selection) {
    if (selection.size() > cp::kMaxSelected) selection.resize(cp::kMaxSelected);
    selection_ = std::move(selection);
}

void DrawHost::reconcile(const cs::Spec& spec) {
    std::vector<Record> next;
    std::map<std::string, cs::CameraMode> cameras;   // the first program's camera of each panel
    for (const cs::Program& p : spec.programs) {
        if (!p.piece || !p.panel || p.panel->empty()) continue;
        Record r;
        r.program = p.id;
        r.panel = *p.panel;
        r.spec = p;
        r.draw.instance.program = p.id;
        r.draw.instance.panel = *p.panel;
        r.draw.instance.bind = p.bind;
        if (!resolver_) {
            r.draw.refused = refusal(cp::kPieceMissing, "The app has no store of pieces to resolve " + p.piece->name + ".");
        } else if (Verdict v = resolver_(spec, p, &r.draw.piece); !v.ok()) {
            r.draw.refused = v;
        } else if (!r.draw.piece.draw) {
            continue;   // a piece that only publishes draws nothing in a viewer
        } else if (Verdict v = review(spec, p, r.draw.piece); !v.ok()) {
            r.draw.refused = v;   // nothing of a foreign piece compiles before a review of its hash
        } else {
            const cs::CameraMode camera = r.draw.piece.draw->camera;
            const auto it = cameras.find(r.panel);
            if (it == cameras.end()) {
                cameras[r.panel] = camera;
            } else if (it->second != camera) {
                r.draw.refused = refusal(cp::kProgramCameraConflict,
                                         "The program " + p.id + " selects the camera " + cs::camera_mode_name(camera) +
                                             ", and another program of the panel " + r.panel + " selects " +
                                             cs::camera_mode_name(it->second) + ".",
                                         "draw.camera");
            }
        }
        // The pick id stays with the program id; a changed program installs again.
        const Record* old = nullptr;
        for (const Record& o : records_)
            if (o.program == r.program) old = &o;
        r.draw.pick_id = old ? old->draw.pick_id : next_pick_++;
        const bool same = old && old->panel == r.panel && old->draw.piece == r.draw.piece &&
                          old->draw.instance.bind == r.draw.instance.bind &&
                          old->draw.refused.code == r.draw.refused.code;
        r.draw.revision = same ? old->draw.revision : ++revision_;
        next.push_back(std::move(r));
    }
    bool changed = next.size() != records_.size();
    for (std::size_t i = 0; !changed && i < next.size(); ++i)
        changed = next[i].program != records_[i].program || next[i].draw.revision != records_[i].draw.revision;
    // Params change without a new revision: the feed packs them at each frame.
    records_ = std::move(next);
    if (changed) ++revision_;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = status_.begin(); it != status_.end();) {
            bool kept = false;
            for (const Record& r : records_) kept = kept || r.program == it->first;
            if (kept) ++it;
            else {
                reported_.erase(it->first);
                it = status_.erase(it);
            }
        }
    }
    for (const Record& r : records_) {
        if (r.draw.refused.ok()) {
            // A program that left review waits for its view to install it; only the host sets review.
            bool was_review = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto it = status_.find(r.program);
                was_review = it != status_.end() && it->second.state == DrawState::review;
            }
            if (was_review) report(r.program, r.panel, DrawStatus{});
            continue;
        }
        DrawStatus s;
        if (r.draw.refused.code == cp::kProgramNeedsReview) {
            s.state = DrawState::review;   // a state, not an error: no event
        } else {
            s.state = DrawState::refused;
            s.error = r.draw.refused;
        }
        report(r.program, r.panel, s);
    }
}

DrawFeed* DrawHost::feed(const std::string& panel) {
    auto it = panels_.find(panel);
    if (it == panels_.end()) it = panels_.emplace(panel, std::make_unique<Panel>(*this, panel)).first;
    return it->second.get();
}

std::map<std::string, DrawStatus> DrawHost::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

void DrawHost::report(const std::string& program, const std::string& panel, const DrawStatus& s) {
    DrawState before = DrawState::waiting;
    bool known = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        status_[program] = s;
        const auto it = reported_.find(program);
        known = it != reported_.end();
        if (known) before = it->second;
        reported_[program] = s.state;
    }
    if (known && before == s.state) return;
    const char* change = s.state == DrawState::compiled  ? "compiled"
                         : s.state == DrawState::refused ? "refused"
                         : s.state == DrawState::stopped ? "stopped"
                                                         : nullptr;
    if (change == nullptr || events_ == nullptr) return;
    events_->push(draw_event(program, panel, s, change));
}

std::vector<std::uint8_t> DrawHost::pack_params(const Record& r) const {
    const std::vector<cs::PieceParam>& params = r.draw.piece.params;
    std::vector<std::uint8_t> out(params.size() * 4, 0);
    for (std::size_t i = 0; i < params.size(); ++i) {
        const cs::PieceParam& p = params[i];
        double value = p.default_value;
        const auto given = r.spec.params.find(p.name);
        if (given != r.spec.params.end()) {
            if (const double* d = std::get_if<double>(&given->second)) {
                value = *d;
            } else if (evaluate_) {
                const nlohmann::json j = evaluate_(std::get<cs::Reference>(given->second));
                if (j.is_number()) value = j.get<double>();
                else if (j.is_boolean()) value = j.get<bool>() ? 1.0 : 0.0;
            }
        }
        if (p.min && value < *p.min) value = *p.min;
        if (p.max && value > *p.max) value = *p.max;
        std::uint32_t bits = 0;
        if (p.type == cs::ParamType::f32) {
            const float f = float(value);
            std::memcpy(&bits, &f, 4);
        } else if (p.type == cs::ParamType::i32) {
            const double c = std::min(2147483647.0, std::max(-2147483648.0, value));
            bits = std::uint32_t(std::int32_t(c));
        } else {
            bits = std::uint32_t(std::min(4294967295.0, std::max(0.0, value)));
        }
        std::memcpy(out.data() + 4 * i, &bits, 4);
    }
    return out;
}

}  // namespace caliper_draw
