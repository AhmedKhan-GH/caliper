// web_programs.cpp: the draw halves of pieces in the browser module (web_programs.h).
#include "web_programs.h"

#include "program_frames.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/piece.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <utility>

namespace caliper_web {

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
using caliper_net::Json;

namespace {

cd::Verdict refused(const char* code, std::string message, std::string where = {}) {
    cd::Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.where = std::move(where);
    return v;
}

std::string text_of(WGPUStringView s) {
    if (!s.data) return {};
    return std::string(s.data, s.length == WGPU_STRLEN ? std::strlen(s.data) : s.length);
}

const char* state_name(cd::DrawState s) {
    switch (s) {
        case cd::DrawState::waiting: return "waiting";
        case cd::DrawState::compiled: return "compiled";
        case cd::DrawState::running: return "running";
        case cd::DrawState::refused: return "refused";
        case cd::DrawState::stopped: return "stopped";
    }
    return "waiting";
}

}  // namespace

// ---- ModuleCheck ------------------------------------------------------------------------------------

struct ModuleCheck::State {
    std::vector<cp::Composed> composed;   // one module for each pass
    std::size_t pending = 0;
    bool started = false;
    cd::Verdict first;                    // the first error, by pass
    std::size_t first_pass = ~std::size_t(0);
};

ModuleCheck::~ModuleCheck() = default;   // a pending callback keeps the state it shares

cd::Verdict ModuleCheck::start(WGPUDevice device, const cs::Piece& piece, const cd::DrawChecks& checks) {
    state_ = std::make_shared<State>();
    // Check 1: the name, the version and each file's name.
    if (cd::Verdict v = cp::check_piece_name(piece.name, piece.version); !v.ok()) return v;
    for (const auto& [name, text] : piece.files)
        if (cd::Verdict v = cp::check_piece_file(piece, name); !v.ok()) return v;
    // Check 2: the manifest, with the limits of WebGPU.
    if (!checks.manifest || !checks.source || !checks.resolve)
        return refused(cp::kProgramNotBuilt, "The checks of pieces are not available.");
    if (cd::Verdict v = checks.manifest(piece); !v.ok()) return v;
    if (!piece.draw) return refused(cp::kProgramBadPass, "The piece " + piece.name + " has no draw half.", "draw");
    // Check 4: each file.
    for (const auto& [name, text] : piece.files)
        if (cd::Verdict v = checks.source(text, cs::piece_limits::kFileBytes); !v.ok()) {
            if (v.where.empty()) v.where = name;
            return v;
        }
    // Check 5: the imports (the hash was checked against the pin when the message was decoded).
    const cp::Resolved resolved = checks.resolve(piece);
    if (!resolved.verdict.ok()) return resolved.verdict;
    for (std::size_t k = 0; k < piece.draw->passes.size(); ++k) {
        cp::Composed c = cp::compose_draw(piece, resolved, k);
        if (!c.verdict.ok()) return c.verdict;
        state_->composed.push_back(std::move(c));
    }

    // Each module to WebGPU; its compilation messages come back on a later turn of the page.
    struct Ask {
        std::shared_ptr<State> state;
        std::size_t pass;
        WGPUShaderModule module;
    };
    state_->started = true;
    state_->pending = state_->composed.size();
    for (std::size_t k = 0; k < state_->composed.size(); ++k) {
        WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
        src.code = WGPUStringView{state_->composed[k].text.data(), state_->composed[k].text.size()};
        WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        md.nextInChain = &src.chain;
        const std::string label = piece.name + " draw.passes[" + std::to_string(k) + "] check";
        md.label = WGPUStringView{label.data(), label.size()};
        // The module's validation error goes to this scope, not to the device's uncaptured errors:
        // its compilation messages say what and where.
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &md);
        WGPUPopErrorScopeCallbackInfo pop = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
        pop.mode = WGPUCallbackMode_AllowSpontaneous;
        pop.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType, WGPUStringView, void*, void*) {};
        wgpuDevicePopErrorScope(device, pop);
        if (!module) {
            --state_->pending;
            if (k < state_->first_pass) {
                state_->first_pass = k;
                state_->first = refused(cp::kProgramBrowserRefused,
                                        "WebGPU made no module of draw pass " + std::to_string(k) + ".",
                                        "draw.passes[" + std::to_string(k) + "]");
            }
            continue;
        }
        WGPUCompilationInfoCallbackInfo cb = WGPU_COMPILATION_INFO_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowSpontaneous;
        cb.callback = [](WGPUCompilationInfoRequestStatus status, const WGPUCompilationInfo* info, void* u1, void*) {
            std::unique_ptr<Ask> ask(static_cast<Ask*>(u1));
            State& s = *ask->state;
            const std::size_t k = ask->pass;
            if (s.pending > 0) --s.pending;
            wgpuShaderModuleRelease(ask->module);
            if (k > s.first_pass) return;   // an earlier pass already refused the program
            if (status != WGPUCompilationInfoRequestStatus_Success || info == nullptr) {
                s.first_pass = k;
                s.first = refused(cp::kProgramBrowserRefused,
                                  "WebGPU gave no compilation result for draw pass " + std::to_string(k) + ".",
                                  "draw.passes[" + std::to_string(k) + "]");
                return;
            }
            for (std::size_t i = 0; i < info->messageCount; ++i) {
                const WGPUCompilationMessage& m = info->messages[i];
                if (m.type != WGPUCompilationMessageType_Error) continue;
                const cp::SourcePlace at =
                    cp::source_place(s.composed[k], std::uint32_t(m.lineNum), std::uint32_t(m.linePos));
                cd::Verdict v = refused(cp::kProgramBrowserRefused,
                                        "Draw pass " + std::to_string(k) + " does not compile in the browser, at " +
                                            at.part + " line " + std::to_string(at.line) + ", column " +
                                            std::to_string(at.column) + ": " + text_of(m.message),
                                        "draw.passes[" + std::to_string(k) + "]");
                v.line = at.line;
                v.column = at.column;
                s.first_pass = k;
                s.first = std::move(v);
                return;
            }
        };
        cb.userdata1 = new Ask{state_, k, module};
        wgpuShaderModuleGetCompilationInfo(module, cb);
    }
    return {};
}

bool ModuleCheck::done() const { return state_ && state_->started && state_->pending == 0; }

cd::Verdict ModuleCheck::verdict() const { return state_ ? state_->first : cd::Verdict{}; }

// ---- WebPrograms ------------------------------------------------------------------------------------

struct WebPrograms::Program {
    std::string id, panel;
    cs::Piece piece;
    cd::Instance instance;
    Json values = Json::object();     // the instance's params by name
    std::uint32_t pick_id = 0;
    std::uint64_t revision = 0;
    ModuleCheck check;
    bool checked = false;             // the browser check finished
    cd::Verdict refused;              // the decode, the browser check or a refused pipeline
    std::map<std::string, Version> versions;   // resource id: the newest version
    std::map<std::string, PublishedSize> published;   // the resources it keeps versions of
    std::uint64_t epoch = 0;          // a version at an earlier step than the newest starts a new epoch
    cd::DrawStatus shown;             // the scene view's last report
    bool reported = false;
};

class WebPrograms::PanelFeed final : public cd::DrawFeed {
public:
    PanelFeed(WebPrograms& owner, std::string panel) : owner_(owner), panel_(std::move(panel)) {}

    std::uint64_t revision() const override { return owner_.revision_; }

    std::vector<cd::PanelProgram> programs() const override {
        std::vector<cd::PanelProgram> out;
        for (const auto& [id, p] : owner_.programs_) {
            if (p->panel != panel_ || (!p->checked && p->refused.ok())) continue;   // still checking
            cd::PanelProgram pp;
            pp.piece = p->piece;
            pp.instance = p->instance;
            pp.pick_id = p->pick_id;
            pp.revision = p->revision;
            pp.refused = p->refused;
            out.push_back(std::move(pp));
        }
        return out;
    }

    void frame(cd::DrawRuntime& rt, cd::FrameInputs* in) override {
        auto it = owner_.programs_.find(rt.instance().program);
        if (it == owner_.programs_.end()) return;
        Program& p = *it->second;
        std::int64_t latest = -1;
        auto give = [&](const Program& from, const std::string& resource, const std::string& as) {
            const auto v = from.versions.find(resource);
            if (v == from.versions.end()) return;
            latest = std::max(latest, v->second.step);
            const auto version = std::make_pair(from.epoch, v->second.serial);
            if (rt.resource_step(as) == v->second.step && given_[p.id + "/" + as] == version) return;
            cd::ResourceVersion r;
            r.resource = as;
            r.step = v->second.step;
            r.serial = v->second.serial;
            r.count = v->second.count;
            r.width = v->second.width;
            r.height = v->second.height;
            r.format = v->second.format;
            r.elements = v->second.elements;
            r.bytes = v->second.bytes->data();
            r.size = v->second.bytes->size();
            if (rt.set_resource(r).ok()) given_[p.id + "/" + as] = version;
        };
        std::uint64_t epoch = p.epoch;
        for (const cs::Resource& res : p.piece.resources)
            if (res.publish) give(p, res.id, res.id);
        for (const cs::PieceInput& input : p.piece.inputs) {
            if (input.kind != cs::InputKind::buffer) continue;
            const auto b = p.instance.bind.find(input.name);
            const std::string prefix = "program:";
            if (b == p.instance.bind.end() || b->second.compare(0, prefix.size(), prefix) != 0) continue;
            const std::string rest = b->second.substr(prefix.size());
            const std::size_t slash = rest.find('/');
            if (slash == std::string::npos) continue;
            const auto from = owner_.programs_.find(rest.substr(0, slash));
            if (from == owner_.programs_.end()) continue;
            give(*from->second, rest.substr(slash + 1), input.name);
            epoch = std::max(epoch, from->second->epoch);
        }
        in->params = pack_params(p.piece, p.values);
        in->selection = owner_.selection_;
        // The module keeps no history: the newest version is the cursor's, and the view is live.
        in->cursor_step = latest;
        in->latest_step = latest;
        in->live = true;
        in->cursor_moved = false;
        in->epoch = epoch;
    }

    void status(const cd::DrawRuntime& rt) override {
        auto it = owner_.programs_.find(rt.instance().program);
        if (it == owner_.programs_.end()) return;
        it->second->shown = rt.status();
        it->second->reported = true;
    }

private:
    WebPrograms& owner_;
    std::string panel_;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> given_;
};

WebPrograms::WebPrograms(WGPUDevice device) : device_(device) {}

std::uint32_t WebPrograms::pick_id_of(const std::string& program) const {
    const auto it = programs_.find(program);
    return it == programs_.end() ? 0u : it->second->pick_id;
}
WebPrograms::~WebPrograms() = default;

void WebPrograms::on_program(const caliper_net::wire::Program& m) {
    if (m.op == CALIPER_NET_PROGRAM_REMOVE) {
        if (programs_.erase(m.id)) ++revision_;
        return;
    }
    if (m.op == CALIPER_NET_PROGRAM_PARAMS) {
        auto it = programs_.find(m.id);
        if (it != programs_.end() && m.params.is_object())
            for (const auto& [name, v] : m.params.members()) it->second->values.set(name, v);
        return;
    }
    if (m.op != CALIPER_NET_PROGRAM_SET || !m.piece.is_object()) return;   // the one-pass form draws no piece
    if (!takes_program(programs_, m.id)) return;   // past the most programs of one producer
    auto p = std::make_unique<Program>();
    p->id = m.id;
    p->panel = m.panel.value_or("");
    if (m.params.is_object()) p->values = m.params;
    // A program set again keeps its pick id and its resources' newest versions.
    if (auto old = programs_.find(m.id); old != programs_.end()) {
        p->pick_id = old->second->pick_id;
        p->versions = std::move(old->second->versions);
        p->epoch = old->second->epoch + 1;
    } else {
        p->pick_id = next_pick_++;
    }
    p->revision = ++revision_;
    p->refused = decode_program(m, &p->piece, &p->instance);
    if (p->refused.ok()) p->published = published_sizes(p->piece);
    // The kept versions of a program set again: only those that its new piece publishes, within their sizes.
    for (auto v = p->versions.begin(); v != p->versions.end();) {
        const auto d = p->published.find(v->first);
        const bool fits = d != p->published.end() && v->second.count <= d->second.count &&
                          v->second.bytes->size() <= d->second.bytes;
        v = fits ? std::next(v) : p->versions.erase(v);
    }
    if (p->refused.ok()) p->refused = p->check.start(device_, p->piece);
    if (!p->refused.ok()) p->checked = true;
    programs_[m.id] = std::move(p);
}

void WebPrograms::on_resource(const caliper_net::wire::Resource& part) {
    // The parts of a version in parts are put together first: a version is shown whole or not at all.
    caliper_net::wire::WholeResource whole;
    std::string err;
    if (parts_.add(part, &whole, &err) != caliper_net::wire::SplitResources::Result::Whole) return;
    const caliper_net::wire::Resource& r = whole.resource;
    auto it = programs_.find(r.program);
    if (it == programs_.end()) return;   // a resource before its program: the next version follows it
    Program& p = *it->second;
    if (!keeps_version(p.published, r)) return;   // no resource that the piece publishes, or above its size
    // All kept versions stay within kKeptBytesMax, as the app's step history keeps its resources: a version
    // that would pass it is not kept, and the program draws the version it has.
    std::uint64_t kept = 0;
    for (const auto& [id, q] : programs_)
        for (const auto& [rid, v] : q->versions)
            if (v.bytes && !(q.get() == &p && rid == r.resource)) kept += v.bytes->size();
    if (kept + r.data.size > kKeptBytesMax) return;
    Version& v = p.versions[r.resource];
    const std::int64_t step = std::int64_t(std::min<std::uint64_t>(r.step, std::uint64_t(INT64_MAX)));
    if (v.step >= 0 && step < v.step) ++p.epoch;   // the run went back (a new attachment or a fork)
    v.step = step;
    v.serial = ++resource_serial_;
    v.count = r.count;
    v.width = r.width.value_or(0);
    v.height = r.height.value_or(0);
    v.format = r.format.value_or(std::string());
    v.elements = r.elements;
    v.bytes = whole.keep ? whole.keep
                         : std::make_shared<const std::vector<std::uint8_t>>(r.data.data, r.data.data + r.data.size);
}

void WebPrograms::pump(const std::vector<std::string>& device_errors) {
    for (auto& [id, p] : programs_) {
        if (p->checked || !p->check.done()) continue;
        p->checked = true;
        p->refused = p->check.verdict();
        ++revision_;
    }
    if (device_errors.size() < errors_seen_) errors_seen_ = 0;   // the list is capped and may restart
    for (std::size_t i = errors_seen_; i < device_errors.size(); ++i)
        for (auto& [id, p] : programs_) {
            if (!p->refused.ok()) continue;
            cd::Verdict v = refusal_of_error(id, device_errors[i]);
            if (v.ok()) continue;
            p->refused = std::move(v);
            p->checked = true;
            ++revision_;
        }
    errors_seen_ = device_errors.size();
}

cd::DrawFeed* WebPrograms::feed(const std::string& panel) {
    std::unique_ptr<PanelFeed>& f = feeds_[panel];
    if (!f) f = std::make_unique<PanelFeed>(*this, panel);
    return f.get();
}

std::vector<std::string> WebPrograms::panels() const {
    std::vector<std::string> out;
    for (const auto& [id, p] : programs_)
        if (!p->panel.empty() && std::find(out.begin(), out.end(), p->panel) == out.end()) out.push_back(p->panel);
    return out;
}

std::string WebPrograms::status_json() const {
    Json out = Json::array();
    for (const auto& [id, p] : programs_) {
        Json j = Json::object();
        j.set("id", id);
        j.set("panel", p->panel);
        const cd::Verdict& err = !p->refused.ok() ? p->refused : p->shown.error;
        std::string state = !p->refused.ok() ? "refused" : !p->checked ? "checking"
                                                       : p->reported ? state_name(p->shown.state)
                                                                     : "waiting";
        j.set("state", state);
        j.set("code", err.code);
        j.set("message", err.message);
        j.set("where", err.where);
        j.set("line", std::uint64_t(err.line));
        j.set("column", std::uint64_t(err.column));
        j.set("warning_code", p->shown.warning.code);
        j.set("warning", p->shown.warning.message);
        std::int64_t step = -1;
        for (const auto& [rid, v] : p->versions) step = std::max(step, v.step);
        j.set("step", step);
        j.set("frames", std::uint64_t(p->shown.frames));
        out.push(std::move(j));
    }
    return out.dump();
}

}  // namespace caliper_web
