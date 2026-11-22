// The app's host of stages (spec_stages.h, spec_stage_host.h, design/gpu-programs.md).
#include "spec_runtime/spec_stages.h"
#include "spec_runtime/spec_stage_host.h"

#include "gfx/shaders/generated/programs.h"
#include "gfx/shaders/generated/scene_lines_wgsl.h"
#include "gfx/shaders/generated/scene_points_wgsl.h"

#if defined(CALIPER_HAS_WGSL)
#include <caliper/wgsl.h>
#endif

#include <algorithm>
#include <array>
#include <utility>

namespace caliper_host {

namespace cb = caliper_builder;
namespace pg = caliper_builder::programs;
namespace spec = caliper_builder::spec;
using caliper_scene::Primitive;
using caliper_scene::StageBuild;

namespace {

uint32_t line_count(const std::string& s) {
    if (s.empty()) return 0;
    return uint32_t(std::count(s.begin(), s.end(), '\n') + (s.back() == '\n' ? 0 : 1));
}

[[maybe_unused]] uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

pg::Verdict refusal(const char* code, std::string message, uint32_t line = 0, uint32_t column = 0) {
    pg::Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.line = line;
    v.column = column;
    return v;
}

std::string entity_key(std::string_view path, Primitive p) {
    std::string k(path);
    k += '\0';
    k += char('0' + int(p));
    return k;
}

constexpr Primitive kStagePrimitives[] = {Primitive::Points, Primitive::Lines};

}  // namespace

// ---- one stage program

std::string merge_msl(const std::vector<std::string>& texts) {
    if (texts.empty()) return std::string();
    size_t n = 0;   // the common leading lines
    for (;;) {
        const size_t e = texts[0].find('\n', n);
        if (e == std::string::npos) break;
        bool same = true;
        for (size_t i = 1; i < texts.size() && same; ++i) same = texts[i].compare(0, e + 1, texts[0], 0, e + 1) == 0;
        if (!same) break;
        n = e + 1;
    }
    std::string out = texts[0];
    for (size_t i = 1; i < texts.size(); ++i) out.append(texts[i], n, std::string::npos);
    return out;
}

StageCompile compile_stage_program(Primitive primitive, const std::string* material, const std::string* vertex,
                                   const std::string& label) {
    StageCompile out;
    if (!material && !vertex) {
        out.verdict = refusal(pg::kStageCompileFailed, "A stage program needs a material or a vertex stage.");
        return out;
    }
    const bool points = primitive == Primitive::Points;
    if (!points && primitive != Primitive::Lines) {
        out.verdict = refusal(pg::kStageCompileFailed, "Only points and lines take stages.");
        return out;
    }
    // The stage sources come right after the prelude, thus a compiler line maps back to them.
    struct Range { int which; uint32_t first, lines; };
    std::vector<Range> ranges;
    std::string src(pg::stage_prelude());
    uint32_t next = pg::stage_prelude_lines() + 1;
    for (int which = 0; which < 2; ++which) {
        const std::string* s = which == 0 ? material : vertex;
        if (!s || s->empty()) continue;
        ranges.push_back({which, next, line_count(*s)});
        src += *s;
        if (s->back() != '\n') src += '\n';
        next += line_count(*s);
    }
    src += '\n';
    src += points ? caliper_gfx::shaders::kScenePointsWgsl : caliper_gfx::shaders::kSceneLinesWgsl;
    src += '\n';
    src += caliper_scene::stage_entries(primitive, material != nullptr, vertex != nullptr);

#if defined(CALIPER_HAS_WGSL)
    auto place = [&](uint32_t line, uint32_t column, std::string message) {
        for (const Range& r : ranges)
            if (line >= r.first && line < r.first + r.lines) {
                out.culprit = r.which;
                out.verdict = refusal(pg::kStageCompileFailed, std::move(message), line - r.first + 1, column);
                return;
            }
        out.verdict = refusal(pg::kStageCompileFailed,
                              "The viewer's program with this stage does not compile: " + message);
    };
    if (caliper_wgsl_abi_version() != CALIPER_WGSL_ABI_VERSION) {
        out.verdict = refusal(pg::kProgramCompilerUnavailable, "The WGSL compiler library has another interface version.");
        return out;
    }
    caliper_wgsl_module* module = nullptr;
    caliper_wgsl_error error{};
    if (caliper_wgsl_module_create(src.data(), src.size(), nullptr, &module, &error) != CALIPER_WGSL_OK) {
        place(error.line, error.column, error.message ? error.message : "the source does not compile");
        caliper_wgsl_error_free(&error);
        return out;
    }
    // The stage program binds what the built-in program binds, with its own stage bits and indices.
    const caliper_gfx::shaders::ProgramId builtin =
        points ? caliper_gfx::shaders::ProgramId::scene_points : caliper_gfx::shaders::ProgramId::scene_lines;
    const caliper_gfx::shaders::ProgramDescription& bd = caliper_gfx::shaders::kProgramDescriptions[uint32_t(builtin)];
    std::vector<caliper_gfx::BindingDesc> bindings(bd.bindings, bd.bindings + bd.binding_count);
    const size_t count = caliper_wgsl_module_binding_count(module);
    size_t matched = 0;
    for (size_t i = 0; i < count; ++i) {
        caliper_wgsl_binding b{};
        if (caliper_wgsl_module_binding(module, i, &b) != CALIPER_WGSL_OK) continue;
        for (caliper_gfx::BindingDesc& d : bindings)
            if (d.group == b.group && d.binding == b.binding) {
                d.stages = uint8_t(b.stages & 3u);
                d.msl_index = uint8_t(b.metal_index);
                ++matched;
            }
    }
    if (matched != count || count != bindings.size()) {
        caliper_wgsl_module_destroy(module);
        out.verdict = refusal(pg::kStageCompileFailed, "The program with this stage binds other resources than the "
                                                       "viewer's built-in program.");
        return out;
    }
    std::vector<std::string> msl;
    const std::pair<const char*, caliper_wgsl_stage> entries[3] = {
        {caliper_scene::kStageVertexEntry, CALIPER_WGSL_STAGE_VERTEX},
        {caliper_scene::kStageFragmentEntry, CALIPER_WGSL_STAGE_FRAGMENT},
        {caliper_scene::kStagePickEntry, CALIPER_WGSL_STAGE_FRAGMENT},
    };
    for (const auto& [name, stage] : entries) {
        caliper_wgsl_buffer text{};
        if (caliper_wgsl_translate_msl(module, name, stage, &text, &error) != CALIPER_WGSL_OK) {
            place(error.line, error.column,
                  std::string("MSL of ") + name + ": " + (error.message ? error.message : "the translation failed"));
            caliper_wgsl_error_free(&error);
            caliper_wgsl_module_destroy(module);
            return out;
        }
        msl.emplace_back(reinterpret_cast<const char*>(text.data), text.size);
        caliper_wgsl_buffer_free(&text);
    }
    caliper_wgsl_module_destroy(module);

    auto build = std::make_shared<StageBuild>();
    build->primitive = primitive;
    build->builtin = builtin;
    build->label = label;
    build->msl = merge_msl(msl);
    build->wgsl = std::move(src);
    build->bindings = std::move(bindings);
    build->material = material != nullptr;
    build->vertex = vertex != nullptr;
    for (const std::string* s : {material, vertex})
        if (s && (s->find("time_s") != std::string::npos || s->find("step") != std::string::npos)) build->animated = true;
    build->source_hash = fnv1a(build->wgsl);
    out.build = std::move(build);
    return out;
#else
    (void)label;
    out.verdict = refusal(pg::kProgramCompilerUnavailable,
                          "This build has no WGSL compiler, so the viewer cannot compile stages.");
    return out;
#endif
}

// ---- the time watchdog

StageWatchdog::Action StageWatchdog::frames(double gpu_ms, uint32_t count, bool staged) {
    if (count == 0) return Action::none;
    if (!staged) {
        baseline_ms_ = gpu_ms;
        if (!probing_) return Action::none;
        probing_ = false;
        if (last_ms_ > baseline_ms_) return Action::stop;
        over_ = 0;   // the panel is as slow without its stages
        return Action::none;
    }
    if (probing_) return Action::none;   // frames from before the probe
    if (gpu_ms <= kLimitMs) {
        over_ = 0;
        return Action::none;
    }
    over_ += count;
    last_ms_ = gpu_ms;
    if (over_ < kFrames) return Action::none;
    probing_ = true;   // one frame without the stages gives the panel's own time
    return Action::probe;
}

// ---- the host

struct AppStageHost::Record {
    spec::Stage stage;
    std::string key;              // what the checks saw: the source, the panel's kind, the review
    cb::ProgramStatus status;
    bool usable = false;          // passed every check
    bool stopped = false;         // the watchdog stopped it
    bool composed = false;        // refused when composed with its entity's other stage: checked again
};

class AppStageHost::Panel final : public caliper_scene::StageSource {
public:
    Panel(AppStageHost& host, std::string id) : host_(host), id_(std::move(id)) {}

    std::shared_ptr<const StageBuild> build(std::string_view path, Primitive p) const override {
        if (off_ || dog_.probing()) return nullptr;
        const auto it = builds_.find(entity_key(path, p));
        return it == builds_.end() ? nullptr : it->second;
    }
    uint64_t revision() const override { return revision_; }
    float time_s() const override {
        return std::chrono::duration<float>(std::chrono::steady_clock::now() - origin_).count();
    }
    void refused(const StageBuild& build, const std::string& message) override;
    void frames_done(double gpu_ms, uint32_t frames, bool staged) override;

    // The builds of the panel's entities, by entity_key(); the sources each entity was built from.
    std::map<std::string, std::shared_ptr<const StageBuild>> builds_;
    std::map<std::string, std::string> sources_;
    uint64_t revision_ = 1;
    bool off_ = false;
    StageWatchdog dog_;
    std::chrono::steady_clock::time_point origin_ = std::chrono::steady_clock::now();

private:
    AppStageHost& host_;
    std::string id_;
};

void AppStageHost::Panel::refused(const StageBuild& build, const std::string& message) {
    std::string entity;
    for (auto it = builds_.begin(); it != builds_.end(); ++it)
        if (it->second.get() == &build) entity = it->first.substr(0, it->first.find('\0'));
    std::vector<cb::ProgramStatus> refused;
    {
        std::lock_guard<std::mutex> lock(host_.mutex_);
        for (Record& r : host_.records_) {
            if (r.stage.panel != id_ || r.stage.entity != entity || !r.usable) continue;
            r.usable = false;
            r.status.state = "refused";
            r.status.error = cb::ProgramError{pg::kStageCompileFailed,
                                              "The graphics layer refused the stage program: " + message, 0, 0};
            refused.push_back(r.status);
        }
    }
    for (auto it = builds_.begin(); it != builds_.end();)
        it = it->first.compare(0, entity.size() + 1, entity + '\0') == 0 ? builds_.erase(it) : std::next(it);
    ++revision_;
    for (const cb::ProgramStatus& s : refused) host_.push(s, "refused");
}

void AppStageHost::Panel::frames_done(double gpu_ms, uint32_t frames, bool staged) {
    if (builds_.empty() || off_) return;
    std::vector<cb::ProgramStatus> stopped;
    {
        std::lock_guard<std::mutex> lock(host_.mutex_);
        if (staged)
            for (Record& r : host_.records_)
                if (r.stage.panel == id_ && r.usable && !r.stopped) {
                    r.status.runs += frames;
                    r.status.last_pass_ms = gpu_ms;
                    r.status.state = "running";
                }
        const bool probing = dog_.probing();
        const StageWatchdog::Action a = dog_.frames(gpu_ms, frames, staged);
        if (dog_.probing() != probing) ++revision_;
        if (a == StageWatchdog::Action::stop) {
            off_ = true;
            ++revision_;
            for (Record& r : host_.records_)
                if (r.stage.panel == id_ && r.usable && !r.stopped) {
                    r.stopped = true;
                    r.status.state = "stopped";
                    r.status.error = cb::ProgramError{
                        pg::kStageOverTime,
                        "The panel took more than 16.7 ms of GPU time a frame with its stages for 30 frames in a row, "
                        "more than without them, so it draws without its stages.",
                        0, 0};
                    stopped.push_back(r.status);
                }
        }
    }
    for (const cb::ProgramStatus& s : stopped) host_.push(s, "stopped");
}

StageChecks StageChecks::sandbox() {
    StageChecks c;
    c.declaration = &pg::check_stage_declaration;
    c.provenance = &pg::check_provenance;
    c.source = &pg::check_source;
    c.compile = &pg::compile_and_reflect;
    c.stage = &pg::check_stage;
    return c;
}

AppStageHost::AppStageHost(StageChecks checks) : checks_(checks) {}
AppStageHost::~AppStageHost() = default;

void AppStageHost::set_events(cb::EventSink* sink) { events_ = sink; }

void AppStageHost::push(const cb::ProgramStatus& status, const char* change) {
    if (events_) cb::push_program_event(*events_, status, change);
}

void AppStageHost::reconcile(const spec::Spec& spec) {
    std::vector<Record> next;
    std::vector<std::pair<size_t, const char*>> events;   // a record and its change
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Steps 1 to 6 of the sandbox for each stage whose source, panel or review changed.
        for (const spec::Stage& st : spec.stages) {
            Record r;
            r.stage = st;
            const std::string sha = pg::sha256_hex(st.wgsl);
            const spec::Panel* panel = spec.find_panel(st.panel);
            const bool scene = panel && panel->primitive == spec::Primitive::scene;
            r.key = st.wgsl + '\0' + (scene ? "s" : "-") + '\0' +
                    (spec.provenance.foreign ? checks_.provenance(spec.provenance, spec::stage_target(st), sha).code
                                             : std::string());
            const std::string id = spec::stage_id(st);
            auto prev = std::find_if(records_.begin(), records_.end(),
                                     [&](const Record& p) { return p.status.id == id; });
            if (prev != records_.end() && prev->key == r.key && !prev->composed) {
                r.status = prev->status;
                r.usable = prev->usable;
                r.stopped = prev->stopped;
                next.push_back(std::move(r));
                continue;
            }
            r.status.id = id;
            r.status.kind = spec::stage_kind_name(st.kind);
            r.status.host = "viewer";
            r.status.panel = st.panel;
            r.status.sha256 = sha;
            pg::Verdict v = checks_.declaration(st);
            if (v.ok() && !scene)
                v = refusal(pg::kStageNotScenePanel, "The panel " + st.panel + " is not a scene panel.");
            if (v.ok()) v = checks_.provenance(spec.provenance, spec::stage_target(st), sha);
            if (v.ok()) v = checks_.source(st.wgsl, spec::program_limits::kStageSourceBytes);
            if (v.ok()) {
                pg::Compiled c = checks_.compile(std::string(pg::stage_prelude()) + st.wgsl);
                v = c.verdict;
                if (v.ok()) v = checks_.stage(st, c.reflection);
                if (v.code == pg::kProgramCompileFailed) v.code = pg::kStageCompileFailed;
                v.line = v.line > pg::stage_prelude_lines() ? v.line - pg::stage_prelude_lines() : 0;
                if (v.line == 0) v.column = 0;
            }
            if (v.code == pg::kProgramNeedsReview) {
                r.status.state = "review";
            } else if (!v.ok()) {
                r.status.state = "refused";
                r.status.error = cb::ProgramError{v.code, v.message, v.line, v.column};
                events.emplace_back(next.size(), "refused");
            } else {
                r.status.state = "compiled";
                r.usable = true;
                events.emplace_back(next.size(), "compiled");
            }
            next.push_back(std::move(r));
        }

        // The programs of each entity with a usable stage, for points and lines. A refusal in one
        // stage's source refuses that stage, and the entity is composed again without it.
        std::map<std::string, std::map<std::string, std::shared_ptr<const StageBuild>>> builds;   // panel, key
        std::map<std::string, std::map<std::string, std::string>> sources;                       // panel, entity
        std::map<std::pair<std::string, std::string>, std::array<int, 2>> entities;               // material, vertex
        for (size_t i = 0; i < next.size(); ++i) {
            if (!next[i].usable || next[i].stopped) continue;
            auto& e = entities.try_emplace({next[i].stage.panel, next[i].stage.entity}, std::array<int, 2>{-1, -1})
                          .first->second;
            e[next[i].stage.kind == spec::StageKind::vertex ? 1 : 0] = int(i);
        }
        for (auto& item : entities) {
            const std::string& panel_id = item.first.first;
            const std::string& entity = item.first.second;
            std::array<int, 2>& idx = item.second;
            for (int attempt = 0; attempt < 3; ++attempt) {
                const std::string* mat = idx[0] >= 0 ? &next[size_t(idx[0])].stage.wgsl : nullptr;
                const std::string* vtx = idx[1] >= 0 ? &next[size_t(idx[1])].stage.wgsl : nullptr;
                if (!mat && !vtx) break;
                const std::string src_key = (mat ? *mat : std::string()) + '\0' + (vtx ? *vtx : std::string());
                // An entity whose sources did not change keeps its builds.
                const auto pit = panels_.find(panel_id);
                if (pit != panels_.end()) {
                    const auto sit = pit->second->sources_.find(entity);
                    if (sit != pit->second->sources_.end() && sit->second == src_key) {
                        for (Primitive p : kStagePrimitives) {
                            const auto bit = pit->second->builds_.find(entity_key(entity, p));
                            if (bit != pit->second->builds_.end()) builds[panel_id][bit->first] = bit->second;
                        }
                        sources[panel_id][entity] = src_key;
                        break;
                    }
                }
                std::map<std::string, std::shared_ptr<const StageBuild>> made;
                StageCompile failed;
                for (Primitive p : kStagePrimitives) {
                    StageCompile c = compile_stage_program(
                        p, mat, vtx, "stage " + panel_id + "/" + entity + " " + caliper_scene::primitive_name(p));
                    if (!c.build) {
                        failed = std::move(c);
                        break;
                    }
                    made[entity_key(entity, p)] = std::move(c.build);
                }
                if (!failed.verdict.code.empty()) {
                    // The stage the line is in, else every stage of the entity.
                    for (int k = 0; k < 2; ++k) {
                        if (idx[k] < 0 || (failed.culprit >= 0 && failed.culprit != k)) continue;
                        Record& r = next[size_t(idx[k])];
                        r.usable = false;
                        r.composed = true;
                        r.status.state = "refused";
                        r.status.error = cb::ProgramError{failed.verdict.code, failed.verdict.message,
                                                          failed.verdict.line, failed.verdict.column};
                        auto ev = std::find_if(events.begin(), events.end(),
                                               [&](const auto& x) { return x.first == size_t(idx[k]); });
                        if (ev != events.end()) ev->second = "refused";
                        else events.emplace_back(size_t(idx[k]), "refused");
                        idx[k] = -1;
                    }
                    continue;
                }
                for (auto& m : made) builds[panel_id][m.first] = std::move(m.second);
                sources[panel_id][entity] = src_key;
                // A stage whose entity is built again compiles again.
                for (int k = 0; k < 2; ++k)
                    if (idx[k] >= 0 && std::none_of(events.begin(), events.end(),
                                                    [&](const auto& x) { return x.first == size_t(idx[k]); })) {
                        next[size_t(idx[k])].status.state = "compiled";
                        events.emplace_back(size_t(idx[k]), "compiled");
                    }
                break;
            }
        }

        // Each panel's source takes its builds; a panel whose builds changed starts again.
        for (auto& [panel_id, made] : builds)
            if (!panels_.count(panel_id)) panels_[panel_id] = std::make_unique<Panel>(*this, panel_id);
        for (auto& [panel_id, panel] : panels_) {
            auto& mine = builds[panel_id];
            if (mine == panel->builds_ && sources[panel_id] == panel->sources_) continue;
            panel->builds_ = std::move(mine);
            panel->sources_ = std::move(sources[panel_id]);
            panel->off_ = false;
            panel->dog_.reset();
            panel->origin_ = std::chrono::steady_clock::now();
            ++panel->revision_;
        }
        records_ = std::move(next);
    }
    std::vector<cb::ProgramStatus> pushed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [i, change] : events) pushed.push_back(records_[i].status);
    }
    for (size_t k = 0; k < pushed.size(); ++k) push(pushed[k], events[k].second);
}

std::vector<cb::ProgramStatus> AppStageHost::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<cb::ProgramStatus> out;
    out.reserve(records_.size());
    for (const Record& r : records_) out.push_back(r.status);
    return out;
}

caliper_scene::StageSource* AppStageHost::source(const std::string& panel) {
    const auto it = panels_.find(panel);
    return it == panels_.end() || it->second->builds_.empty() ? nullptr : it->second.get();
}

std::string AppStageHost::message(const std::string& panel) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Record& r : records_) {
        if (r.stage.panel != panel || !r.status.error) continue;
        const cb::ProgramError& e = *r.status.error;
        std::string text = "Stage " + r.status.kind + " on " + r.stage.entity + ": " + e.code;
        if (e.line > 0) text += " at line " + std::to_string(e.line) + ", column " + std::to_string(e.column);
        return text + ": " + e.message;
    }
    return std::string();
}

AppStageHost* app_stage_host_impl() {
    static AppStageHost* host = new AppStageHost();   // lives until the process ends
    return host;
}

caliper_builder::StageHost* app_stage_host() { return app_stage_host_impl(); }

}  // namespace caliper_host
