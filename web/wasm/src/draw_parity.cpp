// draw_parity.cpp: the browser's side of the pixel comparison (draw_parity.h).
#include "draw_parity.h"

#include "../../../tests/draw_parity/parity_draw.h"
#include "program_frames.h"
#include "web_programs.h"

#include <map>
#include <memory>

namespace caliper_web {

namespace {

namespace cd = caliper_draw;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace wire = caliper_net::wire;

struct Entry {
    dp::ProgramPlan plan;
    cs::Piece piece;
    cd::Instance instance;
    caliper_net::Json values;
    ModuleCheck check;
    std::unique_ptr<cd::DrawRuntime> runtime;
    std::unique_ptr<dp::Target> target;
    caliper_gfx::Readback color_rb, id_rb;
    bool color_in = false, id_in = false;
    enum class Stage { checking, reading, done } stage = Stage::checking;
    ParityResult result;
};

struct Run {
    std::vector<std::uint8_t> frames;   // the RESOURCE sections point into it
    dp::Plan plan;
    std::map<std::string, wire::Program> programs;
    std::map<std::string, std::vector<wire::Resource>> resources;
    std::vector<std::unique_ptr<Entry>> entries;
    std::vector<ParityResult> results;
    bool running = false;
};
Run& run() {
    static Run r;
    return r;
}

void finish(Entry& e, const cd::Verdict& v) {
    e.result.code = v.code;
    e.result.message = v.message;
    e.result.where = v.where;
    e.result.line = v.line;
    e.result.column = v.column;
    e.stage = Entry::Stage::done;
}

// Installs the program, gives it its resources, draws the plan's frames and asks for both targets.
void draw(caliper_gfx::Backend& b, const Run& r, Entry& e) {
    e.runtime = std::make_unique<cd::DrawRuntime>(b, true);
    const cd::Verdict v = e.runtime->install(e.piece, e.instance, e.plan.pick_id);
    if (!v.ok()) return finish(e, v);
    const auto res = r.resources.find(e.plan.id);
    if (res != r.resources.end())
        for (const wire::Resource& x : res->second)
            if (const cd::Verdict s = e.runtime->set_resource(resource_version(x)); !s.ok()) return finish(e, s);
    e.target = std::make_unique<dp::Target>(b, r.plan.size);
    if (!e.target->ok) {
        cd::Verdict t;
        t.code = "parity.target";
        t.message = "the host target was refused: " + b.last_error();
        return finish(e, t);
    }
    std::string why;
    for (int i = 0; i < r.plan.frames; ++i) {
        std::vector<caliper_gfx::Serial> serials;
        if (!e.target->frame(*e.runtime, dp::frame_inputs(e.plan, r.plan, i, pack_params(e.piece, e.values)), &serials,
                             &why)) {
            cd::Verdict f = e.runtime->status().error;
            if (f.ok()) {
                f.code = "parity.frame";
                f.message = why;
            }
            return finish(e, f);
        }
    }
    e.color_rb = b.read_texture(e.target->host.color, 0, 0, r.plan.size, r.plan.size);
    e.id_rb = b.read_texture(e.target->host.id, 0, 0, r.plan.size, r.plan.size);
    if (!e.color_rb || !e.id_rb) {
        cd::Verdict f;
        f.code = "parity.readback";
        f.message = "no readback: " + b.last_error();
        return finish(e, f);
    }
    e.stage = Entry::Stage::reading;
}

}  // namespace

bool draw_parity_start(caliper_gfx::Backend& b, WGPUDevice device, const std::uint8_t* frames, std::size_t n,
                       const std::string& plan_json, std::string* why) {
    Run& r = run();
    if (r.running) return (*why = "a comparison is running", false);
    draw_parity_stop();
    r.frames.assign(frames, frames + n);
    if (!dp::read_plan(plan_json, &r.plan, why)) return false;
    std::vector<std::pair<const std::uint8_t*, std::size_t>> list;
    if (!dp::split_frames(r.frames.data(), r.frames.size(), &list, why)) return false;
    for (const auto& [data, len] : list) {
        wire::Message m;
        const wire::DecodeResult d = wire::decode(data, len, &m);
        if (!d.ok()) return (*why = "a frame does not decode: " + d.reason + d.error_code + " " + d.message, false);
        if (auto* p = std::get_if<wire::Program>(&m)) r.programs[p->id] = *p;
        else if (auto* x = std::get_if<wire::Resource>(&m)) r.resources[x->program].push_back(*x);
        else return (*why = "the fixture holds a frame that is neither PROGRAM nor RESOURCE", false);
    }
    for (const dp::ProgramPlan& pp : r.plan.programs) {
        auto e = std::make_unique<Entry>();
        e->plan = pp;
        e->result.id = pp.id;
        const auto pm = r.programs.find(pp.id);
        if (pm == r.programs.end()) return (*why = "no PROGRAM of " + pp.id + " in the fixture", false);
        e->values = pm->second.params;
        cd::Verdict v = decode_program(pm->second, &e->piece, &e->instance);
        if (v.ok()) v = e->check.start(device, e->piece);
        if (!v.ok()) finish(*e, v);
        r.entries.push_back(std::move(e));
    }
    (void)b;
    r.running = true;
    return true;
}

bool draw_parity_poll(caliper_gfx::Backend& b, const std::vector<std::string>& device_errors, std::string* report) {
    Run& r = run();
    if (!r.running) return false;
    bool all = true;
    for (auto& ep : r.entries) {
        Entry& e = *ep;
        if (e.stage == Entry::Stage::checking && e.check.done()) {
            const cd::Verdict v = e.check.verdict();
            if (!v.ok()) finish(e, v);
            else draw(b, r, e);
        }
        if (e.stage == Entry::Stage::reading) {
            if (!e.color_in) e.color_in = b.take_readback(e.color_rb, &e.result.color);
            if (!e.id_in) e.id_in = b.take_readback(e.id_rb, &e.result.ids);
            if (e.color_in && e.id_in) e.stage = Entry::Stage::done;
        }
        all = all && e.stage == Entry::Stage::done;
    }
    if (!all) return false;
    // A module or pipeline that WebGPU refused after its check reaches the backend's error list.
    for (auto& ep : r.entries)
        for (const std::string& err : device_errors) {
            if (!ep->result.code.empty()) break;
            const cd::Verdict v = refusal_of_error(ep->plan.id, err);
            if (!v.ok()) finish(*ep, v);
        }
    bool ok = true;
    std::string list;
    r.results.clear();
    for (auto& ep : r.entries) {
        const Entry& e = *ep;
        const bool as_planned = e.result.code == e.plan.expect;
        ok = ok && as_planned;
        list += " " + e.plan.id + "=" + (e.result.code.empty() ? std::string("drawn") : e.result.code);
        r.results.push_back(e.result);
    }
    *report = std::string(ok ? "ok" : "fail") + list;
    r.running = false;
    return true;
}

bool draw_parity_running() { return run().running; }

const std::vector<ParityResult>& draw_parity_results() { return run().results; }

void draw_parity_stop() {
    Run& r = run();
    r.running = false;
    r.entries.clear();   // the runtimes and the targets, before the backend goes
    r.programs.clear();
    r.resources.clear();
    r.frames.clear();
}

}  // namespace caliper_web
