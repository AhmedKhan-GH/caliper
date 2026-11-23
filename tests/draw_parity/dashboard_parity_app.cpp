// caliper_dashboard_parity_app: the app's side of the pixel comparison of a whole dashboard
// between the app and the browser's spec page (web/test/draw_parity_dashboard.spec.ts).
//
//   caliper_dashboard_parity_app <fixture> <pieces> <out> [--size WxH]
//
// The fixture is a recording of a short seeded run of the Attention Arc Tower
// (tests/draw_parity/dashboard/): the saved spec (spec.toml, spec.json) and the panel.data
// answer of every leaf of its layout (panels/<id>.json), as the app gave them to a viewer:
// the series as their lines, the scene panels' instances of pieces with the newest version of
// each published resource, and the control's state. <pieces> holds the workspace pieces that
// the spec pins (demos/arc_tower). The tool draws the spec with the app's spec runtime into an
// offscreen canvas on Metal, from that data only: the series through a panel source that gives
// the recorded lines as the window sources' tensors, the programs through the app's draw host
// with the recorded resources in a step history. No script runs and nothing trains.
//
// It draws frames until every program's draw half has drawn the recorded step, then two more,
// and writes <out>/app/dashboard.rgba (RGBA8, top row first), dashboard.png and report.json
// (the size, the step, the frames drawn, the layout's rectangles, the tab strips, each
// program's state, and each scene panel's rectangle and the labels it drew with their boxes). The browser draws the same answers, served as panel.data, at the same size.
// Without a Metal device it writes <out>/skipped.txt and exits 77, which ctest counts as
// skipped. Exit 1 on anything else that fails.
#include "spec_runtime/spec_layout.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"
#include "spec_runtime/spec_stage_host.h"

#include "export_service.h"
#include "live/live_views_scene.h"   // DrawnLabel
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/format.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace spec = caliper_builder::spec;
using caliper_builder::PanelData;
using caliper_builder::SlotData;
using caliper_builder::TensorData;
using nlohmann::json;

namespace {

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool write_file(const fs::path& p, const void* data, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

std::vector<std::uint8_t> unbase64(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<std::uint8_t> out;
    out.reserve(s.size() / 4 * 3);
    unsigned acc = 0;
    int bits = 0;
    for (char c : s) {
        const int v = val(c);
        if (v < 0) continue;
        acc = (acc << 6) | unsigned(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(std::uint8_t((acc >> bits) & 0xff));
        }
    }
    return out;
}

// A wire array of panel.data ({"dtype", "shape", "base64"}) as doubles.
std::vector<double> doubles(const json& a) {
    const std::vector<std::uint8_t> b = unbase64(a.value("base64", ""));
    std::vector<double> out;
    if (a.value("dtype", "") == "f64") {
        out.resize(b.size() / 8);
        std::memcpy(out.data(), b.data(), out.size() * 8);
    } else {
        std::vector<float> f(b.size() / 4);
        std::memcpy(f.data(), b.data(), f.size() * 4);
        out.assign(f.begin(), f.end());
    }
    return out;
}

TensorData f64_tensor(const std::vector<double>& v) {
    std::vector<std::uint8_t> bytes(v.size() * 8);
    std::memcpy(bytes.data(), v.data(), bytes.size());
    caliper_builder::TensorResult r =
        TensorData::from_bytes(caliper_builder::DType::f64, {std::int64_t(v.size())}, std::move(bytes));
    return r.error ? TensorData() : std::move(r.tensor);
}

// The spec source that a binding "source.column" names.
std::string source_of(const std::string& binding) {
    const size_t dot = binding.find('.');
    return dot == std::string::npos ? binding : binding.substr(0, dot);
}

// The recorded series as the resolved data of the app's spec runtime: each line a window
// source's tensor of its y values over the steps of its x, as the live window sources give them,
// so that the app names each line by its source and draws the recorded points.
class FixtureSource final : public caliper_host::SpecPanelSource {
public:
    explicit FixtureSource(std::map<std::string, json> answers) : answers_(std::move(answers)) {}
    PanelData resolve(const spec::Spec& s, std::string_view panel_id) override {
        PanelData d;
        d.panel = std::string(panel_id);
        const spec::Panel* p = s.find_panel(d.panel);
        if (p) d.primitive = p->primitive;
        auto it = answers_.find(d.panel);
        if (!p || it == answers_.end() || !it->second["data"].is_object()) return d;
        const json& data = it->second["data"];
        if (p->primitive != spec::Primitive::series) return d;
        const json& lines = data["lines"];
        if (!lines.is_array() || lines.empty()) return d;
        const std::vector<double> xs = doubles(lines[0]["x"]);
        if (!data.value("x_index", false)) {
            SlotData x;
            x.slot = "x";
            const auto b = p->bind.find("x");
            x.binding = b != p->bind.end() && !b->second.bindings.empty() ? b->second.bindings.front() : "step";
            x.source = source_of(x.binding);
            x.tensor = f64_tensor(xs);
            d.slots.push_back(std::move(x));
        }
        std::size_t k = 0;
        for (const json& l : lines) {
            if (doubles(l["x"]) != xs) mismatched_ = true;
            SlotData y;
            y.slot = "y";
            y.index = k++;
            y.binding = l.value("binding", "");
            y.source = source_of(y.binding);
            y.tensor = f64_tensor(doubles(l["y"]));
            d.slots.push_back(std::move(y));
        }
        return d;
    }
    bool mismatched() const { return mismatched_; }

private:
    std::map<std::string, json> answers_;
    bool mismatched_ = false;
};

caliper_host::SpecRuntime::SceneLayer canvas_layer(caliper_host::SpecOffscreen* c) {
    return [c](std::unique_ptr<caliper_gfx::Backend>* owned, caliper_gfx::Backend** gfx,
               const caliper_gfx::ProgramDesc** programs, std::string* why) {
        *owned = caliper_gfx::make_metal_backend(c->native_device(), c->native_queue());
        if (!*owned) {
            *why = "no Metal backend";
            return false;
        }
        *gfx = owned->get();
        *programs = caliper_gfx::shaders::kMetalPrograms;
        return true;
    };
}

int skip(const fs::path& out, const std::string& why) {
    write_file(out / "skipped.txt", why.data(), why.size());
    std::printf("skipped: %s\n", why.c_str());
    return 77;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: caliper_dashboard_parity_app <fixture> <pieces> <out> [--size WxH]\n");
        return 2;
    }
    const fs::path fixture = argv[1], pieces = argv[2], out = argv[3];
    int W = 1280, H = 750;
    for (int i = 4; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--size") == 0 && std::sscanf(argv[i + 1], "%dx%d", &W, &H) != 2) return 2;
    std::error_code ec;
    fs::create_directories(out / "app", ec);
    fs::remove(out / "skipped.txt", ec);

    const spec::ParseResult parsed = spec::parse_toml(read_file(fixture / "spec.toml"));
    if (!parsed.ok()) {
        std::fprintf(stderr, "spec.toml: %s\n", parsed.diagnostics.empty() ? "?" : parsed.diagnostics[0].message.c_str());
        return 1;
    }
    const spec::Spec& s = parsed.spec;
    std::map<std::string, json> answers;
    for (const auto& e : fs::directory_iterator(fixture / "panels", ec)) {
        if (e.path().extension() != ".json") continue;
        json j = json::parse(read_file(e.path()), nullptr, false);
        if (j.is_discarded()) {
            std::fprintf(stderr, "%s: not JSON\n", e.path().filename().string().c_str());
            return 1;
        }
        answers[e.path().stem().string()] = std::move(j);
    }

    // The workspace: the pieces that the spec pins, as piece.update wrote them.
    const fs::path ws = out / "workspace";
    fs::remove_all(ws, ec);
    for (const spec::Program& p : s.programs) {
        if (!p.piece || p.piece->from != spec::PieceFrom::workspace) continue;
        const fs::path to = ws / "pieces" / p.piece->name / std::to_string(p.piece->version);
        fs::create_directories(to, ec);
        for (const auto& e : fs::directory_iterator(pieces / p.piece->name, ec))
            fs::copy_file(e.path(), to / e.path().filename(), fs::copy_options::overwrite_existing, ec);
    }

    // The recorded resources in a step history, the cursor live; the values at their defaults.
    caliper_builder::StepHistory history;
    history.configure(s);   // the spec's instances, whose resources it keeps
    caliper_builder::ValueStore values;
    values.reset(s);
    std::int64_t step = -1;
    std::size_t programs = 0;
    for (const auto& [id, a] : answers) {
        const json& progs = a["data"].is_object() ? a["data"].value("programs", json::array()) : json::array();
        for (const json& p : progs) {
            ++programs;
            for (const json& r : p.value("resources", json::array())) {
                caliper_builder::ResourceVersion v;
                v.step = r.value("step", std::int64_t(0));
                v.count = r.value("count", 0u);
                v.stride = r.value("stride", 0u);
                v.width = r.value("width", 0u);
                v.height = r.value("height", 0u);
                v.bytes = std::make_shared<const std::vector<std::uint8_t>>(unbase64(r.value("base64", "")));
                if (v.bytes->size() != std::size_t(v.count) * v.stride) {
                    std::fprintf(stderr, "%s/%s: %zu bytes, not count * stride\n", p.value("id", "").c_str(),
                                 r.value("resource", "").c_str(), v.bytes->size());
                    return 1;
                }
                step = std::max(step, v.step);
                history.append_resource(p.value("id", ""), r.value("resource", ""), 0, {}, std::move(v));
            }
        }
    }
    std::printf("fixture: %zu leaves, %zu programs, resources at step %lld\n", answers.size(), programs, (long long)step);

    std::string why;
    std::unique_ptr<caliper_host::SpecOffscreen> c = caliper_host::SpecOffscreen::create(W, H, 1.0f, &why);
    if (!c) return skip(out, "no Metal device (" + why + ")");
    if (!c->native_device()) return skip(out, "no Metal device (the canvas has no device)");

    FixtureSource source(answers);
    caliper_host::SpecRuntime rt(c->textures());
    rt.set_scene_layer(canvas_layer(c.get()));
    rt.set_source(&source);
    rt.set_instrument_state(&values, &history);
    const std::string wss = ws.string();
    caliper_host::set_program_workspace([wss] { return wss; });
    rt.set_spec(s);

    // Frames until every program's draw half drew the recorded step, then two more.
    std::vector<std::uint8_t> px;
    auto drawn = [&] {
        std::size_t n = 0;
        for (const caliper_builder::ProgramStatus& st : caliper_host::app_stage_host_impl()->status())
            if (st.draw && st.draw->state == "running" && st.draw->resource_step && *st.draw->resource_step == step) ++n;
        return n;
    };
    int frames = 0;
    for (; frames < 240 && drawn() < programs; ++frames)
        if (!c->render(1, [&](const ImRect& area) { rt.draw(area, c->look()); }, &px, &why)) {
            std::fprintf(stderr, "render: %s\n", why.c_str());
            return 1;
        }
    if (drawn() < programs) {
        json st = json::array();
        for (const auto& x : caliper_host::app_stage_host_impl()->status()) st.push_back(caliper_builder::program_status_json(x));
        std::fprintf(stderr, "the programs did not draw step %lld in %d frames: %s\n", (long long)step, frames, st.dump().c_str());
        return 1;
    }
    if (!c->render(2, [&](const ImRect& area) { rt.draw(area, c->look()); }, &px, &why)) {
        std::fprintf(stderr, "render: %s\n", why.c_str());
        return 1;
    }
    frames += 2;
    if (source.mismatched()) std::fprintf(stderr, "note: the lines of a chart have different steps; the first's are used\n");

    json layout = json::array(), tabs = json::array(), progs = json::array();
    for (const auto& p : rt.layout().panels)
        layout.push_back({{"id", p.id}, {"path", p.path}, {"tab", p.tab}, {"control", p.control},
                          {"rect", {{"x", p.rect.x}, {"y", p.rect.y}, {"w", p.rect.w}, {"h", p.rect.h}}}});
    for (const auto& t : rt.layout().tabs)
        tabs.push_back({{"path", t.path}, {"labels", t.labels}, {"active", t.active},
                        {"bar", {{"x", t.bar.x}, {"y", t.bar.y}, {"w", t.bar.w}, {"h", t.bar.h}}}});
    for (const auto& x : caliper_host::app_stage_host_impl()->status()) progs.push_back(caliper_builder::program_status_json(x));
    // The labels that each scene panel drew over its programs, with their text boxes, and where its scene
    // drew, in the picture's pixels: the browser compares them as text, apart from the pixels.
    json labels = json::object(), scenes = json::object();
    for (const auto& p : rt.layout().panels) {
        float r[4];
        if (!rt.scene_rect(p.id, r)) continue;
        scenes[p.id] = {r[0], r[1], r[2], r[3]};
        json list = json::array();
        for (const caliper_live_views::DrawnLabel& l : rt.scene_labels(p.id))
            list.push_back({{"text", l.text}, {"x0", l.x0}, {"y0", l.y0}, {"x1", l.x1}, {"y1", l.y1}});
        labels[p.id] = std::move(list);
    }
    const json report = {{"width", W}, {"height", H}, {"step", step}, {"frames", frames}, {"layout", layout},
                         {"tabs", tabs}, {"programs", progs}, {"scenes", scenes}, {"labels", labels}};
    const std::string rtext = report.dump(1);
    std::string png;
    if (!write_file(out / "app" / "dashboard.rgba", px.data(), px.size()) ||
        !write_file(out / "app" / "report.json", rtext.data(), rtext.size()) ||
        !caliper_host::export_encode_png(px.data(), std::uint32_t(W), std::uint32_t(H), &png) ||
        !write_file(out / "app" / "dashboard.png", png.data(), png.size())) {
        std::fprintf(stderr, "could not write into %s\n", (out / "app").string().c_str());
        return 1;
    }
    std::printf("dashboard drawn: %dx%d, %d frames, programs at step %lld\n", W, H, frames, (long long)step);
    return 0;
}
