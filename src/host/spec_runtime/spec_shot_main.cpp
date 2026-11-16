// caliper-spec-shot: a development tool that draws a spec with the spec
// runtime into an offscreen canvas and writes a PNG, as `caliper shot` does
// for an applet page. It opens no window and is not installed.
//
//   caliper-spec-shot --workspace <dir> --applet <id> [options]
//   caliper-spec-shot --spec <file> [--workspace <dir>] [options]
//
// Options: --size WxH (1440x900), --out <file.png> (spec-shot.png),
// --frames N (4), --text-scale S (1), and --tab <path>=<index>, repeated,
// to show another tab of a tabs node (the path is the node's child indices
// from the root joined by '.', "" or "root" for the root).
//
// With --applet it reads the saved spec applet applets/<id>/ of the
// workspace, through its manifest, without changing the workspace: the spec
// is not made the open spec and no command runs. It resolves each panel over
// the workspace's loader outputs with a builder Session. A spec given with
// --spec and no workspace draws each panel's state.
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"

#include "export_service.h"

#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/format.h>

#include <toml++/toml.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace spec = caliper_builder::spec;
using namespace caliper_host;

namespace {

int usage(const char* why) {
    std::fprintf(stderr, "caliper-spec-shot: %s\n", why);
    std::fprintf(stderr,
                 "usage: caliper-spec-shot (--workspace <dir> --applet <id> | --spec <file> [--workspace <dir>])\n"
                 "                         [--size WxH] [--out <file.png>] [--frames N] [--text-scale S]\n"
                 "                         [--tab <path>=<index>]...\n");
    return 2;
}

int fail(const std::string& message) {
    std::fprintf(stderr, "caliper-spec-shot: %s\n", message.c_str());
    return 1;
}

bool read_file(const fs::path& path, std::string* out, size_t max_bytes) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return false;
    if (fs::file_size(path, ec) > max_bytes) return false;
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return (bool)in || in.eof();
}

// The spec file of a saved spec applet, through its manifest.
bool applet_spec_path(const fs::path& workspace, const std::string& id, fs::path* out, std::string* why) {
    if (!std::regex_match(id, std::regex("[a-z][a-z0-9_-]{0,63}"))) {
        *why = "the applet id " + id + " is not a workspace id";
        return false;
    }
    const fs::path dir = workspace / "applets" / id;
    std::string text;
    if (!read_file(dir / (id + ".caliper.toml"), &text, 64 << 10)) {
        *why = "the workspace has no applet " + id + " (no manifest " + (dir / (id + ".caliper.toml")).string() + ")";
        return false;
    }
    toml::table parsed;
#if TOML_EXCEPTIONS
    try {
        parsed = toml::parse(text);
    } catch (const toml::parse_error&) {
        *why = "the manifest of " + id + " is not TOML";
        return false;
    }
#else
    toml::parse_result result = toml::parse(text);
    if (!result) {
        *why = "the manifest of " + id + " is not TOML";
        return false;
    }
    parsed = std::move(result).table();
#endif
    const auto kind = parsed["applet"]["kind"].value<std::string>();
    const auto file = parsed["applet"]["spec"].value<std::string>();
    if (!kind || *kind != "spec" || !file) {
        *why = id + " is not a spec applet";
        return false;
    }
    if (file->empty() || file->find('/') != std::string::npos || file->find('\\') != std::string::npos ||
        file->find("..") != std::string::npos) {
        *why = "the manifest of " + id + " names a spec file outside its folder";
        return false;
    }
    *out = dir / *file;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string workspace, applet, spec_file, out = "spec-shot.png";
    int width = 1440, height = 900, frames = 4;
    float text_scale = 1.0f;
    SpecTabChoice tabs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) return nullptr;
            (void)name;
            return argv[++i];
        };
        const char* v = nullptr;
        if (a == "--workspace" && (v = next("--workspace"))) workspace = v;
        else if (a == "--applet" && (v = next("--applet"))) applet = v;
        else if (a == "--spec" && (v = next("--spec"))) spec_file = v;
        else if (a == "--out" && (v = next("--out"))) out = v;
        else if (a == "--frames" && (v = next("--frames"))) frames = std::atoi(v);
        else if (a == "--text-scale" && (v = next("--text-scale"))) text_scale = (float)std::atof(v);
        else if (a == "--size" && (v = next("--size"))) {
            if (std::sscanf(v, "%dx%d", &width, &height) != 2) return usage("--size takes WxH, such as 1440x900");
        } else if (a == "--tab" && (v = next("--tab"))) {
            const std::string t = v;
            const size_t eq = t.rfind('=');
            if (eq == std::string::npos) return usage("--tab takes <path>=<index>");
            std::string path = t.substr(0, eq);
            if (path == "root") path.clear();
            tabs[path] = std::atoi(t.c_str() + eq + 1);
        } else {
            return usage(("unknown or incomplete option " + a).c_str());
        }
    }
    if (applet.empty() == spec_file.empty()) return usage("give either --applet with --workspace, or --spec");
    if (!applet.empty() && workspace.empty()) return usage("--applet needs --workspace");
    if (width < 64 || height < 64 || width > 8192 || height > 8192) return usage("--size must be from 64x64 to 8192x8192");
    if (frames < 1 || frames > 120) return usage("--frames must be from 1 to 120");
    if (!(text_scale >= 0.5f && text_scale <= 4.0f)) return usage("--text-scale must be from 0.5 to 4");

    std::error_code ec;
    const fs::path out_path = fs::absolute(out, ec).lexically_normal();
    if (ec || !out_path.has_filename() || !fs::is_directory(out_path.parent_path(), ec))
        return fail("cannot write " + out + ": its folder does not exist");

    // The spec.
    fs::path spec_path = spec_file;
    if (!applet.empty()) {
        std::string why;
        if (!applet_spec_path(workspace, applet, &spec_path, &why)) return fail(why);
    }
    std::string text;
    if (!read_file(spec_path, &text, 512 << 10)) return fail("cannot read the spec " + spec_path.string());
    spec::ParseResult parsed = spec::parse_toml(text);
    if (!parsed.ok()) {
        std::string first = parsed.diagnostics.empty() ? "" : parsed.diagnostics[0].code + ": " + parsed.diagnostics[0].message;
        return fail("the spec " + spec_path.string() + " does not parse: " + first);
    }

    // The data: a session over the workspace, which resolves each panel.
    std::unique_ptr<caliper_builder::Session> session;
    std::unique_ptr<SessionPanelSource> source;
    if (!workspace.empty()) {
        auto opened = caliper_builder::Session::open(workspace);
        if (opened.error) return fail("cannot open the workspace " + workspace + ": " + opened.error.message);
        session = std::move(opened.session);
        source = std::make_unique<SessionPanelSource>(*session);
    }

    std::string why;
    std::unique_ptr<SpecOffscreen> canvas = SpecOffscreen::create(width, height, text_scale, &why);
    if (!canvas) return fail("cannot draw offscreen: " + why);
    std::vector<uint8_t> rgba;
    std::vector<std::string> bounds;
    {
        std::unique_ptr<SpecRuntime> runtime;
        canvas->with_context([&] {
            runtime = std::make_unique<SpecRuntime>(canvas->textures());
            runtime->set_source(source.get());
            runtime->set_spec(parsed.spec);
            runtime->tabs() = tabs;
        });
        if (!canvas->render(frames, [&](const ImRect& area) { runtime->draw(area, canvas->look()); }, &rgba, &why))
            return fail(why);
        for (const auto& p : parsed.spec.panels)
            if (const PreparedPanel* pp = runtime->prepared(p.id)) {
                for (const auto& b : pp->bounds) bounds.push_back(p.id + ": " + b);
                if (!pp->state.ready())
                    bounds.push_back(p.id + ": " + pp->state.label + " (" + pp->state.detail + ")");
            }
        canvas->with_context([&] { runtime.reset(); });
    }
    if (!export_write_png_atomic(out_path.string(), rgba.data(), (uint32_t)width, (uint32_t)height, &why))
        return fail("cannot write " + out_path.string() + (why.empty() ? "" : ": " + why));
    std::printf("wrote %s: %s at %dx%d with the %s renderer\n", out_path.string().c_str(),
                parsed.spec.applet.id.c_str(), width, height, canvas->renderer_name());
    for (const auto& b : bounds) std::printf("  %s\n", b.c_str());
    canvas.reset();
    if (session) session->close();
    return 0;
}
