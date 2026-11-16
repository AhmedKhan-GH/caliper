// caliper shot (shot.h): the app's applet page, drawn into an offscreen
// canvas and written as a PNG.
#include "shot.h"

#include "api/app_api.h"
#include "app_paths.h"
#include "applet_loader.h"
#include "background_process.h"
#include "builder/app_hooks.h"
#include "core_lifecycle.h"
#include "export_service.h"
#include "host_services.h"
#include "job_system.h"
#include "negotiation.h"
#include "renderer/host_renderer.h"
#include "ui/applet_page.h"
#include "ui/dock_layout.h"
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"
#include "ui/ui_recovery.h"

#include <imgui.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace caliper_host {

namespace {

namespace fs = std::filesystem;

int fail(const std::string& message) {
    std::fprintf(stderr, "caliper: %s\n", message.c_str());
    return 1;
}

// With Caliper API on, SIGINT and SIGTERM end the frames after the one being
// drawn, and the shot writes its image as usual.
volatile std::sig_atomic_t g_stop = 0;
extern "C" void shot_on_signal(int) { g_stop = 1; }

// A new, empty folder under the system's temporary folder, or "" when none
// could be made.
fs::path make_temp_root() {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec) return {};
    std::random_device rd;
    for (int attempt = 0; attempt < 16; ++attempt) {
        char name[48];
        std::snprintf(name, sizeof name, "caliper-shot-%08x%08x", rd(), rd());
        const fs::path dir = base / name;
        if (fs::create_directory(dir, ec) && !ec) return dir;
    }
    return {};
}

// Everything a shot sets up, torn down in the app's order (CaliperApp::
// cleanup) whichever way the shot ends.
struct ShotSession {
    fs::path temp_root;                      // removed last, when set
    std::unique_ptr<AppApi> api;             // stopped first: waiting calls get 503
    std::unique_ptr<AppSession> session;     // closed next, before the host's stores
    std::unique_ptr<AppletLoader> loader;
    std::unique_ptr<AppletPage> page;        // the dock-hint sink points into it
    std::unique_ptr<HostRenderer> renderer;
    bool ui_context = false;
    bool canvas = false;
    bool services = false;

    ~ShotSession() {
        if (api) api->stop();
        if (session) session->close();
        if (services) {
            // Workers first: a job may hold a pointer into applet state.
            host_job_system().cancel_all_and_join();
            if (loader) loader->close_all();
            set_dock_hint_sink(nullptr, nullptr);
            services_set_renderer(nullptr);
            services_shutdown();
        }
        loader.reset();
        page.reset();
        if (canvas) renderer->canvas_shutdown();
        renderer.reset();
        if (ui_context) core_destroy_ui_context();
        if (!temp_root.empty()) {
            caliper::set_app_data_dir_override("");
            std::error_code ec;
            fs::remove_all(temp_root, ec);
        }
    }
};

} // namespace

int run_shot(const CommandLine& cl) {
    // No Dock icon and no menu bar, before anything that could create either.
    use_background_activation_policy();

    const std::string& id = cl.launch_id;
    const int W = cl.width, H = cl.height;

    // The image's folder must exist; checked first, so a typo costs no render.
    std::error_code ec;
    const fs::path out = fs::absolute(fs::path(cl.out), ec).lexically_normal();
    if (ec || !out.has_filename())
        return fail("cannot write " + cl.out + ": not a file name");
    if (fs::is_directory(out, ec))
        return fail("cannot write " + out.string() + ": it is a folder");
    if (!fs::is_directory(out.parent_path(), ec))
        return fail("cannot write " + out.string() + ": the folder " +
                    out.parent_path().string() + " does not exist");

    // The folders the app searches, worked out before any folder is created:
    // the user applets folder is the app's, even when the data root below is
    // a temporary one.
    const std::string user_dir = caliper::user_applets_dir();
    const std::string shipped_dir = caliper::shipped_applets_dir();
    const std::string user_data_folder = caliper::app_data_location();   // the API file's
    const char* data_env = std::getenv("CALIPER_DATA_DIR");
    const bool own_root = !(data_env && *data_env);

    ShotSession s;
    if (own_root) {
        s.temp_root = make_temp_root();
        if (s.temp_root.empty())
            return fail("cannot create a temporary data folder for the shot");
    }
    const fs::path data_root =
        own_root ? s.temp_root : fs::path(caliper::app_data_location());

    // The applet, found as the app finds it. Scanning reads manifests only.
    s.loader = std::make_unique<AppletLoader>(host_caps(), (data_root / "data").string());
    AppletLoader& loader = *s.loader;
    loader.scan(applet_search_dirs(cl.applet_dirs, std::getenv("CALIPER_APPLET_PATH"),
                                   user_dir, shipped_dir));
    const int idx = loader.find(id);
    if (idx < 0)
        return fail("no applet with the id " + id +
                    " in the search folders (caliper list shows them)");
    if (loader.at(idx).status != AppletStatus::Ready)
        return fail(id + " cannot be launched (" +
                    applet_status_word(loader.at(idx).status) +
                    "): " + loader.at(idx).status_text);

    // The stores, the applet's data folder and the display font path resolve
    // under the temporary root from here on.
    if (own_root) caliper::set_app_data_dir_override(s.temp_root.string());

    // The renderer's offscreen canvas: Metal on macOS, Vulkan on Windows. The
    // GL fallback has none.
    s.renderer = core_select_renderer();
    if (!s.renderer->canvas_supported())
        return fail(std::string("shot cannot draw offscreen: the ") + s.renderer->name() +
                    " renderer has no offscreen canvas (shot needs Metal or Vulkan)");

    // The UI contexts and the house style, as the app makes them, but with no
    // imgui.ini: nothing is read, so the layout is a first run's, and nothing
    // is written.
    core_create_ui_context();
    s.ui_context = true;
    ImGui::GetIO().IniFilename = nullptr;
    sternbild::FontPaths fonts = sternbild::default_font_paths();
    // The display face draws only the landing's wordmark, which the applet
    // page has not; it is loaded only when CALIPER_DISPLAY_FONT names it.
    if (const char* f = std::getenv("CALIPER_DISPLAY_FONT"); !f || !*f)
        fonts.display_font.clear();
    sternbild::load_fonts(fonts);
    sternbild::apply_style();

    if (!s.renderer->canvas_init(nullptr, HostRenderer::CANVAS_OFFSCREEN, W, H))
        return fail(std::string("shot cannot draw offscreen: the ") + s.renderer->name() +
                    " renderer could not make a " + std::to_string(W) + "x" +
                    std::to_string(H) + " canvas (no GPU?)");
    s.canvas = true;

    // Services, the dock-hint sink and the bridge, in the app's order.
    s.page = std::make_unique<AppletPage>();
    AppletPage& page = *s.page;
    services_init();
    s.services = true;
    set_dock_hint_sink(
        [](void* layout, const char* title, uint32_t role) {
            static_cast<DockLayout*>(layout)->hint(title, role);
        },
        &page.dock_layout());
    services_set_renderer(s.renderer.get());

    bool chromeless = false;
    if (const char* c = std::getenv("CALIPER_CHROMELESS")) chromeless = std::strcmp(c, "1") == 0;

    // Launched as CALIPER_AUTOLAUNCH launches it, before the first frame.
    {
        const UiCheckpoint ui_checkpoint = UiCheckpoint::store();
        if (!loader.launch(idx, applet_host_proto())) {
            ui_checkpoint.recover();
            return fail(id + " failed to launch: " + loader.at(idx).status_text);
        }
    }
    page.enter();
    int current = idx;   // the applet the page shows; a launch through the API changes it

    // Caliper API (--api) while the frames are drawn. The canvas is offscreen,
    // so a capture is read back from it at the end of a frame, as the image is.
    void (*old_int)(int) = SIG_DFL;
    void (*old_term)(int) = SIG_DFL;
    if (api_wanted(cl)) {
        s.api = std::make_unique<AppApi>();
        std::string why;
        if (!s.api->start(cl, "", user_data_folder, std::cout, &why))
            return fail("cannot start Caliper API: " + why);
        g_stop = 0;
        old_int = std::signal(SIGINT, shot_on_signal);
        old_term = std::signal(SIGTERM, shot_on_signal);
    }

    // A launch through the API or the session's app.launch: the page shows
    // that applet from this frame on. The applet shown goes first, as Home
    // takes it in the app; when the new one fails to launch, the previous one
    // is launched again, and `lost` is set when that fails too and the shot
    // has no applet left.
    bool lost = false;
    auto launch = [&](const std::string& want_id) -> LaunchOutcome {
        const int want = loader.find(want_id);
        if (want < 0)
            return {404, "no applet with the id " + want_id + " in the search folders (GET /v0/applets lists them)",
                    "app.applet_not_found"};
        const AppletEntry& w = loader.at(want);
        if (want == current) return {200, launched_json(w.manifest.id, w.manifest.version, true), ""};
        if (w.status != AppletStatus::Ready)
            return {409, want_id + " cannot be launched (" + applet_status_word(w.status) + "): " + w.status_text,
                    "app.launch_refused"};
        host_job_system().cancel_all_and_join();
        loader.teardown(current);
        {
            const UiCheckpoint ui_checkpoint = UiCheckpoint::store();
            if (loader.launch(want, applet_host_proto())) {
                current = want;
                page.enter();
                return {200, launched_json(loader.at(want).manifest.id, loader.at(want).manifest.version, false),
                        ""};
            }
            ui_checkpoint.recover();
        }
        const LaunchOutcome failed{409, want_id + " failed to launch: " + loader.at(want).status_text,
                                   "app.launch_failed"};
        const UiCheckpoint ui_checkpoint = UiCheckpoint::store();
        if (loader.launch(current, applet_host_proto())) {
            page.enter();
            return failed;
        }
        ui_checkpoint.recover();
        lost = true;
        return failed;
    };

    // The builder session (builder/app_session.h) for the workspace of the
    // command line, drained where the API's calls are taken. Its capture.frame
    // reads back the canvas at the end of the frame drawn next. Without
    // CALIPER_DATA_DIR the default workspace is under the temporary root.
    bool capture_wanted = false;
    s.session = std::make_unique<AppSession>();
    {
        AppSession::Hooks h = app_session_hooks();
        h.applets_json = [&] { return applets_json(loader); };
        h.launch = launch;
        h.request_capture = [&] { capture_wanted = true; };
        s.session->start(app_session_options(cl), std::move(h));
    }

    // N frames at about 60 a second, as the app draws them, so jobs and
    // workers the applet starts get the time they get in the app.
    ImGuiIO& io = ImGui::GetIO();
    JobsTray tray;
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    double last = 0.0;
    bool alive = true;
    int frames = 0;
    // A command script keeps the frames coming until it is done.
    for (int f = 0; (f < cl.frames || s.session->script_running()) && alive && !g_stop; ++f) {
        // The API's calls of this frame: the applets and launches now, before
        // the frame is drawn, and captures once it is rendered.
        std::vector<uint64_t> captures;
        if (s.api) {
            for (const caliper_api::Call& c : s.api->take()) {
                if (c.kind == caliper_api::Call::Kind::Applets) {
                    s.api->reply(c.ticket, 200, applets_json(loader));
                } else if (c.kind == caliper_api::Call::Kind::Capture) {
                    captures.push_back(c.ticket);
                } else {
                    const LaunchOutcome o = launch(c.applet_id);
                    if (o.status == 200) s.api->reply(c.ticket, 200, o.body);
                    else s.api->reply_error(c.ticket, o.status, o.body);
                }
            }
        }
        s.session->frame();
        if (lost)
            return fail(loader.at(current).manifest.id + " failed to launch again after a launch: " +
                        loader.at(current).status_text);
        if (f > 0)
            std::this_thread::sleep_until(start + std::chrono::microseconds(
                                                      (int64_t)f * 1000000 / 60));
        const double now = std::chrono::duration<double>(clock::now() - start).count();
        const double dt = f == 0 ? 1.0 / 60.0 : now - last;
        last = now;
        io.DisplaySize = ImVec2((float)W, (float)H);
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.DeltaTime = dt > 0.0 ? (float)dt : 1.0f / 60.0f;

        CaliperFrameInfo fi{};
        fi.struct_size = sizeof fi;
        fi.fb_width = W;
        fi.fb_height = H;
        fi.dpi_scale = 1.0f;
        fi.time_sec = now;
        fi.delta_sec = dt;

        s.renderer->canvas_new_frame();
        auto views = host_job_system().views();
        tray.note_failures(views);
        tray.begin_frame(std::move(views), now);
        const AppletPage::Result r =
            page.draw(loader, current, fi, s.renderer->name(), chromeless, &tray);
        s.renderer->canvas_render();
        alive = r.alive;
        ++frames;

        // The session's capture: this frame, read back as the image is, opaque.
        if (capture_wanted) {
            capture_wanted = false;
            std::vector<uint8_t> px((size_t)W * (size_t)H * 4);
            if (s.renderer->canvas_read_pixels(px.data(), W * 4)) {
                for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
                s.session->capture_ready((uint32_t)W, (uint32_t)H, std::move(px));
            } else {
                s.session->capture_failed(std::string("could not read the frame back from the ") +
                                          s.renderer->name() + " renderer");
            }
        }

        // Captures: this frame, read back as the image is, opaque.
        if (!captures.empty()) {
            std::vector<uint8_t> px((size_t)W * (size_t)H * 4);
            const bool read = s.renderer->canvas_read_pixels(px.data(), W * 4);
            if (read)
                for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
            for (size_t i = 0; i < captures.size(); ++i) {
                if (!read)
                    s.api->reply_error(captures[i], 500,
                                       std::string("could not read the frame back from the ") +
                                           s.renderer->name() + " renderer");
                else
                    s.api->reply_capture(captures[i], (uint32_t)W, (uint32_t)H,
                                         i + 1 == captures.size() ? std::move(px) : px);
            }
        }
    }
    if (s.api) {
        std::signal(SIGINT, old_int);
        std::signal(SIGTERM, old_term);
    }
    const std::string shown = loader.at(current).manifest.id;
    if (!alive)
        return fail(shown + " was quarantined: " + loader.at(current).status_text);

    // The last frame, opaque as the window shows it.
    std::vector<uint8_t> px((size_t)W * (size_t)H * 4);
    if (!s.renderer->canvas_read_pixels(px.data(), W * 4))
        return fail("could not read the frame back from the " +
                    std::string(s.renderer->name()) + " renderer");
    for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
    std::string why;
    if (!export_write_png_atomic(out.string(), px.data(), (uint32_t)W, (uint32_t)H, &why))
        return fail("cannot write " + out.string() + (why.empty() ? "" : ": " + why));

    std::printf("wrote %s: %s at %dx%d after %d frames\n", out.string().c_str(), shown.c_str(),
                W, H, frames);
    return 0;
}

} // namespace caliper_host
