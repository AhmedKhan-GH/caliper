// caliper shot (shot.h): the app's applet page, drawn into an offscreen
// canvas and written as a PNG.
#include "shot.h"

#include "app_paths.h"
#include "applet_loader.h"
#include "background_process.h"
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
    std::unique_ptr<AppletLoader> loader;
    std::unique_ptr<AppletPage> page;        // the dock-hint sink points into it
    std::unique_ptr<HostRenderer> renderer;
    bool ui_context = false;
    bool canvas = false;
    bool services = false;

    ~ShotSession() {
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

    // N frames at about 60 a second, as the app draws them, so jobs and
    // workers the applet starts get the time they get in the app.
    ImGuiIO& io = ImGui::GetIO();
    JobsTray tray;
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    double last = 0.0;
    bool alive = true;
    for (int f = 0; f < cl.frames && alive; ++f) {
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
            page.draw(loader, idx, fi, s.renderer->name(), chromeless, &tray);
        s.renderer->canvas_render();
        alive = r.alive;
    }
    if (!alive)
        return fail(id + " was quarantined: " + loader.at(idx).status_text);

    // The last frame, opaque as the window shows it.
    std::vector<uint8_t> px((size_t)W * (size_t)H * 4);
    if (!s.renderer->canvas_read_pixels(px.data(), W * 4))
        return fail("could not read the frame back from the " +
                    std::string(s.renderer->name()) + " renderer");
    for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
    std::string why;
    if (!export_write_png_atomic(out.string(), px.data(), (uint32_t)W, (uint32_t)H, &why))
        return fail("cannot write " + out.string() + (why.empty() ? "" : ": " + why));

    std::printf("wrote %s: %s at %dx%d after %d frames\n", out.string().c_str(), id.c_str(),
                W, H, cl.frames);
    return 0;
}

} // namespace caliper_host
