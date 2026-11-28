// ===========================================================================
// examples/embed_host — what main.mm and main_win32.cpp share: the version
// check and the offscreen mode.
//
// The offscreen mode (CALIPER_EMBED_OFFSCREEN=1) makes the same five calls with
// an offscreen canvas instead of a native view. main() takes this path before
// it creates any application object or window, so nothing appears on screen:
// no window, no Dock icon, no taskbar button. It pumps frames for
// CALIPER_EMBED_EXIT_AFTER seconds (default 1), reads the last frame back,
// prints what it did and returns the exit code: 0 when the instrument drew, 1
// on any refusal or a blank frame.
// ===========================================================================
#pragma once

#include <caliper/embed.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// The library must have the header's major and at least its minor: an older
// library lacks calls this program was compiled to make.
inline bool embed_version_ok() {
    const uint32_t lib = caliper_embed_api_version();
    if ((lib >> 16) == CALIPER_EMBED_VERSION_MAJOR &&
        ((lib >> 8) & 0xffu) >= CALIPER_EMBED_VERSION_MINOR)
        return true;
    std::fprintf(stderr, "[embed-host] the library implements embed %u.%u.%u, "
                         "this program needs %d.%d or a later minor\n",
                 lib >> 16, (lib >> 8) & 0xffu, lib & 0xffu,
                 CALIPER_EMBED_VERSION_MAJOR, CALIPER_EMBED_VERSION_MINOR);
    return false;
}

inline bool embed_offscreen_requested() {
    const char* v = std::getenv("CALIPER_EMBED_OFFSCREEN");
    return v && *v && std::strcmp(v, "0") != 0;
}

inline int embed_run_offscreen(const char* instrument_id, CaliperLogFn log_fn,
                               CaliperCrashFn crash_fn) {
    const int W = 1280, H = 800;
    const uint32_t lib = caliper_embed_api_version();
    std::printf("[embed-host] offscreen mode, embed %u.%u.%u\n",
                lib >> 16, (lib >> 8) & 0xffu, lib & 0xffu);
    std::fflush(stdout);   // keep this line ahead of the core's stderr lines

    // STEP 1: create the core, as the window mode does.
    CaliperCoreDesc desc{};
    desc.struct_size     = sizeof desc;
    desc.renderer        = CALIPER_RENDERER_DEFAULT;
    desc.instruments_dir = std::getenv("CALIPER_EMBED_INSTRUMENTS");  // may be NULL
    desc.log_fn          = log_fn;
    desc.crash_fn        = crash_fn;
    CaliperCore* core = caliper_core_create(&desc);
    if (!core) { std::fprintf(stderr, "[embed-host] core create failed\n"); return 1; }

    // STEP 2: an offscreen canvas takes no native view.
    CaliperCanvasDesc canvas{};
    canvas.struct_size   = sizeof canvas;
    canvas.mode          = CALIPER_CANVAS_OFFSCREEN;
    canvas.width         = W;
    canvas.height        = H;
    canvas.content_scale = 1.0f;
    if (!caliper_core_attach_canvas(core, nullptr, &canvas)) {
        std::fprintf(stderr, "[embed-host] attach_canvas failed: %s\n",
                     caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return 1;
    }

    // STEP 3: load the instrument.
    if (!caliper_core_load_instrument(core, instrument_id)) {
        std::fprintf(stderr, "[embed-host] load_instrument '%s' failed: %s\n",
                     instrument_id, caliper_core_last_error(core));
        caliper_core_shutdown(core);
        return 1;
    }

    // STEP 4: pump about 60 frames a second from this loop.
    double seconds = 1.0;
    if (const char* s = std::getenv("CALIPER_EMBED_EXIT_AFTER")) seconds = std::atof(s);
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    int frames = 0;
    double elapsed = 0.0;
    do {
        caliper_core_frame(core);
        ++frames;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        elapsed = std::chrono::duration<double>(clock::now() - t0).count();
    } while (elapsed < seconds);

    // Read the last frame back and count the pixels the instrument drew: any
    // channel above 80 is well clear of the dark background.
    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    int drawn = 0;
    if (caliper_core_read_pixels(core, px.data(), W * 4)) {
        for (size_t i = 0; i + 3 < px.size(); i += 4)
            if (px[i] > 80 || px[i + 1] > 80 || px[i + 2] > 80) ++drawn;
    }
    std::printf("[embed-host] offscreen: %d frames of %s at %dx%d in %.2f s, "
                "%d drawn pixels in the last frame\n",
                frames, instrument_id, W, H, elapsed, drawn);

    // STEP 5: tear down.
    caliper_core_shutdown(core);
    return drawn > 0 ? 0 : 1;
}
