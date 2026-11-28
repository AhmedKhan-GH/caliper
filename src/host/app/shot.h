#pragma once
// ============================================================================
// caliper shot: renders the app's instrument page for one instrument and
// writes the last frame as a PNG, with nothing on screen.
//
//   caliper shot [--instruments <dir>]... --launch <id> [--size <W>x<H>]
//                [--frames <N>] --out <file.png> [--api [--api-port <n>]]
//
// It runs the app's own code for the page (pages/instrument_page.h: the page
// bar and the status bar unless CALIPER_CHROMELESS=1, the dock space, docking
// by role, the instrument's frame and the jobs tray), finds instruments in the
// app's search order, launches the instrument with that id, draws N frames (60
// by default) at about 60 a second on a W by H canvas (1600 by 1000 by default)
// at a scale of 1, reads the last frame back and writes it as a PNG.
//
// Nothing becomes visible. The page is drawn into the renderer's offscreen
// canvas (HostRenderer::canvas_init with CANVAS_OFFSCREEN, the canvas the
// embed core uses) and read back with canvas_read_pixels: shot never calls
// glfwInit() and creates no window of any kind. On macOS it also sets the
// background activation policy first (background_process.h), so the process
// has no Dock icon and no menu bar.
//
// A shot never reads or writes the saved window layout: ImGui keeps no
// imgui.ini, so every shot shows the layout of a first run. With
// CALIPER_DATA_DIR set, the stores and the instrument's data folder are under
// it, as in the app. Without it, they are under a temporary folder that shot
// creates and removes, so a shot writes nothing in the app-data folder. The
// user instruments folder is searched in either case, as the app searches it.
//
// With --api, shot is the app's windowless mode: it serves Caliper API
// (api/app_api.h) while it draws, answering the API's calls between frames
// and its captures at the end of a frame with the same read-back as the
// image. A launch through the API changes the instrument the page shows.
// SIGINT and SIGTERM end the frames early, and the image is written as usual.
// The API file goes to the app-data folder that is the user's, also when the
// stores are under a temporary one.
//
// Exit codes: 0 when the PNG is written; 1 when the instrument is not found,
// cannot be launched, fails to launch or is quarantined, when the renderer
// cannot draw offscreen (CALIPER_RENDERER=gl, or Metal or Vulkan could not
// start, and there is no fallback renderer), or when the image cannot be
// written. A wrong command line is caught by parse_command_line (exit 2).
// ============================================================================
#include "host_cli.h"

namespace caliper_host {

int run_shot(const CommandLine& cl);

} // namespace caliper_host
