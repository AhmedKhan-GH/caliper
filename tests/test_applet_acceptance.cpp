// The headless acceptance check for applets built outside the core. Skipped
// unless CALIPER_TEST_APPLET_DIR names a folder of built applets (manifests
// beside their libraries). Then it scans that folder with the real loader and
// the capabilities the caliper executable advertises (host_caps()), and for
// each applet runs negotiation, dlopen, the descriptor checks, create() and
// destroy() (AppletLoader::probe). initialize() is not run: it needs the
// host's services and UI contexts, which a headless test does not have. One
// line per applet goes to stdout; the case fails if any applet is not
// accepted, or if the folder holds none.
//
//   CALIPER_TEST_APPLET_DIR=<folder> \
//       ./cmake-build-release/tests/caliper_tests -tc="acceptance: applets*"
#include <doctest/doctest.h>
#include "applet_loader.h"
#include "test_temp.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;
using namespace caliper_host;

namespace {
const char* applet_dir() { return std::getenv("CALIPER_TEST_APPLET_DIR"); }
const char* status_word(AppletStatus s) {
    switch (s) {
        case AppletStatus::Ready:       return "ACCEPTED";
        case AppletStatus::Active:      return "ACTIVE";
        case AppletStatus::Refused:     return "REFUSED";
        case AppletStatus::Failed:      return "FAILED";
        case AppletStatus::Quarantined: return "QUARANTINED";
        case AppletStatus::Shadowed:    return "SHADOWED";
    }
    return "UNKNOWN";
}
} // namespace

TEST_CASE("acceptance: applets in CALIPER_TEST_APPLET_DIR load headless" *
          doctest::skip(applet_dir() == nullptr || !*applet_dir())) {
    const std::string dir = applet_dir();
    REQUIRE_MESSAGE(fs::is_directory(dir), "not a folder: " << dir);
    // A private data root: the check never touches the user's applet data.
    const fs::path data = caliper_test::unique_temp_path("caliper-acceptance-data");
    std::error_code ec;
    fs::remove_all(data, ec);

    const HostCaps caps = host_caps();
    std::printf("host %s, ABI epoch %u, libtorch %s, UI stack %s\n",
                caps.version.c_str(), caps.abi_epoch,
                caps.torch_version.empty() ? "none" : caps.torch_version.c_str(),
                caps.ui_stack_hash.empty() ? "not checked" : caps.ui_stack_hash.c_str());
    AppletLoader L(caps, data.string());
    L.scan(dir);
    CHECK_MESSAGE(L.count() > 0, "no applet manifests in " << dir);

    int rejected = 0;
    for (int i = 0; i < L.count(); i++) {
        if (L.at(i).status == AppletStatus::Ready || L.at(i).status == AppletStatus::Shadowed)
            L.probe(i);
        const AppletEntry& a = L.at(i);
        const bool ok = a.status == AppletStatus::Ready || a.status == AppletStatus::Shadowed;
        rejected += ok ? 0 : 1;
        std::printf("%-12s %s %s  %s%s%s\n", status_word(a.status),
                    a.manifest.id.empty() ? a.manifest.name.c_str() : a.manifest.id.c_str(),
                    a.manifest.version.empty() ? "-" : a.manifest.version.c_str(),
                    a.folder.c_str(), a.status_text.empty() ? "" : "\n             ",
                    a.status_text.c_str());
        CHECK_MESSAGE(ok, a.manifest.name << ": " << a.status_text);
    }
    std::printf("%d of %d applets accepted\n", L.count() - rejected, L.count());
    std::fflush(stdout);
    L.close_all();
    fs::remove_all(data, ec);
}
