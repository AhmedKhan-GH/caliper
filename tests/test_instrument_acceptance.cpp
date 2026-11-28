// The headless acceptance check for instruments built outside the core. Skipped
// unless CALIPER_TEST_INSTRUMENT_DIR names a folder of built instruments (manifests
// beside their libraries). Then it scans that folder with the real loader and
// the capabilities the caliper executable advertises (host_caps()), and for
// each instrument runs negotiation, dlopen, the descriptor checks, create() and
// destroy() (InstrumentLoader::probe). initialize() is not run: it needs the
// host's services and UI contexts, which a headless test does not have. One
// line per instrument goes to stdout; the case fails if any instrument is not
// accepted, or if the folder holds none.
//
//   CALIPER_TEST_INSTRUMENT_DIR=<folder> \
//       ./cmake-build-release/tests/caliper_tests -tc="acceptance: instruments*"
#include <doctest/doctest.h>
#include "instrument_loader.h"
#include "test_temp.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;
using namespace caliper_host;

namespace {
const char* instrument_dir() { return std::getenv("CALIPER_TEST_INSTRUMENT_DIR"); }
const char* status_word(InstrumentStatus s) {
    switch (s) {
        case InstrumentStatus::Ready:       return "ACCEPTED";
        case InstrumentStatus::Active:      return "ACTIVE";
        case InstrumentStatus::Refused:     return "REFUSED";
        case InstrumentStatus::Failed:      return "FAILED";
        case InstrumentStatus::Quarantined: return "QUARANTINED";
        case InstrumentStatus::Shadowed:    return "SHADOWED";
    }
    return "UNKNOWN";
}
} // namespace

TEST_CASE("acceptance: instruments in CALIPER_TEST_INSTRUMENT_DIR load headless" *
          doctest::skip(instrument_dir() == nullptr || !*instrument_dir())) {
    const std::string dir = instrument_dir();
    REQUIRE_MESSAGE(fs::is_directory(dir), "not a folder: " << dir);
    // A private data root: the check never touches the user's instrument data.
    const fs::path data = caliper_test::unique_temp_path("caliper-acceptance-data");
    std::error_code ec;
    fs::remove_all(data, ec);

    const HostCaps caps = host_caps();
    std::printf("host %s, ABI epoch %u, libtorch %s, UI stack %s\n",
                caps.version.c_str(), caps.abi_epoch,
                caps.torch_version.empty() ? "none" : caps.torch_version.c_str(),
                caps.ui_stack_hash.empty() ? "not checked" : caps.ui_stack_hash.c_str());
    InstrumentLoader L(caps, data.string());
    L.scan(dir);
    CHECK_MESSAGE(L.count() > 0, "no instrument manifests in " << dir);

    int rejected = 0;
    for (int i = 0; i < L.count(); i++) {
        if (L.at(i).status == InstrumentStatus::Ready || L.at(i).status == InstrumentStatus::Shadowed)
            L.probe(i);
        const InstrumentEntry& a = L.at(i);
        const bool ok = a.status == InstrumentStatus::Ready || a.status == InstrumentStatus::Shadowed;
        rejected += ok ? 0 : 1;
        std::printf("%-12s %s %s  %s%s%s\n", status_word(a.status),
                    a.manifest.id.empty() ? a.manifest.name.c_str() : a.manifest.id.c_str(),
                    a.manifest.version.empty() ? "-" : a.manifest.version.c_str(),
                    a.folder.c_str(), a.status_text.empty() ? "" : "\n             ",
                    a.status_text.c_str());
        CHECK_MESSAGE(ok, a.manifest.name << ": " << a.status_text);
    }
    std::printf("%d of %d instruments accepted\n", L.count() - rejected, L.count());
    std::fflush(stdout);
    L.close_all();
    fs::remove_all(data, ec);
}
