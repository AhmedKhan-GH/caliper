// Test fixture: a hand-written epoch-2 descriptor (abi.h) that throws a C++
// exception INTO the host from a chosen entry point. Deliberately NOT the
// CALIPER_APPLET sugar — its thunks catch at the ABI boundary, as every
// well-behaved applet must (§6c) — because the crash guard's catch(...) exists
// for exactly the applet that doesn't. Built into ${CMAKE_BINARY_DIR}/test-applets,
// never the host's applets dir, so no card for it ever appears in the app.
#include <caliper/abi.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

// CALIPER_THROWING_WHERE names the entry point that throws: "create",
// "initialize", "frame" (the default) or "cleanup".
bool throws_in(const char* where) {
    const char* w = std::getenv("CALIPER_THROWING_WHERE");
    return std::strcmp(w ? w : "frame", where) == 0;
}

struct Instance { int frames = 0; };

void* create(void) {
    if (throws_in("create")) throw std::runtime_error("throwing_applet.create");
    return new Instance();
}
void destroy(void* self) { delete static_cast<Instance*>(self); }
bool initialize(void*, const CaliperHost*) {
    if (throws_in("initialize"))
        throw std::runtime_error("throwing_applet.initialize");
    return true;
}
void frame(void* self, const CaliperFrameInfo*) {
    static_cast<Instance*>(self)->frames++;
    if (throws_in("frame")) throw std::runtime_error("throwing_applet.frame");
}
void cleanup(void*) {
    if (throws_in("cleanup")) throw std::runtime_error("throwing_applet.cleanup");
}

const char* const kServices[] = {nullptr};

} // namespace

extern "C" CALIPER_EXPORT const CaliperAppletDescriptor*
caliper_applet_descriptor(void) {
    static const CaliperAppletDescriptor kDesc = {
        sizeof(CaliperAppletDescriptor),
        CALIPER_ABI_EPOCH,
        "dev.caliper.throwing",
        "0.1.0",
        "Throwing",
        "Test fixture: throws a C++ exception across the C ABI.",
        "Test",
        kServices,
        {sizeof(CaliperAppletAPI), &create, &destroy, &initialize, &frame,
         &cleanup},
    };
    return &kDesc;
}
