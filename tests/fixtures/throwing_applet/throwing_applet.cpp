// Test fixture: a hand-written epoch-2 descriptor (abi.h) that throws a C++
// exception INTO the host from a chosen entry point. Deliberately NOT the
// CALIPER_APPLET sugar — its thunks catch at the ABI boundary, as every
// well-behaved applet must (§6c) — because the crash guard's catch(...) exists
// for exactly the applet that doesn't. Built into ${CMAKE_BINARY_DIR}/test-applets,
// never the host's applets dir, so no card for it ever appears in the app.
#include <caliper/abi.h>
#include <caliper/caliper.hpp>   // caliper::ui::connect only (and the UI headers)
#include <caliper/services/jobs_v1.h>
#include <caliper/services/log_v1.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

// CALIPER_THROWING_WHERE names the entry point that throws: "create",
// "initialize", "frame" (the default) or "cleanup". Two more modes fail in a
// background job instead: on its first frame() the applet submits a
// caliper.jobs.v1 job that throws ("job") or writes through null
// ("job-fault"), and the entry point itself returns normally. "frame-fault"
// writes through null in frame() itself. "frame-imgui" and
// "frame-imgui-fault" fail in frame() the same two ways, but only after
// opening a window, a plot and a pile of pushes on the host's UI contexts
// (caliper.ui.v1), all left open. Any other value ("none") fails nowhere.
bool throws_in(const char* where) {
    const char* w = std::getenv("CALIPER_THROWING_WHERE");
    return std::strcmp(w ? w : "frame", where) == 0;
}

struct Instance {
    int frames = 0;
    const CaliperJobsV1* jobs = nullptr;   // NULL when the host vends none
    const CaliperLogV1* log = nullptr;     // NULL when the host vends none
    bool ui = false;                       // contexts connected (ui.v1)
};

void say(const Instance* inst, const char* msg) {
    if (inst->log && inst->log->log) inst->log->log(CALIPER_LOG_INFO, msg);
}

void write_through_null() {
    volatile int* p = nullptr;
    *p = 1;
}

// One of nearly everything the host has to unwind after a fault, begun and
// left open: a window holding an ID, a style colour and var, a font, a group,
// a disabled block and a child window; inside the child a table, a tree node,
// a 2-D plot under ImPlot style and colormap pushes and a 3-D plot under
// ImPlot3D's. Sized so neither plot is clipped (each must really be open).
void open_ui_and_leave_it() {
    ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f));
    ImGui::SetNextWindowSize(ImVec2(720.0f, 600.0f));
    ImGui::Begin("Throwing: half-finished");
    ImGui::PushID("unfinished");
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0xe8, 0xe0, 0xd0, 0xff));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 4.0f));
    ImGui::PushFont(nullptr, 18.0f);
    ImGui::BeginGroup();
    ImGui::BeginDisabled();
    ImGui::BeginChild("plots", ImVec2(0.0f, 0.0f));
    if (ImGui::BeginTable("cells", 1)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("node", ImGuiTreeNodeFlags_DefaultOpen);
        ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32(0x12, 0x11, 0x0f, 0xff));
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotBorderSize, 2.0f);
        ImPlot::PushColormap(ImPlotColormap_Greys);
        static const double xs[] = {0.0, 1.0, 2.0};
        if (ImPlot::BeginPlot("2-D", ImVec2(320.0f, 200.0f)))
            ImPlot::PlotLine("line", xs, xs, 3);
        ImPlot3D::PushStyleColor(ImPlot3DCol_FrameBg, IM_COL32(0x12, 0x11, 0x0f, 0xff));
        ImPlot3D::PushStyleVar(ImPlot3DStyleVar_LineWeight, 2.0f);
        ImPlot3D::PushColormap(ImPlot3DColormap_Greys);
        static const float ps[] = {0.0f, 1.0f, 2.0f};
        if (ImPlot3D::BeginPlot("3-D", ImVec2(320.0f, 200.0f)))
            ImPlot3D::PlotLine("line", ps, ps, ps, 3);
    }
}

void job_throws(void*, const CaliperJobControl*) {
    throw std::runtime_error("throwing_applet.job");
}
void job_faults(void*, const CaliperJobControl*) {
    volatile int* p = nullptr;
    *p = 1;
}

void* create(void) {
    if (throws_in("create")) throw std::runtime_error("throwing_applet.create");
    return new Instance();
}
// destroy() and cleanup() say so through log.v1, so a test can see a
// restart release the old instance.
void destroy(void* self) {
    auto* inst = static_cast<Instance*>(self);
    say(inst, "throwing.destroy");
    delete inst;
}
bool initialize(void* self, const CaliperHost* host) {
    auto* inst = static_cast<Instance*>(self);
    if (host && host->get_service) {
        inst->jobs = static_cast<const CaliperJobsV1*>(
            host->get_service(host, CALIPER_JOBS_V1));
        inst->log = static_cast<const CaliperLogV1*>(
            host->get_service(host, CALIPER_LOG_V1));
        inst->ui = caliper::ui::connect(host);
    }
    // After the lookups, so an instance whose initialize() threw can still
    // say when it is destroyed.
    if (throws_in("initialize"))
        throw std::runtime_error("throwing_applet.initialize");
    say(inst, "throwing.initialize");
    return true;
}
void frame(void* self, const CaliperFrameInfo*) {
    auto* inst = static_cast<Instance*>(self);
    if (inst->frames++ == 0 && inst->jobs) {
        if (throws_in("job"))
            inst->jobs->submit("throwing_applet: job", &job_throws, nullptr);
        if (throws_in("job-fault"))
            inst->jobs->submit("throwing_applet: job-fault", &job_faults, nullptr);
    }
    if (throws_in("frame")) throw std::runtime_error("throwing_applet.frame");
    if (throws_in("frame-fault")) write_through_null();
    const bool imgui_throw = throws_in("frame-imgui");
    const bool imgui_fault = throws_in("frame-imgui-fault");
    if ((imgui_throw || imgui_fault) && inst->ui) {
        open_ui_and_leave_it();
        if (imgui_throw) throw std::runtime_error("throwing_applet.frame-imgui");
        write_through_null();
    }
}
void cleanup(void* self) {
    say(static_cast<Instance*>(self), "throwing.cleanup");
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
