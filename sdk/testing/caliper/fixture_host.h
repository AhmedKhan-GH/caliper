#pragma once
// Headless fake CaliperHost (docs/explanations/testing.md, "fixture host"): TDD applets and
// sugar without launching UI. Vends log.v1, and ui.v1_1 for its dock hints
// only: that table has no UI contexts (its context getters return NULL), and
// it records every dock_hint() call, so a test can assert the roles an applet
// declares (dock_hints(), dock_role()). get_service returns NULL for
// everything else, caliper.ui.v1 included, so ui::connect() leaves ImGui
// alone; a test that needs real contexts provides its own ui.v1 table. ONE
// live fixture per process (C tables carry no user data, so the thunks route
// through a static active pointer).
#include <caliper/abi.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/ui_v1_1.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace caliper::testing {

class FixtureHost {
public:
    FixtureHost() {
        active_ = this;
        log_table_.struct_size = sizeof(CaliperLogV1);
        log_table_.log = &FixtureHost::log_thunk;
        ui_table_.struct_size = sizeof(CaliperUiV1_1);
        ui_table_.imgui_context = [] { return (ImGuiContext*)nullptr; };
        ui_table_.implot_context = [] { return (ImPlotContext*)nullptr; };
        ui_table_.implot3d_context = [] { return (ImPlot3DContext*)nullptr; };
        ui_table_.imgui_allocators = [](CaliperImGuiAllocFn* a, CaliperImGuiFreeFn* f,
                                        void** u) {
            if (a) *a = nullptr;
            if (f) *f = nullptr;
            if (u) *u = nullptr;
        };
        ui_table_.dock_hint = &FixtureHost::dock_hint_thunk;
        host_.struct_size = sizeof(CaliperHost);
        host_.abi_epoch = CALIPER_ABI_EPOCH;
        host_.host_version = (0u << 16) | (6u << 8) | 0u;
        host_.applet_data_dir = data_dir_.c_str();
        host_.get_service = &FixtureHost::get_service_thunk;
    }
    ~FixtureHost() { if (active_ == this) active_ = nullptr; }

    const CaliperHost* host() const { return &host_; }
    const std::vector<std::string>& log_lines() const { return lines_; }
    bool log_contains(const std::string& needle) const {
        for (const auto& l : lines_)
            if (l.find(needle) != std::string::npos) return true;
        return false;
    }

    // Inject any service table (e.g. a fake jobs.v1) for wrapper tests. A
    // provided table wins over the fixture's own ui.v1_1.
    void provide(const char* id, const void* table) {
        provided_.emplace_back(id, table);
    }

    // Every dock_hint() call, in order: (window title, role).
    const std::vector<std::pair<std::string, uint32_t>>& dock_hints() const {
        return hints_;
    }
    // The role of `window_title` as the host sees it, which is the first one
    // hinted for that title (later hints change nothing); -1 when it was
    // never hinted.
    int64_t dock_role(const std::string& window_title) const {
        for (const auto& [title, role] : hints_)
            if (title == window_title) return role;
        return -1;
    }

private:
    static void log_thunk(CaliperLogLevel, const char* msg) {
        if (active_ && msg) active_->lines_.emplace_back(msg);
    }
    static void dock_hint_thunk(const char* title, uint32_t role) {
        if (active_ && title) active_->hints_.emplace_back(title, role);
    }
    static const void* get_service_thunk(const CaliperHost*, const char* id) {
        if (!active_ || !id) return nullptr;
        if (std::string(id) == CALIPER_LOG_V1) return &active_->log_table_;
        for (const auto& [pid, table] : active_->provided_)
            if (pid == id) return table;
        if (std::string(id) == CALIPER_UI_V1_1) return &active_->ui_table_;
        return nullptr;
    }
    inline static FixtureHost* active_ = nullptr;
    CaliperHost host_{};
    CaliperLogV1 log_table_{};
    CaliperUiV1_1 ui_table_{};
    std::string data_dir_ = "/tmp/caliper-fixture-data";
    std::vector<std::string> lines_;
    std::vector<std::pair<std::string, uint32_t>> hints_;
    std::vector<std::pair<std::string, const void*>> provided_;
};

} // namespace caliper::testing
