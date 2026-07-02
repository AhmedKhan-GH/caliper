#pragma once
// Headless fake CaliperHost (PLATFORM.md §16 "fixture host"): TDD applets and
// sugar without launching UI. Vends log.v1 only; get_service returns NULL for
// everything else. ONE live fixture per process (C tables carry no user data,
// so the thunks route through a static active pointer).
#include <caliper/abi.h>
#include <caliper/services/log_v1.h>
#include <string>
#include <vector>

namespace caliper::testing {

class FixtureHost {
public:
    FixtureHost() {
        active_ = this;
        log_table_.struct_size = sizeof(CaliperLogV1);
        log_table_.log = &FixtureHost::log_thunk;
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

private:
    static void log_thunk(CaliperLogLevel, const char* msg) {
        if (active_ && msg) active_->lines_.emplace_back(msg);
    }
    static const void* get_service_thunk(const CaliperHost*, const char* id) {
        if (active_ && id && std::string(id) == CALIPER_LOG_V1)
            return &active_->log_table_;
        return nullptr;
    }
    inline static FixtureHost* active_ = nullptr;
    CaliperHost host_{};
    CaliperLogV1 log_table_{};
    std::string data_dir_ = "/tmp/caliper-fixture-data";
    std::vector<std::string> lines_;
};

} // namespace caliper::testing
