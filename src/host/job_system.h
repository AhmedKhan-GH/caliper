#pragma once
#include <caliper/services/jobs_v1.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caliper_host {

// Thread-per-job v1 (PLATFORM.md §7.5), the pattern extracted from repnet's
// train_engine (atomic stop flag polled by the loop; state published under a
// mutex). Job fns are trusted code on worker threads — NOT crash-guarded
// (the guard is UI-thread-only, see crash_guard.h preconditions).
class JobSystem {
public:
    // Ctor + dtor are out-of-line: member jobs_ holds unique_ptr<Job> and Job
    // is only forward-declared here, so any inline special member that could
    // destroy jobs_ (ctor rollback, dtor) needs Job complete — which it is at
    // the .cpp definition point. ~JobSystem cancels + joins all threads.
    JobSystem();
    ~JobSystem();

    uint64_t submit(std::string label, CaliperJobFn fn, void* user);
    void     request_cancel(uint64_t id);
    bool     is_running(uint64_t id) const;
    float    progress_of(uint64_t id) const;

    struct JobView {
        uint64_t id = 0;
        std::string label;
        float progress = 0.0f;
        std::string message;
        bool running = false;
    };
    std::vector<JobView> views() const;   // running jobs first, then recent

    void cancel_all_and_join();

private:
    struct Job;
    Job* find(uint64_t id) const;         // caller holds mutex_
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Job>> jobs_;
    uint64_t next_id_ = 1;
};

} // namespace caliper_host
