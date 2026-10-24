#pragma once
#include <caliper/services/jobs_v1.h>
#include <chrono>
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
// (the guard is UI-thread-only, see crash_guard.h preconditions), so a fault
// in a job still takes the host down. A C++ exception does not: it is caught
// on the worker, recorded as the job's failure, and surfaced via views().
class JobSystem {
public:
    // Ctor + dtor are out-of-line: member jobs_ holds unique_ptr<Job> and Job
    // is only forward-declared here, so any inline special member that could
    // destroy jobs_ (ctor rollback, dtor) needs Job complete — which it is at
    // the .cpp definition point. ~JobSystem cancels + joins all threads
    // (bounded by kShutdownDeadline, see cancel_all_and_join).
    JobSystem();
    ~JobSystem();

    uint64_t submit(std::string label, CaliperJobFn fn, void* user);
    void     request_cancel(uint64_t id);
    bool     is_running(uint64_t id) const;
    bool     failed(uint64_t id) const;    // fn threw; message holds the what()
    float    progress_of(uint64_t id) const;

    struct JobView {
        uint64_t id = 0;
        std::string label;
        float progress = 0.0f;
        std::string message;
        bool running = false;
        bool failed = false;
    };
    std::vector<JobView> views() const;   // running jobs first, then recent

    // Drop finished entries from the list (the Jobs panel's "clear" button).
    // Finished threads are joined before their Job is destroyed; running jobs
    // are untouched.
    void clear_finished();

    // Join the threads of jobs whose fn has returned, keeping their entries
    // (so the panel still shows them). submit() and views() do this on the
    // host's behalf every call, so a finished worker never lingers unjoined
    // until someone presses "clear". Returns how many were joined.
    int reap_finished();

    // Cancel every job and wait for the workers to exit — but not forever: a
    // fn that ignores ctl->cancelled must not hang shutdown. Workers still
    // running at the deadline are reported on stderr and detached (their Job
    // is leaked on purpose; see the .cpp). Workers that do honor cancel are
    // joined for real, however long their cleanup takes within the deadline.
    static constexpr std::chrono::milliseconds kShutdownDeadline{5000};
    void cancel_all_and_join(std::chrono::milliseconds deadline = kShutdownDeadline);

private:
    struct Job;
    Job* find(uint64_t id) const;         // caller holds mutex_
    int  reap_finished_locked();          // caller holds mutex_
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Job>> jobs_;
    uint64_t next_id_ = 1;
};

} // namespace caliper_host
