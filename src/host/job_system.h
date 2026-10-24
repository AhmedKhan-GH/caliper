#pragma once
#include <caliper/services/jobs_v1.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caliper_host {

// The applet a guarded job runs for (submit_guarded). The loader makes one per
// applet instance and puts it in scope around every call into the instance
// (JobOwnerScope); a jobs.v1 submit made inside such a call carries it to the
// worker. A fault there is recorded here, and the loader quarantines the
// instance on its own thread before calling it again — a worker never touches
// loader state. Shared ownership: a worker that outlives the instance still
// writes somewhere valid.
class JobOwner {
public:
    // First fault wins (later ones are still reported per job on stderr).
    void record_fault(const std::string& label, const std::string& fault);
    bool faulted() const { return faulted_.load(std::memory_order_acquire); }
    // e.g. "background job \"train\": SIGSEGV (invalid memory access)"
    std::string fault() const;

private:
    mutable std::mutex mu_;
    std::string fault_;
    std::atomic<bool> faulted_{false};
};

// Puts `owner` in scope on this thread for jobs.v1 submissions; restores the
// previous one on exit (scopes nest). `owner` must outlive the scope — it is
// held by address, so a per-frame scope costs no refcount traffic.
class JobOwnerScope {
public:
    explicit JobOwnerScope(const std::shared_ptr<JobOwner>& owner);
    ~JobOwnerScope();
    JobOwnerScope(const JobOwnerScope&) = delete;
    JobOwnerScope& operator=(const JobOwnerScope&) = delete;

    // The owner a submission on this thread is attributed to; null outside
    // any scope (a thread the applet made itself, an embedder's own call).
    static std::shared_ptr<JobOwner> current();

private:
    const std::shared_ptr<JobOwner>* prev_;
};

// Thread-per-job v1 (PLATFORM.md §7.5), the pattern extracted from repnet's
// train_engine (atomic stop flag polled by the loop; state published under a
// mutex). Two ways in:
//   submit()          host-internal work: trusted, NOT crash-guarded, so a
//                     fault in it still takes the host down (fail loudly). A
//                     C++ exception does not: it is caught on the worker,
//                     recorded as the job's failure, surfaced via views().
//   submit_guarded()  caliper.jobs.v1 (applet code): fn runs under the crash
//                     guard, so a fault or an exception fails the job with the
//                     guard's reason, and the submitting applet's JobOwner (if
//                     one was in scope) is told.
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
    // Attributed to JobOwnerScope::current() on the calling thread.
    uint64_t submit_guarded(std::string label, CaliperJobFn fn, void* user);
    void     request_cancel(uint64_t id);
    bool     is_running(uint64_t id) const;
    bool     failed(uint64_t id) const;    // fn threw/faulted; message says why
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
    uint64_t start(std::string label, CaliperJobFn fn, void* user, bool guarded,
                   std::shared_ptr<JobOwner> owner);
    Job* find(uint64_t id) const;         // caller holds mutex_
    int  reap_finished_locked();          // caller holds mutex_
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Job>> jobs_;
    uint64_t next_id_ = 1;
};

} // namespace caliper_host
