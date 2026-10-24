#include "job_system.h"
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <exception>

namespace caliper_host {

// The C control block handed to job fns. ctl is the FIRST member of a
// standard-layout struct, so the thunks recover the block from the ctl
// pointer with a cast — the same first-member idiom the fixture host uses.
struct ControlBlock {
    CaliperJobControl ctl{};
    std::atomic<bool> cancelled{false};
    std::atomic<float> progress{0.0f};
    std::mutex msg_mutex;
    std::string message;
};
static_assert(offsetof(ControlBlock, ctl) == 0);

struct JobSystem::Job {
    uint64_t id = 0;
    std::string label;
    ControlBlock block;
    std::atomic<bool> running{true};
    std::atomic<bool> failed{false};
    std::thread thread;
};

JobSystem::JobSystem() = default;
JobSystem::~JobSystem() { cancel_all_and_join(); }

namespace {
bool ctl_cancelled(const CaliperJobControl* ctl) {
    return reinterpret_cast<const ControlBlock*>(ctl)->cancelled.load();
}
void ctl_progress(const CaliperJobControl* ctl, float frac, const char* msg) {
    auto* b = const_cast<ControlBlock*>(
        reinterpret_cast<const ControlBlock*>(ctl));
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    b->progress.store(frac);
    if (msg) {
        std::lock_guard<std::mutex> lk(b->msg_mutex);
        b->message = msg;
    }
}
} // namespace

uint64_t JobSystem::submit(std::string label, CaliperJobFn fn, void* user) {
    if (!fn) return 0;
    std::lock_guard<std::mutex> lk(mutex_);
    reap_finished_locked();
    auto job = std::make_unique<Job>();
    job->id = next_id_++;
    job->label = std::move(label);
    job->block.ctl.struct_size = sizeof(CaliperJobControl);
    job->block.ctl.cancelled = &ctl_cancelled;
    job->block.ctl.progress = &ctl_progress;
    Job* raw = job.get();
    job->thread = std::thread([raw, fn, user] {
        // A fn that threw: flag it, put the reason where the Jobs panel
        // already looks (the message), and say so once on stderr.
        auto fail = [raw](const char* what) {
            {
                std::lock_guard<std::mutex> lk(raw->block.msg_mutex);
                raw->block.message = std::string("failed: ") + what;
            }
            raw->failed.store(true);
            std::fprintf(stderr, "[jobs] job #%" PRIu64 " \"%s\" failed: %s\n",
                         raw->id, raw->label.c_str(), what);
        };
        // Trusted code, not signal-guarded (§15) — but an exception escaping
        // a std::thread is std::terminate for the whole host, so that much is
        // contained here: the job fails, the host lives.
        try {
            fn(user, &raw->block.ctl);
        } catch (const std::exception& e) {
            fail(e.what());
        } catch (...) {
            fail("non-std exception");
        }
        raw->running.store(false);
    });
    jobs_.push_back(std::move(job));
    return raw->id;
}

JobSystem::Job* JobSystem::find(uint64_t id) const {
    for (auto& j : jobs_)
        if (j->id == id) return j.get();
    return nullptr;
}

void JobSystem::request_cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (Job* j = find(id)) j->block.cancelled.store(true);
}

bool JobSystem::is_running(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    Job* j = find(id);
    return j && j->running.load();
}

bool JobSystem::failed(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    Job* j = find(id);
    return j && j->failed.load();
}

float JobSystem::progress_of(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    Job* j = find(id);
    return j ? j->block.progress.load() : 0.0f;
}

std::vector<JobSystem::JobView> JobSystem::views() const {
    std::lock_guard<std::mutex> lk(mutex_);
    // The panel calls this every frame: the natural place to reap. The const
    // is about the job list, not the thread handles.
    const_cast<JobSystem*>(this)->reap_finished_locked();
    std::vector<JobView> out;
    out.reserve(jobs_.size());
    for (auto& j : jobs_) {
        JobView v;
        v.id = j->id;
        v.label = j->label;
        v.progress = j->block.progress.load();
        {
            std::lock_guard<std::mutex> mk(j->block.msg_mutex);
            v.message = j->block.message;
        }
        v.running = j->running.load();
        v.failed = j->failed.load();
        if (v.running) out.insert(out.begin(), v); else out.push_back(v);
    }
    return out;
}

int JobSystem::reap_finished_locked() {
    // running==false is the worker's last store, so these joins only wait for
    // the thread to fall off the end of its lambda — instant.
    int n = 0;
    for (auto& j : jobs_)
        if (!j->running.load() && j->thread.joinable()) {
            j->thread.join();
            n++;
        }
    return n;
}

int JobSystem::reap_finished() {
    std::lock_guard<std::mutex> lk(mutex_);
    return reap_finished_locked();
}

void JobSystem::clear_finished() {
    std::vector<std::unique_ptr<Job>> done;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = jobs_.begin();
        while (it != jobs_.end()) {
            if (!(*it)->running.load()) {
                done.push_back(std::move(*it));
                it = jobs_.erase(it);
            } else ++it;
        }
    }
    // Join outside the lock: the fn has already returned (running==false),
    // so these joins are instant (usually already reaped) — but join we must
    // before ~Job.
    for (auto& j : done)
        if (j->thread.joinable()) j->thread.join();
}

void JobSystem::cancel_all_and_join(std::chrono::milliseconds deadline) {
    std::vector<std::unique_ptr<Job>> taken;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto& j : jobs_) j->block.cancelled.store(true);
        taken.swap(jobs_);
    }
    // Poll the running flags, joining each worker the moment it exits. A
    // worker that honors cancel is joined for real (its post-cancel cleanup
    // finishes before we return); one that ignores it only costs us the
    // deadline.
    const auto until = std::chrono::steady_clock::now() + deadline;
    for (;;) {
        bool any_running = false;
        for (auto& j : taken) {
            if (!j) continue;
            if (j->running.load()) { any_running = true; continue; }
            if (j->thread.joinable()) j->thread.join();
            j.reset();
        }
        if (!any_running) return;
        if (std::chrono::steady_clock::now() >= until) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Stragglers: say which, detach, and LEAK their Job on purpose. The
    // detached worker keeps reading its ControlBlock through ctl; destroying
    // the Job here would be a use-after-free on a thread we no longer control.
    // The leak is bounded by the process lifetime, which is where we are.
    int n = 0;
    std::string names;
    for (auto& j : taken) {
        if (!j) continue;
        if (n++) names += ", ";
        names += "\"" + j->label + "\" (#" + std::to_string(j->id) + ")";
        if (j->thread.joinable()) j->thread.detach();
        (void)j.release();
    }
    std::fprintf(stderr,
                 "[jobs] %d job(s) still running %lld ms after cancel; "
                 "detaching: %s\n",
                 n, static_cast<long long>(deadline.count()), names.c_str());
}

} // namespace caliper_host
