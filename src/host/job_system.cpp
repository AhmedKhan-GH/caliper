#include "job_system.h"
#include <atomic>

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
    auto job = std::make_unique<Job>();
    job->id = next_id_++;
    job->label = std::move(label);
    job->block.ctl.struct_size = sizeof(CaliperJobControl);
    job->block.ctl.cancelled = &ctl_cancelled;
    job->block.ctl.progress = &ctl_progress;
    Job* raw = job.get();
    job->thread = std::thread([raw, fn, user] {
        fn(user, &raw->block.ctl);      // trusted code, unguarded (§15)
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

float JobSystem::progress_of(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    Job* j = find(id);
    return j ? j->block.progress.load() : 0.0f;
}

std::vector<JobSystem::JobView> JobSystem::views() const {
    std::lock_guard<std::mutex> lk(mutex_);
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
        if (v.running) out.insert(out.begin(), v); else out.push_back(v);
    }
    return out;
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
    // so these joins are instant — but join we must before ~Job.
    for (auto& j : done)
        if (j->thread.joinable()) j->thread.join();
}

void JobSystem::cancel_all_and_join() {
    std::vector<std::unique_ptr<Job>> taken;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto& j : jobs_) j->block.cancelled.store(true);
        taken.swap(jobs_);
    }
    for (auto& j : taken)
        if (j->thread.joinable()) j->thread.join();
}

} // namespace caliper_host
