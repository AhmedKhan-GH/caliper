// The kept steps of the open spec and the time cursor
// (caliper_builder/session/history.h). Not built yet: nothing is kept, and
// the cursor stays live at step 0.
#include <caliper_builder/session/history.h>

#include <mutex>

namespace caliper_builder {

struct StepHistory::Impl {
    mutable std::mutex mutex;
    CursorState cursor;
};

StepHistory::StepHistory() : impl_(std::make_unique<Impl>()) {}
StepHistory::~StepHistory() = default;

void StepHistory::configure(const spec::Spec& spec) { (void)spec; }

void StepHistory::clear() {}

void StepHistory::append(const std::string& catalog_id, std::int64_t step, const TensorData& value) {
    (void)catalog_id;
    (void)step;
    (void)value;
}

std::optional<std::pair<std::int64_t, TensorData>> StepHistory::at(const std::string& source,
                                                                    std::int64_t step) const {
    (void)source;
    (void)step;
    return std::nullopt;
}

std::vector<std::pair<std::int64_t, TensorData>> StepHistory::window(const std::string& source, std::int64_t lo,
                                                                     std::int64_t hi) const {
    (void)source;
    (void)lo;
    (void)hi;
    return {};
}

std::vector<KeptSource> StepHistory::kept() const { return {}; }

std::vector<std::int64_t> StepHistory::kept_steps() const { return {}; }

CursorState StepHistory::cursor() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->cursor;
}

CursorState StepHistory::set_cursor(const CursorState& wanted) {
    (void)wanted;
    return cursor();
}

CursorState StepHistory::step_cursor(std::int64_t count) {
    (void)count;
    return cursor();
}

void StepHistory::tick(double seconds) { (void)seconds; }

nlohmann::json cursor_json(const CursorState& c) {
    return {{"step", c.step},       {"first", c.first},     {"latest", c.latest},
            {"live", c.live},       {"playing", c.playing}, {"speed", c.speed}};
}

}  // namespace caliper_builder
