#pragma once

#include <memory>
#include <string>

namespace caliper_host { class HostRenderer; }

namespace caliper_live {

// Optional controls advertised by a script over its live request channel.
// Owns no model data beyond the metadata, output text and sampled overview.
class Inspector {
public:
    Inspector();
    ~Inspector();
    Inspector(const Inspector&) = delete;
    Inspector& operator=(const Inspector&) = delete;

    // Called on the frame thread. Socket work is bounded and nonblocking.
    void poll(const std::string& session, bool alive);
    void draw(float x, float y, float width, float height,
              caliper_host::HostRenderer* renderer = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace caliper_live
