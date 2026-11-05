#pragma once
// The Trajectories pane: each token of a traced run as a path up the lens layers, drawn as the run streams in.
// The host fits one plane on the prompt (a PCA shared by all lens layers, of residuals centred on the prompt) and
// places every position's residual after each layer in it as the position arrives; the height is the layer. A
// path turns teal from the layer where the Jacobian lens reads the model's own next token and keeps reading it to
// the last layer; hovering a path lists what the lens read at every layer.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace caliper_host { class HostRenderer; }

namespace caliper_live {

struct TraceRun;

// Per position, the first lens layer (an index into run.layers) from which the J-lens's top token equals the
// model's own top output and stays equal to the last lens layer; -1 where it never settles on it.
std::vector<int> lock_layers(const TraceRun& run);

class TrajectoryView {
public:
    TrajectoryView();
    ~TrajectoryView();
    // Follows run, the run on show, as it grows; cheap when nothing changed, so it is called every frame.
    void show(const TraceRun& run);
    std::string subtitle() const;
    // Draws into the rest of the current window; word(id) gives a token's text.
    void draw(caliper_host::HostRenderer* renderer, const std::function<std::string(int32_t)>& word);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace caliper_live
