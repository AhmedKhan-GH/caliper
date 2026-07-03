# Your first ML applet

The ladder from an empty window to a live machine-learning demonstration,
one capability per stage. We build **SineScope**: a tiny MLP that learns
`y = sin(x)` in front of you — live loss curve, the model's prediction
bending toward the target in real time, and its weight matrix rendered as a
GPU heatmap. Synthetic data keeps it self-contained (no downloads); the
last two stages show where real datasets and the remaining services attach.

Prereqs: [Development basics](development-basics.md) (the mental model),
[Your first applet](first-applet.md) (the hello walkthrough). The finished
staircase — same patterns, full scale — is `applets/embed_scope/`, with the
[cookbook](../howto/ml-applet-cookbook.md) as its field guide.

## Stage 0 — the skeleton

Start from the smallest applet
([Development basics](development-basics.md#the-smallest-complete-applet-in-full)),
renamed to `sine_scope`. Manifest: require what training needs, mark the
bridge optional so the applet still runs without it:

```toml
[services]
required = ["caliper.ui.v1", "caliper.log.v1",
            "caliper.jobs.v1", "caliper.device.v1"]
optional = ["caliper.tensor_bridge.v1", "caliper.metrics.v1"]
```

CMake: hello's file plus the torch lines (this is the *entire* ML build
delta — copy it exactly):

```cmake
add_library(sine_scope SHARED sine_scope.cpp)
target_link_libraries(sine_scope PRIVATE
    caliper::sdk caliper::ui_stack
    "${TORCH_LIBRARIES}")                  # vendored third_party/libtorch
target_compile_definitions(sine_scope PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(sine_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(APPLE)   # let the dylib find libtorch at runtime
    set_target_properties(sine_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()
add_custom_command(TARGET sine_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/sine_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/sine_scope.caliper.toml)
```

## Stage 1 — state + services: the spine before any ML

Every live applet has the same skeleton state: service wrappers, a mutex,
published vectors, a generation counter.

```cpp
#include <caliper/caliper.hpp>
#include <caliper/adapters/torch.hpp>
#include <torch/torch.h>
#include <atomic>
#include <mutex>
#include <vector>

struct SineState {
    caliper::Host*   host = nullptr;
    caliper::Jobs    jobs;
    caliper::Device  device;
    caliper::Bridge  bridge;      // optional — probed, may be falsy
    caliper::Metrics metrics;     // optional

    uint64_t job_id = 0;

    // -- cross-thread, under mtx --
    std::mutex mtx;
    std::vector<float> loss_hist;
    std::vector<float> pred_ys;       // model(x) on a fixed grid
    torch::Tensor      disp_w;        // weight display tensor (handle)
    float              w_max = 1e-6f;
    uint64_t           gen = 0;

    // -- frame-thread only --
    uint64_t seen_gen = 0, tex_gen = 0;
    CaliperTextureId w_tex = 0;
    bool follow = true;
};

bool SineScope::on_init(caliper::Host& host) {
    auto* st = s_.get();
    st->host    = &host;
    st->jobs    = caliper::Jobs(host);      // required — manifest enforced
    st->device  = caliper::Device(host);
    st->bridge  = caliper::Bridge(host);    // optional — falsy if absent
    st->metrics = caliper::Metrics(host);
    host.log_info("sine_scope: on_init");
    return true;
}
```

The manifest already guaranteed the required services exist — no null
checks needed for jobs/device. The optional ones are **falsy-inert**: you
can call them unconditionally and they no-op when absent, but good demos
*show* the degradation (Stage 4).

## Stage 2 — background compute: the training job

Never compute on the frame thread. A job is a free function
`void(void*, const CaliperJobControl*)` — submit it, poll it, cancel it:

```cpp
void train_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<SineState*>(user);

    // device.v1 -> torch device: METAL means MPS on Apple.
    auto d = caliper::Device::query(*st->host);
    torch::Device dev = (d.kind == CALIPER_DEV_METAL) ? torch::kMPS
                                                      : torch::kCPU;

    // Synthetic dataset: 256 points of y = sin(x) on [-pi, pi].
    auto X = torch::linspace(-M_PI, M_PI, 256, dev).unsqueeze(1);
    auto Y = torch::sin(X);

    auto model = torch::nn::Sequential(
        torch::nn::Linear(1, 32), torch::nn::Tanh(),
        torch::nn::Linear(32, 1));
    model->to(dev);
    torch::optim::Adam opt(model->parameters(), 1e-2);

    for (int step = 0; step < 2000; step++) {
        if (ctl->cancelled(ctl)) return;            // ≤100ms contract
        opt.zero_grad();
        auto loss = torch::mse_loss(model->forward(X), Y);
        loss.backward();
        opt.step();
        publish(st, model, X, loss.item<float>());  // Stage 3
        ctl->progress(ctl, step / 2000.f, "fitting sin(x)");
    }
}

// in draw_ui: submit on click, show progress, offer cancel
const bool running = st->job_id && st->jobs.is_running(st->job_id);
if (!running) {
    if (ImGui::Button("Train"))
        st->job_id = st->jobs.submit("sine_scope: fit", &train_job, st);
} else {
    if (ImGui::Button("Cancel")) st->jobs.cancel(st->job_id);
    ImGui::SameLine();
    ImGui::ProgressBar(st->jobs.progress_of(st->job_id), {-1, 0});
}
```

You already get for free: the job appears in the host's **jobs tray**, and
Cancel genuinely stops it (the per-step check is the contract).

## Stage 3 — publish and plot: watching it learn

The worker publishes owned copies + tensor handles under the mutex; the
frame consumes when the generation moves. This is the
[threading spine](../howto/ml-applet-cookbook.md#1-the-threading-spine):

```cpp
void publish(SineState* st, torch::nn::Sequential& model,
             const torch::Tensor& X, float loss) {
    torch::NoGradGuard ng;
    auto pred = model->forward(X).to(torch::kCPU).contiguous();  // (256,1)
    // weight display tensor: (32,1) -> 8x4 grid -> x16 blocks, ON DEVICE
    auto w = model[0]->as<torch::nn::Linear>()->weight.detach()  // (32,1)
                 .reshape({8, 4})
                 .repeat_interleave(16, 0).repeat_interleave(16, 1)
                 .contiguous();
    float wmax = w.abs().max().item<float>();

    std::lock_guard<std::mutex> lk(st->mtx);
    st->loss_hist.push_back(loss);
    st->pred_ys.assign(pred.data_ptr<float>(), pred.data_ptr<float>() + 256);
    st->disp_w = w;                              // handle swap, no copy
    st->w_max  = std::max(wmax, 1e-6f);
    st->gen++;
}
```

Frame side — the prediction curve bending toward the target is the "it's
alive" moment, and it's just two `PlotLine`s:

```cpp
// copy out under the mutex (gen-gated), then:
ImGui::Checkbox("follow", &st->follow);          // viewport policy, §6
const ImPlotAxisFlags f = st->follow ? ImPlotAxisFlags_AutoFit : 0;
if (ImPlot::BeginPlot("fit", {-1, 240})) {
    ImPlot::SetupAxes("x", "y", 0, 0);
    ImPlot::PlotLine("target sin(x)", xs.data(), target.data(), 256);
    if (!pred.empty())
        ImPlot::PlotLine("model(x)", xs.data(), pred.data(), 256);
    ImPlot::EndPlot();
}
if (ImPlot::BeginPlot("loss", {-1, 160})) {
    ImPlot::SetupAxes("step", "MSE", f, f);
    if (!loss.empty()) ImPlot::PlotLine("mse", loss.data(), (int)loss.size());
    ImPlot::EndPlot();
}
```

## Stage 4 — the bridge: a tensor as pixels

The platform's signature move: the weight matrix as a colormapped texture,
device-resident on Metal. Frame thread only, gen-gated, released in cleanup:

```cpp
if (st->bridge && lgen != st->tex_gen && disp_w.defined()) {
    if (st->w_tex) st->bridge.release_texture(st->w_tex);
    auto ct = caliper::adapters::to_tensor(disp_w);       // handle -> ABI type
    st->w_tex = ct ? st->bridge.texture_from_tensor_mapped(
                         &*ct, CALIPER_CMAP_RDBU, -w_max, w_max) : 0;
    st->tex_gen = lgen;
}
if (st->w_tex)
    ImGui::Image(caliper::Bridge::imtex(st->w_tex), ImVec2(128, 256));
else
    ImGui::TextDisabled("tensor_bridge.v1 absent (ok) — no weight heatmap");
```

That last line is the degradation idiom: optional services fail *visibly
and politely*, never silently and never fatally.

Cleanup grows its symmetric duties:

```cpp
void SineScope::on_cleanup() {
    if (st->job_id) {
        st->jobs.cancel(st->job_id);
        for (int i = 0; i < 1000 && st->jobs.is_running(st->job_id); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->w_tex) st->bridge.release_texture(st->w_tex);
}
```

**Run it:** `cmake --build build --target sine_scope`, then
`CALIPER_AUTOLAUNCH=dev.example.sine-scope ./build/caliper`. You should see
the flat line snap into a sine within seconds while the heatmap's blocks
reorganize.

## Stage 5 — real data instead of synthetic

Everything above holds; only acquisition changes. The rules
([cookbook §8](../howto/ml-applet-cookbook.md#8-data-acquisition-the-download-recipe)):
fetch **inside the job**, cache in `host.data_dir()`, write atomically
(`.tmp` + rename), self-heal corrupt caches, make the transfer cancellable
via curl's progress callback, and add `CURL::libcurl`/`ZLIB::ZLIB` to the
CMake links. The exemplar's `ensure_dataset` + `mnist_path` are the
copy-paste source — including the sibling-cache trick (reuse another
applet's MNIST download rather than duplicating 11 MB).

## Stage 6 — the rest of the platform, one line each

Each remaining service is a small delta from here, and the exemplar shows
all of them finished:

- **`metrics.v1`** — persistence + the Runs dashboard for two lines:
  `run = metrics.begin_run("sine", "mlp32")` once, then
  `metrics.scalar(run, "train/loss", step, loss)` in the loop. Your run now
  survives restarts and plots in the host's Runs window.
- **`artifacts.v1`** — Save/Load buttons so a trained model outlives the
  process ([cookbook §9](../howto/ml-applet-cookbook.md#9-checkpoints-via-artifactsv1)).
  Load-then-eval *without retraining* is the demo magic.
- **`data.v1`** — when your published state is genuinely tabular, register
  it and ask SQL questions
  ([cookbook §10](../howto/ml-applet-cookbook.md#10-sql-over-live-data-datav1)).

When all of these feel natural, read `applets/embed_scope/` end to end —
it is exactly this tutorial's patterns at full scale: a real dataset, a 3-D
learned embedding, per-step device pulls, and all eight services in ~900
annotated lines.
