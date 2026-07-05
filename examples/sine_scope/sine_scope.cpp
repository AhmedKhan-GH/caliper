// SineScope — the "Your first ML applet" tutorial, assembled and buildable.
// docs/wiki/tutorials/first-ml-applet.md embeds the marked sections below
// VERBATIM (pymdownx.snippets sections), so the tutorial and this source
// cannot drift: the strict docs build fails if they do.
//
// A tiny two-layer net learns y = sin(x) live: loss curve, the prediction
// bending toward the target, and the first layer's weights as a heatmap.
// Synthetic data keeps it self-contained. The grown-up sibling is
// applets/embed_scope/ (real dataset, 3-D embeddings, all 8 services).
#include <caliper/caliper.hpp>
#include <caliper/adapters/torch.hpp>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace sinescope {

// --8<-- [start:model]
// Two layers are enough to bend a line into a sine.
struct SineNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    SineNetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(1, 32));
        fc2 = register_module("fc2", torch::nn::Linear(32, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        return fc2->forward(torch::tanh(fc1->forward(x)));
    }
};
TORCH_MODULE(SineNet);
// --8<-- [end:model]

// --8<-- [start:state]
// The spine every live applet shares: service wrappers, one mutex,
// published copies, a generation counter (cookbook §1).
struct SineState {
    caliper::Host*   host = nullptr;
    caliper::Jobs    jobs;        // required (manifest-enforced)
    caliper::Bridge  bridge;      // optional — falsy-inert when absent

    uint64_t job_id = 0;

    // -- cross-thread, under mtx --
    std::mutex mtx;
    std::vector<float> loss_hist;
    std::vector<float> pred_ys;   // model(x) on the fixed 256-point grid
    torch::Tensor      disp_w;    // weight display tensor (handle swap)
    float              w_max = 1e-6f;
    uint64_t           gen = 0;   // bumped per publish (0 = none yet)

    // -- frame-thread only --
    uint64_t seen_gen = 0, tex_gen = 0;
    CaliperTextureId w_tex = 0;
    bool follow = true;
};
// --8<-- [end:state]

// --8<-- [start:publish]
// WORKER side: compute outside the lock, swap inside it, bump the gen.
void publish(SineState* st, SineNet& model, const torch::Tensor& X,
             float loss) {
    torch::NoGradGuard ng;
    auto pred = model->forward(X).to(torch::kCPU).contiguous();   // (256,1)
    // First-layer weights (32,1) -> 8x4 grid -> x16 hard blocks (cookbook
    // §4), staged to CPU for tutorial simplicity — the exemplar shows the
    // zero-copy device pull (cookbook §3).
    auto w = model->fc1->weight.detach()
                 .reshape({8, 4})
                 .repeat_interleave(16, 0).repeat_interleave(16, 1)
                 .to(torch::kCPU).contiguous();                   // (128,64)
    const float wmax = w.abs().max().item<float>();

    std::lock_guard<std::mutex> lk(st->mtx);
    st->loss_hist.push_back(loss);
    st->pred_ys.assign(pred.data_ptr<float>(),
                       pred.data_ptr<float>() + 256);
    st->disp_w = w;                       // handle swap — no data copy
    st->w_max  = std::max(wmax, 1e-6f);
    st->gen++;
}
// --8<-- [end:publish]

// --8<-- [start:job]
// The training job: a plain function the host runs on a worker thread.
void train_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<SineState*>(user);

    // device.v1 -> torch device: METAL means MPS on Apple, CUDA on Windows/Linux.
    auto d = caliper::Device::query(*st->host);
    torch::Device dev =
        (d.kind == CALIPER_DEV_CUDA && torch::cuda::is_available())
            ? torch::Device(torch::kCUDA)
        : (d.kind == CALIPER_DEV_METAL) ? torch::Device(torch::kMPS)
                                        : torch::Device(torch::kCPU);

    // Synthetic dataset: 256 points of y = sin(x) on [-pi, pi].
    auto X = torch::linspace(-M_PI, M_PI, 256,
                             torch::TensorOptions().device(dev))
                 .unsqueeze(1);
    auto Y = torch::sin(X);

    SineNet model;
    model->to(dev);
    torch::optim::Adam opt(model->parameters(),
                           torch::optim::AdamOptions(1e-2));

    for (int step = 0; step < 2000; step++) {
        if (ctl->cancelled(ctl)) return;             // <=100 ms contract
        opt.zero_grad();
        auto loss = torch::mse_loss(model->forward(X), Y);
        loss.backward();
        opt.step();
        publish(st, model, X, loss.item<float>());
        ctl->progress(ctl, step / 2000.f, "fitting sin(x)");
    }
    ctl->progress(ctl, 1.f, "done — the line is a sine");
}
// --8<-- [end:job]

} // namespace sinescope

namespace sinescope {

class SineScope final : public caliper::Applet {
public:
    SineScope() : s_(std::make_unique<SineState>()) {}

    // --8<-- [start:init]
    bool on_init(caliper::Host& host) override {
        auto* st = s_.get();
        st->host   = &host;
        st->jobs   = caliper::Jobs(host);     // required — manifest enforced
        st->bridge = caliper::Bridge(host);   // optional — may be falsy
        host.log_info("sine_scope: on_init");
        return true;
    }
    // --8<-- [end:init]

    void on_frame(const caliper::Frame&) override {
        auto* st = s_.get();

        // Copy the latest published state out under the mutex (gen-gated).
        std::vector<float> loss, pred;
        torch::Tensor disp_w; float w_max = 1e-6f;
        uint64_t gen;
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            gen  = st->gen;
            loss = st->loss_hist;
            pred = st->pred_ys;
            if (gen != 0 && gen != st->tex_gen) {
                disp_w = st->disp_w;          // co-owning handle copy
                w_max  = st->w_max;
            }
        }

        ImGui::Begin("SineScope");

        // Dev hook (not part of the tutorial): CALIPER_SINE_AUTOTRAIN=1
        // presses Train on the first frame, for headless verification.
        static bool autotrain_fired = false;
        const bool running = st->job_id && st->jobs.is_running(st->job_id);
        if (!autotrain_fired && !running &&
            std::getenv("CALIPER_SINE_AUTOTRAIN")) {
            autotrain_fired = true;
            st->job_id = st->jobs.submit("sine_scope: fit", &train_job, st);
        }

        // --8<-- [start:controls]
        if (!running) {
            if (ImGui::Button("Train"))
                st->job_id = st->jobs.submit("sine_scope: fit",
                                             &train_job, st);
        } else {
            if (ImGui::Button("Cancel")) st->jobs.request_cancel(st->job_id);
            ImGui::SameLine();
            ImGui::ProgressBar(st->jobs.progress_of(st->job_id), {-1, 0});
        }
        // --8<-- [end:controls]

        // --8<-- [start:plots]
        // These plots are for *viewing*, not editing: ImPlot is interactive by
        // default (drag-pan, scroll-zoom, box-select, right-click menu), so
        // lock every input off. Read-only plots always want these four flags.
        constexpr ImPlotFlags kLockedPlot =
            ImPlotFlags_NoInputs | ImPlotFlags_NoMenus |
            ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText;

        // The fixed grid + target are frame-side constants.
        static std::vector<float> xs, target;
        if (xs.empty()) {
            xs.resize(256); target.resize(256);
            for (int i = 0; i < 256; i++) {
                xs[i] = -3.14159265f + i / 255.0f * 6.2831853f;
                target[i] = std::sin(xs[i]);
            }
        }
        if (ImPlot::BeginPlot("fit", {-1, 240}, kLockedPlot)) {
            ImPlot::SetupAxes("x", "y", 0, 0);
            ImPlot::PlotLine("target sin(x)", xs.data(), target.data(), 256);
            if (!pred.empty())
                ImPlot::PlotLine("model(x)", xs.data(), pred.data(), 256);
            ImPlot::EndPlot();
        }
        ImGui::Checkbox("follow", &st->follow);   // viewport policy, §6
        const ImPlotAxisFlags f =
            st->follow ? ImPlotAxisFlags_AutoFit : 0;
        if (ImPlot::BeginPlot("loss", {-1, 160}, kLockedPlot)) {
            ImPlot::SetupAxes("step", "MSE", f, f);
            if (!loss.empty())
                ImPlot::PlotLine("mse", loss.data(), (int)loss.size());
            ImPlot::EndPlot();
        }
        // --8<-- [end:plots]

        // --8<-- [start:bridge]
        // The weight matrix as pixels — gen-gated rebuild, frame thread only.
        if (st->bridge && gen != 0 && gen != st->tex_gen &&
            disp_w.defined()) {
            if (st->w_tex) st->bridge.release_texture(st->w_tex);
            auto ct = caliper::adapters::to_tensor(disp_w);
            st->w_tex = ct ? st->bridge.texture_from_tensor_mapped(
                                 &*ct, CALIPER_CMAP_RDBU, -w_max, w_max)
                           : 0;
            st->tex_gen = gen;
        }
        if (st->w_tex) {
            ImGui::TextDisabled("fc1 weights (32x1 as 8x4 blocks, RdBu)");
            ImGui::Image(caliper::Bridge::imtex(st->w_tex), ImVec2(128, 256));
        } else {
            ImGui::TextDisabled(
                "tensor_bridge.v1 absent (ok) — no weight heatmap");
        }
        // --8<-- [end:bridge]

        ImGui::End();
    }

    // --8<-- [start:cleanup]
    void on_cleanup() override {
        auto* st = s_.get();
        if (st->job_id) {
            st->jobs.request_cancel(st->job_id);
            for (int i = 0; i < 1000 && st->jobs.is_running(st->job_id); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (st->w_tex) { st->bridge.release_texture(st->w_tex);
                         st->w_tex = 0; }
        if (st->host) st->host->log_info("sine_scope: on_cleanup");
    }
    // --8<-- [end:cleanup]

private:
    std::unique_ptr<SineState> s_;
};

} // namespace sinescope

CALIPER_APPLET(sinescope::SineScope,
    .id       = "dev.example.sine-scope",
    .version  = "0.1.0",
    .name     = "SineScope",
    .summary  = "Tutorial applet: a tiny MLP learns sin(x) live — loss curve, "
                "prediction vs target, first-layer weights as a heatmap.",
    .tag      = "Demo",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
