#pragma once
// EmbedScope model + applet facade (Phase 2F′, Task F5).
//
// The net has a LEARNED 3-D bottleneck: the activations of Linear(64->3) ARE the
// coordinates the ImPlot3D scatter draws — no post-hoc projection (PCA/t-SNE).
// You watch a single gray blob split into ten colored lobes as training runs.
//
//   conv(1->8,3x3) -> ReLU -> maxpool2
//   conv(8->16,3x3) -> ReLU -> maxpool2
//   flatten (16*5*5 = 400)
//   Linear(400->64) -> ReLU
//   Linear(64->3)                <- the 3-D embedding (no ReLU: signed coords)
//   Linear(3->10)                <- classifier head over the embedding
//
// Facade shape mirrors gpt_scope: the heavy state (mutex, model, snapshot
// vectors, curl) is a forward-declared struct defined in embed_scope.cpp, so
// plugin.cpp includes only this header.
#include <torch/torch.h>

#include <memory>

namespace caliper { class Host; }   // fwd only — see the facade below

namespace embedscope {

// 28x28 -> conv1(26) -> pool(13) -> conv2(11) -> pool(5): 16*5*5 = 400.
struct EmbedNetImpl : torch::nn::Module {
    torch::nn::Conv2d  conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear  fc1{nullptr}, fc_embed{nullptr}, fc_out{nullptr};

    EmbedNetImpl() {
        conv1    = register_module("conv1", torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(1, 8, 3)));
        conv2    = register_module("conv2", torch::nn::Conv2d(
                       torch::nn::Conv2dOptions(8, 16, 3)));
        fc1      = register_module("fc1",      torch::nn::Linear(400, 64));
        fc_embed = register_module("fc_embed", torch::nn::Linear(64, 3));
        fc_out   = register_module("fc_out",   torch::nn::Linear(3, 10));
    }

    // (N,1,28,28) -> (N,3): the learned embedding. No ReLU — signed coordinates.
    torch::Tensor embed(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = x.flatten(1);
        x = torch::relu(fc1->forward(x));
        return fc_embed->forward(x);
    }

    // (N,1,28,28) -> (N,10) logits (classifier head over the 3-D embedding).
    torch::Tensor forward(torch::Tensor x) {
        return fc_out->forward(embed(x));
    }
};
TORCH_MODULE(EmbedNet);

struct EmbedScopeState;   // defined in embed_scope.cpp (mutex, model, curl…)

// Applet facade (epoch-2). Logic lives in embed_scope.cpp; plugin.cpp is the
// thin ABI bridge. id/version there are byte-identical to the manifest.
class EmbedScopeApplet {
public:
    EmbedScopeApplet();
    ~EmbedScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();

private:
    std::unique_ptr<EmbedScopeState> s_;
};

} // namespace embedscope
