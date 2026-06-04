// Golden-file test for the native PerLeadCNN libtorch module.
// Verifies param count, named param/buffer shapes, state_dict loading, and
// forward-pass numerical match against Python-exported goldens.
#include "perlead_cnn.h"
#include "golden_io.h"

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <map>
#include <string>
#include <vector>

using nlohmann::json;

int main() {
    golden::Harness H;

    // 1. Build model; count parameters; assert == 29490 (meta.json n_params).
    plcnn::PerLeadCNN model(12, {16, 32, 48}, {31, 21, 11}, 0.15, 2);
    int64_t n_params = 0;
    for (const auto& p : model->parameters()) n_params += p.numel();
    H.check(n_params == 29490,
            "param count == 29490 (got " + std::to_string(n_params) + ")");

    // 2. named_parameters + named_buffers must include every name in
    //    state_dict/index.json with matching shapes.
    std::map<std::string, std::vector<int64_t>> named;
    for (const auto& p : model->named_parameters()) {
        named[p.key()] = p.value().sizes().vec();
    }
    for (const auto& b : model->named_buffers()) {
        named[b.key()] = b.value().sizes().vec();
    }

    json index = json::parse(golden::load_text("state_dict/index.json"));
    for (const auto& e : index["entries"]) {
        std::string name = e["name"].get<std::string>();
        std::vector<int64_t> want = e["shape"].get<std::vector<int64_t>>();
        auto it = named.find(name);
        bool present = (it != named.end());
        H.check(present, "named tensor present: " + name);
        if (present) {
            H.check(it->second == want, "shape matches for " + name);
        }
    }

    // 3. Load state_dict; eval mode.
    int loaded = plcnn::load_state_dict_bins(model, golden::path("state_dict"));
    H.check(loaded == static_cast<int>(index["entries"].size()),
            "loaded all tensors (got " + std::to_string(loaded) + ")");
    model->eval();

    // 4. Forward perlead_in.bin (12,2500) -> (1,12,2500); compare to logits.
    torch::NoGradGuard ng;
    torch::Tensor in = golden::load_bin("perlead_in.bin");  // (12, 2500)
    H.check(in.sizes() == (std::vector<int64_t>{12, 2500}),
            "perlead_in is (12,2500)");
    torch::Tensor x = in.unsqueeze(0);  // (1,12,2500)
    torch::Tensor logits = model->forward(x);  // (1,2)
    torch::Tensor ref = golden::load_bin("perlead_logits.bin");  // (1,2)
    H.check(logits.sizes() == ref.sizes(), "logits shape matches ref (1,2)");
    double max_err = (logits - ref).abs().max().item<double>();
    std::fprintf(stderr, "  achieved max logit abs error: %.3e\n", max_err);
    H.check(max_err < 1e-4, "max logit abs error < 1e-4");

    // 5. Forward-shape check on random (4,12,2500) -> (4,2).
    torch::Tensor rnd = torch::randn({4, 12, 2500});
    torch::Tensor out = model->forward(rnd);
    H.check(out.sizes() == (std::vector<int64_t>{4, 2}),
            "random batch forward shape (4,2)");

    return H.report("test_perlead");
}
