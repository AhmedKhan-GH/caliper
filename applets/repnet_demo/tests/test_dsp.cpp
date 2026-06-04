// Golden test for the Training Lab DSP preprocessing unit.
// For each ecg_id: load <id>_raw.bin (12x5000), run dsp::preprocess_5k,
// compare to <id>_pre.bin (12x2500). Assert max abs error < 1e-4.
//
// Achieved max abs error is printed below ("max abs err = ...").

#include "golden_io.h"
#include "train/dsp.h"

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// ecg_ids from dsp_cases/index.json (baked: do not parse json at runtime).
const std::vector<long> kEcgIds = {960040, 1611723, 2147188, 2274130};

}  // namespace

int main() {
    golden::Harness H;

    // --- sub-step sanity: sosfiltfilt against a small Python-generated fixture ---
    // HP SOS (same constants baked in dsp.cpp), 40-sample random input.
    {
        const std::vector<std::array<double, 6>> hp = {
            {0.9918242120005331, -1.9836484240010661, 0.9918242120005331, 1.0,
             -1.9884180173746582, 0.9884572678187328},
            {1.0, -2.0, 1.0, 1.0, -1.9951632412838627, 0.9952026248755107}};
        const std::vector<double> xin = {
            1.7640523460, 0.4001572084, 0.9787379841, 2.2408931992,  1.8675579901,
            -0.9772778799, 0.9500884175, -0.1513572083, -0.1032188518, 0.4105985019,
            0.1440435712, 1.4542735070, 0.7610377251, 0.1216750165, 0.4438632327,
            0.3336743274, 1.4940790732, -0.2051582638, 0.3130677017, -0.8540957393,
            -2.5529898158, 0.6536185954, 0.8644361989, -0.7421650204, 2.2697546240,
            -1.4543656746, 0.0457585173, -0.1871838500, 1.5327792144, 1.4693587699,
            0.1549474257, 0.3781625196, -0.8877857476, -1.9807964682, -0.3479121493,
            0.1563489691, 1.2302906807, 1.2023798488, -0.3873268174, -0.3023027506};
        const std::vector<double> yref = {
            1.1714566054, -0.1412874215, 0.4888960492, 1.8031078311, 1.4822853542,
            -1.3095793193, 0.6712189441, -0.3763316355, -0.2738328358, 0.2948126804,
            0.0835559598, 1.4495564873, 0.8125660188, 0.2299256907, 0.6093157061,
            0.5568103760, 1.7753828362, 0.1347997217, 0.7121687923, -0.3953602800,
            -2.0341263382, 1.2331061333, 1.5050462368, -0.0399316387, 3.0341146027,
            -0.6273734301, 0.9358911174, 0.7665996227, 2.5507265096, 2.5519852761,
            1.3027709758, 1.5917033969, 0.3919951963, -0.6342502565, 1.0659269992,
            1.6380111969, 2.7803086097, 2.8212885860, 1.3010103260, 1.4560028938};
        auto xt = torch::from_blob(const_cast<double*>(xin.data()),
                                   {static_cast<long>(xin.size())}, torch::kFloat64)
                      .clone();
        auto yt = dsp::sosfiltfilt(hp, xt);
        double mx = 0.0;
        for (size_t i = 0; i < yref.size(); ++i)
            mx = std::max(mx, std::abs(yt[static_cast<long>(i)].item<double>() - yref[i]));
        std::fprintf(stderr, "[test_dsp] sosfiltfilt fixture max abs err = %.3e\n", mx);
        H.check(mx < 1e-6, "sosfiltfilt matches scipy fixture (<1e-6)");
    }

    double global_max = 0.0;
    for (long id : kEcgIds) {
        const std::string base = "dsp_cases/" + std::to_string(id);
        torch::Tensor raw = golden::load_bin(base + "_raw.bin");   // (12,5000) f32
        torch::Tensor pre = golden::load_bin(base + "_pre.bin");   // (12,2500) f32

        H.check(raw.dim() == 2 && raw.size(0) == 12 && raw.size(1) == 5000,
                "raw shape (12,5000) for " + std::to_string(id));

        torch::Tensor out = dsp::preprocess_5k(raw);

        H.check(out.dim() == 2 && out.size(0) == 12 && out.size(1) == 2500,
                "output shape (12,2500) for " + std::to_string(id));
        H.check(out.dtype() == torch::kFloat32, "output dtype float32 for " + std::to_string(id));
        H.check(torch::isfinite(out).all().item<bool>(),
                "output finite for " + std::to_string(id));

        double mx =
            (out.to(torch::kFloat64) - pre.to(torch::kFloat64)).abs().max().item<double>();
        global_max = std::max(global_max, mx);
        std::fprintf(stderr, "[test_dsp] ecg %ld max abs err = %.3e\n", id, mx);
        H.check(mx < 1e-4, "preprocess_5k within 1e-4 for " + std::to_string(id));
    }
    std::fprintf(stderr, "[test_dsp] GLOBAL max abs err = %.3e\n", global_max);

    return H.report("test_dsp");
}
