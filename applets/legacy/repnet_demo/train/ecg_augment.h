// On-the-fly ECG augmentation for the Training Lab (native C++ libtorch).
// Semantic port of repnet train.py augment_ecg: 7 stochastic transforms applied
// in order to a single (12, T) float32 waveform. Randomness comes from a
// caller-supplied std::mt19937_64 so tests are deterministic. Live training is
// "close, not bit-identical" to Python, so numpy-RNG exactness is NOT required.
#pragma once

#include <torch/torch.h>

#include <random>

namespace augment {

// Augmentation config. Defaults mirror repnet AUG_CFG.
struct AugCfg {
    double p_noise = 0.5;
    double noise_sigma_lo = 0.01;
    double noise_sigma_hi = 0.05;

    double p_amp_scale = 0.5;
    double amp_scale_range = 0.15;  // scale = 1 + U(-r, r) per lead

    double p_time_shift = 0.3;
    int max_time_shift = 150;  // shift ~ randint(-max, max+1) inclusive

    double p_lead_drop = 0.10;
    double lead_drop_p = 0.12;  // keep lead where rand > lead_drop_p

    double p_cutout = 0.25;
    int cutout_len_lo = 50;   // length ~ randint(lo, hi)  -> [lo, hi)
    int cutout_len_hi = 200;

    double p_wander = 0.2;
    double wander_amp = 0.2;       // amp ~ U(0, wander_amp)
    double wander_freq_lo = 0.05;  // freq ~ U(lo, hi)
    double wander_freq_hi = 0.5;

    double p_resample = 0.10;
    double resample_rate = 0.05;  // rate = 1 + U(-r, r)

    double fs = 500.0;  // sampling rate used in the wander term
};

// Applies the 7 transforms in order (noise, amp_scale, time_shift, lead_drop,
// cutout, wander, resample) to a clone of x (shape (12, T)); returns (12, T).
// Does NOT modify the input tensor.
torch::Tensor augment_ecg(const torch::Tensor& x, const AugCfg& cfg,
                          std::mt19937_64& rng);

}  // namespace augment
