// Training Lab DSP preprocessing (native C++ libtorch).
// Bit-faithful reproduction of the repnet Python ECG preprocessing:
//   1. 4th-order Butterworth high-pass 0.5 Hz @ fs=500, sosfiltfilt (zero-phase)
//   2. 60 Hz notch (Q=30) @ fs=500, sosfiltfilt (zero-phase)
//   3. per-lead z-norm over time: (x-mean)/(std+1e-8), population std (ddof=0)
//   4. downsample x[:, ::2] -> 250 Hz
#pragma once

#include <torch/torch.h>

#include <array>
#include <vector>

namespace dsp {

// raw: (12, 5000) float32 @500Hz. returns (12, 2500) float32 @250Hz.
torch::Tensor preprocess_5k(const torch::Tensor& raw);

// Zero-phase forward-backward filter, replicating scipy.signal.sosfiltfilt
// (padtype='odd', default padlen). Operates along the last dim of x.
// Each entry of `sos` is one biquad section [b0,b1,b2,a0,a1,a2].
torch::Tensor sosfiltfilt(const std::vector<std::array<double, 6>>& sos,
                          const torch::Tensor& x);

}  // namespace dsp
