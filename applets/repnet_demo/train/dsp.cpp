// Training Lab DSP preprocessing (native C++ libtorch).
//
// Bit-faithful reproduction of repnet's Python ECG preprocessing, matching
// scipy.signal.sosfiltfilt exactly. All filtering is done in float64 (scipy
// uses float64); only the final result is cast to float32.
//
// Achieved accuracy vs the golden scipy output (see test_dsp.cpp output):
//   GLOBAL max abs err = 1.907e-06 across all golden ECG cases (well under the
//   1e-4 budget). The 1-D sosfiltfilt sub-step matches scipy to ~1e-10; the
//   ~1e-6 end-to-end residual is just float32 quantization of the golden .bin.
//
// SOS coefficients below are BAKED from tests/golden/sos_coeffs.json (HP =
// 4th-order Butterworth high-pass 0.5 Hz @ fs=500; notch = 60 Hz Q=30 @ fs=500
// via iirnotch->tf2sos). They are NOT parsed at runtime.

#include "train/dsp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dsp {
namespace {

// 4th-order Butterworth high-pass, 0.5 Hz, fs=500 (2 biquad sections).
const std::vector<std::array<double, 6>> kHpSos = {
    {0.9918242120005331, -1.9836484240010661, 0.9918242120005331, 1.0,
     -1.9884180173746582, 0.9884572678187328},
    {1.0, -2.0, 1.0, 1.0, -1.9951632412838627, 0.9952026248755107}};

// 60 Hz notch, Q=30, fs=500 (1 biquad section).
const std::vector<std::array<double, 6>> kNotchSos = {
    {0.9875889380903247, -1.4398427053125469, 0.9875889380903251, 1.0,
     -1.4398427053125467, 0.9751778761806491}};

// --- scipy.signal.lfilter_zi for a single normalized biquad (a0 == 1). ---
// For order-2 the state is length 2; solving zi = A*zi + B with
//   A = companion(a).T, B = b[1:] - a[1:]*b[0].
// Closed form (n=3 coefficients):
//   IminusA = [[1+a1, -1],[a2, 1]];  B = [b1 - a1*b0, b2 - a2*b0]
//   det = (1+a1) + a2
//   zi0 = (B0 + B1) / det
//   zi1 = a2*zi0 - B1     (from the explicit-formula path in scipy)
std::array<double, 2> lfilter_zi_biquad(const std::array<double, 6>& s) {
    const double b0 = s[0], b1 = s[1], b2 = s[2];
    const double a1 = s[4], a2 = s[5];  // a0 == 1
    const double B0 = b1 - a1 * b0;
    const double B1 = b2 - a2 * b0;
    const double det = (1.0 + a1) + a2;
    const double zi0 = (B0 + B1) / det;
    // scipy explicit path: zi[1] = asum*zi0 - csum, asum = 1 + a1, csum = B0.
    const double zi1 = (1.0 + a1) * zi0 - B0;
    return {zi0, zi1};
}

// scipy.signal.sosfilt_zi: per-section lfilter_zi scaled by cumulative DC gain.
std::vector<std::array<double, 2>> sosfilt_zi(
    const std::vector<std::array<double, 6>>& sos) {
    std::vector<std::array<double, 2>> zi(sos.size());
    double scale = 1.0;
    for (size_t k = 0; k < sos.size(); ++k) {
        const auto& s = sos[k];
        auto z = lfilter_zi_biquad(s);
        zi[k] = {scale * z[0], scale * z[1]};
        const double bsum = s[0] + s[1] + s[2];
        const double asum = s[3] + s[4] + s[5];
        scale *= bsum / asum;
    }
    return zi;
}

// scipy _sosfilt: Direct-Form-II-transposed cascade, in place on `x`.
// `zi` (state) is per-section, length 2, and is updated in place.
// a0 is assumed 1 (true for all our baked SOS).
void sosfilt(const std::vector<std::array<double, 6>>& sos, std::vector<double>& x,
             std::vector<std::array<double, 2>>& zi) {
    const int64_t n = static_cast<int64_t>(x.size());
    for (size_t s = 0; s < sos.size(); ++s) {
        const double b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2];
        const double a1 = sos[s][4], a2 = sos[s][5];
        double z0 = zi[s][0], z1 = zi[s][1];
        for (int64_t i = 0; i < n; ++i) {
            const double xc = x[i];
            const double xn = b0 * xc + z0;
            z0 = b1 * xc - a1 * xn + z1;
            z1 = b2 * xc - a2 * xn;
            x[i] = xn;
        }
        zi[s][0] = z0;
        zi[s][1] = z1;
    }
}

// scipy odd_ext: reflect-and-negate-about-endpoint extension of length `edge`
// at each end. Front: 2*x[0] - x[edge..1]; back: 2*x[-1] - x[-2..-edge-1].
std::vector<double> odd_ext(const std::vector<double>& x, int edge) {
    const int64_t n = static_cast<int64_t>(x.size());
    std::vector<double> ext;
    ext.reserve(static_cast<size_t>(n) + 2 * edge);
    const double x0 = x[0];
    for (int j = edge; j >= 1; --j) ext.push_back(2.0 * x0 - x[j]);
    for (int64_t i = 0; i < n; ++i) ext.push_back(x[i]);
    const double xl = x[n - 1];
    for (int j = 2; j <= edge + 1; ++j) ext.push_back(2.0 * xl - x[n - 1 - j + 1]);
    return ext;
}

// Run sosfiltfilt on a single 1-D double signal.
std::vector<double> sosfiltfilt_1d(const std::vector<std::array<double, 6>>& sos,
                                   const std::vector<double>& x) {
    const int64_t n = static_cast<int64_t>(x.size());
    const int ntaps = 2 * static_cast<int>(sos.size()) + 1;  // no a/b zeros at origin here
    const int edge = ntaps * 3;

    // odd padding
    std::vector<double> ext = odd_ext(x, edge);
    const int64_t m = static_cast<int64_t>(ext.size());

    const auto zi_base = sosfilt_zi(sos);

    // forward pass, zi scaled by ext[0]
    std::vector<std::array<double, 2>> zi = zi_base;
    for (auto& z : zi) {
        z[0] *= ext[0];
        z[1] *= ext[0];
    }
    sosfilt(sos, ext, zi);

    // reverse
    std::reverse(ext.begin(), ext.end());

    // backward pass, zi scaled by (new) ext[0] == previous last sample
    zi = zi_base;
    for (auto& z : zi) {
        z[0] *= ext[0];
        z[1] *= ext[0];
    }
    sosfilt(sos, ext, zi);

    // reverse back
    std::reverse(ext.begin(), ext.end());

    // remove padding
    std::vector<double> y(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) y[static_cast<size_t>(i)] = ext[edge + i];
    (void)m;
    return y;
}

}  // namespace

torch::Tensor sosfiltfilt(const std::vector<std::array<double, 6>>& sos,
                          const torch::Tensor& x) {
    auto xd = x.to(torch::kFloat64).contiguous();
    const int64_t T = xd.size(-1);
    const int64_t rows = xd.numel() / T;
    auto x2 = xd.reshape({rows, T});
    auto out = torch::empty_like(x2);
    for (int64_t r = 0; r < rows; ++r) {
        const double* src = x2[r].data_ptr<double>();
        std::vector<double> row(src, src + T);
        std::vector<double> filt = sosfiltfilt_1d(sos, row);
        std::copy(filt.begin(), filt.end(), out[r].data_ptr<double>());
    }
    return out.reshape(xd.sizes());
}

torch::Tensor preprocess_5k(const torch::Tensor& raw) {
    // 1. high-pass, 2. notch (both zero-phase, float64)
    torch::Tensor x = sosfiltfilt(kHpSos, raw);
    x = sosfiltfilt(kNotchSos, x);

    // 3. per-lead z-norm over time, population std (ddof=0).
    auto mean = x.mean(/*dim=*/-1, /*keepdim=*/true);
    auto std = x.std(/*dim=*/-1, /*unbiased=*/false, /*keepdim=*/true);
    x = (x - mean) / (std + 1e-8);

    // 4. downsample [:, ::2]
    x = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)});

    return x.to(torch::kFloat32).contiguous();
}

}  // namespace dsp
