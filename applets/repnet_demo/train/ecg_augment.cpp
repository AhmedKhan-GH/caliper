// On-the-fly ECG augmentation — semantic port of repnet train.py augment_ecg.
// Operates on a clone of a (12, T) float32 waveform; all randomness from the
// caller-supplied std::mt19937_64. Transform order matches Python exactly:
// noise, amp_scale, time_shift, lead_drop, cutout, wander, resample.
#include "train/ecg_augment.h"

#include <algorithm>
#include <cmath>

namespace augment {
namespace {

// U(a, b) inclusive of a, semantically like np.random.uniform(a, b).
double uniform(std::mt19937_64& rng, double a, double b) {
    std::uniform_real_distribution<double> d(a, b);
    return d(rng);
}
// Bernoulli draw in [0,1), like np.random.rand().
double rand01(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(rng);
}
// np.random.randint(lo, hi) -> integer in [lo, hi).
long randint(std::mt19937_64& rng, long lo, long hi) {
    if (hi <= lo) return lo;  // guard; matches max(...,1) usage upstream
    std::uniform_int_distribution<long> d(lo, hi - 1);
    return d(rng);
}

}  // namespace

torch::Tensor augment_ecg(const torch::Tensor& x_in, const AugCfg& cfg,
                          std::mt19937_64& rng) {
    torch::Tensor x = x_in.clone();
    const long C = x.size(0);
    const long T = x.size(1);

    // 1. Additive Gaussian noise: sigma ~ U(lo,hi); x += N(0,sigma).
    if (rand01(rng) < cfg.p_noise) {
        double sigma = uniform(rng, cfg.noise_sigma_lo, cfg.noise_sigma_hi);
        std::normal_distribution<double> nd(0.0, sigma);
        torch::Tensor noise = torch::empty_like(x);
        auto acc = noise.accessor<float, 2>();
        for (long i = 0; i < C; ++i)
            for (long j = 0; j < T; ++j) acc[i][j] = static_cast<float>(nd(rng));
        x = x + noise;
    }

    // 2. Per-lead amplitude scale: scale = 1 + U(-r, r), shape (C,1).
    if (rand01(rng) < cfg.p_amp_scale) {
        torch::Tensor scale = torch::empty({C, 1}, torch::kFloat32);
        auto acc = scale.accessor<float, 2>();
        for (long i = 0; i < C; ++i)
            acc[i][0] = static_cast<float>(
                1.0 + uniform(rng, -cfg.amp_scale_range, cfg.amp_scale_range));
        x = x * scale;
    }

    // 3. Time shift: shift ~ randint(-max, max+1) inclusive; roll along time.
    if (rand01(rng) < cfg.p_time_shift) {
        long shift = randint(rng, -cfg.max_time_shift, cfg.max_time_shift + 1);
        x = torch::roll(x, /*shifts=*/{shift}, /*dims=*/{1});
    }

    // 4. Lead drop: keep lead where rand > lead_drop_p; zero the rest.
    if (rand01(rng) < cfg.p_lead_drop) {
        torch::Tensor mask = torch::empty({C, 1}, torch::kFloat32);
        auto acc = mask.accessor<float, 2>();
        for (long i = 0; i < C; ++i)
            acc[i][0] = (rand01(rng) > cfg.lead_drop_p) ? 1.0f : 0.0f;
        x = x * mask;
    }

    // 5. Cutout: zero a contiguous time window across all leads.
    if (rand01(rng) < cfg.p_cutout) {
        long length = randint(rng, cfg.cutout_len_lo, cfg.cutout_len_hi);  // [lo,hi)
        long hi = std::max<long>(T - length, 1);
        long start = randint(rng, 0, hi);
        long end = std::min<long>(start + length, T);
        if (end > start)
            x.index_put_({torch::indexing::Slice(),
                          torch::indexing::Slice(start, end)},
                         0);
    }

    // 6. Baseline wander: amp*sin(2*pi*freq*t/fs) added to all leads.
    if (rand01(rng) < cfg.p_wander) {
        double freq = uniform(rng, cfg.wander_freq_lo, cfg.wander_freq_hi);
        double amp = uniform(rng, 0.0, cfg.wander_amp);
        torch::Tensor t = torch::arange(T, torch::kFloat32);
        torch::Tensor wander =
            (t * (2.0 * M_PI * freq / cfg.fs)).sin() * static_cast<float>(amp);
        x = x + wander.unsqueeze(0);  // broadcast (1,T) over leads
    }

    // 7. Resample: rate = 1 + U(-r, r); linear resample then crop/zero-pad to T.
    if (rand01(rng) < cfg.p_resample) {
        double rate = 1.0 + uniform(rng, -cfg.resample_rate, cfg.resample_rate);
        long new_len = static_cast<long>(static_cast<double>(T) * rate);  // int() truncates toward 0
        new_len = std::max<long>(new_len, 1);
        // Linear interpolation along time (semantic match; scipy uses Fourier).
        torch::Tensor resampled = torch::nn::functional::interpolate(
            x.unsqueeze(0),  // (1,C,T)
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{new_len})
                .mode(torch::kLinear)
                .align_corners(false)).squeeze(0);  // (C,new_len)

        if (new_len >= T) {
            x = resampled.index({torch::indexing::Slice(),
                                 torch::indexing::Slice(0, T)})
                    .contiguous();
        } else {
            torch::Tensor pad = torch::zeros({C, T - new_len}, torch::kFloat32);
            x = torch::cat({resampled, pad}, /*dim=*/1);
        }
    }

    return x.to(torch::kFloat32).contiguous();
}

}  // namespace augment
