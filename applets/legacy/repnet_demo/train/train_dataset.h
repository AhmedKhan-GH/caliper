// Training Lab data loader: load raw ECG CSVs, preprocess with the C++ DSP
// unit, encode patient groups, and reproduce the Python train/val/test splits.
// Reproduces repnet reproduce.ipynb load_ecg_data + preprocess + downsample,
// bit-faithfully, so the full C++ pipeline matches the recorded Python result.
#pragma once

#include <torch/torch.h>

#include <functional>
#include <string>
#include <vector>

namespace tdata {

struct Dataset {
    torch::Tensor X;             // (N,12,2500) float32, PREPROCESSED (DSP applied)
    std::vector<int> y;          // N
    std::vector<int> groups_inv; // N, 0..n_groups-1 (numpy-unique-sorted patient ids)
    std::vector<long> ecg_ids;   // N
    int n_groups = 0;
};

// Loads + preprocesses all valid records from data_dir. progress optional.
Dataset load_and_preprocess(const std::string& data_dir,
                            std::function<void(int done, int total)> progress = {});

// scipy.signal.resample(x, 5000, axis=1) for x:(12,2500) -> (12,5000).
torch::Tensor resample_to_5000(const torch::Tensor& x2500);

struct Split {
    std::vector<int> train, val, test;
};

// Reproduce split_i: outer 5-fold seed=split_i*7+1000 -> fold0=test;
// inner 8-fold seed+1 on dev -> fold0=val.
Split make_split(const Dataset& d, int split_i);

}  // namespace tdata
