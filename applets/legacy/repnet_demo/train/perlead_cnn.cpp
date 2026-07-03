#include "perlead_cnn.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace plcnn {

PerLeadCNNImpl::PerLeadCNNImpl(int n_leads, std::vector<int> filters,
                               std::vector<int> kernels, double dropout,
                               int n_classes)
    : n_leads_(n_leads) {
    if (filters.size() != kernels.size()) {
        throw std::runtime_error("PerLeadCNN: filters/kernels size mismatch");
    }

    // backbone = Sequential of, per (f,k): Conv1d(in,f,k,stride=2,
    // padding=k/2,bias=false), BatchNorm1d(f), Mish(). in_ch starts at 1.
    // Child indices 0..8 must match golden state_dict keys (backbone.0.*, ...).
    torch::nn::Sequential seq;
    int in_ch = 1;
    for (size_t i = 0; i < filters.size(); ++i) {
        int f = filters[i];
        int k = kernels[i];
        auto conv_opts = torch::nn::Conv1dOptions(in_ch, f, k)
                             .stride(2)
                             .padding(k / 2)
                             .bias(false);
        seq->push_back(torch::nn::Conv1d(conv_opts));
        seq->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(f)));
        seq->push_back(torch::nn::Mish());
        in_ch = f;
    }
    backbone = register_module("backbone", seq);

    pool = register_module("pool", torch::nn::AdaptiveAvgPool1d(
                                       torch::nn::AdaptiveAvgPool1dOptions(1)));
    head_drop = register_module(
        "head_drop", torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));

    int last_f = filters.back();
    fc = register_module(
        "fc", torch::nn::Linear(
                  torch::nn::LinearOptions(last_f * n_leads, n_classes)));
}

torch::Tensor PerLeadCNNImpl::forward(torch::Tensor x) {
    // x: (B, n_leads, T)
    const int64_t B = x.size(0);
    const int64_t L = x.size(1);
    const int64_t T = x.size(2);

    // (B, L, T) -> (B*L, 1, T)
    x = x.reshape({B * L, 1, T});
    x = backbone->forward(x);          // (B*L, C, T')
    x = pool->forward(x);              // (B*L, C, 1)
    x = x.squeeze(-1);                 // (B*L, C)
    x = x.reshape({B, L * x.size(1)}); // (B, L*C) == (B, 576)
    x = head_drop->forward(x);
    x = fc->forward(x);               // (B, n_classes)
    return x;
}

int load_state_dict_bins(PerLeadCNN& model, const std::string& state_dict_dir) {
    using nlohmann::json;

    std::ifstream idx_f(state_dict_dir + "/index.json");
    if (!idx_f) {
        throw std::runtime_error("cannot open state_dict index.json in " +
                                 state_dict_dir);
    }
    json index = json::parse(idx_f);

    // Build name -> tensor map covering params and buffers.
    std::unordered_map<std::string, torch::Tensor> tensors;
    for (const auto& p : model->named_parameters()) tensors[p.key()] = p.value();
    for (const auto& b : model->named_buffers()) tensors[b.key()] = b.value();

    torch::NoGradGuard ng;
    int count = 0;
    for (const auto& e : index["entries"]) {
        const std::string name = e["name"].get<std::string>();
        const std::string file = e["file"].get<std::string>();

        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("state_dict tensor not found in model: " +
                                     name);
        }

        // Read .bin (same format as golden::load_bin):
        // int32 ndim, int32 dims[ndim], float32 data[prod].
        std::ifstream f(state_dict_dir + "/" + file, std::ios::binary);
        if (!f) {
            throw std::runtime_error("cannot open state_dict bin: " + file);
        }
        int32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        std::vector<int64_t> shape(ndim);
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) {
            int32_t d = 0;
            f.read(reinterpret_cast<char*>(&d), 4);
            shape[i] = d;
            n *= d;
        }
        std::vector<float> data(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
        if (!f) throw std::runtime_error("short read on state_dict bin: " + file);

        torch::Tensor src =
            torch::from_blob(data.data(), shape, torch::kFloat32).clone();
        it->second.copy_(src.to(it->second.dtype()));
        ++count;
    }
    return count;
}

}  // namespace plcnn
