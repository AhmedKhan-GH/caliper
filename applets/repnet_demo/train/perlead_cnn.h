// PerLeadCNN: native libtorch (torch::nn) port of the Python eager module used
// in the caliper Training Lab. Module/parameter names match the Python module so
// the exported state_dict (best_model.pt) loads directly.
#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

namespace plcnn {

struct PerLeadCNNImpl : torch::nn::Module {
    PerLeadCNNImpl(int n_leads = 12, std::vector<int> filters = {16, 32, 48},
                   std::vector<int> kernels = {31, 21, 11}, double dropout = 0.15,
                   int n_classes = 2);
    torch::Tensor forward(torch::Tensor x);

    int n_leads_ = 12;
    torch::nn::Sequential backbone{nullptr};
    torch::nn::AdaptiveAvgPool1d pool{nullptr};
    torch::nn::Dropout head_drop{nullptr};
    torch::nn::Linear fc{nullptr};
};

// Hand-rolled module holder (instead of TORCH_MODULE) so that braced-init-list
// constructor arguments (e.g. {16,32,48}) bind to the explicit signature rather
// than the variadic ModuleHolder forwarding constructor (which can't deduce
// std::initializer_list from a brace-enclosed list).
class PerLeadCNN : public torch::nn::ModuleHolder<PerLeadCNNImpl> {
   public:
    using torch::nn::ModuleHolder<PerLeadCNNImpl>::ModuleHolder;
    PerLeadCNN(int n_leads = 12, std::vector<int> filters = {16, 32, 48},
               std::vector<int> kernels = {31, 21, 11}, double dropout = 0.15,
               int n_classes = 2)
        : torch::nn::ModuleHolder<PerLeadCNNImpl>(std::make_shared<PerLeadCNNImpl>(
              n_leads, std::move(filters), std::move(kernels), dropout,
              n_classes)) {}
};

// Loads named .bin tensors from `state_dict_dir` into model (eval-ready).
// Returns the number of tensors loaded.
int load_state_dict_bins(PerLeadCNN& model, const std::string& state_dict_dir);

}  // namespace plcnn
