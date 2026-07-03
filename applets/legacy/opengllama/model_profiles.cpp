#include "model_profiles.h"

#include <llama.h>

#include <algorithm>
#include <cstring>

namespace {

std::string meta_str(const llama_model* model, const char* key) {
    char buf[256] = {};
    int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n < 0) return {};
    return std::string(buf, std::min((size_t)n, sizeof(buf) - 1));
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

ModelProfile make_qwen3_coder_30b() {
    ModelProfile p;
    p.id             = "qwen3-coder-30b";
    p.display_name   = "Qwen3-Coder 30B";
    p.context_size   = 32768;
    p.n_gpu_layers   = 99;
    p.temperature    = 0.7f;
    p.top_k          = 20;
    p.top_p          = 0.8f;
    p.min_p          = 0.0f;
    p.repeat_penalty = 1.0f;
    p.repeat_last_n  = 0;
    p.supports_thinking = true;
    p.system_prompt  = "You are Qwen, a helpful coding assistant created by Alibaba Cloud.";
    return p;
}

ModelProfile make_qwen3_generic() {
    ModelProfile p;
    p.id             = "qwen3";
    p.display_name   = "Qwen3";
    p.context_size   = 32768;
    p.n_gpu_layers   = 99;
    p.temperature    = 0.7f;
    p.top_k          = 20;
    p.top_p          = 0.8f;
    p.min_p          = 0.0f;
    p.repeat_penalty = 1.0f;
    p.repeat_last_n  = 0;
    p.supports_thinking = true;
    p.system_prompt  = "You are Qwen, a helpful assistant created by Alibaba Cloud.";
    return p;
}

ModelProfile make_default() {
    return ModelProfile{};
}

} // anonymous namespace

ModelProfile detect_model_profile(const llama_model* model) {
    if (!model) return make_default();

    std::string arch = to_lower(meta_str(model, "general.architecture"));
    std::string name = to_lower(meta_str(model, "general.name"));

    char desc_buf[256] = {};
    llama_model_desc(model, desc_buf, sizeof(desc_buf));
    std::string desc = to_lower(desc_buf);

    bool is_qwen3 = (arch.find("qwen3") != std::string::npos) ||
                    (desc.find("qwen3") != std::string::npos);

    if (is_qwen3) {
        bool is_coder = (name.find("coder") != std::string::npos) ||
                        (desc.find("coder") != std::string::npos);

        if (is_coder) return make_qwen3_coder_30b();
        return make_qwen3_generic();
    }

    ModelProfile fallback = make_default();

    int n_ctx_train = llama_model_n_ctx_train(model);
    fallback.context_size = std::min(n_ctx_train, 4096);

    return fallback;
}
