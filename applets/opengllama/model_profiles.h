#pragma once

#include <string>
#include <cstdint>

struct llama_model;

struct ModelProfile {
    std::string id;
    std::string display_name;

    int context_size        = 4096;
    int n_gpu_layers        = 99;

    float temperature       = 0.8f;
    int top_k               = 40;
    float top_p             = 0.95f;
    float min_p             = 0.05f;
    float repeat_penalty    = 1.1f;
    int repeat_last_n       = 64;

    bool supports_thinking  = false;
    std::string system_prompt = "You are a helpful assistant.";
};

ModelProfile detect_model_profile(const llama_model* model);
