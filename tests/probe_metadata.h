#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace caliper_test {

// Descriptors only. These cases need no decoder weights or forward pass.
inline nlohmann::json probe_descriptors() {
    using nlohmann::json;
    json rows = json::array({
        {{"point", "head"}, {"kind", "module"}, {"module", "head"},
         {"head", nullptr}, {"type", "Linear"}, {"shape", nullptr}},
        {{"point", "layers.0.self_attn#1"}, {"kind", "head"}, {"module", "layers.0.self_attn"},
         {"head", 1}, {"type", "Attention"}, {"shape", {1, 2, 16}}}});
    for (const char* probe : {"resid_pre", "mlp_in", "mlp_out", "attn_pattern"}) {
        const bool attention = std::string(probe) == "attn_pattern";
        rows.push_back({{"point", std::string("layers.0.") + probe}, {"kind", "module"},
                        {"module", attention ? "layers.0.self_attn" : "layers.0.mlp"},
                        {"head", nullptr}, {"type", attention ? "Attention" : "TranscoderMLP"}, {"shape", nullptr},
                        {"probe", probe}, {"layer", 0}, {"pattern", attention ? json("softmax") : json()}});
    }
    rows[2]["module"] = "layers.0";
    rows[2]["type"] = "Block";
    for (const bool full : {false, true}) {
        const std::string block = full ? "model.layers.3" : "model.layers.0";
        rows.push_back({{"point", block + ".attn_pattern"}, {"kind", "module"},
                        {"module", block + (full ? ".self_attn" : ".linear_attn")}, {"head", nullptr},
                        {"type", full ? "Qwen3_5Attention" : "Qwen3_5GatedDeltaNet"}, {"shape", nullptr},
                        {"probe", "attn_pattern"}, {"layer", full ? 3 : 0},
                        {"pattern", full ? "softmax" : "signed_value_mixing"}});
    }
    return rows;
}

}  // namespace caliper_test
