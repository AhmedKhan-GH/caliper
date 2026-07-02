#pragma once
// gpt_model.h — a nanoGPT-style char-level transformer (Task E1, Phase 2E′).
//
// The attention is MANUAL on purpose: att = softmax(mask(QKᵀ/√d)) is built as an
// explicit sequence of tensors, never torch's fused sdpa. That keeps the (n_head,
// T,T) weight matrix reachable, so a PROBE forward can hand it back per layer for
// E2's attention panel. The training path passes need_weights=false and retains
// nothing (no clone, no graph on the weights) — the visualization tax is paid
// only when someone is looking.
//
// Config is fixed by the brief: 4 layers / 4 heads / 128 embd, block 128, dropout
// 0.1. Only vocab_size is data-driven (built from the corpus at train time).
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace caliper { class Host; }   // fwd only — see the applet facade at the end

namespace gptscope {

struct GPTScopeState;   // defined in gpt_scope.cpp (heavy: mutex, model, curl)

struct GPTConfig {
    int64_t vocab_size = 0;    // set from the corpus (char-level)
    int64_t n_layer    = 4;
    int64_t n_head     = 4;
    int64_t n_embd     = 128;
    int64_t block_size = 128;
    double  dropout    = 0.1;
};

// ---------------------------------------------------------------------------
// Manual multi-head causal self-attention.
// ---------------------------------------------------------------------------
class CausalSelfAttentionImpl : public torch::nn::Module {
public:
    explicit CausalSelfAttentionImpl(const GPTConfig& cfg)
        : n_head_(cfg.n_head), n_embd_(cfg.n_embd),
          c_attn_(torch::nn::LinearOptions(cfg.n_embd, 3 * cfg.n_embd)),
          c_proj_(torch::nn::LinearOptions(cfg.n_embd, cfg.n_embd)),
          attn_drop_(cfg.dropout), resid_drop_(cfg.dropout) {
        register_module("c_attn", c_attn_);
        register_module("c_proj", c_proj_);
        register_module("attn_drop", attn_drop_);
        register_module("resid_drop", resid_drop_);
        // Lower-triangular causal mask (block,block) as a BUFFER so it rides
        // model->to(device) alongside the parameters (never a stray CPU tensor
        // in the middle of a device matmul).
        mask_ = register_buffer(
            "mask", torch::tril(torch::ones({cfg.block_size, cfg.block_size})));
    }

    // x: (B,T,C). need_weights=false on the training path (retains nothing).
    // When true and out_att != nullptr, appends this layer's (n_head,T,T)
    // attention for the FIRST batch element (the probe) — E2's panel.
    torch::Tensor forward(const torch::Tensor& x, bool need_weights,
                          std::vector<torch::Tensor>* out_att) {
        const auto B = x.size(0), T = x.size(1), C = x.size(2);
        const auto hs = C / n_head_;
        auto qkv = c_attn_->forward(x).split(n_embd_, 2);       // 3 x (B,T,C)
        auto heads = [&](const torch::Tensor& t) {
            return t.view({B, T, n_head_, hs}).transpose(1, 2); // (B,nh,T,hs)
        };
        auto q = heads(qkv[0]), k = heads(qkv[1]), v = heads(qkv[2]);

        // QKᵀ/√d + causal mask + softmax, kept as explicit tensors.
        auto att = torch::matmul(q, k.transpose(-2, -1))
                   * (1.0 / std::sqrt(static_cast<double>(hs)));   // (B,nh,T,T)
        att = att.masked_fill(mask_.slice(0, 0, T).slice(1, 0, T) == 0,
                              -std::numeric_limits<float>::infinity());
        att = torch::softmax(att, -1);
        if (need_weights && out_att)
            out_att->push_back(att.select(0, 0).detach().clone());  // (nh,T,T)
        att = attn_drop_->forward(att);

        auto y = torch::matmul(att, v);                        // (B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view({B, T, C});
        return resid_drop_->forward(c_proj_->forward(y));
    }

private:
    int64_t n_head_, n_embd_;
    torch::nn::Linear  c_attn_, c_proj_;
    torch::nn::Dropout attn_drop_, resid_drop_;
    torch::Tensor      mask_;
};
TORCH_MODULE(CausalSelfAttention);

// ---------------------------------------------------------------------------
// Transformer block: pre-LN attention + pre-LN MLP (4×), both residual.
// ---------------------------------------------------------------------------
class BlockImpl : public torch::nn::Module {
public:
    explicit BlockImpl(const GPTConfig& cfg)
        : ln1_(torch::nn::LayerNormOptions({cfg.n_embd})),
          attn_(cfg),
          ln2_(torch::nn::LayerNormOptions({cfg.n_embd})),
          fc_(torch::nn::LinearOptions(cfg.n_embd, 4 * cfg.n_embd)),
          proj_(torch::nn::LinearOptions(4 * cfg.n_embd, cfg.n_embd)),
          drop_(cfg.dropout) {
        register_module("ln1", ln1_);   register_module("attn", attn_);
        register_module("ln2", ln2_);   register_module("fc", fc_);
        register_module("proj", proj_); register_module("drop", drop_);
    }

    torch::Tensor forward(torch::Tensor x, bool need_weights,
                          std::vector<torch::Tensor>* out_att) {
        x = x + attn_->forward(ln1_->forward(x), need_weights, out_att);
        auto h = proj_->forward(torch::gelu(fc_->forward(ln2_->forward(x))));
        return x + drop_->forward(h);
    }

private:
    torch::nn::LayerNorm ln1_;
    CausalSelfAttention  attn_;
    torch::nn::LayerNorm ln2_;
    torch::nn::Linear    fc_, proj_;
    torch::nn::Dropout   drop_;
};
TORCH_MODULE(Block);

// ---------------------------------------------------------------------------
// GPT: token + positional embedding, N blocks, final LayerNorm, LM head.
// ---------------------------------------------------------------------------
class GPTImpl : public torch::nn::Module {
public:
    explicit GPTImpl(const GPTConfig& cfg)
        : cfg_(cfg),
          wte_(torch::nn::EmbeddingOptions(cfg.vocab_size, cfg.n_embd)),
          wpe_(torch::nn::EmbeddingOptions(cfg.block_size, cfg.n_embd)),
          drop_(cfg.dropout),
          ln_f_(torch::nn::LayerNormOptions({cfg.n_embd})),
          lm_head_(torch::nn::LinearOptions(cfg.n_embd, cfg.vocab_size)
                       .bias(false)) {
        register_module("wte", wte_);
        register_module("wpe", wpe_);
        register_module("drop", drop_);
        blocks_ = register_module("blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < cfg.n_layer; ++i) {
            Block b(cfg);
            blocks_->push_back(b);      // registers for state / to(device)
            block_vec_.push_back(b);    // typed handle for forward()
        }
        register_module("ln_f", ln_f_);
        register_module("lm_head", lm_head_);
    }

    // idx: (B,T) int64 on the model's device. Returns logits (B,T,vocab).
    // need_weights collects per-layer (n_head,T,T) attention (the probe forward
    // E2 consumes); false on every training/generation forward.
    torch::Tensor forward(const torch::Tensor& idx, bool need_weights = false,
                          std::vector<torch::Tensor>* out_att = nullptr) {
        const auto T = idx.size(1);
        auto pos = torch::arange(
            T, torch::TensorOptions(idx.device()).dtype(torch::kLong));
        auto x = drop_->forward(wte_->forward(idx) + wpe_->forward(pos));
        for (auto& b : block_vec_)
            x = b->forward(x, need_weights, out_att);
        x = ln_f_->forward(x);
        return lm_head_->forward(x);                            // (B,T,vocab)
    }

    // Autoregressive sampling: seeded by `idx` (1,t0), grows by max_new tokens
    // at `temperature`. no_grad + eval; crops the context to block_size. Restores
    // the prior train/eval mode so callers can resume stepping.
    torch::Tensor generate(torch::Tensor idx, int64_t max_new,
                           double temperature) {
        torch::NoGradGuard ng;
        const bool was_training = is_training();
        eval();
        for (int64_t i = 0; i < max_new; ++i) {
            auto cond = idx.size(1) <= cfg_.block_size
                            ? idx
                            : idx.slice(1, idx.size(1) - cfg_.block_size);
            auto logits = forward(cond);                         // (1,T,V)
            auto last = logits.select(1, logits.size(1) - 1)     // (1,V)
                        / std::max(temperature, 1e-6);
            auto probs = torch::softmax(last, -1);
            auto next = torch::multinomial(probs, 1);            // (1,1)
            idx = torch::cat({idx, next}, 1);
        }
        if (was_training) train();
        return idx;
    }

    const GPTConfig& config() const { return cfg_; }

private:
    GPTConfig             cfg_;
    torch::nn::Embedding  wte_, wpe_;
    torch::nn::Dropout    drop_;
    torch::nn::ModuleList blocks_{nullptr};
    std::vector<Block>    block_vec_;   // co-owns the ModuleList's blocks
    torch::nn::LayerNorm  ln_f_;
    torch::nn::Linear     lm_head_;
};
TORCH_MODULE(GPT);

// ---------------------------------------------------------------------------
// The applet facade (pImpl). Declared in this header because the fixed 5-file
// layout gives the applet a single header: the heavy State (mutex, the live
// loss/sample snapshot, curl, the model itself) lives entirely in gpt_scope.cpp.
// plugin.cpp wraps this in a caliper::Applet — the established adapter shape.
// ---------------------------------------------------------------------------
class GPTScopeApplet {
public:
    GPTScopeApplet();
    ~GPTScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();

private:
    std::unique_ptr<GPTScopeState> s_;
};

} // namespace gptscope
