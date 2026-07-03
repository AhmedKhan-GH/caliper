#pragma once
// gpt_model.h — a nanoGPT-style char transformer, extended for mechanistic
// interpretability (GPTScope 2, id dev.caliper.gpt-scope 0.2.0).
//
// The attention is MANUAL on purpose: att = softmax(mask(QKᵀ/√d)) is an explicit
// tensor sequence, never torch's fused sdpa. That keeps the (n_head,T,T) weight
// matrix reachable for the probe. GPTScope 2 adds forward_full(), which returns —
// for a single probe batch (1,T) — the per-depth residual stream, the per-layer
// attention weights, AND the L2 write-norm of each sublayer into the residual
// stream. From those three the applet reconstructs the logit lens, head roles,
// embedding geometry, and the residual accounting. The training forward is
// UNCHANGED (need_weights=false retains nothing — the interp tax is paid only
// when someone is looking).
//
// Config is fixed: 4 layers / 4 heads / 128 embd, block 128, dropout 0.1. Only
// vocab_size is data-driven (built from the char corpus at train time).
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
        mask_ = register_buffer(
            "mask", torch::tril(torch::ones({cfg.block_size, cfg.block_size})));
    }

    // x: (B,T,C). need_weights=false on the training path (retains nothing).
    // When true and out_att != nullptr, appends this layer's (n_head,T,T)
    // attention for the FIRST batch element (the probe).
    torch::Tensor forward(const torch::Tensor& x, bool need_weights,
                          std::vector<torch::Tensor>* out_att) {
        const auto B = x.size(0), T = x.size(1), C = x.size(2);
        const auto hs = C / n_head_;
        auto qkv = c_attn_->forward(x).split(n_embd_, 2);       // 3 x (B,T,C)
        auto heads = [&](const torch::Tensor& t) {
            return t.view({B, T, n_head_, hs}).transpose(1, 2); // (B,nh,T,hs)
        };
        auto q = heads(qkv[0]), k = heads(qkv[1]), v = heads(qkv[2]);

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

    // Probe variant (eval + no_grad on the caller): collects the (nh,T,T) probe
    // attention AND the L2 norm of each sublayer's WRITE into the residual
    // stream, meaned over positions (batch element 0). Dropout is identity in
    // eval, so the probe write norms match the deterministic forward.
    torch::Tensor forward_probe(torch::Tensor x,
                                std::vector<torch::Tensor>* out_att,
                                float& attn_wnorm, float& mlp_wnorm) {
        auto a = attn_->forward(ln1_->forward(x), /*need_weights=*/true, out_att);
        attn_wnorm = a.select(0, 0).norm(2, -1).mean().item<float>();
        x = x + a;
        auto m = proj_->forward(torch::gelu(fc_->forward(ln2_->forward(x))));
        mlp_wnorm = m.select(0, 0).norm(2, -1).mean().item<float>();
        return x + m;
    }

private:
    torch::nn::LayerNorm ln1_;
    CausalSelfAttention  attn_;
    torch::nn::LayerNorm ln2_;
    torch::nn::Linear    fc_, proj_;
    torch::nn::Dropout   drop_;
};
TORCH_MODULE(Block);

// The whole mechanistic bundle a single probe forward returns (batch element 0).
struct ForwardFull {
    std::vector<torch::Tensor> resid;      // (T,C) per depth 0..n_layer (0 = emb)
    std::vector<torch::Tensor> attn;       // (n_head,T,T) per layer
    std::vector<float>         attn_wnorm; // per-layer attention write norm
    std::vector<float>         mlp_wnorm;  // per-layer MLP write norm
};

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

    // The mechanistic probe: run on idx (1,T), returning the residual stream at
    // every depth, per-layer attention, and per-layer sublayer write norms — for
    // batch element 0, on the model's device. eval()+no_grad so dropout never
    // perturbs the picture and no graph is retained; restores prior train/eval.
    // (The logit lens is then ln_f + lm_head applied to each resid[d].)
    ForwardFull forward_full(const torch::Tensor& idx) {
        torch::NoGradGuard ng;
        const bool was_training = is_training();
        eval();
        const auto T = idx.size(1);
        auto pos = torch::arange(
            T, torch::TensorOptions(idx.device()).dtype(torch::kLong));
        auto x = wte_->forward(idx) + wpe_->forward(pos);   // (1,T,C), no dropout
        ForwardFull ff;
        ff.resid.push_back(x.select(0, 0));                 // depth 0 = embeddings
        for (auto& b : block_vec_) {
            float an = 0.f, mn = 0.f;
            x = b->forward_probe(x, &ff.attn, an, mn);
            ff.resid.push_back(x.select(0, 0));             // (T,C) after block
            ff.attn_wnorm.push_back(an);
            ff.mlp_wnorm.push_back(mn);
        }
        if (was_training) train();
        return ff;
    }

    // Autoregressive sampling: seeded by idx (1,t0), grows by max_new tokens at
    // temperature. no_grad + eval; crops context to block_size. Restores mode.
    torch::Tensor generate(torch::Tensor idx, int64_t max_new,
                           double temperature) {
        torch::NoGradGuard ng;
        const bool was_training = is_training();
        eval();
        for (int64_t i = 0; i < max_new; ++i) {
            auto cond = idx.size(1) <= cfg_.block_size
                            ? idx
                            : idx.slice(1, idx.size(1) - cfg_.block_size);
            auto logits = forward(cond);
            auto last = logits.select(1, logits.size(1) - 1)
                        / std::max(temperature, 1e-6);
            auto probs = torch::softmax(last, -1);
            auto next = torch::multinomial(probs, 1);
            idx = torch::cat({idx, next}, 1);
        }
        if (was_training) train();
        return idx;
    }

    const GPTConfig& config() const { return cfg_; }
    torch::nn::LayerNorm& ln_f()   { return ln_f_; }   // the model's OWN final LN
    torch::nn::Linear&    lm_head(){ return lm_head_; }// the model's OWN unembed
    torch::nn::Embedding& wte()    { return wte_; }    // W_E for embedding PCA

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
// The applet facade (pImpl). The heavy State (mutex, snapshots, curl, the model)
// lives entirely in gpt_scope.cpp; plugin.cpp wraps this in a caliper::Applet.
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
