#include "model_viz.h"

#include <cstdio>
#include <cmath>
#include <algorithm>

// ============================================================================
// LAYOUT
// ============================================================================

static constexpr float NODE_W       = 540.0f;
static constexpr float NODE_GAP     = 12.0f;
static constexpr float ARROW_GAP    = 28.0f;
static constexpr float HEADER_H     = 26.0f;
static constexpr float LINE_H       = 16.0f;
static constexpr float PAD_BOT      = 8.0f;
static constexpr float ACT_H        = 36.0f;
static constexpr float PAD_X        = 12.0f;
static constexpr float MIN_NODE_H   = 40.0f;
static constexpr float ROUNDING     = 8.0f;
static constexpr float STAGE_PAD    = 10.0f;
static constexpr float STAGE_GAP    = 18.0f;

// ============================================================================
// COLORS
// ============================================================================

static ImU32 fill_for(const std::string& t) {
    if (t == "conv")       return IM_COL32(25, 55, 130, 255);
    if (t == "attention")  return IM_COL32(140, 75, 5, 255);
    if (t == "fusion")     return IM_COL32(10, 90, 65, 255);
    if (t == "pool")       return IM_COL32(75, 35, 140, 255);
    if (t == "linear")     return IM_COL32(50, 45, 140, 255);
    if (t == "dropout")    return IM_COL32(50, 55, 65, 255);
    return IM_COL32(40, 50, 60, 255);
}

static ImU32 border_for(const std::string& t) {
    if (t == "conv")       return IM_COL32(60, 130, 245, 255);
    if (t == "attention")  return IM_COL32(245, 160, 20, 255);
    if (t == "fusion")     return IM_COL32(20, 190, 130, 255);
    if (t == "pool")       return IM_COL32(140, 90, 245, 255);
    if (t == "linear")     return IM_COL32(100, 100, 245, 255);
    if (t == "dropout")    return IM_COL32(110, 120, 140, 255);
    return IM_COL32(115, 130, 150, 255);
}

static ImU32 header_for(const std::string& t) {
    if (t == "conv")       return IM_COL32(35, 75, 170, 255);
    if (t == "attention")  return IM_COL32(170, 95, 10, 255);
    if (t == "fusion")     return IM_COL32(15, 115, 85, 255);
    if (t == "pool")       return IM_COL32(95, 50, 170, 255);
    if (t == "linear")     return IM_COL32(65, 60, 170, 255);
    if (t == "dropout")    return IM_COL32(65, 70, 80, 255);
    return IM_COL32(55, 65, 80, 255);
}

// ============================================================================
// GRAPH
// ============================================================================

void ModelVisualizer::build_repnet_graph() {
    nodes_.clear();
    edges_.clear();
    stages_.clear();

    float y = 20.0f;
    bool has_act = true;

    auto add = [&](const char* name, const char* type,
                   std::initializer_list<const char*> detail_lines,
                   const char* shape, int64_t params,
                   float pre_gap = 0.0f) -> int {
        y += pre_gap;
        float h = HEADER_H + (float)detail_lines.size() * LINE_H + PAD_BOT;
        if (has_act) h += ACT_H;
        h = std::max(h, MIN_NODE_H);
        int idx = (int)nodes_.size();
        ModelNode n;
        n.name = name;
        n.type = type;
        n.lines.assign(detail_lines.begin(), detail_lines.end());
        n.shape_out = shape;
        n.x = 0;
        n.y = y;
        n.w = NODE_W;
        n.h = h;
        n.fill = fill_for(type);
        n.border = border_for(type);
        n.header_fill = header_for(type);
        n.param_count = params;
        nodes_.push_back(std::move(n));
        y += h + ARROW_GAP + NODE_GAP;
        return idx;
    };

    auto edge = [&](int a, int b, const char* lbl = "") {
        edges_.push_back({a, b, lbl});
    };

    // ── Input ──
    int n_in = add("Input: 12-Lead ECG", "input",
        {"12 leads: I, II, III, aVR, aVL, aVF, V1-V6",
         "250 Hz sampling, up to 2500 samples/lead"},
        "(B, 12, T)", 0);

    // ── Stage 0 ──
    float s0_top = y;
    int n_c0 = add("PerLeadConvBlock  [x12 leads]", "conv",
        {"main:  Conv1d(1->48, k=7) + BN + ReLU -> Conv1d(48->48, k=7) + BN",
         "skip:  Conv1d(1->48, k=1) + BN",
         "merge: Add(main, skip) -> ReLU -> Dropout(0.064) -> MaxPool(2)"},
        "(B,12,48,T/2)", 16944, STAGE_GAP);
    int n_a0 = add("CrossLeadAttention", "attention",
        {"pool:  AdaptiveAvgPool1d(1) -> 12 tokens of dim 48",
         "attn:  MultiheadAttention(embed=48, heads=4)",
         "gate:  residual + LayerNorm -> Linear(48->48) + Sigmoid -> x*gate"},
        "(B,12,48,T/2)", 11856);
    float s0_bot = nodes_.back().y + nodes_.back().h;

    // ── Stage 1 ──
    float s1_top = y;
    int n_c1 = add("PerLeadConvBlock  [x12 leads]", "conv",
        {"main:  Conv1d(48->96, k=5) + BN + ReLU -> Conv1d(96->96, k=5) + BN",
         "skip:  Conv1d(48->96, k=1) + BN",
         "merge: Add(main, skip) -> ReLU -> Dropout(0.064) -> MaxPool(2)"},
        "(B,12,96,T/4)", 74592, STAGE_GAP);
    int n_a1 = add("CrossLeadAttention", "attention",
        {"pool:  AdaptiveAvgPool1d(1) -> 12 tokens of dim 96",
         "attn:  MultiheadAttention(embed=96, heads=4)",
         "gate:  residual + LayerNorm -> Linear(96->96) + Sigmoid -> x*gate"},
        "(B,12,96,T/4)", 46752);
    float s1_bot = nodes_.back().y + nodes_.back().h;

    // ── Stage 2 ──
    float s2_top = y;
    int n_c2 = add("PerLeadConvBlock  [x12 leads]", "conv",
        {"main:  Conv1d(96->192, k=3) + BN + ReLU -> Conv1d(192->192, k=3) + BN",
         "skip:  Conv1d(96->192, k=1) + BN",
         "merge: Add(main, skip) -> ReLU -> Dropout(0.064) -> MaxPool(2)"},
        "(B,12,192,T/8)", 186048, STAGE_GAP);
    int n_a2 = add("CrossLeadAttention", "attention",
        {"pool:  AdaptiveAvgPool1d(1) -> 12 tokens of dim 192",
         "attn:  MultiheadAttention(embed=192, heads=4)",
         "gate:  residual + LayerNorm -> Linear(192->192) + Sigmoid -> x*gate"},
        "(B,12,192,T/8)", 185664);
    float s2_bot = nodes_.back().y + nodes_.back().h;

    // ── Fusion + Head ──
    int n_reshape = add("Lead Concatenation", "fusion",
        {"Reshape: (B,12,192,T/8) -> (B, 2304, T/8)"},
        "(B,2304,T/8)", 0, STAGE_GAP);
    int n_fuse = add("Fusion Conv1d", "fusion",
        {"Conv1d(2304->192, k=1) + BatchNorm + ReLU"},
        "(B,192,T/8)", 442944);
    int n_gap = add("Global Average Pooling", "pool",
        {"AdaptiveAvgPool1d(1): collapse temporal -> single vector"},
        "(B,192)", 0);
    int n_drop = add("Dropout (p=0.064)", "dropout", {}, "(B,192)", 0);
    int n_fc = add("Classifier: Linear(192 -> 2)", "linear",
        {"Output classes: PE (Pulmonary Embolism) vs Normal"},
        "(B,2)", 386);
    int n_out = add("Diagnosis Output", "input",
        {"Softmax probability: PE vs Normal"},
        "(B,2)", 0);

    // ── Edges ──
    edge(n_in,      n_c0,      "unsqueeze -> (B,12,1,T)");
    edge(n_c0,      n_a0,      "(B,12,48,T/2)");
    edge(n_a0,      n_c1,      "(B,12,48,T/2)");
    edge(n_c1,      n_a1,      "(B,12,96,T/4)");
    edge(n_a1,      n_c2,      "(B,12,96,T/4)");
    edge(n_c2,      n_a2,      "(B,12,192,T/8)");
    edge(n_a2,      n_reshape, "(B,12,192,T/8)");
    edge(n_reshape, n_fuse,    "(B,2304,T/8)");
    edge(n_fuse,    n_gap,     "(B,192,T/8)");
    edge(n_gap,     n_drop,    "(B,192)");
    edge(n_drop,    n_fc,      "(B,192)");
    edge(n_fc,      n_out,     "(B,2)");

    // ── Stages ──
    stages_.push_back({s0_top, s0_bot, "Stage 0 — QRS-level features (RF = 32 samples)"});
    stages_.push_back({s1_top, s1_bot, "Stage 1 — ST-level features (RF = 40 samples)"});
    stages_.push_back({s2_top, s2_bot, "Stage 2 — Morphology-level features (RF = 49 samples)"});

    built_ = true;
}

// ============================================================================
// DRAW HELPERS
// ============================================================================

static void draw_activation_bar(ImDrawList* dl, float x, float y, float w,
                                const LayerActivation& act) {
    float bar_h = 10.0f;
    float bar_w = w - PAD_X * 2;

    // Background
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + bar_w, y + bar_h),
                      IM_COL32(15, 15, 25, 200), 3.0f);

    // Activation magnitude bar — scale mean to bar width
    float range = act.max_val - act.min_val;
    if (range < 1e-6f) range = 1.0f;
    float fill_frac = std::clamp((act.mean - act.min_val) / range, 0.0f, 1.0f);
    float fill_w = bar_w * fill_frac;

    // Color based on activation: blue -> green -> yellow
    int r = (int)(50 + 180 * fill_frac);
    int g = (int)(180 - 60 * std::abs(fill_frac - 0.5f));
    int b = (int)(220 * (1.0f - fill_frac));
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + fill_w, y + bar_h),
                      IM_COL32(r, g, b, 200), 3.0f);

    // +/- 1 std dev marker
    float mean_x = x + bar_w * fill_frac;
    float std_px = (act.stddev / range) * bar_w;
    float lo = std::max(x, mean_x - std_px);
    float hi = std::min(x + bar_w, mean_x + std_px);
    dl->AddRectFilled(ImVec2(lo, y + 1), ImVec2(hi, y + bar_h - 1),
                      IM_COL32(255, 255, 255, 40), 2.0f);
}

static void draw_prob_bars(ImDrawList* dl, float x, float y, float w,
                           float prob_normal, float prob_pe) {
    float bar_w = w - PAD_X * 2;
    float bar_h = 12.0f;

    // Normal bar
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + bar_w, y + bar_h),
                      IM_COL32(15, 15, 25, 200), 3.0f);
    float nw = bar_w * prob_normal;
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + nw, y + bar_h),
                      IM_COL32(60, 200, 120, 220), 3.0f);
    char nb[32];
    std::snprintf(nb, sizeof(nb), "Normal: %.1f%%", prob_normal * 100);
    dl->AddText(ImVec2(x + 4, y), IM_COL32(255, 255, 255, 230), nb);

    y += bar_h + 3;

    // PE bar
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + bar_w, y + bar_h),
                      IM_COL32(15, 15, 25, 200), 3.0f);
    float pw = bar_w * prob_pe;
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + pw, y + bar_h),
                      IM_COL32(230, 80, 80, 220), 3.0f);
    char pb[32];
    std::snprintf(pb, sizeof(pb), "PE: %.1f%%", prob_pe * 100);
    dl->AddText(ImVec2(x + 4, y), IM_COL32(255, 255, 255, 230), pb);
}

// ============================================================================
// DRAW
// ============================================================================

void ModelVisualizer::draw(ImVec2 avail, const InferenceOverlay* overlay) {
    if (!built_) build_repnet_graph();
    if (nodes_.empty()) return;

    float total_h = nodes_.back().y + nodes_.back().h + 40.0f;
    float cx = avail.x * 0.5f;

    ImGui::BeginChild("##model_scroll", avail, false,
                      ImGuiWindowFlags_NoBackground);

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    ImGui::Dummy(ImVec2(NODE_W, total_h));

    float off_x = cx - NODE_W * 0.5f;

    // ── Stage group backgrounds ──
    for (const auto& sg : stages_) {
        float sy0 = origin.y + sg.y_top - STAGE_PAD;
        float sy1 = origin.y + sg.y_bot + STAGE_PAD;
        float sx0 = origin.x + off_x - STAGE_PAD;
        float sx1 = origin.x + off_x + NODE_W + STAGE_PAD;
        dl->AddRectFilled(ImVec2(sx0, sy0), ImVec2(sx1, sy1),
                          IM_COL32(40, 60, 100, 40), ROUNDING + 4);
        dl->AddRect(ImVec2(sx0, sy0), ImVec2(sx1, sy1),
                    IM_COL32(60, 90, 140, 60), ROUNDING + 4, 0, 1.0f);
        dl->AddText(ImVec2(sx0 + 8, sy0 + 4),
                    IM_COL32(130, 160, 210, 140), sg.label.c_str());
    }

    // ── Edges ──
    hovered_ = -1;
    ImVec2 mpos = ImGui::GetIO().MousePos;
    bool has_overlay = overlay && overlay->valid;

    for (const auto& e : edges_) {
        const auto& src = nodes_[e.from];
        const auto& dst = nodes_[e.to];
        float ax = origin.x + off_x + NODE_W * 0.5f;
        float ay0 = origin.y + src.y + src.h;
        float ay1 = origin.y + dst.y;
        float mid_y = (ay0 + ay1) * 0.5f;

        ImU32 edge_col = has_overlay
            ? IM_COL32(80, 200, 140, 200) : IM_COL32(90, 110, 140, 180);
        float edge_w = has_overlay ? 2.5f : 2.0f;

        dl->AddLine(ImVec2(ax, ay0), ImVec2(ax, ay1), edge_col, edge_w);

        float aw = 6.0f, ah = 8.0f;
        dl->AddTriangleFilled(
            ImVec2(ax, ay1),
            ImVec2(ax - aw, ay1 - ah),
            ImVec2(ax + aw, ay1 - ah), edge_col);

        if (!e.label.empty()) {
            ImVec2 ts = ImGui::CalcTextSize(e.label.c_str());
            float lx = ax + 12.0f;
            float ly = mid_y - ts.y * 0.5f;
            dl->AddRectFilled(ImVec2(lx - 4, ly - 2),
                              ImVec2(lx + ts.x + 4, ly + ts.y + 2),
                              IM_COL32(20, 25, 35, 200), 4.0f);
            dl->AddText(ImVec2(lx, ly),
                        IM_COL32(120, 190, 140, 220), e.label.c_str());
        }
    }

    // ── Nodes ──
    for (int i = 0; i < (int)nodes_.size(); i++) {
        const auto& n = nodes_[i];
        float nx = origin.x + off_x;
        float ny = origin.y + n.y;

        ImVec2 p0(nx, ny);
        ImVec2 p1(nx + n.w, ny + n.h);

        if (mpos.x >= p0.x && mpos.x <= p1.x &&
            mpos.y >= p0.y && mpos.y <= p1.y)
            hovered_ = i;

        bool hov = (i == hovered_);
        bool sel = (i == selected_);

        // Body
        dl->AddRectFilled(p0, p1, n.fill, ROUNDING);

        // Header bar
        ImVec2 hdr_br(p1.x, ny + HEADER_H);
        dl->AddRectFilled(p0, hdr_br, n.header_fill,
                          ROUNDING, ImDrawFlags_RoundCornersTop);

        // Border — glow when live inference is active
        ImU32 bord = sel ? IM_COL32(255, 220, 80, 255) :
                     hov ? IM_COL32(200, 210, 230, 200) :
                     has_overlay ? IM_COL32(80, 200, 140, 160) : n.border;
        float bw = (sel || hov) ? 2.0f : (has_overlay ? 1.5f : 1.0f);
        dl->AddRect(p0, p1, bord, ROUNDING, 0, bw);

        dl->AddLine(ImVec2(nx, ny + HEADER_H),
                    ImVec2(nx + n.w, ny + HEADER_H),
                    IM_COL32(255, 255, 255, 30));

        // Name
        dl->AddText(ImVec2(nx + PAD_X, ny + 5.0f),
                    IM_COL32(255, 255, 255, 240), n.name.c_str());

        // Param count
        if (n.param_count > 0) {
            char buf[64];
            if (n.param_count >= 1000000)
                std::snprintf(buf, sizeof(buf), "%.1fM", n.param_count / 1e6);
            else if (n.param_count >= 1000)
                std::snprintf(buf, sizeof(buf), "%.1fK", n.param_count / 1e3);
            else
                std::snprintf(buf, sizeof(buf), "%lld", (long long)n.param_count);
            ImVec2 ts = ImGui::CalcTextSize(buf);
            dl->AddText(ImVec2(nx + n.w - ts.x - PAD_X, ny + 5.0f),
                        IM_COL32(180, 190, 210, 150), buf);
        }

        // Detail lines
        float ly = ny + HEADER_H + 4.0f;
        for (const auto& line : n.lines) {
            dl->AddText(ImVec2(nx + PAD_X, ly),
                        IM_COL32(190, 200, 215, 190), line.c_str());
            ly += LINE_H;
        }

        // ── Activation overlay ──
        float act_y = ny + n.h - ACT_H - PAD_BOT + 4;
        bool has_act = has_overlay && i < (int)overlay->layers.size()
                       && overlay->layers[i].valid;

        if (has_act) {
            const auto& act = overlay->layers[i];
            bool is_output = (i == (int)nodes_.size() - 1);

            // Separator line
            dl->AddLine(ImVec2(nx + 4, act_y - 2),
                        ImVec2(nx + n.w - 4, act_y - 2),
                        IM_COL32(255, 255, 255, 25));

            if (is_output && overlay->result_class >= 0) {
                draw_prob_bars(dl, nx + PAD_X, act_y + 2, n.w,
                               overlay->probs[0], overlay->probs[1]);
            } else {
                draw_activation_bar(dl, nx + PAD_X, act_y + 2, n.w, act);

                char stats[128];
                std::snprintf(stats, sizeof(stats),
                              "\xce\xbc=%.3f  \xcf\x83=%.3f  [%.2f, %.2f]",
                              act.mean, act.stddev, act.min_val, act.max_val);
                dl->AddText(ImVec2(nx + PAD_X, act_y + 15),
                            IM_COL32(170, 185, 200, 180), stats);

                if (!act.shape.empty()) {
                    ImVec2 ts = ImGui::CalcTextSize(act.shape.c_str());
                    dl->AddText(ImVec2(nx + n.w - ts.x - PAD_X, act_y + 15),
                                IM_COL32(100, 210, 160, 200), act.shape.c_str());
                }
            }
        } else if (!has_overlay) {
            // No model loaded hint
            dl->AddText(ImVec2(nx + PAD_X, act_y + 6),
                        IM_COL32(100, 110, 130, 80),
                        "Load model for live activations");
        } else {
            dl->AddText(ImVec2(nx + PAD_X, act_y + 6),
                        IM_COL32(100, 110, 130, 100), "...");
        }

        // Output shape (static)
        if (!n.shape_out.empty() && !has_act) {
            ImVec2 ts = ImGui::CalcTextSize(n.shape_out.c_str());
            dl->AddText(ImVec2(nx + n.w - ts.x - PAD_X, ny + n.h - PAD_BOT - 2),
                        IM_COL32(100, 210, 160, 200), n.shape_out.c_str());
        }
    }

    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
        selected_ = hovered_;

    ImGui::EndChild();
}

// ============================================================================
// QUERY
// ============================================================================

const ModelNode* ModelVisualizer::get_node(int idx) const {
    if (idx < 0 || idx >= (int)nodes_.size()) return nullptr;
    return &nodes_[idx];
}
