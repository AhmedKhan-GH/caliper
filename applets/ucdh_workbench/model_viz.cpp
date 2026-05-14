#include "model_viz.h"

#include <cstdio>
#include <cmath>
#include <algorithm>

// ============================================================================
// LAYOUT
// ============================================================================

static constexpr float NODE_W       = 520.0f;
static constexpr float NODE_GAP     = 14.0f;
static constexpr float ARROW_GAP    = 28.0f;
static constexpr float HEADER_H     = 26.0f;
static constexpr float LINE_H       = 16.0f;
static constexpr float PAD_BOT      = 10.0f;
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

    auto add = [&](const char* name, const char* type,
                   std::initializer_list<const char*> detail_lines,
                   const char* shape, int64_t params,
                   float pre_gap = 0.0f) -> int {
        y += pre_gap;
        float h = HEADER_H + (float)detail_lines.size() * LINE_H + PAD_BOT;
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
    int n_out = add("Softmax -> Diagnosis", "input",
        {"Probability distribution over PE / Normal"},
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
// DRAW
// ============================================================================

void ModelVisualizer::draw(ImVec2 avail) {
    if (!built_) build_repnet_graph();
    if (nodes_.empty()) return;

    float total_h = nodes_.back().y + nodes_.back().h + 40.0f;
    float cx = avail.x * 0.5f;

    ImGui::BeginChild("##model_scroll", avail, false,
                      ImGuiWindowFlags_NoBackground);

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Reserve space so scrollbar works
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

    // ── Edges (arrows between nodes) ──
    hovered_ = -1;
    ImVec2 mpos = ImGui::GetIO().MousePos;

    for (const auto& e : edges_) {
        const auto& src = nodes_[e.from];
        const auto& dst = nodes_[e.to];
        float ax = origin.x + off_x + NODE_W * 0.5f;
        float ay0 = origin.y + src.y + src.h;
        float ay1 = origin.y + dst.y;
        float mid_y = (ay0 + ay1) * 0.5f;

        dl->AddLine(ImVec2(ax, ay0), ImVec2(ax, ay1),
                    IM_COL32(90, 110, 140, 180), 2.0f);

        // Arrowhead
        float aw = 6.0f, ah = 8.0f;
        dl->AddTriangleFilled(
            ImVec2(ax, ay1),
            ImVec2(ax - aw, ay1 - ah),
            ImVec2(ax + aw, ay1 - ah),
            IM_COL32(90, 110, 140, 200));

        // Edge label
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

        // Hit test
        if (mpos.x >= p0.x && mpos.x <= p1.x &&
            mpos.y >= p0.y && mpos.y <= p1.y) {
            hovered_ = i;
        }

        bool hov = (i == hovered_);
        bool sel = (i == selected_);

        // Body
        dl->AddRectFilled(p0, p1, n.fill, ROUNDING);

        // Header bar
        ImVec2 hdr_br(p1.x, ny + HEADER_H);
        dl->AddRectFilled(p0, hdr_br, n.header_fill,
                          ROUNDING, ImDrawFlags_RoundCornersTop);

        // Border
        ImU32 bord = sel ? IM_COL32(255, 220, 80, 255) :
                     hov ? IM_COL32(200, 210, 230, 200) : n.border;
        float bw = (sel || hov) ? 2.0f : 1.0f;
        dl->AddRect(p0, p1, bord, ROUNDING, 0, bw);

        // Header separator
        dl->AddLine(ImVec2(nx, ny + HEADER_H),
                    ImVec2(nx + n.w, ny + HEADER_H),
                    IM_COL32(255, 255, 255, 30));

        // Name
        dl->AddText(ImVec2(nx + PAD_X, ny + 5.0f),
                    IM_COL32(255, 255, 255, 240), n.name.c_str());

        // Param count in header
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

        // Output shape
        if (!n.shape_out.empty()) {
            ImVec2 ts = ImGui::CalcTextSize(n.shape_out.c_str());
            dl->AddText(ImVec2(nx + n.w - ts.x - PAD_X, ny + n.h - PAD_BOT - 2),
                        IM_COL32(100, 210, 160, 200), n.shape_out.c_str());
        }
    }

    // Click handling
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0)) {
        selected_ = hovered_;
    }

    ImGui::EndChild();
}

// ============================================================================
// QUERY
// ============================================================================

const ModelNode* ModelVisualizer::get_node(int idx) const {
    if (idx < 0 || idx >= (int)nodes_.size()) return nullptr;
    return &nodes_[idx];
}
