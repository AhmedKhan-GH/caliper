// ============================================================================
// Caliper node-editor sandbox applet.
//
// A minimal Blueprints-style canvas built on thedmd/imgui-node-editor.
// Pre-populates with a few signal-processing flavored nodes (Source → Filter
// → Sink) so the wiring API is exercised end-to-end. From here you can:
//   - Drag from any output pin to any compatible input pin to make a link.
//   - Press Delete (or use the toolbar button) to remove the selected items.
//   - Click "Add Filter" to spawn another inline node.
//
// State model:
//   Nodes / Pins / Links are plain structs in std::vector. IDs are dense
//   monotonic uint64_ts wrapped by ed::NodeId/PinId/LinkId at the boundary.
//   We never let the editor own application state — the editor is just a
//   view onto our tables. This is the pattern from upstream's blueprints
//   sample, distilled.
// ============================================================================

#include "node_editor_applet.h"

#include <imgui.h>
#include <imgui_node_editor.h>

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

namespace ed = ax::NodeEditor;

namespace {

// ── Pin / Node / Link domain types ────────────────────────────────────────

enum class PinType { Signal, Number };

ImU32 pin_color(PinType t) {
    switch (t) {
        case PinType::Signal: return IM_COL32(120, 200, 255, 255); // cyan-ish
        case PinType::Number: return IM_COL32(255, 200, 110, 255); // amber
    }
    return IM_COL32(200, 200, 200, 255);
}

const char* pin_type_name(PinType t) {
    switch (t) {
        case PinType::Signal: return "signal";
        case PinType::Number: return "number";
    }
    return "?";
}

struct Pin {
    uint64_t      id;
    std::string   name;
    PinType       type;
    ed::PinKind   kind;          // Input / Output
};

struct Node {
    uint64_t          id;
    std::string       title;
    ImU32             header_color;
    std::vector<Pin>  inputs;
    std::vector<Pin>  outputs;
    ImVec2            initial_pos = ImVec2(0, 0);
    bool              positioned  = false;
};

struct Link {
    uint64_t id;
    uint64_t start_pin_id;
    uint64_t end_pin_id;
};

} // anonymous namespace

// ── State ─────────────────────────────────────────────────────────────────

struct NodeEditorApplet::State {
    ed::EditorContext* ctx = nullptr;
    uint64_t           next_id = 1;

    std::vector<Node> nodes;
    std::vector<Link> links;

    // Look up the pin record for an ID (returns nullptr if not found).
    // Used only for type-compatibility checks during link creation.
    const Pin* find_pin(uint64_t pin_id) const {
        for (auto& n : nodes) {
            for (auto& p : n.inputs)  if (p.id == pin_id) return &p;
            for (auto& p : n.outputs) if (p.id == pin_id) return &p;
        }
        return nullptr;
    }

    uint64_t mint() { return next_id++; }
};

// ── Helpers for building the starter graph ────────────────────────────────

namespace {

// id-minter lambda lets these helpers stay free functions (no need to see
// the private NodeEditorApplet::State type from this anonymous namespace).
template <class Mint>
Node make_source_node(Mint mint, ImVec2 pos) {
    Node n;
    n.id           = mint();
    n.title        = "ECG Source";
    n.header_color = IM_COL32(80, 130, 200, 220);
    n.outputs.push_back({mint(), "out",      PinType::Signal, ed::PinKind::Output});
    n.outputs.push_back({mint(), "rate Hz",  PinType::Number, ed::PinKind::Output});
    n.initial_pos  = pos;
    return n;
}

template <class Mint>
Node make_filter_node(Mint mint, ImVec2 pos, const char* title = "Bandpass") {
    Node n;
    n.id           = mint();
    n.title        = title;
    n.header_color = IM_COL32(120, 90, 180, 220);
    n.inputs.push_back({mint(), "in",       PinType::Signal, ed::PinKind::Input});
    n.inputs.push_back({mint(), "low Hz",   PinType::Number, ed::PinKind::Input});
    n.inputs.push_back({mint(), "high Hz",  PinType::Number, ed::PinKind::Input});
    n.outputs.push_back({mint(), "out",     PinType::Signal, ed::PinKind::Output});
    n.initial_pos  = pos;
    return n;
}

template <class Mint>
Node make_sink_node(Mint mint, ImVec2 pos) {
    Node n;
    n.id           = mint();
    n.title        = "Plot Sink";
    n.header_color = IM_COL32(180, 100, 110, 220);
    n.inputs.push_back({mint(), "in", PinType::Signal, ed::PinKind::Input});
    n.initial_pos  = pos;
    return n;
}

void draw_pin_row(const Pin& pin) {
    const ImU32 col = pin_color(pin.type);
    if (pin.kind == ed::PinKind::Input) {
        ed::BeginPin(pin.id, ed::PinKind::Input);
            ImGui::TextColored(ImColor(col), "-> ");
            ImGui::SameLine(0, 0);
            ImGui::TextUnformatted(pin.name.c_str());
        ed::EndPin();
    } else {
        // Right-align outputs by padding to the available width.
        const float text_w = ImGui::CalcTextSize(pin.name.c_str()).x
                           + ImGui::CalcTextSize(" ->").x;
        const float avail  = ImGui::GetContentRegionAvail().x;
        if (avail > text_w) ImGui::Dummy(ImVec2(avail - text_w, 0));
        ImGui::SameLine(0, 0);
        ed::BeginPin(pin.id, ed::PinKind::Output);
            ImGui::TextUnformatted(pin.name.c_str());
            ImGui::SameLine(0, 0);
            ImGui::TextColored(ImColor(col), " ->");
        ed::EndPin();
    }
}

void draw_node(const Node& n) {
    ed::BeginNode(n.id);

    // Header row — colored title spanning full node width.
    ImGui::PushStyleColor(ImGuiCol_Text, n.header_color);
    ImGui::TextUnformatted(n.title.c_str());
    ImGui::PopStyleColor();
    ImGui::Dummy(ImVec2(140, 2));   // minimum node width
    ImGui::Separator();

    // Body — inputs in a left column, outputs in a right column. ImGui
    // doesn't lay them out side-by-side cleanly inside a node, so we just
    // alternate rows (input then matching output) when both exist, else
    // print remaining rows from the longer side.
    const size_t rows = std::max(n.inputs.size(), n.outputs.size());
    for (size_t r = 0; r < rows; r++) {
        if (r < n.inputs.size())  draw_pin_row(n.inputs[r]);
        if (r < n.outputs.size()) draw_pin_row(n.outputs[r]);
    }

    ed::EndNode();
}

} // anonymous namespace

// ── Lifecycle ─────────────────────────────────────────────────────────────

bool NodeEditorApplet::initialize() {
    s_ = new State();

    ed::Config cfg;
    cfg.SettingsFile = "node_editor.json";    // persists pan/zoom/positions
    s_->ctx = ed::CreateEditor(&cfg);

    // Seed with a small starter graph.
    auto mint = [s = s_]() { return s->mint(); };
    s_->nodes.push_back(make_source_node(mint, ImVec2( 60, 100)));
    s_->nodes.push_back(make_filter_node(mint, ImVec2(320, 100)));
    s_->nodes.push_back(make_sink_node  (mint, ImVec2(620, 140)));

    // Connect Source.out → Filter.in → Sink.in.
    if (s_->nodes.size() == 3) {
        const auto& src    = s_->nodes[0];
        const auto& filt   = s_->nodes[1];
        const auto& sink   = s_->nodes[2];
        s_->links.push_back({s_->mint(), src.outputs[0].id, filt.inputs[0].id});
        s_->links.push_back({s_->mint(), filt.outputs[0].id, sink.inputs[0].id});
    }

    return true;
}

void NodeEditorApplet::cleanup() {
    if (!s_) return;
    if (s_->ctx) ed::DestroyEditor(s_->ctx);
    delete s_;
    s_ = nullptr;
}

// ── Draw ──────────────────────────────────────────────────────────────────

void NodeEditorApplet::draw_ui(int win_w, int win_h) {
    if (!s_) return;

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("##NodeEditorRoot", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    // ── Toolbar ──
    if (ImGui::Button("<< Back to Menu", ImVec2(140, 28))) {
        exit_requested_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Filter", ImVec2(110, 28))) {
        // Place new nodes near canvas origin; user can drag from there.
        auto mint = [s = s_]() { return s->mint(); };
        s_->nodes.push_back(make_filter_node(mint, ImVec2(80, 360), "Filter"));
    }
    ImGui::SameLine();
    if (ImGui::Button("Fit to Content", ImVec2(120, 28))) {
        ed::SetCurrentEditor(s_->ctx);
        ed::NavigateToContent();
        ed::SetCurrentEditor(nullptr);
    }
    ImGui::SameLine();
    ImGui::TextDisabled(" |  drag pin → pin to link  ·  select + Delete to remove");

    ImGui::Separator();

    // ── Editor canvas ──
    ed::SetCurrentEditor(s_->ctx);
    ed::Begin("Caliper Sandbox", ImVec2(0, 0));

    // Apply initial positions exactly once per node (after that the editor
    // owns layout and persists it via the settings file).
    for (auto& n : s_->nodes) {
        if (!n.positioned) {
            ed::SetNodePosition(n.id, n.initial_pos);
            n.positioned = true;
        }
    }

    // Submit nodes.
    for (const auto& n : s_->nodes) draw_node(n);

    // Submit links.
    for (const auto& lk : s_->links) {
        ed::Link(lk.id, lk.start_pin_id, lk.end_pin_id);
    }

    // ── Handle link creation ──
    if (ed::BeginCreate()) {
        ed::PinId start_pin, end_pin;
        if (ed::QueryNewLink(&start_pin, &end_pin)) {
            // Both pins valid?
            if (start_pin && end_pin) {
                const Pin* a = s_->find_pin(start_pin.Get());
                const Pin* b = s_->find_pin(end_pin.Get());

                bool ok = a && b;
                if (ok && a->kind == b->kind)         ok = false; // in→in / out→out
                if (ok && a->type != b->type)         ok = false; // type mismatch
                if (ok && start_pin == end_pin)       ok = false; // self-loop on same pin

                if (!ok) {
                    ed::RejectNewItem(ImColor(255, 80, 80), 2.0f);
                } else if (ed::AcceptNewItem(ImColor(120, 255, 120), 2.0f)) {
                    s_->links.push_back({
                        s_->mint(), start_pin.Get(), end_pin.Get()
                    });
                }
            }
        }
    }
    ed::EndCreate();

    // ── Handle deletions ──
    if (ed::BeginDelete()) {
        ed::LinkId deleted_link;
        while (ed::QueryDeletedLink(&deleted_link)) {
            if (ed::AcceptDeletedItem()) {
                auto it = std::find_if(s_->links.begin(), s_->links.end(),
                    [&](const Link& l) { return l.id == deleted_link.Get(); });
                if (it != s_->links.end()) s_->links.erase(it);
            }
        }
        ed::NodeId deleted_node;
        while (ed::QueryDeletedNode(&deleted_node)) {
            if (ed::AcceptDeletedItem()) {
                // Erase the node and any links that referenced its pins.
                auto nit = std::find_if(s_->nodes.begin(), s_->nodes.end(),
                    [&](const Node& n) { return n.id == deleted_node.Get(); });
                if (nit != s_->nodes.end()) {
                    std::vector<uint64_t> orphan_pins;
                    for (auto& p : nit->inputs)  orphan_pins.push_back(p.id);
                    for (auto& p : nit->outputs) orphan_pins.push_back(p.id);
                    s_->links.erase(std::remove_if(s_->links.begin(), s_->links.end(),
                        [&](const Link& lk) {
                            return std::find(orphan_pins.begin(), orphan_pins.end(),
                                       lk.start_pin_id) != orphan_pins.end()
                                || std::find(orphan_pins.begin(), orphan_pins.end(),
                                       lk.end_pin_id)   != orphan_pins.end();
                        }),
                        s_->links.end());
                    s_->nodes.erase(nit);
                }
            }
        }
    }
    ed::EndDelete();

    ed::End();
    ed::SetCurrentEditor(nullptr);

    ImGui::End();
    (void)win_w; (void)win_h;
}
