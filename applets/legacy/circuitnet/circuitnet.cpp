#include "circuitnet.h"
#include "circuit_db.h"
#include "circuit_viz.h"
#include "verilog_parser.h"
#include "app_paths.h"

#include <imgui.h>
#include <imgui_node_editor.h>
#include <implot.h>
#include <implot3d.h>
#include <ImGuiFileDialog.h>

namespace ned = ax::NodeEditor;

#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace fs = std::filesystem;

// ============================================================================

struct CircuitNetApplet::State {
    CircuitDB db;
    std::vector<DesignEntry> designs;
    int selected_design = -1;

    // Current design data
    CircuitGraph current_graph;
    GraphLayout current_layout;
    std::string current_netlist_source;

    // Graph view state
    int selected_gate = -1;
    enum ColorMode { ByType, ByDelay, ByFanout, BySlew } color_mode = ByType;

    // RTL module view
    VerilogModule rtl_module;
    ned::EditorContext* module_editor_ctx = nullptr;
    int module_layout_frames = -1;

    // Statistics cache
    std::vector<float> power_values;
    std::unordered_map<std::string, int> cell_type_counts;

    // Dataset loading
    std::atomic<bool> loading{false};
    std::atomic<int> load_progress{0};
    std::atomic<int> load_total{0};
    std::string dataset_path;
    bool db_ready = false;

    // Node editor
    ned::EditorContext* node_editor_ctx = nullptr;
    int layout_frames = -1;

    // UI state
    int active_tab = 0;
    char filter_buf[256] = "";
};

// ============================================================================

CircuitNetApplet::CircuitNetApplet() : s_(std::make_unique<State>()) {}
CircuitNetApplet::~CircuitNetApplet() = default;

bool CircuitNetApplet::initialize() {
    s_->db.open();
    ned::Config config;
    config.SettingsFile = nullptr;
    config.NavigateButtonIndex = 1;
    config.SelectButtonIndex = 0;
    config.DragButtonIndex = 0;
    config.ContextMenuButtonIndex = 2;
    config.EnableSmoothZoom = true;
    s_->node_editor_ctx = ned::CreateEditor(&config);
    s_->module_editor_ctx = ned::CreateEditor(&config);

    std::ifstream f(caliper::app_data_path("circuitnet_dataset.txt"));
    std::string last_dir;
    if (f.is_open() && std::getline(f, last_dir) && fs::is_directory(last_dir)) {
        open_dataset(last_dir);
    }

    return true;
}

void CircuitNetApplet::draw_ui(int /*win_w*/, int /*win_h*/) {
    auto* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("CircuitNet 3.0 Explorer", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_MenuBar);

    // Menu bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Dataset...")) {
                IGFD::FileDialogConfig config;
                config.path = ".";
                config.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseDataset", "Select CircuitNet Dataset Directory", nullptr, config);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Color: Cell Type", nullptr, s_->color_mode == State::ByType))
                s_->color_mode = State::ByType;
            if (ImGui::MenuItem("Color: Delay", nullptr, s_->color_mode == State::ByDelay))
                s_->color_mode = State::ByDelay;
            if (ImGui::MenuItem("Color: Fanout", nullptr, s_->color_mode == State::ByFanout))
                s_->color_mode = State::ByFanout;
            if (ImGui::MenuItem("Color: Slew", nullptr, s_->color_mode == State::BySlew))
                s_->color_mode = State::BySlew;
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // File dialog handling — must be called every frame regardless of state
    ImVec2 min_sz(600, 400);
    ImVec2 max_sz(FLT_MAX, FLT_MAX);
    if (ImGuiFileDialog::Instance()->Display("ChooseDataset",
            ImGuiWindowFlags_NoCollapse, min_sz, max_sz)) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path = ImGuiFileDialog::Instance()->GetCurrentPath();
            open_dataset(path);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Loading indicator
    if (s_->loading) {
        ImGui::Text("Loading dataset... %d / %d designs", s_->load_progress.load(), s_->load_total.load());
        float frac = s_->load_total > 0 ? (float)s_->load_progress / s_->load_total : 0;
        ImGui::ProgressBar(frac);
        ImGui::End();
        return;
    }

    if (!s_->db_ready) {
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextWrapped("Open a CircuitNet 3.0 dataset directory to begin.");
        ImGui::TextWrapped("Expected structure: dataset/Final/{design_name}/{feature.json, final_netlist.v, power_summary.txt}");
        ImGui::Spacing();
        if (ImGui::Button("Open Dataset...")) {
            IGFD::FileDialogConfig config;
            config.path = ".";
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog("ChooseDataset", "Select CircuitNet Dataset Directory", nullptr, config);
        }
        ImGui::End();
        return;
    }

    // Left sidebar: design browser (always visible)
    float sidebar_w = 300;
    ImGui::BeginChild("##sidebar", {sidebar_w, 0}, true);
    draw_browser_panel();
    ImGui::EndChild();

    ImGui::SameLine();

    // Right: tabbed views
    ImGui::BeginChild("##main_view", {0, 0}, false);
    draw_design_info();
    if (ImGui::BeginTabBar("MainTabs")) {
        if (ImGui::BeginTabItem("Circuit Graph")) {
            draw_circuit_graph();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Module View")) {
            draw_module_view();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Netlist")) {
            draw_netlist_viewer();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("RTL Source")) {
            if (s_->rtl_module.valid && !s_->rtl_module.source.empty()) {
                ImGui::Text("RTL Verilog (%s, %d bytes)",
                            s_->rtl_module.name.c_str(), (int)s_->rtl_module.source.size());
                ImGui::Separator();
                ImGui::BeginChild("##rtl_scroll", {0, 0}, false, ImGuiWindowFlags_HorizontalScrollbar);
                ImGui::TextUnformatted(s_->rtl_module.source.c_str(),
                                       s_->rtl_module.source.c_str() + s_->rtl_module.source.size());
                ImGui::EndChild();
            } else {
                ImGui::TextWrapped("No RTL source found for this design.");
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Statistics")) {
            draw_statistics();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    ImGui::End();
}

void CircuitNetApplet::cleanup() {
    if (s_->module_editor_ctx) {
        ned::DestroyEditor(s_->module_editor_ctx);
    }
    if (s_->node_editor_ctx) {
        ned::DestroyEditor(s_->node_editor_ctx);
    }
    s_.reset();
    s_ = std::make_unique<State>();
}

// ============================================================================
// Dataset loading
// ============================================================================

void CircuitNetApplet::open_dataset(const std::string& dir) {
    s_->dataset_path = dir;
    s_->loading = true;
    s_->load_progress = 0;

    std::ofstream f(caliper::app_data_path("circuitnet_dataset.txt"));
    if (f.is_open()) f << dir;

    std::thread([this]() {
        s_->db.ingest_dataset(s_->dataset_path, [this](int cur, int total) {
            s_->load_progress = cur;
            s_->load_total = total;
        });

        s_->designs = s_->db.get_designs(100000, 0);
        s_->db_ready = true;
        s_->loading = false;

        // Cache power values for statistics
        auto res = s_->db.query("SELECT total_power FROM designs WHERE total_power > 0");
        if (res.ok) {
            s_->power_values.clear();
            for (auto& row : res.rows) {
                try { s_->power_values.push_back(std::stof(row[0])); } catch (...) {}
            }
        }
    }).detach();
}

void CircuitNetApplet::select_design(int idx) {
    if (idx < 0 || idx >= (int)s_->designs.size()) return;
    s_->selected_design = idx;
    parse_current_netlist();
}

void CircuitNetApplet::parse_current_netlist() {
    if (s_->selected_design < 0) return;
    auto& design = s_->designs[s_->selected_design];

    fs::path dir(design.path);
    fs::path netlist_path = dir / "final_netlist.v";
    fs::path feature_path = dir / "feature.json";

    s_->current_graph = parse_verilog_netlist(netlist_path.string());

    if (fs::exists(feature_path)) {
        annotate_features(s_->current_graph, feature_path.string());
    }

    s_->current_graph.total_power = design.total_power;

    // Read netlist source for viewer
    if (fs::exists(netlist_path)) {
        std::ifstream f(netlist_path);
        std::ostringstream ss;
        ss << f.rdbuf();
        s_->current_netlist_source = ss.str();
    }

    s_->current_layout = compute_layout(s_->current_graph, 200, 80, 100, 40);

    // Cell type distribution
    s_->cell_type_counts.clear();
    for (auto& g : s_->current_graph.gates) {
        s_->cell_type_counts[g.cell_type]++;
    }

    s_->selected_gate = -1;
    s_->layout_frames = 0;

    // Load RTL module if available
    std::string rtl_path = find_rtl_file(design.path);
    if (!rtl_path.empty()) {
        s_->rtl_module = parse_rtl_module(rtl_path);
        s_->module_layout_frames = 0;
    } else {
        s_->rtl_module = {};
    }
}

// ============================================================================
// Browser Panel
// ============================================================================

void CircuitNetApplet::draw_browser_panel() {
    ImGui::Text("Designs (%d)", s_->db.design_count());
    ImGui::Separator();

    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##filter", "Filter...", s_->filter_buf, sizeof(s_->filter_buf));

    std::string filter(s_->filter_buf);

    ImGui::BeginChild("##design_list", {0, 0});
    for (int i = 0; i < (int)s_->designs.size(); i++) {
        auto& d = s_->designs[i];
        if (!filter.empty() && d.name.find(filter) == std::string::npos) continue;

        bool selected = (i == s_->selected_design);
        if (ImGui::Selectable(d.name.c_str(), selected)) {
            select_design(i);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%.2f W  |  %d gates", d.total_power, d.num_gates);
            ImGui::EndTooltip();
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// Design Info (main view header)
// ============================================================================

void CircuitNetApplet::draw_design_info() {
    if (s_->selected_design < 0 || s_->selected_design >= (int)s_->designs.size()) return;

    auto& d = s_->designs[s_->selected_design];

    if (ImGui::CollapsingHeader("Design Info", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Columns(3, "##info_cols", false);
        ImGui::Text("Module: %s", s_->current_graph.module_name.c_str());
        ImGui::Text("Power: %.4f W", d.total_power);
        ImGui::NextColumn();
        ImGui::Text("Gates: %d", (int)s_->current_graph.gates.size());
        ImGui::Text("Edges: %d", (int)s_->current_graph.edges.size());
        ImGui::NextColumn();
        ImGui::Text("Inputs: %d", s_->current_graph.num_inputs);
        ImGui::Text("Outputs: %d", s_->current_graph.num_outputs);
        ImGui::Columns(1);
    }

    if (s_->selected_gate >= 0 && s_->selected_gate < (int)s_->current_graph.gates.size()) {
        auto& g = s_->current_graph.gates[s_->selected_gate];
        if (ImGui::CollapsingHeader("Selected Gate", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Columns(3, "##gate_cols", false);
            ImGui::Text("%s (%s X%d)", g.inst_name.c_str(), g.cell_type.c_str(), g.drive_strength);
            ImGui::Text("Delay: %.4f", g.delay);
            ImGui::NextColumn();
            ImGui::Text("Fanout: %d  Load: %.4f", g.fanout_number, g.fanout_load);
            ImGui::Text("Fanout Res: %.4f", g.fanout_resistance);
            ImGui::NextColumn();
            ImGui::Text("In Slew: %.4f  Out Slew: %.4f", g.input_slew, g.output_slew);
            ImGui::Text("Out: %s  Ins: %d", g.output_net.c_str(), (int)g.input_nets.size());
            ImGui::Columns(1);
        }
    }
}

// ============================================================================
// Circuit Graph Visualization
// ============================================================================

void CircuitNetApplet::draw_circuit_graph() {
    if (!s_->current_graph.valid) {
        ImGui::TextWrapped("Select a design from the sidebar to visualize its circuit graph.");
        return;
    }

    if (!s_->current_layout.valid) {
        ImGui::TextWrapped("No layout available for this circuit.");
        return;
    }

    if (!s_->node_editor_ctx) return;

    ned::SetCurrentEditor(s_->node_editor_ctx);
    ned::Begin("CircuitGraph", ImGui::GetContentRegionAvail());

    // Frame 0: set node positions; Frame 1: fit view after nodes have been measured
    if (s_->layout_frames == 0) {
        for (int i = 0; i < (int)s_->current_graph.gates.size(); i++) {
            auto& pos = s_->current_layout.positions[i];
            ned::SetNodePosition(ned::NodeId(i + 1), ImVec2(pos.x, pos.y));
        }
        s_->layout_frames = 1;
    } else if (s_->layout_frames == 1) {
        ned::NavigateToContent();
        s_->layout_frames = -1;
    }

    // Find max values for heatmap normalization
    float max_delay = 0, max_fanout = 0, max_slew = 0;
    for (auto& g : s_->current_graph.gates) {
        max_delay = std::max(max_delay, g.delay);
        max_fanout = std::max(max_fanout, (float)g.fanout_number);
        max_slew = std::max(max_slew, g.output_slew);
    }

    // Draw nodes
    for (int i = 0; i < (int)s_->current_graph.gates.size(); i++) {
        auto& gate = s_->current_graph.gates[i];

        ImU32 color;
        switch (s_->color_mode) {
            case State::ByType:   color = cell_type_color(gate.cell_type); break;
            case State::ByDelay:  color = power_heatmap_color(max_delay > 0 ? gate.delay / max_delay : 0); break;
            case State::ByFanout: color = power_heatmap_color(max_fanout > 0 ? (float)gate.fanout_number / max_fanout : 0); break;
            case State::BySlew:   color = power_heatmap_color(max_slew > 0 ? gate.output_slew / max_slew : 0); break;
        }
        ImVec4 col4 = ImGui::ColorConvertU32ToFloat4(color);
        ned::PushStyleColor(ned::StyleColor_NodeBg, col4);

        ned::BeginNode(ned::NodeId(i + 1));

        ImGui::Text("%s", gate.cell_type.c_str());

        ImGui::BeginGroup();
        for (int j = 0; j < (int)gate.input_nets.size(); j++) {
            ned::BeginPin(ned::PinId((uintptr_t)(i + 1) * 100000 + j + 1), ned::PinKind::Input);
            ImGui::Bullet();
            ned::EndPin();
        }
        ImGui::EndGroup();

        ImGui::SameLine(0, 20);

        ImGui::BeginGroup();
        if (!gate.output_net.empty()) {
            ned::BeginPin(ned::PinId((uintptr_t)(i + 1) * 100000), ned::PinKind::Output);
            ImGui::Bullet();
            ned::EndPin();
        }
        ImGui::EndGroup();

        ned::EndNode();
        ned::PopStyleColor();
    }

    // Draw links
    for (int i = 0; i < (int)s_->current_graph.edges.size(); i++) {
        auto& e = s_->current_graph.edges[i];

        int input_pin_idx = 0;
        if (e.to_gate >= 0 && e.to_gate < (int)s_->current_graph.gates.size()) {
            auto& sink_gate = s_->current_graph.gates[e.to_gate];
            for (int j = 0; j < (int)sink_gate.input_nets.size(); j++) {
                if (sink_gate.input_nets[j] == e.net_name) {
                    input_pin_idx = j;
                    break;
                }
            }
        }

        ned::PinId from_pin((uintptr_t)(e.from_gate + 1) * 100000);
        ned::PinId to_pin((uintptr_t)(e.to_gate + 1) * 100000 + input_pin_idx + 1);

        ned::Link(ned::LinkId(i + 1), from_pin, to_pin, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    }

    // Sync selection back to our state
    if (ned::HasSelectionChanged()) {
        ned::NodeId sel[1];
        int count = ned::GetSelectedNodes(sel, 1);
        s_->selected_gate = count > 0 ? (int)sel[0].Get() - 1 : -1;
    }

    ned::End();
    ned::SetCurrentEditor(nullptr);
}

// ============================================================================
// Module View (RTL block diagram)
// ============================================================================

void CircuitNetApplet::draw_module_view() {
    if (!s_->rtl_module.valid) {
        ImGui::TextWrapped("No RTL source found for this design. Select a design with an RTL file.");
        return;
    }

    if (!s_->module_editor_ctx) return;

    auto& mod = s_->rtl_module;

    ImGui::Text("Module: %s  |  %d ports  |  %d blocks",
                mod.name.c_str(), (int)mod.ports.size(), (int)mod.blocks.size());
    ImGui::Separator();

    ned::SetCurrentEditor(s_->module_editor_ctx);
    ned::Begin("ModuleView", ImGui::GetContentRegionAvail());

    // Node ID scheme:
    //   1..num_inputs          = input port nodes
    //   num_inputs+1..num_io   = output port nodes
    //   num_io+1..num_io+blocks = block nodes
    // Pin ID scheme: node_id * 10000 + pin_index  (output pins use +0, input pins use +1..N)

    std::vector<VerilogPort> inputs, outputs;
    for (auto& p : mod.ports) {
        if (p.direction == "input") inputs.push_back(p);
        else if (p.direction == "output") outputs.push_back(p);
    }

    int num_in = (int)inputs.size();
    int num_out = (int)outputs.size();
    int num_blocks = (int)mod.blocks.size();
    int id_block_base = num_in + num_out;

    // Map signal name -> list of output pin IDs that produce it
    // Map signal name -> list of input pin IDs that consume it
    struct PinRef { int node_id; int pin_idx; };
    std::unordered_map<std::string, std::vector<PinRef>> signal_sources;
    std::unordered_map<std::string, std::vector<PinRef>> signal_sinks;

    // Input ports are sources
    for (int i = 0; i < num_in; i++) {
        int nid = i + 1;
        signal_sources[inputs[i].name].push_back({nid, 0});
    }

    // Output ports are sinks
    for (int i = 0; i < num_out; i++) {
        int nid = num_in + i + 1;
        signal_sinks[outputs[i].name].push_back({nid, 1});
    }

    // Blocks: writes are sources, reads are sinks
    for (int i = 0; i < num_blocks; i++) {
        int nid = id_block_base + i + 1;
        auto& blk = mod.blocks[i];
        for (int j = 0; j < (int)blk.writes.size(); j++) {
            signal_sources[blk.writes[j]].push_back({nid, 0});
        }
        for (int j = 0; j < (int)blk.reads.size(); j++) {
            signal_sinks[blk.reads[j]].push_back({nid, (int)blk.writes.size() + j + 1});
        }
    }

    // Layout: 3 columns — inputs | blocks | outputs
    if (s_->module_layout_frames == 0) {
        float col_x[] = {0, 400, 800};
        float y_gap = 80;

        for (int i = 0; i < num_in; i++) {
            ned::SetNodePosition(ned::NodeId(i + 1),
                ImVec2(col_x[0], i * y_gap));
        }
        for (int i = 0; i < num_out; i++) {
            ned::SetNodePosition(ned::NodeId(num_in + i + 1),
                ImVec2(col_x[2], i * y_gap));
        }
        for (int i = 0; i < num_blocks; i++) {
            ned::SetNodePosition(ned::NodeId(id_block_base + i + 1),
                ImVec2(col_x[1], i * (y_gap + 40)));
        }
        s_->module_layout_frames = 1;
    } else if (s_->module_layout_frames == 1) {
        ned::NavigateToContent();
        s_->module_layout_frames = -1;
    }

    // Draw input port nodes (green)
    for (int i = 0; i < num_in; i++) {
        int nid = i + 1;
        ned::PushStyleColor(ned::StyleColor_NodeBg, ImVec4(0.2f, 0.5f, 0.2f, 0.9f));
        ned::PushStyleVar(ned::StyleVar_NodeRounding, 12.0f);
        ned::BeginNode(ned::NodeId(nid));

        ImGui::Text("IN");
        ImGui::Text("%s%s", inputs[i].name.c_str(),
                    inputs[i].width.empty() ? "" : (" " + inputs[i].width).c_str());

        ned::BeginPin(ned::PinId((uintptr_t)nid * 10000), ned::PinKind::Output);
        ImGui::Bullet();
        ned::EndPin();

        ned::EndNode();
        ned::PopStyleVar();
        ned::PopStyleColor();
    }

    // Draw output port nodes (orange)
    for (int i = 0; i < num_out; i++) {
        int nid = num_in + i + 1;
        ned::PushStyleColor(ned::StyleColor_NodeBg, ImVec4(0.6f, 0.3f, 0.1f, 0.9f));
        ned::PushStyleVar(ned::StyleVar_NodeRounding, 12.0f);
        ned::BeginNode(ned::NodeId(nid));

        ned::BeginPin(ned::PinId((uintptr_t)nid * 10000 + 1), ned::PinKind::Input);
        ImGui::Bullet();
        ned::EndPin();

        ImGui::Text("OUT");
        ImGui::Text("%s%s", outputs[i].name.c_str(),
                    outputs[i].width.empty() ? "" : (" " + outputs[i].width).c_str());

        ned::EndNode();
        ned::PopStyleVar();
        ned::PopStyleColor();
    }

    // Draw block nodes
    for (int i = 0; i < num_blocks; i++) {
        int nid = id_block_base + i + 1;
        auto& blk = mod.blocks[i];

        ImVec4 bg;
        const char* type_label;
        switch (blk.type) {
            case VerilogBlock::AlwaysFF:
                bg = ImVec4(0.4f, 0.2f, 0.6f, 0.9f);
                type_label = "FF";
                break;
            case VerilogBlock::AlwaysComb:
                bg = ImVec4(0.2f, 0.4f, 0.6f, 0.9f);
                type_label = "COMB";
                break;
            case VerilogBlock::Assign:
                bg = ImVec4(0.3f, 0.5f, 0.3f, 0.9f);
                type_label = "=";
                break;
            case VerilogBlock::Instance:
                bg = ImVec4(0.5f, 0.4f, 0.2f, 0.9f);
                type_label = blk.module_type.c_str();
                break;
        }

        ned::PushStyleColor(ned::StyleColor_NodeBg, bg);
        ned::PushStyleVar(ned::StyleVar_NodeRounding, 4.0f);
        ned::BeginNode(ned::NodeId(nid));

        ImGui::Text("[%s]", type_label);
        if (blk.type == VerilogBlock::Instance) {
            ImGui::Text("%s", blk.inst_name.c_str());
        } else if (!blk.writes.empty()) {
            std::string w;
            for (auto& s : blk.writes) { if (!w.empty()) w += ", "; w += s; }
            ImGui::TextWrapped("%s", w.c_str());
        }

        // Input pins (reads)
        ImGui::BeginGroup();
        for (int j = 0; j < (int)blk.reads.size(); j++) {
            uintptr_t pin_id = (uintptr_t)nid * 10000 + (int)blk.writes.size() + j + 1;
            ned::BeginPin(ned::PinId(pin_id), ned::PinKind::Input);
            ImGui::Text("> %s", blk.reads[j].c_str());
            ned::EndPin();
        }
        ImGui::EndGroup();

        if (!blk.reads.empty() && !blk.writes.empty())
            ImGui::SameLine(0, 30);

        // Output pins (writes)
        ImGui::BeginGroup();
        for (int j = 0; j < (int)blk.writes.size(); j++) {
            uintptr_t pin_id = (uintptr_t)nid * 10000;
            ned::BeginPin(ned::PinId(pin_id), ned::PinKind::Output);
            ImGui::Text("%s >", blk.writes[j].c_str());
            ned::EndPin();
        }
        ImGui::EndGroup();

        ned::EndNode();
        ned::PopStyleVar();
        ned::PopStyleColor();
    }

    // Draw links: for each signal, connect every source to every sink
    int link_id = 1;
    for (auto& [sig, sources] : signal_sources) {
        auto sit = signal_sinks.find(sig);
        if (sit == signal_sinks.end()) continue;
        for (auto& src : sources) {
            for (auto& dst : sit->second) {
                ned::PinId from_pin((uintptr_t)src.node_id * 10000 + src.pin_idx);
                ned::PinId to_pin((uintptr_t)dst.node_id * 10000 + dst.pin_idx);
                ned::Link(ned::LinkId(link_id++), from_pin, to_pin,
                          ImVec4(0.6f, 0.6f, 0.6f, 0.8f));
            }
        }
    }

    ned::End();
    ned::SetCurrentEditor(nullptr);
}

// ============================================================================
// Netlist Viewer
// ============================================================================

void CircuitNetApplet::draw_netlist_viewer() {
    if (s_->current_netlist_source.empty()) {
        ImGui::TextWrapped("Select a design to view its Verilog netlist.");
        return;
    }

    ImGui::Text("Verilog Netlist (%d bytes)", (int)s_->current_netlist_source.size());
    ImGui::Separator();

    ImGui::BeginChild("##netlist_scroll", {0, 0}, false, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::TextUnformatted(s_->current_netlist_source.c_str(),
                           s_->current_netlist_source.c_str() + std::min((size_t)100000, s_->current_netlist_source.size()));
    ImGui::EndChild();
}

// ============================================================================
// Statistics
// ============================================================================

void CircuitNetApplet::draw_statistics() {
    ImGui::Text("Dataset Statistics (%d designs loaded)", s_->db.design_count());
    ImGui::Separator();

    // Power distribution histogram
    if (!s_->power_values.empty() && ImPlot::BeginPlot("Power Distribution", {-1, 250})) {
        ImPlot::SetupAxes("Total Power", "Count");
        ImPlot::PlotHistogram("Power", s_->power_values.data(), (int)s_->power_values.size(), 50);
        ImPlot::EndPlot();
    }

    // Gate count vs power scatter
    {
        auto res = s_->db.query("SELECT num_gates, total_power FROM designs WHERE total_power > 0 AND num_gates > 0 LIMIT 2000");
        if (res.ok && !res.rows.empty()) {
            std::vector<double> gates_v, power_v;
            for (auto& row : res.rows) {
                try {
                    gates_v.push_back(std::stod(row[0]));
                    power_v.push_back(std::stod(row[1]));
                } catch (...) {}
            }

            if (!gates_v.empty() && ImPlot::BeginPlot("Gates vs Power", {-1, 250})) {
                ImPlot::SetupAxes("Gate Count", "Total Power");
                ImPlot::PlotScatter("Designs", gates_v.data(), power_v.data(), (int)gates_v.size());
                ImPlot::EndPlot();
            }
        }
    }

    // Top power consumers
    {
        auto res = s_->db.query("SELECT name, total_power FROM designs ORDER BY total_power DESC LIMIT 15");
        if (res.ok && !res.rows.empty()) {
            ImGui::Spacing();
            ImGui::Text("Top Power Consumers:");
            if (ImGui::BeginTable("##top_power", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Design");
                ImGui::TableSetupColumn("Power");
                ImGui::TableHeadersRow();
                for (auto& row : res.rows) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(row[0].c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextUnformatted(row[1].c_str());
                }
                ImGui::EndTable();
            }
        }
    }
}

