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

// Module-level abstracted graph
struct ModuleNode {
    std::string family;
    int instance_count = 0;
    ImU32 color = IM_COL32(180, 180, 180, 200);
};

struct ModuleEdge {
    int from_node;
    int to_node;
    int connection_count = 0;
};

struct ModuleGraph {
    std::vector<ModuleNode> nodes;
    std::vector<ModuleEdge> edges;
    std::unordered_map<std::string, int> family_to_idx;
    bool valid = false;
};

static std::string gate_family(const std::string& cell_type) {
    if (cell_type.find("NAND") != std::string::npos) return "NAND";
    if (cell_type.find("NOR") != std::string::npos)  return "NOR";
    if (cell_type.find("XOR") != std::string::npos)  return "XOR";
    if (cell_type.find("XNOR") != std::string::npos) return "XNOR";
    if (cell_type.find("AND") != std::string::npos)  return "AND";
    if (cell_type.find("OR") != std::string::npos)   return "OR";
    if (cell_type.find("INV") != std::string::npos)  return "INV";
    if (cell_type.find("BUF") != std::string::npos)  return "BUF";
    if (cell_type.find("DFF") != std::string::npos)  return "FF";
    if (cell_type.find("LATCH") != std::string::npos) return "LATCH";
    if (cell_type.find("MX") != std::string::npos)   return "MUX";
    if (cell_type.find("AO") != std::string::npos)   return "AO";
    if (cell_type.find("OA") != std::string::npos)   return "OA";
    if (cell_type.find("HA") != std::string::npos)   return "HA";
    if (cell_type.find("FA") != std::string::npos)   return "FA";
    return "OTHER";
}

struct CircuitNetApplet::State {
    CircuitDB db;
    std::vector<DesignEntry> designs;
    int selected_design = -1;
    int page = 0;
    int page_size = 200;

    // Current design data
    CircuitGraph current_graph;
    GraphLayout current_layout;
    std::string current_netlist_source;

    // Graph view state
    int selected_gate = -1;
    enum ColorMode { ByType, ByDelay, ByFanout, BySlew } color_mode = ByType;

    // Module view
    ModuleGraph module_graph;
    ned::EditorContext* module_editor_ctx = nullptr;
    int module_layout_frames = -1;

    // SQL console
    char sql_buf[4096] = "SELECT name, num_gates, total_power FROM designs ORDER BY total_power DESC LIMIT 20";
    QueryResult last_query;

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
        if (ImGui::BeginTabItem("Statistics")) {
            draw_statistics();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("SQL Console")) {
            draw_sql_console();
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

        s_->designs = s_->db.get_designs(s_->page_size, 0);
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

    // Build module-level abstracted graph
    {
        ModuleGraph& mg = s_->module_graph;
        mg = {};
        auto& graph = s_->current_graph;

        // Create family nodes
        auto get_or_create = [&](const std::string& fam) -> int {
            auto it = mg.family_to_idx.find(fam);
            if (it != mg.family_to_idx.end()) return it->second;
            int idx = (int)mg.nodes.size();
            ModuleNode mn;
            mn.family = fam;
            mn.color = cell_type_color(fam);
            mg.nodes.push_back(mn);
            mg.family_to_idx[fam] = idx;
            return idx;
        };

        // Add I/O port pseudo-nodes
        int input_idx = get_or_create("INPUTS");
        mg.nodes[input_idx].instance_count = graph.num_inputs;
        mg.nodes[input_idx].color = IM_COL32(100, 200, 255, 220);
        int output_idx = get_or_create("OUTPUTS");
        mg.nodes[output_idx].instance_count = graph.num_outputs;
        mg.nodes[output_idx].color = IM_COL32(255, 150, 100, 220);

        // Map each gate to its family
        std::vector<int> gate_family_idx(graph.gates.size());
        for (int i = 0; i < (int)graph.gates.size(); i++) {
            std::string fam = gate_family(graph.gates[i].cell_type);
            int fi = get_or_create(fam);
            gate_family_idx[i] = fi;
            mg.nodes[fi].instance_count++;
        }

        // Find primary I/O nets (driven externally or consumed externally)
        std::unordered_set<std::string> driven_nets, consumed_nets;
        for (auto& [net, _] : graph.net_to_drivers) driven_nets.insert(net);
        for (auto& [net, _] : graph.net_to_sinks) consumed_nets.insert(net);

        // Nets consumed but not driven by any gate = primary inputs
        for (auto& [net, sinks] : graph.net_to_sinks) {
            if (driven_nets.find(net) == driven_nets.end()) {
                for (int sink : sinks) {
                    int to_fam = gate_family_idx[sink];
                    // Edge from INPUTS to this family
                    bool found = false;
                    for (auto& e : mg.edges) {
                        if (e.from_node == input_idx && e.to_node == to_fam) {
                            e.connection_count++;
                            found = true;
                            break;
                        }
                    }
                    if (!found) mg.edges.push_back({input_idx, to_fam, 1});
                }
            }
        }

        // Nets driven but not consumed by any gate = primary outputs
        for (auto& [net, drivers] : graph.net_to_drivers) {
            if (consumed_nets.find(net) == consumed_nets.end()) {
                for (int drv : drivers) {
                    int from_fam = gate_family_idx[drv];
                    bool found = false;
                    for (auto& e : mg.edges) {
                        if (e.from_node == from_fam && e.to_node == output_idx) {
                            e.connection_count++;
                            found = true;
                            break;
                        }
                    }
                    if (!found) mg.edges.push_back({from_fam, output_idx, 1});
                }
            }
        }

        // Internal edges between families
        for (auto& edge : graph.edges) {
            int from_fam = gate_family_idx[edge.from_gate];
            int to_fam = gate_family_idx[edge.to_gate];
            if (from_fam == to_fam) continue;
            bool found = false;
            for (auto& e : mg.edges) {
                if (e.from_node == from_fam && e.to_node == to_fam) {
                    e.connection_count++;
                    found = true;
                    break;
                }
            }
            if (!found) mg.edges.push_back({from_fam, to_fam, 1});
        }

        mg.valid = !mg.nodes.empty();
        s_->module_layout_frames = 0;
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

    ImGui::BeginChild("##design_list", {0, -ImGui::GetFrameHeightWithSpacing()});
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

    // Pagination
    if (ImGui::Button("<<") && s_->page > 0) {
        s_->page--;
        s_->designs = s_->db.get_designs(s_->page_size, s_->page * s_->page_size);
    }
    ImGui::SameLine();
    ImGui::Text("Page %d", s_->page + 1);
    ImGui::SameLine();
    if (ImGui::Button(">>")) {
        s_->page++;
        s_->designs = s_->db.get_designs(s_->page_size, s_->page * s_->page_size);
    }
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
// Module View (cell-family abstraction)
// ============================================================================

void CircuitNetApplet::draw_module_view() {
    if (!s_->module_graph.valid) {
        ImGui::TextWrapped("Select a design from the sidebar to view its module-level structure.");
        return;
    }

    if (!s_->module_editor_ctx) return;

    auto& mg = s_->module_graph;

    ned::SetCurrentEditor(s_->module_editor_ctx);
    ned::Begin("ModuleView", ImGui::GetContentRegionAvail());

    // Layout: arrange nodes in a circle for now, refine on frame 1
    if (s_->module_layout_frames == 0) {
        int n = (int)mg.nodes.size();
        float cx = 400, cy = 300;
        float radius = 150.0f + n * 20.0f;
        for (int i = 0; i < n; i++) {
            float angle = 2.0f * 3.14159f * i / n - 3.14159f / 2.0f;
            // Put INPUTS on far left, OUTPUTS on far right
            float x, y;
            if (mg.nodes[i].family == "INPUTS") {
                x = 0; y = cy;
            } else if (mg.nodes[i].family == "OUTPUTS") {
                x = cx * 2; y = cy;
            } else {
                x = cx + radius * cosf(angle);
                y = cy + radius * sinf(angle);
            }
            ned::SetNodePosition(ned::NodeId(i + 1), ImVec2(x, y));
        }
        s_->module_layout_frames = 1;
    } else if (s_->module_layout_frames == 1) {
        ned::NavigateToContent();
        s_->module_layout_frames = -1;
    }

    // Draw nodes
    for (int i = 0; i < (int)mg.nodes.size(); i++) {
        auto& mn = mg.nodes[i];
        ImVec4 col4 = ImGui::ColorConvertU32ToFloat4(mn.color);
        ned::PushStyleColor(ned::StyleColor_NodeBg, col4);
        ned::PushStyleVar(ned::StyleVar_NodeRounding, 8.0f);

        ned::BeginNode(ned::NodeId(i + 1));

        ImGui::Text("%s", mn.family.c_str());
        if (mn.family != "INPUTS" && mn.family != "OUTPUTS") {
            ImGui::Text("%d gates", mn.instance_count);
        } else {
            ImGui::Text("%d ports", mn.instance_count);
        }

        // Single input and output pin per family node
        ned::BeginPin(ned::PinId((uintptr_t)(i + 1) * 1000 + 1), ned::PinKind::Input);
        ImGui::Dummy({1, 1});
        ned::EndPin();
        ImGui::SameLine();
        ned::BeginPin(ned::PinId((uintptr_t)(i + 1) * 1000), ned::PinKind::Output);
        ImGui::Dummy({1, 1});
        ned::EndPin();

        ned::EndNode();
        ned::PopStyleVar();
        ned::PopStyleColor();
    }

    // Draw links with thickness based on connection count
    int max_conn = 1;
    for (auto& e : mg.edges) max_conn = std::max(max_conn, e.connection_count);

    for (int i = 0; i < (int)mg.edges.size(); i++) {
        auto& e = mg.edges[i];
        float thickness = 1.0f + 4.0f * (float)e.connection_count / max_conn;
        float alpha = 0.3f + 0.7f * (float)e.connection_count / max_conn;

        ned::PinId from_pin((uintptr_t)(e.from_node + 1) * 1000);
        ned::PinId to_pin((uintptr_t)(e.to_node + 1) * 1000 + 1);

        ned::Link(ned::LinkId(i + 1), from_pin, to_pin,
                  ImVec4(0.7f, 0.7f, 0.7f, alpha), thickness);
    }

    // Tooltip on hovered link
    for (int i = 0; i < (int)mg.edges.size(); i++) {
        if (ned::GetHoveredLink() == ned::LinkId(i + 1)) {
            auto& e = mg.edges[i];
            ned::Suspend();
            ImGui::BeginTooltip();
            ImGui::Text("%s -> %s: %d connections",
                        mg.nodes[e.from_node].family.c_str(),
                        mg.nodes[e.to_node].family.c_str(),
                        e.connection_count);
            ImGui::EndTooltip();
            ned::Resume();
            break;
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

// ============================================================================
// SQL Console
// ============================================================================

void CircuitNetApplet::draw_sql_console() {
    ImGui::Text("DuckDB SQL Console");
    ImGui::Separator();

    ImGui::InputTextMultiline("##sql", s_->sql_buf, sizeof(s_->sql_buf), {-1, 80});

    if (ImGui::Button("Execute") || (ImGui::IsItemFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter, false))) {
        s_->last_query = s_->db.query(s_->sql_buf);
    }

    ImGui::SameLine();
    ImGui::TextDisabled("(Tables: designs, gates)");

    ImGui::Separator();

    if (!s_->last_query.error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, {1, 0.3f, 0.3f, 1});
        ImGui::TextWrapped("Error: %s", s_->last_query.error.c_str());
        ImGui::PopStyleColor();
    }

    if (s_->last_query.ok && !s_->last_query.columns.empty()) {
        ImGui::Text("%d rows", (int)s_->last_query.rows.size());

        int ncols = (int)s_->last_query.columns.size();
        if (ImGui::BeginTable("##results", ncols,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                              {0, ImGui::GetContentRegionAvail().y})) {

            for (auto& col : s_->last_query.columns) {
                ImGui::TableSetupColumn(col.c_str());
            }
            ImGui::TableHeadersRow();

            for (auto& row : s_->last_query.rows) {
                ImGui::TableNextRow();
                for (int c = 0; c < ncols && c < (int)row.size(); c++) {
                    ImGui::TableSetColumnIndex(c);
                    ImGui::TextUnformatted(row[c].c_str());
                }
            }
            ImGui::EndTable();
        }
    }
}
