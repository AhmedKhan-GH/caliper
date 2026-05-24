#include "circuitnet.h"
#include "circuit_db.h"
#include "circuit_viz.h"
#include "verilog_parser.h"

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

namespace fs = std::filesystem;

// ============================================================================

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
    bool layout_applied = false;

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
    s_->node_editor_ctx = ned::CreateEditor(&config);
    return true;
}

void CircuitNetApplet::draw_ui(int win_w, int win_h) {
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({(float)win_w, (float)win_h});
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

    s_->current_layout = compute_layout(s_->current_graph, 70, 30, 40, 20);

    // Cell type distribution
    s_->cell_type_counts.clear();
    for (auto& g : s_->current_graph.gates) {
        s_->cell_type_counts[g.cell_type]++;
    }

    s_->selected_gate = -1;
    s_->layout_applied = false;
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

    // Set initial positions from layout on first frame after design load
    if (!s_->layout_applied) {
        for (int i = 0; i < (int)s_->current_graph.gates.size(); i++) {
            auto& pos = s_->current_layout.positions[i];
            ned::SetNodePosition(ned::NodeId(i + 1), ImVec2(pos.x, pos.y));
        }
        s_->layout_applied = true;
        ned::NavigateToContent();
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

        ImGui::TextUnformatted(gate.cell_type.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("%s", gate.inst_name.c_str());

        ImGui::BeginGroup();
        for (int j = 0; j < (int)gate.input_nets.size(); j++) {
            ned::BeginPin(ned::PinId((uintptr_t)(i + 1) * 100000 + j + 1), ned::PinKind::Input);
            ImGui::Text("-> %s", gate.input_nets[j].c_str());
            ned::EndPin();
        }
        ImGui::EndGroup();

        ImGui::SameLine();

        ImGui::BeginGroup();
        if (!gate.output_net.empty()) {
            ned::BeginPin(ned::PinId((uintptr_t)(i + 1) * 100000), ned::PinKind::Output);
            ImGui::Text("%s ->", gate.output_net.c_str());
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
