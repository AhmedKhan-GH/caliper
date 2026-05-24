#pragma once

#include <memory>
#include <string>

class CircuitNetApplet {
public:
    CircuitNetApplet();
    ~CircuitNetApplet();

    bool initialize();
    void draw_ui(int win_w, int win_h);
    void cleanup();

private:
    void draw_browser_panel();
    void draw_design_info();
    void draw_netlist_viewer();
    void draw_circuit_graph();
    void draw_statistics();
    void draw_sql_console();

    void open_dataset(const std::string& dir);
    void select_design(int idx);
    void parse_current_netlist();

    struct State;
    std::unique_ptr<State> s_;
};
