#pragma once

#include <string>
#include <vector>
#include <unordered_map>

struct Gate {
    int id = -1;
    std::string inst_name;
    std::string cell_type;
    int drive_strength = 1;
    std::vector<std::string> input_nets;
    std::string output_net;

    float fanout_load = 0;
    float fanout_resistance = 0;
    int fanout_number = 0;
    float input_slew = 0;
    float output_slew = 0;
    float delay = 0;
};

struct CircuitEdge {
    int from_gate;
    int to_gate;
    std::string net_name;
};

struct CircuitGraph {
    std::vector<Gate> gates;
    std::vector<CircuitEdge> edges;
    std::unordered_map<std::string, std::vector<int>> net_to_drivers;
    std::unordered_map<std::string, std::vector<int>> net_to_sinks;
    std::string module_name;
    int num_inputs = 0;
    int num_outputs = 0;
    float total_power = 0;
    bool valid = false;
};

CircuitGraph parse_verilog_netlist(const std::string& verilog_path);
void annotate_features(CircuitGraph& graph, const std::string& feature_json_path);
float read_power(const std::string& power_path);

// RTL module structure (behavioral Verilog)
struct VerilogPort {
    std::string name;
    std::string direction; // "input", "output", "inout"
    std::string width;     // "[31:0]" or ""
    bool is_reg = false;
};

struct VerilogBlock {
    enum Type { AlwaysFF, AlwaysComb, Assign, Instance };
    Type type;
    std::string label;
    std::string body;
    std::vector<std::string> reads;
    std::vector<std::string> writes;
    // For Instance type
    std::string module_type;
    std::string inst_name;
};

struct VerilogModule {
    std::string name;
    std::vector<VerilogPort> ports;
    std::vector<std::string> wires;
    std::vector<std::string> regs;
    std::vector<std::string> params;
    std::vector<VerilogBlock> blocks;
    std::string source;
    bool valid = false;
};

VerilogModule parse_rtl_module(const std::string& verilog_path);
std::string find_rtl_file(const std::string& design_path);
