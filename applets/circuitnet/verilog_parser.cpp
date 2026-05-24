#include "verilog_parser.h"

#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>
#include <unordered_set>

static int extract_drive_strength(const std::string& cell_type) {
    auto pos = cell_type.find('X');
    if (pos == std::string::npos || pos + 1 >= cell_type.size()) return 1;
    std::string num_str = cell_type.substr(pos + 1);
    try { return std::stoi(num_str); } catch (...) { return 1; }
}

static const std::unordered_set<std::string> OUTPUT_PINS = {"Y", "Q", "QN", "S", "CO", "SN", "Z"};

// Verilog keywords that are NOT cell instantiations
static const std::unordered_set<std::string> VERILOG_KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg",
    "assign", "parameter", "localparam", "generate", "endgenerate",
    "always", "initial", "begin", "end", "if", "else", "for", "while",
    "case", "endcase", "default", "posedge", "negedge", "supply0", "supply1"
};

CircuitGraph parse_verilog_netlist(const std::string& verilog_path) {
    CircuitGraph graph;

    std::ifstream file(verilog_path);
    if (!file.is_open()) return graph;

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    // Extract module name: "module NAME (" or "module NAME\n("
    std::regex module_re(R"(module\s+(\w+)\s*[\(;])");
    std::smatch module_match;
    if (std::regex_search(content, module_match, module_re)) {
        graph.module_name = module_match[1].str();
    }

    // Count input/output ports
    {
        std::regex input_re(R"(input\s+(?:\[\d+:\d+\]\s*)?(\w+))");
        auto it = std::sregex_iterator(content.begin(), content.end(), input_re);
        for (; it != std::sregex_iterator(); ++it) graph.num_inputs++;
    }
    {
        std::regex output_re(R"(output\s+(?:\[\d+:\d+\]\s*)?(\w+))");
        auto it = std::sregex_iterator(content.begin(), content.end(), output_re);
        for (; it != std::sregex_iterator(); ++it) graph.num_outputs++;
    }

    // Extract gate instances: CELL_TYPE INST_NAME ( .PORT(NET), ... );
    // Instances can span multiple lines. Find each "WORD WORD (" ... ");" block.
    std::regex inst_re(R"((\w+)\s+(\w+)\s*\(([^;]*)\)\s*;)");
    auto it = std::sregex_iterator(content.begin(), content.end(), inst_re);

    for (; it != std::sregex_iterator(); ++it) {
        std::string cell_type = (*it)[1].str();
        std::string inst_name = (*it)[2].str();
        std::string ports_str = (*it)[3].str();

        if (VERILOG_KEYWORDS.count(cell_type)) continue;

        // Must have named port connections (.PORT(NET))
        if (ports_str.find('.') == std::string::npos) continue;

        Gate g;
        g.id = (int)graph.gates.size();
        g.cell_type = cell_type;
        g.inst_name = inst_name;
        g.drive_strength = extract_drive_strength(cell_type);

        // Parse port connections: .PORT_NAME(NET_NAME)
        std::regex port_re(R"(\.(\w+)\s*\(\s*(\w+(?:\[\d+\])?)\s*\))");
        auto pit = std::sregex_iterator(ports_str.begin(), ports_str.end(), port_re);
        for (; pit != std::sregex_iterator(); ++pit) {
            std::string port_name = (*pit)[1].str();
            std::string net_name = (*pit)[2].str();

            if (OUTPUT_PINS.count(port_name)) {
                g.output_net = net_name;
            } else {
                g.input_nets.push_back(net_name);
            }
        }

        if (!g.output_net.empty()) {
            graph.net_to_drivers[g.output_net].push_back(g.id);
        }
        for (auto& net : g.input_nets) {
            graph.net_to_sinks[net].push_back(g.id);
        }

        graph.gates.push_back(std::move(g));
    }

    // Build edges: for each net, connect driver gates to sink gates
    for (auto& [net, drivers] : graph.net_to_drivers) {
        auto sink_it = graph.net_to_sinks.find(net);
        if (sink_it == graph.net_to_sinks.end()) continue;

        for (int drv : drivers) {
            for (int snk : sink_it->second) {
                if (drv != snk) {
                    graph.edges.push_back({drv, snk, net});
                }
            }
        }
    }

    graph.valid = !graph.gates.empty();
    return graph;
}

// ============================================================================
// Feature annotation (manual JSON parsing — no external dependency)
// ============================================================================

static std::string extract_field(const std::string& obj_str, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = obj_str.find(needle);
    if (pos == std::string::npos) return "";

    pos = obj_str.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;

    while (pos < obj_str.size() && (obj_str[pos] == ' ' || obj_str[pos] == '\t')) pos++;
    if (pos >= obj_str.size()) return "";

    if (obj_str[pos] == '"') {
        pos++;
        auto end = obj_str.find('"', pos);
        if (end == std::string::npos) return "";
        return obj_str.substr(pos, end - pos);
    }

    size_t start = pos;
    while (pos < obj_str.size() && obj_str[pos] != ',' && obj_str[pos] != '}' &&
           obj_str[pos] != ' ' && obj_str[pos] != '\n') pos++;
    return obj_str.substr(start, pos - start);
}

static float first_float(const std::string& s) {
    if (s.empty()) return 0;
    std::istringstream iss(s);
    float v = 0;
    iss >> v;
    return v;
}

static float avg_floats(const std::string& s) {
    if (s.empty()) return 0;
    std::istringstream iss(s);
    float sum = 0; int count = 0;
    float v;
    while (iss >> v) { sum += v; count++; }
    return count > 0 ? sum / count : 0;
}

void annotate_features(CircuitGraph& graph, const std::string& feature_json_path) {
    std::ifstream file(feature_json_path);
    if (!file.is_open()) return;

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    std::unordered_map<std::string, int> name_to_idx;
    for (auto& g : graph.gates) {
        name_to_idx[g.inst_name] = g.id;
    }

    size_t pos = 0;
    while (pos < content.size()) {
        auto obj_start = content.find('{', pos);
        if (obj_start == std::string::npos) break;

        int depth = 0;
        size_t obj_end = obj_start;
        for (size_t i = obj_start; i < content.size(); i++) {
            if (content[i] == '{') depth++;
            else if (content[i] == '}') {
                depth--;
                if (depth == 0) { obj_end = i; break; }
            }
        }

        std::string obj_str = content.substr(obj_start, obj_end - obj_start + 1);
        pos = obj_end + 1;

        std::string inst_name = extract_field(obj_str, "InstName");
        if (inst_name.empty()) continue;

        auto it = name_to_idx.find(inst_name);
        if (it == name_to_idx.end()) continue;

        Gate& gate = graph.gates[it->second];

        gate.fanout_load = first_float(extract_field(obj_str, "fanoutLoad (rise fall)"));
        gate.fanout_resistance = first_float(extract_field(obj_str, "fanoutRes"));

        std::string fn = extract_field(obj_str, "fanoutNum");
        if (!fn.empty()) { try { gate.fanout_number = std::stoi(fn); } catch (...) {} }

        // Input slew
        {
            auto slew_pos = obj_str.find("Input slew");
            if (slew_pos != std::string::npos) {
                auto colon = obj_str.find(':', slew_pos);
                if (colon != std::string::npos) {
                    auto quote = obj_str.find('"', colon + 1);
                    if (quote != std::string::npos) {
                        auto end_quote = obj_str.find('"', quote + 1);
                        if (end_quote != std::string::npos) {
                            gate.input_slew = avg_floats(obj_str.substr(quote + 1, end_quote - quote - 1));
                        }
                    }
                }
            }
        }
        // Output slew
        {
            auto slew_pos = obj_str.find("Output slew");
            if (slew_pos != std::string::npos) {
                auto colon = obj_str.find(':', slew_pos);
                if (colon != std::string::npos) {
                    auto quote = obj_str.find('"', colon + 1);
                    if (quote != std::string::npos) {
                        auto end_quote = obj_str.find('"', quote + 1);
                        if (end_quote != std::string::npos) {
                            gate.output_slew = avg_floats(obj_str.substr(quote + 1, end_quote - quote - 1));
                        }
                    }
                }
            }
        }
        // Delay
        {
            auto delay_pos = obj_str.find("Delay");
            if (delay_pos != std::string::npos) {
                auto colon = obj_str.find(':', delay_pos);
                if (colon != std::string::npos) {
                    auto quote = obj_str.find('"', colon + 1);
                    if (quote != std::string::npos) {
                        auto end_quote = obj_str.find('"', quote + 1);
                        if (end_quote != std::string::npos) {
                            gate.delay = avg_floats(obj_str.substr(quote + 1, end_quote - quote - 1));
                        }
                    }
                }
            }
        }
    }
}

float read_power(const std::string& power_path) {
    std::ifstream file(power_path);
    if (!file.is_open()) return 0;

    std::string line;
    while (std::getline(file, line)) {
        auto pos = line.find("Total Power:");
        if (pos == std::string::npos) pos = line.find("Total Power :");
        if (pos != std::string::npos) {
            std::string after = line.substr(pos);
            std::regex num_re(R"([\d]+\.[\d]+|[\d]+)");
            std::smatch m;
            if (std::regex_search(after, m, num_re)) {
                try { return std::stof(m[0].str()); } catch (...) {}
            }
        }
    }
    return 0;
}
