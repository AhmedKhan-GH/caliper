#include "verilog_parser.h"

#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>
#include <unordered_set>

#include <slang/ast/Compilation.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/PortSymbols.h>
#include <slang/ast/symbols/VariableSymbols.h>
#include <slang/ast/symbols/MemberSymbols.h>
#include <slang/ast/expressions/MiscExpressions.h>
#include <slang/ast/ASTVisitor.h>
#include <slang/syntax/SyntaxTree.h>

using namespace slang;
using namespace slang::ast;
using namespace slang::syntax;

static int extract_drive_strength(const std::string& cell_type) {
    auto pos = cell_type.find('X');
    if (pos == std::string::npos || pos + 1 >= cell_type.size()) return 1;
    std::string num_str = cell_type.substr(pos + 1);
    try { return std::stoi(num_str); } catch (...) { return 1; }
}

static const std::unordered_set<std::string> OUTPUT_PINS = {"Y", "Q", "QN", "S", "CO", "SN", "Z"};

CircuitGraph parse_verilog_netlist(const std::string& verilog_path) {
    CircuitGraph graph;

    auto result = SyntaxTree::fromFile(std::string_view(verilog_path));
    if (!result) return graph;

    auto tree = std::move(*result);

    Compilation compilation;
    compilation.addSyntaxTree(tree);

    auto& root = compilation.getRoot();
    if (root.topInstances.empty()) return graph;

    // Take the first top-level instance
    auto* top = root.topInstances[0];
    graph.module_name = std::string(top->name);

    // Count ports
    for (auto& member : top->body.members()) {
        if (member.kind == SymbolKind::Port) {
            auto& port = member.as<PortSymbol>();
            if (port.direction == ArgumentDirection::In)
                graph.num_inputs++;
            else if (port.direction == ArgumentDirection::Out)
                graph.num_outputs++;
        }
    }

    // Extract gate instances
    for (auto& member : top->body.members()) {
        if (member.kind != SymbolKind::Instance) continue;

        auto& inst = member.as<InstanceSymbol>();
        std::string cell_type = std::string(inst.getDefinition().name);
        std::string inst_name = std::string(inst.name);

        Gate g;
        g.id = (int)graph.gates.size();
        g.cell_type = cell_type;
        g.inst_name = inst_name;
        g.drive_strength = extract_drive_strength(cell_type);

        // Extract port connections
        auto connections = inst.getPortConnections();
        for (auto* conn : connections) {
            if (!conn) continue;

            std::string port_name = std::string(conn->port.name);
            const Expression* expr = conn->getExpression();
            if (!expr) continue;

            // Get the connected net name from the expression
            std::string net_name;
            if (expr->kind == ExpressionKind::NamedValue ||
                expr->kind == ExpressionKind::HierarchicalValue) {
                auto& sym = expr->as<ValueExpressionBase>().symbol;
                net_name = std::string(sym.name);
            }

            if (net_name.empty()) continue;

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
