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

// ============================================================================
// RTL file discovery
// ============================================================================

std::string find_rtl_file(const std::string& design_path) {
    namespace fs = std::filesystem;
    fs::path dp(design_path);
    std::string dir_name = dp.filename().string();

    // Extract design ID (numeric prefix before first underscore)
    std::string design_id;
    for (char c : dir_name) {
        if (std::isdigit(c)) design_id += c;
        else break;
    }
    if (design_id.empty()) return "";

    // Go up to dataset root, then look in RTL/
    // design_path = .../dataset/Final/100_LeftShift/ or .../dataset_augment/Final/100_LeftShift.e1/
    fs::path final_dir = dp.parent_path();          // .../Final/
    fs::path dataset_dir = final_dir.parent_path();  // .../dataset/
    fs::path rtl_dir = dataset_dir / "RTL";

    if (!fs::is_directory(rtl_dir)) {
        // Try parent's sibling (for augmented, RTL might be in dataset/RTL/)
        fs::path parent_root = dataset_dir.parent_path();
        rtl_dir = parent_root / "dataset" / "RTL";
    }
    if (!fs::is_directory(rtl_dir)) return "";

    // Find file matching {design_id}.*.v
    std::string prefix = design_id + ".";
    for (auto& entry : fs::directory_iterator(rtl_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.substr(0, prefix.size()) == prefix &&
            fname.size() > 2 && fname.substr(fname.size() - 2) == ".v") {
            return entry.path().string();
        }
    }
    return "";
}

// ============================================================================
// RTL module parser
// ============================================================================

static void extract_signal_refs(const std::string& body,
                                const std::unordered_set<std::string>& signals,
                                std::vector<std::string>& reads,
                                std::vector<std::string>& writes) {
    std::unordered_set<std::string> read_set, write_set;

    // Find LHS of <= or = assignments
    std::regex assign_re(R"((\w+)\s*(?:<=|=[^=]))");
    auto it = std::sregex_iterator(body.begin(), body.end(), assign_re);
    for (; it != std::sregex_iterator(); ++it) {
        std::string lhs = (*it)[1].str();
        if (signals.count(lhs)) write_set.insert(lhs);
    }

    // Find all identifiers that are signals (reads = referenced but not only written)
    std::regex id_re(R"(\b(\w+)\b)");
    it = std::sregex_iterator(body.begin(), body.end(), id_re);
    for (; it != std::sregex_iterator(); ++it) {
        std::string id = (*it)[1].str();
        if (signals.count(id) && !write_set.count(id)) {
            read_set.insert(id);
        }
    }

    reads.assign(read_set.begin(), read_set.end());
    writes.assign(write_set.begin(), write_set.end());
    std::sort(reads.begin(), reads.end());
    std::sort(writes.begin(), writes.end());
}

VerilogModule parse_rtl_module(const std::string& verilog_path) {
    VerilogModule mod;
    namespace fs = std::filesystem;

    std::ifstream file(verilog_path);
    if (!file.is_open()) return mod;

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();
    mod.source = content;

    // Extract module name
    std::regex module_re(R"(module\s+(\w+))");
    std::smatch m;
    if (std::regex_search(content, m, module_re)) {
        mod.name = m[1].str();
    }

    // Extract ports: input/output [signed] [width] name
    std::regex port_re(R"((input|output|inout)\s+(reg\s+)?(signed\s+)?(\[\d+:\d+\]\s*)?(\w+))");
    auto pit = std::sregex_iterator(content.begin(), content.end(), port_re);
    for (; pit != std::sregex_iterator(); ++pit) {
        VerilogPort p;
        p.direction = (*pit)[1].str();
        p.is_reg = (*pit)[2].matched;
        p.width = (*pit)[4].matched ? (*pit)[4].str() : "";
        // Trim whitespace from width
        while (!p.width.empty() && p.width.back() == ' ') p.width.pop_back();
        p.name = (*pit)[5].str();
        mod.ports.push_back(p);
    }

    // Extract wire declarations
    std::regex wire_re(R"(wire\s+(?:signed\s+)?(?:\[\d+:\d+\]\s*)?(\w+))");
    auto wit = std::sregex_iterator(content.begin(), content.end(), wire_re);
    for (; wit != std::sregex_iterator(); ++wit) {
        mod.wires.push_back((*wit)[1].str());
    }

    // Extract reg declarations (not part of port)
    std::regex reg_re(R"((?:^|\n)\s*reg\s+(?:signed\s+)?(?:\[\d+:\d+\]\s*)?(\w+))");
    auto rit = std::sregex_iterator(content.begin(), content.end(), reg_re);
    for (; rit != std::sregex_iterator(); ++rit) {
        mod.regs.push_back((*rit)[1].str());
    }

    // Extract parameter/localparam declarations
    std::regex param_re(R"((parameter|localparam)\s+(?:\[\d+:\d+\]\s*)?(\w+)\s*=\s*([^;]+))");
    auto pait = std::sregex_iterator(content.begin(), content.end(), param_re);
    for (; pait != std::sregex_iterator(); ++pait) {
        std::string val = (*pait)[3].str();
        while (!val.empty() && (val.back() == ' ' || val.back() == '\n')) val.pop_back();
        mod.params.push_back((*pait)[2].str() + " = " + val);
    }

    // Build signal set for reference analysis
    std::unordered_set<std::string> signals;
    for (auto& p : mod.ports) signals.insert(p.name);
    for (auto& w : mod.wires) signals.insert(w);
    for (auto& r : mod.regs) signals.insert(r);

    // Extract always blocks by scanning for balanced begin/end
    {
        std::regex always_start_re(R"(always\s*@\s*\([^)]*\))");
        auto ait = std::sregex_iterator(content.begin(), content.end(), always_start_re);
        for (; ait != std::sregex_iterator(); ++ait) {
            std::string sensitivity = ait->str();
            size_t block_start = ait->position() + ait->length();

            // Find begin/end block
            size_t begin_pos = content.find("begin", block_start);
            if (begin_pos == std::string::npos || begin_pos > block_start + 20) {
                // Single-statement always (no begin/end)
                size_t semi = content.find(';', block_start);
                if (semi != std::string::npos) {
                    std::string body = content.substr(block_start, semi - block_start + 1);
                    VerilogBlock blk;
                    blk.type = (sensitivity.find("posedge") != std::string::npos ||
                                sensitivity.find("negedge") != std::string::npos)
                               ? VerilogBlock::AlwaysFF : VerilogBlock::AlwaysComb;
                    blk.label = sensitivity;
                    blk.body = body;
                    extract_signal_refs(body, signals, blk.reads, blk.writes);
                    mod.blocks.push_back(std::move(blk));
                }
                continue;
            }

            // Track begin/end nesting
            int depth = 0;
            size_t end_pos = begin_pos;
            for (size_t i = begin_pos; i < content.size() - 2; i++) {
                if (content.substr(i, 5) == "begin") { depth++; i += 4; }
                else if (content.substr(i, 3) == "end" &&
                         (i + 3 >= content.size() || !std::isalnum(content[i + 3]))) {
                    depth--;
                    if (depth == 0) { end_pos = i + 3; break; }
                }
            }

            std::string body = content.substr(begin_pos, end_pos - begin_pos);
            VerilogBlock blk;
            blk.type = (sensitivity.find("posedge") != std::string::npos ||
                        sensitivity.find("negedge") != std::string::npos)
                       ? VerilogBlock::AlwaysFF : VerilogBlock::AlwaysComb;
            blk.label = sensitivity;
            blk.body = body;
            extract_signal_refs(body, signals, blk.reads, blk.writes);
            mod.blocks.push_back(std::move(blk));
        }
    }

    // Extract assign statements
    {
        std::regex assign_re(R"(assign\s+(\w+(?:\[\d+(?::\d+)?\])?)\s*=\s*([^;]+);)");
        auto ait = std::sregex_iterator(content.begin(), content.end(), assign_re);
        for (; ait != std::sregex_iterator(); ++ait) {
            VerilogBlock blk;
            blk.type = VerilogBlock::Assign;
            blk.label = "assign " + (*ait)[1].str();
            blk.body = (*ait)[0].str();
            std::string lhs = (*ait)[1].str();
            // Remove bit select for signal matching
            auto bracket = lhs.find('[');
            if (bracket != std::string::npos) lhs = lhs.substr(0, bracket);
            if (signals.count(lhs)) blk.writes.push_back(lhs);
            // RHS reads
            std::string rhs = (*ait)[2].str();
            std::regex id_re(R"(\b(\w+)\b)");
            auto iit = std::sregex_iterator(rhs.begin(), rhs.end(), id_re);
            std::unordered_set<std::string> seen;
            for (; iit != std::sregex_iterator(); ++iit) {
                std::string id = (*iit)[1].str();
                if (signals.count(id) && !seen.count(id)) {
                    blk.reads.push_back(id);
                    seen.insert(id);
                }
            }
            mod.blocks.push_back(std::move(blk));
        }
    }

    // Extract submodule instantiations (MODULE_TYPE [#(params)] INST_NAME (connections);)
    {
        static const std::unordered_set<std::string> kw = {
            "module", "endmodule", "input", "output", "inout", "wire", "reg",
            "assign", "parameter", "localparam", "always", "initial",
            "begin", "end", "if", "else", "for", "while", "case", "generate",
            "function", "task", "integer", "real", "genvar"
        };
        std::regex inst_re(R"((\w+)\s+(?:#\s*\([^)]*\)\s*)?(\w+)\s*\(([^;]*)\)\s*;)");
        auto iit = std::sregex_iterator(content.begin(), content.end(), inst_re);
        for (; iit != std::sregex_iterator(); ++iit) {
            std::string mod_type = (*iit)[1].str();
            std::string inst_name = (*iit)[2].str();
            std::string ports_str = (*iit)[3].str();
            if (kw.count(mod_type)) continue;
            if (ports_str.find('.') == std::string::npos) continue;

            VerilogBlock blk;
            blk.type = VerilogBlock::Instance;
            blk.module_type = mod_type;
            blk.inst_name = inst_name;
            blk.label = mod_type + " " + inst_name;
            blk.body = (*iit)[0].str();

            // Extract connected signals from port map
            std::regex port_conn_re(R"(\.(\w+)\s*\(\s*(\w+(?:\[\d+(?::\d+)?\])?)\s*\))");
            auto pit = std::sregex_iterator(ports_str.begin(), ports_str.end(), port_conn_re);
            std::unordered_set<std::string> seen;
            for (; pit != std::sregex_iterator(); ++pit) {
                std::string sig = (*pit)[2].str();
                auto bracket = sig.find('[');
                if (bracket != std::string::npos) sig = sig.substr(0, bracket);
                if (signals.count(sig) && !seen.count(sig)) {
                    blk.reads.push_back(sig);
                    seen.insert(sig);
                }
            }
            mod.blocks.push_back(std::move(blk));
        }
    }

    mod.valid = !mod.blocks.empty() || !mod.ports.empty();
    return mod;
}

// ============================================================================
// Power reader
// ============================================================================

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
