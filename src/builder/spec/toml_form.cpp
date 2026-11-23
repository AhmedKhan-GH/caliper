// The TOML form of a spec: parse_toml turns the text into the JSON tree that
// json_form.cpp decodes, recording the position of every value; to_toml writes
// the canonical TOML text from the canonical JSON form.
#include <caliper_builder/spec/format.h>

#include <cmath>
#include <cstdio>

#include <toml++/toml.hpp>

#include "internal.h"

namespace caliper_builder::spec {

using detail::json;

namespace {

// ---- reading ------------------------------------------------------------------

// toml++ bounds the nesting of inline arrays and tables, but not the number of
// segments of a dotted key, and each segment is a nested table that its
// destructor frees recursively. This scan finds the longest chain of key
// segments joined by dots outside strings and comments before toml++ sees the
// text. A float or a date has at most one dot, so a longer chain is always a
// dotted key.
struct Chain {
    std::size_t length = 0;
    Position at;
};

Chain longest_dotted_chain(std::string_view s) {
    Chain best;
    std::size_t chain = 0;  // dots in the current chain
    Position chain_at{1, 1};
    bool after_segment = false;  // the last token was a bare run or a quoted key
    int line = 1, column = 1;
    std::size_t i = 0;
    auto advance = [&](std::size_t n) {
        for (std::size_t k = 0; k < n && i < s.size(); ++k, ++i) {
            if (s[i] == '\n') {
                ++line;
                column = 1;
            } else if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
                ++column;
            }
        }
    };
    auto reset = [&] {
        chain = 0;
        after_segment = false;
    };
    while (i < s.size()) {
        const char c = s[i];
        if (c == '#') {  // a comment runs to the end of the line
            while (i < s.size() && s[i] != '\n') advance(1);
            reset();
            continue;
        }
        if (c == '"' || c == '\'') {
            const bool multi = s.compare(i, 3, std::string(3, c)) == 0;
            advance(multi ? 3 : 1);
            while (i < s.size()) {
                if (c == '"' && s[i] == '\\') {
                    advance(2);
                    continue;
                }
                if (multi) {
                    if (s.compare(i, 3, std::string(3, c)) == 0) {
                        advance(3);
                        while (i < s.size() && s[i] == c) advance(1);  // up to two quotes may close it
                        break;
                    }
                } else if (s[i] == c || s[i] == '\n') {
                    advance(1);
                    break;
                }
                advance(1);
            }
            after_segment = true;
            continue;
        }
        if (c == ' ' || c == '\t') {
            advance(1);
            continue;
        }
        if (c == '.') {
            if (after_segment) {
                if (chain == 0) chain_at = {line, column};
                ++chain;
                if (chain > best.length) best = {chain, chain_at};
            }
            after_segment = false;
            advance(1);
            continue;
        }
        const bool bare = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
                          c == '-' || c == '+' || c == ':';
        if (bare) {
            while (i < s.size()) {
                const char d = s[i];
                const bool more = (d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') || (d >= '0' && d <= '9') ||
                                  d == '_' || d == '-' || d == '+' || d == ':';
                if (!more) break;
                advance(1);
            }
            after_segment = true;
            continue;
        }
        reset();
        advance(1);
    }
    return best;
}

class Reader {
public:
    Reader(SourceMap& positions, detail::Sink& sink) : positions_(positions), sink_(sink) {}

    bool convert(const toml::node& n, const std::string& path, int depth, json& out) {
        const auto& begin = n.source().begin;
        positions_[path] = {static_cast<int>(begin.line), static_cast<int>(begin.column)};
        if (++count_ > limits::kValues) {
            sink_.error("spec.too_many", "The document holds more than " + std::to_string(limits::kValues) + " values.",
                        path);
            return false;
        }
        if ((n.is_table() || n.is_array()) && depth > limits::kDepth) {
            sink_.error("spec.too_deep",
                        path + " nests more than " + std::to_string(limits::kDepth) + " levels of tables and arrays.",
                        path);
            return false;
        }
        switch (n.type()) {
            case toml::node_type::table: {
                out = json::object();
                bool ok = true;
                for (auto&& [key, value] : *n.as_table()) {
                    const std::string k(key.str());
                    json child;
                    if (convert(value, detail::key_path(path, k), depth + 1, child)) {
                        out[k] = std::move(child);
                    } else {
                        ok = false;
                        if (stop()) return false;
                    }
                }
                return ok;
            }
            case toml::node_type::array: {
                out = json::array();
                bool ok = true;
                std::size_t index = 0;
                for (const toml::node& value : *n.as_array()) {
                    json child;
                    if (convert(value, detail::index_path(path, index++), depth + 1, child)) {
                        out.push_back(std::move(child));
                    } else {
                        ok = false;
                        if (stop()) return false;
                    }
                }
                return ok;
            }
            case toml::node_type::string: out = n.as_string()->get(); return true;
            case toml::node_type::integer: out = n.as_integer()->get(); return true;
            case toml::node_type::floating_point: out = n.as_floating_point()->get(); return true;
            case toml::node_type::boolean: out = n.as_boolean()->get(); return true;
            case toml::node_type::date:
            case toml::node_type::time:
            case toml::node_type::date_time:
                sink_.error("spec.wrong_type",
                            (path.empty() ? std::string("A value") : path) +
                                " is a date or a time, which a spec cannot hold; write it as a string.",
                            path);
                return false;
            default:
                sink_.error("spec.wrong_type", path + " is not a value a spec can hold.", path);
                return false;
        }
    }

private:
    bool stop() const { return count_ > limits::kValues || sink_.full(); }

    SourceMap& positions_;
    detail::Sink& sink_;
    std::size_t count_ = 0;
};

// ---- writing ------------------------------------------------------------------

std::string number_text(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
    // nlohmann/json writes the shortest text that reads back to the same
    // double, and always with a '.' or an exponent, as TOML needs.
    return json(d).dump();
}

std::string string_text(std::string_view raw, bool allow_multiline) {
    const std::string s = detail::valid_utf8(raw) ? std::string(raw) : detail::to_valid_utf8(raw);
    const bool multi = allow_multiline && s.find('\n') != std::string::npos;
    std::string out = multi ? "\"\"\"\n" : "\"";
    for (char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += multi ? "\n" : "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20 || c == 0x7F) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04X", c);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    out += multi ? "\"\"\"" : "\"";
    return out;
}

std::string key_text(const std::string& k) {
    bool bare = !k.empty();
    for (char c : k)
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-'))
            bare = false;
    return bare ? k : string_text(k, false);
}

const std::vector<std::string>& layout_order() {
    static const std::vector<std::string> v = {"panel", "min", "split", "tabs", "ratio", "shares", "title", "children", "x"};
    return v;
}

class Writer {
public:
    std::string document(const json& j) {
        line("caliper_spec", j.at("caliper_spec"), false);
        auto layout = j.find("layout");
        if (layout != j.end() && layout->is_string()) line("layout", *layout, false);
        if (auto x = j.find("x"); x != j.end()) line("x", *x, false);

        out_ += "\n[applet]\n";
        table(j.at("applet"), {"id", "title", "description", "x"}, false, "");

        if (auto sources = j.find("source"); sources != j.end()) {
            for (const json& s : *sources) {
                out_ += "\n[[source]]\n";
                table(s, {"id", "catalog", "sql", "from", "history", "x"}, false, "");
                if (auto ts = s.find("transform"); ts != s.end()) {
                    for (const json& t : *ts) {
                        out_ += "  [[source.transform]]\n";
                        table(t,
                              {"op", "axis", "index", "start", "stop", "step", "fn", "keepdims", "mode", "steps",
                               "max_cells", "pool", "columns", "max_points", "x"},
                              false, "  ");
                    }
                }
            }
        }
        if (auto states = j.find("state"); states != j.end()) {
            for (const json& v : *states) {
                out_ += "\n[[state]]\n";
                table(v, {"id", "title", "type", "default", "min", "max", "choices", "x"}, false, "");
            }
        }
        if (auto panels = j.find("panel"); panels != j.end()) {
            for (const json& p : *panels) {
                out_ += "\n[[panel]]\n";
                table(p, {"id", "primitive", "title", "bind", "options", "x"}, false, "");
                if (auto es = p.find("entity"); es != p.end()) {
                    for (const json& e : *es) {
                        out_ += "  [[panel.entity]]\n";
                        table(e, {"path", "draw", "title", "bind", "options", "pose", "x"}, false, "  ");
                    }
                }
                if (auto as = p.find("annotation"); as != p.end()) {
                    for (const json& a : *as) {
                        out_ += "  [[panel.annotation]]\n";
                        table(a,
                              {"id", "kind", "axis", "value", "range", "at", "text", "fields", "leader", "emphasis",
                               "x"},
                              false, "  ");
                    }
                }
            }
        }
        if (auto controls = j.find("control"); controls != j.end()) {
            for (const json& c : *controls) {
                out_ += "\n[[control]]\n";
                table(c, {"id", "kind", "title", "bind", "options", "x"}, false, "");
            }
        }
        // A program's and a stage's WGSL come last in their tables, as
        // multi-line strings, so that the short keys stay together above them.
        if (auto programs = j.find("program"); programs != j.end()) {
            for (const json& p : *programs) {
                out_ += "\n[[program]]\n";
                table(p,
                      {"id", "title", "piece", "panel", "bind", "entry", "inputs", "output", "limits", "workgroups",
                       "params", "x", "wgsl"},
                      false, "");
            }
        }
        if (auto stages = j.find("stage"); stages != j.end()) {
            for (const json& s : *stages) {
                out_ += "\n[[stage]]\n";
                table(s, {"panel", "entity", "kind", "params", "x", "wgsl"}, false, "");
            }
        }
        // Inline pieces (design/programmable-graphics.md, Piece Files): the
        // manifest's tables under piece., and the WGSL last in [piece.files].
        if (auto pieces = j.find("piece"); pieces != j.end()) {
            for (const json& p : *pieces) {
                out_ += "\n[[piece]]\n";
                piece(p, "piece.");
            }
        }
        if (auto provenance = j.find("provenance"); provenance != j.end()) {
            out_ += "\n[provenance]\n";
            table(*provenance, {"foreign", "from", "reviewed", "x"}, false, "");
        }
        if (layout != j.end() && layout->is_object()) {
            out_ += "\n[layout]\n";
            table(*layout, layout_order(), true, "");
        }
        return out_;
    }

    // A piece's keys and tables, with its tables' names under `prefix`: ""
    // in a piece.toml, "piece." in a spec. Its files come last, as
    // multi-line strings, when the JSON has them.
    std::string piece(const json& j, const std::string& prefix) {
        for (const char* k : {"name", "version", "description", "imports", "hash"})
            if (auto it = j.find(k); it != j.end()) wrapped(k, *it);
        const std::pair<const char*, std::vector<std::string>> lists[] = {
            {"inputs", {"name", "kind", "dtype", "rank_min", "rank_max", "element", "struct", "fields", "title", "optional"}},
            {"params", {"name", "type", "default", "min", "max", "title"}},
            {"resources",
             {"id", "kind", "element", "struct", "fields", "count", "counted", "width", "height", "format", "texels",
              "publish", "clear", "keep"}},
            {"depends", {"name", "version"}}};
        for (const auto& [key, order] : lists) {
            auto it = j.find(key);
            if (it == j.end()) continue;
            for (const json& t : *it) {
                out_ += "\n[[" + prefix + key + "]]\n";
                table(t, order, false, "");
            }
        }
        const std::pair<const char*, std::vector<std::string>> halves[] = {
            {"derive", {"time_ms", "memory_mb", "every", "passes"}},
            {"draw", {"camera", "frame_ms", "memory_mb", "buffers", "targets", "passes", "pick", "text"}}};
        for (const auto& [key, order] : halves) {
            auto it = j.find(key);
            if (it == j.end()) continue;
            out_ += "\n[" + prefix + key + "]\n";
            for (const std::string& k : order)
                if (auto e = it->find(k); e != it->end()) wrapped(k, *e);
        }
        if (auto it = j.find("sample"); it != j.end()) {
            for (const json& t : *it) {
                out_ += "\n[[" + prefix + "sample]]\n";
                table(t, {"input", "shape"}, false, "");
            }
        }
        if (auto it = j.find("files"); it != j.end()) {
            out_ += "\n[" + prefix + "files]\n";
            table(*it, {"common", "derive", "draw"}, false, "");
        }
        return out_;
    }

private:
    // A key whose array, when its inline text is long, takes one element a
    // line.
    void wrapped(const std::string& key, const json& v) {
        const std::string inline_text = value(v, false);
        if (!v.is_array() || v.empty() || inline_text.size() + key.size() <= 96) {
            out_ += key_text(key) + " = " + inline_text + "\n";
            return;
        }
        out_ += key_text(key) + " = [\n";
        for (const json& e : v) out_ += "  " + value(e, false) + ",\n";
        out_ += "]\n";
    }

    void line(const std::string& key, const json& v, bool in_layout, const std::string& indent = "") {
        out_ += indent + key_text(key) + " = " + value(v, in_layout) + "\n";
    }

    // The keys that a source and a panel write as their own arrays of tables.
    static bool is_nested_array(const std::string& key) {
        return key == "transform" || key == "entity" || key == "annotation";
    }

    // Writes the keys of `t` in `order`; keys not in the order follow in byte order.
    void table(const json& t, const std::vector<std::string>& order, bool in_layout, const std::string& indent) {
        for (const std::string& k : order)
            if (auto it = t.find(k); it != t.end()) line(k, *it, in_layout && (k == "children" || k == "tabs"), indent);
        for (auto it = t.begin(); it != t.end(); ++it)
            if (!detail::contains(order, it.key()) && !is_nested_array(it.key()))
                line(it.key(), it.value(), false, indent);
    }

    std::string value(const json& v, bool in_layout) {
        switch (v.type()) {
            case json::value_t::string: return string_text(v.get_ref<const std::string&>(), true);
            case json::value_t::boolean: return v.get<bool>() ? "true" : "false";
            case json::value_t::number_integer: return std::to_string(v.get<std::int64_t>());
            case json::value_t::number_unsigned: return std::to_string(v.get<std::uint64_t>());
            case json::value_t::number_float: return number_text(v.get<double>());
            case json::value_t::array: {
                std::string s = "[";
                bool first = true;
                for (const json& e : v) {
                    if (!first) s += ", ";
                    first = false;
                    s += value(e, in_layout);
                }
                return s + "]";
            }
            case json::value_t::object: {
                if (v.empty()) return "{}";
                std::string s = "{ ";
                bool first = true;
                auto item = [&](const std::string& k, const json& e, bool layout_child) {
                    if (!first) s += ", ";
                    first = false;
                    s += key_text(k) + " = " + value(e, layout_child);
                };
                if (in_layout) {  // a nested layout node keeps the layout order
                    for (const std::string& k : layout_order())
                        if (auto it = v.find(k); it != v.end()) item(k, *it, k == "children" || k == "tabs");
                    for (auto it = v.begin(); it != v.end(); ++it)
                        if (!detail::contains(layout_order(), it.key())) item(it.key(), it.value(), false);
                } else {
                    for (auto it = v.begin(); it != v.end(); ++it) item(it.key(), it.value(), false);
                }
                return s + " }";
            }
            default: return "\"\"";
        }
    }

    std::string out_;
};

}  // namespace

ParseResult parse_toml(std::string_view text) {
    ParseResult r;
    try {
        if (text.size() > limits::kTextBytes) {
            r.diagnostics.push_back({Severity::error, "spec.too_large",
                                     "The spec is " + std::to_string(text.size()) + " bytes long; the limit is " +
                                         std::to_string(limits::kTextBytes) + " bytes.",
                                     ""});
            return r;
        }
        const Chain chain = longest_dotted_chain(text);
        if (chain.length >= static_cast<std::size_t>(limits::kDepth)) {
            Diagnostic d{Severity::error, "spec.too_deep",
                         "A dotted key has more than " + std::to_string(limits::kDepth) + " parts.", ""};
            d.line = chain.at.line;
            d.column = chain.at.column;
            r.diagnostics.push_back(d);
            return r;
        }

        toml::table root;
#if TOML_EXCEPTIONS
        try {
            root = toml::parse(text);
        } catch (const toml::parse_error& e) {
            Diagnostic d{Severity::error, "spec.syntax", "The text is not valid TOML: " + std::string(e.description()) + ".",
                         ""};
            d.line = static_cast<int>(e.source().begin.line);
            d.column = static_cast<int>(e.source().begin.column);
            r.diagnostics.push_back(d);
            return r;
        }
#else
        toml::parse_result parsed = toml::parse(text);
        if (!parsed) {
            const auto& e = parsed.error();
            Diagnostic d{Severity::error, "spec.syntax", "The text is not valid TOML: " + std::string(e.description()) + ".",
                         ""};
            d.line = static_cast<int>(e.source().begin.line);
            d.column = static_cast<int>(e.source().begin.column);
            r.diagnostics.push_back(d);
            return r;
        }
        root = std::move(parsed).table();
#endif

        json doc;
        detail::Sink sink(r.diagnostics);
        Reader reader(r.positions, sink);
        const bool converted = reader.convert(root, "", 0, doc);
        r.positions[""] = {1, 1};
        if (converted) {
            std::size_t count = 0;
            if (detail::check_values(doc, "", 0, count, sink)) {
                ParseResult decoded = from_json(doc);
                if (decoded.ok()) {
                    r.spec = std::move(decoded.spec);
                } else {
                    for (Diagnostic& d : decoded.diagnostics) r.diagnostics.push_back(std::move(d));
                }
            }
        }
        locate(r.diagnostics, r.positions);
    } catch (...) {
        r.diagnostics.push_back({Severity::error, "spec.internal", "The spec library failed unexpectedly.", ""});
    }
    if (!r.ok()) r.spec = Spec{};
    return r;
}

std::string to_toml(const Spec& spec) {
    try {
        return Writer().document(to_json(spec));
    } catch (...) {
        return std::string();
    }
}

PieceParse piece_from_toml(std::string_view text) {
    PieceParse r;
    try {
        if (text.size() > limits::kTextBytes) {
            r.diagnostics.push_back({Severity::error, "spec.too_large",
                                     "The manifest is " + std::to_string(text.size()) + " bytes long; the limit is " +
                                         std::to_string(limits::kTextBytes) + " bytes.",
                                     ""});
            return r;
        }
        const Chain chain = longest_dotted_chain(text);
        if (chain.length >= static_cast<std::size_t>(limits::kDepth)) {
            Diagnostic d{Severity::error, "spec.too_deep",
                         "A dotted key has more than " + std::to_string(limits::kDepth) + " parts.", ""};
            d.line = chain.at.line;
            d.column = chain.at.column;
            r.diagnostics.push_back(d);
            return r;
        }
        toml::table root;
#if TOML_EXCEPTIONS
        try {
            root = toml::parse(text);
        } catch (const toml::parse_error& e) {
            Diagnostic d{Severity::error, "spec.syntax", "The text is not valid TOML: " + std::string(e.description()) + ".",
                         ""};
            d.line = static_cast<int>(e.source().begin.line);
            d.column = static_cast<int>(e.source().begin.column);
            r.diagnostics.push_back(d);
            return r;
        }
#else
        toml::parse_result parsed = toml::parse(text);
        if (!parsed) {
            const auto& e = parsed.error();
            Diagnostic d{Severity::error, "spec.syntax", "The text is not valid TOML: " + std::string(e.description()) + ".",
                         ""};
            d.line = static_cast<int>(e.source().begin.line);
            d.column = static_cast<int>(e.source().begin.column);
            r.diagnostics.push_back(d);
            return r;
        }
        root = std::move(parsed).table();
#endif
        json doc;
        SourceMap positions;
        detail::Sink sink(r.diagnostics);
        Reader reader(positions, sink);
        if (reader.convert(root, "", 0, doc)) {
            PieceParse decoded = piece_from_json_value(doc, "");
            r.piece = std::move(decoded.piece);
            for (Diagnostic& d : decoded.diagnostics) r.diagnostics.push_back(std::move(d));
        }
        positions[""] = {1, 1};
        locate(r.diagnostics, positions);
    } catch (...) {
        r.diagnostics.push_back({Severity::error, "spec.internal", "The spec library failed unexpectedly.", ""});
    }
    if (!r.ok()) r.piece = Piece{};
    return r;
}

std::string piece_to_toml(const Piece& piece) {
    try {
        json j = to_json(piece);
        j.erase("files");
        return Writer().piece(j, "");
    } catch (...) {
        return std::string();
    }
}

void locate(std::vector<Diagnostic>& diagnostics, const SourceMap& positions) {
    for (Diagnostic& d : diagnostics) {
        if (d.line > 0) continue;
        std::string p = d.path;
        while (true) {
            auto it = positions.find(p);
            if (it != positions.end()) {
                d.line = it->second.line;
                d.column = it->second.column;
                break;
            }
            if (p.empty()) break;
            // Drop the last part: "a.b[2]" -> "a.b" -> "a" -> "".
            const std::size_t cut = p.find_last_of(".[");
            p = cut == std::string::npos ? std::string() : p.substr(0, cut);
        }
    }
}

}  // namespace caliper_builder::spec
