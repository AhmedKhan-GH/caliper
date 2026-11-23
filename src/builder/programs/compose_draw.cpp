// The composer of a piece's draw half: the prelude and the generated header
// of each pass (programs/pieces.h, design/programmable-graphics.md, Generated
// Header). This file needs only the standard library and spec/piece.h,
// because the browser module compiles it too.
//
// A composed pass is, in this order: the draw prelude, the generated header,
// the imported lib modules, common.wgsl and draw.wgsl. The header declares,
// in a fixed order that the app and the browser module bind without
// reflection:
//  - group 0, binding 5: `params: Params`, when the piece has params;
//  - group 1, binding 0: `counts: CaliperCounts`, then from binding 1 each
//    published resource in the manifest's order and after them each buffer
//    input in the manifest's order, read-only (a texture resource as
//    texture_2d<f32>);
//  - group 2: each draw buffer at its index, read_write only in a compute
//    pass that names it in `writes`, and each offscreen target at 4 + its
//    index, as texture_2d<f32>; the host binds a target only in a pass that
//    reads it.
// The structs of the layouts come before the bindings, each once.
//
// WGSL checks every function of a module, not only those that an entry point
// calls. Two rules keep each pass's module valid, although draw.wgsl holds the
// functions of every pass: the header declares every target in every pass,
// so that a function that samples a target compiles in the passes that write
// it, and each pass keeps only its own entry points of common.wgsl and
// draw.wgsl, the others blanked to spaces with their lines kept, so that a
// compute entry that writes a draw buffer is not in a render pass whose
// vertex stage reads that buffer read-only. A function that is no entry point
// and writes a draw buffer thus compiles only in the passes that write it.
#include <caliper_builder/programs/pieces.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace caliper_builder::programs {

namespace {

Verdict refuse(const char* code, std::string message, std::string where) {
    Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.where = std::move(where);
    return v;
}

std::uint32_t count_lines(const std::string& text) {
    std::uint32_t n = 0;
    for (char c : text)
        if (c == '\n') ++n;
    return n;
}

// Appends one part with a closing newline and records its segment.
void append_part(Composed* c, std::string part, const std::string& text, std::uint32_t* line) {
    if (text.empty()) return;
    std::string body = text;
    if (body.back() != '\n') body += '\n';
    Segment s;
    s.part = std::move(part);
    s.first_line = *line;
    s.lines = count_lines(body);
    c->segments.push_back(s);
    c->text += body;
    *line += s.lines;
}

// The WGSL type of one element of a layout: its plain type or its struct's name.
std::string element_type(const spec::Layout& layout) {
    if (layout.type_name) return *layout.type_name;
    if (layout.element) return *layout.element;
    return {};
}

// Declares the struct of a layout once; false with *why when a struct of the
// same name with other fields was declared already.
bool declare_struct(const spec::Layout& layout, std::vector<std::pair<std::string, spec::Layout>>* declared,
                    std::string* out, std::string* why) {
    if (!layout.type_name) return true;
    for (const auto& d : *declared) {
        if (d.first != *layout.type_name) continue;
        if (d.second.fields == layout.fields) return true;
        *why = "the struct " + *layout.type_name + " is declared twice with other fields";
        return false;
    }
    declared->emplace_back(*layout.type_name, layout);
    *out += "struct " + *layout.type_name + " { ";
    for (std::size_t i = 0; i < layout.fields.size(); ++i) {
        if (i) *out += ", ";
        *out += layout.fields[i].name + ": " + layout.fields[i].type;
    }
    *out += " }\n";
    return true;
}

// Whether text[i, i + word) is the identifier `word`, not a part of a longer one.
bool word_at(const std::string& text, std::size_t i, const char* word) {
    const std::size_t n = std::char_traits<char>::length(word);
    if (text.compare(i, n, word) != 0) return false;
    auto ident = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    if (i > 0 && ident(text[i - 1])) return false;
    return i + n >= text.size() || !ident(text[i + n]);
}

// Blanks each entry point of a WGSL text whose name is not in `keep`: from its first attribute
// to the closing brace of its body, every character but a newline becomes a space, thus every
// line and column of the rest stays. Comments (nested block comments too) are skipped.
std::string blank_other_entries(const std::string& text, const std::set<std::string>& keep) {
    const std::size_t n = text.size();
    std::vector<bool> code(n, true);
    for (std::size_t i = 0; i < n;) {
        if (text.compare(i, 2, "//") == 0) {
            while (i < n && text[i] != '\n') code[i++] = false;
        } else if (text.compare(i, 2, "/*") == 0) {
            int depth = 0;
            do {
                if (text.compare(i, 2, "/*") == 0) {
                    ++depth;
                    code[i] = code[i + 1] = false;
                    i += 2;
                } else if (text.compare(i, 2, "*/") == 0) {
                    --depth;
                    code[i] = code[i + 1] = false;
                    i += 2;
                } else {
                    code[i++] = false;
                }
            } while (i < n && depth > 0);
        } else {
            ++i;
        }
    }
    std::string out = text;
    int depth = 0;
    std::size_t item = 0;   // where the current item of the top level starts
    for (std::size_t i = 0; i < n; ++i) {
        if (!code[i]) continue;
        const char c = text[i];
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            if (--depth == 0) item = i + 1;
        } else if (depth == 0 && c == ';') {
            item = i + 1;
        } else if (depth == 0 && word_at(text, i, "fn")) {
            std::size_t j = i + 2;
            while (j < n && std::isspace(static_cast<unsigned char>(text[j]))) ++j;
            std::size_t e = j;
            while (e < n && (std::isalnum(static_cast<unsigned char>(text[e])) || text[e] == '_')) ++e;
            const std::string name = text.substr(j, e - j);
            bool entry = false;
            std::size_t first = i;
            for (std::size_t a = item; a < i; ++a) {
                if (!code[a] || text[a] != '@') continue;
                first = std::min(first, a);
                if (word_at(text, a + 1, "vertex") || word_at(text, a + 1, "fragment") || word_at(text, a + 1, "compute"))
                    entry = true;
            }
            // The body: the first brace after the name, to its match.
            std::size_t b = e;
            while (b < n && !(code[b] && text[b] == '{')) ++b;
            int d = 0;
            std::size_t end = b;
            for (; end < n; ++end) {
                if (!code[end]) continue;
                if (text[end] == '{') ++d;
                if (text[end] == '}' && --d == 0) break;
            }
            if (end >= n) return out;   // no closing brace: the compiler reports the text as it is
            if (entry && keep.count(name) == 0)
                for (std::size_t k = first; k <= end; ++k)
                    if (out[k] != '\n') out[k] = ' ';
            i = end;
            item = end + 1;
        }
    }
    return out;
}

}  // namespace

Composed compose_draw(const spec::Piece& piece, const Resolved& resolved, std::size_t pass) {
    Composed c;
    const std::string where = "draw.passes[" + std::to_string(pass) + "]";
    if (!resolved.verdict.ok()) {
        c.verdict = resolved.verdict;
        return c;
    }
    if (!piece.draw) {
        c.verdict = refuse(kProgramBadPass, "The piece " + piece.name + " has no draw half.", "draw");
        return c;
    }
    const spec::Draw& draw = *piece.draw;
    if (pass >= draw.passes.size()) {
        c.verdict = refuse(kProgramBadPass,
                           "The draw half of " + piece.name + " has " + std::to_string(draw.passes.size()) +
                               " passes, thus no pass " + std::to_string(pass) + ".",
                           where);
        return c;
    }
    const spec::DrawPass& p = draw.passes[pass];
    const auto draw_file = resolved.files.find("draw.wgsl");
    if (draw_file == resolved.files.end() || draw_file->second.empty()) {
        c.verdict = refuse(kProgramEntryStage, "The piece " + piece.name + " has a draw half but no draw.wgsl.", where);
        return c;
    }

    // The generated header: params, the counts, the layouts' structs, then
    // the bindings of groups 1 and 2.
    std::string params, counts, structs, bindings, why;
    std::vector<std::pair<std::string, spec::Layout>> declared;
    if (!piece.params.empty()) {
        params += "struct Params { ";
        for (std::size_t i = 0; i < piece.params.size(); ++i) {
            if (i) params += ", ";
            params += piece.params[i].name + ": " + spec::param_type_name(piece.params[i].type);
        }
        params += " }\n";
        params += "@group(0) @binding(" + std::to_string(kBindingParams) + ") var<uniform> params: Params;\n";
    }

    // Group 1: the counts, the published resources, then the buffer inputs.
    std::vector<std::string> counted;
    std::string group1;
    std::uint32_t binding = 1;
    for (const spec::Resource& r : piece.resources) {
        if (!r.publish) continue;
        counted.push_back(r.id);
        if (r.kind == spec::ResourceKind::texture) {
            group1 += "@group(1) @binding(" + std::to_string(binding++) + ") var " + r.id + ": texture_2d<f32>;\n";
            continue;
        }
        if (!declare_struct(r.layout, &declared, &structs, &why)) {
            c.verdict = refuse(kProgramBadLayout, "The resource " + r.id + ": " + why + ".", "resources");
            return c;
        }
        group1 += "@group(1) @binding(" + std::to_string(binding++) + ") var<storage, read> " + r.id + ": array<" +
                  element_type(r.layout) + ">;\n";
    }
    for (const spec::PieceInput& in : piece.inputs) {
        if (in.kind != spec::InputKind::buffer) continue;
        counted.push_back(in.name);
        if (!declare_struct(in.layout, &declared, &structs, &why)) {
            c.verdict = refuse(kProgramBadLayout, "The input " + in.name + ": " + why + ".", "inputs");
            return c;
        }
        group1 += "@group(1) @binding(" + std::to_string(binding++) + ") var<storage, read> " + in.name + ": array<" +
                  element_type(in.layout) + ">;\n";
    }
    if (!counted.empty()) {
        counts += "struct CaliperCounts { ";
        for (std::size_t i = 0; i < counted.size(); ++i) {
            if (i) counts += ", ";
            counts += counted[i] + ": u32";
        }
        counts += " }\n";
        bindings += "@group(1) @binding(0) var<uniform> counts: CaliperCounts;\n";
    }
    bindings += group1;

    // Group 2: the draw buffers, then the targets that this pass reads.
    const std::set<std::string> writes(p.writes.begin(), p.writes.end());
    for (std::size_t i = 0; i < draw.buffers.size(); ++i) {
        const spec::DrawBuffer& b = draw.buffers[i];
        if (!declare_struct(b.layout, &declared, &structs, &why)) {
            c.verdict = refuse(kProgramBadLayout, "The draw buffer " + b.id + ": " + why + ".",
                               "draw.buffers[" + std::to_string(i) + "]");
            return c;
        }
        const bool rw = p.kind == spec::PassKind::compute && writes.count(b.id) != 0;
        bindings += "@group(2) @binding(" + std::to_string(i) + ") var<storage, " + (rw ? "read_write" : "read") +
                    "> " + b.id + ": array<" + element_type(b.layout) + ">;\n";
    }
    for (const std::string& read : p.reads) {
        bool known = false;
        for (const spec::DrawTarget& t : draw.targets) known = known || t.id == read;
        if (!known) {
            c.verdict = refuse(kProgramUnknownResource, "The pass reads " + read + ", which is no target of the piece.",
                               where);
            return c;
        }
    }
    for (std::size_t t = 0; t < draw.targets.size(); ++t)
        bindings += "@group(2) @binding(" + std::to_string(4 + t) + ") var " + draw.targets[t].id + ": texture_2d<f32>;\n";

    std::uint32_t line = 1;
    append_part(&c, "prelude", std::string(draw_prelude()), &line);
    append_part(&c, "header", params + counts + structs + bindings, &line);
    // The lib modules, with their segments moved to their place in the module.
    if (!resolved.lib_text.empty()) {
        const std::uint32_t first = line;
        std::string lib = resolved.lib_text;
        if (lib.back() != '\n') lib += '\n';
        for (Segment s : resolved.lib_segments) {
            s.first_line += first - 1;
            c.segments.push_back(s);
        }
        if (resolved.lib_segments.empty()) {
            Segment s;
            s.part = "lib";
            s.first_line = first;
            s.lines = count_lines(lib);
            c.segments.push_back(s);
        }
        c.text += lib;
        line += count_lines(lib);
    }
    // This pass's own entry points; the prelude's caliper_fullscreen stays in every pass.
    std::set<std::string> keep;
    if (p.kind == spec::PassKind::compute) {
        keep.insert(p.entry);
    } else {
        if (p.vertex) keep.insert(*p.vertex);
        keep.insert(p.fragment);
    }
    const auto common = resolved.files.find("common.wgsl");
    if (common != resolved.files.end()) append_part(&c, "common.wgsl", blank_other_entries(common->second, keep), &line);
    append_part(&c, "draw.wgsl", blank_other_entries(draw_file->second, keep), &line);
    return c;
}

}  // namespace caliper_builder::programs
