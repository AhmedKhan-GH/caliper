// program_frames.h: the draw half of a piece's instance from its wire messages
// (design/programmable-graphics.md, Resource Message and Browser Checks).
//
// A viewer gets each draw half as the instance form of PROGRAM set and each
// version of a published resource as RESOURCE. These helpers turn the decoded
// messages (src/host/net/wire.h) into what the draw runtime takes
// (src/host/draw/draw_runtime.h): the declaration that DrawRuntime::decode()
// reads, with the hash of the pin checked there, the packed bytes of the
// struct Params from the instance's values, and a ResourceVersion over a
// RESOURCE's section. The browser module and the pixel comparison's app side
// (tests/draw_parity/) include this one header, so that both decode the same
// frames the same way. Header only, and without WebGPU.
#pragma once

#include "draw/draw_runtime.h"
#include "net/wire.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace caliper_web {

// The declaration of an instance for DrawRuntime::decode(): its id, panel,
// pin, manifest and bindings, from the metadata of PROGRAM set.
inline std::string program_declaration(const caliper_net::wire::Program& m) {
    caliper_net::Json d = caliper_net::Json::object();
    d.set("id", m.id);
    d.set("panel", m.panel.value_or(""));
    if (m.piece.is_object()) d.set("piece", m.piece);
    if (m.manifest.is_object()) d.set("manifest", m.manifest);
    if (m.bind.is_object()) d.set("bind", m.bind);
    return d.dump();
}

// Decodes the piece and the instance of PROGRAM set with the checks of the
// draw runtime: piece.bad_manifest for a broken declaration, and
// piece.hash_mismatch when the files that the sections bring do not have the
// pin's hash. A message without the section draw is no draw half
// (piece.bad_manifest).
inline caliper_draw::Verdict decode_program(const caliper_net::wire::Program& m, caliper_builder::spec::Piece* piece,
                                            caliper_draw::Instance* instance,
                                            const caliper_draw::DrawChecks& checks = caliper_draw::DrawChecks::pieces()) {
    if (!m.draw || m.draw->empty()) {
        caliper_draw::Verdict v;
        v.code = caliper_builder::programs::kPieceBadManifest;
        v.message = "The program " + m.id + " came without its draw half.";
        return v;
    }
    return caliper_draw::DrawRuntime::decode(program_declaration(m), m.common.value_or(""), *m.draw, piece, instance,
                                             checks, m.derive.value_or(""));
}

// The bytes of the struct Params in the manifest's order, from the values by
// name (a number, or a boolean as 0 or 1), each clamped to its range, and the
// default for a value that is absent, as the app's draw host packs them.
inline std::vector<std::uint8_t> pack_params(const caliper_builder::spec::Piece& piece, const caliper_net::Json& values) {
    namespace cs = caliper_builder::spec;
    std::vector<std::uint8_t> out(piece.params.size() * 4, 0);
    for (std::size_t i = 0; i < piece.params.size(); ++i) {
        const cs::PieceParam& p = piece.params[i];
        double value = p.default_value;
        if (const caliper_net::Json* v = values.is_object() ? values.find(p.name) : nullptr) {
            if (v->is_number()) value = v->as_double();
            else if (v->is_bool()) value = v->as_bool() ? 1.0 : 0.0;
        }
        if (p.min && value < *p.min) value = *p.min;
        if (p.max && value > *p.max) value = *p.max;
        std::uint32_t bits = 0;
        if (p.type == cs::ParamType::f32) {
            const float f = float(value);
            std::memcpy(&bits, &f, 4);
        } else if (p.type == cs::ParamType::i32) {
            const double c = std::min(2147483647.0, std::max(-2147483648.0, value));
            bits = std::uint32_t(std::int32_t(c));
        } else {
            bits = std::uint32_t(std::min(4294967295.0, std::max(0.0, value)));
        }
        std::memcpy(out.data() + 4 * i, &bits, 4);
    }
    return out;
}

// A version of a resource over the section of a RESOURCE message; the bytes
// stay the message's.
inline caliper_draw::ResourceVersion resource_version(const caliper_net::wire::Resource& r) {
    caliper_draw::ResourceVersion v;
    v.resource = r.resource;
    v.step = std::int64_t(std::min<std::uint64_t>(r.step, std::uint64_t(INT64_MAX)));
    v.count = r.count;
    v.bytes = static_cast<const std::uint8_t*>(r.data.data);
    v.size = r.data.size;
    return v;
}

// The refusal of a program whose module or pipeline WebGPU refused, from an
// entry of the graphics layer's error list: "program <label>: ...", "render
// pipeline <label>: ..." or "compute pipeline <label>: ...", where the draw
// runtime's label of a pass is "<program> draw.passes[k]". An ok verdict when
// the entry names another program.
inline caliper_draw::Verdict refusal_of_error(const std::string& program, const std::string& error) {
    static const char* const kinds[] = {"program", "render pipeline", "compute pipeline"};
    for (const char* kind : kinds) {
        const std::string head = std::string(kind) + " " + program + " ";
        if (error.compare(0, head.size(), head) != 0) continue;
        const std::string pass = "draw.passes[";
        if (error.compare(head.size(), pass.size(), pass) != 0) continue;
        const std::size_t close = error.find(']', head.size());
        if (close == std::string::npos) continue;
        const std::size_t colon = error.find(": ", close);
        caliper_draw::Verdict v;
        v.code = caliper_builder::programs::kProgramBrowserRefused;
        v.where = error.substr(head.size(), close + 1 - head.size());
        v.message = "WebGPU refused the " + std::string(kind) + " of " + v.where + ": " +
                    (colon == std::string::npos ? error : error.substr(colon + 2));
        return v;
    }
    return {};
}

}  // namespace caliper_web
