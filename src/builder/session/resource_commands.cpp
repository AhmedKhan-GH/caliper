// program.read (resource_commands.h, design/programmable-graphics.md, Program
// Commands): the elements of a published resource of a program at a step,
// from the versions that the step history keeps. Every element and field is
// read inside the version's bytes: an element's offset is its index times the
// stride, both bounded by the wire (2^32 elements of at most 256 bytes), and a
// field ends inside its element.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resource_commands.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace caliper_builder {

namespace {

using json = nlohmann::json;

constexpr std::uint64_t kReadDefault = 16;
constexpr std::uint64_t kReadMax = 1024;

// The components and the scalar kind of a field type; 0 components for a type
// that program.read does not know.
struct FieldShape {
    std::uint32_t components = 0;
    char kind = 'f';   // 'f' f32, 'i' i32, 'u' u32
};

FieldShape field_shape(const std::string& type) {
    if (type == "f32") return {1, 'f'};
    if (type == "i32") return {1, 'i'};
    if (type == "u32") return {1, 'u'};
    if (type == "vec2<f32>") return {2, 'f'};
    if (type == "vec4<f32>") return {4, 'f'};
    if (type == "vec2<u32>") return {2, 'u'};
    if (type == "vec4<u32>") return {4, 'u'};
    return {};
}

void read_scalar(const std::uint8_t* p, char kind, json& out) {
    if (kind == 'f') {
        float f;
        std::memcpy(&f, p, 4);
        out.push_back(f);
    } else if (kind == 'i') {
        std::int32_t i;
        std::memcpy(&i, p, 4);
        out.push_back(i);
    } else {
        std::uint32_t u;
        std::memcpy(&u, p, 4);
        out.push_back(u);
    }
}

// The numbers of one element: each field's components in order, or without
// fields the element's 32-bit words as f32.
json read_element(const std::uint8_t* element, std::uint32_t stride, const std::vector<ResourceField>& fields) {
    json out = json::array();
    if (fields.empty()) {
        for (std::uint32_t w = 0; w + 4 <= stride; w += 4) read_scalar(element + w, 'f', out);
        return out;
    }
    for (const ResourceField& field : fields) {
        const FieldShape shape = field_shape(field.type);
        const std::uint64_t end = std::uint64_t(field.offset) + std::uint64_t(shape.components) * 4;
        if (shape.components == 0 || end > stride) continue;   // never past the element
        for (std::uint32_t c = 0; c < shape.components; ++c) read_scalar(element + field.offset + 4 * c, shape.kind, out);
    }
    return out;
}

// The inline piece that a program pins, or null when the piece lives in the
// library or the workspace.
const spec::Piece* inline_piece(const spec::Spec& s, const spec::PieceRef& ref) {
    if (ref.from != spec::PieceFrom::inline_piece) return nullptr;
    for (const spec::Piece& p : s.pieces)
        if (p.name == ref.name && p.version == ref.version) return &p;
    return nullptr;
}

}  // namespace

CommandOutput resource_command(const CommandContext& context, const spec::Spec& spec, const StepHistory& history) {
    const json& args = context.invocation.args;
    const std::string id = args.value("id", std::string());
    const std::string resource = args.value("resource", std::string());
    const spec::Program* program = nullptr;
    for (const spec::Program& p : spec.programs)
        if (p.id == id) program = &p;
    if (!program)
        return CommandOutput::fail(Error(programs::kProgramNotFound, "The open spec has no program " + id + ".", "args.id"));
    if (!program->piece)
        return CommandOutput::fail(Error(programs::kProgramUnknownResource,
                                         "The program " + id +
                                             " is not an instance of a piece, thus it publishes no resource.",
                                         "args.id"));
    if (const spec::Piece* piece = inline_piece(spec, *program->piece)) {
        const bool published = std::any_of(piece->resources.begin(), piece->resources.end(),
                                           [&](const spec::Resource& r) { return r.id == resource && r.publish; });
        if (!published)
            return CommandOutput::fail(Error(programs::kProgramUnknownResource,
                                             "The piece " + piece->name + " of the program " + id +
                                                 " publishes no resource " + resource + ".",
                                             "args.resource"));
    }
    const std::string key = id + "/" + resource;
    const std::int64_t step = args.contains("step") ? args.at("step").get<std::int64_t>() : history.cursor().step;
    const std::uint64_t first = args.value("first", std::uint64_t(0));
    const std::uint64_t want = std::clamp<std::uint64_t>(args.value("count", kReadDefault), 1, kReadMax);

    json kept = {{"versions", 0}, {"first", nullptr}, {"last", nullptr}};
    std::vector<ResourceField> fields;
    for (const KeptResource& k : history.kept_resources()) {
        if (k.key != key) continue;
        kept["versions"] = k.versions;
        if (k.versions > 0) {
            kept["first"] = k.first;
            kept["last"] = k.last;
        }
        fields = k.fields;
    }
    json out = {{"id", id}, {"resource", resource}, {"step", nullptr}, {"count", 0}, {"elements", json::array()},
                {"kept", kept}};
    const std::optional<ResourceVersion> version = history.resource_at(key, step);
    if (!version || !version->bytes) return CommandOutput::unchanged(std::move(out));
    out["step"] = version->step;
    out["count"] = version->count;
    const std::vector<std::uint8_t>& bytes = *version->bytes;
    const std::uint64_t stride = version->stride;
    const std::uint64_t valid = version->count;
    for (std::uint64_t i = first; stride > 0 && i < valid && i - first < want; ++i) {
        const std::uint64_t at = i * stride;   // i < 2^32 and stride <= 256: no wrap
        if (at + stride > bytes.size()) break;
        out["elements"].push_back(read_element(bytes.data() + at, (std::uint32_t)stride, fields));
    }
    return CommandOutput::unchanged(std::move(out));
}

}  // namespace caliper_builder
