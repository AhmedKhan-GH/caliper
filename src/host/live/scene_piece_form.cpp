#include "live/scene_piece_form.h"

#include <cstring>
#include <algorithm>
#include <cmath>

namespace caliper_live {

namespace wire = caliper_net::wire;

namespace {
// At most this many versions wait for their update: every batch of a scene times the resources of a
// piece, so that a producer that never sends the updates cannot grow the inbox without bound.
constexpr std::size_t kHeldMax = std::size_t(CALIPER_NET_MAX_BATCHES) * CALIPER_NET_MAX_RESOURCES;

const wire::Section* paired(const wire::SceneUpdate& u, const char* id) {
    for (const wire::Section& s : u.paired)
        if (s.name == id) return &s;
    return nullptr;
}

bool has(const wire::SceneUpdate& u, const char* name) {
    for (const std::string& n : u.names)
        if (n == name) return true;
    return false;
}
}  // namespace

bool ScenePieceInbox::take(const wire::Resource& r, std::shared_ptr<const std::vector<uint8_t>> bytes) {
    if (!wire::scene_program_id(r.program)) return false;
    if (held_.size() >= kHeldMax && !held_.count(r.program + "/" + r.resource)) held_.clear();
    Held& h = held_[r.program + "/" + r.resource];
    h.count = r.count;
    h.stride = r.stride;
    const bool rgba8 = !r.format || *r.format == CALIPER_NET_TEXTURE_RGBA8;
    h.uint_texels = r.format && *r.format == CALIPER_NET_TEXTURE_R32UINT;
    h.width = r.width && (rgba8 || h.uint_texels) ? *r.width : 0;
    h.height = r.height && (rgba8 || h.uint_texels) ? *r.height : 0;
    h.bytes = std::move(bytes);
    return true;
}

std::shared_ptr<const void> ScenePieceInbox::pair(wire::SceneUpdate& u, std::shared_ptr<const void> frame,
                                                  std::string* why) {
    struct Owner {
        std::shared_ptr<const void> frame;
        std::vector<std::shared_ptr<const std::vector<uint8_t>>> bytes;
    };
    auto owner = std::make_shared<Owner>();
    owner->frame = std::move(frame);
    std::vector<wire::Section> out;
    for (const auto& kv : u.resources.members()) {
        const auto it = held_.find(*u.program + "/" + kv.first);
        uint64_t count = 0;
        kv.second.get_u64(&count);
        if (it == held_.end() || it->second.count != count) {
            if (why)
                *why = "the resource " + kv.first + " of " + *u.program + (it == held_.end() ? " did not arrive" : " has another count");
            return nullptr;
        }
        const Held& h = it->second;
        wire::Section s;
        s.name = kv.first;
        s.dtype = CALIPER_NET_DTYPE_U8;
        if (h.width && h.uint_texels) {
            s.dtype = CALIPER_NET_DTYPE_I32;
            s.shape = {int64_t(h.height), int64_t(h.width)};
        } else if (h.width)   // an RGBA texture has four bytes per texel
            s.shape = {int64_t(h.height), int64_t(h.width), 4};
        else
            s.shape = {int64_t(h.count), int64_t(h.stride)};
        s.data = h.bytes ? h.bytes->data() : nullptr;
        s.size = h.bytes ? h.bytes->size() : 0;
        out.push_back(s);
        owner->bytes.push_back(h.bytes);
    }
    for (const auto& kv : u.resources.members()) held_.erase(*u.program + "/" + kv.first);
    u.paired = std::move(out);
    return owner;
}

bool scene_piece_arrays(const wire::SceneUpdate& u, uint32_t dims, std::vector<wire::Section>* out,
                        std::vector<std::vector<uint8_t>>* owned, std::string* why) {
    const std::string name = u.piece.is_object() && u.piece.find("name") ? u.piece.find("name")->as_string() : "";
    const caliper_net::Json* version = u.piece.is_object() ? u.piece.find("version") : nullptr;
    if (!version || version->as_double() != 1.0) return *why = "a piece " + name + " of another version", false;
    const bool image_piece = name == "image" || name == "depth" || name == "segmentation";
    const int part = wire::scene_part_index(u.part);
    if (part < 0 || u.part != (image_piece ? "image" : name)) return *why = "piece and part differ", false;
    const char* const* names = wire::scene_part_sections(part);
    int last = -1;
    if (u.names.empty()) return *why = "piece arrays are empty", false;
    for (const auto& field : u.names) {
        int at = -1;
        for (int i = 0; names[i]; ++i) if (field == names[i]) at = i;
        if (at <= last || (last < 0 && at != 0)) return *why = "piece arrays do not match the part", false;
        last = at;
    }
    if (name == "points") {
        const wire::Section* r = paired(u, "points");
        if (!r || r->shape.size() != 2 || r->shape[0] < 0 || r->shape[0] > CALIPER_NET_MAX_BATCH_POINTS ||
            (r->size && !r->data) || r->shape[1] != kScenePointStride ||
            r->size != std::size_t(r->shape[0]) * kScenePointStride || (dims != 2 && dims != 3))
            return *why = "points@1's resource points of another layout", false;
        const std::size_t n = std::size_t(r->shape[0]);
        auto add = [&](const char* nm, int64_t dtype, std::vector<int64_t> shape, std::size_t bytes) {
            owned->emplace_back(bytes);
            out->push_back(wire::Section::make(nm, dtype, std::move(shape), owned->back().data(), bytes));
            return owned->back().data();
        };
        owned->reserve(5);
        uint8_t* pos = add("positions", CALIPER_NET_DTYPE_F32, {int64_t(n), int64_t(dims)}, n * dims * 4);
        uint8_t *colors = nullptr, *values = nullptr, *ids = nullptr, *radii = nullptr;
        if (has(u, "colors")) colors = add("colors", CALIPER_NET_DTYPE_U8, {int64_t(n), 4}, n * 4);
        if (has(u, "values")) values = add("values", CALIPER_NET_DTYPE_F32, {int64_t(n)}, n * 4);
        if (has(u, "class_ids")) ids = add("class_ids", CALIPER_NET_DTYPE_I32, {int64_t(n)}, n * 4);
        if (has(u, "radii")) radii = add("radii", CALIPER_NET_DTYPE_F32, {int64_t(n)}, n * 4);
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t* p = r->data + i * kScenePointStride;
            std::memcpy(pos + i * dims * 4, p, dims * 4);
            if (colors) std::memcpy(colors + i * 4, p + 16, 4);
            if (values) std::memcpy(values + i * 4, p + 16, 4);
            if (ids) {
                const wire::Section* source = paired(u, "source_class_ids");
                if (has(u, "colors") || has(u, "values")) {
                    if (!source || source->shape != std::vector<int64_t>{int64_t(n), 4} || source->size != n * 4)
                        return *why = "point class ids do not match their rows", false;
                    std::memcpy(ids + i * 4, source->data + i * 4, 4);
                } else std::memcpy(ids + i * 4, p + 16, 4);
            }
            if (radii) std::memcpy(radii + i * 4, p + 12, 4);
        }
        return true;
    }
    if (name == "image") {
        // image@1: "pixels" the texels of an rgba8unorm texture of the image's size, one packed RGBA a
        // pixel, and "info" the width, the height and the channels.
        const wire::Section* px = paired(u, "pixels");
        const wire::Section* info = paired(u, "info");
        if (!px || !info || px->shape.size() != 3 || px->shape[2] != 4 || info->shape.size() != 2 ||
            info->shape[0] != 1 || info->shape[1] != kSceneImageInfoStride || info->size != kSceneImageInfoStride)
            return *why = "image@1's resources of another layout", false;
        uint32_t size[4];
        std::memcpy(size, info->data, 16);
        const uint64_t w = size[0], h = size[1], n = w * h;
        if (w == 0 || h == 0 || w > CALIPER_NET_MAX_IMAGE_SIDE || h > CALIPER_NET_MAX_IMAGE_SIDE ||
            (size[2] != 1 && size[2] != 4) || uint64_t(px->shape[0]) != h || uint64_t(px->shape[1]) != w ||
            px->size != n * 4)
            return *why = "image@1's pixels do not fit the size in its info", false;
        if (size[2] == 4) {
            owned->emplace_back(px->data, px->data + px->size);
            out->push_back(wire::Section::make("pixels", CALIPER_NET_DTYPE_U8, {int64_t(h), int64_t(w), 4},
                                               owned->back().data(), px->size));
        } else {
            owned->emplace_back(std::size_t(n));
            for (uint64_t i = 0; i < n; ++i) owned->back()[std::size_t(i)] = px->data[i * 4];
            out->push_back(wire::Section::make("pixels", CALIPER_NET_DTYPE_U8, {int64_t(h), int64_t(w)},
                                               owned->back().data(), std::size_t(n)));
        }
        return true;
    }
    // Source buffers preserve data that expanded library resources cannot encode, such as an
    // unused indexed-line vertex. They are bounded host inputs, never draw graph resources.
    auto buffer = [&](const char* id, uint32_t stride, uint64_t limit) -> const wire::Section* {
        const auto* r = paired(u, id);
        if (!r || r->shape.size() != 2 || r->shape[0] < 0 || r->shape[1] != stride ||
            uint64_t(r->shape[0]) > limit || r->size != uint64_t(r->shape[0]) * stride ||
            (r->size && !r->data)) return nullptr;
        return r;
    };
    auto add = [&](const char* id, int64_t dtype, std::vector<int64_t> shape, size_t bytes) {
        owned->emplace_back(bytes);
        out->push_back(wire::Section::make(id, dtype, std::move(shape), owned->back().data(), bytes));
        return owned->back().data();
    };
    auto copy = [&](const char* id, int64_t dtype, std::vector<int64_t> shape, const wire::Section& r) {
        auto* p = add(id, dtype, std::move(shape), r.size);
        if (r.size) std::memcpy(p, r.data, r.size);
    };
    if (dims != 2 && dims != 3) return *why = "scene rows need two or three dimensions", false;
    if (name == "lines") {
        const auto* segments = buffer("segments", 32, CALIPER_NET_MAX_BATCH_POINTS);
        const auto* info = buffer("info", 12, 1);
        const auto* picks = buffer("picks", 8, CALIPER_NET_MAX_BATCH_POINTS);
        if (!segments || !picks || picks->shape[0] != segments->shape[0] || !info || info->shape[0] != 1)
            return *why = "lines@1 resources have another layout", false;
        const auto* positions = buffer("source_positions", dims * 4, CALIPER_NET_MAX_BATCH_POINTS);
        if (!positions) return *why = "lines@1 has no valid source positions", false;
        const int64_t n = positions->shape[0];
        const auto* mode = u.params.find("mode");
        const std::string m = mode ? mode->as_string() : "segments";
        const auto* per = u.params.find("color_per");
        const std::string p = per ? per->as_string() : "vertex";
        const auto* lengths = buffer("source_strip_lengths", 4, CALIPER_NET_MAX_BATCH_POINTS);
        const auto* indices = buffer("source_indices", 8, CALIPER_NET_MAX_BATCH_POINTS);
        int64_t color_rows = n;
        if (m == "strips") {
            if (!lengths) return *why = "line strips need source lengths", false;
            uint64_t total = 0;
            for (int64_t i = 0; i < lengths->shape[0]; ++i) {
                int32_t v; std::memcpy(&v, lengths->data + i * 4, 4);
                if (v < 1) return *why = "a strip length must be positive", false;
                total += uint32_t(v);
            }
            if (total != uint64_t(n) || segments->shape[0] != std::max<int64_t>(0, n - 1))
                return *why = "line strips do not match their source rows", false;
            if (p == "strip") color_rows = lengths->shape[0];
        } else if (m == "edges") {
            if (!indices || segments->shape[0] != (n ? indices->shape[0] : 0))
                return *why = "line edges do not match their source indices", false;
            if (p == "edge") color_rows = indices->shape[0];
        } else if (m != "segments" || n % 2 || segments->shape[0] != n / 2) {
            return *why = "line segments do not match their source rows", false;
        }
        for (const auto& array : u.names) {
            const uint32_t stride = array == "positions" ? dims * 4 : array == "indices" ? 8 : 4;
            const auto* r = buffer(("source_" + array).c_str(), stride, CALIPER_NET_MAX_BATCH_POINTS);
            if (!r) return *why = "a line source resource has another layout", false;
            const int64_t rows = r->shape[0];
            if ((array == "colors" || array == "values") && rows != color_rows && rows != 1)
                return *why = "line colors do not match their source rows", false;
            std::vector<int64_t> shape{rows};
            if (array == "positions") shape.push_back(dims);
            if (array == "indices") shape.push_back(2);
            if (array == "colors") shape.push_back(4);
            const int64_t dtype = array == "colors" ? CALIPER_NET_DTYPE_U8 :
                (array == "indices" || array == "strip_lengths") ? CALIPER_NET_DTYPE_I32 : CALIPER_NET_DTYPE_F32;
            copy(array.c_str(), dtype, shape, *r);
        }
        return true;
    }
    if (name == "arrows" || name == "boxes" || name == "mesh") {
        const bool arrow = name == "arrows", box = name == "boxes";
        const uint32_t stride = box ? 64 : 32;
        const auto* r = buffer(arrow ? "arrows" : box ? "boxes" : "vertices", stride,
                               arrow || box ? CALIPER_NET_MAX_BATCH_ITEMS : CALIPER_NET_MAX_BATCH_POINTS);
        if (!r) return *why = name + "@1 rows have another layout", false;
        const int64_t n = r->shape[0];
        auto column = [&](const char* id, int64_t dtype, uint32_t cols, uint32_t offset) {
            std::vector<int64_t> shape{n};
            if (cols > 1) shape.push_back(cols);
            uint8_t* to = add(id, dtype, shape, size_t(n) * cols * 4);
            for (int64_t i = 0; i < n; ++i) std::memcpy(to + i * cols * 4, r->data + i * stride + offset, cols * 4);
        };
        column(arrow ? "origins" : box ? "centers" : "positions", CALIPER_NET_DTYPE_F32, dims, 0);
        if (arrow) column("vectors", CALIPER_NET_DTYPE_F32, dims, 16);
        if (box) column("half_sizes", CALIPER_NET_DTYPE_F32, dims, 16);
        if (has(u, "rotations")) column("rotations", CALIPER_NET_DTYPE_F32, 4, 32);
        if (has(u, "normals")) column("normals", CALIPER_NET_DTYPE_F32, 3, 16);

        if (has(u, "values")) column("values", CALIPER_NET_DTYPE_F32, 1, 12);
        if (has(u, "colors")) {
            uint8_t* to = add("colors", CALIPER_NET_DTYPE_U8, {n, 4}, size_t(n) * 4);
            const uint32_t off = arrow ? 12 : box ? 48 : 28;
            for (int64_t i = 0; i < n; ++i) std::memcpy(to + i * 4, r->data + i * stride + off, 4);
        }
        if (has(u, "class_ids")) {
            const auto* ids = buffer("source_class_ids", 4, CALIPER_NET_MAX_BATCH_ITEMS);
            if (!box || !ids || ids->shape[0] != n) return *why = "box class ids do not match their rows", false;
            copy("class_ids", CALIPER_NET_DTYPE_I32, {n}, *ids);
        }
        if (!arrow && !box) {
            const auto* attributes = buffer("attributes", 12, CALIPER_NET_MAX_BATCH_POINTS);
            const auto* info = buffer("info", 4, 1);
            if (!attributes || attributes->shape[0] != n || !info || info->shape[0] != 1)
                return *why = "mesh attributes do not match their vertices", false;
            if (has(u, "uvs")) {
                auto* uv = add("uvs", CALIPER_NET_DTYPE_F32, {n, 2}, size_t(n) * 8);
                for (int64_t i = 0; i < n; ++i) std::memcpy(uv + i * 8, attributes->data + i * 12 + 4, 8);
            }
            const auto* faces = buffer("faces", 4, uint64_t(CALIPER_NET_MAX_BATCH_POINTS) * 3);
            if (!faces || faces->shape[0] % 3) return *why = "mesh faces have another layout", false;
            if (has(u, "indices")) copy("indices", CALIPER_NET_DTYPE_I32, {faces->shape[0] / 3, 3}, *faces);
            else if (faces->shape[0] != n || n % 3) return *why = "mesh faces do not match their vertices", false;
            if (has(u, "texture")) {
                const auto* tex = paired(u, "tex");
                if (!tex || tex->shape.size() != 3 || tex->shape[2] != 4 || tex->shape[0] < 1 || tex->shape[1] < 1 ||
                    tex->shape[0] > CALIPER_NET_MAX_IMAGE_SIDE || tex->shape[1] > CALIPER_NET_MAX_IMAGE_SIDE ||
                    uint64_t(tex->shape[0]) * tex->shape[1] * 4 != tex->size)
                    return *why = "mesh texture has another layout", false;
                copy("texture", CALIPER_NET_DTYPE_U8, tex->shape, *tex);
            }
        }
        return true;
    }
    if (name == "depth" || name == "segmentation") {
        const bool depth = name == "depth";
        const auto* pixels = paired(u, "pixels");
        const auto* info = buffer("info", depth ? 96 : 80, 1);
        if (!pixels || !info || info->shape[0] != 1 || pixels->dtype != CALIPER_NET_DTYPE_I32 || pixels->shape.size() != 2)
            return *why = name + " resources have another layout", false;
        uint32_t size[4]; std::memcpy(size, info->data, 16);
        const uint64_t n = uint64_t(size[0]) * size[1];
        if (!size[0] || !size[1] || size[0] > CALIPER_NET_MAX_IMAGE_SIDE || size[1] > CALIPER_NET_MAX_IMAGE_SIDE ||
            pixels->shape != std::vector<int64_t>{int64_t(size[1]), int64_t(size[0])} || pixels->size != n * 4 ||
            !pixels->data || size[2] != 1 || (depth ? size[3] != 4 : size[3] != 1 && size[3] != 2))
            return *why = name + " image size does not match its pixels", false;
        if (depth) {
            // Library pixels already hold depth in scene units. The caller sets scale to one.
            copy("pixels", CALIPER_NET_DTYPE_F32, {int64_t(size[1]), int64_t(size[0])}, *pixels);
            return true;
        }
        const int64_t dtype = size[3] == 2 ? CALIPER_NET_DTYPE_U16 : CALIPER_NET_DTYPE_U8;
        auto* to = add("pixels", dtype, {int64_t(size[1]), int64_t(size[0])}, size_t(n) * size[3]);
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t v; std::memcpy(&v, pixels->data + i * 4, 4);
            if (v > (size[3] == 1 ? 255u : 65535u)) return *why = "mask class id exceeds its dtype", false;
            if (size[3] == 2) { uint16_t small = uint16_t(v); std::memcpy(to + i * 2, &small, 2); }
            else to[i] = uint8_t(v);
        }
        return true;
    }
    return *why = "no piece form of " + u.part + " with the piece " + name, false;
}

}  // namespace caliper_live
