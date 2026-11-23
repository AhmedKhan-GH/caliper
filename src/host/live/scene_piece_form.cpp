#include "live/scene_piece_form.h"

#include <cstring>

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
    if (name == "points") {
        const wire::Section* r = paired(u, "points");
        if (!r || r->shape.size() != 2 || r->shape[1] != kScenePointStride ||
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
            if (ids) std::memcpy(ids + i * 4, p + 16, 4);
            if (radii) std::memcpy(radii + i * 4, p + 12, 4);
        }
        return true;
    }
    if (name == "image") {
        // image@1: "pixels" one packed RGBA a pixel, "info" the width, the height and the channels.
        const wire::Section* px = paired(u, "pixels");
        const wire::Section* info = paired(u, "info");
        if (!px || !info || px->shape.size() != 2 || px->shape[1] != 4 || info->shape.size() != 2 ||
            info->shape[0] != 1 || info->shape[1] != kSceneImageInfoStride || info->size != kSceneImageInfoStride)
            return *why = "image@1's resources of another layout", false;
        uint32_t size[4];
        std::memcpy(size, info->data, 16);
        const uint64_t w = size[0], h = size[1], n = w * h;
        if (w == 0 || h == 0 || w > CALIPER_NET_MAX_IMAGE_SIDE || h > CALIPER_NET_MAX_IMAGE_SIDE ||
            (size[2] != 1 && size[2] != 4) || uint64_t(px->shape[0]) != n || px->size != n * 4)
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
    return *why = "no piece form of " + u.part + " with the piece " + name, false;
}

}  // namespace caliper_live
