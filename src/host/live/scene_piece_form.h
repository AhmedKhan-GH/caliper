#pragma once
// The piece form of SCENE_UPDATE on the receiving side (wire protocol 3.2, capability "scene_pieces").
//
// A producer sends an entity part's arrays as the RESOURCE versions of a scene program "scene.<n>", an
// instance of the part's library piece, and right after them the SCENE_UPDATE in the piece form that names
// them with their counts. ScenePieceInbox keeps the newest version of each scene program's resource that a
// connection received, apart from the resources of spec programs, and pairs it with the update that names
// it (network thread). scene_piece_arrays() then rebuilds the part's arrays from the resource bytes on the
// frame thread, where the dimension of the rows is known, so that the scene view stores the same batch as
// from the per-shape part; the scene view's pieces draw the received bytes themselves.
#include "net/wire.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace caliper_live {

class ScenePieceInbox {
public:
    // A whole RESOURCE version (its parts put together) of a scene program; false, and nothing kept, for
    // the resource of any other program. The newest version of each resource replaces an older one.
    bool take(const caliper_net::wire::Resource& r, std::shared_ptr<const std::vector<uint8_t>> bytes);
    // A piece-form update: its versions, each with the count that `resources` names, move into u.paired,
    // and the result keeps their bytes and `frame` alive; null, with *why, when a version is missing or
    // its count differs, which drops the update.
    std::shared_ptr<const void> pair(caliper_net::wire::SceneUpdate& u, std::shared_ptr<const void> frame,
                                     std::string* why);
    void clear() { held_.clear(); }
    std::size_t held() const { return held_.size(); }

private:
    struct Held {
        uint32_t count = 0, stride = 0;
        uint32_t width = 0, height = 0;   // a texture's size, 0 for a buffer
        bool uint_texels = false;
        std::shared_ptr<const std::vector<uint8_t>> bytes;
    };
    std::map<std::string, Held> held_;   // by program/resource
};

// The arrays of the classic form of a piece-form update `u` (whose `paired` holds its versions) as
// sections over buffers that `owned` keeps: the rows in `dims` dimensions (2 or 3). False, with *why, for
// a piece or a resource layout that this viewer does not read.
bool scene_piece_arrays(const caliper_net::wire::SceneUpdate& u, uint32_t dims, std::vector<caliper_net::wire::Section>* out,
                        std::vector<std::vector<uint8_t>>* owned, std::string* why);

// The layout of each piece's resource that the piece form carries: points@1's "points", 32 bytes a point.
inline constexpr uint32_t kScenePointStride = 32;
// image@1's "info": ImageInfo {size: vec4<u32>, c0, c1, c2, c3: vec4<f32>}, 80 bytes.
inline constexpr uint32_t kSceneImageInfoStride = 80;

}  // namespace caliper_live
