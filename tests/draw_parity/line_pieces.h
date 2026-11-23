// line_pieces.h: the library pieces lines@1, arrows@1 and path@1 in the form that PROGRAM carries
// them, for the pixel comparison of the app and the browser module (line_fixtures.h).
//
// The comparison's targets have no TOML reader, thus each piece comes here as its canonical
// manifest and its hash, and the files come from the build's library (library_files()). The
// matches of the pieces (tests/lines_match/, tests/arrows_match/, tests/path_match/) check both
// against the library's piece.toml, so that a change of a piece shows here at once.
#pragma once

#include "draw/draw_runtime.h"
#include "net/json.h"

#include <caliper_builder/programs/pieces.h>

#include <map>
#include <string>
#include <string_view>

namespace draw_parity {

// The canonical manifest (RFC 8785 JSON) and the hash of each library piece.
inline constexpr std::string_view kLinesManifest = R"json({"derive":{"memory_mb":256,"passes":[{"entry":"sum_chunks","workgroups":[4,1,1],"writes":["scratch"]},{"entry":"start_chunks","writes":["scratch"]},{"entry":"end_strips","workgroups":[4,1,1],"writes":["scratch"]},{"entry":"build_segments","workgroups":[8192,1,1],"writes":["segments"]},{"entry":"scan_values","workgroups":[16,1,1],"writes":["scratch"]},{"entry":"finish_info","writes":["info"]}],"time_ms":50},"description":"Line segments, strips or index-pair edges of a width in pixels, colored by one color, by a value through a colormap, or by a packed RGBA, at each vertex, strip or edge","draw":{"memory_mb":96,"passes":[{"blend":"alpha","depth":"write","fragment":"line_fs","instances":"segments","kind":"render","pick":true,"vertex":"opaque_vs","vertices":6},{"blend":"alpha","depth":"test","fragment":"line_fs","instances":"segments","kind":"render","pick":true,"vertex":"translucent_vs","vertices":6}],"pick":{"fields":["index_a","index_b"],"resource":"segments"}},"imports":["pick@1","camera@1"],"inputs":[{"kind":"tensor","name":"positions","rank_max":2,"rank_min":2,"title":"Positions (V, 3), or (V, 2) with z 0"},{"kind":"tensor","name":"values","rank_max":1,"title":"Values of the vertices, strips or edges, or one for all: a value for the colormap, or a packed RGBA in its bits"},{"dtype":"i32","kind":"tensor","name":"topology","rank_max":2,"title":"Strip lengths (S) for strips, index pairs (E, 2) for edges; segments read none"}],"name":"lines","params":[{"default":2,"max":128,"min":0,"name":"width","title":"Width in Pixels"},{"max":2,"min":0,"name":"mode","title":"Mode (0 segments of vertices 2k and 2k + 1, 1 strips, 2 edges)","type":"u32"},{"max":2,"min":0,"name":"color_mode","title":"Color (0 one color, 1 colormap, 2 packed RGBA)","type":"u32"},{"default":4292664550,"name":"color","title":"One Color as Packed RGBA","type":"u32"},{"max":1,"min":0,"name":"color_per","title":"Values Belong To (0 vertices, 1 strips or edges)","type":"u32"},{"max":3,"min":0,"name":"colormap","title":"Colormap (0 viridis, 1 magma, 2 rdbu, 3 greys)","type":"u32"},{"default":3,"max":3,"min":0,"name":"range","title":"Ends From the Values (0 none, 1 start, 2 end, 3 both)","type":"u32"},{"name":"vmin","title":"Value at the Colormap's Start"},{"default":1,"name":"vmax","title":"Value at the Colormap's End"},{"max":8,"min":-8,"name":"depth_offset","title":"Depth Offset Toward the Camera","type":"i32"}],"resources":[{"count":2097152,"counted":true,"fields":["ax: f32","ay: f32","az: f32","color_a: u32","bx: f32","by: f32","bz: f32","color_b: u32","index_a: u32","index_b: u32"],"id":"segments","kind":"buffer","struct":"Segment"},{"count":1,"fields":["lo: f32","hi: f32","flags: u32"],"id":"info","kind":"buffer","struct":"Info"},{"count":1061888,"element":"u32","id":"scratch","kind":"buffer","publish":false}],"version":1})json";
inline constexpr std::string_view kLinesHash = "sha256:cc0438cac5ba1061207bf686e475c02d4a63f6722f87262c7721db6ee2146502";
inline constexpr std::string_view kArrowsManifest = R"json({"derive":{"memory_mb":128,"passes":[{"entry":"build_arrows","workgroups":[4096,1,1],"writes":["arrows"]},{"entry":"scan_values","workgroups":[16,1,1],"writes":["scratch"]},{"entry":"finish_info","writes":["info"]}],"time_ms":50},"description":"Arrows from origins along vectors, a shaft of a width in pixels and a flat head, colored by one color, by a value through a colormap, or by a packed RGBA","draw":{"memory_mb":64,"passes":[{"blend":"alpha","depth":"write","fragment":"arrow_fs","instances":"arrows","kind":"render","pick":true,"vertex":"opaque_vs","vertices":9},{"blend":"alpha","depth":"test","fragment":"arrow_fs","instances":"arrows","kind":"render","pick":true,"vertex":"translucent_vs","vertices":9}],"pick":{"fields":["index"],"resource":"arrows"}},"imports":["pick@1","camera@1"],"inputs":[{"kind":"tensor","name":"origins","rank_max":2,"rank_min":2,"title":"Origins (N, 3), or (N, 2) with z 0"},{"kind":"tensor","name":"vectors","rank_max":2,"rank_min":2,"title":"Vectors (N, 3), or (N, 2) with z 0: each arrow's tip is its origin plus its vector"},{"kind":"tensor","name":"values","rank_max":1,"title":"Values of the arrows, or one for all: a value for the colormap, or a packed RGBA in its bits"}],"name":"arrows","params":[{"default":2,"max":128,"min":0,"name":"width","title":"Width in Pixels"},{"max":2,"min":0,"name":"color_mode","title":"Color (0 one color, 1 colormap, 2 packed RGBA)","type":"u32"},{"default":4292664550,"name":"color","title":"One Color as Packed RGBA","type":"u32"},{"max":3,"min":0,"name":"colormap","title":"Colormap (0 viridis, 1 magma, 2 rdbu, 3 greys)","type":"u32"},{"default":3,"max":3,"min":0,"name":"range","title":"Ends From the Values (0 none, 1 start, 2 end, 3 both)","type":"u32"},{"name":"vmin","title":"Value at the Colormap's Start"},{"default":1,"name":"vmax","title":"Value at the Colormap's End"},{"max":8,"min":-8,"name":"depth_offset","title":"Depth Offset Toward the Camera","type":"i32"}],"resources":[{"count":1048576,"counted":true,"fields":["ox: f32","oy: f32","oz: f32","color: u32","vx: f32","vy: f32","vz: f32","index: u32"],"id":"arrows","kind":"buffer","struct":"Arrow"},{"count":1,"fields":["lo: f32","hi: f32","flags: u32"],"id":"info","kind":"buffer","struct":"Info"},{"count":12288,"element":"u32","id":"scratch","kind":"buffer","publish":false}],"version":1})json";
inline constexpr std::string_view kArrowsHash = "sha256:8b7cd49b75a2293a05d8bf65a7c519a8efea3be942813654fefc91ab84c39313";
inline constexpr std::string_view kPathManifest = R"json({"derive":{"memory_mb":256,"passes":[{"entry":"keep_rows","workgroups":[4096,1,1],"writes":["vertices"]}],"time_ms":50},"description":"A path through the rows of a tensor, every s-th row at most max_points of them, as a line strip of a width in pixels colored by the row through a colormap or by one color","draw":{"memory_mb":64,"passes":[{"blend":"alpha","depth":"write","fragment":"path_fs","instances":"vertices","kind":"render","pick":true,"vertex":"opaque_vs","vertices":6},{"blend":"alpha","depth":"test","fragment":"path_fs","instances":"vertices","kind":"render","pick":true,"vertex":"translucent_vs","vertices":6}],"pick":{"fields":["row","value","x","y","z"],"resource":"vertices"}},"imports":["pick@1","camera@1"],"inputs":[{"kind":"tensor","name":"rows","rank_max":2,"rank_min":2,"title":"Rows (R, D): each kept row is a vertex at its columns"}],"name":"path","params":[{"max":65535,"min":0,"name":"x","title":"Column of x","type":"i32"},{"default":1,"max":65535,"min":0,"name":"y","title":"Column of y","type":"i32"},{"default":2,"max":65535,"min":-1,"name":"z","title":"Column of z, or -1 or a column past the rows for z 0","type":"i32"},{"default":65536,"max":1048576,"min":2,"name":"max_points","title":"Most Vertices, at most 1,048,576 drawn","type":"u32"},{"default":2,"max":128,"min":0,"name":"width","title":"Width in Pixels"},{"default":1,"max":1,"min":0,"name":"color_mode","title":"Color (0 one color, 1 the row through the colormap)","type":"u32"},{"default":4292664550,"name":"color","title":"One Color as Packed RGBA","type":"u32"},{"max":3,"min":0,"name":"colormap","title":"Colormap (0 viridis, 1 magma, 2 rdbu, 3 greys)","type":"u32"},{"default":3,"max":3,"min":0,"name":"range","title":"Ends From the Rows (0 none, 1 start, 2 end, 3 both)","type":"u32"},{"name":"vmin","title":"Value at the Colormap's Start"},{"default":1,"name":"vmax","title":"Value at the Colormap's End"},{"default":1,"max":8,"min":-8,"name":"depth_offset","title":"Depth Offset Toward the Camera","type":"i32"}],"resources":[{"count":1048576,"counted":true,"fields":["x: f32","y: f32","z: f32","value: f32","row: u32"],"id":"vertices","kind":"buffer","struct":"Vertex"}],"version":1})json";
inline constexpr std::string_view kPathHash = "sha256:f21d1a641fdd4d3907fb52b1bd25cae1c767b5ee373f908e9922aaa452a553c7";

// The library piece `name`@1 from its canonical manifest, its hash and the build's files, decoded
// as a viewer decodes PROGRAM, which checks the hash.
inline bool line_piece(const std::string& name, std::string_view manifest, std::string_view hash,
                       caliper_builder::spec::Piece* piece, std::string* why) {
    namespace cp = caliper_builder::programs;
    std::map<std::string, std::string> files;
    const std::string prefix = "pieces/" + name + "/1/";
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string_view path = f.path;
        if (path.substr(0, prefix.size()) == prefix) files[std::string(path.substr(prefix.size()))] = std::string(f.text);
    }
    caliper_net::Json m;
    std::string err;
    if (!caliper_net::Json::parse(manifest.data(), manifest.size(), &m, &err))
        return (*why = name + ": the manifest does not parse: " + err, false);
    caliper_net::Json decl = caliper_net::Json::object();
    decl.set("id", name);
    decl.set("panel", "scene");
    caliper_net::Json pin = caliper_net::Json::object();
    pin.set("name", name);
    pin.set("version", std::uint64_t(1));
    pin.set("hash", std::string(hash));
    decl.set("piece", pin);
    decl.set("manifest", m);
    caliper_draw::Instance instance;
    const caliper_draw::Verdict v = caliper_draw::DrawRuntime::decode(
        decl.dump(), files["common.wgsl"], files["draw.wgsl"], piece, &instance,
        caliper_draw::DrawChecks::pieces(), files["derive.wgsl"]);
    if (!v.ok()) return (*why = name + ": " + v.code + " " + v.message, false);
    return true;
}

}  // namespace draw_parity
