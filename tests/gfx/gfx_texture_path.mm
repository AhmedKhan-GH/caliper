// gfx_texture_path.mm: the texture path's rows of the graphics suite (caliper_gfx_texture_tests).
//
// Each case puts one tensor into a texture through one of the texture path's clients, as an applet,
// a host or a training script does, and reads the texture back: the tensor bridge
// (texture_from_tensor_mapped, texture_from_tensor, update_texture, alloc_shared, and
// import_allocation with update_texture_from_alloc), the embed views (caliper_core_view_tensor) and
// the heatmap and image live views (caliper_core_live_apply). The cases cover lattices of every
// size class, from one texel to 1024 x 768 and long thin ones, against the compute pass's 16 x 16
// and 16 x 8 workgroups; every colormap; fixed ranges and ranges taken from statistics; increasing,
// decreasing, degenerate and infinite ranges, ranges near the subnormals and ranges whose span
// overflows; values at every colormap level boundary of the range and beside it, NaN of each kind,
// both infinities, both zeros, subnormals and the largest floats; CPU tensors and MTLBuffers in
// shared and in private storage, read directly or through an imported allocation at a byte offset;
// and images of one, three and four channels.
//
// Every case has a CPU reference: map_f32_to_rgba8 for f32 data, expand_u8_to_rgba8 for u8 data.
//
// The reference mode (the graphics suite's, design/renderer-core.md, "The rule of every step"): the
// committed record tests/gfx/references/texture_path.txt was made at the first commit of migration
// step 8 from the old path, before any code moved. A line per case holds the readback's FNV-1a
// hash, the path label, the renderer's copies and bytes for the update, and how many texels of the
// old path differ from the CPU reference (cpu_ref_differs, "none" when it equals it). The first row
// holds the texture path to it: every texel equals the CPU reference, the readback equals the
// record byte for byte wherever the record equals the CPU reference too, and the path label,
// copies and bytes are the record's. The second row compares with the old path's images. The line
// format is the one of the points view's rows:
//
//   <case> <width>x<height> fnv1a64=<16 hex digits> <key>=<value> ... info=<text to the line's end>
//
//   CALIPER_GFX_RECORD=<file>              writes every case's measured line there, sorted by case
//   CALIPER_GFX_REFERENCES=<file>          reads that file instead of the committed one
//   CALIPER_GFX_CAPTURE_DIR=<folder>       writes each readback there as texture-<case>.png, its
//                                          exact texels
//   CALIPER_GFX_TEXTURE_CASES=<substring>  runs only the cases whose names contain it
#define DOCTEST_CONFIG_IMPLEMENT
#include "../gpu_skip.h"

#include <caliper/embed.h>
#include <caliper/live_views.h>
#include <caliper/services/tensor_bridge_v1_2.h>

#include "gfx/colormap_keys.h"      // derive_keys: the level boundaries of a range
#include "host_services.h"          // services_bridge, services_renderer
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"          // TensorBridge, map_f32_to_rgba8, expand_u8_to_rgba8, colormap_lut

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <stb_image.h>
#include <stb_image_write.h>
#pragma clang diagnostic pop

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using caliper_host::TensorBridge;

// ---- The reference mode --------------------------------------------------------------------------

const char* env(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? v : nullptr;
}

uint64_t fnv1a64(const std::vector<uint8_t>& bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : bytes) {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string hex16(uint64_t v) {
    char t[17];
    std::snprintf(t, sizeof t, "%016llx", (unsigned long long)v);
    return t;
}

struct Line {
    std::string name;
    int width = 0, height = 0;
    std::string hash;
    std::vector<std::pair<std::string, std::string>> fields;
    std::string info;

    std::string field(const std::string& key) const {
        for (const auto& kv : fields)
            if (kv.first == key) return kv.second;
        return {};
    }
    std::string text() const {
        std::string s = name + " " + std::to_string(width) + "x" + std::to_string(height) + " fnv1a64=" + hash;
        for (const auto& kv : fields) s += " " + kv.first + "=" + kv.second;
        return s + " info=" + info;
    }
    static bool parse(const std::string& text, Line* out) {
        Line l;
        const size_t at = text.find(" info=");
        const std::string head = at == std::string::npos ? text : text.substr(0, at);
        if (at != std::string::npos) l.info = text.substr(at + 6);
        std::istringstream in(head);
        std::string size, tok;
        if (!(in >> l.name >> size)) return false;
        const size_t x = size.find('x');
        if (x == std::string::npos) return false;
        l.width = std::atoi(size.c_str());
        l.height = std::atoi(size.c_str() + x + 1);
        while (in >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "fnv1a64") l.hash = v;
            else l.fields.emplace_back(k, v);
        }
        *out = l;
        return true;
    }
};

class References {
public:
    References() {
        const char* other = env("CALIPER_GFX_REFERENCES");
        path_ = other ? other : std::string(CALIPER_TEST_SOURCE_ROOT) + "/tests/gfx/references/texture_path.txt";
        std::ifstream in(path_);
        std::string text;
        while (std::getline(in, text)) {
            if (text.empty() || text[0] == '#') continue;
            Line l;
            if (Line::parse(text, &l)) lines_[l.name] = l;
        }
    }
    ~References() { flush(); }
    const std::string& path() const { return path_; }
    bool recording() const { return env("CALIPER_GFX_RECORD") != nullptr; }
    const Line* find(const std::string& name) const {
        const auto it = lines_.find(name);
        return it == lines_.end() ? nullptr : &it->second;
    }
    void record(const Line& l) { recorded_[l.name] = l; }
    void flush() {
        const char* out = env("CALIPER_GFX_RECORD");
        if (!out || recorded_.empty()) return;
        std::ofstream f(out);
        f << "# The texture path's references (tests/gfx/gfx_texture_path.mm): one line per case, the readback's\n"
             "# FNV-1a hash, its path label, the renderer's copies and bytes for the update, and how many\n"
             "# texels differ from the CPU reference (cpu_ref_differs).\n";
        for (const auto& kv : recorded_) f << kv.second.text() << "\n";
        std::fprintf(stderr, "[gfx-reference] %zu lines written to %s\n", recorded_.size(), out);
        recorded_.clear();
    }

private:
    std::string path_;
    std::map<std::string, Line> lines_;
    std::map<std::string, Line> recorded_;
};

// The readback's exact texels as texture-<case>.png in CALIPER_GFX_CAPTURE_DIR.
void capture(const std::string& name, const std::vector<uint8_t>& rgba, int w, int h) {
    const char* dir = env("CALIPER_GFX_CAPTURE_DIR");
    if (dir == nullptr || rgba.size() != (size_t)w * h * 4) return;
    const std::string path = std::string(dir) + "/texture-" + name + ".png";
    CHECK_MESSAGE(stbi_write_png(path.c_str(), w, h, 4, rgba.data(), w * 4) != 0, "could not write ", path);
}

// ---- The offscreen core and its renderer's device -------------------------------------------------

struct Core {
    CaliperCore* core = nullptr;
    bool gpu = false;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    Core() {
        CaliperCoreDesc d{};
        d.struct_size = sizeof d;
        d.renderer = CALIPER_RENDERER_DEFAULT;
        core = caliper_core_create(&d);
        REQUIRE(core != nullptr);
        if (caliper_test::force_no_gpu()) return;
        CaliperCanvasDesc c{};
        c.struct_size = sizeof c;
        c.mode = CALIPER_CANVAS_OFFSCREEN;
        c.width = 256;
        c.height = 256;
        c.content_scale = 1.0f;
        if (caliper_core_attach_canvas(core, nullptr, &c) != 1) return;
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) return;
        queue = [device newCommandQueue];
        gpu = true;
    }
    ~Core() { caliper_core_shutdown(core); }
};

enum class Src { Cpu, Shared, Private };
const char* src_name(Src s) { return s == Src::Cpu ? "cpu" : s == Src::Shared ? "shared" : "private"; }

// An MTLBuffer holding `bytes` after `offset` bytes of 0xab filler, in shared or private storage
// (a private one is filled by a blit, as a torch MPS buffer would be written by its kernels).
id<MTLBuffer> metal_buffer(Core& c, Src s, const void* bytes, size_t len, size_t offset) {
    std::vector<uint8_t> blob(offset + len, 0xab);
    std::memcpy(blob.data() + offset, bytes, len);
    id<MTLBuffer> shared = [c.device newBufferWithBytes:blob.data() length:blob.size()
                                                options:MTLResourceStorageModeShared];
    REQUIRE(bool(shared != nil));
    if (s == Src::Shared) return shared;
    id<MTLBuffer> priv = [c.device newBufferWithLength:blob.size() options:MTLResourceStorageModePrivate];
    REQUIRE(bool(priv != nil));
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:shared sourceOffset:0 toBuffer:priv destinationOffset:0 size:blob.size()];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return priv;
}

CaliperTensor tensor(void* data, CaliperDType dt, std::vector<int64_t> shape, CaliperDeviceKind dev) {
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = data;
    t.dtype = dt;
    t.ndim = (int32_t)shape.size();
    for (int i = 0; i < t.ndim; ++i) t.shape[i] = shape[i];
    int64_t st = 1;
    for (int i = t.ndim - 1; i >= 0; --i) { t.strides[i] = st; st *= t.shape[i]; }
    t.device = dev;
    return t;
}

// ---- The cases -----------------------------------------------------------------------------------

enum class Client {
    BridgeCreate,   // texture_from_tensor(_mapped) with the case's tensor
    BridgeUpdate,   // created from zeros on the CPU, then update_texture with the case's tensor
    BridgeAlloc,    // created from zeros, then import_allocation and update_texture_from_alloc
    BridgeShared,   // alloc_shared, written, then update_texture with the shared tensor
    Embed,          // caliper_core_view_tensor, the embed v1.2 view
    LiveHeatmap,    // caliper_core_live_apply, kind heatmap
    LiveImage,      // caliper_core_live_apply, kind image
};

struct Case {
    std::string name;
    Client client = Client::BridgeCreate;
    Src src = Src::Cpu;
    int w = 1, h = 1;
    int channels = 0;              // 0: f32 (H, W); 1, 3 or 4: u8 (H, W, C)
    int32_t colormap = CALIPER_CMAP_VIRIDIS;
    float vmin = 0.0f, vmax = 1.0f;
    uint64_t offset = 0;           // the byte offset in the MTLBuffer (alloc, embed and live clients)
    bool auto_range = false;       // live heatmap: the range from stats
    std::string info;
};

constexpr float kInf = std::numeric_limits<float>::infinity();

struct NamedRange { const char* name; float vmin, vmax; };
const NamedRange kRanges[] = {
    {"unit", 0.0f, 1.0f},
    {"sym", -1.0f, 1.0f},
    {"wide", -3.0f, 5.0f},
    {"dec", 1.0f, -1.0f},
    {"flat", 2.0f, 2.0f},
    {"inf", 0.0f, kInf},
    {"ninf", -kInf, 0.0f},
    {"tiny", 1e-38f, 2e-38f},
    {"huge", -3e38f, 3e38f},
    {"third", 0.0f, 3.0f},       // spans that are not powers of two, whose division rounds
    {"frac", -0.5f, 0.75f},
};

const char* cm_name(int32_t cm) {
    return cm == CALIPER_CMAP_VIRIDIS ? "viridis" : cm == CALIPER_CMAP_MAGMA ? "magma" : "rdbu";
}

std::string range_text(float a, float b) {
    char t[96];
    std::snprintf(t, sizeof t, "[%g, %g]", (double)a, (double)b);
    return t;
}

std::vector<Case> all_cases() {
    std::vector<Case> out;
    const std::pair<int, int> sizes[] = {{1, 1}, {3, 2}, {17, 9}, {64, 48}, {255, 129},
                                         {256, 256}, {1000, 7}, {7, 1000}, {1024, 768}};
    // Lattices of every size class, created from a CPU tensor and from an MTLBuffer.
    for (Src s : {Src::Cpu, Src::Shared})
        for (const auto& wh : sizes) {
            Case c;
            c.client = Client::BridgeCreate;
            c.src = s;
            c.w = wh.first;
            c.h = wh.second;
            c.name = std::string("bridge-create-") + src_name(s) + "-viridis-unit-" + std::to_string(c.w) + "x" +
                     std::to_string(c.h);
            c.info = "texture_from_tensor_mapped viridis [0, 1] from a " + std::string(src_name(s)) + " tensor";
            out.push_back(c);
        }
    // Every colormap over a symmetric range, and viridis over every kind of range, by update.
    for (Src s : {Src::Cpu, Src::Shared, Src::Private}) {
        for (int32_t cm : {CALIPER_CMAP_VIRIDIS, CALIPER_CMAP_MAGMA, CALIPER_CMAP_RDBU})
            for (const NamedRange& r : kRanges) {
                if (cm != CALIPER_CMAP_VIRIDIS && std::string(r.name) != "sym") continue;
                Case c;
                c.client = Client::BridgeUpdate;
                c.src = s;
                c.w = 64;
                c.h = 48;
                c.colormap = cm;
                c.vmin = r.vmin;
                c.vmax = r.vmax;
                c.name = std::string("bridge-update-") + src_name(s) + "-" + cm_name(cm) + "-" + r.name + "-64x48";
                c.info = std::string("update_texture ") + cm_name(cm) + " " + range_text(r.vmin, r.vmax) + " from a " +
                         src_name(s) + " tensor";
                out.push_back(c);
            }
    }
    // Imported allocations at byte offsets 0, 20 (element 5) and 4096.
    for (Src s : {Src::Shared, Src::Private}) {
        for (uint64_t off : {0ull, 20ull, 4096ull}) {
            Case c;
            c.client = Client::BridgeAlloc;
            c.src = s;
            c.w = 255;
            c.h = 129;
            c.vmin = -3.0f;
            c.vmax = 5.0f;
            c.offset = off;
            c.name = std::string("bridge-alloc-") + src_name(s) + "-viridis-wide-255x129-at" + std::to_string(off);
            c.info = "update_texture_from_alloc viridis [-3, 5] from a " + std::string(src_name(s)) +
                     " MTLBuffer at byte " + std::to_string(off);
            out.push_back(c);
        }
        Case c;
        c.client = Client::BridgeAlloc;
        c.src = s;
        c.w = 17;
        c.h = 9;
        c.colormap = CALIPER_CMAP_MAGMA;
        c.vmin = 1.0f;
        c.vmax = -1.0f;
        c.offset = 20;
        c.name = std::string("bridge-alloc-") + src_name(s) + "-magma-dec-17x9-at20";
        c.info = "update_texture_from_alloc magma [1, -1] from a " + std::string(src_name(s)) + " MTLBuffer at byte 20";
        out.push_back(c);
    }
    {
        Case c;
        c.client = Client::BridgeShared;
        c.w = 64;
        c.h = 48;
        c.name = "bridge-shared-viridis-unit-64x48";
        c.info = "alloc_shared f32, written, then update_texture viridis [0, 1]";
        out.push_back(c);
        c.channels = 4;
        c.w = 17;
        c.h = 9;
        c.name = "bridge-shared-u8c4-17x9";
        c.info = "alloc_shared u8 (H, W, 4), written, then update_texture";
        out.push_back(c);
    }
    // Images: one, three and four channels from the CPU; four from MTLBuffers, directly and imported.
    for (int ch : {1, 3, 4})
        for (const auto& wh : {std::pair<int, int>{3, 2}, {17, 9}, {256, 256}}) {
            Case c;
            c.client = Client::BridgeCreate;
            c.channels = ch;
            c.w = wh.first;
            c.h = wh.second;
            c.name = "bridge-create-cpu-u8c" + std::to_string(ch) + "-" + std::to_string(c.w) + "x" + std::to_string(c.h);
            c.info = "texture_from_tensor u8 (H, W, " + std::to_string(ch) + ") from a cpu tensor";
            out.push_back(c);
        }
    for (Src s : {Src::Shared, Src::Private})
        for (const auto& wh : {std::pair<int, int>{17, 9}, {256, 256}}) {
            Case c;
            c.client = Client::BridgeCreate;
            c.src = s;
            c.channels = 4;
            c.w = wh.first;
            c.h = wh.second;
            c.name = std::string("bridge-create-") + src_name(s) + "-u8c4-" + std::to_string(c.w) + "x" + std::to_string(c.h);
            c.info = "texture_from_tensor u8 (H, W, 4) from a " + std::string(src_name(s)) + " MTLBuffer";
            out.push_back(c);
        }
    for (Src s : {Src::Shared, Src::Private})
        for (uint64_t off : {0ull, 20ull}) {
            Case c;
            c.client = Client::BridgeAlloc;
            c.src = s;
            c.channels = 4;
            c.w = 64;
            c.h = 48;
            c.offset = off;
            c.name = std::string("bridge-alloc-") + src_name(s) + "-u8c4-64x48-at" + std::to_string(off);
            c.info = "update_texture_from_alloc u8 (H, W, 4) from a " + std::string(src_name(s)) + " MTLBuffer at byte " +
                     std::to_string(off);
            out.push_back(c);
        }
    // The embed views.
    struct EmbedCase { const char* name; Src src; int w, h, ch; int32_t cm; float vmin, vmax; uint64_t off; };
    const EmbedCase embeds[] = {
        {"embed-cpu-viridis-unit-64x48", Src::Cpu, 64, 48, 0, CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, 0},
        {"embed-shared-viridis-unit-64x48-at64", Src::Shared, 64, 48, 0, CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f, 64},
        {"embed-private-magma-sym-255x129", Src::Private, 255, 129, 0, CALIPER_CMAP_MAGMA, -1.0f, 1.0f, 0},
        {"embed-shared-rdbu-dec-17x9-at20", Src::Shared, 17, 9, 0, CALIPER_CMAP_RDBU, 1.0f, -1.0f, 20},
        {"embed-cpu-rdbu-flat-1000x7", Src::Cpu, 1000, 7, 0, CALIPER_CMAP_RDBU, 2.0f, 2.0f, 0},
        {"embed-shared-viridis-tiny-1024x768", Src::Shared, 1024, 768, 0, CALIPER_CMAP_VIRIDIS, 1e-38f, 2e-38f, 0},
        {"embed-cpu-u8c4-17x9", Src::Cpu, 17, 9, 4, 0, 0.0f, 1.0f, 0},
        {"embed-shared-u8c4-64x48-at64", Src::Shared, 64, 48, 4, 0, 0.0f, 1.0f, 64},
        {"embed-private-u8c4-256x256", Src::Private, 256, 256, 4, 0, 0.0f, 1.0f, 0},
    };
    for (const EmbedCase& e : embeds) {
        Case c;
        c.client = Client::Embed;
        c.name = e.name;
        c.src = e.src;
        c.w = e.w;
        c.h = e.h;
        c.channels = e.ch;
        c.colormap = e.cm;
        c.vmin = e.vmin;
        c.vmax = e.vmax;
        c.offset = e.off;
        c.info = e.ch ? "caliper_core_view_tensor u8 (H, W, 4) from a " + std::string(src_name(e.src)) + " tensor"
                      : std::string("caliper_core_view_tensor ") + cm_name(e.cm) + " " + range_text(e.vmin, e.vmax) +
                            " from a " + src_name(e.src) + " tensor";
        out.push_back(c);
    }
    // The live views: heatmaps with a range from statistics and with a fixed one, and images.
    struct LiveCase { const char* name; Client client; Src src; int w, h; int32_t cm; float vmin, vmax; bool auto_range; uint64_t off; };
    const LiveCase lives[] = {
        {"live-heatmap-cpu-viridis-auto-64x48", Client::LiveHeatmap, Src::Cpu, 64, 48, CALIPER_CMAP_VIRIDIS, -2.0f, 3.0f, true, 0},
        {"live-heatmap-shared-magma-auto-255x129-at64", Client::LiveHeatmap, Src::Shared, 255, 129, CALIPER_CMAP_MAGMA, -0.5f, 0.75f, true, 64},
        {"live-heatmap-private-rdbu-fixed-sym-17x9", Client::LiveHeatmap, Src::Private, 17, 9, CALIPER_CMAP_RDBU, -1.0f, 1.0f, false, 0},
        {"live-image-cpu-u8c4-64x48", Client::LiveImage, Src::Cpu, 64, 48, 0, 0.0f, 1.0f, false, 0},
        {"live-image-shared-u8c4-17x9-at64", Client::LiveImage, Src::Shared, 17, 9, 0, 0.0f, 1.0f, false, 64},
    };
    for (const LiveCase& l : lives) {
        Case c;
        c.client = l.client;
        c.name = l.name;
        c.src = l.src;
        c.w = l.w;
        c.h = l.h;
        c.channels = l.client == Client::LiveImage ? 4 : 0;
        c.colormap = l.cm;
        c.vmin = l.vmin;
        c.vmax = l.vmax;
        c.auto_range = l.auto_range;
        c.offset = l.off;
        c.info = l.client == Client::LiveImage
                     ? "caliper_core_live_apply image from a " + std::string(src_name(l.src)) + " tensor"
                     : std::string("caliper_core_live_apply heatmap ") + cm_name(l.cm) + " " +
                           (l.auto_range ? "range from stats " : "fixed ") + range_text(l.vmin, l.vmax) + " from a " +
                           src_name(l.src) + " tensor";
        out.push_back(c);
    }
    return out;
}

// ---- The values of a case ------------------------------------------------------------------------

uint32_t bits_of(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
float float_of(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

// A small deterministic generator, so that every build draws the same values.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9e3779b97f4a7c15ull + 1) {}
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return uint32_t(s >> 16);
    }
};

uint64_t seed_of(const std::string& name) {
    std::vector<uint8_t> b(name.begin(), name.end());
    return fnv1a64(b);
}

// The f32 values of a case: the special values, every level boundary of the range and its two
// neighbours (keys from derive_keys, so a decreasing range's are unflipped), a ramp from a quarter
// below the range to a quarter above it, and random bit patterns, NaN among them, laid out in turn
// over the texture so that a small texture still sees each kind.
std::vector<float> f32_values(const Case& c) {
    const float a = c.vmin, b = c.vmax;
    std::vector<uint32_t> special = {
        0x7fc00000u, 0x7f800001u, 0xffc00000u, 0xff800001u,   // quiet, signalling and negative NaN
        bits_of(kInf), bits_of(-kInf), 0x00000000u, 0x80000000u,
        0x00000001u, 0x007fffffu, 0x80000001u, 0x807fffffu,   // subnormals
        bits_of(std::numeric_limits<float>::max()), bits_of(-std::numeric_limits<float>::max()),
        bits_of(a), bits_of(b), bits_of(std::nextafter(a, kInf)), bits_of(std::nextafter(a, -kInf)),
        bits_of(std::nextafter(b, kInf)), bits_of(std::nextafter(b, -kInf)),
        bits_of(0.5f * a + 0.5f * b),
    };
    std::vector<uint32_t> boundary;
    const caliper_gfx::cmap::Keys keys = caliper_gfx::cmap::derive_keys(a, b);
    for (uint32_t k = 1; k < 256; ++k) {
        const uint32_t fk = keys.key[k];
        if (fk == caliper_gfx::cmap::kSentinel) continue;
        for (int d = -1; d <= 1; ++d) {
            const uint32_t f = fk + (uint32_t)d;
            boundary.push_back(caliper_gfx::cmap::bits_of(f ^ keys.flip));
        }
    }
    std::vector<uint32_t> ramp;
    const bool finite_span = std::isfinite(b - a) && std::isfinite(a) && std::isfinite(b);
    const float lo = finite_span ? a : -10.0f, span = finite_span ? b - a : 20.0f;
    for (int i = 0; i < 512; ++i) ramp.push_back(bits_of(lo + span * (-0.25f + 1.5f * (float)i / 511.0f)));
    Rng rng(seed_of(c.name));
    std::vector<uint32_t> random;
    for (int i = 0; i < 1024; ++i) random.push_back(rng.next());
    const std::vector<uint32_t>* pools[] = {&special, &boundary, &ramp, &random};
    std::vector<float> out((size_t)c.w * c.h);
    size_t at[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < out.size(); ++i) {
        // Each texel takes the next value of a pool in turn; an empty pool (a range without
        // boundaries) passes its turn to the ramp.
        size_t p = i % 4;
        if (pools[p]->empty()) p = 2;
        out[i] = float_of((*pools[p])[at[p]++ % pools[p]->size()]);
    }
    return out;
}

std::vector<uint8_t> u8_values(const Case& c) {
    std::vector<uint8_t> out((size_t)c.w * c.h * c.channels);
    Rng rng(seed_of(c.name));
    for (size_t i = 0; i < out.size(); ++i) out[i] = (i % 3 == 0) ? (uint8_t)(i * 7 + 3) : (uint8_t)rng.next();
    return out;
}

// ---- Running a case ------------------------------------------------------------------------------

struct Result {
    bool ok = false;
    std::vector<uint8_t> px;       // the readback, tightly packed RGBA8, top row first
    std::vector<uint8_t> want;     // the CPU reference
    std::string path;
    uint64_t copies = 0, bytes = 0;
    int imported = -1;             // embed and live views: what the view reports; -1 for the bridge
};

// The renderer's copy counter, which the texture path's uploads feed.
caliper_host::HostRenderer::CopyCount copies_now() {
    caliper_host::HostRenderer* r = caliper_host::services_renderer();
    return r ? r->copy_count() : caliper_host::HostRenderer::CopyCount{};
}

// A bridge texture's texels, read back through the graphics layer after every update made before it.
std::vector<uint8_t> bridge_readback(CaliperTextureId id, int w, int h) {
    std::vector<uint8_t> px;
    if (!caliper_host::services_bridge()->readback(id, &px) || px.size() != (size_t)w * h * 4) px.clear();
    return px;
}

// The path label of the bridge's last update of a texture, as the graphics layer returned it.
std::string bridge_path(const Case&) { return caliper_host::services_bridge()->last_path(); }

Result run_bridge(Core& core, const Case& c) {
    Result r;
    TensorBridge* bridge = caliper_host::services_bridge();
    REQUIRE(bridge != nullptr);
    const bool f32 = c.channels == 0;
    std::vector<float> fv;
    std::vector<uint8_t> uv;
    const void* bytes = nullptr;
    size_t len = 0;
    if (f32) { fv = f32_values(c); bytes = fv.data(); len = fv.size() * 4; }
    else { uv = u8_values(c); bytes = uv.data(); len = uv.size(); }
    const std::vector<int64_t> shape = f32 ? std::vector<int64_t>{c.h, c.w} : std::vector<int64_t>{c.h, c.w, c.channels};
    const CaliperDType dt = f32 ? CALIPER_DT_F32 : CALIPER_DT_U8;

    // The CPU reference.
    r.want.assign((size_t)c.w * c.h * 4, 0);
    if (f32) caliper_host::map_f32_to_rgba8(fv.data(), c.w, c.h, caliper_host::colormap_lut(c.colormap), c.vmin, c.vmax,
                                            r.want.data());
    else caliper_host::expand_u8_to_rgba8(uv.data(), c.w, c.h, c.channels, r.want.data());

    std::vector<uint8_t> zeros(len, 0);
    CaliperTensor zero = tensor(zeros.data(), dt, shape, CALIPER_DEV_CPU);
    auto create = [&](CaliperTensor* t) -> CaliperTextureId {
        return f32 ? bridge->texture_from_tensor_mapped(t, c.colormap, c.vmin, c.vmax, 0) : bridge->texture_from_tensor(t, 0);
    };

    id<MTLBuffer> buf = nil;
    CaliperTextureId tex = 0;
    CaliperAllocId alloc = 0;
    caliper_host::HostRenderer::CopyCount c0{};
    if (c.client == Client::BridgeShared) {
        CaliperTensor out{};
        REQUIRE(bridge->alloc_shared(dt, (int32_t)shape.size(), shape.data(), &out, &tex));
        REQUIRE(tex != 0);
        std::memcpy(out.data, bytes, len);
        c0 = copies_now();
        r.ok = bridge->update_texture(tex, &out);
        r.path = bridge_path(c);
    } else if (c.client == Client::BridgeAlloc) {
        buf = metal_buffer(core, c.src, bytes, len, (size_t)c.offset);
        tex = create(&zero);
        REQUIRE(tex != 0);
        alloc = bridge->import_allocation((__bridge void*)buf, buf.length, CALIPER_ALLOC_HANDLE_MTLBUFFER);
        REQUIRE(alloc != 0);
        CaliperTensor desc = tensor(nullptr, dt, shape, CALIPER_DEV_METAL);
        c0 = copies_now();
        r.ok = bridge->update_texture_from_alloc(tex, alloc, c.offset, &desc);
        r.path = bridge_path(c);
    } else {
        void* data = const_cast<void*>(bytes);
        CaliperDeviceKind dev = CALIPER_DEV_CPU;
        if (c.src != Src::Cpu) {
            buf = metal_buffer(core, c.src, bytes, len, 0);
            data = (__bridge void*)buf;
            dev = CALIPER_DEV_METAL;
        }
        CaliperTensor t = tensor(data, dt, shape, dev);
        if (c.client == Client::BridgeCreate) {
            c0 = copies_now();
            tex = create(&t);
            r.ok = tex != 0;
        } else {
            tex = create(&zero);
            REQUIRE(tex != 0);
            c0 = copies_now();
            r.ok = bridge->update_texture(tex, &t);
        }
        r.path = bridge_path(c);
    }
    const caliper_host::HostRenderer::CopyCount c1 = copies_now();
    r.copies = c1.copies - c0.copies;
    r.bytes = c1.bytes - c0.bytes;
    if (r.ok) r.px = bridge_readback(tex, c.w, c.h);
    if (alloc) bridge->release_allocation(alloc);
    if (tex) bridge->release_texture(tex);
    return r;
}

Result run_view(Core& core, const Case& c) {
    Result r;
    const bool f32 = c.channels == 0;
    std::vector<float> fv;
    std::vector<uint8_t> uv;
    const void* bytes = nullptr;
    size_t len = 0;
    if (f32) { fv = f32_values(c); bytes = fv.data(); len = fv.size() * 4; }
    else { uv = u8_values(c); bytes = uv.data(); len = uv.size(); }
    const std::vector<int64_t> shape = f32 ? std::vector<int64_t>{c.h, c.w} : std::vector<int64_t>{c.h, c.w, c.channels};
    const CaliperDType dt = f32 ? CALIPER_DT_F32 : CALIPER_DT_U8;

    // A CPU tensor's bytes sit at the same offset in a host blob, as a slot of a session would.
    std::vector<uint8_t> blob(c.offset + len, 0xab);
    std::memcpy(blob.data() + c.offset, bytes, len);
    id<MTLBuffer> buf = nil;
    CaliperTensor t = tensor(blob.data(), dt, shape, CALIPER_DEV_CPU);
    if (c.src != Src::Cpu) {
        buf = metal_buffer(core, c.src, bytes, len, (size_t)c.offset);
        t = tensor((__bridge void*)buf, dt, shape, CALIPER_DEV_METAL);
    }

    // The range the view draws with: a range from statistics is the producer's [min, max].
    float vmin = c.vmin, vmax = c.vmax;
    const float stats[2] = {c.vmin, c.vmax};
    r.want.assign((size_t)c.w * c.h * 4, 0);
    if (f32) caliper_host::map_f32_to_rgba8(fv.data(), c.w, c.h, caliper_host::colormap_lut(c.colormap), vmin, vmax,
                                            r.want.data());
    else caliper_host::expand_u8_to_rgba8(uv.data(), c.w, c.h, c.channels, r.want.data());

    const caliper_host::HostRenderer::CopyCount c0 = copies_now();
    if (c.client == Client::Embed) {
        CaliperViewDesc d{};
        d.struct_size = sizeof d;
        d.name = c.name.c_str();
        d.tensor = &t;
        d.byte_offset = c.src == Src::Cpu ? 0 : c.offset;
        if (c.src == Src::Cpu) t.data = blob.data() + c.offset;
        d.colormap = c.colormap;
        d.vmin = vmin;
        d.vmax = vmax;
        r.ok = caliper_core_view_tensor(core.core, &d) == 1;
        INFO(caliper_core_last_error(core.core));
        CHECK(r.ok);
        if (r.ok) {
            CaliperViewInfo vi{};
            vi.struct_size = sizeof vi;
            REQUIRE(caliper_core_view_info(core.core, c.name.c_str(), &vi) == 1);
            r.path = vi.path ? vi.path : "";
            r.imported = vi.imported;
        }
    } else {
        CaliperLiveSnapshot s{};
        s.struct_size = sizeof s;
        s.name = c.name.c_str();
        s.kind = c.client == Client::LiveImage ? CALIPER_LIVE_KIND_IMAGE : CALIPER_LIVE_KIND_HEATMAP;
        s.data = &t;
        s.data_offset = c.offset;
        s.colormap = c.colormap;
        s.vmin = c.auto_range ? 0.0f : vmin;
        s.vmax = c.auto_range ? 1.0f : vmax;
        if (c.auto_range) {
            s.stats = stats;
            s.stats_count = 2;
            s.flags = CALIPER_LIVE_APPLY_AUTO_VMIN | CALIPER_LIVE_APPLY_AUTO_VMAX;
        }
        s.seq = 1;
        r.ok = caliper_core_live_apply(core.core, &s) == 1;
        INFO(caliper_core_last_error(core.core));
        CHECK(r.ok);
        if (r.ok) {
            CaliperLiveViewInfo vi{};
            vi.struct_size = sizeof vi;
            REQUIRE(caliper_core_live_view_info(core.core, c.name.c_str(), &vi) == 1);
            r.path = vi.path ? vi.path : "";
            r.imported = vi.imported;
        }
    }
    const caliper_host::HostRenderer::CopyCount c1 = copies_now();
    r.copies = c1.copies - c0.copies;
    r.bytes = c1.bytes - c0.bytes;
    if (r.ok) {
        r.px.assign((size_t)c.w * c.h * 4, 0);
        const int got = c.client == Client::Embed
                            ? caliper_core_view_readback(core.core, c.name.c_str(), r.px.data(), r.px.size())
                            : caliper_core_live_view_readback(core.core, c.name.c_str(), r.px.data(), r.px.size());
        INFO(caliper_core_last_error(core.core));
        CHECK(got == 1);
        if (got != 1) r.px.clear();
    }
    if (c.client == Client::Embed) caliper_core_view_remove(core.core, c.name.c_str());
    else caliper_core_live_view_remove(core.core, c.name.c_str());
    return r;
}

Result run_case(Core& core, const Case& c) {
    @autoreleasepool {
        switch (c.client) {
            case Client::BridgeCreate:
            case Client::BridgeUpdate:
            case Client::BridgeAlloc:
            case Client::BridgeShared: return run_bridge(core, c);
            case Client::Embed:
            case Client::LiveHeatmap:
            case Client::LiveImage: return run_view(core, c);
        }
    }
    return {};
}

// The texels where a readback differs from the CPU reference.
std::vector<size_t> differing(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
    std::vector<size_t> out;
    if (got.size() != want.size()) return out;
    for (size_t i = 0; i < got.size() / 4; ++i)
        if (std::memcmp(&got[i * 4], &want[i * 4], 4) != 0) out.push_back(i);
    return out;
}

Line line_of(const Case& c, const Result& r) {
    Line l;
    l.name = c.name;
    l.width = c.w;
    l.height = c.h;
    l.hash = hex16(fnv1a64(r.px));
    l.fields.emplace_back("path", r.path.empty() ? "-" : r.path);
    if (r.imported >= 0) l.fields.emplace_back("imported", std::to_string(r.imported));
    l.fields.emplace_back("copies", std::to_string(r.copies));
    l.fields.emplace_back("bytes", std::to_string(r.bytes));
    const size_t n = differing(r.px, r.want).size();
    l.fields.emplace_back("cpu_ref_differs", n == 0 ? std::string("none") : std::to_string(n));
    l.info = c.info;
    return l;
}

bool selected(const Case& c) {
    const char* only = env("CALIPER_GFX_TEXTURE_CASES");
    return only == nullptr || c.name.find(only) != std::string::npos;
}

}  // namespace

TEST_CASE("gfx/Metal texture path: every case against its reference record") {
    Core core;
    if (!core.gpu) CALIPER_SKIP_NO_GPU("no Metal device");
    References refs;
    const std::vector<Case> cases = all_cases();
    size_t run = 0;
    for (const Case& c : cases) {
        if (!selected(c)) continue;
        ++run;
        INFO("case " << c.name);
        const Result r = run_case(core, c);
        REQUIRE(r.ok);
        REQUIRE(r.px.size() == (size_t)c.w * c.h * 4);
        capture(c.name, r.px, c.w, c.h);
        const Line got = line_of(c, r);
        if (refs.recording()) {
            refs.record(got);
            continue;
        }
        const Line* want = refs.find(c.name);
        REQUIRE_MESSAGE(want != nullptr, "no reference line in ", refs.path());
        CHECK(got.width == want->width);
        CHECK(got.height == want->height);
        // Every texel is the CPU reference's. Where the old path's record equals the CPU reference
        // too, the readback is the old path's, byte for byte; where it does not (its fast math put
        // values one level off or flushed subnormals), the second row lists those pixels.
        const std::vector<size_t> off = differing(r.px, r.want);
        CHECK_MESSAGE(off.empty(), off.size(), " texels differ from the CPU reference");
        if (want->field("cpu_ref_differs") == "none") CHECK(got.hash == want->hash);
        for (const char* key : {"path", "imported", "copies", "bytes"})
            CHECK_MESSAGE(got.field(key) == want->field(key), key);
    }
    MESSAGE(run, " cases run");
}

namespace {

// The level of a colormap whose colour a texel has, nearest to `near` where levels share a colour;
// -1 for a colour that is no level's.
int level_of(const uint8_t* px, int32_t colormap, int near) {
    const uint32_t* lut = caliper_host::colormap_lut(colormap);
    const uint32_t c = px[0] | (px[1] << 8) | (px[2] << 16) | (uint32_t(px[3]) << 24);
    int best = -1;
    for (int k = 0; k < 256; ++k)
        if (lut[k] == c && (best < 0 || std::abs(k - near) < std::abs(best - near))) best = k;
    return best;
}

bool subnormal(float f) { return f != 0.0f && std::fpclassify(f) == FP_SUBNORMAL; }

std::string coords(const std::vector<size_t>& at, int w, size_t limit) {
    std::string s;
    for (size_t i = 0; i < at.size() && i < limit; ++i)
        s += (i ? ", (" : "(") + std::to_string(at[i] % w) + ", " + std::to_string(at[i] / w) + ")";
    if (at.size() > limit) s += " and " + std::to_string(at.size() - limit) + " more";
    return s;
}

}  // namespace

// The comparison of migration step 8: each case's readback against the old path's image of the
// same case, read from CALIPER_GFX_REFERENCE_IMAGES (the PNG files that CALIPER_GFX_CAPTURE_DIR
// wrote at the step's first commit), pixel by pixel. A pixel may differ only where the new image
// equals the CPU reference and the old one does not; each case prints one sentence with the
// number of differing pixels, their reasons and their coordinates (the first 24, and every one in
// texture-<case>-differs.txt beside the captures when CALIPER_GFX_CAPTURE_DIR is set).
TEST_CASE("gfx/Metal texture path: the new path's images against the old path's, pixel by pixel") {
    const char* old_dir = env("CALIPER_GFX_REFERENCE_IMAGES");
    if (old_dir == nullptr) {
        MESSAGE("CALIPER_GFX_REFERENCE_IMAGES names no folder of the old path's images; nothing compared");
        return;
    }
    Core core;
    if (!core.gpu) CALIPER_SKIP_NO_GPU("no Metal device");
    size_t run = 0, explained_total = 0;
    for (const Case& c : all_cases()) {
        if (!selected(c)) continue;
        ++run;
        INFO("case " << c.name);
        const std::string file = std::string(old_dir) + "/texture-" + c.name + ".png";
        int w = 0, h = 0, n = 0;
        uint8_t* old = stbi_load(file.c_str(), &w, &h, &n, 4);
        REQUIRE_MESSAGE(old != nullptr, "no old image ", file);
        std::vector<uint8_t> was(old, old + (size_t)w * h * 4);
        stbi_image_free(old);
        REQUIRE(w == c.w);
        REQUIRE(h == c.h);
        const Result r = run_case(core, c);
        REQUIRE(r.ok);
        REQUIRE(r.px.size() == was.size());
        const std::vector<float> values = c.channels == 0 ? f32_values(c) : std::vector<float>{};
        const bool flushed_range = subnormal(c.vmin) || subnormal(c.vmax);
        std::vector<size_t> flushed, boundary, other_old, wrong_new;
        size_t differ = 0;
        for (size_t i = 0; i < was.size() / 4; ++i) {
            const uint8_t* o = &was[i * 4];
            const uint8_t* nw = &r.px[i * 4];
            const uint8_t* ref = &r.want[i * 4];
            const bool new_ref = std::memcmp(nw, ref, 4) == 0;
            if (!new_ref) wrong_new.push_back(i);
            if (std::memcmp(o, nw, 4) == 0) continue;
            ++differ;
            if (!new_ref) continue;
            // The new texel is the reference's and the old one is not: say why the old was wrong.
            if (c.channels == 0 && (flushed_range || subnormal(values[i]))) {
                flushed.push_back(i);
                continue;
            }
            const int want_level = c.channels == 0 ? level_of(ref, c.colormap, 0) : -1;
            const int old_level = c.channels == 0 ? level_of(o, c.colormap, want_level) : -1;
            if (want_level >= 0 && old_level >= 0 && std::abs(old_level - want_level) == 1) boundary.push_back(i);
            else other_old.push_back(i);
        }
        const size_t total = was.size() / 4;
        std::string sentence = "case " + c.name + ": " + std::to_string(differ) + " of " + std::to_string(total) +
                               " pixels differ between the old path's image and the new one";
        if (differ == 0) {
            sentence += ".";
        } else {
            sentence += "; " + std::to_string(flushed.size() + boundary.size() + other_old.size()) +
                        " of them are pixels where the new image equals the CPU reference map_f32_to_rgba8 and the old one "
                        "does not: " + std::to_string(flushed.size()) +
                        " because the old shader's fast math flushed a subnormal range end or value to zero" +
                        (flushed.empty() ? "" : " at " + coords(flushed, c.w, 24)) + ", " + std::to_string(boundary.size()) +
                        " at a colormap level boundary that the old shader's division put one level off" +
                        (boundary.empty() ? "" : " at " + coords(boundary, c.w, 24)) + ", and " +
                        std::to_string(other_old.size()) + " for another reason" +
                        (other_old.empty() ? "" : " at " + coords(other_old, c.w, 24)) + "; " +
                        std::to_string(differ - flushed.size() - boundary.size() - other_old.size()) +
                        " differ where the new image is not the CPU reference.";
        }
        std::printf("[texture-compare] %s\n", sentence.c_str());
        if (const char* dir = env("CALIPER_GFX_CAPTURE_DIR"); dir != nullptr && differ > 0) {
            std::ofstream f(std::string(dir) + "/texture-" + c.name + "-differs.txt");
            auto list = [&](const char* why, const std::vector<size_t>& at) {
                for (size_t i : at) f << (i % c.w) << " " << (i / c.w) << " " << why << "\n";
            };
            list("flushed-subnormal", flushed);
            list("level-boundary", boundary);
            list("other", other_old);
        }
        explained_total += flushed.size() + boundary.size();
        CHECK_MESSAGE(wrong_new.empty(), wrong_new.size(), " pixels of the new image differ from the CPU reference at ",
                      coords(wrong_new, c.w, 24));
        CHECK_MESSAGE(other_old.empty(), "pixels that differ for no known reason at ", coords(other_old, c.w, 24));
    }
    MESSAGE(run, " cases compared with the old path's images; ", explained_total,
            " pixels differ, each where the old path differed from the CPU reference");
}

// The canvas: a heatmap live view drawn raw (one texel on one canvas pixel at the top-left corner)
// shows its layer texture through ImGui exactly. A windowed heatmap and an image are then drawn
// beside it, and with CALIPER_GFX_CAPTURE_DIR set the canvas is written as texture-canvas.png.
TEST_CASE("gfx/Metal texture path: the views draw their layer textures on the canvas") {
    Core core;
    if (!core.gpu) CALIPER_SKIP_NO_GPU("no Metal device");
    const int W = 96, H = 64;
    std::vector<float> ramp((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) ramp[(size_t)y * W + x] = -1.0f + 2.0f * (float)x / (float)(W - 1) + 0.25f * (float)y / (float)H;
    CaliperTensor t = tensor(ramp.data(), CALIPER_DT_F32, {H, W}, CALIPER_DEV_CPU);
    CaliperLiveSnapshot s{};
    s.struct_size = sizeof s;
    s.name = "raw";
    s.kind = CALIPER_LIVE_KIND_HEATMAP;
    s.data = &t;
    s.colormap = CALIPER_CMAP_RDBU;
    s.vmin = -1.0f;
    s.vmax = 1.25f;
    s.flags = CALIPER_LIVE_APPLY_RAW;
    s.seq = 1;
    REQUIRE_MESSAGE(caliper_core_live_apply(core.core, &s) == 1, caliper_core_last_error(core.core));
    for (int i = 0; i < 3; ++i) caliper_core_frame(core.core);
    std::vector<uint8_t> canvas(256 * 256 * 4);
    REQUIRE(caliper_core_read_pixels(core.core, canvas.data(), 256 * 4) == 1);
    std::vector<uint8_t> want((size_t)W * H * 4);
    caliper_host::map_f32_to_rgba8(ramp.data(), W, H, caliper_host::colormap_lut(CALIPER_CMAP_RDBU), -1.0f, 1.25f,
                                   want.data());
    size_t bad = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            bad += std::memcmp(&canvas[((size_t)y * 256 + x) * 4], &want[((size_t)y * W + x) * 4], 4) != 0;
    CHECK_MESSAGE(bad == 0, bad, " of ", W * H, " pixels of the raw heatmap differ from the CPU reference on the canvas");

    // A windowed heatmap from an MTLBuffer and an image, drawn in the views' grid.
    id<MTLBuffer> buf = metal_buffer(core, Src::Shared, ramp.data(), ramp.size() * 4, 0);
    CaliperTensor tm = tensor((__bridge void*)buf, CALIPER_DT_F32, {H, W}, CALIPER_DEV_METAL);
    s.name = "heatmap";
    s.data = &tm;
    s.colormap = CALIPER_CMAP_VIRIDIS;
    s.flags = 0;
    REQUIRE_MESSAGE(caliper_core_live_apply(core.core, &s) == 1, caliper_core_last_error(core.core));
    std::vector<uint8_t> img((size_t)W * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint8_t* p = &img[((size_t)y * W + x) * 4];
            p[0] = (uint8_t)(x * 255 / (W - 1));
            p[1] = (uint8_t)(y * 255 / (H - 1));
            p[2] = (uint8_t)((x ^ y) & 0xff);
            p[3] = 255;
        }
    CaliperTensor ti = tensor(img.data(), CALIPER_DT_U8, {H, W, 4}, CALIPER_DEV_CPU);
    s.name = "image";
    s.kind = CALIPER_LIVE_KIND_IMAGE;
    s.data = &ti;
    REQUIRE_MESSAGE(caliper_core_live_apply(core.core, &s) == 1, caliper_core_last_error(core.core));
    for (int i = 0; i < 3; ++i) caliper_core_frame(core.core);
    REQUIRE(caliper_core_read_pixels(core.core, canvas.data(), 256 * 4) == 1);
    capture("canvas", canvas, 256, 256);
}

CALIPER_GPU_TEST_MAIN()
