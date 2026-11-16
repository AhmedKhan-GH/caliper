// gfx.h: the graphics layer of the renderer core (design/renderer-core.md, "The graphics layer").
//
// A thin internal layer, namespace caliper_gfx, shaped to WebGPU's model and default limits:
// buffers, textures, samplers, programs, pipelines, bind groups, draw lists and frames. Each
// backend (Metal, Vulkan, WebGPU in a browser and on Dawn) implements it once. Nothing of it
// crosses the ABI. Nothing draws through it yet; the renderers keep their own paths until each
// migration step moves a client.
//
// Decisions taken where the design's sketch was incomplete or inconsistent (each is also in the
// report of the step that added this header):
//
//  1. Validation lives in the core, once. Backend's public methods are non-virtual: they check
//     the call against caps().limits and the live handle tables, then call a protected hook
//     that the backend implements (the hooks are at the end of the class). A refused call
//     returns a null handle, false or serial 0 and leaves the reason in last_error().
//  2. A handle encodes its kind as well as its slot and generation (bits 0-23 slot, 24-59
//     generation, 60-63 kind), so release(uint64_t) and last_use(uint64_t) know the table.
//     A slot is reused with a new generation and retired when its generation would wrap, so
//     an id is never reused.
//  3. create_program takes a ProgramDesc (sources and the bind group description) instead of
//     a program id. The shader generator emits a table of ProgramDescs; its id is an index
//     into that table. Until the generator lands, tests pass hand-written descriptors.
//  4. A bind group is made for a program and a group index, not for a pipeline, and is valid
//     with every pipeline of that program and with any program that declares the same group
//     layout (WebGPU's group equivalence for explicit layouts). Group 0 can then be shared.
//  5. The Metal binding map is part of the program's description: each binding names its MSL
//     index ([[buffer(n)]], [[texture(n)]] or [[sampler(n)]], the same in every stage), buffer
//     indices stay below kMetalSizesIndex, and vertex buffer slot i is Metal buffer index
//     kMetalVertexBufferBase + i, read through [[stage_in]] with [[attribute(location)]]. Naga's
//     MSL reads the byte size of each runtime-sized storage binding from a sizes buffer at
//     kMetalSizesIndex: the Metal backend binds one uint per Storage or StorageReadWrite
//     binding of the program, in (group, binding) order, to every stage that has one.
//  6. The uniform arena is a pseudo-buffer, kArena. A dynamic uniform binding whose buffer is
//     kArena binds the arena of the pass it is used in: a DrawList's own arena for its draws,
//     the Frame's arena for its dispatches. DrawList::uniform copies into the list's CPU arena
//     at a 256-byte-aligned offset; at submit the backend uploads each arena once and rebases
//     the offsets. The bind group itself never changes from frame to frame.
//  7. A draw has up to four dynamic offsets, not two: the frame block of group 0 lives in the
//     arena too, beside the draw block and the colormap block of group 1. They apply to the
//     dynamic bindings of the bound groups in (group, binding) order.
//  8. DrawRecord and RenderPipelineDesc both allow eight vertex buffers (WebGPU's limit); the
//     sketch had six in one and eight in the other.
//  9. DrawRecord::pick_id is data for the draw map and the tests. The backend writes nothing
//     from it: a program writes the ID target from its own parameter block.
// 10. For an indexed draw, first_vertex is the base vertex added to each index, and
//     index_offset (bytes, a multiple of 4) selects the first index. Indices are u32.
// 11. Pipelines are cached by descriptor and reference-counted: creating an identical
//     descriptor again returns the same handle, and each create needs one release.
// 12. write_buffer and write_texture return false when refused, instead of void.
// 13. A whole Frame is validated before anything is encoded; one bad draw refuses the submit
//     and leaves every target untouched (the geometry service's atomic frame).
// 14. read_texture returns the region tightly packed, top row first, in the texture's format
//     (4 bytes a texel for RGBA8, 8 for RG32Uint, 4 for Depth32Float).
// 15. update_texture_from_tensor and fill_texture_from_buffer are declared with their final
//     shape; their portable defaults need the colormap and fill programs of the shader
//     generator, so until then the defaults refuse, except fill_texture_from_buffer when a
//     plain copy does it (row pitch a multiple of 256).
// 16. Override constants are refused (constant_count must be 0): Naga's MSL cannot declare
//     function constants, so a value that varies goes in a uniform block. The fields stay so
//     that the descriptor keeps its layout.
// 17. The layer is used from one thread. Only completion is reported from other threads.
// 18. The header and src/host/gfx/ build as C++17, the host's standard: View<T>, a pointer
//     and a count, stands in for the C++20 span.
#pragma once

#include <caliper/tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace caliper_gfx {

// ---- A read-only view of contiguous elements, a pointer and a count (decision 18) -----------

template <class T> struct View {
    const T* data = nullptr;
    size_t size = 0;
    View() = default;
    View(const T* p, size_t n) : data(p), size(n) {}
    View(const std::vector<T>& v) : data(v.data()), size(v.size()) {}
    template <size_t N> View(const T (&a)[N]) : data(a), size(N) {}
    const T* begin() const { return data; }
    const T* end() const { return data + size; }
    const T& operator[](size_t i) const { return data[i]; }
    bool empty() const { return size == 0; }
};

// ---- Handles -------------------------------------------------------------------------------

enum class HandleKind : uint32_t {
    None = 0, Buffer = 1, Texture = 2, Sampler = 3, Program = 4, Pipeline = 5, BindGroup = 6,
    Readback = 7,
};

// A handle is a slot, a generation and a kind in 64 bits. 0 is none, and an id is never
// reused, so a use after release is refused instead of reaching another resource.
template <class Tag> struct Handle {
    uint64_t id = 0;
    explicit operator bool() const { return id != 0; }
    friend bool operator==(Handle a, Handle b) { return a.id == b.id; }
    friend bool operator!=(Handle a, Handle b) { return a.id != b.id; }
};
using Buffer = Handle<struct BufferTag>;
using Texture = Handle<struct TextureTag>;
using Sampler = Handle<struct SamplerTag>;
using Program = Handle<struct ProgramTag>;
using Pipeline = Handle<struct PipelineTag>;
using BindGroup = Handle<struct BindGroupTag>;
using Readback = Handle<struct ReadbackTag>;
using Serial = uint64_t;   // one per submit, increasing from 1; 0 is none (a refused submit)

template <class Tag> struct HandleTraits;
template <> struct HandleTraits<BufferTag> { static constexpr HandleKind kind = HandleKind::Buffer; };
template <> struct HandleTraits<TextureTag> { static constexpr HandleKind kind = HandleKind::Texture; };
template <> struct HandleTraits<SamplerTag> { static constexpr HandleKind kind = HandleKind::Sampler; };
template <> struct HandleTraits<ProgramTag> { static constexpr HandleKind kind = HandleKind::Program; };
template <> struct HandleTraits<PipelineTag> { static constexpr HandleKind kind = HandleKind::Pipeline; };
template <> struct HandleTraits<BindGroupTag> { static constexpr HandleKind kind = HandleKind::BindGroup; };
template <> struct HandleTraits<ReadbackTag> { static constexpr HandleKind kind = HandleKind::Readback; };

inline constexpr uint64_t kHandleSlotBits = 24, kHandleGenerationBits = 36;
inline constexpr uint64_t kHandleSlotMask = (1ull << kHandleSlotBits) - 1;
inline constexpr HandleKind handle_kind(uint64_t id) { return HandleKind(id >> 60); }
inline constexpr uint32_t handle_slot(uint64_t id) { return uint32_t(id & kHandleSlotMask); }

// The uniform arena of the pass a binding is used in (decision 6). Never a real slot.
inline constexpr Buffer kArena{(uint64_t(HandleKind::Buffer) << 60) | kHandleSlotMask};

// ---- Fixed sizes of the model ----------------------------------------------------------------

inline constexpr uint32_t kMaxBindGroups = 4;          // 0 frame, 1 draw, 2 images, 3 storage
inline constexpr uint32_t kMaxBindingsPerGroup = 8;
inline constexpr uint32_t kMaxDynamicOffsets = 4;      // per draw or dispatch (decision 7)
inline constexpr uint32_t kMaxVertexBuffers = 8;       // WebGPU's limit (decision 8)
inline constexpr uint32_t kMaxAttributesPerBuffer = 4;
inline constexpr uint32_t kMaxConstants = 8;
inline constexpr uint32_t kUniformAlign = 256;         // arena slices and dynamic offsets
inline constexpr uint32_t kCopyRowPitchAlign = 256;    // buffer-to-texture copies (WebGPU)
inline constexpr uint32_t kMetalVertexBufferBase = 16; // decision 5
inline constexpr uint32_t kMetalSizesIndex = 15;       // Naga's sizes buffer (decision 5)
inline constexpr uint32_t kNoOffset = 0xffffffffu;     // DrawList::uniform refused

// ---- Descriptors -----------------------------------------------------------------------------

enum BufferUsage : uint32_t {
    kVertex = 1, kIndex = 2, kUniform = 4, kStorage = 8, kCopySrc = 16, kCopyDst = 32,
};
enum TextureUsage : uint32_t {
    kSampled = 1, kStorageTex = 2, kRenderTarget = 4, kCopySrcTex = 8, kCopyDstTex = 16,
};
enum class Format : uint32_t {
    RGBA8Unorm, BGRA8Unorm, R8Unorm, R16Uint, R32Uint, RG32Uint, R32Float, Depth32Float,
};
uint32_t format_bytes(Format f);   // bytes per texel

struct BufferDesc {
    uint64_t size = 0;             // a multiple of 4, at most limits.max_buffer_size
    uint32_t usage = 0;
    bool host_visible = false;     // mapped() returns its memory with kCapPersistentMap
    const char* label = nullptr;
};
struct TextureDesc {
    uint32_t width = 0, height = 0;
    Format format = Format::RGBA8Unorm;
    uint32_t usage = 0;
    const char* label = nullptr;
};
struct SamplerDesc { bool linear = true; };   // clamp to edge, no mipmaps, as the geometry service fixes

// stride 0 makes a stream constant: every vertex or instance reads the first element.
enum class VertexFormat : uint32_t { F32, F32x2, F32x3, F32x4, Unorm8x4, U8, U32, S32, U32x2 };
uint32_t vertex_format_bytes(VertexFormat f);
enum class Step : uint32_t { Vertex, Instance };
struct VertexAttribute { uint32_t location = 0; VertexFormat format = VertexFormat::F32; uint32_t offset = 0; };
struct VertexLayout {
    uint32_t stride = 0;           // a multiple of 4 up to limits.max_vertex_stride, or 0
    Step step = Step::Vertex;
    uint8_t attribute_count = 0;
    VertexAttribute attributes[kMaxAttributesPerBuffer];
};

enum class Topology : uint32_t { TriangleList, TriangleStrip, LineList, LineStrip, PointList };
enum class Blend : uint32_t { Opaque, Alpha, Additive };   // the geometry service's three equations

struct RenderPipelineDesc {
    Program program;
    const char* vertex_entry = nullptr;
    const char* fragment_entry = nullptr;
    uint8_t vertex_layout_count = 0;
    VertexLayout vertex_layouts[kMaxVertexBuffers];   // layout i reads vertex buffer slot i
    Topology topology = Topology::TriangleList;
    Format color_format = Format::RGBA8Unorm;
    Blend blend = Blend::Opaque;
    bool id_target = false;        // a second attachment, RG32Uint: pick id and item
    bool depth = false, depth_test = false, depth_write = false;   // D32 float, less-or-equal
    uint8_t constant_count = 0;
    struct Constant { uint16_t id; float value; } constants[kMaxConstants];   // refused (decision 16)
    const char* label = nullptr;
};
struct ComputePipelineDesc { Program program; const char* entry = nullptr; const char* label = nullptr; };

struct BindEntry {
    uint32_t binding = 0;
    Buffer buffer;                 // or kArena for a dynamic uniform binding
    uint64_t offset = 0, size = 0; // a buffer range; size 0: a uniform's declared block, or the
                                   // rest of the buffer for a storage binding, as in WebGPU
    Texture texture;
    Sampler sampler;
};
struct BindGroupDesc {
    Program program;               // decision 4
    uint32_t group = 0;
    uint8_t entry_count = 0;
    BindEntry entries[kMaxBindingsPerGroup];
    const char* label = nullptr;
};

// ---- Programs: what the shader generator emits for each program (decisions 3 and 5) --------

enum Stage : uint8_t { kStageVertex = 1, kStageFragment = 2, kStageCompute = 4 };
enum class BindingType : uint8_t {
    Uniform,         // var<uniform>
    Storage,         // var<storage, read>
    Texture,         // texture_2d<f32>, filterable
    UintTexture,     // texture_2d<u32>
    Sampler,         // sampler, filtering
    StorageTexture,  // texture_storage_2d<format, write>
    StorageReadWrite,  // var<storage, read_write>: the index clamp's and edge gather's outputs
};
struct BindingDesc {
    uint8_t group = 0, binding = 0;
    BindingType type = BindingType::Uniform;
    uint8_t stages = 0;            // Stage bits
    bool dynamic = false;          // a uniform at a dynamic offset
    uint32_t size = 0;             // the uniform or storage block's size, the minimum binding size
    Format storage_format = Format::RGBA8Unorm;   // StorageTexture only
    uint8_t msl_index = 0;         // [[buffer(n)]], [[texture(n)]] or [[sampler(n)]], every stage
};
struct EntryDesc {
    const char* name = nullptr;    // the same in the WGSL, the MSL and the SPIR-V
    uint8_t stage = 0;             // one Stage bit
    uint16_t workgroup_size[3] = {1, 1, 1};   // compute entries
};
struct ProgramDesc {
    const char* label = nullptr;
    const EntryDesc* entries = nullptr;
    uint8_t entry_count = 0;
    const BindingDesc* bindings = nullptr;
    uint8_t binding_count = 0;
    const char* wgsl = nullptr;           // the WebGPU backend, in a browser and on Dawn
    const char* msl = nullptr;            // the Metal backend
    const uint32_t* spirv = nullptr;      // the Vulkan backend
    uint32_t spirv_bytes = 0;
};

// ---- Capabilities and limits -----------------------------------------------------------------

struct Limits {   // WebGPU's defaults are the floor; a native backend reports its own, larger values
    uint32_t max_texture_dimension_2d = 8192, max_bind_groups = 4, max_dynamic_uniform_buffers = 8;
    uint32_t max_storage_buffers_per_stage = 8, max_storage_buffers_in_vertex_stage = 8;  // 0 in compatibility mode
    uint32_t max_uniform_buffer_binding_size = 65536;
    uint64_t max_storage_buffer_binding_size = 134217728, max_buffer_size = 268435456;
    uint32_t min_uniform_buffer_offset_alignment = 256, min_storage_buffer_offset_alignment = 256;
    uint32_t max_vertex_buffers = 8, max_vertex_attributes = 16, max_vertex_stride = 2048, max_color_attachments = 8;
};

enum Cap : uint64_t {
    kCapVertexStorage      = 1ull << 0,   // storage buffers readable in the vertex stage
    kCapPersistentMap      = 1ull << 1,   // host-visible buffers stay mapped
    kCapBlockingWait       = 1ull << 2,   // wait(serial) may block; never in a browser
    kCapTimestamps         = 1ull << 3,   // GPU time per pass
    kCapImportMetalBuffer  = 1ull << 8,   // an id<MTLBuffer> of this device, in place
    kCapImportHostMemory   = 1ull << 9,   // page-aligned host memory wrapped in place
    kCapImportWin32Memory  = 1ull << 10,  // an exportable CUDA allocation, through Vulkan external memory
    kCapImportMetalTexture = 1ull << 11,
    kCapCudaWritesTexture  = 1ull << 12,  // a CUDA kernel writes a layer texture (the direct path)
    kCapWaitMetalEvent     = 1ull << 13,  // reads ordered after an MTLSharedEvent value
    kCapWaitCudaStream     = 1ull << 14,  // reads ordered after a CUDA stream, by a shared timeline semaphore
};

struct Caps {
    uint64_t bits = 0;
    Limits limits;
    const char* backend = "";      // metal, vulkan, webgpu (browser), dawn, recording
    const char* adapter = "";
    bool has(Cap c) const { return (bits & c) != 0; }
};

// ---- Draw lists ------------------------------------------------------------------------------

struct VertexBinding { Buffer buffer; uint64_t offset = 0; };   // offset a multiple of 4

struct DrawRecord {
    Pipeline pipeline;
    BindGroup groups[kMaxBindGroups];             // 0 frame, 1 draw, 2 images, 3 storage (vertex pulling only)
    uint32_t dynamic_offsets[kMaxDynamicOffsets] = {};   // decision 7
    VertexBinding vertex[kMaxVertexBuffers];      // slot i feeds the pipeline's vertex layout i
    Buffer index_buffer;                          // u32 indices; none for a non-indexed draw
    uint64_t index_offset = 0;                    // bytes, a multiple of 4
    uint32_t count = 0;                           // vertices, or indices, per instance
    uint32_t instance_count = 1, first_vertex = 0, first_instance = 0;   // decision 10
    uint32_t pick_id = 0;                         // decision 9; 0 is not pickable
    uint16_t scissor[4] = {0, 0, 0xffff, 0xffff}; // x, y, width, height in pixels, clamped to the target
};

struct RenderTarget {
    Texture color;
    Texture id;                    // RG32Uint, cleared to 0 (not pickable)
    Texture depth;                 // Depth32Float, cleared to 1.0
    uint32_t clear_rgba = 0;       // r | g << 8 | b << 16 | a << 24
    bool clear = true;             // false: load what the targets hold
};

// The backend-neutral list of draws for one target. Plain data: a view builds one without a
// backend, and a headless test can inspect one.
class DrawList {
public:
    DrawList() = default;
    explicit DrawList(const RenderTarget& target) : target_(target) {}
    // Copies size bytes into the list's arena at the next 256-byte boundary and returns that
    // dynamic offset, or kNoOffset (and the list is then refused at submit) for size 0 or a
    // block larger than 64 KiB, WebGPU's default uniform binding size. Each slice is padded
    // to a multiple of 256 bytes, so a 192-byte block takes 256 and a 2,080-byte one 2,304.
    uint32_t uniform(const void* data, uint32_t size);
    void draw(const DrawRecord& record) { records_.push_back(record); }
    void reserve(size_t draws, size_t arena_bytes);
    void reset(const RenderTarget& target);       // empties the list and keeps its capacity

    const RenderTarget& target() const { return target_; }
    View<DrawRecord> records() const { return records_; }
    View<uint8_t> arena() const { return arena_; }
    const std::string& error() const { return error_; }

private:
    RenderTarget target_;
    std::vector<DrawRecord> records_;
    std::vector<uint8_t> arena_;
    std::string error_;
};

// ---- Import (design/renderer-core.md, "Zero-copy as a backend capability") -----------------

enum class Route : uint32_t { InPlace, CopiedOnGpu, CopiedFromHost, Refused };
const char* route_name(Route r);   // in-place, copied-on-gpu, copied-from-host, refused

enum class ExternalKind : uint32_t { MetalBuffer, HostMemory, Win32Memory, MetalTexture };

struct ExternalSync {
    enum Kind : uint32_t { Drained, MetalEvent, CudaStream } kind = Drained;
    void* object = nullptr;        // id<MTLSharedEvent>, or a CUstream
    uint64_t value = 0;            // the event value the producer signals
};

struct ExternalBuffer {
    ExternalKind kind = ExternalKind::HostMemory;
    void* handle = nullptr;        // id<MTLBuffer>, a page-aligned host pointer, or a Win32 HANDLE (duplicated)
    uint64_t size = 0;
    uint32_t usage = 0;            // kVertex | kIndex | kStorage | kCopySrc
    bool allow_copy = true;        // false: refuse rather than copy (geometry service applets ask for this)
    const char* label = nullptr;
};
struct ExternalTexture {
    ExternalKind kind = ExternalKind::MetalTexture;
    void* handle = nullptr;
    uint32_t width = 0, height = 0;
    Format format = Format::RGBA8Unorm;
    uint32_t usage = 0;
};

struct ImportResult {
    Buffer buffer;                 // the one that was imported
    Texture texture;
    Route route = Route::Refused;
    uint64_t bytes_copied = 0;     // 0 in place
    const char* path = "refused";  // the label the view's info line shows: route_name(route)
    std::string reason;            // why it copied or refused
};

// ---- Frames ----------------------------------------------------------------------------------

struct WaitPass { ExternalSync sync; };
struct DispatchPass {
    Pipeline pipeline;
    BindGroup groups[kMaxBindGroups];
    uint32_t dynamic_offsets[kMaxDynamicOffsets] = {};   // kArena bindings: the Frame's arena
    uint32_t x = 1, y = 1, z = 1;  // workgroups
    std::vector<Texture> writes;   // what it writes, for the barriers a Vulkan backend derives
};
struct CopyPass { Buffer src; uint64_t offset = 0; uint32_t row_pitch = 0; Texture dst; };   // the whole texture
struct RenderPass { DrawList list; };
using Pass = std::variant<WaitPass, DispatchPass, CopyPass, RenderPass>;

// The passes of one submit, in order; Vulkan derives its barriers from them.
class Frame {
public:
    void wait(const ExternalSync& sync);             // honour a producer's event before the next pass
    void dispatch(Pipeline p, View<BindGroup> groups, View<uint32_t> offsets,
                  uint32_t x, uint32_t y, uint32_t z, View<Texture> writes = {});
    void copy_buffer_to_texture(Buffer src, uint64_t offset, uint32_t row_pitch, Texture dst);
    void render(DrawList&& list);
    uint32_t uniform(const void* data, uint32_t size);   // the frame's arena, for dispatches

    View<Pass> passes() const { return passes_; }
    View<uint8_t> arena() const { return arena_; }
    const std::string& error() const { return error_; }

private:
    std::vector<Pass> passes_;
    std::vector<uint8_t> arena_;
    std::string error_;
};

// ---- Results of the operations built on the interface ----------------------------------------

struct Stats {   // since the last take_stats; the frame's sum feeds the info line and the benchmarks
    uint32_t submits = 0, imports_in_place = 0, imports_copied = 0, copies = 0, draws = 0, dispatches = 0;
    uint64_t bytes_copied_from_host = 0, bytes_copied_on_gpu = 0;
    double encode_cpu_ms = 0, gpu_ms = -1;   // gpu_ms < 0 without kCapTimestamps
};

struct TensorSource {
    Buffer buffer; uint64_t offset = 0; uint32_t w = 0, h = 0, channels = 1;
    CaliperDType dtype = CALIPER_DT_F32;
    int64_t stride_x = 1, stride_y = 0;   // in elements
    const CaliperTensor* tensor = nullptr;
    ExternalSync sync;
};
struct TextureMapping { const uint32_t* lut256 = nullptr; float vmin = 0, vmax = 1; };   // ignored for u8 data
struct UpdateResult { bool ok = false; Route route = Route::Refused; const char* path = "refused"; Serial serial = 0; };

// Backend-private records the core keeps per resource (defined in backend_impl.h).
struct ProgramInfo;
struct PipelineInfo;
struct BindGroupInfo;
struct BufferInfo;
struct TextureInfo;
class Core;
class CompletionSink;

// ---- The backend interface -------------------------------------------------------------------

class Backend {
public:
    virtual ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    const Caps& caps() const;

    Buffer    create_buffer(const BufferDesc& desc);
    Texture   create_texture(const TextureDesc& desc);
    Sampler   create_sampler(const SamplerDesc& desc);
    Program   create_program(const ProgramDesc& desc);
    Pipeline  create_render_pipeline(const RenderPipelineDesc& desc);   // cached by descriptor (decision 11)
    Pipeline  create_compute_pipeline(const ComputePipelineDesc& desc);
    BindGroup create_bind_group(const BindGroupDesc& desc);
    // The handle is dead at once; the object is destroyed once the last submit that used it
    // completed. Releasing a Readback cancels it.
    void release(uint64_t handle);
    template <class Tag> void release(Handle<Tag> h) { release(h.id); }

    // Ordered before the next submit, after the submits already made (queue order).
    bool  write_buffer(Buffer b, uint64_t offset, const void* data, uint64_t size);
    bool  write_texture(Texture t, const void* data, uint32_t row_pitch);
    void* mapped(Buffer b);                          // with kCapPersistentMap and host_visible, else nullptr

    ImportResult import_buffer(const ExternalBuffer& ext);
    ImportResult import_texture(const ExternalTexture& ext);

    Serial submit(Frame&& frame);                    // 0: refused and nothing encoded (decision 13)
    bool   completed(Serial s) const;                // never blocks
    bool   wait(Serial s);                           // tests and WAIT_GPU; false without kCapBlockingWait
    Serial last_use(uint64_t handle) const;          // the last submit that read or wrote it; 0 none
    template <class Tag> Serial last_use(Handle<Tag> h) const { return last_use(h.id); }

    // Ordered after every submit made before it. take_readback is false until the bytes are
    // ready, then fills out (decision 14) once and frees the handle.
    Readback read_texture(Texture t, uint32_t x, uint32_t y, uint32_t w, uint32_t h);
    bool     take_readback(Readback r, std::vector<uint8_t>* out);

    uint64_t imgui_texture_id(Texture t);            // what ImGui::Image takes on this backend
    Stats    take_stats();

    // Operations with a portable default built from the calls above (decision 15). A backend
    // overrides one only for a faster private path with the same result.
    virtual UpdateResult update_texture_from_tensor(Texture dst, const TensorSource& src,
                                                    const TextureMapping& m);
    virtual UpdateResult fill_texture_from_buffer(Texture dst, Buffer src, uint64_t offset,
                                                  uint32_t row_pitch);

    const std::string& last_error() const;
    bool alive(uint64_t handle) const;               // a live handle of its kind
    template <class Tag> bool alive(Handle<Tag> h) const {
        return handle_kind(h.id) == HandleTraits<Tag>::kind && alive(h.id);
    }
    void poll();                                     // retires completed work; submit and completed call it

protected:
    explicit Backend(const Caps& caps);

    // Called by a backend when the submit with this serial completed on the GPU. gpu_ms < 0
    // when unknown. Serials may complete out of order. Completion handlers on other threads
    // hold the sink by shared pointer instead, so a late one never reaches a dead backend.
    void complete(Serial s, double gpu_ms = -1);
    std::shared_ptr<CompletionSink> completion_sink() const;
    // Read access to what the core keeps per handle, for the hooks.
    const ProgramInfo&   program_info(Program p) const;
    const PipelineInfo&  pipeline_info(Pipeline p) const;
    const BindGroupInfo& bind_group_info(BindGroup g) const;
    const BufferInfo&    buffer_info(Buffer b) const;
    const TextureInfo&   texture_info(Texture t) const;

    // ---- Hooks, one set per backend (decision 1). Each runs after the core validated the call
    // and allocated the handle; slot is handle_slot(handle). A hook that fails sets *error and
    // returns false, and the core frees the handle.
    virtual bool on_create_buffer(uint32_t slot, const BufferDesc& d, std::string* error) = 0;
    virtual bool on_create_texture(uint32_t slot, const TextureDesc& d, std::string* error) = 0;
    virtual bool on_create_sampler(uint32_t slot, const SamplerDesc& d, std::string* error) = 0;
    virtual bool on_create_program(uint32_t slot, const ProgramDesc& d, std::string* error) = 0;
    virtual bool on_create_render_pipeline(uint32_t slot, const RenderPipelineDesc& d, std::string* error) = 0;
    virtual bool on_create_compute_pipeline(uint32_t slot, const ComputePipelineDesc& d, std::string* error) = 0;
    virtual bool on_create_bind_group(uint32_t slot, const BindGroupDesc& d, std::string* error) = 0;
    virtual void on_destroy(uint64_t handle) = 0;    // after its last use completed
    // in_use: a submit that uses the resource has not completed, so the write must be ordered
    // on the queue; otherwise the backend may write the memory at once.
    virtual void on_write_buffer(uint32_t slot, uint64_t offset, const void* data, uint64_t size,
                                 bool in_use) = 0;
    virtual void on_write_texture(uint32_t slot, const void* data, uint32_t row_pitch, bool in_use) = 0;
    virtual void* on_mapped(uint32_t slot) = 0;
    // Fills route, bytes_copied and reason; the core sets the handle, path and the stats.
    virtual bool on_import_buffer(uint32_t slot, const ExternalBuffer& ext, ImportResult* r) = 0;
    virtual bool on_import_texture(uint32_t slot, const ExternalTexture& ext, ImportResult* r) = 0;
    // Encodes and commits a validated frame. Reports completion with complete(serial).
    virtual bool on_submit(Serial s, const Frame& frame, Stats* stats, std::string* error) = 0;
    virtual bool on_wait(Serial s) = 0;
    // Encodes a copy of the region, ordered after serial s - 1, and completes as serial s.
    virtual bool on_read_texture(Serial s, uint32_t readback_slot, uint32_t texture_slot,
                                 uint32_t x, uint32_t y, uint32_t w, uint32_t h, std::string* error) = 0;
    virtual void on_take_readback(uint32_t readback_slot, std::vector<uint8_t>* out) = 0;
    virtual uint64_t on_imgui_texture_id(uint32_t slot) = 0;

private:
    std::unique_ptr<Core> core_;
};

}  // namespace caliper_gfx
