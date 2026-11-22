// The producer runtime on Metal (caliper_producer.h, design/gpu-programs.md,
// design/producer-runtime.md): the graphics layer without a window on the
// device of the script's tensors, with its own command queue. It imports each
// tensor in place with the layer's read_only flag, checks a program with the
// sandbox of caliper_builder_programs and its own binder, translates it with
// caliper-wgsl, and runs one submit for each due program, waited for before
// caliper_producer_run() returns.
//
// Without the caliper-wgsl library, caliper_producer_compile() answers
// CALIPER_PRODUCER_UNSUPPORTED with program.compiler_unavailable. Without the
// sandbox library (a build without the builder), the runtime's own checks
// below still hold the read-only rule, the limits and the bindings.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "producer/caliper_producer.h"

#include "gfx/gfx.h"
#include "gfx/metal_backend.h"

#if defined(CALIPER_PRODUCER_HAVE_WGSL)
#include <caliper/wgsl.h>
#endif
#if defined(CALIPER_PRODUCER_HAVE_SANDBOX)
#include <caliper_builder/programs/sandbox.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace caliper_gfx;

namespace {

// ---- codes and limits (design/gpu-programs.md; the same strings and numbers as
// caliper_builder/programs/sandbox.h and spec/program.h, which a build without the
// builder lacks)
constexpr const char* kTooLarge = "program.too_large";
constexpr const char* kCompileFailed = "program.compile_failed";
constexpr const char* kNoEntry = "program.no_entry";
constexpr const char* kBindingUndeclared = "program.binding_undeclared";
constexpr const char* kWritesInput = "program.writes_input";
constexpr const char* kOutputMismatch = "program.output_mismatch";
constexpr const char* kOverLimits = "program.over_limits";
constexpr const char* kCompilerUnavailable = "program.compiler_unavailable";
constexpr const char* kUnknownInput = "program.unknown_input";
constexpr const char* kImportRefused = "program.import_refused";
constexpr const char* kOverMemory = "program.over_memory";
constexpr const char* kLayerRefused = "program.layer_refused";
constexpr const char* kOverTime = "program.over_time";
constexpr const char* kTimeout = "program.timeout";
constexpr const char* kDeviceLost = "program.device_lost";
constexpr const char* kUnavailable = "program.unavailable";
constexpr const char* kParamsMismatch = "program.params_mismatch";
constexpr uint64_t kTensorElementsMax = 1000000;
constexpr size_t kParamsMax = 16;

constexpr size_t kSourceBytes = 64 * 1024;
constexpr size_t kInputs = 8;
constexpr uint32_t kMaxCount = 1000000;
constexpr uint32_t kPixelsSide = 2048;
constexpr double kTimeMsMax = 50.0;
constexpr uint64_t kMemoryMax = uint64_t(512) << 20;
[[maybe_unused]] constexpr uint32_t kWorkgroupInvocations = 256;
constexpr uint32_t kWorkgroupsPerDimension = 65535;
constexpr uint32_t kFrameBytes = 32;
constexpr uint32_t kOverTimePasses = 3;
constexpr double kTimeoutFloorMs = 100.0;
constexpr double kTimeoutFactor = 10.0;

#if defined(CALIPER_PRODUCER_HAVE_SANDBOX)
namespace pl = caliper_builder::spec::program_limits;
static_assert(kSourceBytes == pl::kProgramSourceBytes && kInputs == pl::kInputs && kMaxCount == pl::kMaxCount &&
                  kPixelsSide == pl::kPixelsSide && kWorkgroupInvocations == pl::kWorkgroupInvocations &&
                  kFrameBytes == pl::kFrameUniformBytes && kOverTimePasses == pl::kOverTimePasses,
              "the runtime's limits are those of spec/program.h");
#endif

// The uniform `caliper` (design/gpu-programs.md, Frame Uniform).
struct CaliperFrame {
    uint32_t step, pass_index, max_count, width, height;
    float time_s;
    uint32_t every, pad;
};
static_assert(sizeof(CaliperFrame) == kFrameBytes, "CaliperFrame is 32 bytes");

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

char* dup_text(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

caliper_producer_status fail(caliper_producer_error* e, caliper_producer_status s, const char* code,
                             const std::string& message, uint32_t line = 0, uint32_t column = 0) {
    if (e) {   // the caller frees an earlier message (caliper_producer_error_free)
        e->status = s;
        e->code = code;
        e->line = line;
        e->column = column;
        e->message = dup_text(message);
    }
    return s;
}

// A static copy of a sandbox code: the error's code is never freed.
[[maybe_unused]] const char* static_code(const std::string& code) {
    static const char* const kKnown[] = {
        kTooLarge, kCompileFailed, kNoEntry, kBindingUndeclared, kWritesInput, kOutputMismatch, kOverLimits,
        kCompilerUnavailable, kUnknownInput, kImportRefused, kOverMemory, kLayerRefused, kOverTime, kTimeout,
        kDeviceLost, kUnavailable, kParamsMismatch, "program.directive_not_allowed", "program.needs_review"};
    for (const char* k : kKnown)
        if (code == k) return k;
    return "program.compile_failed";
}

uint64_t align4(uint64_t n) { return (n + 3) & ~uint64_t(3); }

enum class Kind { Storage, Uniform, Other };

struct Bound {   // one binding of the reflection, neutral of caliper-wgsl's header
    uint32_t group = 0, binding = 0, msl_index = 0;
    Kind kind = Kind::Other;
    bool read = false, write = false;
    uint64_t size = 0;
    std::string name;
};

struct Import {
    bool live = false;
    std::string name;
    Buffer buffer;
    uint64_t byte_offset = 0, byte_size = 0;
};

struct OutputArray {
    std::string name;
    Buffer buffer;
    uint32_t words = 1;
    uint64_t bytes = 0;
};

// One field of the uniform `params`: its name, its type (0 f32, 1 i32, 2 u32,
// as sandbox.h's ParamType) and its value now.
struct ParamSlot {
    std::string name;
    uint32_t type = 0;
    double value = 0;
};

struct Prog {
    bool live = false;
    std::string id;
    uint32_t kind = CALIPER_PRODUCER_GEOMETRY;
    uint32_t max_count = 0, width = 0, height = 0;
    uint64_t tensor_elements = 0;      // tensor only: the elements of `out`
    std::vector<ParamSlot> params;     // the fields of struct Params, in struct order
    Buffer params_buffer;              // the uniform `params`, host-visible; null without params
    std::vector<OutputArray> arrays;   // geometry arrays, the one array "pixels", or "out"
    Buffer count;                      // geometry only
    Program program;
    Pipeline pipeline;
    std::vector<BindGroup> groups;     // indexed by group number
    bool frame = false;                // binds the uniform `caliper` from the frame's arena
    uint32_t dispatch[3] = {1, 1, 1};
    std::vector<uint32_t> tensors;     // the import handles it binds
    double time_limit_ms = 4.0;
    uint64_t memory_bytes = 0;
    uint32_t state = CALIPER_PRODUCER_COMPILED;
    uint64_t runs = 0;
    int64_t last_step = -1;
    double last_pass_ms = -1.0;
    uint32_t over_time = 0;
    uint32_t valid = 0;                // min(count, max_count) after the last pass
    const char* stop_code = nullptr;
};

}  // namespace

namespace {

// Writes the values of `p`'s params into its uniform buffer, 4 bytes each in
// struct order: f32, or i32 and u32 rounded and clamped (sandbox.h,
// pack_params()). The pass that reads them has completed, because a pass
// waits before it returns.
void write_params(Backend& be, const Prog& p) {
    auto* m = static_cast<uint8_t*>(be.mapped(p.params_buffer));
    if (!m) return;
    for (size_t i = 0; i < p.params.size(); ++i) {
        const ParamSlot& s = p.params[i];
        uint32_t bits = 0;
        if (s.type == 0) {
            const float f = float(s.value);
            std::memcpy(&bits, &f, 4);
        } else if (s.type == 1) {
            const int32_t n = int32_t(std::llround(std::clamp(s.value, -2147483648.0, 2147483647.0)));
            std::memcpy(&bits, &n, 4);
        } else {
            bits = uint32_t(std::llround(std::clamp(s.value, 0.0, 4294967295.0)));
        }
        std::memcpy(m + i * 4, &bits, 4);
    }
}

}  // namespace

struct caliper_producer {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unique_ptr<Backend> be;
    std::vector<Import> imports;   // handle = index + 1
    std::vector<Prog> programs;    // handle = index + 1
    double created_ms = 0;
    const char* dead = nullptr;    // the code that stopped every program: a timeout or a lost device
};

namespace {

bool open_device(caliper_producer* rt, id<MTLDevice> device) {
    if (rt->be) return true;
    if (device == nil) return false;
    rt->device = device;
    rt->queue = [device newCommandQueue];
    if (rt->queue == nil) return false;
    rt->be = make_metal_backend((__bridge void*)rt->device, (__bridge void*)rt->queue);
    return rt->be != nullptr;
}

Import* import_of(caliper_producer* rt, uint32_t h) {
    if (h == 0 || h > rt->imports.size() || !rt->imports[h - 1].live) return nullptr;
    return &rt->imports[h - 1];
}

Prog* prog_of(caliper_producer* rt, uint32_t h) {
    if (h == 0 || h > rt->programs.size() || !rt->programs[h - 1].live) return nullptr;
    return &rt->programs[h - 1];
}

void free_prog(caliper_producer* rt, Prog& p) {
    if (!rt->be) return;
    for (BindGroup g : p.groups)
        if (g) rt->be->release(g);
    p.groups.clear();
    if (p.pipeline) rt->be->release(p.pipeline);
    if (p.program) rt->be->release(p.program);
    for (OutputArray& a : p.arrays)
        if (a.buffer) rt->be->release(a.buffer);
    p.arrays.clear();
    if (p.count) rt->be->release(p.count);
    if (p.params_buffer) rt->be->release(p.params_buffer);
    p.pipeline = {};
    p.program = {};
    p.count = {};
    p.params_buffer = {};
}

void stop(Prog& p, const char* code) {
    if (p.state == CALIPER_PRODUCER_STOPPED) return;
    p.state = CALIPER_PRODUCER_STOPPED;
    p.stop_code = code;
}

// A timeout or a lost device: every program stops, and the imports are released.
void stop_all(caliper_producer* rt, const char* code) {
    rt->dead = code;
    for (Prog& p : rt->programs)
        if (p.live) stop(p, code);
    if (!rt->be) return;
    for (Import& im : rt->imports)
        if (im.live) {
            rt->be->release(im.buffer);
            im.live = false;
        }
}

caliper_producer_status refuse_dead(caliper_producer* rt, caliper_producer_error* error) {
    const bool timeout = std::strcmp(rt->dead, kTimeout) == 0;
    return fail(error, timeout ? CALIPER_PRODUCER_TIMEOUT : CALIPER_PRODUCER_DEVICE_LOST, rt->dead,
                "The runtime stopped every program after a timeout or a lost device; create it again.");
}

bool is_array_name(const char* n) {
    return n && (!std::strcmp(n, "positions") || !std::strcmp(n, "colors") || !std::strcmp(n, "values") ||
                 !std::strcmp(n, "radii"));
}

// 1. The declaration: counts, sizes and limits inside their maximums.
caliper_producer_status check_desc(const caliper_producer_program_desc* d, caliper_producer_error* error) {
    if (!d->id || !*d->id) return fail(error, CALIPER_PRODUCER_ARGUMENT, kOverLimits, "The program has no id.");
    if (!d->wgsl) return fail(error, CALIPER_PRODUCER_ARGUMENT, kCompileFailed, "The program has no WGSL source.");
    if (d->input_count > kInputs || (d->input_count && !d->inputs))
        return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "A program has at most 8 inputs.");
    for (size_t i = 0; i < d->input_count; ++i)
        if (!d->inputs[i].var || !*d->inputs[i].var)
            return fail(error, CALIPER_PRODUCER_ARGUMENT, kOverLimits, "An input has no WGSL variable.");
    if (d->wgsl_size > kSourceBytes)
        return fail(error, CALIPER_PRODUCER_REFUSED, kTooLarge,
                    "The source is " + std::to_string(d->wgsl_size) + " bytes, above the limit of 65536.");
    if (d->output_kind == CALIPER_PRODUCER_GEOMETRY) {
        if (d->part != CALIPER_PRODUCER_POINTS && d->part != CALIPER_PRODUCER_LINES)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch, "A geometry output has the part points or lines.");
        if (d->max_count < 1 || d->max_count > kMaxCount)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "output.max_count is between 1 and 1000000.");
        if (d->array_count == 0 || !d->arrays)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch, "A geometry output needs the array positions.");
        bool positions = false;
        for (size_t i = 0; i < d->array_count; ++i) {
            if (!is_array_name(d->arrays[i]))
                return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                            "A geometry array is positions, colors, values or radii.");
            if (!std::strcmp(d->arrays[i], "radii") && d->part != CALIPER_PRODUCER_POINTS)
                return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch, "The array radii is for points only.");
            for (size_t j = 0; j < i; ++j)
                if (!std::strcmp(d->arrays[i], d->arrays[j]))
                    return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                                std::string("The array ") + d->arrays[i] + " is declared twice.");
            positions = positions || !std::strcmp(d->arrays[i], "positions");
        }
        if (!positions)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch, "A geometry output needs the array positions.");
    } else if (d->output_kind == CALIPER_PRODUCER_PIXELS) {
        if (d->width < 1 || d->width > kPixelsSide || d->height < 1 || d->height > kPixelsSide)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "A pixels output is 1 to 2048 pixels on each side.");
    } else if (d->output_kind == CALIPER_PRODUCER_TENSOR) {
        if (d->tensor_ndim < 1 || d->tensor_ndim > 4)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "A tensor output has 1 to 4 axes.");
        uint64_t n = 1;
        for (uint32_t i = 0; i < d->tensor_ndim; ++i) {
            if (d->tensor_shape[i] < 1 || d->tensor_shape[i] > int64_t(kTensorElementsMax))
                return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "Each axis of a tensor output is at least 1.");
            n *= uint64_t(d->tensor_shape[i]);
            if (n > kTensorElementsMax)
                return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits,
                            "A tensor output holds at most 1,000,000 elements.");
        }
    } else {
        return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch, "The output is of kind geometry, pixels or tensor.");
    }
    if (d->param_count > kParamsMax || (d->param_count && !d->params))
        return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "A program has at most 16 params.");
    for (size_t i = 0; i < d->param_count; ++i)
        if (!d->params[i].name || !*d->params[i].name || !std::isfinite(d->params[i].value))
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "Each param has a name and a finite value.");
    if (!(d->time_limit_ms > 0) || d->time_limit_ms > kTimeMsMax)
        return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "limits.time_ms is above 0 and at most 50.");
    if (d->memory_limit_bytes == 0 || d->memory_limit_bytes > kMemoryMax)
        return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "limits.memory_mb is between 1 and 512.");
    for (uint32_t w : d->workgroups)
        if (w > kWorkgroupsPerDimension)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, "Each workgroup count is at most 65535.");
    return CALIPER_PRODUCER_OK;
}

#if defined(CALIPER_PRODUCER_HAVE_SANDBOX)
namespace sb = caliper_builder::programs;
namespace sp = caliper_builder::spec;

sp::Program to_spec(const caliper_producer_program_desc* d, const std::vector<Import>& imports) {
    sp::Program p;
    p.id = d->id;
    p.wgsl.assign(d->wgsl, d->wgsl_size);
    if (d->entry) p.entry = std::string(d->entry);
    for (size_t i = 0; i < d->input_count; ++i) {
        const uint32_t h = d->inputs[i].tensor;
        const bool known = h > 0 && h <= imports.size() && imports[h - 1].live;
        p.inputs.push_back({d->inputs[i].var, std::string("live:") + (known ? imports[h - 1].name : d->inputs[i].var)});
    }
    if (d->output_kind == CALIPER_PRODUCER_PIXELS) {
        p.output.kind = sp::OutputKind::pixels;
        p.output.width = d->width;
        p.output.height = d->height;
    } else if (d->output_kind == CALIPER_PRODUCER_TENSOR) {
        p.output.kind = sp::OutputKind::tensor;
        for (uint32_t i = 0; i < d->tensor_ndim && i < 4; ++i) p.output.shape.push_back(d->tensor_shape[i]);
    } else {
        p.output.kind = sp::OutputKind::geometry;
        p.output.part = d->part == CALIPER_PRODUCER_LINES ? sp::GeometryPart::lines : sp::GeometryPart::points;
        p.output.max_count = d->max_count;
        for (size_t i = 0; i < d->array_count; ++i)
            if (auto a = sp::geometry_array_from_name(d->arrays[i])) p.output.arrays.push_back(*a);
        p.output.panel = "script";   // the sandbox's declaration check wants a panel; a pixels output has none
    }
    for (size_t i = 0; i < d->param_count; ++i) p.params[d->params[i].name] = d->params[i].value;
    p.limits.time_ms = d->time_limit_ms;
    p.limits.memory_mb = uint32_t(std::max<uint64_t>(1, d->memory_limit_bytes >> 20));
    if (d->workgroups[0] || d->workgroups[1] || d->workgroups[2])
        p.workgroups = std::array<uint32_t, 3>{d->workgroups[0], d->workgroups[1], d->workgroups[2]};
    return p;
}

// The sandbox's verdict. A check whose body this build does not have yet answers
// command.not_implemented; the runtime's own binder and the layer's read_only flag then
// hold the rule, thus that answer passes here.
bool sandbox_refused(const sb::Verdict& v, caliper_producer_error* error, caliper_producer_status* out) {
    if (v.ok() || v.code == "command.not_implemented") return false;
    *out = fail(error, CALIPER_PRODUCER_REFUSED, static_code(v.code), v.message, v.line, v.column);
    return true;
}
#endif

}  // namespace

extern "C" {

uint32_t caliper_producer_abi_version(void) { return CALIPER_PRODUCER_ABI_VERSION; }

caliper_producer_status caliper_producer_create(void* device, caliper_producer** out, caliper_producer_error* error) {
    if (!out) return fail(error, CALIPER_PRODUCER_ARGUMENT, kUnavailable, "The out pointer is NULL.");
    *out = nullptr;
    @autoreleasepool {
        auto rt = std::make_unique<caliper_producer>();
        rt->created_ms = now_ms();
        if (device) {
            id<MTLDevice> d = (__bridge id<MTLDevice>)device;
            if (![(id)d conformsToProtocol:@protocol(MTLDevice)])
                return fail(error, CALIPER_PRODUCER_ARGUMENT, kUnavailable, "The device is not an MTLDevice.");
            if (!open_device(rt.get(), d))
                return fail(error, CALIPER_PRODUCER_UNSUPPORTED, kUnavailable,
                            "The graphics layer could not open a Metal backend on the device.");
        }
        *out = rt.release();
    }
    return CALIPER_PRODUCER_OK;
}

caliper_producer_status caliper_producer_import(caliper_producer* rt, const caliper_producer_tensor* t,
                                                uint32_t* handle, caliper_producer_error* error) {
    if (!rt || !t || !handle) return fail(error, CALIPER_PRODUCER_ARGUMENT, kImportRefused, "A pointer is NULL.");
    *handle = 0;
    if (rt->dead) return refuse_dead(rt, error);
    const std::string name = t->name ? t->name : "";
    @autoreleasepool {
        id obj = (__bridge id)t->buffer;
        if (obj == nil || ![obj conformsToProtocol:@protocol(MTLBuffer)])
            return fail(error, CALIPER_PRODUCER_IMPORT_REFUSED, kImportRefused,
                        "The tensor " + name + " is not held by an MTLBuffer, thus it cannot be imported in place.");
        id<MTLBuffer> b = (id<MTLBuffer>)obj;
        if (!open_device(rt, b.device))
            return fail(error, CALIPER_PRODUCER_UNSUPPORTED, kUnavailable,
                        "The graphics layer could not open a Metal backend on the tensor's device.");
        if (b.device.registryID != rt->device.registryID)
            return fail(error, CALIPER_PRODUCER_IMPORT_REFUSED, kImportRefused,
                        "The tensor " + name + " is on another device than the runtime.");
        if (t->ndim > CALIPER_PRODUCER_MAX_DIMS || t->dtype > CALIPER_PRODUCER_U8 || t->byte_size == 0)
            return fail(error, CALIPER_PRODUCER_ARGUMENT, kImportRefused,
                        "The tensor " + name + " has no bytes, too many dimensions or an unknown type.");
        const uint64_t end = t->byte_offset + t->byte_size;
        if (end < t->byte_offset || end > (uint64_t)b.length)
            return fail(error, CALIPER_PRODUCER_IMPORT_REFUSED, kImportRefused,
                        "The buffer of the tensor " + name + " is shorter than the tensor.");
        const uint32_t align = rt->be->caps().limits.min_storage_buffer_offset_alignment;
        if (align && t->byte_offset % align)
            return fail(error, CALIPER_PRODUCER_IMPORT_REFUSED, kImportRefused,
                        "The tensor " + name + " starts at byte " + std::to_string(t->byte_offset) +
                            " of its storage, which is not a multiple of " + std::to_string(align) + ".");
        ExternalBuffer ext;
        ext.kind = ExternalKind::MetalBuffer;
        ext.handle = t->buffer;
        ext.size = std::min<uint64_t>(align4(end), (uint64_t)b.length & ~uint64_t(3));
        ext.usage = kStorage;
        ext.allow_copy = false;   // in place, or refused: never a copy of a training tensor
        ext.read_only = true;     // the layer refuses it in a read_write binding
        ext.label = "training tensor";
        ImportResult r = rt->be->import_buffer(ext);
        if (!r.buffer || r.route != Route::InPlace || r.bytes_copied != 0) {
            if (r.buffer) rt->be->release(r.buffer);
            return fail(error, CALIPER_PRODUCER_IMPORT_REFUSED, kImportRefused,
                        "The layer refused the import of " + name + ": " + r.reason);
        }
        Import im;
        im.live = true;
        im.name = name;
        im.buffer = r.buffer;
        im.byte_offset = t->byte_offset;
        im.byte_size = t->byte_size;
        rt->imports.push_back(std::move(im));
        *handle = uint32_t(rt->imports.size());
    }
    return CALIPER_PRODUCER_OK;
}

void caliper_producer_release(caliper_producer* rt, uint32_t tensor) {
    if (!rt) return;
    Import* im = import_of(rt, tensor);
    if (!im) return;
    for (Prog& p : rt->programs)
        if (p.live && std::find(p.tensors.begin(), p.tensors.end(), tensor) != p.tensors.end())
            stop(p, kImportRefused);
    if (rt->be) rt->be->release(im->buffer);
    im->live = false;
    im->buffer = {};
}

caliper_producer_status caliper_producer_compile(caliper_producer* rt, const caliper_producer_program_desc* d,
                                                 uint32_t* handle, caliper_producer_error* error) {
    if (!rt || !d || !handle) return fail(error, CALIPER_PRODUCER_ARGUMENT, kCompileFailed, "A pointer is NULL.");
    *handle = 0;
    if (rt->dead) return refuse_dead(rt, error);
    // 1. The declaration.
    if (caliper_producer_status s = check_desc(d, error)) return s;
    const std::string wgsl(d->wgsl, d->wgsl_size);
#if defined(CALIPER_PRODUCER_HAVE_SANDBOX)
    const sp::Program spec = to_spec(d, rt->imports);
    caliper_producer_status refused = CALIPER_PRODUCER_OK;
    if (sandbox_refused(sb::check_declaration(spec), error, &refused)) return refused;
    // 3. The source: its size and no directive.
    if (sandbox_refused(sb::check_source(wgsl, kSourceBytes), error, &refused)) return refused;
#endif
    // The fields of struct Params, from the source (sandbox.h); the binder
    // below gives the uniform `params` these bytes.
    std::vector<ParamSlot> param_slots;
#if defined(CALIPER_PRODUCER_HAVE_SANDBOX)
    {
        std::vector<sb::ParamField> fields;
        sb::Verdict why;
        sb::params_fields(wgsl, &fields, &why);
        for (const sb::ParamField& f : fields) {
            ParamSlot slot{f.name, uint32_t(f.type), 0.0};
            for (size_t i = 0; i < d->param_count; ++i)
                if (f.name == d->params[i].name) slot.value = d->params[i].value;
            param_slots.push_back(std::move(slot));
        }
    }
#else
    if (d->param_count)
        return fail(error, CALIPER_PRODUCER_UNSUPPORTED, kParamsMismatch,
                    "This build of the runtime has no sandbox, which reads the struct Params, so params cannot run.");
#endif
#if !defined(CALIPER_PRODUCER_HAVE_WGSL)
    return fail(error, CALIPER_PRODUCER_UNSUPPORTED, kCompilerUnavailable,
                "This build of the runtime has no WGSL compiler, so a program cannot run.");
#else
    @autoreleasepool {
        // 4. The compile.
        caliper_wgsl_options opt;
        caliper_wgsl_options_init(&opt);
        opt.max_source_bytes = kSourceBytes;
        caliper_wgsl_module* mod = nullptr;
        caliper_wgsl_error we{};
        if (caliper_wgsl_module_create(wgsl.data(), wgsl.size(), &opt, &mod, &we) != CALIPER_WGSL_OK) {
            const std::string msg = we.message ? we.message : "The compiler refused the source.";
            const uint32_t line = we.line, column = we.column;
            caliper_wgsl_error_free(&we);
            return fail(error, CALIPER_PRODUCER_REFUSED, kCompileFailed, msg, line, column);
        }
        std::unique_ptr<caliper_wgsl_module, void (*)(caliper_wgsl_module*)> owner(mod, &caliper_wgsl_module_destroy);

        // 5. The entry point.
        std::string entry;
        uint32_t wg[3] = {1, 1, 1};
        size_t computes = 0;
        for (size_t i = 0, n = caliper_wgsl_module_entry_point_count(mod); i < n; ++i) {
            caliper_wgsl_entry_point ep{};
            if (caliper_wgsl_module_entry_point(mod, i, &ep) != CALIPER_WGSL_OK || ep.stage != CALIPER_WGSL_STAGE_COMPUTE)
                continue;
            ++computes;
            if ((d->entry && !std::strcmp(d->entry, ep.name)) || (!d->entry && computes == 1)) {
                entry = ep.name;
                std::memcpy(wg, ep.workgroup_size, sizeof wg);
            }
        }
        if (entry.empty() || (!d->entry && computes != 1))
            return fail(error, CALIPER_PRODUCER_REFUSED, kNoEntry,
                        d->entry ? std::string("The source has no compute entry point named ") + d->entry + "."
                                 : std::string("The source needs exactly one compute entry point, or `entry`."));
        if (uint64_t(wg[0]) * wg[1] * wg[2] > kWorkgroupInvocations)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits,
                        "The entry point's workgroup has more than 256 invocations.");

        // 6. The bindings: the sandbox's check, then the runtime's own binder.
        std::vector<Bound> bound;
        for (size_t i = 0, n = caliper_wgsl_module_binding_count(mod); i < n; ++i) {
            caliper_wgsl_binding wb{};
            if (caliper_wgsl_module_binding(mod, i, &wb) != CALIPER_WGSL_OK) continue;
            Bound b;
            b.group = wb.group;
            b.binding = wb.binding;
            b.msl_index = wb.metal_index;
            b.kind = wb.kind == CALIPER_WGSL_BINDING_STORAGE_BUFFER   ? Kind::Storage
                     : wb.kind == CALIPER_WGSL_BINDING_UNIFORM_BUFFER ? Kind::Uniform
                                                                      : Kind::Other;
            b.read = wb.access & CALIPER_WGSL_ACCESS_READ;
            b.write = wb.access & CALIPER_WGSL_ACCESS_WRITE;
            b.size = wb.size;
            b.name = wb.name ? wb.name : "";
            bound.push_back(std::move(b));
        }
#if defined(CALIPER_PRODUCER_HAVE_SANDBOX)
        {
            sb::Reflection refl;
            for (size_t i = 0, n = caliper_wgsl_module_binding_count(mod); i < n; ++i) {
                caliper_wgsl_binding wb{};
                if (caliper_wgsl_module_binding(mod, i, &wb) != CALIPER_WGSL_OK) continue;
                sb::ReflectedBinding rb;
                rb.group = wb.group;
                rb.binding = wb.binding;
                rb.kind = sb::BindingKind(wb.kind);
                rb.read = wb.access & CALIPER_WGSL_ACCESS_READ;
                rb.write = wb.access & CALIPER_WGSL_ACCESS_WRITE;
                rb.size = wb.size;
                rb.name = wb.name ? wb.name : "";
                if (wb.storage_format) rb.storage_format = wb.storage_format;
                refl.bindings.push_back(std::move(rb));
            }
            for (size_t i = 0, n = caliper_wgsl_module_entry_point_count(mod); i < n; ++i) {
                caliper_wgsl_entry_point ep{};
                if (caliper_wgsl_module_entry_point(mod, i, &ep) != CALIPER_WGSL_OK) continue;
                refl.entries.push_back({ep.name, sb::EntryStage(ep.stage),
                                        {ep.workgroup_size[0], ep.workgroup_size[1], ep.workgroup_size[2]}});
            }
            if (sandbox_refused(sb::check_program(spec, refl), error, &refused)) return refused;
        }
#endif
        const bool geometry = d->output_kind == CALIPER_PRODUCER_GEOMETRY;
        const bool tensor = d->output_kind == CALIPER_PRODUCER_TENSOR;
        uint64_t elements = 0;
        if (tensor) {
            elements = 1;
            for (uint32_t i = 0; i < d->tensor_ndim; ++i) elements *= uint64_t(d->tensor_shape[i]);
        }
        auto input_of = [&](const std::string& n) -> const caliper_producer_input* {
            for (size_t i = 0; i < d->input_count; ++i)
                if (n == d->inputs[i].var) return &d->inputs[i];
            return nullptr;
        };
        auto declared_array = [&](const std::string& n) {
            for (size_t i = 0; i < d->array_count; ++i)
                if (n == d->arrays[i]) return true;
            return false;
        };
        bool has_count = false, has_pixels = false, has_out = false, frame = false;
        std::vector<std::string> arrays_bound;
        std::vector<uint32_t> tensors;
        for (const Bound& b : bound) {
            const std::string where = " (" + b.name + ", group " + std::to_string(b.group) + " binding " +
                                      std::to_string(b.binding) + ")";
            if (const caliper_producer_input* in = input_of(b.name)) {
                // The read-only rule: an input is var<storage, read>, never anything that writes.
                if (b.kind != Kind::Storage || b.write)
                    return fail(error, CALIPER_PRODUCER_REFUSED, kWritesInput,
                                "The input " + b.name + " must be declared var<storage, read>, because a program "
                                "never writes a tensor of the script" + where + ".");
                if (!import_of(rt, in->tensor))
                    return fail(error, CALIPER_PRODUCER_REFUSED, kUnknownInput,
                                "The input " + b.name + " has no imported tensor.");
                if (std::find(tensors.begin(), tensors.end(), in->tensor) == tensors.end())
                    tensors.push_back(in->tensor);
                continue;
            }
            const bool output = geometry ? (declared_array(b.name) || b.name == "count")
                                : tensor ? b.name == "out"
                                         : b.name == "pixels";
            if (output) {
                if (b.kind != Kind::Storage || !b.write)
                    return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                                "The output " + b.name + " must be declared var<storage, read_write>" + where + ".");
                if (b.name == "count") has_count = true;
                else if (b.name == "pixels") has_pixels = true;
                else if (b.name == "out") has_out = true;
                else arrays_bound.push_back(b.name);
                continue;
            }
            if (b.name == "caliper") {
                if (b.kind != Kind::Uniform || b.size > kFrameBytes)
                    return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                                "The binding caliper must be var<uniform> of the 32-byte struct CaliperFrame" + where + ".");
                frame = true;
                continue;
            }
            if (b.name == "params" && !param_slots.empty()) {
                if (b.kind != Kind::Uniform || b.size > param_slots.size() * 4 + 15)
                    return fail(error, CALIPER_PRODUCER_REFUSED, kParamsMismatch,
                                "The binding params must be var<uniform> params: Params" + where + ".");
                continue;
            }
            return fail(error, CALIPER_PRODUCER_REFUSED, kBindingUndeclared,
                        "The binding " + b.name + " is not an input, an output, params or caliper" + where + ".");
        }
        if (geometry) {
            if (!has_count)
                return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                            "A geometry program must write var<storage, read_write> count: atomic<u32>.");
            for (size_t i = 0; i < d->array_count; ++i)
                if (std::find(arrays_bound.begin(), arrays_bound.end(), d->arrays[i]) == arrays_bound.end())
                    return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                                std::string("The declared array ") + d->arrays[i] + " is not written by the entry point.");
        } else if (tensor) {
            if (!has_out)
                return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                            "A tensor program must write var<storage, read_write> out: array<f32>.");
        } else if (!has_pixels) {
            return fail(error, CALIPER_PRODUCER_REFUSED, kOutputMismatch,
                        "A pixels program must write var<storage, read_write> pixels: array<u32>.");
        }

        // 8. The memory: the output buffers and the bound tensors.
        uint64_t memory = geometry ? 4 : tensor ? elements * 4 : uint64_t(d->width) * d->height * 4;
        if (geometry)
            for (size_t i = 0; i < d->array_count; ++i)
                memory += uint64_t(d->max_count) * (std::strcmp(d->arrays[i], "positions") ? 4 : 12);
        for (uint32_t h : tensors) memory += import_of(rt, h)->byte_size;
        if (memory > d->memory_limit_bytes)
            return fail(error, CALIPER_PRODUCER_OVER_BUDGET, kOverMemory,
                        "The outputs and the bound tensors take " + std::to_string(memory) +
                            " bytes, above limits.memory_mb of " + std::to_string(d->memory_limit_bytes) + " bytes.");

        // The dispatch: the declared counts, else derived from the output.
        uint32_t dispatch[3] = {d->workgroups[0], d->workgroups[1], d->workgroups[2]};
        if (!dispatch[0] && !dispatch[1] && !dispatch[2]) {
            if (geometry) {
                dispatch[0] = (d->max_count + wg[0] - 1) / wg[0];
                dispatch[1] = dispatch[2] = 1;
            } else if (tensor) {
                dispatch[0] = uint32_t(std::min<uint64_t>((elements + wg[0] - 1) / wg[0], UINT32_MAX));
                dispatch[1] = dispatch[2] = 1;
            } else {
                dispatch[0] = (d->width + wg[0] - 1) / wg[0];
                dispatch[1] = (d->height + wg[1] - 1) / wg[1];
                dispatch[2] = 1;
            }
        }
        for (uint32_t& x : dispatch) x = std::max(1u, x);
        if (dispatch[0] > kWorkgroupsPerDimension || dispatch[1] > kWorkgroupsPerDimension ||
            dispatch[2] > kWorkgroupsPerDimension)
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits,
                        "The derived workgroup count is above 65535; give `workgroups` or a larger workgroup size.");

        if (!rt->be && !open_device(rt, MTLCreateSystemDefaultDevice()))
            return fail(error, CALIPER_PRODUCER_UNSUPPORTED, kUnavailable, "This machine has no Metal device.");
        Backend& be = *rt->be;

        // The MSL of the entry point.
        caliper_wgsl_buffer msl{};
        if (caliper_wgsl_translate_msl(mod, entry.c_str(), CALIPER_WGSL_STAGE_COMPUTE, &msl, &we) != CALIPER_WGSL_OK) {
            const std::string msg = we.message ? we.message : "The MSL writer refused the program.";
            const uint32_t line = we.line, column = we.column;
            caliper_wgsl_error_free(&we);
            return fail(error, CALIPER_PRODUCER_REFUSED, kCompileFailed, msg, line, column);
        }
        const std::string msl_text(reinterpret_cast<const char*>(msl.data), msl.size);
        caliper_wgsl_buffer_free(&msl);

        // 9. The layer: the program, the pipeline, the outputs and the bind groups.
        Prog p;
        p.live = true;
        p.id = d->id;
        p.kind = d->output_kind;
        p.max_count = geometry ? d->max_count : 0;
        p.width = geometry || tensor ? 0 : d->width;
        p.height = geometry || tensor ? 0 : d->height;
        p.tensor_elements = elements;
        p.params = param_slots;
        p.frame = frame;
        std::memcpy(p.dispatch, dispatch, sizeof dispatch);
        p.tensors = tensors;
        p.time_limit_ms = d->time_limit_ms;
        p.memory_bytes = memory;
        auto layer_fail = [&](const std::string& what) {
            const std::string why = be.last_error();
            free_prog(rt, p);
            return fail(error, CALIPER_PRODUCER_REFUSED, kLayerRefused,
                        "The graphics layer refused the " + what + (why.empty() ? std::string(".") : ": " + why));
        };

        std::vector<BindingDesc> descs;
        for (const Bound& b : bound) {
            BindingDesc bd{};
            bd.group = uint8_t(b.group);
            bd.binding = uint8_t(b.binding);
            bd.type = b.kind == Kind::Uniform ? BindingType::Uniform
                      : b.write               ? BindingType::StorageReadWrite
                                              : BindingType::Storage;
            bd.stages = kStageCompute;
            bd.dynamic = b.name == "caliper";
            bd.size = uint32_t(b.name == "caliper" ? kFrameBytes : std::min<uint64_t>(b.size, UINT32_MAX));
            bd.msl_index = uint8_t(b.msl_index);
            descs.push_back(bd);
        }
        EntryDesc ed;
        ed.name = entry.c_str();
        ed.stage = kStageCompute;
        for (int i = 0; i < 3; ++i) ed.workgroup_size[i] = uint16_t(wg[i]);
        ProgramDesc pd;
        const std::string label = "program " + p.id;
        pd.label = label.c_str();
        pd.entries = &ed;
        pd.entry_count = 1;
        pd.bindings = descs.data();
        pd.binding_count = uint8_t(descs.size());
        pd.msl = msl_text.c_str();
        p.program = be.create_program(pd);
        if (!p.program) return layer_fail("program");
        ComputePipelineDesc cd{p.program, entry.c_str(), label.c_str()};
        p.pipeline = be.create_compute_pipeline(cd);
        if (!p.pipeline) return layer_fail("pipeline");

        auto make_output = [&](uint64_t bytes, const char* name) -> Buffer {
            BufferDesc bd;
            bd.size = std::max<uint64_t>(16, align4(bytes));
            bd.usage = kStorage | kCopySrc;
            bd.host_visible = true;
            bd.label = name;
            Buffer b = be.create_buffer(bd);
            if (b) {
                if (void* m = be.mapped(b)) std::memset(m, 0, size_t(bd.size));
            }
            return b;
        };
        if (geometry) {
            p.count = make_output(4, "count");
            if (!p.count) return layer_fail("counter");
            for (size_t i = 0; i < d->array_count; ++i) {
                OutputArray a;
                a.name = d->arrays[i];
                a.words = a.name == "positions" ? 3 : 1;
                a.bytes = uint64_t(d->max_count) * a.words * 4;
                a.buffer = make_output(a.bytes, d->arrays[i]);
                if (!a.buffer) return layer_fail("output " + a.name);
                p.arrays.push_back(std::move(a));
            }
        } else if (tensor) {
            OutputArray a;
            a.name = "out";
            a.bytes = elements * 4;
            a.buffer = make_output(a.bytes, "out");
            if (!a.buffer) return layer_fail("output out");
            p.arrays.push_back(std::move(a));
        } else {
            OutputArray a;
            a.name = "pixels";
            a.bytes = uint64_t(d->width) * d->height * 4;
            a.buffer = make_output(a.bytes, "pixels");
            if (!a.buffer) return layer_fail("output pixels");
            p.arrays.push_back(std::move(a));
        }

        if (!p.params.empty()) {
            BufferDesc bd;
            bd.size = std::max<uint64_t>(16, ((p.params.size() * 4 + 15) / 16) * 16);
            bd.usage = kUniform;
            bd.host_visible = true;
            bd.label = "params";
            p.params_buffer = be.create_buffer(bd);
            if (!p.params_buffer) return layer_fail("params buffer");
            write_params(be, p);
        }

        uint32_t groups = 0;
        for (const Bound& b : bound) groups = std::max(groups, b.group + 1);
        if (groups > kMaxBindGroups) return layer_fail("bind groups: a program uses groups 0 to 3");
        p.groups.assign(groups, BindGroup{});
        for (uint32_t g = 0; g < groups; ++g) {
            BindGroupDesc gd;
            gd.program = p.program;
            gd.group = g;
            gd.label = label.c_str();
            for (const Bound& b : bound) {
                if (b.group != g) continue;
                if (gd.entry_count >= kMaxBindingsPerGroup) return layer_fail("bind group: at most 8 bindings in a group");
                BindEntry& e = gd.entries[gd.entry_count++];
                e.binding = b.binding;
                if (const caliper_producer_input* in = input_of(b.name)) {
                    const Import* im = import_of(rt, in->tensor);
                    e.buffer = im->buffer;
                    e.offset = im->byte_offset;
                    e.size = align4(im->byte_size);
                } else if (b.name == "caliper") {
                    e.buffer = kArena;
                } else if (b.name == "count") {
                    e.buffer = p.count;
                } else if (b.name == "params" && p.params_buffer) {
                    e.buffer = p.params_buffer;
                    e.size = std::max<uint64_t>(16, ((p.params.size() * 4 + 15) / 16) * 16);
                } else {
                    for (const OutputArray& a : p.arrays)
                        if (a.name == b.name) e.buffer = a.buffer;
                }
            }
            if (gd.entry_count == 0) continue;
            p.groups[g] = be.create_bind_group(gd);
            if (!p.groups[g]) return layer_fail("bind group " + std::to_string(g));
        }
        rt->programs.push_back(std::move(p));
        *handle = uint32_t(rt->programs.size());
    }
    return CALIPER_PRODUCER_OK;
#endif
}

caliper_producer_status caliper_producer_run(caliper_producer* rt, const uint32_t* programs, size_t count,
                                             const caliper_producer_frame* frame, double timeout_ms,
                                             caliper_producer_pass* pass, caliper_producer_error* error) {
    if (pass) *pass = caliper_producer_pass{0, -1.0, 0, 0};
    if (!rt || (count && !programs) || !frame)
        return fail(error, CALIPER_PRODUCER_ARGUMENT, kUnavailable, "A pointer is NULL.");
    if (rt->dead) return refuse_dead(rt, error);
    if (!rt->be || count == 0) return CALIPER_PRODUCER_OK;
    Backend& be = *rt->be;
    double limits = 0;
    for (size_t i = 0; i < count; ++i)
        if (Prog* p = prog_of(rt, programs[i])) limits += p->time_limit_ms;
    if (!(timeout_ms > 0)) timeout_ms = std::max(kTimeoutFloorMs, kTimeoutFactor * limits);
    const double start = now_ms();
    const double deadline = start + timeout_ms;
    double gpu_sum = -1.0;
    (void)be.take_stats();   // the GPU times below are this pass's only
    for (size_t i = 0; i < count; ++i) {
        Prog* p = prog_of(rt, programs[i]);
        if (!p || p->state == CALIPER_PRODUCER_STOPPED) continue;
        @autoreleasepool {
            if (p->count) {
                if (auto* c = static_cast<uint32_t*>(be.mapped(p->count))) *c = 0;
            }
            Frame f;
            uint32_t offsets[1] = {0};
            uint32_t offset_count = 0;
            if (p->frame) {
                CaliperFrame cf{};
                cf.step = uint32_t(frame->step < 0 ? 0 : frame->step);
                cf.pass_index = uint32_t(p->runs);
                cf.max_count = p->max_count;
                cf.width = p->width;
                cf.height = p->height;
                cf.time_s = float(frame->time_s);
                cf.every = frame->every;
                offsets[0] = f.uniform(&cf, sizeof cf);
                offset_count = 1;
            }
            f.dispatch(p->pipeline, View<BindGroup>(p->groups.data(), p->groups.size()),
                       View<uint32_t>(offsets, offset_count), p->dispatch[0], p->dispatch[1], p->dispatch[2]);
            const Serial s = be.submit(std::move(f));
            if (s == 0) {
                stop(*p, kLayerRefused);
                if (pass) pass->stopped++;
                continue;
            }
            // Caliper's work complete: the wait at once, bounded by the deadline of the pass.
            int spins = 0;
            while (!be.completed(s)) {
                if (now_ms() > deadline) {
                    stop_all(rt, kTimeout);
                    if (pass) pass->wall_ms = now_ms() - start;
                    return fail(error, CALIPER_PRODUCER_TIMEOUT, kTimeout,
                                "The pass of " + p->id + " did not finish in " + std::to_string(timeout_ms) +
                                    " ms; every program stopped, and training continues without them.");
                }
                if (++spins < 2000) std::this_thread::yield();
                else std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            const Stats st = be.take_stats();
            p->last_pass_ms = st.gpu_submits > 0 ? st.gpu_ms : -1.0;
            if (p->last_pass_ms >= 0) gpu_sum = (gpu_sum < 0 ? 0 : gpu_sum) + p->last_pass_ms;
            p->runs++;
            p->last_step = frame->step;
            p->state = CALIPER_PRODUCER_RUNNING;
            if (p->count) {
                const auto* c = static_cast<const uint32_t*>(be.mapped(p->count));
                p->valid = c ? std::min(*c, p->max_count) : 0;
            } else if (p->kind == CALIPER_PRODUCER_TENSOR) {
                p->valid = uint32_t(p->tensor_elements);
            } else {
                p->valid = p->width * p->height;
            }
            if (pass) pass->ran++;
            // The time watchdog: over time_ms on 3 passes in a row stops the program.
            if (p->last_pass_ms > p->time_limit_ms) {
                if (++p->over_time >= kOverTimePasses) {
                    stop(*p, kOverTime);
                    if (pass) pass->stopped++;
                }
            } else if (p->last_pass_ms >= 0) {
                p->over_time = 0;
            }
        }
    }
    if (pass) {
        pass->wall_ms = now_ms() - start;
        pass->gpu_ms = gpu_sum;
    }
    return CALIPER_PRODUCER_OK;
}

caliper_producer_status caliper_producer_read_output(caliper_producer* rt, uint32_t program, const char* name,
                                                     caliper_producer_output* out, caliper_producer_error* error) {
    if (!rt || !name || !out) return fail(error, CALIPER_PRODUCER_ARGUMENT, kOutputMismatch, "A pointer is NULL.");
    *out = caliper_producer_output{};
    Prog* p = prog_of(rt, program);
    if (!p) return fail(error, CALIPER_PRODUCER_ARGUMENT, kOutputMismatch, "The program handle is unknown.");
    for (const OutputArray& a : p->arrays) {
        if (a.name != name) continue;
        out->buffer = metal_buffer(*rt->be, a.buffer);
        out->byte_offset = 0;
        out->count = p->runs ? p->valid : 0;
        out->words = a.words;
        out->byte_size = uint64_t(out->count) * a.words * 4;
        return CALIPER_PRODUCER_OK;
    }
    return fail(error, CALIPER_PRODUCER_ARGUMENT, kOutputMismatch,
                std::string("The program ") + p->id + " has no output " + name + ".");
}

caliper_producer_status caliper_producer_set_params(caliper_producer* rt, uint32_t program,
                                                    const caliper_producer_param* params, size_t count,
                                                    caliper_producer_error* error) {
    if (!rt || (count && !params)) return fail(error, CALIPER_PRODUCER_ARGUMENT, kParamsMismatch, "A pointer is NULL.");
    Prog* p = prog_of(rt, program);
    if (!p) return fail(error, CALIPER_PRODUCER_ARGUMENT, kParamsMismatch, "The program handle is unknown.");
    // Every name first, so that a refusal changes nothing.
    for (size_t i = 0; i < count; ++i) {
        const char* n = params[i].name;
        const bool known = n && std::any_of(p->params.begin(), p->params.end(),
                                            [&](const ParamSlot& s) { return s.name == n; });
        if (!known)
            return fail(error, CALIPER_PRODUCER_REFUSED, kParamsMismatch,
                        std::string("The program ") + p->id + " has no param " + (n ? n : "(null)") + ".");
        if (!std::isfinite(params[i].value))
            return fail(error, CALIPER_PRODUCER_REFUSED, kOverLimits, std::string("The param ") + n + " is not finite.");
    }
    for (size_t i = 0; i < count; ++i)
        for (ParamSlot& s : p->params)
            if (s.name == params[i].name) s.value = params[i].value;
    if (rt->be && p->params_buffer) write_params(*rt->be, *p);
    return CALIPER_PRODUCER_OK;
}

caliper_producer_status caliper_producer_read_status(caliper_producer* rt, uint32_t program,
                                                     caliper_producer_program_status* out) {
    if (!rt || !out) return CALIPER_PRODUCER_ARGUMENT;
    Prog* p = prog_of(rt, program);
    if (!p) return CALIPER_PRODUCER_ARGUMENT;
    out->state = p->state;
    out->runs = p->runs;
    out->last_step = p->last_step;
    out->last_pass_ms = p->last_pass_ms;
    out->over_time_passes = p->over_time;
    out->memory_bytes = p->memory_bytes;
    out->stop_code = p->stop_code;
    return CALIPER_PRODUCER_OK;
}

void caliper_producer_remove(caliper_producer* rt, uint32_t program) {
    if (!rt) return;
    Prog* p = prog_of(rt, program);
    if (!p) return;
    free_prog(rt, *p);
    p->live = false;
}

void caliper_producer_destroy(caliper_producer* rt) {
    if (!rt) return;
    @autoreleasepool {
        if (rt->be) {
            for (Prog& p : rt->programs)
                if (p.live) free_prog(rt, p);
            for (Import& im : rt->imports)
                if (im.live) rt->be->release(im.buffer);
            // The last submit may still read an import; the layer frees nothing under it.
            rt->be.reset();
        }
        rt->queue = nil;
        rt->device = nil;
        delete rt;
    }
}

void caliper_producer_error_free(caliper_producer_error* error) {
    if (!error) return;
    std::free(error->message);
    error->message = nullptr;
}

}  // extern "C"
