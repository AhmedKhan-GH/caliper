// caliper._producer: the producer runtime of GPU programs in the script's process
// (design/gpu-programs.md, design/producer-runtime.md), a nanobind module over the C
// interface src/host/producer/caliper_producer.h.
//
// The facade python/caliper/_programs.py loads it on the first program only. It imports a
// tensor of the script from its DLPack capsule in place and read-only, compiles a program,
// runs a pass of the due programs and waits for it, and gives each output as a DLPack capsule
// over the runtime's own buffer, thus no byte of a tensor or an output is copied here.
//
// Errors: a refusal raises ProducerError, a RuntimeError whose args are (code, message, line,
// column, status), with a stable code of sandbox.h such as "program.writes_input".
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "dlpack.h"
#include "producer/caliper_producer.h"

#include <array>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

// DLPack >= 1.0 managed tensor, as in _native.mm (the vendored dlpack.h predates it).
struct ProducerDLPackVersion { uint32_t major, minor; };
struct ProducerDLManagedTensorVersioned {
    ProducerDLPackVersion version;
    void* manager_ctx;
    void (*deleter)(ProducerDLManagedTensorVersioned* self);
    uint64_t flags;
    DLTensor dl_tensor;
};

namespace {

PyObject* g_error_type = nullptr;   // caliper._producer.ProducerError

[[noreturn]] void raise(caliper_producer_status s, caliper_producer_error& e) {
    const std::string code = e.code ? e.code : "program.unavailable";
    const std::string message = e.message ? e.message : "The producer runtime refused the call.";
    const uint32_t line = e.line, column = e.column;
    caliper_producer_error_free(&e);
    PyObject* args = Py_BuildValue("(ssIIi)", code.c_str(), message.c_str(), line, column, int(s));
    PyErr_SetObject(g_error_type, args);
    Py_XDECREF(args);
    throw nb::python_error();
}

[[noreturn]] void raise(caliper_producer_status s, const char* code, const std::string& message) {
    caliper_producer_error e{};
    e.code = code;
    e.message = static_cast<char*>(std::malloc(message.size() + 1));
    if (e.message) std::memcpy(e.message, message.c_str(), message.size() + 1);
    raise(s, e);
}

const DLTensor* tensor_of_capsule(nb::handle cap) {
    PyObject* o = cap.ptr();
    if (PyCapsule_IsValid(o, "dltensor"))
        return &static_cast<DLManagedTensor*>(PyCapsule_GetPointer(o, "dltensor"))->dl_tensor;
    if (PyCapsule_IsValid(o, "dltensor_versioned")) {
        auto* m = static_cast<ProducerDLManagedTensorVersioned*>(PyCapsule_GetPointer(o, "dltensor_versioned"));
        if (m->version.major > 1) return nullptr;
        return &m->dl_tensor;
    }
    return nullptr;
}

bool dtype_of(const DLDataType& t, uint32_t* out, uint32_t* item) {
    if (t.lanes != 1) return false;
    struct Row { uint8_t code, bits; uint32_t dtype; } const rows[] = {
        {kDLFloat, 32, CALIPER_PRODUCER_F32}, {kDLFloat, 16, CALIPER_PRODUCER_F16},
        {kDLBfloat, 16, CALIPER_PRODUCER_BF16}, {kDLInt, 32, CALIPER_PRODUCER_I32},
        {kDLUInt, 32, CALIPER_PRODUCER_U32}, {kDLInt, 64, CALIPER_PRODUCER_I64}, {kDLUInt, 8, CALIPER_PRODUCER_U8}};
    for (const Row& r : rows)
        if (t.code == r.code && t.bits == r.bits) {
            *out = r.dtype;
            *item = r.bits / 8;
            return true;
        }
    return false;
}

// A "dltensor" capsule over an output buffer: kDLMetal (data = the MTLBuffer) or kDLCPU (data =
// its contents in shared storage). The capsule's tensor retains the buffer.
struct OutputCtx {
    DLManagedTensor managed{};
    int64_t shape[3] = {0, 0, 0};
    int64_t strides[3] = {0, 0, 0};
    void* retained = nullptr;
};

void output_deleter(DLManagedTensor* m) {
    auto* ctx = static_cast<OutputCtx*>(m->manager_ctx);
    if (ctx->retained) CFRelease(ctx->retained);
    delete ctx;
}

void output_capsule_destructor(PyObject* cap) {
    if (PyCapsule_IsValid(cap, "dltensor")) {   // only an unconsumed capsule owns its tensor
        auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (m && m->deleter) m->deleter(m);
    }
}

nb::object output_capsule(id<MTLBuffer> buffer, uint64_t byte_offset, bool metal, DLDataType dtype,
                          const std::vector<int64_t>& shape) {
    auto* ctx = new OutputCtx();
    ctx->retained = (void*)CFBridgingRetain(buffer);
    int64_t stride = 1;
    for (int i = int(shape.size()) - 1; i >= 0; --i) {
        ctx->shape[i] = shape[i];
        ctx->strides[i] = stride;
        stride *= shape[i];
    }
    DLTensor& t = ctx->managed.dl_tensor;
    t.data = metal ? (__bridge void*)buffer : [buffer contents];
    t.device = DLDevice{metal ? kDLMetal : kDLCPU, 0};
    t.ndim = int(shape.size());
    t.dtype = dtype;
    t.shape = ctx->shape;
    t.strides = ctx->strides;
    t.byte_offset = byte_offset;
    ctx->managed.manager_ctx = ctx;
    ctx->managed.deleter = &output_deleter;
    PyObject* cap = PyCapsule_New(&ctx->managed, "dltensor", &output_capsule_destructor);
    if (!cap) {
        output_deleter(&ctx->managed);
        throw nb::python_error();
    }
    return nb::steal(cap);
}

uint32_t output_kind_of(const std::string& kind) {
    if (kind == "geometry") return CALIPER_PRODUCER_GEOMETRY;
    if (kind == "pixels") return CALIPER_PRODUCER_PIXELS;
    return 0;
}

}  // namespace

// The runtime of one process, on the device of the first bound tensor. Every call runs on the
// thread that holds the facade's lock; the runtime adds no thread.
class Runtime {
public:
    Runtime() {
        caliper_producer_error e{};
        const caliper_producer_status s = caliper_producer_create(nullptr, &rt_, &e);
        if (s != CALIPER_PRODUCER_OK) raise(s, e);
    }
    ~Runtime() { close(); }

    // Imports the tensor of a DLPack capsule (tensor.__dlpack__()) in place and read-only. The
    // runtime keeps the capsule, thus the framework cannot give the storage to another tensor
    // until release().
    uint32_t bind(const std::string& name, nb::object capsule) {
        alive();
        const DLTensor* dl = tensor_of_capsule(capsule);
        if (!dl) raise(CALIPER_PRODUCER_ARGUMENT, "program.import_refused",
                       "The input " + name + " is not an unconsumed DLPack capsule.");
        if (dl->device.device_type != kDLMetal)
            raise(CALIPER_PRODUCER_IMPORT_REFUSED, "program.import_refused",
                  "The tensor " + name + " is not on the Metal device (DLPack device type " +
                      std::to_string(int(dl->device.device_type)) + ").");
        uint32_t dtype = 0, item = 0;
        if (!dtype_of(dl->dtype, &dtype, &item))
            raise(CALIPER_PRODUCER_IMPORT_REFUSED, "program.import_refused",
                  "The tensor " + name + " has an element type that a program cannot read.");
        if (dl->ndim < 0 || dl->ndim > int(CALIPER_PRODUCER_MAX_DIMS))
            raise(CALIPER_PRODUCER_IMPORT_REFUSED, "program.import_refused",
                  "The tensor " + name + " has more than 8 dimensions.");
        int64_t n = 1;
        for (int i = 0; i < dl->ndim; ++i) n *= dl->shape[i];
        if (dl->strides) {
            int64_t expect = 1;
            for (int i = dl->ndim - 1; i >= 0; --i) {
                if (dl->shape[i] != 1 && dl->strides[i] != expect)
                    raise(CALIPER_PRODUCER_IMPORT_REFUSED, "program.import_refused",
                          "The tensor " + name + " is not contiguous, and the runtime never copies a tensor.");
                expect *= dl->shape[i];
            }
        }
        caliper_producer_tensor t{};
        t.name = name.c_str();
        t.buffer = dl->data;
        t.byte_offset = dl->byte_offset;
        t.byte_size = uint64_t(n) * item;
        t.dtype = dtype;
        t.ndim = uint32_t(dl->ndim);
        for (int i = 0; i < dl->ndim; ++i) t.shape[i] = dl->shape[i];
        uint32_t handle = 0;
        caliper_producer_error e{};
        const caliper_producer_status s = caliper_producer_import(rt_, &t, &handle, &e);
        if (s != CALIPER_PRODUCER_OK) {
            ++imports_refused_;
            raise(s, e);
        }
        capsules_[handle] = std::move(capsule);
        ++imports_in_place_;
        return handle;
    }

    // Releases an import and the capsule it holds; a program that binds it stops.
    void release(uint32_t handle) {
        if (rt_) caliper_producer_release(rt_, handle);
        capsules_.erase(handle);
    }

    // Checks and compiles a program; `inputs` pairs each WGSL variable with an import's handle.
    uint32_t program(const std::string& id, const std::string& wgsl,
                     const std::vector<std::pair<std::string, uint32_t>>& inputs, const std::string& kind,
                     std::optional<std::string> entry, const std::string& part, uint32_t max_count,
                     const std::vector<std::string>& arrays, uint32_t width, uint32_t height,
                     std::array<uint32_t, 3> workgroups, double time_ms, uint64_t memory_bytes) {
        alive();
        std::vector<caliper_producer_input> ins;
        for (const auto& [var, h] : inputs) ins.push_back({var.c_str(), h});
        std::vector<const char*> names;
        for (const std::string& a : arrays) names.push_back(a.c_str());
        caliper_producer_program_desc d{};
        d.id = id.c_str();
        d.wgsl = wgsl.data();
        d.wgsl_size = wgsl.size();
        d.entry = entry ? entry->c_str() : nullptr;
        d.inputs = ins.data();
        d.input_count = ins.size();
        d.output_kind = output_kind_of(kind);
        d.part = part == "lines" ? CALIPER_PRODUCER_LINES : part == "points" ? CALIPER_PRODUCER_POINTS : 0;
        d.max_count = max_count;
        d.arrays = names.data();
        d.array_count = names.size();
        d.width = width;
        d.height = height;
        std::memcpy(d.workgroups, workgroups.data(), sizeof d.workgroups);
        d.time_limit_ms = time_ms;
        d.memory_limit_bytes = memory_bytes;
        uint32_t handle = 0;
        caliper_producer_error e{};
        const caliper_producer_status s = caliper_producer_compile(rt_, &d, &handle, &e);
        if (s != CALIPER_PRODUCER_OK) raise(s, e);
        shapes_[handle] = {d.output_kind, width, height};
        return handle;
    }

    // One pass of `programs`, one submit each, waited for before it returns. A timeout raises
    // ProducerError with program.timeout after every program stopped.
    nb::dict run(const std::vector<uint32_t>& programs, int64_t step, double time_s, uint32_t every,
                 double timeout_ms) {
        alive();
        caliper_producer_frame frame{step, time_s, every};
        caliper_producer_pass pass{};
        caliper_producer_error e{};
        caliper_producer_status s;
        {
            nb::gil_scoped_release unlocked;   // the wait lets other Python threads run
            s = caliper_producer_run(rt_, programs.data(), programs.size(), &frame, timeout_ms, &pass, &e);
        }
        if (s != CALIPER_PRODUCER_OK) raise(s, e);
        ++passes_;
        nb::dict out;
        out["wall_ms"] = pass.wall_ms;
        out["gpu_ms"] = pass.gpu_ms < 0 ? nb::none() : nb::cast(pass.gpu_ms);
        out["ran"] = pass.ran;
        out["stopped"] = pass.stopped;
        return out;
    }

    // An output after the last pass as (capsule, count): positions float32 (count, 3), values
    // and radii float32 (count,), colors uint8 (count, 4), pixels uint8 (height, width, 4).
    // device "mps" gives kDLMetal, "cpu" the shared storage's contents as kDLCPU.
    std::pair<nb::object, uint32_t> output(uint32_t program, const std::string& name, const std::string& device) {
        alive();
        const bool metal = device == "mps" || device == "metal";
        if (!metal && device != "cpu")
            raise(CALIPER_PRODUCER_ARGUMENT, "program.output_mismatch", "device is 'mps' or 'cpu'.");
        caliper_producer_output o{};
        caliper_producer_error e{};
        const caliper_producer_status s = caliper_producer_read_output(rt_, program, name.c_str(), &o, &e);
        if (s != CALIPER_PRODUCER_OK) raise(s, e);
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)o.buffer;
        const Shape& sh = shapes_[program];
        std::vector<int64_t> shape;
        DLDataType dtype{kDLFloat, 32, 1};
        if (name == "pixels") {
            dtype = DLDataType{kDLUInt, 8, 1};
            shape = {int64_t(sh.height), int64_t(sh.width), 4};
        } else if (name == "colors") {
            dtype = DLDataType{kDLUInt, 8, 1};
            shape = {int64_t(o.count), 4};
        } else if (o.words == 3) {
            shape = {int64_t(o.count), 3};
        } else {
            shape = {int64_t(o.count)};
        }
        ++outputs_exported_;
        return {output_capsule(buffer, o.byte_offset, metal, dtype, shape), o.count};
    }

    // The state, times and memory of a program, as the C interface reports them.
    nb::dict status(uint32_t program) {
        caliper_producer_program_status st{};
        nb::dict out;
        if (!rt_ || caliper_producer_read_status(rt_, program, &st) != CALIPER_PRODUCER_OK) return out;
        out["state"] = st.state == CALIPER_PRODUCER_STOPPED ? "stopped"
                       : st.state == CALIPER_PRODUCER_RUNNING ? "running"
                                                              : "compiled";
        out["runs"] = st.runs;
        out["last_step"] = st.last_step < 0 ? nb::none() : nb::cast(st.last_step);
        out["last_pass_ms"] = st.last_pass_ms < 0 ? nb::none() : nb::cast(st.last_pass_ms);
        out["over_time_passes"] = st.over_time_passes;
        out["memory_bytes"] = st.memory_bytes;
        out["stop_code"] = st.stop_code ? nb::cast(std::string(st.stop_code)) : nb::none();
        return out;
    }

    void remove(uint32_t program) {
        if (rt_) caliper_producer_remove(rt_, program);
        shapes_.erase(program);
    }

    // The counters of the routes: each import in place, never a copied byte, and each output
    // given as a capsule over the runtime's own buffer.
    nb::dict stats() const {
        nb::dict out;
        out["imports"] = capsules_.size();
        out["imports_in_place"] = imports_in_place_;
        out["imports_refused"] = imports_refused_;
        out["bytes_copied"] = 0;   // an import that cannot be in place is refused, never copied
        out["outputs_exported"] = outputs_exported_;
        out["output_bytes_copied"] = 0;
        out["passes"] = passes_;
        return out;
    }

    // Stops every program, releases every import and frees the runtime; later calls raise.
    void close() {
        if (rt_) caliper_producer_destroy(rt_);
        rt_ = nullptr;
        capsules_.clear();
        shapes_.clear();
    }

    bool closed() const { return rt_ == nullptr; }

private:
    struct Shape { uint32_t kind = 0, width = 0, height = 0; };

    void alive() const {
        if (!rt_) raise(CALIPER_PRODUCER_ARGUMENT, "program.unavailable", "The runtime is closed.");
    }

    caliper_producer* rt_ = nullptr;
    std::map<uint32_t, nb::object> capsules_;
    std::map<uint32_t, Shape> shapes_;
    uint64_t imports_in_place_ = 0, imports_refused_ = 0, outputs_exported_ = 0, passes_ = 0;
};

NB_MODULE(_producer, m) {
    m.doc() = "The producer runtime of GPU programs (caliper_producer.h): in-place, read-only imports of the "
              "script's tensors, programs checked by the sandbox, passes and their outputs.";
    nb::set_leak_warnings(false);
    g_error_type = PyErr_NewExceptionWithDoc(
        "caliper._producer.ProducerError",
        "A refusal of the producer runtime; args are (code, message, line, column, status).", PyExc_RuntimeError,
        nullptr);
    m.attr("ProducerError") = nb::borrow(g_error_type);
    m.attr("ABI_VERSION") = caliper_producer_abi_version();

    nb::class_<Runtime>(m, "Runtime", "The producer runtime of this process, on the device of the first bound tensor.")
        .def(nb::init<>())
        .def("bind", &Runtime::bind, nb::arg("name"), nb::arg("capsule"),
             "Import a tensor's DLPack capsule in place and read-only; returns its handle.")
        .def("release", &Runtime::release, nb::arg("handle"), "Release an import and its capsule.")
        .def("program", &Runtime::program, nb::arg("id"), nb::arg("wgsl"), nb::arg("inputs"), nb::arg("kind"),
             nb::arg("entry") = nb::none(), nb::arg("part") = "points", nb::arg("max_count") = 0,
             nb::arg("arrays") = std::vector<std::string>{}, nb::arg("width") = 0, nb::arg("height") = 0,
             nb::arg("workgroups") = std::array<uint32_t, 3>{0, 0, 0}, nb::arg("time_ms") = 4.0,
             nb::arg("memory_bytes") = uint64_t(64) << 20,
             "Check and compile a program; returns its handle or raises ProducerError.")
        .def("run", &Runtime::run, nb::arg("programs"), nb::arg("step"), nb::arg("time_s"), nb::arg("every"),
             nb::arg("timeout_ms") = 0.0, "Run one pass of the programs and wait for it.")
        .def("output", &Runtime::output, nb::arg("program"), nb::arg("name"), nb::arg("device") = "mps",
             "An output as (DLPack capsule, count) over the runtime's buffer.")
        .def("status", &Runtime::status, nb::arg("program"), "The state, times and memory of a program.")
        .def("remove", &Runtime::remove, nb::arg("program"), "Remove a program and free its outputs.")
        .def("stats", &Runtime::stats, "The counters of imports, outputs and passes.")
        .def("close", &Runtime::close, "Stop every program, release every import and free the runtime.")
        .def_prop_ro("closed", &Runtime::closed, "Whether close() ran.");
}
