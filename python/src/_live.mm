// caliper._live: nanobind binding of the live CLIENT API (caliper/live.h) for
// the training process, plus session discovery. Python glue:
// python/caliper/live.py (SessionTransport).
//
// What crosses into Python:
//   Client   one live session (the control segment + the session file);
//            attached() is the near-free "is a viewer watching" check.
//   Segment  one view's shared-memory segment (3 slots; SERIES 1).
//            region(role, slot, device, shape, dtype) returns a DLPack capsule
//            that torch.from_dlpack turns into a tensor ALIASING the shared
//            pages of that slot's payload / aux / stats region:
//              device "mps": kDLMetal, data = the segment's MTLBuffer,
//                            byte_offset = the region's offset (torch maps it
//                            to storage_offset), so copy_/aminmax(out=) run on
//                            torch's own stream straight into the segment;
//              device "cpu": kDLCPU over the CPU mapping.
//            claims() / publish_slot() are the slot protocol as the Python
//            ring uses it (live.h); acquire/publish/abandon the C-side choice.
//   list_sessions() / data_root(): caliper_live_list_sessions (reclaims stale
//            sessions as a side effect).
//
// DLPack, producer side: each capsule owns a heap DLManagedTensor whose
// manager_ctx holds a +1 reference on the segment, released by the deleter,
// i.e. when the consumer (torch) frees the tensor, or by the capsule
// destructor if nobody consumed it. So a torch tensor keeps the mapping (and
// the MTLBuffer) alive however long it lives, even past remove_view/close.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <caliper/live.h>
#include <caliper/tensor.h>

#include "dlpack.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

std::string live_error(const char* what) {
    const char* e = caliper_live_last_error();
    return (e && e[0]) ? std::string(e) : std::string(what) + " failed";
}

int64_t slot_or_minus(uint32_t s) { return s == CALIPER_LIVE_SLOT_NONE ? -1 : (int64_t)s; }

// --- DLPack producer ---------------------------------------------------------
struct CapsuleCtx {
    DLManagedTensor     managed{};
    int64_t             shape[4] = {};
    int64_t             strides[4] = {};
    CaliperLiveSegment* seg = nullptr;   // +1, released by the deleter
};

void ctx_deleter(DLManagedTensor* self) {
    auto* ctx = static_cast<CapsuleCtx*>(self->manager_ctx);
    caliper_live_segment_release(ctx->seg);
    delete ctx;
}

// Capsule destructor: if no consumer renamed it to "used_dltensor", we still
// own the managed tensor and must run its deleter (DLPack protocol).
void capsule_destructor(PyObject* cap) {
    if (!PyCapsule_IsValid(cap, "dltensor")) return;   // consumed (or not ours)
    PyObject *t, *v, *tb;
    PyErr_Fetch(&t, &v, &tb);
    auto* m = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
    if (m && m->deleter) m->deleter(m);
    PyErr_Restore(t, v, tb);
}

nb::object make_capsule(CaliperLiveSegment* seg, bool metal, void* mtl_buffer, uint8_t* host,
                        uint64_t offset, uint8_t code, uint8_t bits, const std::vector<int64_t>& shape) {
    auto* ctx = new CapsuleCtx();
    caliper_live_segment_retain(seg);
    ctx->seg = seg;
    const int nd = (int)shape.size();
    int64_t st = 1;
    for (int i = nd - 1; i >= 0; --i) { ctx->shape[i] = shape[i]; ctx->strides[i] = st; st *= shape[i]; }
    DLTensor& t = ctx->managed.dl_tensor;
    if (metal) {
        t.data = mtl_buffer;              // the MTLBuffer object (kDLMetal convention)
        t.device = {kDLMetal, 0};
        t.byte_offset = offset;
    } else {
        t.data = host + offset;
        t.device = {kDLCPU, 0};
        t.byte_offset = 0;
    }
    t.ndim = nd;
    t.dtype = {code, bits, 1};
    t.shape = ctx->shape;
    t.strides = ctx->strides;
    ctx->managed.manager_ctx = ctx;
    ctx->managed.deleter = &ctx_deleter;
    PyObject* cap = PyCapsule_New(&ctx->managed, "dltensor", &capsule_destructor);
    if (!cap) {
        ctx_deleter(&ctx->managed);
        throw nb::python_error();
    }
    return nb::steal(cap);
}

struct DT { uint8_t code; uint8_t bits; size_t bytes; };
DT parse_dtype(const std::string& s) {
    if (s == "float32") return {(uint8_t)kDLFloat, 32, 4};
    if (s == "uint8") return {(uint8_t)kDLUInt, 8, 1};
    throw std::invalid_argument("dtype must be 'float32' or 'uint8'");
}

// --- Segment -------------------------------------------------------------------
class Segment {
public:
    Segment(CaliperLiveSegment* s, std::string name, uint32_t kind, int32_t dtype,
            std::vector<int64_t> shape, int64_t aux_count)
        : seg_(s), name_(std::move(name)), kind_(kind), dtype_(dtype), shape_(std::move(shape)),
          aux_count_(aux_count) {
        caliper_live_segment_retain(seg_);
        refresh();
    }
    ~Segment() { caliper_live_segment_release(seg_); }
    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    CaliperLiveSegment* raw() const { return seg_; }

    // A tensor-shaped window on one region of one slot.
    nb::object region(const std::string& role, int slot, const std::string& device,
                      std::vector<int64_t> shape, const std::string& dtype) {
        check_slot(slot);
        const bool metal = want_metal(device);
        const DT dt = parse_dtype(dtype);
        if (shape.empty() || shape.size() > 4) throw std::invalid_argument("ndim must be 1..4");
        uint64_t bytes = dt.bytes;
        for (int64_t d : shape) {
            if (d < 1) throw std::invalid_argument("dims must be >= 1");
            bytes *= (uint64_t)d;
        }
        uint64_t off = 0, cap = 0;
        if (role == "payload") { off = info_.slot_offset[slot]; cap = info_.payload_bytes; }
        else if (role == "aux") { off = info_.aux_offset[slot]; cap = info_.aux_bytes; }
        else if (role == "stats") { off = info_.stats_offset[slot]; cap = info_.stats_bytes; }
        else throw std::invalid_argument("role must be 'payload', 'aux' or 'stats'");
        if (cap == 0) throw std::invalid_argument("view '" + name_ + "' has no " + role + " region");
        if (bytes > cap)
            throw std::invalid_argument(role + " region of view '" + name_ + "' holds " + std::to_string(cap) +
                                        " bytes; asked for " + std::to_string(bytes));
        return make_capsule(seg_, metal, info_.mtl_buffer, (uint8_t*)info_.host_ptr, off, dt.code, dt.bits, shape);
    }

    nb::object payload(int slot, const std::string& device) {
        return region("payload", slot, device, shape_, dtype_ == CALIPER_DT_F32 ? "float32" : "uint8");
    }

    nb::object stats(int slot, const std::string& device) {
        return region("stats", slot, device, {2}, "float32");
    }

    // A slot index, or -1 when busy (every candidate slot holds an in-flight
    // snapshot: drop this one). Raises only on a removed segment.
    int acquire() {
        int busy = 0;
        const int s = caliper_live_segment_acquire(seg_, &busy);
        if (s < 0 && !busy) throw std::runtime_error(live_error("acquire"));
        return s;
    }

    static CaliperLivePublish make_pub(uint64_t stamp_ns, int colormap, uint32_t flags, float vmin, float vmax,
                                       uint64_t series_total, int points_dims, int label_count, int stats_count) {
        CaliperLivePublish p{};
        p.struct_size = sizeof p;
        p.stamp_ns = stamp_ns;
        p.colormap = colormap;
        p.flags = flags;
        p.vmin = vmin;
        p.vmax = vmax;
        p.series_total = series_total;
        p.points_dims = points_dims;
        p.label_count = label_count;
        p.stats_count = stats_count;
        return p;
    }

    void publish(int slot, uint64_t stamp_ns, int colormap, uint32_t flags, float vmin, float vmax) {
        const CaliperLivePublish p = make_pub(stamp_ns, colormap, flags, vmin, vmax, 0, 0, 0, 0);
        if (!caliper_live_segment_publish(seg_, slot, &p))
            throw std::runtime_error(live_error("publish"));
    }

    void publish_slot(int slot, uint64_t stamp_ns, int colormap, uint32_t flags, float vmin, float vmax,
                      uint64_t series_total, int points_dims, int label_count, int stats_count) {
        const CaliperLivePublish p = make_pub(stamp_ns, colormap, flags, vmin, vmax, series_total, points_dims,
                                              label_count, stats_count);
        if (!caliper_live_segment_publish_slot(seg_, slot, &p))
            throw std::runtime_error(live_error("publish_slot"));
    }

    // (viewer_reading_slot, viewer_prev_slot), -1 = free. seq_cst loads.
    std::tuple<int64_t, int64_t> claims() const {
        uint32_t r = CALIPER_LIVE_SLOT_NONE, p = CALIPER_LIVE_SLOT_NONE;
        caliper_live_segment_claims(seg_, &r, &p);
        return {slot_or_minus(r), slot_or_minus(p)};
    }

    // (published_slot or -1, seq): what the client last published.
    std::tuple<int64_t, uint64_t> published() const {
        auto* h = (const CaliperLiveSegmentHeader*)info_.host_ptr;
        return {slot_or_minus(__atomic_load_n(&h->published_slot, __ATOMIC_SEQ_CST)),
                __atomic_load_n(&h->seq, __ATOMIC_ACQUIRE)};
    }

    void count_drop() { caliper_live_segment_count_drop(seg_); }
    void abandon(int slot) { caliper_live_segment_abandon(seg_, slot); }

    // Tests only: act as a viewer's claims (seq_cst stores).
    void set_claims(int64_t reading, int64_t prev) {
        auto* h = (CaliperLiveSegmentHeader*)info_.host_ptr;
        __atomic_store_n(&h->viewer_prev_slot, prev < 0 ? CALIPER_LIVE_SLOT_NONE : (uint32_t)prev, __ATOMIC_SEQ_CST);
        __atomic_store_n(&h->viewer_reading_slot, reading < 0 ? CALIPER_LIVE_SLOT_NONE : (uint32_t)reading,
                         __ATOMIC_SEQ_CST);
    }

    nb::dict slot_meta(int slot) {
        check_slot(slot);
        auto* h = (const CaliperLiveSegmentHeader*)info_.host_ptr;
        const CaliperLiveSlotMeta& m = h->slot[slot];
        nb::dict d;
        d["seq"] = m.seq;
        d["stamp_ns"] = m.stamp_ns;
        d["colormap"] = m.colormap;
        d["flags"] = m.flags;
        d["vmin"] = m.vmin;
        d["vmax"] = m.vmax;
        d["series_total"] = m.series_total;
        d["points_dims"] = m.points_dims;
        d["label_count"] = m.label_count;
        d["stats_count"] = m.stats_count;
        return d;
    }

    nb::dict info() {
        refresh();
        nb::dict d;
        d["name"] = name_;
        d["kind"] = info_.kind;
        d["segment"] = std::string(info_.segment_name ? info_.segment_name : "");
        d["segment_bytes"] = info_.segment_bytes;
        d["payload_bytes"] = info_.payload_bytes;
        d["aux_bytes"] = info_.aux_bytes;
        d["stats_bytes"] = info_.stats_bytes;
        d["slot_count"] = info_.slot_count;
        d["slot_offsets"] = std::vector<uint64_t>(info_.slot_offset, info_.slot_offset + info_.slot_count);
        d["aux_offsets"] = std::vector<uint64_t>(info_.aux_offset, info_.aux_offset + info_.slot_count);
        d["stats_offsets"] = std::vector<uint64_t>(info_.stats_offset, info_.stats_offset + info_.slot_count);
        d["host_ptr"] = (uintptr_t)info_.host_ptr;
        d["mtl_buffer"] = (uintptr_t)info_.mtl_buffer;
        d["table_index"] = info_.table_index;
        d["seq"] = info_.seq;
        d["published_slot"] = slot_or_minus(info_.published_slot);
        d["viewer_reading_slot"] = slot_or_minus(info_.viewer_reading_slot);
        d["viewer_prev_slot"] = slot_or_minus(info_.viewer_prev_slot);
        d["dropped_client"] = info_.dropped_client;
        d["viewer_seq"] = info_.viewer_seq;
        d["skipped_viewer"] = info_.skipped_viewer;
        return d;
    }

    const std::string& name() const { return name_; }
    uint32_t kind() const { return kind_; }
    std::vector<int64_t> shape() const { return shape_; }
    int64_t aux_count() const { return aux_count_; }
    uint32_t slot_count() const { return info_.slot_count; }
    bool has_metal() const { return info_.mtl_buffer != nullptr; }

private:
    void refresh() {
        info_ = CaliperLiveSegmentInfo{};
        info_.struct_size = sizeof info_;
        if (!caliper_live_segment_info(seg_, &info_)) throw std::runtime_error(live_error("segment_info"));
    }
    void check_slot(int slot) const {
        if (slot < 0 || slot >= (int)info_.slot_count) throw std::out_of_range("slot out of range");
    }
    bool want_metal(const std::string& device) const {
        if (device == "mps") {
            if (!info_.mtl_buffer) throw std::runtime_error("no Metal device: the segment has no MTLBuffer");
            return true;
        }
        if (device == "cpu") return false;
        throw std::invalid_argument("device must be 'mps' or 'cpu'");
    }

    CaliperLiveSegment*    seg_ = nullptr;   // +1 (released in the destructor)
    std::string            name_;
    uint32_t               kind_ = 0;
    int32_t                dtype_ = 0;
    std::vector<int64_t>   shape_;
    int64_t                aux_count_ = 0;
    CaliperLiveSegmentInfo info_{};
};

// --- Client --------------------------------------------------------------------
class Client {
public:
    Client(std::optional<std::string> session, uintptr_t mtl_buffer_ptr, std::optional<std::string> script,
           std::optional<std::string> data_dir, bool session_file) {
        CaliperLiveClientDesc d{};
        d.struct_size = sizeof d;
        std::string s = session.value_or("");
        d.session = s.empty() ? nullptr : s.c_str();
        id<MTLDevice> dev = nil;
        if (mtl_buffer_ptr) {
            // The producer's device, read off one of its buffers (e.g. a torch
            // MPS tensor's storage MTLBuffer), so the segment MTLBuffers live
            // on exactly torch's device.
            id<MTLBuffer> b = (__bridge id<MTLBuffer>)(void*)mtl_buffer_ptr;
            dev = b.device;
        }
        d.mtl_device = (__bridge void*)dev;
        const std::string sc = script.value_or("");
        const std::string dd = data_dir.value_or("");
        d.script = sc.empty() ? nullptr : sc.c_str();
        d.data_dir = dd.empty() ? nullptr : dd.c_str();
        d.flags = session_file ? 0u : (uint32_t)CALIPER_LIVE_CLIENT_NO_SESSION_FILE;
        c_ = caliper_live_client_open(&d);
        if (!c_) throw std::runtime_error(live_error("live client open"));
        device_name_ = dev ? std::string(dev.name.UTF8String) : std::string();
        registry_id_ = dev ? dev.registryID : 0;
    }
    ~Client() { close(); }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    std::string session() const { check(); return caliper_live_client_session(c_); }
    std::string session_file() const { check(); return caliper_live_client_session_file(c_); }

    // The near-free check: one atomic load + one clock read (False once closed).
    bool attached() const { return caliper_live_client_attached(c_) != 0; }

    Segment* add_view(const std::string& name, uint32_t kind, const std::string& dtype,
                      std::vector<int64_t> shape, uint32_t slots, int64_t aux_count) {
        check();
        CaliperLiveViewDesc v{};
        v.struct_size = sizeof v;
        v.name = name.c_str();
        v.kind = kind;
        if (dtype == "float32") v.dtype = CALIPER_DT_F32;
        else if (dtype == "uint8") v.dtype = CALIPER_DT_U8;
        else throw std::invalid_argument("dtype must be 'float32' or 'uint8'");
        if (shape.empty() || shape.size() > 4) throw std::invalid_argument("ndim must be 1..4");
        v.ndim = (int32_t)shape.size();
        for (size_t i = 0; i < shape.size(); ++i) v.shape[i] = shape[i];
        v.slots = slots;
        v.aux_count = aux_count;
        CaliperLiveSegment* s = caliper_live_client_add_view(c_, &v);
        if (!s) throw std::runtime_error(live_error("add_view"));
        return new Segment(s, name, kind, v.dtype, std::move(shape), aux_count);
    }

    void remove_view(Segment& s) {
        check();
        if (!caliper_live_client_remove_view(c_, s.raw()))
            throw std::runtime_error(live_error("remove_view"));
    }

    // (viewer_pid or 0, seconds since its last poll or None)
    std::tuple<int, std::optional<double>> viewer() const {
        check();
        int32_t pid = 0;
        uint64_t age = 0;
        caliper_live_client_viewer(c_, &pid, &age);
        std::optional<double> a;
        if (age != UINT64_MAX) a = (double)age * 1e-9;
        return {pid, a};
    }

    void close() {
        if (c_) {
            caliper_live_client_close(c_);
            c_ = nullptr;
        }
    }
    bool closed() const { return c_ == nullptr; }
    std::string device_name() const { return device_name_; }
    uint64_t registry_id() const { return registry_id_; }

private:
    void check() const {
        if (!c_) throw std::runtime_error("live client is closed");
    }
    CaliperLiveClient* c_ = nullptr;
    std::string        device_name_;
    uint64_t           registry_id_ = 0;
};

uint64_t registry_id_of_buffer(uintptr_t ptr) {
    if (!ptr) return 0;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)(void*)ptr;
    return b.device.registryID;
}

nb::list list_sessions(std::optional<std::string> data_root) {
    const std::string root = data_root.value_or("");
    const char* r = root.empty() ? nullptr : root.c_str();
    const int n = caliper_live_list_sessions_in(r, nullptr, 0);
    if (n < 0) throw std::runtime_error(live_error("list_sessions"));
    std::vector<CaliperLiveSessionInfo> v((size_t)n + 8);
    for (auto& e : v) e.struct_size = sizeof e;
    const int m = caliper_live_list_sessions_in(r, v.data(), (int)v.size());
    if (m < 0) throw std::runtime_error(live_error("list_sessions"));
    nb::list out;
    for (int i = 0; i < m && i < (int)v.size(); ++i) {
        const auto& e = v[(size_t)i];
        nb::dict d;
        d["pid"] = e.pid;
        d["alive"] = e.alive != 0;
        d["view_count"] = e.view_count;
        d["protocol"] = e.protocol;
        d["compatible"] = e.compatible != 0;
        d["start_unix_ms"] = e.start_unix_ms;
        d["viewer_pid"] = e.viewer_pid;
        d["session"] = std::string(e.session);
        d["script"] = std::string(e.script);
        d["path"] = std::string(e.path);
        out.append(d);
    }
    return out;
}

bool shm_exists(const std::string& name) {
    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) return false;
    close(fd);
    return true;
}

}  // namespace

NB_MODULE(_live, m) {
    m.doc() = "Caliper live-session client (caliper/live.h): shared-memory snapshot segments and session "
              "files a viewer process (caliper-viewer, the Caliper app) lists, attaches to and maps. "
              "Python glue: caliper.live.SessionTransport.";
    m.attr("PROTOCOL_VERSION") = CALIPER_LIVE_PROTOCOL_VERSION;
    m.attr("SLOTS") = CALIPER_LIVE_SLOTS;
    m.attr("MAX_VIEWS") = CALIPER_LIVE_MAX_VIEWS;
    m.attr("STATS_BYTES") = CALIPER_LIVE_STATS_BYTES;
    m.attr("ATTACH_WINDOW_NS") = (uint64_t)CALIPER_LIVE_ATTACH_WINDOW_NS;
    m.attr("KIND_HEATMAP") = (uint32_t)CALIPER_LIVE_KIND_HEATMAP;
    m.attr("KIND_IMAGE") = (uint32_t)CALIPER_LIVE_KIND_IMAGE;
    m.attr("KIND_POINTS") = (uint32_t)CALIPER_LIVE_KIND_POINTS;
    m.attr("KIND_SERIES") = (uint32_t)CALIPER_LIVE_KIND_SERIES;
    m.attr("KIND_HISTOGRAM") = (uint32_t)CALIPER_LIVE_KIND_HISTOGRAM;
    m.attr("SNAP_AUTO_VMIN") = (uint32_t)CALIPER_LIVE_SNAP_AUTO_VMIN;
    m.attr("SNAP_AUTO_VMAX") = (uint32_t)CALIPER_LIVE_SNAP_AUTO_VMAX;
    m.attr("SNAP_RAW") = (uint32_t)CALIPER_LIVE_SNAP_RAW;

    m.def("now_ns", []() { return caliper_live_now_ns(); },
          "CLOCK_UPTIME_RAW in ns (the clock of time.monotonic_ns() on macOS).");
    m.def("registry_id_of_buffer", &registry_id_of_buffer, "mtl_buffer_ptr"_a,
          "registryID of an MTLBuffer's device (e.g. a torch MPS storage data_ptr()).");
    m.def("data_root", []() { return std::string(caliper_live_data_root()); },
          "The default data root: $CALIPER_DATA_DIR, else Caliper's app-data folder. Sessions live in <root>/live.");
    m.def("list_sessions", &list_sessions, "data_root"_a = nb::none(),
          "Live sessions under <data_root>/live, newest first (stale ones are reclaimed, not listed).");
    m.def("shm_exists", &shm_exists, "name"_a, "Whether the POSIX shm name exists (tests/diagnostics).");

    nb::class_<Segment>(m, "Segment")
        .def("region", &Segment::region, "role"_a, "slot"_a, "device"_a, "shape"_a, "dtype"_a,
             "DLPack capsule aliasing one region ('payload' | 'aux' | 'stats') of a slot.")
        .def("payload", &Segment::payload, "slot"_a, "device"_a,
             "DLPack capsule aliasing the slot payload at the view's shape (device 'mps' or 'cpu').")
        .def("stats", &Segment::stats, "slot"_a, "device"_a,
             "DLPack capsule aliasing the slot's first two stats floats (vmin, vmax).")
        .def("acquire", &Segment::acquire, "slot index, or -1 when busy (drop this snapshot)")
        .def("publish", &Segment::publish, "slot"_a, "stamp_ns"_a, "colormap"_a, "flags"_a,
             "vmin"_a, "vmax"_a)
        .def("publish_slot", &Segment::publish_slot, "slot"_a, "stamp_ns"_a, "colormap"_a, "flags"_a,
             "vmin"_a, "vmax"_a, "series_total"_a = 0, "points_dims"_a = 0, "label_count"_a = 0,
             "stats_count"_a = 0, "Publish a slot the caller chose (its GPU writes completed).")
        .def("claims", &Segment::claims, "(viewer_reading_slot, viewer_prev_slot); -1 = free")
        .def("published", &Segment::published, "(published_slot or -1, seq)")
        .def("count_drop", &Segment::count_drop)
        .def("abandon", &Segment::abandon, "slot"_a)
        .def("slot_meta", &Segment::slot_meta, "slot"_a)
        .def("_set_claims", &Segment::set_claims, "reading"_a, "prev"_a, "Tests only: write the viewer's claims.")
        .def("info", &Segment::info)
        .def_prop_ro("name", &Segment::name)
        .def_prop_ro("kind", &Segment::kind)
        .def_prop_ro("shape", &Segment::shape)
        .def_prop_ro("aux_count", &Segment::aux_count)
        .def_prop_ro("slot_count", &Segment::slot_count)
        .def_prop_ro("has_metal", &Segment::has_metal);

    nb::class_<Client>(m, "Client")
        .def(nb::init<std::optional<std::string>, uintptr_t, std::optional<std::string>, std::optional<std::string>,
                      bool>(),
             "session"_a = nb::none(), "mtl_buffer_ptr"_a = 0, "script"_a = nb::none(), "data_dir"_a = nb::none(),
             "session_file"_a = true)
        .def("attached", &Client::attached,
             "True while a viewer heartbeat is younger than ATTACH_WINDOW_NS (one atomic load + clock read).")
        .def("add_view", &Client::add_view, "name"_a, "kind"_a, "dtype"_a, "shape"_a, "slots"_a = 0,
             "aux_count"_a = 0, nb::rv_policy::take_ownership)
        .def("remove_view", &Client::remove_view, "segment"_a)
        .def("viewer", &Client::viewer)
        .def("close", &Client::close)
        .def_prop_ro("session", &Client::session)
        .def_prop_ro("session_file", &Client::session_file)
        .def_prop_ro("closed", &Client::closed)
        .def_prop_ro("device_name", &Client::device_name)
        .def_prop_ro("registry_id", &Client::registry_id);
}
