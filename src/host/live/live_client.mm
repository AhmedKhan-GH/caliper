// live_client.mm: the CLIENT half of the live protocol (caliper/live.h), used
// by the training process through python/src/_live.mm.
//
// A client owns one session: the control segment "/<session>-ctl", one segment
// per view and the session file <data_root>/live/<pid>.json that lets a viewer
// find it. It creates and unlinks every name; a viewer only maps. Each segment
// mapping is also wrapped as an MTLBuffer on the producer's device
// (newBufferWithBytesNoCopy, shared storage), which the Python glue hands to
// torch as kDLMetal DLPack tensors (one per slot payload / aux / stats region),
// so torch's own copy_/aminmax kernels write the shared pages directly.
//
// Lifetime: a segment is reference counted. The client's table row holds one
// reference; every DLPack capsule (and so every torch tensor aliasing a slot)
// holds another. remove_view/close unlink the NAME at once (a viewer that
// already mapped it keeps its own mapping), and the pages go away when the last
// reference drops. The MTLBuffer's deallocator does the munmap, so pages that a
// still-running GPU command buffer references outlive the segment object.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "caliper/live.h"
#include "live/live_shm.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace caliper_live;

namespace {

template <typename R> R fail(R r, std::string why) {
    live_err() = std::move(why);
    return r;
}

// Create, size and map a NEW shm object (O_EXCL: never adopt a live name).
// The fd is closed right away; the mapping keeps the object alive.
void* create_shm(const std::string& name, uint64_t len, int* err_no) {
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) { *err_no = errno; return nullptr; }
    if (ftruncate(fd, (off_t)len) != 0) {
        *err_no = errno;
        close(fd);
        shm_unlink(name.c_str());
        return nullptr;
    }
    void* p = mmap(nullptr, (size_t)len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        *err_no = errno;
        shm_unlink(name.c_str());
        return nullptr;
    }
    return p;
}

// Session-file paths held by open clients of THIS process: a second concurrent
// client (tests) gets "<pid>-<token>.json" instead of clobbering "<pid>.json".
std::mutex& files_mu() { static std::mutex m; return m; }
std::set<std::string>& files_open() { static std::set<std::string> s; return s; }

std::string default_token() {
    char b[24];
    std::snprintf(b, sizeof b, "clp-%d-%06x", (int)getpid(), arc4random_uniform(0x1000000u));
    return b;
}

}  // namespace

struct CaliperLiveSegment {
    std::atomic<int> refs{1};
    std::string      name;               // shm name, "/<session>-v<id>"
    std::string      view_name;
    void*            base = nullptr;     // this process's mapping
    uint64_t         len = 0;
    CaliperLiveSegmentHeader* hdr = nullptr;
    // MTLBuffer over [base, base+len). Its deallocator munmaps; nil on a
    // device-less machine (then the destructor munmaps).
    id<MTLBuffer>    buffer = nil;
    Layout           layout;
    uint32_t         kind = 0;
    int              table_index = -1;   // control-table row, -1 once removed
    uint32_t         inflight = 0;       // client-local: acquired, unpublished slots
    bool             removed = false;

    ~CaliperLiveSegment() {
        if (buffer != nil) buffer = nil;           // pages freed by the deallocator
        else if (base) munmap(base, (size_t)len);
    }
};

struct CaliperLiveClient {
    std::string          session;
    std::string          ctl_shm;
    std::string          file;               // session file path, "" without one
    SessionFile          meta;               // what the file says (views/segments refreshed)
    CaliperLiveControl*  ctl = nullptr;
    uint64_t             ctl_len = 0;
    id<MTLDevice>        device = nil;
    CaliperLiveSegment*  rows[CALIPER_LIVE_MAX_VIEWS] = {};
    uint64_t             next_view_id = 1;

    // Seqlock writer over ctl->views[] (the viewer retries a torn read).
    void table_begin() {
        const uint64_t g = __atomic_load_n(&ctl->generation, __ATOMIC_RELAXED);
        __atomic_store_n(&ctl->generation, g + 1, __ATOMIC_RELAXED);   // odd
        __atomic_thread_fence(__ATOMIC_RELEASE);
    }
    void table_end() {
        const uint64_t g = __atomic_load_n(&ctl->generation, __ATOMIC_RELAXED);
        __atomic_store_n(&ctl->generation, g + 1, __ATOMIC_RELEASE);   // even
    }

    // Rewrite the session file with the current views (best effort: a failed
    // rewrite leaves the previous, still valid, file).
    void write_file() {
        if (file.empty()) return;
        meta.views.clear();
        meta.segments.assign(1, ctl_shm);
        for (auto* s : rows) {
            if (!s) continue;
            meta.views.push_back(s->view_name);
            meta.segments.push_back(s->name);
        }
        write_session_file(file, meta, nullptr);
    }
};

extern "C" {

const char* caliper_live_last_error(void) { return live_err().c_str(); }
uint64_t caliper_live_now_ns(void) { return now_ns(); }

CaliperLiveClient* caliper_live_client_open(const CaliperLiveClientDesc* desc) {
    using D = CaliperLiveClientDesc;
    if (!desc || desc->struct_size < sizeof(D))
        return fail<CaliperLiveClient*>(nullptr, "live_client_open: null/short CaliperLiveClientDesc");
    const char* script = desc->script;
    const char* droot = desc->data_dir;
    const uint32_t cflags = desc->flags;

    std::string session = desc->session && desc->session[0] ? std::string(desc->session) : default_token();
    if (!valid_session(session.c_str()))
        return fail<CaliperLiveClient*>(nullptr,
            "live_client_open: session must be 1..20 chars of [A-Za-z0-9._-]; got '" + session + "'");

    const std::string ctl_shm = ctl_name(session);
    const uint64_t len = ctl_bytes();
    int err_no = 0;
    void* p = create_shm(ctl_shm, len, &err_no);
    if (!p && err_no == EEXIST) {
        // A leftover name. Adopt it only if its owner is gone (a crashed client
        // never unlinked); a live owner keeps its session.
        int fd = shm_open(ctl_shm.c_str(), O_RDONLY, 0);
        int32_t owner = 0;
        uint32_t state = 0;
        std::vector<std::string> stale;   // segment names the dead owner never unlinked
        if (fd >= 0) {
            struct stat st{};
            const bool big = fstat(fd, &st) == 0 && (uint64_t)st.st_size >= len;
            void* q = big ? mmap(nullptr, (size_t)len, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
            close(fd);
            if (q != MAP_FAILED) {
                auto* old = (const CaliperLiveControl*)q;
                if (old->magic == CALIPER_LIVE_CTL_MAGIC && old->struct_size >= sizeof(CaliperLiveControl)) {
                    owner = old->client_pid;
                    state = load_acquire(&old->state);
                    const std::string prefix = "/" + session + "-v";
                    for (const auto& e : old->views) {
                        if (e.state != CALIPER_LIVE_ENTRY_ACTIVE) continue;
                        std::string n(e.segment, strnlen(e.segment, sizeof e.segment));
                        if (n.compare(0, prefix.size(), prefix) == 0) stale.push_back(n);
                    }
                }
                munmap(q, (size_t)len);
            }
        }
        if (pid_alive(owner) && state != CALIPER_LIVE_SESSION_CLOSED)
            return fail<CaliperLiveClient*>(nullptr, "live_client_open: session '" + session +
                                                         "' is in use by pid " + std::to_string(owner));
        for (const auto& n : stale) shm_unlink(n.c_str());
        shm_unlink(ctl_shm.c_str());
        p = create_shm(ctl_shm, len, &err_no);
    }
    if (!p)
        return fail<CaliperLiveClient*>(nullptr, "live_client_open: shm_open('" + ctl_shm +
                                                     "') failed: " + std::strerror(err_no));

    auto* c = new CaliperLiveClient();
    c->session = session;
    c->ctl_shm = ctl_shm;
    c->ctl = (CaliperLiveControl*)p;
    c->ctl_len = len;
    std::memset(p, 0, (size_t)len);
    CaliperLiveControl* ctl = c->ctl;
    ctl->version = CALIPER_LIVE_PROTOCOL_VERSION;
    ctl->struct_size = sizeof(CaliperLiveControl);
    ctl->entry_size = sizeof(CaliperLiveViewEntry);
    ctl->max_views = CALIPER_LIVE_MAX_VIEWS;
    ctl->client_pid = (int32_t)getpid();
    ctl->start_unix_ms = unix_ms();
    std::snprintf(ctl->session, sizeof ctl->session, "%s", session.c_str());
    std::snprintf(ctl->script, sizeof ctl->script, "%s", script && script[0] ? script : getprogname());
    store_release(&ctl->state, (uint32_t)CALIPER_LIVE_SESSION_OPEN);
    store_release(&ctl->magic, (uint64_t)CALIPER_LIVE_CTL_MAGIC);   // last: "initialised"

    c->device = desc->mtl_device ? (__bridge id<MTLDevice>)desc->mtl_device
                                 : MTLCreateSystemDefaultDevice();

    if (!(cflags & CALIPER_LIVE_CLIENT_NO_SESSION_FILE)) {
        const std::string dir = sessions_dir(data_root(droot));
        if (!ensure_dir(dir)) {
            const std::string why = "live_client_open: cannot create '" + dir + "': " + std::strerror(errno);
            caliper_live_client_close(c);
            return fail<CaliperLiveClient*>(nullptr, why);
        }
        const std::string pid = std::to_string((int)getpid());
        {
            std::lock_guard<std::mutex> lk(files_mu());
            std::string path = dir + "/" + pid + ".json";
            if (files_open().count(path)) path = dir + "/" + pid + "-" + session + ".json";
            files_open().insert(path);
            c->file = path;
        }
        SessionFile& f = c->meta;
        f.protocol = (int32_t)CALIPER_LIVE_PROTOCOL_VERSION;
        f.pid = ctl->client_pid;
        f.proc_start_us = proc_start_us(f.pid);
        f.session = session;
        f.script = ctl->script;
        f.start_unix_ms = ctl->start_unix_ms;
        f.segments.assign(1, ctl_shm);
        std::string why;
        if (!write_session_file(c->file, f, &why)) {
            caliper_live_client_close(c);
            return fail<CaliperLiveClient*>(nullptr, "live_client_open: " + why);
        }
    }
    live_err().clear();
    return c;
}

const char* caliper_live_client_session(const CaliperLiveClient* c) {
    return c ? c->session.c_str() : "";
}

const char* caliper_live_client_session_file(const CaliperLiveClient* c) {
    return c ? c->file.c_str() : "";
}

int caliper_live_client_attached(const CaliperLiveClient* c) {
    if (!c) return 0;
    const uint64_t hb = load_acquire(&c->ctl->viewer_heartbeat_ns);
    if (hb == 0) return 0;
    const uint64_t now = now_ns();
    return (now < hb || now - hb < CALIPER_LIVE_ATTACH_WINDOW_NS) ? 1 : 0;
}

int caliper_live_client_viewer(const CaliperLiveClient* c, int32_t* viewer_pid, uint64_t* age_ns) {
    if (!c) return fail(0, "live_client_viewer: null client");
    const int32_t pid = load_acquire(&c->ctl->viewer_pid);
    const uint64_t hb = load_acquire(&c->ctl->viewer_heartbeat_ns);
    if (viewer_pid) *viewer_pid = pid;
    if (age_ns) {
        const uint64_t now = now_ns();
        *age_ns = hb == 0 ? UINT64_MAX : (now > hb ? now - hb : 0);
    }
    return 1;
}

CaliperLiveSegment* caliper_live_client_add_view(CaliperLiveClient* c,
                                                 const CaliperLiveViewDesc* d) {
    using R = CaliperLiveSegment*;
    using D = CaliperLiveViewDesc;
    if (!c) return fail<R>(nullptr, "live_add_view: null client");
    if (!d || d->struct_size < sizeof(D))
        return fail<R>(nullptr, "live_add_view: null/short CaliperLiveViewDesc");
    if (!d->name || !d->name[0] || std::strlen(d->name) >= CALIPER_LIVE_NAME_MAX)
        return fail<R>(nullptr, "live_add_view: name must be 1..63 bytes");
    const std::string name = d->name;
    auto vfail = [&](const std::string& why) { return fail<R>(nullptr, "live_add_view('" + name + "'): " + why); };
    const uint32_t want_slots = d->slots;
    const int64_t aux_count = d->aux_count;

    auto dims_ok = [&](int64_t lo, int64_t hi) {
        for (int i = 0; i < d->ndim; ++i)
            if (d->shape[i] < lo || d->shape[i] > hi) return false;
        return true;
    };
    switch (d->kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
            if (d->dtype != CALIPER_DT_F32 || d->ndim != 2) return vfail("a heatmap is a 2-D f32 (H,W)");
            if (!dims_ok(1, 16384)) return vfail("dims must be in 1..16384");
            break;
        case CALIPER_LIVE_KIND_IMAGE:
            if (d->dtype != CALIPER_DT_U8 || d->ndim != 3 || d->shape[2] != 4)
                return vfail("an image is a 3-D u8 (H,W,4) RGBA");
            if (!dims_ok(1, 16384)) return vfail("dims must be in 1..16384");
            break;
        case CALIPER_LIVE_KIND_POINTS:
            if (d->dtype != CALIPER_DT_F32 || d->ndim != 2 || d->shape[1] != 3)
                return vfail("points are a 2-D f32 (N,3) (2-D callers pad z)");
            if (d->shape[0] < 1 || d->shape[0] > (1 << 24)) return vfail("N must be in 1..16777216");
            if (aux_count != 0 && aux_count != d->shape[0]) return vfail("aux_count must be 0 or N");
            break;
        case CALIPER_LIVE_KIND_SERIES:
        case CALIPER_LIVE_KIND_HISTOGRAM:
            if (d->dtype != CALIPER_DT_F32 || d->ndim != 1)
                return vfail(d->kind == CALIPER_LIVE_KIND_SERIES ? "a series is a 1-D f32 (M,) ring"
                                                                 : "a histogram is a 1-D f32 (B,) of bin counts");
            if (d->shape[0] < 1 || d->shape[0] > (1 << 24)) return vfail("length must be in 1..16777216");
            break;
        default:
            return vfail("unknown view kind " + std::to_string(d->kind));
    }
    if (aux_count && d->kind != CALIPER_LIVE_KIND_POINTS) return vfail("only points carry an aux region");
    const uint32_t slots = kind_slots(d->kind);
    if (want_slots && want_slots != slots)
        return vfail("a view of this kind has " + std::to_string(slots) + " slot(s), not " +
                     std::to_string(want_slots));

    int row = -1;
    for (int i = 0; i < CALIPER_LIVE_MAX_VIEWS; ++i) {
        if (c->rows[i] && c->rows[i]->view_name == name)
            return vfail("a view with this name exists (remove it first)");
        if (!c->rows[i] && row < 0) row = i;
    }
    if (row < 0) return vfail("the session already has 64 views");
    if (c->next_view_id > kMaxViewId) return vfail("view id space exhausted for this session");

    uint64_t payload = dtype_bytes(d->dtype);
    for (int i = 0; i < d->ndim; ++i) payload *= (uint64_t)d->shape[i];
    const uint64_t aux = (uint64_t)aux_count * sizeof(float);
    const Layout L = make_layout(payload, aux, slots);
    const uint64_t id = c->next_view_id++;
    const std::string shm = seg_name(c->session, id);

    int err_no = 0;
    void* base = create_shm(shm, L.segment_bytes, &err_no);
    if (!base && err_no == EEXIST) {   // leaked by a crashed owner of this session
        shm_unlink(shm.c_str());
        base = create_shm(shm, L.segment_bytes, &err_no);
    }
    if (!base) return vfail("shm_open('" + shm + "') failed: " + std::strerror(err_no));

    auto* s = new CaliperLiveSegment();
    s->name = shm;
    s->view_name = name;
    s->base = base;
    s->len = L.segment_bytes;
    s->layout = L;
    s->kind = d->kind;
    s->hdr = (CaliperLiveSegmentHeader*)base;
    std::memset(base, 0, (size_t)L.header_bytes);   // ftruncate zero-fills the rest
    CaliperLiveSegmentHeader* h = s->hdr;
    h->magic = CALIPER_LIVE_SEG_MAGIC;
    h->version = CALIPER_LIVE_PROTOCOL_VERSION;
    h->struct_size = sizeof(CaliperLiveSegmentHeader);
    h->writer_pid = (int32_t)getpid();
    h->slot_count = slots;
    h->kind = d->kind;
    h->view_id = id;
    h->payload_bytes = L.payload_bytes;
    h->aux_bytes = L.aux_bytes;
    h->stats_bytes = L.stats_bytes;
    h->segment_bytes = L.segment_bytes;
    for (uint32_t i = 0; i < slots; ++i) {
        h->slot_offset[i] = L.slot_offset[i];
        h->aux_offset[i] = L.aux_offset[i];
        h->stats_offset[i] = L.stats_offset[i];
    }
    h->published_slot = CALIPER_LIVE_SLOT_NONE;
    h->viewer_prev_slot = CALIPER_LIVE_SLOT_NONE;
    store_release(&h->viewer_reading_slot, CALIPER_LIVE_SLOT_NONE);

    if (c->device != nil) {
        // One MTLBuffer over the whole mapping (page-aligned pointer and length,
        // as newBufferWithBytesNoCopy requires). No copy: torch writes the
        // shared pages through it. The deallocator unmaps once Metal (and every
        // command buffer that referenced the buffer) has let go.
        s->buffer = [c->device newBufferWithBytesNoCopy:base
                                                 length:(NSUInteger)L.segment_bytes
                                                options:MTLResourceStorageModeShared
                                            deallocator:^(void* ptr, NSUInteger n) {
                                                munmap(ptr, (size_t)n);
                                            }];
        if (s->buffer == nil) {
            shm_unlink(shm.c_str());
            delete s;   // buffer nil -> the destructor unmaps
            return vfail("newBufferWithBytesNoCopy over the shm mapping failed");
        }
    }

    c->table_begin();
    CaliperLiveViewEntry& e = c->ctl->views[row];
    std::memset(&e, 0, sizeof e);
    e.struct_size = sizeof(CaliperLiveViewEntry);
    e.kind = d->kind;
    e.dtype = d->dtype;
    e.ndim = (uint32_t)d->ndim;
    e.slot_count = slots;
    for (int i = 0; i < 4; ++i) e.shape[i] = i < d->ndim ? d->shape[i] : 0;
    e.view_id = id;
    e.slot_bytes = L.payload_bytes;
    e.stats_bytes = L.stats_bytes;
    e.segment_bytes = L.segment_bytes;
    e.aux_bytes = L.aux_bytes;
    e.aux_count = aux_count;
    std::snprintf(e.name, sizeof e.name, "%s", d->name);
    std::snprintf(e.segment, sizeof e.segment, "%s", shm.c_str());
    e.state = CALIPER_LIVE_ENTRY_ACTIVE;
    c->table_end();

    s->table_index = row;
    c->rows[row] = s;
    c->write_file();
    live_err().clear();
    return s;
}

static void unlink_row(CaliperLiveClient* c, int row) {
    CaliperLiveSegment* s = c->rows[row];
    if (!s) return;
    shm_unlink(s->name.c_str());
    s->removed = true;
    s->table_index = -1;
    c->rows[row] = nullptr;
    caliper_live_segment_release(s);
}

int caliper_live_client_remove_view(CaliperLiveClient* c, CaliperLiveSegment* s) {
    if (!c || !s) return fail(0, "live_remove_view: null client/segment");
    const int row = s->table_index;
    if (row < 0 || row >= CALIPER_LIVE_MAX_VIEWS || c->rows[row] != s)
        return fail(0, "live_remove_view: segment is not in this session");
    c->table_begin();
    std::memset(&c->ctl->views[row], 0, sizeof(CaliperLiveViewEntry));
    c->table_end();
    unlink_row(c, row);
    c->write_file();
    live_err().clear();
    return 1;
}

void caliper_live_client_close(CaliperLiveClient* c) {
    if (!c) return;
    // The file goes first: a lister never sees a closing session as live.
    if (!c->file.empty()) {
        unlink(c->file.c_str());
        std::lock_guard<std::mutex> lk(files_mu());
        files_open().erase(c->file);
    }
    store_release(&c->ctl->state, (uint32_t)CALIPER_LIVE_SESSION_CLOSED);
    c->table_begin();
    for (int i = 0; i < CALIPER_LIVE_MAX_VIEWS; ++i)
        if (c->rows[i]) c->ctl->views[i].state = CALIPER_LIVE_ENTRY_FREE;
    c->table_end();
    for (int i = 0; i < CALIPER_LIVE_MAX_VIEWS; ++i) unlink_row(c, i);
    shm_unlink(c->ctl_shm.c_str());
    munmap(c->ctl, (size_t)c->ctl_len);
    delete c;
}

void caliper_live_segment_retain(CaliperLiveSegment* s) {
    if (s) s->refs.fetch_add(1, std::memory_order_relaxed);
}

void caliper_live_segment_release(CaliperLiveSegment* s) {
    if (s && s->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete s;
}

int caliper_live_segment_info(CaliperLiveSegment* s, CaliperLiveSegmentInfo* o) {
    using I = CaliperLiveSegmentInfo;
    if (!s || !o || o->struct_size < sizeof(I))
        return fail(0, "live_segment_info: null segment or null/short CaliperLiveSegmentInfo");
    const CaliperLiveSegmentHeader* h = s->hdr;
    o->mtl_buffer = (__bridge void*)s->buffer;
    o->host_ptr = s->base;
    o->segment_bytes = s->len;
    o->payload_bytes = s->layout.payload_bytes;
    o->stats_bytes = s->layout.stats_bytes;
    for (int i = 0; i < CALIPER_LIVE_SLOTS; ++i) {
        o->slot_offset[i] = s->layout.slot_offset[i];
        o->stats_offset[i] = s->layout.stats_offset[i];
    }
    o->slot_count = s->layout.slot_count;
    o->table_index = s->table_index;
    o->segment_name = s->name.c_str();
    o->seq = load_acquire(&h->seq);
    o->published_slot = load_acquire(&h->published_slot);
    o->viewer_reading_slot = load_acquire(&h->viewer_reading_slot);
    o->dropped_client = load_acquire(&h->dropped_client);
    o->viewer_seq = load_acquire(&h->viewer_seq);
    o->skipped_viewer = load_acquire(&h->skipped_viewer);
    {
        o->kind = s->kind;
        o->viewer_prev_slot = load_acquire(&h->viewer_prev_slot);
        o->aux_bytes = s->layout.aux_bytes;
        for (int i = 0; i < CALIPER_LIVE_SLOTS; ++i) o->aux_offset[i] = s->layout.aux_offset[i];
    }
    return 1;
}

int caliper_live_segment_claims(CaliperLiveSegment* s, uint32_t* reading, uint32_t* prev) {
    if (!s) return fail(0, "live_claims: null segment");
    // seq_cst: pairs with the viewer's seq_cst claim (see live.h).
    if (reading) *reading = load_seqcst(&s->hdr->viewer_reading_slot);
    if (prev) *prev = load_seqcst(&s->hdr->viewer_prev_slot);
    return 1;
}

void caliper_live_segment_count_drop(CaliperLiveSegment* s) {
    if (s) fetch_add(&s->hdr->dropped_client, (uint64_t)1);
}

int caliper_live_segment_acquire(CaliperLiveSegment* s, int* busy) {
    if (busy) *busy = 0;
    if (!s || s->removed) return fail(-1, "live_acquire: segment removed");
    if (s->layout.slot_count == 1) return 0;   // SERIES: one append-only slot, never claimed
    CaliperLiveSegmentHeader* h = s->hdr;
    // seq_cst loads: pair with the viewer's seq_cst claims (see live.h). The
    // claim words are read after published_slot: the viewer moves prev before
    // reading, so a slot is never missed between the two loads.
    const uint32_t P = load_seqcst(&h->published_slot);
    const uint32_t R = load_seqcst(&h->viewer_reading_slot);
    const uint32_t R2 = load_seqcst(&h->viewer_prev_slot);
    for (int i = 0; i < CALIPER_LIVE_SLOTS; ++i) {
        const uint32_t u = (uint32_t)i;
        if (u == P || u == R || u == R2 || (s->inflight & (1u << i))) continue;
        s->inflight |= 1u << i;
        return i;
    }
    // Every candidate holds an unpublished snapshot of ours or a claim: drop
    // this one. (Overwriting an in-flight slot would be memory-safe on torch's
    // in-order stream, but under a sustained GPU backlog it starves publishing.)
    if (busy) *busy = 1;
    fetch_add(&h->dropped_client, (uint64_t)1);
    live_err() = "live_acquire: every free slot holds an unpublished snapshot or a viewer claim (busy)";
    return -1;
}

static int publish_impl(CaliperLiveSegment* s, int slot, const CaliperLivePublish* m, bool need_acquired) {
    using P = CaliperLivePublish;
    if (!s || s->removed) return fail(0, "live_publish: segment removed");
    if (!m || m->struct_size < sizeof(P))
        return fail(0, "live_publish: null/short CaliperLivePublish");
    if (slot < 0 || slot >= (int)s->layout.slot_count)
        return fail(0, "live_publish: slot " + std::to_string(slot) + " out of range");
    if (need_acquired && s->layout.slot_count > 1 && !(s->inflight & (1u << slot)))
        return fail(0, "live_publish: slot " + std::to_string(slot) + " was not acquired");
    CaliperLiveSegmentHeader* h = s->hdr;
    const uint64_t seq = h->seq + 1;   // the client is the only writer of seq
    CaliperLiveSlotMeta& meta = h->slot[slot];
    meta.seq = seq;
    meta.stamp_ns = m->stamp_ns;
    meta.colormap = m->colormap;
    meta.flags = m->flags;
    meta.vmin = m->vmin;
    meta.vmax = m->vmax;
    meta.series_total = m->series_total;
    meta.points_dims = m->points_dims;
    meta.label_count = m->label_count;
    meta.stats_count = m->stats_count > 0
                           ? std::min<int32_t>(m->stats_count, (int32_t)(h->stats_bytes / sizeof(float)))
                           : 2;
    store_seqcst(&h->published_slot, (uint32_t)slot);   // releases the meta above
    store_release(&h->stamp_ns, m->stamp_ns);
    store_release(&h->seq, seq);
    s->inflight &= ~(1u << slot);
    return 1;
}

int caliper_live_segment_publish(CaliperLiveSegment* s, int slot, const CaliperLivePublish* m) {
    return publish_impl(s, slot, m, /*need_acquired=*/true);
}

int caliper_live_segment_publish_slot(CaliperLiveSegment* s, int slot, const CaliperLivePublish* m) {
    return publish_impl(s, slot, m, /*need_acquired=*/false);
}

void caliper_live_segment_abandon(CaliperLiveSegment* s, int slot) {
    if (s && slot >= 0 && slot < CALIPER_LIVE_SLOTS) s->inflight &= ~(1u << slot);
}

}  // extern "C"
