// live_session.mm: the viewer-side shared-memory transport (a caliper_live::
// Source) and session discovery (caliper_live_list_sessions).
//
// THE SOURCE maps another process's live session (caliper/live.h) and hands
// live_core.cpp completed snapshots as "MTLBuffer + byte offsets". The viewer
// never creates or unlinks a name. It maps the control segment and each view
// segment read-write (it writes only the words that belong to it: the two
// claims, viewer_seq, skipped_viewer, the heartbeat) and wraps each view
// segment once as an MTLBuffer on the RENDERER's device, so the embed core's
// import (which refuses a buffer from another device) takes it as is.
// Unmapping is tied to the MTLBuffer's lifetime through its deallocator, so a
// view removed while a GPU pass still reads it stays mapped until that command
// buffer retires.
//
// DISCOVERY reads <root>/live/*.json (written by live_client.mm). A file whose
// pid is gone, or runs with another start time (pid reuse), is stale: the
// names it lists under its own "/<token>-" prefix (plus the active rows of its
// control table) are unlinked and the file is removed. A token another live
// client has since re-opened is left alone (only the stale file goes).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "caliper/live.h"
#include "live/live_shm.h"
#include "live/live_source.h"

#include <caliper/tensor.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace caliper_live {
namespace {

// Map an existing shm object read-write. len is what the caller expects; the
// object must be at least that large.
void* map_existing(const std::string& name, uint64_t len, std::string* why) {
    int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) {
        *why = "shm_open('" + name + "'): " + std::strerror(errno);
        return nullptr;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < len) {
        close(fd);
        *why = "shm '" + name + "' is smaller than its declared size";
        return nullptr;
    }
    void* p = mmap(nullptr, (size_t)len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        *why = "mmap('" + name + "'): " + std::strerror(errno);
        return nullptr;
    }
    return p;
}

std::string cstr(const char* p, size_t cap) { return std::string(p, strnlen(p, cap)); }

class ShmSource final : public Source {
public:
    ShmSource(std::string session, CaliperLiveControl* ctl, uint64_t ctl_len, id<MTLDevice> dev)
        : session_(std::move(session)), ctl_(ctl), ctl_len_(ctl_len), device_(dev) {
        pid_ = ctl_->client_pid;
        script_ = cstr(ctl_->script, sizeof ctl_->script);
        start_ms_ = ctl_->start_unix_ms;
    }

    ~ShmSource() override {
        // Give the producer every slot back and stop the heartbeat at once: its
        // attached() turns false now, not a window later.
        for (auto& m : maps_) {
            store_seqcst(&m.hdr->viewer_prev_slot, CALIPER_LIVE_SLOT_NONE);
            store_seqcst(&m.hdr->viewer_reading_slot, CALIPER_LIVE_SLOT_NONE);
        }
        if (load_acquire(&ctl_->viewer_pid) == (int32_t)getpid()) {
            store_release(&ctl_->viewer_heartbeat_ns, (uint64_t)0);
            store_release(&ctl_->viewer_pid, (int32_t)0);
        }
        maps_.clear();   // MTLBuffer deallocators unmap the view segments
        munmap(ctl_, (size_t)ctl_len_);
    }

    std::string label() const override { return session_; }
    std::string script() const override { return script_; }
    uint64_t start_unix_ms() const override { return start_ms_; }
    int32_t producer_pid() const override { return pid_; }

    bool producer_alive() override {
        if (load_acquire(&ctl_->state) == CALIPER_LIVE_SESSION_CLOSED) return false;
        return pid_alive(pid_);
    }

    void heartbeat() override {
        store_release(&ctl_->viewer_pid, (int32_t)getpid());
        store_release(&ctl_->viewer_heartbeat_ns, now_ns());
    }

    bool sync(std::vector<SourceView>* views) override {
        // Seqlock read of the table: equal, even generations around the copy.
        const uint64_t g1 = load_acquire(&ctl_->generation);
        if (g1 == gen_seen_ || (g1 & 1)) return false;
        std::vector<CaliperLiveViewEntry> table(CALIPER_LIVE_MAX_VIEWS);
        std::memcpy(table.data(), ctl_->views, sizeof(CaliperLiveViewEntry) * CALIPER_LIVE_MAX_VIEWS);
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        const uint64_t g2 = __atomic_load_n(&ctl_->generation, __ATOMIC_RELAXED);
        if (g1 != g2) return false;   // torn; retry next frame
        gen_seen_ = g1;
        // A closing client clears its table AFTER storing state = CLOSED: a
        // viewer keeps what it received, so a closed session's table is ignored.
        if (load_acquire(&ctl_->state) == CALIPER_LIVE_SESSION_CLOSED) return false;

        std::vector<Mapped> next;
        for (auto& e : table) {
            if (e.state != CALIPER_LIVE_ENTRY_ACTIVE) continue;
            e.name[CALIPER_LIVE_NAME_MAX - 1] = '\0';
            e.segment[CALIPER_LIVE_SEGNAME_MAX - 1] = '\0';
            Mapped* have = find(e.view_id);
            if (have) { next.push_back(std::move(*have)); continue; }
            Mapped m;
            if (map_view(e, &m)) next.push_back(std::move(m));
        }
        maps_ = std::move(next);   // dropped Mapped entries release their buffers

        views->clear();
        for (const auto& m : maps_) views->push_back({m.key, m.name, m.kind});
        return true;
    }

    bool take(uint64_t key, uint64_t after_seq, Snapshot* out) override {
        Mapped* m = find(key);
        if (!m) return false;
        CaliperLiveSegmentHeader* h = m->hdr;
        uint64_t seq = load_acquire(&h->seq);
        if (seq == 0 || seq == after_seq) return false;

        if (m->slot_count == 1) {
            // SERIES: one never-claimed slot. Read its meta between two equal
            // seq loads (the client rewrites it on every publish).
            CaliperLiveSlotMeta meta{};
            bool ok = false;
            for (int tries = 0; tries < 4 && !ok; ++tries) {
                meta = h->slot[0];
                __atomic_thread_fence(__ATOMIC_ACQUIRE);
                const uint64_t again = load_acquire(&h->seq);
                ok = again == seq && meta.seq == seq;
                seq = again;
            }
            if (!ok || meta.seq <= after_seq) return false;
            fill(*m, 0, meta, out);
            return true;
        }

        // Keep protecting what the view shows (a points view draws its payload
        // every frame) while claiming the newest: prev first, then the claim
        // (the viewer half of the Dekker pair in live.h).
        const uint32_t old = m->cur;
        store_seqcst(&h->viewer_prev_slot, old);
        m->prev = old;
        uint32_t s = CALIPER_LIVE_SLOT_NONE;
        for (int tries = 0; tries < 8; ++tries) {
            const uint32_t p = load_seqcst(&h->published_slot);
            if (p >= m->slot_count) break;
            store_seqcst(&h->viewer_reading_slot, p);
            if (load_seqcst(&h->published_slot) == p) { s = p; break; }
        }
        if (s == CALIPER_LIVE_SLOT_NONE) {   // client is publishing fast; next frame
            restore(*m);
            return false;
        }
        const CaliperLiveSlotMeta meta = h->slot[s];
        if (meta.seq <= after_seq) {
            restore(*m);
            return false;
        }
        m->cur = s;
        fill(*m, s, meta, out);
        return true;
    }

    void release_prev(uint64_t key) override {
        Mapped* m = find(key);
        if (!m || m->prev == CALIPER_LIVE_SLOT_NONE) return;
        store_seqcst(&m->hdr->viewer_prev_slot, CALIPER_LIVE_SLOT_NONE);
        m->prev = CALIPER_LIVE_SLOT_NONE;
    }

    void unclaim(uint64_t key) override {
        Mapped* m = find(key);
        if (!m || m->slot_count == 1) return;
        restore(*m);
    }

    void drew(uint64_t key, uint64_t seq, uint64_t skipped) override {
        Mapped* m = find(key);
        if (!m) return;
        store_release(&m->hdr->viewer_seq, seq);
        if (skipped) fetch_add(&m->hdr->skipped_viewer, skipped);
    }

    uint64_t producer_dropped(uint64_t key) override {
        Mapped* m = find(key);
        return m ? load_acquire(&m->hdr->dropped_client) : 0;
    }

private:
    struct Mapped {
        uint64_t    key = 0;
        std::string name;
        uint32_t    kind = 0;
        int32_t     dtype = 0;
        int32_t     ndim = 0;
        int64_t     shape[4] = {};
        int64_t     aux_count = 0;
        uint32_t    slot_count = CALIPER_LIVE_SLOTS;
        uint32_t    cur = CALIPER_LIVE_SLOT_NONE;    // our viewer_reading_slot
        uint32_t    prev = CALIPER_LIVE_SLOT_NONE;   // our viewer_prev_slot
        void*       base = nullptr;
        uint64_t    len = 0;
        CaliperLiveSegmentHeader* hdr = nullptr;
        id<MTLBuffer> buffer = nil;   // its deallocator unmaps [base, base+len)
    };

    // Back to the claims before the last take(): the old current claim is what
    // the view still shows.
    void restore(Mapped& m) {
        store_seqcst(&m.hdr->viewer_reading_slot, m.prev);
        store_seqcst(&m.hdr->viewer_prev_slot, CALIPER_LIVE_SLOT_NONE);
        m.cur = m.prev;
        m.prev = CALIPER_LIVE_SLOT_NONE;
    }

    void fill(const Mapped& m, uint32_t s, const CaliperLiveSlotMeta& meta, Snapshot* out) {
        const CaliperLiveSegmentHeader* h = m.hdr;
        out->mtl_buffer = (__bridge void*)m.buffer;
        out->host_base = m.base;
        out->byte_offset = h->slot_offset[s];
        out->aux_offset = m.aux_count ? h->aux_offset[s] : 0;
        out->aux_count = m.aux_count;
        out->kind = m.kind;
        out->dtype = m.dtype;
        out->ndim = m.ndim;
        for (int i = 0; i < 4; ++i) out->shape[i] = m.shape[i];
        out->colormap = meta.colormap;
        out->flags = meta.flags;
        out->vmin = meta.vmin;
        out->vmax = meta.vmax;
        // Written by the client's GPU work (aminmax, bbox, histogram range);
        // complete because the client publishes only after its event completed.
        out->stats = (const float*)((const uint8_t*)m.base + h->stats_offset[s]);
        const int32_t cap = (int32_t)(h->stats_bytes / sizeof(float));
        out->stats_count = std::max(0, std::min(meta.stats_count > 0 ? meta.stats_count : 2, cap));
        out->series_total = meta.series_total;
        out->points_dims = meta.points_dims;
        out->label_count = meta.label_count;
        out->slot = s;
        out->seq = meta.seq;
        out->stamp_ns = meta.stamp_ns;
    }

    Mapped* find(uint64_t key) {
        for (auto& m : maps_)
            if (m.key == key) return &m;
        return nullptr;
    }

    bool map_view(const CaliperLiveViewEntry& e, Mapped* m) {
        // Only names of THIS session (a table is data from another process).
        const std::string prefix = "/" + session_ + "-v";
        if (std::strncmp(e.segment, prefix.c_str(), prefix.size()) != 0) return false;
        if (e.kind < CALIPER_LIVE_KIND_HEATMAP || e.kind > CALIPER_LIVE_KIND_HISTOGRAM) return false;
        if (e.ndim < 1 || e.ndim > 4) return false;
        uint64_t want = dtype_bytes(e.dtype);
        for (uint32_t i = 0; i < e.ndim; ++i) {
            if (e.shape[i] < 1) return false;
            want *= (uint64_t)e.shape[i];
        }
        if (want == 0 || want != e.slot_bytes) return false;
        if (e.aux_count < 0 || (uint64_t)e.aux_count * sizeof(float) > e.aux_bytes) return false;
        if (e.segment_bytes == 0 || e.segment_bytes % page_size() != 0) return false;
        std::string why;
        void* base = map_existing(e.segment, e.segment_bytes, &why);
        if (!base) return false;   // removed before we got here: next generation drops it
        auto* h = (CaliperLiveSegmentHeader*)base;
        if (!header_ok(h, e.segment_bytes, &why) || h->view_id != e.view_id || h->kind != e.kind ||
            h->payload_bytes != e.slot_bytes || h->aux_bytes != e.aux_bytes) {
            munmap(base, (size_t)e.segment_bytes);
            return false;
        }
        id<MTLBuffer> buf = [device_ newBufferWithBytesNoCopy:base
                                                       length:(NSUInteger)e.segment_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:^(void* ptr, NSUInteger n) {
                                                      munmap(ptr, (size_t)n);
                                                  }];
        if (buf == nil) {
            munmap(base, (size_t)e.segment_bytes);
            return false;
        }
        // A fresh mapping: claims a previous (dead or detached) viewer left are
        // ours to clear; nothing of ours reads this segment yet.
        store_seqcst(&h->viewer_prev_slot, CALIPER_LIVE_SLOT_NONE);
        store_seqcst(&h->viewer_reading_slot, CALIPER_LIVE_SLOT_NONE);
        m->key = e.view_id;
        m->name = e.name;
        m->kind = e.kind;
        m->dtype = e.dtype;
        m->ndim = (int32_t)e.ndim;
        for (int i = 0; i < 4; ++i) m->shape[i] = e.shape[i];
        m->aux_count = e.aux_count;
        m->slot_count = h->slot_count;
        m->base = base;
        m->len = e.segment_bytes;
        m->hdr = h;
        m->buffer = buf;
        return true;
    }

    std::string         session_;
    std::string         script_;
    uint64_t            start_ms_ = 0;
    CaliperLiveControl* ctl_ = nullptr;
    uint64_t            ctl_len_ = 0;
    id<MTLDevice>       device_ = nil;
    int32_t             pid_ = 0;
    uint64_t            gen_seen_ = UINT64_MAX;
    std::vector<Mapped> maps_;
};

// --- discovery -------------------------------------------------------------------

// Read-only look at a session's control segment. false when absent/foreign.
struct CtlPeek {
    bool     ok = false;
    int32_t  client_pid = 0;
    uint32_t state = 0;
    int32_t  views = 0;
    int32_t  viewer_pid = 0;
    uint64_t heartbeat_age_ns = UINT64_MAX;
    std::vector<std::string> segments;   // active rows' names under the session prefix
};

CtlPeek peek_ctl(const std::string& session) {
    CtlPeek r;
    int fd = shm_open(ctl_name(session).c_str(), O_RDONLY, 0);
    if (fd < 0) return r;
    struct stat st{};
    const uint64_t len = ctl_bytes();
    if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < len) { close(fd); return r; }
    void* p = mmap(nullptr, (size_t)len, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return r;
    const auto* c = (const CaliperLiveControl*)p;
    if (load_acquire(&c->magic) == CALIPER_LIVE_CTL_MAGIC && c->version == CALIPER_LIVE_PROTOCOL_VERSION &&
        c->struct_size >= sizeof(CaliperLiveControl)) {
        r.ok = true;
        r.client_pid = c->client_pid;
        r.state = load_acquire(&c->state);
        r.viewer_pid = load_acquire(&c->viewer_pid);
        const uint64_t hb = load_acquire(&c->viewer_heartbeat_ns);
        const uint64_t now = now_ns();
        r.heartbeat_age_ns = hb == 0 ? UINT64_MAX : (now > hb ? now - hb : 0);
        const std::string prefix = seg_prefix(session);
        for (const auto& e : c->views) {
            if (e.state != CALIPER_LIVE_ENTRY_ACTIVE) continue;
            ++r.views;
            std::string n = cstr(e.segment, sizeof e.segment);
            if (n.compare(0, prefix.size(), prefix) == 0) r.segments.push_back(n);
        }
    }
    munmap(p, (size_t)len);
    return r;
}

bool same_process(const SessionFile& f) {
    if (!pid_alive(f.pid)) return false;
    if (!f.proc_start_us) return true;
    const uint64_t now = proc_start_us(f.pid);
    return now == 0 || now == f.proc_start_us;
}

void reclaim(const std::string& path, const SessionFile& f) {
    const CtlPeek c = peek_ctl(f.session);
    const bool reused = c.ok && c.client_pid != f.pid && pid_alive(c.client_pid);
    if (!reused) {
        const std::string prefix = seg_prefix(f.session);
        std::vector<std::string> names = f.segments;
        names.insert(names.end(), c.segments.begin(), c.segments.end());
        names.push_back(ctl_name(f.session));
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        for (const auto& n : names)
            if (n.compare(0, prefix.size(), prefix) == 0 && n.size() < CALIPER_LIVE_SEGNAME_MAX)
                shm_unlink(n.c_str());
    }
    unlink(path.c_str());
}

}  // namespace

std::unique_ptr<Source> open_shm_source(const std::string& session, void* mtl_device,
                                        std::string* err, OpenFail* fail) {
    OpenFail dummy;
    OpenFail& f = fail ? *fail : dummy;
    f = OpenFail::Refused;
    if (!valid_session(session.c_str())) {
        *err = "session must be 1..20 chars of [A-Za-z0-9._-]";
        return nullptr;
    }
    if (!mtl_device) {
        *err = "no Metal device";
        return nullptr;
    }
    const std::string name = ctl_name(session);
    int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) {
        if (errno == ENOENT) f = OpenFail::Missing;
        *err = "no live session '" + session + "' (shm_open('" + name + "'): " +
               std::strerror(errno) + ")";
        return nullptr;
    }
    struct stat st{};
    const uint64_t len = ctl_bytes();
    if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < len) {
        close(fd);
        // ftruncate not done yet (still starting) or an older, smaller layout.
        if (st.st_size == 0) f = OpenFail::Starting;
        *err = "session '" + session + "': control segment is too small (another protocol version?)";
        return nullptr;
    }
    void* p = mmap(nullptr, (size_t)len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        *err = std::string("mmap(control): ") + std::strerror(errno);
        return nullptr;
    }
    auto* ctl = (CaliperLiveControl*)p;
    std::string why;
    const uint64_t magic = load_acquire(&ctl->magic);
    if (magic == 0) {
        f = OpenFail::Starting;
        why = "session is still starting";
    } else if (magic != CALIPER_LIVE_CTL_MAGIC) {
        why = "bad control magic";
    } else if (ctl->version != CALIPER_LIVE_PROTOCOL_VERSION) {
        why = "protocol version mismatch (session v" + std::to_string(ctl->version) + ", viewer v" +
              std::to_string(CALIPER_LIVE_PROTOCOL_VERSION) + ")";
    } else if (ctl->struct_size < sizeof(CaliperLiveControl) ||
               ctl->entry_size < sizeof(CaliperLiveViewEntry) ||
               ctl->max_views != CALIPER_LIVE_MAX_VIEWS) {
        why = "control layout mismatch";
    } else {
        // One viewer per session: the claims are single-reader words.
        const int32_t vp = load_acquire(&ctl->viewer_pid);
        const uint64_t hb = load_acquire(&ctl->viewer_heartbeat_ns);
        const uint64_t now = now_ns();
        if (vp > 0 && vp != (int32_t)getpid() && pid_alive(vp) && hb &&
            (now < hb || now - hb < CALIPER_LIVE_ATTACH_WINDOW_NS))
            why = "session is being viewed by pid " + std::to_string(vp) + " (one viewer per session)";
    }
    if (!why.empty()) {
        munmap(p, (size_t)len);
        *err = "session '" + session + "': " + why;
        return nullptr;
    }
    f = OpenFail::None;
    return std::make_unique<ShmSource>(session, ctl, len, (__bridge id<MTLDevice>)mtl_device);
}

bool list_sessions(const std::string& root, std::vector<SessionEntry>* out, std::string* err) {
    out->clear();
    const std::string dir = sessions_dir(root);
    DIR* d = opendir(dir.c_str());
    if (!d) {
        if (errno == ENOENT) return true;   // nobody ever opened a session here
        if (err) *err = "opendir('" + dir + "'): " + std::strerror(errno);
        return false;
    }
    std::vector<std::string> names;
    while (dirent* de = readdir(d)) names.emplace_back(de->d_name);
    closedir(d);
    for (const auto& n : names) {
        const std::string path = dir + "/" + n;
        const size_t tmp = n.find(".json.tmp.");
        if (tmp != std::string::npos) {   // an interrupted atomic write
            const int32_t writer = (int32_t)std::atoi(n.c_str() + tmp + 10);
            if (!pid_alive(writer)) unlink(path.c_str());
            continue;
        }
        if (n.size() < 6 || n.compare(n.size() - 5, 5, ".json") != 0) continue;
        SessionFile f;
        if (!read_session_file(path, &f)) continue;   // not ours / unreadable: leave it
        if (!same_process(f)) {
            reclaim(path, f);
            continue;
        }
        SessionEntry e;
        e.path = path;
        e.session = f.session;
        e.script = f.script;
        e.pid = f.pid;
        e.protocol = f.protocol;
        e.start_unix_ms = f.start_unix_ms;
        e.view_count = (int32_t)f.views.size();
        if (f.protocol == (int32_t)CALIPER_LIVE_PROTOCOL_VERSION) {
            const CtlPeek c = peek_ctl(f.session);
            e.alive = c.ok && c.client_pid == f.pid && c.state == CALIPER_LIVE_SESSION_OPEN;
            e.starting = !c.ok;   // absent or not initialised yet
            if (c.ok) e.view_count = c.views;
            if (c.ok && c.viewer_pid > 0 && c.heartbeat_age_ns < CALIPER_LIVE_ATTACH_WINDOW_NS &&
                pid_alive(c.viewer_pid))
                e.viewer_pid = c.viewer_pid;
        } else {
            e.alive = true;   // its process runs; this viewer cannot attach (compatible = 0)
        }
        out->push_back(std::move(e));
    }
    std::sort(out->begin(), out->end(), [](const SessionEntry& a, const SessionEntry& b) {
        return a.start_unix_ms != b.start_unix_ms ? a.start_unix_ms > b.start_unix_ms : a.pid > b.pid;
    });
    return true;
}

}  // namespace caliper_live

// ---------------------------------------------------------------------------
// C ABI: discovery
// ---------------------------------------------------------------------------
extern "C" const char* caliper_live_data_root(void) {
    static thread_local std::string s;
    s = caliper_live::data_root(nullptr);
    return s.c_str();
}

extern "C" int caliper_live_list_sessions_in(const char* root, CaliperLiveSessionInfo* out, int cap) {
    using namespace caliper_live;
    if (cap < 0 || (cap > 0 && !out)) {
        live_err() = "live_list_sessions: null out with cap > 0";
        return -1;
    }
    for (int i = 0; i < cap; ++i)
        if (out[i].struct_size < sizeof(CaliperLiveSessionInfo)) {
            live_err() = "live_list_sessions: short CaliperLiveSessionInfo";
            return -1;
        }
    std::vector<SessionEntry> v;
    std::string err;
    if (!list_sessions(caliper_live::data_root(root), &v, &err)) {
        live_err() = "live_list_sessions: " + err;
        return -1;
    }
    for (int i = 0; i < cap && i < (int)v.size(); ++i) {
        const SessionEntry& e = v[(size_t)i];
        CaliperLiveSessionInfo& o = out[i];
        const size_t sz = o.struct_size;
        o = CaliperLiveSessionInfo{};
        o.struct_size = sz;
        o.pid = e.pid;
        o.alive = e.alive ? 1 : 0;
        o.view_count = e.view_count;
        o.protocol = e.protocol;
        o.start_unix_ms = e.start_unix_ms;
        o.viewer_pid = e.viewer_pid;
        o.compatible = e.protocol == (int32_t)CALIPER_LIVE_PROTOCOL_VERSION ? 1 : 0;
        std::snprintf(o.session, sizeof o.session, "%s", e.session.c_str());
        std::snprintf(o.script, sizeof o.script, "%s", e.script.c_str());
        std::snprintf(o.path, sizeof o.path, "%s", e.path.c_str());
    }
    live_err().clear();
    return (int)v.size();
}

extern "C" int caliper_live_list_sessions(CaliperLiveSessionInfo* out, int cap) {
    return caliper_live_list_sessions_in(nullptr, out, cap);
}
