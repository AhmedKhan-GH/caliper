// live_shm.h: host-internal helpers shared by the live client (live_client.mm),
// the viewer-side shm source + session lister (live_session.mm) and the core
// glue (live_core.cpp): segment layout, shm names, the atomic accessors for the
// shared sync words, header validation, process identity, the session files
// (<data_root>/live/<pid>.json) and the small JSON reader/writer they need.
// Plain C++ (no ObjC), header-only; nothing here crosses the C seam.
#pragma once

#include "caliper/live.h"

#include <caliper/tensor.h>

#include "app_paths.h"   // caliper::app_data_dir (honours the embed data_dir override)

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <libproc.h>
#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace caliper_live {

// caliper_live_last_error()'s thread-local string (one instance for every TU).
inline std::string& live_err() {
    static thread_local std::string e;
    return e;
}

// --- atomics on the shared mapping ------------------------------------------
// Both processes see the same physical pages; ARM64 atomics on coherent memory
// work across processes. The __atomic builtins accept plain fields, which keeps
// the protocol structs C (live.h) instead of std::atomic members.
template <typename T> inline T load_acquire(const T* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
template <typename T> inline T load_seqcst(const T* p)  { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
template <typename T> inline void store_release(T* p, T v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
template <typename T> inline void store_seqcst(T* p, T v)  { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
template <typename T> inline T fetch_add(T* p, T v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }

// CLOCK_UPTIME_RAW == mach_absolute_time in ns == Python's time.monotonic_ns()
// on macOS (checked in python/bench/results/B-clock.txt).
inline uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
inline uint64_t unix_ms() {
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}

inline uint64_t page_size() {
    static const uint64_t p = (uint64_t)getpagesize();   // 16 KB on Apple silicon
    return p;
}
inline uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// --- process identity ---------------------------------------------------------
inline bool pid_alive(int32_t pid) {
    if (pid <= 0) return false;
    return kill((pid_t)pid, 0) == 0 || errno == EPERM;
}
// Start time of `pid` in µs since the epoch (0 when unknown). A session file
// records its writer's; a listed pid that runs with another start time is a
// reused pid, i.e. the session is stale.
inline uint64_t proc_start_us(int32_t pid) {
    struct proc_bsdinfo bi{};
    if (pid <= 0 || proc_pidinfo((pid_t)pid, PROC_PIDTBSDINFO, 0, &bi, (int)sizeof bi) != (int)sizeof bi)
        return 0;
    return (uint64_t)bi.pbi_start_tvsec * 1000000ull + (uint64_t)bi.pbi_start_tvusec;
}

// --- names ------------------------------------------------------------------
// macOS caps POSIX shm names at 31 chars (PSHMNAMLEN). Session tokens are at
// most 20 chars, so "/<session>-ctl" and "/<session>-v<id>" (id < 100000) fit.
inline bool valid_session(const char* s) {
    if (!s) return false;
    const size_t n = std::strlen(s);
    if (n < 1 || n > CALIPER_LIVE_SESSION_MAX) return false;
    for (size_t i = 0; i < n; ++i) {
        const char c = s[i];
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}
inline std::string ctl_name(const std::string& session) { return "/" + session + "-ctl"; }
inline std::string seg_name(const std::string& session, uint64_t id) {
    return "/" + session + "-v" + std::to_string(id);
}
inline std::string seg_prefix(const std::string& session) { return "/" + session + "-"; }
constexpr uint64_t kMaxViewId = 99999;

// --- layout -----------------------------------------------------------------
// [header pages][slot 0: payload (page-aligned) | aux (256-aligned) | stats
// (256-aligned)][slot 1 ...][slot 2 ...]. Each slot is rounded up to whole pages
// so the next payload is page-aligned too. Offsets are relative to the start of
// the mapping, i.e. they are the byte_offset into either process's MTLBuffer.
struct Layout {
    uint32_t slot_count = CALIPER_LIVE_SLOTS;
    uint64_t header_bytes = 0;
    uint64_t payload_bytes = 0;
    uint64_t aux_bytes = 0;
    uint64_t stats_bytes = CALIPER_LIVE_STATS_BYTES;
    uint64_t slot_stride = 0;
    uint64_t segment_bytes = 0;
    uint64_t slot_offset[CALIPER_LIVE_SLOTS] = {};
    uint64_t aux_offset[CALIPER_LIVE_SLOTS] = {};
    uint64_t stats_offset[CALIPER_LIVE_SLOTS] = {};
};

inline size_t dtype_bytes(int32_t dt) {
    switch (dt) {
        case CALIPER_DT_F32: return 4;
        case CALIPER_DT_U8:  return 1;
        default:             return 0;   // other dtypes are not live payloads
    }
}

inline Layout make_layout(uint64_t payload_bytes, uint64_t aux_bytes = 0,
                          uint32_t slot_count = CALIPER_LIVE_SLOTS) {
    Layout L;
    const uint64_t pg = page_size();
    L.slot_count = slot_count;
    L.header_bytes = round_up(sizeof(CaliperLiveSegmentHeader), pg);
    L.payload_bytes = payload_bytes;
    L.aux_bytes = aux_bytes;
    const uint64_t pay = round_up(payload_bytes, 256);
    const uint64_t aux = round_up(aux_bytes, 256);
    L.slot_stride = round_up(pay + aux + L.stats_bytes, pg);
    for (uint32_t i = 0; i < slot_count; ++i) {
        L.slot_offset[i] = L.header_bytes + (uint64_t)i * L.slot_stride;
        L.aux_offset[i] = aux_bytes ? L.slot_offset[i] + pay : 0;
        L.stats_offset[i] = L.slot_offset[i] + pay + aux;
    }
    L.segment_bytes = L.header_bytes + slot_count * L.slot_stride;
    return L;
}

inline uint64_t ctl_bytes() { return round_up(sizeof(CaliperLiveControl), page_size()); }

inline uint32_t kind_slots(uint32_t kind) {
    return kind == CALIPER_LIVE_KIND_SERIES ? 1u : (uint32_t)CALIPER_LIVE_SLOTS;
}

// --- validation ---------------------------------------------------------------
// The viewer maps what another process wrote: every field it uses to address
// memory is checked against the mapping length before use.
inline bool header_ok(const CaliperLiveSegmentHeader* h, uint64_t map_len, std::string* why) {
    auto bad = [&](const char* w) { if (why) *why = w; return false; };
    if (h->magic != CALIPER_LIVE_SEG_MAGIC) return bad("bad segment magic");
    if (h->version != CALIPER_LIVE_PROTOCOL_VERSION) return bad("segment protocol version mismatch");
    if (h->struct_size < sizeof(CaliperLiveSegmentHeader)) return bad("short segment header");
    if (h->slot_count != kind_slots(h->kind)) return bad("unexpected slot_count for the kind");
    if (h->segment_bytes > map_len) return bad("segment larger than its mapping");
    if (h->stats_bytes < 2 * sizeof(float) || h->stats_bytes > CALIPER_LIVE_STATS_BYTES)
        return bad("bad stats region size");
    for (uint32_t i = 0; i < h->slot_count; ++i) {
        if (h->slot_offset[i] % 4 || h->slot_offset[i] + h->payload_bytes > map_len)
            return bad("slot payload out of bounds");
        if (h->aux_bytes && (h->aux_offset[i] % 4 || h->aux_offset[i] + h->aux_bytes > map_len))
            return bad("slot aux out of bounds");
        if (h->stats_offset[i] % 4 || h->stats_offset[i] + h->stats_bytes > map_len)
            return bad("slot stats out of bounds");
    }
    return true;
}

// --- data root / session files -------------------------------------------------
// <root>/live/<pid>.json. root: explicit, else $CALIPER_DATA_DIR, else the app
// data dir (caliper::app_data_dir, which an embed core's data_dir overrides).
inline std::string data_root(const char* explicit_root) {
    if (explicit_root && explicit_root[0]) return explicit_root;
    const char* env = std::getenv("CALIPER_DATA_DIR");
    if (env && env[0]) return env;
    return caliper::app_data_dir();
}
inline std::string sessions_dir(const std::string& root) { return root + "/live"; }

struct SessionFile {
    int32_t                  protocol = 0;
    int32_t                  pid = 0;
    uint64_t                 proc_start_us = 0;
    uint64_t                 start_unix_ms = 0;
    std::string              session;
    std::string              script;
    std::vector<std::string> views;
    std::vector<std::string> segments;   // shm names, the control segment first
};

inline void json_str(std::string& o, const std::string& s) {
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    o += '"';
}

inline std::string session_json(const SessionFile& f) {
    std::string o = "{\n";
    o += "  \"protocol\": " + std::to_string(f.protocol) + ",\n";
    o += "  \"pid\": " + std::to_string(f.pid) + ",\n";
    o += "  \"proc_start_us\": " + std::to_string(f.proc_start_us) + ",\n";
    o += "  \"session\": "; json_str(o, f.session); o += ",\n";
    o += "  \"script\": "; json_str(o, f.script); o += ",\n";
    o += "  \"start_unix_ms\": " + std::to_string(f.start_unix_ms) + ",\n";
    auto arr = [&](const char* key, const std::vector<std::string>& v) {
        o += std::string("  \"") + key + "\": [";
        for (size_t i = 0; i < v.size(); ++i) { if (i) o += ", "; json_str(o, v[i]); }
        o += "]";
    };
    arr("views", f.views);
    o += ",\n";
    arr("segments", f.segments);
    o += "\n}\n";
    return o;
}

// A small reader for exactly what session_json writes (a flat object of
// numbers, strings and string arrays). Anything else: false.
class JsonReader {
public:
    explicit JsonReader(const std::string& s) : s_(s) {}
    bool parse(SessionFile* f) {
        ws();
        if (!eat('{')) return false;
        ws();
        if (eat('}')) return true;
        for (;;) {
            std::string key;
            ws();
            if (!str(&key)) return false;
            ws();
            if (!eat(':')) return false;
            ws();
            if (peek() == '"') {
                std::string v;
                if (!str(&v)) return false;
                if (key == "session") f->session = v;
                else if (key == "script") f->script = v;
            } else if (peek() == '[') {
                std::vector<std::string> v;
                if (!strs(&v)) return false;
                if (key == "views") f->views = std::move(v);
                else if (key == "segments") f->segments = std::move(v);
            } else {
                uint64_t n = 0;
                if (!num(&n)) return false;
                if (key == "protocol") f->protocol = (int32_t)n;
                else if (key == "pid") f->pid = (int32_t)n;
                else if (key == "proc_start_us") f->proc_start_us = n;
                else if (key == "start_unix_ms") f->start_unix_ms = n;
            }
            ws();
            if (eat(',')) continue;
            if (eat('}')) return true;
            return false;
        }
    }

private:
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    bool eat(char c) { if (peek() == c) { ++i_; return true; } return false; }
    void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_; }
    bool num(uint64_t* out) {
        size_t start = i_;
        uint64_t v = 0;
        while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') { v = v * 10 + (uint64_t)(s_[i_] - '0'); ++i_; }
        *out = v;
        return i_ > start && i_ - start <= 19;
    }
    bool str(std::string* out) {
        if (!eat('"')) return false;
        out->clear();
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') return true;
            if (c != '\\') { *out += c; continue; }
            if (i_ >= s_.size()) return false;
            char e = s_[i_++];
            switch (e) {
                case '"': *out += '"'; break;
                case '\\': *out += '\\'; break;
                case '/': *out += '/'; break;
                case 'n': *out += '\n'; break;
                case 'r': *out += '\r'; break;
                case 't': *out += '\t'; break;
                case 'u': {
                    if (i_ + 4 > s_.size()) return false;
                    unsigned v = (unsigned)std::strtoul(s_.substr(i_, 4).c_str(), nullptr, 16);
                    i_ += 4;
                    *out += v < 0x80 ? (char)v : '?';
                    break;
                }
                default: return false;
            }
        }
        return false;
    }
    bool strs(std::vector<std::string>* out) {
        if (!eat('[')) return false;
        ws();
        if (eat(']')) return true;
        for (;;) {
            ws();
            std::string v;
            if (!str(&v)) return false;
            out->push_back(std::move(v));
            ws();
            if (eat(',')) continue;
            return eat(']');
        }
    }
    const std::string& s_;
    size_t i_ = 0;
};

inline bool read_session_file(const std::string& path, SessionFile* f) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0 && s.size() < (1u << 20)) s.append(buf, n);
    std::fclose(fp);
    *f = SessionFile{};
    return JsonReader(s).parse(f) && f->pid > 0 && valid_session(f->session.c_str());
}

// Atomic replace: write <path>.tmp.<pid>, fsync-free (a viewer only needs a
// complete file, not a durable one), rename over <path>. 0600.
inline bool write_session_file(const std::string& path, const SessionFile& f, std::string* why) {
    const std::string tmp = path + ".tmp." + std::to_string((int)getpid());
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { if (why) *why = "open('" + tmp + "'): " + std::strerror(errno); return false; }
    const std::string body = session_json(f);
    const char* p = body.data();
    size_t left = body.size();
    while (left) {
        const ssize_t w = write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) continue;
            if (why) *why = std::string("write(session file): ") + std::strerror(errno);
            close(fd);
            unlink(tmp.c_str());
            return false;
        }
        p += w;
        left -= (size_t)w;
    }
    close(fd);
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        if (why) *why = std::string("rename(session file): ") + std::strerror(errno);
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

inline bool ensure_dir(const std::string& d) {
    struct stat st{};
    if (stat(d.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    // parents first (the data root may not exist yet)
    const size_t slash = d.find_last_of('/');
    if (slash != std::string::npos && slash > 0) ensure_dir(d.substr(0, slash));
    return mkdir(d.c_str(), 0700) == 0 || errno == EEXIST;
}

}  // namespace caliper_live
