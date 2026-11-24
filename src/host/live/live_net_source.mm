// live_net_source.mm: the wire protocol v3 Source of live_net_source.h. The
// network thread speaks the protocol and fills slots; the frame thread claims
// them through the Source methods. See the header for the rules.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "live/live_net_source.h"
#include "live/live_series_ring.h"
#include "live/scene_piece_form.h"
#include "live/scene_queue.h"

#include "host_version.h"
#include "live/live_shm.h"   // now_ns
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace caliper_live {
namespace {

// The program epochs of every stream of this process, so that an attachment to
// another stream is a new epoch for the app's sync of programs
// (live_programs.h) even when that stream's own count would repeat the last
// stream's: a switch back to a run that the app showed before sends its
// programs again.
std::atomic<uint64_t> g_program_epochs{0};

namespace wire = caliper_net::wire;
using caliper_net::Bytes;
using caliper_net::Client;
using caliper_net::Json;
using caliper_net::monotonic_ms;

constexpr uint32_t kNone = CALIPER_LIVE_SLOT_NONE;
constexpr size_t kStatsFloats = CALIPER_LIVE_STATS_BYTES / sizeof(float);   // 64
constexpr int kPollMs = 50;
constexpr int kStopFlushMs = 300;
constexpr int kBackoffFirstMs = 250;
constexpr int kBackoffMaxMs = 2000;

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

bool valid_token(const std::string& t) {
    if (t.size() < CALIPER_NET_TOKEN_MIN || t.size() > CALIPER_NET_TOKEN_MAX) return false;
    for (unsigned char c : t)
        if (!(std::isalnum(c) || c == '_' || c == '-') || c >= 0x80) return false;
    return true;
}

bool valid_session_id(const std::string& s) {
    if (s.empty() || s.size() > CALIPER_NET_SESSION_ID_MAX) return false;
    for (unsigned char c : s)
        if (!(std::isalnum(c) || c == '.' || c == '_' || c == '-') || c >= 0x80) return false;
    return true;
}

// What a slot holds (written by the network thread before it publishes).
struct SlotMeta {
    uint32_t kind = 0;
    int32_t  dtype = 0;
    int32_t  ndim = 0;
    int64_t  shape[4] = {};
    int32_t  colormap = 0;
    uint32_t flags = 0;
    float    vmin = 0.0f, vmax = 1.0f;
    int32_t  stats_count = 0;
    int32_t  points_dims = 0;
    int32_t  label_count = 0;
    int64_t  aux_count = 0;
    uint64_t aux_offset = 0;
    uint64_t stats_offset = 0;
    uint64_t seq = 0;        // the local publish counter
    uint64_t stamp_ns = 0;
    int64_t  step = INT64_MIN;   // the step stamp; INT64_MIN when none
    uint64_t copied = 0;     // bytes received into the slot (Snapshot::copied_bytes)
};

struct Slot {
    id<MTLBuffer> buf = nil;
    uint64_t      cap = 0;
    SlotMeta      meta;
    PayloadMap    map;      // the snapshot's payload axes (Snapshot::map points here while claimed)
};

std::string str_or(const Json* j, const char* fallback) {
    return j && j->is_string() ? j->as_string() : std::string(fallback);
}
int64_t int_or(const Json* j, int64_t fallback) {
    int64_t v = 0;
    return j && j->get_i64(&v) ? v : fallback;
}

// A snapshot's axes and tile (wire protocol v3, "Payload axes").
PayloadMap payload_map(const Json& axes, const Json& tile) {
    PayloadMap m;
    if (axes.is_array())
        for (size_t i = 0; i < axes.size(); ++i) {
            const Json& a = axes.at(i);
            if (!a.is_object()) continue;
            PayloadAxis ax;
            ax.dim = str_or(a.find("dim"), "");
            ax.start = int_or(a.find("start"), 0);
            ax.step = int_or(a.find("step"), 1);
            ax.count = int_or(a.find("count"), 0);
            ax.size = int_or(a.find("size"), 0);
            m.axes.push_back(std::move(ax));
        }
    if (tile.is_object()) {
        m.tile_rows = int_or(tile.find("rows"), 0);
        m.tile_cols = int_or(tile.find("cols"), 0);
    }
    return m;
}

bool has_cap(const std::vector<std::string>& caps, const char* cap) {
    return std::find(caps.begin(), caps.end(), cap) != caps.end();
}

// CONTROLS as the frame thread reads it (wire protocol 3.1, "controls").
Controls parse_controls(const wire::Controls& c) {
    Controls out;
    out.run_state = str_or(c.run.find("state"), CALIPER_NET_RUN_NONE);
    uint64_t u = 0;
    if (const Json* st = c.run.find("step"); st && st->get_u64(&u)) {
        out.has_step = true;
        out.step = u;
    }
    if (const Json* sl = c.run.find("steps_left"); sl && sl->get_u64(&u)) out.steps_left = u;
    for (const Json& p : c.params.items()) {
        ControlParam cp;
        cp.name = str_or(p.find("name"), "");
        cp.type = str_or(p.find("type"), "");
        cp.help = str_or(p.find("help"), "");
        if (const Json* v = p.find("value")) cp.value = *v;
        if (const Json* v = p.find("pending")) cp.pending = *v;
        if (const Json* v = p.find("min"); v && v->is_number()) {
            cp.has_min = true;
            cp.min = v->as_double();
        }
        if (const Json* v = p.find("max"); v && v->is_number()) {
            cp.has_max = true;
            cp.max = v->as_double();
        }
        if (const Json* v = p.find("log"); v && v->is_bool()) cp.log = v->as_bool();
        if (const Json* ch = p.find("choices"); ch && ch->is_array())
            for (const Json& x : ch->items())
                if (x.is_string()) cp.choices.push_back(x.as_string());
        out.params.push_back(std::move(cp));
    }
    for (const Json& a : c.actions.items()) {
        ControlAction ca;
        ca.name = str_or(a.find("name"), "");
        ca.help = str_or(a.find("help"), "");
        if (const Json* v = a.find("running"); v && v->is_bool()) ca.running = v->as_bool();
        if (const Json* v = a.find("queued"); v && v->is_bool()) ca.queued = v->as_bool();
        out.actions.push_back(std::move(ca));
    }
    out.interventions = c.interventions;
    return out;
}

// What the frame thread asks the network thread to send (the network thread
// owns the socket and the subscriptions, and fills in what only it knows).
struct Outgoing {
    enum class Kind { Control, Event, ViewRequest, Unsubscribe, Program } kind = Kind::Control;
    wire::Control control;
    wire::Program program;     // Program
    uint64_t key = 0;          // Event, ViewRequest: the view
    std::string type;          // Event
    Json body;                 // Event: detail; ViewRequest: the data view
};

struct SeriesDelta {
    uint64_t start = 0;
    std::vector<float> values;
};

struct NetView {
    // Set at creation, never changed.
    uint64_t    key = 0;
    std::string name;
    uint32_t    kind = 0;
    uint64_t    view_id = 0;

    // Network thread only.
    uint32_t sub = 0;
    bool     subscribed = false;
    bool     stopped = false;      // a scene view the frame thread refused for good: never subscribed again
    uint64_t last_producer_seq = 0;
    Json     shown_data_view;      // the data view of the newest snapshot (events name it)
    Json     shown_axes = Json::array(), shown_tile;   // and its payload axes (select events)

    // Guarded by NetSourceImpl::mu_ (3.1 and data views).
    Json     offer;                // the view's VIEWS entry
    Json     resolved;             // the data view its subscription resolved to

    // Guarded by NetSourceImpl::mu_.
    Slot     slot[NetSource::kSlots];
    uint32_t published = kNone, reading = kNone, prev = kNone;
    uint64_t seq = 0;        // local publishes: completed snapshots, or queued series deltas
    uint64_t dropped = 0;    // snapshots the producer (or a hub) replaced before sending
    std::deque<SeriesDelta> pending;
    uint64_t pending_end = 0;   // the largest end among the queued deltas
    uint64_t series_stamp = 0;  // stamp_ns of the newest SERIES message
    SceneQueue scene;              // a scene view's messages, for take_scene (scene_queue.h)
    caliper_net::wire::SplitUpdates splits;   // a scene view's split updates being put together (network thread)

    // Frame thread only (series).
    id<MTLBuffer> ring = nil;
    uint64_t      ring_total = 0;
};

class NetSourceImpl final : public NetSource {
public:
    NetSourceImpl(StreamUrl url, id<MTLDevice> device) : url_(std::move(url)), device_(device) {
        label_ = url_.display;
        client_ = std::make_unique<Client>();
        thread_ = std::thread([this] { run(); });
    }

    ~NetSourceImpl() override {
        stop_ = true;
        client_->wake();
        if (thread_.joinable()) thread_.join();
    }

    // ---- Source (frame thread) -------------------------------------------------------------
    std::string label() const override {
        std::lock_guard<std::mutex> g(mu_);
        return label_;
    }
    std::string script() const override {
        std::lock_guard<std::mutex> g(mu_);
        return script_;
    }
    uint64_t start_unix_ms() const override {
        std::lock_guard<std::mutex> g(mu_);
        return start_ms_;
    }
    int32_t producer_pid() const override {
        std::lock_guard<std::mutex> g(mu_);
        return pid_;
    }
    bool producer_alive() override {
        std::lock_guard<std::mutex> g(mu_);
        return !failed_ && !ended_ && (attached_ || !ever_attached_);
    }
    void heartbeat() override {}   // the stream needs none: PING and PONG keep it alive

    bool sync(std::vector<SourceView>* views) override {
        std::lock_guard<std::mutex> g(mu_);
        if (views_gen_ == seen_gen_) return false;
        seen_gen_ = views_gen_;
        frame_views_ = views_;
        views->clear();
        for (const auto& v : frame_views_) views->push_back({v->key, v->name, v->kind});
        return true;
    }

    bool take(uint64_t key, uint64_t after_seq, Snapshot* out) override {
        NetView* v = find(key);
        if (!v) return false;
        if (v->kind == CALIPER_LIVE_KIND_SERIES) return take_series(*v, after_seq, out);
        std::lock_guard<std::mutex> g(mu_);
        if (v->published == kNone) return false;
        const Slot& s = v->slot[v->published];
        if (s.meta.seq <= after_seq) return false;
        v->prev = v->reading;
        v->reading = v->published;
        fill(s, v->reading, out);
        return true;
    }

    void release_prev(uint64_t key) override {
        NetView* v = find(key);
        if (!v) return;
        std::lock_guard<std::mutex> g(mu_);
        v->prev = kNone;
    }

    void unclaim(uint64_t key) override {
        NetView* v = find(key);
        if (!v || v->kind == CALIPER_LIVE_KIND_SERIES) return;
        std::lock_guard<std::mutex> g(mu_);
        v->reading = v->prev;
        v->prev = kNone;
    }

    void drew(uint64_t, uint64_t, uint64_t) override {}

    uint64_t producer_dropped(uint64_t key) override {
        NetView* v = find(key);
        if (!v) return 0;
        std::lock_guard<std::mutex> g(mu_);
        return v->dropped;
    }

    // ---- outbound and wire protocol 3.1 (frame thread) -------------------------------------
    // The frame thread never touches the socket: it queues what to send and
    // wakes the network thread, which sends it in order between its polls.
    bool send_event(uint64_t key, const std::string& type, const std::string& detail_json) override {
        if (!find(key)) return false;
        Json detail;
        if (!Json::parse(detail_json.data(), detail_json.size(), &detail, nullptr) || !detail.is_object())
            return false;
        Outgoing o;
        o.kind = Outgoing::Kind::Event;
        o.key = key;
        o.type = type;
        o.body = std::move(detail);
        return enqueue(std::move(o));
    }

    bool request_data_view(uint64_t key, const std::string& data_view_json, std::string* err) override {
        if (!find(key)) {
            if (err) *err = "no such view";
            return false;
        }
        Json dv;
        std::string perr;
        if (!Json::parse(data_view_json.data(), data_view_json.size(), &dv, &perr) || !dv.is_object()) {
            if (err) *err = "the data view must be a JSON object" + (perr.empty() ? "" : " (" + perr + ")");
            return false;
        }
        Outgoing o;
        o.kind = Outgoing::Kind::ViewRequest;
        o.key = key;
        o.body = std::move(dv);
        if (!enqueue(std::move(o))) {
            if (err) *err = "too many requests wait to be sent";
            return false;
        }
        return true;
    }

    bool controls(Controls* out) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!have_controls_) return false;
        *out = controls_;
        return true;
    }

    ControlAccess control_access() override {
        std::lock_guard<std::mutex> g(mu_);
        return access_locked();
    }

    bool supports_interpretation() override {
        std::lock_guard<std::mutex> g(mu_);
        return has_cap(session_caps_, CALIPER_NET_CAP_INTERPRETATION);
    }

    uint32_t send_control(const std::string& op, const std::string& name, const Json& value, uint32_t count,
                          std::string* err) override {
        {
            std::lock_guard<std::mutex> g(mu_);
            const ControlAccess a = access_locked();
            const bool read = op == CALIPER_NET_OP_INTERPRET && value.find("method") &&
                (value.find("method")->as_string().rfind("feature.", 0) == 0 ||
                 value.find("method")->as_string() == "trace.read" || value.find("method")->as_string() == "trace.status");
            if (op == CALIPER_NET_OP_INTERPRET && !has_cap(session_caps_, CALIPER_NET_CAP_INTERPRETATION)) {
                if (err) *err = "The script did not announce interpretation.";
                return 0;
            }
            if ((!a.allowed && !read) || failed_ || !connected_ || !attached_ || ended_) {
                if (err) *err = a.why;
                return 0;
            }
        }
        Outgoing o;
        o.kind = Outgoing::Kind::Control;
        o.control.request_id = next_control_id_++;
        if (o.control.request_id == 0) o.control.request_id = next_control_id_++;
        o.control.op = op;
        if (op == CALIPER_NET_OP_SET || op == CALIPER_NET_OP_RUN) o.control.name = name;
        if (op == CALIPER_NET_OP_SET) o.control.value = value;
        if (op == CALIPER_NET_OP_STEP && count > 0) o.control.count = count;
        if (op == CALIPER_NET_OP_INTERVENE) o.control.intervene = value;
        if (op == CALIPER_NET_OP_INTERPRET) o.control.interpretation = value;
        if (op == CALIPER_NET_OP_INTERPRET) {
            std::vector<uint8_t> bytes;
            if (!wire::encode(o.control, &bytes, err)) return 0;
        }
        const uint32_t id = o.control.request_id;
        if (!enqueue(std::move(o))) {
            if (err) *err = "too many controls wait to be sent";
            return 0;
        }
        return id;
    }

    // ---- 3.2 draft: GPU programs (frame thread) ----------------------------------------------
    bool programs_ready(std::string* why) override {
        std::lock_guard<std::mutex> g(mu_);
        std::string no;
        if (failed_ || !connected_ || !attached_) no = "not attached to a session";
        else if (ended_) no = "the session has ended";
        else if (!server_programs_) no = "the server takes no programs";
        else if (!welcome_control_) no = "the server does not let this viewer control (a hub needs --allow-control)";
        else if (!has_cap(session_caps_, CALIPER_NET_CAP_PROGRAMS))
            no = "the script takes no programs (it needs caliper.watch(model, optimizer))";
        if (why) *why = no;
        return no.empty();
    }

    uint64_t program_epoch() override {
        std::lock_guard<std::mutex> g(mu_);
        return program_epoch_;
    }

    uint32_t send_program(wire::Program msg) override {
        if (!programs_ready(nullptr)) return 0;
        Outgoing o;
        o.kind = Outgoing::Kind::Program;
        msg.request_id = next_control_id_++;
        if (msg.request_id == 0) msg.request_id = next_control_id_++;
        const uint32_t id = msg.request_id;
        o.program = std::move(msg);
        return enqueue(std::move(o)) ? id : 0;
    }

    // ---- 3.2 draft: resources ------------------------------------------------------------
    bool takes_pieces() override {
        if (!programs_ready(nullptr)) return false;
        std::lock_guard<std::mutex> g(mu_);
        return server_resources_ && has_cap(session_caps_, CALIPER_NET_CAP_RESOURCES);
    }

    std::vector<ReceivedResource> take_resources() override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<std::pair<uint64_t, ReceivedResource>> waiting;
        waiting.reserve(resources_.size());
        for (auto& kv : resources_) waiting.push_back(std::move(kv.second));
        resources_.clear();
        std::sort(waiting.begin(), waiting.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<ReceivedResource> out;
        out.reserve(waiting.size());
        for (auto& w : waiting) out.push_back(std::move(w.second));
        return out;
    }

    // The resources of every program, at most: a new resource past it waits for the next take.
    static constexpr size_t kMaxWaitingResources = size_t(CALIPER_NET_MAX_PROGRAMS) * CALIPER_NET_MAX_RESOURCES;

    // A RESOURCE (network thread): its valid bytes are copied out of the frame, and the newest
    // version of each resource waits for the frame thread, in the place of an older one. The parts
    // of a version in parts are put together first (wire::SplitResources), so that only a whole
    // version waits; a part that is dropped or refused is counted, and nothing of it is shown.
    void on_resource(const wire::Resource& part) {
        wire::WholeResource whole;
        std::string err;
        const wire::SplitResources::Result got = resource_parts_.add(part, &whole, &err);
        if (got != wire::SplitResources::Result::Whole) {
            if (got == wire::SplitResources::Result::Dropped || got == wire::SplitResources::Result::Refused)
                resource_parts_refused_.fetch_add(1);
            return;
        }
        const wire::Resource& r = whole.resource;
        // A scene program's version waits for the piece-form update that names it (network thread).
        if (wire::scene_program_id(r.program)) {
            scene_inbox_.take(r, whole.keep ? whole.keep
                                            : std::make_shared<const std::vector<uint8_t>>(r.data.data, r.data.data + r.data.size));
            return;
        }
        ReceivedResource out;
        out.program = r.program;
        out.resource = r.resource;
        out.step = (int64_t)r.step;   // at most 2^53 - 1 on the wire
        out.count = r.count;
        out.stride = r.stride;
        out.width = r.width.value_or(0);
        out.height = r.height.value_or(0);
        out.format = r.format.value_or(std::string());
        out.elements = r.elements;
        out.bytes = whole.keep ? whole.keep
                               : std::make_shared<const std::vector<uint8_t>>(r.data.data, r.data.data + r.data.size);
        const std::string key = r.program + "/" + r.resource;
        std::lock_guard<std::mutex> g(mu_);
        auto it = resources_.find(key);
        if (it != resources_.end()) {
            it->second.second = std::move(out);
        } else if (resources_.size() < kMaxWaitingResources) {
            resources_.emplace(key, std::make_pair(++resource_order_, std::move(out)));
        }
    }

    std::vector<ProgramAnswer> take_program_answers() override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<ProgramAnswer> out(program_answers_.begin(), program_answers_.end());
        program_answers_.clear();
        return out;
    }

    static constexpr size_t kMaxProgramAnswers = 64;   // answers the frame thread has not read

    void push_program_answer(ProgramAnswer a) {
        std::lock_guard<std::mutex> g(mu_);
        if (program_answers_.size() >= kMaxProgramAnswers) program_answers_.pop_front();
        program_answers_.push_back(std::move(a));
    }

    std::vector<ControlReply> take_control_replies() override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<ControlReply> out(replies_.begin(), replies_.end());
        replies_.clear();
        return out;
    }

    uint64_t selections(std::vector<LinkedSelection>* out) override {
        std::lock_guard<std::mutex> g(mu_);
        out->clear();
        for (const auto& kv : selections_) out->push_back(kv.second);
        return sel_version_;
    }

    std::vector<ItemAxis> item_axes() override {
        std::lock_guard<std::mutex> g(mu_);
        return item_axes_;
    }

    std::map<std::string, std::string> view_cadences() override {
        std::lock_guard<std::mutex> g(mu_);
        return cadences_;
    }

    bool data_view_of(uint64_t key, Json* resolved, Json* offer) override {
        NetView* v = find(key);
        if (!v) return false;
        std::lock_guard<std::mutex> g(mu_);
        if (resolved) *resolved = v->resolved;
        if (offer) *offer = v->offer;
        return true;
    }

    bool take_scene(uint64_t key, std::vector<SceneItem>* out) override {
        NetView* v = find(key);
        if (!v || v->kind != CALIPER_NET_KIND_SCENE) return false;
        std::deque<SceneItem> got;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (v->scene.items.empty()) return false;
            got.swap(v->scene.items);
        }
        for (auto& it : got) out->push_back(std::move(it));
        return true;
    }

    void stop_scene(uint64_t key) override {
        NetView* v = find(key);
        if (!v || v->kind != CALIPER_NET_KIND_SCENE) return;
        {
            std::lock_guard<std::mutex> g(mu_);
            v->scene.items.clear();
            v->scene.stopped = true;
        }
        Outgoing o;
        o.kind = Outgoing::Kind::Unsubscribe;
        o.key = key;
        enqueue(std::move(o));
    }

    // ---- NetSource ------------------------------------------------------------------------
    Ready wait_ready(int timeout_ms, std::string* err) override {
        std::unique_lock<std::mutex> g(mu_);
        cv_.wait_for(g, std::chrono::milliseconds(std::max(0, timeout_ms)),
                     [&] { return ever_attached_ || failed_; });
        if (ever_attached_) return Ready::Attached;
        if (failed_) {
            if (err) *err = problem_;
            return Ready::Failed;
        }
        return Ready::Pending;
    }

    bool pending() const override {
        std::lock_guard<std::mutex> g(mu_);
        return !ever_attached_ && !failed_;
    }

    std::string problem() const override {
        std::lock_guard<std::mutex> g(mu_);
        return problem_;
    }

private:
    // ---- 3.1 helpers -----------------------------------------------------------------------------
    static constexpr size_t kMaxOutgoing = 1024;   // queued for the network thread
    static constexpr size_t kMaxReplies = 256;     // CONTROL_REPLYs the frame thread has not read

    bool enqueue(Outgoing o) {
        {
            std::lock_guard<std::mutex> g(out_mu_);
            if (outbox_.size() >= kMaxOutgoing) return false;
            outbox_.push_back(std::move(o));
        }
        client_->wake();
        return true;
    }

    // Who may control (wire protocol 3.1, "Who may control"); mu_ held.
    ControlAccess access_locked() const {
        if (failed_ || !connected_ || !attached_) return {false, "not attached to a session"};
        if (ended_) return {false, "the session has ended"};
        if (!server_controls_) return {false, "the server takes no controls (wire protocol 3.0)"};
        if (!welcome_control_)
            return {false, "the server does not let this viewer control (a hub needs --allow-control)"};
        if (!has_cap(session_caps_, CALIPER_NET_CAP_CONTROLS)) return {false, "the script takes no controls"};
        return {true, ""};
    }

    void push_reply(ControlReply r) {
        std::lock_guard<std::mutex> g(mu_);
        r.controls_version = controls_.version;
        if (replies_.size() >= kMaxReplies) replies_.pop_front();
        replies_.push_back(std::move(r));
    }

    // ---- frame thread helpers ------------------------------------------------------------------
    NetView* find(uint64_t key) {
        for (auto& v : frame_views_)
            if (v->key == key) return v.get();
        return nullptr;
    }

    void fill(const Slot& s, uint32_t index, Snapshot* out) {
        const SlotMeta& m = s.meta;
        out->mtl_buffer = (__bridge void*)s.buf;
        out->host_base = [s.buf contents];
        out->byte_offset = 0;
        out->aux_offset = m.aux_offset;
        out->aux_count = m.aux_count;
        out->kind = m.kind;
        out->dtype = m.dtype;
        out->ndim = m.ndim;
        for (int i = 0; i < 4; ++i) out->shape[i] = m.shape[i];
        out->colormap = m.colormap;
        out->flags = m.flags;
        out->vmin = m.vmin;
        out->vmax = m.vmax;
        out->stats = (const float*)((const uint8_t*)[s.buf contents] + m.stats_offset);
        out->stats_count = m.stats_count;
        out->series_total = 0;
        out->points_dims = m.points_dims;
        out->label_count = m.label_count;
        out->slot = index;
        out->seq = m.seq;
        out->stamp_ns = m.stamp_ns;
        out->step = m.step;
        out->map = s.map.axes.empty() ? nullptr : &s.map;
        out->copied_bytes = m.copied;
    }

    // Applies the queued deltas to the view's ring (frame thread only) and hands
    // the ring over, never claimed.
    bool take_series(NetView& v, uint64_t after_seq, Snapshot* out) {
        std::deque<SeriesDelta> deltas;
        uint64_t seq, stamp;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (v.seq <= after_seq) return false;
            deltas.swap(v.pending);
            v.pending_end = 0;
            seq = v.seq;
            stamp = v.series_stamp;
        }
        constexpr uint64_t M = kSeriesRing;
        if (v.ring == nil) {
            v.ring = [device_ newBufferWithLength:M * sizeof(float) options:MTLResourceStorageModeShared];
            if (v.ring == nil) return false;
            float* r = (float*)[v.ring contents];
            for (uint64_t i = 0; i < M; ++i) r[i] = NAN;
        }
        float* ring = (float*)[v.ring contents];
        // Gaps read as NaN, overlaps replace values (live_series_ring.h).
        uint64_t copied = 0;
        for (const SeriesDelta& d : deltas) {
            series_ring_apply(ring, M, &v.ring_total, d.start, d.values.data(), d.values.size());
            copied += d.values.size() * sizeof(float);
        }
        out->copied_bytes = copied;
        out->mtl_buffer = (__bridge void*)v.ring;
        out->host_base = ring;
        out->byte_offset = 0;
        out->aux_offset = 0;
        out->aux_count = 0;
        out->kind = CALIPER_LIVE_KIND_SERIES;
        out->dtype = CALIPER_DT_F32;
        out->ndim = 1;
        out->shape[0] = (int64_t)M;
        for (int i = 1; i < 4; ++i) out->shape[i] = 0;
        out->colormap = 0;
        out->flags = 0;
        out->vmin = 0.0f;
        out->vmax = 1.0f;
        out->stats = nullptr;
        out->stats_count = 0;
        out->series_total = v.ring_total;
        out->points_dims = 0;
        out->label_count = 0;
        out->slot = 0;
        out->seq = seq;
        out->stamp_ns = stamp;
        return true;
    }

    // ---- the network thread ---------------------------------------------------------------------
    void set_problem(const std::string& p) {
        std::lock_guard<std::mutex> g(mu_);
        problem_ = p;
    }

    bool send(const wire::Message& m) {
        auto b = std::make_shared<Bytes>();
        if (!wire::encode(m, b.get())) return false;
        return client_->send(std::shared_ptr<const Bytes>(std::move(b)));
    }

    void protocol_error(const std::string& why) {
        set_problem("protocol error: " + why);
        client_->close(CALIPER_NET_CLOSE_PROTOCOL_ERROR, why.substr(0, 120));
    }

    void run() {
        int backoff = kBackoffFirstMs;
        while (!stop_) {
            bool attached_this_time = false;
            const bool final = run_connection(&attached_this_time);
            if (stop_ || final) break;
            if (attached_this_time) backoff = kBackoffFirstMs;
            // Wait before the next attempt; wake() (the destructor) ends the wait.
            const uint64_t until = monotonic_ms() + (uint64_t)backoff;
            while (!stop_ && monotonic_ms() < until) client_->poll((int)std::min<uint64_t>(until - monotonic_ms(), 100));
            backoff = std::min(backoff * 2, kBackoffMaxMs);
        }
        std::lock_guard<std::mutex> g(mu_);
        connected_ = false;
        attached_ = false;
        cv_.notify_all();
    }

    // One connection from connect to close. True when no retry should follow.
    bool run_connection(bool* attached_here) {
        std::string err;
        welcomed_ = false;
        attach_sent_ = false;
        session_id_.clear();
        by_sub_.clear();
        for (auto& v : views_snapshot()) v->subscribed = false;
        if (!client_->connect(url_.ws_url, &err)) return fail_final("cannot connect to " + url_.display + ": " + err);
        bool open = false;
        uint64_t hello_ms = 0, ping_ms = 0;
        while (!stop_) {
            std::vector<caliper_net::Event> events = client_->poll(kPollMs);
            for (auto& e : events) {
                switch (e.type) {
                    case caliper_net::Event::Type::Open: {
                        open = true;
                        // 3.1: it shows controls and draws linked selections; 3.2 draft:
                        // it draws scenes.
                        wire::Hello h;
                        h.role = CALIPER_NET_ROLE_VIEWER;
                        h.capabilities = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION, CALIPER_NET_CAP_SCENES,
                                          CALIPER_NET_CAP_PROGRAMS, CALIPER_NET_CAP_INTERVENTIONS, CALIPER_NET_CAP_INTERPRETATION, CALIPER_NET_CAP_TEXT_PARAMS,
                                          CALIPER_NET_CAP_TIMELINES, CALIPER_NET_CAP_RESOURCES, CALIPER_NET_CAP_SCENE_PIECES};
                        Json sw = Json::object();
                        sw.set("name", "caliper-app");
                        sw.set("version", caliper_host::kHostVersionStr);
                        h.software = sw;
                        send(h);
                        hello_ms = monotonic_ms();
                        std::lock_guard<std::mutex> g(mu_);
                        connected_ = true;
                        break;
                    }
                    case caliper_net::Event::Type::Message:
                        if (!e.binary) {
                            set_problem("protocol error: a text message");
                            client_->close(CALIPER_NET_CLOSE_UNSUPPORTED_DATA, "a text message");
                            break;
                        }
                        on_frame(std::make_shared<const Bytes>(std::move(e.data)));
                        break;
                    case caliper_net::Event::Type::Closed: {
                        const bool attached = session_seen_on_connection_;
                        session_seen_on_connection_ = false;
                        *attached_here = attached;
                        return on_closed(e, open);
                    }
                    default:
                        break;
                }
            }
            drain_outbox();
            // Timers: the handshake, the protocol's ping and idle rules.
            const uint64_t now = monotonic_ms();
            if (open && !welcomed_ && hello_ms && now - hello_ms > CALIPER_NET_HANDSHAKE_TIMEOUT_MS) {
                set_problem("no WELCOME within " + std::to_string(CALIPER_NET_HANDSHAKE_TIMEOUT_MS / 1000) + " s");
                client_->close(CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT, "no welcome");
            }
            if (open && welcomed_) {
                const uint64_t idle = now - client_->last_input_ms();
                if (idle > CALIPER_NET_IDLE_TIMEOUT_MS) {
                    set_problem("nothing received for " + std::to_string(CALIPER_NET_IDLE_TIMEOUT_MS / 1000) + " s");
                    client_->close(CALIPER_NET_CLOSE_IDLE_TIMEOUT, "idle");
                } else if (idle > CALIPER_NET_PING_INTERVAL_MS && now - ping_ms > CALIPER_NET_PING_INTERVAL_MS) {
                    ping_ms = now;
                    send(wire::Ping{++ping_nonce_});
                }
            }
        }
        // Stopping: say goodbye, close, and give the close a moment to go out.
        if (open && client_->state() == Client::State::Open) {
            send(wire::Bye{CALIPER_NET_BYE_NORMAL, "the viewer detached"});
            client_->close(CALIPER_NET_CLOSE_NORMAL, "detached");
            const uint64_t end = monotonic_ms() + kStopFlushMs;
            while (monotonic_ms() < end) {
                bool closed = false;
                for (auto& e : client_->poll(20)) closed |= e.type == caliper_net::Event::Type::Closed;
                if (closed) break;
            }
        }
        return true;
    }

    bool fail_final(const std::string& why) {
        std::lock_guard<std::mutex> g(mu_);
        problem_ = why;
        if (!ever_attached_) failed_ = true;
        ended_ = true;
        connected_ = false;
        attached_ = false;
        cv_.notify_all();
        return true;
    }

    // The connection ended. True when the source should not reconnect.
    bool on_closed(const caliper_net::Event& e, bool was_open) {
        {
            std::lock_guard<std::mutex> g(mu_);
            connected_ = false;
            attached_ = false;
        }
        if (stop_) return true;
        if (e.http_status == 403)
            return fail_final("the server refused the connection (HTTP 403): the session token is wrong, or it "
                              "does not accept this Host or Origin");
        if (e.http_status)
            return fail_final("the server refused the upgrade (HTTP " + std::to_string(e.http_status) + ") at " +
                              url_.display);
        if (e.code == CALIPER_NET_CLOSE_UNSUPPORTED_MAJOR)
            return fail_final("the server speaks another major version of the protocol");
        if (e.code == CALIPER_NET_CLOSE_ROLE_REFUSED) return fail_final("the server does not take viewers");
        bool ever;
        {
            std::lock_guard<std::mutex> g(mu_);
            ever = ever_attached_;
        }
        if (!ever) {
            // Nothing was ever attached: report why and stop (the attach fails).
            std::string why = problem();
            if (why.empty())
                why = was_open ? "the server closed the connection (code " + std::to_string(e.code) + ": " + e.reason + ")"
                               : "cannot connect to " + url_.display + ": " + e.reason;
            return fail_final(why);
        }
        set_problem(bye_reason_.empty() ? "disconnected (code " + std::to_string(e.code) + "); reconnecting"
                                        : "the producer said goodbye (" + bye_reason_ + "); reconnecting");
        bye_reason_.clear();
        return false;
    }

    std::vector<std::shared_ptr<NetView>> views_snapshot() {
        std::lock_guard<std::mutex> g(mu_);
        return views_;
    }

    void on_frame(std::shared_ptr<const Bytes> keep) {
        const Bytes& data = *keep;
        wire::Message m;
        const wire::DecodeResult r = wire::decode(data.data(), data.size(), &m);
        if (r.status == wire::DecodeResult::Status::BadFrame) {
            set_problem("a frame the protocol refuses (" + r.reason + ": " + r.message + ")");
            client_->close((uint16_t)r.close_code, r.reason);
            return;
        }
        if (r.status == wire::DecodeResult::Status::BadMessage) {
            if (r.type == CALIPER_NET_MSG_CONTROL_REPLY) {
                ControlReply reply;
                const auto* id = r.frame.meta.find("request_id");
                uint64_t request = 0;
                if (id && id->get_u64(&request) && request <= UINT32_MAX) reply.request_id = uint32_t(request);
                reply.status = CALIPER_NET_STATUS_ERROR;
                reply.error_code = CALIPER_NET_ERR_BAD_REQUEST;
                reply.error_message = "The producer sent an invalid interpretation reply. Check its protocol version.";
                push_reply(std::move(reply));
            }
            wire::Error e;
            e.code = r.error_code;
            e.message = r.message;
            e.in_reply_to = r.type;
            send(e);
            // A scene message this viewer does not build is ignored, and its view says so.
            if (r.type == CALIPER_NET_MSG_SCENE_UPDATE && r.error_code == CALIPER_NET_ERR_UNSUPPORTED)
                scene_note(r.frame.meta, "ignored: " + r.message);
            return;
        }
        const uint16_t type = wire::message_type(m);
        if (!welcomed_) {
            if (auto* w = std::get_if<wire::Welcome>(&m)) on_welcome(*w);
            else if (auto* b = std::get_if<wire::Bye>(&m)) on_bye(*b);
            else if (!std::holds_alternative<wire::Unknown>(m))
                protocol_error(std::string("'") + wire::type_name(type) + "' before WELCOME");
            return;
        }
        if (auto* s = std::get_if<wire::Sessions>(&m)) on_sessions(*s);
        else if (auto* v = std::get_if<wire::Views>(&m)) on_views(*v);
        else if (auto* p = std::get_if<wire::Snapshot>(&m)) on_snapshot(*p);
        else if (auto* q = std::get_if<wire::Series>(&m)) on_series(*q);
        else if (auto* vr = std::get_if<wire::ViewReply>(&m)) on_view_reply(*vr);
        else if (auto* pi = std::get_if<wire::Ping>(&m)) send(wire::Pong{pi->nonce});
        else if (std::holds_alternative<wire::Pong>(m)) {}
        else if (auto* er = std::get_if<wire::Error>(&m)) on_error(*er);
        else if (auto* b = std::get_if<wire::Bye>(&m)) on_bye(*b);
        else if (std::holds_alternative<wire::Unknown>(m)) {}
        else if (auto* c = std::get_if<wire::Controls>(&m)) on_controls(*c);
        else if (auto* cr = std::get_if<wire::ControlReply>(&m)) on_control_reply(*cr);
        else if (auto* sel = std::get_if<wire::Selection>(&m)) on_selection(*sel);
        else if (auto* ps = std::get_if<wire::ProgramState>(&m)) on_program_state(*ps);
        else if (auto* rs = std::get_if<wire::Resource>(&m)) on_resource(*rs);
        // A relayed draw half: the app composes each draw half from its own spec, and reads none.
        else if (std::holds_alternative<wire::Program>(m)) {}
        else if (auto* su = std::get_if<wire::SceneUpdate>(&m)) {
            if (!su->split.is_null()) {
                on_piece(r.frame.meta, r.type, *su);
            } else if (su->program) {
                // The piece form: the update takes the versions of its scene program that came just before it.
                std::string why;
                std::shared_ptr<const void> owner = scene_inbox_.pair(*su, keep, &why);
                if (owner) on_scene(owner, r.frame.sections, std::move(*su));
                else scene_note(r.frame.meta, "dropped: " + why);
            } else {
                const std::string why = scene_not_drawn(*su);
                if (why.empty()) on_scene(keep, r.frame.sections, std::move(*su));
                else refuse_scene(r.frame.meta, su->sub, su->view, r.type, why);
            }
        }
        else if (auto* sc = std::get_if<wire::SceneClear>(&m)) on_scene(keep, r.frame.sections, std::move(*sc));
        else if (auto* st = std::get_if<wire::SceneTransforms>(&m)) on_scene(keep, r.frame.sections, std::move(*st));
        else if (auto* sk = std::get_if<wire::SceneClasses>(&m)) on_scene(keep, r.frame.sections, std::move(*sk));
        else protocol_error(std::string("'") + wire::type_name(type) + "' sent to a viewer");
    }

    // ---- 3.2 draft: scenes -------------------------------------------------------------
    // A scene message goes to its view's queue as it arrived, with the frame its sections
    // point into; the frame thread applies it (take_scene).
    void on_scene(const std::shared_ptr<const void>& keep, const std::vector<wire::Section>& sections,
                  std::variant<wire::SceneUpdate, wire::SceneClear, wire::SceneTransforms, wire::SceneClasses> msg) {
        const uint32_t sub = std::visit([](const auto& x) { return x.sub; }, msg);
        const auto it = by_sub_.find(sub);
        if (it == by_sub_.end() || it->second->kind != CALIPER_NET_KIND_SCENE) return;
        NetView& v = *it->second;
        std::string ignored;   // what this view does not use of an update (a stamp, class ids, normals, ...)
        if (const auto* u = std::get_if<wire::SceneUpdate>(&msg)) ignored = scene_ignored(*u, sections);
        bool again = false;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (!ignored.empty()) again = scene_queue_push(v.scene, SceneItem{nullptr, ignored});
            SceneItem item;
            item.keep = keep;
            std::visit([&](auto& x) { item.msg = std::move(x); }, msg);
            again = scene_queue_push(v.scene, std::move(item)) || again;
            ++v.seq;
        }
        if (again) resubscribe(v);
    }

    // The scene's queue passed its bound while the display was paused and was emptied: a
    // SUBSCRIBE on the same sub replaces the subscription, and the sender answers with the
    // scene's state, whose reset the queue waits for.
    void resubscribe(const NetView& v) {
        wire::Subscribe s;
        s.sub = v.sub;
        s.view = v.name;
        s.request_id = next_request_++;
        send(s);
    }

    // A piece of a split update: the view's SplitUpdates keeps it until the update is whole, which is
    // then queued as any update; a piece that contradicts its update is answered ERROR bad_request.
    void on_piece(const Json& meta, uint16_t type, const wire::SceneUpdate& piece) {
        const auto it = by_sub_.find(piece.sub);
        if (it == by_sub_.end() || it->second->kind != CALIPER_NET_KIND_SCENE) return;
        wire::WholeUpdate whole;
        std::string err;
        switch (it->second->splits.add(piece, &whole, &err)) {
            case wire::SplitUpdates::Result::Pending: return;
            case wire::SplitUpdates::Result::Dropped:
                scene_note(meta, "ignored: a piece of a split update whose first piece is not held");
                return;
            case wire::SplitUpdates::Result::Refused: {
                wire::Error e;
                e.code = CALIPER_NET_ERR_BAD_REQUEST;
                e.message = err;
                e.in_reply_to = type;
                e.sub = piece.sub;
                e.view = piece.view;
                send(e);
                scene_note(meta, "ignored: " + err);
                return;
            }
            case wire::SplitUpdates::Result::Whole: break;
        }
        const std::string why = scene_not_drawn(whole.update);
        if (!why.empty()) {
            refuse_scene(meta, piece.sub, piece.view, type, why);
            return;
        }
        const std::vector<wire::Section> arrays = whole.update.arrays;
        on_scene(whole.keep, arrays, std::move(whole.update));
    }

    // A scene update of a part this view does not draw yet: answered ERROR unsupported, not queued, and noted.
    void refuse_scene(const Json& meta, uint32_t sub, const std::string& view, uint16_t type, const std::string& why) {
        wire::Error e;
        e.code = CALIPER_NET_ERR_UNSUPPORTED;
        e.message = why;
        e.in_reply_to = type;
        e.sub = sub;
        e.view = view;
        send(e);
        scene_note(meta, "ignored: " + why);
    }

    void scene_note(const Json& meta, const std::string& text) {
        const Json* sub = meta.find("sub");
        uint64_t s = 0;
        if (!sub || !sub->get_u64(&s)) return;
        const auto it = by_sub_.find((uint32_t)s);
        if (it == by_sub_.end() || it->second->kind != CALIPER_NET_KIND_SCENE) return;
        bool again = false;
        {
            std::lock_guard<std::mutex> g(mu_);
            again = scene_queue_push(it->second->scene, SceneItem{nullptr, text});
        }
        if (again) resubscribe(*it->second);
    }

    void on_welcome(const wire::Welcome& w) {
        welcomed_ = true;
        wire::Identity id;
        if (wire::Identity::parse(w.session, &id)) remember_identity(id);
        std::lock_guard<std::mutex> g(mu_);
        server_controls_ = has_cap(w.capabilities, CALIPER_NET_CAP_CONTROLS);
        server_selection_ = has_cap(w.capabilities, CALIPER_NET_CAP_SELECTION);
        server_programs_ = has_cap(w.capabilities, CALIPER_NET_CAP_PROGRAMS);
        server_resources_ = has_cap(w.capabilities, CALIPER_NET_CAP_RESOURCES);
        resource_parts_ = wire::SplitResources();   // a new connection holds no part of a version
        scene_inbox_.clear();
        program_fresh_ = true;   // a new connection: the next attachment is a new epoch
        welcome_control_ = w.control.value_or(false);
        // A server that is itself the producer (caliper.serve()) names its own
        // capabilities; a hub's session list names the session's.
        if (w.role == CALIPER_NET_ROLE_PRODUCER) session_caps_ = w.capabilities;
    }

    // ---- 3.1 (network thread) --------------------------------------------------------------
    void on_controls(const wire::Controls& c) {
        Controls parsed = parse_controls(c);
        std::lock_guard<std::mutex> g(mu_);
        parsed.version = controls_.version + 1;
        controls_ = std::move(parsed);
        have_controls_ = true;
    }

    void on_control_reply(const wire::ControlReply& r) {
        ControlReply out;
        out.request_id = r.request_id;
        out.op = r.op;
        out.status = r.status;
        out.name = r.name.value_or("");
        out.value = r.value;
        out.result = r.result;
        out.message = r.message.value_or("");
        if (!r.error.is_null()) {
            out.error_code = str_or(r.error.find("code"), "");
            out.error_message = str_or(r.error.find("message"), "");
        }
        push_reply(std::move(out));
    }

    // 3.2 draft: GPU programs.
    void on_program_state(const wire::ProgramState& ps) {
        ProgramAnswer a;
        a.request_id = ps.request_id;
        a.programs = ps.programs;
        a.step = ps.step;
        a.step_ms = ps.step_ms;
        push_program_answer(std::move(a));
    }

    void on_selection(const wire::Selection& s) {
        LinkedSelection out;
        out.space = s.space;
        out.view = s.view.value_or("");
        for (const Json& x : s.items.items()) {
            uint64_t u = 0;
            if (x.get_u64(&u)) out.items.push_back(u);
        }
        out.truncated = s.truncated;
        out.details = s.details;
        out.details_error = s.details_error.value_or("");
        std::lock_guard<std::mutex> g(mu_);
        if (selections_.size() >= CALIPER_NET_MAX_ITEM_SPACES && !selections_.count(s.space)) return;
        out.seq = ++sel_version_;
        selections_[s.space] = std::move(out);
    }

    // What the frame thread queued, sent in order. Without an attached session
    // a CONTROL is answered here, as the server would, so no request waits.
    void drain_outbox() {
        std::deque<Outgoing> out;
        {
            std::lock_guard<std::mutex> g(out_mu_);
            out.swap(outbox_);
        }
        const bool ready = welcomed_ && !session_id_.empty() && client_->state() == Client::State::Open;
        for (Outgoing& o : out) {
            switch (o.kind) {
                case Outgoing::Kind::Control: {
                    if (!ready) {
                        ControlReply r;
                        r.request_id = o.control.request_id;
                        r.op = o.control.op;
                        r.status = CALIPER_NET_STATUS_ERROR;
                        r.name = o.control.name.value_or("");
                        r.error_code = CALIPER_NET_ERR_NOT_ATTACHED;
                        r.error_message = "the stream is not attached to a session";
                        push_reply(std::move(r));
                        break;
                    }
                    send(o.control);
                    break;
                }
                case Outgoing::Kind::Program: {   // 3.2 draft
                    if (!ready) {   // answered here, as the server would, so no request waits
                        ProgramAnswer a;
                        a.request_id = o.program.request_id;
                        a.error_code = CALIPER_NET_ERR_NOT_ATTACHED;
                        a.error_message = "the stream is not attached to a session";
                        push_program_answer(std::move(a));
                        break;
                    }
                    send(o.program);
                    break;
                }
                case Outgoing::Kind::Event: {
                    std::shared_ptr<NetView> v = view_by_key(o.key);
                    if (!ready || !v || !v->subscribed) break;
                    wire::Event e;
                    e.view = v->name;
                    e.type = o.type;
                    e.detail = std::move(o.body);
                    if (server_selection_ && e.type == "select") add_items(*v, &e.detail);
                    e.sub = v->sub;
                    if (v->last_producer_seq) e.seq = v->last_producer_seq;
                    if (v->shown_data_view.is_object()) e.data_view = v->shown_data_view;
                    send(e);
                    break;
                }
                case Outgoing::Kind::Unsubscribe: {   // a scene view refused for good
                    std::shared_ptr<NetView> v = view_by_key(o.key);
                    if (!v) break;
                    v->stopped = true;
                    if (!v->subscribed) break;
                    by_sub_.erase(v->sub);   // what is in flight for it is dropped
                    if (ready) {
                        wire::Unsubscribe u;
                        u.sub = v->sub;
                        send(u);
                    }
                    break;
                }
                case Outgoing::Kind::ViewRequest: {
                    std::shared_ptr<NetView> v = view_by_key(o.key);
                    if (!ready || !v || !v->subscribed) break;
                    wire::ViewRequest r;
                    r.request_id = next_request_++;
                    r.sub = v->sub;
                    r.view = v->name;
                    r.data_view = std::move(o.body);
                    send(r);
                    break;
                }
            }
        }
    }

    // ---- 3.1 "Selecting" (network thread) ----------------------------------------------------
    // The item space of dim `dim` of the source `source` ("" the first) in the
    // view's offer, "" when the dim has none. With `dim` "", the last dim (the
    // point dim of a points source).
    static std::string dim_space(const NetView& v, const std::string& source, const std::string& dim) {
        const Json* srcs = v.offer.find("sources");
        if (!srcs || !srcs->is_array()) return "";
        for (const Json& s : srcs->items()) {
            if (!source.empty() && str_or(s.find("name"), "") != source) continue;
            const Json* dims = s.find("dims");
            if (!dims || !dims->is_array() || dims->size() == 0) return "";
            if (dim.empty()) return str_or(dims->at(dims->size() - 1).find("items"), "");
            for (const Json& d : dims->items())
                if (str_or(d.find("name"), "") == dim) return str_or(d.find("items"), "");
            return "";
        }
        return "";
    }

    // The keys `space`, `items` and `truncated` of a select event, from the
    // view's item spaces and the payload axes of the snapshot it shows: the
    // row_ids of points, the source rows (else columns) of a heatmap or an
    // image, the steps of a series. A select that clears ("{}") clears the
    // view's space. A view without an item space keeps its detail unchanged.
    static void add_items(const NetView& v, Json* detail) {
        const std::string source = str_or(v.shown_data_view.find("source"), "");
        std::string space;
        std::vector<uint64_t> items;
        bool truncated = false;
        const size_t kMax = CALIPER_NET_MAX_EVENT_INDICES;
        auto take_range = [&](int64_t a, int64_t b, int64_t step) {
            if (step <= 0) step = 1;
            for (int64_t i = std::max<int64_t>(a, 0); i < b; i += step) {
                if (items.size() >= kMax) {
                    truncated = true;
                    break;
                }
                items.push_back((uint64_t)i);
            }
        };
        auto pair_of = [](const Json* p, int64_t* a, int64_t* b) {
            return p && p->is_array() && p->size() == 2 && p->at(0).get_i64(a) && p->at(1).get_i64(b);
        };
        int64_t a = 0, b = 0;
        if (v.kind == CALIPER_NET_KIND_SERIES) {
            space = str_or(v.offer.find("items"), "");
            if (pair_of(detail->find("steps"), &a, &b)) take_range(a, b, 1);
        } else if (v.kind == CALIPER_NET_KIND_POINTS) {
            space = dim_space(v, source, "");
            if (const Json* ids = detail->find("row_ids"); ids && ids->is_array())
                for (const Json& x : ids->items()) {
                    uint64_t u = 0;
                    if (!x.get_u64(&u)) continue;
                    if (items.size() >= kMax) {
                        truncated = true;
                        break;
                    }
                    items.push_back(u);
                }
            std::sort(items.begin(), items.end());
            items.erase(std::unique(items.begin(), items.end()), items.end());
            if (const Json* t = detail->find("truncated"); t && t->is_bool() && t->as_bool()) truncated = true;
        } else if (v.kind == CALIPER_NET_KIND_HEATMAP || v.kind == CALIPER_NET_KIND_IMAGE) {
            // The axes as the view layer reads them: tile, rows, columns for a
            // mosaic, else rows and columns.
            const Json& ax = v.shown_axes;
            const bool mosaic = v.shown_tile.is_object() && ax.is_array() && ax.size() == 3;
            const size_t ri = mosaic ? 1 : 0, ci = mosaic ? 2 : 1;
            const std::string row_dim = ax.is_array() && ax.size() > ri ? str_or(ax.at(ri).find("dim"), "") : "";
            const std::string col_dim = ax.is_array() && ax.size() > ci ? str_or(ax.at(ci).find("dim"), "") : "";
            const std::string row_space = row_dim.empty() ? "" : dim_space(v, source, row_dim);
            const std::string col_space = col_dim.empty() ? "" : dim_space(v, source, col_dim);
            const bool rows = !row_space.empty();
            space = rows ? row_space : col_space;
            const Json* range = detail->find(rows ? "rows" : "cols");
            const int64_t step = int_or(ax.is_array() && ax.size() > (rows ? ri : ci)
                                            ? ax.at(rows ? ri : ci).find("step") : nullptr, 1);
            if (!space.empty() && pair_of(range, &a, &b)) take_range(a, b, step);
        }
        if (space.empty()) return;
        detail->set("space", space);
        Json arr = Json::array();
        for (uint64_t i : items) arr.push(i);
        detail->set("items", arr);
        if (truncated) detail->set("truncated", true);
    }

    std::shared_ptr<NetView> view_by_key(uint64_t key) {
        std::lock_guard<std::mutex> g(mu_);
        for (const auto& v : views_)
            if (v->key == key) return v;
        return nullptr;
    }

    // The attached session's capabilities, from a hub's session list.
    void note_session_caps() {
        if (session_id_.empty()) return;
        for (const Json& e : sessions_json_.items()) {
            const Json* id = e.find("id");
            if (!id || !id->is_string() || id->as_string() != session_id_) continue;
            // An entry without the key keeps what WELCOME said (a producer's own).
            const Json* c = e.find("capabilities");
            if (!c || !c->is_array()) continue;
            std::vector<std::string> caps;
            for (const Json& x : c->items())
                if (x.is_string()) caps.push_back(x.as_string());
            std::lock_guard<std::mutex> g(mu_);
            session_caps_ = std::move(caps);
        }
    }

    void remember_identity(const wire::Identity& id) {
        std::lock_guard<std::mutex> g(mu_);
        label_ = id.id;
        script_ = id.script;
        pid_ = (int32_t)id.pid;
        start_ms_ = id.start_unix_ms;
    }

    void on_bye(const wire::Bye& b) {
        bye_reason_ = b.reason;
        std::lock_guard<std::mutex> g(mu_);
        if (b.reason == CALIPER_NET_BYE_SHUTDOWN || b.reason == CALIPER_NET_BYE_NORMAL) ended_ = true;
    }

    void on_error(const wire::Error& e) {
        if (!e.request_id && !e.sub && !e.view && !e.session &&
            (!e.in_reply_to || *e.in_reply_to == CALIPER_NET_MSG_CONTROL)) {
            // A decoder error can omit the request ID. End queued interpretation
            // requests through the reply seam, and retain the connection notice.
            ControlReply r;
            r.status = CALIPER_NET_STATUS_ERROR;
            r.error_code = e.code;
            r.error_message = e.message;
            push_reply(std::move(r));
        }
        if (e.in_reply_to && *e.in_reply_to == CALIPER_NET_MSG_CONTROL && e.request_id) {
            ControlReply r;
            r.request_id = *e.request_id;
            r.status = CALIPER_NET_STATUS_ERROR;
            r.error_code = e.code;
            r.error_message = e.message;
            push_reply(std::move(r));
            return;
        }
        if (e.in_reply_to && *e.in_reply_to == CALIPER_NET_MSG_PROGRAM && e.request_id) {
            ProgramAnswer a;   // a refused PROGRAM: its program's error, not the stream's
            a.request_id = e.request_id;
            a.error_code = e.code.empty() ? std::string(CALIPER_NET_ERR_BAD_REQUEST) : e.code;
            a.error_message = e.message;
            push_program_answer(std::move(a));
            return;
        }
        set_problem("the server reported " + e.code + (e.message.empty() ? "" : ": " + e.message));
    }

    void on_sessions(const wire::Sessions& s) {
        sessions_ = s.list;
        sessions_json_ = s.sessions;
        const std::string& want = url_.session;
        if (!want.empty()) {
            const auto it = std::find_if(sessions_.begin(), sessions_.end(),
                                         [&](const wire::SessionInfo& x) { return x.identity.id == want; });
            if (it == sessions_.end()) {
                set_problem("the server holds no session '" + want + "'");
            } else if (session_id_ != want && !attach_sent_) {
                attach_sent_ = true;
                send(wire::Attach{want});
            }
        } else if (session_id_.empty()) {
            int live = 0;
            for (const auto& x : sessions_) live += x.state == CALIPER_NET_SESSION_LIVE;
            if (live == 0) set_problem("the server holds no live session yet");
            else if (live > 1)
                set_problem("the server holds " + std::to_string(live) +
                            " live sessions; name one with session=<id> in the link");
        }
        // The attached session's state, identity and capabilities.
        for (const auto& x : sessions_) {
            if (x.identity.id != session_id_ || session_id_.empty()) continue;
            remember_identity(x.identity);
            std::lock_guard<std::mutex> g(mu_);
            ended_ = x.state == CALIPER_NET_SESSION_ENDED;
        }
        note_session_caps();
    }

    // A viewer drops its controls and selections when VIEWS names another
    // session or none (wire protocol 3.1).
    void forget_session_state() {
        std::lock_guard<std::mutex> g(mu_);
        cadences_.clear();
        have_controls_ = false;
        const uint64_t version = controls_.version + 1;
        controls_ = Controls{};
        controls_.version = version;
        if (!selections_.empty()) ++sel_version_;
        selections_.clear();
        item_axes_.clear();
    }

    void on_views(const wire::Views& v) {
        // The step is the first value of a stamp only when the first timeline
        // is the sequence `step`; the codec refuses other VIEWS, and this
        // reads nothing as the step when a timeline of another kind is first.
        const Json* first = v.timelines.is_array() && !v.timelines.items().empty() ? &v.timelines.items().front() : nullptr;
        const Json* kind = first && first->is_object() ? first->find("kind") : nullptr;
        const Json* name = first && first->is_object() ? first->find("name") : nullptr;
        timelines_listed_ = kind && kind->is_string() && kind->as_string() == CALIPER_NET_TIMELINE_SEQUENCE && name &&
                            name->is_string() && name->as_string() == "step";
        if (!v.session) {   // detached: the views keep their last pixels
            session_id_.clear();
            forget_session_state();
            std::lock_guard<std::mutex> g(mu_);
            attached_ = false;
            return;
        }
        if (!views_session_.empty() && views_session_ != *v.session) forget_session_state();
        session_id_ = *v.session;
        note_session_caps();
        session_seen_on_connection_ = true;
        bool live = true;
        for (const auto& x : sessions_)
            if (x.identity.id == session_id_) {
                remember_identity(x.identity);
                live = x.state != CALIPER_NET_SESSION_ENDED;
            }
        std::vector<std::shared_ptr<NetView>> old = views_snapshot();
        const bool same_session = views_session_ == session_id_;
        views_session_ = session_id_;
        std::vector<std::shared_ptr<NetView>> next;
        for (const wire::ViewInfo& vi : v.list) {
            if ((vi.kind < CALIPER_LIVE_KIND_HEATMAP || vi.kind > CALIPER_LIVE_KIND_HISTOGRAM) &&
                vi.kind != CALIPER_NET_KIND_SCENE)
                continue;
            std::shared_ptr<NetView> keep;
            if (same_session)
                for (const auto& o : old)
                    if (o->name == vi.name && o->view_id == vi.id && o->kind == (uint32_t)vi.kind) keep = o;
            if (!keep) {
                keep = std::make_shared<NetView>();
                keep->key = next_key_++;
                keep->name = vi.name;
                keep->kind = (uint32_t)vi.kind;
                keep->view_id = vi.id;
            }
            next.push_back(keep);
        }
        // Subscriptions of views that went away end with them.
        for (auto it = by_sub_.begin(); it != by_sub_.end();) {
            const bool still = std::find(next.begin(), next.end(), it->second) != next.end();
            it = still ? std::next(it) : by_sub_.erase(it);
        }
        // The views' offers and item spaces (3.1).
        std::vector<ItemAxis> axes;
        for (size_t i = 0; i < v.list.size() && i < v.views.size(); ++i) {
            const Json& entry = v.views.at(i);
            const std::string& name = v.list[i].name;
            if (const Json* it = entry.find("items"); it && it->is_string())
                axes.push_back({name, "", "", it->as_string()});
            for (const wire::SourceInfo& src : v.list[i].sources)
                for (const wire::DimInfo& d : src.dims)
                    if (!d.items.empty()) axes.push_back({name, src.name, d.name, d.items});
        }
        {
            std::lock_guard<std::mutex> g(mu_);
            for (const auto& view : next)
                for (size_t i = 0; i < v.list.size() && i < v.views.size(); ++i)
                    if (v.list[i].name == view->name) view->offer = v.views.at(i);
            item_axes_ = std::move(axes);
            views_ = next;
            ++views_gen_;
            if (program_fresh_ || program_session_ != session_id_) {
                program_epoch_ = ++g_program_epochs;   // the script may hold none of the programs sent before
                program_fresh_ = false;
                program_session_ = session_id_;
            }
            attached_ = true;
            ever_attached_ = true;
            ended_ = !live;
            problem_.clear();
            cv_.notify_all();
        }
        for (const auto& view : next) {
            if (view->subscribed || view->stopped) continue;
            view->sub = next_sub_++;
            view->subscribed = true;
            view->last_producer_seq = 0;
            by_sub_[view->sub] = view;
            wire::Subscribe s;
            s.sub = view->sub;
            s.view = view->name;
            s.request_id = next_request_++;
            send(s);
        }
    }

    void on_view_reply(const wire::ViewReply& r) {
        if (r.error.is_null()) {
            const auto it = by_sub_.find(r.sub);
            if (it != by_sub_.end()) {
                std::lock_guard<std::mutex> g(mu_);
                it->second->resolved = r.data_view;
            }
            return;
        }
        const Json* code = r.error.find("code");
        const Json* msg = r.error.find("message");
        set_problem("view '" + r.view + "': " + (code ? code->as_string() : "?") +
                    (msg && msg->is_string() ? ": " + msg->as_string() : ""));
        by_sub_.erase(r.sub);
    }

    void unsupported(const std::string& view, const std::string& why) {
        wire::Error e;
        e.code = CALIPER_NET_ERR_UNSUPPORTED;
        e.message = why;
        e.in_reply_to = CALIPER_NET_MSG_SNAPSHOT;
        e.view = view;
        send(e);
    }

    void on_snapshot(const wire::Snapshot& s) {
        const auto it = by_sub_.find(s.sub);
        if (it == by_sub_.end()) return;   // ended, or another subscription's late snapshot
        NetView& v = *it->second;
        if (s.kind != (int64_t)v.kind || s.kind == CALIPER_NET_KIND_SERIES) {
            set_problem("view '" + v.name + "': a snapshot of kind " + std::to_string(s.kind) +
                        " for a view of kind " + std::to_string(v.kind));
            return;
        }
        const wire::Section& d = s.data;
        if (d.encoding != CALIPER_NET_ENCODING_RAW || (s.aux && s.aux->encoding != CALIPER_NET_ENCODING_RAW)) {
            unsupported(v.name, "section encoding " + std::to_string(d.encoding));
            return;
        }
        if (s.cadence.is_object()) {   // 3.2 draft: the script's shared cadence of its snapshots
            std::lock_guard<std::mutex> g(mu_);
            cadences_[v.name] = s.cadence.dump();
        }
        for (int64_t x : d.shape)
            if (x <= 0) return;   // nothing to draw
        if (d.shape.size() > 4) return;

        // The slot layout: payload, aux, stats (64 f32).
        SlotMeta m;
        m.kind = v.kind;
        m.colormap = (int32_t)s.colormap;
        m.flags = (uint32_t)(s.flags & (CALIPER_LIVE_SNAP_AUTO_VMIN | CALIPER_LIVE_SNAP_AUTO_VMAX | CALIPER_LIVE_SNAP_RAW));
        m.vmin = (float)s.vmin;
        m.vmax = (float)s.vmax;
        m.stamp_ns = s.stamp_ns;
        // The step is the first value of the stamp, when VIEWS listed the sequence `step` first.
        if (timelines_listed_ && s.stamp && s.stamp->is(CALIPER_NET_DTYPE_I64, 1) && s.stamp->shape[0] >= 1) {
            int64_t step = 0;
            std::memcpy(&step, s.stamp->data, sizeof step);
            if (step != CALIPER_NET_STAMP_ABSENT) m.step = step;
        }
        uint64_t payload = d.size;
        int64_t n_points = 0, dims = 0;
        if (v.kind == CALIPER_LIVE_KIND_POINTS) {
            n_points = d.shape[0];
            dims = d.shape[1];
            payload = (uint64_t)n_points * 3 * sizeof(float);
            m.dtype = CALIPER_DT_F32;
            m.ndim = 2;
            m.shape[0] = n_points;
            m.shape[1] = 3;
            m.points_dims = (int32_t)dims;
            m.label_count = s.aux ? (int32_t)std::min<uint64_t>(s.label_count.value_or(0), INT32_MAX) : 0;
        } else {
            m.dtype = v.kind == CALIPER_LIVE_KIND_IMAGE ? CALIPER_DT_U8 : CALIPER_DT_F32;
            m.ndim = (int32_t)d.shape.size();
            for (size_t i = 0; i < d.shape.size(); ++i) m.shape[i] = d.shape[i];
        }
        const uint64_t aux_bytes = s.aux ? s.aux->size : 0;
        m.aux_count = s.aux ? (int64_t)(aux_bytes / sizeof(float)) : 0;
        m.aux_offset = s.aux ? align_up(payload, 256) : 0;
        m.stats_offset = align_up(s.aux ? m.aux_offset + aux_bytes : payload, 256);
        const uint64_t need = m.stats_offset + CALIPER_LIVE_STATS_BYTES;
        m.stats_count = (int32_t)std::min<size_t>(s.stats.size(), kStatsFloats);

        // A slot that is neither claimed nor the newest completed one.
        uint32_t idx = kNone;
        std::shared_ptr<NetView> hold = it->second;   // the frame thread may drop the view meanwhile
        {
            std::lock_guard<std::mutex> g(mu_);
            for (uint32_t i = 0; i < kSlots; ++i)
                if (i != v.published && i != v.reading && i != v.prev) {
                    idx = i;
                    break;
                }
        }
        if (idx == kNone) return;   // cannot happen with kSlots = 4 (two claims, one published)
        Slot& slot = v.slot[idx];
        if (slot.cap < need) {
            const uint64_t cap = align_up(need, 64 * 1024);
            id<MTLBuffer> b = [device_ newBufferWithLength:(NSUInteger)cap options:MTLResourceStorageModeShared];
            if (b == nil) {
                set_problem("view '" + v.name + "': no memory for a " + std::to_string(cap) + "-byte slot");
                return;
            }
            slot.buf = b;
            slot.cap = cap;
        }
        uint8_t* base = (uint8_t*)[slot.buf contents];
        if (v.kind == CALIPER_LIVE_KIND_POINTS) {
            float* dst = (float*)base;
            if (dims == 3) {
                std::memcpy(dst, d.data, (size_t)payload);
            } else {
                for (int64_t i = 0; i < n_points; ++i) {
                    float xy[2];
                    std::memcpy(xy, d.data + (size_t)i * 8, 8);
                    dst[i * 3 + 0] = xy[0];
                    dst[i * 3 + 1] = xy[1];
                    dst[i * 3 + 2] = 0.0f;
                }
            }
            if (s.aux) std::memcpy(base + m.aux_offset, s.aux->data, (size_t)aux_bytes);
        } else {
            std::memcpy(base, d.data, d.size);
        }
        float* stats = (float*)(base + m.stats_offset);
        for (size_t i = 0; i < kStatsFloats; ++i) stats[i] = i < s.stats.size() ? (float)s.stats[i] : NAN;

        slot.map = payload_map(s.axes, s.tile);
        v.shown_data_view = s.data_view;
        v.shown_axes = s.axes;
        v.shown_tile = s.tile;
        std::lock_guard<std::mutex> g(mu_);
        if (v.last_producer_seq && s.seq > v.last_producer_seq + 1) v.dropped += s.seq - v.last_producer_seq - 1;
        v.last_producer_seq = s.seq;
        m.seq = ++v.seq;
        m.copied = payload + aux_bytes;   // received into the slot: one copy
        slot.meta = m;
        v.published = idx;
    }

    void on_series(const wire::Series& s) {
        const auto it = by_sub_.find(s.sub);
        if (it == by_sub_.end()) return;
        NetView& v = *it->second;
        if (v.kind != CALIPER_LIVE_KIND_SERIES) return;
        if (s.values.encoding != CALIPER_NET_ENCODING_RAW) {
            unsupported(v.name, "section encoding " + std::to_string(s.values.encoding));
            return;
        }
        SeriesDelta delta;
        delta.start = s.start;
        delta.values.resize(s.values.size / sizeof(float));
        std::memcpy(delta.values.data(), s.values.data, delta.values.size() * sizeof(float));
        const uint64_t end = s.start + delta.values.size();
        std::lock_guard<std::mutex> g(mu_);
        v.series_stamp = s.stamp_ns;
        v.pending_end = std::max(v.pending_end, end);
        v.pending.push_back(std::move(delta));
        // Values below the window of the newest end can never show: drop them,
        // so a frame thread that is not polling holds at most a window's worth.
        while (!v.pending.empty() && v.pending_end > kSeriesRing &&
               v.pending.front().start + v.pending.front().values.size() <= v.pending_end - kSeriesRing)
            v.pending.pop_front();
        ++v.seq;
    }

    // ---- state ------------------------------------------------------------------------------
    const StreamUrl url_;
    id<MTLDevice> device_ = nil;
    std::unique_ptr<Client> client_;   // the network thread's; wake() from any thread
    std::thread thread_;
    std::atomic<bool> stop_{false};
    bool timelines_listed_ = false;   // the newest VIEWS listed the sequence `step` first (network thread only)

    mutable std::mutex mu_;
    std::condition_variable cv_;
    // Guarded by mu_.
    std::vector<std::shared_ptr<NetView>> views_;
    uint64_t views_gen_ = 0;
    std::string label_, script_, problem_;
    uint64_t start_ms_ = 0;
    int32_t pid_ = 0;
    bool connected_ = false, attached_ = false, ever_attached_ = false, ended_ = false, failed_ = false;
    // 3.1, guarded by mu_.
    bool server_controls_ = false, welcome_control_ = false, have_controls_ = false;
    bool server_selection_ = false;   // network thread only after WELCOME
    std::vector<std::string> session_caps_;
    // 3.2 draft, GPU programs; guarded by mu_.
    bool server_programs_ = false, program_fresh_ = false;
    uint64_t program_epoch_ = 0;
    std::string program_session_;
    std::deque<ProgramAnswer> program_answers_;
    // 3.2 draft, resources; guarded by mu_: the server announced `resources`, and the newest
    // received version of each "<program>/<resource>" with the order in which it began to wait.
    bool server_resources_ = false;
    std::map<std::string, std::pair<uint64_t, ReceivedResource>> resources_;
    uint64_t resource_order_ = 0;
    wire::SplitResources resource_parts_;   // versions in parts being put together (network thread)
    ScenePieceInbox scene_inbox_;           // scene programs' versions until their update (network thread)
    std::atomic<uint64_t> resource_parts_refused_{0};
    Controls controls_;
    std::deque<ControlReply> replies_;
    std::map<std::string, LinkedSelection> selections_;
    uint64_t sel_version_ = 0;
    std::vector<ItemAxis> item_axes_;
    std::map<std::string, std::string> cadences_;   // 3.2 draft: view name -> its cadence (JSON)
    // The frame thread's requests for the network thread.
    std::mutex out_mu_;
    std::deque<Outgoing> outbox_;
    std::atomic<uint32_t> next_control_id_{1};

    // Frame thread only.
    uint64_t seen_gen_ = UINT64_MAX;
    std::vector<std::shared_ptr<NetView>> frame_views_;

    // Network thread only.
    std::map<uint32_t, std::shared_ptr<NetView>> by_sub_;
    std::vector<wire::SessionInfo> sessions_;
    Json sessions_json_ = Json::array();   // the same list as sent, for the capabilities
    std::string session_id_;        // attached on this connection ("" = not attached)
    std::string views_session_;     // the session the views belong to (kept across reconnects)
    std::string bye_reason_;
    bool welcomed_ = false, attach_sent_ = false, session_seen_on_connection_ = false;
    uint32_t next_sub_ = 1, next_request_ = 1, ping_nonce_ = 0;
    uint64_t next_key_ = 1;
};

}  // namespace

bool parse_stream_url(const std::string& in, StreamUrl* out, std::string* err) {
    std::string url = in;
    while (!url.empty() && (url.back() == ' ' || url.back() == '\n' || url.back() == '\r' || url.back() == '\t'))
        url.pop_back();
    while (!url.empty() && (url.front() == ' ' || url.front() == '\t')) url.erase(0, 1);
    std::string lower = url;
    for (char& c : lower) c = (char)std::tolower((unsigned char)c);
    bool page = false;
    if (lower.rfind("http://", 0) == 0) {
        url = "ws://" + url.substr(7);   // the page link: same host, port and token
        page = true;
    } else if (lower.rfind("https://", 0) == 0 || lower.rfind("wss://", 0) == 0) {
        *err = "the stream has no TLS: use the ws:// or http:// link on 127.0.0.1";
        return false;
    }
    caliper_net::WsUrl u;
    if (!caliper_net::parse_ws_url(url, &u, err)) return false;
    if (u.host != "127.0.0.1" && u.host != "localhost") {
        *err = "the stream is loopback only (127.0.0.1 or localhost), not " + u.host;
        return false;
    }
    if (!page && u.path != CALIPER_NET_PATH) {
        *err = "the stream path is " CALIPER_NET_PATH ", not " + u.path;
        return false;
    }
    caliper_net::HttpRequest q;
    q.query = u.query;
    StreamUrl s;
    if (!q.query_param(CALIPER_NET_TOKEN_PARAM, &s.token)) {
        *err = "the link has no token (…?token=<session token>)";
        return false;
    }
    if (!valid_token(s.token)) {
        *err = "the token must be 22 to 128 characters of A-Z, a-z, 0-9, _ and -";
        return false;
    }
    if (q.query_param("session", &s.session) && !valid_session_id(s.session)) {
        *err = "session must be 1 to 63 characters of A-Z, a-z, 0-9, ., _ and -";
        return false;
    }
    s.host = u.host;
    s.port = u.port;
    s.display = "ws://" + u.host + ":" + std::to_string(u.port) + CALIPER_NET_PATH;
    s.ws_url = s.display + "?" CALIPER_NET_TOKEN_PARAM "=" + s.token;
    *out = s;
    return true;
}

std::unique_ptr<NetSource> open_net_source(const std::string& url, void* mtl_device, std::string* err) {
    StreamUrl u;
    if (!parse_stream_url(url, &u, err)) return nullptr;
    if (!mtl_device) {
        *err = "no Metal device";
        return nullptr;
    }
    return std::make_unique<NetSourceImpl>(u, (__bridge id<MTLDevice>)mtl_device);
}

}  // namespace caliper_live
