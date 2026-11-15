// Caliper API v0 (api_server.h): the routes, the checks of a plain request,
// and the server around an embedded hub.
#include "api/api_server.h"

#include "net/json.h"
#include "net/wire.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

namespace caliper_api {

using caliper_net::ConnId;
using caliper_net::HttpRequest;

namespace {

const char kPrefix[] = "/v0/";
const char kLaunchHead[] = "/v0/applets/";
const char kLaunchTail[] = "/launch";

bool starts_with(const std::string& s, const char* p) {
    const size_t n = std::char_traits<char>::length(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

bool ends_with(const std::string& s, const char* p) {
    const size_t n = std::char_traits<char>::length(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

}  // namespace

// ---- routing ------------------------------------------------------------------------------

Routed route(const std::string& method, const std::string& path) {
    Routed r;
    if (path == "/v0/info") {
        r.route = Route::Info;
    } else if (path == "/v0/applets") {
        r.route = Route::Applets;
    } else if (path == "/v0/capture.png") {
        r.route = Route::Capture;
    } else if (path == kStreamPath) {
        r.route = Route::Stream;
    } else if (starts_with(path, kLaunchHead) && ends_with(path, kLaunchTail)) {
        const size_t a = sizeof kLaunchHead - 1;
        const size_t b = path.size() - (sizeof kLaunchTail - 1);
        if (b > a && path.find('/', a) == b) {
            r.route = Route::Launch;
            r.applet_id = path.substr(a, b - a);
        }
    }
    if (r.route == Route::None) return r;
    r.allow = r.route == Route::Launch ? "POST" : "GET";
    r.method_ok = method == r.allow;
    return r;
}

bool plausible_applet_id(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
              c == '_' || c == '-'))
            return false;
    return true;
}

bool token_ok(const HttpRequest& r, const std::string& token, bool query_ok) {
    if (token.empty()) return false;
    bool has_header = false;
    for (const auto& h : r.headers)
        if (h.first == "authorization") has_header = true;
    if (has_header) {
        // "Bearer <token>", the scheme in any case. Only the first header counts.
        const std::string v = trim(r.header("authorization"));
        const size_t sp = v.find_first_of(" \t");
        if (sp == std::string::npos || lower(v.substr(0, sp)) != "bearer") return false;
        return caliper_net::wire::token_equal(trim(v.substr(sp + 1)), token);
    }
    std::string q;
    if (!query_ok || !r.query_param(CALIPER_NET_TOKEN_PARAM, &q)) return false;
    return caliper_net::wire::token_equal(q, token);
}

Verdict check(const HttpRequest& r, const std::string& token) {
    Verdict v;
    if (r.has_origin && !caliper_net::wire::origin_allowed(r.origin)) {
        v.status = 403;
        v.message = "this Origin is not allowed";
        return v;
    }
    v.routed = route(r.method, r.path);
    if (v.routed.route == Route::None) {
        v.status = 404;
        v.message = starts_with(r.path, kPrefix) ? "no such endpoint in Caliper API v0"
                                                 : "Caliper API serves only /v0/";
        return v;
    }
    if (!v.routed.method_ok) {
        v.status = 405;
        v.message = std::string(r.path) + " takes " + v.routed.allow;
        return v;
    }
    if (!token_ok(r, token, r.method == "GET")) {
        v.status = 401;
        v.message = r.method == "GET" ? "the session token is missing or wrong"
                                      : "the session token is missing or wrong (a POST takes it in "
                                        "the Authorization header only)";
        return v;
    }
    const std::string length = trim(r.header("content-length"));
    if ((!length.empty() && length != "0") || !r.header("transfer-encoding").empty()) {
        v.status = 413;
        v.message = "Caliper API v0 takes no request body";
        return v;
    }
    if (v.routed.route == Route::Launch && !plausible_applet_id(v.routed.applet_id)) {
        v.status = 400;
        v.message = "not an applet id: an id is 1 to 128 characters of A-Z, a-z, 0-9, '.', '_' and '-'";
        return v;
    }
    return v;
}

std::string error_json(const std::string& message) {
    caliper_net::Json j = caliper_net::Json::object();
    j.set("error", message);
    return j.dump();
}

// ---- the server ---------------------------------------------------------------------------

struct Server::Impl {
    struct Pending {
        ConnId conn = 0;
        Call::Kind kind = Call::Kind::Applets;
        uint64_t deadline_ms = 0;
    };
    struct Encode {
        uint64_t ticket = 0;
        uint32_t w = 0, h = 0;
        std::vector<uint8_t> rgba;
    };

    const Options& opt;
    std::unique_ptr<caliper_net::Hub> hub;
    std::thread hub_thread;

    mutable std::mutex mu;               // everything below
    std::map<uint64_t, Pending> pending;
    std::vector<Call> queue;             // not yet taken by the frame thread
    uint64_t next_ticket = 1;
    bool stopping = false;
    Stats st;
    std::atomic<size_t> queued{0};       // queue.size(), read without the lock

    std::mutex enc_mu;
    std::condition_variable enc_cv;
    std::deque<Encode> enc_jobs;
    bool enc_stop = false;
    std::thread enc_thread;

    explicit Impl(const Options& o) : opt(o) {}

    // Every answer of a plain request goes through here. The WebSocket layer
    // adds Cache-Control: no-store and Connection: close to every response.
    void answer(ConnId c, int status, const std::string& type, std::string body,
                std::vector<std::pair<std::string, std::string>> extra = {}) {
        extra.emplace_back("X-Content-Type-Options", "nosniff");
        {
            std::lock_guard<std::mutex> g(mu);
            if (status >= 200 && status < 300) ++st.answered;
            else if (status >= 400 && status < 500) ++st.refused;
            else if (status == 503) ++st.busy;
            else if (status == 504) ++st.timeouts;
        }
        hub->respond(c, status, type, std::move(body), std::move(extra));
    }

    void answer_error(ConnId c, int status, const std::string& message,
                      std::vector<std::pair<std::string, std::string>> extra = {}) {
        answer(c, status, "application/json", error_json(message), std::move(extra));
    }

    // The hub's thread: a request on any path but the stream's.
    void on_request(ConnId c, const HttpRequest& r) {
        {
            std::lock_guard<std::mutex> g(mu);
            ++st.requests;
        }
        const Verdict v = check(r, opt.token);
        if (v.status == 401) {
            answer_error(c, 401, v.message, {{"WWW-Authenticate", "Bearer realm=\"caliper\""}});
            return;
        }
        if (v.status == 405) {
            answer_error(c, 405, v.message, {{"Allow", v.routed.allow}});
            return;
        }
        if (v.status != 0) {
            answer_error(c, v.status, v.message);
            return;
        }
        Call call;
        switch (v.routed.route) {
            case Route::Info:
                answer(c, 200, "application/json", opt.info_json);
                return;
            case Route::Capture:
                if (!opt.capture_unavailable.empty()) {
                    answer_error(c, 501, opt.capture_unavailable);
                    return;
                }
                call.kind = Call::Kind::Capture;
                break;
            case Route::Applets:
                call.kind = Call::Kind::Applets;
                break;
            case Route::Launch:
                call.kind = Call::Kind::Launch;
                call.applet_id = v.routed.applet_id;
                break;
            default:   // the stream path never reaches here (http_door.h)
                answer_error(c, 404, "no such endpoint in Caliper API v0");
                return;
        }
        std::string busy;
        {
            std::lock_guard<std::mutex> g(mu);
            if (stopping) {
                busy = "the app is closing";
            } else if (pending.size() >= opt.max_pending) {
                busy = "the app is busy: " + std::to_string(pending.size()) +
                       " requests are waiting for it";
            } else {
                call.ticket = next_ticket++;
                pending[call.ticket] = {c, call.kind,
                                        caliper_net::monotonic_ms() + (uint64_t)opt.frame_timeout_ms};
                queue.push_back(call);
                queued.store(queue.size());
            }
        }
        if (!busy.empty()) answer_error(c, 503, busy, {{"Retry-After", "1"}});
    }

    // The hub's thread, every round: calls past their deadline get 504.
    void on_tick() {
        const uint64_t now = caliper_net::monotonic_ms();
        std::vector<ConnId> late;
        {
            std::lock_guard<std::mutex> g(mu);
            for (auto it = pending.begin(); it != pending.end();) {
                if (it->second.deadline_ms > now) {
                    ++it;
                    continue;
                }
                late.push_back(it->second.conn);
                const uint64_t t = it->first;
                queue.erase(std::remove_if(queue.begin(), queue.end(),
                                           [t](const Call& q) { return q.ticket == t; }),
                            queue.end());
                it = pending.erase(it);
            }
            queued.store(queue.size());
        }
        const std::string why = "the app did not answer within " +
                                std::to_string(opt.frame_timeout_ms) + " ms";
        for (ConnId c : late) answer_error(c, 504, why);
    }

    // Takes a call out of `pending`; false when it timed out or was answered.
    bool claim(uint64_t ticket, ConnId* conn) {
        std::lock_guard<std::mutex> g(mu);
        auto it = pending.find(ticket);
        if (it == pending.end()) return false;
        *conn = it->second.conn;
        pending.erase(it);
        return true;
    }

    void encoder() {
        for (;;) {
            Encode job;
            {
                std::unique_lock<std::mutex> g(enc_mu);
                enc_cv.wait(g, [this] { return enc_stop || !enc_jobs.empty(); });
                if (enc_stop) return;
                job = std::move(enc_jobs.front());
                enc_jobs.pop_front();
            }
            {
                // Skip the work for a call that has timed out meanwhile.
                std::lock_guard<std::mutex> g(mu);
                if (!pending.count(job.ticket)) continue;
            }
            std::string png;
            const bool ok = opt.encode_png && opt.encode_png(job.rgba.data(), job.w, job.h, &png);
            ConnId c = 0;
            if (!claim(job.ticket, &c)) continue;
            if (ok) answer(c, 200, "image/png", std::move(png));
            else answer_error(c, 500, "the frame could not be encoded as a PNG");
        }
    }
};

Server::Server(Options o) : opt_(std::move(o)) {}

Server::~Server() { stop(); }

bool Server::start(std::string* err) {
    if (impl_) {
        *err = "Caliper API is running already";
        return false;
    }
    if (!caliper_net::valid_token(opt_.token)) {
        *err = "the session token must be 22 to 128 characters of A-Z, a-z, 0-9, '_' and '-'";
        return false;
    }
    auto m = std::make_unique<Impl>(opt_);
    Impl* p = m.get();
    caliper_net::HubOptions h;
    h.port = opt_.port;
    h.token = opt_.token;
    h.software_version = opt_.software_version;
    h.log = opt_.log;
    h.stream_path = kStreamPath;
    h.on_request = [p](ConnId c, const HttpRequest& r) { p->on_request(c, r); };
    h.on_tick = [p] { p->on_tick(); };
    h.tick_ms = 50;
    m->hub = std::make_unique<caliper_net::Hub>(std::move(h));
    if (!m->hub->listen(err)) return false;
    m->enc_thread = std::thread([p] { p->encoder(); });
    m->hub_thread = std::thread([p] { p->hub->run(); });
    impl_ = std::move(m);
    return true;
}

void Server::stop() {
    if (!impl_) return;
    Impl& m = *impl_;
    std::vector<ConnId> waiting;
    {
        std::lock_guard<std::mutex> g(m.mu);
        m.stopping = true;
        for (auto& kv : m.pending) waiting.push_back(kv.second.conn);
        m.pending.clear();
        m.queue.clear();
        m.queued.store(0);
    }
    for (ConnId c : waiting) m.answer_error(c, 503, "the app is closing");
    {
        std::lock_guard<std::mutex> g(m.enc_mu);
        m.enc_stop = true;
    }
    m.enc_cv.notify_all();
    if (m.enc_thread.joinable()) m.enc_thread.join();
    m.hub->stop();
    if (m.hub_thread.joinable()) m.hub_thread.join();
    impl_.reset();
}

uint16_t Server::port() const { return impl_ ? impl_->hub->port() : 0; }

std::string Server::base_url() const { return "http://127.0.0.1:" + std::to_string(port()) + "/v0"; }

std::string Server::stream_url() const {
    return "ws://127.0.0.1:" + std::to_string(port()) + kStreamPath;
}

std::vector<Call> Server::take() {
    std::vector<Call> out;
    if (!impl_ || impl_->queued.load() == 0) return out;
    std::lock_guard<std::mutex> g(impl_->mu);
    out.swap(impl_->queue);
    impl_->queued.store(0);
    return out;
}

void Server::reply(uint64_t ticket, int status, std::string json) {
    if (!impl_) return;
    ConnId c = 0;
    if (!impl_->claim(ticket, &c)) return;
    impl_->answer(c, status, "application/json", std::move(json));
}

void Server::reply_capture(uint64_t ticket, uint32_t w, uint32_t h, std::vector<uint8_t> rgba) {
    if (!impl_) return;
    {
        std::lock_guard<std::mutex> g(impl_->mu);
        if (!impl_->pending.count(ticket)) return;
    }
    {
        std::lock_guard<std::mutex> g(impl_->enc_mu);
        impl_->enc_jobs.push_back({ticket, w, h, std::move(rgba)});
    }
    impl_->enc_cv.notify_one();
}

Stats Server::stats() const {
    if (!impl_) return {};
    std::lock_guard<std::mutex> g(impl_->mu);
    Stats s = impl_->st;
    s.pending = impl_->pending.size();
    return s;
}

caliper_net::HubStats Server::hub_stats() const { return impl_ ? impl_->hub->stats() : caliper_net::HubStats(); }

}  // namespace caliper_api
