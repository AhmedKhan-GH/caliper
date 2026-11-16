// caliper_attention_producer: a test and demo tool, not installed. It stands
// in for a training script that streams its attention over wire protocol v3
// (docs/references/wire-protocol.md), with no torch and no Python: a hub of
// its own on 127.0.0.1 (the hub of `caliper serve`, src/host/net/hub.h) and a
// producer connection to it, which publishes, about `--hz` times a second, the
// attention of a model whose heads sharpen as it trains, deeper layers
// sooner, so that the entropy falls layer by layer and over the steps.
//
//   caliper_attention_producer [--layers 6] [--heads 8] [--tokens 64] [--hz 8]
//                              [--stream-file <data dir>]
//
// It prints `[producer] hub <link>` once its hub listens; the app attaches to
// that link with --url. With --stream-file it also writes the stream file
// that caliper.serve() writes (python/caliper/_net.py, StreamFile), as
// <data dir>/streams/<pid>.json with mode 0600, so that the app's session
// list (F4) lists the stream, and removes it when it stops. SIGTERM or SIGINT
// stops it. Its views:
//
//   attn          heatmap of the source attn [layer, head, query, key], whose
//                 default data view reduces the heads by their mean on the
//                 producer and tiles the layers: the app receives one T x T
//                 cell per layer, f32 [layers, T, T], the head-mean attention
//   attn.entropy  heatmap of the source entropy [layer, head], the mean over
//                 the queries of each head's attention entropy, in nats,
//                 computed here, since a heatmap data view carries at most
//                 three dims and the spec cannot lay cells back out
//   entropy.l<k>  a series per layer: that layer's mean entropy over its
//                 heads, one value per step, so a viewer has its history
#include "caliper/live_net.h"
#include "net/hub.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace wire = caliper_net::wire;
using caliper_net::Json;

namespace {

volatile std::sig_atomic_t g_stop = 0;
extern "C" void on_signal(int) { g_stop = 1; }

uint64_t mono_ns() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

Json obj(std::initializer_list<std::pair<const char*, Json>> kv) {
    Json j = Json::object();
    for (auto& x : kv) j.set(x.first, x.second);
    return j;
}

Json dim(const char* name, int64_t size) { return obj({{"name", Json(name)}, {"size", Json(size)}}); }

Json axis(const char* dim, int64_t count) {
    return obj({{"dim", Json(dim)}, {"start", Json(0)}, {"step", Json(1)}, {"count", Json(count)}, {"size", Json(count)}});
}

struct Model {
    int L = 6, H = 8, T = 64;
    std::mt19937 rng{42};
    std::vector<float> attn;     // [L, H, T, T]
    std::vector<float> entropy;  // [L, H]

    // One training step: each head mixes a pattern of its own (a fixed
    // offset back, the first token, or itself) with uniform attention over
    // the causal keys; the pattern's weight grows with the step, faster in
    // deeper layers, so the rows sharpen and their entropy falls.
    void step(uint64_t s) {
        attn.assign((size_t)L * H * T * T, 0.0f);
        entropy.assign((size_t)L * H, 0.0f);
        std::normal_distribution<float> noise(0.0f, 0.15f);
        for (int l = 0; l < L; ++l)
            for (int h = 0; h < H; ++h) {
                const double pace = 120.0 / (1.0 + 0.6 * l);   // steps to half strength
                const double strength = 6.0 * (double)s / ((double)s + pace) * (0.75 + 0.05 * h);
                double ent_sum = 0.0;
                for (int q = 0; q < T; ++q) {
                    float* row = &attn[(((size_t)l * H + h) * T + q) * T];
                    const int kind = (h + l) % 4;
                    const int target = kind == 0 ? std::max(0, q - 1 - h % 3)
                                       : kind == 1 ? 0
                                       : kind == 2 ? q
                                                   : (q / 8) * 8;
                    double sum = 0.0;
                    for (int k = 0; k <= q; ++k) {
                        double logit = noise(rng);
                        if (k == target) logit += strength;
                        else if (std::abs(k - target) == 1) logit += 0.4 * strength;
                        row[k] = (float)std::exp(logit);
                        sum += row[k];
                    }
                    double ent = 0.0;
                    for (int k = 0; k <= q; ++k) {
                        row[k] = (float)(row[k] / sum);
                        if (row[k] > 0.0f) ent -= row[k] * std::log((double)row[k]);
                    }
                    ent_sum += ent;
                }
                entropy[(size_t)l * H + h] = (float)(ent_sum / T);
            }
    }

    // The head-mean attention laid out as the tile mosaic of the wire
    // protocol: cols = ceil(sqrt(L)), rows = ceil(L / cols), cell l at row
    // l / cols and column l % cols, the cells left over NaN.
    std::vector<float> mosaic(int* rows, int* cols) const {
        *cols = (int)std::ceil(std::sqrt((double)L));
        *rows = (L + *cols - 1) / *cols;
        const int W = *cols * T;
        std::vector<float> m((size_t)*rows * T * W, std::nanf(""));
        for (int l = 0; l < L; ++l) {
            const int r0 = (l / *cols) * T, c0 = (l % *cols) * T;
            for (int q = 0; q < T; ++q)
                for (int k = 0; k < T; ++k) {
                    double v = 0.0;
                    for (int h = 0; h < H; ++h) v += attn[(((size_t)l * H + h) * T + q) * T + k];
                    m[(size_t)(r0 + q) * W + c0 + k] = (float)(v / H);
                }
        }
        return m;
    }
};

class Producer {
public:
    Producer(const Model& m) : model_(m) {}

    bool connect(const std::string& url) {
        std::string err;
        if (!client_.connect(url, &err)) {
            std::fprintf(stderr, "[producer] cannot connect: %s\n", err.c_str());
            return false;
        }
        return true;
    }

    bool closed() const { return closed_; }

    void poll(int ms) {
        for (auto& e : client_.poll(ms)) {
            using T = caliper_net::Event::Type;
            if (e.type == T::Open) {
                hello();
            } else if (e.type == T::Closed) {
                closed_ = true;
            } else if (e.type == T::Message) {
                wire::Message msg;
                const wire::DecodeResult r = wire::decode(e.data.data(), e.data.size(), &msg);
                if (r.ok()) on(msg);
            }
        }
    }

    // Everything of one step to the open subscriptions.
    void publish(uint64_t step) {
        ++seq_;
        const int L = model_.L, H = model_.H, T = model_.T;
        std::vector<float> layer_mean((size_t)L, 0.0f);
        for (int l = 0; l < L; ++l) {
            for (int h = 0; h < H; ++h) layer_mean[(size_t)l] += model_.entropy[(size_t)l * H + h] / (float)H;
            series_[l].push_back(layer_mean[(size_t)l]);
        }
        int rows = 0, cols = 0;
        const std::vector<float> m = model_.mosaic(&rows, &cols);
        for (auto& [sub, s] : subs_) {
            if (s.view == "attn") {
                wire::Snapshot snap = base(sub, s);
                snap.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {(int64_t)rows * T, (int64_t)cols * T},
                                                m.data(), m.size() * 4);
                snap.axes = Json::array();
                snap.axes.push(axis("layer", L));
                snap.axes.push(axis("query", T));
                snap.axes.push(axis("key", T));
                snap.tile = obj({{"dim", Json("layer")}, {"rows", Json(rows)}, {"cols", Json(cols)}});
                snap.stats = range(m);
                send(snap);
            } else if (s.view == "attn.entropy") {
                wire::Snapshot snap = base(sub, s);
                snap.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {(int64_t)L, (int64_t)H},
                                                model_.entropy.data(), model_.entropy.size() * 4);
                snap.axes = Json::array();
                snap.axes.push(axis("layer", L));
                snap.axes.push(axis("head", H));
                snap.stats = range(model_.entropy);
                send(snap);
            } else if (s.view.rfind("entropy.l", 0) == 0) {
                const int l = std::atoi(s.view.c_str() + 9);
                send_series(sub, s.view, l, series_[l].size() - 1, series_[l].size());
            }
        }
        (void)step;
    }

private:
    struct Sub {
        std::string view;
        Json resolved;
    };

    void send(const wire::Message& m) {
        std::vector<uint8_t> b;
        if (wire::encode(m, &b)) client_.send(b.data(), b.size());
    }

    void hello() {
        wire::Hello h;
        h.role = CALIPER_NET_ROLE_PRODUCER;
        wire::Identity id;
        id.id = "attention-" + std::to_string((long)getpid());
        id.script = "attention_train.py";
        id.pid = (uint64_t)getpid();
        id.start_unix_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
        h.session = id.to_json();
        send(h);
    }

    Json views() const {
        const int L = model_.L, H = model_.H, T = model_.T;
        Json v = Json::array();
        Json attn_dims = Json::array();
        attn_dims.push(dim("layer", L));
        attn_dims.push(dim("head", H));
        attn_dims.push(dim("query", T));
        attn_dims.push(dim("key", T));
        Json attn_src = Json::array();
        attn_src.push(obj({{"name", Json("attn")}, {"dtype", Json("f32")}, {"dims", attn_dims}}));
        v.push(obj({{"id", Json(1)},
                    {"name", Json("attn")},
                    {"kind", Json((int)CALIPER_NET_KIND_HEATMAP)},
                    {"sources", attn_src},
                    {"default", obj({{"reduce", obj({{"head", Json("mean")}})}, {"tile", Json("layer")}})}}));
        Json ent_dims = Json::array();
        ent_dims.push(dim("layer", L));
        ent_dims.push(dim("head", H));
        Json ent_src = Json::array();
        ent_src.push(obj({{"name", Json("entropy")}, {"dtype", Json("f32")}, {"dims", ent_dims}}));
        v.push(obj({{"id", Json(2)}, {"name", Json("attn.entropy")}, {"kind", Json((int)CALIPER_NET_KIND_HEATMAP)},
                    {"sources", ent_src}}));
        for (int l = 0; l < L; ++l)
            v.push(obj({{"id", Json(3 + l)},
                        {"name", Json("entropy.l" + std::to_string(l))},
                        {"kind", Json((int)CALIPER_NET_KIND_SERIES)},
                        {"sources", Json::array()}}));
        return v;
    }

    // The resolved data view of a subscription: the default, which is all
    // this producer offers; a request for any other is refused.
    bool resolve(const std::string& view, const Json& requested, Json* out, std::string* why) const {
        const bool plain = !requested.is_object() || requested.size() == 0;
        if (view == "attn") {
            if (!plain && !(requested.has("tile") || requested.has("reduce") || requested.has("budget"))) {
                *why = "this producer serves the default data view of attn only";
                return false;
            }
            *out = obj({{"source", Json("attn")},
                        {"index", Json::object()},
                        {"reduce", obj({{"head", Json("mean")}})},
                        {"tile", Json("layer")},
                        {"budget", obj({{"max_extent", Json(512)}, {"pool", Json("mean")}})}});
            return true;
        }
        if (view == "attn.entropy") {
            *out = obj({{"source", Json("entropy")},
                        {"index", Json::object()},
                        {"reduce", Json::object()},
                        {"budget", obj({{"max_extent", Json(512)}, {"pool", Json("mean")}})}});
            return true;
        }
        if (view.rfind("entropy.l", 0) == 0) {
            *out = obj({{"budget", obj({{"max_backfill", Json((int64_t)1 << 20)}})}});
            return true;
        }
        *why = "no view " + view;
        return false;
    }

    void on(const wire::Message& msg) {
        if (std::get_if<wire::Welcome>(&msg)) {
            wire::Views v;
            v.views = views();
            send(v);
        } else if (auto* s = std::get_if<wire::Subscribe>(&msg)) {
            wire::ViewReply r;
            r.request_id = s->request_id;
            r.sub = s->sub;
            r.view = s->view;
            Json resolved;
            std::string why;
            if (!resolve(s->view, s->data_view, &resolved, &why)) {
                r.error = obj({{"code", Json(CALIPER_NET_ERR_BAD_DATA_VIEW)}, {"message", Json(why)}});
                send(r);
                return;
            }
            r.data_view = resolved;
            subs_[s->sub] = Sub{s->view, resolved};
            send(r);
            if (s->view.rfind("entropy.l", 0) == 0) {
                const int l = std::atoi(s->view.c_str() + 9);
                send_series(s->sub, s->view, l, 0, series_[l].size());   // the backfill
            }
        } else if (auto* u = std::get_if<wire::Unsubscribe>(&msg)) {
            subs_.erase(u->sub);
        } else if (auto* p = std::get_if<wire::Ping>(&msg)) {
            send(wire::Pong{p->nonce});
        }
    }

    // The data's range before pooling, the empty cells left out, for autoscale.
    static std::vector<double> range(const std::vector<float>& v) {
        double lo = INFINITY, hi = -INFINITY;
        for (float x : v)
            if (std::isfinite(x)) {
                lo = std::min(lo, (double)x);
                hi = std::max(hi, (double)x);
            }
        if (!(lo <= hi)) lo = hi = 0.0;
        return {lo, hi};
    }

    wire::Snapshot base(uint32_t sub, const Sub& s) const {
        wire::Snapshot snap;
        snap.sub = sub;
        snap.view = s.view;
        snap.seq = seq_;
        snap.stamp_ns = mono_ns();
        snap.kind = CALIPER_NET_KIND_HEATMAP;
        snap.data_view = s.resolved;
        snap.colormap = CALIPER_NET_COLORMAP_MAGMA;
        snap.flags = CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX;
        return snap;
    }

    void send_series(uint32_t sub, const std::string& view, int l, size_t a, size_t b) {
        const std::vector<float>& v = series_[l];
        while (a < b) {
            const size_t e = std::min(b, a + (size_t)CALIPER_NET_SERIES_MAX_VALUES);
            wire::Series s;
            s.sub = sub;
            s.view = view;
            s.start = a;
            s.total = v.size();
            s.stamp_ns = mono_ns();
            s.values = wire::Section::make("values", CALIPER_NET_DTYPE_F32, {(int64_t)(e - a)}, v.data() + a, (e - a) * 4);
            send(s);
            a = e;
        }
    }

    const Model& model_;
    caliper_net::Client client_;
    bool closed_ = false;
    uint64_t seq_ = 0;
    std::map<uint32_t, Sub> subs_;
    std::map<int, std::vector<float>> series_;
};

// The stream file of caliper.serve() for this stream: <dir>/streams/<pid>.json,
// mode 0600 in a folder of mode 0700, with the keys the app reads. Its path,
// or "" when it could not be written.
std::string write_stream_file(const std::string& dir, uint16_t port, const std::string& url, const Model& m) {
    const std::string folder = dir + "/streams";
    ::mkdir(dir.c_str(), 0700);
    ::mkdir(folder.c_str(), 0700);
    char host[256] = {0};
    ::gethostname(host, sizeof host - 1);
    Json views = Json::array();
    views.push(Json("attn"));
    views.push(Json("attn.entropy"));
    for (int l = 0; l < m.L; ++l) views.push(Json("entropy.l" + std::to_string(l)));
    const uint64_t now = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
    Json f = obj({{"version", Json(1)},
                  {"pid", Json((int64_t)getpid())},
                  {"host", Json(std::string(host))},
                  {"session", Json("attention-" + std::to_string((long)getpid()))},
                  {"script", Json("attention_train.py")},
                  {"start_unix_ms", Json(now)},
                  {"port", Json((int64_t)port)},
                  {"stream_url", Json(url)},
                  {"control", Json(false)},
                  {"views", views}});
    const std::string path = folder + "/" + std::to_string((long)getpid()) + ".json";
    const std::string text = f.dump() + "\n";
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return {};
    const bool ok = ::write(fd, text.data(), text.size()) == (ssize_t)text.size();
    ::close(fd);
    if (!ok) {
        ::unlink(path.c_str());
        return {};
    }
    return path;
}

}  // namespace

int main(int argc, char** argv) {
    Model model;
    double hz = 8.0;
    std::string stream_dir;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--layers") model.L = std::max(1, std::min(32, std::atoi(argv[i + 1])));
        else if (a == "--heads") model.H = std::max(1, std::min(64, std::atoi(argv[i + 1])));
        else if (a == "--tokens") model.T = std::max(2, std::min(256, std::atoi(argv[i + 1])));
        else if (a == "--hz") hz = std::max(0.5, std::min(60.0, std::atof(argv[i + 1])));
        else if (a == "--stream-file") stream_dir = argv[i + 1];
        else {
            std::fprintf(stderr, "usage: caliper_attention_producer [--layers N] [--heads N] [--tokens N] [--hz F] "
                                 "[--stream-file DIR]\n");
            return 2;
        }
    }
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);

    // The hub, on a thread of its own, with a token of this run.
    caliper_net::HubOptions o;
    std::random_device rd;
    char token[33];
    std::snprintf(token, sizeof token, "%08x%08x%08x%08x", rd(), rd(), rd(), rd());
    o.token = token;
    caliper_net::Hub hub(o);
    std::string err;
    if (!hub.listen(&err)) {
        std::fprintf(stderr, "[producer] the hub cannot listen: %s\n", err.c_str());
        return 1;
    }
    std::thread hub_thread([&] { hub.run(); });
    const std::string url = "ws://127.0.0.1:" + std::to_string(hub.port()) + "/v3/stream?token=" + token;

    Producer p(model);
    std::string stream_file;
    int rc = 0;
    if (!p.connect(url)) {
        rc = 1;
    } else {
        if (!stream_dir.empty()) stream_file = write_stream_file(stream_dir, hub.port(), url, model);
        std::printf("[producer] hub %s\n", url.c_str());
        std::fflush(stdout);
        const auto period = std::chrono::nanoseconds((int64_t)(1e9 / hz));
        auto next = std::chrono::steady_clock::now();
        uint64_t step = 0;
        while (!g_stop && !p.closed()) {
            p.poll(5);
            if (std::chrono::steady_clock::now() < next) continue;
            next += period;
            model.step(++step);
            p.publish(step);
            if (step % 50 == 0) {
                std::printf("[producer] step %llu\n", (unsigned long long)step);
                std::fflush(stdout);
            }
        }
    }
    if (!stream_file.empty()) ::unlink(stream_file.c_str());
    hub.stop();
    hub_thread.join();
    return rc;
}
