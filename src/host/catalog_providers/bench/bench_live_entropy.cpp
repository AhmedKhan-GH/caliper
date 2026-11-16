// caliper_catalog_providers_bench: what a live panel costs per snapshot and per
// frame, for the acceptance case "plot attention entropy per layer while it
// trains". A [layers, heads, tokens, tokens] attention tensor (softmax rows)
// arrives as a live view through LiveCatalog::apply(), and a Session with the
// providers registered resolves a heatmap of its entropy over the last axis
// and mean over the query tokens, on the CPU reference transforms. Printed,
// per size: the snapshot's copy on the frame thread, a resolve after a new
// snapshot (read, transforms, cache store), a resolve with nothing new (the
// cached result), the generation probe, and the two transforms alone. Not a
// test: run it by hand on a quiet machine, with the load average beside it.
#include "catalog_providers/catalog_providers.h"

#include <caliper/live.h>
#include <caliper_builder/loaders/tensor_ops.h>
#include <caliper_builder/session/session.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace caliper_builder;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Stat {
    double median = 0, p95 = 0;
};
Stat stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stat s;
    if (v.empty()) return s;
    s.median = v[v.size() / 2];
    s.p95 = v[std::min(v.size() - 1, (v.size() * 95) / 100)];
    return s;
}

std::vector<float> attention(int64_t rows, int64_t tokens, int salt) {
    std::vector<float> a(static_cast<std::size_t>(rows * tokens));
    for (int64_t r = 0; r < rows; ++r) {
        float* row = &a[static_cast<std::size_t>(r * tokens)];
        double sum = 0;
        for (int64_t k = 0; k < tokens; ++k) {
            row[k] = std::exp(0.25f * static_cast<float>((r * 31 + k * 7 + salt) % 11));
            sum += row[k];
        }
        for (int64_t k = 0; k < tokens; ++k) row[k] = static_cast<float>(row[k] / sum);
    }
    return a;
}

CommandResult run(Session& s, const std::string& name, nlohmann::json args) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    return s.invoke(std::move(inv));
}

void bench(const std::filesystem::path& root, std::vector<int64_t> shape, int reps) {
    int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
    const int64_t tokens = shape.back();
    std::vector<std::vector<float>> payloads = {attention(rows, tokens, 0), attention(rows, tokens, 5)};
    const double mb = static_cast<double>(payloads[0].size()) * 4.0 / (1 << 20);

    std::string label;
    for (std::size_t i = 0; i < shape.size(); ++i) label += (i ? "x" : "") + std::to_string(shape[i]);
    const std::filesystem::path ws = root / ("ws-" + label);
    std::filesystem::remove_all(ws);
    auto opened = Session::open(ws.string());
    if (!opened.error.ok()) {
        std::printf("open failed: %s\n", opened.error.message.c_str());
        return;
    }
    Session& s = *opened.session;
    caliper_host::LiveCatalog::Options lo;
    lo.max_view_bytes = std::uint64_t(512) << 20;
    caliper_host::MetricsCatalog::Options mo;
    mo.thread = false;
    caliper_host::CatalogProviders p = caliper_host::register_catalog_providers(s, nullptr, nullptr, lo, mo);
    auto frame = [&](std::uint64_t updates) {
        caliper_host::LiveCatalog::ViewFrame f;
        f.name = "attn.weights";
        f.kind = CALIPER_LIVE_KIND_HEATMAP;
        f.updates = updates;
        f.has_payload = true;
        f.dtype = DType::f32;
        f.shape = shape;
        const std::vector<float>& v = payloads[updates % 2];
        f.data = reinterpret_cast<const std::uint8_t*>(v.data());
        f.payload_bytes = v.size() * 4;
        return f;
    };
    p.live->apply(true, "bench", {frame(1)});
    run(s, "spec.new", {{"id", "bench"}});
    run(s, "source.add", {{"id", "attn"}, {"catalog", "live:attn.weights"}});
    // Entropy over the keys, then the mean over the query tokens.
    run(s, "source.add", {{"id", "entropy"}, {"from", "attn"},
                          {"transform", {{{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}},
                                         {{"op", "reduce"}, {"axis", -1}, {"fn", "mean"}}}}});
    CommandResult made = run(s, "panel.create", {{"id", "ent"}, {"primitive", shape.size() == 4 ? "heatmap" : "series"},
                                                 {"bind", shape.size() == 4 ? nlohmann::json{{"values", "entropy"}}
                                                                            : nlohmann::json{{"y", "entropy"}}}});
    if (!made.ok) {
        std::printf("panel.create failed: %s\n", made.error.message.c_str());
        return;
    }
    const spec::Spec spec = s.open_spec()->spec;

    std::vector<double> copy_ms, fresh_ms, cached_ms, probe_us, transforms_ms;
    std::uint64_t updates = 1;
    for (int i = 0; i < reps; ++i) {
        ++updates;
        auto t0 = Clock::now();
        p.live->apply(true, "bench", {frame(updates)});   // the view is wanted: the copy is made here
        copy_ms.push_back(ms_since(t0));
        t0 = Clock::now();
        PanelData d = s.resolve("ent");
        fresh_ms.push_back(ms_since(t0));
        if (!d.error.ok()) {
            std::printf("resolve failed: %s\n", d.error.message.c_str());
            return;
        }
        t0 = Clock::now();
        PanelData again = s.resolve("ent");
        cached_ms.push_back(ms_since(t0));
        t0 = Clock::now();
        const std::uint64_t g = panel_generation(spec, "ent", s.data_sources());
        probe_us.push_back(ms_since(t0) * 1000.0);
        if (g != d.generation || again.generation != d.generation) std::printf("generation mismatch\n");
    }
    // The two transforms alone, on the copy.
    ServedData snap = p.live->read("live:attn.weights", {std::uint64_t(1) << 40, std::uint64_t(1) << 40,
                                                         std::uint64_t(1) << 40});
    TensorTransform entropy, mean;
    transform_from_json({{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}}, &entropy);
    transform_from_json({{"op", "reduce"}, {"axis", -1}, {"fn", "mean"}}, &mean);
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        TensorResult r = apply_transforms(snap.tensor, {entropy, mean});
        transforms_ms.push_back(ms_since(t0));
        if (r.error) std::printf("transform failed: %s\n", r.error.message.c_str());
    }
    const Stat c = stat(copy_ms), f = stat(fresh_ms), k = stat(cached_ms), pr = stat(probe_us), t = stat(transforms_ms);
    const auto cache = s.data_sources().cache_stats();
    std::printf("%-16s %7.1f MB | copy %7.2f / %7.2f ms | resolve new %8.2f / %8.2f ms | transforms %8.2f / %8.2f ms"
                " | resolve cached %6.3f / %6.3f ms | probe %5.2f / %5.2f us | computed %llu reused %llu\n",
                label.c_str(), mb, c.median, c.p95, f.median, f.p95, t.median, t.p95, k.median, k.p95, pr.median,
                pr.p95, static_cast<unsigned long long>(cache.computed), static_cast<unsigned long long>(cache.reused));
    std::fflush(stdout);
    caliper_host::unregister_catalog_providers(s);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <scratch folder> [reps]\n", argv[0]);
        return 2;
    }
    const std::filesystem::path root = argv[1];
    std::filesystem::create_directories(root);
    const int reps = argc > 2 ? std::max(3, std::atoi(argv[2])) : 30;
    std::printf("median / p95 of %d runs; sizes are [layers, heads, tokens, tokens] or a tiled view's [cells, rows, cols]\n",
                reps);
    bench(root, {12, 128, 128}, reps);        // a tiled heatmap: heads reduced on the producer, one cell per layer
    bench(root, {12, 512, 512}, reps);
    bench(root, {12, 12, 64, 64}, reps);
    bench(root, {12, 12, 128, 128}, reps);    // GPT-2 small at 128 tokens
    bench(root, {12, 12, 256, 256}, reps);
    bench(root, {24, 16, 256, 256}, reps);
    bench(root, {12, 12, 512, 512}, reps);
    return 0;
}
