// The derive half of a piece in the producer runtime on the Metal device
// (caliper_producer_derive.h, design/programmable-graphics.md): the arcs of the plan's piece
// over a fixed tensor, several passes in order with kept contents, clears and counters, the
// rules of one-pass programs for every pass (read-only tensors, bounds checks, sizes bounded
// before allocation, the time watchdog, the timeout without a wait and the memory budget), a
// rebind, and a texture. A shared MTLBuffer stands for the training tensors. Without a Metal
// device or the WGSL compiler, the cases return early with a message.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <doctest/doctest.h>

#include "producer/caliper_producer.h"
#include "producer/caliper_producer_derive.h"

#if CALIPER_PRODUCER_HAVE_LIB_MODULES
#include <caliper_builder/programs/pieces.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

constexpr uint32_t kL = 2, kH = 4, kT = 32, kP = 12, kS = 4;
constexpr uint32_t kAttn = kH * kT * kT;   // floats of one layer's weights, batch row 0

// The arcs piece of design/programmable-graphics.md, Arc Example.
const char* kArcCommon = R"(const L: u32 = 2u;
const H: u32 = 4u;
const T: u32 = 32u;
const P: u32 = 12u;
const S: u32 = 4u;

fn ring_point(layer: u32, token: u32) -> vec3<f32> {
    let a = 6.2831853 * f32(token) / f32(T);
    return vec3<f32>(cos(a), f32(layer) * 0.6, sin(a));
}
)";

const char* kArcDerive = R"(fn attn(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    let k = (head * T + i) * T + j;
    if (layer == 0u) {
        return attn0[k];
    }
    return attn1[k];
}

@compute @workgroup_size(8)
fn scores(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H) {
        return;
    }
    var sum = 0.0;
    for (var t = S + P; t < S + 2u * P; t = t + 1u) {
        sum = sum + attn(id.x / H, id.x % H, t, t - P + 1u);
    }
    heads[id.x] = sum / f32(P);
}

@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H * T * T) {
        return;
    }
    let j = id.x % T;
    let i = (id.x / T) % T;
    let head = (id.x / (T * T)) % H;
    let layer = id.x / (T * T * H);
    if (j >= i) {
        return;
    }
    let w = attn(layer, head, i, j);
    if (w < params.threshold) {
        return;
    }
    let slot = atomicAdd(&arcs_count, 1u);
    if (slot >= arrayLength(&arcs)) {
        return;
    }
    arcs[slot] = Arc(j, i, layer, head, w, heads[layer * H + head]);
}

@compute @workgroup_size(32)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= T) {
        return;
    }
    labels[id.x] = Label(vec4<f32>(ring_point(0u, id.x) * 1.15, 1.0), id.x);
}
)";

// The weight from token i to token j of one head, causal, in [0, 0.16].
float weight(uint32_t layer, uint32_t head, uint32_t i, uint32_t j) {
    return j <= i ? float((i * 7 + j * 13 + head * 5 + layer * 3) % 17) / 100.0f : 0.0f;
}

// A derive declaration with its strings and arrays kept alive.
struct Piece {
    std::string id = "arcs";
    std::string wgsl, common;
    std::vector<caliper_producer_input> inputs;
    std::vector<const char*> input_types;
    std::vector<caliper_producer_resource_desc> resources;
    std::vector<std::vector<const char*>> fields;
    std::vector<caliper_producer_derive_pass> passes;
    std::vector<std::vector<const char*>> writes;
    std::vector<caliper_producer_param> params;
    std::vector<const char*> param_types;
    double time_ms = 50.0;
    uint64_t memory = 64ull << 20;

    caliper_producer_resource_desc& buffer(const char* rid, const char* element, uint32_t count) {
        caliper_producer_resource_desc r{};
        r.id = rid;
        r.kind = CALIPER_PRODUCER_RESOURCE_BUFFER;
        r.element = element;
        r.count = count;
        r.publish = 1;
        resources.push_back(r);
        fields.emplace_back();
        return resources.back();
    }
    caliper_producer_resource_desc& record(const char* rid, const char* name, std::vector<const char*> f,
                                           uint32_t count) {
        caliper_producer_resource_desc& r = buffer(rid, nullptr, count);
        r.struct_name = name;
        fields.back() = std::move(f);
        return r;
    }
    void pass(const char* entry, uint32_t x, std::vector<const char*> w = {}) {
        caliper_producer_derive_pass p{};
        p.entry = entry;
        p.workgroups[0] = x;
        p.workgroups[1] = p.workgroups[2] = 1;
        passes.push_back(p);
        writes.push_back(std::move(w));
    }
    caliper_producer_derive_desc desc() {
        for (size_t i = 0; i < resources.size(); ++i) {
            resources[i].fields = fields[i].empty() ? nullptr : fields[i].data();
            resources[i].field_count = fields[i].size();
        }
        for (size_t i = 0; i < passes.size(); ++i) {
            passes[i].writes = writes[i].empty() ? nullptr : writes[i].data();
            passes[i].write_count = writes[i].size();
        }
        caliper_producer_derive_desc d{};
        d.id = id.c_str();
        d.wgsl = wgsl.c_str();
        d.wgsl_size = wgsl.size();
        d.common = common.empty() ? nullptr : common.c_str();
        d.common_size = common.size();
        d.inputs = inputs.data();
        d.input_count = inputs.size();
        d.input_types = input_types.empty() ? nullptr : input_types.data();
        d.resources = resources.data();
        d.resource_count = resources.size();
        d.passes = passes.data();
        d.pass_count = passes.size();
        d.params = params.data();
        d.param_types = param_types.data();
        d.param_count = params.size();
        d.time_limit_ms = time_ms;
        d.memory_limit_bytes = memory;
        return d;
    }
};

// Two layers of attention weights in one shared buffer, imported in place as attn0 and attn1.
struct Fixture {
    id<MTLDevice> device = nil;
    id<MTLBuffer> tensor = nil;
    caliper_producer* rt = nullptr;
    uint32_t attn[2] = {0, 0};

    Fixture() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) return;
        tensor = [device newBufferWithLength:2 * kAttn * 4 options:MTLResourceStorageModeShared];
        auto* w = static_cast<float*>([tensor contents]);
        for (uint32_t l = 0; l < kL; ++l)
            for (uint32_t h = 0; h < kH; ++h)
                for (uint32_t i = 0; i < kT; ++i)
                    for (uint32_t j = 0; j < kT; ++j) w[l * kAttn + (h * kT + i) * kT + j] = weight(l, h, i, j);
        caliper_producer_error e{};
        REQUIRE(caliper_producer_create(nullptr, &rt, &e) == CALIPER_PRODUCER_OK);
        for (uint32_t l = 0; l < kL; ++l) {
            caliper_producer_tensor t{};
            t.name = l ? "blocks.1.attn.weights.act" : "blocks.0.attn.weights.act";
            t.buffer = (__bridge void*)tensor;
            t.byte_offset = uint64_t(l) * kAttn * 4;
            t.byte_size = uint64_t(kAttn) * 4;
            t.dtype = CALIPER_PRODUCER_F32;
            t.ndim = 4;
            t.shape[0] = 1;
            t.shape[1] = kH;
            t.shape[2] = kT;
            t.shape[3] = kT;
            REQUIRE(caliper_producer_import(rt, &t, &attn[l], &e) == CALIPER_PRODUCER_OK);
        }
        caliper_producer_error_free(&e);
    }
    ~Fixture() { caliper_producer_destroy(rt); }

    // A piece that reads attn0 (and attn1 with two inputs).
    Piece piece(const std::string& derive, int inputs = 1) const {
        Piece p;
        p.wgsl = derive;
        static const char* const kNames[2] = {"attn0", "attn1"};
        for (int i = 0; i < inputs; ++i) {
            p.inputs.push_back({kNames[i], attn[i]});
            p.input_types.push_back("f32");
        }
        return p;
    }
    Piece arcs() const {
        Piece p = piece(kArcDerive, 2);
        p.common = kArcCommon;
        p.buffer("heads", "f32", 8);
        p.record("arcs", "Arc", {"from_token: u32", "to_token: u32", "layer: u32", "head: u32", "weight: f32",
                                 "score: f32"}, 4096)
            .counted = 1;
        p.record("labels", "Label", {"position: vec4<f32>", "string: u32"}, 32);
        p.pass("scores", 1, {"heads"});
        p.pass("build", 128, {"arcs"});
        p.pass("place_labels", 1);
        p.params.push_back({"threshold", 0.05});
        p.params.push_back({"glow", 1.0});
        p.param_types = {"f32", "f32"};
        return p;
    }
};

// Compiles, or ends the case with a message when this build has no WGSL compiler.
bool compile_or_skip(caliper_producer* rt, const caliper_producer_derive_desc& d, uint32_t* prog,
                     caliper_producer_error* e) {
    const caliper_producer_status s = caliper_producer_compile_derive(rt, &d, prog, e);
    if (s == CALIPER_PRODUCER_UNSUPPORTED) {
        MESSAGE("no WGSL compiler or composer: ", e->message ? e->message : "");
        return false;
    }
    if (s != CALIPER_PRODUCER_OK) FAIL("compile_derive: ", e->code, ": ", e->message ? e->message : "");
    return true;
}

// The code of a refused compile, or "ok".
std::string refused(caliper_producer* rt, Piece& p, caliper_producer_status want = CALIPER_PRODUCER_REFUSED) {
    caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 9;
    caliper_producer_error e{};
    const caliper_producer_status s = caliper_producer_compile_derive(rt, &d, &prog, &e);
    std::string code = s == CALIPER_PRODUCER_OK ? "ok" : (e.code ? e.code : "?");
    if (s != CALIPER_PRODUCER_OK) {
        CHECK(s == want);
        CHECK(prog == 0);
    }
    caliper_producer_error_free(&e);
    return code;
}

template <typename T>
std::vector<T> read(caliper_producer* rt, uint32_t prog, const char* name, caliper_producer_output* out) {
    caliper_producer_error e{};
    REQUIRE(caliper_producer_read_resource(rt, prog, name, out, &e) == CALIPER_PRODUCER_OK);
    const auto* base = static_cast<const uint8_t*>([(__bridge id<MTLBuffer>)out->buffer contents]);
    std::vector<T> v(out->byte_size / sizeof(T));
    if (!v.empty()) std::memcpy(v.data(), base + out->byte_offset, out->byte_size);
    return v;
}

}  // namespace

TEST_CASE("the arcs piece publishes the arcs of a fixed tensor in three passes") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Piece p = f.arcs();
    const caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_output out{};
    CHECK(read<float>(f.rt, prog, "heads", &out).empty());   // nothing before the first run
    caliper_producer_frame frame{100, 0.0, 3};
    caliper_producer_pass pass{};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    CHECK(pass.ran == 1);

    // The reference on the CPU: the induction score of each head, then each arc above 0.05.
    std::vector<float> heads(kL * kH);
    for (uint32_t x = 0; x < kL * kH; ++x) {
        float sum = 0;
        for (uint32_t t = kS + kP; t < kS + 2 * kP; ++t) sum += weight(x / kH, x % kH, t, t - kP + 1);
        heads[x] = sum / float(kP);
    }
    using Arc = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, float>;   // layer, head, to, from, weight
    std::vector<Arc> want;
    for (uint32_t l = 0; l < kL; ++l)
        for (uint32_t h = 0; h < kH; ++h)
            for (uint32_t i = 0; i < kT; ++i)
                for (uint32_t j = 0; j < i; ++j)
                    if (!(weight(l, h, i, j) < 0.05f)) want.emplace_back(l, h, i, j, weight(l, h, i, j));

    const std::vector<float> got_heads = read<float>(f.rt, prog, "heads", &out);
    CHECK(out.count == 8);
    CHECK(out.words == 1);
    REQUIRE(got_heads.size() == heads.size());
    for (size_t i = 0; i < heads.size(); ++i) CHECK(got_heads[i] == doctest::Approx(heads[i]).epsilon(1e-5));

    struct ArcBytes { uint32_t from, to, layer, head; float weight, score; };
    static_assert(sizeof(ArcBytes) == 24, "an Arc is 24 bytes");
    const std::vector<ArcBytes> arcs = read<ArcBytes>(f.rt, prog, "arcs", &out);
    CHECK(out.words == 6);
    CHECK(out.count == want.size());
    CHECK(out.byte_size == uint64_t(want.size()) * 24);
    MESSAGE(want.size(), " arcs of 24 bytes");
    std::vector<Arc> got;
    for (const ArcBytes& a : arcs) {
        got.emplace_back(a.layer, a.head, a.to, a.from, a.weight);
        CHECK(a.score == doctest::Approx(heads[a.layer * kH + a.head]).epsilon(1e-5));
    }
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    CHECK(got == want);

    struct LabelBytes { float position[4]; uint32_t string, pad[3]; };
    static_assert(sizeof(LabelBytes) == 32, "a Label is 32 bytes");
    const std::vector<LabelBytes> labels = read<LabelBytes>(f.rt, prog, "labels", &out);
    CHECK(out.words == 8);
    REQUIRE(labels.size() == kT);
    for (uint32_t i = 0; i < kT; ++i) {
        CHECK(labels[i].string == i);
        CHECK(labels[i].position[3] == 1.0f);
    }

    // A second run counts again from zero: the same arcs, not twice as many.
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    CHECK(read<ArcBytes>(f.rt, prog, "arcs", &out).size() == want.size());
    // A param moves the threshold without a compile.
    const caliper_producer_param none{"threshold", 1.0};
    REQUIRE(caliper_producer_set_params(f.rt, prog, &none, 1, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    CHECK(read<ArcBytes>(f.rt, prog, "arcs", &out).empty());

    caliper_producer_program_status st{};
    REQUIRE(caliper_producer_read_status(f.rt, prog, &st) == CALIPER_PRODUCER_OK);
    CHECK(st.runs == 3);
    CHECK(st.last_step == 100);
    CHECK(st.memory_bytes == 8 * 4 + 4096 * 24 + 4 + 32 * 32 + 2ull * kAttn * 4);
    caliper_producer_error_free(&e);
}

TEST_CASE("passes run in order, resources keep their contents, clears zero them and counters restart") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Piece p = f.piece(R"(@compute @workgroup_size(64)
fn grow(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 64u) { a[id.x] = a[id.x] + 1.0; secret[id.x] = 1u; }
}
@compute @workgroup_size(64)
fn twice(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 64u) { b[id.x] = b[id.x] + 2.0 * a[id.x]; }
}
@compute @workgroup_size(64)
fn keep(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 64u && a[id.x] > 1.5) {
        let i = atomicAdd(&c_count, 1u);
        c[i] = caliper.step + id.x;
    }
}
)", 0);
    p.id = "state";
    p.buffer("a", "f32", 64);
    p.buffer("b", "f32", 64).clear = 1;
    p.buffer("c", "u32", 64).counted = 1;
    p.buffer("secret", "u32", 64).publish = 0;
    p.pass("grow", 1, {"a"});
    p.pass("twice", 1, {"b"});
    p.pass("keep", 1, {"c"});
    const caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_output out{};
    for (int run = 1; run <= 3; ++run) {
        caliper_producer_frame frame{run * 10, 0.0, 3};
        REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
        const std::vector<float> a = read<float>(f.rt, prog, "a", &out);
        const std::vector<float> b = read<float>(f.rt, prog, "b", &out);
        const std::vector<uint32_t> c = read<uint32_t>(f.rt, prog, "c", &out);
        REQUIRE(a.size() == 64);
        CHECK(a[5] == float(run));            // kept between runs
        CHECK(b[5] == 2.0f * float(run));     // zeroed before each run, then read a of the same run
        CHECK(c.size() == (run == 1 ? 0u : 64u));   // the counter restarts at each run
        if (!c.empty()) CHECK(*std::min_element(c.begin(), c.end()) == uint32_t(run * 10));
    }
    CHECK(caliper_producer_read_resource(f.rt, prog, "secret", &out, &e) == CALIPER_PRODUCER_REFUSED);
    CHECK(std::string(e.code) == "program.unknown_resource");
    caliper_producer_error_free(&e);
}

TEST_CASE("the compile refuses a store into a tensor and a binding of a derive pass's own") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Piece p = f.piece("@compute @workgroup_size(1)\nfn run() {\n    attn0[0] = 1.0;\n    a[0] = attn0[1];\n}\n");
    p.buffer("a", "f32", 4);
    p.pass("run", 1, {"a"});
    caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    const caliper_producer_status s = caliper_producer_compile_derive(f.rt, &d, &prog, &e);
    if (s == CALIPER_PRODUCER_UNSUPPORTED) { MESSAGE("no WGSL compiler"); caliper_producer_error_free(&e); return; }
    CHECK(s == CALIPER_PRODUCER_REFUSED);
    CHECK(std::string(e.code) == "program.compile_failed");
    CHECK(e.line == 3);   // the line of the store in derive.wgsl, not in the composed module
    CHECK(std::string(e.message).rfind("derive.wgsl, line 3", 0) == 0);
    caliper_producer_error_free(&e);

    Piece own = f.piece(R"(@group(3) @binding(0) var<storage, read_write> sneak: array<f32>;
@compute @workgroup_size(1)
fn run() { sneak[0] = attn0[0]; a[0] = 1.0; }
)");
    own.buffer("a", "f32", 4);
    own.pass("run", 1, {"a"});
    CHECK(refused(f.rt, own) == "program.binding_not_allowed");
    // Neither piece compiled, thus neither ran; the next case runs passes against the tensor.
}

TEST_CASE("an access out of range stays inside its binding in every pass") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    auto* w = static_cast<const float*>([f.tensor contents]);
    const std::vector<float> before(w, w + 2 * kAttn);
    Piece p = f.piece(R"(@compute @workgroup_size(64)
fn spill(@builtin(global_invocation_id) id: vec3<u32>) {
    a[id.x] = 7.0;
    a[id.x + 4000000000u] = 9.0;
}
@compute @workgroup_size(4)
fn far(@builtin(global_invocation_id) id: vec3<u32>) {
    b[id.x] = 1.0 + attn0[id.x + 100000000u] + attn0[4096u + id.x];
}
)");
    p.buffer("a", "f32", 4);
    p.buffer("b", "f32", 4);
    p.buffer("c", "f32", 4);
    p.pass("spill", 16, {"a"});
    p.pass("far", 1, {"b"});
    const caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 3};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_output out{};
    CHECK(read<float>(f.rt, prog, "a", &out) == std::vector<float>(4, 7.0f));
    // attn1 starts at element 4096 of the same storage: a load past attn0 reads zero, not attn1.
    CHECK(read<float>(f.rt, prog, "b", &out) == std::vector<float>(4, 1.0f));
    CHECK(read<float>(f.rt, prog, "c", &out) == std::vector<float>(4, 0.0f));
    CHECK(std::equal(before.begin(), before.end(), w));
}

TEST_CASE("sizes, layouts, passes and the memory budget are refused before allocation") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    const char* kOne = "@compute @workgroup_size(1)\nfn run() { a[0] = attn0[0]; }\n";
    auto base = [&] {
        Piece p = f.piece(kOne);
        p.pass("run", 1, {"a"});
        return p;
    };
    Piece p = base();
    p.buffer("a", "f32", 4194305);
    CHECK(refused(f.rt, p) == "program.bad_resource");
    p = base();
    p.record("a", "Big", {"x: vec4<f32>", "y: vec4<f32>"}, 4194304);   // 4,194,304 x 32 bytes: 128 MiB
    CHECK(refused(f.rt, p) == "program.bad_resource");
    p = base();
    p.buffer("a", "f32", 4);
    caliper_producer_resource_desc& t = p.buffer("t", nullptr, 0);
    t.kind = CALIPER_PRODUCER_RESOURCE_TEXTURE;
    t.width = 4097;
    t.height = 1;
    CHECK(refused(f.rt, p) == "program.bad_resource");
    p = base();
    p.record("a", "Wide", std::vector<const char*>(17, "x: f32"), 4);
    CHECK(refused(f.rt, p) == "program.bad_layout");
    p = base();
    p.record("a", "V3", {"x: vec3<f32>"}, 4);
    CHECK(refused(f.rt, p) == "program.bad_layout");
    p = base();   // a field that would write a binding into the header
    p.record("a", "Y", {"y: f32 } @group(2) @binding(9) var<storage, read_write> z: array<f32>; struct Z { z: f32"},
             4);
    CHECK(refused(f.rt, p) == "program.bad_layout");
    p = base();
    p.buffer("a__b", "f32", 4);
    CHECK(refused(f.rt, p) != "ok");
    p = base();
    p.buffer("a", "f32", 4);
    p.buffer("a", "u32", 4);
    CHECK(refused(f.rt, p) == "program.bad_resource");
    p = base();
    p.buffer("a", "f32", 4).counted = 1;
    p.buffer("a_count", "u32", 4);
    CHECK(refused(f.rt, p) == "program.bad_resource");
    p = base();
    p.buffer("a", "f32", 4);
    p.writes[0] = {"b"};
    CHECK(refused(f.rt, p) == "program.unknown_resource");
    p = base();
    p.buffer("a", "f32", 4);
    for (int i = 0; i < 8; ++i) p.pass("run", 1);
    CHECK(refused(f.rt, p) == "program.bad_pass");
    p = base();
    p.buffer("a", "f32", 4);
    p.passes[0].entry = "main";
    CHECK(refused(f.rt, p) == "program.bad_pass");
    p = base();
    p.buffer("a", "f32", 4);
    p.passes[0].entry = "absent";
    CHECK(refused(f.rt, p) == "program.entry_stage");
    p = base();
    p.buffer("a", "f32", 4);
    p.input_types[0] = "i32";
    CHECK(refused(f.rt, p) == "piece.bad_binding");
    p = base();
    p.buffer("a", "f32", 1 << 19);   // 2 MiB
    p.memory = 1 << 20;
    CHECK(refused(f.rt, p, CALIPER_PRODUCER_OVER_BUDGET) == "program.over_memory");
    p = base();
    p.buffer("a", "f32", 4);
    p.time_ms = 51;
    CHECK(refused(f.rt, p) == "program.over_limits");
}

TEST_CASE("the watchdog stops a derive program over its time, and a timeout goes without a wait") {
    using clock = std::chrono::steady_clock;
    const auto settle = clock::now() + std::chrono::seconds(20);   // the reaper of earlier cases
    while (caliper_producer_reaping() != 0 && clock::now() < settle)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    REQUIRE(caliper_producer_reaping() == 0);
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    // The middle pass loops n times in one invocation; n comes from a param.
    Piece p = f.piece(R"(@compute @workgroup_size(1)
fn first() { a[0] = 1.0; }
@compute @workgroup_size(1)
fn slow() {
    var s = a[0];
    for (var k = 0u; k < params.n; k++) { s = s * 0.999999 + attn0[k % 8u]; }
    a[1] = s;
}
@compute @workgroup_size(1)
fn last() { a[2] = a[1]; }
)");
    p.buffer("a", "f32", 4);
    p.pass("first", 1, {"a"});
    p.pass("slow", 1, {"a"});
    p.pass("last", 1, {"a"});
    p.params.push_back({"n", 100000});
    p.param_types = {"u32"};
    p.time_ms = 0.01;
    const caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 3};
    caliper_producer_pass pass{};
    for (int i = 0; i < 3; ++i) REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    CHECK(pass.stopped == 1);
    caliper_producer_program_status st{};
    REQUIRE(caliper_producer_read_status(f.rt, prog, &st) == CALIPER_PRODUCER_OK);
    CHECK(st.state == CALIPER_PRODUCER_STOPPED);
    REQUIRE(st.stop_code);
    CHECK(std::string(st.stop_code) == "program.over_time");
    REQUIRE(st.last_pass_ms > 0.0);

    // The same piece with a pass of about 1.5 s by the measured pass, 30 times the timeout of
    // 50 ms, so that a slow runner that measured the pass wrong still runs it past the timeout.
    p.id = "slow";
    p.time_ms = 50.0;
    p.params[0].value = std::min(4e9, 100000.0 * 1500.0 / st.last_pass_ms);
    const caliper_producer_derive_desc d2 = p.desc();
    uint32_t again = 0;
    REQUIRE(caliper_producer_compile_derive(f.rt, &d2, &again, &e) == CALIPER_PRODUCER_OK);
    const auto t0 = clock::now();
    CHECK(caliper_producer_run(f.rt, &again, 1, &frame, 50.0, nullptr, &e) == CALIPER_PRODUCER_TIMEOUT);
    REQUIRE(e.code);
    CHECK(std::string(e.code) == "program.timeout");
    caliper_producer_error_free(&e);
    const auto t1 = clock::now();
    caliper_producer_destroy(f.rt);   // no wait for the pass that still runs
    f.rt = nullptr;
    const auto t2 = clock::now();
    // Neither the timeout nor the destroy waited: the pass still runs on the GPU after both
    // returned, and its objects wait in the reaper until it ends. The wall times are reported, not
    // held to a bound, because they depend on the runner.
    const bool running_after = caliper_producer_reaping() != 0;
    const auto deadline = clock::now() + std::chrono::seconds(20);
    while (caliper_producer_reaping() != 0 && clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const auto t3 = clock::now();
    const auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    MESSAGE("the timeout came after ", ms(t1 - t0), " ms; the destroy took ", ms(t2 - t1),
            " ms; the pass ended ", ms(t3 - t0), " ms after the run began");
    CHECK(running_after);
    CHECK(caliper_producer_reaping() == 0);
}

TEST_CASE("a rebind gives a derive program new storage and its shape, and a texture is packed texels") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Piece p = f.piece(R"(@compute @workgroup_size(8)
fn look(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x == 0u) { s[0] = shapes.attn0; v[0] = attn0[162]; }
    if (id.x < 8u) { img[id.x] = pack4x8unorm(vec4<f32>(f32(id.x) / 7.0, 0.0, 1.0, 1.0)); }
}
)");
    p.buffer("s", "vec4<u32>", 1);
    p.buffer("v", "f32", 1);
    caliper_producer_resource_desc& img = p.buffer("img", nullptr, 0);
    img.kind = CALIPER_PRODUCER_RESOURCE_TEXTURE;
    img.width = 4;
    img.height = 2;
    p.pass("look", 1, {"s", "v", "img"});
    const caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 3};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_output out{};
    CHECK(read<uint32_t>(f.rt, prog, "s", &out) == std::vector<uint32_t>{1, kH, kT, kT});
    CHECK(read<float>(f.rt, prog, "v", &out)[0] == weight(0, 0, 5, 2));
    const std::vector<uint32_t> texels = read<uint32_t>(f.rt, prog, "img", &out);
    CHECK(out.count == 8);
    CHECK(out.words == 1);
    REQUIRE(texels.size() == 8);
    CHECK(texels[0] == 0xff'ff'00'00u);   // r 0, g 0, b 255, a 255, packed little-endian
    CHECK(texels[7] == 0xff'ff'00'ffu);

    // The second layer's storage as a tensor of another shape and the same bytes.
    caliper_producer_tensor t{};
    t.name = "moved";
    t.buffer = (__bridge void*)f.tensor;
    t.byte_offset = uint64_t(kAttn) * 4;
    t.byte_size = uint64_t(kAttn) * 4;
    t.dtype = CALIPER_PRODUCER_F32;
    t.ndim = 2;
    t.shape[0] = kT * kT;
    t.shape[1] = kH;
    uint32_t moved = 0;
    REQUIRE(caliper_producer_import(f.rt, &t, &moved, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(caliper_producer_rebind(f.rt, prog, f.attn[0], moved, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    CHECK(read<uint32_t>(f.rt, prog, "s", &out) == std::vector<uint32_t>{1, 1, kT * kT, kH});
    CHECK(read<float>(f.rt, prog, "v", &out)[0] == weight(1, 0, 5, 2));
    caliper_producer_error_free(&e);
}

TEST_CASE("a resource and an input that no pass uses still compile, and a private resource stays in the script") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Piece p = f.piece("@compute @workgroup_size(1)\nfn run() { a[0] = 2.0; }\n", 2);
    p.buffer("a", "f32", 4);
    p.buffer("unused", "vec4<f32>", 8).counted = 1;
    p.pass("run", 1, {"a"});
    const caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 3};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_output out{};
    CHECK(read<float>(f.rt, prog, "a", &out)[0] == 2.0f);
    CHECK(read<float>(f.rt, prog, "unused", &out).empty());   // counted, and nothing counted it
    caliper_producer_error_free(&e);
}

#if CALIPER_PRODUCER_HAVE_LIB_MODULES
TEST_CASE("a derive half compiles when common carries lib modules that read the draw prelude") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    // The section common as the app sends it: the lib modules in the order of imports, then common.wgsl.
    std::string lib;
    for (const char* m : {"colormap@1", "easing@1", "pick@1", "camera@1"}) {
        std::string body(caliper_builder::programs::lib_module(m));
        if (!body.empty() && body.back() != '\n') body += '\n';
        lib += body;
    }
    Piece p = f.piece("@compute @workgroup_size(1)\nfn run() { a[0] = easing__smooth(0.5) + K; }\n", 2);
    p.common = lib + "const K: f32 = 1.0;\n";
    p.buffer("a", "f32", 4);
    p.pass("run", 1, {"a"});
    const caliper_producer_derive_desc d = p.desc();
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 3};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_output out{};
    CHECK(read<float>(f.rt, prog, "a", &out)[0] == doctest::Approx(1.5f));
    caliper_producer_error_free(&e);
}
#endif
