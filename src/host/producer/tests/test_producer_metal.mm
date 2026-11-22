// The producer runtime on the Metal device (caliper_producer.h, design/gpu-programs.md): a
// shared MTLBuffer stands for a training tensor. The cases import it in place and read-only,
// compile programs through the sandbox and caliper-wgsl, run passes, read the outputs, and
// check each refusal with its code. Without a Metal device or the WGSL compiler, the cases that
// need them return early with a message.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <doctest/doctest.h>

#include "producer/caliper_producer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kRows = 4, kCols = 8;

// One point for each row of w: x the row, y the row's norm, z the step of the frame uniform.
const char* kRowNorms = R"(struct CaliperFrame { step: u32, pass_index: u32, max_count: u32, width: u32,
                         height: u32, time_s: f32, every: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> count: atomic<u32>;
@group(0) @binding(3) var<uniform> caliper: CaliperFrame;
@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= 4u) { return; }
    var s = 0.0;
    for (var c = 0u; c < 8u; c++) { let v = w[row * 8u + c]; s += v * v; }
    let i = atomicAdd(&count, 1u);
    positions[i * 3u] = f32(row);
    positions[i * 3u + 1u] = sqrt(s);
    positions[i * 3u + 2u] = f32(caliper.step);
}
)";

struct Fixture {
    id<MTLDevice> device = nil;
    id<MTLBuffer> tensor = nil;
    caliper_producer* rt = nullptr;
    uint32_t w = 0;

    Fixture() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) return;
        tensor = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        fill(1.0f);
        caliper_producer_error e{};
        REQUIRE(caliper_producer_create(nullptr, &rt, &e) == CALIPER_PRODUCER_OK);
        caliper_producer_tensor t = desc_of(0, kRows * kCols * 4);
        REQUIRE(caliper_producer_import(rt, &t, &w, &e) == CALIPER_PRODUCER_OK);
        CHECK(w != 0);
        caliper_producer_error_free(&e);
    }
    ~Fixture() { caliper_producer_destroy(rt); }

    // Row r holds r + scale in each column, thus its norm is (r + scale) * sqrt(8).
    void fill(float scale) {
        auto* f = static_cast<float*>([tensor contents]);
        for (uint32_t r = 0; r < kRows; ++r)
            for (uint32_t c = 0; c < kCols; ++c) f[r * kCols + c] = float(r) + scale;
    }
    caliper_producer_tensor desc_of(uint64_t offset, uint64_t bytes) const {
        caliper_producer_tensor t{};
        t.name = "layers.0.weight";
        t.buffer = (__bridge void*)tensor;
        t.byte_offset = offset;
        t.byte_size = bytes;
        t.dtype = CALIPER_PRODUCER_F32;
        t.ndim = 2;
        t.shape[0] = kRows;
        t.shape[1] = kCols;
        return t;
    }
};

struct Desc {
    caliper_producer_input input{"w", 0};
    const char* arrays[1] = {"positions"};
    caliper_producer_program_desc d{};
    std::string src;

    Desc(const char* wgsl, uint32_t tensor) {
        src = wgsl;
        input.tensor = tensor;
        d.id = "norms";
        d.wgsl = src.c_str();
        d.wgsl_size = src.size();
        d.inputs = &input;
        d.input_count = 1;
        d.output_kind = CALIPER_PRODUCER_GEOMETRY;
        d.part = CALIPER_PRODUCER_POINTS;
        d.max_count = kRows;
        d.arrays = arrays;
        d.array_count = 1;
        d.time_limit_ms = 50.0;
        d.memory_limit_bytes = 64u << 20;
    }
    void set(const char* wgsl) {
        src = wgsl;
        d.wgsl = src.c_str();
        d.wgsl_size = src.size();
    }
};

std::string replaced(std::string s, const std::string& from, const std::string& to) {
    const size_t at = s.find(from);
    REQUIRE(at != std::string::npos);
    s.replace(at, from.size(), to);
    return s;
}

// Compiles; false when this build has no WGSL compiler (the case then ends).
bool compile_or_skip(caliper_producer* rt, const caliper_producer_program_desc& d, uint32_t* h,
                     caliper_producer_error* e) {
    const caliper_producer_status s = caliper_producer_compile(rt, &d, h, e);
    if (s == CALIPER_PRODUCER_UNSUPPORTED && e->code && std::string(e->code) == "program.compiler_unavailable") {
        MESSAGE("no WGSL compiler in this build: ", e->message);
        return false;
    }
    if (s != CALIPER_PRODUCER_OK) {
        const std::string why = std::string(e->code ? e->code : "") + ": " + (e->message ? e->message : "");
        FAIL_CHECK(why);
    }
    return s == CALIPER_PRODUCER_OK;
}

caliper_producer_status compile_code(caliper_producer* rt, const caliper_producer_program_desc& d, std::string* code,
                                     uint32_t* line = nullptr) {
    uint32_t h = 0;
    caliper_producer_error e{};
    const caliper_producer_status s = caliper_producer_compile(rt, &d, &h, &e);
    *code = e.code ? e.code : "";
    if (line) *line = e.line;
    if (s == CALIPER_PRODUCER_OK) CHECK(h != 0);
    else CHECK(h == 0);
    caliper_producer_error_free(&e);
    return s;
}

}  // namespace

TEST_CASE("the runtime imports a tensor in place, runs a geometry pass and reads its output") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Desc d(kRowNorms, f.w);
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &prog, &e)) { caliper_producer_error_free(&e); return; }

    caliper_producer_frame frame{7, 0.5, 3};
    caliper_producer_pass pass{};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    CHECK(pass.ran == 1);
    CHECK(pass.stopped == 0);
    CHECK(pass.wall_ms >= 0);

    caliper_producer_output out{};
    REQUIRE(caliper_producer_read_output(f.rt, prog, "positions", &out, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(out.buffer != nullptr);
    CHECK(out.count == kRows);
    CHECK(out.words == 3);
    CHECK(out.byte_size == kRows * 12);
    auto check_norms = [&](float scale, float step) {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)out.buffer;
        const float* p = static_cast<const float*>([b contents]);
        std::vector<bool> seen(kRows, false);
        for (uint32_t i = 0; i < out.count; ++i) {
            const auto row = uint32_t(p[i * 3]);
            REQUIRE(row < kRows);
            seen[row] = true;
            CHECK(p[i * 3 + 1] == doctest::Approx((float(row) + scale) * std::sqrt(8.0f)).epsilon(1e-5));
            CHECK(p[i * 3 + 2] == step);
        }
        for (bool s : seen) CHECK(s);
    };
    check_norms(1.0f, 7.0f);

    // The script changes the tensor in place: the same import reads the new values.
    f.fill(2.0f);
    frame.step = 8;
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(caliper_producer_read_output(f.rt, prog, "positions", &out, &e) == CALIPER_PRODUCER_OK);
    check_norms(2.0f, 8.0f);

    caliper_producer_program_status st{};
    REQUIRE(caliper_producer_read_status(f.rt, prog, &st) == CALIPER_PRODUCER_OK);
    CHECK(st.state == CALIPER_PRODUCER_RUNNING);
    CHECK(st.runs == 2);
    CHECK(st.last_step == 8);
    CHECK(st.memory_bytes == kRows * 12 + 4 + kRows * kCols * 4);
    CHECK(st.stop_code == nullptr);

    CHECK(caliper_producer_read_output(f.rt, prog, "colors", &out, &e) == CALIPER_PRODUCER_ARGUMENT);
    caliper_producer_remove(f.rt, prog);
    CHECK(caliper_producer_read_status(f.rt, prog, &st) == CALIPER_PRODUCER_ARGUMENT);
    caliper_producer_error_free(&e);
}

TEST_CASE("an input bound again to new storage of the same size runs without a compile") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    // A prepare compiles the source ahead; the compile then takes its library.
    caliper_producer_error e{};
    const caliper_producer_status prepared = caliper_producer_prepare(nullptr, kRowNorms, std::strlen(kRowNorms), nullptr, &e);
    caliper_producer_error_free(&e);
    Desc d(kRowNorms, f.w);
    uint32_t prog = 0;
    if (!compile_or_skip(f.rt, d.d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    CHECK(prepared == CALIPER_PRODUCER_OK);
    caliper_producer_frame frame{1, 0.0, 1};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);

    // The gradient's new storage: rows of r + 5.
    id<MTLBuffer> moved = [f.device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    auto* m = static_cast<float*>([moved contents]);
    for (uint32_t r = 0; r < kRows; ++r)
        for (uint32_t c = 0; c < kCols; ++c) m[r * kCols + c] = float(r) + 5.0f;
    caliper_producer_tensor t = f.desc_of(0, kRows * kCols * 4);
    t.buffer = (__bridge void*)moved;
    uint32_t h = 0;
    REQUIRE(caliper_producer_import(f.rt, &t, &h, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(caliper_producer_rebind(f.rt, prog, f.w, h, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_release(f.rt, f.w);   // the old storage goes; the program no longer binds it
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_program_status st{};
    REQUIRE(caliper_producer_read_status(f.rt, prog, &st) == CALIPER_PRODUCER_OK);
    CHECK(st.state == CALIPER_PRODUCER_RUNNING);
    CHECK(st.runs == 2);   // the same program: its count goes on
    caliper_producer_output out{};
    REQUIRE(caliper_producer_read_output(f.rt, prog, "positions", &out, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(out.count == kRows);
    const float* p = static_cast<const float*>([(__bridge id<MTLBuffer>)out.buffer contents]);
    for (uint32_t i = 0; i < kRows; ++i)
        CHECK(p[i * 3 + 1] == doctest::Approx((p[i * 3] + 5.0f) * std::sqrt(8.0f)).epsilon(1e-5));

    // Another size needs a compile: refused, and the program keeps its binding.
    t.byte_size = 64;
    uint32_t small = 0;
    REQUIRE(caliper_producer_import(f.rt, &t, &small, &e) == CALIPER_PRODUCER_OK);
    CHECK(caliper_producer_rebind(f.rt, prog, h, small, &e) == CALIPER_PRODUCER_REFUSED);
    REQUIRE(e.code);
    CHECK(std::string(e.code) == "program.import_refused");
    caliper_producer_error_free(&e);
    CHECK(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_error_free(&e);
}

TEST_CASE("a pixels program writes its packed RGBA8 image") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Desc d(R"(@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
@compute @workgroup_size(8, 8)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 4u || id.y >= 2u) { return; }
    pixels[id.y * 4u + id.x] = pack4x8unorm(vec4<f32>(f32(id.x) / 3.0, f32(id.y), 0.0, 1.0));
}
)", f.w);
    d.d.input_count = 0;
    d.d.output_kind = CALIPER_PRODUCER_PIXELS;
    d.d.width = 4;
    d.d.height = 2;
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 1};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_output out{};
    REQUIRE(caliper_producer_read_output(f.rt, prog, "pixels", &out, &e) == CALIPER_PRODUCER_OK);
    CHECK(out.count == 8);
    CHECK(out.byte_size == 32);
    const uint32_t* px = static_cast<const uint32_t*>([(__bridge id<MTLBuffer>)out.buffer contents]);
    CHECK(px[0] == 0xFF000000u);       // x 0, y 0: black, opaque
    CHECK(px[7] == 0xFF00FFFFu);       // x 3, y 1: red and green, opaque (R in the low byte)
    caliper_producer_error_free(&e);
}

// A store far outside the output values, at an index the script gives through the input aim:
// the index that, without bounds checks, lands in the tensor guard, which the pass also binds.
const char* kFarStore = R"(@group(0) @binding(0) var<storage, read> aim: array<i32>;
@group(0) @binding(1) var<storage, read> guard: array<f32>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> count: atomic<u32>;
@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 4u) { return; }
    let i = atomicAdd(&count, 1u);
    positions[i * 3u] = f32(id.x);
    positions[i * 3u + 1u] = guard[0];
    positions[i * 3u + 2u] = 0.0;
    values[i] = f32(id.x) + 10.0;
    if (id.x == 0u) {
        values[aim[0]] = 12345.0;
        values[u32(aim[0]) + 4000000000u] = 12345.0;
    }
}
)";

TEST_CASE("a store far outside an output leaves a tensor after it and the rest of the output unchanged") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    constexpr float kSentinel = 7.0f;
    id<MTLBuffer> aim = [f.device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLBuffer> guard = [f.device newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    auto* g = static_cast<float*>([guard contents]);
    for (size_t i = 0; i < 65536 / 4; ++i) g[i] = kSentinel;
    std::memset([aim contents], 0, 4096);
    auto import = [&](id<MTLBuffer> b, const char* name, uint64_t bytes, uint32_t dtype) {
        caliper_producer_tensor t{};
        t.name = name;
        t.buffer = (__bridge void*)b;
        t.byte_size = bytes;
        t.dtype = dtype;
        t.ndim = 1;
        t.shape[0] = int64_t(bytes / 4);
        uint32_t h = 0;
        caliper_producer_error e{};
        REQUIRE(caliper_producer_import(f.rt, &t, &h, &e) == CALIPER_PRODUCER_OK);
        return h;
    };
    const uint32_t aim_h = import(aim, "aim", 16, CALIPER_PRODUCER_I32);
    const uint32_t guard_h = import(guard, "guard", 65536, CALIPER_PRODUCER_F32);

    caliper_producer_input inputs[2] = {{"aim", aim_h}, {"guard", guard_h}};
    const char* arrays[2] = {"positions", "values"};
    Desc d(kFarStore, 0);
    d.d.inputs = inputs;
    d.d.input_count = 2;
    d.d.arrays = arrays;
    d.d.array_count = 2;
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &prog, &e)) { caliper_producer_error_free(&e); return; }

    // The index from values' first element to the middle of guard, in elements of 4 bytes.
    caliper_producer_output out{};
    REQUIRE(caliper_producer_read_output(f.rt, prog, "values", &out, &e) == CALIPER_PRODUCER_OK);
    id<MTLBuffer> values = (__bridge id<MTLBuffer>)out.buffer;
    const int64_t delta = int64_t(guard.gpuAddress + 32768) - int64_t(values.gpuAddress);
    REQUIRE(delta % 4 == 0);
    if (delta / 4 > INT32_MAX || delta / 4 < INT32_MIN) {
        MESSAGE("guard is more than 8 GiB from the output: the case cannot aim at it");
        return;
    }
    static_cast<int32_t*>([aim contents])[0] = int32_t(delta / 4);
    MESSAGE("the store aims ", delta, " bytes from the output's first element");

    caliper_producer_frame frame{1, 0.0, 1};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    size_t changed = 0;
    for (size_t i = 0; i < 65536 / 4; ++i) changed += g[i] != kSentinel;
    CHECK(changed == 0);
    REQUIRE(caliper_producer_read_output(f.rt, prog, "values", &out, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(out.count == 4);
    const float* v = static_cast<const float*>([values contents]);
    std::vector<bool> seen(4, false);
    for (uint32_t i = 0; i < 4; ++i) {
        const float x = v[i] - 10.0f;
        REQUIRE((x >= 0.0f && x < 4.0f));
        seen[size_t(x)] = true;
    }
    for (bool s : seen) CHECK(s);
    REQUIRE(caliper_producer_read_output(f.rt, prog, "positions", &out, &e) == CALIPER_PRODUCER_OK);
    const float* pos = static_cast<const float*>([(__bridge id<MTLBuffer>)out.buffer contents]);
    for (uint32_t i = 0; i < 4; ++i) CHECK(pos[i * 3 + 1] == kSentinel);
    caliper_producer_error_free(&e);
}

TEST_CASE("a load far past an input reads zero, not the next tensor in the same storage") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    // The import w covers bytes 0 to 127 of the storage; another tensor fills the rest of it.
    auto* t = static_cast<float*>([f.tensor contents]);
    for (size_t i = kRows * kCols; i < 4096 / 4; ++i) t[i] = 9.0f;
    Desc d(R"(@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> count: atomic<u32>;
@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 4u) { return; }
    let i = atomicAdd(&count, 1u);
    positions[i * 3u] = f32(id.x);
    positions[i * 3u + 1u] = w[id.x * 8u];
    positions[i * 3u + 2u] = w[200u + id.x];
}
)", f.w);
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 1};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_output out{};
    REQUIRE(caliper_producer_read_output(f.rt, prog, "positions", &out, &e) == CALIPER_PRODUCER_OK);
    REQUIRE(out.count == 4);
    const float* p = static_cast<const float*>([(__bridge id<MTLBuffer>)out.buffer contents]);
    for (uint32_t i = 0; i < 4; ++i) {
        const auto row = uint32_t(p[i * 3]);
        CHECK(p[i * 3 + 1] == float(row) + 1.0f);   // in range: the row's first column, row + 1
        CHECK(p[i * 3 + 2] == 0.0f);                // out of range: zero, not 9
    }
    caliper_producer_error_free(&e);
}

TEST_CASE("the read-only rule refuses an input declared read_write and a store to a read input") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Desc d(kRowNorms, f.w);
    uint32_t probe = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &probe, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_remove(f.rt, probe);
    caliper_producer_error_free(&e);

    std::string code;
    d.set(replaced(kRowNorms, "var<storage, read> w", "var<storage, read_write> w").c_str());
    CHECK(compile_code(f.rt, d.d, &code) == CALIPER_PRODUCER_REFUSED);
    CHECK(code == "program.writes_input");

    // Naga refuses a store to var<storage, read>, with the line of the store.
    d.set(replaced(kRowNorms, "let i = atomicAdd", "w[row] = 0.0;\n    let i = atomicAdd").c_str());
    uint32_t line = 0;
    CHECK(compile_code(f.rt, d.d, &code, &line) == CALIPER_PRODUCER_REFUSED);
    CHECK(code == "program.compile_failed");
    CHECK(line == 13);
}

TEST_CASE("the binder refuses undeclared bindings, missing outputs, unknown inputs and the memory budget") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    Desc d(kRowNorms, f.w);
    uint32_t probe = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &probe, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_error_free(&e);
    std::string code;

    std::string other = replaced(kRowNorms, "@compute",
                                 "@group(1) @binding(0) var<storage, read_write> other: array<f32>;\n@compute");
    other = replaced(other, "positions[i * 3u] = f32(row);", "positions[i * 3u] = f32(row);\n    other[0] = 1.0;");
    d.set(other.c_str());
    CHECK(compile_code(f.rt, d.d, &code) == CALIPER_PRODUCER_REFUSED);
    CHECK(code == "program.binding_undeclared");

    d.set(replaced(kRowNorms, "let i = atomicAdd(&count, 1u);", "let i = row;").c_str());
    CHECK(compile_code(f.rt, d.d, &code) == CALIPER_PRODUCER_REFUSED);
    CHECK(code == "program.output_mismatch");

    d.set(kRowNorms);
    d.input.tensor = 99;
    CHECK(compile_code(f.rt, d.d, &code) == CALIPER_PRODUCER_REFUSED);
    CHECK(code == "program.unknown_input");

    d.input.tensor = f.w;
    d.d.memory_limit_bytes = 64;   // 48 + 4 bytes of outputs and 128 of the tensor
    CHECK(compile_code(f.rt, d.d, &code) == CALIPER_PRODUCER_OVER_BUDGET);
    CHECK(code == "program.over_memory");

    d.d.memory_limit_bytes = 64u << 20;
    d.d.max_count = 2000000;
    CHECK(compile_code(f.rt, d.d, &code) == CALIPER_PRODUCER_REFUSED);
    CHECK(code == "program.over_limits");
}

TEST_CASE("an import that cannot be in place is refused, never copied") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    caliper_producer_error e{};
    uint32_t h = 0;
    caliper_producer_tensor t = f.desc_of(0, 8192);   // longer than its buffer
    CHECK(caliper_producer_import(f.rt, &t, &h, &e) == CALIPER_PRODUCER_IMPORT_REFUSED);
    CHECK(std::string(e.code) == "program.import_refused");
    caliper_producer_error_free(&e);
    t = f.desc_of(4, 64);                              // not at the storage offset alignment
    CHECK(caliper_producer_import(f.rt, &t, &h, &e) == CALIPER_PRODUCER_IMPORT_REFUSED);
    caliper_producer_error_free(&e);
    NSObject* other = [NSObject new];
    t = f.desc_of(0, 64);
    t.buffer = (__bridge void*)other;                  // not an MTLBuffer
    CHECK(caliper_producer_import(f.rt, &t, &h, &e) == CALIPER_PRODUCER_IMPORT_REFUSED);
    CHECK(h == 0);
    caliper_producer_error_free(&e);
}

TEST_CASE("the watchdog stops a program over its time limit, and a timeout stops every program") {
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    // A long loop over the row, thus every pass is over the sandbox's least limit of 0.01 ms.
    const std::string slow = replaced(kRowNorms, "c < 8u; c++) { let v = w[row * 8u + c];",
                                      "c < 200000u; c++) { let v = w[row * 8u + c % 8u];");
    Desc d(slow.c_str(), f.w);
    d.d.time_limit_ms = 0.01;
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 1};
    caliper_producer_pass pass{};
    for (int i = 0; i < 3; ++i) REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    CHECK(pass.stopped == 1);
    caliper_producer_program_status st{};
    REQUIRE(caliper_producer_read_status(f.rt, prog, &st) == CALIPER_PRODUCER_OK);
    CHECK(st.state == CALIPER_PRODUCER_STOPPED);
    CHECK(st.runs == 3);
    REQUIRE(st.stop_code);
    CHECK(std::string(st.stop_code) == "program.over_time");
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK);
    CHECK(pass.ran == 0);   // a stopped program does not run again

    d.d.time_limit_ms = 50.0;
    d.d.id = "norms2";
    uint32_t again = 0;
    REQUIRE(caliper_producer_compile(f.rt, &d.d, &again, &e) == CALIPER_PRODUCER_OK);
    CHECK(caliper_producer_run(f.rt, &again, 1, &frame, 1e-9, &pass, &e) == CALIPER_PRODUCER_TIMEOUT);
    CHECK(std::string(e.code) == "program.timeout");
    caliper_producer_error_free(&e);
    REQUIRE(caliper_producer_read_status(f.rt, again, &st) == CALIPER_PRODUCER_OK);
    CHECK(st.state == CALIPER_PRODUCER_STOPPED);
    uint32_t after = 0;
    CHECK(caliper_producer_compile(f.rt, &d.d, &after, &e) == CALIPER_PRODUCER_TIMEOUT);
    caliper_producer_error_free(&e);
}

TEST_CASE("after a timeout the runtime goes without a wait while its pass still runs on the GPU") {
    using clock = std::chrono::steady_clock;
    const auto settle = clock::now() + std::chrono::seconds(20);   // the reaper of earlier cases
    while (caliper_producer_reaping() != 0 && clock::now() < settle)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    REQUIRE(caliper_producer_reaping() == 0);
    Fixture f;
    if (f.device == nil) { MESSAGE("no Metal device"); return; }
    // One invocation that loops w[1] times: the case sets the count so that the pass lasts
    // about 400 ms, far past a timeout of 50 ms.
    Desc d(R"(@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> count: atomic<u32>;
@compute @workgroup_size(1)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    var s = w[0];
    let n = u32(w[1]);
    for (var k = 0u; k < n; k++) { s = s * 0.999999 + w[k % 8u]; }
    let i = atomicAdd(&count, 1u);
    positions[i * 3u] = s;
    positions[i * 3u + 1u] = 0.0;
    positions[i * 3u + 2u] = 0.0;
}
)", f.w);
    d.d.max_count = 1;
    d.d.workgroups[0] = d.d.workgroups[1] = d.d.workgroups[2] = 1;
    auto* w = static_cast<float*>([f.tensor contents]);
    w[1] = 100000.0f;
    uint32_t prog = 0;
    caliper_producer_error e{};
    if (!compile_or_skip(f.rt, d.d, &prog, &e)) { caliper_producer_error_free(&e); return; }
    caliper_producer_frame frame{1, 0.0, 1};
    REQUIRE(caliper_producer_run(f.rt, &prog, 1, &frame, 5000.0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_program_status st{};
    REQUIRE(caliper_producer_read_status(f.rt, prog, &st) == CALIPER_PRODUCER_OK);
    REQUIRE(st.last_pass_ms > 0.0);
    w[1] = float(std::min(1e9, 100000.0 * 400.0 / st.last_pass_ms));
    MESSAGE("100000 iterations took ", st.last_pass_ms, " ms; the long pass runs ", w[1], " of them");

    const auto t0 = clock::now();
    CHECK(caliper_producer_run(f.rt, &prog, 1, &frame, 50.0, nullptr, &e) == CALIPER_PRODUCER_TIMEOUT);
    REQUIRE(e.code);
    CHECK(std::string(e.code) == "program.timeout");
    CHECK(std::string(e.message).find("cannot be stopped") != std::string::npos);
    caliper_producer_error_free(&e);
    const auto t1 = clock::now();
    caliper_producer_destroy(f.rt);   // no wait for the pass that still runs
    f.rt = nullptr;
    const double destroy_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
    const size_t reaping = caliper_producer_reaping();
    MESSAGE("the timeout came after ", std::chrono::duration<double, std::milli>(t1 - t0).count(),
            " ms; the destroy took ", destroy_ms, " ms");
    CHECK(destroy_ms < 50.0);
    CHECK(reaping == 1);
    // The reaper lets the runtime's objects go when the GPU ends the pass.
    const auto deadline = clock::now() + std::chrono::seconds(20);
    while (caliper_producer_reaping() != 0 && clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(caliper_producer_reaping() == 0);
}
