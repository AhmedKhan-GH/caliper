// The producer tests of the operators, the tensor output and the params
// (design/instrument-interface.md, Operators, Tensor Output and Program
// Parameters): the built-in WGSL of `surface` and `path` runs on the device
// over a shared buffer that stands for a training tensor, and its `out`
// matches the CPU reference of caliper_builder/programs/operator_kernels.h,
// which the app's operators run. A change of params applies at the next pass
// with no compile.
//
// The WGSL sources are read from the source tree, which the case finds above
// its working directory. Without a Metal device, the WGSL compiler or the
// sources, a case returns early with a message.
#include <doctest/doctest.h>

#include "producer/caliper_producer.h"

#include "../../../builder/include/caliper_builder/programs/operator_kernels.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__)
// test_producer_metal.mm
void* caliper_test_shared_buffer(size_t bytes, void** contents);
const void* caliper_test_buffer_contents(void* buffer);
void caliper_test_release_buffer(void* buffer);
#endif

namespace {

namespace k = caliper_builder::programs::operator_kernels;

// The text of src/builder/operators/wgsl/<op>.wgsl, found in a parent of the
// working directory; empty when it is not there.
std::string operator_wgsl(const char* op) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir = fs::current_path(ec);
    for (int up = 0; up < 8 && !dir.empty(); ++up) {
        const fs::path file = dir / "src" / "builder" / "operators" / "wgsl" / (std::string(op) + ".wgsl");
        if (fs::exists(file, ec)) {
            std::ifstream in(file);
            std::stringstream ss;
            ss << in.rdbuf();
            return ss.str();
        }
        if (!dir.has_parent_path() || dir.parent_path() == dir) break;
        dir = dir.parent_path();
    }
    return {};
}

#if defined(__APPLE__)

// A runtime with one imported (rows, cols) f32 tensor, filled by `fill`.
struct Device {
    void* buffer = nullptr;
    float* data = nullptr;
    caliper_producer* rt = nullptr;
    uint32_t tensor = 0;
    int64_t rows = 0, cols = 0;

    Device(int64_t r, int64_t c) : rows(r), cols(c) {
        void* contents = nullptr;
        buffer = caliper_test_shared_buffer(size_t(r * c * 4), &contents);
        if (!buffer) return;
        data = static_cast<float*>(contents);
        caliper_producer_error e{};
        REQUIRE(caliper_producer_create(nullptr, &rt, &e) == CALIPER_PRODUCER_OK);
        caliper_producer_tensor t{};
        t.name = "blocks.0.attn.proj.weight";
        t.buffer = buffer;
        t.byte_size = uint64_t(r * c * 4);
        t.dtype = CALIPER_PRODUCER_F32;
        t.ndim = 2;
        t.shape[0] = r;
        t.shape[1] = c;
        REQUIRE(caliper_producer_import(rt, &t, &tensor, &e) == CALIPER_PRODUCER_OK);
        caliper_producer_error_free(&e);
    }
    ~Device() {
        caliper_producer_destroy(rt);
        caliper_test_release_buffer(buffer);
    }
    bool ok() const { return buffer && rt; }
};

// An operator program as the app declares it (operators.h, operator_program()).
struct OperatorDesc {
    std::string wgsl;
    caliper_producer_input input{"input", 0};
    std::vector<caliper_producer_param> params;
    std::vector<std::string> names;
    caliper_producer_program_desc d{};

    OperatorDesc(const char* op, uint32_t tensor, std::vector<std::pair<std::string, double>> values,
                 std::vector<int64_t> shape) {
        wgsl = operator_wgsl(op);
        input.tensor = tensor;
        for (auto& [n, v] : values) names.push_back(n);
        for (std::size_t i = 0; i < values.size(); ++i) params.push_back({names[i].c_str(), values[i].second});
        d.id = op[0] == 's' ? "operator.terrain.0" : "operator.trail.0";
        d.wgsl = wgsl.c_str();
        d.wgsl_size = wgsl.size();
        d.entry = "run";
        d.inputs = &input;
        d.input_count = 1;
        d.output_kind = CALIPER_PRODUCER_TENSOR;
        d.tensor_ndim = uint32_t(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i) d.tensor_shape[i] = shape[i];
        d.params = params.data();
        d.param_count = params.size();
        uint64_t n = 1;
        for (int64_t s : shape) n *= uint64_t(s);
        d.workgroups[0] = uint32_t((n + k::kWorkgroupSize - 1) / k::kWorkgroupSize);
        d.workgroups[1] = d.workgroups[2] = 1;
        d.time_limit_ms = 50.0;
        d.memory_limit_bytes = 64u << 20;
    }
};

// Compiles; false (with a message) when this build has no WGSL compiler.
bool compile(caliper_producer* rt, const caliper_producer_program_desc& d, uint32_t* h) {
    caliper_producer_error e{};
    const caliper_producer_status s = caliper_producer_compile(rt, &d, h, &e);
    const std::string code = e.code ? e.code : "";
    const std::string message = e.message ? e.message : "";
    caliper_producer_error_free(&e);
    if (s == CALIPER_PRODUCER_UNSUPPORTED && code == "program.compiler_unavailable") {
        MESSAGE("no WGSL compiler in this build: ", message);
        return false;
    }
    if (s != CALIPER_PRODUCER_OK) FAIL_CHECK(std::string(code + ": " + message));
    return s == CALIPER_PRODUCER_OK;
}

// Runs one pass at `step` and gives the elements of `out`.
std::vector<float> pass(caliper_producer* rt, uint32_t prog, int64_t step) {
    caliper_producer_error e{};
    caliper_producer_frame frame{step, 0.0, 1};
    REQUIRE(caliper_producer_run(rt, &prog, 1, &frame, 0, nullptr, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_output out{};
    REQUIRE(caliper_producer_read_output(rt, prog, "out", &out, &e) == CALIPER_PRODUCER_OK);
    caliper_producer_error_free(&e);
    CHECK(out.words == 1);
    const float* f = static_cast<const float*>(caliper_test_buffer_contents(out.buffer));
    REQUIRE(f != nullptr);
    return std::vector<float>(f, f + out.count);
}

// The device's pooled heights against the reference: NaN at the same cells,
// else equal up to the rounding of a sum of at most f * f values.
void check_heights(const std::vector<float>& device, const std::vector<float>& reference) {
    REQUIRE(device.size() == reference.size());
    std::size_t nan = 0;
    for (std::size_t i = 0; i < device.size(); ++i) {
        const bool a = k::is_nan_bits(device[i]), b = k::is_nan_bits(reference[i]);
        CHECK(a == b);
        if (a || b) {
            ++nan;
            continue;
        }
        CHECK(std::fabs(device[i] - reference[i]) <= 1e-5f * std::max(1.0f, std::fabs(reference[i])));
    }
    MESSAGE(device.size(), " cells compared, ", nan, " of them NaN");
}

#endif

}  // namespace

TEST_CASE("operators: the built-in WGSL sources are in the source tree") {
    const std::string surface = operator_wgsl("surface"), path = operator_wgsl("path");
    if (surface.empty()) {
        MESSAGE("the source tree is not above the working directory; the device cases skip");
        return;
    }
    CHECK(surface.find("struct Params") != std::string::npos);
    CHECK(path.find("struct Params") != std::string::npos);
}

#if defined(__APPLE__)

TEST_CASE("operators: surface on the device matches the CPU reference for each pool, and params apply at the next pass") {
    // A (75, 41) tensor with NaN cells and one block of NaN only, pooled into at most 300 cells.
    const int64_t rows = 75, cols = 41, max_cells = 300;
    Device dev(rows, cols);
    if (!dev.ok()) { MESSAGE("no Metal device"); return; }
    if (operator_wgsl("surface").empty()) { MESSAGE("no WGSL sources above the working directory"); return; }
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c)
            dev.data[r * cols + c] = float(std::sin(0.21 * double(r)) * std::cos(0.13 * double(c)) + 0.01 * double(r));
    // f is 4, thus the last block of the first block row holds only the column 40: NaN there
    // makes a block of NaN only, and one more NaN sits inside a block of numbers.
    const int64_t f = k::pool_factor(rows, cols, max_cells);
    CHECK(f == 4);
    dev.data[3 * cols + 5] = k::quiet_nan();
    for (int64_t r = 0; r < f; ++r) dev.data[r * cols + 40] = k::quiet_nan();
    const int64_t out_rows = k::ceil_div(rows, f), out_cols = k::ceil_div(cols, f);
    CHECK(out_rows * out_cols <= max_cells);
    OperatorDesc od("surface", dev.tensor,
                    {{"rows", double(rows)},
                     {"cols", double(cols)},
                     {"factor", double(f)},
                     {"out_rows", double(out_rows)},
                     {"out_cols", double(out_cols)},
                     {"pool", double(k::kPoolMean)}},
                    {out_rows, out_cols});
    uint32_t prog = 0;
    if (!compile(dev.rt, od.d, &prog)) return;

    std::vector<float> ref;
    k::surface_heights(dev.data, rows, cols, f, k::kPoolMean, &ref);
    check_heights(pass(dev.rt, prog, 10), ref);

    // A new pool through the params: the next pass reads it, with no compile.
    for (uint32_t pool : {k::kPoolMax, k::kPoolMin}) {
        caliper_producer_param p{"pool", double(pool)};
        caliper_producer_error e{};
        REQUIRE(caliper_producer_set_params(dev.rt, prog, &p, 1, &e) == CALIPER_PRODUCER_OK);
        caliper_producer_error_free(&e);
        k::surface_heights(dev.data, rows, cols, f, pool, &ref);
        check_heights(pass(dev.rt, prog, 11 + pool), ref);
    }
    // An unknown name changes nothing.
    caliper_producer_param bad{"level", 1.0};
    caliper_producer_error e{};
    CHECK(caliper_producer_set_params(dev.rt, prog, &bad, 1, &e) == CALIPER_PRODUCER_REFUSED);
    CHECK(std::string(e.code ? e.code : "") == "program.params_mismatch");
    caliper_producer_error_free(&e);
    caliper_producer_program_status st{};
    REQUIRE(caliper_producer_read_status(dev.rt, prog, &st) == CALIPER_PRODUCER_OK);
    CHECK(st.runs == 3);
}

TEST_CASE("operators: a surface over a tensor that grows by one row each step follows the reference at each pass") {
    // The terrain of a run: a (steps, 16) tensor whose row k holds step k, filled one row a step.
    const int64_t max_rows = 48, cols = 16;
    Device dev(max_rows, cols);
    if (!dev.ok()) { MESSAGE("no Metal device"); return; }
    if (operator_wgsl("surface").empty()) { MESSAGE("no WGSL sources above the working directory"); return; }
    std::memset(dev.data, 0, size_t(max_rows * cols * 4));
    for (int64_t step = 8; step <= max_rows; step += 8) {
        for (int64_t r = 0; r < step; ++r)
            for (int64_t c = 0; c < cols; ++c)
                dev.data[r * cols + c] = float(std::exp(-0.05 * double(r)) * std::sin(0.4 * double(c) + 0.1 * double(r)));
        // The app sends the operator again when the catalog's shape changes; the runtime then
        // binds the first `step` rows of the same buffer.
        OperatorDesc od("surface", dev.tensor,
                        {{"rows", double(step)}, {"cols", double(cols)}, {"factor", 1.0},
                         {"out_rows", double(step)}, {"out_cols", double(cols)}, {"pool", 0.0}},
                        {step, cols});
        uint32_t prog = 0;
        if (!compile(dev.rt, od.d, &prog)) return;
        std::vector<float> ref;
        k::surface_heights(dev.data, step, cols, 1, k::kPoolMean, &ref);
        check_heights(pass(dev.rt, prog, step), ref);
        caliper_producer_remove(dev.rt, prog);
    }
}

TEST_CASE("operators: path on the device keeps every s-th row as the CPU reference does") {
    // 1,000 rows of 5 columns, at most 64 points: the stride is 16, and the output keeps the
    // first 4 columns, because the path reads the columns 3, 0 and 1.
    const int64_t rows = 1000, cols = 5, max_points = 64;
    Device dev(rows, cols);
    if (!dev.ok()) { MESSAGE("no Metal device"); return; }
    if (operator_wgsl("path").empty()) { MESSAGE("no WGSL sources above the working directory"); return; }
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) dev.data[r * cols + c] = float(r) * 0.5f + float(c) * 100.0f;
    const int64_t s = k::path_stride(rows, max_points), points = k::ceil_div(rows, s), width = 4;
    CHECK(s == 16);
    CHECK(points == 63);
    OperatorDesc od("path", dev.tensor,
                    {{"rows", double(rows)}, {"cols", double(cols)}, {"stride", double(s)},
                     {"points", double(points)}, {"out_cols", double(width)}},
                    {points, width});
    uint32_t prog = 0;
    if (!compile(dev.rt, od.d, &prog)) return;
    const std::vector<float> device = pass(dev.rt, prog, 1);
    std::vector<float> ref;
    k::path_rows(dev.data, rows, cols, width, s, &ref);
    REQUIRE(device.size() == ref.size());
    CHECK(std::memcmp(device.data(), ref.data(), ref.size() * 4) == 0);
    // The app's reference over the output with the step's columns gives the path itself.
    std::vector<float> from_output, from_input;
    k::path_positions(device.data(), points, width, {3, 0, 1}, 3, 1, &from_output);
    k::path_positions(dev.data, rows, cols, {3, 0, 1}, 3, s, &from_input);
    CHECK(from_output == from_input);
}

TEST_CASE("operators: the runtime refuses params that differ from struct Params") {
    Device dev(4, 4);
    if (!dev.ok()) { MESSAGE("no Metal device"); return; }
    if (operator_wgsl("surface").empty()) { MESSAGE("no WGSL sources above the working directory"); return; }
    OperatorDesc od("surface", dev.tensor,
                    {{"rows", 4.0}, {"cols", 4.0}, {"factor", 1.0}, {"out_rows", 4.0}, {"out_cols", 4.0}, {"level", 0.0}},
                    {4, 4});
    uint32_t prog = 0;
    caliper_producer_error e{};
    const caliper_producer_status s = caliper_producer_compile(dev.rt, &od.d, &prog, &e);
    const std::string code = e.code ? e.code : "";
    caliper_producer_error_free(&e);
    if (code == "program.compiler_unavailable") { MESSAGE("no WGSL compiler"); return; }
    CHECK(s == CALIPER_PRODUCER_REFUSED);
    CHECK(code == "program.params_mismatch");
    CHECK(prog == 0);
}

#endif
