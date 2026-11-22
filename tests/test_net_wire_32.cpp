// The scenes of the 3.2 draft in the C++ codec (src/host/net/wire.h) against
// the golden frames of tests/fixtures/wire_v3/v3_2, which python/caliper/
// _wire.py wrote: every frame decodes to what its .json says and encodes back
// to the same bytes at minor 2 (except the one that carries a section no part
// has), every invalid frame is refused as v3_2/invalid/expect.json says, every
// part, stamps, entity transforms and class tables carry their golden values,
// the limits hold, the scene view helpers follow python/tests/test_wire_32.py,
// a forwarder can rewrite `sub` and keep the sections, and truncations and bit
// flips never crash. The instance form of PROGRAM and RESOURCE carry their
// golden values, and a hub forwards both to the viewers that announced
// `resources` and keeps the newest of each for a late viewer.
#include <doctest/doctest.h>

#include "net/json.h"
#include "net/wire.h"
#if !defined(_WIN32)   // the hub and the WebSocket layer are not built on Windows yet
#include "live/live_programs.h"
#include "net/hub.h"
#include "net/ws.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <thread>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace caliper_net;
using namespace caliper_net::wire;
namespace fs = std::filesystem;

namespace {

fs::path v32_dir() {
    return fs::path(CALIPER_TEST_SOURCE_ROOT) / "tests" / "fixtures" / "wire_v3" / "v3_2";
}

std::vector<uint8_t> read_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

Json read_json(const fs::path& p) {
    const std::vector<uint8_t> b = read_bytes(p);
    Json j;
    std::string err;
    REQUIRE_MESSAGE(Json::parse((const char*)b.data(), b.size(), &j, &err, 64), p.string() << ": " << err);
    return j;
}

std::vector<std::string> v32_names() {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(v32_dir()))
        if (e.path().extension() == ".bin") out.push_back(e.path().stem().string());
    std::sort(out.begin(), out.end());
    return out;
}

bool decode_only(const std::string& n) { return n == "scene_update_points_future"; }

uint64_t ju(const Json& j, const char* key) {
    uint64_t v = 0;
    REQUIRE(j.find(key) != nullptr);
    REQUIRE(j.find(key)->get_u64(&v));
    return v;
}

template <typename T>
T at(const Section& s, size_t k) {
    T v;
    std::memcpy(&v, s.data + sizeof(T) * k, sizeof(T));
    return v;
}

DecodeResult decode_bytes(const std::vector<uint8_t>& b, Message* m) { return decode(b.data(), b.size(), m); }

// A scene_update frame over caller-owned bytes, for the limit cases.
std::vector<uint8_t> update_frame(const std::string& meta_json, const std::vector<Section>& sections) {
    Json meta;
    std::string err;
    REQUIRE_MESSAGE(Json::parse(meta_json.data(), meta_json.size(), &meta, &err), err);
    std::vector<uint8_t> out;
    FrameError fe;
    REQUIRE_MESSAGE(encode_frame(CALIPER_NET_MSG_SCENE_UPDATE, meta, sections, &out, &fe, CALIPER_NET_MINOR_3_2), fe.message);
    return out;
}

const char* kBase = "\"sub\":1,\"view\":\"v\",\"seq\":1,\"stamp_ns\":1,\"path\":\"a\"";

}  // namespace

TEST_CASE("net/wire 3.2: every golden frame decodes as described and encodes back to the same bytes") {
    const std::vector<std::string> names = v32_names();
    REQUIRE(names.size() >= 66);
    for (const std::string& name : names) {
        CAPTURE(name);
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / (name + ".bin"));
        const Json want = read_json(v32_dir() / (name + ".json"));
        Frame f;
        FrameError fe;
        REQUIRE_MESSAGE(decode_frame(bin.data(), bin.size(), &f, &fe), fe.reason << ": " << fe.message);
        const Json& h = *want.find("header");
        CHECK(f.type == ju(h, "type"));
        CHECK(f.major == ju(h, "major"));
        CHECK(f.minor == CALIPER_NET_MINOR_3_2);
        CHECK(f.minor == ju(h, "minor"));
        CHECK(f.flags == ju(h, "flags"));
        CHECK(bin.size() == ju(h, "frame_len"));
        CHECK(*want.find("meta") == f.meta);
        const Json& secs = *want.find("sections");
        REQUIRE(f.sections.size() == secs.size());
        for (size_t i = 0; i < f.sections.size(); ++i) {
            const Section& s = f.sections[i];
            const Json& d = secs.at(i);
            CHECK(s.name == d.find("name")->as_string());
            CHECK(s.dtype == (int64_t)ju(d, "dtype"));
            CHECK(s.offset == ju(d, "offset"));
            CHECK(s.size == ju(d, "length"));
            const Json& shape = *d.find("shape");
            REQUIRE(s.shape.size() == shape.size());
            for (size_t k = 0; k < shape.size(); ++k) CHECK(s.shape[k] == (int64_t)shape.at(k).as_double());
            const Json& vals = *d.find("values");
            REQUIRE((int64_t)vals.size() == s.count());
            for (size_t k = 0; k < vals.size(); ++k) {
                if (s.dtype == CALIPER_NET_DTYPE_F32) CHECK(at<float>(s, k) == (float)vals.at(k).as_double());
                else if (s.dtype == CALIPER_NET_DTYPE_U8) CHECK(at<uint8_t>(s, k) == (uint8_t)vals.at(k).as_double());
                else if (s.dtype == CALIPER_NET_DTYPE_I32) CHECK(at<int32_t>(s, k) == (int32_t)vals.at(k).as_double());
                else if (s.dtype == CALIPER_NET_DTYPE_U16) CHECK(at<uint16_t>(s, k) == (uint16_t)vals.at(k).as_double());
                else if (s.dtype == CALIPER_NET_DTYPE_I64) {
                    int64_t want = 0;
                    REQUIRE(vals.at(k).get_i64(&want));
                    CHECK(at<int64_t>(s, k) == want);
                }
            }
        }

        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        REQUIRE_MESSAGE(r.ok(), r.error_code << ": " << r.message);
        CHECK(message_type(m) == f.type);
        CHECK(std::string(type_name(f.type)) == want.find("type_name")->as_string());
        if (f.type == CALIPER_NET_MSG_SCENE_UPDATE) CHECK(std::holds_alternative<SceneUpdate>(m));
        if (f.type == CALIPER_NET_MSG_SCENE_CLEAR) CHECK(std::holds_alternative<SceneClear>(m));
        if (f.type == CALIPER_NET_MSG_SCENE_TRANSFORMS) CHECK(std::holds_alternative<SceneTransforms>(m));
        if (f.type == CALIPER_NET_MSG_SCENE_CLASSES) CHECK(std::holds_alternative<SceneClasses>(m));
        if (f.type == CALIPER_NET_MSG_PROGRAM) CHECK(std::holds_alternative<Program>(m));
        if (f.type == CALIPER_NET_MSG_PROGRAM_STATE) CHECK(std::holds_alternative<ProgramState>(m));
        if (f.type == CALIPER_NET_MSG_RESOURCE) CHECK(std::holds_alternative<Resource>(m));

        std::vector<uint8_t> again;
        std::string err;
        REQUIRE_MESSAGE(encode(m, &again, &err, CALIPER_NET_MINOR_3_2), err);
        if (decode_only(name)) CHECK(again != bin);   // a section no part has is not written
        else CHECK(again == bin);
    }
}

TEST_CASE("net/wire 3.2 programs") {
    auto load = [](const char* name) {
        static std::vector<std::vector<uint8_t>> keep;   // a decoded Section points into its buffer
        keep.push_back(read_bytes(v32_dir() / (std::string(name) + ".bin")));
        Message m;
        const DecodeResult r = decode_bytes(keep.back(), &m);
        REQUIRE_MESSAGE(r.ok(), name << ": " << r.error_code << ": " << r.message);
        return m;
    };
    {
        const Message m = load("program_set_geometry");
        const Program& p = std::get<Program>(m);
        CHECK(p.op == CALIPER_NET_PROGRAM_SET);
        CHECK(p.id == "rows");
        CHECK(p.scene.value_or("") == "world");
        CHECK(p.request_id == 11);
        REQUIRE(p.wgsl.has_value());
        CHECK(!p.wgsl->empty());
        CHECK(p.wgsl->back() == '\n');
        CHECK(p.sha256.has_value());
        CHECK(p.declaration.find("output")->find("max_count")->as_double() == 4096);
    }
    {
        const Message m = load("program_set_pixels");
        const Program& p = std::get<Program>(m);
        CHECK(!p.scene.has_value());
        CHECK(p.declaration.find("output")->find("width")->as_double() == 64);
    }
    {
        const Message m = load("program_remove");
        const Program& p = std::get<Program>(m);
        CHECK(p.op == CALIPER_NET_PROGRAM_REMOVE);
        CHECK(p.declaration.is_null());
        CHECK(!p.wgsl.has_value());
        CHECK(!p.sha256.has_value());
    }
    {
        const Message m = load("program_state_reply");
        const ProgramState& s = std::get<ProgramState>(m);
        CHECK(s.request_id.value_or(0) == 12);
        CHECK(s.step.value_or(0) == 1200);
        CHECK(s.step_ms.value_or(0) == 31.25);
        CHECK(s.programs.size() == 2);
        const Json& run = s.programs.items().front();   // the cadence keys of a running program
        REQUIRE(run.find("overhead_share"));
        REQUIRE(run.find("passes_per_s"));
        CHECK(run.find("overhead_share")->as_double() == 0.03125);
        CHECK(run.find("passes_per_s")->as_double() == 2.5);
    }
    {
        const Message m = load("program_state_waiting");
        const ProgramState& s = std::get<ProgramState>(m);
        CHECK(!s.request_id.has_value());
        CHECK(!s.step.has_value());
        CHECK(s.programs.size() == 1);
    }
    {
        const Message m = load("control_intervene_clear_all");
        const Control& c = std::get<Control>(m);
        CHECK(c.op == CALIPER_NET_OP_INTERVENE);
        CHECK(c.intervene.find("clear_all")->as_bool());
    }
    {
        const Message m = load("controls_interventions");
        const Controls& c = std::get<Controls>(m);
        REQUIRE(c.interventions.is_object());
        CHECK(c.interventions.find("points")->size() == 2);
    }
    {
        const Message m = load("control_reply_intervene");
        CHECK(std::get<ControlReply>(m).op == CALIPER_NET_OP_INTERVENE);
    }
}

// Encodes `p` and decodes it again, the way a peer would receive it.
DecodeResult program_round_trip(const Program& p, Message* out) {
    static std::vector<std::vector<uint8_t>> keep;   // a decoded Section points into its buffer
    keep.emplace_back();
    std::string err;
    REQUIRE_MESSAGE(encode(Message(p), &keep.back(), &err, CALIPER_NET_MINOR_3_2), err);
    return decode_bytes(keep.back(), out);
}

Json parse_json(const char* text) {
    Json j;
    std::string err;
    REQUIRE_MESSAGE(Json::parse(text, std::strlen(text), &j, &err, 64), err);
    return j;
}

Program params_program(const char* params_text) {
    Program p;
    p.request_id = 5;
    p.op = CALIPER_NET_PROGRAM_PARAMS;
    p.id = "rows";
    if (params_text) p.params = parse_json(params_text);
    return p;
}

// A valid pixels set from the golden frame, with its declaration replaced by `decl_text` when given.
Program set_program(const char* decl_text, const char* id = nullptr) {
    static std::vector<uint8_t> keep;
    keep = read_bytes(v32_dir() / "program_set_pixels.bin");
    Message m;
    const DecodeResult r = decode_bytes(keep, &m);
    REQUIRE(r.ok());
    Program p = std::get<Program>(m);
    if (decl_text) p.declaration = parse_json(decl_text);
    if (id) p.id = id;
    return p;
}

TEST_CASE("net/wire 3.2 program op params") {
    {   // A round trip keeps the values.
        Message m;
        const DecodeResult r = program_round_trip(params_program(R"({"gain": 2.5, "_k1": -1, "A_b": 0})"), &m);
        REQUIRE_MESSAGE(r.ok(), r.error_code << ": " << r.message);
        const Program& p = std::get<Program>(m);
        CHECK(p.op == CALIPER_NET_PROGRAM_PARAMS);
        CHECK(p.request_id == 5);
        CHECK(p.id == "rows");
        CHECK(p.declaration.is_null());
        CHECK(!p.wgsl.has_value());
        CHECK(p.params.size() == 3);
        CHECK(p.params.find("gain")->as_double() == 2.5);
        CHECK(p.params.find("_k1")->as_double() == -1);
    }
    Message m;
    CHECK(!program_round_trip(params_program(nullptr), &m).ok());   // no params
    CHECK(!program_round_trip(params_program("{}"), &m).ok());      // no name
    {
        std::string seventeen = "{";
        for (int i = 0; i < 17; ++i) seventeen += std::string(i ? "," : "") + "\"p" + std::to_string(i) + "\": 1";
        seventeen += "}";
        CHECK(!program_round_trip(params_program(seventeen.c_str()), &m).ok());
        std::string sixteen = "{";
        for (int i = 0; i < 16; ++i) sixteen += std::string(i ? "," : "") + "\"p" + std::to_string(i) + "\": 1";
        sixteen += "}";
        CHECK(program_round_trip(params_program(sixteen.c_str()), &m).ok());
    }
    CHECK(!program_round_trip(params_program(R"({"1a": 1})"), &m).ok());      // bad name
    CHECK(!program_round_trip(params_program(R"({"a-b": 1})"), &m).ok());
    CHECK(!program_round_trip(params_program(R"({"__x": 1})"), &m).ok());
    CHECK(!program_round_trip(params_program(R"({"a": "1"})"), &m).ok());     // not a number
    {
        Program p = params_program(R"({"a": 1})");
        p.params.set("a", std::nan(""));
        CHECK(!program_round_trip(p, &m).ok());   // not finite
        p.params.set("a", INFINITY);
        CHECK(!program_round_trip(p, &m).ok());
    }
    {
        Program p = params_program(R"({"a": 1})");
        p.declaration = parse_json(R"({"output": {"kind": "pixels", "width": 4, "height": 4}})");
        CHECK(!program_round_trip(p, &m).ok());   // a declaration
        p = params_program(R"({"a": 1})");
        const std::string w = "fn main() {}";
        p.wgsl = w;
        CHECK(!program_round_trip(p, &m).ok());   // a section
    }
}

TEST_CASE("net/wire 3.2 program declaration keys and ids") {
    Message m;
    CHECK(program_round_trip(set_program(nullptr), &m).ok());
    CHECK(program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [4, 8]},
        "params": {"gain": 1.5}, "builtin": "surface@1"})"), &m).ok());
    CHECK(program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [1000000]}})"), &m).ok());
    CHECK(program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [100, 100, 10, 1]}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [1, 1, 1, 1, 1]}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [1000001]}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [1001, 1000]}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": []}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [0, 4]}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "tensor"}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "tensor", "shape": [4], "width": 4}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "pixels", "width": 4, "height": 4},
        "params": {"__a": 1}})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "pixels", "width": 4, "height": 4},
        "builtin": ""})"), &m).ok());
    CHECK(!program_round_trip(set_program(R"({"output": {"kind": "pixels", "width": 4, "height": 4},
        "builtin": "0123456789012345678901234567890123"})"), &m).ok());

    CHECK(program_round_trip(set_program(nullptr, "operator.terrain.0"), &m).ok());
    CHECK(program_round_trip(set_program(nullptr, "operator.a-b_1.12345"), &m).ok());
    CHECK(!program_round_trip(set_program(nullptr, "operator.Terrain.0"), &m).ok());
    CHECK(!program_round_trip(set_program(nullptr, "operator.terrain."), &m).ok());
    CHECK(!program_round_trip(set_program(nullptr, "operator..0"), &m).ok());
    CHECK(!program_round_trip(set_program(nullptr, "operator.terrain.123456"), &m).ok());
    CHECK(!program_round_trip(set_program(nullptr, "operator.1x.0"), &m).ok());
    CHECK(!program_round_trip(set_program(nullptr, "operator.terrain.x"), &m).ok());
    CHECK(!program_round_trip(set_program(nullptr, ("operator." + std::string(50, 'a') + ".0").c_str()), &m).ok());
    CHECK(program_round_trip(set_program(nullptr, "operatorx"), &m).ok());   // an ordinary id
    CHECK(!program_round_trip(set_program(nullptr, "Rows"), &m).ok());
}

TEST_CASE("net/wire 3.2 program status items of a tensor and of an operator") {
    for (const char* kind : {"tensor", "operator", "mesh"}) {
        ProgramState st;
        st.programs = parse_json((std::string(R"([{"id": "operator.terrain.0", "kind": ")") + kind +
                                  R"(", "state": "running", "runs": 3}])").c_str());
        std::vector<uint8_t> bytes;
        std::string err;
        INFO(kind);
        const bool valid = std::string(kind) != "mesh";
        if (!encode(Message(st), &bytes, &err, CALIPER_NET_MINOR_3_2)) {   // the encoder may refuse it first
            CHECK_FALSE(valid);
            continue;
        }
        Message m;
        CHECK(decode_bytes(bytes, &m).ok() == valid);
    }
}

TEST_CASE("net/wire 3.2: instances of pieces and resources carry their golden values") {
    std::vector<std::vector<uint8_t>> keep;   // a decoded Section points into its buffer
    auto load = [&](const char* name) {
        keep.push_back(read_bytes(v32_dir() / (std::string(name) + ".bin")));
        Message m;
        const DecodeResult r = decode_bytes(keep.back(), &m);
        REQUIRE_MESSAGE(r.ok(), name << ": " << r.error_code << ": " << r.message);
        return m;
    };
    const Program p = std::get<Program>(load("program_set_piece"));
    CHECK(p.request_id == 31);
    CHECK(p.id == "heads");
    CHECK(p.piece.find("name")->as_string() == "head_bars");
    CHECK(p.piece.find("hash")->as_string().compare(0, 7, "sha256:") == 0);
    CHECK(p.bind.find("attn")->as_string() == "live:blocks.0.attn");
    CHECK(p.panel == std::optional<std::string>("bars"));
    CHECK(p.segments.size() == 2);
    CHECK(p.declaration.is_null());
    CHECK_FALSE(p.wgsl);
    REQUIRE(p.common);
    REQUIRE(p.derive);
    REQUIRE(p.draw);
    CHECK(p.derive->find("mean_heads") != std::string::npos);
    CHECK(p.draw->find("\xc2\xb5") != std::string::npos);
    const Program relay = std::get<Program>(load("program_set_piece_draw"));
    CHECK(relay.request_id == 0);
    CHECK_FALSE(relay.derive);
    CHECK(relay.draw == p.draw);
    const ProgramState st = std::get<ProgramState>(load("program_state_piece"));
    CHECK(st.programs.at(0).find("kind")->as_string() == "piece");

    const std::vector<uint8_t>* frame = nullptr;
    const Resource b = std::get<Resource>(load("resource_buffer"));
    frame = &keep.back();
    CHECK(b.program == "heads");
    CHECK(b.step == 1200);
    CHECK(b.count == 4);
    CHECK(b.stride == 4);
    CHECK_FALSE(b.width);
    REQUIRE(b.data.size == 16);
    // The section is a view of the frame's bytes, not a copy.
    CHECK(b.data.data >= frame->data());
    CHECK(b.data.data + b.data.size <= frame->data() + frame->size());
    float first = 0;
    std::memcpy(&first, b.data.data, 4);
    CHECK(first == 0.5f);
    const Resource t = std::get<Resource>(load("resource_texture"));
    CHECK(t.width == std::optional<uint32_t>(2));
    CHECK(t.height == std::optional<uint32_t>(2));
    CHECK(t.step == 1210);
    const Resource e = std::get<Resource>(load("resource_empty"));
    CHECK(e.count == 0);
    CHECK(e.stride == 24);
    CHECK(e.data.size == 0);
}

TEST_CASE("net/wire 3.2: a resource whose count times stride wraps in 32 bits is refused") {
    // count * stride = 2^32 + 256: a 32-bit product would equal the section's 256 bytes.
    const std::vector<uint8_t> zeros(256, 0);
    Resource r;
    r.program = "arcs";
    r.resource = "arcs";
    r.count = 16777217u;
    r.stride = 256;
    r.data = Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8, {256}, zeros.data(), zeros.size());
    std::vector<uint8_t> bytes;
    std::string err;
    REQUIRE(encode(Message(r), &bytes, &err, CALIPER_NET_MINOR_3_2));
    Message m;
    const DecodeResult res = decode_bytes(bytes, &m);
    CHECK(res.status == DecodeResult::Status::BadMessage);
    CHECK(res.error_code == "bad_request");
    // At the byte limit's count the check passes and only the section's length decides.
    r.count = CALIPER_NET_RESOURCE_MAX_BYTES / 256;
    REQUIRE(encode(Message(r), &bytes, &err, CALIPER_NET_MINOR_3_2));
    CHECK(decode_bytes(bytes, &m).message.find("count * stride") != std::string::npos);
    r.count = 1;
    REQUIRE(encode(Message(r), &bytes, &err, CALIPER_NET_MINOR_3_2));
    REQUIRE(decode_bytes(bytes, &m).ok());
    CHECK(std::get<Resource>(m).data.size == 256);
}

TEST_CASE("net/wire 3.2: decoded scene messages carry the golden values") {
    {
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "scene_update_mesh.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        CHECK(u.sub == 2);
        CHECK(u.view == "landscape");
        CHECK(u.seq == 39);
        CHECK(u.stamp_ns == 81234567893623ull);
        CHECK(u.path == "surface");
        CHECK(u.part == CALIPER_NET_PART_MESH);
        CHECK_FALSE(u.is_static);
        CHECK_FALSE(u.state);
        CHECK(u.axes.is_null());
        CHECK_FALSE(u.items.has_value());
        REQUIRE(u.arrays.size() == 3);
        CHECK(u.arrays[0].name == "positions");
        CHECK(u.arrays[1].name == "indices");
        CHECK(u.arrays[2].name == "values");
        const Section* idx = u.array("indices");
        REQUIRE(idx != nullptr);
        CHECK(idx->shape == std::vector<int64_t>{2, 3});
        const int32_t want[6] = {0, 1, 2, 1, 3, 2};
        for (size_t k = 0; k < 6; ++k) CHECK(at<int32_t>(*idx, k) == want[k]);
        CHECK(u.array("colors") == nullptr);
        // the sections point into the frame
        CHECK(idx->data >= bin.data());
        CHECK(idx->data + idx->size <= bin.data() + bin.size());
        CHECK(u.params.find("vmin")->as_double() == -4.0);
        CHECK(u.params.find("vmax")->as_double() == 2.0);
        const Json& stats = *u.params.find("stats");
        REQUIRE(stats.size() == 8);
        CHECK(stats.at(0).as_double() == -1.5);
    }
    {
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "scene_update_lines_strips.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        CHECK(u.part == CALIPER_NET_PART_LINES);
        CHECK(u.path == "world/tracks");
        CHECK(u.params.find("mode")->as_string() == "strips");
        CHECK(u.params.find("color_per")->as_string() == "strip");
        CHECK(u.params.find("width")->as_double() == 3.0);
        const Section* sl = u.array("strip_lengths");
        REQUIRE(sl != nullptr);
        REQUIRE(sl->shape == std::vector<int64_t>{2});
        CHECK(at<int32_t>(*sl, 0) == 3);
        CHECK(at<int32_t>(*sl, 1) == 2);
        REQUIRE(u.arrays.size() == 3);
        CHECK(u.arrays[2].name == "colors");
    }
    {
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "scene_clear_reset.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const SceneClear& c = std::get<SceneClear>(m);
        CHECK(c.reset);
        CHECK(c.state);
        CHECK_FALSE(c.recursive);
        CHECK_FALSE(c.is_static);
        CHECK_FALSE(c.parts.has_value());
        CHECK(c.path.empty());
        CHECK(c.view == "landscape");
        CHECK(c.seq == 41);
    }
    {
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "scene_clear.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const SceneClear& c = std::get<SceneClear>(m);
        CHECK_FALSE(c.reset);
        CHECK_FALSE(c.path.empty());
    }
    {   // the decode-only frame: a section no part has is left out
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "scene_update_points_future.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        REQUIRE(u.arrays.size() == 1);
        CHECK(u.array("future") == nullptr);
    }
    {   // a null stat decodes to NaN and encodes back to null
        std::vector<float> pos = {0, 0, 1, 1};
        Section p = Section::make("positions", CALIPER_NET_DTYPE_F32, {2, 2}, pos.data(), 16);
        const std::vector<uint8_t> bin = update_frame(std::string("{") + kBase + ",\"part\":\"points\",\"stats\":[null,1,2.5],\"vmin\":1}", {p});
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        const Json& st = *u.params.find("stats");
        CHECK(std::isnan(st.at(0).as_double()));
        CHECK(st.at(2).as_double() == 2.5);
        std::vector<uint8_t> again;
        REQUIRE(encode(m, &again, nullptr, CALIPER_NET_MINOR_3_2));
        Message m2;
        REQUIRE(decode_bytes(again, &m2).ok());
        Frame f;
        FrameError fe;
        REQUIRE(decode_frame(again.data(), again.size(), &f, &fe));
        CHECK(f.meta.find("stats")->at(0).is_null());
        CHECK(f.meta.find("vmin")->as_double() == 1.0);
    }
}

TEST_CASE("net/wire 3.2: the other parts, stamps, entity transforms and class tables carry their golden values") {
    auto load = [](const char* name, Message* m, std::vector<uint8_t>* keep) {
        *keep = read_bytes(v32_dir() / (std::string(name) + ".bin"));
        const DecodeResult r = decode_bytes(*keep, m);
        REQUIRE_MESSAGE(r.ok(), name << ": " << r.error_code << ": " << r.message);
    };
    std::vector<uint8_t> b;
    Message m;
    load("scene_update_points_stamped", &m, &b);
    {
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        REQUIRE(u.stamp.has_value());
        CHECK(u.stamp->shape == std::vector<int64_t>{2});
        CHECK(at<int64_t>(*u.stamp, 0) == 1200);
        CHECK(at<int64_t>(*u.stamp, 1) == 1763200000123456789LL);
        CHECK(u.array("stamp") == nullptr);   // the stamp is not one of the part's arrays
    }
    load("scene_update_lines_edges", &m, &b);
    {
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        CHECK(u.params.find("mode")->as_string() == "edges");
        REQUIRE(u.array("indices") != nullptr);
        CHECK(u.array("indices")->shape == std::vector<int64_t>{3, 2});
        CHECK(u.array("values")->shape == std::vector<int64_t>{3});
    }
    load("scene_update_boxes", &m, &b);
    {
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        CHECK(u.is_static);
        CHECK_FALSE(u.stamp.has_value());
        CHECK(u.array("rotations") != nullptr);
        CHECK(u.params.find("labels")->size() == 1);
    }
    load("scene_update_image_depth", &m, &b);
    {
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        const Section* p = u.array("pixels");
        REQUIRE(p != nullptr);
        CHECK(p->dtype == CALIPER_NET_DTYPE_U16);
        CHECK(at<uint16_t>(*p, 5) == 2500);
        CHECK(u.params.find("scale")->as_double() == 0.001);
        CHECK(u.params.find("cloud")->as_bool());
    }
    load("scene_update_camera", &m, &b);
    {
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        CHECK(u.arrays.empty());
        CHECK(u.params.find("width")->is_int());
        CHECK(u.params.find("fx")->as_double() == 525.0);
        CHECK(u.params.find("convention")->as_string() == "rdf");
    }
    load("scene_transforms_stamps", &m, &b);
    {
        const SceneTransforms& t = std::get<SceneTransforms>(m);
        CHECK(t.paths == std::vector<std::string>{"a", "b"});
        CHECK(t.matrices.shape == std::vector<int64_t>{2, 16});
        CHECK(at<float>(t.matrices, 12) == 1.0f);   // column-major: the translation is in elements 12 to 14
        REQUIRE(t.stamps.has_value());
        CHECK_FALSE(t.stamp.has_value());
        CHECK(at<int64_t>(*t.stamps, 2) == 1201);
    }
    load("scene_transforms_static", &m, &b);
    CHECK(std::get<SceneTransforms>(m).is_static);
    CHECK(std::get<SceneTransforms>(m).state);
    load("scene_classes", &m, &b);
    {
        const SceneClasses& k = std::get<SceneClasses>(m);
        REQUIRE(k.classes.size() == 3);
        CHECK(k.classes.at(2).find("id")->as_double() == 65535);
        CHECK(k.is_static);
    }
    load("scene_clear_stamped", &m, &b);
    {
        const SceneClear& c = std::get<SceneClear>(m);
        REQUIRE(c.stamp.has_value());
        REQUIRE(c.parts.has_value());
        CHECK((*c.parts)[0] == "transform");
    }
    CHECK(scene_part_index("camera") == 6);
    CHECK(scene_part_index("blob") == -1);
    CHECK(std::string(scene_part_sections(scene_part_index("mesh"))[6]) == "texture");
}

TEST_CASE("net/wire 3.2: timelines, text logs, bar charts and the recording messages carry their golden values") {
    std::vector<uint8_t> b;
    Message m;
    auto load = [&](const char* name) {
        b = read_bytes(v32_dir() / (std::string(name) + ".bin"));
        const DecodeResult r = decode_bytes(b, &m);
        REQUIRE_MESSAGE(r.ok(), name << ": " << r.error_code << ": " << r.message);
    };
    load("views_timelines");
    REQUIRE(std::get<Views>(m).timelines.size() == 3);
    CHECK(std::get<Views>(m).timelines.at(2).find("name")->as_string() == "episode");
    load("series_stamps");
    {
        const Series& s = std::get<Series>(m);
        REQUIRE(s.stamps.has_value());
        CHECK(s.stamps->shape == std::vector<int64_t>{3, 2});
        CHECK(at<int64_t>(*s.stamps, 5) == 1763200000133456789LL);
    }
    load("series_decimated");
    CHECK(std::get<Series>(m).decimated == uint64_t(1024));
    CHECK(at<int64_t>(*std::get<Series>(m).indices, 2) == 4095);
    load("snapshot_empty");
    CHECK(std::get<Snapshot>(m).empty);
    CHECK(std::get<Snapshot>(m).cursor == uint32_t(3));
    load("snapshot_bars");
    {
        const Snapshot& s = std::get<Snapshot>(m);
        CHECK(s.kind == CALIPER_NET_KIND_BARS);
        CHECK(s.horizontal);
        CHECK(s.labels.size() == 3);
        REQUIRE(s.colors.has_value());
        CHECK(s.colors->shape == std::vector<int64_t>{3, 4});
    }
    load("snapshot_static");
    CHECK(std::get<Snapshot>(m).is_static);
    load("controls_stamped");
    CHECK(std::get<Controls>(m).stamp.has_value());
    load("text");
    {
        const Text& t = std::get<Text>(m);
        CHECK(t.line(0) == "epoch 1 starts");
        CHECK(t.line(1) == "loss rose\nto 2.5");
        CHECK(t.line(2) == "pas de donn\xc3\xa9" "es \xc3\xa9");
        CHECK(t.levels.data[2] == CALIPER_NET_LEVEL_ERROR);
    }
    load("cursor_play");
    {
        const Cursor& c = std::get<Cursor>(m);
        CHECK(c.op == "play");
        CHECK(*c.speed == 2.0);
        CHECK(c.loop.at(1).as_string() == "5000");
    }
    load("time_state_following");
    CHECK_FALSE(std::get<TimeState>(m).cursor.has_value());
    load("recordings");
    CHECK(std::get<Recordings>(m).recordings.size() == 1);
    load("sessions_recording");
    CHECK(std::get<Sessions>(m).list.size() == 1);
}

TEST_CASE("net/wire 3.2: stamp text, and a cursor keeps the fields of its op") {
    for (const char* ok : {"0", "-1", "1763200000123456789", "9223372036854775807", "-9223372036854775808", "007"})
        CHECK_MESSAGE(stamp_text_ok(ok), ok);
    for (const char* no : {"", "-", "+1", "1.5", "1e3", " 1", "9223372036854775808", "-9223372036854775809",
                           "111111111111111111111"})
        CHECK_MESSAGE(!stamp_text_ok(no), no);
    Json meta;
    std::string err;
    const char* text = "{\"request_id\":1,\"op\":\"pause\",\"timeline\":\"step\",\"value\":\"5\",\"count\":3}";
    REQUIRE(Json::parse(text, std::strlen(text), &meta, &err));
    std::vector<uint8_t> b;
    REQUIRE(encode_frame(CALIPER_NET_MSG_CURSOR, meta, {}, &b));
    Message m;
    REQUIRE(decode_bytes(b, &m).ok());
    const Cursor& c = std::get<Cursor>(m);
    CHECK_FALSE(c.timeline.has_value());
    CHECK_FALSE(c.count.has_value());
    std::vector<uint8_t> again;
    REQUIRE(encode(m, &again));
    Frame f;
    FrameError fe;
    REQUIRE(decode_frame(again.data(), again.size(), &f, &fe));
    CHECK(f.meta.dump() == "{\"op\":\"pause\",\"request_id\":1}");
}

TEST_CASE("net/wire 3.2: the pieces of a split update are put together and checked") {
    std::vector<std::vector<uint8_t>> frames;
    std::vector<SceneUpdate> pieces;
    for (int k = 0; k < 5; ++k) {
        frames.push_back(read_bytes(v32_dir() / ("scene_update_split_" + std::to_string(k) + ".bin")));
    }
    for (const auto& fr : frames) {
        Message m;
        REQUIRE(decode_bytes(fr, &m).ok());
        pieces.push_back(std::get<SceneUpdate>(m));
        CHECK_FALSE(pieces.back().split.is_null());
    }
    CHECK(pieces[0].params.find("radius") != nullptr);
    CHECK(pieces[1].params.size() == 0);
    SplitUpdates su;
    WholeUpdate whole;
    std::string err;
    for (int k = 0; k < 4; ++k) CHECK(su.add(pieces[(size_t)k], &whole, &err) == SplitUpdates::Result::Pending);
    REQUIRE_MESSAGE(su.add(pieces[4], &whole, &err) == SplitUpdates::Result::Whole, err);
    CHECK(su.held() == 0);
    const SceneUpdate& u = whole.update;
    CHECK(u.split.is_null());
    CHECK(u.path == "world/split");
    REQUIRE(u.array("positions") != nullptr);
    CHECK(u.array("positions")->shape == std::vector<int64_t>{20, 3});
    for (int i = 0; i < 60; ++i) CHECK(at<float>(*u.array("positions"), (size_t)i) == (float)i / 8.0f);
    REQUIRE(u.array("colors") != nullptr);
    for (int i = 0; i < 80; ++i) CHECK(at<uint8_t>(*u.array("colors"), (size_t)i) == (uint8_t)(i * 3 % 256));
    REQUIRE(u.stamp.has_value());
    CHECK(at<int64_t>(*u.stamp, 0) == 1200);
    CHECK(u.params.find("radius")->as_double() == 0.25);
    // The whole update encodes as an unsplit one: a single frame.
    std::vector<uint8_t> enc;
    REQUIRE(encode(Message(u), &enc));
    Message back;
    REQUIRE(decode_bytes(enc, &back).ok());
    CHECK(std::get<SceneUpdate>(back).array("colors")->shape == std::vector<int64_t>{20, 4});

    // A piece without its piece 0 is dropped; a piece twice is refused.
    SplitUpdates a;
    CHECK(a.add(pieces[1], &whole, &err) == SplitUpdates::Result::Dropped);
    CHECK(a.dropped() == 1);
    CHECK(a.add(pieces[0], &whole, &err) == SplitUpdates::Result::Pending);
    CHECK(a.add(pieces[1], &whole, &err) == SplitUpdates::Result::Pending);
    CHECK(a.add(pieces[1], &whole, &err) == SplitUpdates::Result::Refused);
    CHECK(a.held() == 0);
    // A newer update of the same entity part discards the incomplete one; at most two are held.
    SplitUpdates c;
    CHECK(c.add(pieces[0], &whole, &err) == SplitUpdates::Result::Pending);
    SceneUpdate newer = pieces[0];
    newer.split.set("update", (uint64_t)5);
    CHECK(c.add(newer, &whole, &err) == SplitUpdates::Result::Pending);
    CHECK(c.held() == 1);
    CHECK(c.discarded() == 1);
    CHECK(c.add(pieces[1], &whole, &err) == SplitUpdates::Result::Dropped);
    SplitUpdates d;
    for (int k = 0; k < 3; ++k) {
        SceneUpdate p = pieces[0];
        p.path = std::string(1, char('a' + k));
        p.split.set("update", (uint64_t)(10 + k));
        CHECK(d.add(p, &whole, &err) == SplitUpdates::Result::Pending);
    }
    CHECK(d.held() == 2);
    // A piece whose seq differs from piece 0 is refused.
    SplitUpdates e;
    CHECK(e.add(pieces[0], &whole, &err) == SplitUpdates::Result::Pending);
    SceneUpdate odd = pieces[1];
    odd.seq = 38;
    CHECK(e.add(odd, &whole, &err) == SplitUpdates::Result::Refused);
}

TEST_CASE("net/wire 3.2: a split piece whose rows pass the update's rows is refused before any copy") {
    std::vector<std::vector<uint8_t>> frames;
    std::vector<SceneUpdate> pieces;
    for (int k = 0; k < 2; ++k)
        frames.push_back(read_bytes(v32_dir() / ("scene_update_split_" + std::to_string(k) + ".bin")));
    for (const auto& fr : frames) {
        Message m;
        REQUIRE(decode_bytes(fr, &m).ok());
        pieces.push_back(std::get<SceneUpdate>(m));
    }
    REQUIRE_FALSE(pieces[1].arrays.empty());
    const std::string name = pieces[1].arrays[0].name;
    const uint64_t rows = 20, n = (uint64_t)pieces[1].arrays[0].shape[0];
    REQUIRE(n >= 1);
    // A first row of 2^64 - 1 was -1 as int64_t and wrote before the buffer; 2^64 - n wraps first + n
    // to 0; rows itself and rows + 1 - n pass the end by n rows and by one row.
    for (const uint64_t f : {UINT64_MAX, UINT64_MAX - n + 1, rows, rows + 1 - n}) {
        CAPTURE(f);
        SceneUpdate p = pieces[1];
        Json first = *p.split.find("first");
        first.set(name, f);
        p.split.set("first", first);
        SplitUpdates su;
        WholeUpdate whole;
        std::string err;
        CHECK(su.add(pieces[0], &whole, &err) == SplitUpdates::Result::Pending);
        CHECK(su.add(p, &whole, &err) == SplitUpdates::Result::Refused);
        CHECK(err.find("pass its rows") != std::string::npos);
        CHECK(su.held() == 0);
        CHECK(su.discarded() == 1);
    }
    // A piece whose bytes disagree with its shape, and one with a negative row count, are refused too.
    for (int k = 0; k < 2; ++k) {
        SceneUpdate p = pieces[1];
        if (k == 0) p.arrays[0].size -= 1;
        else p.arrays[0].shape[0] = -1;
        SplitUpdates su;
        WholeUpdate whole;
        std::string err;
        CHECK(su.add(pieces[0], &whole, &err) == SplitUpdates::Result::Pending);
        CHECK(su.add(p, &whole, &err) == SplitUpdates::Result::Refused);
        CHECK(err.find(k == 0 ? "disagree with its shape" : "has no rows") != std::string::npos);
    }
    // The last rows of the update still fit: first = rows - n is accepted.
    SceneUpdate last = pieces[1];
    Json first = *last.split.find("first");
    first.set(name, rows - n);
    last.split.set("first", first);
    SplitUpdates su;
    WholeUpdate whole;
    std::string err;
    CHECK(su.add(pieces[0], &whole, &err) == SplitUpdates::Result::Pending);
    CHECK(su.add(last, &whole, &err) == SplitUpdates::Result::Pending);
}

TEST_CASE("net/wire 3.2: every invalid golden frame is refused as expect.json says") {
    const Json expect = read_json(v32_dir() / "invalid" / "expect.json");
    REQUIRE(expect.size() >= 93);
    size_t frames = 0, messages = 0;
    for (const auto& kv : expect.members()) {
        CAPTURE(kv.first);
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "invalid" / kv.first);
        REQUIRE(!bin.empty());
        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        if (const Json* e = kv.second.find("error")) {
            ++messages;
            CHECK(r.status == DecodeResult::Status::BadMessage);
            CHECK(r.error_code == e->as_string());
        } else {
            ++frames;
            CHECK(r.status == DecodeResult::Status::BadFrame);
            CHECK(r.close_code == (int)ju(kv.second, "close_code"));
            CHECK(r.reason == kv.second.find("reason")->as_string());
        }
    }
    CHECK(frames == 3);
    CHECK(messages >= 90);
}

TEST_CASE("net/wire 3.2: every truncation of a scene_update frame is refused") {
    const std::vector<uint8_t> bin = read_bytes(v32_dir() / "scene_update_mesh.bin");
    for (size_t n = 0; n < bin.size(); ++n) {
        CAPTURE(n);
        std::unique_ptr<uint8_t[]> exact(new uint8_t[n ? n : 1]);
        if (n) std::memcpy(exact.get(), bin.data(), n);
        Message m;
        const DecodeResult r = decode(exact.get(), n, &m);
        CHECK(r.status == DecodeResult::Status::BadFrame);
        CHECK(r.close_code == 4000);
    }
}

TEST_CASE("net/wire 3.2: the limits of a batch") {
    std::string err;
    {   // 4194305 2-D points
        std::vector<float> pos((size_t)(CALIPER_NET_MAX_BATCH_POINTS + 1) * 2, 0.f);
        Section p = Section::make("positions", CALIPER_NET_DTYPE_F32, {(int64_t)CALIPER_NET_MAX_BATCH_POINTS + 1, 2},
                                  pos.data(), pos.size() * 4);
        const std::vector<uint8_t> bin = update_frame(std::string("{") + kBase + ",\"part\":\"points\"}", {p});
        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        CHECK(r.status == DecodeResult::Status::BadMessage);
        CHECK(r.error_code == "bad_request");
        // the limit itself is fine
        pos.resize((size_t)CALIPER_NET_MAX_BATCH_POINTS * 2);
        p = Section::make("positions", CALIPER_NET_DTYPE_F32, {(int64_t)CALIPER_NET_MAX_BATCH_POINTS, 2}, pos.data(),
                          pos.size() * 4);
        const std::vector<uint8_t> okb = update_frame(std::string("{") + kBase + ",\"part\":\"points\"}", {p});
        CHECK(decode_bytes(okb, &m).ok());
    }
    {   // 1048577 strips
        std::vector<float> pos = {0, 0};
        std::vector<int32_t> lens((size_t)CALIPER_NET_MAX_BATCH_ITEMS + 1, 1);
        Section p = Section::make("positions", CALIPER_NET_DTYPE_F32, {1, 2}, pos.data(), 8);
        Section s = Section::make("strip_lengths", CALIPER_NET_DTYPE_I32, {(int64_t)lens.size()}, lens.data(), lens.size() * 4);
        const std::vector<uint8_t> bin =
            update_frame(std::string("{") + kBase + ",\"part\":\"lines\",\"mode\":\"strips\"}", {p, s});
        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        CHECK(r.status == DecodeResult::Status::BadMessage);
        CHECK(r.error_code == "bad_request");
        CHECK(r.message.find("strips") != std::string::npos);
    }
    {   // 4194305 mesh triangles
        std::vector<float> pos = {0, 0, 1, 0, 0, 1};
        std::vector<int32_t> idx((size_t)(CALIPER_NET_MAX_BATCH_POINTS + 1) * 3, 0);
        Section p = Section::make("positions", CALIPER_NET_DTYPE_F32, {3, 2}, pos.data(), 24);
        Section s = Section::make("indices", CALIPER_NET_DTYPE_I32, {(int64_t)CALIPER_NET_MAX_BATCH_POINTS + 1, 3},
                                  idx.data(), idx.size() * 4);
        const std::vector<uint8_t> bin = update_frame(std::string("{") + kBase + ",\"part\":\"mesh\"}", {p, s});
        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        CHECK(r.status == DecodeResult::Status::BadMessage);
        CHECK(r.error_code == "bad_request");
        CHECK(r.message.find("triangles") != std::string::npos);
    }
}

TEST_CASE("net/wire 3.2: entity paths") {
    std::vector<std::string> good = {"", "a", "world/cloud", "a.b-c_d/E9", "x", std::string(63, 'x')};
    {
        std::string s16;
        for (int i = 0; i < 16; ++i) s16 += (i ? "/s" : "s");
        good.push_back(s16);
        std::string y;
        for (int i = 0; i < 4; ++i) y += (i ? "/" : "") + std::string(63, 'y');
        good.push_back(y.substr(0, 255));
        CHECK(good.back().size() == 255);
    }
    std::vector<std::string> bad = {"/", "a/", "/a", "a//b", ".", "a/..", "a/./b", "a b", "\xc3\xa9", std::string(64, 'x'),
                                    std::string("a\nb"), std::string("a\0b", 3)};
    {
        std::string s17;
        for (int i = 0; i < 17; ++i) s17 += (i ? "/s" : "s");
        bad.push_back(s17);
        std::string y;
        for (int i = 0; i < 5; ++i) y += (i ? "/" : "") + std::string(63, 'y');
        bad.push_back(y);
        bad.push_back(good[good.size() - 1] + "z");   // 256 bytes
    }
    for (const std::string& p : good) CHECK_MESSAGE(entity_path_ok(p), p);
    for (const std::string& p : bad) CHECK_MESSAGE(!entity_path_ok(p), p);
}

TEST_CASE("net/wire 3.2: scene view info and the scene data view") {
    auto parse = [](const char* s) {
        Json j;
        std::string err;
        REQUIRE(Json::parse(s, std::strlen(s), &j, &err));
        return j;
    };
    SceneView v;
    std::string err;
    REQUIRE(scene_view_info(parse("{\"kind\":6}"), &v, &err));
    CHECK(v.space == "3d");
    CHECK(v.up == "+y");
    CHECK(v.units == "m");
    REQUIRE(scene_view_info(parse("{\"kind\":6,\"scene\":{\"space\":\"2d\"}}"), &v, &err));
    CHECK(v.up == "-y");
    CHECK(v.view_camera.is_null());
    REQUIRE(scene_view_info(parse("{\"kind\":6,\"scene\":{\"view_camera\":{\"azimuth\":30}}}"), &v, &err));
    CHECK(v.up == "+y");
    CHECK(v.view_camera == parse("{\"azimuth\":30}"));
    // the starting camera: kept when a viewer can read it, else null, and never in 2-D
    for (const char* vc : {"{}", "{\"target\":[0,1,2.5]}", "{\"distance\":3}", "{\"elevation\":-89}",
                           "{\"elevation\":89,\"azimuth\":-720.5,\"target\":[0,0,0],\"distance\":0.5}", "\"rig/left\""}) {
        CAPTURE(vc);
        REQUIRE(scene_view_info(parse((std::string("{\"kind\":6,\"scene\":{\"view_camera\":") + vc + "}}").c_str()),
                                &v, &err));
        CHECK(v.view_camera == parse(vc));
    }
    for (const char* vc : {"{\"target\":[0,1]}", "{\"distance\":0}", "{\"elevation\":90}", "{\"azimuth\":\"x\"}",
                           "{\"fov\":45}", "\"\"", "\"a/../b\"", "3", "[1,2,3]"}) {
        CAPTURE(vc);
        REQUIRE(scene_view_info(parse((std::string("{\"kind\":6,\"scene\":{\"view_camera\":") + vc + "}}").c_str()),
                                &v, &err));
        CHECK(v.view_camera.is_null());
    }
    REQUIRE(scene_view_info(parse("{\"kind\":6,\"scene\":{\"space\":\"2d\",\"view_camera\":{}}}"), &v, &err));
    CHECK(v.view_camera.is_null());
    REQUIRE(scene_view_info(parse("{\"kind\":6,\"scene\":{\"up\":\"+z\",\"units\":\"\"}}"), &v, &err));
    CHECK(v.up == "+z");
    CHECK(v.units.empty());
    for (const char* sc : {"{\"space\":\"4d\"}", "{\"space\":\"2d\",\"up\":\"+z\"}", "{\"up\":\"z\"}",
                           "{\"units\":\"xxxxxxxxxxxxxxxx\"}", "{\"units\":3}", "\"3d\""}) {
        CAPTURE(sc);
        err.clear();
        CHECK_FALSE(scene_view_info(parse((std::string("{\"kind\":6,\"scene\":") + sc + "}").c_str()), &v, &err));
        CHECK_FALSE(err.empty());
    }
    // the golden VIEWS with a starting camera and a 2-D scene
    {
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "views_scene_camera.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const Views& vs = std::get<Views>(m);
        REQUIRE(scene_view_info(vs.views.at(0), &v, &err));
        CHECK(v.view_camera == parse("{\"target\":[0.0,0.0,0.5],\"azimuth\":30.0,\"elevation\":25.0}"));
        REQUIRE(scene_view_info(vs.views.at(1), &v, &err));
        CHECK(v.space == "2d");
        CHECK(v.units == "px");
        CHECK(v.view_camera.is_null());
    }
    // the views entries of the golden VIEWS frame
    {
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "views_scene.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const Views& vs = std::get<Views>(m);
        bool seen = false;
        for (const Json& e : vs.views.items())
            if (e.find("kind") && e.find("kind")->as_double() == CALIPER_NET_KIND_SCENE) {
                seen = true;
                CHECK(scene_view_info(e, &v, &err));
            }
        CHECK(seen);
    }
    // A null key means the same as an absent one: the golden VIEWS whose scene keys are all null.
    {
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "views_scene_nulls.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const Views& vs = std::get<Views>(m);
        REQUIRE(vs.views.size() == 1);
        v = SceneView{};
        err.clear();
        CHECK_MESSAGE(scene_view_info(vs.views.at(0), &v, &err), err);
        CHECK(v.space == "3d");
        CHECK(v.up == "+y");
        CHECK(v.units == "m");
    }

    const Json dv = scene_data_view();
    CHECK(dv.dump() ==
          "{\"budget\":{\"max_extent\":1024,\"max_items\":100000,\"max_points\":100000,\"max_vertices\":1000000},"
          "\"entities\":{\"exclude\":[],\"include\":[\"**\"]}}");
    const Json view = parse("{\"id\":1,\"name\":\"s\",\"kind\":6,\"sources\":[]}");
    Json out;
    for (const Json& req : {Json(), Json::object(), dv}) {
        out = Json();
        REQUIRE(resolve_scene_data_view(view, req, &out, &err));
        CHECK(out == dv);
    }
    for (const char* req : {"{\"budget\":{\"max_points\":10}}", "{\"entities\":{\"include\":[\"a/**\"]}}",
                            "{\"source\":\"data\"}", "[1]"}) {
        CAPTURE(req);
        CHECK_FALSE(resolve_scene_data_view(view, parse(req), &out, &err));
    }
    // a view's own default: absent, {} and the resolved default are fine, anything else is not built
    Json with_default = view;
    with_default.set("default", dv);
    CHECK(resolve_scene_data_view(with_default, Json(), &out, &err));
    with_default.set("default", parse("{\"budget\":{\"max_points\":10}}"));
    CHECK_FALSE(resolve_scene_data_view(with_default, Json(), &out, &err));
}

TEST_CASE("net/wire 3.2: a forwarder rewrites sub and keeps the original sections") {
    for (const std::string& name : v32_names()) {
        if (name.rfind("scene_update", 0) != 0) continue;   // a frame without sections has no tail to share
        CAPTURE(name);
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / (name + ".bin"));
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const std::vector<Section>& arrays = std::get<SceneUpdate>(m).arrays;
        Frame f;
        FrameError fe;
        REQUIRE(decode_frame(bin.data(), bin.size(), &f, &fe));
        if (f.sections.empty()) continue;   // a camera: no tail to share
        for (uint64_t sub : {uint64_t(0), uint64_t(7), uint64_t(4294967295u)}) {
            Json meta = f.meta;
            meta.set("sub", sub);
            std::vector<uint32_t> lens;
            for (const auto& s : f.sections) lens.push_back((uint32_t)s.size);
            const size_t tail = f.sections.front().offset;
            std::vector<uint8_t> head;
            REQUIRE(encode_head(f.type, meta, lens, bin.size() - tail, &head, nullptr, f.minor, f.flags));
            CHECK(head.size() % 8 == 0);
            std::vector<uint8_t> whole = head;
            whole.insert(whole.end(), bin.begin() + (std::ptrdiff_t)tail, bin.end());
            Message m2;
            const DecodeResult r = decode_bytes(whole, &m2);
            REQUIRE_MESSAGE(r.ok(), r.error_code << ": " << r.message);
            const SceneUpdate& u2 = std::get<SceneUpdate>(m2);
            CHECK(u2.sub == sub);
            REQUIRE(u2.arrays.size() == arrays.size());
            for (size_t i = 0; i < arrays.size(); ++i) {
                CHECK(u2.arrays[i].name == arrays[i].name);
                CHECK(u2.arrays[i].shape == arrays[i].shape);
                REQUIRE(u2.arrays[i].size == arrays[i].size);
                CHECK(std::memcmp(u2.arrays[i].data, arrays[i].data, arrays[i].size) == 0);
            }
            if (sub == ju(f.meta, "sub")) CHECK(whole == bin);
        }
    }
}

#ifndef CALIPER_NET_FUZZ_ITERATIONS
#define CALIPER_NET_FUZZ_ITERATIONS 20000   // the sanitizer build raises it
#endif

TEST_CASE("net/wire 3.2: truncations and bit flips of the 3.2 frames never crash") {
    std::vector<std::vector<uint8_t>> frames;
    for (const std::string& n : v32_names()) frames.push_back(read_bytes(v32_dir() / (n + ".bin")));
    std::mt19937 g(32032);
    size_t ok = 0, bad_frame = 0, bad_msg = 0;
    const int iterations = CALIPER_NET_FUZZ_ITERATIONS;
    for (int it = 0; it < iterations; ++it) {
        std::vector<uint8_t> b = frames[(size_t)it % frames.size()];
        switch (g() % 4) {
            case 0: b.resize(g() % (b.size() + 1)); break;
            case 1:
                for (int k = 0; k < 1 + (int)(g() % 4); ++k) b[g() % b.size()] ^= uint8_t(1u << (g() % 8));
                break;
            case 2: b[g() % b.size()] = uint8_t(g()); break;
            case 3: {
                const size_t lim = std::min<size_t>(b.size(), 200);
                b[g() % lim] ^= uint8_t(1u << (g() % 8));
                break;
            }
        }
        std::unique_ptr<uint8_t[]> exact(new uint8_t[b.size() ? b.size() : 1]);
        if (!b.empty()) std::memcpy(exact.get(), b.data(), b.size());
        Message m;
        const DecodeResult r = decode(exact.get(), b.size(), &m);
        if (r.ok()) {
            ++ok;
            std::vector<uint8_t> again;
            encode(m, &again, nullptr, CALIPER_NET_MINOR_3_2);   // whatever decoded must encode
        } else if (r.status == DecodeResult::Status::BadFrame) {
            ++bad_frame;
            CHECK((r.close_code == 4000 || r.close_code == 4001 || r.close_code == 1009));
        } else {
            ++bad_msg;
            CHECK((r.error_code == "bad_request" || r.error_code == "unsupported" || r.error_code == "bad_cursor"));
        }
    }
    MESSAGE(iterations << " mutated frames: " << ok << " decoded, " << bad_frame << " refused as frames, " << bad_msg
                       << " refused as messages");
    CHECK(ok + bad_frame + bad_msg == (size_t)iterations);
}

// ---- the hub and the capability "resources" ------------------------------------------------------------
#if !defined(_WIN32)
namespace {
namespace hubres {

const std::string kToken = "hub_resource_token-0123456789abcdef";

struct Runner {
    std::unique_ptr<caliper_net::Hub> hub;
    std::thread thread;
    Runner() {
        caliper_net::HubOptions o;
        o.token = kToken;
        o.allow_control = true;
        hub = std::make_unique<caliper_net::Hub>(std::move(o));
        std::string err;
        REQUIRE_MESSAGE(hub->listen(&err), err);
        thread = std::thread([this] { hub->run(); });
    }
    ~Runner() {
        hub->stop();
        thread.join();
    }
    std::string url() const { return "ws://127.0.0.1:" + std::to_string(hub->port()) + "/v3/stream?token=" + kToken; }
};

// A peer that keeps each decoded message with the frame its sections point into.
struct Peer {
    caliper_net::Client client;
    std::vector<std::string> caps;
    bool producer = false;
    bool open = false, welcomed = false;
    std::deque<std::pair<std::shared_ptr<caliper_net::Bytes>, Message>> inbox;

    void send(const Message& m) {
        std::vector<uint8_t> b;
        REQUIRE(encode(m, &b, nullptr, CALIPER_NET_MINOR_3_2));
        client.send(b.data(), b.size());
    }
    void poll() {
        for (auto& e : client.poll(0)) {
            if (e.type == caliper_net::Event::Type::Open) {
                open = true;
                Hello h;
                h.role = producer ? CALIPER_NET_ROLE_PRODUCER : CALIPER_NET_ROLE_VIEWER;
                h.capabilities = caps;
                if (producer) {
                    Identity id;
                    id.id = "clp-res";
                    id.script = "res.py";
                    id.pid = 9;
                    id.start_unix_ms = 1763200000000ull;
                    h.session = id.to_json();
                }
                send(h);
            } else if (e.type == caliper_net::Event::Type::Message) {
                auto buf = std::make_shared<caliper_net::Bytes>(std::move(e.data));
                Message m;
                if (!decode(buf->data(), buf->size(), &m).ok()) continue;
                if (auto* pi = std::get_if<Ping>(&m)) send(Pong{pi->nonce});
                if (producer && std::holds_alternative<Welcome>(m)) {
                    welcomed = true;
                    send(Views{});
                }
                inbox.emplace_back(buf, std::move(m));
            }
        }
    }
    template <typename T>
    std::vector<const T*> all() const {
        std::vector<const T*> v;
        for (const auto& x : inbox)
            if (auto* p = std::get_if<T>(&x.second)) v.push_back(p);
        return v;
    }
};

struct World {
    std::vector<std::unique_ptr<Peer>> peers;
    bool until(const std::function<bool()>& done, int ms = 5000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            for (auto& p : peers) p->poll();
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        return done();
    }
    // Polls until the hub has sent everything it holds for each viewer: a PING answered by its PONG
    // comes after every message queued before it on the same connection.
    Peer& add(const Runner& h, std::vector<std::string> caps, bool producer) {
        peers.push_back(std::make_unique<Peer>());
        Peer& p = *peers.back();
        p.caps = std::move(caps);
        p.producer = producer;
        std::string err;
        REQUIRE_MESSAGE(p.client.connect(h.url(), &err), err);
        if (producer) REQUIRE(until([&] { return p.welcomed; }));
        else REQUIRE(until([&] { return !p.all<Views>().empty() && p.all<Views>().back()->session; }));
        return p;
    }
    void settle(Peer& v) {
        const uint32_t nonce = 4242 + (uint32_t)v.inbox.size();
        v.send(Ping{nonce});
        REQUIRE(until([&] {
            for (auto* x : v.all<Pong>())
                if (x->nonce == nonce) return true;
            return false;
        }));
    }
};

Message golden(const char* name) {
    static std::vector<std::vector<uint8_t>> keep;
    keep.push_back(read_bytes(v32_dir() / (std::string(name) + ".bin")));
    Message m;
    REQUIRE(decode_bytes(keep.back(), &m).ok());
    return m;
}

Resource heads_at(uint64_t step, const std::vector<float>& values) {
    Resource r = std::get<Resource>(golden("resource_buffer"));
    static std::deque<std::vector<float>> keep;
    keep.push_back(values);
    r.step = step;
    r.count = (uint32_t)values.size();
    r.data = Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8, {(int64_t)(values.size() * 4)},
                           keep.back().data(), values.size() * 4);
    return r;
}

const Error* error_for(const Peer& v, uint32_t request_id) {
    for (auto* e : v.all<Error>())
        if (e->in_reply_to && *e->in_reply_to == CALIPER_NET_MSG_PROGRAM && e->request_id && *e->request_id == request_id)
            return e;
    return nullptr;
}

}  // namespace hubres
}  // namespace

TEST_CASE("net/hub 3.2: draw halves and resources reach the viewers that announced resources, and a late viewer gets the newest") {
    using namespace hubres;
    const std::vector<std::string> caps = {CALIPER_NET_CAP_PROGRAMS, CALIPER_NET_CAP_RESOURCES};
    Runner h;
    World w;
    Peer& p = w.add(h, caps, true);
    Peer& a = w.add(h, caps, false);
    Peer& b = w.add(h, {CALIPER_NET_CAP_PROGRAMS}, false);
    const auto welcome = a.all<Welcome>();
    REQUIRE(!welcome.empty());
    CHECK(std::count(welcome[0]->capabilities.begin(), welcome[0]->capabilities.end(), CALIPER_NET_CAP_RESOURCES) == 1);

    p.send(golden("program_set_piece_draw"));
    p.send(heads_at(1200, {1, 2, 3, 4}));
    REQUIRE(w.until([&] { return !a.all<Resource>().empty(); }));
    p.send(heads_at(1210, {5, 6}));
    REQUIRE(w.until([&] { return a.all<Resource>().back()->step == 1210; }));
    const auto relays = a.all<Program>();
    REQUIRE(relays.size() == 1);
    CHECK(relays[0]->request_id == 0);
    CHECK(relays[0]->draw);
    CHECK_FALSE(relays[0]->derive);
    const Resource* last = a.all<Resource>().back();
    CHECK(last->count == 2);
    REQUIRE(last->data.size == 8);
    float second = 0;
    std::memcpy(&second, last->data.data + 4, 4);
    CHECK(second == 6.0f);
    w.settle(b);
    CHECK(b.all<Resource>().empty());
    CHECK(b.all<Program>().empty());

    // A late viewer with resources gets the draw half and only the newest version; one without, neither.
    Peer& late = w.add(h, caps, false);
    Peer& late_none = w.add(h, {}, false);
    REQUIRE(w.until([&] { return !late.all<Resource>().empty(); }));
    w.settle(late);
    REQUIRE(late.all<Program>().size() == 1);
    REQUIRE(late.all<Resource>().size() == 1);
    CHECK(late.all<Resource>()[0]->step == 1210);
    w.settle(late_none);
    CHECK(late_none.all<Resource>().empty());

    // The same piece relayed again (the script's next attachment) keeps the newest version for a late viewer.
    p.send(golden("program_set_piece_draw"));
    REQUIRE(w.until([&] { return a.all<Program>().size() == 2; }));
    Peer& again = w.add(h, caps, false);
    REQUIRE(w.until([&] { return !again.all<Resource>().empty(); }));
    CHECK(again.all<Resource>()[0]->step == 1210);
    CHECK(again.all<Program>().size() == 1);

    // A removal reaches the viewers and drops what the hub keeps of the program.
    Program rm = std::get<Program>(golden("program_remove_piece"));
    p.send(rm);
    REQUIRE(w.until([&] { return a.all<Program>().size() == 3 && late.all<Program>().size() == 3; }));
    CHECK(a.all<Program>().back()->op == CALIPER_NET_PROGRAM_REMOVE);
    Peer& after = w.add(h, caps, false);
    w.settle(after);
    CHECK(after.all<Program>().empty());
    CHECK(after.all<Resource>().empty());
    // The hub publishes its counters at most every 20 ms: wait for them.
    CHECK(w.until([&] {
        const caliper_net::HubStats st = h.hub->stats();
        return st.resources_in == 2 && st.program_relays_in == 3 && st.retained_resource_bytes == 0;
    }));
}

TEST_CASE("net/hub 3.2: a producer without resources has its RESOURCE and relays ignored") {
    using namespace hubres;
    const std::vector<std::string> caps = {CALIPER_NET_CAP_PROGRAMS, CALIPER_NET_CAP_RESOURCES};
    Runner h;
    World w;
    Peer& p = w.add(h, {CALIPER_NET_CAP_PROGRAMS}, true);
    Peer& a = w.add(h, caps, false);
    p.send(golden("program_set_piece_draw"));
    p.send(heads_at(5, {1}));
    w.settle(p);   // the hub read both before it answered the producer's PING
    w.settle(a);
    CHECK(h.hub->stats().resources_in == 0);
    CHECK(h.hub->stats().program_relays_in == 0);
    CHECK(a.all<Resource>().empty());
    CHECK(a.all<Program>().empty());
}

TEST_CASE("net/hub 3.2: an instance of a piece goes to a producer with resources, and is refused without") {
    using namespace hubres;
    const std::vector<std::string> caps = {CALIPER_NET_CAP_PROGRAMS, CALIPER_NET_CAP_RESOURCES};
    {
        Runner h;
        World w;
        Peer& p = w.add(h, {CALIPER_NET_CAP_PROGRAMS}, true);
        Peer& v = w.add(h, caps, false);
        Program inst = std::get<Program>(golden("program_set_piece"));
        inst.request_id = 41;
        v.send(inst);
        REQUIRE(w.until([&] { return error_for(v, 41) != nullptr; }));
        CHECK(error_for(v, 41)->code == CALIPER_NET_ERR_UNSUPPORTED);
        CHECK(p.all<Program>().empty());
    }
    {
        Runner h;
        World w;
        Peer& p = w.add(h, caps, true);
        Peer& plain = w.add(h, {CALIPER_NET_CAP_PROGRAMS}, false);
        Peer& v = w.add(h, caps, false);
        Program inst = std::get<Program>(golden("program_set_piece"));
        inst.request_id = 42;
        plain.send(inst);
        REQUIRE(w.until([&] { return error_for(plain, 42) != nullptr; }));
        CHECK(error_for(plain, 42)->code == CALIPER_NET_ERR_UNSUPPORTED);
        inst.request_id = 43;
        v.send(inst);
        REQUIRE(w.until([&] { return !p.all<Program>().empty(); }));
        const Program* got = p.all<Program>()[0];
        CHECK(got->request_id != 0);
        CHECK(got->piece.find("name")->as_string() == "head_bars");
        CHECK(got->derive);
        CHECK(got->draw);
        CHECK(got->common);
    }
}

// ---- the app's program sync and instances of pieces --------------------------------------------------------
namespace {
namespace syncres {

// A link that records what the sync sends; it takes pieces when told to.
struct FakeLink final : caliper_live::ProgramLink {
    bool pieces = false;
    uint64_t epoch_n = 1;
    uint32_t next = 1;
    std::vector<Program> sent;
    bool ready() override { return true; }
    uint64_t epoch() override { return epoch_n; }
    uint32_t send(Program msg) override {
        msg.request_id = next++;
        sent.push_back(std::move(msg));
        return sent.back().request_id;
    }
    std::vector<caliper_live::ProgramAnswer> take() override { return {}; }
    bool takes_pieces() override { return pieces; }
};

caliper_live::WantedProgram instance(double gain) {
    const Program golden = std::get<Program>(hubres::golden("program_set_piece"));
    caliper_live::WantedProgram w;
    w.id = "heads";
    w.kind = "piece";
    w.panel = "bars";
    w.piece = golden.piece;
    w.sha256 = golden.piece.find("hash")->as_string().substr(7);
    w.manifest = golden.manifest;
    w.bind = golden.bind;
    w.segments = golden.segments;
    w.common = *golden.common;
    w.derive = *golden.derive;
    w.draw = *golden.draw;
    w.params = Json::object();
    w.params.set("gain", gain);
    return w;
}

}  // namespace syncres
}  // namespace

TEST_CASE("live programs: an instance of a piece goes out in the instance form, only to a script that takes pieces") {
    using namespace syncres;
    caliper_live::ProgramSync sync;
    FakeLink link;
    sync.want({instance(2.0)});
    sync.poll(link);
    CHECK(link.sent.empty());
    REQUIRE(sync.records().size() == 1);
    CHECK(sync.records()[0].state == "refused");
    CHECK(sync.records()[0].error_code == "program.resources_unavailable");
    const auto changes = sync.take_changes();
    REQUIRE(changes.size() == 1);
    CHECK(changes[0].change == "refused");

    // Once the script takes pieces, the set goes out once, as the golden frame's instance with its own request id.
    link.pieces = true;
    sync.poll(link);
    sync.poll(link);
    REQUIRE(link.sent.size() == 1);
    Program m = link.sent[0];
    CHECK(m.op == CALIPER_NET_PROGRAM_SET);
    CHECK(m.declaration.is_null());
    CHECK_FALSE(m.wgsl);
    CHECK_FALSE(m.sha256);
    std::vector<uint8_t> bytes;
    std::string err;
    REQUIRE(encode(Message(m), &bytes, &err, CALIPER_NET_MINOR_3_2));
    Message back;
    REQUIRE(decode_bytes(bytes, &back).ok());
    const Program& p = std::get<Program>(back);
    const Program golden = std::get<Program>(hubres::golden("program_set_piece"));
    CHECK(p.piece == golden.piece);
    CHECK(p.manifest == golden.manifest);
    CHECK(p.bind == golden.bind);
    CHECK(p.params == golden.params);
    CHECK(p.panel == golden.panel);
    CHECK(p.segments == golden.segments);
    CHECK(p.common == golden.common);
    CHECK(p.derive == golden.derive);
    CHECK(p.draw == golden.draw);

    // A new value alone goes out as op params; another binding sends the set again.
    sync.want({instance(3.0)});
    sync.poll(link);
    REQUIRE(link.sent.size() == 2);
    CHECK(link.sent[1].op == CALIPER_NET_PROGRAM_PARAMS);
    caliper_live::WantedProgram rebound = instance(3.0);
    rebound.bind.set("attn", "live:blocks.1.attn");
    sync.want({rebound});
    sync.poll(link);
    REQUIRE(link.sent.size() == 3);
    CHECK(link.sent[2].op == CALIPER_NET_PROGRAM_SET);
    CHECK(link.sent[2].bind.find("attn")->as_string() == "live:blocks.1.attn");
}
#endif   // !_WIN32

