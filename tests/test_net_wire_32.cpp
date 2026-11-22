// The scenes of the 3.2 draft in the C++ codec (src/host/net/wire.h) against
// the golden frames of tests/fixtures/wire_v3/v3_2, which python/caliper/
// _wire.py wrote: every frame decodes to what its .json says and encodes back
// to the same bytes at minor 2 (except the one that carries a section no part
// has), every invalid frame is refused as v3_2/invalid/expect.json says, every
// part, stamps, entity transforms and class tables carry their golden values,
// the limits hold, the scene view helpers follow python/tests/test_wire_32.py,
// a forwarder can rewrite `sub` and keep the sections, and truncations and bit
// flips never crash.
#include <doctest/doctest.h>

#include "net/json.h"
#include "net/wire.h"

#include <algorithm>
#include <cmath>
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
