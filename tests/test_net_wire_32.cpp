// The scenes subset of the 3.2 draft in the C++ codec (src/host/net/wire.h)
// against the golden frames of tests/fixtures/wire_v3/v3_2, which
// python/caliper/_wire.py wrote: every frame decodes to what its .json says
// and encodes back to the same bytes at minor 2 (except the two that carry
// sections the subset ignores), every invalid frame is refused as
// v3_2/invalid/expect.json says, the batch limits hold, the scene view helpers
// follow python/tests/test_wire_32.py, a forwarder can rewrite `sub` and keep
// the sections, and truncations and bit flips never crash.
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

bool decode_only(const std::string& n) { return n == "scene_update_points_stamped" || n == "scene_update_mesh_normals"; }

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
    REQUIRE(names.size() >= 16);
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
            }
        }

        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        REQUIRE_MESSAGE(r.ok(), r.error_code << ": " << r.message);
        CHECK(message_type(m) == f.type);
        CHECK(std::string(type_name(f.type)) == want.find("type_name")->as_string());
        if (f.type == CALIPER_NET_MSG_SCENE_UPDATE) CHECK(std::holds_alternative<SceneUpdate>(m));
        if (f.type == CALIPER_NET_MSG_SCENE_CLEAR) CHECK(std::holds_alternative<SceneClear>(m));

        std::vector<uint8_t> again;
        std::string err;
        REQUIRE_MESSAGE(encode(m, &again, &err, CALIPER_NET_MINOR_3_2), err);
        if (decode_only(name)) CHECK(again != bin);   // the sections the subset ignores are not written
        else CHECK(again == bin);
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
    {   // the decode-only frames: the sections the subset ignores are left out
        const std::vector<uint8_t> bin = read_bytes(v32_dir() / "scene_update_points_stamped.bin");
        Message m;
        REQUIRE(decode_bytes(bin, &m).ok());
        const SceneUpdate& u = std::get<SceneUpdate>(m);
        REQUIRE(u.arrays.size() == 2);
        CHECK(u.array("stamp") == nullptr);
        const std::vector<uint8_t> bin2 = read_bytes(v32_dir() / "scene_update_mesh_normals.bin");
        REQUIRE(decode_bytes(bin2, &m).ok());
        CHECK(std::get<SceneUpdate>(m).array("normals") == nullptr);
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

TEST_CASE("net/wire 3.2: every invalid golden frame is refused as expect.json says") {
    const Json expect = read_json(v32_dir() / "invalid" / "expect.json");
    REQUIRE(expect.size() >= 18);
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
    CHECK(frames == 2);
    CHECK(messages >= 16);
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
    REQUIRE(scene_view_info(parse("{\"kind\":6,\"scene\":{\"view_camera\":{\"azimuth\":30}}}"), &v, &err));
    CHECK(v.up == "+y");
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
            CHECK((r.error_code == "bad_request" || r.error_code == "unsupported"));
        }
    }
    MESSAGE(iterations << " mutated frames: " << ok << " decoded, " << bad_frame << " refused as frames, " << bad_msg
                       << " refused as messages");
    CHECK(ok + bad_frame + bad_msg == (size_t)iterations);
}
