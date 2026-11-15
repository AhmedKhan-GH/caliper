// The C++ codec of wire protocol v3 (src/host/net/wire.h, json.h) against the
// golden frames of tests/fixtures/wire_v3, which python/caliper/_wire.py wrote:
// every valid frame decodes to what its .json description says and encodes
// back to the same bytes (the golden frames are canonical), every invalid one
// is refused with its close code and reason, message-level errors answer
// bad_request, and random truncations and bit flips of the golden frames never
// crash (the same loop runs under AddressSanitizer outside this suite).
#include <doctest/doctest.h>

#include "net/json.h"
#include "net/wire.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace caliper_net;
using namespace caliper_net::wire;
namespace fs = std::filesystem;

namespace {

fs::path fixtures() { return fs::path(CALIPER_TEST_SOURCE_ROOT) / "tests" / "fixtures" / "wire_v3"; }

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

std::vector<std::string> valid_names() {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(fixtures()))
        if (e.path().extension() == ".bin") out.push_back(e.path().stem().string());
    std::sort(out.begin(), out.end());
    return out;
}

uint64_t ju(const Json& j, const char* key) {
    uint64_t v = 0;
    REQUIRE(j.find(key) != nullptr);
    REQUIRE(j.find(key)->get_u64(&v));
    return v;
}

DecodeResult decode_bytes(const std::vector<uint8_t>& b, Message* m) { return decode(b.data(), b.size(), m); }

// A frame from a metadata object and sections, for the message-level cases.
std::vector<uint8_t> frame_of(uint16_t type, const std::string& meta_json, const std::vector<Section>& sections = {}) {
    Json meta;
    std::string err;
    REQUIRE_MESSAGE(Json::parse(meta_json.data(), meta_json.size(), &meta, &err), err);
    std::vector<uint8_t> out;
    REQUIRE(encode_frame(type, meta, sections, &out));
    return out;
}

}  // namespace

TEST_CASE("net/json: strict reader, canonical writer") {
    auto parses = [](const std::string& s, Json* out = nullptr) {
        Json j;
        std::string err;
        const bool ok = Json::parse(s.data(), s.size(), &j, &err);
        if (out) *out = j;
        return ok;
    };
    Json j;
    REQUIRE(parses(" {\"b\": [1, -2, 3.5, 1e3, true, false, null], \"a\": \"x\\u00e9\\ud83d\\ude00\"}\r\n", &j));
    CHECK(j.dump() == "{\"a\":\"x\xc3\xa9\xf0\x9f\x98\x80\",\"b\":[1,-2,3.5,1000.0,true,false,null]}");
    CHECK(j.find("b")->at(0).is_int());
    CHECK_FALSE(j.find("b")->at(3).is_int());   // an exponent makes a double

    // Refused as Python's json.loads (after a strict UTF-8 decode) refuses.
    for (const char* bad : {"", "{", "{\"a\":1,}", "[1,]", "{\"a\" 1}", "01", "-", "1.", ".5", "+1", "1e",
                            "NaN", "Infinity", "-Infinity", "[nan]", "{\"a\":1} x", "\"\\x\"", "\"\\u12g4\"",
                            "\"tab\there\"", "{'a':1}", "tru", "nul"}) {
        CAPTURE(bad);
        CHECK_FALSE(parses(bad));
    }
    CHECK_FALSE(parses("\xef\xbb\xbf{}"));               // a byte order mark
    CHECK_FALSE(parses("{\"a\":\"\xc0\xaf\"}"));         // overlong UTF-8
    CHECK_FALSE(parses("{\"a\":\"\xed\xa0\x80\"}"));     // an encoded surrogate
    CHECK_FALSE(parses("{\"a\":\"\xf4\x90\x80\x80\"}")); // above U+10FFFF
    CHECK_FALSE(parses("{\"a\":\"\xe2\x82\"}"));         // truncated sequence
    // Lone surrogate escapes are accepted (as by Python) and become U+FFFD.
    REQUIRE(parses("\"\\ud800x\"", &j));
    CHECK(j.as_string() == "\xef\xbf\xbdx");
    // A repeated name keeps its first position and its last value.
    REQUIRE(parses("{\"k\":1,\"j\":2,\"k\":3}", &j));
    CHECK(j.size() == 2);
    CHECK(j.members()[0].first == "k");
    CHECK(j.find("k")->as_double() == 3.0);
    // Depth: 16 nested containers pass, 17 do not.
    std::string d16 = std::string(15, '[') + "{}" + std::string(15, ']');
    std::string d17 = std::string(16, '[') + "{}" + std::string(16, ']');
    CHECK(parses(d16));
    CHECK_FALSE(parses(d17));
    CHECK_FALSE(parses(std::string(100000, '[') + std::string(100000, ']')));   // no recursion trouble
    // Integers: exact in int64 and uint64; beyond, a double.
    REQUIRE(parses("[9223372036854775807,-9223372036854775808,18446744073709551615,18446744073709551616]", &j));
    int64_t i;
    uint64_t u;
    CHECK((j.at(0).get_i64(&i) && i == INT64_MAX));
    CHECK((j.at(1).get_i64(&i) && i == INT64_MIN));
    CHECK((j.at(2).get_u64(&u) && u == UINT64_MAX));
    CHECK_FALSE(j.at(3).is_int());
    CHECK(j.dump() == "[9223372036854775807,-9223372036854775808,18446744073709551615,1.8446744073709552e+19]");

    // The writer: sorted keys, Python's escapes, Python's float repr, null for non-finite.
    Json o = Json::object();
    o.set("z", 1);
    o.set("\xc3\xa9", "\x01\x1f\"\\\n\t\x7f");
    o.set("a", Json::array());
    o.set("f", std::vector<double>{}.empty() ? Json(0.1) : Json());
    o.set("n", std::nan(""));
    CHECK(o.dump() == "{\"a\":[],\"f\":0.1,\"n\":null,\"z\":1,\"\xc3\xa9\":\"\\u0001\\u001f\\\"\\\\\\n\\t\x7f\"}");
    CHECK(format_double(1.0) == "1.0");
    CHECK(format_double(-0.0) == "-0.0");
    CHECK(format_double(1e-5) == "1e-05");
    CHECK(format_double(1e-4) == "0.0001");
    CHECK(format_double(1e16) == "1e+16");
    CHECK(format_double(123456789012345.0) == "123456789012345.0");
    CHECK(format_double(1.5e300) == "1.5e+300");
    CHECK(format_double(0.30000000000000004) == "0.30000000000000004");
}

TEST_CASE("net/wire: every golden frame decodes as described and encodes back to the same bytes") {
    const std::vector<std::string> names = valid_names();
    REQUIRE(names.size() >= 20);
    for (const std::string& name : names) {
        CAPTURE(name);
        const std::vector<uint8_t> bin = read_bytes(fixtures() / (name + ".bin"));
        const Json want = read_json(fixtures() / (name + ".json"));
        Frame f;
        FrameError fe;
        REQUIRE_MESSAGE(decode_frame(bin.data(), bin.size(), &f, &fe), fe.reason << ": " << fe.message);
        const Json& h = *want.find("header");
        CHECK(f.type == ju(h, "type"));
        CHECK(f.major == ju(h, "major"));
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
            CHECK(s.offset % 8 == 0);
            const Json& vals = *d.find("values");
            REQUIRE((int64_t)vals.size() == s.count());
            for (size_t k = 0; k < vals.size(); ++k) {
                if (s.dtype == CALIPER_NET_DTYPE_F32) {
                    float v;
                    std::memcpy(&v, s.data + 4 * k, 4);
                    CHECK(v == (float)vals.at(k).as_double());
                } else if (s.dtype == CALIPER_NET_DTYPE_U8) {
                    CHECK(s.data[k] == (uint8_t)vals.at(k).as_double());
                }
            }
        }

        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        REQUIRE_MESSAGE(r.ok(), r.error_code << ": " << r.message);
        CHECK(message_type(m) == f.type);
        if (f.type <= CALIPER_NET_MSG_EVENT && std::string(type_name(f.type)) != "?")
            CHECK(std::string(type_name(f.type)) == want.find("type_name")->as_string());

        // The golden frames are canonical: re-encoding gives the same bytes,
        // except for the two frames of a later minor version, which a 3.0
        // codec reads but does not reproduce (unknown keys, flags, type).
        std::vector<uint8_t> again;
        std::string err;
        REQUIRE_MESSAGE(encode(m, &again, &err, f.minor), err);
        if (name == "ping_future") {
            REQUIRE(std::holds_alternative<Ping>(m));
            CHECK(std::get<Ping>(m).nonce == 17);
        } else if (name == "unknown_type") {
            REQUIRE(std::holds_alternative<Unknown>(m));
            CHECK(std::get<Unknown>(m).meta.find("future") != nullptr);
            CHECK(again == bin);   // Unknown keeps the metadata and sections as they came
        } else {
            CHECK(again == bin);
        }
    }
}

TEST_CASE("net/wire: decoded messages carry the golden values") {
    Message m;
    auto load = [&](const char* name) {
        static std::vector<std::vector<uint8_t>> keep;   // sections point into these
        keep.push_back(read_bytes(fixtures() / (std::string(name) + ".bin")));
        REQUIRE(decode_bytes(keep.back(), &m).ok());
    };
    load("snapshot");
    {
        const Snapshot& s = std::get<Snapshot>(m);
        CHECK(s.kind == CALIPER_NET_KIND_HEATMAP);
        CHECK(s.view == "attn");
        CHECK(s.sub == 1);
        CHECK(s.seq == 42);
        CHECK(s.colormap == CALIPER_NET_COLORMAP_RDBU);
        CHECK(s.flags == (CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX));
        CHECK(s.stats == std::vector<double>{-1.5, 1.0});
        CHECK(s.data.shape == std::vector<int64_t>{2, 3});
        CHECK(s.data.is(CALIPER_NET_DTYPE_F32, 2));
        CHECK(s.data_view.find("source")->as_string() == "layer0");
        CHECK_FALSE(s.aux.has_value());
    }
    load("snapshot_points");
    {
        const Snapshot& s = std::get<Snapshot>(m);
        CHECK(s.kind == CALIPER_NET_KIND_POINTS);
        CHECK(s.points_dims == 2);
        CHECK(s.label_count == 3u);
        REQUIRE(s.aux.has_value());
        CHECK(s.aux->shape == std::vector<int64_t>{4});
        CHECK(s.stats.size() == 8);
    }
    load("series");
    {
        const Series& s = std::get<Series>(m);
        CHECK(s.start == 0);
        CHECK(s.total == 3);
        CHECK(s.values.shape == std::vector<int64_t>{3});
        float v[3];
        std::memcpy(v, s.values.data, 12);
        CHECK(v[0] == 2.5f);
        CHECK(v[2] == 0.75f);
    }
    load("views");
    {
        const Views& v = std::get<Views>(m);
        CHECK(v.session == std::string("clp-4242-a1b2c3"));
        REQUIRE(v.list.size() == 3);
        CHECK(v.list[0].name == "attn");
        CHECK(v.list[0].kind == CALIPER_NET_KIND_HEATMAP);
        REQUIRE(v.list[0].sources.size() == 2);
        CHECK(v.list[0].sources[1].name == "layer1");
        CHECK(v.list[0].sources[0].dims.size() == 3);
        CHECK(v.list[0].sources[0].dims[0].name == "head");
        CHECK(v.list[0].sources[0].dims[0].size == 4);
        CHECK(v.list[0].default_view.find("index") != nullptr);
        CHECK(v.list[2].kind == CALIPER_NET_KIND_SERIES);
        CHECK(v.list[2].default_view.is_null());
    }
    load("sessions");
    {
        const Sessions& s = std::get<Sessions>(m);
        REQUIRE(s.list.size() == 2);
        CHECK(s.list[0].identity.id == "clp-4242-a1b2c3");
        CHECK(s.list[0].identity.pid == 4242);
        CHECK(s.list[0].state == "live");
        CHECK(s.list[1].state == "ended");
        CHECK(s.list[0].to_json() == s.sessions.at(0));
    }
    load("welcome");
    {
        const Welcome& w = std::get<Welcome>(m);
        CHECK(w.role == "producer");
        Identity id;
        REQUIRE(Identity::parse(w.session, &id));
        CHECK(id.script == "train.py");
        CHECK(id.start_unix_ms == 1763200000000ull);
    }
    load("view_reply_error");
    CHECK(std::get<ViewReply>(m).error.find("code")->as_string() == "bad_data_view");
    load("attach");
    CHECK(std::get<Attach>(m).session == std::string("clp-4242-a1b2c3"));
    load("error");
    CHECK(std::get<Error>(m).in_reply_to == (uint32_t)CALIPER_NET_MSG_SUBSCRIBE);
}

TEST_CASE("net/wire: every invalid golden frame is refused with its close code and reason") {
    const Json expect = read_json(fixtures() / "invalid" / "expect.json");
    REQUIRE(expect.size() >= 15);
    for (const auto& kv : expect.members()) {
        CAPTURE(kv.first);
        const std::vector<uint8_t> bin = read_bytes(fixtures() / "invalid" / kv.first);
        REQUIRE(!bin.empty());
        Frame f;
        FrameError fe;
        CHECK_FALSE(decode_frame(bin.data(), bin.size(), &f, &fe));
        CHECK(fe.close_code == (int)ju(kv.second, "close_code"));
        CHECK(fe.reason == kv.second.find("reason")->as_string());
        Message m;
        const DecodeResult r = decode_bytes(bin, &m);
        CHECK(r.status == DecodeResult::Status::BadFrame);
        CHECK(r.close_code == fe.close_code);
    }
}

TEST_CASE("net/wire: frame rules beyond the golden set") {
    std::vector<uint8_t> ok = frame_of(CALIPER_NET_MSG_PING, "{\"nonce\":1}");
    Frame f;
    FrameError fe;
    REQUIRE(decode_frame(ok.data(), ok.size(), &f, &fe));
    // minor, flags and reserved never reject.
    std::vector<uint8_t> b = ok;
    b[6] = 7;
    b[10] = 0xff;
    b[22] = 0x55;
    CHECK(decode_frame(b.data(), b.size(), &f, &fe));
    // "sections" in a frame without sections: only absent, null or [] pass.
    for (const char* meta : {"{\"nonce\":1,\"sections\":null}", "{\"nonce\":1,\"sections\":[]}"}) {
        b = ok;
        std::string m = meta;
        std::vector<uint8_t> raw(24 + m.size());
        std::memcpy(raw.data(), ok.data(), 24);
        const uint32_t fl = (uint32_t)raw.size(), ml = (uint32_t)m.size();
        std::memcpy(raw.data() + 12, &fl, 4);
        std::memcpy(raw.data() + 16, &ml, 4);
        std::memcpy(raw.data() + 24, m.data(), m.size());
        CHECK(decode_frame(raw.data(), raw.size(), &f, &fe));
    }
    {
        std::string m = "{\"nonce\":1,\"sections\":[{}]}";
        std::vector<uint8_t> raw(24 + m.size());
        std::memcpy(raw.data(), ok.data(), 24);
        const uint32_t fl = (uint32_t)raw.size(), ml = (uint32_t)m.size();
        std::memcpy(raw.data() + 12, &fl, 4);
        std::memcpy(raw.data() + 16, &ml, 4);
        std::memcpy(raw.data() + 24, m.data(), m.size());
        CHECK_FALSE(decode_frame(raw.data(), raw.size(), &f, &fe));
        CHECK(fe.reason == "malformed");
    }
    // Sections: unknown dtypes and encodings pass without a length check;
    // repeated names and a negative shape entry do not.
    const uint8_t bytes[16] = {};
    std::vector<uint8_t> out;
    REQUIRE(encode_frame(0x7777, Json::object(), {Section::make("x", 99, {3}, bytes, 5)}, &out));
    CHECK(decode_frame(out.data(), out.size(), &f, &fe));
    Section arrow = Section::make("t", CALIPER_NET_DTYPE_F32, {100}, bytes, 7);
    arrow.encoding = CALIPER_NET_ENCODING_ARROW_IPC;
    REQUIRE(encode_frame(0x7777, Json::object(), {arrow}, &out));
    CHECK(decode_frame(out.data(), out.size(), &f, &fe));
    REQUIRE(encode_frame(0x7777, Json::object(),
                         {Section::make("x", 0, {1}, bytes, 4), Section::make("x", 0, {1}, bytes, 4)}, &out));
    CHECK_FALSE(decode_frame(out.data(), out.size(), &f, &fe));
    REQUIRE(encode_frame(0x7777, Json::object(), {Section::make("x", 0, {2}, bytes, 4)}, &out));
    CHECK_FALSE(decode_frame(out.data(), out.size(), &f, &fe));   // 2 f32 need 8 bytes
    CHECK(fe.reason == "malformed");
    // Padding: every section starts at a multiple of 8, the gaps are zero.
    REQUIRE(encode_frame(0x7777, Json::object(),
                         {Section::make("a", 5, {3}, "abc", 3), Section::make("b", 5, {1}, "d", 1)}, &out));
    REQUIRE(decode_frame(out.data(), out.size(), &f, &fe));
    CHECK(f.sections[0].offset % 8 == 0);
    CHECK(f.sections[1].offset % 8 == 0);
    CHECK(f.sections[1].offset == f.sections[0].offset + 8);
    CHECK(out[f.sections[0].offset + 3] == 0);
    // The encoder refuses what the decoder would refuse as too big.
    std::vector<Section> nine(9, Section::make("s", 5, {0}, bytes, 0));
    for (int i = 0; i < 9; ++i) nine[(size_t)i].name = "s" + std::to_string(i);
    CHECK_FALSE(encode_frame(0x7777, Json::object(), nine, &out, &fe));
    CHECK(fe.close_code == CALIPER_NET_CLOSE_TOO_BIG);
    Json huge = Json::object();
    huge.set("k", std::string(CALIPER_NET_MAX_META_BYTES, 'x'));
    CHECK_FALSE(encode_frame(0x7777, huge, {}, &out, &fe));
    CHECK(fe.close_code == CALIPER_NET_CLOSE_TOO_BIG);
}

TEST_CASE("net/wire: invalid messages are answered with bad_request, the frame being well formed") {
    const float f4[4] = {1, 2, 3, 4};
    const uint8_t px[16] = {};
    auto msg_error = [](const std::vector<uint8_t>& b) {
        Message m;
        const DecodeResult r = decode(b.data(), b.size(), &m);
        return r.status == DecodeResult::Status::BadMessage ? r.error_code : std::string("ok");
    };
    const std::string snap = "\"sub\":1,\"view\":\"v\",\"seq\":1,\"stamp_ns\":2,\"data_view\":{}";
    auto data = [&](std::vector<int64_t> shape, int64_t dt = 0) {
        return std::vector<Section>{Section::make("data", dt, shape, dt == 5 ? (const void*)px : (const void*)f4,
                                                  dt == 5 ? 16 : 16)};
    };
    struct Case {
        const char* what;
        std::vector<uint8_t> frame;
        const char* code;
    };
    std::vector<Case> cases = {
        {"hello without role", frame_of(CALIPER_NET_MSG_HELLO, "{}"), "bad_request"},
        {"hello, unknown role", frame_of(CALIPER_NET_MSG_HELLO, "{\"role\":\"hub\"}"), "bad_request"},
        {"producer hello without session", frame_of(CALIPER_NET_MSG_HELLO, "{\"role\":\"producer\"}"), "bad_request"},
        {"welcome as viewer", frame_of(CALIPER_NET_MSG_WELCOME, "{\"role\":\"viewer\"}"), "bad_request"},
        {"ping nonce too big", frame_of(CALIPER_NET_MSG_PING, "{\"nonce\":4294967296}"), "bad_request"},
        {"ping nonce true", frame_of(CALIPER_NET_MSG_PING, "{\"nonce\":true}"), "bad_request"},
        {"ping nonce 1.0", frame_of(CALIPER_NET_MSG_PING, "{\"nonce\":1.0}"), "bad_request"},
        {"attach without session", frame_of(CALIPER_NET_MSG_ATTACH, "{}"), "bad_request"},
        {"attach bad id", frame_of(CALIPER_NET_MSG_ATTACH, "{\"session\":\"a b\"}"), "bad_request"},
        {"attach null", frame_of(CALIPER_NET_MSG_ATTACH, "{\"session\":null}"), "ok"},
        {"subscribe rate 0", frame_of(CALIPER_NET_MSG_SUBSCRIBE, "{\"sub\":1,\"view\":\"v\",\"request_id\":1,\"max_rate\":0}"), "bad_request"},
        {"subscribe long name", frame_of(CALIPER_NET_MSG_SUBSCRIBE, "{\"sub\":1,\"view\":\"" + std::string(64, 'v') + "\",\"request_id\":1}"), "bad_request"},
        {"view_reply with both", frame_of(CALIPER_NET_MSG_VIEW_REPLY, "{\"request_id\":1,\"sub\":1,\"view\":\"v\",\"data_view\":{},\"error\":{\"code\":\"x\"}}"), "bad_request"},
        {"view_reply with neither", frame_of(CALIPER_NET_MSG_VIEW_REPLY, "{\"request_id\":1,\"sub\":1,\"view\":\"v\"}"), "bad_request"},
        {"views entry without sources", frame_of(CALIPER_NET_MSG_VIEWS, "{\"views\":[{\"id\":1,\"name\":\"v\",\"kind\":1}]}"), "bad_request"},
        {"sessions bad state", frame_of(CALIPER_NET_MSG_SESSIONS, "{\"sessions\":[{\"id\":\"a\",\"script\":\"s\",\"pid\":1,\"start_unix_ms\":1,\"state\":\"gone\",\"views\":0}]}"), "bad_request"},
        {"snapshot without data", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":1," + snap + "}"), "bad_request"},
        {"heatmap of rank 1", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":1," + snap + "}", data({4})), "bad_request"},
        {"image of f32", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":2," + snap + "}", data({1, 1, 4}, 0)), "bad_request"},
        {"image with one channel", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":2," + snap + "}", data({2, 8, 1}, 5)), "bad_request"},
        {"image of 2 x 2 x 4 u8", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":2," + snap + "}", data({2, 2, 4}, 5)), "ok"},
        {"points (1, 4)", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":3," + snap + "}", data({1, 4})), "bad_request"},
        {"points dims disagree", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":3,\"points_dims\":3," + snap + "}", data({2, 2})), "bad_request"},
        {"points dims 2.0", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":3,\"points_dims\":2.0," + snap + "}", data({2, 2})), "ok"},
        {"stats with a string", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":1,\"stats\":[\"x\"]," + snap + "}", data({2, 2})), "bad_request"},
        {"stats too long", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":1,\"stats\":[" + [] { std::string s = "0"; for (int i = 0; i < 64; ++i) s += ",0"; return s; }() + "]," + snap + "}", data({2, 2})), "bad_request"},
        {"unknown kind passes", frame_of(CALIPER_NET_MSG_SNAPSHOT, "{\"kind\":77," + snap + "}", data({4})), "ok"},
        {"series past total", frame_of(CALIPER_NET_MSG_SERIES, "{\"sub\":1,\"view\":\"v\",\"start\":2,\"total\":5,\"stamp_ns\":0}", {Section::make("values", 0, {4}, f4, 16)}), "bad_request"},
        {"series of u8", frame_of(CALIPER_NET_MSG_SERIES, "{\"sub\":1,\"view\":\"v\",\"start\":0,\"total\":16,\"stamp_ns\":0}", {Section::make("values", 5, {16}, px, 16)}), "bad_request"},
        {"event with a negative index", frame_of(CALIPER_NET_MSG_EVENT, "{\"view\":\"v\",\"type\":\"select\",\"detail\":{\"indices\":[-1]}}"), "bad_request"},
        {"error with fatal 1", frame_of(CALIPER_NET_MSG_ERROR, "{\"code\":\"x\",\"fatal\":1}"), "bad_request"},
        {"unknown type with anything", frame_of(0x0999, "{\"x\":[1]}"), "ok"},
    };
    for (const Case& c : cases) {
        const std::string what = c.what;
        CAPTURE(what);
        CHECK(msg_error(c.frame) == std::string(c.code));
    }
}

TEST_CASE("net/wire: messages built in C++ round-trip") {
    const float heat[6] = {-1.5f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f};
    Snapshot s;
    s.sub = 7;
    s.view = "w";
    s.seq = 3;
    s.stamp_ns = 123456789;
    s.kind = CALIPER_NET_KIND_HEATMAP;
    s.data = Section::make("data", CALIPER_NET_DTYPE_F32, {2, 3}, heat, sizeof heat);
    s.flags = CALIPER_NET_SNAP_AUTO_VMIN;
    s.vmax = 2.0;
    s.stats = {-1.5, std::nan("")};
    std::vector<uint8_t> b;
    REQUIRE(encode(s, &b));
    Message m;
    REQUIRE(decode(b.data(), b.size(), &m).ok());
    const Snapshot& d = std::get<Snapshot>(m);
    CHECK(d.view == "w");
    CHECK(d.vmax == 2.0);
    CHECK(d.stats.size() == 2);
    CHECK(std::isnan(d.stats[1]));
    CHECK(std::memcmp(d.data.data, heat, sizeof heat) == 0);
    std::vector<uint8_t> b2;
    REQUIRE(encode(d, &b2));
    CHECK(b == b2);

    Views v;
    v.session = "s1";
    ViewInfo vi;
    Json entry = Json::object();
    entry.set("id", 1);
    entry.set("name", "w");
    entry.set("kind", 1);
    entry.set("sources", Json::array());
    v.views.push(entry);
    REQUIRE(encode(v, &b));
    REQUIRE(decode(b.data(), b.size(), &m).ok());
    CHECK(std::get<Views>(m).list.at(0).name == "w");
    Views none;   // session null: written as null, read back as absent
    REQUIRE(encode(none, &b));
    REQUIRE(decode(b.data(), b.size(), &m).ok());
    CHECK_FALSE(std::get<Views>(m).session.has_value());
}

#ifndef CALIPER_NET_FUZZ_ITERATIONS
#define CALIPER_NET_FUZZ_ITERATIONS 20000   // the sanitizer build raises it
#endif

TEST_CASE("net/wire: truncations and bit flips of the golden frames never crash") {
    std::vector<std::vector<uint8_t>> frames;
    for (const std::string& n : valid_names()) frames.push_back(read_bytes(fixtures() / (n + ".bin")));
    std::mt19937 g(12345);
    size_t ok = 0, bad_frame = 0, bad_msg = 0;
    const int iterations = CALIPER_NET_FUZZ_ITERATIONS;
    for (int it = 0; it < iterations; ++it) {
        std::vector<uint8_t> b = frames[(size_t)it % frames.size()];
        switch (g() % 4) {
            case 0: b.resize(g() % (b.size() + 1)); break;   // truncate
            case 1:
                for (int k = 0; k < 1 + (int)(g() % 4); ++k) b[g() % b.size()] ^= uint8_t(1u << (g() % 8));
                break;
            case 2: b[g() % b.size()] = uint8_t(g()); break;
            case 3: {   // flip inside the metadata or the header only
                const size_t lim = std::min<size_t>(b.size(), 200);
                b[g() % lim] ^= uint8_t(1u << (g() % 8));
                break;
            }
        }
        // An exact-size copy, so a read past the end is a real out-of-bounds read
        // under AddressSanitizer.
        std::unique_ptr<uint8_t[]> exact(new uint8_t[b.size() ? b.size() : 1]);
        if (!b.empty()) std::memcpy(exact.get(), b.data(), b.size());
        Message m;
        const DecodeResult r = decode(exact.get(), b.size(), &m);
        if (r.ok()) {
            ++ok;
            std::vector<uint8_t> again;
            encode(m, &again);   // whatever decoded must encode
        } else if (r.status == DecodeResult::Status::BadFrame) {
            ++bad_frame;
            CHECK((r.close_code == 4000 || r.close_code == 4001 || r.close_code == 1009));
        } else {
            ++bad_msg;
        }
    }
    MESSAGE(iterations << " mutated frames: " << ok << " decoded, " << bad_frame << " refused as frames, " << bad_msg
                                     << " refused as messages");
    CHECK(ok + bad_frame + bad_msg == (size_t)iterations);
}
