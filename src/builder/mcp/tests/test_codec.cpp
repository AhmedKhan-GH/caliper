// The small codecs of the mcp module: base64, the PNG size, the nesting check
// and redaction.
#include "../codec.h"

#include <doctest/doctest.h>

#include <string>

using namespace caliper_builder::mcp::detail;

TEST_CASE("base64 round-trips every length and refuses malformed text") {
    std::string bytes;
    for (int i = 0; i < 300; ++i) {
        const std::string enc = base64_encode(bytes);
        CHECK(enc.size() % 4 == 0);
        std::string back;
        REQUIRE(base64_decode(enc, &back));
        CHECK(back == bytes);
        bytes.push_back(char(i * 37 + 11));
    }
    CHECK(base64_encode("Man") == "TWFu");
    CHECK(base64_encode("Ma") == "TWE=");
    CHECK(base64_encode("M") == "TQ==");
    std::string out;
    CHECK_FALSE(base64_decode("TWF", &out));       // length
    CHECK_FALSE(base64_decode("TW=u", &out));      // padding inside
    CHECK_FALSE(base64_decode("T===", &out));      // three pads
    CHECK_FALSE(base64_decode("TQ==TWFu", &out));  // padding before the end
    CHECK_FALSE(base64_decode("TW\nu", &out));     // a character outside the alphabet
    CHECK(base64_decode("", &out));
    CHECK(out.empty());
}

TEST_CASE("png_size reads the IHDR size and refuses what is not a PNG") {
    std::string png = std::string("\x89PNG\r\n\x1a\n", 8) + std::string("\0\0\0\x0d", 4) + "IHDR" +
                      std::string("\0\0\x01\x40", 4) + std::string("\0\0\0\xf0", 4) + std::string(5, '\0');
    std::uint32_t w = 0, h = 0;
    REQUIRE(png_size(png, &w, &h));
    CHECK(w == 320);
    CHECK(h == 240);
    CHECK_FALSE(png_size(png.substr(0, 20), &w, &h));
    std::string bad = png;
    bad[1] = 'X';
    CHECK_FALSE(png_size(bad, &w, &h));
    bad = png;
    bad[12] = 'J';
    CHECK_FALSE(png_size(bad, &w, &h));
    std::string zero = png;
    zero[19] = '\0';
    zero[18] = '\0';
    CHECK_FALSE(png_size(zero, &w, &h));
}

TEST_CASE("json_depth_ok counts brackets outside strings only") {
    CHECK(json_depth_ok("{\"a\":[1,2,{\"b\":3}]}", 3));
    CHECK_FALSE(json_depth_ok("{\"a\":[1,2,{\"b\":3}]}", 2));
    CHECK(json_depth_ok("{\"a\":\"[[[[[[[[\"}", 1));
    CHECK(json_depth_ok("{\"a\":\"\\\"[[[[\"}", 1));
    CHECK_FALSE(json_depth_ok(std::string(1000, '['), 128));
    CHECK(json_depth_ok(std::string(128, '['), 128));
}

TEST_CASE("redact replaces a secret everywhere and leaves short ones alone") {
    std::string text = "a abcdefghijklmnop b abcdefghijklmnop";
    redact(&text, "abcdefghijklmnop");
    CHECK(text == "a <redacted> b <redacted>");
    std::string short_one = "abc abc";
    redact(&short_one, "abc");
    CHECK(short_one == "abc abc");
}
