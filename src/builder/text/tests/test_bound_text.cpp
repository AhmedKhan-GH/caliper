// The tests of the format language of bound texts
// (caliper_builder/text/bound_text.h): each format and each refusal of the
// grammar.
#include <doctest/doctest.h>

#include <caliper_builder/text/bound_text.h>

#include <string>

using caliper_builder::text::format_bound_text;
using caliper_builder::text::parse_bound_text;
using caliper_builder::text::ParsedText;
using nlohmann::json;

namespace {

// Formats `format` with one field `v` holding `value`.
std::string one(const std::string& format, const json& value) {
    ParsedText t;
    const caliper_builder::Error e = parse_bound_text(format, {"v"}, &t);
    REQUIRE_MESSAGE(!e, e.message);
    return format_bound_text(t, {{"v", value}});
}

// The code of the refusal of `format` with the fields `names`, or "".
std::string refusal(const std::string& format, const std::vector<std::string>& names = {"v"}) {
    ParsedText t;
    return parse_bound_text(format, names, &t).code;
}

}  // namespace

TEST_CASE("each number format gives the output of the design table") {
    const double x = 12345.678;
    CHECK(one("{v:d}", x) == "12346");
    CHECK(one("{v:,d}", x) == "12,346");
    CHECK(one("{v:.2f}", x) == "12345.68");
    CHECK(one("{v:.2e}", x) == "1.23e+04");
    CHECK(one("{v:.3g}", x) == "1.23e+04");
    CHECK(one("{v:.1%}", x) == "1234567.8%");
    CHECK(one("{v:si.3}", x) == "12.3k");
    CHECK(one("{v:s}", "abc") == "abc");
    CHECK(one("{v}", x) == "12345.7");
}

TEST_CASE("number formats round, group and carry") {
    CHECK(one("{v:,.2f}", 1234567.891) == "1,234,567.89");
    CHECK(one("{v:,d}", -1234567) == "-1,234,567");
    CHECK(one("{v:d}", -0.4) == "0");
    CHECK(one("{v:.0%}", 0.25) == "25%");
    CHECK(one("{v:f}", 1.5) == "1.500000");
    CHECK(one("{v:si}", 999950.0) == "1.00M");
    CHECK(one("{v:si.2}", 0.00123) == "1.2m");
    CHECK(one("{v:si}", -4.2e-6) == "-4.20\xC2\xB5");
    CHECK(one("{v:si}", 0) == "0");
    CHECK(one("{v:si.1}", 5.0) == "5");
    CHECK(one("{v:.3f}", std::nan("")) == "nan");
    CHECK(one("{v:d}", -INFINITY) == "-inf");
    CHECK(one("{v:s}", 3) == "3");
    CHECK(one("{v:s}", 0.125) == "0.125");
    CHECK(one("{v}", true) == "true");
}

TEST_CASE("a missing, unfitting or larger value") {
    CHECK(one("Loss {v:.3f}", nullptr) == "Loss \xE2\x80\x93");
    CHECK(one("{v:.3f}", "text") == "?");
    CHECK(one("{v:d}", true) == "?");
    CHECK(one("{v:d}", json::object()) == "?");
    CHECK(one("{v:.2f}", json::array({0.5})) == "0.50");
    CHECK(one("{v:.2f}", json::array({1, 2, 3})) == "[3]");
    CHECK(one("{v}", json::array({json::array({1, 2}), json::array({3, 4})})) == "[2, 2]");
    CHECK(one("{v:.2f}", json{{"shape", {8, 64}}}) == "[8, 64]");
    ParsedText t;
    REQUIRE(!parse_bound_text("{a} {b}", {"a", "b"}, &t));
    CHECK(format_bound_text(t, {{"a", 1}}) == "1 \xE2\x80\x93");
}

TEST_CASE("literals, doubled braces and several fields") {
    ParsedText t;
    REQUIRE(!parse_bound_text("Step {step:,d}  Loss {loss:.3f}  Token {token:d}", {"step", "loss", "token"}, &t));
    CHECK(t.fields.size() == 3);
    CHECK(format_bound_text(t, {{"step", 12000}, {"loss", 0.41234}, {"token", 7}}) ==
          "Step 12,000  Loss 0.412  Token 7");
    REQUIRE(!parse_bound_text("{{v}} = {v} }}", {"v"}, &t));
    CHECK(format_bound_text(t, {{"v", 2}}) == "{v} = 2 }");
    REQUIRE(!parse_bound_text("", {}, &t));
    CHECK(format_bound_text(t, {}).empty());
    REQUIRE(!parse_bound_text("{v} and {v:.1f}", {"v"}, &t));
    CHECK(format_bound_text(t, {{"v", 2.25}}) == "2.25 and 2.2");
}

TEST_CASE("each refusal of the grammar") {
    CHECK(refusal("{w}") == "spec.bad_format");            // unknown name
    CHECK(refusal("{}") == "spec.bad_format");             // no name
    CHECK(refusal("{v") == "spec.bad_format");             // open brace
    CHECK(refusal("{v {v}}") == "spec.bad_format");        // nested brace
    CHECK(refusal("a } b") == "spec.bad_format");          // lone closing brace
    CHECK(refusal("{v:}") == "spec.bad_format");           // empty format
    CHECK(refusal("{v:x}") == "spec.bad_format");          // unknown type
    CHECK(refusal("{v:.2d}") == "spec.bad_format");        // d takes no precision
    CHECK(refusal("{v:,s}") == "spec.bad_format");         // s takes no comma
    CHECK(refusal("{v:.3s}") == "spec.bad_format");        // s takes no precision
    CHECK(refusal("{v:,si}") == "spec.bad_format");        // si takes no comma
    CHECK(refusal("{v:si.0}") == "spec.bad_format");       // si needs one digit
    CHECK(refusal("{v:.21f}") == "spec.bad_format");       // precision above 20
    CHECK(refusal("{v:.f}") == "spec.bad_format");         // a point without digits
    CHECK(refusal("{v:10.2f}") == "spec.bad_format");      // no width
    CHECK(refusal("{v:>8}") == "spec.bad_format");         // no alignment
    CHECK(refusal("{v.real}") == "spec.bad_format");       // no attributes
    CHECK(refusal("{v[0]}") == "spec.bad_format");         // no index
    CHECK(refusal("{v!r}") == "spec.bad_format");          // no conversion
    CHECK(refusal("{v:.2f:.2f}") == "spec.bad_format");
    CHECK(refusal("{v:.20f}").empty());
}

TEST_CASE("the limits of a format and of its output") {
    CHECK(refusal(std::string(1024, 'a')).empty());
    CHECK(refusal(std::string(1025, 'a')) == "spec.bad_format");
    // Characters, not bytes: 1,024 two-byte letters fit.
    std::string wide;
    for (int i = 0; i < 1024; ++i) wide += "\xC3\xA9";
    CHECK(refusal(wide).empty());
    std::string sixteen, seventeen;
    for (int i = 0; i < 16; ++i) sixteen += "{v}";
    seventeen = sixteen + "{v}";
    CHECK(refusal(sixteen).empty());
    CHECK(refusal(seventeen) == "spec.bad_format");

    // A long output ends with an ellipsis at 1,024 characters.
    ParsedText t;
    REQUIRE(!parse_bound_text(std::string(1000, 'x') + "{v}", {"v"}, &t));
    const std::string out = format_bound_text(t, {{"v", std::string(100, 'y')}});
    CHECK(out.size() == 1023 + 3);
    CHECK(out.substr(out.size() - 3) == "\xE2\x80\xA6");
    CHECK(format_bound_text(t, {{"v", "ok"}}).size() == 1002);
}

TEST_CASE("a refusal names the character position and clears the output") {
    ParsedText t;
    t.parts = {"old"};
    const caliper_builder::Error e = parse_bound_text("ab{zz}", {"v"}, &t);
    CHECK(e.code == "spec.bad_format");
    CHECK(e.message.find("character 3") != std::string::npos);
    CHECK(t.parts.empty());
    CHECK(t.fields.empty());
}
