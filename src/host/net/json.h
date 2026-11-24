#pragma once
// json.h: the JSON of wire protocol v3 metadata (caliper/live_net.h): a strict
// RFC 8259 reader with the protocol's depth limit, and a writer of the
// protocol's canonical encoding. Host-internal C++17, no dependency.
//
// READER. The input must be valid UTF-8 (overlong forms, surrogates and code
// points above U+10FFFF are refused) with no byte order mark; whitespace is
// space, tab, CR and LF; numbers follow the RFC grammar exactly (no NaN,
// Infinity, leading zeros, '+' or bare '.'); strings refuse raw control
// characters and unknown escapes. A number without a fraction or an exponent
// is an integer (kept exactly when it fits int64, or uint64 above that).
// Larger integers use a double with integer-token metadata; other numbers
// use a double. Objects keep their members in order; a repeated
// name keeps its first position and its last value. Containers nest at most
// `max_depth` deep (the top-level container is depth 1). These are the rules
// of python/caliper/_wire.py (json.loads after a strict UTF-8 decode), so the
// native reader refuses what the reference codec refuses.
//
// WRITER. dump() writes the canonical form of _wire.canonical_json: compact
// separators, object members sorted by name (UTF-8 byte order, which is code
// point order), non-ASCII characters raw, the escapes \" \\ \b \f \n \r \t and
// \u00XX (lower-case hex) for the other control characters, integers in
// decimal, and doubles as Python's repr() writes them (the shortest digits that
// read back to the same double: "1.0", "0.25", "1e-05", "1.5e+20"). A
// non-finite double is written as null.
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace caliper_net {

class Json {
public:
    enum class Type : uint8_t { Null, Bool, Int, UInt, Double, String, Array, Object };
    using Member = std::pair<std::string, Json>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : type_(Type::Bool), b_(b) {}
    // Any integer type except bool.
    template <typename T,
              typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type = 0>
    Json(T v) {
        if (std::is_signed<T>::value) {
            type_ = Type::Int;
            i_ = (int64_t)v;
        } else {
            set_unsigned((uint64_t)v);
        }
    }
    Json(double v) : type_(Type::Double), d_(v) {}
    Json(const char* s) : type_(Type::String), s_(s) {}
    Json(std::string s) : type_(Type::String), s_(std::move(s)) {}

    static Json array() { Json j; j.type_ = Type::Array; return j; }
    static Json object() { Json j; j.type_ = Type::Object; return j; }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_int() const { return type_ == Type::Int || type_ == Type::UInt; }   // stored exactly
    // Integer values and parsed tokens without a fraction or exponent, even after overflow.
    // This metadata does not change the value, equality, or canonical encoding.
    bool is_integer_token() const { return is_int() || integer_token_; }
    bool is_number() const { return is_int() || type_ == Type::Double; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool() const { return b_; }
    double as_double() const;                 // any number
    const std::string& as_string() const { return s_; }
    // Integer accessors: false unless is_int() and the value fits.
    bool get_i64(int64_t* out) const;
    bool get_u64(uint64_t* out) const;

    // Arrays.
    size_t size() const { return type_ == Type::Array ? a_.size() : type_ == Type::Object ? o_.size() : 0; }
    const Json& at(size_t i) const { return a_[i]; }
    const std::vector<Json>& items() const { return a_; }
    Json& push(Json v) { a_.push_back(std::move(v)); return a_.back(); }

    // Objects.
    const std::vector<Member>& members() const { return o_; }
    const Json* find(const std::string& key) const;   // nullptr when absent
    bool has(const std::string& key) const { return find(key) != nullptr; }
    Json& set(const std::string& key, Json v);       // replaces an existing member
    // An object of these members in their order, where a repeated name keeps
    // its first position and its last value, as the reader's rule; in
    // O(n log n) for n members, where n calls of set() take O(n^2).
    static Json object_of(std::vector<Member> members);
    void erase(const std::string& key);

    // The canonical encoding (see above).
    std::string dump() const;
    void dump_to(std::string& out) const;

    // Strict parse of [p, p + n). False with *err (and *out untouched) on any
    // violation of the rules above.
    static bool parse(const char* p, size_t n, Json* out, std::string* err, unsigned max_depth = 16);

    bool operator==(const Json& o) const;
    bool operator!=(const Json& o) const { return !(*this == o); }

private:
    friend struct JsonParser;
    void set_unsigned(uint64_t v);
    Type type_ = Type::Null;
    bool b_ = false;
    bool integer_token_ = false;
    int64_t i_ = 0;
    uint64_t u_ = 0;
    double d_ = 0.0;
    std::string s_;
    std::vector<Json> a_;
    std::vector<Member> o_;
};

// True when [p, p + n) is valid UTF-8 (RFC 3629).
bool valid_utf8(const char* p, size_t n);
// Python's repr() of a finite double (see above).
std::string format_double(double v);
// Appends s as a JSON string literal in the canonical escaping.
void json_quote(std::string& out, const std::string& s);

}  // namespace caliper_net
