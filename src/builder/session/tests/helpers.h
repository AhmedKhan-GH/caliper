#pragma once
// Helpers for the session tests: a fresh folder under the build tree for each
// test, a safetensors writer (parquet_writer.h writes Parquet), a
// session opener, a command runner that checks results against the table's
// schemas, and a small HTTP client for the stand-in host. Nothing here touches
// a path outside the build folder.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace session_test {

namespace fs = std::filesystem;
using nlohmann::json;

// A new, empty folder <build>/test-tmp/<name>, returned as a canonical path.
inline std::string fresh_dir(const std::string& name) {
    fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "session" / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return fs::canonical(dir).string();
}

inline void write_text(const std::string& path, const std::string& text) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

inline std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::string fixture_path(const std::string& name) {
    return std::string(CALIPER_BUILDER_SOURCE_DIR) + "/spec/tests/fixtures/" + name;
}

inline caliper_builder::spec::Spec parse_fixture(const std::string& name) {
    auto parsed = caliper_builder::spec::parse_toml(read_text(fixture_path(name)));
    REQUIRE_MESSAGE(parsed.ok(), name);
    return parsed.spec;
}

// Writes a safetensors file of one f32 tensor.
inline void write_safetensors_f32(const std::string& path, const std::string& name, const std::vector<int64_t>& shape,
                                  const std::vector<float>& values) {
    json header;
    header[name] = {{"dtype", "F32"}, {"shape", shape}, {"data_offsets", {0, values.size() * 4}}};
    std::string h = header.dump();
    while (h.size() % 8) h += ' ';
    std::vector<char> bytes(8 + h.size() + values.size() * 4);
    const std::uint64_t n = h.size();
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<char>((n >> (8 * i)) & 0xFF);
    std::memcpy(bytes.data() + 8, h.data(), h.size());
    if (!values.empty()) std::memcpy(bytes.data() + 8 + h.size(), values.data(), values.size() * 4);
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

inline std::unique_ptr<caliper_builder::Session> open_session(const std::string& ws) {
    caliper_builder::SessionOptions options;
    options.registry.check_results = true;  // every result must pass its schema
    auto opened = caliper_builder::Session::open(ws, options);
    REQUIRE_MESSAGE(opened.error.ok(), opened.error.code << ": " << opened.error.message);
    return std::move(opened.session);
}

inline caliper_builder::CommandResult invoke(caliper_builder::Session& s, const std::string& name, json args = json::object(),
                                             caliper_builder::CommandOrigin origin = caliper_builder::CommandOrigin::ui,
                                             const std::string& group = {}, const std::string& merge_key = {}) {
    caliper_builder::CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = origin;
    inv.group = group;
    inv.merge_key = merge_key;
    return s.invoke(std::move(inv));
}

// Runs a command that must succeed, and returns its result.
inline json ok(caliper_builder::Session& s, const std::string& name, json args = json::object(),
               caliper_builder::CommandOrigin origin = caliper_builder::CommandOrigin::ui, const std::string& group = {}) {
    auto r = invoke(s, name, std::move(args), origin, group);
    INFO(name << ": " << r.error.code << " at " << r.error.where << ": " << r.error.message);
    REQUIRE(r.ok);
    return r.result;
}

// Runs a command that must fail, and returns its error code.
inline std::string err(caliper_builder::Session& s, const std::string& name, json args = json::object(),
                       caliper_builder::CommandOrigin origin = caliper_builder::CommandOrigin::ui) {
    auto r = invoke(s, name, std::move(args), origin);
    INFO(name << " answered " << (r.ok ? r.result.dump() : r.error.message));
    CHECK_FALSE(r.ok);
    return r.error.code;
}

#ifndef _WIN32

struct HttpReply {
    int status = 0;
    std::string head;
    std::string body;
    json json_body() const { return json::parse(body, nullptr, false); }
};

// Sends raw bytes to 127.0.0.1:<port> and reads until the server closes.
// With `close_write` false the client keeps its side open, as a slow client does.
inline HttpReply http_raw(int port, const std::string& request, int timeout_ms = 20000, bool close_write = true) {
    HttpReply reply;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return reply;
#ifdef SO_NOSIGPIPE
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        ::close(fd);
        return reply;
    }
    std::size_t sent = 0;
    while (sent < request.size()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const ssize_t n = ::send(fd, request.data() + sent, request.size() - sent, flags);
        if (n <= 0) break;  // the server may answer before it reads everything
        sent += static_cast<std::size_t>(n);
    }
    if (close_write) ::shutdown(fd, SHUT_WR);
    std::string data;
    char buf[65536];
    for (;;) {
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, timeout_ms) <= 0) break;
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        data.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);
    const auto end = data.find("\r\n\r\n");
    if (end == std::string::npos) return reply;
    reply.head = data.substr(0, end);
    reply.body = data.substr(end + 4);
    if (reply.head.size() > 12) reply.status = std::atoi(reply.head.substr(9, 3).c_str());
    return reply;
}

inline HttpReply http(int port, const std::string& method, const std::string& path, const std::string& token,
                      const std::string& body = {}, const std::vector<std::string>& extra = {},
                      int timeout_ms = 20000) {
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) + "\r\n";
    if (!token.empty()) req += "Authorization: Bearer " + token + "\r\n";
    for (const auto& h : extra) req += h + "\r\n";
    if (method == "POST") {
        req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    }
    req += "\r\n" + body;
    return http_raw(port, req, timeout_ms);
}

#endif

}  // namespace session_test
