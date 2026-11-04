#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/live_requests.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

TEST_CASE("request channel frames partial replies and detects disconnect") {
    const auto dir = "/tmp/caliper-" + std::to_string(getuid());
    mkdir(dir.c_str(), 0700);
    const auto token = "cpp-" + std::to_string(getpid());
    const auto path = dir + "/" + token + ".sock";
    int listener = socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(listener >= 0);
    sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    std::strcpy(addr.sun_path, path.c_str());
    REQUIRE(bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    REQUIRE(listen(listener, 1) == 0);
    caliper_live::RequestChannel channel;
    REQUIRE(channel.connect(token));
    int peer = accept(listener, nullptr, nullptr);
    REQUIRE(peer >= 0);
    REQUIRE(channel.send("{\"id\":1,\"op\":\"hello\"}"));
    channel.poll();
    char buf[100]{};
    const int n = read(peer, buf, sizeof buf);
    REQUIRE(n > 0);
    CHECK(std::string(buf, n) == "{\"id\":1,\"op\":\"hello\"}\n");
    write(peer, "{\"id\":", 6);
    CHECK(channel.poll().empty());
    const std::string tail = "1}\n{\"id\":2}\n";
    write(peer, tail.data(), tail.size());
    auto replies = channel.poll();
    REQUIRE(replies.size() == 2);
    CHECK(replies[0] == "{\"id\":1}");
    CHECK(replies[1] == "{\"id\":2}");
    CHECK_FALSE(channel.send("bad\nframe"));
    CHECK_FALSE(channel.send(std::string(4 * 1024 * 1024, 'x')));
    close(peer);
    channel.poll();
    CHECK_FALSE(channel.connected());
    CHECK_FALSE(channel.error().empty());
    close(listener); unlink(path.c_str());
    CHECK_FALSE(channel.connect("../../escape"));
}
