// CPU-only discovery: metadata and socket file attributes, with no connection.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/request_discovery.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
using namespace caliper_live;

std::string metadata(const std::string& token, int pid, const std::string& capabilities =
                     "[\"generate\",\"list_tensors\",\"show_tensor\"]") {
    return "{\"protocol\":1,\"session\":\"" + token + "\",\"pid\":" + std::to_string(pid) +
           ",\"capabilities\":" + capabilities + "}";
}
struct Files {
    std::string directory, token = "discovery-test", socket_path, advertisement;
    int listener = -1;
    SessionEntry session;

    Files() {
        char pattern[] = "/tmp/caliper-discovery-XXXXXX";
        const char* made = mkdtemp(pattern);
        REQUIRE(made != nullptr);
        directory = made;
        socket_path = directory + "/" + token + ".sock";
        advertisement = directory + "/" + token + ".requests.json";
        listener = socket(AF_UNIX, SOCK_STREAM, 0);
        REQUIRE(listener >= 0);
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        REQUIRE(socket_path.size() < sizeof address.sun_path);
        std::strcpy(address.sun_path, socket_path.c_str());
        REQUIRE(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof address) == 0);
        REQUIRE(chmod(socket_path.c_str(), 0600) == 0);
        REQUIRE(listen(listener, 1) == 0);
        REQUIRE(fcntl(listener, F_SETFL, O_NONBLOCK) == 0);
        session.session = token;
        session.pid = getpid();
        session.script = "my_model.py";
        session.protocol = CALIPER_LIVE_PROTOCOL_VERSION;
        session.alive = true;
    }
    ~Files() {
        if (listener >= 0) close(listener);
        unlink(advertisement.c_str());
        unlink(socket_path.c_str());
        chmod(directory.c_str(), 0700);
        rmdir(directory.c_str());
    }
    void publish(const std::string& body) {
        const int fd = open(advertisement.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
        REQUIRE(fd >= 0);
        REQUIRE(write(fd, body.data(), body.size()) == static_cast<ssize_t>(body.size()));
        close(fd);
    }
    RequestOffer discover() const { return discover_requests(session, directory); }
    void no_connection() const {
        const int peer = accept(listener, nullptr, nullptr);
        const int error = errno;
        if (peer >= 0) close(peer);
        CHECK(peer == -1);
        CHECK((error == EAGAIN || error == EWOULDBLOCK));
    }
};
} // namespace

TEST_CASE("request discovery validates capability identities without process access") {
    RequestAdvertisement advertisement;
    CHECK(parse_request_advertisement(metadata("test", 100), "test", 100, &advertisement));
    CHECK(advertisement.pid == 100);
    CHECK(advertisement.session == "test");
    CHECK(request_offer(advertisement) == RequestOffer::Inspector);
    CHECK_FALSE(parse_request_advertisement(metadata("test", 100), "test", 101, &advertisement));
    CHECK_FALSE(parse_request_advertisement(metadata("test", 100), "other", 100, &advertisement));
    CHECK_FALSE(parse_request_advertisement(metadata("../test", 100), "../test", 100, &advertisement));
    CHECK_FALSE(parse_request_advertisement("[]", "test", 100, &advertisement));
    CHECK_FALSE(parse_request_advertisement("{broken}", "test", 100, &advertisement));
    CHECK_FALSE(parse_request_advertisement(metadata("test", 100, "[12]"), "test", 100, &advertisement));
    CHECK_FALSE(parse_request_advertisement(metadata("test", 100, "[\"\"]"), "test", 100, &advertisement));
    CHECK_FALSE(parse_request_advertisement(std::string(65537, ' '), "test", 100, &advertisement));
    CHECK(parse_request_advertisement(metadata("test", 100, "[\"echo\"]"), "test", 100, &advertisement));
    CHECK(request_offer(advertisement) == RequestOffer::Controls);
    CHECK(parse_request_advertisement(metadata("test", 100, "[]"), "test", 100, &advertisement));
    CHECK(request_offer(advertisement) == RequestOffer::None);
    CHECK_FALSE(parse_request_advertisement(
        "{\"protocol\":2,\"session\":\"test\",\"pid\":100,\"capabilities\":[]}", "test", 100, &advertisement));
    CHECK_FALSE(parse_request_advertisement(
        "{\"protocol\":1,\"session\":\"test\",\"pid\":100.5,\"capabilities\":[]}", "test", 100, &advertisement));
}

TEST_CASE("request discovery finds legacy controls and advertised inspectors without connecting") {
    Files files;
    CHECK(files.discover() == RequestOffer::Controls);
    files.publish(metadata(files.token, files.session.pid));
    CHECK(files.discover() == RequestOffer::Inspector);
    files.session.viewer_pid = 12345;
    auto entries = discover_request_entries({files.session}, files.directory);
    REQUIRE(entries.size() == 1);
    CHECK(entries.front().offer == RequestOffer::Inspector);
    CHECK(entries.front().session.script == "my_model.py");
    CHECK(entries.front().session.viewer_pid == 12345);
    CHECK_FALSE(can_attach(entries.front().session));
    files.no_connection();
    files.session.viewer_pid = 0;
    CHECK(can_attach(files.session));
    files.session.alive = false;
    CHECK(files.discover() == RequestOffer::None);
    files.session.starting = true;
    CHECK(files.discover() == RequestOffer::Inspector);
    files.publish(metadata(files.token, files.session.pid + 1));
    CHECK(files.discover() == RequestOffer::None);
    files.publish("invalid");
    CHECK(files.discover() == RequestOffer::None); // invalid metadata is not a legacy offer
}

TEST_CASE("request discovery refuses unsafe paths and permissions") {
    Files files;
    files.publish(metadata(files.token, files.session.pid));
    REQUIRE(chmod(files.advertisement.c_str(), 0644) == 0);
    CHECK(files.discover() == RequestOffer::None);
    REQUIRE(chmod(files.advertisement.c_str(), 0600) == 0);
    REQUIRE(chmod(files.socket_path.c_str(), 0666) == 0);
    CHECK(files.discover() == RequestOffer::None);
    REQUIRE(chmod(files.socket_path.c_str(), 0600) == 0);
    REQUIRE(chmod(files.directory.c_str(), 0755) == 0);
    CHECK(files.discover() == RequestOffer::None);
    REQUIRE(chmod(files.directory.c_str(), 0700) == 0);
    REQUIRE(unlink(files.advertisement.c_str()) == 0);
    REQUIRE(symlink("/dev/null", files.advertisement.c_str()) == 0);
    CHECK(files.discover() == RequestOffer::None);
    REQUIRE(unlink(files.advertisement.c_str()) == 0);
    files.publish(std::string(65537, 'x'));
    CHECK(files.discover() == RequestOffer::None);
    files.session.session = "../escape";
    CHECK(files.discover() == RequestOffer::None);
    files.no_connection();
}
