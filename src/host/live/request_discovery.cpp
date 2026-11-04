#include "live/request_discovery.h"

#include <crude_json.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace caliper_live {
namespace {
using Json = crude_json::value;
constexpr size_t max_advertisement_bytes = 65536;

bool valid_token(const std::string& token) {
    return !token.empty() && token.size() <= 20 &&
        token.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") == std::string::npos;
}
const Json& field(const Json& value, const char* name) {
    static const Json empty;
    return value.is_object() && value.contains(name) ? value[name] : empty;
}
bool integer(const Json& value, int32_t* result) {
    if (!value.is_number()) return false;
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::floor(number) != number || number < 1 ||
        number > std::numeric_limits<int32_t>::max()) return false;
    *result = static_cast<int32_t>(number);
    return true;
}
#ifndef _WIN32
bool private_owner(const struct stat& info) {
    return info.st_uid == getuid() && (info.st_mode & 0077) == 0;
}
bool read_advertisement(const std::string& path, std::string* body) {
    const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat info{};
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || !private_owner(info) ||
        info.st_size <= 0 || static_cast<uint64_t>(info.st_size) > max_advertisement_bytes) {
        close(fd);
        return false;
    }
    body->clear();
    char buffer[4096];
    for (;;) {
        const ssize_t count = read(fd, buffer, sizeof buffer);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) { close(fd); return false; }
        if (count == 0) break;
        body->append(buffer, static_cast<size_t>(count));
        if (body->size() > max_advertisement_bytes) { close(fd); return false; }
    }
    close(fd);
    return true;
}
#endif
} // namespace

bool parse_request_advertisement(const std::string& body, const std::string& session,
                                  int32_t pid, RequestAdvertisement* out) {
    if (!out || !valid_token(session) || pid <= 0 || body.size() > max_advertisement_bytes) return false;
    const Json json = Json::parse(body);
    const Json& token = field(json, "session");
    const Json& capabilities = field(json, "capabilities");
    RequestAdvertisement next;
    if (!integer(field(json, "protocol"), &next.protocol) || next.protocol != 1 ||
        !integer(field(json, "pid"), &next.pid) || next.pid != pid ||
        !token.is_string() || token.get<std::string>() != session ||
        !capabilities.is_array() || capabilities.get<crude_json::array>().size() > 128) return false;
    next.session = session;
    for (const Json& capability : capabilities.get<crude_json::array>()) {
        if (!capability.is_string()) return false;
        const std::string& name = capability.get<std::string>();
        if (name.empty() || name.size() > 128) return false;
        next.capabilities.push_back(name);
    }
    *out = std::move(next);
    return true;
}

RequestOffer request_offer(const RequestAdvertisement& advertisement) {
    if (advertisement.protocol != 1 || advertisement.pid <= 0 || !valid_token(advertisement.session))
        return RequestOffer::None;
    const auto has = [&](const char* name) {
        return std::find(advertisement.capabilities.begin(), advertisement.capabilities.end(), name) !=
               advertisement.capabilities.end();
    };
    if (has("generate") && has("list_tensors") && has("show_tensor")) return RequestOffer::Inspector;
    return advertisement.capabilities.empty() ? RequestOffer::None : RequestOffer::Controls;
}

RequestOffer discover_requests(const SessionEntry& session, const std::string& directory) {
#ifndef _WIN32
    if (!valid_token(session.session) || session.pid <= 0 || (!session.alive && !session.starting))
        return RequestOffer::None;
    const std::string root = directory.empty() ? "/tmp/caliper-" + std::to_string(getuid()) : directory;
    struct stat info{};
    if (lstat(root.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) || !private_owner(info)) return RequestOffer::None;
    const std::string stem = root + "/" + session.session;
    if (lstat((stem + ".sock").c_str(), &info) != 0 || !S_ISSOCK(info.st_mode) || !private_owner(info))
        return RequestOffer::None;
    const std::string metadata = stem + ".requests.json";
    if (lstat(metadata.c_str(), &info) != 0)
        return errno == ENOENT ? RequestOffer::Controls : RequestOffer::None;
    if (!S_ISREG(info.st_mode) || !private_owner(info)) return RequestOffer::None;
    std::string body;
    RequestAdvertisement advertisement;
    if (!read_advertisement(metadata, &body) ||
        !parse_request_advertisement(body, session.session, session.pid, &advertisement)) return RequestOffer::None;
    return request_offer(advertisement);
#else
    (void)session; (void)directory;
    return RequestOffer::None;
#endif
}

std::vector<RequestEntry> discover_request_entries(const std::vector<SessionEntry>& sessions,
                                                   const std::string& directory) {
    std::vector<RequestEntry> entries;
    for (const auto& session : sessions) {
        const RequestOffer offer = discover_requests(session, directory);
        if (offer != RequestOffer::None) entries.push_back({session, offer});
    }
    return entries;
}

} // namespace caliper_live
