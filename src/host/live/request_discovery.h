#pragma once

#include "live/live_source.h"

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_live {

// Local request discovery never opens a connection. A controller may already
// occupy the socket, so even a hello probe would interfere with a live session.
enum class RequestOffer { None, Controls, Inspector };
struct RequestAdvertisement {
    int32_t protocol = 0;
    int32_t pid = 0;
    std::string session;
    std::vector<std::string> capabilities;
};
struct RequestEntry {
    SessionEntry session;
    RequestOffer offer = RequestOffer::None;
};

// Pure validation and classification, also used by the filesystem reader.
bool parse_request_advertisement(const std::string& json, const std::string& session,
                                  int32_t pid, RequestAdvertisement* out);
RequestOffer request_offer(const RequestAdvertisement& advertisement);

// `directory` is normally /tmp/caliper-<uid>; tests can supply a private dir.
// The directory, socket and metadata must belong to the current user and have
// no group/other permissions. A private socket without metadata is a legacy
// Controls offer; an invalid metadata file is never treated as a legacy host.
RequestOffer discover_requests(const SessionEntry& session,
                                const std::string& directory = {});
std::vector<RequestEntry> discover_request_entries(const std::vector<SessionEntry>& sessions,
                                                   const std::string& directory = {});

} // namespace caliper_live
