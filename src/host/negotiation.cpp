#include "negotiation.h"
#include "host_version.h"
#include <caliper/abi.h>
#include <caliper/services/artifacts_v1.h>
#include <caliper/services/data_v1.h>
#include <caliper/services/device_v1.h>
#include <caliper/services/feed_v1.h>
#include <caliper/services/jobs_v1.h>
#include <caliper/services/log_v1.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/services/metrics_v1_1.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/ui_v1.h>
#include <caliper/services/ui_v1_1.h>
#include <cstdio>

// Defined by the build on the host library; empty when it does not.
#ifndef CALIPER_HOST_TORCH_VERSION
#define CALIPER_HOST_TORCH_VERSION ""
#endif
#ifndef CALIPER_HOST_UI_STACK_HASH
#define CALIPER_HOST_UI_STACK_HASH ""
#endif
#ifndef CALIPER_HOST_SDK_VERSION
#define CALIPER_HOST_SDK_VERSION ""
#endif

namespace caliper_host {

int semver_cmp(const std::string& a, const std::string& b) {
    int av[3] = {0,0,0}, bv[3] = {0,0,0};
    std::sscanf(a.c_str(), "%d.%d.%d", &av[0], &av[1], &av[2]);
    std::sscanf(b.c_str(), "%d.%d.%d", &bv[0], &bv[1], &bv[2]);
    for (int i = 0; i < 3; i++)
        if (av[i] != bv[i]) return av[i] < bv[i] ? -1 : 1;
    return 0;
}

namespace {
// A UI-stack hash is long; the card shows its first 12 characters, as a
// short commit id does.
std::string short_hash(const std::string& h) {
    return h.size() > 12 ? h.substr(0, 12) : h;
}
} // namespace

Negotiation negotiate(const InstrumentManifest& m, const HostCaps& caps,
                      uint32_t stored_data_schema) {
    Negotiation n;
    if (m.abi_epoch != caps.abi_epoch) {
        n.reason = "Built for ABI epoch " + std::to_string(m.abi_epoch) +
                   "; this host speaks " + std::to_string(caps.abi_epoch) +
                   " — check for an instrument update.";
        return n;
    }
    if (!m.min_host.empty() && semver_cmp(caps.version, m.min_host) < 0) {
        n.reason = "Requires host " + m.min_host + " or newer; this host is " +
                   caps.version + ".";
        return n;
    }
    if (!m.ui_stack.empty() && !caps.ui_stack_hash.empty() &&
        m.ui_stack != caps.ui_stack_hash) {
        n.reason = "Built against UI stack " + short_hash(m.ui_stack) +
                   "; this host uses " + short_hash(caps.ui_stack_hash) +
                   " — rebuild the instrument against this host's SDK.";
        return n;
    }
    if (!m.torch.empty() && m.torch != caps.torch_version) {
        n.reason = "Built against libtorch " + m.torch + "; this host provides " +
                   (caps.torch_version.empty() ? std::string("no libtorch")
                                               : caps.torch_version) +
                   ".";
        return n;
    }
    for (const auto& svc : m.required_services) {
        if (!caps.services.count(svc)) {
            n.reason = missing_service_reason(svc);
            return n;
        }
    }
    if (m.data_schema != 0 && m.data_schema < stored_data_schema) {
        n.reason = data_schema_reason(stored_data_schema, m.data_schema);
        return n;
    }
    n.ok = true;
    return n;
}

std::string missing_service_reason(const std::string& svc) {
    return "Requires a capability this host doesn't have: " + svc + ".";
}

std::string data_schema_reason(uint32_t stored, uint32_t instrument) {
    return "Stored data is schema " + std::to_string(stored) +
           "; this instrument reads schema " + std::to_string(instrument) + ".";
}

const std::set<std::string>& host_service_ids() {
    static const std::set<std::string> ids = {
        CALIPER_UI_V1,
        CALIPER_UI_V1_1,
        CALIPER_LOG_V1,
        CALIPER_JOBS_V1,
        CALIPER_DEVICE_V1,
        CALIPER_METRICS_V1,
        CALIPER_METRICS_V1_1,
        CALIPER_TENSOR_BRIDGE_V1,
        CALIPER_TENSOR_BRIDGE_V1_1,
        CALIPER_TENSOR_BRIDGE_V1_2,
        CALIPER_ARTIFACTS_V1,
        CALIPER_DATA_V1,
        CALIPER_FEED_V1,
    };
    return ids;
}

const char* host_torch_version() { return CALIPER_HOST_TORCH_VERSION; }
const char* host_ui_stack_hash() { return CALIPER_HOST_UI_STACK_HASH; }
const char* host_sdk_version() { return CALIPER_HOST_SDK_VERSION; }

HostCaps host_caps() {
    return HostCaps{CALIPER_ABI_EPOCH, kHostVersionStr, host_service_ids(),
                    host_torch_version(), host_ui_stack_hash()};
}

} // namespace caliper_host
