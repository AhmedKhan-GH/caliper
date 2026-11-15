#pragma once
#include "applet_manifest.h"
#include <set>
#include <string>

namespace caliper_host {

struct HostCaps {
    uint32_t abi_epoch;
    std::string version;              // host semver, e.g. "0.6.0"
    std::set<std::string> services;   // ids this host can vend
    std::string torch_version;        // libtorch the host provides; "" = none
    std::string ui_stack_hash;        // pinned UI-stack hash; "" = not checked
};

struct Negotiation {
    bool ok = false;
    std::string reason;               // friendly card text when !ok
};

// The checks made before dlopen (docs/explanations/compatibility.md), in this
// order: ABI epoch, min_host, UI-stack hash, libtorch version, required
// services, data schema. `stored_data_schema` is the schema recorded in the
// applet's data folder (read_data_schema_stamp in applet_loader.h), 0 when
// none is recorded; it is compared only when the manifest declares a schema.
Negotiation negotiate(const AppletManifest& m, const HostCaps& caps,
                      uint32_t stored_data_schema = 0);

// Card text for a required service this host can't vend. One string, two
// callers: negotiate() for the manifest's list (pre-dlopen) and the loader for
// the descriptor's list (post-dlopen, the binary's own declaration).
std::string missing_service_reason(const std::string& svc);
// Card text for an applet whose data schema is older than the stored data's.
std::string data_schema_reason(uint32_t stored, uint32_t applet);

int semver_cmp(const std::string& a, const std::string& b); // <0, 0, >0

// The service ids this host serves (host_services.cpp vends a table for each;
// its service_ids() returns this set).
const std::set<std::string>& host_service_ids();
// The libtorch version and UI-stack hash this build provides, from the
// compile definitions CALIPER_HOST_TORCH_VERSION and
// CALIPER_HOST_UI_STACK_HASH; "" for one the build does not define.
const char* host_torch_version();
const char* host_ui_stack_hash();
// The version of the SDK built with this host, from the compile definition
// CALIPER_HOST_SDK_VERSION; "" when the build does not define it.
const char* host_sdk_version();
// The capabilities the caliper executable and the embed core advertise.
HostCaps host_caps();
}
