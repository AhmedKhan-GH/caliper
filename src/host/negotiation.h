#pragma once
#include "applet_manifest.h"
#include <set>
#include <string>

namespace caliper_host {

struct HostCaps {
    uint32_t abi_epoch;
    std::string version;              // host semver, e.g. "0.6.0"
    std::set<std::string> services;   // ids this host can vend
};

struct Negotiation {
    bool ok = false;
    std::string reason;               // friendly card text when !ok
};

// PLATFORM.md §14 order (Phase-1 subset — packs/platform checks arrive
// Phase 4): epoch supported → min_host satisfied → required services present.
Negotiation negotiate(const AppletManifest& m, const HostCaps& caps);

int semver_cmp(const std::string& a, const std::string& b); // <0, 0, >0
}
