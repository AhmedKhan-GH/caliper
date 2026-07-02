#include "negotiation.h"
#include <cstdio>

namespace caliper_host {

int semver_cmp(const std::string& a, const std::string& b) {
    int av[3] = {0,0,0}, bv[3] = {0,0,0};
    std::sscanf(a.c_str(), "%d.%d.%d", &av[0], &av[1], &av[2]);
    std::sscanf(b.c_str(), "%d.%d.%d", &bv[0], &bv[1], &bv[2]);
    for (int i = 0; i < 3; i++)
        if (av[i] != bv[i]) return av[i] < bv[i] ? -1 : 1;
    return 0;
}

Negotiation negotiate(const AppletManifest& m, const HostCaps& caps) {
    Negotiation n;
    if (m.abi_epoch != caps.abi_epoch) {
        n.reason = "Built for ABI epoch " + std::to_string(m.abi_epoch) +
                   "; this host speaks " + std::to_string(caps.abi_epoch) +
                   " — check for an applet update.";
        return n;
    }
    if (!m.min_host.empty() && semver_cmp(caps.version, m.min_host) < 0) {
        n.reason = "Requires host " + m.min_host + " or newer; this host is " +
                   caps.version + ".";
        return n;
    }
    for (const auto& svc : m.required_services) {
        if (!caps.services.count(svc)) {
            n.reason = "Requires a capability this host doesn't have: " + svc + ".";
            return n;
        }
    }
    n.ok = true;
    return n;
}

} // namespace caliper_host
