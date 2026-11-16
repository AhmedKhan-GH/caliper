#pragma once
// ============================================================================
// app_providers.h: the catalog providers of the sources the app serves, for
// the app's builder session (src/host/catalog_providers/catalog_providers.h,
// design/builder-session.md, The session in the app). Once the session has
// been adopted, frame() registers the providers of live views, feed channels
// and metrics over the host's metrics and feed stores, and from then on hands
// the live provider the app's attachment every frame, on every page, after
// the live poll and the page's draw, so that a new snapshot, an attach and a
// detach are seen. close() removes them before the session closes, and so
// before the host's stores shut down.
//
// Frame thread only. While the session's worker runs a long command the
// providers are not registered yet (the catalog is the worker's then); the
// live update itself touches only the provider. Built with
// CALIPER_WITH_BUILDER; without it both calls do nothing.
// ============================================================================
#include <memory>
#include <string>
#include <vector>

namespace caliper_live {
class Attachment;
}

namespace caliper_host {

class AppSession;

class AppProviders {
public:
    AppProviders();
    ~AppProviders();
    AppProviders(const AppProviders&) = delete;
    AppProviders& operator=(const AppProviders&) = delete;

    void frame(AppSession& session, const caliper_live::Attachment& live);
    void close(AppSession& session);
    // The names of a live view's dims ("live:attn": layer, query, key), one
    // per dim of its catalog shape, as the stream's payload axes give them;
    // empty for any other source and for a view whose axes carry none.
    std::vector<std::string> dim_names(const std::string& catalog_id) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
