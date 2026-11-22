#include "catalog_providers/catalog_providers.h"

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/session.h>

namespace caliper_host {

using namespace caliper_builder;

CatalogProviders register_catalog_providers(Session& session, MetricsStore* metrics, FeedStore* feed,
                                            LiveCatalog::Options live_options,
                                            MetricsCatalog::Options metrics_options) {
    CatalogProviders out;
    auto keep = [&](Error e) {
        if (e && out.error.ok()) out.error = std::move(e);
    };
    // The live catalog keeps the session's steps (design/instrument-interface.md,
    // Step History): it appends each snapshot of a kept view to the history at
    // its step stamp, else at the run's step when it arrived.
    if (!live_options.history) live_options.history = &session.history();
    if (!live_options.events) live_options.events = &session.events();
    if (!live_options.run_step)
        live_options.run_step = [&session]() -> std::optional<std::int64_t> {
            RunControl* run = session.run_control();
            if (!run) return std::nullopt;
            return run->state().step;
        };
    out.live = std::make_shared<LiveCatalog>(std::move(live_options));
    keep(session.catalog().add_provider("live", out.live));
    session.data_sources().set(CatalogKind::live, out.live);
    if (feed) {
        out.feed = std::make_shared<FeedCatalog>(*feed);
        keep(session.catalog().add_provider("feed", out.feed));
        session.data_sources().set(CatalogKind::feed, out.feed);
    }
    if (metrics) {
        out.metrics = std::make_shared<MetricsCatalog>(*metrics, metrics_options);
        keep(session.catalog().add_provider("metrics", out.metrics));
        session.data_sources().set(CatalogKind::metric, out.metrics);
    }
    return out;
}

void unregister_catalog_providers(Session& session) {
    for (const char* name : {"live", "feed", "metrics"}) session.catalog().remove_provider(name);
    for (CatalogKind kind : {CatalogKind::live, CatalogKind::feed, CatalogKind::metric})
        session.data_sources().set(kind, nullptr);
}

}  // namespace caliper_host
