#include "catalog_providers/catalog_providers.h"

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
