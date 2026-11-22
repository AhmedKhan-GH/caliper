// catalog_providers.h: the one call that gives a builder session the sources
// the app serves (design/builder-session.md, the wiring): the catalog
// providers of live views, feed channels and metrics, and the data sources
// that let Session::resolve() read them. Host-internal; built only with
// CALIPER_WITH_BUILDER.
//
// The app, after it opens its session:
//
//   providers_ = caliper_host::register_catalog_providers(*session,
//                    &caliper_host::host_metrics_store(), &caliper_host::host_feed_store());
//
// once per frame, after the live poll and whatever page is shown (so that a
// detach is seen):
//
//   providers_.live->update(live_);   // live_ is the app's caliper_live::Attachment
//
// and before services_shutdown() closes the stores:
//
//   caliper_host::unregister_catalog_providers(*session);
//   providers_ = {};                  // joins the metrics summary's thread
//
// The providers are named `live`, `feed` and `metrics`, at priority 0 (their
// ids never meet the loader outputs', whose kinds are table and tensor). A
// null store leaves its kind out.
#pragma once

#include "catalog_providers/feed_catalog.h"
#include "catalog_providers/live_catalog.h"
#include "catalog_providers/metrics_catalog.h"

#include <caliper_builder/error.h>

#include <memory>

namespace caliper_builder { class Session; }

namespace caliper_host {

class FeedStore;
class MetricsStore;

struct CatalogProviders {
    std::shared_ptr<LiveCatalog> live;        // always made; the app calls live->update() every frame
    std::shared_ptr<FeedCatalog> feed;        // null without a feed store
    std::shared_ptr<MetricsCatalog> metrics;  // null without a metrics store
    caliper_builder::Error error;             // a provider the catalog refused (a name already taken)
};

CatalogProviders register_catalog_providers(caliper_builder::Session& session, MetricsStore* metrics,
                                            FeedStore* feed, LiveCatalog::Options live_options = {},
                                            MetricsCatalog::Options metrics_options = {});

// Removes the three providers and data sources from the session again.
void unregister_catalog_providers(caliper_builder::Session& session);

}  // namespace caliper_host
