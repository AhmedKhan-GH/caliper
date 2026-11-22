// The catalog providers of the sources the app serves (app_providers.h).
#include "builder/app_providers.h"

#include "builder/app_session.h"

#if CALIPER_WITH_BUILDER
#include "catalog_providers/catalog_providers.h"
#include "host_services.h"
#include "live/live_attach.h"

#include <caliper_builder/session/session.h>

#include <iostream>
#endif

namespace caliper_host {

#if CALIPER_WITH_BUILDER

struct AppProviders::Impl {
    caliper_builder::Session* session = nullptr;   // the session the providers are registered with
    CatalogProviders providers;
};

AppProviders::AppProviders() : impl_(std::make_unique<Impl>()) {}

AppProviders::~AppProviders() = default;

void AppProviders::frame(AppSession& session, const caliper_live::Attachment& live) {
    Impl& im = *impl_;
    if (!im.session) {
        caliper_builder::Session* s = session.session();
        if (!s || session.worker_busy()) return;
        im.providers = register_catalog_providers(*s, &host_metrics_store(), &host_feed_store());
        im.session = s;
        if (!im.providers.error.ok())
            std::cerr << "[session] catalog providers: " << im.providers.error.message << std::endl;
    }
    if (im.providers.live) im.providers.live->update(live);
}

std::vector<std::string> AppProviders::dim_names(const std::string& catalog_id) const {
    const Impl& im = *impl_;
    return im.providers.live ? im.providers.live->dim_names(catalog_id) : std::vector<std::string>();
}

void AppProviders::close(AppSession& session) {
    Impl& im = *impl_;
    if (im.session && im.session == session.session()) unregister_catalog_providers(*im.session);
    im.providers = {};   // joins the metrics summary's thread
    im.session = nullptr;
}

#else  // !CALIPER_WITH_BUILDER

struct AppProviders::Impl {};
AppProviders::AppProviders() : impl_(std::make_unique<Impl>()) {}
AppProviders::~AppProviders() = default;
void AppProviders::frame(AppSession&, const caliper_live::Attachment&) {}
void AppProviders::close(AppSession&) {}
std::vector<std::string> AppProviders::dim_names(const std::string&) const { return {}; }

#endif

}  // namespace caliper_host
