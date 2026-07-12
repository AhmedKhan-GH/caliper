#include <caliper/caliper.hpp>
#include <caliper/services/feed_v1.h>   // CALIPER_FEED_V1 (not in caliper.hpp)
#include "pulse_scope.h"

class PulseScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    pulsescope::PulseScopeApplet impl_;
};

CALIPER_APPLET(PulseScopePlugin,
    .id       = "dev.caliper.pulse-scope",
    .version  = "0.1.0",
    .name     = "PulseScope",
    .summary  = "This machine's own vitals, live: PulseScope enumerates every "
                "caliper.feed.v1 channel the host vends (CPU/GPU utilization, "
                "memory, thermal state, fan, temperature, power on this Mac) and "
                "draws one input-locked strip chart per channel from a job-thread "
                "poller. Current value + units, a climbing \"last sample N.N s "
                "ago\" staleness label, and a cumulative \"lost M samples\" gap "
                "counter make the honest-loss contract visible. No sensors on "
                "this host -> one honest line, nothing faked. CPU + ImPlot only.",
    .tag      = "Telemetry",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1, CALIPER_FEED_V1})
