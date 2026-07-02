#pragma once
// caliper_host::render_runs_dashboard — host UI over the tested MetricsStore
// readers (runs(), scalar_tags(), scalars()). This is glue: it lives in the
// `caliper` executable (not caliper_host_lib) because it pulls in ImGui/ImPlot.
//
// Draws a Runs window while *p_open: a left run-list pane (experiment/name,
// ● while a run is not done), and a right pane with the selected run's hparams
// plus one ImPlot per scalar tag, each with an EMA smoothing slider (0–0.99)
// that overlays a smoothed line on the faint raw series. Queries run per-frame
// only while the window is open (the store serializes internally).
namespace caliper_host {

class MetricsStore;

void render_runs_dashboard(MetricsStore& store, bool* p_open);

}  // namespace caliper_host
