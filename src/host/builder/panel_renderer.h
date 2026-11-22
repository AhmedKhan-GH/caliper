#pragma once
// ============================================================================
// panel_renderer.h: the seam of panel.render (app_session.h). The app
// implements PanelRenderSeam over the spec runtime's OffscreenPanelRenderer
// (spec_page.h) and gives it to AppSession::set_panel_renderer(). It is not
// called PanelRenderer, the name of the runtime's own interface
// (spec_runtime/spec_offscreen.h), which returns a PNG rather than a result.
// The session has already found the panel in the open spec and resolved its
// data (caliper_builder::Session::resolve()), which may carry a resolve error
// that the renderer draws as the panel's state. render_panel() runs on the
// frame thread, draws that data off screen at width x height pixels and
// answers {"png_base64", "width", "height"}, or panel.render_failed. Until a
// renderer is set, panel.render answers command.not_available. Built only
// with CALIPER_WITH_BUILDER.
// ============================================================================
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/spec.h>

namespace caliper_host {

class PanelRenderSeam {
public:
    virtual ~PanelRenderSeam() = default;
    virtual caliper_builder::CommandOutput render_panel(const caliper_builder::spec::Spec& spec,
                                                        const caliper_builder::PanelData& data, int width,
                                                        int height) = 0;
};

}  // namespace caliper_host
