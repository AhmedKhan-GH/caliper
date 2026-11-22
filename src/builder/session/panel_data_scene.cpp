// The entities of a scene panel in panel.data (panel_data.h). Not built
// yet: the validator refuses entities (spec.not_built), so a scene panel's
// data holds only its live scene.
#include <caliper_builder/session/panel_data.h>

namespace caliper_builder {

void encode_scene_entities(const spec::Panel& panel, const PanelData& data, const PanelDataLimits& limits,
                           nlohmann::json* out) {
    (void)panel;
    (void)data;
    (void)limits;
    (void)out;
}

}  // namespace caliper_builder
