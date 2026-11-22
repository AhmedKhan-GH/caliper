// The entities of a scene panel (spec_entities.h). Not built yet: the
// validator refuses entities (spec.not_built), so a SpecScene holds nothing.
#include "spec_runtime/spec_entities.h"

namespace caliper_host {

struct SpecScene::Impl {};

SpecScene::SpecScene() : impl_(std::make_unique<Impl>()) {}
SpecScene::~SpecScene() = default;

void SpecScene::set_entities(const caliper_builder::spec::Panel& panel, const caliper_builder::PanelData& data) {
    (void)panel;
    (void)data;
}

void SpecScene::set_overlay_batch(const std::string& path, const caliper_scene::Batch& batch) {
    (void)path;
    (void)batch;
}

void SpecScene::clear(const std::string& path) { (void)path; }

std::optional<caliper_builder::PickFields> SpecScene::pick_fields(const std::string& path, std::uint64_t item) const {
    (void)path;
    (void)item;
    return std::nullopt;
}

}  // namespace caliper_host
