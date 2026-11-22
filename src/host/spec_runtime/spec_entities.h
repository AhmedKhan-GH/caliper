#pragma once
// The entities of a scene panel in the spec runtime
// (design/instrument-interface.md, Scene Entities).
//
// A SpecScene holds the batches that one scene panel builds from the spec's
// own sources: each entity under the reserved root "spec/" of the panel's
// scene, and the overlays of annotations under "spec-overlay/". The runtime
// writes an entity again when its source has a new generation, when the
// cursor moves or when a reference that its chain reads changes, at most
// once a frame.
//
// Threads: the frame thread only. Defined in spec_entities.cpp.
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <caliper_builder/session/resolve.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/spec/spec.h>

namespace caliper_scene {
struct Batch;  // scene/scene.h
}

namespace caliper_host {

class SpecScene {
public:
    SpecScene();
    ~SpecScene();
    SpecScene(const SpecScene&) = delete;
    SpecScene& operator=(const SpecScene&) = delete;

    // Writes the batches of the panel's entities from their resolved arrays
    // (PanelData::entities); an entity with an error shows nothing.
    void set_entities(const caliper_builder::spec::Panel& panel, const caliper_builder::PanelData& data);
    // An overlay batch at "spec-overlay/<path>": the outline of a threshold's
    // plane or a band's slab on the data box (spec_annotations.h).
    void set_overlay_batch(const std::string& path, const caliper_scene::Batch& batch);
    // Removes the batch at an entity's or an overlay's path.
    void clear(const std::string& path);
    // The pick fields of a hit on item `item` of the entity at `path`, or
    // empty when the path is not an entity of the panel.
    std::optional<caliper_builder::PickFields> pick_fields(const std::string& path, std::uint64_t item) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
