#pragma once
// MeshScope — geometry.v1_1 exemplar: a live deforming height-field mesh drawn
// from imported tensor allocations as triangles plus a wireframe overlay.
#include <memory>

namespace caliper { class Host; }

namespace meshscope {

struct MeshScopeState;

class MeshScopeApplet {
public:
    MeshScopeApplet();
    ~MeshScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();

private:
    std::unique_ptr<MeshScopeState> s_;
};

}  // namespace meshscope
