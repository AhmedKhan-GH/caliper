// The draw half of a program in a viewer (draw_runtime.h). Not built yet:
// decode() and install() answer program.not_built, and the runtime draws
// nothing, until the work of design/programmable-graphics.md fills it.
#include "draw/draw_runtime.h"

namespace caliper_draw {

namespace cp = caliper_builder::programs;

namespace {

Verdict not_built(const char* what) {
    Verdict v;
    v.code = cp::kProgramNotBuilt;
    v.message = std::string(what) + " of a program's draw half is not built yet.";
    return v;
}

}  // namespace

struct DrawRuntime::Impl {
    caliper_gfx::Backend* gfx = nullptr;
    bool webgpu = false;
    DrawStatus status;
};

DrawRuntime::DrawRuntime(caliper_gfx::Backend& gfx, bool webgpu) : impl_(std::make_unique<Impl>()) {
    impl_->gfx = &gfx;
    impl_->webgpu = webgpu;
}

DrawRuntime::~DrawRuntime() = default;

Verdict DrawRuntime::decode(std::string_view, std::string_view, std::string_view, caliper_builder::spec::Piece* piece,
                            Instance* instance) {
    if (piece) *piece = caliper_builder::spec::Piece{};
    if (instance) *instance = Instance{};
    return not_built("The decoder");
}

Verdict DrawRuntime::install(const caliper_builder::spec::Piece&, const Instance&, std::uint32_t) {
    impl_->status = DrawStatus{};
    impl_->status.state = DrawState::refused;
    impl_->status.error = not_built("The render graph");
    return impl_->status.error;
}

void DrawRuntime::remove() { impl_->status = DrawStatus{}; }

Verdict DrawRuntime::set_resource(const ResourceVersion&) { return not_built("A resource"); }

bool DrawRuntime::encode(caliper_gfx::Frame&, const FrameInputs&, const HostTarget&) { return false; }

void DrawRuntime::frame_done(double) {}

void DrawRuntime::frame_pending(double) {}

std::optional<PickedItem> DrawRuntime::pick(std::uint32_t) const { return std::nullopt; }

std::vector<Label> DrawRuntime::labels() const { return {}; }

DrawStatus DrawRuntime::status() const { return impl_->status; }

}  // namespace caliper_draw
