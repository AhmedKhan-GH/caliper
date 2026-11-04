// CPU-only cache and lazy-loading contract. No Metal device or model is opened.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/live_model3d.h"

#include <imgui_internal.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
using caliper_live::Model3D;
using caliper_live::ModelTensor;

ModelTensor matrix(const std::string& name, uint64_t revision = 0) {
    ModelTensor tensor;
    tensor.name = name;
    tensor.shape = {4, 3};
    tensor.rows = 4;
    tensor.cols = 3;
    tensor.revision = revision;
    return tensor;
}

ModelTensor vector(const std::string& name) {
    ModelTensor tensor;
    tensor.name = name;
    tensor.shape = {1024};
    tensor.rows = 1024;
    tensor.cols = 1;
    return tensor;
}

struct Ui {
    int errors = 0;
    std::string captured;

    Ui() {
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.DisplaySize = ImVec2(1000, 700);
        io.DeltaTime = 1.0f / 60.0f;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.ConfigErrorRecoveryEnableAssert = false;
        io.ConfigErrorRecoveryEnableTooltip = false;
        io.ConfigErrorRecoveryEnableDebugLog = false;
        ImGui::GetCurrentContext()->ErrorCallback = [](ImGuiContext*, void* userdata, const char*) {
            ++static_cast<Ui*>(userdata)->errors;
        };
        ImGui::GetCurrentContext()->ErrorCallbackUserData = this;
    }

    ~Ui() {
        CHECK(errors == 0);
        if (ImGui::GetCurrentContext()->WithinFrameScope) ImGui::EndFrame();
        ImGui::DestroyContext();
    }

    std::string draw_frame(Model3D& model, const std::string& selected = {}, bool collapsed = false,
                           float width = 900, float height = 600) {
        ImGui::GetIO().DisplaySize = ImVec2(width + 100, height + 100);
        if (ImGui::FindWindowByName("Model 3D###caliper.inspector.model3d"))
            ImGui::SetWindowSize("Model 3D###caliper.inspector.model3d", ImVec2(width, height));
        ImGui::NewFrame();
        ImGui::SetNextWindowCollapsed(collapsed, ImGuiCond_Always);
        ImGui::LogToBuffer();
        const auto clicked = model.draw(nullptr, 0, 0, width, height, selected);
        captured = ImGui::GetCurrentContext()->LogBuffer.c_str();
        ImGui::LogFinish();
        ImGui::Render();
        // Service ImGui's font-atlas lifecycle without uploading a GPU texture.
        for (ImTextureData* texture : ImGui::GetPlatformIO().Textures) {
            if (texture->Status == ImTextureStatus_WantCreate || texture->Status == ImTextureStatus_WantUpdates) {
                texture->SetTexID(static_cast<ImTextureID>(1));
                texture->SetStatus(ImTextureStatus_OK);
            } else if (texture->Status == ImTextureStatus_WantDestroy) {
                texture->SetTexID(ImTextureID_Invalid);
                texture->SetStatus(ImTextureStatus_Destroyed);
            }
        }
        return clicked;
    }

    void frame(Model3D& model, const std::string& selected = {}, bool collapsed = false,
               float width = 900, float height = 600) {
        CHECK(draw_frame(model, selected, collapsed, width, height).empty());
    }

    void activate(Model3D& model, const std::string& selected, const char* label) {
        auto* window = ImGui::FindWindowByName("Model 3D###caliper.inspector.model3d");
        REQUIRE(window != nullptr);
        ImGui::FocusWindow(window);
        ImGui::ActivateItemByID(window->GetID(label));
        frame(model, selected);
    }
};

ModelTensor embedding() {
    ModelTensor tensor = matrix("embedding");
    tensor.shape = {248320, 1024}; tensor.rows = 248320; tensor.cols = 1024;
    tensor.role = "embedding"; tensor.tied = true;
    return tensor;
}

std::vector<ModelTensor> framing_tensors() {
    auto a = matrix("layer.0.matrix"), b = matrix("layer.1.matrix");
    a.shape = {1024, 3584}; a.rows = 1024; a.cols = 3584;
    a.layer_index = 0; a.layer_type = "linear_attention";
    b.shape = {3584, 1024}; b.rows = 3584; b.cols = 1024;
    b.layer_index = 1; b.layer_type = "full_attention";
    return {embedding(), a, b};
}
} // namespace

TEST_CASE("model3d cache accepts only bounded finite thumbnails") {
    Model3D model;
    ModelTensor large = matrix("matrix");
    large.shape = {256, 256};
    large.rows = large.cols = 256;
    model.set_tensors({large, vector("vector")});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    CHECK_FALSE(model.set_thumbnail("missing", 1, 1, {0}, 0, 1));
    CHECK_FALSE(model.set_thumbnail("matrix", 0, 1, {}, 0, 1));
    CHECK_FALSE(model.set_thumbnail("matrix", 1, 0, {}, 0, 1));
    CHECK_FALSE(model.set_thumbnail("matrix", 129, 1, std::vector<float>(129), 0, 1));
    CHECK_FALSE(model.set_thumbnail("matrix", 1, 129, std::vector<float>(129), 0, 1));
    CHECK_FALSE(model.set_thumbnail("matrix", 2, 2, {0, 1, 2}, 0, 2));
    CHECK_FALSE(model.set_thumbnail("matrix", 1, 1, {nan}, 0, 1));
    CHECK_FALSE(model.set_thumbnail("matrix", 1, 1, {0}, -infinity, 1));
    CHECK_FALSE(model.set_thumbnail("matrix", 1, 1, {0}, 0, infinity));
    CHECK_FALSE(model.set_thumbnail("matrix", 1, 1, {0}, 1, 0));
    CHECK_FALSE(model.set_thumbnail("vector", 128, 2, std::vector<float>(256), 0, 1));

    CHECK(model.set_thumbnail("matrix", 128, 128, std::vector<float>(128 * 128, 0.5f), 0, 1));
    CHECK(model.set_thumbnail("vector", 128, 1, std::vector<float>(128, 1.0f), 1, 1));
    CHECK(model.next_thumbnail().empty()); // A cache update never opens the window.
}

TEST_CASE("model3d requests visible thumbnails once and prioritizes selection") {
    Ui ui;
    Model3D model;
    model.set_tensors({matrix("first"), matrix("selected"), vector("last")});

    CHECK(model.next_thumbnail().empty());
    ui.frame(model, "selected", true);
    CHECK(model.next_thumbnail().empty());
    ui.frame(model, "selected");
    CHECK(model.next_thumbnail() == "selected");
    CHECK(model.next_thumbnail() == "first");
    CHECK(model.next_thumbnail() == "last");
    CHECK(model.next_thumbnail().empty());

    // A failed request is not continuously retried while that manifest remains.
    model.thumbnail_failed("selected");
    CHECK(model.next_thumbnail().empty());
    ui.frame(model, "selected", true);
    CHECK(model.next_thumbnail().empty());
}

TEST_CASE("model3d manifest refresh retries unfinished samples and preserves complete samples") {
    Ui ui;
    Model3D model;
    const std::vector<ModelTensor> tensors{matrix("complete"), matrix("interrupted"), matrix("failed")};
    model.set_tensors(tensors);
    CHECK(model.set_thumbnail("complete", 1, 1, {0.25f}, 0.25, 0.25));
    ui.frame(model);
    CHECK(model.next_thumbnail() == "interrupted");
    CHECK(model.next_thumbnail() == "failed");
    model.thumbnail_failed("failed");
    CHECK(model.next_thumbnail().empty());

    // Reconnection obtains another manifest. Its unfinished requests must be
    // admitted again, while the already cached tensor must not be downloaded.
    model.set_tensors(tensors);
    CHECK(model.next_thumbnail() == "interrupted");
    CHECK(model.set_thumbnail("interrupted", 1, 1, {0.5f}, 0.5, 0.5));
    CHECK(model.next_thumbnail() == "failed");
    CHECK(model.set_thumbnail("failed", 1, 1, {0.75f}, 0.75, 0.75));
    CHECK(model.next_thumbnail().empty());

    model.invalidate("complete", true);
    CHECK(model.next_thumbnail() == "complete");
    CHECK(model.set_thumbnail("complete", 1, 1, {1.0f}, 1, 1));
    CHECK(model.next_thumbnail().empty());
}

TEST_CASE("model3d invalidates changed revisions and removes absent tensors") {
    Ui ui;
    Model3D model;
    ModelTensor tensor = matrix("weight", 3);
    model.set_tensors({tensor});
    ui.frame(model, tensor.name);
    CHECK(model.next_thumbnail() == tensor.name);
    CHECK_FALSE(model.set_thumbnail(tensor.name, 1, 1, {0}, 0, 0, 2));
    CHECK(model.next_thumbnail() == tensor.name);
    CHECK(model.set_thumbnail(tensor.name, 1, 1, {0}, 0, 0, 3));
    CHECK(model.next_thumbnail().empty());

    tensor.revision = 4;
    tensor.edited = true;
    model.set_tensors({tensor});
    CHECK(model.next_thumbnail() == tensor.name);
    CHECK_FALSE(model.set_thumbnail(tensor.name, 1, 1, {0}, 0, 0, 3));
    CHECK(model.set_thumbnail(tensor.name, 1, 1, {1}, 1, 1, 4));
    CHECK(model.next_thumbnail().empty());

    model.set_tensors({matrix("replacement")});
    CHECK_FALSE(model.set_thumbnail(tensor.name, 1, 1, {0}, 0, 0, 4));
    CHECK(model.next_thumbnail() == "replacement");
}

TEST_CASE("model3d folded thumbnail atlas follows original row blocks with bounded storage") {
    using namespace caliper_live::model3d;
    const auto tensor = embedding();
    REQUIRE(row_blocks(tensor) == 16);
    const auto shape = thumbnail_shape(tensor);
    REQUIRE(shape[0] == 2048); REQUIRE(shape[1] == 8);
    std::vector<float> source(static_cast<size_t>(shape[0]) * shape[1]);
    for (int r = 0; r < shape[0]; ++r) for (int c = 0; c < shape[1]; ++c)
        source[static_cast<size_t>(r) * shape[1] + c] = static_cast<float>(r * 100 + c);
    ThumbnailAtlas atlas;
    REQUIRE(thumbnail_atlas(tensor, shape[0], shape[1], source, &atlas));
    REQUIRE(atlas.width == 128); REQUIRE(atlas.height == 128);
    for (int block = 0; block < 16; ++block) for (int y : {0, 17, 127}) for (int c : {0, 3, 7}) {
        CHECK(atlas.values[static_cast<size_t>(y) * atlas.width + block * 8 + c] ==
              source[static_cast<size_t>(block * 128 + y) * 8 + c]);
    }

    // A legacy host's square sample is resampled within each folded block.
    std::vector<float> legacy(128 * 128);
    for (size_t i = 0; i < legacy.size(); ++i) legacy[i] = static_cast<float>(i);
    REQUIRE(thumbnail_atlas(tensor, 128, 128, legacy, &atlas));
    CHECK(atlas.width == 128); CHECK(atlas.height == 8);
    CHECK(atlas.values.front() == legacy[8]);
    CHECK(atlas.values[8] == legacy[8 * 128 + 8]);
    CHECK(atlas.values.back() == legacy[127 * 128 + 120]);
    CHECK(atlas.values.size() <= 128 * 128);
    CHECK_FALSE(thumbnail_atlas(tensor, 2048, 128, std::vector<float>(2048 * 128), &atlas));
    CHECK_FALSE(thumbnail_atlas(tensor, 2049, 1, std::vector<float>(2049), &atlas));

    Model3D model;
    model.set_tensors({tensor});
    CHECK(model.set_thumbnail(tensor.name, shape[0], shape[1], source, 0, source.back()));
}

TEST_CASE("model3d Focus uses the final scene viewport and preserves complete panels on resize") {
    using namespace caliper_live::model3d;
    Ui ui;
    Model3D model;
    const auto tensors = framing_tensors();
    const auto scene = layout(tensors);
    model.set_tensors(tensors);
    const std::string selected = tensors[1].name;
    ui.frame(model, selected);
    ui.activate(model, selected, "Focus selected");
    const auto panel = std::find_if(scene.panels.begin(), scene.panels.end(), [](const auto& p) { return p.tensor == 1; });
    REQUIRE(panel != scene.panels.end());
    for (const ImVec2 size : {ImVec2(900, 600), ImVec2(520, 850), ImVec2(1500, 450)}) {
        ui.frame(model, selected, false, size.x, size.y);
        const auto state = model.view_state();
        REQUIRE(state.viewport.width > 0); REQUIRE(state.viewport.height > 0);
        CHECK(state.viewport.width < size.x - 50); // The label column is excluded.
        const double aspect = state.viewport.width / state.viewport.height;
        for (const auto corner : corners(*panel)) {
            const auto p = project(corner, state.camera, aspect, state.viewport);
            CHECK(p.visible);
            CHECK(p.x >= state.viewport.x + state.viewport.width * .04);
            CHECK(p.x <= state.viewport.x + state.viewport.width * .96);
            CHECK(p.y >= state.viewport.y + state.viewport.height * .04);
            CHECK(p.y <= state.viewport.y + state.viewport.height * .96);
        }
    }
}

TEST_CASE("model3d panel hover and click share scene geometry and reject drag or label-column presses") {
    using namespace caliper_live::model3d;
    Ui ui;
    Model3D model;
    const auto tensors = framing_tensors();
    const auto scene = layout(tensors);
    model.set_tensors(tensors);
    const auto selected = tensors[1].name;
    ui.frame(model, selected);
    ui.activate(model, selected, "Focus selected");
    const auto panel = std::find_if(scene.panels.begin(), scene.panels.end(), [](const auto& p) { return p.tensor == 1; });
    REQUIRE(panel != scene.panels.end());
    const auto center = [&] {
        const auto state = model.view_state();
        const auto p = project(panel->center, state.camera, state.viewport.width / state.viewport.height, state.viewport);
        REQUIRE(p.visible);
        const auto hit = hit_test(scene, state.camera, state.viewport, p.x, p.y);
        REQUIRE(hit.panel >= 0);
        REQUIRE(scene.panels[static_cast<size_t>(hit.panel)].tensor == 1);
        return ImVec2(static_cast<float>(p.x), static_cast<float>(p.y));
    };
    auto& io = ImGui::GetIO();
    ImVec2 point = center();
    io.AddMousePosEvent(point.x, point.y); ui.frame(model, selected);
    CHECK(ui.captured.find(selected) != std::string::npos);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); ui.frame(model, selected);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    CHECK(ui.draw_frame(model, selected) == selected);

    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); ui.frame(model, selected);
    io.AddMousePosEvent(point.x + 30, point.y + 15); ui.frame(model, selected);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); ui.frame(model, selected);

    ui.activate(model, selected, "Fit");
    point = center();
    const auto state = model.view_state();
    REQUIRE(state.labels.width > 0); REQUIRE(state.labels.height > 0);
    const double overlap = std::min(state.labels.width, state.viewport.x + state.viewport.width - state.labels.x);
    REQUIRE(overlap > 0); // Exercise the moved column inside the scene button.
    const float label_x = static_cast<float>(state.labels.x + overlap * .5);
    const float label_y = static_cast<float>(std::clamp(static_cast<double>(point.y), state.labels.y + 1,
                                                       state.labels.y + state.labels.height - 1));
    io.AddMousePosEvent(label_x, label_y);
    ui.frame(model, selected);
    CHECK(ui.captured.find(selected) == std::string::npos);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); ui.frame(model, selected);
    io.AddMousePosEvent(point.x, point.y); ui.frame(model, selected);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); ui.frame(model, selected);
    const auto unchanged = [&](const Camera& current) {
        CHECK(current.azimuth == state.camera.azimuth);
        CHECK(current.elevation == state.camera.elevation);
        CHECK(current.distance == state.camera.distance);
        CHECK(current.target.x == state.camera.target.x);
        CHECK(current.target.y == state.camera.target.y);
        CHECK(current.target.z == state.camera.target.z);
    };
    unchanged(model.view_state().camera);

    io.AddMousePosEvent(label_x, label_y); ui.frame(model, selected);
    io.AddMouseButtonEvent(ImGuiMouseButton_Middle, true); ui.frame(model, selected);
    io.AddMousePosEvent(point.x, point.y); ui.frame(model, selected);
    io.AddMouseButtonEvent(ImGuiMouseButton_Middle, false); ui.frame(model, selected);
    unchanged(model.view_state().camera);

    io.AddMousePosEvent(label_x, label_y); ui.frame(model, selected);
    io.AddMouseWheelEvent(0, 1); ui.frame(model, selected);
    unchanged(model.view_state().camera);
}

TEST_CASE("model3d strong frames use contained coplanar bands with stable projected thickness") {
    using namespace caliper_live::model3d;
    Panel panel;
    panel.center = {0.3, -0.2, 1}; panel.width = 4; panel.height = 3;
    Camera camera;
    camera.target = panel.center; camera.distance = 12; camera.radius = 8;
    const Rect viewport{17, 29, 960, 640};
    const double aspect = viewport.width / viewport.height;
    const auto containment = [&](const std::array<FrameBand, 4>& bands) {
        for (const auto& band : bands) {
            CHECK(std::isfinite(band.width)); CHECK(std::isfinite(band.height));
            CHECK(band.width > 0); CHECK(band.height > 0);
            CHECK(band.center.z == panel.center.z);
            CHECK(band.center.x - band.width * .5 >= panel.center.x - panel.width * .5 - 1e-10);
            CHECK(band.center.x + band.width * .5 <= panel.center.x + panel.width * .5 + 1e-10);
            CHECK(band.center.y - band.height * .5 >= panel.center.y - panel.height * .5 - 1e-10);
            CHECK(band.center.y + band.height * .5 <= panel.center.y + panel.height * .5 + 1e-10);
        }
        CHECK(bands[0].width <= panel.width * .2); CHECK(bands[1].width <= panel.width * .2);
        CHECK(bands[2].height <= panel.height * .2); CHECK(bands[3].height <= panel.height * .2);
    };
    for (const auto angles : {std::array<double, 2>{1.5707963267948966, 0}, {1.1, .35}}) {
        camera.azimuth = angles[0]; camera.elevation = angles[1];
        for (double pixels : {2.0, 2.5}) {
            const auto bands = frame_bands(panel, camera, viewport, pixels);
            containment(bands);
            for (size_t i = 0; i < bands.size(); ++i) {
                const bool vertical = i < 2;
                Vec3 a = bands[i].center, b = a, tangent = a;
                if (vertical) { a.x -= bands[i].width * .5; b.x += bands[i].width * .5; tangent.y += .01; }
                else { a.y -= bands[i].height * .5; b.y += bands[i].height * .5; tangent.x += .01; }
                const auto pa = project(a, camera, aspect, viewport), pb = project(b, camera, aspect, viewport);
                const auto pc = project(bands[i].center, camera, aspect, viewport), pt = project(tangent, camera, aspect, viewport);
                const double tx = pt.x - pc.x, ty = pt.y - pc.y;
                const double projected = std::fabs((pb.x - pa.x) * ty - (pb.y - pa.y) * tx) / std::hypot(tx, ty);
                CHECK(projected == doctest::Approx(pixels).epsilon(.05));
            }
        }
    }
    camera.azimuth = 0; camera.elevation = 0; // Exactly edge-on.
    containment(frame_bands(panel, camera, viewport, 2.5));
    camera.azimuth = 1e-12;
    containment(frame_bands(panel, camera, viewport, 2.5));
    const auto invalid = frame_bands(panel, camera, viewport, 0);
    for (const auto& band : invalid) { CHECK(band.width == 0); CHECK(band.height == 0); }
}
