// CPU-only cache and lazy-loading contract. No Metal device or model is opened.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/live_model3d.h"

#include <imgui_internal.h>

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

    void frame(Model3D& model, const std::string& selected = {}, bool collapsed = false) {
        ImGui::NewFrame();
        ImGui::SetNextWindowCollapsed(collapsed, ImGuiCond_Always);
        CHECK(model.draw(nullptr, 0, 0, 900, 600, selected).empty());
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
    }
};
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
