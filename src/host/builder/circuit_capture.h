#pragma once
// Bounded hidden-test evidence from the exact frame that capture.frame requests.
#include "spec_runtime/spec_runtime.h"
#include "live/live_scene.h"
#include "live/live_views_scene.h"
#include "draw/draw_runtime.h"
#include "circuit_reference.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

namespace caliper_host {
class CircuitCapture {
    using Json = nlohmann::json;
    struct Panel {
        std::string name;
        caliper_live_views::SceneView* view = nullptr;
        uint64_t serial = 0;
        std::shared_ptr<caliper_live_views::SceneView::CaptureReadback> ids, color;
        std::vector<uint8_t> id_bytes, color_bytes;
        uint64_t pixels = 0;
    };
    static constexpr uint64_t max_pixels = 8 * 1024 * 1024;
    std::filesystem::path root_;
    unsigned sequence_ = 0;
    bool active_ = false, window_ready_ = false;
    Json metadata_;
    std::vector<Panel> panels_;
    std::chrono::steady_clock::time_point started_;
    static constexpr uint64_t max_atlas_pixels = 16 * 1024 * 1024;

    void bytes(const std::string& name, const std::vector<uint8_t>& value) {
        std::ofstream file(root_ / name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(value.data()), std::streamsize(value.size()));
        if (!file.good()) throw std::runtime_error("The circuit capture artifact could not be saved.");
    }
    std::string prefix() const { return "capture-" + std::to_string(sequence_); }
    void error(const std::string& value) { metadata_["error"] = value; }
public:
    CircuitCapture() = default;
    CircuitCapture(const CircuitCapture&) = delete;
    CircuitCapture& operator=(const CircuitCapture&) = delete;
    ~CircuitCapture() = default; // Panel leases release pending reads, even after view cancellation.
    void cancel() { panels_.clear(); active_ = false; window_ready_ = false; }
    // The caller permits this option only for an owned hidden window.
    void begin(SpecRuntime& runtime, const char* folder) {
        if (!folder || !*folder || active_) return;
        root_ = folder;
        if (++sequence_ > 16) return;
        active_ = true;
        window_ready_ = false;
        started_ = std::chrono::steady_clock::now();
        metadata_ = {{"capture", sequence_}, {"panels", Json::object()}};
        const ImGuiIO& io = ImGui::GetIO();
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        metadata_["origin"] = {viewport->Pos.x, viewport->Pos.y};
        metadata_["pixel_scale"] = {io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y};
        uint64_t total = 0;
        try {
            std::filesystem::create_directories(root_);
            // Validate the entire readback size before the first allocation.
            ImFontAtlas* atlas = nullptr;
            size_t glyph_quads = 0;
            for (const char* name : {"graph", "examples", "output"}) {
                auto* view = runtime.scene_view_of(name);
                float rect[4];
                if (!view || !view->id_target() || !view->color() || !runtime.scene_rect(name, rect))
                    throw std::runtime_error(std::string("No rendered target for ") + name + ".");
                const uint64_t pixels = uint64_t(view->width()) * view->height();
                total += pixels;
                if (!pixels || total > max_pixels) throw std::runtime_error("The circuit readback exceeds eight million pixels.");
                Panel panel;
                panel.name = name;
                panel.view = view;
                panel.serial = view->serial();
                panel.pixels = pixels;
                panels_.push_back(std::move(panel));
                Json labels = Json::array(), programs = Json::array(), clip = Json::array();
                for (const auto& label : runtime.scene_labels(name)) {
                    if (labels.size() >= 512) throw std::runtime_error("The capture exceeds 512 labels per panel.");
                    if (!label.font || (atlas && label.font->OwnerAtlas != atlas))
                        throw std::runtime_error("The capture requires one label font atlas.");
                    atlas = label.font->OwnerAtlas;
                    labels.push_back(CircuitReference::text(label));
                    glyph_quads += labels.back()["quads"].size();
                    if (glyph_quads > 65536) throw std::runtime_error("The capture exceeds 65536 reference glyph quads.");
                }
                for (const auto* program : view->draw_runtimes())
                    programs.push_back({{"id", program->instance().program}, {"pick_id", program->pick_id()},
                                        {"step", program->status().resource_step}});
                for (unsigned i = 0; i < 16; ++i) clip.push_back(view->program_clip().m[i]);
                metadata_["panels"][name] = {{"width", view->width()}, {"height", view->height()},
                    {"view_serial", panel.serial}, {"generation", view->generation()}, {"step", view->program_step()},
                    {"rect", {rect[0], rect[1], rect[2], rect[3]}}, {"clip_from_world", clip},
                    {"composition", CircuitReference::image(ImTextureID(view->imgui_texture()))},
                    {"labels", labels}, {"programs", programs},
                    {"ids", prefix() + "-" + name + ".id"}, {"color", prefix() + "-" + name + ".rgba"}};
            }
            const ImTextureData* font = atlas ? atlas->TexData : nullptr;
            if (!font || !font->Pixels || uint64_t(font->Width) * font->Height > max_atlas_pixels ||
                font->Format != ImTextureFormat_RGBA32)
                throw std::runtime_error("No bounded RGBA font atlas is available.");
            metadata_["font_atlas"] = {{"width", font->Width}, {"height", font->Height}, {"file", prefix() + "-font.rgba"}};
            bytes(prefix() + "-font.rgba", std::vector<uint8_t>(font->Pixels, font->Pixels + uint64_t(font->Width) * font->Height * 4));
            for (auto& panel : panels_) {
                // Metal's read_texture implementation owns an autorelease scope.
                panel.ids = panel.view->capture_readback(panel.view->id_target());
                panel.color = panel.view->capture_readback(panel.view->color());
                if (!panel.ids || !panel.color) throw std::runtime_error("The circuit target readback was refused.");
            }
        } catch (const std::exception& e) {
            error(e.what());
            for (auto& panel : panels_) { panel.ids.reset(); panel.color.reset(); }
        }
    }
    void window(uint32_t width, uint32_t height, const std::vector<uint8_t>& rgba) {
        if (!active_ || window_ready_) return;
        window_ready_ = true;
        try {
            if (uint64_t(width) * height > max_pixels || rgba.size() != uint64_t(width) * height * 4)
                throw std::runtime_error("The window capture exceeds the diagnostic limit.");
            metadata_["window"] = prefix() + "-window.rgba";
            metadata_["width"] = width;
            metadata_["height"] = height;
            bytes(metadata_["window"].get<std::string>(), rgba);
        } catch (const std::exception& e) { error(e.what()); }
    }
    void poll(SpecRuntime& runtime) {
        if (!active_) return;
        if (std::chrono::steady_clock::now() - started_ > std::chrono::seconds(10))
            error("The circuit readback did not complete before its deadline.");
        bool ready = window_ready_;
        for (auto& panel : panels_) {
            auto* current = runtime.scene_view_of(panel.name);
            if (!current || current != panel.view || current->serial() != panel.serial) {
                error("The captured panel changed before readback completion.");
                continue;
            }
            if (panel.ids && panel.ids->take(&panel.id_bytes)) panel.ids.reset();
            if (panel.color && panel.color->take(&panel.color_bytes)) panel.color.reset();
            ready = ready && !panel.ids && !panel.color;
        }
        if (!ready && !metadata_.contains("error")) return;
        try {
            for (auto& panel : panels_) {
                panel.ids.reset();
                panel.color.reset();
                if (panel.id_bytes.size() != panel.pixels * 8 || panel.color_bytes.size() != panel.pixels * 4) {
                    error("The circuit readback has incomplete target bytes.");
                    continue;
                }
                bytes(prefix() + "-" + panel.name + ".id", panel.id_bytes);
                bytes(prefix() + "-" + panel.name + ".rgba", panel.color_bytes);
            }
            const auto pending = root_ / (prefix() + ".pending");
            std::ofstream file(pending);
            file << metadata_.dump(2) << '\n';
            file.close();
            if (!file.good()) throw std::runtime_error("The circuit capture metadata could not be saved.");
            std::filesystem::rename(pending, root_ / (prefix() + ".json"));
        } catch (const std::exception&) {
            // The driver treats an absent completion artifact as a failed capture.
        }
        panels_.clear();
        active_ = false;
    }
};
} // namespace caliper_host
