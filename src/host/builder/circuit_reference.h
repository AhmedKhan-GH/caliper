#pragma once
// Same-frame image placement and independent font references for hidden captures.
#include "imgui_internal.h"
#include "live/live_views_scene.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace caliper_host {
class CircuitReference {
    using Json = nlohmann::json;
public:
    static Json color(ImU32 c) {
        return {(c >> IM_COL32_R_SHIFT) & 255, (c >> IM_COL32_G_SHIFT) & 255,
                (c >> IM_COL32_B_SHIFT) & 255, (c >> IM_COL32_A_SHIFT) & 255};
    }
    static Json quad(const ImDrawList& list, const ImDrawCmd& cmd, unsigned first) {
        if (first + 6 > unsigned(list.IdxBuffer.Size)) throw std::runtime_error("Incomplete capture quad.");
        const ImDrawVert* vertices[6];
        for (unsigned i = 0; i < 6; ++i) {
            const unsigned index = cmd.VtxOffset + list.IdxBuffer[first + i];
            if (index >= unsigned(list.VtxBuffer.Size)) throw std::runtime_error("Invalid capture vertex.");
            vertices[i] = &list.VtxBuffer[index];
        }
        const auto &a = *vertices[0], &b = *vertices[1], &c = *vertices[2], &d = *vertices[5];
        if (vertices[0] != vertices[3] || vertices[2] != vertices[4] ||
            a.pos.x != d.pos.x || a.pos.y != b.pos.y || c.pos.x != b.pos.x || c.pos.y != d.pos.y ||
            a.uv.x != d.uv.x || a.uv.y != b.uv.y || c.uv.x != b.uv.x || c.uv.y != d.uv.y ||
            a.col != b.col || a.col != c.col || a.col != d.col)
            throw std::runtime_error("The capture reference requires an axis-aligned image quad.");
        return {{"rect", {a.pos.x, a.pos.y, c.pos.x, c.pos.y}},
                {"uv", {a.uv.x, a.uv.y, c.uv.x, c.uv.y}}, {"tint", color(a.col)},
                {"clip", {cmd.ClipRect.x, cmd.ClipRect.y, cmd.ClipRect.z, cmd.ClipRect.w}}};
    }
    static Json image(ImTextureID texture) {
        Json result;
        for (const ImGuiWindow* window : ImGui::GetCurrentContext()->Windows) {
            if (!window->Active || window->Hidden) continue;
            const ImDrawList& list = *window->DrawList;
            for (const ImDrawCmd& cmd : list.CmdBuffer) {
                if (cmd.TexRef._TexData || cmd.TexRef._TexID != texture || !cmd.ElemCount) continue;
                if (cmd.UserCallback || cmd.ElemCount != 6 || !result.is_null())
                    throw std::runtime_error("The panel image has unsupported composition.");
                result = quad(list, cmd, cmd.IdxOffset);
            }
        }
        if (result.is_null()) throw std::runtime_error("The panel image is absent from the window draw lists.");
        result["sampler"] = "linear";
        result["blend"] = "source_alpha";
        return result;
    }
    static Json text(const caliper_live_views::DrawnLabel& label) {
        if (!label.font || label.text.size() > 256 || !(label.font_size > 0))
            throw std::runtime_error("The label has no bounded font reference.");
        // This private list never enters the window. It renders the expected string
        // from the saved font state, independently of the actual label draw calls.
        ImDrawList reference(ImGui::GetDrawListSharedData());
        reference._ResetForNewFrame();
        reference.Flags = label.draw_flags;
        reference.PushTexture(label.font->OwnerAtlas->TexRef);
        reference.PushClipRect(ImVec2(label.clip.x, label.clip.y), ImVec2(label.clip.z, label.clip.w));
        struct Density {
            ImFont& font;
            float previous;
            ~Density() { font.CurrentRasterizerDensity = previous; }
        } restore{*label.font, label.font->CurrentRasterizerDensity};
        label.font->CurrentRasterizerDensity = label.font_density;
        // Keep every expected glyph. The saved command clip still applies in the
        // CPU oracle, which rejects a required label if that clip removes text.
        const ImVec4 complete_clip(-FLT_MAX, -FLT_MAX, FLT_MAX, FLT_MAX);
        label.font->RenderText(&reference, label.font_size, ImVec2(label.x0 + 1, label.y0 + 1),
                               label.shade, complete_clip, label.text.c_str(), nullptr);
        label.font->RenderText(&reference, label.font_size, ImVec2(label.x0, label.y0),
                               label.ink, complete_clip, label.text.c_str(), nullptr);
        if (reference.IdxBuffer.Size > 512 * 6) throw std::runtime_error("The label exceeds the glyph limit.");
        Json quads = Json::array();
        for (const auto& cmd : reference.CmdBuffer) {
            if (cmd.ElemCount % 6 || cmd.UserCallback) throw std::runtime_error("Unsupported glyph geometry.");
            for (unsigned i = 0; i < cmd.ElemCount; i += 6) quads.push_back(quad(reference, cmd, cmd.IdxOffset + i));
        }
        return {{"text", label.text}, {"box", {label.x0, label.y0, label.x1, label.y1}},
                {"font_size", label.font_size}, {"font_density", label.font_density}, {"quads", quads}};
    }
};
} // namespace caliper_host
