#pragma once

#include "live/camera_drag.h"
#include "mouse_latency.h"
#include <imgui.h>
#include <imgui_internal.h>

namespace caliper_live_views {

// ImGui stops at motion after a button edge when ConfigInputTrickleEventQueue is true.
// Let the camera that owns the press consume the adjacent motion in this frame.
// Keep every button edge, key, wheel and focus event in order for ImGui's next frame.
// Thus a fast release still produces a click, and other widgets retain normal input.
template <class V>
void camera_press_input(CameraPress<typename V::State>& press, V& view, int button, bool released) {
    ImGuiContext& context = *ImGui::GetCurrentContext();
    ImGuiIO& io = context.IO;
    camera_press_frame(press, view, ImGui::IsMousePosValid(), io.MousePos.x, io.MousePos.y, released);
    if (!press.down || !ImGui::IsItemActive() || !io.MouseDown[button] || io.WantSetMousePos)
        return;

    int count = 0;
    for (const ImGuiInputEvent& event : context.InputEventsQueue) {
        if (event.Type != ImGuiInputEventType_MousePos || event.MousePos.MouseSource != ImGuiMouseSource_Mouse)
            break;
        const ImVec2 pos(event.MousePos.PosX, event.MousePos.PosY);
        if (!ImGui::IsMousePosValid(&pos)) break;
        io.MousePos = context.MouseLastValidPos = ImFloor(pos);
        io.MouseSource = event.MousePos.MouseSource;
        io.MouseDelta = ImGui::IsMousePosValid(&io.MousePosPrev)
            ? ImVec2(io.MousePos.x - io.MousePosPrev.x, io.MousePos.y - io.MousePosPrev.y) : ImVec2();
        for (int b = 0; b < ImGuiMouseButton_COUNT; ++b) {
            if (!io.MouseDown[b]) continue;
            const float dx = io.MousePos.x - io.MouseClickedPos[b].x;
            const float dy = io.MousePos.y - io.MouseClickedPos[b].y;
            io.MouseDragMaxDistanceSqr[b] = ImMax(io.MouseDragMaxDistanceSqr[b], dx * dx + dy * dy);
            io.MouseDragMaxDistanceAbs[b].x = ImMax(io.MouseDragMaxDistanceAbs[b].x, ImAbs(dx));
            io.MouseDragMaxDistanceAbs[b].y = ImMax(io.MouseDragMaxDistanceAbs[b].y, ImAbs(dy));
        }
        // Apply every point: an excursion and return in one batch is still a drag.
        camera_press_frame(press, view, true, io.MousePos.x, io.MousePos.y, false);
        context.InputEventsTrail.push_back(event);
        ++count;
    }
    if (count) {
        context.InputEventsQueue.erase(context.InputEventsQueue.Data, context.InputEventsQueue.Data + count);
        context.MouseStationaryTimer = 0.0f;
        context.NavHighlightItemUnderNav = false;
        caliper_host::mouse_latency_consume();
    }
}

} // namespace caliper_live_views
