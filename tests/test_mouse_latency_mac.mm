// Synthetic NSEvents exercise the desktop route from GLFW's Cocoa view handlers.
// The window stays hidden. A recording graphics backend needs no GPU or live producer.
#import <AppKit/AppKit.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <backends/imgui_impl_glfw.h>

#include "glfw_background.h"
#include "gfx/recording_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "live/live_scene.h"
#include "live/live_views_scene.h"
#include "mouse_latency.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Harness {
    GLFWwindow* window = nullptr;
    caliper_gfx::RecordingBackend gfx;
    caliper_live_views::SceneView scene{gfx, caliper_gfx::shaders::kMetalPrograms};
    caliper_live_views::SceneRectState state;
    int frame_number = 0;
    ImVec2 press_position{100, 100};

    Harness() {
        require(caliper_test::glfw_init_in_background(), "GLFW initialization failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
        glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
        window = glfwCreateWindow(400, 300, "Input Test", nullptr, nullptr);
        require(window != nullptr, "Hidden window creation failed");
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = io.LogFilename = nullptr;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        // The app supplies the cursor position at a press. Use the event's fixed test
        // position because the hidden window does not own the physical cursor.
        glfwSetWindowUserPointer(window, this);
        glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int, int action, int) {
            auto& h = *static_cast<Harness*>(glfwGetWindowUserPointer(w));
            if (action == GLFW_PRESS)
                ImGui::GetIO().AddMousePosEvent(h.press_position.x, h.press_position.y);
        });
        require(ImGui_ImplGlfw_InitForOther(window, true), "ImGui GLFW initialization failed");
        NSView* native = ((NSWindow*)glfwGetCocoaWindow(window)).contentView;
        std::printf("pressure=%ld recognizers=%lu accepts_first_mouse=%d mouse_moves_window=%d\n",
                    native.pressureConfiguration ? long(native.pressureConfiguration.pressureBehavior) : -1L,
                    (unsigned long)native.gestureRecognizers.count, int([native acceptsFirstMouse:nil]),
                    int(native.mouseDownCanMoveWindow));
        caliper_host::mouse_latency_start(window);
        std::string why;
        require(scene.init(&why), why.c_str());
        scene.describe({"3d", "+y", "", {}});
        frame();
        frame();
    }

    ~Harness() {
        caliper_host::mouse_latency_stop();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void post(NSEventType type, float x, float y) {
        NSWindow* native = glfwGetCocoaWindow(window);
        NSEvent* event = [NSEvent mouseEventWithType:type
            location:NSMakePoint(x, native.contentView.bounds.size.height - y)
            modifierFlags:0 timestamp:NSProcessInfo.processInfo.systemUptime
            windowNumber:native.windowNumber context:nil eventNumber:0
            clickCount:1 pressure:0];
        [NSApp postEvent:event atStart:NO];
    }

    void frame() {
        // AppKit discards a synthetic press addressed to a hidden, inactive window.
        // Dequeue native events as GLFW does, then call the GLFW view's responders.
        // This preserves NSEvent.timestamp and NSApp.currentEvent without activation.
        // WindowServer delivery and hardware gesture recognition need a manual test.
        NSWindow* native = glfwGetCocoaWindow(window);
        for (;;) {
            NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny untilDate:NSDate.distantPast
                                                inMode:NSDefaultRunLoopMode dequeue:YES];
            if (!event) break;
            if (event.window != native) { [NSApp sendEvent:event]; continue; }
            switch (event.type) {
            case NSEventTypeMouseMoved: [native.contentView mouseMoved:event]; break;
            case NSEventTypeLeftMouseDown: [native.contentView mouseDown:event]; break;
            case NSEventTypeLeftMouseDragged: [native.contentView mouseDragged:event]; break;
            case NSEventTypeLeftMouseUp: [native.contentView mouseUp:event]; break;
            case NSEventTypeRightMouseDown: [native.contentView rightMouseDown:event]; break;
            case NSEventTypeRightMouseDragged: [native.contentView rightMouseDragged:event]; break;
            case NSEventTypeRightMouseUp: [native.contentView rightMouseUp:event]; break;
            default: [NSApp sendEvent:event]; break;
            }
        }
        require(!glfwGetWindowAttrib(window, GLFW_VISIBLE), "Test window became visible");
        require(!glfwGetWindowAttrib(window, GLFW_FOCUSED), "Test window took focus");
        require(caliper_host::background_activation_policy_active(), "Background policy changed");
        ImGui_ImplGlfw_NewFrame();
        ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(400, 300));
        ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                          ImGuiWindowFlags_NoSavedSettings);
        const auto result = caliper_live_views::draw_scene_view_in_rect(
            scene, 1, "picture", ImVec2(20, 20), ImVec2(320, 220), {}, state, {});
        require(result == caliper_live_views::SceneRectResult::drawn, "Scene widget did not draw");
        ImGui::End();
        ImGui::Render();
        ++frame_number;
    }

    int drag(int button, bool burst) {
        // Separate gestures in ImGui time to exclude the scene's double-click fit.
        for (int i = 0; i < 20; ++i) frame();
        const auto down = button == 0 ? NSEventTypeLeftMouseDown : NSEventTypeRightMouseDown;
        const auto move = button == 0 ? NSEventTypeLeftMouseDragged : NSEventTypeRightMouseDragged;
        const auto up = button == 0 ? NSEventTypeLeftMouseUp : NSEventTypeRightMouseUp;
        post(NSEventTypeMouseMoved, 100, 100);
        frame();
        const auto before = scene.camera();
        post(down, 100, 100);
        if (!burst) {
            frame();
            require(state.down && state.button == button, "Press did not reach the scene");
            require(caliper_live_views::same_camera(before, scene.camera()), "Press moved the camera");
        }
        const int posted = frame_number;
        post(move, 101, 100);
        frame();
        if (caliper_live_views::same_camera(before, scene.camera())) frame();
        require(!caliper_live_views::same_camera(before, scene.camera()), "First point did not move the camera");
        const int frames = frame_number - posted;
        require(frames <= (burst ? 2 : 1), "Motion waited more than one extra frame");
        const auto& a = std::get<caliper_scene::Orbit>(before);
        const auto& b = std::get<caliper_scene::Orbit>(scene.camera());
        require(button == 0 ? b.azimuth != a.azimuth : b.target[0] != a.target[0] || b.target[2] != a.target[2],
                "Wrong camera action");
        post(move, 108, 100);
        frame();
        post(up, 108, 100);
        frame();
        require(!state.down, "Release did not reach the scene");
        std::printf("button=%d burst=%d motion_frames=%d\n", button, int(burst), frames);
        return frames;
    }

    void click(int button) {
        for (int i = 0; i < 20; ++i) frame();
        post(NSEventTypeMouseMoved, 100, 100);
        frame();
        post(button == 0 ? NSEventTypeLeftMouseDown : NSEventTypeRightMouseDown, 100, 100);
        post(button == 0 ? NSEventTypeLeftMouseUp : NSEventTypeRightMouseUp, 100, 100);
        frame();
        require(ImGui::IsMouseClicked(button), "Trickle queue lost a fast press");
        frame();
        require(ImGui::IsMouseReleased(button), "Trickle queue lost a fast release");
    }
};

} // namespace

int main() {
    @autoreleasepool {
        try {
            Harness h;
            for (const bool burst : {false, true}) {
                const int left = h.drag(0, burst);
                const int right = h.drag(1, burst);
                require(std::abs(left - right) <= 1, "Left and right latency differs by more than one frame");
            }
            h.click(0);
            h.click(1);
            std::puts("PASS: native events, scene cameras, button parity, and fast clicks");
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL: %s\n", e.what());
            return 1;
        }
    }
}
