// Synthetic NSEvents exercise the desktop route from GLFW's Cocoa view handlers.
// The window stays hidden. Camera cases use a recording backend. A Metal case checks poll order.
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
#include "renderer/host_renderer.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    bool controls = false, menu_selected = false;
    char text[32]{};
    ImVec2 text_position{}, menu_position{}, item_position{};
    std::unique_ptr<caliper_host::HostRenderer> renderer;

    explicit Harness(bool desktop = false) {
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
        if (desktop) {
            controls = true;
            renderer = caliper_host::make_metal_renderer();
            require(renderer->init(window), "Metal renderer initialization failed");
        } else {
            require(ImGui_ImplGlfw_InitForOther(window, true), "ImGui GLFW initialization failed");
        }
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
        if (renderer) renderer->shutdown();
        else ImGui_ImplGlfw_Shutdown();
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

    void poll() {
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
    }

    void frame() {
        if (renderer) {
            bool polled = false;
            renderer->new_frame_with_input([&] {
                require(renderer->frame_waits().drawable_ms >= 0, "Input preceded drawable acquisition");
                poll();
                polled = true;
            });
            require(polled, "The renderer did not poll input");
        } else {
            poll();
            ImGui_ImplGlfw_NewFrame();
            ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
            ImGui::NewFrame();
        }
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(400, 300));
        ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                          ImGuiWindowFlags_NoSavedSettings);
        if (controls) {
            ImGui::InputText("Text", text, sizeof(text));
            text_position = ImGui::GetItemRectMin();
            text_position.x += 10;
            text_position.y += 10;
            if (ImGui::Button("Menu")) ImGui::OpenPopup("Actions");
            menu_position = ImGui::GetItemRectMin();
            menu_position.x += 10;
            menu_position.y += 5;
            if (ImGui::BeginPopup("Actions")) {
                if (ImGui::MenuItem("Select")) menu_selected = true;
                item_position = ImGui::GetItemRectMin();
                item_position.x += 10;
                item_position.y += 5;
                ImGui::EndPopup();
            }
        } else {
            const auto result = caliper_live_views::draw_scene_view_in_rect(
                scene, 1, "picture", ImVec2(20, 20), ImVec2(320, 220), {}, state, {});
            require(result == caliper_live_views::SceneRectResult::drawn, "Scene widget did not draw");
        }
        ImGui::End();
        if (renderer) renderer->render(400, 300);
        else ImGui::Render();
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
        require(!caliper_live_views::same_camera(before, scene.camera()), "First point did not move the camera");
        const int frames = frame_number - posted;
        require(frames == 1, "Motion did not reach the camera in its arrival frame");
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

    void burst_release(int button) {
        for (int i = 0; i < 20; ++i) frame();
        post(NSEventTypeMouseMoved, 100, 100);
        frame();
        const auto before = scene.camera();
        const auto move = button == 0 ? NSEventTypeLeftMouseDragged : NSEventTypeRightMouseDragged;
        post(button == 0 ? NSEventTypeLeftMouseDown : NSEventTypeRightMouseDown, 100, 100);
        post(move, 130, 100);
        post(move, 101, 100);
        post(button == 0 ? NSEventTypeLeftMouseUp : NSEventTypeRightMouseUp, 101, 100);
        post(NSEventTypeMouseMoved, 300, 100);
        frame();
        require(state.down && state.press.max_d2 == 900, "The burst lost its press or excursion");
        require(ImGui::GetIO().MousePos.x == 101, "Motion crossed a queued release");
        const auto after = scene.camera();
        require(!caliper_live_views::same_camera(before, after), "The burst lost its motion");
        frame();
        require(ImGui::IsMouseReleased(button) && !state.down, "The burst lost its release");
        require(caliper_live_views::same_camera(after, scene.camera()), "Motion repeated or became a click");
        frame();
        require(ImGui::GetIO().MousePos.x == 300, "Motion after release was lost");
    }

    void fast_click_at(ImVec2 pos) {
        press_position = pos;
        post(NSEventTypeMouseMoved, pos.x, pos.y);
        post(NSEventTypeLeftMouseDown, pos.x, pos.y);
        post(NSEventTypeLeftMouseUp, pos.x, pos.y);
        frame();
        require(ImGui::IsMouseClicked(0), "Widget lost its press");
        frame();
        require(ImGui::IsMouseReleased(0), "Widget lost its release");
    }

    void widgets() {
        controls = true;
        for (int i = 0; i < 20; ++i) frame();
        fast_click_at(text_position);
        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiKey_A, true);
        io.AddInputCharactersUTF8("a");
        io.AddKeyEvent(ImGuiKey_A, false);
        io.AddKeyEvent(ImGuiKey_LeftArrow, true);
        io.AddKeyEvent(ImGuiKey_LeftArrow, false);
        io.AddInputCharactersUTF8("b");
        for (int i = 0; i < 6; ++i) frame();
        require(std::strcmp(text, "ba") == 0, "Text and key order changed");
        fast_click_at(menu_position);
        frame();
        require(ImGui::IsPopupOpen("Actions", ImGuiPopupFlags_AnyPopupId), "Fast click did not open the menu");
        fast_click_at(item_position);
        require(menu_selected, "Fast click did not select the menu item");
        require(io.ConfigInputTrickleEventQueue, "Global input trickle was disabled");
    }
};

} // namespace

int main() {
    @autoreleasepool {
        try {
            {
                Harness h;
                for (const bool burst : {false, true}) {
                    const int left = h.drag(0, burst);
                    const int right = h.drag(1, burst);
                    require(left == right, "Left and right latency differs");
                }
                h.click(0);
                h.click(1);
                h.burst_release(0);
                h.burst_release(1);
                h.widgets();
            }
            {
                Harness desktop(true);
                desktop.fast_click_at(desktop.text_position);
                require(ImGui::IsAnyItemActive(), "Desktop input did not reach the text field");
            }
            std::puts("PASS: same-frame cameras, fast clicks, burst releases, text input, and menus");
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL: %s\n", e.what());
            return 1;
        }
    }
}
