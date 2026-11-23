#include "mouse_latency.h"

#import <AppKit/AppKit.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_map>

namespace caliper_host {
namespace {

double uptime() { return NSProcessInfo.processInfo.systemUptime; }

struct Sample {
    unsigned sequence = 0, event_id = 0;
    int type = 0, imgui_type = 0, button = -1;
    double event = -1, callback = -1, consume_lo = -1, consume = -1;
    double x = 0, y = 0;
};

struct Trace {
    FILE* file = nullptr;
    GLFWwindow* window = nullptr;
    ImGuiContext* context = nullptr;
    GLFWmousebuttonfun button_callback = nullptr;
    GLFWcursorposfun position_callback = nullptr;
    GLFWscrollfun scroll_callback = nullptr;
    ImGuiID pre_hook = 0, post_hook = 0;
    id pressure_monitor = nil;
    unsigned sequence = 0, lines = 0;
    double consume_lo = -1;
    std::unordered_map<unsigned, Sample> pending;
    std::array<Sample, 3> press{}, first_motion{}, motion{};
    std::array<bool, 3> down{}, changed{};

    // Times share NSEvent.timestamp's uptime clock. NewFrame starts the consumption interval.
    // A camera widget can consume adjacent motion after the NewFrame hook.
    void line(const char* stage, const Sample& s, double camera = -1) {
        if (!file || lines >= 60000) return;
        std::fprintf(file, "%s\t%u\t%u\t%d\t%d\t%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\t%d\t%.3f\t%.3f\n",
                     stage, s.sequence, s.event_id, s.type, s.imgui_type, s.button, s.event, s.callback,
                     s.consume_lo, s.consume, camera, context->FrameCount, s.x, s.y);
        if (++lines == 60000) std::fputs("# trace limit reached\n", file);
    }

    Sample begin(int button) {
        Sample s;
        s.callback = uptime();
        s.sequence = ++sequence;
        s.button = button;
        NSEvent* event = NSApp.currentEvent;
        NSWindow* native = glfwGetCocoaWindow(window);
        if (event && event.windowNumber == native.windowNumber) {
            switch (event.type) {
            case NSEventTypeLeftMouseDown: case NSEventTypeLeftMouseUp: case NSEventTypeLeftMouseDragged:
                s.button = 0; break;
            case NSEventTypeRightMouseDown: case NSEventTypeRightMouseUp: case NSEventTypeRightMouseDragged:
                s.button = 1; break;
            case NSEventTypeOtherMouseDown: case NSEventTypeOtherMouseUp: case NSEventTypeOtherMouseDragged:
                s.button = int(event.buttonNumber); break;
            case NSEventTypeMouseMoved: case NSEventTypeScrollWheel: break;
            default: return s;
            }
            s.type = int(event.type);
            s.event = event.timestamp;
            s.x = event.locationInWindow.x;
            s.y = native.contentView.bounds.size.height - event.locationInWindow.y;
        }
        return s;
    }

    void queued(Sample s, unsigned first_id) {
        line("callback", s);
        // A button callback can add a press position and modifier events before the button.
        // Match IDs, never coordinates, because a fast drag can revisit the same point.
        for (const ImGuiInputEvent& e : context->InputEventsQueue) {
            if (e.EventId < first_id) continue;
            if (e.Type != ImGuiInputEventType_MousePos && e.Type != ImGuiInputEventType_MouseButton &&
                e.Type != ImGuiInputEventType_MouseWheel) continue;
            s.event_id = e.EventId;
            s.imgui_type = int(e.Type);
            pending.emplace(e.EventId, s);
        }
        if (pending.size() > 4096) {
            std::fputs("# pending event limit reached; unmatched events discarded\n", file);
            pending.clear();
        }
    }

    void consumed() {
        const double now = uptime();
        for (const ImGuiInputEvent& e : context->InputEventsTrail) {
            auto it = pending.find(e.EventId);
            if (it == pending.end()) continue;
            Sample s = it->second;
            pending.erase(it);
            s.consume_lo = consume_lo;
            s.consume = now;
            line("consume", s);
            if (e.Type == ImGuiInputEventType_MouseButton && e.MouseButton.Button < 3) {
                const int b = e.MouseButton.Button;
                down[b] = e.MouseButton.Down;
                if (down[b]) {
                    press[b] = s;
                    first_motion[b] = {};
                    motion[b] = {};
                    changed[b] = false;
                }
            } else if (e.Type == ImGuiInputEventType_MousePos) {
                for (int b = 0; b < 3; ++b) {
                    if (!down[b]) continue;
                    if (!first_motion[b].sequence) first_motion[b] = s;
                    motion[b] = s;
                }
            }
        }
        // Flush after timestamps, once a second, to limit disk work during a drag.
        if (context->FrameCount % 60 == 0) std::fflush(file);
    }
};

std::unique_ptr<Trace> trace;

void button_callback(GLFWwindow* w, int b, int action, int mods) {
    Trace& t = *trace;
    if (t.lines >= 60000) {
        if (t.button_callback) t.button_callback(w, b, action, mods);
        return;
    }
    Sample s = t.begin(b);
    const unsigned first = t.context->InputEventsNextEventId;
    if (t.button_callback) t.button_callback(w, b, action, mods);
    t.queued(s, first);
}

void position_callback(GLFWwindow* w, double x, double y) {
    Trace& t = *trace;
    if (t.lines >= 60000) {
        if (t.position_callback) t.position_callback(w, x, y);
        return;
    }
    Sample s = t.begin(-1);
    const unsigned first = t.context->InputEventsNextEventId;
    if (t.position_callback) t.position_callback(w, x, y);
    t.queued(s, first);
}

void scroll_callback(GLFWwindow* w, double x, double y) {
    Trace& t = *trace;
    if (t.lines >= 60000) {
        if (t.scroll_callback) t.scroll_callback(w, x, y);
        return;
    }
    Sample s = t.begin(-1);
    const unsigned first = t.context->InputEventsNextEventId;
    if (t.scroll_callback) t.scroll_callback(w, x, y);
    t.queued(s, first);
}

} // namespace

void mouse_latency_start(GLFWwindow* window) {
    const char* path = std::getenv("CALIPER_MOUSE_LOG");
    if (trace || !path || !*path || !ImGui::GetCurrentContext()) return;
    FILE* file = std::fopen(path, "a");
    if (!file) return;
    trace = std::make_unique<Trace>();
    Trace& t = *trace;
    t.file = file;
    t.window = window;
    t.context = ImGui::GetCurrentContext();
    NSView* view = ((NSWindow*)glfwGetCocoaWindow(window)).contentView;
    std::fprintf(file, "# uptime_seconds pressure=%ld recognizers=%lu key=%d trickle=%d\n",
                 view.pressureConfiguration ? long(view.pressureConfiguration.pressureBehavior) : -1L,
                 (unsigned long)view.gestureRecognizers.count, int(view.window.isKeyWindow),
                 int(t.context->IO.ConfigInputTrickleEventQueue));
    std::fputs("stage\tsequence\tevent_id\tns_type\timgui_type\tbutton\tevent_s\tcallback_s\tconsume_lo_s\tconsume_s\tcamera_s\tframe\tx\ty\n", file);
    t.button_callback = glfwSetMouseButtonCallback(window, button_callback);
    t.position_callback = glfwSetCursorPosCallback(window, position_callback);
    t.scroll_callback = glfwSetScrollCallback(window, scroll_callback);
    ImGuiContextHook hook;
    hook.Type = ImGuiContextHookType_NewFramePre;
    hook.Callback = [](ImGuiContext*, ImGuiContextHook*) { trace->consume_lo = uptime(); };
    t.pre_hook = ImGui::AddContextHook(t.context, &hook);
    hook.Type = ImGuiContextHookType_NewFramePost;
    hook.Callback = [](ImGuiContext*, ImGuiContextHook*) { trace->consumed(); };
    t.post_hook = ImGui::AddContextHook(t.context, &hook);
    // GLFW does not forward pressure events. Keep their native behavior in the trace
    // to distinguish the device's effective configuration from a nil view property.
    t.pressure_monitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskPressure
        handler:^NSEvent*(NSEvent* event) {
            if (trace && trace->lines < 60000 && event.window == glfwGetCocoaWindow(trace->window)) {
                const double now = uptime();
                std::fprintf(trace->file, "# pressure event_s=%.9f delivery_s=%.9f behavior=%ld stage=%ld\n",
                             event.timestamp, now, long(event.pressureBehavior), long(event.stage));
                ++trace->lines;
            }
            return event;
        }];
}

void mouse_latency_stop() {
    if (!trace) return;
    Trace& t = *trace;
    [NSEvent removeMonitor:t.pressure_monitor];
    glfwSetMouseButtonCallback(t.window, t.button_callback);
    glfwSetCursorPosCallback(t.window, t.position_callback);
    glfwSetScrollCallback(t.window, t.scroll_callback);
    ImGui::RemoveContextHook(t.context, t.pre_hook);
    ImGui::RemoveContextHook(t.context, t.post_hook);
    std::fclose(t.file);
    trace.reset();
}

bool mouse_latency_enabled() { return trace && trace->lines < 60000; }

void mouse_latency_consume() {
    if (mouse_latency_enabled()) trace->consumed();
}

void mouse_latency_camera(int button) {
    if (!mouse_latency_enabled() || button < 0 || button >= 3) return;
    Trace& t = *trace;
    if (t.changed[button] || !t.press[button].sequence || !t.motion[button].sequence) return;
    const double now = uptime();
    t.changed[button] = true;
    t.line("camera_press", t.press[button], now);
    t.line("camera_first_motion", t.first_motion[button], now);
    t.line("camera_motion", t.motion[button], now);
}

} // namespace caliper_host
