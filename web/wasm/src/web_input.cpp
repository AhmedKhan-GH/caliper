// web_input.cpp: see web_input.h.
#include "web_input.h"

#include <imgui.h>

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace {

ImGuiKey key_from_code(const char* c) {
    if (!c) return ImGuiKey_None;
    // KeyA..KeyZ, Digit0..Digit9, F1..F12
    if (std::strncmp(c, "Key", 3) == 0 && c[3] >= 'A' && c[3] <= 'Z' && !c[4])
        return (ImGuiKey)(ImGuiKey_A + (c[3] - 'A'));
    if (std::strncmp(c, "Digit", 5) == 0 && c[5] >= '0' && c[5] <= '9' && !c[6])
        return (ImGuiKey)(ImGuiKey_0 + (c[5] - '0'));
    if (std::strncmp(c, "Numpad", 6) == 0 && c[6] >= '0' && c[6] <= '9' && !c[7])
        return (ImGuiKey)(ImGuiKey_Keypad0 + (c[6] - '0'));
    if (c[0] == 'F' && c[1] >= '1' && c[1] <= '9') {
        const int n = c[2] ? (c[1] - '0') * 10 + (c[2] - '0') : c[1] - '0';
        if (n >= 1 && n <= 12 && (!c[2] || !c[3])) return (ImGuiKey)(ImGuiKey_F1 + n - 1);
    }
    static const struct { const char* code; ImGuiKey key; } kMap[] = {
        {"ArrowLeft", ImGuiKey_LeftArrow}, {"ArrowRight", ImGuiKey_RightArrow},
        {"ArrowUp", ImGuiKey_UpArrow}, {"ArrowDown", ImGuiKey_DownArrow},
        {"PageUp", ImGuiKey_PageUp}, {"PageDown", ImGuiKey_PageDown},
        {"Home", ImGuiKey_Home}, {"End", ImGuiKey_End}, {"Insert", ImGuiKey_Insert},
        {"Delete", ImGuiKey_Delete}, {"Backspace", ImGuiKey_Backspace}, {"Space", ImGuiKey_Space},
        {"Enter", ImGuiKey_Enter}, {"Escape", ImGuiKey_Escape}, {"Tab", ImGuiKey_Tab},
        {"ShiftLeft", ImGuiKey_LeftShift}, {"ShiftRight", ImGuiKey_RightShift},
        {"ControlLeft", ImGuiKey_LeftCtrl}, {"ControlRight", ImGuiKey_RightCtrl},
        {"AltLeft", ImGuiKey_LeftAlt}, {"AltRight", ImGuiKey_RightAlt},
        {"MetaLeft", ImGuiKey_LeftSuper}, {"MetaRight", ImGuiKey_RightSuper},
        {"Quote", ImGuiKey_Apostrophe}, {"Comma", ImGuiKey_Comma}, {"Minus", ImGuiKey_Minus},
        {"Period", ImGuiKey_Period}, {"Slash", ImGuiKey_Slash}, {"Semicolon", ImGuiKey_Semicolon},
        {"Equal", ImGuiKey_Equal}, {"BracketLeft", ImGuiKey_LeftBracket},
        {"Backslash", ImGuiKey_Backslash}, {"BracketRight", ImGuiKey_RightBracket},
        {"Backquote", ImGuiKey_GraveAccent}, {"CapsLock", ImGuiKey_CapsLock},
        {"NumpadEnter", ImGuiKey_KeypadEnter}, {"NumpadAdd", ImGuiKey_KeypadAdd},
        {"NumpadSubtract", ImGuiKey_KeypadSubtract}, {"NumpadMultiply", ImGuiKey_KeypadMultiply},
        {"NumpadDivide", ImGuiKey_KeypadDivide}, {"NumpadDecimal", ImGuiKey_KeypadDecimal},
    };
    for (const auto& m : kMap)
        if (std::strcmp(c, m.code) == 0) return m.key;
    return ImGuiKey_None;
}

void mods(int m) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiMod_Ctrl, (m & 1) != 0);
    io.AddKeyEvent(ImGuiMod_Shift, (m & 2) != 0);
    io.AddKeyEvent(ImGuiMod_Alt, (m & 4) != 0);
    io.AddKeyEvent(ImGuiMod_Super, (m & 8) != 0);
}

double g_last_ms = 0.0;
float g_last_dt = 0.0f;   // the previous frame's io.DeltaTime
int g_cursor = -2;
caliper_web::InputCounts g_counts;

}  // namespace

// ---- entry points for the listeners below (this module only) ----------------------
extern "C" {

EMSCRIPTEN_KEEPALIVE void caliper_input_pointer(float x, float y, int source) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseSourceEvent(source == 1 ? ImGuiMouseSource_TouchScreen
                           : source == 2 ? ImGuiMouseSource_Pen : ImGuiMouseSource_Mouse);
    io.AddMousePosEvent(x, y);
    ++g_counts.pointer;
}

EMSCRIPTEN_KEEPALIVE void caliper_input_leave() {
    ImGui::GetIO().AddMousePosEvent(-FLT_MAX, -FLT_MAX);
}

// DOM button 0 left, 1 middle, 2 right -> ImGui 0 left, 2 middle, 1 right.
EMSCRIPTEN_KEEPALIVE void caliper_input_button(int button, int down, int m) {
    const int b = button == 1 ? 2 : button == 2 ? 1 : button;
    if (b < 0 || b >= ImGuiMouseButton_COUNT) return;
    mods(m);
    ImGui::GetIO().AddMouseButtonEvent(b, down != 0);
    ++g_counts.buttons;
}

// Returns 1 when ImGui takes the wheel (the page must not scroll then).
EMSCRIPTEN_KEEPALIVE int caliper_input_wheel(float dx, float dy) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseWheelEvent(dx, dy);
    ++g_counts.wheel;
    return io.WantCaptureMouse ? 1 : 0;
}

// Returns 1 when ImGui takes the key (the page must not act on it then).
EMSCRIPTEN_KEEPALIVE int caliper_input_key(const char* code, int down, int m) {
    ImGuiIO& io = ImGui::GetIO();
    mods(m);
    const ImGuiKey k = key_from_code(code);
    if (k != ImGuiKey_None) {
        io.AddKeyEvent(k, down != 0);
        ++g_counts.keys;
        if (down) g_counts.last_key = ImGui::GetKeyName(k);
    }
    return io.WantCaptureKeyboard ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE void caliper_input_char(unsigned int codepoint) {
    ImGui::GetIO().AddInputCharacter(codepoint);
    ++g_counts.chars;
}

EMSCRIPTEN_KEEPALIVE void caliper_input_focus(int focused) {
    ImGui::GetIO().AddFocusEvent(focused != 0);
}

}  // extern "C"

// The listeners, on this instance's canvas only. A pointer drag keeps its
// events through setPointerCapture, so it may leave the canvas. Wheel and keys
// are taken from the page only while ImGui wants them.
EM_JS(void, caliper_listen, (), {
    var c = Module["canvas"];
    var list = Module["caliperListeners"] = [];
    function on(type, fn, opts) { c.addEventListener(type, fn, opts); list.push([type, fn, opts]); }
    function src(e) { return e.pointerType === "touch" ? 1 : e.pointerType === "pen" ? 2 : 0; }
    function mods(e) { return (e.ctrlKey ? 1 : 0) | (e.shiftKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0); }
    function move(e) {
        var r = c.getBoundingClientRect();
        _caliper_input_pointer(e.clientX - r.left, e.clientY - r.top, src(e));
    }
    on("pointermove", move);
    on("pointerdown", function(e) {
        move(e);
        try { c.setPointerCapture(e.pointerId); } catch (x) {}
        c.focus({preventScroll: true});
        _caliper_input_button(e.button, 1, mods(e));
        e.preventDefault();
    });
    on("pointerup", function(e) { move(e); _caliper_input_button(e.button, 0, mods(e)); });
    on("pointercancel", function(e) { _caliper_input_button(e.button, 0, mods(e)); });
    on("pointerleave", function(e) { if (!c.hasPointerCapture(e.pointerId)) _caliper_input_leave(); });
    on("contextmenu", function(e) { e.preventDefault(); });
    on("wheel", function(e) {
        var s = e.deltaMode === 1 ? 1 / 3 : e.deltaMode === 2 ? 3 : 1 / 100;
        move(e);
        if (_caliper_input_wheel(-e.deltaX * s, -e.deltaY * s)) e.preventDefault();
    }, {passive: false});
    function key(e, down) {
        var sp = stackSave();
        var used = _caliper_input_key(stringToUTF8OnStack(e.code), down, mods(e));
        stackRestore(sp);
        if (down && !e.ctrlKey && !e.metaKey && e.key && Array.from(e.key).length === 1) {
            _caliper_input_char(e.key.codePointAt(0));
        }
        if (used) e.preventDefault();
    }
    on("keydown", function(e) { key(e, 1); });
    on("keyup", function(e) { key(e, 0); });
    on("focus", function() { _caliper_input_focus(1); });
    on("blur", function() { _caliper_input_focus(0); });
    if (!c.hasAttribute("tabindex")) c.tabIndex = 0;
    c.style.touchAction = "none";
    c.style.outline = "none";
});

EM_JS(void, caliper_unlisten, (), {
    var c = Module["canvas"];
    var list = Module["caliperListeners"] || [];
    for (var i = 0; i < list.length; ++i) c.removeEventListener(list[i][0], list[i][1], list[i][2]);
    Module["caliperListeners"] = [];
});

EM_JS(void, caliper_set_cursor, (int cursor), {
    var names = ["default", "text", "move", "ns-resize", "ew-resize", "nesw-resize", "nwse-resize",
                 "pointer", "wait", "progress", "not-allowed"];
    Module["canvas"].style.cursor = cursor < 0 ? "none" : (names[cursor] || "default");
});

namespace caliper_web {

void input_install() {
    ImGuiIO& io = ImGui::GetIO();
    io.BackendPlatformName = "caliper_web";
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    caliper_listen();
    g_last_ms = emscripten_get_now();
}

void input_uninstall() { caliper_unlisten(); }

InputCounts input_counts() { return g_counts; }

FrameSize input_new_frame() {
    FrameSize f;
    double w = 0.0, h = 0.0;
    emscripten_get_element_css_size("!canvas", &w, &h);
    const double dpr = emscripten_get_device_pixel_ratio();
    f.css_w = (float)w;
    f.css_h = (float)h;
    f.fb_w = (int)std::lround(w * dpr);
    f.fb_h = (int)std::lround(h * dpr);
    int cw = 0, ch = 0;
    emscripten_get_canvas_element_size("!canvas", &cw, &ch);
    if ((cw != f.fb_w || ch != f.fb_h) && f.fb_w > 0 && f.fb_h > 0)
        emscripten_set_canvas_element_size("!canvas", f.fb_w, f.fb_h);

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(f.css_w, f.css_h);
    io.DisplayFramebufferScale = ImVec2(f.css_w > 0.0f ? (float)f.fb_w / f.css_w : 1.0f,
                                        f.css_h > 0.0f ? (float)f.fb_h / f.css_h : 1.0f);
    const double now = emscripten_get_now();
    io.DeltaTime = now > g_last_ms ? (float)((now - g_last_ms) * 1e-3) : 1.0f / 60.0f;
    g_last_ms = now;
    // ImGui applies one change of a mouse button per frame, so the second press
    // of a double click reaches it two frames after the first, however quick
    // the hand. Below about 7 frames a second (a software adapter) those two
    // frames outlast its 0.3 s double-click time, and the WebAssembly views
    // missed every double click that the element's own views take: allow them.
    io.MouseDoubleClickTime = std::max(0.30f, 1.01f * (g_last_dt + io.DeltaTime));
    g_last_dt = io.DeltaTime;
    return f;
}

void input_end_frame() {
    const ImGuiIO& io = ImGui::GetIO();
    const int c = io.MouseDrawCursor ? -1 : (int)ImGui::GetMouseCursor();
    if (c != g_cursor) {
        g_cursor = c;
        caliper_set_cursor(c == ImGuiMouseCursor_None ? -1 : c);
    }
}

}  // namespace caliper_web
