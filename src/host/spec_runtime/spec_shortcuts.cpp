// The keyboard shortcuts of an instrument (spec_shortcuts.h,
// design/instrument-interface.md, Keyboard Shortcuts).
//
// A choice of a segmented group may have a key: one letter or digit with an
// optional Shift. The key chooses its choice, or turns it on or off in a
// group with `multiple`, through the session's values, as a click does. The
// `control` event it adds carries the key. A key acts only while the
// runtime's window has focus, no text field holds the keyboard and no
// Command, Control or Option key is down; the validator keeps the app's own
// keys out of the spec (spec.reserved_shortcut).
//
// The runtime calls handle_instrument_keys() once a frame before it draws the
// controls, and so it also hands the session's values the cursor of the kept
// steps, which the bounds of a range may read.
#include "spec_runtime/spec_shortcuts.h"

#include <caliper_builder/spec/control.h>

#include <string>

namespace caliper_host {

namespace {

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using nlohmann::json;

// Passes events on to the ring, adding the key to a `control` event.
class KeyedSink final : public cb::EventSink {
public:
    KeyedSink(cb::EventSink* inner, std::string key) : inner_(inner), key_(std::move(key)) {}
    std::uint64_t push(cb::Event e) override {
        if (!inner_) return 0;
        if (e.kind == cb::EventKind::control) e.fields["key"] = key_;
        return inner_->push(std::move(e));
    }

private:
    cb::EventSink* inner_;
    std::string key_;
};

// The ImGui key of a shortcut's letter or digit, or ImGuiKey_None.
ImGuiKey key_of(char c) {
    if (c >= 'a' && c <= 'z') return static_cast<ImGuiKey>(ImGuiKey_A + (c - 'a'));
    if (c >= '0' && c <= '9') return static_cast<ImGuiKey>(ImGuiKey_0 + (c - '0'));
    return ImGuiKey_None;
}

bool same(const json& a, const json& b) {
    if (a.is_number() && b.is_number()) return a.get<double>() == b.get<double>();
    return a == b;
}

// The value of group `c` after its choice `ch` is pressed.
json pressed_value(const spec::Control& c, const spec::Choice& ch, const json& now) {
    if (!(c.options.multiple && *c.options.multiple)) return ch.value;
    bool on = false;
    if (now.is_array())
        for (const auto& v : now) on = on || same(v, ch.value);
    json list = json::array();
    for (const auto& other : c.options.choices) {
        bool keep = false;
        if (now.is_array())
            for (const auto& v : now) keep = keep || same(v, other.value);
        if (&other == &ch) keep = !on;
        if (keep) list.push_back(other.value);
    }
    return list;
}

}  // namespace

bool handle_instrument_keys(const caliper_builder::spec::Spec& spec, SpecLiveHost& host) {
    if (host.values && host.history) host.values->set_cursor(host.history->cursor());
    if (!host.values) return false;
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput || io.KeyCtrl || io.KeySuper || io.KeyAlt) return false;
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) return false;
    bool changed = false;
    for (const spec::Control& c : spec.controls) {
        if (c.kind != spec::ControlKind::segmented) continue;
        for (const spec::Choice& ch : c.options.choices) {
            if (!ch.key) continue;
            std::string k = *ch.key;
            const bool shift = k.rfind("shift+", 0) == 0;
            if (shift) k = k.substr(6);
            if (k.size() != 1 || shift != io.KeyShift) continue;
            const ImGuiKey key = key_of(k[0]);
            if (key == ImGuiKey_None || !ImGui::IsKeyPressed(key, false)) continue;
            KeyedSink sink(host.events, *ch.key);
            const json next = pressed_value(c, ch, host.values->control_value(c.id));
            changed = host.values->set_control(c.id, next, cb::CommandOrigin::ui, &sink).ok() || changed;
        }
    }
    return changed;
}

}  // namespace caliper_host
