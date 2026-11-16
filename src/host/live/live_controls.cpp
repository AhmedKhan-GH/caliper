// live_controls.cpp: see live_controls.h.
#include "live/live_controls.h"

#include "caliper/live_net.h"
#include "live/live_views_state.h"   // the views' payload map and geometry (the selection hook)

#include <imgui_internal.h>   // SeparatorEx

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace caliper_live {

using caliper_net::Json;

namespace {

constexpr uint64_t kSetIntervalNs = 1000000000ull / CALIPER_NET_CONTROL_RATE_HZ;
constexpr uint64_t kResultShowNs = 15ull * 1000000000ull;   // an action's result stays in the bar
constexpr uint64_t kNoticeShowNs = 10ull * 1000000000ull;   // and a refusal
constexpr size_t kMaxLog = 256;

std::map<std::string, WidgetRect> g_widgets;   // the bar's widgets, last frame

void remember(const std::string& name) {
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    g_widgets[name] = WidgetRect{a.x, a.y, b.x, b.y};
}

// A value as the bar writes it.
std::string text_of(const Json& v) {
    char buf[64];
    if (v.is_bool()) return v.as_bool() ? "on" : "off";
    if (v.is_string()) return v.as_string();
    if (v.is_int()) {
        int64_t i = 0;
        if (v.get_i64(&i)) {
            std::snprintf(buf, sizeof buf, "%lld", (long long)i);
            return buf;
        }
    }
    if (v.is_number()) {
        std::snprintf(buf, sizeof buf, "%.4g", v.as_double());
        return buf;
    }
    return "-";
}

bool same_value(const Json& a, const Json& b) {
    if (a.is_number() && b.is_number()) return a.as_double() == b.as_double();
    return a == b;
}

std::string reply_error(const ControlReply& r) {
    std::string s = r.error_code.empty() ? "refused" : r.error_code;
    if (!r.error_message.empty()) s += ": " + r.error_message;
    return s;
}

std::string state_text(const Controls& c) {
    std::string s;
    const std::string& st = c.run_state;
    if (st == CALIPER_NET_RUN_NONE) s = "no checkpoint yet";
    else if (st == CALIPER_NET_RUN_STEPPING) s = "stepping, " + std::to_string(c.steps_left) + " left";
    else s = st;
    if (c.has_step && st != CALIPER_NET_RUN_NONE) s += " at step " + std::to_string(c.step);
    return s;
}

// ---- look fallbacks ---------------------------------------------------------------------
bool look_button(const ControlsLook& look, const char* label, bool enabled, const char* reason) {
    if (look.button) return look.button(label, enabled, reason);
    ImGui::BeginDisabled(!enabled);
    const bool clicked = ImGui::Button(label);
    ImGui::EndDisabled();
    if (!enabled && reason && *reason && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", reason);
    return clicked && enabled;
}

void look_label(const ControlsLook& look, const char* text) {
    if (look.label) {
        look.label(text);
        return;
    }
    ImGui::PushStyleColor(ImGuiCol_Text, look.secondary);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void coloured(ImU32 c, const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, c);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void vertical_rule() {
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();
}

// Text cut to `max_w` with an ellipsis; the whole text in a tooltip when cut.
void fitted(ImU32 c, const std::string& s, float max_w) {
    if (max_w <= 8.0f) return;
    if (ImGui::CalcTextSize(s.c_str()).x <= max_w) {
        coloured(c, s.c_str());
        return;
    }
    std::string cut = s;
    while (!cut.empty() && ImGui::CalcTextSize((cut + "...").c_str()).x > max_w) cut.pop_back();
    coloured(c, (cut + "...").c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", s.c_str());
}

constexpr float kSliderW = 120.0f;
constexpr float kFieldW = 90.0f;

// One parameter: its label, then its control by type, then whether a value waits.
void draw_param(RunControls& rc, const ControlsLook& look, const ControlParam& p) {
    ImGui::PushID(p.name.c_str());
    look_label(look, p.name.c_str());
    ImGui::SameLine();
    const std::string why = rc.why_not(CALIPER_NET_OP_SET);
    const bool enabled = why.empty();
    const Json shown = rc.shown_value(p);
    ImGui::BeginDisabled(!enabled);
    if (p.type == CALIPER_NET_PARAM_BOOLEAN) {
        bool on = shown.is_bool() && shown.as_bool();
        if (ImGui::Checkbox("##v", &on)) rc.set(p.name, Json(on));
    } else if (p.type == CALIPER_NET_PARAM_CHOICE) {
        const std::string cur = shown.is_string() ? shown.as_string() : "";
        ImGui::SetNextItemWidth(kSliderW);
        if (ImGui::BeginCombo("##v", cur.c_str())) {
            for (const std::string& c : p.choices)
                if (ImGui::Selectable(c.c_str(), c == cur)) rc.set(p.name, Json(c));
            ImGui::EndCombo();
        }
    } else if (p.type == CALIPER_NET_PARAM_INTEGER) {
        int64_t v = 0;
        if (shown.is_number()) v = (int64_t)std::llround(shown.as_double());
        if (p.has_min && p.has_max) {
            ImGui::SetNextItemWidth(kSliderW);
            int64_t lo = (int64_t)std::llround(p.min), hi = (int64_t)std::llround(p.max);
            const ImGuiSliderFlags f = (p.log ? ImGuiSliderFlags_Logarithmic : 0) | ImGuiSliderFlags_AlwaysClamp;
            if (ImGui::SliderScalar("##v", ImGuiDataType_S64, &v, &lo, &hi, "%lld", f)) rc.set(p.name, Json(v));
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(v));
        } else {
            ImGui::SetNextItemWidth(kFieldW);
            ImGui::InputScalar("##v", ImGuiDataType_S64, &v);
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(v));
        }
    } else {   // number
        double v = shown.is_number() ? shown.as_double() : 0.0;
        if (p.has_min && p.has_max) {
            ImGui::SetNextItemWidth(kSliderW);
            double lo = p.min, hi = p.max;
            const ImGuiSliderFlags f = (p.log ? ImGuiSliderFlags_Logarithmic : 0) | ImGuiSliderFlags_AlwaysClamp;
            if (ImGui::SliderScalar("##v", ImGuiDataType_Double, &v, &lo, &hi, "%.4g", f)) rc.set(p.name, Json(v));
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(v));
        } else {
            ImGui::SetNextItemWidth(kFieldW);
            ImGui::InputDouble("##v", &v, 0.0, 0.0, "%.4g");
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(v));
        }
    }
    ImGui::EndDisabled();
    remember("param:" + p.name);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        std::string tip = p.help;
        if (!enabled) tip += (tip.empty() ? "" : "\n") + why;
        if (!tip.empty()) ImGui::SetTooltip("%s", tip.c_str());
    }
    if (rc.param_waiting(p) || rc.param_pending(p)) {
        ImGui::SameLine();
        coloured(look.faint, "pending");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", rc.param_waiting(p) ? "the script has not confirmed this value yet"
                                                        : "the script reads it at its next checkpoint");
    }
    ImGui::PopID();
}

// One action: its button, then its state or its last result.
void draw_action(RunControls& rc, const ControlsLook& look, const ControlAction& a, float result_w) {
    ImGui::PushID(a.name.c_str());
    const bool busy = rc.action_busy(a);
    std::string why = rc.why_not(CALIPER_NET_OP_RUN);
    if (why.empty() && busy) why = a.running ? "it runs now" : "it waits for the script's next checkpoint";
    if (look_button(look, a.name.c_str(), why.empty(), why.c_str())) rc.run(a.name);
    remember("action:" + a.name);
    if (why.empty() && !a.help.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", a.help.c_str());
    const RunControls::Result* r = rc.action_result(a.name);
    if (busy) {
        ImGui::SameLine();
        coloured(look.secondary, a.running ? "running" : "queued");
    } else if (r && result_w > 0.0f && rc.now_ns() - r->at_ns < kResultShowNs) {
        ImGui::SameLine();
        fitted(r->error ? look.text : look.secondary, r->error ? "FAILED: " + r->text : r->text, result_w);
    }
    ImGui::PopID();
}

}  // namespace

// ---- RunControls ------------------------------------------------------------------------------
void RunControls::reset() {
    *this = RunControls();
}

void RunControls::update(Source* src, uint64_t now_ns) {
    src_ = src;
    now_ns_ = now_ns;
    if (!src) {
        if (have_ || !locals_.empty()) log_.push_back("controls gone: not attached");
        have_ = false;
        controls_ = Controls{};
        access_ = ControlAccess{false, "not attached to a session"};
        locals_.clear();
        runs_.clear();
        ops_.clear();
        selections_.clear();
        item_axes_.clear();
        return;
    }
    const ControlAccess access = src->control_access();
    if (access.allowed != access_.allowed || access.why != access_.why)
        log_.push_back(access.allowed ? "controls on" : "controls off: " + access.why);
    access_ = access;
    // Replies first: each carries the CONTROLS version it arrived after, so a
    // CONTROLS read below confirms only what came after its reply.
    for (const ControlReply& r : src->take_control_replies()) on_reply(r);
    Controls c;
    if (src->controls(&c)) {
        if (!have_ || c.version != controls_.version) {
            const Controls before = controls_;
            const bool first = !have_;
            controls_ = std::move(c);
            have_ = true;
            on_controls(first ? Controls{} : before);
        }
    } else if (have_) {
        have_ = false;
        controls_ = Controls{};
        locals_.clear();
        log_.push_back("controls gone");
    }
    // A value is confirmed by a CONTROLS that arrived after its `ok`.
    for (auto it = locals_.begin(); it != locals_.end();) {
        Local& l = it->second;
        if (l.unsent && now_ns_ - l.sent_ns >= kSetIntervalNs) send_set(it->first, l);
        if (!l.unsent && l.replied && controls_.version > l.reply_controls) it = locals_.erase(it);
        else ++it;
    }
    std::vector<LinkedSelection> sel;
    const uint64_t v = src->selections(&sel);
    if (v != sel_version_) {
        // The space whose selection arrived last decides the details shown.
        uint64_t newest = 0;
        for (const LinkedSelection& s : sel) {
            const LinkedSelection* old = nullptr;
            for (const LinkedSelection& o : selections_)
                if (o.space == s.space) old = &o;
            if (!old || old->seq != s.seq) {
                if (s.seq >= newest) {
                    newest = s.seq;
                    details_space_ = s.space;
                }
                if (old && old->items == s.items && old->details == s.details &&
                    old->details_error == s.details_error)
                    continue;   // sent again unchanged
                char line[160];
                std::snprintf(line, sizeof line, "selection %s: %zu item%s%s", s.space.c_str(), s.items.size(),
                              s.items.size() == 1 ? "" : "s",
                              s.details.is_object() ? ", with details"
                              : !s.details_error.empty() ? ", details failed" : "");
                log_.push_back(line);
            }
        }
        selections_ = std::move(sel);
        sel_version_ = v;
    }
    item_axes_ = src->item_axes();
    if (log_.size() > kMaxLog) log_.erase(log_.begin(), log_.begin() + (std::ptrdiff_t)(log_.size() - kMaxLog));
}

void RunControls::on_controls(const Controls& before) {
    if (before.run_state != controls_.run_state) {
        std::string line = "run " + controls_.run_state;
        if (controls_.has_step) line += " at step " + std::to_string(controls_.step);
        log_.push_back(line);
    }
    for (const ControlParam& p : controls_.params) {
        const ControlParam* old = nullptr;
        for (const ControlParam& o : before.params)
            if (o.name == p.name) old = &o;
        if (!old || !same_value(old->value, p.value)) log_.push_back("param " + p.name + " = " + text_of(p.value));
        if (!p.pending.is_null() && (!old || !same_value(old->pending, p.pending)))
            log_.push_back("param " + p.name + " pending " + text_of(p.pending));
    }
}

std::string RunControls::why_not(const char* op) const {
    if (!access_.allowed) return access_.why;
    if (!have_) return "the script has sent no controls";
    const bool needs_checkpoint = std::strcmp(op, CALIPER_NET_OP_PAUSE) == 0 ||
                                  std::strcmp(op, CALIPER_NET_OP_STEP) == 0 ||
                                  std::strcmp(op, CALIPER_NET_OP_RESUME) == 0 ||
                                  std::strcmp(op, CALIPER_NET_OP_RUN) == 0;
    if (needs_checkpoint && controls_.run_state == CALIPER_NET_RUN_NONE)
        return "the script has called no checkpoint yet";
    return "";
}

void RunControls::refuse(const std::string& what, const std::string& why) {
    notice_ = Result{what + ": " + why, true, now_ns_};
    log_.push_back("refused " + what + ": " + why);
}

uint32_t RunControls::send(const char* op, const std::string& name, const Json& value, uint32_t count) {
    const std::string why = why_not(op);
    const std::string what = name.empty() ? std::string(op) : std::string(op) + " " + name;
    if (!why.empty()) {
        refuse(what, why);
        return 0;
    }
    std::string err;
    const uint32_t id = src_ ? src_->send_control(op, name, value, count, &err) : 0;
    if (!id) {
        refuse(what, err.empty() ? "not sent" : err);
        return 0;
    }
    ++counts_.sent;
    return id;
}

bool RunControls::pause() {
    const uint32_t id = send(CALIPER_NET_OP_PAUSE, "", Json(), 0);
    if (id) ops_[id] = CALIPER_NET_OP_PAUSE;
    return id != 0;
}

bool RunControls::resume() {
    const uint32_t id = send(CALIPER_NET_OP_RESUME, "", Json(), 0);
    if (id) ops_[id] = CALIPER_NET_OP_RESUME;
    return id != 0;
}

bool RunControls::step(uint32_t count) {
    const uint32_t id = send(CALIPER_NET_OP_STEP, "", Json(), count);
    if (id) ops_[id] = CALIPER_NET_OP_STEP;
    return id != 0;
}

bool RunControls::run(const std::string& name) {
    const uint32_t id = send(CALIPER_NET_OP_RUN, name, Json(), 0);
    if (id) {
        runs_[id] = name;
        results_.erase(name);
    }
    return id != 0;
}

bool RunControls::set(const std::string& name, const Json& value) {
    const std::string why = why_not(CALIPER_NET_OP_SET);
    if (!why.empty()) {
        refuse("set " + name, why);
        return false;
    }
    // The value the control already shows (a slider held still, the edit
    // that ends a drag) is not sent again.
    const auto it = locals_.find(name);
    if (it == locals_.end()) {
        for (const ControlParam& p : controls_.params)
            if (p.name == name && same_value(shown_value(p), value)) return true;
    } else if (!it->second.unsent && same_value(it->second.value, value)) {
        return true;
    }
    Local& l = locals_[name];
    l.value = value;
    l.unsent = true;
    if (now_ns_ - l.sent_ns >= kSetIntervalNs || l.request == 0) send_set(name, l);
    return true;
}

void RunControls::send_set(const std::string& name, Local& l) {
    const uint32_t id = send(CALIPER_NET_OP_SET, name, l.value, 0);
    l.unsent = false;
    l.sent_ns = now_ns_;
    if (!id) {
        l.replied = true;   // dropped at the next update
        l.reply_controls = 0;
        l.request = 0;
        return;
    }
    ++counts_.sets;
    l.request = id;
    l.replied = false;
}

void RunControls::on_reply(const ControlReply& r) {
    ++counts_.replies;
    if (r.status == CALIPER_NET_STATUS_ERROR) ++counts_.refused;
    if (r.op == CALIPER_NET_OP_SET) {
        for (auto it = locals_.begin(); it != locals_.end(); ++it) {
            Local& l = it->second;
            if (l.request != r.request_id) continue;
            if (r.status == CALIPER_NET_STATUS_ERROR) {
                refuse("set " + it->first, reply_error(r));
                if (!l.unsent) locals_.erase(it);
                return;
            }
            if (!r.value.is_null()) l.value = r.value;   // the clamped value
            l.replied = true;
            l.reply_controls = r.controls_version;
            log_.push_back("set " + it->first + " = " + text_of(l.value));
            return;
        }
        if (r.status == CALIPER_NET_STATUS_ERROR) refuse("set " + r.name, reply_error(r));
        return;
    }
    if (r.op == CALIPER_NET_OP_RUN) {
        const auto it = runs_.find(r.request_id);
        const std::string name = it != runs_.end() ? it->second : r.name;
        if (r.status == CALIPER_NET_STATUS_QUEUED) return;   // one more reply follows
        if (it != runs_.end()) runs_.erase(it);
        if (r.status == CALIPER_NET_STATUS_OK) {
            results_[name] = Result{r.message.empty() ? "done" : r.message, false, now_ns_};
            log_.push_back(name + ": " + results_[name].text);
        } else {
            results_[name] = Result{reply_error(r), true, now_ns_};
            log_.push_back(name + " failed: " + results_[name].text);
        }
        return;
    }
    const auto it = ops_.find(r.request_id);
    const std::string op = it != ops_.end() ? it->second : r.op;
    if (it != ops_.end()) ops_.erase(it);
    if (r.status == CALIPER_NET_STATUS_ERROR) refuse(op, reply_error(r));
}

Json RunControls::shown_value(const ControlParam& p) const {
    const auto it = locals_.find(p.name);
    if (it != locals_.end()) return it->second.value;
    if (!p.pending.is_null()) return p.pending;
    return p.value;
}

bool RunControls::param_waiting(const ControlParam& p) const {
    return locals_.count(p.name) != 0;
}

bool RunControls::action_busy(const ControlAction& a) const {
    if (a.running || a.queued) return true;
    for (const auto& kv : runs_)
        if (kv.second == a.name) return true;
    return false;
}

const RunControls::Result* RunControls::action_result(const std::string& name) const {
    const auto it = results_.find(name);
    return it == results_.end() ? nullptr : &it->second;
}

const LinkedSelection* RunControls::details() const {
    for (const LinkedSelection& s : selections_)
        if (s.space == details_space_ && !s.items.empty() && (s.details.is_object() || !s.details_error.empty()))
            return &s;
    return nullptr;
}

std::vector<std::string> RunControls::take_log() {
    std::vector<std::string> out;
    out.swap(log_);
    return out;
}

// ---- drawing ----------------------------------------------------------------------------
const std::map<std::string, WidgetRect>& bar_widgets() { return g_widgets; }

void draw_run_bar(RunControls& rc, const ControlsLook& look, float width) {
    g_widgets.clear();
    const float x0 = ImGui::GetCursorScreenPos().x;
    const float right = x0 + width;
    const ImGuiStyle& st = ImGui::GetStyle();
    const Controls& c = rc.state();

    // The run state, then the run buttons.
    coloured(look.secondary, rc.have() ? state_text(c).c_str() : "no controls");
    ImGui::SameLine(0.0f, st.ItemSpacing.x * 2.0f);
    const bool paused = c.run_state == CALIPER_NET_RUN_PAUSED;
    if (paused) {
        const std::string why = rc.why_not(CALIPER_NET_OP_RESUME);
        if (look_button(look, "Resume", why.empty(), why.c_str())) rc.resume();
        remember("resume");
    } else {
        std::string why = rc.why_not(CALIPER_NET_OP_PAUSE);
        if (why.empty() && c.run_state == CALIPER_NET_RUN_PAUSING) why = "the script pauses at its next checkpoint";
        if (look_button(look, "Pause", why.empty(), why.c_str())) rc.pause();
        remember("pause");
    }
    ImGui::SameLine();
    {
        const std::string why = rc.why_not(CALIPER_NET_OP_STEP);
        if (look_button(look, "Step", why.empty(), why.c_str())) rc.step(1);
        remember("step");
    }

    // The parameters and actions that fit, then "+N more" for the rest.
    struct Item {
        const ControlParam* p = nullptr;
        const ControlAction* a = nullptr;
        float w = 0.0f;
    };
    std::vector<Item> items;
    const float pending_w = ImGui::CalcTextSize("pending").x + st.ItemSpacing.x;
    for (const ControlParam& p : c.params) {
        float w = ImGui::CalcTextSize(p.name.c_str()).x + st.ItemSpacing.x;
        if (p.type == CALIPER_NET_PARAM_BOOLEAN) w += ImGui::GetFrameHeight();
        else if (p.type == CALIPER_NET_PARAM_CHOICE || (p.has_min && p.has_max)) w += kSliderW;
        else w += kFieldW;
        items.push_back({&p, nullptr, w + pending_w});
    }
    const float result_w = 180.0f;
    for (const ControlAction& a : c.actions)
        items.push_back({nullptr, &a,
                         ImGui::CalcTextSize(a.name.c_str()).x + 2.0f * st.FramePadding.x + 16.0f +
                             ImGui::CalcTextSize("running").x + st.ItemSpacing.x});
    const float more_w = ImGui::CalcTextSize("+99 more").x + 2.0f * st.FramePadding.x + st.ItemSpacing.x;
    const float notice_min = 160.0f;
    // Every item starts with SameLine, so the bar stays one line and the
    // cursor ends after its last item.
    size_t shown = 0;
    float x = ImGui::GetItemRectMax().x;
    for (; shown < items.size(); ++shown) {
        const bool last = shown + 1 == items.size();
        const float need = 2.0f * st.ItemSpacing.x + items[shown].w + (last ? 0.0f : more_w);
        if (x + need > right - notice_min) break;
        if (shown == 0 || (items[shown].a && items[shown - 1].p)) vertical_rule();
        else ImGui::SameLine(0.0f, st.ItemSpacing.x * 2.0f);
        if (items[shown].p) draw_param(rc, look, *items[shown].p);
        else draw_action(rc, look, *items[shown].a, result_w);
        x = ImGui::GetItemRectMax().x;
    }
    if (shown < items.size()) {
        if (shown == 0) vertical_rule();
        else ImGui::SameLine(0.0f, st.ItemSpacing.x * 2.0f);
        char label[32];
        std::snprintf(label, sizeof label, "+%zu more", items.size() - shown);
        if (look_button(look, label, true, nullptr)) ImGui::OpenPopup("##caliper_controls_more");
        remember("more");
        if (ImGui::BeginPopup("##caliper_controls_more")) {
            for (size_t i = shown; i < items.size(); ++i) {
                if (items[i].p) draw_param(rc, look, *items[i].p);
                else draw_action(rc, look, *items[i].a, 320.0f);
            }
            ImGui::EndPopup();
        }
    }

    // Right: the last refusal while it is fresh, else why the controls are off.
    std::string note;
    ImU32 note_c = look.faint;
    const RunControls::Result& n = rc.notice();
    const std::string off = rc.why_not(CALIPER_NET_OP_SET);
    if (!off.empty()) note = "controls off: " + off;
    if (!n.text.empty() && rc.now_ns() - n.at_ns < kNoticeShowNs) {
        note = n.text;
        note_c = look.text;
    }
    if (!note.empty()) {
        ImGui::SameLine(0.0f, st.ItemSpacing.x * 2.0f);
        fitted(note_c, note, right - ImGui::GetCursorScreenPos().x);
    }
}

void draw_selection_details(const RunControls& rc, const ControlsLook& look) {
    const LinkedSelection* s = rc.details();
    if (!s) return;
    char head[160];
    std::snprintf(head, sizeof head, "%s: %zu%s selected%s", s->space.c_str(), s->items.size(),
                  s->truncated ? "+" : "", s->view.empty() ? " by the script" : (" in " + s->view).c_str());
    coloured(look.secondary, head);
    if (!s->details.is_object()) {
        coloured(look.text, ("no details: " + s->details_error).c_str());
        return;
    }
    const Json* fields = s->details.find("fields");
    const Json* ids = s->details.find("items");
    const Json* values = s->details.find("values");
    if (!fields || !fields->is_array() || !ids || !ids->is_array() || !values || !values->is_array()) return;
    const int cols = 1 + (int)fields->size();
    const bool open = look.begin_table ? look.begin_table("##caliper_selection_details", cols)
                                       : ImGui::BeginTable("##caliper_selection_details", cols,
                                                           ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                                               ImGuiTableFlags_BordersInnerH);
    if (!open) return;
    ImGui::TableSetupColumn("item");
    for (const Json& f : fields->items())
        ImGui::TableSetupColumn(f.is_string() ? f.as_string().c_str() : "?");
    if (look.header_row) look.header_row();
    else ImGui::TableHeadersRow();
    for (size_t r = 0; r < ids->size() && r < values->size(); ++r) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(text_of(ids->at(r)).c_str());
        const Json& row = values->at(r);
        for (size_t f = 0; f < fields->size(); ++f) {
            ImGui::TableNextColumn();
            const Json v = row.is_array() && f < row.size() ? row.at(f) : Json();
            if (v.is_bool()) ImGui::TextUnformatted(v.as_bool() ? "true" : "false");
            else ImGui::TextUnformatted(text_of(v).c_str());
        }
    }
    if (look.end_table) look.end_table();
    else ImGui::EndTable();
    if (s->items.size() > ids->size()) {
        char more[96];
        std::snprintf(more, sizeof more, "details of the first %zu of %zu items", ids->size(), s->items.size());
        coloured(look.faint, more);
    }
}

// ==== THE SELECTION HOOK =================================================================
namespace {

// The payload positions [p0, p1) that show source index `item` along `ax`
// (wire protocol v3, "Payload axes"); false when the item is not shown.
bool payload_cell(const PayloadAxis& ax, uint64_t item, int64_t* cell) {
    const int64_t step = ax.step > 0 ? ax.step : 1;
    const int64_t i = (int64_t)item - ax.start;
    if (i < 0 || i % step != 0) return false;
    const int64_t k = i / step;   // position among the selected source indices
    if (k >= ax.count || ax.count <= 0) return false;
    *cell = ax.size > 0 ? k * ax.size / ax.count : k;
    return true;
}

// Consecutive runs [a, b) of a sorted list of cells.
std::vector<std::pair<int64_t, int64_t>> runs_of(std::vector<int64_t> cells) {
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    std::vector<std::pair<int64_t, int64_t>> out;
    for (int64_t c : cells) {
        if (!out.empty() && out.back().second == c) out.back().second = c + 1;
        else out.push_back({c, c + 1});
    }
    return out;
}

}  // namespace

void draw_linked_selection(CaliperCore* core, const RunControls& rc, const ControlsLook& look) {
    if (!core) return;
    const ImU32 fill = (look.accent & ~IM_COL32_A_MASK) | ((ImU32)56 << IM_COL32_A_SHIFT);
    const ImU32 edge = (look.accent & ~IM_COL32_A_MASK) | ((ImU32)200 << IM_COL32_A_SHIFT);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    for (const LinkedSelection& s : rc.selections()) {
        if (s.items.empty()) continue;
        for (const ItemAxis& a : rc.item_axes()) {
            if (a.space != s.space) continue;
            const caliper_live_views::View* v = caliper_live_views::find(core, a.view.c_str());
            if (!v || !v->geo_valid) continue;
            const float ax0 = v->geo_area[0], ay0 = v->geo_area[1];
            const float aw = v->geo_area[2], ah = v->geo_area[3];
            const double wx0 = v->geo_window[0], wx1 = v->geo_window[1];
            const double wy0 = v->geo_window[2], wy1 = v->geo_window[3];
            if (aw <= 0 || ah <= 0 || wx1 == wx0 || wy1 == wy0) continue;
            dl->PushClipRect(ImVec2(ax0, ay0), ImVec2(ax0 + aw, ay0 + ah), true);
            auto sx = [&](double x) { return ax0 + (float)((x - wx0) / (wx1 - wx0)) * aw; };
            auto sy = [&](double y) {
                const float t = (float)((y - wy0) / (wy1 - wy0));
                return v->geo_ydown ? ay0 + t * ah : ay0 + ah - t * ah;
            };
            if (v->kind == CALIPER_LIVE_KIND_SERIES && a.dim.empty()) {
                // A series: the steps, as vertical bands.
                std::vector<int64_t> steps(s.items.begin(), s.items.end());
                for (const auto& r : runs_of(std::move(steps))) {
                    const float xa = sx((double)r.first - 0.5), xb = sx((double)r.second - 0.5);
                    dl->AddRectFilled(ImVec2(std::min(xa, xb), ay0), ImVec2(std::max(xa, xb), ay0 + ah), fill);
                }
            } else if ((v->kind == CALIPER_LIVE_KIND_HEATMAP || v->kind == CALIPER_LIVE_KIND_IMAGE) &&
                       !(v->map.tile_cols > 0)) {
                // Rows or columns of a heatmap or an image (a pooled cell stands
                // for several items and shows when any of them is selected).
                const auto& axes = v->map.axes;
                // The payload's own extent: a band covers the image, not the
                // margins its aspect leaves in the area.
                const double rows = axes.size() > 0 && axes[0].size > 0 ? (double)axes[0].size : wy1;
                const double cols = axes.size() > 1 && axes[1].size > 0 ? (double)axes[1].size : wx1;
                for (size_t i = 0; i < axes.size() && i < 2; ++i) {
                    if (axes[i].dim != a.dim) continue;
                    std::vector<int64_t> cells;
                    for (uint64_t item : s.items) {
                        int64_t cell = 0;
                        if (payload_cell(axes[i], item, &cell)) cells.push_back(cell);
                    }
                    for (const auto& r : runs_of(std::move(cells))) {
                        ImVec2 p0, p1;
                        if (i == 0) {   // rows
                            const float ya = sy((double)r.first), yb = sy((double)r.second);
                            const float xa = sx(0.0), xb = sx(cols);
                            p0 = ImVec2(std::min(xa, xb), std::min(ya, yb));
                            p1 = ImVec2(std::max(xa, xb), std::max(ya, yb));
                        } else {        // columns
                            const float xa = sx((double)r.first), xb = sx((double)r.second);
                            const float ya = sy(0.0), yb = sy(rows);
                            p0 = ImVec2(std::min(xa, xb), std::min(ya, yb));
                            p1 = ImVec2(std::max(xa, xb), std::max(ya, yb));
                        }
                        dl->AddRectFilled(p0, p1, fill);
                        dl->AddRect(p0, p1, edge);
                    }
                }
            }
            dl->PopClipRect();
        }
    }
}

}  // namespace caliper_live
