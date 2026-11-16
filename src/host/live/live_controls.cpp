// live_controls.cpp: see live_controls.h.
#include "live/live_controls.h"

#include "caliper/live_net.h"
#include "live/live_views_state.h"   // the views' payload maps and set_link (linked selections)

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
            // The slider carries no text; the exact value is in its own field beside it.
            ImGui::SetNextItemWidth(kSliderW);
            int64_t lo = (int64_t)std::llround(p.min), hi = (int64_t)std::llround(p.max);
            const ImGuiSliderFlags f = (p.log ? ImGuiSliderFlags_Logarithmic : 0) | ImGuiSliderFlags_AlwaysClamp |
                                       ImGuiSliderFlags_NoInput;
            ImGui::PushStyleColor(ImGuiCol_Text, 0);   // the field beside it writes the value
            const bool moved = ImGui::SliderScalar("##v", ImGuiDataType_S64, &v, &lo, &hi, "%lld", f);
            ImGui::PopStyleColor();
            if (moved) rc.set(p.name, Json(v));
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(v));
            remember("slider:" + p.name);
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::SetNextItemWidth(kFieldW);
            int64_t e = v;
            ImGui::InputScalar("##exact", ImGuiDataType_S64, &e);
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(std::clamp(e, lo, hi)));
        } else {
            ImGui::SetNextItemWidth(kFieldW);
            ImGui::InputScalar("##v", ImGuiDataType_S64, &v);
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(v));
        }
    } else {   // number
        double v = shown.is_number() ? shown.as_double() : 0.0;
        if (p.has_min && p.has_max) {
            // The slider carries no text; the exact value is in its own field beside it.
            ImGui::SetNextItemWidth(kSliderW);
            double lo = p.min, hi = p.max;
            // The format sets the slider's precision (a log slider treats values
            // under its last digit as zero), so it carries one, drawn in no ink.
            const ImGuiSliderFlags f = (p.log ? ImGuiSliderFlags_Logarithmic : 0) | ImGuiSliderFlags_AlwaysClamp |
                                       ImGuiSliderFlags_NoInput | ImGuiSliderFlags_NoRoundToFormat;
            ImGui::PushStyleColor(ImGuiCol_Text, 0);
            const bool moved = ImGui::SliderScalar("##v", ImGuiDataType_Double, &v, &lo, &hi, "%.12g", f);
            ImGui::PopStyleColor();
            if (moved) rc.set(p.name, Json(v));
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(v));
            remember("slider:" + p.name);
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::SetNextItemWidth(kFieldW);
            double e = v;
            ImGui::InputDouble("##exact", &e, 0.0, 0.0, "%.4g");
            if (ImGui::IsItemDeactivatedAfterEdit()) rc.set(p.name, Json(std::clamp(e, lo, hi)));
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
        else if (p.type == CALIPER_NET_PARAM_CHOICE) w += kSliderW;
        else if (p.has_min && p.has_max) w += kSliderW + 4.0f + kFieldW;
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

// ---- the data view picker -------------------------------------------------------------------
namespace {

constexpr int64_t kMaxIndexEntries = 256;   // past this a dimension's index is a field
constexpr uint64_t kPickWaitNs = 2000000000ull;
const char* const kReductions[] = {CALIPER_NET_REDUCE_MEAN, CALIPER_NET_REDUCE_SUM, CALIPER_NET_REDUCE_MAX,
                                   CALIPER_NET_REDUCE_MIN, CALIPER_NET_REDUCE_NORM};

struct DimOffer {
    std::string name;
    int64_t size = 0;
};
struct SourceOffer {
    std::string name;
    std::vector<DimOffer> dims;
};

std::vector<SourceOffer> sources_of(const Json& offer) {
    std::vector<SourceOffer> out;
    const Json* srcs = offer.find("sources");
    if (!srcs || !srcs->is_array()) return out;
    for (const Json& s : srcs->items()) {
        SourceOffer so;
        if (const Json* n = s.find("name"); n && n->is_string()) so.name = n->as_string();
        if (const Json* dims = s.find("dims"); dims && dims->is_array())
            for (const Json& d : dims->items()) {
                DimOffer dm;
                if (const Json* n = d.find("name"); n && n->is_string()) dm.name = n->as_string();
                if (const Json* z = d.find("size")) z->get_i64(&dm.size);
                so.dims.push_back(dm);
            }
        out.push_back(std::move(so));
    }
    return out;
}

int min_kept(uint32_t kind) {
    return kind == CALIPER_LIVE_KIND_HEATMAP || kind == CALIPER_LIVE_KIND_IMAGE ? 2
           : kind == CALIPER_LIVE_KIND_POINTS                                    ? 1
                                                                                 : 0;
}

// A compact menu: the preview and a small chevron in the faint ink, in place
// of ImGui's square arrow button.
bool small_combo(const char* id, const char* preview, float width, ImU32 faint) {
    ImGui::SetNextItemWidth(width);
    const bool open = ImGui::BeginCombo(id, preview, ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge);
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    const float cx = b.x - 8.0f, cy = std::floor((a.y + b.y) * 0.5f) + 0.5f;
    const ImVec2 pts[3] = {ImVec2(cx - 3.5f, cy - 1.5f), ImVec2(cx, cy + 2.0f), ImVec2(cx + 3.5f, cy - 1.5f)};
    ImGui::GetWindowDrawList()->AddPolyline(pts, 3, faint, 0, 1.0f);
    return open;
}

std::string choice_label(int what, int64_t index, const std::string& op) {
    return what == 1 ? std::to_string(index) : what == 2 ? op : "all";
}

}  // namespace

float DataViewPicker::draw(Source* src, uint64_t key, const std::string& view, uint32_t kind, float width,
                           ImU32 faint, float size, uint64_t now_ns) {
    Json resolved, offer;
    if (!src || !key || !src->data_view_of(key, &resolved, &offer)) return 0.0f;
    const std::vector<SourceOffer> sources = sources_of(offer);
    if (sources.empty()) return 0.0f;
    State& st = views_[view];
    // The menus follow what the producer resolved, except while a pick waits.
    const bool fresh = resolved.is_object() && resolved != st.shown;
    if (fresh || (st.source.empty() && st.pending_until == 0) || (st.pending_until && now_ns > st.pending_until)) {
        const Json& dv = resolved.is_object() ? resolved : offer.find("default") ? *offer.find("default") : resolved;
        const Json* sname = dv.is_object() ? dv.find("source") : nullptr;
        st.source = sname && sname->is_string() ? sname->as_string() : sources[0].name;
        st.choices.clear();
        const Json* index = dv.is_object() ? dv.find("index") : nullptr;
        const Json* reduce = dv.is_object() ? dv.find("reduce") : nullptr;
        for (const SourceOffer& so : sources)
            if (so.name == st.source)
                for (const DimOffer& d : so.dims) {
                    Choice c;
                    int64_t i = 0;
                    const Json* ix = index && index->is_object() ? index->find(d.name) : nullptr;
                    const Json* op = reduce && reduce->is_object() ? reduce->find(d.name) : nullptr;
                    if (ix && ix->is_int() && ix->get_i64(&i)) {
                        c.what = 1;
                        c.index = i;
                    } else if (op && op->is_string()) {
                        c.what = 2;
                        c.op = op->as_string();
                    }
                    st.choices[d.name] = c;
                }
        if (resolved.is_object()) st.shown = resolved;
        st.pending_until = 0;
    }
    const SourceOffer* cur = nullptr;
    for (const SourceOffer& so : sources)
        if (so.name == st.source) cur = &so;
    if (!cur) cur = &sources[0];
    const bool choosable = (int)cur->dims.size() > min_kept(kind);
    if (sources.size() < 2 && !choosable) return 0.0f;

    // The data view the menus describe, sent as soon as one changes.
    auto send = [&]() {
        Json index = Json::object(), reduce = Json::object();
        std::vector<std::string> kept;
        for (const DimOffer& d : cur->dims) {
            const Choice& c = st.choices[d.name];
            if (c.what == 1) index.set(d.name, c.index);
            else if (c.what == 2) reduce.set(d.name, c.op);
            else kept.push_back(d.name);
        }
        Json dv = Json::object();
        dv.set("source", cur->name);
        dv.set("index", index);
        dv.set("reduce", reduce);
        if (kind == CALIPER_LIVE_KIND_HEATMAP && kept.size() == 3) dv.set("tile", kept[0]);
        std::string err;
        if (src->request_data_view(key, dv.dump(), &err)) st.pending_until = now_ns + kPickWaitNs;
    };

    if (size > 0.0f) ImGui::PushFont(nullptr, size);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 1.0f));
    ImGui::PushID(("caliper.picker." + view).c_str());
    const ImGuiStyle& gs = ImGui::GetStyle();
    const float x0 = ImGui::GetCursorScreenPos().x, y0 = ImGui::GetCursorScreenPos().y;
    const float right = x0 + width;
    bool first = true;
    // Places a control of width `w` on this row, or on the next when it does not fit.
    auto place = [&](float w) {
        if (first) {
            first = false;
            return;
        }
        ImGui::SameLine(0.0f, 10.0f);
        if (ImGui::GetCursorScreenPos().x + w > right) ImGui::NewLine();
    };
    const float combo_w = ImGui::CalcTextSize("norm").x + 2.0f * gs.FramePadding.x + 22.0f;
    const std::string prefix = "picker:" + view + ":";

    if (sources.size() > 1) {
        float w = 0.0f;
        for (const SourceOffer& so : sources) w = std::max(w, ImGui::CalcTextSize(so.name.c_str()).x);
        w += 2.0f * gs.FramePadding.x + 22.0f;
        place(w);
        if (small_combo("##source", cur->name.c_str(), w, faint)) {
            for (const SourceOffer& so : sources) {
                if (ImGui::Selectable(so.name.c_str(), so.name == cur->name) && so.name != cur->name) {
                    // Another source: its dims keep the choices of dims of the same name.
                    std::map<std::string, Choice> next;
                    for (const DimOffer& d : so.dims) {
                        auto it = st.choices.find(d.name);
                        next[d.name] = it != st.choices.end() ? it->second : Choice{};
                    }
                    st.source = so.name;
                    st.choices = std::move(next);
                    for (const SourceOffer& s2 : sources)
                        if (s2.name == so.name) cur = &s2;
                    send();
                }
                remember(prefix + "source=" + so.name);
            }
            ImGui::EndCombo();
        }
        remember(prefix + "source");
    }
    if (choosable)
        for (const DimOffer& d : cur->dims) {
            Choice& c = st.choices[d.name];
            const bool field = d.size > kMaxIndexEntries;
            const float label_w = ImGui::CalcTextSize(d.name.c_str()).x;
            place(label_w + 4.0f + combo_w + (field && c.what == 1 ? 64.0f : 0.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, faint);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(d.name.c_str());
            ImGui::PopStyleColor();
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::PushID(d.name.c_str());
            const std::string shown = field && c.what == 1 ? "index" : choice_label(c.what, c.index, c.op);
            if (small_combo("##dim", shown.c_str(), combo_w, faint)) {
                auto entry = [&](const std::string& label, const Choice& pick) {
                    if (ImGui::Selectable(label.c_str(), c == pick) && !(c == pick)) {
                        c = pick;
                        send();
                    }
                    remember(prefix + d.name + "=" + label);
                };
                entry("all", Choice{});
                ImGui::SeparatorText("reduce");
                for (const char* op : kReductions) entry(op, Choice{2, 0, op});
                ImGui::SeparatorText("index");
                if (field) {
                    if (ImGui::Selectable("index", c.what == 1) && c.what != 1) {
                        c = Choice{1, 0, ""};
                        send();
                    }
                    remember(prefix + d.name + "=index");
                } else {
                    for (int64_t i = 0; i < d.size; ++i) entry(std::to_string(i), Choice{1, i, ""});
                }
                ImGui::EndCombo();
            }
            remember(prefix + d.name);
            if (field && c.what == 1) {
                ImGui::SameLine(0.0f, 4.0f);
                ImGui::SetNextItemWidth(60.0f);
                int64_t i = c.index;
                ImGui::InputScalar("##index", ImGuiDataType_S64, &i);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    c.index = std::clamp<int64_t>(i, 0, d.size - 1);
                    send();
                }
            }
            ImGui::PopID();
        }
    ImGui::PopID();
    ImGui::PopStyleVar();
    if (size > 0.0f) ImGui::PopFont();
    const float h = ImGui::GetItemRectMax().y - y0 + gs.ItemSpacing.y;
    ImGui::SetCursorScreenPos(ImVec2(x0, y0 + h));
    return h;
}

// ---- linked selection ---------------------------------------------------------------------
namespace {

// The source a view shows now: the resolved data view's, else the offer's first.
std::string shown_source(Source* src, uint64_t key) {
    Json resolved, offer;
    if (!src || !key || !src->data_view_of(key, &resolved, &offer)) return {};
    if (const Json* s = resolved.find("source"); s && s->is_string()) return s->as_string();
    if (const Json* srcs = offer.find("sources"); srcs && srcs->is_array() && srcs->size() > 0)
        if (const Json* n = srcs->at(0).find("name"); n && n->is_string()) return n->as_string();
    return {};
}

// Which axis of view `v` (as the view layer names it) an item axis is: the
// steps of a series, the point dim of points, the rows or columns of a
// heatmap or an image by its payload axes (the cell's two, after the tile of
// a mosaic). "" when it is none of them yet (no snapshot, another source).
const char* link_axis_of(const caliper_live_views::View& v, const ItemAxis& a, const std::string& source) {
    if (a.dim.empty()) return v.kind == CALIPER_LIVE_KIND_SERIES ? "steps" : "";
    if (!a.source.empty() && !source.empty() && a.source != source) return "";
    const auto& ax = v.map.axes;
    if (v.kind == CALIPER_LIVE_KIND_POINTS) return !ax.empty() && ax.back().dim == a.dim ? "points" : "";
    if (v.kind != CALIPER_LIVE_KIND_HEATMAP && v.kind != CALIPER_LIVE_KIND_IMAGE) return "";
    const bool mosaic = v.map.tile_cols > 0 && ax.size() == 3;
    const size_t r = mosaic ? 1 : 0, c = mosaic ? 2 : 1;
    if (ax.size() > r && ax[r].dim == a.dim) return "rows";
    if (ax.size() > c && ax[c].dim == a.dim) return "columns";
    return "";
}

}  // namespace

void RunControls::link_views(CaliperCore* core, const std::function<uint64_t(const std::string&)>& key_of) {
    if (!core || !core->live_views) return;
    for (const auto& vp : core->live_views->views) {
        const caliper_live_views::View& v = *vp;
        const std::string source = v.kind == CALIPER_LIVE_KIND_SERIES ? std::string() : shown_source(src_, key_of(v.name));
        // The newest selection of a space that one of the view's axes carries.
        const LinkedSelection* best = nullptr;
        const char* axis = "";
        for (const ItemAxis& a : item_axes_) {
            if (a.view != v.name) continue;
            const char* ax = link_axis_of(v, a, source);
            if (!*ax) continue;
            for (const LinkedSelection& s : selections_)
                if (s.space == a.space && (!best || s.seq > best->seq)) {
                    best = &s;
                    axis = ax;
                }
        }
        const std::string sig = best ? std::string(axis) + "|" + best->space + "|" + std::to_string(best->seq) : "";
        auto it = linked_.find(v.name);
        if (it != linked_.end() ? it->second == sig : sig.empty()) continue;
        std::vector<int64_t> items;
        if (best) items.assign(best->items.begin(), best->items.end());
        if (caliper_live_views::set_link(core, v.name.c_str(), best ? axis : "", std::move(items))) linked_[v.name] = sig;
    }
}

}  // namespace caliper_live
