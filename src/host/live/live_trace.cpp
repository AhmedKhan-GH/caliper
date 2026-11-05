#include "live/live_trace.h"
#include "ui/sternbild_theme.h"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace caliper_live {
namespace {
using Json = crude_json::value;
using Object = crude_json::object;
using Array = crude_json::array;

const Json& field(const Json& object, const char* key) {
    static const Json empty;
    return object.is_object() && object.contains(key) ? object[key] : empty;
}
int integer(const Json& value, int fallback) {
    if (!value.is_number()) return fallback;
    const double n = value.get<double>();
    return std::isfinite(n) && n >= 0 && n <= std::numeric_limits<int32_t>::max() && std::floor(n) == n
        ? static_cast<int>(n) : fallback;
}
bool ints(const Json& value, size_t count, std::vector<int32_t>& out) {
    if (!value.is_array() || value.get<Array>().size() != count) return false;
    out.clear(); out.reserve(count);
    for (const Json& v : value.get<Array>()) {
        const int n = integer(v, -1);
        if (n < 0) return false;
        out.push_back(n);
    }
    return true;
}
bool probabilities(const Json& value, size_t count, std::vector<float>& out) {
    if (!value.is_array() || value.get<Array>().size() != count) return false;
    out.clear(); out.reserve(count);
    for (const Json& v : value.get<Array>()) {
        if (!v.is_number() || !(v.get<double>() >= 0) || v.get<double>() > 1.0001) return false;
        out.push_back(static_cast<float>(v.get<double>()));
    }
    return true;
}
ImVec2 point(double x, double y) { return ImVec2(static_cast<float>(x), static_cast<float>(y)); }
void faint(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}
void label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}
// A word in the ink its probability earns: certain words read first.
ImU32 confidence(float p) {
    return p >= 0.3f ? sternbild::col::parchment : p >= 0.1f ? sternbild::col::secondary : sternbild::col::faint;
}
// The model's own next word, wherever a layer already holds it: a teal lighter than the palette's, to read
// on the blue probability shade.
constexpr ImU32 kAnswer = IM_COL32(0x7f, 0xe0, 0xd2, 255);
// The speed slider in log10 layer steps a second: from 30 s a layer to everything at once.
constexpr float kSlowest = -1.48f, kMaxSpeed = 3.0f;
constexpr const char* kKinds[] = {"Ablate J-space", "Remove words", "Swap two words", "Inject word"};
constexpr const char* kKindIds[] = {"ablate", "remove", "swap", "inject"};
} // namespace

bool decode_base64(const std::string& text, std::vector<uint8_t>& out) {
    static const auto table = [] {
        std::array<int8_t, 256> t{}; t.fill(-1);
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) t[static_cast<uint8_t>(alphabet[i])] = static_cast<int8_t>(i);
        return t;
    }();
    if (text.size() % 4) return false;
    out.clear(); out.reserve(text.size() / 4 * 3);
    for (size_t i = 0; i < text.size(); i += 4) {
        uint32_t n = 0; int padding = 0;
        for (size_t k = 0; k < 4; ++k) {
            const char c = text[i + k];
            if (c == '=' && i + 4 == text.size() && k >= 2) { ++padding; n <<= 6; continue; }
            const int v = table[static_cast<uint8_t>(c)];
            if (v < 0 || padding) return false;
            n = n << 6 | static_cast<uint32_t>(v);
        }
        out.push_back(static_cast<uint8_t>(n >> 16));
        if (padding < 2) out.push_back(static_cast<uint8_t>(n >> 8));
        if (padding < 1) out.push_back(static_cast<uint8_t>(n));
    }
    return true;
}

float TraceRun::weight(int position, int layer, int key, int head) const {
    if (position < 0 || position >= positions() || layer < 0 || layer >= static_cast<int>(attention_layers.size()) ||
        key < 0 || key >= keys[position] || heads <= 0) return 0;
    const uint8_t* row = attention.data() + attention_at[position] +
                         static_cast<size_t>(layer) * heads * keys[position];
    if (head >= 0) return row[static_cast<size_t>(std::min(head, heads - 1)) * keys[position] + key] / 255.0f;
    float sum = 0;
    for (int h = 0; h < heads; ++h) sum += row[static_cast<size_t>(h) * keys[position] + key];
    return sum / (255.0f * heads);
}

bool read_trace_event(const Json& event, TraceRun& run, std::map<int32_t, std::string>& words, std::string& error) {
    const auto fail = [&](const char* why) { error = why; return false; };
    if (!event.is_object()) return fail("trace event is not an object");
    const int first = integer(field(event, "first"), -1), count = integer(field(event, "count"), -1);
    const int k = integer(field(event, "k"), -1), prompt = integer(field(event, "prompt"), -1);
    const Json& layer_values = field(event, "layers");
    if (first != run.positions() || count < 1 || count > 4096 || k < 1 || k > 50 || prompt < 0 ||
        !layer_values.is_array() || layer_values.get<Array>().empty() || layer_values.get<Array>().size() > 256)
        return fail("trace event does not continue the run");
    std::vector<int> layers;
    for (const Json& v : layer_values.get<Array>()) {
        const int layer = integer(v, -1);
        if (layer < 0 || (!layers.empty() && layer <= layers.back())) return fail("trace layers are not increasing");
        layers.push_back(layer);
    }
    if (run.positions() > 0 && (layers != run.layers || k != run.k || prompt != run.prompt))
        return fail("trace event changes the run's shape");
    const size_t cells = static_cast<size_t>(count) * layers.size() * k, outputs = static_cast<size_t>(count) * k;
    std::vector<int32_t> tokens, jlens, logit, output;
    std::vector<float> jprob, lprob, oprob;
    if (!ints(field(event, "tokens"), count, tokens) || !ints(field(event, "jlens"), cells, jlens) ||
        !ints(field(event, "logit"), cells, logit) || !ints(field(event, "output"), outputs, output) ||
        !probabilities(field(event, "jprob"), cells, jprob) || !probabilities(field(event, "lprob"), cells, lprob) ||
        !probabilities(field(event, "oprob"), outputs, oprob))
        return fail("trace event arrays do not match its shape");
    // Attention: count x layers x heads x keys bytes, or keys 0 when none were sent.
    std::vector<int> attention_layers;
    int heads = 0, keys = 0;
    std::vector<uint8_t> attention;
    if (field(event, "attention").is_string()) {
        const Json& listed = field(event, "attention_layers");
        heads = integer(field(event, "heads"), -1);
        keys = integer(field(event, "keys"), -1);
        if (!listed.is_array() || listed.get<Array>().size() > 64 || heads < 1 || heads > 128 || keys < 0 || keys > 65536)
            return fail("trace attention is malformed");
        for (const Json& v : listed.get<Array>()) attention_layers.push_back(integer(v, -1));
        if (!decode_base64(field(event, "attention").get<std::string>(), attention) ||
            attention.size() != static_cast<size_t>(count) * attention_layers.size() * heads * keys)
            return fail("trace attention does not match its shape");
        if (run.positions() > 0 && !run.attention_layers.empty() &&
            (attention_layers != run.attention_layers || heads != run.heads))
            return fail("trace attention changes the run's shape");
    }
    // The trajectory plane: count x layers x 2 little-endian float32, in every event of a run or in none.
    std::vector<float> plane;
    if (field(event, "plane").is_string()) {
        std::vector<uint8_t> bytes;
        if (!decode_base64(field(event, "plane").get<std::string>(), bytes) ||
            bytes.size() != static_cast<size_t>(count) * layers.size() * 2 * sizeof(float))
            return fail("trace plane does not match its shape");
        plane.resize(bytes.size() / sizeof(float));
        std::memcpy(plane.data(), bytes.data(), bytes.size());
        for (float v : plane)
            if (!std::isfinite(v)) return fail("trace plane is not finite");
    }
    if (run.positions() > 0 && plane.empty() != run.plane.empty()) return fail("trace plane comes and goes");
    const Json& names = field(event, "words");
    std::map<int32_t, std::string> added;
    if (names.is_object()) {
        for (const auto& [key, value] : names.get<Object>()) {
            if (key.empty() || key.size() > 9 || key.find_first_not_of("0123456789") != std::string::npos ||
                !value.is_string() || value.get<std::string>().size() > 256)
                return fail("trace words are malformed");
            added[std::stoi(key)] = value.get<std::string>();
        }
    }
    if (run.positions() == 0) { run.layers = layers; run.k = k; run.prompt = prompt; }
    const auto append = [](auto& to, const auto& from) { to.insert(to.end(), from.begin(), from.end()); };
    append(run.tokens, tokens); append(run.jlens, jlens); append(run.logit, logit); append(run.output, output);
    append(run.jprob, jprob); append(run.lprob, lprob); append(run.oprob, oprob); append(run.plane, plane);
    for (auto& [id, text] : added) words[id] = std::move(text);
    if (!attention_layers.empty()) { run.attention_layers = attention_layers; run.heads = heads; }
    const size_t per_position = attention_layers.size() * heads * keys;
    for (int i = 0; i < count; ++i) {
        run.keys.push_back(keys);
        run.attention_at.push_back(run.attention.size());
        if (per_position) run.attention.insert(run.attention.end(), attention.begin() + i * per_position,
                                               attention.begin() + (i + 1) * per_position);
    }
    return true;
}

std::string display_word(const std::string& word) {
    std::string out;
    for (char c : word) {
        if (c == '\n') out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    const auto start = out.find_first_not_of(' ');
    if (start == std::string::npos) return out.empty() ? out : "\xC2\xB7";
    return out.substr(start);
}

Object TraceIntervention::fields() const {
    Object f{{"kind", kind}, {"first", static_cast<double>(first)}, {"last", static_cast<double>(last)}};
    if (kind == "ablate") f["k"] = static_cast<double>(k);
    if (kind == "inject") f["strength"] = strength;
    if (!tokens.empty()) {
        Array ids;
        for (int32_t t : tokens) ids.push_back(static_cast<double>(t));
        f["tokens"] = std::move(ids);
    }
    return f;
}

std::string TraceIntervention::label() const {
    char band[48];
    std::snprintf(band, sizeof band, "  layers %d to %d", first, last);
    if (kind == "ablate") return "Ablate J-space, top " + std::to_string(k) + band;
    if (kind == "remove") return "Remove " + words + band;
    if (kind == "swap") return "Swap " + words + band;
    char strength_text[32];
    std::snprintf(strength_text, sizeof strength_text, " x%g", strength);
    return "Inject " + words + strength_text + band;
}

void TracePane::started(int run) {
    runs_[run] = {};
    runs_[run].running = true;
}

bool TracePane::event(int run, const Json& event, std::string& error) {
    return read_trace_event(event, runs_[run], words_, error);
}

void TracePane::finished(int run, const Json& result) {
    runs_[run].running = false;
    runs_[run].done = true;
    const Json& finish = field(result, "finish_reason");
    runs_[run].finish = finish.is_string() ? finish.get<std::string>() : std::string();
}

void TracePane::failed(int run, const std::string& error) {
    runs_[run].running = false;
    runs_[run].error = error;
    second_pending_ = false;
}

bool TracePane::tokenized(const Json& result, std::string& error) {
    tokenizing_ = false;
    std::vector<int32_t> ids;
    const Json& values = field(result, "ids");
    if (!values.is_array() || !ints(values, values.get<Array>().size(), ids) || ids.empty()) {
        error = "Invalid tokens from script"; return false;
    }
    const Json& pieces = field(result, "words");
    std::string shown;
    if (pieces.is_array())
        for (const Json& piece : pieces.get<Array>())
            if (piece.is_string()) shown += (shown.empty() ? "" : " + ") + display_word(piece.get<std::string>());
    TraceIntervention change = pending_;
    if (change.kind == "swap" && ids.size() != 2) { error_ = "Swap needs two words of one token each: " + shown; return true; }
    if (change.kind == "inject") ids.resize(1);
    change.tokens = ids;
    change.words = shown;
    interventions_.push_back(change);
    error_.clear();
    return true;
}

std::string TracePane::text(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += word(id);
    return out;
}

const std::string& TracePane::word(int32_t id) const {
    static const std::string unknown = "?";
    const auto it = words_.find(id);
    return it == words_.end() ? unknown : it->second;
}

double TracePane::total_steps(const TraceRun& r) const {
    return r.positions() == 0 ? 0 : static_cast<double>(r.layers.size() + 1) * (1 + r.generated());
}

bool playback_shows(double steps, int layers, int prompt, int position, int layer) {
    const double per_token = layers + 1.0, index = layer < 0 ? layers : layer;
    const double before = position < prompt ? 0 : per_token * (1 + position - prompt);
    return std::floor(steps) >= before + index + 1;
}

int playback_positions(double steps, int layers, int prompt, int positions) {
    const double per_token = layers + 1.0, n = std::floor(steps);
    if (n <= per_token) return std::min(prompt, positions);
    return static_cast<int>(std::min<double>(positions, prompt + std::ceil((n - per_token) / per_token)));
}

bool TracePane::revealed(const TraceRun& r, int position, int layer) const {
    return &r != &display() || playback_shows(revealed_, static_cast<int>(r.layers.size()), r.prompt, position, layer);
}

int TracePane::revealed_positions(const TraceRun& r) const {
    return &r != &display() ? r.positions()
        : playback_positions(revealed_, static_cast<int>(r.layers.size()), r.prompt, r.positions());
}

std::string TracePane::revealed_answer(int run) const {
    const TraceRun& r = runs_[run];
    const int end = &r == &display() ? revealed_positions(r) : r.positions();
    if (end <= r.prompt) return {};
    return text(std::vector<int32_t>(r.tokens.begin() + r.prompt, r.tokens.begin() + end));
}

void TracePane::playback() {
    const TraceRun& r = display();
    const int index = &r == &runs_[0] ? 0 : 1;
    if (index != played_) {
        // Another run on show: one the user switched to is shown whole; one that began streaming (the
        // intervened run after the intact one) plays from its start.
        revealed_ = picked_ ? 1e18 : 0;
        played_ = index;
    }
    picked_ = false;
    if (speed_ >= kMaxSpeed) revealed_ = 1e18;
    else if (!paused_) revealed_ += ImGui::GetIO().DeltaTime * std::pow(10.0, static_cast<double>(speed_));
    // Never ahead of the computation: at the edge, the view is live.
    revealed_ = std::min(revealed_, total_steps(r));
}

TracePane::Ask TracePane::generate(const Object& settings) {
    settings_ = settings;
    runs_[0] = {}; runs_[1] = {};
    revealed_ = speed_ >= kMaxSpeed ? 1e18 : 0;
    const bool intervened = !interventions_.empty();
    second_pending_ = intervened && compare_;
    shown_ = intervened ? 1 : 0;
    // The first run to stream plays first: the intact one when comparing.
    played_ = intervened && compare_ ? 0 : shown_;
    cursor_position_ = cursor_layer_ = -1;
    follow_ = true;
    return trace_request(intervened && !compare_ ? 1 : 0);
}

// Both runs share the settings and one seed, so the intervened run samples like the intact one until their
// distributions part.
TracePane::Ask TracePane::trace_request(int run) const {
    Ask ask;
    ask.op = "trace"; ask.run = run;
    Array changes;
    if (run == 1) for (const auto& change : interventions_) changes.push_back(change.fields());
    ask.fields = settings_;
    ask.fields["k"] = 10.0;
    ask.fields["seed"] = 0.0;
    ask.fields["attention"] = true;
    ask.fields["interventions"] = std::move(changes);
    return ask;
}

TracePane::Ask TracePane::draw(bool can_send, bool tracing) {
    playback();
    Ask ask = controls(can_send, tracing);
    intervention_editor(ask, can_send);
    answers();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float side = std::clamp(ImGui::GetContentRegionAvail().x * 0.3f, 220.0f, 460.0f);
    const float grid_width = std::max(120.0f, ImGui::GetContentRegionAvail().x - side - spacing);
    tape(grid_width);
    const ImVec2 available = ImGui::GetContentRegionAvail();
    grid(grid_width, std::max(80.0f, available.y));
    ImGui::SameLine();
    side_list(side, std::max(80.0f, available.y), ask, can_send);
    // The intervened run follows the intact one, prompt for prompt.
    if (ask.op.empty() && second_pending_ && can_send && runs_[0].done) {
        ask = trace_request(1);
        second_pending_ = false;
    }
    return ask;
}

TracePane::Ask TracePane::controls(bool, bool tracing) {
    Ask ask;
    const ImGuiStyle& style = ImGui::GetStyle();
    ImGui::BeginDisabled(!tracing);
    if (sternbild::word_button("Stop")) ask.op = "stop";
    ImGui::EndDisabled();
    // The clock: one slider from 30 seconds a layer, through layers and tokens a second, to full speed.
    ImGui::SameLine(0, style.ItemSpacing.x * 3);
    label("Speed");
    ImGui::SameLine();
    const double steps = static_cast<double>((display().layers.empty() ? 23 : display().layers.size()) + 1);
    const double rate = std::pow(10.0, static_cast<double>(speed_));
    char speed[48];
    if (speed_ >= kMaxSpeed) std::snprintf(speed, sizeof speed, "Full speed");
    else if (rate >= steps) std::snprintf(speed, sizeof speed, "%.3g tokens/s", rate / steps);
    else if (rate >= 1) std::snprintf(speed, sizeof speed, "%.3g layers/s", rate);
    else std::snprintf(speed, sizeof speed, "%.3g s per layer", 1 / rate);
    ImGui::SetNextItemWidth(ImGui::CalcTextSize("00.0 s per layer").x * 1.6f);
    ImGui::SliderFloat("##trace.speed", &speed_, kSlowest, kMaxSpeed, speed, ImGuiSliderFlags_NoInput);
    ImGui::SameLine();
    if (sternbild::word_button(paused_ ? "Play" : "Pause")) paused_ = !paused_;
    ImGui::SameLine();
    ImGui::BeginDisabled(speed_ >= kMaxSpeed);
    if (sternbild::word_button("Step")) { paused_ = true; revealed_ = std::floor(revealed_) + 1; }
    ImGui::EndDisabled();
    ImGui::SameLine(0, style.ItemSpacing.x * 3);

    // Which lens and which run the grid shows.
    label("Lens");
    ImGui::SameLine();
    for (int i = 0; i < 2; ++i) {
        ImGui::PushStyleColor(ImGuiCol_Text, jacobian_ == (i == 0) ? sternbild::col::bronze : sternbild::col::secondary);
        if (sternbild::word_button(i == 0 ? "J-lens" : "Logit lens")) jacobian_ = i == 0;
        ImGui::PopStyleColor();
        ImGui::SameLine();
    }
    {
        ImGui::SameLine(0, style.ItemSpacing.x * 3);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float h = ImGui::GetTextLineHeight(), y = at.y + ImGui::GetStyle().FramePadding.y;
        auto* legend = ImGui::GetWindowDrawList();
        for (int i = 0; i < 5; ++i)
            legend->AddRectFilled(point(at.x + i * 9, y + 2), point(at.x + i * 9 + 8, y + h - 2),
                                  sternbild::with_alpha(sternbild::col::vehicle, 18 + i * 47));
        ImGui::Dummy(ImVec2(46, 1));
        ImGui::SameLine(); faint("probability");
        ImGui::SameLine(0, style.ItemSpacing.x * 2);
        ImGui::PushStyleColor(ImGuiCol_Text, kAnswer);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("next word");
        ImGui::PopStyleColor();
        if (runs_[1].positions() > 0) {
            ImGui::SameLine(0, style.ItemSpacing.x * 2);
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::cat);
            ImGui::TextUnformatted("changed by the intervention");
            ImGui::PopStyleColor();
        }
    }
    const bool both = runs_[0].positions() > 0 && runs_[1].positions() > 0;
    if (both) {
        ImGui::SameLine(0, style.ItemSpacing.x * 4);
        label("Show");
        ImGui::SameLine();
        for (int i = 0; i < 2; ++i) {
            ImGui::PushStyleColor(ImGuiCol_Text, shown_ == i ? sternbild::col::bronze : sternbild::col::secondary);
            if (sternbild::word_button(i == 0 ? "Intact" : "Intervened")) { shown_ = i; picked_ = true; }
            ImGui::PopStyleColor();
            ImGui::SameLine();
        }
    }
    const TraceRun& r = display();
    if (r.heads > 0) {
        ImGui::SameLine(0, style.ItemSpacing.x * 4);
        label("Attention heads");
        ImGui::SameLine();
        char current[16];
        if (head_ < 0) std::snprintf(current, sizeof current, "Mean");
        else std::snprintf(current, sizeof current, "%d", head_ + 1);
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("Mean").x + ImGui::GetFrameHeight() + 2 * style.FramePadding.x);
        if (ImGui::BeginCombo("##trace.head", current)) {
            if (ImGui::Selectable("Mean", head_ < 0)) head_ = -1;
            for (int h = 0; h < r.heads; ++h) {
                char name[8];
                std::snprintf(name, sizeof name, "%d", h + 1);
                if (ImGui::Selectable(name, head_ == h)) head_ = h;
            }
            ImGui::EndCombo();
        }
        if (head_ >= r.heads) head_ = -1;
    }
    char facts[128] = "";
    if (r.positions() > 0)
        std::snprintf(facts, sizeof facts, "%zu layers   %d prompt tokens   %d generated%s", r.layers.size(), r.prompt,
                      r.generated(), r.running ? "" : r.finish == "stop" ? "   ended" : "");
    else if (r.running) std::snprintf(facts, sizeof facts, "Reading the prompt");
    const float width = ImGui::CalcTextSize(facts).x;
    const float x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - width;
    if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
    ImGui::AlignTextToFramePadding();
    faint(facts);
    return ask;
}

void TracePane::intervention_editor(Ask& ask, bool can_send) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const float number = ImGui::CalcTextSize("000").x + 2 * style.FramePadding.x;
    label("Intervention");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::CalcTextSize("Swap two words").x + ImGui::GetFrameHeight() + 2 * style.FramePadding.x);
    ImGui::Combo("##trace.kind", &kind_, kKinds, 4);
    if (kind_ != 0) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("Paris London").x * 1.6f);
        ImGui::InputTextWithHint("##trace.words", kind_ == 2 ? "Two words" : kind_ == 1 ? "Words" : "Word",
                                 words_text_, sizeof words_text_);
    }
    ImGui::SameLine(); label("Layers"); ImGui::SameLine(); ImGui::SetNextItemWidth(number);
    ImGui::InputInt("##trace.first", &first_, 0, 0);
    ImGui::SameLine(); label("to"); ImGui::SameLine(); ImGui::SetNextItemWidth(number);
    ImGui::InputInt("##trace.last", &last_, 0, 0);
    const TraceRun& r = runs_[0].positions() > 0 ? runs_[0] : runs_[1];
    const int top = r.layers.empty() ? 22 : r.layers.back();
    first_ = std::clamp(first_, 0, top); last_ = std::clamp(last_, first_, top);
    if (kind_ == 0) {
        ImGui::SameLine(); label("Top"); ImGui::SameLine(); ImGui::SetNextItemWidth(number);
        ImGui::InputInt("##trace.k", &k_, 0, 0);
        k_ = std::clamp(k_, 1, 50);
    } else if (kind_ == 3) {
        ImGui::SameLine(); label("Strength"); ImGui::SameLine(); ImGui::SetNextItemWidth(number * 1.4f);
        ImGui::InputFloat("##trace.strength", &strength_, 0, 0, "%.3g");
        if (!std::isfinite(strength_)) strength_ = 4;
        strength_ = std::clamp(strength_, -64.0f, 64.0f);
    }
    ImGui::SameLine();
    const bool needs_words = kind_ != 0;
    ImGui::BeginDisabled(tokenizing_ || (needs_words && (words_text_[0] == '\0' || !can_send)) ||
                         interventions_.size() >= 16);
    if (sternbild::word_button("Add")) {
        TraceIntervention change;
        change.kind = kKindIds[kind_]; change.first = first_; change.last = last_; change.k = k_;
        change.strength = strength_;
        if (!needs_words) interventions_.push_back(change);
        else {
            // Words mid-sentence carry a leading space, as the lens reads them.
            pending_ = change;
            tokenizing_ = true;
            ask.op = "tokenize";
            ask.fields = Object{{"text", " " + std::string(words_text_)}};
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, style.ItemSpacing.x * 3);
    ImGui::Checkbox("Compare with intact", &compare_);
    if (!error_.empty()) { ImGui::SameLine(); faint(error_.c_str()); }
    for (size_t i = 0; i < interventions_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (i) ImGui::SameLine(0, style.ItemSpacing.x * 3);
        ImGui::AlignTextToFramePadding();
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::bronze);
        ImGui::TextUnformatted(interventions_[i].label().c_str());
        ImGui::PopStyleColor();
        ImGui::SameLine();
        const bool remove = sternbild::word_button("Remove");
        ImGui::PopID();
        if (remove) { interventions_.erase(interventions_.begin() + static_cast<long>(i)); break; }
    }
}

void TracePane::answers() {
    for (int i = 0; i < 2; ++i) {
        const TraceRun& r = runs_[i];
        if (r.positions() == 0 && !r.running) continue;
        // The answer as far as the clock has reached it.
        const std::string answer = revealed_answer(i);
        label(i == 0 ? "Intact" : "Intervened");
        ImGui::SameLine();
        ImGui::PushTextWrapPos(0);
        if (!r.error.empty()) faint(r.error.c_str());
        else if (answer.empty()) faint(r.running || r.generated() > 0 ? "..." : "No tokens");
        else ImGui::TextUnformatted(answer.c_str());
        ImGui::PopTextWrapPos();
    }
}

// The whole tape at a glance: every position the clock has reached (across, squeezed to the width) by layer (up,
// the output on top), coloured like the grid, with the grid's window framed in bronze and the cursor marked.
// Click or drag to move the grid and the cursor there.
void TracePane::tape(float width) {
    const TraceRun& r = display();
    if (r.positions() == 0) return;
    const int layers = static_cast<int>(r.layers.size()), rows = layers + 1;
    const int seen = std::max(1, revealed_positions(r));
    const float row = 2, height = rows * row;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##trace.tape", ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
    auto* list = ImGui::GetWindowDrawList();
    list->AddRectFilled(origin, point(origin.x + width, origin.y + height), sternbild::col::deep);
    const auto x_of = [&](double position) { return origin.x + width * position / seen; };
    const auto& ids = jacobian_ ? r.jlens : r.logit;
    const auto& probs = jacobian_ ? r.jprob : r.lprob;
    // One column per position, or per point when positions outnumber points (the middle position stands in).
    const int columns = std::max(1, std::min(seen, static_cast<int>(width)));
    for (int c = 0; c < columns; ++c) {
        const int p = std::min(seen - 1, static_cast<int>((c + 0.5) * seen / columns));
        const float x0 = origin.x + width * c / columns, x1 = origin.x + width * (c + 1) / columns;
        const int32_t answer = r.output[static_cast<size_t>(p) * r.k];
        for (int i = 0; i < rows; ++i) {
            const int layer = i == 0 ? -1 : layers - i;
            if (!revealed(r, p, layer)) continue;
            const size_t at = layer < 0 ? static_cast<size_t>(p) * r.k : r.at(p, layer);
            const int32_t id = layer < 0 ? r.output[at] : ids[at];
            const float prob = layer < 0 ? r.oprob[at] : probs[at];
            const ImU32 fill = differs(p, layer) ? sternbild::with_alpha(sternbild::col::cat, 200)
                : id == answer ? sternbild::with_alpha(kAnswer, static_cast<int>(70 + 185 * std::clamp(prob, 0.0f, 1.0f)))
                : sternbild::with_alpha(sternbild::col::vehicle, static_cast<int>(18 + 200 * std::clamp(prob, 0.0f, 1.0f)));
            list->AddRectFilled(point(x0, origin.y + row * i), point(std::max(x1, x0 + 1), origin.y + row * (i + 1)), fill);
        }
    }
    if (r.prompt > 0 && r.prompt < seen)
        list->AddLine(point(x_of(r.prompt), origin.y), point(x_of(r.prompt), origin.y + height), sternbild::col::rule);
    const float a = static_cast<float>(x_of(visible_first_)), b = std::max(static_cast<float>(x_of(visible_last_ + 1)), a + 3);
    list->AddRect(point(a, origin.y - 1), point(b, origin.y + height + 1), sternbild::col::bronze, 0, 0, 1.5f);
    if (cursor_position_ >= 0 && cursor_position_ < seen) {
        const float x = x_of(cursor_position_ + 0.5);
        list->AddLine(point(x, origin.y), point(x, origin.y + height), sternbild::col::parchment);
    }
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const int under = std::clamp(static_cast<int>((mouse.x - origin.x) / width * seen), 0, seen - 1);
    if (hovered && !active)
        ImGui::SetTooltip("Position %d (%s) of %d\nClick or drag to move the grid here", under,
                          display_word(word(r.tokens[under])).c_str(), seen);
    if (active) {
        // Centre the grid on the position under the pointer, and put the cursor there.
        const float cell = 72, label_width = ImGui::CalcTextSize("Out").x + 14;
        const float view = std::max(1.0f, (visible_last_ - visible_first_ + 1) * cell);
        scroll_to_ = std::max(0.0f, label_width + cell * under - view * 0.5f);
        cursor_position_ = under;
        follow_ = false;
    }
}

bool TracePane::differs(int position, int layer) const {
    const TraceRun& a = runs_[0];
    const TraceRun& b = runs_[1];
    if (shown_ != 1 || position >= a.positions() || position >= b.positions() || a.tokens[position] != b.tokens[position])
        return false;
    const auto& ids_a = layer < 0 ? a.output : jacobian_ ? a.jlens : a.logit;
    const auto& ids_b = layer < 0 ? b.output : jacobian_ ? b.jlens : b.logit;
    const size_t ia = layer < 0 ? static_cast<size_t>(position) * a.k : a.at(position, layer);
    const size_t ib = layer < 0 ? static_cast<size_t>(position) * b.k : b.at(position, layer);
    return ids_a[ia] != ids_b[ib];
}

// Layers up the side (the highest on top, the model's output above them), token positions across, each cell
// its residual's top lens word. The intervened run marks cells whose word differs from the intact run's.
void TracePane::grid(float width, float height) {
    const TraceRun& r = display();
    if (r.positions() == 0) {
        ImGui::BeginChild("##trace.grid", ImVec2(width, height));
        faint(r.running ? "Reading the prompt" : "Generate in the Inference pane to read every layer through the lens");
        ImGui::EndChild();
        return;
    }
    const int layers = static_cast<int>(r.layers.size()), rows = layers + 1;
    const float row = ImGui::GetTextLineHeight() + 6, cell = 72;
    const float label_width = ImGui::CalcTextSize("Out").x + 14;
    const float token_row = ImGui::GetTextLineHeight() + 8;
    const int seen = std::max(1, revealed_positions(r));
    if (follow_) cursor_position_ = seen - 1;
    cursor_position_ = std::clamp(cursor_position_, 0, seen - 1);
    cursor_layer_ = std::clamp(cursor_layer_, -1, layers - 1);

    if (scroll_to_ >= 0) { ImGui::SetNextWindowScroll(ImVec2(scroll_to_, -1)); scroll_to_ = -1; }
    ImGui::BeginChild("##trace.grid", ImVec2(width, height), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float scroll = ImGui::GetScrollX();
    const float view = ImGui::GetWindowWidth();
    const float content = label_width + cell * seen;
    // Below the tokens, the attention of the cursor's token over the tokens before it, one row per full-attention
    // layer, aligned with the columns.
    const int attention_rows = static_cast<int>(r.attention_layers.size());
    const float attention_top = rows * row + token_row + row;
    const float total_height = attention_rows ? attention_top + attention_rows * row : rows * row + token_row;
    ImGui::Dummy(ImVec2(content, total_height));
    if (follow_ && (r.running || revealed_ < total_steps(r))) ImGui::SetScrollX(std::max(0.0f, content - view + 8));
    auto* list = ImGui::GetWindowDrawList();
    const auto& ids = jacobian_ ? r.jlens : r.logit;
    const auto& probs = jacobian_ ? r.jprob : r.lprob;
    const int first = std::max(0, static_cast<int>((scroll - label_width) / cell));
    const int last = std::min(seen - 1, static_cast<int>((scroll + view) / cell) + 1);
    visible_first_ = std::clamp(static_cast<int>(scroll / cell), 0, seen - 1);
    visible_last_ = std::clamp(static_cast<int>((scroll + view - label_width) / cell), visible_first_, seen - 1);
    ImFont* font = ImGui::GetFont();
    const float font_size = ImGui::GetFontSize();
    const auto x_of = [&](int position) { return origin.x + label_width + cell * position; };
    // Row r_ is the output for r_ == 0, else layer (layers - r_).
    const auto layer_of = [&](int row_index) { return row_index == 0 ? -1 : layers - row_index; };
    for (int p = first; p <= last; ++p) {
        const float x = x_of(p);
        if (p == r.prompt && r.prompt > 0)
            list->AddLine(point(x, origin.y), point(x, origin.y + rows * row + token_row), sternbild::col::rule);
        for (int i = 0; i < rows; ++i) {
            const int layer = layer_of(i);
            const float y = origin.y + row * i;
            if (!revealed(r, p, layer)) continue;     // the clock has not reached this layer yet
            const size_t at = layer < 0 ? static_cast<size_t>(p) * r.k : r.at(p, layer);
            const int32_t id = layer < 0 ? r.output[at] : ids[at];
            const float prob = layer < 0 ? r.oprob[at] : probs[at];
            // Shade: the word's probability. Rose: the intervened run's word differs from the intact run's.
            const ImU32 fill = differs(p, layer) ? sternbild::with_alpha(sternbild::col::cat, 150)
                : sternbild::with_alpha(sternbild::col::vehicle, static_cast<int>(18 + 190 * std::clamp(prob, 0.0f, 1.0f)));
            list->AddRectFilled(point(x + 1, y + 1), point(x + cell - 1, y + row - 1), fill);
            const std::string shown = display_word(word(id));
            const ImVec4 clip(x + 3, y, x + cell - 3, y + row);
            // Teal: the word the model outputs at this position, wherever a layer already holds it.
            const bool answer = id == r.output[static_cast<size_t>(p) * r.k];
            list->AddText(font, font_size, point(x + 4, y + 3), answer ? kAnswer : sternbild::col::parchment,
                          shown.c_str(), nullptr, 0, &clip);
        }
        // The token at this position, under its column; generated tokens in the primary ink.
        const float y = origin.y + rows * row + 3;
        const bool generated = p >= r.prompt;
        const bool split = runs_[0].positions() > p && runs_[1].positions() > p && runs_[0].tokens[p] != runs_[1].tokens[p];
        const ImVec4 clip(x + 3, y, x + cell - 3, y + token_row);
        list->AddText(font, font_size, point(x + 4, y), split ? sternbild::col::bronze : generated ? sternbild::col::parchment : sternbild::col::secondary,
                      display_word(word(r.tokens[p])).c_str(), nullptr, 0, &clip);
    }
    if (attention_rows && cursor_position_ < r.positions()) {
        const int p = cursor_position_;
        char heading[160];
        if (r.keys[p] > 0)
            std::snprintf(heading, sizeof heading, "Attention from position %d (%s), %s", p,
                          display_word(word(r.tokens[p])).c_str(), head_ < 0 ? "mean of heads" : "one head");
        else std::snprintf(heading, sizeof heading, "Attention not sent for position %d: the context is too long", p);
        list->AddText(point(origin.x + scroll + label_width, origin.y + attention_top - row + 3), sternbild::col::secondary, heading);
        for (int a = 0; a < attention_rows; ++a) {
            const float y = origin.y + attention_top + row * a;
            const int block = r.attention_layers[a];
            if (!revealed(r, p, block < static_cast<int>(r.layers.size()) ? block : -1)) continue;
            for (int key = first; key <= std::min(last, r.keys[p] - 1); ++key) {
                const float w = r.weight(p, a, key, head_);
                const float x = x_of(key);
                // Square root, so the small weights beside an attention sink still show.
                list->AddRectFilled(point(x + 1, y + 1), point(x + cell - 1, y + row - 1),
                                    sternbild::with_alpha(sternbild::col::vehicle, static_cast<int>(10 + 235 * std::sqrt(w))));
                if (w >= 0.995f || (cell >= 40 && w >= 0.005f)) {
                    char percent[16];
                    std::snprintf(percent, sizeof percent, "%.0f%%", w * 100);
                    list->AddText(point(x + 4, y + 3), w > 0.3f ? sternbild::col::parchment : sternbild::col::secondary, percent);
                }
            }
        }
    }
    // The cursor cell.
    {
        const float x = x_of(cursor_position_), y = origin.y + row * (cursor_layer_ < 0 ? 0 : layers - cursor_layer_);
        list->AddRect(point(x, y), point(x + cell, y + row), sternbild::col::bronze, 0, 0, 1.5f);
    }
    // Layer labels stay at the left edge while the grid scrolls.
    const float left = origin.x + scroll;
    list->AddRectFilled(point(left, origin.y), point(left + label_width, origin.y + rows * row + token_row), sternbild::col::ground);
    for (int i = 0; i < rows; ++i) {
        const int layer = layer_of(i);
        char name[16];
        if (layer < 0) std::snprintf(name, sizeof name, "Out");
        else std::snprintf(name, sizeof name, "%d", r.layers[layer]);
        list->AddText(point(left + 2, origin.y + row * i + 3), layer < 0 ? sternbild::col::secondary : sternbild::col::faint, name);
    }
    list->AddLine(point(left + label_width - 4, origin.y), point(left + label_width - 4, origin.y + rows * row), sternbild::col::rule);
    if (attention_rows) {
        list->AddRectFilled(point(left, origin.y + attention_top), point(left + label_width, origin.y + attention_top + attention_rows * row),
                            sternbild::col::ground);
        for (int a = 0; a < attention_rows; ++a) {
            char name[16];
            std::snprintf(name, sizeof name, "%d", r.attention_layers[a]);
            list->AddText(point(left + 2, origin.y + attention_top + row * a + 3), sternbild::col::faint, name);
        }
    }

    // Hover shows the cell's top words; a click moves the cursor; arrows move it while the grid is hovered.
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    if (ImGui::IsWindowHovered() && mouse.x > left + label_width && attention_rows &&
        mouse.y >= origin.y + attention_top && mouse.y < origin.y + attention_top + attention_rows * row) {
        const int key = static_cast<int>(std::floor((mouse.x - origin.x - label_width) / cell));
        const int a = static_cast<int>(std::floor((mouse.y - origin.y - attention_top) / row));
        const int p = cursor_position_;
        if (key >= 0 && key < r.positions() && a >= 0 && a < attention_rows && p >= 0 && key < r.keys[p]) {
            ImGui::BeginTooltip();
            ImGui::Text("Layer %d: position %d (%s) attends to position %d (%s)", r.attention_layers[a], p,
                        display_word(word(r.tokens[p])).c_str(), key, display_word(word(r.tokens[key])).c_str());
            ImGui::Text("%s %.2f%%", head_ < 0 ? "Mean over heads" : "Head", r.weight(p, a, key, head_) * 100);
            if (head_ < 0) {
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                for (int h = 0; h < r.heads; ++h) ImGui::Text("Head %d  %.2f%%", h + 1, r.weight(p, a, key, h) * 100);
                ImGui::PopStyleColor();
            }
            ImGui::EndTooltip();
        }
    } else if (ImGui::IsWindowHovered() && mouse.x > left + label_width) {
        const int p = static_cast<int>(std::floor((mouse.x - origin.x - label_width) / cell));
        const int i = static_cast<int>(std::floor((mouse.y - origin.y) / row));
        if (p >= 0 && p < r.positions() && i >= 0 && i < rows) {
            const int layer = layer_of(i);
            ImGui::BeginTooltip();
            if (layer < 0) ImGui::Text("Position %d, model output", p);
            else ImGui::Text("Position %d, layer %d, %s", p, r.layers[layer], jacobian_ ? "J-lens" : "logit lens");
            for (int rank = 0; rank < r.k; ++rank) {
                const size_t at = layer < 0 ? static_cast<size_t>(p) * r.k + rank : r.at(p, layer, rank);
                const int32_t id = layer < 0 ? r.output[at] : ids[at];
                const float prob = layer < 0 ? r.oprob[at] : probs[at];
                ImGui::PushStyleColor(ImGuiCol_Text, confidence(prob));
                ImGui::Text("%2d  %5.1f%%  %s", rank + 1, prob * 100, display_word(word(id)).c_str());
                ImGui::PopStyleColor();
            }
            if (differs(p, layer)) {
                const TraceRun& a = runs_[0];
                const size_t at = layer < 0 ? static_cast<size_t>(p) * a.k : a.at(p, layer);
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
                ImGui::Text("Intact: %s", display_word(word(layer < 0 ? a.output[at] : (jacobian_ ? a.jlens : a.logit)[at])).c_str());
                ImGui::PopStyleColor();
            }
            ImGui::EndTooltip();
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { cursor_position_ = p; cursor_layer_ = layer; follow_ = false; }
        }
    }
    if (ImGui::IsWindowHovered()) {
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) { cursor_position_ = std::max(0, cursor_position_ - 1); follow_ = false; }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) { cursor_position_ = std::min(r.positions() - 1, cursor_position_ + 1); follow_ = false; }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) cursor_layer_ = cursor_layer_ < 0 ? -1 : cursor_layer_ + 1 >= layers ? -1 : cursor_layer_ + 1;
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) cursor_layer_ = cursor_layer_ < 0 ? layers - 1 : std::max(0, cursor_layer_ - 1);
    }
    ImGui::EndChild();
}

// Every layer's top words at the cursor's position, the output first: the reference viewer's "by layer"
// column. Click a word to remove or inject its direction.
void TracePane::side_list(float width, float height, Ask& ask, bool can_send) {
    ImGui::BeginChild("##trace.side", ImVec2(width, height), ImGuiChildFlags_Borders);
    const TraceRun& r = display();
    if (r.positions() == 0 || cursor_position_ < 0 || cursor_position_ >= r.positions()) { ImGui::EndChild(); return; }
    const int p = cursor_position_;
    ImGui::Text("Position %d", p);
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::bronze);
    ImGui::TextUnformatted(display_word(word(r.tokens[p])).c_str());
    ImGui::PopStyleColor();
    const auto& ids = jacobian_ ? r.jlens : r.logit;
    const auto& probs = jacobian_ ? r.jprob : r.lprob;
    const float row_label = ImGui::CalcTextSize("Out").x + 10;
    const float start = ImGui::GetCursorPosX(), limit = start + ImGui::GetContentRegionAvail().x;
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    for (int layer = -1; layer < static_cast<int>(r.layers.size()); ++layer) {
        const int shown_layer = layer < 0 ? -1 : static_cast<int>(r.layers.size()) - 1 - layer;
        ImGui::PushID(layer);
        ImGui::PushStyleColor(ImGuiCol_Text, shown_layer < 0 ? sternbild::col::secondary : sternbild::col::faint);
        if (shown_layer < 0) ImGui::TextUnformatted("Out");
        else ImGui::Text("%d", r.layers[shown_layer]);
        ImGui::PopStyleColor();
        float x = start + row_label;
        const int32_t answer = r.output[static_cast<size_t>(p) * r.k];
        for (int rank = 0; rank < r.k; ++rank) {
            const size_t at = shown_layer < 0 ? static_cast<size_t>(p) * r.k + rank : r.at(p, shown_layer, rank);
            const int32_t id = shown_layer < 0 ? r.output[at] : ids[at];
            const float prob = shown_layer < 0 ? r.oprob[at] : probs[at];
            const std::string text_shown = display_word(word(id));
            const float w = std::min(ImGui::CalcTextSize(text_shown.c_str()).x, limit - start - row_label);
            if (rank > 0 && x + w > limit) break;
            ImGui::SameLine(x);
            x += w + gap;
            ImGui::PushID(rank);
            ImGui::PushStyleColor(ImGuiCol_Text, shown_layer == cursor_layer_ ? sternbild::col::bronze
                                                 : id == answer ? kAnswer : confidence(prob));
            if (ImGui::Selectable(text_shown.c_str(), false, 0, ImVec2(w, 0))) ImGui::OpenPopup("##trace.word");
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%.1f%%", prob * 100);
            word_menu(id, ask, can_send);
            ImGui::PopID();
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
}

void TracePane::word_menu(int32_t id, Ask&, bool) {
    if (!ImGui::BeginPopup("##trace.word")) return;
    const std::string shown = display_word(word(id));
    ImGui::TextUnformatted(shown.c_str());
    ImGui::Separator();
    char band[64];
    std::snprintf(band, sizeof band, " (layers %d to %d)", first_, last_);
    TraceIntervention change;
    change.first = first_; change.last = last_; change.tokens = {id}; change.words = shown; change.strength = strength_;
    if (ImGui::MenuItem(("Remove its direction" + std::string(band)).c_str()) && interventions_.size() < 16) {
        change.kind = "remove"; interventions_.push_back(change);
    }
    if (ImGui::MenuItem(("Inject it" + std::string(band)).c_str()) && interventions_.size() < 16) {
        change.kind = "inject"; interventions_.push_back(change);
    }
    if (ImGui::MenuItem("Add to the word field")) {
        std::string text = words_text_;
        text += (text.empty() ? "" : " ") + shown;
        std::snprintf(words_text_, sizeof words_text_, "%s", text.c_str());
    }
    ImGui::EndPopup();
}

} // namespace caliper_live
