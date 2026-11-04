#include "live/live_trace.h"
#include "ui/sternbild_theme.h"

#include <imgui.h>

#include <algorithm>
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
constexpr const char* kKinds[] = {"Ablate J-space", "Remove words", "Swap two words", "Inject word"};
constexpr const char* kKindIds[] = {"ablate", "remove", "swap", "inject"};
} // namespace

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
    append(run.jprob, jprob); append(run.lprob, lprob); append(run.oprob, oprob);
    for (auto& [id, text] : added) words[id] = std::move(text);
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

TracePane::Ask TracePane::trace_request(int run) const {
    Ask ask;
    ask.op = "trace"; ask.run = run;
    Array changes;
    if (run == 1) for (const auto& change : interventions_) changes.push_back(change.fields());
    ask.fields = Object{{"prompt", std::string(prompt_)}, {"max_new_tokens", static_cast<double>(max_new_tokens_)},
                        {"k", 10.0}, {"temperature", 0.0}, {"seed", 0.0}, {"thinking", false},
                        {"interventions", std::move(changes)}};
    return ask;
}

TracePane::Ask TracePane::draw(bool can_send, bool tracing) {
    Ask ask = controls(can_send, tracing);
    intervention_editor(ask, can_send);
    answers();
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float side = std::clamp(available.x * 0.3f, 220.0f, 460.0f);
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    grid(std::max(120.0f, available.x - side - spacing), std::max(80.0f, available.y));
    ImGui::SameLine();
    side_list(side, std::max(80.0f, available.y), ask, can_send);
    // The intervened run follows the intact one, prompt for prompt.
    if (ask.op.empty() && second_pending_ && can_send && runs_[0].done) {
        ask = trace_request(1);
        second_pending_ = false;
    }
    return ask;
}

TracePane::Ask TracePane::controls(bool can_send, bool tracing) {
    Ask ask;
    const ImGuiStyle& style = ImGui::GetStyle();
    const float number = ImGui::CalcTextSize("0000").x + 2 * style.FramePadding.x;
    const float tail = ImGui::CalcTextSize("Max tokens").x + number + ImGui::CalcTextSize("Trace").x + 24 +
                       ImGui::CalcTextSize("Stop").x + 2 * style.FramePadding.x + 5 * style.ItemSpacing.x;
    ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x - tail));
    ImGui::InputTextWithHint("##trace.prompt", "Prompt", prompt_, sizeof prompt_);
    ImGui::SameLine(); label("Max tokens"); ImGui::SameLine(); ImGui::SetNextItemWidth(number);
    ImGui::InputInt("##trace.max", &max_new_tokens_, 0, 0);
    max_new_tokens_ = std::clamp(max_new_tokens_, 0, 256);
    ImGui::SameLine();
    ImGui::BeginDisabled(!can_send || prompt_[0] == '\0');
    if (sternbild::primary_button("Trace")) {
        runs_[0] = {}; runs_[1] = {};
        const bool intervened = !interventions_.empty();
        ask = trace_request(intervened && !compare_ ? 1 : 0);
        second_pending_ = intervened && compare_;
        shown_ = intervened ? 1 : 0;
        cursor_position_ = cursor_layer_ = -1;
        follow_ = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!tracing);
    if (sternbild::word_button("Stop")) ask.op = "stop";
    ImGui::EndDisabled();

    // Which lens and which run the grid shows.
    label("Lens");
    ImGui::SameLine();
    for (int i = 0; i < 2; ++i) {
        ImGui::PushStyleColor(ImGuiCol_Text, jacobian_ == (i == 0) ? sternbild::col::bronze : sternbild::col::secondary);
        if (sternbild::word_button(i == 0 ? "J-lens" : "Logit lens")) jacobian_ = i == 0;
        ImGui::PopStyleColor();
        ImGui::SameLine();
    }
    const bool both = runs_[0].positions() > 0 && runs_[1].positions() > 0;
    if (both) {
        ImGui::SameLine(0, style.ItemSpacing.x * 4);
        label("Show");
        ImGui::SameLine();
        for (int i = 0; i < 2; ++i) {
            ImGui::PushStyleColor(ImGuiCol_Text, shown_ == i ? sternbild::col::bronze : sternbild::col::secondary);
            if (sternbild::word_button(i == 0 ? "Intact" : "Intervened")) shown_ = i;
            ImGui::PopStyleColor();
            ImGui::SameLine();
        }
    }
    const TraceRun& r = runs_[shown_].positions() > 0 ? runs_[shown_] : runs_[1 - shown_];
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
        const std::vector<int32_t> generated(r.tokens.begin() + std::min(r.prompt, r.positions()), r.tokens.end());
        label(i == 0 ? "Intact" : "Intervened");
        ImGui::SameLine();
        ImGui::PushTextWrapPos(0);
        if (!r.error.empty()) faint(r.error.c_str());
        else if (generated.empty()) faint(r.running ? "..." : "No tokens");
        else ImGui::TextUnformatted(text(generated).c_str());
        ImGui::PopTextWrapPos();
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
    const TraceRun& r = runs_[shown_].positions() > 0 ? runs_[shown_] : runs_[1 - shown_];
    if (r.positions() == 0) {
        ImGui::BeginChild("##trace.grid", ImVec2(width, height));
        faint(r.running ? "Reading the prompt" : "Trace a prompt to read every layer through the lens");
        ImGui::EndChild();
        return;
    }
    const int layers = static_cast<int>(r.layers.size()), rows = layers + 1;
    const float row = ImGui::GetTextLineHeight() + 6, cell = 72;
    const float label_width = ImGui::CalcTextSize("Out").x + 14;
    const float token_row = ImGui::GetTextLineHeight() + 8;
    if (follow_ && r.positions() > 0) { cursor_position_ = r.positions() - 1; if (cursor_layer_ < 0) cursor_layer_ = -1; }
    cursor_position_ = std::clamp(cursor_position_, 0, r.positions() - 1);
    cursor_layer_ = std::clamp(cursor_layer_, -1, layers - 1);

    ImGui::BeginChild("##trace.grid", ImVec2(width, height), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float scroll = ImGui::GetScrollX();
    const float view = ImGui::GetWindowWidth();
    const float content = label_width + cell * r.positions();
    ImGui::Dummy(ImVec2(content, rows * row + token_row));
    if (follow_ && r.running) ImGui::SetScrollX(std::max(0.0f, content - view + 8));
    auto* list = ImGui::GetWindowDrawList();
    const auto& ids = jacobian_ ? r.jlens : r.logit;
    const auto& probs = jacobian_ ? r.jprob : r.lprob;
    const int first = std::max(0, static_cast<int>((scroll - label_width) / cell));
    const int last = std::min(r.positions() - 1, static_cast<int>((scroll + view) / cell) + 1);
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
            const size_t at = layer < 0 ? static_cast<size_t>(p) * r.k : r.at(p, layer);
            const int32_t id = layer < 0 ? r.output[at] : ids[at];
            const float prob = layer < 0 ? r.oprob[at] : probs[at];
            if (differs(p, layer))
                list->AddRectFilled(point(x + 1, y + 1), point(x + cell - 1, y + row - 1), sternbild::with_alpha(sternbild::col::cat, 90));
            const std::string shown = display_word(word(id));
            const ImVec4 clip(x + 3, y, x + cell - 3, y + row);
            list->AddText(font, font_size, point(x + 4, y + 3), confidence(prob), shown.c_str(), nullptr, 0, &clip);
        }
        // The token at this position, under its column; generated tokens in the primary ink.
        const float y = origin.y + rows * row + 3;
        const bool generated = p >= r.prompt;
        const bool split = runs_[0].positions() > p && runs_[1].positions() > p && runs_[0].tokens[p] != runs_[1].tokens[p];
        const ImVec4 clip(x + 3, y, x + cell - 3, y + token_row);
        list->AddText(font, font_size, point(x + 4, y), split ? sternbild::col::bronze : generated ? sternbild::col::parchment : sternbild::col::secondary,
                      display_word(word(r.tokens[p])).c_str(), nullptr, 0, &clip);
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

    // Hover shows the cell's top words; a click moves the cursor; arrows move it while the grid is hovered.
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    if (ImGui::IsWindowHovered() && mouse.x > left + label_width) {
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
    const TraceRun& r = runs_[shown_].positions() > 0 ? runs_[shown_] : runs_[1 - shown_];
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
    for (int layer = -1; layer < static_cast<int>(r.layers.size()); ++layer) {
        const int shown_layer = layer < 0 ? -1 : static_cast<int>(r.layers.size()) - 1 - layer;
        ImGui::PushID(layer);
        ImGui::PushStyleColor(ImGuiCol_Text, shown_layer < 0 ? sternbild::col::secondary : sternbild::col::faint);
        if (shown_layer < 0) ImGui::TextUnformatted("Out");
        else ImGui::Text("%d", r.layers[shown_layer]);
        ImGui::PopStyleColor();
        for (int rank = 0; rank < r.k; ++rank) {
            const size_t at = shown_layer < 0 ? static_cast<size_t>(p) * r.k + rank : r.at(p, shown_layer, rank);
            const int32_t id = shown_layer < 0 ? r.output[at] : ids[at];
            const float prob = shown_layer < 0 ? r.oprob[at] : probs[at];
            const std::string text_shown = display_word(word(id));
            const float w = ImGui::CalcTextSize(text_shown.c_str()).x;
            ImGui::SameLine(rank == 0 ? row_label : 0);
            if (rank > 0 && ImGui::GetContentRegionAvail().x < w + 4) break;
            ImGui::PushID(rank);
            ImGui::PushStyleColor(ImGuiCol_Text, shown_layer == cursor_layer_ ? sternbild::col::bronze : confidence(prob));
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
