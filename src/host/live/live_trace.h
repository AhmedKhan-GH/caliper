#pragma once
// The Trace pane: the model read through the Jacobian lens while it generates, after Anthropic's
// "Verbalizable Representations Form a Global Workspace in Language Models" (2026) and its slice viewer.
// A grid of layers by token positions shows each residual's top lens word; a side list shows every layer's
// top words at the cursor's position. Interventions (J-space ablation, removing, swapping or injecting word
// directions across a band of layers) run beside an intact run, and the grid marks where they differ.
#include <crude_json.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace caliper_live {

// One run of the host's trace request, filled event by event. Per position: the token, then for every
// source layer the top k lens words of the J-lens and of the logit lens, then the model's own top k.
struct TraceRun {
    std::vector<int> layers;
    int k = 0, prompt = 0;
    std::vector<int32_t> tokens, jlens, logit, output;
    std::vector<float> jprob, lprob, oprob;
    // Attention patterns of the full-attention layers, when sent: per position, layer and head, the weights
    // (0-255) over that position's keys. keys[p] is 0 where none were sent (a context too long).
    std::vector<int> attention_layers;
    int heads = 0;
    std::vector<int> keys;
    std::vector<size_t> attention_at;
    std::vector<uint8_t> attention;
    bool running = false, done = false;
    std::string finish, error;
    int positions() const { return static_cast<int>(tokens.size()); }
    int generated() const { return positions() > prompt ? positions() - prompt : 0; }
    // The top word index into jlens/logit (rank 0 is the most likely).
    size_t at(int position, int layer, int rank = 0) const {
        return (static_cast<size_t>(position) * layers.size() + static_cast<size_t>(layer)) * k + rank;
    }
    // The weight (0-1) position gives key at an attention layer: one head's, or the heads' mean for head < 0.
    float weight(int position, int layer, int key, int head = -1) const;
};

// Standard base64 (with padding) to bytes; false if the text is not base64.
bool decode_base64(const std::string& text, std::vector<uint8_t>& out);

// Appends one trace event to run and its new words to words. False, with a reason, if the event does not
// continue the run where it ended or its arrays do not match its shape.
bool read_trace_event(const crude_json::value& event, TraceRun& run, std::map<int32_t, std::string>& words,
                      std::string& error);

// A word as a grid cell shows it: line breaks and tabs escaped, leading spaces dropped, a lone space a dot.
std::string display_word(const std::string& word);

struct TraceIntervention {
    std::string kind;                 // ablate, remove, swap, inject
    std::vector<int32_t> tokens;
    std::string words;                // as typed
    int first = 0, last = 0, k = 10;
    double strength = 4;
    crude_json::object fields() const;
    std::string label() const;
};

class TracePane {
public:
    // What the pane asks the inspector to send: op is empty for nothing. run is 0 (intact) or 1 (intervened).
    struct Ask { std::string op; crude_json::object fields; int run = -1; };

    // Starts a generation with the Inference pane's prompt and sampling (prompt, max_new_tokens, temperature,
    // top_p, top_k, thinking): the intact run, the intervened run, or the intact then the intervened run.
    Ask generate(const crude_json::object& settings);
    void started(int run);
    bool event(int run, const crude_json::value& event, std::string& error);
    void finished(int run, const crude_json::value& result);
    void failed(int run, const std::string& error);
    bool tokenized(const crude_json::value& result, std::string& error);
    // Draws the pane's contents below its header. can_send: the channel is free for a request.
    Ask draw(bool can_send, bool tracing);

    const TraceRun& run(int index) const { return runs_[index]; }
    const std::vector<TraceIntervention>& interventions() const { return interventions_; }
    std::string text(const std::vector<int32_t>& ids) const;

private:
    Ask controls(bool can_send, bool tracing);
    void intervention_editor(Ask& ask, bool can_send);
    void answers();
    void grid(float width, float height);
    void side_list(float width, float height, Ask& ask, bool can_send);
    void word_menu(int32_t id, Ask& ask, bool can_send);
    Ask trace_request(int run) const;
    bool differs(int position, int layer) const;
    const std::string& word(int32_t id) const;

    TraceRun runs_[2];
    std::map<int32_t, std::string> words_;
    std::vector<TraceIntervention> interventions_;
    crude_json::object settings_;
    bool jacobian_ = true, compare_ = true, second_pending_ = false;
    int head_ = -1;
    int shown_ = 0, cursor_position_ = -1, cursor_layer_ = -1;
    bool follow_ = true;
    // The intervention being written.
    int kind_ = 0, first_ = 5, last_ = 17, k_ = 10;
    float strength_ = 4;
    char words_text_[128] = "";
    bool tokenizing_ = false;
    TraceIntervention pending_;
    std::string error_;
};

} // namespace caliper_live
