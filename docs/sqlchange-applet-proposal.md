# SQLChange Applet: SQL Mutation Interpretability via Real-Time Attention Analysis

A proposal for a new Caliper applet that combines the SQLChange dataset pipeline with OpenGLlama's eval-callback interpretability infrastructure to study how a language model reasons about SQL mutations — which tokens it attends to, where its prediction crystallizes, and how those patterns shift between original and mutated queries.

**Target model**: `llama3.1:8b` (32 layers, 4096 hidden dim, 32 heads / 8 KV heads via GQA)

---

## Motivation

The SQLChange pipeline (ECS 189G) generates 334 structured SQL mutation pairs — an original query paired with a specific structural change (join swap, where drop, etc.) — and classifies each along three dimensions: semantic impact, performance impact, and risk. Currently, this classification is either rule-based (deterministic heuristics) or delegates to an LLM as a black-box judge.

OpenGLlama already intercepts `l_out-N`, `attn_out-N`, and `kq_soft_max-N` tensors during inference, rendering live attention heatmaps, logit lens, and semantic drift charts. The infrastructure for observing *how* a model reasons is built — it just needs a domain-specific frontend that understands SQL structure.

The SQLChange applet bridges these: feed a mutation classification prompt into llama.cpp, capture the full internal state, and render it with SQL-aware annotations that show which query tokens the model attends to when deciding "this mutation is high-risk" versus "this is safe."

---

## Architecture: Fork, Don't Rewrite

The applet forks `applets/opengllama/` into `applets/sqlchange/`. This is a fork, not a wrapper — the eval callback, threading model, and texture pipeline are copied and extended, not called through an indirection layer.

### What stays the same

| Component | Source | Reuse rationale |
|-----------|--------|-----------------|
| `eval_callback` | `opengllama.cpp:59-195` | Tensor interception logic is model-agnostic. The same `l_out-N`, `attn_out-N`, `kq_soft_max-N` filtering works for any llama.cpp model on any prompt. |
| `LayerActivation` struct | `opengllama.h:14-26` | Hidden state capture format. No SQL-specific changes needed. |
| `TokenAttn` struct | `opengllama.h:119-121` | Per-layer attention vector storage. Unchanged. |
| `TokenLogitInfo` struct | `opengllama.h:28-33` | Token confidence and top-k capture. Unchanged. |
| `run_inference_async` | `opengllama.cpp:1469-1690` | Inference loop with pause/step/speed control, sampler chain, per-token activation snapshot. The playback controls are essential for step-through analysis. |
| `load_model_async` | `opengllama.cpp:1341-1413` | Ollama model discovery and async GGUF loading with progress bar. |
| `update_*_texture` methods | `opengllama.cpp:1111-1301` | Heatmap pixel generation and GL texture upload. Color ramps stay. |
| `DrawTextVertical` | `opengllama.cpp:21-53` | Vertical token label rendering for heatmap axes. |

### What changes

| Component | Change | Why |
|-----------|--------|-----|
| **Prompt construction** | New | Builds the classification prompt from a SQLChange JSON record instead of free-text input |
| **Token annotation layer** | New | Maps tokenizer output positions back to SQL structural categories (keyword, table name, column, operator, literal, mutation site) |
| **Record browser panel** | New | Loads `sqlchange_dataset*.json`, shows record list with filters by mutation type / complexity / domain |
| **Dual-run differential mode** | New | Runs original and mutated SQL through the same prompt template sequentially, stores both attention histories, renders a difference heatmap |
| **SQL syntax-highlighted display** | New | Replaces the plain-text output box with a color-coded SQL view using ImGuiColorTextEdit (already a submodule in `third_party/`) |
| **draw_inference_view** | Modified | Restructured into SQL-specific panels (see UI Layout below) |

---

## Adapting for llama3.1:8b

An 8B model with 32 layers is not GPT-4. The prompt design and visualization interpretation must account for its limitations.

### Prompt strategy: minimal, structured, direct

The full SQLChange labeling prompt from `reasoning_pipeline.py:_build_llm_prompt()` sends the complete record including ER graph, join keys, where details, and the prior rule-based result. That prompt can exceed 1500 tokens — too much context noise for an 8B model to attend to meaningfully.

Instead, the applet constructs a **focused prompt** that isolates the classification task:

```
You are analyzing a SQL query mutation. The original query was modified by
applying a "{mutation_type}" operation.

Schema:
{compact_schema}

Original SQL:
{original_sql}

Modified SQL:
{modified_sql}

Classify this mutation:
- Semantic impact (equivalent, narrower, broader, different):
- Performance impact (improves, degrades, neutral):
- Risk level (low, medium, high):

Respond with one line per dimension. Be concise.
```

**Why this works for 8B:**

1. **~300-600 tokens** depending on query complexity. Leaves headroom in the 2048-token context window for generation and keeps heatmaps readable.
2. **No ER graph or rule hints** in the prompt. The ER graph is displayed in the UI as reference, but excluded from the prompt. This is intentional: we want to see whether the model can identify structural relationships from the SQL alone — that is the research question.
3. **Schema is compacted** to table names and column names only — no type annotations. Reduces token count without losing the information the model needs for relationship inference.
4. **"Be concise"** steers 8B away from verbose reasoning that dilutes attention signal. We want the model to commit to labels quickly so the logit lens shows a clean crystallization profile.

### What 8B can and cannot show

**Expect to see clearly:**
- Attention concentration on SQL keywords at the mutation site (`LEFT JOIN` vs `INNER JOIN`, presence/absence of `WHERE` clause)
- Layer stratification: early layers attending to syntax, later layers to semantics (this pattern holds even in small models)
- Decision crystallization differences between "easy" mutations (limit_add — always narrower) and "hard" mutations (join_swap — requires reasoning about data flow)

**Expect noise in:**
- Fine-grained schema reasoning. An 8B model may not reliably infer implicit foreign key relationships from column names. That is a finding, not a bug.
- Multi-step reasoning about cross-table risk. The model may attend to table names without connecting them through join paths. Compare this against the ER graph metadata to see where the model's understanding breaks down.

**Specific 8B considerations in the code:**
- `context_size_` set to `2048` (sufficient for compact prompts, keeps attention maps manageable)
- `n_gpu_layers_` set to `99` (8B fits entirely in Apple Silicon unified memory or a modest GPU)
- `temperature_` set to `0.0` — greedy decoding, since we want deterministic output for fair comparison across mutations. As discussed, temperature does not affect attention patterns during the forward pass.
- `max_tokens_` set to `64` — the classification response is three short lines. Limiting generation keeps the attention timeline focused.

---

## Data Flow

```
                   sqlchange_dataset_gptoss.json
                              |
                              v
                    ┌─────────────────────┐
                    │   Record Browser     │
                    │  (filter by mutation  │
                    │   type, complexity,   │
                    │   domain)             │
                    └─────────┬───────────┘
                              |
                    user selects a record
                              |
                              v
                    ┌─────────────────────┐
                    │  Prompt Builder      │
                    │  (compact schema +   │
                    │   original SQL +     │
                    │   modified SQL)      │
                    └─────────┬───────────┘
                              |
                    ┌─────────┴───────────┐
                    |                     |
                    v                     v
           ┌──────────────┐     ┌──────────────┐
           │ Run A:        │     │ Run B:        │
           │ "original"    │     │ "mutated"     │
           │ prompt with   │     │ prompt with   │
           │ original SQL  │     │ modified SQL  │
           │ in both slots │     │ in both slots │
           └──────┬───────┘     └──────┬───────┘
                  |                     |
                  v                     v
           attn_history_A[]       attn_history_B[]
           context_map_A[][]      context_map_B[][]
           activations_A[]        activations_B[]
           token_logits_A[]       token_logits_B[]
                  |                     |
                  └─────────┬───────────┘
                            |
                            v
                  ┌─────────────────────┐
                  │   Differential       │
                  │   Visualization      │
                  │   (A vs B heatmaps,  │
                  │    attention delta,   │
                  │    drift comparison)  │
                  └─────────────────────┘
```

**Run A** is the control: the prompt shows the original SQL in both the "Original" and "Modified" slots. The model should classify this as equivalent/neutral/low — a no-op mutation. This establishes a baseline attention pattern.

**Run B** is the experiment: the prompt shows the actual original and modified SQL. The attention difference between Run A and Run B reveals exactly where the model's reasoning changes in response to the mutation.

This A/B design isolates the mutation signal from the prompt-template signal. Without it, you cannot distinguish "the model attends to the WHERE clause because it's important" from "the model attends to the WHERE clause because it's always near the end of the prompt."

---

## Token Annotation Layer

OpenGLlama renders token labels as opaque strings from `llama_token_to_piece`. The SQLChange applet adds a classification step after tokenization that maps each token position to a structural category:

```cpp
enum class SqlTokenRole {
    Keyword,       // SELECT, FROM, JOIN, WHERE, GROUP BY, LIMIT, ON, AND, OR
    TableName,     // Identifiers matching a table name in the schema context
    ColumnName,    // Identifiers matching a column name in the schema context
    Operator,      // =, <, >, !=, IS, IN, BETWEEN, LIKE
    Literal,       // String literals, numbers
    MutationSite,  // Tokens that differ between original_sql and modified_sql
    Punctuation,   // (, ), ., *, ;
    PromptFrame,   // "You are analyzing...", "Classify this mutation:"
    Other
};
```

**How classification works:**

1. After `llama_tokenize`, iterate over `context_tokens_` (the string pieces)
2. The prompt is constructed from known template fragments — the boundary indices between "prompt frame", "schema", "original SQL", and "modified SQL" sections are known at construction time
3. Within SQL sections, match each token piece against:
   - A static set of SQL keywords (case-insensitive)
   - The table names from `record["context"]` keys
   - The column names from `record["context"][table]["columns"]` values
4. For mutation site detection: diff the `original_sql` and `modified_sql` token sequences (LCS-based) and mark tokens present in one but not the other

**Rendering:**

Each `SqlTokenRole` maps to a color. In the context attention heatmap, token labels on the X-axis are colored by role instead of uniform gray. In the SQL display panel, tokens are syntax-highlighted with the same color scheme. The mutation site tokens get a contrasting background (e.g., orange underline).

This means when you hover over a bright spot in the attention heatmap, the tooltip shows not just "Token 47: `retailers`" but "Token 47: `retailers` [TableName, in original SQL section]" — immediately interpretable.

---

## UI Layout

The applet replaces OpenGLlama's single-column scrolling layout with a structured panel arrangement. All rendering uses ImGui docking (already available in the Caliper ImGui build).

```
┌─────────────────────────────────────────────────────────────────┐
│ Model: llama3.1:8b  | 32 layers | 2048 ctx | [Unload]          │
├────────────────────────┬────────────────────────────────────────┤
│                        │                                        │
│   RECORD BROWSER       │   SQL COMPARISON                       │
│                        │                                        │
│   [Filter: mutation]   │   ┌─ Original ──────────────────────┐  │
│   [Filter: complexity] │   │ SELECT retailers.name            │  │
│   [Filter: domain]     │   │ FROM retailers                   │  │
│                        │   │ LEFT JOIN retailer_products ...   │  │
│   ┌────────────────┐   │   └──────────────────────────────────┘  │
│   │ #0  join_swap  │   │   ┌─ Modified ──────────────────────┐  │
│   │ #1  where_drop │◄──│   │ SELECT retailers.name            │  │
│   │ #2  limit_add  │   │   │ FROM retailers                   │  │
│   │ #3  join_drop  │   │   │ INNER JOIN retailer_products ... │  │
│   │ ...            │   │   └──────────────────────────────────┘  │
│   └────────────────┘   │                                        │
│                        │   Mutation: join_swap                   │
│   Record metadata:     │   Domain: retail                       │
│   complexity: multi    │   Rule labels: semantic=different       │
│   tables: 3            │               performance=improves     │
│   cross_table: true    │               risk=high                │
│   graph_depth: 3       │                                        │
│                        │   [Run Classification]  [Run A/B Diff] │
├────────────────────────┴────────────────────────────────────────┤
│                                                                 │
│   ATTENTION & ACTIVATION PANELS (same as OpenGLlama, below)     │
│                                                                 │
│   ┌─ Token Confidence ────────────────────────────────────────┐ │
│   │  [bar chart — green/red per generated token]              │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│   ┌─ Context Attention (SQL-annotated) ───────────────────────┐ │
│   │  [heatmap: layers × tokens, X-axis colored by SqlRole]    │ │
│   │  mutation-site tokens marked with orange indicator         │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│   ┌─ Decision Crystallization ────────────────────────────────┐ │
│   │  [logit lens bar chart: cosine-to-final per layer]        │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│   ┌─ Semantic Drift ─────────────────────────────────────────┐  │
│   │  [bar chart: cosine between adjacent layers]              │  │
│   └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─ Attention Difference (A/B mode only) ────────────────────┐ │
│   │  [heatmap: attn_B[layer][tok] - attn_A[layer][tok]]       │ │
│   │  blue = less attention than control, red = more            │ │
│   │  mutation-site tokens highlighted on X-axis               │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│   ┌─ Embedding Flow (per-layer heatmaps + arrows) ───────────┐ │
│   │  [identical to OpenGLlama]                                │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Differential Attention Heatmap

This is the primary new visualization. After completing both Run A and Run B:

```cpp
// Both runs produce: attn_history_[token_idx].layer_attn[layer][kv_pos]
// The differential is computed per output token, per layer, per KV position:

std::vector<std::vector<float>> attn_diff;  // [n_layers][n_kv]

for (int l = 0; l < n_layers; ++l) {
    attn_diff[l].resize(n_kv, 0.0f);
    for (int k = 0; k < n_kv; ++k) {
        float a = run_a_attn.layer_attn[l][k];  // control (no mutation)
        float b = run_b_attn.layer_attn[l][k];  // experiment (with mutation)
        attn_diff[l][k] = b - a;                 // positive = model attends MORE here
    }
}
```

**Color mapping:**
- Diverging blue-white-red ramp
- Blue: model attends *less* to this token when the mutation is present
- White: no change
- Red: model attends *more* to this token when the mutation is present

**Interpretation for SQL mutations:**

For a `join_swap` (LEFT JOIN -> INNER JOIN):
- Expect red at the JOIN keyword position and the table name after it
- Expect blue at the WHERE clause if the model redistributes attention
- If the model shows *no* differential at the mutation site, it is not detecting the structural change — a capability limitation of 8B visible in the data

For a `where_drop` (removed filter condition):
- Expect blue at the position where the WHERE clause used to be (absent in modified SQL, but present in the "original SQL" section of the prompt)
- Expect red at the remaining query structure as the model compensates

---

## New Data Structures

Beyond the inherited OpenGLlama structs, the applet adds:

```cpp
struct SqlChangeRecord {
    int unique_id;
    int source_id;
    std::string domain;
    std::string complexity;
    std::string mutation_type;
    std::string original_sql;
    std::string modified_sql;
    std::string context_json;        // raw schema JSON string

    // Pre-parsed from context for token annotation
    std::vector<std::string> table_names;
    std::vector<std::string> column_names;

    // Rule-based labels (from reasoning_pipeline.py)
    std::string rule_semantic;
    std::string rule_performance;
    std::string rule_risk;
};

struct TokenAnnotation {
    int position;                    // index in context_tokens_
    SqlTokenRole role;
    std::string section;             // "prompt_frame", "schema", "original_sql", "modified_sql"
    bool is_mutation_site;           // true if this token differs between original and modified
};

struct DualRunState {
    // Run A (control)
    std::vector<TokenAttn> attn_history_a;
    std::vector<std::vector<float>> context_map_a;
    std::vector<LayerActivation> activations_a;
    std::vector<TokenLogitInfo> logits_a;
    std::string output_a;

    // Run B (experiment)
    std::vector<TokenAttn> attn_history_b;
    std::vector<std::vector<float>> context_map_b;
    std::vector<LayerActivation> activations_b;
    std::vector<TokenLogitInfo> logits_b;
    std::string output_b;
};
```

---

## File Structure

```
applets/sqlchange/
├── CMakeLists.txt
├── plugin.cpp                  # ABI exports (boilerplate from APPLETS.md)
├── sqlchange_applet.h          # Main applet class
├── sqlchange_applet.cpp        # UI, inference orchestration, differential logic
├── record_store.h              # JSON loading and filtering for SqlChangeRecord
├── record_store.cpp
├── token_annotator.h           # SqlTokenRole classification after tokenization
├── token_annotator.cpp
└── sql_diff.h                  # LCS-based token diff for mutation site detection
```

**Dependencies** (link in CMakeLists.txt):
- `caliper_applet_sdk` (ImGui, ImPlot, ImPlot3D, ImGuiFileDialog)
- `llama` (llama.cpp — inference, tokenizer, eval callback)
- `imgui_color_text_edit` (syntax-highlighted SQL display)

No libtorch dependency. No DuckDB dependency. The applet reads JSON directly with a lightweight parser (nlohmann/json header-only, or hand-rolled since the schema is fixed).

---

## Implementation Plan

### Phase 1: Scaffold and Single-Run Mode

Fork OpenGLlama into `applets/sqlchange/`. Strip the free-text prompt input. Add the record browser panel (load JSON, display filterable list). Construct the compact classification prompt from a selected record. Run inference and display the standard OpenGLlama visualizations (attention, logit lens, drift) with no SQL-specific annotation yet.

**Deliverable**: Select a SQLChange record, click Run, see the same visualizations as OpenGLlama but with the SQL classification prompt.

### Phase 2: Token Annotation

Implement `TokenAnnotator`. After tokenization, classify each token position. Modify the context attention heatmap X-axis labels to use role-based colors. Modify tooltips to show role and section. Highlight mutation-site tokens with an orange marker bar below the heatmap.

**Deliverable**: Hovering over the attention heatmap shows "Token 23: `JOIN` [Keyword, in modified SQL section]" and mutation-site tokens are visually distinct.

### Phase 3: Dual-Run Differential

Implement `DualRunState`. Add "Run A/B Diff" button that queues two sequential inference runs (control then experiment). After both complete, compute the per-layer, per-position attention difference. Render the diverging blue-white-red differential heatmap. Add a side-by-side logit lens comparison (two bar charts stacked vertically, one per run).

**Deliverable**: Click "Run A/B Diff" on a join_swap record. See a red hotspot at the JOIN keyword position in the differential heatmap. See whether the logit lens crystallizes at a different layer for the mutation case.

### Phase 4: Batch Comparison and Export

Add the ability to queue multiple records (e.g., all 23 join_swap mutations) and collect summary statistics: average attention weight on mutation-site tokens by layer, average crystallization layer by mutation type. Export captured data as JSON for offline analysis (e.g., in the SQLChange notebook).

**Deliverable**: A summary table showing "join_swap mutations: average crystallization at layer 22, mean mutation-site attention 0.34" vs "limit_add mutations: average crystallization at layer 14, mean mutation-site attention 0.08."

---

## Research Questions This Enables

Ordered by what llama3.1:8b can realistically answer:

1. **Does the model attend to the mutation site?** Compare average attention at mutation-site tokens vs. non-mutation tokens across all 334 records. If the model does not attend to the site of change, its classification is based on shallow pattern matching.

2. **Does crystallization depth correlate with mutation difficulty?** `limit_add` is semantically simple (always "narrower"). `join_swap` requires understanding data flow. If the logit lens shows earlier crystallization for limit_add, the model is deploying different computational depth for different mutation types.

3. **Do attention patterns match the rule-based signals?** The deterministic rules in `reasoning_pipeline.py` check `cross_table_risk`, `join_keys`, and `where_details`. If the model attends to the same structural features — table names involved in joins, WHERE clause predicates — then the LLM and the rules are reasoning about the same evidence. If not, one of them is wrong.

4. **Where does 8B break down?** For queries with 3+ tables and deep join chains, does the model's attention diffuse across all tables equally (failure to trace relationships) or concentrate on the correct pair? This directly measures schema reasoning capability in a small model.

5. **Attention redistribution under mutation**: When a WHERE clause is dropped, does the model shift attention to other filtering mechanisms (HAVING, subquery predicates), or does it simply attend less overall? The differential heatmap answers this per-layer.

---

## Relevant Reading

From `docs/opengllama-reading-list.md`, directly applicable to this work:

- **Understanding Hidden Computations in CoT Reasoning** (Dec 2024) — Uses logit lens to decode hidden states. Directly applicable: the model's CoT about SQL may not reflect its actual attention patterns.
- **Circuit Tracing** (Lindsey et al., Mar 2025) — Attribution graphs in production models. The attention heatmaps are a coarser version of this; future work could integrate Anthropic's cross-layer transcoder approach.
- **Base Models Know How to Reason, Thinking Models Learn When** (Oct 2025) — If reasoning circuits are present in all models, llama3.1:8b should show SQL reasoning patterns in its attention even without explicit reasoning training on SQL.
- **Alignment Faking is a Linear Feature** (Jan 2026) — If misalignment is a single direction in activation space, analogous "mutation-type reasoning" directions may exist. The per-layer activation captures could support future linear probe experiments.

---

## Constraints and Non-Goals

- **Not a production classification tool**. The applet is a research instrument for understanding model internals, not a replacement for the rule-based labeling pipeline.
- **Not multi-model comparison** (initially). Phase 1-3 target llama3.1:8b only. Multi-model comparison (e.g., 8B vs 27B) is future work that requires no architectural changes — just loading a different GGUF.
- **Not fine-tuning**. The applet observes a frozen model. No gradient computation, no weight modification, no LoRA.
- **No libtorch dependency**. The RepNet applet uses libtorch for ECG inference. This applet uses llama.cpp only. Keeping the dependency footprint minimal means faster builds and simpler debugging.
- **No network calls**. All data is local JSON files. All inference is local via llama.cpp. No Ollama HTTP API, no cloud LLM calls.
