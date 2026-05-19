# OpenGllama: Real-Time LLM Interpretability & Alignment Research

OpenGllama is a real-time inference visualization applet built on [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Caliper](../README.md). It intercepts the computation graph during inference to expose the internal mechanics of large language models — attention patterns, activation dynamics, and prediction formation — as they happen, token by token.

This document covers the tool's current capabilities, the research questions they address, and the roadmap for future interpretability features.

---

## Architecture

OpenGllama hooks into llama.cpp's `ggml_backend_sched_eval_callback` mechanism. During each forward pass, the callback is invoked for every tensor in the computation graph. By filtering on tensor names (`l_out-N`, `attn_out-N`, `kq_soft_max-N`), the applet captures:

| Tensor | Shape | What it contains |
|--------|-------|------------------|
| `l_out-{layer}` | `[d_model, n_tokens]` | Residual stream output after layer N |
| `attn_out-{layer}` | `[d_model, n_tokens]` | Attention block output before residual add |
| `kq_soft_max-{layer}` | `[n_kv, n_tokens, n_heads]` | Softmax attention weights (QK^T post-softmax) |

Flash attention is explicitly disabled (`LLAMA_FLASH_ATTN_TYPE_DISABLED`) so that `kq_soft_max` tensors are materialized as discrete nodes rather than fused into an opaque kernel. This trades some inference speed for full observability.

Inference runs on a background thread with pause, step, and speed control. The UI thread snapshots captured data under a mutex and renders it each frame.

---

## Current Visualizations

### 1. Token Confidence (ImPlot Bar Chart)

For each generated token, the applet computes:

- **Probability**: softmax probability of the sampled token
- **Entropy**: Shannon entropy of the full output distribution (in bits)
- **Top-K alternatives**: the 5 highest-probability tokens and their probabilities

Each bar is colored by confidence: green (high probability, model is certain) through yellow to red (high entropy, model is uncertain).

**Research applications:**

- **Hallucination detection**: Sequences of low-confidence, high-entropy tokens often correlate with hallucinated content. A sudden entropy spike mid-generation may indicate the model has moved out of distribution.
- **Calibration analysis**: Compare reported probabilities against empirical accuracy across many generations. Well-calibrated models should hallucinate proportionally to their uncertainty.
- **Sampling regime diagnosis**: Observe how temperature, top-p, and top-k interact with the raw distribution. High-temperature sampling from a peaked distribution behaves differently than from a flat one.

### 2. Context Attention Map (Texture Heatmap)

A 2D heatmap of shape `[n_layers × n_context]` showing the attention pattern of the **most recently generated token**. Each cell represents how much attention a given layer pays to a given context position, averaged across all attention heads.

Values are normalized per-layer (each row's maximum maps to full brightness) so that attention patterns remain visible regardless of how peaked or spread the distribution is. The BOS token (position 0) is excluded as it consistently acts as an attention sink that dominates the color scale.

**Research applications:**

- **Attention sink identification**: Some tokens (punctuation, BOS, repeated tokens) consistently attract disproportionate attention across layers. This visualization makes sinks immediately visible as bright vertical columns.
- **Induction head detection**: Induction heads implement copying behavior — they attend to tokens that followed a similar token earlier in context. Look for diagonal attention stripes where the model attends to positions whose local context matches the current generation context.
- **Layer specialization**: Different layers attend to different aspects of the input. Early layers often attend locally (adjacent tokens), middle layers attend to syntactic structure, and late layers attend to semantically relevant content. The heatmap makes this stratification visible.
- **Prompt injection & adversarial inputs**: Adversarial suffixes or injected instructions may produce anomalous attention patterns — layers that normally attend broadly might concentrate on the injected region.

### 3. Logit Lens — Decision Crystallization

For each layer, the applet stores the full hidden state of the last token and computes its cosine similarity to the final layer's hidden state. This is a proxy for the "logit lens" technique: projecting intermediate hidden states through the output head to see what each layer would predict if it were the last layer.

Since llama.cpp does not expose the output projection matrix (`model.output`) through its public API, cosine similarity with the final hidden state serves as an approximation. A cosine value near 1.0 means that layer's representation has already converged to the final prediction.

The visualization is a bar chart across layers, colored from blue (divergent, still exploring) through cyan/green to yellow (converged, prediction locked in).

**Research applications:**

- **Early exit feasibility**: If the logit lens shows convergence at layer 16 of 32, the remaining layers are not changing the prediction. This directly measures the potential for early-exit inference optimizations.
- **Difficulty estimation**: Easy predictions (common next words, predictable syntax) converge early. Hard predictions (factual recall, reasoning steps, ambiguous continuations) converge late. The convergence layer is a token-level difficulty metric.
- **Deceptive alignment signals**: In alignment research, a model that "changes its mind" in later layers — where the logit lens shows one prediction in early/middle layers but switches in the final layers — may be exhibiting mesa-optimization or deceptive behavior.
- **Knowledge localization**: Factual knowledge tends to be retrieved at specific layers. By comparing logit lens profiles across factual vs. non-factual prompts, you can identify which layers are responsible for knowledge retrieval.

### 4. Semantic Drift (Cosine Similarity Between Adjacent Layers)

Each layer's output is compared to the previous layer's output via cosine similarity. High drift (low cosine) means that layer is making a large transformation to the representation. Low drift (high cosine) means the representation passes through nearly unchanged.

Displayed as a bar chart: red indicates high drift (significant semantic transformation), green indicates low drift (representation preserved).

**Research applications:**

- **Computation localization**: Not all layers contribute equally. Some layers are "active" (high drift) while others are nearly residual pass-throughs. This pattern varies by input and reveals which layers are doing the heavy lifting for a given prompt.
- **Emotion and sentiment processing**: Layers with high semantic drift on emotionally charged inputs but low drift on neutral inputs are candidates for affect processing. This helps localize where the model builds emotional/sentiment representations.
- **Layer pruning candidates**: Layers that consistently show near-zero drift across diverse inputs may be candidates for removal or distillation without significant capability loss.

### 5. Per-Layer Activation Heatmaps

Raw activation values from `l_out` and `attn_out` tensors, rendered as color-mapped heatmap tiles. Each tile shows a slice of the hidden state (up to 256 dimensions) for the current token.

Connected by flow arrows to visualize the residual stream as data flows through the network.

**Research applications:**

- **Dead neuron detection**: Dimensions that are consistently near-zero across layers and tokens may indicate dead or underutilized capacity.
- **Activation magnitude monitoring**: Unusually large activations can indicate numerical instability, adversarial inputs, or outlier features. The norm bar provides an at-a-glance summary.

### 6. Activation Norm Profile

A bar chart showing the RMS norm of each layer's output. Provides a compact summary of activation magnitude across the network.

**Research applications:**

- **Norm growth patterns**: Healthy transformers typically show a characteristic norm profile — gradual increase through layers with possible plateaus. Deviations may indicate training instabilities or architectural issues.
- **Input anomaly detection**: Out-of-distribution inputs often produce anomalous norm profiles compared to typical text.

---

## Playback Controls for Research

OpenGllama provides fine-grained control over inference execution:

| Control | Function |
|---------|----------|
| **Pause / Resume** | Freeze inference to examine the current state in detail |
| **Step** | Advance exactly one token — observe how each token changes the activation landscape |
| **Token Delay** | Slow inference to a human-observable rate (0–2000ms per token) |
| **Seed** | Fix the random seed for reproducible generation |

Step-through mode is particularly valuable for mechanistic interpretability. By advancing one token at a time and examining the full visualization state, you can trace how specific predictions form:

1. Pause inference after a prompt
2. Step one token
3. Examine which context positions the attention map highlights
4. Check whether the logit lens had already converged before this layer
5. Note the semantic drift profile — which layers did the work
6. Step again and observe how the pattern shifts

---

## Sampling Hyperparameters

All inference hyperparameters are exposed and adjustable between runs:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Max Tokens | 16–2048 | 256 | Generation length |
| Temperature | 0.0–2.0 | 0.8 | Distribution sharpness (0 = greedy) |
| Top-K | 1–200 | 40 | Restrict to top K candidates |
| Top-P | 0.0–1.0 | 0.95 | Nucleus sampling threshold |
| Min-P | 0.0–0.5 | 0.05 | Minimum probability cutoff |
| Repeat Penalty | 1.0–2.0 | 1.1 | Penalize recent token repetition |
| Repeat Window | 0–256 | 64 | How far back the repeat penalty looks |
| Seed | 0–9999 | 0 (random) | Deterministic generation when non-zero |

These enable controlled experiments: fix the seed and vary temperature to observe how sampling strategy affects which tokens are chosen and how the model's internal state changes in response.

---

## Model Compatibility

OpenGllama loads GGUF models via llama.cpp. Any architecture supported by llama.cpp works, including:

- **Llama 2/3** (all sizes)
- **Qwen 2/2.5** (recommended: `qwen2.5:32b` via Ollama — 20GB, fully supported)
- **Mistral / Mixtral**
- **Phi-2/3**
- **Gemma**

Models are loaded from Ollama's local blob storage or from GGUF files on disk. GPU offloading via Metal (macOS) or CUDA is supported, with configurable layer count.

Note: Some architectures may not be supported by the version of llama.cpp bundled with Caliper. If a model fails to load, the UI displays the error. Common issues include unsupported architecture names and rope dimension mismatches.

---

## Roadmap: Future Interpretability Features

### Per-Head Attention Decomposition

The current attention map averages across all heads. Individual attention heads implement distinct circuits:

- **Induction heads**: Copy patterns from earlier in context
- **Inhibition heads**: Suppress certain token predictions
- **Positional heads**: Attend based on relative position rather than content
- **Rare token heads**: Activate selectively for uncommon vocabulary items

Exposing per-head attention (selectable via dropdown or grid view) would allow researchers to identify and study these specialized circuits directly.

### Activation Patching / Causal Tracing

Activation patching replaces a layer's output with a baseline (e.g., from a corrupted prompt) and measures the effect on downstream predictions. This reveals causal structure: which layers and positions are *necessary* for a given prediction, not just correlated with it.

Implementation: run inference twice (clean and corrupted), cache hidden states, then re-run with selective layer substitutions. The eval callback already captures the required hidden states.

### Sparse Autoencoder (SAE) Integration

Raw hidden states are high-dimensional and entangled. Sparse autoencoders decompose them into interpretable, monosemantic features — individual dimensions that correspond to human-understandable concepts (e.g., "code syntax," "French language," "deception").

Integration would involve loading pre-trained SAE weights and projecting captured hidden states through the encoder in real-time, displaying active features and their magnitudes alongside the raw activations.

### Comparative / Differential Mode

Run two prompts through the same model and display their visualizations side-by-side with a diff overlay. This directly answers questions like:

- "What changes in the model's internals when I add 'please be honest' to the system prompt?"
- "Which layers behave differently when the model refuses vs. complies?"
- "Where does the representation of a true statement diverge from a false one?"

### Residual Stream Decomposition

The residual stream at any layer is the sum of all previous layers' contributions plus the embedding. Decomposing it into per-layer contributions (attention output + FFN output per layer) would show which layers are *adding* vs. *modifying* the representation.

### Probe Training

Train lightweight linear probes on captured hidden states to detect specific properties (sentiment, truthfulness, language, topic). Run probes in real-time during inference to annotate each layer with detected features. This bridges the gap between raw activations and human-interpretable concepts without requiring pre-trained SAEs.

### Token-Level Attribution

Gradient-based or attention-based attribution showing which input tokens most influenced the current prediction. Displayed as a highlight overlay on the prompt text, providing an intuitive "why did the model say this" explanation.

---

## Research Workflows

### Workflow 1: Hallucination Forensics

1. Load a model known to hallucinate on specific queries
2. Enter a factual question with a known answer
3. Run inference and observe the token confidence chart
4. Identify the token where confidence drops / entropy spikes
5. Pause at that token and examine the attention map
6. Check: is the model attending to relevant context, or has attention drifted to irrelevant positions?
7. Check the logit lens: did the correct answer ever appear in early layers before being overwritten?

### Workflow 2: Refusal Mechanism Analysis

1. Prepare two prompts: one the model will answer, one it will refuse
2. Run both with the same seed and temperature
3. Compare the logit lens convergence profiles — at which layer does refusal crystallize?
4. Compare attention maps — do refused prompts show distinctive attention patterns (e.g., strong attention to safety-trained token positions)?
5. Compare semantic drift — are specific layers consistently responsible for the refusal decision?

### Workflow 3: Induction Head Identification

1. Construct a prompt with repeated patterns: "A B C D ... A B C"
2. Step through inference, pausing at the token after "C" (should predict "D")
3. Examine the attention map layer by layer
4. Look for layers where attention strongly targets the first occurrence of "C" (or the token after it)
5. These are candidate induction heads — they copy the pattern from earlier context

### Workflow 4: Knowledge Retrieval Depth

1. Prepare prompts requiring different types of knowledge:
   - Syntactic: "The cat sat on the ___" (shallow, early convergence expected)
   - Factual: "The capital of Mongolia is ___" (deep retrieval, late convergence expected)
   - Reasoning: "If all A are B and all B are C, then all A are ___" (multi-step, very late convergence)
2. Run each and record the logit lens convergence layer
3. Map knowledge types to network depth — this reveals the model's computational architecture for different cognitive tasks

---

## Technical Notes

### Performance Impact

Intercepting tensors via the eval callback adds overhead proportional to the amount of data copied from GPU to CPU. The primary costs are:

- `kq_soft_max`: `n_kv × n_heads` floats per layer per token (attention weights)
- `l_out`: `d_model` floats per layer per token (hidden states for logit lens)
- `attn_out`: visualization slice only (256 floats per layer)

For a 32-layer model with 4096-dim hidden states and 32 heads on a 2048-token context, this is approximately 8MB per generated token. On Apple Silicon with unified memory, the copy cost is minimal. On discrete GPUs with PCIe transfers, overhead is higher.

Flash attention is disabled to expose `kq_soft_max`, which prevents the use of optimized fused attention kernels. This typically results in 10-30% slower inference depending on the model and backend.

### Thread Safety

The eval callback runs on the inference thread during `llama_decode()`. Captured data is written to pending buffers (`pending_activations_`, `context_attention_`) without locking. After each decode call, the inference thread copies pending data to the shared state under `output_mutex_`. The UI thread only reads shared state under the same mutex, ensuring consistency.

### Numerical Considerations

- Cosine similarity can be unreliable when comparing vectors with very different magnitudes. The logit lens proxy (cosine to final layer) is most meaningful for `l_out` tensors, which have been through layer normalization.
- Attention weights are already normalized (sum to 1 per head per query position) via softmax. The per-layer normalization in the heatmap rescales to visual range but does not change the relative distribution within a layer.
- Entropy is computed in bits (log base 2). For a vocabulary of 128K tokens, maximum entropy is ~17 bits. Typical natural language entropy is 2-6 bits per token.
