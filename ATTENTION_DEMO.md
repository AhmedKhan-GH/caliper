# Complete Transformer Architecture Visualization

## What Does This Show?

This demo provides a **complete 3D visualization** of a transformer neural network, showing:
- Token and positional embeddings
- Multi-head self-attention mechanisms (Q, K, V projections)
- Attention weight patterns between tokens
- Feed-forward neural networks
- Layer normalization and residual connections
- The full data flow from input to output

## Running the Demo

```bash
./cmake-build-debug/caliper
```

The window opens in **Transformer Attention** mode by default.

## Understanding the Visualization

### The 3D Architecture View

The 3D viewport shows the **entire transformer architecture** from bottom to top as data flows through the network:

1. **Token Embeddings (Blue)** - Each word is converted to a vector
2. **Positional Embeddings (Orange)** - Position information is encoded
3. **Combined Embeddings (Green)** - Token + position information merged
4. **For the selected layer, you'll see:**
   - **Q/K/V Projections** (Red/Green/Blue) - The query, key, and value transformations
   - **Attention Connections** - Lines showing which tokens attend to each other
   - **Attention Output** (Purple) - The result after applying attention weights
   - **After LayerNorm1** (Yellow-green) - After first normalization + residual
   - **Feed-Forward Intermediate** (Orange, expanded) - The expanded hidden layer
   - **Feed-Forward Output** (Orange) - Projected back to model dimension
   - **After LayerNorm2** (Light green) - Final layer output
5. **Pooling Layer** (Cyan) - Aggregated representation
6. **Classifier Output (Top)** - Final class predictions

### The Attention Heatmap (Side Panel)

Each cell `(i, j)` shows how much token `i` attends to token `j`:
- **Rows** = Query tokens ("who is looking")
- **Columns** = Key tokens ("what they're looking at")
- **Brightness** = attention weight (brighter = stronger)

### Example: "the cat sat on the mat"

When you select this sentence, you might see patterns like:
- **"sat"** attends to **"cat"** (subject-verb relationship)
- **"sat"** attends to **"mat"** (verb-location relationship)
- **"the"** attends to nearby nouns ("cat", "mat")
- All tokens attend somewhat to themselves (diagonal)

### Multiple Attention Heads

The transformer has 4 attention heads, each learning different patterns:
- **Head 0** might focus on syntactic structure (grammar)
- **Head 1** might focus on semantic relationships (meaning)
- **Head 2** might focus on positional patterns (word order)
- **Head 3** might capture other contextual information

Try different heads with the slider - each learns unique patterns!

### Multiple Layers

Layer 0 learns basic patterns, Layer 1 refines them with more abstract features.

## Example Sentences

1. **"the cat sat on the mat"** - Simple subject-verb-location
2. **"a quick brown dog ran in the park"** - Multiple adjectives and complex structure
3. **"the big cat and small dog played"** - Coordination with "and"
4. **"lazy cat slept on the red mat"** - Adjective-noun patterns

## Camera Controls

- **Arrow Keys** - Move forward/backward/left/right
- **Space/Shift** - Move up/down
- **Mouse Drag** - Look around (rotate view)
- **Scroll Wheel** - Zoom in/out
- **Reset Camera Button** - Return to default view

**Tip**: Fly through the layers to see how information flows from embeddings at the bottom to predictions at the top!

## What to Look For

### In the 3D View:

1. **Layer Progression**: Notice how representations change depth as you move up through layers
2. **Attention Connections**: Lines between tokens show information flow
3. **Dimension Expansion**: The feed-forward layer expands to 2048 dimensions then projects back
4. **Activation Patterns**: Brighter spheres = higher activation values
5. **Multi-Head Diversity**: Different heads show different attention patterns

### In the Attention Heatmap:

1. **Subject-Verb**: Verbs attend to their subjects
   - In "cat sat", expect "sat" → "cat" to be bright

2. **Adjective-Noun**: Adjectives attend to the nouns they modify
   - In "big cat", expect "big" → "cat" to be bright

3. **Prepositional Phrases**: Prepositions link nouns
   - In "on the mat", expect "on" → "mat" to be bright

4. **Determiners**: "the", "a" attend to nearby nouns

5. **Coordination**: "and" attends to both coordinated elements
   - In "cat and dog", expect "and" → both "cat" and "dog"

## Technical Details

- **Model**: 2-layer transformer encoder
- **Dimensions**: 128d embeddings
- **Attention**: 4 heads, scaled dot-product
- **Vocab**: 38 common English words

## Note

This is a **randomly initialized** model (not trained), so attention patterns may not be linguistically meaningful yet. In a trained model (like BERT or GPT), you'd see much clearer linguistic patterns!

To see learned patterns, you would need to:
1. Train on a language modeling task
2. Load pre-trained weights
3. Then the attention would show actual grammatical/semantic relationships
