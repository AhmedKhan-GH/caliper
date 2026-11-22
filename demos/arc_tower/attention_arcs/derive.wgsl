// The weight of a query token i on a key token j, in batch row 0 of the (B, H, T, T) tensors,
// whose first H * T * T values are that row.
fn attn(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    let k = (head * T + i) * T + j;
    if (layer == 0u) {
        return attn0[k];
    }
    return attn1[k];
}

// The induction score of each head: its mean weight from a token of the
// second copy to the token after the same token's first occurrence.
@compute @workgroup_size(8)
fn scores(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H) {
        return;
    }
    var sum = 0.0;
    for (var t = S + P; t < S + 2u * P; t = t + 1u) {
        sum = sum + attn(id.x / H, id.x % H, t, t - P + 1u);
    }
    heads[id.x] = sum / f32(P);
}

// One arc for each weight above the threshold, from the key token to the query token.
@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H * T * T) {
        return;
    }
    let j = id.x % T;
    let i = (id.x / T) % T;
    let head = (id.x / (T * T)) % H;
    let layer = id.x / (T * T * H);
    if (j >= i) {
        return;                            // causal, and no arc to itself
    }
    let w = attn(layer, head, i, j);
    if (w < params.threshold) {
        return;
    }
    let slot = atomicAdd(&arcs_count, 1u);
    if (slot >= arrayLength(&arcs)) {
        return;
    }
    arcs[slot] = Arc(j, i, layer, head, w, heads[layer * H + head]);
}

// One label anchor for each token, just outside the lower ring.
@compute @workgroup_size(32)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= T) {
        return;
    }
    labels[id.x] = Label(vec4<f32>(ring_point(0u, id.x) * 1.15, 1.0), id.x);
}
