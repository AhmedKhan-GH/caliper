// Batch row 0 of the (B, H, T, T) weights of each layer, which the script publishes as
// (B * H * T, T) rows, is their first H * T * T values: layer 0's row first, then layer 1's.
@compute @workgroup_size(256)
fn copy_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = H * T * T;
    if (id.x >= n) {
        return;
    }
    pattern[id.x] = attn0[id.x];
    pattern[n + id.x] = attn1[id.x];
}

// One label for each token: its word at the left of its row, beside the grid.
@compute @workgroup_size(32)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= T) {
        return;
    }
    let cell_y = (GY1 - GY0) / f32(T);
    // At the left edge, half a row up: the anchor compensates for the label placer's offset.
    labels[id.x] = Label(clip_of(vec2<f32>(0.0, GY0 + (f32(id.x) - 0.5) * cell_y)), id.x);
}
