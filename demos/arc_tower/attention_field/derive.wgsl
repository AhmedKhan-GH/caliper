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
