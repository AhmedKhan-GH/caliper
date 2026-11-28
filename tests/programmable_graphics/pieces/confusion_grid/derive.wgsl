// Compute a 10x10 row-normalized confusion matrix directly from live probe logits and labels.
@compute @workgroup_size(10)
fn compute_confusion(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= CLASSES * CLASSES) {
        return;
    }
    let row = idx / CLASSES;
    let col = idx % CLASSES;
    let n_log = shapes.logits.z;
    let c_log = shapes.logits.w;
    let n_tgt = shapes.targets.w;
    let n = min(n_log, n_tgt);
    if (n == 0u || c_log < CLASSES) {
        matrix[idx] = 0.0;
        return;
    }
    var row_total = 0u;
    var cell = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        if (targets[i] == i32(row)) {
            row_total = row_total + 1u;
            let base = i * c_log;
            var best = 0u;
            var best_val = logits[base];
            for (var c = 1u; c < CLASSES; c = c + 1u) {
                let v = logits[base + c];
                if (v > best_val) {
                    best_val = v;
                    best = c;
                }
            }
            if (best == col) {
                cell = cell + 1u;
            }
        }
    }
    matrix[idx] = select(0.0, f32(cell) / f32(row_total), row_total > 0u);
}

// Place one label for each true-digit row along the left margin.
@compute @workgroup_size(10)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= CLASSES) {
        return;
    }
    let cell_y = (GY1 - GY0) / f32(CLASSES);
    labels[row] = Label(clip_of(vec2<f32>(0.01, GY0 + (f32(row) + 0.5) * cell_y)), row);
}
