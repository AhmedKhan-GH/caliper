// Compute per-digit probe accuracy, error count, and mean true-class probability on GPU.
@compute @workgroup_size(10)
fn compute_metrics(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = id.x;
    if (d >= CLASSES) {
        return;
    }
    let n_log = shapes.logits.z;
    let c_log = shapes.logits.w;
    let n_tgt = shapes.targets.w;
    let n = min(n_log, n_tgt);
    if (n == 0u || c_log < CLASSES) {
        metrics[d] = DigitMetric(d, 0u, 0u, 0u, 0u, 0.0, 0.0);
        return;
    }
    var total = 0u;
    var correct = 0u;
    var conf_sum = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        if (targets[i] == i32(d)) {
            total = total + 1u;
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
            if (best == d) {
                correct = correct + 1u;
            }
            conf_sum = conf_sum + exp(clamp(logits[base + d], -20.0, 0.0));
        }
    }
    let errors = total - correct;
    let has_errors = select(0u, 1u, errors > 0u);
    let acc = select(0.0, f32(correct) / f32(total), total > 0u);
    let mean_conf = select(0.0, conf_sum / f32(total), total > 0u);
    metrics[d] = DigitMetric(d, total, correct, errors, has_errors, acc, mean_conf);
}

// Place one label anchor just outside each digit column on the ring.
@compute @workgroup_size(10)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = id.x;
    if (d >= CLASSES) {
        return;
    }
    labels[d] = Label(vec4<f32>(ring_pos(d, RING_R * 1.28, BASE_Y), 1.0), d);
}
