// Fixed trace shapes and valid counts gate every pass. Padding never becomes a node.
fn node_count() -> u32 {
    if (shapes.graph_counts.w != 4u || shapes.node_i.z != 512u || shapes.node_i.w != 4u ||
        shapes.node_f.z != 512u || shapes.node_f.w != 4u || shapes.edge_i.z != 8192u ||
        shapes.edge_i.w != 2u || shapes.edge_f.z != 8192u || shapes.edge_f.w != 2u) { return 0u; }
    return u32(clamp(graph_counts[0], 0, 512));
}
fn edge_count() -> u32 { return u32(clamp(graph_counts[1], 0, 8192)); }
fn token_count() -> u32 { return u32(clamp(graph_counts[2], 1, 32)); }
fn layer_count() -> u32 { return u32(clamp(graph_counts[3], 1, 512)); }
fn finite_value(v: f32) -> f32 { return select(0.0, v, abs(v) <= 3.402823e38); }

@compute @workgroup_size(64)
fn assign_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = node_count();
    if (i == 0u) { atomicStore(&nodes_count, n); }
    if (i >= n) { return; }
    let k = u32(clamp(node_i[4u*i], 0, 3));
    let t = clamp(node_i[4u*i+2u], 0, i32(token_count())-1);
    var l = clamp(node_i[4u*i+1u], 0, i32(layer_count())-1);
    if (k == 0u) { l = -1; }
    if (k == 3u) { l = i32(layer_count()); }
    let width = min(0.2, 1.8 / f32(token_count()));
    let x = select(0.0, -0.88 + 1.76*f32(t)/f32(max(token_count()-1u,1u)), token_count()>1u);
    let center = vec2<f32>(x, -0.82 + 1.64*f32(l+1)/f32(layer_count()+1u));
    let influence = finite_value(node_f[4u*i+1u]);
    cells[i] = Cell(center, width, influence, u32(l+1)*32u+u32(t));
    nodes[i] = Node(vec4<f32>(center, 0.0, 0.0), i, k, l, t, node_i[4u*i+3u], influence);
}

// Each row owns its full text element. An absent text tensor uses a numeric label.
fn copy_text(row: u32, number: u32, prefix: u32) {
    var words: array<u32, 16>;
    if (shapes.label_text.z == 544u && shapes.label_text.w == 16u) {
        for (var w = 0u; w < 16u; w++) { words[w] = label_text[row*16u+w]; }
    } else {
        words[0] = prefix;
        var value = number;
        var digits: array<u32, 10>;
        var count = 0u;
        loop {
            digits[count] = 48u + value % 10u;
            count++;
            value = value / 10u;
            if (value == 0u || count == 10u) { break; }
        }
        for (var j = 0u; j < count; j++) {
            let b = j+4u;
            words[b/4u] = words[b/4u] | (digits[count-j-1u] << (8u*(b%4u)));
        }
    }
    strings[row] = bounded_text(words);
}

@compute @workgroup_size(64)
fn order_influence(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = node_count();
    if (i < 32u) {
        copy_text(i, i, 0x206b6f54u); // Tok
        let x = select(0.0, -0.88 + 1.76*f32(i)/f32(max(token_count()-1u,1u)), token_count()>1u);
        labels[i] = Label(vec4<f32>(x, -0.88, 0.0, 1.0), select(0xffffffffu, i, i<token_count() && n>0u), 1.0);
    }
    if (i == 0u) {
        labels[32] = Label(vec4<f32>(-0.4,0.9,0.0,1.0), 0xffffffffu, 3.0);
        labels[33] = Label(vec4<f32>(0.4,0.9,0.0,1.0), 0xffffffffu, 4.0);
    }
    if (i >= n) { return; }
    let c = cells[i];
    var rank = 0u;
    var count = 0u;
    for (var j = 0u; j < n; j++) {
        let other = cells[j];
        if (other.key == c.key) {
            count++;
            if (other.influence > c.influence || (other.influence == c.influence && j < i)) { rank++; }
        }
    }
    let spacing = 0.8*c.width/f32(count);
    let x = c.center.x - 0.4*c.width + (f32(rank)+0.5)*spacing;
    // Fixed lanes preserve the influence order and leave a strict gap after all relaxations.
    offset_a[i] = vec4<f32>(x, x, spacing, min(0.25*spacing, 0.24*1.64/f32(layer_count()+1u)));
    var prefix = 0x20727446u; // Feature prefix.
    if (nodes[i].kind == 0u) { prefix = 0x206b6f54u; }
    if (nodes[i].kind == 1u) { prefix = 0x20727446u; } // Feature
    if (nodes[i].kind == 2u) { prefix = 0x20727245u; } // Err
    if (nodes[i].kind == 3u) { prefix = 0x20676f4cu; } // Log
    copy_text(32u+i, u32(max(nodes[i].feature, 0)), prefix);
}

fn relaxed(i: u32, use_b: bool) -> vec4<f32> {
    var own: vec4<f32>;
    if (use_b) { own = offset_b[i]; } else { own = offset_a[i]; }
    var total = 0.0;
    var count = 0.0;
    let n = node_count();
    for (var e = 0u; e < edge_count(); e++) {
        let s = edge_i[2u*e];
        let t = edge_i[2u*e+1u];
        if (s < 0 || t < 0 || s >= i32(n) || t >= i32(n)) { continue; }
        var neighbor = -1;
        if (s == i32(i)) { neighbor = t; }
        if (t == i32(i)) { neighbor = s; }
        if (neighbor < 0) { continue; }
        var x: f32;
        if (use_b) { x = offset_b[u32(neighbor)].x; } else { x = offset_a[u32(neighbor)].x; }
        total += x;
        count += 1.0;
    }
    if (count > 0.0) { own.x = clamp(mix(own.x, total/count, 0.5), own.y-0.15*own.z, own.y+0.15*own.z); }
    return own;
}
@compute @workgroup_size(64)
fn relax_one(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x < node_count()) { offset_b[id.x] = relaxed(id.x, false); } }
@compute @workgroup_size(64)
fn relax_two(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x < node_count()) { offset_a[id.x] = relaxed(id.x, true); } }
@compute @workgroup_size(64)
fn relax_three(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x < node_count()) { offset_b[id.x] = relaxed(id.x, false); } }
@compute @workgroup_size(64)
fn relax_four(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= node_count()) { return; }
    let result = relaxed(i, true);
    offset_a[i] = result;
    nodes[i].position = vec4<f32>(result.x, cells[i].center.y, 0.0, result.w);
}

@compute @workgroup_size(64)
fn edge_curves(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = node_count();
    let count = select(0u, edge_count(), n>0u);
    if (i == 0u) { atomicStore(&curves_count, count); }
    if (i >= count) { return; }
    let s = edge_i[2u*i];
    let t = edge_i[2u*i+1u];
    if (s<0 || t<0 || s>=i32(n) || t>=i32(n)) { curves[i] = Curve(vec4<f32>(0.0),vec4<f32>(0.0),0.0,0.0,0u,0u); return; }
    let a = nodes[u32(s)].position.xy;
    let b = nodes[u32(t)].position.xy;
    let mid = 0.5*(a.y+b.y);
    let weight = finite_value(edge_f[2u*i]);
    let normalized = clamp(finite_value(edge_f[2u*i+1u]), -1.0, 1.0);
    curves[i] = Curve(vec4<f32>(a, a.x, mid), vec4<f32>(b.x, mid, b), weight, normalized, u32(s), u32(t));
}
