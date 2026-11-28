// Derive half of image_grid@1: selects one page of samples (optionally only misclassified ones),
// formats each tile's "pred <p> (true <t>)" UTF-8 label, and copies the selected images into `atlas`.

fn append_char(buf: ptr<function, array<u32, 64>>, pos: ptr<function, u32>, ch: u32) {
    let i = *pos;
    if (i < 63u) {
        (*buf)[i] = ch;
        *pos = i + 1u;
    }
}

fn append_int(buf: ptr<function, array<u32, 64>>, pos: ptr<function, u32>, value: i32) {
    var v = value;
    if (v < 0) {
        append_char(buf, pos, 45u); // '-'
        v = -v;
    }
    var digits: array<u32, 12>;
    var count = 0u;
    var u = u32(v);
    loop {
        digits[count] = 48u + (u % 10u);
        count = count + 1u;
        u = u / 10u;
        if (u == 0u || count >= 12u) { break; }
    }
    for (var k = 0u; k < count; k = k + 1u) {
        append_char(buf, pos, digits[count - 1u - k]);
    }
}

fn format_label(pred: i32, truth: i32) -> Text {
    var b: array<u32, 64>;
    for (var i = 0u; i < 64u; i = i + 1u) {
        b[i] = 0u;
    }
    var pos = 0u;
    // "pred <p> (true <t>)"
    append_char(&b, &pos, 112u); // 'p'
    append_char(&b, &pos, 114u); // 'r'
    append_char(&b, &pos, 101u); // 'e'
    append_char(&b, &pos, 100u); // 'd'
    append_char(&b, &pos, 32u);  // ' '
    append_int(&b, &pos, pred);
    append_char(&b, &pos, 32u);  // ' '
    append_char(&b, &pos, 40u);  // '('
    append_char(&b, &pos, 116u); // 't'
    append_char(&b, &pos, 114u); // 'r'
    append_char(&b, &pos, 117u); // 'u'
    append_char(&b, &pos, 101u); // 'e'
    append_char(&b, &pos, 32u);  // ' '
    append_int(&b, &pos, truth);
    append_char(&b, &pos, 41u);  // ')'

    var words: array<u32, 16>;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let base = w * 4u;
        words[w] = (b[base] & 255u)
            | ((b[base + 1u] & 255u) << 8u)
            | ((b[base + 2u] & 255u) << 16u)
            | ((b[base + 3u] & 255u) << 24u);
    }
    return Text(
        vec4<u32>(words[0], words[1], words[2], words[3]),
        vec4<u32>(words[4], words[5], words[6], words[7]),
        vec4<u32>(words[8], words[9], words[10], words[11]),
        vec4<u32>(words[12], words[13], words[14], words[15]),
    );
}

fn empty_text() -> Text {
    let z = vec4<u32>(0u, 0u, 0u, 0u);
    return Text(z, z, z, z);
}

fn grid_dims() -> vec4<u32> {
    let img_s = shapes.images;
    let single_channel = (img_s.x == 1u || img_s.y == 1u);
    let n_img = select(0u, img_s.x * img_s.y, single_channel);
    let n_tgt = shapes.targets.w;
    let n_log = shapes.logits.z;
    let classes = shapes.logits.w;
    let n = min(n_img, min(n_tgt, n_log));
    let h = img_s.z;
    let w = img_s.w;
    if (n == 0u || classes == 0u || h == 0u || w == 0u) {
        return vec4<u32>(0u, 0u, 0u, 0u);
    }
    return vec4<u32>(n, classes, h, w);
}

@compute @workgroup_size(1)
fn select_tiles() {
    let cols = clamp(params.cols, 1u, 8u);
    let rows = clamp(params.rows, 1u, 8u);
    let per_page = min(cols * rows, 64u);
    let page = params.page;
    let only_err = params.only_errors != 0u;
    let d = grid_dims();
    let n = d.x;
    let classes = d.y;
    let img_h = d.z;
    let img_w = d.w;
    let fits_atlas = img_w > 0u && img_h > 0u && cols * img_w <= 512u && rows * img_h <= 512u;

    for (var s = 0u; s < 64u; s = s + 1u) {
        labels[s] = Label(vec4<f32>(0.0, 0.0, 0.0, 1.0), 0xffffffffu);
        strings[s] = empty_text();
        selected[s] = 0xffffffffu;
    }

    if (n == 0u || !fits_atlas) {
        atomicStore(&tiles_count, 0u);
        summary[0] = Summary(n, 0u, 0u, page);
        return;
    }

    var total_errors = 0u;
    var matched = 0u;
    var shown = 0u;
    let skip = page * per_page;

    let cell_w = 1.92 / f32(cols);
    let cell_h = 1.92 / f32(rows);
    let pad_x = cell_w * 0.05;
    let pad_y = cell_h * 0.05;
    let inv_cols = 1.0 / f32(cols);
    let inv_rows = 1.0 / f32(rows);

    for (var i = 0u; i < n; i = i + 1u) {
        let base = i * classes;
        var best = 0u;
        var best_val = logits[base];
        for (var c = 1u; c < classes; c = c + 1u) {
            let v = logits[base + c];
            if (v > best_val) {
                best_val = v;
                best = c;
            }
        }
        let tgt = targets[i];
        let pred = i32(best);
        let is_wrong = pred != tgt;
        if (is_wrong) {
            total_errors = total_errors + 1u;
        }
        if (!only_err || is_wrong) {
            if (matched >= skip && shown < per_page) {
                let col = shown % cols;
                let row = shown / cols;
                let x0 = -0.96 + f32(col) * cell_w + pad_x;
                let x1 = -0.96 + f32(col + 1u) * cell_w - pad_x;
                let y1 = 0.96 - f32(row) * cell_h - pad_y;
                let y0 = 0.96 - f32(row + 1u) * cell_h + pad_y;
                let u0 = f32(col) * inv_cols;
                let u1 = f32(col + 1u) * inv_cols;
                let v0 = f32(row) * inv_rows;
                let v1 = f32(row + 1u) * inv_rows;
                tiles[shown] = Tile(
                    vec4<f32>(x0, y0, x1, y1),
                    vec4<f32>(u0, v0, u1, v1),
                    i,
                    tgt,
                    pred,
                    select(0u, 1u, is_wrong),
                );
                selected[shown] = i;
                // Place the centered text anchor inside the tile's bottom caption band.
                let label_y = y0 + (y1 - y0) * 0.11;
                labels[shown] = Label(vec4<f32>((x0 + x1) * 0.5, label_y, 0.0, 1.0), shown);
                strings[shown] = format_label(pred, tgt);
                shown = shown + 1u;
            }
            matched = matched + 1u;
        }
    }

    atomicStore(&tiles_count, shown);
    summary[0] = Summary(n, total_errors, shown, page);
}

@compute @workgroup_size(64)
fn copy_atlas(@builtin(global_invocation_id) id: vec3<u32>) {
    let cols = clamp(params.cols, 1u, 8u);
    let rows = clamp(params.rows, 1u, 8u);
    let d = grid_dims();
    let img_h = d.z;
    let img_w = d.w;
    let shown = summary[0].shown;
    let atlas_w = cols * img_w;
    let atlas_h = rows * img_h;
    let valid = shown > 0u && img_w > 0u && img_h > 0u && atlas_w <= 512u && atlas_h <= 512u;

    if (id.x == 0u) {
        atlas_size = select(vec2<u32>(0u, 0u), vec2<u32>(atlas_w, atlas_h), valid);
    }
    if (!valid) {
        return;
    }

    let total_px = atlas_w * atlas_h;
    let stride = 64u * 64u;
    let pixels_per_img = img_w * img_h;
    for (var px = id.x; px < total_px; px = px + stride) {
        let ax = px % atlas_w;
        let ay = px / atlas_w;
        let col = ax / img_w;
        let row = ay / img_h;
        let slot = row * cols + col;
        var rgba = 0xff161618u;
        if (slot < shown) {
            let sample_idx = selected[slot];
            if (sample_idx != 0xffffffffu) {
                let lx = ax % img_w;
                let ly = ay % img_h;
                let raw = images[sample_idx * pixels_per_img + ly * img_w + lx] * params.scale;
                let clamped = clamp(select(0.0, raw, raw > 0.0), 0.0, 1.0);
                let g = min(u32(clamped * 255.0 + 0.5), 255u);
                rgba = 0xff000000u | g | (g << 8u) | (g << 16u);
            }
        }
        atlas[px] = rgba;
    }
}
