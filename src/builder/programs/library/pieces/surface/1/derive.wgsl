// The derive half of surface@1: the operator surface over `height` (R, C). When R times C is above
// max_cells, it pools blocks of f by f cells with the smallest f for which the pooled grid fits,
// skipping NaN values; a block of NaN values gives NaN. One element of `cells` for each pooled
// cell, in row order: x and y are the mean row and column indices of the block, z the pooled
// height, value the pooled `values` when that tensor has the shape of `height` (else the height),
// row and column the indices that x and y round to, step -1 and cols the pooled columns. The
// arithmetic is that of the CPU reference (caliper_builder/programs/operator_kernels.h).

fn is_nan(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// ceil(a / b) for b above 0, without the sum that can wrap.
fn ceil_div(a: u32, b: u32) -> u32 {
    return a / b + select(0u, 1u, a % b != 0u);
}

// Whether ceil(rows / f) * ceil(cols / f) is above m, without the product that can wrap.
fn too_many(rows: u32, cols: u32, f: u32, m: u32) -> bool {
    return ceil_div(rows, f) > m / ceil_div(cols, f);
}

// The smallest pool factor for which the pooled grid holds at most m cells. The start is a lower
// bound, as in the reference: no f below sqrt(rows * cols / m) fits.
fn pool_factor(rows: u32, cols: u32, m: u32) -> u32 {
    if (!too_many(rows, cols, 1u, m)) {
        return 1u;
    }
    var f = max(1u, u32(floor(sqrt(f32(rows) * f32(cols) / f32(m)))));
    let most = max(rows, cols);
    loop {
        if (f >= most || !too_many(rows, cols, f, m)) {
            break;
        }
        f = f + 1u;
    }
    return f;
}

// The pool of the block [r0, r1) x [c0, c1) of a row-major tensor with `cols` columns, visiting
// rows then columns as the reference does.
fn pool_height(r0: u32, r1: u32, c0: u32, c1: u32, cols: u32) -> f32 {
    var acc = 0.0;
    var n = 0u;
    for (var r = r0; r < r1; r = r + 1u) {
        for (var c = c0; c < c1; c = c + 1u) {
            let v = height[r * cols + c];
            if (is_nan(v)) {
                continue;
            }
            if (n == 0u) {
                acc = v;
            } else if (params.pool == 1u) {
                acc = max(acc, v);
            } else if (params.pool == 2u) {
                acc = min(acc, v);
            } else {
                acc = acc + v;
            }
            n = n + 1u;
        }
    }
    // A var, not a constant: a constant NaN is an error of the shader.
    var nan_bits = 0x7fc00000u;
    var h = bitcast<f32>(nan_bits);
    if (n > 0u) {
        h = acc;
        if (params.pool != 1u && params.pool != 2u) {
            h = acc / f32(n);
        }
    }
    return h;
}

fn pool_value(r0: u32, r1: u32, c0: u32, c1: u32, cols: u32) -> f32 {
    var acc = 0.0;
    var n = 0u;
    for (var r = r0; r < r1; r = r + 1u) {
        for (var c = c0; c < c1; c = c + 1u) {
            let v = values[r * cols + c];
            if (is_nan(v)) {
                continue;
            }
            if (n == 0u) {
                acc = v;
            } else if (params.pool == 1u) {
                acc = max(acc, v);
            } else if (params.pool == 2u) {
                acc = min(acc, v);
            } else {
                acc = acc + v;
            }
            n = n + 1u;
        }
    }
    var nan_bits = 0x7fc00000u;
    var h = bitcast<f32>(nan_bits);
    if (n > 0u) {
        h = acc;
        if (params.pool != 1u && params.pool != 2u) {
            h = acc / f32(n);
        }
    }
    return h;
}

@compute @workgroup_size(256)
fn pool_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let rows = shapes.height.z;
    let cols = shapes.height.w;
    // An empty tensor, or one whose elements a u32 cannot count, gives no cells.
    var ok = rows > 0u && cols > 0u && rows <= 0xffffffffu / cols;
    if (ok) {
        ok = rows * cols <= arrayLength(&height);
    }
    let m = max(1u, min(params.max_cells, arrayLength(&cells)));
    var f = 1u;
    var out_rows = 0u;
    var out_cols = 0u;
    if (ok) {
        f = pool_factor(rows, cols, m);
        out_rows = ceil_div(rows, f);
        out_cols = ceil_div(cols, f);
    }
    var n = 0u;
    if (ok && !too_many(rows, cols, f, m)) {
        n = out_rows * out_cols;
    }
    if (id.x == 0u) {
        atomicStore(&cells_count, n);
    }
    if (id.x >= n) {
        return;
    }
    let i = id.x / out_cols;
    let j = id.x % out_cols;
    let r0 = i * f;
    let c0 = j * f;
    let r1 = min(r0 + f, rows);
    let c1 = min(c0 + f, cols);
    let z = pool_height(r0, r1, c0, c1, cols);
    var value = z;
    if (shapes.values.z == rows && shapes.values.w == cols && rows * cols <= arrayLength(&values)) {
        value = pool_value(r0, r1, c0, c1, cols);
    }
    // The mean index of a block, (lo + hi - 1) / 2, and the index that it rounds to, half up.
    let x = f32(r0 + r1 - 1u) * 0.5;
    let y = f32(c0 + c1 - 1u) * 0.5;
    cells[id.x] = Cell(x, y, z, value, i32((r0 + r1) / 2u), i32((c0 + c1) / 2u), -1, out_cols);
}
