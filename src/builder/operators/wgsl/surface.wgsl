// The operator surface (design/instrument-interface.md, Surface Operator):
// pools a row-major (rows, cols) tensor into a height field of out_rows by
// out_cols cells, blocks of factor by factor cells, skipping NaN values; a
// block of NaN values gives NaN. One invocation computes one cell, visiting
// rows then columns as the CPU reference does
// (caliper_builder/programs/operator_kernels.h).
struct Params { rows: u32, cols: u32, factor: u32, out_rows: u32, out_cols: u32, pool: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// A NaN read from its bits, which a compiler's fast math cannot fold away.
fn is_nan(v: f32) -> bool {
  let b = bitcast<u32>(v);
  return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
  let cell = id.x;
  if (cell >= params.out_rows * params.out_cols) { return; }
  let i = cell / params.out_cols;
  let j = cell % params.out_cols;
  let r1 = min((i + 1u) * params.factor, params.rows);
  let c1 = min((j + 1u) * params.factor, params.cols);
  var acc = 0.0;
  var n = 0u;
  for (var r = i * params.factor; r < r1; r = r + 1u) {
    for (var c = j * params.factor; c < c1; c = c + 1u) {
      let v = input[r * params.cols + c];
      if (is_nan(v)) { continue; }
      if (n == 0u) { acc = v; }
      else if (params.pool == 1u) { acc = max(acc, v); }
      else if (params.pool == 2u) { acc = min(acc, v); }
      else { acc = acc + v; }
      n = n + 1u;
    }
  }
  // A var, not a constant: a constant NaN is an error of the shader.
  var nan_bits = 0x7fc00000u;
  var h = bitcast<f32>(nan_bits);
  if (n > 0u) {
    h = acc;
    if (params.pool == 0u) { h = acc / f32(n); }
  }
  out[cell] = h;
}
