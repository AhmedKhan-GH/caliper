// The operator path (design/instrument-interface.md, Path Operator): keeps
// every stride-th row of a row-major (rows, cols) tensor, at most `points`
// rows, and copies the first out_cols columns of each kept row, so that the
// app's reference over the result with the step's columns gives the path's
// positions. One invocation copies one kept row, as the CPU reference does
// (caliper_builder/programs/operator_kernels.h).
struct Params { rows: u32, cols: u32, stride: u32, points: u32, out_cols: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x;
  if (p >= params.points) { return; }
  let row = p * params.stride;
  if (row >= params.rows) { return; }
  let width = min(params.out_cols, params.cols);
  for (var c = 0u; c < width; c = c + 1u) {
    out[p * params.out_cols + c] = input[row * params.cols + c];
  }
}
