// Geometry program that declares the model tensor writable. The sandbox refuses it with program.writes_input.
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> count: atomic<u32>;
@compute @workgroup_size(64) fn grid(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 512u) { return; }
  let k = atomicAdd(&count, 1u);
  positions[3u * k] = f32(i % 8u) * 0.25 - 0.875;
  positions[3u * k + 1u] = f32(i / 8u) * (2.0 / 64.0) - 1.0;
  positions[3u * k + 2u] = w[i];
  values[k] = w[i];
}
