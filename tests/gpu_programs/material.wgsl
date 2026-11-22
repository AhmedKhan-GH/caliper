// Material stage: the points of the geometry program in orange and yellow by their value.
fn material(in: MaterialIn) -> vec4<f32> {
  return vec4<f32>(1.0, 0.55 + 0.45 * clamp(in.value * 4.0, -1.0, 1.0), 0.05, 1.0);
}
