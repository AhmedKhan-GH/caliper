// Draw half of image_grid@1: framed cards with crisp nearest-sampled tiles and a lower caption band
// for Caliper's text labels.

struct TileOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) item: u32,
}

@vertex
fn tile_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> TileOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let uv = corners[min(v, 5u)];
    let rect = tiles[i].rect;
    var out: TileOut;
    out.position = camera::clip(vec3<f32>(mix(rect.xy, rect.zw, uv), 0.0));
    out.uv = uv;
    out.item = i;
    return out;
}

@fragment
fn tile_fs(in: TileOut) -> CaliperPicked {
    let t = tiles[in.item];
    let img_x0 = 0.07;
    let img_x1 = 0.93;
    let img_y0 = 0.24;
    let img_y1 = 0.94;
    let img_u = clamp((in.uv.x - img_x0) / (img_x1 - img_x0), 0.0, 1.0);
    let img_v = clamp((in.uv.y - img_y0) / (img_y1 - img_y0), 0.0, 1.0);
    // Flip v inside the tile so row 0 of each image sits at the top of the card.
    let tex_uv = vec2<f32>(
        mix(t.uv_rect.x, t.uv_rect.z, img_u),
        mix(t.uv_rect.w, t.uv_rect.y, img_v),
    );
    let sampled = textureSampleLevel(atlas, caliper_nearest, tex_uv, 0.0);

    var color = vec3<f32>(0.10, 0.10, 0.12);
    let in_img = in.uv.x >= img_x0 && in.uv.x <= img_x1 && in.uv.y >= img_y0 && in.uv.y <= img_y1;
    if (in_img) {
        color = sampled.rgb;
    }

    let border = in.uv.x < 0.025 || in.uv.x > 0.975 || in.uv.y < 0.025 || in.uv.y > 0.975;
    if (border) {
        let err_col = vec3<f32>(0.85, 0.36, 0.25);
        let ok_col = vec3<f32>(0.34, 0.42, 0.48);
        color = select(ok_col, err_col, t.wrong != 0u);
    }

    if (pick::selected(in.item)) {
        color = mix(color, vec3<f32>(0.96, 0.78, 0.35), 0.35);
    }

    var out: CaliperPicked;
    out.color = vec4<f32>(color, 1.0);
    out.id = pick::id(in.item);
    return out;
}
