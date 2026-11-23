// The derive half of image@1: one texel of the texture `pixels` for each pixel of `image`, row by
// row, as packed RGBA8 the way the producer converts an image for the built-in draw, and the
// texture's size in `pixels_size`. A u8 channel is its byte, as the built-in takes it. An f32 value v
// becomes trunc(clamp(v * scale, 0, 1) * 255 + 0.5), and NaN becomes 0. A grey image (H, W) repeats
// its value in red, green and blue, and an image without alpha gets 255. `info` holds the width,
// the height, the channels and the pose. An image of another shape, or larger than `pixels`,
// publishes a version without texels and a size of 0. The derive half sees the last four axes of a
// tensor and not its rank, thus (1, W, 3) and (1, W, 4) read as color images of one row, and so
// does a grey image (H, W) of width 3 or 4.

const THREADS: u32 = 65536u;   // 256 workgroups of 256

// The width, the height and the channels of `image`, with a size of 0 when the piece cannot hold it.
fn image_dims() -> vec3<u32> {
    let s = shapes.image;
    var d = vec3<u32>(s.w, s.z, 1u);
    if (s.w == 3u || s.w == 4u) {
        d = vec3<u32>(s.z, s.y, s.w);
    } else if (s.y != 1u) {
        return vec3<u32>(0u, 0u, s.w);
    }
    if (d.x == 0u || d.y == 0u || d.x > 8192u || d.y > 8192u || d.y > arrayLength(&pixels) / d.x) {
        return vec3<u32>(0u, 0u, d.z);
    }
    return d;
}

// Channel i of the image from 0 to 255: the byte of a u8 image, or an f32 value times the scale.
fn channel(i: u32) -> u32 {
    let v = image_at(i);
    if (image_type == CALIPER_U8) {
        return u32(v);
    }
    let x = v * params.scale;
    if (!(x > 0.0)) {
        return 0u;
    }
    if (x >= 1.0) {
        return 255u;
    }
    return min(u32(x * 255.0 + 0.5), 255u);
}

// Column c of the pose, a (4, 4) or (3, 4) tensor row by row, or of the identity for another shape.
fn pose_column(c: u32) -> vec4<f32> {
    let s = shapes.pose;
    var col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (s.w != 4u || (s.z != 3u && s.z != 4u)) {
        col[c] = 1.0;
        return col;
    }
    col = vec4<f32>(pose[c], pose[4u + c], pose[8u + c], 0.0);
    if (s.z == 4u) {
        col.w = pose[12u + c];
    } else if (c == 3u) {
        col.w = 1.0;
    }
    return col;
}

@compute @workgroup_size(256)
fn copy_pixels(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = image_dims();
    let n = d.x * d.y;
    if (id.x == 0u) {
        pixels_size = d.xy;   // 0 by 0: a version without texels
    }
    for (var i = id.x; i < n; i = i + THREADS) {
        var rgba = 0xff000000u;
        if (d.z == 1u) {
            let g = channel(i);
            rgba = rgba | g | (g << 8u) | (g << 16u);
        } else {
            let b = i * d.z;
            rgba = channel(b) | (channel(b + 1u) << 8u) | (channel(b + 2u) << 16u);
            if (d.z == 4u) {
                rgba = rgba | (channel(b + 3u) << 24u);
            } else {
                rgba = rgba | 0xff000000u;
            }
        }
        pixels[i] = rgba;
    }
}

@compute @workgroup_size(1)
fn place_image() {
    let d = image_dims();
    info[0] = ImageInfo(vec4<u32>(d, 0u), pose_column(0u), pose_column(1u), pose_column(2u), pose_column(3u));
}
