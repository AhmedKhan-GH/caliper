// The common part of segmentation@1, in both halves.

// Color k % 16 of the built-in palette, as packed RGBA, r | g << 8 | b << 16 | a << 24.
fn palette_color(k: u32) -> u32 {
    var table = array<u32, 16>(
        0xffb4771fu, 0xff0e7fffu, 0xff2ca02cu, 0xff2827d6u, 0xffbd6794u, 0xff4b568cu, 0xffc277e3u, 0xff7f7f7fu,
        0xff22bdbcu, 0xffcfbe17u, 0xffe8c7aeu, 0xff78bbffu, 0xff8adf98u, 0xff9698ffu, 0xffd5b0c5u, 0xff949cc4u);
    return table[k % 16u];
}
