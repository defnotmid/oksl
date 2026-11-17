// texelFetch of iChannel3 at levels -1 to 6, each level's texels from two
// before its edges to two past them, p.y counted from that level's bottom
// row: tools/gpu_parity and tools/d3d12_parity bind a non-square image
// there (48 x 20 with 4 of its 6 levels, Run_Options.channel_levels on the
// CPU), so the flip takes each level's own height and levels 4 and up
// read 0. The top half fetches through a sampler parameter with an int
// lod, the bottom half directly with a uint one (-1 as 2^31).
vec4 fetch(sampler2D s, ivec2 p, int l) {
    return texelFetch(s, p, l);
}

void mainImage(out vec4 c, in vec2 f) {
    ivec2 cell = ivec2(f) / ivec2(32, 72);
    int lod = cell.x - 1;
    ivec2 size = max(textureSize(iChannel3, 0) >> max(lod, 0), ivec2(1));
    ivec2 q = ivec2(f) - cell * ivec2(32, 72);
    ivec2 p = q * (size + 4) / ivec2(32, 72) - 2;
    vec4 t;
    if (cell.y == 0) {
        t = fetch(iChannel3, p, lod);
    } else {
        t = texelFetch(iChannel3, p, lod < 0 ? 0x80000000u : uint(lod));
    }
    c = vec4(t.rg, t.b * 0.5 + t.a * 0.5, 1.0);
}
