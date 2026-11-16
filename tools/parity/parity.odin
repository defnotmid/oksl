// The CPU side of the GPU parity checks, shared by tools/gpu_parity
// (SDL_GPU: SPIR-V or MSL) and tools/d3d12_parity (Direct3D 12: DXBC), so
// both compare the GPU with one reference: the test data (the channel
// textures and their mips, the bound float images, the storage buffers'
// words), the CPU sampler, the uniform block, the layer quad, and per
// check the shaders, the CPU's result and the comparison. Each tool only
// draws.
package parity

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
import "core:strings"
import "core:thread"
import oksl "../.."
import stbi "vendor:stb/image"

W :: 256
H :: 144
TIME :: 1.5
TEX :: 64
LAYERS :: 3
LEVELS :: 7

// The storage buffers' words: floats in [0, 1], which shaders read as
// floats or as bits.
WORDS :: 4096
test_words: [WORDS]u32

// The channel texture: a colour checker and its mip chain, each level the
// 2x2 box average of the one above, in 8 bits (so both sides see the same
// values). Layer l of it has its colours rotated l places: a
// sampler2DArray's layers, and sampler2D channel n's image is layer n %
// LAYERS (so channels differ).
texture_pixels: [(TEX * TEX * 4 * 4 - 4) / 3]u8

BOUND_R_W :: 16
BOUND_R_H :: 8
BOUND_RG_W :: 12
BOUND_RG_H :: 10
BOUND_RGBA_W :: 20
BOUND_RGBA_H :: 14

bound_r_texels: [BOUND_R_W * BOUND_R_H]f32
bound_rg_texels: [BOUND_RG_W * BOUND_RG_H][2]f32
bound_rgba_texels: [BOUND_RGBA_W * BOUND_RGBA_H][4]f32

// The bound byte and half formats (bound_formats_compare): a 37 x 23 image
// with its whole chain of levels, each its own pseudo-random texels.
BOUND_FMT_W :: 37
BOUND_FMT_H :: 23
BOUND_FMT_LEVELS :: 6
BOUND_FORMATS :: [4]oksl.Channel_Format{.Rgba8, .Gray8, .Gray16, .Rgba16f}
bound_fmt_texels: [4][BOUND_FMT_LEVELS][]u8
// What bound_formats.frag samples at each pixel: (u, v, lod, 0).
bound_fmt_queries: [W * H][4]f32

// The layer quad: position in target pixels, uv; counter-clockwise on
// screen, so front-facing under SDL_GPU's default front face (as the CPU's
// gl_FrontFacing is).
Vertex :: struct {
	pos, uv: [2]f32,
}

QUAD := [6]Vertex{{{0, 0}, {0, 0}}, {{0, H}, {0, 1}}, {{W, H}, {1, 1}}, {{0, 0}, {0, 0}}, {{W, H}, {1, 1}}, {{W, 0}, {1, 0}}}

// Level l's size of the bound format image.
bound_fmt_size :: proc(l: int) -> (w, h: int) {
	return max(BOUND_FMT_W >> uint(l), 1), max(BOUND_FMT_H >> uint(l), 1)
}

make_bound_fmt_data :: proc() {
	state := u32(12345)
	next :: proc(s: ^u32) -> u32 {
		s^ = s^ * 1664525 + 1013904223
		return s^ >> 8
	}
	formats := BOUND_FORMATS
	for f, fi in formats {
		tbs := [4]int{4, 1, 2, 8}
		tb := tbs[fi]
		for l in 0 ..< BOUND_FMT_LEVELS {
			w, h := bound_fmt_size(l)
			b := make([]u8, w * h * tb)
			if f == .Rgba16f {
				for i := 0; i < len(b); i += 2 {
					hv := transmute(u16)f16(f32(i32(next(&state) & 0xffff) - 0x8000) / 8192)
					b[i], b[i + 1] = u8(hv), u8(hv >> 8)
				}
			} else {
				for &x in b {
					x = u8(next(&state))
				}
			}
			bound_fmt_texels[fi][l] = b
		}
	}
	// Quarter by quarter: magnified past the edges, huge (repeating),
	// ties of the weights' 1/256 steps, and minified across the levels.
	for &q, i in bound_fmt_queries {
		x, y := i % W, i / W
		r0, r1, r2 := f32(next(&state)) / 16777216, f32(next(&state)) / 16777216, f32(next(&state)) / 16777216
		switch (y * 4) / H {
		case 0: q = {r0 * 1.6 - 0.3, r1 * 1.6 - 0.3, 0, 0}
		case 1: q = {r0 * 80 - 40, r1 * 80 - 40, 0, 0}
		case 2: q = {(f32(x % 64) + 0.5 + f32(x % 7) / 256) / BOUND_FMT_W, (f32(y % 23) + 0.5 + f32(x % 5) / 512) / BOUND_FMT_H, 0, 0}
		case:   q = {r0 * 1.4 - 0.2, r1 * 1.4 - 0.2, r2 * (BOUND_FMT_LEVELS + 0.5) - 0.5, 0}
		}
	}
}

// Format fi's image with `filter` and `wrap`, all its levels.
bound_fmt_image :: proc(fi: int, filter: oksl.Channel_Filter, wrap: [2]oksl.Channel_Wrap) -> oksl.Channel_Image {
	levels := make([]oksl.Channel_Level, BOUND_FMT_LEVELS - 1, context.temp_allocator)
	for &lv, i in levels {
		w, h := bound_fmt_size(i + 1)
		lv = {texels = bound_fmt_texels[fi][i + 1], width = w, height = h}
	}
	formats := BOUND_FORMATS
	return {format = formats[fi], pixels = bound_fmt_texels[fi][0], width = BOUND_FMT_W, height = BOUND_FMT_H, filter = filter, wrap = wrap, levels = levels}
}

// Samples the bound image at each pixel's query.
BOUND_FMT_FRAG :: `#version 310 es
precision highp float;
readonly buffer Q { vec4 q[]; };
uniform sampler2D s;
layout(location = 0) out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 c = q[p.y * 256 + p.x];
    o = textureLod(s, c.xy, c.z);
}`

// The wraps bound_formats_compare draws: both clamped, both repeating, u
// repeating with v clamped.
BOUND_FMT_WRAPS :: [3][2]oksl.Channel_Wrap{{.Clamp, .Clamp}, {.Repeat, .Repeat}, {.Repeat, .Clamp}}

// One draw of BOUND_FMT_FRAG with a bound format image against the CPU
// (bound_compare).
bound_formats_compare :: proc(m: ^oksl.Module, fi: int, filter: oksl.Channel_Filter, wrap: [2]oksl.Channel_Wrap, gpu_f: [][4]f32, scratch: ^oksl.Scratch, metal: bool) -> (string, bool) {
	return bound_compare(m, bound_fmt_image(fi, filter, wrap), bound_fmt_queries[:], gpu_f, scratch, metal)
}

// ---- texelFetch out of range (bound_fetch_compare) ----------------------------------------
//
// A measurement, not oksl's rule against itself: what a GPU's texelFetch
// returns outside the level or the texture, for the bound format images
// bound as SDL binds a texture. oksl returns (0, 0, 0, 0) there.

// texelFetch at each pixel's query: (x, y, level, 0).
BOUND_FETCH_FRAG :: `#version 310 es
precision highp float;
precision highp int;
readonly buffer Q { ivec4 q[]; };
uniform sampler2D s;
layout(location = 0) out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec4 c = q[p.y * 256 + p.x];
    o = texelFetch(s, c.xy, c.z);
}`

// The queries, and each one's coordinates moved into the texture: the
// level clamp(l, 0, top), then x and y clamped to its edges, or wrapped
// (floored modulo).
bound_fetch_queries: [W * H][4]i32
bound_fetch_clamped: [W * H][4]i32
bound_fetch_wrapped: [W * H][4]i32

// Every pairing of x and y from -2^20, -2, -1, 0, the middle, n - 1, n and
// n + 5 (n the level's size, the top level's past the top), at levels -1,
// 0, 2, the top, the top + 1 and the top + 3, repeated over the target.
make_bound_fetch_data :: proc() {
	LEVELS :: [6]i32{-1, 0, 2, BOUND_FMT_LEVELS - 1, BOUND_FMT_LEVELS, BOUND_FMT_LEVELS + 3}
	cases := make([dynamic][4]i32, context.temp_allocator)
	for l in LEVELS {
		w, h := bound_fmt_size(int(clamp(l, 0, BOUND_FMT_LEVELS - 1)))
		coords :: proc(n: int) -> [8]i32 {
			return {-1 << 20, -2, -1, 0, i32(n / 2), i32(n - 1), i32(n), i32(n + 5)}
		}
		for y in coords(h) {
			for x in coords(w) {
				append(&cases, [4]i32{x, y, l, 0})
			}
		}
	}
	for &q, i in bound_fetch_queries {
		q = cases[i % len(cases)]
		lc := clamp(q.z, 0, BOUND_FMT_LEVELS - 1)
		w, h := bound_fmt_size(int(lc))
		bound_fetch_clamped[i] = {clamp(q.x, 0, i32(w - 1)), clamp(q.y, 0, i32(h - 1)), lc, 0}
		bound_fetch_wrapped[i] = {q.x %% i32(w), q.y %% i32(h), lc, 0}
	}
}

// Sorts a GPU's texelFetch results at bound_fetch_queries: inside (level
// and texel in range), and outside the texture or the level, each result
// as zero (rgb 0, alpha 0 or 1), the clamped texel, the wrapped one, more
// than one of those, or other. Within 1e-3 of the CPU's texels (decoded
// exactly by both). The verdict: every inside result the CPU's texel.
bound_fetch_compare :: proc(m: ^oksl.Module, fi: int, gpu_f: [][4]f32, scratch: ^oksl.Scratch) -> (string, bool) {
	im := bound_fmt_image(fi, .Nearest, {.Clamp, .Clamp})
	clamped := make([][4]f32, W * H, context.temp_allocator)
	wrapped := make([][4]f32, W * H, context.temp_allocator)
	oksl.run_f32(&m.program, nil, W, H, clamped, scratch, {channel_images = {im}, buffers = {slice.reinterpret([]u32, bound_fetch_clamped[:])}})
	oksl.run_f32(&m.program, nil, W, H, wrapped, scratch, {channel_images = {im}, buffers = {slice.reinterpret([]u32, bound_fetch_wrapped[:])}})
	near :: proc(a, b: [4]f32) -> bool {
		for k in 0 ..< 4 {
			if !(abs(a[k] - b[k]) <= 1e-3) {
				return false
			}
		}
		return true
	}
	Tally :: struct {
		n, zero0, zero1, clamped, wrapped, several, other: int,
		example:                                          string,
	}
	inside: Tally
	coord, level: Tally
	for i in 0 ..< W * H {
		q := bound_fetch_queries[i]
		g := gpu_f[i]
		lw, lh := bound_fmt_size(int(clamp(q.z, 0, BOUND_FMT_LEVELS - 1)))
		level_in := q.z >= 0 && q.z < BOUND_FMT_LEVELS
		texel_in := q.x >= 0 && q.y >= 0 && int(q.x) < lw && int(q.y) < lh
		t := &inside if level_in && texel_in else (&coord if level_in else &level)
		t.n += 1
		zero := g.xyz == {0, 0, 0} && (g.w == 0 || g.w == 1)
		cl, wr := near(g, clamped[i]), near(g, wrapped[i])
		switch {
		case int(zero) + int(cl) + int(wr) > 1 && !(cl && wr && !zero && t == &inside): t.several += 1
		case zero && g.w == 0: t.zero0 += 1
		case zero:             t.zero1 += 1
		case cl:               t.clamped += 1
		case wr:               t.wrapped += 1
		case:
			t.other += 1
			if t.example == "" {
				t.example = fmt.tprintf(" (e.g. %v at %v; clamped %v)", g, q, clamped[i])
			}
		}
	}
	line :: proc(t: Tally) -> string {
		return fmt.tprintf("%d: zero %d + %d (alpha 0 + 1), clamped %d, wrapped %d, several %d, other %d%s", t.n, t.zero0, t.zero1, t.clamped, t.wrapped, t.several, t.other, t.example)
	}
	inside_ok := inside.clamped + inside.several == inside.n && inside.zero0 + inside.zero1 + inside.wrapped + inside.other == 0
	return fmt.tprintf("inside %s; texel out %s; level out %s", line(inside), line(coord), line(level)), inside_ok
}

// Metal's level of detail (Apple GPUs, measured on the M4 with
// tools/gpu_parity's queries): the lod rounded to a half float, its
// fraction truncated to 1/64 (oksl's rule and RADV round it to 1/256, to
// nearest even). A query's lod given the CPU as this, the CPU blends
// levels with Metal's weights; texelFetch's (lod <= -1000) as they are,
// and lods of 16 and up (past any level; a half overflows to inf at 65520).
metal_lod :: proc(lod: f32) -> f32 {
	if !(lod > 0) || lod >= 16 {
		return lod
	}
	l := f32(f16(lod))
	i := math.floor(l)
	return i + math.floor((l - i) * 64) / 64
}

// A draw of `im` at `queries` ((u, v, lod, layer) per pixel, the
// module's storage buffer) against the CPU: the worst error and the share
// of exact components. Within 1e-6 (the GPU's last bits where it blends
// texels or levels), Linear Rgba16f 2e-3 (RADV filters halves in less than
// f32's precision). On Metal, which blends levels with its own weights
// (metal_lod, given the CPU), rounds its texel weights half up from an
// f32 coordinate (`u * n - 0.5`: one 1/256 step from the CPU's at and
// near ties) and returns a filtered 8-bit texel to 1/16 of a step, a
// 16-bit one to a step and a half to a half: Nearest within 1.3e-4
// (8-bit), 8e-6 (Gray16), 1e-3 (halves in [-4, 4]: half a half's ulp);
// Linear one weight step of the texels' range more, 4.1e-3 (unorm, [0, 1])
// or 0.033 (halves).
bound_compare :: proc(m: ^oksl.Module, im: oksl.Channel_Image, queries: [][4]f32, gpu_f: [][4]f32, scratch: ^oksl.Scratch, metal: bool) -> (string, bool) {
	cpu_f := make([][4]f32, W * H, context.temp_allocator)
	q := queries
	if metal {
		q = slice.clone(queries, context.temp_allocator)
		for &x in q {
			x.z = metal_lod(x.z)
		}
	}
	oksl.run_f32(&m.program, nil, W, H, cpu_f, scratch, {channel_images = {im}, buffers = {slice.reinterpret([]u32, q)}})
	tol: f32 = 2e-3 if im.filter == .Linear && im.format == .Rgba16f else 1e-6
	if metal {
		#partial switch im.format {
		case .Rgba8, .Gray8: tol = 4.1e-3 if im.filter == .Linear else 1.3e-4
		case .Gray16:        tol = 4.1e-3 if im.filter == .Linear else 8e-6
		case .Rgba16f:       tol = 0.033 if im.filter == .Linear else 1e-3
		}
	}
	worst: f32
	exact, bad := 0, 0
	for i in 0 ..< W * H {
		for k in 0 ..< 4 {
			g, c := gpu_f[i][k], cpu_f[i][k]
			d := abs(g - c)
			if g == c {
				exact += 1
			}
			if !(d <= tol) {
				bad += 1
			}
			worst = max(worst, d)
		}
	}
	return fmt.tprintf("worst %.3g, exact %.2f%%, over %.2g: %d", worst, 100 * f64(exact) / f64(W * H * 4), tol, bad), bad == 0
}

// ---- layered bound images (bound_layers_compare) -------------------------------------------

// The bound format images with BOUND_LAYERS layers: layer 0 is
// bound_fmt_texels', the others their own pseudo-random texels; packed for
// the CPU as each level's layers one after another.
BOUND_LAYERS :: 3
bound_layer_texels: [4][BOUND_LAYERS][BOUND_FMT_LEVELS][]u8
bound_layer_packed: [4][BOUND_FMT_LEVELS][]u8
// What BOUND_LAYERS_FRAG samples at each pixel: (u, v, lod, layer), or
// with lod <= -1000 texelFetch's (x, y, -1000 - level, layer).
bound_layer_queries: [W * H][4]f32

make_bound_layer_data :: proc() {
	state := u32(777)
	next :: proc(s: ^u32) -> u32 {
		s^ = s^ * 1664525 + 1013904223
		return s^ >> 8
	}
	formats := BOUND_FORMATS
	for f, fi in formats {
		for k in 0 ..< BOUND_LAYERS {
			for l in 0 ..< BOUND_FMT_LEVELS {
				if k == 0 {
					bound_layer_texels[fi][k][l] = bound_fmt_texels[fi][l]
					continue
				}
				b := make([]u8, len(bound_fmt_texels[fi][l]))
				if f == .Rgba16f {
					for i := 0; i < len(b); i += 2 {
						hv := transmute(u16)f16(f32(i32(next(&state) & 0xffff) - 0x8000) / 8192)
						b[i], b[i + 1] = u8(hv), u8(hv >> 8)
					}
				} else {
					for &x in b {
						x = u8(next(&state))
					}
				}
				bound_layer_texels[fi][k][l] = b
			}
		}
		for l in 0 ..< BOUND_FMT_LEVELS {
			n := len(bound_fmt_texels[fi][l])
			b := make([]u8, n * BOUND_LAYERS)
			for k in 0 ..< BOUND_LAYERS {
				copy(b[k * n:], bound_layer_texels[fi][k][l])
			}
			bound_layer_packed[fi][l] = b
		}
	}
	// Quarter by quarter: magnified at every tie of the layer coordinate
	// and either side of it; minified across the levels at random layers;
	// NaN, infinite, huge and negative layers; texelFetch at in-range
	// texels, levels and layers.
	specials := [?]f32{math.nan_f32(), math.inf_f32(1), math.inf_f32(-1), 1e30, -1e30, -0.0, 0.5, 1.5, 2.5, -0.5, 3.5, 0.49999997, 0.50000006, 1.4999999, 1.5000001}
	for &q, i in bound_layer_queries {
		x, y := i % W, i / W
		r0, r1, r2 := f32(next(&state)) / 16777216, f32(next(&state)) / 16777216, f32(next(&state)) / 16777216
		switch (y * 4) / H {
		case 0:
			t := f32(x % 12) * 0.5 - 1.5
			e := [3]f32{0, 1.2e-7, -1.2e-7}
			q = {r0 * 1.6 - 0.3, r1 * 1.6 - 0.3, 0, t + e[y % 3] * max(1, abs(t))}
		case 1:
			q = {r0 * 1.4 - 0.2, r1 * 1.4 - 0.2, r2 * (BOUND_FMT_LEVELS + 0.5) - 0.5, f32(next(&state) % 900) / 150 - 1.5}
		case 2:
			q = {r0 * 1.4 - 0.2, r1 * 1.4 - 0.2, r2 * 3 - 0.5, specials[(x + y) % len(specials)]}
		case:
			l := int(next(&state) % BOUND_FMT_LEVELS)
			w, h := bound_fmt_size(l)
			q = {f32(next(&state) % u32(w)), f32(next(&state) % u32(h)), -1000 - f32(l), f32(next(&state) % BOUND_LAYERS)}
		}
	}
}

// Format fi's layered image with `filter` and `wrap`, all its levels.
bound_layer_image :: proc(fi: int, filter: oksl.Channel_Filter, wrap: [2]oksl.Channel_Wrap) -> oksl.Channel_Image {
	levels := make([]oksl.Channel_Level, BOUND_FMT_LEVELS - 1, context.temp_allocator)
	for &lv, i in levels {
		w, h := bound_fmt_size(i + 1)
		lv = {texels = bound_layer_packed[fi][i + 1], width = w, height = h}
	}
	formats := BOUND_FORMATS
	return {format = formats[fi], pixels = bound_layer_packed[fi][0], width = BOUND_FMT_W, height = BOUND_FMT_H, filter = filter, wrap = wrap, levels = levels, layers = BOUND_LAYERS}
}

// Samples the layered image at each pixel's query.
BOUND_LAYERS_FRAG :: `#version 310 es
precision highp float;
readonly buffer Q { vec4 q[]; };
uniform sampler2DArray s;
layout(location = 0) out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 c = q[p.y * 256 + p.x];
    if (c.z <= -1000.0) {
        o = texelFetch(s, ivec3(c.xyw), int(-1000.0 - c.z));
    } else {
        o = textureLod(s, c.xyw, c.z);
    }
}`

// One draw of BOUND_LAYERS_FRAG with a layered format image against the
// CPU (bound_compare).
bound_layers_compare :: proc(m: ^oksl.Module, fi: int, filter: oksl.Channel_Filter, wrap: [2]oksl.Channel_Wrap, gpu_f: [][4]f32, scratch: ^oksl.Scratch, metal: bool) -> (string, bool) {
	return bound_compare(m, bound_layer_image(fi, filter, wrap), bound_layer_queries[:], gpu_f, scratch, metal)
}

// ---- built levels (built_levels_compare) --------------------------------------------------

// The formats and sizes (width, height, layers) built_levels checks
// channel_build_levels against GenerateMipmapsForGPUTexture with: even,
// odd and 1-pixel sizes, 1280 x 720 and 960 x 540 as o3d's frame maps.
Built_Format :: struct {
	format: oksl.Channel_Format,
	comps:  int,
}
BUILT_FORMATS :: [7]Built_Format{{.Rgba8, 0}, {.Gray8, 0}, {.Gray16, 0}, {.Rgba16f, 0}, {.F32, 1}, {.F32, 2}, {.F32, 4}}
BUILT_SIZES :: [?][3]int{{37, 23, 2}, {64, 32, 1}, {1, 37, 1}, {37, 1, 1}, {2, 1, 1}, {5, 7, 1}, {15, 15, 1}, {513, 5, 1}, {257, 3, 1}, {300, 200, 1}, {960, 540, 1}, {1280, 720, 1}}

// n bytes, 16-byte aligned (a []u8 from an arena may start anywhere; the
// formats read u16 and f32).
built_bytes :: proc(n: int, allocator := context.allocator) -> []u8 {
	return slice.to_bytes(make([][4]u32, (n + 15) / 16, allocator))[:n]
}

built_texel_bytes :: proc(bf: Built_Format) -> int {
	switch bf.format {
	case .F32:     return 4 * bf.comps
	case .Rgba8:   return 4
	case .Gray8:   return 1
	case .Gray16:  return 2
	case .Rgba16f: return 8
	}
	return 0
}

built_components :: proc(bf: Built_Format) -> int {
	return bf.comps if bf.format == .F32 else (4 if bf.format == .Rgba8 || bf.format == .Rgba16f else 1)
}

// Level 0 of the size's image, its layers one after another: bytes,
// halves in [-4, 4] (one in 64 times 15000 larger), floats in [-4, 4].
built_level0 :: proc(bf: Built_Format, size: [3]int, seed: u32) -> []u8 {
	state := seed * 2654435761 + 1
	next :: proc(s: ^u32) -> u32 {
		s^ = s^ * 1664525 + 1013904223
		return s^ >> 8
	}
	b := built_bytes(size[0] * size[1] * size[2] * built_texel_bytes(bf))
	#partial switch bf.format {
	case .Rgba16f:
		for i := 0; i < len(b); i += 2 {
			v := f32(next(&state)) / 2097152 - 4
			if next(&state) % 64 == 0 {
				v *= 15000
			}
			(^f16)(&b[i])^ = f16(v)
		}
	case .F32:
		for i := 0; i < len(b); i += 4 {
			(^f32)(&b[i])^ = f32(next(&state)) / 2097152 - 4
		}
	case:
		for &x in b {
			x = u8(next(&state) >> 16) // the LCG's top bits
		}
	}
	return b
}

// A component of texel i of a level (`bytes`, its layers one after
// another) as the GPU reads it: unorm as its integer, halves and floats.
built_value :: proc(bf: Built_Format, bytes: []u8, i, k: int) -> f64 {
	tb := built_texel_bytes(bf)
	p := bytes[i * tb:]
	switch bf.format {
	case .F32:     return f64((^f32)(&p[4 * k])^)
	case .Rgba8:   return f64(p[k])
	case .Gray8:   return f64(p[0])
	case .Gray16:  return f64(u16(p[0]) | u16(p[1]) << 8)
	case .Rgba16f: return f64(f32((^f16)(&p[2 * k])^))
	}
	return 0
}

// Two values apart in the format's steps: unorm integers, halves' and
// floats' ulps.
built_steps :: proc(bf: Built_Format, a, b: f64) -> f64 {
	if a == b || (a != a && b != b) {
		return 0
	}
	#partial switch bf.format {
	case .Rgba8, .Gray8, .Gray16:
		return abs(a - b)
	case .Rgba16f:
		ord16 :: proc(x: f64) -> i64 {
			h := transmute(u16)f16(f32(x))
			return i64(h & 0x7fff) if h & 0x8000 == 0 else -i64(h & 0x7fff)
		}
		return f64(abs(ord16(a) - ord16(b)))
	}
	ord32 :: proc(x: f64) -> i64 {
		u := transmute(u32)f32(x)
		return i64(u & 0x7fffffff) if u & 0x80000000 == 0 else -i64(u & 0x7fffffff)
	}
	return f64(abs(ord32(a) - ord32(b)))
}

// How far one format's built levels are from the GPU's.
Built_Stats :: struct {
	total, exact, one: int, // components, equal, within one step
	worst_steps:       f64, // each level built from the GPU's level above
	worst_chain:       f64, // the whole chain built from level 0
	worst_rel:         f64, // |cpu - gpu| over the largest of its four parents
	over:              int, // past the tolerance (built_levels_compare)
	worst_where:       string,
	unwritten:         [dynamic]string, // levels the GPU left as they were
}

built_stats_destroy :: proc(st: ^Built_Stats) {
	delete(st.worst_where)
	for s in st.unwritten {
		delete(s)
	}
	delete(st.unwritten)
}

// One size of format bf against the GPU's levels `gpu` (each level's
// layers one after another; written[l] false where the GPU left level l
// as it was): channel_build_levels from each GPU level the one below
// (the rule, step by step) and from level 0 (the chain). A component
// passes within a weight step of its four parents' spread plus two of
// the format's steps (RADV writes halves toward zero and filters them in
// less than f32's precision: Rgba16f also 2e-3 of the largest parent).
built_levels_compare :: proc(bf: Built_Format, size: [3]int, gpu: [][]u8, written: []bool, st: ^Built_Stats) {
	w0, h0, layers := size[0], size[1], size[2]
	n := len(gpu)
	tb := built_texel_bytes(bf)
	nc := built_components(bf)
	chain := make([]oksl.Channel_Level, n - 1, context.temp_allocator)
	chain_bytes := built_bytes(oksl.channel_levels_bytes(bf.format, bf.comps, w0, h0, n, layers), context.temp_allocator)
	oksl.channel_levels_layout(chain, chain_bytes, bf.format, bf.comps, w0, h0, layers)
	image :: proc(bf: Built_Format, texels: []u8, w, h, layers: int, levels: []oksl.Channel_Level) -> oksl.Channel_Image {
		im := oksl.Channel_Image{format = bf.format, components = bf.comps, width = w, height = h, layers = layers, levels = levels}
		if bf.format == .F32 {
			im.data = slice.reinterpret([]f32, texels)
		} else {
			im.pixels = texels
		}
		return im
	}
	oksl.channel_build_levels(image(bf, gpu[0], w0, h0, layers, chain))
	for l in 1 ..< n {
		pw, ph := max(w0 >> uint(l - 1), 1), max(h0 >> uint(l - 1), 1)
		lw, lh := max(w0 >> uint(l), 1), max(h0 >> uint(l), 1)
		if !written[l] {
			append(&st.unwritten, fmt.aprintf("%dx%d L%d", w0, h0, l))
			continue
		}
		step := built_bytes(lw * lh * layers * tb, context.temp_allocator)
		one := [1]oksl.Channel_Level{{texels = step, width = lw, height = lh}}
		oksl.channel_build_levels(image(bf, gpu[l - 1], pw, ph, layers, one[:]))
		for k in 0 ..< layers {
			for y in 0 ..< lh {
				for x in 0 ..< lw {
					i := (k * lh + y) * lw + x
					// The parents the rule reads (its taps' clamped texels).
					px0 := min(int(math.floor((f64(x) + 0.5) * f64(pw) / f64(lw) - 0.5)), pw - 1)
					py0 := min(int(math.floor((f64(y) + 0.5) * f64(ph) / f64(lh) - 0.5)), ph - 1)
					for c in 0 ..< nc {
						g := built_value(bf, gpu[l], i, c)
						v := built_value(bf, step, i, c)
						lo, hi, big := math.INF_F64, -math.INF_F64, 0.0
						for dy in 0 ..< 2 {
							for dx in 0 ..< 2 {
								p := built_value(bf, gpu[l - 1], (k * ph + min(py0 + dy, ph - 1)) * pw + min(px0 + dx, pw - 1), c)
								lo, hi, big = min(lo, p), max(hi, p), max(big, abs(p))
							}
						}
						d := built_steps(bf, v, g)
						st.total += 1
						if d == 0 {
							st.exact += 1
						} else if d <= 1 {
							st.one += 1
						}
						unit := 1.0
						#partial switch bf.format {
						case .Rgba16f: unit = abs(v) * (1.0 / 1024) + 1.0 / 16777216
						case .F32:     unit = abs(v) * (1.0 / 8388608) + 1e-45
						}
						tol := (hi - lo) / 256 + 2 * unit + (2e-3 * big if bf.format == .Rgba16f else 0)
						if !(abs(v - g) <= tol) {
							st.over += 1
						}
						st.worst_rel = max(st.worst_rel, abs(v - g) / max(big, 1e-30))
						if d > st.worst_steps {
							st.worst_steps = d
							delete(st.worst_where)
							st.worst_where = fmt.aprintf("%dx%d L%d layer %d (%d, %d)[%d]: cpu %v gpu %v", w0, h0, l, k, x, y, c, v, g)
						}
						cv := built_value(bf, chain[l - 1].texels, i, c)
						st.worst_chain = max(st.worst_chain, built_steps(bf, cv, g))
					}
				}
			}
		}
	}
}

make_data :: proc() {
	make_bound_fmt_data()
	make_bound_fetch_data()
	make_bound_layer_data()
	for &w, i in test_words {
		w = transmute(u32)(f32((i * 37 + 11) % 256) / 255)
	}
	for y in 0 ..< BOUND_R_H {
		for x in 0 ..< BOUND_R_W {
			bound_r_texels[y * BOUND_R_W + x] = f32(((x * 5 + y * 3) % 25) - 8) * 0.125
		}
	}
	for y in 0 ..< BOUND_RG_H {
		for x in 0 ..< BOUND_RG_W {
			bound_rg_texels[y * BOUND_RG_W + x] = {f32(((x * 3 + y * 7 + 2) % 25) - 8) * 0.125, f32(((x * 7 + y * 5 + 9) % 25) - 8) * 0.125}
		}
	}
	for y in 0 ..< BOUND_RGBA_H {
		for x in 0 ..< BOUND_RGBA_W {
			bound_rgba_texels[y * BOUND_RGBA_W + x] = {
				f32(((x * 3 + y * 5 + 1) % 25) - 8) * 0.125,
				f32(((x * 5 + y * 7 + 4) % 25) - 8) * 0.125,
				f32(((x * 7 + y * 3 + 11) % 25) - 8) * 0.125,
				f32(((x * 2 + y * 3 + 5) % 17) - 2) * 0.125,
			}
		}
	}
	for y in 0 ..< TEX {
		for x in 0 ..< TEX {
			i := (y * TEX + x) * 4
			on := ((x / 8) + (y / 8)) % 2 == 0
			texture_pixels[i + 0] = u8(x * 4)
			texture_pixels[i + 1] = u8(y * 4)
			texture_pixels[i + 2] = 230 if on else 50
			texture_pixels[i + 3] = 255
		}
	}
	for l in 1 ..< LEVELS {
		src, dst := level_offset(l - 1), level_offset(l)
		n := TEX >> uint(l)
		for y in 0 ..< n {
			for x in 0 ..< n {
				for k in 0 ..< 4 {
					sum := 2
					for d in 0 ..< 4 {
						sum += int(texture_pixels[src + ((2 * y + d / 2) * 2 * n + 2 * x + d % 2) * 4 + k])
					}
					texture_pixels[dst + (y * n + x) * 4 + k] = u8(sum / 4)
				}
			}
		}
	}
}

// Where level l starts in texture_pixels.
level_offset :: proc(l: int) -> int {
	o := 0
	for i in 0 ..< l {
		o += (TEX >> uint(i)) * (TEX >> uint(i)) * 4
	}
	return o
}

// Byte i of layer `layer`: the checker with its colours rotated `layer`
// places (layer 0 is the checker).
layer_pixel :: proc(i, layer: int) -> u8 {
	k := i % 4
	return texture_pixels[i - k + (k + layer) % 3] if k < 3 else texture_pixels[i]
}

texel :: proc(l, x, y, layer: int) -> [4]f32 {
	n := TEX >> uint(l)
	xx := ((x % n) + n) % n
	yy := ((y % n) + n) % n
	i := level_offset(l) + (yy * n + xx) * 4
	return {f32(layer_pixel(i, layer)), f32(layer_pixel(i + 1, layer)), f32(layer_pixel(i + 2, layer)), f32(layer_pixel(i + 3, layer))} / 255
}

bilinear :: proc(l: int, uv: [2]f32, layer: int) -> [4]f32 {
	n := f32(int(TEX) >> uint(l))
	fx := uv.x * n - 0.5
	fy := uv.y * n - 0.5
	x0 := int(fx) if fx >= 0 else int(fx) - 1
	y0 := int(fy) if fy >= 0 else int(fy) - 1
	tx := fx - f32(x0)
	ty := fy - f32(y0)
	a, b := texel(l, x0, y0, layer), texel(l, x0 + 1, y0, layer)
	c, d := texel(l, x0, y0 + 1, layer), texel(l, x0 + 1, y0 + 1, layer)
	return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
}

// The layer a channel's sample reads: the array's (clamped) for a
// sampler2DArray, channel % LAYERS for a sampler2D. `arrays` is the
// module's Module.channel_arrays.
channel_layer :: proc(arrays: bit_set[0 ..< oksl.SAMPLER_COUNT], channel, layer: int) -> int {
	return clamp(layer, 0, LAYERS - 1) if channel in arrays else channel % LAYERS
}

// Trilinear and repeating, the level clamped to the chain: what the GPU
// sampler the tools bind does. Run_Options.user is the module's
// channel_arrays.
cpu_sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
	ly := channel_layer((^bit_set[0 ..< oksl.SAMPLER_COUNT])(user)^, channel, layer)
	if !(lod > 0) {
		return bilinear(0, uv, ly)
	}
	level := min(lod, LEVELS - 1)
	l0 := int(level)
	l1 := min(l0 + 1, LEVELS - 1)
	t := level - f32(l0)
	return bilinear(l0, uv, ly) * (1 - t) + bilinear(l1, uv, ly) * t
}

// The fragment shaders' uniform block: the built-ins for a W x H target
// and the testdata shaders' own uniforms.
uniform_block :: proc(m: ^oksl.Module) -> []u8 {
	block := make([]u8, m.block_size)
	ch := [3]f32{TEX, TEX, 1}
	oksl.write_builtins(block, {resolution = {W, H, 1}, time = TIME, time_delta = 1.0 / 60, frame = 90, frame_rate = 60, channel_resolution = {ch, ch, ch, ch}})
	oksl.set_uniform(m, block, "uSpeed", {0.5})
	oksl.set_uniform(m, block, "uScale", {1.5})
	oksl.set_uniform(m, block, "uTint", {1, 0.9, 0.8})
	oksl.set_uniform(m, block, "uPalette", {1, 0.2, 0.2, 1}, 0)
	oksl.set_uniform(m, block, "uPalette", {0.2, 1, 0.2, 1}, 1)
	oksl.set_uniform(m, block, "uPalette", {0.2, 0.2, 1, 1}, 2)
	oksl.set_uniform(m, block, "uRot", {0.8, 0.6, -0.6, 0.8})
	oksl.set_uniform(m, block, "uMode", {1})
	oksl.set_uniform(m, block, "uInvert", {0})
	return block
}

// The CPU's run options for a fragment shader on the test data: the
// sampler, the channels' sizes and every storage buffer the test words.
// `arrays` must outlive the run (Run_Options.user points at it).
run_options :: proc(m: ^oksl.Module, arrays: ^bit_set[0 ..< oksl.SAMPLER_COUNT], sizes: ^[oksl.SAMPLER_COUNT][3]i32) -> oksl.Run_Options {
	arrays^ = m.channel_arrays
	for &s, ch in sizes {
		s = {TEX, TEX, LAYERS if ch in m.channel_arrays else 1}
	}
	buffers := make([][]u32, len(m.storage_buffers), context.temp_allocator)
	for &b in buffers {
		b = test_words[:]
	}
	return {sampler = cpu_sampler, user = arrays, channel_sizes = sizes[:], buffers = buffers}
}

write_png :: proc(path: string, px: []u8) {
	stbi.write_png(strings.clone_to_cstring(path, context.temp_allocator), W, H, 4, raw_data(px), W * 4)
}

// ---- a fragment shader on the layer quad ------------------------------------------------

// The CPU's RGBA8 render of a fragment shader on the layer quad (discarded
// pixels keep the target's clear, opaque black), in the temp allocator.
fragment_reference :: proc(m: ^oksl.Module, block: []u8, scratch: ^oksl.Scratch) -> []u8 {
	cpu_px := make([]u8, W * H * 4, context.temp_allocator)
	for i in 0 ..< W * H {
		cpu_px[i * 4 + 3] = 255
	}
	arrays: bit_set[0 ..< oksl.SAMPLER_COUNT]
	sizes: [oksl.SAMPLER_COUNT][3]i32
	oksl.run(&m.program, block, W, H, cpu_px, W * 4, scratch, run_options(m, &arrays, &sizes))
	return cpu_px
}

// Compares RGBA8 renders: within 3/255 everywhere, or differences
// confined to a few pixels where a threshold (step, comparisons) meets
// transcendental results that differ by ulps between the GPU and the CPU.
// Writes <name>_<gpu_tag>.png, <name>_cpu.png and <name>_diff.png and
// prints a line; true when they match.
compare_rgba8 :: proc(name, gpu_tag, out_dir: string, cpu_px, gpu_px: []u8) -> bool {
	max_d, sum, over2, over8 := 0, 0, 0, 0
	diff := make([]u8, W * H * 4, context.temp_allocator)
	for i := 0; i < len(cpu_px); i += 4 {
		pd := 0
		for k in 0 ..< 4 {
			d := abs(int(cpu_px[i + k]) - int(gpu_px[i + k]))
			pd = max(pd, d)
			sum += d
		}
		max_d = max(max_d, pd)
		if pd > 2 {over2 += 1}
		if pd > 8 {over8 += 1}
		v := u8(min(pd * 16, 255))
		diff[i], diff[i + 1], diff[i + 2], diff[i + 3] = v, v, v, 255
	}
	mean := f64(sum) / f64(len(cpu_px))
	verdict := "ok"
	if max_d > 3 {
		verdict = "ok (threshold edges)" if over8 <= W * H / 100 && mean < 1 else "MISMATCH"
	}
	fmt.printfln("%-20s max diff %d, mean %.4f, pixels >2: %d, >8: %d  %s", name, max_d, mean, over2, over8, verdict)
	write_png(fmt.tprintf("%s/%s_%s.png", out_dir, name, gpu_tag), gpu_px)
	write_png(fmt.tprintf("%s/%s_cpu.png", out_dir, name), cpu_px)
	write_png(fmt.tprintf("%s/%s_diff.png", out_dir, name), diff)
	return verdict != "MISMATCH"
}

// ---- float renders ----------------------------------------------------------------------

// The CPU's float render of a fragment shader on the layer quad, the
// target cleared to (0, 0, 0, 1), in the temp allocator.
fragment_reference_f32 :: proc(m: ^oksl.Module, block: []u8, scratch: ^oksl.Scratch) -> [][4]f32 {
	px := make([][4]f32, W * H, context.temp_allocator)
	for &p in px {
		p = {0, 0, 0, 1}
	}
	arrays: bit_set[0 ..< oksl.SAMPLER_COUNT]
	sizes: [oksl.SAMPLER_COUNT][3]i32
	oksl.run_f32(&m.program, block, W, H, px, scratch, run_options(m, &arrays, &sizes))
	return px
}

// Float tolerances: `abs` absolute or `rel` relative; NaN matches NaN,
// infinities match exactly, and a zero only a zero of either sign (GLSL
// does not require signed zeros, and the JIT and the interpreter differ on
// some).
Tolerance :: struct {
	abs, rel: f32,
}

close :: proc(gpu, cpu: f32, tol: Tolerance) -> bool {
	switch {
	case math.is_nan(cpu):
		return math.is_nan(gpu)
	case math.is_inf(cpu) || cpu == 0 && tol.abs == 0:
		return gpu == cpu
	}
	d := abs(gpu - cpu)
	return d <= tol.abs || d <= tol.rel * abs(cpu)
}

// Compares float renders component by component under `tol`, skipping the
// pixels `skip` names; prints a line and up to 4 mismatches.
compare_f32 :: proc(name: string, cpu, gpu: [][4]f32, tol: Tolerance, skip: proc(x, y: int) -> bool = nil) -> bool {
	bad := 0
	worst: f32
	for y in 0 ..< H {
		for x in 0 ..< W {
			if skip != nil && skip(x, y) {
				continue
			}
			c, g := cpu[y * W + x], gpu[y * W + x]
			for k in 0 ..< 4 {
				if !close(g[k], c[k], tol) {
					if bad < 4 {
						fmt.printfln("  %s (%d, %d)[%d]: gpu %v (%x), cpu %v (%x)", name, x, y, k, g[k], transmute(u32)g[k], c[k], transmute(u32)c[k])
					}
					bad += 1
				} else if !math.is_nan(c[k]) && !math.is_inf(c[k]) {
					worst = max(worst, abs(g[k] - c[k]))
				}
			}
		}
	}
	fmt.printfln("%-20s float, worst %.3g, mismatches %d  %s", name, worst, bad, "ok" if bad == 0 else "MISMATCH")
	return bad == 0
}

// ---- bound float channels ---------------------------------------------------------------

// The GPU binds channels 0-2 to the R32F, RG32F and RGBA32F images with a
// nearest sampler and 3-5 to them with a linear one, all clamped.
bound_channel_images :: proc() -> [6]oksl.Channel_Image {
	return {
		oksl.channel_image(bound_r_texels[:], BOUND_R_W, BOUND_R_H, .Nearest),
		oksl.channel_image(bound_rg_texels[:], BOUND_RG_W, BOUND_RG_H, .Nearest),
		oksl.channel_image(bound_rgba_texels[:], BOUND_RGBA_W, BOUND_RGBA_H, .Nearest),
		oksl.channel_image(bound_r_texels[:], BOUND_R_W, BOUND_R_H, .Linear),
		oksl.channel_image(bound_rg_texels[:], BOUND_RG_W, BOUND_RG_H, .Linear),
		oksl.channel_image(bound_rgba_texels[:], BOUND_RGBA_W, BOUND_RGBA_H, .Linear),
	}
}

// testdata/bound_channels.frag: nearest rows (the top half) exact, linear
// rows within 1e-6 (the GPU's last bits) except the components a narrow
// image fills (exact). With `coarse_weights` (WARP, which D3D's 8-bit
// minimum lets filter with other weights) a linear row may be one 1/256
// weight step off on each axis: within 2 * range / 256, range the
// texels' max - min (3, so 0.0234). Writes its PNGs (values mapped from
// [-2, 2]) and the GPU's and CPU's floats (bound_channels_*.f32, W x H x 4
// little-endian, for a probe); a detail line and the verdict.
bound_channels_compare :: proc(m: ^oksl.Module, block: []u8, gpu_f: [][4]f32, gpu_tag, out_dir: string, scratch: ^oksl.Scratch, coarse_weights: bool) -> (string, bool) {
	cpu_f := make([][4]f32, W * H, context.temp_allocator)
	cimgs := bound_channel_images()
	oksl.run_f32(&m.program, block, W, H, cpu_f, scratch, {channel_images = cimgs[:]})
	lin_tol: f32 = 1e-6
	if coarse_weights {
		lo, hi := f32(max(f32)), f32(min(f32))
		for t in bound_r_texels {
			lo, hi = min(lo, t), max(hi, t)
		}
		for t in bound_rg_texels {
			lo, hi = min(lo, t[0], t[1]), max(hi, t[0], t[1])
		}
		for t in bound_rgba_texels {
			lo, hi = min(lo, t[0], t[1], t[2], t[3]), max(hi, t[0], t[1], t[2], t[3])
		}
		lin_tol = 2 * (hi - lo) / 256 + 1e-6
	}
	near_err, lin_err: f32
	mismatches := 0
	gpu_png := make([]u8, W * H * 4, context.temp_allocator)
	cpu_png := make([]u8, W * H * 4, context.temp_allocator)
	diff_png := make([]u8, W * H * 4, context.temp_allocator)
	for y in 0 ..< H {
		for x in 0 ..< W {
			idx := y * W + x
			gp, cp := gpu_f[idx], cpu_f[idx]
			pd: f32
			for k in 0 ..< 4 {
				d := abs(gp[k] - cp[k])
				pd = max(pd, d)
				if y < 72 {
					near_err = max(near_err, d)
					if !(gp[k] == cp[k]) {
						mismatches += 1
					}
				} else {
					lin_err = max(lin_err, d)
					exact_fill := (x < 64 && k >= 1) || (x >= 64 && x < 128 && k >= 2)
					tol: f32 = 0 if exact_fill else lin_tol
					if !(d <= tol) {
						mismatches += 1
					}
				}
				gpu_png[idx * 4 + k] = u8(clamp(gp[k] * 0.25 + 0.5, 0, 1) * 255 + 0.5)
				cpu_png[idx * 4 + k] = u8(clamp(cp[k] * 0.25 + 0.5, 0, 1) * 255 + 0.5)
			}
			gpu_png[idx * 4 + 3] = 255
			cpu_png[idx * 4 + 3] = 255
			dv := u8(min(pd * 255 * 16, 255))
			diff_png[idx * 4 + 0], diff_png[idx * 4 + 1], diff_png[idx * 4 + 2], diff_png[idx * 4 + 3] = dv, dv, dv, 255
		}
	}
	write_png(fmt.tprintf("%s/bound_channels_%s.png", out_dir, gpu_tag), gpu_png)
	write_png(fmt.tprintf("%s/bound_channels_cpu.png", out_dir), cpu_png)
	write_png(fmt.tprintf("%s/bound_channels_diff.png", out_dir), diff_png)
	_ = os.write_entire_file(fmt.tprintf("%s/bound_channels_%s.f32", out_dir, gpu_tag), slice.to_bytes(gpu_f))
	_ = os.write_entire_file(fmt.tprintf("%s/bound_channels_cpu.f32", out_dir), slice.to_bytes(cpu_f))
	return fmt.tprintf("nearest err %.6f, linear err %.6f (within %.2g), mismatches %d", near_err, lin_err, lin_tol, mismatches), mismatches == 0
}

// ---- a tap of the layer vertex shader ---------------------------------------------------

TAP_PRELUDE :: "in vec2 openPosition;\nout vec4 openClip;\n"
TAP_FRAGMENT :: "layout(location = 1) in vec4 openClip;\nout vec4 c;\nvoid main() { c = vec4(openClip.xy * 0.5 + 0.5, 0.0, 1.0); }"

// The layer vertex shader with one tap, whose twin of position reads uv
// and whose environment is a 1x1 target, so its position spans clip space.
tap_options :: proc(dxbc: bool) -> oksl.Options {
	return {profile = .Glsl, stage = .Vertex, prelude = TAP_PRELUDE, dxbc = dxbc, taps = TAPS}
}

TAP_ATTRIBUTES := [][2]string{{"position", "openPosition"}}
TAPS := []oksl.Tap{{position = "openClip", attributes = TAP_ATTRIBUTES}}

// The tap's vertex block: the W x H target, then the 1x1 environment.
tap_block :: proc(vm: ^oksl.Module) -> []u8 {
	block := make([]u8, vm.block_size, context.temp_allocator)
	oksl.set_uniform(vm, block, "size", {W, H, 0, 0})
	oksl.set_uniform(vm, block[vm.block_stride:], "size", {1, 1, 0, 0})
	return block
}

// The GPU drew the tap's position as colour (attribute position at offset
// 0 of the quad's vertex, its twin at uv's); run_vertex gives it at every
// pixel centre (it is affine in the pixel, so interpolation agrees). The
// largest difference in 8 bits.
tap_compare :: proc(vm: ^oksl.Module, block: []u8, gpu_px: []u8) -> (max_d: int) {
	centres := make([][2][2]f32, W * H, context.temp_allocator)
	for y in 0 ..< H {
		for x in 0 ..< W {
			p := [2]f32{f32(x) + 0.5, f32(y) + 0.5}
			centres[y * W + x] = {p, p / {W, H}}
		}
	}
	cpu_attrs: [3]oksl.Attribute
	for a in vm.stage_inputs {
		cpu_attrs[a.location] = {&centres[0][1] if a.name != "position" else &centres[0][0], size_of([2][2]f32), .Float2}
	}
	positions := make([][4]f32, W * H, context.temp_allocator)
	varyings := make([]f32, W * H * vm.varying_floats, context.temp_allocator)
	at := 0
	for o in vm.stage_outputs {
		if o.name == "openClip" {at = o.offset}
	}
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	oksl.run_vertex(&vm.program, block, cpu_attrs[:], W * H, positions, varyings, &scratch)
	for i in 0 ..< W * H {
		clip := varyings[i * vm.varying_floats + at:]
		for k in 0 ..< 2 {
			want := int(clamp(clip[k] * 0.5 + 0.5, 0, 1) * 255 + 0.5)
			max_d = max(max_d, abs(want - int(gpu_px[i * 4 + k])))
		}
	}
	return
}

// ---- gl_VertexID ------------------------------------------------------------------------

VERTEX_ID_VS :: `#version 310 es
struct Corner { vec4 position; vec4 color; };
readonly buffer Corners { Corner corners[]; };
uniform vec4 size;
out vec4 vColor;
void main() {
    Corner c = corners[gl_VertexID];
    vColor = c.color;
    gl_Position = vec4(2.0 * c.position.x / size.x - 1.0, 1.0 - 2.0 * c.position.y / size.y, 0.0, 1.0);
}`

VERTEX_ID_FS :: "in vec4 vColor;\nout vec4 c;\nvoid main() { c = vColor; }"

Corner :: struct {
	position, color: [4]f32,
}

// The storage buffer of a vertex shader with no attributes: gl_VertexID
// picks the quad's corners and their colours (affine in the pixel) from
// it, the quad starting at corner `first` (the corners before it are
// junk).
vertex_id_corners :: proc(first: int) -> []Corner {
	corners := make([]Corner, first + 6, context.temp_allocator)
	for i in 0 ..< first {
		corners[i] = {{1e6 * f32(1 - 2 * (i % 2)), 1e6, 0, 0}, {1, 0, 1, 1}}
	}
	for v, i in QUAD {
		corners[first + i] = {{v.pos.x, v.pos.y, 0, 1}, {v.pos.x / W, v.pos.y / H, 0.25, 1}}
	}
	return corners
}

vertex_id_block :: proc(vm: ^oksl.Module) -> []u8 {
	block := make([]u8, vm.block_size, context.temp_allocator)
	oksl.set_uniform(vm, block, "size", {W, H, 0, 0})
	return block
}

// run_vertex gives the six vertices from `first` (the GPU drew them from
// its gl_VertexID `first`); the GPU's colours against the CPU's
// interpolated at every pixel centre, the largest difference in 8 bits.
vertex_id_compare :: proc(vm: ^oksl.Module, block: []u8, corners: []Corner, first: int, gpu_px: []u8) -> (max_d: int) {
	positions: [6][4]f32
	record := vm.varying_floats
	varyings := make([]f32, 6 * record, context.temp_allocator)
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	oksl.run_vertex(&vm.program, block, nil, 6, positions[:], varyings, &scratch, {buffers = {slice.reinterpret([]u32, corners)}, first_vertex = first})
	px: [6][2]f32
	for p, i in positions {
		px[i] = {(p.x / p.w + 1) * 0.5 * W, (1 - p.y / p.w) * 0.5 * H}
	}
	for y in 0 ..< H {
		for x in 0 ..< W {
			c := [2]f32{f32(x) + 0.5, f32(y) + 0.5}
			for tri in 0 ..< 2 {
				l, inside := barycentric(px[tri * 3], px[tri * 3 + 1], px[tri * 3 + 2], c)
				if !inside {
					continue
				}
				for k in 0 ..< 4 {
					v := l[0] * varyings[(tri * 3) * record + k] + l[1] * varyings[(tri * 3 + 1) * record + k] + l[2] * varyings[(tri * 3 + 2) * record + k]
					want := int(clamp(v, 0, 1) * 255 + 0.5)
					max_d = max(max_d, abs(want - int(gpu_px[(y * W + x) * 4 + k])))
				}
				break
			}
		}
	}
	return
}

// Screen-space barycentric coordinates of `p` in triangle abc, and whether
// it is inside (with a 1e-4 margin).
barycentric :: proc(a, b, d, p: [2]f32) -> (l: [3]f32, inside: bool) {
	area := (b.x - a.x) * (d.y - a.y) - (d.x - a.x) * (b.y - a.y)
	l[1] = ((p.x - a.x) * (d.y - a.y) - (d.x - a.x) * (p.y - a.y)) / area
	l[2] = ((b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y)) / area
	l[0] = 1 - l[1] - l[2]
	return l, l[0] >= -1e-4 && l[1] >= -1e-4 && l[2] >= -1e-4
}

// ---- gl_SampleMask ----------------------------------------------------------------------

SAMPLE_MASK_FS :: `#version 300 es
precision highp float;
out vec4 c;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) / 8;
    gl_SampleMask[0] = (p.x + 3 * p.y) & 15;
    c = vec4(1.0, 0.5, 0.25, 1.0);
}`

// A fragment shader keeping some of a 4x MSAA pixel's samples, resolved
// over black: each pixel is its colour times the kept samples / 4. The
// GPU's against run_fragment's colours and masks over the target's quads,
// the largest difference in 8 bits.
sample_mask_compare :: proc(m: ^oksl.Module, gpu_px: []u8) -> (max_d: int) {
	Q :: (W / 2) * (H / 2)
	origin := make([][2]i32, Q, context.temp_allocator)
	for q in 0 ..< Q {
		origin[q] = {i32(q % (W / 2) * 2), i32(q / (W / 2) * 2)}
	}
	out := make([][4]f32, Q * 4, context.temp_allocator)
	shaded := make([]oksl.Quad_Lanes, Q, context.temp_allocator)
	masks := make([]u32, Q * 4, context.temp_allocator)
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	block := make([]u8, m.block_size, context.temp_allocator)
	targets := [1][][4]f32{out}
	oksl.run_fragment(&m.program, block, {count = Q, origin = origin}, targets[:], shaded, &scratch, {}, masks)
	for q in 0 ..< Q {
		for i in 0 ..< 4 {
			x, y := int(origin[q].x) + (i & 1), int(origin[q].y) + (i >> 1)
			kept := f32(card(transmute(bit_set[0 ..< 32;u32])(masks[q * 4 + i] & 15))) / 4
			for k in 0 ..< 4 {
				// Each sample stores the colour in 8 bits; the resolve
				// averages them.
				want := int(f32(int(clamp(out[q * 4 + i][k], 0, 1) * 255 + 0.5)) * kept + 0.5)
				max_d = max(max_d, abs(want - int(gpu_px[(y * W + x) * 4 + k])))
			}
		}
	}
	return
}

// ---- facing, perspective and flat integers ----------------------------------------------

// Two triangles in clip space with w != 1 and z != 0, drawn without
// culling and without vertex buffers: the left one counter-clockwise on
// screen (front-facing under SDL_GPU's default), the right one clockwise.
// Each passes a flat ivec2 (negative, its own per triangle) to the
// fragment shader, which writes gl_FrontFacing (1 or -1),
// gl_FragCoord.z, gl_FragCoord.w and the integers (exact as a float) to
// a float target.
FACING_VS :: `#version 310 es
layout(location = 0) flat out ivec2 vId;
void main() {
    vec4 p[6] = vec4[6](
        vec4(-0.9, 0.8, 0.2, 1.0), vec4(-1.8, -1.6, 1.0, 2.0), vec4(-0.4, -3.2, 3.2, 4.0),
        vec4(0.3, 2.4, 0.9, 3.0), vec4(1.35, -1.2, 0.9, 1.5), vec4(0.1, -0.8, 0.1, 1.0));
    gl_Position = p[gl_VertexID];
    int tri = gl_VertexID / 3;
    vId = ivec2(tri * 1000 - 7, -3001 * (tri + 1));
}`

FACING_FS :: `#version 310 es
precision highp float;
layout(location = 0) flat in ivec2 vId;
out vec4 c;
void main() {
    c = vec4(gl_FrontFacing ? 1.0 : -1.0, gl_FragCoord.z, gl_FragCoord.w, float(vId.x) + 2048.0 * float(vId.y));
}`

// The expected pixels from run_vertex's positions and flat integers
// (rasterized here: the CPU's run_f32 has gl_FragCoord.w = 1 and every
// pixel front-facing): front where the triangle is counter-clockwise on
// screen, z the screen-linear z / w, w the screen-linear 1 / w. Pixels
// within 1e-3 of an edge are skipped; the others outside both are the
// target's clear, (0, 0, 0, 1). z and w within 1e-4 relative, the rest exact.
facing_compare :: proc(vm: ^oksl.Module, gpu_f: [][4]f32) -> (string, bool) {
	positions: [6][4]f32
	record := vm.varying_floats
	varyings := make([]f32, 6 * record, context.temp_allocator)
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	oksl.run_vertex(&vm.program, nil, nil, 6, positions[:], varyings, &scratch)
	at := vm.stage_outputs[0].offset
	px: [6][2]f32
	for p, i in positions {
		px[i] = {(p.x / p.w + 1) * 0.5 * W, (1 - p.y / p.w) * 0.5 * H}
	}
	bad, checked, fronts, backs := 0, 0, 0, 0
	worst: f32
	for y in 0 ..< H {
		for x in 0 ..< W {
			c := [2]f32{f32(x) + 0.5, f32(y) + 0.5}
			// Outside both: the target's clear.
			want := [4]f32{0, 0, 0, 1}
			edge := false
			for tri in 0 ..< 2 {
				a, b, d := px[tri * 3], px[tri * 3 + 1], px[tri * 3 + 2]
				l, _ := barycentric(a, b, d, c)
				if l[0] < -1e-3 || l[1] < -1e-3 || l[2] < -1e-3 {
					continue
				}
				if l[0] < 1e-3 || l[1] < 1e-3 || l[2] < 1e-3 {
					edge = true
					break
				}
				// y down: a negative area is counter-clockwise on screen.
				area := (b.x - a.x) * (d.y - a.y) - (d.x - a.x) * (b.y - a.y)
				front := area < 0
				z, rw: f32
				for k in 0 ..< 3 {
					p := positions[tri * 3 + k]
					z += l[k] * p.z / p.w
					rw += l[k] / p.w
				}
				ids := transmute([2]i32)[2]f32{varyings[tri * 3 * record + at], varyings[tri * 3 * record + at + 1]}
				want = {1 if front else -1, z, rw, f32(ids.x) + 2048 * f32(ids.y)}
				if front {fronts += 1} else {backs += 1}
			}
			if edge {
				continue
			}
			checked += 1
			g := gpu_f[y * W + x]
			ok := g.x == want.x && g.w == want.w
			for k in 1 ..= 2 {
				d := abs(g[k] - want[k])
				ok &&= d <= 1e-4 * abs(want[k])
				if want.x != 0 {
					worst = max(worst, d / abs(want[k]))
				}
			}
			if !ok {
				if bad < 4 {
					fmt.printfln("  facing (%d, %d): gpu %v, want %v", x, y, g, want)
				}
				bad += 1
			}
		}
	}
	return fmt.tprintf("%d pixels (%d front, %d back), z and w worst %.3g relative, mismatches %d", checked, fronts, backs, worst, bad), bad == 0 && fronts > 0 && backs > 0
}

// ---- compute ----------------------------------------------------------------------------

// A compute check's shape: read-write images rw_w x rw_h (read-only ones
// twice that, so non-multiples of 8 exercise partial edge workgroups),
// `buf_words` words per storage buffer, and a grid of exactly the
// read-write image (else 3 more invocations each way, and at least 2x2).
Compute_Shape :: struct {
	rw_w, rw_h, buf_words: int,
	exact_groups:          bool,
}

// image_war.comp needs >= 128x128 to catch a missing fence (gpu_parity
// on Metal).
compute_shape :: proc(name: string) -> Compute_Shape {
	if name == "image_war.comp" {
		return {128, 128, 16384, true}
	}
	return {29, 17, 4096, false}
}

// A compute check's data, in the temporary allocator: the uniform block,
// the grid, every storage buffer's and image's first contents (what the
// GPU uploads, in Module.storage_buffers and .storage_images order) and the
// CPU's copies (run_compute writes them).
Compute_Data :: struct {
	block:       []u8,
	groups:      [3]int,
	buf_words:   int,
	// Which words of a buffer's element are floats (compared within 1e-4).
	float_words: [][]bool,
	init_bufs:   [][]u32,
	cpu_bufs:    [][]u32,
	init_imgs:   [][]u8,
	cpu_imgs:    []oksl.Storage_Image,
}

compute_data :: proc(m: ^oksl.Module, shape: Compute_Shape) -> (cd: Compute_Data, err: string) {
	rw_w, rw_h := shape.rw_w, shape.rw_h
	ro_w, ro_h := rw_w * 2, rw_h * 2
	y_stride := (ro_w + 7) &~ 7
	tx, ty := int(m.threadcount.x), int(m.threadcount.y)
	extra := 0 if shape.exact_groups else 3
	cd.groups = {max((rw_w + extra + tx - 1) / tx, 2), max((rw_h + extra + ty - 1) / ty, 2), 3 if m.threadcount.z > 1 else 1}
	cd.buf_words = shape.buf_words

	cd.block = make([]u8, m.block_size, context.temp_allocator)
	oksl.set_uniform(m, cd.block, "uScale", {1.25})
	oksl.set_uniform(m, cd.block, "uDt", {0.25})
	oksl.set_uniform(m, cd.block, "uCount", {500})
	oksl.set_uniform(m, cd.block, "uMatArr", {1, 2, 3, 4}, 0)
	oksl.set_uniform(m, cd.block, "uMatArr", {5, 6, 7, 8}, 1)
	oksl.set_uniform(m, cd.block, "ky", {13933, 46871, 4732, 1081344})
	oksl.set_uniform(m, cd.block, "ku", {-7509, -25259, 32768, 33685504})
	oksl.set_uniform(m, cd.block, "kv", {32768, -29763, -3005, 255})
	oksl.set_uniform(m, cd.block, "size", {f32(ro_w), f32(ro_h), f32(y_stride), f32(y_stride * ro_h)})
	// Options.num_work_groups_uniform's group count (the GPU's; the CPU
	// reads groups).
	if m.num_work_groups_offset >= 0 {
		counts := [3]u32{u32(cd.groups.x), u32(cd.groups.y), u32(cd.groups.z)}
		copy(cd.block[m.num_work_groups_offset:], slice.to_bytes(counts[:]))
	}

	nb, ni := len(m.storage_buffers), len(m.storage_images)
	cd.float_words = make([][]bool, nb, context.temp_allocator)
	cd.init_bufs = make([][]u32, nb, context.temp_allocator)
	cd.cpu_bufs = make([][]u32, nb, context.temp_allocator)
	for sb, bi in m.storage_buffers {
		stride_words := max(sb.stride / 4, 1)
		fw := make([]bool, stride_words, context.temp_allocator)
		if !mark_float_words(m, sb.element, 0, fw) {
			return cd, fmt.tprintf("missing struct reflection for buffer %s", sb.name)
		}
		init_w := make([]u32, shape.buf_words, context.temp_allocator)
		for &w, i in init_w {
			if !fw[i % stride_words] {
				w = 0xdead_0000 | u32((bi * 31 + i * 17) & 0xffff)
			} else {
				w = transmute(u32)(f32(((bi * 29 + i * 37 + 11) % 64) + 8) * 0.25)
			}
		}
		cd.float_words[bi] = fw
		cd.init_bufs[bi] = init_w
		cd.cpu_bufs[bi] = slice.clone(init_w, context.temp_allocator)
	}
	cd.init_imgs = make([][]u8, ni, context.temp_allocator)
	cd.cpu_imgs = make([]oksl.Storage_Image, ni, context.temp_allocator)
	for im, ii in m.storage_images {
		iw := ro_w if im.readonly else rw_w
		ih := ro_h if im.readonly else rw_h
		init_d := make([]u8, iw * ih * image_texel_bytes(im.format), context.temp_allocator)
		switch im.format {
		case .None:
		case .Rgba8:
			for &b, i in init_d {
				b = u8((ii * 53 + i * 41 + 19) & 255)
			}
		case .Rgba16f:
			for &h, i in slice.reinterpret([]u16, init_d) {
				h = u16(0x3800 + ((ii * 31 + i * 19) & 0x07ff))
			}
		case .Rgba32f, .R32f:
			for &f, i in slice.reinterpret([]f32, init_d) {
				f = f32((ii * 23 + i * 43 + 7) % 128) / 64.0
			}
		case .R32ui:
			for &v, i in slice.reinterpret([]u32, init_d) {
				v = u32((ii * 13 + i * 17 + 5) & 1023)
			}
		case .R32i:
			for &v, i in slice.reinterpret([]i32, init_d) {
				v = i32((ii * 13 + i * 19) % 511) - 255
			}
		}
		cd.init_imgs[ii] = init_d
		cd.cpu_imgs[ii] = {data = slice.clone(init_d, context.temp_allocator), width = iw, height = ih, format = im.format}
	}
	return cd, ""
}

image_texel_bytes :: proc(f: oksl.Image_Format) -> int {
	switch f {
	case .None:
		return 0
	case .Rgba8, .R32f, .R32ui, .R32i:
		return 4
	case .Rgba16f:
		return 8
	case .Rgba32f:
		return 16
	}
	return 0
}

mark_float_words :: proc(m: ^oksl.Module, t: oksl.Type, byte_off: int, float_words: []bool) -> bool {
	if t.base == .Struct {
		if t.strukt < 0 || int(t.strukt) >= len(m.structs) || len(m.structs[t.strukt].fields) == 0 {
			return false
		}
		for f in m.structs[t.strukt].fields {
			if !mark_float_words(m, f.type, byte_off + f.offset, float_words) {
				return false
			}
		}
		return true
	}
	if t.base == .Float {
		words := int(t.rows) * max(int(t.cols), 1)
		for k in 0 ..< words {
			idx := byte_off / 4 + k
			if idx >= 0 && idx < len(float_words) {
				float_words[idx] = true
			}
		}
	}
	return true
}

@(private = "file")
Range_Job :: struct {
	program: ^oksl.Program,
	block:   []u8,
	groups:  [3]int,
	opts:    oksl.Run_Options,
	first:   int,
	count:   int,
}

// Runs the check on the CPU, split across three concurrent OS threads (each
// with its own Scratch), into cd.cpu_bufs and cd.cpu_imgs.
compute_cpu :: proc(m: ^oksl.Module, cd: ^Compute_Data) {
	sizes: [oksl.SAMPLER_COUNT][3]i32
	for &s, ch in sizes {
		s = {TEX, TEX, LAYERS if ch in m.channel_arrays else 1}
	}
	arrays := m.channel_arrays
	opts := oksl.Run_Options{sampler = cpu_sampler, user = &arrays, channel_sizes = sizes[:], buffers = cd.cpu_bufs, images = cd.cpu_imgs}
	total := cd.groups.x * cd.groups.y * cd.groups.z
	third := total / 3
	jobs := [3]Range_Job {
		{program = &m.program, block = cd.block, groups = cd.groups, opts = opts, first = 0, count = third},
		{program = &m.program, block = cd.block, groups = cd.groups, opts = opts, first = third, count = third},
		{program = &m.program, block = cd.block, groups = cd.groups, opts = opts, first = third * 2, count = -1},
	}
	threads: [3]^thread.Thread
	for &th, ti in threads {
		th = thread.create_and_start_with_poly_data(&jobs[ti], proc(job: ^Range_Job) {
			s: oksl.Scratch
			defer oksl.scratch_destroy(&s)
			oksl.run_compute(job.program, job.block, job.groups, &s, job.opts, job.first, job.count)
		})
	}
	for th in threads {
		thread.destroy(th)
	}
}

parity_f16_to_f32 :: proc(h: u16) -> f32 {
	sign := u32(h & 0x8000) << 16
	exp := u32((h >> 10) & 0x1f)
	mant := u32(h & 0x03ff)
	if exp == 0 {
		if mant == 0 {
			return transmute(f32)sign
		}
		e := u32(113)
		for (mant & 0x0400) == 0 {
			mant <<= 1
			e -= 1
		}
		return transmute(f32)(sign | (e << 23) | ((mant & 0x03ff) << 13))
	}
	if exp == 31 {
		return transmute(f32)(sign | 0x7f80_0000 | (mant << 13))
	}
	return transmute(f32)(sign | ((exp + 112) << 23) | (mant << 13))
}

// Compares what the GPU wrote (`gpu_bufs`: the read-write storage buffers'
// bytes, `gpu_imgs`: the read-write images' tightly packed rows, each in
// binding order) with the CPU's: integer words, r32ui and r32i texels
// exactly, float words and rgba32f / r32f texels within 1e-4 (0.02 when the
// shader samples, whose filtering differs), rgba16f within 2e-3 or one ulp
// and rgba8 within 1 of 255. Atomics' results the testdata shaders fold so they do
// not depend on order (atomics.comp, histogram.comp).
compute_compare :: proc(m: ^oksl.Module, cd: ^Compute_Data, gpu_bufs, gpu_imgs: [][]u8) -> (string, bool) {
	ro_buf, ro_tex := m.num_readonly_storage_buffers, m.num_readonly_storage_textures
	max_f_err: f32 = 0
	mismatches := 0
	// Prints the first 4 mismatches.
	report :: proc(mismatches: ^int, what: string, i, k: int, gpu, cpu: u32) {
		if mismatches^ < 4 {
			fmt.printfln("  %s %d [%d]: gpu %x, cpu %x", what, i, k, gpu, cpu)
		}
		mismatches^ += 1
	}
	for gb, i in gpu_bufs {
		bi := ro_buf + i
		fw := cd.float_words[bi]
		gpu_w := slice.reinterpret([]u32, gb)
		cpu_w := cd.cpu_bufs[bi]
		for w, k in cpu_w {
			if gpu_w[k] == w {
				continue
			}
			if !fw[k % len(fw)] {
				report(&mismatches, "buffer", bi, k, gpu_w[k], w)
				continue
			}
			if same_float(transmute(f32)gpu_w[k], transmute(f32)w) {
				continue
			}
			d := abs(transmute(f32)gpu_w[k] - transmute(f32)w)
			max_f_err = max(max_f_err, d)
			if !(d <= 1e-4) {
				report(&mismatches, "buffer", bi, k, gpu_w[k], w)
			}
		}
	}
	for gpu_b, i in gpu_imgs {
		ii := ro_tex + i
		cpu_b := cd.cpu_imgs[ii].data
		switch m.storage_images[ii].format {
		case .None:
		case .R32ui, .R32i:
			gw, cw := slice.reinterpret([]u32, gpu_b), slice.reinterpret([]u32, cpu_b)
			for k in 0 ..< len(gw) {
				if gw[k] != cw[k] {
					report(&mismatches, "image", ii, k, gw[k], cw[k])
				}
			}
		case .Rgba8:
			for k in 0 ..< len(gpu_b) {
				d := abs(int(gpu_b[k]) - int(cpu_b[k]))
				max_f_err = max(max_f_err, f32(d) / 255.0)
				if d > 1 {
					report(&mismatches, "image", ii, k, u32(gpu_b[k]), u32(cpu_b[k]))
				}
			}
		case .Rgba16f:
			gh, ch := slice.reinterpret([]u16, gpu_b), slice.reinterpret([]u16, cpu_b)
			for k in 0 ..< len(gh) {
				gf, cf := parity_f16_to_f32(gh[k]), parity_f16_to_f32(ch[k])
				if same_float(gf, cf) {
					continue
				}
				d := abs(gf - cf)
				max_f_err = max(max_f_err, d)
				// One half-float ulp (2^-10 relative) either way: the CPU
				// rounds to nearest, D3D's conversion toward zero.
				if !(d <= 2e-3 || d <= abs(cf) / 1024) {
					report(&mismatches, "image", ii, k, u32(gh[k]), u32(ch[k]))
				}
			}
		case .Rgba32f, .R32f:
			gf, cf := slice.reinterpret([]f32, gpu_b), slice.reinterpret([]f32, cpu_b)
			tol: f32 = 0.02 if m.num_samplers > 0 else 1e-4
			for k in 0 ..< len(gf) {
				if same_float(gf[k], cf[k]) {
					continue
				}
				d := abs(gf[k] - cf[k])
				max_f_err = max(max_f_err, d)
				if !(d <= tol) {
					report(&mismatches, "image", ii, k, transmute(u32)gf[k], transmute(u32)cf[k])
				}
			}
		}
	}
	return fmt.tprintf("max float err %.6f, mismatches %d", max_f_err, mismatches), mismatches == 0
}

// Equal (infinities of a sign, either zero) or both NaN: before any
// difference, which inf - inf would make NaN.
same_float :: proc(a, b: f32) -> bool {
	return a == b || math.is_nan(a) && math.is_nan(b)
}

// Every storage image format, read-only and read-write, and a single
// channel format's missing components ((0, 0, 1)).
IMAGE_FORMATS_CS :: `#version 310 es
layout(local_size_x = 8, local_size_y = 4) in;

layout(r32f) readonly uniform image2D inF;
layout(r32i) readonly uniform iimage2D inI;
layout(r32f) uniform image2D rwF;
layout(r32i) uniform iimage2D rwI;
layout(rgba8) writeonly uniform image2D out8;
writeonly buffer Checks { uint ok_bits[]; } checks;

void main() {
    ivec2 sz = imageSize(out8);
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= sz.x || p.y >= sz.y) {
        return;
    }
    vec4 vf = imageLoad(inF, p);
    ivec4 vi = imageLoad(inI, p);
    vec4 rf = imageLoad(rwF, p);
    ivec4 ri = imageLoad(rwI, p);
    imageStore(rwF, p, vec4(rf.r + vf.r * 0.5 + vf.a, 9.0, 9.0, 9.0));
    imageStore(rwI, p, ivec4(ri.r - vi.r + vi.a * 7, 3, 4, 5));
    uint sub_ok = (vf.g == 0.0 && vf.b == 0.0 && vf.a == 1.0 && vi.g == 0 && vi.b == 0 && vi.a == 1) ? 1u : 0u;
    checks.ok_bits[p.y * sz.x + p.x] = sub_ok * 100u + uint(vi.a + ri.a);
    imageStore(out8, p, vec4(clamp(vf.r * 0.25, 0.0, 1.0), vf.g, vf.b, vf.a * 0.5));
}`
