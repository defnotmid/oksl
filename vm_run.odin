package oksl

import "base:intrinsics"
import "core:math"
import "core:math/linalg"

// CPU backend, part 2: the interpreter. Pixels run in batches of LANES:
// 2x2 quads in a BATCH_W x BATCH_H block, so derivatives use the quad
// neighbours like a GPU's fine derivatives, and a block of neighbouring
// pixels diverges little.

// Samples channel `channel` at normalized `uv` (origin top-left) and mip
// level `lod`; returns RGBA. The level texture() picks is fractional and
// may be below 0 (magnified, -inf where uv is constant): clamp it to the
// levels the channel has. `layer` is a sampler2DArray's, its coordinate
// rounded to nearest even (0 for a sampler2D): clamp it to the layers.
Sampler_Proc :: #type proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32

Run_Options :: struct {
	sampler:  Sampler_Proc,
	user:     rawptr,
	// Region to shade in layer pixels, clipped to the layer; zero means
	// the whole layer. Its pixels come out as a whole-layer run's do.
	x, y:           int,
	w, h:           int,
	// When set on run_f32 or run_targets, each non-nil target is sized to
	// the region (len >= w * h) rather than the whole layer, and pixel
	// (px, py) is written to target[(py - y) * w + (px - x)]. Requires
	// w > 0, h > 0 and the region inside 0..width x 0..height; the shader
	// still sees full-layer coordinates, uv, varyings and derivatives.
	region_targets: bool,
	// The stage inputs by location. Absent ones are zero, except location
	// 0, which is the layer's uv (what the layer vertex shader writes).
	varyings: []Varying,
	// Each channel's size in texels (level 0) and its layers (a
	// sampler2DArray's; ignored for a sampler2D): textureSize's in GLSL
	// shaders (Shadertoy's is iChannelResolution), the level texture() and
	// textureGrad pick, and in every profile, Shadertoy's too, the texels
	// texelFetch reads: the sampler is called only for a texel, level and
	// layer inside them (a full mip chain of w x h, level l max(w >> l, 1)
	// x max(h >> l, 1) while max(w, h) >> l is not 0, or channel_levels'
	// levels of it), and (0, 0, 0, 0) is read anywhere else. A channel
	// without one is 0x0: texture() samples it at its bias, and texelFetch
	// reads 0.
	channel_sizes: [][3]i32,
	// Each channel's mip levels for texelFetch: n > 0 means levels 0 ..< n
	// of channel_sizes' chain (n at most 16, as a Channel_Image's levels,
	// ensured once per run), so
	// texelFetch reads 0 at a level from n on without calling the sampler;
	// 0, or a channel past the end, the full chain. Set it to the GPU
	// texture's num_levels for the CPU to match the GPU: a texture of one
	// level reads 0 there at every lod above 0, where the full chain calls
	// the sampler. texture(), textureLod and textureGrad pass the sampler
	// their level whatever it is, and a bound image has its own levels.
	channel_levels: []i32,
	// run_vertex: gl_VertexID of its first vertex (vertex i's is
	// first_vertex + i), as a draw's first vertex counts on the GPU.
	first_vertex: int,
	// The storage buffers' words, in SDL_GPU binding order
	// (Module.storage_buffers: read-only first, then read-write/writeonly).
	// A read past a buffer's words gives 0 here and a write past them is
	// dropped; on the GPU out-of-bounds access is undefined (Metal reads past
	// the buffer).
	buffers: [][]u32,
	// The storage images for run_compute, in SDL_GPU binding order
	// (Module.storage_images: read-only first, then read-write/writeonly).
	// Out-of-bounds loads read 0 and out-of-bounds stores are dropped.
	images:         []Storage_Image,
	// Optional 2D images bound to sampled channels, indexed by channel
	// (Channel_Image); the others, and entries without texels, call
	// `sampler`.
	channel_images: []Channel_Image,
}

// How a bound channel filters, as GPU texture units do (measured against
// RADV with tools/gpu_parity's bound images). The coordinate is scaled to
// the level's texels exactly (`x = f64(u) * f64(n)`, minus 0.5 for Linear):
//   Nearest: the texel at floor(x);
//   Linear:  the four around it, blended by each axis's fraction `t`
//            rounded to 1/256 (to nearest even: 257 steps, as GPU
//            filtering's 8 bits), in f32 as `a + (b - a) * t`: the row
//            at floor(y) across x, the next row across x, then the two
//            across y.
// Both blend the texels' stored values (bytes and u16 as integers, halves
// and floats as they are); the unsigned normalized formats are divided by
// 255 or 65535 last. This is what the GPU computes to within its last
// bits.
Channel_Filter :: enum u8 {
	Nearest,
	Linear,
}

// A bound channel's texels. F32 takes `components` floats a texel from
// `data`; the others read `pixels`: Rgba8 and Gray8 bytes and Gray16
// little-endian u16 as unsigned normalized (`/ 255`, `/ 65535`), Rgba16f
// IEEE half floats. Missing components read (0, 0, 0, 1): Gray8 and Gray16
// are (g, 0, 0, 1).
Channel_Format :: enum u8 {
	F32,
	Rgba8,
	Gray8,
	Gray16,
	Rgba16f,
}

// How a bound channel addresses one axis, from the texel index `i =
// floor(x)` (NaN as -2^31, within [-2^31, 2^31 - 2]) on an axis of `n`:
//   Clamp:  clamp(i, 0, n - 1), and i + 1 likewise;
//   Repeat: i mod n, floored, and the next one (0 after n - 1).
Channel_Wrap :: enum u8 {
	Clamp,
	Repeat,
}

// The most levels a bound channel has: level 0 and 15 more.
CHANNEL_MAX_LEVELS :: 16

// One mip level above level 0, in the image's format: `max(width >> l, 1)
// x max(height >> l, 1)` texels for level l, rows `stride` bytes apart (0:
// tightly packed), and with layers each layer `layer_stride` bytes after
// the one before (0: right after it, `height` rows on).
Channel_Level :: struct {
	texels:        []u8,
	width, height: int,
	stride:        int,
	layer_stride:  int,
}

// A 2D image bound to a sampled channel on the CPU: `width x height`
// texels (level 0) of `format`, rows `stride` bytes apart (0: tightly
// packed), each axis addressed by `wrap` ({u, v}), filtered by `filter`,
// with the mip levels above level 0 in `levels`. An entry with neither
// `data` nor `pixels` leaves the channel unbound so `Run_Options.sampler`
// is used. Coordinates address it as the GPU addresses a texture: uv (0, 0)
// is the first texel of the first row.
//
// The level: texture() takes the bias plus the level the quad's
// derivatives pick in level 0's texels (as it passes the sampler),
// textureLod and textureGrad theirs. With more than one level, a level
// `lod` that is not above 0 (NaN included) reads level 0; otherwise `l =
// min(lod, levels - 1)` reads level `l0 = trunc(l)`, blended toward level
// `l0 + 1` (when there is one) as `c0 + (c1 - c0) * w / 256`, `w` the
// fraction `l - l0` rounded to 1/256 (to nearest even) as GPUs blend
// levels, where w > 0.
// `texelFetch` reads its level's texel, (0, 0, 0, 0) out of range;
// `textureSize` gives level `clamp(lod, 0, levels - 1)`'s size.
//
// Layers: a sampler2DArray channel takes `layers` of them (0 or 1: one),
// each the same size, levels, format and wrap, layer i of each level
// `i * layer_stride` bytes after its first (0: right after the layer
// before, `height` rows on; Channel_Level.layer_stride for the levels
// above). A sample reads layer `clamp(round_even(r), 0, layers - 1)` for
// its coordinate r (NaN as 0, as RADV takes it); texelFetch reads layer
// p.z, (0, 0, 0, 0) outside [0, layers); textureSize's z is `layers`. A
// sampler2D channel takes one layer.
Channel_Image :: struct {
	data:          []f32,
	width, height: int,
	components:    int,
	filter:        Channel_Filter,
	format:        Channel_Format,
	pixels:        []u8,
	stride:        int,
	wrap:          [2]Channel_Wrap,
	levels:        []Channel_Level,
	layers:        int,
	layer_stride:  int,
}

// A single-level F32 image from tightly packed 1-, 2- or 4-component
// texels, both axes clamped to the edge.
channel_image :: proc "contextless" (texels: []$T, width, height: int, filter := Channel_Filter.Nearest) -> Channel_Image
	where T == f32 || T == [2]f32 || T == [4]f32 {
	N :: size_of(T) / size_of(f32)
	return {
		data       = ([^]f32)(raw_data(texels))[:len(texels) * N],
		width      = width,
		height     = height,
		components = N,
		filter     = filter,
	}
}

// A mip level from its texels (any element type: `u8`, `[4]u8`, `u16`,
// `[4]f16`, `f32`, `[2]f32`, `[4]f32`, ...), tightly packed.
channel_level :: proc "contextless" (texels: []$T, width, height: int) -> Channel_Level {
	return {texels = ([^]u8)(raw_data(texels))[:len(texels) * size_of(T)], width = width, height = height}
}

// The levels a bound image of `width x height` can have, level 0
// included: its whole chain down to 1x1, at most CHANNEL_MAX_LEVELS.
channel_level_count :: proc "contextless" (width, height: int) -> int {
	n := 1
	for w, h := width, height; (w > 1 || h > 1) && n < CHANNEL_MAX_LEVELS; n += 1 {
		w, h = max(w >> 1, 1), max(h >> 1, 1)
	}
	return n
}

// The bytes levels 1 ..< `count` of a `width x height` image of `format`
// (`components` floats a texel for F32: 1, 2 or 4) take tightly packed,
// each with `layers` (0 or 1: one): what channel_levels_layout lays out.
// `count` is at most CHANNEL_MAX_LEVELS.
channel_levels_bytes :: proc(format: Channel_Format, components, width, height, count, layers: int) -> int {
	ensure(format >= .F32 && format <= .Rgba16f, "oksl: channel_levels_bytes: invalid Channel_Format")
	ensure(format != .F32 || components == 1 || components == 2 || components == 4, "oksl: channel_levels_bytes: F32 components must be 1, 2 or 4")
	ensure(count <= CHANNEL_MAX_LEVELS, "oksl: channel_levels_bytes: more than 16 levels")
	tb := channel_texel_bytes(format, components if format == .F32 else (4 if format == .Rgba8 || format == .Rgba16f else 1))
	total := 0
	for l in 1 ..< count {
		total += max(width >> uint(l), 1) * max(height >> uint(l), 1) * tb * max(layers, 1)
	}
	return total
}

// Lays levels 1 ..< len(levels) + 1 out in `texels` (caller-owned, at
// least channel_levels_bytes long, of any element type, aligned as the
// format reads: 4 bytes for F32, 2 for Gray16 and Rgba16f), tightly packed
// one after the other, each level's layers one after the other. Allocating
// `texels` as `[]f32`, `[]u16` or `[][4]f16` aligns it on any allocator.
channel_levels_layout :: proc(levels: []Channel_Level, texels: []$T, format: Channel_Format, components, width, height, layers: int) {
	bytes := ([^]u8)(raw_data(texels))[:len(texels) * size_of(T)]
	ensure(len(bytes) >= channel_levels_bytes(format, components, width, height, len(levels) + 1, layers), "oksl: channel_levels_layout texels are shorter than the levels")
	align := uintptr(4 if format == .F32 else (2 if format == .Gray16 || format == .Rgba16f else 1))
	ensure(uintptr(raw_data(bytes)) % align == 0, "oksl: channel_levels_layout texels must be 4-byte aligned for F32, 2-byte for Gray16 and Rgba16f")
	tb := channel_texel_bytes(format, components if format == .F32 else (4 if format == .Rgba8 || format == .Rgba16f else 1))
	at := 0
	for &lv, i in levels {
		l := uint(i + 1)
		w, h := max(width >> l, 1), max(height >> l, 1)
		n := w * h * tb * max(layers, 1)
		lv = {texels = bytes[at:][:n], width = w, height = h}
		at += n
	}
}

// Fills `im.levels` (caller-owned, sized as Channel_Image requires, e.g.
// by channel_levels_layout) from level 0, each level from the one above
// and each layer on its own, as SDL_GPU's GenerateMipmapsForGPUTexture's
// linear blit does (measured against RADV, tools/gpu_parity's
// built_levels): texel (x, y) of a `w x h` level reads its `sw x sh`
// parent at the centre's coordinate `(x + 0.5) * sw / w - 0.5` (exactly;
// likewise y), the four texels around it clamped to the edge, blended by
// each axis's fraction rounded to 1/256 (to nearest even, as the
// sampling rule's Linear) in exact arithmetic on the texels as the GPU
// reads them (bytes / 255, u16 / 65535, halves and floats) and rounded
// once to f32, then stored: F32 as it is, Rgba16f to the nearest half
// (even), the unsigned normalized formats `round_even(v * 255)` (65535).
// Even sizes average each 2x2 block; odd ones spread over the whole
// level. Allocates nothing.
channel_build_levels :: proc(im: Channel_Image) {
	comps, tb := validate_channel_image(im)
	layers := max(im.layers, 1)
	src := Build_Level{raw_data(im.data) if im.format == .F32 else raw_data(im.pixels), im.width, im.height, im.stride if im.stride != 0 else im.width * tb, im.layer_stride}
	if src.lstride == 0 {
		src.lstride = src.pitch * im.height
	}
	for lv in im.levels {
		dst := Build_Level{raw_data(lv.texels), lv.width, lv.height, lv.stride if lv.stride != 0 else lv.width * tb, lv.layer_stride}
		if dst.lstride == 0 {
			dst.lstride = dst.pitch * lv.height
		}
		for layer in 0 ..< layers {
			s, d := src, dst
			s.ptr = &([^]u8)(src.ptr)[layer * src.lstride]
			d.ptr = &([^]u8)(dst.ptr)[layer * dst.lstride]
			switch im.format {
			case .F32:
				switch comps {
				case 1: build_level(s, d, .F32, 1)
				case 2: build_level(s, d, .F32, 2)
				case:   build_level(s, d, .F32, 4)
				}
			case .Rgba8:   build_level(s, d, .Rgba8, 4)
			case .Gray8:   build_level(s, d, .Gray8, 1)
			case .Gray16:  build_level(s, d, .Gray16, 1)
			case .Rgba16f: build_level(s, d, .Rgba16f, 4)
			}
		}
		src = dst
	}
}

// One level (one layer) for channel_build_levels: its first texel, size,
// row pitch and layer stride in bytes.
@(private)
Build_Level :: struct {
	ptr:      rawptr,
	w, h:     int,
	pitch:    int,
	lstride:  int,
}

// Where texel d of an axis of n reads its parent's sn: the two texels and
// the weight toward the second, w / 256.
@(private)
Build_Tap :: struct {
	i0, i1: i32,
	t:      f64,
}

// In integers, exactly: the centre is x = num / 2n with num = (2d + 1) *
// sn - n, its texel i = floor(x) and its weight q = round_even(128 * rem
// / n) for the remainder rem in [0, 2n); a weight of 256 is the next
// texel's 0. Level sizes keep sn >= n (sn is 2n, 2n + 1, or n = 1), so
// num >= sn - n >= 0 and truncating division is floor; i64 holds (2d +
// 1) * sn for sides below 2^31. x < sn - 1/2, so only the second texel
// reaches past the edge.
@(private)
build_tap :: #force_inline proc "contextless" (d, sn, n: int) -> Build_Tap {
	n2 := 2 * i64(n)
	num := (2 * i64(d) + 1) * i64(sn) - i64(n)
	i := num / n2
	rem := num - i * n2
	q, r := 128 * rem / i64(n), 128 * rem % i64(n)
	if 2 * r > i64(n) || (2 * r == i64(n) && q & 1 == 1) {
		q += 1
	}
	if q == 256 {
		i, q = i + 1, 0
	}
	top := i64(sn - 1)
	return {i32(min(i, top)), i32(min(i + 1, top)), f64(q) * (1.0 / 256)}
}

// A texel as the GPU's texture unit reads it, components past the format's 0.
@(private)
build_read :: #force_inline proc "contextless" (p: [^]u8, $F: Channel_Format, $C: int) -> [4]f64 #no_bounds_check {
	when F == .F32 {
		t := ([^]f32)(p)
		when C == 1 {
			return {f64(t[0]), 0, 0, 0}
		} else when C == 2 {
			return {f64(t[0]), f64(t[1]), 0, 0}
		} else {
			return {f64(t[0]), f64(t[1]), f64(t[2]), f64(t[3])}
		}
	} else when F == .Rgba8 {
		return {f64(UNORM8_TO_F32[p[0]]), f64(UNORM8_TO_F32[p[1]]), f64(UNORM8_TO_F32[p[2]]), f64(UNORM8_TO_F32[p[3]])}
	} else when F == .Gray8 {
		return {f64(UNORM8_TO_F32[p[0]]), 0, 0, 0}
	} else when F == .Gray16 {
		return {f64(f32(u16(p[0]) | u16(p[1]) << 8) / 65535), 0, 0, 0}
	} else {
		h := ([^]u16)(p)
		return {f64(f16_to_f32(h[0])), f64(f16_to_f32(h[1])), f64(f16_to_f32(h[2])), f64(f16_to_f32(h[3]))}
	}
}

// Stores a blended texel, rounded once to f32, in the format.
@(private)
build_write :: #force_inline proc "contextless" (p: [^]u8, v: [4]f64, $F: Channel_Format, $C: int) #no_bounds_check {
	when F == .F32 {
		t := ([^]f32)(p)
		for k in 0 ..< C {
			t[k] = f32(v[k])
		}
	} else when F == .Rgba8 || F == .Gray8 {
		for k in 0 ..< C {
			p[k] = u8(build_unorm(f32(v[k]), 255))
		}
	} else when F == .Gray16 {
		g := u16(build_unorm(f32(v[0]), 65535))
		p[0], p[1] = u8(g), u8(g >> 8)
	} else {
		h := ([^]u16)(p)
		for k in 0 ..< 4 {
			h[k] = f32_to_f16(f32(v[k]))
		}
	}
}

// round_even(v * m) for v in [0, 1] (the blend of values in it), m 255
// or 65535. v * m is exact in f64, and its only half is at v = 0.5
// (127.5, 32767.5: v = (2k + 1) / 2m is a dyadic f32 only there, m being
// odd), where (m - 1) / 2 is odd and rounding half up rounds to even too:
// so it is floor(v * m + 0.5), the sum exact (every f32 in [0, 1] checked).
@(private)
build_unorm :: #force_inline proc "contextless" (v: f32, m: f64) -> u32 {
	return u32(f64(v) * m + 0.5)
}

// Level `dst` from `src` (channel_build_levels), the columns' taps made
// once a run of BUILD_COLUMNS and each row's once.
@(private)
BUILD_COLUMNS :: 256

@(private)
build_level :: proc "contextless" (src, dst: Build_Level, $F: Channel_Format, $C: int) #no_bounds_check {
	TB := channel_texel_bytes(F, C)
	sp, dp := ([^]u8)(src.ptr), ([^]u8)(dst.ptr)
	xs: [BUILD_COLUMNS]Build_Tap = ---
	for x0 := 0; x0 < dst.w; x0 += BUILD_COLUMNS {
		n := min(BUILD_COLUMNS, dst.w - x0)
		for i in 0 ..< n {
			xs[i] = build_tap(x0 + i, src.w, dst.w)
		}
		for y in 0 ..< dst.h {
			ty := build_tap(y, src.h, dst.h)
			r0, r1 := sp[int(ty.i0) * src.pitch:], sp[int(ty.i1) * src.pitch:]
			out := dp[y * dst.pitch + x0 * TB:]
			Y := ty.t
			for i in 0 ..< n {
				tx := xs[i]
				a, b := int(tx.i0) * TB, int(tx.i1) * TB
				X := tx.t
				c00, c10 := build_read(r0[a:], F, C), build_read(r0[b:], F, C)
				c01, c11 := build_read(r1[a:], F, C), build_read(r1[b:], F, C)
				v := (c00 * (1 - X) + c10 * X) * (1 - Y) + (c01 * (1 - X) + c11 * X) * Y
				build_write(out[i * TB:], v, F, C)
			}
		}
	}
}

// A bound channel as the samplers read it: its levels' addresses and
// sizes, made once per run.
@(private)
Chan_Level :: struct {
	// The level's first texel; Gray8 and Gray16 aligned down to 4 bytes
	// (moved down `mis` bytes) for whole-word reads.
	ptr:   [^]u8,
	mis:   i32,
	pitch: i32, // texels a row
	size:  [2]i32,
	max:   [2]i32,
	fs:    [2]f64, // f64(size)
	inv:   [2]f64, // 1 / size, for Repeat
}

#assert(size_of(Chan_Level) == 64)

@(private)
Chan_Img :: struct {
	ptr:      rawptr, // level 0's texels; nil: unbound
	format:   Channel_Format,
	filter:   Channel_Filter,
	wrap:     [2]Channel_Wrap,
	comps:    u8, // components a texel holds (F32: 1, 2 or 4; Gray: 1; Rgba: 4)
	shift:    u8, // F32: log2(comps)
	held:     u8, // (1 << comps) - 1
	// 0: one level, Nearest, both axes Clamp (the JIT's short path); 1: the rest.
	route:    u8,
	n_levels: i32,
	top:      f32, // f32(n_levels - 1)
	unorm:    f32, // 255, 65535; 0 for halves and floats
	// 0: F32 with one level (the JIT stubs' inline texelFetch); 1: the rest.
	fetch_route: u8,
	// 0: F32, one level, Nearest, both axes Clamp (the stubs' inline read);
	// 1: another format so with sides below 2^24 (sample_sub's FMA floor).
	near_route: u8,
	// 0: F32, one level, Linear, both axes Clamp (the JIT's bilinear_sub).
	lin_route: u8,
	// 0: lin_route 0 with both sides at most 32768 (bilinear_sub in f32
	// with FMA).
	lin_f32:  u8,
	fn512:    [2]f32, // 512 * level 0's size (bilinear_sub in f32)
	n_layers: i32, // 1 for a 2D image
	ltop:     f32, // f32(n_layers - 1)
	_:        [2]u32,
	half:     f64, // what x subtracts: 0.5 for Linear, 0 for Nearest
	fs32:     [2]f32, // level 0's size as f32 (near_route: exact)
	fmax32:   [2]f32, // and its largest index
	// Each level's layer pitch in texels (0 with one layer).
	lpitch:   [CHANNEL_MAX_LEVELS]i32,
	levels:   [CHANNEL_MAX_LEVELS]Chan_Level,
}

#assert(size_of(Chan_Img) == 144 + CHANNEL_MAX_LEVELS * 64)

@(private)
channel_texel_bytes :: proc "contextless" (format: Channel_Format, components: int) -> int {
	switch format {
	case .F32:     return 4 * components
	case .Rgba8:   return 4
	case .Gray8:   return 1
	case .Gray16:  return 2
	case .Rgba16f: return 8
	}
	return 0
}

@(private)
channel_bound :: #force_inline proc "contextless" (im: Channel_Image) -> bool {
	return len(im.data) > 0 || len(im.pixels) > 0
}

// One level's texels, `w x h` of `tb` bytes in rows `stride` bytes apart
// (0: w * tb), checked to be there and addressable in 32 bits.
// With `layers` of them, each `layer_stride` bytes after the one before (0:
// `h` rows on).
@(private)
validate_channel_level :: proc(texels: int, w, h, stride, tb: int, format: Channel_Format, ptr: rawptr, layers, layer_stride: int) {
	ensure(w > 0 && h > 0, "oksl: Channel_Image dimensions out of range")
	row, r_ovf := intrinsics.overflow_mul(w, tb)
	ensure(!r_ovf && row <= int(max(i32)), "oksl: Channel_Image dimensions out of range")
	pitch := stride if stride != 0 else row
	ensure(pitch >= row && pitch % tb == 0, "oksl: Channel_Image.stride must be 0 or a multiple of the texel size, at least width texels")
	span, s_ovf := intrinsics.overflow_mul(pitch, h - 1)
	need, n_ovf := intrinsics.overflow_add(span, row)
	if layers > 1 {
		// The layers before the last, then the last one's rows.
		lstride, l_ovf := intrinsics.overflow_mul(pitch, h)
		if layer_stride != 0 {
			lstride, l_ovf = layer_stride, false
		}
		ensure(!s_ovf && !n_ovf && !l_ovf && lstride >= need && lstride % tb == 0, "oksl: Channel_Image.layer_stride must be 0 or a multiple of the texel size, at least a layer's rows")
		before, b_ovf := intrinsics.overflow_mul(lstride, layers - 1)
		need, n_ovf = intrinsics.overflow_add(before, need)
		n_ovf ||= b_ovf
	}
	// F32 reads index floats, the byte formats bytes (and a word around a
	// Gray texel): both within 31 bits.
	limit := int(max(i32)) * 4 if format == .F32 else int(max(i32)) - 3
	ensure(!s_ovf && !n_ovf && need <= limit, "oksl: Channel_Image dimensions out of range")
	ensure(texels >= need, "oksl: Channel_Image texels are shorter than its rows")
	// Halves load as u16s (rows stay aligned: pitch is whole texels).
	if format == .Gray16 || format == .Rgba16f {
		ensure(uintptr(ptr) % 2 == 0, "oksl: Channel_Image Gray16 and Rgba16f texels must be 2-byte aligned")
	}
}

// The channels' options, once per run: channel_levels and each bound
// image.
@(private)
validate_channels :: proc(p: ^Program, opts: Run_Options) {
	for n in opts.channel_levels {
		ensure(n >= 0 && n <= CHANNEL_MAX_LEVELS, "oksl: Run_Options.channel_levels must be 0 to 16")
	}
	for im, ch in opts.channel_images {
		if !channel_bound(im) {
			continue
		}
		validate_channel_image(im)
		ensure(im.layers <= 1 || ch in p.channel_arrays, "oksl: a Channel_Image with layers binds to a sampler2DArray channel")
	}
}

// One bound image, whatever it binds to: its components and texel bytes.
@(private)
validate_channel_image :: proc(im: Channel_Image) -> (comps, tb: int) {
	// f32(layers - 1) exact, for the clamp.
	ensure(im.layers >= 0 && im.layers <= 1 << 24, "oksl: Channel_Image.layers out of range")
	layers := max(im.layers, 1)
	ensure(im.filter == .Nearest || im.filter == .Linear, "oksl: invalid Channel_Image.filter")
	ensure(im.format >= .F32 && im.format <= .Rgba16f, "oksl: invalid Channel_Image.format")
	for wr in im.wrap {
		ensure(wr == .Clamp || wr == .Repeat, "oksl: invalid Channel_Image.wrap")
	}
	comps = im.components
	if im.format == .F32 {
		ensure(comps == 1 || comps == 2 || comps == 4, "oksl: Channel_Image.components must be 1, 2 or 4")
		ensure(len(im.pixels) == 0, "oksl: an F32 Channel_Image takes its texels in data, not pixels")
	} else {
		ensure(len(im.data) == 0, "oksl: a byte-format Channel_Image takes its texels in pixels, not data")
		fc := 4 if im.format == .Rgba8 || im.format == .Rgba16f else 1
		ensure(comps == 0 || comps == fc, "oksl: Channel_Image.components must be 0 or the format's")
		comps = fc
	}
	tb = channel_texel_bytes(im.format, comps)
	if im.format == .F32 {
		validate_channel_level(len(im.data) * 4, im.width, im.height, im.stride, tb, .F32, raw_data(im.data), layers, im.layer_stride)
	} else {
		validate_channel_level(len(im.pixels), im.width, im.height, im.stride, tb, im.format, raw_data(im.pixels), layers, im.layer_stride)
	}
	ensure(len(im.levels) < CHANNEL_MAX_LEVELS, "oksl: Channel_Image has more than 16 levels")
	for lv, i in im.levels {
		l := uint(i + 1)
		ensure(lv.width == max(im.width >> l, 1) && lv.height == max(im.height >> l, 1), "oksl: Channel_Level l must be max(width >> l, 1) x max(height >> l, 1)")
		validate_channel_level(len(lv.texels), lv.width, lv.height, lv.stride, tb, im.format, raw_data(lv.texels), layers, lv.layer_stride)
		if im.format == .F32 {
			ensure(uintptr(raw_data(lv.texels)) % 4 == 0, "oksl: Channel_Level F32 texels must be 4-byte aligned")
		}
	}
	return
}

// The level's record and its layer pitch in texels (0 with one layer).
@(private)
fill_chan_level :: proc "contextless" (dst: ^Chan_Level, texels: rawptr, w, h, stride, tb: int, format: Channel_Format, layers, layer_stride: int) -> (lpitch: i32) {
	ptr := uintptr(texels)
	mis := 0
	if format == .Gray8 || format == .Gray16 {
		mis = int(ptr & 3)
		ptr -= uintptr(mis)
	}
	pitch := (stride if stride != 0 else w * tb) / tb
	if layers > 1 {
		lpitch = i32(layer_stride / tb if layer_stride != 0 else pitch * h)
	}
	dst^ = {
		ptr   = ([^]u8)(ptr),
		mis   = i32(mis),
		pitch = i32(pitch),
		size  = {i32(w), i32(h)},
		max   = {i32(w - 1), i32(h - 1)},
		fs    = {f64(w), f64(h)},
		inv   = {1 / f64(w), 1 / f64(h)},
	}
	return
}

// Channels past `images` and unbound entries get ptr = nil; the rest
// their levels (validated by validate_channels).
@(private)
fill_chan_imgs :: proc "contextless" (dst: []Chan_Img, images: []Channel_Image) {
	for i in 0 ..< len(dst) {
		if i >= len(images) || !channel_bound(images[i]) {
			dst[i].ptr = nil
			continue
		}
		im := images[i]
		d := &dst[i]
		comps := im.components
		switch im.format {
		case .F32:
		case .Rgba8, .Rgba16f: comps = 4
		case .Gray8, .Gray16:  comps = 1
		}
		tb := channel_texel_bytes(im.format, comps)
		n := 1 + len(im.levels)
		d.ptr = raw_data(im.data) if im.format == .F32 else raw_data(im.pixels)
		unorm: f32 = 255 if im.format == .Rgba8 || im.format == .Gray8 else (65535 if im.format == .Gray16 else 0)
		d.unorm = unorm
		d.half = 0.5 if im.filter == .Linear else 0
		d.format, d.filter, d.wrap = im.format, im.filter, im.wrap
		d.comps = u8(comps)
		d.shift = u8(comps >> 1)
		d.held = u8(1 << uint(comps)) - 1
		// Layers take the general paths (sample_sub and fetch_sub in the JIT).
		layers := max(im.layers, 1)
		one := layers == 1
		d.route = 0 if one && n == 1 && im.filter == .Nearest && im.wrap == {.Clamp, .Clamp} else 1
		d.fetch_route = 0 if one && n == 1 && im.format == .F32 else 1
		d.near_route = 2
		if d.route == 0 && im.width < 1 << 24 && im.height < 1 << 24 {
			d.near_route = 0 if im.format == .F32 else 1
		}
		d.fs32 = {f32(im.width), f32(im.height)}
		d.fmax32 = {f32(im.width - 1), f32(im.height - 1)}
		d.fn512 = {f32(512 * im.width), f32(512 * im.height)}
		// (bilinear_sub's indexes stay under 2^23)
		d.lin_route = 0 if one && n == 1 && im.format == .F32 && im.filter == .Linear && im.wrap == {.Clamp, .Clamp} && im.width < 1 << 22 && im.height < 1 << 22 else 1
		d.lin_f32 = 0 if d.lin_route == 0 && im.width <= 1 << 15 && im.height <= 1 << 15 else 1
		d.n_levels = i32(n)
		d.top = f32(n - 1)
		d.n_layers = i32(layers)
		d.ltop = f32(layers - 1)
		d.lpitch[0] = fill_chan_level(&d.levels[0], d.ptr, im.width, im.height, im.stride, tb, im.format, layers, im.layer_stride)
		for lv, l in im.levels {
			d.lpitch[l + 1] = fill_chan_level(&d.levels[l + 1], raw_data(lv.texels), lv.width, lv.height, lv.stride, tb, im.format, layers, lv.layer_stride)
		}
	}
}

// No channel bound: what runs without channel_images reads.
@(private, rodata)
CHAN_IMGS_NONE: [SAMPLER_COUNT]Chan_Img

// A 2D storage image for run_compute: `width x height` texels in `data`,
// `stride` bytes per row (0 means tightly packed: width * texel bytes).
Storage_Image :: struct {
	data:          []u8,
	width, height: int,
	stride:        int,
	format:        Image_Format,
}

// A stage input across the layer, as one triangle's rasterization gives
// it: per component, a + dx * x + dy * y at the pixel centre (x + 0.5,
// y + 0.5). Integer inputs take the value converted.
Varying :: struct {
	a, dx, dy: [4]f32,
}

// Shades a `width x height` layer into `out` (RGBA8, `stride` bytes per
// row, top-down) from the output at location 0. `uniforms` is the block
// (`Module.block_size` bytes, with iResolution set to the layer size).
// Discarded pixels are left untouched. `scratch` is the caller's working
// memory (one per thread running at once).
run :: proc(p: ^Program, uniforms: []u8, width, height: int, out: []u8, stride: int, scratch: ^Scratch, opts := Run_Options{}) {
	ensure(!opts.region_targets, "oksl: region_targets applies to float targets (run_f32 and run_targets), not run")
	runner_run(p, uniforms, width, height, opts, out, stride, nil, scratch)
}

// Like `run`, writing unclamped float RGBA (`out` has width * height
// entries, or opts.w * opts.h when opts.region_targets is set, row-major).
run_f32 :: proc(p: ^Program, uniforms: []u8, width, height: int, out: [][4]f32, scratch: ^Scratch, opts := Run_Options{}) {
	empty: [1][4]f32
	targets := [1][][4]f32{empty[:0] if opts.region_targets && out == nil else out}
	runner_run(p, uniforms, width, height, opts, nil, 0, targets[:], scratch)
}

// Like `run_f32` for every output: `targets[n]` (width * height entries,
// or opts.w * opts.h when opts.region_targets is set; nil to skip) receives
// the output at location n, integer outputs converted to float, missing
// components 0 (alpha 1).
run_targets :: proc(p: ^Program, uniforms: []u8, width, height: int, targets: [][][4]f32, scratch: ^Scratch, opts := Run_Options{}) {
	runner_run(p, uniforms, width, height, opts, nil, 0, targets, scratch)
}

@(private)
Reg :: [LANES]u32

// Working memory for running programs (registers and masks). A Program is
// only read while it runs, so threads may run one at once, each with its
// own Scratch; it grows to the largest program it has run and is reused,
// so runs allocate nothing once it has.
Scratch :: struct {
	regs:   [dynamic]Reg,
	masks:  [dynamic]Lane_Mask,
	shared: [dynamic]u32,
	chans:  [dynamic]Chan_Img,
}

scratch_destroy :: proc(s: ^Scratch) {
	delete(s.regs)
	delete(s.masks)
	delete(s.shared)
	delete(s.chans)
}

@(private)
Runner :: struct {
	p:             ^Program,
	regs:          []Reg,
	masks:         []Lane_Mask,
	shared:        []u32,
	uniforms:      []u8,
	opts:          Run_Options,
	chan_imgs:     ^[SAMPLER_COUNT]Chan_Img,
	// The batch's lanes: the exec mask it starts with.
	lanes:         Lane_Mask,
	split_barrier: bool,
}

// A runner over `scratch` with the program's constants and uniforms loaded,
// and what follows from them alone computed.
@(private)
runner_begin :: proc(p: ^Program, scratch: ^Scratch, uniforms: []u8, opts: Run_Options) -> Runner {
	if len(scratch.regs) < max(p.num_slots, 1) {
		resize(&scratch.regs, max(p.num_slots, 1))
	}
	if len(scratch.masks) < max(p.num_masks, 2) {
		resize(&scratch.masks, max(p.num_masks, 2))
	}
	r := Runner{p = p, regs = scratch.regs[:], masks = scratch.masks[:], uniforms = uniforms, opts = opts, chan_imgs = &CHAN_IMGS_NONE}
	if len(opts.channel_images) > 0 {
		if len(scratch.chans) < SAMPLER_COUNT {
			resize(&scratch.chans, SAMPLER_COUNT)
		}
		r.chan_imgs = (^[SAMPLER_COUNT]Chan_Img)(raw_data(scratch.chans))
		fill_chan_imgs(r.chan_imgs[:], opts.channel_images)
	}
	for c in p.consts {
		r.regs[c[0]] = c[1]
	}
	for f in p.fills {
		bits: u32
		if int(f.offset) + 4 <= len(uniforms) {
			bits = (^u32)(&uniforms[f.offset])^
		}
		if f.base == .Bool {
			bits = u32(bits != 0)
		}
		r.regs[f.slot] = bits
	}
	// What is the same in every lane and batch, once.
	execute(&r, p.code[:p.init_len])
	return r
}

// A register's lanes from floats, converted to `base`.
@(private)
fill_reg :: #force_inline proc "contextless" (reg: ^Reg, col: F_Lanes, base: Base) {
	#partial switch base {
	case .Int:
		for l in 0 ..< LANES {reg[l] = u32(f_to_i(col[l]))}
	case .Uint:
		for l in 0 ..< LANES {reg[l] = f_to_u(col[l])}
	case .Bool:
		for l in 0 ..< LANES {reg[l] = u32(col[l] != 0)}
	case:
		fl(reg)^ = col
	}
}

// A component's bits converted between bases, as GLSL's constructors do
// (ints and uints share bits).
@(private)
convert_bits :: #force_inline proc "contextless" (bits: u32, from, to: Base) -> u32 {
	switch {
	case from == to || (from != .Float && to != .Float):
		return bits
	case to == .Float:
		return fbits(float_of(bits, from))
	case to == .Int:
		return u32(f_to_i(fval(bits)))
	}
	return f_to_u(fval(bits))
}

// Component bits of `base` as a float.
@(private)
float_of :: #force_inline proc "contextless" (bits: u32, base: Base) -> f32 {
	#partial switch base {
	case .Int:
		return f32(i32(bits))
	case .Uint:
		return f32(bits)
	case .Bool:
		return 1 if bits != 0 else 0
	}
	return fval(bits)
}

// The layer runner's batch: the lanes' quads in rows of BATCH_W / 2.
BATCH_W :: 8
BATCH_H :: LANES / BATCH_W

@(private)
validate_region_targets :: proc(width, height: int, opts: Run_Options, targets: [][][4]f32) {
	ensure(opts.w > 0 && opts.h > 0, "oksl: region_targets requires opts.w > 0 and opts.h > 0")
	ensure(
		width > 0 && height > 0 &&
		opts.x >= 0 && opts.y >= 0 &&
		opts.x <= width && opts.w <= width - opts.x &&
		opts.y <= height && opts.h <= height - opts.y,
		"oksl: region_targets region lies outside the layer",
	)
	need, ovf := intrinsics.overflow_mul(opts.w, opts.h)
	ensure(!ovf, "oksl: region_targets dimensions overflow")
	for target in targets {
		if target != nil {
			ensure(len(target) >= need, "oksl: target is shorter than opts.w * opts.h")
		}
	}
}

@(private)
runner_run :: proc(p: ^Program, uniforms: []u8, width, height: int, opts: Run_Options, out: []u8, stride: int, targets: [][][4]f32, scratch: ^Scratch) {
	ensure(p.stage == .Fragment, "oksl: run requires a fragment program")
	validate_channels(p, opts)
	if opts.region_targets {
		validate_region_targets(width, height, opts, targets)
	}
	if p.jit_body != nil {
		jit_runner_run(p, uniforms, width, height, opts, out, stride, targets, scratch)
		return
	}
	x0, y0 := opts.x, opts.y
	w, h := opts.w, opts.h
	if w <= 0 || h <= 0 {
		x0, y0, w, h = 0, 0, width, height
	}
	fw, fh := f32(width), f32(height)
	// The region within the layer. Batches start on even pixels, so a
	// region's quads, and so its derivatives, are the whole layer's.
	start_x, start_y := max(x0, 0), max(y0, 0)
	end_x := min(x0 + w, width)
	end_y := min(y0 + h, height)
	if start_x >= end_x || start_y >= end_y {
		return
	}
	out_x, out_y, out_stride := 0, 0, width
	if opts.region_targets {
		out_x, out_y, out_stride = start_x, start_y, w
	}
	r := runner_begin(p, scratch, uniforms, opts)
	// Each lane's pixel in the batch.
	lx, ly: [LANES]int
	lxf, lyf: F_Lanes
	for lane in 0 ..< LANES {
		q := lane >> 2
		lx[lane] = q % (BATCH_W / 2) * 2 + (lane & 1)
		ly[lane] = q / (BATCH_W / 2) * 2 + ((lane >> 1) & 1)
		lxf[lane] = f32(lx[lane]) + 0.5
		lyf[lane] = f32(ly[lane]) + 0.5
	}
	for by := start_y &~ 1; by < end_y; by += BATCH_H {
		for bx := start_x &~ 1; bx < end_x; bx += BATCH_W {
			// Pixel centres of the batch's lanes, then each input a whole
			// register at a time.
			px := f32(bx) + lxf
			py := f32(by) + lyf
			for v in p.inputs {
				for k in 0 ..< int(v.n) {
					col: F_Lanes
					switch {
					case v.kind == .Frag_Coord:
						col = px if k == 0 else (py if k == 1 else (1 if k == 3 else 0))
					case v.kind == .Front_Facing:
						// A layer faces the camera.
						col = 1
					case int(v.location) < len(opts.varyings):
						vy := opts.varyings[v.location]
						col = vy.a[k] + vy.dx[k] * px + vy.dy[k] * py
					case v.location == 0 && k < 2:
						col = px / fw if k == 0 else py / fh
					}
					fill_reg(&r.regs[v.slot + i32(k)], col, v.base)
				}
			}
			// The quads the region touches run, all four lanes of each (as
			// a GPU's helpers); the batch's others sample nothing.
			inside := bx >= start_x && by >= start_y && bx + BATCH_W <= end_x && by + BATCH_H <= end_y
			exec := ALL_LANES
			if !inside {
				exec = {}
				for lane in 0 ..< LANES {
					qx, qy := bx + lx[lane] &~ 1, by + ly[lane] &~ 1
					if qx + 1 >= start_x && qx < end_x && qy + 1 >= start_y && qy < end_y {
						exec += {lane}
					}
				}
			}
			r.masks[MASK_EXEC] = exec
			r.lanes = exec
			r.masks[MASK_DISCARD] = {}
			execute(&r, p.code[p.init_len:])
			alive := ~r.masks[MASK_DISCARD]
			full_tile := inside && alive == ALL_LANES
			for v in p.outputs {
				if v.kind != .Stage_Out {
					continue
				}
				target := targets[v.location] if int(v.location) < len(targets) else nil
				bytes := out if v.location == 0 else nil
				if target == nil && bytes == nil {
					continue
				}
				if full_tile && v.base == .Float && v.n == 4 {
					r0 := &r.regs[v.slot]
					r1 := &r.regs[v.slot + 1]
					r2 := &r.regs[v.slot + 2]
					r3 := &r.regs[v.slot + 3]
					if target != nil {
						for lane in 0 ..< LANES {
							target[(by + ly[lane] - out_y) * out_stride + (bx + lx[lane] - out_x)] = {fval(r0[lane]), fval(r1[lane]), fval(r2[lane]), fval(r3[lane])}
						}
					}
					if bytes != nil {
						for lane in 0 ..< LANES {
							o := (by + ly[lane]) * stride + (bx + lx[lane]) * 4
							c := [4]f32{fval(r0[lane]), fval(r1[lane]), fval(r2[lane]), fval(r3[lane])}
							for k in 0 ..< 4 {
								cv := c[k]
								if !(cv > 0) {cv = 0}
								bytes[o + k] = u8(min(cv, 1) * 255 + 0.5)
							}
						}
					}
					continue
				}
				for lane in 0 ..< LANES {
					x := bx + lx[lane]
					y := by + ly[lane]
					if x < start_x || y < start_y || x >= end_x || y >= end_y || lane not_in alive {
						continue
					}
					c := [4]f32{0, 0, 0, 1}
					for k in 0 ..< int(v.n) {
						c[k] = float_of(r.regs[v.slot + i32(k)][lane], v.base)
					}
					if target != nil {
						target[(y - out_y) * out_stride + (x - out_x)] = c
					}
					if bytes != nil {
						o := y * stride + x * 4
						for k in 0 ..< 4 {
							cv := c[k]
							if !(cv > 0) {cv = 0}
							bytes[o + k] = u8(min(cv, 1) * 255 + 0.5)
						}
					}
				}
			}
		}
	}
}

// ---- vertices and quads: for hosts that rasterize ----------------------------------------

// What an attribute's data holds per vertex: 1 to 4 components of f32, i32
// or u32, or four packed ones as SDL_GPU's formats of the name have them.
Vertex_Format :: enum u8 {
	Float,
	Float2,
	Float3,
	Float4,
	Int,
	Int2,
	Int3,
	Int4,
	Uint,
	Uint2,
	Uint3,
	Uint4,
	// Four u8, each / 255, as floats.
	Ubyte4_Norm,
	// Four u16, as uints.
	Ushort4,
}

@(private, rodata)
FORMAT_BASES := [3]Base{.Float, .Int, .Uint}

// A vertex attribute: vertex i's value at `data + i * stride` bytes.
Attribute :: struct {
	data:   rawptr,
	stride: int,
	format: Vertex_Format,
}

// Runs a vertex program over `count` vertices. `attributes[n]` feeds the
// input at location n (absent or nil data: (0, 0, 0, 1)); `position[i]`
// receives vertex i's gl_Position and `varyings[i * p.varying_floats:]`
// its outputs at their offsets (Stage_Var_Info.offset), integers (flat) as
// their bits, so every 32-bit value is exact. Threads may run one program
// at once, each with its own scratch.
run_vertex :: proc(p: ^Program, uniforms: []u8, attributes: []Attribute, count: int, position: [][4]f32, varyings: []f32, scratch: ^Scratch, opts := Run_Options{}) {
	ensure(p.stage == .Vertex, "oksl: run_vertex requires a vertex program")
	validate_channels(p, opts)
	if count <= 0 {
		return
	}
	if p.jit_body != nil {
		jit_run_vertex(p, uniforms, attributes, count, position, varyings, scratch, opts)
		return
	}
	r := runner_begin(p, scratch, uniforms, opts)
	n := p.varying_floats
	for first := 0; first < count; first += LANES {
		lanes := min(count - first, LANES)
		// Per component, one of: absent (0, 0, 0, 1), as stored, converted.
		for v in p.inputs {
			if v.kind == .Vertex_ID {
				reg := &r.regs[v.slot]
				for l in 0 ..< lanes {reg[l] = u32(i32(opts.first_vertex + first + l))}
				continue
			}
			a := attributes[v.location] if int(v.location) < len(attributes) else Attribute{}
			comps := 4
			if a.format <= .Uint4 {
				comps = int(a.format) % 4 + 1
			}
			for k in 0 ..< int(v.n) {
				reg := &r.regs[v.slot + i32(k)]
				if a.data == nil || k >= comps {
					fill := convert_bits(1 if k == 3 else 0, .Int, v.base)
					for l in 0 ..< lanes {reg[l] = fill}
					continue
				}
				vertex := uintptr(a.data) + uintptr(first * a.stride)
				#partial switch a.format {
				case .Ubyte4_Norm:
					for l in 0 ..< lanes {reg[l] = convert_bits(fbits(f32((^u8)(vertex + uintptr(l * a.stride + k))^) / 255), .Float, v.base)}
				case .Ushort4:
					for l in 0 ..< lanes {reg[l] = convert_bits(u32((^u16)(vertex + uintptr(l * a.stride + k * 2))^), .Uint, v.base)}
				case:
					from := FORMAT_BASES[int(a.format) / 4]
					words := vertex + uintptr(k * 4)
					if from == v.base {
						for l in 0 ..< lanes {reg[l] = (^u32)(words + uintptr(l * a.stride))^}
					} else {
						for l in 0 ..< lanes {reg[l] = convert_bits((^u32)(words + uintptr(l * a.stride))^, from, v.base)}
					}
				}
			}
		}
		r.masks[MASK_EXEC] = transmute(Lane_Mask)Lane_Bits((1 << uint(lanes)) - 1)
		r.lanes = r.masks[MASK_EXEC]
		r.masks[MASK_DISCARD] = {}
		execute(&r, p.code[p.init_len:])
		for v in p.outputs {
			if v.kind == .Position {
				for k in 0 ..< 4 {
					reg := fl(&r.regs[v.slot + i32(k)])
					for l in 0 ..< lanes {position[first + l][k] = reg[l]}
				}
				continue
			}
			for k in 0 ..< int(v.n) {
				reg := &r.regs[v.slot + i32(k)]
				out := varyings[first * n + int(v.offset) + k:]
				for l in 0 ..< lanes {out[l * n] = fval(reg[l])}
			}
		}
	}
}

// Lanes of a 2x2 quad: lane i is pixel (i & 1, i >> 1) from its top-left.
Quad_Lanes :: bit_set[0 ..< 4;u8]

// Quads of fragments a host has rasterized, for run_fragment.
Fragment_Quads :: struct {
	count:        int,
	// Each quad's top-left pixel. gl_FragCoord is a lane's pixel centre
	// (z 0, w 1).
	origin:       [][2]i32,
	// Per quad; nil: every quad faces the camera.
	front_facing: []bool,
	// Per quad, the lanes outside the primitive: they run (derivatives read
	// them) but produce nothing. Nil: none.
	helpers:      []Quad_Lanes,
	// The interpolated varyings: float k of the record (Stage_Var_Info.offset)
	// for lane j (quad j / 4, lane j % 4) at varyings[k * count * 4 + j].
	// Integer (flat) inputs are their bits, as run_vertex writes them:
	// take the provoking vertex's.
	varyings:     []f32,
	// Per lane (quad j / 4, lane j % 4), the rasterizer's depth: gl_FragCoord.z.
	// Nil: 0.
	depth:        []f32,
}

// Shades quads with a fragment program, four at a time like a GPU's 2x2
// quads (dFdx / dFdy within each). `targets[n][j]` (nil to skip) receives
// lane j's output at location n (four lanes per quad; integer outputs
// converted to float, missing components 0, alpha 1, (0, 0, 0, 1) where
// the shader writes no location n), `shaded[q]` the lanes of quad q that
// produced one (not helpers, not discarded), and `sample_masks[j]` (when
// given) the samples lane j keeps: gl_SampleMask[0]'s bits, or all of them
// when the shader does not write it, for the host to AND with the
// fragment's coverage. Threads may run one program at once, each with its
// own scratch.
run_fragment :: proc(p: ^Program, uniforms: []u8, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, scratch: ^Scratch, opts := Run_Options{}, sample_masks: []u32 = nil) {
	ensure(p.stage == .Fragment, "oksl: run_fragment requires a fragment program")
	validate_channels(p, opts)
	if quads.count <= 0 {
		return
	}
	if p.jit_body != nil {
		jit_run_fragment(p, uniforms, quads, targets, shaded, scratch, opts, sample_masks)
		return
	}
	r := runner_begin(p, scratch, uniforms, opts)
	total := quads.count * 4
	// The output at each location (n == 0: none).
	outs: [FRAGMENT_OUTPUTS]Program_Var
	mask_v: Program_Var
	has_mask := false
	for v in p.outputs {
		if v.kind == .Stage_Out && int(v.location) < len(outs) {
			outs[v.location] = v
		}
		if v.kind == .Sample_Mask {
			mask_v = v
			has_mask = true
		}
	}
	for first := 0; first < quads.count; first += LANES / 4 {
		nq := min(quads.count - first, LANES / 4)
		lanes := nq * 4
		for v in p.inputs {
			for k in 0 ..< int(v.n) {
				col: F_Lanes
				#partial switch v.kind {
				case .Frag_Coord:
					switch k {
					case 0:
						for l in 0 ..< lanes {col[l] = f32(quads.origin[first + l / 4].x + i32(l & 1)) + 0.5}
					case 1:
						for l in 0 ..< lanes {col[l] = f32(quads.origin[first + l / 4].y + i32((l >> 1) & 1)) + 0.5}
					case 2:
						if quads.depth != nil {
							for l in 0 ..< lanes {col[l] = quads.depth[first * 4 + l]}
						}
					case 3:
						for l in 0 ..< lanes {col[l] = 1}
					}
				case .Front_Facing:
					if quads.front_facing == nil {
						for l in 0 ..< lanes {col[l] = 1}
					} else {
						for l in 0 ..< lanes {col[l] = 1 if quads.front_facing[first + l / 4] else 0}
					}
				case .Stage_In:
					// The record's bits: floats, and integers as run_vertex
					// writes them.
					src := quads.varyings[(int(v.offset) + k) * total + first * 4:]
					reg := fl(&r.regs[v.slot + i32(k)])
					for l in 0 ..< lanes {reg[l] = src[l]}
					continue
				}
				fill_reg(&r.regs[v.slot + i32(k)], col, v.base)
			}
		}
		r.masks[MASK_EXEC] = transmute(Lane_Mask)Lane_Bits((1 << uint(lanes)) - 1)
		r.lanes = r.masks[MASK_EXEC]
		r.masks[MASK_DISCARD] = {}
		execute(&r, p.code[p.init_len:])
		alive := ~r.masks[MASK_DISCARD]
		for qi in 0 ..< nq {
			q := first + qi
			lanes_out: Quad_Lanes
			helpers := quads.helpers[q] if quads.helpers != nil else {}
			for i in 0 ..< 4 {
				l := qi * 4 + i
				if i in helpers || l not_in alive {
					continue
				}
				lanes_out += {i}
				for target, n in targets {
					if target == nil {
						continue
					}
					c := [4]f32{0, 0, 0, 1}
					if n < len(outs) {
						v := outs[n]
						for k in 0 ..< int(v.n) {
							c[k] = float_of(r.regs[v.slot + i32(k)][l], v.base)
						}
					}
					target[q * 4 + i] = c
				}
				if sample_masks != nil {
					sample_masks[q * 4 + i] = r.regs[mask_v.slot][l] if has_mask else ~u32(0)
				}
			}
			shaded[q] = lanes_out
		}
	}
}

// Runs a compute program over `groups.x * groups.y * groups.z` workgroups
// (or `[first_group, first_group + group_count)` when `group_count >= 0`),
// one workgroup at a time in 64-lane chunks. Workgroup `g` has
// `gl_WorkGroupID = (g % groups.x, (g / groups.x) % groups.y, g / (groups.x * groups.y))`.
// Local sizes below 64 leave the rest of a batch's lanes idle; threads may
// run disjoint workgroup ranges at once, each with its own scratch. Panics
// when `opts.buffers` or `opts.images` is missing declared resources or a
// `Storage_Image` has a mismatched format, negative dimensions, or an
// undersized `stride` / `data` slice.
run_compute :: proc(p: ^Program, uniforms: []u8, groups: [3]int, scratch: ^Scratch, opts := Run_Options{}, first_group := 0, group_count := -1) {
	ensure(p.stage == .Compute, "oksl: run_compute requires a compute program")
	ensure(len(opts.buffers) >= p.num_buffers, "oksl: Run_Options.buffers has fewer buffers than declared")
	ensure(len(opts.images) >= len(p.image_formats), "oksl: Run_Options.images has fewer images than declared")
	validate_channels(p, opts)
	for want_fmt, i in p.image_formats {
		im := opts.images[i]
		ensure(im.format == want_fmt, "oksl: Storage_Image.format does not match shader declaration")
		ensure(im.width >= 0 && im.height >= 0 && im.width <= int(max(i32)) && im.height <= int(max(i32)), "oksl: Storage_Image dimensions out of range")
		tb := image_texel_bytes(want_fmt)
		min_stride, s_ovf := intrinsics.overflow_mul(im.width, tb)
		ensure(!s_ovf, "oksl: Storage_Image width overflows")
		row_stride := im.stride if im.stride != 0 else min_stride
		ensure(row_stride >= min_stride, "oksl: Storage_Image.stride is smaller than width * texel_bytes")
		if im.width > 0 && im.height > 0 {
			prev_rows, r_ovf := intrinsics.overflow_mul(im.height - 1, row_stride)
			need, n_ovf := intrinsics.overflow_add(prev_rows, min_stride)
			ensure(!r_ovf && !n_ovf && len(im.data) >= need, "oksl: Storage_Image.data is too small for width, height and stride")
		}
	}
	if groups.x <= 0 || groups.y <= 0 || groups.z <= 0 {
		return
	}
	gxy, xy_ovf := intrinsics.overflow_mul(groups.x, groups.y)
	total_groups, tg_ovf := intrinsics.overflow_mul(gxy, groups.z)
	ensure(!xy_ovf && !tg_ovf, "oksl: dispatch group count overflows")
	start := max(first_group, 0)
	if start >= total_groups || group_count == 0 {
		return
	}
	end := total_groups if group_count < 0 else start + min(group_count, total_groups - start)
	if first_group < 0 && group_count >= 0 {
		end = min(first_group + group_count, total_groups)
		if start >= end {
			return
		}
	}
	if p.jit_body != nil && images_fit_i32(opts.images[:len(p.image_formats)]) {
		jit_run_compute(p, uniforms, groups, start, end, scratch, opts)
		return
	}
	ls := [3]int{max(int(p.local_size[0]), 1), max(int(p.local_size[1]), 1), max(int(p.local_size[2]), 1)}
	local_total := ls.x * ls.y * ls.z
	local_coords: [1024][3]u16
	ti := 0
	for lz in 0 ..< ls.z {
		for ly in 0 ..< ls.y {
			for lx in 0 ..< ls.x {
				local_coords[ti] = {u16(lx), u16(ly), u16(lz)}
				ti += 1
			}
		}
	}
	r := runner_begin(p, scratch, uniforms, opts)
	if p.shared_words > 0 {
		if len(scratch.shared) < p.shared_words {
			resize(&scratch.shared, p.shared_words)
		}
		r.shared = scratch.shared[:p.shared_words]
	}
	single_chunk := local_total <= LANES
	gx_n := groups.x
	gxy_n := groups.x * groups.y
	body := p.code[p.init_len:]
	if p.has_barrier && !single_chunk {
		MAX_CHUNKS :: 1024 / LANES
		num_chunks := (local_total + LANES - 1) / LANES
		reg_stride := max(p.num_slots, 1)
		mask_stride := max(p.num_masks, 2)
		if len(scratch.regs) < num_chunks * reg_stride {
			resize(&scratch.regs, num_chunks * reg_stride)
		}
		if len(scratch.masks) < num_chunks * mask_stride {
			resize(&scratch.masks, num_chunks * mask_stride)
		}
		c0 := scratch.regs[:reg_stride]
		for v in p.inputs {
			if v.kind == .Num_Work_Groups {
				for k in 0 ..< 3 {
					c0[v.slot + i32(k)] = u32(groups[k])
				}
			}
		}
		for c in 1 ..< num_chunks {
			copy(scratch.regs[c * reg_stride : (c + 1) * reg_stride], c0)
		}
		chunk_lanes: [MAX_CHUNKS]Lane_Mask
		for c in 0 ..< num_chunks {
			first := c * LANES
			lanes := min(local_total - first, LANES)
			chunk_lanes[c] = ALL_LANES if lanes == LANES else transmute(Lane_Mask)Lane_Bits((Lane_Bits(1) << uint(lanes)) - 1)
			cregs := scratch.regs[c * reg_stride : (c + 1) * reg_stride]
			for v in p.inputs {
				#partial switch v.kind {
				case .Local_Invocation_ID:
					for k in 0 ..< 3 {
						reg := &cregs[v.slot + i32(k)]
						for l in 0 ..< LANES {
							reg[l] = u32(local_coords[first + l][k])
						}
					}
				case .Local_Invocation_Index:
					reg := &cregs[v.slot]
					for l in 0 ..< LANES {
						reg[l] = u32(first + l)
					}
				}
			}
		}
		r.split_barrier = true
		chunk_pc: [MAX_CHUNKS]int
		for g in start ..< end {
			if len(r.shared) > 0 {
				intrinsics.mem_zero(raw_data(r.shared), len(r.shared) * size_of(u32))
			}
			wg := [3]u32{u32(g % gx_n), u32((g / gx_n) % groups.y), u32(g / gxy_n)}
			base_gid := [3]u32{wg[0] * u32(ls.x), wg[1] * u32(ls.y), wg[2] * u32(ls.z)}
			for c in 0 ..< num_chunks {
				first := c * LANES
				cregs := scratch.regs[c * reg_stride : (c + 1) * reg_stride]
				cmasks := scratch.masks[c * mask_stride : (c + 1) * mask_stride]
				for v in p.inputs {
					#partial switch v.kind {
					case .Work_Group_ID:
						for k in 0 ..< 3 {
							cregs[v.slot + i32(k)] = wg[k]
						}
					case .Global_Invocation_ID:
						for k in 0 ..< 3 {
							reg := &cregs[v.slot + i32(k)]
							bg := base_gid[k]
							for l in 0 ..< LANES {
								reg[l] = bg + u32(local_coords[first + l][k])
							}
						}
					}
				}
				cmasks[MASK_EXEC] = chunk_lanes[c]
				cmasks[MASK_DISCARD] = {}
				chunk_pc[c] = 0
			}
			for {
				any_running := false
				for c in 0 ..< num_chunks {
					if chunk_pc[c] < len(body) {
						r.regs = scratch.regs[c * reg_stride : (c + 1) * reg_stride]
						r.masks = scratch.masks[c * mask_stride : (c + 1) * mask_stride]
						r.lanes = chunk_lanes[c]
						chunk_pc[c] = execute(&r, body, chunk_pc[c])
						if chunk_pc[c] < len(body) {
							any_running = true
						}
					}
				}
				if !any_running {
					break
				}
			}
		}
		return
	}
	for v in p.inputs {
		#partial switch v.kind {
		case .Num_Work_Groups:
			for k in 0 ..< 3 {
				r.regs[v.slot + i32(k)] = u32(groups[k])
			}
		case .Local_Invocation_ID:
			if single_chunk {
				for k in 0 ..< 3 {
					reg := &r.regs[v.slot + i32(k)]
					for l in 0 ..< LANES {
						reg[l] = u32(local_coords[l][k])
					}
				}
			}
		case .Local_Invocation_Index:
			if single_chunk {
				reg := &r.regs[v.slot]
				for l in 0 ..< LANES {
					reg[l] = u32(l)
				}
			}
		}
	}
	for g in start ..< end {
		if len(r.shared) > 0 {
			intrinsics.mem_zero(raw_data(r.shared), len(r.shared) * size_of(u32))
		}
		wg := [3]u32{u32(g % gx_n), u32((g / gx_n) % groups.y), u32(g / gxy_n)}
		base_gid := [3]u32{wg[0] * u32(ls.x), wg[1] * u32(ls.y), wg[2] * u32(ls.z)}
		for v in p.inputs {
			if v.kind == .Work_Group_ID {
				for k in 0 ..< 3 {
					r.regs[v.slot + i32(k)] = wg[k]
				}
			}
		}
		for first := 0; first < local_total; first += LANES {
			lanes := min(local_total - first, LANES)
			for v in p.inputs {
				#partial switch v.kind {
				case .Global_Invocation_ID:
					for k in 0 ..< 3 {
						reg := &r.regs[v.slot + i32(k)]
						bg := base_gid[k]
						for l in 0 ..< LANES {
							reg[l] = bg + u32(local_coords[first + l][k])
						}
					}
				case .Local_Invocation_ID:
					if !single_chunk {
						for k in 0 ..< 3 {
							reg := &r.regs[v.slot + i32(k)]
							for l in 0 ..< LANES {
								reg[l] = u32(local_coords[first + l][k])
							}
						}
					}
				case .Local_Invocation_Index:
					if !single_chunk {
						reg := &r.regs[v.slot]
						for l in 0 ..< LANES {
							reg[l] = u32(first + l)
						}
					}
				}
			}
			exec := ALL_LANES if lanes == LANES else transmute(Lane_Mask)Lane_Bits((Lane_Bits(1) << uint(lanes)) - 1)
			r.masks[MASK_EXEC] = exec
			r.lanes = exec
			r.masks[MASK_DISCARD] = {}
			execute(&r, body)
		}
	}
}

// Whether every image's texels lie in its first 2^31 bytes, so the JIT's
// gathers can address them with signed 32-bit byte offsets; a dispatch with a
// larger image runs on the interpreter.
@(private)
images_fit_i32 :: proc "contextless" (images: []Storage_Image) -> bool {
	for im in images {
		if im.width <= 0 || im.height <= 0 {
			continue
		}
		tb := image_texel_bytes(im.format)
		row := im.stride if im.stride != 0 else im.width * tb
		if (im.height - 1) * row + im.width * tb > 1 << 31 {
			return false
		}
	}
	return true
}

// floor without core:math's modf: values of 2^23 and up (and inf/NaN) are
// already integral.
@(private)
ffloor :: #force_inline proc "contextless" (x: f32) -> f32 {
	if !(abs(x) < 8388608) || x == 0 {
		return x
	}
	t := f32(i32(x))
	return t - 1 if t > x else t
}

@(private)
fceil :: #force_inline proc "contextless" (x: f32) -> f32 {
	return -ffloor(-x)
}

// A register's lanes as floats.
@(private)
F_Lanes :: [LANES]f32

@(private)
Simd_U8 :: #simd[8]u32

@(private)
LANE_BIT :: Simd_U8{1, 2, 4, 8, 16, 32, 64, 128}

@(private)
fl :: #force_inline proc "contextless" (r: ^Reg) -> ^F_Lanes {
	return (^F_Lanes)(r)
}

@(private)
fbits :: #force_inline proc "contextless" (v: f32) -> u32 {
	return transmute(u32)v
}

@(private)
fval :: #force_inline proc "contextless" (v: u32) -> f32 {
	return transmute(f32)v
}

// A float's bits with every NaN as 0x7fc00000 (`canonical`).
@(private)
canon_bits :: #force_inline proc "contextless" (b: u32) -> u32 {
	return b if b & 0x7fff_ffff <= 0x7f80_0000 else 0x7fc0_0000
}

@(private)
canon8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	return intrinsics.simd_select(intrinsics.simd_lanes_ne(x, x), F8(fval(0x7fc0_0000)), x)
}

@(private)
f_to_i :: proc "contextless" (f: f32) -> i32 {
	if !(f == f) {
		return 0
	}
	if f >= 2147483647 {
		return max(i32)
	}
	if f <= -2147483648 {
		return min(i32)
	}
	return i32(f)
}

@(private)
f_to_u :: proc "contextless" (f: f32) -> u32 {
	if !(f > 0) {
		return 0
	}
	if f >= 4294967295 {
		return max(u32)
	}
	return u32(f)
}

// Two-operand integer, comparison and logic ops over whole registers. OP is
// a compile-time constant, so each use is its own branch-free loop.
@(private)
bin_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $OP: VM_Op) {
	for i in 0 ..< i32(in_.n) {
		x, y, d := &regs[in_.a + i], &regs[in_.b + i], &regs[in_.dst + i]
		for l in 0 ..< LANES {
			d[l] = bin_lane(x[l], y[l], OP)
		}
	}
}

@(private)
bin_lane :: #force_inline proc "contextless" (x, y: u32, $OP: VM_Op) -> u32 {
	when OP == .I_Add {
		return x + y
	} else when OP == .I_Sub {
		return x - y
	} else when OP == .I_Mul {
		return x * y
	} else when OP == .I_Div {
		sx, sy := i32(x), i32(y)
		return 0 if sy == 0 else (x if sy == -1 && sx == min(i32) else u32(sx / sy))
	} else when OP == .U_Div {
		return 0 if y == 0 else x / y
	} else when OP == .I_Rem {
		sx, sy := i32(x), i32(y)
		return 0 if sy == 0 || sy == -1 else u32(sx % sy)
	} else when OP == .U_Rem {
		return 0 if y == 0 else x % y
	} else when OP == .Bit_And {
		return x & y
	} else when OP == .Bit_Or {
		return x | y
	} else when OP == .Bit_Xor {
		return x ~ y
	} else when OP == .Shl {
		return x << (y & 31)
	} else when OP == .Shr_I {
		return u32(i32(x) >> (y & 31))
	} else when OP == .Shr_U {
		return x >> (y & 31)
	} else when OP == .I_Lt {
		return u32(i32(x) < i32(y))
	} else when OP == .I_Le {
		return u32(i32(x) <= i32(y))
	} else when OP == .I_Gt {
		return u32(i32(x) > i32(y))
	} else when OP == .I_Ge {
		return u32(i32(x) >= i32(y))
	} else when OP == .U_Lt {
		return u32(x < y)
	} else when OP == .U_Le {
		return u32(x <= y)
	} else when OP == .U_Gt {
		return u32(x > y)
	} else when OP == .U_Ge {
		return u32(x >= y)
	} else when OP == .I_Eq {
		return u32(x == y)
	} else when OP == .I_Ne {
		return u32(x != y)
	} else when OP == .B_And {
		return u32(x != 0 && y != 0)
	} else when OP == .B_Or {
		return u32(x != 0 || y != 0)
	} else {
		#panic("not a two-operand lane op")
	}
}

// Conversions over whole registers, like bin_lanes.
@(private)
un_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $OP: VM_Op) {
	for i in 0 ..< i32(in_.n) {
		x, d := &regs[in_.a + i], &regs[in_.dst + i]
		for l in 0 ..< LANES {
			d[l] = un_lane(x[l], OP)
		}
	}
}

@(private)
un_lane :: #force_inline proc "contextless" (x: u32, $OP: VM_Op) -> u32 {
	when OP == .F_To_I {
		return u32(f_to_i(fval(x)))
	} else when OP == .F_To_U {
		return f_to_u(fval(x))
	} else when OP == .I_To_F {
		return fbits(f32(i32(x)))
	} else when OP == .U_To_F {
		return fbits(f32(x))
	} else when OP == .B_To_F {
		return fbits(1 if x != 0 else 0)
	} else when OP == .F_To_B {
		return u32(!(fval(x) == 0))
	} else when OP == .I_To_B {
		return u32(x != 0)
	} else {
		#panic("not a conversion")
	}
}

// The float built-ins (k = Builtin) over whole registers: a switch per
// instruction, then a loop specialized for that built-in.
@(private)
float_fn1 :: proc "contextless" (regs: []Reg, in_: Instr) {
	#partial switch Builtin(in_.k) {
	case .Radians:
		fn1_lanes(regs, in_, .Radians)
	case .Degrees:
		fn1_lanes(regs, in_, .Degrees)
	case .Sin:
		fn1_lanes(regs, in_, .Sin)
	case .Cos:
		fn1_lanes(regs, in_, .Cos)
	case .Tan:
		fn1_lanes(regs, in_, .Tan)
	case .Asin:
		fn1_lanes(regs, in_, .Asin)
	case .Acos:
		fn1_lanes(regs, in_, .Acos)
	case .Atan:
		fn1_lanes(regs, in_, .Atan)
	case .Sinh:
		fn1_lanes(regs, in_, .Sinh)
	case .Cosh:
		fn1_lanes(regs, in_, .Cosh)
	case .Tanh:
		fn1_lanes(regs, in_, .Tanh)
	case .Asinh:
		fn1_lanes(regs, in_, .Asinh)
	case .Acosh:
		fn1_lanes(regs, in_, .Acosh)
	case .Atanh:
		fn1_lanes(regs, in_, .Atanh)
	case .Exp:
		fn1_lanes(regs, in_, .Exp)
	case .Log:
		fn1_lanes(regs, in_, .Log)
	case .Exp2:
		fn1_lanes(regs, in_, .Exp2)
	case .Log2:
		fn1_lanes(regs, in_, .Log2)
	case .Sqrt:
		fn1_lanes(regs, in_, .Sqrt)
	case .Inversesqrt:
		fn1_lanes(regs, in_, .Inversesqrt)
	case .Floor:
		fn1_lanes(regs, in_, .Floor)
	case .Ceil:
		fn1_lanes(regs, in_, .Ceil)
	case .Fract:
		fn1_lanes(regs, in_, .Fract)
	case .Trunc:
		fn1_lanes(regs, in_, .Trunc)
	case .Round:
		fn1_lanes(regs, in_, .Round)
	case .Round_Even:
		fn1_lanes(regs, in_, .Round_Even)
	case .Abs:
		fn1_lanes(regs, in_, .Abs)
	case .Sign:
		fn1_lanes(regs, in_, .Sign)
	case .Isnan:
		fn1_lanes(regs, in_, .Isnan)
	case .Isinf:
		fn1_lanes(regs, in_, .Isinf)
	case .Float_Bits_To_Int, .Float_Bits_To_Uint:
		fn1_lanes(regs, in_, .Float_Bits_To_Uint)
	}
}

@(private)
fn1_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $B: Builtin) {
	for i in 0 ..< i32(in_.n) {
		x, d := fl(&regs[in_.a + i]), &regs[in_.dst + i]
		when B >= .Sin && B <= .Log2 {
			for l := 0; l < LANES; l += 8 {
				intrinsics.unaligned_store((^F8)(&d[l]), fn1_8(intrinsics.unaligned_load((^F8)(&x[l])), B))
			}
		} else {
			for l in 0 ..< LANES {
				d[l] = fn1(x[l], B)
			}
		}
	}
}

// The float built-ins vm_math.odin computes, eight lanes at a time (the
// interpreter and the JIT's helpers both).
@(private)
fn1_8 :: #force_inline proc "contextless" (x: F8, $B: Builtin) -> F8 {
	when B == .Sin {
		return fsin8(x)
	} else when B == .Cos {
		return fcos8(x)
	} else when B == .Tan {
		return ftan8(x)
	} else when B == .Asin {
		return fasin8(x)
	} else when B == .Acos {
		return facos8(x)
	} else when B == .Atan {
		return fatan8(x)
	} else when B == .Sinh {
		return fsinh8(x)
	} else when B == .Cosh {
		return fcosh8(x)
	} else when B == .Tanh {
		return ftanh8(x)
	} else when B == .Asinh {
		return fasinh8(x)
	} else when B == .Acosh {
		return facosh8(x)
	} else when B == .Atanh {
		return fatanh8(x)
	} else when B == .Exp {
		return fexp8(x)
	} else when B == .Log {
		return flog8(x)
	} else when B == .Exp2 {
		return fexp28(x)
	} else when B == .Log2 {
		return flog28(x)
	} else {
		#panic("not a vm_math.odin built-in")
	}
}

// One float built-in of one operand, as result bits (isnan and isinf give
// bools).
@(private)
fn1 :: #force_inline proc "contextless" (x: f32, $B: Builtin) -> u32 {
	when B == .Radians {
		return fbits(x * (math.PI / 180))
	} else when B == .Degrees {
		return fbits(x * (180 / math.PI))
	} else when B == .Sqrt {
		return fbits(math.sqrt(x))
	} else when B == .Inversesqrt {
		return fbits(1 / math.sqrt(x))
	} else when B == .Floor {
		return fbits(ffloor(x))
	} else when B == .Ceil {
		return fbits(fceil(x))
	} else when B == .Fract {
		return fbits(x - ffloor(x))
	} else when B == .Trunc {
		return fbits(math.trunc(x))
	} else when B == .Round {
		return fbits(math.round(x))
	} else when B == .Round_Even {
		r := math.round(x)
		if abs(x - math.trunc(x)) == 0.5 {
			r = 2 * math.round(x / 2)
		}
		return fbits(r)
	} else when B == .Abs {
		return fbits(x) & 0x7fff_ffff
	} else when B == .Sign {
		return fbits(1 if x > 0 else (-1 if x < 0 else 0))
	} else when B == .Isnan {
		return u32(math.is_nan(x))
	} else when B == .Isinf {
		return u32(math.is_inf(x))
	} else when B == .Float_Bits_To_Uint {
		return canon_bits(fbits(x))
	} else {
		#panic("not a one-operand float built-in")
	}
}

@(private)
float_fn2 :: proc "contextless" (regs: []Reg, in_: Instr) {
	#partial switch Builtin(in_.k) {
	case .Atan2:
		fn2_lanes(regs, in_, .Atan2)
	case .Pow:
		fn2_lanes(regs, in_, .Pow)
	case .Mod:
		fn2_lanes(regs, in_, .Mod)
	case .Min:
		fn2_lanes(regs, in_, .Min)
	case .Max:
		fn2_lanes(regs, in_, .Max)
	case .Step:
		fn2_lanes(regs, in_, .Step)
	}
}

@(private)
fn2_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $B: Builtin) {
	for i in 0 ..< i32(in_.n) {
		x, y, d := fl(&regs[in_.a + i]), fl(&regs[in_.b + i]), fl(&regs[in_.dst + i])
		when B == .Atan2 || B == .Pow {
			for l := 0; l < LANES; l += 8 {
				intrinsics.unaligned_store((^F8)(&d[l]), fn2_8(intrinsics.unaligned_load((^F8)(&x[l])), intrinsics.unaligned_load((^F8)(&y[l])), B))
			}
		} else {
			for l in 0 ..< LANES {
				d[l] = fn2(x[l], y[l], B)
			}
		}
	}
}

@(private)
fn2_8 :: #force_inline proc "contextless" (x, y: F8, $B: Builtin) -> F8 {
	when B == .Atan2 {
		return fatan28(x, y)
	} else when B == .Pow {
		return fpow8(x, y)
	} else {
		#panic("not a vm_math.odin built-in")
	}
}

@(private)
fn2 :: #force_inline proc "contextless" (x, y: f32, $B: Builtin) -> f32 {
	when B == .Mod {
		return x - y * ffloor(x / y)
	} else when B == .Min {
		return y if y < x else x
	} else when B == .Max {
		return y if y > x else x
	} else when B == .Step {
		return 0 if y < x else 1
	} else {
		#panic("not a two-operand float built-in")
	}
}

@(private)
float_fn3 :: proc "contextless" (regs: []Reg, in_: Instr) {
	#partial switch Builtin(in_.k) {
	case .Clamp:
		fn3_lanes(regs, in_, .Clamp)
	case .Mix:
		fn3_lanes(regs, in_, .Mix)
	case .Smoothstep:
		fn3_lanes(regs, in_, .Smoothstep)
	}
}

@(private)
fn3_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $B: Builtin) {
	for i in 0 ..< i32(in_.n) {
		x, y, z, d := fl(&regs[in_.a + i]), fl(&regs[in_.b + i]), fl(&regs[in_.c + i]), fl(&regs[in_.dst + i])
		for l in 0 ..< LANES {
			d[l] = fn3(x[l], y[l], z[l], B)
		}
	}
}

@(private)
fn3 :: #force_inline proc "contextless" (x, y, z: f32, $B: Builtin) -> f32 {
	when B == .Clamp {
		return clamp_f32(x, y, z)
	} else when B == .Mix {
		return x * (1 - z) + y * z
	} else when B == .Smoothstep {
		t := clamp((z - x) / (y - x), 0, 1)
		return t * t * (3 - 2 * t)
	} else {
		#panic("not a three-operand float built-in")
	}
}

// GLSL ES 3.10's clamp, min(max(x, lo), hi), with its max(x, y) = `y if
// x < y else x` and min(x, y) = `y if y < x else x`: a bound equal to x
// leaves x, so clamp(-0.0, 0.0, 1.0) is -0.0. A NaN operand gives the other
// one (clamp(NaN, 0, 1) = 0). Spelled out rather than Odin's min and max,
// which are LLVM's minnum and maxnum and leave the sign of an equal zero to
// the target. The JIT's F_Fn3 Clamp matches it.
@(private)
clamp_f32 :: #force_inline proc "contextless" (x, lo, hi: f32) -> f32 {
	t := lo if x != x || lo > x else x
	return hi if t != t || hi < t else t
}

@(private)
atomic_word_shared :: #force_inline proc "contextless" (ptr: ^u32, op: Atomic_Op, v1, v2: u32) -> u32 {
	old := ptr^
	switch op {
	case .Add:       ptr^ = old + v1
	case .U_Min:     ptr^ = min(old, v1)
	case .S_Min:     ptr^ = u32(min(i32(old), i32(v1)))
	case .U_Max:     ptr^ = max(old, v1)
	case .S_Max:     ptr^ = u32(max(i32(old), i32(v1)))
	case .And:       ptr^ = old & v1
	case .Or:        ptr^ = old | v1
	case .Xor:       ptr^ = old ~ v1
	case .Exchange:  ptr^ = v1
	case .Comp_Swap: if old == v1 {ptr^ = v2}
	}
	return old
}

@(private)
atomic_word_buf :: #force_inline proc "contextless" (ptr: ^u32, op: Atomic_Op, v1, v2: u32) -> u32 {
	switch op {
	case .Add:
		return intrinsics.atomic_add_explicit(ptr, v1, .Relaxed)
	case .And:
		return intrinsics.atomic_and_explicit(ptr, v1, .Relaxed)
	case .Or:
		return intrinsics.atomic_or_explicit(ptr, v1, .Relaxed)
	case .Xor:
		return intrinsics.atomic_xor_explicit(ptr, v1, .Relaxed)
	case .Exchange:
		return intrinsics.atomic_exchange_explicit(ptr, v1, .Relaxed)
	case .Comp_Swap:
		old, _ := intrinsics.atomic_compare_exchange_strong_explicit(ptr, v1, v2, .Relaxed, .Relaxed)
		return old
	case .U_Min, .S_Min, .U_Max, .S_Max:
		old := intrinsics.atomic_load_explicit(ptr, .Relaxed)
		for {
			next: u32
			#partial switch op {
			case .U_Min: next = min(old, v1)
			case .S_Min: next = u32(min(i32(old), i32(v1)))
			case .U_Max: next = max(old, v1)
			case .S_Max: next = u32(max(i32(old), i32(v1)))
			}
			got, ok := intrinsics.atomic_compare_exchange_weak_explicit(ptr, old, next, .Relaxed, .Relaxed)
			if ok {
				return got
			}
			old = got
		}
	}
	return 0
}

// Unchecked: every slot and mask an instruction names was allocated by the
// lowering (dynamic offsets are clamped there), within num_slots and
// num_masks, which runner_begin sizes the scratch to.
@(private)
execute :: proc(r: ^Runner, code: []Instr, start_pc := 0) -> int #no_bounds_check {
	regs := r.regs
	masks := r.masks
	pc := start_pc
	for pc < len(code) {
		in_ := code[pc]
		pc += 1
		n := int(in_.n)
		dst, a, b, c := in_.dst, in_.a, in_.b, in_.c
		exec := masks[MASK_EXEC]
		switch in_.op {
		case .Nop:
		case .Mov:
			for i in 0 ..< i32(n) {
				regs[dst + i] = regs[a + i]
			}
		case .Splat:
			for i in 0 ..< i32(n) {
				regs[dst + i] = regs[a]
			}
		case .Gather:
			regs[dst] = regs[a]
			if n > 1 {regs[dst + 1] = regs[b]}
			if n > 2 {regs[dst + 2] = regs[c]}
			if n > 3 {regs[dst + 3] = regs[in_.d]}
		case .Store:
			if exec == ALL_LANES {
				for i in 0 ..< i32(n) {
					regs[dst + i] = regs[a + i]
				}
				break
			}
			if exec == {} {
				break
			}
			ebits := u64(transmute(Lane_Bits)exec)
			km: [LANES / 8]Simd_U8
			#unroll for k in 0 ..< LANES / 8 {
				km[k] = intrinsics.simd_lanes_ne(Simd_U8(u32(u8(ebits >> uint(k * 8)))) & LANE_BIT, Simd_U8(0))
			}
			for i in 0 ..< i32(n) {
				d, x := &regs[dst + i], &regs[a + i]
				#unroll for k in 0 ..< LANES / 8 {
					dp := (^Simd_U8)(&d[k * 8])
					xp := (^Simd_U8)(&x[k * 8])
					intrinsics.unaligned_store(dp, intrinsics.simd_select(km[k], intrinsics.unaligned_load(xp), intrinsics.unaligned_load(dp)))
				}
			}
		case .Store_Zero:
			if exec == ALL_LANES {
				for i in 0 ..< i32(n) {
					regs[dst + i] = {}
				}
				break
			}
			if exec == {} {
				break
			}
			ebits := u64(transmute(Lane_Bits)exec)
			km: [LANES / 8]Simd_U8
			#unroll for k in 0 ..< LANES / 8 {
				km[k] = intrinsics.simd_lanes_eq(Simd_U8(u32(u8(ebits >> uint(k * 8)))) & LANE_BIT, Simd_U8(0))
			}
			for i in 0 ..< i32(n) {
				d := &regs[dst + i]
				#unroll for k in 0 ..< LANES / 8 {
					dp := (^Simd_U8)(&d[k * 8])
					intrinsics.unaligned_store(dp, intrinsics.unaligned_load(dp) & km[k])
				}
			}
		case .Load_Dyn:
			off0 := regs[b][0]
			uniform := true
			for l in 1 ..< LANES {
				if regs[b][l] != off0 {
					uniform = false
					break
				}
			}
			if uniform {
				off := i32(off0)
				for i in 0 ..< i32(n) {
					regs[dst + i] = regs[a + off + i]
				}
				break
			}
			for l in 0 ..< LANES {
				off := i32(regs[b][l])
				for i in 0 ..< i32(n) {
					regs[dst + i][l] = regs[a + off + i][l]
				}
			}
		case .Store_Dyn:
			if exec == {} {
				break
			}
			if exec == ALL_LANES {
				off0 := regs[b][0]
				uniform := true
				for l in 1 ..< LANES {
					if regs[b][l] != off0 {
						uniform = false
						break
					}
				}
				if uniform {
					off := i32(off0)
					for i in 0 ..< i32(n) {
						regs[dst + off + i] = regs[a + i]
					}
					break
				}
			}
			for l in 0 ..< LANES {
				if l not_in exec {
					continue
				}
				off := i32(regs[b][l])
				for i in 0 ..< i32(n) {
					regs[dst + off + i][l] = regs[a + i][l]
				}
			}
		// Whole-register array ops: no per-lane branches, vectorized.
		// A scalar operand (k) has stride 0.
		case .F_Add:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ + fl(&regs[b + i * sb])^
			}
		case .F_Sub:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ - fl(&regs[b + i * sb])^
			}
		case .F_Mul:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ * fl(&regs[b + i * sb])^
			}
		case .F_Div:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ / fl(&regs[b + i * sb])^
			}
		case .F_Neg:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = regs[a + i][l] ~ 0x8000_0000
				}
			}
		// Integer, comparison, logic and conversion ops: one loop each, the
		// operation picked at compile time (bin_lanes, un_lanes).
		case .I_Add:
			bin_lanes(regs, in_, .I_Add)
		case .I_Sub:
			bin_lanes(regs, in_, .I_Sub)
		case .I_Mul:
			bin_lanes(regs, in_, .I_Mul)
		case .I_Div:
			bin_lanes(regs, in_, .I_Div)
		case .U_Div:
			bin_lanes(regs, in_, .U_Div)
		case .I_Rem:
			bin_lanes(regs, in_, .I_Rem)
		case .U_Rem:
			bin_lanes(regs, in_, .U_Rem)
		case .Bit_And:
			bin_lanes(regs, in_, .Bit_And)
		case .Bit_Or:
			bin_lanes(regs, in_, .Bit_Or)
		case .Bit_Xor:
			bin_lanes(regs, in_, .Bit_Xor)
		case .Shl:
			bin_lanes(regs, in_, .Shl)
		case .Shr_I:
			bin_lanes(regs, in_, .Shr_I)
		case .Shr_U:
			bin_lanes(regs, in_, .Shr_U)
		case .I_Neg:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = 0 - regs[a + i][l]
				}
			}
		case .Bit_Not:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = ~regs[a + i][l]
				}
			}
		case .F_Lt:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] < y[l])}
			}
		case .F_Le:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] <= y[l])}
			}
		case .F_Gt:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] > y[l])}
			}
		case .F_Ge:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] >= y[l])}
			}
		case .F_Eq:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] == y[l])}
			}
		case .F_Ne:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(!(x[l] == y[l]))}
			}
		case .I_Lt:
			bin_lanes(regs, in_, .I_Lt)
		case .I_Le:
			bin_lanes(regs, in_, .I_Le)
		case .I_Gt:
			bin_lanes(regs, in_, .I_Gt)
		case .I_Ge:
			bin_lanes(regs, in_, .I_Ge)
		case .U_Lt:
			bin_lanes(regs, in_, .U_Lt)
		case .U_Le:
			bin_lanes(regs, in_, .U_Le)
		case .U_Gt:
			bin_lanes(regs, in_, .U_Gt)
		case .U_Ge:
			bin_lanes(regs, in_, .U_Ge)
		case .I_Eq:
			bin_lanes(regs, in_, .I_Eq)
		case .I_Ne:
			bin_lanes(regs, in_, .I_Ne)
		case .B_And:
			bin_lanes(regs, in_, .B_And)
		case .B_Or:
			bin_lanes(regs, in_, .B_Or)
		case .B_Not:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = u32(regs[a + i][l] == 0)
				}
			}
		case .F_To_I:
			un_lanes(regs, in_, .F_To_I)
		case .F_To_U:
			un_lanes(regs, in_, .F_To_U)
		case .I_To_F:
			un_lanes(regs, in_, .I_To_F)
		case .U_To_F:
			un_lanes(regs, in_, .U_To_F)
		case .B_To_F:
			un_lanes(regs, in_, .B_To_F)
		case .F_To_B:
			un_lanes(regs, in_, .F_To_B)
		case .I_To_B:
			un_lanes(regs, in_, .I_To_B)
		case .Select:
			for i in 0 ..< i32(n) {
				ra, rb, rd := &regs[a + i], &regs[b + i], &regs[dst + i]
				rc := &regs[c] if in_.k == 1 else &regs[c + i]
				for l in 0 ..< LANES {
					m := u32(0) - u32(rc[l] != 0)
					rd[l] = (ra[l] & m) | (rb[l] & ~m)
				}
			}
		case .All, .Any:
			for l in 0 ..< LANES {
				all, any := true, false
				for i in 0 ..< i32(n) {
					v := regs[a + i][l] != 0
					all &&= v
					any ||= v
				}
				regs[dst][l] = u32(all) if in_.op == .All else u32(any)
			}
		case .F_Fn1:
			float_fn1(regs, in_)
		case .F_Fn2:
			float_fn2(regs, in_)
		case .F_Fn3:
			float_fn3(regs, in_)
		case .I_Fn1:
			bi := Builtin(in_.k)
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					x := i32(regs[a + i][l])
					v := x
					if bi == .Abs {
						v = -x if x < 0 else x
					} else {
						v = 1 if x > 0 else (-1 if x < 0 else 0)
					}
					regs[dst + i][l] = u32(v)
				}
			}
		case .I_Fn2, .U_Fn2:
			bi := Builtin(in_.k)
			signed := in_.op == .I_Fn2
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					x, y := regs[a + i][l], regs[b + i][l]
					less := i32(y) < i32(x) if signed else y < x
					regs[dst + i][l] = (y if less else x) if bi == .Min else (x if less else y)
				}
			}
		case .I_Fn3, .U_Fn3:
			signed := in_.op == .I_Fn3
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					x, lo, hi := regs[a + i][l], regs[b + i][l], regs[c + i][l]
					if signed {
						regs[dst + i][l] = u32(min(max(i32(x), i32(lo)), i32(hi)))
					} else {
						regs[dst + i][l] = min(max(x, lo), hi)
					}
				}
			}
		case .Dot:
			sum := fl(&regs[a])^ * fl(&regs[b])^
			for i in 1 ..< i32(n) {
				sum += fl(&regs[a + i])^ * fl(&regs[b + i])^
			}
			fl(&regs[dst])^ = sum
		case .Length:
			x0 := fl(&regs[a])^
			sum := x0 * x0
			for i in 1 ..< i32(n) {
				x := fl(&regs[a + i])^
				sum += x * x
			}
			for l in 0 ..< LANES {
				sum[l] = math.sqrt(sum[l])
			}
			fl(&regs[dst])^ = sum
		case .Distance:
			d0 := fl(&regs[a])^ - fl(&regs[b])^
			sum := d0 * d0
			for i in 1 ..< i32(n) {
				d := fl(&regs[a + i])^ - fl(&regs[b + i])^
				sum += d * d
			}
			for l in 0 ..< LANES {
				sum[l] = math.sqrt(sum[l])
			}
			fl(&regs[dst])^ = sum
		case .Normalize:
			x0 := fl(&regs[a])^
			sum := x0 * x0
			for i in 1 ..< i32(n) {
				x := fl(&regs[a + i])^
				sum += x * x
			}
			for l in 0 ..< LANES {
				sum[l] = 1 / math.sqrt(sum[l])
			}
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i])^ * sum
			}
		case .Cross:
			x0, x1, x2 := fl(&regs[a])^, fl(&regs[a + 1])^, fl(&regs[a + 2])^
			y0, y1, y2 := fl(&regs[b])^, fl(&regs[b + 1])^, fl(&regs[b + 2])^
			fl(&regs[dst])^ = x1 * y2 - y1 * x2
			fl(&regs[dst + 1])^ = x2 * y0 - y2 * x0
			fl(&regs[dst + 2])^ = x0 * y1 - y0 * x1
		case .Faceforward, .Reflect, .Refract:
			// N/I in a, I/N in b (GLSL argument order); faceforward's
			// dot is Nref's (c) with I.
			x := c if in_.op == .Faceforward else a
			d := fl(&regs[x])^ * fl(&regs[b])^
			for i in 1 ..< i32(n) {
				d += fl(&regs[x + i])^ * fl(&regs[b + i])^
			}
			#partial switch in_.op {
			case .Faceforward:
				for i in 0 ..< i32(n) {
					v := fl(&regs[a + i])^
					for l in 0 ..< LANES {
						v[l] = v[l] if d[l] < 0 else -v[l]
					}
					fl(&regs[dst + i])^ = v
				}
			case .Reflect:
				for i in 0 ..< i32(n) {
					fl(&regs[dst + i])^ = fl(&regs[a + i])^ - 2 * d * fl(&regs[b + i])^
				}
			case:
				eta := fl(&regs[c])^
				k := 1 - eta * eta * (1 - d * d)
				s: F_Lanes
				for l in 0 ..< LANES {
					s[l] = eta[l] * d[l] + math.sqrt(k[l])
				}
				for i in 0 ..< i32(n) {
					v := eta * fl(&regs[a + i])^ - s * fl(&regs[b + i])^
					for l in 0 ..< LANES {
						v[l] = v[l] if k[l] >= 0 else 0
					}
					fl(&regs[dst + i])^ = v
				}
			}
		case .Mat_Mul:
			rows, inner, cols := n, int(in_.k), int(in_.d)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					sum := fl(&regs[a + i32(row)])^ * fl(&regs[b + i32(col * inner)])^
					for k in 1 ..< inner {
						sum += fl(&regs[a + i32(k * rows + row)])^ * fl(&regs[b + i32(col * inner + k)])^
					}
					fl(&regs[dst + i32(col * rows + row)])^ = sum
				}
			}
		case .Vec_Mat:
			rows, cols := n, int(in_.k)
			for col in 0 ..< cols {
				sum := fl(&regs[a])^ * fl(&regs[b + i32(col * rows)])^
				for row in 1 ..< rows {
					sum += fl(&regs[a + i32(row)])^ * fl(&regs[b + i32(col * rows + row)])^
				}
				fl(&regs[dst + i32(col)])^ = sum
			}
		case .Transpose:
			rows, cols := n, int(in_.k)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					regs[dst + i32(row * cols + col)] = regs[a + i32(col * rows + row)]
				}
			}
		case .Outer:
			rows, cols := n, int(in_.k)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					fl(&regs[dst + i32(col * rows + row)])^ = fl(&regs[a + i32(row)])^ * fl(&regs[b + i32(col)])^
				}
			}
		case .Determinant, .Inverse:
			switch in_.k {
			case 2:
				lane_matrices(regs, in_, 2)
			case 3:
				lane_matrices(regs, in_, 3)
			case:
				lane_matrices(regs, in_, 4)
			}
		case .Deriv_X, .Deriv_Y, .Fwidth:
			for i in 0 ..< i32(n) {
				src := regs[a + i]
				for q in 0 ..< LANES / 4 {
					o := q * 4
					v00, v10, v01, v11 := fval(src[o]), fval(src[o + 1]), fval(src[o + 2]), fval(src[o + 3])
					dx0, dx1 := v10 - v00, v11 - v01
					dy0, dy1 := v01 - v00, v11 - v10
					res: [4]f32
					#partial switch in_.op {
					case .Deriv_X:
						res = {dx0, dx0, dx1, dx1}
					case .Deriv_Y:
						res = {dy0, dy1, dy0, dy1}
					case:
						res = {abs(dx0) + abs(dy0), abs(dx0) + abs(dy1), abs(dx1) + abs(dy0), abs(dx1) + abs(dy1)}
					}
					for k in 0 ..< 4 {
						regs[dst + i][o + k] = fbits(res[k])
					}
				}
			}
		case .Texture, .Texture_Bias, .Texel_Fetch:
			sample_texture(r, in_, exec)
		case .Load_Buf:
			// The word index in 32 bits, wrapping as SPIR-V's and Metal's
			// do; past the host's words, 0.
			words := r.opts.buffers[in_.k] if int(in_.k) < len(r.opts.buffers) else nil
			idx := &regs[in_.a]
			for i in 0 ..< i32(in_.n) {
				d := &regs[in_.dst + i]
				for l in 0 ..< LANES {
					w := idx[l] * u32(in_.b) + u32(in_.c + i)
					d[l] = words[w] if uint(w) < uint(len(words)) else 0
				}
			}
		case .Store_Buf:
			if exec == {} || int(in_.k) >= len(r.opts.buffers) {
				break
			}
			store_buf_lanes(raw_data(regs), r.opts.buffers[in_.k], in_, transmute(Lane_Bits)exec)
		case .Image_Load:
			load_image(r, in_, exec)
		case .Image_Store:
			store_image(r, in_, exec)
		case .Image_Size:
			w, h: i32
			if int(in_.k) < len(r.opts.images) {
				im := r.opts.images[in_.k]
				w, h = i32(max(im.width, 0)), i32(max(im.height, 0))
			}
			regs[dst] = u32(w)
			regs[dst + 1] = u32(h)
		case .Texture_Size:
			ch := int(in_.k)
			if im := &r.chan_imgs[ch]; im.ptr != nil {
				bound_size_lanes(im, in_.d != 0, raw_data(&regs[a]), raw_data(&regs[dst]), raw_data(&regs[dst + 1]), raw_data(&regs[dst + 2]) if in_.n == 3 else nil, LANES)
				break
			}
			size := channel_size(r, ch)
			for l in 0 ..< LANES {
				// Negative lods shift everything out.
				lod := regs[a][l]
				for i in 0 ..< i32(2) {
					regs[dst + i][l] = u32(max(size[i] >> lod, 1) if size[i] > 0 else 0)
				}
				if in_.n == 3 {
					regs[dst + 2][l] = u32(size.z)
				}
			}
		case .Mask_Copy:
			masks[dst] = masks[a]
		case .Mask_Cond:
			lanes: Lane_Mask
			for l in 0 ..< LANES {
				if regs[b][l] != 0 {
					lanes += {l}
				}
			}
			masks[dst] = masks[a] - lanes if in_.k == 1 else masks[a] & lanes
		case .Mask_Restore:
			kill := masks[MASK_DISCARD]
			if b >= 0 {kill += masks[b]}
			if c >= 0 {kill += masks[c]}
			if in_.d >= 0 {kill += masks[in_.d]}
			masks[dst] = masks[a] - kill
		case .Mask_Or:
			masks[dst] = masks[a] + masks[b]
		case .Mask_Clear:
			masks[dst] = {}
		case .Mask_Kill:
			masks[dst] += exec
			masks[MASK_EXEC] = {}
			if dst == MASK_DISCARD && r.lanes <= masks[MASK_DISCARD] {
				return len(code)
			}
		case .Jump:
			pc = int(a)
		case .Jump_If_None:
			if exec == {} {
				pc = int(a)
			}
		case .Load_Shared:
			sh := r.shared
			if in_.k == 0 {
				for i in 0 ..< i32(n) {
					regs[dst + i] = sh[c + i]
				}
				break
			}
			off0 := regs[a][0]
			uniform := true
			for l in 1 ..< LANES {
				if regs[a][l] != off0 {
					uniform = false
					break
				}
			}
			if uniform {
				base := c + i32(off0)
				for i in 0 ..< i32(n) {
					regs[dst + i] = sh[base + i]
				}
				break
			}
			for i in 0 ..< i32(n) {
				d := &regs[dst + i]
				base := c + i
				for l in 0 ..< LANES {
					d[l] = sh[base + i32(regs[a][l])]
				}
			}
		case .Store_Shared:
			if exec == {} {
				break
			}
			store_shared_lanes(raw_data(regs), raw_data(r.shared), in_, transmute(Lane_Bits)exec)
		case .Atomic_Shared:
			atomic_shared_lanes(raw_data(regs), raw_data(r.shared), in_, transmute(Lane_Bits)exec)
		case .Atomic_Buf:
			words := r.opts.buffers[in_.d] if int(in_.d) < len(r.opts.buffers) else nil
			atomic_buf_lanes(raw_data(regs), words, in_, transmute(Lane_Bits)exec)
		case .Barrier:
			if r.split_barrier {
				return pc
			}
		}
	}
	return len(code)
}

@(private)
MAX_TEXEL_F32 :: f32(2147483520.0) // 0x4effffff, largest exact f32 < 2^31

// The texel index floor(x) gives: NaN as -2^31, within [-2^31, 2^31 - 2].
@(private)
bound_index :: #force_inline proc "contextless" (g: f64) -> i32 {
	h := g if g > -2147483648.0 else -2147483648.0
	h = h if h < 2147483646.0 else 2147483646.0
	return i32(h)
}

// Texel index i on an axis of n texels, and the next one, as `W` addresses them.
@(private)
bound_wrap :: #force_inline proc "contextless" (i, n: i32, $W: Channel_Wrap) -> (i0, i1: i32) {
	when W == .Repeat {
		i0 = i %% n
		i1 = 0 if i0 + 1 == n else i0 + 1
	} else {
		a, b := i if i > 0 else 0, i + 1 if i + 1 > 0 else 0
		i0 = a if a < n - 1 else n - 1
		i1 = b if b < n - 1 else n - 1
	}
	return
}

// A fraction's weight of 256: t * 256 (t in [0, 1)) rounded to nearest even.
@(private)
bound_weight :: #force_inline proc "contextless" (t: f64) -> i32 {
	y := t * 256
	f := i32(y)
	d := y - f64(f)
	return f + (i32(d > 0.5) | (i32(d == 0.5) & f & 1)) // without a branch
}

// A texel's stored values (the unsigned normalized formats' integers),
// components past the format's 0; `lo` texels on (its layer's offset).
@(private)
bound_raw :: #force_inline proc "contextless" (lv: ^Chan_Level, x, y: i32, lo: int, $F: Channel_Format, $C: int) -> [4]f32 #no_bounds_check {
	i := int(y) * int(lv.pitch) + int(x) + lo
	when F == .F32 {
		t := ([^]f32)(lv.ptr)[i * C:]
		when C == 1 {
			return {t[0], 0, 0, 0}
		} else when C == 2 {
			return {t[0], t[1], 0, 0}
		} else {
			return {t[0], t[1], t[2], t[3]}
		}
	} else when F == .Rgba8 {
		b := lv.ptr[i * 4:]
		return {f32(b[0]), f32(b[1]), f32(b[2]), f32(b[3])}
	} else when F == .Gray8 {
		return {f32(lv.ptr[int(lv.mis) + i]), 0, 0, 0}
	} else when F == .Gray16 {
		b := lv.ptr[int(lv.mis) + i * 2:]
		return {f32(u16(b[0]) | u16(b[1]) << 8), 0, 0, 0}
	} else {
		h := ([^]u16)(&lv.ptr[i * 8])
		return {f16_to_f32(h[0]), f16_to_f32(h[1]), f16_to_f32(h[2]), f16_to_f32(h[3])}
	}
}

// Texel (x, y) of a level, decoded, missing components (0, 0, 0, 1);
// `lo` texels on (its layer's offset).
@(private)
bound_texel :: #force_inline proc "contextless" (lv: ^Chan_Level, x, y: i32, lo: int, $F: Channel_Format, $C: int) -> [4]f32 #no_bounds_check {
	i := int(y) * int(lv.pitch) + int(x) + lo
	when F == .F32 {
		t := ([^]f32)(lv.ptr)[i * C:]
		when C == 1 {
			return {t[0], 0, 0, 1}
		} else when C == 2 {
			return {t[0], t[1], 0, 1}
		} else {
			return {t[0], t[1], t[2], t[3]}
		}
	} else when F == .Rgba8 {
		b := lv.ptr[i * 4:]
		return {UNORM8_TO_F32[b[0]], UNORM8_TO_F32[b[1]], UNORM8_TO_F32[b[2]], UNORM8_TO_F32[b[3]]}
	} else when F == .Gray8 {
		return {UNORM8_TO_F32[lv.ptr[int(lv.mis) + i]], 0, 0, 1}
	} else when F == .Gray16 {
		b := lv.ptr[int(lv.mis) + i * 2:]
		return {f32(u16(b[0]) | u16(b[1]) << 8) / 65535, 0, 0, 1}
	} else {
		h := ([^]u16)(&lv.ptr[i * 8])
		return {f16_to_f32(h[0]), f16_to_f32(h[1]), f16_to_f32(h[2]), f16_to_f32(h[3])}
	}
}

// One axis of every lane's tap at its level `lvl`: the texel indexes and,
// for Linear, the fraction w / 256.
@(private)
Bound_Axis :: struct {
	i0, i1: [LANES]i32,
	t:      F_Lanes,
}

@(private)
bound_axis_lanes :: proc "contextless" (im: ^Chan_Img, c: ^F_Lanes, lvl: ^[LANES]i32, ax: int, out: ^Bound_Axis, $FI: Channel_Filter, $W: Channel_Wrap, $MIPS: bool) #no_bounds_check {
	half := im.half
	lv0 := &im.levels[0]
	for l in 0 ..< LANES {
		lv := lv0
		when MIPS {
			lv = &im.levels[lvl[l]]
		}
		// x within +-2^52 (NaN as -2^52), where every f64 is whole: the
		// same index and weight as floor(x) and x - floor(x) give, without
		// a branch.
		x := f64(c[l]) * lv.fs[ax] - half
		x = x if x > -4503599627370496 else -4503599627370496
		x = x if x < 4503599627370496 else 4503599627370496
		t := f64(i64(x))
		g := t - 1 if t > x else t
		out.i0[l], out.i1[l] = bound_wrap(bound_index(g), lv.size[ax], W)
		when FI == .Linear {
			out.t[l] = f32(bound_weight(x - g)) * (1.0 / 256)
		}
	}
}

@(private)
bound_axis :: proc "contextless" (im: ^Chan_Img, c: ^F_Lanes, lvl: ^[LANES]i32, ax: int, out: ^Bound_Axis, $FI: Channel_Filter, $MIPS: bool) {
	if im.wrap[ax] == .Repeat {
		bound_axis_lanes(im, c, lvl, ax, out, FI, .Repeat, MIPS)
	} else {
		bound_axis_lanes(im, c, lvl, ax, out, FI, .Clamp, MIPS)
	}
}

// Every lane's tap at its level `lvl` (and with layers, `L`, its layer
// `lay`): the stored values blended (Channel_Filter), not yet divided by an
// unsigned normalized format's largest; the components a format lacks
// unset.
@(private)
bound_tap_lanes :: proc "contextless" (im: ^Chan_Img, u, v: ^F_Lanes, lvl, lay: ^[LANES]i32, out: ^[LANES][4]f32, $F: Channel_Format, $C: int, $FI: Channel_Filter, $MIPS, $L: bool) #no_bounds_check {
	ax, ay: Bound_Axis = ---, ---
	bound_axis(im, u, lvl, 0, &ax, FI, MIPS)
	bound_axis(im, v, lvl, 1, &ay, FI, MIPS)
	lv0 := &im.levels[0]
	for l in 0 ..< LANES {
		lv := lv0
		lo := 0
		when MIPS {
			lv = &im.levels[lvl[l]]
		}
		when L {
			lo = int(lay[l]) * int(im.lpitch[lvl[l] if MIPS else 0])
		}
		when FI == .Nearest {
			out[l] = bound_raw(lv, ax.i0[l], ay.i0[l], lo, F, C)
		} else {
			tx, ty := ax.t[l], ay.t[l]
			top := bound_raw(lv, ax.i0[l], ay.i0[l], lo, F, C)
			top += (bound_raw(lv, ax.i1[l], ay.i0[l], lo, F, C) - top) * tx
			bot := bound_raw(lv, ax.i0[l], ay.i1[l], lo, F, C)
			bot += (bound_raw(lv, ax.i1[l], ay.i1[l], lo, F, C) - bot) * tx
			out[l] = top + (bot - top) * ty
		}
	}
}

// Every lane's sample: level l0's tap, blended toward level l0 + 1's by
// w / 256 where w > 0, then divided by the unsigned normalized formats'
// largest stored value.
@(private)
bound_sample_lanes :: proc "contextless" (im: ^Chan_Img, u, v: ^F_Lanes, lvl0, wl, lay: ^[LANES]i32, blend: bool, out: ^[4]F_Lanes, $F: Channel_Format, $C: int, $FI: Channel_Filter, $L: bool) #no_bounds_check {
	s: [LANES][4]f32 = ---
	if im.n_levels > 1 {
		bound_tap_lanes(im, u, v, lvl0, lay, &s, F, C, FI, true, L)
	} else {
		bound_tap_lanes(im, u, v, lvl0, lay, &s, F, C, FI, false, L)
	}
	if blend {
		s1: [LANES][4]f32 = ---
		lvl1: [LANES]i32 = ---
		for l in 0 ..< LANES {
			lvl1[l] = lvl0[l] + 1 if wl[l] > 0 else lvl0[l]
		}
		bound_tap_lanes(im, u, v, &lvl1, lay, &s1, F, C, FI, true, L)
		for l in 0 ..< LANES {
			if wl[l] > 0 {
				s[l] += (s1[l] - s[l]) * (f32(wl[l]) * (1.0 / 256))
			}
		}
	}
	for k in 0 ..< C {
		for l in 0 ..< LANES {
			out[k][l] = s[l][k]
		}
	}
	when F == .Rgba8 || F == .Gray8 || F == .Gray16 {
		unorm := im.unorm
		for k in 0 ..< C {
			for l in 0 ..< LANES {
				out[k][l] /= unorm
			}
		}
	}
	when C == 1 {
		out[1], out[2] = {}, {}
		out[3] = 1
	} else when C == 2 {
		out[2] = {}
		out[3] = 1
	}
}

@(private)
bound_sample_filter :: proc "contextless" (im: ^Chan_Img, u, v: ^F_Lanes, lvl0, wl, lay: ^[LANES]i32, blend: bool, out: ^[4]F_Lanes, $F: Channel_Format, $C: int) {
	if im.n_layers > 1 {
		if im.filter == .Nearest {
			bound_sample_lanes(im, u, v, lvl0, wl, lay, blend, out, F, C, .Nearest, true)
		} else {
			bound_sample_lanes(im, u, v, lvl0, wl, lay, blend, out, F, C, .Linear, true)
		}
	} else if im.filter == .Nearest {
		bound_sample_lanes(im, u, v, lvl0, wl, lay, blend, out, F, C, .Nearest, false)
	} else {
		bound_sample_lanes(im, u, v, lvl0, wl, lay, blend, out, F, C, .Linear, false)
	}
}

// The layer a sample at coordinate r reads: clamp(round_even(r), 0, top),
// NaN as 0 (the sampler callback's layer, clamped to the layers).
@(private)
bound_layer :: #force_inline proc "contextless" (r, top: f32) -> i32 {
	t := fval(fn1(r, .Round_Even))
	// NaN as 0. Off x86 LLVM makes this clamp fmaxnm, which gives a
	// signaling NaN back quiet (then the last layer); round_even passes one
	// through in integers, so it is quieted first, as bound_clamped_f32's
	// axis does (-0 becomes +0, which the clamp makes 0 anyway).
	when ODIN_ARCH != .amd64 && ODIN_ARCH != .i386 {
		t = t + 0
	}
	t = t if t > 0 else 0
	t = t if t < top else top
	return i32(t)
}

// The level the derivatives of each quad of eight lanes (two quads) pick
// in `w x h` texels, in all four of its lanes: 0.5 * log2 of the larger
// squared length of uv's differences along the quad's top row and left
// column (max as maxnum), by quad_level8. texture() of a callback channel
// (sampler_args8) and of a bound one alike.
@(private)
quad_levels8 :: #force_inline proc "contextless" (u, v: F8, w, h: f32) -> F8 {
	u0, v0 := intrinsics.simd_shuffle(u, u, 0, 0, 0, 0, 4, 4, 4, 4), intrinsics.simd_shuffle(v, v, 0, 0, 0, 0, 4, 4, 4, 4)
	u1, v1 := intrinsics.simd_shuffle(u, u, 1, 1, 1, 1, 5, 5, 5, 5), intrinsics.simd_shuffle(v, v, 1, 1, 1, 1, 5, 5, 5, 5)
	u2, v2 := intrinsics.simd_shuffle(u, u, 2, 2, 2, 2, 6, 6, 6, 6), intrinsics.simd_shuffle(v, v, 2, 2, 2, 2, 6, 6, 6, 6)
	dxx, dxy := (u1 - u0) * w, (v1 - v0) * h
	dyx, dyy := (u2 - u0) * w, (v2 - v0) * h
	x2, y2 := dxx * dxx + dxy * dxy, dyx * dyx + dyy * dyy
	// max (maxnum): y2 where x2 is NaN or smaller.
	m := intrinsics.simd_select(intrinsics.simd_lanes_ne(x2, x2) | intrinsics.simd_lanes_gt(y2, x2), y2, x2)
	return quad_level8(m)
}

// lod += quad_levels8 for eight lanes at u, v and lod (by pointer, for
// the JIT's helpers too).
@(private)
add_quad_levels8 :: #force_inline proc "contextless" (u, v, lod: ^f32, w, h: f32) {
	l := intrinsics.unaligned_load((^F8)(lod)) + quad_levels8(intrinsics.unaligned_load((^F8)(u)), intrinsics.unaligned_load((^F8)(v)), w, h)
	intrinsics.unaligned_store((^F8)(lod), l)
}

// A lane's levels at lod `L` of a chain whose top level is `top`: level
// l0, and l0 + 1's weight of 256 (0 without one).
@(private)
bound_levels :: #force_inline proc "contextless" (L, top: f32) -> (l0, w: i32) {
	if !(L > 0) {
		return 0, 0
	}
	lc := L if L < top else top
	l0 = i32(lc)
	w = bound_weight(f64(lc - f32(l0))) // 0 at the top level, where lc = l0
	return
}

// F32 with one level, both axes clamped: every lane in one loop. Clamped
// to the edge, a coordinate outside [0, 1] (or NaN) reads the edge texel
// twice, as one at 0 or 1 does, whatever the weight; so the coordinate is
// clamped to [0, 1] first and T = f64(c) * 256n - 128 (Linear) stays small:
// the index floor(T) >> 8, the weight round_even(T) - 256 * index, as the
// JIT's bilinear_sub takes them. The same rule as bound_sample_lanes.
@(private)
bound_clamped_f32 :: proc "contextless" (im: ^Chan_Img, u, v: ^F_Lanes, out: ^[4]F_Lanes, $C: int, $FI: Channel_Filter) #no_bounds_check {
	lv := &im.levels[0]
	t := ([^]f32)(lv.ptr)
	pitch := int(lv.pitch)
	mx, my := lv.max[0], lv.max[1]
	when FI == .Nearest {
		fx, fy := lv.fs[0], lv.fs[1]
		for l in 0 ..< LANES {
			x, y := f64(u[l]) * fx, f64(v[l]) * fy
			x = x if x > 0 else 0
			y = y if y > 0 else 0
			x = x if x < 2147483646 else 2147483646
			y = y if y < 2147483646 else 2147483646
			ix, iy := i32(x), i32(y)
			ix = ix if ix < mx else mx
			iy = iy if iy < my else my
			p := t[(int(iy) * pitch + int(ix)) * C:]
			out[0][l] = p[0]
			when C >= 2 {out[1][l] = p[1]}
			when C == 4 {out[2][l], out[3][l] = p[2], p[3]}
		}
	} else {
		nx, ny := lv.fs[0] * 256, lv.fs[1] * 256
		axis :: #force_inline proc "contextless" (c: f32, n256: f64, mx: i32) -> (i0, i1: i32, tw: f32) {
			// NaN as 0. x86's maxps does that for every NaN; elsewhere LLVM
			// turns these compare-and-select clamps into maxnum and minnum
			// (fmaxnm and fminnm on arm64), which give a signaling NaN back
			// quiet and then 1 (v0.0.22 read the last texel there), so a
			// signaling NaN is quieted first: + 0 (LLVM keeps it, for -0;
			// -0 becomes +0, which the clamp makes 0 anyway). Not on x86,
			// where it would cost 2.6% of a Linear tap for nothing.
			cc := c
			when ODIN_ARCH != .amd64 && ODIN_ARCH != .i386 {
				cc = c + 0
			}
			cc = cc if cc > 0 else 0
			cc = cc if cc < 1 else 1
			T := f64(cc) * n256 - 128
			// round_even(T) by 1.5 * 2^52 (|T| < 2^31), then floor(T) from it.
			rt := (T + 6755399441055744.0) - 6755399441055744.0
			ft := rt - 1 if rt > T else rt
			i := i32(ft) >> 8
			w := i32(rt) - (i << 8)
			i0 = i if i > 0 else 0
			i1 = i + 1 if i + 1 < mx else mx
			tw = f32(w) * (1.0 / 256)
			return
		}
		ax, ay: Bound_Axis = ---, ---
		for l in 0 ..< LANES {
			ax.i0[l], ax.i1[l], ax.t[l] = axis(u[l], nx, mx)
		}
		for l in 0 ..< LANES {
			ay.i0[l], ay.i1[l], ay.t[l] = axis(v[l], ny, my)
		}
		for l in 0 ..< LANES {
			x0, x1, tx := ax.i0[l], ax.i1[l], ax.t[l]
			y0, y1, ty := ay.i0[l], ay.i1[l], ay.t[l]
			r0, r1 := int(y0) * pitch, int(y1) * pitch
			p00, p10 := t[(r0 + int(x0)) * C:], t[(r0 + int(x1)) * C:]
			p01, p11 := t[(r1 + int(x0)) * C:], t[(r1 + int(x1)) * C:]
			when C == 4 {
				c00, c10 := ((^[4]f32)(p00))^, ((^[4]f32)(p10))^
				c01, c11 := ((^[4]f32)(p01))^, ((^[4]f32)(p11))^
				top := c00 + (c10 - c00) * tx
				bot := c01 + (c11 - c01) * tx
				res := top + (bot - top) * ty
				out[0][l], out[1][l], out[2][l], out[3][l] = res[0], res[1], res[2], res[3]
			} else {
				for k in 0 ..< C {
					top := p00[k] + (p10[k] - p00[k]) * tx
					bot := p01[k] + (p11[k] - p01[k]) * tx
					out[k][l] = top + (bot - top) * ty
				}
			}
		}
	}
	when C == 1 {
		out[1], out[2] = {}, {}
		out[3] = 1
	} else when C == 2 {
		out[2] = {}
		out[3] = 1
	}
}

// texture(), textureLod and textureGrad of a bound channel, every lane.
@(private)
sample_bound :: proc(r: ^Runner, im: ^Chan_Img, bias: bool, a, b, dst: i32) #no_bounds_check {
	regs := r.regs
	u, v := fl(&regs[a]), fl(&regs[a + 1])
	lvl0, wl: [LANES]i32
	blend := false
	if im.n_levels > 1 {
		lod := fl(&regs[b])
		level := lod^
		if bias {
			w, h := f32(im.levels[0].size[0]), f32(im.levels[0].size[1])
			for g := 0; g < LANES; g += 8 {
				add_quad_levels8(&u[g], &v[g], &level[g], w, h)
			}
		}
		for l in 0 ..< LANES {
			lvl0[l], wl[l] = bound_levels(level[l], im.top)
			blend |= wl[l] > 0
		}
	}
	out := (^[4]F_Lanes)(&regs[dst])
	// With layers (a sampler2DArray, so its coordinate has a layer at a + 2), each lane's layer.
	lay: [LANES]i32 = ---
	if im.n_layers > 1 {
		r := fl(&regs[a + 2])
		for l in 0 ..< LANES {
			lay[l] = bound_layer(r[l], im.ltop)
		}
	}
	if im.n_layers == 1 && im.n_levels == 1 && im.format == .F32 && im.wrap == {.Clamp, .Clamp} && im.levels[0].size[0] < 1 << 22 && im.levels[0].size[1] < 1 << 22 {
		switch im.comps {
		case 1:  if im.filter == .Nearest {bound_clamped_f32(im, u, v, out, 1, .Nearest)} else {bound_clamped_f32(im, u, v, out, 1, .Linear)}
		case 2:  if im.filter == .Nearest {bound_clamped_f32(im, u, v, out, 2, .Nearest)} else {bound_clamped_f32(im, u, v, out, 2, .Linear)}
		case:    if im.filter == .Nearest {bound_clamped_f32(im, u, v, out, 4, .Nearest)} else {bound_clamped_f32(im, u, v, out, 4, .Linear)}
		}
		return
	}
	switch im.format {
	case .F32:
		switch im.comps {
		case 1: bound_sample_filter(im, u, v, &lvl0, &wl, &lay, blend, out, .F32, 1)
		case 2: bound_sample_filter(im, u, v, &lvl0, &wl, &lay, blend, out, .F32, 2)
		case:   bound_sample_filter(im, u, v, &lvl0, &wl, &lay, blend, out, .F32, 4)
		}
	case .Rgba8:   bound_sample_filter(im, u, v, &lvl0, &wl, &lay, blend, out, .Rgba8, 4)
	case .Gray8:   bound_sample_filter(im, u, v, &lvl0, &wl, &lay, blend, out, .Gray8, 1)
	case .Gray16:  bound_sample_filter(im, u, v, &lvl0, &wl, &lay, blend, out, .Gray16, 1)
	case .Rgba16f: bound_sample_filter(im, u, v, &lvl0, &wl, &lay, blend, out, .Rgba16f, 4)
	}
}

// `ARR`: a sampler2DArray's texelFetch, its layer p.z at a + 2, in range
// below n_layers.
@(private)
fetch_bound_lanes :: proc "contextless" (im: ^Chan_Img, regs: []Reg, a, b, dst: i32, flip: bool, $F: Channel_Format, $C: int, $ARR: bool) #no_bounds_check {
	px, py, lb := &regs[a], &regs[a + 1], &regs[b]
	d0, d1, d2, d3 := fl(&regs[dst]), fl(&regs[dst + 1]), fl(&regs[dst + 2]), fl(&regs[dst + 3])
	n := u32(im.n_levels)
	for l in 0 ..< LANES {
		// A negative int lod is a huge uint one: out of range either way.
		ok := lb[l] < n
		lo := 0
		when ARR {
			z := regs[a + 2][l]
			ok &&= z < u32(im.n_layers)
			lo = int(z) * int(im.lpitch[lb[l] if ok else 0])
		}
		lv := &im.levels[lb[l] if ok else 0]
		x := i32(px[l])
		y := i32(u32(lv.max[1]) - py[l]) if flip else i32(py[l])
		c: [4]f32
		if ok && u32(x) <= u32(lv.max[0]) && u32(y) <= u32(lv.max[1]) {
			c = bound_texel(lv, x, y, lo, F, C)
		}
		d0[l], d1[l], d2[l], d3[l] = c[0], c[1], c[2], c[3]
	}
}

// texelFetch of a bound channel: its level's texel, 0 out of range.
// texelFetch of F32 with one level: level 0's texel where the lod is 0.
@(private)
fetch_bound_f32 :: proc "contextless" (im: ^Chan_Img, regs: []Reg, a, b, dst: i32, flip: bool, $C: int) #no_bounds_check {
	px, py, lb := &regs[a], &regs[a + 1], &regs[b]
	d0, d1, d2, d3 := fl(&regs[dst]), fl(&regs[dst + 1]), fl(&regs[dst + 2]), fl(&regs[dst + 3])
	lv := &im.levels[0]
	t := ([^]f32)(lv.ptr)
	pitch, mx, my := lv.pitch, lv.max[0], lv.max[1]
	for l in 0 ..< LANES {
		x := i32(px[l])
		y := i32(u32(my) - py[l]) if flip else i32(py[l])
		if lb[l] != 0 || u32(x) > u32(mx) || u32(y) > u32(my) {
			d0[l], d1[l], d2[l], d3[l] = 0, 0, 0, 0
			continue
		}
		off := int(y * pitch + x) * C
		when C == 1 {
			d0[l], d1[l], d2[l], d3[l] = t[off], 0, 0, 1
		} else when C == 2 {
			d0[l], d1[l], d2[l], d3[l] = t[off], t[off + 1], 0, 1
		} else {
			c := ((^[4]f32)(&t[off]))^
			d0[l], d1[l], d2[l], d3[l] = c[0], c[1], c[2], c[3]
		}
	}
}

@(private)
fetch_bound :: proc "contextless" (im: ^Chan_Img, regs: []Reg, a, b, dst: i32, flip, arr: bool) {
	if arr {
		fetch_bound_format(im, regs, a, b, dst, flip, true)
	} else {
		fetch_bound_format(im, regs, a, b, dst, flip, false)
	}
}

@(private)
fetch_bound_format :: proc "contextless" (im: ^Chan_Img, regs: []Reg, a, b, dst: i32, flip: bool, $ARR: bool) {
	switch im.format {
	case .F32:
		if im.n_levels == 1 && !ARR {
			switch im.comps {
			case 1: fetch_bound_f32(im, regs, a, b, dst, flip, 1)
			case 2: fetch_bound_f32(im, regs, a, b, dst, flip, 2)
			case:   fetch_bound_f32(im, regs, a, b, dst, flip, 4)
			}
			return
		}
		switch im.comps {
		case 1: fetch_bound_lanes(im, regs, a, b, dst, flip, .F32, 1, ARR)
		case 2: fetch_bound_lanes(im, regs, a, b, dst, flip, .F32, 2, ARR)
		case:   fetch_bound_lanes(im, regs, a, b, dst, flip, .F32, 4, ARR)
		}
	case .Rgba8:   fetch_bound_lanes(im, regs, a, b, dst, flip, .Rgba8, 4, ARR)
	case .Gray8:   fetch_bound_lanes(im, regs, a, b, dst, flip, .Gray8, 1, ARR)
	case .Gray16:  fetch_bound_lanes(im, regs, a, b, dst, flip, .Gray16, 1, ARR)
	case .Rgba16f: fetch_bound_lanes(im, regs, a, b, dst, flip, .Rgba16f, 4, ARR)
	}
}

// textureSize of a bound channel: level clamp(lod, 0, levels - 1)'s size,
// an int lod clamped signed and a uint one unsigned (Texture_Size's d), and
// its layers into dz (a sampler2DArray's; nil for a sampler2D).
@(private)
bound_size_lanes :: proc "contextless" (im: ^Chan_Img, uint_lod: bool, lods: [^]u32, dx, dy, dz: [^]u32, n: int) {
	if uint_lod {
		bound_size_lod_lanes(im, lods, dx, dy, n, true)
	} else {
		bound_size_lod_lanes(im, lods, dx, dy, n, false)
	}
	if dz != nil {
		for l in 0 ..< n {
			dz[l] = u32(im.n_layers)
		}
	}
}

@(private)
bound_size_lod_lanes :: proc "contextless" (im: ^Chan_Img, lods: [^]u32, dx, dy: [^]u32, n: int, $UNSIGNED: bool) #no_bounds_check {
	top := u32(im.n_levels - 1)
	for l in 0 ..< n {
		L := lods[l]
		when !UNSIGNED {
			L = u32(max(i32(L), 0))
		}
		lv := &im.levels[min(L, top)]
		dx[l], dy[l] = u32(lv.size[0]), u32(lv.size[1])
	}
}

// Of eight lanes (two 2x2 quads) with `run` running, the ones a fragment
// shader's texelFetch of a channel without an image fetches for (of those
// the batch has): every lane of a quad with a running lane, as a GPU's
// helper invocations fetch for the quad's derivatives.
@(private)
fetch_quad_lanes8 :: #force_inline proc "contextless" (run: u32) -> u32 {
	return u32(0xf if run & 0xf != 0 else 0) | u32(0xf0 if run & 0xf0 != 0 else 0)
}

// texelFetch of a channel without an image, eight lanes of texel (x, y),
// layer z (a sampler2DArray's) and lod L: what the sampler callback
// receives (uv, lod), and in `ok` bit l whether lane l is inside a full
// mip chain of the channel's size w x h (levels max(w >> l, 1) x max(h >>
// l, 1) while max(w, h) >> l is not 0), or its first `levels` levels when
// that is not 0 (Run_Options.channel_levels), its level and `size.z`
// layers; outside, texelFetch reads (0, 0, 0, 0) without a call, as GPUs
// and bound images do. Inside, uv is the texel's centre in its level
// (flipped: 1 - v), never NaN, and lod L. The interpreter and the JIT's
// helper both.
@(private)
texel_fetch_args8 :: #force_inline proc "contextless" (px, py, pz, pl: ^u32, size: [3]i32, levels: i32, arr, flip: bool, cu, cv, cl: ^f32) -> (ok: u32) {
	U :: #simd[8]u32
	I :: #simd[8]i32
	x, y, z, L := intrinsics.unaligned_load((^U)(px)), intrinsics.unaligned_load((^U)(py)), intrinsics.unaligned_load((^U)(pz)), intrinsics.unaligned_load((^U)(pl))
	w, h := u32(max(size.x, 0)), u32(max(size.y, 0))
	if w == 0 || h == 0 {
		return 0
	}
	// Shifts of 32 or more give 0.
	lw := intrinsics.simd_max(intrinsics.simd_shr(U(w), L), U(1))
	lh := intrinsics.simd_max(intrinsics.simd_shr(U(h), L), U(1))
	in_ := intrinsics.simd_lanes_ne(intrinsics.simd_shr(U(max(w, h)), L), U(0)) & intrinsics.simd_lanes_lt(x, lw) & intrinsics.simd_lanes_lt(y, lh)
	if levels > 0 {
		in_ &= intrinsics.simd_lanes_lt(L, U(u32(levels)))
	}
	if arr {
		in_ &= intrinsics.simd_lanes_lt(z, U(u32(max(size.z, 0))))
	}
	// Inside, x, y, L and the level's sides are below 2^31: converted as
	// ints.
	u := (cast(F8)transmute(I)x + 0.5) / cast(F8)transmute(I)lw
	v := (cast(F8)transmute(I)y + 0.5) / cast(F8)transmute(I)lh
	if flip {
		v = v * -1 + 1
	}
	intrinsics.unaligned_store((^F8)(cu), u)
	intrinsics.unaligned_store((^F8)(cv), v)
	intrinsics.unaligned_store((^F8)(cl), cast(F8)transmute(I)L)
	return intrinsics.simd_reduce_or(in_ & U{1, 2, 4, 8, 16, 32, 64, 128})
}

// A sample: dst = channel k at uv a, lod b (Texture_Bias: b plus the
// level the uv derivatives pick in the channel's texels, one per quad from
// its top row and left column, as GPUs' texture units pick it). Quads
// with a running lane (`exec`) are sampled; the others read 0.
@(private)
sample_texture :: proc(r: ^Runner, in_: Instr, exec: Lane_Mask) #no_bounds_check {
	regs := r.regs
	ch := int(in_.k)
	dst := in_.dst
	if im := &r.chan_imgs[ch]; im.ptr != nil {
		if in_.op == .Texel_Fetch {
			fetch_bound(im, regs, in_.a, in_.b, in_.dst, (in_.d & 1) != 0, in_.n == 3)
		} else {
			// Fields, not the Instr: an Instr passed by reference out of
			// line would make execute keep every instruction in memory.
			sample_bound(r, im, in_.op == .Texture_Bias, in_.a, in_.b, in_.dst)
		}
		return
	}
	for i in 0 ..< i32(4) {
		regs[dst + i] = {}
	}
	if r.opts.sampler == nil {
		return
	}
	size := channel_size(r, ch)
	running, have := transmute(Lane_Bits)exec, transmute(Lane_Bits)r.lanes
	if in_.op == .Texel_Fetch {
		px, py, pz, lb := &regs[in_.a], &regs[in_.a + 1], &regs[in_.a + 2 if in_.n == 3 else in_.a], &regs[in_.b]
		arr, flip := in_.n == 3, (in_.d & 1) != 0
		levels := r.opts.channel_levels[ch] if ch < len(r.opts.channel_levels) else 0
		for g := 0; g < LANES; g += 8 {
			// Compute has no quads: its running lanes fetch.
			run8 := u32(running >> uint(g)) & 0xff
			calls := run8
			if r.p.stage != .Compute {
				calls = fetch_quad_lanes8(run8) & u32(have >> uint(g))
			}
			if calls == 0 {
				continue
			}
			u, v, lod: [8]f32
			calls &= texel_fetch_args8(&px[g], &py[g], &pz[g], &lb[g], size, levels, arr, flip, &u[0], &v[0], &lod[0])
			for ; calls != 0; calls &= calls - 1 {
				k := int(intrinsics.count_trailing_zeros(calls))
				l := g + k
				c := r.opts.sampler(r.opts.user, ch, {u[k], v[k]}, lod[k], int(pz[l]) if arr else 0)
				for i in 0 ..< i32(4) {
					regs[dst + i][l] = fbits(c[i])
				}
			}
		}
		return
	}
	a, b := in_.a, in_.b
	// What the callback receives: canonical, with texture()'s level.
	u, v, lod: F_Lanes
	implicit := in_.op == .Texture_Bias && size.xy != {} && r.p.stage != .Compute
	for g := 0; g < LANES; g += 8 {
		sampler_args8(&regs[a][g], &regs[a + 1][g], &regs[b][g], size, implicit, &u[g], &v[g], &lod[g])
	}
	if r.p.stage == .Compute {
		if in_.n == 3 {
			z := &regs[a + 2]
			for l in 0 ..< LANES {
				if (running >> uint(l)) & 1 == 0 {
					continue
				}
				layer := int(f_to_i(fval(fn1(fval(z[l]), .Round_Even))))
				c := r.opts.sampler(r.opts.user, ch, {u[l], v[l]}, lod[l], layer)
				for i in 0 ..< i32(4) {
					regs[dst + i][l] = fbits(c[i])
				}
			}
		} else {
			for l in 0 ..< LANES {
				if (running >> uint(l)) & 1 == 0 {
					continue
				}
				c := r.opts.sampler(r.opts.user, ch, {u[l], v[l]}, lod[l], 0)
				for i in 0 ..< i32(4) {
					regs[dst + i][l] = fbits(c[i])
				}
			}
		}
		return
	}
	for o := 0; o < LANES; o += 4 {
		// A quad with a running lane samples all four lanes the batch has,
		// as a GPU quad's helpers do, so derivatives of the result are the
		// quad's like any temporary's; an idle quad samples nothing.
		if (running >> uint(o)) & 0xf == 0 {
			continue
		}
		for l in o ..< o + 4 {
			if (have >> uint(l)) & 1 == 0 {
				continue
			}
			// The layer as GPUs take it, rounded to nearest even (NaN and
			// out-of-range values as F_To_I takes them).
			layer := int(f_to_i(fval(fn1(fval(regs[a + 2][l]), .Round_Even)))) if in_.n == 3 else 0
			c := r.opts.sampler(r.opts.user, ch, {u[l], v[l]}, lod[l], layer)
			for i in 0 ..< i32(4) {
				regs[dst + i][l] = fbits(c[i])
			}
		}
	}
}

// What a sampler callback receives for eight lanes (two quads) of uv and
// lod: each lane's u, v and lod, plus, with `level` (texture() in a
// fragment shader, of a sized channel), the level its quad picks: 0.5 *
// log2 of the larger squared length, in texels, of the uv derivatives along
// the quad's top row and left column. Every NaN is 0x7fc00000
// (`canonical`). The interpreter and the JIT's helpers both.
@(private)
sampler_args8 :: #force_inline proc "contextless" (pu, pv, plod: ^u32, size: [3]i32, level: bool, cu, cv, cl: ^f32) {
	u, v, lod := intrinsics.unaligned_load((^F8)(pu)), intrinsics.unaligned_load((^F8)(pv)), intrinsics.unaligned_load((^F8)(plod))
	l := lod
	if level {
		l = lod + quad_levels8(u, v, f32(size.x), f32(size.y))
	}
	intrinsics.unaligned_store((^F8)(cu), canon8(u))
	intrinsics.unaligned_store((^F8)(cv), canon8(v))
	intrinsics.unaligned_store((^F8)(cl), canon8(l))
}

// 0.5 * log2(m), to within 2e-6 plus the result's own rounding to f32
// (half an ulp, 3.8e-6 from |level| 64; checked for every m), exact where
// m is a power of 4: m = 2^k * (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2))
// and log2(1 + f) as f times a degree-5 minimax polynomial q, in f32 (a
// level needs neither f64 nor a correctly rounded log2, and is waited for
// by every callback): 0.5 * f * q(f) on [sqrt(2)/2 - 1, sqrt(2) - 1],
// minimax absolute error 1.85e-6.
// log2(+-0) = -inf, of +inf +inf, of NaN or m < 0 NaN.
@(private)
quad_level8 :: #force_inline proc "contextless" (m: F8) -> F8 {
	W :: #simd[8]u32
	I :: #simd[8]i32
	C0, C1, C2 :: f32(0h3FB8AA72), f32(0hBF38A0F9), f32(0h3EF5A9EC)
	C3, C4, C5 :: f32(0hBEBBA56E), f32(0h3EA2EAB7), f32(0hBE531AE5)
	sub := intrinsics.simd_lanes_lt(m, F8(1.1754943508222875e-38))
	ms := intrinsics.simd_select(sub, m * 16777216, m)
	b := transmute(W)ms
	k := intrinsics.simd_shr(transmute(I)(b - W(0x3f35_04f3)), W(23))
	f := transmute(F8)(b - transmute(W)intrinsics.simd_shl(k, W(23))) - 1
	f2 := f * f
	q := (C0 + f * C1) + f2 * ((C2 + f * C3) + f2 * (C4 + f * C5))
	e := cast(F8)k - intrinsics.simd_select(sub, F8(24), F8(0))
	l := 0.5 * (e + f * q)
	l = intrinsics.simd_select(intrinsics.simd_lanes_eq(m, F8(0)), F8(fval(0xff80_0000)), l)
	l = intrinsics.simd_select(intrinsics.simd_lanes_eq(m, F8(fval(0x7f80_0000))), m, l)
	return intrinsics.simd_select(intrinsics.simd_lanes_ge(m, F8(0)), l, F8(fval(0x7fc0_0000)))
}

// Channel ch's size from the host (0x0 without one).
@(private)
channel_size :: #force_inline proc "contextless" (r: ^Runner, ch: int) -> [3]i32 {
	return r.opts.channel_sizes[ch] if ch >= 0 && ch < len(r.opts.channel_sizes) else {}
}

// The determinant or the inverse of each lane's N x N matrix, read from
// column-major registers at `a`.
@(private)
lane_matrices :: proc "contextless" (regs: []Reg, in_: Instr, $N: int) #no_bounds_check {
	for l in 0 ..< LANES {
		m: matrix[N, N]f32
		for c in 0 ..< N {
			for r in 0 ..< N {
				m[r, c] = fval(regs[in_.a + i32(c * N + r)][l])
			}
		}
		if in_.op == .Determinant {
			regs[in_.dst][l] = fbits(linalg.determinant(m))
			continue
		}
		inv := linalg.inverse(m)
		for c in 0 ..< N {
			for r in 0 ..< N {
				regs[in_.dst + i32(c * N + r)][l] = fbits(inv[r, c])
			}
		}
	}
}

@(private)
image_texel_bytes :: #force_inline proc "contextless" (f: Image_Format) -> int {
	#partial switch f {
	case .Rgba8, .R32f, .R32ui, .R32i:
		return 4
	case .Rgba16f:
		return 8
	case .Rgba32f:
		return 16
	}
	return 0
}

// IEEE-754 binary32 -> binary16, round-to-nearest-even.
@(private)
f32_to_f16 :: proc "contextless" (x: f32) -> u16 {
	b := transmute(u32)x
	sign := u16((b >> 16) & 0x8000)
	abs_b := b & 0x7fff_ffff
	if abs_b > 0x7f80_0000 {
		return sign | 0x7e00
	}
	if abs_b >= 0x4780_0000 {
		return sign | 0x7c00
	}
	exp32 := int(abs_b >> 23)
	mant32 := abs_b & 0x7f_ffff
	exp16 := exp32 - 112
	if exp16 >= 1 {
		lsb := (mant32 >> 13) & 1
		rounded := (mant32 + 0xfff + lsb) >> 13
		return sign + (u16(exp16) << 10) + u16(rounded)
	}
	if exp16 < -10 {
		return sign
	}
	m := mant32 | 0x80_0000
	shift := uint(14 - exp16)
	lsb := (m >> shift) & 1
	half_minus_1 := (u32(1) << (shift - 1)) - 1
	return sign | u16((m + half_minus_1 + lsb) >> shift)
}

@(private)
f16_to_f32 :: proc "contextless" (h: u16) -> f32 {
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
		if mant == 0 {
			return transmute(f32)(sign | 0x7f80_0000)
		}
		return transmute(f32)(sign | 0x7fc0_0000 | (mant << 13))
	}
	return transmute(f32)(sign | ((exp + 112) << 23) | (mant << 13))
}

// Exact IEEE-754 single-precision `f32(b) / 255.0` for every byte `0..=255`.
@(private, rodata)
UNORM8_TO_F32 := [256]f32 {
	0.0 / 255.0, 1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 4.0 / 255.0, 5.0 / 255.0, 6.0 / 255.0, 7.0 / 255.0,
	8.0 / 255.0, 9.0 / 255.0, 10.0 / 255.0, 11.0 / 255.0, 12.0 / 255.0, 13.0 / 255.0, 14.0 / 255.0, 15.0 / 255.0,
	16.0 / 255.0, 17.0 / 255.0, 18.0 / 255.0, 19.0 / 255.0, 20.0 / 255.0, 21.0 / 255.0, 22.0 / 255.0, 23.0 / 255.0,
	24.0 / 255.0, 25.0 / 255.0, 26.0 / 255.0, 27.0 / 255.0, 28.0 / 255.0, 29.0 / 255.0, 30.0 / 255.0, 31.0 / 255.0,
	32.0 / 255.0, 33.0 / 255.0, 34.0 / 255.0, 35.0 / 255.0, 36.0 / 255.0, 37.0 / 255.0, 38.0 / 255.0, 39.0 / 255.0,
	40.0 / 255.0, 41.0 / 255.0, 42.0 / 255.0, 43.0 / 255.0, 44.0 / 255.0, 45.0 / 255.0, 46.0 / 255.0, 47.0 / 255.0,
	48.0 / 255.0, 49.0 / 255.0, 50.0 / 255.0, 51.0 / 255.0, 52.0 / 255.0, 53.0 / 255.0, 54.0 / 255.0, 55.0 / 255.0,
	56.0 / 255.0, 57.0 / 255.0, 58.0 / 255.0, 59.0 / 255.0, 60.0 / 255.0, 61.0 / 255.0, 62.0 / 255.0, 63.0 / 255.0,
	64.0 / 255.0, 65.0 / 255.0, 66.0 / 255.0, 67.0 / 255.0, 68.0 / 255.0, 69.0 / 255.0, 70.0 / 255.0, 71.0 / 255.0,
	72.0 / 255.0, 73.0 / 255.0, 74.0 / 255.0, 75.0 / 255.0, 76.0 / 255.0, 77.0 / 255.0, 78.0 / 255.0, 79.0 / 255.0,
	80.0 / 255.0, 81.0 / 255.0, 82.0 / 255.0, 83.0 / 255.0, 84.0 / 255.0, 85.0 / 255.0, 86.0 / 255.0, 87.0 / 255.0,
	88.0 / 255.0, 89.0 / 255.0, 90.0 / 255.0, 91.0 / 255.0, 92.0 / 255.0, 93.0 / 255.0, 94.0 / 255.0, 95.0 / 255.0,
	96.0 / 255.0, 97.0 / 255.0, 98.0 / 255.0, 99.0 / 255.0, 100.0 / 255.0, 101.0 / 255.0, 102.0 / 255.0, 103.0 / 255.0,
	104.0 / 255.0, 105.0 / 255.0, 106.0 / 255.0, 107.0 / 255.0, 108.0 / 255.0, 109.0 / 255.0, 110.0 / 255.0, 111.0 / 255.0,
	112.0 / 255.0, 113.0 / 255.0, 114.0 / 255.0, 115.0 / 255.0, 116.0 / 255.0, 117.0 / 255.0, 118.0 / 255.0, 119.0 / 255.0,
	120.0 / 255.0, 121.0 / 255.0, 122.0 / 255.0, 123.0 / 255.0, 124.0 / 255.0, 125.0 / 255.0, 126.0 / 255.0, 127.0 / 255.0,
	128.0 / 255.0, 129.0 / 255.0, 130.0 / 255.0, 131.0 / 255.0, 132.0 / 255.0, 133.0 / 255.0, 134.0 / 255.0, 135.0 / 255.0,
	136.0 / 255.0, 137.0 / 255.0, 138.0 / 255.0, 139.0 / 255.0, 140.0 / 255.0, 141.0 / 255.0, 142.0 / 255.0, 143.0 / 255.0,
	144.0 / 255.0, 145.0 / 255.0, 146.0 / 255.0, 147.0 / 255.0, 148.0 / 255.0, 149.0 / 255.0, 150.0 / 255.0, 151.0 / 255.0,
	152.0 / 255.0, 153.0 / 255.0, 154.0 / 255.0, 155.0 / 255.0, 156.0 / 255.0, 157.0 / 255.0, 158.0 / 255.0, 159.0 / 255.0,
	160.0 / 255.0, 161.0 / 255.0, 162.0 / 255.0, 163.0 / 255.0, 164.0 / 255.0, 165.0 / 255.0, 166.0 / 255.0, 167.0 / 255.0,
	168.0 / 255.0, 169.0 / 255.0, 170.0 / 255.0, 171.0 / 255.0, 172.0 / 255.0, 173.0 / 255.0, 174.0 / 255.0, 175.0 / 255.0,
	176.0 / 255.0, 177.0 / 255.0, 178.0 / 255.0, 179.0 / 255.0, 180.0 / 255.0, 181.0 / 255.0, 182.0 / 255.0, 183.0 / 255.0,
	184.0 / 255.0, 185.0 / 255.0, 186.0 / 255.0, 187.0 / 255.0, 188.0 / 255.0, 189.0 / 255.0, 190.0 / 255.0, 191.0 / 255.0,
	192.0 / 255.0, 193.0 / 255.0, 194.0 / 255.0, 195.0 / 255.0, 196.0 / 255.0, 197.0 / 255.0, 198.0 / 255.0, 199.0 / 255.0,
	200.0 / 255.0, 201.0 / 255.0, 202.0 / 255.0, 203.0 / 255.0, 204.0 / 255.0, 205.0 / 255.0, 206.0 / 255.0, 207.0 / 255.0,
	208.0 / 255.0, 209.0 / 255.0, 210.0 / 255.0, 211.0 / 255.0, 212.0 / 255.0, 213.0 / 255.0, 214.0 / 255.0, 215.0 / 255.0,
	216.0 / 255.0, 217.0 / 255.0, 218.0 / 255.0, 219.0 / 255.0, 220.0 / 255.0, 221.0 / 255.0, 222.0 / 255.0, 223.0 / 255.0,
	224.0 / 255.0, 225.0 / 255.0, 226.0 / 255.0, 227.0 / 255.0, 228.0 / 255.0, 229.0 / 255.0, 230.0 / 255.0, 231.0 / 255.0,
	232.0 / 255.0, 233.0 / 255.0, 234.0 / 255.0, 235.0 / 255.0, 236.0 / 255.0, 237.0 / 255.0, 238.0 / 255.0, 239.0 / 255.0,
	240.0 / 255.0, 241.0 / 255.0, 242.0 / 255.0, 243.0 / 255.0, 244.0 / 255.0, 245.0 / 255.0, 246.0 / 255.0, 247.0 / 255.0,
	248.0 / 255.0, 249.0 / 255.0, 250.0 / 255.0, 251.0 / 255.0, 252.0 / 255.0, 253.0 / 255.0, 254.0 / 255.0, 255.0 / 255.0,
}

// run_compute validates opts.images up front, so width/height/stride/data are
// guaranteed in-bounds for every (x, y) in [0, width) x [0, height).
//
// The image, buffer, shared-memory and atomic kernels below are generic over
// the lane count: the interpreter instantiates them with 64-lane registers and
// the JIT's helpers with 8-lane slots, so both apply lanes in ascending order
// with the same rules.
@(private)
load_image :: proc "contextless" (r: ^Runner, in_: Instr, exec: Lane_Mask) #no_bounds_check {
	image_load_lanes(raw_data(r.regs), &r.opts.images[in_.k], in_, transmute(Lane_Bits)exec)
}

@(private)
store_image :: proc "contextless" (r: ^Runner, in_: Instr, exec: Lane_Mask) #no_bounds_check {
	image_store_lanes(raw_data(r.regs), &r.opts.images[in_.k], in_, transmute(Lane_Bits)exec)
}

// dst (4 registers) = image texels at the lanes' (a, a + 1) in `ebits`, 0 in
// the other lanes and out of bounds.
@(private)
image_load_lanes :: #force_inline proc "contextless" (regs: [^][$N]u32, im: ^Storage_Image, in_: Instr, ebits: $B) #no_bounds_check {
	dst := in_.dst
	for i in 0 ..< i32(4) {
		regs[dst + i] = {}
	}
	if ebits == 0 || im.width <= 0 || im.height <= 0 {
		return
	}
	#partial switch im.format {
	case .Rgba8:   image_load_fmt(regs, im, in_, ebits, Image_Format.Rgba8)
	case .Rgba16f: image_load_fmt(regs, im, in_, ebits, Image_Format.Rgba16f)
	case .Rgba32f: image_load_fmt(regs, im, in_, ebits, Image_Format.Rgba32f)
	case .R32f:    image_load_fmt(regs, im, in_, ebits, Image_Format.R32f)
	case .R32ui:   image_load_fmt(regs, im, in_, ebits, Image_Format.R32ui)
	case .R32i:    image_load_fmt(regs, im, in_, ebits, Image_Format.R32i)
	}
}

@(private)
image_load_fmt :: #force_inline proc "contextless" (regs: [^][$N]u32, im: ^Storage_Image, in_: Instr, ebits: $B, $F: Image_Format) #no_bounds_check {
	TB :: 4 when F == .Rgba8 || F == .R32f || F == .R32ui || F == .R32i else (8 when F == .Rgba16f else 16)
	w, h := u32(im.width), u32(im.height)
	row_stride := im.stride if im.stride != 0 else im.width * TB
	dst := in_.dst
	cx, cy := &regs[in_.a], &regs[in_.a + 1]
	data := raw_data(im.data)
	for l in 0 ..< N {
		if (ebits >> uint(l)) & 1 == 0 {
			continue
		}
		x, y := cx[l], cy[l]
		if x >= w || y >= h {
			continue
		}
		p := data[int(y) * row_stride + int(x) * TB:]
		when F == .Rgba8 {
			lut := &UNORM8_TO_F32
			for i in 0 ..< i32(4) {
				regs[dst + i][l] = fbits(lut[p[i]])
			}
		} else when F == .Rgba16f {
			for i in 0 ..< i32(4) {
				hv := u16(p[i * 2]) | (u16(p[i * 2 + 1]) << 8)
				regs[dst + i][l] = fbits(f16_to_f32(hv))
			}
		} else when F == .Rgba32f {
			for i in 0 ..< i32(4) {
				regs[dst + i][l] = intrinsics.unaligned_load((^u32)(&p[i * 4]))
			}
		} else {
			// Single-channel formats fill .gba = (0, 0, 1).
			regs[dst][l] = intrinsics.unaligned_load((^u32)(p))
			regs[dst + 3][l] = 0x3f80_0000 when F == .R32f else 1
		}
	}
}

// Image texels at the lanes' (a, a + 1) in `ebits` = b (4 registers); out of
// bounds dropped.
@(private)
image_store_lanes :: #force_inline proc "contextless" (regs: [^][$N]u32, im: ^Storage_Image, in_: Instr, ebits: $B) #no_bounds_check {
	if ebits == 0 || im.width <= 0 || im.height <= 0 {
		return
	}
	#partial switch im.format {
	case .Rgba8:   image_store_fmt(regs, im, in_, ebits, Image_Format.Rgba8)
	case .Rgba16f: image_store_fmt(regs, im, in_, ebits, Image_Format.Rgba16f)
	case .Rgba32f: image_store_fmt(regs, im, in_, ebits, Image_Format.Rgba32f)
	case .R32f, .R32ui, .R32i: image_store_fmt(regs, im, in_, ebits, Image_Format.R32ui)
	}
}

@(private)
image_store_fmt :: #force_inline proc "contextless" (regs: [^][$N]u32, im: ^Storage_Image, in_: Instr, ebits: $B, $F: Image_Format) #no_bounds_check {
	TB :: 4 when F == .Rgba8 || F == .R32ui else (8 when F == .Rgba16f else 16)
	w, h := u32(im.width), u32(im.height)
	row_stride := im.stride if im.stride != 0 else im.width * TB
	b := in_.b
	cx, cy := &regs[in_.a], &regs[in_.a + 1]
	data := raw_data(im.data)
	for l in 0 ..< N {
		if (ebits >> uint(l)) & 1 == 0 {
			continue
		}
		x, y := cx[l], cy[l]
		if x >= w || y >= h {
			continue
		}
		p := data[int(y) * row_stride + int(x) * TB:]
		when F == .Rgba8 {
			for i in 0 ..< i32(4) {
				cv := fval(regs[b + i][l])
				if !(cv > 0) {
					cv = 0
				}
				p[i] = u8(math.round(min(cv, 1) * 255))
			}
		} else when F == .Rgba16f {
			for i in 0 ..< i32(4) {
				hv := f32_to_f16(fval(regs[b + i][l]))
				p[i * 2] = u8(hv)
				p[i * 2 + 1] = u8(hv >> 8)
			}
		} else when F == .Rgba32f {
			for i in 0 ..< i32(4) {
				intrinsics.unaligned_store((^u32)(&p[i * 4]), regs[b + i][l])
			}
		} else {
			intrinsics.unaligned_store((^u32)(p), regs[b][l])
		}
	}
}

// Buffer k's words a * c + d + i = b + i (i < n) in the lanes of `ebits`,
// in ascending lane order; words past the buffer are dropped.
@(private)
store_buf_lanes :: #force_inline proc "contextless" (regs: [^][$N]u32, words: []u32, in_: Instr, ebits: $B) #no_bounds_check {
	idx := &regs[in_.a]
	n, b := i32(in_.n), in_.b
	stride, word := u32(in_.c), u32(in_.d)
	for l in 0 ..< N {
		if (ebits >> uint(l)) & 1 == 0 {
			continue
		}
		base_w := idx[l] * stride + word
		for i in 0 ..< n {
			w := base_w + u32(i)
			if uint(w) < uint(len(words)) {
				words[w] = regs[b + i][l]
			}
		}
	}
}

// shared[c + (k != 0 ? b-lane : 0) + i] = a + i in the lanes of `ebits`, in
// ascending lane order (the lowering clamps dynamic offsets into range).
@(private)
store_shared_lanes :: #force_inline proc "contextless" (regs: [^][$N]u32, sh: [^]u32, in_: Instr, ebits: $B) #no_bounds_check {
	n := i32(in_.n)
	a, c := in_.a, in_.c
	if in_.k == 0 {
		for l in 0 ..< N {
			if (ebits >> uint(l)) & 1 != 0 {
				for i in 0 ..< n {
					sh[c + i] = regs[a + i][l]
				}
			}
		}
		return
	}
	offs := &regs[in_.b]
	for l in 0 ..< N {
		if (ebits >> uint(l)) & 1 == 0 {
			continue
		}
		off := c + i32(offs[l])
		for i in 0 ..< n {
			sh[off + i] = regs[a + i][l]
		}
	}
}

// dst = the old word of atomic op k at shared word a-lane with b (and c),
// lane by lane in ascending order; 0 in the other lanes.
@(private)
atomic_shared_lanes :: #force_inline proc "contextless" (regs: [^][$N]u32, sh: [^]u32, in_: Instr, ebits: $B) #no_bounds_check {
	dst, a, b, c := in_.dst, in_.a, in_.b, in_.c
	regs[dst] = {}
	if ebits == 0 {
		return
	}
	aop := Atomic_Op(in_.k)
	for l in 0 ..< N {
		if (ebits >> uint(l)) & 1 != 0 {
			regs[dst][l] = atomic_word_shared(&sh[regs[a][l]], aop, regs[b][l], regs[c][l])
		}
	}
}

// The same on buffer words with host atomics; 0 past the buffer.
@(private)
atomic_buf_lanes :: #force_inline proc "contextless" (regs: [^][$N]u32, words: []u32, in_: Instr, ebits: $B) #no_bounds_check {
	dst, a, b, c := in_.dst, in_.a, in_.b, in_.c
	regs[dst] = {}
	if ebits == 0 {
		return
	}
	aop := Atomic_Op(in_.k)
	for l in 0 ..< N {
		if (ebits >> uint(l)) & 1 == 0 {
			continue
		}
		w := regs[a][l]
		if uint(w) < uint(len(words)) {
			regs[dst][l] = atomic_word_buf(&words[w], aop, regs[b][l], regs[c][l])
		}
	}
}
