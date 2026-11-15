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
import "core:slice"
import "core:strings"
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

// The layer quad: position in target pixels, uv; counter-clockwise on
// screen, so front-facing under SDL_GPU's default front face (as the CPU's
// gl_FrontFacing is).
Vertex :: struct {
	pos, uv: [2]f32,
}

QUAD := [6]Vertex{{{0, 0}, {0, 0}}, {{0, H}, {0, 1}}, {{W, H}, {1, 1}}, {{0, 0}, {0, 0}}, {{W, H}, {1, 1}}, {{W, 0}, {1, 0}}}

make_data :: proc() {
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
// rows within 0.02 except the components a narrow image fills (exact).
// Writes its PNGs (values mapped from [-2, 2]); a detail line and the
// verdict.
bound_channels_compare :: proc(m: ^oksl.Module, block: []u8, gpu_f: [][4]f32, gpu_tag, out_dir: string, scratch: ^oksl.Scratch) -> (string, bool) {
	cpu_f := make([][4]f32, W * H, context.temp_allocator)
	cimgs := bound_channel_images()
	oksl.run_f32(&m.program, block, W, H, cpu_f, scratch, {channel_images = cimgs[:]})
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
					tol: f32 = 0 if exact_fill else 0.02
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
	return fmt.tprintf("nearest err %.6f, linear err %.6f, mismatches %d", near_err, lin_err, mismatches), mismatches == 0
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
