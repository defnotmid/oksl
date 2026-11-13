// Renders every shader in testdata/ (.sksl, .glsl Shadertoy code and .frag
// GLSL fragment shaders, by oksl.SOURCE_EXTENSIONS) with SDL_GPU (the
// generated SPIR-V, or MSL on Metal, drawn through oksl.LAYER_VERTEX_GLSL,
// whose uv a .frag reads at location 0) and with the CPU interpreter, and
// compares the colour at location 0: sampler2Ds read a checker texture,
// sampler2DArrays LAYERS copies of it with their colours rotated, and
// every storage buffer the test words. It also checks a tap of the layer
// vertex shader (Options.taps), its position drawn as colour, and a vertex
// shader placing the quad's corners by gl_VertexID from a storage buffer,
// against run_vertex's, and gl_SampleMask resolved from 4x MSAA against
// run_fragment's masks.
// Writes <name>_gpu.png, <name>_cpu.png and <name>_diff.png.
//
//     odin run tools/gpu_parity -- [out_dir] [testdata_dir]
package gpu_parity

import "core:fmt"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"
import "core:thread"
import oksl "../.."
import sdl "vendor:sdl3"
import stbi "vendor:stb/image"

W :: 256
H :: 144
TIME :: 1.5
TEX :: 64
LAYERS :: 3

// The storage buffers' words: floats in [0, 1], which shaders read as
// floats or as bits.
WORDS :: 4096
test_words: [WORDS]u32

make_words :: proc() {
	for &w, i in test_words {
		w = transmute(u32)(f32((i * 37 + 11) % 256) / 255)
	}
}

// The channel texture: a colour checker and its mip chain, each level the
// 2x2 box average of the one above, in 8 bits (so both sides see the same
// values).
LEVELS :: 7
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

make_bound_textures :: proc() {
	for y in 0 ..< BOUND_R_H {
		for x in 0 ..< BOUND_R_W {
			bound_r_texels[y * BOUND_R_W + x] = f32(((x * 5 + y * 3) % 25) - 8) * 0.125
		}
	}
	for y in 0 ..< BOUND_RG_H {
		for x in 0 ..< BOUND_RG_W {
			bound_rg_texels[y * BOUND_RG_W + x] = {
				f32(((x * 3 + y * 7 + 2) % 25) - 8) * 0.125,
				f32(((x * 7 + y * 5 + 9) % 25) - 8) * 0.125,
			}
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
}

// Where level l starts in texture_pixels.
level_offset :: proc(l: int) -> int {
	o := 0
	for i in 0 ..< l {
		o += (TEX >> uint(i)) * (TEX >> uint(i)) * 4
	}
	return o
}

make_texture :: proc() {
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

// Layer `layer` of the array texture: the checker with its colours
// rotated `layer` places (layer 0 is the 2D texture).
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

// Trilinear and repeating, the level clamped to the chain and the layer to
// the array: what the GPU sampler below does.
cpu_sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
	ly := clamp(layer, 0, LAYERS - 1)
	if !(lod > 0) {
		return bilinear(0, uv, ly)
	}
	level := min(lod, LEVELS - 1)
	l0 := int(level)
	l1 := min(l0 + 1, LEVELS - 1)
	t := level - f32(l0)
	return bilinear(l0, uv, ly) * (1 - t) + bilinear(l1, uv, ly) * t
}

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

Gpu :: struct {
	device:         ^sdl.GPUDevice,
	msl:            bool,
	vert:           ^sdl.GPUShader,
	target:         ^sdl.GPUTexture,
	target_f32:     ^sdl.GPUTexture,
	tex:            ^sdl.GPUTexture,
	layers:         ^sdl.GPUTexture,
	tex_r32f:       ^sdl.GPUTexture,
	tex_rg32f:      ^sdl.GPUTexture,
	tex_rgba32f:    ^sdl.GPUTexture,
	sampler:        ^sdl.GPUSampler,
	sampler_near:   ^sdl.GPUSampler,
	sampler_linear: ^sdl.GPUSampler,
	quad:           ^sdl.GPUBuffer,
	words:          ^sdl.GPUBuffer,
}

main :: proc() {
	args := os.args[1:]
	out_dir := args[0] if len(args) > 0 else "parity_out"
	data_dir := args[1] if len(args) > 1 else "testdata"
	os.make_directory_all(out_dir)
	make_texture()
	make_bound_textures()
	make_words()

	g: Gpu
	if !gpu_init(&g) {
		fmt.eprintln("no SDL_GPU device:", sdl.GetError())
		os.exit(2)
	}
	fmt.println("SDL_GPU driver:", sdl.GetGPUDeviceDriver(g.device))

	infos, _ := os.read_all_directory_by_path(data_dir, context.allocator)
	names := make([dynamic]string)
	comp_names := make([dynamic]string)
	for fi in infos {
		if o, known := oksl.options_of_path(fi.name); known {
			if o.stage == .Fragment {
				append(&names, fi.fullpath)
			} else if o.stage == .Compute {
				append(&comp_names, fi.fullpath)
			}
		}
	}
	slice.sort(names[:])
	slice.sort(comp_names[:])

	failed := 0
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	for path in names {
		name := filepath.stem(path)
		src, _ := os.read_entire_file(path, context.allocator)
		m: ^oksl.Module
		errors: []oksl.Error
		ok: bool
		options, _ := oksl.options_of_path(path)
		m, errors, ok = oksl.compile(string(src), options)
		if !ok {
			fmt.printfln("%-16s compile error:\n%s", name, oksl.format_errors(errors, context.temp_allocator))
			failed += 1
			continue
		}
		block := uniform_block(m)
		if name == "bound_channels" {
			detail, bc_ok := bound_channels_check(&g, m, block, out_dir, &scratch)
			if !bc_ok {
				failed += 1
			}
			fmt.printfln("%-16s %s  %s", name, detail, "ok" if bc_ok else "MISMATCH")
			oksl.destroy(m)
			continue
		}
		gpu_px, gpu_ok := gpu_render(&g, m, block)
		if !gpu_ok {
			fmt.printfln("%-16s GPU render failed: %s", name, sdl.GetError())
			failed += 1
			oksl.destroy(m)
			continue
		}
		cpu_px := make([]u8, W * H * 4)
		for i in 0 ..< W * H {
			cpu_px[i * 4 + 3] = 255
		}
		sizes: [oksl.SAMPLER_COUNT][3]i32
		for &s, ch in sizes {
			s = {TEX, TEX, LAYERS if ch in m.channel_arrays else 1}
		}
		buffers := make([][]u32, len(m.storage_buffers), context.temp_allocator)
		for &b in buffers {
			b = test_words[:]
		}
		oksl.run(&m.program, block, W, H, cpu_px, W * 4, &scratch, {sampler = cpu_sampler, channel_sizes = sizes[:], buffers = buffers})

		max_d, sum, over2, over8 := 0, 0, 0, 0
		diff := make([]u8, W * H * 4)
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
		// Within 3/255 everywhere, or differences confined to a few pixels
		// where a threshold (step, comparisons) meets transcendental results
		// that differ by ulps between the GPU and the CPU.
		verdict := "ok"
		if max_d > 3 {
			verdict = "ok (threshold edges)" if over8 <= W * H / 100 && mean < 1 else "MISMATCH"
		}
		if verdict == "MISMATCH" {
			failed += 1
		}
		fmt.printfln("%-16s max diff %d, mean %.4f, pixels >2: %d, >8: %d  %s", name, max_d, mean, over2, over8, verdict)
		write_png(fmt.tprintf("%s/%s_gpu.png", out_dir, name), gpu_px)
		write_png(fmt.tprintf("%s/%s_cpu.png", out_dir, name), cpu_px)
		write_png(fmt.tprintf("%s/%s_diff.png", out_dir, name), diff)
		oksl.destroy(m)
	}
	max_d, tap_ok := tap_check(&g)
	if !tap_ok || max_d > 2 {
		failed += 1
	}
	fmt.printfln("%-16s max diff %d  %s", "layer.vert tap", max_d, "ok" if tap_ok && max_d <= 2 else "MISMATCH")
	id_d, id_ok := vertex_id_check(&g)
	if !id_ok || id_d > 2 {
		failed += 1
	}
	fmt.printfln("%-16s max diff %d  %s", "gl_VertexID", id_d, "ok" if id_ok && id_d <= 2 else "MISMATCH")
	mask_d, mask_ok := sample_mask_check(&g)
	if !mask_ok || mask_d > 2 {
		failed += 1
	}
	fmt.printfln("%-16s max diff %d  %s", "gl_SampleMask", mask_d, "ok" if mask_ok && mask_d <= 2 else "MISMATCH")
	for path in comp_names {
		name := filepath.base(path)
		detail, comp_ok := compute_file_check(&g, path, &scratch)
		if !comp_ok {
			failed += 1
		}
		fmt.printfln("%-16s %s  %s", name, detail, "ok" if comp_ok else "MISMATCH")
	}
	fmt_detail, fmt_ok := image_formats_check(&g, &scratch)
	if !fmt_ok {
		failed += 1
	}
	fmt.printfln("%-16s %s  %s", "image_formats", fmt_detail, "ok" if fmt_ok else "MISMATCH")
	checks := len(names) + 3 + len(comp_names) + 1
	fmt.printfln("%d/%d shaders match (images in %s)", checks - failed, checks, out_dir)
	if failed > 0 {
		os.exit(1)
	}
}

write_png :: proc(path: string, px: []u8) {
	stbi.write_png(strings.clone_to_cstring(path, context.temp_allocator), W, H, 4, raw_data(px), W * 4)
}

gpu_init :: proc(g: ^Gpu) -> bool {
	if !sdl.Init({.VIDEO}) {
		sdl.SetHint(sdl.HINT_VIDEO_DRIVER, "offscreen")
		if !sdl.Init({.VIDEO}) {
			return false
		}
	}
	g.device = sdl.CreateGPUDevice({.SPIRV, .MSL}, false, nil)
	if g.device == nil {
		return false
	}
	g.msl = .SPIRV not_in sdl.GetGPUShaderFormats(g.device)
	m, errors, ok := oksl.compile(oksl.LAYER_VERTEX_GLSL, {profile = .Glsl, stage = .Vertex})
	defer oksl.destroy(m)
	if !ok {
		fmt.eprint(oksl.format_errors(errors))
		return false
	}
	vert := sdl.GPUShaderCreateInfo {
		code_size = len(m.spirv) * 4,
		code = ([^]u8)(raw_data(m.spirv)),
		entrypoint = oksl.SPIRV_ENTRY,
		format = {.SPIRV},
		stage = .VERTEX,
		num_uniform_buffers = 1,
	}
	if g.msl {
		vert.code_size = len(m.msl)
		vert.code = raw_data(m.msl)
		vert.entrypoint = oksl.MSL_ENTRY
		vert.format = {.MSL}
	}
	g.vert = sdl.CreateGPUShader(g.device, vert)
	g.target = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R8G8B8A8_UNORM, usage = {.COLOR_TARGET}, width = W, height = H, layer_count_or_depth = 1, num_levels = 1})
	g.target_f32 = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R32G32B32A32_FLOAT, usage = {.COLOR_TARGET}, width = W, height = H, layer_count_or_depth = 1, num_levels = 1})
	g.tex = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R8G8B8A8_UNORM, usage = {.SAMPLER}, width = TEX, height = TEX, layer_count_or_depth = 1, num_levels = LEVELS})
	g.layers = sdl.CreateGPUTexture(g.device, {type = .D2_ARRAY, format = .R8G8B8A8_UNORM, usage = {.SAMPLER}, width = TEX, height = TEX, layer_count_or_depth = LAYERS, num_levels = LEVELS})
	g.tex_r32f = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R32_FLOAT, usage = {.SAMPLER}, width = BOUND_R_W, height = BOUND_R_H, layer_count_or_depth = 1, num_levels = 1})
	g.tex_rg32f = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R32G32_FLOAT, usage = {.SAMPLER}, width = BOUND_RG_W, height = BOUND_RG_H, layer_count_or_depth = 1, num_levels = 1})
	g.tex_rgba32f = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R32G32B32A32_FLOAT, usage = {.SAMPLER}, width = BOUND_RGBA_W, height = BOUND_RGBA_H, layer_count_or_depth = 1, num_levels = 1})
	g.sampler = sdl.CreateGPUSampler(g.device, {
		min_filter = .LINEAR,
		mag_filter = .LINEAR,
		mipmap_mode = .LINEAR,
		address_mode_u = .REPEAT,
		address_mode_v = .REPEAT,
		address_mode_w = .REPEAT,
		max_lod = LEVELS - 1,
	})
	g.sampler_near = sdl.CreateGPUSampler(g.device, {
		min_filter = .NEAREST,
		mag_filter = .NEAREST,
		mipmap_mode = .NEAREST,
		address_mode_u = .CLAMP_TO_EDGE,
		address_mode_v = .CLAMP_TO_EDGE,
		address_mode_w = .CLAMP_TO_EDGE,
	})
	g.sampler_linear = sdl.CreateGPUSampler(g.device, {
		min_filter = .LINEAR,
		mag_filter = .LINEAR,
		mipmap_mode = .NEAREST,
		address_mode_u = .CLAMP_TO_EDGE,
		address_mode_v = .CLAMP_TO_EDGE,
		address_mode_w = .CLAMP_TO_EDGE,
	})
	Vertex :: struct {
		pos, uv: [2]f32,
	}
	quad := [6]Vertex{{{0, 0}, {0, 0}}, {{W, 0}, {1, 0}}, {{W, H}, {1, 1}}, {{0, 0}, {0, 0}}, {{W, H}, {1, 1}}, {{0, H}, {0, 1}}}
	g.quad = sdl.CreateGPUBuffer(g.device, {usage = {.VERTEX}, size = size_of(quad)})
	g.words = sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = size_of(test_words)})
	if g.vert == nil || g.target == nil || g.target_f32 == nil || g.tex == nil || g.layers == nil ||
	   g.tex_r32f == nil || g.tex_rg32f == nil || g.tex_rgba32f == nil ||
	   g.sampler == nil || g.sampler_near == nil || g.sampler_linear == nil || g.quad == nil || g.words == nil {
		return false
	}

	// Upload the quad, the words, the texture, the array's layers and the float textures.
	pixels := len(texture_pixels)
	at_words := size_of(quad)
	at_layers := at_words + size_of(test_words)
	at_r32 := at_layers + LAYERS * pixels
	at_rg32 := at_r32 + size_of(bound_r_texels)
	at_rgba32 := at_rg32 + size_of(bound_rg_texels)
	total_upload := at_rgba32 + size_of(bound_rgba_texels)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(total_upload)})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	mem.copy(mapped, &quad, size_of(quad))
	mem.copy(mapped[at_words:], &test_words, size_of(test_words))
	for layer in 0 ..< LAYERS {
		for i in 0 ..< pixels {
			mapped[at_layers + layer * pixels + i] = layer_pixel(i, layer)
		}
	}
	mem.copy(mapped[at_r32:], &bound_r_texels, size_of(bound_r_texels))
	mem.copy(mapped[at_rg32:], &bound_rg_texels, size_of(bound_rg_texels))
	mem.copy(mapped[at_rgba32:], &bound_rgba_texels, size_of(bound_rgba_texels))
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUBuffer(pass, {transfer_buffer = transfer}, {buffer = g.quad, size = size_of(quad)}, false)
	sdl.UploadToGPUBuffer(pass, {transfer_buffer = transfer, offset = u32(at_words)}, {buffer = g.words, size = size_of(test_words)}, false)
	for l in 0 ..< LEVELS {
		n := u32(TEX >> uint(l))
		for layer in 0 ..< LAYERS {
			// Layer 0 is the 2D texture's pixels.
			src := sdl.GPUTextureTransferInfo{transfer_buffer = transfer, offset = u32(at_layers + layer * pixels + level_offset(l)), pixels_per_row = n, rows_per_layer = n}
			sdl.UploadToGPUTexture(pass, src, {texture = g.layers, mip_level = u32(l), layer = u32(layer), w = n, h = n, d = 1}, false)
			if layer == 0 {
				sdl.UploadToGPUTexture(pass, src, {texture = g.tex, mip_level = u32(l), w = n, h = n, d = 1}, false)
			}
		}
	}
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, offset = u32(at_r32), pixels_per_row = BOUND_R_W, rows_per_layer = BOUND_R_H}, {texture = g.tex_r32f, w = BOUND_R_W, h = BOUND_R_H, d = 1}, false)
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, offset = u32(at_rg32), pixels_per_row = BOUND_RG_W, rows_per_layer = BOUND_RG_H}, {texture = g.tex_rg32f, w = BOUND_RG_W, h = BOUND_RG_H, d = 1}, false)
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, offset = u32(at_rgba32), pixels_per_row = BOUND_RGBA_W, rows_per_layer = BOUND_RGBA_H}, {texture = g.tex_rgba32f, w = BOUND_RGBA_W, h = BOUND_RGBA_H, d = 1}, false)
	sdl.EndGPUCopyPass(pass)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)
	return true
}

gpu_render :: proc(g: ^Gpu, m: ^oksl.Module, block: []u8) -> ([]u8, bool) {
	info := sdl.GPUShaderCreateInfo {
		code_size = len(m.spirv) * 4,
		code = ([^]u8)(raw_data(m.spirv)),
		entrypoint = oksl.SPIRV_ENTRY,
		format = {.SPIRV},
		stage = .FRAGMENT,
		num_samplers = u32(m.num_samplers),
		num_storage_buffers = u32(len(m.storage_buffers)),
		num_uniform_buffers = u32(m.num_uniform_buffers),
	}
	if g.msl {
		info.code_size = len(m.msl)
		info.code = raw_data(m.msl)
		info.entrypoint = oksl.MSL_ENTRY
		info.format = {.MSL}
	}
	frag := sdl.CreateGPUShader(g.device, info)
	if frag == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUShader(g.device, frag)
	buffer := sdl.GPUVertexBufferDescription{slot = 0, pitch = 16, input_rate = .VERTEX}
	attrs := [2]sdl.GPUVertexAttribute{{location = 0, buffer_slot = 0, format = .FLOAT2, offset = 0}, {location = 1, buffer_slot = 0, format = .FLOAT2, offset = 8}}
	color := sdl.GPUColorTargetDescription{format = .R8G8B8A8_UNORM}
	pipeline := sdl.CreateGPUGraphicsPipeline(g.device, {
		vertex_shader = g.vert,
		fragment_shader = frag,
		vertex_input_state = {vertex_buffer_descriptions = &buffer, num_vertex_buffers = 1, vertex_attributes = &attrs[0], num_vertex_attributes = 2},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
		multisample_state = {sample_count = ._1},
		target_info = {color_target_descriptions = &color, num_color_targets = 1},
	})
	if pipeline == nil {
		return nil, false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)

	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	target := sdl.GPUColorTargetInfo{texture = g.target, load_op = .CLEAR, store_op = .STORE, clear_color = {0, 0, 0, 1}}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	binding := sdl.GPUBufferBinding{buffer = g.quad}
	sdl.BindGPUVertexBuffers(pass, 0, &binding, 1)
	if m.num_samplers > 0 {
		samplers := make([]sdl.GPUTextureSamplerBinding, m.num_samplers, context.temp_allocator)
		for &s, ch in samplers {
			s = {texture = g.layers if ch in m.channel_arrays else g.tex, sampler = g.sampler}
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(samplers), u32(len(samplers)))
	}
	if n := len(m.storage_buffers); n > 0 {
		buffers := make([]^sdl.GPUBuffer, n, context.temp_allocator)
		for &b in buffers {
			b = g.words
		}
		sdl.BindGPUFragmentStorageBuffers(pass, 0, raw_data(buffers), u32(n))
	}
	view := [4]f32{W, H, 0, 0}
	sdl.PushGPUVertexUniformData(cmd, 0, &view, size_of(view))
	if m.num_uniform_buffers > 0 {
		sdl.PushGPUFragmentUniformData(cmd, 0, raw_data(block), u32(len(block)))
	}
	sdl.DrawGPUPrimitives(pass, 6, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
	return download(g, cmd), true
}

// Submits `cmd` with a copy of the target and returns its pixels.
download :: proc(g: ^Gpu, cmd: ^sdl.GPUCommandBuffer) -> []u8 {
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .DOWNLOAD, size = W * H * 4})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	copy_pass := sdl.BeginGPUCopyPass(cmd)
	sdl.DownloadFromGPUTexture(copy_pass, {texture = g.target, w = W, h = H, d = 1}, {transfer_buffer = transfer, pixels_per_row = W, rows_per_layer = H})
	sdl.EndGPUCopyPass(copy_pass)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	px := make([]u8, W * H * 4)
	copy(px, mapped[:W * H * 4])
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	return px
}

bound_channels_check :: proc(g: ^Gpu, m: ^oksl.Module, block: []u8, out_dir: string, scratch: ^oksl.Scratch) -> (string, bool) {
	info := sdl.GPUShaderCreateInfo {
		code_size           = len(m.msl) if g.msl else len(m.spirv) * 4,
		code                = raw_data(m.msl) if g.msl else ([^]u8)(raw_data(m.spirv)),
		entrypoint          = oksl.MSL_ENTRY if g.msl else oksl.SPIRV_ENTRY,
		format              = {.MSL} if g.msl else {.SPIRV},
		stage               = .FRAGMENT,
		num_samplers        = u32(m.num_samplers),
		num_uniform_buffers = u32(m.num_uniform_buffers),
	}
	frag := sdl.CreateGPUShader(g.device, info)
	if frag == nil {
		return fmt.tprintf("shader failed: %s", sdl.GetError()), false
	}
	defer sdl.ReleaseGPUShader(g.device, frag)
	buffer := sdl.GPUVertexBufferDescription{slot = 0, pitch = 16, input_rate = .VERTEX}
	attrs := [2]sdl.GPUVertexAttribute{{location = 0, buffer_slot = 0, format = .FLOAT2, offset = 0}, {location = 1, buffer_slot = 0, format = .FLOAT2, offset = 8}}
	color := sdl.GPUColorTargetDescription{format = .R32G32B32A32_FLOAT}
	pipeline := sdl.CreateGPUGraphicsPipeline(g.device, {
		vertex_shader      = g.vert,
		fragment_shader    = frag,
		vertex_input_state = {vertex_buffer_descriptions = &buffer, num_vertex_buffers = 1, vertex_attributes = &attrs[0], num_vertex_attributes = 2},
		primitive_type     = .TRIANGLELIST,
		rasterizer_state   = {fill_mode = .FILL, cull_mode = .NONE},
		multisample_state  = {sample_count = ._1},
		target_info        = {color_target_descriptions = &color, num_color_targets = 1},
	})
	if pipeline == nil {
		return fmt.tprintf("pipeline failed: %s", sdl.GetError()), false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)

	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	target := sdl.GPUColorTargetInfo{texture = g.target_f32, load_op = .CLEAR, store_op = .STORE, clear_color = {0, 0, 0, 1}}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	binding := sdl.GPUBufferBinding{buffer = g.quad}
	sdl.BindGPUVertexBuffers(pass, 0, &binding, 1)
	samplers := [6]sdl.GPUTextureSamplerBinding {
		{texture = g.tex_r32f, sampler = g.sampler_near},
		{texture = g.tex_rg32f, sampler = g.sampler_near},
		{texture = g.tex_rgba32f, sampler = g.sampler_near},
		{texture = g.tex_r32f, sampler = g.sampler_linear},
		{texture = g.tex_rg32f, sampler = g.sampler_linear},
		{texture = g.tex_rgba32f, sampler = g.sampler_linear},
	}
	sdl.BindGPUFragmentSamplers(pass, 0, &samplers[0], len(samplers))
	view := [4]f32{W, H, 0, 0}
	sdl.PushGPUVertexUniformData(cmd, 0, &view, size_of(view))
	if m.num_uniform_buffers > 0 {
		sdl.PushGPUFragmentUniformData(cmd, 0, raw_data(block), u32(len(block)))
	}
	sdl.DrawGPUPrimitives(pass, 6, 1, 0, 0)
	sdl.EndGPURenderPass(pass)

	nbytes :: W * H * size_of([4]f32)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .DOWNLOAD, size = nbytes})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	cp := sdl.BeginGPUCopyPass(cmd)
	sdl.DownloadFromGPUTexture(cp, {texture = g.target_f32, w = W, h = H, d = 1}, {transfer_buffer = transfer, pixels_per_row = W, rows_per_layer = H})
	sdl.EndGPUCopyPass(cp)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)

	gpu_f := make([][4]f32, W * H, context.temp_allocator)
	mapped := ([^][4]f32)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	copy(gpu_f, mapped[:W * H])
	sdl.UnmapGPUTransferBuffer(g.device, transfer)

	cpu_f := make([][4]f32, W * H, context.temp_allocator)
	cimgs := [6]oksl.Channel_Image {
		oksl.channel_image(bound_r_texels[:], BOUND_R_W, BOUND_R_H, .Nearest),
		oksl.channel_image(bound_rg_texels[:], BOUND_RG_W, BOUND_RG_H, .Nearest),
		oksl.channel_image(bound_rgba_texels[:], BOUND_RGBA_W, BOUND_RGBA_H, .Nearest),
		oksl.channel_image(bound_r_texels[:], BOUND_R_W, BOUND_R_H, .Linear),
		oksl.channel_image(bound_rg_texels[:], BOUND_RG_W, BOUND_RG_H, .Linear),
		oksl.channel_image(bound_rgba_texels[:], BOUND_RGBA_W, BOUND_RGBA_H, .Linear),
	}
	oksl.run_f32(&m.program, block, W, H, cpu_f, scratch, {channel_images = cimgs[:]})

	near_err, lin_err: f32
	mismatches := 0
	gpu_png := make([]u8, W * H * 4, context.temp_allocator)
	cpu_png := make([]u8, W * H * 4, context.temp_allocator)
	diff_png := make([]u8, W * H * 4, context.temp_allocator)
	for y in 0 ..< H {
		for x in 0 ..< W {
			idx := y * W + x
			gp, cp_v := gpu_f[idx], cpu_f[idx]
			pd: f32
			for k in 0 ..< 4 {
				d := abs(gp[k] - cp_v[k])
				pd = max(pd, d)
				if y < 72 {
					near_err = max(near_err, d)
					if !(gp[k] == cp_v[k]) {
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
				cpu_png[idx * 4 + k] = u8(clamp(cp_v[k] * 0.25 + 0.5, 0, 1) * 255 + 0.5)
			}
			gpu_png[idx * 4 + 3] = 255
			cpu_png[idx * 4 + 3] = 255
			dv := u8(min(pd * 255 * 16, 255))
			diff_png[idx * 4 + 0], diff_png[idx * 4 + 1], diff_png[idx * 4 + 2], diff_png[idx * 4 + 3] = dv, dv, dv, 255
		}
	}
	write_png(fmt.tprintf("%s/bound_channels_gpu.png", out_dir), gpu_png)
	write_png(fmt.tprintf("%s/bound_channels_cpu.png", out_dir), cpu_png)
	write_png(fmt.tprintf("%s/bound_channels_diff.png", out_dir), diff_png)
	return fmt.tprintf("nearest err %.6f, linear err %.6f, mismatches %d", near_err, lin_err, mismatches), mismatches == 0
}

// A module's shader with the resources it declares (no samplers).
make_shader :: proc(g: ^Gpu, m: ^oksl.Module, stage: sdl.GPUShaderStage) -> ^sdl.GPUShader {
	info := sdl.GPUShaderCreateInfo {
		code_size = len(m.spirv) * 4,
		code = ([^]u8)(raw_data(m.spirv)),
		entrypoint = oksl.SPIRV_ENTRY,
		format = {.SPIRV},
		stage = stage,
		num_storage_buffers = u32(len(m.storage_buffers)),
		num_uniform_buffers = u32(m.num_uniform_buffers),
	}
	if g.msl {
		info.code_size = len(m.msl)
		info.code = raw_data(m.msl)
		info.entrypoint = oksl.MSL_ENTRY
		info.format = {.MSL}
	}
	return sdl.CreateGPUShader(g.device, info)
}

TAP_PRELUDE :: "in vec2 openPosition;\nout vec4 openClip;\n"
TAP_FRAGMENT :: "layout(location = 1) in vec4 openClip;\nout vec4 c;\nvoid main() { c = vec4(openClip.xy * 0.5 + 0.5, 0.0, 1.0); }"

// The layer vertex shader with one tap, whose twin of position reads uv
// and whose environment is a 1x1 target, so its position spans clip space:
// the GPU draws it as colour, and run_vertex gives it at every pixel centre
// (it is affine in the pixel, so interpolation agrees). The largest
// difference in 8 bits.
tap_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	options := oksl.Options{profile = .Glsl, stage = .Vertex, prelude = TAP_PRELUDE}
	options.taps = {{position = "openClip", attributes = {{"position", "openPosition"}}}}
	vm, verrors, vok := oksl.compile(oksl.LAYER_VERTEX_GLSL, options)
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(TAP_FRAGMENT, {profile = .Glsl})
	defer oksl.destroy(fm)
	if !vok || !fok {
		fmt.eprint(oksl.format_errors(verrors), oksl.format_errors(ferrors))
		return 0, false
	}
	vert := make_shader(g, vm, .VERTEX)
	frag := make_shader(g, fm, .FRAGMENT)
	defer sdl.ReleaseGPUShader(g.device, vert)
	defer sdl.ReleaseGPUShader(g.device, frag)
	// The twin reads the quad's uv.
	buffer := sdl.GPUVertexBufferDescription{slot = 0, pitch = 16, input_rate = .VERTEX}
	attrs: [3]sdl.GPUVertexAttribute
	for a, i in vm.stage_inputs {
		attrs[i] = {location = u32(a.location), buffer_slot = 0, format = .FLOAT2, offset = 0 if a.name == "position" else 8}
	}
	color := sdl.GPUColorTargetDescription{format = .R8G8B8A8_UNORM}
	pipeline := sdl.CreateGPUGraphicsPipeline(g.device, {
		vertex_shader = vert,
		fragment_shader = frag,
		vertex_input_state = {vertex_buffer_descriptions = &buffer, num_vertex_buffers = 1, vertex_attributes = &attrs[0], num_vertex_attributes = u32(len(vm.stage_inputs))},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
		multisample_state = {sample_count = ._1},
		target_info = {color_target_descriptions = &color, num_color_targets = 1},
	})
	if vert == nil || frag == nil || pipeline == nil {
		return 0, false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)
	block := make([]u8, vm.block_size)
	defer delete(block)
	oksl.set_uniform(vm, block, "size", {W, H, 0, 0})
	oksl.set_uniform(vm, block[vm.block_stride:], "size", {1, 1, 0, 0})

	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	target := sdl.GPUColorTargetInfo{texture = g.target, load_op = .CLEAR, store_op = .STORE, clear_color = {0, 0, 0, 1}}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	binding := sdl.GPUBufferBinding{buffer = g.quad}
	sdl.BindGPUVertexBuffers(pass, 0, &binding, 1)
	sdl.PushGPUVertexUniformData(cmd, 0, raw_data(block), u32(len(block)))
	sdl.DrawGPUPrimitives(pass, 6, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
	gpu_px := download(g, cmd)
	defer delete(gpu_px)

	// A vertex at every pixel centre.
	centres := make([][2][2]f32, W * H)
	defer delete(centres)
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
	positions := make([][4]f32, W * H)
	defer delete(positions)
	varyings := make([]f32, W * H * vm.varying_floats)
	defer delete(varyings)
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
	return max_d, true
}

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

// A vertex shader with no attributes: gl_VertexID, counted from the draw's
// first vertex (2), picks the quad's corners and their colours (affine in
// the pixel) from a storage buffer. run_vertex gives the six from the same
// first vertex; the GPU's colours against the CPU's interpolated at every
// pixel centre, the largest difference in 8 bits.
vertex_id_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	vm, verrors, vok := oksl.compile(VERTEX_ID_VS, {profile = .Glsl, stage = .Vertex})
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(VERTEX_ID_FS, {profile = .Glsl})
	defer oksl.destroy(fm)
	if !vok || !fok {
		fmt.eprint(oksl.format_errors(verrors), oksl.format_errors(ferrors))
		return 0, false
	}
	FIRST :: 2
	Corner :: struct {
		position, color: [4]f32,
	}
	corners: [FIRST + 6]Corner
	corners[0], corners[1] = {{1e6, 1e6, 0, 0}, {1, 0, 1, 1}}, {{-1e6, 0, 0, 0}, {0, 1, 1, 1}}
	quad := [6][2]f32{{0, 0}, {W, 0}, {W, H}, {0, 0}, {W, H}, {0, H}}
	for p, i in quad {
		corners[FIRST + i] = {{p.x, p.y, 0, 1}, {p.x / W, p.y / H, 0.25, 1}}
	}
	words := slice.reinterpret([]u32, corners[:])

	vert := make_shader(g, vm, .VERTEX)
	frag := make_shader(g, fm, .FRAGMENT)
	defer sdl.ReleaseGPUShader(g.device, vert)
	defer sdl.ReleaseGPUShader(g.device, frag)
	color := sdl.GPUColorTargetDescription{format = .R8G8B8A8_UNORM}
	pipeline := sdl.CreateGPUGraphicsPipeline(g.device, {
		vertex_shader = vert,
		fragment_shader = frag,
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
		multisample_state = {sample_count = ._1},
		target_info = {color_target_descriptions = &color, num_color_targets = 1},
	})
	buffer := sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = size_of(corners)})
	if vert == nil || frag == nil || pipeline == nil || buffer == nil {
		return 0, false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)
	defer sdl.ReleaseGPUBuffer(g.device, buffer)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = size_of(corners)})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mem.copy(sdl.MapGPUTransferBuffer(g.device, transfer, false), &corners, size_of(corners))
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	block := make([]u8, vm.block_size)
	defer delete(block)
	oksl.set_uniform(vm, block, "size", {W, H, 0, 0})

	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	copy_pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUBuffer(copy_pass, {transfer_buffer = transfer}, {buffer = buffer, size = size_of(corners)}, false)
	sdl.EndGPUCopyPass(copy_pass)
	target := sdl.GPUColorTargetInfo{texture = g.target, load_op = .CLEAR, store_op = .STORE, clear_color = {0, 0, 0, 1}}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	sdl.BindGPUVertexStorageBuffers(pass, 0, &buffer, 1)
	sdl.PushGPUVertexUniformData(cmd, 0, raw_data(block), u32(len(block)))
	sdl.DrawGPUPrimitives(pass, 6, 1, FIRST, 0)
	sdl.EndGPURenderPass(pass)
	gpu_px := download(g, cmd)
	defer delete(gpu_px)

	positions: [6][4]f32
	record := vm.varying_floats
	varyings := make([]f32, 6 * record)
	defer delete(varyings)
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	oksl.run_vertex(&vm.program, block, nil, 6, positions[:], varyings, &scratch, {buffers = {words}, first_vertex = FIRST})
	// The CPU's triangles in pixels, each pixel centre's colour by its
	// barycentric coordinates.
	px: [6][2]f32
	for p, i in positions {
		px[i] = {(p.x / p.w + 1) * 0.5 * W, (1 - p.y / p.w) * 0.5 * H}
	}
	for y in 0 ..< H {
		for x in 0 ..< W {
			c := [2]f32{f32(x) + 0.5, f32(y) + 0.5}
			for tri in 0 ..< 2 {
				a, b, d := px[tri * 3], px[tri * 3 + 1], px[tri * 3 + 2]
				area := (b.x - a.x) * (d.y - a.y) - (d.x - a.x) * (b.y - a.y)
				w1 := ((c.x - a.x) * (d.y - a.y) - (d.x - a.x) * (c.y - a.y)) / area
				w2 := ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / area
				w0 := 1 - w1 - w2
				if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
					continue
				}
				for k in 0 ..< 4 {
					v := w0 * varyings[(tri * 3) * record + k] + w1 * varyings[(tri * 3 + 1) * record + k] + w2 * varyings[(tri * 3 + 2) * record + k]
					want := int(clamp(v, 0, 1) * 255 + 0.5)
					max_d = max(max_d, abs(want - int(gpu_px[(y * W + x) * 4 + k])))
				}
				break
			}
		}
	}
	return max_d, true
}

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
sample_mask_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	m, errors, cok := oksl.compile(SAMPLE_MASK_FS, {profile = .Glsl})
	defer oksl.destroy(m)
	if !cok {
		fmt.eprint(oksl.format_errors(errors))
		return 0, false
	}
	if !sdl.GPUTextureSupportsSampleCount(g.device, .R8G8B8A8_UNORM, ._4) {
		return 0, false
	}
	frag := make_shader(g, m, .FRAGMENT)
	defer sdl.ReleaseGPUShader(g.device, frag)
	msaa := sdl.CreateGPUTexture(g.device, {type = .D2, format = .R8G8B8A8_UNORM, usage = {.COLOR_TARGET}, width = W, height = H, layer_count_or_depth = 1, num_levels = 1, sample_count = ._4})
	defer sdl.ReleaseGPUTexture(g.device, msaa)
	buffer := sdl.GPUVertexBufferDescription{slot = 0, pitch = 16, input_rate = .VERTEX}
	attrs := [2]sdl.GPUVertexAttribute{{location = 0, buffer_slot = 0, format = .FLOAT2, offset = 0}, {location = 1, buffer_slot = 0, format = .FLOAT2, offset = 8}}
	color := sdl.GPUColorTargetDescription{format = .R8G8B8A8_UNORM}
	pipeline := sdl.CreateGPUGraphicsPipeline(g.device, {
		vertex_shader = g.vert,
		fragment_shader = frag,
		vertex_input_state = {vertex_buffer_descriptions = &buffer, num_vertex_buffers = 1, vertex_attributes = &attrs[0], num_vertex_attributes = 2},
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
		multisample_state = {sample_count = ._4},
		target_info = {color_target_descriptions = &color, num_color_targets = 1},
	})
	if frag == nil || msaa == nil || pipeline == nil {
		return 0, false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	target := sdl.GPUColorTargetInfo{texture = msaa, load_op = .CLEAR, store_op = .RESOLVE, resolve_texture = g.target, clear_color = {0, 0, 0, 0}}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	binding := sdl.GPUBufferBinding{buffer = g.quad}
	sdl.BindGPUVertexBuffers(pass, 0, &binding, 1)
	view := [4]f32{W, H, 0, 0}
	sdl.PushGPUVertexUniformData(cmd, 0, &view, size_of(view))
	sdl.DrawGPUPrimitives(pass, 6, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
	gpu_px := download(g, cmd)
	defer delete(gpu_px)

	Q :: (W / 2) * (H / 2)
	origin := make([][2]i32, Q)
	defer delete(origin)
	for q in 0 ..< Q {
		origin[q] = {i32(q % (W / 2) * 2), i32(q / (W / 2) * 2)}
	}
	out := make([][4]f32, Q * 4)
	defer delete(out)
	shaded := make([]oksl.Quad_Lanes, Q)
	defer delete(shaded)
	masks := make([]u32, Q * 4)
	defer delete(masks)
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	block := make([]u8, m.block_size)
	defer delete(block)
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
	return max_d, true
}

sdl_image_format :: proc(f: oksl.Image_Format) -> (sdl.GPUTextureFormat, int) {
	switch f {
	case .None:
		return .INVALID, 0
	case .Rgba8:
		return .R8G8B8A8_UNORM, 4
	case .Rgba16f:
		return .R16G16B16A16_FLOAT, 8
	case .Rgba32f:
		return .R32G32B32A32_FLOAT, 16
	case .R32f:
		return .R32_FLOAT, 4
	case .R32ui:
		return .R32_UINT, 4
	case .R32i:
		return .R32_INT, 4
	}
	return .INVALID, 0
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

compute_file_check :: proc(g: ^Gpu, path: string, scratch: ^oksl.Scratch) -> (string, bool) {
	src, err := os.read_entire_file(path, context.temp_allocator)
	if err != nil {
		return "read error", false
	}
	options, _ := oksl.options_of_path(path)
	m, errs, ok := oksl.compile(string(src), options)
	defer oksl.destroy(m)
	if !ok {
		return strings.trim_space(oksl.format_errors(errs, context.temp_allocator)), false
	}
	if filepath.base(path) == "image_war.comp" {
		return compute_module_check(g, m, scratch, rw_w = 128, rw_h = 128, buf_words = 16384, exact_groups = true)
	}
	return compute_module_check(g, m, scratch)
}

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

image_formats_check :: proc(g: ^Gpu, scratch: ^oksl.Scratch) -> (string, bool) {
	m, errs, ok := oksl.compile(IMAGE_FORMATS_CS, {profile = .Glsl, stage = .Compute})
	defer oksl.destroy(m)
	if !ok {
		return strings.trim_space(oksl.format_errors(errs, context.temp_allocator)), false
	}
	return compute_module_check(g, m, scratch)
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

Range_Job :: struct {
	program: ^oksl.Program,
	block:   []u8,
	groups:  [3]int,
	opts:    oksl.Run_Options,
	first:   int,
	count:   int,
}

run_range_job :: proc(job: ^Range_Job) {
	s: oksl.Scratch
	defer oksl.scratch_destroy(&s)
	oksl.run_compute(job.program, job.block, job.groups, &s, job.opts, job.first, job.count)
}

compute_module_check :: proc(g: ^Gpu, m: ^oksl.Module, scratch: ^oksl.Scratch, rw_w := 29, rw_h := 17, buf_words := 4096, exact_groups := false) -> (string, bool) {
	_ = scratch
	info := sdl.GPUComputePipelineCreateInfo {
		code_size                      = uint(len(m.msl)) if g.msl else uint(len(m.spirv) * 4),
		code                           = raw_data(m.msl) if g.msl else ([^]u8)(raw_data(m.spirv)),
		entrypoint                     = oksl.MSL_ENTRY if g.msl else oksl.SPIRV_ENTRY,
		format                         = {.MSL} if g.msl else {.SPIRV},
		num_samplers                   = u32(m.num_samplers),
		num_readonly_storage_textures  = u32(m.num_readonly_storage_textures),
		num_readonly_storage_buffers   = u32(m.num_readonly_storage_buffers),
		num_readwrite_storage_textures = u32(m.num_readwrite_storage_textures),
		num_readwrite_storage_buffers  = u32(m.num_readwrite_storage_buffers),
		num_uniform_buffers            = u32(m.num_uniform_buffers),
		threadcount_x                  = m.threadcount.x,
		threadcount_y                  = m.threadcount.y,
		threadcount_z                  = m.threadcount.z,
	}
	pipeline := sdl.CreateGPUComputePipeline(g.device, info)
	if pipeline == nil {
		return fmt.tprintf("pipeline failed: %s", sdl.GetError()), false
	}
	defer sdl.ReleaseGPUComputePipeline(g.device, pipeline)

	// Non-multiple-of-8 default dimensions exercise partial edge workgroups.
	ro_w, ro_h := rw_w * 2, rw_h * 2
	y_stride := (ro_w + 7) &~ 7

	block := make([]u8, m.block_size, context.temp_allocator)
	oksl.set_uniform(m, block, "uScale", {1.25})
	oksl.set_uniform(m, block, "uDt", {0.25})
	oksl.set_uniform(m, block, "uCount", {500})
	oksl.set_uniform(m, block, "uMatArr", {1, 2, 3, 4}, 0)
	oksl.set_uniform(m, block, "uMatArr", {5, 6, 7, 8}, 1)
	oksl.set_uniform(m, block, "ky", {13933, 46871, 4732, 1081344})
	oksl.set_uniform(m, block, "ku", {-7509, -25259, 32768, 33685504})
	oksl.set_uniform(m, block, "kv", {32768, -29763, -3005, 255})
	oksl.set_uniform(m, block, "size", {f32(ro_w), f32(ro_h), f32(y_stride), f32(y_stride * ro_h)})

	nb := len(m.storage_buffers)
	ni := len(m.storage_images)
	cpu_bufs := make([][]u32, nb, context.temp_allocator)
	gpu_init_bufs := make([][]u32, nb, context.temp_allocator)
	buf_float_words := make([][]bool, nb, context.temp_allocator)
	gpu_bufs := make([]^sdl.GPUBuffer, nb, context.temp_allocator)
	defer for b in gpu_bufs {
		if b != nil {sdl.ReleaseGPUBuffer(g.device, b)}
	}
	for sb, bi in m.storage_buffers {
		stride_words := max(sb.stride / 4, 1)
		fw := make([]bool, stride_words, context.temp_allocator)
		if !mark_float_words(m, sb.element, 0, fw) {
			return fmt.tprintf("missing struct reflection for buffer %s", sb.name), false
		}
		buf_float_words[bi] = fw
		init_w := make([]u32, buf_words, context.temp_allocator)
		cpu_w := make([]u32, buf_words, context.temp_allocator)
		for i in 0 ..< buf_words {
			if !fw[i % stride_words] {
				init_w[i] = 0xdead_0000 | u32((bi * 31 + i * 17) & 0xffff)
			} else {
				init_w[i] = transmute(u32)(f32(((bi * 29 + i * 37 + 11) % 64) + 8) * 0.25)
			}
		}
		copy(cpu_w, init_w)
		gpu_init_bufs[bi] = init_w
		cpu_bufs[bi] = cpu_w
		usage: sdl.GPUBufferUsageFlags = {.COMPUTE_STORAGE_READ} if sb.readonly else {.COMPUTE_STORAGE_READ, .COMPUTE_STORAGE_WRITE}
		gpu_bufs[bi] = sdl.CreateGPUBuffer(g.device, {usage = usage, size = u32(buf_words * 4)})
		if gpu_bufs[bi] == nil {
			return fmt.tprintf("buffer failed: %s", sdl.GetError()), false
		}
	}

	cpu_imgs := make([]oksl.Storage_Image, ni, context.temp_allocator)
	gpu_init_imgs := make([][]u8, ni, context.temp_allocator)
	gpu_texs := make([]^sdl.GPUTexture, ni, context.temp_allocator)
	img_dims := make([][3]int, ni, context.temp_allocator)
	defer for tx in gpu_texs {
		if tx != nil {sdl.ReleaseGPUTexture(g.device, tx)}
	}
	for im, ii in m.storage_images {
		sfmt, tb := sdl_image_format(im.format)
		iw := ro_w if im.readonly else rw_w
		ih := ro_h if im.readonly else rw_h
		img_dims[ii] = {iw, ih, tb}
		nbytes := iw * ih * tb
		init_d := make([]u8, nbytes, context.temp_allocator)
		cpu_d := make([]u8, nbytes, context.temp_allocator)
		switch im.format {
		case .None:
		case .Rgba8:
			for &b, i in init_d {
				b = u8((ii * 53 + i * 41 + 19) & 255)
			}
		case .Rgba16f:
			u16s := slice.reinterpret([]u16, init_d)
			for &h, i in u16s {
				h = u16(0x3800 + ((ii * 31 + i * 19) & 0x07ff))
			}
		case .Rgba32f, .R32f:
			f32s := slice.reinterpret([]f32, init_d)
			for &f, i in f32s {
				f = f32((ii * 23 + i * 43 + 7) % 128) / 64.0
			}
		case .R32ui:
			u32s := slice.reinterpret([]u32, init_d)
			for &v, i in u32s {
				v = u32((ii * 13 + i * 17 + 5) & 1023)
			}
		case .R32i:
			i32s := slice.reinterpret([]i32, init_d)
			for &v, i in i32s {
				v = i32((ii * 13 + i * 19) % 511) - 255
			}
		}
		copy(cpu_d, init_d)
		gpu_init_imgs[ii] = init_d
		cpu_imgs[ii] = {data = cpu_d, width = iw, height = ih, format = im.format}
		usage: sdl.GPUTextureUsageFlags
		if im.readonly {
			usage = {.COMPUTE_STORAGE_READ}
		} else if im.writeonly {
			usage = {.COMPUTE_STORAGE_WRITE}
		} else {
			usage = {.COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE}
		}
		gpu_texs[ii] = sdl.CreateGPUTexture(g.device, {
			type                 = .D2,
			format               = sfmt,
			usage                = usage,
			width                = u32(iw),
			height               = u32(ih),
			layer_count_or_depth = 1,
			num_levels           = 1,
		})
		if gpu_texs[ii] == nil {
			return fmt.tprintf("texture failed: %s", sdl.GetError()), false
		}
	}

	upload_bytes := nb * buf_words * 4
	for d in gpu_init_imgs {
		upload_bytes += len(d)
	}
	if upload_bytes > 0 {
		up := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(upload_bytes)})
		defer sdl.ReleaseGPUTransferBuffer(g.device, up)
		mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, up, false))
		off := 0
		for w in gpu_init_bufs {
			mem.copy(mapped[off:], raw_data(w), len(w) * 4)
			off += len(w) * 4
		}
		for d in gpu_init_imgs {
			mem.copy(mapped[off:], raw_data(d), len(d))
			off += len(d)
		}
		sdl.UnmapGPUTransferBuffer(g.device, up)
		cmd := sdl.AcquireGPUCommandBuffer(g.device)
		cp := sdl.BeginGPUCopyPass(cmd)
		off = 0
		for w, bi in gpu_init_bufs {
			sdl.UploadToGPUBuffer(cp, {transfer_buffer = up, offset = u32(off)}, {buffer = gpu_bufs[bi], size = u32(len(w) * 4)}, false)
			off += len(w) * 4
		}
		for d, ii in gpu_init_imgs {
			iw, ih := u32(img_dims[ii][0]), u32(img_dims[ii][1])
			sdl.UploadToGPUTexture(cp, {transfer_buffer = up, offset = u32(off), pixels_per_row = iw, rows_per_layer = ih}, {texture = gpu_texs[ii], w = iw, h = ih, d = 1}, false)
			off += len(d)
		}
		sdl.EndGPUCopyPass(cp)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
		sdl.ReleaseGPUFence(g.device, fence)
	}

	tx, ty := int(m.threadcount.x), int(m.threadcount.y)
	extra := 0 if exact_groups else 3
	groups := [3]int {
		max((rw_w + extra + tx - 1) / tx, 2),
		max((rw_h + extra + ty - 1) / ty, 2),
		3 if m.threadcount.z > 1 else 1,
	}

	ro_tex := m.num_readonly_storage_textures
	rw_tex := m.num_readwrite_storage_textures
	ro_buf := m.num_readonly_storage_buffers
	rw_buf := m.num_readwrite_storage_buffers

	rw_tex_binds := make([]sdl.GPUStorageTextureReadWriteBinding, rw_tex, context.temp_allocator)
	for i in 0 ..< rw_tex {
		rw_tex_binds[i] = {texture = gpu_texs[ro_tex + i]}
	}
	rw_buf_binds := make([]sdl.GPUStorageBufferReadWriteBinding, rw_buf, context.temp_allocator)
	for i in 0 ..< rw_buf {
		rw_buf_binds[i] = {buffer = gpu_bufs[ro_buf + i]}
	}

	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	if m.num_uniform_buffers > 0 {
		sdl.PushGPUComputeUniformData(cmd, 0, raw_data(block), u32(len(block)))
	}
	pass := sdl.BeginGPUComputePass(cmd, raw_data(rw_tex_binds), u32(rw_tex), raw_data(rw_buf_binds), u32(rw_buf))
	sdl.BindGPUComputePipeline(pass, pipeline)
	if m.num_samplers > 0 {
		samplers := make([]sdl.GPUTextureSamplerBinding, m.num_samplers, context.temp_allocator)
		for &s, ch in samplers {
			s = {texture = g.layers if ch in m.channel_arrays else g.tex, sampler = g.sampler}
		}
		sdl.BindGPUComputeSamplers(pass, 0, raw_data(samplers), u32(len(samplers)))
	}
	if ro_tex > 0 {
		sdl.BindGPUComputeStorageTextures(pass, 0, raw_data(gpu_texs[:ro_tex]), u32(ro_tex))
	}
	if ro_buf > 0 {
		sdl.BindGPUComputeStorageBuffers(pass, 0, raw_data(gpu_bufs[:ro_buf]), u32(ro_buf))
	}
	sdl.DispatchGPUCompute(pass, u32(groups.x), u32(groups.y), u32(groups.z))
	sdl.EndGPUComputePass(pass)

	down_bytes := rw_buf * buf_words * 4
	for i in 0 ..< rw_tex {
		down_bytes += len(gpu_init_imgs[ro_tex + i])
	}
	down := sdl.CreateGPUTransferBuffer(g.device, {usage = .DOWNLOAD, size = u32(max(down_bytes, 4))})
	defer sdl.ReleaseGPUTransferBuffer(g.device, down)
	cp := sdl.BeginGPUCopyPass(cmd)
	off := 0
	for i in 0 ..< rw_buf {
		sdl.DownloadFromGPUBuffer(cp, {buffer = gpu_bufs[ro_buf + i], size = u32(buf_words * 4)}, {transfer_buffer = down, offset = u32(off)})
		off += buf_words * 4
	}
	for i in 0 ..< rw_tex {
		ii := ro_tex + i
		iw, ih := u32(img_dims[ii][0]), u32(img_dims[ii][1])
		sdl.DownloadFromGPUTexture(cp, {texture = gpu_texs[ii], w = iw, h = ih, d = 1}, {transfer_buffer = down, offset = u32(off), pixels_per_row = iw, rows_per_layer = ih})
		off += len(gpu_init_imgs[ii])
	}
	sdl.EndGPUCopyPass(cp)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)

	sizes: [oksl.SAMPLER_COUNT][3]i32
	for &s, ch in sizes {
		s = {TEX, TEX, LAYERS if ch in m.channel_arrays else 1}
	}
	opts := oksl.Run_Options{sampler = cpu_sampler, channel_sizes = sizes[:], buffers = cpu_bufs, images = cpu_imgs}
	// Split across three concurrent OS threads, each with its own Scratch.
	total_groups := groups.x * groups.y * groups.z
	third := total_groups / 3
	jobs := [3]Range_Job {
		{program = &m.program, block = block, groups = groups, opts = opts, first = 0, count = third},
		{program = &m.program, block = block, groups = groups, opts = opts, first = third, count = third},
		{program = &m.program, block = block, groups = groups, opts = opts, first = third * 2, count = -1},
	}
	threads: [3]^thread.Thread
	for &th, ti in threads {
		th = thread.create_and_start_with_poly_data(&jobs[ti], run_range_job)
	}
	for th in threads {
		thread.destroy(th)
	}

	got := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, down, false))
	defer sdl.UnmapGPUTransferBuffer(g.device, down)

	max_f_err: f32 = 0
	int_mismatches := 0
	off = 0
	for i in 0 ..< rw_buf {
		bi := ro_buf + i
		fw := buf_float_words[bi]
		stride_words := len(fw)
		gpu_w := slice.reinterpret([]u32, got[off : off + buf_words * 4])
		cpu_w := cpu_bufs[bi]
		off += buf_words * 4
		for w_idx in 0 ..< buf_words {
			if gpu_w[w_idx] == cpu_w[w_idx] {
				continue
			}
			if !fw[w_idx % stride_words] {
				int_mismatches += 1
				continue
			}
			gf := transmute(f32)gpu_w[w_idx]
			cf := transmute(f32)cpu_w[w_idx]
			d := abs(gf - cf)
			max_f_err = max(max_f_err, d)
			if !(d <= 1e-4) {
				int_mismatches += 1
			}
		}
	}
	for i in 0 ..< rw_tex {
		ii := ro_tex + i
		im := m.storage_images[ii]
		nbytes := len(gpu_init_imgs[ii])
		gpu_b := got[off : off + nbytes]
		cpu_b := cpu_imgs[ii].data
		off += nbytes
		switch im.format {
		case .None:
		case .R32ui, .R32i:
			gw := slice.reinterpret([]u32, gpu_b)
			cw := slice.reinterpret([]u32, cpu_b)
			for k in 0 ..< len(gw) {
				if gw[k] != cw[k] {
					int_mismatches += 1
				}
			}
		case .Rgba8:
			for k in 0 ..< nbytes {
				d := abs(int(gpu_b[k]) - int(cpu_b[k]))
				max_f_err = max(max_f_err, f32(d) / 255.0)
				if d > 1 {
					int_mismatches += 1
				}
			}
		case .Rgba16f:
			gh := slice.reinterpret([]u16, gpu_b)
			ch := slice.reinterpret([]u16, cpu_b)
			for k in 0 ..< len(gh) {
				d := abs(parity_f16_to_f32(gh[k]) - parity_f16_to_f32(ch[k]))
				max_f_err = max(max_f_err, d)
				if !(d <= 2e-3) {
					int_mismatches += 1
				}
			}
		case .Rgba32f, .R32f:
			gf := slice.reinterpret([]f32, gpu_b)
			cf := slice.reinterpret([]f32, cpu_b)
			tol: f32 = 0.02 if m.num_samplers > 0 else 1e-4
			for k in 0 ..< len(gf) {
				d := abs(gf[k] - cf[k])
				max_f_err = max(max_f_err, d)
				if !(d <= tol) {
					int_mismatches += 1
				}
			}
		}
	}
	return fmt.tprintf("max float err %.6f, mismatches %d", max_f_err, int_mismatches), int_mismatches == 0
}
