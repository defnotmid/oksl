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
	device:  ^sdl.GPUDevice,
	msl:     bool,
	vert:    ^sdl.GPUShader,
	target:  ^sdl.GPUTexture,
	tex:     ^sdl.GPUTexture,
	layers:  ^sdl.GPUTexture,
	sampler: ^sdl.GPUSampler,
	quad:    ^sdl.GPUBuffer,
	words:   ^sdl.GPUBuffer,
}

main :: proc() {
	args := os.args[1:]
	out_dir := args[0] if len(args) > 0 else "parity_out"
	data_dir := args[1] if len(args) > 1 else "testdata"
	os.make_directory_all(out_dir)
	make_texture()
	make_words()

	g: Gpu
	if !gpu_init(&g) {
		fmt.eprintln("no SDL_GPU device:", sdl.GetError())
		os.exit(2)
	}
	fmt.println("SDL_GPU driver:", sdl.GetGPUDeviceDriver(g.device))

	infos, _ := os.read_all_directory_by_path(data_dir, context.allocator)
	names := make([dynamic]string)
	for fi in infos {
		if o, known := oksl.options_of_path(fi.name); known && o.stage == .Fragment {
			append(&names, fi.fullpath)
		}
	}
	slice.sort(names[:])

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
	checks := len(names) + 3
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
	g.tex = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R8G8B8A8_UNORM, usage = {.SAMPLER}, width = TEX, height = TEX, layer_count_or_depth = 1, num_levels = LEVELS})
	g.layers = sdl.CreateGPUTexture(g.device, {type = .D2_ARRAY, format = .R8G8B8A8_UNORM, usage = {.SAMPLER}, width = TEX, height = TEX, layer_count_or_depth = LAYERS, num_levels = LEVELS})
	g.sampler = sdl.CreateGPUSampler(g.device, {
		min_filter = .LINEAR,
		mag_filter = .LINEAR,
		mipmap_mode = .LINEAR,
		address_mode_u = .REPEAT,
		address_mode_v = .REPEAT,
		address_mode_w = .REPEAT,
		max_lod = LEVELS - 1,
	})
	Vertex :: struct {
		pos, uv: [2]f32,
	}
	quad := [6]Vertex{{{0, 0}, {0, 0}}, {{W, 0}, {1, 0}}, {{W, H}, {1, 1}}, {{0, 0}, {0, 0}}, {{W, H}, {1, 1}}, {{0, H}, {0, 1}}}
	g.quad = sdl.CreateGPUBuffer(g.device, {usage = {.VERTEX}, size = size_of(quad)})
	g.words = sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = size_of(test_words)})
	if g.vert == nil || g.target == nil || g.tex == nil || g.layers == nil || g.sampler == nil || g.quad == nil || g.words == nil {
		return false
	}

	// Upload the quad, the words, the texture and the array's layers.
	pixels := len(texture_pixels)
	at_words := size_of(quad)
	at_layers := at_words + size_of(test_words)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(at_layers + LAYERS * pixels)})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	mem.copy(mapped, &quad, size_of(quad))
	mem.copy(mapped[at_words:], &test_words, size_of(test_words))
	for layer in 0 ..< LAYERS {
		for i in 0 ..< pixels {
			mapped[at_layers + layer * pixels + i] = layer_pixel(i, layer)
		}
	}
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
