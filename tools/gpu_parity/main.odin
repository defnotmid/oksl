// Renders every shader in testdata/ (.sksl, .glsl Shadertoy code and .frag
// GLSL fragment shaders, by oksl.SOURCE_EXTENSIONS) with SDL_GPU (the
// generated SPIR-V, or MSL on Metal, and the layer vertex shader, whose uv
// a .frag reads at location 0) and with the CPU interpreter, and compares
// the colour at location 0.
//
// `-oksl-vertex` draws with testdata/layer.vert compiled by oksl (its GLSL
// vertex stage) instead of the committed layer vertex shader.
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

// The channel texture: a colour checker (8-bit, so both sides see the same
// values).
texture_pixels: [TEX * TEX * 4]u8

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
}

texel :: proc(x, y: int) -> [4]f32 {
	xx := ((x % TEX) + TEX) % TEX
	yy := ((y % TEX) + TEX) % TEX
	i := (yy * TEX + xx) * 4
	return {f32(texture_pixels[i]), f32(texture_pixels[i + 1]), f32(texture_pixels[i + 2]), f32(texture_pixels[i + 3])} / 255
}

// Bilinear, repeat, no mipmaps: what the GPU sampler below does.
cpu_sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
	fx := uv.x * TEX - 0.5
	fy := uv.y * TEX - 0.5
	x0 := int(fx) if fx >= 0 else int(fx) - 1
	y0 := int(fy) if fy >= 0 else int(fy) - 1
	tx := fx - f32(x0)
	ty := fy - f32(y0)
	a, b := texel(x0, y0), texel(x0 + 1, y0)
	c, d := texel(x0, y0 + 1), texel(x0 + 1, y0 + 1)
	return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
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
	sampler: ^sdl.GPUSampler,
	quad:    ^sdl.GPUBuffer,
}

main :: proc() {
	args := make([dynamic]string)
	oksl_vertex := false
	for a in os.args[1:] {
		if a == "-oksl-vertex" {
			oksl_vertex = true
		} else {
			append(&args, a)
		}
	}
	out_dir := args[0] if len(args) > 0 else "parity_out"
	data_dir := args[1] if len(args) > 1 else "testdata"
	os.make_directory_all(out_dir)
	make_texture()

	g: Gpu
	if !gpu_init(&g, fmt.tprintf("%s/layer.vert", data_dir) if oksl_vertex else "") {
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
		oksl.run(&m.program, block, W, H, cpu_px, W * 4, {sampler = cpu_sampler})

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
	fmt.printfln("%d/%d shaders match (images in %s)", len(names) - failed, len(names), out_dir)
	if failed > 0 {
		os.exit(1)
	}
}

write_png :: proc(path: string, px: []u8) {
	stbi.write_png(strings.clone_to_cstring(path, context.temp_allocator), W, H, 4, raw_data(px), W * 4)
}

gpu_init :: proc(g: ^Gpu, vertex_path: string) -> bool {
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
	vert := sdl.GPUShaderCreateInfo {
		code_size = len(oksl.LAYER_VERTEX_SPIRV),
		code = raw_data(oksl.LAYER_VERTEX_SPIRV),
		entrypoint = oksl.LAYER_VERTEX_SPIRV_ENTRY,
		format = {.SPIRV},
		stage = .VERTEX,
		num_uniform_buffers = 1,
	}
	if g.msl {
		vert.code_size = len(oksl.LAYER_VERTEX_MSL)
		vert.code = raw_data(oksl.LAYER_VERTEX_MSL)
		vert.entrypoint = oksl.LAYER_VERTEX_MSL_ENTRY
		vert.format = {.MSL}
	}
	if vertex_path != "" {
		src, err := os.read_entire_file(vertex_path, context.allocator)
		if err != nil {
			fmt.eprintln("can not read", vertex_path)
			return false
		}
		options, _ := oksl.options_of_path(vertex_path)
		m, errors, ok := oksl.compile(string(src), options)
		if !ok {
			fmt.eprint(oksl.format_errors(errors))
			return false
		}
		fmt.println("vertex shader:", vertex_path, "(oksl)")
		vert.code_size = len(m.spirv) * 4
		vert.code = ([^]u8)(raw_data(m.spirv))
		vert.entrypoint = oksl.SPIRV_ENTRY
		if g.msl {
			vert.code_size = len(m.msl)
			vert.code = raw_data(m.msl)
			vert.entrypoint = oksl.MSL_ENTRY
		}
	}
	g.vert = sdl.CreateGPUShader(g.device, vert)
	g.target = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R8G8B8A8_UNORM, usage = {.COLOR_TARGET}, width = W, height = H, layer_count_or_depth = 1, num_levels = 1})
	g.tex = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R8G8B8A8_UNORM, usage = {.SAMPLER}, width = TEX, height = TEX, layer_count_or_depth = 1, num_levels = 1})
	g.sampler = sdl.CreateGPUSampler(g.device, {
		min_filter = .LINEAR,
		mag_filter = .LINEAR,
		mipmap_mode = .NEAREST,
		address_mode_u = .REPEAT,
		address_mode_v = .REPEAT,
		address_mode_w = .REPEAT,
	})
	Vertex :: struct {
		pos, uv: [2]f32,
	}
	quad := [6]Vertex{{{0, 0}, {0, 0}}, {{W, 0}, {1, 0}}, {{W, H}, {1, 1}}, {{0, 0}, {0, 0}}, {{W, H}, {1, 1}}, {{0, H}, {0, 1}}}
	g.quad = sdl.CreateGPUBuffer(g.device, {usage = {.VERTEX}, size = size_of(quad)})
	if g.vert == nil || g.target == nil || g.tex == nil || g.sampler == nil || g.quad == nil {
		return false
	}

	// Upload the quad and the texture.
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(size_of(quad) + len(texture_pixels))})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	mem.copy(mapped, &quad, size_of(quad))
	mem.copy(mapped[size_of(quad):], &texture_pixels, len(texture_pixels))
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUBuffer(pass, {transfer_buffer = transfer}, {buffer = g.quad, size = size_of(quad)}, false)
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, offset = size_of(quad), pixels_per_row = TEX, rows_per_layer = TEX}, {texture = g.tex, w = TEX, h = TEX, d = 1}, false)
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
		num_uniform_buffers = 1,
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
		for &s in samplers {
			s = {texture = g.tex, sampler = g.sampler}
		}
		sdl.BindGPUFragmentSamplers(pass, 0, raw_data(samplers), u32(len(samplers)))
	}
	view := [4]f32{W, H, 0, 0}
	sdl.PushGPUVertexUniformData(cmd, 0, &view, size_of(view))
	sdl.PushGPUFragmentUniformData(cmd, 0, raw_data(block), u32(len(block)))
	sdl.DrawGPUPrimitives(pass, 6, 1, 0, 0)
	sdl.EndGPURenderPass(pass)

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
	return px, true
}
