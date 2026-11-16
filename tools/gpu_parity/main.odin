// Renders every shader in testdata/ (.sksl, .glsl Shadertoy code and .frag
// GLSL fragment shaders, by oksl.SOURCE_EXTENSIONS) with SDL_GPU (the
// generated SPIR-V, or MSL on Metal, drawn through oksl.LAYER_VERTEX_GLSL,
// whose uv a .frag reads at location 0) and with the CPU interpreter, and
// compares the colour at location 0: sampler2DArrays read LAYERS copies
// of a checker texture with their colours rotated, sampler2D channel n
// layer n % LAYERS, and every storage buffer the test words. It also
// checks a tap of the layer vertex shader (Options.taps), its position
// drawn as colour, a vertex shader placing the quad's corners by
// gl_VertexID from a storage buffer, against run_vertex's, gl_SampleMask
// resolved from 4x MSAA against run_fragment's masks, and both windings
// with perspective and flat integers, and every testdata compute shader
// and image_formats dispatched against run_compute. The CPU side (test
// data, sampler, references, compute data and comparisons) is
// tools/parity's, shared with d3d12_parity.
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
import parity "../parity"
import sdl "vendor:sdl3"

W :: parity.W
H :: parity.H
TEX :: parity.TEX
LAYERS :: parity.LAYERS
LEVELS :: parity.LEVELS
cpu_sampler :: parity.cpu_sampler
write_png :: parity.write_png

Gpu :: struct {
	device:         ^sdl.GPUDevice,
	msl:            bool,
	vert:           ^sdl.GPUShader,
	target:         ^sdl.GPUTexture,
	target_f32:     ^sdl.GPUTexture,
	// The checker's layers as 2D textures (sampler2D channel n reads tex[n
	// % LAYERS]), and the array of them.
	tex:            [LAYERS]^sdl.GPUTexture,
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
	parity.make_data()

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
			fmt.printfln("%-20s compile error:\n%s", name, oksl.format_errors(errors, context.temp_allocator))
			failed += 1
			continue
		}
		block := parity.uniform_block(m)
		if name == "bound_channels" {
			detail, bc_ok := bound_channels_check(&g, m, block, out_dir, &scratch)
			if !bc_ok {
				failed += 1
			}
			fmt.printfln("%-20s %s  %s", name, detail, "ok" if bc_ok else "MISMATCH")
			oksl.destroy(m)
			continue
		}
		gpu_px, gpu_ok := gpu_render(&g, m, block)
		if !gpu_ok {
			fmt.printfln("%-20s GPU render failed: %s", name, sdl.GetError())
			failed += 1
			oksl.destroy(m)
			continue
		}
		if !parity.compare_rgba8(name, "gpu", out_dir, parity.fragment_reference(m, block, &scratch), gpu_px) {
			failed += 1
		}
		delete(gpu_px)
		oksl.destroy(m)
		free_all(context.temp_allocator)
	}
	report :: proc(name: string, d: int, ok: bool, failed: ^int) {
		good := ok && d <= 2
		if !good {
			failed^ += 1
		}
		fmt.printfln("%-20s max diff %d  %s", name, d, "ok" if good else "MISMATCH")
	}
	max_d, tap_ok := tap_check(&g)
	report("layer.vert tap", max_d, tap_ok, &failed)
	id_d, id_ok := vertex_id_check(&g)
	report("gl_VertexID", id_d, id_ok, &failed)
	mask_d, mask_ok := sample_mask_check(&g)
	report("gl_SampleMask", mask_d, mask_ok, &failed)
	face_detail, face_ok := facing_check(&g)
	if !face_ok {
		failed += 1
	}
	fmt.printfln("%-20s %s  %s", "facing", face_detail, "ok" if face_ok else "MISMATCH")
	for path in comp_names {
		name := filepath.base(path)
		detail, comp_ok := compute_file_check(&g, path)
		if !comp_ok {
			failed += 1
		}
		fmt.printfln("%-20s %s  %s", name, detail, "ok" if comp_ok else "MISMATCH")
	}
	bf_checks := bound_formats_check(&g, &scratch, &failed)
	fmt_detail, fmt_ok := image_formats_check(&g)
	if !fmt_ok {
		failed += 1
	}
	fmt.printfln("%-20s %s  %s", "image_formats", fmt_detail, "ok" if fmt_ok else "MISMATCH")
	checks := len(names) + 4 + len(comp_names) + 1 + bf_checks
	fmt.printfln("%d/%d shaders match (images in %s)", checks - failed, checks, out_dir)
	if failed > 0 {
		os.exit(1)
	}
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
	for &t in g.tex {
		t = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R8G8B8A8_UNORM, usage = {.SAMPLER}, width = TEX, height = TEX, layer_count_or_depth = 1, num_levels = LEVELS})
	}
	g.layers = sdl.CreateGPUTexture(g.device, {type = .D2_ARRAY, format = .R8G8B8A8_UNORM, usage = {.SAMPLER}, width = TEX, height = TEX, layer_count_or_depth = LAYERS, num_levels = LEVELS})
	g.tex_r32f = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R32_FLOAT, usage = {.SAMPLER}, width = parity.BOUND_R_W, height = parity.BOUND_R_H, layer_count_or_depth = 1, num_levels = 1})
	g.tex_rg32f = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R32G32_FLOAT, usage = {.SAMPLER}, width = parity.BOUND_RG_W, height = parity.BOUND_RG_H, layer_count_or_depth = 1, num_levels = 1})
	g.tex_rgba32f = sdl.CreateGPUTexture(g.device, {type = .D2, format = .R32G32B32A32_FLOAT, usage = {.SAMPLER}, width = parity.BOUND_RGBA_W, height = parity.BOUND_RGBA_H, layer_count_or_depth = 1, num_levels = 1})
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
	quad := parity.QUAD
	g.quad = sdl.CreateGPUBuffer(g.device, {usage = {.VERTEX}, size = size_of(quad)})
	g.words = sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = size_of(parity.test_words)})
	if g.vert == nil || g.target == nil || g.target_f32 == nil || g.tex[0] == nil || g.tex[1] == nil || g.tex[2] == nil || g.layers == nil ||
	   g.tex_r32f == nil || g.tex_rg32f == nil || g.tex_rgba32f == nil ||
	   g.sampler == nil || g.sampler_near == nil || g.sampler_linear == nil || g.quad == nil || g.words == nil {
		return false
	}

	// Upload the quad, the words, the texture, the array's layers and the float textures.
	pixels := len(parity.texture_pixels)
	at_words := size_of(quad)
	at_layers := at_words + size_of(parity.test_words)
	at_r32 := at_layers + LAYERS * pixels
	at_rg32 := at_r32 + size_of(parity.bound_r_texels)
	at_rgba32 := at_rg32 + size_of(parity.bound_rg_texels)
	total_upload := at_rgba32 + size_of(parity.bound_rgba_texels)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(total_upload)})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	mem.copy(mapped, &quad, size_of(quad))
	mem.copy(mapped[at_words:], &parity.test_words, size_of(parity.test_words))
	for layer in 0 ..< LAYERS {
		for i in 0 ..< pixels {
			mapped[at_layers + layer * pixels + i] = parity.layer_pixel(i, layer)
		}
	}
	mem.copy(mapped[at_r32:], &parity.bound_r_texels, size_of(parity.bound_r_texels))
	mem.copy(mapped[at_rg32:], &parity.bound_rg_texels, size_of(parity.bound_rg_texels))
	mem.copy(mapped[at_rgba32:], &parity.bound_rgba_texels, size_of(parity.bound_rgba_texels))
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUBuffer(pass, {transfer_buffer = transfer}, {buffer = g.quad, size = size_of(quad)}, false)
	sdl.UploadToGPUBuffer(pass, {transfer_buffer = transfer, offset = u32(at_words)}, {buffer = g.words, size = size_of(parity.test_words)}, false)
	for l in 0 ..< LEVELS {
		n := u32(TEX >> uint(l))
		for layer in 0 ..< LAYERS {
			src := sdl.GPUTextureTransferInfo{transfer_buffer = transfer, offset = u32(at_layers + layer * pixels + parity.level_offset(l)), pixels_per_row = n, rows_per_layer = n}
			sdl.UploadToGPUTexture(pass, src, {texture = g.layers, mip_level = u32(l), layer = u32(layer), w = n, h = n, d = 1}, false)
			sdl.UploadToGPUTexture(pass, src, {texture = g.tex[layer], mip_level = u32(l), w = n, h = n, d = 1}, false)
		}
	}
	BR_W, BR_H :: parity.BOUND_R_W, parity.BOUND_R_H
	BRG_W, BRG_H :: parity.BOUND_RG_W, parity.BOUND_RG_H
	BRGBA_W, BRGBA_H :: parity.BOUND_RGBA_W, parity.BOUND_RGBA_H
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, offset = u32(at_r32), pixels_per_row = BR_W, rows_per_layer = BR_H}, {texture = g.tex_r32f, w = BR_W, h = BR_H, d = 1}, false)
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, offset = u32(at_rg32), pixels_per_row = BRG_W, rows_per_layer = BRG_H}, {texture = g.tex_rg32f, w = BRG_W, h = BRG_H, d = 1}, false)
	sdl.UploadToGPUTexture(pass, {transfer_buffer = transfer, offset = u32(at_rgba32), pixels_per_row = BRGBA_W, rows_per_layer = BRGBA_H}, {texture = g.tex_rgba32f, w = BRGBA_W, h = BRGBA_H, d = 1}, false)
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
			s = {texture = g.layers if ch in m.channel_arrays else g.tex[ch % LAYERS], sampler = g.sampler}
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
	return parity.bound_channels_compare(m, block, gpu_f, "gpu", out_dir, scratch, false)
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

// The layer vertex shader with a tap, its position drawn as colour.
tap_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	vm, verrors, vok := oksl.compile(oksl.LAYER_VERTEX_GLSL, parity.tap_options(false))
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(parity.TAP_FRAGMENT, {profile = .Glsl})
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
	block := parity.tap_block(vm)

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
	return parity.tap_compare(vm, block, gpu_px), true
}

// A vertex shader with no attributes: gl_VertexID, counted from the draw's
// first vertex (2), picks the quad's corners and their colours from a
// storage buffer.
vertex_id_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	vm, verrors, vok := oksl.compile(parity.VERTEX_ID_VS, {profile = .Glsl, stage = .Vertex})
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(parity.VERTEX_ID_FS, {profile = .Glsl})
	defer oksl.destroy(fm)
	if !vok || !fok {
		fmt.eprint(oksl.format_errors(verrors), oksl.format_errors(ferrors))
		return 0, false
	}
	FIRST :: 2
	corners := parity.vertex_id_corners(FIRST)
	size := u32(len(corners) * size_of(parity.Corner))

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
	buffer := sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = size})
	if vert == nil || frag == nil || pipeline == nil || buffer == nil {
		return 0, false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)
	defer sdl.ReleaseGPUBuffer(g.device, buffer)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = size})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mem.copy(sdl.MapGPUTransferBuffer(g.device, transfer, false), raw_data(corners), int(size))
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	block := parity.vertex_id_block(vm)

	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	copy_pass := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUBuffer(copy_pass, {transfer_buffer = transfer}, {buffer = buffer, size = size}, false)
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
	return parity.vertex_id_compare(vm, block, corners, FIRST, gpu_px), true
}

// gl_SampleMask, 4x MSAA resolved over black.
sample_mask_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	m, errors, cok := oksl.compile(parity.SAMPLE_MASK_FS, {profile = .Glsl})
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
	return parity.sample_mask_compare(m, gpu_px), true
}

// Both windings, perspective (w != 1, z != 0) and flat integers
// (parity.FACING_VS / FACING_FS) on a float target, SDL_GPU's default
// (counter-clockwise) front face.
facing_check :: proc(g: ^Gpu) -> (string, bool) {
	vm, verrors, vok := oksl.compile(parity.FACING_VS, {profile = .Glsl, stage = .Vertex})
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(parity.FACING_FS, {profile = .Glsl})
	defer oksl.destroy(fm)
	if !vok || !fok {
		return fmt.tprint(oksl.format_errors(verrors), oksl.format_errors(ferrors)), false
	}
	vert := make_shader(g, vm, .VERTEX)
	frag := make_shader(g, fm, .FRAGMENT)
	defer sdl.ReleaseGPUShader(g.device, vert)
	defer sdl.ReleaseGPUShader(g.device, frag)
	color := sdl.GPUColorTargetDescription{format = .R32G32B32A32_FLOAT}
	pipeline := sdl.CreateGPUGraphicsPipeline(g.device, {
		vertex_shader = vert,
		fragment_shader = frag,
		primitive_type = .TRIANGLELIST,
		rasterizer_state = {fill_mode = .FILL, cull_mode = .NONE},
		multisample_state = {sample_count = ._1},
		target_info = {color_target_descriptions = &color, num_color_targets = 1},
	})
	if vert == nil || frag == nil || pipeline == nil {
		return fmt.tprintf("pipeline failed: %s", sdl.GetError()), false
	}
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	target := sdl.GPUColorTargetInfo{texture = g.target_f32, load_op = .CLEAR, store_op = .STORE, clear_color = {0, 0, 0, 1}}
	pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
	sdl.BindGPUGraphicsPipeline(pass, pipeline)
	sdl.DrawGPUPrimitives(pass, 6, 1, 0, 0)
	sdl.EndGPURenderPass(pass)
	return parity.facing_compare(vm, download_f32(g, cmd))
}

// Submits `cmd` with a copy of the float target and returns its pixels,
// in the temp allocator.
download_f32 :: proc(g: ^Gpu, cmd: ^sdl.GPUCommandBuffer) -> [][4]f32 {
	nbytes :: W * H * size_of([4]f32)
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .DOWNLOAD, size = nbytes})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	cp := sdl.BeginGPUCopyPass(cmd)
	sdl.DownloadFromGPUTexture(cp, {texture = g.target_f32, w = W, h = H, d = 1}, {transfer_buffer = transfer, pixels_per_row = W, rows_per_layer = H})
	sdl.EndGPUCopyPass(cp)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)
	px := make([][4]f32, W * H, context.temp_allocator)
	mapped := ([^][4]f32)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	copy(px, mapped[:W * H])
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	return px
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

compute_file_check :: proc(g: ^Gpu, path: string) -> (string, bool) {
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
	return compute_module_check(g, m, parity.compute_shape(filepath.base(path)))
}

image_formats_check :: proc(g: ^Gpu) -> (string, bool) {
	m, errs, ok := oksl.compile(parity.IMAGE_FORMATS_CS, {profile = .Glsl, stage = .Compute})
	defer oksl.destroy(m)
	if !ok {
		return strings.trim_space(oksl.format_errors(errs, context.temp_allocator)), false
	}
	return compute_module_check(g, m, parity.compute_shape(""))
}

// Dispatches the module on SDL_GPU with tools/parity's data and compares
// with the CPU's run.
compute_module_check :: proc(g: ^Gpu, m: ^oksl.Module, shape: parity.Compute_Shape) -> (string, bool) {
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

	cd, cd_err := parity.compute_data(m, shape)
	if cd_err != "" {
		return cd_err, false
	}
	buf_words := cd.buf_words
	nb := len(m.storage_buffers)
	ni := len(m.storage_images)
	gpu_bufs := make([]^sdl.GPUBuffer, nb, context.temp_allocator)
	defer for b in gpu_bufs {
		if b != nil {sdl.ReleaseGPUBuffer(g.device, b)}
	}
	for sb, bi in m.storage_buffers {
		usage: sdl.GPUBufferUsageFlags = {.COMPUTE_STORAGE_READ} if sb.readonly else {.COMPUTE_STORAGE_READ, .COMPUTE_STORAGE_WRITE}
		gpu_bufs[bi] = sdl.CreateGPUBuffer(g.device, {usage = usage, size = u32(buf_words * 4)})
		if gpu_bufs[bi] == nil {
			return fmt.tprintf("buffer failed: %s", sdl.GetError()), false
		}
	}

	gpu_texs := make([]^sdl.GPUTexture, ni, context.temp_allocator)
	defer for tx in gpu_texs {
		if tx != nil {sdl.ReleaseGPUTexture(g.device, tx)}
	}
	for im, ii in m.storage_images {
		sfmt, _ := sdl_image_format(im.format)
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
			width                = u32(cd.cpu_imgs[ii].width),
			height               = u32(cd.cpu_imgs[ii].height),
			layer_count_or_depth = 1,
			num_levels           = 1,
		})
		if gpu_texs[ii] == nil {
			return fmt.tprintf("texture failed: %s", sdl.GetError()), false
		}
	}

	upload_bytes := nb * buf_words * 4
	for d in cd.init_imgs {
		upload_bytes += len(d)
	}
	if upload_bytes > 0 {
		up := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(upload_bytes)})
		defer sdl.ReleaseGPUTransferBuffer(g.device, up)
		mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, up, false))
		off := 0
		for w in cd.init_bufs {
			mem.copy(mapped[off:], raw_data(w), len(w) * 4)
			off += len(w) * 4
		}
		for d in cd.init_imgs {
			mem.copy(mapped[off:], raw_data(d), len(d))
			off += len(d)
		}
		sdl.UnmapGPUTransferBuffer(g.device, up)
		cmd := sdl.AcquireGPUCommandBuffer(g.device)
		cp := sdl.BeginGPUCopyPass(cmd)
		off = 0
		for w, bi in cd.init_bufs {
			sdl.UploadToGPUBuffer(cp, {transfer_buffer = up, offset = u32(off)}, {buffer = gpu_bufs[bi], size = u32(len(w) * 4)}, false)
			off += len(w) * 4
		}
		for d, ii in cd.init_imgs {
			iw, ih := u32(cd.cpu_imgs[ii].width), u32(cd.cpu_imgs[ii].height)
			sdl.UploadToGPUTexture(cp, {transfer_buffer = up, offset = u32(off), pixels_per_row = iw, rows_per_layer = ih}, {texture = gpu_texs[ii], w = iw, h = ih, d = 1}, false)
			off += len(d)
		}
		sdl.EndGPUCopyPass(cp)
		fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
		_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
		sdl.ReleaseGPUFence(g.device, fence)
	}

	groups := cd.groups
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
		sdl.PushGPUComputeUniformData(cmd, 0, raw_data(cd.block), u32(len(cd.block)))
	}
	pass := sdl.BeginGPUComputePass(cmd, raw_data(rw_tex_binds), u32(rw_tex), raw_data(rw_buf_binds), u32(rw_buf))
	sdl.BindGPUComputePipeline(pass, pipeline)
	if m.num_samplers > 0 {
		samplers := make([]sdl.GPUTextureSamplerBinding, m.num_samplers, context.temp_allocator)
		for &s, ch in samplers {
			s = {texture = g.layers if ch in m.channel_arrays else g.tex[ch % LAYERS], sampler = g.sampler}
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
		down_bytes += len(cd.init_imgs[ro_tex + i])
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
		iw, ih := u32(cd.cpu_imgs[ii].width), u32(cd.cpu_imgs[ii].height)
		sdl.DownloadFromGPUTexture(cp, {texture = gpu_texs[ii], w = iw, h = ih, d = 1}, {transfer_buffer = down, offset = u32(off), pixels_per_row = iw, rows_per_layer = ih})
		off += len(cd.init_imgs[ii])
	}
	sdl.EndGPUCopyPass(cp)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)

	parity.compute_cpu(m, &cd)

	got := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, down, false))
	defer sdl.UnmapGPUTransferBuffer(g.device, down)
	gpu_b := make([][]u8, rw_buf, context.temp_allocator)
	gpu_i := make([][]u8, rw_tex, context.temp_allocator)
	off = 0
	for &b in gpu_b {
		b = got[off:][:buf_words * 4]
		off += buf_words * 4
	}
	for &im, i in gpu_i {
		n := len(cd.init_imgs[ro_tex + i])
		im = got[off:][:n]
		off += n
	}
	return parity.compute_compare(m, &cd, gpu_b, gpu_i)
}

// The bound byte and half formats (parity.bound_formats_compare): each
// format with its levels uploaded, sampled with textureLod at
// parity.bound_fmt_queries by a nearest and a linear sampler in each of
// parity.BOUND_FMT_WRAPS. Returns the checks it ran.
bound_formats_check :: proc(g: ^Gpu, scratch: ^oksl.Scratch, failed: ^int) -> int {
	m, errors, ok := oksl.compile(parity.BOUND_FMT_FRAG, {profile = .Glsl, stage = .Fragment})
	defer oksl.destroy(m)
	if !ok {
		fmt.println(oksl.format_errors(errors, context.temp_allocator))
		failed^ += 1
		return 1
	}
	info := sdl.GPUShaderCreateInfo {
		code_size           = len(m.msl) if g.msl else len(m.spirv) * 4,
		code                = raw_data(m.msl) if g.msl else ([^]u8)(raw_data(m.spirv)),
		entrypoint          = oksl.MSL_ENTRY if g.msl else oksl.SPIRV_ENTRY,
		format              = {.MSL} if g.msl else {.SPIRV},
		stage               = .FRAGMENT,
		num_samplers        = 1,
		num_storage_buffers = 1,
	}
	frag := sdl.CreateGPUShader(g.device, info)
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
	defer sdl.ReleaseGPUGraphicsPipeline(g.device, pipeline)
	if frag == nil || pipeline == nil {
		fmt.println("bound_formats: shader or pipeline failed:", sdl.GetError())
		failed^ += 1
		return 1
	}
	// The queries and every format's levels.
	qbytes := size_of(parity.bound_fmt_queries)
	qbuf := sdl.CreateGPUBuffer(g.device, {usage = {.GRAPHICS_STORAGE_READ}, size = u32(qbytes)})
	defer sdl.ReleaseGPUBuffer(g.device, qbuf)
	sdl_formats := [4]sdl.GPUTextureFormat{.R8G8B8A8_UNORM, .R8_UNORM, .R16_UNORM, .R16G16B16A16_FLOAT}
	texs: [4]^sdl.GPUTexture
	total := qbytes
	for fi in 0 ..< 4 {
		for l in 0 ..< parity.BOUND_FMT_LEVELS {
			total += (len(parity.bound_fmt_texels[fi][l]) + 15) &~ 15
		}
	}
	transfer := sdl.CreateGPUTransferBuffer(g.device, {usage = .UPLOAD, size = u32(total)})
	defer sdl.ReleaseGPUTransferBuffer(g.device, transfer)
	mapped := ([^]u8)(sdl.MapGPUTransferBuffer(g.device, transfer, false))
	mem.copy(mapped, &parity.bound_fmt_queries, qbytes)
	at := qbytes
	offsets: [4][parity.BOUND_FMT_LEVELS]int
	for fi in 0 ..< 4 {
		for l in 0 ..< parity.BOUND_FMT_LEVELS {
			b := parity.bound_fmt_texels[fi][l]
			offsets[fi][l] = at
			mem.copy(mapped[at:], raw_data(b), len(b))
			at += (len(b) + 15) &~ 15 // Vulkan wants texel-aligned buffer offsets
		}
	}
	sdl.UnmapGPUTransferBuffer(g.device, transfer)
	cmd := sdl.AcquireGPUCommandBuffer(g.device)
	cp := sdl.BeginGPUCopyPass(cmd)
	sdl.UploadToGPUBuffer(cp, {transfer_buffer = transfer}, {buffer = qbuf, size = u32(qbytes)}, false)
	for fi in 0 ..< 4 {
		texs[fi] = sdl.CreateGPUTexture(g.device, {type = .D2, format = sdl_formats[fi], usage = {.SAMPLER}, width = parity.BOUND_FMT_W, height = parity.BOUND_FMT_H, layer_count_or_depth = 1, num_levels = parity.BOUND_FMT_LEVELS})
		for l in 0 ..< parity.BOUND_FMT_LEVELS {
			w, h := parity.bound_fmt_size(l)
			sdl.UploadToGPUTexture(cp, {transfer_buffer = transfer, offset = u32(offsets[fi][l]), pixels_per_row = u32(w), rows_per_layer = u32(h)}, {texture = texs[fi], mip_level = u32(l), w = u32(w), h = u32(h), d = 1}, false)
		}
	}
	sdl.EndGPUCopyPass(cp)
	fence := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
	_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
	sdl.ReleaseGPUFence(g.device, fence)
	defer for t in texs {
		sdl.ReleaseGPUTexture(g.device, t)
	}

	checks := 0
	formats := parity.BOUND_FORMATS
	wraps := parity.BOUND_FMT_WRAPS
	for fi in 0 ..< 4 {
		for filter in oksl.Channel_Filter {
			for wrap in wraps {
				mode := [2]sdl.GPUSamplerAddressMode{.REPEAT if wrap[0] == .Repeat else .CLAMP_TO_EDGE, .REPEAT if wrap[1] == .Repeat else .CLAMP_TO_EDGE}
				f: sdl.GPUFilter = .NEAREST if filter == .Nearest else .LINEAR
				smp := sdl.CreateGPUSampler(g.device, {min_filter = f, mag_filter = f, mipmap_mode = .LINEAR, address_mode_u = mode[0], address_mode_v = mode[1], address_mode_w = .CLAMP_TO_EDGE, max_lod = 1000})
				cmd = sdl.AcquireGPUCommandBuffer(g.device)
				target := sdl.GPUColorTargetInfo{texture = g.target_f32, load_op = .CLEAR, store_op = .STORE}
				pass := sdl.BeginGPURenderPass(cmd, &target, 1, nil)
				sdl.BindGPUGraphicsPipeline(pass, pipeline)
				binding := sdl.GPUBufferBinding{buffer = g.quad}
				sdl.BindGPUVertexBuffers(pass, 0, &binding, 1)
				sb := sdl.GPUTextureSamplerBinding{texture = texs[fi], sampler = smp}
				sdl.BindGPUFragmentSamplers(pass, 0, &sb, 1)
				qb := qbuf
				sdl.BindGPUFragmentStorageBuffers(pass, 0, &qb, 1)
				view := [4]f32{W, H, 0, 0}
				sdl.PushGPUVertexUniformData(cmd, 0, &view, size_of(view))
				sdl.DrawGPUPrimitives(pass, 6, 1, 0, 0)
				sdl.EndGPURenderPass(pass)
				nbytes :: W * H * size_of([4]f32)
				dl := sdl.CreateGPUTransferBuffer(g.device, {usage = .DOWNLOAD, size = nbytes})
				dp := sdl.BeginGPUCopyPass(cmd)
				sdl.DownloadFromGPUTexture(dp, {texture = g.target_f32, w = W, h = H, d = 1}, {transfer_buffer = dl, pixels_per_row = W, rows_per_layer = H})
				sdl.EndGPUCopyPass(dp)
				fence = sdl.SubmitGPUCommandBufferAndAcquireFence(cmd)
				_ = sdl.WaitForGPUFences(g.device, true, &fence, 1)
				sdl.ReleaseGPUFence(g.device, fence)
				gpu_f := make([][4]f32, W * H, context.temp_allocator)
				mp := ([^][4]f32)(sdl.MapGPUTransferBuffer(g.device, dl, false))
				copy(gpu_f, mp[:W * H])
				sdl.UnmapGPUTransferBuffer(g.device, dl)
				sdl.ReleaseGPUTransferBuffer(g.device, dl)
				sdl.ReleaseGPUSampler(g.device, smp)
				detail, bf_ok := parity.bound_formats_compare(m, fi, filter, wrap, gpu_f, scratch, g.msl)
				if !bf_ok {
					failed^ += 1
				}
				checks += 1
				fmt.printfln("%-20s %v %v %v: %s  %s", "bound_formats", formats[fi], filter, wrap, detail, "ok" if bf_ok else "MISMATCH")
			}
		}
	}
	return checks
}
