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
// with perspective and flat integers. The CPU side (test data, sampler,
// references, comparisons) is tools/parity's, shared with d3d12_parity.
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
		detail, comp_ok := compute_file_check(&g, path, &scratch)
		if !comp_ok {
			failed += 1
		}
		fmt.printfln("%-20s %s  %s", name, detail, "ok" if comp_ok else "MISMATCH")
	}
	fmt_detail, fmt_ok := image_formats_check(&g, &scratch)
	if !fmt_ok {
		failed += 1
	}
	fmt.printfln("%-20s %s  %s", "image_formats", fmt_detail, "ok" if fmt_ok else "MISMATCH")
	checks := len(names) + 4 + len(comp_names) + 1
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
	return parity.bound_channels_compare(m, block, gpu_f, "gpu", out_dir, scratch)
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
	arrays := m.channel_arrays
	opts := oksl.Run_Options{sampler = cpu_sampler, user = &arrays, channel_sizes = sizes[:], buffers = cpu_bufs, images = cpu_imgs}
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
