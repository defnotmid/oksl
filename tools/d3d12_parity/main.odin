#+build linux amd64, windows amd64
// Renders every vertex and fragment shader check of tools/gpu_parity
// through Direct3D 12 with oksl's DXBC (Module.dxbc), on vkd3d over Vulkan
// (libvkd3d, libvkd3d-utils; d3d12.odin) on Linux, or on Microsoft's
// runtime on Windows (d3d12_windows.odin: WARP, or with -hardware the first
// hardware adapter, under the debug layer, whose errors fail the check they
// come from), and compares the colours with
// the CPU interpreter's under gpu_parity's rules: every testdata fragment
// shader drawn through oksl.LAYER_VERTEX_GLSL (sampler2Ds read a checker
// texture with its mip chain, sampler2DArrays LAYERS copies of it with
// their colours rotated, storage buffers the test words), the bound float
// channels, a tap of the layer vertex shader, gl_VertexID from a storage
// buffer, gl_SampleMask resolved from 4x MSAA, both windings with
// perspective and flat integers, and the DXBC emitter's own cases
// (edge.odin: early returns, loops, indexing and bounds, integers,
// evaluation order, bools, discard, channels through functions, built-ins
// at their edges); then every testdata compute shader, image_formats and
// the emitter's compute cases (edge_compute.odin), dispatched with SDL_GPU's
// compute root signature (compute.odin), every writable buffer word and
// image texel compared with the CPU's run_compute. The CPU side is
// tools/parity's, shared with gpu_parity.
//
// Bindings follow SDL_GPU's D3D12 rules: vertex t[n] / s[n] in space0 and
// b0 in space1, pixel t[n] / s[n] in space2 and b0 in space3 (root CBVs and
// descriptor tables, as SDL's root signature has them), vertex inputs
// TEXCOORD<location>; compute t / s in space0, u in space1, b0 in space2.
//
// Writes <name>_d3d12.png, <name>_cpu.png and <name>_diff.png.
//
//     odin run tools/d3d12_parity -- [-hardware] [out_dir] [testdata_dir]
package d3d12_parity

import "core:fmt"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:slice"
import oksl "../.."
import parity "../parity"
import d3d12 "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"

W :: parity.W
H :: parity.H
TEX :: parity.TEX
LAYERS :: parity.LAYERS
LEVELS :: parity.LEVELS

// ---- the device -------------------------------------------------------------------------

// Tex0-2: the checker's layers 0-2 as 2D textures (sampler2D channel n
// reads Tex0 + n % LAYERS).
Srv :: enum {
	Tex0,
	Tex1,
	Tex2,
	Layers,
	Words,
	Corners,
	R32f,
	Rg32f,
	Rgba32f,
}

Smp :: enum {
	Repeat,
	Near_Clamp,
	Linear_Clamp,
}

Target :: enum {
	Rgba8,
	Rgba32f,
	Msaa4,
}

TARGET_FORMATS := [Target]dxgi.FORMAT {
	.Rgba8   = .R8G8B8A8_UNORM,
	.Rgba32f = .R32G32B32A32_FLOAT,
	.Msaa4   = .R8G8B8A8_UNORM,
}

// Each target's clear colour, also its optimized clear value.
TARGET_CLEARS := [Target][4]f32 {
	.Rgba8   = {0, 0, 0, 1},
	.Rgba32f = {0, 0, 0, 1},
	.Msaa4   = {0, 0, 0, 0},
}

Gpu :: struct {
	device, queue, allocator, list, fence: rawptr,
	fence_value:                           u64,
	srv_heap, sampler_heap, rtv_heap:      rawptr,
	srv_step, sampler_step, rtv_step:      u32,
	targets:                               [Target]rawptr,
	resolved:                              rawptr,
	readback, constants, quad:             rawptr,
	resources:                             [Srv]rawptr,
	vs:                                    []u8,
	// WARP (Windows without -hardware): its Linear weights differ from
	// the 1/256 rounding at some coordinates, as D3D's 8-bit minimum
	// allows (parity.bound_channels_compare's coarse_weights).
	warp:                                  bool,
}

SHADER_READ :: d3d12.RESOURCE_STATES{.NON_PIXEL_SHADER_RESOURCE, .PIXEL_SHADER_RESOURCE}

buffer_desc :: proc(size: int) -> d3d12.RESOURCE_DESC {
	return {Dimension = .BUFFER, Width = u64(size), Height = 1, DepthOrArraySize = 1, MipLevels = 1, SampleDesc = {Count = 1}, Layout = .ROW_MAJOR}
}

texture_desc :: proc(format: dxgi.FORMAT, w, h, layers, levels: int, samples: u32 = 1) -> d3d12.RESOURCE_DESC {
	return {Dimension = .TEXTURE2D, Width = u64(w), Height = u32(h), DepthOrArraySize = u16(layers), MipLevels = u16(levels), Format = format, SampleDesc = {Count = samples}}
}

// Records, runs and waits for one command list.
submit :: proc(g: ^Gpu) {
	list_close(g.list)
	execute(g.queue, g.list)
	g.fence_value += 1
	queue_signal(g.queue, g.fence, g.fence_value)
	fence_wait(g.fence, g.fence_value)
	allocator_reset(g.allocator)
	list_reset(g.list, g.allocator)
}

// Uploads `data` into a new default-heap buffer (`flags`) in `state`.
upload_buffer :: proc(g: ^Gpu, data: []u8, state := SHADER_READ, flags := d3d12.RESOURCE_FLAGS{}) -> rawptr {
	desc := buffer_desc(len(data))
	// Created in COMMON (D3D12 ignores a buffer's initial state); the
	// copy promotes it to COPY_DEST.
	staging := create_committed_resource(g.device, .UPLOAD, &desc, d3d12.RESOURCE_STATE_GENERIC_READ)
	desc.Flags = flags
	buf := create_committed_resource(g.device, .DEFAULT, &desc, {})
	defer release(staging)
	copy(resource_map(staging)[:len(data)], data)
	resource_unmap(staging)
	copy_buffer_region(g.list, buf, 0, staging, 0, u64(len(data)))
	transition(g.list, buf, {.COPY_DEST}, state)
	submit(g)
	return buf
}

// A texture from `levels` mips of `layers` layers: subresource
// (level, layer)'s rows come from `rows` (at layer first_layer + layer,
// given `user`); created with `flags`, left in `state`.
upload_texture :: proc(g: ^Gpu, format: dxgi.FORMAT, w, h, layers, levels, bpp: int, rows: proc(level, layer, y: int, out: []u8, user: rawptr), first_layer := 0, user: rawptr = nil, flags := d3d12.RESOURCE_FLAGS{}, state := SHADER_READ) -> rawptr {
	desc := texture_desc(format, w, h, layers, levels)
	desc.Flags = flags
	tex := create_committed_resource(g.device, .DEFAULT, &desc, {.COPY_DEST})
	n := layers * levels
	footprints := make([]d3d12.PLACED_SUBRESOURCE_FOOTPRINT, n, context.temp_allocator)
	total := get_copyable_footprints(g.device, &desc, 0, u32(n), raw_data(footprints))
	sdesc := buffer_desc(int(total))
	staging := create_committed_resource(g.device, .UPLOAD, &sdesc, d3d12.RESOURCE_STATE_GENERIC_READ)
	defer release(staging)
	mapped := resource_map(staging)
	for layer in 0 ..< layers {
		for level in 0 ..< levels {
			fp := footprints[layer * levels + level]
			lw, lh := max(w >> uint(level), 1), max(h >> uint(level), 1)
			for y in 0 ..< lh {
				rows(level, first_layer + layer, y, mapped[int(fp.Offset) + y * int(fp.Footprint.RowPitch):][:lw * bpp], user)
			}
		}
	}
	resource_unmap(staging)
	for i in 0 ..< n {
		dst := d3d12.TEXTURE_COPY_LOCATION{pResource = (^d3d12.IResource)(tex), Type = .SUBRESOURCE_INDEX}
		dst.SubresourceIndex = u32(i)
		src := d3d12.TEXTURE_COPY_LOCATION{pResource = (^d3d12.IResource)(staging), Type = .PLACED_FOOTPRINT}
		src.PlacedFootprint = footprints[i]
		copy_texture_region(g.list, &dst, &src)
	}
	transition(g.list, tex, {.COPY_DEST}, state)
	submit(g)
	return tex
}

gpu_init :: proc(g: ^Gpu, hardware: bool) -> bool {
	name: string
	g.device, name = create_device(hardware)
	fmt.println("D3D12 on", name)
	if g.device == nil {
		return false
	}
	qd := d3d12.COMMAND_QUEUE_DESC{Type = .DIRECT}
	g.queue = create_command_queue(g.device, &qd)
	g.allocator = create_command_allocator(g.device)
	g.list = create_command_list(g.device, g.allocator)
	g.fence = create_fence(g.device)
	srv := d3d12.DESCRIPTOR_HEAP_DESC{Type = .CBV_SRV_UAV, NumDescriptors = 256, Flags = {.SHADER_VISIBLE}}
	smp := d3d12.DESCRIPTOR_HEAP_DESC{Type = .SAMPLER, NumDescriptors = 32, Flags = {.SHADER_VISIBLE}}
	rtv := d3d12.DESCRIPTOR_HEAP_DESC{Type = .RTV, NumDescriptors = len(Target)}
	g.srv_heap = create_descriptor_heap(g.device, &srv)
	g.sampler_heap = create_descriptor_heap(g.device, &smp)
	g.rtv_heap = create_descriptor_heap(g.device, &rtv)
	g.srv_step = descriptor_increment(g.device, .CBV_SRV_UAV)
	g.sampler_step = descriptor_increment(g.device, .SAMPLER)
	g.rtv_step = descriptor_increment(g.device, .RTV)
	for t in Target {
		desc := texture_desc(TARGET_FORMATS[t], W, H, 1, 1, 4 if t == .Msaa4 else 1)
		desc.Flags = {.ALLOW_RENDER_TARGET}
		clear := d3d12.CLEAR_VALUE{Format = TARGET_FORMATS[t]}
		clear.Color = TARGET_CLEARS[t]
		g.targets[t] = create_committed_resource(g.device, .DEFAULT, &desc, {.RENDER_TARGET}, &clear)
		h := heap_cpu_start(g.rtv_heap)
		h.ptr += uint(int(t) * int(g.rtv_step))
		create_render_target_view(g.device, g.targets[t], h)
	}
	rd := texture_desc(.R8G8B8A8_UNORM, W, H, 1, 1)
	g.resolved = create_committed_resource(g.device, .DEFAULT, &rd, {.RESOLVE_DEST})
	rb := buffer_desc(W * H * 16)
	g.readback = create_committed_resource(g.device, .READBACK, &rb, {.COPY_DEST})
	cb := buffer_desc(1 << 16)
	g.constants = create_committed_resource(g.device, .UPLOAD, &cb, d3d12.RESOURCE_STATE_GENERIC_READ)
	qb := buffer_desc(size_of(parity.QUAD))
	g.quad = create_committed_resource(g.device, .UPLOAD, &qb, d3d12.RESOURCE_STATE_GENERIC_READ)
	mem.copy(resource_map(g.quad), &parity.QUAD, size_of(parity.QUAD))
	resource_unmap(g.quad)

	g.resources[.Words] = upload_buffer(g, slice.to_bytes(parity.test_words[:]))
	layer_rows :: proc(level, layer, y: int, out: []u8, _: rawptr) {
		n := TEX >> uint(level)
		for i in 0 ..< len(out) {
			out[i] = parity.layer_pixel(parity.level_offset(level) + y * n * 4 + i, layer)
		}
	}
	for s, l in ([3]Srv{.Tex0, .Tex1, .Tex2}) {
		g.resources[s] = upload_texture(g, .R8G8B8A8_UNORM, TEX, TEX, 1, LEVELS, 4, layer_rows, first_layer = l)
	}
	g.resources[.Layers] = upload_texture(g, .R8G8B8A8_UNORM, TEX, TEX, LAYERS, LEVELS, 4, layer_rows)
	g.resources[.R32f] = upload_texture(g, .R32_FLOAT, parity.BOUND_R_W, parity.BOUND_R_H, 1, 1, 4, proc(level, layer, y: int, out: []u8, _: rawptr) {
		copy(out, slice.to_bytes(parity.bound_r_texels[y * parity.BOUND_R_W:][:parity.BOUND_R_W]))
	})
	g.resources[.Rg32f] = upload_texture(g, .R32G32_FLOAT, parity.BOUND_RG_W, parity.BOUND_RG_H, 1, 1, 8, proc(level, layer, y: int, out: []u8, _: rawptr) {
		copy(out, slice.to_bytes(parity.bound_rg_texels[y * parity.BOUND_RG_W:][:parity.BOUND_RG_W]))
	})
	g.resources[.Rgba32f] = upload_texture(g, .R32G32B32A32_FLOAT, parity.BOUND_RGBA_W, parity.BOUND_RGBA_H, 1, 1, 16, proc(level, layer, y: int, out: []u8, _: rawptr) {
		copy(out, slice.to_bytes(parity.bound_rgba_texels[y * parity.BOUND_RGBA_W:][:parity.BOUND_RGBA_W]))
	})

	m, errors, ok := oksl.compile(oksl.LAYER_VERTEX_GLSL, {profile = .Glsl, stage = .Vertex, dxbc = true})
	if !ok {
		fmt.eprint(oksl.format_errors(errors))
		return false
	}
	g.vs = slice.clone(m.dxbc)
	oksl.destroy(m)
	return true
}

// ---- drawing ----------------------------------------------------------------------------

Draw :: struct {
	vs, ps:             []u8,
	vs_block, ps_block: []u8,
	ps_srvs:            []Srv,
	ps_samplers:        []Smp,
	vs_srvs:            []Srv,
	// Vertex attributes: TEXCOORD<location> from the quad's position (0)
	// or uv (8).
	attributes:         [][2]u32,
	vertices, first:    u32,
	target:             Target,
}

LAYER_ATTRIBUTES := [][2]u32{{0, 0}, {1, 8}}

write_srv :: proc(g: ^Gpu, s: Srv, slot: int) {
	desc := d3d12.SHADER_RESOURCE_VIEW_DESC{Shader4ComponentMapping = 0x1688}
	switch s {
	case .Tex0, .Tex1, .Tex2:
		desc.Format, desc.ViewDimension = .R8G8B8A8_UNORM, .TEXTURE2D
		desc.Texture2D = {MipLevels = LEVELS}
	case .Layers:
		desc.Format, desc.ViewDimension = .R8G8B8A8_UNORM, .TEXTURE2DARRAY
		desc.Texture2DArray = {MipLevels = LEVELS, ArraySize = LAYERS}
	case .R32f, .Rg32f, .Rgba32f:
		formats := [?]dxgi.FORMAT{.R32_FLOAT, .R32G32_FLOAT, .R32G32B32A32_FLOAT}
		desc.Format = formats[int(s) - int(Srv.R32f)]
		desc.ViewDimension = .TEXTURE2D
		desc.Texture2D = {MipLevels = 1}
	case .Words, .Corners:
		desc.Format, desc.ViewDimension = .R32_TYPELESS, .BUFFER
		desc.Buffer = {NumElements = parity.WORDS if s == .Words else len(parity.QUAD) * size_of(parity.Corner) / 4, Flags = {.RAW}}
	}
	h := heap_cpu_start(g.srv_heap)
	h.ptr += uint(slot * int(g.srv_step))
	create_shader_resource_view(g.device, g.resources[s], &desc, h)
}

write_sampler :: proc(g: ^Gpu, s: Smp, slot: int) {
	desc: d3d12.SAMPLER_DESC
	switch s {
	case .Repeat:
		desc = {Filter = .MIN_MAG_MIP_LINEAR, AddressU = .WRAP, AddressV = .WRAP, AddressW = .WRAP, MaxLOD = LEVELS - 1}
	case .Near_Clamp:
		desc = {Filter = .MIN_MAG_MIP_POINT, AddressU = .CLAMP, AddressV = .CLAMP, AddressW = .CLAMP}
	case .Linear_Clamp:
		desc = {Filter = .MIN_MAG_LINEAR_MIP_POINT, AddressU = .CLAMP, AddressV = .CLAMP, AddressW = .CLAMP}
	}
	h := heap_cpu_start(g.sampler_heap)
	h.ptr += uint(slot * int(g.sampler_step))
	create_sampler(g.device, &desc, h)
}

// Draws and returns the target's pixels (RGBA8, or RGBA32F's bytes), or
// why it could not.
draw :: proc(g: ^Gpu, dr: Draw) -> (px: []u8, err: string) {
	// The root signature: SDL_GPU's spaces.
	params := make([dynamic]d3d12.ROOT_PARAMETER, context.temp_allocator)
	cbv :: proc(space: u32, vis: d3d12.SHADER_VISIBILITY) -> d3d12.ROOT_PARAMETER {
		p := d3d12.ROOT_PARAMETER{ParameterType = .CBV, ShaderVisibility = vis}
		p.Descriptor = {ShaderRegister = 0, RegisterSpace = space}
		return p
	}
	append(&params, cbv(1, .VERTEX), cbv(3, .PIXEL))
	ranges: [3]d3d12.DESCRIPTOR_RANGE
	table :: proc(params: ^[dynamic]d3d12.ROOT_PARAMETER, r: ^d3d12.DESCRIPTOR_RANGE, vis: d3d12.SHADER_VISIBILITY) -> u32 {
		p := d3d12.ROOT_PARAMETER{ParameterType = .DESCRIPTOR_TABLE, ShaderVisibility = vis}
		p.DescriptorTable = {NumDescriptorRanges = 1, pDescriptorRanges = r}
		append(params, p)
		return u32(len(params) - 1)
	}
	ps_srv_param, ps_smp_param, vs_srv_param := -1, -1, -1
	if len(dr.ps_srvs) > 0 {
		ranges[0] = {RangeType = .SRV, NumDescriptors = u32(len(dr.ps_srvs)), RegisterSpace = 2}
		ps_srv_param = int(table(&params, &ranges[0], .PIXEL))
	}
	if len(dr.ps_samplers) > 0 {
		ranges[1] = {RangeType = .SAMPLER, NumDescriptors = u32(len(dr.ps_samplers)), RegisterSpace = 2}
		ps_smp_param = int(table(&params, &ranges[1], .PIXEL))
	}
	if len(dr.vs_srvs) > 0 {
		ranges[2] = {RangeType = .SRV, NumDescriptors = u32(len(dr.vs_srvs)), RegisterSpace = 0}
		vs_srv_param = int(table(&params, &ranges[2], .VERTEX))
	}
	rsd := d3d12.ROOT_SIGNATURE_DESC{NumParameters = u32(len(params)), pParameters = raw_data(params), Flags = {.ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT}}
	blob, eblob: rawptr
	if D3D12SerializeRootSignature(&rsd, ._1_0, &blob, &eblob) < 0 {
		return nil, "root signature serialization failed"
	}
	bytes := blob_data(blob)
	rs, hr := create_root_signature(g.device, raw_data(bytes), len(bytes))
	release(blob)
	if hr < 0 {
		return nil, "CreateRootSignature failed"
	}
	defer release(rs)

	elems := make([]d3d12.INPUT_ELEMENT_DESC, len(dr.attributes), context.temp_allocator)
	for a, i in dr.attributes {
		elems[i] = {SemanticName = "TEXCOORD", SemanticIndex = a[0], Format = .R32G32_FLOAT, AlignedByteOffset = a[1], InputSlotClass = .PER_VERTEX_DATA}
	}
	pd := d3d12.GRAPHICS_PIPELINE_STATE_DESC {
		pRootSignature = (^d3d12.IRootSignature)(rs),
		VS = {raw_data(dr.vs), uint(len(dr.vs))},
		PS = {raw_data(dr.ps), uint(len(dr.ps))},
		SampleMask = 0xffffffff,
		// SDL_GPU's default front face: counter-clockwise on screen.
		RasterizerState = {FillMode = .SOLID, CullMode = .NONE, FrontCounterClockwise = true, DepthClipEnable = true},
		InputLayout = {raw_data(elems), u32(len(elems))},
		PrimitiveTopologyType = .TRIANGLE,
		NumRenderTargets = 1,
		SampleDesc = {Count = 4 if dr.target == .Msaa4 else 1},
	}
	pd.RTVFormats[0] = TARGET_FORMATS[dr.target]
	pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf
	pd.BlendState.RenderTarget[0].LogicOp = .NOOP
	pso, phr := create_graphics_pipeline_state(g.device, &pd)
	if phr < 0 {
		return nil, fmt.tprintf("CreateGraphicsPipelineState failed: hr %x", u32(phr))
	}
	defer release(pso)

	// Descriptors: the pixel shader's SRVs, then the vertex shader's.
	for s, i in dr.ps_srvs {
		write_srv(g, s, i)
	}
	for s, i in dr.vs_srvs {
		write_srv(g, s, len(dr.ps_srvs) + i)
	}
	for s, i in dr.ps_samplers {
		write_sampler(g, s, i)
	}
	cmap := resource_map(g.constants)
	copy(cmap[0:1 << 15], dr.vs_block)
	copy(cmap[1 << 15:1 << 16], dr.ps_block)
	resource_unmap(g.constants)
	cbase := gpu_address(g.constants)

	l := g.list
	rt := g.targets[dr.target]
	rtv := heap_cpu_start(g.rtv_heap)
	rtv.ptr += uint(int(dr.target) * int(g.rtv_step))
	heaps := [2]rawptr{g.srv_heap, g.sampler_heap}
	set_descriptor_heaps(l, heaps[:])
	set_graphics_root_signature(l, rs)
	set_pipeline_state(l, pso)
	set_graphics_root_cbv(l, 0, cbase)
	set_graphics_root_cbv(l, 1, cbase + 1 << 15)
	srv_gpu := heap_gpu_start(g.srv_heap)
	if ps_srv_param >= 0 {
		set_graphics_root_table(l, u32(ps_srv_param), srv_gpu)
	}
	if ps_smp_param >= 0 {
		set_graphics_root_table(l, u32(ps_smp_param), heap_gpu_start(g.sampler_heap))
	}
	if vs_srv_param >= 0 {
		h := srv_gpu
		h.ptr += u64(len(dr.ps_srvs) * int(g.srv_step))
		set_graphics_root_table(l, u32(vs_srv_param), h)
	}
	set_topology(l, .TRIANGLELIST)
	view := d3d12.VERTEX_BUFFER_VIEW{gpu_address(g.quad), 6 * 16, 16}
	set_vertex_buffer(l, &view)
	set_viewport_scissor(l, W, H)
	set_render_target(l, &rtv)
	clear_render_target(l, rtv, TARGET_CLEARS[dr.target])
	draw_instanced(l, dr.vertices, dr.first)

	src := rt
	bpp := 16 if dr.target == .Rgba32f else 4
	if dr.target == .Msaa4 {
		transition(l, rt, {.RENDER_TARGET}, {.RESOLVE_SOURCE})
		resolve_subresource(l, g.resolved, rt, .R8G8B8A8_UNORM)
		transition(l, rt, {.RESOLVE_SOURCE}, {.RENDER_TARGET})
		transition(l, g.resolved, {.RESOLVE_DEST}, {.COPY_SOURCE})
		src = g.resolved
	} else {
		transition(l, rt, {.RENDER_TARGET}, {.COPY_SOURCE})
	}
	dst := d3d12.TEXTURE_COPY_LOCATION{pResource = (^d3d12.IResource)(g.readback), Type = .PLACED_FOOTPRINT}
	dst.PlacedFootprint = {Footprint = {Format = .R32G32B32A32_FLOAT if bpp == 16 else .R8G8B8A8_UNORM, Width = W, Height = H, Depth = 1, RowPitch = u32(W * bpp)}}
	from := d3d12.TEXTURE_COPY_LOCATION{pResource = (^d3d12.IResource)(src), Type = .SUBRESOURCE_INDEX}
	copy_texture_region(l, &dst, &from)
	if dr.target == .Msaa4 {
		transition(l, g.resolved, {.COPY_SOURCE}, {.RESOLVE_DEST})
	} else {
		transition(l, rt, {.COPY_SOURCE}, {.RENDER_TARGET})
	}
	submit(g)
	mapped := resource_map(g.readback)
	px = slice.clone(mapped[:W * H * bpp])
	resource_unmap(g.readback)
	return px, ""
}

// What a module reads: its channels (the array, or sampler2D channel n's
// layer n % LAYERS) and storage buffers.
module_srvs :: proc(m: ^oksl.Module) -> (srvs: []Srv, samplers: []Smp) {
	s := make([dynamic]Srv, context.temp_allocator)
	for ch in 0 ..< m.num_samplers {
		append(&s, Srv.Layers if ch in m.channel_arrays else Srv(int(Srv.Tex0) + ch % LAYERS))
	}
	for _ in m.storage_buffers {
		append(&s, Srv.Words)
	}
	smp := make([]Smp, m.num_samplers, context.temp_allocator)
	return s[:], smp
}

// ---- the checks -------------------------------------------------------------------------

main :: proc() {
	args := os.args[1:]
	hardware := len(args) > 0 && args[0] == "-hardware"
	if hardware {
		args = args[1:]
	}
	out_dir := args[0] if len(args) > 0 else "d3d12_parity_out"
	data_dir := args[1] if len(args) > 1 else "testdata"
	os.make_directory_all(out_dir)
	parity.make_data()
	g := Gpu{warp = ODIN_OS == .Windows && !hardware}
	if !gpu_init(&g, hardware) || debug_messages() > 0 {
		fmt.eprintln("no D3D12 device")
		os.exit(2)
	}

	infos, _ := os.read_all_directory_by_path(data_dir, context.allocator)
	names := make([dynamic]string)
	comp_names := make([dynamic]string)
	for fi in infos {
		if o, known := oksl.options_of_path(fi.name); known && o.stage == .Fragment {
			append(&names, fi.fullpath)
		} else if known && o.stage == .Compute {
			append(&comp_names, fi.fullpath)
		}
	}
	slice.sort(names[:])
	slice.sort(comp_names[:])
	failed := 0
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	for path in names {
		src, _ := os.read_entire_file(path, context.allocator)
		options, _ := oksl.options_of_path(path)
		ok := fragment_check(&g, filepath.stem(path), string(src), options, {}, out_dir, &scratch)
		if debug_messages() > 0 || !ok {
			failed += 1
		}
		delete(src)
		free_all(context.temp_allocator)
	}
	// The emitter's own cases (edge.odin).
	for e in EDGE_SHADERS {
		// A float check's Shadertoy colour as it is (not clamped).
		ok := fragment_check(&g, e.name, e.source, {profile = .Glsl if e.glsl else .Shadertoy, unclamped = e.f32}, e, out_dir, &scratch)
		if debug_messages() > 0 || !ok {
			failed += 1
		}
		free_all(context.temp_allocator)
	}
	report :: proc(name: string, d: int, ok: bool, failed: ^int) {
		good := ok && d <= 2
		fmt.printfln("%-20s max diff %d  %s", name, d, "ok" if good else "MISMATCH")
		if debug_messages() > 0 || !good {
			failed^ += 1
		}
	}
	tap_d, tap_ok := tap_check(&g)
	report("layer.vert tap", tap_d, tap_ok, &failed)
	id_d, id_ok := vertex_id_check(&g)
	report("gl_VertexID", id_d, id_ok, &failed)
	mask_d, mask_ok := sample_mask_check(&g)
	report("gl_SampleMask", mask_d, mask_ok, &failed)
	face_detail, face_ok := facing_check(&g)
	fmt.printfln("%-20s %s  %s", "facing", face_detail, "ok" if face_ok else "MISMATCH")
	if debug_messages() > 0 || !face_ok {
		failed += 1
	}
	// Compute: every testdata compute shader, image_formats, the emitter's
	// own cases (edge_compute.odin).
	compute_report :: proc(name, detail: string, ok: bool, failed: ^int) {
		fmt.printfln("%-20s %s  %s", name, detail, "ok" if ok else "MISMATCH")
		if debug_messages() > 0 || !ok {
			failed^ += 1
		}
		free_all(context.temp_allocator)
	}
	for path in comp_names {
		src, _ := os.read_entire_file(path, context.allocator)
		options, _ := oksl.options_of_path(path)
		name := filepath.base(path)
		detail, ok := compute_check(&g, name, string(src), options, parity.compute_shape(name))
		compute_report(name, detail, ok, &failed)
		delete(src)
	}
	fmt_detail, fmt_ok := compute_check(&g, "image_formats", parity.IMAGE_FORMATS_CS, {profile = .Glsl, stage = .Compute}, parity.compute_shape(""))
	compute_report("image_formats", fmt_detail, fmt_ok, &failed)
	for e in EDGE_COMPUTE {
		detail, ok := compute_check(&g, e.name, e.source, {profile = .Glsl, stage = .Compute}, e.shape)
		compute_report(e.name, detail, ok, &failed)
	}
	checks := len(names) + len(EDGE_SHADERS) + 4 + len(comp_names) + 1 + len(EDGE_COMPUTE)
	fmt.printfln("%d/%d checks match (images in %s)", checks - failed, checks, out_dir)
	if failed > 0 {
		os.exit(1)
	}
}

// Renders a fragment shader on the layer quad through D3D12 and with the
// CPU and compares them: in 8 bits under gpu_parity's rule, or (an edge
// shader's `f32`) on a float target.
fragment_check :: proc(g: ^Gpu, name, src: string, options: oksl.Options, edge: Edge_Shader, out_dir: string, scratch: ^oksl.Scratch) -> bool {
	view := [4]f32{W, H, 0, 0}
	view_block := slice.to_bytes(view[:])
	o := options
	o.dxbc = true
	m, errors, ok := oksl.compile(src, o)
	defer oksl.destroy(m)
	if !ok {
		fmt.printfln("%-20s compile error:\n%s", name, oksl.format_errors(errors, context.temp_allocator))
		return false
	}
	block := parity.uniform_block(m)
	defer delete(block)
	if edge.setup != nil {
		edge.setup(m, block)
	}
	srvs, samplers := module_srvs(m)
	if name == "bound_channels" {
		srvs = {.R32f, .Rg32f, .Rgba32f, .R32f, .Rg32f, .Rgba32f}
		samplers = {.Near_Clamp, .Near_Clamp, .Near_Clamp, .Linear_Clamp, .Linear_Clamp, .Linear_Clamp}
	}
	f32_target := edge.f32 || name == "bound_channels"
	gpu_px, err := draw(g, {vs = g.vs, ps = m.dxbc, vs_block = view_block, ps_block = block, ps_srvs = srvs, ps_samplers = samplers, attributes = LAYER_ATTRIBUTES, vertices = 6, target = .Rgba32f if f32_target else .Rgba8})
	if err != "" {
		fmt.printfln("%-20s D3D12 draw failed: %s", name, err)
		return false
	}
	defer delete(gpu_px)
	if name == "bound_channels" {
		detail, bc_ok := parity.bound_channels_compare(m, block, slice.reinterpret([][4]f32, gpu_px), "d3d12", out_dir, scratch, g.warp)
		fmt.printfln("%-20s %s  %s", name, detail, "ok" if bc_ok else "MISMATCH")
		return bc_ok
	}
	if edge.f32 {
		return parity.compare_f32(name, parity.fragment_reference_f32(m, block, scratch), slice.reinterpret([][4]f32, gpu_px), edge.tol, edge.skip)
	}
	return parity.compare_rgba8(name, "d3d12", out_dir, parity.fragment_reference(m, block, scratch), gpu_px)
}

// The layer vertex shader with a tap, its position drawn as colour.
tap_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	vm, verrors, vok := oksl.compile(oksl.LAYER_VERTEX_GLSL, parity.tap_options(true))
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(parity.TAP_FRAGMENT, {profile = .Glsl, dxbc = true})
	defer oksl.destroy(fm)
	if !vok || !fok {
		fmt.eprint(oksl.format_errors(verrors), oksl.format_errors(ferrors))
		return 0, false
	}
	attrs := make([][2]u32, len(vm.stage_inputs), context.temp_allocator)
	for a, i in vm.stage_inputs {
		attrs[i] = {u32(a.location), 0 if a.name == "position" else 8}
	}
	block := parity.tap_block(vm)
	gpu_px, err := draw(g, {vs = vm.dxbc, ps = fm.dxbc, vs_block = block, attributes = attrs, vertices = 6})
	if err != "" {
		fmt.eprintln(err)
		return 0, false
	}
	defer delete(gpu_px)
	return parity.tap_compare(vm, block, gpu_px), true
}

// gl_VertexID from vertex 0: D3D's SV_VertexID leaves out the draw's
// first vertex (vkd3d's VertexIndex - BaseVertex), where Vulkan's
// VertexIndex and Metal's vertex_id count it (gpu_parity draws from 2).
vertex_id_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	vm, verrors, vok := oksl.compile(parity.VERTEX_ID_VS, {profile = .Glsl, stage = .Vertex, dxbc = true})
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(parity.VERTEX_ID_FS, {profile = .Glsl, dxbc = true})
	defer oksl.destroy(fm)
	if !vok || !fok {
		fmt.eprint(oksl.format_errors(verrors), oksl.format_errors(ferrors))
		return 0, false
	}
	corners := parity.vertex_id_corners(0)
	if g.resources[.Corners] == nil {
		g.resources[.Corners] = upload_buffer(g, slice.to_bytes(corners))
	}
	block := parity.vertex_id_block(vm)
	gpu_px, err := draw(g, {vs = vm.dxbc, ps = fm.dxbc, vs_block = block, vs_srvs = {.Corners}, vertices = 6})
	if err != "" {
		fmt.eprintln(err)
		return 0, false
	}
	defer delete(gpu_px)
	return parity.vertex_id_compare(vm, block, corners, 0, gpu_px), true
}

// gl_SampleMask, 4x MSAA resolved over black.
sample_mask_check :: proc(g: ^Gpu) -> (max_d: int, ok: bool) {
	m, errors, cok := oksl.compile(parity.SAMPLE_MASK_FS, {profile = .Glsl, dxbc = true})
	defer oksl.destroy(m)
	if !cok {
		fmt.eprint(oksl.format_errors(errors))
		return 0, false
	}
	view := [4]f32{W, H, 0, 0}
	gpu_px, err := draw(g, {vs = g.vs, ps = m.dxbc, vs_block = slice.to_bytes(view[:]), attributes = LAYER_ATTRIBUTES, vertices = 6, target = .Msaa4})
	if err != "" {
		fmt.eprintln(err)
		return 0, false
	}
	defer delete(gpu_px)
	return parity.sample_mask_compare(m, gpu_px), true
}

// Both windings, perspective (w != 1, z != 0) and flat integers
// (parity.FACING_VS / FACING_FS) on a float target.
facing_check :: proc(g: ^Gpu) -> (string, bool) {
	vm, verrors, vok := oksl.compile(parity.FACING_VS, {profile = .Glsl, stage = .Vertex, dxbc = true})
	defer oksl.destroy(vm)
	fm, ferrors, fok := oksl.compile(parity.FACING_FS, {profile = .Glsl, dxbc = true})
	defer oksl.destroy(fm)
	if !vok || !fok {
		return fmt.tprint(oksl.format_errors(verrors), oksl.format_errors(ferrors)), false
	}
	gpu_px, err := draw(g, {vs = vm.dxbc, ps = fm.dxbc, vertices = 6, target = .Rgba32f})
	if err != "" {
		return err, false
	}
	defer delete(gpu_px)
	return parity.facing_compare(vm, slice.reinterpret([][4]f32, gpu_px))
}
