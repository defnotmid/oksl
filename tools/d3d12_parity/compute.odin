#+build linux amd64, windows amd64
package d3d12_parity

// Compute checks: a compute shader's DXBC dispatched with SDL_GPU's D3D12
// compute root signature (SDL_gpu_d3d12.c's
// D3D12_INTERNAL_CreateComputeRootSignature: one table each for the
// samplers (s0.. space0), the sampled textures (t0.. space0), the
// read-only storage textures (t from num_samplers, space0), the read-only
// storage buffers (t after them), the read-write storage textures (u0..
// space1) and the read-write storage buffers (u after them), and the
// uniform block as a root CBV at b0 space2), on tools/parity's data, and
// compared with the CPU under gpu_parity's rules (parity.compute_compare).

import "core:fmt"
import "core:mem"
import "core:slice"
import "core:strings"
import oksl "../.."
import parity "../parity"
import d3d12 "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"

image_dxgi_format :: proc(f: oksl.Image_Format) -> dxgi.FORMAT {
	switch f {
	case .None:
	case .Rgba8:
		return .R8G8B8A8_UNORM
	case .Rgba16f:
		return .R16G16B16A16_FLOAT
	case .Rgba32f:
		return .R32G32B32A32_FLOAT
	case .R32f:
		return .R32_FLOAT
	case .R32ui:
		return .R32_UINT
	case .R32i:
		return .R32_SINT
	}
	return .UNKNOWN
}

// Compiles `src` with DXBC, gl_NumWorkGroups from the block
// (Options.num_work_groups_uniform), and runs compute_module_check.
compute_check :: proc(g: ^Gpu, name, src: string, options: oksl.Options, shape: parity.Compute_Shape) -> (string, bool) {
	o := options
	o.dxbc = true
	o.num_work_groups_uniform = true
	m, errs, ok := oksl.compile(src, o)
	defer oksl.destroy(m)
	if !ok {
		return strings.trim_space(oksl.format_errors(errs, context.temp_allocator)), false
	}
	return compute_module_check(g, m, shape)
}

compute_module_check :: proc(g: ^Gpu, m: ^oksl.Module, shape: parity.Compute_Shape) -> (string, bool) {
	cd, cd_err := parity.compute_data(m, shape)
	if cd_err != "" {
		return cd_err, false
	}
	nb, ni := len(m.storage_buffers), len(m.storage_images)
	ns := m.num_samplers
	ro_tex, ro_buf := m.num_readonly_storage_textures, m.num_readonly_storage_buffers
	rw_tex, rw_buf := ni - ro_tex, nb - ro_buf

	// SDL_GPU's root signature: its tables in its order, then the CBV.
	params := make([dynamic]d3d12.ROOT_PARAMETER, context.temp_allocator)
	ranges: [6]d3d12.DESCRIPTOR_RANGE
	Table :: struct {
		param, first: int,
	}
	tables := make([dynamic]Table, context.temp_allocator)
	next_desc := 0
	add_table :: proc(params: ^[dynamic]d3d12.ROOT_PARAMETER, tables: ^[dynamic]Table, r: ^d3d12.DESCRIPTOR_RANGE, first: int) {
		p := d3d12.ROOT_PARAMETER{ParameterType = .DESCRIPTOR_TABLE, ShaderVisibility = .ALL}
		p.DescriptorTable = {NumDescriptorRanges = 1, pDescriptorRanges = r}
		append(params, p)
		append(tables, Table{len(params) - 1, first})
	}
	sampler_param := -1
	if ns > 0 {
		ranges[0] = {RangeType = .SAMPLER, NumDescriptors = u32(ns), RegisterSpace = 0}
		p := d3d12.ROOT_PARAMETER{ParameterType = .DESCRIPTOR_TABLE, ShaderVisibility = .ALL}
		p.DescriptorTable = {NumDescriptorRanges = 1, pDescriptorRanges = &ranges[0]}
		append(&params, p)
		sampler_param = len(params) - 1
		ranges[1] = {RangeType = .SRV, NumDescriptors = u32(ns), RegisterSpace = 0}
		add_table(&params, &tables, &ranges[1], next_desc)
		next_desc += ns
	}
	if ro_tex > 0 {
		ranges[2] = {RangeType = .SRV, NumDescriptors = u32(ro_tex), BaseShaderRegister = u32(ns), RegisterSpace = 0}
		add_table(&params, &tables, &ranges[2], next_desc)
		next_desc += ro_tex
	}
	if ro_buf > 0 {
		ranges[3] = {RangeType = .SRV, NumDescriptors = u32(ro_buf), BaseShaderRegister = u32(ns + ro_tex), RegisterSpace = 0}
		add_table(&params, &tables, &ranges[3], next_desc)
		next_desc += ro_buf
	}
	if rw_tex > 0 {
		ranges[4] = {RangeType = .UAV, NumDescriptors = u32(rw_tex), RegisterSpace = 1}
		add_table(&params, &tables, &ranges[4], next_desc)
		next_desc += rw_tex
	}
	if rw_buf > 0 {
		ranges[5] = {RangeType = .UAV, NumDescriptors = u32(rw_buf), BaseShaderRegister = u32(rw_tex), RegisterSpace = 1}
		add_table(&params, &tables, &ranges[5], next_desc)
		next_desc += rw_buf
	}
	cbv_param := -1
	if m.num_uniform_buffers > 0 {
		p := d3d12.ROOT_PARAMETER{ParameterType = .CBV, ShaderVisibility = .ALL}
		p.Descriptor = {ShaderRegister = 0, RegisterSpace = 2}
		append(&params, p)
		cbv_param = len(params) - 1
	}
	rsd := d3d12.ROOT_SIGNATURE_DESC{NumParameters = u32(len(params)), pParameters = raw_data(params)}
	blob, eblob: rawptr
	if D3D12SerializeRootSignature(&rsd, ._1_0, &blob, &eblob) < 0 {
		return "root signature serialization failed", false
	}
	rs_bytes := blob_data(blob)
	rs, rhr := create_root_signature(g.device, raw_data(rs_bytes), len(rs_bytes))
	release(blob)
	if rhr < 0 {
		return "CreateRootSignature failed", false
	}
	defer release(rs)
	pd := d3d12.COMPUTE_PIPELINE_STATE_DESC {
		pRootSignature = (^d3d12.IRootSignature)(rs),
		CS             = {raw_data(m.dxbc), uint(len(m.dxbc))},
	}
	pso, phr := create_compute_pipeline_state(g.device, &pd)
	if phr < 0 {
		return fmt.tprintf("CreateComputePipelineState failed: hr %x", u32(phr)), false
	}
	defer release(pso)

	// The resources, their contents uploaded; the writable ones in
	// UNORDERED_ACCESS.
	UAV_STATE :: d3d12.RESOURCE_STATES{.UNORDERED_ACCESS}
	bufs := make([]rawptr, nb, context.temp_allocator)
	imgs := make([]rawptr, ni, context.temp_allocator)
	defer {
		for b in bufs {
			release(b)
		}
		for t in imgs {
			release(t)
		}
	}
	for sb, bi in m.storage_buffers {
		data := slice.to_bytes(cd.init_bufs[bi])
		if sb.readonly {
			bufs[bi] = upload_buffer(g, data)
		} else {
			bufs[bi] = upload_buffer(g, data, UAV_STATE, {.ALLOW_UNORDERED_ACCESS})
		}
	}
	Image_Rows :: struct {
		data:   []u8,
		stride: int,
	}
	for im, ii in m.storage_images {
		ci := cd.cpu_imgs[ii]
		bpp := parity.image_texel_bytes(im.format)
		rows := Image_Rows{cd.init_imgs[ii], ci.width * bpp}
		copy_rows :: proc(level, layer, y: int, out: []u8, user: rawptr) {
			r := (^Image_Rows)(user)
			copy(out, r.data[y * r.stride:][:r.stride])
		}
		if im.readonly {
			imgs[ii] = upload_texture(g, image_dxgi_format(im.format), ci.width, ci.height, 1, 1, bpp, copy_rows, user = &rows)
		} else {
			imgs[ii] = upload_texture(g, image_dxgi_format(im.format), ci.width, ci.height, 1, 1, bpp, copy_rows, user = &rows, flags = {.ALLOW_UNORDERED_ACCESS}, state = UAV_STATE)
		}
	}

	// The descriptors, in the tables' order.
	slot := 0
	handle :: proc(g: ^Gpu, slot: int) -> d3d12.CPU_DESCRIPTOR_HANDLE {
		h := heap_cpu_start(g.srv_heap)
		h.ptr += uint(slot * int(g.srv_step))
		return h
	}
	for ch in 0 ..< ns {
		write_srv(g, Srv.Layers if ch in m.channel_arrays else Srv(int(Srv.Tex0) + ch % LAYERS), slot)
		write_sampler(g, .Repeat, ch)
		slot += 1
	}
	texture_srv :: proc(f: oksl.Image_Format) -> d3d12.SHADER_RESOURCE_VIEW_DESC {
		desc := d3d12.SHADER_RESOURCE_VIEW_DESC{Format = image_dxgi_format(f), ViewDimension = .TEXTURE2D, Shader4ComponentMapping = 0x1688}
		desc.Texture2D = {MipLevels = 1}
		return desc
	}
	for ii in 0 ..< ro_tex {
		desc := texture_srv(m.storage_images[ii].format)
		create_shader_resource_view(g.device, imgs[ii], &desc, handle(g, slot))
		slot += 1
	}
	for bi in 0 ..< ro_buf {
		desc := d3d12.SHADER_RESOURCE_VIEW_DESC{Format = .R32_TYPELESS, ViewDimension = .BUFFER, Shader4ComponentMapping = 0x1688}
		desc.Buffer = {NumElements = u32(cd.buf_words), Flags = {.RAW}}
		create_shader_resource_view(g.device, bufs[bi], &desc, handle(g, slot))
		slot += 1
	}
	for ii in ro_tex ..< ni {
		desc := d3d12.UNORDERED_ACCESS_VIEW_DESC{Format = image_dxgi_format(m.storage_images[ii].format), ViewDimension = .TEXTURE2D}
		create_unordered_access_view(g.device, imgs[ii], &desc, handle(g, slot))
		slot += 1
	}
	for bi in ro_buf ..< nb {
		desc := d3d12.UNORDERED_ACCESS_VIEW_DESC{Format = .R32_TYPELESS, ViewDimension = .BUFFER}
		desc.Buffer = {NumElements = u32(cd.buf_words), Flags = {.RAW}}
		create_unordered_access_view(g.device, bufs[bi], &desc, handle(g, slot))
		slot += 1
	}
	cmap := resource_map(g.constants)
	copy(cmap[:1 << 16], cd.block)
	resource_unmap(g.constants)

	l := g.list
	heaps := [2]rawptr{g.srv_heap, g.sampler_heap}
	set_descriptor_heaps(l, heaps[:])
	set_compute_root_signature(l, rs)
	set_pipeline_state(l, pso)
	if sampler_param >= 0 {
		set_compute_root_table(l, u32(sampler_param), heap_gpu_start(g.sampler_heap))
	}
	for t in tables {
		h := heap_gpu_start(g.srv_heap)
		h.ptr += u64(t.first * int(g.srv_step))
		set_compute_root_table(l, u32(t.param), h)
	}
	if cbv_param >= 0 {
		set_compute_root_cbv(l, u32(cbv_param), gpu_address(g.constants))
	}
	dispatch(l, u32(cd.groups.x), u32(cd.groups.y), u32(cd.groups.z))

	// The writable resources back: buffers, then each image at its
	// footprint (rows RowPitch apart).
	footprints := make([]d3d12.PLACED_SUBRESOURCE_FOOTPRINT, rw_tex, context.temp_allocator)
	total := u64(rw_buf * cd.buf_words * 4)
	for i in 0 ..< rw_tex {
		ci := cd.cpu_imgs[ro_tex + i]
		desc := texture_desc(image_dxgi_format(ci.format), ci.width, ci.height, 1, 1)
		size := get_copyable_footprints(g.device, &desc, 0, 1, &footprints[i])
		total = (total + 511) &~ 511
		footprints[i].Offset = total
		total += size
	}
	rbd := buffer_desc(int(max(total, 4)))
	readback := create_committed_resource(g.device, .READBACK, &rbd, {.COPY_DEST})
	defer release(readback)
	for i in 0 ..< rw_buf {
		b := bufs[ro_buf + i]
		transition(l, b, UAV_STATE, {.COPY_SOURCE})
		copy_buffer_region(l, readback, u64(i * cd.buf_words * 4), b, 0, u64(cd.buf_words * 4))
	}
	for i in 0 ..< rw_tex {
		t := imgs[ro_tex + i]
		transition(l, t, UAV_STATE, {.COPY_SOURCE})
		dst := d3d12.TEXTURE_COPY_LOCATION{pResource = (^d3d12.IResource)(readback), Type = .PLACED_FOOTPRINT}
		dst.PlacedFootprint = footprints[i]
		src := d3d12.TEXTURE_COPY_LOCATION{pResource = (^d3d12.IResource)(t), Type = .SUBRESOURCE_INDEX}
		copy_texture_region(l, &dst, &src)
	}
	submit(g)

	parity.compute_cpu(m, &cd)

	got := resource_map(readback)
	defer resource_unmap(readback)
	gpu_b := make([][]u8, rw_buf, context.temp_allocator)
	for &b, i in gpu_b {
		b = got[i * cd.buf_words * 4:][:cd.buf_words * 4]
	}
	gpu_i := make([][]u8, rw_tex, context.temp_allocator)
	for &im, i in gpu_i {
		ci := cd.cpu_imgs[ro_tex + i]
		row := ci.width * parity.image_texel_bytes(ci.format)
		fp := footprints[i]
		im = make([]u8, row * ci.height, context.temp_allocator)
		for y in 0 ..< ci.height {
			mem.copy(&im[y * row], &got[int(fp.Offset) + y * int(fp.Footprint.RowPitch)], row)
		}
	}
	return parity.compute_compare(m, &cd, gpu_b, gpu_i)
}
