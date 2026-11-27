#+build windows amd64
package d3d12_parity

// The few Direct3D 12 calls the harness makes, on Microsoft's runtime
// through vendor:directx/d3d12 and vendor:directx/dxgi: the same procs as
// d3d12.odin (vkd3d), on WARP by default or the first hardware adapter,
// with the debug layer and GPU-based validation on. debug_messages reads
// the debug layer's info queue after each check.

import "core:fmt"
import win32 "core:sys/windows"
import d3d12 "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"

// The device's info queue (debug_messages).
info_queue: ^d3d12.IInfoQueue

check :: proc(hr: dxgi.HRESULT, what: string, loc := #caller_location) {
	if hr < 0 {
		fmt.eprintfln("%s failed: hr %x (%v)", what, u32(hr), loc)
		if info_queue != nil {
			debug_messages()
		}
		panic("D3D12 call failed")
	}
}

release :: proc(obj: rawptr) {
	if obj != nil {
		u := (^dxgi.IUnknown)(obj)
		u->Release()
	}
}

// WARP, or the first adapter that is not a software one, with the debug
// layer and GPU-based validation enabled before the device; returns the
// adapter's description.
create_device :: proc(hardware: bool) -> (device: rawptr, name: string) {
	debug: ^d3d12.IDebug1
	check(d3d12.GetDebugInterface(d3d12.IDebug1_UUID, (^rawptr)(&debug)), "GetDebugInterface")
	debug->EnableDebugLayer()
	debug->SetEnableGPUBasedValidation(true)
	release(debug)
	factory: ^dxgi.IFactory4
	check(dxgi.CreateDXGIFactory2({.DEBUG}, dxgi.IFactory4_UUID, (^rawptr)(&factory)), "CreateDXGIFactory2")
	defer release(factory)
	adapter: ^dxgi.IAdapter1
	desc: dxgi.ADAPTER_DESC1
	if hardware {
		for i: u32 = 0; factory->EnumAdapters1(i, &adapter) >= 0; i += 1 {
			check(adapter->GetDesc1(&desc), "GetDesc1")
			// The Basic Render Driver (Microsoft's vendor ID, 0x1414) is
			// WARP, and may come without the SOFTWARE flag.
			if .SOFTWARE not_in desc.Flags && desc.VendorId != 0x1414 {
				break
			}
			release(adapter)
			adapter = nil
		}
		if adapter == nil {
			return nil, "no hardware adapter"
		}
	} else {
		check(factory->EnumWarpAdapter(dxgi.IAdapter1_UUID, (^rawptr)(&adapter)), "EnumWarpAdapter")
		check(adapter->GetDesc1(&desc), "GetDesc1")
	}
	defer release(adapter)
	name, _ = win32.wstring_to_utf8(win32.wstring(&desc.Description[0]), -1, context.allocator)
	if d3d12.CreateDevice((^dxgi.IUnknown)(adapter), ._11_0, d3d12.IDevice_UUID, &device) < 0 {
		return nil, name
	}
	dev := (^d3d12.IDevice)(device)
	check(dev->QueryInterface(d3d12.IInfoQueue_UUID, (^rawptr)(&info_queue)), "QueryInterface(ID3D12InfoQueue)")
	return device, name
}

// Prints the debug layer's messages stored since the last call (warnings
// and above), clears them and returns how many are errors or corruption.
debug_messages :: proc() -> (errors: int) {
	n := info_queue->GetNumStoredMessages()
	for i in 0 ..< n {
		size: uint
		info_queue->GetMessageA(i, nil, &size)
		msg := (^d3d12.MESSAGE)(raw_data(make([]u8, size, context.temp_allocator)))
		if info_queue->GetMessageA(i, msg, &size) < 0 || msg.Severity > .WARNING {
			continue
		}
		if msg.Severity < .WARNING {
			errors += 1
		}
		fmt.printfln("    d3d12 %v %d: %s", msg.Severity, int(msg.ID), msg.pDescription)
	}
	info_queue->ClearStoredMessages()
	return
}

D3D12SerializeRootSignature :: proc(desc: ^d3d12.ROOT_SIGNATURE_DESC, version: d3d12.ROOT_SIGNATURE_VERSION, blob: ^rawptr, error_blob: ^rawptr) -> dxgi.HRESULT {
	return d3d12.SerializeRootSignature(desc, version, (^^d3d12.IBlob)(blob), (^^d3d12.IBlob)(error_blob))
}

// ---- ID3D12Device -----------------------------------------------------------------------

create_command_queue :: proc(dev: rawptr, desc: ^d3d12.COMMAND_QUEUE_DESC) -> (out: rawptr) {
	d := (^d3d12.IDevice)(dev)
	check(d->CreateCommandQueue(desc, d3d12.ICommandQueue_UUID, &out), "CreateCommandQueue")
	return
}

create_command_allocator :: proc(dev: rawptr) -> (out: rawptr) {
	d := (^d3d12.IDevice)(dev)
	check(d->CreateCommandAllocator(.DIRECT, d3d12.ICommandAllocator_UUID, &out), "CreateCommandAllocator")
	return
}

create_graphics_pipeline_state :: proc(dev: rawptr, desc: ^d3d12.GRAPHICS_PIPELINE_STATE_DESC) -> (out: rawptr, hr: dxgi.HRESULT) {
	d := (^d3d12.IDevice)(dev)
	hr = d->CreateGraphicsPipelineState(desc, d3d12.IPipelineState_UUID, &out)
	return
}

create_compute_pipeline_state :: proc(dev: rawptr, desc: ^d3d12.COMPUTE_PIPELINE_STATE_DESC) -> (out: rawptr, hr: dxgi.HRESULT) {
	d := (^d3d12.IDevice)(dev)
	hr = d->CreateComputePipelineState(desc, d3d12.IPipelineState_UUID, &out)
	return
}

create_command_list :: proc(dev: rawptr, allocator: rawptr) -> (out: rawptr) {
	d := (^d3d12.IDevice)(dev)
	check(d->CreateCommandList(0, .DIRECT, (^d3d12.ICommandAllocator)(allocator), nil, d3d12.IGraphicsCommandList_UUID, &out), "CreateCommandList")
	return
}

create_descriptor_heap :: proc(dev: rawptr, desc: ^d3d12.DESCRIPTOR_HEAP_DESC) -> (out: rawptr) {
	d := (^d3d12.IDevice)(dev)
	check(d->CreateDescriptorHeap(desc, d3d12.IDescriptorHeap_UUID, &out), "CreateDescriptorHeap")
	return
}

descriptor_increment :: proc(dev: rawptr, type: d3d12.DESCRIPTOR_HEAP_TYPE) -> u32 {
	d := (^d3d12.IDevice)(dev)
	return d->GetDescriptorHandleIncrementSize(type)
}

create_root_signature :: proc(dev: rawptr, data: rawptr, size: uint) -> (out: rawptr, hr: dxgi.HRESULT) {
	d := (^d3d12.IDevice)(dev)
	hr = d->CreateRootSignature(0, data, size, d3d12.IRootSignature_UUID, &out)
	return
}

create_shader_resource_view :: proc(dev: rawptr, res: rawptr, desc: ^d3d12.SHADER_RESOURCE_VIEW_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	d := (^d3d12.IDevice)(dev)
	d->CreateShaderResourceView((^d3d12.IResource)(res), desc, handle)
}

create_unordered_access_view :: proc(dev: rawptr, res: rawptr, desc: ^d3d12.UNORDERED_ACCESS_VIEW_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	d := (^d3d12.IDevice)(dev)
	d->CreateUnorderedAccessView((^d3d12.IResource)(res), nil, desc, handle)
}

create_render_target_view :: proc(dev: rawptr, res: rawptr, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	d := (^d3d12.IDevice)(dev)
	d->CreateRenderTargetView((^d3d12.IResource)(res), nil, handle)
}

create_sampler :: proc(dev: rawptr, desc: ^d3d12.SAMPLER_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	d := (^d3d12.IDevice)(dev)
	d->CreateSampler(desc, handle)
}

create_committed_resource :: proc(dev: rawptr, heap: d3d12.HEAP_TYPE, desc: ^d3d12.RESOURCE_DESC, state: d3d12.RESOURCE_STATES, clear: ^d3d12.CLEAR_VALUE = nil) -> (out: rawptr) {
	d := (^d3d12.IDevice)(dev)
	props := d3d12.HEAP_PROPERTIES{Type = heap}
	check(d->CreateCommittedResource(&props, {}, desc, state, clear, d3d12.IResource_UUID, &out), "CreateCommittedResource")
	return
}

create_fence :: proc(dev: rawptr) -> (out: rawptr) {
	d := (^d3d12.IDevice)(dev)
	check(d->CreateFence(0, {}, d3d12.IFence_UUID, &out), "CreateFence")
	return
}

get_copyable_footprints :: proc(dev: rawptr, desc: ^d3d12.RESOURCE_DESC, first, count: u32, layouts: [^]d3d12.PLACED_SUBRESOURCE_FOOTPRINT) -> (total: u64) {
	d := (^d3d12.IDevice)(dev)
	d->GetCopyableFootprints(desc, first, count, 0, layouts, nil, nil, &total)
	return
}

// ---- queue, allocator, fence, blob, heap, resource ------------------------------------

execute :: proc(queue: rawptr, list: rawptr) {
	q := (^d3d12.ICommandQueue)(queue)
	l := (^d3d12.ICommandList)(list)
	q->ExecuteCommandLists(1, &l)
}

queue_signal :: proc(queue: rawptr, fence: rawptr, value: u64) {
	q := (^d3d12.ICommandQueue)(queue)
	check(q->Signal((^d3d12.IFence)(fence), value), "Signal")
}

allocator_reset :: proc(a: rawptr) {
	ca := (^d3d12.ICommandAllocator)(a)
	check(ca->Reset(), "CommandAllocator::Reset")
}

// Blocks until the fence reaches `value` (a nil event waits).
fence_wait :: proc(fence: rawptr, value: u64) {
	f := (^d3d12.IFence)(fence)
	check(f->SetEventOnCompletion(value, nil), "SetEventOnCompletion")
}

blob_data :: proc(blob: rawptr) -> []u8 {
	b := (^d3d12.IBlob)(blob)
	return ([^]u8)(b->GetBufferPointer())[:b->GetBufferSize()]
}

heap_cpu_start :: proc(heap: rawptr) -> (h: d3d12.CPU_DESCRIPTOR_HANDLE) {
	dh := (^d3d12.IDescriptorHeap)(heap)
	dh->GetCPUDescriptorHandleForHeapStart(&h)
	return
}

heap_gpu_start :: proc(heap: rawptr) -> (h: d3d12.GPU_DESCRIPTOR_HANDLE) {
	dh := (^d3d12.IDescriptorHeap)(heap)
	dh->GetGPUDescriptorHandleForHeapStart(&h)
	return
}

resource_map :: proc(res: rawptr) -> [^]u8 {
	r := (^d3d12.IResource)(res)
	p: rawptr
	check(r->Map(0, nil, &p), "Map")
	return ([^]u8)(p)
}

resource_unmap :: proc(res: rawptr) {
	r := (^d3d12.IResource)(res)
	r->Unmap(0, nil)
}

gpu_address :: proc(res: rawptr) -> u64 {
	r := (^d3d12.IResource)(res)
	return r->GetGPUVirtualAddress()
}

// ---- ID3D12GraphicsCommandList ----------------------------------------------------------

list_close :: proc(l: rawptr) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	check(gl->Close(), "Close")
}

list_reset :: proc(l: rawptr, allocator: rawptr) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	check(gl->Reset((^d3d12.ICommandAllocator)(allocator), nil), "CommandList::Reset")
}

draw_instanced :: proc(l: rawptr, vertices, first: u32, instances: u32 = 1) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->DrawInstanced(vertices, instances, first, 0)
}

dispatch :: proc(l: rawptr, x, y, z: u32) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->Dispatch(x, y, z)
}

copy_buffer_region :: proc(l: rawptr, dst: rawptr, dst_offset: u64, src: rawptr, src_offset: u64, size: u64) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->CopyBufferRegion((^d3d12.IResource)(dst), dst_offset, (^d3d12.IResource)(src), src_offset, size)
}

copy_texture_region :: proc(l: rawptr, dst: ^d3d12.TEXTURE_COPY_LOCATION, src: ^d3d12.TEXTURE_COPY_LOCATION) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->CopyTextureRegion(dst, 0, 0, 0, src, nil)
}

resolve_subresource :: proc(l: rawptr, dst: rawptr, src: rawptr, format: dxgi.FORMAT) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->ResolveSubresource((^d3d12.IResource)(dst), 0, (^d3d12.IResource)(src), 0, format)
}

set_topology :: proc(l: rawptr, topology: d3d12.PRIMITIVE_TOPOLOGY) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->IASetPrimitiveTopology(topology)
}

set_viewport_scissor :: proc(l: rawptr, w, h: u32) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	vp := d3d12.VIEWPORT{0, 0, f32(w), f32(h), 0, 1}
	rect := d3d12.RECT{0, 0, i32(w), i32(h)}
	gl->RSSetViewports(1, &vp)
	gl->RSSetScissorRects(1, &rect)
}

set_pipeline_state :: proc(l: rawptr, pso: rawptr) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetPipelineState((^d3d12.IPipelineState)(pso))
}

transition :: proc(l: rawptr, res: rawptr, before, after: d3d12.RESOURCE_STATES) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	b := d3d12.RESOURCE_BARRIER{Type = .TRANSITION}
	b.Transition = {pResource = (^d3d12.IResource)(res), Subresource = 0xffffffff, StateBefore = before, StateAfter = after}
	gl->ResourceBarrier(1, &b)
}

set_descriptor_heaps :: proc(l: rawptr, heaps: []rawptr) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetDescriptorHeaps(u32(len(heaps)), ([^]^d3d12.IDescriptorHeap)(raw_data(heaps)))
}

set_compute_root_signature :: proc(l: rawptr, rs: rawptr) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetComputeRootSignature((^d3d12.IRootSignature)(rs))
}

set_compute_root_table :: proc(l: rawptr, index: u32, h: d3d12.GPU_DESCRIPTOR_HANDLE) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetComputeRootDescriptorTable(index, h)
}

set_compute_root_cbv :: proc(l: rawptr, index: u32, address: u64) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetComputeRootConstantBufferView(index, address)
}

set_graphics_root_signature :: proc(l: rawptr, rs: rawptr) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetGraphicsRootSignature((^d3d12.IRootSignature)(rs))
}

set_graphics_root_table :: proc(l: rawptr, index: u32, h: d3d12.GPU_DESCRIPTOR_HANDLE) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetGraphicsRootDescriptorTable(index, h)
}

set_graphics_root_cbv :: proc(l: rawptr, index: u32, address: u64) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->SetGraphicsRootConstantBufferView(index, address)
}

set_vertex_buffer :: proc(l: rawptr, view: ^d3d12.VERTEX_BUFFER_VIEW) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->IASetVertexBuffers(0, 1, view)
}

set_render_target :: proc(l: rawptr, h: ^d3d12.CPU_DESCRIPTOR_HANDLE) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	gl->OMSetRenderTargets(1, h, false, nil)
}

clear_render_target :: proc(l: rawptr, h: d3d12.CPU_DESCRIPTOR_HANDLE, color: [4]f32) {
	gl := (^d3d12.IGraphicsCommandList)(l)
	c := color
	gl->ClearRenderTargetView(h, &c, 0, nil)
}
