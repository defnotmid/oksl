#+build linux amd64
package d3d12_parity

// The few Direct3D 12 calls the harness makes, on vkd3d (Wine's D3D12 on
// Vulkan) through libvkd3d-utils. Structs and constants are
// vendor:directx/d3d12's (the same layout on Linux); its methods are
// declared "system", which is SysV here, while vkd3d's COM methods and
// entry points are ms_abi (vkd3d_windows.h: __stdcall is
// __attribute__((ms_abi)) on x86-64), so the calls go through the vtables
// below with the "win64" convention. Indices are the C vtables' in
// vkd3d_d3d12.h.

import "core:fmt"
import d3d12 "vendor:directx/d3d12"
import dxgi "vendor:directx/dxgi"

foreign import vkd3d_utils "system:vkd3d-utils"

@(default_calling_convention = "win64")
foreign vkd3d_utils {
	D3D12CreateDeviceVKD3D :: proc(adapter: rawptr, level: d3d12.FEATURE_LEVEL, iid: ^d3d12.IID, device: ^rawptr, api_version: u32) -> i32 ---
	D3D12SerializeRootSignature :: proc(desc: ^d3d12.ROOT_SIGNATURE_DESC, version: d3d12.ROOT_SIGNATURE_VERSION, blob: ^rawptr, error_blob: ^rawptr) -> i32 ---
}

// vkd3d's device on the first Vulkan device (`hardware` has no meaning
// here).
create_device :: proc(hardware: bool) -> (device: rawptr, name: string) {
	iid := d3d12.IDevice_UUID^
	if D3D12CreateDeviceVKD3D(nil, ._11_0, &iid, &device, 0) < 0 {
		return nil, "vkd3d"
	}
	return device, "vkd3d"
}

// vkd3d has no debug layer: no messages.
debug_messages :: proc() -> (errors: int) {
	return 0
}

vtbl :: #force_inline proc "contextless" (obj: rawptr) -> [^]rawptr {
	return (^[^]rawptr)(obj)^
}

check :: proc(hr: i32, what: string, loc := #caller_location) {
	if hr < 0 {
		fmt.eprintfln("%s failed: hr %x (%v)", what, u32(hr), loc)
		panic("D3D12 call failed")
	}
}

release :: proc(obj: rawptr) {
	if obj != nil {
		F :: proc "win64" (this: rawptr) -> u32
		(F)(vtbl(obj)[2])(obj)
	}
}

// ---- ID3D12Device -----------------------------------------------------------------------

create_command_queue :: proc(dev: rawptr, desc: ^d3d12.COMMAND_QUEUE_DESC) -> (out: rawptr) {
	F :: proc "win64" (this: rawptr, desc: ^d3d12.COMMAND_QUEUE_DESC, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.ICommandQueue_UUID^
	check((F)(vtbl(dev)[8])(dev, desc, &iid, &out), "CreateCommandQueue")
	return
}

create_command_allocator :: proc(dev: rawptr) -> (out: rawptr) {
	F :: proc "win64" (this: rawptr, type: d3d12.COMMAND_LIST_TYPE, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.ICommandAllocator_UUID^
	check((F)(vtbl(dev)[9])(dev, .DIRECT, &iid, &out), "CreateCommandAllocator")
	return
}

create_graphics_pipeline_state :: proc(dev: rawptr, desc: ^d3d12.GRAPHICS_PIPELINE_STATE_DESC) -> (out: rawptr, hr: i32) {
	F :: proc "win64" (this: rawptr, desc: ^d3d12.GRAPHICS_PIPELINE_STATE_DESC, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.IPipelineState_UUID^
	hr = (F)(vtbl(dev)[10])(dev, desc, &iid, &out)
	return
}

create_compute_pipeline_state :: proc(dev: rawptr, desc: ^d3d12.COMPUTE_PIPELINE_STATE_DESC) -> (out: rawptr, hr: i32) {
	F :: proc "win64" (this: rawptr, desc: ^d3d12.COMPUTE_PIPELINE_STATE_DESC, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.IPipelineState_UUID^
	hr = (F)(vtbl(dev)[11])(dev, desc, &iid, &out)
	return
}

create_command_list :: proc(dev: rawptr, allocator: rawptr) -> (out: rawptr) {
	F :: proc "win64" (this: rawptr, node_mask: u32, type: d3d12.COMMAND_LIST_TYPE, allocator: rawptr, pso: rawptr, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.IGraphicsCommandList_UUID^
	check((F)(vtbl(dev)[12])(dev, 0, .DIRECT, allocator, nil, &iid, &out), "CreateCommandList")
	return
}

create_descriptor_heap :: proc(dev: rawptr, desc: ^d3d12.DESCRIPTOR_HEAP_DESC) -> (out: rawptr) {
	F :: proc "win64" (this: rawptr, desc: ^d3d12.DESCRIPTOR_HEAP_DESC, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.IDescriptorHeap_UUID^
	check((F)(vtbl(dev)[14])(dev, desc, &iid, &out), "CreateDescriptorHeap")
	return
}

descriptor_increment :: proc(dev: rawptr, type: d3d12.DESCRIPTOR_HEAP_TYPE) -> u32 {
	F :: proc "win64" (this: rawptr, type: d3d12.DESCRIPTOR_HEAP_TYPE) -> u32
	return (F)(vtbl(dev)[15])(dev, type)
}

create_root_signature :: proc(dev: rawptr, data: rawptr, size: uint) -> (out: rawptr, hr: i32) {
	F :: proc "win64" (this: rawptr, node_mask: u32, data: rawptr, size: uint, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.IRootSignature_UUID^
	hr = (F)(vtbl(dev)[16])(dev, 0, data, size, &iid, &out)
	return
}

create_shader_resource_view :: proc(dev: rawptr, res: rawptr, desc: ^d3d12.SHADER_RESOURCE_VIEW_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, res: rawptr, desc: ^d3d12.SHADER_RESOURCE_VIEW_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE)
	(F)(vtbl(dev)[18])(dev, res, desc, handle)
}

create_unordered_access_view :: proc(dev: rawptr, res: rawptr, desc: ^d3d12.UNORDERED_ACCESS_VIEW_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, res: rawptr, counter: rawptr, desc: ^d3d12.UNORDERED_ACCESS_VIEW_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE)
	(F)(vtbl(dev)[19])(dev, res, nil, desc, handle)
}

create_render_target_view :: proc(dev: rawptr, res: rawptr, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, res: rawptr, desc: rawptr, handle: d3d12.CPU_DESCRIPTOR_HANDLE)
	(F)(vtbl(dev)[20])(dev, res, nil, handle)
}

create_sampler :: proc(dev: rawptr, desc: ^d3d12.SAMPLER_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, desc: ^d3d12.SAMPLER_DESC, handle: d3d12.CPU_DESCRIPTOR_HANDLE)
	(F)(vtbl(dev)[22])(dev, desc, handle)
}

create_committed_resource :: proc(dev: rawptr, heap: d3d12.HEAP_TYPE, desc: ^d3d12.RESOURCE_DESC, state: d3d12.RESOURCE_STATES, clear: ^d3d12.CLEAR_VALUE = nil) -> (out: rawptr) {
	F :: proc "win64" (this: rawptr, props: ^d3d12.HEAP_PROPERTIES, flags: d3d12.HEAP_FLAGS, desc: ^d3d12.RESOURCE_DESC, state: d3d12.RESOURCE_STATES, clear: ^d3d12.CLEAR_VALUE, iid: ^d3d12.IID, out: ^rawptr) -> i32
	props := d3d12.HEAP_PROPERTIES{Type = heap}
	iid := d3d12.IResource_UUID^
	check((F)(vtbl(dev)[27])(dev, &props, {}, desc, state, clear, &iid, &out), "CreateCommittedResource")
	return
}

create_fence :: proc(dev: rawptr) -> (out: rawptr) {
	F :: proc "win64" (this: rawptr, initial: u64, flags: d3d12.FENCE_FLAGS, iid: ^d3d12.IID, out: ^rawptr) -> i32
	iid := d3d12.IFence_UUID^
	check((F)(vtbl(dev)[36])(dev, 0, {}, &iid, &out), "CreateFence")
	return
}

get_copyable_footprints :: proc(dev: rawptr, desc: ^d3d12.RESOURCE_DESC, first, count: u32, layouts: [^]d3d12.PLACED_SUBRESOURCE_FOOTPRINT) -> (total: u64) {
	F :: proc "win64" (this: rawptr, desc: ^d3d12.RESOURCE_DESC, first, count: u32, base: u64, layouts: [^]d3d12.PLACED_SUBRESOURCE_FOOTPRINT, rows: rawptr, row_sizes: rawptr, total: ^u64)
	(F)(vtbl(dev)[38])(dev, desc, first, count, 0, layouts, nil, nil, &total)
	return
}

// ---- queue, allocator, fence, blob, heap, resource ------------------------------------

execute :: proc(queue: rawptr, list: rawptr) {
	F :: proc "win64" (this: rawptr, count: u32, lists: ^rawptr)
	l := list
	(F)(vtbl(queue)[10])(queue, 1, &l)
}

queue_signal :: proc(queue: rawptr, fence: rawptr, value: u64) {
	F :: proc "win64" (this: rawptr, fence: rawptr, value: u64) -> i32
	check((F)(vtbl(queue)[14])(queue, fence, value), "Signal")
}

allocator_reset :: proc(a: rawptr) {
	F :: proc "win64" (this: rawptr) -> i32
	check((F)(vtbl(a)[8])(a), "CommandAllocator::Reset")
}

// Blocks until the fence reaches `value` (vkd3d waits on a nil event).
fence_wait :: proc(fence: rawptr, value: u64) {
	F :: proc "win64" (this: rawptr, value: u64, event: rawptr) -> i32
	check((F)(vtbl(fence)[9])(fence, value, nil), "SetEventOnCompletion")
}

blob_data :: proc(blob: rawptr) -> []u8 {
	P :: proc "win64" (this: rawptr) -> rawptr
	S :: proc "win64" (this: rawptr) -> uint
	return ([^]u8)((P)(vtbl(blob)[3])(blob))[:(S)(vtbl(blob)[4])(blob)]
}

heap_cpu_start :: proc(heap: rawptr) -> (h: d3d12.CPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, ret: ^d3d12.CPU_DESCRIPTOR_HANDLE) -> ^d3d12.CPU_DESCRIPTOR_HANDLE
	(F)(vtbl(heap)[9])(heap, &h)
	return
}

heap_gpu_start :: proc(heap: rawptr) -> (h: d3d12.GPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, ret: ^d3d12.GPU_DESCRIPTOR_HANDLE) -> ^d3d12.GPU_DESCRIPTOR_HANDLE
	(F)(vtbl(heap)[10])(heap, &h)
	return
}

resource_map :: proc(res: rawptr) -> [^]u8 {
	F :: proc "win64" (this: rawptr, sub: u32, range: rawptr, data: ^rawptr) -> i32
	p: rawptr
	check((F)(vtbl(res)[8])(res, 0, nil, &p), "Map")
	return ([^]u8)(p)
}

resource_unmap :: proc(res: rawptr) {
	F :: proc "win64" (this: rawptr, sub: u32, range: rawptr)
	(F)(vtbl(res)[9])(res, 0, nil)
}

gpu_address :: proc(res: rawptr) -> u64 {
	F :: proc "win64" (this: rawptr) -> u64
	return (F)(vtbl(res)[11])(res)
}

// ---- ID3D12GraphicsCommandList ----------------------------------------------------------

list_close :: proc(l: rawptr) {
	F :: proc "win64" (this: rawptr) -> i32
	check((F)(vtbl(l)[9])(l), "Close")
}

list_reset :: proc(l: rawptr, allocator: rawptr) {
	F :: proc "win64" (this: rawptr, allocator: rawptr, pso: rawptr) -> i32
	check((F)(vtbl(l)[10])(l, allocator, nil), "CommandList::Reset")
}

draw_instanced :: proc(l: rawptr, vertices, first: u32) {
	F :: proc "win64" (this: rawptr, vertices, instances, first_vertex, first_instance: u32)
	(F)(vtbl(l)[12])(l, vertices, 1, first, 0)
}

dispatch :: proc(l: rawptr, x, y, z: u32) {
	F :: proc "win64" (this: rawptr, x, y, z: u32)
	(F)(vtbl(l)[14])(l, x, y, z)
}

copy_buffer_region :: proc(l: rawptr, dst: rawptr, dst_offset: u64, src: rawptr, src_offset: u64, size: u64) {
	F :: proc "win64" (this: rawptr, dst: rawptr, dst_offset: u64, src: rawptr, src_offset: u64, size: u64)
	(F)(vtbl(l)[15])(l, dst, dst_offset, src, src_offset, size)
}

copy_texture_region :: proc(l: rawptr, dst: ^d3d12.TEXTURE_COPY_LOCATION, src: ^d3d12.TEXTURE_COPY_LOCATION) {
	F :: proc "win64" (this: rawptr, dst: ^d3d12.TEXTURE_COPY_LOCATION, x, y, z: u32, src: ^d3d12.TEXTURE_COPY_LOCATION, box: rawptr)
	(F)(vtbl(l)[16])(l, dst, 0, 0, 0, src, nil)
}

resolve_subresource :: proc(l: rawptr, dst: rawptr, src: rawptr, format: dxgi.FORMAT) {
	F :: proc "win64" (this: rawptr, dst: rawptr, dst_sub: u32, src: rawptr, src_sub: u32, format: dxgi.FORMAT)
	(F)(vtbl(l)[19])(l, dst, 0, src, 0, format)
}

set_topology :: proc(l: rawptr, topology: d3d12.PRIMITIVE_TOPOLOGY) {
	F :: proc "win64" (this: rawptr, topology: d3d12.PRIMITIVE_TOPOLOGY)
	(F)(vtbl(l)[20])(l, topology)
}

set_viewport_scissor :: proc(l: rawptr, w, h: u32) {
	V :: proc "win64" (this: rawptr, count: u32, vp: ^d3d12.VIEWPORT)
	R :: proc "win64" (this: rawptr, count: u32, rect: ^d3d12.RECT)
	vp := d3d12.VIEWPORT{0, 0, f32(w), f32(h), 0, 1}
	rect := d3d12.RECT{0, 0, i32(w), i32(h)}
	(V)(vtbl(l)[21])(l, 1, &vp)
	(R)(vtbl(l)[22])(l, 1, &rect)
}

set_pipeline_state :: proc(l: rawptr, pso: rawptr) {
	F :: proc "win64" (this: rawptr, pso: rawptr)
	(F)(vtbl(l)[25])(l, pso)
}

transition :: proc(l: rawptr, res: rawptr, before, after: d3d12.RESOURCE_STATES) {
	F :: proc "win64" (this: rawptr, count: u32, barriers: ^d3d12.RESOURCE_BARRIER)
	b := d3d12.RESOURCE_BARRIER{Type = .TRANSITION}
	b.Transition = {pResource = (^d3d12.IResource)(res), Subresource = 0xffffffff, StateBefore = before, StateAfter = after}
	(F)(vtbl(l)[26])(l, 1, &b)
}

set_descriptor_heaps :: proc(l: rawptr, heaps: []rawptr) {
	F :: proc "win64" (this: rawptr, count: u32, heaps: [^]rawptr)
	(F)(vtbl(l)[28])(l, u32(len(heaps)), raw_data(heaps))
}

set_compute_root_signature :: proc(l: rawptr, rs: rawptr) {
	F :: proc "win64" (this: rawptr, rs: rawptr)
	(F)(vtbl(l)[29])(l, rs)
}

set_compute_root_table :: proc(l: rawptr, index: u32, h: d3d12.GPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, index: u32, h: d3d12.GPU_DESCRIPTOR_HANDLE)
	(F)(vtbl(l)[31])(l, index, h)
}

set_compute_root_cbv :: proc(l: rawptr, index: u32, address: u64) {
	F :: proc "win64" (this: rawptr, index: u32, address: u64)
	(F)(vtbl(l)[37])(l, index, address)
}

set_graphics_root_signature :: proc(l: rawptr, rs: rawptr) {
	F :: proc "win64" (this: rawptr, rs: rawptr)
	(F)(vtbl(l)[30])(l, rs)
}

set_graphics_root_table :: proc(l: rawptr, index: u32, h: d3d12.GPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, index: u32, h: d3d12.GPU_DESCRIPTOR_HANDLE)
	(F)(vtbl(l)[32])(l, index, h)
}

set_graphics_root_cbv :: proc(l: rawptr, index: u32, address: u64) {
	F :: proc "win64" (this: rawptr, index: u32, address: u64)
	(F)(vtbl(l)[38])(l, index, address)
}

set_vertex_buffer :: proc(l: rawptr, view: ^d3d12.VERTEX_BUFFER_VIEW) {
	F :: proc "win64" (this: rawptr, start, count: u32, views: ^d3d12.VERTEX_BUFFER_VIEW)
	(F)(vtbl(l)[44])(l, 0, 1, view)
}

set_render_target :: proc(l: rawptr, h: ^d3d12.CPU_DESCRIPTOR_HANDLE) {
	F :: proc "win64" (this: rawptr, count: u32, handles: ^d3d12.CPU_DESCRIPTOR_HANDLE, single: i32, dsv: rawptr)
	(F)(vtbl(l)[46])(l, 1, h, 0, nil)
}

clear_render_target :: proc(l: rawptr, h: d3d12.CPU_DESCRIPTOR_HANDLE, color: [4]f32) {
	F :: proc "win64" (this: rawptr, h: d3d12.CPU_DESCRIPTOR_HANDLE, color: ^[4]f32, count: u32, rects: rawptr)
	c := color
	(F)(vtbl(l)[48])(l, h, &c, 0, nil)
}
