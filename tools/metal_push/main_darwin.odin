#+build darwin
// Compiles oksl's Layout.Oplat MSL 3.0 shaders with Metal's runtime compiler
// (checking every static_assert(sizeof / alignof / __builtin_offsetof)) and
// draws one full-screen triangle coloured from setVertexBytes / setFragmentBytes
// at buffer 0, comparing the pixels against oksl's CPU VM.
//
//     odin run tools/metal_push
package metal_push

import "core:fmt"
import "core:os"
import "core:slice"
import "core:strings"
import "core:time"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import oksl "../.."

Push_Draw :: struct {
	view_proj: matrix[4, 4]f32,
	tint:      [4]f32,
	count:     u32,
}

#assert(size_of(Push_Draw) == 96)
#assert(align_of(Push_Draw) == 16)
#assert(offset_of(Push_Draw, view_proj) == 0)
#assert(offset_of(Push_Draw, tint) == 64)
#assert(offset_of(Push_Draw, count) == 80)

VS_SRC :: `#version 300 es
push Push_Draw {
	mat4 view_proj;
	vec4 tint;
	uint count;
} draw;
layout(location = 0) out vec4 v_color;
void main() {
	vec2 pos = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
	gl_Position = draw.view_proj * vec4(pos, 0.0, 1.0);
	v_color = draw.tint * float(draw.count) * 0.25;
}
`

FS_SRC :: `#version 300 es
precision highp float;
push Push_Draw {
	mat4 view_proj;
	vec4 tint;
	uint count;
} draw;
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 out_color;
void main() {
	out_color = v_color;
}
`

All_Types_Inner :: struct {
	a: [2]f32,
	b: f32,
}

All_Types_Push :: struct {
	v3:  [3]f32,
	m2:  matrix[2, 2]f32,
	iv:  [2]i32,
	uv:  [3]u32,
	v4:  [4]f32,
	m3:  matrix[3, 3]f32,
	arr: [2]All_Types_Inner,
}

#assert(size_of(All_Types_Inner) == 12)
#assert(align_of(All_Types_Inner) == 4)
#assert(offset_of(All_Types_Inner, a) == 0)
#assert(offset_of(All_Types_Inner, b) == 8)
#assert(size_of(All_Types_Push) == 128)
#assert(align_of(All_Types_Push) == 8)
#assert(offset_of(All_Types_Push, v3) == 0)
#assert(offset_of(All_Types_Push, m2) == 16)
#assert(offset_of(All_Types_Push, iv) == 32)
#assert(offset_of(All_Types_Push, uv) == 40)
#assert(offset_of(All_Types_Push, v4) == 52)
#assert(offset_of(All_Types_Push, m3) == 68)
#assert(offset_of(All_Types_Push, arr) == 104)

ALL_TYPES_FS :: `#version 300 es
precision highp float;
struct All_Types_Inner {
	vec2 a;
	float b;
};
push All_Types_Push {
	vec3 v3;
	mat2 m2;
	ivec2 iv;
	uvec3 uv;
	vec4 v4;
	mat3 m3;
	All_Types_Inner arr[2];
} p;
layout(location = 0) out vec4 out_color;
void main() {
	int idx = int(gl_FragCoord.x) & 1;
	int band = int(gl_FragCoord.y) & 3;
	All_Types_Inner item = p.arr[idx];
	if (band == 0) {
		out_color = vec4(p.m3 * vec3(1.0, 0.0, 0.0) + vec3(item.a.x * 0.25, 0.0, 0.0), 1.0);
	} else if (band == 1) {
		out_color = vec4(p.m3 * vec3(0.0, 1.0, 0.0) + vec3(0.0, item.a.y * 0.25, 0.0), 1.0);
	} else if (band == 2) {
		out_color = vec4(p.m3 * vec3(0.0, 0.0, 1.0) + vec3(0.0, 0.0, item.b * 0.25), 1.0);
	} else {
		vec2 q = p.m2 * p.v3.xy;
		float f0 = clamp(q.x + float(p.iv.x) * 0.05 + p.v4.w * 0.2, 0.0, 1.0);
		float f1 = clamp(q.y + float(p.iv.y) * 0.05 + p.v3.z * 0.2 + float(p.uv.x) * 0.05, 0.0, 1.0);
		float f2 = clamp(p.v4.x + p.v4.y * 0.25 + float(p.uv.y + p.uv.z) * 0.04 + item.b * 0.2, 0.0, 1.0);
		out_color = vec4(f0, f1, f2, 1.0);
	}
}
`

Push_Discard :: struct {
	iters: i32,
	seed:  f32,
}

#assert(size_of(Push_Discard) == 8)
#assert(align_of(Push_Discard) == 4)
#assert(offset_of(Push_Discard, iters) == 0)
#assert(offset_of(Push_Discard, seed) == 4)

DISCARD_TERM_FS :: `#version 300 es
precision highp float;
push Push_Discard {
	int iters;
	float seed;
} u;
layout(location = 0) out vec4 out_color;
void maybe_discard(int i) {
	if (i == 0 && u.seed > -10.0 && gl_FragCoord.x >= 0.0) {
		discard;
	}
}
void main() {
	float acc = u.seed;
	for (int i = 0; i < u.iters; i++) {
		maybe_discard(i);
		acc = fract(acc * 1.000001 + float(i) * 1e-7 + u.seed);
	}
	out_color = vec4(acc, 1.0 - acc, 0.0, 1.0);
}
`

compile_msl_lib :: proc(device: ^MTL.Device, label: string, msl_src: string) -> (lib: ^MTL.Library, ok: bool) {
	opts := MTL.new(MTL.CompileOptions)
	defer opts->release()
	opts->setLanguageVersion(._3_0)
	src_ns := NS.to_ns_string(msl_src)
	defer src_ns->release()
	err: ^NS.Error
	lib, err = device->newLibraryWithSource(src_ns, opts)
	if lib == nil {
		msg := NS.to_odin_string(err->localizedDescription(), context.temp_allocator) if err != nil else "unknown Metal error"
		fmt.eprintfln("%s: Metal 3.0 compilation failed: %s", label, msg)
		return nil, false
	}
	return lib, true
}

render_and_check_triangle :: proc(device: ^MTL.Device, queue: ^MTL.CommandQueue, p: ^oksl.Pipeline, push_bytes: []u8) -> bool {
	vlib, vok := compile_msl_lib(device, "triangle.vert", p.vertex.msl)
	if !vok {
		return false
	}
	defer vlib->release()
	flib, fok := compile_msl_lib(device, "triangle.frag", p.fragment.msl)
	if !fok {
		return false
	}
	defer flib->release()

	entry_ns := NS.to_ns_string(oksl.MSL_ENTRY)
	defer entry_ns->release()
	vfn := vlib->newFunctionWithName(entry_ns)
	ffn := flib->newFunctionWithName(entry_ns)
	if vfn == nil || ffn == nil {
		fmt.eprintln("triangle: missing oksl_main in Metal library")
		return false
	}
	defer vfn->release()
	defer ffn->release()

	W :: 16
	H :: 16
	td := MTL.TextureDescriptor.texture2DDescriptorWithPixelFormat(.RGBA8Unorm, W, H, false)
	td->setUsage({.RenderTarget})
	td->setStorageMode(.Shared)
	tex := device->newTextureWithDescriptor(td)
	if tex == nil {
		fmt.eprintln("triangle: failed to create render target texture")
		return false
	}
	defer tex->release()

	rpd := MTL.new(MTL.RenderPipelineDescriptor)
	defer rpd->release()
	rpd->setVertexFunction(vfn)
	rpd->setFragmentFunction(ffn)
	rpd->colorAttachments()->object(0)->setPixelFormat(.RGBA8Unorm)
	pso, perr := device->newRenderPipelineStateWithDescriptor(rpd)
	if pso == nil {
		msg := NS.to_odin_string(perr->localizedDescription(), context.temp_allocator) if perr != nil else "unknown PSO error"
		fmt.eprintfln("triangle: PSO failed: %s", msg)
		return false
	}
	defer pso->release()

	pass := MTL.RenderPassDescriptor.renderPassDescriptor()
	ca := pass->colorAttachments()->object(0)
	ca->setTexture(tex)
	ca->setLoadAction(.Clear)
	ca->setStoreAction(.Store)
	ca->setClearColor(MTL.ClearColor{0, 0, 0, 1})

	cb := queue->commandBuffer()
	enc := cb->renderCommandEncoderWithDescriptor(pass)
	enc->setRenderPipelineState(pso)
	enc->setVertexBytes(push_bytes, 0)
	enc->setFragmentBytes(push_bytes, 0)
	enc->drawPrimitives(.Triangle, 0, 3)
	enc->endEncoding()
	cb->commit()
	cb->waitUntilCompleted()

	gpu_px := make([]u8, W * H * 4)
	defer delete(gpu_px)
	tex->getBytes(raw_data(gpu_px), W * 4, MTL.Region{origin = {0, 0, 0}, size = {W, H, 1}}, 0)

	// Reference on CPU VM.
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	pos := make([][4]f32, 3)
	defer delete(pos)
	varys := make([]f32, 3 * p.vertex.varying_floats)
	defer delete(varys)
	oksl.run_vertex(&p.vertex.program, push_bytes, nil, 3, pos, varys, &scratch)

	// Constant tint across full-screen triangle: check all pixels against vertex 0's varying.
	v0 := [4]f32{varys[0], varys[1], varys[2], varys[3]}
	fvar := []oksl.Varying{{a = v0}}
	cpu_px := make([]u8, W * H * 4)
	defer delete(cpu_px)
	oksl.run(&p.fragment.program, push_bytes, W, H, cpu_px, W * 4, &scratch, {varyings = fvar})

	max_diff := 0
	for i in 0 ..< len(cpu_px) {
		d := abs(int(gpu_px[i]) - int(cpu_px[i]))
		max_diff = max(max_diff, d)
	}
	fmt.printfln("triangle parity: max channel diff = %d (expected <= 1)", max_diff)
	return max_diff <= 1
}

render_and_check_all_types :: proc(device: ^MTL.Device, queue: ^MTL.CommandQueue, m: ^oksl.Module, push_bytes: []u8) -> bool {
	lib, ok := compile_msl_lib(device, "all_types.frag", m.msl)
	if !ok {
		return false
	}
	defer lib->release()
	entry_ns := NS.to_ns_string(oksl.MSL_ENTRY)
	defer entry_ns->release()
	ffn := lib->newFunctionWithName(entry_ns)
	vs := `#version 300 es
void main() {
	vec2 pos = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
	gl_Position = vec4(pos, 0.0, 1.0);
}`
	vm, verrs, vok := oksl.compile(vs, {profile = .Glsl, stage = .Vertex, layout = .Oplat})
	defer oksl.destroy(vm)
	if !vok {
		fmt.eprint(oksl.format_errors(verrs))
		return false
	}
	vlib, vlok := compile_msl_lib(device, "all_types.vert", vm.msl)
	if !vlok {
		return false
	}
	defer vlib->release()
	vfn := vlib->newFunctionWithName(entry_ns)
	if vfn == nil || ffn == nil {
		fmt.eprintln("all_types: missing oksl_main in Metal library")
		return false
	}
	defer vfn->release()
	defer ffn->release()

	W :: 8
	H :: 8
	td := MTL.TextureDescriptor.texture2DDescriptorWithPixelFormat(.RGBA8Unorm, W, H, false)
	td->setUsage({.RenderTarget})
	td->setStorageMode(.Shared)
	tex := device->newTextureWithDescriptor(td)
	if tex == nil {
		fmt.eprintln("all_types: failed to create render target texture")
		return false
	}
	defer tex->release()

	rpd := MTL.new(MTL.RenderPipelineDescriptor)
	defer rpd->release()
	rpd->setVertexFunction(vfn)
	rpd->setFragmentFunction(ffn)
	rpd->colorAttachments()->object(0)->setPixelFormat(.RGBA8Unorm)
	pso, perr := device->newRenderPipelineStateWithDescriptor(rpd)
	if pso == nil {
		msg := NS.to_odin_string(perr->localizedDescription(), context.temp_allocator) if perr != nil else "unknown PSO error"
		fmt.eprintfln("all_types: PSO failed: %s", msg)
		return false
	}
	defer pso->release()

	pass := MTL.RenderPassDescriptor.renderPassDescriptor()
	ca := pass->colorAttachments()->object(0)
	ca->setTexture(tex)
	ca->setLoadAction(.Clear)
	ca->setStoreAction(.Store)
	ca->setClearColor(MTL.ClearColor{0, 0, 0, 1})

	cb := queue->commandBuffer()
	enc := cb->renderCommandEncoderWithDescriptor(pass)
	enc->setRenderPipelineState(pso)
	enc->setFragmentBytes(push_bytes, 0)
	enc->drawPrimitives(.Triangle, 0, 3)
	enc->endEncoding()
	cb->commit()
	cb->waitUntilCompleted()

	gpu_px := make([]u8, W * H * 4)
	defer delete(gpu_px)
	tex->getBytes(raw_data(gpu_px), W * 4, MTL.Region{origin = {0, 0, 0}, size = {W, H, 1}}, 0)

	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	cpu_px := make([]u8, W * H * 4)
	defer delete(cpu_px)
	oksl.run(&m.program, push_bytes, W, H, cpu_px, W * 4, &scratch)

	max_diff := 0
	for i in 0 ..< len(cpu_px) {
		d := abs(int(gpu_px[i]) - int(cpu_px[i]))
		max_diff = max(max_diff, d)
	}
	fmt.printfln("all_types parity: max channel diff = %d (expected <= 1)", max_diff)
	return max_diff <= 1
}

render_and_check_discard_terminate :: proc(device: ^MTL.Device, queue: ^MTL.CommandQueue, m: ^oksl.Module) -> bool {
	if !strings.contains(m.msl, "bool _discarded;") ||
	   !strings.contains(m.msl, "discard_fragment();") ||
	   !strings.contains(m.msl, "g._discarded = true;") ||
	   !strings.contains(m.msl, "if (g._discarded) return;") ||
	   !strings.contains(m.msl, "if (g._discarded) return Oksl_Out{};") {
		fmt.eprintfln("discard_term: missing early-return discard guards in MSL:\n%s", m.msl)
		return false
	}

	lib, ok := compile_msl_lib(device, "discard_term.frag", m.msl)
	if !ok {
		return false
	}
	defer lib->release()
	entry_ns := NS.to_ns_string(oksl.MSL_ENTRY)
	defer entry_ns->release()
	ffn := lib->newFunctionWithName(entry_ns)
	vs := `#version 300 es
void main() {
	vec2 pos = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
	gl_Position = vec4(pos, 0.0, 1.0);
}`
	vm, verrs, vok := oksl.compile(vs, {profile = .Glsl, stage = .Vertex, layout = .Oplat})
	defer oksl.destroy(vm)
	if !vok {
		fmt.eprint(oksl.format_errors(verrs))
		return false
	}
	vlib, vlok := compile_msl_lib(device, "discard_term.vert", vm.msl)
	if !vlok {
		return false
	}
	defer vlib->release()
	vfn := vlib->newFunctionWithName(entry_ns)
	defer vfn->release()
	defer ffn->release()

	W :: 16
	H :: 16
	td := MTL.TextureDescriptor.texture2DDescriptorWithPixelFormat(.RGBA8Unorm, W, H, false)
	td->setUsage({.RenderTarget})
	td->setStorageMode(.Shared)
	tex := device->newTextureWithDescriptor(td)
	defer tex->release()

	rpd := MTL.new(MTL.RenderPipelineDescriptor)
	defer rpd->release()
	rpd->setVertexFunction(vfn)
	rpd->setFragmentFunction(ffn)
	rpd->colorAttachments()->object(0)->setPixelFormat(.RGBA8Unorm)
	pso, _ := device->newRenderPipelineStateWithDescriptor(rpd)
	if pso == nil {
		return false
	}
	defer pso->release()

	pd := Push_Discard{iters = 16_000_000, seed = 0.12345}
	push_bytes := slice.bytes_from_ptr(&pd, size_of(pd))
	pass := MTL.RenderPassDescriptor.renderPassDescriptor()
	ca := pass->colorAttachments()->object(0)
	ca->setTexture(tex)
	ca->setLoadAction(.Clear)
	ca->setStoreAction(.Store)
	ca->setClearColor(MTL.ClearColor{0.2, 0.4, 0.6, 1.0})

	start := time.now()
	cb := queue->commandBuffer()
	enc := cb->renderCommandEncoderWithDescriptor(pass)
	enc->setRenderPipelineState(pso)
	enc->setFragmentBytes(push_bytes, 0)
	enc->drawPrimitives(.Triangle, 0, 3)
	enc->endEncoding()
	cb->commit()
	cb->waitUntilCompleted()
	elapsed := time.since(start)

	gpu_px := make([]u8, W * H * 4)
	defer delete(gpu_px)
	tex->getBytes(raw_data(gpu_px), W * 4, MTL.Region{origin = {0, 0, 0}, size = {W, H, 1}}, 0)

	// Every pixel discarded on iteration 0: texture must keep the clear colour (51, 102, 153, 255).
	want_clear := [4]u8{51, 102, 153, 255}
	for p in 0 ..< W * H {
		for c in 0 ..< 4 {
			if abs(int(gpu_px[p * 4 + c]) - int(want_clear[c])) > 2 {
				fmt.eprintfln("discard_term: pixel %d ch %d expected clear %d, got %d", p, c, want_clear[c], gpu_px[p * 4 + c])
				return false
			}
		}
	}
	ms := f64(time.duration_nanoseconds(elapsed)) / 1e6
	if elapsed > 500 * time.Millisecond {
		fmt.eprintfln("discard_term: 16M-iter loop kept running after discard (%.2f ms)", ms)
		return false
	}
	fmt.printfln("discard_term: Metal 3.0 discard exits caller loop immediately (%.2f ms): ok", ms)
	return true
}

main :: proc() {
	device := MTL.CreateSystemDefaultDevice()
	if device == nil {
		fmt.eprintln("no Metal device")
		os.exit(1)
	}
	defer device->release()
	queue := device->newCommandQueue()
	defer queue->release()

	p, errs, ok := oksl.compile_pipeline({vertex = VS_SRC, fragment = FS_SRC}, {layout = .Oplat})
	defer oksl.destroy_pipeline(p)
	if !ok {
		fmt.eprint(oksl.format_errors(errs))
		os.exit(1)
	}
	draw := Push_Draw {
		view_proj = {
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
		tint  = {0.2, 0.4, 0.1, 0.25},
		count = 2,
	}
	if !render_and_check_triangle(device, queue, p, slice.bytes_from_ptr(&draw, size_of(draw))) {
		os.exit(1)
	}

	am, aerrs, aok := oksl.compile(ALL_TYPES_FS, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
	defer oksl.destroy(am)
	if !aok {
		fmt.eprint(oksl.format_errors(aerrs))
		os.exit(1)
	}
	all_types := All_Types_Push {
		v3 = {0.35, 0.25, 0.45},
		m2 = matrix[2, 2]f32{
			0.8, 0.4,
			0.3, 0.9,
		},
		iv = {2, 3},
		uv = {1, 2, 3},
		v4 = {0.15, 0.35, 0.2, 0.5},
		m3 = matrix[3, 3]f32{
			0.12, 0.45, 0.78,
			0.23, 0.56, 0.89,
			0.34, 0.67, 0.91,
		},
		arr = {
			{a = {0.2, 0.6}, b = 0.4},
			{a = {0.7, 0.1}, b = 0.8},
		},
	}
	if !render_and_check_all_types(device, queue, am, slice.bytes_from_ptr(&all_types, size_of(all_types))) {
		os.exit(1)
	}
	fmt.println("all_types.frag: Metal 3.0 render parity & static_asserts ok")

	dm, derrs, dok := oksl.compile(DISCARD_TERM_FS, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
	defer oksl.destroy(dm)
	if !dok {
		fmt.eprint(oksl.format_errors(derrs))
		os.exit(1)
	}
	if !render_and_check_discard_terminate(device, queue, dm) {
		os.exit(1)
	}
	fmt.println("metal_push: all checks passed")
}
