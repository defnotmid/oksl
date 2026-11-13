package oksl

import "base:intrinsics"
import "core:math"
import "core:math/linalg"

// CPU backend, part 2: the interpreter. Pixels run in batches of LANES:
// 2x2 quads in a BATCH_W x BATCH_H block, so derivatives use the quad
// neighbours like a GPU's fine derivatives, and a block of neighbouring
// pixels diverges little.

// Samples channel `channel` at normalized `uv` (origin top-left) and mip
// level `lod`; returns RGBA. The level texture() picks is fractional and
// may be below 0 (magnified, -inf where uv is constant): clamp it to the
// levels the channel has. `layer` is a sampler2DArray's, its coordinate
// rounded to nearest even (0 for a sampler2D): clamp it to the layers.
Sampler_Proc :: #type proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32

Run_Options :: struct {
	sampler:  Sampler_Proc,
	user:     rawptr,
	// Region to shade in layer pixels, clipped to the layer; zero means
	// the whole layer. Its pixels come out as a whole-layer run's do.
	x, y:     int,
	w, h:     int,
	// The stage inputs by location. Absent ones are zero, except location
	// 0, which is the layer's uv (what the layer vertex shader writes).
	varyings: []Varying,
	// Each channel's size in texels (level 0) and its layers (a
	// sampler2DArray's; ignored for a sampler2D), for textureSize and
	// texelFetch in GLSL shaders and the level texture() and textureGrad
	// pick. A channel without one is 0x0, and texture() samples it at its
	// bias.
	channel_sizes: [][3]i32,
	// run_vertex: gl_VertexID of its first vertex (vertex i's is
	// first_vertex + i), as a draw's first vertex counts on the GPU.
	first_vertex: int,
	// The storage buffers' words, in SDL_GPU binding order
	// (Module.storage_buffers: read-only first, then read-write/writeonly).
	// A read past a buffer's words gives 0 here and a write past them is
	// dropped; on the GPU out-of-bounds access is undefined (Metal reads past
	// the buffer).
	buffers: [][]u32,
	// The storage images for run_compute, in SDL_GPU binding order
	// (Module.storage_images: read-only first, then read-write/writeonly).
	// Out-of-bounds loads read 0 and out-of-bounds stores are dropped.
	images:         []Storage_Image,
	// Optional single-level 2D float images indexed by channel. Channels with
	// `len(data) > 0` sample directly from the image (all mip levels reading
	// level 0, `texelFetch` returning 0 out of range, `textureSize` returning
	// `width x height`); unbound channels continue to call `sampler`.
	channel_images: []Channel_Image,
}

Channel_Filter :: enum u8 {
	Nearest,
	Linear,
}

// A 2D single-level float image bound to a sampled channel on the CPU:
// `width x height` tightly packed texels of `components` (1, 2 or 4) floats
// in `data`, clamped to the edge, missing components reading (0, 0, 0, 1).
// An entry with `len(data) == 0` leaves the channel unbound so
// `Run_Options.sampler` is used.
Channel_Image :: struct {
	data:          []f32,
	width, height: int,
	components:    int,
	filter:        Channel_Filter,
}

channel_image :: proc "contextless" (texels: []$T, width, height: int, filter := Channel_Filter.Nearest) -> Channel_Image
	where T == f32 || T == [2]f32 || T == [4]f32 {
	N :: size_of(T) / size_of(f32)
	return {
		data       = ([^]f32)(raw_data(texels))[:len(texels) * N],
		width      = width,
		height     = height,
		components = N,
		filter     = filter,
	}
}

@(private)
Chan_Img :: struct {
	ptr:    [^]f32,
	fw:     f32,
	fh:     f32,
	w:      i32,
	max_x:  i32,
	max_y:  i32,
	comps:  u8,
	shift:  u8,
	filter: Channel_Filter,
	_pad:   u8,
}

#assert(size_of(Chan_Img) == 32)

@(private)
validate_channel_images :: proc(p: ^Program, opts: Run_Options) {
	for im, ch in opts.channel_images {
		if len(im.data) == 0 {
			continue
		}
		ensure(ch not_in p.channel_arrays, "oksl: Channel_Image cannot bind to a sampler2DArray channel")
		ensure(im.components == 1 || im.components == 2 || im.components == 4, "oksl: Channel_Image.components must be 1, 2 or 4")
		ensure(im.filter == .Nearest || im.filter == .Linear, "oksl: invalid Channel_Image.filter")
		ensure(im.width > 0 && im.height > 0, "oksl: Channel_Image dimensions out of range")
		px, p_ovf := intrinsics.overflow_mul(im.width, im.height)
		need, n_ovf := intrinsics.overflow_mul(px, im.components)
		ensure(!p_ovf && !n_ovf && need <= int(max(i32)), "oksl: Channel_Image dimensions out of range")
		ensure(len(im.data) >= need, "oksl: Channel_Image.data is shorter than width * height * components")
	}
}

@(private)
fill_chan_imgs :: proc "contextless" (dst: ^[SAMPLER_COUNT]Chan_Img, images: []Channel_Image) {
	n := min(len(images), SAMPLER_COUNT)
	for i in 0 ..< n {
		im := images[i]
		if len(im.data) == 0 {
			continue
		}
		dst[i] = {
			ptr    = raw_data(im.data),
			fw     = f32(im.width),
			fh     = f32(im.height),
			w      = i32(im.width),
			max_x  = i32(im.width - 1),
			max_y  = i32(im.height - 1),
			comps  = u8(im.components),
			shift  = u8(im.components >> 1),
			filter = im.filter,
		}
	}
}

// A 2D storage image for run_compute: `width x height` texels in `data`,
// `stride` bytes per row (0 means tightly packed: width * texel bytes).
Storage_Image :: struct {
	data:          []u8,
	width, height: int,
	stride:        int,
	format:        Image_Format,
}

// A stage input across the layer, as one triangle's rasterization gives
// it: per component, a + dx * x + dy * y at the pixel centre (x + 0.5,
// y + 0.5). Integer inputs take the value converted.
Varying :: struct {
	a, dx, dy: [4]f32,
}

// Shades a `width x height` layer into `out` (RGBA8, `stride` bytes per
// row, top-down) from the output at location 0. `uniforms` is the block
// (`Module.block_size` bytes, with iResolution set to the layer size).
// Discarded pixels are left untouched. `scratch` is the caller's working
// memory (one per thread running at once).
run :: proc(p: ^Program, uniforms: []u8, width, height: int, out: []u8, stride: int, scratch: ^Scratch, opts := Run_Options{}) {
	runner_run(p, uniforms, width, height, opts, out, stride, nil, scratch)
}

// Like `run`, writing unclamped float RGBA (`out` has width * height
// entries, row-major).
run_f32 :: proc(p: ^Program, uniforms: []u8, width, height: int, out: [][4]f32, scratch: ^Scratch, opts := Run_Options{}) {
	targets := [1][][4]f32{out}
	runner_run(p, uniforms, width, height, opts, nil, 0, targets[:], scratch)
}

// Like `run_f32` for every output: `targets[n]` (width * height entries,
// nil to skip) receives the output at location n, integer outputs
// converted to float, missing components 0 (alpha 1).
run_targets :: proc(p: ^Program, uniforms: []u8, width, height: int, targets: [][][4]f32, scratch: ^Scratch, opts := Run_Options{}) {
	runner_run(p, uniforms, width, height, opts, nil, 0, targets, scratch)
}

@(private)
Reg :: [LANES]u32

// Working memory for running programs (registers and masks). A Program is
// only read while it runs, so threads may run one at once, each with its
// own Scratch; it grows to the largest program it has run and is reused,
// so runs allocate nothing once it has.
Scratch :: struct {
	regs:  [dynamic]Reg,
	masks: [dynamic]Lane_Mask,
}

scratch_destroy :: proc(s: ^Scratch) {
	delete(s.regs)
	delete(s.masks)
}

@(private)
Runner :: struct {
	p:         ^Program,
	regs:      []Reg,
	masks:     []Lane_Mask,
	uniforms:  []u8,
	opts:      Run_Options,
	chan_imgs: [SAMPLER_COUNT]Chan_Img,
	// The batch's lanes: the exec mask it starts with.
	lanes:     Lane_Mask,
}

// A runner over `scratch` with the program's constants and uniforms loaded,
// and what follows from them alone computed.
@(private)
runner_begin :: proc(p: ^Program, scratch: ^Scratch, uniforms: []u8, opts: Run_Options) -> Runner {
	if len(scratch.regs) < max(p.num_slots, 1) {
		resize(&scratch.regs, max(p.num_slots, 1))
	}
	if len(scratch.masks) < max(p.num_masks, 2) {
		resize(&scratch.masks, max(p.num_masks, 2))
	}
	r := Runner{p = p, regs = scratch.regs[:], masks = scratch.masks[:], uniforms = uniforms, opts = opts}
	fill_chan_imgs(&r.chan_imgs, opts.channel_images)
	for c in p.consts {
		r.regs[c[0]] = c[1]
	}
	for f in p.fills {
		bits: u32
		if int(f.offset) + 4 <= len(uniforms) {
			bits = (^u32)(&uniforms[f.offset])^
		}
		if f.base == .Bool {
			bits = u32(bits != 0)
		}
		r.regs[f.slot] = bits
	}
	// What is the same in every lane and batch, once.
	execute(&r, p.code[:p.init_len])
	return r
}

// A register's lanes from floats, converted to `base`.
@(private)
fill_reg :: #force_inline proc "contextless" (reg: ^Reg, col: F_Lanes, base: Base) {
	#partial switch base {
	case .Int:
		for l in 0 ..< LANES {reg[l] = u32(f_to_i(col[l]))}
	case .Uint:
		for l in 0 ..< LANES {reg[l] = f_to_u(col[l])}
	case .Bool:
		for l in 0 ..< LANES {reg[l] = u32(col[l] != 0)}
	case:
		fl(reg)^ = col
	}
}

// A component's bits converted between bases, as GLSL's constructors do
// (ints and uints share bits).
@(private)
convert_bits :: #force_inline proc "contextless" (bits: u32, from, to: Base) -> u32 {
	switch {
	case from == to || (from != .Float && to != .Float):
		return bits
	case to == .Float:
		return fbits(float_of(bits, from))
	case to == .Int:
		return u32(f_to_i(fval(bits)))
	}
	return f_to_u(fval(bits))
}

// Component bits of `base` as a float.
@(private)
float_of :: #force_inline proc "contextless" (bits: u32, base: Base) -> f32 {
	#partial switch base {
	case .Int:
		return f32(i32(bits))
	case .Uint:
		return f32(bits)
	case .Bool:
		return 1 if bits != 0 else 0
	}
	return fval(bits)
}

// The layer runner's batch: the lanes' quads in rows of BATCH_W / 2.
BATCH_W :: 8
BATCH_H :: LANES / BATCH_W

@(private)
runner_run :: proc(p: ^Program, uniforms: []u8, width, height: int, opts: Run_Options, out: []u8, stride: int, targets: [][][4]f32, scratch: ^Scratch) {
	ensure(p.stage == .Fragment, "oksl: run requires a fragment program")
	validate_channel_images(p, opts)
	if p.jit_body != nil {
		jit_runner_run(p, uniforms, width, height, opts, out, stride, targets, scratch)
		return
	}
	x0, y0 := opts.x, opts.y
	w, h := opts.w, opts.h
	if w <= 0 || h <= 0 {
		x0, y0, w, h = 0, 0, width, height
	}
	fw, fh := f32(width), f32(height)
	// The region within the layer. Batches start on even pixels, so a
	// region's quads, and so its derivatives, are the whole layer's.
	start_x, start_y := max(x0, 0), max(y0, 0)
	end_x := min(x0 + w, width)
	end_y := min(y0 + h, height)
	if start_x >= end_x || start_y >= end_y {
		return
	}
	r := runner_begin(p, scratch, uniforms, opts)
	// Each lane's pixel in the batch.
	lx, ly: [LANES]int
	lxf, lyf: F_Lanes
	for lane in 0 ..< LANES {
		q := lane >> 2
		lx[lane] = q % (BATCH_W / 2) * 2 + (lane & 1)
		ly[lane] = q / (BATCH_W / 2) * 2 + ((lane >> 1) & 1)
		lxf[lane] = f32(lx[lane]) + 0.5
		lyf[lane] = f32(ly[lane]) + 0.5
	}
	for by := start_y &~ 1; by < end_y; by += BATCH_H {
		for bx := start_x &~ 1; bx < end_x; bx += BATCH_W {
			// Pixel centres of the batch's lanes, then each input a whole
			// register at a time.
			px := f32(bx) + lxf
			py := f32(by) + lyf
			for v in p.inputs {
				for k in 0 ..< int(v.n) {
					col: F_Lanes
					switch {
					case v.kind == .Frag_Coord:
						col = px if k == 0 else (py if k == 1 else (1 if k == 3 else 0))
					case v.kind == .Front_Facing:
						// A layer faces the camera.
						col = 1
					case int(v.location) < len(opts.varyings):
						vy := opts.varyings[v.location]
						col = vy.a[k] + vy.dx[k] * px + vy.dy[k] * py
					case v.location == 0 && k < 2:
						col = px / fw if k == 0 else py / fh
					}
					fill_reg(&r.regs[v.slot + i32(k)], col, v.base)
				}
			}
			// The quads the region touches run, all four lanes of each (as
			// a GPU's helpers); the batch's others sample nothing.
			inside := bx >= start_x && by >= start_y && bx + BATCH_W <= end_x && by + BATCH_H <= end_y
			exec := ALL_LANES
			if !inside {
				exec = {}
				for lane in 0 ..< LANES {
					qx, qy := bx + lx[lane] &~ 1, by + ly[lane] &~ 1
					if qx + 1 >= start_x && qx < end_x && qy + 1 >= start_y && qy < end_y {
						exec += {lane}
					}
				}
			}
			r.masks[MASK_EXEC] = exec
			r.lanes = exec
			r.masks[MASK_DISCARD] = {}
			execute(&r, p.code[p.init_len:])
			alive := ~r.masks[MASK_DISCARD]
			full_tile := inside && alive == ALL_LANES
			for v in p.outputs {
				if v.kind != .Stage_Out {
					continue
				}
				target := targets[v.location] if int(v.location) < len(targets) else nil
				bytes := out if v.location == 0 else nil
				if target == nil && bytes == nil {
					continue
				}
				if full_tile && v.base == .Float && v.n == 4 {
					r0 := &r.regs[v.slot]
					r1 := &r.regs[v.slot + 1]
					r2 := &r.regs[v.slot + 2]
					r3 := &r.regs[v.slot + 3]
					if target != nil {
						for lane in 0 ..< LANES {
							target[(by + ly[lane]) * width + bx + lx[lane]] = {fval(r0[lane]), fval(r1[lane]), fval(r2[lane]), fval(r3[lane])}
						}
					}
					if bytes != nil {
						for lane in 0 ..< LANES {
							o := (by + ly[lane]) * stride + (bx + lx[lane]) * 4
							c := [4]f32{fval(r0[lane]), fval(r1[lane]), fval(r2[lane]), fval(r3[lane])}
							for k in 0 ..< 4 {
								cv := c[k]
								if !(cv > 0) {cv = 0}
								bytes[o + k] = u8(min(cv, 1) * 255 + 0.5)
							}
						}
					}
					continue
				}
				for lane in 0 ..< LANES {
					x := bx + lx[lane]
					y := by + ly[lane]
					if x < start_x || y < start_y || x >= end_x || y >= end_y || lane not_in alive {
						continue
					}
					c := [4]f32{0, 0, 0, 1}
					for k in 0 ..< int(v.n) {
						c[k] = float_of(r.regs[v.slot + i32(k)][lane], v.base)
					}
					if target != nil {
						target[y * width + x] = c
					}
					if bytes != nil {
						o := y * stride + x * 4
						for k in 0 ..< 4 {
							cv := c[k]
							if !(cv > 0) {cv = 0}
							bytes[o + k] = u8(min(cv, 1) * 255 + 0.5)
						}
					}
				}
			}
		}
	}
}

// ---- vertices and quads: for hosts that rasterize ----------------------------------------

// What an attribute's data holds per vertex: 1 to 4 components of f32, i32
// or u32, or four packed ones as SDL_GPU's formats of the name have them.
Vertex_Format :: enum u8 {
	Float,
	Float2,
	Float3,
	Float4,
	Int,
	Int2,
	Int3,
	Int4,
	Uint,
	Uint2,
	Uint3,
	Uint4,
	// Four u8, each / 255, as floats.
	Ubyte4_Norm,
	// Four u16, as uints.
	Ushort4,
}

@(private, rodata)
FORMAT_BASES := [3]Base{.Float, .Int, .Uint}

// A vertex attribute: vertex i's value at `data + i * stride` bytes.
Attribute :: struct {
	data:   rawptr,
	stride: int,
	format: Vertex_Format,
}

// Runs a vertex program over `count` vertices. `attributes[n]` feeds the
// input at location n (absent or nil data: (0, 0, 0, 1)); `position[i]`
// receives vertex i's gl_Position and `varyings[i * p.varying_floats:]`
// its outputs at their offsets (Stage_Var_Info.offset), integers (flat) as
// their bits, so every 32-bit value is exact. Threads may run one program
// at once, each with its own scratch.
run_vertex :: proc(p: ^Program, uniforms: []u8, attributes: []Attribute, count: int, position: [][4]f32, varyings: []f32, scratch: ^Scratch, opts := Run_Options{}) {
	ensure(p.stage == .Vertex, "oksl: run_vertex requires a vertex program")
	validate_channel_images(p, opts)
	if count <= 0 {
		return
	}
	if p.jit_body != nil {
		jit_run_vertex(p, uniforms, attributes, count, position, varyings, scratch, opts)
		return
	}
	r := runner_begin(p, scratch, uniforms, opts)
	n := p.varying_floats
	for first := 0; first < count; first += LANES {
		lanes := min(count - first, LANES)
		// Per component, one of: absent (0, 0, 0, 1), as stored, converted.
		for v in p.inputs {
			if v.kind == .Vertex_ID {
				reg := &r.regs[v.slot]
				for l in 0 ..< lanes {reg[l] = u32(i32(opts.first_vertex + first + l))}
				continue
			}
			a := attributes[v.location] if int(v.location) < len(attributes) else Attribute{}
			comps := 4
			if a.format <= .Uint4 {
				comps = int(a.format) % 4 + 1
			}
			for k in 0 ..< int(v.n) {
				reg := &r.regs[v.slot + i32(k)]
				if a.data == nil || k >= comps {
					fill := convert_bits(1 if k == 3 else 0, .Int, v.base)
					for l in 0 ..< lanes {reg[l] = fill}
					continue
				}
				vertex := uintptr(a.data) + uintptr(first * a.stride)
				#partial switch a.format {
				case .Ubyte4_Norm:
					for l in 0 ..< lanes {reg[l] = convert_bits(fbits(f32((^u8)(vertex + uintptr(l * a.stride + k))^) / 255), .Float, v.base)}
				case .Ushort4:
					for l in 0 ..< lanes {reg[l] = convert_bits(u32((^u16)(vertex + uintptr(l * a.stride + k * 2))^), .Uint, v.base)}
				case:
					from := FORMAT_BASES[int(a.format) / 4]
					words := vertex + uintptr(k * 4)
					if from == v.base {
						for l in 0 ..< lanes {reg[l] = (^u32)(words + uintptr(l * a.stride))^}
					} else {
						for l in 0 ..< lanes {reg[l] = convert_bits((^u32)(words + uintptr(l * a.stride))^, from, v.base)}
					}
				}
			}
		}
		r.masks[MASK_EXEC] = transmute(Lane_Mask)Lane_Bits((1 << uint(lanes)) - 1)
		r.lanes = r.masks[MASK_EXEC]
		r.masks[MASK_DISCARD] = {}
		execute(&r, p.code[p.init_len:])
		for v in p.outputs {
			if v.kind == .Position {
				for k in 0 ..< 4 {
					reg := fl(&r.regs[v.slot + i32(k)])
					for l in 0 ..< lanes {position[first + l][k] = reg[l]}
				}
				continue
			}
			for k in 0 ..< int(v.n) {
				reg := &r.regs[v.slot + i32(k)]
				out := varyings[first * n + int(v.offset) + k:]
				for l in 0 ..< lanes {out[l * n] = fval(reg[l])}
			}
		}
	}
}

// Lanes of a 2x2 quad: lane i is pixel (i & 1, i >> 1) from its top-left.
Quad_Lanes :: bit_set[0 ..< 4;u8]

// Quads of fragments a host has rasterized, for run_fragment.
Fragment_Quads :: struct {
	count:        int,
	// Each quad's top-left pixel. gl_FragCoord is a lane's pixel centre
	// (z 0, w 1).
	origin:       [][2]i32,
	// Per quad; nil: every quad faces the camera.
	front_facing: []bool,
	// Per quad, the lanes outside the primitive: they run (derivatives read
	// them) but produce nothing. Nil: none.
	helpers:      []Quad_Lanes,
	// The interpolated varyings: float k of the record (Stage_Var_Info.offset)
	// for lane j (quad j / 4, lane j % 4) at varyings[k * count * 4 + j].
	// Integer (flat) inputs are their bits, as run_vertex writes them:
	// take the provoking vertex's.
	varyings:     []f32,
	// Per lane (quad j / 4, lane j % 4), the rasterizer's depth: gl_FragCoord.z.
	// Nil: 0.
	depth:        []f32,
}

// Shades quads with a fragment program, four at a time like a GPU's 2x2
// quads (dFdx / dFdy within each). `targets[n][j]` (nil to skip) receives
// lane j's output at location n (four lanes per quad; integer outputs
// converted to float, missing components 0, alpha 1, (0, 0, 0, 1) where
// the shader writes no location n), `shaded[q]` the lanes of quad q that
// produced one (not helpers, not discarded), and `sample_masks[j]` (when
// given) the samples lane j keeps: gl_SampleMask[0]'s bits, or all of them
// when the shader does not write it, for the host to AND with the
// fragment's coverage. Threads may run one program at once, each with its
// own scratch.
run_fragment :: proc(p: ^Program, uniforms: []u8, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, scratch: ^Scratch, opts := Run_Options{}, sample_masks: []u32 = nil) {
	ensure(p.stage == .Fragment, "oksl: run_fragment requires a fragment program")
	validate_channel_images(p, opts)
	if quads.count <= 0 {
		return
	}
	if p.jit_body != nil {
		jit_run_fragment(p, uniforms, quads, targets, shaded, scratch, opts, sample_masks)
		return
	}
	r := runner_begin(p, scratch, uniforms, opts)
	total := quads.count * 4
	// The output at each location (n == 0: none).
	outs: [FRAGMENT_OUTPUTS]Program_Var
	mask_v: Program_Var
	has_mask := false
	for v in p.outputs {
		if v.kind == .Stage_Out && int(v.location) < len(outs) {
			outs[v.location] = v
		}
		if v.kind == .Sample_Mask {
			mask_v = v
			has_mask = true
		}
	}
	for first := 0; first < quads.count; first += LANES / 4 {
		nq := min(quads.count - first, LANES / 4)
		lanes := nq * 4
		for v in p.inputs {
			for k in 0 ..< int(v.n) {
				col: F_Lanes
				#partial switch v.kind {
				case .Frag_Coord:
					switch k {
					case 0:
						for l in 0 ..< lanes {col[l] = f32(quads.origin[first + l / 4].x + i32(l & 1)) + 0.5}
					case 1:
						for l in 0 ..< lanes {col[l] = f32(quads.origin[first + l / 4].y + i32((l >> 1) & 1)) + 0.5}
					case 2:
						if quads.depth != nil {
							for l in 0 ..< lanes {col[l] = quads.depth[first * 4 + l]}
						}
					case 3:
						for l in 0 ..< lanes {col[l] = 1}
					}
				case .Front_Facing:
					if quads.front_facing == nil {
						for l in 0 ..< lanes {col[l] = 1}
					} else {
						for l in 0 ..< lanes {col[l] = 1 if quads.front_facing[first + l / 4] else 0}
					}
				case .Stage_In:
					// The record's bits: floats, and integers as run_vertex
					// writes them.
					src := quads.varyings[(int(v.offset) + k) * total + first * 4:]
					reg := fl(&r.regs[v.slot + i32(k)])
					for l in 0 ..< lanes {reg[l] = src[l]}
					continue
				}
				fill_reg(&r.regs[v.slot + i32(k)], col, v.base)
			}
		}
		r.masks[MASK_EXEC] = transmute(Lane_Mask)Lane_Bits((1 << uint(lanes)) - 1)
		r.lanes = r.masks[MASK_EXEC]
		r.masks[MASK_DISCARD] = {}
		execute(&r, p.code[p.init_len:])
		alive := ~r.masks[MASK_DISCARD]
		for qi in 0 ..< nq {
			q := first + qi
			lanes_out: Quad_Lanes
			helpers := quads.helpers[q] if quads.helpers != nil else {}
			for i in 0 ..< 4 {
				l := qi * 4 + i
				if i in helpers || l not_in alive {
					continue
				}
				lanes_out += {i}
				for target, n in targets {
					if target == nil {
						continue
					}
					c := [4]f32{0, 0, 0, 1}
					if n < len(outs) {
						v := outs[n]
						for k in 0 ..< int(v.n) {
							c[k] = float_of(r.regs[v.slot + i32(k)][l], v.base)
						}
					}
					target[q * 4 + i] = c
				}
				if sample_masks != nil {
					sample_masks[q * 4 + i] = r.regs[mask_v.slot][l] if has_mask else ~u32(0)
				}
			}
			shaded[q] = lanes_out
		}
	}
}

// Runs a compute program over `groups.x * groups.y * groups.z` workgroups
// (or `[first_group, first_group + group_count)` when `group_count >= 0`),
// one workgroup at a time in 64-lane chunks. Workgroup `g` has
// `gl_WorkGroupID = (g % groups.x, (g / groups.x) % groups.y, g / (groups.x * groups.y))`.
// Local sizes below 64 leave the rest of a batch's lanes idle; threads may
// run disjoint workgroup ranges at once, each with its own scratch. Panics
// when `opts.buffers` or `opts.images` is missing declared resources or a
// `Storage_Image` has a mismatched format, negative dimensions, or an
// undersized `stride` / `data` slice.
run_compute :: proc(p: ^Program, uniforms: []u8, groups: [3]int, scratch: ^Scratch, opts := Run_Options{}, first_group := 0, group_count := -1) {
	ensure(p.stage == .Compute, "oksl: run_compute requires a compute program")
	ensure(len(opts.buffers) >= p.num_buffers, "oksl: Run_Options.buffers has fewer buffers than declared")
	ensure(len(opts.images) >= len(p.image_formats), "oksl: Run_Options.images has fewer images than declared")
	validate_channel_images(p, opts)
	for want_fmt, i in p.image_formats {
		im := opts.images[i]
		ensure(im.format == want_fmt, "oksl: Storage_Image.format does not match shader declaration")
		ensure(im.width >= 0 && im.height >= 0 && im.width <= int(max(i32)) && im.height <= int(max(i32)), "oksl: Storage_Image dimensions out of range")
		tb := image_texel_bytes(want_fmt)
		min_stride, s_ovf := intrinsics.overflow_mul(im.width, tb)
		ensure(!s_ovf, "oksl: Storage_Image width overflows")
		row_stride := im.stride if im.stride != 0 else min_stride
		ensure(row_stride >= min_stride, "oksl: Storage_Image.stride is smaller than width * texel_bytes")
		if im.width > 0 && im.height > 0 {
			prev_rows, r_ovf := intrinsics.overflow_mul(im.height - 1, row_stride)
			need, n_ovf := intrinsics.overflow_add(prev_rows, min_stride)
			ensure(!r_ovf && !n_ovf && len(im.data) >= need, "oksl: Storage_Image.data is too small for width, height and stride")
		}
	}
	if groups.x <= 0 || groups.y <= 0 || groups.z <= 0 {
		return
	}
	gxy, xy_ovf := intrinsics.overflow_mul(groups.x, groups.y)
	total_groups, tg_ovf := intrinsics.overflow_mul(gxy, groups.z)
	ensure(!xy_ovf && !tg_ovf, "oksl: dispatch group count overflows")
	start := max(first_group, 0)
	if start >= total_groups || group_count == 0 {
		return
	}
	end := total_groups if group_count < 0 else start + min(group_count, total_groups - start)
	if first_group < 0 && group_count >= 0 {
		end = min(first_group + group_count, total_groups)
		if start >= end {
			return
		}
	}
	ls := [3]int{max(int(p.local_size[0]), 1), max(int(p.local_size[1]), 1), max(int(p.local_size[2]), 1)}
	local_total := ls.x * ls.y * ls.z
	local_coords: [1024][3]u16
	ti := 0
	for lz in 0 ..< ls.z {
		for ly in 0 ..< ls.y {
			for lx in 0 ..< ls.x {
				local_coords[ti] = {u16(lx), u16(ly), u16(lz)}
				ti += 1
			}
		}
	}
	r := runner_begin(p, scratch, uniforms, opts)
	single_chunk := local_total <= LANES
	for v in p.inputs {
		#partial switch v.kind {
		case .Num_Work_Groups:
			for k in 0 ..< 3 {
				r.regs[v.slot + i32(k)] = u32(groups[k])
			}
		case .Local_Invocation_ID:
			if single_chunk {
				for k in 0 ..< 3 {
					reg := &r.regs[v.slot + i32(k)]
					for l in 0 ..< LANES {
						reg[l] = u32(local_coords[l][k])
					}
				}
			}
		case .Local_Invocation_Index:
			if single_chunk {
				reg := &r.regs[v.slot]
				for l in 0 ..< LANES {
					reg[l] = u32(l)
				}
			}
		}
	}
	gx_n := groups.x
	gxy_n := groups.x * groups.y
	for g in start ..< end {
		wg := [3]u32{u32(g % gx_n), u32((g / gx_n) % groups.y), u32(g / gxy_n)}
		base_gid := [3]u32{wg[0] * u32(ls.x), wg[1] * u32(ls.y), wg[2] * u32(ls.z)}
		for v in p.inputs {
			if v.kind == .Work_Group_ID {
				for k in 0 ..< 3 {
					r.regs[v.slot + i32(k)] = wg[k]
				}
			}
		}
		for first := 0; first < local_total; first += LANES {
			lanes := min(local_total - first, LANES)
			for v in p.inputs {
				#partial switch v.kind {
				case .Global_Invocation_ID:
					for k in 0 ..< 3 {
						reg := &r.regs[v.slot + i32(k)]
						bg := base_gid[k]
						for l in 0 ..< LANES {
							reg[l] = bg + u32(local_coords[first + l][k])
						}
					}
				case .Local_Invocation_ID:
					if !single_chunk {
						for k in 0 ..< 3 {
							reg := &r.regs[v.slot + i32(k)]
							for l in 0 ..< LANES {
								reg[l] = u32(local_coords[first + l][k])
							}
						}
					}
				case .Local_Invocation_Index:
					if !single_chunk {
						reg := &r.regs[v.slot]
						for l in 0 ..< LANES {
							reg[l] = u32(first + l)
						}
					}
				}
			}
			exec := ALL_LANES if lanes == LANES else transmute(Lane_Mask)Lane_Bits((Lane_Bits(1) << uint(lanes)) - 1)
			r.masks[MASK_EXEC] = exec
			r.lanes = exec
			r.masks[MASK_DISCARD] = {}
			execute(&r, p.code[p.init_len:])
		}
	}
}

// floor without core:math's modf: values of 2^23 and up (and inf/NaN) are
// already integral.
@(private)
ffloor :: #force_inline proc "contextless" (x: f32) -> f32 {
	if !(abs(x) < 8388608) || x == 0 {
		return x
	}
	t := f32(i32(x))
	return t - 1 if t > x else t
}

@(private)
fceil :: #force_inline proc "contextless" (x: f32) -> f32 {
	return -ffloor(-x)
}

// A register's lanes as floats.
@(private)
F_Lanes :: [LANES]f32

@(private)
Simd_U8 :: #simd[8]u32

@(private)
LANE_BIT :: Simd_U8{1, 2, 4, 8, 16, 32, 64, 128}

@(private)
fl :: #force_inline proc "contextless" (r: ^Reg) -> ^F_Lanes {
	return (^F_Lanes)(r)
}

@(private)
fbits :: #force_inline proc "contextless" (v: f32) -> u32 {
	return transmute(u32)v
}

@(private)
fval :: #force_inline proc "contextless" (v: u32) -> f32 {
	return transmute(f32)v
}

@(private)
f_to_i :: proc "contextless" (f: f32) -> i32 {
	if !(f == f) {
		return 0
	}
	if f >= 2147483647 {
		return max(i32)
	}
	if f <= -2147483648 {
		return min(i32)
	}
	return i32(f)
}

@(private)
f_to_u :: proc "contextless" (f: f32) -> u32 {
	if !(f > 0) {
		return 0
	}
	if f >= 4294967295 {
		return max(u32)
	}
	return u32(f)
}

// Two-operand integer, comparison and logic ops over whole registers. OP is
// a compile-time constant, so each use is its own branch-free loop.
@(private)
bin_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $OP: VM_Op) {
	for i in 0 ..< i32(in_.n) {
		x, y, d := &regs[in_.a + i], &regs[in_.b + i], &regs[in_.dst + i]
		for l in 0 ..< LANES {
			d[l] = bin_lane(x[l], y[l], OP)
		}
	}
}

@(private)
bin_lane :: #force_inline proc "contextless" (x, y: u32, $OP: VM_Op) -> u32 {
	when OP == .I_Add {
		return x + y
	} else when OP == .I_Sub {
		return x - y
	} else when OP == .I_Mul {
		return x * y
	} else when OP == .I_Div {
		sx, sy := i32(x), i32(y)
		return 0 if sy == 0 else (x if sy == -1 && sx == min(i32) else u32(sx / sy))
	} else when OP == .U_Div {
		return 0 if y == 0 else x / y
	} else when OP == .I_Rem {
		sx, sy := i32(x), i32(y)
		return 0 if sy == 0 || sy == -1 else u32(sx % sy)
	} else when OP == .U_Rem {
		return 0 if y == 0 else x % y
	} else when OP == .Bit_And {
		return x & y
	} else when OP == .Bit_Or {
		return x | y
	} else when OP == .Bit_Xor {
		return x ~ y
	} else when OP == .Shl {
		return x << (y & 31)
	} else when OP == .Shr_I {
		return u32(i32(x) >> (y & 31))
	} else when OP == .Shr_U {
		return x >> (y & 31)
	} else when OP == .I_Lt {
		return u32(i32(x) < i32(y))
	} else when OP == .I_Le {
		return u32(i32(x) <= i32(y))
	} else when OP == .I_Gt {
		return u32(i32(x) > i32(y))
	} else when OP == .I_Ge {
		return u32(i32(x) >= i32(y))
	} else when OP == .U_Lt {
		return u32(x < y)
	} else when OP == .U_Le {
		return u32(x <= y)
	} else when OP == .U_Gt {
		return u32(x > y)
	} else when OP == .U_Ge {
		return u32(x >= y)
	} else when OP == .I_Eq {
		return u32(x == y)
	} else when OP == .I_Ne {
		return u32(x != y)
	} else when OP == .B_And {
		return u32(x != 0 && y != 0)
	} else when OP == .B_Or {
		return u32(x != 0 || y != 0)
	} else {
		#panic("not a two-operand lane op")
	}
}

// Conversions over whole registers, like bin_lanes.
@(private)
un_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $OP: VM_Op) {
	for i in 0 ..< i32(in_.n) {
		x, d := &regs[in_.a + i], &regs[in_.dst + i]
		for l in 0 ..< LANES {
			d[l] = un_lane(x[l], OP)
		}
	}
}

@(private)
un_lane :: #force_inline proc "contextless" (x: u32, $OP: VM_Op) -> u32 {
	when OP == .F_To_I {
		return u32(f_to_i(fval(x)))
	} else when OP == .F_To_U {
		return f_to_u(fval(x))
	} else when OP == .I_To_F {
		return fbits(f32(i32(x)))
	} else when OP == .U_To_F {
		return fbits(f32(x))
	} else when OP == .B_To_F {
		return fbits(1 if x != 0 else 0)
	} else when OP == .F_To_B {
		return u32(!(fval(x) == 0))
	} else when OP == .I_To_B {
		return u32(x != 0)
	} else {
		#panic("not a conversion")
	}
}

// The float built-ins (k = Builtin) over whole registers: a switch per
// instruction, then a loop specialized for that built-in.
@(private)
float_fn1 :: proc "contextless" (regs: []Reg, in_: Instr) {
	#partial switch Builtin(in_.k) {
	case .Radians:
		fn1_lanes(regs, in_, .Radians)
	case .Degrees:
		fn1_lanes(regs, in_, .Degrees)
	case .Sin:
		fn1_lanes(regs, in_, .Sin)
	case .Cos:
		fn1_lanes(regs, in_, .Cos)
	case .Tan:
		fn1_lanes(regs, in_, .Tan)
	case .Asin:
		fn1_lanes(regs, in_, .Asin)
	case .Acos:
		fn1_lanes(regs, in_, .Acos)
	case .Atan:
		fn1_lanes(regs, in_, .Atan)
	case .Sinh:
		fn1_lanes(regs, in_, .Sinh)
	case .Cosh:
		fn1_lanes(regs, in_, .Cosh)
	case .Tanh:
		fn1_lanes(regs, in_, .Tanh)
	case .Asinh:
		fn1_lanes(regs, in_, .Asinh)
	case .Acosh:
		fn1_lanes(regs, in_, .Acosh)
	case .Atanh:
		fn1_lanes(regs, in_, .Atanh)
	case .Exp:
		fn1_lanes(regs, in_, .Exp)
	case .Log:
		fn1_lanes(regs, in_, .Log)
	case .Exp2:
		fn1_lanes(regs, in_, .Exp2)
	case .Log2:
		fn1_lanes(regs, in_, .Log2)
	case .Sqrt:
		fn1_lanes(regs, in_, .Sqrt)
	case .Inversesqrt:
		fn1_lanes(regs, in_, .Inversesqrt)
	case .Floor:
		fn1_lanes(regs, in_, .Floor)
	case .Ceil:
		fn1_lanes(regs, in_, .Ceil)
	case .Fract:
		fn1_lanes(regs, in_, .Fract)
	case .Trunc:
		fn1_lanes(regs, in_, .Trunc)
	case .Round:
		fn1_lanes(regs, in_, .Round)
	case .Round_Even:
		fn1_lanes(regs, in_, .Round_Even)
	case .Abs:
		fn1_lanes(regs, in_, .Abs)
	case .Sign:
		fn1_lanes(regs, in_, .Sign)
	case .Isnan:
		fn1_lanes(regs, in_, .Isnan)
	case .Isinf:
		fn1_lanes(regs, in_, .Isinf)
	}
}

@(private)
fn1_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $B: Builtin) {
	for i in 0 ..< i32(in_.n) {
		x, d := fl(&regs[in_.a + i]), &regs[in_.dst + i]
		for l in 0 ..< LANES {
			d[l] = fn1(x[l], B)
		}
	}
}

// One float built-in of one operand, as result bits (isnan and isinf give
// bools).
@(private)
fn1 :: #force_inline proc "contextless" (x: f32, $B: Builtin) -> u32 {
	when B == .Radians {
		return fbits(x * (math.PI / 180))
	} else when B == .Degrees {
		return fbits(x * (180 / math.PI))
	} else when B == .Sin {
		return fbits(math.sin(x))
	} else when B == .Cos {
		return fbits(math.cos(x))
	} else when B == .Tan {
		return fbits(math.tan(x))
	} else when B == .Asin {
		return fbits(math.asin(x))
	} else when B == .Acos {
		return fbits(math.acos(x))
	} else when B == .Atan {
		return fbits(math.atan(x))
	} else when B == .Sinh {
		return fbits(math.sinh(x))
	} else when B == .Cosh {
		return fbits(math.cosh(x))
	} else when B == .Tanh {
		return fbits(math.tanh(x))
	} else when B == .Asinh {
		return fbits(math.asinh(x))
	} else when B == .Acosh {
		return fbits(math.acosh(x))
	} else when B == .Atanh {
		return fbits(math.atanh(x))
	} else when B == .Exp {
		return fbits(math.exp(x))
	} else when B == .Log {
		return fbits(math.ln(x))
	} else when B == .Exp2 {
		return fbits(math.pow(f32(2), x))
	} else when B == .Log2 {
		return fbits(math.log2(x))
	} else when B == .Sqrt {
		return fbits(math.sqrt(x))
	} else when B == .Inversesqrt {
		return fbits(1 / math.sqrt(x))
	} else when B == .Floor {
		return fbits(ffloor(x))
	} else when B == .Ceil {
		return fbits(fceil(x))
	} else when B == .Fract {
		return fbits(x - ffloor(x))
	} else when B == .Trunc {
		return fbits(math.trunc(x))
	} else when B == .Round {
		return fbits(math.round(x))
	} else when B == .Round_Even {
		r := math.round(x)
		if abs(x - math.trunc(x)) == 0.5 {
			r = 2 * math.round(x / 2)
		}
		return fbits(r)
	} else when B == .Abs {
		return fbits(x) & 0x7fff_ffff
	} else when B == .Sign {
		return fbits(1 if x > 0 else (-1 if x < 0 else 0))
	} else when B == .Isnan {
		return u32(math.is_nan(x))
	} else when B == .Isinf {
		return u32(math.is_inf(x))
	} else {
		#panic("not a one-operand float built-in")
	}
}

@(private)
float_fn2 :: proc "contextless" (regs: []Reg, in_: Instr) {
	#partial switch Builtin(in_.k) {
	case .Atan2:
		fn2_lanes(regs, in_, .Atan2)
	case .Pow:
		fn2_lanes(regs, in_, .Pow)
	case .Mod:
		fn2_lanes(regs, in_, .Mod)
	case .Min:
		fn2_lanes(regs, in_, .Min)
	case .Max:
		fn2_lanes(regs, in_, .Max)
	case .Step:
		fn2_lanes(regs, in_, .Step)
	}
}

@(private)
fn2_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $B: Builtin) {
	for i in 0 ..< i32(in_.n) {
		x, y, d := fl(&regs[in_.a + i]), fl(&regs[in_.b + i]), fl(&regs[in_.dst + i])
		for l in 0 ..< LANES {
			d[l] = fn2(x[l], y[l], B)
		}
	}
}

@(private)
fn2 :: #force_inline proc "contextless" (x, y: f32, $B: Builtin) -> f32 {
	when B == .Atan2 {
		return math.atan2(x, y)
	} else when B == .Pow {
		return math.pow(x, y)
	} else when B == .Mod {
		return x - y * ffloor(x / y)
	} else when B == .Min {
		return y if y < x else x
	} else when B == .Max {
		return y if y > x else x
	} else when B == .Step {
		return 0 if y < x else 1
	} else {
		#panic("not a two-operand float built-in")
	}
}

@(private)
float_fn3 :: proc "contextless" (regs: []Reg, in_: Instr) {
	#partial switch Builtin(in_.k) {
	case .Clamp:
		fn3_lanes(regs, in_, .Clamp)
	case .Mix:
		fn3_lanes(regs, in_, .Mix)
	case .Smoothstep:
		fn3_lanes(regs, in_, .Smoothstep)
	}
}

@(private)
fn3_lanes :: #force_inline proc "contextless" (regs: []Reg, in_: Instr, $B: Builtin) {
	for i in 0 ..< i32(in_.n) {
		x, y, z, d := fl(&regs[in_.a + i]), fl(&regs[in_.b + i]), fl(&regs[in_.c + i]), fl(&regs[in_.dst + i])
		for l in 0 ..< LANES {
			d[l] = fn3(x[l], y[l], z[l], B)
		}
	}
}

@(private)
fn3 :: #force_inline proc "contextless" (x, y, z: f32, $B: Builtin) -> f32 {
	when B == .Clamp {
		return min(max(x, y), z)
	} else when B == .Mix {
		return x * (1 - z) + y * z
	} else when B == .Smoothstep {
		t := clamp((z - x) / (y - x), 0, 1)
		return t * t * (3 - 2 * t)
	} else {
		#panic("not a three-operand float built-in")
	}
}

// Unchecked: every slot and mask an instruction names was allocated by the
// lowering (dynamic offsets are clamped there), within num_slots and
// num_masks, which runner_begin sizes the scratch to.
@(private)
execute :: proc(r: ^Runner, code: []Instr) #no_bounds_check {
	regs := r.regs
	masks := r.masks
	pc := 0
	for pc < len(code) {
		in_ := code[pc]
		pc += 1
		n := int(in_.n)
		dst, a, b, c := in_.dst, in_.a, in_.b, in_.c
		exec := masks[MASK_EXEC]
		switch in_.op {
		case .Nop:
		case .Mov:
			for i in 0 ..< i32(n) {
				regs[dst + i] = regs[a + i]
			}
		case .Splat:
			for i in 0 ..< i32(n) {
				regs[dst + i] = regs[a]
			}
		case .Gather:
			regs[dst] = regs[a]
			if n > 1 {regs[dst + 1] = regs[b]}
			if n > 2 {regs[dst + 2] = regs[c]}
			if n > 3 {regs[dst + 3] = regs[in_.d]}
		case .Store:
			if exec == ALL_LANES {
				for i in 0 ..< i32(n) {
					regs[dst + i] = regs[a + i]
				}
				break
			}
			if exec == {} {
				break
			}
			ebits := u64(transmute(Lane_Bits)exec)
			km: [LANES / 8]Simd_U8
			#unroll for k in 0 ..< LANES / 8 {
				km[k] = intrinsics.simd_lanes_ne(Simd_U8(u32(u8(ebits >> uint(k * 8)))) & LANE_BIT, Simd_U8(0))
			}
			for i in 0 ..< i32(n) {
				d, x := &regs[dst + i], &regs[a + i]
				#unroll for k in 0 ..< LANES / 8 {
					dp := (^Simd_U8)(&d[k * 8])
					xp := (^Simd_U8)(&x[k * 8])
					intrinsics.unaligned_store(dp, intrinsics.simd_select(km[k], intrinsics.unaligned_load(xp), intrinsics.unaligned_load(dp)))
				}
			}
		case .Store_Zero:
			if exec == ALL_LANES {
				for i in 0 ..< i32(n) {
					regs[dst + i] = {}
				}
				break
			}
			if exec == {} {
				break
			}
			ebits := u64(transmute(Lane_Bits)exec)
			km: [LANES / 8]Simd_U8
			#unroll for k in 0 ..< LANES / 8 {
				km[k] = intrinsics.simd_lanes_eq(Simd_U8(u32(u8(ebits >> uint(k * 8)))) & LANE_BIT, Simd_U8(0))
			}
			for i in 0 ..< i32(n) {
				d := &regs[dst + i]
				#unroll for k in 0 ..< LANES / 8 {
					dp := (^Simd_U8)(&d[k * 8])
					intrinsics.unaligned_store(dp, intrinsics.unaligned_load(dp) & km[k])
				}
			}
		case .Load_Dyn:
			off0 := regs[b][0]
			uniform := true
			for l in 1 ..< LANES {
				if regs[b][l] != off0 {
					uniform = false
					break
				}
			}
			if uniform {
				off := i32(off0)
				for i in 0 ..< i32(n) {
					regs[dst + i] = regs[a + off + i]
				}
				break
			}
			for l in 0 ..< LANES {
				off := i32(regs[b][l])
				for i in 0 ..< i32(n) {
					regs[dst + i][l] = regs[a + off + i][l]
				}
			}
		case .Store_Dyn:
			if exec == {} {
				break
			}
			if exec == ALL_LANES {
				off0 := regs[b][0]
				uniform := true
				for l in 1 ..< LANES {
					if regs[b][l] != off0 {
						uniform = false
						break
					}
				}
				if uniform {
					off := i32(off0)
					for i in 0 ..< i32(n) {
						regs[dst + off + i] = regs[a + i]
					}
					break
				}
			}
			for l in 0 ..< LANES {
				if l not_in exec {
					continue
				}
				off := i32(regs[b][l])
				for i in 0 ..< i32(n) {
					regs[dst + off + i][l] = regs[a + i][l]
				}
			}
		// Whole-register array ops: no per-lane branches, vectorized.
		// A scalar operand (k) has stride 0.
		case .F_Add:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ + fl(&regs[b + i * sb])^
			}
		case .F_Sub:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ - fl(&regs[b + i * sb])^
			}
		case .F_Mul:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ * fl(&regs[b + i * sb])^
			}
		case .F_Div:
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i * sa])^ / fl(&regs[b + i * sb])^
			}
		case .F_Neg:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = regs[a + i][l] ~ 0x8000_0000
				}
			}
		// Integer, comparison, logic and conversion ops: one loop each, the
		// operation picked at compile time (bin_lanes, un_lanes).
		case .I_Add:
			bin_lanes(regs, in_, .I_Add)
		case .I_Sub:
			bin_lanes(regs, in_, .I_Sub)
		case .I_Mul:
			bin_lanes(regs, in_, .I_Mul)
		case .I_Div:
			bin_lanes(regs, in_, .I_Div)
		case .U_Div:
			bin_lanes(regs, in_, .U_Div)
		case .I_Rem:
			bin_lanes(regs, in_, .I_Rem)
		case .U_Rem:
			bin_lanes(regs, in_, .U_Rem)
		case .Bit_And:
			bin_lanes(regs, in_, .Bit_And)
		case .Bit_Or:
			bin_lanes(regs, in_, .Bit_Or)
		case .Bit_Xor:
			bin_lanes(regs, in_, .Bit_Xor)
		case .Shl:
			bin_lanes(regs, in_, .Shl)
		case .Shr_I:
			bin_lanes(regs, in_, .Shr_I)
		case .Shr_U:
			bin_lanes(regs, in_, .Shr_U)
		case .I_Neg:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = 0 - regs[a + i][l]
				}
			}
		case .Bit_Not:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = ~regs[a + i][l]
				}
			}
		case .F_Lt:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] < y[l])}
			}
		case .F_Le:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] <= y[l])}
			}
		case .F_Gt:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] > y[l])}
			}
		case .F_Ge:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] >= y[l])}
			}
		case .F_Eq:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(x[l] == y[l])}
			}
		case .F_Ne:
			for i in 0 ..< i32(n) {
				x, y, d := fl(&regs[a + i]), fl(&regs[b + i]), &regs[dst + i]
				for l in 0 ..< LANES {d[l] = u32(!(x[l] == y[l]))}
			}
		case .I_Lt:
			bin_lanes(regs, in_, .I_Lt)
		case .I_Le:
			bin_lanes(regs, in_, .I_Le)
		case .I_Gt:
			bin_lanes(regs, in_, .I_Gt)
		case .I_Ge:
			bin_lanes(regs, in_, .I_Ge)
		case .U_Lt:
			bin_lanes(regs, in_, .U_Lt)
		case .U_Le:
			bin_lanes(regs, in_, .U_Le)
		case .U_Gt:
			bin_lanes(regs, in_, .U_Gt)
		case .U_Ge:
			bin_lanes(regs, in_, .U_Ge)
		case .I_Eq:
			bin_lanes(regs, in_, .I_Eq)
		case .I_Ne:
			bin_lanes(regs, in_, .I_Ne)
		case .B_And:
			bin_lanes(regs, in_, .B_And)
		case .B_Or:
			bin_lanes(regs, in_, .B_Or)
		case .B_Not:
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] = u32(regs[a + i][l] == 0)
				}
			}
		case .F_To_I:
			un_lanes(regs, in_, .F_To_I)
		case .F_To_U:
			un_lanes(regs, in_, .F_To_U)
		case .I_To_F:
			un_lanes(regs, in_, .I_To_F)
		case .U_To_F:
			un_lanes(regs, in_, .U_To_F)
		case .B_To_F:
			un_lanes(regs, in_, .B_To_F)
		case .F_To_B:
			un_lanes(regs, in_, .F_To_B)
		case .I_To_B:
			un_lanes(regs, in_, .I_To_B)
		case .Select:
			for i in 0 ..< i32(n) {
				ra, rb, rd := &regs[a + i], &regs[b + i], &regs[dst + i]
				rc := &regs[c] if in_.k == 1 else &regs[c + i]
				for l in 0 ..< LANES {
					m := u32(0) - u32(rc[l] != 0)
					rd[l] = (ra[l] & m) | (rb[l] & ~m)
				}
			}
		case .All, .Any:
			for l in 0 ..< LANES {
				all, any := true, false
				for i in 0 ..< i32(n) {
					v := regs[a + i][l] != 0
					all &&= v
					any ||= v
				}
				regs[dst][l] = u32(all) if in_.op == .All else u32(any)
			}
		case .F_Fn1:
			float_fn1(regs, in_)
		case .F_Fn2:
			float_fn2(regs, in_)
		case .F_Fn3:
			float_fn3(regs, in_)
		case .I_Fn1:
			bi := Builtin(in_.k)
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					x := i32(regs[a + i][l])
					v := x
					if bi == .Abs {
						v = -x if x < 0 else x
					} else {
						v = 1 if x > 0 else (-1 if x < 0 else 0)
					}
					regs[dst + i][l] = u32(v)
				}
			}
		case .I_Fn2, .U_Fn2:
			bi := Builtin(in_.k)
			signed := in_.op == .I_Fn2
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					x, y := regs[a + i][l], regs[b + i][l]
					less := i32(y) < i32(x) if signed else y < x
					regs[dst + i][l] = (y if less else x) if bi == .Min else (x if less else y)
				}
			}
		case .I_Fn3, .U_Fn3:
			signed := in_.op == .I_Fn3
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					x, lo, hi := regs[a + i][l], regs[b + i][l], regs[c + i][l]
					if signed {
						regs[dst + i][l] = u32(min(max(i32(x), i32(lo)), i32(hi)))
					} else {
						regs[dst + i][l] = min(max(x, lo), hi)
					}
				}
			}
		case .Dot:
			sum := fl(&regs[a])^ * fl(&regs[b])^
			for i in 1 ..< i32(n) {
				sum += fl(&regs[a + i])^ * fl(&regs[b + i])^
			}
			fl(&regs[dst])^ = sum
		case .Length:
			x0 := fl(&regs[a])^
			sum := x0 * x0
			for i in 1 ..< i32(n) {
				x := fl(&regs[a + i])^
				sum += x * x
			}
			for l in 0 ..< LANES {
				sum[l] = math.sqrt(sum[l])
			}
			fl(&regs[dst])^ = sum
		case .Distance:
			d0 := fl(&regs[a])^ - fl(&regs[b])^
			sum := d0 * d0
			for i in 1 ..< i32(n) {
				d := fl(&regs[a + i])^ - fl(&regs[b + i])^
				sum += d * d
			}
			for l in 0 ..< LANES {
				sum[l] = math.sqrt(sum[l])
			}
			fl(&regs[dst])^ = sum
		case .Normalize:
			x0 := fl(&regs[a])^
			sum := x0 * x0
			for i in 1 ..< i32(n) {
				x := fl(&regs[a + i])^
				sum += x * x
			}
			for l in 0 ..< LANES {
				sum[l] = 1 / math.sqrt(sum[l])
			}
			for i in 0 ..< i32(n) {
				fl(&regs[dst + i])^ = fl(&regs[a + i])^ * sum
			}
		case .Cross:
			x0, x1, x2 := fl(&regs[a])^, fl(&regs[a + 1])^, fl(&regs[a + 2])^
			y0, y1, y2 := fl(&regs[b])^, fl(&regs[b + 1])^, fl(&regs[b + 2])^
			fl(&regs[dst])^ = x1 * y2 - y1 * x2
			fl(&regs[dst + 1])^ = x2 * y0 - y2 * x0
			fl(&regs[dst + 2])^ = x0 * y1 - y0 * x1
		case .Faceforward, .Reflect, .Refract:
			// N/I in a, I/N in b (GLSL argument order); faceforward's
			// dot is Nref's (c) with I.
			x := c if in_.op == .Faceforward else a
			d := fl(&regs[x])^ * fl(&regs[b])^
			for i in 1 ..< i32(n) {
				d += fl(&regs[x + i])^ * fl(&regs[b + i])^
			}
			#partial switch in_.op {
			case .Faceforward:
				for i in 0 ..< i32(n) {
					v := fl(&regs[a + i])^
					for l in 0 ..< LANES {
						v[l] = v[l] if d[l] < 0 else -v[l]
					}
					fl(&regs[dst + i])^ = v
				}
			case .Reflect:
				for i in 0 ..< i32(n) {
					fl(&regs[dst + i])^ = fl(&regs[a + i])^ - 2 * d * fl(&regs[b + i])^
				}
			case:
				eta := fl(&regs[c])^
				k := 1 - eta * eta * (1 - d * d)
				s: F_Lanes
				for l in 0 ..< LANES {
					s[l] = eta[l] * d[l] + math.sqrt(k[l])
				}
				for i in 0 ..< i32(n) {
					v := eta * fl(&regs[a + i])^ - s * fl(&regs[b + i])^
					for l in 0 ..< LANES {
						v[l] = v[l] if k[l] >= 0 else 0
					}
					fl(&regs[dst + i])^ = v
				}
			}
		case .Mat_Mul:
			rows, inner, cols := n, int(in_.k), int(in_.d)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					sum := fl(&regs[a + i32(row)])^ * fl(&regs[b + i32(col * inner)])^
					for k in 1 ..< inner {
						sum += fl(&regs[a + i32(k * rows + row)])^ * fl(&regs[b + i32(col * inner + k)])^
					}
					fl(&regs[dst + i32(col * rows + row)])^ = sum
				}
			}
		case .Vec_Mat:
			rows, cols := n, int(in_.k)
			for col in 0 ..< cols {
				sum := fl(&regs[a])^ * fl(&regs[b + i32(col * rows)])^
				for row in 1 ..< rows {
					sum += fl(&regs[a + i32(row)])^ * fl(&regs[b + i32(col * rows + row)])^
				}
				fl(&regs[dst + i32(col)])^ = sum
			}
		case .Transpose:
			rows, cols := n, int(in_.k)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					regs[dst + i32(row * cols + col)] = regs[a + i32(col * rows + row)]
				}
			}
		case .Outer:
			rows, cols := n, int(in_.k)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					fl(&regs[dst + i32(col * rows + row)])^ = fl(&regs[a + i32(row)])^ * fl(&regs[b + i32(col)])^
				}
			}
		case .Determinant, .Inverse:
			switch in_.k {
			case 2:
				lane_matrices(regs, in_, 2)
			case 3:
				lane_matrices(regs, in_, 3)
			case:
				lane_matrices(regs, in_, 4)
			}
		case .Deriv_X, .Deriv_Y, .Fwidth:
			for i in 0 ..< i32(n) {
				src := regs[a + i]
				for q in 0 ..< LANES / 4 {
					o := q * 4
					v00, v10, v01, v11 := fval(src[o]), fval(src[o + 1]), fval(src[o + 2]), fval(src[o + 3])
					dx0, dx1 := v10 - v00, v11 - v01
					dy0, dy1 := v01 - v00, v11 - v10
					res: [4]f32
					#partial switch in_.op {
					case .Deriv_X:
						res = {dx0, dx0, dx1, dx1}
					case .Deriv_Y:
						res = {dy0, dy1, dy0, dy1}
					case:
						res = {abs(dx0) + abs(dy0), abs(dx0) + abs(dy1), abs(dx1) + abs(dy0), abs(dx1) + abs(dy1)}
					}
					for k in 0 ..< 4 {
						regs[dst + i][o + k] = fbits(res[k])
					}
				}
			}
		case .Texture, .Texture_Bias, .Texel_Fetch:
			sample_texture(r, in_, exec)
		case .Load_Buf:
			// The word index in 32 bits, wrapping as SPIR-V's and Metal's
			// do; past the host's words, 0.
			words := r.opts.buffers[in_.k] if int(in_.k) < len(r.opts.buffers) else nil
			idx := &regs[in_.a]
			for i in 0 ..< i32(in_.n) {
				d := &regs[in_.dst + i]
				for l in 0 ..< LANES {
					w := idx[l] * u32(in_.b) + u32(in_.c + i)
					d[l] = words[w] if uint(w) < uint(len(words)) else 0
				}
			}
		case .Store_Buf:
			if exec == {} || int(in_.k) >= len(r.opts.buffers) {
				break
			}
			words := r.opts.buffers[in_.k]
			idx := &regs[in_.a]
			ebits := transmute(Lane_Bits)exec
			for l in 0 ..< LANES {
				if (ebits >> uint(l)) & 1 == 0 {
					continue
				}
				base_w := idx[l] * u32(in_.c) + u32(in_.d)
				for i in 0 ..< i32(n) {
					w := base_w + u32(i)
					if uint(w) < uint(len(words)) {
						words[w] = regs[b + i][l]
					}
				}
			}
		case .Image_Load:
			load_image(r, in_, exec)
		case .Image_Store:
			store_image(r, in_, exec)
		case .Image_Size:
			w, h: i32
			if int(in_.k) < len(r.opts.images) {
				im := r.opts.images[in_.k]
				w, h = i32(max(im.width, 0)), i32(max(im.height, 0))
			}
			regs[dst] = u32(w)
			regs[dst + 1] = u32(h)
		case .Texture_Size:
			ch := int(in_.k)
			if r.chan_imgs[ch].ptr != nil {
				im := &r.chan_imgs[ch]
				regs[dst] = u32(im.w)
				regs[dst + 1] = u32(im.max_y + 1)
				break
			}
			size := channel_size(r, ch)
			for l in 0 ..< LANES {
				// Negative lods shift everything out.
				lod := regs[a][l]
				for i in 0 ..< i32(2) {
					regs[dst + i][l] = u32(max(size[i] >> lod, 1) if size[i] > 0 else 0)
				}
				if in_.n == 3 {
					regs[dst + 2][l] = u32(size.z)
				}
			}
		case .Mask_Copy:
			masks[dst] = masks[a]
		case .Mask_Cond:
			lanes: Lane_Mask
			for l in 0 ..< LANES {
				if regs[b][l] != 0 {
					lanes += {l}
				}
			}
			masks[dst] = masks[a] - lanes if in_.k == 1 else masks[a] & lanes
		case .Mask_Restore:
			kill := masks[MASK_DISCARD]
			if b >= 0 {kill += masks[b]}
			if c >= 0 {kill += masks[c]}
			if in_.d >= 0 {kill += masks[in_.d]}
			masks[dst] = masks[a] - kill
		case .Mask_Or:
			masks[dst] = masks[a] + masks[b]
		case .Mask_Clear:
			masks[dst] = {}
		case .Mask_Kill:
			masks[dst] += exec
			masks[MASK_EXEC] = {}
			if dst == MASK_DISCARD && r.lanes <= masks[MASK_DISCARD] {
				return
			}
		case .Jump:
			pc = int(a)
		case .Jump_If_None:
			if exec == {} {
				pc = int(a)
			}
		}
	}
}

@(private)
MAX_TEXEL_F32 :: f32(2147483520.0) // 0x4effffff, largest exact f32 < 2^31

@(private)
coord_nearest :: #force_inline proc "contextless" (u, size_f: f32, max_i: i32) -> i32 {
	p := u * size_f
	p = p if p > 0 else 0
	p = p if p < MAX_TEXEL_F32 else MAX_TEXEL_F32
	i := i32(p)
	return i if i < max_i else max_i
}

@(private)
coord_linear :: #force_inline proc "contextless" (u, size_f: f32, max_i: i32) -> (i0, i1: i32, f: f32) {
	p := u * size_f - 0.5
	fl := ffloor(p)
	df := p - fl
	f = df if df > 0 else 0
	fl_c := fl if fl > -1 else -1
	fl_c = fl_c if fl_c < MAX_TEXEL_F32 else MAX_TEXEL_F32
	iv := i32(fl_c)
	c0 := iv if iv > 0 else 0
	i0 = c0 if c0 < max_i else max_i
	c1 := iv + 1
	i1 = c1 if c1 < max_i else max_i
	return
}

@(private)
sample_bound_nearest :: #force_inline proc "contextless" (im: ^Chan_Img, regs: []Reg, a, dst: i32, $C: int) #no_bounds_check {
	u, v := fl(&regs[a]), fl(&regs[a + 1])
	d0 := fl(&regs[dst])
	when C >= 2 {d1 := fl(&regs[dst + 1])}
	when C == 4 {d2, d3 := fl(&regs[dst + 2]), fl(&regs[dst + 3])}
	fw, fh, w, max_x, max_y, ptr := im.fw, im.fh, im.w, im.max_x, im.max_y, im.ptr
	for l in 0 ..< LANES {
		x := coord_nearest(u[l], fw, max_x)
		y := coord_nearest(v[l], fh, max_y)
		off := int(y * w + x) * C
		d0[l] = ptr[off]
		when C >= 2 {d1[l] = ptr[off + 1]}
		when C == 4 {
			d2[l] = ptr[off + 2]
			d3[l] = ptr[off + 3]
		}
	}
	when C == 1 {
		regs[dst + 1] = 0
		regs[dst + 2] = 0
		regs[dst + 3] = 0x3f800000
	} else when C == 2 {
		regs[dst + 2] = 0
		regs[dst + 3] = 0x3f800000
	}
}

@(private)
sample_bound_linear :: #force_inline proc "contextless" (im: ^Chan_Img, regs: []Reg, a, dst: i32, $C: int) #no_bounds_check {
	u, v := fl(&regs[a]), fl(&regs[a + 1])
	d0 := fl(&regs[dst])
	when C >= 2 {d1 := fl(&regs[dst + 1])}
	when C == 4 {d2, d3 := fl(&regs[dst + 2]), fl(&regs[dst + 3])}
	fw, fh, w, max_x, max_y, ptr := im.fw, im.fh, im.w, im.max_x, im.max_y, im.ptr
	for l in 0 ..< LANES {
		x0, x1, fx := coord_linear(u[l], fw, max_x)
		y0, y1, fy := coord_linear(v[l], fh, max_y)
		r0, r1 := y0 * w, y1 * w
		o00, o10 := int(r0 + x0) * C, int(r0 + x1) * C
		o01, o11 := int(r1 + x0) * C, int(r1 + x1) * C
		when C == 1 {
			top := ptr[o00] + (ptr[o10] - ptr[o00]) * fx
			bot := ptr[o01] + (ptr[o11] - ptr[o01]) * fx
			d0[l] = top + (bot - top) * fy
		} else when C == 2 {
			c00, c10 := ((^[2]f32)(&ptr[o00]))^, ((^[2]f32)(&ptr[o10]))^
			c01, c11 := ((^[2]f32)(&ptr[o01]))^, ((^[2]f32)(&ptr[o11]))^
			top := c00 + (c10 - c00) * fx
			bot := c01 + (c11 - c01) * fx
			res := top + (bot - top) * fy
			d0[l], d1[l] = res[0], res[1]
		} else {
			c00, c10 := ((^[4]f32)(&ptr[o00]))^, ((^[4]f32)(&ptr[o10]))^
			c01, c11 := ((^[4]f32)(&ptr[o01]))^, ((^[4]f32)(&ptr[o11]))^
			top := c00 + (c10 - c00) * fx
			bot := c01 + (c11 - c01) * fx
			res := top + (bot - top) * fy
			d0[l], d1[l], d2[l], d3[l] = res[0], res[1], res[2], res[3]
		}
	}
	when C == 1 {
		regs[dst + 1] = 0
		regs[dst + 2] = 0
		regs[dst + 3] = 0x3f800000
	} else when C == 2 {
		regs[dst + 2] = 0
		regs[dst + 3] = 0x3f800000
	}
}

@(private)
fetch_bound_lanes :: #force_inline proc "contextless" (im: ^Chan_Img, regs: []Reg, a, dst: i32, flip: bool, $C: int) #no_bounds_check {
	px, py := &regs[a], &regs[a + 1]
	d0, d1, d2, d3 := fl(&regs[dst]), fl(&regs[dst + 1]), fl(&regs[dst + 2]), fl(&regs[dst + 3])
	w, max_x, max_y, ptr := im.w, im.max_x, im.max_y, im.ptr
	for l in 0 ..< LANES {
		x := i32(px[l])
		y := i32(u32(max_y) - py[l]) if flip else i32(py[l])
		if u32(x) > u32(max_x) || u32(y) > u32(max_y) {
			d0[l], d1[l], d2[l], d3[l] = 0, 0, 0, 0
			continue
		}
		off := int(y * w + x) * C
		when C == 1 {
			d0[l], d1[l], d2[l], d3[l] = ptr[off], 0, 0, 1
		} else when C == 2 {
			d0[l], d1[l], d2[l], d3[l] = ptr[off], ptr[off + 1], 0, 1
		} else {
			t := ((^[4]f32)(&ptr[off]))^
			d0[l], d1[l], d2[l], d3[l] = t[0], t[1], t[2], t[3]
		}
	}
}

@(private)
texel_fetch_mip :: #force_inline proc "contextless" (size: [3]i32, lod_bits: u32) -> (sx, sy: f32) {
	return f32(max(size.x >> lod_bits, 1) if size.x > 0 else 0), f32(max(size.y >> lod_bits, 1) if size.y > 0 else 0)
}

@(private)
texel_fetch_uv :: #force_inline proc "contextless" (px, py: i32, sx, sy: f32, flags: i32) -> [2]f32 {
	ux := (f32(px) + 0.5) / sx
	uy := (f32(py) + 0.5) / sy
	if (flags & 1) != 0 {
		uy = uy * -1 + 1
	}
	return {ux, uy}
}

// A sample: dst = channel k at uv a, lod b (Texture_Bias: b plus the
// level the uv derivatives pick in the channel's texels, one per quad from
// its top row and left column, as GPUs' texture units pick it). Quads
// with a running lane (`exec`) are sampled; the others read 0.
@(private)
sample_texture :: proc(r: ^Runner, in_: Instr, exec: Lane_Mask) #no_bounds_check {
	regs := r.regs
	ch := int(in_.k)
	dst := in_.dst
	if r.chan_imgs[ch].ptr != nil {
		im := &r.chan_imgs[ch]
		if in_.op == .Texel_Fetch {
			flip := (in_.d & 1) != 0
			switch im.comps {
			case 1: fetch_bound_lanes(im, regs, in_.a, dst, flip, 1)
			case 2: fetch_bound_lanes(im, regs, in_.a, dst, flip, 2)
			case:   fetch_bound_lanes(im, regs, in_.a, dst, flip, 4)
			}
		} else if im.filter == .Nearest {
			switch im.comps {
			case 1: sample_bound_nearest(im, regs, in_.a, dst, 1)
			case 2: sample_bound_nearest(im, regs, in_.a, dst, 2)
			case:   sample_bound_nearest(im, regs, in_.a, dst, 4)
			}
		} else {
			switch im.comps {
			case 1: sample_bound_linear(im, regs, in_.a, dst, 1)
			case 2: sample_bound_linear(im, regs, in_.a, dst, 2)
			case:   sample_bound_linear(im, regs, in_.a, dst, 4)
			}
		}
		return
	}
	for i in 0 ..< i32(4) {
		regs[dst + i] = {}
	}
	if r.opts.sampler == nil {
		return
	}
	size := channel_size(r, ch)
	running, have := transmute(Lane_Bits)exec, transmute(Lane_Bits)r.lanes
	if in_.op == .Texel_Fetch {
		px, py, lb := &regs[in_.a], &regs[in_.a + 1], &regs[in_.b]
		uint_lod := (in_.d & 4) != 0
		if r.p.stage == .Compute {
			for l in 0 ..< LANES {
				if (running >> uint(l)) & 1 == 0 {
					continue
				}
				sx, sy := texel_fetch_mip(size, lb[l])
				flod := f32(lb[l]) if uint_lod else f32(i32(lb[l]))
				c := r.opts.sampler(r.opts.user, ch, texel_fetch_uv(i32(px[l]), i32(py[l]), sx, sy, 0), flod, 0)
				for i in 0 ..< i32(4) {
					regs[dst + i][l] = fbits(c[i])
				}
			}
			return
		}
		hdr := (in_.d & 2) != 0
		for o := 0; o < LANES; o += 4 {
			if (running >> uint(o)) & 0xf == 0 {
				continue
			}
			for l in o ..< o + 4 {
				if (have >> uint(l)) & 1 == 0 {
					continue
				}
				sx, sy := texel_fetch_mip(size, lb[l])
				if hdr {
					sx, sy = fval(regs[in_.c][l]), fval(regs[in_.c + 1][l])
				}
				flod := f32(lb[l]) if uint_lod else f32(i32(lb[l]))
				c := r.opts.sampler(r.opts.user, ch, texel_fetch_uv(i32(px[l]), i32(py[l]), sx, sy, in_.d), flod, 0)
				for i in 0 ..< i32(4) {
					regs[dst + i][l] = fbits(c[i])
				}
			}
		}
		return
	}
	a, b := in_.a, in_.b
	lod := fl(&regs[b])
	u, v := fl(&regs[a]), fl(&regs[a + 1])
	if r.p.stage == .Compute {
		if in_.n == 3 {
			z := &regs[a + 2]
			for l in 0 ..< LANES {
				if (running >> uint(l)) & 1 == 0 {
					continue
				}
				layer := int(f_to_i(fval(fn1(fval(z[l]), .Round_Even))))
				c := r.opts.sampler(r.opts.user, ch, {u[l], v[l]}, lod[l], layer)
				for i in 0 ..< i32(4) {
					regs[dst + i][l] = fbits(c[i])
				}
			}
		} else {
			for l in 0 ..< LANES {
				if (running >> uint(l)) & 1 == 0 {
					continue
				}
				c := r.opts.sampler(r.opts.user, ch, {u[l], v[l]}, lod[l], 0)
				for i in 0 ..< i32(4) {
					regs[dst + i][l] = fbits(c[i])
				}
			}
		}
		return
	}
	for o := 0; o < LANES; o += 4 {
		// A quad with a running lane samples all four lanes the batch has,
		// as a GPU quad's helpers do, so derivatives of the result are the
		// quad's like any temporary's; an idle quad samples nothing.
		if (running >> uint(o)) & 0xf == 0 {
			continue
		}
		level: f32 = 0
		if in_.op == .Texture_Bias && size.xy != {} {
			w, h := f32(size.x), f32(size.y)
			dx := [2]f32{(u[o + 1] - u[o]) * w, (v[o + 1] - v[o]) * h}
			dy := [2]f32{(u[o + 2] - u[o]) * w, (v[o + 2] - v[o]) * h}
			level = 0.5 * math.log2(max(dx.x * dx.x + dx.y * dx.y, dy.x * dy.x + dy.y * dy.y))
		}
		for l in o ..< o + 4 {
			if (have >> uint(l)) & 1 == 0 {
				continue
			}
			// The layer as GPUs take it, rounded to nearest even (NaN and
			// out-of-range values as F_To_I takes them).
			layer := int(f_to_i(fval(fn1(fval(regs[a + 2][l]), .Round_Even)))) if in_.n == 3 else 0
			c := r.opts.sampler(r.opts.user, ch, {u[l], v[l]}, lod[l] + level, layer)
			for i in 0 ..< i32(4) {
				regs[dst + i][l] = fbits(c[i])
			}
		}
	}
}

// Channel ch's size from the host (0x0 without one).
@(private)
channel_size :: #force_inline proc "contextless" (r: ^Runner, ch: int) -> [3]i32 {
	return r.opts.channel_sizes[ch] if ch >= 0 && ch < len(r.opts.channel_sizes) else {}
}

// The determinant or the inverse of each lane's N x N matrix, read from
// column-major registers at `a`.
@(private)
lane_matrices :: proc "contextless" (regs: []Reg, in_: Instr, $N: int) #no_bounds_check {
	for l in 0 ..< LANES {
		m: matrix[N, N]f32
		for c in 0 ..< N {
			for r in 0 ..< N {
				m[r, c] = fval(regs[in_.a + i32(c * N + r)][l])
			}
		}
		if in_.op == .Determinant {
			regs[in_.dst][l] = fbits(linalg.determinant(m))
			continue
		}
		inv := linalg.inverse(m)
		for c in 0 ..< N {
			for r in 0 ..< N {
				regs[in_.dst + i32(c * N + r)][l] = fbits(inv[r, c])
			}
		}
	}
}

@(private)
image_texel_bytes :: #force_inline proc "contextless" (f: Image_Format) -> int {
	#partial switch f {
	case .Rgba8, .R32f, .R32ui, .R32i:
		return 4
	case .Rgba16f:
		return 8
	case .Rgba32f:
		return 16
	}
	return 0
}

// IEEE-754 binary32 -> binary16, round-to-nearest-even.
@(private)
f32_to_f16 :: proc "contextless" (x: f32) -> u16 {
	b := transmute(u32)x
	sign := u16((b >> 16) & 0x8000)
	abs_b := b & 0x7fff_ffff
	if abs_b > 0x7f80_0000 {
		return sign | 0x7e00
	}
	if abs_b >= 0x4780_0000 {
		return sign | 0x7c00
	}
	exp32 := int(abs_b >> 23)
	mant32 := abs_b & 0x7f_ffff
	exp16 := exp32 - 112
	if exp16 >= 1 {
		lsb := (mant32 >> 13) & 1
		rounded := (mant32 + 0xfff + lsb) >> 13
		return sign + (u16(exp16) << 10) + u16(rounded)
	}
	if exp16 < -10 {
		return sign
	}
	m := mant32 | 0x80_0000
	shift := uint(14 - exp16)
	lsb := (m >> shift) & 1
	half_minus_1 := (u32(1) << (shift - 1)) - 1
	return sign | u16((m + half_minus_1 + lsb) >> shift)
}

@(private)
f16_to_f32 :: proc "contextless" (h: u16) -> f32 {
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
		if mant == 0 {
			return transmute(f32)(sign | 0x7f80_0000)
		}
		return transmute(f32)(sign | 0x7fc0_0000 | (mant << 13))
	}
	return transmute(f32)(sign | ((exp + 112) << 23) | (mant << 13))
}

// Exact IEEE-754 single-precision `f32(b) / 255.0` for every byte `0..=255`.
@(private, rodata)
UNORM8_TO_F32 := [256]f32 {
	0.0 / 255.0, 1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 4.0 / 255.0, 5.0 / 255.0, 6.0 / 255.0, 7.0 / 255.0,
	8.0 / 255.0, 9.0 / 255.0, 10.0 / 255.0, 11.0 / 255.0, 12.0 / 255.0, 13.0 / 255.0, 14.0 / 255.0, 15.0 / 255.0,
	16.0 / 255.0, 17.0 / 255.0, 18.0 / 255.0, 19.0 / 255.0, 20.0 / 255.0, 21.0 / 255.0, 22.0 / 255.0, 23.0 / 255.0,
	24.0 / 255.0, 25.0 / 255.0, 26.0 / 255.0, 27.0 / 255.0, 28.0 / 255.0, 29.0 / 255.0, 30.0 / 255.0, 31.0 / 255.0,
	32.0 / 255.0, 33.0 / 255.0, 34.0 / 255.0, 35.0 / 255.0, 36.0 / 255.0, 37.0 / 255.0, 38.0 / 255.0, 39.0 / 255.0,
	40.0 / 255.0, 41.0 / 255.0, 42.0 / 255.0, 43.0 / 255.0, 44.0 / 255.0, 45.0 / 255.0, 46.0 / 255.0, 47.0 / 255.0,
	48.0 / 255.0, 49.0 / 255.0, 50.0 / 255.0, 51.0 / 255.0, 52.0 / 255.0, 53.0 / 255.0, 54.0 / 255.0, 55.0 / 255.0,
	56.0 / 255.0, 57.0 / 255.0, 58.0 / 255.0, 59.0 / 255.0, 60.0 / 255.0, 61.0 / 255.0, 62.0 / 255.0, 63.0 / 255.0,
	64.0 / 255.0, 65.0 / 255.0, 66.0 / 255.0, 67.0 / 255.0, 68.0 / 255.0, 69.0 / 255.0, 70.0 / 255.0, 71.0 / 255.0,
	72.0 / 255.0, 73.0 / 255.0, 74.0 / 255.0, 75.0 / 255.0, 76.0 / 255.0, 77.0 / 255.0, 78.0 / 255.0, 79.0 / 255.0,
	80.0 / 255.0, 81.0 / 255.0, 82.0 / 255.0, 83.0 / 255.0, 84.0 / 255.0, 85.0 / 255.0, 86.0 / 255.0, 87.0 / 255.0,
	88.0 / 255.0, 89.0 / 255.0, 90.0 / 255.0, 91.0 / 255.0, 92.0 / 255.0, 93.0 / 255.0, 94.0 / 255.0, 95.0 / 255.0,
	96.0 / 255.0, 97.0 / 255.0, 98.0 / 255.0, 99.0 / 255.0, 100.0 / 255.0, 101.0 / 255.0, 102.0 / 255.0, 103.0 / 255.0,
	104.0 / 255.0, 105.0 / 255.0, 106.0 / 255.0, 107.0 / 255.0, 108.0 / 255.0, 109.0 / 255.0, 110.0 / 255.0, 111.0 / 255.0,
	112.0 / 255.0, 113.0 / 255.0, 114.0 / 255.0, 115.0 / 255.0, 116.0 / 255.0, 117.0 / 255.0, 118.0 / 255.0, 119.0 / 255.0,
	120.0 / 255.0, 121.0 / 255.0, 122.0 / 255.0, 123.0 / 255.0, 124.0 / 255.0, 125.0 / 255.0, 126.0 / 255.0, 127.0 / 255.0,
	128.0 / 255.0, 129.0 / 255.0, 130.0 / 255.0, 131.0 / 255.0, 132.0 / 255.0, 133.0 / 255.0, 134.0 / 255.0, 135.0 / 255.0,
	136.0 / 255.0, 137.0 / 255.0, 138.0 / 255.0, 139.0 / 255.0, 140.0 / 255.0, 141.0 / 255.0, 142.0 / 255.0, 143.0 / 255.0,
	144.0 / 255.0, 145.0 / 255.0, 146.0 / 255.0, 147.0 / 255.0, 148.0 / 255.0, 149.0 / 255.0, 150.0 / 255.0, 151.0 / 255.0,
	152.0 / 255.0, 153.0 / 255.0, 154.0 / 255.0, 155.0 / 255.0, 156.0 / 255.0, 157.0 / 255.0, 158.0 / 255.0, 159.0 / 255.0,
	160.0 / 255.0, 161.0 / 255.0, 162.0 / 255.0, 163.0 / 255.0, 164.0 / 255.0, 165.0 / 255.0, 166.0 / 255.0, 167.0 / 255.0,
	168.0 / 255.0, 169.0 / 255.0, 170.0 / 255.0, 171.0 / 255.0, 172.0 / 255.0, 173.0 / 255.0, 174.0 / 255.0, 175.0 / 255.0,
	176.0 / 255.0, 177.0 / 255.0, 178.0 / 255.0, 179.0 / 255.0, 180.0 / 255.0, 181.0 / 255.0, 182.0 / 255.0, 183.0 / 255.0,
	184.0 / 255.0, 185.0 / 255.0, 186.0 / 255.0, 187.0 / 255.0, 188.0 / 255.0, 189.0 / 255.0, 190.0 / 255.0, 191.0 / 255.0,
	192.0 / 255.0, 193.0 / 255.0, 194.0 / 255.0, 195.0 / 255.0, 196.0 / 255.0, 197.0 / 255.0, 198.0 / 255.0, 199.0 / 255.0,
	200.0 / 255.0, 201.0 / 255.0, 202.0 / 255.0, 203.0 / 255.0, 204.0 / 255.0, 205.0 / 255.0, 206.0 / 255.0, 207.0 / 255.0,
	208.0 / 255.0, 209.0 / 255.0, 210.0 / 255.0, 211.0 / 255.0, 212.0 / 255.0, 213.0 / 255.0, 214.0 / 255.0, 215.0 / 255.0,
	216.0 / 255.0, 217.0 / 255.0, 218.0 / 255.0, 219.0 / 255.0, 220.0 / 255.0, 221.0 / 255.0, 222.0 / 255.0, 223.0 / 255.0,
	224.0 / 255.0, 225.0 / 255.0, 226.0 / 255.0, 227.0 / 255.0, 228.0 / 255.0, 229.0 / 255.0, 230.0 / 255.0, 231.0 / 255.0,
	232.0 / 255.0, 233.0 / 255.0, 234.0 / 255.0, 235.0 / 255.0, 236.0 / 255.0, 237.0 / 255.0, 238.0 / 255.0, 239.0 / 255.0,
	240.0 / 255.0, 241.0 / 255.0, 242.0 / 255.0, 243.0 / 255.0, 244.0 / 255.0, 245.0 / 255.0, 246.0 / 255.0, 247.0 / 255.0,
	248.0 / 255.0, 249.0 / 255.0, 250.0 / 255.0, 251.0 / 255.0, 252.0 / 255.0, 253.0 / 255.0, 254.0 / 255.0, 255.0 / 255.0,
}

// run_compute validates opts.images up front, so width/height/stride/data are
// guaranteed in-bounds for every (x, y) in [0, width) x [0, height).
@(private)
load_image :: proc "contextless" (r: ^Runner, in_: Instr, exec: Lane_Mask) #no_bounds_check {
	regs := r.regs
	dst, a := in_.dst, in_.a
	for i in 0 ..< i32(4) {
		regs[dst + i] = {}
	}
	if exec == {} {
		return
	}
	im := r.opts.images[in_.k]
	if im.width <= 0 || im.height <= 0 {
		return
	}
	w, h := u32(im.width), u32(im.height)
	tb := image_texel_bytes(im.format)
	row_stride := im.stride if im.stride != 0 else im.width * tb
	cx, cy := &regs[a], &regs[a + 1]
	ebits := transmute(Lane_Bits)exec
	data := raw_data(im.data)
	#partial switch im.format {
	case .Rgba8:
		r0, r1, r2, r3 := &regs[dst], &regs[dst + 1], &regs[dst + 2], &regs[dst + 3]
		lut := &UNORM8_TO_F32
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			p := data[int(y) * row_stride + int(x) * 4:]
			r0[l] = fbits(lut[p[0]])
			r1[l] = fbits(lut[p[1]])
			r2[l] = fbits(lut[p[2]])
			r3[l] = fbits(lut[p[3]])
		}
	case .Rgba16f:
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 8
			for i in 0 ..< i32(4) {
				o := off + int(i) * 2
				hv := u16(data[o]) | (u16(data[o + 1]) << 8)
				regs[dst + i][l] = fbits(f16_to_f32(hv))
			}
		}
	case .Rgba32f:
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 16
			for i in 0 ..< i32(4) {
				regs[dst + i][l] = intrinsics.unaligned_load((^u32)(&data[off + int(i) * 4]))
			}
		}
	case .R32f:
		r0, r3 := &regs[dst], &regs[dst + 3]
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 4
			r0[l] = intrinsics.unaligned_load((^u32)(&data[off]))
			r3[l] = 0x3f80_0000
		}
	case .R32ui, .R32i:
		r0, r3 := &regs[dst], &regs[dst + 3]
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 4
			r0[l] = intrinsics.unaligned_load((^u32)(&data[off]))
			r3[l] = 1
		}
	}
}

@(private)
store_image :: proc "contextless" (r: ^Runner, in_: Instr, exec: Lane_Mask) #no_bounds_check {
	if exec == {} {
		return
	}
	im := r.opts.images[in_.k]
	if im.width <= 0 || im.height <= 0 {
		return
	}
	w, h := u32(im.width), u32(im.height)
	tb := image_texel_bytes(im.format)
	row_stride := im.stride if im.stride != 0 else im.width * tb
	regs := r.regs
	a, b := in_.a, in_.b
	cx, cy := &regs[a], &regs[a + 1]
	ebits := transmute(Lane_Bits)exec
	data := raw_data(im.data)
	#partial switch im.format {
	case .Rgba8:
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 4
			for i in 0 ..< i32(4) {
				cv := fval(regs[b + i][l])
				if !(cv > 0) {
					cv = 0
				}
				data[off + int(i)] = u8(math.round(min(cv, 1) * 255))
			}
		}
	case .Rgba16f:
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 8
			for i in 0 ..< i32(4) {
				hv := f32_to_f16(fval(regs[b + i][l]))
				o := off + int(i) * 2
				data[o] = u8(hv)
				data[o + 1] = u8(hv >> 8)
			}
		}
	case .Rgba32f:
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 16
			for i in 0 ..< i32(4) {
				intrinsics.unaligned_store((^u32)(&data[off + int(i) * 4]), regs[b + i][l])
			}
		}
	case .R32f, .R32ui, .R32i:
		b0 := &regs[b]
		for l in 0 ..< LANES {
			if (ebits >> uint(l)) & 1 == 0 {
				continue
			}
			x, y := cx[l], cy[l]
			if x >= w || y >= h {
				continue
			}
			off := int(y) * row_stride + int(x) * 4
			intrinsics.unaligned_store((^u32)(&data[off]), b0[l])
		}
	}
}
