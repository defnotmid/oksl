package oksl

import "core:math"

// CPU backend, part 2: the interpreter. Pixels run in batches of LANES:
// 2x2 quads in a BATCH_W x BATCH_H block, so derivatives use the quad
// neighbours like a GPU's fine derivatives, and a block of neighbouring
// pixels diverges little.

// Samples channel `channel` at normalized `uv` (origin top-left) and mip
// level `lod`; returns RGBA. The level texture() picks is fractional and
// may be below 0 (magnified, -inf where uv is constant): clamp it to the
// levels the channel has.
Sampler_Proc :: #type proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32

Run_Options :: struct {
	sampler:  Sampler_Proc,
	user:     rawptr,
	// Region to shade in layer pixels; zero means the whole layer.
	x, y:     int,
	w, h:     int,
	// The stage inputs by location. Absent ones are zero, except location
	// 0, which is the layer's uv (what the layer vertex shader writes).
	varyings: []Varying,
	// Each channel's size in texels (level 0), for textureSize and
	// texelFetch in GLSL shaders and the level texture() picks in fragment
	// shaders. A channel without one is 0x0, and texture() samples it at
	// its bias.
	channel_sizes: [][2]i32,
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
	p:        ^Program,
	regs:     []Reg,
	masks:    []Lane_Mask,
	uniforms: []u8,
	opts:     Run_Options,
}

// A runner over `scratch` with the program's constants and uniforms loaded.
@(private)
runner_begin :: proc(p: ^Program, scratch: ^Scratch, uniforms: []u8, opts: Run_Options) -> Runner {
	if len(scratch.regs) < max(p.num_slots, 1) {
		resize(&scratch.regs, max(p.num_slots, 1))
	}
	if len(scratch.masks) < max(p.num_masks, 2) {
		resize(&scratch.masks, max(p.num_masks, 2))
	}
	r := Runner{p, scratch.regs[:], scratch.masks[:], uniforms, opts}
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
	r := runner_begin(p, scratch, uniforms, opts)
	x0, y0 := opts.x, opts.y
	w, h := opts.w, opts.h
	if w <= 0 || h <= 0 {
		x0, y0, w, h = 0, 0, width, height
	}
	fw, fh := f32(width), f32(height)
	// Each lane's pixel in the batch.
	lx, ly: [LANES]int
	for lane in 0 ..< LANES {
		q := lane >> 2
		lx[lane] = q % (BATCH_W / 2) * 2 + (lane & 1)
		ly[lane] = q / (BATCH_W / 2) * 2 + ((lane >> 1) & 1)
	}
	for by := y0; by < y0 + h; by += BATCH_H {
		for bx := x0; bx < x0 + w; bx += BATCH_W {
			// Pixel centres of the batch's lanes, then each input a whole
			// register at a time.
			px, py: F_Lanes
			for lane in 0 ..< LANES {
				px[lane] = f32(bx + lx[lane]) + 0.5
				py[lane] = f32(by + ly[lane]) + 0.5
			}
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
			r.masks[MASK_EXEC] = ALL_LANES
			r.masks[MASK_DISCARD] = {}
			execute(&r)
			alive := ~r.masks[MASK_DISCARD]
			for v in p.outputs {
				if v.kind != .Stage_Out {
					continue
				}
				target := targets[v.location] if int(v.location) < len(targets) else nil
				bytes := out if v.location == 0 else nil
				if target == nil && bytes == nil {
					continue
				}
				for lane in 0 ..< LANES {
					x := bx + lx[lane]
					y := by + ly[lane]
					if x >= x0 + w || y >= y0 + h || x >= width || y >= height || lane not_in alive {
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
// or u32.
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
// its outputs at their offsets (Stage_Var_Info.offset), integers as
// floats. Threads may run one program at once, each with its own scratch.
run_vertex :: proc(p: ^Program, uniforms: []u8, attributes: []Attribute, count: int, position: [][4]f32, varyings: []f32, scratch: ^Scratch, opts := Run_Options{}) {
	r := runner_begin(p, scratch, uniforms, opts)
	n := p.varying_floats
	for first := 0; first < count; first += LANES {
		lanes := min(count - first, LANES)
		for v in p.inputs {
			a := attributes[v.location] if int(v.location) < len(attributes) else Attribute{}
			comps := int(a.format) % 4 + 1
			from := FORMAT_BASES[int(a.format) / 4]
			for k in 0 ..< int(v.n) {
				reg := &r.regs[v.slot + i32(k)]
				for l in 0 ..< lanes {
					if a.data == nil || k >= comps {
						reg[l] = convert_bits(1 if k == 3 else 0, .Int, v.base)
						continue
					}
					word := ([^]u32)(rawptr(uintptr(a.data) + uintptr((first + l) * a.stride)))[k]
					reg[l] = convert_bits(word, from, v.base)
				}
			}
		}
		r.masks[MASK_EXEC] = transmute(Lane_Mask)Lane_Bits((1 << uint(lanes)) - 1)
		r.masks[MASK_DISCARD] = {}
		execute(&r)
		for v in p.outputs {
			for l in 0 ..< lanes {
				i := first + l
				if v.kind == .Position {
					for k in 0 ..< 4 {
						position[i][k] = fval(r.regs[v.slot + i32(k)][l])
					}
					continue
				}
				for k in 0 ..< int(v.n) {
					varyings[i * n + int(v.offset) + k] = float_of(r.regs[v.slot + i32(k)][l], v.base)
				}
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
	varyings:     []f32,
}

// Shades quads with a fragment program, four at a time like a GPU's 2x2
// quads (dFdx / dFdy within each). `out[j]` receives lane j's colour at
// location 0 (four per quad), `shaded[q]` the lanes of quad q that
// produced one (not helpers, not discarded). Threads may run one program
// at once, each with its own scratch.
run_fragment :: proc(p: ^Program, uniforms: []u8, quads: Fragment_Quads, out: [][4]f32, shaded: []Quad_Lanes, scratch: ^Scratch, opts := Run_Options{}) {
	r := runner_begin(p, scratch, uniforms, opts)
	total := quads.count * 4
	for first := 0; first < quads.count; first += LANES / 4 {
		nq := min(quads.count - first, LANES / 4)
		lanes := nq * 4
		for v in p.inputs {
			for k in 0 ..< int(v.n) {
				col: F_Lanes
				for l in 0 ..< lanes {
					q := first + l / 4
					switch v.kind {
					case .Frag_Coord:
						o := quads.origin[q]
						switch k {
						case 0:
							col[l] = f32(o.x + i32(l & 1)) + 0.5
						case 1:
							col[l] = f32(o.y + i32((l >> 1) & 1)) + 0.5
						case 3:
							col[l] = 1
						}
					case .Front_Facing:
						col[l] = 1 if quads.front_facing == nil || quads.front_facing[q] else 0
					case .Stage_In:
						col[l] = quads.varyings[(int(v.offset) + k) * total + first * 4 + l]
					case .Stage_Out, .Position, .Uniform_Block, .Channel:
					}
				}
				fill_reg(&r.regs[v.slot + i32(k)], col, v.base)
			}
		}
		r.masks[MASK_EXEC] = transmute(Lane_Mask)Lane_Bits((1 << uint(lanes)) - 1)
		r.masks[MASK_DISCARD] = {}
		execute(&r)
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
				c := [4]f32{0, 0, 0, 1}
				for v in p.outputs {
					if v.kind == .Stage_Out && v.location == 0 {
						for k in 0 ..< int(v.n) {
							c[k] = float_of(r.regs[v.slot + i32(k)][l], v.base)
						}
					}
				}
				out[q * 4 + i] = c
			}
			shaded[q] = lanes_out
		}
	}
}

// floor without core:math's modf: values of 2^23 and up (and inf/NaN) are
// already integral.
@(private)
ffloor :: #force_inline proc "contextless" (x: f32) -> f32 {
	if !(abs(x) < 8388608) {
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
execute :: proc(r: ^Runner) #no_bounds_check {
	code := r.p.code
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
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					keep := u32(0) - u32(l in exec)
					regs[dst + i][l] = (regs[a + i][l] & keep) | (regs[dst + i][l] & ~keep)
				}
			}
		case .Store_Zero:
			if exec == ALL_LANES {
				for i in 0 ..< i32(n) {
					regs[dst + i] = {}
				}
				break
			}
			for i in 0 ..< i32(n) {
				for l in 0 ..< LANES {
					regs[dst + i][l] &= u32(0) - u32(l not_in exec)
				}
			}
		case .Load_Dyn:
			for l in 0 ..< LANES {
				off := i32(regs[b][l])
				for i in 0 ..< i32(n) {
					regs[dst + i][l] = regs[a + off + i][l]
				}
			}
		case .Store_Dyn:
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
				ra, rb, rc, rd := &regs[a + i], &regs[b + i], &regs[c + i], &regs[dst + i]
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
			sum: F_Lanes
			for i in 0 ..< i32(n) {
				sum += fl(&regs[a + i])^ * fl(&regs[b + i])^
			}
			fl(&regs[dst])^ = sum
		case .Length:
			sum: F_Lanes
			for i in 0 ..< i32(n) {
				x := fl(&regs[a + i])^
				sum += x * x
			}
			for l in 0 ..< LANES {
				sum[l] = math.sqrt(sum[l])
			}
			fl(&regs[dst])^ = sum
		case .Distance:
			for l in 0 ..< LANES {
				sum: f32 = 0
				for i in 0 ..< i32(n) {
					d := fval(regs[a + i][l]) - fval(regs[b + i][l])
					sum += d * d
				}
				regs[dst][l] = fbits(math.sqrt(sum))
			}
		case .Normalize:
			for l in 0 ..< LANES {
				sum: f32 = 0
				for i in 0 ..< i32(n) {
					x := fval(regs[a + i][l])
					sum += x * x
				}
				inv := 1 / math.sqrt(sum)
				for i in 0 ..< i32(n) {
					regs[dst + i][l] = fbits(fval(regs[a + i][l]) * inv)
				}
			}
		case .Cross:
			for l in 0 ..< LANES {
				x0, x1, x2 := fval(regs[a][l]), fval(regs[a + 1][l]), fval(regs[a + 2][l])
				y0, y1, y2 := fval(regs[b][l]), fval(regs[b + 1][l]), fval(regs[b + 2][l])
				regs[dst][l] = fbits(x1 * y2 - y1 * x2)
				regs[dst + 1][l] = fbits(x2 * y0 - y2 * x0)
				regs[dst + 2][l] = fbits(x0 * y1 - y0 * x1)
			}
		case .Faceforward, .Reflect, .Refract:
			for l in 0 ..< LANES {
				// N/I in a, I/N in b (GLSL argument order).
				d: f32 = 0
				for i in 0 ..< i32(n) {
					#partial switch in_.op {
					case .Faceforward:
						d += fval(regs[c + i][l]) * fval(regs[b + i][l])
					case:
						d += fval(regs[a + i][l]) * fval(regs[b + i][l])
					}
				}
				#partial switch in_.op {
				case .Faceforward:
					for i in 0 ..< i32(n) {
						x := fval(regs[a + i][l])
						regs[dst + i][l] = fbits(x if d < 0 else -x)
					}
				case .Reflect:
					for i in 0 ..< i32(n) {
						regs[dst + i][l] = fbits(fval(regs[a + i][l]) - 2 * d * fval(regs[b + i][l]))
					}
				case:
					eta := fval(regs[c][l])
					k := 1 - eta * eta * (1 - d * d)
					for i in 0 ..< i32(n) {
						v: f32 = 0
						if k >= 0 {
							v = eta * fval(regs[a + i][l]) - (eta * d + math.sqrt(k)) * fval(regs[b + i][l])
						}
						regs[dst + i][l] = fbits(v)
					}
				}
			}
		case .Mat_Mul:
			rows, inner, cols := n, int(in_.k), int(in_.d)
			for l in 0 ..< LANES {
				for col in 0 ..< cols {
					for row in 0 ..< rows {
						sum: f32 = 0
						for k in 0 ..< inner {
							sum += fval(regs[a + i32(k * rows + row)][l]) * fval(regs[b + i32(col * inner + k)][l])
						}
						regs[dst + i32(col * rows + row)][l] = fbits(sum)
					}
				}
			}
		case .Vec_Mat:
			rows, cols := n, int(in_.k)
			for l in 0 ..< LANES {
				for col in 0 ..< cols {
					sum: f32 = 0
					for row in 0 ..< rows {
						sum += fval(regs[a + i32(row)][l]) * fval(regs[b + i32(col * rows + row)][l])
					}
					regs[dst + i32(col)][l] = fbits(sum)
				}
			}
		case .Transpose:
			rows, cols := n, int(in_.k)
			for l in 0 ..< LANES {
				for col in 0 ..< cols {
					for row in 0 ..< rows {
						regs[dst + i32(row * cols + col)][l] = regs[a + i32(col * rows + row)][l]
					}
				}
			}
		case .Outer:
			rows, cols := n, int(in_.k)
			for l in 0 ..< LANES {
				for col in 0 ..< cols {
					for row in 0 ..< rows {
						regs[dst + i32(col * rows + row)][l] = fbits(fval(regs[a + i32(row)][l]) * fval(regs[b + i32(col)][l]))
					}
				}
			}
		case .Determinant, .Inverse:
			dim := int(in_.k)
			for l in 0 ..< LANES {
				m: [16]f32
				for i in 0 ..< dim * dim {
					m[i] = fval(regs[a + i32(i)][l])
				}
				if in_.op == .Determinant {
					regs[dst][l] = fbits(mat_det(m, dim))
				} else {
					inv := mat_inverse(m, dim)
					for i in 0 ..< dim * dim {
						regs[dst + i32(i)][l] = fbits(inv[i])
					}
				}
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
		case .Texture, .Texture_Bias:
			sample_texture(r, in_)
		case .Texture_Size:
			size := channel_size(r, int(in_.k))
			for l in 0 ..< LANES {
				// Negative lods shift everything out.
				lod := regs[a][l]
				for i in 0 ..< i32(2) {
					regs[dst + i][l] = u32(max(size[i] >> lod, 1) if size[i] > 0 else 0)
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
		case .Jump:
			pc = int(a)
		case .Jump_If_None:
			if exec == {} {
				pc = int(a)
			}
		}
	}
}

// A sample: dst = channel k at uv a, lod b (Texture_Bias: b plus the
// level the uv derivatives pick in the channel's texels, one per quad from
// its top row and left column, as GPUs' texture units pick it).
@(private)
sample_texture :: proc(r: ^Runner, in_: Instr) {
	regs := r.regs
	ch := int(in_.k)
	dst, a, b := in_.dst, in_.a, in_.b
	lod := fl(&regs[b])^
	if size := channel_size(r, ch); in_.op == .Texture_Bias && size != {} {
		w, h := f32(size.x), f32(size.y)
		u, v := fl(&regs[a]), fl(&regs[a + 1])
		for o := 0; o < LANES; o += 4 {
			dx := [2]f32{(u[o + 1] - u[o]) * w, (v[o + 1] - v[o]) * h}
			dy := [2]f32{(u[o + 2] - u[o]) * w, (v[o + 2] - v[o]) * h}
			level := 0.5 * math.log2(max(dx.x * dx.x + dx.y * dx.y, dy.x * dy.x + dy.y * dy.y))
			for k in 0 ..< 4 {
				lod[o + k] += level
			}
		}
	}
	for l in 0 ..< LANES {
		c: [4]f32
		if r.opts.sampler != nil && ch >= 0 {
			c = r.opts.sampler(r.opts.user, ch, {fval(regs[a][l]), fval(regs[a + 1][l])}, lod[l])
		}
		for i in 0 ..< 4 {
			regs[dst + i32(i)][l] = fbits(c[i])
		}
	}
}

// Channel ch's size from the host (0x0 without one).
@(private)
channel_size :: #force_inline proc "contextless" (r: ^Runner, ch: int) -> [2]i32 {
	return r.opts.channel_sizes[ch] if ch >= 0 && ch < len(r.opts.channel_sizes) else {}
}

// Column-major dim x dim matrices.
@(private)
mat_det :: proc(m: [16]f32, dim: int) -> f32 {
	switch dim {
	case 2:
		return m[0] * m[3] - m[2] * m[1]
	case 3:
		return m[0] * (m[4] * m[8] - m[7] * m[5]) - m[3] * (m[1] * m[8] - m[7] * m[2]) + m[6] * (m[1] * m[5] - m[4] * m[2])
	}
	// 4x4 by cofactors of the first column.
	det: f32 = 0
	for i in 0 ..< 4 {
		det += m[i] * cofactor4(m, 0, i)
	}
	return det
}

// Cofactor of element (col, row) of a 4x4 column-major matrix.
@(private)
cofactor4 :: proc(m: [16]f32, col, row: int) -> f32 {
	sub: [9]f32
	k := 0
	for c in 0 ..< 4 {
		if c == col {continue}
		for r in 0 ..< 4 {
			if r == row {continue}
			sub[k] = m[c * 4 + r]
			k += 1
		}
	}
	d := sub[0] * (sub[4] * sub[8] - sub[7] * sub[5]) - sub[3] * (sub[1] * sub[8] - sub[7] * sub[2]) + sub[6] * (sub[1] * sub[5] - sub[4] * sub[2])
	return d if (col + row) % 2 == 0 else -d
}

@(private)
mat_inverse :: proc(m: [16]f32, dim: int) -> (inv: [16]f32) {
	det := mat_det(m, dim)
	id := 1 / det
	switch dim {
	case 2:
		inv[0] = m[3] * id
		inv[1] = -m[1] * id
		inv[2] = -m[2] * id
		inv[3] = m[0] * id
	case 3:
		// inverse = adjugate / det; adj[c][r] = cofactor(r, c).
		at :: proc(m: [16]f32, c, r: int) -> f32 {return m[c * 3 + r]}
		for c in 0 ..< 3 {
			for r in 0 ..< 3 {
				c1, c2 := (r + 1) % 3, (r + 2) % 3
				r1, r2 := (c + 1) % 3, (c + 2) % 3
				inv[c * 3 + r] = (at(m, c1, r1) * at(m, c2, r2) - at(m, c2, r1) * at(m, c1, r2)) * id
			}
		}
	case 4:
		for c in 0 ..< 4 {
			for r in 0 ..< 4 {
				inv[c * 4 + r] = cofactor4(m, r, c) * id
			}
		}
	}
	return
}
