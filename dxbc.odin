package oksl

import "base:runtime"
import "core:slice"

// DXBC emission for Direct3D 12 through SDL_GPU (SDL_GPU_SHADERFORMAT_DXBC,
// "DXBC SM5_1 shaders for D3D12"): a container with the input and output
// signatures (ISGN, OSGN) and a Shader Model 5.1 program (SHEX), vs_5_1 or
// ps_5_1, entry point "main". Registers follow SDL_GPU's rules:
//
//     vertex:   t[n] space0 (channel n, then storage buffers), s[n] space0, b0 space1
//     fragment: t[n] space2 (channel n, then storage buffers), s[n] space2, b0 space3
//
// Stage variables at location N are TEXCOORD N in register N (SDL_GPU's
// vertex semantics), so a vertex shader's outputs and a fragment shader's
// inputs agree by register and by semantic whichever of them a stage reads.
// SV_Position takes register DX_POSITION_REGISTER after them.
//
// Every function is inlined (DXBC subroutines are optional for drivers and
// vkd3d does not take them); see dxbc_emit.odin for statements and
// expressions. Values live in vec4 temporaries, one register per scalar or
// vector and one per matrix column, struct field or array element; a
// variable whose type holds an array, or a matrix indexed dynamically,
// lives in an indexable temporary (x#).

// The register after the 16 locations: SV_Position (VS output, PS input),
// SV_VertexID (VS input).
@(private)
DX_POSITION_REGISTER :: 16
// SV_IsFrontFace's PS input register.
@(private)
DX_FRONT_FACE_REGISTER :: 17

@(private)
Dx_Op :: enum u32 {
	Add                 = 0x00,
	And                 = 0x01,
	Break               = 0x02,
	Breakc              = 0x03,
	Case                = 0x06,
	Continue            = 0x07,
	Default             = 0x0a,
	Discard             = 0x0d,
	Div                 = 0x0e,
	Dp2                 = 0x0f,
	Dp3                 = 0x10,
	Dp4                 = 0x11,
	Else                = 0x12,
	End_If              = 0x15,
	End_Loop            = 0x16,
	End_Switch          = 0x17,
	Eq                  = 0x18,
	Exp                 = 0x19,
	Frc                 = 0x1a,
	Ftoi                = 0x1b,
	Ftou                = 0x1c,
	Ge                  = 0x1d,
	Iadd                = 0x1e,
	If                  = 0x1f,
	Ieq                 = 0x20,
	Ige                 = 0x21,
	Ilt                 = 0x22,
	Imax                = 0x24,
	Imin                = 0x25,
	Imul                = 0x26,
	Ine                 = 0x27,
	Ineg                = 0x28,
	Ishl                = 0x29,
	Ishr                = 0x2a,
	Itof                = 0x2b,
	Log                 = 0x2f,
	Loop                = 0x30,
	Lt                  = 0x31,
	Mad                 = 0x32,
	Min                 = 0x33,
	Max                 = 0x34,
	Mov                 = 0x36,
	Movc                = 0x37,
	Mul                 = 0x38,
	Ne                  = 0x39,
	Not                 = 0x3b,
	Or                  = 0x3c,
	Resinfo             = 0x3d,
	Ret                 = 0x3e,
	Round_Ne            = 0x40,
	Round_Ni            = 0x41,
	Round_Pi            = 0x42,
	Round_Z             = 0x43,
	Rsq                 = 0x44,
	Sample              = 0x45,
	Sample_L            = 0x48,
	Sample_D            = 0x49,
	Sample_B            = 0x4a,
	Sqrt                = 0x4b,
	Switch              = 0x4c,
	Sincos              = 0x4d,
	Udiv                = 0x4e,
	Ult                 = 0x4f,
	Uge                 = 0x50,
	Umax                = 0x53,
	Umin                = 0x54,
	Ushr                = 0x55,
	Utof                = 0x56,
	Xor                 = 0x57,
	Dcl_Resource        = 0x58,
	Dcl_Constant_Buffer = 0x59,
	Dcl_Sampler         = 0x5a,
	Dcl_Input           = 0x5f,
	Dcl_Input_Sgv       = 0x60,
	Dcl_Input_Ps        = 0x62,
	Dcl_Input_Ps_Sgv    = 0x63,
	Dcl_Input_Ps_Siv    = 0x64,
	Dcl_Output          = 0x65,
	Dcl_Output_Siv      = 0x67,
	Dcl_Temps           = 0x68,
	Dcl_Indexable_Temp  = 0x69,
	Dcl_Global_Flags    = 0x6a,
	Deriv_Rtx_Fine      = 0x7b,
	Deriv_Rty_Fine      = 0x7d,
	Rcp                 = 0x81,
	Dcl_Resource_Raw    = 0xa1,
	Ld_Raw              = 0xa5,
}

// Opcode token controls (bits 11..23).
@(private)
DX_SATURATE :: 1 << 13
// The _nz form of if, breakc, discard, retc.
@(private)
DX_NZ :: 1 << 18
// resinfo_uint.
@(private)
DX_RESINFO_UINT :: 2 << 11
@(private)
DX_CB_DYNAMIC :: 1 << 11

// Operand types (register files).
@(private)
Dx_File :: enum u8 {
	Temp      = 0,
	Input     = 1,
	Output    = 2,
	Indexable = 3,
	Imm       = 4,
	Sampler   = 6,
	Resource  = 7,
	Cbuffer   = 8,
	Null      = 13,
	Omask     = 15,
}

// Indices an operand of each file takes in an SM5.1 instruction: r/v/o
// [reg], x [array][reg], cb [id][slot][element], t/s [id][register].
@(private, rodata)
DX_DIMS := #partial #sparse [Dx_File]u8 {
	.Temp      = 1,
	.Input     = 1,
	.Output    = 1,
	.Indexable = 2,
	.Sampler   = 2,
	.Resource  = 2,
	.Cbuffer   = 3,
}

// A source operand, or a value: a register (its last index steps through
// an aggregate's registers) whose components `swz` hold the value's
// components in order, or an immediate (`imm` in the value's order).
@(private)
Dx_Src :: struct {
	file:  Dx_File,
	idx:   [3]u32,
	// A relative index added to the last index: temp register `rel`,
	// component `rel_c`; -1 for none.
	rel:   i32,
	rel_c: u8,
	swz:   [4]u8,
	neg:   bool,
	abs:   bool,
	imm:   [4]u32,
}

@(private)
DX_XYZW :: [4]u8{0, 1, 2, 3}

@(private)
dx_reg :: proc "contextless" (file: Dx_File, reg: u32) -> Dx_Src {
	return {file = file, idx = {reg, 0, 0}, rel = -1, swz = DX_XYZW}
}

// Register k of an aggregate starting at `s`.
@(private)
dx_at :: proc "contextless" (s: Dx_Src, k: int) -> Dx_Src {
	r := s
	if s.file == .Imm {
		return r
	}
	r.idx[DX_DIMS[s.file] - 1] += u32(k)
	r.swz = DX_XYZW
	return r
}

// A value of 4 identical components (one per component).
@(private)
dx_imm :: proc "contextless" (bits: u32) -> Dx_Src {
	return {file = .Imm, rel = -1, swz = DX_XYZW, imm = {bits, bits, bits, bits}}
}

@(private)
dx_imm_f :: proc "contextless" (v: f32) -> Dx_Src {
	return dx_imm(transmute(u32)v)
}

@(private)
dx_imm_i :: proc "contextless" (v: i32) -> Dx_Src {
	return dx_imm(u32(v))
}

// Component j of a value, broadcast.
@(private)
dx_comp :: proc "contextless" (s: Dx_Src, j: int) -> Dx_Src {
	r := s
	c := s.swz[j]
	r.swz = {c, c, c, c}
	if s.file == .Imm {
		v := s.imm[c]
		r.imm = {v, v, v, v}
		r.swz = DX_XYZW
	}
	return r
}

// A value of `rows` components read by an operation on more: a scalar
// broadcasts, a vector repeats its last component past its end.
@(private)
dx_n :: proc "contextless" (s: Dx_Src, rows: int) -> Dx_Src {
	if rows == 1 {
		return dx_comp(s, 0)
	}
	r := s
	for d in rows ..< 4 {
		r.swz[d] = s.swz[rows - 1]
	}
	if s.file == .Imm {
		for d in 0 ..< 4 {
			r.imm[d] = s.imm[r.swz[d]]
		}
		r.swz = DX_XYZW
	}
	return r
}

// Components `pick` of a value (a swizzle of it).
@(private)
dx_swizzle :: proc "contextless" (s: Dx_Src, pick: [4]u8, n: int) -> Dx_Src {
	r := s
	for d in 0 ..< 4 {
		r.swz[d] = s.swz[pick[min(d, n - 1)]]
	}
	if s.file == .Imm {
		for d in 0 ..< 4 {
			r.imm[d] = s.imm[r.swz[d]]
		}
		r.swz = DX_XYZW
	}
	return r
}

@(private)
dx_mask :: proc "contextless" (n: int) -> u8 {
	return u8((1 << uint(n)) - 1)
}

// ---- encoding ---------------------------------------------------------------------------

@(private)
dx_token :: proc "contextless" (file: Dx_File, ncomp: u32, mode: u32, sel: u32, dims: u32, reps: [3]u32) -> u32 {
	return ncomp | mode << 2 | sel << 4 | u32(file) << 12 | dims << 20 | reps[0] << 22 | reps[1] << 25 | reps[2] << 28
}

// Appends the operand's indices (and the relative register of its last).
@(private)
dx_put_indices :: proc(buf: ^[dynamic]u32, s: Dx_Src) {
	dims := int(DX_DIMS[s.file])
	for i in 0 ..< dims {
		append(buf, s.idx[i])
		if i == dims - 1 && s.rel >= 0 {
			append(buf, dx_token(.Temp, 2, 2, u32(s.rel_c), 1, {}), u32(s.rel))
		}
	}
}

@(private)
dx_reps :: proc "contextless" (s: Dx_Src) -> (reps: [3]u32) {
	if s.rel >= 0 {
		reps[DX_DIMS[s.file] - 1] = 3 // immediate plus relative
	}
	return
}

// A destination operand: register `s`, components `mask`.
@(private)
dx_put_dst :: proc(buf: ^[dynamic]u32, s: Dx_Src, mask: u8) {
	switch s.file {
	case .Null:
		append(buf, dx_token(.Null, 0, 0, 0, 0, {}))
		return
	case .Omask:
		append(buf, dx_token(.Omask, 1, 0, 0, 0, {}))
		return
	case .Temp, .Input, .Output, .Indexable, .Imm, .Sampler, .Resource, .Cbuffer:
	}
	append(buf, dx_token(s.file, 2, 0, u32(mask), u32(DX_DIMS[s.file]), dx_reps(s)))
	dx_put_indices(buf, s)
}

// A source operand read for destination components `mask`: one component
// (select_1) when the mask has one, else a swizzle. `vector` reads a
// swizzle whatever the mask (dot products, sample coordinates).
@(private)
dx_put_src :: proc(buf: ^[dynamic]u32, s: Dx_Src, mask: u8, vector := false) {
	single := !vector && (mask == 1 || mask == 2 || mask == 4 || mask == 8)
	d := 0
	if single {
		for mask & (1 << uint(d)) == 0 {
			d += 1
		}
	}
	if s.file == .Imm {
		// Modifiers on an immediate are float ones (integer negation is
		// folded where it is made).
		value :: proc "contextless" (s: Dx_Src, k: int) -> u32 {
			v := s.imm[s.swz[k]]
			if s.abs {
				v &= 0x7fffffff
			}
			if s.neg {
				v ~= 0x80000000
			}
			return v
		}
		if single {
			append(buf, dx_token(.Imm, 1, 0, 0, 0, {}), value(s, d))
		} else {
			append(buf, dx_token(.Imm, 2, 0, 0, 0, {}))
			for k in 0 ..< 4 {
				append(buf, value(s, k))
			}
		}
		return
	}
	if s.file == .Sampler {
		append(buf, dx_token(.Sampler, 0, 0, 0, 2, {}))
		dx_put_indices(buf, s)
		return
	}
	mod: u32
	if s.neg || s.abs {
		mod = 1 | (u32(s.neg) | u32(s.abs) << 1) << 6
	}
	tok: u32
	if single {
		tok = dx_token(s.file, 2, 2, u32(s.swz[d]), u32(DX_DIMS[s.file]), dx_reps(s))
	} else {
		sw := u32(s.swz[0]) | u32(s.swz[1]) << 2 | u32(s.swz[2]) << 4 | u32(s.swz[3]) << 6
		tok = dx_token(s.file, 2, 1, sw, u32(DX_DIMS[s.file]), dx_reps(s))
	}
	if mod != 0 {
		append(buf, tok | 1 << 31, mod)
	} else {
		append(buf, tok)
	}
	dx_put_indices(buf, s)
}

// Patches the instruction started at `start` with its length.
@(private)
dx_end :: proc(buf: ^[dynamic]u32, start: int) {
	buf[start] |= u32(len(buf) - start) << 24
}

// An instruction writing `dst` (components `mask`) from `srcs`.
@(private)
dx_ins :: proc(buf: ^[dynamic]u32, op: Dx_Op, ctl: u32, dst: Dx_Src, mask: u8, srcs: ..Dx_Src) {
	start := len(buf)
	append(buf, u32(op) | ctl)
	dx_put_dst(buf, dst, mask)
	for s in srcs {
		dx_put_src(buf, s, mask)
	}
	dx_end(buf, start)
}

// Two destinations (imul, udiv, sincos), sources read for `mask`.
@(private)
dx_ins2 :: proc(buf: ^[dynamic]u32, op: Dx_Op, d0, d1: Dx_Src, mask: u8, srcs: ..Dx_Src) {
	start := len(buf)
	append(buf, u32(op))
	dx_put_dst(buf, d0, 0 if d0.file == .Null else mask)
	dx_put_dst(buf, d1, 0 if d1.file == .Null else mask)
	for s in srcs {
		dx_put_src(buf, s, mask)
	}
	dx_end(buf, start)
}

// An instruction without destination: if, breakc, discard, switch, case
// (one scalar source), or none (else, loop, break, ...).
@(private)
dx_flow :: proc(buf: ^[dynamic]u32, op: Dx_Op, ctl: u32 = 0, cond: Maybe(Dx_Src) = nil) {
	start := len(buf)
	append(buf, u32(op) | ctl)
	if c, ok := cond.?; ok {
		dx_put_src(buf, c, 1)
	}
	dx_end(buf, start)
}

// ---- container --------------------------------------------------------------------------

// A chunk: its tag and its data, the concatenation of `parts`.
@(private)
Dx_Chunk :: struct {
	tag:   [4]u8,
	parts: [][]u8,
}

// A DXBC container of `chunks` in one allocation of exactly its size: the
// header ("DXBC", checksum, version 1, size, chunk count and offsets), then
// each chunk (tag, size, data, at 4-byte aligned offsets), written once
// and signed in place with dxbc_checksum.
@(private)
dxbc_container :: proc(chunks: []Dx_Chunk, allocator := context.allocator) -> []u8 {
	chunk_size :: proc(ch: Dx_Chunk) -> int {
		n := 0
		for p in ch.parts {
			n += len(p)
		}
		return n
	}
	size := 32 + 4 * len(chunks)
	for ch in chunks {
		size += 8 + round_up(chunk_size(ch), 4)
	}
	out := make([]u8, size, allocator)
	put :: proc(out: []u8, at: int, v: u32) {
		out[at], out[at + 1], out[at + 2], out[at + 3] = u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24)
	}
	copy(out[0:], "DXBC")
	put(out, 20, 1)
	put(out, 24, u32(size))
	put(out, 28, u32(len(chunks)))
	at := 32 + 4 * len(chunks)
	for ch, i in chunks {
		put(out, 32 + 4 * i, u32(at))
		tag := ch.tag
		copy(out[at:], tag[:])
		n := chunk_size(ch)
		put(out, at + 4, u32(n))
		w := at + 8
		for p in ch.parts {
			copy(out[w:], p)
			w += len(p)
		}
		at += 8 + round_up(n, 4)
	}
	sum := dxbc_checksum(out[20:])
	for w, i in sum {
		put(out, 4 + 4 * i, w)
	}
	return out
}

// The container checksum over `data` (the container from byte 20): MD5
// with DXBC's own final block, which puts the bit count first and
// (bit count >> 2) | 1 last instead of the 64-bit count at the end.
dxbc_checksum :: proc(data: []u8) -> (state: [4]u32) {
	state = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476}
	block :: proc(state: ^[4]u32, b: []u8) {
		w: [16]u32
		for i in 0 ..< 16 {
			w[i] = u32(b[4 * i]) | u32(b[4 * i + 1]) << 8 | u32(b[4 * i + 2]) << 16 | u32(b[4 * i + 3]) << 24
		}
		md5_block(state, w)
	}
	n := len(data)
	full := n / 64 * 64
	for i := 0; i < full; i += 64 {
		block(&state, data[i:][:64])
	}
	rest := data[full:]
	bits := u32(n * 8)
	last: [64]u8
	put :: proc(b: []u8, at: int, v: u32) {
		b[at], b[at + 1], b[at + 2], b[at + 3] = u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24)
	}
	if len(rest) < 56 {
		put(last[:], 0, bits)
		copy(last[4:], rest)
		last[4 + len(rest)] = 0x80
		put(last[:], 60, bits >> 2 | 1)
		block(&state, last[:])
	} else {
		copy(last[:], rest)
		last[len(rest)] = 0x80
		block(&state, last[:])
		last = {}
		put(last[:], 0, bits)
		put(last[:], 60, bits >> 2 | 1)
		block(&state, last[:])
	}
	return
}

@(private, rodata)
MD5_K := [64]u32 {
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
}

@(private, rodata)
MD5_S := [64]u32 {
	7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
	5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
	4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
	6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
}

// One MD5 block (RFC 1321).
@(private)
md5_block :: proc "contextless" (state: ^[4]u32, w: [16]u32) {
	a, b, c, d := state[0], state[1], state[2], state[3]
	for i in 0 ..< 64 {
		f: u32
		g: int
		switch i / 16 {
		case 0:
			f, g = d ~ (b & (c ~ d)), i
		case 1:
			f, g = c ~ (d & (b ~ c)), (5 * i + 1) % 16
		case 2:
			f, g = b ~ c ~ d, (3 * i + 5) % 16
		case:
			f, g = c ~ (b | ~d), (7 * i) % 16
		}
		x := a + f + MD5_K[i] + w[g]
		a, d, c = d, c, b
		b += x << MD5_S[i] | x >> (32 - MD5_S[i])
	}
	state[0] += a
	state[1] += b
	state[2] += c
	state[3] += d
}

// ---- signatures -------------------------------------------------------------------------

// A signature element: `semantic` index `index`, system value `sysval`
// (D3D_NAME: 0, POSITION 1, VERTEX_ID 6, IS_FRONT_FACE 9), component type
// (1 uint, 2 int, 3 float), register (0xffffffff for oMask) and mask.
@(private)
Dx_Sig :: struct {
	semantic: string,
	index:    u32,
	sysval:   u32,
	ctype:    u32,
	register: u32,
	mask:     u8,
}

// ISGN or OSGN data: the count, 8, one 24-byte element each, then the
// names. An input's second mask byte is the components read (all of the
// element's, as vkd3d's own compiler writes them), an output's those never
// written.
@(private)
dxbc_signature :: proc(elems: []Dx_Sig, output: bool) -> []u8 {
	out := make([dynamic]u8)
	put :: proc(out: ^[dynamic]u8, v: u32) {
		append(out, u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24))
	}
	put(&out, u32(len(elems)))
	put(&out, 8)
	name_at := 8 + 24 * len(elems)
	for e in elems {
		used := u32(e.mask)
		if output {
			used = 0xf ~ used
		}
		put(&out, u32(name_at))
		name_at += round_up(len(e.semantic) + 1, 4)
		put(&out, e.index)
		put(&out, e.sysval)
		put(&out, e.ctype)
		put(&out, e.register)
		put(&out, u32(e.mask) | used << 8)
	}
	for e in elems {
		append(&out, e.semantic)
		append(&out, 0)
		for len(out) % 4 != 0 {
			append(&out, 0xab)
		}
	}
	return out[:]
}

@(private)
dx_ctype :: proc "contextless" (b: Base) -> u32 {
	#partial switch b {
	case .Uint:
		return 1
	case .Int:
		return 2
	}
	return 3
}

// ---- emitter state ------------------------------------------------------------------------

@(private)
Dx_Flow_Kind :: enum u8 {
	Loop,
	Switch,
	// An inlined function with early returns runs in a loop its returns
	// break out of.
	Return,
}

@(private)
Dx_Flow :: struct {
	kind:     Dx_Flow_Kind,
	// A return inside set the function's flag and broke out: the
	// construct breaks again when the flag is set.
	returned: bool,
}

// An inlined call in progress.
@(private)
Dx_Call :: struct {
	ret:      Dx_Src,
	ret_type: Type,
	// The index in `flow` of its return loop, -1 when the body returns only
	// at its end.
	flow:     int,
	// The temp whose x is set when a return leaves nested constructs.
	flag:     u32,
}

// A stage output's temporary, copied to the output register at the end.
@(private)
Dx_Out :: struct {
	temp: Dx_Src,
	dst:  Dx_Src,
	mask: u8,
}

@(private)
Dx :: struct {
	u:          ^Unit,
	checked:    ^Checked,
	code:       [dynamic]u32,
	next_temp:  u32,
	max_temp:   u32,
	// Indexable temps: registers of each x#, the next free one.
	x_sizes:    [dynamic]u32,
	next_x:     u32,
	// Where each symbol lives; channels and sampler parameters: the channel.
	sym_loc:    []Dx_Src,
	sym_chan:   []i8,
	// Symbols kept in an indexable temp though their type has no array
	// (matrices indexed dynamically).
	sym_x:      []bool,
	// SM5.1 range ids: per channel, per storage buffer.
	tex_id:     [SAMPLER_COUNT]u32,
	buf_id:     []u32,
	buf_reg:    []u32,
	cb_dynamic: bool,
	terminated: bool,
	// Where the code ended after the last break (a switch case needs one
	// last).
	last_break: int,
	flow:       [dynamic]Dx_Flow,
	calls:      [dynamic]Dx_Call,
	outputs:    [dynamic]Dx_Out,
	sig_in:     [dynamic]Dx_Sig,
	sig_out:    [dynamic]Dx_Sig,
	// Input and output declarations, and the code that copies the inputs
	// needing a fix-up (gl_FragCoord.w, gl_FrontFacing).
	io_decls:   [dynamic]u32,
	prologue:   [dynamic]u32,
	// The operand stack (dx_operands) and a call's out / inout arguments,
	// used by index.
	vals:       [dynamic]Dx_Src,
	lvs:        [dynamic]Dx_LV,
}

@(private)
dx_regs :: proc(u: ^Unit, t: Type) -> int {
	if t.array > 0 {
		return int(t.array) * dx_regs(u, elem_type(t))
	}
	if t.base == .Struct {
		n := 0
		for f in struct_fields(u, t.strukt) {
			n += dx_regs(u, ty(u, f.type))
		}
		return n
	}
	return int(t.cols) if t.cols > 0 else 1
}

@(private)
dx_has_array :: proc(u: ^Unit, t: Type) -> bool {
	if t.array > 0 {
		return true
	}
	if t.base == .Struct {
		for f in struct_fields(u, t.strukt) {
			if dx_has_array(u, ty(u, f.type)) {
				return true
			}
		}
	}
	return false
}

// The register offset of field `field` in struct `st`.
@(private)
dx_field_reg :: proc(u: ^Unit, st: i32, field: i32) -> int {
	n := 0
	for f, i in struct_fields(u, st) {
		if i32(i) == field {
			break
		}
		n += dx_regs(u, ty(u, f.type))
	}
	return n
}

@(private)
Dx_Mark :: struct {
	temp, x: u32,
}

@(private)
dx_mark :: proc "contextless" (d: ^Dx) -> Dx_Mark {
	return {d.next_temp, d.next_x}
}

@(private)
dx_release :: proc "contextless" (d: ^Dx, m: Dx_Mark) {
	d.next_temp, d.next_x = m.temp, m.x
}

@(private)
dx_temps :: proc(d: ^Dx, n: int) -> Dx_Src {
	r := dx_reg(.Temp, d.next_temp)
	d.next_temp += u32(n)
	d.max_temp = max(d.max_temp, d.next_temp)
	return r
}

@(private)
dx_temp :: proc(d: ^Dx) -> Dx_Src {
	return dx_temps(d, 1)
}

// An indexable temp of `n` registers.
@(private)
dx_indexable :: proc(d: ^Dx, n: int) -> Dx_Src {
	x := d.next_x
	d.next_x += 1
	if int(x) >= len(d.x_sizes) {
		append(&d.x_sizes, 0)
	}
	d.x_sizes[x] = max(d.x_sizes[x], u32(n))
	return {file = .Indexable, idx = {x, 0, 0}, rel = -1, swz = DX_XYZW}
}

// Registers for a value of type `t`: an indexable temp when it holds an
// array (or `indexable`), else consecutive temps.
@(private)
dx_alloc :: proc(d: ^Dx, t: Type, indexable := false) -> Dx_Src {
	n := dx_regs(d.u, t)
	if indexable || dx_has_array(d.u, t) {
		return dx_indexable(d, n)
	}
	return dx_temps(d, n)
}

// ---- module ------------------------------------------------------------------------------

// The DXBC container of a vertex or fragment shader (module_front rejects
// Options.dxbc for compute), in the context's allocator: it is the only
// allocation there, sized exactly; the emitter's working tables live in the
// temporary allocator and are released on return.
emit_dxbc :: proc(u: ^Unit, c: ^Checked) -> []u8 {
	assert(u.stage != .Compute)
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	out_allocator := context.allocator
	context.allocator = context.temp_allocator
	d := Dx {
		u       = u,
		checked = c,
	}
	d.code = make([dynamic]u32)
	d.x_sizes = make([dynamic]u32)
	d.flow = make([dynamic]Dx_Flow)
	d.calls = make([dynamic]Dx_Call)
	d.outputs = make([dynamic]Dx_Out)
	d.sig_in = make([dynamic]Dx_Sig)
	d.sig_out = make([dynamic]Dx_Sig)
	d.io_decls = make([dynamic]u32)
	d.prologue = make([dynamic]u32)
	d.vals = make([dynamic]Dx_Src)
	d.lvs = make([dynamic]Dx_LV)
	d.sym_loc = make([]Dx_Src, len(u.syms))
	d.sym_chan = make([]i8, len(u.syms))
	d.sym_x = make([]bool, len(u.syms))
	for &ch in d.sym_chan {
		ch = -1
	}
	d.buf_id = make([]u32, len(c.buffers))
	d.buf_reg = make([]u32, len(c.buffers))
	dx_find_indexed_matrices(&d)

	resources := make([dynamic]u32)
	dx_declare_interface(&d, &resources)
	// Globals (zero, like SPIR-V's null-initialized privates), then the
	// entry's body.
	for g in c.globals {
		d.sym_loc[g] = dx_alloc(&d, sym_type(u, g), d.sym_x[g])
		dx_zero(&d, d.sym_loc[g], sym_type(u, g))
	}
	dx_block(&d, u.func_decls[u.funcs[c.entry].decl].body)
	for o in d.outputs {
		dx_ins(&d.code, .Mov, 0, o.dst, o.mask, o.temp)
	}
	dx_flow(&d.code, .Ret)

	// SHEX: the version and length, the global flags (refactoringAllowed,
	// as fxc sets it: drivers may contract and reorder float arithmetic,
	// as SPIR-V's and MSL's do; without it a driver treats every
	// instruction as precise), the declarations, then the code, written
	// straight into the container.
	is_vertex := u.stage == .Vertex
	head := [3]u32{0x51 | (u32(1) if is_vertex else 0) << 16, 0, u32(Dx_Op.Dcl_Global_Flags) | 1 << 11 | 1 << 24}
	if d.cb_dynamic && len(resources) > 0 && resources[0] & 0x7ff == u32(Dx_Op.Dcl_Constant_Buffer) {
		resources[0] |= DX_CB_DYNAMIC
	}
	temps := make([dynamic]u32, 0, 2 + 4 * len(d.x_sizes))
	if d.max_temp > 0 {
		append(&temps, u32(Dx_Op.Dcl_Temps) | 2 << 24, d.max_temp)
	}
	for size, x in d.x_sizes {
		append(&temps, u32(Dx_Op.Dcl_Indexable_Temp) | 4 << 24, u32(x), size, 4)
	}
	shex := [?][]u32{head[:], resources[:], d.io_decls[:], temps[:], d.prologue[:], d.code[:]}
	words := 0
	for p in shex {
		words += len(p)
	}
	head[1] = u32(words)

	slice.sort_by(d.sig_in[:], proc(a, b: Dx_Sig) -> bool {return a.register < b.register})
	slice.sort_by(d.sig_out[:], proc(a, b: Dx_Sig) -> bool {return a.register < b.register})
	shex_bytes: [len(shex)][]u8
	for p, i in shex {
		shex_bytes[i] = slice.to_bytes(p)
	}
	isgn := [1][]u8{dxbc_signature(d.sig_in[:], false)}
	osgn := [1][]u8{dxbc_signature(d.sig_out[:], true)}
	chunks := [?]Dx_Chunk{{{'I', 'S', 'G', 'N'}, isgn[:]}, {{'O', 'S', 'G', 'N'}, osgn[:]}, {{'S', 'H', 'E', 'X'}, shex_bytes[:]}}
	return dxbc_container(chunks[:], out_allocator)
}

// Marks symbols whose matrices an expression indexes dynamically: they
// live in an indexable temp.
@(private)
dx_find_indexed_matrices :: proc(d: ^Dx) {
	u := d.u
	for e in u.exprs {
		if e.kind != .Index {
			continue
		}
		ix := u.exprs[e.b]
		if ix.kind == .Int_Lit || ix.kind == .Uint_Lit || !is_matrix(expr_type(u, e.a)) {
			continue
		}
		if s := target_sym(u, e.a); s != 0 {
			d.sym_x[s] = true
		}
	}
}

// Writes zero to every register of a value of type `t` at `at`.
@(private)
dx_zero :: proc(d: ^Dx, at: Dx_Src, t: Type) {
	for k in 0 ..< dx_regs(d.u, t) {
		dx_ins(&d.code, .Mov, 0, dx_at(at, k), 0xf, dx_imm(0))
	}
}

// Declares the interface rows: resources (in `res`, written before the
// inputs), inputs and outputs, with their signature elements.
@(private)
dx_declare_interface :: proc(d: ^Dx, res: ^[dynamic]u32) {
	u := d.u
	c := d.checked
	fragment := u.stage == .Fragment
	space_tex: u32 = 2 if fragment else 0
	space_cb: u32 = 3 if fragment else 1
	next_tex: u32 = 0
	ins_decl :: proc(buf: ^[dynamic]u32, op: Dx_Op, ctl: u32, words: ..u32) {
		start := len(buf)
		append(buf, u32(op) | ctl)
		append(buf, ..words)
		dx_end(buf, start)
	}
	// The constant buffer first, then samplers, textures and buffers.
	for row in c.interface {
		if row.kind == .Uniform_Block {
			size := u32(round_up(c.block_size, 16) / 16)
			ins_decl(res, .Dcl_Constant_Buffer, 0, dx_token(.Cbuffer, 2, 1, 0xe4, 3, {}), 0, u32(row.binding), u32(row.binding), size, space_cb)
		}
	}
	for row in c.interface {
		if row.kind == .Channel {
			ch := u32(row.binding)
			ins_decl(res, .Dcl_Sampler, 0, dx_token(.Sampler, 0, 0, 0, 3, {}), next_tex, ch, ch, space_tex)
			d.tex_id[ch] = next_tex
			d.sym_chan[row.sym] = i8(ch)
			next_tex += 1
		}
	}
	for row in c.interface {
		if row.kind == .Channel {
			ch := u32(row.binding)
			dim: u32 = 8 if sym_type(u, row.sym) == T_SAMPLER_ARRAY else 3
			ins_decl(res, .Dcl_Resource, dim << 11, dx_token(.Resource, 1, 0, 0, 3, {}), d.tex_id[ch], ch, ch, 0x5555, space_tex)
		}
	}
	for row in c.interface {
		if row.kind == .Buffer {
			i := u.syms[row.sym].index
			reg := u32(row.binding)
			ins_decl(res, .Dcl_Resource_Raw, 0, dx_token(.Resource, 1, 0, 0, 3, {}), next_tex, reg, reg, space_tex)
			d.buf_id[i] = next_tex
			d.buf_reg[i] = reg
			next_tex += 1
		}
	}

	io := &d.io_decls
	// The outputs are declared first (fxc's order), then the inputs.
	for row in c.interface {
		sym := u.syms[row.sym]
		t := ty(u, sym.type)
		#partial switch row.kind {
		case .Stage_Out:
			reg := u32(row.binding)
			mask := dx_mask(int(t.rows))
			ins_decl(io, .Dcl_Output, 0, dx_token(.Output, 2, 0, u32(mask), 1, {}), reg)
			append(&d.sig_out, Dx_Sig{"SV_Target" if fragment else "TEXCOORD", reg, 0, dx_ctype(t.base), reg, mask})
			dx_shadow_output(d, row.sym, dx_reg(.Output, reg), mask)
		case .Position:
			reg := u32(DX_POSITION_REGISTER)
			ins_decl(io, .Dcl_Output_Siv, 0, dx_token(.Output, 2, 0, 0xf, 1, {}), reg, 1)
			append(&d.sig_out, Dx_Sig{"SV_Position", 0, 1, 3, reg, 0xf})
			dx_shadow_output(d, row.sym, dx_reg(.Output, reg), 0xf)
		case .Sample_Mask:
			append(io, u32(Dx_Op.Dcl_Output) | 2 << 24, dx_token(.Omask, 1, 0, 0, 0, {}))
			append(&d.sig_out, Dx_Sig{"SV_Coverage", 0, 0, 1, 0xffffffff, 1})
			loc := dx_alloc(d, t)
			d.sym_loc[row.sym] = loc
			// All samples kept when unwritten, as on the CPU.
			dx_ins(&d.prologue, .Mov, 0, loc, 1, dx_imm(0xffffffff))
			append(&d.outputs, Dx_Out{loc, Dx_Src{file = .Omask, rel = -1, swz = DX_XYZW}, 1})
		}
	}
	for row in c.interface {
		sym := u.syms[row.sym]
		t := ty(u, sym.type)
		#partial switch row.kind {
		case .Stage_In:
			reg := u32(row.binding)
			mask := dx_mask(int(t.rows))
			if fragment {
				interp: u32 = 1 if row.flat || t.base != .Float else 2 // constant, linear
				ins_decl(io, .Dcl_Input_Ps, interp << 11, dx_token(.Input, 2, 0, u32(mask), 1, {}), reg)
			} else {
				ins_decl(io, .Dcl_Input, 0, dx_token(.Input, 2, 0, u32(mask), 1, {}), reg)
			}
			append(&d.sig_in, Dx_Sig{"TEXCOORD", reg, 0, dx_ctype(t.base), reg, mask})
			d.sym_loc[row.sym] = dx_reg(.Input, reg)
		case .Frag_Coord:
			reg := u32(DX_POSITION_REGISTER)
			// linear noperspective
			ins_decl(io, .Dcl_Input_Ps_Siv, 4 << 11, dx_token(.Input, 2, 0, 0xf, 1, {}), reg, 1)
			append(&d.sig_in, Dx_Sig{"SV_Position", 0, 1, 3, reg, 0xf})
			// D3D's SV_Position.w is the clip w; gl_FragCoord.w is 1 / w.
			loc := dx_temp(d)
			dx_ins(&d.prologue, .Mov, 0, loc, 0x7, dx_reg(.Input, reg))
			dx_ins(&d.prologue, .Rcp, 0, loc, 0x8, dx_reg(.Input, reg))
			d.sym_loc[row.sym] = loc
		case .Front_Facing:
			reg := u32(DX_FRONT_FACE_REGISTER)
			ins_decl(io, .Dcl_Input_Ps_Sgv, 1 << 11, dx_token(.Input, 2, 0, 0x1, 1, {}), reg, 9)
			append(&d.sig_in, Dx_Sig{"SV_IsFrontFace", 0, 9, 1, reg, 1})
			loc := dx_temp(d)
			dx_ins(&d.prologue, .Ine, 0, loc, 1, dx_reg(.Input, reg), dx_imm(0))
			d.sym_loc[row.sym] = loc
		case .Vertex_ID:
			reg := u32(DX_POSITION_REGISTER)
			ins_decl(io, .Dcl_Input_Sgv, 0, dx_token(.Input, 2, 0, 0x1, 1, {}), reg, 6)
			append(&d.sig_in, Dx_Sig{"SV_VertexID", 0, 6, 1, reg, 1})
			d.sym_loc[row.sym] = dx_reg(.Input, reg)
		}
	}
}

// A stage output lives in a temporary (outputs cannot be read), zero until
// written and copied to `dst` at the end.
@(private)
dx_shadow_output :: proc(d: ^Dx, sym: Sym_Id, dst: Dx_Src, mask: u8) {
	loc := dx_temp(d)
	d.sym_loc[sym] = loc
	dx_ins(&d.prologue, .Mov, 0, loc, 0xf, dx_imm(0))
	append(&d.outputs, Dx_Out{loc, dst, mask})
}
