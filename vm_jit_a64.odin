package oksl

import "base:runtime"
import "core:math"

// CPU backend, part 3c: the arm64 NEON emitter (vm_jit.odin has the shared
// front end, the slot register cache and the runners).
//
// A value is eight lanes as a pair of 128-bit V registers, quad 0 (lanes 0-3)
// in the first, so a register index r of the shared cache (0..15) is
// v(2r), v(2r + 1): 0..7 the scratch and work registers (v0..v15, 7 the exec
// mask's lanes), 8..15 the slot cache (v16..v31). Pinned GPRs:
//   x19 : ^Jit_Ctx
//   x20 : [^]Jit_Slot (32 bytes a slot: lanes 0-3, then 4-7)
//   x21 : [^]u8 masks (masks[1] = DISCARD, masks[2..] = saved masks)
//   w22 : the live MASK_EXEC, 0..255 (loaded from ctx.lanes)
//   x23 : ^Jit_A64_Tables
// x16 and x17 are the emitter's address scratch (x17 for large offsets only).
//
// Floats follow NEON's own semantics where they differ from the
// interpreter's (README): every float min, max and clamp takes fminnm and
// fmaxnm, the shader's and the lowering's (the Shadertoy output clamp,
// saturate, textureGrad's lod, smoothstep's clamp of its interpolant, where
// they agree): a quiet NaN operand gives the other, a signaling one NaN,
// and +0 is above -0. NaN payloads inside a program are the hardware's
// (what reaches the host or an integer is canonical, as everywhere). Each
// multiply and add rounds apart, as in the interpreter (no fused
// multiply-add).

_ :: runtime
_ :: math

when JIT_A64 {

@(private)
JIT_PAD :: u8(0)

@(private)
Jit_A64_Tables :: struct #align(64) {
	// Each exec mask's eight lanes as all-ones words.
	mask32:    [256][8]u32,
	bits8:     [8]u16, // lane l's bit, for the mask of eight compares
	inf:       [4]u32,
	canon:     [4]u32, // the canonical NaN, 0x7fc00000
	deg2rad:   [4]f32,
	rad2deg:   [4]f32,
	// Bound channels (the sample and fetch subroutines).
	d_half:    [2]f64,
	d256:      [2]f64,
	d128:      [2]f64,
	d_lo:      [2]f64, // -2^31 and 2^31 - 2: a texel index's range
	d_hi:      [2]f64,
	f256:      [4]f32,
	inv256:    [4]f32,
	// quad_level8's constants.
	ql:        [10][4]u32,
}

@(private) A64_OFF_BITS8 :: u32(offset_of(Jit_A64_Tables, bits8))
@(private) A64_OFF_INF :: u32(offset_of(Jit_A64_Tables, inf))
@(private) A64_OFF_CANON :: u32(offset_of(Jit_A64_Tables, canon))
@(private) A64_OFF_DHALF :: u32(offset_of(Jit_A64_Tables, d_half))
@(private) A64_OFF_D256 :: u32(offset_of(Jit_A64_Tables, d256))
@(private) A64_OFF_D128 :: u32(offset_of(Jit_A64_Tables, d128))
@(private) A64_OFF_DLO :: u32(offset_of(Jit_A64_Tables, d_lo))
@(private) A64_OFF_DHI :: u32(offset_of(Jit_A64_Tables, d_hi))
@(private) A64_OFF_F256 :: u32(offset_of(Jit_A64_Tables, f256))
@(private) A64_OFF_INV256 :: u32(offset_of(Jit_A64_Tables, inv256))
@(private) A64_OFF_QL :: u32(offset_of(Jit_A64_Tables, ql))
@(private) A64_OFF_DEG2RAD :: u32(offset_of(Jit_A64_Tables, deg2rad))
@(private) A64_OFF_RAD2DEG :: u32(offset_of(Jit_A64_Tables, rad2deg))

#assert(offset_of(Jit_A64_Tables, bits8) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, inf) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, canon) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, deg2rad) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, rad2deg) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, d_half) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, d256) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, d128) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, d_lo) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, d_hi) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, f256) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, inv256) % 16 == 0)
#assert(offset_of(Jit_A64_Tables, ql) % 16 == 0)

#assert(align_of(Jit_A64_Tables) >= 16)

@(private)
jit_a64_tables: Jit_A64_Tables

@(private)
jit_a64_init_tables :: proc "contextless" () {
	t := &jit_a64_tables
	for m in 0 ..< 256 {
		for l in 0 ..< 8 {
			t.mask32[m][l] = 0xffff_ffff if (m & (1 << uint(l))) != 0 else 0
		}
	}
	for l in 0 ..< 8 {
		t.bits8[l] = u16(1) << uint(l)
	}
	t.inf = 0x7f80_0000
	t.canon = 0x7fc0_0000
	t.deg2rad = math.PI / 180.0
	t.rad2deg = 180.0 / math.PI
	t.d_half = 0.5
	t.d256 = 256
	t.d128 = 128
	t.d_lo = -2147483648.0
	t.d_hi = 2147483646.0
	t.f256 = 256
	t.inv256 = 1.0 / 256
	// FLT_MIN, 2^24, sqrt(0.5)'s bits, C0..C5, 24.
	ql := [10]u32{0x0080_0000, 0x4b80_0000, 0x3f35_04f3, 0x3FB8AA72, 0xBF38A0F9, 0x3EF5A9EC, 0xBEBBA56E, 0x3EA2EAB7, 0xBE531AE5, 0x41c0_0000}
	for v, i in ql {
		t.ql[i] = v
	}
}

// Cold callback stub for an in-block .Texture/.Texture_Bias/.Texel_Fetch when
// its channel has no bound image (ctx.chan[k].ptr == nil).
@(private)
A64_Tex_Stub :: struct {
	unbound_patch: int,
	ret_off:       int,
	stub_off:      int,
	stub_end:      int,
	pc:            i32,
	comp_mask:     u8,
	n_dirty:       u8,
	n_save:        u8,
	dirty_slots:   [8]i32,
	dirty_regs:    [8]u8,
	save_regs:     [8]u8,
}

@(private)
Jit_Arch_Emitter :: struct {
	unbound_tex: [dynamic]A64_Tex_Stub,
}

@(private)
jit_arch_init :: proc(e: ^Jit_Emitter, p: ^Program, num_slots: int, fma, f16c: bool, heap: runtime.Allocator) {
	assert(uintptr(&jit_a64_tables) % 16 == 0)
	e.unbound_tex = make([dynamic]A64_Tex_Stub, 0, 8, heap)
}

@(private)
jit_arch_reset :: proc(e: ^Jit_Emitter) {
	clear(&e.unbound_tex)
}

@(private)
jit_arch_destroy :: proc(e: ^Jit_Emitter, heap: runtime.Allocator) {
	delete(e.unbound_tex)
}

// ---- executable memory ------------------------------------------------------------------

when ODIN_OS == .Darwin {
	foreign import jit_libsystem "system:System"
	@(private, default_calling_convention = "c")
	foreign jit_libsystem {
		@(link_name = "mmap") jit_darwin_mmap :: proc(addr: rawptr, length: uint, prot, flags, fd: i32, offset: i64) -> rawptr ---
		@(link_name = "pthread_jit_write_protect_np") jit_darwin_write_protect :: proc(enabled: i32) ---
		@(link_name = "sys_icache_invalidate") jit_darwin_icache_invalidate :: proc(start: rawptr, length: uint) ---
	}
} else {
	foreign import jit_libc "system:c"
	@(private, default_calling_convention = "c")
	foreign jit_libc {
		// compiler-rt's and libgcc's.
		@(link_name = "__clear_cache") jit_clear_cache :: proc(start, end: rawptr) ---
	}
}

// macOS's jit_map_code: a MAP_JIT mapping, written with this thread's JIT
// write protection off and then on again (pthread has no call that reads
// the state to restore), its instruction cache invalidated.
when ODIN_OS == .Darwin {
	@(private)
	jit_map_code_darwin :: proc(code: []u8) -> []u8 {
		n := len(code)
		PROT_RWX :: 7
		MAP_PRIVATE_ANON_JIT :: 0x0002 | 0x1000 | 0x0800
		size := (uint(n) + 16383) &~ uint(16383)
		ptr := jit_darwin_mmap(nil, size, PROT_RWX, MAP_PRIVATE_ANON_JIT, -1, 0)
		if ptr == nil || uintptr(ptr) == ~uintptr(0) {
			return nil
		}
		mem_slice := ([^]u8)(ptr)[:size]
		jit_darwin_write_protect(0)
		copy(mem_slice, code)
		jit_darwin_write_protect(1)
		jit_darwin_icache_invalidate(ptr, uint(n))
		// The code's length (munmap in jit_destroy rounds it to pages).
		return mem_slice[:n]
	}
}

// ---- arm64 encoding ---------------------------------------------------------------------

@(private) A_X0 :: u8(0)
@(private) A_X1 :: u8(1)
@(private) A_X2 :: u8(2)
@(private) A_X3 :: u8(3)
@(private) A_X16 :: u8(16)
@(private) A_X17 :: u8(17)
@(private) A_CTX :: u8(19)
@(private) A_SLOTS :: u8(20)
@(private) A_MASKS :: u8(21)
@(private) A_EXEC :: u8(22)
@(private) A_TAB :: u8(23)
@(private) A_SP :: u8(31) // or wzr/xzr, by the instruction
@(private) A_ZR :: u8(31)

// Condition codes.
@(private) A_EQ :: 0
@(private) A_HS :: 2
@(private) A_LS :: 9

// Vector ops on .4s (.16b for the bitwise ones), Rd | Rn << 5 | Rm << 16.
@(private) V_FADD :: u32(0x4e20d400)
@(private) V_FSUB :: u32(0x4ea0d400)
@(private) V_FMUL :: u32(0x6e20dc00)
@(private) V_FDIV :: u32(0x6e20fc00)
@(private) V_FMAXNM :: u32(0x4e20c400)
@(private) V_FMINNM :: u32(0x4ea0c400)
@(private) V_FCMEQ :: u32(0x4e20e400)
@(private) V_FCMGE :: u32(0x6e20e400)
@(private) V_FCMGT :: u32(0x6ea0e400)
@(private) V_FABD :: u32(0x6ea0d400)
@(private) V_ADD :: u32(0x4ea08400)
@(private) V_SUB :: u32(0x6ea08400)
@(private) V_MUL :: u32(0x4ea09c00)
@(private) V_AND :: u32(0x4e201c00)
@(private) V_ORR :: u32(0x4ea01c00)
@(private) V_EOR :: u32(0x6e201c00)
@(private) V_BIC :: u32(0x4e601c00)
@(private) V_CMEQ :: u32(0x6ea08c00)
@(private) V_CMGT :: u32(0x4ea03400)
@(private) V_CMGE :: u32(0x4ea03c00)
@(private) V_CMHI :: u32(0x6ea03400)
@(private) V_CMHS :: u32(0x6ea03c00)
@(private) V_CMTST :: u32(0x4ea08c00)
@(private) V_SMIN :: u32(0x4ea06c00)
@(private) V_SMAX :: u32(0x4ea06400)
@(private) V_UMIN :: u32(0x6ea06c00)
@(private) V_UMAX :: u32(0x6ea06400)
@(private) V_USHL :: u32(0x6ea04400)
@(private) V_SSHL :: u32(0x4ea04400)
@(private) V_BSL :: u32(0x6e601c00)
@(private) V_BIT :: u32(0x6ea01c00)
@(private) V_BIF :: u32(0x6ee01c00)
@(private) V_TRN1 :: u32(0x4e802800)
@(private) V_TRN2 :: u32(0x4e806800)
@(private) V_ZIP1_2D :: u32(0x4ec03800)
@(private) V_ZIP2_2D :: u32(0x4ec07800)
@(private) V_UZP1_8H :: u32(0x4e401800)
@(private) V_FDIV_2D :: u32(0x6e60fc00)
// Unary, Rd | Rn << 5.
@(private) V_FNEG :: u32(0x6ea0f800)
@(private) V_FABS :: u32(0x4ea0f800)
@(private) V_FSQRT :: u32(0x6ea1f800)
@(private) V_FRINTM :: u32(0x4e219800)
@(private) V_FRINTP :: u32(0x4ea18800)
@(private) V_FRINTZ :: u32(0x4ea19800)
@(private) V_FRINTN :: u32(0x4e218800)
@(private) V_FRINTA :: u32(0x6e218800)
@(private) V_SCVTF :: u32(0x4e21d800)
@(private) V_UCVTF :: u32(0x6e21d800)
@(private) V_FCVTZS :: u32(0x4ea1b800)
@(private) V_FCVTZU :: u32(0x6ea1b800)
@(private) V_FCMEQ0 :: u32(0x4ea0d800)
@(private) V_FCMGT0 :: u32(0x4ea0c800)
@(private) V_FCMLT0 :: u32(0x4ea0e800)
@(private) V_NEG :: u32(0x6ea0b800)
@(private) V_ABS :: u32(0x4ea0b800)
@(private) V_MVN :: u32(0x6e205800)
@(private) V_CMEQ0 :: u32(0x4ea09800)
@(private) V_CMGT0 :: u32(0x4ea08800)
@(private) V_CMLT0 :: u32(0x4ea0a800)
@(private) V_UMINV :: u32(0x6eb1a800)
@(private) V_ADDV_8H :: u32(0x4e71b800)
@(private) V_XTN :: u32(0x0ea12800)
@(private) V_XTN2 :: u32(0x4ea12800)
@(private) V_SXTL :: u32(0x0f20a400)
@(private) V_SXTL2 :: u32(0x4f20a400)
@(private) V_UXTL :: u32(0x2f20a400)
@(private) V_UXTL2 :: u32(0x6f20a400)
@(private) V_SCVTF_2D :: u32(0x4e61d800)
@(private) V_UCVTF_2D :: u32(0x6e61d800)
@(private) V_FCVTZS_2D :: u32(0x4ee1b800)
@(private) V_FCVTZU_2D :: u32(0x6ee1b800)
@(private) V_USHR31 :: u32(0x6f210400)
// Vector immediates (Rd | op).
@(private) V_ZERO :: u32(0x6f00e400)
@(private) FMOV_ONE :: u32(0x4f03f600)
@(private) FMOV_NEG_ONE :: u32(0x4f07f600)
@(private) FMOV_TWO :: u32(0x4f00f400)
@(private) FMOV_THREE :: u32(0x4f00f500)
@(private) V_XTN_4H :: u32(0x0e612800)
@(private) V_XTN2_8H :: u32(0x4e612800)
@(private) MOVI_256_8H :: u32(0x4f00a420)
@(private) V_SUB_8H :: u32(0x6e608400)
@(private) V_UXTL_8H :: u32(0x2f08a400)
@(private) V_MUL_8H :: u32(0x4e609c00)
@(private) V_MLA_8H :: u32(0x4e609400)
@(private) V_UMULL_4S :: u32(0x2e60c000)
@(private) V_UMLAL_4S :: u32(0x2e608000)
@(private) V_UMULL2_4S :: u32(0x6e60c000)
@(private) V_UMLAL2_4S :: u32(0x6e608000)
@(private) V_UCVTF_F16 :: u32(0x6f30e400)
@(private) V_MUL_8H_E :: u32(0x4f408000)
@(private) V_MLA_8H_E :: u32(0x6f400000)
@(private) V_UMULL_4S_E :: u32(0x2f40a000)
@(private) V_UMLAL2_4S_E :: u32(0x6f402000)
@(private) MOVI_31 :: u32(0x4f0007e0)

@(private)
ai :: #force_inline proc(e: ^Jit_Emitter, w: u32) {
	emit_u32(&e.buf, w)
}

@(private)
a_v3 :: #force_inline proc(e: ^Jit_Emitter, op: u32, d, n, m: u8) {
	ai(e, op | u32(d) | u32(n) << 5 | u32(m) << 16)
}

@(private)
a_v2 :: #force_inline proc(e: ^Jit_Emitter, op: u32, d, n: u8) {
	ai(e, op | u32(d) | u32(n) << 5)
}

// Pair forms of registers r (v(2r), v(2r + 1)); q is a single V register
// (a constant for both halves).
@(private)
p3 :: #force_inline proc(e: ^Jit_Emitter, op: u32, d, a, b: u8) {
	a_v3(e, op, 2 * d, 2 * a, 2 * b)
	a_v3(e, op, 2 * d + 1, 2 * a + 1, 2 * b + 1)
}

@(private)
p3q :: #force_inline proc(e: ^Jit_Emitter, op: u32, d, a, q: u8) {
	a_v3(e, op, 2 * d, 2 * a, q)
	a_v3(e, op, 2 * d + 1, 2 * a + 1, q)
}

@(private)
pq3 :: #force_inline proc(e: ^Jit_Emitter, op: u32, d, q, b: u8) {
	a_v3(e, op, 2 * d, q, 2 * b)
	a_v3(e, op, 2 * d + 1, q, 2 * b + 1)
}

@(private)
p2 :: #force_inline proc(e: ^Jit_Emitter, op: u32, d, a: u8) {
	a_v2(e, op, 2 * d, 2 * a)
	a_v2(e, op, 2 * d + 1, 2 * a + 1)
}

@(private)
pmov :: #force_inline proc(e: ^Jit_Emitter, d, a: u8) {
	if d != a {
		p3(e, V_ORR, d, a, a)
	}
}

@(private)
pzero :: #force_inline proc(e: ^Jit_Emitter, d: u8) {
	ai(e, V_ZERO | u32(2 * d))
	ai(e, V_ZERO | u32(2 * d + 1))
}

// All-ones lanes to 1 (ushr #31).
@(private)
pone :: #force_inline proc(e: ^Jit_Emitter, d, a: u8) {
	p2(e, V_USHR31, d, a)
}

// d = m ? t : f lane by lane (m's lanes all ones or zero), for pairs.
@(private)
psel :: proc(e: ^Jit_Emitter, d, m, t, f: u8) {
	if d == m {
		p3(e, V_BSL, d, t, f)
	} else if d == f {
		p3(e, V_BIT, d, t, m)
	} else if d == t {
		p3(e, V_BIF, d, f, m)
	} else {
		pmov(e, d, f)
		p3(e, V_BIT, d, t, m)
	}
}

@(private)
a_imm :: #force_inline proc(e: ^Jit_Emitter, op: u32, q: u8) {
	ai(e, op | u32(q))
}

@(private)
a_elem_h :: #force_inline proc(e: ^Jit_Emitter, op: u32, d, n, m: u8, l: u32) {
	assert(m < 16)
	ai(e, op | ((l & 3) << 20) | u32(m) << 16 | ((l >> 2) << 11) | u32(n) << 5 | u32(d))
}

// ldr q, [x23, #off] (a table constant).
@(private)
a_tab :: #force_inline proc(e: ^Jit_Emitter, q: u8, off: u32) {
	ai(e, 0x3dc00000 | (off / 16) << 10 | u32(A_TAB) << 5 | u32(q))
}

// x = imm (movz and movk).
@(private)
a_mov_imm :: proc(e: ^Jit_Emitter, x: u8, v: u64) {
	ai(e, 0xd2800000 | u32(v & 0xffff) << 5 | u32(x))
	for hw in u32(1) ..< 4 {
		part := (v >> (16 * hw)) & 0xffff
		if part != 0 {
			ai(e, 0xf2800000 | hw << 21 | u32(part) << 5 | u32(x))
		}
	}
}

// w = imm.
@(private)
a_mov_w :: proc(e: ^Jit_Emitter, w: u8, v: u32) {
	ai(e, 0x52800000 | (v & 0xffff) << 5 | u32(w))
	if v >> 16 != 0 {
		ai(e, 0x72a00000 | (v >> 16) << 5 | u32(w))
	}
}

// mov wd, wm and mov xd, xm.
@(private) a_mov_ww :: #force_inline proc(e: ^Jit_Emitter, d, m: u8) {ai(e, 0x2a0003e0 | u32(m) << 16 | u32(d))}
@(private) a_mov_xx :: #force_inline proc(e: ^Jit_Emitter, d, m: u8) {ai(e, 0xaa0003e0 | u32(m) << 16 | u32(d))}

// A register pair r loaded from or stored to [base + off]: ldp or stp in
// range, else two ldr or str with unsigned offsets, else through x17.
@(private)
a_pair_mem :: proc(e: ^Jit_Emitter, load: bool, r, base: u8, off: i64) {
	q0, q1 := u32(2 * r), u32(2 * r + 1)
	pair_op := u32(0xad400000) if load else 0xad000000
	if off % 16 == 0 && off >= -1024 && off <= 1008 {
		ai(e, pair_op | (u32(off / 16) & 0x7f) << 15 | q1 << 10 | u32(base) << 5 | q0)
		return
	}
	if off % 16 == 0 && off >= 0 && off + 16 <= 65520 {
		op := u32(0x3dc00000) if load else 0x3d800000
		ai(e, op | u32(off / 16) << 10 | u32(base) << 5 | q0)
		ai(e, op | u32(off / 16 + 1) << 10 | u32(base) << 5 | q1)
		return
	}
	a_mov_imm(e, A_X17, u64(off))
	// The extended-register add, whose Rn of 31 is sp (the shifted one's is xzr).
	ai(e, 0x8b206000 | u32(A_X17) << 16 | u32(base) << 5 | u32(A_X17)) // add x17, base, x17, uxtx
	ai(e, pair_op | q1 << 10 | u32(A_X17) << 5 | q0)
}

// The slot cache's moves (vm_jit.odin).
@(private)
jit_ld_slot :: proc(e: ^Jit_Emitter, dst: u8, slot: i32) {
	a_pair_mem(e, true, dst, A_SLOTS, i64(slot) * 32)
}

@(private)
jit_st_slot :: proc(e: ^Jit_Emitter, slot: i32, src: u8) {
	a_pair_mem(e, false, src, A_SLOTS, i64(slot) * 32)
}

@(private)
jit_mov :: proc(e: ^Jit_Emitter, dst, src: u8) {
	pmov(e, dst, src)
}

// ldrb w, [x21 + m] and strb.
@(private) a_ldrb_mask :: #force_inline proc(e: ^Jit_Emitter, w: u8, m: i32) {ai(e, 0x39400000 | u32(m) << 10 | u32(A_MASKS) << 5 | u32(w))}
@(private) a_strb_mask :: #force_inline proc(e: ^Jit_Emitter, w: u8, m: i32) {ai(e, 0x39000000 | u32(m) << 10 | u32(A_MASKS) << 5 | u32(w))}

// A forward branch (cond < 0: b), its position for a_land.
@(private)
a_branch :: proc(e: ^Jit_Emitter, cond: int) -> int {
	pos := len(e.buf)
	if cond < 0 {
		ai(e, 0x14000000)
	} else {
		ai(e, 0x54000000 | u32(cond))
	}
	return pos
}

// cbz w (cbnz w when nonzero).
@(private)
a_cbz :: proc(e: ^Jit_Emitter, w: u8, nonzero: bool) -> int {
	pos := len(e.buf)
	ai(e, (0x35000000 if nonzero else 0x34000000) | u32(w))
	return pos
}

// Points the branch at pos to target (b and bl: 26 bits of words, tb(n)z:
// 14, b.cond and cb(n)z: 19); one out of range marks the program
// unsupported.
@(private)
a_patch :: proc(e: ^Jit_Emitter, pos, target: int) {
	w := u32(e.buf[pos]) | u32(e.buf[pos + 1]) << 8 | u32(e.buf[pos + 2]) << 16 | u32(e.buf[pos + 3]) << 24
	d := (target - pos) / 4
	if (w & 0x7c000000) == 0x14000000 {
		if d < -(1 << 25) || d >= (1 << 25) {
			e.unsupported = true
			return
		}
		w = (w & 0xfc000000) | (u32(d) & 0x03ff_ffff)
	} else if (w & 0x7e000000) == 0x36000000 {
		if d < -(1 << 13) || d >= (1 << 13) {
			e.unsupported = true
			return
		}
		w = (w & 0xfff8001f) | (u32(d) & 0x3fff) << 5
	} else {
		if d < -(1 << 18) || d >= (1 << 18) {
			e.unsupported = true
			return
		}
		w = (w & 0xff00001f) | (u32(d) & 0x7ffff) << 5
	}
	patch_i32(e.buf[:], pos, i32(w))
}

@(private)
a_land :: #force_inline proc(e: ^Jit_Emitter, pos: int) {
	a_patch(e, pos, len(e.buf))
}

// The exec mask's lanes in register 7 (v14, v15).
@(private)
a_exec_lanes :: proc(e: ^Jit_Emitter) -> u8 {
	if !e.exec_ymask_valid {
		ai(e, 0x8b000000 | u32(A_EXEC) << 16 | 5 << 10 | u32(A_TAB) << 5 | u32(A_X16)) // add x16, x23, x22, lsl #5
		ai(e, 0xad400000 | 15 << 10 | u32(A_X16) << 5 | 14)                              // ldp q14, q15, [x16]
		e.exec_ymask_valid = true
	}
	return 7
}

// w = the bit mask of pair r's lanes (all ones or zero each), through pair t.
@(private)
a_movmsk :: proc(e: ^Jit_Emitter, w, r, t: u8) {
	a_v3(e, V_UZP1_8H, 2 * t, 2 * r, 2 * r + 1)
	a_tab(e, 2 * t + 1, A64_OFF_BITS8)
	a_v3(e, V_AND, 2 * t, 2 * t, 2 * t + 1)
	a_v2(e, V_ADDV_8H, 2 * t, 2 * t)
	ai(e, 0x0e023c00 | u32(2 * t) << 5 | u32(w)) // umov w, v.h[0]
}

// w0 = all ones if every lane of pair r has lane 0's value, else 0; w =
// lane 0's value (its x zero-extended). Clobbers v2..v4.
@(private)
a_uniform_lane0 :: proc(e: ^Jit_Emitter, r, w: u8) {
	ai(e, 0x4e040400 | u32(2 * r) << 5 | 4) // dup v4.4s, v(2r).s[0]
	a_v3(e, V_CMEQ, 2, 2 * r, 4)
	a_v3(e, V_CMEQ, 3, 2 * r + 1, 4)
	a_v3(e, V_AND, 2, 2, 3)
	a_v2(e, V_UMINV, 2, 2)
	ai(e, 0x1e260000 | 2 << 5 | u32(A_X0))       // fmov w0, s2
	ai(e, 0x0e043c00 | u32(2 * r) << 5 | u32(w)) // umov w, v(2r).s[0]
}

// w0 = mask m.
@(private)
a_load_mask_w0 :: proc(e: ^Jit_Emitter, m: i32) {
	if is_mask_eq_exec(e, m) {
		a_mov_ww(e, A_X0, A_EXEC)
	} else {
		a_ldrb_mask(e, A_X0, m)
	}
}

@(private)
a_store_mask_w0 :: proc(e: ^Jit_Emitter, m: i32) {
	if m == MASK_EXEC {
		a_mov_ww(e, A_EXEC, A_X0)
	} else {
		a_strb_mask(e, A_X0, m)
	}
}

@(private)
a_copy_mask :: proc(e: ^Jit_Emitter, dst, a: i32) {
	if dst == a || (dst == MASK_EXEC && is_mask_eq_exec(e, a)) {
		return
	}
	if is_mask_eq_exec(e, a) {
		a_strb_mask(e, A_EXEC, dst)
		e.mask_eq_gen[dst] = e.exec_gen
	} else if dst == MASK_EXEC {
		a_ldrb_mask(e, A_EXEC, a)
		invalidate_mask_write(e, MASK_EXEC)
		e.mask_eq_gen[a] = e.exec_gen
	} else {
		a_load_mask_w0(e, a)
		a_store_mask_w0(e, dst)
		e.mask_eq_gen[dst] = 0
	}
}

@(private)
a_finish_fused_cmp :: proc(e: ^Jit_Emitter, cond: i32, pc_mc: int, inv: bool) {
	for &cs in e.cache {
		if cs.slot == cond {
			cs = {slot = -1, last_use = -1, dirty = false, stamp = 0}
		}
	}
	a_movmsk(e, A_X1, 0, 2)
	e.fused_mc_pc = i32(pc_mc)
	e.fused_mc_inv = inv
}

// fn(ctx, in_ptr, arg) under AAPCS64: x0 = ctx, x1 = the Instr, w2 = arg
// (the exec mask when arg < 0).
@(private)
a_call :: proc(e: ^Jit_Emitter, fn: rawptr, in_ptr: ^Instr, arg: i32 = -1) {
	a_mov_xx(e, A_X0, A_CTX)
	a_mov_imm(e, A_X1, u64(uintptr(in_ptr)))
	if arg < 0 {
		a_mov_ww(e, A_X2, A_EXEC)
	} else {
		a_mov_w(e, A_X2, u32(arg))
	}
	a_mov_imm(e, A_X16, u64(uintptr(fn)))
	ai(e, 0xd63f0000 | u32(A_X16) << 5) // blr x16
}

// A call inside a basic block (as emit_math_call on x86): the dirty cached
// slots it reads stored, the cache registers live after this instruction
// saved around it on the stack (AAPCS64 keeps only the low halves of
// v8..v15), the exec lanes reloaded after. The slots [dst_lo, dst_lo +
// dst_n) it writes leave the cache.
@(private)
a_call_in_block :: proc(e: ^Jit_Emitter, fn: rawptr, in_ptr: ^Instr, arg: i32, reads: []Slots, dst_lo, dst_n: i32) {
	for &cs, idx in e.cache {
		if cs.slot < 0 || !cs.dirty {
			continue
		}
		for r in reads {
			if cs.slot >= r.start && cs.slot < r.start + r.n {
				jit_st_slot(e, cs.slot, u8(8 + idx))
				cs.dirty = false
				break
			}
		}
	}
	save_regs: [8]u8
	n_save := 0
	for &cs, idx in e.cache {
		if cs.slot < 0 {
			continue
		}
		if (cs.slot >= dst_lo && cs.slot < dst_lo + dst_n) || cs.last_use <= e.pc {
			cs = {slot = -1, last_use = -1, dirty = false, stamp = 0}
		} else {
			save_regs[n_save] = u8(8 + idx)
			n_save += 1
		}
	}
	frame := u32(n_save * 32)
	if n_save > 0 {
		ai(e, 0xd10003ff | frame << 10) // sub sp, sp, #frame
		for k in 0 ..< n_save {
			a_pair_mem(e, false, save_regs[k], A_SP, i64(k * 32))
		}
	}
	a_call(e, fn, in_ptr, arg)
	if n_save > 0 {
		for k in 0 ..< n_save {
			a_pair_mem(e, true, save_regs[k], A_SP, i64(k * 32))
		}
		ai(e, 0x910003ff | frame << 10) // add sp, sp, #frame
	}
	e.exec_ymask_valid = false
}

// Component i of a vm_math.odin built-in through the interpreter's math_8,
// into ctx.math_out.
@(private, enable_target_feature = JIT_FEATURE)
jit_help_math_a64 :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, i: i32) #no_bounds_check {
	math_8(in_.op, Builtin(in_.k), &ctx.slots[in_.a + i], &ctx.slots[in_.b + i], (^[8]u32)(&ctx.math_out))
}

@(private)
a_math_call :: proc(e: ^Jit_Emitter, in_ptr: ^Instr, i: i32) {
	in_ := in_ptr^
	n := i32(in_.n)
	reads := [2]Slots{{in_.a, n}, {in_.b, n if in_.op == .F_Fn2 else 0}}
	a_call_in_block(e, rawptr(jit_help_math_a64), in_ptr, i, reads[:], in_.dst + i, 1)
	a_pair_mem(e, true, 0, A_CTX, i64(OFF_CTX_MATH_OUT))
	copy_to_dst(e, in_.dst + i, 0)
}

// ---- bound channels -----------------------------------------------------------------------
//
// The rule of both engines (Channel_Filter, vm_run.odin), in x86's routes:
// sample_sub (texture(), textureLod, textureGrad: one level Nearest with
// both axes clamped in registers, else every level a lane uses, ascending,
// its taps decoded and blended), fetch_sub (texelFetch of levels or
// layers) and, in each texelFetch, one level inline; decode_sub reads
// texels. A texture instruction stays in its basic block, so the
// subroutines preserve v16..v31 (the slot cache), x17 and x19..x23, take
// v0..v15 and x0..x16, and keep their per-lane state in those or a frame
// on the stack.
//
// NEON has no gather: each lane's texel index goes to w3..w10 (umov) and
// each lane is loaded alone, with an ldr of that register offset (the
// element's shift applied), all eight issued before the zips that gather
// them; ldr q of a whole texel and a 4x4 transpose for F32x4. A lane that
// reads nothing (texelFetch out of range) takes index 0, a texel every
// image has, and its result is masked to 0 after.
// Coordinates scale in f64 (fcvtl), as on x86 without FMA3, exactly.

@(private) V_FCVTL_2D :: u32(0x0e617800)
@(private) V_FCVTL2_2D :: u32(0x4e617800)
@(private) V_FCVTL_4S :: u32(0x0e217800)
@(private) V_FRINTM_2D :: u32(0x4e619800)
@(private) V_FRINTN_2D :: u32(0x4e618800)
@(private) V_FMUL_2D :: u32(0x6e60dc00)
@(private) V_FSUB_2D :: u32(0x4ee0d400)
@(private) V_FMAXNM_2D :: u32(0x4e60c400)
@(private) V_FMINNM_2D :: u32(0x4ee0c400)
@(private) V_FCMGE_2D :: u32(0x6e60e400)
@(private) V_TRN1_2D :: u32(0x4ec02800)
@(private) V_TRN2_2D :: u32(0x4ec06800)
@(private) V_UZP1_4S :: u32(0x4e801800)
@(private) V_UZP2_4S :: u32(0x4e805800)
@(private) V_UZP2_8H :: u32(0x4e405800)
@(private) MOVI_255 :: u32(0x4f0707e0)
@(private) MOVI_ONE :: u32(0x4f000420)

// umov w, v.s[i].
@(private) a_umov_s :: #force_inline proc(e: ^Jit_Emitter, w, v: u8, i: u32) {ai(e, 0x0e043c00 | ((i << 3) | 4) << 16 | u32(v) << 5 | u32(w))}
// ldr q, x, w, h, b with an unsigned offset; str x.
@(private) a_ldr_q :: #force_inline proc(e: ^Jit_Emitter, q, x: u8, off: i32) {ai(e, 0x3dc00000 | u32(off / 16) << 10 | u32(x) << 5 | u32(q))}
@(private) a_ldr_x :: #force_inline proc(e: ^Jit_Emitter, t, x: u8, off: i32) {ai(e, 0xf9400000 | u32(off / 8) << 10 | u32(x) << 5 | u32(t))}
@(private) a_str_x :: #force_inline proc(e: ^Jit_Emitter, t, x: u8, off: i32) {ai(e, 0xf9000000 | u32(off / 8) << 10 | u32(x) << 5 | u32(t))}
@(private) a_ldr_w :: #force_inline proc(e: ^Jit_Emitter, t, x: u8, off: i32) {ai(e, 0xb9400000 | u32(off / 4) << 10 | u32(x) << 5 | u32(t))}
@(private) a_ldrb :: #force_inline proc(e: ^Jit_Emitter, t, x: u8, off: i32) {ai(e, 0x39400000 | u32(off) << 10 | u32(x) << 5 | u32(t))}
// add xd, xn, #imm (imm < 4096).
@(private) a_add_imm :: #force_inline proc(e: ^Jit_Emitter, d, n: u8, imm: i32) {ai(e, 0x91000000 | u32(imm) << 10 | u32(n) << 5 | u32(d))}
// cmp w, #imm.
@(private) a_cmp_w :: #force_inline proc(e: ^Jit_Emitter, w: u8, imm: u32) {ai(e, 0x7100001f | imm << 10 | u32(w) << 5)}
// dup v.4s, w and dup v.2d, x.
@(private) a_dup_s :: #force_inline proc(e: ^Jit_Emitter, v, w: u8) {ai(e, 0x4e040c00 | u32(w) << 5 | u32(v))}
@(private) a_dup_d :: #force_inline proc(e: ^Jit_Emitter, v, x: u8) {ai(e, 0x4e080c00 | u32(x) << 5 | u32(v))}
// v.4s = the word at [x + off] (through w16), and v.2d the doubleword.
@(private) a_bcast_w :: #force_inline proc(e: ^Jit_Emitter, v, x: u8, off: i32) {a_ldr_w(e, A_X16, x, off); a_dup_s(e, v, A_X16)}
@(private) a_bcast_d :: #force_inline proc(e: ^Jit_Emitter, v, x: u8, off: i32) {a_ldr_x(e, A_X16, x, off); a_dup_d(e, v, A_X16)}

// tbz w, #bit (tbnz when `set`), patched by a_land.
@(private)
a_tbz :: proc(e: ^Jit_Emitter, w: u8, bit: u32, set: bool) -> int {
	pos := len(e.buf)
	ai(e, (0x37000000 if set else 0x36000000) | bit << 19 | u32(w))
	return pos
}

// bl to a subroutine emitted later: its position in `calls`.
@(private)
a_bl :: proc(e: ^Jit_Emitter, calls: ^[dynamic]int) {
	append(calls, len(e.buf))
	ai(e, 0x94000000)
}


// w = the OR of pair r's eight words, through pair t.
@(private)
a_or_lanes :: proc(e: ^Jit_Emitter, w, r, t: u8) {
	a_v3(e, V_ORR, 2 * t, 2 * r, 2 * r + 1)
	a_v3(e, 0x6e004000, 2 * t + 1, 2 * t, 2 * t) // ext #8
	a_v3(e, V_ORR, 2 * t, 2 * t, 2 * t + 1)
	a_v3(e, 0x6e002000, 2 * t + 1, 2 * t, 2 * t) // ext #4
	a_v3(e, V_ORR, 2 * t, 2 * t, 2 * t + 1)
	a_umov_s(e, w, 2 * t, 0)
}

// The level loop's head: w14 = the lowest level in w13, cleared from it;
// x2 = that level's record.
@(private)
a_next_level :: proc(e: ^Jit_Emitter) {
	ai(e, 0x5ac001ae) // rbit w14, w13
	ai(e, 0x5ac011ce) // clz w14, w14
	ai(e, 0x510005af) // sub w15, w13, #1
	ai(e, 0x0a0f01ad) // and w13, w13, w15
	a_add_imm(e, A_X2, A_X0, OFF_CI_LEVELS)
	ai(e, 0x8b0e1842) // add x2, x2, x14, lsl #6
}

// w3..w10 = lane l of pair r (texel indexes, for the loads' register offsets).
@(private)
a_lane_idx :: proc(e: ^Jit_Emitter, r: u8) {
	for l in u32(0) ..< 8 {
		a_umov_s(e, u8(3 + l), 2 * r + u8(l / 4), l % 4)
	}
}

// Register-offset loads of lane l's element at [x15 + w(3 + l) << shift]
// into a whole V register (the rest zeroed): ldr b, h, s, d, q.
@(private) A_LDR_B :: u32(0x3c604800)
@(private) A_LDR_H :: u32(0x7c605800)
@(private) A_LDR_S :: u32(0xbc605800)
@(private) A_LDR_D :: u32(0xfc605800)
@(private) A_LDR_Q :: u32(0x3ce05800)
@(private) V_ZIP1_4S :: u32(0x4e803800)

@(private)
a_ldr_lane :: #force_inline proc(e: ^Jit_Emitter, op: u32, v: u8, l: u32) {
	ai(e, op | u32(3 + l) << 16 | 15 << 5 | u32(v))
}

// Sets w11 = w1 (requested comp_mask | flags) and w1 = w11 & img.held
// (the components to actually decode/blend). Clobbers x16.
@(private)
a_live_held :: proc(e: ^Jit_Emitter) {
	a_mov_ww(e, 11, A_X1)
	a_ldrb(e, A_X16, A_X0, OFF_CI_HELD)
	ai(e, 0x0a100161) // and w1, w11, w16
}

// Pair d (0 or 6) = each lane's element of up to 4 bytes (`op`: ldr b, h or s),
// issuing all eight independent scalar loads into v4..v11 before zipping.
// Preserves v14..v31.
@(private)
a_gather_s8 :: proc(e: ^Jit_Emitter, op: u32, d: u8) {
	assert(d == 0 || d == 6)
	for i in u32(0) ..< 8 {
		a_ldr_lane(e, op, 4 + u8(i), i)
	}
	a_v3(e, V_ZIP1_4S, 4, 4, 5)
	a_v3(e, V_ZIP1_4S, 6, 6, 7)
	a_v3(e, V_ZIP1_2D, 2 * d, 4, 6)
	a_v3(e, V_ZIP1_4S, 8, 8, 9)
	a_v3(e, V_ZIP1_4S, 10, 10, 11)
	a_v3(e, V_ZIP1_2D, 2 * d + 1, 8, 10)
}

// Half h's four lanes' doublewords into v12 (lanes 0, 1) and v13 (2, 3).
// Clobbers v8..v11. Preserves v14..v31.
@(private)
a_gather_d :: proc(e: ^Jit_Emitter, h: u32) {
	for i in u32(0) ..< 4 {
		a_ldr_lane(e, A_LDR_D, 8 + u8(i), 4 * h + i)
	}
	a_v3(e, V_ZIP1_2D, 12, 8, 9)
	a_v3(e, V_ZIP1_2D, 13, 10, 11)
}

// decode_sub: the texels at in-range indexes pair 4 of level x2 of image x0
// as stored values in f32 (the unsigned normalized formats' integers,
// halves converted by fcvtl, which gives f16_to_f32's bits), for components
// present in w1 (`comp_mask & img.held`), into pairs 0..3. Each lane's
// element is its own register-offset load, independent of the others.
// Preserves v14..v31, x0, x1, x2, x11..x14 and x17..x28.
@(private)
a_emit_decode_sub :: proc(e: ^Jit_Emitter) {
	a_ldr_x(e, 15, A_X2, OFF_LV_PTR)
	a_lane_idx(e, 4)
	a_ldrb(e, A_X16, A_X0, OFF_CI_FORMAT)
	not_f32 := a_cbz(e, A_X16, true)
	// F32: 1, 2 or 4 words a texel.
	a_ldrb(e, A_X16, A_X0, OFF_CI_COMPS)
	a_cmp_w(e, A_X16, 2)
	c2 := a_branch(e, 0)  // b.eq
	c4 := a_branch(e, 8)  // b.hi
	skip1 := a_cbz(e, A_X1, false)
	a_gather_s8(e, A_LDR_S, 0)
	a_land(e, skip1)
	ai(e, 0xd65f03c0)

	a_land(e, c2)
	skip2 := a_cbz(e, A_X1, false)
	for h in u32(0) ..< 2 {
		a_gather_d(e, h)
		s0 := a_tbz(e, A_X1, 0, false)
		a_v3(e, V_UZP1_4S, u8(h), 12, 13)
		a_land(e, s0)
		s1 := a_tbz(e, A_X1, 1, false)
		a_v3(e, V_UZP2_4S, 2 + u8(h), 12, 13)
		a_land(e, s1)
	}
	a_land(e, skip2)
	ai(e, 0xd65f03c0)

	a_land(e, c4)
	skip4 := a_cbz(e, A_X1, false)
	for h in u8(0) ..< 2 {
		for i in u8(0) ..< 4 {
			a_ldr_lane(e, A_LDR_Q, 8 + i, u32(4 * h + i))
		}
		a_v3(e, V_TRN1, 12, 8, 9)
		a_v3(e, V_TRN2, 13, 8, 9)
		a_v3(e, V_TRN1, 8, 10, 11)
		a_v3(e, V_TRN2, 9, 10, 11)
		s0 := a_tbz(e, A_X1, 0, false); a_v3(e, V_TRN1_2D, h, 12, 8);     a_land(e, s0)
		s2 := a_tbz(e, A_X1, 2, false); a_v3(e, V_TRN2_2D, 4 + h, 12, 8); a_land(e, s2)
		s1 := a_tbz(e, A_X1, 1, false); a_v3(e, V_TRN1_2D, 2 + h, 13, 9); a_land(e, s1)
		s3 := a_tbz(e, A_X1, 3, false); a_v3(e, V_TRN2_2D, 6 + h, 13, 9); a_land(e, s3)
	}
	a_land(e, skip4)
	ai(e, 0xd65f03c0)

	a_land(e, not_f32)
	a_cmp_w(e, A_X16, u32(Channel_Format.Rgba8))
	not_rgba8 := a_branch(e, 1) // b.ne
	// Rgba8: byte k of the texel's word.
	skip8 := a_cbz(e, A_X1, false)
	a_gather_s8(e, A_LDR_S, 6)
	a_imm(e, MOVI_255, 11)
	a_cmp_w(e, A_X1, 0xf)
	partial8 := a_branch(e, 1)
	p2(e, 0x6f000400 | (64 - 8) << 16, 1, 6)
	p2(e, 0x6f000400 | (64 - 16) << 16, 2, 6)
	p2(e, 0x6f000400 | (64 - 24) << 16, 3, 6)
	p3q(e, V_AND, 0, 6, 11)
	p3q(e, V_AND, 1, 1, 11)
	p3q(e, V_AND, 2, 2, 11)
	for k in u8(0) ..< 4 {
		p2(e, V_UCVTF, k, k)
	}
	a_land(e, skip8)
	ai(e, 0xd65f03c0)

	a_land(e, partial8)
	for k in u32(0) ..< 4 {
		sk := a_tbz(e, A_X1, k, false)
		if k > 0 {
			p2(e, 0x6f000400 | (64 - 8 * k) << 16, u8(k), 6)
		}
		if k == 0 {
			p3q(e, V_AND, 0, 6, 11)
		} else if k < 3 {
			p3q(e, V_AND, u8(k), u8(k), 11)
		}
		p2(e, V_UCVTF, u8(k), u8(k))
		a_land(e, sk)
	}
	ai(e, 0xd65f03c0)

	a_land(e, not_rgba8)
	a_cmp_w(e, A_X16, u32(Channel_Format.Rgba16f))
	not_half := a_branch(e, 1)
	// Rgba16f: each texel's four halves, deinterleaved (uzp), converted.
	skiph := a_cbz(e, A_X1, false)
	for h in u8(0) ..< 2 {
		a_gather_d(e, u32(h))
		a_v3(e, V_UZP1_8H, 10, 12, 13) // halves 0 and 2
		a_v3(e, V_UZP2_8H, 11, 12, 13) // 1 and 3
		s0 := a_tbz(e, A_X1, 0, false); a_v3(e, V_UZP1_8H, 12, 10, 10); a_v2(e, V_FCVTL_4S, h, 12);     a_land(e, s0)
		s2 := a_tbz(e, A_X1, 2, false); a_v3(e, V_UZP2_8H, 12, 10, 10); a_v2(e, V_FCVTL_4S, 4 + h, 12); a_land(e, s2)
		s1 := a_tbz(e, A_X1, 1, false); a_v3(e, V_UZP1_8H, 12, 11, 11); a_v2(e, V_FCVTL_4S, 2 + h, 12); a_land(e, s1)
		s3 := a_tbz(e, A_X1, 3, false); a_v3(e, V_UZP2_8H, 12, 11, 11); a_v2(e, V_FCVTL_4S, 6 + h, 12); a_land(e, s3)
	}
	a_land(e, skiph)
	ai(e, 0xd65f03c0)

	// Gray8 and Gray16: byte or halfword mis + index (* 2) of the level.
	a_land(e, not_half)
	skipg := a_cbz(e, A_X1, false)
	a_cmp_w(e, A_X16, u32(Channel_Format.Gray16))
	g16 := a_branch(e, 0)
	for gray in 0 ..< 2 {
		if gray == 1 {
			a_land(e, g16)
		}
		a_ldr_w(e, A_X16, A_X2, OFF_LV_MIS)
		ai(e, 0x8b3041ef) // add x15, x15, w16, uxtw
		a_gather_s8(e, A_LDR_B if gray == 0 else A_LDR_H, 0)
		p2(e, V_UCVTF, 0, 0)
		if gray == 0 {
			ai(e, 0xd65f03c0)
		}
	}
	a_land(e, skipg)
	ai(e, 0xd65f03c0)
}

// After a decode: the unsigned normalized formats' live components (w1) /
// 255 or 65535; lacking live components (w11 ^ w1) set to (0, 0, 1).
// Preserves v14..v31. Clobbers v12 and w15..w16.
@(private)
a_chan_finish :: proc(e: ^Jit_Emitter) {
	a_ldr_w(e, 15, A_X0, OFF_CI_UNORM)
	float := a_cbz(e, 15, false)
	a_dup_s(e, 12, 15)
	a_cmp_w(e, A_X1, 0xf)
	partial := a_branch(e, 1)
	for k in u8(0) ..< 4 {
		p3q(e, V_FDIV, k, k, 12)
	}
	done := a_branch(e, -1)
	a_land(e, partial)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		p3q(e, V_FDIV, k, k, 12)
		a_land(e, sk)
	}
	a_land(e, done)
	a_land(e, float)
	ai(e, 0x4a010170) // eor w16, w11, w1
	s1 := a_tbz(e, A_X16, 1, false); pzero(e, 1);                          a_land(e, s1)
	s2 := a_tbz(e, A_X16, 2, false); pzero(e, 2);                          a_land(e, s2)
	s3 := a_tbz(e, A_X16, 3, false); a_imm(e, FMOV_ONE, 6); a_imm(e, FMOV_ONE, 7); a_land(e, s3)
}

// Nearest, both axes clamped, level x2: pairs 0 (u) and 1 (v) to their
// texel indexes clamp(floor(f64(c) * n), 0, n - 1). Clobbers v8..v11, v14
// and v15.
@(private)
a_chan_nearest_xy :: proc(e: ^Jit_Emitter) {
	for ax in u8(0) ..< 2 {
		a_bcast_d(e, 14, A_X2, OFF_LV_FS + i32(ax) * 8)
		a_v2(e, V_FCVTL_2D, 8, 2 * ax)
		a_v2(e, V_FCVTL2_2D, 9, 2 * ax)
		a_v2(e, V_FCVTL_2D, 10, 2 * ax + 1)
		a_v2(e, V_FCVTL2_2D, 11, 2 * ax + 1)
		for d in u8(8) ..< 12 {
			a_v3(e, V_FMUL_2D, d, d, 14)
			a_v2(e, V_FRINTM_2D, d, d)
		}
		a_tab(e, 15, A64_OFF_DLO)
		for d in u8(8) ..< 12 {a_v3(e, V_FMAXNM_2D, d, d, 15)}
		a_tab(e, 15, A64_OFF_DHI)
		for d in u8(8) ..< 12 {
			a_v3(e, V_FMINNM_2D, d, d, 15)
			a_v2(e, V_FCVTZS_2D, d, d)
		}
		a_v2(e, V_XTN, 2 * ax, 8)
		a_v2(e, V_XTN2, 2 * ax, 9)
		a_v2(e, V_XTN, 2 * ax + 1, 10)
		a_v2(e, V_XTN2, 2 * ax + 1, 11)
		a_imm(e, V_ZERO, 15)
		a_bcast_w(e, 14, A_X2, OFF_LV_MAX + i32(ax) * 4)
		p3q(e, V_SMAX, ax, ax, 15)
		p3q(e, V_SMIN, ax, ax, 14)
	}
}

// sample_sub's general-path stack frame (32-byte slots from sp; short route
// uses only a 16-byte frame for x30).
@(private) A_SF_U :: i64(0)
@(private) A_SF_V :: i64(32)
@(private) A_SF_L0 :: i64(64)    // each lane's level l0
@(private) A_SF_WL :: i64(96)    // and l0 + 1's weight of 256
@(private) A_SF_LAYER :: i64(128) // each lane's layer
@(private) A_SF_B3 :: i64(160)   // bottom-row component 3 spill across Corner 3 decode
@(private) A_SF_C0 :: i64(192)   // [4] each lane's level l0 tap
@(private) A_SF_C1 :: i64(320)   // [4] and l0 + 1's
@(private) A_SF_Q :: i64(448)    // [8] saved q16..q31 (pairs 8..15)
@(private) A_SF_GEN :: u32(704)

// One axis of the tap at level x2 from the coordinate at [sp + f_c]: x =
// f64(c) * size (- 0.5 for Linear), leaves texel index floor(x) clamped to
// [-2^31, 2^31 - 2] in pair 6 and, Linear, the next one in pair 5, as the
// axis's wrap addresses them, and the weight round_even(fract(x) * 256) / 256
// in pair 4. Rows (ax 1) times the pitch, plus the layer's offset with layers.
// Preserves v16..v31, w1, w11..w14, x0 and x2.
@(private)
a_chan_axis :: proc(e: ^Jit_Emitter, ax: i32, f_c: i64, nearest: bool, int_weight := false) {
	a_pair_mem(e, true, 0, A_SP, f_c)
	a_bcast_d(e, 14, A_X2, OFF_LV_FS + ax * 8)
	a_v2(e, V_FCVTL_2D, 2, 0)
	a_v2(e, V_FCVTL2_2D, 3, 0)
	a_v2(e, V_FCVTL_2D, 4, 1)
	a_v2(e, V_FCVTL2_2D, 5, 1)
	for d in u8(2) ..< 6 {a_v3(e, V_FMUL_2D, d, d, 14)}
	if !nearest {
		a_tab(e, 15, A64_OFF_DHALF)
		for d in u8(2) ..< 6 {a_v3(e, V_FSUB_2D, d, d, 15)}
	}
	for d in u8(2) ..< 6 {a_v2(e, V_FRINTM_2D, d + 4, d)} // g in v6..v9
	if !nearest {
		// The weight: round_even((x - g) * 256) (NaN: 0), in v2..v5.
		a_tab(e, 15, A64_OFF_D256)
		for d in u8(2) ..< 6 {a_v3(e, V_FSUB_2D, d, d, d + 4)}
		for d in u8(2) ..< 6 {a_v3(e, V_FMUL_2D, d, d, 15)}
		for d in u8(2) ..< 6 {a_v2(e, V_FRINTN_2D, d, d)}
		for d in u8(2) ..< 6 {a_v2(e, V_FCVTZS_2D, d, d)}
	}
	a_tab(e, 15, A64_OFF_DLO) // NaN: -2^31
	for g in u8(6) ..< 10 {a_v3(e, V_FMAXNM_2D, g, g, 15)}
	a_tab(e, 15, A64_OFF_DHI)
	for g in u8(6) ..< 10 {a_v3(e, V_FMINNM_2D, g, g, 15)}
	a_ldrb(e, 15, A_X0, OFF_CI_WRAP + ax)
	clamp := a_cbz(e, 15, false)
	// Repeat: g mod n, floored, in f64 (exact): g - n * floor(g / n), with
	// g * (1 / n) at most one step low (where g / n is whole).
	a_bcast_d(e, 15, A_X2, OFF_LV_INV + ax * 8)
	for g in u8(6) ..< 10 {a_v3(e, V_FMUL_2D, g + 4, g, 15)}
	for g in u8(6) ..< 10 {a_v2(e, V_FRINTM_2D, g + 4, g + 4)}
	for g in u8(6) ..< 10 {a_v3(e, V_FMUL_2D, g + 4, g + 4, 14)}
	for g in u8(6) ..< 10 {a_v3(e, V_FSUB_2D, g, g, g + 4)}
	for g in u8(6) ..< 10 {a_v3(e, V_FCMGE_2D, g + 4, g, 14)} // m >= n
	for g in u8(6) ..< 10 {a_v3(e, V_AND, g + 4, g + 4, 14)}
	for g in u8(6) ..< 10 {a_v3(e, V_FSUB_2D, g, g, g + 4)}
	a_chan_axis_i32(e)
	if !nearest {
		// The next one: 0 after n - 1.
		a_imm(e, MOVI_ONE, 15)
		a_bcast_w(e, 14, A_X2, OFF_LV_SIZE + ax * 4)
		p3q(e, V_ADD, 5, 6, 15)
		p3q(e, V_CMEQ, 3, 5, 14)
		p3(e, V_BIC, 5, 5, 3)
	}
	done := a_branch(e, -1)
	a_land(e, clamp)
	a_chan_axis_i32(e)
	a_imm(e, V_ZERO, 15)
	a_bcast_w(e, 14, A_X2, OFF_LV_MAX + ax * 4)
	if !nearest {
		a_imm(e, MOVI_ONE, 1)
		p3q(e, V_ADD, 5, 6, 1)
		p3q(e, V_SMAX, 5, 5, 15)
		p3q(e, V_SMIN, 5, 5, 14)
	}
	p3q(e, V_SMAX, 6, 6, 15)
	p3q(e, V_SMIN, 6, 6, 14)
	a_land(e, done)
	if ax == 1 {
		a_bcast_w(e, 14, A_X2, OFF_LV_PITCH)
		p3q(e, V_MUL, 6, 6, 14)
		if !nearest {
			p3q(e, V_MUL, 5, 5, 14)
		}
		// With layers: + each lane's layer times level w14's layer pitch.
		a_ldr_w(e, A_X16, A_X0, OFF_CI_NLAYERS)
		a_cmp_w(e, A_X16, 1)
		one := a_branch(e, A_LS)
		a_add_imm(e, A_X16, A_X0, OFF_CI_LPITCH)
		ai(e, 0xb86e7a10) // ldr w16, [x16, x14, lsl #2]
		a_dup_s(e, 14, A_X16)
		a_pair_mem(e, true, 4, A_SP, A_SF_LAYER)
		p3q(e, V_MUL, 4, 4, 14)
		p3(e, V_ADD, 6, 6, 4)
		if !nearest {
			p3(e, V_ADD, 5, 5, 4)
		}
		a_land(e, one)
	}
	if !nearest {
		a_v2(e, V_XTN, 8, 2)
		a_v2(e, V_XTN2, 8, 3)
		a_v2(e, V_XTN, 9, 4)
		a_v2(e, V_XTN2, 9, 5)
		if !int_weight {
			p2(e, V_SCVTF, 4, 4)
			a_tab(e, 15, A64_OFF_INV256)
			p3q(e, V_FMUL, 4, 4, 15)
		}
	}
}

// The axis's indexes g (v6..v9, f64) as i32 into pair 6.
@(private)
a_chan_axis_i32 :: proc(e: ^Jit_Emitter) {
	for g in u8(6) ..< 10 {a_v2(e, V_FCVTZS_2D, g, g)}
	a_v2(e, V_XTN, 12, 6)
	a_v2(e, V_XTN2, 12, 7)
	a_v2(e, V_XTN, 13, 8)
	a_v2(e, V_XTN2, 13, 9)
}

// fmul by element: d = n * m.s[i], 4 lanes (q) or 2.
@(private)
a_fmul_elem :: #force_inline proc(e: ^Jit_Emitter, d, n, m: u8, i: u32, q: bool) {
	ai(e, (0x4f809000 if q else 0x0f809000) | (i & 1) << 21 | u32(m >> 4) << 20 | u32(m & 15) << 16 | (i >> 1) << 11 | u32(n) << 5 | u32(d))
}

// One clamped Linear axis for bilinear_sub, in registers: c (pair `c`)
// clamped to [0, 1] (NaN as 0), T = f64(c) * 256n - 128 (below 2^31 for n
// < 2^22), the index floor(T) >> 8 (from -1 to n - 1) clamped to the edge
// and the next one, and the weight round_even(T) - 256 * index as w / 256,
// as bound_clamped_f32 takes them. Into pairs i0, i1 and t; rows (ax 1)
// times the pitch. Clobbers pairs 2..6 and 10, 11, v14 and v15.
@(private)
a_chan_axis_lin :: proc(e: ^Jit_Emitter, ax: i32, c, i0, i1, t: u8) {
	p2(e, V_FCMGT0, 6, c)
	p3(e, V_AND, 6, c, 6)
	a_imm(e, FMOV_ONE, 14)
	p3q(e, V_FMINNM, 6, 6, 14)
	a_bcast_d(e, 14, A_X2, OFF_LV_FS + ax * 8)
	a_tab(e, 15, A64_OFF_D256)
	a_v3(e, V_FMUL_2D, 14, 14, 15)
	a_tab(e, 15, A64_OFF_D128)
	a_v2(e, V_FCVTL_2D, 8, 12)
	a_v2(e, V_FCVTL2_2D, 9, 12)
	a_v2(e, V_FCVTL_2D, 10, 13)
	a_v2(e, V_FCVTL2_2D, 11, 13)
	for d in u8(8) ..< 12 {
		a_v3(e, V_FMUL_2D, d, d, 14)
		a_v3(e, V_FSUB_2D, d, d, 15)
		a_v2(e, V_FRINTM_2D, d + 12, d) // floor(T) in v20..v23
		a_v2(e, V_FRINTN_2D, d, d)      // round_even(T)
		a_v2(e, V_FCVTZS_2D, d + 12, d + 12)
		a_v2(e, V_FCVTZS_2D, d, d)
	}
	a_v2(e, V_XTN, 4, 20)
	a_v2(e, V_XTN2, 4, 21)
	a_v2(e, V_XTN, 5, 22)
	a_v2(e, V_XTN2, 5, 23)
	a_v2(e, V_XTN, 6, 8)
	a_v2(e, V_XTN2, 6, 9)
	a_v2(e, V_XTN, 7, 10)
	a_v2(e, V_XTN2, 7, 11)
	p2(e, 0x4f380400, 2, 2)  // sshr #8: the index
	p2(e, 0x4f285400, 4, 2)  // shl #8
	p3(e, V_SUB, 3, 3, 4)    // the weight
	p2(e, V_SCVTF, t, 3)
	a_tab(e, 15, A64_OFF_INV256)
	p3q(e, V_FMUL, t, t, 15)
	a_imm(e, V_ZERO, 15)
	a_imm(e, MOVI_ONE, 14)
	p3q(e, V_ADD, 4, 2, 14)
	p3q(e, V_SMAX, i0, 2, 15)
	a_bcast_w(e, 14, A_X2, OFF_LV_MAX + ax * 4)
	p3q(e, V_SMIN, i1, 4, 14)
	if ax == 1 {
		a_bcast_w(e, 14, A_X2, OFF_LV_PITCH)
		p3q(e, V_MUL, i0, i0, 14)
		p3q(e, V_MUL, i1, i1, 14)
	}
}

// bilinear_sub: sample_sub's Linear for F32 with one level and both axes
// clamped (Chan_Img.lin_route 0), in registers: F32x4 and F32x2 a lane at a
// time (each corner's texel one ldr, blended by the lane's weights by
// element, then transposed), F32x1 a component across the lanes.
//   in:  pair 0 = u, 1 = v, x0 = ^Chan_Img, w1 = live components
//   out: pairs 0..3; preserves v16..v31.
@(private)
a_emit_bilinear_sub :: proc(e: ^Jit_Emitter) {
	a_live_held(e)
	ai(e, 0xd10403ff) // sub sp, sp, #256
	for r in u8(8) ..< 16 {
		a_pair_mem(e, false, r, A_SP, i64(r - 8) * 32)
	}
	a_add_imm(e, A_X2, A_X0, OFF_CI_LEVELS)
	a_chan_axis_lin(e, 0, 0, 8, 9, 12)
	a_chan_axis_lin(e, 1, 1, 10, 11, 13)
	// The corners' texel indexes: 00 in pair 14, 10 in 15, 01 in 8, 11 in 9.
	p3(e, V_ADD, 14, 10, 8)
	p3(e, V_ADD, 15, 10, 9)
	p3(e, V_ADD, 8, 11, 8)
	p3(e, V_ADD, 9, 11, 9)
	a_ldr_x(e, 15, A_X2, OFF_LV_PTR)
	a_ldrb(e, A_X16, A_X0, OFF_CI_COMPS)
	a_cmp_w(e, A_X16, 2)
	c2 := a_branch(e, 0)
	c4 := a_branch(e, 8)
	// F32x1: each corner's word across the lanes (pairs 2, 3, 6, 7), then
	// the blends (skipped if component 0 is dead).
	skip1 := a_tbz(e, A_X1, 0, false)
	corners := [4]u8{14, 15, 8, 9}
	for cr, ci in corners {
		a_lane_idx(e, cr)
		a_gather_s8(e, A_LDR_S, 0)
		if ci < 3 {
			pmov(e, cr, 0)
		}
	}
	p3(e, V_FSUB, 4, 15, 14)
	p3(e, V_FMUL, 4, 4, 12)
	p3(e, V_FADD, 2, 14, 4) // top
	p3(e, V_FSUB, 4, 0, 8)
	p3(e, V_FMUL, 4, 4, 12)
	p3(e, V_FADD, 6, 8, 4)  // bottom
	p3(e, V_FSUB, 4, 6, 2)
	p3(e, V_FMUL, 4, 4, 13)
	p3(e, V_FADD, 0, 2, 4)
	a_land(e, skip1)
	one_done := a_branch(e, -1)
	// F32x2 and F32x4: each lane's four texels (v4..v7), blended into v(8 + l).
	done2 := 0
	for wide in 0 ..< 2 {
		a_land(e, c2 if wide == 0 else c4)
		q := wide == 1
		ld := u32(0x3ce05800) if q else 0xfc605800 // ldr q / ldr d, [x15, w, uxtw #4 / #3]
		sub := V_FSUB if q else 0x0ea0d400
		add := V_FADD if q else 0x0e20d400
		for l in u32(0) ..< 8 {
			h, i := u8(l / 4), l % 4
			for cr, ci in corners {
				a_umov_s(e, A_X3 + u8(ci), 2 * cr + h, i)
				ai(e, ld | u32(A_X3 + u8(ci)) << 16 | 15 << 5 | u32(4 + ci))
			}
			a_v3(e, sub, 20, 5, 4)
			a_fmul_elem(e, 20, 20, 24 + h, i, q)
			a_v3(e, add, 20, 4, 20) // top
			a_v3(e, sub, 21, 7, 6)
			a_fmul_elem(e, 21, 21, 24 + h, i, q)
			a_v3(e, add, 21, 6, 21) // bottom
			a_v3(e, sub, 21, 21, 20)
			a_fmul_elem(e, 21, 21, 26 + h, i, q)
			a_v3(e, add, 8 + u8(l), 20, 21)
		}
		for h in u8(0) ..< 2 {
			b := 8 + 4 * h
			if q {
				a_v3(e, V_TRN1, 20, b, b + 1)
				a_v3(e, V_TRN2, 21, b, b + 1)
				a_v3(e, V_TRN1, 22, b + 2, b + 3)
				a_v3(e, V_TRN2, 23, b + 2, b + 3)
				a_v3(e, V_TRN1_2D, h, 20, 22)
				a_v3(e, V_TRN2_2D, 4 + h, 20, 22)
				a_v3(e, V_TRN1_2D, 2 + h, 21, 23)
				a_v3(e, V_TRN2_2D, 6 + h, 21, 23)
			} else {
				a_v3(e, V_ZIP1_2D, 20, b, b + 1)
				a_v3(e, V_ZIP1_2D, 21, b + 2, b + 3)
				a_v3(e, V_UZP1_4S, h, 20, 21)
				a_v3(e, V_UZP2_4S, 2 + h, 20, 21)
			}
		}
		if wide == 0 {
			done2 = a_branch(e, -1)
		}
	}
	a_land(e, done2)
	a_land(e, one_done)
	a_chan_finish(e)
	for r in u8(8) ..< 16 {
		a_pair_mem(e, true, r, A_SP, i64(r - 8) * 32)
	}
	ai(e, 0x910403ff) // add sp, sp, #256
	ai(e, 0xd65f03c0)
}

// The level each quad's derivatives pick in level 0's texels, added to pair
// 2 (texture()'s bias), as add_quad_levels8 and quad_level8 take it: from
// pair 0 (u) and 1 (v), operation for operation (no fused multiply-add),
// so the same bits. Preserves v0..v3, v6..v7 (layer) and v16..v31. Clobbers v8..v15.
@(private)
a_emit_quad_levels :: proc(e: ^Jit_Emitter) {
	ql :: proc(e: ^Jit_Emitter, v: u8, i: u32) {a_tab(e, v, A64_OFF_QL + 16 * i)}
	dup :: proc(e: ^Jit_Emitter, d, n: u8, i: u32) {ai(e, 0x4e040400 | ((i << 3) | 4) << 16 | u32(n) << 5 | u32(d))}
	for q in u8(0) ..< 2 {
		a_ldr_w(e, A_X16, A_X0, OFF_CI_LEVELS + OFF_LV_SIZE)
		a_dup_s(e, 14, A_X16)
		a_v2(e, V_SCVTF, 14, 14) // w
		a_ldr_w(e, A_X16, A_X0, OFF_CI_LEVELS + OFF_LV_SIZE + 4)
		a_dup_s(e, 15, A_X16)
		a_v2(e, V_SCVTF, 15, 15) // h
		dup(e, 10, q, 0)
		dup(e, 11, q, 1)
		dup(e, 12, q, 2)
		a_v3(e, V_FSUB, 11, 11, 10)
		a_v3(e, V_FMUL, 11, 11, 14) // dxx
		a_v3(e, V_FSUB, 10, 12, 10)
		a_v3(e, V_FMUL, 10, 10, 14) // dyx
		dup(e, 12, 2 + q, 0)
		dup(e, 13, 2 + q, 1)
		dup(e, 14, 2 + q, 2)
		a_v3(e, V_FSUB, 13, 13, 12)
		a_v3(e, V_FMUL, 13, 13, 15) // dxy
		a_v3(e, V_FSUB, 12, 14, 12)
		a_v3(e, V_FMUL, 12, 12, 15) // dyy
		a_v3(e, V_FMUL, 11, 11, 11)
		a_v3(e, V_FMUL, 13, 13, 13)
		a_v3(e, V_FADD, 8, 11, 13)  // x2
		a_v3(e, V_FMUL, 10, 10, 10)
		a_v3(e, V_FMUL, 12, 12, 12)
		a_v3(e, V_FADD, 9, 10, 12)  // y2
		// m into v10: y2 where x2 is NaN or smaller. fmaxnm gives those bits,
		// since a sum of squares is never -0 or a signaling NaN.
		a_v3(e, V_FMAXNM, 10, 8, 9)
		// quad_level8(m): 0.5 * log2(m).
		ql(e, 8, 0)
		a_v3(e, V_FCMGT, 8, 8, 10) // sub = m < FLT_MIN
		ql(e, 11, 1)
		a_v3(e, V_FMUL, 12, 10, 11)
		a_v3(e, V_ORR, 11, 8, 8)
		a_v3(e, V_BSL, 11, 12, 10) // ms
		ql(e, 12, 2)
		a_v3(e, V_SUB, 9, 11, 12)
		a_v2(e, 0x4f290400, 9, 9)  // sshr #23: k
		a_v2(e, 0x4f375400, 12, 9) // shl #23
		a_v3(e, V_SUB, 12, 11, 12)
		a_imm(e, FMOV_ONE, 13)
		a_v3(e, V_FSUB, 11, 12, 13) // f
		a_v2(e, V_SCVTF, 9, 9)
		ql(e, 12, 9)
		a_v3(e, V_AND, 8, 8, 12)
		a_v3(e, V_FSUB, 8, 9, 8)   // e
		a_v3(e, V_FMUL, 9, 11, 11) // f2
		for c in u32(0) ..< 3 {
			ql(e, 15, 4 + 2 * c)
			a_v3(e, V_FMUL, 12 + u8(c), 11, 15)
			ql(e, 15, 3 + 2 * c)
			a_v3(e, V_FADD, 12 + u8(c), 15, 12 + u8(c)) // C(2c) + f * C(2c + 1)
		}
		a_v3(e, V_FMUL, 14, 9, 14)
		a_v3(e, V_FADD, 13, 13, 14)
		a_v3(e, V_FMUL, 13, 9, 13)
		a_v3(e, V_FADD, 12, 12, 13) // q
		a_v3(e, V_FMUL, 12, 11, 12)
		a_v3(e, V_FADD, 12, 8, 12)
		a_imm(e, 0x4f03f400, 15)   // fmov #0.5
		a_v3(e, V_FMUL, 12, 15, 12) // l
		a_v2(e, V_FCMEQ0, 13, 10)
		a_tab(e, 15, A64_OFF_INF)
		a_v2(e, V_FNEG, 14, 15)
		a_v3(e, V_BIT, 12, 14, 13) // m == 0: -inf
		a_v3(e, V_FCMEQ, 13, 10, 15)
		a_v3(e, V_BIT, 12, 10, 13) // m == inf: m
		a_v2(e, 0x6ea0c800, 13, 10) // m >= 0
		a_tab(e, 15, A64_OFF_CANON)
		a_v3(e, V_BIF, 12, 15, 13) // else NaN
		a_v3(e, V_FADD, 4 + q, 4 + q, 12)
	}
}

// sample_sub: texture(), textureLod and textureGrad of a bound channel.
//   in:  pair 0 = u, 1 = v, 2 = lod, 3 = the layer (with layers: clamped,
//        as i32), x0 = ^Chan_Img, w1 = live components
//   out: pairs 0..3; preserves v16..v31.
@(private)
a_emit_sample_sub :: proc(e: ^Jit_Emitter) {
	ai(e, 0xd10043ff) // sub sp, sp, #16
	a_str_x(e, 30, A_SP, 0)
	a_live_held(e)
	// One level, Nearest, both axes clamped: the indexes in registers.
	a_ldrb(e, 15, A_X0, OFF_CI_ROUTE)
	general := a_cbz(e, 15, true)
	a_add_imm(e, A_X2, A_X0, OFF_CI_LEVELS)
	a_chan_nearest_xy(e)
	a_bcast_w(e, 14, A_X2, OFF_LV_PITCH)
	p3q(e, V_MUL, 4, 1, 14)
	p3(e, V_ADD, 4, 4, 0)
	a_bl(e, &e.decode_calls)
	short_done := a_branch(e, -1)

	a_land(e, general)
	ai(e, 0xd10003ff | A_SF_GEN << 10) // sub sp, sp, #A_SF_GEN
	for r in u8(8) ..< 16 {
		a_pair_mem(e, false, r, A_SP, A_SF_Q + i64(r - 8) * 32)
	}
	a_pair_mem(e, false, 0, A_SP, A_SF_U)
	a_pair_mem(e, false, 1, A_SP, A_SF_V)
	a_pair_mem(e, false, 3, A_SP, A_SF_LAYER)
	// Levels: l0 and l0 + 1's weight w per lane (Channel_Image), the levels
	// some lane uses as bits in w13; w12 = 0 (1 level), 1 (2 levels, iter 0),
	// 2 (2 levels, iter 1), or 3 (>= 3 levels across the quad).
	a_mov_w(e, 12, 0)
	a_mov_w(e, 13, 1)
	a_ldr_w(e, A_X16, A_X0, OFF_CI_NLEV)
	a_cmp_w(e, A_X16, 1)
	single := a_branch(e, 0)
	a_bcast_w(e, 14, A_X0, OFF_CI_TOP)
	p2(e, V_FCMGT0, 5, 2)      // lod > 0
	p3q(e, V_FMINNM, 6, 2, 14) // l = min(lod, top)
	p2(e, V_FCVTZS, 8, 6)
	p3(e, V_AND, 8, 8, 5)      // l0
	p2(e, V_SCVTF, 3, 8)
	p3(e, V_FSUB, 6, 6, 3)
	a_tab(e, 15, A64_OFF_F256)
	p3q(e, V_FMUL, 6, 6, 15)
	p2(e, V_FRINTN, 6, 6)
	p2(e, V_FCVTZS, 6, 6)
	p3(e, V_AND, 6, 6, 5)      // w (0 at the top level, where l = l0)
	a_pair_mem(e, false, 8, A_SP, A_SF_L0)
	a_pair_mem(e, false, 6, A_SP, A_SF_WL)
	a_imm(e, MOVI_ONE, 15)
	pq3(e, V_USHL, 1, 15, 8)   // 1 << l0
	p3q(e, V_ADD, 2, 8, 15)
	pq3(e, V_USHL, 2, 15, 2)   // 1 << (l0 + 1)
	p2(e, V_CMEQ0, 3, 6)
	p3(e, V_BIC, 2, 2, 3)      // where w != 0
	p3(e, V_ORR, 1, 1, 2)
	a_or_lanes(e, 13, 1, 0)
	ai(e, 0x510005af)          // sub w15, w13, #1
	ai(e, 0x0a0f01af)          // and w15, w13, w15 -> t = w13 & (w13 - 1)
	one_bit := a_cbz(e, 15, false)
	a_mov_w(e, 12, 1)
	ai(e, 0x510005f0)          // sub w16, w15, #1
	ai(e, 0x0a1001ef)          // and w15, w15, w16 -> 0 iff popcount(w13) == 2
	two_bits := a_cbz(e, 15, false)
	a_mov_w(e, 12, 3)
	a_land(e, two_bits)
	a_land(e, one_bit)
	a_land(e, single)

	loop_top := len(e.buf)
	a_next_level(e)
	a_ldrb(e, 15, A_X0, OFF_CI_FILTER)
	linear := a_cbz(e, 15, true)
	a_chan_axis(e, 0, A_SF_U, true)
	pmov(e, 8, 6)
	a_chan_axis(e, 1, A_SF_V, true)
	p3(e, V_ADD, 4, 6, 8)
	a_bl(e, &e.decode_calls)
	tapped := a_branch(e, -1)

	a_land(e, linear)
	// Linear in registers: axis 0 -> (X0=8, X1=9, WX=10), axis 1 -> (Y0=6, Y1=5, WY=11),
	// then four corner indices (c00=4, c10=7, c01=8, c11=9).
	a_chan_axis(e, 0, A_SF_U, false, true)
	pmov(e, 8, 6)
	pmov(e, 9, 5)
	pmov(e, 10, 4)
	a_chan_axis(e, 1, A_SF_V, false, true)
	pmov(e, 11, 4)
	p3(e, V_ADD, 4, 6, 8)  // c00
	p3(e, V_ADD, 7, 6, 9)  // c10
	p3(e, V_ADD, 8, 5, 8)  // c01
	p3(e, V_ADD, 9, 5, 9)  // c11
	a_ldrb(e, A_X16, A_X0, OFF_CI_FORMAT)
	a_cmp_w(e, A_X16, u32(Channel_Format.Rgba8))
	not_rgba8 := a_branch(e, 1) // b.ne
	// Rgba8 per-level linear tap: integer bilinear via uxtw loads and ucvtf #16.
	rgba8_dead := a_cbz(e, A_X1, false)
	a_ldr_x(e, 15, A_X2, OFF_LV_PTR)
	a_v2(e, V_XTN_4H, 0, 20)
	a_v2(e, V_XTN2_8H, 0, 21)
	a_imm(e, MOVI_256_8H, 24)
	a_v3(e, V_SUB_8H, 1, 24, 0)
	a_v2(e, V_XTN_4H, 2, 22)
	a_v2(e, V_XTN2_8H, 2, 23)
	a_v3(e, V_SUB_8H, 3, 24, 2)
	for l in u32(0) ..< 8 {
		h, i := u8(l / 4), l % 4
		a_umov_s(e, A_X3, 8 + h, i)
		a_umov_s(e, A_X3 + 1, 14 + h, i)
		a_umov_s(e, A_X3 + 2, 16 + h, i)
		a_umov_s(e, A_X3 + 3, 18 + h, i)
		ai(e, A_LDR_S | u32(A_X3) << 16 | 15 << 5 | 4)
		ai(e, A_LDR_S | u32(A_X3 + 1) << 16 | 15 << 5 | 5)
		ai(e, A_LDR_S | u32(A_X3 + 2) << 16 | 15 << 5 | 6)
		ai(e, A_LDR_S | u32(A_X3 + 3) << 16 | 15 << 5 | 7)
		ai(e, 0x6e0c0400 | 6 << 5 | 4) // ins v4.s[1], v6.s[0]
		ai(e, 0x6e0c0400 | 7 << 5 | 5) // ins v5.s[1], v7.s[0]
		a_v2(e, V_UXTL_8H, 4, 4)
		a_v2(e, V_UXTL_8H, 5, 5)
		a_elem_h(e, V_MUL_8H_E, 4, 4, 1, l)
		a_elem_h(e, V_MLA_8H_E, 4, 5, 0, l)
		a_elem_h(e, V_UMULL_4S_E, 24 + u8(l), 4, 3, l)
		a_elem_h(e, V_UMLAL2_4S_E, 24 + u8(l), 4, 2, l)
		a_v2(e, V_UCVTF_F16, 24 + u8(l), 24 + u8(l))
	}
	for h in u8(0) ..< 2 {
		b := 24 + 4 * h
		a_v3(e, V_TRN1, 20, b, b + 1)
		a_v3(e, V_TRN2, 21, b, b + 1)
		a_v3(e, V_TRN1, 22, b + 2, b + 3)
		a_v3(e, V_TRN2, 23, b + 2, b + 3)
		a_v3(e, V_TRN1_2D, h, 20, 22)
		a_v3(e, V_TRN2_2D, 4 + h, 20, 22)
		a_v3(e, V_TRN1_2D, 2 + h, 21, 23)
		a_v3(e, V_TRN2_2D, 6 + h, 21, 23)
	}
	a_land(e, rgba8_dead)
	rgba8_done := a_branch(e, -1)

	a_land(e, not_rgba8)
	a_cmp_w(e, A_X16, u32(Channel_Format.Gray8))
	not_gray8 := a_branch(e, 1) // b.ne
	g8_dead := a_tbz(e, A_X1, 0, false)
	a_ldr_x(e, 15, A_X2, OFF_LV_PTR)
	a_ldr_w(e, A_X16, A_X2, OFF_LV_MIS)
	ai(e, 0x8b3041ef) // add x15, x15, w16, uxtw
	a_v2(e, V_XTN_4H, 24, 20)
	a_v2(e, V_XTN2_8H, 24, 21)
	a_imm(e, MOVI_256_8H, 12)
	a_v3(e, V_SUB_8H, 25, 12, 24)
	a_v2(e, V_XTN_4H, 26, 22)
	a_v2(e, V_XTN2_8H, 26, 23)
	a_v3(e, V_SUB_8H, 27, 12, 26)
	for cr, ci in ([4]u8{4, 7, 8, 9}) {
		a_lane_idx(e, cr)
		a_gather_s8(e, A_LDR_B, 0)
		a_v2(e, V_XTN_4H, 28 + u8(ci), 0)
		a_v2(e, V_XTN2_8H, 28 + u8(ci), 1)
	}
	a_v3(e, V_MUL_8H, 28, 28, 25)
	a_v3(e, V_MLA_8H, 28, 29, 24)
	a_v3(e, V_MUL_8H, 30, 30, 25)
	a_v3(e, V_MLA_8H, 30, 31, 24)
	a_v3(e, V_UMULL_4S, 0, 28, 27)
	a_v3(e, V_UMLAL_4S, 0, 30, 26)
	a_v3(e, V_UMULL2_4S, 1, 28, 27)
	a_v3(e, V_UMLAL2_4S, 1, 30, 26)
	a_v2(e, V_UCVTF_F16, 0, 0)
	a_v2(e, V_UCVTF_F16, 1, 1)
	a_land(e, g8_dead)
	g8_done := a_branch(e, -1)

	// General float fallback (F32, Rgba16f, Gray16): convert integer weights -> f32 / 256.
	// Pairs 4 (v8..9), 7 (v14..15), 8..11 (v16..23) hold corners and weights, so use v12 for 1/256.
	a_land(e, not_gray8)
	a_tab(e, 12, A64_OFF_INV256)
	p2(e, V_SCVTF, 10, 10)
	p3q(e, V_FMUL, 10, 10, 12)
	p2(e, V_SCVTF, 11, 11)
	p3q(e, V_FMUL, 11, 11, 12)
	// Corner 00 -> pair (12 + k).
	a_bl(e, &e.decode_calls)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		pmov(e, 12 + k, k)
		a_land(e, sk)
	}
	// Corner 10 (pair 7 freed): A = c00 + (c10 - c00) * TX in pair (12 + k).
	pmov(e, 4, 7)
	a_bl(e, &e.decode_calls)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		p3(e, V_FSUB, 5, k, 12 + k)
		p3(e, V_FMUL, 5, 5, 10)
		p3(e, V_FADD, 12 + k, 12 + k, 5)
		a_land(e, sk)
	}
	// Corner 01 (pair 8 freed); queue c11 into pair 4 (pair 9 freed -> holds b0[2]),
	// leaving b0[0..2] in pairs 7..9 and spilling only b0[3] when live.
	pmov(e, 4, 8)
	a_bl(e, &e.decode_calls)
	pmov(e, 4, 9)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		if k < 3 {
			pmov(e, 7 + k, k)
		} else {
			a_pair_mem(e, false, 3, A_SP, A_SF_B3)
		}
		a_land(e, sk)
	}
	// Corner 11: B = b0 + (c11 - b0) * TX, final = A + (B - A) * TY in pair k.
	a_bl(e, &e.decode_calls)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		b0 := u8(7 + k) if k < 3 else u8(6)
		if k == 3 {
			a_pair_mem(e, true, 6, A_SP, A_SF_B3)
		}
		p3(e, V_FSUB, 5, k, b0)
		p3(e, V_FMUL, 5, 5, 10)
		p3(e, V_FADD, 5, b0, 5)
		p3(e, V_FSUB, 5, 5, 12 + k)
		p3(e, V_FMUL, 5, 5, 11)
		p3(e, V_FADD, k, 12 + k, 5)
		a_land(e, sk)
	}

	a_land(e, g8_done)
	a_land(e, rgba8_done)
	a_land(e, tapped)
	one_level := a_cbz(e, 12, false)
	a_cmp_w(e, 12, 2)
	two_iter0 := a_branch(e, 3) // b.lo: w12 == 1 (first of two levels L_a < L_b)
	two_iter1 := a_branch(e, 0) // b.eq: w12 == 2 (second of two levels)
	// General (>= 3 levels across lanes): this tap is level l0's for the
	// lanes at l0, l0 + 1's for the lanes blending toward it.
	a_dup_s(e, 14, 14)
	a_imm(e, MOVI_ONE, 15)
	a_pair_mem(e, true, 5, A_SP, A_SF_L0)
	p3q(e, V_CMEQ, 6, 5, 14)
	p3q(e, V_ADD, 5, 5, 15)
	p3q(e, V_CMEQ, 8, 5, 14)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		a_pair_mem(e, true, 4, A_SP, A_SF_C0 + i64(k) * 32)
		psel(e, 4, 6, k, 4)
		a_pair_mem(e, false, 4, A_SP, A_SF_C0 + i64(k) * 32)
		a_pair_mem(e, true, 4, A_SP, A_SF_C1 + i64(k) * 32)
		psel(e, 4, 8, k, 4)
		a_pair_mem(e, false, 4, A_SP, A_SF_C1 + i64(k) * 32)
		a_land(e, sk)
	}
	back := a_cbz(e, 13, true)
	a_patch(e, back, loop_top)
	// c0 + (c1 - c0) * w / 256 where w > 0, else c0.
	a_pair_mem(e, true, 6, A_SP, A_SF_WL)
	p2(e, V_SCVTF, 9, 6)
	a_tab(e, 15, A64_OFF_INV256)
	p3q(e, V_FMUL, 9, 9, 15)
	p2(e, V_CMGT0, 6, 6)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		a_pair_mem(e, true, k, A_SP, A_SF_C0 + i64(k) * 32)
		a_pair_mem(e, true, 4, A_SP, A_SF_C1 + i64(k) * 32)
		p3(e, V_FSUB, 4, 4, k)
		p3(e, V_FMUL, 4, 4, 9)
		p3(e, V_FADD, 4, k, 4)
		psel(e, k, 6, 4, k)
		a_land(e, sk)
	}
	gen_done := a_branch(e, -1)

	// Two levels (L_a < L_b): lanes with l0 == L_b must have w == 0, while
	// lanes with w > 0 have l0 == L_a and L_b == L_a + 1. Keep tap(L_a) in C0
	// and blend directly against tap(L_b) in pairs 0..3 without touching C1.
	a_land(e, two_iter0)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		a_pair_mem(e, false, k, A_SP, A_SF_C0 + i64(k) * 32)
		a_land(e, sk)
	}
	a_mov_w(e, 12, 2)
	back2 := a_branch(e, -1)
	a_patch(e, back2, loop_top)

	a_land(e, two_iter1)
	a_dup_s(e, 14, 14)
	a_pair_mem(e, true, 5, A_SP, A_SF_L0)
	p3q(e, V_CMEQ, 5, 5, 14) // l0 == L_b
	a_pair_mem(e, true, 6, A_SP, A_SF_WL)
	p2(e, V_SCVTF, 9, 6)
	a_tab(e, 15, A64_OFF_INV256)
	p3q(e, V_FMUL, 9, 9, 15)
	p2(e, V_CMGT0, 6, 6)     // wl > 0
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, A_X1, u32(k), false)
		a_pair_mem(e, true, 4, A_SP, A_SF_C0 + i64(k) * 32)
		p3(e, V_FSUB, 7, k, 4)
		p3(e, V_FMUL, 7, 7, 9)
		p3(e, V_FADD, 7, 4, 7)
		psel(e, 4, 6, 7, 4) // w > 0 ? C0 + (tap_b - C0) * w / 256 : C0
		psel(e, k, 5, k, 4) // l0 == L_b ? tap_b : above
		a_land(e, sk)
	}

	a_land(e, gen_done)
	a_land(e, one_level)
	for r in u8(8) ..< 16 {
		a_pair_mem(e, true, r, A_SP, A_SF_Q + i64(r - 8) * 32)
	}
	ai(e, 0x910003ff | A_SF_GEN << 10) // add sp, sp, #A_SF_GEN
	a_land(e, short_done)
	a_chan_finish(e)
	a_ldr_x(e, 30, A_SP, 0)
	ai(e, 0x910043ff) // add sp, sp, #16
	ai(e, 0xd65f03c0)
}

// fetch_sub: texelFetch of a bound channel, each lane's texel at its level,
// (0, 0, 0, 0) out of range.
//   in:  pair 0 = x, 1 = y, 2 = level (a layer outside the image's as a
//        level outside its levels), 3 = layer, x0 = ^Chan_Img, w1 = live
//        components | (flip_y << 8)
//   out: pairs 0..3; preserves v16..v31.
@(private)
a_emit_fetch_sub :: proc(e: ^Jit_Emitter) {
	a_live_held(e)
	ai(e, 0xd10003ff | u32(272) << 10) // sub sp, sp, #272
	a_str_x(e, 30, A_SP, 256)
	for r in u8(8) ..< 16 {
		a_pair_mem(e, false, r, A_SP, i64(r - 8) * 32)
	}
	pmov(e, 13, 0)
	pmov(e, 14, 1)
	pmov(e, 15, 2)
	pmov(e, 12, 3)
	for k in u8(8) ..< 12 {pzero(e, k)}
	// Levels in range (u32(level) < n_levels), as bits in w13.
	a_bcast_w(e, 14, A_X0, OFF_CI_NLEV)
	pq3(e, V_CMHI, 5, 14, 15)
	a_imm(e, MOVI_ONE, 14)
	pq3(e, V_USHL, 6, 14, 15)
	p3(e, V_AND, 6, 6, 5)
	a_or_lanes(e, 13, 6, 0)
	none := a_cbz(e, 13, false)

	loop_top := len(e.buf)
	a_next_level(e)
	a_dup_s(e, 12, 14)
	p3q(e, V_CMEQ, 7, 15, 12) // this level's lanes
	pmov(e, 0, 13)
	pmov(e, 1, 14)
	no_flip := a_tbz(e, 11, 8, false)
	a_bcast_w(e, 12, A_X2, OFF_LV_MAX + 4)
	pq3(e, V_SUB, 1, 12, 1) // max_y - y
	a_land(e, no_flip)
	a_bcast_w(e, 12, A_X2, OFF_LV_MAX)
	pq3(e, V_CMHS, 5, 12, 0)
	p3(e, V_AND, 7, 7, 5)
	a_bcast_w(e, 12, A_X2, OFF_LV_MAX + 4)
	pq3(e, V_CMHS, 5, 12, 1)
	p3(e, V_AND, 7, 7, 5)     // and in range
	a_bcast_w(e, 12, A_X2, OFF_LV_PITCH)
	p3q(e, V_MUL, 4, 1, 12)
	p3(e, V_ADD, 4, 4, 0)
	a_ldr_w(e, A_X16, A_X0, OFF_CI_NLAYERS)
	a_cmp_w(e, A_X16, 1)
	one := a_branch(e, A_LS)
	a_add_imm(e, A_X16, A_X0, OFF_CI_LPITCH)
	ai(e, 0xb86e7a10) // ldr w16, [x16, x14, lsl #2]
	a_dup_s(e, 12, A_X16)
	p3q(e, V_MUL, 5, 12, 12)
	p3(e, V_ADD, 4, 4, 5)
	a_land(e, one)
	p3(e, V_AND, 4, 4, 7)     // other lanes read texel 0
	a_bl(e, &e.decode_calls)
	a_chan_finish(e)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, 11, u32(k), false)
		psel(e, 8 + k, 7, k, 8 + k)
		a_land(e, sk)
	}
	back := a_cbz(e, 13, true)
	a_patch(e, back, loop_top)
	a_land(e, none)
	for k in u8(0) ..< 4 {
		sk := a_tbz(e, 11, u32(k), false)
		pmov(e, k, 8 + k)
		a_land(e, sk)
	}
	for r in u8(8) ..< 16 {
		a_pair_mem(e, true, r, A_SP, i64(r - 8) * 32)
	}
	a_ldr_x(e, 30, A_SP, 256)
	ai(e, 0x910003ff | u32(272) << 10) // add sp, sp, #272
	ai(e, 0xd65f03c0)
}

// Inline per-format level-0 texelFetch of a 2D image (`fetch_route !=
// FETCH_GENERAL`), decoding only compile-time `comp_mask` components and
// returning (0, 0, 0, 0) out of range; else calls `fetch_sub`.
// Preserves v16..v31.
@(private)
a_emit_fetch_one :: proc(e: ^Jit_Emitter, comp_mask: u8, flip: bool) {
	live :: #force_inline proc(m: u8, k: int) -> bool {return (m & (u8(1) << uint(k))) != 0}
	if comp_mask == 0 {
		return
	}
	fill_one :: proc(e: ^Jit_Emitter, mask: u8) {
		a_imm(e, FMOV_ONE, 6)
		a_imm(e, FMOV_ONE, 7)
		p3(e, V_AND, 3, 3, mask)
	}

	done: [5]int
	n_done := 0
	a_ldrb(e, 11, A_X0, OFF_CI_FETCH_ROUTE)
	a_cmp_w(e, 11, u32(FETCH_GENERAL))
	general := a_branch(e, A_EQ)

	// Pair 7 = (lod == 0 && x <= max_x && y <= max_y),
	// pair 4 = (y * pitch + x) & pair 7, x2 = &levels[0],
	// x15 = levels[0].ptr, w3..w10 = lane texel indices.
	a_add_imm(e, A_X2, A_X0, OFF_CI_LEVELS)
	if flip {
		a_bcast_w(e, 12, A_X2, OFF_LV_MAX + 4)
		pq3(e, V_SUB, 1, 12, 1)
	}
	p2(e, V_CMEQ0, 7, 2)
	a_bcast_w(e, 12, A_X2, OFF_LV_MAX)
	pq3(e, V_CMHS, 5, 12, 0)
	p3(e, V_AND, 7, 7, 5)
	a_bcast_w(e, 12, A_X2, OFF_LV_MAX + 4)
	pq3(e, V_CMHS, 5, 12, 1)
	p3(e, V_AND, 7, 7, 5)
	a_bcast_w(e, 12, A_X2, OFF_LV_PITCH)
	p3q(e, V_MUL, 4, 1, 12)
	a_ldr_x(e, 15, A_X2, OFF_LV_PTR)
	p3(e, V_ADD, 4, 4, 0)
	p3(e, V_AND, 4, 4, 7)
	a_lane_idx(e, 4)

	not_f32 := a_cbz(e, 11, true)
	// F32: comps 1, 2 or 4.
	a_ldrb(e, A_X16, A_X0, OFF_CI_COMPS)
	a_cmp_w(e, A_X16, 2)
	c2 := a_branch(e, 0)  // b.eq
	c4 := a_branch(e, 8)  // b.hi
	if live(comp_mask, 0) {
		a_gather_s8(e, A_LDR_S, 0)
		p3(e, V_AND, 0, 0, 7)
	}
	if live(comp_mask, 1) {pzero(e, 1)}
	if live(comp_mask, 2) {pzero(e, 2)}
	if live(comp_mask, 3) {fill_one(e, 7)}
	done[n_done] = a_branch(e, -1); n_done += 1

	a_land(e, c2)
	if live(comp_mask, 0) || live(comp_mask, 1) {
		for h in u32(0) ..< 2 {
			a_gather_d(e, h)
			if live(comp_mask, 0) {a_v3(e, V_UZP1_4S, u8(h), 12, 13)}
			if live(comp_mask, 1) {a_v3(e, V_UZP2_4S, 2 + u8(h), 12, 13)}
		}
		if live(comp_mask, 0) {p3(e, V_AND, 0, 0, 7)}
		if live(comp_mask, 1) {p3(e, V_AND, 1, 1, 7)}
	}
	if live(comp_mask, 2) {pzero(e, 2)}
	if live(comp_mask, 3) {fill_one(e, 7)}
	done[n_done] = a_branch(e, -1); n_done += 1

	a_land(e, c4)
	for h in u8(0) ..< 2 {
		for i in u8(0) ..< 4 {
			a_ldr_lane(e, A_LDR_Q, 8 + i, u32(4 * h + i))
		}
		a_v3(e, V_TRN1, 12, 8, 9)
		a_v3(e, V_TRN2, 13, 8, 9)
		a_v3(e, V_TRN1, 8, 10, 11)
		a_v3(e, V_TRN2, 9, 10, 11)
		if live(comp_mask, 0) {a_v3(e, V_TRN1_2D, h, 12, 8)}
		if live(comp_mask, 2) {a_v3(e, V_TRN2_2D, 4 + h, 12, 8)}
		if live(comp_mask, 1) {a_v3(e, V_TRN1_2D, 2 + h, 13, 9)}
		if live(comp_mask, 3) {a_v3(e, V_TRN2_2D, 6 + h, 13, 9)}
	}
	for k in 0 ..< 4 {
		if live(comp_mask, k) {
			p3(e, V_AND, u8(k), u8(k), 7)
		}
	}
	done[n_done] = a_branch(e, -1); n_done += 1

	// Rgba8: live bytes / 255, masked by pair 7.
	a_land(e, not_f32)
	a_cmp_w(e, 11, u32(Channel_Format.Rgba8))
	not_rgba8 := a_branch(e, 1)
	a_gather_s8(e, A_LDR_S, 6)
	a_imm(e, MOVI_255, 11)
	a_bcast_w(e, 10, A_X0, OFF_CI_UNORM)
	for k in 0 ..< 4 {
		if !live(comp_mask, k) {continue}
		r := u8(k)
		if k > 0 {
			p2(e, 0x6f000400 | (64 - 8 * u32(k)) << 16, r, 6)
		}
		if k == 0 {
			p3q(e, V_AND, 0, 6, 11)
		} else if k < 3 {
			p3q(e, V_AND, r, r, 11)
		}
	}
	for k in 0 ..< 4 {
		if live(comp_mask, k) {p2(e, V_UCVTF, u8(k), u8(k))}
	}
	for k in 0 ..< 4 {
		if live(comp_mask, k) {p3q(e, V_FDIV, u8(k), u8(k), 10)}
	}
	for k in 0 ..< 4 {
		if live(comp_mask, k) {p3(e, V_AND, u8(k), u8(k), 7)}
	}
	done[n_done] = a_branch(e, -1); n_done += 1

	// Gray8 and Gray16.
	a_land(e, not_rgba8)
	a_cmp_w(e, 11, u32(Channel_Format.Rgba16f))
	is_half := a_branch(e, A_EQ)
	if live(comp_mask, 0) {
		a_ldr_w(e, A_X16, A_X2, OFF_LV_MIS)
		ai(e, 0x8b3041ef) // add x15, x15, w16, uxtw
		a_bcast_w(e, 12, A_X0, OFF_CI_UNORM)
		a_cmp_w(e, 11, u32(Channel_Format.Gray16))
		g16 := a_branch(e, A_EQ)
		a_gather_s8(e, A_LDR_B, 0)
		g_done := a_branch(e, -1)
		a_land(e, g16)
		a_gather_s8(e, A_LDR_H, 0)
		a_land(e, g_done)
		p2(e, V_UCVTF, 0, 0)
		p3q(e, V_FDIV, 0, 0, 12)
		p3(e, V_AND, 0, 0, 7)
	}
	if live(comp_mask, 1) {pzero(e, 1)}
	if live(comp_mask, 2) {pzero(e, 2)}
	if live(comp_mask, 3) {fill_one(e, 7)}
	done[n_done] = a_branch(e, -1); n_done += 1

	// Rgba16f.
	a_land(e, is_half)
	for h in u8(0) ..< 2 {
		a_gather_d(e, u32(h))
		if live(comp_mask, 0) || live(comp_mask, 2) {
			a_v3(e, V_UZP1_8H, 10, 12, 13)
		}
		if live(comp_mask, 1) || live(comp_mask, 3) {
			a_v3(e, V_UZP2_8H, 11, 12, 13)
		}
		if live(comp_mask, 0) {a_v3(e, V_UZP1_8H, 12, 10, 10); a_v2(e, V_FCVTL_4S, h, 12)}
		if live(comp_mask, 2) {a_v3(e, V_UZP2_8H, 12, 10, 10); a_v2(e, V_FCVTL_4S, 4 + h, 12)}
		if live(comp_mask, 1) {a_v3(e, V_UZP1_8H, 12, 11, 11); a_v2(e, V_FCVTL_4S, 2 + h, 12)}
		if live(comp_mask, 3) {a_v3(e, V_UZP2_8H, 12, 11, 11); a_v2(e, V_FCVTL_4S, 6 + h, 12)}
	}
	for k in 0 ..< 4 {
		if live(comp_mask, k) {p3(e, V_AND, u8(k), u8(k), 7)}
	}
	gen_done := a_branch(e, -1)

	a_land(e, general)
	a_mov_w(e, A_X1, u32(comp_mask) | u32(flip) << 8)
	a_bl(e, &e.fetch_calls)
	for p in done[:n_done] {a_land(e, p)}
	a_land(e, gen_done)
}

// A bound-channel texture op inside a basic block: inputs already in
// pairs 0..2 (and 3 when in_.n == 3), x0 = ^Chan_Img; returns live `comp_mask`
// results in pairs 0..3 while preserving v16..v31.
@(private)
a_emit_bound_tex :: proc(e: ^Jit_Emitter, in_ptr: ^Instr, comp_mask: u8) {
	in_ := in_ptr^
	flip := (in_.d & 1) != 0
	if in_.op == .Texel_Fetch && in_.n == 3 {
		// A sampler2DArray's: its layer p.z, outside [0, n_layers) as a
		// level outside the levels (all ones).
		a_ldr_w(e, A_X16, A_X0, OFF_CI_NLAYERS)
		ai(e, 0x51000610) // sub w16, w16, #1
		a_dup_s(e, 12, A_X16)
		p3q(e, V_CMHI, 5, 3, 12)
		p3(e, V_ORR, 2, 2, 5)
		a_mov_w(e, A_X1, u32(comp_mask) | u32(flip) << 8)
		a_bl(e, &e.fetch_calls)
	} else if in_.op == .Texel_Fetch {
		a_emit_fetch_one(e, comp_mask, flip)
	} else {
		if in_.op == .Texture_Bias {
			// With levels, the bias plus the level the quad's derivatives pick.
			a_ldr_w(e, A_X16, A_X0, OFF_CI_NLEV)
			a_cmp_w(e, A_X16, 1)
			one := a_branch(e, A_LS)
			a_emit_quad_levels(e)
			a_land(e, one)
		}
		if in_.n == 3 {
			// A sampler2DArray's layer: clamp(round_even(r), 0, layers - 1),
			// NaN as 0 (bound_layer; frintn quiets a signaling one).
			p2(e, V_FRINTN, 3, 3)
			a_imm(e, V_ZERO, 12)
			p3q(e, V_FMAXNM, 3, 3, 12)
			a_bcast_w(e, 12, A_X0, OFF_CI_LTOP)
			p3q(e, V_FMINNM, 3, 3, 12)
			p2(e, V_FCVTZS, 3, 3)
		}
		a_mov_w(e, A_X1, u32(comp_mask))
		// F32, one level, Linear, both axes clamped: bilinear_sub.
		a_ldrb(e, 15, A_X0, OFF_CI_LIN_ROUTE)
		not_lin := a_cbz(e, 15, true)
		a_bl(e, &e.bilinear_calls)
		lin_done := a_branch(e, -1)
		a_land(e, not_lin)
		a_bl(e, &e.sample_calls)
		a_land(e, lin_done)
	}
}

// x0 = channel k's record (below the slots, as jit_begin fills them).
@(private)
a_chan_record :: proc(e: ^Jit_Emitter, k: i32) {
	off := u32(JIT_CHAN_BYTES - k * size_of(Chan_Img))
	if off >> 12 != 0 {
		ai(e, 0xd1400000 | (off >> 12) << 10 | u32(A_SLOTS) << 5 | u32(A_X0)) // sub x0, x20, #(off >> 12), lsl #12
		if lo := off & 0xfff; lo != 0 {
			ai(e, 0xd1000000 | lo << 10 | u32(A_X0) << 5 | u32(A_X0)) // sub x0, x0, #lo
		}
	} else {
		ai(e, 0xd1000000 | off << 10 | u32(A_SLOTS) << 5 | u32(A_X0)) // sub x0, x20, #off
	}
}

// ---- IR -> NEON lowering ----------------------------------------------------------------

@(private)
jit_emit_fn :: proc(e: ^Jit_Emitter, p: ^Program, code: []Instr, is_init: bool) {
	if p.num_masks > 4096 {
		e.unsupported = true
		return
	}
	entry_exec_end := jit_plan_fn(e, p, code, is_init)
	n_code := len(code)
	pc_off := e.pc_off[:n_code + 1]

	// Prologue: x29, x30, x19..x23 and d8..d15 (AAPCS64's callee-saved ones
	// this code uses).
	ai(e, 0xa9b87bfd) // stp x29, x30, [sp, #-128]!
	ai(e, 0x910003fd) // mov x29, sp
	ai(e, 0xa90153f3) // stp x19, x20, [sp, #16]
	ai(e, 0xa9025bf5) // stp x21, x22, [sp, #32]
	ai(e, 0xf9001bf7) // str x23, [sp, #48]
	ai(e, 0x6d0427e8) // stp d8, d9, [sp, #64]
	ai(e, 0x6d052fea) // stp d10, d11, [sp, #80]
	ai(e, 0x6d0637ec) // stp d12, d13, [sp, #96]
	ai(e, 0x6d073fee) // stp d14, d15, [sp, #112]
	a_mov_xx(e, A_CTX, A_X0)
	ai(e, 0xf9400000 | u32(OFF_CTX_SLOTS / 8) << 10 | u32(A_CTX) << 5 | u32(A_SLOTS)) // ldr x20, [x19, #slots]
	ai(e, 0xf9400000 | u32(OFF_CTX_MASKS / 8) << 10 | u32(A_CTX) << 5 | u32(A_MASKS)) // ldr x21, [x19, #masks]
	ai(e, 0x39400000 | u32(OFF_CTX_LANES) << 10 | u32(A_CTX) << 5 | u32(A_EXEC))      // ldrb w22, [x19, #lanes]
	a_mov_imm(e, A_TAB, u64(uintptr(&jit_a64_tables)))

	cache_reset(e)

	for pc := 0; pc < n_code; pc += 1 {
		pc_off[pc] = len(e.buf)
		jit_enter_pc(e, pc, n_code)

		e.pc = i32(pc)
		in_ := code[pc]
		e.cur_dst_base = in_.dst
		e.cur_def_off = e.def_off[pc]
		e.cur_mem_mask = e.mem_mask[pc]
		n := i32(in_.n)
		dst, a, b, c := in_.dst, in_.a, in_.b, in_.c

		switch in_.op {
		case .Nop:

		case .Mov:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 0)
				copy_to_dst(e, dst + i, ya, pin_bit(ya))
			}

		case .Splat:
			ya := read_slot(e, a, 0)
			pa := pin_bit(ya)
			for i in 0 ..< n {
				copy_to_dst(e, dst + i, ya, pa, pa if i + 1 < n else 0)
			}

		case .Gather:
			srcs := [4]i32{a, b, c, in_.d}
			for i in 0 ..< n {
				ys := read_slot(e, srcs[i], 0)
				copy_to_dst(e, dst + i, ys, pin_bit(ys))
			}

		case .Store:
			if pc < entry_exec_end {
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 0)
					copy_to_dst(e, dst + i, ya, pin_bit(ya))
				}
				break
			}
			ym := a_exec_lanes(e)
			for i in 0 ..< n {
				yold := read_slot(e, dst + i, 1)
				ya := read_slot(e, a + i, 2, pin_bit(yold))
				yd := prep_dst(e, dst + i, 0, pin_bit(yold) | pin_bit(ya))
				psel(e, yd, ym, ya, yold)
				commit_dst(e, dst + i, yd)
			}

		case .Store_Zero:
			if pc < entry_exec_end {
				for i in 0 ..< n {
					yd := prep_dst(e, dst + i, 0)
					pzero(e, yd)
					commit_dst(e, dst + i, yd)
				}
				break
			}
			ym := a_exec_lanes(e)
			for i in 0 ..< n {
				yold := read_slot(e, dst + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(yold))
				p3(e, V_BIC, yd, yold, ym)
				commit_dst(e, dst + i, yd)
			}

		case .Load_Dyn:
			// A block of its own (is_helper_instr): every lane at one offset
			// copies the element's slots, else the helper.
			if n > 4 {
				a_call(e, rawptr(jit_help_load_dyn), &code[pc])
				break
			}
			a_pair_mem(e, true, 0, A_SLOTS, i64(b) * 32)
			a_uniform_lane0(e, 0, A_X1)
			div := a_cbz(e, A_X0, false)
			ai(e, 0x8b001400 | u32(A_X1) << 16 | u32(A_SLOTS) << 5 | u32(A_X16)) // add x16, x20, x1, lsl #5
			for i in 0 ..< n {
				a_pair_mem(e, true, 0, A_X16, i64(a + i) * 32)
				a_pair_mem(e, false, 0, A_SLOTS, i64(dst + i) * 32)
			}
			done := a_branch(e, -1)
			a_land(e, div)
			a_call(e, rawptr(jit_help_load_dyn), &code[pc])
			a_land(e, done)

		case .Store_Dyn:
			// A block of its own: lane by lane over the running lanes.
			a_call(e, rawptr(jit_help_store_dyn), &code[pc])

		case .F_Add, .F_Sub, .F_Mul, .F_Div:
			op := V_FADD
			#partial switch in_.op {
			case .F_Sub: op = V_FSUB
			case .F_Mul: op = V_FMUL
			case .F_Div: op = V_FDIV
			}
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			hoist_a := sa == 0 && (a < dst || a >= dst + n)
			hoist_b := sb == 0 && (b < dst || b >= dst + n)
			ya_s: u8 = 0
			yb_s: u8 = 0
			if hoist_a {
				ya_s = read_slot(e, a, 6)
			}
			if hoist_b {
				yb_s = read_slot(e, b, 5, pin_bit(ya_s))
			}
			for i in 0 ..< n {
				ya := ya_s if hoist_a else read_slot(e, a + i * sa, 1, pin_bit(yb_s))
				yb := yb_s if hoist_b else read_slot(e, b + i * sb, 2, pin_bit(ya))
				keep := u8(0)
				if i + 1 < n {
					if hoist_a {keep |= pin_bit(ya)}
					if hoist_b {keep |= pin_bit(yb)}
				}
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb), keep)
				p3(e, op, yd, ya, yb)
				commit_dst(e, dst + i, yd)
			}

		case .F_Neg:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p2(e, V_FNEG, yd, ya)
				commit_dst(e, dst + i, yd)
			}

		case .I_Add, .I_Sub, .I_Mul, .Bit_And, .Bit_Or, .Bit_Xor:
			op := V_ADD
			#partial switch in_.op {
			case .I_Sub:   op = V_SUB
			case .I_Mul:   op = V_MUL
			case .Bit_And: op = V_AND
			case .Bit_Or:  op = V_ORR
			case .Bit_Xor: op = V_EOR
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				p3(e, op, yd, ya, yb)
				commit_dst(e, dst + i, yd)
			}

		case .I_Div, .U_Div, .I_Rem, .U_Rem:
			// In f64, two lanes a register, where a 32-bit quotient's
			// division and truncation are exact; 0 where the divisor is 0,
			// and min(i32) / -1 the low word of 2^31, min(i32), as bin_lane.
			e.exec_ymask_valid = false
			is_u := in_.op == .U_Div || in_.op == .U_Rem
			is_rem := in_.op == .I_Rem || in_.op == .U_Rem
			ext_lo := V_UXTL if is_u else V_SXTL
			ext_hi := V_UXTL2 if is_u else V_SXTL2
			cvt := V_UCVTF_2D if is_u else V_SCVTF_2D
			back := V_FCVTZU_2D if is_u else V_FCVTZS_2D
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 6)
				yb := read_slot(e, b + i, 7, pin_bit(ya))
				// v0, v1 = the quotients of each half's four lanes.
				for k in u8(0) ..< 2 {
					va, vb := 2 * ya + k, 2 * yb + k
					a_v2(e, ext_lo, 2, va)
					a_v2(e, ext_hi, 3, va)
					a_v2(e, ext_lo, 4, vb)
					a_v2(e, ext_hi, 5, vb)
					a_v2(e, cvt, 2, 2)
					a_v2(e, cvt, 3, 3)
					a_v2(e, cvt, 4, 4)
					a_v2(e, cvt, 5, 5)
					a_v3(e, V_FDIV_2D, 2, 2, 4)
					a_v3(e, V_FDIV_2D, 3, 3, 5)
					a_v2(e, back, 2, 2)
					a_v2(e, back, 3, 3)
					a_v2(e, V_XTN, k, 2)
					a_v2(e, V_XTN2, k, 3)
				}
				if is_rem {
					p3(e, V_MUL, 1, 0, yb)
					p3(e, V_SUB, 0, ya, 1)
				}
				p3(e, V_CMTST, 2, yb, yb) // the divisor != 0
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				p3(e, V_AND, yd, 0, 2)
				commit_dst(e, dst + i, yd)
			}

		case .I_Neg:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p2(e, V_NEG, yd, ya)
				commit_dst(e, dst + i, yd)
			}

		case .Bit_Not:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p2(e, V_MVN, yd, ya)
				commit_dst(e, dst + i, yd)
			}

		case .Shl, .Shr_I, .Shr_U:
			// ushl and sshl shift by each lane's count, right when it's
			// negative: the count & 31, negated for the right shifts.
			a_imm(e, MOVI_31, 8)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				p3q(e, V_AND, 3, yb, 8)
				if in_.op != .Shl {
					p2(e, V_NEG, 3, 3)
				}
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				p3(e, V_SSHL if in_.op == .Shr_I else V_USHL, yd, ya, 3)
				commit_dst(e, dst + i, yd)
			}

		case .F_Lt, .F_Le, .F_Gt, .F_Ge, .F_Eq, .F_Ne,
		     .I_Lt, .I_Le, .I_Gt, .I_Ge, .I_Eq, .I_Ne,
		     .U_Lt, .U_Le, .U_Gt, .U_Ge:
			// a < b as b > a; a != b as !(a == b) (true for a NaN).
			op := u32(0)
			swap, inv := false, false
			#partial switch in_.op {
			case .F_Lt: op = V_FCMGT; swap = true
			case .F_Le: op = V_FCMGE; swap = true
			case .F_Gt: op = V_FCMGT
			case .F_Ge: op = V_FCMGE
			case .F_Eq: op = V_FCMEQ
			case .F_Ne: op = V_FCMEQ; inv = true
			case .I_Lt: op = V_CMGT; swap = true
			case .I_Le: op = V_CMGE; swap = true
			case .I_Gt: op = V_CMGT
			case .I_Ge: op = V_CMGE
			case .I_Eq: op = V_CMEQ
			case .I_Ne: op = V_CMEQ; inv = true
			case .U_Lt: op = V_CMHI; swap = true
			case .U_Le: op = V_CMHS; swap = true
			case .U_Gt: op = V_CMHI
			case .U_Ge: op = V_CMHS
			}
			if pc_mc, fused := can_fuse_cmp_mask_cond(e, code, pc, is_init); fused {
				ya := read_slot(e, a, 1)
				yb := read_slot(e, b, 2, pin_bit(ya))
				p3(e, op, 0, yb if swap else ya, ya if swap else yb)
				a_finish_fused_cmp(e, dst, pc_mc, inv)
				break
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				p3(e, op, yd, yb if swap else ya, ya if swap else yb)
				if inv {
					p2(e, V_MVN, yd, yd)
				}
				pone(e, yd, yd)
				commit_dst(e, dst + i, yd)
			}

		case .B_And:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				p3(e, V_CMTST, 3, ya, ya)
				p3(e, V_CMTST, 4, yb, yb)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				p3(e, V_AND, yd, 3, 4)
				pone(e, yd, yd)
				commit_dst(e, dst + i, yd)
			}

		case .B_Or:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				p3(e, V_ORR, 3, ya, yb)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				p3(e, V_CMTST, yd, 3, 3)
				pone(e, yd, yd)
				commit_dst(e, dst + i, yd)
			}

		case .B_Not:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p2(e, V_CMEQ0, yd, ya)
				pone(e, yd, yd)
				commit_dst(e, dst + i, yd)
			}

		case .I_To_F, .U_To_F, .F_To_I, .F_To_U:
			// scvtf and ucvtf round to nearest; fcvtzs and fcvtzu truncate,
			// saturate and take NaN to 0, as f_to_i and f_to_u.
			op := V_SCVTF
			#partial switch in_.op {
			case .U_To_F: op = V_UCVTF
			case .F_To_I: op = V_FCVTZS
			case .F_To_U: op = V_FCVTZU
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p2(e, op, yd, ya)
				commit_dst(e, dst + i, yd)
			}

		case .B_To_F:
			a_imm(e, FMOV_ONE, 8)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p3(e, V_CMTST, yd, ya, ya)
				p3q(e, V_AND, yd, yd, 8)
				commit_dst(e, dst + i, yd)
			}

		case .F_To_B:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p2(e, V_FCMEQ0, yd, ya)
				p2(e, V_MVN, yd, yd) // !(x == 0.0): a NaN is true
				pone(e, yd, yd)
				commit_dst(e, dst + i, yd)
			}

		case .I_To_B:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				p3(e, V_CMTST, yd, ya, ya)
				pone(e, yd, yd)
				commit_dst(e, dst + i, yd)
			}

		case .Select:
			// c != 0 ? a : b.
			if in_.k == 1 {
				yc := read_slot(e, c, 3)
				p3(e, V_CMTST, 3, yc, yc)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					psel(e, yd, 3, ya, yb)
					commit_dst(e, dst + i, yd)
				}
			} else {
				for i in 0 ..< n {
					yc := read_slot(e, c + i, 3)
					p3(e, V_CMTST, 3, yc, yc)
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					psel(e, yd, 3, ya, yb)
					commit_dst(e, dst + i, yd)
				}
			}

		case .All, .Any:
			// The components' != 0, and-ed or or-ed, to 1.
			ya0 := read_slot(e, a, 1)
			p3(e, V_CMTST, 0, ya0, ya0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				p3(e, V_CMTST, 3, yai, yai)
				p3(e, V_AND if in_.op == .All else V_ORR, 0, 0, 3)
			}
			yd := prep_dst(e, dst, 0)
			pone(e, yd, 0)
			commit_dst(e, dst, yd)

		case .F_Fn1:
			bi := Builtin(in_.k)
			op1 := u32(0)
			#partial switch bi {
			case .Abs:        op1 = V_FABS
			case .Sqrt:       op1 = V_FSQRT
			case .Floor:      op1 = V_FRINTM
			case .Ceil:       op1 = V_FRINTP
			case .Trunc:      op1 = V_FRINTZ
			case .Round_Even: op1 = V_FRINTN
			case .Round:      op1 = V_FRINTA // half away from zero, as math.round
			}
			if op1 != 0 {
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p2(e, op1, yd, ya)
					commit_dst(e, dst + i, yd)
				}
				break
			}
			#partial switch bi {
			case .Inversesqrt:
				a_imm(e, FMOV_ONE, 8)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p2(e, V_FSQRT, yd, ya)
					pq3(e, V_FDIV, yd, 8, yd)
					commit_dst(e, dst + i, yd)
				}
			case .Fract:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					p2(e, V_FRINTM, 2, ya)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p3(e, V_FSUB, yd, ya, 2)
					commit_dst(e, dst + i, yd)
				}
			case .Radians, .Degrees:
				a_tab(e, 8, A64_OFF_DEG2RAD if bi == .Radians else A64_OFF_RAD2DEG)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p3q(e, V_FMUL, yd, ya, 8)
					commit_dst(e, dst + i, yd)
				}
			case .Sign:
				a_imm(e, FMOV_ONE, 10)
				a_imm(e, FMOV_NEG_ONE, 11)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					p2(e, V_FCMGT0, 3, ya)
					p2(e, V_FCMLT0, 4, ya)
					p3q(e, V_AND, 3, 3, 10)
					p3q(e, V_AND, 4, 4, 11)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p3(e, V_ORR, yd, 3, 4)
					commit_dst(e, dst + i, yd)
				}
			case .Isnan:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p3(e, V_FCMEQ, yd, ya, ya)
					p2(e, V_MVN, yd, yd)
					pone(e, yd, yd)
					commit_dst(e, dst + i, yd)
				}
			case .Isinf:
				a_tab(e, 8, A64_OFF_INF)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p2(e, V_FABS, yd, ya)
					p3q(e, V_CMEQ, yd, yd, 8)
					pone(e, yd, yd)
					commit_dst(e, dst + i, yd)
				}
			case .Float_Bits_To_Int, .Float_Bits_To_Uint:
				// canon_bits: every NaN as 0x7fc00000 (README: the host and
				// integers see NaNs canonical; the program's float outputs
				// pass through here).
				a_tab(e, 8, A64_OFF_CANON)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					p3(e, V_FCMEQ, 2, ya, ya) // not NaN
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					pmov(e, yd, ya)
					a_v3(e, V_BIF, 2 * yd, 8, 4)
					a_v3(e, V_BIF, 2 * yd + 1, 8, 5)
					commit_dst(e, dst + i, yd)
				}
			case:
				for i in 0 ..< n {
					a_math_call(e, &code[pc], i)
				}
			}

		case .F_Fn2:
			bi := Builtin(in_.k)
			#partial switch bi {
			case .Min, .Max:
				// fminnm and fmaxnm: a NaN operand gives the other.
				op := V_FMINNM if bi == .Min else V_FMAXNM
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					p3(e, op, yd, ya, yb)
					commit_dst(e, dst + i, yd)
				}
			case .Step:
				// x < edge ? 0 : 1 (a = edge, b = x).
				a_imm(e, FMOV_ONE, 8)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					p3(e, V_FCMGT, yd, ya, yb)
					pq3(e, V_BIC, yd, 8, yd)
					commit_dst(e, dst + i, yd)
				}
			case .Mod:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					p3(e, V_FDIV, 3, ya, yb)
					p2(e, V_FRINTM, 3, 3)
					p3(e, V_FMUL, 3, yb, 3)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					p3(e, V_FSUB, yd, ya, 3)
					commit_dst(e, dst + i, yd)
				}
			case:
				for i in 0 ..< n {
					a_math_call(e, &code[pc], i)
				}
			}

		case .F_Fn3:
			bi := Builtin(in_.k)
			#partial switch bi {
			case .Clamp:
				// fminnm(fmaxnm(x, lo), hi).
				for i in 0 ..< n {
					yx := read_slot(e, a + i, 1)
					ylo := read_slot(e, b + i, 2, pin_bit(yx))
					yhi := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(ylo))
					p3(e, V_FMAXNM, 4, yx, ylo)
					yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(ylo) | pin_bit(yhi))
					p3(e, V_FMINNM, yd, 4, yhi)
					commit_dst(e, dst + i, yd)
				}
			case .Mix:
				// x * (1 - z) + y * z
				a_imm(e, FMOV_ONE, 10)
				for i in 0 ..< n {
					yx := read_slot(e, a + i, 1)
					yy := read_slot(e, b + i, 2, pin_bit(yx))
					yz := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(yy))
					pq3(e, V_FSUB, 4, 10, yz)
					p3(e, V_FMUL, 4, yx, 4)
					p3(e, V_FMUL, 6, yy, yz)
					yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(yy) | pin_bit(yz))
					p3(e, V_FADD, yd, 4, 6)
					commit_dst(e, dst + i, yd)
				}
			case .Smoothstep:
				// t = clamp((x - e0) / (e1 - e0), 0, 1) by fmaxnm and fminnm
				// (NaN to 0), then t * t * (3 - 2 * t); constants in v10..v13.
				a_imm(e, V_ZERO, 10)
				a_imm(e, FMOV_TWO, 11)
				a_imm(e, FMOV_THREE, 12)
				a_imm(e, FMOV_ONE, 13)
				for i in 0 ..< n {
					yx := read_slot(e, a + i, 1)
					yy := read_slot(e, b + i, 2, pin_bit(yx))
					yz := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(yy))
					p3(e, V_FSUB, 0, yz, yx)
					p3(e, V_FSUB, 4, yy, yx)
					p3(e, V_FDIV, 0, 0, 4)
					p3q(e, V_FMAXNM, 0, 0, 10)
					p3q(e, V_FMINNM, 0, 0, 13)
					p3q(e, V_FMUL, 4, 0, 11)
					pq3(e, V_FSUB, 4, 12, 4)
					p3(e, V_FMUL, 0, 0, 0)
					yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(yy) | pin_bit(yz))
					p3(e, V_FMUL, yd, 0, 4)
					commit_dst(e, dst + i, yd)
				}
			}

		case .I_Fn1:
			if Builtin(in_.k) == .Abs {
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p2(e, V_ABS, yd, ya)
					commit_dst(e, dst + i, yd)
				}
			} else {
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					p2(e, V_CMGT0, 3, ya) // -1 where x > 0
					p2(e, V_CMLT0, 4, ya) // -1 where x < 0
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					p3(e, V_SUB, yd, 4, 3)
					commit_dst(e, dst + i, yd)
				}
			}

		case .I_Fn2, .U_Fn2:
			op := u32(0)
			if in_.op == .I_Fn2 {
				op = V_SMIN if Builtin(in_.k) == .Min else V_SMAX
			} else {
				op = V_UMIN if Builtin(in_.k) == .Min else V_UMAX
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				p3(e, op, yd, ya, yb)
				commit_dst(e, dst + i, yd)
			}

		case .I_Fn3, .U_Fn3:
			max_op := V_SMAX if in_.op == .I_Fn3 else V_UMAX
			min_op := V_SMIN if in_.op == .I_Fn3 else V_UMIN
			for i in 0 ..< n {
				yx := read_slot(e, a + i, 1)
				ylo := read_slot(e, b + i, 2, pin_bit(yx))
				yhi := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(ylo))
				p3(e, max_op, 0, yx, ylo)
				yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(ylo) | pin_bit(yhi))
				p3(e, min_op, yd, 0, yhi)
				commit_dst(e, dst + i, yd)
			}

		case .Dot:
			ya0 := read_slot(e, a, 1)
			yb0 := read_slot(e, b, 2, pin_bit(ya0))
			if n == 1 {
				yd := prep_dst(e, dst, 0, pin_bit(ya0) | pin_bit(yb0))
				p3(e, V_FMUL, yd, ya0, yb0)
				commit_dst(e, dst, yd)
			} else {
				p3(e, V_FMUL, 0, ya0, yb0)
				for i in 1 ..< n - 1 {
					yai := read_slot(e, a + i, 1)
					ybi := read_slot(e, b + i, 2, pin_bit(yai))
					p3(e, V_FMUL, 3, yai, ybi)
					p3(e, V_FADD, 0, 0, 3)
				}
				yal := read_slot(e, a + n - 1, 1)
				ybl := read_slot(e, b + n - 1, 2, pin_bit(yal))
				p3(e, V_FMUL, 3, yal, ybl)
				yd := prep_dst(e, dst, 0, pin_bit(yal) | pin_bit(ybl))
				p3(e, V_FADD, yd, 0, 3)
				commit_dst(e, dst, yd)
			}

		case .Length:
			ya0 := read_slot(e, a, 1)
			p3(e, V_FMUL, 0, ya0, ya0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				p3(e, V_FMUL, 2, yai, yai)
				p3(e, V_FADD, 0, 0, 2)
			}
			yd := prep_dst(e, dst, 0)
			p2(e, V_FSQRT, yd, 0)
			commit_dst(e, dst, yd)

		case .Distance:
			ya0 := read_slot(e, a, 1)
			yb0 := read_slot(e, b, 2, pin_bit(ya0))
			p3(e, V_FSUB, 0, ya0, yb0)
			p3(e, V_FMUL, 0, 0, 0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				ybi := read_slot(e, b + i, 2, pin_bit(yai))
				p3(e, V_FSUB, 3, yai, ybi)
				p3(e, V_FMUL, 3, 3, 3)
				p3(e, V_FADD, 0, 0, 3)
			}
			yd := prep_dst(e, dst, 0)
			p2(e, V_FSQRT, yd, 0)
			commit_dst(e, dst, yd)

		case .Normalize:
			ya0 := read_slot(e, a, 0)
			p3(e, V_FMUL, 4, ya0, ya0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 0)
				p3(e, V_FMUL, 5, yai, yai)
				p3(e, V_FADD, 4, 4, 5)
			}
			p2(e, V_FSQRT, 4, 4)
			a_imm(e, FMOV_ONE, 10)
			pq3(e, V_FDIV, 4, 10, 4)
			for i in 0 ..< n {
				yai := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(yai))
				p3(e, V_FMUL, yd, yai, 4)
				commit_dst(e, dst + i, yd)
			}

		case .Cross:
			e.exec_ymask_valid = false
			x0 := read_slot(e, a, 0);     pmov(e, 0, x0)
			x1 := read_slot(e, a + 1, 1); pmov(e, 1, x1)
			x2 := read_slot(e, a + 2, 2); pmov(e, 2, x2)
			y0 := read_slot(e, b, 3);     pmov(e, 3, y0)
			y1 := read_slot(e, b + 1, 4); pmov(e, 4, y1)
			y2 := read_slot(e, b + 2, 5); pmov(e, 5, y2)

			p3(e, V_FMUL, 6, 1, 5)
			p3(e, V_FMUL, 7, 4, 2)
			yd0 := prep_dst(e, dst, 6)
			p3(e, V_FSUB, yd0, 6, 7)
			commit_dst(e, dst, yd0)

			p3(e, V_FMUL, 6, 2, 3)
			p3(e, V_FMUL, 7, 5, 0)
			yd1 := prep_dst(e, dst + 1, 6)
			p3(e, V_FSUB, yd1, 6, 7)
			commit_dst(e, dst + 1, yd1)

			p3(e, V_FMUL, 6, 0, 4)
			p3(e, V_FMUL, 7, 3, 1)
			yd2 := prep_dst(e, dst + 2, 6)
			p3(e, V_FSUB, yd2, 6, 7)
			commit_dst(e, dst + 2, yd2)

		case .Reflect:
			ya0 := read_slot(e, a, 1)
			yb0 := read_slot(e, b, 2, pin_bit(ya0))
			p3(e, V_FMUL, 4, ya0, yb0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				ybi := read_slot(e, b + i, 2, pin_bit(yai))
				p3(e, V_FMUL, 5, yai, ybi)
				p3(e, V_FADD, 4, 4, 5)
			}
			a_imm(e, FMOV_TWO, 12)
			p3q(e, V_FMUL, 4, 4, 12) // 2 * dot(I, N)
			for i in 0 ..< n {
				yai := read_slot(e, a + i, 1)
				ybi := read_slot(e, b + i, 2, pin_bit(yai))
				p3(e, V_FMUL, 5, 4, ybi)
				yd := prep_dst(e, dst + i, 0, pin_bit(yai) | pin_bit(ybi))
				p3(e, V_FSUB, yd, yai, 5)
				commit_dst(e, dst + i, yd)
			}

		case .Faceforward, .Refract:
			a_call(e, rawptr(jit_help_geom), &code[pc])

		case .Mat_Mul:
			rows, inner, cols := n, i32(in_.k), in_.d
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					out_s := dst + col * rows + row
					ya0 := read_slot(e, a + row, 1)
					yb0 := read_slot(e, b + col * inner, 2, pin_bit(ya0))
					if inner == 1 {
						yd := prep_dst(e, out_s, 0, pin_bit(ya0) | pin_bit(yb0))
						p3(e, V_FMUL, yd, ya0, yb0)
						commit_dst(e, out_s, yd)
					} else {
						p3(e, V_FMUL, 0, ya0, yb0)
						for k in 1 ..< inner - 1 {
							yak := read_slot(e, a + k * rows + row, 1)
							ybk := read_slot(e, b + col * inner + k, 2, pin_bit(yak))
							p3(e, V_FMUL, 3, yak, ybk)
							p3(e, V_FADD, 0, 0, 3)
						}
						kl := inner - 1
						yal := read_slot(e, a + kl * rows + row, 1)
						ybl := read_slot(e, b + col * inner + kl, 2, pin_bit(yal))
						p3(e, V_FMUL, 3, yal, ybl)
						yd := prep_dst(e, out_s, 0, pin_bit(yal) | pin_bit(ybl))
						p3(e, V_FADD, yd, 0, 3)
						commit_dst(e, out_s, yd)
					}
				}
			}

		case .Vec_Mat:
			rows, cols := n, i32(in_.k)
			for col in 0 ..< cols {
				ya0 := read_slot(e, a, 1)
				yb0 := read_slot(e, b + col * rows, 2, pin_bit(ya0))
				if rows == 1 {
					yd := prep_dst(e, dst + col, 0, pin_bit(ya0) | pin_bit(yb0))
					p3(e, V_FMUL, yd, ya0, yb0)
					commit_dst(e, dst + col, yd)
				} else {
					p3(e, V_FMUL, 0, ya0, yb0)
					for row in 1 ..< rows - 1 {
						yar := read_slot(e, a + row, 1)
						ybr := read_slot(e, b + col * rows + row, 2, pin_bit(yar))
						p3(e, V_FMUL, 3, yar, ybr)
						p3(e, V_FADD, 0, 0, 3)
					}
					rl := rows - 1
					yal := read_slot(e, a + rl, 1)
					ybl := read_slot(e, b + col * rows + rl, 2, pin_bit(yal))
					p3(e, V_FMUL, 3, yal, ybl)
					yd := prep_dst(e, dst + col, 0, pin_bit(yal) | pin_bit(ybl))
					p3(e, V_FADD, yd, 0, 3)
					commit_dst(e, dst + col, yd)
				}
			}

		case .Transpose:
			rows, cols := n, i32(in_.k)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					ya := read_slot(e, a + col * rows + row, 0)
					copy_to_dst(e, dst + row * cols + col, ya, pin_bit(ya))
				}
			}

		case .Outer:
			rows, cols := n, i32(in_.k)
			for col in 0 ..< cols {
				ybc := read_slot(e, b + col, 2)
				pb := pin_bit(ybc)
				for row in 0 ..< rows {
					yar := read_slot(e, a + row, 1, pb)
					keep := pb if row + 1 < rows else 0
					yd := prep_dst(e, dst + col * rows + row, 0, pb | pin_bit(yar), keep)
					p3(e, V_FMUL, yd, yar, ybc)
					commit_dst(e, dst + col * rows + row, yd)
				}
			}

		case .Determinant, .Inverse:
			a_call(e, rawptr(jit_help_mat), &code[pc])

		case .Deriv_X, .Deriv_Y, .Fwidth:
			// In each quad (one V register) lanes 0, 1 are the top row and
			// 2, 3 the bottom one. x: trn2 - trn1 ([a1 a1 a3 a3] - [a0 a0 a2
			// a2]); y: zip2 - zip1 of the doublewords ([a2 a3 a2 a3] - [a0
			// a1 a0 a1]); fwidth |x| + |y| by fabd (a subtract's rounding,
			// then the abs).
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				if in_.op != .Deriv_Y {
					p3(e, V_TRN2, 2, ya, ya)
					p3(e, V_TRN1, 3, ya, ya)
				}
				if in_.op != .Deriv_X {
					p3(e, V_ZIP2_2D, 4, ya, ya)
					p3(e, V_ZIP1_2D, 5, ya, ya)
				}
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				#partial switch in_.op {
				case .Deriv_X:
					p3(e, V_FSUB, yd, 2, 3)
				case .Deriv_Y:
					p3(e, V_FSUB, yd, 4, 5)
				case:
					p3(e, V_FABD, 2, 2, 3)
					p3(e, V_FABD, 4, 4, 5)
					p3(e, V_FADD, yd, 2, 4)
				}
				commit_dst(e, dst + i, yd)
			}

		case .Texture, .Texture_Bias, .Texel_Fetch:
			e.exec_ymask_valid = false
			comp_mask := e.tex_mask[pc]
			if comp_mask != 0 {
				ya0 := read_slot(e, a, 0)
				pins := pin_bit(ya0)
				ya1 := read_slot(e, a + 1, 1, pins)
				pins |= pin_bit(ya1)
				yb := read_slot(e, b, 2, pins)
				pins |= pin_bit(yb)
				ya2 := u8(3)
				if in_.n == 3 {
					ya2 = read_slot(e, a + 2, 3, pins)
				}
				pmov(e, 0, ya0)
				pmov(e, 1, ya1)
				pmov(e, 2, yb)
				if in_.n == 3 {
					pmov(e, 3, ya2)
				}
			}
			stub := A64_Tex_Stub {
				pc        = i32(pc),
				comp_mask = comp_mask,
			}
			for cs, idx in e.cache {
				if cs.slot < 0 || !cs.dirty {
					continue
				}
				for r in e.rw[pc].reads {
					if cs.slot >= r.start && cs.slot < r.start + r.n {
						stub.dirty_slots[stub.n_dirty] = cs.slot
						stub.dirty_regs[stub.n_dirty] = u8(8 + idx)
						stub.n_dirty += 1
						break
					}
				}
			}
			for &cs, idx in e.cache {
				if cs.slot < 0 {
					continue
				}
				if (cs.slot >= dst && cs.slot < dst + 4) || cs.last_use <= e.pc {
					cs = {slot = -1, last_use = -1, dirty = false, stamp = 0}
				} else {
					stub.save_regs[stub.n_save] = u8(8 + idx)
					stub.n_save += 1
				}
			}
			a_chan_record(e, i32(in_.k))
			a_ldr_x(e, 15, A_X0, OFF_CHAN_PTR)
			stub.unbound_patch = len(e.buf)
			ai(e, 0xb400000f) // cbz x15, unbound_stub
			if comp_mask != 0 {
				a_emit_bound_tex(e, &code[pc], comp_mask)
			}
			stub.ret_off = len(e.buf)
			append(&e.unbound_tex, stub)
			for k in i32(0) ..< 4 {
				if (comp_mask & (u8(1) << uint(k))) != 0 {
					copy_to_dst(e, dst + k, u8(k))
				}
			}

		case .Texture_Size:
			a_call(e, rawptr(jit_help_texture_size), &code[pc])

		case .Load_Buf:
			// A block of its own. Every lane at one index into one of the
			// first 4 buffers, all n words inside it: broadcasts; else the
			// helper.
			if in_.k < 4 {
				k_buf := u32(in_.k)
				ai(e, 0xf9400000 | (u32(OFF_CTX_BUF_PTR) / 8 + k_buf) << 10 | u32(A_CTX) << 5 | u32(A_X16)) // ldr x16, buf_ptr[k]
				ai(e, 0xf9400000 | (u32(OFF_CTX_BUF_LEN) / 8 + k_buf) << 10 | u32(A_CTX) << 5 | u32(A_X2))  // ldr x2, buf_len[k]
				a_pair_mem(e, true, 0, A_SLOTS, i64(a) * 32)
				a_uniform_lane0(e, 0, A_X1)
				div := a_cbz(e, A_X0, false)
				// w1 = index * stride + offset, wrapping in 32 bits.
				if stride := u32(in_.b); stride != 1 {
					a_mov_w(e, A_X3, stride)
					ai(e, 0x1b007c00 | u32(A_X3) << 16 | u32(A_X1) << 5 | u32(A_X1)) // mul w1, w1, w3
				}
				if off := u32(in_.c); off != 0 {
					a_mov_w(e, A_X3, off)
					ai(e, 0x0b000000 | u32(A_X3) << 16 | u32(A_X1) << 5 | u32(A_X1)) // add w1, w1, w3
				}
				// buf_len is at most 2^32, so x1 + n - 1 < buf_len holds iff
				// the n words neither wrap 32 bits nor leave the buffer.
				ai(e, 0x91000000 | u32(n - 1) << 10 | u32(A_X1) << 5 | u32(A_X3)) // add x3, x1, #(n - 1)
				ai(e, 0xeb00001f | u32(A_X2) << 16 | u32(A_X3) << 5)              // cmp x3, x2
				oob := a_branch(e, A_HS)
				ai(e, 0x8b000800 | u32(A_X1) << 16 | u32(A_X16) << 5 | u32(A_X16)) // add x16, x16, x1, lsl #2
				for i in 0 ..< n {
					ai(e, 0xb9400000 | u32(i) << 10 | u32(A_X16) << 5 | u32(A_X3)) // ldr w3, [x16, #4i]
					ai(e, 0x4e040c00 | u32(A_X3) << 5)                              // dup v0.4s, w3
					ai(e, 0x4ea01c01)                                               // mov v1.16b, v0.16b
					a_pair_mem(e, false, 0, A_SLOTS, i64(dst + i) * 32)
				}
				done := a_branch(e, -1)
				a_land(e, div)
				a_land(e, oob)
				a_call(e, rawptr(jit_help_load_buf), &code[pc])
				a_land(e, done)
			} else {
				a_call(e, rawptr(jit_help_load_buf), &code[pc])
			}

		case .Mask_Copy:
			a_copy_mask(e, dst, a)

		case .Mask_Cond:
			if e.fused_mc_pc == i32(pc) {
				e.fused_mc_pc = -1
				if (in_.k == 1) != e.fused_mc_inv {
					ai(e, 0x2a2003e0 | u32(A_X1) << 16 | u32(A_X1)) // mvn w1, w1
				}
			} else {
				yb := read_slot(e, b, 1)
				p2(e, V_CMEQ0, 0, yb) // the lanes where b == 0
				a_movmsk(e, A_X1, 0, 2)
				if in_.k == 0 {
					ai(e, 0x52001c21) // eor w1, w1, #0xff
				}
			}
			if dst == MASK_EXEC && is_mask_eq_exec(e, a) {
				ai(e, 0x0a000000 | u32(A_X1) << 16 | u32(A_EXEC) << 5 | u32(A_EXEC)) // and w22, w22, w1
			} else {
				a_load_mask_w0(e, a)
				ai(e, 0x0a000000 | u32(A_X1) << 16 | u32(A_X0) << 5 | u32(A_X0)) // and w0, w0, w1
				a_store_mask_w0(e, dst)
			}
			invalidate_mask_write(e, dst)

		case .Mask_Restore:
			cands := [4]i32{MASK_DISCARD, b, c, in_.d}
			kills: [4]i32
			n_kills := 0
			for km in cands {
				if km < 0 || !e.mask_written[km] {
					continue
				}
				dup := false
				for prev in kills[:n_kills] {
					if prev == km {
						dup = true
						break
					}
				}
				if !dup {
					kills[n_kills] = km
					n_kills += 1
				}
			}
			if n_kills == 0 {
				a_copy_mask(e, dst, a)
			} else {
				k0 := kills[0]
				if is_mask_eq_exec(e, k0) {
					a_mov_ww(e, A_X1, A_EXEC)
				} else {
					a_ldrb_mask(e, A_X1, k0)
				}
				for km in kills[1:n_kills] {
					a_load_mask_w0(e, km)
					ai(e, 0x2a000000 | u32(A_X0) << 16 | u32(A_X1) << 5 | u32(A_X1)) // orr w1, w1, w0
				}
				if dst == MASK_EXEC && is_mask_eq_exec(e, a) {
					ai(e, 0x0a200000 | u32(A_X1) << 16 | u32(A_EXEC) << 5 | u32(A_EXEC)) // bic w22, w22, w1
				} else {
					a_load_mask_w0(e, a)
					ai(e, 0x0a200000 | u32(A_X1) << 16 | u32(A_X0) << 5 | u32(A_X0)) // bic w0, w0, w1
					a_store_mask_w0(e, dst)
				}
				invalidate_mask_write(e, dst)
			}

		case .Mask_Or:
			a_load_mask_w0(e, b)
			a_mov_ww(e, A_X1, A_X0)
			a_load_mask_w0(e, a)
			ai(e, 0x2a000000 | u32(A_X1) << 16 | u32(A_X0) << 5 | u32(A_X0)) // orr w0, w0, w1
			a_store_mask_w0(e, dst)
			invalidate_mask_write(e, dst)

		case .Mask_Clear:
			if dst == MASK_EXEC {
				a_mov_ww(e, A_EXEC, A_ZR)
			} else {
				a_strb_mask(e, A_ZR, dst)
			}
			invalidate_mask_write(e, dst)

		case .Mask_Kill:
			if dst != MASK_EXEC {
				a_load_mask_w0(e, dst)
				ai(e, 0x2a000000 | u32(A_EXEC) << 16 | u32(A_X0) << 5 | u32(A_X0)) // orr w0, w0, w22
				a_store_mask_w0(e, dst)
			}
			a_mov_ww(e, A_EXEC, A_ZR)
			invalidate_mask_write(e, dst)
			invalidate_mask_write(e, MASK_EXEC)
			if dst == MASK_DISCARD && !is_init {
				// Every entry lane discarded: done.
				ai(e, 0x39400000 | u32(OFF_CTX_LANES) << 10 | u32(A_CTX) << 5 | u32(A_X1)) // ldrb w1, [x19, #lanes]
				ai(e, 0x6a200000 | u32(A_X0) << 16 | u32(A_X1) << 5 | u32(A_ZR))            // bics wzr, w1, w0
				append(&e.epilogues, a_branch(e, A_EQ))
			}

		case .Jump:
			append(&e.patches, Jump_Patch{a_branch(e, -1), a})

		case .Jump_If_None:
			skip := a_cbz(e, A_EXEC, true)
			append(&e.patches, Jump_Patch{a_branch(e, -1), a})
			a_land(e, skip)

		case .Store_Buf:
			a_call(e, rawptr(jit_help_store_buf), &code[pc])
		case .Image_Load:
			a_call(e, rawptr(jit_help_image_load), &code[pc])
		case .Image_Store:
			a_call(e, rawptr(jit_help_image_store), &code[pc])
		case .Image_Size:
			a_call(e, rawptr(jit_help_image_size), &code[pc])
		case .Load_Shared:
			a_call(e, rawptr(jit_help_load_shared), &code[pc])
		case .Store_Shared:
			a_call(e, rawptr(jit_help_store_shared), &code[pc])
		case .Atomic_Shared:
			a_call(e, rawptr(jit_help_atomic_shared), &code[pc])
		case .Atomic_Buf:
			a_call(e, rawptr(jit_help_atomic_buf), &code[pc])
		case .Barrier:
			// Only compiled when one batch is the whole workgroup (jit_compute_eligible).
		}
	}

	pc_off[n_code] = len(e.buf)
	epilogue_off := len(e.buf)
	e.epilogue_off = epilogue_off

	ai(e, 0x6d473fee) // ldp d14, d15, [sp, #112]
	ai(e, 0x6d4637ec) // ldp d12, d13, [sp, #96]
	ai(e, 0x6d452fea) // ldp d10, d11, [sp, #80]
	ai(e, 0x6d4427e8) // ldp d8, d9, [sp, #64]
	ai(e, 0xf9401bf7) // ldr x23, [sp, #48]
	ai(e, 0xa9425bf5) // ldp x21, x22, [sp, #32]
	ai(e, 0xa94153f3) // ldp x19, x20, [sp, #16]
	ai(e, 0xa8c87bfd) // ldp x29, x30, [sp], #128
	ai(e, 0xd65f03c0) // ret

	for jp in e.patches {
		a_patch(e, jp.patch_pos, pc_off[jp.target_pc])
	}
	for ep in e.epilogues {
		a_patch(e, ep, epilogue_off)
	}
	for &stub in e.unbound_tex {
		stub.stub_off = len(e.buf)
		a_land(e, stub.unbound_patch)
		in_ := code[stub.pc]
		for i in 0 ..< int(stub.n_dirty) {
			jit_st_slot(e, stub.dirty_slots[i], stub.dirty_regs[i])
		}
		n_save := int(stub.n_save)
		frame := u32(n_save * 32)
		if n_save > 0 {
			ai(e, 0xd10003ff | frame << 10) // sub sp, sp, #frame
			for i in 0 ..< n_save {
				a_pair_mem(e, false, stub.save_regs[i], A_SP, i64(i * 32))
			}
		}
		if p.stage == .Compute {
			ai(e, 0x39000000 | u32(OFF_CTX_LANES) << 10 | u32(A_CTX) << 5 | u32(A_EXEC)) // strb w22, [x19, #lanes]
		}
		fn := rawptr(jit_help_texel_fetch) if in_.op == .Texel_Fetch else rawptr(jit_help_texture)
		a_call(e, fn, &code[stub.pc])
		if n_save > 0 {
			for i in 0 ..< n_save {
				a_pair_mem(e, true, stub.save_regs[i], A_SP, i64(i * 32))
			}
			ai(e, 0x910003ff | frame << 10) // add sp, sp, #frame
		}
		for k in u8(0) ..< 4 {
			if (stub.comp_mask & (u8(1) << k)) != 0 {
				a_pair_mem(e, true, k, A_SLOTS, i64(in_.dst + i32(k)) * 32)
			}
		}
		back := a_branch(e, -1)
		a_patch(e, back, stub.ret_off)
		stub.stub_end = len(e.buf)
	}
	// The bound channels' subroutines the code calls (decode_sub last: the
	// others call it).
	if len(e.sample_calls) > 0 {
		e.chan_sample_off = len(e.buf)
		a_emit_sample_sub(e)
		for cp in e.sample_calls {a_patch(e, cp, e.chan_sample_off)}
	}
	if len(e.bilinear_calls) > 0 {
		e.chan_bilinear_off = len(e.buf)
		a_emit_bilinear_sub(e)
		for cp in e.bilinear_calls {a_patch(e, cp, e.chan_bilinear_off)}
	}
	if len(e.fetch_calls) > 0 {
		e.chan_fetch_off = len(e.buf)
		a_emit_fetch_sub(e)
		for cp in e.fetch_calls {a_patch(e, cp, e.chan_fetch_off)}
	}
	if len(e.decode_calls) > 0 {
		e.chan_decode_off = len(e.buf)
		a_emit_decode_sub(e)
		for cp in e.decode_calls {a_patch(e, cp, e.chan_decode_off)}
	}
}

}
