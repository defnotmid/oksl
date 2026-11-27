package oksl

import "base:intrinsics"
import "base:runtime"
import "core:math"

// CPU backend, part 3b: the x86-64 AVX2 emitter (vm_jit.odin has the
// shared front end and the runners).

_ :: intrinsics
_ :: runtime
_ :: math

when JIT_X64 {

// Between the init and the body functions: int3.
@(private)
JIT_PAD :: u8(0xCC)

@(private)
Jit_Tables :: struct #align(32) {
	mask_lut:      [256][8]i8,
	sign_mask:     [8]u32,
	abs_mask:      [8]u32,
	pos_inf:       [8]u32,
	one_u:         [8]u32,
	three_u:       [8]u32,
	mask31_u:      [8]u32,
	all_ones:      [8]u32,
	lane_iota:     [8]u32,
	zero_f:        [8]f32,
	one_f:         [8]f32,
	neg_one_f:     [8]f32,
	two_f:         [8]f32,
	three_f:       [8]f32,
	deg_to_rad:    [8]f32,
	rad_to_deg:    [8]f32,
	f2i_hi:        [8]f32,
	f2u_hi:        [8]f32,
	half_below:    [8]f32, // 0.49999997, the f32 below 0.5
	i32_max_f:     [8]f32,
	neg_i32_max_f: [8]f32,
	f255:          [8]f32,
	byte_mask:     [8]u32,
	canon_nan:     [8]u32,
	two52_d:       [4]u64,
	// Bound channels (sample_sub, decode_sub).
	i256:          [8]u32,
	not3:          [8]u32,
	u16_mask:      [8]u32,
	h_em:          [8]u32,
	h_sign:        [8]u32,
	h_bias:        [8]u32,
	h_sub:         [8]f32,
	h_sublim:      [8]u32,
	h_inflim:      [8]u32,
	h_mant:        [8]u32,
	h_quiet:       [8]u32,
	f256:          [8]f32,
	inv256:        [8]f32,
	d256:          [4]f64,
	d_half:        [4]f64,
	d128:          [4]f64,
	d_lo:          [4]f64,
	d_hi:          [4]f64,
	// vpshufb: in each 128-bit lane, the four dwords' low words, then
	// their high words (fetch_one's F16C path).
	h_pack:        [32]u8,
	// quad_level8 constants (appended so existing OFF_TAB_* stay identical).
	ql_flt_min:    [8]u32,
	ql_two24:      [8]f32,
	ql_sqrt_half:  [8]u32,
	ql_c:          [6][8]u32,
	ql_24:         [8]f32,
	ql_half:       [8]f32,
	ql_neg_inf:    [8]u32,
	// Packed u16 0x00FF and 2^-16 scale for RGBA8/Gray8 integer bilinear.
	w_00ff:        [8]u32,
	inv65536:      [8]f32,
}

#assert(offset_of(Jit_Tables, sign_mask) % 32 == 0)
#assert(offset_of(Jit_Tables, two52_d) % 32 == 0)
#assert(offset_of(Jit_Tables, ql_flt_min) % 32 == 0)
#assert(offset_of(Jit_Tables, w_00ff) % 32 == 0)

// FMA3, for the bound channels' exact products in f32 (else f64).
@(private)
jit_fma_ready: bool

// F16C, for fetch_one's half decode (else x_half's).
@(private)
jit_f16c_ready: bool

@(private)
jit_tables: Jit_Tables

@(private)
jit_x64_init :: proc "contextless" () -> bool {
	for m in 0 ..< 256 {
		for l in 0 ..< 8 {
			jit_tables.mask_lut[m][l] = -1 if (m & (1 << uint(l))) != 0 else 0
		}
	}
	jit_tables.sign_mask = 0x8000_0000
	jit_tables.abs_mask = 0x7fff_ffff
	jit_tables.pos_inf = 0x7f80_0000
	jit_tables.one_u = 1
	jit_tables.three_u = 3
	jit_tables.mask31_u = 31
	jit_tables.all_ones = 0xffff_ffff
	jit_tables.lane_iota = {0, 1, 2, 3, 4, 5, 6, 7}
	jit_tables.half_below = 0.49999997
	jit_tables.one_f = 1.0
	jit_tables.neg_one_f = -1.0
	jit_tables.two_f = 2.0
	jit_tables.three_f = 3.0
	jit_tables.deg_to_rad = math.PI / 180.0
	jit_tables.rad_to_deg = 180.0 / math.PI
	jit_tables.i32_max_f = MAX_TEXEL_F32
	jit_tables.neg_i32_max_f = -MAX_TEXEL_F32
	jit_tables.f2i_hi = 2147483648.0
	jit_tables.f2u_hi = 4294967296.0
	jit_tables.two52_d = 0x4330_0000_0000_0000
	jit_tables.f255 = 255.0
	jit_tables.byte_mask = 0xFF
	jit_tables.i256 = 256
	jit_tables.not3 = 0xFFFF_FFFC
	jit_tables.u16_mask = 0xFFFF
	jit_tables.h_em = 0x7FFF
	jit_tables.h_sign = 0x8000
	jit_tables.h_bias = (127 - 15) << 23
	jit_tables.h_sub = 1.0 / 16777216
	jit_tables.h_sublim = 0x400
	jit_tables.h_inflim = 0x7BFF
	jit_tables.h_mant = 0x3FF
	jit_tables.h_quiet = 0x40_0000
	jit_tables.f256 = 256
	jit_tables.inv256 = 1.0 / 256
	jit_tables.d256 = 256
	jit_tables.d_half = 0.5
	jit_tables.d128 = 128
	jit_tables.d_lo = -2147483648.0
	jit_tables.d_hi = 2147483646.0
	jit_tables.canon_nan = 0x7fc0_0000
	for i in 0 ..< 32 {
		// Byte j of a lane: of dword (j & 7) / 2, its low word's (j < 8)
		// or its high word's.
		j := i & 15
		jit_tables.h_pack[i] = u8(((j & 7) / 2) * 4 + (2 if j >= 8 else 0) + (j & 1))
	}
	jit_tables.ql_flt_min = 0x0080_0000
	jit_tables.ql_two24 = 16777216
	jit_tables.ql_sqrt_half = 0x3f35_04f3
	ql_c := [6]u32{0x3FB8AA72, 0xBF38A0F9, 0x3EF5A9EC, 0xBEBBA56E, 0x3EA2EAB7, 0xBE531AE5}
	for v, i in ql_c {
		jit_tables.ql_c[i] = v
	}
	jit_tables.ql_24 = 24
	jit_tables.ql_half = 0.5
	jit_tables.ql_neg_inf = 0xff80_0000
	jit_tables.w_00ff = 0x00FF_00FF
	jit_tables.inv65536 = 1.0 / 65536.0

	max_id, _, _, _ := intrinsics.x86_cpuid(0, 0)
	if max_id < 7 {
		return false
	}
	_, _, ecx1, _ := intrinsics.x86_cpuid(1, 0)
	avx_bits :: u32((1 << 26) | (1 << 27) | (1 << 28))
	if (ecx1 & avx_bits) != avx_bits {
		return false
	}
	xcr0, _ := intrinsics.x86_xgetbv(0)
	if (xcr0 & 0x6) != 0x6 {
		return false
	}
	_, ebx7, _, _ := intrinsics.x86_cpuid(7, 0)
	jit_fma_ready = (ecx1 & (1 << 12)) != 0
	jit_f16c_ready = (ecx1 & (1 << 29)) != 0
	return (ebx7 & (1 << 5)) != 0
}

@(private) OFF_TAB_SIGN :: i32(offset_of(Jit_Tables, sign_mask))
@(private) OFF_TAB_ABS :: i32(offset_of(Jit_Tables, abs_mask))
@(private) OFF_TAB_INF :: i32(offset_of(Jit_Tables, pos_inf))
@(private) OFF_TAB_ONE_U :: i32(offset_of(Jit_Tables, one_u))
@(private) OFF_TAB_THREE_U :: i32(offset_of(Jit_Tables, three_u))
@(private) OFF_TAB_MASK31 :: i32(offset_of(Jit_Tables, mask31_u))
@(private) OFF_TAB_ALL1 :: i32(offset_of(Jit_Tables, all_ones))
@(private) OFF_TAB_IOTA :: i32(offset_of(Jit_Tables, lane_iota))
@(private) OFF_TAB_HALF_BELOW :: i32(offset_of(Jit_Tables, half_below))
@(private) OFF_TAB_ONE_F :: i32(offset_of(Jit_Tables, one_f))
@(private) OFF_TAB_NEG1_F :: i32(offset_of(Jit_Tables, neg_one_f))
@(private) OFF_TAB_TWO_F :: i32(offset_of(Jit_Tables, two_f))
@(private) OFF_TAB_THREE_F :: i32(offset_of(Jit_Tables, three_f))
@(private) OFF_TAB_DEG2RAD :: i32(offset_of(Jit_Tables, deg_to_rad))
@(private) OFF_TAB_RAD2DEG :: i32(offset_of(Jit_Tables, rad_to_deg))
@(private) OFF_TAB_I32MAX :: i32(offset_of(Jit_Tables, i32_max_f))
@(private) OFF_TAB_NEG_I32MAX :: i32(offset_of(Jit_Tables, neg_i32_max_f))
@(private) OFF_TAB_F2I_HI :: i32(offset_of(Jit_Tables, f2i_hi))
@(private) OFF_TAB_F2U_HI :: i32(offset_of(Jit_Tables, f2u_hi))
@(private) OFF_TAB_TWO52_D :: i32(offset_of(Jit_Tables, two52_d))
@(private) OFF_TAB_F255 :: i32(offset_of(Jit_Tables, f255))
@(private) OFF_TAB_BYTE :: i32(offset_of(Jit_Tables, byte_mask))
@(private) OFF_TAB_CANON :: i32(offset_of(Jit_Tables, canon_nan))
@(private) OFF_TAB_H_PACK :: i32(offset_of(Jit_Tables, h_pack))
@(private) OFF_TAB_QL_FLT_MIN :: i32(offset_of(Jit_Tables, ql_flt_min))
@(private) OFF_TAB_QL_TWO24 :: i32(offset_of(Jit_Tables, ql_two24))
@(private) OFF_TAB_QL_SQRT_HALF :: i32(offset_of(Jit_Tables, ql_sqrt_half))
@(private) OFF_TAB_QL_C :: i32(offset_of(Jit_Tables, ql_c))
@(private) OFF_TAB_QL_24 :: i32(offset_of(Jit_Tables, ql_24))
@(private) OFF_TAB_QL_HALF :: i32(offset_of(Jit_Tables, ql_half))
@(private) OFF_TAB_QL_NEG_INF :: i32(offset_of(Jit_Tables, ql_neg_inf))
@(private) OFF_TAB_W00FF :: i32(offset_of(Jit_Tables, w_00ff))
@(private) OFF_TAB_INV65536 :: i32(offset_of(Jit_Tables, inv65536))

@(private) OFF_CI_NEAR_ROUTE :: i32(offset_of(Chan_Img, near_route))
@(private) OFF_CI_FS32 :: i32(offset_of(Chan_Img, fs32))
@(private) OFF_CI_FMAX32 :: i32(offset_of(Chan_Img, fmax32))
@(private) OFF_CI_LIN_F32 :: i32(offset_of(Chan_Img, lin_f32))
@(private) OFF_CI_FN512 :: i32(offset_of(Chan_Img, fn512))

@(private)
Bound_Tex_Stub :: struct {
	bound_patch: int,
	ret_off:     int,
	stub_off:    int,
	stub_end:    int,
	pc:          i32,
	comp_mask:   u8,
}

@(private)
Load_Buf_Stub :: struct {
	unif_patch: int,
	big_patch:  int,
	ret_off:    int,
	stub_off:   int,
	stub_end:   int,
	pc:         i32,
	yd:         [4]u8,
}

@(private)
Jit_Arch_Emitter :: struct {
	// Slots holding a constant that isn't NaN (Program.consts; nothing
	// else writes them).
	num_const:        []bool,
	helper_ymm_calls: [dynamic]int,
	unif_oob_calls:   [dynamic]int,
	big_buf_calls:    [dynamic]int,
	bound_tex:        [dynamic]Bound_Tex_Stub,
	load_bufs:        [dynamic]Load_Buf_Stub,
	helper_ymm_off:   int,
	unif_oob_off:     int,
	big_buf_off:      int,
	fma:              bool, // FMA3 for the bound channels' exact products
	f16c:             bool, // F16C for fetch_one's halves
}

@(private)
jit_arch_init :: proc(e: ^Jit_Emitter, p: ^Program, num_slots: int, fma, f16c: bool, heap: runtime.Allocator) {
	e.num_const = make([]bool, num_slots, heap)
	for c in p.consts {
		e.num_const[c[0]] = (c[1] & 0x7fff_ffff) <= 0x7f80_0000
	}
	e.helper_ymm_calls = make([dynamic]int, 0, 8, heap)
	e.unif_oob_calls = make([dynamic]int, 0, 8, heap)
	e.big_buf_calls = make([dynamic]int, 0, 8, heap)
	e.bound_tex = make([dynamic]Bound_Tex_Stub, 0, 8, heap)
	e.load_bufs = make([dynamic]Load_Buf_Stub, 0, 8, heap)
	e.fma = fma && jit_fma_ready
	e.f16c = f16c && jit_f16c_ready
}

@(private)
jit_arch_reset :: proc(e: ^Jit_Emitter) {
	clear(&e.helper_ymm_calls)
	clear(&e.unif_oob_calls)
	clear(&e.big_buf_calls)
	clear(&e.bound_tex)
	clear(&e.load_bufs)
	e.helper_ymm_off = 0
	e.unif_oob_off = 0
	e.big_buf_off = 0
}

@(private)
jit_arch_destroy :: proc(e: ^Jit_Emitter, heap: runtime.Allocator) {
	delete(e.num_const, heap)
	delete(e.helper_ymm_calls)
	delete(e.unif_oob_calls)
	delete(e.big_buf_calls)
	delete(e.bound_tex)
	delete(e.load_bufs)
}

// The slot cache's moves (vm_jit.odin): a slot to and from a register, and
// a register to another.
@(private)
jit_ld_slot :: #force_inline proc(e: ^Jit_Emitter, dst: u8, slot: i32) {vmovaps_load(&e.buf, dst, slot)}
@(private)
jit_st_slot :: #force_inline proc(e: ^Jit_Emitter, slot: i32, src: u8) {vmovaps_store(&e.buf, slot, src)}
@(private)
jit_mov :: #force_inline proc(e: ^Jit_Emitter, dst, src: u8) {vmovaps_rr(&e.buf, dst, src)}

// ---- x86-64 AVX2 machine-code emitter ---------------------------------------------------

// Pinned registers inside generated functions (System V AMD64 ABI):
//   rbp (5)  : ^Jit_Ctx
//   rbx (3)  : [^]Jit_Slot (32-byte aligned)
//   r12 (12) : [^]u8       (masks; masks[1] = DISCARD, masks[2..] = saved masks)
//   r13 (13) : u32         (live MASK_EXEC in 0..255, loaded from ctx.lanes)
//   r14 (14) : ^Jit_Tables (32-byte aligned lookup tables & vector constants)
//   ymm0..7  : scratch / work registers
//   ymm8..15 : basic-block slot register cache
@(private)
GPR_RAX :: u8(0)
@(private)
GPR_RCX :: u8(1)
@(private)
GPR_RDX :: u8(2)
@(private)
GPR_RBX :: u8(3)
@(private)
GPR_RSP :: u8(4)
@(private)
GPR_RBP :: u8(5)
@(private)
GPR_RSI :: u8(6)
@(private)
GPR_RDI :: u8(7)
@(private)
GPR_R12 :: u8(12)
@(private)
GPR_R13 :: u8(13)
@(private)
GPR_R14 :: u8(14)

@(private)
emit_modrm_reg :: #force_inline proc(buf: ^[dynamic]u8, reg, rm: u8) {
	append(buf, 0xC0 | ((reg & 7) << 3) | (rm & 7))
}

@(private)
emit_modrm_mem :: proc(buf: ^[dynamic]u8, reg, base: u8, disp: i32) {
	b7 := base & 7
	r7 := (reg & 7) << 3
	if disp == 0 && b7 != 5 {
		append(buf, r7 | b7)
		if b7 == 4 {
			append(buf, 0x24)
		}
	} else if disp >= -128 && disp <= 127 {
		append(buf, 0x40 | r7 | b7)
		if b7 == 4 {
			append(buf, 0x24)
		}
		append(buf, u8(i8(disp)))
	} else {
		append(buf, 0x80 | r7 | b7)
		if b7 == 4 {
			append(buf, 0x24)
		}
		emit_i32(buf, disp)
	}
}

@(private)
emit_modrm_sib :: proc(buf: ^[dynamic]u8, reg, base, index, scale_shift: u8, disp: i32) {
	b7 := base & 7
	i7 := index & 7
	r7 := (reg & 7) << 3
	sib := (scale_shift << 6) | (i7 << 3) | b7
	if disp == 0 && b7 != 5 {
		append(buf, r7 | 4, sib)
	} else if disp >= -128 && disp <= 127 {
		append(buf, 0x40 | r7 | 4, sib, u8(i8(disp)))
	} else {
		append(buf, 0x80 | r7 | 4, sib)
		emit_i32(buf, disp)
	}
}

@(private)
emit_vex_rr :: proc(buf: ^[dynamic]u8, mmmmm, pp, L, W, op, reg, vvvv, rm: u8) {
	if mmmmm == 1 && W == 0 && (rm & 8) == 0 {
		b1 := ((~reg & 8) << 4) | ((~vvvv & 0xF) << 3) | (L << 2) | pp
		append(buf, 0xC5, b1, op)
	} else {
		b1 := ((~reg & 8) << 4) | 0x40 | ((~rm & 8) << 2) | mmmmm
		b2 := (W << 7) | ((~vvvv & 0xF) << 3) | (L << 2) | pp
		append(buf, 0xC4, b1, b2, op)
	}
	emit_modrm_reg(buf, reg, rm)
}

@(private)
emit_vex_rm :: proc(buf: ^[dynamic]u8, mmmmm, pp, L, W, op, reg, vvvv, base: u8, disp: i32) {
	if mmmmm == 1 && W == 0 && (base & 8) == 0 {
		b1 := ((~reg & 8) << 4) | ((~vvvv & 0xF) << 3) | (L << 2) | pp
		append(buf, 0xC5, b1, op)
	} else {
		b1 := ((~reg & 8) << 4) | 0x40 | ((~base & 8) << 2) | mmmmm
		b2 := (W << 7) | ((~vvvv & 0xF) << 3) | (L << 2) | pp
		append(buf, 0xC4, b1, b2, op)
	}
	emit_modrm_mem(buf, reg, base, disp)
}

@(private)
emit_vex_sib :: proc(buf: ^[dynamic]u8, mmmmm, pp, L, W, op, reg, vvvv, base, index, scale_shift: u8, disp: i32) {
	b1 := ((~reg & 8) << 4) | ((~index & 8) << 3) | ((~base & 8) << 2) | mmmmm
	b2 := (W << 7) | ((~vvvv & 0xF) << 3) | (L << 2) | pp
	append(buf, 0xC4, b1, b2, op)
	emit_modrm_sib(buf, reg, base, index, scale_shift, disp)
}

// Common 256-bit AVX / AVX2 instruction shorthands.
@(private)
vmovaps_rr :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {
	if dst != src {
		emit_vex_rr(buf, 1, 0, 1, 0, 0x28, dst, 0, src)
	}
}

@(private)
vmovaps_load :: #force_inline proc(buf: ^[dynamic]u8, dst: u8, slot: i32) {
	emit_vex_rm(buf, 1, 0, 1, 0, 0x28, dst, 0, GPR_RBX, slot * 32)
}

@(private)
vmovaps_store :: #force_inline proc(buf: ^[dynamic]u8, slot: i32, src: u8) {
	emit_vex_rm(buf, 1, 0, 1, 0, 0x29, src, 0, GPR_RBX, slot * 32)
}

@(private)
vzero :: #force_inline proc(buf: ^[dynamic]u8, dst: u8) {
	emit_vex_rr(buf, 1, 0, 1, 0, 0x57, dst, dst, dst)
}

@(private)
vop_ps :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, b: u8) {
	emit_vex_rr(buf, 1, 0, 1, 0, op, dst, a, b)
}

@(private)
vop_ps_tab :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a: u8, tab_off: i32) {
	emit_vex_rm(buf, 1, 0, 1, 0, op, dst, a, GPR_R14, tab_off)
}

@(private)
vop_d :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, b: u8) {
	emit_vex_rr(buf, 1, 1, 1, 0, op, dst, a, b)
}

@(private)
vop_d_tab :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a: u8, tab_off: i32) {
	emit_vex_rm(buf, 1, 1, 1, 0, op, dst, a, GPR_R14, tab_off)
}

@(private)
vop38_d :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, b: u8) {
	emit_vex_rr(buf, 2, 1, 1, 0, op, dst, a, b)
}

@(private)
vcmpps :: #force_inline proc(buf: ^[dynamic]u8, dst, a, b, pred: u8) {
	emit_vex_rr(buf, 1, 0, 1, 0, 0xC2, dst, a, b)
	append(buf, pred)
}

@(private)
vcmpps_tab :: #force_inline proc(buf: ^[dynamic]u8, dst, a: u8, tab_off: i32, pred: u8) {
	emit_vex_rm(buf, 1, 0, 1, 0, 0xC2, dst, a, GPR_R14, tab_off)
	append(buf, pred)
}

// vpslld (ext 6), vpsrld (ext 2) or vpsrad (ext 4) dst, src, imm.
@(private)
vshift_imm :: #force_inline proc(buf: ^[dynamic]u8, ext, dst, src, imm: u8) {
	emit_vex_rr(buf, 1, 1, 1, 0, 0x72, ext, dst, src)
	append(buf, imm)
}

@(private)
vroundps :: #force_inline proc(buf: ^[dynamic]u8, dst, src, mode: u8) {
	emit_vex_rr(buf, 3, 1, 1, 0, 0x08, dst, 0, src)
	append(buf, mode)
}

@(private)
vpblendvb :: #force_inline proc(buf: ^[dynamic]u8, dst, false_v, true_v, mask_v: u8) {
	emit_vex_rr(buf, 3, 1, 1, 0, 0x4C, dst, false_v, true_v)
	append(buf, mask_v << 4)
}

@(private)
vblendvps :: #force_inline proc(buf: ^[dynamic]u8, dst, false_v, true_v, mask_v: u8) {
	emit_vex_rr(buf, 3, 1, 1, 0, 0x4A, dst, false_v, true_v)
	append(buf, mask_v << 4)
}

// Expands live `r13d` (u8 exec mask) into 8x32-bit lane masks in `dst` via `jit_tables.mask_lut`.
@(private)
emit_exec_ymask :: #force_inline proc(buf: ^[dynamic]u8, dst: u8) {
	emit_vex_sib(buf, 2, 1, 1, 0, 0x21, dst, 0, GPR_R14, GPR_R13, 3, 0)
}

@(private)
vbroadcastss_mem :: #force_inline proc(buf: ^[dynamic]u8, dst, base: u8, disp: i32) {
	emit_vex_rm(buf, 2, 1, 1, 0, 0x18, dst, 0, base, disp)
}

@(private)
vpbroadcastd_mem :: #force_inline proc(buf: ^[dynamic]u8, dst, base: u8, disp: i32) {
	emit_vex_rm(buf, 2, 1, 1, 0, 0x58, dst, 0, base, disp)
}

// ---- bound channels: the sampling subroutines ------------------------------------------
//
// One rule in both engines (Channel_Filter): sample_sub taps each level some
// lane uses (ascending), for every lane, and keeps each lane's l0 and l0 + 1
// taps to blend; a tap's coordinates are scaled in f64, its blends f32.

@(private) OFF_TAB_ZERO :: i32(offset_of(Jit_Tables, zero_f))
@(private) OFF_TAB_I256 :: i32(offset_of(Jit_Tables, i256))
@(private) OFF_TAB_NOT3 :: i32(offset_of(Jit_Tables, not3))
@(private) OFF_TAB_U16 :: i32(offset_of(Jit_Tables, u16_mask))
@(private) OFF_TAB_H_EM :: i32(offset_of(Jit_Tables, h_em))
@(private) OFF_TAB_H_SIGN :: i32(offset_of(Jit_Tables, h_sign))
@(private) OFF_TAB_H_BIAS :: i32(offset_of(Jit_Tables, h_bias))
@(private) OFF_TAB_H_SUB :: i32(offset_of(Jit_Tables, h_sub))
@(private) OFF_TAB_H_SUBLIM :: i32(offset_of(Jit_Tables, h_sublim))
@(private) OFF_TAB_H_INFLIM :: i32(offset_of(Jit_Tables, h_inflim))
@(private) OFF_TAB_H_MANT :: i32(offset_of(Jit_Tables, h_mant))
@(private) OFF_TAB_H_QUIET :: i32(offset_of(Jit_Tables, h_quiet))
@(private) OFF_TAB_F256 :: i32(offset_of(Jit_Tables, f256))
@(private) OFF_TAB_D256 :: i32(offset_of(Jit_Tables, d256))
@(private) OFF_TAB_DHALF :: i32(offset_of(Jit_Tables, d_half))
@(private) OFF_TAB_D128 :: i32(offset_of(Jit_Tables, d128))
@(private) OFF_TAB_INV256 :: i32(offset_of(Jit_Tables, inv256))
@(private) OFF_TAB_DLO :: i32(offset_of(Jit_Tables, d_lo))
@(private) OFF_TAB_DHI :: i32(offset_of(Jit_Tables, d_hi))

// 256-bit loads and stores, unaligned.
@(private) x_ld :: #force_inline proc(buf: ^[dynamic]u8, dst, base: u8, disp: i32) {emit_vex_rm(buf, 1, 0, 1, 0, 0x10, dst, 0, base, disp)}
@(private) x_st :: #force_inline proc(buf: ^[dynamic]u8, base: u8, disp: i32, src: u8) {emit_vex_rm(buf, 1, 0, 1, 0, 0x11, src, 0, base, disp)}
// Packed doubles (66 0F op), register and memory forms.
@(private) x_pd :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, b: u8) {emit_vex_rr(buf, 1, 1, 1, 0, op, dst, a, b)}
@(private) x_pd_m :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, base: u8, disp: i32) {emit_vex_rm(buf, 1, 1, 1, 0, op, dst, a, base, disp)}
// Packed dwords with a 256-bit memory operand (66 0F op, 66 0F38 op).
@(private) x_d_m :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, base: u8, disp: i32) {emit_vex_rm(buf, 1, 1, 1, 0, op, dst, a, base, disp)}
@(private) x_d38_m :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, base: u8, disp: i32) {emit_vex_rm(buf, 2, 1, 1, 0, op, dst, a, base, disp)}
@(private) x_movdqa_tab :: #force_inline proc(buf: ^[dynamic]u8, dst: u8, off: i32) {emit_vex_rm(buf, 1, 1, 1, 0, 0x6F, dst, 0, GPR_R14, off)}
@(private) x_roundpd :: #force_inline proc(buf: ^[dynamic]u8, dst, src, mode: u8) {emit_vex_rr(buf, 3, 1, 1, 0, 0x09, dst, 0, src); append(buf, mode)}
@(private) x_cmppd :: #force_inline proc(buf: ^[dynamic]u8, dst, a, b, pred: u8) {emit_vex_rr(buf, 1, 1, 1, 0, 0xC2, dst, a, b); append(buf, pred)}
@(private) x_blendvpd :: #force_inline proc(buf: ^[dynamic]u8, dst, f, t, m: u8) {emit_vex_rr(buf, 3, 1, 1, 0, 0x4B, dst, f, t); append(buf, m << 4)}
// vcvtps2pd ymm, xmm / m128; vcvtdq2pd ymm, xmm; vcvttpd2dq xmm, ymm;
// vcvtpd2ps xmm, ymm; vcvttps2dq and vcvtdq2ps ymm, ymm.
@(private) x_ps2pd :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {emit_vex_rr(buf, 1, 0, 1, 0, 0x5A, dst, 0, src)}
@(private) x_ps2pd_m :: #force_inline proc(buf: ^[dynamic]u8, dst, base: u8, disp: i32) {emit_vex_rm(buf, 1, 0, 1, 0, 0x5A, dst, 0, base, disp)}
@(private) x_dq2pd :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {emit_vex_rr(buf, 1, 2, 1, 0, 0xE6, dst, 0, src)}
@(private) x_pd2dq :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {emit_vex_rr(buf, 1, 1, 1, 0, 0xE6, dst, 0, src)}
@(private) x_pd2ps :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {emit_vex_rr(buf, 1, 1, 1, 0, 0x5A, dst, 0, src)}
@(private) x_ps2dq_t :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {emit_vex_rr(buf, 1, 2, 1, 0, 0x5B, dst, 0, src)}
@(private) x_dq2ps :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {emit_vex_rr(buf, 1, 0, 1, 0, 0x5B, dst, 0, src)}
// vextractf128 xmm, ymm, 1 and vinsertf128 ymm, ymm, xmm, 1.
@(private) x_hi128 :: #force_inline proc(buf: ^[dynamic]u8, dst, src: u8) {emit_vex_rr(buf, 3, 1, 1, 0, 0x19, src, 0, dst); append(buf, 1)}
@(private) x_ins128 :: #force_inline proc(buf: ^[dynamic]u8, dst, lo, hi: u8) {emit_vex_rr(buf, 3, 1, 1, 0, 0x18, dst, lo, hi); append(buf, 1)}
@(private) x_bcast_sd :: #force_inline proc(buf: ^[dynamic]u8, dst, base: u8, disp: i32) {emit_vex_rm(buf, 2, 1, 1, 0, 0x19, dst, 0, base, disp)}
// vmovd xmm, r32 then vpbroadcastd ymm, xmm.
@(private) x_bcast_gpr :: #force_inline proc(buf: ^[dynamic]u8, dst, gpr: u8) {
	emit_vex_rr(buf, 1, 1, 0, 0, 0x6E, dst, 0, gpr)
	emit_vex_rr(buf, 2, 1, 1, 0, 0x58, dst, 0, dst)
}
// Gathers dwords [base + idx << scale + disp] for every lane (mask: all ones).
@(private) x_gather :: #force_inline proc(buf: ^[dynamic]u8, dst, base, idx, mask, scale: u8, disp: i32) {
	vop_d(buf, 0x76, mask, mask, mask)
	vzero(buf, dst)
	emit_vex_sib(buf, 2, 1, 1, 0, 0x90, dst, mask, base, idx, scale, disp)
}
@(private) x_call :: #force_inline proc(buf: ^[dynamic]u8, calls: ^[dynamic]int) {
	emit_b(buf, 0xE8)
	append(calls, len(buf^))
	emit_i32(buf, 0)
}

// Packed words (66 0F op) and word shifts vpsllw (ext 6) / vpsrlw (ext 2).
@(private) vop_w :: #force_inline proc(buf: ^[dynamic]u8, op, dst, a, b: u8) {emit_vex_rr(buf, 1, 1, 1, 0, op, dst, a, b)}
@(private)
vshift_w_imm :: #force_inline proc(buf: ^[dynamic]u8, ext, dst, src, imm: u8) {
	emit_vex_rr(buf, 1, 1, 1, 0, 0x71, ext, dst, src)
	append(buf, imm)
}
// Unaligned 128-bit SIB load (vmovups xmm, [base + idx]) and high insert (vinsertf128 ymm, ymm, [base + idx], 1).
@(private) x_vmovups_xmm_sib :: #force_inline proc(buf: ^[dynamic]u8, dst, base, idx: u8) {emit_vex_sib(buf, 1, 0, 0, 0, 0x10, dst, 0, base, idx, 0, 0)}
@(private)
x_vinsertf128_sib :: #force_inline proc(buf: ^[dynamic]u8, dst, src, base, idx: u8) {
	emit_vex_sib(buf, 3, 1, 1, 0, 0x18, dst, src, base, idx, 0, 0)
	append(buf, 1)
}
@(private)
x_shufps :: #force_inline proc(buf: ^[dynamic]u8, dst, a, b, imm: u8) {
	emit_vex_rr(buf, 1, 0, 1, 0, 0xC6, dst, a, b)
	append(buf, imm)
}

@(private) JCC_JB :: u8(0x82)
@(private) JCC_JE :: u8(0x84)
@(private) JCC_JNE :: u8(0x85)
@(private) JCC_JBE :: u8(0x86)
@(private) JCC_JA :: u8(0x87)

// A forward rel32 jump (cc: the 0F 8x opcode byte; 0: jmp), to patch with x_land.
@(private)
x_jump :: proc(buf: ^[dynamic]u8, cc: u8) -> int {
	if cc == 0 {
		emit_b(buf, 0xE9)
	} else {
		emit_bytes(buf, 0x0F, cc)
	}
	emit_i32(buf, 0)
	return len(buf^) - 4
}

@(private)
x_land :: #force_inline proc(buf: ^[dynamic]u8, at: int) {
	patch_i32(buf[:], at, i32(len(buf^) - (at + 4)))
}

// cmp byte [base + disp], imm
@(private)
x_cmp_b :: proc(buf: ^[dynamic]u8, base: u8, disp: i32, imm: u8) {
	emit_b(buf, 0x80)
	emit_modrm_mem(buf, 7, base, disp)
	emit_b(buf, imm)
}

// Skips what follows unless dl has component k: the subroutines take dl
// as the live components the image holds (x_live_held); `held` is for the
// reader. Land both patches after it.
@(private)
x_if_comp :: proc(buf: ^[dynamic]u8, k: int, held: bool) -> (skip_live, skip_held: int) {
	emit_bytes(buf, 0xF6, 0xC2, u8(1) << uint(k)) // test dl, 1 << k
	skip_live = x_jump(buf, JCC_JE)
	skip_held = -1
	return
}

// r11d = dl (the live components); dl &= the components image rdi holds.
@(private)
x_live_held :: proc(buf: ^[dynamic]u8) {
	emit_bytes(buf, 0x44, 0x0F, 0xB6, 0xDA) // movzx r11d, dl
	emit_bytes(buf, 0x22, 0x57)             // and dl, byte [rdi + held]
	emit_b(buf, u8(OFF_CI_HELD))
}

@(private)
x_end_if :: proc(buf: ^[dynamic]u8, skip_live, skip_held: int) {
	x_land(buf, skip_live)
	if skip_held >= 0 {
		x_land(buf, skip_held)
	}
}

// ymm(h)'s 16-bit half floats to f32 as f16_to_f32 converts them
// (subnormals exact, NaN quieted), with four temporaries.
@(private)
x_half :: proc(buf: ^[dynamic]u8, h, ta, tb, tc, td: u8) {
	vop_d_tab(buf, 0xDB, ta, h, OFF_TAB_H_EM) // em = h & 0x7fff
	vop_d_tab(buf, 0xDB, tb, h, OFF_TAB_H_SIGN)
	vshift_imm(buf, 6, tb, tb, 16)             // sign
	vshift_imm(buf, 6, h, ta, 13)
	vop_d_tab(buf, 0xFE, h, h, OFF_TAB_H_BIAS) // a normal half's bits
	x_dq2ps(buf, tc, ta)
	vop_ps_tab(buf, 0x59, tc, tc, OFF_TAB_H_SUB) // a subnormal one: em * 2^-24
	x_movdqa_tab(buf, td, OFF_TAB_H_SUBLIM)
	vop_d(buf, 0x66, td, td, ta)               // em < 0x400
	vblendvps(buf, h, h, tc, td)
	vshift_imm(buf, 6, tc, ta, 13)
	vop_d_tab(buf, 0xEB, tc, tc, OFF_TAB_INF)  // inf and NaN: exponent 255,
	vop_d_tab(buf, 0xDB, td, ta, OFF_TAB_H_MANT)
	vop_d_tab(buf, 0x76, td, td, OFF_TAB_ZERO)
	vop_d_tab(buf, 0xDF, td, td, OFF_TAB_H_QUIET) // NaN quieted
	vop_d(buf, 0xEB, tc, tc, td)
	vop_d_tab(buf, 0x66, td, ta, OFF_TAB_H_INFLIM) // em > 0x7bff
	vblendvps(buf, h, h, tc, td)
	vop_d(buf, 0xEB, h, h, tb)
}

// Half k of an Rgba16f texel (ymm6: halves 0 and 1, ymm5: 2 and 3) into
// ymm(k), as f16_to_f32 gives it. Called for halves 3 and 2 first, then 1
// and 0: each takes as temporaries the registers that hold nothing needed
// after it.
@(private)
x_half_comp :: proc(buf: ^[dynamic]u8, k: int) {
	temps := [4][4]u8{{4, 5, 6, 7}, {0, 4, 5, 7}, {0, 1, 4, 7}, {0, 1, 4, 7}}
	src := u8(5 if k >= 2 else 6)
	if k & 1 == 1 {
		vshift_imm(buf, 2, u8(k), src, 16)
	} else {
		vop_d_tab(buf, 0xDB, u8(k), src, OFF_TAB_U16)
	}
	t := temps[k]
	x_half(buf, u8(k), t[0], t[1], t[2], t[3])
}

// decode_sub: the texels at in-range texel indexes ymm4 of level rsi of image
// rdi, as stored values in f32 (the unsigned normalized formats' integers,
// halves decoded), into ymm0..3 for the components dl has live and the image
// holds. Clobbers rax, rcx and ymm4..7.
@(private)
emit_chan_decode_sub :: proc(buf: ^[dynamic]u8) {
	emit_bytes(buf, 0x48, 0x8B) // mov rax, [rsi + ptr]
	emit_modrm_mem(buf, GPR_RAX, GPR_RSI, OFF_LV_PTR)
	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [rdi + format]
	emit_modrm_mem(buf, GPR_RCX, GPR_RDI, OFF_CI_FORMAT)
	emit_bytes(buf, 0x85, 0xC9) // test ecx, ecx
	not_f32 := x_jump(buf, JCC_JNE)

	// F32: element e = index << shift; component k at [rax + e * 4 + 4k].
	emit_bytes(buf, 0x0F, 0xB6)
	emit_modrm_mem(buf, GPR_RCX, GPR_RDI, OFF_CI_SHIFT)
	x_bcast_gpr(buf, 5, GPR_RCX)
	vop38_d(buf, 0x47, 5, 4, 5)
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		x_gather(buf, u8(k), GPR_RAX, 5, 7, 2, i32(k * 4))
		x_end_if(buf, s0, s1)
	}
	emit_b(buf, 0xC3)

	x_land(buf, not_f32)
	emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Rgba8)) // cmp ecx, Rgba8
	not_rgba8 := x_jump(buf, JCC_JNE)
	x_gather(buf, 6, GPR_RAX, 4, 7, 2, 0)
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, false)
		vshift_imm(buf, 2, u8(k), 6, u8(8 * k))
		vop_d_tab(buf, 0xDB, u8(k), u8(k), OFF_TAB_BYTE)
		x_dq2ps(buf, u8(k), u8(k))
		x_end_if(buf, s0, s1)
	}
	emit_b(buf, 0xC3)

	x_land(buf, not_rgba8)
	emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Rgba16f))
	not_half := x_jump(buf, JCC_JNE)
	// Rgba16f: the texel's two words, ymm6 (halves 0, 1) and ymm5 (2, 3).
	x_gather(buf, 6, GPR_RAX, 4, 7, 3, 0)
	x_gather(buf, 5, GPR_RAX, 4, 7, 3, 4)
	for k := 3; k >= 0; k -= 1 {
		s0, s1 := x_if_comp(buf, k, false)
		x_half_comp(buf, k)
		x_end_if(buf, s0, s1)
	}
	emit_b(buf, 0xC3)

	// Gray8 and Gray16: the aligned dword around byte offset o = index *
	// bytes + mis, shifted right by (o & 3) * 8 and masked.
	x_land(buf, not_half)
	emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Gray16))
	is16 := x_jump(buf, JCC_JE)
	vmovaps_rr(buf, 5, 4)
	gray := x_jump(buf, 0)
	x_land(buf, is16)
	vop_d(buf, 0xFE, 5, 4, 4)
	x_land(buf, gray)
	vpbroadcastd_mem(buf, 6, GPR_RSI, OFF_LV_MIS)
	vop_d(buf, 0xFE, 5, 5, 6)
	vop_d_tab(buf, 0xDB, 6, 5, OFF_TAB_THREE_U)
	vshift_imm(buf, 6, 6, 6, 3)
	vop_d_tab(buf, 0xDB, 5, 5, OFF_TAB_NOT3)
	x_gather(buf, 0, GPR_RAX, 5, 7, 0, 0)
	vop38_d(buf, 0x45, 0, 0, 6) // vpsrlvd
	emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Gray16))
	m16 := x_jump(buf, JCC_JE)
	vop_d_tab(buf, 0xDB, 0, 0, OFF_TAB_BYTE)
	cvt := x_jump(buf, 0)
	x_land(buf, m16)
	vop_d_tab(buf, 0xDB, 0, 0, OFF_TAB_U16)
	x_land(buf, cvt)
	x_dq2ps(buf, 0, 0)
	emit_b(buf, 0xC3)
}

// Gathers the live components (`comp_mask` bits 0..3) from 1-, 2-, or 4-component
// `[rsi + yidx*4]` under `ymask` into `ymm0..3`, filling missing image channels
// with `(0, 0, 0, 1)` on masked-in lanes.
@(private)
emit_chan_gather_rgba :: proc(buf: ^[dynamic]u8, chan_base: u8, chan_disp: i32, yidx, ymask, comp_mask: u8) {
	if comp_mask == 0 {
		return
	}
	if (comp_mask & 1) != 0 {
		vmovaps_rr(buf, 7, ymask)
		vzero(buf, 0)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, 0, 7, GPR_RSI, yidx, 2, 0)
	}
	if (comp_mask & 0xE) == 0 {
		return
	}

	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [chan_base + chan_disp + OFF_CI_COMPS]
	emit_modrm_mem(buf, GPR_RCX, chan_base, chan_disp + OFF_CI_COMPS)
	emit_bytes(buf, 0x83, 0xF9, 0x02) // cmp ecx, 2
	emit_b(buf, 0x72)                 // jb comp1
	comp1_patch := len(buf^)
	emit_b(buf, 0)

	if (comp_mask & 2) != 0 {
		vmovaps_rr(buf, 7, ymask)
		vzero(buf, 1)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, 1, 7, GPR_RSI, yidx, 2, 4)
	}
	if (comp_mask & 0xC) == 0 {
		emit_b(buf, 0xEB) // jmp done
		end_rg := len(buf^)
		emit_b(buf, 0)
		buf[comp1_patch] = u8(len(buf^) - (comp1_patch + 1))
		vzero(buf, 1)
		buf[end_rg] = u8(len(buf^) - (end_rg + 1))
		return
	}

	emit_b(buf, 0x74) // je comp2 (ZF preserved from cmp ecx, 2)
	comp2_patch := len(buf^)
	emit_b(buf, 0)

	if (comp_mask & 4) != 0 {
		vmovaps_rr(buf, 7, ymask)
		vzero(buf, 2)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, 2, 7, GPR_RSI, yidx, 2, 8)
	}
	if (comp_mask & 8) != 0 {
		vmovaps_rr(buf, 7, ymask)
		vzero(buf, 3)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, 3, 7, GPR_RSI, yidx, 2, 12)
	}
	emit_b(buf, 0xEB) // jmp done
	end_patch := len(buf^)
	emit_b(buf, 0)

	buf[comp1_patch] = u8(len(buf^) - (comp1_patch + 1))
	if (comp_mask & 2) != 0 {
		vzero(buf, 1)
	}
	buf[comp2_patch] = u8(len(buf^) - (comp2_patch + 1))
	if (comp_mask & 4) != 0 {
		vzero(buf, 2)
	}
	if (comp_mask & 8) != 0 {
		vop_ps_tab(buf, 0x54, 3, ymask, OFF_TAB_ONE_F)
	}
	buf[end_patch] = u8(len(buf^) - (end_patch + 1))
}

// sample_sub's frame: 32-byte slots from rsp.
@(private) SF_U :: i32(0)
@(private) SF_V :: i32(32)
@(private) SF_L0 :: i32(64)  // each lane's level l0
@(private) SF_WL :: i32(96)  // and l0 + 1's weight of 256
@(private) SF_X0 :: i32(128)
@(private) SF_X1 :: i32(160)
@(private) SF_LAYER :: i32(192) // each lane's layer
@(private) SF_Y0 :: i32(224) // rows: y * pitch
@(private) SF_Y1 :: i32(256)
@(private) SF_TX :: i32(320) // the weights as fractions, w / 256
@(private) SF_TY :: i32(352)
@(private) SF_A :: i32(384)  // [4] the top row's blend
@(private) SF_B :: i32(512)  // [4] the bottom row's
@(private) SF_C0 :: i32(640) // [4] each lane's level l0 tap
@(private) SF_C1 :: i32(768) // [4] and l0 + 1's
@(private) SF_FRAME :: i32(896)

// One axis of the tap at level rsi: from the coordinate at [rsp + f_c],
// x = f64(c) * size - half, the texel indexes floor(x) and the next one as
// the axis's wrap addresses them, and the weight round_even(fract(x) * 256).
@(private)
emit_chan_axis :: proc(buf: ^[dynamic]u8, ax: i32, f_c, f_i0, f_i1, f_w: i32, nearest: bool, int_weight := false) {
	x_bcast_sd(buf, 3, GPR_RSI, OFF_LV_FS + ax * 8)
	for h in 0 ..< i32(2) {
		x_ps2pd_m(buf, 0, GPR_RSP, f_c + h * 16)
		x_pd(buf, 0x59, 0, 0, 3)
		if !nearest {
			x_pd_m(buf, 0x5C, 0, 0, GPR_R14, OFF_TAB_DHALF)
		}
		x_roundpd(buf, 2, 0, 0x09)            // g = floor(x)
		if !nearest {
			x_pd(buf, 0x5C, 0, 0, 2)
			x_pd_m(buf, 0x59, 0, 0, GPR_R14, OFF_TAB_D256)
			x_roundpd(buf, 0, 0, 0x08)            // to nearest even
			x_pd2dq(buf, u8(6 + h), 0)            // the weight (NaN: 0x80000000)
		}
		x_pd_m(buf, 0x5F, 2, 2, GPR_R14, OFF_TAB_DLO) // NaN: -2^31
		x_pd_m(buf, 0x5D, u8(4 + h), 2, GPR_R14, OFF_TAB_DHI)
	}
	x_cmp_b(buf, GPR_RDI, OFF_CI_WRAP + ax, u8(Channel_Wrap.Clamp))
	clamp := x_jump(buf, JCC_JE)
	// Repeat: g mod n, floored, in f64 (exact): g - n * floor(g / n). With
	// |g| < 2^31, g * (1 / n) misses floor(g / n) only where g / n is whole
	// (it can land just below), so one step down suffices.
	x_bcast_sd(buf, 1, GPR_RSI, OFF_LV_INV + ax * 8)
	x_bcast_sd(buf, 2, GPR_RSI, OFF_LV_FS + ax * 8)
	for h in u8(0) ..< 2 {
		r := 4 + h
		x_pd(buf, 0x59, 0, r, 1)
		x_roundpd(buf, 0, 0, 0x09)
		x_pd(buf, 0x59, 0, 0, 2)
		x_pd(buf, 0x5C, r, r, 0)
		x_cmppd(buf, 0, r, 2, 0x1D) // m >= n
		x_pd(buf, 0x54, 0, 0, 2)
		x_pd(buf, 0x5C, r, r, 0)
	}
	x_pd2dq(buf, 4, 4)
	x_pd2dq(buf, 5, 5)
	x_ins128(buf, 4, 4, 5)
	if !nearest {
		vop_d_tab(buf, 0xFE, 5, 4, OFF_TAB_ONE_U)
		vpbroadcastd_mem(buf, 0, GPR_RSI, OFF_LV_SIZE + ax * 4)
		vop_d(buf, 0x76, 0, 5, 0)
		vop_d(buf, 0xDF, 5, 0, 5) // the next one, 0 after n - 1
	}
	done := x_jump(buf, 0)
	x_land(buf, clamp)
	x_pd2dq(buf, 4, 4)
	x_pd2dq(buf, 5, 5)
	x_ins128(buf, 4, 4, 5)
	vzero(buf, 3)
	vpbroadcastd_mem(buf, 0, GPR_RSI, OFF_LV_MAX + ax * 4)
	if !nearest {
		vop_d_tab(buf, 0xFE, 5, 4, OFF_TAB_ONE_U)
		vop38_d(buf, 0x3D, 5, 5, 3)
		vop38_d(buf, 0x39, 5, 5, 0)
	}
	vop38_d(buf, 0x3D, 4, 4, 3)
	vop38_d(buf, 0x39, 4, 4, 0)
	x_land(buf, done)
	if ax == 1 {
		// Rows: y * pitch (and the layer's offset), for the texels'
		// indexes y * pitch + x.
		vpbroadcastd_mem(buf, 0, GPR_RSI, OFF_LV_PITCH)
		vop38_d(buf, 0x40, 4, 4, 0)
		if !nearest {
			vop38_d(buf, 0x40, 5, 5, 0)
		}
		emit_chan_layer_rows(buf, SF_LAYER, 4, 0 if nearest else 5, 0)
	}
	x_st(buf, GPR_RSP, f_i0, 4)
	if !nearest {
		x_ins128(buf, 6, 6, 7)
		vzero(buf, 3)
		vop38_d(buf, 0x3D, 6, 6, 3)
		if !int_weight {
			x_dq2ps(buf, 6, 6)
			vop_ps_tab(buf, 0x59, 6, 6, OFF_TAB_INV256) // the fraction w / 256
		}
		x_st(buf, GPR_RSP, f_i1, 5)
		x_st(buf, GPR_RSP, f_w, 6)
	}
}

// With layers (n_layers > 1): ymm(a) (and ymm(b) unless it is 0) += each
// lane's layer at [rsp + f_layer] times level r9d's layer pitch, through
// ymm(t). Clobbers eax.
@(private)
emit_chan_layer_rows :: proc(buf: ^[dynamic]u8, f_layer: i32, a, b, t: u8) {
	emit_b(buf, 0x83) // cmp dword [rdi + n_layers], 1
	emit_modrm_mem(buf, 7, GPR_RDI, OFF_CI_NLAYERS)
	emit_b(buf, 1)
	one := x_jump(buf, JCC_JBE)
	emit_bytes(buf, 0x42, 0x8B, 0x84, 0x8F) // mov eax, [rdi + r9 * 4 + lpitch]
	emit_i32(buf, OFF_CI_LPITCH)
	x_bcast_gpr(buf, t, GPR_RAX)
	x_d38_m(buf, 0x40, t, t, GPR_RSP, f_layer) // vpmulld
	vop_d(buf, 0xFE, a, a, t)
	if b != 0 {
		vop_d(buf, 0xFE, b, b, t)
	}
	x_land(buf, one)
}

// Decodes the texel at ([rsp + f_x], [rsp + f_y]) of level rsi into ymm0..3.
@(private)
emit_chan_texel :: proc(buf: ^[dynamic]u8, f_x, f_y: i32, decode_calls: ^[dynamic]int) {
	x_ld(buf, 4, GPR_RSP, f_y)
	x_d_m(buf, 0xFE, 4, 4, GPR_RSP, f_x)
	x_call(buf, decode_calls)
}

// For each component the image holds and dl has live: [rsp + f + 32k] =
// [rsp + f + 32k] + (ymm(k) - [rsp + f + 32k]) * [rsp + f_t], into ymm(k)
// as well when `keep`.
@(private)
emit_chan_lerp :: proc(buf: ^[dynamic]u8, f, f_t: i32, keep: bool) {
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		x_ld(buf, 4, GPR_RSP, f + i32(k) * 32)
		vop_ps(buf, 0x5C, 5, u8(k), 4)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x59, 5, 5, GPR_RSP, f_t)
		vop_ps(buf, 0x58, 4, 4, 5)
		if keep {
			vmovaps_rr(buf, u8(k), 4)
		} else {
			x_st(buf, GPR_RSP, f + i32(k) * 32, 4)
		}
		x_end_if(buf, s0, s1)
	}
}

// Stores ymm0..3's held, live components at [rsp + f + 32k].
@(private)
emit_chan_store4 :: proc(buf: ^[dynamic]u8, f: i32) {
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		x_st(buf, GPR_RSP, f + i32(k) * 32, u8(k))
		x_end_if(buf, s0, s1)
	}
}

// ORs ymm1's eight dwords into r8d (clobbers ymm0).
@(private)
x_hor_or_r8 :: proc(buf: ^[dynamic]u8) {
	emit_vex_rr(buf, 3, 1, 1, 0, 0x39, 1, 0, 0) // vextracti128 xmm0, ymm1, 1
	append(buf, 1)
	emit_vex_rr(buf, 1, 1, 0, 0, 0xEB, 1, 1, 0) // vpor xmm1, xmm1, xmm0
	emit_vex_rr(buf, 1, 1, 0, 0, 0x70, 0, 0, 1) // vpshufd xmm0, xmm1, 0x4e
	append(buf, 0x4E)
	emit_vex_rr(buf, 1, 1, 0, 0, 0xEB, 1, 1, 0)
	emit_vex_rr(buf, 1, 1, 0, 0, 0x70, 0, 0, 1) // vpshufd xmm0, xmm1, 0xb1
	append(buf, 0xB1)
	emit_vex_rr(buf, 1, 1, 0, 0, 0xEB, 1, 1, 0)
	emit_vex_rr(buf, 1, 1, 0, 0, 0x7E, 1, 0, 8) // vmovd r8d, xmm1
}

// The level loop's head: r9d = the lowest level in r8d, cleared from it;
// rsi = that level's record.
@(private)
x_next_level :: proc(buf: ^[dynamic]u8) {
	emit_bytes(buf, 0x45, 0x0F, 0xBC, 0xC8) // bsf r9d, r8d
	emit_bytes(buf, 0x41, 0x8D, 0x40, 0xFF) // lea eax, [r8 - 1]
	emit_bytes(buf, 0x41, 0x21, 0xC0)       // and r8d, eax
	emit_bytes(buf, 0x44, 0x89, 0xCE)       // mov esi, r9d
	emit_bytes(buf, 0x48, 0xC1, 0xE6, 0x06) // shl rsi, 6
	emit_bytes(buf, 0x48, 0x8D, 0xB4, 0x37) // lea rsi, [rdi + rsi + levels]
	emit_i32(buf, OFF_CI_LEVELS)
}

// Nearest, both axes clamped, on the level record at [base + disp]: ymm0 =
// u, ymm1 = v in; their texel indexes clamp(floor(f64(c) * n), 0, n - 1)
// out. Clobbers ymm2..4 and ymm7.
@(private)
emit_chan_nearest_xy :: proc(buf: ^[dynamic]u8, base: u8, disp: i32) {
	vzero(buf, 7)
	for ax in i32(0) ..< 2 {
		c := u8(ax)
		x_bcast_sd(buf, 4, base, disp + OFF_LV_FS + ax * 8)
		x_hi128(buf, 3, c)
		x_ps2pd(buf, 2, c)
		x_ps2pd(buf, 3, 3)
		for r in u8(2) ..< 4 {
			x_pd(buf, 0x59, r, r, 4)
			x_roundpd(buf, r, r, 0x09)
			x_pd_m(buf, 0x5F, r, r, GPR_R14, OFF_TAB_DLO)
			x_pd_m(buf, 0x5D, r, r, GPR_R14, OFF_TAB_DHI)
			x_pd2dq(buf, r, r)
		}
		x_ins128(buf, c, 2, 3)
		vpbroadcastd_mem(buf, 4, base, disp + OFF_LV_MAX + ax * 4)
		vop38_d(buf, 0x3D, c, c, 7)
		vop38_d(buf, 0x39, c, c, 4)
	}
}

// emit_chan_nearest_xy with FMA, for an image of fewer than 2^24 texels a
// side (n exact in f32), header at [base + disp]: k = floor(c * n) and r =
// fma(c, n, -k), whose sign is the exact product's against k, so floor(c *
// n) = k - (r < 0); the same indexes in f32.
@(private)
emit_chan_nearest_xy_fma :: proc(buf: ^[dynamic]u8, base: u8, disp: i32) {
	vzero(buf, 7)
	for ax in i32(0) ..< 2 {
		c := u8(ax)
		vbroadcastss_mem(buf, 4, base, disp + OFF_CI_FS32 + ax * 4)
		vop_ps(buf, 0x59, 2, c, 4)
		vroundps(buf, 2, 2, 0x09)                     // k = floor(hi)
		emit_vex_rr(buf, 2, 1, 1, 0, 0xAA, c, 4, 2)   // vfmsub213ps: c = c * n - k
		vcmpps(buf, c, c, 7, 0x11)                    // < 0: the product is below k
		vop_ps_tab(buf, 0x54, c, c, OFF_TAB_NEG1_F)
		vop_ps(buf, 0x58, 2, 2, c)                    // k - 1 there (NaN stays)
		vop_ps(buf, 0x5F, 2, 2, 7)                    // max(k, 0): NaN as 0
		vbroadcastss_mem(buf, 4, base, disp + OFF_CI_FMAX32 + ax * 4)
		vop_ps(buf, 0x5D, 2, 2, 4)                    // min(k, n - 1)
		x_ps2dq_t(buf, c, 2)
	}
}

// One clamped Linear axis for bilinear_sub. Clamped to the edge, a
// coordinate outside [0, 1] (or NaN) reads the edge texel twice, as one at
// 0 or 1 does, whatever the weight: so c = clamp(c, 0, 1) (NaN as 0), then T
// = f64(c) * 256n - 128 (below 2^31 for n < 2^22), the index floor(T) >> 8
// (from -1 to n - 1) and the weight round_even(T) - 256 * index.
//
// With `fma` (Chan_Img.lin_f32 0: n at most 32768, so N = 512n is exact
// in f32 and so is every whole number up to it) the same in f32, eight
// lanes at once, from 2x = 2(T + 128) = c * N: q = c * N rounded and e =
// fma(c, N, -q) its exact error (TwoProduct), so q + e = 2x exactly, and
// |e| is at most half q's ulp, below 1. floor(2x) is floor(q), less 1 where
// q is whole and e < 0, so A = floor(2x) - 256 = floor(2T), the index is A
// >> 9 and round_even(T) is (A + 1) >> 1, but on a tie (q whole, e = 0, A
// odd), where it is that less its low bit. The FMA only forms the exact
// error term: the indexes and weights are the f64 path's bit for bit
// (checked over all 2^32 f32 c for 19 sizes from 1 to 32768), so output
// stays identical across platforms, FMA or not.
@(private)
emit_chan_axis_clamp_lin :: proc(buf: ^[dynamic]u8, ax: i32, f_c, f_i0, f_i1, f_t: i32, fma: bool, from_level := false, int_weight := false) {
	x_ld(buf, 0, GPR_RSP, f_c)
	vzero(buf, 1)
	vop_ps(buf, 0x5F, 0, 0, 1)
	vop_ps_tab(buf, 0x5D, 0, 0, OFF_TAB_ONE_F)
	if fma {
		if from_level {
			vpbroadcastd_mem(buf, 3, GPR_RSI, OFF_LV_SIZE + ax * 4)
			vshift_imm(buf, 6, 3, 3, 9) // 512 * size (exact in f32 for size <= 32768)
			x_dq2ps(buf, 3, 3)
		} else {
			vbroadcastss_mem(buf, 3, GPR_RDI, OFF_CI_FN512 + ax * 4)
		}
		vop_ps(buf, 0x59, 1, 0, 3)                    // q = c * N
		emit_vex_rr(buf, 2, 1, 1, 0, 0xAA, 0, 3, 1)   // vfmsub213ps: e = N * c - q
		vroundps(buf, 2, 1, 0x09)                     // floor(q)
		vcmpps(buf, 1, 1, 2, 0x00)                    // q whole
		vzero(buf, 3)
		vcmpps(buf, 4, 0, 3, 0x11)                    // e < 0
		vcmpps(buf, 5, 0, 3, 0x00)                    // e = 0
		vop_ps(buf, 0x54, 4, 4, 1)
		vop_ps(buf, 0x54, 5, 5, 1)                    // whole, e = 0: a tie where A is odd
		vop_ps_tab(buf, 0x54, 4, 4, OFF_TAB_NEG1_F)
		vop_ps(buf, 0x58, 2, 2, 4)                    // floor(2x)
		x_ps2dq_t(buf, 2, 2)
		vop_d_tab(buf, 0xFA, 2, 2, OFF_TAB_I256)      // A = floor(2T)
		vshift_imm(buf, 4, 4, 2, 9)                   // index = A >> 9
		vop_d_tab(buf, 0xFE, 6, 2, OFF_TAB_ONE_U)
		vshift_imm(buf, 4, 6, 6, 1)                   // (A + 1) >> 1
		vop_d(buf, 0xDB, 2, 2, 6)
		vop_d(buf, 0xDB, 2, 2, 5)
		vop_d_tab(buf, 0xDB, 2, 2, OFF_TAB_ONE_U)
		vop_d(buf, 0xFA, 6, 6, 2)                     // round_even(T)
	} else {
		x_bcast_sd(buf, 3, GPR_RSI, OFF_LV_FS + ax * 8)
		x_pd_m(buf, 0x59, 3, 3, GPR_R14, OFF_TAB_D256)
		x_hi128(buf, 7, 0)
		for h in u8(0) ..< 2 {
			x_ps2pd(buf, 1, 7 if h == 1 else 0)
			x_pd(buf, 0x59, 1, 1, 3)
			x_pd_m(buf, 0x5C, 1, 1, GPR_R14, OFF_TAB_D128)
			x_roundpd(buf, 2, 1, 0x09)
			x_roundpd(buf, 1, 1, 0x08)
			x_pd2dq(buf, 4 + h, 2)
			x_pd2dq(buf, 2, 1)
			if h == 0 {
				vmovaps_rr(buf, 6, 2)
			} else {
				x_ins128(buf, 6, 6, 2)
			}
		}
		x_ins128(buf, 4, 4, 5)
		vshift_imm(buf, 4, 4, 4, 8)  // index = floor(T) >> 8
	}
	vshift_imm(buf, 6, 5, 4, 8)
	vop_d(buf, 0xFA, 6, 6, 5)    // weight
	if !int_weight {
		x_dq2ps(buf, 6, 6)
		vop_ps_tab(buf, 0x59, 6, 6, OFF_TAB_INV256)
	}
	x_st(buf, GPR_RSP, f_t, 6)
	vzero(buf, 3)
	vpbroadcastd_mem(buf, 0, GPR_RSI, OFF_LV_MAX + ax * 4)
	vop_d_tab(buf, 0xFE, 5, 4, OFF_TAB_ONE_U)
	vop38_d(buf, 0x3D, 4, 4, 3)
	vop38_d(buf, 0x39, 5, 5, 0)
	if ax == 1 {
		vpbroadcastd_mem(buf, 0, GPR_RSI, OFF_LV_PITCH)
		vop38_d(buf, 0x40, 4, 4, 0)
		vop38_d(buf, 0x40, 5, 5, 0)
		if from_level {
			emit_chan_layer_rows(buf, SF_LAYER, 4, 5, 0)
		}
	}
	x_st(buf, GPR_RSP, f_i0, 4)
	x_st(buf, GPR_RSP, f_i1, 5)
}

// The live components the image lacks: 0, 1 for alpha (r11d = the live
// components, dl those the image holds).
@(private)
emit_chan_lacking :: proc(buf: ^[dynamic]u8) {
	emit_bytes(buf, 0x41, 0x30, 0xD3) // xor r11b, dl: the live ones it lacks
	none := x_jump(buf, JCC_JE)
	for k in 1 ..< 4 {
		emit_bytes(buf, 0x41, 0xF6, 0xC3, u8(1) << uint(k)) // test r11b, 1 << k
		dead := x_jump(buf, JCC_JE)
		if k == 3 {
			x_movdqa_tab(buf, 3, OFF_TAB_ONE_F)
		} else {
			vzero(buf, u8(k))
		}
		x_land(buf, dead)
	}
	x_land(buf, none)
}

// bilinear_sub's frame.
@(private) BF_U :: i32(0)
@(private) BF_V :: i32(32)
@(private) BF_X0 :: i32(64)
@(private) BF_X1 :: i32(96)
@(private) BF_Y0 :: i32(128) // rows: y * pitch
@(private) BF_Y1 :: i32(160)
@(private) BF_TX :: i32(192)
@(private) BF_TY :: i32(224)
@(private) BF_E :: i32(256) // [4] the corners' element indexes
@(private) BF_FRAME :: i32(384)

// bilinear_sub: sample_sub's Linear for F32 with one level and both axes
// clamped (Chan_Img.lin_route 0), the blends in registers.
//   in:  ymm0 = u, ymm1 = v, rdi = ^Chan_Img, dl = live components
//   out: ymm0..3; preserves rbx, rbp, r12..r14 and ymm8..15.
@(private)
emit_chan_bilinear_sub :: proc(buf: ^[dynamic]u8, fma: bool) {
	x_live_held(buf)
	emit_bytes(buf, 0x48, 0x81, 0xEC) // sub rsp, frame
	emit_i32(buf, BF_FRAME)
	x_st(buf, GPR_RSP, BF_U, 0)
	x_st(buf, GPR_RSP, BF_V, 1)
	emit_b(buf, 0x48) // lea rsi, [rdi + levels]
	emit_b(buf, 0x8D)
	emit_modrm_mem(buf, GPR_RSI, GPR_RDI, OFF_CI_LEVELS)
	if fma {
		x_cmp_b(buf, GPR_RDI, OFF_CI_LIN_F32, 0)
		wide := x_jump(buf, JCC_JNE)
		emit_chan_axis_clamp_lin(buf, 0, BF_U, BF_X0, BF_X1, BF_TX, true)
		emit_chan_axis_clamp_lin(buf, 1, BF_V, BF_Y0, BF_Y1, BF_TY, true)
		axes := x_jump(buf, 0)
		x_land(buf, wide)
		emit_chan_axis_clamp_lin(buf, 0, BF_U, BF_X0, BF_X1, BF_TX, false)
		emit_chan_axis_clamp_lin(buf, 1, BF_V, BF_Y0, BF_Y1, BF_TY, false)
		x_land(buf, axes)
	} else {
		emit_chan_axis_clamp_lin(buf, 0, BF_U, BF_X0, BF_X1, BF_TX, false)
		emit_chan_axis_clamp_lin(buf, 1, BF_V, BF_Y0, BF_Y1, BF_TY, false)
	}
	corners := [4][2]i32{{BF_X0, BF_Y0}, {BF_X1, BF_Y0}, {BF_X0, BF_Y1}, {BF_X1, BF_Y1}}
	for c, i in corners {
		x_ld(buf, 4, GPR_RSP, c[1])
		x_d_m(buf, 0xFE, 4, 4, GPR_RSP, c[0])
		x_st(buf, GPR_RSP, BF_E + i32(i) * 32, 4)
	}
	emit_bytes(buf, 0x48, 0x8B) // mov rax, [rsi + ptr]
	emit_modrm_mem(buf, GPR_RAX, GPR_RSI, OFF_LV_PTR)
	emit_bytes(buf, 0x84, 0xD2) // test dl, dl
	dead := x_jump(buf, JCC_JE)
	x_cmp_b(buf, GPR_RDI, OFF_CI_COMPS, 4)
	not_v4 := x_jump(buf, JCC_JNE)
	// F32x4: load whole 128-bit texels for paired lanes (i, i + 4), blend, transpose.
	x_ld(buf, 6, GPR_RSP, BF_TX)
	x_ld(buf, 7, GPR_RSP, BF_TY)
	for i in 0 ..< 4 {
		for ci in 0 ..< 4 {
			disp_lo := BF_E + i32(ci) * 32 + i32(i) * 4
			disp_hi := disp_lo + 16
			emit_b(buf, 0x8B) // mov ecx, [rsp + disp_lo]
			emit_modrm_mem(buf, GPR_RCX, GPR_RSP, disp_lo)
			emit_bytes(buf, 0x44, 0x8B) // mov r8d, [rsp + disp_hi]
			emit_modrm_mem(buf, 8, GPR_RSP, disp_hi)
			emit_bytes(buf, 0x48, 0xC1, 0xE1, 0x04) // shl rcx, 4
			emit_bytes(buf, 0x49, 0xC1, 0xE0, 0x04) // shl r8, 4
			x_vmovups_xmm_sib(buf, u8(ci), GPR_RAX, GPR_RCX)
			x_vinsertf128_sib(buf, u8(ci), u8(ci), GPR_RAX, 8)
		}
		x_shufps(buf, 4, 6, 6, u8(i) * 0x55)
		x_shufps(buf, 5, 7, 7, u8(i) * 0x55)
		vop_ps(buf, 0x5C, 1, 1, 0)
		vop_ps(buf, 0x59, 1, 1, 4)
		vop_ps(buf, 0x58, 0, 0, 1) // top
		vop_ps(buf, 0x5C, 3, 3, 2)
		vop_ps(buf, 0x59, 3, 3, 4)
		vop_ps(buf, 0x58, 2, 2, 3) // bottom
		vop_ps(buf, 0x5C, 2, 2, 0)
		vop_ps(buf, 0x59, 2, 2, 5)
		vop_ps(buf, 0x58, 0, 0, 2) // top + (bottom - top) * ty
		if i < 3 {
			x_st(buf, GPR_RSP, BF_U + i32(i) * 32, 0)
		} else {
			vmovaps_rr(buf, 3, 0)
		}
	}
	x_ld(buf, 0, GPR_RSP, BF_U)
	x_ld(buf, 1, GPR_RSP, BF_U + 32)
	x_ld(buf, 2, GPR_RSP, BF_U + 64)
	vop_ps(buf, 0x14, 4, 0, 1)
	vop_ps(buf, 0x15, 5, 0, 1)
	vop_ps(buf, 0x14, 6, 2, 3)
	vop_ps(buf, 0x15, 7, 2, 3)
	x_shufps(buf, 0, 4, 6, 0x44)
	x_shufps(buf, 1, 4, 6, 0xEE)
	x_shufps(buf, 2, 5, 7, 0x44)
	x_shufps(buf, 3, 5, 7, 0xEE)
	v4_done := x_jump(buf, 0)

	x_land(buf, not_v4)
	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [rdi + shift]
	emit_modrm_mem(buf, GPR_RCX, GPR_RDI, OFF_CI_SHIFT)
	x_bcast_gpr(buf, 6, GPR_RCX)
	for i in 0 ..< 4 {
		x_ld(buf, 4, GPR_RSP, BF_E + i32(i) * 32)
		vop38_d(buf, 0x47, 4, 4, 6)
		x_st(buf, GPR_RSP, BF_E + i32(i) * 32, 4)
	}
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		yk := u8(k)
		d := i32(k) * 4
		x_ld(buf, 6, GPR_RSP, BF_E)
		x_gather(buf, yk, GPR_RAX, 6, 7, 2, d)
		x_ld(buf, 6, GPR_RSP, BF_E + 32)
		x_gather(buf, 4, GPR_RAX, 6, 7, 2, d)
		vop_ps(buf, 0x5C, 4, 4, yk)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x59, 4, 4, GPR_RSP, BF_TX)
		vop_ps(buf, 0x58, yk, yk, 4) // top = c00 + (c10 - c00) * tx
		x_ld(buf, 6, GPR_RSP, BF_E + 64)
		x_gather(buf, 4, GPR_RAX, 6, 7, 2, d)
		x_ld(buf, 6, GPR_RSP, BF_E + 96)
		x_gather(buf, 5, GPR_RAX, 6, 7, 2, d)
		vop_ps(buf, 0x5C, 5, 5, 4)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x59, 5, 5, GPR_RSP, BF_TX)
		vop_ps(buf, 0x58, 4, 4, 5)   // bottom
		vop_ps(buf, 0x5C, 4, 4, yk)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x59, 4, 4, GPR_RSP, BF_TY)
		vop_ps(buf, 0x58, yk, yk, 4) // top + (bottom - top) * ty
		x_end_if(buf, s0, s1)
	}
	x_land(buf, v4_done)
	x_land(buf, dead)
	emit_chan_lacking(buf)
	emit_bytes(buf, 0x48, 0x81, 0xC4) // add rsp, frame
	emit_i32(buf, BF_FRAME)
	emit_b(buf, 0xC3)
}

// sample_sub: texture(), textureLod and textureGrad of a bound channel.
//   in:  ymm0 = u, ymm1 = v, ymm2 = lod, ymm3 = the layer (with layers:
//        clamped, as i32), rdi = ^Chan_Img, dl = live components
//   out: ymm0..3; preserves rbx, rbp, r12..r14 and ymm8..15.
@(private)
emit_chan_sample_sub :: proc(buf: ^[dynamic]u8, decode_calls: ^[dynamic]int, fma: bool) {
	x_live_held(buf)
	// One level, Nearest, both axes clamped: floor(f64(c) * n) clamped, in
	// registers, the texel decoded.
	x_cmp_b(buf, GPR_RDI, OFF_CI_ROUTE, 0)
	general := x_jump(buf, JCC_JNE)
	emit_b(buf, 0x48) // lea rsi, [rdi + levels]
	emit_b(buf, 0x8D)
	emit_modrm_mem(buf, GPR_RSI, GPR_RDI, OFF_CI_LEVELS)
	if fma {
		x_cmp_b(buf, GPR_RDI, OFF_CI_NEAR_ROUTE, 1)
		wide := x_jump(buf, JCC_JNE)
		emit_chan_nearest_xy_fma(buf, GPR_RDI, 0)
		xy := x_jump(buf, 0)
		x_land(buf, wide)
		emit_chan_nearest_xy(buf, GPR_RSI, 0)
		x_land(buf, xy)
	} else {
		emit_chan_nearest_xy(buf, GPR_RSI, 0)
	}
	vpbroadcastd_mem(buf, 4, GPR_RSI, OFF_LV_PITCH)
	vop38_d(buf, 0x40, 4, 1, 4)
	vop_d(buf, 0xFE, 4, 4, 0)
	x_call(buf, decode_calls)
	short_done := x_jump(buf, 0)
	x_land(buf, general)
	emit_bytes(buf, 0x48, 0x81, 0xEC) // sub rsp, frame
	emit_i32(buf, SF_FRAME)
	x_st(buf, GPR_RSP, SF_U, 0)
	x_st(buf, GPR_RSP, SF_V, 1)
	x_st(buf, GPR_RSP, SF_LAYER, 3)

	// Levels: l0 and l0 + 1's weight w per lane (Channel_Image), the levels
	// some lane uses as bits in r8d; r10d = 0 (1 level), 1 (2 levels iter 0),
	// 2 (2 levels iter 1), or 3 (>= 3 levels across the quad).
	emit_bytes(buf, 0x45, 0x31, 0xD2) // xor r10d, r10d
	emit_bytes(buf, 0x41, 0xB8)       // mov r8d, 1
	emit_u32(buf, 1)
	emit_b(buf, 0x83) // cmp dword [rdi + n_levels], 1
	emit_modrm_mem(buf, 7, GPR_RDI, OFF_CI_NLEV)
	emit_b(buf, 1)
	single := x_jump(buf, JCC_JE)
	vbroadcastss_mem(buf, 4, GPR_RDI, OFF_CI_TOP)
	vzero(buf, 3)
	vcmpps(buf, 5, 2, 3, 0x1E) // lod > 0
	vop_ps(buf, 0x5D, 6, 2, 4) // l = min(lod, top)
	x_ps2dq_t(buf, 7, 6)
	vop_d(buf, 0xDB, 7, 7, 5)  // l0
	x_dq2ps(buf, 1, 7)
	vop_ps(buf, 0x5C, 6, 6, 1)
	vop_ps_tab(buf, 0x59, 6, 6, OFF_TAB_F256)
	vroundps(buf, 6, 6, 0x08)
	x_ps2dq_t(buf, 6, 6)
	vop_d(buf, 0xDB, 6, 6, 5)  // w (0 at the top level, where l = l0)
	x_st(buf, GPR_RSP, SF_L0, 7)
	x_st(buf, GPR_RSP, SF_WL, 6)
	x_movdqa_tab(buf, 0, OFF_TAB_ONE_U)
	vop38_d(buf, 0x47, 1, 0, 7)
	vop_d_tab(buf, 0xFE, 2, 7, OFF_TAB_ONE_U)
	vop38_d(buf, 0x47, 2, 0, 2)
	vop_d(buf, 0x76, 3, 6, 3)  // w == 0
	vop_d(buf, 0xDF, 2, 3, 2)
	vop_d(buf, 0xEB, 1, 1, 2)
	x_hor_or_r8(buf)
	emit_bytes(buf, 0x41, 0x8D, 0x40, 0xFF) // lea eax, [r8 - 1]
	emit_bytes(buf, 0x44, 0x21, 0xC0)       // and eax, r8d -> t = r8d & (r8d - 1)
	one_bit := x_jump(buf, JCC_JE)
	emit_bytes(buf, 0x41, 0xBA) // mov r10d, 1
	emit_u32(buf, 1)
	emit_bytes(buf, 0x8D, 0x48, 0xFF) // lea ecx, [rax - 1]
	emit_bytes(buf, 0x85, 0xC8)       // test eax, ecx -> 0 iff popcount(r8d) == 2
	two_bits := x_jump(buf, JCC_JE)
	emit_bytes(buf, 0x41, 0xBA) // mov r10d, 3
	emit_u32(buf, 3)
	x_land(buf, two_bits)
	x_land(buf, one_bit)
	x_land(buf, single)

	loop_top := len(buf^)
	x_next_level(buf)
	x_cmp_b(buf, GPR_RDI, OFF_CI_FILTER, u8(Channel_Filter.Nearest))
	linear := x_jump(buf, JCC_JNE)
	emit_chan_axis(buf, 0, SF_U, SF_X0, SF_X1, SF_TX, true)
	emit_chan_axis(buf, 1, SF_V, SF_Y0, SF_Y1, SF_TY, true)
	emit_chan_texel(buf, SF_X0, SF_Y0, decode_calls)
	tapped := x_jump(buf, 0)

	x_land(buf, linear)
	if fma {
		emit_b(buf, 0x66) // cmp word [rdi + wrap], 0 ({.Clamp, .Clamp})
		emit_b(buf, 0x83)
		emit_modrm_mem(buf, 7, GPR_RDI, OFF_CI_WRAP)
		emit_b(buf, 0)
		f64_ax0 := x_jump(buf, JCC_JNE)
		emit_b(buf, 0x81) // cmp dword [rsi + size.x], 32768
		emit_modrm_mem(buf, 7, GPR_RSI, OFF_LV_SIZE)
		emit_u32(buf, 32768)
		f64_ax1 := x_jump(buf, JCC_JA)
		emit_b(buf, 0x81) // cmp dword [rsi + size.y], 32768
		emit_modrm_mem(buf, 7, GPR_RSI, OFF_LV_SIZE + 4)
		emit_u32(buf, 32768)
		f64_ax2 := x_jump(buf, JCC_JA)
		emit_chan_axis_clamp_lin(buf, 0, SF_U, SF_X0, SF_X1, SF_TX, true, true, true)
		emit_chan_axis_clamp_lin(buf, 1, SF_V, SF_Y0, SF_Y1, SF_TY, true, true, true)
		axes_done := x_jump(buf, 0)
		x_land(buf, f64_ax0)
		x_land(buf, f64_ax1)
		x_land(buf, f64_ax2)
		emit_chan_axis(buf, 0, SF_U, SF_X0, SF_X1, SF_TX, false, true)
		emit_chan_axis(buf, 1, SF_V, SF_Y0, SF_Y1, SF_TY, false, true)
		x_land(buf, axes_done)
	} else {
		emit_chan_axis(buf, 0, SF_U, SF_X0, SF_X1, SF_TX, false, true)
		emit_chan_axis(buf, 1, SF_V, SF_Y0, SF_Y1, SF_TY, false, true)
	}

	x_cmp_b(buf, GPR_RDI, OFF_CI_FORMAT, u8(Channel_Format.Rgba8))
	not_rgba8 := x_jump(buf, JCC_JNE)
	// Rgba8 per-level linear tap: integer bilinear in 16/32 bits, exact to f32.
	emit_bytes(buf, 0x84, 0xD2) // test dl, dl
	rgba8_dead := x_jump(buf, JCC_JE)
	emit_bytes(buf, 0x48, 0x8B) // mov rax, [rsi + ptr]
	emit_modrm_mem(buf, GPR_RAX, GPR_RSI, OFF_LV_PTR)
	x_ld(buf, 6, GPR_RSP, SF_TX)
	vshift_imm(buf, 6, 7, 6, 16)
	vop_d(buf, 0xEB, 6, 6, 7) // wx_pair = (wx << 16) | wx
	for pair in 0 ..< 2 {
		fy := SF_Y0 if pair == 0 else SF_Y1
		x_ld(buf, 4, GPR_RSP, fy)
		x_d_m(buf, 0xFE, 4, 4, GPR_RSP, SF_X0)
		x_gather(buf, 0, GPR_RAX, 4, 7, 2, 0)
		x_ld(buf, 4, GPR_RSP, fy)
		x_d_m(buf, 0xFE, 4, 4, GPR_RSP, SF_X1)
		x_gather(buf, 1, GPR_RAX, 4, 7, 2, 0)
		if pair == 0 {
			// T_rb -> ymm2, T_ga -> ymm3
			vop_d_tab(buf, 0xDB, 4, 0, OFF_TAB_W00FF)
			vop_d_tab(buf, 0xDB, 5, 1, OFF_TAB_W00FF)
			vop_w(buf, 0xF9, 5, 5, 4)
			vop_w(buf, 0xD5, 5, 5, 6)
			vshift_w_imm(buf, 6, 4, 4, 8)
			vop_w(buf, 0xFD, 2, 4, 5)
			vshift_w_imm(buf, 2, 4, 0, 8)
			vshift_w_imm(buf, 2, 5, 1, 8)
			vop_w(buf, 0xF9, 5, 5, 4)
			vop_w(buf, 0xD5, 5, 5, 6)
			vshift_w_imm(buf, 6, 4, 4, 8)
			vop_w(buf, 0xFD, 3, 4, 5)
		} else {
			// B_rb -> ymm4, B_ga -> ymm5 (preserve ymm2, ymm3)
			vop_d_tab(buf, 0xDB, 4, 0, OFF_TAB_W00FF)
			vop_d_tab(buf, 0xDB, 5, 1, OFF_TAB_W00FF)
			vop_w(buf, 0xF9, 5, 5, 4)
			vop_w(buf, 0xD5, 5, 5, 6)
			vshift_w_imm(buf, 6, 4, 4, 8)
			vop_w(buf, 0xFD, 4, 4, 5)
			vshift_w_imm(buf, 2, 0, 0, 8)
			vshift_w_imm(buf, 2, 1, 1, 8)
			vop_w(buf, 0xF9, 1, 1, 0)
			vop_w(buf, 0xD5, 1, 1, 6)
			vshift_w_imm(buf, 6, 0, 0, 8)
			vop_w(buf, 0xFD, 5, 0, 1)
		}
	}
	x_ld(buf, 6, GPR_RSP, SF_TY)
	// Vertical blend in order R (0), B (2), G (1), A (3) so ymm2/ymm3 stay valid until both halves are read.
	Rgba8_Comp :: struct {
		dst, t_src, b_src: u8,
		hi:                bool,
	}
	for rc in ([4]Rgba8_Comp{{0, 2, 4, false}, {2, 2, 4, true}, {1, 3, 5, false}, {3, 3, 5, true}}) {
		s0, s1 := x_if_comp(buf, int(rc.dst), false)
		if rc.hi {
			vshift_imm(buf, 2, rc.dst, rc.t_src, 16)
			vshift_imm(buf, 2, 7, rc.b_src, 16)
		} else {
			vop_d_tab(buf, 0xDB, rc.dst, rc.t_src, OFF_TAB_U16)
			vop_d_tab(buf, 0xDB, 7, rc.b_src, OFF_TAB_U16)
		}
		vop_d(buf, 0xFA, 7, 7, rc.dst)
		vop38_d(buf, 0x40, 7, 7, 6)
		vshift_imm(buf, 6, rc.dst, rc.dst, 8)
		vop_d(buf, 0xFE, rc.dst, rc.dst, 7)
		x_dq2ps(buf, rc.dst, rc.dst)
		vop_ps_tab(buf, 0x59, rc.dst, rc.dst, OFF_TAB_INV65536)
		x_end_if(buf, s0, s1)
	}
	x_land(buf, rgba8_dead)
	rgba8_done := x_jump(buf, 0)

	x_land(buf, not_rgba8)
	x_cmp_b(buf, GPR_RDI, OFF_CI_FORMAT, u8(Channel_Format.Gray8))
	not_gray8 := x_jump(buf, JCC_JNE)
	// Gray8 per-level linear tap: four byte gathers + integer bilinear.
	emit_bytes(buf, 0xF6, 0xC2, 0x01) // test dl, 1
	g8_dead := x_jump(buf, JCC_JE)
	emit_bytes(buf, 0x48, 0x8B) // mov rax, [rsi + ptr]
	emit_modrm_mem(buf, GPR_RAX, GPR_RSI, OFF_LV_PTR)
	g8_corners := [4][2]i32{{SF_X0, SF_Y0}, {SF_X1, SF_Y0}, {SF_X0, SF_Y1}, {SF_X1, SF_Y1}}
	for gc, ci in g8_corners {
		x_ld(buf, 4, GPR_RSP, gc[1])
		x_d_m(buf, 0xFE, 4, 4, GPR_RSP, gc[0])
		vpbroadcastd_mem(buf, 5, GPR_RSI, OFF_LV_MIS)
		vop_d(buf, 0xFE, 4, 4, 5)
		vop_d_tab(buf, 0xDB, 5, 4, OFF_TAB_THREE_U)
		vshift_imm(buf, 6, 5, 5, 3)
		vop_d_tab(buf, 0xDB, 4, 4, OFF_TAB_NOT3)
		x_gather(buf, u8(ci), GPR_RAX, 4, 7, 0, 0)
		vop38_d(buf, 0x45, u8(ci), u8(ci), 5)
		vop_d_tab(buf, 0xDB, u8(ci), u8(ci), OFF_TAB_BYTE)
	}
	x_ld(buf, 6, GPR_RSP, SF_TX)
	x_ld(buf, 7, GPR_RSP, SF_TY)
	vop_d(buf, 0xFA, 1, 1, 0)
	vop38_d(buf, 0x40, 1, 1, 6)
	vshift_imm(buf, 6, 0, 0, 8)
	vop_d(buf, 0xFE, 0, 0, 1) // T
	vop_d(buf, 0xFA, 3, 3, 2)
	vop38_d(buf, 0x40, 3, 3, 6)
	vshift_imm(buf, 6, 2, 2, 8)
	vop_d(buf, 0xFE, 2, 2, 3) // B
	vop_d(buf, 0xFA, 2, 2, 0)
	vop38_d(buf, 0x40, 2, 2, 7)
	vshift_imm(buf, 6, 0, 0, 8)
	vop_d(buf, 0xFE, 0, 0, 2) // R
	x_dq2ps(buf, 0, 0)
	vop_ps_tab(buf, 0x59, 0, 0, OFF_TAB_INV65536)
	x_land(buf, g8_dead)
	g8_done := x_jump(buf, 0)

	// General linear fallback (F32, Rgba16f, Gray16): weights i32 -> f32 / 256.
	x_land(buf, not_gray8)
	for fw in ([2]i32{SF_TX, SF_TY}) {
		x_ld(buf, 6, GPR_RSP, fw)
		x_dq2ps(buf, 6, 6)
		vop_ps_tab(buf, 0x59, 6, 6, OFF_TAB_INV256)
		x_st(buf, GPR_RSP, fw, 6)
	}
	// The rows' blends across x, then theirs across y.
	emit_chan_texel(buf, SF_X0, SF_Y0, decode_calls)
	emit_chan_store4(buf, SF_A)
	emit_chan_texel(buf, SF_X1, SF_Y0, decode_calls)
	emit_chan_lerp(buf, SF_A, SF_TX, false)
	emit_chan_texel(buf, SF_X0, SF_Y1, decode_calls)
	emit_chan_store4(buf, SF_B)
	emit_chan_texel(buf, SF_X1, SF_Y1, decode_calls)
	emit_chan_lerp(buf, SF_B, SF_TX, true)
	// ymm(k) = A + (B - A) * ty: B is in ymm(k).
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		x_ld(buf, 4, GPR_RSP, SF_A + i32(k) * 32)
		vop_ps(buf, 0x5C, 5, u8(k), 4)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x59, 5, 5, GPR_RSP, SF_TY)
		vop_ps(buf, 0x58, u8(k), 4, 5)
		x_end_if(buf, s0, s1)
	}
	x_land(buf, g8_done)
	x_land(buf, rgba8_done)
	x_land(buf, tapped)

	emit_bytes(buf, 0x45, 0x85, 0xD2) // test r10d, r10d
	one_level := x_jump(buf, JCC_JE)
	emit_bytes(buf, 0x41, 0x83, 0xFA, 0x02) // cmp r10d, 2
	two_iter0 := x_jump(buf, JCC_JB)
	two_iter1 := x_jump(buf, JCC_JE)
	// General (>= 3 levels across lanes): this tap is level l0's for the
	// lanes at l0, l0 + 1's for the lanes blending toward it.
	x_bcast_gpr(buf, 4, 9)
	x_ld(buf, 5, GPR_RSP, SF_L0)
	vop_d(buf, 0x76, 6, 5, 4)
	vop_d_tab(buf, 0xFE, 5, 5, OFF_TAB_ONE_U)
	vop_d(buf, 0x76, 7, 5, 4)
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		x_ld(buf, 4, GPR_RSP, SF_C0 + i32(k) * 32)
		vblendvps(buf, 4, 4, u8(k), 6)
		x_st(buf, GPR_RSP, SF_C0 + i32(k) * 32, 4)
		x_ld(buf, 4, GPR_RSP, SF_C1 + i32(k) * 32)
		vblendvps(buf, 4, 4, u8(k), 7)
		x_st(buf, GPR_RSP, SF_C1 + i32(k) * 32, 4)
		x_end_if(buf, s0, s1)
	}
	emit_bytes(buf, 0x45, 0x85, 0xC0) // test r8d, r8d
	emit_bytes(buf, 0x0F, 0x85)       // jnz loop_top
	emit_i32(buf, i32(loop_top - (len(buf^) + 4)))
	// c0 + (c1 - c0) * w / 256 where w > 0, else c0.
	x_ld(buf, 6, GPR_RSP, SF_WL)
	x_dq2ps(buf, 7, 6)
	vop_ps_tab(buf, 0x59, 7, 7, OFF_TAB_INV256)
	vzero(buf, 5)
	vop_d(buf, 0x66, 6, 6, 5) // w > 0
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		x_ld(buf, u8(k), GPR_RSP, SF_C0 + i32(k) * 32)
		x_ld(buf, 4, GPR_RSP, SF_C1 + i32(k) * 32)
		vop_ps(buf, 0x5C, 4, 4, u8(k))
		vop_ps(buf, 0x59, 4, 4, 7)
		vop_ps(buf, 0x58, 4, u8(k), 4)
		vblendvps(buf, u8(k), u8(k), 4, 6)
		x_end_if(buf, s0, s1)
	}
	gen_done := x_jump(buf, 0)

	// Two levels (L_a < L_b): keep tap(L_a) in SF_C0 and blend directly
	// against tap(L_b) in ymm0..3 without touching SF_C1's array.
	x_land(buf, two_iter0)
	emit_chan_store4(buf, SF_C0)
	emit_bytes(buf, 0x41, 0xBA) // mov r10d, 2
	emit_u32(buf, 2)
	emit_b(buf, 0xE9) // jmp loop_top
	emit_i32(buf, i32(loop_top - (len(buf^) + 4)))

	x_land(buf, two_iter1)
	x_bcast_gpr(buf, 4, 9)
	x_ld(buf, 5, GPR_RSP, SF_L0)
	vop_d(buf, 0x76, 5, 5, 4) // l0 == L_b
	x_st(buf, GPR_RSP, SF_C1, 5)
	x_ld(buf, 6, GPR_RSP, SF_WL)
	x_dq2ps(buf, 7, 6)
	vop_ps_tab(buf, 0x59, 7, 7, OFF_TAB_INV256)
	vzero(buf, 5)
	vop_d(buf, 0x66, 6, 6, 5) // wl > 0
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		x_ld(buf, 4, GPR_RSP, SF_C0 + i32(k) * 32)
		vop_ps(buf, 0x5C, 5, u8(k), 4)
		vop_ps(buf, 0x59, 5, 5, 7)
		vop_ps(buf, 0x58, 5, 4, 5)
		vblendvps(buf, 4, 4, 5, 6)
		x_ld(buf, 5, GPR_RSP, SF_C1)
		vblendvps(buf, u8(k), 4, u8(k), 5)
		x_end_if(buf, s0, s1)
	}

	x_land(buf, gen_done)
	x_land(buf, one_level)
	emit_bytes(buf, 0x48, 0x81, 0xC4) // add rsp, frame
	emit_i32(buf, SF_FRAME)
	x_land(buf, short_done)

	// The unsigned normalized formats' stored values / 255 or 65535; the
	// components the image lacks 0 (1 for alpha).
	emit_b(buf, 0x83) // cmp dword [rdi + unorm], 0
	emit_modrm_mem(buf, 7, GPR_RDI, OFF_CI_UNORM)
	emit_b(buf, 0)
	float := x_jump(buf, JCC_JE)
	vbroadcastss_mem(buf, 5, GPR_RDI, OFF_CI_UNORM)
	for k in 0 ..< 4 {
		s0, s1 := x_if_comp(buf, k, true)
		vop_ps(buf, 0x5E, u8(k), u8(k), 5)
		x_end_if(buf, s0, s1)
	}
	x_land(buf, float)
	emit_chan_lacking(buf)
	emit_b(buf, 0xC3)
}

// quad_levels8(u, v, w, h) added to slots[bias_slot], into ymm2 (without FMA,
// matching vm_run.odin bit for bit). Preserves ymm0, ymm1, rbx, rbp, rdi, r12..r14.
@(private)
emit_chan_quad_levels :: proc(buf: ^[dynamic]u8, bias_slot: i32) {
	shuf :: #force_inline proc(buf: ^[dynamic]u8, dst, src, imm: u8) {
		emit_vex_rr(buf, 1, 0, 1, 0, 0xC6, dst, src, src)
		append(buf, imm)
	}
	vpbroadcastd_mem(buf, 6, GPR_RDI, OFF_CI_LEVELS + OFF_LV_SIZE)
	x_dq2ps(buf, 6, 6) // w
	vpbroadcastd_mem(buf, 7, GPR_RDI, OFF_CI_LEVELS + OFF_LV_SIZE + 4)
	x_dq2ps(buf, 7, 7) // h
	shuf(buf, 2, 0, 0x00)
	shuf(buf, 3, 0, 0x55)
	vop_ps(buf, 0x5C, 3, 3, 2)
	vop_ps(buf, 0x59, 3, 3, 6) // dxx
	shuf(buf, 4, 0, 0xAA)
	vop_ps(buf, 0x5C, 4, 4, 2)
	vop_ps(buf, 0x59, 4, 4, 6) // dyx
	shuf(buf, 2, 1, 0x00)
	shuf(buf, 5, 1, 0x55)
	vop_ps(buf, 0x5C, 5, 5, 2)
	vop_ps(buf, 0x59, 5, 5, 7) // dxy
	shuf(buf, 6, 1, 0xAA)
	vop_ps(buf, 0x5C, 6, 6, 2)
	vop_ps(buf, 0x59, 6, 6, 7) // dyy
	vop_ps(buf, 0x59, 3, 3, 3)
	vop_ps(buf, 0x59, 5, 5, 5)
	vop_ps(buf, 0x58, 3, 3, 5) // x2
	vop_ps(buf, 0x59, 4, 4, 4)
	vop_ps(buf, 0x59, 6, 6, 6)
	vop_ps(buf, 0x58, 4, 4, 6) // y2
	// m (maxnum): y2 where x2 is NaN or y2 > x2, into ymm2.
	vcmpps(buf, 5, 3, 3, 0x03)
	vop_ps(buf, 0x5F, 2, 4, 3)
	vblendvps(buf, 2, 2, 4, 5)
	// quad_level8(m): 0.5 * log2(m).
	vcmpps_tab(buf, 3, 2, OFF_TAB_QL_FLT_MIN, 0x11)
	vop_ps_tab(buf, 0x59, 4, 2, OFF_TAB_QL_TWO24)
	vblendvps(buf, 4, 2, 4, 3) // ms
	vop_d_tab(buf, 0xFA, 5, 4, OFF_TAB_QL_SQRT_HALF)
	vshift_imm(buf, 4, 5, 5, 23) // k
	vshift_imm(buf, 6, 6, 5, 23)
	vop_d(buf, 0xFA, 4, 4, 6)
	vop_ps_tab(buf, 0x5C, 4, 4, OFF_TAB_ONE_F) // f
	x_dq2ps(buf, 5, 5)
	vop_ps_tab(buf, 0x54, 3, 3, OFF_TAB_QL_24)
	vop_ps(buf, 0x5C, 3, 5, 3)  // e
	vop_ps(buf, 0x59, 5, 4, 4)  // f2
	for c := i32(2); c >= 0; c -= 1 {
		dst := u8(6 if c == 2 else 7)
		vop_ps_tab(buf, 0x59, dst, 4, OFF_TAB_QL_C + (2 * c + 1) * 32)
		vop_ps_tab(buf, 0x58, dst, dst, OFF_TAB_QL_C + (2 * c) * 32)
		if c < 2 {
			vop_ps(buf, 0x59, 6, 5, 6)
			vop_ps(buf, 0x58, 6, 7, 6)
		}
	}
	vop_ps(buf, 0x59, 6, 4, 6)
	vop_ps(buf, 0x58, 6, 3, 6)
	vop_ps_tab(buf, 0x59, 6, 6, OFF_TAB_QL_HALF) // l
	vzero(buf, 7)
	vcmpps(buf, 5, 2, 7, 0x00)
	x_movdqa_tab(buf, 4, OFF_TAB_QL_NEG_INF)
	vblendvps(buf, 6, 6, 4, 5) // m == 0: -inf
	vcmpps_tab(buf, 5, 2, OFF_TAB_INF, 0x00)
	vblendvps(buf, 6, 6, 2, 5) // m == +inf: m
	vcmpps(buf, 5, 2, 7, 0x1D)
	x_movdqa_tab(buf, 4, OFF_TAB_CANON)
	vblendvps(buf, 6, 4, 6, 5) // else NaN
	vmovaps_load(buf, 2, bias_slot)
	vop_ps(buf, 0x58, 2, 2, 6)
}

// fetch_sub's frame.
@(private) FF_X :: i32(0)
@(private) FF_Y :: i32(32)
@(private) FF_LOD :: i32(64)
@(private) FF_R :: i32(96) // [4] results
@(private) FF_M :: i32(224) // this level's lanes in range
@(private) FF_LAYER :: i32(256)
@(private) FF_FRAME :: i32(288)

// fetch_sub: texelFetch of a bound channel, each lane's texel at its level,
// (0, 0, 0, 0) out of range.
//   in:  ymm0 = x, ymm1 = y, ymm2 = level (a layer outside the image's as
//        a level outside its levels), ymm3 = layer (read with layers
//        only), rdi = ^Chan_Img, dl = live components, dh bit 0 = flip y
//   out: ymm0..3; preserves rbx, rbp, r12..r14 and ymm8..15.
@(private)
emit_chan_fetch_sub :: proc(buf: ^[dynamic]u8, decode_calls: ^[dynamic]int) {
	emit_bytes(buf, 0x48, 0x81, 0xEC)
	emit_i32(buf, FF_FRAME)
	x_st(buf, GPR_RSP, FF_X, 0)
	x_st(buf, GPR_RSP, FF_Y, 1)
	x_st(buf, GPR_RSP, FF_LAYER, 3)
	x_st(buf, GPR_RSP, FF_LOD, 2)
	x_live_held(buf)
	vzero(buf, 0)
	for k in i32(0) ..< 4 {
		x_st(buf, GPR_RSP, FF_R + k * 32, 0)
	}
	// Levels in range (u32(level) < n_levels), as bits in r8d.
	vpbroadcastd_mem(buf, 3, GPR_RDI, OFF_CI_NLEV)
	vop_d_tab(buf, 0xFA, 3, 3, OFF_TAB_ONE_U)
	vop38_d(buf, 0x3B, 4, 2, 3)
	vop_d(buf, 0x76, 4, 4, 2)
	x_movdqa_tab(buf, 0, OFF_TAB_ONE_U)
	vop38_d(buf, 0x47, 1, 0, 2)
	vop_d(buf, 0xDB, 1, 1, 4)
	x_hor_or_r8(buf)
	emit_bytes(buf, 0x45, 0x85, 0xC0) // test r8d, r8d
	none := x_jump(buf, JCC_JE)

	loop_top := len(buf^)
	x_next_level(buf)
	x_bcast_gpr(buf, 4, 9)
	x_ld(buf, 2, GPR_RSP, FF_LOD)
	vop_d(buf, 0x76, 2, 2, 4) // this level's lanes
	x_ld(buf, 0, GPR_RSP, FF_X)
	x_ld(buf, 1, GPR_RSP, FF_Y)
	emit_bytes(buf, 0xF6, 0xC6, 0x01) // test dh, 1
	no_flip := x_jump(buf, JCC_JE)
	vpbroadcastd_mem(buf, 5, GPR_RSI, OFF_LV_MAX + 4)
	vop_d(buf, 0xFA, 1, 5, 1) // max_y - y
	x_land(buf, no_flip)
	vpbroadcastd_mem(buf, 5, GPR_RSI, OFF_LV_MAX)
	vop38_d(buf, 0x3B, 6, 0, 5)
	vop_d(buf, 0x76, 6, 6, 0)
	vpbroadcastd_mem(buf, 5, GPR_RSI, OFF_LV_MAX + 4)
	vop38_d(buf, 0x3B, 7, 1, 5)
	vop_d(buf, 0x76, 7, 7, 1)
	vop_d(buf, 0xDB, 2, 2, 6)
	vop_d(buf, 0xDB, 2, 2, 7) // and in range
	vpbroadcastd_mem(buf, 5, GPR_RSI, OFF_LV_PITCH)
	vop38_d(buf, 0x40, 4, 1, 5)
	vop_d(buf, 0xFE, 4, 4, 0)
	emit_chan_layer_rows(buf, FF_LAYER, 4, 0, 5)
	vop_d(buf, 0xDB, 4, 4, 2) // other lanes read texel 0
	x_st(buf, GPR_RSP, FF_M, 2)
	x_call(buf, decode_calls)
	for k in 0 ..< 4 {
		emit_bytes(buf, 0x41, 0xF6, 0xC3, u8(1) << uint(k)) // test r11b, 1 << k
		s0 := x_jump(buf, JCC_JE)
		r := FF_R + i32(k) * 32
		x_cmp_b(buf, GPR_RDI, OFF_CI_COMPS, u8(k))
		lacks := x_jump(buf, JCC_JBE)
		emit_b(buf, 0x83) // cmp dword [rdi + unorm], 0
		emit_modrm_mem(buf, 7, GPR_RDI, OFF_CI_UNORM)
		emit_b(buf, 0)
		float := x_jump(buf, JCC_JE)
		vbroadcastss_mem(buf, 5, GPR_RDI, OFF_CI_UNORM)
		vop_ps(buf, 0x5E, u8(k), u8(k), 5)
		x_land(buf, float)
		x_ld(buf, 6, GPR_RSP, r)
		x_ld(buf, 5, GPR_RSP, FF_M)
		vblendvps(buf, 6, 6, u8(k), 5)
		x_st(buf, GPR_RSP, r, 6)
		done := x_jump(buf, 0)
		x_land(buf, lacks)
		if k == 3 {
			x_ld(buf, 6, GPR_RSP, r)
			x_ld(buf, 5, GPR_RSP, FF_M)
			x_movdqa_tab(buf, 7, OFF_TAB_ONE_F)
			vblendvps(buf, 6, 6, 7, 5)
			x_st(buf, GPR_RSP, r, 6)
		}
		x_land(buf, done)
		x_land(buf, s0)
	}
	emit_bytes(buf, 0x45, 0x85, 0xC0) // test r8d, r8d
	emit_bytes(buf, 0x0F, 0x85)       // jnz loop_top
	emit_i32(buf, i32(loop_top - (len(buf^) + 4)))
	x_land(buf, none)
	for k in i32(0) ..< 4 {
		x_ld(buf, u8(k), GPR_RSP, FF_R + k * 32)
	}
	emit_bytes(buf, 0x48, 0x81, 0xC4)
	emit_i32(buf, FF_FRAME)
	emit_b(buf, 0xC3)
}

// fetch_one: a 2D image's texelFetch, the stub's whole body. One level
// (fetch_route its format) inline, with fetch_sub's rule: the texel at
// level 0, (0, 0, 0, 0) out of range (the gathers' masks leave those lanes
// 0, which every format decodes to 0), the components the format lacks
// (0, 0, 0, 1); the rest (FETCH_GENERAL) through fetch_sub. F32 is tested
// first and the general route before the bounds, so neither pays for the
// other formats; each route ends where the caller stores the result.
//   in:  ymm0 = x, ymm1 = y, ymm2 = lod, rdi = ^Chan_Img, rsi = its texels
//   out: ymm0..3's live components (`comp_mask`); clobbers rax, rcx, rdx
//        and ymm4..7.
@(private)
emit_chan_fetch_one :: proc(buf: ^[dynamic]u8, fetch_calls: ^[dynamic]int, comp_mask: u8, flip, f16c: bool) {
	live :: #force_inline proc(m: u8, k: int) -> bool {return (m & (u8(1) << uint(k))) != 0}
	if comp_mask == 0 {
		return
	}
	// In range (unsigned: min_u(p, max) == p; and the lod 0) into ymm5,
	// y * pitch + x into ymm4; clobbers ymm1, ymm3 and ymm6.
	bounds :: proc(buf: ^[dynamic]u8, flip: bool) {
		lv0 := OFF_CI_LEVELS
		if flip {
			vpbroadcastd_mem(buf, 6, GPR_RDI, lv0 + OFF_LV_MAX + 4)
			vop_d(buf, 0xFA, 1, 6, 1) // max_y - py
		}
		vpbroadcastd_mem(buf, 6, GPR_RDI, lv0 + OFF_LV_MAX)
		vop38_d(buf, 0x3B, 3, 0, 6)
		vop_d(buf, 0x76, 3, 3, 0)
		vpbroadcastd_mem(buf, 6, GPR_RDI, lv0 + OFF_LV_MAX + 4)
		vop38_d(buf, 0x3B, 4, 1, 6)
		vop_d(buf, 0x76, 4, 4, 1)
		vop_d(buf, 0xDB, 5, 3, 4)
		vzero(buf, 6)
		vop_d(buf, 0x76, 6, 6, 2)
		vop_d(buf, 0xDB, 5, 5, 6)
		vpbroadcastd_mem(buf, 6, GPR_RDI, lv0 + OFF_LV_PITCH)
		vop38_d(buf, 0x40, 1, 1, 6)
		vop_d(buf, 0xFE, 4, 1, 0)
	}
	// A dword per lane in range at [base + ymm(idx) << scale + disp] into
	// ymm(dst), 0 elsewhere (ymm7: the mask's copy).
	gather :: proc(buf: ^[dynamic]u8, dst, base, idx, scale: u8, disp: i32) {
		vmovaps_rr(buf, 7, 5)
		vzero(buf, dst)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, dst, 7, base, idx, scale, disp)
	}
	done: [4]int
	n_done := 0

	// F32: element index << shift, its components (missing: 0, 0, 0, 1).
	x_cmp_b(buf, GPR_RDI, OFF_CI_FETCH_ROUTE, u8(Channel_Format.F32))
	not_f32 := x_jump(buf, JCC_JNE)
	bounds(buf, flip)
	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [rdi + shift]
	emit_modrm_mem(buf, GPR_RCX, GPR_RDI, OFF_CI_SHIFT)
	x_bcast_gpr(buf, 6, GPR_RCX)
	vop38_d(buf, 0x47, 4, 4, 6)
	emit_chan_gather_rgba(buf, GPR_RDI, 0, 4, 5, comp_mask)
	done[n_done] = x_jump(buf, 0)
	n_done += 1

	x_land(buf, not_f32)
	x_cmp_b(buf, GPR_RDI, OFF_CI_FETCH_ROUTE, FETCH_GENERAL)
	general := x_jump(buf, JCC_JE)
	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [rdi + fetch_route]
	emit_modrm_mem(buf, GPR_RCX, GPR_RDI, OFF_CI_FETCH_ROUTE)
	bounds(buf, flip)

	// Rgba8: byte k of the texel's dword, / 255.
	emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Rgba8)) // cmp ecx, Rgba8
	not_rgba8 := x_jump(buf, JCC_JNE)
	gather(buf, 6, GPR_RSI, 4, 2, 0)
	for k in 0 ..< 4 {
		if !live(comp_mask, k) {
			continue
		}
		r := u8(k)
		if k == 0 {
			vop_d_tab(buf, 0xDB, r, 6, OFF_TAB_BYTE)
		} else {
			vshift_imm(buf, 2, r, 6, u8(8 * k))
			if k < 3 {
				vop_d_tab(buf, 0xDB, r, r, OFF_TAB_BYTE)
			}
		}
		x_dq2ps(buf, r, r)
		vop_ps_tab(buf, 0x5E, r, r, OFF_TAB_F255)
	}
	done[n_done] = x_jump(buf, 0)
	n_done += 1

	// Gray8 and Gray16: the aligned dword around byte offset o = index *
	// bytes + mis of level 0's aligned texels, shifted right (o & 3) * 8
	// and masked, / 255 or / 65535; then (r, 0, 0, 1).
	x_land(buf, not_rgba8)
	emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Rgba16f))
	is_half := x_jump(buf, JCC_JE)
	if live(comp_mask, 0) {
		emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Gray16))
		gray8 := x_jump(buf, JCC_JNE)
		vop_d(buf, 0xFE, 4, 4, 4) // Gray16: 2 bytes a texel
		x_land(buf, gray8)
		emit_bytes(buf, 0x48, 0x8B) // mov rax, [rdi + levels[0].ptr]
		emit_modrm_mem(buf, GPR_RAX, GPR_RDI, OFF_CI_LEVELS + OFF_LV_PTR)
		vpbroadcastd_mem(buf, 6, GPR_RDI, OFF_CI_LEVELS + OFF_LV_MIS)
		vop_d(buf, 0xFE, 4, 4, 6)
		vop_d_tab(buf, 0xDB, 6, 4, OFF_TAB_THREE_U)
		vshift_imm(buf, 6, 6, 6, 3)
		vop_d_tab(buf, 0xDB, 4, 4, OFF_TAB_NOT3)
		gather(buf, 0, GPR_RAX, 4, 0, 0)
		vop38_d(buf, 0x45, 0, 0, 6) // vpsrlvd
		emit_bytes(buf, 0x83, 0xF9, u8(Channel_Format.Gray16))
		m8 := x_jump(buf, JCC_JNE)
		vop_d_tab(buf, 0xDB, 0, 0, OFF_TAB_U16)
		m16 := x_jump(buf, 0)
		x_land(buf, m8)
		vop_d_tab(buf, 0xDB, 0, 0, OFF_TAB_BYTE)
		x_land(buf, m16)
		x_dq2ps(buf, 0, 0)
		vbroadcastss_mem(buf, 6, GPR_RDI, OFF_CI_UNORM)
		vop_ps(buf, 0x5E, 0, 0, 6)
	}
	if live(comp_mask, 1) {
		vzero(buf, 1)
	}
	if live(comp_mask, 2) {
		vzero(buf, 2)
	}
	if live(comp_mask, 3) {
		vop_ps_tab(buf, 0x54, 3, 5, OFF_TAB_ONE_F)
	}
	done[n_done] = x_jump(buf, 0)
	n_done += 1

	// Rgba16f: the texel's two dwords, ymm6 (halves 0, 1) and ymm5 (2, 3).
	x_land(buf, is_half)
	if live(comp_mask, 0) || live(comp_mask, 1) {
		gather(buf, 6, GPR_RSI, 4, 3, 0)
	}
	if live(comp_mask, 2) || live(comp_mask, 3) {
		gather(buf, 5, GPR_RSI, 4, 3, 4)
	}
	if f16c {
		// Each pair's eight words in order (vpshufb, then vpermq 0xd8:
		// the first's in the low 128 bits, the second's in the high),
		// then vcvtph2ps, which gives f16_to_f32's bits for every half.
		for k0 in ([2]int{0, 2}) {
			lo, hi := live(comp_mask, k0), live(comp_mask, k0 + 1)
			if !lo && !hi {
				continue
			}
			src := u8(6 if k0 == 0 else 5)
			x_d38_m(buf, 0x00, 7, src, GPR_R14, OFF_TAB_H_PACK)
			emit_vex_rr(buf, 3, 1, 1, 1, 0x00, 7, 0, 7) // vpermq ymm7, ymm7, 0xd8
			append(buf, 0xD8)
			if lo {
				emit_vex_rr(buf, 2, 1, 1, 0, 0x13, u8(k0), 0, 7) // vcvtph2ps
			}
			if hi {
				emit_vex_rr(buf, 3, 1, 1, 0, 0x39, 7, 0, 4) // vextracti128 xmm4, ymm7, 1
				append(buf, 1)
				emit_vex_rr(buf, 2, 1, 1, 0, 0x13, u8(k0 + 1), 0, 4)
			}
		}
	} else {
		for k := 3; k >= 0; k -= 1 {
			if live(comp_mask, k) {
				x_half_comp(buf, k)
			}
		}
	}
	done[n_done] = x_jump(buf, 0)
	n_done += 1

	x_land(buf, general)
	emit_b(buf, 0xBA) // mov edx, live components | flip y << 8
	emit_u32(buf, u32(comp_mask) | u32(flip) << 8)
	x_call(buf, fetch_calls)
	for i in 0 ..< n_done {
		x_land(buf, done[i])
	}
}

@(private)
emit_store_mask_al :: proc(buf: ^[dynamic]u8, m: i32) {
	if m == MASK_EXEC {
		emit_bytes(buf, 0x41, 0x89, 0xC5) // mov r13d, eax
	} else {
		emit_bytes(buf, 0x41, 0x88) // mov byte [r12 + m], al
		emit_modrm_mem(buf, GPR_RAX, GPR_R12, m)
	}
}

@(private)
emit_helper_call :: proc(buf: ^[dynamic]u8, fn: Jit_Help_Fn, in_ptr: ^Instr) {
	emit_bytes(buf, 0xC5, 0xF8, 0x77) // vzeroupper
	emit_bytes(buf, 0x48, 0x89, 0xEF) // mov rdi, rbp
	emit_bytes(buf, 0x48, 0xBE)       // movabs rsi, in_ptr
	emit_u64(buf, u64(uintptr(in_ptr)))
	emit_bytes(buf, 0x44, 0x89, 0xEA) // mov edx, r13d
	emit_bytes(buf, 0x48, 0xB8)       // movabs rax, fn
	emit_u64(buf, u64(uintptr(rawptr(fn))))
	emit_bytes(buf, 0xFF, 0xD0)       // call rax
}

// Internal SysV-bridging subroutine for cold unbound texture callbacks when
// 3+ `ymm8..15` registers hold live basic-block values:
//   in:  rbp = ^Jit_Ctx, rsi = ^Instr, r13d = exec, rax = Jit_Help_Fn
//   out: preserves rbp, rbx, r12..r14 and ymm8..15.
@(private)
emit_helper_ymm_sub :: proc(buf: ^[dynamic]u8) {
	// Entry rsp % 16 == 8 (after `call`); 264 + 8 = 272 == 0 (mod 16) before `call rax`.
	emit_bytes(buf, 0x48, 0x81, 0xEC) // sub rsp, 264
	emit_u32(buf, 264)
	for i in 0 ..< 8 {
		emit_vex_rm(buf, 1, 0, 1, 0, 0x11, u8(8 + i), 0, GPR_RSP, i32(i * 32))
	}
	emit_bytes(buf, 0xC5, 0xF8, 0x77) // vzeroupper
	emit_bytes(buf, 0x48, 0x89, 0xEF) // mov rdi, rbp
	emit_bytes(buf, 0x44, 0x89, 0xEA) // mov edx, r13d
	emit_bytes(buf, 0xFF, 0xD0)       // call rax
	for i in 0 ..< 8 {
		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, u8(8 + i), 0, GPR_RSP, i32(i * 32))
	}
	emit_bytes(buf, 0x48, 0x81, 0xC4) // add rsp, 264
	emit_u32(buf, 264)
	emit_b(buf, 0xC3) // ret
}

// Internal AVX2 subroutine for cold uniform Load_Buf edge cases (n in 2..4,
// straddling buf_len or wrapping u32(max)):
//   in:  rax = u64(u32(w0)), rcx = buf_len, rsi = buf_ptr, edx = n (2..4)
//   out: ymm0..ymm(n-1) = broadcast or zeroed components; preserves rbp, rbx, r12..r14 and ymm8..15.
@(private)
emit_unif_oob_sub :: proc(buf: ^[dynamic]u8) {
	for i in 0 ..< i32(4) {
		if i >= 2 {
			emit_bytes(buf, 0x83, 0xFA, u8(i)) // cmp edx, i
			emit_bytes(buf, 0x77, 0x01, 0xC3)  // ja +1; ret
		}
		if i > 0 {
			emit_bytes(buf, 0x83, 0xC0, 0x01) // add eax, 1 (32-bit wrap + zero-extend rax)
		}
		ui := u8(i)
		emit_bytes(buf, 0x48, 0x39, 0xC8) // cmp rax, rcx
		emit_b(buf, 0x73)                 // jae oob
		oob_p := len(buf^)
		emit_b(buf, 0)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x18, ui, 0, GPR_RSI, GPR_RAX, 2, 0) // vbroadcastss ymm(i), [rsi + rax*4]
		emit_b(buf, 0xEB) // jmp next_i
		next_p := len(buf^)
		emit_b(buf, 0)
		buf[oob_p] = u8(len(buf^) - (oob_p + 1))
		vzero(buf, ui)
		buf[next_p] = u8(len(buf^) - (next_p + 1))
	}
	emit_b(buf, 0xC3) // ret
}

// Internal AVX2 subroutine for cold divergent Load_Buf gathers on > 2^31-word buffers:
//   in:  rcx = buf_len (> 2^31), rdx = 0x8000_0000, rsi = buf_ptr, ymm4 = w0, eax = n (1..4)
//   out: ymm0..ymm(n-1) = gathered components; preserves rbp, rbx, r12..r14 and ymm8..15.
@(private)
emit_big_buf_sub :: proc(buf: ^[dynamic]u8) {
	emit_bytes(buf, 0x48, 0x29, 0xD1) // sub rcx, rdx -> rem in 1..2^31
	emit_bytes(buf, 0x48, 0x8D)       // lea rdi, [rsi + rdx*4] -> rsi + 8 GiB
	emit_modrm_sib(buf, GPR_RDI, GPR_RSI, GPR_RDX, 2, 0)
	emit_vex_rr(buf, 1, 1, 0, 0, 0x6E, 5, 0, GPR_RCX)
	emit_vex_rr(buf, 2, 1, 1, 0, 0x58, 5, 0, 5)
	for i in 0 ..< i32(4) {
		if i > 0 {
			emit_bytes(buf, 0x83, 0xF8, u8(i)) // cmp eax, i
			emit_bytes(buf, 0x77, 0x01, 0xC3)  // ja +1; ret
			vop_d_tab(buf, 0xFE, 4, 4, OFF_TAB_ONE_U)
		}
		ui := u8(i)
		vzero(buf, ui)
		vop_d_tab(buf, 0xDB, 6, 4, OFF_TAB_ABS)  // ymm6 = w & 0x7fffffff (non-negative)
		vop_d_tab(buf, 0xEF, 7, 4, OFF_TAB_ALL1) // MSB=1 where w < 2^31
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, ui, 7, GPR_RSI, 6, 2, 0)
		vop38_d(buf, 0x3B, 7, 6, 5)              // vpminud(w_lo, rem)
		vop_d(buf, 0x76, 7, 7, 5)                // ~0 where w_lo >= rem
		vop_d(buf, 0xDF, 7, 7, 4)                // vpandn: MSB=1 where w >= 2^31 && w_lo < rem
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, ui, 7, GPR_RDI, 6, 2, 0)
	}
	emit_b(buf, 0xC3) // ret
}

@(private)
ensure_exec_ymask :: #force_inline proc(e: ^Jit_Emitter) -> u8 {
	if !e.exec_ymask_valid {
		emit_exec_ymask(&e.buf, 7)
		e.exec_ymask_valid = true
	}
	return 7
}

@(private)
emit_load_mask_eax :: proc(e: ^Jit_Emitter, m: i32) {
	if is_mask_eq_exec(e, m) {
		emit_bytes(&e.buf, 0x44, 0x89, 0xE8) // mov eax, r13d
	} else {
		emit_bytes(&e.buf, 0x41, 0x0F, 0xB6) // movzx eax, byte [r12 + m]
		emit_modrm_mem(&e.buf, GPR_RAX, GPR_R12, m)
	}
}

@(private)
emit_copy_mask :: proc(e: ^Jit_Emitter, dst, a: i32) {
	if dst == a || (dst == MASK_EXEC && is_mask_eq_exec(e, a)) {
		return
	}
	if is_mask_eq_exec(e, a) {
		emit_bytes(&e.buf, 0x45, 0x88) // mov byte [r12 + dst], r13b
		emit_modrm_mem(&e.buf, GPR_R13, GPR_R12, dst)
		e.mask_eq_gen[dst] = e.exec_gen
	} else if dst == MASK_EXEC {
		emit_bytes(&e.buf, 0x45, 0x0F, 0xB6) // movzx r13d, byte [r12 + a]
		emit_modrm_mem(&e.buf, GPR_R13, GPR_R12, a)
		invalidate_mask_write(e, MASK_EXEC)
		e.mask_eq_gen[a] = e.exec_gen
	} else {
		emit_load_mask_eax(e, a)
		emit_store_mask_al(&e.buf, dst)
		e.mask_eq_gen[dst] = 0
	}
}

// Component i of a vm_math.odin built-in (F_Fn1, F_Fn2), through the
// interpreter's math_8: the same procs, a register (eight lanes) at a time.
// The operands are read from their slots; the result is returned in ymm0
// (emit_math_call).
@(private, enable_target_feature = JIT_FEATURE)
jit_help_math :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, i: i32) -> F8 #no_bounds_check {
	r: [8]u32
	math_8(in_.op, Builtin(in_.k), &ctx.slots[in_.a + i], &ctx.slots[in_.b + i], &r)
	return transmute(F8)r
}

// A jit_help_math call that doesn't end the basic block (as texture
// callbacks): dirty cached operands are stored to their slots, the
// ymm8..15 values still live after this instruction saved around the call
// (the SysV ABI clobbers every YMM register), and the result taken from
// ymm0 into the destination like any other op's.
@(private)
emit_math_call :: proc(e: ^Jit_Emitter, in_ptr: ^Instr, i: i32) {
	in_ := in_ptr^
	dst := in_.dst + i
	n := i32(in_.n)
	// Every operand register the helper reads, now or for a later i, to
	// memory: an operand whose last use is this instruction leaves the cache
	// below without a store.
	for &cs, idx in e.cache {
		if cs.slot >= 0 && cs.dirty && ((cs.slot >= in_.a && cs.slot < in_.a + n) || (in_.op == .F_Fn2 && cs.slot >= in_.b && cs.slot < in_.b + n)) {
			vmovaps_store(&e.buf, cs.slot, u8(8 + idx))
			cs.dirty = false
		}
	}
	save_regs: [8]u8
	n_save := 0
	for &cs, idx in e.cache {
		if cs.slot < 0 {
			continue
		}
		if cs.slot == dst || cs.last_use <= e.pc {
			cs = {slot = -1, last_use = -1, dirty = false, stamp = 0}
		} else {
			save_regs[n_save] = u8(8 + idx)
			n_save += 1
		}
	}
	frame := i32(n_save * 32)
	if n_save > 0 {
		emit_bytes(&e.buf, 0x48, 0x81, 0xEC) // sub rsp, frame
		emit_u32(&e.buf, u32(frame))
		for k in 0 ..< n_save {
			emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x11, save_regs[k], 0, GPR_RSP, i32(k * 32))
		}
	}
	emit_bytes(&e.buf, 0x48, 0x89, 0xEF) // mov rdi, rbp
	emit_bytes(&e.buf, 0x48, 0xBE)       // movabs rsi, in_ptr
	emit_u64(&e.buf, u64(uintptr(in_ptr)))
	emit_b(&e.buf, 0xBA)                 // mov edx, i
	emit_u32(&e.buf, u32(i))
	emit_bytes(&e.buf, 0x48, 0xB8)       // movabs rax, jit_help_math
	emit_u64(&e.buf, u64(uintptr(rawptr(jit_help_math))))
	emit_bytes(&e.buf, 0xFF, 0xD0)       // call rax
	if n_save > 0 {
		for k in 0 ..< n_save {
			emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x10, save_regs[k], 0, GPR_RSP, i32(k * 32))
		}
		emit_bytes(&e.buf, 0x48, 0x81, 0xC4) // add rsp, frame
		emit_u32(&e.buf, u32(frame))
	}
	e.exec_ymask_valid = false
	copy_to_dst(e, dst, 0)
}

// Store_Dyn (its own basic block, so its operands are in memory): where every
// lane has the same offset (a loop counter's index), masked blends into the
// element's slots; otherwise jit_help_store_dyn.
@(private)
emit_store_dyn :: proc(e: ^Jit_Emitter, code: []Instr, pc: int) {
	in_ := code[pc]
	n := i32(in_.n)
	if n > 4 {
		emit_helper_call(&e.buf, jit_help_store_dyn, &code[pc])
		return
	}
	yoff := read_slot(e, in_.b, 6)
	emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 4, 0, yoff)    // vpbroadcastd ymm4, xmm(yoff)
	vop_d(&e.buf, 0x76, 4, 4, yoff)
	emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x50, GPR_RDX, 0, 4) // vmovmskps edx, ymm4
	emit_bytes(&e.buf, 0x81, 0xFA, 0xFF, 0x00, 0x00, 0x00)
	emit_bytes(&e.buf, 0x0F, 0x85) // jne divergent
	div_patch := len(e.buf)
	emit_i32(&e.buf, 0)
	emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x7E, yoff, 0, GPR_RAX) // vmovd eax, xmm(yoff)
	emit_bytes(&e.buf, 0x48, 0xC1, 0xE0, 0x05)              // shl rax, 5 (off * 32)
	ym := ensure_exec_ymask(e)
	for i in 0 ..< n {
		ya := read_slot(e, in_.a + i, 1)
		emit_vex_sib(&e.buf, 1, 0, 1, 0, 0x28, 2, 0, GPR_RBX, GPR_RAX, 0, (in_.dst + i) * 32) // vmovaps ymm2, [rbx + rax + slot]
		vpblendvb(&e.buf, 2, 2, ya, ym)
		emit_vex_sib(&e.buf, 1, 0, 1, 0, 0x29, 2, 0, GPR_RBX, GPR_RAX, 0, (in_.dst + i) * 32)
	}
	emit_b(&e.buf, 0xE9) // jmp done
	done_patch := len(e.buf)
	emit_i32(&e.buf, 0)
	patch_i32(e.buf[:], div_patch, i32(len(e.buf) - (div_patch + 4)))
	emit_helper_call(&e.buf, jit_help_store_dyn, &code[pc])
	patch_i32(e.buf[:], done_patch, i32(len(e.buf) - (done_patch + 4)))
}

// Store_Buf (its own basic block, so its operands are in memory) of up to 4
// words into one of the first 4 buffers: lane by lane in ascending order over
// the running lanes, as store_buf_lanes does, words past the buffer dropped.
// buffer_stores splits a std430 element (scalars, vectors and structs of them)
// into scalars and vectors, so n is 1 to 4 and the stride at least 1 word.
@(private)
emit_store_buf :: proc(e: ^Jit_Emitter, in_: Instr) {
	n := i32(in_.n)
	k := i32(in_.k)
	emit_bytes(&e.buf, 0x48, 0x8B) // mov rsi, [rbp + buf_ptr[k]]
	emit_modrm_mem(&e.buf, GPR_RSI, GPR_RBP, OFF_CTX_BUF_PTR + k * 8)
	emit_bytes(&e.buf, 0x48, 0x8B) // mov rcx, [rbp + buf_len[k]]
	emit_modrm_mem(&e.buf, GPR_RCX, GPR_RBP, OFF_CTX_BUF_LEN + k * 8)
	emit_bytes(&e.buf, 0x44, 0x89, 0xEA) // mov edx, r13d
	emit_bytes(&e.buf, 0x85, 0xD2)       // test edx, edx
	emit_bytes(&e.buf, 0x0F, 0x84)       // jz done
	done_patch := len(e.buf)
	emit_i32(&e.buf, 0)
	loop_top := len(e.buf)
	emit_bytes(&e.buf, 0x0F, 0xBC, 0xC2) // bsf eax, edx
	emit_b(&e.buf, 0x8B)                 // mov edi, [rbx + rax*4 + a*32]
	emit_modrm_sib(&e.buf, GPR_RDI, GPR_RBX, GPR_RAX, 2, in_.a * 32)
	stride, word := u32(in_.c), u32(in_.d)
	if stride != 1 {
		emit_bytes(&e.buf, 0x69, 0xFF) // imul edi, edi, stride
		emit_u32(&e.buf, stride)
	}
	if word != 0 {
		emit_bytes(&e.buf, 0x81, 0xC7) // add edi, word
		emit_u32(&e.buf, word)
	}
	for i in 0 ..< n {
		if i == 0 {
			emit_bytes(&e.buf, 0x41, 0x89, 0xF8) // mov r8d, edi
		} else {
			emit_bytes(&e.buf, 0x44, 0x8D, 0x47, u8(i)) // lea r8d, [rdi + i] (32-bit wrap)
		}
		emit_bytes(&e.buf, 0x49, 0x39, 0xC8) // cmp r8, rcx
		emit_b(&e.buf, 0x73)                 // jae skip
		skip_patch := len(e.buf)
		emit_b(&e.buf, 0)
		emit_bytes(&e.buf, 0x44, 0x8B) // mov r9d, [rbx + rax*4 + (b + i)*32]
		emit_modrm_sib(&e.buf, 9, GPR_RBX, GPR_RAX, 2, (in_.b + i) * 32)
		emit_bytes(&e.buf, 0x46, 0x89, 0x0C, 0x86) // mov [rsi + r8*4], r9d
		e.buf[skip_patch] = u8(len(e.buf) - (skip_patch + 1))
	}
	emit_bytes(&e.buf, 0x44, 0x8D, 0x52, 0xFF) // lea r10d, [rdx - 1]
	emit_bytes(&e.buf, 0x44, 0x21, 0xD2)       // and edx, r10d
	emit_bytes(&e.buf, 0x0F, 0x85)             // jnz loop_top
	emit_i32(&e.buf, i32(loop_top - (len(e.buf) + 4)))
	patch_i32(e.buf[:], done_patch, i32(len(e.buf) - (done_patch + 4)))
}

// Image_Load of a 4- or 16-byte texel format (rgba8, r32f, r32ui, r32i,
// rgba32f) with AVX2 gathers, as image_load_fmt does it: running lanes in
// bounds read their texel, the others 0.
@(private)
emit_image_load :: proc(e: ^Jit_Emitter, in_: Instr, fmt: Image_Format) {
	dst := in_.dst
	emit_image_addr(e, in_, fmt)
	if fmt == .Rgba32f {
		for i in i32(0) ..< 4 {
			if dead_write(e, dst + i) {
				continue
			}
			yd := prep_dst(e, dst + i, 3)
			vmovaps_rr(&e.buf, 6, 2)
			vzero(&e.buf, yd)
			emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, yd, 6, GPR_RSI, 4, 0, i * 4) // vpgatherdd yd, [rsi + ymm4 + 4i], ymm6
			commit_dst(e, dst + i, yd)
		}
		return
	}
	vmovaps_rr(&e.buf, 6, 2)
	vzero(&e.buf, 0)
	emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, 0, 6, GPR_RSI, 4, 0, 0) // vpgatherdd ymm0, [rsi + ymm4], ymm6
	if fmt == .Rgba8 {
		// f32(byte) / 255 rounds exactly as UNORM8_TO_F32 (test_jit_image_load_rgba8_lut).
		for i in i32(0) ..< 4 {
			if dead_write(e, dst + i) {
				continue
			}
			yd := prep_dst(e, dst + i, 3)
			src := u8(0)
			if i > 0 {
				vshift_imm(&e.buf, 2, yd, 0, u8(8 * i))
				src = yd
			}
			if i < 3 {
				vop_d_tab(&e.buf, 0xDB, yd, src, OFF_TAB_BYTE)
			}
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x5B, yd, 0, yd) // vcvtdq2ps
			vop_ps_tab(&e.buf, 0x5E, yd, yd, OFF_TAB_F255)
			commit_dst(e, dst + i, yd)
		}
		return
	}
	// Single-channel formats: (texel, 0, 0, 1) in bounds.
	if !dead_write(e, dst) {
		copy_to_dst(e, dst, 0)
	}
	for i in i32(1) ..< 3 {
		if dead_write(e, dst + i) {
			continue
		}
		yd := prep_dst(e, dst + i, 3)
		vzero(&e.buf, yd)
		commit_dst(e, dst + i, yd)
	}
	if !dead_write(e, dst + 3) {
		yd := prep_dst(e, dst + 3, 3)
		vop_d_tab(&e.buf, 0xDB, yd, 2, OFF_TAB_ONE_F if fmt == .R32f else OFF_TAB_ONE_U)
		commit_dst(e, dst + 3, yd)
	}
}

// Image_Store (its own basic block, so its operands are in memory) of a 4- or
// 16-byte texel format: offsets and rgba8 texels computed 8 lanes at once,
// then stored lane by lane in ascending order over the running lanes in
// bounds, as image_store_fmt does it.
@(private)
emit_image_store :: proc(e: ^Jit_Emitter, in_: Instr, fmt: Image_Format) {
	b := in_.b
	emit_image_addr(e, in_, fmt)
	emit_bytes(&e.buf, 0x48, 0x83, 0xEC, 64)           // sub rsp, 64
	emit_vex_rm(&e.buf, 1, 2, 1, 0, 0x7F, 4, 0, GPR_RSP, 0) // vmovdqu [rsp], ymm4
	if fmt == .Rgba8 {
		// round(min(max(v, 0), 1) * 255) per channel (NaN to 0), packed.
		vzero(&e.buf, 5)
		for i in i32(0) ..< 4 {
			y := read_slot(e, b + i, 1)
			vop_ps(&e.buf, 0x5F, 1, y, 5)                    // max(v, 0): NaN gives the second operand, 0
			vop_ps_tab(&e.buf, 0x5D, 1, 1, OFF_TAB_ONE_F)
			vop_ps_tab(&e.buf, 0x59, 1, 1, OFF_TAB_F255)
			vop_ps_tab(&e.buf, 0x58, 3, 1, OFF_TAB_HALF_BELOW) // x >= 0: round's x + 0.49999997
			emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 3, 0, 3)   // vcvttps2dq truncates
			if i == 0 {
				vmovaps_rr(&e.buf, 0, 3)
			} else {
				vshift_imm(&e.buf, 6, 3, 3, u8(8 * i))
				vop_d(&e.buf, 0xEB, 0, 0, 3)
			}
		}
		emit_vex_rm(&e.buf, 1, 2, 1, 0, 0x7F, 0, 0, GPR_RSP, 32) // vmovdqu [rsp + 32], ymm0
	}
	emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x50, GPR_RDX, 0, 2) // vmovmskps edx, ymm2
	emit_bytes(&e.buf, 0x85, 0xD2)                       // test edx, edx
	emit_bytes(&e.buf, 0x0F, 0x84)                       // jz done
	done_patch := len(e.buf)
	emit_i32(&e.buf, 0)
	loop_top := len(e.buf)
	emit_bytes(&e.buf, 0x0F, 0xBC, 0xC2) // bsf eax, edx
	emit_b(&e.buf, 0x8B)                 // mov ecx, [rsp + rax*4]
	emit_modrm_sib(&e.buf, GPR_RCX, GPR_RSP, GPR_RAX, 2, 0)
	words := 4 if fmt == .Rgba32f else 1
	for i in 0 ..< words {
		emit_bytes(&e.buf, 0x44, 0x8B) // mov r9d, [rsp + 32 + rax*4] or [rbx + rax*4 + (b + i)*32]
		if fmt == .Rgba8 {
			emit_modrm_sib(&e.buf, 9, GPR_RSP, GPR_RAX, 2, 32)
		} else {
			emit_modrm_sib(&e.buf, 9, GPR_RBX, GPR_RAX, 2, (b + i32(i)) * 32)
		}
		emit_bytes(&e.buf, 0x44, 0x89) // mov [rsi + rcx + 4i], r9d
		emit_modrm_sib(&e.buf, 9, GPR_RSI, GPR_RCX, 0, i32(i) * 4)
	}
	emit_bytes(&e.buf, 0x44, 0x8D, 0x52, 0xFF) // lea r10d, [rdx - 1]
	emit_bytes(&e.buf, 0x44, 0x21, 0xD2)       // and edx, r10d
	emit_bytes(&e.buf, 0x0F, 0x85)             // jnz loop_top
	emit_i32(&e.buf, i32(loop_top - (len(e.buf) + 4)))
	patch_i32(e.buf[:], done_patch, i32(len(e.buf) - (done_patch + 4)))
	emit_bytes(&e.buf, 0x48, 0x83, 0xC4, 64) // add rsp, 64
}

// rsi = image k's data, ymm2 = the running lanes whose (a, a + 1) is in
// bounds, ymm4 = their texels' byte offsets (y * stride + x * texel bytes).
@(private)
emit_image_addr :: proc(e: ^Jit_Emitter, in_: Instr, fmt: Image_Format) {
	a := in_.a
	disp := OFF_CTX_IMGS + i32(in_.k) * size_of(Jit_Img)
	yx := read_slot(e, a, 0)
	yy := read_slot(e, a + 1, 1, pin_bit(yx))
	ym := ensure_exec_ymask(e)
	emit_bytes(&e.buf, 0x48, 0x8B) // mov rsi, [rbp + img.ptr]
	emit_modrm_mem(&e.buf, GPR_RSI, GPR_RBP, disp + OFF_IMG_PTR)
	vop_d_tab(&e.buf, 0xEF, 2, yx, OFF_TAB_SIGN) // x ^ 2^31
	vpbroadcastd_mem(&e.buf, 3, GPR_RBP, disp + OFF_IMG_WB)
	vop_d(&e.buf, 0x66, 2, 3, 2)                 // u32(x) < width
	vop_d_tab(&e.buf, 0xEF, 4, yy, OFF_TAB_SIGN)
	vpbroadcastd_mem(&e.buf, 3, GPR_RBP, disp + OFF_IMG_HB)
	vop_d(&e.buf, 0x66, 4, 3, 4)                 // u32(y) < height
	vop_d(&e.buf, 0xDB, 2, 2, 4)
	vop_d(&e.buf, 0xDB, 2, 2, ym)                // ymm2 = in bounds and running
	vpbroadcastd_mem(&e.buf, 3, GPR_RBP, disp + OFF_IMG_STRIDE)
	vop38_d(&e.buf, 0x40, 4, yy, 3)              // y * stride
	vshift_imm(&e.buf, 6, 5, yx, 4 if fmt == .Rgba32f else 2)
	vop_d(&e.buf, 0xFE, 4, 4, 5)                 // ymm4 = texel byte offset
}

@(private)
finish_fused_cmp :: #force_inline proc(e: ^Jit_Emitter, cond: i32, pc_mc: int, inv: bool) {
	for &cs in e.cache {
		if cs.slot == cond {
			cs = {slot = -1, last_use = -1, dirty = false, stamp = 0}
		}
	}
	emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x50, GPR_RCX, 0, 0) // vmovmskps ecx, ymm0
	e.fused_mc_pc = i32(pc_mc)
	e.fused_mc_inv = inv
}

@(private)
jit_emit_fn :: proc(e: ^Jit_Emitter, p: ^Program, code: []Instr, is_init: bool) {
	entry_exec_end := jit_plan_fn(e, p, code, is_init)
	n_code := len(code)
	rw := e.rw[:n_code]
	pc_off := e.pc_off[:n_code + 1]

	// Prologue: 5 callee-saved GPRs (40 B + 8 B ret addr = 48 B -> 16-byte aligned rsp).
	emit_bytes(&e.buf, 0x55, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56)
	emit_bytes(&e.buf, 0x48, 0x89, 0xFD) // mov rbp, rdi
	emit_bytes(&e.buf, 0x48, 0x8B)       // mov rbx, [rbp + OFF_CTX_SLOTS]
	emit_modrm_mem(&e.buf, GPR_RBX, GPR_RBP, OFF_CTX_SLOTS)
	emit_bytes(&e.buf, 0x4C, 0x8B)       // mov r12, [rbp + OFF_CTX_MASKS]
	emit_modrm_mem(&e.buf, GPR_R12, GPR_RBP, OFF_CTX_MASKS)
	emit_bytes(&e.buf, 0x44, 0x0F, 0xB6) // movzx r13d, byte [rbp + OFF_CTX_LANES]
	emit_modrm_mem(&e.buf, GPR_R13, GPR_RBP, OFF_CTX_LANES)
	emit_bytes(&e.buf, 0x49, 0xBE)       // movabs r14, &jit_tables
	emit_u64(&e.buf, u64(uintptr(&jit_tables)))

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
			if !is_init && n <= 4 && (a + n <= dst || dst + n <= a) {
				ya: [4]u8
				old_reg: [4]u8
				yd: [4]u8
				pins: u8 = 0
				for i in 0 ..< n {
					ya[i] = read_slot(e, a + i, u8(2 + i), pins)
					pins |= pin_bit(ya[i])
				}
				old_pins: u8 = 0
				for i in 0 ..< n {
					old_reg[i] = 255
					for c, ci in e.cache {
						if c.slot == dst + i {
							old_reg[i] = u8(8 + ci)
							old_pins |= u8(1) << uint(ci)
							break
						}
					}
				}
				for i in 0 ..< n {
					yd[i] = prep_dst(e, dst + i, 0, pins | old_pins)
					if yd[i] >= 8 {
						pins |= pin_bit(yd[i])
						e.stamp += 1
						e.cache[yd[i] - 8] = {slot = dst + i, last_use = write_lu(e, dst + i), dirty = !need_mem_write(e, dst + i), stamp = e.stamp}
					}
				}
				was_valid := e.exec_ymask_valid
				emit_bytes(&e.buf, 0x41, 0x81, 0xFD, 0xFF, 0x00, 0x00, 0x00) // cmp r13d, 0xFF
				emit_b(&e.buf, 0x75) // jne masked
				masked_patch := len(e.buf)
				emit_b(&e.buf, 0)
				for i in 0 ..< n {
					if yd[i] >= 8 {
						if yd[i] != ya[i] {
							vmovaps_rr(&e.buf, yd[i], ya[i])
						}
						if need_mem_write(e, dst + i) {
							vmovaps_store(&e.buf, dst + i, yd[i])
						}
					} else if need_mem_write(e, dst + i) {
						vmovaps_store(&e.buf, dst + i, ya[i])
					}
				}
				emit_b(&e.buf, 0xEB) // jmp done
				done_patch := len(e.buf)
				emit_b(&e.buf, 0)
				e.buf[masked_patch] = u8(len(e.buf) - (masked_patch + 1))
				ym := ensure_exec_ymask(e)
				for i in 0 ..< n {
					if yd[i] == 0 && !need_mem_write(e, dst + i) {
						continue
					}
					yold := old_reg[i]
					if yold == 255 {
						vmovaps_load(&e.buf, 1, dst + i)
						yold = 1
					}
					vpblendvb(&e.buf, yd[i], yold, ya[i], ym)
					if need_mem_write(e, dst + i) {
						vmovaps_store(&e.buf, dst + i, yd[i])
					}
				}
				e.buf[done_patch] = u8(len(e.buf) - (done_patch + 1))
				e.exec_ymask_valid = was_valid
				break
			}
			ym := ensure_exec_ymask(e)
			for i in 0 ..< n {
				yold := read_slot(e, dst + i, 1)
				ya := read_slot(e, a + i, 2, pin_bit(yold))
				yd := prep_dst(e, dst + i, 0, pin_bit(yold) | pin_bit(ya))
				vpblendvb(&e.buf, yd, yold, ya, ym)
				commit_dst(e, dst + i, yd)
			}

		case .Store_Zero:
			if pc < entry_exec_end {
				z0_ready := false
				for i in 0 ..< n {
					yd := prep_dst(e, dst + i, 0)
					if yd != 0 || !z0_ready {
						vzero(&e.buf, yd)
						if yd == 0 {
							z0_ready = true
						}
					}
					commit_dst(e, dst + i, yd)
				}
				break
			}
			if !is_init && n <= 4 {
				old_reg: [4]u8
				yd: [4]u8
				old_pins: u8 = 0
				for i in 0 ..< n {
					old_reg[i] = 255
					for c, ci in e.cache {
						if c.slot == dst + i {
							old_reg[i] = u8(8 + ci)
							old_pins |= u8(1) << uint(ci)
							break
						}
					}
				}
				pins := old_pins
				for i in 0 ..< n {
					yd[i] = prep_dst(e, dst + i, 0, pins)
					if yd[i] >= 8 {
						pins |= pin_bit(yd[i])
						e.stamp += 1
						e.cache[yd[i] - 8] = {slot = dst + i, last_use = write_lu(e, dst + i), dirty = !need_mem_write(e, dst + i), stamp = e.stamp}
					}
				}
				was_valid := e.exec_ymask_valid
				emit_bytes(&e.buf, 0x41, 0x81, 0xFD, 0xFF, 0x00, 0x00, 0x00) // cmp r13d, 0xFF
				emit_b(&e.buf, 0x75) // jne masked
				masked_patch := len(e.buf)
				emit_b(&e.buf, 0)
				z0_ready := false
				for i in 0 ..< n {
					if yd[i] >= 8 {
						vzero(&e.buf, yd[i])
						if need_mem_write(e, dst + i) {
							vmovaps_store(&e.buf, dst + i, yd[i])
						}
					} else if need_mem_write(e, dst + i) {
						if !z0_ready {
							vzero(&e.buf, 0)
							z0_ready = true
						}
						vmovaps_store(&e.buf, dst + i, 0)
					}
				}
				emit_b(&e.buf, 0xEB) // jmp done
				done_patch := len(e.buf)
				emit_b(&e.buf, 0)
				e.buf[masked_patch] = u8(len(e.buf) - (masked_patch + 1))
				ym := ensure_exec_ymask(e)
				for i in 0 ..< n {
					if yd[i] == 0 && !need_mem_write(e, dst + i) {
						continue
					}
					yold := old_reg[i]
					if yold == 255 {
						vmovaps_load(&e.buf, 1, dst + i)
						yold = 1
					}
					vop_d(&e.buf, 0xDF, yd[i], ym, yold)
					if need_mem_write(e, dst + i) {
						vmovaps_store(&e.buf, dst + i, yd[i])
					}
				}
				e.buf[done_patch] = u8(len(e.buf) - (done_patch + 1))
				e.exec_ymask_valid = was_valid
				break
			}
			ym := ensure_exec_ymask(e)
			for i in 0 ..< n {
				yold := read_slot(e, dst + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(yold))
				vop_d(&e.buf, 0xDF, yd, ym, yold) // vpandn yd, ymm7, yold
				commit_dst(e, dst + i, yd)
			}

		case .Load_Dyn:
			e.exec_ymask_valid = false
			if n > 4 {
				emit_helper_call(&e.buf, jit_help_load_dyn, &code[pc])
				break
			}
			yoff := read_slot(e, b, 6)
			pins := pin_bit(yoff)
			yd: [4]u8
			for i in 0 ..< n {
				yd[i] = prep_dst(e, dst + i, u8(i), pins)
				pins |= pin_bit(yd[i])
			}
			emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 4, 0, yoff)
			vop_d(&e.buf, 0x76, 4, 4, yoff)
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x50, GPR_RDX, 0, 4)
			emit_bytes(&e.buf, 0x81, 0xFA, 0xFF, 0x00, 0x00, 0x00)
			emit_bytes(&e.buf, 0x0F, 0x85) // jne divergent
			div_dyn := len(e.buf)
			emit_i32(&e.buf, 0)

			emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x7E, yoff, 0, GPR_RAX) // vmovd eax, xmm(yoff)
			emit_bytes(&e.buf, 0x48, 0xC1, 0xE0, 0x05)              // shl rax, 5 (off * 32)
			for i in 0 ..< n {
				emit_vex_sib(&e.buf, 1, 0, 1, 0, 0x28, yd[i], 0, GPR_RBX, GPR_RAX, 0, (a + i) * 32)
			}
			emit_b(&e.buf, 0xE9) // jmp done
			done_dyn := len(e.buf)
			emit_i32(&e.buf, 0)

			patch_i32(e.buf[:], div_dyn, i32(len(e.buf) - (div_dyn + 4)))
			emit_vex_rm(&e.buf, 2, 1, 1, 0, 0x47, 5, yoff, GPR_R14, OFF_TAB_THREE_U) // off << 3
			vop_d_tab(&e.buf, 0xFE, 5, 5, OFF_TAB_IOTA)                              // off*8 + lane
			for i in 0 ..< n {
				emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 7, 0, GPR_R14, OFF_TAB_ALL1)
				vzero(&e.buf, yd[i])
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, yd[i], 7, GPR_RBX, 5, 2, (a + i) * 32)
			}

			patch_i32(e.buf[:], done_dyn, i32(len(e.buf) - (done_dyn + 4)))
			for i in 0 ..< n {
				commit_dst(e, dst + i, yd[i])
			}

		case .Store_Dyn:
			emit_store_dyn(e, code, pc)

		case .F_Add, .F_Sub, .F_Mul, .F_Div:
			op_byte: u8 = 0x58
			#partial switch in_.op {
			case .F_Sub: op_byte = 0x5C
			case .F_Mul: op_byte = 0x59
			case .F_Div: op_byte = 0x5E
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
				vop_ps(&e.buf, op_byte, yd, ya, yb)
				commit_dst(e, dst + i, yd)
			}

		case .F_Neg:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_ps_tab(&e.buf, 0x57, yd, ya, OFF_TAB_SIGN) // vxorps
				commit_dst(e, dst + i, yd)
			}

		case .I_Add, .I_Sub, .I_Mul, .Bit_And, .Bit_Or, .Bit_Xor:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				#partial switch in_.op {
				case .I_Add:   vop_d(&e.buf, 0xFE, yd, ya, yb)
				case .I_Sub:   vop_d(&e.buf, 0xFA, yd, ya, yb)
				case .I_Mul:   vop38_d(&e.buf, 0x40, yd, ya, yb)
				case .Bit_And: vop_d(&e.buf, 0xDB, yd, ya, yb)
				case .Bit_Or:  vop_d(&e.buf, 0xEB, yd, ya, yb)
				case .Bit_Xor: vop_d(&e.buf, 0xEF, yd, ya, yb)
				}
				commit_dst(e, dst + i, yd)
			}

		case .I_Div, .U_Div, .I_Rem, .U_Rem:
			e.exec_ymask_valid = false
			is_u := in_.op == .U_Div || in_.op == .U_Rem
			is_rem := in_.op == .I_Rem || in_.op == .U_Rem
			cvt_pd :: #force_inline proc(buf: ^[dynamic]u8, y_dst, x_src: u8, unsigned: bool) {
				if unsigned {
					emit_vex_rr(buf, 2, 1, 1, 0, 0x35, y_dst, 0, x_src)     // vpmovzxdq ymm, xmm
					vop_d_tab(buf, 0xEB, y_dst, y_dst, OFF_TAB_TWO52_D)     // vpor ymm, [two52_d]
					vop_d_tab(buf, 0x5C, y_dst, y_dst, OFF_TAB_TWO52_D)     // vsubpd ymm, [two52_d]
				} else {
					emit_vex_rr(buf, 1, 2, 1, 0, 0xE6, y_dst, 0, x_src)     // vcvtdq2pd ymm, xmm
				}
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 6)
				yb := read_slot(e, b + i, 7, pin_bit(ya))
				// Low 4 lanes -> xmm0, high 4 lanes -> xmm3, combined in ymm0.
				cvt_pd(&e.buf, 1, ya, is_u)
				cvt_pd(&e.buf, 2, yb, is_u)
				vop_d(&e.buf, 0x5E, 0, 1, 2)                            // vdivpd ymm0, ymm1, ymm2
				emit_vex_rr(&e.buf, 1, 1, 1, 0, 0xE6, 0, 0, 0)          // vcvttpd2dq xmm0, ymm0
				emit_vex_rr(&e.buf, 3, 1, 1, 0, 0x39, ya, 0, 1)         // vextracti128 xmm1, ya, 1
				emit_b(&e.buf, 1)
				emit_vex_rr(&e.buf, 3, 1, 1, 0, 0x39, yb, 0, 2)         // vextracti128 xmm2, yb, 1
				emit_b(&e.buf, 1)
				cvt_pd(&e.buf, 1, 1, is_u)
				cvt_pd(&e.buf, 2, 2, is_u)
				vop_d(&e.buf, 0x5E, 3, 1, 2)                            // vdivpd ymm3, ymm1, ymm2
				emit_vex_rr(&e.buf, 1, 1, 1, 0, 0xE6, 3, 0, 3)          // vcvttpd2dq xmm3, ymm3
				emit_vex_rr(&e.buf, 3, 1, 1, 0, 0x38, 0, 0, 3)          // vinserti128 ymm0, ymm0, xmm3, 1
				emit_b(&e.buf, 1)
				if is_u {
					// u32 quotients >= 2^31 occur iff yb == 1 and ya >= 2^31 (where ya / 1 == ya).
					vop_d_tab(&e.buf, 0x76, 4, yb, OFF_TAB_ONE_U)       // vpcmpeqd ymm4, yb, [one_u]
					vpblendvb(&e.buf, 0, 0, ya, 4)
				}
				if is_rem {
					vop38_d(&e.buf, 0x40, 1, 0, yb)                     // vpmulld ymm1, ymm0, yb
					vop_d(&e.buf, 0xFA, 0, ya, 1)                       // vpsubd ymm0, ya, ymm1
				}
				vzero(&e.buf, 4)
				vop_d(&e.buf, 0x76, 4, yb, 4)                           // vpcmpeqd ymm4, yb, 0
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				vop_d(&e.buf, 0xDF, yd, 4, 0)                           // vpandn yd, ymm4, ymm0
				commit_dst(e, dst + i, yd)
			}

		case .I_Neg:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_d(&e.buf, 0xFA, yd, 2, ya)
				commit_dst(e, dst + i, yd)
			}

		case .Bit_Not:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_d_tab(&e.buf, 0xEF, yd, ya, OFF_TAB_ALL1)
				commit_dst(e, dst + i, yd)
			}

		case .Shl, .Shr_I, .Shr_U:
			op_byte: u8 = 0x47
			if in_.op == .Shr_I {
				op_byte = 0x46
			} else if in_.op == .Shr_U {
				op_byte = 0x45
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				vop_d_tab(&e.buf, 0xDB, 3, yb, OFF_TAB_MASK31)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				vop38_d(&e.buf, op_byte, yd, ya, 3)
				commit_dst(e, dst + i, yd)
			}

		case .F_Lt, .F_Le, .F_Gt, .F_Ge, .F_Eq, .F_Ne:
			pred: u8 = 0x00
			#partial switch in_.op {
			case .F_Lt: pred = 0x11
			case .F_Le: pred = 0x12
			case .F_Gt: pred = 0x1E
			case .F_Ge: pred = 0x1D
			case .F_Eq: pred = 0x00
			case .F_Ne: pred = 0x04 // NEQ_UQ: true when unordered (NaN)
			}
			if pc_mc, fused := can_fuse_cmp_mask_cond(e, code, pc, is_init); fused {
				ya := read_slot(e, a, 1)
				yb := read_slot(e, b, 2, pin_bit(ya))
				vcmpps(&e.buf, 0, ya, yb, pred)
				finish_fused_cmp(e, dst, pc_mc, false)
				break
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				vcmpps(&e.buf, yd, ya, yb, pred)
				vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
				commit_dst(e, dst + i, yd)
			}

		case .I_Lt, .I_Le, .I_Gt, .I_Ge, .I_Eq, .I_Ne:
			if pc_mc, fused := can_fuse_cmp_mask_cond(e, code, pc, is_init); fused {
				ya := read_slot(e, a, 1)
				yb := read_slot(e, b, 2, pin_bit(ya))
				inv := false
				#partial switch in_.op {
				case .I_Gt: vop_d(&e.buf, 0x66, 0, ya, yb)
				case .I_Lt: vop_d(&e.buf, 0x66, 0, yb, ya)
				case .I_Le: vop_d(&e.buf, 0x66, 0, ya, yb); inv = true
				case .I_Ge: vop_d(&e.buf, 0x66, 0, yb, ya); inv = true
				case .I_Eq: vop_d(&e.buf, 0x76, 0, ya, yb)
				case .I_Ne: vop_d(&e.buf, 0x76, 0, ya, yb); inv = true
				}
				finish_fused_cmp(e, dst, pc_mc, inv)
				break
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				#partial switch in_.op {
				case .I_Gt:
					vop_d(&e.buf, 0x66, yd, ya, yb)
					vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
				case .I_Lt:
					vop_d(&e.buf, 0x66, yd, yb, ya)
					vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
				case .I_Le:
					vop_d(&e.buf, 0x66, yd, ya, yb)
					vop_d_tab(&e.buf, 0xDF, yd, yd, OFF_TAB_ONE_U)
				case .I_Ge:
					vop_d(&e.buf, 0x66, yd, yb, ya)
					vop_d_tab(&e.buf, 0xDF, yd, yd, OFF_TAB_ONE_U)
				case .I_Eq:
					vop_d(&e.buf, 0x76, yd, ya, yb)
					vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
				case .I_Ne:
					vop_d(&e.buf, 0x76, yd, ya, yb)
					vop_d_tab(&e.buf, 0xDF, yd, yd, OFF_TAB_ONE_U)
				}
				commit_dst(e, dst + i, yd)
			}

		case .U_Lt, .U_Le, .U_Gt, .U_Ge:
			if pc_mc, fused := can_fuse_cmp_mask_cond(e, code, pc, is_init); fused {
				ya := read_slot(e, a, 1)
				yb := read_slot(e, b, 2, pin_bit(ya))
				inv := false
				#partial switch in_.op {
				case .U_Le: vop38_d(&e.buf, 0x3B, 0, ya, yb); vop_d(&e.buf, 0x76, 0, 0, ya)
				case .U_Ge: vop38_d(&e.buf, 0x3F, 0, ya, yb); vop_d(&e.buf, 0x76, 0, 0, ya)
				case .U_Gt: vop38_d(&e.buf, 0x3B, 0, ya, yb); vop_d(&e.buf, 0x76, 0, 0, ya); inv = true
				case .U_Lt: vop38_d(&e.buf, 0x3F, 0, ya, yb); vop_d(&e.buf, 0x76, 0, 0, ya); inv = true
				}
				finish_fused_cmp(e, dst, pc_mc, inv)
				break
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				#partial switch in_.op {
				case .U_Le:
					vop38_d(&e.buf, 0x3B, 0, ya, yb) // vpminud
					vop_d(&e.buf, 0x76, 0, 0, ya)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vop_d_tab(&e.buf, 0xDB, yd, 0, OFF_TAB_ONE_U)
					commit_dst(e, dst + i, yd)
				case .U_Ge:
					vop38_d(&e.buf, 0x3F, 0, ya, yb) // vpmaxud
					vop_d(&e.buf, 0x76, 0, 0, ya)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vop_d_tab(&e.buf, 0xDB, yd, 0, OFF_TAB_ONE_U)
					commit_dst(e, dst + i, yd)
				case .U_Gt:
					vop38_d(&e.buf, 0x3B, 0, ya, yb)
					vop_d(&e.buf, 0x76, 0, 0, ya)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vop_d_tab(&e.buf, 0xDF, yd, 0, OFF_TAB_ONE_U)
					commit_dst(e, dst + i, yd)
				case .U_Lt:
					vop38_d(&e.buf, 0x3F, 0, ya, yb)
					vop_d(&e.buf, 0x76, 0, 0, ya)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vop_d_tab(&e.buf, 0xDF, yd, 0, OFF_TAB_ONE_U)
					commit_dst(e, dst + i, yd)
				}
			}

		case .B_And:
			vzero(&e.buf, 3)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				vop_d(&e.buf, 0x76, 4, ya, 3)
				vop_d(&e.buf, 0x76, 5, yb, 3)
				vop_d(&e.buf, 0xEB, 0, 4, 5)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				vop_d_tab(&e.buf, 0xDF, yd, 0, OFF_TAB_ONE_U)
				commit_dst(e, dst + i, yd)
			}

		case .B_Or:
			vzero(&e.buf, 3)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				vop_d(&e.buf, 0xEB, 0, ya, yb)
				vop_d(&e.buf, 0x76, 0, 0, 3)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				vop_d_tab(&e.buf, 0xDF, yd, 0, OFF_TAB_ONE_U)
				commit_dst(e, dst + i, yd)
			}

		case .B_Not:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_d(&e.buf, 0x76, yd, ya, 2)
				vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
				commit_dst(e, dst + i, yd)
			}

		case .I_To_F:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x5B, yd, 0, ya) // vcvtdq2ps
				commit_dst(e, dst + i, yd)
			}

		case .U_To_F:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x5B, 2, 0, ya) // lo = vcvtdq2ps(u)
				emit_vex_rm(&e.buf, 2, 1, 1, 0, 0x45, 3, ya, GPR_R14, OFF_TAB_ONE_U) // u >> 1
				vop_d_tab(&e.buf, 0xDB, 4, ya, OFF_TAB_ONE_U)   // u & 1
				vop_d(&e.buf, 0xEB, 3, 3, 4)                    // (u >> 1) | (u & 1)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x5B, 3, 0, 3)  // vcvtdq2ps
				vop_ps(&e.buf, 0x58, 3, 3, 3)                   // * 2.0
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vblendvps(&e.buf, yd, 2, 3, ya)                 // sign(u) ? hi : lo
				commit_dst(e, dst + i, yd)
			}

		case .F_To_I:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 2, 0, ya)      // vcvttps2dq
				vcmpps_tab(&e.buf, 3, ya, OFF_TAB_F2I_HI, 0x1D)      // ya >= 2^31
				vop_d(&e.buf, 0xEF, 2, 2, 3)                         // 0x80000000 -> 0x7fffffff
				vcmpps(&e.buf, 3, ya, ya, 0x07)                      // ORD_Q (!isnan)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_d(&e.buf, 0xDB, yd, 2, 3)                        // NaN -> 0
				commit_dst(e, dst + i, yd)
			}

		case .F_To_U:
			vzero(&e.buf, 5)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vop_ps(&e.buf, 0x5F, 2, ya, 5)                       // pos = (ya > 0) ? ya : 0 (NaN -> 0)
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 3, 0, 2)       // cvt0 = vcvttps2dq(pos)
				vop_ps_tab(&e.buf, 0x5C, 4, 2, OFF_TAB_F2I_HI)       // pos - 2^31
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 4, 0, 4)       // vcvttps2dq(pos - 2^31)
				vop_d_tab(&e.buf, 0xEF, 4, 4, OFF_TAB_SIGN)          // ^ 0x80000000
				vcmpps_tab(&e.buf, 6, 2, OFF_TAB_F2I_HI, 0x1D)       // pos >= 2^31
				vpblendvb(&e.buf, 0, 3, 4, 6)
				vcmpps_tab(&e.buf, 6, 2, OFF_TAB_F2U_HI, 0x1D)       // pos >= 2^32
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_d(&e.buf, 0xEB, yd, 0, 6)                        // clamp to 0xffffffff
				commit_dst(e, dst + i, yd)
			}

		case .B_To_F:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_d(&e.buf, 0x76, yd, ya, 2)
				vop_ps_tab(&e.buf, 0x55, yd, yd, OFF_TAB_ONE_F) // vandnps
				commit_dst(e, dst + i, yd)
			}

		case .F_To_B:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vcmpps(&e.buf, yd, ya, 2, 0x04) // NEQ_UQ: !(x == 0.0)
				vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
				commit_dst(e, dst + i, yd)
			}

		case .I_To_B:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_d(&e.buf, 0x76, yd, ya, 2)
				vop_d_tab(&e.buf, 0xDF, yd, yd, OFF_TAB_ONE_U)
				commit_dst(e, dst + i, yd)
			}

		case .Select:
			vzero(&e.buf, 4)
			if in_.k == 1 {
				yc := read_slot(e, c, 3)
				vop_d(&e.buf, 0x76, 3, yc, 4) // ymm3 = (c == 0)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vpblendvb(&e.buf, yd, ya, yb, 3)
					commit_dst(e, dst + i, yd)
				}
			} else {
				for i in 0 ..< n {
					yc := read_slot(e, c + i, 3)
					vop_d(&e.buf, 0x76, 3, yc, 4)
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vpblendvb(&e.buf, yd, ya, yb, 3)
					commit_dst(e, dst + i, yd)
				}
			}

		case .All:
			vzero(&e.buf, 2)
			ya0 := read_slot(e, a, 1)
			if n == 1 {
				yd := prep_dst(e, dst, 0, pin_bit(ya0))
				vop_d(&e.buf, 0x76, yd, ya0, 2)
				vop_d_tab(&e.buf, 0xDF, yd, yd, OFF_TAB_ONE_U)
				commit_dst(e, dst, yd)
			} else {
				vop_d(&e.buf, 0x76, 0, ya0, 2)
				vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				for i in 1 ..< n - 1 {
					yai := read_slot(e, a + i, 1)
					vop_d(&e.buf, 0x76, 3, yai, 2)
					vop_d(&e.buf, 0xDF, 0, 3, 0)
				}
				yal := read_slot(e, a + n - 1, 1)
				vop_d(&e.buf, 0x76, 3, yal, 2)
				yd := prep_dst(e, dst, 0, pin_bit(yal))
				vop_d(&e.buf, 0xDF, yd, 3, 0)
				commit_dst(e, dst, yd)
			}

		case .Any:
			vzero(&e.buf, 2)
			ya0 := read_slot(e, a, 0)
			if n == 1 {
				yd := prep_dst(e, dst, 0, pin_bit(ya0))
				vop_d(&e.buf, 0x76, yd, ya0, 2)
				vop_d_tab(&e.buf, 0xDF, yd, yd, OFF_TAB_ONE_U)
				commit_dst(e, dst, yd)
			} else {
				ya1 := read_slot(e, a + 1, 1, pin_bit(ya0))
				vop_d(&e.buf, 0xEB, 0, ya0, ya1)
				for i in 2 ..< n {
					yai := read_slot(e, a + i, 1)
					vop_d(&e.buf, 0xEB, 0, 0, yai)
				}
				vop_d(&e.buf, 0x76, 0, 0, 2)
				yd := prep_dst(e, dst, 0)
				vop_d_tab(&e.buf, 0xDF, yd, 0, OFF_TAB_ONE_U)
				commit_dst(e, dst, yd)
			}

		case .F_Fn1:
			bi := Builtin(in_.k)
			#partial switch bi {
			case .Abs:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vop_ps_tab(&e.buf, 0x54, yd, ya, OFF_TAB_ABS)
					commit_dst(e, dst + i, yd)
				}
			case .Sqrt:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, yd, 0, ya)
					commit_dst(e, dst + i, yd)
				}
			case .Inversesqrt:
				emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 2, 0, GPR_R14, OFF_TAB_ONE_F)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, yd, 0, ya)
					vop_ps(&e.buf, 0x5E, yd, 2, yd)
					commit_dst(e, dst + i, yd)
				}
			case .Floor, .Ceil, .Trunc, .Round_Even:
				mode: u8 = 0x08
				#partial switch bi {
				case .Floor: mode = 0x09
				case .Ceil:  mode = 0x0A
				case .Trunc: mode = 0x0B
				}
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vroundps(&e.buf, yd, ya, mode)
					commit_dst(e, dst + i, yd)
				}
			case .Round:
				// Half away from zero, as math.round: trunc(x + copysign(0.49999997, x)),
				// the f32 below 0.5, so that the sum's rounding to nearest even
				// carries x to the next integer exactly when |fract(x)| >= 0.5
				// (the same bits as trunc plus a compare and add, NaN payloads
				// included, for all 2^32 inputs).
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vop_ps_tab(&e.buf, 0x54, 2, ya, OFF_TAB_SIGN)
					vop_ps_tab(&e.buf, 0x56, 2, 2, OFF_TAB_HALF_BELOW) // copysign(0.49999997, x)
					vop_ps(&e.buf, 0x58, 2, ya, 2)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vroundps(&e.buf, yd, 2, 0x0B)
					commit_dst(e, dst + i, yd)
				}
			case .Fract:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vroundps(&e.buf, 2, ya, 0x09)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vop_ps(&e.buf, 0x5C, yd, ya, 2)
					commit_dst(e, dst + i, yd)
				}
			case .Radians:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vop_ps_tab(&e.buf, 0x59, yd, ya, OFF_TAB_DEG2RAD)
					commit_dst(e, dst + i, yd)
				}
			case .Degrees:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vop_ps_tab(&e.buf, 0x59, yd, ya, OFF_TAB_RAD2DEG)
					commit_dst(e, dst + i, yd)
				}
			case .Sign:
				vzero(&e.buf, 2)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vcmpps(&e.buf, 3, ya, 2, 0x1E) // ya > 0
					vcmpps(&e.buf, 4, ya, 2, 0x11) // ya < 0
					vop_ps_tab(&e.buf, 0x54, 3, 3, OFF_TAB_ONE_F)
					vop_ps_tab(&e.buf, 0x54, 4, 4, OFF_TAB_NEG1_F)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vop_ps(&e.buf, 0x56, yd, 3, 4)
					commit_dst(e, dst + i, yd)
				}
			case .Isnan:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vcmpps(&e.buf, yd, ya, ya, 0x03) // UNORD_Q
					vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
					commit_dst(e, dst + i, yd)
				}
			case .Isinf:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vop_ps_tab(&e.buf, 0x54, yd, ya, OFF_TAB_ABS)
					vop_d_tab(&e.buf, 0x76, yd, yd, OFF_TAB_INF)
					vop_d_tab(&e.buf, 0xDB, yd, yd, OFF_TAB_ONE_U)
					commit_dst(e, dst + i, yd)
				}
			case .Float_Bits_To_Int, .Float_Bits_To_Uint:
				// canon_bits: NaN lanes (unordered with themselves) to 0x7fc00000.
				emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 3, 0, GPR_R14, OFF_TAB_CANON)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vcmpps(&e.buf, 2, ya, ya, 0x03) // UNORD_Q
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vblendvps(&e.buf, yd, ya, 3, 2)
					commit_dst(e, dst + i, yd)
				}
			case:
				for i in 0 ..< n {
					emit_math_call(e, &code[pc], i)
				}
			}

		case .F_Fn2:
			bi := Builtin(in_.k)
			#partial switch bi {
			case .Min:
				// (y < x) ? y : x -> vvvv = y, rm = x so min(NaN, u0) == NaN and min(u0, NaN) == u0.
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vop_ps(&e.buf, 0x5D, yd, yb, ya)
					commit_dst(e, dst + i, yd)
				}
			case .Max:
				// (y > x) ? y : x -> vvvv = y, rm = x.
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vop_ps(&e.buf, 0x5F, yd, yb, ya)
					commit_dst(e, dst + i, yd)
				}
			case .Step:
				// (y < x) ? 0.0 : 1.0
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vcmpps(&e.buf, yd, yb, ya, 0x11)
					vop_ps_tab(&e.buf, 0x55, yd, yd, OFF_TAB_ONE_F)
					commit_dst(e, dst + i, yd)
				}
			case .Mod:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					vop_ps(&e.buf, 0x5E, 3, ya, yb)
					vroundps(&e.buf, 3, 3, 0x09)
					vop_ps(&e.buf, 0x59, 3, yb, 3)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
					vop_ps(&e.buf, 0x5C, yd, ya, 3)
					commit_dst(e, dst + i, yd)
				}
			case:
				for i in 0 ..< n {
					emit_math_call(e, &code[pc], i)
				}
			}

		case .F_Fn3:
			bi := Builtin(in_.k)
			#partial switch bi {
			case .Clamp:
				// clamp_f32 (vm_run.odin): GLSL's min(max(x, lo), hi) with
				// max(x, y) = `y if x < y else x`, so a bound equal to x (a
				// zero of the other sign) leaves x; a NaN operand gives the other.
				for i in 0 ..< n {
					yx := read_slot(e, a + i, 1)
					ylo := read_slot(e, b + i, 2, pin_bit(yx))
					yhi := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(ylo))
					vop_ps(&e.buf, 0x5F, 4, ylo, yx) // lo > x ? lo : x
					vcmpps(&e.buf, 5, yx, yx, 0x03)
					vblendvps(&e.buf, 0, 4, ylo, 5)  // x NaN: lo
					yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(ylo) | pin_bit(yhi))
					if e.num_const[b + i] {
						// t is NaN only where x and lo both are, so never
						// with a constant lo: hi < t ? hi : t is all.
						vop_ps(&e.buf, 0x5D, yd, yhi, 0)
					} else {
						vop_ps(&e.buf, 0x5D, 4, yhi, 0)
						vcmpps(&e.buf, 5, 0, 0, 0x03)
						vblendvps(&e.buf, yd, 4, yhi, 5) // t NaN: hi
					}
					commit_dst(e, dst + i, yd)
				}
			case .Mix:
				// x * (1 - z) + y * z
				emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 5, 0, GPR_R14, OFF_TAB_ONE_F)
				for i in 0 ..< n {
					yx := read_slot(e, a + i, 1)
					yy := read_slot(e, b + i, 2, pin_bit(yx))
					yz := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(yy))
					vop_ps(&e.buf, 0x5C, 4, 5, yz)
					vop_ps(&e.buf, 0x59, 4, yx, 4)
					vop_ps(&e.buf, 0x59, 6, yy, yz)
					yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(yy) | pin_bit(yz))
					vop_ps(&e.buf, 0x58, yd, 4, 6)
					commit_dst(e, dst + i, yd)
				}
			case .Smoothstep:
				vzero(&e.buf, 5)
				emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 6, 0, GPR_R14, OFF_TAB_THREE_F)
				for i in 0 ..< n {
					yx := read_slot(e, a + i, 1)
					yy := read_slot(e, b + i, 2, pin_bit(yx))
					yz := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(yy))
					vop_ps(&e.buf, 0x5C, 0, yz, yx)
					vop_ps(&e.buf, 0x5C, 4, yy, yx)
					vop_ps(&e.buf, 0x5E, 0, 0, 4)
					vop_ps(&e.buf, 0x5F, 0, 0, 5)                 // clamp(NaN, 0, 1) -> 0
					vop_ps_tab(&e.buf, 0x5D, 0, 0, OFF_TAB_ONE_F)
					vop_ps_tab(&e.buf, 0x59, 4, 0, OFF_TAB_TWO_F) // 2 * t
					vop_ps(&e.buf, 0x5C, 4, 6, 4)                 // 3 - 2 * t
					vop_ps(&e.buf, 0x59, 0, 0, 0)                 // t * t
					yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(yy) | pin_bit(yz))
					vop_ps(&e.buf, 0x59, yd, 0, 4)
					commit_dst(e, dst + i, yd)
				}
			}

		case .I_Fn1:
			if Builtin(in_.k) == .Abs {
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x1E, yd, 0, ya) // vpabsd
					commit_dst(e, dst + i, yd)
				}
			} else {
				vzero(&e.buf, 2)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vop_d(&e.buf, 0x66, 3, ya, 2) // pos = (x > 0) ? -1 : 0
					vop_d(&e.buf, 0x66, 4, 2, ya) // neg = (0 > x) ? -1 : 0
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vop_d(&e.buf, 0xFA, yd, 4, 3) // neg - pos = sign(x)
					commit_dst(e, dst + i, yd)
				}
			}

		case .I_Fn2, .U_Fn2:
			op_byte: u8 = 0x39
			if in_.op == .I_Fn2 {
				op_byte = 0x39 if Builtin(in_.k) == .Min else 0x3D
			} else {
				op_byte = 0x3B if Builtin(in_.k) == .Min else 0x3F
			}
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				yd := prep_dst(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
				vop38_d(&e.buf, op_byte, yd, ya, yb)
				commit_dst(e, dst + i, yd)
			}

		case .I_Fn3, .U_Fn3:
			max_op: u8 = 0x3D if in_.op == .I_Fn3 else 0x3F
			min_op: u8 = 0x39 if in_.op == .I_Fn3 else 0x3B
			for i in 0 ..< n {
				yx := read_slot(e, a + i, 1)
				ylo := read_slot(e, b + i, 2, pin_bit(yx))
				yhi := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(ylo))
				vop38_d(&e.buf, max_op, 0, yx, ylo)
				yd := prep_dst(e, dst + i, 0, pin_bit(yx) | pin_bit(ylo) | pin_bit(yhi))
				vop38_d(&e.buf, min_op, yd, 0, yhi)
				commit_dst(e, dst + i, yd)
			}

		case .Dot:
			ya0 := read_slot(e, a, 1)
			yb0 := read_slot(e, b, 2, pin_bit(ya0))
			if n == 1 {
				yd := prep_dst(e, dst, 0, pin_bit(ya0) | pin_bit(yb0))
				vop_ps(&e.buf, 0x59, yd, ya0, yb0)
				commit_dst(e, dst, yd)
			} else {
				vop_ps(&e.buf, 0x59, 0, ya0, yb0)
				for i in 1 ..< n - 1 {
					yai := read_slot(e, a + i, 1)
					ybi := read_slot(e, b + i, 2, pin_bit(yai))
					vop_ps(&e.buf, 0x59, 3, yai, ybi)
					vop_ps(&e.buf, 0x58, 0, 0, 3)
				}
				yal := read_slot(e, a + n - 1, 1)
				ybl := read_slot(e, b + n - 1, 2, pin_bit(yal))
				vop_ps(&e.buf, 0x59, 3, yal, ybl)
				yd := prep_dst(e, dst, 0, pin_bit(yal) | pin_bit(ybl))
				vop_ps(&e.buf, 0x58, yd, 0, 3)
				commit_dst(e, dst, yd)
			}

		case .Length:
			ya0 := read_slot(e, a, 1)
			vop_ps(&e.buf, 0x59, 0, ya0, ya0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				vop_ps(&e.buf, 0x59, 2, yai, yai)
				vop_ps(&e.buf, 0x58, 0, 0, 2)
			}
			yd := prep_dst(e, dst, 0)
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, yd, 0, 0)
			commit_dst(e, dst, yd)

		case .Distance:
			ya0 := read_slot(e, a, 1)
			yb0 := read_slot(e, b, 2, pin_bit(ya0))
			vop_ps(&e.buf, 0x5C, 0, ya0, yb0)
			vop_ps(&e.buf, 0x59, 0, 0, 0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				ybi := read_slot(e, b + i, 2, pin_bit(yai))
				vop_ps(&e.buf, 0x5C, 3, yai, ybi)
				vop_ps(&e.buf, 0x59, 3, 3, 3)
				vop_ps(&e.buf, 0x58, 0, 0, 3)
			}
			yd := prep_dst(e, dst, 0)
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, yd, 0, 0)
			commit_dst(e, dst, yd)

		case .Normalize:
			ya0 := read_slot(e, a, 0)
			vop_ps(&e.buf, 0x59, 4, ya0, ya0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 0)
				vop_ps(&e.buf, 0x59, 5, yai, yai)
				vop_ps(&e.buf, 0x58, 4, 4, 5)
			}
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, 4, 0, 4)
			emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 5, 0, GPR_R14, OFF_TAB_ONE_F)
			vop_ps(&e.buf, 0x5E, 4, 5, 4)
			for i in 0 ..< n {
				yai := read_slot(e, a + i, 1)
				yd := prep_dst(e, dst + i, 0, pin_bit(yai))
				vop_ps(&e.buf, 0x59, yd, yai, 4)
				commit_dst(e, dst + i, yd)
			}

		case .Cross:
			e.exec_ymask_valid = false
			x0 := read_slot(e, a, 0);     if x0 != 0 {vmovaps_rr(&e.buf, 0, x0)}
			x1 := read_slot(e, a + 1, 1); if x1 != 1 {vmovaps_rr(&e.buf, 1, x1)}
			x2 := read_slot(e, a + 2, 2); if x2 != 2 {vmovaps_rr(&e.buf, 2, x2)}
			y0 := read_slot(e, b, 3);     if y0 != 3 {vmovaps_rr(&e.buf, 3, y0)}
			y1 := read_slot(e, b + 1, 4); if y1 != 4 {vmovaps_rr(&e.buf, 4, y1)}
			y2 := read_slot(e, b + 2, 5); if y2 != 5 {vmovaps_rr(&e.buf, 5, y2)}

			vop_ps(&e.buf, 0x59, 6, 1, 5)
			vop_ps(&e.buf, 0x59, 7, 4, 2)
			yd0 := prep_dst(e, dst, 6)
			vop_ps(&e.buf, 0x5C, yd0, 6, 7)
			commit_dst(e, dst, yd0)

			vop_ps(&e.buf, 0x59, 6, 2, 3)
			vop_ps(&e.buf, 0x59, 7, 5, 0)
			yd1 := prep_dst(e, dst + 1, 6)
			vop_ps(&e.buf, 0x5C, yd1, 6, 7)
			commit_dst(e, dst + 1, yd1)

			vop_ps(&e.buf, 0x59, 6, 0, 4)
			vop_ps(&e.buf, 0x59, 7, 3, 1)
			yd2 := prep_dst(e, dst + 2, 6)
			vop_ps(&e.buf, 0x5C, yd2, 6, 7)
			commit_dst(e, dst + 2, yd2)

		case .Reflect:
			ya0 := read_slot(e, a, 1)
			yb0 := read_slot(e, b, 2, pin_bit(ya0))
			vop_ps(&e.buf, 0x59, 4, ya0, yb0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				ybi := read_slot(e, b + i, 2, pin_bit(yai))
				vop_ps(&e.buf, 0x59, 5, yai, ybi)
				vop_ps(&e.buf, 0x58, 4, 4, 5)
			}
			vop_ps_tab(&e.buf, 0x59, 4, 4, OFF_TAB_TWO_F) // 2 * dot(I, N)
			for i in 0 ..< n {
				yai := read_slot(e, a + i, 1)
				ybi := read_slot(e, b + i, 2, pin_bit(yai))
				vop_ps(&e.buf, 0x59, 5, 4, ybi)
				yd := prep_dst(e, dst + i, 0, pin_bit(yai) | pin_bit(ybi))
				vop_ps(&e.buf, 0x5C, yd, yai, 5)
				commit_dst(e, dst + i, yd)
			}

		case .Faceforward, .Refract:
			emit_helper_call(&e.buf, jit_help_geom, &code[pc])

		case .Mat_Mul:
			rows, inner, cols := n, i32(in_.k), in_.d
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					out_s := dst + col * rows + row
					ya0 := read_slot(e, a + row, 1)
					yb0 := read_slot(e, b + col * inner, 2, pin_bit(ya0))
					if inner == 1 {
						yd := prep_dst(e, out_s, 0, pin_bit(ya0) | pin_bit(yb0))
						vop_ps(&e.buf, 0x59, yd, ya0, yb0)
						commit_dst(e, out_s, yd)
					} else {
						vop_ps(&e.buf, 0x59, 0, ya0, yb0)
						for k in 1 ..< inner - 1 {
							yak := read_slot(e, a + k * rows + row, 1)
							ybk := read_slot(e, b + col * inner + k, 2, pin_bit(yak))
							vop_ps(&e.buf, 0x59, 3, yak, ybk)
							vop_ps(&e.buf, 0x58, 0, 0, 3)
						}
						kl := inner - 1
						yal := read_slot(e, a + kl * rows + row, 1)
						ybl := read_slot(e, b + col * inner + kl, 2, pin_bit(yal))
						vop_ps(&e.buf, 0x59, 3, yal, ybl)
						yd := prep_dst(e, out_s, 0, pin_bit(yal) | pin_bit(ybl))
						vop_ps(&e.buf, 0x58, yd, 0, 3)
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
					vop_ps(&e.buf, 0x59, yd, ya0, yb0)
					commit_dst(e, dst + col, yd)
				} else {
					vop_ps(&e.buf, 0x59, 0, ya0, yb0)
					for row in 1 ..< rows - 1 {
						yar := read_slot(e, a + row, 1)
						ybr := read_slot(e, b + col * rows + row, 2, pin_bit(yar))
						vop_ps(&e.buf, 0x59, 3, yar, ybr)
						vop_ps(&e.buf, 0x58, 0, 0, 3)
					}
					rl := rows - 1
					yal := read_slot(e, a + rl, 1)
					ybl := read_slot(e, b + col * rows + rl, 2, pin_bit(yal))
					vop_ps(&e.buf, 0x59, 3, yal, ybl)
					yd := prep_dst(e, dst + col, 0, pin_bit(yal) | pin_bit(ybl))
					vop_ps(&e.buf, 0x58, yd, 0, 3)
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
					vop_ps(&e.buf, 0x59, yd, yar, ybc)
					commit_dst(e, dst + col * rows + row, yd)
				}
			}

		case .Determinant, .Inverse:
			emit_helper_call(&e.buf, jit_help_mat, &code[pc])

		case .Deriv_X:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x16, 2, 0, ya) // vmovshdup
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x12, 3, 0, ya) // vmovsldup
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_ps(&e.buf, 0x5C, yd, 2, 3)
				commit_dst(e, dst + i, yd)
			}

		case .Deriv_Y:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0xC6, 2, ya, ya); emit_b(&e.buf, 0xEE)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0xC6, 3, ya, ya); emit_b(&e.buf, 0x44)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_ps(&e.buf, 0x5C, yd, 2, 3)
				commit_dst(e, dst + i, yd)
			}

		case .Fwidth:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x16, 2, 0, ya)
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x12, 3, 0, ya)
				vop_ps(&e.buf, 0x5C, 2, 2, 3)
				vop_ps_tab(&e.buf, 0x54, 2, 2, OFF_TAB_ABS)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0xC6, 3, ya, ya); emit_b(&e.buf, 0xEE)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0xC6, 4, ya, ya); emit_b(&e.buf, 0x44)
				vop_ps(&e.buf, 0x5C, 3, 3, 4)
				vop_ps_tab(&e.buf, 0x54, 3, 3, OFF_TAB_ABS)
				yd := prep_dst(e, dst + i, 0, pin_bit(ya))
				vop_ps(&e.buf, 0x58, yd, 2, 3)
				commit_dst(e, dst + i, yd)
			}

		case .Texture, .Texture_Bias, .Texel_Fetch:
			e.exec_ymask_valid = false
			comp_mask := e.tex_mask[pc]
			if comp_mask != 0 {
				// The bound path's inputs: the coordinates and the lod.
				ya0 := read_slot(e, a, 0)
				ya1 := read_slot(e, a + 1, 1, pin_bit(ya0))
				if ya0 != 0 {
					vmovaps_rr(&e.buf, 0, ya0)
				}
				if ya1 != 1 {
					vmovaps_rr(&e.buf, 1, ya1)
				}
				if in_.op == .Texel_Fetch {
					yb := read_slot(e, b, 2, pin_bit(ya0) | pin_bit(ya1))
					if yb != 2 {
						vmovaps_rr(&e.buf, 2, yb)
					}
				} else {
					// The lod in memory, for the stub's paths that read it.
					for &cs, idx in e.cache {
						if cs.slot == b && cs.dirty {
							vmovaps_store(&e.buf, b, u8(8 + idx))
							cs.dirty = false
						}
					}
				}
				if in_.n == 3 {
					// A sampler2DArray's layer coordinate in memory, for the stub.
					for &cs, idx in e.cache {
						if cs.slot == a + 2 && cs.dirty {
							vmovaps_store(&e.buf, a + 2, u8(8 + idx))
							cs.dirty = false
						}
					}
				}
			}
			chan_disp := i32(in_.k) * size_of(Chan_Img) - JIT_CHAN_BYTES
			emit_bytes(&e.buf, 0x48, 0x8B) // mov rsi, [rbx + chan_disp + OFF_CHAN_PTR]
			emit_modrm_mem(&e.buf, GPR_RSI, GPR_RBX, chan_disp + OFF_CHAN_PTR)
			emit_bytes(&e.buf, 0x48, 0x85, 0xF6) // test rsi, rsi
			emit_bytes(&e.buf, 0x0F, 0x85)       // jnz bound_tex
			bound_patch := len(e.buf)
			emit_i32(&e.buf, 0)

			// Unbound callback path: flush any dirty cached input slots to [rbx]
			// and save/restore only the ymm8..15 registers still live after `pc`.
			for cs, idx in e.cache {
				if cs.slot < 0 || !cs.dirty {
					continue
				}
				for r in rw[pc].reads {
					if cs.slot >= r.start && cs.slot < r.start + r.n {
						vmovaps_store(&e.buf, cs.slot, u8(8 + idx))
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
				if (cs.slot >= dst && cs.slot < dst + 4) || cs.last_use <= e.pc {
					cs = {slot = -1, last_use = -1, dirty = false, stamp = 0}
				} else {
					save_regs[n_save] = u8(8 + idx)
					n_save += 1
				}
			}
			fn := jit_help_texel_fetch if in_.op == .Texel_Fetch else jit_help_texture
			if p.stage == .Compute {
				// No quads: the callback runs for the running lanes only, as in
				// the interpreter (the helpers call it for ctx.lanes, which
				// nothing else reads after the prologue in compute).
				emit_bytes(&e.buf, 0x44, 0x88) // mov byte [rbp + lanes], r13b
				emit_modrm_mem(&e.buf, GPR_R13, GPR_RBP, OFF_CTX_LANES)
			}
			if n_save == 0 {
				emit_helper_call(&e.buf, fn, &code[pc])
			} else if n_save <= 2 {
				frame := u8(n_save * 32)
				emit_bytes(&e.buf, 0x48, 0x83, 0xEC, frame)
				for i in 0 ..< n_save {
					emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x11, save_regs[i], 0, GPR_RSP, i32(i * 32))
				}
				emit_helper_call(&e.buf, fn, &code[pc])
				for i in 0 ..< n_save {
					emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x10, save_regs[i], 0, GPR_RSP, i32(i * 32))
				}
				emit_bytes(&e.buf, 0x48, 0x83, 0xC4, frame)
			} else {
				emit_bytes(&e.buf, 0x48, 0xBE) // movabs rsi, &code[pc]
				emit_u64(&e.buf, u64(uintptr(&code[pc])))
				emit_bytes(&e.buf, 0x48, 0xB8) // movabs rax, fn
				emit_u64(&e.buf, u64(uintptr(rawptr(fn))))
				emit_b(&e.buf, 0xE8) // call helper_ymm_sub
				append(&e.helper_ymm_calls, len(e.buf))
				emit_i32(&e.buf, 0)
			}
			if comp_mask == 0 {
				patch_i32(e.buf[:], bound_patch, i32(len(e.buf) - (bound_patch + 4)))
			} else {
				append(&e.bound_tex, Bound_Tex_Stub {
					bound_patch = bound_patch,
					ret_off     = len(e.buf),
					pc          = i32(pc),
					comp_mask   = comp_mask,
				})
			}

		case .Texture_Size:
			emit_helper_call(&e.buf, jit_help_texture_size, &code[pc])

		case .Load_Buf:
			e.exec_ymask_valid = false
			k_buf := i32(in_.k)
			if k_buf >= 4 {
				emit_helper_call(&e.buf, jit_help_load_buf, &code[pc])
				break
			}
			yidx := read_slot(e, a, 6)
			pins := pin_bit(yidx)
			yd: [4]u8
			for i in 0 ..< n {
				yd[i] = prep_dst(e, dst + i, u8(i), pins)
				pins |= pin_bit(yd[i])
			}
			emit_bytes(&e.buf, 0x48, 0x8B) // mov rcx, [rbp + OFF_CTX_BUF_LEN + k*8]
			emit_modrm_mem(&e.buf, GPR_RCX, GPR_RBP, OFF_CTX_BUF_LEN + k_buf * 8)
			emit_bytes(&e.buf, 0x48, 0x8B) // mov rsi, [rbp + OFF_CTX_BUF_PTR + k*8]
			emit_modrm_mem(&e.buf, GPR_RSI, GPR_RBP, OFF_CTX_BUF_PTR + k_buf * 8)

			// Check if all 8 lanes share lane 0's index (99% of buffer reads in ovg).
			emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 4, 0, yidx) // vpbroadcastd ymm4, xmm(yidx)
			vop_d(&e.buf, 0x76, 4, 4, yidx)                   // vpcmpeqd
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x50, GPR_RDX, 0, 4) // vmovmskps edx, ymm4
			emit_bytes(&e.buf, 0x81, 0xFA, 0xFF, 0x00, 0x00, 0x00) // cmp edx, 0xFF
			emit_bytes(&e.buf, 0x0F, 0x85) // jne divergent
			div_patch := len(e.buf)
			emit_i32(&e.buf, 0)

			// Uniform fast path: 64-bit unsigned bounds-checked vbroadcastss into yd[0..n-1].
			emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x7E, yidx, 0, GPR_RAX) // vmovd eax, xmm(yidx)
			// A std430 element is at least one word (declare_buffer), so the
			// stride is at least 1, as in emit_store_buf.
			stride := u32(in_.b)
			off := u32(in_.c)
			if stride != 1 {
				emit_bytes(&e.buf, 0x69, 0xC0) // imul eax, eax, stride
				emit_u32(&e.buf, stride)
			}
			if off != 0 {
				emit_b(&e.buf, 0x05) // add eax, off
				emit_u32(&e.buf, off)
			}
			// ctx.buf_len is clamped at 2^32, so in 64 bits `rax + (n - 1) < rcx` holds iff
			// `eax + u32(n - 1)` does not wrap 32 bits and all `n` words lie within `buf_len`.
			if n == 1 {
				emit_bytes(&e.buf, 0x48, 0x39, 0xC8) // cmp rax, rcx
			} else {
				emit_bytes(&e.buf, 0x48, 0x8D) // lea rdx, [rax + (n - 1)]
				emit_modrm_mem(&e.buf, GPR_RDX, GPR_RAX, n - 1)
				emit_bytes(&e.buf, 0x48, 0x39, 0xCA) // cmp rdx, rcx
			}
			emit_bytes(&e.buf, 0x0F, 0x83) // jae unif_oob_stub
			unif_patch := len(e.buf)
			emit_i32(&e.buf, 0)
			for i in 0 ..< n {
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x18, yd[i], 0, GPR_RSI, GPR_RAX, 2, i * 4) // vbroadcastss yd[i], [rsi + rax*4 + i*4]
			}
			emit_b(&e.buf, 0xE9) // jmp done
			done_patch := len(e.buf)
			emit_i32(&e.buf, 0)

			// Divergent path: hardware AVX2 vpgatherdd (< 2^31 words inline; > 2^31 words out of line).
			patch_i32(e.buf[:], div_patch, i32(len(e.buf) - (div_patch + 4)))
			if stride == 1 {
				vmovaps_rr(&e.buf, 4, yidx)
			} else {
				emit_b(&e.buf, 0xB8) // mov eax, stride
				emit_u32(&e.buf, stride)
				emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x6E, 4, 0, GPR_RAX) // vmovd xmm4, eax
				emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 4, 0, 4)       // vpbroadcastd ymm4, xmm4
				vop38_d(&e.buf, 0x40, 4, yidx, 4)                    // vpmulld
			}
			if off != 0 {
				emit_b(&e.buf, 0xB8)
				emit_u32(&e.buf, off)
				emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x6E, 5, 0, GPR_RAX)
				emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 5, 0, 5)
				vop_d(&e.buf, 0xFE, 4, 4, 5)
			}
			// vpgatherdd sign-extends 32-bit indices; if eff_len > 2^31, jump to out-of-line split gather.
			emit_b(&e.buf, 0xBA) // mov edx, 0x80000000 (zero-extended into rdx)
			emit_u32(&e.buf, 0x8000_0000)
			emit_bytes(&e.buf, 0x48, 0x39, 0xD1) // cmp rcx, rdx
			emit_bytes(&e.buf, 0x0F, 0x87)       // ja big_buf_stub
			big_patch := len(e.buf)
			emit_i32(&e.buf, 0)

			emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x6E, 5, 0, GPR_RCX)
			emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 5, 0, 5)
			for i in 0 ..< n {
				if i > 0 {
					vop_d_tab(&e.buf, 0xFE, 4, 4, OFF_TAB_ONE_U)
				}
				vop38_d(&e.buf, 0x3B, 7, 4, 5)              // vpminud ymm7, w, len
				vop_d(&e.buf, 0x76, 7, 7, 5)                // ~0 where w >= len
				vop_d_tab(&e.buf, 0xEF, 7, 7, OFF_TAB_ALL1) // ~0 where w < len (so bit 31 of w is 0)
				vop_d(&e.buf, 0xDB, 6, 4, 7)                // non-negative safe index
				vzero(&e.buf, yd[i])
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, yd[i], 7, GPR_RSI, 6, 2, 0) // vpgatherdd yd[i], [rsi + ymm6*4], ymm7
			}

			ret_off := len(e.buf)
			patch_i32(e.buf[:], done_patch, i32(ret_off - (done_patch + 4)))
			append(&e.load_bufs, Load_Buf_Stub {
				unif_patch = unif_patch,
				big_patch  = big_patch,
				ret_off    = ret_off,
				pc         = i32(pc),
				yd         = yd,
			})
			for i in 0 ..< n {
				commit_dst(e, dst + i, yd[i])
			}

		case .Mask_Copy:
			emit_copy_mask(e, dst, a)

		case .Mask_Cond:
			if e.fused_mc_pc == i32(pc) {
				e.fused_mc_pc = -1
				if (in_.k == 1) != e.fused_mc_inv {
					emit_bytes(&e.buf, 0xF7, 0xD1) // not ecx
				}
			} else {
				yb := read_slot(e, b, 1)
				vzero(&e.buf, 0)
				vop_d(&e.buf, 0x76, 0, yb, 0)                         // ~0 where b == 0
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x50, GPR_RCX, 0, 0)  // vmovmskps ecx, ymm0
				if in_.k == 0 {
					emit_bytes(&e.buf, 0x81, 0xF1, 0xFF, 0x00, 0x00, 0x00) // xor ecx, 0xFF
				}
			}
			if dst == MASK_EXEC && is_mask_eq_exec(e, a) {
				emit_bytes(&e.buf, 0x41, 0x21, 0xCD) // and r13d, ecx
			} else {
				emit_load_mask_eax(e, a)
				emit_bytes(&e.buf, 0x21, 0xC8) // and eax, ecx
				emit_store_mask_al(&e.buf, dst)
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
				emit_copy_mask(e, dst, a)
			} else {
				k0 := kills[0]
				if is_mask_eq_exec(e, k0) {
					emit_bytes(&e.buf, 0x44, 0x89, 0xE9) // mov ecx, r13d
				} else {
					emit_bytes(&e.buf, 0x41, 0x0F, 0xB6) // movzx ecx, byte [r12 + k0]
					emit_modrm_mem(&e.buf, GPR_RCX, GPR_R12, k0)
				}
				for km in kills[1:n_kills] {
					emit_load_mask_eax(e, km)
					emit_bytes(&e.buf, 0x09, 0xC1) // or ecx, eax
				}
				emit_bytes(&e.buf, 0xF7, 0xD1) // not ecx
				if dst == MASK_EXEC && is_mask_eq_exec(e, a) {
					emit_bytes(&e.buf, 0x41, 0x21, 0xCD) // and r13d, ecx
				} else {
					emit_load_mask_eax(e, a)
					emit_bytes(&e.buf, 0x21, 0xC8) // and eax, ecx
					emit_store_mask_al(&e.buf, dst)
				}
				invalidate_mask_write(e, dst)
			}

		case .Mask_Or:
			emit_load_mask_eax(e, b)
			emit_bytes(&e.buf, 0x89, 0xC1) // mov ecx, eax
			emit_load_mask_eax(e, a)
			emit_bytes(&e.buf, 0x09, 0xC8) // or eax, ecx
			emit_store_mask_al(&e.buf, dst)
			invalidate_mask_write(e, dst)

		case .Mask_Clear:
			if dst == MASK_EXEC {
				emit_bytes(&e.buf, 0x45, 0x31, 0xED) // xor r13d, r13d
			} else {
				emit_bytes(&e.buf, 0x41, 0xC6)
				emit_modrm_mem(&e.buf, 0, GPR_R12, dst)
				emit_b(&e.buf, 0)
			}
			invalidate_mask_write(e, dst)

		case .Mask_Kill:
			if dst != MASK_EXEC {
				emit_load_mask_eax(e, dst)
				emit_bytes(&e.buf, 0x44, 0x09, 0xE8) // or eax, r13d
				emit_store_mask_al(&e.buf, dst)
			}
			emit_bytes(&e.buf, 0x45, 0x31, 0xED) // xor r13d, r13d
			invalidate_mask_write(e, dst)
			invalidate_mask_write(e, MASK_EXEC)
			if dst == MASK_DISCARD && !is_init {
				emit_bytes(&e.buf, 0xF7, 0xD0) // not eax
				emit_bytes(&e.buf, 0x0F, 0xB6) // movzx ecx, byte [rbp + OFF_CTX_LANES]
				emit_modrm_mem(&e.buf, GPR_RCX, GPR_RBP, OFF_CTX_LANES)
				emit_bytes(&e.buf, 0x85, 0xC8) // test eax, ecx
				emit_bytes(&e.buf, 0x0F, 0x84) // jz epilogue
				append(&e.epilogues, len(e.buf))
				emit_i32(&e.buf, 0)
			}

		case .Jump:
			emit_b(&e.buf, 0xE9)
			append(&e.patches, Jump_Patch{len(e.buf), a})
			emit_i32(&e.buf, 0)

		case .Jump_If_None:
			emit_bytes(&e.buf, 0x45, 0x85, 0xED) // test r13d, r13d
			emit_bytes(&e.buf, 0x0F, 0x84)       // jz target
			append(&e.patches, Jump_Patch{len(e.buf), a})
			emit_i32(&e.buf, 0)

		case .Store_Buf:
			if in_.k >= 4 {
				emit_helper_call(&e.buf, jit_help_store_buf, &code[pc])
			} else {
				emit_store_buf(e, in_)
			}
		case .Image_Load:
			if p.image_formats[in_.k] == .Rgba16f {
				emit_helper_call(&e.buf, jit_help_image_load, &code[pc])
			} else {
				emit_image_load(e, in_, p.image_formats[in_.k])
			}
		case .Image_Store:
			if p.image_formats[in_.k] == .Rgba16f {
				emit_helper_call(&e.buf, jit_help_image_store, &code[pc])
			} else {
				emit_image_store(e, in_, p.image_formats[in_.k])
			}
		case .Image_Size:
			emit_helper_call(&e.buf, jit_help_image_size, &code[pc])
		case .Load_Shared:
			emit_helper_call(&e.buf, jit_help_load_shared, &code[pc])
		case .Store_Shared:
			emit_helper_call(&e.buf, jit_help_store_shared, &code[pc])
		case .Atomic_Shared:
			emit_helper_call(&e.buf, jit_help_atomic_shared, &code[pc])
		case .Atomic_Buf:
			emit_helper_call(&e.buf, jit_help_atomic_buf, &code[pc])
		case .Barrier:
			// Only compiled when one batch is the whole workgroup (jit_compute_eligible),
			// where every lane reaches it together: nothing to wait for.
		}
	}

	pc_off[n_code] = len(e.buf)
	epilogue_off := len(e.buf)
	e.epilogue_off = epilogue_off

	// Epilogue: vzeroupper, restore callee-saved GPRs.
	emit_bytes(&e.buf, 0xC5, 0xF8, 0x77) // vzeroupper
	emit_bytes(&e.buf, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3)

	for jp in e.patches {
		rel := i32(pc_off[jp.target_pc] - (jp.patch_pos + 4))
		patch_i32(e.buf[:], jp.patch_pos, rel)
	}
	for ep in e.epilogues {
		rel := i32(epilogue_off - (ep + 4))
		patch_i32(e.buf[:], ep, rel)
	}
	for &stub in e.bound_tex {
		stub.stub_off = len(e.buf)
		patch_i32(e.buf[:], stub.bound_patch, i32(len(e.buf) - (stub.bound_patch + 4)))
		in_ := code[stub.pc]
		chan_disp := i32(in_.k) * size_of(Chan_Img) - JIT_CHAN_BYTES
		emit_bytes(&e.buf, 0x48, 0x8D) // lea rdi, [rbx + chan_disp]
		emit_modrm_mem(&e.buf, GPR_RDI, GPR_RBX, chan_disp)
		if in_.op == .Texel_Fetch && in_.n == 3 {
			// A sampler2DArray's: its layer p.z, outside [0, n_layers) as a
			// level outside the levels (all ones).
			vmovaps_load(&e.buf, 3, in_.a + 2)
			vpbroadcastd_mem(&e.buf, 5, GPR_RDI, OFF_CI_NLAYERS)
			vop_d_tab(&e.buf, 0xFA, 5, 5, OFF_TAB_ONE_U)
			vop38_d(&e.buf, 0x3B, 6, 3, 5) // vpminud
			vop_d(&e.buf, 0x76, 6, 6, 3)   // in range
			vop_d(&e.buf, 0x76, 7, 7, 7)
			vop_d(&e.buf, 0xEF, 6, 6, 7)
			vop_d(&e.buf, 0xEB, 2, 2, 6)
			emit_b(&e.buf, 0xBA) // mov edx, live components | flip y << 8
			emit_u32(&e.buf, u32(stub.comp_mask) | u32(in_.d & 1) << 8)
			x_call(&e.buf, &e.fetch_calls)
		} else if in_.op == .Texel_Fetch {
			// A 2D image's: one level inline (rsi = its texels).
			emit_chan_fetch_one(&e.buf, &e.fetch_calls, stub.comp_mask, (in_.d & 1) != 0, e.f16c)
		} else {
			// F32, one level, Nearest, both axes clamped: inline (rsi = its texels).
			x_cmp_b(&e.buf, GPR_RDI, OFF_CI_NEAR_ROUTE, 0)
			general := x_jump(&e.buf, JCC_JNE)
			l0 := chan_disp + OFF_CI_LEVELS
			if e.fma {
				emit_chan_nearest_xy_fma(&e.buf, GPR_RBX, chan_disp)
			} else {
				emit_chan_nearest_xy(&e.buf, GPR_RBX, l0)
			}
			vpbroadcastd_mem(&e.buf, 6, GPR_RBX, l0 + OFF_LV_PITCH)
			vop38_d(&e.buf, 0x40, 1, 1, 6)
			vop_d(&e.buf, 0xFE, 4, 1, 0)
			emit_bytes(&e.buf, 0x0F, 0xB6) // movzx ecx, byte [rbx + shift]
			emit_modrm_mem(&e.buf, GPR_RCX, GPR_RBX, chan_disp + OFF_CI_SHIFT)
			x_bcast_gpr(&e.buf, 6, GPR_RCX)
			vop38_d(&e.buf, 0x47, 4, 4, 6)
			emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 5, 0, GPR_R14, OFF_TAB_ALL1)
			emit_chan_gather_rgba(&e.buf, GPR_RBX, chan_disp, 4, 5, stub.comp_mask)
			sampled := x_jump(&e.buf, 0)
			x_land(&e.buf, general)
			if in_.op == .Texture_Bias {
				// With levels, the bias plus the level the quad's derivatives pick.
				emit_b(&e.buf, 0x83) // cmp dword [rdi + n_levels], 1
				emit_modrm_mem(&e.buf, 7, GPR_RDI, OFF_CI_NLEV)
				emit_b(&e.buf, 1)
				one := x_jump(&e.buf, JCC_JBE)
				emit_chan_quad_levels(&e.buf, in_.b)
				x_land(&e.buf, one)
			} else {
				vmovaps_load(&e.buf, 2, in_.b) // the lod (in memory: the main site flushed it)
			}
			if in_.n == 3 {
				// A sampler2DArray's layer: clamp(round_even(r), 0,
				// layers - 1), NaN as 0 (bound_layer).
				vmovaps_load(&e.buf, 3, in_.a + 2)
				vroundps(&e.buf, 3, 3, 0x08)
				vzero(&e.buf, 4)
				vop_ps(&e.buf, 0x5F, 3, 3, 4)
				vbroadcastss_mem(&e.buf, 4, GPR_RDI, OFF_CI_LTOP)
				vop_ps(&e.buf, 0x5D, 3, 3, 4)
				x_ps2dq_t(&e.buf, 3, 3)
			}
			emit_b(&e.buf, 0xBA) // mov edx, live components
			emit_u32(&e.buf, u32(stub.comp_mask))
			x_cmp_b(&e.buf, GPR_RDI, OFF_CI_LIN_ROUTE, 0)
			not_lin := x_jump(&e.buf, JCC_JNE)
			x_call(&e.buf, &e.bilinear_calls)
			lin_done := x_jump(&e.buf, 0)
			x_land(&e.buf, not_lin)
			x_call(&e.buf, &e.sample_calls)
			x_land(&e.buf, lin_done)
			x_land(&e.buf, sampled)
		}
		for i in 0 ..< i32(4) {
			if (stub.comp_mask & (u8(1) << uint(i))) != 0 {
				vmovaps_store(&e.buf, in_.dst + i, u8(i))
			}
		}
		emit_b(&e.buf, 0xE9) // jmp ret_off
		ret_patch := len(e.buf)
		emit_i32(&e.buf, 0)
		patch_i32(e.buf[:], ret_patch, i32(stub.ret_off - (ret_patch + 4)))
		stub.stub_end = len(e.buf)
	}
	for &stub in e.load_bufs {
		stub.stub_off = len(e.buf)
		patch_i32(e.buf[:], stub.unif_patch, i32(len(e.buf) - (stub.unif_patch + 4)))
		in_ := code[stub.pc]
		n := i32(in_.n)
		if n == 1 {
			vzero(&e.buf, stub.yd[0])
			emit_b(&e.buf, 0xE9) // jmp ret_off
			p := len(e.buf)
			emit_i32(&e.buf, 0)
			patch_i32(e.buf[:], p, i32(stub.ret_off - (p + 4)))

			patch_i32(e.buf[:], stub.big_patch, i32(len(e.buf) - (stub.big_patch + 4)))
			emit_b(&e.buf, 0xB8) // mov eax, 1
			emit_u32(&e.buf, 1)
			emit_b(&e.buf, 0xE8) // call big_buf_sub
			append(&e.big_buf_calls, len(e.buf))
			emit_i32(&e.buf, 0)
			vmovaps_rr(&e.buf, stub.yd[0], 0)
		} else {
			emit_b(&e.buf, 0xBA) // mov edx, u32(n)
			emit_u32(&e.buf, u32(n))
			emit_b(&e.buf, 0xE8) // call unif_oob_sub
			append(&e.unif_oob_calls, len(e.buf))
			emit_i32(&e.buf, 0)
			emit_b(&e.buf, 0xEB) // jmp copy_out
			copy_jmp := len(e.buf)
			emit_b(&e.buf, 0)

			patch_i32(e.buf[:], stub.big_patch, i32(len(e.buf) - (stub.big_patch + 4)))
			emit_b(&e.buf, 0xB8) // mov eax, u32(n)
			emit_u32(&e.buf, u32(n))
			emit_b(&e.buf, 0xE8) // call big_buf_sub
			append(&e.big_buf_calls, len(e.buf))
			emit_i32(&e.buf, 0)

			e.buf[copy_jmp] = u8(len(e.buf) - (copy_jmp + 1))
			for i in 0 ..< n {
				vmovaps_rr(&e.buf, stub.yd[i], u8(i))
			}
		}
		emit_b(&e.buf, 0xE9) // jmp ret_off
		ret_p := len(e.buf)
		emit_i32(&e.buf, 0)
		patch_i32(e.buf[:], ret_p, i32(stub.ret_off - (ret_p + 4)))
		stub.stub_end = len(e.buf)
	}
	if len(e.sample_calls) > 0 {
		e.chan_sample_off = len(e.buf)
		emit_chan_sample_sub(&e.buf, &e.decode_calls, e.fma)
		for cp in e.sample_calls {
			patch_i32(e.buf[:], cp, i32(e.chan_sample_off - (cp + 4)))
		}
	}
	if len(e.bilinear_calls) > 0 {
		e.chan_bilinear_off = len(e.buf)
		emit_chan_bilinear_sub(&e.buf, e.fma)
		for cp in e.bilinear_calls {
			patch_i32(e.buf[:], cp, i32(e.chan_bilinear_off - (cp + 4)))
		}
	}
	if len(e.fetch_calls) > 0 {
		e.chan_fetch_off = len(e.buf)
		emit_chan_fetch_sub(&e.buf, &e.decode_calls)
		for cp in e.fetch_calls {
			patch_i32(e.buf[:], cp, i32(e.chan_fetch_off - (cp + 4)))
		}
	}
	if len(e.decode_calls) > 0 {
		e.chan_decode_off = len(e.buf)
		emit_chan_decode_sub(&e.buf)
		for cp in e.decode_calls {
			patch_i32(e.buf[:], cp, i32(e.chan_decode_off - (cp + 4)))
		}
	}
	if len(e.helper_ymm_calls) > 0 {
		helper_ymm_off := len(e.buf)
		e.helper_ymm_off = helper_ymm_off
		emit_helper_ymm_sub(&e.buf)
		for cp in e.helper_ymm_calls {
			patch_i32(e.buf[:], cp, i32(helper_ymm_off - (cp + 4)))
		}
	}
	if len(e.unif_oob_calls) > 0 {
		unif_oob_off := len(e.buf)
		e.unif_oob_off = unif_oob_off
		emit_unif_oob_sub(&e.buf)
		for cp in e.unif_oob_calls {
			patch_i32(e.buf[:], cp, i32(unif_oob_off - (cp + 4)))
		}
	}
	if len(e.big_buf_calls) > 0 {
		big_buf_off := len(e.buf)
		e.big_buf_off = big_buf_off
		emit_big_buf_sub(&e.buf)
		for cp in e.big_buf_calls {
			patch_i32(e.buf[:], cp, i32(big_buf_off - (cp + 4)))
		}
	}
}


}
