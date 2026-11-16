package oksl

import "base:intrinsics"
import "base:runtime"
import "core:math"
import "core:math/linalg"
import "core:mem/virtual"

// CPU backend, part 3: native x86-64 AVX2 JIT compiler and 8-lane runners.
// Eight lanes pack two 2x2 quads into a single 256-bit YMM register (quad 0
// in the low 128-bit lane, quad 1 in the high 128-bit lane), so fine quad
// derivatives stay in-lane, divergence shrinks to two quads per batch, and
// temporary values live directly in YMM registers.

JIT_ENABLED :: #config(OKSL_JIT, true) && ODIN_ARCH == .amd64 && (ODIN_OS == .Linux || ODIN_OS == .Darwin)

when !JIT_ENABLED {

_ :: intrinsics
_ :: runtime
_ :: math
_ :: linalg
_ :: virtual

@(private)
jit_destroy :: proc(p: ^Program) {}

@(private)
jit_compile :: proc(p: ^Program) {}

@(private)
jit_runner_run :: proc(p: ^Program, uniforms: []u8, width, height: int, opts: Run_Options, out: []u8, stride: int, targets: [][][4]f32, scratch: ^Scratch) {}

@(private)
jit_run_vertex :: proc(p: ^Program, uniforms: []u8, attributes: []Attribute, count: int, position: [][4]f32, varyings: []f32, scratch: ^Scratch, opts: Run_Options) {}

@(private)
jit_run_fragment :: proc(p: ^Program, uniforms: []u8, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, scratch: ^Scratch, opts: Run_Options, sample_masks: []u32) {}

@(private)
jit_run_compute :: proc(p: ^Program, uniforms: []u8, groups: [3]int, start, end: int, scratch: ^Scratch, opts: Run_Options) {}

} else {

JIT_LANES :: 8
JIT_BATCH_W :: 4
JIT_BATCH_H :: 2

@(private)
Jit_Slot :: [JIT_LANES]u32

@(private)
Jit_Floats :: [JIT_LANES]f32

@(private)
Jit_Ctx :: struct {
	slots:    [^]Jit_Slot,
	masks:    [^]u8,
	opts:     ^Run_Options,
	lanes:    u8,
	buf_ptr:  [4][^]u32,
	buf_len:  [4]uint,
	host_ctx: runtime.Context,
	// Compute: the workgroup's shared memory (Program.shared_words) and the
	// storage images (Run_Options.images).
	shared:   [^]u32,
	imgs:     [IMAGE_COUNT]Jit_Img,
}

// A storage image as generated code reads it: width and height biased by 2^31
// (unsigned compares as signed), the row stride in bytes. The runner keeps
// every image's data under 2^31 bytes, so texel byte offsets fit the gathers'
// signed 32-bit indices.
@(private)
Jit_Img :: struct {
	ptr:    [^]u8,
	wb, hb: u32,
	stride: u32,
	_:      u32,
}

#assert(size_of(Chan_Img) == size_of(Jit_Slot))

@(private)
Jit_Fn :: #type proc "c" (ctx: ^Jit_Ctx)

@(private)
Jit_Help_Fn :: #type proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32)

@(private)
Jit_Tables :: struct #align(32) {
	mask_lut:   [256][8]i8,
	sign_mask:  [8]u32,
	abs_mask:   [8]u32,
	pos_inf:    [8]u32,
	one_u:      [8]u32,
	three_u:    [8]u32,
	mask31_u:   [8]u32,
	all_ones:   [8]u32,
	lane_iota:  [8]u32,
	zero_f:     [8]f32,
	one_f:      [8]f32,
	neg_one_f:  [8]f32,
	two_f:      [8]f32,
	three_f:    [8]f32,
	deg_to_rad: [8]f32,
	rad_to_deg: [8]f32,
	f2i_hi:     [8]f32,
	f2u_hi:     [8]f32,
	half_f:     [8]f32,
	i32_max_f:  [8]f32,
	f255:       [8]f32,
	byte_mask:  [8]u32,
	two52_d:    [4]u64,
}

#assert(offset_of(Jit_Tables, sign_mask) % 32 == 0)
#assert(offset_of(Jit_Tables, two52_d) % 32 == 0)

@(private)
jit_avx2_ready: bool

@(private)
jit_tables: Jit_Tables

@(init, private)
init_jit :: proc "contextless" () {
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
	jit_tables.half_f = 0.5
	jit_tables.one_f = 1.0
	jit_tables.neg_one_f = -1.0
	jit_tables.two_f = 2.0
	jit_tables.three_f = 3.0
	jit_tables.deg_to_rad = math.PI / 180.0
	jit_tables.rad_to_deg = 180.0 / math.PI
	jit_tables.i32_max_f = MAX_TEXEL_F32
	jit_tables.f2i_hi = 2147483648.0
	jit_tables.f2u_hi = 4294967296.0
	jit_tables.two52_d = 0x4330_0000_0000_0000
	jit_tables.f255 = 255.0
	jit_tables.byte_mask = 0xFF

	max_id, _, _, _ := intrinsics.x86_cpuid(0, 0)
	if max_id < 7 {
		return
	}
	_, _, ecx1, _ := intrinsics.x86_cpuid(1, 0)
	avx_bits :: u32((1 << 26) | (1 << 27) | (1 << 28))
	if (ecx1 & avx_bits) != avx_bits {
		return
	}
	xcr0, _ := intrinsics.x86_xgetbv(0)
	if (xcr0 & 0x6) != 0x6 {
		return
	}
	_, ebx7, _, _ := intrinsics.x86_cpuid(7, 0)
	jit_avx2_ready = (ebx7 & (1 << 5)) != 0
}

@(private)
Jit_Var_Span :: struct {
	offset: i32,
	slot:   i32,
	n:      i32,
}

@(private)
Jit_Meta :: struct {
	var_spans:        []Jit_Var_Span,
	max_var_col:      i32,
	fc_slot:          i32,
	fc_n:             i32,
	fc_base:          Base,
	ff_slot:          i32,
	ff_n:             i32,
	ff_base:          Base,
	outs:             [FRAGMENT_OUTPUTS]Program_Var,
	has_sample_mask:  bool,
	sample_mask_slot: i32,
	num_chans:        u8,
	num_buf_binds:    u8,
}

@(private)
jit_destroy :: proc(p: ^Program) {
	if p.jit_mem != nil {
		virtual.release(raw_data(p.jit_mem), len(p.jit_mem))
		p.jit_mem = nil
		p.jit_init = nil
		p.jit_body = nil
	}
	if p.jit_meta != nil {
		heap := runtime.default_allocator()
		meta := (^Jit_Meta)(p.jit_meta)
		delete(meta.var_spans, heap)
		free(meta, heap)
		p.jit_meta = nil
	}
}

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
OFF_CTX_SLOTS :: i32(offset_of(Jit_Ctx, slots))
@(private)
OFF_CTX_MASKS :: i32(offset_of(Jit_Ctx, masks))
@(private)
OFF_CTX_BUF_PTR :: i32(offset_of(Jit_Ctx, buf_ptr))
@(private)
OFF_CTX_BUF_LEN :: i32(offset_of(Jit_Ctx, buf_len))
@(private)
OFF_CTX_LANES :: i32(offset_of(Jit_Ctx, lanes))
@(private)
OFF_CTX_IMGS :: i32(offset_of(Jit_Ctx, imgs))
@(private)
OFF_IMG_PTR :: i32(offset_of(Jit_Img, ptr))
@(private)
OFF_IMG_WB :: i32(offset_of(Jit_Img, wb))
@(private)
OFF_IMG_HB :: i32(offset_of(Jit_Img, hb))
@(private)
OFF_IMG_STRIDE :: i32(offset_of(Jit_Img, stride))

@(private)
OFF_CHAN_PTR :: i32(offset_of(Chan_Img, ptr))
@(private)
OFF_CHAN_FW :: i32(offset_of(Chan_Img, fw))
@(private)
OFF_CHAN_FH :: i32(offset_of(Chan_Img, fh))
@(private)
OFF_CHAN_W :: i32(offset_of(Chan_Img, w))
@(private)
OFF_CHAN_MAX_X :: i32(offset_of(Chan_Img, max_x))
@(private)
OFF_CHAN_MAX_Y :: i32(offset_of(Chan_Img, max_y))
@(private)
OFF_CHAN_COMPS :: i32(offset_of(Chan_Img, comps))
@(private)
OFF_CHAN_SHIFT :: i32(offset_of(Chan_Img, shift))
@(private)
OFF_CHAN_FILTER :: i32(offset_of(Chan_Img, filter))

@(private)
OFF_TAB_SIGN :: i32(offset_of(Jit_Tables, sign_mask))
@(private)
OFF_TAB_ABS :: i32(offset_of(Jit_Tables, abs_mask))
@(private)
OFF_TAB_INF :: i32(offset_of(Jit_Tables, pos_inf))
@(private)
OFF_TAB_ONE_U :: i32(offset_of(Jit_Tables, one_u))
@(private)
OFF_TAB_THREE_U :: i32(offset_of(Jit_Tables, three_u))
@(private)
OFF_TAB_MASK31 :: i32(offset_of(Jit_Tables, mask31_u))
@(private)
OFF_TAB_ALL1 :: i32(offset_of(Jit_Tables, all_ones))
@(private)
OFF_TAB_IOTA :: i32(offset_of(Jit_Tables, lane_iota))
@(private)
OFF_TAB_HALF_F :: i32(offset_of(Jit_Tables, half_f))
@(private)
OFF_TAB_ONE_F :: i32(offset_of(Jit_Tables, one_f))
@(private)
OFF_TAB_NEG1_F :: i32(offset_of(Jit_Tables, neg_one_f))
@(private)
OFF_TAB_TWO_F :: i32(offset_of(Jit_Tables, two_f))
@(private)
OFF_TAB_THREE_F :: i32(offset_of(Jit_Tables, three_f))
@(private)
OFF_TAB_DEG2RAD :: i32(offset_of(Jit_Tables, deg_to_rad))
@(private)
OFF_TAB_RAD2DEG :: i32(offset_of(Jit_Tables, rad_to_deg))
@(private)
OFF_TAB_I32MAX :: i32(offset_of(Jit_Tables, i32_max_f))
@(private)
OFF_TAB_F2I_HI :: i32(offset_of(Jit_Tables, f2i_hi))
@(private)
OFF_TAB_F2U_HI :: i32(offset_of(Jit_Tables, f2u_hi))
@(private)
OFF_TAB_TWO52_D :: i32(offset_of(Jit_Tables, two52_d))
@(private)
OFF_TAB_F255 :: i32(offset_of(Jit_Tables, f255))
@(private)
OFF_TAB_BYTE :: i32(offset_of(Jit_Tables, byte_mask))

@(private)
emit_b :: #force_inline proc(buf: ^[dynamic]u8, b0: u8) {
	append(buf, b0)
}

@(private)
emit_bytes :: #force_inline proc(buf: ^[dynamic]u8, bs: ..u8) {
	append(buf, ..bs)
}

@(private)
emit_u32 :: #force_inline proc(buf: ^[dynamic]u8, v: u32) {
	append(buf, u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24))
}

@(private)
emit_i32 :: #force_inline proc(buf: ^[dynamic]u8, v: i32) {
	emit_u32(buf, u32(v))
}

@(private)
emit_u64 :: #force_inline proc(buf: ^[dynamic]u8, v: u64) {
	emit_u32(buf, u32(v))
	emit_u32(buf, u32(v >> 32))
}

@(private)
patch_i32 :: #force_inline proc(buf: []u8, pos: int, v: i32) {
	u := u32(v)
	buf[pos] = u8(u)
	buf[pos + 1] = u8(u >> 8)
	buf[pos + 2] = u8(u >> 16)
	buf[pos + 3] = u8(u >> 24)
}

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

	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [chan_base + chan_disp + OFF_CHAN_COMPS]
	emit_modrm_mem(buf, GPR_RCX, chan_base, chan_disp + OFF_CHAN_COMPS)
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

// Internal AVX2 subroutine for bilinear bound-channel taps:
//   in:  ymm0 = u, ymm1 = v, rsi = im.ptr, rdi = ^Chan_Img, dl = comp_mask (bits 0..3)
//   out: ymm0..3 = live RGBA components; preserves rbp, rbx, r12..r14 and ymm8..15.
@(private)
emit_bilinear_sub :: proc(buf: ^[dynamic]u8) {
	emit_bytes(buf, 0x48, 0x81, 0xEC) // sub rsp, 192
	emit_u32(buf, 192)

	// (x0, x1, fx) from ymm0 (u); keep ymm1 (v) live.
	vbroadcastss_mem(buf, 2, GPR_RDI, OFF_CHAN_FW)
	vop_ps(buf, 0x59, 2, 0, 2)
	vop_ps_tab(buf, 0x5C, 2, 2, OFF_TAB_HALF_F)
	vroundps(buf, 3, 2, 0x09)
	vop_ps(buf, 0x5C, 4, 2, 3)
	vzero(buf, 6)
	vop_ps(buf, 0x5F, 4, 4, 6)
	emit_vex_rm(buf, 1, 0, 1, 0, 0x11, 4, 0, GPR_RSP, 128) // [rsp + 128] = fx
	vop_ps_tab(buf, 0x5F, 3, 3, OFF_TAB_NEG1_F)
	vop_ps_tab(buf, 0x5D, 3, 3, OFF_TAB_I32MAX)
	emit_vex_rr(buf, 1, 2, 1, 0, 0x5B, 3, 0, 3)
	vpbroadcastd_mem(buf, 5, GPR_RDI, OFF_CHAN_MAX_X)
	vop38_d(buf, 0x3D, 0, 3, 6)
	vop38_d(buf, 0x39, 0, 0, 5) // ymm0 = x0
	vop_d_tab(buf, 0xFE, 2, 3, OFF_TAB_ONE_U)
	vop38_d(buf, 0x39, 2, 2, 5) // ymm2 = x1

	// (y0, y1, fy) from ymm1 (v).
	vbroadcastss_mem(buf, 3, GPR_RDI, OFF_CHAN_FH)
	vop_ps(buf, 0x59, 3, 1, 3)
	vop_ps_tab(buf, 0x5C, 3, 3, OFF_TAB_HALF_F)
	vroundps(buf, 4, 3, 0x09)
	vop_ps(buf, 0x5C, 5, 3, 4)
	vop_ps(buf, 0x5F, 5, 5, 6)
	emit_vex_rm(buf, 1, 0, 1, 0, 0x11, 5, 0, GPR_RSP, 160) // [rsp + 160] = fy
	vop_ps_tab(buf, 0x5F, 4, 4, OFF_TAB_NEG1_F)
	vop_ps_tab(buf, 0x5D, 4, 4, OFF_TAB_I32MAX)
	emit_vex_rr(buf, 1, 2, 1, 0, 0x5B, 4, 0, 4)
	vpbroadcastd_mem(buf, 5, GPR_RDI, OFF_CHAN_MAX_Y)
	vop38_d(buf, 0x3D, 1, 4, 6)
	vop38_d(buf, 0x39, 1, 1, 5) // ymm1 = y0
	vop_d_tab(buf, 0xFE, 3, 4, OFF_TAB_ONE_U)
	vop38_d(buf, 0x39, 3, 3, 5) // ymm3 = y1

	// Scaled float indices [rsp + 0, 32, 64, 96] = (y*w + x) << shift.
	vpbroadcastd_mem(buf, 4, GPR_RDI, OFF_CHAN_W)
	vop38_d(buf, 0x40, 1, 1, 4) // y0 * w
	vop38_d(buf, 0x40, 3, 3, 4) // y1 * w
	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [rdi + OFF_CHAN_SHIFT]
	emit_modrm_mem(buf, GPR_RCX, GPR_RDI, OFF_CHAN_SHIFT)
	emit_vex_rr(buf, 1, 1, 0, 0, 0x6E, 4, 0, GPR_RCX)
	emit_vex_rr(buf, 2, 1, 1, 0, 0x58, 4, 0, 4)

	corners := [4][3]i32{{1, 0, 0}, {1, 2, 32}, {3, 0, 64}, {3, 2, 96}}
	for co in corners {
		vop_d(buf, 0xFE, 5, u8(co[0]), u8(co[1]))
		vop38_d(buf, 0x47, 5, 5, 4)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x11, 5, 0, GPR_RSP, co[2])
	}

	emit_bytes(buf, 0x0F, 0xB6) // movzx ecx, byte [rdi + OFF_CHAN_COMPS]
	emit_modrm_mem(buf, GPR_RCX, GPR_RDI, OFF_CHAN_COMPS)

	comp1_patch, comp2_patch := 0, 0
	done_patches: [3]int
	n_done := 0
	for c in 0 ..< i32(4) {
		emit_bytes(buf, 0xF6, 0xC2, u8(1 << uint(c))) // test dl, (1 << c)
		emit_bytes(buf, 0x0F, 0x84)                   // jz skip_c
		skip_patch := len(buf^)
		emit_i32(buf, 0)

		yd := u8(c)
		// c00 -> yd, c10 -> ymm4, top -> yd
		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, 6, 0, GPR_RSP, 0)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x28, 7, 0, GPR_R14, OFF_TAB_ALL1)
		vzero(buf, yd)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, yd, 7, GPR_RSI, 6, 2, c * 4)

		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, 6, 0, GPR_RSP, 32)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x28, 7, 0, GPR_R14, OFF_TAB_ALL1)
		vzero(buf, 4)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, 4, 7, GPR_RSI, 6, 2, c * 4)

		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, 6, 0, GPR_RSP, 128) // fx
		vop_ps(buf, 0x5C, 4, 4, yd)
		vop_ps(buf, 0x59, 4, 4, 6)
		vop_ps(buf, 0x58, yd, yd, 4)

		// c01 -> ymm4, c11 -> ymm5, bot -> ymm4, out -> yd
		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, 6, 0, GPR_RSP, 64)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x28, 7, 0, GPR_R14, OFF_TAB_ALL1)
		vzero(buf, 4)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, 4, 7, GPR_RSI, 6, 2, c * 4)

		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, 6, 0, GPR_RSP, 96)
		emit_vex_rm(buf, 1, 0, 1, 0, 0x28, 7, 0, GPR_R14, OFF_TAB_ALL1)
		vzero(buf, 5)
		emit_vex_sib(buf, 2, 1, 1, 0, 0x90, 5, 7, GPR_RSI, 6, 2, c * 4)

		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, 6, 0, GPR_RSP, 128) // fx
		vop_ps(buf, 0x5C, 5, 5, 4)
		vop_ps(buf, 0x59, 5, 5, 6)
		vop_ps(buf, 0x58, 4, 4, 5)

		emit_vex_rm(buf, 1, 0, 1, 0, 0x10, 6, 0, GPR_RSP, 160) // fy
		vop_ps(buf, 0x5C, 4, 4, yd)
		vop_ps(buf, 0x59, 4, 4, 6)
		vop_ps(buf, 0x58, yd, yd, 4)

		patch_i32(buf[:], skip_patch, i32(len(buf^) - (skip_patch + 4)))
		if c == 0 {
			emit_bytes(buf, 0xF6, 0xC2, 0x0E) // test dl, 0x0E
			emit_bytes(buf, 0x0F, 0x84)       // jz done
			done_patches[n_done] = len(buf^)
			n_done += 1
			emit_i32(buf, 0)

			emit_bytes(buf, 0x83, 0xF9, 0x02) // cmp ecx, 2
			emit_bytes(buf, 0x0F, 0x82)       // jb comp1
			comp1_patch = len(buf^)
			emit_i32(buf, 0)
		} else if c == 1 {
			emit_bytes(buf, 0xF6, 0xC2, 0x0C) // test dl, 0x0C
			emit_bytes(buf, 0x0F, 0x84)       // jz done
			done_patches[n_done] = len(buf^)
			n_done += 1
			emit_i32(buf, 0)

			emit_bytes(buf, 0x83, 0xF9, 0x02) // cmp ecx, 2
			emit_bytes(buf, 0x0F, 0x84)       // je comp2
			comp2_patch = len(buf^)
			emit_i32(buf, 0)
		}
	}
	emit_b(buf, 0xEB) // jmp done
	done_short := len(buf^)
	emit_b(buf, 0)

	patch_i32(buf[:], comp1_patch, i32(len(buf^) - (comp1_patch + 4)))
	vzero(buf, 1)
	patch_i32(buf[:], comp2_patch, i32(len(buf^) - (comp2_patch + 4)))
	vzero(buf, 2)
	emit_vex_rm(buf, 1, 0, 1, 0, 0x28, 3, 0, GPR_R14, OFF_TAB_ONE_F)

	buf[done_short] = u8(len(buf^) - (done_short + 1))
	for dp in done_patches[:n_done] {
		patch_i32(buf[:], dp, i32(len(buf^) - (dp + 4)))
	}
	emit_bytes(buf, 0x48, 0x81, 0xC4) // add rsp, 192
	emit_u32(buf, 192)
	emit_b(buf, 0xC3) // ret
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

// ---- basic-block slot register cache (ymm8..ymm15) --------------------------------------

@(private)
Cache_Slot :: struct {
	slot:     i32,
	last_use: i32,
	dirty:    bool,
	stamp:    u32,
}

@(private)
Instr_RW :: struct {
	writes:     Slots,
	full_write: bool,
	reads:      [4]Slots,
}

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
Jit_Emitter :: struct {
	buf:              [dynamic]u8,
	mem_mask:         []u16,
	tex_mask:         []u8,
	// Slots holding a constant that isn't NaN (Program.consts; nothing
	// else writes them).
	num_const:        []bool,
	pc_to_bb:         []i32,
	last_use:         []i32,
	def_off:          []i32,
	def_lu:           [dynamic]i32,
	bb_start:         []bool,
	mask_written:     []bool,
	mask_eq_gen:      []u32,
	pc_off:           []int,
	rw:               []Instr_RW,
	patches:          [dynamic]Jump_Patch,
	epilogues:        [dynamic]int,
	bilinear_calls:   [dynamic]int,
	helper_ymm_calls: [dynamic]int,
	unif_oob_calls:   [dynamic]int,
	big_buf_calls:    [dynamic]int,
	bound_tex:        [dynamic]Bound_Tex_Stub,
	load_bufs:        [dynamic]Load_Buf_Stub,
	epilogue_off:     int,
	bilinear_off:     int,
	helper_ymm_off:   int,
	unif_oob_off:     int,
	big_buf_off:      int,
	cache:            [8]Cache_Slot,
	stamp:            u32,
	exec_gen:         u32,
	pc:               i32,
	cur_dst_base:     i32,
	cur_def_off:      i32,
	cur_mem_mask:     u16,
	fused_mc_pc:      i32,
	fused_mc_inv:     bool,
	exec_ymask_valid: bool,
}

@(private)
cache_reset :: proc(e: ^Jit_Emitter) {
	for &c in e.cache {
		c = {slot = -1, last_use = -1, dirty = false, stamp = 0}
	}
	e.exec_gen += 1
	e.fused_mc_pc = -1
	e.fused_mc_inv = false
	e.exec_ymask_valid = false
}

@(private)
is_mask_eq_exec :: #force_inline proc(e: ^Jit_Emitter, m: i32) -> bool {
	return m == MASK_EXEC || e.mask_eq_gen[m] == e.exec_gen
}

@(private)
invalidate_mask_write :: #force_inline proc(e: ^Jit_Emitter, dst: i32) {
	if dst == MASK_EXEC {
		e.exec_gen += 1
		e.exec_ymask_valid = false
	} else if dst > 0 {
		e.mask_eq_gen[dst] = 0
	}
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

@(private)
pin_bit :: #force_inline proc(ymm: u8) -> u8 {
	return (u8(1) << (ymm - 8)) if ymm >= 8 else 0
}

@(private)
write_lu :: #force_inline proc(e: ^Jit_Emitter, slot: i32) -> i32 {
	return e.def_lu[e.cur_def_off + (slot - e.cur_dst_base)]
}

@(private)
alloc_cache_idx :: proc(e: ^Jit_Emitter, pinned: u8) -> int {
	// Prefer empty or already-dead entries first (zero spill cost).
	for c, i in e.cache {
		if (pinned & (u8(1) << uint(i))) != 0 {
			continue
		}
		if c.slot < 0 || c.last_use < e.pc {
			e.cache[i] = {slot = -1, last_use = -1, dirty = false, stamp = 0}
			return i
		}
	}
	best := -1
	best_stamp := max(u32)
	for c, i in e.cache {
		if (pinned & (u8(1) << uint(i))) != 0 {
			continue
		}
		if c.stamp < best_stamp {
			best_stamp = c.stamp
			best = i
		}
	}
	if e.cache[best].dirty {
		vmovaps_store(&e.buf, e.cache[best].slot, u8(8 + best))
	}
	e.cache[best] = {slot = -1, last_use = -1, dirty = false, stamp = 0}
	return best
}

@(private)
read_slot :: proc(e: ^Jit_Emitter, slot: i32, scratch: u8, pinned: u8 = 0) -> u8 {
	for &c, i in e.cache {
		if c.slot == slot {
			e.stamp += 1
			c.stamp = e.stamp
			return u8(8 + i)
		}
	}
	rlu := e.last_use[slot]
	if rlu > e.pc {
		idx := alloc_cache_idx(e, pinned)
		r := u8(8 + idx)
		vmovaps_load(&e.buf, r, slot)
		e.stamp += 1
		e.cache[idx] = {slot = slot, last_use = rlu, dirty = false, stamp = e.stamp}
		return r
	}
	vmovaps_load(&e.buf, scratch, slot)
	return scratch
}

// Selects the destination YMM register for `slot` (`ymm8..15` when read again
// in this block, else `scratch` in `ymm0..7`).
// - `pinned`: `ymm8..15` bits read by this step (never evicted).
// - `no_alias`: `ymm8..15` bits read after `yd` is written (never reused as `yd`).
@(private)
prep_dst :: proc(e: ^Jit_Emitter, slot: i32, scratch: u8 = 0, pinned: u8 = 0, no_alias: u8 = 0) -> u8 {
	wlu := write_lu(e, slot)
	for &c, i in e.cache {
		if c.slot == slot {
			bit := u8(1) << uint(i)
			if (no_alias & bit) != 0 || wlu <= e.pc {
				c = {slot = -1, last_use = -1, dirty = false, stamp = 0}
				break
			}
			c.last_use = wlu
			return u8(8 + i)
		}
	}
	if wlu > e.pc {
		idx := alloc_cache_idx(e, pinned | no_alias)
		e.cache[idx].slot = slot
		e.cache[idx].last_use = wlu
		return u8(8 + idx)
	}
	return scratch
}

@(private)
need_mem_write :: #force_inline proc(e: ^Jit_Emitter, slot: i32) -> bool {
	idx := uint(slot - e.cur_dst_base)
	if idx >= 16 {
		return true
	}
	return (e.cur_mem_mask & (u16(1) << idx)) != 0
}

@(private)
commit_dst :: proc(e: ^Jit_Emitter, slot: i32, yd: u8) {
	need_mem := need_mem_write(e, slot)
	wlu := write_lu(e, slot)
	if yd >= 8 && wlu > e.pc {
		idx := yd - 8
		e.stamp += 1
		e.cache[idx] = {slot = slot, last_use = wlu, dirty = !need_mem, stamp = e.stamp}
	}
	if need_mem {
		vmovaps_store(&e.buf, slot, yd)
	}
}

@(private)
copy_to_dst :: proc(e: ^Jit_Emitter, slot: i32, src: u8, pinned: u8 = 0, no_alias: u8 = 0) {
	yd := prep_dst(e, slot, src, pinned, no_alias)
	if yd != src {
		vmovaps_rr(&e.buf, yd, src)
	}
	commit_dst(e, slot, yd)
}

@(private)
is_helper_instr :: proc(p: ^Program, in_: Instr) -> bool {
	#partial switch in_.op {
	case .Image_Load:
		return p.image_formats[in_.k] == .Rgba16f
	case .Store_Dyn,
	     .Faceforward, .Refract,
	     .Determinant, .Inverse,
	     .Texture_Size,
	     .Store_Buf, .Image_Store, .Image_Size,
	     .Load_Shared, .Store_Shared, .Atomic_Shared, .Atomic_Buf:
		return true
	case .Load_Dyn:
		return int(in_.n) > 4
	case .Load_Buf:
		return int(in_.k) >= 4
	case .F_Fn1:
		#partial switch Builtin(in_.k) {
		case .Radians, .Degrees, .Sqrt, .Inversesqrt,
		     .Floor, .Ceil, .Fract, .Trunc, .Round, .Round_Even,
		     .Abs, .Sign, .Isnan, .Isinf:
			return false
		}
		return true
	case .F_Fn2:
		b := Builtin(in_.k)
		return b == .Atan2 || b == .Pow
	}
	return false
}

@(private)
instr_read_write :: proc(in_: Instr) -> (writes: Slots, full_write: bool, reads: [4]Slots) {
	w, r, pure := instr_slots(in_)
	if pure {
		return w, w.n > 0, r
	}
	n := i32(in_.n)
	#partial switch in_.op {
	case .Texture, .Texture_Bias, .Texel_Fetch:
		reads[0] = {in_.a, n}
		reads[1] = {in_.b, 1}
		if in_.d & 2 != 0 {
			reads[2] = {in_.c, 2}
		}
		return {in_.dst, 4}, true, reads
	case .Deriv_X, .Deriv_Y, .Fwidth:
		reads[0] = {in_.a, n}
		return {in_.dst, n}, true, reads
	case .Load_Buf:
		reads[0] = {in_.a, 1}
		return {in_.dst, n}, true, reads
	case .Load_Dyn:
		reads[0] = {in_.b, 1}
		return {in_.dst, n}, true, reads
	case .Store_Dyn:
		reads[0] = {in_.a, n}
		reads[1] = {in_.b, 1}
		return {}, false, reads
	case .Store:
		reads[0] = {in_.a, n}
		reads[1] = {in_.dst, n}
		return {in_.dst, n}, false, reads
	case .Store_Zero:
		reads[0] = {in_.dst, n}
		return {in_.dst, n}, false, reads
	case .Mask_Cond:
		reads[0] = {in_.b, 1}
		return {}, false, reads
	case .Store_Buf:
		reads[0] = {in_.a, 1}
		reads[1] = {in_.b, n}
		return {}, false, reads
	case .Image_Load:
		reads[0] = {in_.a, 2}
		return {in_.dst, 4}, true, reads
	case .Image_Store:
		reads[0] = {in_.a, 2}
		reads[1] = {in_.b, 4}
		return {}, false, reads
	case .Load_Shared:
		if in_.k != 0 {
			reads[0] = {in_.a, 1}
		}
		return {in_.dst, n}, true, reads
	case .Store_Shared:
		reads[0] = {in_.a, n}
		if in_.k != 0 {
			reads[1] = {in_.b, 1}
		}
		return {}, false, reads
	case .Atomic_Shared, .Atomic_Buf:
		reads[0] = {in_.a, 1}
		reads[1] = {in_.b, 1}
		reads[2] = {in_.c, 1}
		return {in_.dst, 1}, true, reads
	}
	return {}, false, reads
}

// ---- cold instruction helpers (`proc "c"`) ----------------------------------------------

@(private, enable_target_feature = "avx2")
jit_help_load_dyn :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	n := i32(in_.n)
	for l in 0 ..< JIT_LANES {
		off := i32(slots[in_.b][l])
		for i in 0 ..< n {
			slots[in_.dst + i][l] = slots[in_.a + off + i][l]
		}
	}
}

@(private, enable_target_feature = "avx2")
jit_help_store_dyn :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	if exec == 0 {
		return
	}
	slots := ctx.slots
	n := i32(in_.n)
	for l in 0 ..< JIT_LANES {
		if (exec & (1 << uint(l))) == 0 {
			continue
		}
		off := i32(slots[in_.b][l])
		for i in 0 ..< n {
			slots[in_.dst + off + i][l] = slots[in_.a + i][l]
		}
	}
}

@(private, enable_target_feature = "avx2")
jit_help_sin :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	for i in 0 ..< i32(in_.n) {
		x := (^Jit_Floats)(&slots[in_.a + i])
		d := (^Jit_Floats)(&slots[in_.dst + i])
		for l in 0 ..< JIT_LANES {d[l] = math.sin(x[l])}
	}
}

@(private, enable_target_feature = "avx2")
jit_help_cos :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	for i in 0 ..< i32(in_.n) {
		x := (^Jit_Floats)(&slots[in_.a + i])
		d := (^Jit_Floats)(&slots[in_.dst + i])
		for l in 0 ..< JIT_LANES {d[l] = math.cos(x[l])}
	}
}

@(private, enable_target_feature = "avx2")
jit_help_fn1 :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	b := Builtin(in_.k)
	for i in 0 ..< i32(in_.n) {
		x := (^Jit_Floats)(&slots[in_.a + i])
		d := &slots[in_.dst + i]
		#partial switch b {
		case .Tan:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Tan)}
		case .Asin:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Asin)}
		case .Acos:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Acos)}
		case .Atan:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Atan)}
		case .Sinh:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Sinh)}
		case .Cosh:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Cosh)}
		case .Tanh:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Tanh)}
		case .Asinh:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Asinh)}
		case .Acosh:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Acosh)}
		case .Atanh:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Atanh)}
		case .Exp:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Exp)}
		case .Log:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Log)}
		case .Exp2:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Exp2)}
		case .Log2:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Log2)}
		}
	}
}

@(private, enable_target_feature = "avx2")
jit_atan2_4 :: #force_inline proc "contextless" (y, x: #simd[4]f64) -> #simd[4]f64 {
	// Exact SIMD evaluation of core:math.atan2_f64 (Stephen L. Moshier) for finite non-zero lanes.
	PI        :: 0h4009_21fb_5444_2d18
	MORE_BITS :: 6.123233995736765886130e-17
	TAN3PI08  :: 2.41421356237309504880
	P0        :: -8.750608600031904122785e-01
	P1        :: -1.615753718733365076637e+01
	P2        :: -7.500855792314704667340e+01
	P3        :: -1.228866684490136173410e+02
	P4        :: -6.485021904942025371773e+01
	Q0        :: +2.485846490142306297962e+01
	Q1        :: +1.650270098316988542046e+02
	Q2        :: +4.328810604912902668951e+02
	Q3        :: +4.853903996359136964868e+02
	Q4        :: +1.945506571482613964425e+02

	r := y / x
	ax := intrinsics.simd_abs(r)
	m_small := intrinsics.simd_lanes_le(ax, 0.66)
	m_large := intrinsics.simd_lanes_gt(ax, TAN3PI08)
	u := intrinsics.simd_select(m_small, ax, intrinsics.simd_select(m_large, 1.0 / ax, (ax - 1.0) / (ax + 1.0)))
	z := u * u
	num := z * ((((P0 * z + P1) * z + P2) * z + P3) * z + P4)
	den := ((((z + Q0) * z + Q1) * z + Q2) * z + Q3) * z + Q4
	xa := u * (num / den) + u
	sa := intrinsics.simd_select(
		m_small,
		xa,
		intrinsics.simd_select(m_large, (PI / 2 - xa) + MORE_BITS, (PI / 4 + xa) + 0.5 * MORE_BITS),
	)
	q := intrinsics.simd_select(intrinsics.simd_lanes_gt(r, 0), sa, -sa)
	q_neg := intrinsics.simd_select(intrinsics.simd_lanes_le(q, 0), q + PI, q - PI)
	return intrinsics.simd_select(intrinsics.simd_lanes_lt(x, 0), q_neg, q)
}

@(private, enable_target_feature = "avx2")
jit_help_atan2 :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	for i in 0 ..< i32(in_.n) {
		yf := (^Jit_Floats)(&slots[in_.a + i])
		xf := (^Jit_Floats)(&slots[in_.b + i])
		d := (^Jit_Floats)(&slots[in_.dst + i])
		yb := transmute(#simd[8]i32)((^#simd[8]u32)(yf)^ & 0x7FFF_FFFF)
		xb := transmute(#simd[8]i32)((^#simd[8]u32)(xf)^ & 0x7FFF_FFFF)
		bad := intrinsics.simd_lanes_le(yb, 0) |
			intrinsics.simd_lanes_ge(yb, 0x7F80_0000) |
			intrinsics.simd_lanes_le(xb, 0) |
			intrinsics.simd_lanes_ge(xb, 0x7F80_0000)
		bad_mask := transmute(u8)intrinsics.simd_extract_msbs(bad)
		if (bad_mask & 0x0F) == 0 {
			ans := transmute([4]f64)jit_atan2_4(#simd[4]f64{f64(yf[0]), f64(yf[1]), f64(yf[2]), f64(yf[3])}, #simd[4]f64{f64(xf[0]), f64(xf[1]), f64(xf[2]), f64(xf[3])})
			d[0], d[1], d[2], d[3] = f32(ans[0]), f32(ans[1]), f32(ans[2]), f32(ans[3])
		} else {
			for l in 0 ..< 4 {d[l] = math.atan2(yf[l], xf[l])}
		}
		if (bad_mask & 0xF0) == 0 {
			ans := transmute([4]f64)jit_atan2_4(#simd[4]f64{f64(yf[4]), f64(yf[5]), f64(yf[6]), f64(yf[7])}, #simd[4]f64{f64(xf[4]), f64(xf[5]), f64(xf[6]), f64(xf[7])})
			d[4], d[5], d[6], d[7] = f32(ans[0]), f32(ans[1]), f32(ans[2]), f32(ans[3])
		} else {
			for l in 4 ..< 8 {d[l] = math.atan2(yf[l], xf[l])}
		}
	}
}

@(private, enable_target_feature = "avx2")
jit_help_pow :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	for i in 0 ..< i32(in_.n) {
		x := (^Jit_Floats)(&slots[in_.a + i])
		y := (^Jit_Floats)(&slots[in_.b + i])
		d := (^Jit_Floats)(&slots[in_.dst + i])
		for l in 0 ..< JIT_LANES {d[l] = math.pow(x[l], y[l])}
	}
}

@(private, enable_target_feature = "avx2")
jit_help_geom :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	n := i32(in_.n)
	d := (^Jit_Floats)(&slots[in_.a])^ * (^Jit_Floats)(&slots[in_.b])^
	if in_.op == .Faceforward {
		d = (^Jit_Floats)(&slots[in_.c])^ * (^Jit_Floats)(&slots[in_.b])^
	}
	x := in_.c if in_.op == .Faceforward else in_.a
	for i in 1 ..< n {
		d += (^Jit_Floats)(&slots[x + i])^ * (^Jit_Floats)(&slots[in_.b + i])^
	}
	if in_.op == .Faceforward {
		for i in 0 ..< n {
			v := (^Jit_Floats)(&slots[in_.a + i])^
			for l in 0 ..< JIT_LANES {
				v[l] = v[l] if d[l] < 0 else -v[l]
			}
			(^Jit_Floats)(&slots[in_.dst + i])^ = v
		}
		return
	}
	eta := (^Jit_Floats)(&slots[in_.c])^
	k := 1 - eta * eta * (1 - d * d)
	s: Jit_Floats
	for l in 0 ..< JIT_LANES {
		s[l] = eta[l] * d[l] + math.sqrt(k[l])
	}
	for i in 0 ..< n {
		v := eta * (^Jit_Floats)(&slots[in_.a + i])^ - s * (^Jit_Floats)(&slots[in_.b + i])^
		for l in 0 ..< JIT_LANES {
			v[l] = v[l] if k[l] >= 0 else 0
		}
		(^Jit_Floats)(&slots[in_.dst + i])^ = v
	}
}

@(private)
jit_mat_n :: proc "contextless" (slots: [^]Jit_Slot, in_: ^Instr, $N: int) #no_bounds_check {
	for l in 0 ..< JIT_LANES {
		m: matrix[N, N]f32
		for c in 0 ..< N {
			for r in 0 ..< N {
				m[r, c] = fval(slots[in_.a + i32(c * N + r)][l])
			}
		}
		if in_.op == .Determinant {
			slots[in_.dst][l] = fbits(linalg.determinant(m))
			continue
		}
		inv := linalg.inverse(m)
		for c in 0 ..< N {
			for r in 0 ..< N {
				slots[in_.dst + i32(c * N + r)][l] = fbits(inv[r, c])
			}
		}
	}
}

@(private)
jit_help_mat :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	switch in_.k {
	case 2:
		jit_mat_n(ctx.slots, in_, 2)
	case 3:
		jit_mat_n(ctx.slots, in_, 3)
	case:
		jit_mat_n(ctx.slots, in_, 4)
	}
}

@(private)
jit_help_texture :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	dst, a, b := in_.dst, in_.a, in_.b
	for i in 0 ..< i32(4) {
		slots[dst + i] = {}
	}
	opts := ctx.opts
	if opts.sampler == nil {
		return
	}
	ch := int(in_.k)
	context = ctx.host_ctx
	size := opts.channel_sizes[ch] if ch < len(opts.channel_sizes) else [3]i32{}
	lod := (^Jit_Floats)(&slots[b])
	u, v := (^Jit_Floats)(&slots[a]), (^Jit_Floats)(&slots[a + 1])
	have := u32(ctx.lanes)
	for o := 0; o < JIT_LANES; o += 4 {
		if (exec >> uint(o)) & 0xf == 0 {
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
			layer := int(f_to_i(fval(fn1(fval(slots[a + 2][l]), .Round_Even)))) if in_.n == 3 else 0
			c := opts.sampler(opts.user, ch, {u[l], v[l]}, lod[l] + level, layer)
			for i in 0 ..< i32(4) {
				slots[dst + i][l] = fbits(c[i])
			}
		}
	}
}

@(private, enable_target_feature = "avx2")
jit_help_texel_fetch :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	dst := in_.dst
	for i in 0 ..< i32(4) {
		slots[dst + i] = {}
	}
	opts := ctx.opts
	if opts.sampler == nil {
		return
	}
	ch := int(in_.k)
	context = ctx.host_ctx
	size := opts.channel_sizes[ch] if ch < len(opts.channel_sizes) else [3]i32{}
	px := (^Simd_I8)(&slots[in_.a])^
	py := (^Simd_I8)(&slots[in_.a + 1])^
	lb := &slots[in_.b]
	lb_v := (^Simd_U8)(lb)^
	sx, sy: Simd_F8
	if (in_.d & 2) != 0 {
		sx = (^Simd_F8)(&slots[in_.c])^
		sy = (^Simd_F8)(&slots[in_.c + 1])^
	} else {
		if size.x > 0 {
			sx = (Simd_F8)((Simd_I8)(intrinsics.simd_max(intrinsics.simd_shr(Simd_U8(u32(size.x)), lb_v), Simd_U8(1))))
		}
		if size.y > 0 {
			sy = (Simd_F8)((Simd_I8)(intrinsics.simd_max(intrinsics.simd_shr(Simd_U8(u32(size.y)), lb_v), Simd_U8(1))))
		}
	}
	ux := transmute([8]f32)(((Simd_F8)(px) + 0.5) / sx)
	uy_v := (((Simd_F8)(py) + 0.5) / sy)
	if (in_.d & 1) != 0 {
		uy_v = uy_v * -1 + 1
	}
	uy := transmute([8]f32)uy_v
	uint_lod := (in_.d & 4) != 0
	have := u32(ctx.lanes)
	for o := 0; o < JIT_LANES; o += 4 {
		if (exec >> uint(o)) & 0xf == 0 {
			continue
		}
		for l in o ..< o + 4 {
			if (have >> uint(l)) & 1 == 0 {
				continue
			}
			flod := f32(lb[l]) if uint_lod else f32(i32(lb[l]))
			c := opts.sampler(opts.user, ch, {ux[l], uy[l]}, flod, 0)
			for i in 0 ..< i32(4) {
				slots[dst + i][l] = fbits(c[i])
			}
		}
	}
}

@(private)
jit_help_texture_size :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	ch := int(in_.k)
	chan_imgs := ([^]Chan_Img)(uintptr(slots) - SAMPLER_COUNT * size_of(Chan_Img))
	if chan_imgs[ch].ptr != nil {
		im := &chan_imgs[ch]
		slots[in_.dst] = u32(im.w)
		slots[in_.dst + 1] = u32(im.max_y + 1)
		return
	}
	opts := ctx.opts
	size := opts.channel_sizes[ch] if ch < len(opts.channel_sizes) else [3]i32{}
	for l in 0 ..< JIT_LANES {
		lod := slots[in_.a][l]
		for i in 0 ..< i32(2) {
			slots[in_.dst + i][l] = u32(max(size[i] >> lod, 1) if size[i] > 0 else 0)
		}
		if in_.n == 3 {
			slots[in_.dst + 2][l] = u32(size.z)
		}
	}
}

@(private)
jit_help_load_buf :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	opts := ctx.opts
	words := opts.buffers[in_.k] if int(in_.k) < len(opts.buffers) else nil
	idx := &slots[in_.a]
	for i in 0 ..< i32(in_.n) {
		d := &slots[in_.dst + i]
		for l in 0 ..< JIT_LANES {
			w := idx[l] * u32(in_.b) + u32(in_.c + i)
			d[l] = words[w] if int(w) < len(words) else 0
		}
	}
}

// Compute memory ops: the interpreter's lane kernels (vm_run.odin) over 8 lanes,
// in ascending lane order.

@(private)
jit_help_store_buf :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	if exec == 0 || int(in_.k) >= len(ctx.opts.buffers) {
		return
	}
	store_buf_lanes(ctx.slots, ctx.opts.buffers[in_.k], in_^, exec)
}

@(private)
jit_help_image_load :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	image_load_lanes(ctx.slots, &ctx.opts.images[in_.k], in_^, exec)
}

@(private)
jit_help_image_store :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	image_store_lanes(ctx.slots, &ctx.opts.images[in_.k], in_^, exec)
}

@(private)
jit_help_image_size :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	w, h: i32
	if int(in_.k) < len(ctx.opts.images) {
		im := ctx.opts.images[in_.k]
		w, h = i32(max(im.width, 0)), i32(max(im.height, 0))
	}
	ctx.slots[in_.dst] = u32(w)
	ctx.slots[in_.dst + 1] = u32(h)
}

@(private)
jit_help_load_shared :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots, sh := ctx.slots, ctx.shared
	dst, c := in_.dst, in_.c
	if in_.k == 0 {
		for i in 0 ..< i32(in_.n) {
			slots[dst + i] = sh[c + i]
		}
		return
	}
	off := &slots[in_.a]
	for i in 0 ..< i32(in_.n) {
		d := &slots[dst + i]
		for l in 0 ..< JIT_LANES {
			d[l] = sh[c + i + i32(off[l])]
		}
	}
}

@(private)
jit_help_store_shared :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	store_shared_lanes(ctx.slots, ctx.shared, in_^, exec)
}

@(private)
jit_help_atomic_shared :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	atomic_shared_lanes(ctx.slots, ctx.shared, in_^, exec)
}

@(private)
jit_help_atomic_buf :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	words := ctx.opts.buffers[in_.d] if int(in_.d) < len(ctx.opts.buffers) else nil
	atomic_buf_lanes(ctx.slots, words, in_^, exec)
}

// ---- IR -> AVX2 lowering ----------------------------------------------------------------

// Whether nothing reads `slot` after this instruction writes it.
@(private)
dead_write :: #force_inline proc(e: ^Jit_Emitter, slot: i32) -> bool {
	return write_lu(e, slot) <= e.pc && !need_mem_write(e, slot)
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
			vroundps(&e.buf, 3, 1, 0x0B)                     // t = trunc(x), x >= 0
			vop_ps(&e.buf, 0x5C, 1, 1, 3)                    // x - t
			vcmpps_tab(&e.buf, 1, 1, OFF_TAB_HALF_F, 0x1D)   // >= 0.5
			vop_ps_tab(&e.buf, 0x54, 1, 1, OFF_TAB_ONE_F)
			vop_ps(&e.buf, 0x58, 3, 3, 1)                    // t + 1 where rounding up
			emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 3, 0, 3)   // vcvttps2dq
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
Jump_Patch :: struct {
	patch_pos: int,
	target_pc: i32,
}

@(private)
can_fuse_cmp_mask_cond :: #force_inline proc(e: ^Jit_Emitter, code: []Instr, pc: int, is_init: bool) -> (pc_mc: int, ok: bool) {
	if is_init || code[pc].n != 1 {
		return -1, false
	}
	cond := code[pc].dst
	if (e.mem_mask[pc] & 1) != 0 {
		return -1, false
	}
	n_code := len(code)
	bb_start := e.bb_start[:n_code + 1]
	pc_mc = -1
	if pc + 1 < n_code && !bb_start[pc + 1] {
		if code[pc + 1].op == .Mask_Cond {
			pc_mc = pc + 1
		} else if code[pc + 1].op == .Mask_Copy && pc + 2 < n_code && !bb_start[pc + 2] && code[pc + 2].op == .Mask_Cond {
			pc_mc = pc + 2
		}
	}
	if pc_mc < 0 || code[pc_mc].b != cond || write_lu(e, cond) != i32(pc_mc) {
		return -1, false
	}
	return pc_mc, true
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
bset_set :: #force_inline proc(bits: []u64, slot: i32) {
	u := uint(slot)
	bits[u >> 6] |= u64(1) << (u & 63)
}

@(private)
bset_clear :: #force_inline proc(bits: []u64, slot: i32) {
	u := uint(slot)
	bits[u >> 6] &~= u64(1) << (u & 63)
}

@(private)
bset_test :: #force_inline proc(bits: []u64, slot: i32) -> bool {
	u := uint(slot)
	return (bits[u >> 6] & (u64(1) << (u & 63))) != 0
}

// Backward CFG liveness over basic blocks: marks `e.mem_mask[pc]` bit `k` iff
// `writes.start + k` is live out of `pc`'s block without a subsequent full_write
// in the same block (or is in `p.dyn_slots` / `is_init`).
@(private)
jit_compute_liveness :: proc(e: ^Jit_Emitter, p: ^Program, code: []Instr, is_init: bool) {
	n_code := len(code)
	if n_code == 0 {
		return
	}
	mem_mask := e.mem_mask[:n_code]
	tex_mask := e.tex_mask[:n_code]
	if is_init {
		for &mm in mem_mask {
			mm = 0xFFFF
		}
		for &tm in tex_mask {
			tm = 0xF
		}
		return
	}

	bb_start := e.bb_start[:n_code + 1]
	rw := e.rw[:n_code]
	pc_to_bb := e.pc_to_bb[:n_code + 1]
	n_bb := 0
	for pc in 0 ..< n_code {
		if bb_start[pc] {
			n_bb += 1
		}
		pc_to_bb[pc] = i32(n_bb - 1)
	}
	exit_bb := i32(n_bb)
	pc_to_bb[n_code] = exit_bb

	heap := runtime.default_allocator()
	bb_meta := make([][4]i32, n_bb, heap)
	defer delete(bb_meta, heap)

	bi := 0
	for pc := 0; pc < n_code; {
		end := pc + 1
		for end < n_code && !bb_start[end] {
			end += 1
		}
		last := code[end - 1]
		s0 := i32(bi + 1)
		s1 := i32(-1)
		#partial switch last.op {
		case .Jump:
			s0 = pc_to_bb[last.a] if last.a >= 0 && int(last.a) <= n_code else exit_bb
		case .Jump_If_None:
			s1 = pc_to_bb[last.a] if last.a >= 0 && int(last.a) <= n_code else exit_bb
		case .Mask_Kill:
			if last.dst == MASK_DISCARD {
				s1 = exit_bb
			}
		}
		bb_meta[bi] = {i32(pc), i32(end), s0, s1}
		bi += 1
		pc = end
	}

	n_words := (max(p.num_slots, 1) + 63) / 64
	total_words := (3 * n_bb + 4) * n_words
	words_buf := make([]u64, total_words, heap)
	defer delete(words_buf, heap)

	live_in := words_buf[:(n_bb + 1) * n_words]
	off1 := (n_bb + 1) * n_words
	bb_use := words_buf[off1 : off1 + n_bb * n_words]
	off2 := off1 + n_bb * n_words
	bb_def := words_buf[off2 : off2 + n_bb * n_words]
	off3 := off2 + n_bb * n_words
	always_mem := words_buf[off3 : off3 + n_words]
	cur := words_buf[off3 + n_words : off3 + 2 * n_words]
	live := words_buf[off3 + 2 * n_words : off3 + 3 * n_words]

	for ds in p.dyn_slots {
		for s in ds.start ..< ds.start + ds.n {
			bset_set(always_mem, s)
		}
	}
	exit_in := live_in[n_bb * n_words : (n_bb + 1) * n_words]
	for v in p.outputs {
		for k in 0 ..< v.n {
			bset_set(exit_in, v.slot + k)
		}
	}

	for b in 0 ..< n_bb {
		u := bb_use[b * n_words : (b + 1) * n_words]
		d := bb_def[b * n_words : (b + 1) * n_words]
		first, end := int(bb_meta[b][0]), int(bb_meta[b][1])
		for pc := end - 1; pc >= first; pc -= 1 {
			info := rw[pc]
			if info.full_write {
				for s in info.writes.start ..< info.writes.start + info.writes.n {
					bset_clear(u, s)
					bset_set(d, s)
				}
			}
			for r in info.reads {
				for s in r.start ..< r.start + r.n {
					bset_set(u, s)
					bset_clear(d, s)
				}
			}
		}
		copy(live_in[b * n_words : (b + 1) * n_words], u)
	}

	for {
		changed := false
		for b := n_bb - 1; b >= 0; b -= 1 {
			s0, s1 := int(bb_meta[b][2]), int(bb_meta[b][3])
			base := b * n_words
			s0_base := s0 * n_words
			if s1 >= 0 {
				s1_base := s1 * n_words
				for w in 0 ..< n_words {
					lo := live_in[s0_base + w] | live_in[s1_base + w]
					li := (lo & ~bb_def[base + w]) | bb_use[base + w]
					if li != live_in[base + w] {
						live_in[base + w] = li
						changed = true
					}
				}
			} else {
				for w in 0 ..< n_words {
					lo := live_in[s0_base + w]
					li := (lo & ~bb_def[base + w]) | bb_use[base + w]
					if li != live_in[base + w] {
						live_in[base + w] = li
						changed = true
					}
				}
			}
		}
		if !changed {
			break
		}
	}

	for b in 0 ..< n_bb {
		s0, s1 := int(bb_meta[b][2]), int(bb_meta[b][3])
		s0_base := s0 * n_words
		for w in 0 ..< n_words {
			lo := live_in[s0_base + w]
			if s1 >= 0 {
				lo |= live_in[s1 * n_words + w]
			}
			v := lo | always_mem[w]
			cur[w] = v
			live[w] = v
		}
		first, end := int(bb_meta[b][0]), int(bb_meta[b][1])
		for pc := end - 1; pc >= first; pc -= 1 {
			info := rw[pc]
			op := code[pc].op
			if op == .Texture || op == .Texture_Bias || op == .Texel_Fetch {
				tm: u8 = 0
				for k in 0 ..< 4 {
					if bset_test(live, info.writes.start + i32(k)) {
						tm |= u8(1) << uint(k)
					}
				}
				tex_mask[pc] = tm
			}
			wn := int(info.writes.n)
			mask: u16 = 0
			if wn > 0 {
				lim := min(wn, 16)
				ws := info.writes.start
				for k in 0 ..< lim {
					if bset_test(cur, ws + i32(k)) {
						mask |= u16(1) << uint(k)
					}
				}
				if info.full_write {
					for k in 0 ..< wn {
						s := ws + i32(k)
						if !bset_test(always_mem, s) {
							bset_clear(cur, s)
							bset_clear(live, s)
						}
					}
				}
			}
			for r in info.reads {
				for s in r.start ..< r.start + r.n {
					bset_set(live, s)
				}
			}
			mem_mask[pc] = mask
		}
	}
}

@(private)
jit_emit_fn :: proc(e: ^Jit_Emitter, p: ^Program, code: []Instr, is_init: bool) {
	n_code := len(code)
	bb_start := e.bb_start[:n_code + 1]
	for &b in bb_start {
		b = false
	}
	for in_ in code {
		if (in_.op == .Jump || in_.op == .Jump_If_None) && in_.a >= 0 && int(in_.a) <= n_code {
			bb_start[in_.a] = true
		}
	}

	// From pc = 0 to the first jump target or the first write to MASK_EXEC,
	// MASK_EXEC is provably the entry mask (`ctx.lanes`), so Store and Store_Zero
	// run unmasked: lanes outside the entry mask never execute, derivatives are
	// unaffected because fragment quads are whole (`0x0F`, `0xF0`, or `0xFF`),
	// and runner output write-back masks by alive entry lanes.
	entry_exec_end := 0
	if !is_init {
		for pc in 0 ..< n_code {
			if bb_start[pc] {
				break
			}
			in_ := code[pc]
			if in_.op == .Jump || in_.op == .Mask_Kill {
				break
			}
			if (in_.op == .Mask_Cond || in_.op == .Mask_Restore || in_.op == .Mask_Or || in_.op == .Mask_Clear) && in_.dst == MASK_EXEC {
				break
			}
			if in_.op == .Mask_Copy && in_.dst == MASK_EXEC && in_.a != MASK_EXEC {
				break
			}
			entry_exec_end = pc + 1
		}
	}

	bb_start[0] = true
	rw := e.rw[:n_code]
	for in_, pc in code {
		w, fw, r := instr_read_write(in_)
		if pc < entry_exec_end {
			if in_.op == .Store {
				w, fw, r = {in_.dst, i32(in_.n)}, true, {{in_.a, i32(in_.n)}, {}, {}, {}}
			} else if in_.op == .Store_Zero {
				w, fw, r = {in_.dst, i32(in_.n)}, true, {}
			}
		}
		rw[pc] = {w, fw, r}
		if in_.op == .Jump || in_.op == .Jump_If_None {
			bb_start[pc + 1] = true
		} else if in_.op == .Mask_Kill {
			bb_start[pc + 1] = true
		} else if is_helper_instr(p, in_) {
			bb_start[pc] = true
			bb_start[pc + 1] = true
		}
	}
	bb_start[n_code] = true

	jit_compute_liveness(e, p, code, is_init)

	for &lu in e.last_use {
		lu = -1
	}

	for &mw in e.mask_written {
		mw = false
	}
	for &mg in e.mask_eq_gen {
		mg = 0
	}
	e.exec_gen = 0
	e.mask_written[MASK_EXEC] = true
	for in_ in code {
		#partial switch in_.op {
		case .Mask_Kill, .Mask_Or, .Mask_Cond:
			if in_.dst >= 0 {
				e.mask_written[in_.dst] = true
			}
		case .Mask_Copy, .Mask_Restore:
			if in_.dst >= 0 && in_.dst != in_.a {
				e.mask_written[in_.dst] = true
			}
		}
	}

	pc_off := e.pc_off[:n_code + 1]
	clear(&e.patches)
	clear(&e.epilogues)
	clear(&e.bilinear_calls)
	clear(&e.helper_ymm_calls)
	clear(&e.unif_oob_calls)
	clear(&e.big_buf_calls)
	clear(&e.bound_tex)
	clear(&e.load_bufs)
	e.epilogue_off = 0
	e.bilinear_off = 0
	e.helper_ymm_off = 0
	e.unif_oob_off = 0
	e.big_buf_off = 0

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
		if bb_start[pc] {
			cache_reset(e)
			end := pc + 1
			for end < n_code && !bb_start[end] {
				end += 1
			}
			clear(&e.def_lu)
			for q in pc ..< end {
				e.def_off[q] = i32(len(e.def_lu))
				wn := int(rw[q].writes.n)
				if wn > 0 {
					resize(&e.def_lu, len(e.def_lu) + wn)
				}
				for r in rw[q].reads {
					for s in r.start ..< r.start + r.n {
						e.last_use[s] = -1
					}
				}
				for s in rw[q].writes.start ..< rw[q].writes.start + rw[q].writes.n {
					e.last_use[s] = -1
				}
			}
			for q := end - 1; q >= pc; q -= 1 {
				info := rw[q]
				doff := e.def_off[q]
				for k in i32(0) ..< info.writes.n {
					e.def_lu[doff + k] = e.last_use[info.writes.start + k]
					if info.full_write {
						e.last_use[info.writes.start + k] = -1
					}
				}
				for r in info.reads {
					for s in r.start ..< r.start + r.n {
						if e.last_use[s] < 0 {
							e.last_use[s] = i32(q)
						}
					}
				}
			}
		} else {
			prev := rw[pc - 1]
			doff := e.def_off[pc - 1]
			for k in i32(0) ..< prev.writes.n {
				e.last_use[prev.writes.start + k] = e.def_lu[doff + k]
			}
		}

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
				// Half away from zero, as math.round: t = trunc(x), and where
				// |x - t| >= 0.5 (exact), t + copysign(1, x); NaN and inf keep t.
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vroundps(&e.buf, 2, ya, 0x0B)
					vop_ps(&e.buf, 0x5C, 3, ya, 2)                   // x - t
					vop_ps_tab(&e.buf, 0x54, 3, 3, OFF_TAB_ABS)
					vcmpps_tab(&e.buf, 3, 3, OFF_TAB_HALF_F, 0x1D)   // |x - t| >= 0.5 (GE_OQ)
					vop_ps_tab(&e.buf, 0x54, 4, ya, OFF_TAB_SIGN)
					vop_ps_tab(&e.buf, 0x56, 4, 4, OFF_TAB_ONE_F)    // copysign(1, x)
					vop_ps(&e.buf, 0x58, 4, 2, 4)
					yd := prep_dst(e, dst + i, 0, pin_bit(ya))
					vblendvps(&e.buf, yd, 2, 4, 3)
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
			case .Sin:
				emit_helper_call(&e.buf, jit_help_sin, &code[pc])
			case .Cos:
				emit_helper_call(&e.buf, jit_help_cos, &code[pc])
			case:
				emit_helper_call(&e.buf, jit_help_fn1, &code[pc])
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
			case .Atan2:
				emit_helper_call(&e.buf, jit_help_atan2, &code[pc])
			case:
				emit_helper_call(&e.buf, jit_help_pow, &code[pc])
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
				ya0 := read_slot(e, a, 0)
				ya1 := read_slot(e, a + 1, 1, pin_bit(ya0))
				if ya0 != 0 {
					vmovaps_rr(&e.buf, 0, ya0)
				}
				if ya1 != 1 {
					vmovaps_rr(&e.buf, 1, ya1)
				}
			}
			chan_disp := (i32(in_.k) - SAMPLER_COUNT) * size_of(Chan_Img)
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
		chan_disp := (i32(in_.k) - SAMPLER_COUNT) * size_of(Chan_Img)
		if in_.op == .Texel_Fetch {
			if (in_.d & 1) != 0 {
				vpbroadcastd_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_MAX_Y)
				vop_d(&e.buf, 0xFA, 1, 6, 1) // max_y - py
			}
			vpbroadcastd_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_MAX_X)
			vop38_d(&e.buf, 0x3B, 2, 0, 6) // min_u(px, max_x) == px iff u32(px) <= u32(max_x)
			vop_d(&e.buf, 0x76, 2, 2, 0)
			vpbroadcastd_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_MAX_Y)
			vop38_d(&e.buf, 0x3B, 3, 1, 6)
			vop_d(&e.buf, 0x76, 3, 3, 1)
			vop_d(&e.buf, 0xDB, 5, 2, 3) // ymm5 = in_bounds
			vpbroadcastd_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_W)
			vop38_d(&e.buf, 0x40, 1, 1, 6)
			vop_d(&e.buf, 0xFE, 4, 1, 0)
			emit_bytes(&e.buf, 0x0F, 0xB6)
			emit_modrm_mem(&e.buf, GPR_RCX, GPR_RBX, chan_disp + OFF_CHAN_SHIFT)
			emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x6E, 6, 0, GPR_RCX)
			emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 6, 0, 6)
			vop38_d(&e.buf, 0x47, 4, 4, 6)
			emit_chan_gather_rgba(&e.buf, GPR_RBX, chan_disp, 4, 5, stub.comp_mask)
		} else {
			emit_b(&e.buf, 0x80) // cmp byte [rbx + chan_disp + OFF_CHAN_FILTER], 0
			emit_modrm_mem(&e.buf, 7, GPR_RBX, chan_disp + OFF_CHAN_FILTER)
			emit_b(&e.buf, 0)
			emit_bytes(&e.buf, 0x0F, 0x85) // jne linear
			lin_patch := len(e.buf)
			emit_i32(&e.buf, 0)

			// Nearest fast path.
			vbroadcastss_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_FW)
			vop_ps(&e.buf, 0x59, 0, 0, 6)
			vbroadcastss_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_FH)
			vop_ps(&e.buf, 0x59, 1, 1, 6)
			vroundps(&e.buf, 0, 0, 0x09)
			vroundps(&e.buf, 1, 1, 0x09)
			vzero(&e.buf, 6)
			vop_ps(&e.buf, 0x5F, 0, 0, 6)
			vop_ps(&e.buf, 0x5F, 1, 1, 6)
			vop_ps_tab(&e.buf, 0x5D, 0, 0, OFF_TAB_I32MAX)
			vop_ps_tab(&e.buf, 0x5D, 1, 1, OFF_TAB_I32MAX)
			emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 0, 0, 0)
			emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 1, 0, 1)
			vpbroadcastd_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_MAX_X)
			vop38_d(&e.buf, 0x39, 0, 0, 6)
			vpbroadcastd_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_MAX_Y)
			vop38_d(&e.buf, 0x39, 1, 1, 6)
			vpbroadcastd_mem(&e.buf, 6, GPR_RBX, chan_disp + OFF_CHAN_W)
			vop38_d(&e.buf, 0x40, 1, 1, 6)
			vop_d(&e.buf, 0xFE, 4, 1, 0)
			emit_bytes(&e.buf, 0x0F, 0xB6)
			emit_modrm_mem(&e.buf, GPR_RCX, GPR_RBX, chan_disp + OFF_CHAN_SHIFT)
			emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x6E, 6, 0, GPR_RCX)
			emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 6, 0, 6)
			vop38_d(&e.buf, 0x47, 4, 4, 6)
			emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 5, 0, GPR_R14, OFF_TAB_ALL1)
			emit_chan_gather_rgba(&e.buf, GPR_RBX, chan_disp, 4, 5, stub.comp_mask)
			emit_b(&e.buf, 0xE9) // jmp bound_done
			done_near := len(e.buf)
			emit_i32(&e.buf, 0)

			// Bilinear path via per-function register-ABI subroutine.
			patch_i32(e.buf[:], lin_patch, i32(len(e.buf) - (lin_patch + 4)))
			emit_bytes(&e.buf, 0x48, 0x8D) // lea rdi, [rbx + chan_disp]
			emit_modrm_mem(&e.buf, GPR_RDI, GPR_RBX, chan_disp)
			emit_b(&e.buf, 0xBA)           // mov edx, u32(stub.comp_mask)
			emit_u32(&e.buf, u32(stub.comp_mask))
			emit_b(&e.buf, 0xE8)           // call bilinear_sub
			append(&e.bilinear_calls, len(e.buf))
			emit_i32(&e.buf, 0)

			patch_i32(e.buf[:], done_near, i32(len(e.buf) - (done_near + 4)))
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
	if len(e.bilinear_calls) > 0 {
		bilinear_off := len(e.buf)
		e.bilinear_off = bilinear_off
		emit_bilinear_sub(&e.buf)
		for cp in e.bilinear_calls {
			patch_i32(e.buf[:], cp, i32(bilinear_off - (cp + 4)))
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

// Whether a compute program runs in the JIT. Its 8-lane batches run each
// workgroup batch by batch, where the interpreter runs 64-lane chunks
// instruction by instruction, so only results that do not depend on the order
// between invocations of a workgroup are the same: a workgroup of at most 8
// invocations is one batch, in the interpreter's order, so barriers are no-ops
// and atomics apply in the same order; larger ones run here only without
// barrier() (which needs every batch to reach it before any passes) and
// without atomics (whose returned values and non-commutative results follow
// the order lanes apply them).
@(private)
jit_compute_eligible :: proc(p: ^Program) -> bool {
	ls := [3]u32{max(p.local_size[0], 1), max(p.local_size[1], 1), max(p.local_size[2], 1)}
	if ls.x * ls.y * ls.z <= JIT_LANES {
		return true
	}
	if p.has_barrier {
		return false
	}
	for in_ in p.code {
		if in_.op == .Atomic_Shared || in_.op == .Atomic_Buf {
			return false
		}
	}
	return true
}

@(private)
jit_compile :: proc(p: ^Program) {
	if !jit_avx2_ready || len(p.code) == 0 || (p.stage == .Compute && !jit_compute_eligible(p)) {
		return
	}
	heap := runtime.default_allocator()
	num_slots := max(p.num_slots, 1)
	max_code := max(p.init_len, len(p.code) - p.init_len)
	e := Jit_Emitter {
		buf              = make([dynamic]u8, 0, max(len(p.code) * 24, 256), heap),
		mem_mask         = make([]u16, max_code, heap),
		tex_mask         = make([]u8, max_code, heap),
		num_const        = make([]bool, num_slots, heap),
		pc_to_bb         = make([]i32, max_code + 1, heap),
		last_use         = make([]i32, num_slots, heap),
		def_off          = make([]i32, max_code, heap),
		def_lu           = make([dynamic]i32, 0, 256, heap),
		bb_start         = make([]bool, max_code + 1, heap),
		mask_written     = make([]bool, max(p.num_masks, 2), heap),
		mask_eq_gen      = make([]u32, max(p.num_masks, 2), heap),
		pc_off           = make([]int, max_code + 1, heap),
		rw               = make([]Instr_RW, max_code, heap),
		patches          = make([dynamic]Jump_Patch, 0, 64, heap),
		epilogues        = make([dynamic]int, 0, 8, heap),
		bilinear_calls   = make([dynamic]int, 0, 8, heap),
		helper_ymm_calls = make([dynamic]int, 0, 8, heap),
		unif_oob_calls   = make([dynamic]int, 0, 8, heap),
		big_buf_calls    = make([dynamic]int, 0, 8, heap),
		bound_tex        = make([dynamic]Bound_Tex_Stub, 0, 8, heap),
		load_bufs        = make([dynamic]Load_Buf_Stub, 0, 8, heap),
	}
	defer delete(e.buf)
	defer delete(e.mem_mask, heap)
	defer delete(e.tex_mask, heap)
	defer delete(e.num_const, heap)
	for c in p.consts {
		e.num_const[c[0]] = (c[1] & 0x7fff_ffff) <= 0x7f80_0000
	}
	defer delete(e.pc_to_bb, heap)
	defer delete(e.last_use, heap)
	defer delete(e.def_off, heap)
	defer delete(e.def_lu)
	defer delete(e.bb_start, heap)
	defer delete(e.mask_written, heap)
	defer delete(e.mask_eq_gen, heap)
	defer delete(e.pc_off, heap)
	defer delete(e.rw, heap)
	defer delete(e.patches)
	defer delete(e.epilogues)
	defer delete(e.bilinear_calls)
	defer delete(e.helper_ymm_calls)
	defer delete(e.unif_oob_calls)
	defer delete(e.big_buf_calls)
	defer delete(e.bound_tex)
	defer delete(e.load_bufs)

	has_init := p.init_len > 0
	init_bytes := 0
	if has_init {
		jit_emit_fn(&e, p, p.code[:p.init_len], true)
		init_bytes = len(e.buf)
		for (len(e.buf) & 31) != 0 {
			append(&e.buf, 0xCC)
		}
	}
	body_off := len(e.buf)
	jit_emit_fn(&e, p, p.code[p.init_len:], false)

	total := len(e.buf)
	mem_slice, err := virtual.reserve_and_commit(uint(total))
	if err != .None || len(mem_slice) < total {
		return
	}
	copy(mem_slice, e.buf[:])
	if !virtual.protect(raw_data(mem_slice), uint(len(mem_slice)), {.Read, .Execute}) {
		virtual.release(raw_data(mem_slice), uint(len(mem_slice)))
		return
	}
	jit_dump_program(p, &e, mem_slice, init_bytes, body_off, total)

	meta := new(Jit_Meta, heap)
	for in_ in p.code {
		#partial switch in_.op {
		case .Texture, .Texture_Bias, .Texel_Fetch, .Texture_Size:
			if in_.k + 1 > meta.num_chans {
				meta.num_chans = in_.k + 1
			}
		case .Load_Buf, .Store_Buf:
			if in_.k < 4 && in_.k + 1 > meta.num_buf_binds {
				meta.num_buf_binds = in_.k + 1
			}
		}
	}

	if p.stage == .Fragment {
		spans := make([dynamic]Jit_Var_Span, 0, len(p.inputs), heap)
		for v in p.inputs {
			#partial switch v.kind {
			case .Stage_In:
				if v.n <= 0 {
					continue
				}
				meta.max_var_col = max(meta.max_var_col, v.offset + v.n)
				if len(spans) > 0 {
					last := &spans[len(spans) - 1]
					if last.offset + last.n == v.offset && last.slot + last.n == v.slot {
						last.n += v.n
						continue
					}
				}
				append(&spans, Jit_Var_Span{offset = v.offset, slot = v.slot, n = v.n})
			case .Frag_Coord:
				meta.fc_slot = v.slot
				meta.fc_n = v.n
				meta.fc_base = v.base
			case .Front_Facing:
				meta.ff_slot = v.slot
				meta.ff_n = v.n
				meta.ff_base = v.base
			}
		}
		meta.var_spans = spans[:]
		for v in p.outputs {
			if v.kind == .Stage_Out && int(v.location) < len(meta.outs) {
				meta.outs[v.location] = v
			} else if v.kind == .Sample_Mask {
				meta.has_sample_mask = true
				meta.sample_mask_slot = v.slot
			}
		}
	}

	p.jit_meta = meta
	p.jit_mem = mem_slice
	base := uintptr(raw_data(mem_slice))
	if has_init {
		p.jit_init = rawptr(base)
	}
	p.jit_body = rawptr(base + uintptr(body_off))
}

// ---- 8-lane runners ---------------------------------------------------------------------

@(private, enable_target_feature = "avx2")
jit_begin :: proc(p: ^Program, scratch: ^Scratch, uniforms: []u8, opts: ^Run_Options) -> Jit_Ctx #no_bounds_check {
	needed_regs := (max(p.num_slots, 1) + SAMPLER_COUNT + 7) / 8 + 1
	if len(scratch.regs) < needed_regs {
		resize(&scratch.regs, needed_regs)
	}
	needed_masks := (max(p.num_masks, 2) + 7) / 8
	if len(scratch.masks) < needed_masks {
		resize(&scratch.masks, needed_masks)
	}
	meta := (^Jit_Meta)(p.jit_meta)
	aligned := (uintptr(raw_data(scratch.regs)) + 31) &~ uintptr(31)
	if meta.num_chans > 0 {
		chan_imgs := (^[SAMPLER_COUNT]Chan_Img)(aligned)
		n_ch := int(meta.num_chans)
		for i in 0 ..< n_ch {
			chan_imgs[i].ptr = nil
		}
		if len(opts.channel_images) > 0 {
			fill_chan_imgs(chan_imgs, opts.channel_images[:min(len(opts.channel_images), n_ch)])
		}
	}
	slots := ([^]Jit_Slot)(aligned + SAMPLER_COUNT * size_of(Chan_Img))
	masks := ([^]u8)(raw_data(scratch.masks))

	ctx := Jit_Ctx {
		slots = slots,
		masks = masks,
		opts  = opts,
		lanes = 0xFF,
	}
	if meta.num_chans > 0 && opts.sampler != nil {
		ctx.host_ctx = context
	}
	if meta.num_buf_binds > 0 {
		n_buf := min(len(opts.buffers), int(meta.num_buf_binds))
		for i in 0 ..< n_buf {
			b := opts.buffers[i]
			ctx.buf_ptr[i] = raw_data(b)
			// Word indices wrap in 32 bits, so clamping len at 2^32 is exact for all u32 indices.
			ctx.buf_len[i] = min(uint(len(b)), uint(1) << 32)
		}
	}

	for c in p.consts {
		slots[c[0]] = c[1]
	}
	for f in p.fills {
		bits: u32
		if int(f.offset) + 4 <= len(uniforms) {
			bits = (^u32)(&uniforms[f.offset])^
		}
		if f.base == .Bool {
			bits = u32(bits != 0)
		}
		slots[f.slot] = bits
	}
	if p.jit_init != nil {
		masks[MASK_DISCARD] = 0
		(Jit_Fn(p.jit_init))(&ctx)
	}
	return ctx
}

@(private, enable_target_feature = "avx2")
jit_fill_slot :: #force_inline proc "contextless" (slot: ^Jit_Slot, col: Jit_Floats, base: Base) {
	#partial switch base {
	case .Int:
		for l in 0 ..< JIT_LANES {slot[l] = u32(f_to_i(col[l]))}
	case .Uint:
		for l in 0 ..< JIT_LANES {slot[l] = f_to_u(col[l])}
	case .Bool:
		for l in 0 ..< JIT_LANES {slot[l] = u32(col[l] != 0)}
	case:
		(^Jit_Floats)(slot)^ = col
	}
}

@(private)
Simd_F8 :: #simd[8]f32
@(private)
Simd_I8 :: #simd[8]i32

@(private, enable_target_feature = "avx2")
jit_pack_u8 :: #force_inline proc "contextless" (v: Simd_F8) -> Simd_U8 {
	z := Simd_F8(0)
	clamped := intrinsics.simd_select(intrinsics.simd_lanes_gt(v, z), intrinsics.simd_min(v, Simd_F8(1)), z)
	return transmute(Simd_U8)Simd_I8(clamped * Simd_F8(255) + Simd_F8(0.5))
}

@(private, rodata)
JIT_LX := [JIT_LANES]int{0, 1, 0, 1, 2, 3, 2, 3}
@(private, rodata)
JIT_LY := [JIT_LANES]int{0, 0, 1, 1, 0, 0, 1, 1}
@(private, rodata)
JIT_LXF := Jit_Floats{0.5, 1.5, 0.5, 1.5, 2.5, 3.5, 2.5, 3.5}
@(private, rodata)
JIT_LYF := Jit_Floats{0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5}
@(private, rodata)
JIT_ROW_LANES := [2][4]int{{0, 1, 4, 5}, {2, 3, 6, 7}}

@(private)
jit_validate_run :: proc(p: ^Program, out_x, out_y, out_stride, start_x, start_y, end_x, end_y: int, out: []u8, stride: int, targets: [][][4]f32) {
	first_px := (start_y - out_y) * out_stride + (start_x - out_x)
	last_px := (end_y - 1 - out_y) * out_stride + (end_x - 1 - out_x)
	for v in p.outputs {
		if v.kind != .Stage_Out {
			continue
		}
		if int(v.location) < len(targets) {
			if target := targets[v.location]; target != nil {
				_ = target[first_px]
				_ = target[last_px]
			}
		}
		if v.location == 0 && out != nil {
			_ = out[start_y * stride + start_x * 4]
			_ = out[start_y * stride + (end_x - 1) * 4 + 3]
			_ = out[(end_y - 1) * stride + start_x * 4]
			_ = out[(end_y - 1) * stride + (end_x - 1) * 4 + 3]
		}
	}
}

@(private)
jit_validate_vertex :: proc(p: ^Program, count: int, position: [][4]f32, varyings: []f32) {
	n := p.varying_floats
	for v in p.outputs {
		if v.kind == .Position {
			_ = position[count - 1]
		} else if v.n > 0 {
			_ = varyings[(count - 1) * n + int(v.offset) + int(v.n) - 1]
		}
	}
}

@(private)
jit_validate_fragment :: proc(p: ^Program, meta: ^Jit_Meta, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, sample_masks: []u32) {
	total := quads.count * 4
	_ = shaded[quads.count - 1]
	if quads.helpers != nil {
		_ = quads.helpers[quads.count - 1]
	}
	if meta.max_var_col > 0 {
		_ = quads.varyings[int(meta.max_var_col) * total - 1]
	}
	if meta.fc_n > 0 {
		_ = quads.origin[quads.count - 1]
		if meta.fc_n > 2 && quads.depth != nil {
			_ = quads.depth[total - 1]
		}
	}
	if meta.ff_n > 0 && quads.front_facing != nil {
		_ = quads.front_facing[quads.count - 1]
	}
	for target in targets {
		if target != nil {
			_ = target[total - 1]
		}
	}
	if sample_masks != nil {
		_ = sample_masks[total - 1]
	}
}

@(private, enable_target_feature = "avx2")
jit_runner_run :: proc(p: ^Program, uniforms: []u8, width, height: int, opts: Run_Options, out: []u8, stride: int, targets: [][][4]f32, scratch: ^Scratch) #no_bounds_check {
	x0, y0 := opts.x, opts.y
	w, h := opts.w, opts.h
	if w <= 0 || h <= 0 {
		x0, y0, w, h = 0, 0, width, height
	}
	start_x, start_y := max(x0, 0), max(y0, 0)
	end_x := min(x0 + w, width)
	end_y := min(y0 + h, height)
	if start_x >= end_x || start_y >= end_y {
		return
	}
	out_x, out_y, out_stride := 0, 0, width
	if opts.region_targets {
		out_x, out_y, out_stride = start_x, start_y, w
	}
	jit_validate_run(p, out_x, out_y, out_stride, start_x, start_y, end_x, end_y, out, stride, targets)
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	slots := ctx.slots
	masks := ctx.masks
	body_fn := Jit_Fn(p.jit_body)
	fw, fh := f32(width), f32(height)

	// Hoist constant Frag_Coord z/w and Front_Facing fills out of the tile loop.
	for v in p.inputs {
		if v.kind == .Frag_Coord && v.base == .Float {
			if v.n > 2 {(^Jit_Floats)(&slots[v.slot + 2])^ = 0}
			if v.n > 3 {(^Jit_Floats)(&slots[v.slot + 3])^ = 1}
		} else if v.kind == .Front_Facing {
			for k in 0 ..< int(v.n) {
				jit_fill_slot(&slots[v.slot + i32(k)], 1, v.base)
			}
		}
	}

	for by := start_y &~ 1; by < end_y; by += JIT_BATCH_H {
		py := f32(by) + JIT_LYF
		for v in p.inputs {
			if v.kind == .Frag_Coord && v.base == .Float && v.n > 1 {
				(^Jit_Floats)(&slots[v.slot + 1])^ = py
			}
		}
		for bx := start_x &~ 1; bx < end_x; bx += JIT_BATCH_W {
			px := f32(bx) + JIT_LXF
			for v in p.inputs {
				if v.kind == .Frag_Coord && v.base == .Float {
					(^Jit_Floats)(&slots[v.slot])^ = px
					continue
				}
				if v.kind == .Front_Facing {
					continue
				}
				for k in 0 ..< int(v.n) {
					col: Jit_Floats
					switch {
					case v.kind == .Frag_Coord:
						col = px if k == 0 else (py if k == 1 else (1 if k == 3 else 0))
					case int(v.location) < len(opts_local.varyings):
						vy := opts_local.varyings[v.location]
						col = vy.a[k] + vy.dx[k] * px + vy.dy[k] * py
					case v.location == 0 && k < 2:
						col = px / fw if k == 0 else py / fh
					}
					jit_fill_slot(&slots[v.slot + i32(k)], col, v.base)
				}
			}

			inside := bx >= start_x && by >= start_y && bx + JIT_BATCH_W <= end_x && by + JIT_BATCH_H <= end_y
			exec := u8(0xFF)
			if !inside {
				exec = 0
				if bx + 1 >= start_x && bx < end_x && by + 1 >= start_y && by < end_y {
					exec |= 0x0F
				}
				if bx + 3 >= start_x && bx + 2 < end_x && by + 1 >= start_y && by < end_y {
					exec |= 0xF0
				}
			}
			ctx.lanes = exec
			masks[MASK_DISCARD] = 0
			body_fn(&ctx)

			alive := ~masks[MASK_DISCARD]
			if alive == 0 {
				continue
			}
			full_tile := inside && alive == 0xFF
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
					r0 := (^Jit_Floats)(&slots[v.slot])
					r1 := (^Jit_Floats)(&slots[v.slot + 1])
					r2 := (^Jit_Floats)(&slots[v.slot + 2])
					r3 := (^Jit_Floats)(&slots[v.slot + 3])
					if target != nil {
						tptr := ([^][4]f32)(raw_data(target))
						for ry in 0 ..< 2 {
							base_px := (by + ry - out_y) * out_stride + (bx - out_x)
							lanes_r := JIT_ROW_LANES[ry]
							for rx in 0 ..< 4 {
								l := lanes_r[rx]
								tptr[base_px + rx] = {r0[l], r1[l], r2[l], r3[l]}
							}
						}
					}
					if bytes != nil {
						bptr := raw_data(bytes)
						rgba := jit_pack_u8((^Simd_F8)(r0)^) |
						        intrinsics.simd_shl(jit_pack_u8((^Simd_F8)(r1)^), Simd_U8(8)) |
						        intrinsics.simd_shl(jit_pack_u8((^Simd_F8)(r2)^), Simd_U8(16)) |
						        intrinsics.simd_shl(jit_pack_u8((^Simd_F8)(r3)^), Simd_U8(24))
						p8 := transmute([8]u32)rgba
						intrinsics.unaligned_store((^[4]u32)(&bptr[by * stride + bx * 4]), [4]u32{p8[0], p8[1], p8[4], p8[5]})
						intrinsics.unaligned_store((^[4]u32)(&bptr[(by + 1) * stride + bx * 4]), [4]u32{p8[2], p8[3], p8[6], p8[7]})
					}
					continue
				}
				for lane in 0 ..< JIT_LANES {
					x := bx + JIT_LX[lane]
					y := by + JIT_LY[lane]
					if x < start_x || y < start_y || x >= end_x || y >= end_y || (alive & (1 << uint(lane))) == 0 {
						continue
					}
					c := [4]f32{0, 0, 0, 1}
					for k in 0 ..< int(v.n) {
						c[k] = float_of(slots[v.slot + i32(k)][lane], v.base)
					}
					if target != nil {
						target[(y - out_y) * out_stride + (x - out_x)] = c
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

@(private, enable_target_feature = "avx2")
jit_run_vertex :: proc(p: ^Program, uniforms: []u8, attributes: []Attribute, count: int, position: [][4]f32, varyings: []f32, scratch: ^Scratch, opts: Run_Options) #no_bounds_check {
	jit_validate_vertex(p, count, position, varyings)
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	slots := ctx.slots
	masks := ctx.masks
	body_fn := Jit_Fn(p.jit_body)
	n := p.varying_floats

	for first := 0; first < count; first += JIT_LANES {
		lanes := min(count - first, JIT_LANES)
		for v in p.inputs {
			if v.kind == .Vertex_ID {
				reg := &slots[v.slot]
				for l in 0 ..< lanes {reg[l] = u32(i32(opts_local.first_vertex + first + l))}
				continue
			}
			a := attributes[v.location] if int(v.location) < len(attributes) else Attribute{}
			comps := 4
			if a.format <= .Uint4 {
				comps = int(a.format) % 4 + 1
			}
			for k in 0 ..< int(v.n) {
				reg := &slots[v.slot + i32(k)]
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
		ctx.lanes = u8((u32(1) << uint(lanes)) - 1)
		masks[MASK_DISCARD] = 0
		body_fn(&ctx)

		for v in p.outputs {
			if v.kind == .Position {
				for k in 0 ..< 4 {
					reg := (^Jit_Floats)(&slots[v.slot + i32(k)])
					for l in 0 ..< lanes {position[first + l][k] = reg[l]}
				}
				continue
			}
			for k in 0 ..< int(v.n) {
				reg := &slots[v.slot + i32(k)]
				out := varyings[first * n + int(v.offset) + k:]
				for l in 0 ..< lanes {out[l * n] = fval(reg[l])}
			}
		}
	}
}

@(private)
Simd_F4 :: #simd[4]f32

@(private)
Jit_Out_Target :: struct {
	dst: [^][4]f32,
	c0:  ^Jit_Floats,
	c1:  ^Jit_Floats,
	c2:  ^Jit_Floats,
	c3:  ^Jit_Floats,
}

@(private, enable_target_feature = "avx2")
jit_store_pair :: #force_inline proc "contextless" (t: Jit_Out_Target, base_out: int) #no_bounds_check {
	r := (^Simd_F8)(t.c0)^
	g := (^Simd_F8)(t.c1)^
	b := (^Simd_F8)(t.c2)^
	a := (^Simd_F8)(t.c3)^
	rg_lo := intrinsics.simd_shuffle(r, g, 0, 8, 1, 9, 4, 12, 5, 13)
	ba_lo := intrinsics.simd_shuffle(b, a, 0, 8, 1, 9, 4, 12, 5, 13)
	rg_hi := intrinsics.simd_shuffle(r, g, 2, 10, 3, 11, 6, 14, 7, 15)
	ba_hi := intrinsics.simd_shuffle(b, a, 2, 10, 3, 11, 6, 14, 7, 15)
	p04 := intrinsics.simd_shuffle(rg_lo, ba_lo, 0, 1, 8, 9, 4, 5, 12, 13)
	p15 := intrinsics.simd_shuffle(rg_lo, ba_lo, 2, 3, 10, 11, 6, 7, 14, 15)
	p26 := intrinsics.simd_shuffle(rg_hi, ba_hi, 0, 1, 8, 9, 4, 5, 12, 13)
	p37 := intrinsics.simd_shuffle(rg_hi, ba_hi, 2, 3, 10, 11, 6, 7, 14, 15)
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 0]), intrinsics.simd_shuffle(p04, p15, 0, 1, 2, 3, 8, 9, 10, 11))
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 2]), intrinsics.simd_shuffle(p26, p37, 0, 1, 2, 3, 8, 9, 10, 11))
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 4]), intrinsics.simd_shuffle(p04, p15, 4, 5, 6, 7, 12, 13, 14, 15))
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 6]), intrinsics.simd_shuffle(p26, p37, 4, 5, 6, 7, 12, 13, 14, 15))
}

@(private, enable_target_feature = "avx2")
jit_store_quad :: #force_inline proc "contextless" (t: Jit_Out_Target, base_out, qi: int, qalive: u8) #no_bounds_check {
	r := ((^[2]Simd_F4)(t.c0))[qi]
	g := ((^[2]Simd_F4)(t.c1))[qi]
	b := ((^[2]Simd_F4)(t.c2))[qi]
	a := ((^[2]Simd_F4)(t.c3))[qi]
	rg_lo := intrinsics.simd_shuffle(r, g, 0, 4, 1, 5)
	ba_lo := intrinsics.simd_shuffle(b, a, 0, 4, 1, 5)
	rg_hi := intrinsics.simd_shuffle(r, g, 2, 6, 3, 7)
	ba_hi := intrinsics.simd_shuffle(b, a, 2, 6, 3, 7)
	p0 := intrinsics.simd_shuffle(rg_lo, ba_lo, 0, 1, 4, 5)
	p1 := intrinsics.simd_shuffle(rg_lo, ba_lo, 2, 3, 6, 7)
	p2 := intrinsics.simd_shuffle(rg_hi, ba_hi, 0, 1, 4, 5)
	p3 := intrinsics.simd_shuffle(rg_hi, ba_hi, 2, 3, 6, 7)
	if qalive == 0xF {
		intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 0]), p0)
		intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 1]), p1)
		intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 2]), p2)
		intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 3]), p3)
		return
	}
	if (qalive & 1) != 0 {intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 0]), p0)}
	if (qalive & 2) != 0 {intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 1]), p1)}
	if (qalive & 4) != 0 {intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 2]), p2)}
	if (qalive & 8) != 0 {intrinsics.unaligned_store((^Simd_F4)(&t.dst[base_out + 3]), p3)}
}

@(private)
JIT_FRAG_DX :: Simd_F8{0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5}
@(private)
JIT_FRAG_DY :: Simd_F8{0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5}

@(private)
jit_write_fragment_slow :: proc(meta: ^Jit_Meta, slots: [^]Jit_Slot, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, sample_masks: []u32, first, nq: int, alive: u8) #no_bounds_check {
	if alive == 0 {
		shaded[first] = {}
		if nq == 2 {
			shaded[first + 1] = {}
		}
		return
	}
	for qi in 0 ..< nq {
		q := first + qi
		lanes_out: Quad_Lanes
		helpers := quads.helpers[q] if quads.helpers != nil else {}
		for i in 0 ..< 4 {
			l := qi * 4 + i
			if i in helpers || (alive & (1 << uint(l))) == 0 {
				continue
			}
			lanes_out += {i}
			for target, n_t in targets {
				if target == nil {
					continue
				}
				c := [4]f32{0, 0, 0, 1}
				if n_t < len(meta.outs) {
					v := meta.outs[n_t]
					for k in 0 ..< int(v.n) {
						c[k] = float_of(slots[v.slot + i32(k)][l], v.base)
					}
				}
				target[q * 4 + i] = c
			}
			if sample_masks != nil {
				sample_masks[q * 4 + i] = slots[meta.sample_mask_slot][l] if meta.has_sample_mask else ~u32(0)
			}
		}
		shaded[q] = lanes_out
	}
}

@(private, enable_target_feature = "avx2")
jit_run_fragment :: proc(p: ^Program, uniforms: []u8, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, scratch: ^Scratch, opts: Run_Options, sample_masks: []u32) #no_bounds_check {
	meta := (^Jit_Meta)(p.jit_meta)
	jit_validate_fragment(p, meta, quads, targets, shaded, sample_masks)
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	slots := ctx.slots
	masks := ctx.masks
	body_fn := Jit_Fn(p.jit_body)
	total := quads.count * 4

	active_outs: [FRAGMENT_OUTPUTS]Jit_Out_Target
	n_active := 0
	fast_outs := sample_masks == nil && len(targets) <= FRAGMENT_OUTPUTS
	if fast_outs {
		for target, n_t in targets {
			if target == nil {
				continue
			}
			v := meta.outs[n_t]
			if v.n > 0 && v.base != .Float {
				fast_outs = false
				break
			}
			active_outs[n_active] = {
				dst = raw_data(target),
				c0  = (^Jit_Floats)(&slots[v.slot]) if v.n > 0 else &jit_tables.zero_f,
				c1  = (^Jit_Floats)(&slots[v.slot + 1]) if v.n > 1 else &jit_tables.zero_f,
				c2  = (^Jit_Floats)(&slots[v.slot + 2]) if v.n > 2 else &jit_tables.zero_f,
				c3  = (^Jit_Floats)(&slots[v.slot + 3]) if v.n > 3 else &jit_tables.one_f,
			}
			n_active += 1
		}
	}

	fc_n := meta.fc_n
	fc_slot := meta.fc_slot
	fc_base := meta.fc_base
	ff_n := meta.ff_n
	ff_slot := meta.ff_slot
	ff_base := meta.ff_base
	dptr := raw_data(quads.depth)
	ffptr := raw_data(quads.front_facing)

	// Hoist constant Frag_Coord z/w and Front_Facing fills out of the quad loop.
	if fc_n > 2 {
		if dptr == nil {
			jit_fill_slot(&slots[fc_slot + 2], 0, fc_base)
		}
		if fc_n > 3 {
			jit_fill_slot(&slots[fc_slot + 3], 1, fc_base)
		}
	}
	if ff_n > 0 && ffptr == nil {
		for k in 0 ..< ff_n {
			jit_fill_slot(&slots[ff_slot + k], 1, ff_base)
		}
	}

	var_spans := meta.var_spans
	vptr := uintptr(raw_data(quads.varyings))
	col_stride := uintptr(total) * size_of(f32)
	optr := raw_data(quads.origin)
	hptr := raw_data(quads.helpers)
	sptr := raw_data(shaded)

	first := 0
	for ; first + 2 <= quads.count; first += 2 {
		pair_off := uintptr(first * 4) * size_of(f32)
		for sp in var_spans {
			src := vptr + uintptr(sp.offset) * col_stride + pair_off
			dst := slots[sp.slot:]
			for k in 0 ..< int(sp.n) {
				(^Simd_F8)(&dst[k])^ = intrinsics.unaligned_load((^Simd_F8)(src))
				src += col_stride
			}
		}
		if fc_n > 0 {
			if fc_base == .Float {
				of := (Simd_F4)(intrinsics.unaligned_load((^#simd[4]i32)(&optr[first])))
				(^Simd_F8)(&slots[fc_slot])^ = intrinsics.simd_shuffle(of, of, 0, 0, 0, 0, 2, 2, 2, 2) + JIT_FRAG_DX
				if fc_n > 1 {
					(^Simd_F8)(&slots[fc_slot + 1])^ = intrinsics.simd_shuffle(of, of, 1, 1, 1, 1, 3, 3, 3, 3) + JIT_FRAG_DY
					if fc_n > 2 && dptr != nil {
						(^Simd_F8)(&slots[fc_slot + 2])^ = intrinsics.unaligned_load((^Simd_F8)(&dptr[first * 4]))
					}
				}
			} else {
				o0, o1 := optr[first], optr[first + 1]
				ox0, ox1 := f32(o0.x), f32(o1.x)
				jit_fill_slot(&slots[fc_slot], {ox0 + 0.5, ox0 + 1.5, ox0 + 0.5, ox0 + 1.5, ox1 + 0.5, ox1 + 1.5, ox1 + 0.5, ox1 + 1.5}, fc_base)
				if fc_n > 1 {
					oy0, oy1 := f32(o0.y), f32(o1.y)
					jit_fill_slot(&slots[fc_slot + 1], {oy0 + 0.5, oy0 + 0.5, oy0 + 1.5, oy0 + 1.5, oy1 + 0.5, oy1 + 0.5, oy1 + 1.5, oy1 + 1.5}, fc_base)
					if fc_n > 2 && dptr != nil {
						jit_fill_slot(&slots[fc_slot + 2], intrinsics.unaligned_load((^Jit_Floats)(&dptr[first * 4])), fc_base)
					}
				}
			}
		}
		if ff_n > 0 && ffptr != nil {
			f0 := f32(1) if ffptr[first] else 0
			f1 := f32(1) if ffptr[first + 1] else 0
			col := Jit_Floats{f0, f0, f0, f0, f1, f1, f1, f1}
			for k in 0 ..< ff_n {
				jit_fill_slot(&slots[ff_slot + k], col, ff_base)
			}
		}

		masks[MASK_DISCARD] = 0
		body_fn(&ctx)

		alive := ~masks[MASK_DISCARD]
		if fast_outs {
			if hptr != nil {
				alive &= ~((transmute(u8)hptr[first] & 0xF) | ((transmute(u8)hptr[first + 1] & 0xF) << 4))
			}
			sptr[first] = transmute(Quad_Lanes)(alive & 0xF)
			sptr[first + 1] = transmute(Quad_Lanes)(alive >> 4)
			if alive == 0xFF {
				base_out := first * 4
				for ti in 0 ..< n_active {
					jit_store_pair(active_outs[ti], base_out)
				}
			} else if alive != 0 {
				if q0 := alive & 0xF; q0 != 0 {
					base_out := first * 4
					for ti in 0 ..< n_active {
						jit_store_quad(active_outs[ti], base_out, 0, q0)
					}
				}
				if q1 := alive >> 4; q1 != 0 {
					base_out := (first + 1) * 4
					for ti in 0 ..< n_active {
						jit_store_quad(active_outs[ti], base_out, 1, q1)
					}
				}
			}
		} else {
			jit_write_fragment_slow(meta, slots, quads, targets, shaded, sample_masks, first, 2, alive)
		}
	}

	if first < quads.count {
		quad_off := uintptr(first * 4) * size_of(f32)
		for sp in var_spans {
			src := vptr + uintptr(sp.offset) * col_stride + quad_off
			dst := slots[sp.slot:]
			for k in 0 ..< int(sp.n) {
				((^[2]Simd_F4)(&dst[k]))[0] = intrinsics.unaligned_load((^Simd_F4)(src))
				src += col_stride
			}
		}
		if fc_n > 0 {
			ox := f32(optr[first].x)
			jit_fill_slot(&slots[fc_slot], {ox + 0.5, ox + 1.5, ox + 0.5, ox + 1.5, 0, 0, 0, 0}, fc_base)
			if fc_n > 1 {
				oy := f32(optr[first].y)
				jit_fill_slot(&slots[fc_slot + 1], {oy + 0.5, oy + 0.5, oy + 1.5, oy + 1.5, 0, 0, 0, 0}, fc_base)
				if fc_n > 2 && dptr != nil {
					col: Jit_Floats
					for l in 0 ..< 4 {col[l] = dptr[first * 4 + l]}
					jit_fill_slot(&slots[fc_slot + 2], col, fc_base)
				}
			}
		}
		if ff_n > 0 && ffptr != nil {
			f0 := f32(1) if ffptr[first] else 0
			col := Jit_Floats{f0, f0, f0, f0, 0, 0, 0, 0}
			for k in 0 ..< ff_n {
				jit_fill_slot(&slots[ff_slot + k], col, ff_base)
			}
		}

		ctx.lanes = 0x0F
		masks[MASK_DISCARD] = 0
		body_fn(&ctx)

		alive := ~masks[MASK_DISCARD] & 0x0F
		if fast_outs {
			if hptr != nil {
				alive &= ~(transmute(u8)hptr[first] & 0xF)
			}
			sptr[first] = transmute(Quad_Lanes)alive
			if alive != 0 {
				base_out := first * 4
				for ti in 0 ..< n_active {
					jit_store_quad(active_outs[ti], base_out, 0, alive)
				}
			}
		} else {
			jit_write_fragment_slow(meta, slots, quads, targets, shaded, sample_masks, first, 1, alive)
		}
	}
}

// Runs workgroups [start, end) of a compute dispatch (run_compute validated the
// options and the range): each workgroup's invocations in 8-lane batches of
// consecutive gl_LocalInvocationIndex, the last batch partial.
@(private, enable_target_feature = "avx2")
jit_run_compute :: proc(p: ^Program, uniforms: []u8, groups: [3]int, start, end: int, scratch: ^Scratch, opts: Run_Options) #no_bounds_check {
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	slots := ctx.slots
	masks := ctx.masks
	body_fn := Jit_Fn(p.jit_body)
	shared: []u32
	if p.shared_words > 0 {
		if len(scratch.shared) < p.shared_words {
			resize(&scratch.shared, p.shared_words)
		}
		shared = scratch.shared[:p.shared_words]
		ctx.shared = raw_data(shared)
	}
	for im, i in opts.images[:len(p.image_formats)] {
		tb := image_texel_bytes(im.format)
		ctx.imgs[i] = {
			ptr    = raw_data(im.data),
			wb     = u32(max(im.width, 0)) ~ 0x8000_0000,
			hb     = u32(max(im.height, 0)) ~ 0x8000_0000,
			stride = u32(im.stride if im.stride != 0 else im.width * tb),
		}
	}
	ls := [3]u32{max(p.local_size[0], 1), max(p.local_size[1], 1), max(p.local_size[2], 1)}
	local_total := int(ls.x * ls.y * ls.z)
	n_batches := (local_total + JIT_LANES - 1) / JIT_LANES

	// Each batch's gl_LocalInvocationID (x, y, z) and gl_LocalInvocationIndex,
	// once per run; lanes past the workgroup read 0 as in the interpreter.
	local: [1024 / JIT_LANES][4]Jit_Slot = ---
	{
		i := 0
		for lz in 0 ..< ls.z {
			for ly in 0 ..< ls.y {
				for lx in 0 ..< ls.x {
					local[i / JIT_LANES][0][i % JIT_LANES] = lx
					local[i / JIT_LANES][1][i % JIT_LANES] = ly
					local[i / JIT_LANES][2][i % JIT_LANES] = lz
					i += 1
				}
			}
		}
		for j in 0 ..< n_batches * JIT_LANES {
			if j >= local_total {
				for k in 0 ..< 3 {
					local[j / JIT_LANES][k][j % JIT_LANES] = 0
				}
			}
			local[j / JIT_LANES][3][j % JIT_LANES] = u32(j)
		}
	}
	lid_slot, lidx_slot, gid_slot, wg_slot := i32(-1), i32(-1), i32(-1), i32(-1)
	for v in p.inputs {
		#partial switch v.kind {
		case .Num_Work_Groups:
			for k in 0 ..< 3 {
				slots[v.slot + i32(k)] = u32(groups[k])
			}
		case .Local_Invocation_ID:
			lid_slot = v.slot
		case .Local_Invocation_Index:
			lidx_slot = v.slot
		case .Global_Invocation_ID:
			gid_slot = v.slot
		case .Work_Group_ID:
			wg_slot = v.slot
		}
	}
	gx_n := groups.x
	gxy_n := groups.x * groups.y
	for g in start ..< end {
		if len(shared) > 0 {
			intrinsics.mem_zero(raw_data(shared), len(shared) * size_of(u32))
		}
		wg := [3]u32{u32(g % gx_n), u32((g / gx_n) % groups.y), u32(g / gxy_n)}
		if wg_slot >= 0 {
			for k in 0 ..< i32(3) {
				slots[wg_slot + k] = wg[k]
			}
		}
		base_gid := [3]Simd_U8{Simd_U8(wg[0] * ls.x), Simd_U8(wg[1] * ls.y), Simd_U8(wg[2] * ls.z)}
		for b in 0 ..< n_batches {
			lb := &local[b]
			if lid_slot >= 0 {
				for k in 0 ..< i32(3) {
					slots[lid_slot + k] = lb[k]
				}
			}
			if lidx_slot >= 0 {
				slots[lidx_slot] = lb[3]
			}
			if gid_slot >= 0 {
				for k in 0 ..< i32(3) {
					(^Simd_U8)(&slots[gid_slot + k])^ = intrinsics.unaligned_load((^Simd_U8)(&lb[k])) + base_gid[k]
				}
			}
			lanes := min(local_total - b * JIT_LANES, JIT_LANES)
			ctx.lanes = u8((u32(1) << uint(lanes)) - 1)
			masks[MASK_DISCARD] = 0
			body_fn(&ctx)
		}
	}
}

}
