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
	buf_ptr:  [4][^]u32,
	buf_len:  [4]uint,
	host_ctx: runtime.Context,
	lanes:    u8,
}

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
	one_f:      [8]f32,
	neg_one_f:  [8]f32,
	two_f:      [8]f32,
	three_f:    [8]f32,
	deg_to_rad: [8]f32,
	rad_to_deg: [8]f32,
	f2i_hi:     [8]f32,
	f2u_hi:     [8]f32,
}

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
	jit_tables.one_f = 1.0
	jit_tables.neg_one_f = -1.0
	jit_tables.two_f = 2.0
	jit_tables.three_f = 3.0
	jit_tables.deg_to_rad = math.PI / 180.0
	jit_tables.rad_to_deg = 180.0 / math.PI
	jit_tables.f2i_hi = 2147483648.0
	jit_tables.f2u_hi = 4294967296.0

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
jit_destroy :: proc(p: ^Program) {
	if p.jit_mem != nil {
		virtual.release(raw_data(p.jit_mem), len(p.jit_mem))
		p.jit_mem = nil
		p.jit_init = nil
		p.jit_body = nil
	}
}

// ---- x86-64 AVX2 machine-code emitter ---------------------------------------------------

// Pinned registers inside generated functions (System V AMD64 ABI):
//   rbp (5)  : ^Jit_Ctx
//   rbx (3)  : [^]Jit_Slot (32-byte aligned)
//   r12 (12) : [^]u8       (masks; masks[0] = EXEC, masks[1] = DISCARD)
//   r13 (13) : u32         (live masks[MASK_EXEC] in 0..255)
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
OFF_TAB_F2I_HI :: i32(offset_of(Jit_Tables, f2i_hi))
@(private)
OFF_TAB_F2U_HI :: i32(offset_of(Jit_Tables, f2u_hi))

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
emit_load_mask_eax :: proc(buf: ^[dynamic]u8, m: i32) {
	if m == MASK_EXEC {
		emit_bytes(buf, 0x44, 0x89, 0xE8) // mov eax, r13d
	} else {
		emit_bytes(buf, 0x41, 0x0F, 0xB6) // movzx eax, byte [r12 + m]
		emit_modrm_mem(buf, GPR_RAX, GPR_R12, m)
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

// ---- basic-block slot register cache (ymm8..ymm15) --------------------------------------

@(private)
Cache_Slot :: struct {
	slot:  i32,
	dirty: bool,
	stamp: u32,
}

@(private)
Instr_RW :: struct {
	writes:     Slots,
	full_write: bool,
	reads:      [4]Slots,
}

@(private)
Jit_Emitter :: struct {
	buf:          [dynamic]u8,
	cross_bb:     []bool,
	last_use:     []i32,
	def_bb:       []i32,
	bb_start:     []bool,
	pc_off:       []int,
	rw:           []Instr_RW,
	patches:      [dynamic]Jump_Patch,
	epilogues:    [dynamic]int,
	cache:        [8]Cache_Slot,
	stamp:        u32,
	pc:           i32,
}

@(private)
cache_reset :: proc(e: ^Jit_Emitter) {
	for &c in e.cache {
		c = {slot = -1, dirty = false, stamp = 0}
	}
}

@(private)
pin_bit :: #force_inline proc(ymm: u8) -> u8 {
	return (u8(1) << (ymm - 8)) if ymm >= 8 else 0
}

@(private)
alloc_cache_idx :: proc(e: ^Jit_Emitter, pinned: u8) -> int {
	// Prefer empty or already-dead entries first (zero spill cost).
	for c, i in e.cache {
		if (pinned & (u8(1) << uint(i))) != 0 {
			continue
		}
		if c.slot < 0 || e.last_use[c.slot] < e.pc {
			e.cache[i] = {slot = -1, dirty = false, stamp = 0}
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
	e.cache[best] = {slot = -1, dirty = false, stamp = 0}
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
	if e.last_use[slot] > e.pc {
		idx := alloc_cache_idx(e, pinned)
		r := u8(8 + idx)
		vmovaps_load(&e.buf, r, slot)
		e.stamp += 1
		e.cache[idx] = {slot = slot, dirty = false, stamp = e.stamp}
		return r
	}
	vmovaps_load(&e.buf, scratch, slot)
	return scratch
}

@(private)
write_slot :: proc(e: ^Jit_Emitter, slot: i32, src: u8, pinned: u8 = 0) {
	for &c, i in e.cache {
		if c.slot == slot {
			r := u8(8 + i)
			vmovaps_rr(&e.buf, r, src)
			e.stamp += 1
			c.stamp = e.stamp
			if e.cross_bb[slot] {
				vmovaps_store(&e.buf, slot, r)
				c.dirty = false
			} else {
				c.dirty = true
			}
			return
		}
	}
	if e.last_use[slot] > e.pc {
		idx := alloc_cache_idx(e, pinned)
		r := u8(8 + idx)
		vmovaps_rr(&e.buf, r, src)
		e.stamp += 1
		if e.cross_bb[slot] {
			vmovaps_store(&e.buf, slot, r)
			e.cache[idx] = {slot = slot, dirty = false, stamp = e.stamp}
		} else {
			e.cache[idx] = {slot = slot, dirty = true, stamp = e.stamp}
		}
		return
	}
	if e.cross_bb[slot] {
		vmovaps_store(&e.buf, slot, src)
	}
}

@(private)
is_helper_instr :: proc(in_: Instr) -> bool {
	#partial switch in_.op {
	case .Store_Dyn,
	     .I_Div, .U_Div, .I_Rem, .U_Rem,
	     .Faceforward, .Refract,
	     .Determinant, .Inverse,
	     .Texture, .Texture_Bias, .Texture_Size:
		return true
	case .Load_Dyn:
		return int(in_.n) > 4
	case .Load_Buf:
		return int(in_.k) >= 4
	case .F_Fn1:
		#partial switch Builtin(in_.k) {
		case .Radians, .Degrees, .Sqrt, .Inversesqrt,
		     .Floor, .Ceil, .Fract, .Trunc, .Round_Even,
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
	case .Texture, .Texture_Bias:
		reads[0] = {in_.a, n}
		reads[1] = {in_.b, 1}
		return {in_.dst, 4}, true, reads
	case .Deriv_X, .Deriv_Y, .Fwidth:
		reads[0] = {in_.a, n}
		return {in_.dst, n}, true, reads
	case .Load_Dyn:
		reads[0] = {in_.b, 1}
		return {in_.dst, n}, true, reads
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
jit_help_idiv :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	for i in 0 ..< i32(in_.n) {
		a, b, d := &slots[in_.a + i], &slots[in_.b + i], &slots[in_.dst + i]
		#partial switch in_.op {
		case .I_Div:
			for l in 0 ..< JIT_LANES {d[l] = bin_lane(a[l], b[l], .I_Div)}
		case .U_Div:
			for l in 0 ..< JIT_LANES {d[l] = bin_lane(a[l], b[l], .U_Div)}
		case .I_Rem:
			for l in 0 ..< JIT_LANES {d[l] = bin_lane(a[l], b[l], .I_Rem)}
		case .U_Rem:
			for l in 0 ..< JIT_LANES {d[l] = bin_lane(a[l], b[l], .U_Rem)}
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
		case .Round:
			for l in 0 ..< JIT_LANES {d[l] = fn1(x[l], .Round)}
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
	ch := int(in_.k)
	if opts.sampler == nil || ch < 0 {
		return
	}
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

@(private)
jit_help_texture_size :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	opts := ctx.opts
	ch := int(in_.k)
	size := opts.channel_sizes[ch] if ch >= 0 && ch < len(opts.channel_sizes) else [3]i32{}
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

// ---- IR -> AVX2 lowering ----------------------------------------------------------------

@(private)
Jump_Patch :: struct {
	patch_pos: int,
	target_pc: i32,
}

@(private)
jit_emit_fn :: proc(e: ^Jit_Emitter, p: ^Program, code: []Instr, is_init: bool) {
	n_code := len(code)
	bb_start := e.bb_start[:n_code + 1]
	for &b in bb_start {
		b = false
	}
	bb_start[0] = true
	has_dyn := false
	rw := e.rw[:n_code]
	for in_, pc in code {
		w, fw, r := instr_read_write(in_)
		rw[pc] = {w, fw, r}
		if in_.op == .Load_Dyn || in_.op == .Store_Dyn {
			has_dyn = true
		}
		if in_.op == .Jump || in_.op == .Jump_If_None {
			if in_.a >= 0 && int(in_.a) <= n_code {
				bb_start[in_.a] = true
			}
			bb_start[pc + 1] = true
		} else if in_.op == .Mask_Kill {
			bb_start[pc + 1] = true
		} else if is_helper_instr(in_) {
			bb_start[pc] = true
			bb_start[pc + 1] = true
		}
	}
	bb_start[n_code] = true

	for &cb in e.cross_bb {
		cb = is_init || has_dyn
	}
	if !is_init && !has_dyn {
		for v in p.outputs {
			for k in 0 ..< v.n {
				e.cross_bb[v.slot + k] = true
			}
		}
		for &d in e.def_bb {
			d = 0
		}
		bb_id := i32(0)
		for _, pc in code {
			if bb_start[pc] {
				bb_id += 1
			}
			info := rw[pc]
			for r in info.reads {
				for s in r.start ..< r.start + r.n {
					if e.def_bb[s] != bb_id {
						e.cross_bb[s] = true
					}
				}
			}
			if info.full_write {
				for s in info.writes.start ..< info.writes.start + info.writes.n {
					e.def_bb[s] = bb_id
				}
			}
		}
	}

	for &lu in e.last_use {
		lu = -1
	}

	pc_off := e.pc_off[:n_code + 1]
	clear(&e.patches)
	clear(&e.epilogues)

	// Prologue: 5 callee-saved GPRs (40 B + 8 B ret addr = 48 B -> 16-byte aligned rsp).
	emit_bytes(&e.buf, 0x55, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56)
	emit_bytes(&e.buf, 0x48, 0x89, 0xFD) // mov rbp, rdi
	emit_bytes(&e.buf, 0x48, 0x8B)       // mov rbx, [rbp + OFF_CTX_SLOTS]
	emit_modrm_mem(&e.buf, GPR_RBX, GPR_RBP, OFF_CTX_SLOTS)
	emit_bytes(&e.buf, 0x4C, 0x8B)       // mov r12, [rbp + OFF_CTX_MASKS]
	emit_modrm_mem(&e.buf, GPR_R12, GPR_RBP, OFF_CTX_MASKS)
	emit_bytes(&e.buf, 0x45, 0x0F, 0xB6, 0x2C, 0x24) // movzx r13d, byte [r12]
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
			for q in pc ..< end {
				for r in rw[q].reads {
					for s in r.start ..< r.start + r.n {
						e.last_use[s] = i32(q)
					}
				}
			}
		}

		e.pc = i32(pc)
		in_ := code[pc]
		n := i32(in_.n)
		dst, a, b, c := in_.dst, in_.a, in_.b, in_.c

		switch in_.op {
		case .Nop:

		case .Mov:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 0)
				write_slot(e, dst + i, ya, pin_bit(ya))
			}

		case .Splat:
			ya := read_slot(e, a, 0)
			pa := pin_bit(ya)
			for i in 0 ..< n {
				write_slot(e, dst + i, ya, pa)
			}

		case .Gather:
			srcs := [4]i32{a, b, c, in_.d}
			for i in 0 ..< n {
				ys := read_slot(e, srcs[i], u8(i))
				vmovaps_rr(&e.buf, u8(i), ys)
			}
			for i in 0 ..< n {
				write_slot(e, dst + i, u8(i))
			}

		case .Store:
			emit_exec_ymask(&e.buf, 0)
			for i in 0 ..< n {
				yd := read_slot(e, dst + i, 1)
				ya := read_slot(e, a + i, 2, pin_bit(yd))
				vpblendvb(&e.buf, 3, yd, ya, 0)
				write_slot(e, dst + i, 3)
			}

		case .Store_Zero:
			emit_exec_ymask(&e.buf, 0)
			for i in 0 ..< n {
				yd := read_slot(e, dst + i, 1)
				vop_d(&e.buf, 0xDF, 2, 0, yd) // vpandn ymm2, ymm0, yd
				write_slot(e, dst + i, 2)
			}

		case .Load_Dyn:
			if n > 4 {
				emit_helper_call(&e.buf, jit_help_load_dyn, &code[pc])
				break
			}
			yoff := read_slot(e, b, 6)
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
				emit_vex_sib(&e.buf, 1, 0, 1, 0, 0x28, u8(i), 0, GPR_RBX, GPR_RAX, 0, (a + i) * 32)
			}
			emit_b(&e.buf, 0xE9) // jmp done
			done_dyn := len(e.buf)
			emit_i32(&e.buf, 0)

			patch_i32(e.buf[:], div_dyn, i32(len(e.buf) - (div_dyn + 4)))
			emit_vex_rm(&e.buf, 2, 1, 1, 0, 0x47, 5, yoff, GPR_R14, OFF_TAB_THREE_U) // off << 3
			vop_d_tab(&e.buf, 0xFE, 5, 5, OFF_TAB_IOTA)                              // off*8 + lane
			for i in 0 ..< n {
				emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 7, 0, GPR_R14, OFF_TAB_ALL1)
				vzero(&e.buf, u8(i))
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, u8(i), 7, GPR_RBX, 5, 2, (a + i) * 32)
			}

			patch_i32(e.buf[:], done_dyn, i32(len(e.buf) - (done_dyn + 4)))
			for i in 0 ..< n {
				write_slot(e, dst + i, u8(i))
			}

		case .Store_Dyn:
			emit_helper_call(&e.buf, jit_help_store_dyn, &code[pc])

		case .F_Add, .F_Sub, .F_Mul, .F_Div:
			op_byte: u8 = 0x58
			#partial switch in_.op {
			case .F_Sub: op_byte = 0x5C
			case .F_Mul: op_byte = 0x59
			case .F_Div: op_byte = 0x5E
			}
			sa, sb := 1 - i32(in_.k & 1), 1 - i32(in_.k >> 1)
			for i in 0 ..< n {
				ya := read_slot(e, a + i * sa, 1)
				yb := read_slot(e, b + i * sb, 2, pin_bit(ya))
				vop_ps(&e.buf, op_byte, 0, ya, yb)
				write_slot(e, dst + i, 0, pin_bit(ya) | pin_bit(yb))
			}

		case .F_Neg:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vop_ps_tab(&e.buf, 0x57, 0, ya, OFF_TAB_SIGN) // vxorps
				write_slot(e, dst + i, 0)
			}

		case .I_Add, .I_Sub, .I_Mul, .Bit_And, .Bit_Or, .Bit_Xor:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				#partial switch in_.op {
				case .I_Add:   vop_d(&e.buf, 0xFE, 0, ya, yb)
				case .I_Sub:   vop_d(&e.buf, 0xFA, 0, ya, yb)
				case .I_Mul:   vop38_d(&e.buf, 0x40, 0, ya, yb)
				case .Bit_And: vop_d(&e.buf, 0xDB, 0, ya, yb)
				case .Bit_Or:  vop_d(&e.buf, 0xEB, 0, ya, yb)
				case .Bit_Xor: vop_d(&e.buf, 0xEF, 0, ya, yb)
				}
				write_slot(e, dst + i, 0)
			}

		case .I_Div, .U_Div, .I_Rem, .U_Rem:
			emit_helper_call(&e.buf, jit_help_idiv, &code[pc])

		case .I_Neg:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vop_d(&e.buf, 0xFA, 0, 2, ya)
				write_slot(e, dst + i, 0)
			}

		case .Bit_Not:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vop_d_tab(&e.buf, 0xEF, 0, ya, OFF_TAB_ALL1)
				write_slot(e, dst + i, 0)
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
				vop38_d(&e.buf, op_byte, 0, ya, 3)
				write_slot(e, dst + i, 0)
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
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				vcmpps(&e.buf, 0, ya, yb, pred)
				vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				write_slot(e, dst + i, 0)
			}

		case .I_Lt, .I_Le, .I_Gt, .I_Ge, .I_Eq, .I_Ne:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				#partial switch in_.op {
				case .I_Gt:
					vop_d(&e.buf, 0x66, 0, ya, yb)
					vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				case .I_Lt:
					vop_d(&e.buf, 0x66, 0, yb, ya)
					vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				case .I_Le:
					vop_d(&e.buf, 0x66, 0, ya, yb)
					vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				case .I_Ge:
					vop_d(&e.buf, 0x66, 0, yb, ya)
					vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				case .I_Eq:
					vop_d(&e.buf, 0x76, 0, ya, yb)
					vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				case .I_Ne:
					vop_d(&e.buf, 0x76, 0, ya, yb)
					vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				}
				write_slot(e, dst + i, 0)
			}

		case .U_Lt, .U_Le, .U_Gt, .U_Ge:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				#partial switch in_.op {
				case .U_Le:
					vop38_d(&e.buf, 0x3B, 0, ya, yb) // vpminud
					vop_d(&e.buf, 0x76, 0, 0, ya)
					vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				case .U_Ge:
					vop38_d(&e.buf, 0x3F, 0, ya, yb) // vpmaxud
					vop_d(&e.buf, 0x76, 0, 0, ya)
					vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				case .U_Gt:
					vop38_d(&e.buf, 0x3B, 0, ya, yb)
					vop_d(&e.buf, 0x76, 0, 0, ya)
					vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				case .U_Lt:
					vop38_d(&e.buf, 0x3F, 0, ya, yb)
					vop_d(&e.buf, 0x76, 0, 0, ya)
					vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				}
				write_slot(e, dst + i, 0)
			}

		case .B_And:
			vzero(&e.buf, 3)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				vop_d(&e.buf, 0x76, 4, ya, 3)
				vop_d(&e.buf, 0x76, 5, yb, 3)
				vop_d(&e.buf, 0xEB, 0, 4, 5)
				vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				write_slot(e, dst + i, 0)
			}

		case .B_Or:
			vzero(&e.buf, 3)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				yb := read_slot(e, b + i, 2, pin_bit(ya))
				vop_d(&e.buf, 0xEB, 0, ya, yb)
				vop_d(&e.buf, 0x76, 0, 0, 3)
				vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				write_slot(e, dst + i, 0)
			}

		case .B_Not:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vop_d(&e.buf, 0x76, 0, ya, 2)
				vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				write_slot(e, dst + i, 0)
			}

		case .I_To_F:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x5B, 0, 0, ya) // vcvtdq2ps
				write_slot(e, dst + i, 0)
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
				vblendvps(&e.buf, 0, 2, 3, ya)                  // sign(u) ? hi : lo
				write_slot(e, dst + i, 0)
			}

		case .F_To_I:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x5B, 2, 0, ya)      // vcvttps2dq
				vcmpps_tab(&e.buf, 3, ya, OFF_TAB_F2I_HI, 0x1D)      // ya >= 2^31
				vop_d(&e.buf, 0xEF, 2, 2, 3)                         // 0x80000000 -> 0x7fffffff
				vcmpps(&e.buf, 3, ya, ya, 0x07)                      // ORD_Q (!isnan)
				vop_d(&e.buf, 0xDB, 0, 2, 3)                         // NaN -> 0
				write_slot(e, dst + i, 0)
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
				vop_d(&e.buf, 0xEB, 0, 0, 6)                         // clamp to 0xffffffff
				write_slot(e, dst + i, 0)
			}

		case .B_To_F:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vop_d(&e.buf, 0x76, 0, ya, 2)
				vop_ps_tab(&e.buf, 0x55, 0, 0, OFF_TAB_ONE_F) // vandnps
				write_slot(e, dst + i, 0)
			}

		case .F_To_B:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vcmpps(&e.buf, 0, ya, 2, 0x04) // NEQ_UQ: !(x == 0.0)
				vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
				write_slot(e, dst + i, 0)
			}

		case .I_To_B:
			vzero(&e.buf, 2)
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				vop_d(&e.buf, 0x76, 0, ya, 2)
				vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
				write_slot(e, dst + i, 0)
			}

		case .Select:
			vzero(&e.buf, 4)
			if in_.k == 1 {
				yc := read_slot(e, c, 3)
				vop_d(&e.buf, 0x76, 3, yc, 4) // ymm3 = (c == 0)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					vpblendvb(&e.buf, 0, ya, yb, 3)
					write_slot(e, dst + i, 0)
				}
			} else {
				for i in 0 ..< n {
					yc := read_slot(e, c + i, 3)
					vop_d(&e.buf, 0x76, 3, yc, 4)
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					vpblendvb(&e.buf, 0, ya, yb, 3)
					write_slot(e, dst + i, 0)
				}
			}

		case .All:
			vzero(&e.buf, 2)
			ya0 := read_slot(e, a, 1)
			vop_d(&e.buf, 0x76, 0, ya0, 2)
			vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				vop_d(&e.buf, 0x76, 3, yai, 2)
				vop_d(&e.buf, 0xDF, 0, 3, 0)
			}
			write_slot(e, dst, 0)

		case .Any:
			vzero(&e.buf, 2)
			ya0 := read_slot(e, a, 0)
			vmovaps_rr(&e.buf, 0, ya0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				vop_d(&e.buf, 0xEB, 0, 0, yai)
			}
			vop_d(&e.buf, 0x76, 0, 0, 2)
			vop_d_tab(&e.buf, 0xDF, 0, 0, OFF_TAB_ONE_U)
			write_slot(e, dst, 0)

		case .F_Fn1:
			bi := Builtin(in_.k)
			#partial switch bi {
			case .Abs:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vop_ps_tab(&e.buf, 0x54, 0, ya, OFF_TAB_ABS)
					write_slot(e, dst + i, 0)
				}
			case .Sqrt:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, 0, 0, ya)
					write_slot(e, dst + i, 0)
				}
			case .Inversesqrt:
				emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 2, 0, GPR_R14, OFF_TAB_ONE_F)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, 0, 0, ya)
					vop_ps(&e.buf, 0x5E, 0, 2, 0)
					write_slot(e, dst + i, 0)
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
					vroundps(&e.buf, 0, ya, mode)
					write_slot(e, dst + i, 0)
				}
			case .Fract:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vroundps(&e.buf, 2, ya, 0x09)
					vop_ps(&e.buf, 0x5C, 0, ya, 2)
					write_slot(e, dst + i, 0)
				}
			case .Radians:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vop_ps_tab(&e.buf, 0x59, 0, ya, OFF_TAB_DEG2RAD)
					write_slot(e, dst + i, 0)
				}
			case .Degrees:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vop_ps_tab(&e.buf, 0x59, 0, ya, OFF_TAB_RAD2DEG)
					write_slot(e, dst + i, 0)
				}
			case .Sign:
				vzero(&e.buf, 2)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vcmpps(&e.buf, 3, ya, 2, 0x1E) // ya > 0
					vcmpps(&e.buf, 4, ya, 2, 0x11) // ya < 0
					vop_ps_tab(&e.buf, 0x54, 3, 3, OFF_TAB_ONE_F)
					vop_ps_tab(&e.buf, 0x54, 4, 4, OFF_TAB_NEG1_F)
					vop_ps(&e.buf, 0x56, 0, 3, 4)
					write_slot(e, dst + i, 0)
				}
			case .Isnan:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vcmpps(&e.buf, 0, ya, ya, 0x03) // UNORD_Q
					vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
					write_slot(e, dst + i, 0)
				}
			case .Isinf:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vop_ps_tab(&e.buf, 0x54, 0, ya, OFF_TAB_ABS)
					vop_d_tab(&e.buf, 0x76, 0, 0, OFF_TAB_INF)
					vop_d_tab(&e.buf, 0xDB, 0, 0, OFF_TAB_ONE_U)
					write_slot(e, dst + i, 0)
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
					vop_ps(&e.buf, 0x5D, 0, yb, ya)
					write_slot(e, dst + i, 0)
				}
			case .Max:
				// (y > x) ? y : x -> vvvv = y, rm = x.
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					vop_ps(&e.buf, 0x5F, 0, yb, ya)
					write_slot(e, dst + i, 0)
				}
			case .Step:
				// (y < x) ? 0.0 : 1.0
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					vcmpps(&e.buf, 0, yb, ya, 0x11)
					vop_ps_tab(&e.buf, 0x55, 0, 0, OFF_TAB_ONE_F)
					write_slot(e, dst + i, 0)
				}
			case .Mod:
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					yb := read_slot(e, b + i, 2, pin_bit(ya))
					vop_ps(&e.buf, 0x5E, 3, ya, yb)
					vroundps(&e.buf, 3, 3, 0x09)
					vop_ps(&e.buf, 0x59, 3, yb, 3)
					vop_ps(&e.buf, 0x5C, 0, ya, 3)
					write_slot(e, dst + i, 0)
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
				// Match Odin's min(max(x, lo), hi) (IEEE maxNum / minNum):
				// return the non-NaN operand whenever either argument is NaN.
				for i in 0 ..< n {
					yx := read_slot(e, a + i, 1)
					ylo := read_slot(e, b + i, 2, pin_bit(yx))
					yhi := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(ylo))
					vop_ps(&e.buf, 0x5F, 4, yx, ylo)
					vcmpps(&e.buf, 5, ylo, ylo, 0x03)
					vblendvps(&e.buf, 0, 4, yx, 5)
					vop_ps(&e.buf, 0x5D, 4, 0, yhi)
					vcmpps(&e.buf, 5, yhi, yhi, 0x03)
					vblendvps(&e.buf, 0, 4, 0, 5)
					write_slot(e, dst + i, 0)
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
					vop_ps(&e.buf, 0x58, 0, 4, 6)
					write_slot(e, dst + i, 0)
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
					vop_ps(&e.buf, 0x59, 0, 0, 4)
					write_slot(e, dst + i, 0)
				}
			}

		case .I_Fn1:
			if Builtin(in_.k) == .Abs {
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x1E, 0, 0, ya) // vpabsd
					write_slot(e, dst + i, 0)
				}
			} else {
				vzero(&e.buf, 2)
				for i in 0 ..< n {
					ya := read_slot(e, a + i, 1)
					vop_d(&e.buf, 0x66, 3, ya, 2) // pos = (x > 0) ? -1 : 0
					vop_d(&e.buf, 0x66, 4, 2, ya) // neg = (0 > x) ? -1 : 0
					vop_d(&e.buf, 0xFA, 0, 4, 3) // neg - pos = sign(x)
					write_slot(e, dst + i, 0)
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
				vop38_d(&e.buf, op_byte, 0, ya, yb)
				write_slot(e, dst + i, 0)
			}

		case .I_Fn3, .U_Fn3:
			max_op: u8 = 0x3D if in_.op == .I_Fn3 else 0x3F
			min_op: u8 = 0x39 if in_.op == .I_Fn3 else 0x3B
			for i in 0 ..< n {
				yx := read_slot(e, a + i, 1)
				ylo := read_slot(e, b + i, 2, pin_bit(yx))
				yhi := read_slot(e, c + i, 3, pin_bit(yx) | pin_bit(ylo))
				vop38_d(&e.buf, max_op, 0, yx, ylo)
				vop38_d(&e.buf, min_op, 0, 0, yhi)
				write_slot(e, dst + i, 0)
			}

		case .Dot:
			ya0 := read_slot(e, a, 1)
			yb0 := read_slot(e, b, 2, pin_bit(ya0))
			vop_ps(&e.buf, 0x59, 0, ya0, yb0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				ybi := read_slot(e, b + i, 2, pin_bit(yai))
				vop_ps(&e.buf, 0x59, 3, yai, ybi)
				vop_ps(&e.buf, 0x58, 0, 0, 3)
			}
			write_slot(e, dst, 0)

		case .Length:
			ya0 := read_slot(e, a, 1)
			vop_ps(&e.buf, 0x59, 0, ya0, ya0)
			for i in 1 ..< n {
				yai := read_slot(e, a + i, 1)
				vop_ps(&e.buf, 0x59, 2, yai, yai)
				vop_ps(&e.buf, 0x58, 0, 0, 2)
			}
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, 0, 0, 0)
			write_slot(e, dst, 0)

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
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, 0, 0, 0)
			write_slot(e, dst, 0)

		case .Normalize:
			for i in 0 ..< n {
				yai := read_slot(e, a + i, u8(i))
				vmovaps_rr(&e.buf, u8(i), yai)
			}
			vop_ps(&e.buf, 0x59, 4, 0, 0)
			for i in 1 ..< n {
				vop_ps(&e.buf, 0x59, 5, u8(i), u8(i))
				vop_ps(&e.buf, 0x58, 4, 4, 5)
			}
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x51, 4, 0, 4)
			emit_vex_rm(&e.buf, 1, 0, 1, 0, 0x28, 5, 0, GPR_R14, OFF_TAB_ONE_F)
			vop_ps(&e.buf, 0x5E, 4, 5, 4)
			for i in 0 ..< n {
				vop_ps(&e.buf, 0x59, u8(i), u8(i), 4)
				write_slot(e, dst + i, u8(i))
			}

		case .Cross:
			x0 := read_slot(e, a, 0)
			vmovaps_rr(&e.buf, 0, x0)
			x1 := read_slot(e, a + 1, 1)
			vmovaps_rr(&e.buf, 1, x1)
			x2 := read_slot(e, a + 2, 2)
			vmovaps_rr(&e.buf, 2, x2)
			y0 := read_slot(e, b, 3)
			vmovaps_rr(&e.buf, 3, y0)
			y1 := read_slot(e, b + 1, 4)
			vmovaps_rr(&e.buf, 4, y1)
			y2 := read_slot(e, b + 2, 5)
			vmovaps_rr(&e.buf, 5, y2)

			vop_ps(&e.buf, 0x59, 6, 1, 5)
			vop_ps(&e.buf, 0x59, 7, 4, 2)
			vop_ps(&e.buf, 0x5C, 6, 6, 7)
			write_slot(e, dst, 6)

			vop_ps(&e.buf, 0x59, 6, 2, 3)
			vop_ps(&e.buf, 0x59, 7, 5, 0)
			vop_ps(&e.buf, 0x5C, 6, 6, 7)
			write_slot(e, dst + 1, 6)

			vop_ps(&e.buf, 0x59, 6, 0, 4)
			vop_ps(&e.buf, 0x59, 7, 3, 1)
			vop_ps(&e.buf, 0x5C, 6, 6, 7)
			write_slot(e, dst + 2, 6)

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
				vop_ps(&e.buf, 0x5C, 0, yai, 5)
				write_slot(e, dst + i, 0)
			}

		case .Faceforward, .Refract:
			emit_helper_call(&e.buf, jit_help_geom, &code[pc])

		case .Mat_Mul:
			rows, inner, cols := n, i32(in_.k), in_.d
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					ya0 := read_slot(e, a + row, 1)
					yb0 := read_slot(e, b + col * inner, 2, pin_bit(ya0))
					vop_ps(&e.buf, 0x59, 0, ya0, yb0)
					for k in 1 ..< inner {
						yak := read_slot(e, a + k * rows + row, 1)
						ybk := read_slot(e, b + col * inner + k, 2, pin_bit(yak))
						vop_ps(&e.buf, 0x59, 3, yak, ybk)
						vop_ps(&e.buf, 0x58, 0, 0, 3)
					}
					write_slot(e, dst + col * rows + row, 0)
				}
			}

		case .Vec_Mat:
			rows, cols := n, i32(in_.k)
			for col in 0 ..< cols {
				ya0 := read_slot(e, a, 1)
				yb0 := read_slot(e, b + col * rows, 2, pin_bit(ya0))
				vop_ps(&e.buf, 0x59, 0, ya0, yb0)
				for row in 1 ..< rows {
					yar := read_slot(e, a + row, 1)
					ybr := read_slot(e, b + col * rows + row, 2, pin_bit(yar))
					vop_ps(&e.buf, 0x59, 3, yar, ybr)
					vop_ps(&e.buf, 0x58, 0, 0, 3)
				}
				write_slot(e, dst + col, 0)
			}

		case .Transpose:
			rows, cols := n, i32(in_.k)
			for col in 0 ..< cols {
				for row in 0 ..< rows {
					ya := read_slot(e, a + col * rows + row, 0)
					write_slot(e, dst + row * cols + col, ya, pin_bit(ya))
				}
			}

		case .Outer:
			rows, cols := n, i32(in_.k)
			for col in 0 ..< cols {
				ybc := read_slot(e, b + col, 2)
				pb := pin_bit(ybc)
				for row in 0 ..< rows {
					yar := read_slot(e, a + row, 1, pb)
					vop_ps(&e.buf, 0x59, 0, yar, ybc)
					write_slot(e, dst + col * rows + row, 0, pb)
				}
			}

		case .Determinant, .Inverse:
			emit_helper_call(&e.buf, jit_help_mat, &code[pc])

		case .Deriv_X:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x16, 2, 0, ya) // vmovshdup
				emit_vex_rr(&e.buf, 1, 2, 1, 0, 0x12, 3, 0, ya) // vmovsldup
				vop_ps(&e.buf, 0x5C, 0, 2, 3)
				write_slot(e, dst + i, 0)
			}

		case .Deriv_Y:
			for i in 0 ..< n {
				ya := read_slot(e, a + i, 1)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0xC6, 2, ya, ya); emit_b(&e.buf, 0xEE)
				emit_vex_rr(&e.buf, 1, 0, 1, 0, 0xC6, 3, ya, ya); emit_b(&e.buf, 0x44)
				vop_ps(&e.buf, 0x5C, 0, 2, 3)
				write_slot(e, dst + i, 0)
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
				vop_ps(&e.buf, 0x58, 0, 2, 3)
				write_slot(e, dst + i, 0)
			}

		case .Texture, .Texture_Bias:
			emit_helper_call(&e.buf, jit_help_texture, &code[pc])

		case .Texture_Size:
			emit_helper_call(&e.buf, jit_help_texture_size, &code[pc])

		case .Load_Buf:
			k_buf := i32(in_.k)
			if k_buf >= 4 {
				emit_helper_call(&e.buf, jit_help_load_buf, &code[pc])
				break
			}
			yidx := read_slot(e, a, 6)
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

			// Uniform fast path: 64-bit unsigned bounds-checked vbroadcastss into ymm0..ymm(n-1).
			emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x7E, yidx, 0, GPR_RAX) // vmovd eax, xmm(yidx)
			stride := u32(in_.b)
			off := u32(in_.c)
			if stride == 0 {
				emit_bytes(&e.buf, 0x31, 0xC0) // xor eax, eax
			} else if stride != 1 {
				emit_bytes(&e.buf, 0x69, 0xC0) // imul eax, eax, stride
				emit_u32(&e.buf, stride)
			}
			if off != 0 {
				emit_b(&e.buf, 0x05) // add eax, off
				emit_u32(&e.buf, off)
			}
			for i in 0 ..< n {
				if i > 0 {
					emit_bytes(&e.buf, 0x83, 0xC0, 0x01) // add eax, 1 (32-bit wrap + zero-extend rax)
				}
				emit_bytes(&e.buf, 0x48, 0x39, 0xC8) // cmp rax, rcx
				emit_b(&e.buf, 0x73)                 // jae oob
				oob_patch := len(e.buf)
				emit_b(&e.buf, 0)
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x18, u8(i), 0, GPR_RSI, GPR_RAX, 2, 0) // vbroadcastss ymm(i), [rsi + rax*4]
				emit_b(&e.buf, 0xEB) // jmp next_i
				next_patch := len(e.buf)
				emit_b(&e.buf, 0)
				e.buf[oob_patch] = u8(len(e.buf) - (oob_patch + 1))
				vzero(&e.buf, u8(i))
				e.buf[next_patch] = u8(len(e.buf) - (next_patch + 1))
			}
			emit_b(&e.buf, 0xE9) // jmp done
			done_patch := len(e.buf)
			emit_i32(&e.buf, 0)

			// Divergent path: hardware AVX2 vpgatherdd.
			patch_i32(e.buf[:], div_patch, i32(len(e.buf) - (div_patch + 4)))
			if stride == 0 {
				vzero(&e.buf, 4)
			} else if stride == 1 {
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
			// vpgatherdd sign-extends 32-bit indices; if eff_len > 2^31, split low/high 8 GiB halves.
			emit_b(&e.buf, 0xBA) // mov edx, 0x80000000 (zero-extended into rdx)
			emit_u32(&e.buf, 0x8000_0000)
			emit_bytes(&e.buf, 0x48, 0x39, 0xD1) // cmp rcx, rdx
			emit_bytes(&e.buf, 0x0F, 0x87)       // ja big_buf
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
				vzero(&e.buf, u8(i))
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, u8(i), 7, GPR_RSI, 6, 2, 0) // vpgatherdd ymm(i), [rsi + ymm6*4], ymm7
			}
			emit_b(&e.buf, 0xE9) // jmp done2
			done2_patch := len(e.buf)
			emit_i32(&e.buf, 0)

			// Large buffer (> 2^31 words): gather w < 2^31 from rsi and w >= 2^31 from rsi + 8 GiB.
			patch_i32(e.buf[:], big_patch, i32(len(e.buf) - (big_patch + 4)))
			emit_bytes(&e.buf, 0x48, 0x29, 0xD1) // sub rcx, rdx -> rem in 1..2^31
			emit_bytes(&e.buf, 0x48, 0x8D)       // lea rdi, [rsi + rdx*4] -> rsi + 8 GiB
			emit_modrm_sib(&e.buf, GPR_RDI, GPR_RSI, GPR_RDX, 2, 0)
			emit_vex_rr(&e.buf, 1, 1, 0, 0, 0x6E, 5, 0, GPR_RCX)
			emit_vex_rr(&e.buf, 2, 1, 1, 0, 0x58, 5, 0, 5)
			for i in 0 ..< n {
				if i > 0 {
					vop_d_tab(&e.buf, 0xFE, 4, 4, OFF_TAB_ONE_U)
				}
				vzero(&e.buf, u8(i))
				vop_d_tab(&e.buf, 0xDB, 6, 4, OFF_TAB_ABS)  // ymm6 = w & 0x7fffffff (non-negative)
				vop_d_tab(&e.buf, 0xEF, 7, 4, OFF_TAB_ALL1) // MSB=1 where w < 2^31
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, u8(i), 7, GPR_RSI, 6, 2, 0)
				vop38_d(&e.buf, 0x3B, 7, 6, 5)              // vpminud(w_lo, rem)
				vop_d(&e.buf, 0x76, 7, 7, 5)                // ~0 where w_lo >= rem
				vop_d(&e.buf, 0xDF, 7, 7, 4)                // vpandn: MSB=1 where w >= 2^31 && w_lo < rem
				emit_vex_sib(&e.buf, 2, 1, 1, 0, 0x90, u8(i), 7, GPR_RDI, 6, 2, 0)
			}

			patch_i32(e.buf[:], done_patch, i32(len(e.buf) - (done_patch + 4)))
			patch_i32(e.buf[:], done2_patch, i32(len(e.buf) - (done2_patch + 4)))
			for i in 0 ..< n {
				write_slot(e, dst + i, u8(i))
			}

		case .Mask_Copy:
			if dst != a {
				emit_load_mask_eax(&e.buf, a)
				emit_store_mask_al(&e.buf, dst)
			}

		case .Mask_Cond:
			yb := read_slot(e, b, 1)
			vzero(&e.buf, 0)
			vop_d(&e.buf, 0x76, 0, yb, 0)                         // ~0 where b == 0
			emit_vex_rr(&e.buf, 1, 0, 1, 0, 0x50, GPR_RCX, 0, 0)  // vmovmskps ecx, ymm0
			if in_.k == 0 {
				emit_bytes(&e.buf, 0x81, 0xF1, 0xFF, 0x00, 0x00, 0x00) // xor ecx, 0xFF
			}
			emit_load_mask_eax(&e.buf, a)
			emit_bytes(&e.buf, 0x21, 0xC8) // and eax, ecx
			emit_store_mask_al(&e.buf, dst)

		case .Mask_Restore:
			emit_bytes(&e.buf, 0x41, 0x0F, 0xB6) // movzx ecx, byte [r12 + MASK_DISCARD]
			emit_modrm_mem(&e.buf, GPR_RCX, GPR_R12, MASK_DISCARD)
			kills := [3]i32{b, c, in_.d}
			for km in kills {
				if km >= 0 {
					emit_load_mask_eax(&e.buf, km)
					emit_bytes(&e.buf, 0x09, 0xC1) // or ecx, eax
				}
			}
			emit_bytes(&e.buf, 0xF7, 0xD1) // not ecx
			emit_load_mask_eax(&e.buf, a)
			emit_bytes(&e.buf, 0x21, 0xC8) // and eax, ecx
			emit_store_mask_al(&e.buf, dst)

		case .Mask_Or:
			emit_load_mask_eax(&e.buf, b)
			emit_bytes(&e.buf, 0x89, 0xC1) // mov ecx, eax
			emit_load_mask_eax(&e.buf, a)
			emit_bytes(&e.buf, 0x09, 0xC8) // or eax, ecx
			emit_store_mask_al(&e.buf, dst)

		case .Mask_Clear:
			if dst == MASK_EXEC {
				emit_bytes(&e.buf, 0x45, 0x31, 0xED) // xor r13d, r13d
			} else {
				emit_bytes(&e.buf, 0x41, 0xC6)
				emit_modrm_mem(&e.buf, 0, GPR_R12, dst)
				emit_b(&e.buf, 0)
			}

		case .Mask_Kill:
			if dst != MASK_EXEC {
				emit_load_mask_eax(&e.buf, dst)
				emit_bytes(&e.buf, 0x44, 0x09, 0xE8) // or eax, r13d
				emit_store_mask_al(&e.buf, dst)
			}
			emit_bytes(&e.buf, 0x45, 0x31, 0xED) // xor r13d, r13d
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
		}
	}

	pc_off[n_code] = len(e.buf)
	epilogue_off := len(e.buf)

	// Epilogue: persist r13b -> masks[MASK_EXEC], vzeroupper, restore callee-saved GPRs.
	emit_bytes(&e.buf, 0x45, 0x88, 0x2C, 0x24) // mov byte [r12], r13b
	emit_bytes(&e.buf, 0xC5, 0xF8, 0x77)       // vzeroupper
	emit_bytes(&e.buf, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3)

	for jp in e.patches {
		rel := i32(pc_off[jp.target_pc] - (jp.patch_pos + 4))
		patch_i32(e.buf[:], jp.patch_pos, rel)
	}
	for ep in e.epilogues {
		rel := i32(epilogue_off - (ep + 4))
		patch_i32(e.buf[:], ep, rel)
	}
}

@(private)
jit_compile :: proc(p: ^Program) {
	if !jit_avx2_ready || len(p.code) == 0 {
		return
	}
	heap := runtime.default_allocator()
	num_slots := max(p.num_slots, 1)
	max_code := max(p.init_len, len(p.code) - p.init_len)
	e := Jit_Emitter {
		buf       = make([dynamic]u8, 0, max(len(p.code) * 24, 256), heap),
		cross_bb  = make([]bool, num_slots, heap),
		last_use  = make([]i32, num_slots, heap),
		def_bb    = make([]i32, num_slots, heap),
		bb_start  = make([]bool, max_code + 1, heap),
		pc_off    = make([]int, max_code + 1, heap),
		rw        = make([]Instr_RW, max_code, heap),
		patches   = make([dynamic]Jump_Patch, 0, 64, heap),
		epilogues = make([dynamic]int, 0, 8, heap),
	}
	defer delete(e.buf)
	defer delete(e.cross_bb, heap)
	defer delete(e.last_use, heap)
	defer delete(e.def_bb, heap)
	defer delete(e.bb_start, heap)
	defer delete(e.pc_off, heap)
	defer delete(e.rw, heap)
	defer delete(e.patches)
	defer delete(e.epilogues)

	has_init := p.init_len > 0
	if has_init {
		jit_emit_fn(&e, p, p.code[:p.init_len], true)
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
	needed_regs := (max(p.num_slots, 1) + 7) / 8 + 1
	if len(scratch.regs) < needed_regs {
		resize(&scratch.regs, needed_regs)
	}
	needed_masks := (max(p.num_masks, 2) + 7) / 8
	if len(scratch.masks) < needed_masks {
		resize(&scratch.masks, needed_masks)
	}
	aligned := (uintptr(raw_data(scratch.regs)) + 31) &~ uintptr(31)
	slots := ([^]Jit_Slot)(aligned)
	masks := ([^]u8)(raw_data(scratch.masks))

	ctx := Jit_Ctx {
		slots    = slots,
		masks    = masks,
		opts     = opts,
		host_ctx = context,
		lanes    = 0xFF,
	}
	n_buf := min(len(opts.buffers), 4)
	for i in 0 ..< n_buf {
		b := opts.buffers[i]
		ctx.buf_ptr[i] = raw_data(b)
		// Word indices wrap in 32 bits, so clamping len at 2^32 is exact for all u32 indices.
		ctx.buf_len[i] = min(uint(len(b)), uint(1) << 32)
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
		masks[MASK_EXEC] = 0xFF
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
jit_validate_run :: proc(p: ^Program, width, start_x, start_y, end_x, end_y: int, out: []u8, stride: int, targets: [][][4]f32) {
	first_px := start_y * width + start_x
	last_px := (end_y - 1) * width + (end_x - 1)
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
jit_validate_fragment :: proc(p: ^Program, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, sample_masks: []u32) {
	total := quads.count * 4
	_ = shaded[quads.count - 1]
	if quads.helpers != nil {
		_ = quads.helpers[quads.count - 1]
	}
	for v in p.inputs {
		#partial switch v.kind {
		case .Stage_In:
			if v.n > 0 {
				_ = quads.varyings[(int(v.offset) + int(v.n)) * total - 1]
			}
		case .Frag_Coord:
			if v.n > 0 {
				_ = quads.origin[quads.count - 1]
			}
			if v.n > 2 && quads.depth != nil {
				_ = quads.depth[total - 1]
			}
		case .Front_Facing:
			if v.n > 0 && quads.front_facing != nil {
				_ = quads.front_facing[quads.count - 1]
			}
		}
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
	jit_validate_run(p, width, start_x, start_y, end_x, end_y, out, stride, targets)
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
			masks[MASK_EXEC] = exec
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
							base_px := (by + ry) * width + bx
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
		exec := u8((u32(1) << uint(lanes)) - 1)
		masks[MASK_EXEC] = exec
		ctx.lanes = exec
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

@(private, enable_target_feature = "avx2")
jit_run_fragment :: proc(p: ^Program, uniforms: []u8, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, scratch: ^Scratch, opts: Run_Options, sample_masks: []u32) #no_bounds_check {
	jit_validate_fragment(p, quads, targets, shaded, sample_masks)
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	slots := ctx.slots
	masks := ctx.masks
	body_fn := Jit_Fn(p.jit_body)
	total := quads.count * 4

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
	fast_rgba := len(targets) == 1 && targets[0] != nil && sample_masks == nil && outs[0].n == 4 && outs[0].base == .Float
	out0_slot := outs[0].slot

	for first := 0; first < quads.count; first += 2 {
		nq := min(quads.count - first, 2)
		lanes := nq * 4
		for v in p.inputs {
			if v.kind == .Stage_In {
				base_idx := int(v.offset) * total + first * 4
				vptr := raw_data(quads.varyings)
				if nq == 2 {
					for k in 0 ..< int(v.n) {
						(^Jit_Floats)(&slots[v.slot + i32(k)])^ = intrinsics.unaligned_load((^Jit_Floats)(&vptr[base_idx + k * total]))
					}
				} else {
					for k in 0 ..< int(v.n) {
						src := &vptr[base_idx + k * total]
						reg := (^Jit_Floats)(&slots[v.slot + i32(k)])
						for l in 0 ..< 4 {reg[l] = ([^]f32)(src)[l]}
					}
				}
				continue
			}
			for k in 0 ..< int(v.n) {
				col: Jit_Floats
				#partial switch v.kind {
				case .Frag_Coord:
					switch k {
					case 0:
						ox0 := f32(quads.origin[first].x)
						ox1 := f32(quads.origin[first + 1].x) if nq == 2 else 0
						col = {ox0 + 0.5, ox0 + 1.5, ox0 + 0.5, ox0 + 1.5, ox1 + 0.5, ox1 + 1.5, ox1 + 0.5, ox1 + 1.5}
					case 1:
						oy0 := f32(quads.origin[first].y)
						oy1 := f32(quads.origin[first + 1].y) if nq == 2 else 0
						col = {oy0 + 0.5, oy0 + 0.5, oy0 + 1.5, oy0 + 1.5, oy1 + 0.5, oy1 + 0.5, oy1 + 1.5, oy1 + 1.5}
					case 2:
						if quads.depth != nil {
							for l in 0 ..< lanes {col[l] = quads.depth[first * 4 + l]}
						}
					case 3:
						col = 1
					}
				case .Front_Facing:
					if quads.front_facing == nil {
						col = 1
					} else {
						f0 := f32(1) if quads.front_facing[first] else 0
						f1 := (f32(1) if quads.front_facing[first + 1] else 0) if nq == 2 else 0
						col = {f0, f0, f0, f0, f1, f1, f1, f1}
					}
				}
				jit_fill_slot(&slots[v.slot + i32(k)], col, v.base)
			}
		}

		exec := u8(0xFF) if nq == 2 else u8(0x0F)
		masks[MASK_EXEC] = exec
		ctx.lanes = exec
		masks[MASK_DISCARD] = 0
		body_fn(&ctx)

		alive := ~masks[MASK_DISCARD] & exec
		if alive == 0 {
			shaded[first] = {}
			if nq == 2 {
				shaded[first + 1] = {}
			}
			continue
		}

		if fast_rgba {
			t0 := targets[0]
			r0 := (^Jit_Floats)(&slots[out0_slot])
			r1 := (^Jit_Floats)(&slots[out0_slot + 1])
			r2 := (^Jit_Floats)(&slots[out0_slot + 2])
			r3 := (^Jit_Floats)(&slots[out0_slot + 3])
			for qi in 0 ..< nq {
				q := first + qi
				hbits := transmute(u8)(quads.helpers[q]) if quads.helpers != nil else 0
				qalive := ((alive >> uint(qi * 4)) & 0xF) &~ hbits
				shaded[q] = transmute(Quad_Lanes)qalive
				if qalive == 0 {
					continue
				}
				base_l := qi * 4
				base_out := q * 4
				for i in 0 ..< 4 {
					if (qalive & (1 << uint(i))) != 0 {
						l := base_l + i
						t0[base_out + i] = {r0[l], r1[l], r2[l], r3[l]}
					}
				}
			}
			continue
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
					if n_t < len(outs) {
						v := outs[n_t]
						for k in 0 ..< int(v.n) {
							c[k] = float_of(slots[v.slot + i32(k)][l], v.base)
						}
					}
					target[q * 4 + i] = c
				}
				if sample_masks != nil {
					sample_masks[q * 4 + i] = slots[mask_v.slot][l] if has_mask else ~u32(0)
				}
			}
			shaded[q] = lanes_out
		}
	}
}

}
