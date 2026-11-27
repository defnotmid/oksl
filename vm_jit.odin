package oksl

import "base:intrinsics"
import "base:runtime"
import "core:math"
import "core:math/linalg"
import "core:mem/virtual"

// CPU backend, part 3: native JIT compilers (x86-64 AVX2 in vm_jit_x64.odin,
// arm64 NEON in vm_jit_a64.odin) and their shared front end (liveness, the
// slot register cache, eligibility, helpers) and 8-lane runners. Eight lanes
// pack two 2x2 quads into one value: one 256-bit YMM register on x86-64
// (quad 0 in the low 128-bit lane, quad 1 in the high one), a pair of
// 128-bit V registers on arm64 (quad 0 in the first), so fine quad
// derivatives stay in-register, divergence shrinks to two quads per batch,
// and temporary values live directly in registers.

JIT_ENABLED :: #config(OKSL_JIT, true) && (ODIN_ARCH == .amd64 || ODIN_ARCH == .arm64) && (ODIN_OS == .Linux || ODIN_OS == .Darwin)
JIT_X64 :: JIT_ENABLED && ODIN_ARCH == .amd64
JIT_A64 :: JIT_ENABLED && ODIN_ARCH == .arm64

// The target feature the helpers and runners are built with.
@(private)
JIT_FEATURE :: "avx2" when ODIN_ARCH == .amd64 else "neon"

when !JIT_ENABLED {

_ :: intrinsics
_ :: runtime
_ :: math
_ :: linalg
_ :: virtual

@(private)
jit_destroy :: proc(p: ^Program) {}

@(private)
jit_compile :: proc(p: ^Program, fma := true, f16c := true) {}

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
Jit_Ctx :: struct #align(16) {
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
	// arm64's in-block math helper's result (jit_help_math_a64), read back
	// with one ldp of two q registers.
	math_out: Jit_Floats_Q,
}

@(private)
Jit_Floats_Q :: struct #align(16) {
	f: Jit_Floats,
}

#assert(align_of(Jit_Ctx) >= 16)
#assert(size_of(Jit_Slot) == 32)

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

// The bound channels' table, just below the slots in Scratch.regs.
@(private)
JIT_CHAN_BYTES :: SAMPLER_COUNT * size_of(Chan_Img)

#assert(JIT_CHAN_BYTES % size_of(Jit_Slot) == 0)

@(private)
Jit_Vec32 :: struct #align(32) {
	v: Jit_Floats,
}

@(private, rodata)
jit_zero_f: Jit_Vec32

@(private, rodata)
jit_one_f := Jit_Vec32{{1, 1, 1, 1, 1, 1, 1, 1}}

#assert(align_of(Jit_Vec32) >= 32)

@(private)
Jit_Fn :: #type proc "c" (ctx: ^Jit_Ctx)

@(private)
Jit_Help_Fn :: #type proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32)

// Whether jit_compile compiles on this CPU: AVX2 on x86-64, always on arm64.
@(private)
jit_ready: bool

@(init, private)
init_jit :: proc "contextless" () {
	when JIT_A64 {
		jit_a64_init_tables()
		jit_ready = true
	} else {
		jit_ready = jit_x64_init()
	}
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
OFF_CTX_MATH_OUT :: i32(offset_of(Jit_Ctx, math_out))
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

// A bound channel's record (Chan_Img) and its levels' (Chan_Level), as the
// emitters read them.
@(private) OFF_CI_FORMAT :: i32(offset_of(Chan_Img, format))
@(private) OFF_CI_FILTER :: i32(offset_of(Chan_Img, filter))
@(private) OFF_CI_WRAP :: i32(offset_of(Chan_Img, wrap))
@(private) OFF_CI_COMPS :: i32(offset_of(Chan_Img, comps))
@(private) OFF_CI_SHIFT :: i32(offset_of(Chan_Img, shift))
@(private) OFF_CI_HELD :: i32(offset_of(Chan_Img, held))
@(private) OFF_CI_ROUTE :: i32(offset_of(Chan_Img, route))
@(private) OFF_CI_FETCH_ROUTE :: i32(offset_of(Chan_Img, fetch_route))
@(private) OFF_CI_LIN_ROUTE :: i32(offset_of(Chan_Img, lin_route))
@(private) OFF_CI_NLEV :: i32(offset_of(Chan_Img, n_levels))
@(private) OFF_CI_TOP :: i32(offset_of(Chan_Img, top))
@(private) OFF_CI_UNORM :: i32(offset_of(Chan_Img, unorm))
@(private) OFF_CI_LEVELS :: i32(offset_of(Chan_Img, levels))
@(private) OFF_CI_NLAYERS :: i32(offset_of(Chan_Img, n_layers))
@(private) OFF_CI_LTOP :: i32(offset_of(Chan_Img, ltop))
@(private) OFF_CI_LPITCH :: i32(offset_of(Chan_Img, lpitch))
@(private) OFF_LV_PTR :: i32(offset_of(Chan_Level, ptr))
@(private) OFF_LV_MIS :: i32(offset_of(Chan_Level, mis))
@(private) OFF_LV_PITCH :: i32(offset_of(Chan_Level, pitch))
@(private) OFF_LV_SIZE :: i32(offset_of(Chan_Level, size))
@(private) OFF_LV_MAX :: i32(offset_of(Chan_Level, max))
@(private) OFF_LV_FS :: i32(offset_of(Chan_Level, fs))
@(private) OFF_LV_INV :: i32(offset_of(Chan_Level, inv))


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

// ---- basic-block slot register cache ----------------------------------------------------
//
// Registers 0..7 are scratch and 8..15 cache slots within a basic block:
// ymm0..15 on x86-64, the V register pairs v(2r), v(2r + 1) on arm64. The
// emitters move values with jit_ld_slot, jit_st_slot and jit_mov.

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
Jit_Emitter :: struct {
	buf:               [dynamic]u8,
	mem_mask:          []u16,
	tex_mask:          []u8,
	pc_to_bb:          []i32,
	last_use:          []i32,
	def_off:           []i32,
	def_lu:            [dynamic]i32,
	bb_start:          []bool,
	mask_written:      []bool,
	mask_eq_gen:       []u32,
	pc_off:            []int,
	rw:                []Instr_RW,
	patches:           [dynamic]Jump_Patch,
	epilogues:         [dynamic]int,
	sample_calls:      [dynamic]int,
	bilinear_calls:    [dynamic]int,
	fetch_calls:       [dynamic]int,
	decode_calls:      [dynamic]int,
	epilogue_off:      int,
	chan_sample_off:   int,
	chan_bilinear_off: int,
	chan_fetch_off:    int,
	chan_decode_off:   int,
	cache:             [8]Cache_Slot,
	unsupported:       bool, // set by an emitter when lowering fails (e.g. branch range on arm64)
	stamp:             u32,
	exec_gen:          u32,
	pc:                i32,
	cur_dst_base:      i32,
	cur_def_off:       i32,
	cur_mem_mask:      u16,
	fused_mc_pc:       i32,
	fused_mc_inv:      bool,
	exec_ymask_valid:  bool,
	using arch:        Jit_Arch_Emitter,
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
		jit_st_slot(e, e.cache[best].slot, u8(8 + best))
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
		jit_ld_slot(e, r, slot)
		e.stamp += 1
		e.cache[idx] = {slot = slot, last_use = rlu, dirty = false, stamp = e.stamp}
		return r
	}
	jit_ld_slot(e, scratch, slot)
	return scratch
}

// Selects the destination register for `slot` (8..15 when read again in this
// block, else `scratch` in 0..7).
// - `pinned`: 8..15 bits read by this step (never evicted).
// - `no_alias`: 8..15 bits read after `yd` is written (never reused as `yd`).
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
		jit_st_slot(e, slot, yd)
	}
}

@(private)
copy_to_dst :: proc(e: ^Jit_Emitter, slot: i32, src: u8, pinned: u8 = 0, no_alias: u8 = 0) {
	yd := prep_dst(e, slot, src, pinned, no_alias)
	if yd != src {
		jit_mov(e, yd, src)
	}
	commit_dst(e, slot, yd)
}

@(private)
is_helper_instr :: proc(p: ^Program, in_: Instr) -> bool {
	when JIT_A64 {
		// arm64 lowers these in a block of their own (operands and results
		// in memory): helpers and a uniform fast path.
		#partial switch in_.op {
		case .Load_Dyn, .Load_Buf, .Image_Load:
			return true
		}
	}
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
	}
	// F_Fn1 and F_Fn2's vm_math.odin built-ins call a helper inside their
	// basic block (x86's emit_math_call, arm64's a_math_call).
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

@(private, enable_target_feature = JIT_FEATURE)
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

@(private, enable_target_feature = JIT_FEATURE)
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

@(private, enable_target_feature = JIT_FEATURE)
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
	u, v, lod: [8]f32
	sampler_args8(&slots[a][0], &slots[a + 1][0], &slots[b][0], size, in_.op == .Texture_Bias && size.xy != {}, &u[0], &v[0], &lod[0])
	have := u32(ctx.lanes)
	for o := 0; o < JIT_LANES; o += 4 {
		if (exec >> uint(o)) & 0xf == 0 {
			continue
		}
		for l in o ..< o + 4 {
			if (have >> uint(l)) & 1 == 0 {
				continue
			}
			layer := int(f_to_i(fval(fn1(fval(slots[a + 2][l]), .Round_Even)))) if in_.n == 3 else 0
			c := opts.sampler(opts.user, ch, {u[l], v[l]}, lod[l], layer)
			for i in 0 ..< i32(4) {
				slots[dst + i][l] = fbits(c[i])
			}
		}
	}
}

// texelFetch of a channel without an image: the interpreter's lanes
// (texel_fetch_args8, which takes its vectors by pointer: see math_8).
// In compute ctx.lanes is the running lanes, which then call alone.
@(private, enable_target_feature = JIT_FEATURE)
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
	levels := opts.channel_levels[ch] if ch < len(opts.channel_levels) else 0
	pz := &slots[in_.a + 2 if in_.n == 3 else in_.a]
	arr := in_.n == 3
	calls := fetch_quad_lanes8(exec) & u32(ctx.lanes)
	if calls == 0 {
		return
	}
	u, v, lod: [8]f32
	calls &= texel_fetch_args8(&slots[in_.a][0], &slots[in_.a + 1][0], &pz[0], &slots[in_.b][0], size, levels, arr, (in_.d & 1) != 0, &u[0], &v[0], &lod[0])
	for ; calls != 0; calls &= calls - 1 {
		l := int(intrinsics.count_trailing_zeros(calls))
		c := opts.sampler(opts.user, ch, {u[l], v[l]}, lod[l], int(pz[l]) if arr else 0)
		for i in 0 ..< i32(4) {
			slots[dst + i][l] = fbits(c[i])
		}
	}
}

@(private)
jit_help_texture_size :: proc "c" (ctx: ^Jit_Ctx, in_: ^Instr, exec: u32) #no_bounds_check {
	slots := ctx.slots
	ch := int(in_.k)
	chan_imgs := ([^]Chan_Img)(uintptr(slots) - JIT_CHAN_BYTES)
	if im := &chan_imgs[ch]; im.ptr != nil {
		bound_size_lanes(im, in_.d != 0, raw_data(&slots[in_.a]), raw_data(&slots[in_.dst]), raw_data(&slots[in_.dst + 1]), raw_data(&slots[in_.dst + 2]) if in_.n == 3 else nil, JIT_LANES)
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

// ---- IR lowering, shared ----------------------------------------------------------------

// Whether nothing reads `slot` after this instruction writes it.
@(private)
dead_write :: #force_inline proc(e: ^Jit_Emitter, slot: i32) -> bool {
	return write_lu(e, slot) <= e.pc && !need_mem_write(e, slot)
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


// The per-function plan both emitters share: the basic blocks (jump targets,
// helper instructions alone), where MASK_EXEC is still the entry mask
// (returned: Store and Store_Zero before it run unmasked), each
// instruction's reads and writes, the liveness, the masks written, and the
// emitter's per-function lists cleared.
@(private)
jit_plan_fn :: proc(e: ^Jit_Emitter, p: ^Program, code: []Instr, is_init: bool) -> (entry_exec_end: int) {
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
	entry_exec_end = 0
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

	clear(&e.patches)
	clear(&e.epilogues)
	clear(&e.sample_calls)
	clear(&e.bilinear_calls)
	clear(&e.fetch_calls)
	clear(&e.decode_calls)
	e.epilogue_off = 0
	e.chan_sample_off = 0
	e.chan_bilinear_off = 0
	e.chan_fetch_off = 0
	e.chan_decode_off = 0
	jit_arch_reset(e)

	return
}

// At instruction pc: a block's start resets the register cache and computes
// each slot's last use within the block (and each write's, in def_lu);
// inside a block, the previous instruction's writes take their last uses.
@(private)
jit_enter_pc :: proc(e: ^Jit_Emitter, pc, n_code: int) {
	bb_start := e.bb_start[:n_code + 1]
	rw := e.rw[:n_code]
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
}

// The emitted code in executable memory (nil if it can't be mapped): on
// macOS arm64 jit_map_code_darwin's MAP_JIT mapping; elsewhere read-write
// pages, the code copied in, (on arm64) the instruction cache invalidated
// over it, then read-execute.
@(private)
jit_map_code :: proc(code: []u8) -> []u8 {
	when JIT_A64 && ODIN_OS == .Darwin {
		return jit_map_code_darwin(code)
	} else {
		mem_slice, err := virtual.reserve_and_commit(uint(len(code)))
		if err != .None || len(mem_slice) < len(code) {
			return nil
		}
		copy(mem_slice, code)
		when JIT_A64 {
			jit_clear_cache(raw_data(mem_slice), rawptr(uintptr(raw_data(mem_slice)) + uintptr(len(code))))
		}
		if !virtual.protect(raw_data(mem_slice), uint(len(mem_slice)), {.Read, .Execute}) {
			virtual.release(raw_data(mem_slice), uint(len(mem_slice)))
			return nil
		}
		return mem_slice
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

// fma: false compiles without FMA3 where the CPU has it (the tests' other
// path).
@(private)
jit_compile :: proc(p: ^Program, fma := true, f16c := true) {
	if !jit_ready || len(p.code) == 0 || (p.stage == .Compute && !jit_compute_eligible(p)) {
		return
	}
	heap := runtime.default_allocator()
	num_slots := max(p.num_slots, 1)
	max_code := max(p.init_len, len(p.code) - p.init_len)
	e := Jit_Emitter {
		buf            = make([dynamic]u8, 0, max(len(p.code) * 24, 256), heap),
		mem_mask       = make([]u16, max_code, heap),
		tex_mask       = make([]u8, max_code, heap),
		pc_to_bb       = make([]i32, max_code + 1, heap),
		last_use       = make([]i32, num_slots, heap),
		def_off        = make([]i32, max_code, heap),
		def_lu         = make([dynamic]i32, 0, 256, heap),
		bb_start       = make([]bool, max_code + 1, heap),
		mask_written   = make([]bool, max(p.num_masks, 2), heap),
		mask_eq_gen    = make([]u32, max(p.num_masks, 2), heap),
		pc_off         = make([]int, max_code + 1, heap),
		rw             = make([]Instr_RW, max_code, heap),
		patches        = make([dynamic]Jump_Patch, 0, 64, heap),
		epilogues      = make([dynamic]int, 0, 8, heap),
		sample_calls   = make([dynamic]int, 0, 8, heap),
		bilinear_calls = make([dynamic]int, 0, 8, heap),
		fetch_calls    = make([dynamic]int, 0, 8, heap),
		decode_calls   = make([dynamic]int, 0, 16, heap),
	}
	jit_arch_init(&e, p, num_slots, fma, f16c, heap)
	defer delete(e.buf)
	defer delete(e.mem_mask, heap)
	defer delete(e.tex_mask, heap)
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
	defer delete(e.sample_calls)
	defer delete(e.bilinear_calls)
	defer delete(e.fetch_calls)
	defer delete(e.decode_calls)
	defer jit_arch_destroy(&e, heap)

	has_init := p.init_len > 0
	init_bytes := 0
	if has_init {
		jit_emit_fn(&e, p, p.code[:p.init_len], true)
		if e.unsupported {
			return
		}
		init_bytes = len(e.buf)
		for (len(e.buf) & 31) != 0 {
			append(&e.buf, JIT_PAD)
		}
	}
	body_off := len(e.buf)
	jit_emit_fn(&e, p, p.code[p.init_len:], false)
	if e.unsupported {
		// An instruction this JIT doesn't lower: the program runs in the
		// interpreter.
		return
	}

	total := len(e.buf)
	mem_slice := jit_map_code(e.buf[:])
	if mem_slice == nil {
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

@(private, enable_target_feature = JIT_FEATURE)
jit_begin :: proc(p: ^Program, scratch: ^Scratch, uniforms: []u8, opts: ^Run_Options) -> Jit_Ctx #no_bounds_check {
	needed_regs := (max(p.num_slots, 1) * size_of(Jit_Slot) + JIT_CHAN_BYTES + size_of(Reg) - 1) / size_of(Reg) + 1
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
		fill_chan_imgs(chan_imgs[:meta.num_chans], opts.channel_images)
	}
	slots := ([^]Jit_Slot)(aligned + JIT_CHAN_BYTES)
	assert(uintptr(slots) % 32 == 0)
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

@(private, enable_target_feature = JIT_FEATURE)
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

@(private, enable_target_feature = JIT_FEATURE)
jit_pack_u8 :: #force_inline proc "contextless" (v: Simd_F8) -> Simd_U8 {
	z := Simd_F8(0)
	clamped := intrinsics.simd_select(intrinsics.simd_lanes_gt(v, z), intrinsics.simd_min(v, Simd_F8(1)), z)
	return transmute(Simd_U8)Simd_I8(clamped * Simd_F8(255) + Simd_F8(0.5))
}

// The first and last pixel the run writes in every target and in the RGBA8
// output, read with bounds checks (jit_runner_run has none), so a short
// buffer traps before anything is written.
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

// The layer runner's tile: 4x2 pixels, quads side by side, lane l at
// (JIT_RUN_DX[l], JIT_RUN_DY[l]) - 0.5 from its top-left pixel.
@(private)
JIT_RUN_DX :: Simd_F8{0.5, 1.5, 0.5, 1.5, 2.5, 3.5, 2.5, 3.5}
@(private)
JIT_RUN_DY :: Simd_F8{0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5}
@(private, rodata)
JIT_RUN_LX := [JIT_LANES]int{0, 1, 0, 1, 2, 3, 2, 3}
@(private, rodata)
JIT_RUN_LY := [JIT_LANES]int{0, 0, 1, 1, 0, 0, 1, 1}

// An input component from a Varying: a + dx * x + dy * y at the lanes' pixel centres.
@(private)
Jit_Run_Varying :: struct {
	dst:       ^Simd_F8,
	a, dx, dy: f32,
}

// A component converted from float columns to `base`, every tile: an integer
// input staged as floats, or an integer output for the float stores.
@(private)
Jit_Run_Conv :: struct {
	src:  ^Jit_Slot,
	dst:  ^Jit_Slot,
	base: Base,
}

// Input components at most: 16 locations of 4, gl_FragCoord's 4 and
// gl_FrontFacing.
@(private)
JIT_RUN_MAX_IN :: 16 * 4 + 4 + 1

// What one call fills per row and per tile: each input component as a
// destination, written as floats (staged, then converted, when its base
// isn't float). gl_FragCoord and location 0 occur at most once, so their
// destinations are single (nil when absent); the counts say what is set.
@(private)
Jit_Run_Inputs :: struct {
	x, y:    ^Simd_F8, // gl_FragCoord.x, .y
	ux, uy:  ^Simd_F8, // location 0's uv without varyings
	vars:    [JIT_RUN_MAX_IN]Jit_Run_Varying,
	convs:   [JIT_RUN_MAX_IN]Jit_Run_Conv,
	staged:  [JIT_RUN_MAX_IN]Simd_F8,
	nvars:   int,
	nconvs:  int,
	nstaged: int,
}

@(private)
jit_run_input_dst :: proc(ins: ^Jit_Run_Inputs, slot: ^Jit_Slot, base: Base) -> ^Simd_F8 {
	if base == .Float {
		return (^Simd_F8)(slot)
	}
	staged := &ins.staged[ins.nstaged]
	ins.nstaged += 1
	ins.convs[ins.nconvs] = {src = (^Jit_Slot)(staged), dst = slot, base = base}
	ins.nconvs += 1
	return staged
}

// The inputs' plan, and their components that are the same in every tile
// (gl_FragCoord.zw, gl_FrontFacing, inputs without a varying) filled once:
// input slots are only read.
@(private, enable_target_feature = JIT_FEATURE)
jit_run_plan_inputs :: proc(ins: ^Jit_Run_Inputs, p: ^Program, slots: [^]Jit_Slot, varyings: []Varying) {
	ins.x, ins.y, ins.ux, ins.uy = nil, nil, nil, nil
	ins.nvars, ins.nconvs, ins.nstaged = 0, 0, 0
	for v in p.inputs {
		for k in 0 ..< int(v.n) {
			slot := &slots[v.slot + i32(k)]
			#partial switch v.kind {
			case .Frag_Coord:
				switch k {
				case 0:
					ins.x = jit_run_input_dst(ins, slot, v.base)
				case 1:
					ins.y = jit_run_input_dst(ins, slot, v.base)
				case:
					jit_fill_slot(slot, 1 if k == 3 else 0, v.base)
				}
			case .Front_Facing:
				// A layer faces the camera.
				jit_fill_slot(slot, 1, v.base)
			case:
				if int(v.location) < len(varyings) {
					vy := varyings[v.location]
					ins.vars[ins.nvars] = {jit_run_input_dst(ins, slot, v.base), vy.a[k], vy.dx[k], vy.dy[k]}
					ins.nvars += 1
				} else if v.location == 0 && k == 0 {
					ins.ux = jit_run_input_dst(ins, slot, v.base)
				} else if v.location == 0 && k == 1 {
					ins.uy = jit_run_input_dst(ins, slot, v.base)
				} else {
					jit_fill_slot(slot, 0, v.base)
				}
			}
		}
	}
}

// Runs a fragment program over a layer (runner_run validated the options) in
// 4x2 tiles, two quads each. What a call needs is planned once: the input
// columns by how they step (per tile with x, per row with y, constant), the
// targets with their four columns (missing components from constant
// columns, integer outputs converted into floats), and the lanes' pixel
// offsets; each tile then fills its spans, runs, and stores the pixels its
// region and discards leave: per float target one 4x2 transpose, then each
// pixel's [4]f32 (32-byte pair stores measured slower: on Zen 5 the next
// tile's body stalls on a full retire queue), and RGBA8 a row at a time.
@(private, enable_target_feature = JIT_FEATURE)
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
	when JIT_A64 {
		assert(uintptr(&ctx.math_out) % 16 == 0)
	}
	slots := ctx.slots
	masks := ctx.masks
	body_fn := Jit_Fn(p.jit_body)

	ins: Jit_Run_Inputs = ---
	jit_run_plan_inputs(&ins, p, slots, opts.varyings)

	// The outputs: a float target per written location, RGBA8 from location 0.
	targs: [FRAGMENT_OUTPUTS]Jit_Out_Target
	n_targs := 0
	rgba8: Jit_Out_Target
	has_rgba8 := false
	out_convs: [FRAGMENT_OUTPUTS * 4]Jit_Run_Conv
	out_cols: [FRAGMENT_OUTPUTS * 4]Simd_F8
	n_out_convs := 0
	for v in p.outputs {
		if v.kind != .Stage_Out {
			continue
		}
		target := targets[v.location] if int(v.location) < len(targets) else nil
		bytes := v.location == 0 && out != nil
		if target == nil && !bytes {
			continue
		}
		cols := [4]^Jit_Floats{&jit_zero_f.v, &jit_zero_f.v, &jit_zero_f.v, &jit_one_f.v}
		for k in 0 ..< int(v.n) {
			slot := &slots[v.slot + i32(k)]
			if v.base == .Float {
				cols[k] = (^Jit_Floats)(slot)
				continue
			}
			col := &out_cols[n_out_convs]
			out_convs[n_out_convs] = {src = slot, dst = (^Jit_Slot)(col), base = v.base}
			n_out_convs += 1
			cols[k] = (^Jit_Floats)(col)
		}
		if target != nil {
			targs[n_targs] = {dst = raw_data(target), c0 = cols[0], c1 = cols[1], c2 = cols[2], c3 = cols[3]}
			n_targs += 1
		}
		if bytes {
			rgba8 = {c0 = cols[0], c1 = cols[1], c2 = cols[2], c3 = cols[3]}
			has_rgba8 = true
		}
	}
	bptr := raw_data(out)
	// Each lane's pixel from its tile's top-left: in a target, in the bytes.
	lane_off, lane_boff: [JIT_LANES]int
	for l in 0 ..< JIT_LANES {
		lane_off[l] = JIT_RUN_LY[l] * out_stride + JIT_RUN_LX[l]
		lane_boff[l] = JIT_RUN_LY[l] * stride + JIT_RUN_LX[l] * 4
	}

	fw, fh := Simd_F8(f32(width)), Simd_F8(f32(height))
	for by := start_y &~ 1; by < end_y; by += 2 {
		py := Simd_F8(f32(by)) + JIT_RUN_DY
		if ins.y != nil {ins.y^ = py}
		if ins.uy != nil {ins.uy^ = py / fh}
		// The row's lanes in the region, and its pixel (0, by) in a target and the bytes.
		row_mask := (u8(0x33) if by >= start_y else 0) | (u8(0xCC) if by + 1 < end_y else 0)
		row_px := (by - out_y) * out_stride - out_x
		row_byte := by * stride
		for bx := start_x &~ 1; bx < end_x; bx += 4 {
			px := Simd_F8(f32(bx)) + JIT_RUN_DX
			if ins.x != nil {ins.x^ = px}
			if ins.ux != nil {ins.ux^ = px / fw}
			for vr in ins.vars[:ins.nvars] {
				vr.dst^ = Simd_F8(vr.a) + Simd_F8(vr.dx) * px + Simd_F8(vr.dy) * py
			}
			for c in ins.convs[:ins.nconvs] {
				jit_fill_slot(c.dst, (^Jit_Floats)(c.src)^, c.base)
			}

			// The quads the region touches run, all four lanes of each (as
			// a GPU's helpers): the left one always, the right one unless
			// it starts past the region.
			ctx.lanes = 0x0F | (u8(0xF0) if bx + 2 < end_x else 0)
			masks[MASK_DISCARD] = 0
			body_fn(&ctx)

			col_mask := (u8(0x05) if bx >= start_x else 0) | (u8(0x0A) if bx + 1 < end_x else 0) | (u8(0x50) if bx + 2 < end_x else 0) | (u8(0xA0) if bx + 3 < end_x else 0)
			write := ~masks[MASK_DISCARD] & row_mask & col_mask
			if write == 0 {
				continue
			}
			for c in out_convs[:n_out_convs] {
				for l in 0 ..< JIT_LANES {
					(^Jit_Floats)(c.dst)[l] = float_of(c.src[l], c.base)
				}
			}
			tile_px := row_px + bx
			tile_byte := row_byte + bx * 4
			if write == 0xFF {
				for t in targs[:n_targs] {
					pixels := transmute([JIT_LANES][4]f32)jit_out_pixels(t)
					for l in 0 ..< JIT_LANES {
						t.dst[tile_px + lane_off[l]] = pixels[l]
					}
				}
				if has_rgba8 {
					rgba := jit_pack_rgba8(rgba8)
					intrinsics.unaligned_store((^[4]u32)(&bptr[tile_byte]), [4]u32{rgba[0], rgba[1], rgba[4], rgba[5]})
					intrinsics.unaligned_store((^[4]u32)(&bptr[tile_byte + stride]), [4]u32{rgba[2], rgba[3], rgba[6], rgba[7]})
				}
				continue
			}
			for t in targs[:n_targs] {
				pixels := transmute([JIT_LANES][4]f32)jit_out_pixels(t)
				for m := write; m != 0; m &= m - 1 {
					l := intrinsics.count_trailing_zeros(m)
					t.dst[tile_px + lane_off[l]] = pixels[l]
				}
			}
			if has_rgba8 {
				rgba := jit_pack_rgba8(rgba8)
				for m := write; m != 0; m &= m - 1 {
					l := intrinsics.count_trailing_zeros(m)
					intrinsics.unaligned_store((^u32)(&bptr[tile_byte + lane_boff[l]]), rgba[l])
				}
			}
		}
	}
}

@(private, enable_target_feature = JIT_FEATURE)
jit_run_vertex :: proc(p: ^Program, uniforms: []u8, attributes: []Attribute, count: int, position: [][4]f32, varyings: []f32, scratch: ^Scratch, opts: Run_Options) #no_bounds_check {
	jit_validate_vertex(p, count, position, varyings)
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	when JIT_A64 {
		assert(uintptr(&ctx.math_out) % 16 == 0)
	}
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
			if v.kind == .Instance_ID {
				slots[v.slot] = u32(i32(opts_local.instance))
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

// An output's 8 lanes as RGBA pixels, two lanes a register: lanes 0-1, 2-3,
// 4-5 and 6-7.
@(private, enable_target_feature = JIT_FEATURE)
jit_out_pixels :: #force_inline proc "contextless" (t: Jit_Out_Target) -> [4]Simd_F8 #no_bounds_check {
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
	return {
		intrinsics.simd_shuffle(p04, p15, 0, 1, 2, 3, 8, 9, 10, 11),
		intrinsics.simd_shuffle(p26, p37, 0, 1, 2, 3, 8, 9, 10, 11),
		intrinsics.simd_shuffle(p04, p15, 4, 5, 6, 7, 12, 13, 14, 15),
		intrinsics.simd_shuffle(p26, p37, 4, 5, 6, 7, 12, 13, 14, 15),
	}
}

// Two quads' pixels, consecutive from base_out (run_fragment).
@(private, enable_target_feature = JIT_FEATURE)
jit_store_pair :: #force_inline proc "contextless" (t: Jit_Out_Target, base_out: int) #no_bounds_check {
	px := jit_out_pixels(t)
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 0]), px[0])
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 2]), px[1])
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 4]), px[2])
	intrinsics.unaligned_store((^Simd_F8)(&t.dst[base_out + 6]), px[3])
}

// An output's 8 lanes as RGBA8 pixels, as run rounds them.
@(private, enable_target_feature = JIT_FEATURE)
jit_pack_rgba8 :: #force_inline proc "contextless" (t: Jit_Out_Target) -> [JIT_LANES]u32 #no_bounds_check {
	rgba := jit_pack_u8((^Simd_F8)(t.c0)^) |
	        intrinsics.simd_shl(jit_pack_u8((^Simd_F8)(t.c1)^), Simd_U8(8)) |
	        intrinsics.simd_shl(jit_pack_u8((^Simd_F8)(t.c2)^), Simd_U8(16)) |
	        intrinsics.simd_shl(jit_pack_u8((^Simd_F8)(t.c3)^), Simd_U8(24))
	return transmute([JIT_LANES]u32)rgba
}

@(private, enable_target_feature = JIT_FEATURE)
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

// The pair's sample masks from base, for its alive lanes only, as its
// colors: the shader's gl_SampleMask, or every sample (src all ones).
@(private)
jit_store_sample_masks :: #force_inline proc "contextless" (dst: []u32, src: ^Jit_Slot, base: int, alive: u8) #no_bounds_check {
	if alive == 0xFF {
		intrinsics.unaligned_store((^Jit_Slot)(&dst[base]), src^)
		return
	}
	for l in 0 ..< JIT_LANES {
		if alive & (1 << uint(l)) != 0 {
			dst[base + l] = src[l]
		}
	}
}

@(private, rodata)
jit_all_samples := Jit_Slot{~u32(0), ~u32(0), ~u32(0), ~u32(0), ~u32(0), ~u32(0), ~u32(0), ~u32(0)}

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

@(private, enable_target_feature = JIT_FEATURE)
jit_run_fragment :: proc(p: ^Program, uniforms: []u8, quads: Fragment_Quads, targets: [][][4]f32, shaded: []Quad_Lanes, scratch: ^Scratch, opts: Run_Options, sample_masks: []u32) #no_bounds_check {
	meta := (^Jit_Meta)(p.jit_meta)
	jit_validate_fragment(p, meta, quads, targets, shaded, sample_masks)
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	when JIT_A64 {
		assert(uintptr(&ctx.math_out) % 16 == 0)
	}
	slots := ctx.slots
	masks := ctx.masks
	body_fn := Jit_Fn(p.jit_body)
	total := quads.count * 4

	active_outs: [FRAGMENT_OUTPUTS]Jit_Out_Target
	n_active := 0
	fast_outs := len(targets) <= FRAGMENT_OUTPUTS
	mask_src := &slots[meta.sample_mask_slot] if meta.has_sample_mask else &jit_all_samples
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
				c0  = (^Jit_Floats)(&slots[v.slot]) if v.n > 0 else &jit_zero_f.v,
				c1  = (^Jit_Floats)(&slots[v.slot + 1]) if v.n > 1 else &jit_zero_f.v,
				c2  = (^Jit_Floats)(&slots[v.slot + 2]) if v.n > 2 else &jit_zero_f.v,
				c3  = (^Jit_Floats)(&slots[v.slot + 3]) if v.n > 3 else &jit_one_f.v,
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
			if sample_masks != nil {
				jit_store_sample_masks(sample_masks, mask_src, first * 4, alive)
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
			if sample_masks != nil {
				jit_store_sample_masks(sample_masks, mask_src, first * 4, alive)
			}
		} else {
			jit_write_fragment_slow(meta, slots, quads, targets, shaded, sample_masks, first, 1, alive)
		}
	}
}

// Runs workgroups [start, end) of a compute dispatch (run_compute validated the
// options and the range): each workgroup's invocations in 8-lane batches of
// consecutive gl_LocalInvocationIndex, the last batch partial.
@(private, enable_target_feature = JIT_FEATURE)
jit_run_compute :: proc(p: ^Program, uniforms: []u8, groups: [3]int, start, end: int, scratch: ^Scratch, opts: Run_Options) #no_bounds_check {
	opts_local := opts
	ctx := jit_begin(p, scratch, uniforms, &opts_local)
	when JIT_A64 {
		assert(uintptr(&ctx.math_out) % 16 == 0)
	}
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
