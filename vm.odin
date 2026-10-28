package oksl

// CPU backend, part 1: lowering. Functions are inlined (recursion is
// rejected) into one flat instruction array over SoA registers: a slot is
// one scalar component for LANES pixels. Divergent control flow runs every
// path with execution masks; only stores to variables honour the mask, so
// temporaries are computed for all lanes (helper lanes keep derivatives
// right).

LANES :: 64
// A set of lanes (execution, discard, break, ... masks).
Lane_Bits :: u16 when LANES <= 16 else (u32 when LANES <= 32 else u64)
Lane_Mask :: bit_set[0 ..< LANES;Lane_Bits]
ALL_LANES :: ~Lane_Mask{}

// Mask registers 0 and 1: the execution mask and the discarded lanes.
MASK_EXEC :: 0
MASK_DISCARD :: 1

VM_Op :: enum u8 {
	Nop,
	// Unmasked moves into temporaries.
	Mov,
	Splat,
	// dst + i = slot a, b, c or d for i < n (n <= 4): swizzles and
	// constructors from scattered components.
	Gather,
	// Masked stores into variables: dst = a for the lanes in exec.
	Store,
	Store_Zero,
	// Per-lane offsets in slot b (components): dst = a[off], a[off] = src.
	Load_Dyn,
	Store_Dyn,
	// k marks an operand that is a scalar, read for every component (1: a,
	// 2: b), instead of broadcast into a temporary.
	F_Add,
	F_Sub,
	F_Mul,
	F_Div,
	F_Neg,
	I_Add,
	I_Sub,
	I_Mul,
	I_Div,
	U_Div,
	I_Rem,
	U_Rem,
	I_Neg,
	F_Lt,
	F_Le,
	F_Gt,
	F_Ge,
	F_Eq,
	F_Ne,
	I_Lt,
	I_Le,
	I_Gt,
	I_Ge,
	U_Lt,
	U_Le,
	U_Gt,
	U_Ge,
	// Integer and bool (in)equality.
	I_Eq,
	I_Ne,
	B_And,
	B_Or,
	B_Not,
	Bit_And,
	Bit_Or,
	Bit_Xor,
	Bit_Not,
	Shl,
	Shr_I,
	Shr_U,
	F_To_I,
	F_To_U,
	I_To_F,
	U_To_F,
	B_To_F,
	F_To_B,
	I_To_B,
	// dst = c-lane ? a : b, per component (cond has n components).
	Select,
	// Scalar bool: all / any of n components.
	All,
	Any,
	// k = Builtin: one, two or three float operands per component.
	F_Fn1,
	F_Fn2,
	F_Fn3,
	// k = Builtin: integer abs/sign/min/max/clamp (signed); U_* unsigned.
	I_Fn1,
	I_Fn2,
	I_Fn3,
	U_Fn2,
	U_Fn3,
	Dot,
	Length,
	Distance,
	Normalize,
	Cross,
	Faceforward,
	Reflect,
	Refract,
	// dst (n x d) = a (n rows x k cols) * b (k rows x d cols), column-major.
	Mat_Mul,
	// dst (k) = a (row vector of n) * b (n rows x k cols).
	Vec_Mat,
	// a is n rows x k cols.
	Transpose,
	Determinant,
	Inverse,
	// dst (n rows x k cols) = a (n) * b (k)^T.
	Outer,
	Deriv_X,
	Deriv_Y,
	Fwidth,
	// dst vec4 = channel k sampled at uv a, explicit lod b.
	Texture,
	// The same at the lod a's quad derivatives pick, plus bias b.
	Texture_Bias,
	// dst ivec2 = channel k's size at lod a.
	Texture_Size,
	// Masks: dst = a.
	Mask_Copy,
	// dst = the lanes of a where cond slot b holds (k = 1: where it does not).
	Mask_Cond,
	// dst = a without the lanes of b, c, d and the discarded ones; unused
	// kill masks are -1.
	Mask_Restore,
	// dst = a + b (union).
	Mask_Or,
	Mask_Clear,
	// dst += exec, exec = {} (break, continue, return, discard).
	Mask_Kill,
	Jump,
	// Jump to a when exec is empty.
	Jump_If_None,
}

// The ops that take a Builtin in k.
FN_OPS :: bit_set[VM_Op;u128]{.F_Fn1, .F_Fn2, .F_Fn3, .I_Fn1, .I_Fn2, .I_Fn3, .U_Fn2, .U_Fn3}

Instr :: struct {
	op:            VM_Op,
	n:             u8,
	k:             u8,
	dst, a, b, c:  i32,
	d:             i32,
}

// Copies a uniform from the block into packed slots before a run.
Uniform_Fill :: struct {
	slot:   i32,
	offset: i32,
	base:   Base,
}

Program :: struct {
	code:       []Instr,
	num_slots:  int,
	num_masks:  int,
	// (slot, bits) constants written once per run.
	consts:     [][2]u32,
	fills:      []Uniform_Fill,
	// The stage inputs the runner fills per pixel (Stage_In, Frag_Coord)
	// and the outputs it reads back (Stage_Out).
	inputs:     []Program_Var,
	outputs:    []Program_Var,
	// Floats per vertex in the varyings' record.
	varying_floats: int,
}

// A stage variable's registers.
Program_Var :: struct {
	kind:     Interface_Kind,
	location: i32,
	slot:     i32,
	n:        i32,
	base:     Base,
	// Varyings: the float offset in the record (Interface_Var.offset).
	offset:   i32,
}

@(private)
Val :: struct {
	slot: i32,
	t:    Type,
}

// A place in registers: `base` (+ per-lane component offsets in `dyn`),
// optionally swizzled.
@(private)
LV :: struct {
	base:    i32,
	dyn:     i32,
	t:       Type,
	swz:     [4]u8,
	swz_len: int,
}

@(private)
Lower :: struct {
	u:         ^Unit,
	checked:   ^Checked,
	code:      [dynamic]Instr,
	next_slot: i32,
	next_mask: i32,
	sym_slot:  []i32,
	// Channel each sampler symbol refers to.
	sym_chan:  []i32,
	consts:    map[u32]i32,
	const_list: [dynamic][2]u32,
	fills:     [dynamic]Uniform_Fill,
	// Kill masks of the innermost loop and inlined function (-1: none).
	brk, cont: i32,
	ret:       i32,
	// Where a framed function's returns store (-1: unframed, or void).
	ret_slot:  i32,
	// An unframed function's value, from its only return (-1: none yet),
	// and the first slot made during its call.
	ret_val:   i32,
	ret_mark:  i32,
	// The symbols the program writes (assigned, incremented or passed out).
	written:   []bool,
	// The slots of the outputs and globals, [lo, hi): what an inlined
	// function can write besides its own variables.
	globals:   [2]i32,
}

build_program :: proc(u: ^Unit, c: ^Checked) -> Program {
	l := Lower {
		u         = u,
		checked   = c,
		code      = make([dynamic]Instr),
		next_mask = 2,
		sym_slot  = make([]i32, len(u.syms)),
		sym_chan  = make([]i32, len(u.syms)),
		consts    = make(map[u32]i32),
		const_list = make([dynamic][2]u32),
		fills     = make([dynamic]Uniform_Fill),
		brk       = -1,
		cont      = -1,
		ret       = -1,
		ret_slot  = -1,
		ret_val   = -1,
	}
	for &ch in l.sym_chan {
		ch = -1
	}
	l.written = written_syms(u)
	p: Program
	inputs := make([dynamic]Program_Var)
	outputs := make([dynamic]Program_Var)
	stage_var :: proc(l: ^Lower, row: Interface_Var) -> Program_Var {
		t := sym_type(l.u, row.sym)
		v := Program_Var{row.kind, row.binding, alloc_slots(l, cc(l, t)), i32(cc(l, t)), t.base, row.offset}
		l.sym_slot[row.sym] = v.slot
		return v
	}
	// The inputs, the uniforms, then the outputs and globals in one range.
	for row in c.interface {
		#partial switch row.kind {
		case .Channel:
			l.sym_chan[row.sym] = row.binding
		case .Stage_In, .Frag_Coord, .Front_Facing:
			append(&inputs, stage_var(&l, row))
		}
	}
	p.inputs = inputs[:]
	p.varying_floats = c.varying_floats
	// Uniform slots, filled from the block before each run.
	for sym, i in u.syms {
		// Uniforms left out of the block (unused, index -1) have no slots.
		if i == 0 || (sym.storage != .Uniform && sym.storage != .Builtin_Uniform) || sym.index < 0 {
			continue
		}
		t := ty(u, sym.type)
		slot := alloc_slots(&l, component_count(u, t))
		l.sym_slot[i] = slot
		elem := elem_type(t)
		count := max(int(t.array), 1)
		stride := std140_array_stride(t) if t.array > 0 else 0
		k := slot
		for e in 0 ..< count {
			for col in 0 ..< max(int(elem.cols), 1) {
				for row in 0 ..< int(elem.rows) {
					off := i32(sym.index + e * stride + col * 16 + row * 4)
					append(&l.fills, Uniform_Fill{k, off, elem.base})
					k += 1
				}
			}
		}
	}

	// Outputs and globals start at zero; the entry runs the globals'
	// initializers.
	l.globals[0] = l.next_slot
	for row in c.interface {
		if row.kind == .Stage_Out || row.kind == .Position {
			append(&outputs, stage_var(&l, row))
		}
	}
	p.outputs = outputs[:]
	for v in p.outputs {
		emit(&l, .Store_Zero, n = int(v.n), dst = v.slot)
	}
	for g in c.globals {
		t := sym_type(u, g)
		l.sym_slot[g] = alloc_slots(&l, component_count(u, t))
		emit(&l, .Store_Zero, n = component_count(u, t), dst = l.sym_slot[g])
	}
	l.globals[1] = l.next_slot
	inline_call(&l, c.entry, nil, nil)

	p.code = l.code[:]
	p.num_slots = int(l.next_slot)
	p.num_masks = int(l.next_mask)
	p.consts = l.const_list[:]
	p.fills = l.fills[:]
	return p
}

// The symbols some expression writes: the targets of assignments and
// increments and the arguments of out and inout parameters (each the
// variable at the root of its swizzles, fields and indices).
@(private)
written_syms :: proc(u: ^Unit) -> []bool {
	written := make([]bool, len(u.syms))
	for e, id in u.exprs {
		#partial switch e.kind {
		case .Assign, .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
			written[target_sym(u, e.a)] = true
		case .Call:
			if e.call != .User {
				continue
			}
			ps := fparams(u, e.func)
			for a, i in args_of(u, Expr_Id(id)) {
				if i < len(ps) && ps[i].qual != .In {
					written[target_sym(u, a)] = true
				}
			}
		}
	}
	return written
}

// ---- helpers ----------------------------------------------------------------------------

@(private)
emit :: proc(l: ^Lower, op: VM_Op, n := 1, k := 0, dst: i32 = 0, a: i32 = 0, b: i32 = 0, c: i32 = 0, d: i32 = 0) -> int {
	append(&l.code, Instr{op = op, n = u8(n), k = u8(k), dst = dst, a = a, b = b, c = c, d = d})
	return len(l.code) - 1
}

@(private)
here :: proc(l: ^Lower) -> i32 {
	return i32(len(l.code))
}

@(private)
alloc_slots :: proc(l: ^Lower, n: int) -> i32 {
	s := l.next_slot
	l.next_slot += i32(max(n, 1))
	return s
}

@(private)
alloc_mask :: proc(l: ^Lower) -> i32 {
	m := l.next_mask
	l.next_mask += 1
	return m
}

@(private)
const_slot :: proc(l: ^Lower, bits: u32) -> i32 {
	if s, ok := l.consts[bits]; ok {
		return s
	}
	s := alloc_slots(l, 1)
	l.consts[bits] = s
	append(&l.const_list, [2]u32{u32(s), bits})
	return s
}

// The bits slot `s` holds when it is a constant. Slots are allocated in
// increasing order, so const_list is in slot order: a binary search.
@(private)
slot_const :: proc(l: ^Lower, s: i32) -> (bits: u32, ok: bool) {
	lo, hi := 0, len(l.const_list)
	for lo < hi {
		mid := (lo + hi) / 2
		if l.const_list[mid][0] < u32(s) {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == len(l.const_list) || l.const_list[lo][0] != u32(s) {
		return 0, false
	}
	return l.const_list[lo][1], true
}

// Consecutive slots holding `bits`, written once per run like any constant
// (a broadcast or a constructor of constants costs nothing per batch).
@(private)
const_vec :: proc(l: ^Lower, bits: []u32) -> i32 {
	if len(bits) == 1 {
		return const_slot(l, bits[0])
	}
	s := alloc_slots(l, len(bits))
	for b, i in bits {
		append(&l.const_list, [2]u32{u32(s + i32(i)), b})
	}
	return s
}

// The constants these slots hold, when every one is a constant.
@(private)
consts_of :: proc(l: ^Lower, slots: []i32) -> (bits: []u32, ok: bool) {
	for s in slots {
		slot_const(l, s) or_return
	}
	bits = make([]u32, len(slots), context.temp_allocator)
	for s, i in slots {
		bits[i], _ = slot_const(l, s)
	}
	return bits, true
}

@(private)
const_val :: proc(l: ^Lower, base: Base, v: f64) -> Val {
	bits: u32
	switch base {
	case .Float:
		bits = transmute(u32)f32(v)
	case .Int:
		bits = u32(i32(v))
	case .Uint, .Bool:
		bits = u32(v)
	case .Void, .Sampler, .Struct:
	}
	return {const_slot(l, bits), {base = base, rows = 1}}
}

@(private)
cc :: proc(l: ^Lower, t: Type) -> int {
	return component_count(l.u, t)
}

// A temporary holding `v` broadcast to `n` components.
@(private)
widen :: proc(l: ^Lower, v: Val, n: int) -> i32 {
	if cc(l, v.t) == n || n == 1 {
		return v.slot
	}
	if bits, ok := slot_const(l, v.slot); ok {
		splat := make([]u32, n, context.temp_allocator)
		for &b in splat {
			b = bits
		}
		return const_vec(l, splat)
	}
	s := alloc_slots(l, n)
	emit(l, .Splat, n = n, dst = s, a = v.slot)
	return s
}

// Masked store of a value's components.
@(private)
store_val :: proc(l: ^Lower, dst: i32, v: Val) {
	emit(l, .Store, n = cc(l, v.t), dst = dst, a = v.slot)
}

@(private)
kills :: proc(l: ^Lower) -> (i32, i32, i32) {
	return l.brk, l.cont, l.ret
}

// ---- lvalues ----------------------------------------------------------------------------

@(private)
lower_lvalue :: proc(l: ^Lower, id: Expr_Id) -> LV {
	u := l.u
	e := u.exprs[id]
	#partial switch e.kind {
	case .Ident:
		return {base = l.sym_slot[e.sym], dyn = -1, t = sym_type(u, e.sym)}
	case .Index:
		lv := lower_lvalue(l, e.a)
		bt := lv.t
		limit: int
		rt := expr_type(u, id)
		switch {
		case is_array(bt):
			limit = int(bt.array)
		case is_matrix(bt):
			limit = int(bt.cols)
		case:
			limit = int(bt.rows)
		}
		size := cc(l, rt)
		ix := u.exprs[e.b]
		if ix.kind == .Int_Lit || ix.kind == .Uint_Lit {
			lv.base += i32(int(ix.value) * size)
		} else {
			idx := lower_expr(l, e.b)
			lo := const_val(l, .Int, 0)
			hi := const_val(l, .Int, f64(limit - 1))
			clamped := alloc_slots(l, 1)
			emit(l, .I_Fn3, n = 1, k = int(Builtin.Clamp), dst = clamped, a = idx.slot, b = lo.slot, c = hi.slot)
			off := clamped
			if size != 1 {
				off = alloc_slots(l, 1)
				emit(l, .I_Mul, n = 1, dst = off, a = clamped, b = const_val(l, .Int, f64(size)).slot)
			}
			if lv.dyn >= 0 {
				sum := alloc_slots(l, 1)
				emit(l, .I_Add, n = 1, dst = sum, a = lv.dyn, b = off)
				off = sum
			}
			lv.dyn = off
		}
		lv.t = rt
		return lv
	case .Field:
		lv := lower_lvalue(l, e.a)
		if e.swizzle_len > 0 {
			lv.swz = e.swizzle
			lv.swz_len = int(e.swizzle_len)
			return lv
		}
		off := 0
		for f, i in struct_fields(u, lv.t.strukt) {
			if i32(i) == e.field {
				break
			}
			off += cc(l, ty(u, f.type))
		}
		lv.base += i32(off)
		lv.t = expr_type(u, id)
		return lv
	}
	return {dyn = -1}
}

@(private)
load_lv :: proc(l: ^Lower, lv: LV) -> Val {
	if lv.swz_len > 0 {
		n := lv.swz_len
		rt := vec_type(lv.t.base, n) if n > 1 else scalar_of(lv.t)
		// Ascending consecutive components (.xy, .rgb, .yz) are the
		// variable's own slots, read in place like a whole variable.
		consecutive := lv.dyn < 0
		for i in 1 ..< n {
			consecutive &&= lv.swz[i] == lv.swz[0] + u8(i)
		}
		if consecutive {
			return {lv.base + i32(lv.swz[0]), rt}
		}
		s := alloc_slots(l, n)
		if lv.dyn < 0 {
			srcs: [4]i32
			for i in 0 ..< n {
				srcs[i] = lv.base + i32(lv.swz[i])
			}
			gather(l, s, srcs[:n])
			return {s, rt}
		}
		for i in 0 ..< n {
			emit(l, .Load_Dyn, n = 1, dst = s + i32(i), a = lv.base + i32(lv.swz[i]), b = lv.dyn)
		}
		return {s, rt}
	}
	if lv.dyn < 0 {
		return {lv.base, lv.t}
	}
	n := cc(l, lv.t)
	s := alloc_slots(l, n)
	emit(l, .Load_Dyn, n = n, dst = s, a = lv.base, b = lv.dyn)
	return {s, lv.t}
}

@(private)
store_lv :: proc(l: ^Lower, lv: LV, v: Val) {
	if lv.swz_len > 0 {
		src := widen(l, v, lv.swz_len)
		// A source read in place from the same variable (v.yz = v.xy) is
		// copied first: storing component by component would overwrite it.
		if lv.dyn < 0 && src < lv.base + i32(cc(l, lv.t)) && src + i32(lv.swz_len) > lv.base {
			tmp := alloc_slots(l, lv.swz_len)
			emit(l, .Mov, n = lv.swz_len, dst = tmp, a = src)
			src = tmp
		}
		for i in 0 ..< lv.swz_len {
			dst := lv.base + i32(lv.swz[i])
			if lv.dyn >= 0 {
				emit(l, .Store_Dyn, n = 1, dst = dst, a = src + i32(i), b = lv.dyn)
			} else {
				emit(l, .Store, n = 1, dst = dst, a = src + i32(i))
			}
		}
		return
	}
	n := cc(l, lv.t)
	src := widen(l, v, n)
	if lv.dyn >= 0 {
		emit(l, .Store_Dyn, n = n, dst = lv.base, a = src, b = lv.dyn)
	} else {
		emit(l, .Store, n = n, dst = lv.base, a = src)
	}
}

@(private)
addressable_vm :: proc(l: ^Lower, id: Expr_Id) -> bool {
	e := l.u.exprs[id]
	#partial switch e.kind {
	case .Ident:
		return l.u.syms[e.sym].storage != .Channel
	case .Index:
		return addressable_vm(l, e.a)
	case .Field:
		return addressable_vm(l, e.a)
	}
	return false
}

// ---- statements -------------------------------------------------------------------------

@(private)
lower_stmt :: proc(l: ^Lower, id: Stmt_Id) {
	u := l.u
	if id == 0 {
		return
	}
	st := u.stmts[id]
	switch st.kind {
	case .Empty, .Block:
		for b in list(u, st.body) {
			lower_stmt(l, Stmt_Id(b))
		}
	case .Decl:
		for d in list(u, st.decls) {
			v := u.vars[d]
			t := sym_type(u, v.sym)
			n := cc(l, t)
			if v.init == 0 {
				l.sym_slot[v.sym] = alloc_slots(l, n)
				emit(l, .Store_Zero, n = n, dst = l.sym_slot[v.sym])
				continue
			}
			// The checker resolves the initializer before declaring the
			// variable, so it never reads the variable's slot.
			mark := l.next_slot
			val := lower_expr(l, v.init)
			// A value made for this initializer alone is the variable.
			if owned(l, val, mark) {
				l.sym_slot[v.sym] = val.slot
				continue
			}
			l.sym_slot[v.sym] = alloc_slots(l, n)
			emit(l, .Store, n = n, dst = l.sym_slot[v.sym], a = val.slot)
		}
	case .Expr:
		lower_expr(l, st.expr)
	case .If:
		c := lower_expr(l, st.expr)
		save := alloc_mask(l)
		emit(l, .Mask_Copy, dst = save, a = MASK_EXEC)
		emit(l, .Mask_Cond, dst = MASK_EXEC, a = save, b = c.slot)
		skip_then := emit(l, .Jump_If_None)
		lower_stmt(l, st.then_)
		l.code[skip_then].a = here(l)
		if st.else_ != 0 {
			emit(l, .Mask_Cond, k = 1, dst = MASK_EXEC, a = save, b = c.slot)
			skip_else := emit(l, .Jump_If_None)
			lower_stmt(l, st.else_)
			l.code[skip_else].a = here(l)
		}
		b, cn, r := kills(l)
		emit(l, .Mask_Restore, dst = MASK_EXEC, a = save, b = b, c = cn, d = r)
	case .For, .While, .Do_While:
		lower_loop(l, st)
	case .Switch:
		lower_switch(l, st)
	case .Case, .Default:
		// Handled by lower_switch.
	case .Break:
		emit(l, .Mask_Kill, dst = l.brk)
	case .Continue:
		emit(l, .Mask_Kill, dst = l.cont)
	case .Discard:
		emit(l, .Mask_Kill, dst = MASK_DISCARD)
	case .Return:
		if st.expr != 0 {
			v := lower_expr(l, st.expr)
			if l.ret_slot >= 0 {
				store_val(l, l.ret_slot, v)
			} else if owned(l, v, l.ret_mark) {
				// An unframed function's only return, at its end: a value
				// made during the call is the call's value.
				l.ret_val = v.slot
			} else {
				l.ret_val = alloc_slots(l, cc(l, v.t))
				store_val(l, l.ret_val, v)
			}
		}
		// Without a return mask the return is the function's last
		// statement: nothing follows it.
		if l.ret >= 0 {
			emit(l, .Mask_Kill, dst = l.ret)
		}
	}
}

// Whether a `return` can leave a function before its end (anywhere but as
// its last top-level statement). Calls to one that can't need no return
// mask.
@(private)
returns_early :: proc(u: ^Unit, body: Stmt_Id) -> bool {
	stmts := list(u, u.stmts[body].body)
	for s, i in stmts {
		if i == len(stmts) - 1 && u.stmts[s].kind == .Return {
			break
		}
		if has_return(u, Stmt_Id(s)) {
			return true
		}
	}
	return false
}

@(private)
has_return :: proc(u: ^Unit, id: Stmt_Id) -> bool {
	if id == 0 {
		return false
	}
	st := u.stmts[id]
	#partial switch st.kind {
	case .Return:
		return true
	case .Block, .Switch:
		for s in list(u, st.body) {
			if has_return(u, Stmt_Id(s)) {
				return true
			}
		}
	case .If:
		return has_return(u, st.then_) || has_return(u, st.else_)
	case .For, .While, .Do_While:
		return has_return(u, st.then_)
	}
	return false
}

@(private)
lower_loop :: proc(l: ^Lower, st: Stmt) {
	lower_stmt(l, st.init)
	entry := alloc_mask(l)
	live := alloc_mask(l)
	brk := alloc_mask(l)
	cont := alloc_mask(l)
	emit(l, .Mask_Copy, dst = entry, a = MASK_EXEC)
	emit(l, .Mask_Copy, dst = live, a = MASK_EXEC)
	emit(l, .Mask_Clear, dst = brk)
	outer_brk, outer_cont := l.brk, l.cont
	l.brk, l.cont = brk, cont

	top := here(l)
	emit(l, .Mask_Clear, dst = cont)
	emit(l, .Mask_Copy, dst = MASK_EXEC, a = live)
	if st.kind != .Do_While && st.expr != 0 {
		c := lower_expr(l, st.expr)
		emit(l, .Mask_Cond, dst = live, a = live, b = c.slot)
		emit(l, .Mask_Copy, dst = MASK_EXEC, a = live)
	}
	exit := emit(l, .Jump_If_None)
	lower_stmt(l, st.then_)
	// Continue point: lanes that broke or returned stop iterating.
	emit(l, .Mask_Restore, dst = live, a = live, b = brk, c = l.ret, d = -1)
	emit(l, .Mask_Copy, dst = MASK_EXEC, a = live)
	if st.kind == .Do_While {
		c := lower_expr(l, st.expr)
		emit(l, .Mask_Cond, dst = live, a = live, b = c.slot)
	} else if st.step != 0 {
		lower_expr(l, st.step)
	}
	emit(l, .Jump, a = top)
	l.code[exit].a = here(l)

	l.brk, l.cont = outer_brk, outer_cont
	b, cn, r := kills(l)
	emit(l, .Mask_Restore, dst = MASK_EXEC, a = entry, b = b, c = cn, d = r)
}

// A switch under masks: nothing runs before the first label; at each label
// the lanes whose selector matches it join the running ones (the default
// takes the lanes no case matches), so earlier lanes fall through. break
// kills lanes into the switch's own mask, and they resume after it.
@(private)
lower_switch :: proc(l: ^Lower, st: Stmt) {
	u := l.u
	sel := lower_expr(l, st.expr)
	body := list(u, st.body)
	// One bool slot per label; `any` holds whether some case matched.
	conds := make([]i32, len(body), context.temp_allocator)
	any := const_val(l, .Bool, 0).slot
	for b, i in body {
		if u.stmts[b].kind != .Case {
			continue
		}
		conds[i] = alloc_slots(l, 1)
		emit(l, .I_Eq, dst = conds[i], a = sel.slot, b = const_slot(l, u32(u.stmts[b].value)))
		next := alloc_slots(l, 1)
		emit(l, .B_Or, dst = next, a = any, b = conds[i])
		any = next
	}
	entry := alloc_mask(l)
	joining := alloc_mask(l)
	brk := alloc_mask(l)
	emit(l, .Mask_Copy, dst = entry, a = MASK_EXEC)
	emit(l, .Mask_Clear, dst = brk)
	emit(l, .Mask_Clear, dst = MASK_EXEC)
	outer_brk := l.brk
	l.brk = brk
	for b, i in body {
		#partial switch u.stmts[b].kind {
		case .Case:
			emit(l, .Mask_Cond, dst = joining, a = entry, b = conds[i])
			emit(l, .Mask_Or, dst = MASK_EXEC, a = MASK_EXEC, b = joining)
		case .Default:
			emit(l, .Mask_Cond, k = 1, dst = joining, a = entry, b = any)
			emit(l, .Mask_Or, dst = MASK_EXEC, a = MASK_EXEC, b = joining)
		case:
			lower_stmt(l, Stmt_Id(b))
		}
	}
	l.brk = outer_brk
	b, cn, r := kills(l)
	emit(l, .Mask_Restore, dst = MASK_EXEC, a = entry, b = b, c = cn, d = r)
}

// Inlines a call: `vals` for `in` arguments, `lvs` for out/inout ones.
@(private)
inline_call :: proc(l: ^Lower, f: Func_Id, vals: []Val, lvs: []LV, sampler_args: []i32 = nil) -> Val {
	u := l.u
	mark := l.next_slot
	ps := fparams(u, f)
	for p, i in ps {
		if p.sym == 0 {
			continue
		}
		t := ty(u, p.type)
		if t.base == .Sampler {
			l.sym_chan[p.sym] = sampler_args[i] if sampler_args != nil else -1
			continue
		}
		n := cc(l, t)
		if p.qual == .In {
			arg := widen(l, vals[i], n)
			// A parameter the function never writes reads its argument in
			// place, unless that is an output or a global, which the
			// function could write meanwhile.
			if !l.written[p.sym] && (arg < l.globals[0] || arg >= l.globals[1]) {
				l.sym_slot[p.sym] = arg
				continue
			}
			l.sym_slot[p.sym] = alloc_slots(l, n)
			emit(l, .Store, n = n, dst = l.sym_slot[p.sym], a = arg)
			continue
		}
		slot := alloc_slots(l, n)
		l.sym_slot[p.sym] = slot
		#partial switch p.qual {
		case .Inout:
			emit(l, .Store, n = n, dst = slot, a = load_lv(l, lvs[i]).slot)
		case .Out:
			emit(l, .Store_Zero, n = n, dst = slot)
		}
	}
	ret_t := ty(u, u.funcs[f].ret)
	result := Val{-1, ret_t}
	outer := [6]i32{l.brk, l.cont, l.ret, l.ret_slot, l.ret_val, l.ret_mark}
	body := u.func_decls[u.funcs[f].decl].body
	// A return mask only when a return can leave early.
	framed := returns_early(u, body)
	l.brk, l.cont = -1, -1
	l.ret = alloc_mask(l) if framed else -1
	l.ret_slot, l.ret_val, l.ret_mark = -1, -1, mark
	if ret_t != T_VOID && framed {
		result.slot = alloc_slots(l, cc(l, ret_t))
		l.ret_slot = result.slot
	}
	save: i32
	if framed {
		save = alloc_mask(l)
		emit(l, .Mask_Copy, dst = save, a = MASK_EXEC)
		emit(l, .Mask_Clear, dst = l.ret)
	}
	lower_stmt(l, body)
	if ret_t != T_VOID && !framed {
		// Its only return's value (none: whatever its slot holds).
		result.slot = l.ret_val if l.ret_val >= 0 else alloc_slots(l, cc(l, ret_t))
	}
	l.brk, l.cont, l.ret, l.ret_slot, l.ret_val, l.ret_mark = outer[0], outer[1], outer[2], outer[3], outer[4], outer[5]
	if framed {
		b, cn, r := kills(l)
		emit(l, .Mask_Restore, dst = MASK_EXEC, a = save, b = b, c = cn, d = r)
	}
	for p, i in ps {
		t := ty(u, p.type)
		if p.qual != .In && t.base != .Sampler && p.sym != 0 {
			store_lv(l, lvs[i], {l.sym_slot[p.sym], t})
		}
	}
	return result
}

// ---- expressions ------------------------------------------------------------------------

@(private)
lower_expr :: proc(l: ^Lower, id: Expr_Id) -> Val {
	u := l.u
	e := u.exprs[id]
	t := ty(u, e.type)
	switch e.kind {
	case .None:
		return {}
	case .Int_Lit:
		return {const_slot(l, u32(e.value)), T_INT}
	case .Uint_Lit:
		return {const_slot(l, u32(e.value)), T_UINT}
	case .Float_Lit:
		return {const_slot(l, transmute(u32)f32(lit_f64(e))), T_FLOAT}
	case .Bool_Lit:
		return {const_slot(l, u32(e.value)), T_BOOL}
	case .Ident:
		return {l.sym_slot[e.sym], t}
	case .Convert:
		v := lower_expr(l, e.a)
		return convert_val(l, v, t)
	case .Comma:
		lower_expr(l, e.a)
		return lower_expr(l, e.b)
	case .Length:
		at := expr_type(u, e.a)
		n := at.array if is_array(at) else (i32(at.cols) if is_matrix(at) else i32(at.rows))
		return const_val(l, .Int, f64(n))
	case .Unary:
		v := lower_expr(l, e.a)
		n := cc(l, t)
		s := alloc_slots(l, n)
		#partial switch e.op {
		case .Plus:
			return v
		case .Minus:
			emit(l, .F_Neg if t.base == .Float else .I_Neg, n = n, dst = s, a = v.slot)
		case .Bang:
			emit(l, .B_Not, n = n, dst = s, a = v.slot)
		case .Tilde:
			emit(l, .Bit_Not, n = n, dst = s, a = v.slot)
		}
		return {s, t}
	case .Binary:
		#partial switch e.op {
		case .And_And, .Or_Or:
			return lower_short_circuit(l, e)
		}
		a := lower_expr(l, e.a)
		b := lower_expr(l, e.b)
		return binop_vm(l, e.op, a, b, t)
	case .Assign:
		lv := lower_lvalue(l, e.a)
		v := lower_expr(l, e.b)
		if e.op != .Eq {
			old := load_lv(l, lv)
			v = binop_vm(l, compound_op(e.op), old, v, expr_type(u, e.a))
		}
		store_lv(l, lv, v)
		return v
	case .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
		lv := lower_lvalue(l, e.a)
		cur := load_lv(l, lv)
		n := cc(l, t)
		old := alloc_slots(l, n)
		emit(l, .Mov, n = n, dst = old, a = cur.slot)
		inc := e.kind == .Pre_Inc || e.kind == .Post_Inc
		nv := binop_vm(l, .Plus if inc else .Minus, {old, t}, const_val(l, t.base, 1), t)
		store_lv(l, lv, nv)
		return nv if e.kind == .Pre_Inc || e.kind == .Pre_Dec else Val{old, t}
	case .Ternary:
		c := lower_expr(l, e.a)
		n := cc(l, t)
		res := alloc_slots(l, n)
		save := alloc_mask(l)
		emit(l, .Mask_Copy, dst = save, a = MASK_EXEC)
		emit(l, .Mask_Cond, dst = MASK_EXEC, a = save, b = c.slot)
		skip_a := emit(l, .Jump_If_None)
		va := lower_expr(l, e.b)
		emit(l, .Store, n = n, dst = res, a = widen(l, va, n))
		l.code[skip_a].a = here(l)
		emit(l, .Mask_Cond, k = 1, dst = MASK_EXEC, a = save, b = c.slot)
		skip_b := emit(l, .Jump_If_None)
		vb := lower_expr(l, e.c)
		emit(l, .Store, n = n, dst = res, a = widen(l, vb, n))
		l.code[skip_b].a = here(l)
		b, cn, r := kills(l)
		emit(l, .Mask_Restore, dst = MASK_EXEC, a = save, b = b, c = cn, d = r)
		return {res, t}
	case .Index:
		if addressable_vm(l, e.a) {
			return load_lv(l, lower_lvalue(l, id))
		}
		base := lower_expr(l, e.a)
		bt := base.t
		size := cc(l, t)
		ix := u.exprs[e.b]
		if ix.kind == .Int_Lit || ix.kind == .Uint_Lit {
			return {base.slot + i32(int(ix.value) * size), t}
		}
		lv := LV{base = base.slot, dyn = -1, t = bt}
		limit := int(bt.array) if is_array(bt) else (int(bt.cols) if is_matrix(bt) else int(bt.rows))
		idx := lower_expr(l, e.b)
		clamped := alloc_slots(l, 1)
		emit(l, .I_Fn3, n = 1, k = int(Builtin.Clamp), dst = clamped, a = idx.slot, b = const_val(l, .Int, 0).slot, c = const_val(l, .Int, f64(limit - 1)).slot)
		off := clamped
		if size != 1 {
			off = alloc_slots(l, 1)
			emit(l, .I_Mul, n = 1, dst = off, a = clamped, b = const_val(l, .Int, f64(size)).slot)
		}
		lv.dyn = off
		lv.t = t
		return load_lv(l, lv)
	case .Field:
		if addressable_vm(l, e.a) {
			return load_lv(l, lower_lvalue(l, id))
		}
		base := lower_expr(l, e.a)
		if e.swizzle_len > 0 {
			lv := LV{base = base.slot, dyn = -1, t = base.t, swz = e.swizzle, swz_len = int(e.swizzle_len)}
			if is_scalar(base.t) {
				// Swizzling a scalar repeats it.
				return {widen(l, base, int(e.swizzle_len)), t}
			}
			return load_lv(l, lv)
		}
		off := 0
		for f, i in struct_fields(u, base.t.strukt) {
			if i32(i) == e.field {
				break
			}
			off += cc(l, ty(u, f.type))
		}
		return {base.slot + i32(off), t}
	case .Call:
		switch e.call {
		case .User:
			return lower_user_call(l, id)
		case .Builtin:
			return lower_builtin(l, id)
		case .Constructor:
			return lower_constructor(l, id)
		case .None:
		}
	}
	return {}
}

@(private)
lower_short_circuit :: proc(l: ^Lower, e: Expr) -> Val {
	a := lower_expr(l, e.a)
	res := alloc_slots(l, 1)
	emit(l, .Mov, dst = res, a = a.slot)
	save := alloc_mask(l)
	emit(l, .Mask_Copy, dst = save, a = MASK_EXEC)
	// && evaluates b where a holds, || where it does not.
	emit(l, .Mask_Cond, k = 0 if e.op == .And_And else 1, dst = MASK_EXEC, a = save, b = a.slot)
	skip := emit(l, .Jump_If_None)
	b := lower_expr(l, e.b)
	emit(l, .Store, dst = res, a = b.slot)
	l.code[skip].a = here(l)
	bk, cn, r := kills(l)
	emit(l, .Mask_Restore, dst = MASK_EXEC, a = save, b = bk, c = cn, d = r)
	return {res, T_BOOL}
}

@(private)
convert_val :: proc(l: ^Lower, v: Val, to: Type) -> Val {
	if v.t.base == to.base {
		return {v.slot, to}
	}
	n := cc(l, to)
	op := VM_CONVERT[to.base][v.t.base]
	slots: [16]i32
	for i in 0 ..< n {
		slots[i] = v.slot + i32(i)
	}
	if bits, ok := consts_of(l, slots[:n]); ok {
		for &b in bits {
			b = convert_const(b, op)
		}
		return {const_vec(l, bits), to}
	}
	s := alloc_slots(l, n)
	emit(l, op, n = n, dst = s, a = v.slot)
	return {s, to}
}

// A constant's conversion, made by the instruction's own lane function.
@(private)
convert_const :: proc(bits: u32, op: VM_Op) -> u32 {
	#partial switch op {
	case .I_To_F:
		return un_lane(bits, .I_To_F)
	case .U_To_F:
		return un_lane(bits, .U_To_F)
	case .B_To_F:
		return un_lane(bits, .B_To_F)
	case .F_To_I:
		return un_lane(bits, .F_To_I)
	case .F_To_U:
		return un_lane(bits, .F_To_U)
	case .F_To_B:
		return un_lane(bits, .F_To_B)
	case .I_To_B:
		return un_lane(bits, .I_To_B)
	}
	// Mov: ints, uints and bools share their bits.
	return bits
}

// Base conversions, [to][from]. Bools are 0 or 1 and ints and uints share
// their bits, so those move unchanged.
@(rodata)
VM_CONVERT := #partial [Base][Base]VM_Op {
	.Float = #partial {.Int = .I_To_F, .Uint = .U_To_F, .Bool = .B_To_F},
	.Int   = #partial {.Float = .F_To_I, .Uint = .Mov, .Bool = .Mov},
	.Uint  = #partial {.Float = .F_To_U, .Int = .Mov, .Bool = .Mov},
	.Bool  = #partial {.Float = .F_To_B, .Int = .I_To_B, .Uint = .I_To_B},
}

@(private)
binop_vm :: proc(l: ^Lower, op: Token_Kind, a, b: Val, result: Type) -> Val {
	ta, tb := a.t, b.t
	if is_matrix(ta) || is_matrix(tb) {
		if op == .Star && !is_scalar(ta) && !is_scalar(tb) {
			s := alloc_slots(l, cc(l, result))
			switch {
			case is_matrix(ta) && is_matrix(tb):
				emit(l, .Mat_Mul, n = int(ta.rows), k = int(ta.cols), dst = s, a = a.slot, b = b.slot, d = i32(tb.cols))
			case is_matrix(ta):
				emit(l, .Mat_Mul, n = int(ta.rows), k = int(ta.cols), dst = s, a = a.slot, b = b.slot, d = 1)
			case:
				emit(l, .Vec_Mat, n = int(tb.rows), k = int(tb.cols), dst = s, a = a.slot, b = b.slot)
			}
			return {s, result}
		}
	}
	vop := VM_BINOPS[op][ta.base]
	if op == .Eq_Eq || op == .Not_Eq {
		// Per component, then all (==) or any (!=).
		cn := cc(l, ta)
		cmp := alloc_slots(l, cn)
		emit(l, vop, n = cn, dst = cmp, a = a.slot, b = b.slot)
		if cn == 1 {
			return {cmp, T_BOOL}
		}
		s := alloc_slots(l, 1)
		emit(l, .All if op == .Eq_Eq else .Any, n = cn, dst = s, a = cmp)
		return {s, T_BOOL}
	}
	n := cc(l, result)
	s := alloc_slots(l, n)
	if vop == .F_Add || vop == .F_Sub || vop == .F_Mul || vop == .F_Div {
		k := (1 if cc(l, ta) == 1 && n > 1 else 0) | (2 if cc(l, tb) == 1 && n > 1 else 0)
		emit(l, vop, n = n, k = k, dst = s, a = a.slot, b = b.slot)
		return {s, result}
	}
	emit(l, vop, n = n, dst = s, a = widen(l, a, n), b = widen(l, b, n))
	return {s, result}
}

// The instruction for a binary operator, by its operands' base (after the
// checker's conversions both have the same one). Matrix products and the
// reduction of vector == and != are lowered by hand above; && and || are
// short-circuited before they get here.
@(rodata)
VM_BINOPS := #partial [Token_Kind][Base]VM_Op {
	.Plus    = #partial {.Float = .F_Add, .Int = .I_Add, .Uint = .I_Add},
	.Minus   = #partial {.Float = .F_Sub, .Int = .I_Sub, .Uint = .I_Sub},
	.Star    = #partial {.Float = .F_Mul, .Int = .I_Mul, .Uint = .I_Mul},
	.Slash   = #partial {.Float = .F_Div, .Int = .I_Div, .Uint = .U_Div},
	.Percent = #partial {.Int = .I_Rem, .Uint = .U_Rem},
	.Amp     = #partial {.Int = .Bit_And, .Uint = .Bit_And},
	.Pipe    = #partial {.Int = .Bit_Or, .Uint = .Bit_Or},
	.Caret   = #partial {.Int = .Bit_Xor, .Uint = .Bit_Xor},
	.Shl     = #partial {.Int = .Shl, .Uint = .Shl},
	.Shr     = #partial {.Int = .Shr_I, .Uint = .Shr_U},
	.Lt      = #partial {.Float = .F_Lt, .Int = .I_Lt, .Uint = .U_Lt},
	.Gt      = #partial {.Float = .F_Gt, .Int = .I_Gt, .Uint = .U_Gt},
	.Lt_Eq   = #partial {.Float = .F_Le, .Int = .I_Le, .Uint = .U_Le},
	.Gt_Eq   = #partial {.Float = .F_Ge, .Int = .I_Ge, .Uint = .U_Ge},
	.Eq_Eq   = #partial {.Float = .F_Eq, .Int = .I_Eq, .Uint = .I_Eq, .Bool = .I_Eq},
	.Not_Eq  = #partial {.Float = .F_Ne, .Int = .I_Ne, .Uint = .I_Ne, .Bool = .I_Ne},
	.Xor_Xor = #partial {.Bool = .I_Ne},
}

// ---- calls ------------------------------------------------------------------------------

@(private)
lower_user_call :: proc(l: ^Lower, id: Expr_Id) -> Val {
	u := l.u
	f := u.exprs[id].func
	args := args_of(u, id)
	ps := fparams(u, f)
	vals := make([]Val, len(args), context.temp_allocator)
	lvs := make([]LV, len(args), context.temp_allocator)
	chans := make([]i32, len(args), context.temp_allocator)
	for p, i in ps {
		t := ty(u, p.type)
		switch {
		case t.base == .Sampler:
			chans[i] = l.sym_chan[u.exprs[args[i]].sym]
		case p.qual == .In:
			v := lower_expr(l, args[i])
			// Copy: the callee may write its parameter.
			vals[i] = v
		case:
			lvs[i] = lower_lvalue(l, args[i])
		}
	}
	return inline_call(l, f, vals, lvs, chans)
}

// Components of `v` converted to `base`, as slots.
@(private)
flatten :: proc(l: ^Lower, v: Val, base: Base, out: ^[dynamic]i32) {
	n := cc(l, v.t)
	conv := v
	if v.t.base != base {
		conv = convert_val(l, v, with_base(v.t, base))
	}
	for i in 0 ..< n {
		append(out, conv.slot + i32(i))
	}
}

@(private)
lower_constructor :: proc(l: ^Lower, id: Expr_Id) -> Val {
	u := l.u
	t := expr_type(u, id)
	args := args_of(u, id)
	vals := make([]Val, len(args), context.temp_allocator)
	// Each argument's instruction when it was made for this constructor
	// alone (producer), and where its components start in the result.
	made := make([]int, len(args), context.temp_allocator)
	start := make([]int, len(args), context.temp_allocator)
	for a, i in args {
		mark := l.next_slot
		vals[i] = lower_expr(l, a)
		made[i] = producer(l, vals[i], mark)
	}
	n := cc(l, t)
	// Each component's source slot.
	comps := make([dynamic]i32, 0, n, context.temp_allocator)
	whole := t.array > 0 || t.base == .Struct
	splat := false
	// Each argument's components are consecutive in the result (not a
	// matrix's diagonal or a matrix resized).
	in_order := true
	switch {
	case whole:
		for v, i in vals {
			start[i] = len(comps)
			for k in 0 ..< cc(l, v.t) {
				append(&comps, v.slot + i32(k))
			}
		}
	case len(vals) == 1 && is_matrix(t) && is_matrix(vals[0].t):
		in_order = false
		v := vals[0]
		one := const_val(l, .Float, 1).slot
		zero := const_val(l, .Float, 0).slot
		for c in 0 ..< int(t.cols) {
			for r in 0 ..< int(t.rows) {
				if c < int(v.t.cols) && r < int(v.t.rows) {
					append(&comps, v.slot + i32(c * int(v.t.rows) + r))
				} else {
					append(&comps, one if c == r else zero)
				}
			}
		}
	case len(vals) == 1 && is_scalar(vals[0].t):
		sv := convert_val(l, vals[0], scalar_of(t))
		if !is_matrix(t) {
			splat = true
			for _ in 0 ..< n {
				append(&comps, sv.slot)
			}
			break
		}
		in_order = false
		zero := const_val(l, .Float, 0).slot
		for c in 0 ..< int(t.cols) {
			for r in 0 ..< int(t.rows) {
				append(&comps, sv.slot if c == r else zero)
			}
		}
	case:
		for v, i in vals {
			start[i] = len(comps)
			mark := l.next_slot
			flatten(l, v, t.base, &comps)
			if v.t.base != t.base {
				// Converted: the conversion made it.
				made[i] = producer(l, {comps[start[i]], with_base(v.t, t.base)}, mark)
			}
		}
	}
	// Of constants: a constant, written once per run.
	if bits, ok := consts_of(l, comps[:n]); ok {
		return {const_vec(l, bits), t}
	}
	s := alloc_slots(l, n)
	if splat {
		emit(l, .Splat, n = n, dst = s, a = comps[0])
		return {s, t}
	}
	// An argument made for this constructor is made in place; the others'
	// components are moved there.
	placed := make([]bool, n, context.temp_allocator)
	if in_order {
		for v, i in vals {
			k := cc(l, v.t)
			if made[i] < 0 || start[i] + k > n {
				continue
			}
			l.code[made[i]].dst = s + i32(start[i])
			for j in start[i] ..< start[i] + k {
				placed[j] = true
			}
		}
	}
	if whole {
		for v, i in vals {
			if !placed[start[i]] {
				emit(l, .Mov, n = cc(l, v.t), dst = s + i32(start[i]), a = v.slot)
			}
		}
		return {s, t}
	}
	for i := 0; i < n; {
		if placed[i] {
			i += 1
			continue
		}
		end := i
		for end < n && !placed[end] {
			end += 1
		}
		gather(l, s + i32(i), comps[i:end])
		i = end
	}
	return {s, t}
}

// Copies these slots into consecutive slots from dst, up to four per
// instruction.
@(private)
gather :: proc(l: ^Lower, dst: i32, srcs: []i32) {
	for first := 0; first < len(srcs); first += 4 {
		k := min(len(srcs) - first, 4)
		src: [4]i32
		copy(src[:], srcs[first:][:k])
		emit(l, .Gather, n = k, dst = dst + i32(first), a = src[0], b = src[1], c = src[2], d = src[3])
	}
}

// Whether `v` was made after slot `mark` and is no constant: a temporary
// nothing else reads (or a variable of a call that has ended), which its
// consumer can keep instead of copying.
@(private)
owned :: proc(l: ^Lower, v: Val, mark: i32) -> bool {
	_, is_const := slot_const(l, v.slot)
	return v.slot >= mark && !is_const
}

// The instruction that has just made `v`, owned: the last one, writing
// every lane of v's slots, which are the last allocated (so v is not part
// of a larger value). It can write where the value is going instead (-1:
// none).
@(private)
producer :: proc(l: ^Lower, v: Val, mark: i32) -> int {
	i := len(l.code) - 1
	if i < 0 || !owned(l, v, mark) || l.code[i].dst != v.slot || v.slot + i32(cc(l, v.t)) != l.next_slot || l.code[i].op in MASKED_OPS {
		return -1
	}
	return i
}

// Instructions whose dst is not a value they make whole: masked stores,
// masks and jumps.
@(private)
MASKED_OPS :: bit_set[VM_Op;u128]{.Nop, .Store, .Store_Zero, .Store_Dyn, .Mask_Copy, .Mask_Cond, .Mask_Restore, .Mask_Or, .Mask_Clear, .Mask_Kill, .Jump, .Jump_If_None}

@(private)
lower_builtin :: proc(l: ^Lower, id: Expr_Id) -> Val {
	u := l.u
	e := u.exprs[id]
	b := e.builtin
	t := ty(u, e.type)
	argx := args_of(u, id)
	vals := make([]Val, len(argx), context.temp_allocator)
	for a, i in argx {
		if expr_type(u, a).base == .Sampler {
			vals[i] = {l.sym_chan[u.exprs[a].sym], T_SAMPLER}
		} else {
			vals[i] = lower_expr(l, a)
		}
	}
	at := vals[0].t
	n := cc(l, t)
	s := alloc_slots(l, n)
	if op := VM_BUILTINS[b][at.base]; op != .Nop {
		// Component-wise over the result, or a reduction over the first
		// argument (dot, length, any, ...) when the result is a scalar.
		count := cc(l, at) if is_scalar(t) else n
		slots: [3]i32
		for v, i in vals {
			slots[i] = widen(l, v, count) if i in BUILTINS[b].broadcast else v.slot
		}
		emit(l, op, n = count, k = int(b) if op in FN_OPS else 0, dst = s, a = slots[0], b = slots[1], c = slots[2])
		return {s, t}
	}
	#partial switch b {
	case .Mix_Bool:
		emit(l, .Select, n = n, dst = s, a = vals[1].slot, b = vals[0].slot, c = vals[2].slot)
	case .Outer_Product:
		emit(l, .Outer, n = int(vals[0].t.rows), k = int(vals[1].t.rows), dst = s, a = vals[0].slot, b = vals[1].slot)
	case .Transpose:
		emit(l, .Transpose, n = int(at.rows), k = int(at.cols), dst = s, a = vals[0].slot)
	case .Determinant:
		emit(l, .Determinant, k = int(at.cols), dst = s, a = vals[0].slot)
	case .Inverse:
		emit(l, .Inverse, k = int(at.cols), dst = s, a = vals[0].slot)
	case .Float_Bits_To_Int, .Float_Bits_To_Uint, .Int_Bits_To_Float, .Uint_Bits_To_Float:
		return {vals[0].slot, t}
	case .Texture_Lod:
		emit(l, .Texture, k = int(vals[0].slot), dst = s, a = vals[1].slot, b = vals[2].slot)
	case .Texture:
		emit(l, .Texture_Bias, k = int(vals[0].slot), dst = s, a = vals[1].slot, b = vals[2].slot)
	case .Texture_Size:
		emit(l, .Texture_Size, k = int(vals[0].slot), dst = s, a = vals[1].slot)
	}
	return {s, t}
}

// The instruction each built-in lowers to, by its first argument's base;
// the Fn ops take the Builtin in k. The rest are lowered by hand above.
@(rodata)
VM_BUILTINS := #partial [Builtin][Base]VM_Op {
	.Radians            = #partial {.Float = .F_Fn1},
	.Degrees            = #partial {.Float = .F_Fn1},
	.Sin                = #partial {.Float = .F_Fn1},
	.Cos                = #partial {.Float = .F_Fn1},
	.Tan                = #partial {.Float = .F_Fn1},
	.Asin               = #partial {.Float = .F_Fn1},
	.Acos               = #partial {.Float = .F_Fn1},
	.Atan               = #partial {.Float = .F_Fn1},
	.Sinh               = #partial {.Float = .F_Fn1},
	.Cosh               = #partial {.Float = .F_Fn1},
	.Tanh               = #partial {.Float = .F_Fn1},
	.Asinh              = #partial {.Float = .F_Fn1},
	.Acosh              = #partial {.Float = .F_Fn1},
	.Atanh              = #partial {.Float = .F_Fn1},
	.Exp                = #partial {.Float = .F_Fn1},
	.Log                = #partial {.Float = .F_Fn1},
	.Exp2               = #partial {.Float = .F_Fn1},
	.Log2               = #partial {.Float = .F_Fn1},
	.Sqrt               = #partial {.Float = .F_Fn1},
	.Inversesqrt        = #partial {.Float = .F_Fn1},
	.Floor              = #partial {.Float = .F_Fn1},
	.Ceil               = #partial {.Float = .F_Fn1},
	.Fract              = #partial {.Float = .F_Fn1},
	.Trunc              = #partial {.Float = .F_Fn1},
	.Round              = #partial {.Float = .F_Fn1},
	.Round_Even         = #partial {.Float = .F_Fn1},
	.Isnan              = #partial {.Float = .F_Fn1},
	.Isinf              = #partial {.Float = .F_Fn1},
	.Dfdx               = #partial {.Float = .Deriv_X},
	.Dfdy               = #partial {.Float = .Deriv_Y},
	.Fwidth             = #partial {.Float = .Fwidth},
	.Abs                = #partial {.Float = .F_Fn1, .Int = .I_Fn1},
	.Sign               = #partial {.Float = .F_Fn1, .Int = .I_Fn1},
	.Atan2              = #partial {.Float = .F_Fn2},
	.Pow                = #partial {.Float = .F_Fn2},
	.Mod                = #partial {.Float = .F_Fn2},
	.Step               = #partial {.Float = .F_Fn2},
	.Min                = #partial {.Float = .F_Fn2, .Int = .I_Fn2, .Uint = .U_Fn2},
	.Max                = #partial {.Float = .F_Fn2, .Int = .I_Fn2, .Uint = .U_Fn2},
	.Clamp              = #partial {.Float = .F_Fn3, .Int = .I_Fn3, .Uint = .U_Fn3},
	.Mix                = #partial {.Float = .F_Fn3},
	.Smoothstep         = #partial {.Float = .F_Fn3},
	.Length             = #partial {.Float = .Length},
	.Distance           = #partial {.Float = .Distance},
	.Dot                = #partial {.Float = .Dot},
	.Normalize          = #partial {.Float = .Normalize},
	.Cross              = #partial {.Float = .Cross},
	.Faceforward        = #partial {.Float = .Faceforward},
	.Reflect            = #partial {.Float = .Reflect},
	.Refract            = #partial {.Float = .Refract},
	.Matrix_Comp_Mult   = #partial {.Float = .F_Mul},
	.Less_Than          = #partial {.Float = .F_Lt, .Int = .I_Lt, .Uint = .U_Lt},
	.Less_Than_Equal    = #partial {.Float = .F_Le, .Int = .I_Le, .Uint = .U_Le},
	.Greater_Than       = #partial {.Float = .F_Gt, .Int = .I_Gt, .Uint = .U_Gt},
	.Greater_Than_Equal = #partial {.Float = .F_Ge, .Int = .I_Ge, .Uint = .U_Ge},
	.Equal              = #partial {.Float = .F_Eq, .Int = .I_Eq, .Uint = .I_Eq, .Bool = .I_Eq},
	.Not_Equal          = #partial {.Float = .F_Ne, .Int = .I_Ne, .Uint = .I_Ne, .Bool = .I_Ne},
	.Any                = #partial {.Bool = .Any},
	.All                = #partial {.Bool = .All},
	.Not                = #partial {.Bool = .B_Not},
}
