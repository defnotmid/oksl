package oksl

// CPU backend, part 1: lowering. Functions are inlined (recursion is
// rejected) into one flat instruction array over SoA registers: a slot is
// one scalar component for LANES pixels. Divergent control flow runs every
// path with execution masks; only stores to variables honour the mask, so
// temporaries are computed for all lanes (helper lanes keep derivatives
// right).

LANES :: 16
// A set of lanes (execution, discard, break, ... masks).
Lane_Mask :: bit_set[0 ..< LANES;u16]
ALL_LANES :: ~Lane_Mask{}

// Mask registers 0 and 1: the execution mask and the discarded lanes.
MASK_EXEC :: 0
MASK_DISCARD :: 1

VM_Op :: enum u8 {
	Nop,
	// Unmasked moves into temporaries.
	Mov,
	Splat,
	// Masked stores into variables: dst = a for the lanes in exec.
	Store,
	Store_Zero,
	// Per-lane offsets in slot b (components): dst = a[off], a[off] = src.
	Load_Dyn,
	Store_Dyn,
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
	ret_slot:  i32,
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
	}
	for &ch in l.sym_chan {
		ch = -1
	}
	p: Program
	inputs := make([dynamic]Program_Var)
	outputs := make([dynamic]Program_Var)
	for row in c.interface {
		#partial switch row.kind {
		case .Channel:
			l.sym_chan[row.sym] = row.binding
		case .Stage_In, .Frag_Coord, .Front_Facing, .Stage_Out, .Position:
			t := sym_type(u, row.sym)
			v := Program_Var{row.kind, row.binding, alloc_slots(&l, cc(&l, t)), i32(cc(&l, t)), t.base, row.offset}
			l.sym_slot[row.sym] = v.slot
			append(&outputs if row.kind == .Stage_Out || row.kind == .Position else &inputs, v)
		}
	}
	p.inputs = inputs[:]
	p.outputs = outputs[:]
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
	for v in p.outputs {
		emit(&l, .Store_Zero, n = int(v.n), dst = v.slot)
	}
	for g in c.globals {
		t := sym_type(u, g)
		l.sym_slot[g] = alloc_slots(&l, component_count(u, t))
		emit(&l, .Store_Zero, n = component_count(u, t), dst = l.sym_slot[g])
	}
	inline_call(&l, c.entry, nil, nil)

	p.code = l.code[:]
	p.num_slots = int(l.next_slot)
	p.num_masks = int(l.next_mask)
	p.consts = l.const_list[:]
	p.fills = l.fills[:]
	return p
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
		for i in 0 ..< n {
			src := lv.base + i32(lv.swz[i])
			if lv.dyn >= 0 {
				emit(l, .Load_Dyn, n = 1, dst = s + i32(i), a = src, b = lv.dyn)
			} else {
				emit(l, .Mov, n = 1, dst = s + i32(i), a = src)
			}
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
			slot := alloc_slots(l, n)
			l.sym_slot[v.sym] = slot
			if v.init != 0 {
				val := lower_expr(l, v.init)
				emit(l, .Store, n = n, dst = slot, a = val.slot)
			} else {
				emit(l, .Store_Zero, n = n, dst = slot)
			}
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
			store_val(l, l.ret_slot, v)
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
		slot := alloc_slots(l, n)
		l.sym_slot[p.sym] = slot
		switch p.qual {
		case .In:
			emit(l, .Store, n = n, dst = slot, a = widen(l, vals[i], n))
		case .Inout:
			emit(l, .Store, n = n, dst = slot, a = load_lv(l, lvs[i]).slot)
		case .Out:
			emit(l, .Store_Zero, n = n, dst = slot)
		}
	}
	ret_t := ty(u, u.funcs[f].ret)
	result := Val{-1, ret_t}
	outer := [4]i32{l.brk, l.cont, l.ret, l.ret_slot}
	body := u.func_decls[u.funcs[f].decl].body
	// A return mask only when a return can leave early.
	framed := returns_early(u, body)
	l.brk, l.cont = -1, -1
	l.ret = alloc_mask(l) if framed else -1
	l.ret_slot = -1
	if ret_t != T_VOID {
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
	l.brk, l.cont, l.ret, l.ret_slot = outer[0], outer[1], outer[2], outer[3]
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
				s := alloc_slots(l, int(e.swizzle_len))
				emit(l, .Splat, n = int(e.swizzle_len), dst = s, a = base.slot)
				return {s, t}
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
	s := alloc_slots(l, n)
	emit(l, VM_CONVERT[to.base][v.t.base], n = n, dst = s, a = v.slot)
	return {s, to}
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
	sa := widen(l, a, n)
	sb := widen(l, b, n)
	s := alloc_slots(l, n)
	emit(l, vop, n = n, dst = s, a = sa, b = sb)
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
	for a, i in args {
		vals[i] = lower_expr(l, a)
	}
	n := cc(l, t)
	s := alloc_slots(l, n)
	if t.array > 0 || t.base == .Struct {
		off := i32(0)
		for v in vals {
			k := cc(l, v.t)
			emit(l, .Mov, n = k, dst = s + off, a = v.slot)
			off += i32(k)
		}
		return {s, t}
	}
	if len(vals) == 1 {
		v := vals[0]
		at := v.t
		if is_matrix(t) && is_matrix(at) {
			one := const_val(l, .Float, 1).slot
			zero := const_val(l, .Float, 0).slot
			for c in 0 ..< int(t.cols) {
				for r in 0 ..< int(t.rows) {
					dst := s + i32(c * int(t.rows) + r)
					if c < int(at.cols) && r < int(at.rows) {
						emit(l, .Mov, dst = dst, a = v.slot + i32(c * int(at.rows) + r))
					} else {
						emit(l, .Mov, dst = dst, a = one if c == r else zero)
					}
				}
			}
			return {s, t}
		}
		if is_scalar(at) {
			sv := convert_val(l, v, scalar_of(t))
			if is_matrix(t) {
				zero := const_val(l, .Float, 0).slot
				for c in 0 ..< int(t.cols) {
					for r in 0 ..< int(t.rows) {
						emit(l, .Mov, dst = s + i32(c * int(t.rows) + r), a = sv.slot if c == r else zero)
					}
				}
				return {s, t}
			}
			emit(l, .Splat, n = n, dst = s, a = sv.slot)
			return {s, t}
		}
	}
	comps := make([dynamic]i32, context.temp_allocator)
	for v in vals {
		flatten(l, v, t.base, &comps)
	}
	for i in 0 ..< n {
		emit(l, .Mov, dst = s + i32(i), a = comps[i])
	}
	return {s, t}
}

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
