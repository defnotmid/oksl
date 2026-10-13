package oksl

// SPIR-V emission of statements and expressions (see spirv.odin for the
// module, types and interface).

// ---- statements ------------------------------------------------------------------------

@(private)
emit_block :: proc(s: ^Spv, id: Stmt_Id) {
	for b in list(s.u, s.u.stmts[id].body) {
		if s.terminated {
			// Statements after return/break/continue/discard never run.
			return
		}
		emit_stmt(s, Stmt_Id(b))
	}
}

@(private)
emit_stmt :: proc(s: ^Spv, id: Stmt_Id) {
	u := s.u
	if id == 0 || s.terminated {
		return
	}
	st := u.stmts[id]
	switch st.kind {
	case .Empty:
	case .Block:
		emit_block(s, id)
	case .Decl:
		for d in list(u, st.decls) {
			v := u.vars[d]
			t := sym_type(u, v.sym)
			ptr := local_var(s, t, v.name)
			s.sym_ids[v.sym] = ptr
			if v.init != 0 {
				inst(&s.body, .Store, ptr, emit_expr(s, v.init))
			} else {
				inst(&s.body, .Store, ptr, const_null(s, type_id(s, t)))
			}
		}
	case .Expr:
		emit_expr(s, st.expr)
	case .If:
		cond := emit_expr(s, st.expr)
		then_l := new_id(s)
		merge := new_id(s)
		else_l := merge
		if st.else_ != 0 {
			else_l = new_id(s)
		}
		inst(&s.body, .Selection_Merge, merge, 0)
		terminate(s, .Branch_Conditional, cond, then_l, else_l)
		label(s, then_l)
		emit_stmt(s, st.then_)
		then_term := s.terminated
		if !s.terminated {
			terminate(s, .Branch, merge)
		}
		else_term := false
		if st.else_ != 0 {
			label(s, else_l)
			emit_stmt(s, st.else_)
			else_term = s.terminated
			if !s.terminated {
				terminate(s, .Branch, merge)
			}
		}
		label(s, merge)
		if then_term && else_term {
			terminate(s, .Unreachable)
		}
	case .For, .While:
		emit_stmt(s, st.init)
		header := new_id(s)
		cond_l := new_id(s)
		body_l := new_id(s)
		cont := new_id(s)
		merge := new_id(s)
		terminate(s, .Branch, header)
		label(s, header)
		inst(&s.body, .Loop_Merge, merge, cont, 0)
		terminate(s, .Branch, cond_l)
		label(s, cond_l)
		append(&s.loops, Loop_Target{merge, cont, st.expr != 0})
		if st.expr != 0 {
			c := emit_expr(s, st.expr)
			terminate(s, .Branch_Conditional, c, body_l, merge)
		} else {
			terminate(s, .Branch, body_l)
		}
		label(s, body_l)
		emit_stmt(s, st.then_)
		if !s.terminated {
			terminate(s, .Branch, cont)
		}
		label(s, cont)
		if st.step != 0 {
			emit_expr(s, st.step)
		}
		terminate(s, .Branch, header)
		target := pop(&s.loops)
		label(s, merge)
		if !target.broke {
			terminate(s, .Unreachable)
		}
	case .Do_While:
		header := new_id(s)
		body_l := new_id(s)
		cont := new_id(s)
		merge := new_id(s)
		terminate(s, .Branch, header)
		label(s, header)
		inst(&s.body, .Loop_Merge, merge, cont, 0)
		terminate(s, .Branch, body_l)
		label(s, body_l)
		append(&s.loops, Loop_Target{merge, cont, true})
		emit_stmt(s, st.then_)
		if !s.terminated {
			terminate(s, .Branch, cont)
		}
		label(s, cont)
		c := emit_expr(s, st.expr)
		terminate(s, .Branch_Conditional, c, header, merge)
		pop(&s.loops)
		label(s, merge)
	case .Break:
		l := &s.loops[len(s.loops) - 1]
		l.broke = true
		terminate(s, .Branch, l.merge)
	case .Continue:
		terminate(s, .Branch, s.loops[len(s.loops) - 1].cont)
	case .Discard:
		terminate(s, .Kill)
	case .Switch:
		emit_switch(s, st)
	case .Case, .Default:
		// Handled by emit_switch.
	case .Return:
		if st.expr != 0 {
			terminate(s, .Return_Value, emit_expr(s, st.expr))
		} else {
			terminate(s, .Return)
		}
	}
}

// OpSwitch over the labels. Structured SPIR-V only lets a case fall into
// the case right after it in operand order (default first), so instead of
// branching between cases each label's block runs the statements from the
// label to the end of the body until one ends the path (break, return,
// ...): fallthrough becomes the following cases' code repeated, and every
// case leaves through the merge block. Adjacent labels share a block.
@(private)
emit_switch :: proc(s: ^Spv, st: Stmt) {
	u := s.u
	sel := emit_expr(s, st.expr)
	body := list(u, st.body)
	merge := new_id(s)
	blocks := make([]u32, len(body), context.temp_allocator)
	starts := make([dynamic]int, context.temp_allocator)
	default_l := merge
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, sel, 0)
	for b, i in body {
		k := u.stmts[b].kind
		if k != .Case && k != .Default {
			continue
		}
		prev := u.stmts[body[i - 1]].kind if i > 0 else Stmt_Kind.Empty
		if i > 0 && (prev == .Case || prev == .Default) {
			blocks[i] = blocks[i - 1]
		} else {
			blocks[i] = new_id(s)
			append(&starts, i)
		}
		if k == .Default {
			default_l = blocks[i]
		} else {
			append(&words, u32(u.stmts[b].value), blocks[i])
		}
	}
	words[1] = default_l
	inst(&s.body, .Selection_Merge, merge, 0)
	terminate(s, .Switch, ..words[:])
	cont := s.loops[len(s.loops) - 1].cont if len(s.loops) > 0 else 0
	append(&s.loops, Loop_Target{merge, cont, false})
	reaches_merge := default_l == merge
	for start in starts {
		label(s, blocks[start])
		for b in body[start:] {
			k := u.stmts[b].kind
			if k == .Case || k == .Default {
				continue
			}
			emit_stmt(s, Stmt_Id(b))
			if s.terminated {
				break
			}
		}
		if !s.terminated {
			terminate(s, .Branch, merge)
			reaches_merge = true
		}
	}
	target := pop(&s.loops)
	label(s, merge)
	if !reaches_merge && !target.broke {
		terminate(s, .Unreachable)
	}
}

// ---- lvalues ---------------------------------------------------------------------------

// True when `e` names memory (so indexing can use access chains).
@(private)
addressable :: proc(u: ^Unit, id: Expr_Id) -> bool {
	e := u.exprs[id]
	#partial switch e.kind {
	case .Ident:
		return e.sym != 0 && u.syms[e.sym].storage != .Channel
	case .Index:
		return addressable(u, e.a)
	case .Field:
		return e.swizzle_len == 0 && addressable(u, e.a)
	}
	return false
}

@(private)
emit_lvalue :: proc(s: ^Spv, id: Expr_Id) -> LValue {
	u := s.u
	e := u.exprs[id]
	#partial switch e.kind {
	case .Ident:
		sym := u.syms[e.sym]
		t := ty(u, sym.type)
		switch sym.storage {
		case .Uniform, .Builtin_Uniform:
			idx := s.members[e.sym] if sym.storage == .Uniform else builtin_member_index(sym.index)
			as_uint := t.base == .Bool
			ptr := new_id(s)
			inst(&s.body, .Access_Chain, ptr_type(s, SC_UNIFORM, s.member_types[idx]), ptr, s.block_var, const_u32(s, idx))
			return {ptr = ptr, type = t, storage = SC_UNIFORM, as_uint = as_uint}
		case .Global:
			return {ptr = s.sym_ids[e.sym], type = t, storage = SC_PRIVATE}
		case .Stage_In:
			return {ptr = s.sym_ids[e.sym], type = t, storage = SC_INPUT}
		case .Stage_Out:
			return {ptr = s.sym_ids[e.sym], type = t, storage = SC_OUTPUT}
		case .Local, .Param:
			return {ptr = s.sym_ids[e.sym], type = t, storage = SC_FUNCTION}
		case .Channel:
		}
	case .Index:
		base := emit_lvalue(s, e.a)
		idx := emit_expr(s, e.b)
		return chain(s, base, idx, ty(u, e.type))
	case .Field:
		base := emit_lvalue(s, e.a)
		if e.swizzle_len > 0 {
			base.swz = e.swizzle
			base.swz_len = int(e.swizzle_len)
			return base
		}
		return chain(s, base, const_i32(s, e.field), ty(u, e.type))
	}
	return {}
}

@(private)
chain :: proc(s: ^Spv, base: LValue, index: u32, result: Type) -> LValue {
	rt := result
	if base.as_uint {
		rt.base = .Uint
	}
	ptr := new_id(s)
	inst(&s.body, .Access_Chain, ptr_type(s, base.storage, type_id(s, rt)), ptr, base.ptr, index)
	return {ptr = ptr, type = result, storage = base.storage, as_uint = base.as_uint}
}

@(private)
load_lvalue :: proc(s: ^Spv, lv: LValue) -> u32 {
	t := lv.type
	if lv.storage == SC_UNIFORM && t.array > 0 {
		// A whole std140 array becomes a plain array value.
		elems := make([]u32, int(t.array), context.temp_allocator)
		for &el, i in elems {
			el = load_lvalue(s, chain(s, lv, const_i32(s, i32(i)), elem_type(t)))
		}
		return construct(s, type_id(s, t), ..elems)
	}
	v: u32
	if lv.as_uint {
		ut := with_base(t, .Uint)
		raw := new_id(s)
		inst(&s.body, .Load, type_id(s, ut), raw, lv.ptr)
		v = convert(s, raw, ut, t)
	} else {
		v = new_id(s)
		inst(&s.body, .Load, type_id(s, t), v, lv.ptr)
	}
	if lv.swz_len > 0 {
		return swizzle_value(s, v, t, lv.swz, lv.swz_len)
	}
	return v
}

@(private)
swizzle_value :: proc(s: ^Spv, v: u32, t: Type, swz: [4]u8, n: int) -> u32 {
	if is_scalar(t) {
		if n == 1 {
			return v
		}
		parts := make([]u32, n, context.temp_allocator)
		for &p in parts {
			p = v
		}
		return construct(s, type_id(s, vec_type(t.base, n)), ..parts)
	}
	if n == 1 {
		return extract(s, type_id(s, scalar_of(t)), v, u32(swz[0]))
	}
	id := new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, type_id(s, vec_type(t.base, n)), id, v, v)
	for i in 0 ..< n {
		append(&words, u32(swz[i]))
	}
	inst_words(&s.body, .Vector_Shuffle, words[:])
	return id
}

@(private)
store_lvalue :: proc(s: ^Spv, lv: LValue, v: u32) {
	t := lv.type
	if lv.swz_len == 0 || is_scalar(t) {
		inst(&s.body, .Store, lv.ptr, v)
		return
	}
	old := new_id(s)
	inst(&s.body, .Load, type_id(s, t), old, lv.ptr)
	nv := new_id(s)
	if lv.swz_len == 1 {
		inst(&s.body, .Composite_Insert, type_id(s, t), nv, v, old, u32(lv.swz[0]))
	} else {
		// Component i comes from the new value where the swizzle writes it,
		// else from the old value.
		words := make([dynamic]u32, context.temp_allocator)
		append(&words, type_id(s, t), nv, old, v)
		for i in 0 ..< int(t.rows) {
			sel := u32(i)
			for k in 0 ..< lv.swz_len {
				if int(lv.swz[k]) == i {
					sel = u32(int(t.rows) + k)
				}
			}
			append(&words, sel)
		}
		inst_words(&s.body, .Vector_Shuffle, words[:])
	}
	inst(&s.body, .Store, lv.ptr, nv)
}

// ---- expressions -----------------------------------------------------------------------

@(private)
emit_expr :: proc(s: ^Spv, id: Expr_Id) -> u32 {
	u := s.u
	e := u.exprs[id]
	t := ty(u, e.type)
	switch e.kind {
	case .None:
		return 0
	case .Int_Lit:
		return const_i32(s, i32(e.value))
	case .Uint_Lit:
		return const_u32(s, u32(e.value))
	case .Float_Lit:
		return const_f32(s, f32(lit_f64(e)))
	case .Bool_Lit:
		return const_bits(s, .Bool, u32(e.value))
	case .Ident:
		if u.syms[e.sym].storage == .Channel {
			return s.sym_ids[e.sym]
		}
		return load_lvalue(s, emit_lvalue(s, id))
	case .Convert:
		return convert(s, emit_expr(s, e.a), expr_type(u, e.a), t)
	case .Comma:
		emit_expr(s, e.a)
		return emit_expr(s, e.b)
	case .Length:
		at := expr_type(u, e.a)
		n := at.array if is_array(at) else (i32(at.cols) if is_matrix(at) else i32(at.rows))
		return const_i32(s, n)
	case .Unary:
		v := emit_expr(s, e.a)
		tt := type_id(s, t)
		#partial switch e.op {
		case .Minus:
			if is_matrix(t) {
				return per_column(s, t, v, 0, proc(s: ^Spv, ct: u32, a, b: u32) -> u32 {return unop(s, .F_Negate, ct, a)})
			}
			return unop(s, .F_Negate if t.base == .Float else .S_Negate, tt, v)
		case .Bang:
			return unop(s, .Logical_Not, tt, v)
		case .Tilde:
			return unop(s, .Not, tt, v)
		}
		return v
	case .Binary:
		#partial switch e.op {
		case .And_And, .Or_Or:
			return emit_short_circuit(s, e)
		}
		a := emit_expr(s, e.a)
		b := emit_expr(s, e.b)
		return binop(s, e.op, a, expr_type(u, e.a), b, expr_type(u, e.b), t)
	case .Assign:
		lv := emit_lvalue(s, e.a)
		v := emit_expr(s, e.b)
		if e.op != .Eq {
			old := load_lvalue(s, lv)
			lt := expr_type(u, e.a)
			v = binop(s, compound_op(e.op), old, lt, v, expr_type(u, e.b), lt)
		}
		store_lvalue(s, lv, v)
		return v
	case .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
		lv := emit_lvalue(s, e.a)
		old := load_lvalue(s, lv)
		inc := e.kind == .Pre_Inc || e.kind == .Post_Inc
		nv := binop(s, .Plus if inc else .Minus, old, t, const_of(s, t.base, 1), scalar_of(t), t)
		store_lvalue(s, lv, nv)
		return nv if e.kind == .Pre_Inc || e.kind == .Pre_Dec else old
	case .Ternary:
		cond := emit_expr(s, e.a)
		then_l, else_l, merge := new_id(s), new_id(s), new_id(s)
		inst(&s.body, .Selection_Merge, merge, 0)
		terminate(s, .Branch_Conditional, cond, then_l, else_l)
		label(s, then_l)
		a := emit_expr(s, e.b)
		a_block := s.cur_block
		terminate(s, .Branch, merge)
		label(s, else_l)
		b := emit_expr(s, e.c)
		b_block := s.cur_block
		terminate(s, .Branch, merge)
		label(s, merge)
		r := new_id(s)
		inst(&s.body, .Phi, type_id(s, t), r, a, a_block, b, b_block)
		return r
	case .Index:
		if addressable(u, e.a) {
			return load_lvalue(s, emit_lvalue(s, id))
		}
		base := emit_expr(s, e.a)
		bt := expr_type(u, e.a)
		ix := u.exprs[e.b]
		if ix.kind == .Int_Lit || ix.kind == .Uint_Lit {
			return extract(s, type_id(s, t), base, u32(ix.value))
		}
		idx := emit_expr(s, e.b)
		if is_vector(bt) {
			r := new_id(s)
			inst(&s.body, .Vector_Extract_Dynamic, type_id(s, t), r, base, idx)
			return r
		}
		// Dynamic index into an array or matrix value: through a temporary.
		tmp := local_var(s, bt)
		inst(&s.body, .Store, tmp, base)
		ptr := new_id(s)
		inst(&s.body, .Access_Chain, ptr_type(s, SC_FUNCTION, type_id(s, t)), ptr, tmp, idx)
		r := new_id(s)
		inst(&s.body, .Load, type_id(s, t), r, ptr)
		return r
	case .Field:
		if addressable(u, e.a) {
			return load_lvalue(s, emit_lvalue(s, id))
		}
		base := emit_expr(s, e.a)
		if e.swizzle_len > 0 {
			return swizzle_value(s, base, expr_type(u, e.a), e.swizzle, int(e.swizzle_len))
		}
		return extract(s, type_id(s, t), base, u32(e.field))
	case .Call:
		switch e.call {
		case .User:
			return emit_user_call(s, id)
		case .Builtin:
			return emit_builtin(s, id)
		case .Constructor:
			return emit_constructor(s, id)
		case .None:
		}
	}
	return 0
}

@(private)
emit_short_circuit :: proc(s: ^Spv, e: Expr) -> u32 {
	a := emit_expr(s, e.a)
	a_block := s.cur_block
	rhs_l, merge := new_id(s), new_id(s)
	inst(&s.body, .Selection_Merge, merge, 0)
	if e.op == .And_And {
		terminate(s, .Branch_Conditional, a, rhs_l, merge)
	} else {
		terminate(s, .Branch_Conditional, a, merge, rhs_l)
	}
	label(s, rhs_l)
	b := emit_expr(s, e.b)
	b_block := s.cur_block
	terminate(s, .Branch, merge)
	label(s, merge)
	r := new_id(s)
	inst(&s.body, .Phi, type_id(s, T_BOOL), r, a, a_block, b, b_block)
	return r
}

// ---- calls -----------------------------------------------------------------------------

@(private)
emit_user_call :: proc(s: ^Spv, id: Expr_Id) -> u32 {
	u := s.u
	f := u.exprs[id].func
	args := args_of(u, id)
	ps := fparams(u, f)
	vals := make([]u32, len(args), context.temp_allocator)
	temps := make([]u32, len(args), context.temp_allocator)
	lvs := make([]LValue, len(args), context.temp_allocator)
	for p, i in ps {
		pt := ty(u, p.type)
		switch {
		case pt.base == .Sampler:
			vals[i] = s.sym_ids[u.exprs[args[i]].sym]
		case p.qual == .In:
			vals[i] = emit_expr(s, args[i])
		case:
			lvs[i] = emit_lvalue(s, args[i])
			tmp := local_var(s, pt)
			if p.qual == .Inout {
				inst(&s.body, .Store, tmp, load_lvalue(s, lvs[i]))
			} else {
				inst(&s.body, .Store, tmp, const_null(s, type_id(s, pt)))
			}
			temps[i] = tmp
			vals[i] = tmp
		}
	}
	r := new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, type_id(s, ty(u, u.funcs[f].ret)), r, s.func_ids[f])
	append(&words, ..vals)
	inst_words(&s.body, .Function_Call, words[:])
	for p, i in ps {
		pt := ty(u, p.type)
		if p.qual != .In && pt.base != .Sampler {
			v := new_id(s)
			inst(&s.body, .Load, type_id(s, pt), v, temps[i])
			store_lvalue(s, lvs[i], v)
		}
	}
	return r
}

// Scalar components of a value, flattened column-major.
@(private)
components :: proc(s: ^Spv, v: u32, t: Type) -> []u32 {
	st := type_id(s, scalar_of(t))
	if is_scalar(t) {
		out := make([]u32, 1, context.temp_allocator)
		out[0] = v
		return out
	}
	if is_vector(t) {
		out := make([]u32, int(t.rows), context.temp_allocator)
		for &o, i in out {
			o = extract(s, st, v, u32(i))
		}
		return out
	}
	out := make([]u32, int(t.rows) * int(t.cols), context.temp_allocator)
	for c in 0 ..< int(t.cols) {
		for r in 0 ..< int(t.rows) {
			out[c * int(t.rows) + r] = extract(s, st, v, u32(c), u32(r))
		}
	}
	return out
}

@(private)
emit_constructor :: proc(s: ^Spv, id: Expr_Id) -> u32 {
	u := s.u
	t := expr_type(u, id)
	tt := type_id(s, t)
	args := args_of(u, id)
	vals := make([]u32, len(args), context.temp_allocator)
	for a, i in args {
		vals[i] = emit_expr(s, a)
	}
	if t.array > 0 || t.base == .Struct {
		return construct(s, tt, ..vals)
	}
	st := scalar_of(t)
	if len(args) == 1 {
		at := expr_type(u, args[0])
		v := vals[0]
		if is_matrix(t) && is_matrix(at) {
			// Resize: overlapping components, identity elsewhere.
			ct := type_id(s, column_type(t))
			cols := make([]u32, int(t.cols), context.temp_allocator)
			for &col, c in cols {
				rows := make([]u32, int(t.rows), context.temp_allocator)
				for &r, ri in rows {
					if c < int(at.cols) && ri < int(at.rows) {
						r = extract(s, type_id(s, T_FLOAT), v, u32(c), u32(ri))
					} else {
						r = const_f32(s, 1 if c == ri else 0)
					}
				}
				col = construct(s, ct, ..rows)
			}
			return construct(s, tt, ..cols)
		}
		if is_scalar(at) {
			sv := convert(s, v, at, st)
			if is_scalar(t) {
				return sv
			}
			if is_vector(t) {
				return splat(s, sv, st, t)
			}
			// Diagonal matrix.
			ct := type_id(s, column_type(t))
			zero := const_f32(s, 0)
			cols := make([]u32, int(t.cols), context.temp_allocator)
			for &col, c in cols {
				rows := make([]u32, int(t.rows), context.temp_allocator)
				for &r, ri in rows {
					r = sv if c == ri else zero
				}
				col = construct(s, ct, ..rows)
			}
			return construct(s, tt, ..cols)
		}
		if is_scalar(t) {
			return convert(s, components(s, v, at)[0], scalar_of(at), st)
		}
		if is_vector(t) && is_vector(at) && at.rows == t.rows {
			return convert(s, v, at, t)
		}
	}
	// Flatten every argument and take the components in order.
	flat := make([dynamic]u32, context.temp_allocator)
	for a, i in args {
		at := expr_type(u, a)
		for comp in components(s, vals[i], at) {
			append(&flat, convert(s, comp, scalar_of(at), st))
		}
	}
	if is_vector(t) {
		return construct(s, tt, ..flat[:int(t.rows)])
	}
	ct := type_id(s, column_type(t))
	cols := make([]u32, int(t.cols), context.temp_allocator)
	for &col, c in cols {
		col = construct(s, ct, ..flat[c * int(t.rows):(c + 1) * int(t.rows)])
	}
	return construct(s, tt, ..cols)
}

@(private)
emit_builtin :: proc(s: ^Spv, id: Expr_Id) -> u32 {
	u := s.u
	e := u.exprs[id]
	b := e.builtin
	t := ty(u, e.type)
	rt := type_id(s, t)
	argx := args_of(u, id)
	args := make([]u32, len(argx), context.temp_allocator)
	types := make([]Type, len(argx), context.temp_allocator)
	for a, i in argx {
		types[i] = expr_type(u, a)
		if types[i].base == .Sampler {
			args[i] = s.sym_ids[u.exprs[a].sym]
		} else {
			args[i] = emit_expr(s, a)
		}
	}
	at := types[0]
	for i in BUILTINS[b].broadcast {
		args[i] = splat(s, args[i], types[i], with_base(t, types[i].base))
	}
	if op := SPV_EXT_BUILTINS[b][at.base]; op != Glsl(0) {
		return ext(s, rt, op, ..args)
	}
	if op := SPV_OP_BUILTINS[b][at.base]; op != Op(0) {
		r := new_id(s)
		words := make([dynamic]u32, context.temp_allocator)
		append(&words, rt, r)
		append(&words, ..args)
		inst_words(&s.body, op, words[:])
		return r
	}
	#partial switch b {
	case .Mod:
		// GLSL: x - y * floor(x / y).
		q := binop_raw(s, .F_Div, rt, args[0], args[1])
		fq := ext(s, rt, .Floor, q)
		m := binop_raw(s, .F_Mul, rt, args[1], fq)
		return binop_raw(s, .F_Sub, rt, args[0], m)
	case .Mix_Bool:
		r := new_id(s)
		inst(&s.body, .Select, rt, r, args[2], args[1], args[0])
		return r
	case .Dot:
		if is_scalar(at) {
			return binop_raw(s, .F_Mul, rt, args[0], args[1])
		}
		return binop_raw(s, .Dot, rt, args[0], args[1])
	case .Matrix_Comp_Mult:
		return per_column(s, t, args[0], args[1], proc(s: ^Spv, ct: u32, a, b: u32) -> u32 {return binop_raw(s, .F_Mul, ct, a, b)})
	case .Texture_Lod:
		img := new_id(s)
		inst(&s.body, .Load, sampled_image_type(s), img, args[0])
		r := new_id(s)
		inst(&s.body, .Image_Sample_Explicit_Lod, rt, r, img, args[1], 0x2, args[2])
		return r
	}
	return 0
}

// The GLSL.std.450 instruction each built-in is, by its first argument's
// base. Arguments in BUILTINS' broadcast sets are splatted first.
@(rodata)
SPV_EXT_BUILTINS := #partial [Builtin][Base]Glsl {
	.Radians     = #partial {.Float = .Radians},
	.Degrees     = #partial {.Float = .Degrees},
	.Sin         = #partial {.Float = .Sin},
	.Cos         = #partial {.Float = .Cos},
	.Tan         = #partial {.Float = .Tan},
	.Asin        = #partial {.Float = .Asin},
	.Acos        = #partial {.Float = .Acos},
	.Atan        = #partial {.Float = .Atan},
	.Atan2       = #partial {.Float = .Atan2},
	.Sinh        = #partial {.Float = .Sinh},
	.Cosh        = #partial {.Float = .Cosh},
	.Tanh        = #partial {.Float = .Tanh},
	.Asinh       = #partial {.Float = .Asinh},
	.Acosh       = #partial {.Float = .Acosh},
	.Atanh       = #partial {.Float = .Atanh},
	.Pow         = #partial {.Float = .Pow},
	.Exp         = #partial {.Float = .Exp},
	.Log         = #partial {.Float = .Log},
	.Exp2        = #partial {.Float = .Exp2},
	.Log2        = #partial {.Float = .Log2},
	.Sqrt        = #partial {.Float = .Sqrt},
	.Inversesqrt = #partial {.Float = .Inverse_Sqrt},
	.Floor       = #partial {.Float = .Floor},
	.Ceil        = #partial {.Float = .Ceil},
	.Fract       = #partial {.Float = .Fract},
	.Trunc       = #partial {.Float = .Trunc},
	.Round       = #partial {.Float = .Round},
	.Round_Even  = #partial {.Float = .Round_Even},
	.Normalize   = #partial {.Float = .Normalize},
	.Abs         = #partial {.Float = .F_Abs, .Int = .S_Abs},
	.Sign        = #partial {.Float = .F_Sign, .Int = .S_Sign},
	.Min         = #partial {.Float = .F_Min, .Int = .S_Min, .Uint = .U_Min},
	.Max         = #partial {.Float = .F_Max, .Int = .S_Max, .Uint = .U_Max},
	.Clamp       = #partial {.Float = .F_Clamp, .Int = .S_Clamp, .Uint = .U_Clamp},
	.Mix         = #partial {.Float = .F_Mix},
	.Step        = #partial {.Float = .Step},
	.Smoothstep  = #partial {.Float = .Smooth_Step},
	.Length      = #partial {.Float = .Length},
	.Distance    = #partial {.Float = .Distance},
	.Cross       = #partial {.Float = .Cross},
	.Faceforward = #partial {.Float = .Face_Forward},
	.Reflect     = #partial {.Float = .Reflect},
	.Refract     = #partial {.Float = .Refract},
	.Determinant = #partial {.Float = .Determinant},
	.Inverse     = #partial {.Float = .Inverse},
}

// The core instruction each remaining built-in is, by its first argument's
// base. The rest are emitted by hand above.
@(rodata)
SPV_OP_BUILTINS := #partial [Builtin][Base]Op {
	.Isnan              = #partial {.Float = .Is_Nan},
	.Isinf              = #partial {.Float = .Is_Inf},
	.Outer_Product      = #partial {.Float = .Outer_Product},
	.Transpose          = #partial {.Float = .Transpose},
	.Less_Than          = #partial {.Float = .F_Ord_Less_Than, .Int = .S_Less_Than, .Uint = .U_Less_Than},
	.Less_Than_Equal    = #partial {.Float = .F_Ord_Less_Than_Equal, .Int = .S_Less_Than_Equal, .Uint = .U_Less_Than_Equal},
	.Greater_Than       = #partial {.Float = .F_Ord_Greater_Than, .Int = .S_Greater_Than, .Uint = .U_Greater_Than},
	.Greater_Than_Equal = #partial {.Float = .F_Ord_Greater_Than_Equal, .Int = .S_Greater_Than_Equal, .Uint = .U_Greater_Than_Equal},
	.Equal              = #partial {.Float = .F_Ord_Equal, .Int = .I_Equal, .Uint = .I_Equal, .Bool = .Logical_Equal},
	.Not_Equal          = #partial {.Float = .F_Unord_Not_Equal, .Int = .I_Not_Equal, .Uint = .I_Not_Equal, .Bool = .Logical_Not_Equal},
	.Any                = #partial {.Bool = .Any},
	.All                = #partial {.Bool = .All},
	.Not                = #partial {.Bool = .Logical_Not},
	.Dfdx               = #partial {.Float = .D_Pdx},
	.Dfdy               = #partial {.Float = .D_Pdy},
	.Fwidth             = #partial {.Float = .Fwidth},
	.Float_Bits_To_Int  = #partial {.Float = .Bitcast},
	.Float_Bits_To_Uint = #partial {.Float = .Bitcast},
	.Int_Bits_To_Float  = #partial {.Int = .Bitcast},
	.Uint_Bits_To_Float = #partial {.Uint = .Bitcast},
}
