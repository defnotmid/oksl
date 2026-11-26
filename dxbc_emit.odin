package oksl

import "core:math"

// DXBC emission of statements and expressions (see dxbc.odin for the
// container, the encoding and the interface). Expressions return a Dx_Src
// naming where their value is (a variable's own registers, a uniform, an
// input, a fresh temporary or an immediate); temporaries are released at
// the end of the statement that made them, variables at the end of their
// block.

// ---- statements ------------------------------------------------------------------------

@(private)
dx_block :: proc(d: ^Dx, id: Stmt_Id) {
	m := dx_mark(d)
	for b in list(d.u, d.u.stmts[id].body) {
		if d.terminated {
			break
		}
		dx_stmt(d, Stmt_Id(b))
	}
	dx_release(d, m)
}

// A statement in its own scope (an if's branch, a loop's body).
@(private)
dx_scoped :: proc(d: ^Dx, id: Stmt_Id) {
	m := dx_mark(d)
	dx_stmt(d, id)
	dx_release(d, m)
}

@(private)
dx_stmt :: proc(d: ^Dx, id: Stmt_Id) {
	u := d.u
	if id == 0 || d.terminated {
		return
	}
	st := u.stmts[id]
	switch st.kind {
	case .Empty, .Case, .Default:
	case .Block:
		dx_block(d, id)
	case .Decl:
		for v in list(u, st.decls) {
			decl := u.vars[v]
			t := sym_type(u, decl.sym)
			loc := dx_alloc(d, t, d.sym_x[decl.sym])
			m := dx_mark(d)
			if decl.init != 0 {
				dx_store(d, loc, t, dx_expr(d, decl.init))
			} else {
				dx_zero(d, loc, t)
			}
			dx_release(d, m)
			d.sym_loc[decl.sym] = loc
		}
	case .Expr:
		m := dx_mark(d)
		dx_expr(d, st.expr)
		dx_release(d, m)
	case .If:
		m := dx_mark(d)
		cond := dx_expr(d, st.expr)
		dx_flow(&d.code, .If, DX_NZ, cond)
		dx_release(d, m)
		dx_scoped(d, st.then_)
		then_term := d.terminated
		d.terminated = false
		else_term := false
		if st.else_ != 0 {
			dx_flow(&d.code, .Else)
			dx_scoped(d, st.else_)
			else_term = d.terminated
			d.terminated = false
		}
		dx_flow(&d.code, .End_If)
		d.terminated = then_term && else_term
	case .For, .While, .Do_While:
		// One shape for every loop, so `continue` is always a bare
		// continue (never code that could land inside a switch):
		//     loop
		//       if !first { step; breakc_z cond (a do-while's) } first = false
		//       breakc_z cond (a for's or while's)
		//       body
		//     endloop
		// A loop with neither a step nor a do-while test needs no flag.
		m := dx_mark(d)
		dx_stmt(d, st.init)
		has_first := st.kind == .Do_While || st.step != 0
		first: Dx_Src
		if has_first {
			first = dx_scalar_temp(d)
			dx_ins(&d.code, .Mov, 0, first, 1, dx_imm(0xffffffff))
		}
		dx_flow(&d.code, .Loop)
		append(&d.flow, Dx_Flow{kind = .Loop})
		if has_first {
			dx_flow(&d.code, .If, 0, first)
			if st.step != 0 {
				sm := dx_mark(d)
				dx_expr(d, st.step)
				dx_release(d, sm)
			}
			if st.kind == .Do_While {
				dx_break_unless(d, st.expr)
			}
			dx_flow(&d.code, .End_If)
			dx_ins(&d.code, .Mov, 0, first, 1, dx_imm(0))
		}
		if st.kind != .Do_While && st.expr != 0 {
			dx_break_unless(d, st.expr)
		}
		dx_scoped(d, st.then_)
		d.terminated = false
		dx_flow(&d.code, .End_Loop)
		dx_end_construct(d)
		dx_release(d, m)
	case .Break:
		dx_flow(&d.code, .Break)
		d.last_break = len(d.code)
		d.terminated = true
	case .Continue:
		dx_flow(&d.code, .Continue)
		d.terminated = true
	case .Discard:
		// GLSL's discard ends the invocation (as SPIR-V's OpKill and the
		// CPU's kill mask do); discard_nz alone only marks the pixel.
		dx_flow(&d.code, .Discard, DX_NZ, dx_imm(0xffffffff))
		dx_flow(&d.code, .Ret)
		d.last_break = len(d.code)
		d.terminated = true
	case .Switch:
		dx_switch(d, st)
	case .Return:
		dx_return(d, st)
	}
}

// `breakc_z cond`: leaves the loop when `cond` is false.
@(private)
dx_break_unless :: proc(d: ^Dx, cond: Expr_Id) {
	m := dx_mark(d)
	c := dx_expr(d, cond)
	dx_flow(&d.code, .Breakc, 0, c)
	dx_release(d, m)
}

// Pops a loop or switch; when a return inside it broke out with the flag
// set, the enclosing construct breaks too (through an if inside a switch:
// a breakc there would leave only the switch on D3D, and vkd3d takes it
// for the loop).
@(private)
dx_end_construct :: proc(d: ^Dx) {
	f := pop(&d.flow)
	if !f.returned {
		return
	}
	flag := dx_reg(.Temp, d.calls[len(d.calls) - 1].flag)
	top := len(d.flow) - 1
	if d.flow[top].kind == .Switch {
		dx_flow(&d.code, .If, DX_NZ, flag)
		dx_flow(&d.code, .Break)
		dx_flow(&d.code, .End_If)
	} else {
		dx_flow(&d.code, .Breakc, DX_NZ, flag)
	}
	if d.flow[top].kind != .Return {
		d.flow[top].returned = true
	}
}

// A switch over the labels. Each group of adjacent labels runs the
// statements from it to the end of the body until one ends the path (as
// the SPIR-V backend does), so no case falls into another.
@(private)
dx_switch :: proc(d: ^Dx, st: Stmt) {
	u := d.u
	m := dx_mark(d)
	sel := dx_expr(d, st.expr)
	dx_flow(&d.code, .Switch, 0, sel)
	dx_release(d, m)
	append(&d.flow, Dx_Flow{kind = .Switch})
	body := list(u, st.body)
	for b, i in body {
		k := u.stmts[b].kind
		if k != .Case && k != .Default {
			continue
		}
		if k == .Default {
			dx_flow(&d.code, .Default)
		} else {
			dx_flow(&d.code, .Case, 0, dx_imm(u32(u.stmts[b].value)))
		}
		next := u.stmts[body[i + 1]].kind if i + 1 < len(body) else Stmt_Kind.Empty
		if i + 1 < len(body) && (next == .Case || next == .Default) {
			continue
		}
		bm := dx_mark(d)
		for s in body[i + 1:] {
			sk := u.stmts[s].kind
			if sk == .Case || sk == .Default {
				continue
			}
			dx_stmt(d, Stmt_Id(s))
			if d.terminated {
				break
			}
		}
		dx_release(d, bm)
		// A case ends with a break unless its last instruction is one
		// (after a continue, a discard or an if that returns on both
		// sides too: the case's block must end there).
		if d.last_break != len(d.code) {
			dx_flow(&d.code, .Break)
		}
		d.terminated = false
	}
	dx_flow(&d.code, .End_Switch)
	dx_end_construct(d)
}

@(private)
dx_return :: proc(d: ^Dx, st: Stmt) {
	call := d.calls[len(d.calls) - 1]
	if st.expr != 0 {
		m := dx_mark(d)
		dx_store(d, call.ret, call.ret_type, dx_expr(d, st.expr))
		dx_release(d, m)
	}
	d.terminated = true
	if call.flow < 0 {
		return
	}
	if len(d.flow) - 1 > call.flow {
		dx_ins(&d.code, .Mov, 0, dx_reg(.Temp, call.flag), 1, dx_imm(0xffffffff))
		d.flow[len(d.flow) - 1].returned = true
	}
	dx_flow(&d.code, .Break)
	d.last_break = len(d.code)
}

// ---- values ----------------------------------------------------------------------------

// Copies a value of type `t` into the registers at `dst`.
@(private)
dx_store :: proc(d: ^Dx, dst: Dx_Src, t: Type, v: Dx_Src) {
	if is_sv(t) {
		dx_ins(&d.code, .Mov, 0, dst, dx_mask(int(t.rows)), dx_n(v, int(t.rows)))
		return
	}
	for k in 0 ..< dx_regs(d.u, t) {
		dx_ins(&d.code, .Mov, 0, dx_at(dst, k), 0xf, dx_at(v, k))
	}
}

// A copy of `v` in fresh registers.
@(private)
dx_copy :: proc(d: ^Dx, v: Dx_Src, t: Type) -> Dx_Src {
	r := dx_alloc(d, t)
	dx_store(d, r, t, v)
	return dx_comp(r, 0) if is_scalar(t) else r
}

// Applies an instruction component-wise: `n` components of a fresh temp
// (a scalar's swizzle broadcasts its x, as every scalar value's does).
@(private)
dx_op :: proc(d: ^Dx, op: Dx_Op, n: int, srcs: ..Dx_Src) -> Dx_Src {
	r := dx_temp(d)
	dx_ins(&d.code, op, 0, r, dx_mask(n), ..srcs)
	return dx_scalar_view(r, n)
}

// A fresh scalar temp, read as its x whatever the destination component.
@(private)
dx_scalar_temp :: proc(d: ^Dx) -> Dx_Src {
	return dx_comp(dx_temp(d), 0)
}

// A value of n components as read: a scalar's one component broadcast.
@(private)
dx_scalar_view :: proc "contextless" (r: Dx_Src, n: int) -> Dx_Src {
	return dx_comp(r, 0) if n == 1 else r
}

@(private)
dx_un :: proc(d: ^Dx, op: Dx_Op, n: int, a: Dx_Src) -> Dx_Src {
	return dx_op(d, op, n, dx_n(a, n))
}

// Expressions are evaluated left to right: an operand naming a variable is
// copied when a later operand may write it. The values go on the operand
// stack (d.vals) from the returned index; the caller pops them with
// dx_pop_operands. Each value is pushed once its expression is done, so
// nested operands have been popped by then.
@(private)
dx_operands :: proc(d: ^Dx, ids: []Expr_Id) -> int {
	base := len(d.vals)
	for id, i in ids {
		v := dx_expr(d, id)
		later := false
		for j in i + 1 ..< len(ids) {
			later ||= may_write(d.u, ids[j])
		}
		if later && v.file != .Imm {
			v = dx_copy(d, v, expr_type(d.u, id))
		}
		append(&d.vals, v)
	}
	return base
}

@(private)
dx_pop_operands :: proc(d: ^Dx, base: int) {
	resize(&d.vals, base)
}

// ---- lvalues ---------------------------------------------------------------------------

@(private)
Dx_LV :: struct {
	loc:     Dx_Src,
	type:    Type,
	// The components a vector's swizzle (or a uniform's place) names.
	swz:     [4]u8,
	swz_len: int,
	// A component chosen at run time: the temp holding it (its register
	// component, after the swizzle), -1 for none.
	comp:    i32,
	// Bools in the uniform block are 0 / 1 words.
	as_uint: bool,
	// Memory (dx_is_mem: shared memory, g0, packed as on the CPU; a storage
	// buffer's t# or u#, std430): the value's word, `word` plus the run-time
	// word offset in temp `dyn`'s x (-1 for none); a vector's swizzle is in
	// `swz`, a component chosen at run time is folded into `dyn`.
	word:    u32,
	dyn:     i32,
}

@(private)
dx_is_mem :: proc "contextless" (lv: Dx_LV) -> bool {
	return lv.loc.file == .Tgsm || lv.loc.file == .Resource || lv.loc.file == .Uav
}

// The clamped index `i` (0 ..< n) times `size`, added to `rel` when set,
// as a temp's x.
@(private)
dx_index :: proc(d: ^Dx, id: Expr_Id, n: int, size: int, rel: i32 = -1, rel_c: u8 = 0) -> u32 {
	i := dx_expr(d, id)
	t := dx_scalar_temp(d)
	dx_ins(&d.code, .Imax, 0, t, 1, i, dx_imm(0))
	dx_ins(&d.code, .Imin, 0, t, 1, t, dx_imm_i(i32(n - 1)))
	if size != 1 {
		dx_ins2(&d.code, .Imul, dx_null(), t, 1, t, dx_imm_i(i32(size)))
	}
	if rel >= 0 {
		dx_ins(&d.code, .Iadd, 0, t, 1, t, Dx_Src{file = .Temp, idx = {u32(rel), 0, 0}, rel = -1, swz = {rel_c, rel_c, rel_c, rel_c}})
	}
	return t.idx[0]
}

@(private)
dx_null :: proc "contextless" () -> Dx_Src {
	return {file = .Null, rel = -1}
}

// `base` indexed by the run-time register offset in temp `r`.x.
@(private)
dx_offset_by :: proc(d: ^Dx, base: Dx_Src, r: u32) -> Dx_Src {
	out := base
	out.rel = i32(r)
	out.rel_c = 0
	if base.file == .Cbuffer {
		d.cb_dynamic = true
	}
	return out
}

@(private)
dx_lvalue :: proc(d: ^Dx, id: Expr_Id) -> Dx_LV {
	u := d.u
	e := u.exprs[id]
	#partial switch e.kind {
	case .Ident:
		sym := u.syms[e.sym]
		t := ty(u, sym.type)
		if sym.storage == .Uniform || sym.storage == .Builtin_Uniform {
			loc := Dx_Src{file = .Cbuffer, idx = {0, 0, u32(sym.index / 16)}, rel = -1, swz = DX_XYZW}
			lv := Dx_LV{loc = loc, type = t, comp = -1, as_uint = elem_type(t).base == .Bool}
			if is_sv(t) {
				c := u8(sym.index % 16 / 4)
				lv.swz_len = int(t.rows)
				for k in 0 ..< 4 {
					lv.swz[k] = min(c + u8(k), 3)
				}
			}
			return lv
		}
		if sym.storage == .Shared {
			return {loc = dx_reg(.Tgsm, 0), type = t, comp = -1, word = d.sym_word[e.sym], dyn = -1}
		}
		return {loc = d.sym_loc[e.sym], type = t, comp = -1, dyn = -1}
	case .Buffer_Load:
		p, _ := buf_path(u, id)
		return dx_buf_lvalue(d, p)
	case .Index:
		if root, _ := target_buffer(u, id); root != 0 {
			p, _ := buf_path(u, id)
			return dx_buf_lvalue(d, p)
		}
		base := dx_lvalue(d, e.a)
		rt := expr_type(u, id)
		ix := u.exprs[e.b]
		bt := base.type
		lit := ix.kind == .Int_Lit || ix.kind == .Uint_Lit
		if is_vector(bt) {
			if base.swz_len == 0 {
				base.swz = DX_XYZW
				base.swz_len = int(bt.rows)
			}
			if lit {
				base.swz[0] = base.swz[clamp(int(ix.value), 0, base.swz_len - 1)]
				base.swz_len = 1
			} else {
				// The component after the swizzle: clamp, then map.
				i := dx_expr(d, e.b)
				t := dx_scalar_temp(d)
				dx_ins(&d.code, .Imax, 0, t, 1, i, dx_imm(0))
				dx_ins(&d.code, .Imin, 0, t, 1, t, dx_imm_i(i32(base.swz_len - 1)))
				mapped := dx_scalar_temp(d)
				dx_ins(&d.code, .Mov, 0, mapped, 1, dx_imm(u32(base.swz[0])))
				c := dx_scalar_temp(d)
				for k in 1 ..< base.swz_len {
					dx_ins(&d.code, .Ieq, 0, c, 1, t, dx_imm(u32(k)))
					dx_ins(&d.code, .Movc, 0, mapped, 1, c, dx_imm(u32(base.swz[k])), mapped)
				}
				if dx_is_mem(base) {
					// The component's word.
					if base.dyn >= 0 {
						dx_ins(&d.code, .Iadd, 0, mapped, 1, mapped, dx_comp(dx_reg(.Temp, u32(base.dyn)), 0))
					}
					base.dyn = i32(mapped.idx[0])
				} else {
					base.comp = i32(mapped.idx[0])
				}
				base.swz_len = 0
			}
			if lit && dx_is_mem(base) {
				// One component: its word.
				base.word += u32(base.swz[0])
				base.swz_len = 0
			}
			base.type = rt
			return base
		}
		n := int(bt.array) if is_array(bt) else int(bt.cols)
		if dx_is_mem(base) {
			// Elements of component_count words (matrices' columns rows).
			words := component_count(u, rt)
			if lit {
				base.word += u32(clamp(int(ix.value), 0, n - 1) * words)
			} else {
				base.dyn = i32(dx_index(d, e.b, n, words, base.dyn, 0))
			}
			base.type = rt
			return base
		}
		size := dx_regs(u, rt)
		if lit {
			base.loc = dx_at(base.loc, clamp(int(ix.value), 0, n - 1) * size)
		} else {
			// Only indexable temps and the constant buffer take run-time
			// indexes (dx_alloc and dx_find_indexed_matrices see to it).
			assert(base.loc.file == .Indexable || base.loc.file == .Cbuffer)
			r := dx_index(d, e.b, n, size, base.loc.rel, base.loc.rel_c)
			base.loc = dx_offset_by(d, base.loc, r)
		}
		base.type = rt
		return base
	case .Field:
		if root, _ := target_buffer(u, id); root != 0 {
			p, _ := buf_path(u, id)
			return dx_buf_lvalue(d, p)
		}
		base := dx_lvalue(d, e.a)
		if e.swizzle_len > 0 {
			if is_scalar(base.type) {
				return base
			}
			if base.swz_len == 0 {
				base.swz = e.swizzle
			} else {
				composed: [4]u8
				for k in 0 ..< int(e.swizzle_len) {
					composed[k] = base.swz[e.swizzle[k]]
				}
				base.swz = composed
			}
			base.swz_len = int(e.swizzle_len)
			base.type = expr_type(u, id)
			return base
		}
		if dx_is_mem(base) {
			base.word += u32(dx_field_words(u, base.type.strukt, e.field))
		} else {
			base.loc = dx_at(base.loc, dx_field_reg(u, base.type.strukt, e.field))
		}
		base.type = expr_type(u, id)
		return base
	}
	return {}
}

// The word offset of field `field` in struct `st` packed (shared memory).
@(private)
dx_field_words :: proc(u: ^Unit, st: i32, field: i32) -> int {
	n := 0
	for f, i in struct_fields(u, st) {
		if i32(i) == field {
			break
		}
		n += component_count(u, ty(u, f.type))
	}
	return n
}

@(private)
dx_load :: proc(d: ^Dx, lv: Dx_LV) -> Dx_Src {
	t := lv.type
	if dx_is_mem(lv) {
		return dx_mem_load(d, lv)
	}
	if lv.comp >= 0 {
		// One component of a vector chosen at run time.
		r := dx_scalar_temp(d)
		c := dx_scalar_temp(d)
		loc := lv.loc
		loc.swz = {0, 0, 0, 0}
		dx_ins(&d.code, .Mov, 0, r, 1, loc)
		for k in 1 ..< 4 {
			loc.swz = {u8(k), u8(k), u8(k), u8(k)}
			dx_ins(&d.code, .Ieq, 0, c, 1, dx_reg(.Temp, u32(lv.comp)), dx_imm(u32(k)))
			dx_ins(&d.code, .Movc, 0, r, 1, c, loc, r)
		}
		return dx_bool_word(d, r, t, lv.as_uint)
	}
	v := lv.loc
	if lv.swz_len > 0 {
		v.swz = lv.swz
	}
	if is_scalar(t) {
		v = dx_comp(v, 0)
	}
	return dx_bool_word(d, v, t, lv.as_uint)
}

// A uniform bool's 0 / 1 word as ~0 / 0.
@(private)
dx_bool_word :: proc(d: ^Dx, v: Dx_Src, t: Type, as_uint: bool) -> Dx_Src {
	if !as_uint {
		return v
	}
	if is_sv(t) {
		return dx_op(d, .Ine, int(t.rows), dx_n(v, int(t.rows)), dx_imm(0))
	}
	r := dx_alloc(d, t)
	for k in 0 ..< dx_regs(d.u, t) {
		dx_ins(&d.code, .Ine, 0, dx_at(r, k), 0xf, dx_at(v, k), dx_imm(0))
	}
	return r
}

@(private)
dx_store_lv :: proc(d: ^Dx, lv: Dx_LV, v: Dx_Src) {
	t := lv.type
	if dx_is_mem(lv) {
		if lv.swz_len > 0 {
			// A swizzle: each component to its word.
			for k in 0 ..< lv.swz_len {
				dx_mem_leaf(d, lv, lv.word + u32(lv.swz[k]), 1, dx_comp(v, 0 if is_scalar(t) else k), true)
			}
			return
		}
		dx_mem_walk(d, lv, lv.word, t, v, true)
		return
	}
	if lv.comp >= 0 {
		c := dx_scalar_temp(d)
		for k in 0 ..< 4 {
			dx_ins(&d.code, .Ieq, 0, c, 1, dx_reg(.Temp, u32(lv.comp)), dx_imm(u32(k)))
			old := lv.loc
			old.swz = {u8(k), u8(k), u8(k), u8(k)}
			dx_ins(&d.code, .Movc, 0, lv.loc, u8(1) << uint(k), c, dx_comp(v, 0), old)
		}
		return
	}
	if lv.swz_len > 0 && !is_scalar(t) || lv.swz_len == 1 {
		mask: u8
		src := v
		for k in 0 ..< lv.swz_len {
			p := lv.swz[k]
			mask |= 1 << p
			src.swz[p] = v.swz[0 if is_scalar(t) else k]
		}
		if v.file == .Imm {
			for k in 0 ..< lv.swz_len {
				src.imm[lv.swz[k]] = v.imm[v.swz[0 if is_scalar(t) else k]]
			}
			src.swz = DX_XYZW
		}
		dx_ins(&d.code, .Mov, 0, lv.loc, mask, src)
		return
	}
	dx_store(d, lv.loc, t, v)
}

// ---- expressions -----------------------------------------------------------------------

@(private)
dx_expr :: proc(d: ^Dx, id: Expr_Id) -> Dx_Src {
	u := d.u
	e := u.exprs[id]
	t := ty(u, e.type)
	switch e.kind {
	case .None:
		return dx_imm(0)
	case .Int_Lit, .Uint_Lit:
		return dx_imm(u32(e.value))
	case .Float_Lit:
		return dx_imm_f(f32(lit_f64(e)))
	case .Bool_Lit:
		return dx_imm(0xffffffff if e.value != 0 else 0)
	case .Ident:
		if u.syms[e.sym].storage == .Work_Group_Size {
			ls := d.checked.local_size
			return {file = .Imm, rel = -1, swz = DX_XYZW, imm = {ls.x, ls.y, ls.z, ls.z}}
		}
		return dx_load(d, dx_lvalue(d, id))
	case .Convert:
		return dx_convert(d, dx_expr(d, e.a), expr_type(u, e.a), t)
	case .Buffer_Load:
		p, _ := buf_path(u, id)
		return dx_buffer_read(d, p)
	case .Comma:
		dx_expr(d, e.a)
		return dx_expr(d, e.b)
	case .Length:
		at := expr_type(u, e.a)
		return dx_imm_i(at.array if is_array(at) else (i32(at.cols) if is_matrix(at) else i32(at.rows)))
	case .Unary:
		v := dx_expr(d, e.a)
		#partial switch e.op {
		case .Minus:
			if t.base == .Float {
				return dx_per_reg(d, t, v, proc(d: ^Dx, n: int, a: Dx_Src) -> Dx_Src {
					x := a
					x.neg = true
					return dx_op(d, .Mov, n, x)
				})
			}
			return dx_un(d, .Ineg, int(t.rows), v)
		case .Bang, .Tilde:
			return dx_un(d, .Not, int(t.rows), v)
		}
		return v
	case .Binary:
		#partial switch e.op {
		case .And_And, .Or_Or:
			return dx_short_circuit(d, e)
		}
		base := dx_operands(d, {e.a, e.b})
		defer dx_pop_operands(d, base)
		return dx_binop(d, e.op, d.vals[base], expr_type(u, e.a), d.vals[base + 1], expr_type(u, e.b), t)
	case .Assign:
		lv := dx_lvalue(d, e.a)
		v := dx_expr(d, e.b)
		if e.op != .Eq {
			lt := expr_type(u, e.a)
			v = dx_binop(d, compound_op(e.op), dx_load(d, lv), lt, v, expr_type(u, e.b), lt)
		}
		dx_store_lv(d, lv, v)
		return v
	case .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
		lv := dx_lvalue(d, e.a)
		old := dx_copy(d, dx_load(d, lv), t)
		inc := e.kind == .Pre_Inc || e.kind == .Post_Inc
		one := dx_imm_f(1) if t.base == .Float else dx_imm(1)
		nv := dx_binop(d, .Plus if inc else .Minus, old, t, one, scalar_of(t), t)
		dx_store_lv(d, lv, nv)
		return nv if e.kind == .Pre_Inc || e.kind == .Pre_Dec else old
	case .Ternary:
		return dx_ternary(d, e, t)
	case .Index:
		if addressable(u, e.a) {
			return dx_load(d, dx_lvalue(d, id))
		}
		if p, is_buf := buf_path(u, id); is_buf {
			return dx_buffer_read(d, p)
		}
		bt := expr_type(u, e.a)
		base := dx_expr(d, e.a)
		ix := u.exprs[e.b]
		lit := ix.kind == .Int_Lit || ix.kind == .Uint_Lit
		if is_vector(bt) {
			if lit {
				return dx_comp(base, clamp(int(ix.value), 0, int(bt.rows) - 1))
			}
			lv := Dx_LV{loc = base, type = t, swz = base.swz, swz_len = int(bt.rows), comp = -1}
			if base.file == .Imm {
				lv.loc = dx_copy(d, base, bt)
				lv.swz = DX_XYZW
			}
			i := dx_expr(d, e.b)
			c := dx_scalar_temp(d)
			dx_ins(&d.code, .Imax, 0, c, 1, i, dx_imm(0))
			dx_ins(&d.code, .Imin, 0, c, 1, c, dx_imm_i(i32(bt.rows) - 1))
			r := dx_comp(dx_temp(d), 0)
			dx_ins(&d.code, .Mov, 0, r, 1, dx_comp(lv.loc, 0))
			k := dx_scalar_temp(d)
			for j in 1 ..< int(bt.rows) {
				dx_ins(&d.code, .Ieq, 0, k, 1, c, dx_imm(u32(j)))
				dx_ins(&d.code, .Movc, 0, r, 1, k, dx_comp(lv.loc, j), r)
			}
			return r
		}
		n := int(bt.array) if is_array(bt) else int(bt.cols)
		size := dx_regs(u, t)
		if lit {
			return dx_at(base, clamp(int(ix.value), 0, n - 1) * size)
		}
		if base.file != .Indexable && base.file != .Cbuffer {
			base = dx_store_indexable(d, base, bt)
		}
		r := dx_index(d, e.b, n, size, base.rel, base.rel_c)
		return dx_offset_by(d, base, r)
	case .Field:
		if addressable(u, e.a) {
			return dx_load(d, dx_lvalue(d, id))
		}
		if p, is_buf := buf_path(u, id); is_buf {
			return dx_buffer_read(d, p)
		}
		base := dx_expr(d, e.a)
		bt := expr_type(u, e.a)
		if e.swizzle_len > 0 {
			if is_scalar(bt) {
				return base
			}
			return dx_swizzle(base, e.swizzle, int(e.swizzle_len))
		}
		return dx_at(base, dx_field_reg(u, bt.strukt, e.field))
	case .Call:
		switch e.call {
		case .User:
			return dx_user_call(d, id)
		case .Builtin:
			return dx_builtin(d, id)
		case .Constructor:
			return dx_constructor(d, id)
		case .None:
		}
	}
	return dx_imm(0)
}

// An aggregate copied into a fresh indexable temp (for dynamic indexing).
@(private)
dx_store_indexable :: proc(d: ^Dx, v: Dx_Src, t: Type) -> Dx_Src {
	r := dx_alloc(d, t, true)
	dx_store(d, r, t, v)
	return r
}

// `f` over each register (matrix column) of a value of type `t`.
@(private)
dx_per_reg :: proc(d: ^Dx, t: Type, v: Dx_Src, f: proc(d: ^Dx, n: int, a: Dx_Src) -> Dx_Src) -> Dx_Src {
	if !is_matrix(t) {
		return f(d, int(t.rows), dx_n(v, int(t.rows)))
	}
	r := dx_temps(d, int(t.cols))
	for c in 0 ..< int(t.cols) {
		dx_store(d, dx_at(r, c), column_type(t), f(d, int(t.rows), dx_at(v, c)))
	}
	return r
}

@(private)
dx_short_circuit :: proc(d: ^Dx, e: Expr) -> Dx_Src {
	if !may_write(d.u, e.b) && !dx_risky(d.u, e.b) {
		base := dx_operands(d, {e.a, e.b})
		defer dx_pop_operands(d, base)
		return dx_op(d, .And if e.op == .And_And else .Or, 1, d.vals[base], d.vals[base + 1])
	}
	r := dx_comp(dx_temp(d), 0)
	dx_ins(&d.code, .Mov, 0, r, 1, dx_expr(d, e.a))
	dx_flow(&d.code, .If, DX_NZ if e.op == .And_And else 0, r)
	m := dx_mark(d)
	dx_ins(&d.code, .Mov, 0, r, 1, dx_expr(d, e.b))
	dx_release(d, m)
	dx_flow(&d.code, .End_If)
	return r
}

// True when `id` calls a function or samples a texture: a ternary or a
// && or || then branches rather than evaluating both sides (any other
// operand is safe to evaluate: run-time indexes are clamped and buffer
// reads past the end give 0).
@(private)
dx_risky :: proc(u: ^Unit, id: Expr_Id) -> bool {
	if id == 0 {
		return false
	}
	e := u.exprs[id]
	if e.kind == .Call && (e.call == .User || e.call == .Builtin && (e.builtin >= .Texture_Lod && e.builtin <= .Texture_Grad || e.builtin == .Texel_Fetch)) {
		return true
	}
	if dx_risky(u, e.a) || dx_risky(u, e.b) || dx_risky(u, e.c) {
		return true
	}
	for a in list(u, e.args) {
		if dx_risky(u, Expr_Id(a)) {
			return true
		}
	}
	return false
}

@(private)
dx_ternary :: proc(d: ^Dx, e: Expr, t: Type) -> Dx_Src {
	u := d.u
	if t != T_VOID && !may_write(u, e.b) && !may_write(u, e.c) && !dx_risky(u, e.b) && !dx_risky(u, e.c) {
		base := dx_operands(d, {e.a, e.b, e.c})
		defer dx_pop_operands(d, base)
		v := [3]Dx_Src{d.vals[base], d.vals[base + 1], d.vals[base + 2]}
		if is_sv(t) {
			return dx_op(d, .Movc, int(t.rows), dx_comp(v[0], 0), dx_n(v[1], int(t.rows)), dx_n(v[2], int(t.rows)))
		}
		r := dx_alloc(d, t)
		for k in 0 ..< dx_regs(u, t) {
			dx_ins(&d.code, .Movc, 0, dx_at(r, k), 0xf, dx_comp(v[0], 0), dx_at(v[1], k), dx_at(v[2], k))
		}
		return r
	}
	r := dx_alloc(d, t) if t != T_VOID else dx_null()
	cond := dx_expr(d, e.a)
	dx_flow(&d.code, .If, DX_NZ, cond)
	m := dx_mark(d)
	a := dx_expr(d, e.b)
	if t != T_VOID {
		dx_store(d, r, t, a)
	}
	dx_release(d, m)
	dx_flow(&d.code, .Else)
	b := dx_expr(d, e.c)
	if t != T_VOID {
		dx_store(d, r, t, b)
	}
	dx_release(d, m)
	dx_flow(&d.code, .End_If)
	return dx_comp(r, 0) if is_scalar(t) else r
}

// A storage buffer element `p` names (buf_path) as an lvalue: word w =
// element * stride (+ a component chosen at run time, clamped and mapped
// through the swizzle before it) in 32 bits, wrapping as on the CPU, then
// the path's word offset per access (dx_mem_addr).
@(private)
dx_buf_lvalue :: proc(d: ^Dx, p: Buf_Path) -> Dx_LV {
	u := d.u
	base := dx_scalar_temp(d)
	dx_ins2(&d.code, .Imul, dx_null(), base, 1, dx_comp(dx_expr(d, p.elem_idx), 0), dx_imm(u32(p.stride_words)))
	if p.dyn_comp != 0 {
		c := dx_scalar_temp(d)
		dx_ins(&d.code, .Imax, 0, c, 1, dx_comp(dx_expr(d, p.dyn_comp), 0), dx_imm(0))
		dx_ins(&d.code, .Imin, 0, c, 1, c, dx_imm_i(i32(p.dyn_rows - 1)))
		if p.dyn_swz_len > 0 {
			mapped := dx_scalar_temp(d)
			k := dx_scalar_temp(d)
			dx_ins(&d.code, .Mov, 0, mapped, 1, dx_imm(u32(p.dyn_swz[0])))
			for j in 1 ..< p.dyn_rows {
				dx_ins(&d.code, .Ieq, 0, k, 1, c, dx_imm(u32(j)))
				dx_ins(&d.code, .Movc, 0, mapped, 1, k, dx_imm(u32(p.dyn_swz[j])), mapped)
			}
			c = mapped
		}
		dx_ins(&d.code, .Iadd, 0, base, 1, base, c)
	}
	res := d.buf_res[u.syms[p.sym].index]
	return {loc = res, type = p.type, swz = p.swz, swz_len = p.swz_len, comp = -1, word = u32(p.word_off), dyn = i32(base.idx[0])}
}

// A storage buffer read, of only the words `p` names (as the CPU reads
// them).
@(private)
dx_buffer_read :: proc(d: ^Dx, p: Buf_Path) -> Dx_Src {
	return dx_load(d, dx_buf_lvalue(d, p))
}

// A value in memory (dx_is_mem) into fresh registers: only the words it
// names, a vector's swizzle as the words up to the last one named.
@(private)
dx_mem_load :: proc(d: ^Dx, lv: Dx_LV) -> Dx_Src {
	t := lv.type
	if t.base == .Struct || is_array(t) || is_matrix(t) {
		r := dx_alloc(d, t)
		dx_mem_walk(d, lv, lv.word, t, r, false)
		return r
	}
	r := dx_temp(d)
	if lv.swz_len == 1 {
		dx_mem_leaf(d, lv, lv.word + u32(lv.swz[0]), 1, r, false)
		return dx_comp(r, 0)
	}
	if lv.swz_len > 0 {
		last := 0
		for k in 0 ..< lv.swz_len {
			last = max(last, int(lv.swz[k]))
		}
		dx_mem_leaf(d, lv, lv.word, last + 1, r, false)
		return dx_swizzle(r, lv.swz, lv.swz_len)
	}
	dx_mem_leaf(d, lv, lv.word, int(t.rows), r, false)
	return dx_scalar_view(r, int(t.rows))
}

// A value of type `t` at `word` of memory `lv` from (`store`) or into the
// registers `regs`: each scalar, vector or matrix column one access, at its
// offset (a buffer's std430 offsets, shared memory's packed words), arrays
// by element and structs by field, recursively.
@(private)
dx_mem_walk :: proc(d: ^Dx, lv: Dx_LV, word: u32, t: Type, regs: Dx_Src, store: bool) {
	u := d.u
	switch {
	case is_array(t):
		et := elem_type(t)
		words, n := u32(component_count(u, et)), dx_regs(u, et)
		for i in 0 ..< int(t.array) {
			dx_mem_walk(d, lv, word + u32(i) * words, et, dx_at(regs, i * n), store)
		}
	case t.base == .Struct:
		reg := 0
		for f, i in struct_fields(u, t.strukt) {
			ft := ty(u, f.type)
			off := std430_field_offset(u, t.strukt, i32(i)) / 4 if lv.loc.file != .Tgsm else dx_field_words(u, t.strukt, i32(i))
			dx_mem_walk(d, lv, word + u32(off), ft, dx_at(regs, reg), store)
			reg += dx_regs(u, ft)
		}
	case is_matrix(t):
		for c in 0 ..< int(t.cols) {
			dx_mem_leaf(d, lv, word + u32(c) * u32(t.rows), int(t.rows), dx_at(regs, c), store)
		}
	case:
		dx_mem_leaf(d, lv, word, int(t.rows), dx_n(regs, int(t.rows)) if store else regs, store)
	}
}

// The byte address of `word` of memory `lv`: shared memory's directly (its
// indexes are clamped into the variable); a buffer's clamped to word
// 0x3ffffffc first, so a word past the buffer stays past it (where ld_raw
// gives 0 and store_raw and the atomics write nothing) and a 4-word access
// at the ceiling does not cross 2^32.
@(private)
dx_mem_addr :: proc(d: ^Dx, lv: Dx_LV, word: u32) -> Dx_Src {
	if lv.loc.file == .Tgsm && lv.dyn < 0 {
		return dx_imm(4 * word)
	}
	a := dx_scalar_temp(d)
	w := dx_comp(dx_reg(.Temp, u32(lv.dyn)), 0)
	if word != 0 {
		dx_ins(&d.code, .Iadd, 0, a, 1, w, dx_imm(word))
		w = a
	}
	if lv.loc.file != .Tgsm {
		dx_ins(&d.code, .Umin, 0, a, 1, w, dx_imm(0x3ffffffc))
		w = a
	}
	dx_ins(&d.code, .Ishl, 0, a, 1, w, dx_imm(2))
	return a
}

// ld_raw of `n` words at `word` of memory `lv` into `reg`, or store_raw of
// `n` words from `reg` (store_raw writes the words inside the buffer of a
// store that straddles its end, as the CPU does).
@(private)
dx_mem_leaf :: proc(d: ^Dx, lv: Dx_LV, word: u32, n: int, reg: Dx_Src, store: bool) {
	assert(n >= 1 && n <= 4)
	a := dx_mem_addr(d, lv, word)
	mask := dx_mask(n)
	start := len(d.code)
	if store {
		append(&d.code, u32(Dx_Op.Store_Raw))
		dx_put_dst(&d.code, lv.loc, mask)
		dx_put_src(&d.code, a, 1)
		dx_put_src(&d.code, reg, mask)
	} else {
		append(&d.code, u32(Dx_Op.Ld_Raw))
		dx_put_dst(&d.code, reg, mask)
		dx_put_src(&d.code, a, 1)
		dx_put_src(&d.code, lv.loc, mask, vector = true)
	}
	dx_end(&d.code, start)
}

// Converts a scalar or vector between bases.
@(private)
dx_convert :: proc(d: ^Dx, v: Dx_Src, from, to: Type) -> Dx_Src {
	if from.base == to.base || is_integer_base(from.base) && is_integer_base(to.base) {
		return v
	}
	n := int(to.rows)
	a := dx_n(v, int(from.rows))
	switch {
	case to.base == .Bool:
		return dx_op(d, .Ne if from.base == .Float else .Ine, n, a, dx_imm(0))
	case from.base == .Bool:
		return dx_op(d, .And, n, a, dx_imm(0x3f800000 if to.base == .Float else 1))
	case to.base == .Float:
		return dx_op(d, .Itof if from.base == .Int else .Utof, n, a)
	case to.base == .Int:
		return dx_op(d, .Ftoi, n, a)
	case to.base == .Uint:
		return dx_op(d, .Ftou, n, a)
	}
	return v
}

// ---- operators -------------------------------------------------------------------------

// `a op b` with operands of a common base (a shift's amount aside);
// scalars broadcast.
@(private)
dx_binop :: proc(d: ^Dx, op: Token_Kind, a: Dx_Src, ta: Type, b: Dx_Src, tb: Type, result: Type) -> Dx_Src {
	if is_matrix(ta) || is_matrix(tb) {
		return dx_matrix_binop(d, op, a, ta, b, tb, result)
	}
	n := int(max(ta.rows, tb.rows))
	x, y := dx_n(a, int(ta.rows)), dx_n(b, int(tb.rows))
	base := ta.base
	#partial switch op {
	case .Eq_Eq, .Not_Eq:
		eq := op == .Eq_Eq
		cmp := dx_op(d, (.Eq if eq else .Ne) if base == .Float else (.Ieq if eq else .Ine), n, x, y)
		return dx_reduce(d, cmp, n, eq)
	case .Slash, .Percent:
		if base != .Float {
			return dx_int_div(d, op == .Slash, base == .Int, n, x, y)
		}
	}
	ins, swap, neg := dx_binop_ins(op, base)
	if swap {
		x, y = y, x
	}
	if neg {
		y.neg = true
		if y.file == .Imm {
			for &w in y.imm {
				w = transmute(u32)(-transmute(f32)w) if base == .Float else u32(-i32(w))
			}
			y.neg = false
		}
	}
	if ins == .Imul {
		r := dx_temp(d)
		dx_ins2(&d.code, .Imul, dx_null(), r, dx_mask(n), x, y)
		return dx_scalar_view(r, n)
	}
	return dx_op(d, ins, n, x, y)
}

// The instruction for an operator and base: swap the operands for > and
// <=, negate the second for -.
@(private)
dx_binop_ins :: proc "contextless" (op: Token_Kind, base: Base) -> (ins: Dx_Op, swap: bool, neg: bool) {
	f := base == .Float
	s := base == .Int
	#partial switch op {
	case .Plus:
		return .Add if f else .Iadd, false, false
	case .Minus:
		return .Add if f else .Iadd, false, true
	case .Star:
		return .Mul if f else .Imul, false, false
	case .Slash:
		return .Div, false, false
	case .Amp:
		return .And, false, false
	case .Pipe:
		return .Or, false, false
	case .Caret, .Xor_Xor:
		return .Xor, false, false
	case .Shl:
		return .Ishl, false, false
	case .Shr:
		return .Ishr if s else .Ushr, false, false
	case .Lt:
		return .Lt if f else (.Ilt if s else .Ult), false, false
	case .Gt:
		return .Lt if f else (.Ilt if s else .Ult), true, false
	case .Lt_Eq:
		return .Ge if f else (.Ige if s else .Uge), true, false
	case .Gt_Eq:
		return .Ge if f else (.Ige if s else .Uge), false, false
	case .And_And:
		return .And, false, false
	case .Or_Or:
		return .Or, false, false
	}
	return .Mov, false, false
}

// All (`all`) or any of n bool components.
@(private)
dx_reduce :: proc(d: ^Dx, v: Dx_Src, n: int, all: bool) -> Dx_Src {
	if n == 1 {
		return v
	}
	r := dx_comp(dx_temp(d), 0)
	dx_ins(&d.code, .And if all else .Or, 0, r, 1, dx_comp(v, 0), dx_comp(v, 1))
	for k in 2 ..< n {
		dx_ins(&d.code, .And if all else .Or, 0, r, 1, r, dx_comp(v, k))
	}
	return r
}

// Integer / and % as on the CPU: x / 0 and x % 0 are 0, INT_MIN / -1 is
// INT_MIN, x % -1 is 0; signed through the magnitudes (udiv), the
// quotient negated when the signs differ, the remainder taking x's sign.
@(private)
dx_int_div :: proc(d: ^Dx, quotient, signed: bool, n: int, x, y: Dx_Src) -> Dx_Src {
	m := dx_mask(n)
	r := dx_temp(d)
	if !signed {
		if quotient {
			dx_ins2(&d.code, .Udiv, r, dx_null(), m, x, y)
		} else {
			dx_ins2(&d.code, .Udiv, dx_null(), r, m, x, y)
		}
	} else {
		ax, ay := dx_temp(d), dx_temp(d)
		dx_ins(&d.code, .Ineg, 0, ax, m, x)
		dx_ins(&d.code, .Imax, 0, ax, m, x, ax)
		dx_ins(&d.code, .Ineg, 0, ay, m, y)
		dx_ins(&d.code, .Imax, 0, ay, m, y, ay)
		if quotient {
			dx_ins2(&d.code, .Udiv, r, dx_null(), m, ax, ay)
		} else {
			dx_ins2(&d.code, .Udiv, dx_null(), r, m, ax, ay)
		}
		neg := dx_temp(d)
		if quotient {
			dx_ins(&d.code, .Xor, 0, neg, m, x, y)
		} else {
			dx_ins(&d.code, .Mov, 0, neg, m, x)
		}
		dx_ins(&d.code, .Ilt, 0, neg, m, neg, dx_imm(0))
		dx_ins(&d.code, .Ineg, 0, ax, m, r)
		dx_ins(&d.code, .Movc, 0, r, m, neg, ax, r)
	}
	z := dx_temp(d)
	dx_ins(&d.code, .Ieq, 0, z, m, y, dx_imm(0))
	dx_ins(&d.code, .Movc, 0, r, m, z, dx_imm(0), r)
	return dx_scalar_view(r, n)
}

@(private)
dx_matrix_binop :: proc(d: ^Dx, op: Token_Kind, a: Dx_Src, ta: Type, b: Dx_Src, tb: Type, result: Type) -> Dx_Src {
	if op == .Star {
		switch {
		case is_matrix(ta) && is_matrix(tb):
			r := dx_temps(d, int(tb.cols))
			for c in 0 ..< int(tb.cols) {
				col := dx_mat_vec(d, a, ta, dx_at(b, c))
				dx_store(d, dx_at(r, c), column_type(result), col)
			}
			return r
		case is_matrix(ta) && is_vector(tb):
			return dx_mat_vec(d, a, ta, b)
		case is_vector(ta) && is_matrix(tb):
			// Component c is dot(v, column c).
			r := dx_temp(d)
			dps := DX_DOTS
			dp := dps[tb.rows]
			for c in 0 ..< int(tb.cols) {
				dx_dot_into(d, dp, r, u8(1) << uint(c), a, dx_at(b, c))
			}
			return r
		case is_matrix(ta) && is_scalar(tb), is_scalar(ta) && is_matrix(tb):
		}
	}
	mt := ta if is_matrix(ta) else tb
	ct := column_type(mt)
	n := int(mt.rows)
	if op == .Eq_Eq || op == .Not_Eq {
		eq := op == .Eq_Eq
		r: Dx_Src
		for c in 0 ..< int(mt.cols) {
			col := dx_reduce(d, dx_op(d, .Eq if eq else .Ne, n, dx_at(a, c), dx_at(b, c)), n, eq)
			if c == 0 {
				r = col
			} else {
				r = dx_op(d, .And if eq else .Or, 1, r, col)
			}
		}
		return r
	}
	r := dx_temps(d, int(mt.cols))
	for c in 0 ..< int(mt.cols) {
		ca := dx_at(a, c) if is_matrix(ta) else a
		cb := dx_at(b, c) if is_matrix(tb) else b
		col := dx_binop(d, op, ca, ct if is_matrix(ta) else ta, cb, ct if is_matrix(tb) else tb, ct)
		dx_store(d, dx_at(r, c), ct, col)
	}
	return r
}

// m * v: the columns weighted by v's components.
@(private)
dx_mat_vec :: proc(d: ^Dx, m: Dx_Src, mt: Type, v: Dx_Src) -> Dx_Src {
	n := int(mt.rows)
	r := dx_temp(d)
	dx_ins(&d.code, .Mul, 0, r, dx_mask(n), dx_at(m, 0), dx_comp(v, 0))
	for c in 1 ..< int(mt.cols) {
		dx_ins(&d.code, .Mad, 0, r, dx_mask(n), dx_at(m, c), dx_comp(v, c), r)
	}
	return r
}

// A dot product of a and b into component(s) `mask` of `r`.
@(private)
dx_dot_into :: proc(d: ^Dx, dp: Dx_Op, r: Dx_Src, mask: u8, a, b: Dx_Src) {
	start := len(d.code)
	append(&d.code, u32(dp))
	dx_put_dst(&d.code, r, mask)
	dx_put_src(&d.code, a, mask, vector = true)
	dx_put_src(&d.code, b, mask, vector = true)
	dx_end(&d.code, start)
}

// ---- calls -----------------------------------------------------------------------------

// What the body's returns need: a return loop when one returns before the
// end, the flag when one returns from inside a loop or switch.
@(private)
dx_scan_returns :: proc(u: ^Unit, id: Stmt_Id, depth: int, last: bool, early, nested: ^bool) {
	if id == 0 {
		return
	}
	st := u.stmts[id]
	#partial switch st.kind {
	case .Return:
		if !last || depth > 0 {
			early^ = true
		}
		if depth > 0 {
			nested^ = true
		}
	case .Block:
		body := list(u, st.body)
		for b, i in body {
			dx_scan_returns(u, Stmt_Id(b), depth, last && i == len(body) - 1, early, nested)
		}
	case .If:
		dx_scan_returns(u, st.then_, depth, false, early, nested)
		dx_scan_returns(u, st.else_, depth, false, early, nested)
	case .For, .While, .Do_While:
		dx_scan_returns(u, st.then_, depth + 1, false, early, nested)
	case .Switch:
		for b in list(u, st.body) {
			dx_scan_returns(u, Stmt_Id(b), depth + 1, false, early, nested)
		}
	}
}

// A user call, inlined: parameters bound to copies of the arguments (out
// and inout copied back after), the body in a loop its returns break out
// of when one returns early.
@(private)
dx_user_call :: proc(d: ^Dx, id: Expr_Id) -> Dx_Src {
	u := d.u
	f := u.exprs[id].func
	args := args_of(u, id)
	ps := fparams(u, f)
	// The parameters' registers and the out / inout arguments' lvalues, on
	// the stacks from vbase and lbase (by index: the body pushes above).
	vbase, lbase := len(d.vals), len(d.lvs)
	resize(&d.vals, vbase + len(args))
	resize(&d.lvs, lbase + len(args))
	defer {
		resize(&d.vals, vbase)
		resize(&d.lvs, lbase)
	}
	for p, i in ps {
		pt := ty(u, p.type)
		switch {
		case pt.base == .Sampler:
			// The channel (or image) the argument names now, bound with the
			// other parameters once every argument is evaluated (a call in a
			// later one may bind this function's parameters too).
			d.vals[vbase + i] = dx_imm(u32(u8(d.sym_chan[u.exprs[args[i]].sym])))
		case pt.base == .Image:
			d.vals[vbase + i] = dx_imm(u32(u8(d.sym_img[u.exprs[args[i]].sym])))
		case p.qual == .In:
			v := dx_expr(d, args[i])
			loc := dx_alloc(d, pt, p.sym != 0 && d.sym_x[p.sym])
			dx_store(d, loc, pt, v)
			d.vals[vbase + i] = loc
		case:
			lv := dx_lvalue(d, args[i])
			d.lvs[lbase + i] = lv
			loc := dx_alloc(d, pt, p.sym != 0 && d.sym_x[p.sym])
			if p.qual == .Inout {
				dx_store(d, loc, pt, dx_load(d, lv))
			} else {
				dx_zero(d, loc, pt)
			}
			d.vals[vbase + i] = loc
		}
	}
	for p, i in ps {
		if p.sym == 0 {
			continue
		}
		#partial switch ty(u, p.type).base {
		case .Sampler:
			d.sym_chan[p.sym] = i8(d.vals[vbase + i].imm[0])
		case .Image:
			d.sym_img[p.sym] = i8(d.vals[vbase + i].imm[0])
		case:
			d.sym_loc[p.sym] = d.vals[vbase + i]
		}
	}
	ret_t := ty(u, u.funcs[f].ret)
	call := Dx_Call{ret_type = ret_t, flow = -1}
	if ret_t != T_VOID {
		call.ret = dx_alloc(d, ret_t)
	}
	body := u.func_decls[u.funcs[f].decl].body
	early, nested := false, false
	dx_scan_returns(u, body, 0, true, &early, &nested)
	ends_with_return := false
	if b := list(u, u.stmts[body].body); len(b) > 0 {
		ends_with_return = u.stmts[b[len(b) - 1]].kind == .Return
	}
	if ret_t != T_VOID && (early || !ends_with_return) {
		dx_zero(d, call.ret, ret_t)
	}
	if early {
		if nested {
			flag := dx_temp(d)
			call.flag = flag.idx[0]
			dx_ins(&d.code, .Mov, 0, flag, 1, dx_imm(0))
		}
		dx_flow(&d.code, .Loop)
		call.flow = len(d.flow)
		append(&d.flow, Dx_Flow{kind = .Return})
	}
	append(&d.calls, call)
	dx_block(d, body)
	if early {
		if !d.terminated {
			dx_flow(&d.code, .Break)
		}
		dx_flow(&d.code, .End_Loop)
		pop(&d.flow)
	}
	pop(&d.calls)
	d.terminated = false
	for p, i in ps {
		pt := ty(u, p.type)
		if p.qual != .In && pt.base != .Sampler && pt.base != .Image {
			dx_store_lv(d, d.lvs[lbase + i], d.vals[vbase + i])
		}
	}
	return dx_comp(call.ret, 0) if is_scalar(ret_t) else call.ret
}

// Scalar components of a value in order (matrices column by column) into
// `out` from `n^`, as many as fit (a constructor uses at most 16).
@(private)
dx_components :: proc "contextless" (v: Dx_Src, t: Type, out: ^[16]Dx_Src, n: ^int) {
	cols := int(t.cols) if is_matrix(t) else 1
	for c in 0 ..< cols {
		col := dx_at(v, c) if is_matrix(t) else v
		for r in 0 ..< int(t.rows) {
			if n^ < len(out) {
				out[n^] = dx_comp(col, r)
				n^ += 1
			}
		}
	}
}

// Writes scalars `parts` to the components of register `dst`, one mov per
// run of components read from the same register.
@(private)
dx_gather :: proc(d: ^Dx, dst: Dx_Src, parts: []Dx_Src) {
	same :: proc(a, b: Dx_Src) -> bool {
		return a.file == b.file && a.file != .Imm && a.idx == b.idx && a.rel == b.rel && a.rel_c == b.rel_c && a.neg == b.neg && a.abs == b.abs
	}
	done: [4]bool
	for i in 0 ..< len(parts) {
		if done[i] {
			continue
		}
		src := parts[i]
		mask: u8
		for j in i ..< len(parts) {
			if !done[j] && (j == i || same(parts[i], parts[j])) {
				mask |= 1 << uint(j)
				src.swz[j] = parts[j].swz[0]
				src.imm[j] = parts[j].imm[0]
				done[j] = true
			}
		}
		if src.file == .Imm {
			src.swz = DX_XYZW
		}
		dx_ins(&d.code, .Mov, 0, dst, mask, src)
	}
}

@(private)
dx_constructor :: proc(d: ^Dx, id: Expr_Id) -> Dx_Src {
	u := d.u
	t := expr_type(u, id)
	args := args_of(u, id)
	base := dx_operands(d, args)
	defer dx_pop_operands(d, base)
	// Read only until the operand stack next grows (nothing below
	// evaluates an expression).
	vals := d.vals[base:][:len(args)]
	if t.array > 0 || t.base == .Struct {
		r := dx_alloc(d, t)
		reg := 0
		for a, i in args {
			at := expr_type(u, a)
			dx_store(d, dx_at(r, reg), at, vals[i])
			reg += dx_regs(u, at)
		}
		return r
	}
	st := scalar_of(t)
	if len(args) == 1 {
		at := expr_type(u, args[0])
		v := vals[0]
		if is_matrix(t) && is_matrix(at) {
			// Resize: overlapping components, identity elsewhere.
			r := dx_temps(d, int(t.cols))
			for c in 0 ..< int(t.cols) {
				parts: [4]Dx_Src
				for ri in 0 ..< int(t.rows) {
					if c < int(at.cols) && ri < int(at.rows) {
						parts[ri] = dx_comp(dx_at(v, c), ri)
					} else {
						parts[ri] = dx_imm_f(1 if c == ri else 0)
					}
				}
				dx_gather(d, dx_at(r, c), parts[:t.rows])
			}
			return r
		}
		if is_scalar(at) {
			sv := dx_convert(d, v, at, st)
			if is_sv(t) {
				return dx_comp(sv, 0)
			}
			// A diagonal matrix.
			r := dx_temps(d, int(t.cols))
			for c in 0 ..< int(t.cols) {
				parts: [4]Dx_Src
				for ri in 0 ..< int(t.rows) {
					parts[ri] = dx_comp(sv, 0) if c == ri else dx_imm_f(0)
				}
				dx_gather(d, dx_at(r, c), parts[:t.rows])
			}
			return r
		}
		if is_sv(t) && is_sv(at) && (is_scalar(t) || at.rows == t.rows) {
			cv := dx_convert(d, v, at, with_base(at, st.base))
			return dx_comp(cv, 0) if is_scalar(t) else cv
		}
	}
	// Every argument's components in order, converted (a matrix column
	// by column).
	flat: [16]Dx_Src
	nflat := 0
	for a, i in args {
		at := expr_type(u, a)
		v := vals[i]
		if is_sv(at) {
			v = dx_convert(d, v, at, with_base(at, st.base))
		} else if st.base != .Float {
			ct := column_type(at)
			conv := dx_temps(d, int(at.cols))
			for c in 0 ..< int(at.cols) {
				dx_store(d, dx_at(conv, c), with_base(ct, st.base), dx_convert(d, dx_at(v, c), ct, with_base(ct, st.base)))
			}
			v = conv
		}
		dx_components(v, at, &flat, &nflat)
	}
	if is_sv(t) {
		r := dx_temp(d)
		dx_gather(d, r, flat[:t.rows])
		return r
	}
	r := dx_temps(d, int(t.cols))
	for c in 0 ..< int(t.cols) {
		dx_gather(d, dx_at(r, c), flat[c * int(t.rows):][:t.rows])
	}
	return r
}

// ---- built-ins -------------------------------------------------------------------------

@(private)
DX_LOG2E :: 1.4426950408889634
@(private)
DX_LN2 :: 0.6931471805599453

@(private)
dx_builtin :: proc(d: ^Dx, id: Expr_Id) -> Dx_Src {
	u := d.u
	e := u.exprs[id]
	b := e.builtin
	t := ty(u, e.type)
	n := int(t.rows)
	argx := args_of(u, id)
	if b >= .Texture_Lod && b <= .Texture_Grad || b == .Texel_Fetch {
		return dx_texture(d, id)
	}
	if is_barrier_builtin(b) {
		// Graphics stages read their buffers only: nothing to order.
		if u.stage == .Compute {
			dx_flow(&d.code, .Sync, DX_SYNC_FLAGS[b])
		}
		return dx_imm(0)
	}
	if is_atomic_builtin(b) {
		return dx_atomic(d, id)
	}
	#partial switch b {
	case .Image_Load, .Image_Store, .Image_Size:
		return dx_image(d, id)
	}
	assert(len(argx) <= 4)
	types_buf: [4]Type
	types := types_buf[:len(argx)]
	for a, i in argx {
		types[i] = expr_type(u, a)
	}
	base := dx_operands(d, argx)
	defer dx_pop_operands(d, base)
	// Read only until the operand stack next grows (the lowering below
	// evaluates no expression).
	args := d.vals[base:][:len(argx)]
	at := types[0]
	an := int(at.rows)
	x := dx_n(args[0], an)
	arg :: proc(args: []Dx_Src, types: []Type, i, n: int) -> Dx_Src {
		return dx_n(args[i], int(types[i].rows))
	}
	#partial switch b {
	case .Radians:
		return dx_op(d, .Mul, n, x, dx_imm_f(math.PI / 180))
	case .Degrees:
		return dx_op(d, .Mul, n, x, dx_imm_f(180 / math.PI))
	case .Sin, .Cos:
		r := dx_temp(d)
		if b == .Sin {
			dx_ins2(&d.code, .Sincos, r, dx_null(), dx_mask(n), x)
		} else {
			dx_ins2(&d.code, .Sincos, dx_null(), r, dx_mask(n), x)
		}
		return dx_scalar_view(r, n)
	case .Tan:
		s, c := dx_temp(d), dx_temp(d)
		dx_ins2(&d.code, .Sincos, s, c, dx_mask(n), x)
		return dx_op(d, .Div, n, s, c)
	case .Asin, .Acos:
		return dx_asin_acos(d, n, x, b == .Asin)
	case .Atan:
		return dx_atan(d, n, x)
	case .Atan2:
		return dx_atan2(d, n, x, arg(args, types, 1, n))
	case .Sinh, .Cosh:
		// (e^|x| +- e^-|x|) / 2 from e = e^|x| >= 1 and 1 / e, with x's
		// sign for sinh: e^-|x| itself would be a denormal, which D3D
		// flushes to 0, where 1 / e is still fine (cosh(-88)); sinh is x
		// itself near 0, where the difference cancels to 0.
		ax := x
		ax.abs = true
		ex := dx_exp(d, n, ax)
		inv := dx_op(d, .Rcp, n, ex)
		inv.neg = b == .Sinh
		h := dx_op(d, .Mul, n, dx_op(d, .Add, n, ex, inv), dx_imm_f(0.5))
		return dx_near_zero(d, n, x, dx_copysign(d, n, h, x)) if b == .Sinh else h
	case .Tanh:
		// 1 - 2 / (e^2x + 1), which saturates instead of inf / inf; x
		// itself near 0.
		e2 := dx_exp(d, n, dx_op(d, .Add, n, x, x))
		den := dx_op(d, .Add, n, e2, dx_imm_f(1))
		q := dx_op(d, .Div, n, dx_imm_f(2), den)
		q.neg = true
		return dx_near_zero(d, n, x, dx_op(d, .Add, n, dx_imm_f(1), q))
	case .Asinh:
		// ln(|x| + sqrt(x^2 + 1)) with x's sign; from 2^16, where the
		// square root is |x| in f32 and x^2 may overflow, ln |x| + ln 2;
		// x itself near 0.
		ax := x
		ax.abs = true
		s := dx_op(d, .Sqrt, n, dx_op(d, .Mad, n, x, x, dx_imm_f(1)))
		l := dx_ln_big(d, n, ax, dx_op(d, .Add, n, ax, s))
		return dx_near_zero(d, n, x, dx_copysign(d, n, l, x))
	case .Acosh:
		// ln(x + sqrt(x^2 - 1)); from 2^16, ln x + ln 2.
		s := dx_op(d, .Sqrt, n, dx_op(d, .Mad, n, x, x, dx_imm_f(-1)))
		return dx_ln_big(d, n, x, dx_op(d, .Add, n, x, s))
	case .Atanh:
		// ln((1 + x) / (1 - x)) / 2; x itself near 0.
		mx := x
		mx.neg = true
		num := dx_op(d, .Add, n, dx_imm_f(1), x)
		den := dx_op(d, .Add, n, dx_imm_f(1), mx)
		l := dx_ln(d, n, dx_op(d, .Div, n, num, den))
		return dx_near_zero(d, n, x, dx_op(d, .Mul, n, l, dx_imm_f(0.5)))
	case .Pow:
		if pow_is_square(u, argx) {
			return dx_op(d, .Mul, n, x, x)
		}
		l := dx_op(d, .Log, n, x)
		return dx_op(d, .Exp, n, dx_op(d, .Mul, n, l, arg(args, types, 1, n)))
	case .Exp:
		return dx_exp(d, n, x)
	case .Log:
		return dx_ln(d, n, x)
	case .Exp2:
		return dx_op(d, .Exp, n, x)
	case .Log2:
		return dx_op(d, .Log, n, x)
	case .Sqrt:
		return dx_op(d, .Sqrt, n, x)
	case .Inversesqrt:
		return dx_op(d, .Rsq, n, x)
	case .Floor:
		return dx_op(d, .Round_Ni, n, x)
	case .Ceil:
		return dx_op(d, .Round_Pi, n, x)
	case .Fract:
		return dx_op(d, .Frc, n, x)
	case .Trunc:
		return dx_op(d, .Round_Z, n, x)
	case .Round_Even:
		return dx_op(d, .Round_Ne, n, x)
	case .Round:
		// Halves away from zero, exactly: trunc(x) plus sign(f) where
		// |f| >= 0.5, f the fraction.
		tr := dx_op(d, .Round_Z, n, x)
		ntr := tr
		ntr.neg = true
		f := dx_op(d, .Add, n, x, ntr)
		af := f
		af.abs = true
		half := dx_op(d, .Ge, n, af, dx_imm_f(0.5))
		neg := dx_op(d, .Lt, n, f, dx_imm_f(0))
		s := dx_op(d, .Movc, n, neg, dx_imm_f(-1), dx_imm_f(1))
		dx_ins(&d.code, .And, 0, s, dx_mask(n), s, half)
		return dx_op(d, .Add, n, tr, s)
	case .Abs:
		if at.base == .Float {
			ax := x
			ax.abs = true
			return dx_op(d, .Mov, n, ax)
		}
		ng := dx_op(d, .Ineg, n, x)
		return dx_op(d, .Imax, n, x, ng)
	case .Sign:
		if at.base == .Float {
			p := dx_op(d, .Lt, n, dx_imm_f(0), x)
			m := dx_op(d, .Lt, n, x, dx_imm_f(0))
			dx_ins(&d.code, .And, 0, p, dx_mask(n), p, dx_imm(0x3f800000))
			dx_ins(&d.code, .And, 0, m, dx_mask(n), m, dx_imm(0x3f800000))
			m.neg = true
			return dx_op(d, .Add, n, p, m)
		}
		p := dx_op(d, .Ilt, n, dx_imm(0), x)
		m := dx_op(d, .Ilt, n, x, dx_imm(0))
		p.neg = true
		return dx_op(d, .Iadd, n, m, p)
	case .Mod:
		y := arg(args, types, 1, n)
		q := dx_op(d, .Round_Ni, n, dx_op(d, .Div, n, x, y))
		m := dx_op(d, .Mul, n, y, q)
		m.neg = true
		return dx_op(d, .Add, n, x, m)
	case .Min, .Max, .Clamp:
		lo, hi := Dx_Op.Min, Dx_Op.Max
		#partial switch at.base {
		case .Int:
			lo, hi = .Imin, .Imax
		case .Uint:
			lo, hi = .Umin, .Umax
		}
		#partial switch b {
		case .Min:
			return dx_op(d, lo, n, x, arg(args, types, 1, n))
		case .Max:
			return dx_op(d, hi, n, x, arg(args, types, 1, n))
		}
		return dx_op(d, lo, n, dx_op(d, hi, n, x, arg(args, types, 1, n)), arg(args, types, 2, n))
	case .Mix:
		// x * (1 - a) + y * a, as the CPU computes it.
		a := arg(args, types, 2, n)
		na := a
		na.neg = true
		one_a := dx_op(d, .Add, n, dx_imm_f(1), na)
		xa := dx_op(d, .Mul, n, x, one_a)
		ya := dx_op(d, .Mul, n, arg(args, types, 1, n), a)
		return dx_op(d, .Add, n, xa, ya)
	case .Mix_Bool:
		return dx_op(d, .Movc, n, arg(args, types, 2, n), arg(args, types, 1, n), x)
	case .Step:
		y := arg(args, types, 1, n)
		r := dx_op(d, .Ge, n, y, x)
		dx_ins(&d.code, .And, 0, r, dx_mask(n), r, dx_imm(0x3f800000))
		return r
	case .Smoothstep:
		e1 := arg(args, types, 1, n)
		v := arg(args, types, 2, n)
		ne0 := x
		ne0.neg = true
		num := dx_op(d, .Add, n, v, ne0)
		den := dx_op(d, .Add, n, e1, ne0)
		tt := dx_temp(d)
		dx_ins(&d.code, .Div, DX_SATURATE, tt, dx_mask(n), num, den)
		k := dx_op(d, .Mad, n, tt, dx_imm_f(-2), dx_imm_f(3))
		return dx_op(d, .Mul, n, dx_op(d, .Mul, n, tt, tt), k)
	case .Isnan:
		return dx_op(d, .Ne, n, x, x)
	case .Isinf:
		ax := x
		ax.abs = true
		return dx_op(d, .Eq, n, ax, dx_imm(0x7f800000))
	case .Length:
		return dx_op(d, .Sqrt, 1, dx_dot(d, args[0], at, args[0]))
	case .Distance:
		y := args[1]
		y.neg = true
		diff := dx_op(d, .Add, an, dx_n(args[0], an), dx_n(y, an))
		return dx_op(d, .Sqrt, 1, dx_dot(d, diff, at, diff))
	case .Dot:
		return dx_dot(d, args[0], at, args[1])
	case .Cross:
		// a.yzx * b.zxy - a.zxy * b.yzx
		a, c := args[0], args[1]
		r := dx_op(d, .Mul, 3, dx_swizzle(a, {2, 0, 1, 1}, 3), dx_swizzle(c, {1, 2, 0, 0}, 3))
		r.neg = true
		return dx_op(d, .Mad, 3, dx_swizzle(a, {1, 2, 0, 0}, 3), dx_swizzle(c, {2, 0, 1, 1}, 3), r)
	case .Normalize:
		l := dx_op(d, .Rsq, 1, dx_dot(d, args[0], at, args[0]))
		return dx_op(d, .Mul, n, x, dx_comp(l, 0))
	case .Faceforward:
		// dot(Nref, I) < 0 ? N : -N
		dt := dx_dot(d, args[2], at, args[1])
		lt := dx_op(d, .Lt, 1, dt, dx_imm_f(0))
		nx := x
		nx.neg = true
		return dx_op(d, .Movc, n, dx_comp(lt, 0), x, nx)
	case .Reflect:
		// I - 2 * dot(N, I) * N
		nn := args[1]
		dt := dx_dot(d, nn, at, x)
		k := dx_op(d, .Mul, 1, dt, dx_imm_f(2))
		kn := dx_op(d, .Mul, n, dx_n(nn, n), dx_comp(k, 0))
		kn.neg = true
		return dx_op(d, .Add, n, x, kn)
	case .Refract:
		// k = 1 - eta^2 (1 - dot(N, I)^2); k < 0 ? 0 : eta I - (eta dot + sqrt(k)) N
		nn := dx_n(args[1], n)
		eta := dx_comp(args[2], 0)
		dt := dx_dot(d, args[1], at, x)
		one := dx_op(d, .Mad, 1, dt, dx_comp(dt, 0), dx_imm_f(-1))
		ee := dx_op(d, .Mul, 1, eta, eta)
		k := dx_op(d, .Mad, 1, ee, one, dx_imm_f(1))
		s := dx_op(d, .Sqrt, 1, k)
		c := dx_op(d, .Mad, 1, eta, dt, s)
		cn := dx_op(d, .Mul, n, nn, dx_comp(c, 0))
		cn.neg = true
		r := dx_op(d, .Mad, n, x, eta, cn)
		z := dx_op(d, .Lt, 1, k, dx_imm_f(0))
		return dx_op(d, .Movc, n, dx_comp(z, 0), dx_imm_f(0), r)
	case .Matrix_Comp_Mult:
		r := dx_temps(d, int(t.cols))
		for c in 0 ..< int(t.cols) {
			dx_ins(&d.code, .Mul, 0, dx_at(r, c), dx_mask(n), dx_at(args[0], c), dx_at(args[1], c))
		}
		return r
	case .Outer_Product:
		// Column c is a * b[c].
		r := dx_temps(d, int(t.cols))
		for c in 0 ..< int(t.cols) {
			dx_ins(&d.code, .Mul, 0, dx_at(r, c), dx_mask(n), dx_n(args[0], an), dx_comp(args[1], c))
		}
		return r
	case .Transpose:
		r := dx_temps(d, int(t.cols))
		for c in 0 ..< int(t.cols) {
			parts: [4]Dx_Src
			for ri in 0 ..< int(t.rows) {
				parts[ri] = dx_comp(dx_at(args[0], ri), c)
			}
			dx_gather(d, dx_at(r, c), parts[:t.rows])
		}
		return r
	case .Determinant:
		return dx_determinant(d, args[0], int(at.cols))
	case .Inverse:
		return dx_inverse(d, args[0], int(at.cols))
	case .Less_Than, .Less_Than_Equal, .Greater_Than, .Greater_Than_Equal, .Equal, .Not_Equal:
		ops := [?]Token_Kind{.Lt, .Lt_Eq, .Gt, .Gt_Eq, .Eq_Eq, .Not_Eq}
		op := ops[int(b) - int(Builtin.Less_Than)]
		y := arg(args, types, 1, n)
		if op == .Eq_Eq || op == .Not_Eq {
			eq := op == .Eq_Eq
			return dx_op(d, (.Eq if eq else .Ne) if at.base == .Float else (.Ieq if eq else .Ine), n, x, y)
		}
		ins, swap, _ := dx_binop_ins(op, at.base)
		if swap {
			return dx_op(d, ins, n, y, x)
		}
		return dx_op(d, ins, n, x, y)
	case .Any, .All:
		return dx_reduce(d, args[0], an, b == .All)
	case .Not:
		return dx_op(d, .Not, n, x)
	case .Dfdx:
		return dx_op(d, .Deriv_Rtx_Fine, n, x)
	case .Dfdy:
		return dx_op(d, .Deriv_Rty_Fine, n, x)
	case .Fwidth:
		dx := dx_op(d, .Deriv_Rtx_Fine, n, x)
		dy := dx_op(d, .Deriv_Rty_Fine, n, x)
		dx.abs, dy.abs = true, true
		return dx_op(d, .Add, n, dx, dy)
	case .Float_Bits_To_Int, .Float_Bits_To_Uint, .Int_Bits_To_Float, .Uint_Bits_To_Float:
		return args[0]
	}
	return dx_imm(0)
}

// e^x and ln x through the hardware's base 2.
@(private)
dx_exp :: proc(d: ^Dx, n: int, x: Dx_Src) -> Dx_Src {
	return dx_op(d, .Exp, n, dx_op(d, .Mul, n, x, dx_imm_f(DX_LOG2E)))
}

@(private)
dx_ln :: proc(d: ^Dx, n: int, x: Dx_Src) -> Dx_Src {
	return dx_op(d, .Mul, n, dx_op(d, .Log, n, x), dx_imm_f(DX_LN2))
}

// ln(arg), or ln(x) + ln 2 where x >= 2^16 (asinh's and acosh's large
// arguments, where arg would be 2x or overflow).
@(private)
dx_ln_big :: proc(d: ^Dx, n: int, x, arg: Dx_Src) -> Dx_Src {
	big := dx_op(d, .Ge, n, x, dx_imm_f(65536))
	l := dx_op(d, .Log, n, dx_op(d, .Movc, n, big, x, arg))
	one := dx_op(d, .And, n, big, dx_imm_f(1))
	return dx_op(d, .Mul, n, dx_op(d, .Add, n, l, one), dx_imm_f(DX_LN2))
}

// `mag` (+0 or more, or NaN) with the sign bit of `x`.
@(private)
dx_copysign :: proc(d: ^Dx, n: int, mag, x: Dx_Src) -> Dx_Src {
	assert(!x.neg && !x.abs)
	s := dx_op(d, .And, n, x, dx_imm(0x80000000))
	return dx_op(d, .Or, n, mag, s)
}

// x itself where |x| < 1e-4 (sinh, tanh, asinh, atanh: their cubic term is
// below half an ulp there, and the formulas lose x to the 1 they add), else
// `r`.
@(private)
dx_near_zero :: proc(d: ^Dx, n: int, x, r: Dx_Src) -> Dx_Src {
	ax := x
	ax.abs = true
	small := dx_op(d, .Lt, n, ax, dx_imm_f(1e-4))
	return dx_op(d, .Movc, n, small, x, r)
}

@(private)
DX_DOTS :: [5]Dx_Op{.Mov, .Mov, .Dp2, .Dp3, .Dp4}

// dot(a, b) of vectors of type t (a product for scalars), as a temp's x.
@(private)
dx_dot :: proc(d: ^Dx, a: Dx_Src, t: Type, b: Dx_Src) -> Dx_Src {
	if is_scalar(t) {
		return dx_op(d, .Mul, 1, a, b)
	}
	r := dx_comp(dx_temp(d), 0)
	dps := DX_DOTS
	dx_dot_into(d, dps[t.rows], r, 1, a, b)
	return r
}

// Horner: c[0] + x * (c[1] + x * (...)), n components.
@(private)
dx_poly :: proc(d: ^Dx, n: int, x: Dx_Src, c: []f32) -> Dx_Src {
	r := dx_op(d, .Mad, n, x, dx_imm_f(c[len(c) - 1]), dx_imm_f(c[len(c) - 2]))
	for i := len(c) - 3; i >= 0; i -= 1 {
		dx_ins(&d.code, .Mad, 0, r, dx_mask(n), r, x, dx_imm_f(c[i]))
	}
	return r
}

// Abramowitz and Stegun 4.4.46: acos(x) = sqrt(1 - x) p(x) on [0, 1]
// (|error| <= 2e-8), reflected for x < 0. asin is pi/2 - acos(|x|) from
// |x| = 0.5, and atan(|x| / sqrt(1 - x^2)) below (pi/2 - acos would cancel
// to nothing near 0), with x's sign.
@(private)
dx_asin_acos :: proc(d: ^Dx, n: int, x: Dx_Src, asin: bool) -> Dx_Src {
	ax := x
	ax.abs = true
	p := dx_poly(d, n, ax, {1.5707963050, -0.2145988016, 0.0889789874, -0.0501743046, 0.0308918810, -0.0170881256, 0.0066700901, -0.0012624911})
	nax := ax
	nax.neg = true
	s := dx_op(d, .Sqrt, n, dx_op(d, .Add, n, dx_imm_f(1), nax))
	r := dx_op(d, .Mul, n, s, p)
	nr := r
	nr.neg = true
	if asin {
		big := dx_op(d, .Add, n, dx_imm_f(math.PI / 2), nr)
		nx := x
		nx.neg = true
		c := dx_op(d, .Sqrt, n, dx_op(d, .Mad, n, nx, x, dx_imm_f(1)))
		small := dx_atan01(d, n, dx_op(d, .Div, n, ax, c))
		half := dx_op(d, .Ge, n, ax, dx_imm_f(0.5))
		return dx_copysign(d, n, dx_op(d, .Movc, n, half, big, small), x)
	}
	neg := dx_op(d, .Lt, n, x, dx_imm_f(0))
	flip := dx_op(d, .Add, n, dx_imm_f(math.PI), nr)
	return dx_op(d, .Movc, n, neg, flip, r)
}

// atan(t) for t in [0, 1]: Abramowitz and Stegun 4.4.49 (|error| <= 2e-8).
@(private)
dx_atan01 :: proc(d: ^Dx, n: int, t: Dx_Src) -> Dx_Src {
	t2 := dx_op(d, .Mul, n, t, t)
	p := dx_poly(d, n, t2, {1, -0.3333314528, 0.1999355085, -0.1420889944, 0.1065626393, -0.0752896400, 0.0429096138, -0.0161657367, 0.0028662257})
	return dx_op(d, .Mul, n, t, p)
}

// pi/2 - r where `cond`, else r.
@(private)
dx_reflect_angle :: proc(d: ^Dx, n: int, cond, r: Dx_Src, angle: f32) -> Dx_Src {
	nr := r
	nr.neg = true
	a := dx_op(d, .Add, n, dx_imm_f(angle), nr)
	return dx_op(d, .Movc, n, cond, a, r)
}

// atan(t) of |x| <= 1, pi/2 - atan(1 / |x|) above (1 / inf is 0), with
// x's sign; NaN stays NaN (no min or max, which would drop it).
@(private)
dx_atan :: proc(d: ^Dx, n: int, x: Dx_Src) -> Dx_Src {
	ax := x
	ax.abs = true
	big := dx_op(d, .Lt, n, dx_imm_f(1), ax)
	t := dx_op(d, .Movc, n, big, dx_op(d, .Rcp, n, ax), ax)
	r := dx_reflect_angle(d, n, big, dx_atan01(d, n, t), math.PI / 2)
	return dx_copysign(d, n, r, x)
}

// atan(y, x): the angle of the smaller over the larger magnitude (1 where
// they are equal, so infinities give the diagonals; 0 at the origin), then
// reflected into its octant, with y's sign; NaN if either is.
@(private)
dx_atan2 :: proc(d: ^Dx, n: int, y, x: Dx_Src) -> Dx_Src {
	ax, ay := x, y
	ax.abs, ay.abs = true, true
	lo := dx_op(d, .Min, n, ax, ay)
	hi := dx_op(d, .Max, n, ax, ay)
	q := dx_op(d, .Div, n, lo, hi)
	same := dx_op(d, .Eq, n, ax, ay)
	dx_ins(&d.code, .Movc, 0, q, dx_mask(n), same, dx_imm_f(1), q)
	z := dx_op(d, .Eq, n, hi, dx_imm_f(0))
	dx_ins(&d.code, .Movc, 0, q, dx_mask(n), z, dx_imm_f(0), q)
	r := dx_atan01(d, n, q)
	r = dx_reflect_angle(d, n, dx_op(d, .Lt, n, ax, ay), r, math.PI / 2)
	r = dx_reflect_angle(d, n, dx_op(d, .Lt, n, x, dx_imm_f(0)), r, math.PI)
	r = dx_copysign(d, n, r, y)
	nan := dx_op(d, .Or, n, dx_op(d, .Ne, n, x, x), dx_op(d, .Ne, n, y, y))
	return dx_op(d, .Movc, n, nan, dx_imm(0x7fc00000), r)
}

// Element (column c, row r) of an n x n matrix.
@(private)
dx_el :: proc "contextless" (m: Dx_Src, c, r: int) -> Dx_Src {
	return dx_comp(dx_at(m, c), r)
}

// The determinant of the minor of `m` keeping columns `cols` and rows
// `rows` (scalar ops, by the first column's cofactors).
@(private)
dx_minor_det :: proc(d: ^Dx, m: Dx_Src, cols, rows: []int) -> Dx_Src {
	if len(cols) == 1 {
		return dx_el(m, cols[0], rows[0])
	}
	if len(cols) == 2 {
		a := dx_op(d, .Mul, 1, dx_el(m, cols[0], rows[0]), dx_el(m, cols[1], rows[1]))
		b := dx_op(d, .Mul, 1, dx_el(m, cols[1], rows[0]), dx_el(m, cols[0], rows[1]))
		b.neg = true
		return dx_op(d, .Add, 1, a, b)
	}
	r: Dx_Src
	sub_buf: [4]int
	sub_rows := sub_buf[:len(rows) - 1]
	for i in 0 ..< len(rows) {
		k := 0
		for rr, j in rows {
			if j != i {
				sub_rows[k] = rr
				k += 1
			}
		}
		minor := dx_minor_det(d, m, cols[1:], sub_rows)
		term := dx_op(d, .Mul, 1, dx_el(m, cols[0], rows[i]), minor)
		term.neg = i % 2 == 1
		if i == 0 {
			r = dx_op(d, .Mov, 1, term)
		} else {
			dx_ins(&d.code, .Add, 0, r, 1, r, term)
		}
	}
	return r
}

@(private)
dx_determinant :: proc(d: ^Dx, m: Dx_Src, n: int) -> Dx_Src {
	all := [4]int{0, 1, 2, 3}
	return dx_minor_det(d, m, all[:n], all[:n])
}

// The adjugate over the determinant: element (c, r) is the cofactor of
// (r, c) over det.
@(private)
dx_inverse :: proc(d: ^Dx, m: Dx_Src, n: int) -> Dx_Src {
	det := dx_determinant(d, m, n)
	inv := dx_op(d, .Div, 1, dx_imm_f(1), det)
	r := dx_temps(d, n)
	cols_buf, rows_buf: [4]int
	cols, rows := cols_buf[:max(n - 1, 0)], rows_buf[:max(n - 1, 0)]
	for c in 0 ..< n {
		for rr in 0 ..< n {
			// Cofactor of element (column rr, row c).
			k := 0
			for j in 0 ..< n {
				if j != rr {
					cols[k] = j
					k += 1
				}
			}
			k = 0
			for j in 0 ..< n {
				if j != c {
					rows[k] = j
					k += 1
				}
			}
			cof := dx_minor_det(d, m, cols, rows) if n > 1 else dx_imm_f(1)
			cof.neg = (rr + c) % 2 == 1
			dx_ins(&d.code, .Mul, 0, dx_at(r, c), u8(1) << uint(rr), dx_comp(cof, 0), dx_comp(inv, 0))
		}
	}
	return r
}

// ---- textures --------------------------------------------------------------------------

@(private)
dx_sample :: proc(d: ^Dx, op: Dx_Op, ch: int, uv: Dx_Src, extra: []Dx_Src) -> Dx_Src {
	r := dx_temp(d)
	start := len(d.code)
	append(&d.code, u32(op))
	dx_put_dst(&d.code, r, 0xf)
	dx_put_src(&d.code, uv, 0xf, vector = true)
	dx_put_src(&d.code, Dx_Src{file = .Resource, idx = {d.tex_id[ch], u32(ch), 0}, rel = -1, swz = DX_XYZW}, 0xf, vector = true)
	dx_put_src(&d.code, Dx_Src{file = .Sampler, idx = {d.tex_id[ch], u32(ch), 0}, rel = -1}, 0xf)
	for x in extra {
		// Gradients are vectors; lod and bias scalars.
		dx_put_src(&d.code, x, 0xf if op == .Sample_D else 1, vector = op == .Sample_D)
	}
	dx_end(&d.code, start)
	return r
}

// resinfo (_uint) of channel `ch` at `lod`.
@(private)
dx_resinfo :: proc(d: ^Dx, ch: int, lod: Dx_Src, as_uint: bool, mask: u8) -> Dx_Src {
	r := dx_temp(d)
	start := len(d.code)
	append(&d.code, u32(Dx_Op.Resinfo) | (DX_RESINFO_UINT if as_uint else 0))
	dx_put_dst(&d.code, r, mask)
	dx_put_src(&d.code, lod, 1)
	dx_put_src(&d.code, Dx_Src{file = .Resource, idx = {d.tex_id[ch], u32(ch), 0}, rel = -1, swz = DX_XYZW}, mask, vector = true)
	dx_end(&d.code, start)
	return r
}

@(private)
dx_texture :: proc(d: ^Dx, id: Expr_Id) -> Dx_Src {
	u := d.u
	e := u.exprs[id]
	argx := args_of(u, id)
	ch := int(d.sym_chan[u.exprs[argx[0]].sym])
	base := dx_operands(d, argx[1:])
	defer dx_pop_operands(d, base)
	// Read only until the operand stack next grows (sampling evaluates no
	// expression).
	args := d.vals[base:][:len(argx) - 1]
	uv := dx_n(args[0], int(expr_type(u, argx[1]).rows))
	#partial switch e.builtin {
	case .Texture:
		if u.stage != .Fragment {
			// No derivatives: level 0 plus the bias.
			return dx_sample(d, .Sample_L, ch, uv, {dx_comp(args[1], 0)})
		}
		bias := u.exprs[argx[2]]
		if bias.kind == .Float_Lit && lit_f64(bias) == 0 {
			return dx_sample(d, .Sample, ch, uv, {})
		}
		return dx_sample(d, .Sample_B, ch, uv, {dx_comp(args[1], 0)})
	case .Texture_Lod:
		return dx_sample(d, .Sample_L, ch, uv, {dx_comp(args[1], 0)})
	case .Texture_Grad:
		return dx_sample(d, .Sample_D, ch, uv, {dx_n(args[1], 2), dx_n(args[2], 2)})
	case .Texture_Size:
		t := expr_type(u, id)
		return dx_resinfo(d, ch, dx_comp(args[0], 0), true, dx_mask(int(t.rows)))
	case .Texel_Fetch:
		// One texel, whatever the sampler: ld's address is p.xy, a
		// sampler2DArray's layer p.z in z, and the level in w, each taken
		// unsigned (a negative int one out of range as a huge uint). Out
		// of range ld reads 0 in the format's components but 1 in a
		// missing alpha (R8's (0, 0, 0, 1)), so w is masked by the texel
		// being inside the level and the levels: resinfo_uint's size at
		// the level (and layers) and its levels in w, which D3D gives
		// past the levels too (vkd3d's size there is RADV's undefined
		// one).
		lod := dx_comp(args[1], 0)
		arr := expr_type(u, argx[0]) == T_SAMPLER_ARRAY
		mask: u8 = 0x7 if arr else 0x3
		at := dx_temp(d)
		dx_ins(&d.code, .Mov, 0, at, mask, dx_n(args[0], 3 if arr else 2))
		size := dx_resinfo(d, ch, lod, true, mask | 0x8)
		if PROFILE_RULES[u.profile].flip_y {
			// p.y counted from the level's bottom row: h + ~p.y.
			dx_ins(&d.code, .Not, 0, at, 0x2, dx_comp(args[0], 1))
			dx_ins(&d.code, .Iadd, 0, at, 0x2, at, dx_comp(size, 1))
		}
		dx_ins(&d.code, .Mov, 0, at, 0x8, lod)
		r := dx_temp(d)
		start := len(d.code)
		append(&d.code, u32(Dx_Op.Ld))
		dx_put_dst(&d.code, r, 0xf)
		dx_put_src(&d.code, at, 0xf, vector = true)
		dx_put_src(&d.code, Dx_Src{file = .Resource, idx = {d.tex_id[ch], u32(ch), 0}, rel = -1, swz = DX_XYZW}, 0xf, vector = true)
		dx_end(&d.code, start)
		dx_ins(&d.code, .Ult, 0, size, mask | 0x8, at, size)
		dx_ins(&d.code, .And, 0, size, 0x1, dx_comp(size, 0), dx_comp(size, 1))
		dx_ins(&d.code, .And, 0, size, 0x1, dx_comp(size, 0), dx_comp(size, 3))
		if arr {
			dx_ins(&d.code, .And, 0, size, 0x1, dx_comp(size, 0), dx_comp(size, 2))
		}
		dx_ins(&d.code, .And, 0, r, 0x8, r, dx_comp(size, 0))
		return r
	}
	return dx_imm(0)
}

// ---- compute ---------------------------------------------------------------------------

// Each barrier's sync, by the SPIR-V backend's scopes and semantics:
// barrier() waits for the thread group and orders shared memory
// (ControlBarrier Workgroup, WorkgroupMemory); memoryBarrier() orders UAVs
// for the device and shared memory (Device, Uniform | Workgroup | Image);
// memoryBarrierShared() shared memory; memoryBarrierBuffer() and
// memoryBarrierImage() UAVs (buffers and images are both u#);
// groupMemoryBarrier() UAVs for the thread group and shared memory
// (Workgroup, Uniform | Workgroup | Image).
@(private, rodata)
DX_SYNC_FLAGS := #partial [Builtin]u32 {
	.Barrier               = DX_SYNC_T | DX_SYNC_G,
	.Memory_Barrier        = DX_SYNC_UGLOBAL | DX_SYNC_G,
	.Memory_Barrier_Shared = DX_SYNC_G,
	.Memory_Barrier_Buffer = DX_SYNC_UGLOBAL,
	.Memory_Barrier_Image  = DX_SYNC_UGLOBAL,
	.Group_Memory_Barrier  = DX_SYNC_UGROUP | DX_SYNC_G,
}

// An atomic on a shared or storage buffer word: the target's address, then
// the operands left to right, then imm_atomic_<op> (the old word, whether
// the shader uses it or not). Out of a buffer's bounds the atomic writes
// nothing and its value is undefined on D3D, so it is masked to 0 by the
// buffer's size (bufinfo), as on the CPU; shared memory's indexes are
// clamped into the variable.
@(private)
dx_atomic :: proc(d: ^Dx, id: Expr_Id) -> Dx_Src {
	u := d.u
	e := u.exprs[id]
	argx := args_of(u, id)
	lv := dx_lvalue(d, argx[0])
	assert(dx_is_mem(lv))
	word := lv.word + (u32(lv.swz[0]) if lv.swz_len == 1 else 0)
	addr := dx_mem_addr(d, lv, word)
	base := dx_operands(d, argx[1:])
	defer dx_pop_operands(d, base)
	signed := ty(u, e.type).base == .Int
	op: Dx_Op
	#partial switch e.builtin {
	case .Atomic_Add:
		op = .Imm_Atomic_Iadd
	case .Atomic_Min:
		op = .Imm_Atomic_Imin if signed else .Imm_Atomic_Umin
	case .Atomic_Max:
		op = .Imm_Atomic_Imax if signed else .Imm_Atomic_Umax
	case .Atomic_And:
		op = .Imm_Atomic_And
	case .Atomic_Or:
		op = .Imm_Atomic_Or
	case .Atomic_Xor:
		op = .Imm_Atomic_Xor
	case .Atomic_Exchange:
		op = .Imm_Atomic_Exch
	case .Atomic_Comp_Swap:
		// imm_atomic_cmp_exch's src0 is the comparand, src1 the value.
		op = .Imm_Atomic_Cmp_Exch
	}
	r := dx_scalar_temp(d)
	start := len(d.code)
	append(&d.code, u32(op))
	dx_put_dst(&d.code, r, 1)
	dx_put_mem(&d.code, lv.loc)
	dx_put_src(&d.code, addr, 1)
	for k in 0 ..< len(argx) - 1 {
		dx_put_src(&d.code, dx_comp(d.vals[base + k], 0), 1)
	}
	dx_end(&d.code, start)
	if lv.loc.file != .Tgsm {
		size := dx_scalar_temp(d)
		bstart := len(d.code)
		append(&d.code, u32(Dx_Op.Bufinfo))
		dx_put_dst(&d.code, size, 1)
		dx_put_src(&d.code, lv.loc, 1, vector = true)
		dx_end(&d.code, bstart)
		dx_ins(&d.code, .Ult, 0, size, 1, addr, size)
		dx_ins(&d.code, .And, 0, r, 1, r, size)
	}
	return r
}

// imageLoad, imageStore and imageSize. A read-only image is an SRV read
// with ld (at mip 0), a writable one a typed UAV (ld_uav_typed: only
// read-write r32 formats, as check allows, which every D3D12 device loads);
// both give 0 in every component of the format out of bounds, and the
// format's defaults (0, 0, 1) in the others, so a single-channel format's
// w is masked to the CPU's 0 out of bounds by the image's size.
@(private)
dx_image :: proc(d: ^Dx, id: Expr_Id) -> Dx_Src {
	u := d.u
	e := u.exprs[id]
	argx := args_of(u, id)
	img := int(d.sym_img[u.exprs[argx[0]].sym])
	res := d.img_res[img]
	format := image_format_of(sym_type(u, d.checked.images[img]))
	size_of_image :: proc(d: ^Dx, res: Dx_Src) -> Dx_Src {
		r := dx_temp(d)
		start := len(d.code)
		append(&d.code, u32(Dx_Op.Resinfo) | DX_RESINFO_UINT)
		dx_put_dst(&d.code, r, 0x3)
		dx_put_src(&d.code, dx_imm(0), 1)
		dx_put_src(&d.code, res, 0x3, vector = true)
		dx_end(&d.code, start)
		return r
	}
	if e.builtin == .Image_Size {
		return size_of_image(d, res)
	}
	base := dx_operands(d, argx[1:])
	defer dx_pop_operands(d, base)
	coord := dx_n(d.vals[base], 2)
	if e.builtin == .Image_Store {
		start := len(d.code)
		append(&d.code, u32(Dx_Op.Store_Uav_Typed))
		dx_put_dst(&d.code, res, 0xf)
		dx_put_src(&d.code, coord, 0xf, vector = true)
		dx_put_src(&d.code, dx_n(d.vals[base + 1], 4), 0xf, vector = true)
		dx_end(&d.code, start)
		return dx_imm(0)
	}
	r := dx_temp(d)
	start := len(d.code)
	if res.file == .Uav {
		append(&d.code, u32(Dx_Op.Ld_Uav_Typed))
		dx_put_dst(&d.code, r, 0xf)
		dx_put_src(&d.code, coord, 0xf, vector = true)
	} else {
		// ld's address: x, y, then the mip level in w.
		at := dx_temp(d)
		dx_ins(&d.code, .Mov, 0, at, 0x3, coord)
		dx_ins(&d.code, .Mov, 0, at, 0xc, dx_imm(0))
		start = len(d.code)
		append(&d.code, u32(Dx_Op.Ld))
		dx_put_dst(&d.code, r, 0xf)
		dx_put_src(&d.code, at, 0xf, vector = true)
	}
	dx_put_src(&d.code, res, 0xf, vector = true)
	dx_end(&d.code, start)
	#partial switch format {
	case .R32f, .R32ui, .R32i:
		inside := size_of_image(d, res)
		dx_ins(&d.code, .Ult, 0, inside, 0x3, coord, inside)
		dx_ins(&d.code, .And, 0, inside, 0x1, dx_comp(inside, 0), dx_comp(inside, 1))
		dx_ins(&d.code, .And, 0, r, 0x8, r, dx_comp(inside, 0))
	}
	return r
}
