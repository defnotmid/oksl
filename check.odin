package oksl

import "core:container/bit_array"
import "core:fmt"
import "core:strings"

// Semantic analysis: resolves names and overloads, types every expression,
// adds implicit conversions, lowers sampling (SkSL's `child.eval`, GLSL's
// texture family) onto one explicit-lod sample and `saturate` onto clamp,
// lays out the uniform block, finds or synthesizes the entry point and
// rejects recursion. Where the profiles differ it consults PROFILE_RULES
// and the features they leave (dialect.odin).

Checked :: struct {
	// Reached functions, callees before callers; the last is the stage's
	// entry point (stage.odin).
	funcs:         []Func_Id,
	entry:         Func_Id,
	// What the entry reads and writes: stage inputs and outputs, then the
	// resources.
	interface:     []Interface_Var,
	// Private globals in declaration order (their decls hold initializers).
	globals:       []Sym_Id,
	uniforms:      []Sym_Id,
	inputs_used:   Inputs,
	// The sampler symbol bound to each channel the shader uses.
	channels:      [CHANNEL_COUNT]Sym_Id,
	block_size:    int,
	// The block starts with the inputs (PROFILE_RULES' header).
	header:        bool,
	// Floats in the varyings' record (Interface_Var.offset).
	varying_floats: int,
}

@(private)
Checker :: struct {
	u:             ^Unit,
	rules:         ^Profile_Rules,
	struct_index:  map[string]i32,
	scopes:        [dynamic]map[string]Sym_Id,
	pending:       map[string][dynamic]Func_Decl_Id,
	by_name:       map[string][dynamic]Func_Id,
	globals:       [dynamic]Sym_Id,
	uniforms:      [dynamic]Sym_Id,
	// The symbol of each input (declared up front, or bound by a uniform
	// declaration), and the ones the shader uses.
	input_syms:    [Input]Sym_Id,
	inputs_used:   Inputs,
	channels:      [CHANNEL_COUNT]Sym_Id,
	// Channels the source names as iChannelN (GLSL keeps them for those).
	named_channels: bit_set[0 ..< CHANNEL_COUNT],
	frag_coord:    Sym_Id,
	gl_frag_coord: Sym_Id,
	gl_position:   Sym_Id,
	gl_front_facing: Sym_Id,
	// The symbols the source names (statically used).
	used:          bit_array.Bit_Array,
	next_offset:   int,
	cur_func:      Func_Id,
	// Loops (for continue), and loops and switches (for break).
	loop_depth:    int,
	break_depth:   int,
	// The source's `in` and `out` variables, in declaration order.
	stage_vars:    [dynamic]Sym_Id,
	// Floats in the varyings' record (Interface_Var.offset).
	varying_floats: int,
	// Set while checking a synthesized entry, which writes gl_FragCoord or
	// sk_FragCoord.
	entry_writes:  bool,
}

check :: proc(u: ^Unit) -> (out: Checked, ok: bool) {
	context.allocator = u.allocator
	c := Checker {
		u            = u,
		rules        = &PROFILE_RULES[u.profile],
		struct_index = make(map[string]i32),
		scopes       = make([dynamic]map[string]Sym_Id),
		pending      = make(map[string][dynamic]Func_Decl_Id),
		by_name      = make(map[string][dynamic]Func_Id),
		globals      = make([dynamic]Sym_Id),
		uniforms     = make([dynamic]Sym_Id),
		next_offset  = OFFSET_USER if PROFILE_RULES[u.profile].header else 0,
	}
	error_count := len(u.errors)
	push_scope(&c)
	declare_inputs(&c)
	for t in u.tops {
		if t.kind == .Func {
			name := u.func_decls[t.fn].name
			l := c.pending[name]
			append(&l, t.fn)
			c.pending[name] = l
		}
	}
	for t in u.tops {
		switch t.kind {
		case .Struct:
			declare_struct(&c, t.strukt)
		case .Vars:
			// Index rather than hold a `list` slice: checking can grow u.ids
			// (saturate, eval), which would leave the slice dangling.
			for k in 0 ..< t.vars.count {
				declare_global(&c, Var_Id(u.ids[t.vars.start + k]))
			}
		case .Func:
			f := resolve_func_decl(&c, t.fn)
			if f != 0 && u.func_decls[t.fn].body != 0 {
				check_body(&c, f, t.fn)
			}
		}
	}

	// The adapters check like user code, so they come before the tables
	// are copied out.
	main: Func_Id
	switch c.rules.entry {
	case .Main:
		main = find_main(&c)
	case .Main_Image:
		main = shadertoy_entry(&c)
	case .Void_Main:
		main = void_main(&c)
	}
	// Every use is known once the source is checked (the entry adds none).
	if !c.rules.header {
		layout_used_uniforms(&c)
	}
	if main != 0 && len(u.errors) == error_count {
		out.entry, out.interface = stage_entry(&c, main)
	}
	out.globals = c.globals[:]
	out.uniforms = c.uniforms[:]
	out.inputs_used = c.inputs_used
	out.channels = c.channels
	out.block_size = round_up(c.next_offset, 16)
	out.header = c.rules.header
	out.varying_floats = c.varying_floats
	if len(u.errors) == error_count && out.entry != 0 {
		out.funcs = order_functions(&c, out.entry)
	}
	return out, len(u.errors) == error_count
}

// SkSL's entry: `half4 main(float2 coord)`.
@(private)
find_main :: proc(c: ^Checker) -> (entry: Func_Id) {
	u := c.u
	v2, v4 := vec_type(.Float, 2), vec_type(.Float, 4)
	mains := c.by_name["main"]
	for f in mains {
		ps := fparams(u, f)
		if ty(u, u.funcs[f].ret) == v4 && len(ps) == 1 && ty(u, ps[0].type) == v2 && ps[0].qual == .In {
			entry = f
		} else {
			add_error(c, u.func_decls[u.funcs[f].decl].pos, .Entry_Signature)
		}
	}
	if entry == 0 && len(mains) == 0 {
		if images := c.by_name["mainImage"]; len(images) > 0 {
			add_error(c, u.func_decls[u.funcs[images[0]].decl].pos, .Sksl_Main_Image)
		} else {
			add_error(c, {line = 1, col = 1}, .No_Entry)
		}
	} else if entry != 0 && !u.funcs[entry].defined {
		add_error(c, u.func_decls[u.funcs[entry].decl].pos, .Entry_No_Body, u.funcs[entry].name)
	}
	return entry
}

@(private)
add_error :: proc(ck: ^Checker, pos: Pos, code: Error_Code, name := "", a := "", b := "", c := "", n: i64 = 0, m: i64 = 0) {
	append(&ck.u.errors, Error{pos = pos, code = code, name = name, a = a, b = b, c = c, n = n, m = m})
}

// Declares the dialect's inputs (GLSL: Shadertoy's) in the global scope.
// They count as used once referenced. Channels the source names keep
// their slot; a `uniform sampler2D` takes a free one.
@(private)
declare_inputs :: proc(c: ^Checker) {
	u := c.u
	for in_ in c.rules.inputs {
		info := INPUTS[in_]
		s := Symbol{name = info.name, type = intern(u, info.type), index = info.offset, is_const = true}
		s.storage = .Channel if in_ in CHANNEL_INPUTS else .Builtin_Uniform
		c.input_syms[in_] = add_sym(c, s)
		declare(c, c.input_syms[in_])
	}
	if c.rules.inputs & CHANNEL_INPUTS == {} {
		return
	}
	for e in u.exprs {
		if e.kind != .Ident || len(e.name) != len("iChannel0") || !strings.has_prefix(e.name, "iChannel") {
			continue
		}
		if n := int(e.name[len(e.name) - 1]) - '0'; n >= 0 && n < CHANNEL_COUNT {
			c.named_channels += {n}
		}
	}
}

// Marks a referenced input symbol used.
@(private)
use_input :: proc(c: ^Checker, s: Sym_Id) {
	sym := c.u.syms[s]
	#partial switch sym.storage {
	case .Builtin_Uniform:
		if in_, ok := input_at_offset(sym.index); ok {
			c.inputs_used += {in_}
		}
	case .Channel:
		c.channels[sym.index] = s
		c.inputs_used += {Input(int(Input.Channel0) + sym.index)}
	}
}

@(private)
tn :: proc(c: ^Checker, t: Type) -> string {
	return type_name(c.u, t)
}

@(private)
push_scope :: proc(c: ^Checker) {
	append(&c.scopes, make(map[string]Sym_Id))
}

@(private)
pop_scope :: proc(c: ^Checker) {
	pop(&c.scopes)
}

@(private)
add_sym :: proc(c: ^Checker, s: Symbol) -> Sym_Id {
	id := Sym_Id(len(c.u.syms))
	append(&c.u.syms, s)
	return id
}

@(private)
declare :: proc(c: ^Checker, id: Sym_Id) {
	s := c.u.syms[id]
	scope := &c.scopes[len(c.scopes) - 1]
	if prev, exists := scope[s.name]; exists {
		pp := c.u.syms[prev].pos
		add_error(c, s.pos, .Already_Declared_At, s.name, n = i64(pp.line), m = i64(pp.col))
		return
	}
	scope[s.name] = id
}

@(private)
lookup :: proc(c: ^Checker, name: string) -> Sym_Id {
	#reverse for scope in c.scopes {
		if s, ok := scope[name]; ok {
			return s
		}
	}
	return builtin_symbol(c, name)
}

// The uniform input a declaration `uniform <t> <name>;` binds to.
@(private)
uniform_input :: proc(name: string) -> (Input, bool) {
	for in_ in UNIFORM_INPUTS {
		if INPUTS[in_].name == name {
			return in_, true
		}
	}
	return {}, false
}

// Names nobody declares: `sk_FragCoord` (SkSL), `gl_FragCoord` (GLSL,
// written by the Shadertoy entry), and the channel sizes sampling lowers to
// (`$` keeps the name out of reach of source code; it shares the
// iChannelResolution input).
@(private)
builtin_symbol :: proc(c: ^Checker, name: string) -> Sym_Id {
	v4 := vec_type(.Float, 4)
	switch name {
	case "sk_FragCoord":
		if .Sk_Frag_Coord not_in c.u.features {
			return 0
		}
		if c.frag_coord == 0 {
			c.frag_coord = add_sym(c, Symbol{name = name, type = intern(c.u, v4), storage = .Global, is_const = true})
			append(&c.globals, c.frag_coord)
		}
		return c.frag_coord
	case "gl_FragCoord":
		if .Gl_Frag_Coord not_in c.u.features {
			return 0
		}
		if c.gl_frag_coord == 0 {
			// Shadertoy's is a global its entry adapter sets; a GLSL
			// fragment shader's is the stage's input.
			s := Symbol{name = name, type = intern(c.u, v4), storage = .Global, is_const = true}
			if c.rules.entry == .Void_Main {
				s.storage = .Stage_In
			}
			c.gl_frag_coord = add_sym(c, s)
			if s.storage == .Global {
				append(&c.globals, c.gl_frag_coord)
			}
		}
		return c.gl_frag_coord
	case "gl_Position":
		if .Gl_Position not_in c.u.features {
			return 0
		}
		if c.gl_position == 0 {
			c.gl_position = add_sym(c, Symbol{name = name, type = intern(c.u, v4), storage = .Stage_Out})
		}
		return c.gl_position
	case "gl_FrontFacing":
		if .Gl_Front_Facing not_in c.u.features {
			return 0
		}
		if c.gl_front_facing == 0 {
			c.gl_front_facing = add_sym(c, Symbol{name = name, type = intern(c.u, T_BOOL), storage = .Stage_In, is_const = true})
		}
		return c.gl_front_facing
	case "$channel_resolution":
		in_ := Input.Channel_Resolution
		if c.input_syms[in_] == 0 {
			info := INPUTS[in_]
			c.input_syms[in_] = add_sym(c, Symbol{name = name, type = intern(c.u, info.type), storage = .Builtin_Uniform, index = info.offset, is_const = true})
		}
		c.inputs_used += {in_}
		return c.input_syms[in_]
	}
	return 0
}

// ---- types ---------------------------------------------------------------------------

// The type of a spec (plus a declarator array size). Unsized arrays get
// array = -1 for the initializer to size.
@(private)
resolve_type :: proc(c: ^Checker, spec_id: Spec_Id, array_size: Expr_Id = 0, unsized := false) -> Type {
	u := c.u
	if spec_id == 0 {
		return T_ERROR
	}
	spec := u.specs[spec_id]
	t: Type
	if bt, is_builtin := builtin_type(spec.name, u.dialect); is_builtin {
		t = bt
	} else if idx, is_struct := c.struct_index[spec.name]; is_struct {
		t = Type{base = .Struct, strukt = idx}
	} else {
		add_error(c, spec.pos, .Unknown_Type, spec.name)
		return T_ERROR
	}
	if spec.array_size != 0 && array_size != 0 {
		add_error(c, spec.pos, .Array_Of_Arrays)
		return T_ERROR
	}
	size_expr := spec.array_size if spec.array_size != 0 else array_size
	if size_expr != 0 {
		check_expr(c, size_expr)
		n, ok := const_int(c, size_expr)
		if !ok {
			add_error(c, u.exprs[size_expr].pos, .Array_Size_Constant)
			return T_ERROR
		}
		if n <= 0 {
			add_error(c, u.exprs[size_expr].pos, .Array_Size_Positive)
			return T_ERROR
		}
		t.array = i32(n)
	} else if spec.unsized || unsized {
		t.array = -1
	}
	return t
}

@(private)
declare_struct :: proc(c: ^Checker, sd_id: Struct_Decl_Id) {
	u := c.u
	sd := u.struct_decls[sd_id]
	if _, exists := c.struct_index[sd.name]; exists {
		add_error(c, sd.pos, .Struct_Redeclared, sd.name)
		return
	}
	fields := make([dynamic]Struct_Field, context.temp_allocator)
	for i in sd.fields.start ..< sd.fields.start + sd.fields.count {
		f := u.vars[i]
		ft := resolve_type(c, f.spec, f.array_size)
		if ft.base == .Sampler {
			add_error(c, f.pos, .Struct_Sampler, a = tn(c, T_SAMPLER))
		}
		append(&fields, Struct_Field{f.name, intern(u, ft)})
	}
	idx := i32(len(u.structs))
	append(&u.structs, Struct_Info{sd.name, {u32(len(u.fields)), u32(len(fields))}})
	append(&u.fields, ..fields[:])
	c.struct_index[sd.name] = idx
}

// Checks a variable's initializer against its type, sizing unsized arrays.
@(private)
check_init :: proc(c: ^Checker, v: Var_Id, t: ^Type) {
	u := c.u
	init := u.vars[v].init
	if init == 0 {
		return
	}
	it := check_expr(c, init)
	if t.array < 0 && is_array(it) {
		t.array = it.array
	}
	if is_error(it) {
		return
	}
	nid, ok := coerce(c, init, t^)
	u.vars[v].init = nid
	if !ok {
		add_error(c, u.exprs[init].pos, .Init_Mismatch, u.vars[v].name, a = tn(c, t^), b = tn(c, it))
	}
}

@(private)
declare_global :: proc(c: ^Checker, v: Var_Id) {
	u := c.u
	d := u.vars[v]
	t := resolve_type(c, d.spec, d.array_size, d.unsized)
	if is_error(t) {
		return
	}
	s := Symbol{name = d.name, pos = d.pos, is_const = d.is_const, decl = v}
	if d.io != .None {
		declare_stage_var(c, v, t)
		return
	}
	if d.is_uniform {
		if d.init != 0 {
			add_error(c, d.pos, .Uniform_Initializer)
		}
		// Code written for other hosts declares the inputs it reads
		// (`uniform vec3 iResolution;`): the same declaration is the input.
		for in_ in c.rules.inputs {
			if INPUTS[in_].name != d.name {
				continue
			}
			if INPUTS[in_].type != t {
				add_error(c, d.pos, .Input_Redeclared, d.name, a = tn(c, INPUTS[in_].type))
				return
			}
			u.vars[v].sym = c.input_syms[in_]
			return
		}
		if t.base == .Sampler {
			if t.array != 0 {
				add_error(c, d.pos, .Sampler_Array, a = tn(c, T_SAMPLER))
				return
			}
			// Samplers and children take channel slots in declaration order,
			// after the iChannelN the source names.
			slot := -1
			for ch in 0 ..< CHANNEL_COUNT {
				if c.channels[ch] == 0 && ch not_in c.named_channels {
					slot = ch
					break
				}
			}
			if slot < 0 {
				add_error(c, d.pos, .Too_Many_Channels, a = tn(c, T_SAMPLER), n = CHANNEL_COUNT)
				return
			}
			s.type = intern(u, t)
			s.storage = .Channel
			s.index = slot
			id := add_sym(c, s)
			c.channels[slot] = id
			u.vars[v].sym = id
			declare(c, id)
			return
		}
		if t.base == .Struct || t.array < 0 {
			add_error(c, d.pos, .Uniform_Kind, d.name)
			return
		}
		if in_, is_input := uniform_input(d.name); is_input && INPUTS[in_].type == t && c.rules.binds_inputs {
			if c.input_syms[in_] != 0 && c.u.syms[c.input_syms[in_]].decl != 0 {
				add_error(c, d.pos, .Already_Declared, d.name)
				return
			}
			s.type = intern(u, t)
			s.storage = .Builtin_Uniform
			s.index = INPUTS[in_].offset
			s.is_const = true
			id := add_sym(c, s)
			c.input_syms[in_] = id
			c.inputs_used += {in_}
			u.vars[v].sym = id
			declare(c, id)
			return
		}
		align, size := std140(t)
		s.type = intern(u, t)
		s.storage = .Uniform
		// Without the inputs header, only used uniforms are laid out, once
		// all are checked (check).
		s.index = -1
		if c.rules.header {
			s.index = round_up(c.next_offset, align)
			c.next_offset = s.index + size
		}
		id := add_sym(c, s)
		u.vars[v].sym = id
		append(&c.uniforms, id)
		declare(c, id)
		return
	}
	if t.base == .Sampler {
		add_error(c, d.pos, .Sampler_Not_Uniform, a = tn(c, T_SAMPLER))
		return
	}
	s.storage = .Global
	check_init(c, v, &t)
	if d.init == 0 && d.is_const {
		add_error(c, d.pos, .Const_Needs_Init, d.name)
	}
	if t.array < 0 {
		add_error(c, d.pos, .Unsized_Needs_Init, d.name)
		return
	}
	s.type = intern(u, t)
	if d.is_const && (t == T_INT || t == T_UINT) {
		s.const_i, s.const_ok = const_int(c, u.vars[v].init)
	}
	id := add_sym(c, s)
	u.vars[v].sym = id
	append(&c.globals, id)
	declare(c, id)
}

// ---- functions -----------------------------------------------------------------------

@(private)
resolve_func_decl :: proc(c: ^Checker, fd_id: Func_Decl_Id) -> Func_Id {
	u := c.u
	fd := u.func_decls[fd_id]
	ret := resolve_type(c, fd.ret)
	params := make([dynamic]Func_Param, context.temp_allocator)
	for pi in fd.params.start ..< fd.params.start + fd.params.count {
		prm := u.params[pi]
		t := resolve_type(c, prm.spec, prm.array_size)
		if t.base == .Sampler && .Sampler_Params not_in c.u.features {
			add_error(c, prm.pos, missing_feature(c.u.profile, c.u.stage, .Sampler_Params))
		}
		append(&params, Func_Param{prm.name, intern(u, t), prm.qual, 0})
	}
	if is_error(ret) {
		return 0
	}
	rid := intern(u, ret)
	l := c.by_name[fd.name]
	for f in l {
		ps := fparams(u, f)
		if len(ps) != len(params) {
			continue
		}
		same := true
		for p, i in ps {
			if p.type != params[i].type {
				same = false
			}
		}
		if !same {
			continue
		}
		if u.funcs[f].decl == fd_id {
			return f
		}
		if u.funcs[f].ret != rid {
			add_error(c, fd.pos, .Return_Type_Redeclared, fd.name)
			return 0
		}
		if fd.body != 0 {
			if u.funcs[f].defined {
				dp := u.func_decls[u.funcs[f].decl].pos
				add_error(c, fd.pos, .Already_Defined_At, fd.name, n = i64(dp.line), m = i64(dp.col))
				return 0
			}
			// The definition's parameter names win over the prototype's.
			u.funcs[f].decl = fd_id
			for p, i in params {
				ps[i].name = p.name
			}
		}
		return f
	}
	f := Func_Id(len(u.funcs))
	append(&u.funcs, Func{name = fd.name, ret = rid, params = {u32(len(u.func_params)), u32(len(params))}, decl = fd_id})
	append(&u.func_params, ..params[:])
	append(&l, f)
	c.by_name[fd.name] = l
	return f
}

// Resolves not-yet-seen declarations of `name`, so calls may precede
// definitions.
@(private)
resolve_pending :: proc(c: ^Checker, name: string) {
	decls, ok := c.pending[name]
	if !ok {
		return
	}
	delete_key(&c.pending, name)
	for fd in decls {
		resolve_func_decl(c, fd)
	}
}

@(private)
check_body :: proc(c: ^Checker, f: Func_Id, fd_id: Func_Decl_Id) {
	u := c.u
	u.funcs[f].defined = true
	u.funcs[f].decl = fd_id
	c.cur_func = f
	defer c.cur_func = 0
	push_scope(c)
	defer pop_scope(c)
	fd := u.func_decls[fd_id]
	for &p, i in fparams(u, f) {
		if p.name == "" {
			continue
		}
		prm := u.params[int(fd.params.start) + i]
		s := Symbol{name = p.name, type = p.type, storage = .Param, qual = p.qual, pos = prm.pos, is_const = prm.is_const}
		p.sym = add_sym(c, s)
		declare(c, p.sym)
	}
	body := u.stmts[fd.body].body
	for k in 0 ..< body.count {
		check_stmt(c, Stmt_Id(u.ids[body.start + k]))
	}
}

// Callees before callers, from the entry and the global initializers;
// recursion is an error.
@(private)
order_functions :: proc(c: ^Checker, entry: Func_Id) -> []Func_Id {
	u := c.u
	order := make([dynamic]Func_Id)
	State :: enum u8 {
		New,
		Active,
		Done,
	}
	Walk :: struct {
		c:     ^Checker,
		state: []State,
		stack: [dynamic]Func_Id,
		order: ^[dynamic]Func_Id,
	}
	w := Walk{c, make([]State, len(u.funcs), context.temp_allocator), make([dynamic]Func_Id, context.temp_allocator), &order}
	visit :: proc(w: ^Walk, f: Func_Id) -> bool {
		u := w.c.u
		switch w.state[f] {
		case .Done:
			return true
		case .Active:
			b := strings.builder_make(context.temp_allocator)
			started := false
			for g in w.stack {
				if g == f {started = true}
				if started {
					fmt.sbprintf(&b, "%s -> ", u.funcs[g].name)
				}
			}
			strings.write_string(&b, u.funcs[f].name)
			add_error(w.c, u.func_decls[u.funcs[f].decl].pos, .Recursion, a = strings.to_string(b))
			return false
		case .New:
		}
		if !u.funcs[f].defined {
			add_error(w.c, u.func_decls[u.funcs[f].decl].pos, .Never_Defined, u.funcs[f].name)
			w.state[f] = .Done
			return false
		}
		w.state[f] = .Active
		append(&w.stack, f)
		ok := true
		for edge in u.calls {
			if edge[0] == f && !visit(w, edge[1]) {
				ok = false
				break
			}
		}
		pop(&w.stack)
		w.state[f] = .Done
		append(w.order, f)
		return ok
	}
	for edge in u.calls {
		if edge[0] == 0 {
			visit(&w, edge[1])
		}
	}
	visit(&w, entry)
	return order[:]
}

// ---- statements ----------------------------------------------------------------------

@(private)
check_stmt :: proc(c: ^Checker, id: Stmt_Id) {
	u := c.u
	if id == 0 {
		return
	}
	s := u.stmts[id]
	switch s.kind {
	case .Empty:
	case .Block:
		push_scope(c)
		for k in 0 ..< s.body.count {
			check_stmt(c, Stmt_Id(u.ids[s.body.start + k]))
		}
		pop_scope(c)
	case .Decl:
		for k in 0 ..< s.decls.count {
			check_local(c, Var_Id(u.ids[s.decls.start + k]))
		}
	case .Expr:
		check_expr(c, s.expr)
	case .If:
		check_condition(c, s.expr)
		push_scope(c)
		check_stmt(c, s.then_)
		pop_scope(c)
		push_scope(c)
		check_stmt(c, s.else_)
		pop_scope(c)
	case .For:
		push_scope(c)
		check_stmt(c, s.init)
		if s.expr != 0 {
			check_condition(c, s.expr)
		}
		if s.step != 0 {
			check_expr(c, s.step)
		}
		c.loop_depth += 1
		c.break_depth += 1
		push_scope(c)
		check_stmt(c, s.then_)
		pop_scope(c)
		c.loop_depth -= 1
		c.break_depth -= 1
		pop_scope(c)
	case .While, .Do_While:
		if s.kind == .While {
			check_condition(c, s.expr)
		}
		c.loop_depth += 1
		c.break_depth += 1
		push_scope(c)
		check_stmt(c, s.then_)
		pop_scope(c)
		c.loop_depth -= 1
		c.break_depth -= 1
		if s.kind == .Do_While {
			check_condition(c, s.expr)
		}
	case .Break:
		if c.break_depth == 0 {
			add_error(c, s.pos, .Break_Outside)
		}
	case .Continue:
		if c.loop_depth == 0 {
			add_error(c, s.pos, .Continue_Outside)
		}
	case .Discard:
		if .Discard not_in c.u.features {
			add_error(c, s.pos, missing_feature(c.u.profile, c.u.stage, .Discard), "discard")
		}
	case .Switch:
		check_switch(c, id)
	case .Case, .Default:
		// Labels are checked by their switch (check_switch).
		add_error(c, s.pos, .Case_Outside, "case" if s.kind == .Case else "default")
	case .Return:
		f := u.funcs[c.cur_func]
		ret := ty(u, f.ret)
		if s.expr == 0 {
			if ret != T_VOID {
				add_error(c, s.pos, .Must_Return, f.name, a = tn(c, ret))
			}
			return
		}
		t := check_expr(c, s.expr)
		if ret == T_VOID {
			add_error(c, s.pos, .Returns_Void, f.name)
			return
		}
		if is_error(t) {
			return
		}
		nid, ok := coerce(c, s.expr, ret)
		u.stmts[id].expr = nid
		if !ok {
			add_error(c, u.exprs[s.expr].pos, .Return_Mismatch, f.name, a = tn(c, ret), b = tn(c, t))
		}
	}
}

// A switch on an int or uint: labels are constant, distinct and come first;
// the body is one scope, `break` leaves it and `continue` the loop around.
@(private)
check_switch :: proc(c: ^Checker, id: Stmt_Id) {
	u := c.u
	s := u.stmts[id]
	t := check_expr(c, s.expr)
	if !is_error(t) && t != T_INT && t != T_UINT {
		add_error(c, u.exprs[s.expr].pos, .Switch_Type, a = tn(c, t))
		t = T_ERROR
	}
	values := make([dynamic]i64, context.temp_allocator)
	has_default := false
	push_scope(c)
	c.break_depth += 1
	for k in 0 ..< s.body.count {
		sid := Stmt_Id(u.ids[s.body.start + k])
		st := u.stmts[sid]
		switch st.kind {
		case .Case:
			check_expr(c, st.expr)
			label := st.expr
			if !is_error(t) {
				label, _ = coerce(c, st.expr, t)
			}
			v, ok := const_int(c, label)
			if !ok {
				add_error(c, u.exprs[st.expr].pos, .Case_Constant)
				continue
			}
			if t == T_UINT {
				v = i64(u32(v))
			}
			for prev in values {
				if prev == v {
					add_error(c, st.pos, .Case_Duplicate, n = v)
				}
			}
			append(&values, v)
			u.stmts[sid].expr = label
			u.stmts[sid].value = v
		case .Default:
			if has_default {
				add_error(c, st.pos, .Default_Duplicate)
			}
			has_default = true
		case .Empty, .Block, .Decl, .Expr, .If, .For, .While, .Do_While, .Break, .Continue, .Return, .Discard, .Switch:
			if k == 0 {
				add_error(c, st.pos, .Statement_Before_Case)
			}
			check_stmt(c, sid)
		}
	}
	c.break_depth -= 1
	pop_scope(c)
}

@(private)
check_condition :: proc(c: ^Checker, e: Expr_Id) {
	t := check_expr(c, e)
	if !is_error(t) && t != T_BOOL {
		add_error(c, c.u.exprs[e].pos, .Condition_Type, a = tn(c, t))
	}
}

@(private)
check_local :: proc(c: ^Checker, v: Var_Id) {
	u := c.u
	d := u.vars[v]
	t := resolve_type(c, d.spec, d.array_size, d.unsized)
	if is_error(t) {
		return
	}
	if t.base == .Sampler {
		add_error(c, d.pos, .Sampler_Not_Uniform, a = tn(c, T_SAMPLER))
		return
	}
	check_init(c, v, &t)
	if d.init == 0 && d.is_const {
		add_error(c, d.pos, .Const_Needs_Init, d.name)
	}
	if t.array < 0 {
		add_error(c, d.pos, .Unsized_Needs_Init, d.name)
		return
	}
	s := Symbol{name = d.name, type = intern(u, t), storage = .Local, pos = d.pos, is_const = d.is_const, decl = v}
	if d.is_const && (t == T_INT || t == T_UINT) {
		s.const_i, s.const_ok = const_int(c, u.vars[v].init)
	}
	id := add_sym(c, s)
	u.vars[v].sym = id
	declare(c, id)
}

// ---- conversions ---------------------------------------------------------------------

// Implicit conversions: int -> uint -> float, same shape.
@(private)
can_convert :: proc(from, to: Type) -> bool {
	if from == to {
		return true
	}
	if from.rows != to.rows || from.cols != to.cols || from.array != to.array || from.base == .Struct || to.base == .Struct {
		return false
	}
	#partial switch from.base {
	case .Int:
		return to.base == .Uint || to.base == .Float
	case .Uint:
		return to.base == .Float
	}
	return false
}

// `e` converted implicitly to `to` (a new expression when needed).
@(private)
coerce :: proc(c: ^Checker, e: Expr_Id, to: Type) -> (Expr_Id, bool) {
	from := expr_type(c.u, e)
	if from == to {
		return e, true
	}
	if !can_convert(from, to) {
		return e, false
	}
	return make_convert(c, e, to), true
}

// A conversion of `e` to `to`; literals fold.
@(private)
make_convert :: proc(c: ^Checker, e: Expr_Id, to: Type) -> Expr_Id {
	u := c.u
	src := u.exprs[e]
	nid := Expr_Id(len(u.exprs))
	if (src.kind == .Int_Lit || src.kind == .Uint_Lit) && is_scalar(to) {
		lit := src
		#partial switch to.base {
		case .Float:
			v := f64(i32(src.value)) if src.kind == .Int_Lit else f64(u32(src.value))
			lit.kind = .Float_Lit
			lit.value = transmute(u64)v
		case .Uint:
			lit.kind = .Uint_Lit
		case .Int:
			lit.kind = .Int_Lit
		}
		lit.type = intern(u, to)
		append(&u.exprs, lit)
		return nid
	}
	append(&u.exprs, Expr{kind = .Convert, pos = src.pos, a = e, type = intern(u, to)})
	return nid
}

// Converts two operands to a common base (int -> uint -> float).
@(private)
unify_bases :: proc(c: ^Checker, a, b: ^Expr_Id) -> bool {
	ta, tb := expr_type(c.u, a^), expr_type(c.u, b^)
	if ta.base == tb.base {
		return true
	}
	if !is_numeric_base(ta.base) || !is_numeric_base(tb.base) {
		return false
	}
	rank :: proc(b: Base) -> int {
		#partial switch b {
		case .Int:
			return 0
		case .Uint:
			return 1
		}
		return 2
	}
	if rank(ta.base) < rank(tb.base) {
		a^ = make_convert(c, a^, with_base(ta, tb.base))
	} else {
		b^ = make_convert(c, b^, with_base(tb, ta.base))
	}
	return true
}

// ---- constants -----------------------------------------------------------------------

@(private)
const_int :: proc(c: ^Checker, id: Expr_Id) -> (i64, bool) {
	u := c.u
	if id == 0 {
		return 0, false
	}
	e := u.exprs[id]
	#partial switch e.kind {
	case .Int_Lit:
		return i64(i32(e.value)), true
	case .Uint_Lit:
		return i64(u32(e.value)), true
	case .Ident:
		s := e.sym if e.sym != 0 else lookup(c, e.name)
		if s != 0 && u.syms[s].const_ok {
			return u.syms[s].const_i, true
		}
	case .Unary:
		v, ok := const_int(c, e.a)
		if !ok {
			return 0, false
		}
		#partial switch e.op {
		case .Minus:
			return -v, true
		case .Plus:
			return v, true
		case .Tilde:
			return ~v, true
		}
	case .Binary:
		a, aok := const_int(c, e.a)
		b, bok := const_int(c, e.b)
		if !aok || !bok {
			return 0, false
		}
		#partial switch e.op {
		case .Plus:
			return a + b, true
		case .Minus:
			return a - b, true
		case .Star:
			return a * b, true
		case .Slash:
			return a / b if b != 0 else 0, b != 0
		case .Percent:
			return a % b if b != 0 else 0, b != 0
		case .Shl:
			return a << u64(b & 63), true
		case .Shr:
			return a >> u64(b & 63), true
		case .Amp:
			return a & b, true
		case .Pipe:
			return a | b, true
		case .Caret:
			return a ~ b, true
		}
	case .Length:
		t := expr_type(u, e.a)
		if is_array(t) {
			return i64(t.array), true
		}
		if is_matrix(t) {
			return i64(t.cols), true
		}
		if is_vector(t) {
			return i64(t.rows), true
		}
	case .Call:
		t := ty(u, e.type)
		if e.call == .Constructor && e.args.count == 1 && (t == T_INT || t == T_UINT) {
			return const_int(c, Expr_Id(list(u, e.args)[0]))
		}
	case .Convert:
		t := ty(u, e.type)
		if t == T_INT || t == T_UINT {
			return const_int(c, e.a)
		}
	}
	return 0, false
}

// ---- expressions ---------------------------------------------------------------------

@(private)
check_expr :: proc(c: ^Checker, id: Expr_Id) -> Type {
	if id == 0 {
		return T_ERROR
	}
	t := check_expr_inner(c, id)
	c.u.exprs[id].type = intern(c.u, t)
	return t
}

@(private)
check_expr_inner :: proc(c: ^Checker, id: Expr_Id) -> Type {
	u := c.u
	e := u.exprs[id]
	switch e.kind {
	case .None:
		return T_ERROR
	case .Int_Lit:
		return T_INT
	case .Uint_Lit:
		return T_UINT
	case .Float_Lit:
		return T_FLOAT
	case .Bool_Lit:
		return T_BOOL
	case .Convert:
		return ty(u, e.type)
	case .Ident:
		s := lookup(c, e.name)
		if s == 0 {
			if _, is_func := c.by_name[e.name]; is_func || builtin_from_name(e.name) != .None {
				add_error(c, e.pos, .Is_Function, e.name)
			} else {
				add_error(c, e.pos, .Undeclared, e.name)
			}
			return T_ERROR
		}
		u.exprs[id].sym = s
		use_input(c, s)
		bit_array.set(&c.used, int(s))
		return sym_type(u, s)
	case .Comma:
		check_expr(c, e.a)
		return check_expr(c, e.b)
	case .Ternary:
		check_condition(c, e.a)
		tb := check_expr(c, e.b)
		tc := check_expr(c, e.c)
		if is_error(tb) || is_error(tc) {
			return T_ERROR
		}
		if tb != tc {
			if nc, ok := coerce(c, e.c, tb); ok {
				u.exprs[id].c = nc
				return tb
			}
			if nb, ok := coerce(c, e.b, tc); ok {
				u.exprs[id].b = nb
				return tc
			}
			add_error(c, e.pos, .Ternary_Mismatch, a = tn(c, tb), b = tn(c, tc))
			return T_ERROR
		}
		return tb
	case .Unary:
		t := check_expr(c, e.a)
		if is_error(t) {
			return t
		}
		#partial switch e.op {
		case .Minus, .Plus:
			if !is_numeric_base(t.base) || is_array(t) {
				add_error(c, e.pos, .Unary_Number, "-" if e.op == .Minus else "+", a = tn(c, t))
				return T_ERROR
			}
		case .Bang:
			if t != T_BOOL {
				add_error(c, e.pos, .Not_Bool, a = tn(c, t))
				return T_ERROR
			}
		case .Tilde:
			if !is_integer_base(t.base) || !is_sv(t) {
				add_error(c, e.pos, .Tilde_Integer, a = tn(c, t))
				return T_ERROR
			}
		}
		return t
	case .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
		t := check_expr(c, e.a)
		if is_error(t) {
			return t
		}
		if !is_numeric_base(t.base) || is_array(t) {
			add_error(c, e.pos, .Inc_Number, a = tn(c, t))
			return T_ERROR
		}
		check_lvalue(c, e.a)
		return t
	case .Binary:
		check_expr(c, e.a)
		check_expr(c, e.b)
		a, b := e.a, e.b
		t := check_binary(c, e.op, &a, &b, e.pos)
		u.exprs[id].a = a
		u.exprs[id].b = b
		return t
	case .Assign:
		tl := check_expr(c, e.a)
		tr := check_expr(c, e.b)
		if is_error(tl) || is_error(tr) {
			return T_ERROR
		}
		check_lvalue(c, e.a)
		if e.op == .Eq {
			nb, ok := coerce(c, e.b, tl)
			u.exprs[id].b = nb
			if !ok {
				add_error(c, e.pos, .Assign_Mismatch, a = tn(c, tr), b = tn(c, tl))
				return T_ERROR
			}
			return tl
		}
		// `a op= b` types as `a op b`, which must have a's type.
		a, b := e.a, e.b
		rt := check_binary(c, compound_op(e.op), &a, &b, e.pos)
		if is_error(rt) {
			return T_ERROR
		}
		u.exprs[id].b = b
		if a != e.a {
			add_error(c, e.pos, .Compound_Changes_Left, token_text(e.op), a = tn(c, tl), b = tn(c, tr))
			return T_ERROR
		}
		if rt != tl {
			add_error(c, e.pos, .Compound_Result, token_text(e.op), a = tn(c, tl), b = tn(c, tr), c = tn(c, rt))
			return T_ERROR
		}
		return tl
	case .Index:
		t := check_expr(c, e.a)
		it := check_expr(c, e.b)
		if is_error(t) || is_error(it) {
			return T_ERROR
		}
		if it != T_INT && it != T_UINT {
			add_error(c, u.exprs[e.b].pos, .Index_Type, a = tn(c, it))
			return T_ERROR
		}
		limit := 0
		result: Type
		switch {
		case is_array(t):
			limit = int(t.array)
			result = elem_type(t)
		case is_matrix(t):
			limit = int(t.cols)
			result = column_type(t)
		case is_vector(t):
			limit = int(t.rows)
			result = scalar_of(t)
		case:
			add_error(c, e.pos, .Not_Indexable, a = tn(c, t))
			return T_ERROR
		}
		if v, ok := const_int(c, e.b); ok && (v < 0 || v >= i64(limit)) {
			add_error(c, u.exprs[e.b].pos, .Index_Range, a = tn(c, t), n = v)
			return T_ERROR
		}
		return result
	case .Length:
		t := check_expr(c, e.a)
		if is_error(t) {
			return t
		}
		if !is_array(t) && !is_vector(t) && !is_matrix(t) {
			add_error(c, e.pos, .Length_Type, a = tn(c, t))
			return T_ERROR
		}
		return T_INT
	case .Field:
		t := check_expr(c, e.a)
		if is_error(t) {
			return t
		}
		if t.base == .Struct && t.array == 0 {
			for f, i in struct_fields(u, t.strukt) {
				if f.name == e.name {
					u.exprs[id].field = i32(i)
					return ty(u, f.type)
				}
			}
			add_error(c, e.pos, .No_Field, e.name, a = tn(c, t))
			return T_ERROR
		}
		if !is_sv(t) {
			add_error(c, e.pos, .No_Field, e.name, a = tn(c, t))
			return T_ERROR
		}
		return check_swizzle(c, id, t)
	case .Call:
		return check_call(c, id)
	}
	return T_ERROR
}

@(private)
token_text :: proc(k: Token_Kind) -> string {
	#partial switch k {
	case .Plus, .Plus_Eq:
		return "+"
	case .Minus, .Minus_Eq:
		return "-"
	case .Star, .Star_Eq:
		return "*"
	case .Slash, .Slash_Eq:
		return "/"
	case .Percent, .Percent_Eq:
		return "%"
	case .Amp, .Amp_Eq:
		return "&"
	case .Pipe, .Pipe_Eq:
		return "|"
	case .Caret, .Caret_Eq:
		return "^"
	case .Shl, .Shl_Eq:
		return "<<"
	case .Shr, .Shr_Eq:
		return ">>"
	case .Eq_Eq:
		return "=="
	case .Not_Eq:
		return "!="
	case .Lt:
		return "<"
	case .Gt:
		return ">"
	case .Lt_Eq:
		return "<="
	case .Gt_Eq:
		return ">="
	case .And_And:
		return "&&"
	case .Or_Or:
		return "||"
	case .Xor_Xor:
		return "^^"
	}
	return "?"
}

@(private)
compound_op :: proc(k: Token_Kind) -> Token_Kind {
	#partial switch k {
	case .Plus_Eq:
		return .Plus
	case .Minus_Eq:
		return .Minus
	case .Star_Eq:
		return .Star
	case .Slash_Eq:
		return .Slash
	case .Percent_Eq:
		return .Percent
	case .Amp_Eq:
		return .Amp
	case .Pipe_Eq:
		return .Pipe
	case .Caret_Eq:
		return .Caret
	case .Shl_Eq:
		return .Shl
	case .Shr_Eq:
		return .Shr
	}
	return .Invalid
}

// Types `a op b`, converting the operands (through the pointers) as needed.
@(private)
check_binary :: proc(c: ^Checker, op: Token_Kind, a, b: ^Expr_Id, pos: Pos) -> Type {
	u := c.u
	ta, tb := expr_type(u, a^), expr_type(u, b^)
	if is_error(ta) || is_error(tb) {
		return T_ERROR
	}
	bad :: proc(c: ^Checker, op: Token_Kind, ta, tb: Type, pos: Pos) -> Type {
		add_error(c, pos, .Binary_Undefined, token_text(op), a = tn(c, ta), b = tn(c, tb))
		return T_ERROR
	}
	#partial switch op {
	case .And_And, .Or_Or, .Xor_Xor:
		if ta != T_BOOL || tb != T_BOOL {
			return bad(c, op, ta, tb, pos)
		}
		return T_BOOL
	case .Eq_Eq, .Not_Eq:
		if ta != tb && !unify_bases(c, a, b) {
			return bad(c, op, ta, tb, pos)
		}
		ta, tb = expr_type(u, a^), expr_type(u, b^)
		if ta != tb {
			return bad(c, op, ta, tb, pos)
		}
		if ta.base == .Struct || is_array(ta) || ta.base == .Sampler {
			add_error(c, pos, .Compare_Unsupported, a = tn(c, ta))
			return T_ERROR
		}
		return T_BOOL
	case .Lt, .Gt, .Lt_Eq, .Gt_Eq:
		if !is_scalar(ta) || !is_scalar(tb) || !is_numeric_base(ta.base) || !is_numeric_base(tb.base) {
			if is_vector(ta) || is_vector(tb) {
				add_error(c, pos, .Compare_Vectors, token_text(op))
				return T_ERROR
			}
			return bad(c, op, ta, tb, pos)
		}
		unify_bases(c, a, b)
		return T_BOOL
	case .Amp, .Pipe, .Caret, .Percent, .Shl, .Shr:
		if !is_integer_base(ta.base) || !is_integer_base(tb.base) || !is_sv(ta) || !is_sv(tb) {
			return bad(c, op, ta, tb, pos)
		}
		if op == .Shl || op == .Shr {
			if is_vector(tb) && (is_scalar(ta) || tb.rows != ta.rows) {
				return bad(c, op, ta, tb, pos)
			}
			return ta
		}
		unify_bases(c, a, b)
		return shape_result(c, op, expr_type(u, a^), expr_type(u, b^), pos)
	case .Plus, .Minus, .Star, .Slash:
		if !is_numeric_base(ta.base) || !is_numeric_base(tb.base) || is_array(ta) || is_array(tb) {
			return bad(c, op, ta, tb, pos)
		}
		unify_bases(c, a, b)
		ta, tb = expr_type(u, a^), expr_type(u, b^)
		if is_matrix(ta) || is_matrix(tb) {
			if ta.base != .Float || tb.base != .Float {
				return bad(c, op, ta, tb, pos)
			}
			if op == .Star {
				switch {
				case is_matrix(ta) && is_matrix(tb):
					if ta.cols != tb.rows {
						return bad(c, op, ta, tb, pos)
					}
					return mat_type(int(tb.cols), int(ta.rows))
				case is_matrix(ta) && is_vector(tb):
					if ta.cols != tb.rows {
						return bad(c, op, ta, tb, pos)
					}
					return vec_type(.Float, int(ta.rows))
				case is_vector(ta) && is_matrix(tb):
					if ta.rows != tb.rows {
						return bad(c, op, ta, tb, pos)
					}
					return vec_type(.Float, int(tb.cols))
				}
			}
			// Component-wise with a scalar or a matrix of the same size.
			if is_scalar(ta) {
				return tb
			}
			if is_scalar(tb) {
				return ta
			}
			if ta == tb {
				return ta
			}
			return bad(c, op, ta, tb, pos)
		}
		return shape_result(c, op, ta, tb, pos)
	}
	return bad(c, op, ta, tb, pos)
}

// Scalar/vector component-wise result: equal shapes, or a scalar with a
// vector of the same base.
@(private)
shape_result :: proc(c: ^Checker, op: Token_Kind, ta, tb: Type, pos: Pos) -> Type {
	if ta == tb {
		return ta
	}
	if is_scalar(ta) && is_vector(tb) && ta.base == tb.base {
		return tb
	}
	if is_vector(ta) && is_scalar(tb) && ta.base == tb.base {
		return ta
	}
	add_error(c, pos, .Binary_Undefined, token_text(op), a = tn(c, ta), b = tn(c, tb))
	return T_ERROR
}

@(private)
check_swizzle :: proc(c: ^Checker, id: Expr_Id, t: Type) -> Type {
	u := c.u
	e := u.exprs[id]
	sets := [3]string{"xyzw", "rgba", "stpq"}
	n := len(e.name)
	if n == 0 || n > 4 {
		add_error(c, e.pos, .Bad_Swizzle, e.name)
		return T_ERROR
	}
	set := -1
	for s, si in sets {
		if strings.index_byte(s, e.name[0]) >= 0 {
			set = si
		}
	}
	if set < 0 {
		add_error(c, e.pos, .No_Field, e.name, a = tn(c, t))
		return T_ERROR
	}
	sw: [4]u8
	for i in 0 ..< n {
		k := strings.index_byte(sets[set], e.name[i])
		if k < 0 {
			add_error(c, e.pos, .Swizzle_Mixed, e.name)
			return T_ERROR
		}
		if k >= int(t.rows) {
			add_error(c, e.pos, .Swizzle_Range, e.name, a = tn(c, t), n = i64(t.rows))
			return T_ERROR
		}
		sw[i] = u8(k)
	}
	u.exprs[id].swizzle = sw
	u.exprs[id].swizzle_len = u8(n)
	return vec_type(t.base, n) if n > 1 else scalar_of(t)
}

// Reports an error when `e` can not be assigned to.
@(private)
check_lvalue :: proc(c: ^Checker, id: Expr_Id) -> bool {
	u := c.u
	e := u.exprs[id]
	#partial switch e.kind {
	case .Ident:
		if e.sym == 0 {
			return false
		}
		s := u.syms[e.sym]
		switch s.storage {
		case .Uniform, .Builtin_Uniform, .Channel, .Stage_In:
			add_error(c, e.pos, .Read_Only, s.name)
			return false
		case .Stage_Out:
		case .Local, .Param, .Global:
			if s.is_const && !(c.entry_writes && (e.sym == c.gl_frag_coord || e.sym == c.frag_coord)) {
				add_error(c, e.pos, .Is_Const, s.name)
				return false
			}
		}
		return true
	case .Index:
		return check_lvalue(c, e.a)
	case .Field:
		for i in 0 ..< int(e.swizzle_len) {
			for j in 0 ..< i {
				if e.swizzle[i] == e.swizzle[j] {
					add_error(c, e.pos, .Swizzle_Repeat, e.name)
					return false
				}
			}
		}
		return check_lvalue(c, e.a)
	}
	add_error(c, e.pos, .Not_Assignable)
	return false
}

// ---- calls ---------------------------------------------------------------------------

@(private)
check_call :: proc(c: ^Checker, id: Expr_Id) -> Type {
	u := c.u
	e := u.exprs[id]
	// Checking an argument can grow u.ids (saturate, eval), so slice after.
	for k in 0 ..< len(args_of(u, id)) {
		check_expr(c, args_of(u, id)[k])
	}
	args := args_of(u, id)
	for a in args {
		if is_error(expr_type(u, a)) {
			return T_ERROR
		}
	}
	if e.ctor != 0 {
		return check_constructor(c, id)
	}
	switch e.name {
	case "$eval":
		if .Children not_in c.u.features {
			add_error(c, e.pos, missing_feature(c.u.profile, c.u.stage, .Children))
			return T_ERROR
		}
		return check_eval(c, id)
	case "texture", "textureLod", "texelFetch", "textureSize", "texture2D":
		if .Samplers not_in c.u.features {
			add_error(c, e.pos, missing_feature(c.u.profile, c.u.stage, .Samplers), e.name)
			return T_ERROR
		}
		if e.name != "texture2D" {
			return check_texture(c, id)
		}
	case "saturate":
		if .Saturate not_in c.u.features {
			break
		}
		resolve_pending(c, e.name)
		if _, has_user := c.by_name[e.name]; has_user {
			break
		}
		// saturate(x) is clamp(x, 0, 1).
		if len(args) != 1 {
			add_error(c, e.pos, .Arg_Count, e.name, n = 1, m = i64(len(args)))
			return T_ERROR
		}
		zero := add_lit(c, 0, e.pos)
		one := add_lit(c, 1, e.pos)
		u.exprs[id].args = push_ids(u, {u32(args[0]), u32(zero), u32(one)})
		u.exprs[id].call = .Builtin
		u.exprs[id].builtin = .Clamp
		return check_builtin(c, id, .Clamp)
	}
	if b := builtin_from_name(e.name); b != .None {
		// A user overload with matching parameters wins over the built-in.
		resolve_pending(c, e.name)
		if _, has_user := c.by_name[e.name]; has_user {
			if f := pick_overload(c, id, false); f != 0 {
				return finish_user_call(c, id, f)
			}
		}
		u.exprs[id].call = .Builtin
		u.exprs[id].builtin = b
		return check_builtin(c, id, b)
	}
	resolve_pending(c, e.name)
	if _, ok := c.by_name[e.name]; !ok {
		add_error(c, e.pos, .Unknown_Function, e.name)
		return T_ERROR
	}
	f := pick_overload(c, id, true)
	if f == 0 {
		return T_ERROR
	}
	return finish_user_call(c, id, f)
}

// A checked float literal.
@(private)
add_lit :: proc(c: ^Checker, v: f64, pos: Pos) -> Expr_Id {
	id := Expr_Id(len(c.u.exprs))
	append(&c.u.exprs, Expr{kind = .Float_Lit, pos = pos, value = transmute(u64)v, type = intern(c.u, T_FLOAT)})
	return id
}

// child.eval(coord) samples a `uniform shader` child at `coord` in the
// child's own pixels; it becomes textureLod(child, coord / size, 0) with the
// child's size from the block (bilinear and clamped, like Skia's image
// shaders without mipmaps).
@(private)
check_eval :: proc(c: ^Checker, id: Expr_Id) -> Type {
	u := c.u
	e := u.exprs[id]
	args := args_of(u, id)
	child := u.exprs[args[0]]
	if child.kind != .Ident || child.sym == 0 || u.syms[child.sym].storage != .Channel {
		add_error(c, e.pos, .Eval_Child, a = tn(c, expr_type(u, args[0])))
		return T_ERROR
	}
	if len(args) != 2 {
		add_error(c, e.pos, .Eval_Args, child.name)
		return T_ERROR
	}
	v2 := vec_type(.Float, 2)
	coord, ok := coerce(c, args[1], v2)
	if !ok {
		add_error(c, u.exprs[args[1]].pos, .Eval_Coord, child.name, a = tn(c, expr_type(u, args[1])))
		return T_ERROR
	}
	res := Expr_Id(len(u.exprs))
	append(&u.exprs, Expr{kind = .Ident, pos = e.pos, name = "$channel_resolution"})
	index := Expr_Id(len(u.exprs))
	append(&u.exprs, Expr{kind = .Int_Lit, pos = e.pos, value = u64(u.syms[child.sym].index)})
	element := Expr_Id(len(u.exprs))
	append(&u.exprs, Expr{kind = .Index, pos = e.pos, a = res, b = index})
	size := Expr_Id(len(u.exprs))
	append(&u.exprs, Expr{kind = .Field, pos = e.pos, a = element, name = "xy"})
	check_expr(c, size)
	uv := Expr_Id(len(u.exprs))
	append(&u.exprs, Expr{kind = .Binary, op = .Slash, pos = e.pos, a = coord, b = size, type = intern(u, v2)})
	// Not inline in the assignment below: add_lit grows u.exprs, and the
	// address of u.exprs[id] is taken before the right-hand side runs.
	lod := add_lit(c, 0, e.pos)
	u.exprs[id].args = push_ids(u, {u32(args[0]), u32(uv), u32(lod)})
	u.exprs[id].call = .Builtin
	u.exprs[id].builtin = .Texture_Lod
	u.exprs[id].name = "eval"
	return check_builtin(c, id, .Texture_Lod)
}

// GLSL's sampling, lowered onto the explicit-lod sample `.eval` uses and,
// for texture() in a fragment shader, the implicit-lod one (the level its
// uv derivatives pick, plus the bias; a vertex shader's samples level 0).
// In Shadertoy code uv (0..1) and texel coordinates start at the bottom
// left and are flipped to the backends' top-left origin as
// uv * (1, -1) + (0, 1), which evaluates uv once (a GLSL fragment shader's
// start at the top left, as Vulkan's do, and are not flipped):
//
//     texture(s, uv [, bias])  ->  sample(s, flip(uv), implicit + bias)
//     textureLod(s, uv, lod)   ->  sample(s, flip(uv), lod)
//     texelFetch(s, p, lod)    ->  sample(s, flip((vec2(p) + 0.5) / size), lod)
//     textureSize(s, lod)      ->  ivec2(size)
//
// With the inputs header (Shadertoy) `size` is the channel's
// iChannelResolution.xy, so texelFetch and textureSize take a channel
// itself rather than a sampler parameter; without it (GLSL shaders) it is
// the image's own size at lod.
@(private)
check_texture :: proc(c: ^Checker, id: Expr_Id) -> Type {
	u := c.u
	e := u.exprs[id]
	args := args_of(u, id)
	want := 3 if e.name == "textureLod" || e.name == "texelFetch" else 2
	if len(args) != want && !(e.name == "texture" && len(args) == 3) {
		add_error(c, e.pos, .Arg_Count, e.name, n = i64(want), m = i64(len(args)))
		return T_ERROR
	}
	if expr_type(u, args[0]) != T_SAMPLER {
		add_error(c, u.exprs[args[0]].pos, .Texture_Sampler, e.name, a = tn(c, expr_type(u, args[0])))
		return T_ERROR
	}
	v2 := vec_type(.Float, 2)
	iv2 := vec_type(.Int, 2)
	// A checked `vec2(x, y)` of literals.
	vec2_lit :: proc(c: ^Checker, x, y: f64, pos: Pos) -> Expr_Id {
		u := c.u
		spec := Spec_Id(len(u.specs))
		append(&u.specs, Type_Spec{pos = pos, name = "vec2" if u.dialect == .GLSL else "float2"})
		lx := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Float_Lit, pos = pos, value = transmute(u64)x})
		ly := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Float_Lit, pos = pos, value = transmute(u64)y})
		args := push_ids(u, {u32(lx), u32(ly)})
		id := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Call, pos = pos, ctor = spec, name = u.specs[spec].name, args = args})
		check_expr(c, id)
		return id
	}
	binary :: proc(c: ^Checker, op: Token_Kind, a, b: Expr_Id, t: Type) -> Expr_Id {
		id := Expr_Id(len(c.u.exprs))
		append(&c.u.exprs, Expr{kind = .Binary, op = op, pos = c.u.exprs[a].pos, a = a, b = b, type = intern(c.u, t)})
		return id
	}
	// textureSize(sampler, lod) as the image reports it.
	image_size :: proc(c: ^Checker, e: Expr, sampler, lod: Expr_Id) -> Expr_Id {
		u := c.u
		args := push_ids(u, {u32(sampler), u32(lod)})
		id := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Call, pos = e.pos, name = "textureSize", args = args, call = .Builtin})
		u.exprs[id].type = intern(u, check_builtin(c, id, .Texture_Size))
		return id
	}
	// The channel's size in texels at `lod` as a vec2: the image's, or
	// with the inputs header iChannelResolution[n].xy of the channel.
	channel_size :: proc(c: ^Checker, e: Expr, sampler, lod: Expr_Id) -> (Expr_Id, bool) {
		u := c.u
		if !c.rules.header {
			return make_convert(c, image_size(c, e, sampler, lod), vec_type(.Float, 2)), true
		}
		s := u.exprs[sampler]
		if s.kind != .Ident || s.sym == 0 || u.syms[s.sym].storage != .Channel {
			add_error(c, s.pos, .Texture_Channel, e.name)
			return 0, false
		}
		res := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Ident, pos = e.pos, name = "$channel_resolution"})
		index := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Int_Lit, pos = e.pos, value = u64(u.syms[s.sym].index)})
		element := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Index, pos = e.pos, a = res, b = index})
		size := Expr_Id(len(u.exprs))
		append(&u.exprs, Expr{kind = .Field, pos = e.pos, a = element, name = "xy"})
		check_expr(c, size)
		return size, true
	}

	if e.name == "textureSize" {
		if lt := expr_type(u, args[1]); lt != T_INT && lt != T_UINT {
			return no_overload(c, e, args)
		}
		if !c.rules.header {
			u.exprs[id].call = .Builtin
			return check_builtin(c, id, .Texture_Size)
		}
		size, ok := channel_size(c, e, args[0], args[1])
		if !ok {
			return T_ERROR
		}
		spec := Spec_Id(len(u.specs))
		append(&u.specs, Type_Spec{pos = e.pos, name = "ivec2"})
		ctor_args := push_ids(u, {u32(size)})
		u.exprs[id].ctor = spec
		u.exprs[id].args = ctor_args
		return check_constructor(c, id)
	}
	// Building nodes below grows u.ids, which `args` points into.
	sampler := args[0]
	lod := args[2] if len(args) == 3 else 0
	uv: Expr_Id
	if e.name == "texelFetch" {
		p, ok := coerce(c, args[1], iv2)
		if !ok || !is_integer_base(expr_type(u, args[2]).base) || !is_scalar(expr_type(u, args[2])) {
			return no_overload(c, e, args)
		}
		size: Expr_Id
		if size, ok = channel_size(c, e, sampler, lod); !ok {
			return T_ERROR
		}
		centre := binary(c, .Plus, make_convert(c, p, v2), add_lit(c, 0.5, e.pos), v2)
		uv = binary(c, .Slash, centre, size, v2)
	} else {
		ok: bool
		uv, ok = coerce(c, args[1], v2)
		if !ok || (len(args) == 3 && !is_scalar(expr_type(u, args[2]))) {
			return no_overload(c, e, args)
		}
	}
	if c.rules.flip_y {
		scale := vec2_lit(c, 1, -1, e.pos)
		offset := vec2_lit(c, 0, 1, e.pos)
		uv = binary(c, .Plus, binary(c, .Star, uv, scale, v2), offset, v2)
	}
	if lod == 0 {
		lod = add_lit(c, 0, e.pos)
	}
	// texture()'s lod is a bias on the level its derivatives pick, in the
	// stages that have them.
	b := Builtin.Texture if e.name == "texture" && .Derivatives in c.u.features else Builtin.Texture_Lod
	sample_args := push_ids(u, {u32(sampler), u32(uv), u32(lod)})
	u.exprs[id].args = sample_args
	u.exprs[id].call = .Builtin
	return check_builtin(c, id, b)
}

@(private)
pick_overload :: proc(c: ^Checker, id: Expr_Id, report: bool) -> Func_Id {
	u := c.u
	e := u.exprs[id]
	args := args_of(u, id)
	exact: Func_Id
	loose := make([dynamic]Func_Id, context.temp_allocator)
	for f in c.by_name[e.name] {
		ps := fparams(u, f)
		if len(ps) != len(args) {
			continue
		}
		is_exact, is_loose := true, true
		for p, i in ps {
			at := expr_type(u, args[i])
			pt := ty(u, p.type)
			if at != pt {
				is_exact = false
				if p.qual != .In || !can_convert(at, pt) {
					is_loose = false
				}
			}
		}
		if is_exact {
			exact = f
		} else if is_loose {
			append(&loose, f)
		}
	}
	if exact != 0 {
		return exact
	}
	if len(loose) == 1 {
		return loose[0]
	}
	if !report {
		return 0
	}
	b := strings.builder_make(context.temp_allocator)
	for a, i in args {
		if i > 0 {strings.write_string(&b, ", ")}
		strings.write_string(&b, tn(c, expr_type(u, a)))
	}
	if len(loose) > 1 {
		add_error(c, e.pos, .Ambiguous_Call, e.name, a = strings.to_string(b))
	} else {
		add_error(c, e.pos, .No_Overload, e.name, a = strings.to_string(b))
	}
	return 0
}

@(private)
finish_user_call :: proc(c: ^Checker, id: Expr_Id, f: Func_Id) -> Type {
	u := c.u
	u.exprs[id].call = .User
	u.exprs[id].func = f
	args := args_of(u, id)
	for p, i in fparams(u, f) {
		pt := ty(u, p.type)
		if p.qual != .In {
			if expr_type(u, args[i]) != pt {
				add_error(c, u.exprs[args[i]].pos, .Out_Arg_Type, a = tn(c, pt))
			}
			check_lvalue(c, args[i])
		} else {
			args[i], _ = coerce(c, args[i], pt)
		}
	}
	append(&u.calls, [2]Func_Id{c.cur_func, f})
	return ty(u, u.funcs[f].ret)
}

@(private)
check_constructor :: proc(c: ^Checker, id: Expr_Id) -> Type {
	u := c.u
	e := u.exprs[id]
	u.exprs[id].call = .Constructor
	args := args_of(u, id)
	t := resolve_type(c, e.ctor)
	if is_error(t) {
		return t
	}
	if t.base == .Void || t.base == .Sampler {
		add_error(c, e.pos, .Not_Constructible, e.name)
		return T_ERROR
	}
	if t.array != 0 {
		elem := elem_type(t)
		if t.array > 0 && int(t.array) != len(args) {
			add_error(c, e.pos, .Array_Ctor_Count, a = tn(c, t), n = i64(t.array), m = i64(len(args)))
			return T_ERROR
		}
		for &a in args {
			na, ok := coerce(c, a, elem)
			if !ok {
				add_error(c, u.exprs[a].pos, .Array_Element_Type, a = tn(c, elem), b = tn(c, expr_type(u, a)))
				return T_ERROR
			}
			a = na
		}
		t.array = i32(len(args))
		return t
	}
	if t.base == .Struct {
		fields := struct_fields(u, t.strukt)
		if len(fields) != len(args) {
			add_error(c, e.pos, .Struct_Ctor_Count, a = tn(c, t), n = i64(len(fields)), m = i64(len(args)))
			return T_ERROR
		}
		for &a, i in args {
			ft := ty(u, fields[i].type)
			na, ok := coerce(c, a, ft)
			if !ok {
				add_error(c, u.exprs[a].pos, .Struct_Field_Type, fields[i].name, a = tn(c, t), b = tn(c, ft), c = tn(c, expr_type(u, a)))
				return T_ERROR
			}
			a = na
		}
		return t
	}
	if len(args) == 0 {
		add_error(c, e.pos, .Ctor_No_Args, a = tn(c, t))
		return T_ERROR
	}
	for a in args {
		at := expr_type(u, a)
		if !is_sv(at) && !is_matrix(at) {
			add_error(c, u.exprs[a].pos, .Ctor_Arg_Type, a = tn(c, at), b = tn(c, t))
			return T_ERROR
		}
	}
	if len(args) == 1 {
		a := expr_type(u, args[0])
		if is_scalar(t) || is_scalar(a) || (is_matrix(t) && is_matrix(a)) {
			return t
		}
		if is_vector(t) && is_vector(a) && a.rows < t.rows {
			add_error(c, e.pos, .Ctor_Too_Few, a = tn(c, a), b = tn(c, t))
			return T_ERROR
		}
		if is_matrix(t) && !is_matrix(a) {
			add_error(c, e.pos, .Ctor_Matrix_Args, a = tn(c, t))
			return T_ERROR
		}
		return t
	}
	need := int(t.rows) * max(int(t.cols), 1)
	have := 0
	for a in args {
		if have >= need {
			add_error(c, u.exprs[a].pos, .Ctor_Too_Many, a = tn(c, t))
			return T_ERROR
		}
		have += component_count(u, expr_type(u, a))
	}
	if have < need {
		add_error(c, e.pos, .Ctor_Components, a = tn(c, t), n = i64(need), m = i64(have))
		return T_ERROR
	}
	return t
}

@(private)
check_builtin :: proc(c: ^Checker, id: Expr_Id, b: Builtin) -> Type {
	u := c.u
	e := u.exprs[id]
	args := args_of(u, id)
	n := len(args)
	b := b
	// Variants chosen by the arguments: atan(y, x) and mix(x, y, bvec).
	if b == .Atan && n == 2 {
		b = .Atan2
	}
	if n != int(BUILTINS[b].argc) {
		add_error(c, e.pos, .Arg_Count, e.name, n = i64(BUILTINS[b].argc), m = i64(n))
		return no_overload(c, e, args)
	}
	if b == .Mix && expr_type(u, args[2]).base == .Bool {
		b = .Mix_Bool
	}
	u.exprs[id].builtin = b
	info := BUILTINS[b]
	for f in info.needs - c.u.features {
		add_error(c, e.pos, missing_feature(c.u.profile, c.u.stage, f), e.name)
		return T_ERROR
	}
	if !builtin_args_ok(c, info, args) {
		return no_overload(c, e, args)
	}
	t0, last := expr_type(u, args[0]), expr_type(u, args[n - 1])
	switch info.result {
	case .First:
		return t0
	case .Last:
		return last
	case .Float:
		return T_FLOAT
	case .Bool:
		return T_BOOL
	case .Vec4:
		return vec_type(.Float, 4)
	case .Bool_Shape:
		return with_base(t0, .Bool)
	case .Int_Shape:
		return with_base(t0, .Int)
	case .Uint_Shape:
		return with_base(t0, .Uint)
	case .Float_Shape:
		return with_base(t0, .Float)
	case .Outer:
		return mat_type(int(last.rows), int(t0.rows))
	case .Transposed:
		return mat_type(int(t0.rows), int(t0.cols))
	case .Ivec2:
		return vec_type(.Int, 2)
	}
	return T_ERROR
}

// The uniforms the source uses, laid out std140 from the block's start in
// declaration order; the others are left out of the block.
@(private)
layout_used_uniforms :: proc(c: ^Checker) {
	u := c.u
	used := make([dynamic]Sym_Id)
	for id in c.uniforms {
		if !bit_array.get(&c.used, int(id)) {
			continue
		}
		align, size := std140(sym_type(u, id))
		u.syms[id].index = round_up(c.next_offset, align)
		c.next_offset = u.syms[id].index + size
		append(&used, id)
	}
	c.uniforms = used
}

// "no overload of f takes (types)", with the arguments' current types.
@(private)
no_overload :: proc(c: ^Checker, e: Expr, args: []Expr_Id) -> Type {
	sb := strings.builder_make(context.temp_allocator)
	for a, i in args {
		if i > 0 {strings.write_string(&sb, ", ")}
		strings.write_string(&sb, tn(c, expr_type(c.u, a)))
	}
	add_error(c, e.pos, .No_Overload, e.name, a = strings.to_string(sb))
	return T_ERROR
}

// Checks a built-in call's arguments against its rule, adding the implicit
// conversions it allows.
@(private)
builtin_args_ok :: proc(c: ^Checker, info: Builtin_Info, args: []Expr_Id) -> bool {
	at :: proc(c: ^Checker, args: []Expr_Id, i: int) -> Type {
		return expr_type(c.u, args[i])
	}
	// Numeric (int) arguments convert to float of the same shape.
	to_float :: proc(c: ^Checker, args: []Expr_Id, i: int) -> bool {
		t := expr_type(c.u, args[i])
		if t.base == .Float {
			return true
		}
		if t.base == .Int || t.base == .Uint {
			args[i] = make_convert(c, args[i], with_base(t, .Float))
			return true
		}
		return false
	}
	floats :: proc(c: ^Checker, args: []Expr_Id) -> bool {
		for i in 0 ..< len(args) {
			if !to_float(c, args, i) {
				return false
			}
		}
		return true
	}
	// Argument i matches args[0], or is a scalar of its base (backends
	// broadcast it).
	same_or_scalar :: proc(c: ^Checker, args: []Expr_Id, i: int) -> bool {
		t0 := expr_type(c.u, args[0])
		ti := expr_type(c.u, args[i])
		if ti == t0 || (is_scalar(ti) && ti.base == t0.base) {
			return true
		}
		if can_convert(ti, t0) {
			args[i] = make_convert(c, args[i], t0)
			return true
		}
		if is_scalar(ti) && can_convert(ti, scalar_of(t0)) {
			args[i] = make_convert(c, args[i], scalar_of(t0))
			return true
		}
		return false
	}
	gen_float :: proc(t: Type) -> bool {
		return is_sv(t) && t.base == .Float
	}
	// The first two arguments with their bases unified.
	unified :: proc(c: ^Checker, args: []Expr_Id) -> bool {
		if at(c, args, 0) != at(c, args, 1) {
			a0, a1 := args[0], args[1]
			if !unify_bases(c, &a0, &a1) {
				return false
			}
			args[0], args[1] = a0, a1
		}
		return true
	}
	n := len(args)
	switch info.args {
	case .None:
		return false
	case .Float:
		return to_float(c, args, 0) && is_sv(at(c, args, 0))
	case .Float_Exact:
		return gen_float(at(c, args, 0))
	case .Float_Or_Int:
		t := at(c, args, 0)
		return is_sv(t) && (t.base == .Float || t.base == .Int)
	case .Exact_Base:
		return is_sv(at(c, args, 0)) && at(c, args, 0).base == info.base
	case .Float_Same:
		if !floats(c, args) || !gen_float(at(c, args, 0)) {
			return false
		}
		for i in 1 ..< n {
			if !(same_or_scalar(c, args, i) if i in info.broadcast else at(c, args, i) == at(c, args, 0)) {
				return false
			}
		}
		return true
	case .Numeric_Same:
		if !is_sv(at(c, args, 0)) || at(c, args, 0).base == .Bool {
			return false
		}
		if at(c, args, 0).base != .Float {
			for i in 1 ..< n {
				if at(c, args, i).base == .Float {
					to_float(c, args, 0)
					break
				}
			}
		}
		for i in 1 ..< n {
			if !same_or_scalar(c, args, i) {
				return false
			}
		}
		return true
	case .Edges:
		if !floats(c, args) || !gen_float(at(c, args, n - 1)) {
			return false
		}
		t := at(c, args, n - 1)
		for i in 0 ..< n - 1 {
			if at(c, args, i) != t && !is_scalar(at(c, args, i)) {
				return false
			}
		}
		return at(c, args, 0) == at(c, args, n - 2)
	case .Mix_Bool:
		if !unified(c, args) {
			return false
		}
		return is_sv(at(c, args, 0)) && at(c, args, 0) == at(c, args, 1) && at(c, args, 2).rows == at(c, args, 0).rows
	case .Vec3:
		v3 := vec_type(.Float, 3)
		return floats(c, args) && at(c, args, 0) == v3 && at(c, args, 1) == v3
	case .Refract:
		return floats(c, args) && gen_float(at(c, args, 0)) && at(c, args, 1) == at(c, args, 0) && at(c, args, 2) == T_FLOAT
	case .Float_Vectors:
		return floats(c, args) && is_vector(at(c, args, 0)) && is_vector(at(c, args, 1))
	case .Matrix:
		return is_matrix(at(c, args, 0))
	case .Matrix_Same:
		return is_matrix(at(c, args, 0)) && at(c, args, 0) == at(c, args, 1)
	case .Square_Matrix:
		return is_matrix(at(c, args, 0)) && at(c, args, 0).cols == at(c, args, 0).rows
	case .Vector_Ordered, .Vector_Equal:
		if !is_vector(at(c, args, 0)) {
			return false
		}
		if info.args == .Vector_Ordered && !is_numeric_base(at(c, args, 0).base) {
			return false
		}
		return unified(c, args) && at(c, args, 0) == at(c, args, 1)
	case .Bool_Vector:
		return is_vector(at(c, args, 0)) && at(c, args, 0).base == .Bool
	case .Texture_Lod:
		return at(c, args, 0) == T_SAMPLER && at(c, args, 1) == vec_type(.Float, 2) && to_float(c, args, 2) && at(c, args, 2) == T_FLOAT
	case .Texture_Size:
		return at(c, args, 0) == T_SAMPLER && (at(c, args, 1) == T_INT || at(c, args, 1) == T_UINT)
	}
	return false
}
