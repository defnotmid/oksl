package oksl

import "core:strconv"

@(private)
Parser :: struct {
	u:                   ^Unit,
	toks:                []Token,
	i:                   int,
	struct_names:        map[string]bool,
	default_buf_packing: string,
	// Set after an error until the parser resynchronizes.
	panicking:           bool,
}

// Parses preprocessed tokens into `u` (tops, expressions, statements...).
// One parser serves both dialects; only the built-in type names differ.
parse :: proc(u: ^Unit, toks: []Token) {
	context.allocator = u.allocator
	p := Parser {
		u            = u,
		toks         = toks,
		struct_names = make(map[string]bool),
	}
	// Structs parsed before (a program's Common unit) name types too.
	for sd in u.struct_decls[1:] {
		p.struct_names[sd.name] = true
	}
	for peek(&p) != .EOF {
		start := p.i
		parse_top(&p)
		if p.panicking {
			// Skip to the end of the declaration.
			depth := 0
			for peek(&p) != .EOF {
				k := peek(&p)
				p.i += 1
				if k == .L_Brace {depth += 1}
				if k == .R_Brace {
					depth -= 1
					if depth <= 0 {break}
				}
				if k == .Semicolon && depth == 0 {break}
			}
			p.panicking = false
		}
		if p.i == start {
			p.i += 1
		}
	}
}

// ---- helpers -------------------------------------------------------------------------

@(private = "file")
peek :: proc(p: ^Parser, off := 0) -> Token_Kind {
	j := p.i + off
	return p.toks[j].kind if j < len(p.toks) else .EOF
}

@(private = "file")
tok :: proc(p: ^Parser, off := 0) -> Token {
	j := min(p.i + off, len(p.toks) - 1)
	return p.toks[j]
}

@(private = "file")
error_at :: proc(p: ^Parser, pos: Pos, code: Error_Code, name := "", a := "", b := "") {
	if p.panicking {
		return
	}
	append(&p.u.errors, Error{pos = pos, code = code, name = name, a = a, b = b})
	p.panicking = true
}

@(private = "file")
accept :: proc(p: ^Parser, k: Token_Kind) -> bool {
	if peek(p) == k {
		p.i += 1
		return true
	}
	return false
}

@(private = "file")
expect :: proc(p: ^Parser, k: Token_Kind, code := Error_Code.Expected) -> bool {
	if peek(p) == k {
		p.i += 1
		return true
	}
	t := tok(p)
	error_at(p, t.pos, code, a = token_text(k), b = t.text)
	return false
}

@(private = "file")
is_ident :: proc(p: ^Parser, text: string, off := 0) -> bool {
	t := tok(p, off)
	return t.kind == .Ident && t.text == text
}

@(private = "file")
is_type_name :: proc(p: ^Parser, off := 0) -> bool {
	t := tok(p, off)
	if t.kind != .Ident {
		return false
	}
	if _, ok := builtin_type(t.text, p.u.dialect); ok {
		return true
	}
	return t.text == "struct" || p.struct_names[t.text]
}

@(private = "file")
skip_precision :: proc(p: ^Parser) {
	for is_ident(p, "highp") || is_ident(p, "mediump") || is_ident(p, "lowp") || is_ident(p, "inline") || is_ident(p, "noinline") {
		p.i += 1
	}
}

@(private = "file")
add_expr :: proc(p: ^Parser, e: Expr) -> Expr_Id {
	id := Expr_Id(len(p.u.exprs))
	append(&p.u.exprs, e)
	return id
}

@(private = "file")
add_stmt :: proc(p: ^Parser, s: Stmt) -> Stmt_Id {
	id := Stmt_Id(len(p.u.stmts))
	append(&p.u.stmts, s)
	return id
}

@(private = "file")
add_spec :: proc(p: ^Parser, s: Type_Spec) -> Spec_Id {
	id := Spec_Id(len(p.u.specs))
	append(&p.u.specs, s)
	return id
}

@(private = "file")
add_var :: proc(p: ^Parser, v: Var_Decl) -> Var_Id {
	id := Var_Id(len(p.u.vars))
	append(&p.u.vars, v)
	return id
}

// ---- declarations ------------------------------------------------------------------

@(private = "file")
Layout_Qual :: struct {
	has_layout:    bool,
	has_location:  bool,
	location_uint: bool,
	bad:           bool,
	empty:         bool,
	non_default:   bool,
	count:         i32,
	layout_pos:    Pos,
	location:      i32,
	packing:       string,
	format:        Image_Format,
	workgroups:    [dynamic]Workgroup_Item,
}

@(private = "file")
parse_layout :: proc(p: ^Parser, lay: ^Layout_Qual) {
	lay.has_layout = true
	lay.layout_pos = tok(p).pos
	p.i += 1
	if !accept(p, .L_Paren) {
		return
	}
	start_count := lay.count
	need_comma := false
	for peek(p) != .R_Paren && peek(p) != .EOF {
		if accept(p, .Comma) {
			if !need_comma || peek(p) == .R_Paren {
				lay.empty = true
			}
			need_comma = false
			continue
		}
		if need_comma {
			expect(p, .Comma)
			lay.bad = true
			return
		}
		need_comma = true
		lay.count += 1
		if is_ident(p, "location") && peek(p, 1) == .Eq && (peek(p, 2) == .Int_Lit || peek(p, 2) == .Uint_Lit) {
			vt := tok(p, 2)
			val, ok := parse_int_text(vt.text)
			if !ok {
				error_at(p, vt.pos, .Invalid_Int_Literal, a = vt.text)
				lay.bad = true
				return
			}
			lay.location = i32(val)
			lay.has_location = true
			lay.location_uint = vt.kind == .Uint_Lit
			lay.non_default = true
			p.i += 3
			continue
		}
		if is_ident(p, "local_size_x") || is_ident(p, "local_size_y") || is_ident(p, "local_size_z") {
			t := tok(p)
			axis := i8(t.text[len("local_size_")] - 'x')
			lay.non_default = true
			p.i += 1
			if !expect(p, .Eq) {
				lay.bad = true
				return
			}
			vt := tok(p)
			next_k := peek(p, 1)
			if (vt.kind == .Int_Lit || vt.kind == .Uint_Lit) && (next_k == .Comma || next_k == .R_Paren || next_k == .Ident) {
				val, ok := parse_int_text(vt.text)
				if !ok {
					error_at(p, vt.pos, .Invalid_Int_Literal, a = vt.text)
					lay.bad = true
					return
				}
				p.i += 1
				if lay.workgroups.allocator.procedure == nil {
					lay.workgroups = make([dynamic]Workgroup_Item, 0, 4, context.temp_allocator)
				}
				lv := i64(val) if vt.kind == .Uint_Lit else i64(i32(val))
				append(&lay.workgroups, Workgroup_Item{pos = t.pos, value = lv, axis = axis})
				continue
			}
			error_at(p, vt.pos, .Local_Size_Constant, t.text)
			lay.bad = true
			return
		}
		if peek(p) == .Ident && peek(p, 1) == .Eq {
			// Other `key = value` layout qualifiers (binding, set, offset, ...): ignored on declarations, rejected as defaults.
			lay.non_default = true
			p.i += 2
			parse_assignment(p)
			continue
		}
		if peek(p) == .Ident {
			t := tok(p)
			switch t.text {
			case "std430", "std140", "shared", "packed":
				lay.packing = t.text
			case "scalar":
				lay.packing = t.text
				lay.non_default = true
			case "row_major", "column_major":
				// Matrix storage order on buffer blocks: accepted.
			case:
				if fmt, ok := image_format_from_name(t.text); ok {
					lay.format = fmt
					lay.non_default = true
				} else if p.u.dialect == .GLSL {
					error_at(p, t.pos, .Image_Format_Unknown, t.text)
					lay.bad = true
				}
			}
			p.i += 1
			continue
		}
		lay.non_default = true
		p.i += 1
	}
	if lay.count == start_count {
		lay.empty = true
	}
	accept(p, .R_Paren)
}

@(private = "file")
parse_top :: proc(p: ^Parser) {
	u := p.u
	if accept(p, .Semicolon) {
		return
	}
	if is_ident(p, "precision") {
		for peek(p) != .Semicolon && peek(p) != .EOF {
			p.i += 1
		}
		accept(p, .Semicolon)
		return
	}
	lay := Layout_Qual{location = -1}
	stage_vars := .Stage_Variables in p.u.features
	buffers := .Storage_Buffers in p.u.features
	is_const, is_uniform, flat, readonly, writeonly := false, false, false, false, false
	has_mem_qual, has_prec, has_interp := false, false, false
	io: Var_Io
	for {
		before_prec := p.i
		skip_precision(p)
		if p.i != before_prec {
			has_prec = true
		}
		default_ok := lay.has_layout && !lay.empty && !lay.non_default && lay.count > 0 && !has_mem_qual && !has_prec && !has_interp && !is_const && io == .None
		if is_ident(p, "layout") && peek(p, 1) == .L_Paren {
			parse_layout(p, &lay)
			if p.panicking {
				return
			}
		} else if buffers && (is_ident(p, "readonly") || is_ident(p, "writeonly") || is_ident(p, "coherent") || is_ident(p, "volatile") || is_ident(p, "restrict")) {
			has_mem_qual = true
			readonly ||= is_ident(p, "readonly")
			writeonly ||= is_ident(p, "writeonly")
			p.i += 1
		} else if default_ok && buffers && is_ident(p, "buffer") && peek(p, 1) == .Semicolon && u.dialect == .GLSL {
			if !lay.bad && lay.packing != "" {
				p.default_buf_packing = lay.packing
			}
			p.i += 2
			return
		} else if buffers && is_ident(p, "buffer") {
			p.i += 1
			parse_buffer_block(p, readonly, writeonly, &lay)
			return
		} else if is_ident(p, "const") {
			is_const = true
			p.i += 1
		} else if default_ok && lay.packing != "std430" && is_ident(p, "uniform") && peek(p, 1) == .Semicolon && u.dialect == .GLSL {
			p.i += 2
			return
		} else if is_ident(p, "uniform") {
			is_uniform = true
			p.i += 1
		} else if is_ident(p, "in") && peek(p, 1) == .Semicolon && u.dialect == .GLSL {
			in_pos := tok(p).pos
			p.i += 2
			if len(lay.workgroups) > 0 {
				r := Range{u32(len(u.workgroups)), u32(len(lay.workgroups))}
				append(&u.workgroups, ..lay.workgroups[:])
				append(&u.tops, Top{kind = .Workgroup, vars = r})
			} else if !lay.bad {
				error_at(p, lay.layout_pos if lay.has_layout else in_pos, .Missing_Local_Size if u.stage == .Compute else .Expected_Type, a = "in")
			}
			return
		} else if is_ident(p, "in") || is_ident(p, "out") || is_ident(p, "varying") || is_ident(p, "attribute") {
			t := tok(p)
			if !stage_vars || t.text == "varying" || t.text == "attribute" {
				code := Error_Code.Varying_Keyword
				if !stage_vars {
					code = .Compute_Stage_Var if u.stage == .Compute && u.dialect == .GLSL else .Global_In_Out
				}
				error_at(p, t.pos, code, t.text)
				return
			}
			io = .In if t.text == "in" else .Out
			p.i += 1
		} else if stage_vars && (is_ident(p, "flat") || is_ident(p, "smooth") || is_ident(p, "centroid")) {
			// Interpolation: flat matters (integer inputs need it), the
			// others are the default.
			has_interp = true
			flat ||= is_ident(p, "flat")
			p.i += 1
		} else {
			break
		}
	}
	start := tok(p)
	spec := parse_type_spec(p)
	if spec == 0 {
		return
	}
	if sd := u.specs[spec].strukt; sd != 0 {
		append(&u.tops, Top{kind = .Struct, strukt = sd})
		if accept(p, .Semicolon) {
			return
		}
	}
	if peek(p) != .Ident {
		error_at(p, tok(p).pos, .Expected_Name, a = start.text)
		return
	}
	name := tok(p)
	if peek(p, 1) == .L_Paren {
		if is_const || is_uniform || io != .None {
			error_at(p, name.pos, .Const_Function)
			return
		}
		p.i += 2
		if fn := parse_function(p, spec, name); fn != 0 {
			append(&u.tops, Top{kind = .Func, fn = fn})
		}
		return
	}
	vars, ok := parse_declarators(p, spec, is_const, is_uniform)
	for k in 0 ..< vars.count {
		v := &u.vars[u.ids[vars.start + k]]
		v.io, v.location, v.flat = io, lay.location, flat
		v.has_location, v.location_uint = lay.has_location, lay.location_uint
		v.readonly, v.writeonly = readonly, writeonly
		v.format, v.bad_layout = lay.format, lay.bad
	}
	if ok {
		append(&u.tops, Top{kind = .Vars, vars = vars})
	}
	expect(p, .Semicolon)
}

// `buffer Block { T name[]; } instance;` after its qualifiers (the
// checker takes one runtime-sized array). Members are parsed first (their
// array sizes may add expressions), then stored contiguously.
@(private = "file")
parse_buffer_block :: proc(p: ^Parser, readonly, writeonly: bool, lay: ^Layout_Qual) {
	u := p.u
	block := tok(p)
	if !expect(p, .Ident, .Expected_Name) || !expect(p, .L_Brace) {
		return
	}
	members := make([dynamic]Var_Decl, context.temp_allocator)
	for peek(p) != .R_Brace && peek(p) != .EOF {
		spec := parse_type_spec(p)
		if spec == 0 {
			return
		}
		name := tok(p)
		if !expect(p, .Ident, .Expected_Field) {
			return
		}
		v := Var_Decl{pos = name.pos, name = name.text, spec = spec, location = -1}
		if accept(p, .L_Bracket) {
			if peek(p) == .R_Bracket {
				v.unsized = true
			} else {
				v.array_size = parse_expr(p)
			}
			expect(p, .R_Bracket)
		}
		append(&members, v)
		if !expect(p, .Semicolon) {
			return
		}
	}
	expect(p, .R_Brace)
	bd := Buffer_Decl {
		pos        = block.pos,
		block      = block.text,
		packing    = lay.packing if lay.packing != "" else p.default_buf_packing,
		readonly   = readonly,
		writeonly  = writeonly,
		bad_layout = lay.bad,
	}
	if peek(p) == .Ident {
		bd.instance = tok(p).text
		p.i += 1
	}
	expect(p, .Semicolon)
	bd.members = {u32(len(u.vars)), u32(len(members))}
	append(&u.vars, ..members[:])
	id := Buffer_Decl_Id(len(u.buffer_decls))
	append(&u.buffer_decls, bd)
	append(&u.tops, Top{kind = .Buffer, buffer = id})
}

@(private = "file")
parse_type_spec :: proc(p: ^Parser) -> Spec_Id {
	u := p.u
	skip_precision(p)
	t := tok(p)
	if t.kind != .Ident {
		error_at(p, t.pos, .Expected_Type, a = t.text)
		return 0
	}
	if t.text == "struct" {
		p.i += 1
		sd := Struct_Decl{pos = t.pos}
		if peek(p) != .Ident {
			error_at(p, tok(p).pos, .Anonymous_Struct)
			return 0
		}
		sd.name = tok(p).text
		p.i += 1
		p.struct_names[sd.name] = true
		if !expect(p, .L_Brace) {
			return 0
		}
		// Fields are parsed first (their array sizes may add expressions),
		// then stored contiguously.
		fields := make([dynamic]Var_Decl, context.temp_allocator)
		for peek(p) != .R_Brace && peek(p) != .EOF {
			fspec := parse_type_spec(p)
			if fspec == 0 {
				return 0
			}
			for {
				ft := tok(p)
				if !expect(p, .Ident, .Expected_Field) {
					return 0
				}
				f := Var_Decl{pos = ft.pos, name = ft.text, spec = fspec}
				if accept(p, .L_Bracket) {
					f.array_size = parse_expr(p)
					expect(p, .R_Bracket)
				}
				append(&fields, f)
				if !accept(p, .Comma) {break}
			}
			if !expect(p, .Semicolon) {
				return 0
			}
		}
		expect(p, .R_Brace)
		sd.fields = {u32(len(u.vars)), u32(len(fields))}
		append(&u.vars, ..fields[:])
		sid := Struct_Decl_Id(len(u.struct_decls))
		append(&u.struct_decls, sd)
		return add_spec(p, Type_Spec{pos = t.pos, name = sd.name, strukt = sid})
	}
	if !is_type_name(p) {
		// The other dialect's sampler types get a hint.
		has := DIALECT_RULES[p.u.dialect].has
		code := Error_Code.Unknown_Type
		switch t.text {
		case "sampler2D":
			code = .Sksl_Sampler2D if .Samplers not_in has else code
		case "shader":
			code = .Glsl_Shader_Type if .Children not_in has else code
		case "colorFilter", "blender":
			code = .Sksl_Child_Kind if .Children in has else code
		}
		error_at(p, t.pos, code, t.text)
		return 0
	}
	spec := Type_Spec{pos = t.pos, name = t.text}
	p.i += 1
	if peek(p) == .L_Bracket {
		p.i += 1
		if accept(p, .R_Bracket) {
			spec.unsized = true
		} else {
			spec.array_size = parse_expr(p)
			expect(p, .R_Bracket)
		}
	}
	return add_spec(p, spec)
}

@(private = "file")
parse_declarators :: proc(p: ^Parser, spec: Spec_Id, is_const, is_uniform: bool) -> (Range, bool) {
	ids := make([dynamic]u32, context.temp_allocator)
	for {
		t := tok(p)
		if !expect(p, .Ident, .Expected_Variable) {
			return {}, false
		}
		v := Var_Decl{pos = t.pos, name = t.text, spec = spec, is_const = is_const, is_uniform = is_uniform}
		if accept(p, .L_Bracket) {
			if accept(p, .R_Bracket) {
				v.unsized = true
			} else {
				v.array_size = parse_expr(p)
				expect(p, .R_Bracket)
			}
		}
		if accept(p, .Eq) {
			v.init = parse_assignment(p)
		}
		append(&ids, u32(add_var(p, v)))
		if !accept(p, .Comma) {
			break
		}
	}
	return push_ids(p.u, ids[:]), true
}

@(private = "file")
parse_function :: proc(p: ^Parser, ret: Spec_Id, name: Token) -> Func_Decl_Id {
	u := p.u
	fn := Func_Decl{pos = name.pos, name = name.text, ret = ret}
	params := make([dynamic]Param, context.temp_allocator)
	if is_ident(p, "void") && peek(p, 1) == .R_Paren {
		p.i += 1
	}
	for peek(p) != .R_Paren && peek(p) != .EOF {
		prm := Param{pos = tok(p).pos}
		for {
			skip_precision(p)
			if is_ident(p, "layout") && peek(p, 1) == .L_Paren {
				lay_pos := tok(p).pos
				p.i += 2
				depth := 1
				for depth > 0 && peek(p) != .EOF {
					if peek(p) == .L_Paren {
						depth += 1
					} else if peek(p) == .R_Paren {
						depth -= 1
					}
					p.i += 1
				}
				error_at(p, lay_pos, .Param_Layout)
				continue
			} else if is_ident(p, "const") {
				prm.is_const = true
			} else if is_ident(p, "in") {
				prm.qual = .In
			} else if is_ident(p, "out") {
				prm.qual = .Out
			} else if is_ident(p, "inout") {
				prm.qual = .Inout
			} else if u.dialect == .GLSL && (is_ident(p, "readonly") || is_ident(p, "writeonly") || is_ident(p, "coherent") || is_ident(p, "volatile") || is_ident(p, "restrict")) {
				prm.readonly ||= is_ident(p, "readonly")
				prm.writeonly ||= is_ident(p, "writeonly")
			} else {
				break
			}
			p.i += 1
		}
		prm.spec = parse_type_spec(p)
		if prm.spec == 0 {
			return 0
		}
		if peek(p) == .Ident {
			prm.name = tok(p).text
			p.i += 1
		}
		if accept(p, .L_Bracket) {
			prm.array_size = parse_expr(p)
			expect(p, .R_Bracket)
		}
		append(&params, prm)
		if !accept(p, .Comma) {
			break
		}
	}
	if !expect(p, .R_Paren) {
		return 0
	}
	fn.params = {u32(len(u.params)), u32(len(params))}
	append(&u.params, ..params[:])
	if !accept(p, .Semicolon) {
		if peek(p) != .L_Brace {
			error_at(p, tok(p).pos, .Expected_Body, name.text)
			return 0
		}
		fn.body = parse_block(p)
	}
	id := Func_Decl_Id(len(u.func_decls))
	append(&u.func_decls, fn)
	return id
}

// ---- statements ----------------------------------------------------------------------

@(private = "file")
parse_block :: proc(p: ^Parser) -> Stmt_Id {
	pos := tok(p).pos
	expect(p, .L_Brace)
	body := make([dynamic]u32, context.temp_allocator)
	for peek(p) != .R_Brace && peek(p) != .EOF {
		start := p.i
		st := parse_stmt(p)
		if p.panicking {
			// Resynchronize at the end of the statement.
			for peek(p) != .EOF && peek(p) != .Semicolon && peek(p) != .R_Brace {
				if peek(p) == .L_Brace {
					depth := 0
					for peek(p) != .EOF {
						k := peek(p)
						p.i += 1
						if k == .L_Brace {depth += 1}
						if k == .R_Brace {depth -= 1; if depth == 0 {break}}
					}
					continue
				}
				p.i += 1
			}
			accept(p, .Semicolon)
			p.panicking = false
		} else if st != 0 {
			append(&body, u32(st))
		}
		if p.i == start {
			p.i += 1
		}
	}
	expect(p, .R_Brace)
	return add_stmt(p, Stmt{kind = .Block, pos = pos, body = push_ids(p.u, body[:])})
}

@(private = "file")
starts_declaration :: proc(p: ^Parser) -> bool {
	off := 0
	for is_ident(p, "highp", off) || is_ident(p, "mediump", off) || is_ident(p, "lowp", off) || is_ident(p, "const", off) {
		if is_ident(p, "const", off) {
			return true
		}
		off += 1
	}
	if !is_type_name(p, off) {
		return false
	}
	if is_ident(p, "struct", off) {
		return true
	}
	// `vec3 v` and `float[3] a`, but not `vec3(...)` (a constructor).
	k := peek(p, off + 1)
	if k == .Ident {
		return true
	}
	if k == .L_Bracket {
		j := off + 2
		depth := 1
		for depth > 0 && peek(p, j) != .EOF {
			if peek(p, j) == .L_Bracket {depth += 1}
			if peek(p, j) == .R_Bracket {depth -= 1}
			j += 1
		}
		return peek(p, j) == .Ident
	}
	return false
}

@(private = "file")
parse_stmt :: proc(p: ^Parser) -> Stmt_Id {
	t := tok(p)
	if t.kind == .L_Brace {
		return parse_block(p)
	}
	if t.kind == .Semicolon {
		p.i += 1
		return add_stmt(p, Stmt{kind = .Empty, pos = t.pos})
	}
	if t.kind == .Ident {
		switch t.text {
		case "if":
			p.i += 1
			s := Stmt{kind = .If, pos = t.pos}
			expect(p, .L_Paren)
			s.expr = parse_expr(p)
			expect(p, .R_Paren)
			s.then_ = parse_stmt(p)
			if is_ident(p, "else") {
				p.i += 1
				s.else_ = parse_stmt(p)
			}
			return add_stmt(p, s)
		case "for":
			p.i += 1
			s := Stmt{kind = .For, pos = t.pos}
			expect(p, .L_Paren)
			if !accept(p, .Semicolon) {
				if starts_declaration(p) {
					s.init = parse_decl_stmt(p)
				} else {
					pos := tok(p).pos
					e := parse_expr(p)
					expect(p, .Semicolon)
					s.init = add_stmt(p, Stmt{kind = .Expr, pos = pos, expr = e})
				}
			}
			if peek(p) != .Semicolon {
				s.expr = parse_expr(p)
			}
			expect(p, .Semicolon)
			if peek(p) != .R_Paren {
				s.step = parse_expr(p)
			}
			expect(p, .R_Paren)
			s.then_ = parse_stmt(p)
			return add_stmt(p, s)
		case "while":
			p.i += 1
			s := Stmt{kind = .While, pos = t.pos}
			expect(p, .L_Paren)
			s.expr = parse_expr(p)
			expect(p, .R_Paren)
			s.then_ = parse_stmt(p)
			return add_stmt(p, s)
		case "do":
			p.i += 1
			s := Stmt{kind = .Do_While, pos = t.pos}
			s.then_ = parse_stmt(p)
			if !is_ident(p, "while") {
				error_at(p, tok(p).pos, .Expected_While)
				return 0
			}
			p.i += 1
			expect(p, .L_Paren)
			s.expr = parse_expr(p)
			expect(p, .R_Paren)
			expect(p, .Semicolon)
			return add_stmt(p, s)
		case "break", "continue", "discard":
			// The checker rejects discard where the dialect has none.
			p.i += 1
			kind := Stmt_Kind.Break if t.text == "break" else (.Continue if t.text == "continue" else .Discard)
			expect(p, .Semicolon)
			return add_stmt(p, Stmt{kind = kind, pos = t.pos})
		case "switch":
			p.i += 1
			s := Stmt{kind = .Switch, pos = t.pos}
			expect(p, .L_Paren)
			s.expr = parse_expr(p)
			expect(p, .R_Paren)
			if peek(p) != .L_Brace {
				error_at(p, tok(p).pos, .Expected_Switch_Body, a = token_text(.L_Brace), b = tok(p).text)
				return 0
			}
			// The labels are statements of the body (parse_stmt below).
			block := parse_block(p)
			s.body = p.u.stmts[block].body
			return add_stmt(p, s)
		case "case":
			p.i += 1
			s := Stmt{kind = .Case, pos = t.pos}
			s.expr = parse_ternary(p)
			expect(p, .Colon, .Expected_Case_Colon)
			return add_stmt(p, s)
		case "default":
			p.i += 1
			expect(p, .Colon, .Expected_Default_Colon)
			return add_stmt(p, Stmt{kind = .Default, pos = t.pos})
		case "return":
			p.i += 1
			s := Stmt{kind = .Return, pos = t.pos}
			if peek(p) != .Semicolon {
				s.expr = parse_expr(p)
			}
			expect(p, .Semicolon)
			return add_stmt(p, s)
		case "precision":
			for peek(p) != .Semicolon && peek(p) != .EOF {
				p.i += 1
			}
			accept(p, .Semicolon)
			return add_stmt(p, Stmt{kind = .Empty, pos = t.pos})
		}
	}
	if starts_declaration(p) {
		return parse_decl_stmt(p)
	}
	e := parse_expr(p)
	expect(p, .Semicolon)
	return add_stmt(p, Stmt{kind = .Expr, pos = t.pos, expr = e})
}

@(private = "file")
parse_decl_stmt :: proc(p: ^Parser) -> Stmt_Id {
	pos := tok(p).pos
	is_const := false
	for {
		skip_precision(p)
		if !is_ident(p, "const") {
			break
		}
		is_const = true
		p.i += 1
	}
	spec := parse_type_spec(p)
	if spec == 0 {
		return 0
	}
	if p.u.specs[spec].strukt != 0 {
		error_at(p, pos, .Local_Struct)
		return 0
	}
	decls, ok := parse_declarators(p, spec, is_const, false)
	expect(p, .Semicolon)
	if !ok {
		return 0
	}
	return add_stmt(p, Stmt{kind = .Decl, pos = pos, decls = decls})
}

// ---- expressions ---------------------------------------------------------------------

@(private = "file")
parse_expr :: proc(p: ^Parser) -> Expr_Id {
	e := parse_assignment(p)
	for peek(p) == .Comma {
		t := tok(p)
		p.i += 1
		b := parse_assignment(p)
		e = add_expr(p, Expr{kind = .Comma, pos = t.pos, a = e, b = b})
	}
	return e
}

@(private = "file")
is_assign_op :: proc(k: Token_Kind) -> bool {
	#partial switch k {
	case .Eq, .Plus_Eq, .Minus_Eq, .Star_Eq, .Slash_Eq, .Percent_Eq, .Amp_Eq, .Pipe_Eq, .Caret_Eq, .Shl_Eq, .Shr_Eq:
		return true
	}
	return false
}

@(private = "file")
parse_assignment :: proc(p: ^Parser) -> Expr_Id {
	lhs := parse_ternary(p)
	if is_assign_op(peek(p)) {
		t := tok(p)
		p.i += 1
		rhs := parse_assignment(p)
		return add_expr(p, Expr{kind = .Assign, op = t.kind, pos = t.pos, a = lhs, b = rhs})
	}
	return lhs
}

@(private = "file")
parse_ternary :: proc(p: ^Parser) -> Expr_Id {
	c := parse_binary(p, 1)
	if peek(p) == .Question {
		t := tok(p)
		p.i += 1
		b := parse_expr(p)
		expect(p, .Colon)
		e := parse_assignment(p)
		return add_expr(p, Expr{kind = .Ternary, pos = t.pos, a = c, b = b, c = e})
	}
	return c
}

@(private)
binary_precedence :: proc(k: Token_Kind) -> int {
	#partial switch k {
	case .Or_Or:
		return 1
	case .Xor_Xor:
		return 2
	case .And_And:
		return 3
	case .Pipe:
		return 4
	case .Caret:
		return 5
	case .Amp:
		return 6
	case .Eq_Eq, .Not_Eq:
		return 7
	case .Lt, .Gt, .Lt_Eq, .Gt_Eq:
		return 8
	case .Shl, .Shr:
		return 9
	case .Plus, .Minus:
		return 10
	case .Star, .Slash, .Percent:
		return 11
	}
	return -1
}

@(private = "file")
parse_binary :: proc(p: ^Parser, min_prec: int) -> Expr_Id {
	lhs := parse_unary(p)
	for {
		k := peek(p)
		prec := binary_precedence(k)
		if prec < min_prec {
			return lhs
		}
		t := tok(p)
		p.i += 1
		rhs := parse_binary(p, prec + 1)
		lhs = add_expr(p, Expr{kind = .Binary, op = k, pos = t.pos, a = lhs, b = rhs})
	}
}

@(private = "file")
parse_unary :: proc(p: ^Parser) -> Expr_Id {
	t := tok(p)
	#partial switch t.kind {
	case .Plus, .Minus, .Bang, .Tilde:
		p.i += 1
		a := parse_unary(p)
		return add_expr(p, Expr{kind = .Unary, op = t.kind, pos = t.pos, a = a})
	case .Inc, .Dec:
		p.i += 1
		a := parse_unary(p)
		return add_expr(p, Expr{kind = .Pre_Inc if t.kind == .Inc else .Pre_Dec, pos = t.pos, a = a})
	}
	return parse_postfix(p)
}

@(private = "file")
parse_postfix :: proc(p: ^Parser) -> Expr_Id {
	e := parse_primary(p)
	for e != 0 {
		t := tok(p)
		#partial switch t.kind {
		case .L_Bracket:
			p.i += 1
			ix := parse_expr(p)
			expect(p, .R_Bracket)
			e = add_expr(p, Expr{kind = .Index, pos = t.pos, a = e, b = ix})
		case .Dot:
			p.i += 1
			name := tok(p)
			if !expect(p, .Ident, .Expected_Field) {
				return e
			}
			if name.text == "length" && peek(p) == .L_Paren {
				p.i += 1
				expect(p, .R_Paren)
				e = add_expr(p, Expr{kind = .Length, pos = name.pos, a = e})
				continue
			}
			if name.text == "eval" && peek(p) == .L_Paren {
				// child.eval(coord): a call of the internal `$eval` with the
				// child as its first argument (the checker lowers it).
				p.i += 1
				args := make([dynamic]u32, context.temp_allocator)
				append(&args, u32(e))
				r := parse_args(p)
				append(&args, ..list(p.u, r))
				e = add_expr(p, Expr{kind = .Call, pos = name.pos, name = "$eval", args = push_ids(p.u, args[:])})
				continue
			}
			e = add_expr(p, Expr{kind = .Field, pos = name.pos, a = e, name = name.text})
		case .Inc, .Dec:
			p.i += 1
			e = add_expr(p, Expr{kind = .Post_Inc if t.kind == .Inc else .Post_Dec, pos = t.pos, a = e})
		case:
			return e
		}
	}
	return e
}

@(private = "file")
parse_args :: proc(p: ^Parser) -> Range {
	args := make([dynamic]u32, context.temp_allocator)
	if is_ident(p, "void") && peek(p, 1) == .R_Paren {
		p.i += 1
	}
	for peek(p) != .R_Paren && peek(p) != .EOF {
		append(&args, u32(parse_assignment(p)))
		if !accept(p, .Comma) {
			break
		}
	}
	expect(p, .R_Paren)
	return push_ids(p.u, args[:])
}

@(private = "file")
parse_primary :: proc(p: ^Parser) -> Expr_Id {
	t := tok(p)
	#partial switch t.kind {
	case .Int_Lit, .Uint_Lit:
		p.i += 1
		val, ok := parse_int_text(t.text)
		if !ok {
			error_at(p, t.pos, .Invalid_Int_Literal, a = t.text)
			return 0
		}
		return add_expr(p, Expr{kind = .Int_Lit if t.kind == .Int_Lit else .Uint_Lit, pos = t.pos, value = u64(val)})
	case .Float_Lit:
		p.i += 1
		text := t.text
		if len(text) > 0 && (text[len(text) - 1] == 'f' || text[len(text) - 1] == 'F') {
			text = text[:len(text) - 1]
		}
		v, _ := strconv.parse_f64(text)
		return add_expr(p, Expr{kind = .Float_Lit, pos = t.pos, value = transmute(u64)v})
	case .L_Paren:
		p.i += 1
		e := parse_expr(p)
		expect(p, .R_Paren)
		return e
	case .Ident:
		if t.text == "true" || t.text == "false" {
			p.i += 1
			return add_expr(p, Expr{kind = .Bool_Lit, pos = t.pos, value = u64(t.text == "true")})
		}
		if is_type_name(p) && !is_ident(p, "struct") {
			spec := parse_type_spec(p)
			if spec == 0 {
				return 0
			}
			if !expect(p, .L_Paren, .Expected_Ctor_Paren) {
				return 0
			}
			args := parse_args(p)
			return add_expr(p, Expr{kind = .Call, pos = t.pos, ctor = spec, name = p.u.specs[spec].name, args = args})
		}
		p.i += 1
		if peek(p) == .L_Paren {
			p.i += 1
			args := parse_args(p)
			return add_expr(p, Expr{kind = .Call, pos = t.pos, name = t.text, args = args})
		}
		return add_expr(p, Expr{kind = .Ident, pos = t.pos, name = t.text})
	}
	error_at(p, t.pos, .Expected_Expression, a = t.text)
	return 0
}

// The value of an integer literal (decimal, 0x hex or 0-prefixed octal, an
// optional u suffix), wrapped to 32 bits; `ok` reports whether every digit is
// valid for the base and the value fits in 32 bits.
@(private)
parse_int_text :: proc(text: string) -> (val: u32, ok: bool) {
	s := text
	if len(s) > 0 && (s[len(s) - 1] == 'u' || s[len(s) - 1] == 'U') {
		s = s[:len(s) - 1]
	}
	base: u64 = 10
	if len(s) >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		base = 16
		s = s[2:]
	} else if len(s) > 1 && s[0] == '0' {
		base = 8
		s = s[1:]
	}
	ok = len(s) > 0
	v: u64
	for c in transmute([]u8)s {
		d: u64
		switch c {
		case '0' ..= '9':
			d = u64(c - '0')
		case 'a' ..= 'f':
			d = u64(c - 'a' + 10)
		case 'A' ..= 'F':
			d = u64(c - 'A' + 10)
		case:
			d = base
		}
		if d >= base || v > (u64(max(u32)) - d) / base {
			ok = false
		}
		v = v * base + d
	}
	return u32(v), ok
}
