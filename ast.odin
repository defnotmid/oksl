package oksl

// The compilation unit: every syntax node, symbol, type and function lives
// in a flat array of plain structs, referenced by integer ids. Index 0 of
// each array is a reserved "none" entry. Children that come in lists
// (arguments, block statements, declarators) are ranges into `ids`.
//
// The checker annotates expressions in place (type, symbol, call target,
// swizzle) and adds `Convert` expressions for implicit conversions, so the
// backends see explicit types everywhere.

Expr_Id :: distinct u32
Stmt_Id :: distinct u32
Var_Id :: distinct u32
Spec_Id :: distinct u32
Func_Decl_Id :: distinct u32
Struct_Decl_Id :: distinct u32
Buffer_Decl_Id :: distinct u32
Sym_Id :: distinct u32
Func_Id :: distinct u32
Type_Id :: distinct u32

// `[start, start + count)` into `Unit.ids` (or another array, as noted).
Range :: struct {
	start, count: u32,
}

Expr_Kind :: enum u8 {
	None,
	Int_Lit,
	Uint_Lit,
	Float_Lit,
	Bool_Lit,
	Ident,
	// a op b
	Binary,
	// op a (Minus, Plus, Bang, Tilde)
	Unary,
	Pre_Inc,
	Pre_Dec,
	Post_Inc,
	Post_Dec,
	// a op= b (op is Eq for plain assignment)
	Assign,
	// a ? b : c
	Ternary,
	// name(args), or a constructor when `ctor` is set
	Call,
	// a[b]
	Index,
	// a.name: struct member or swizzle
	Field,
	// a.length()
	Length,
	// a, b
	Comma,
	// Checker-added conversion of a to the expression's type.
	Convert,
	// Checker-made from a[b] on a buffer: element b of buffer `sym`.
	Buffer_Load,
}

Call_Kind :: enum u8 {
	None,
	User,
	Builtin,
	Constructor,
}

Expr :: struct {
	kind:        Expr_Kind,
	op:          Token_Kind,
	call:        Call_Kind,
	builtin:     Builtin,
	swizzle_len: u8,
	swizzle:     [4]u8,
	pos:         Pos,
	a, b, c:     Expr_Id,
	// Call arguments (Expr_Ids).
	args:        Range,
	ctor:        Spec_Id,
	// Literal bits: integers as is, floats as f64 bits, bools 0/1.
	value:       u64,
	name:        string,
	// Set by the checker.
	type:        Type_Id,
	sym:         Sym_Id,
	func:        Func_Id,
	field:       i32,
}

Type_Spec :: struct {
	pos:        Pos,
	name:       string,
	// `T[n]` in the type itself (`float[3] a` or `float[3](...)`).
	array_size: Expr_Id,
	unsized:    bool,
	// An inline `struct { ... }` definition.
	strukt:     Struct_Decl_Id,
}

Stmt_Kind :: enum u8 {
	Empty,
	Block,
	Decl,
	Expr,
	If,
	For,
	While,
	Do_While,
	Break,
	Continue,
	Return,
	Discard,
	// switch (expr) { body }: the body is a flat statement list in which
	// Case and Default are labels; control falls through to the next one.
	Switch,
	// case expr: (the checker folds expr into `value`).
	Case,
	Default,
}

Stmt :: struct {
	kind:  Stmt_Kind,
	pos:   Pos,
	// Block statements (Stmt_Ids).
	body:  Range,
	// Declarators (Var_Ids).
	decls: Range,
	// Expression, condition or return value.
	expr:  Expr_Id,
	init:  Stmt_Id,
	step:  Expr_Id,
	then_: Stmt_Id,
	else_: Stmt_Id,
	// Case: the label's value (set by the checker).
	value: i64,
}

Var_Decl :: struct {
	pos:        Pos,
	name:       string,
	spec:       Spec_Id,
	// `name[n]` after the declarator.
	array_size: Expr_Id,
	// A stage variable (`in`, `out`), its `layout(location = N)` (-1 when
	// absent) and `flat`.
	location:   i32,
	init:       Expr_Id,
	sym:        Sym_Id,
	unsized:    bool,
	is_const:   bool,
	is_uniform: bool,
	io:         Var_Io,
	flat:       bool,
	// Storage image qualifiers (`layout(rgba8)`, `readonly`, `writeonly`).
	readonly:      bool,
	writeonly:     bool,
	bad_layout:    bool,
	has_location:  bool,
	location_uint: bool,
	format:        Image_Format,
}

Var_Io :: enum u8 {
	None,
	In,
	Out,
}

Param_Qual :: enum u8 {
	In,
	Out,
	Inout,
}

Param :: struct {
	pos:        Pos,
	name:       string,
	spec:       Spec_Id,
	array_size: Expr_Id,
	qual:       Param_Qual,
	is_const:   bool,
	readonly:   bool,
	writeonly:  bool,
}

Func_Decl :: struct {
	pos:    Pos,
	name:   string,
	ret:    Spec_Id,
	// Into `Unit.params`.
	params: Range,
	// 0 for a prototype.
	body:   Stmt_Id,
}

Struct_Decl :: struct {
	pos:    Pos,
	name:   string,
	// Fields as Var_Decls, into `Unit.vars` directly.
	fields: Range,
}

// `readonly buffer Block { T name[]; } instance;`: its members are
// Var_Decls in `Unit.vars` (the checker takes one runtime-sized array).
Buffer_Decl :: struct {
	pos:        Pos,
	block:      string,
	// "" when the members are global names.
	instance:   string,
	// layout(...)'s packing ("" when none is given).
	packing:    string,
	readonly:   bool,
	writeonly:  bool,
	bad_layout: bool,
	// Into `Unit.vars` directly.
	members:    Range,
}

// One item of `layout(local_size_x = X, ...) in;` (`axis` 0..2 for x..z).
Workgroup_Item :: struct {
	pos:   Pos,
	value: i64,
	axis:  i8,
}

Top_Kind :: enum u8 {
	Func,
	Vars,
	Struct,
	Buffer,
	Workgroup,
}

Top :: struct {
	kind:   Top_Kind,
	fn:     Func_Decl_Id,
	strukt: Struct_Decl_Id,
	buffer: Buffer_Decl_Id,
	// Vars: declarators (Var_Ids in Unit.ids). Workgroup: into Unit.workgroups.
	vars:   Range,
}

#assert(size_of(Var_Decl) <= 64)
#assert(size_of(Param) <= 48)
#assert(size_of(Buffer_Decl) <= 80)
#assert(size_of(Top) <= 24)

// ---- semantic tables ------------------------------------------------------------------

Storage :: enum u8 {
	Local,
	Param,
	// Private global variable (initialized at entry).
	Global,
	Uniform,
	// A uniform input (iTime, iResolution, ...: INPUTS) in the block.
	Builtin_Uniform,
	// A `uniform shader` child or a sampler (index: its channel).
	Channel,
	// A storage buffer's runtime-sized array, typed as its element (index:
	// the buffer's place among Module.storage_buffers).
	Buffer,
	// The stage's inputs and outputs (Checked.interface).
	Stage_In,
	Stage_Out,
	// A `uniform image2D` / `uimage2D` / `iimage2D` (index: place among
	// Module.storage_images).
	Storage_Image,
	// Compute's `gl_WorkGroupSize` (`const uvec3`).
	Work_Group_Size,
}

Symbol :: struct {
	name:     string,
	type:     Type_Id,
	storage:  Storage,
	qual:     Param_Qual,
	is_const: bool,
	const_ok: bool,
	pos:      Pos,
	// Uniform/Builtin_Uniform: byte offset in the block. Channel: channel
	// index.
	index:    int,
	// Folded 32-bit pattern of a scalar int, uint or bool constant.
	const_bits: u32,
	decl:       Var_Id,
}

Func :: struct {
	name:         string,
	ret:          Type_Id,
	// Into `Unit.func_params`.
	params:       Range,
	decl:         Func_Decl_Id,
	defined:      bool,
	writes_state: bool,
}

Func_Param :: struct {
	name: string,
	type: Type_Id,
	qual: Param_Qual,
	sym:  Sym_Id,
}

Struct_Field :: struct {
	name: string,
	type: Type_Id,
}

Struct_Info :: struct {
	name:   string,
	// Into `Unit.fields`.
	fields: Range,
}

Unit :: struct {
	allocator:    Allocator,
	profile:      Profile,
	stage:        Stage,
	// The profile's dialect and the features its source can use.
	dialect:      Dialect,
	features:     Features,
	// A Shadertoy buffer pass: the entry adapter keeps mainImage's colour.
	buffer_pass:  bool,
	// A vertex shader's taps (Options.taps): the entry runs a copy of main
	// for each.
	taps:         []Tap,
	errors:       [dynamic]Error,
	// Syntax.
	exprs:        [dynamic]Expr,
	stmts:        [dynamic]Stmt,
	vars:         [dynamic]Var_Decl,
	specs:        [dynamic]Type_Spec,
	params:       [dynamic]Param,
	func_decls:   [dynamic]Func_Decl,
	struct_decls: [dynamic]Struct_Decl,
	buffer_decls: [dynamic]Buffer_Decl,
	workgroups:   [dynamic]Workgroup_Item,
	tops:         [dynamic]Top,
	ids:          [dynamic]u32,
	// Semantics.
	types:        [dynamic]Type,
	type_index:   map[Type]Type_Id,
	structs:      [dynamic]Struct_Info,
	fields:       [dynamic]Struct_Field,
	syms:         [dynamic]Symbol,
	funcs:        [dynamic]Func,
	func_params:  [dynamic]Func_Param,
	// (caller, callee); caller 0 is a global initializer.
	calls:        [dynamic][2]Func_Id,
}

unit_init :: proc(u: ^Unit, profile: Profile, allocator: Allocator, buffer_pass := false, stage := Stage.Fragment, taps: []Tap = nil) {
	context.allocator = allocator
	u.allocator = allocator
	u.profile = profile
	u.dialect = PROFILE_RULES[profile].dialect
	u.stage = stage
	u.features = profile_features(profile, stage)
	u.buffer_pass = buffer_pass
	u.taps = taps
	u.errors = make([dynamic]Error)
	u.exprs = make([dynamic]Expr)
	u.stmts = make([dynamic]Stmt)
	u.vars = make([dynamic]Var_Decl)
	u.specs = make([dynamic]Type_Spec)
	u.params = make([dynamic]Param)
	u.func_decls = make([dynamic]Func_Decl)
	u.struct_decls = make([dynamic]Struct_Decl)
	u.buffer_decls = make([dynamic]Buffer_Decl)
	u.workgroups = make([dynamic]Workgroup_Item)
	u.tops = make([dynamic]Top)
	u.ids = make([dynamic]u32)
	u.types = make([dynamic]Type)
	u.type_index = make(map[Type]Type_Id)
	u.structs = make([dynamic]Struct_Info)
	u.fields = make([dynamic]Struct_Field)
	u.syms = make([dynamic]Symbol)
	u.funcs = make([dynamic]Func)
	u.func_params = make([dynamic]Func_Param)
	u.calls = make([dynamic][2]Func_Id)
	// Reserved "none" entries.
	append(&u.exprs, Expr{})
	append(&u.stmts, Stmt{})
	append(&u.vars, Var_Decl{})
	append(&u.specs, Type_Spec{})
	append(&u.func_decls, Func_Decl{})
	append(&u.struct_decls, Struct_Decl{})
	append(&u.syms, Symbol{})
	append(&u.funcs, Func{})
	// Type 0 is the error type.
	append(&u.types, T_ERROR)
	u.type_index[T_ERROR] = 0
}

// Replaces the syntax of a fresh `dst` with a copy of `src`'s (a program's
// Common unit, parsed once, under each pass). `src` has no semantics yet.
unit_copy_syntax :: proc(dst, src: ^Unit) {
	copy_array :: proc(dst: ^[dynamic]$T, src: [dynamic]T) {
		clear(dst)
		append(dst, ..src[:])
	}
	copy_array(&dst.exprs, src.exprs)
	copy_array(&dst.stmts, src.stmts)
	copy_array(&dst.vars, src.vars)
	copy_array(&dst.specs, src.specs)
	copy_array(&dst.params, src.params)
	copy_array(&dst.func_decls, src.func_decls)
	copy_array(&dst.struct_decls, src.struct_decls)
	copy_array(&dst.buffer_decls, src.buffer_decls)
	copy_array(&dst.workgroups, src.workgroups)
	copy_array(&dst.tops, src.tops)
	copy_array(&dst.ids, src.ids)
}

// The id of `t` in the interned type table.
intern :: proc(u: ^Unit, t: Type) -> Type_Id {
	if id, ok := u.type_index[t]; ok {
		return id
	}
	id := Type_Id(len(u.types))
	append(&u.types, t)
	u.type_index[t] = id
	return id
}

ty :: #force_inline proc "contextless" (u: ^Unit, id: Type_Id) -> Type {
	return u.types[id]
}

expr_type :: #force_inline proc "contextless" (u: ^Unit, id: Expr_Id) -> Type {
	return u.types[u.exprs[id].type]
}

sym_type :: #force_inline proc "contextless" (u: ^Unit, id: Sym_Id) -> Type {
	return u.types[u.syms[id].type]
}

list :: #force_inline proc "contextless" (u: ^Unit, r: Range) -> []u32 {
	return u.ids[r.start:][:r.count]
}

args_of :: #force_inline proc "contextless" (u: ^Unit, id: Expr_Id) -> []Expr_Id {
	r := u.exprs[id].args
	return transmute([]Expr_Id)u.ids[r.start:][:r.count]
}

fparams :: #force_inline proc "contextless" (u: ^Unit, f: Func_Id) -> []Func_Param {
	r := u.funcs[f].params
	return u.func_params[r.start:][:r.count]
}

struct_fields :: #force_inline proc "contextless" (u: ^Unit, s: i32) -> []Struct_Field {
	r := u.structs[s].fields
	return u.fields[r.start:][:r.count]
}

// Appends child ids and returns their range.
push_ids :: proc(u: ^Unit, ids: []u32) -> Range {
	r := Range{u32(len(u.ids)), u32(len(ids))}
	append(&u.ids, ..ids)
	return r
}

// The variable an assignment target or `out` argument writes: `v`, `v[i]`
// and `v.xy` write v (0 for any other expression).
target_sym :: proc "contextless" (u: ^Unit, id: Expr_Id) -> Sym_Id {
	id := id
	for u.exprs[id].kind == .Index || u.exprs[id].kind == .Field {
		id = u.exprs[id].a
	}
	return u.exprs[id].sym if u.exprs[id].kind == .Ident else 0
}

// The root Buffer_Load expression and buffer symbol of a buffer element
// target (`buf[i]`, `buf[i].field`, `buf[i].v.xy`, `buf[i].v[k]`), or 0.
target_buffer :: proc "contextless" (u: ^Unit, id: Expr_Id) -> (Expr_Id, Sym_Id) {
	id := id
	for u.exprs[id].kind == .Index || u.exprs[id].kind == .Field {
		id = u.exprs[id].a
	}
	if u.exprs[id].kind == .Buffer_Load {
		return id, u.exprs[id].sym
	}
	return 0, 0
}

// Decomposed storage-buffer lvalue: element `elem_idx` of `sym` (`stride_words`
// words per element), plus static `word_off` words, addressing either a whole
// `type` (`swz_len == 0 && dyn_comp == 0`), a multi-component vector swizzle
// (`swz[0 ..< swz_len]`), or a dynamically indexed vector component (`dyn_comp`).
Buf_Path :: struct {
	sym:          Sym_Id,
	elem_idx:     Expr_Id,
	stride_words: int,
	word_off:     int,
	type:         Type,
	swz:          [4]u8,
	swz_len:      int,
	dyn_comp:     Expr_Id,
	dyn_rows:     int,
	dyn_swz:      [4]u8,
	dyn_swz_len:  int,
}

buf_path :: proc(u: ^Unit, id: Expr_Id) -> (p: Buf_Path, ok: bool) {
	steps := make([dynamic]Expr_Id, 0, 8, context.temp_allocator)
	cur := id
	for cur != 0 {
		ce := u.exprs[cur]
		if ce.kind == .Buffer_Load {
			p.sym = ce.sym
			p.elem_idx = ce.b
			et := ty(u, ce.type)
			p.stride_words = std430_stride(u, et) / 4
			p.type = et
			ok = true
			break
		}
		if ce.kind == .Field || ce.kind == .Index {
			append(&steps, cur)
			cur = ce.a
		} else {
			return {}, false
		}
	}
	if !ok {
		return {}, false
	}
	#reverse for step in steps {
		se := u.exprs[step]
		#partial switch se.kind {
		case .Field:
			if se.swizzle_len == 0 {
				p.word_off += std430_field_offset(u, p.type.strukt, se.field) / 4
				p.type = ty(u, se.type)
			} else if is_scalar(p.type) {
				p.type = ty(u, se.type)
			} else if se.swizzle_len == 1 {
				c := int(se.swizzle[0])
				p.word_off += int(p.swz[c]) if p.swz_len > 0 else c
				p.swz_len = 0
				p.type = ty(u, se.type)
			} else {
				if p.swz_len == 0 {
					p.swz = se.swizzle
				} else {
					composed: [4]u8
					for k in 0 ..< int(se.swizzle_len) {
						composed[k] = p.swz[se.swizzle[k]]
					}
					p.swz = composed
				}
				p.swz_len = int(se.swizzle_len)
				p.type = ty(u, se.type)
			}
		case .Index:
			ix := u.exprs[se.b]
			if ix.kind == .Int_Lit || ix.kind == .Uint_Lit {
				c := clamp(int(ix.value), 0, int(p.type.rows) - 1)
				p.word_off += int(p.swz[c]) if p.swz_len > 0 else c
				p.swz_len = 0
			} else {
				p.dyn_comp = se.b
				p.dyn_rows = int(p.type.rows)
				p.dyn_swz = p.swz
				p.dyn_swz_len = p.swz_len
				p.swz_len = 0
			}
			p.type = ty(u, se.type)
		}
	}
	return p, true
}

lit_f64 :: #force_inline proc "contextless" (e: Expr) -> f64 {
	return transmute(f64)e.value
}

// True when evaluating `id` may mutate a variable, global, storage buffer or
// storage image (assignment, ++/--, imageStore, or a user call with out/inout
// parameters or side effects).
may_write :: proc "contextless" (u: ^Unit, id: Expr_Id) -> bool {
	if id == 0 {
		return false
	}
	e := u.exprs[id]
	#partial switch e.kind {
	case .Assign, .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
		return true
	case .Unary, .Convert, .Field, .Length:
		return may_write(u, e.a)
	case .Binary, .Index, .Comma, .Buffer_Load:
		return may_write(u, e.a) || may_write(u, e.b)
	case .Ternary:
		return may_write(u, e.a) || may_write(u, e.b) || may_write(u, e.c)
	case .Call:
		if e.call == .Builtin && e.builtin == .Image_Store {
			return true
		}
		if e.call == .User {
			fn := u.funcs[e.func]
			if fn.writes_state {
				return true
			}
			for p in u.func_params[fn.params.start:][:fn.params.count] {
				if p.qual != .In {
					return true
				}
			}
		}
		for i in u.ids[e.args.start:][:e.args.count] {
			if may_write(u, Expr_Id(i)) {
				return true
			}
		}
	}
	return false
}

// ---- traversal ------------------------------------------------------------------------

// Calls `f` on every expression under statement `s` (pre-order).
walk_stmt :: proc(u: ^Unit, s: Stmt_Id, f: proc(u: ^Unit, e: Expr_Id, user: rawptr), user: rawptr) {
	if s == 0 {
		return
	}
	st := u.stmts[s]
	for b in list(u, st.body) {
		walk_stmt(u, Stmt_Id(b), f, user)
	}
	for d in list(u, st.decls) {
		walk_expr(u, u.vars[d].init, f, user)
	}
	walk_expr(u, st.expr, f, user)
	walk_stmt(u, st.init, f, user)
	walk_expr(u, st.step, f, user)
	walk_stmt(u, st.then_, f, user)
	walk_stmt(u, st.else_, f, user)
}

walk_expr :: proc(u: ^Unit, e: Expr_Id, f: proc(u: ^Unit, e: Expr_Id, user: rawptr), user: rawptr) {
	if e == 0 {
		return
	}
	f(u, e, user)
	ex := u.exprs[e]
	walk_expr(u, ex.a, f, user)
	walk_expr(u, ex.b, f, user)
	walk_expr(u, ex.c, f, user)
	for a in list(u, ex.args) {
		walk_expr(u, Expr_Id(a), f, user)
	}
}
