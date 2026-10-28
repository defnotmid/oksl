package oksl

import "core:container/bit_array"
import "core:fmt"

// The stage's interface and entry point. The checker lists what a shader
// exchanges with the pipeline in `Checked.interface` (stage inputs and
// outputs, the uniform block, the channels) and synthesizes the entry point
// as IR, after the profile's own adapter (shadertoy.odin) has made `main`.
// Each backend declares the interface rows and emits the entry like any
// other function; none has stage glue of its own.
//
// The layer profiles (SkSL, Shadertoy) run over a layer whose vertex shader
// writes uv at location 0:
//
//     void $entry() {
//         vec2 coord = $uv * iResolution.xy;         // layer units
//         sk_FragCoord = vec4(coord, 0.0, 1.0);      // if the source reads it
//         <global> = <initializer>;                  // each, in declaration order
//         $color = main(coord);
//     }
//
// (without sk_FragCoord, coord is main's argument expression itself).
//
// A GLSL fragment shader's `void main()` reads and writes its `in` and
// `out` variables and gl_FragCoord itself:
//
//     void $entry() {
//         <global> = <initializer>;
//         main();
//     }
//
// A vertex shader's taps (Options.taps; "taps" below) run between the
// initializers and main: each tap's own initializers, then its copy of main.

Interface_Kind :: enum u8 {
	// A value from the vertex stage, at location `binding`.
	Stage_In,
	// The fragment's window position (gl_FragCoord), origin top-left.
	Frag_Coord,
	// Whether the fragment's triangle faces the camera (gl_FrontFacing).
	Front_Facing,
	// A vertex shader's output or a fragment shader's colour target, at
	// location `binding`.
	Stage_Out,
	// A vertex shader's clip-space position (gl_Position), as written.
	Position,
	// The uniform block: the inputs, then the user uniforms (std140).
	Uniform_Block,
	// A channel's combined image sampler; `binding` is the channel.
	Channel,
}

// The kinds a host binds as resources.
RESOURCE_KINDS :: bit_set[Interface_Kind]{.Uniform_Block, .Channel}

Interface_Var :: struct {
	kind:    Interface_Kind,
	// The variable (0 for the uniform block).
	sym:     Sym_Id,
	// Stage_In / Stage_Out: the location. Resources: the binding (in
	// Metal, the buffer / texture / sampler index).
	binding: i32,
	// Resources: SDL_GPU's descriptor set (SPIR-V).
	set:     i32,
	// Stage_In, Stage_Out: not interpolated.
	flat:    bool,
	// Varyings (a vertex shader's outputs, a fragment shader's inputs): the
	// float offset in the CPU interpreter's per-vertex record (run_vertex,
	// run_fragment): by location, packed. A pipeline's fragment inputs take
	// their vertex outputs'.
	offset:  i32,
}

Stage :: enum u8 {
	Fragment,
	Vertex,
	// Reserved: no profile compiles to it yet.
	Compute,
}

@(rodata)
STAGE_NAMES := [Stage]string {
	.Fragment = "fragment",
	.Vertex   = "vertex",
	.Compute  = "compute",
}

Stage_Rules :: struct {
	// Features the stage takes away, and what using one reports.
	lacks:           Features,
	missing:         [Feature]Error_Code,
	// The locations stage inputs and outputs can use.
	in_locations:    i32,
	out_locations:   i32,
	// SPIR-V: the execution model and the entry's execution modes.
	execution_model: u32,
	execution_modes: []u32,
	// SDL_GPU's descriptor sets for samplers and uniform buffers (SPIR-V).
	sampler_set:     i32,
	uniform_set:     i32,
	// Metal: the entry function's qualifier.
	msl_qualifier:   string,
}

@(rodata)
STAGE_RULES := [Stage]Stage_Rules {
	// OriginUpperLeft: Vulkan's window origin (and Metal's).
	.Fragment = {
		lacks = {.Gl_Position},
		in_locations = 16,
		out_locations = 8,
		execution_model = 4,
		execution_modes = {7},
		sampler_set = 2,
		uniform_set = 3,
		msl_qualifier = "fragment",
	},
	.Vertex = {
		lacks = {.Discard, .Derivatives, .Sk_Frag_Coord, .Gl_Frag_Coord, .Gl_Front_Facing},
		missing = #partial {.Discard = .Fragment_Only, .Derivatives = .Fragment_Only},
		in_locations = 16,
		out_locations = 16,
		execution_model = 0,
		sampler_set = 0,
		uniform_set = 1,
		msl_qualifier = "vertex",
	},
	.Compute = {
		lacks = {.Discard, .Derivatives, .Sk_Frag_Coord, .Gl_Frag_Coord, .Gl_Front_Facing, .Stage_Variables, .Gl_Position},
		execution_model = 5,
		msl_qualifier = "kernel",
	},
}

// An `in` or `out` variable of the source (checked for its type here, its
// location once all are declared).
@(private)
declare_stage_var :: proc(c: ^Checker, v: Var_Id, t: Type) {
	u := c.u
	d := u.vars[v]
	what := "input" if d.io == .In else "output"
	if d.init != 0 {
		add_error(c, d.pos, .Stage_Var_Initializer, d.name, a = what)
	}
	if !is_sv(t) || t.base == .Bool {
		add_error(c, d.pos, .Stage_Var_Type, d.name, a = what, b = tn(c, t))
		return
	}
	// What is interpolated (fragment inputs, vertex outputs) must be
	// float unless flat.
	interpolated := d.io == (.In if c.u.stage == .Fragment else .Out)
	if interpolated && t.base != .Float && !d.flat {
		add_error(c, d.pos, .Flat_Required, d.name, a = what)
	}
	s := Symbol{name = d.name, type = intern(u, t), pos = d.pos, decl = v, index = int(d.location)}
	s.storage = .Stage_In if d.io == .In else .Stage_Out
	s.is_const = d.io == .In
	id := add_sym(c, s)
	u.vars[v].sym = id
	append(&c.stage_vars, id)
	declare(c, id)
}

// GLSL's `void main()`.
@(private)
void_main :: proc(c: ^Checker) -> (entry: Func_Id) {
	u := c.u
	mains := c.by_name["main"]
	if len(mains) == 0 {
		add_error(c, {line = 1, col = 1}, .No_Void_Main)
		return 0
	}
	for f in mains {
		if ty(u, u.funcs[f].ret) == T_VOID && len(fparams(u, f)) == 0 {
			entry = f
		} else {
			add_error(c, u.func_decls[u.funcs[f].decl].pos, .Void_Main_Signature)
		}
	}
	if entry != 0 && !u.funcs[entry].defined {
		add_error(c, u.func_decls[u.funcs[entry].decl].pos, .Entry_No_Body, u.funcs[entry].name)
	}
	return entry
}

// The stage variables' rows: explicit locations as written, the others the
// next free one in declaration order (one output may omit it; several may
// not).
@(private)
stage_var_rows :: proc(c: ^Checker, rows: ^[dynamic]Interface_Var) {
	u := c.u
	stage := STAGE_RULES[c.u.stage]
	if c.gl_frag_coord != 0 {
		append(rows, Interface_Var{kind = .Frag_Coord, sym = c.gl_frag_coord})
	}
	if c.gl_front_facing != 0 {
		append(rows, Interface_Var{kind = .Front_Facing, sym = c.gl_front_facing})
	}
	if c.u.stage == .Vertex {
		// Written or not, a vertex shader outputs its position.
		if c.gl_position == 0 {
			c.gl_position = add_sym(c, Symbol{name = "gl_Position", type = intern(u, vec_type(.Float, 4)), storage = .Stage_Out})
		}
		append(rows, Interface_Var{kind = .Position, sym = c.gl_position})
	}
	outputs := 0
	for s in c.stage_vars {
		outputs += int(u.syms[s].storage == .Stage_Out)
	}
	MAX :: 32
	taken: [Var_Io][MAX]Sym_Id
	io_of :: proc(u: ^Unit, s: Sym_Id) -> Var_Io {
		return .In if u.syms[s].storage == .Stage_In else .Out
	}
	limit :: proc(stage: Stage_Rules, io: Var_Io) -> i32 {
		return stage.in_locations if io == .In else stage.out_locations
	}
	// Inputs the source never reads are left out (a host binds only
	// what is used); outputs stay.
	vars := make([dynamic]Sym_Id, context.temp_allocator)
	for s in c.stage_vars {
		if u.syms[s].storage == .Stage_Out || bit_array.get(&c.used, int(s)) {
			append(&vars, s)
		}
	}
	// Explicit locations first, so implicit ones fill around them.
	for s in vars {
		sym := u.syms[s]
		io := io_of(u, s)
		what := "input" if io == .In else "output"
		if sym.index < 0 {
			if io == .Out && outputs > 1 && c.u.stage == .Fragment {
				add_error(c, sym.pos, .Location_Required, sym.name)
			}
			continue
		}
		if sym.index >= int(limit(stage, io)) {
			add_error(c, sym.pos, .Location_Range, a = what, n = i64(sym.index), m = i64(limit(stage, io) - 1))
			continue
		}
		if other := taken[io][sym.index]; other != 0 {
			add_error(c, sym.pos, .Location_Taken, u.syms[other].name, a = what, n = i64(sym.index))
			continue
		}
		taken[io][sym.index] = s
	}
	for s in vars {
		sym := &u.syms[s]
		io := io_of(u, s)
		if sym.index < 0 {
			for l in 0 ..< int(limit(stage, io)) {
				if taken[io][l] == 0 {
					sym.index = l
					taken[io][l] = s
					break
				}
			}
		}
		if sym.index < 0 {
			add_error(c, sym.pos, .Location_Range, a = "input" if io == .In else "output", n = i64(limit(stage, io)), m = i64(limit(stage, io) - 1))
			continue
		}
		kind := Interface_Kind.Stage_In if io == .In else .Stage_Out
		append(rows, Interface_Var{kind = kind, sym = s, binding = i32(sym.index), flat = u.vars[sym.decl].flat})
	}
	// The varyings' record: what passes between the stages, by location.
	varying := Interface_Kind.Stage_Out if c.u.stage == .Vertex else .Stage_In
	for l in 0 ..< MAX {
		for &row in rows {
			if row.kind == varying && row.binding == i32(l) {
				row.offset = i32(c.varying_floats)
				c.varying_floats += component_count(u, ty(u, u.syms[row.sym].type))
			}
		}
	}
}

// Synthesizes the entry around `main` and returns it with the interface it
// reads and writes.
@(private)
stage_entry :: proc(c: ^Checker, main: Func_Id, funcs: ^[dynamic]Func_Id) -> (entry: Func_Id, iface: []Interface_Var) {
	u := c.u
	pos := u.func_decls[u.funcs[main].decl].pos
	b := Node_Builder{u, pos}
	body := make([dynamic]u32, context.temp_allocator)
	rows := make([dynamic]Interface_Var)
	// `$` names are out of the source's reach.
	hidden := make(map[string]Sym_Id, context.temp_allocator)
	if c.rules.entry == .Void_Main {
		append(&body, u32(stmt_expr(&b, node_call(&b, "main"))))
	} else {
		uv := add_sym(c, Symbol{name = "uv", type = intern(u, vec_type(.Float, 2)), storage = .Stage_In, pos = pos, is_const = true})
		color := add_sym(c, Symbol{name = "color", type = intern(u, vec_type(.Float, 4)), storage = .Stage_Out, pos = pos})
		// The resolution input as the source declared it, or a symbol of
		// its own when the source never names it.
		res := c.input_syms[.Resolution]
		if res == 0 {
			info := INPUTS[.Resolution]
			res = add_sym(c, Symbol{name = info.name, type = intern(u, info.type), storage = .Builtin_Uniform, index = info.offset, is_const = true})
		}
		hidden["$uv"], hidden["$color"], hidden["$resolution"] = uv, color, res
		append(&rows, Interface_Var{kind = .Stage_In, sym = uv, binding = 0})
		append(&rows, Interface_Var{kind = .Stage_Out, sym = color, binding = 0})

		coord := node_binary(&b, .Star, node_ident(&b, "$uv"), node_field(&b, node_ident(&b, "$resolution"), "xy"))
		// A local only when sk_FragCoord reads it too.
		if c.frag_coord != 0 {
			append(&body, u32(stmt_decl(&b, "vec2", "coord", coord)))
			value := node_ctor(&b, "vec4", node_ident(&b, "coord"), node_float(&b, 0), node_float(&b, 1))
			append(&body, u32(stmt_expr(&b, node_assign(&b, node_ident(&b, "sk_FragCoord"), value))))
			coord = node_ident(&b, "coord")
		}
		call := node_call(&b, "main", coord)
		append(&body, u32(stmt_expr(&b, node_assign(&b, node_ident(&b, "$color"), call))))
	}
	block := add_node_stmt(&b, {kind = .Block, body = push_ids(u, body[:])})
	fd := Func_Decl_Id(len(u.func_decls))
	append(&u.func_decls, Func_Decl{pos = pos, name = "$entry", ret = node_spec(&b, "void"), body = block})
	entry = resolve_func_decl(c, fd)
	if entry == 0 {
		return
	}

	// The entry's own read of iResolution is the stage's, not the
	// shader's: inputs_used stays what the source reads.
	push_scope(c)
	for name, s in hidden {
		c.scopes[len(c.scopes) - 1][name] = s
	}
	used := c.inputs_used
	c.entry_writes = true
	check_body(c, entry, fd)
	c.entry_writes = false
	c.inputs_used = used
	pop_scope(c)

	// The global initializers (checked with their declarations) run before
	// main: spliced in ahead of the last statement.
	stmts := make([dynamic]u32, context.temp_allocator)
	append(&stmts, ..body[:len(body) - 1])
	for g in c.globals {
		init := u.vars[u.syms[g].decl].init
		if init == 0 {
			continue
		}
		t := u.syms[g].type
		target := node_expr(&b, {kind = .Ident, name = u.syms[g].name, sym = g, type = t})
		assign := node_expr(&b, {kind = .Assign, op = .Eq, a = target, b = init, type = t})
		append(&stmts, u32(stmt_expr(&b, assign)))
	}
	// The taps run after the initializers, before main. They copy what
	// main reaches (`funcs`, nil when it is unsound) and add the copies.
	if funcs != nil && len(u.taps) > 0 {
		live := tap_live(c, funcs[:], main)
		reached := len(funcs)
		for n in 0 ..< len(u.taps) {
			tap_run(c, &b, &stmts, &live, funcs, reached, main, n)
		}
	}
	append(&stmts, body[len(body) - 1])
	u.stmts[block].body = push_ids(u, stmts[:])

	// After the taps, which read their twins.
	if c.rules.entry == .Void_Main {
		stage_var_rows(c, &rows)
	}
	append_resources(c, &rows)
	return entry, rows[:]
}

// The uniform block and the channels in use, at the stage's sets.
@(private)
append_resources :: proc(c: ^Checker, rows: ^[dynamic]Interface_Var) {
	stage := STAGE_RULES[c.u.stage]
	// No block when there is nothing in it.
	if c.rules.header || len(c.uniforms) > 0 {
		append(rows, Interface_Var{kind = .Uniform_Block, binding = 0, set = stage.uniform_set})
	}
	for ch, i in c.channels {
		if ch != 0 {
			append(rows, Interface_Var{kind = .Channel, sym = ch, binding = i32(i), set = stage.sampler_set})
		}
	}
}

// Compiles a GLSL ES 3.0 fragment shader: `void main()`, `in` and `out`
// variables (`layout(location = N)`), gl_FragCoord, the preprocessor.
compile_glsl :: proc(source: string, options := Options{}) -> (m: ^Module, errors: []Error, ok: bool) {
	o := options
	o.profile = .Glsl
	return compile(source, o)
}


// ---- taps -----------------------------------------------------------------------------
//
// A tap (Options.taps) is a copy of what main does toward gl_Position: of
// the statements of main, of the functions it reaches and of the globals'
// initializers that the position depends on (tap_live), made from the
// checked statements and expressions. In the copy each uniform reads its
// copy in environment N + 1 of the block, the listed attributes read their
// twins, gl_Position is the tap's position, the shader's outputs and
// globals are private globals, and locals and parameters are the copy's
// own. Copies are named "<N + 1>_<name>", which no source name can be.

// What the position depends on, by original symbol, statement and function.
// Flow-insensitive: a symbol is live if a live statement reads it, and a
// statement is live if it writes a live symbol, calls a function that
// does, or decides whether live statements run (the loops, branches,
// breaks and returns around them).
@(private)
Tap_Live :: struct {
	u:       ^Unit,
	syms:    []bool,
	stmts:   []bool,
	// Functions whose run a live statement uses (main's always), and that
	// write a live global or output.
	results: []bool,
	effects: []bool,
	// The function being visited (0: the initializers).
	f:       Func_Id,
	changed: bool,
}

// Visits `funcs` and the initializers until nothing more becomes live.
@(private)
tap_live :: proc(c: ^Checker, funcs: []Func_Id, main: Func_Id) -> Tap_Live {
	u := c.u
	l := Tap_Live {
		u       = u,
		syms    = make([]bool, len(u.syms), context.temp_allocator),
		stmts   = make([]bool, len(u.stmts), context.temp_allocator),
		results = make([]bool, len(u.funcs), context.temp_allocator),
		effects = make([]bool, len(u.funcs), context.temp_allocator),
	}
	l.results[main] = true
	if c.gl_position == 0 {
		return l
	}
	l.syms[c.gl_position] = true
	for l.changed = true; l.changed; {
		l.changed = false
		for f in funcs {
			l.f = f
			live_stmt(&l, u.func_decls[u.funcs[f].decl].body, false, false)
		}
		// An initializer runs when its global is live or it writes one.
		l.f = 0
		for g in c.globals {
			init := u.vars[u.syms[g].decl].init
			if init != 0 && (l.syms[g] || live_writes(&l, init)) {
				live_set(&l, &l.syms[g])
				live_reads(&l, init)
			}
		}
	}
	return l
}

@(private)
live_set :: proc(l: ^Tap_Live, b: ^bool) {
	if !b^ {
		b^ = true
		l.changed = true
	}
}

// Whether statement `id` runs in the copy, marking it and what it reads;
// `breaks` and `continues` say whether the loop or switch they leave does.
// True if anything in it is live.
@(private)
live_stmt :: proc(l: ^Tap_Live, id: Stmt_Id, breaks, continues: bool) -> bool {
	if id == 0 {
		return false
	}
	u := l.u
	s := u.stmts[id]
	live := false
	switch s.kind {
	case .Empty, .Case, .Default:
	case .Block:
		for b in list(u, s.body) {
			live |= live_stmt(l, Stmt_Id(b), breaks, continues)
		}
	case .Decl:
		for d in list(u, s.decls) {
			live |= l.syms[u.vars[d].sym] || live_writes(l, u.vars[d].init)
		}
	case .Expr:
		live = live_writes(l, s.expr)
	case .If:
		then_ := live_stmt(l, s.then_, breaks, continues)
		else_ := live_stmt(l, s.else_, breaks, continues)
		live = then_ || else_ || live_writes(l, s.expr)
	case .For, .While, .Do_While:
		// Its jumps matter once the loop runs (an earlier visit found it).
		body := live_stmt(l, s.then_, l.stmts[id], l.stmts[id])
		init := live_stmt(l, s.init, breaks, continues)
		live = body || init || live_writes(l, s.expr) || live_writes(l, s.step)
	case .Switch:
		for b in list(u, s.body) {
			live |= live_stmt(l, Stmt_Id(b), l.stmts[id], continues)
		}
		live ||= live_writes(l, s.expr)
	case .Break:
		live = breaks
	case .Continue:
		live = continues
	case .Return, .Discard:
		live = l.results[l.f]
	}
	if live {
		live_set(l, &l.stmts[id])
		live_reads(l, s.expr)
		live_reads(l, s.step)
		for d in list(u, s.decls) {
			live_reads(l, u.vars[d].init)
		}
	}
	return live
}

// Whether evaluating `id` writes a live symbol, directly or by a call. A
// live global or output written makes the function one with effects.
@(private)
live_writes :: proc(l: ^Tap_Live, id: Expr_Id) -> bool {
	if id == 0 {
		return false
	}
	u := l.u
	e := u.exprs[id]
	w := live_writes(l, e.a) | live_writes(l, e.b) | live_writes(l, e.c)
	for a in args_of(u, id) {
		w |= live_writes(l, a)
	}
	writes :: proc(l: ^Tap_Live, target: Expr_Id) -> bool {
		s := target_sym(l.u, target)
		if s == 0 || !l.syms[s] {
			return false
		}
		if storage := l.u.syms[s].storage; storage == .Global || storage == .Stage_Out {
			live_set(l, &l.effects[l.f])
		}
		return true
	}
	#partial switch e.kind {
	case .Assign, .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
		w |= writes(l, e.a)
	case .Call:
		if e.call != .User {
			break
		}
		if l.effects[e.func] {
			live_set(l, &l.effects[l.f])
			w = true
		}
		ps := fparams(u, e.func)
		for a, i in args_of(u, id) {
			if ps[i].qual != .In {
				w |= writes(l, a)
			}
		}
	}
	return w
}

// Marks what evaluating `id` reads live: its symbols, the results of the
// functions it calls and their `out` parameters.
@(private)
live_reads :: proc(l: ^Tap_Live, id: Expr_Id) {
	if id == 0 {
		return
	}
	u := l.u
	e := u.exprs[id]
	live_reads(l, e.a)
	live_reads(l, e.b)
	live_reads(l, e.c)
	for a in args_of(u, id) {
		live_reads(l, a)
	}
	if e.sym != 0 {
		live_set(l, &l.syms[e.sym])
	}
	if e.kind == .Call && e.call == .User {
		live_set(l, &l.results[e.func])
		for p in fparams(u, e.func) {
			if p.qual != .In && p.sym != 0 {
				live_set(l, &l.syms[p.sym])
			}
		}
	}
}

@(private)
Tap_Copy :: struct {
	c:     ^Checker,
	live:  ^Tap_Live,
	n:     int,
	// Each original symbol's in the copy (0: not made yet) and each copied
	// function's copy.
	syms:  []Sym_Id,
	funcs: []Func_Id,
}

// Appends tap `n` to the entry's statements (its live globals'
// initializers, then its copy of main) and the copies of the functions in
// `funcs[:reached]` that it calls to `funcs`.
@(private)
tap_run :: proc(c: ^Checker, b: ^Node_Builder, stmts: ^[dynamic]u32, live: ^Tap_Live, funcs: ^[dynamic]Func_Id, reached: int, main: Func_Id, n: int) {
	u := c.u
	t := Tap_Copy{c = c, live = live, n = n, syms = make([]Sym_Id, len(u.syms), context.temp_allocator), funcs = make([]Func_Id, len(u.funcs), context.temp_allocator)}
	if !tap_bind(&t, u.taps[n]) {
		return
	}
	// Every copy exists before any body is copied, so calls can name them.
	for i in 0 ..< reached {
		f := funcs[i]
		if !live.results[f] {
			continue
		}
		fn := u.funcs[f]
		params := make([]Func_Param, fn.params.count, context.temp_allocator)
		copy(params, fparams(u, f))
		for &p in params {
			p.sym = tap_sym(&t, p.sym)
		}
		fn.name = tap_name(&t, fn.name)
		fn.params = {u32(len(u.func_params)), fn.params.count}
		append(&u.func_params, ..params)
		fd := u.func_decls[fn.decl]
		fd.name = fn.name
		fn.decl = Func_Decl_Id(len(u.func_decls))
		append(&u.func_decls, fd)
		t.funcs[f] = Func_Id(len(u.funcs))
		append(&u.funcs, fn)
		append(funcs, t.funcs[f])
	}
	for i in 0 ..< reached {
		f := funcs[i]
		if t.funcs[f] == 0 {
			continue
		}
		body := tap_stmt(&t, u.func_decls[u.funcs[f].decl].body)
		if body == 0 {
			body = add_node_stmt(b, {kind = .Block})
		}
		u.func_decls[u.funcs[t.funcs[f]].decl].body = body
	}
	// Copying an initializer can add private globals: the originals'.
	for i, globals := 0, len(c.globals); i < globals; i += 1 {
		g := c.globals[i]
		init := u.vars[u.syms[g].decl].init
		if init == 0 || !live.syms[g] {
			continue
		}
		target := node_expr(b, {kind = .Ident, name = u.syms[g].name, sym = tap_sym(&t, g), type = u.syms[g].type})
		assign := node_expr(b, {kind = .Assign, op = .Eq, a = target, b = tap_expr(&t, init), type = u.syms[g].type})
		append(stmts, u32(stmt_expr(b, assign)))
	}
	run := t.funcs[main]
	call := node_expr(b, {kind = .Call, call = .User, name = u.funcs[run].name, func = run, type = u.funcs[main].ret})
	append(stmts, u32(stmt_expr(b, call)))
}

// Binds the symbols a tap replaces up front: the uniforms (copied in block
// order), the twins and the position. False when the tap names them wrong.
@(private)
tap_bind :: proc(t: ^Tap_Copy, tap: Tap) -> bool {
	c := t.c
	u := c.u
	error_count := len(u.errors)
	declared :: proc(c: ^Checker, name: string, storage: Storage) -> (s: Sym_Id, pos: Pos) {
		s = c.scopes[0][name]
		if s == 0 {
			return 0, {line = 1, col = 1}
		}
		return s if c.u.syms[s].storage == storage else 0, c.u.syms[s].pos
	}
	pos, at := declared(c, tap.position, .Stage_Out)
	switch {
	case pos == 0 || ty(u, u.syms[pos].type) != vec_type(.Float, 4):
		add_error(c, at, .Tap_Position, tap.position)
	case bit_array.get(&c.used, int(pos)):
		add_error(c, at, .Tap_Position_Used, tap.position)
	case c.gl_position != 0:
		t.syms[c.gl_position] = pos
	}
	for other in u.taps[:t.n] {
		if other.position == tap.position {
			add_error(c, at, .Tap_Position_Taken, tap.position)
			break
		}
	}
	for pair in tap.attributes {
		attr, attr_at := declared(c, pair[0], .Stage_In)
		if attr == 0 {
			add_error(c, attr_at, .Tap_Attribute, pair[0])
			continue
		}
		twin, twin_at := declared(c, pair[1], .Stage_In)
		if twin == 0 || u.syms[twin].type != u.syms[attr].type {
			add_error(c, twin_at, .Tap_Twin, pair[1], a = tn(c, sym_type(u, attr)), b = pair[0])
			continue
		}
		t.syms[attr] = twin
	}
	if len(u.errors) > error_count {
		return false
	}
	// The first environment's uniforms lead the list; earlier taps' copies
	// follow.
	stride := round_up(c.next_offset, 16)
	for i, uniforms := 0, len(c.uniforms); i < uniforms; i += 1 {
		s := c.uniforms[i]
		if u.syms[s].index >= stride {
			break
		}
		dup := u.syms[s]
		dup.name = tap_name(t, dup.name)
		dup.index += (t.n + 1) * stride
		t.syms[s] = add_sym(c, dup)
		append(&c.uniforms, t.syms[s])
	}
	return true
}

// A symbol's in the copy: made on first use for locals and parameters (the
// copy's own) and for outputs and globals (private globals); the rest are
// shared.
@(private)
tap_sym :: proc(t: ^Tap_Copy, s: Sym_Id) -> Sym_Id {
	if s == 0 || t.syms[s] != 0 {
		return t.syms[s]
	}
	sym := t.c.u.syms[s]
	#partial switch sym.storage {
	case .Local, .Param:
		t.syms[s] = add_sym(t.c, sym)
	case .Global, .Stage_Out:
		sym.name = tap_name(t, sym.name)
		sym.storage = .Global
		sym.decl = 0
		t.syms[s] = add_sym(t.c, sym)
		append(&t.c.globals, t.syms[s])
	case:
		t.syms[s] = s
	}
	return t.syms[s]
}

@(private)
tap_name :: proc(t: ^Tap_Copy, name: string) -> string {
	return fmt.aprintf("%d_%s", t.n + 1, name)
}

// Copies statement `id` if the position needs it (0 if not); labels and
// empty statements go with their parent.
@(private)
tap_stmt :: proc(t: ^Tap_Copy, id: Stmt_Id) -> Stmt_Id {
	if id == 0 {
		return 0
	}
	u := t.c.u
	s := u.stmts[id]
	if s.kind != .Case && s.kind != .Default && s.kind != .Empty && !t.live.stmts[id] {
		return 0
	}
	if s.body.count > 0 {
		body := make([dynamic]u32, 0, s.body.count, context.temp_allocator)
		for b in list(u, s.body) {
			if k := tap_stmt(t, Stmt_Id(b)); k != 0 {
				append(&body, u32(k))
			}
		}
		s.body = push_ids(u, body[:])
	}
	if s.decls.count > 0 {
		decls := make([]u32, s.decls.count, context.temp_allocator)
		copy(decls, list(u, s.decls))
		for &d in decls {
			v := u.vars[d]
			v.init = tap_expr(t, v.init)
			v.sym = tap_sym(t, v.sym)
			d = u32(len(u.vars))
			u.syms[v.sym].decl = Var_Id(d)
			append(&u.vars, v)
		}
		s.decls = push_ids(u, decls)
	}
	s.expr, s.step = tap_expr(t, s.expr), tap_expr(t, s.step)
	s.init, s.else_ = tap_stmt(t, s.init), tap_stmt(t, s.else_)
	if s.then_ != 0 {
		// A branch or loop body with nothing live runs nothing.
		then_ := tap_stmt(t, s.then_)
		if then_ == 0 {
			then_ = Stmt_Id(len(u.stmts))
			append(&u.stmts, Stmt{kind = .Empty, pos = s.pos})
		}
		s.then_ = then_
	}
	out := Stmt_Id(len(u.stmts))
	append(&u.stmts, s)
	return out
}

@(private)
tap_expr :: proc(t: ^Tap_Copy, id: Expr_Id) -> Expr_Id {
	if id == 0 {
		return 0
	}
	u := t.c.u
	e := u.exprs[id]
	e.a, e.b, e.c = tap_expr(t, e.a), tap_expr(t, e.b), tap_expr(t, e.c)
	if e.sym != 0 {
		// A twin is an input once the copy reads it.
		s := tap_sym(t, e.sym)
		if s != e.sym && u.syms[s].storage == .Stage_In {
			bit_array.set(&t.c.used, int(s))
		}
		e.sym = s
	}
	if e.call == .User {
		e.func = t.funcs[e.func]
	}
	if e.args.count > 0 {
		args := make([]u32, e.args.count, context.temp_allocator)
		copy(args, list(u, e.args))
		for &a in args {
			a = u32(tap_expr(t, Expr_Id(a)))
		}
		e.args = push_ids(u, args)
	}
	out := Expr_Id(len(u.exprs))
	append(&u.exprs, e)
	return out
}
