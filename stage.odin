package oksl

import "core:container/bit_array"

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
stage_entry :: proc(c: ^Checker, main: Func_Id) -> (entry: Func_Id, iface: []Interface_Var) {
	u := c.u
	pos := u.func_decls[u.funcs[main].decl].pos
	b := Node_Builder{u, pos}
	body := make([dynamic]u32, context.temp_allocator)
	rows := make([dynamic]Interface_Var)
	// `$` names are out of the source's reach.
	hidden := make(map[string]Sym_Id, context.temp_allocator)
	if c.rules.entry == .Void_Main {
		stage_var_rows(c, &rows)
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
	append(&stmts, body[len(body) - 1])
	u.stmts[block].body = push_ids(u, stmts[:])

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

