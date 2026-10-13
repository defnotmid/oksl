package oksl

// Shadertoy's entry adapter. GLSL code defines `void mainImage(out vec4
// fragColor, in vec2 fragCoord)`; the backends call `vec4 main(vec2 coord)`
// with the layer position (origin top-left). The adapter is synthesized as
// Unit nodes after the user's code is checked, then checked like any other
// function, so it reaches every backend as ordinary IR:
//
//     vec4 main(vec2 coord) {
//         vec2 frag = vec2(coord.x, iResolution.y - coord.y);  // bottom-left origin
//         gl_FragCoord = vec4(frag, 0.0, 1.0);                 // if the source reads it
//         vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
//         mainImage(color, frag);
//         return vec4(clamp(color.rgb, 0.0, 1.0), 1.0);         // opaque, clamped
//     }
//
// A buffer pass of a program (program.odin) returns `color` as it is:
// unclamped, with its alpha, for the float textures buffers render into.
//
// The Shadertoy inputs (iResolution, iTime, ..., iChannel0..3) are the
// dialect's input table (inputs.odin), declared before the source.

// Compiles Shadertoy GLSL (the GLSL dialect: mainImage, the Shadertoy
// inputs, the preprocessor). Errors point at the source's lines.
compile_shadertoy :: proc(source: string, options := Options{}) -> (m: ^Module, errors: []Error, ok: bool) {
	o := options
	o.profile = .Shadertoy
	return compile(source, o)
}

// Checks the user's mainImage and synthesizes the `main` that calls it.
@(private)
shadertoy_entry :: proc(c: ^Checker) -> Func_Id {
	u := c.u
	if mains, has_main := c.by_name["main"]; has_main {
		for f in mains {
			add_error(c, u.func_decls[u.funcs[f].decl].pos, .Main_Reserved)
		}
		return 0
	}
	images := c.by_name["mainImage"]
	if len(images) == 0 {
		add_error(c, {line = 1, col = 1}, .No_Main_Image)
		return 0
	}
	v2, v4 := vec_type(.Float, 2), vec_type(.Float, 4)
	image: Func_Id
	for f in images {
		ps := fparams(u, f)
		if ty(u, u.funcs[f].ret) == T_VOID && len(ps) == 2 && ty(u, ps[0].type) == v4 && ps[0].qual != .In && ty(u, ps[1].type) == v2 && ps[1].qual == .In {
			image = f
		}
	}
	pos := u.func_decls[u.funcs[images[0]].decl].pos
	if image == 0 {
		add_error(c, pos, .Main_Image_Signature)
		return 0
	}
	if !u.funcs[image].defined {
		add_error(c, pos, .Entry_No_Body, "mainImage")
		return 0
	}

	b := Node_Builder{u, u.func_decls[u.funcs[image].decl].pos}
	body := make([dynamic]u32, context.temp_allocator)
	res_y := node_field(&b, node_ident(&b, "iResolution"), "y")
	flipped_y := node_binary(&b, .Minus, res_y, node_field(&b, node_ident(&b, "coord"), "y"))
	frag := node_ctor(&b, "vec2", node_field(&b, node_ident(&b, "coord"), "x"), flipped_y)
	append(&body, u32(stmt_decl(&b, "vec2", "frag", frag)))
	if c.gl_frag_coord != 0 {
		value := node_ctor(&b, "vec4", node_ident(&b, "frag"), node_float(&b, 0), node_float(&b, 1))
		assign := node_assign(&b, node_ident(&b, "gl_FragCoord"), value)
		append(&body, u32(stmt_expr(&b, assign)))
	}
	black := node_ctor(&b, "vec4", node_float(&b, 0), node_float(&b, 0), node_float(&b, 0), node_float(&b, 1))
	append(&body, u32(stmt_decl(&b, "vec4", "color", black)))
	call := node_call(&b, "mainImage", node_ident(&b, "color"), node_ident(&b, "frag"))
	append(&body, u32(stmt_expr(&b, call)))
	result: Expr_Id
	if u.buffer_pass {
		result = node_ident(&b, "color")
	} else {
		rgb := node_call(&b, "clamp", node_field(&b, node_ident(&b, "color"), "rgb"), node_float(&b, 0), node_float(&b, 1))
		result = node_ctor(&b, "vec4", rgb, node_float(&b, 1))
	}
	append(&body, u32(add_node_stmt(&b, {kind = .Return, expr = result})))
	block := add_node_stmt(&b, {kind = .Block, body = push_ids(u, body[:])})

	params := Range{u32(len(u.params)), 1}
	append(&u.params, Param{pos = b.pos, name = "coord", spec = node_spec(&b, "vec2")})
	fd := Func_Decl_Id(len(u.func_decls))
	ret := node_spec(&b, "vec4")
	append(&u.func_decls, Func_Decl{pos = b.pos, name = "main", ret = ret, params = params, body = block})
	f := resolve_func_decl(c, fd)
	if f == 0 {
		return 0
	}
	c.entry_writes = true
	check_body(c, f, fd)
	c.entry_writes = false
	return f
}
