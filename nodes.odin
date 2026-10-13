package oksl

// Synthesized syntax: the entry adapters (shadertoy.odin, stage.odin) build
// their functions as Unit nodes, which the checker then checks like
// source.

// Appends unchecked syntax at one position, as the parser would.
@(private)
Node_Builder :: struct {
	u:   ^Unit,
	pos: Pos,
}

@(private)
node_expr :: proc(b: ^Node_Builder, e: Expr) -> Expr_Id {
	e := e
	e.pos = b.pos
	id := Expr_Id(len(b.u.exprs))
	append(&b.u.exprs, e)
	return id
}

@(private)
node_ident :: proc(b: ^Node_Builder, name: string) -> Expr_Id {
	return node_expr(b, {kind = .Ident, name = name})
}

@(private)
node_float :: proc(b: ^Node_Builder, v: f64) -> Expr_Id {
	return node_expr(b, {kind = .Float_Lit, value = transmute(u64)v})
}

@(private)
node_field :: proc(b: ^Node_Builder, a: Expr_Id, name: string) -> Expr_Id {
	return node_expr(b, {kind = .Field, a = a, name = name})
}

@(private)
node_call :: proc(b: ^Node_Builder, name: string, args: ..Expr_Id) -> Expr_Id {
	r := push_ids(b.u, transmute([]u32)args)
	return node_expr(b, {kind = .Call, name = name, args = r})
}

@(private)
node_spec :: proc(b: ^Node_Builder, type_name: string) -> Spec_Id {
	id := Spec_Id(len(b.u.specs))
	append(&b.u.specs, Type_Spec{pos = b.pos, name = type_name})
	return id
}

@(private)
node_ctor :: proc(b: ^Node_Builder, type_name: string, args: ..Expr_Id) -> Expr_Id {
	spec := node_spec(b, type_name)
	r := push_ids(b.u, transmute([]u32)args)
	return node_expr(b, {kind = .Call, ctor = spec, name = type_name, args = r})
}

@(private)
add_node_stmt :: proc(b: ^Node_Builder, s: Stmt) -> Stmt_Id {
	s := s
	s.pos = b.pos
	id := Stmt_Id(len(b.u.stmts))
	append(&b.u.stmts, s)
	return id
}

@(private)
stmt_expr :: proc(b: ^Node_Builder, e: Expr_Id) -> Stmt_Id {
	return add_node_stmt(b, {kind = .Expr, expr = e})
}

@(private)
stmt_decl :: proc(b: ^Node_Builder, type_name, name: string, init: Expr_Id) -> Stmt_Id {
	v := Var_Id(len(b.u.vars))
	spec := node_spec(b, type_name)
	append(&b.u.vars, Var_Decl{pos = b.pos, name = name, spec = spec, init = init})
	return add_node_stmt(b, {kind = .Decl, decls = push_ids(b.u, {u32(v)})})
}

@(private)
node_binary :: proc(b: ^Node_Builder, op: Token_Kind, x, y: Expr_Id) -> Expr_Id {
	return node_expr(b, {kind = .Binary, op = op, a = x, b = y})
}

@(private)
node_assign :: proc(b: ^Node_Builder, target, value: Expr_Id) -> Expr_Id {
	return node_expr(b, {kind = .Assign, op = .Eq, a = target, b = value})
}
