package oksl

import "core:mem"
import "core:mem/virtual"
import "core:strings"

// The token-level preprocessor: a flat token array in, a flat token array
// out. GLSL gets object-like and function-like #define (with ## pasting),
// #undef, #if/#ifdef/#ifndef/#elif/#else/#endif with defined() and integer
// expressions, #line and #error; #version, #extension and #pragma are
// skipped. SkSL has no preprocessor: #version is skipped and any other
// directive is reported (profile_features).
//
// Macros live in a table indexed by `Macro_Id`, their parameters and bodies
// in two flat arrays. Expansion reads from a stack of token frames (the
// source segment, then each macro body being rescanned); a macro is
// disabled while its frame is on the stack, and its name met there is
// marked No_Expand for good, as C does. Arguments are expanded on their
// own before substitution, except next to ##. Expansions are built in a
// scratch arena freed at the end; nothing is allocated per token.
//
// Tokens keep source positions: an argument keeps its own, a token from a
// macro body takes the position of the macro's use, so diagnostics point
// at the user's line and column.
//
// The macro table outlives a unit: `preprocess_unit` can run several
// sources in turn with one set of macros. A program (program.odin) runs its
// Common unit once, marks the table (`preprocess_mark`) and starts each pass
// from that mark (`preprocess_reset`), so Common's macros reach every pass
// and a pass's own stay in it.

Macro_Id :: distinct u32

@(private)
Macro :: struct {
	name:     string,
	// Parameter names, into Preprocessor.params.
	params:   Range,
	// Replacement tokens, into Preprocessor.bodies.
	body:     Range,
	function: bool,
}

// An open #if.
@(private)
Cond :: struct {
	pos:       Pos,
	directive: string,
	// This branch's text is kept.
	active:    bool,
	// A branch was taken already.
	taken:     bool,
	// The text around the #if is kept.
	outer:     bool,
	seen_else: bool,
}

// Token slices being read: the source segment, then macro expansions. A
// frame of a macro's expansion re-enables the macro when it runs out.
@(private)
Frame :: struct {
	toks:  []Token,
	i:     int,
	macro: Macro_Id,
}

@(private)
Reader :: struct {
	frames: [dynamic]Frame,
}

Preprocessor :: struct {
	profile:    Profile,
	features:   Features,
	// Index 0 is reserved.
	macros:     [dynamic]Macro,
	by_name:    map[string]Macro_Id,
	params:     [dynamic]string,
	bodies:     [dynamic]Token,
	// Per macro: its expansion is being read (its name is not expanded).
	active:     [dynamic]bool,
	conds:      [dynamic]Cond,
	// From #line: added to the lines of the tokens that follow.
	line_delta: i32,
	out:        [dynamic]Token,
	errors:     ^[dynamic]Error,
	// Pasted tokens' text and #error messages outlive the preprocessor.
	keep:       mem.Allocator,
	scratch:    mem.Allocator,
	arena:      virtual.Arena,
}

// Preprocesses one source's tokens for `dialect`. The result and the
// strings it points to live in `allocator`.
preprocess :: proc(toks: []Token, profile: Profile, errors: ^[dynamic]Error, allocator := context.allocator) -> []Token {
	pp: Preprocessor
	preprocess_init(&pp, profile, errors, allocator)
	defer preprocess_destroy(&pp)
	preprocess_unit(&pp, toks)
	return preprocess_end(&pp, toks[len(toks) - 1].pos if len(toks) > 0 else {line = 1, col = 1})
}

preprocess_init :: proc(pp: ^Preprocessor, profile: Profile, errors: ^[dynamic]Error, allocator := context.allocator) {
	pp.profile = profile
	// Directives are the dialect's, whatever the stage.
	pp.features = DIALECT_RULES[PROFILE_RULES[profile].dialect].has
	pp.errors = errors
	pp.keep = allocator
	_ = virtual.arena_init_growing(&pp.arena)
	pp.scratch = virtual.arena_allocator(&pp.arena)
	pp.macros = make([dynamic]Macro, pp.scratch)
	pp.by_name = make(map[string]Macro_Id, pp.scratch)
	pp.params = make([dynamic]string, pp.scratch)
	pp.bodies = make([dynamic]Token, pp.scratch)
	pp.active = make([dynamic]bool, pp.scratch)
	pp.conds = make([dynamic]Cond, pp.scratch)
	pp.out = make([dynamic]Token, allocator)
	append(&pp.macros, Macro{})
	append(&pp.active, false)
	dialect := DIALECT_RULES[PROFILE_RULES[profile].dialect]
	for d in dialect.predefined {
		predefine(pp, d)
	}
	for d in PROFILE_RULES[profile].predefined {
		predefine(pp, d)
	}
}

@(private = "file")
predefine :: proc(pp: ^Preprocessor, d: Predefined_Macro) {
	toks := tokenize(d.value, pp.scratch)
	body := Range{u32(len(pp.bodies)), u32(len(toks) - 1)}
	append(&pp.bodies, ..toks[:len(toks) - 1])
	add_macro(pp, Macro{name = d.name, body = body})
}

preprocess_destroy :: proc(pp: ^Preprocessor) {
	virtual.arena_destroy(&pp.arena)
}

// Appends the preprocessed tokens of one source (ending in EOF or not) to
// the output; macros carry over to the next unit.
preprocess_unit :: proc(pp: ^Preprocessor, toks: []Token) {
	i := 0
	for i < len(toks) && toks[i].kind != .EOF {
		// The line (a directive) or the run of lines up to the next one.
		end := i + 1
		directive := toks[i].kind == .Hash && .Line_Start in toks[i].flags
		for end < len(toks) && toks[end].kind != .EOF {
			t := toks[end]
			if directive ? .Line_Start in t.flags : (t.kind == .Hash && .Line_Start in t.flags) {
				break
			}
			end += 1
		}
		chunk := toks[i:end]
		i = end
		if pp.line_delta != 0 {
			for &t in chunk {
				t.pos.line += pp.line_delta
			}
		}
		if directive {
			run_directive(pp, chunk)
		} else if !skipping(pp) {
			if .Preprocessor in pp.features {
				r := Reader{frames = make([dynamic]Frame, pp.scratch)}
				append(&r.frames, Frame{toks = chunk})
				expand(pp, &r, &pp.out)
			} else {
				append(&pp.out, ..chunk)
			}
		}
	}
	for c in pp.conds {
		pp_error(pp, c.pos, .If_Unterminated, c.directive)
	}
	clear(&pp.conds)
}

// The output, ended by an EOF token at `eof`.
preprocess_end :: proc(pp: ^Preprocessor, eof: Pos) -> []Token {
	append(&pp.out, Token{kind = .EOF, flags = {.Line_Start}, pos = eof})
	return pp.out[:]
}

// The macro table at a point (after a program's Common unit), to start
// each later unit from.
Macro_Mark :: struct {
	macros, params, bodies: int,
	by_name:                map[string]Macro_Id,
}

preprocess_mark :: proc(pp: ^Preprocessor) -> Macro_Mark {
	m := Macro_Mark{len(pp.macros), len(pp.params), len(pp.bodies), make(map[string]Macro_Id, len(pp.by_name), pp.scratch)}
	for name, id in pp.by_name {
		m.by_name[name] = id
	}
	return m
}

// Back to the macros of `m` (later #define and #undef are forgotten), with
// an empty output and no #line offset. Entries before the mark never
// change, so dropping the ones after it restores the table.
preprocess_reset :: proc(pp: ^Preprocessor, m: Macro_Mark) {
	resize(&pp.macros, m.macros)
	resize(&pp.active, m.macros)
	resize(&pp.params, m.params)
	resize(&pp.bodies, m.bodies)
	clear(&pp.by_name)
	for name, id in m.by_name {
		pp.by_name[name] = id
	}
	pp.line_delta = 0
	clear(&pp.out)
}

@(private = "file")
pp_error :: proc(pp: ^Preprocessor, pos: Pos, code: Error_Code, name := "", a := "", b := "", n: i64 = 0, m: i64 = 0) {
	append(pp.errors, Error{pos = pos, code = code, name = name, a = a, b = b, n = n, m = m})
}

@(private = "file")
skipping :: proc(pp: ^Preprocessor) -> bool {
	return len(pp.conds) > 0 && !pp.conds[len(pp.conds) - 1].active
}

@(private = "file")
add_macro :: proc(pp: ^Preprocessor, m: Macro) {
	id := Macro_Id(len(pp.macros))
	append(&pp.macros, m)
	append(&pp.active, false)
	pp.by_name[m.name] = id
}

// ---- directives ----------------------------------------------------------------------

@(private = "file")
run_directive :: proc(pp: ^Preprocessor, line: []Token) {
	if len(line) < 2 {
		// A lone `#`.
		return
	}
	hash := line[0]
	name := line[1].text
	args := line[2:]
	if .Preprocessor not_in pp.features {
		if name != "version" && !skipping(pp) {
			pp_error(pp, hash.pos, DIALECT_RULES[PROFILE_RULES[pp.profile].dialect].missing[.Preprocessor], name)
		}
		return
	}
	switch name {
	case "if", "ifdef", "ifndef":
		outer := !skipping(pp)
		v := false
		if outer {
			v = name == "if" ? eval_if(pp, args, name, hash.pos) : macro_test(pp, args, name, hash.pos) == (name == "ifdef")
		}
		append(&pp.conds, Cond{pos = hash.pos, directive = name, active = outer && v, taken = v, outer = outer})
		return
	case "elif", "else":
		if len(pp.conds) == 0 {
			pp_error(pp, hash.pos, .If_Without_If, name)
			return
		}
		c := &pp.conds[len(pp.conds) - 1]
		if c.seen_else {
			pp_error(pp, hash.pos, .Else_After_Else, name)
			return
		}
		if name == "else" {
			c.seen_else = true
			c.active = c.outer && !c.taken
			c.taken = true
			return
		}
		c.active = false
		if c.outer && !c.taken {
			v := eval_if(pp, args, name, hash.pos)
			c.active = v
			c.taken = v
		}
		return
	case "endif":
		if len(pp.conds) == 0 {
			pp_error(pp, hash.pos, .If_Without_If, name)
			return
		}
		pop(&pp.conds)
		return
	}
	if skipping(pp) {
		return
	}
	switch name {
	case "define":
		define(pp, args, hash.pos)
	case "undef":
		if len(args) == 0 || args[0].kind != .Ident {
			pp_error(pp, hash.pos, .Macro_Name_Expected, name)
			return
		}
		delete_key(&pp.by_name, args[0].text)
	case "line":
		if len(args) == 0 || args[0].kind != .Int_Lit {
			pp_error(pp, hash.pos, .Line_Number)
			return
		}
		// The next line is line N.
		physical := hash.pos.line - pp.line_delta + 1
		pp.line_delta = i32(parse_int_text(args[0].text)) - physical
	case "version", "extension", "pragma":
	case "error":
		b := strings.builder_make(pp.keep)
		for t, k in args {
			if k > 0 {
				strings.write_byte(&b, ' ')
			}
			strings.write_string(&b, t.text)
		}
		pp_error(pp, hash.pos, .Error_Directive, a = strings.to_string(b))
	case:
		pp_error(pp, hash.pos, .Unknown_Directive, name)
	}
}

@(private = "file")
define :: proc(pp: ^Preprocessor, args: []Token, pos: Pos) {
	if len(args) == 0 || args[0].kind != .Ident {
		pp_error(pp, pos, .Macro_Name_Expected, "define")
		return
	}
	m := Macro{name = args[0].text}
	rest := args[1:]
	// `F(x)` with no space before the parenthesis takes parameters.
	if len(rest) > 0 && rest[0].kind == .L_Paren && .Space_Before not_in rest[0].flags {
		m.function = true
		start := len(pp.params)
		k := 1
		if k < len(rest) && rest[k].kind == .R_Paren {
			k += 1
		} else {
			for {
				if k >= len(rest) || rest[k].kind != .Ident {
					pp_error(pp, pos, .Macro_Params, m.name)
					return
				}
				append(&pp.params, rest[k].text)
				k += 1
				if k < len(rest) && rest[k].kind == .Comma {
					k += 1
					continue
				}
				if k < len(rest) && rest[k].kind == .R_Paren {
					k += 1
					break
				}
				pp_error(pp, pos, .Macro_Params, m.name)
				return
			}
		}
		m.params = {u32(start), u32(len(pp.params) - start)}
		rest = rest[k:]
	}
	m.body = {u32(len(pp.bodies)), u32(len(rest))}
	append(&pp.bodies, ..rest)
	add_macro(pp, m)
}

// Whether the macro named after #ifdef / #ifndef is defined.
@(private = "file")
macro_test :: proc(pp: ^Preprocessor, args: []Token, directive: string, pos: Pos) -> bool {
	if len(args) == 0 || args[0].kind != .Ident {
		pp_error(pp, pos, .Macro_Name_Expected, directive)
		return false
	}
	return args[0].text in pp.by_name
}

// ---- #if expressions ------------------------------------------------------------------

@(private = "file")
If_Eval :: struct {
	pp:        ^Preprocessor,
	toks:      []Token,
	i:         int,
	directive: string,
	pos:       Pos,
	failed:    bool,
}

// The value of an #if / #elif condition: `defined` is resolved first, then
// macros are expanded and the integer expression evaluated (C's operators;
// an identifier left over is an error, as GLSL ES says).
@(private = "file")
eval_if :: proc(pp: ^Preprocessor, args: []Token, directive: string, pos: Pos) -> bool {
	buf := make([dynamic]Token, 0, len(args), pp.scratch)
	for k := 0; k < len(args); k += 1 {
		t := args[k]
		if t.kind != .Ident || t.text != "defined" {
			append(&buf, t)
			continue
		}
		name := ""
		if k + 1 < len(args) && args[k + 1].kind == .Ident {
			name = args[k + 1].text
			k += 1
		} else if k + 3 < len(args) && args[k + 1].kind == .L_Paren && args[k + 2].kind == .Ident && args[k + 3].kind == .R_Paren {
			name = args[k + 2].text
			k += 3
		} else {
			pp_error(pp, t.pos, .Macro_Name_Expected, "defined")
			return false
		}
		append(&buf, Token{kind = .Int_Lit, pos = t.pos, text = "1" if name in pp.by_name else "0"})
	}
	r := Reader{frames = make([dynamic]Frame, pp.scratch)}
	append(&r.frames, Frame{toks = buf[:]})
	expanded := make([dynamic]Token, pp.scratch)
	expand(pp, &r, &expanded)
	e := If_Eval{pp = pp, toks = expanded[:], directive = directive, pos = pos}
	v := if_ternary(&e)
	if !e.failed && e.i < len(e.toks) {
		if_fail(&e)
	}
	return !e.failed && v != 0
}

@(private = "file")
if_fail :: proc(e: ^If_Eval) {
	if e.failed {
		return
	}
	e.failed = true
	if e.i < len(e.toks) {
		t := e.toks[e.i]
		if t.kind == .Ident {
			pp_error(e.pp, t.pos, .If_Undefined, e.directive, a = t.text)
		} else {
			pp_error(e.pp, t.pos, .If_Expression, e.directive, a = t.text)
		}
	} else {
		pp_error(e.pp, e.pos, .If_Incomplete, e.directive)
	}
}

@(private = "file")
if_peek :: proc(e: ^If_Eval) -> Token_Kind {
	return e.toks[e.i].kind if e.i < len(e.toks) else .EOF
}

@(private = "file")
if_ternary :: proc(e: ^If_Eval) -> i64 {
	c := if_binary(e, 1)
	if if_peek(e) != .Question {
		return c
	}
	e.i += 1
	a := if_ternary(e)
	if if_peek(e) != .Colon {
		if_fail(e)
		return 0
	}
	e.i += 1
	b := if_ternary(e)
	return a if c != 0 else b
}

@(private = "file")
if_binary :: proc(e: ^If_Eval, min_prec: int) -> i64 {
	lhs := if_unary(e)
	for !e.failed {
		op := if_peek(e)
		prec := binary_precedence(op)
		if prec < min_prec {
			break
		}
		e.i += 1
		rhs := if_binary(e, prec + 1)
		#partial switch op {
		case .Or_Or:
			lhs = i64(lhs != 0 || rhs != 0)
		case .Xor_Xor:
			lhs = i64((lhs != 0) != (rhs != 0))
		case .And_And:
			lhs = i64(lhs != 0 && rhs != 0)
		case .Pipe:
			lhs |= rhs
		case .Caret:
			lhs ~= rhs
		case .Amp:
			lhs &= rhs
		case .Eq_Eq:
			lhs = i64(lhs == rhs)
		case .Not_Eq:
			lhs = i64(lhs != rhs)
		case .Lt:
			lhs = i64(lhs < rhs)
		case .Gt:
			lhs = i64(lhs > rhs)
		case .Lt_Eq:
			lhs = i64(lhs <= rhs)
		case .Gt_Eq:
			lhs = i64(lhs >= rhs)
		case .Shl:
			lhs <<= u64(rhs & 63)
		case .Shr:
			lhs >>= u64(rhs & 63)
		case .Plus:
			lhs += rhs
		case .Minus:
			lhs -= rhs
		case .Star:
			lhs *= rhs
		case .Slash, .Percent:
			if rhs == 0 {
				if !e.failed {
					e.failed = true
					pp_error(e.pp, e.pos, .If_Divide, e.directive)
				}
				return 0
			}
			lhs = lhs / rhs if op == .Slash else lhs % rhs
		}
	}
	return lhs
}

@(private = "file")
if_unary :: proc(e: ^If_Eval) -> i64 {
	if e.failed || e.i >= len(e.toks) {
		if_fail(e)
		return 0
	}
	t := e.toks[e.i]
	#partial switch t.kind {
	case .Int_Lit, .Uint_Lit:
		e.i += 1
		return i64(parse_int_text(t.text))
	case .L_Paren:
		e.i += 1
		v := if_ternary(e)
		if if_peek(e) != .R_Paren {
			if_fail(e)
			return 0
		}
		e.i += 1
		return v
	case .Minus:
		e.i += 1
		return -if_unary(e)
	case .Plus:
		e.i += 1
		return if_unary(e)
	case .Tilde:
		e.i += 1
		return ~if_unary(e)
	case .Bang:
		e.i += 1
		return i64(if_unary(e) == 0)
	}
	if_fail(e)
	return 0
}

// ---- expansion -----------------------------------------------------------------------

@(private = "file")
reader_next :: proc(pp: ^Preprocessor, r: ^Reader) -> (Token, bool) {
	for len(r.frames) > 0 {
		f := &r.frames[len(r.frames) - 1]
		if f.i < len(f.toks) {
			f.i += 1
			return f.toks[f.i - 1], true
		}
		if f.macro != 0 {
			pp.active[f.macro] = false
		}
		pop(&r.frames)
	}
	return {}, false
}

@(private = "file")
reader_peek :: proc(r: ^Reader) -> (Token, bool) {
	#reverse for f in r.frames {
		if f.i < len(f.toks) {
			return f.toks[f.i], true
		}
	}
	return {}, false
}

// Expands macros in everything `r` reads, into `out`.
@(private = "file")
expand :: proc(pp: ^Preprocessor, r: ^Reader, out: ^[dynamic]Token) {
	for {
		t := reader_next(pp, r) or_break
		if t.kind != .Ident || .No_Expand in t.flags {
			append(out, t)
			continue
		}
		id, is_macro := pp.by_name[t.text]
		if !is_macro {
			append(out, t)
			continue
		}
		if pp.active[id] {
			// Inside its own expansion: never expanded again.
			t.flags += {.No_Expand}
			append(out, t)
			continue
		}
		m := pp.macros[id]
		args: [][]Token
		if m.function {
			// A function-like macro's name alone is just a name.
			if next, ok := reader_peek(r); !ok || next.kind != .L_Paren {
				append(out, t)
				continue
			}
			reader_next(pp, r)
			ok: bool
			if args, ok = collect_args(pp, r, t, m); !ok {
				continue
			}
		}
		body := substitute(pp, m, t.pos, args)
		pp.active[id] = true
		append(&r.frames, Frame{toks = body, macro = id})
	}
}

// The arguments of a macro use, after its '(': split at top-level commas.
@(private = "file")
collect_args :: proc(pp: ^Preprocessor, r: ^Reader, name: Token, m: Macro) -> ([][]Token, bool) {
	buf := make([dynamic]Token, pp.scratch)
	starts := make([dynamic]int, pp.scratch)
	append(&starts, 0)
	depth := 0
	for {
		t, ok := reader_next(pp, r)
		if !ok {
			pp_error(pp, name.pos, .Macro_Unterminated, m.name)
			return nil, false
		}
		#partial switch t.kind {
		case .L_Paren:
			depth += 1
		case .R_Paren:
			if depth == 0 {
				args := make([][]Token, len(starts), pp.scratch)
				for &a, k in args {
					end := starts[k + 1] if k + 1 < len(starts) else len(buf)
					a = buf[starts[k]:end]
				}
				want := int(m.params.count)
				got := len(args)
				if want == 0 && got == 1 && len(args[0]) == 0 {
					return nil, true
				}
				if got != want {
					pp_error(pp, name.pos, .Macro_Arg_Count, m.name, n = i64(want), m = i64(got))
					return nil, false
				}
				return args, true
			}
			depth -= 1
		case .Comma:
			if depth == 0 {
				append(&starts, len(buf))
				continue
			}
		}
		append(&buf, t)
	}
}

// The replacement of a macro use at `pos`: parameters become the expanded
// arguments (the raw ones next to ##), ## pastes, and body tokens take the
// use's position.
@(private = "file")
substitute :: proc(pp: ^Preprocessor, m: Macro, pos: Pos, args: [][]Token) -> []Token {
	body := pp.bodies[m.body.start:][:m.body.count]
	params := pp.params[m.params.start:][:m.params.count]
	out := make([dynamic]Token, 0, len(body), pp.scratch)
	expanded := make([][]Token, len(params), pp.scratch)
	done := make([]bool, len(params), pp.scratch)
	param :: proc(params: []string, t: Token) -> int {
		if t.kind == .Ident {
			for p, k in params {
				if p == t.text {
					return k
				}
			}
		}
		return -1
	}
	for k := 0; k < len(body); k += 1 {
		t := body[k]
		if t.kind == .Hash_Hash && len(out) > 0 && k + 1 < len(body) {
			k += 1
			next := body[k]
			next.pos = pos
			rhs := ([^]Token)(&next)[:1]
			if p := param(params, body[k]); p >= 0 {
				rhs = args[p]
			}
			if len(rhs) == 0 {
				continue
			}
			lhs := pop(&out)
			append(&out, paste(pp, lhs, rhs[0]))
			append(&out, ..rhs[1:])
			continue
		}
		if p := param(params, t); p >= 0 {
			if k + 1 < len(body) && body[k + 1].kind == .Hash_Hash {
				append(&out, ..args[p])
				continue
			}
			if !done[p] {
				r := Reader{frames = make([dynamic]Frame, pp.scratch)}
				append(&r.frames, Frame{toks = args[p]})
				e := make([dynamic]Token, pp.scratch)
				expand(pp, &r, &e)
				expanded[p] = e[:]
				done[p] = true
			}
			append(&out, ..expanded[p])
			continue
		}
		t.pos = pos
		t.flags -= {.Line_Start}
		append(&out, t)
	}
	return out[:]
}

// `a ## b`: one token spelled like both together.
@(private = "file")
paste :: proc(pp: ^Preprocessor, a, b: Token) -> Token {
	text := strings.concatenate({a.text, b.text}, pp.keep)
	l := Lexer{src = text, line = 1, col = 1}
	kind := lex_one(&l)
	if l.i != len(text) {
		pp_error(pp, a.pos, .Paste_Invalid, a = a.text, b = b.text)
		return a
	}
	return Token{kind = kind, flags = a.flags, pos = a.pos, text = text}
}
