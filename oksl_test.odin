package oksl

import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:testing"

// ---- helpers ------------------------------------------------------------------------

@(private = "file")
errors_of :: proc(src: string) -> string {
	m, errs, ok := compile(src)
	defer destroy(m)
	if ok {
		return ""
	}
	return strings.clone(format_errors(errs, context.temp_allocator), context.temp_allocator)
}

// Runs `body` (statements that set `fragColor`, Shadertoy style with
// `fragCoord` growing upwards) in an SkSL main for one pixel of a 4x2 layer
// and returns pixel (x, y)'s colour.
@(private = "file")
eval_at :: proc(t: ^testing.T, body: string, x := 0, y := 0, prelude := "", loc := #caller_location) -> [4]f32 {
	src := fmt.tprintf(
		"uniform float3 iResolution;\nuniform float iTime;\n%s\nhalf4 main(float2 coord) {{\nfloat2 fragCoord = float2(coord.x, iResolution.y - coord.y);\nfloat4 fragColor = float4(0.0);\n%s\nreturn fragColor;\n}}\n",
		prelude,
		body,
	)
	m, errs, ok := compile(src)
	defer destroy(m)
	if !ok {
		testing.expectf(t, false, "compile failed:\n%s", format_errors(errs, context.temp_allocator), loc = loc)
		return {}
	}
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}, time = 2})
	out := make([][4]f32, 8, context.temp_allocator)
	run_f32(&m.program, block, 4, 2, out, temp_scratch())
	return out[y * 4 + x]
}

@(private = "file")
eval :: proc(t: ^testing.T, body: string, prelude := "", loc := #caller_location) -> [4]f32 {
	return eval_at(t, body, prelude = prelude, loc = loc)
}

// Working memory for one run, freed with the temp allocator.
@(private = "file")
temp_scratch :: proc() -> ^Scratch {
	return new_clone(Scratch{make([dynamic]Reg, context.temp_allocator), make([dynamic]Lane_Mask, context.temp_allocator)}, context.temp_allocator)
}

@(private = "file")
near :: proc(t: ^testing.T, got, want: [4]f32, eps: f32 = 1e-5, loc := #caller_location) {
	for i in 0 ..< 4 {
		if !(abs(got[i] - want[i]) <= eps) {
			testing.expectf(t, false, "got %v, want %v", got, want, loc = loc)
			return
		}
	}
}

// ---- the preprocessor ---------------------------------------------------------------

// The preprocessed token texts of `src` (GLSL), space-separated, and the
// formatted errors.
@(private = "file")
pp_text :: proc(src: string) -> (text: string, errs: string) {
	errors := make([dynamic]Error, context.temp_allocator)
	toks := tokenize(src, context.temp_allocator)
	out := preprocess(toks[:], .Shadertoy, &errors, context.temp_allocator)
	b := strings.builder_make(context.temp_allocator)
	for tk, i in out {
		if tk.kind == .EOF {
			break
		}
		if i > 0 {
			strings.write_byte(&b, ' ')
		}
		strings.write_string(&b, tk.text)
	}
	return strings.to_string(b), format_errors(errors[:], context.temp_allocator)
}

@(test)
test_preprocessor_macros :: proc(t: ^testing.T) {
	expect_pp :: proc(t: ^testing.T, src, want: string, loc := #caller_location) {
		got, errs := pp_text(src)
		testing.expectf(t, got == want && errs == "", "%q:\ngot  %q\nwant %q\n%s", src, got, want, errs, loc = loc)
	}
	// Object-like, using earlier ones, never inside number suffixes.
	expect_pp(t, "#define PI 3.14\n#define TAU (2.0 * PI) // a turn\nfloat a = TAU + PI2;", "float a = ( 2.0 * 3.14 ) + PI2 ;")
	expect_pp(t, "#define e 2.0\nfloat c = 1e5 + e;", "float c = 1e5 + 2.0 ;")
	// Function-like: arguments are expanded first, parentheses nest, and a
	// name without '(' stays a name.
	expect_pp(t, "#define SQ(x) ((x)*(x))\nfloat b = SQ(f(1, 2));", "float b = ( ( f ( 1 , 2 ) ) * ( f ( 1 , 2 ) ) ) ;")
	expect_pp(t, "#define ADD(a, b) a + b\n#define ONE 1\nint i = ADD(ONE, ADD(2, 3));", "int i = 1 + 2 + 3 ;")
	expect_pp(t, "#define F(x) x\nint F = F(4);", "int F = 4 ;")
	expect_pp(t, "#define F (x)\nF", "( x )")
	expect_pp(t, "#define NONE() 7\nint n = NONE();", "int n = 7 ;")
	// Arguments across lines; a macro producing a macro call.
	expect_pp(t, "#define R(a) rot(a)\n#define G R\nG(\n  1.0);", "rot ( 1.0 ) ;")
	// A macro is not expanded inside itself.
	expect_pp(t, "#define x x + 1\nx;", "x + 1 ;")
	expect_pp(t, "#define f(a) a * f(a)\nf(2);", "2 * f ( 2 ) ;")
	// Continued lines and CRLF.
	expect_pp(t, "#define LONG(a) a + \\\r\n  1\r\nint l = LONG(2);", "int l = 2 + 1 ;")
	// Pasting, #undef.
	expect_pp(t, "#define CAT(a, b) a ## b\nfloat CAT(my, Var) = CAT(1, 0);", "float myVar = 10 ;")
	expect_pp(t, "#define A 1\n#undef A\nA", "A")
	// Ignored directives and the predefined macros.
	expect_pp(t, "#version 300 es\n#extension GL_OES_standard_derivatives : enable\n#pragma optimize(on)\nint v = __VERSION__ + GL_ES;", "int v = 300 + 1 ;")
}

@(test)
test_preprocessor_conditionals :: proc(t: ^testing.T) {
	expect_pp :: proc(t: ^testing.T, src, want: string, loc := #caller_location) {
		got, errs := pp_text(src)
		testing.expectf(t, got == want && errs == "", "%q:\ngot  %q\nwant %q\n%s", src, got, want, errs, loc = loc)
	}
	expect_pp(t, "#define Q 2\n#if Q == 1\na\n#elif Q == 2\nb\n#else\nc\n#endif", "b")
	expect_pp(t, "#ifdef NOPE\na\n#else\nb\n#endif\n#ifndef NOPE\nc\n#endif", "b c")
	expect_pp(t, "#define A\n#if defined(A) && !defined B\nyes\n#endif", "yes")
	expect_pp(t, "#if (1 + 2) * 3 == 9 && (7 % 4) == 3 && (1 << 3) == 8 && -1 < 0 && (0 ? 0 : 5) == 5\nok\n#endif", "ok")
	// Nested: the inner #if of a skipped branch is not evaluated.
	expect_pp(t, "#if 0\n#if UNDEFINED_NAME\nx\n#endif\n#elif 1\ny\n#endif", "y")
	expect_pp(t, "#define F(x) (x * 2)\n#if F(3) > 5\nbig\n#endif", "big")
	expect_pp(t, "#if HW_PERFORMANCE == 1\nfast\n#endif", "fast")
	// Errors: positions are the directive's.
	expect_err :: proc(t: ^testing.T, src, want: string, loc := #caller_location) {
		_, errs := pp_text(src)
		testing.expectf(t, strings.contains(errs, want), "%q: got %q, want %q", src, errs, want, loc = loc)
	}
	expect_err(t, "a\n#if 1\nb", "2:1: #if without #endif")
	expect_err(t, "#endif", "1:1: #endif without #if")
	expect_err(t, "#if 1\n#else\n#else\n#endif", "3:1: #else after #else")
	expect_err(t, "#if FOO\n#endif", "1:5: \"FOO\" is not a macro")
	expect_err(t, "#if 1 / 0\n#endif", "division by zero in #if")
	expect_err(t, "#if 1 +\n#endif", "1:1: #if expression ends early")
	expect_err(t, "#define SQ(x) x*x\nSQ(1, 2)", "2:1: macro SQ takes 1 arguments, got 2")
	expect_err(t, "#define SQ(x) x*x\nSQ(1", "2:1: unterminated argument list for macro SQ")
	expect_err(t, "#error stop here", "1:1: #error stop here")
	expect_err(t, "#frobnicate", "1:1: unknown directive #frobnicate")
	// #line renumbers the following lines.
	_, errs := pp_text("#line 100\n#if 1\n")
	testing.expectf(t, strings.contains(errs, "100:1: #if without #endif"), "%q", errs)
}

// ---- Shadertoy (the GLSL dialect) ---------------------------------------------------

// Compiles `src` as Shadertoy code and runs it on a 4x2 layer; the sampler
// reports the uv it was given (and the channel in alpha).
@(private = "file")
run_shadertoy :: proc(t: ^testing.T, src: string, b: Builtins = {resolution = {4, 2, 1}}, loc := #caller_location) -> (out: [][4]f32, m: ^Module) {
	errs: []Error
	ok: bool
	m, errs, ok = compile_shadertoy(src)
	if !ok {
		testing.expectf(t, false, "compile failed:\n%s", format_errors(errs, context.temp_allocator), loc = loc)
		destroy(m)
		return nil, nil
	}
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, b)
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
		return {uv.x, uv.y, lod, f32(channel)}
	}
	out = make([][4]f32, 8, context.temp_allocator)
	for &px in out {
		px = {-1, -1, -1, -1}
	}
	run_f32(&m.program, block, 4, 2, out, temp_scratch(), {sampler = sampler})
	return out, m
}

@(test)
test_shadertoy_entry :: proc(t: ^testing.T) {
	// Origin bottom-left, colour clamped, alpha 1; macros and precision.
	out, m := run_shadertoy(t, "precision highp float;\n#define K 0.5\n#define SCALE(v) ((v) * K)\nvoid mainImage(out vec4 c, in vec2 p) { c = vec4(SCALE(p.y), -1.0, 2.0, 0.0); }")
	if m != nil {
		near(t, out[0], {0.75, 0, 1, 1}) // top row: fragCoord.y = 1.5
		near(t, out[4], {0.25, 0, 1, 1})
		destroy(m)
	}
	// gl_FragCoord is fragCoord; the inputs come from the table.
	src := `
void mainImage(out vec4 c, in vec2 p) {
    c = vec4(gl_FragCoord.xy - p, iChannelTime[2] * 0.25, iSampleRate / 48000.0);
}`
	out, m = run_shadertoy(t, src, {resolution = {4, 2, 1}, sample_rate = 24000, channel_time = {0, 0, 2, 0}})
	if m != nil {
		near(t, out[5], {0, 0, 0.5, 1})
		testing.expect(t, m.inputs_used >= {.Channel_Time, .Sample_Rate, .Resolution})
		destroy(m)
	}
	// Declaring an input again, as code for other hosts does, is allowed.
	out, m = run_shadertoy(t, "uniform vec3 iResolution;\nuniform float iTime;\nvoid mainImage(out vec4 c, in vec2 p) { c = vec4(p / iResolution.xy, 0.0, 1.0); }")
	if m != nil {
		near(t, out[1], {0.375, 0.75, 0, 1})
		destroy(m)
	}
}

@(test)
test_shadertoy_texture :: proc(t: ^testing.T) {
	// texture() takes Shadertoy's uv (origin bottom-left); the sampler sees
	// the top-left uv of the same point.
	src := `
vec4 tap(sampler2D s, vec2 uv) { return texture(s, uv); }
void mainImage(out vec4 c, in vec2 p) {
    vec2 uv = p / iResolution.xy;
    vec4 a = texture(iChannel1, uv);
    vec4 b = textureLod(iChannel1, uv, 2.0);
    vec4 d = tap(iChannel1, uv);
    ivec2 size = textureSize(iChannel1, 0);
    vec4 f = texelFetch(iChannel1, ivec2(1, 0), 0);
    c = vec4(a.xy, b.z, 1.0);
    if (p.x > 3.0) c = vec4(d.xy, float(size.x * 100 + size.y), 1.0);
    if (p.x < 1.0 && p.y < 1.0) c = vec4(f.xy, 0.0, 1.0);
}`
	out, m := run_shadertoy(t, src, {resolution = {4, 2, 1}, channel_resolution = {{}, {8, 4, 1}, {}, {}}})
	if m != nil {
		testing.expect_value(t, m.num_samplers, 2)
		testing.expect_value(t, m.channel_names[1], "iChannel1")
		testing.expect(t, .Channel1 in m.inputs_used && 0 not_in m.channels_used)
		// Pixel (1, 0): fragCoord (1.5, 1.5), uv (0.375, 0.75) -> top-left (0.375, 0.25).
		near(t, out[1], {0.375, 0.25, 1, 1})
		// Pixel (3, 0) through a sampler parameter; textureSize is 8x4.
		near(t, out[3], {0.875, 0.25, 1, 1})
		testing.expect(t, abs(out[7][2] - 1) < 1e-6) // 804 clamped to 1
		// Pixel (0, 1): texel (1, 0) of 8x4 counts from the bottom.
		near(t, out[4], {1.5 / 8, 1 - 0.5 / 4, 0, 1})
		destroy(m)
	}
	// A `uniform sampler2D` takes a channel the source doesn't name.
	out, m = run_shadertoy(t, "uniform sampler2D tex;\nvoid mainImage(out vec4 c, in vec2 p) { c = texture(tex, vec2(0.0)) + texture(iChannel0, vec2(0.0)); }")
	if m != nil {
		testing.expect_value(t, m.channel_names[0], "iChannel0")
		testing.expect_value(t, m.channel_names[1], "tex")
		destroy(m)
	}
}

@(test)
test_program :: proc(t: ^testing.T) {
	// Common's macros, structs and functions reach every pass; a buffer
	// reads itself and a later buffer as the previous frame, an earlier one
	// as this frame's; buffers keep their raw colour.
	desc := Program_Desc {
		common = `
#define GAIN 2.0
struct Cell { float v; };
float shade(Cell c) { return c.v * GAIN; }`,
		sources = #partial {
			.Buffer_A = `
#define LOCAL 1.0
void mainImage(out vec4 c, in vec2 p) {
    Cell cell = Cell(1.5);
    c = vec4(shade(cell), -LOCAL, 0.0, 0.25) + 0.0 * (texture(iChannel0, vec2(0.0)) + texture(iChannel1, vec2(0.0)));
}`,
			.Buffer_B = "void mainImage(out vec4 c, in vec2 p) { c = texture(iChannel0, p / iResolution.xy) * GAIN; }",
			.Image = "void mainImage(out vec4 c, in vec2 p) { c = vec4(shade(Cell(4.0))) + texture(iChannel0, vec2(0.0)) + texture(iChannel3, vec2(0.0)); }",
		},
	}
	desc.channels[.Buffer_A] = {{.Buffer, u8(Pass_Kind.Buffer_A)}, {.Buffer, u8(Pass_Kind.Buffer_B)}, {}, {}}
	desc.channels[.Buffer_B] = {{.Buffer, u8(Pass_Kind.Buffer_A)}, {.External, 9}, {}, {}}
	desc.channels[.Image] = {{.Buffer, u8(Pass_Kind.Buffer_B)}, {}, {}, {.External, 7}}
	g, errs, ok := compile_program(desc)
	defer destroy_program(g)
	if !testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, len(g.passes), 3)
	testing.expect_value(t, g.passes[2].kind, Pass_Kind.Image)
	testing.expect_value(t, g.index[.Buffer_C], -1)
	want := []Edge{{0, 0, .Previous, 0}, {0, 1, .Previous, 1}, {1, 0, .Buffer, 0}, {2, 0, .Buffer, 1}, {2, 3, .External, 7}}
	testing.expect_value(t, len(g.edges), len(want))
	for e, i in want {
		if i < len(g.edges) {
			testing.expect_value(t, g.edges[i], e)
		}
	}
	testing.expect_value(t, g.passes[1].edges, Range{2, 1})
	block := make([]u8, 1024, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}})
	out := make([][4]f32, 8, context.temp_allocator)
	run_f32(&g.passes[0].module.program, block, 4, 2, out, temp_scratch())
	near(t, out[0], {3, -1, 0, 0.25})
	run_f32(&g.passes[2].module.program, block, 4, 2, out, temp_scratch())
	near(t, out[0], {1, 1, 1, 1})

	program_errors :: proc(desc: Program_Desc) -> string {
		g, errs, ok := compile_program(desc)
		defer destroy_program(g)
		return "" if ok else strings.clone(format_errors(errs, context.temp_allocator), context.temp_allocator)
	}
	// A pass's macros stay in it; errors name their source; Common's are
	// reported once; a channel bound to a missing buffer points at its use.
	bad := Program_Desc {
		common = "float twice(float x) { return x * missing; }",
		sources = #partial {
			.Buffer_A = "#define K 1.0\nvoid mainImage(out vec4 c, in vec2 p) { c = vec4(K); }",
			.Image = "void mainImage(out vec4 c, in vec2 p) {\n    c = vec4(K) + texture(iChannel2, p);\n}",
		},
	}
	bad.channels[.Image][2] = {.Buffer, u8(Pass_Kind.Buffer_C)}
	text := program_errors(bad)
	testing.expectf(t, strings.count(text, "Common:1:35: undeclared identifier \"missing\"") == 1, "got:\n%s", text)
	testing.expectf(t, strings.contains(text, "Image:2:14: undeclared identifier \"K\""), "got:\n%s", text)
	bad.common = ""
	bad.sources[.Image] = "void mainImage(out vec4 c, in vec2 p) {\n    c = texture(iChannel2, p);\n}"
	testing.expect_value(t, program_errors(bad), "Image:2:17: iChannel2 reads Buffer C, which the program does not have\n")
	bad.sources[.Image] = ""
	testing.expect_value(t, program_errors(bad), "Image:1:1: a program needs an Image pass\n")
}

@(test)
test_switch :: proc(t: ^testing.T) {
	body := `
int mode = int(fragCoord.x);
float r = 0.0;
switch (mode) {
case 0:
    r += 1.0;
case 1:
    r += 10.0;
    break;
case 2: {
    r = 100.0;
    break;
}
case 3:
default:
    float k = 7.0;
    for (int i = 0; i < 3; i++) {
        if (i == 1) continue;
        r += k;
    }
}
uint u = 2u;
switch (u) { case 1u: r = -1.0; break; case 2u: r += 0.5; }
fragColor = vec4(r);
`
	near(t, eval_at(t, body, 0, 0), {11.5, 11.5, 11.5, 11.5})
	near(t, eval_at(t, body, 1, 0), {10.5, 10.5, 10.5, 10.5})
	near(t, eval_at(t, body, 2, 0), {100.5, 100.5, 100.5, 100.5})
	near(t, eval_at(t, body, 3, 0), {14.5, 14.5, 14.5, 14.5})
	// The same in Shadertoy code, with a return inside a case.
	src := `
float pick(int m) {
    switch (m) {
    case 0: return 0.25;
    case 1:
    case 2: return 0.5;
    }
    return 1.0;
}
void mainImage(out vec4 c, in vec2 p) { c = vec4(pick(int(p.x)), 0.0, 0.0, 1.0); }`
	out, m := run_shadertoy(t, src)
	if m != nil {
		testing.expect(t, out[0][0] == 0.25 && out[1][0] == 0.5 && out[2][0] == 0.5 && out[3][0] == 1)
		testing.expect(t, strings.contains(m.msl, "switch (") && strings.contains(m.msl, "case 1:"))
		destroy(m)
	}
	cases := [?][2]string {
		{"half4 main(float2 p) {\n  switch (p.x) { case 1: break; }\n  return half4(1);\n}", "2:13: switch needs an int or uint, not float"},
		{"half4 main(float2 p) {\n  int k = int(p.x);\n  switch (1) { case k: break; }\n  return half4(1);\n}", "3:21: case label must be a constant integer expression"},
		{"half4 main(float2 p) {\n  switch (1) { case 1: case 1: break; }\n  return half4(1);\n}", "2:24: case 1 is already handled"},
		{"half4 main(float2 p) {\n  switch (1) { default: default: break; }\n  return half4(1);\n}", "2:25: a switch has only one default label"},
		{"half4 main(float2 p) {\n  switch (1) { p.x; case 1: break; }\n  return half4(1);\n}", "2:16: statements in a switch must come after a case or default label"},
		{"half4 main(float2 p) {\n  case 1: return half4(1);\n}", "2:3: case label outside a switch"},
		{"half4 main(float2 p) {\n  switch (1) { case 1: continue; }\n  return half4(1);\n}", "2:24: continue outside a loop"},
	}
	for c in cases {
		got := errors_of(c[0])
		testing.expectf(t, strings.contains(got, c[1]), "%q: got errors %q", c[1], got)
	}
}

@(test)
test_discard :: proc(t: ^testing.T) {
	// Discarded pixels are left untouched (the left half here, through a
	// helper function).
	src := `
void clip(float x) { if (x < 2.0) discard; }
void mainImage(out vec4 c, in vec2 p) { clip(p.x); c = vec4(1.0); }`
	m, errs, ok := compile_shadertoy(src)
	defer destroy(m)
	testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	testing.expect(t, strings.contains(m.msl, "discard_fragment();"))
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}})
	px := make([]u8, 4 * 2 * 4, context.temp_allocator)
	for &b in px {
		b = 9
	}
	run(&m.program, block, 4, 2, px, 16, temp_scratch())
	testing.expect_value(t, px[1 * 4], 9)
	testing.expect_value(t, px[2 * 4], 255)
	testing.expect_value(t, px[(4 + 3) * 4 + 3], 255)
}

@(test)
test_dialect_errors :: proc(t: ^testing.T) {
	shadertoy_errors :: proc(src: string) -> string {
		m, errs, ok := compile_shadertoy(src)
		defer destroy(m)
		return "" if ok else strings.clone(format_errors(errs, context.temp_allocator), context.temp_allocator)
	}
	// Errors point at the user's line, also on lines that use macros (an
	// argument keeps its own column, a macro body the macro's use).
	glsl := [?][2]string {
		{"#define SQ(x) ((x)*(x))\nvoid mainImage(out vec4 c, in vec2 p) {\n  float a = SQ(nope);\n  c = vec4(a);\n}", "3:16: undeclared identifier \"nope\""},
		{"#define BAD (missing * 2.0)\n\nvoid mainImage(out vec4 c, in vec2 p) {\n    c = vec4(BAD);\n}", "4:14: undeclared identifier \"missing\""},
		{"#define K 1.0\nvoid mainImage(out vec4 c, in vec2 p) {\n  c = vec4(nope);\n}", "3:12: undeclared identifier \"nope\""},
		// SkSL's features are not GLSL.
		{"uniform shader s;\nvoid mainImage(out vec4 c, in vec2 p) { c = vec4(1); }", "1:9: `shader` children are SkSL"},
		{"void mainImage(out vec4 c, in vec2 p) { c = iChannel0.eval(p); }", "1:55: .eval is SkSL"},
		{"void mainImage(out vec4 c, in vec2 p) { float2 q = p; c = vec4(q, 0, 1); }", "1:48: expected ';', found \"q\""},
		{"void mainImage(out vec4 c, in vec2 p) { c = vec4(saturate(p.x)); }", "unknown function \"saturate\""},
		{"void mainImage(out vec4 c, in vec2 p) { c = sk_FragCoord; }", "undeclared identifier \"sk_FragCoord\""},
		// The entry point.
		{"void image(out vec4 c, in vec2 p) { c = vec4(1); }", "1:1: no entry point: define void mainImage(out vec4 fragColor, in vec2 fragCoord)"},
		{"void mainImage(vec4 c, vec2 p) { }", "1:6: mainImage must be void mainImage(out vec4 fragColor, in vec2 fragCoord)"},
		{"vec4 main(vec2 p) { return vec4(1); }\nvoid mainImage(out vec4 c, in vec2 p) { c = vec4(1); }", "1:6: main is reserved in Shadertoy code"},
		{"uniform float iResolution;\nvoid mainImage(out vec4 c, in vec2 p) { c = vec4(1); }", "1:15: iResolution is a Shadertoy input of type vec3"},
		{"void mainImage(out vec4 c, in vec2 p) { vec4 x = textureSize(iChannel0, 0); }", "can not initialize vec4 x with ivec2"},
		{"vec4 f(sampler2D s) { return texelFetch(s, ivec2(0), 0); }\nvoid mainImage(out vec4 c, in vec2 p) { c = f(iChannel0); }", "1:41: texelFetch needs iChannel0..3 or a uniform sampler2D itself"},
	}
	for c in glsl {
		got := shadertoy_errors(c[0])
		testing.expectf(t, strings.contains(got, c[1]), "%q: got errors %q", c[1], got)
	}
	// GLSL's features are not SkSL (and its preprocessor is only #version).
	sksl := [?][2]string {
		{"half4 main(float2 p) {\n  if (p.x > 1.0) discard;\n  return half4(1);\n}", "2:18: discard is not allowed in SkSL"},
		{"uniform sampler2D t;\nhalf4 main(float2 p) { return half4(1); }", "1:9: sampler2D is not SkSL"},
		{"uniform shader s;\nhalf4 main(float2 p) { return texture(s, p); }", "2:31: texture() is not SkSL"},
		{"uniform shader s;\nhalf4 main(float2 p) { return texelFetch(s, int2(p), 0); }", "2:31: texelFetch() is not SkSL"},
		{"half4 main(float2 p) { return gl_FragCoord; }", "undeclared identifier \"gl_FragCoord\""},
		{"half4 main(float2 p) { return half4(1); }\n#define X 1\n", "2:1: SkSL has no preprocessor: #define is not supported"},
		{"#ifdef X\n#endif\nhalf4 main(float2 p) { return half4(1); }", "1:1: SkSL has no preprocessor: #ifdef is not supported"},
	}
	for c in sksl {
		got := errors_of(c[0])
		testing.expectf(t, strings.contains(got, c[1]), "%q: got errors %q", c[1], got)
	}
	testing.expect_value(t, errors_of("#version 300\nhalf4 main(float2 p) { return half4(1); }"), "")
}

// ---- errors -------------------------------------------------------------------------

@(test)
test_errors :: proc(t: ^testing.T) {
	cases := [?][2]string {
		{"half4 main(float2 p) {\n  return half4(1.0) +;\n}", "2:22: expected an expression"},
		{"half4 main(float2 p) {\n  return half4(nope);\n}", "2:16: undeclared identifier \"nope\""},
		{"half4 main(float2 p) {\n  float f = float2(1.0);\n  return half4(f);\n}", "2:13: can not initialize float f with float2"},
		{"uniform float u;\nhalf4 main(float2 p) {\n  u = 1.0;\n  return half4(u);\n}", "3:3: u is read-only"},
		{"float f(float x) { return g(x); }\nfloat g(float x) { return f(x); }\nhalf4 main(float2 p) { return half4(f(1.0)); }", "recursion is not supported"},
		{"void foo() {}", "no entry point: define half4 main(float2 coord)"},
		{"void main() {}", "1:6: the entry point must be half4 main(float2 coord)"},
		{"void mainImage(out vec4 c, in vec2 p) { c = vec4(1); }", "mainImage is Shadertoy's entry point"},
		{"half4 main(float2 p) {\n  float3 v = float3(1.0);\n  v.xx = float2(2.0);\n  return half4(v, 1.0);\n}", "repeats a component"},
		{"half4 main(float2 p) {\n  break;\n  return half4(1);\n}", "2:3: break outside a loop"},
		// The parser: what it expected and found (operands, not prose).
		{"half4 main(float2 p) {\n  return half4(1)", "2:18: expected ';', found end of file"},
		{"half4 main(float2 p) {\n  float2 v = p.;\n  return half4(1);\n}", "2:16: expected a field name, found \";\""},
		{"half4 main(float2 p) {\n  switch (1) { case 1 return half4(1); }\n  return half4(0);\n}", "2:23: expected ':' after the case value, found \"return\""},
		{"half4 main(float2 p) {\n  switch (1) return half4(1);\n  return half4(0);\n}", "2:14: expected '{' after switch (...), found \"return\""},
		{"half4 main(float2 p) {\n  float a[3];\n  return half4(a[3]);\n}", "3:18: index 3 is out of range"},
		// SkSL restrictions (GLSL's features: test_dialect_errors).
		{"uniform colorFilter f;\nhalf4 main(float2 p) { return half4(1); }", "only `uniform shader` children are supported"},
		{"uniform shader s;\nhalf4 f(shader x, float2 p) { return x.eval(p); }\nhalf4 main(float2 p) { return f(s, p); }", "2:9: shaders can not be parameters"},
		{"half4 main(float2 p) {\n  float x = 1.0;\n  return x.eval(p);\n}", "3:12: eval needs a `uniform shader` child, not float"},
		{"uniform shader s;\nhalf4 main(float2 p) { return s.eval(1.0); }", "s.eval takes a float2 coordinate, not float"},
		{"half4 main(float2 p) {\n  sk_FragCoord = float4(0);\n  return half4(1);\n}", "2:3: sk_FragCoord is const"},
	}
	for c in cases {
		got := errors_of(c[0])
		testing.expectf(t, strings.contains(got, c[1]), "%q: got errors %q", c[1], got)
	}
	// Several errors are reported at once.
	many := errors_of("half4 main(float2 p) {\n  float a = x1;\n  float b = x2;\n  return half4(1);\n}")
	testing.expect(t, strings.contains(many, "x1") && strings.contains(many, "x2"))
}

// Errors' text lives with the module (or pipeline), not in the temp
// allocator: hosts keep errors across frames that free it.
@(test)
test_errors_outlive_temp :: proc(t: ^testing.T) {
	sources := [?]string {
		"half4 main(float2 p) {\n  float f = float2(1.0);\n  return half4(f);\n}",
		"float f(float x) { return x; }\nhalf4 main(float2 p) { return half4(f(true)); }",
		"float f(float x) { return g(x); }\nfloat g(float x) { return f(x); }\nhalf4 main(float2 p) { return half4(f(1.0)); }",
	}
	wants := [?]string{"can not initialize float f with float2", "no overload of f takes (bool)", "f -> g -> f"}
	modules: [len(sources)]^Module
	errors: [len(sources)][]Error
	for src, i in sources {
		modules[i], errors[i], _ = compile(src)
	}
	defer for m in modules {destroy(m)}
	p, perrs, _ := compile_pipeline({vertex = "out vec2 a;\nvoid main() { a = vec2(0); }", fragment = "in vec3 a;\nout vec4 c;\nvoid main() { c = vec4(a, 1); }"})
	defer destroy_pipeline(p)
	free_all(context.temp_allocator)
	junk := make([]u8, 1 << 16, context.temp_allocator)
	for &b in junk {
		b = '#'
	}
	for want, i in wants {
		got := format_errors(errors[i], context.allocator)
		defer delete(got)
		testing.expectf(t, strings.contains(got, want), "%q: got errors %q", want, got)
	}
	got := format_errors(perrs, context.allocator)
	defer delete(got)
	testing.expectf(t, strings.contains(got, "a is vec2 in the vertex shader and vec3 in the fragment shader"), "got errors %q", got)
}

// ---- SkSL ----------------------------------------------------------------------------

@(test)
test_sksl_types :: proc(t: ^testing.T) {
	near(t, eval(t, "half2 h = half2(0.5, 1.5); float2x2 m = float2x2(1.0, 2.0, 3.0, 4.0); int2 i = int2(3, -4); bool2 b = bool2(true, false); fragColor = float4(m * float2(h), float(i.x + i.y), float(b.x) + float(b.y));"), {5, 7, -1, 1})
	near(t, eval(t, "float3x2 m = float3x2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0); half3x3 h = half3x3(2.0); mat2 g = mat2(1.0); fragColor = float4(m[2], h[1][1], g[1][1]);"), {5, 6, 2, 1})
	near(t, eval(t, "fragColor = float4(saturate(-0.5), saturate(0.25), saturate(float2(2.0, 0.5)));"), {0, 0.25, 1, 0.5})
	near(t, eval(t, "fragColor = sk_FragCoord;", prelude = ""), {0.5, 0.5, 0, 1})
}

@(test)
test_sksl_uniforms_and_children :: proc(t: ^testing.T) {
	src := `
uniform float3 iResolution;
uniform float iTime;
uniform int iFrame;
layout(color) uniform half4 uColor;
uniform float iTimeDelta2;
uniform shader uImage;
uniform shader uOther;

half4 main(float2 coord) {
    half4 c = uImage.eval(coord * 2.0);
    return half4(c.xy, uColor.b + iTime + float(iFrame), uOther.eval(float2(1.0, 3.0)).a);
}`
	m, errs, ok := compile(src)
	defer destroy(m)
	testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	// The built-in names bind to their fixed slots; the rest are user uniforms.
	testing.expect(t, m.inputs_used >= {.Resolution, .Time, .Frame})
	_, has_time := find_uniform(m, "iTime")
	testing.expect(t, !has_time)
	u, found := find_uniform(m, "uColor")
	testing.expect(t, found && u.offset == OFFSET_USER)
	testing.expect_value(t, m.num_samplers, 2)
	testing.expect_value(t, m.channel_names[0], "uImage")
	testing.expect_value(t, m.channel_names[1], "uOther")
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}, time = 0.5, frame = 2, channel_resolution = {{16, 8, 1}, {4, 4, 1}, {}, {}}})
	set_uniform(m, block, "uColor", {0, 0, 0.25, 1})
	// The sampler sees uv = coord / child size: report it back.
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
		return {uv.x, uv.y, lod, f32(channel)}
	}
	out := make([][4]f32, 8, context.temp_allocator)
	run_f32(&m.program, block, 4, 2, out, temp_scratch(), {sampler = sampler})
	// Pixel (1, 1): coord (1.5, 1.5), uImage at (3, 3) / (16, 8); uOther at (1, 3) / (4, 4) on channel 1.
	near(t, out[5], {3.0 / 16, 3.0 / 8, 2.75, 1})
}

// `return child.eval(T(...))`: lowering eval grows the expression array; the
// rewrite of the call must land in the grown array (it once wrote into the
// old one and failed with "eval takes [3] arguments, got 2").
@(test)
test_sksl_return_eval_of_constructor :: proc(t: ^testing.T) {
	src := `
uniform float3 iResolution;
uniform shader uScene;
half4 main(float2 coord) {
  return uScene.eval(float2(iResolution.x - coord.x, coord.y));
}`
	m, errs, ok := compile(src)
	defer destroy(m)
	testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}, channel_resolution = {{4, 2, 1}, {}, {}, {}}})
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
		return {uv.x, uv.y, lod, f32(channel)}
	}
	out := make([][4]f32, 8, context.temp_allocator)
	run_f32(&m.program, block, 4, 2, out, temp_scratch(), {sampler = sampler})
	// Pixel (0, 1): coord (0.5, 1.5), mirrored to (3.5, 1.5) / (4, 2).
	near(t, out[4], {3.5 / 4, 1.5 / 2, 0, 0})
}

// saturate() and eval() append to u.ids while the enclosing call's arguments
// and the enclosing statement list are being checked.
@(test)
test_sksl_ids_grow_inside_args_and_blocks :: proc(t: ^testing.T) {
	src := `
uniform float3 iResolution;
uniform shader uScene;
float g = max(saturate(-1.0), saturate(0.25));
half4 main(float2 coord) {
  float a = max(saturate(coord.x - 10.0), min(saturate(2.0), saturate(0.5)));
  {
    float b = saturate(1.5);
    a += b * max(uScene.eval(coord).x, saturate(-3.0));
  }
  return half4(a, g, uScene.eval(float2(saturate(9.0), saturate(-9.0))).xy);
}`
	m, errs, ok := compile(src)
	defer destroy(m)
	testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}, channel_resolution = {{4, 2, 1}, {}, {}, {}}})
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
		return {uv.x, uv.y, lod, f32(channel)}
	}
	out := make([][4]f32, 8, context.temp_allocator)
	run_f32(&m.program, block, 4, 2, out, temp_scratch(), {sampler = sampler})
	// Pixel (1, 0): a = 0.5 + 1 * eval((1.5, 0.5)).x; eval((1, 0)).xy = (1, 0) / (4, 2).
	near(t, out[1], {0.5 + 1.5 / 4, 0.25, 1.0 / 4, 0})
}

// ---- builtins on the CPU ------------------------------------------------------------

@(test)
test_builtin_math :: proc(t: ^testing.T) {
	near(t, eval(t, "fragColor = vec4(radians(180.0), degrees(3.14159265), sin(0.5), cos(0.5));"), {math.PI, 180, math.sin(f32(0.5)), math.cos(f32(0.5))}, 1e-4)
	near(t, eval(t, "fragColor = vec4(tan(0.3), asin(0.5), acos(0.5), atan(2.0));"), {math.tan(f32(0.3)), math.asin(f32(0.5)), math.acos(f32(0.5)), math.atan(f32(2))}, 1e-5)
	near(t, eval(t, "fragColor = vec4(atan(1.0, -1.0), sinh(1.0), cosh(1.0), tanh(0.5));"), {math.atan2(f32(1), -1), math.sinh(f32(1)), math.cosh(f32(1)), math.tanh(f32(0.5))}, 1e-5)
	near(t, eval(t, "fragColor = vec4(asinh(1.0), acosh(2.0), atanh(0.5), pow(2.0, 10.0));"), {0.8813736, 1.3169579, 0.5493061, 1024}, 1e-4)
	near(t, eval(t, "fragColor = vec4(exp(1.0), log(10.0), exp2(3.0), log2(8.0));"), {math.e, math.ln(f32(10)), 8, 3}, 1e-5)
	near(t, eval(t, "fragColor = vec4(sqrt(2.0), inversesqrt(4.0), abs(-3.0), sign(-2.0));"), {math.SQRT_TWO, 0.5, 3, -1}, 1e-6)
	near(t, eval(t, "fragColor = vec4(floor(-1.5), ceil(-1.5), fract(-1.25), trunc(-1.5));"), {-2, -1, 0.75, -1})
	near(t, eval(t, "fragColor = vec4(round(2.6), roundEven(2.5), roundEven(3.5), mod(-1.5, 1.0));"), {3, 2, 4, 0.5})
	near(t, eval(t, "fragColor = vec4(min(1.0, 2.0), max(vec2(1.0, 5.0), 3.0), clamp(7.0, 0.0, 5.0));"), {1, 3, 5, 5})
	near(t, eval(t, "fragColor = vec4(mix(0.0, 10.0, 0.25), step(0.5, 0.4), step(0.5, 0.6), smoothstep(0.0, 1.0, 0.5));"), {2.5, 0, 1, 0.5})
	near(t, eval(t, "fragColor = vec4(mix(vec2(1.0, 2.0), vec2(3.0, 4.0), bvec2(true, false)), float(isnan(sqrt(-1.0))), float(isinf(1.0 / 0.0)));"), {3, 2, 1, 1})
	near(t, eval(t, "fragColor = vec4(mod(vec2(5.5, -5.5), 2.0), clamp(vec2(-1.0, 2.0), vec2(0.0), vec2(1.0)));"), {1.5, 0.5, 0, 1})
}

@(test)
test_builtin_geometry :: proc(t: ^testing.T) {
	near(t, eval(t, "fragColor = vec4(length(vec3(1.0, 2.0, 2.0)), distance(vec2(1.0), vec2(4.0, 5.0)), dot(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)), 0.0);"), {3, 5, 32, 0})
	near(t, eval(t, "fragColor = vec4(cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)), 0.0);"), {0, 0, 1, 0})
	near(t, eval(t, "fragColor = vec4(normalize(vec3(3.0, 0.0, 4.0)), 0.0);"), {0.6, 0, 0.8, 0}, 1e-6)
	near(t, eval(t, "fragColor = vec4(faceforward(vec2(1.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 0.0)), reflect(vec2(1.0, -1.0), vec2(0.0, 1.0)));"), {-1, 0, 1, 1})
	r := eval(t, "fragColor = vec4(refract(normalize(vec2(1.0, -1.0)), vec2(0.0, 1.0), 0.5), refract(vec2(1.0, 0.0), vec2(0.0, 1.0), 2.0));")
	near(t, r, {0.35355338, -0.9354143, 0, 0}, 1e-5)
}

@(test)
test_builtin_matrix :: proc(t: ^testing.T) {
	near(t, eval(t, "mat2 m = mat2(1.0, 2.0, 3.0, 4.0); fragColor = vec4(m * vec2(1.0, 1.0), vec2(1.0, 1.0) * m);"), {4, 6, 3, 7})
	near(t, eval(t, "mat2 m = mat2(1.0, 2.0, 3.0, 4.0); mat2 i = inverse(m); fragColor = vec4(i[0], i[1]);"), {-2, 1, 1.5, -0.5}, 1e-5)
	near(t, eval(t, "mat3 m = mat3(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0); mat3 i = inverse(m) * m; fragColor = vec4(i[0][0], i[1][1], i[2][0], determinant(m));"), {1, 1, 0, 24}, 1e-5)
	near(t, eval(t, "mat4 m = mat4(2.0); m[3] = vec4(1.0, 2.0, 3.0, 1.0); mat4 i = inverse(m); fragColor = vec4(i[3].xyz, determinant(m));"), {-0.5, -1, -1.5, 8}, 1e-5)
	near(t, eval(t, "mat2x3 m = outerProduct(vec3(1.0, 2.0, 3.0), vec2(4.0, 5.0)); mat3x2 tr = transpose(m); fragColor = vec4(m[1][2], tr[2][1], matrixCompMult(mat2(2.0), mat2(3.0))[0][0], mat2(mat3(5.0))[1][1]);"), {15, 15, 6, 5})
	near(t, eval(t, "mat2 a = mat2(1.0, 2.0, 3.0, 4.0); mat2 b = a * a; fragColor = vec4(b[0], b[1]);"), {7, 10, 15, 22})
}

@(test)
test_builtin_relational_and_bits :: proc(t: ^testing.T) {
	near(t, eval(t, "bvec3 b = lessThan(vec3(1.0, 2.0, 3.0), vec3(2.0)); fragColor = vec4(vec3(b), float(any(b)) + float(all(b)) * 2.0);"), {1, 0, 0, 1})
	near(t, eval(t, "bvec2 e = equal(ivec2(1, 2), ivec2(1, 3)); fragColor = vec4(vec2(e), vec2(not(e)));"), {1, 0, 0, 1})
	near(t, eval(t, "fragColor = vec4(float(floatBitsToInt(1.0)), float(floatBitsToUint(-0.0) >> 31u), intBitsToFloat(1065353216), uintBitsToFloat(0x40000000u));"), {1065353216, 1, 1, 2})
	near(t, eval(t, "int a = -7; fragColor = vec4(float(a / 2), float(a % 3), float(abs(a)), float(sign(a)));"), {-3, -1, 7, -1})
	near(t, eval(t, "uint u = 0xF0u; fragColor = vec4(float(u >> 4u), float(u & 0x3Cu), float(u | 1u), float(~u & 0xFFu));"), {15, 48, 241, 15})
	near(t, eval(t, "int i = 1 << 4; i ^= 3; fragColor = vec4(float(i), float(-8 >> 1), float(min(3, -2)), float(clamp(10u, 2u, 5u)));"), {19, -4, -2, 5})
}

@(test)
test_derivatives :: proc(t: ^testing.T) {
	// Shadertoy's fragCoord grows upwards.
	near(t, eval_at(t, "fragColor = vec4(dFdx(fragCoord.x), dFdy(fragCoord.y), fwidth(fragCoord.x * 3.0), 0.0);", 1, 1), {1, -1, 3, 0})
	src := "half4 main(float2 p) { return half4(dFdx(p.x), dFdy(p.y), 0.0, 1.0); }"
	m, _, ok := compile(src)
	defer destroy(m)
	testing.expect(t, ok)
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}})
	out := make([][4]f32, 8, context.temp_allocator)
	run_f32(&m.program, block, 4, 2, out, temp_scratch())
	near(t, out[5], {1, 1, 0, 1})
}

@(test)
test_control_flow :: proc(t: ^testing.T) {
	near(t, eval(t, `
int count = 0;
for (int i = 0; i < 10; i++) {
    if (i == 2) continue;
    if (i == 6) break;
    count += i;
}
int j = 0;
do { j += 3; } while (j < 10);
int w = 5;
while (w > 0) { w -= 2; }
fragColor = vec4(float(count), float(j), float(w), 0.0);
`), {13, 12, -1, 0})
	near(t, eval(t, "float x = first(5.0); fragColor = vec4(x, 0.0, 0.0, 0.0);", prelude = `
float first(float n) {
    for (int i = 0; i < 100; i++) {
        if (float(i) * 2.0 > n) return float(i);
    }
    return -1.0;
}`), {3, 0, 0, 0})
	// Per-pixel divergence: the left half takes the branch.
	src := "if (fragCoord.x < 2.0) { return half4(1.0); } fragColor = vec4(0.5);"
	near(t, eval_at(t, src, 0, 0), {1, 1, 1, 1})
	near(t, eval_at(t, src, 3, 0), {0.5, 0.5, 0.5, 0.5})
	// Short circuit: the right side runs only where needed.
	near(t, eval(t, "int n = 0; bool b = false && bump(n); bool c = true || bump(n); bool d = true && bump(n); fragColor = vec4(float(n), float(b), float(c), float(d));", prelude = "bool bump(inout int n) { n++; return true; }"), {1, 0, 1, 1})
}

@(test)
test_structs_arrays_params :: proc(t: ^testing.T) {
	near(t, eval(t, `
S s = S(vec2(1.0, 2.0), 3.0);
s.v.y += 1.0;
float a[4] = float[4](1.0, 2.0, 3.0, 4.0);
int k = int(fragCoord.x) + 1;
a[k] = 10.0;
vec3 r;
split(vec3(5.0, 6.0, 7.0), r.x, r.yz);
fragColor = vec4(s.v.y + s.f, a[1], r.x + r.z, float(a.length()));
`, prelude = `
struct S { vec2 v; float f; };
void split(vec3 v, out float a, out vec2 b) { a = v.x; b = v.yz; }
`), {6, 10, 12, 4})
}

@(test)
test_uniforms :: proc(t: ^testing.T) {
	src := `
uniform float uA;
uniform float3 uB;
uniform float2 uArr[2];
uniform float2x2 uM;
uniform bool uFlag;
half4 main(float2 p) {
    if (p.x > 2.0) return half4(0.0);
    float2 m = uM * float2(1.0, 0.0);
    return half4(uA + uArr[1].y, uB.z, m.y, uFlag ? 1.0 : 0.0);
}`
	m, errs, ok := compile(src)
	defer destroy(m)
	testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	// std140 offsets after the built-ins.
	expect_uniform :: proc(t: ^testing.T, m: ^Module, name: string, offset, stride: int) {
		u, found := find_uniform(m, name)
		testing.expectf(t, found && u.offset == offset && u.stride == stride, "%s: %v", name, u)
	}
	expect_uniform(t, m, "uA", OFFSET_USER, 0)
	expect_uniform(t, m, "uB", OFFSET_USER + 16, 0)
	expect_uniform(t, m, "uArr", OFFSET_USER + 32, 16)
	expect_uniform(t, m, "uM", OFFSET_USER + 64, 0)
	expect_uniform(t, m, "uFlag", OFFSET_USER + 96, 0)
	testing.expect_value(t, m.block_size, OFFSET_USER + 112)
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {4, 2, 1}})
	set_uniform(m, block, "uA", {1})
	set_uniform(m, block, "uB", {0, 0, 5})
	set_uniform(m, block, "uArr", {2, 3}, 1)
	set_uniform(m, block, "uM", {7, 8, 9, 10})
	set_uniform(m, block, "uFlag", {1})
	px := make([]u8, 4 * 2 * 4, context.temp_allocator)
	for &b in px {
		b = 9
	}
	run(&m.program, block, 4, 2, px, 16, temp_scratch())
	// x = 0: (1 + 3, 5, 8, 1) clamped to bytes; x = 3: transparent.
	testing.expect_value(t, px[0], 255)
	testing.expect_value(t, px[3], 255)
	testing.expect_value(t, px[3 * 4 + 3], 0)
	f := make([][4]f32, 8, context.temp_allocator)
	run_f32(&m.program, block, 4, 2, f, temp_scratch())
	near(t, f[0], {4, 5, 8, 1})
	// The MSL struct carries the same layout.
	testing.expect(t, strings.contains(m.msl, "packed_float3 ok_uB;"))
	testing.expect(t, strings.contains(m.msl, "float4 ok_uArr[2];"))
	testing.expect(t, strings.contains(m.msl, "float4 ok_uM[2];"))
}

// ---- corpus -------------------------------------------------------------------------

// testdata/ by extension (SOURCE_EXTENSIONS).
@(private = "file")
compile_path :: proc(path: string, src: []u8) -> (^Module, []Error, bool) {
	options, _ := options_of_path(path)
	return compile(string(src), options)
}

@(private = "file")
corpus :: proc() -> []string {
	infos, _ := os.read_all_directory_by_path("testdata", context.temp_allocator)
	out := make([dynamic]string, context.temp_allocator)
	for fi in infos {
		if _, known := options_of_path(fi.name); known {
			append(&out, fi.fullpath)
		}
	}
	return out[:]
}

@(test)
test_corpus :: proc(t: ^testing.T) {
	files := corpus()
	testing.expect(t, len(files) >= 15)
	for path in files {
		src, _ := os.read_entire_file(path, context.temp_allocator)
		m, errs, ok := compile_path(path, src)
		testing.expectf(t, ok, "%s:\n%s", path, format_errors(errs, context.temp_allocator))
		if ok {
			testing.expect(t, len(m.spirv) > 5 && m.spirv[0] == 0x07230203)
			entry := fmt.tprintf("%s Oksl_Out oksl_main(", STAGE_RULES[m.stage].msl_qualifier)
			testing.expectf(t, strings.contains(m.msl, entry), "%s: no %s", path, entry)
			// Internal names ($channel_resolution) are not MSL identifiers.
			testing.expectf(t, !strings.contains(m.msl, "$"), "%s: '$' in MSL", path)
			if m.stage == .Fragment {
				block := make([]u8, m.block_size, context.temp_allocator)
				write_builtins(block, {resolution = {32, 16, 1}, time = 1, channel_resolution = {{8, 8, 1}, {8, 8, 1}, {8, 8, 1}, {8, 8, 1}}})
				px := make([]u8, 32 * 16 * 4, context.temp_allocator)
				run(&m.program, block, 32, 16, px, 32 * 4, temp_scratch())
			}
		}
		destroy(m)
	}
}

@(private = "file")
spirv_files: int

// spirv-val is a development tool: tests that use it skip without it.
@(private = "file")
have_spirv_val :: proc() -> bool {
	_, _, _, err := os.process_exec({command = {"spirv-val", "--version"}}, context.temp_allocator)
	return err == nil
}

@(private = "file")
expect_valid_spirv :: proc(t: ^testing.T, name: string, words: []u32, loc := #caller_location) {
	dir := os.temp_directory(context.temp_allocator) or_else "/tmp"
	// Tests run in parallel: one file per validation.
	out := fmt.tprintf("%s/oksl_test_%d_%d.spv", dir, os.get_pid(), sync.atomic_add(&spirv_files, 1))
	_ = os.write_entire_file(out, ([^]u8)(raw_data(words))[:len(words) * 4])
	state, stdout, stderr, err := os.process_exec({command = {"spirv-val", "--target-env", "vulkan1.0", out}}, context.temp_allocator)
	testing.expectf(t, err == nil && state.exit_code == 0, "%s: %s%s", name, stdout, stderr, loc = loc)
	os.remove(out)
}

// Validates every corpus module with spirv-val.
@(test)
test_corpus_spirv_val :: proc(t: ^testing.T) {
	if !have_spirv_val() {
		fmt.println("spirv-val not installed; skipping")
		return
	}
	for path in corpus() {
		src, _ := os.read_entire_file(path, context.temp_allocator)
		m, _, ok := compile_path(path, src)
		if ok {
			expect_valid_spirv(t, path, m.spirv)
		}
		destroy(m)
	}
}

// ---- GLSL fragment shaders ----------------------------------------------------------

// Several outputs, a flat integer input, implicit locations around an
// explicit one: reflection, the backends' interface and the CPU's
// varyings and targets.
@(test)
test_glsl_fragment :: proc(t: ^testing.T) {
	src := `#version 300 es
precision highp float;
in vec2 ramp;                              // location 1: 0 is taken
layout(location = 0) flat in ivec2 id;
layout(location = 1) out vec4 second;
layout(location = 0) out vec4 first;
layout(location = 2) out uvec2 count;
void main() {
    first = vec4(ramp, gl_FragCoord.xy);
    second = vec4(vec2(id), 0.0, 1.0);
    count = uvec2(gl_FragCoord.xy);
}`
	m, errs, ok := compile_glsl(src)
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, len(m.stage_inputs), 2)
	testing.expect_value(t, len(m.stage_outputs), 3)
	for v in m.stage_inputs {
		switch v.name {
		case "ramp":
			testing.expect(t, v.location == 1 && !v.flat && v.type == vec_type(.Float, 2))
		case "id":
			testing.expect(t, v.location == 0 && v.flat && v.type == vec_type(.Int, 2))
		case:
			testing.expectf(t, false, "unexpected input %s", v.name)
		}
	}
	for want in ([]string{"[[user(locn1)]]", "[[user(locn0), flat]]", "[[position]]", "[[color(1)]]", "[[color(2)]]", "fragment Oksl_Out oksl_main("}) {
		testing.expectf(t, strings.contains(m.msl, want), "MSL lacks %s", want)
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "test_glsl_fragment", m.spirv)
	}

	// 8x2 pixels: ramp = (x, 2y) at pixel centres, id = (3, 4) everywhere.
	W, H :: 8, 2
	targets: [3][W * H][4]f32
	slices := [3][][4]f32{targets[0][:], targets[1][:], targets[2][:]}
	varyings := []Varying{{a = {3, 4, 0, 0}}, {dx = {1, 0, 0, 0}, dy = {0, 2, 0, 0}}}
	block := make([]u8, m.block_size, context.temp_allocator)
	run_targets(&m.program, block, W, H, slices[:], temp_scratch(), {varyings = varyings})
	for y in 0 ..< H {
		for x in 0 ..< W {
			i := y * W + x
			px, py := f32(x) + 0.5, f32(y) + 0.5
			near(t, targets[0][i], {px, 2 * py, px, py})
			near(t, targets[1][i], {3, 4, 0, 1})
			near(t, targets[2][i], {f32(x), f32(y), 0, 1})
		}
	}
}

// texture() in a fragment shader samples at the level its uv derivatives
// pick in the channel's texels (one per quad), plus the bias; a channel
// without a size samples at the bias, and a vertex shader's texture() at
// level 0. textureSize and texelFetch read the sizes the host gives.
@(test)
test_glsl_texture_lod :: proc(t: ^testing.T) {
	src := `#version 300 es
precision highp float;
in vec2 st;
uniform sampler2D a;
uniform sampler2D b;
uniform sampler2D c;
layout(location = 0) out vec4 implicit;
layout(location = 1) out vec4 biased;
layout(location = 2) out vec4 unsized;
layout(location = 3) out vec4 sized;
layout(location = 4) out vec4 fetched;
void main() {
    implicit = texture(a, st);
    biased = texture(b, st, 1.0);
    unsized = texture(c, st, 0.5);
    sized = vec4(textureSize(a, 0), textureSize(b, 2));
    fetched = texelFetch(b, ivec2(3, 1), 2);
}`
	m, errs, ok := compile_glsl(src)
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	for want in ([]string{"ok_a_tex.sample(ok_a_smp, g.in_st, bias(0.0f))", "ok_b_tex.get_width(uint(2))"}) {
		testing.expectf(t, strings.contains(m.msl, want), "MSL lacks %s", want)
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "test_glsl_texture_lod", m.spirv)
	}
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
		return {uv.x, uv.y, lod, f32(channel)}
	}
	// st steps 1/16 across and 1/8 down a pixel: 4 x 4 texels of a
	// (64x32, lod 2) and 1 x 2 of b (16x16, lod 1).
	W, H :: 8, 2
	targets: [5][W * H][4]f32
	slices := [5][][4]f32{targets[0][:], targets[1][:], targets[2][:], targets[3][:], targets[4][:]}
	varyings := []Varying{{dx = {1.0 / 16, 0, 0, 0}, dy = {0, 1.0 / 8, 0, 0}}}
	sizes := [][2]i32{{64, 32}, {16, 16}}
	block := make([]u8, m.block_size, context.temp_allocator)
	run_targets(&m.program, block, W, H, slices[:], temp_scratch(), {sampler = sampler, varyings = varyings, channel_sizes = sizes})
	for i in 0 ..< W * H {
		st := [2]f32{(f32(i % W) + 0.5) / 16, (f32(i / W) + 0.5) / 8}
		near(t, targets[0][i], {st.x, st.y, 2, 0})
		near(t, targets[1][i], {st.x, st.y, 2, 1})
		near(t, targets[2][i], {st.x, st.y, 0.5, 2})
		near(t, targets[3][i], {64, 32, 4, 4})
		near(t, targets[4][i], {3.5 / 4, 1.5 / 4, 2, 1})
	}

	vs := `#version 300 es
in vec2 p;
uniform sampler2D t;
out vec4 v;
void main() { v = texture(t, p); gl_Position = vec4(p, 0.0, 1.0); }`
	vm, verrs, vok := compile_glsl(vs, {stage = .Vertex})
	defer destroy(vm)
	if !testing.expectf(t, vok, "%s", format_errors(verrs, context.temp_allocator)) {
		return
	}
	testing.expectf(t, strings.contains(vm.msl, "level(0.0f)"), "vertex MSL samples level 0")
	points := [2][2]f32{{0.25, 0.5}, {0.75, 0}}
	attrs := []Attribute{{&points[0], size_of([2]f32), .Float2}}
	positions: [2][4]f32
	vary := make([]f32, 2 * vm.varying_floats, context.temp_allocator)
	scratch: Scratch
	defer scratch_destroy(&scratch)
	vblock := make([]u8, max(vm.block_size, 16), context.temp_allocator)
	run_vertex(&vm.program, vblock, attrs, 2, positions[:], vary, &scratch, {sampler = sampler, channel_sizes = sizes})
	for i in 0 ..< 2 {
		r := vary[i * vm.varying_floats:]
		near(t, {r[0], r[1], r[2], r[3]}, {points[i].x, points[i].y, 0, 0})
	}
}

@(test)
test_glsl_fragment_errors :: proc(t: ^testing.T) {
	glsl_errors :: proc(src: string) -> string {
		m, errs, ok := compile_glsl(src)
		defer destroy(m)
		return "" if ok else strings.clone(format_errors(errs, context.temp_allocator), context.temp_allocator)
	}
	cases := [?][2]string {
		{"layout(location = 0) out vec4 a;\nlayout(location = 0) out vec4 b;\nvoid main() {}", "2:31: output location 0 is already used by a"},
		{"out vec4 a;\nout vec4 b;\nvoid main() {}", "1:10: with several outputs, a needs layout(location = N)"},
		{"layout(location = 9) out vec4 c;\nvoid main() {}", "1:31: output location 9 is out of range (0..7)"},
		{"in ivec2 i;\nvoid main() {}", "1:10: integer input i needs the flat qualifier"},
		{"in vec4 x = vec4(1);\nvoid main() {}", "1:9: input variable x can not have an initializer"},
		{"out mat2 m;\nvoid main() {}", "1:10: output variable m can not be mat2"},
		{"out bool b;\nvoid main() {}", "1:10: output variable b can not be bool"},
		{"varying vec2 v;\nvoid main() {}", "1:1: varying variables are GLSL ES 1.0"},
		{"in vec2 uv;\nvoid main() { uv = vec2(0); }", "2:15: uv is read-only"},
		{"out vec4 c;\nvoid main() { gl_FragCoord = vec4(0); }", "2:15: gl_FragCoord is read-only"},
		{"out vec4 c;\nvoid f() {}", "1:1: no entry point: define void main()"},
		{"out vec4 c;\nvec4 main() { return vec4(1); }", "2:6: the entry point must be void main()"},
		{"out vec4 f() { return vec4(1); }\nvoid main() {}", "1:10: a function can not be const, uniform, in or out"},
		{"out vec4 c;\nvoid main() { c = mainImage; }", "undeclared identifier \"mainImage\""},
	}
	for c in cases {
		got := glsl_errors(c[0])
		testing.expectf(t, strings.contains(got, c[1]), "%q: got errors %q", c[1], got)
	}
	// The layer profiles have no stage variables.
	testing.expect(t, strings.contains(errors_of("in float2 x;\nhalf4 main(float2 p) { return half4(1); }"), "1:1: global 'in' variables are not supported"))
	m, errs, ok := compile_shadertoy("out vec4 c;\nvoid mainImage(out vec4 o, in vec2 p) { o = vec4(1); }")
	defer destroy(m)
	testing.expect(t, !ok && strings.contains(format_errors(errs, context.temp_allocator), "1:1: global 'out' variables are not supported"))
}

// The interpreter reads consecutive swizzles (.xy, .yzw) in place; a
// swizzled store from the same variable must still see the old values.
@(test)
test_swizzle_overlap :: proc(t: ^testing.T) {
	near(t, eval(t, "float4 v = float4(1.0, 2.0, 3.0, 4.0); v.yzw = v.xyz; fragColor = v;"), {1, 1, 2, 3})
	near(t, eval(t, "float4 v = float4(1.0, 2.0, 3.0, 4.0); v.xyz = v.yzw; fragColor = v;"), {2, 3, 4, 4})
	near(t, eval(t, "float4 v = float4(1.0, 2.0, 3.0, 4.0); v.zw = v.xy; v.xy = v.zw; fragColor = v;"), {1, 2, 1, 2})
	near(t, eval(t, "float4 v = float4(1.0, 2.0, 3.0, 4.0); float2 a = v.yz; v.y = 9.0; fragColor = float4(a, v.yz);"), {2, 3, 9, 3})
	near(t, eval(t, "float2x2 m = float2x2(1.0, 2.0, 3.0, 4.0); m[1].yx = m[1].xy; fragColor = float4(m[0], m[1]);"), {1, 2, 4, 3})
}

// ---- vertex shaders and pipelines ---------------------------------------------------

// three.js's ShaderMaterial prefixes (abridged): #defines that rename
// keywords and built-ins, and declarations the shader may not use.
@(private = "file")
THREE_VERTEX_PRELUDE :: `#define attribute in
#define varying out
#define texture2D texture
precision highp float;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
`

@(private = "file")
THREE_FRAGMENT_PRELUDE :: `#define varying in
layout(location = 0) out highp vec4 pc_fragColor;
#define gl_FragColor pc_fragColor
#define texture2D texture
precision highp float;
uniform mat4 viewMatrix;
uniform vec3 cameraPosition;
`

@(test)
test_pipeline :: proc(t: ^testing.T) {
	vs := `varying vec2 vUv;
varying vec3 vNormal;
uniform float time;
void main() {
    vNormal = normalize(normalMatrix * normal);
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position + normal * sin(time), 1.0);
}`
	// The fragment shader declares the varyings in the other order.
	fs := `varying vec3 vNormal;
varying vec2 vUv;
uniform vec3 color;
uniform sampler2D map;
void main() {
    gl_FragColor = vec4(color * texture2D(map, vUv).rgb * max(vNormal.z, 0.0), 1.0);
    if (!gl_FrontFacing) gl_FragColor.rgb *= 0.5;
}`
	p, errs, ok := compile_pipeline({vertex = vs, fragment = fs, vertex_prelude = THREE_VERTEX_PRELUDE, fragment_prelude = THREE_FRAGMENT_PRELUDE})
	defer destroy_pipeline(p)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, p.vertex.stage, Stage.Vertex)
	testing.expect_value(t, p.fragment.stage, Stage.Fragment)
	// Only what each stage uses, from offset 0 (std140: mat3 is three
	// 16-byte columns).
	want_uniforms := [?]struct {
		name:   string,
		offset: int,
	}{{"modelViewMatrix", 0}, {"projectionMatrix", 64}, {"normalMatrix", 128}, {"time", 176}}
	testing.expect_value(t, len(p.vertex.uniforms), len(want_uniforms))
	for w in want_uniforms {
		off, _, found := uniform_offset(p.vertex, w.name)
		testing.expectf(t, found && off == w.offset, "%s at %d (found %v), want %d", w.name, off, found, w.offset)
	}
	_, _, found := uniform_offset(p.vertex, "modelMatrix")
	testing.expect(t, !found)
	testing.expect_value(t, p.vertex.block_size, 192)
	off, ct, color_found := uniform_offset(p.fragment, "color")
	testing.expect(t, color_found && off == 0 && ct == vec_type(.Float, 3))
	testing.expect_value(t, p.fragment.num_samplers, 1)
	// Attributes: position, normal, uv (all used), in declaration order.
	testing.expect_value(t, len(p.vertex.stage_inputs), 3)
	for v, i in p.vertex.stage_inputs {
		testing.expect_value(t, v.location, i)
	}
	// Each varying has one location in both stages.
	for fin in p.fragment.stage_inputs {
		matched := false
		for vout in p.vertex.stage_outputs {
			if vout.name == fin.name {
				matched = true
				testing.expectf(t, vout.location == fin.location && vout.type == fin.type, "%s: vertex %d, fragment %d", fin.name, vout.location, fin.location)
			}
		}
		testing.expectf(t, matched, "%s unmatched", fin.name)
	}
	testing.expect_value(t, len(p.fragment.stage_outputs), 1)
	for want in ([]string{"vertex Oksl_Out oksl_main(", "[[attribute(0)]]", "[[attribute(2)]]", "[[position]]", "[[user(locn1)]]"}) {
		testing.expectf(t, strings.contains(p.vertex.msl, want), "vertex MSL lacks %s", want)
	}
	testing.expect(t, strings.contains(p.fragment.msl, "[[front_facing]]"))
	if have_spirv_val() {
		expect_valid_spirv(t, "pipeline vertex", p.vertex.spirv)
		expect_valid_spirv(t, "pipeline fragment", p.fragment.spirv)
	}
}

@(test)
test_pipeline_errors :: proc(t: ^testing.T) {
	pipeline_errors :: proc(vs, fs: string, vp := "", fp := "") -> string {
		p, errs, ok := compile_pipeline({vertex = vs, fragment = fs, vertex_prelude = vp, fragment_prelude = fp})
		defer destroy_pipeline(p)
		return "" if ok else strings.clone(format_errors(errs, context.temp_allocator), context.temp_allocator)
	}
	VS :: "out vec2 a;\nflat out int k;\nvoid main() { a = vec2(0); k = 1; gl_Position = vec4(0); }"
	cases := [?][3]string {
		{VS, "in vec2 b;\nout vec4 c;\nvoid main() { c = vec4(b, 0, 1); }", "Fragment:1:9: fragment input b is not an output of the vertex shader"},
		{VS, "in vec3 a;\nout vec4 c;\nvoid main() { c = vec4(a, 1); }", "Fragment:1:9: a is vec2 in the vertex shader and vec3 in the fragment shader"},
		{VS, "in float k;\nout vec4 c;\nvoid main() { c = vec4(k); }", "Fragment:1:10: k is int in the vertex shader and float in the fragment shader"},
		{"out vec2 a;\nvoid main() { a = vec2(0); }", "flat in vec2 a;\nout vec4 c;\nvoid main() { c = vec4(a, 0, 1); }", "Fragment:1:14: a must be flat in both shaders or in neither"},
		{"out int k;\nvoid main() { k = 1; }", "flat in int k;\nout vec4 c;\nvoid main() { c = vec4(k); }", "Vertex:1:9: integer output k needs the flat qualifier"},
		// The vertex stage has no fragment features.
		{"void main() { discard; }", "out vec4 c;\nvoid main() { c = vec4(1); }", "Vertex:1:15: discard is only available in fragment shaders"},
		{"out float d;\nvoid main() { d = dFdx(1.0); }", "out vec4 c;\nvoid main() { c = vec4(1); }", "Vertex:2:19: dFdx is only available in fragment shaders"},
		{"void main() { gl_Position = gl_FragCoord; }", "out vec4 c;\nvoid main() { c = vec4(1); }", "Vertex:1:29: undeclared identifier \"gl_FragCoord\""},
		{"void main() { }", "out vec4 c;\nvoid main() { gl_Position = vec4(1); }", "Fragment:2:15: undeclared identifier \"gl_Position\""},
		{"void main() { }", "out vec4 c;\nvoid main() { c = vec4(1); }", ""},
		{"void f() { }", "out vec4 c;\nvoid main() { c = vec4(1); }", "Vertex:1:1: no entry point: define void main()"},
	}
	for c in cases {
		got := pipeline_errors(c[0], c[1])
		if c[2] == "" {
			testing.expectf(t, got == "", "expected no errors, got %q", got)
		} else {
			testing.expectf(t, strings.contains(got, c[2]), "%q: got errors %q", c[2], got)
		}
	}
	// Prelude errors name the prelude; its macros apply to the source.
	got := pipeline_errors("void main() { gl_Position = vec4(0); }", "out vec4 c;\nvoid main() { c = vec4(1); }", vp = "uniform mat4 m;\nuniform nope q;")
	testing.expectf(t, strings.contains(got, "Vertex prelude:2:9: unknown type \"nope\""), "got %q", got)
	got = pipeline_errors("void main() { gl_Position = POS; }", "out vec4 c;\nvoid main() { c = vec4(1); }", vp = "#define POS vec4(0.0, 0.0, 0.0, 1.0)\n")
	testing.expectf(t, got == "", "got %q", got)

	// A single compile: the stage must be one the profile has.
	m, errs, ok := compile("half4 main(float2 p) { return half4(1); }", {stage = .Vertex})
	defer destroy(m)
	testing.expect(t, !ok && strings.contains(format_errors(errs, context.temp_allocator), "SkSL source can not be a vertex shader"))
	m2, errs2, ok2 := compile_glsl("out vec4 c;\nvoid main() { c = vec4(X); }", {prelude = "#define X 0.5\nuniform float unused;\n"})
	defer destroy(m2)
	testing.expectf(t, ok2 && len(m2.uniforms) == 0 && m2.block_size == 0, "%s", format_errors(errs2, context.temp_allocator))
}

// ---- the CPU's vertices and quads ---------------------------------------------------

@(private = "file")
CPU_VS :: `in vec3 position;
in vec3 normal;
in vec2 uv;
out vec2 vUv;
out vec3 vNormal;
flat out ivec2 id;
uniform mat4 mvp;
void main() {
    vUv = uv * 2.0;
    vNormal = normal;
    id = ivec2(7, -3);
    gl_Position = mvp * vec4(position, 1.0);
}`

@(private = "file")
CPU_FS :: `in vec2 vUv;
in vec3 vNormal;
out vec4 c;
void main() {
    if (vNormal.x > 0.9) discard;
    c = vec4(vUv.x, gl_FragCoord.y, dFdx(vUv.x), gl_FrontFacing ? 1.0 : 0.5);
}`

// o3d's interleaved vertex: stride 32.
@(private = "file")
Test_Vertex :: struct {
	position: [3]f32,
	normal:   [3]f32,
	uv:       [2]f32,
}

@(test)
test_run_vertex :: proc(t: ^testing.T) {
	p, errs, ok := compile_pipeline({vertex = CPU_VS, fragment = CPU_FS})
	defer destroy_pipeline(p)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	vm := p.vertex
	// 19 vertices: a full batch of 16 and a partial one.
	N :: 19
	verts: [N]Test_Vertex
	for &v, i in verts {
		f := f32(i)
		v = {{f, f * 0.5, -f}, {0.1 * f, 0.2, 0.3}, {f * 0.01, 1 - f * 0.01}}
	}
	mvp := matrix[4, 4]f32{
		2, 0, 0, 1,
		0, 3, 0, -1,
		0, 0, 0.5, 0.25,
		0, 0, -1, 0,
	}
	block := make([]u8, vm.block_size, context.temp_allocator)
	off, _, found := uniform_offset(vm, "mvp")
	testing.expect(t, found)
	m := transmute([16]f32)mvp // column-major, as std140 stores it
	copy(block[off:], ([^]u8)(&m)[:64])
	// Attributes by reflected location; o3d's offsets in the vertex.
	attrs: [3]Attribute
	for a in vm.stage_inputs {
		switch a.name {
		case "position":
			attrs[a.location] = {&verts[0].position, size_of(Test_Vertex), .Float3}
		case "normal":
			attrs[a.location] = {&verts[0].normal, size_of(Test_Vertex), .Float3}
		case "uv":
			attrs[a.location] = {&verts[0].uv, size_of(Test_Vertex), .Float2}
		}
	}
	positions: [N][4]f32
	record := vm.varying_floats
	varyings := make([]f32, N * record, context.temp_allocator)
	scratch: Scratch
	defer scratch_destroy(&scratch)
	run_vertex(&vm.program, block, attrs[:], N, positions[:], varyings, &scratch)
	offsets: map[string]int
	defer delete(offsets)
	for o in vm.stage_outputs {
		offsets[o.name] = o.offset
	}
	testing.expect_value(t, record, 2 + 3 + 2)
	for v, i in verts {
		want := mvp * [4]f32{v.position.x, v.position.y, v.position.z, 1}
		near(t, positions[i], want)
		r := varyings[i * record:]
		near(t, {r[offsets["vUv"]], r[offsets["vUv"] + 1], 0, 0}, {v.uv.x * 2, v.uv.y * 2, 0, 0})
		near(t, {r[offsets["vNormal"]], r[offsets["vNormal"] + 1], r[offsets["vNormal"] + 2], 0}, {v.normal.x, v.normal.y, v.normal.z, 0})
		near(t, {r[offsets["id"]], r[offsets["id"] + 1], 0, 0}, {7, -3, 0, 0})
	}
	// The fragment stage reads the vertex shader's record.
	testing.expect_value(t, p.fragment.varying_floats, record)
	for fin in p.fragment.stage_inputs {
		testing.expect_value(t, fin.offset, offsets[fin.name])
	}
}

@(test)
test_run_fragment :: proc(t: ^testing.T) {
	p, errs, ok := compile_pipeline({vertex = CPU_VS, fragment = CPU_FS})
	defer destroy_pipeline(p)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	fm := p.fragment
	// 5 quads in a row (crossing the batch of 4), alternately facing.
	Q :: 5
	record := fm.varying_floats
	origin: [Q][2]i32
	facing: [Q]bool
	helpers: [Q]Quad_Lanes
	varyings := make([]f32, record * Q * 4, context.temp_allocator)
	uv_off, n_off := -1, -1
	for fin in fm.stage_inputs {
		if fin.name == "vUv" {uv_off = fin.offset}
		if fin.name == "vNormal" {n_off = fin.offset}
	}
	for q in 0 ..< Q {
		origin[q] = {i32(q * 2), 4}
		facing[q] = q % 2 == 0
		for i in 0 ..< 4 {
			j := q * 4 + i
			x := f32(q * 2 + (i & 1))
			varyings[uv_off * Q * 4 + j] = x * 0.25             // dFdx = 0.25
			varyings[(uv_off + 1) * Q * 4 + j] = 0.5
			varyings[n_off * Q * 4 + j] = 0.95 if j == 6 else 0 // lane 2 of quad 1 discards
		}
	}
	helpers[3] = {3}
	out: [Q * 4][4]f32
	shaded: [Q]Quad_Lanes
	scratch: Scratch
	defer scratch_destroy(&scratch)
	block := make([]u8, max(fm.block_size, 16), context.temp_allocator)
	run_fragment(&fm.program, block, {count = Q, origin = origin[:], front_facing = facing[:], helpers = helpers[:], varyings = varyings}, out[:], shaded[:], &scratch)
	for q in 0 ..< Q {
		want_lanes := Quad_Lanes{0, 1, 2, 3}
		if q == 1 {want_lanes -= {2}}
		if q == 3 {want_lanes -= {3}}
		testing.expectf(t, shaded[q] == want_lanes, "quad %d shaded %v", q, shaded[q])
		for i in 0 ..< 4 {
			if i not_in shaded[q] {
				continue
			}
			x := f32(q * 2 + (i & 1))
			y := f32(4 + (i >> 1)) + 0.5
			near(t, out[q * 4 + i], {x * 0.25, y, 0.25, 1 if q % 2 == 0 else 0.5})
		}
	}
}

// Shading a layer's pixels as quads, with uv supplied as a varying, gives
// what the layer runner gives, bit for bit; and so do threads sharing the
// program, each with its own scratch.
@(test)
test_run_fragment_matches_layer :: proc(t: ^testing.T) {
	src, _ := os.read_entire_file("testdata/fragment.frag", context.temp_allocator)
	m, errs, ok := compile_glsl(string(src))
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	W, H :: 16, 8
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
		return {uv.x, uv.y, 0.5, 1}
	}
	block := make([]u8, m.block_size, context.temp_allocator)
	set_uniform(m, block, "uSpeed", {0.5})
	want := make([][4]f32, W * H, context.temp_allocator)
	for &p in want {p = {-1, -1, -1, -1}}
	run_f32(&m.program, block, W, H, want, temp_scratch(), {sampler = sampler})

	Q :: (W / 2) * (H / 2)
	origin := make([][2]i32, Q, context.temp_allocator)
	varyings := make([]f32, 2 * Q * 4, context.temp_allocator)
	for q in 0 ..< Q {
		origin[q] = {i32(q % (W / 2) * 2), i32(q / (W / 2) * 2)}
		for i in 0 ..< 4 {
			x, y := origin[q].x + i32(i & 1), origin[q].y + i32(i >> 1)
			varyings[0 * Q * 4 + q * 4 + i] = (f32(x) + 0.5) / W
			varyings[1 * Q * 4 + q * 4 + i] = (f32(y) + 0.5) / H
		}
	}
	quads := Fragment_Quads{count = Q, origin = origin, varyings = varyings}
	shade :: proc(m: ^Module, block: []u8, quads: Fragment_Quads, out: [][4]f32, shaded: []Quad_Lanes) {
		scratch: Scratch
		defer scratch_destroy(&scratch)
		run_fragment(&m.program, block, quads, out, shaded, &scratch, {sampler = proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {return {uv.x, uv.y, 0.5, 1}}})
	}
	compare :: proc(t: ^testing.T, quads: Fragment_Quads, want, out: [][4]f32, shaded: []Quad_Lanes) {
		for q in 0 ..< quads.count {
			for i in 0 ..< 4 {
				x, y := int(quads.origin[q].x) + (i & 1), int(quads.origin[q].y) + (i >> 1)
				w := want[y * W + x]
				if i in shaded[q] {
					testing.expectf(t, transmute([4]u32)out[q * 4 + i] == transmute([4]u32)w, "(%d, %d): %v, layer %v", x, y, out[q * 4 + i], w)
				} else {
					testing.expectf(t, w == {-1, -1, -1, -1}, "(%d, %d) discarded on quads only", x, y)
				}
			}
		}
	}
	out := make([][4]f32, Q * 4, context.temp_allocator)
	shaded := make([]Quad_Lanes, Q, context.temp_allocator)
	shade(m, block, quads, out, shaded)
	compare(t, quads, want, out, shaded)

	// Four threads on one program, each with its own scratch.
	Job :: struct {
		m:      ^Module,
		block:  []u8,
		quads:  Fragment_Quads,
		out:    [][4]f32,
		shaded: []Quad_Lanes,
	}
	jobs: [4]Job
	threads: [4]^thread.Thread
	for &j, n in jobs {
		j = {m, block, quads, make([][4]f32, Q * 4, context.temp_allocator), make([]Quad_Lanes, Q, context.temp_allocator)}
		threads[n] = thread.create_and_start_with_data(&j, proc(data: rawptr) {
			j := (^Job)(data)
			for _ in 0 ..< 50 {
				shade(j.m, j.block, j.quads, j.out, j.shaded)
			}
		})
	}
	for th in threads {
		thread.join(th)
		thread.destroy(th)
	}
	for j in jobs {
		compare(t, quads, want, j.out, j.shaded)
	}
}

// ---- dialect-neutral pass graphs ----------------------------------------------------

// Two SkSL passes, rendered as a host would: pass 0 into a float target,
// pass 1 reading it through its child (channel 0) as this frame's output.
@(test)
test_make_program_sksl :: proc(t: ^testing.T) {
	m0, e0, ok0 := compile("half4 main(float2 p) { return half4(p.x / 8.0, p.y / 4.0, 0.25, 1.0); }")
	m1, e1, ok1 := compile("uniform shader prev;\nhalf4 main(float2 p) { half4 c = prev.eval(p); return half4(c.g, c.r, c.b * 2.0, 1.0); }")
	testing.expectf(t, ok0 && ok1, "%s%s", format_errors(e0, context.temp_allocator), format_errors(e1, context.temp_allocator))
	g, errs, ok := make_program({{module = m0, name = "first"}, {module = m1, name = "second", channels = {0 = {.Buffer, 0}}}})
	defer destroy_program(g)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, g.output, 1)
	testing.expect_value(t, len(g.edges), 1)
	testing.expect(t, g.edges[0] == Edge{1, 0, .Buffer, 0})
	testing.expect_value(t, g.passes[1].name, "second")
	testing.expect_value(t, g.index[.Image], -1)

	W, H :: 8, 4
	first := make([][4]f32, W * H, context.temp_allocator)
	for p in g.passes {
		block := make([]u8, p.module.block_size, context.temp_allocator)
		write_builtins(p.module, block, {resolution = {W, H, 1}, channel_resolution = {{W, H, 1}, {}, {}, {}}})
		if p.name == "first" {
			run_f32(&p.module.program, block, W, H, first, temp_scratch())
			continue
		}
		out := make([][4]f32, W * H, context.temp_allocator)
		// The channel reads the first pass's target, nearest texel.
		run_f32(&p.module.program, block, W, H, out, temp_scratch(), {user = raw_data(first), sampler = proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
			src := ([^][4]f32)(user)
			x, y := clamp(int(uv.x * W), 0, W - 1), clamp(int(uv.y * H), 0, H - 1)
			return src[y * W + x]
		}})
		for y in 0 ..< H {
			for x in 0 ..< W {
				c := first[y * W + x]
				near(t, out[y * W + x], {c.g, c.r, c.b * 2, 1})
			}
		}
	}
}

// Feedback: a pass reading its own previous frame, and an earlier pass's
// previous frame (what Shadertoy's order rule can't say).
@(test)
test_make_program_feedback :: proc(t: ^testing.T) {
	SRC :: "uniform shader a;\nuniform shader b;\nhalf4 main(float2 p) { return a.eval(p) * 0.5 + b.eval(p) * 0.5; }"
	m0, _, _ := compile("half4 main(float2 p) { return half4(1); }")
	m1, _, _ := compile(SRC)
	g, errs, ok := make_program({{module = m0}, {module = m1, channels = {0 = {.Previous, 1}, 1 = {.Previous, 0}}}})
	defer destroy_program(g)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, len(g.edges), 2)
	testing.expect(t, g.edges[0] == Edge{1, 0, .Previous, 1} && g.edges[1] == Edge{1, 1, .Previous, 0})
}

// SkSL, GLSL and an unclamped Shadertoy pass in one graph; write_builtins
// with the module writes the inputs only where the block has them.
@(test)
test_make_program_mixed :: proc(t: ^testing.T) {
	sk, _, ok0 := compile("uniform float gain;\nhalf4 main(float2 p) { return half4(gain); }")
	gl, _, ok1 := compile_glsl("in vec2 uv;\nout vec4 c;\nuniform float gain;\nuniform sampler2D src;\nvoid main() { c = texture(src, uv) * gain; }")
	st, _, ok2 := compile_shadertoy("void mainImage(out vec4 c, in vec2 p) { c = vec4(3.0, 2.0, 1.0, 0.5); }", {unclamped = true})
	testing.expect(t, ok0 && ok1 && ok2)
	testing.expect(t, sk.profile == .Sksl && gl.profile == .Glsl && st.profile == .Shadertoy)
	g, errs, ok := make_program({{module = st}, {module = sk}, {module = gl, channels = {0 = {.Buffer, 0}}}}, output = 2)
	defer destroy_program(g)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	// The GLSL block is its own uniforms from offset 0: write_builtins
	// leaves it alone.
	off, _, found := uniform_offset(gl, "gain")
	testing.expect(t, found && off == 0)
	block := make([]u8, gl.block_size, context.temp_allocator)
	set_uniform(gl, block, "gain", {0.75})
	write_builtins(gl, block, {resolution = {9, 9, 1}, time = 5})
	write_input(gl, block, .Time, {7})
	testing.expect_value(t, (^f32)(&block[0])^, 0.75)
	// The SkSL block has the inputs ahead of its uniforms.
	sblock := make([]u8, sk.block_size, context.temp_allocator)
	write_builtins(sk, sblock, {resolution = {9, 9, 1}})
	testing.expect_value(t, (^f32)(&sblock[OFFSET_RESOLUTION])^, 9)
	// Unclamped Shadertoy keeps mainImage's colour as it is.
	sblock2 := make([]u8, st.block_size, context.temp_allocator)
	write_builtins(st, sblock2, {resolution = {4, 2, 1}})
	px := make([][4]f32, 8, context.temp_allocator)
	run_f32(&st.program, sblock2, 4, 2, px, temp_scratch())
	near(t, px[0], {3, 2, 1, 0.5})
}

@(test)
test_make_program_errors :: proc(t: ^testing.T) {
	sampling :: proc() -> ^Module {
		m, _, _ := compile("uniform shader s;\nhalf4 main(float2 p) { return s.eval(p); }")
		return m
	}
	plain :: proc() -> ^Module {
		m, _, _ := compile("half4 main(float2 p) { return half4(1); }")
		return m
	}
	program_errors :: proc(passes: []Pass_Spec, output := -1) -> string {
		g, errs, ok := make_program(passes, output)
		defer destroy_program(g) // the graph owns the modules, also on failure
		return "" if ok else strings.clone(format_errors(errs, context.temp_allocator), context.temp_allocator)
	}
	got := program_errors({{module = sampling(), name = "blur", channels = {0 = {.Buffer, 0}}}})
	testing.expectf(t, strings.contains(got, "blur's channel 0 reads pass 0, which does not run before it"), "got %q", got)
	got = program_errors({{module = plain()}, {module = sampling(), channels = {0 = {.Previous, 5}}}})
	testing.expectf(t, strings.contains(got, "pass 1's channel 0 reads pass 5, which does not exist"), "got %q", got)
	got = program_errors({{module = plain()}}, output = 3)
	testing.expectf(t, strings.contains(got, "the output pass 3 is not one of the 1 passes"), "got %q", got)
	many: [PASS_MAX + 1]Pass_Spec
	for &p in many {
		p.module = plain()
	}
	got = program_errors(many[:])
	testing.expectf(t, strings.contains(got, "a program has 1 to 8 passes, not 9"), "got %q", got)
}
