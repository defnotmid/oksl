package oksl

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
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
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
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
	// Cases no lane reaches do not run: their texture reads call no sampler.
	{
		dm, errs, ok := compile_shadertoy(`
void mainImage(out vec4 c, in vec2 p) {
    c = vec4(0.25);
    switch (int(iTime)) {
    case 0:
        c = texture(iChannel0, p / iResolution.xy);
        break;
    case 2:
        c = vec4(0.5);
        break;
    default:
        c = texelFetch(iChannel0, ivec2(0), 0);
    }
}`)
		defer destroy(dm)
		testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
		if ok {
			count := 0
			sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
				(^int)(user)^ += 1
				return {1, 0, 0, 1}
			}
			block := make([]u8, dm.block_size, context.temp_allocator)
			write_builtins(block, {resolution = {4, 2, 1}, time = 2})
			px := make([][4]f32, 8, context.temp_allocator)
			run_f32(&dm.program, block, 4, 2, px, temp_scratch(), {sampler = sampler, user = &count})
			testing.expectf(t, px[0] == {0.5, 0.5, 0.5, 1} && count == 0, "got %v after %d samples", px[0], count)
		}
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
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
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
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
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
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
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

// The CPU lowering reads values in place where it can (an unwritten
// parameter, a variable initialized from a value made for it); where the
// source would see a difference it copies.
@(test)
test_values_in_place :: proc(t: ^testing.T) {
	// A global the function writes, passed to it: the parameter keeps the
	// value at the call.
	near(t, eval(t, "float r = keep(g); fragColor = vec4(r, g, 0.0, 0.0);", prelude = `
float g = 1.0;
float keep(float x) { g = 5.0; return x; }
`), {1, 5, 0, 0})
	// A parameter the function writes is its own; a variable initialized from
	// a constant does not change the constant.
	near(t, eval(t, "float a = 2.0; float b = bump(a); float c = 2.0; c += 1.0; fragColor = vec4(a, b, c, 2.0);", prelude = `
float bump(float x) { x += 1.0; return x; }
`), {2, 3, 3, 2})
	// A returned parameter is the argument's value at the call, though the
	// caller assigns the argument before using it.
	near(t, eval(t, "float a = 1.0; float r = same(a) + (a = 5.0); fragColor = vec4(r, a, 0.0, 0.0);", prelude = `
float same(float x) { return x; }
`), {6, 5, 0, 0})
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
			ret := "void" if m.stage == .Compute else "Oksl_Out"
			entry := fmt.tprintf("%s %s oksl_main(", STAGE_RULES[m.stage].msl_qualifier, ret)
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
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		return {uv.x, uv.y, lod, f32(channel)}
	}
	// st steps 1/16 across and 1/8 down a pixel: 4 x 4 texels of a
	// (64x32, lod 2) and 1 x 2 of b (16x16, lod 1).
	W, H :: 8, 2
	targets: [5][W * H][4]f32
	slices := [5][][4]f32{targets[0][:], targets[1][:], targets[2][:], targets[3][:], targets[4][:]}
	varyings := []Varying{{dx = {1.0 / 16, 0, 0, 0}, dy = {0, 1.0 / 8, 0, 0}}}
	sizes := [][3]i32{{64, 32, 1}, {16, 16, 1}}
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

// The layer vertex shader hosts compile (LAYER_VERTEX_GLSL): position and
// uv at locations 0 and 1, uv out at location 0, the target size alone in
// the block, and target pixels (y down) to clip space.
@(test)
test_layer_vertex :: proc(t: ^testing.T) {
	m, errs, ok := compile(LAYER_VERTEX_GLSL, {profile = .Glsl, stage = .Vertex})
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	locations := [2]int{-1, -1}
	for a in m.stage_inputs {
		switch a.name {
		case "position":
			locations[0] = a.location
		case "uv":
			locations[1] = a.location
		}
	}
	testing.expect_value(t, locations, [2]int{0, 1})
	testing.expect(t, len(m.stage_outputs) == 1 && m.stage_outputs[0].location == 0)
	size, found := find_uniform(m, "size")
	testing.expect(t, found && size.offset == 0 && m.block_size == 16)
	Layer_Vertex :: struct {
		position, uv: [2]f32,
	}
	verts := [2]Layer_Vertex{{{0, 0}, {0, 0}}, {{320, 180}, {1, 1}}}
	attrs := [2]Attribute{{&verts[0].position, size_of(Layer_Vertex), .Float2}, {&verts[0].uv, size_of(Layer_Vertex), .Float2}}
	block := make([]u8, m.block_size, context.temp_allocator)
	set_uniform(m, block, "size", {320, 180, 0, 0})
	positions: [2][4]f32
	varyings := make([]f32, 2 * m.varying_floats, context.temp_allocator)
	scratch: Scratch
	defer scratch_destroy(&scratch)
	run_vertex(&m.program, block, attrs[:], 2, positions[:], varyings, &scratch)
	testing.expectf(t, positions == {{-1, 1, 0, 1}, {1, -1, 0, 1}}, "positions %v", positions)
	uv := varyings[m.varying_floats + m.stage_outputs[0].offset:][:2]
	testing.expectf(t, uv[0] == 1 && uv[1] == 1, "second vertex's uv %v", uv)
}

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
	// A full batch and a partial one.
	N :: LANES + 3
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
		// Integers travel as their bits.
		testing.expect_value(t, transmute([2]i32)[2]f32{r[offsets["id"]], r[offsets["id"] + 1]}, [2]i32{7, -3})
	}
	// The fragment stage reads the vertex shader's record.
	testing.expect_value(t, p.fragment.varying_floats, record)
	for fin in p.fragment.stage_inputs {
		testing.expect_value(t, fin.offset, offsets[fin.name])
	}
}

// A vertex shader for a tap: a helper that counts its calls in a global,
// a global with an initializer, and an output.
@(private = "file")
TAP_VS :: `in vec3 position;
in vec3 normal;
out vec3 vNormal;
uniform mat4 mvp;
uniform float scale;
float calls = 0.0;
float base = 2.0;
float bump(float x) {
    calls += 1.0;
    return x * scale;
}
void main() {
    base += 1.0;
    float s = bump(1.0);
    vNormal = normal * s + vec3(calls, base, 0.0);
    float b = bump(1.0);
    gl_Position = mvp * vec4(position * b * calls, 1.0);
}`

@(private = "file")
TAP_PRELUDE :: "in vec3 openPosition;\nin vec3 openNormal;\nout vec4 openClip;\n"

@(private = "file")
TAP_FS :: "in vec3 vNormal;\nout vec4 c;\nvoid main() { c = vec4(vNormal, 1.0); }"

// A tap runs what main does toward the position again with the next
// environment's uniforms and its twins, into its position, and leaves
// main's own run alone; given the first environment and the same
// attributes it writes gl_Position bit for bit, and it equals a run of the
// untapped shader in its environment.
@(test)
test_vertex_taps :: proc(t: ^testing.T) {
	taps := []Tap{{position = "openClip", attributes = {{"position", "openPosition"}, {"normal", "openNormal"}}}}
	p, errs, ok := compile_pipeline({vertex = TAP_VS, fragment = TAP_FS, vertex_prelude = TAP_PRELUDE}, {taps = taps})
	defer destroy_pipeline(p)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	vm := p.vertex
	testing.expect_value(t, vm.taps, 1)
	testing.expect_value(t, vm.block_size, 2 * vm.block_stride)
	testing.expect_value(t, len(vm.uniforms), 2)
	// bump's call counts for the position, the normal does not: its twin
	// is no input.
	for want in ([]string{"ok_1_main(", "ok_1_bump(", "ok_1_mvp", "g.ok_1_calls"}) {
		testing.expectf(t, strings.contains(vm.msl, want), "tapped MSL lacks %s", want)
	}
	for dead in ([]string{"ok_1_vNormal", "ok_1_base", "openNormal"}) {
		testing.expectf(t, !strings.contains(vm.msl, dead), "tapped MSL has %s", dead)
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "tapped vertex", vm.spirv)
	}

	N :: LANES + 3
	verts: [N]Test_Vertex
	open: [N][3]f32
	for &v, i in verts {
		f := f32(i)
		v = {{f, f * 0.5, -f}, {0.1 * f, 0.2, 0.3}, {}}
		open[i] = {f - 1, f * 0.25, 2 - f}
	}
	mvps := [2]matrix[4, 4]f32 {
		{2, 0, 0, 1, 0, 3, 0, -1, 0, 0, 0.5, 0.25, 0, 0, -1, 0},
		{1, 0, 0, 0.5, 0, 2, 0, 0, 0, 0, 1, -0.5, 0, 0, -1, 1},
	}
	scales := [2]f32{1.5, 0.75}
	block := make([]u8, vm.block_size, context.temp_allocator)
	for e in 0 ..< 2 {
		env := block[e * vm.block_stride:]
		m := transmute([16]f32)mvps[e]
		testing.expect(t, set_uniform(vm, env, "mvp", m[:]))
		testing.expect(t, set_uniform(vm, env, "scale", {scales[e]}))
	}
	attrs: [3]Attribute
	twin := -1
	for a in vm.stage_inputs {
		switch a.name {
		case "position":
			attrs[a.location] = {&verts[0].position, size_of(Test_Vertex), .Float3}
		case "normal":
			attrs[a.location] = {&verts[0].normal, size_of(Test_Vertex), .Float3}
		case "openPosition":
			attrs[a.location] = {&open[0], size_of([3]f32), .Float3}
			twin = a.location
		}
	}
	testing.expect(t, twin >= 0)
	offsets: map[string]int
	defer delete(offsets)
	for o in vm.stage_outputs {
		offsets[o.name] = o.offset
	}
	clip_at :: proc(varyings: []f32, record, at, i: int) -> [4]f32 {
		r := varyings[i * record + at:]
		return {r[0], r[1], r[2], r[3]}
	}
	record := vm.varying_floats
	positions: [N][4]f32
	varyings := make([]f32, N * record, context.temp_allocator)
	scratch: Scratch
	defer scratch_destroy(&scratch)
	run_vertex(&vm.program, block, attrs[:], N, positions[:], varyings, &scratch)
	for v, i in verts {
		// Two calls of bump when the position is made.
		p := v.position * scales[0] * 2
		near(t, positions[i], mvps[0] * [4]f32{p.x, p.y, p.z, 1})
		o := open[i] * scales[1] * 2
		near(t, clip_at(varyings, record, offsets["openClip"], i), mvps[1] * [4]f32{o.x, o.y, o.z, 1})
		// The tap's globals are its own: main sees one call of bump and base 3.
		n := varyings[i * record + offsets["vNormal"]:]
		near(t, {n[0], n[1], n[2], 0}, {v.normal.x * scales[0] + 1, v.normal.y * scales[0] + 3, v.normal.z * scales[0], 0})
	}

	// The untapped shader in the second environment, on the twins.
	plain, plain_errs, plain_ok := compile_pipeline({vertex = TAP_VS, fragment = TAP_FS, vertex_prelude = TAP_PRELUDE})
	defer destroy_pipeline(plain)
	if !testing.expectf(t, plain_ok, "%s", format_errors(plain_errs, context.temp_allocator)) {
		return
	}
	pv := plain.vertex
	testing.expect_value(t, pv.block_size, vm.block_stride)
	plain_attrs: [2]Attribute
	for a in pv.stage_inputs {
		switch a.name {
		case "position":
			plain_attrs[a.location] = {&open[0], size_of([3]f32), .Float3}
		case "normal":
			plain_attrs[a.location] = {&verts[0].normal, size_of(Test_Vertex), .Float3}
		}
	}
	plain_positions: [N][4]f32
	plain_varyings := make([]f32, N * pv.varying_floats, context.temp_allocator)
	run_vertex(&pv.program, block[vm.block_stride:], plain_attrs[:], N, plain_positions[:], plain_varyings, &scratch)
	for i in 0 ..< N {
		got := clip_at(varyings, record, offsets["openClip"], i)
		testing.expectf(t, transmute([4]u32)got == transmute([4]u32)plain_positions[i], "vertex %d: tap %v, untapped %v", i, got, plain_positions[i])
	}

	// The first environment twice, the twins the attributes themselves.
	copy(block[vm.block_stride:], block[:vm.block_stride])
	attrs[twin] = {&verts[0].position, size_of(Test_Vertex), .Float3}
	run_vertex(&vm.program, block, attrs[:], N, positions[:], varyings, &scratch)
	for i in 0 ..< N {
		got := clip_at(varyings, record, offsets["openClip"], i)
		testing.expectf(t, transmute([4]u32)got == transmute([4]u32)positions[i], "vertex %d: tap %v, gl_Position %v", i, got, positions[i])
	}
}

// Every tap, given the first environment and the attributes themselves as
// twins, writes gl_Position bit for bit: over loops whose breaks and
// continues the position depends on, early returns, inout and out
// parameters, a switch and initializers that call functions.
@(test)
test_taps_match :: proc(t: ^testing.T) {
	LOOPS :: `in vec3 position;
in vec3 normal;
out float vDead;
uniform float limit;
float acc = 0.0;
void grow(inout float h, out float d) {
    h += 0.5;
    d = h * 3.0;
}
float shape(float x) {
    if (x < 0.0) return -x;
    vDead = x;
    return x * 2.0;
}
void place(vec3 p) {
    if (p.x < 0.0) {
        gl_Position = vec4(-p, 1.0);
        return;
    }
    gl_Position = vec4(p, 1.0);
}
void main() {
    float h = 0.0;
    float dead;
    for (int i = 0; i < 8; i++) {
        if (h > limit * 8.0) break;
        if (i == 2) continue;
        grow(h, dead);
        h += float(i);
        acc += dead;
    }
    switch (int(limit)) {
    case 0: h += 1.0; break;
    case 1: h += 2.0; break;
    default: vDead = 1.0;
    }
    place(position * shape(h - 20.0));
}`
	INIT :: `in vec3 position;
in vec3 normal;
out vec4 vColor;
uniform float u;
float k = 2.0;
float helper(float x) {
    vColor = vec4(x);
    return x * k + u;
}
float g = helper(1.0);
void main() {
    gl_Position = vec4(position * g, 1.0);
}`
	PRELUDE :: "in vec3 open0;\nin vec3 open1;\nin vec3 openNormal;\nout vec4 tap0;\nout vec4 tap1;\n"
	taps := []Tap {
		{position = "tap0", attributes = {{"position", "open0"}}},
		{position = "tap1", attributes = {{"position", "open1"}, {"normal", "openNormal"}}},
	}
	// No position at all: the taps' stay zero too.
	SILENT :: "in vec3 position;\nin vec3 normal;\nout vec3 vNormal;\nvoid main() { vNormal = normal; }"
	for src in ([]string{TAP_VS, LOOPS, INIT, SILENT}) {
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Vertex, prelude = PRELUDE, taps = taps})
		defer destroy(m)
		if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
			continue
		}
		// One copy per tap of what the position needs.
		testing.expectf(t, !strings.contains(m.msl, "2_1_") && !strings.contains(m.msl, "ok_1_vDead"), "copies of copies or dead outputs:\n%s", m.msl)
		block := make([]u8, m.block_size, context.temp_allocator)
		for u, k in m.uniforms {
			values := [16]f32{}
			for &v, i in values {
				v = 0.75 + f32(k) * 0.5 + f32(i) * 0.125
			}
			set_uniform(m, block, u.name, values[:min(u.size / 4, 16)])
		}
		for e in 1 ..= m.taps {
			copy(block[e * m.block_stride:], block[:m.block_stride])
		}
		N :: LANES + 3
		verts: [N]Test_Vertex
		for &v, i in verts {
			f := f32(i)
			v = {{f * 0.25 - 4, 1 - f * 0.125, f * 0.5}, {0.1 * f, 0.2, 0.3}, {}}
		}
		attrs: [8]Attribute
		for a in m.stage_inputs {
			testing.expectf(t, a.name != "openNormal", "the twin of an unused attribute is an input")
			field := &verts[0].normal if a.name == "normal" else &verts[0].position
			attrs[a.location] = {field, size_of(Test_Vertex), .Float3}
		}
		positions: [N][4]f32
		varyings := make([]f32, N * m.varying_floats, context.temp_allocator)
		scratch: Scratch
		defer scratch_destroy(&scratch)
		run_vertex(&m.program, block, attrs[:], N, positions[:], varyings, &scratch)
		for o in m.stage_outputs {
			if o.name != "tap0" && o.name != "tap1" {
				continue
			}
			for i in 0 ..< N {
				r := varyings[i * m.varying_floats + o.offset:]
				got := [4]f32{r[0], r[1], r[2], r[3]}
				if got != positions[i] {
					testing.expectf(t, false, "%s, vertex %d: %v, gl_Position %v", o.name, i, got, positions[i])
					break
				}
			}
		}
	}
}

@(test)
test_tap_errors :: proc(t: ^testing.T) {
	VS :: "in vec3 position;\nvoid main() { gl_Position = vec4(position, 1.0); }"
	Case :: struct {
		prelude: string,
		tap:     Tap,
		want:    string,
	}
	cases := [?]Case {
		{"", {position = "openClip"}, "a tap's position openClip must be declared `out vec4`"},
		{"out vec3 openClip;\n", {position = "openClip"}, "Prelude:1:10: a tap's position openClip must be declared `out vec4`"},
		{"out vec4 openClip;\n", {position = "openClip", attributes = {{"normal", "openNormal"}}}, "a tap's attribute normal must be an `in` variable"},
		{"out vec4 openClip;\nin vec2 openPosition;\n", {position = "openClip", attributes = {{"position", "openPosition"}}}, "the tap twin openPosition must be declared `in vec3`, like position"},
		{"out vec4 openClip;\nvoid touch() { openClip = vec4(0.0); }\n", {position = "openClip"}, "the shader uses openClip, a tap's position"},
	}
	for c in cases {
		m, errs, ok := compile(VS, {profile = .Glsl, stage = .Vertex, prelude = c.prelude, taps = []Tap{c.tap}})
		got := format_errors(errs, context.temp_allocator)
		testing.expectf(t, !ok && strings.contains(got, c.want), "%q: got errors %q", c.want, got)
		destroy(m)
	}
	{
		twice := []Tap{{position = "openClip"}, {position = "openClip"}}
		m, errs, ok := compile(VS, {profile = .Glsl, stage = .Vertex, prelude = "out vec4 openClip;\n", taps = twice})
		got := format_errors(errs, context.temp_allocator)
		testing.expectf(t, !ok && strings.contains(got, "two taps write openClip"), "got errors %q", got)
		destroy(m)
	}
	{
		// Recursion is the source's error, once, about the source's names.
		src := "in vec3 position;\nfloat f(float x) { return f(x); }\nfloat g = f(1.0);\nvoid main() { gl_Position = vec4(position * g, 1.0); }"
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Vertex, prelude = "out vec4 openClip;\n", taps = []Tap{{position = "openClip"}}})
		got := format_errors(errs, context.temp_allocator)
		testing.expectf(t, !ok && strings.count(got, "recursion") == 1 && !strings.contains(got, "1_f"), "got errors %q", got)
		destroy(m)
	}
	m, errs, ok := compile("out vec4 c;\nvoid main() { c = vec4(1.0); }", {profile = .Glsl, taps = []Tap{{position = "c"}}})
	defer destroy(m)
	got := format_errors(errs, context.temp_allocator)
	testing.expectf(t, !ok && strings.contains(got, "taps are for vertex shaders, not fragment shaders"), "got errors %q", got)
}

@(test)
test_run_fragment :: proc(t: ^testing.T) {
	p, errs, ok := compile_pipeline({vertex = CPU_VS, fragment = CPU_FS})
	defer destroy_pipeline(p)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	fm := p.fragment
	// A row of quads crossing a batch, alternately facing.
	Q :: LANES / 4 + 1
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
	targets := [1][][4]f32{out[:]}
	run_fragment(&fm.program, block, {count = Q, origin = origin[:], front_facing = facing[:], helpers = helpers[:], varyings = varyings}, targets[:], shaded[:], &scratch)
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

// A layer shaded in regions, starting on odd pixels or before the layer
// and running past it, is the layer shaded whole: the quads (and so the
// derivatives) stay the layer's, and each region writes only its pixels.
@(test)
test_run_regions :: proc(t: ^testing.T) {
	m, errs, ok := compile_shadertoy("void mainImage(out vec4 c, in vec2 f) { float v = f.x * f.x + f.y * f.y; c = vec4(dFdx(v), -dFdy(v), 0.0, 1.0) / 64.0; }")
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	W, H :: 16, 16
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {W, H, 1}})
	whole := make([][4]f32, W * H, context.temp_allocator)
	run_f32(&m.program, block, W, H, whole, temp_scratch())
	parts := make([][4]f32, W * H, context.temp_allocator)
	UNSET :: [4]f32{-1, -1, -1, -1}
	for &px in parts {
		px = UNSET
	}
	run_f32(&m.program, block, W, H, parts, temp_scratch(), {x = 3, y = 5, w = 9, h = 7})
	for y in 0 ..< H {
		for x in 0 ..< W {
			want := whole[y * W + x] if x >= 3 && x < 12 && y >= 5 && y < 12 else UNSET
			if parts[y * W + x] != want {
				testing.expectf(t, false, "region at (3, 5): pixel (%d, %d) is %v, want %v", x, y, parts[y * W + x], want)
				return
			}
		}
	}
	regions := [?]Run_Options{{x = -3, y = -1, w = W + 6, h = 6}, {x = -1, y = 5, w = 4, h = 7}, {x = 12, y = 5, w = 9, h = 7}, {x = 0, y = 12, w = W, h = 9}}
	for r in regions {
		run_f32(&m.program, block, W, H, parts, temp_scratch(), r)
	}
	for i in 0 ..< W * H {
		if parts[i] != whole[i] {
			testing.expectf(t, false, "pixel (%d, %d) is %v in regions, %v whole", i % W, i / W, parts[i], whole[i])
			return
		}
	}
}

// What follows from uniforms and constants alone runs once per run, before
// the batches (code[:init_len]), and gives what it gave in place, inside a
// loop or a branch. A variable written later, here through a dynamic index
// whose element changes between batches, keeps registers of its own, so no
// batch reads another's stores.
@(test)
test_uniform_hoisting :: proc(t: ^testing.T) {
	src := `
uniform float uT;
float2x2 rot(float a) { float c = cos(a), s = sin(a); return float2x2(c, s, -s, c); }
half4 main(float2 p) {
    float2 q = float2(1.0, 0.0);
    float acc = 0.0;
    for (int i = 0; i < 3; i++) {
        q = rot(uT) * q;
        acc += q.x;
    }
    if (p.x > 9.0) {
        acc += sin(uT * 2.0);
    }
    float2 v = float2(uT, uT + 1.0);
    v[int(p.x) / 8] = 0.0;
    float w = uT * 2.0;
    if (p.x > 9.0) {
        w += 1.0;
    }
    return half4(acc, v.x + 10.0 * v.y, w, q.y);
}`
	m, errs, ok := compile(src)
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	trig :: proc(in_: Instr) -> bool {
		return in_.op == .F_Fn1 && (Builtin(in_.k) == .Sin || Builtin(in_.k) == .Cos)
	}
	early, late := 0, 0
	for in_, i in m.program.code {
		if trig(in_) {
			if i < m.program.init_len {
				early += 1
			} else {
				late += 1
			}
		}
	}
	testing.expectf(t, early == 3 && late == 0, "sin and cos: %d run once, %d per batch", early, late)
	W, H :: 16, 16
	T :: 0.7
	block := make([]u8, m.block_size, context.temp_allocator)
	write_builtins(block, {resolution = {W, H, 1}})
	set_uniform(m, block, "uT", {T})
	out := make([][4]f32, W * H, context.temp_allocator)
	run_f32(&m.program, block, W, H, out, temp_scratch())
	for y in 0 ..< H {
		for x in 0 ..< W {
			acc := math.cos(f32(T)) + math.cos(f32(2 * T)) + math.cos(f32(3 * T))
			w := f32(2 * T)
			if x >= 9 {
				acc += math.sin(f32(2 * T))
				w += 1
			}
			v := f32(10 * (T + 1)) if x < 8 else f32(T)
			want := [4]f32{acc, v, w, math.sin(f32(3 * T))}
			got := out[y * W + x]
			for k in 0 ..< 4 {
				if abs(got[k] - want[k]) > 1e-5 {
					testing.expectf(t, false, "pixel (%d, %d): got %v, want %v", x, y, got, want)
					return
				}
			}
		}
	}
	// textureSize of a sampler reads only the run's channel sizes, so it
	// runs once too.
	sm, serrs, sok := compile_glsl("#version 300 es\nprecision highp float;\nuniform sampler2D tex;\nout vec4 c;\nvoid main() { c = vec4(vec2(textureSize(tex, 0)) / 64.0, 0.0, 1.0); }")
	defer destroy(sm)
	if !testing.expectf(t, sok, "%s", format_errors(serrs, context.temp_allocator)) {
		return
	}
	sized := false
	for in_ in sm.program.code[:sm.program.init_len] {
		sized ||= in_.op == .Texture_Size
	}
	testing.expect(t, sized)
	sblock := make([]u8, sm.block_size, context.temp_allocator)
	write_builtins(sblock, {resolution = {4, 2, 1}})
	px := make([][4]f32, 8, context.temp_allocator)
	sizes := [][3]i32{{32, 16, 1}}
	run_f32(&sm.program, sblock, 4, 2, px, temp_scratch(), {channel_sizes = sizes})
	testing.expectf(t, px[7] == {0.5, 0.25, 0, 1}, "got %v", px[7])
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
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
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
		targets := [1][][4]f32{out}
		run_fragment(&m.program, block, quads, targets[:], shaded, &scratch, {sampler = proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {return {uv.x, uv.y, 0.5, 1}}})
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
		run_f32(&p.module.program, block, W, H, out, temp_scratch(), {user = raw_data(first), sampler = proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
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

@(test)
test_audit_regressions :: proc(t: ^testing.T) {
	// 1. Nested composite constructors where the inner argument emits a
	// trailing Gather/Mov for its first component after placing later ones.
	near(t, eval(t, "float a = 2.0, b = 5.0; fragColor = float4(float2(b, a + 1.0), 7.0, 9.0);"), {5, 3, 7, 9})
	near(t, eval(t, "float a = 4.0; fragColor = float4(pair(a), 7.0, 9.0);", prelude = `
float2 pair(float x) { return float2(x, x + 1.0); }
`), {4, 5, 7, 9})
	near(t, eval(t, "float a = 2.0, b = 6.0; Outer o = Outer(Inner(b, a + 3.0), 8.0); fragColor = float4(o.p.x, o.p.y, o.z, 1.0);", prelude = `
struct Inner { float x; float y; };
struct Outer { Inner p; float z; };
`), {6, 5, 8, 1})

	// 2. Unused uniform arrays in GLSL profile must not emit broken MSL helpers.
	{
		m, errs, ok := compile_glsl("uniform vec4 unused_v[4];\nuniform mat4 unused_m[2];\nout vec4 c;\nvoid main() { c = vec4(1.0); }")
		defer destroy(m)
		testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
		testing.expect(t, !strings.contains(m.msl, "unused_v") && !strings.contains(m.msl, "unused_m"))
		if have_spirv_val() {
			expect_valid_spirv(t, "unused_uniform_arrays", m.spirv)
		}
	}

	// 3. Preprocessor: token pasting with empty left/middle arguments,
	// leading ## error, multiline comments inside #define, and backslash-newline before '('.
	{
		got, errs := pp_text("#define P(a, b) x + a ## b\n#define P3(a, b, c) x + a ## b ## c\nP(, y) P3(, , z) P3(, y, z) P3(y, , z)")
		testing.expectf(t, got == "x + y x + z x + yz x + yz" && errs == "", "got %q, errs %q", got, errs)
		got, errs = pp_text("#define FOO /*\n*/ 123\n#define FN\\\n(x) ((x) + FOO)\nFN(4)")
		testing.expectf(t, got == "( ( 4 ) + 123 )" && errs == "", "got %q, errs %q", got, errs)
		bad_m, _, bad_ok := compile_shadertoy("#define F(x) ## x\nvoid mainImage(out vec4 c, in vec2 p) { float y = F(1.0); c = vec4(y); }")
		defer destroy(bad_m)
		testing.expect(t, !bad_ok)
	}

	// 4. Non-void functions that fall off the end without returning:
	// deterministic zero in CPU VM (even with dirty scratch), valid SPIR-V and MSL.
	{
		src := `
float empty_fn(float x) { float y = x + 1.0; }
float partial_fn(float x) { if (x > 10.0) return 9.0; }
half4 main(float2 p) {
    return half4(empty_fn(p.x), partial_fn(p.x), partial_fn(20.0), 1.0);
}`
		m, errs, ok := compile(src)
		defer destroy(m)
		testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
		if ok {
			testing.expect(t, strings.contains(m.msl, "return {};"))
			if have_spirv_val() {
				expect_valid_spirv(t, "non_void_falloff", m.spirv)
			}
			scratch := temp_scratch()
			resize(&scratch.regs, max(m.program.num_slots, 16))
			for &r in scratch.regs {
				r = ~u32(0)
			}
			block := make([]u8, m.block_size, context.temp_allocator)
			write_builtins(block, {resolution = {4, 2, 1}})
			out := make([][4]f32, 8, context.temp_allocator)
			run_f32(&m.program, block, 4, 2, out, scratch)
			near(t, out[0], {0, 0, 9, 1})
		}
	}

	// 5. SPIR-V structured CFG: loops with unconditional break/return and
	// short-circuit step/condition, and switch fallthrough with declarations.
	{
		src := `#version 300 es
precision highp float;
uniform int uSel;
out vec4 c;
float early_do(bool a, bool b) {
    do {
        return 2.0;
    } while (a && b);
}
void main() {
    float v = early_do(true, false);
    for (int i = 0; i < 4; i += (uSel > 0 && v > 0.0 ? 1 : 2)) {
        v += 1.0;
        break;
    }
    switch (uSel) {
    case 0:
        int x = 3;
        v += float(x);
    case 1:
        int y = 4;
        v += float(y);
        break;
    default:
        v = -1.0;
        break;
    }
    c = vec4(v, 0.0, 0.0, 1.0);
}`
		m, errs, ok := compile_glsl(src)
		defer destroy(m)
		testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
		if ok {
			if have_spirv_val() {
				expect_valid_spirv(t, "loop_and_switch_cfg", m.spirv)
			}
			block := make([]u8, m.block_size, context.temp_allocator)
			set_uniform(m, block, "uSel", {0})
			out := make([][4]f32, 4, context.temp_allocator)
			run_f32(&m.program, block, 2, 2, out, temp_scratch())
			near(t, out[0], {10, 0, 0, 1})
		}
	}

	// 6. Copy-in / copy-out for out/inout parameters in MSL and VM: swizzles,
	// aliased arguments, an index evaluated once and before the call (a side
	// effect; a global the callee changes), an unwritten out.
	{
		src := `
float g = 3.0;
int k = 0;
float swap_add(inout float2 v, inout float a, out float b) {
    v = v.yx + float2(1.0, 2.0);
    a += g;
    return b + a;
}
void bump_elem(inout float x) {
    k = 1;
    x += 10.0;
}
void unwritten(out float y) {
}
half4 main(float2 p) {
    float4 q = float4(1.0, 2.0, 3.0, 4.0);
    float x = 5.0;
    float r = swap_add(q.zw, x, x);
    float a[2];
    a[0] = 1.0;
    a[1] = 2.0;
    int i = 0;
    bump_elem(a[i++]);
    k = 0;
    bump_elem(a[k]);
    float z = 7.0;
    unwritten(z);
    return half4(a[0] + float(i), a[1], z, r);
}`
		m, errs, ok := compile(src)
		defer destroy(m)
		testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator))
		if ok {
			testing.expect(t, strings.contains(m.msl, "(ok_q.zw = _t0)"))
			testing.expect(t, strings.count(m.msl, "ok_i++") == 1)
			testing.expect(t, strings.contains(m.msl, "((_t6 = g.ok_k), (_t7 = ok_a[_t6]), ok_bump_elem(_t7, u, g), (ok_a[_t6] = _t7))"))
			testing.expect(t, strings.contains(m.msl, "(_t8 = float{}), ok_unwritten(_t8"))
			if have_spirv_val() {
				expect_valid_spirv(t, "copy_in_out_swizzle", m.spirv)
			}
			block := make([]u8, m.block_size, context.temp_allocator)
			write_builtins(block, {resolution = {4, 2, 1}})
			out := make([][4]f32, 8, context.temp_allocator)
			run_f32(&m.program, block, 4, 2, out, temp_scratch())
			near(t, out[0], {22, 2, 0, 8})
		}
	}

	// 7. set_uniform array index bounds checking.
	{
		m, _, ok := compile("uniform float uScalar;\nuniform float2 uArr[2];\nhalf4 main(float2 p) { return half4(uScalar + uArr[0].x); }")
		defer destroy(m)
		testing.expect(t, ok)
		block := make([]u8, m.block_size, context.temp_allocator)
		testing.expect(t, set_uniform(m, block, "uScalar", {1}, 0))
		testing.expect(t, !set_uniform(m, block, "uScalar", {1}, 1))
		testing.expect(t, !set_uniform(m, block, "uScalar", {1}, -1))
		testing.expect(t, set_uniform(m, block, "uArr", {1, 2}, 1))
		testing.expect(t, !set_uniform(m, block, "uArr", {1, 2}, 2))
		testing.expect(t, !set_uniform(m, block, "uArr", {1, 2}, -1))
	}
}

// ---- storage buffers, gl_VertexID, array textures, textureGrad ---------------------

// Read-only storage buffers: struct elements at their std430 offsets, a
// per-lane loop whose trip count is a buffer word, a read from a helper,
// a uniform index, and reads past the host's words (0).
@(test)
test_storage_buffers :: proc(t: ^testing.T) {
	src := `#version 310 es
precision highp float;
struct Row { vec4 color; vec3 probe; float w; ivec2 m; };
layout(std430) readonly buffer Rows { Row rows[]; } rowBuf;
readonly buffer Words { uint words[]; };
uniform int uPick;
out vec4 o;
uint word(uint i) { return words[i]; }
void main() {
    int i = int(gl_FragCoord.x);
    Row r = rowBuf.rows[i];
    uint sum = 0u;
    for (uint k = 0u; k < word(uint(i)); k++) {
        sum += words[8u + k];
    }
    o = vec4(r.color.x + r.probe.z, float(r.m.y), float(sum), rowBuf.rows[uPick].w + float(words[1000]) + float(rowBuf.rows[-i - 1].m.x));
}`
	m, errs, ok := compile_glsl(src)
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	// vec4, vec3 and float, then ivec2 at 32: 40 bytes, 48 with the
	// struct's 16-byte alignment.
	testing.expect_value(t, len(m.storage_buffers), 2)
	testing.expect_value(t, m.storage_buffers[0].name, "Rows")
	testing.expect_value(t, m.storage_buffers[0].stride, 48)
	testing.expect_value(t, m.storage_buffers[1].name, "Words")
	testing.expect_value(t, m.storage_buffers[1].stride, 4)
	testing.expect_value(t, m.num_uniform_buffers, 1)
	for want in ([]string{"const device uint* oksl_buf_rowBuf [[buffer(1)]]", "const device uint* oksl_buf_words [[buffer(2)]]", "ok_Row{float4(as_type<float>(w[b + 0u])", "as_type<int>(w[b + 9u])"}) {
		testing.expectf(t, strings.contains(m.msl, want), "MSL lacks %s", want)
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "test_storage_buffers", m.spirv)
	}
	ROWS :: 4
	rows := make([]u32, ROWS * 12, context.temp_allocator)
	for i in 0 ..< ROWS {
		r := rows[i * 12:]
		for k in 0 ..< 4 {r[k] = transmute(u32)f32(10 * i + k)}
		for k in 0 ..< 3 {r[4 + k] = transmute(u32)f32(100 + i)}
		r[7] = transmute(u32)(0.5 * f32(i))
		r[8], r[9] = u32(i32(-i)), u32(i32(-2 * i))
	}
	words := make([]u32, 16, context.temp_allocator)
	for i in 0 ..< 4 {
		words[i] = u32(i)
		words[8 + i] = 1 << uint(i)
	}
	block := make([]u8, m.block_size, context.temp_allocator)
	testing.expect(t, set_uniform(m, block, "uPick", {2}))
	out := make([][4]f32, 4, context.temp_allocator)
	run_f32(&m.program, block, 4, 1, out, temp_scratch(), {buffers = {rows, words}})
	sums := [4]f32{0, 1, 3, 7}
	for i in 0 ..< 4 {
		near(t, out[i], {f32(11 * i + 100), f32(-2 * i), sums[i], 1})
	}
	// Word addresses wrap at 32 bits, as SPIR-V's and Metal's do: element
	// 2^30 of a 12-word element is element 0.
	wm, werrs, wok := compile_glsl("struct Row { vec4 color; vec3 probe; float w; ivec2 m; };\nreadonly buffer Rows { Row rows[]; };\nuniform uint uWrap;\nout vec4 o;\nvoid main() { o = rows[uWrap].color; }")
	defer destroy(wm)
	if testing.expectf(t, wok, "%s", format_errors(werrs, context.temp_allocator)) {
		wblock := make([]u8, wm.block_size, context.temp_allocator)
		testing.expect(t, set_uniform(wm, wblock, "uWrap", {1 << 30}))
		run_f32(&wm.program, wblock, 4, 1, out, temp_scratch(), {buffers = {rows}})
		near(t, out[0], {0, 1, 2, 3})
	}
	// No buffers bound: every read is 0.
	run_f32(&m.program, block, 4, 1, out, temp_scratch())
	for i in 0 ..< 4 {
		near(t, out[i], {0, 0, 0, 0})
	}

	cases := [?][2]string {
		{"buffer B { uint w[]; } b;\nvoid main() {}", "buffer B is not readonly: buffers are read-only"},
		{"layout(std140) readonly buffer B { uint w[]; } b;\nvoid main() {}", "buffer B is std140: buffers are laid out std430"},
		{"readonly buffer B { uint w[]; uint x[]; } b;\nvoid main() {}", "buffer B holds one runtime-sized array"},
		{"readonly buffer B { uint w[4]; } b;\nvoid main() {}", "buffer B holds one runtime-sized array"},
		{"readonly buffer B { mat4 w[]; } b;\nvoid main() {}", "mat4x4 is not a buffer element"},
		{"readonly buffer B { bool w[]; } b;\nvoid main() {}", "bool is not a buffer element"},
		{"readonly buffer B { uint w[]; } b;\nout vec4 c;\nvoid main() { b.w[0] = 1u; }", "is read-only"},
		{"readonly buffer B { uint w[]; } b;\nout vec4 c;\nvoid main() { uint x = b.w.length(); }", "b.w is a buffer: read an element, b.w[i]"},
		{"readonly buffer B { uint w[]; };\nout vec4 c;\nvoid main() { uint x = w; }", "w is a buffer: read an element, w[i]"},
		{"readonly buffer B { uint w[]; };\nout vec4 c;\nvoid main() { uint x = w[-1]; }", "out of range"},
	}
	for c in cases {
		em, eerrs, eok := compile_glsl(c[0])
		got := "" if eok else format_errors(eerrs, context.temp_allocator)
		destroy(em)
		testing.expectf(t, strings.contains(got, c[1]), "%q: got errors %q", c[1], got)
	}
	// SkSL and Shadertoy have no buffers.
	testing.expect(t, errors_of("readonly buffer B { uint w[]; } b;\nhalf4 main(float2 p) { return half4(1); }") != "")
	sm, _, sok := compile_shadertoy("readonly buffer B { uint w[]; } b;\nvoid mainImage(out vec4 o, in vec2 p) { o = vec4(1); }")
	destroy(sm)
	testing.expect(t, !sok)
}

// gl_VertexID counts from Run_Options.first_vertex, as a draw's first
// vertex counts on the GPU, and indexes a buffer of morph deltas.
@(test)
test_vertex_id :: proc(t: ^testing.T) {
	vs := `#version 310 es
struct Delta { vec4 position; vec4 normal; };
readonly buffer Morph { Delta d[]; } morph;
uniform int uBase;
in vec3 position;
out vec4 n;
flat out int vid;
void main() {
    Delta d = morph.d[uBase + gl_VertexID];
    n = d.normal;
    vid = gl_VertexID;
    gl_Position = vec4(position, 1.0) + d.position;
}`
	m, errs, ok := compile_glsl(vs, {stage = .Vertex})
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, len(m.stage_inputs), 1)
	for want in ([]string{"uint oksl_vertex_id [[vertex_id]]", "g.in_gl_VertexID = int(oksl_vertex_id);", "const device uint* oksl_buf_morph [[buffer(1)]]"}) {
		testing.expectf(t, strings.contains(m.msl, want), "MSL lacks %s", want)
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "test_vertex_id", m.spirv)
	}
	// A full batch and a partial one; the deltas end before the last
	// vertices, which read 0.
	N :: LANES + 3
	FIRST, BASE, DELTAS :: 5, 2, LANES + 4
	pos := make([][3]f32, N, context.temp_allocator)
	for &p, i in pos {
		p = {f32(i), 0, 0}
	}
	deltas := make([]u32, DELTAS * 8, context.temp_allocator)
	for i in 0 ..< DELTAS {
		for k in 0 ..< 8 {deltas[i * 8 + k] = transmute(u32)f32(1000 * i + k)}
	}
	block := make([]u8, m.block_size, context.temp_allocator)
	testing.expect(t, set_uniform(m, block, "uBase", {BASE}))
	positions := make([][4]f32, N, context.temp_allocator)
	record := m.varying_floats
	vary := make([]f32, N * record, context.temp_allocator)
	attrs := []Attribute{{raw_data(pos), size_of([3]f32), .Float3}}
	scratch: Scratch
	defer scratch_destroy(&scratch)
	run_vertex(&m.program, block, attrs, N, positions, vary, &scratch, {buffers = {deltas}, first_vertex = FIRST})
	offsets: map[string]int
	defer delete(offsets)
	for o in m.stage_outputs {
		offsets[o.name] = o.offset
	}
	for i in 0 ..< N {
		d := BASE + FIRST + i
		delta := [4]f32{f32(1000 * d), f32(1000 * d + 1), f32(1000 * d + 2), f32(1000 * d + 3)} if d < DELTAS else {}
		near(t, positions[i], [4]f32{f32(i), 0, 0, 1} + delta)
		r := vary[i * record:]
		testing.expect_value(t, transmute(i32)r[offsets["vid"]], i32(FIRST + i))
		normal := [4]f32{f32(1000 * d + 4), f32(1000 * d + 5), f32(1000 * d + 6), f32(1000 * d + 7)} if d < DELTAS else {}
		near(t, {r[offsets["n"]], r[offsets["n"] + 1], r[offsets["n"] + 2], r[offsets["n"] + 3]}, normal)
	}
	// Vertex shaders only.
	fm, ferrs, fok := compile_glsl("out vec4 c;\nvoid main() { c = vec4(float(gl_VertexID)); }")
	defer destroy(fm)
	testing.expect(t, !fok && strings.contains(format_errors(ferrs, context.temp_allocator), "gl_VertexID"))
}

// sampler2DArray: texture, textureLod and textureSize, the layer rounded
// to nearest even for the host's sampler; textureGrad picks the level
// texture() picks from the same derivatives.
@(test)
test_sampler_array :: proc(t: ^testing.T) {
	src := `#version 300 es
precision highp float;
in vec2 st;
uniform sampler2DArray layers;
uniform sampler2D flat2d;
uniform float uLayer;
layout(location = 0) out vec4 implicit;
layout(location = 1) out vec4 explicit;
layout(location = 2) out vec4 size;
layout(location = 3) out vec4 grad;
layout(location = 4) out vec4 grad2d;
vec4 pick(sampler2DArray s, vec3 c) { return textureLod(s, c, 1.0); }
void main() {
    implicit = texture(layers, vec3(st, uLayer));
    explicit = pick(layers, vec3(st, uLayer + 1.0));
    size = vec4(textureSize(layers, 1), 0.0);
    grad = textureGrad(layers, vec3(st, 0.0), dFdx(st), dFdy(st)) - texture(layers, vec3(st, 0.0));
    grad2d = textureGrad(flat2d, st, dFdx(st) * 4.0, dFdy(st)) - texture(flat2d, st, 2.0);
}`
	m, errs, ok := compile_glsl(src)
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	for want in ([]string{"texture2d_array<float> ok_layers_tex [[texture(0)]]", "oksl_sample_bias(ok_layers_tex, ok_layers_smp, float3(g.in_st, u.ok_uLayer)", "rint(z)", "ok_layers_tex.get_array_size()", "gradient2d("}) {
		testing.expectf(t, strings.contains(m.msl, want), "MSL lacks %s", want)
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "test_sampler_array", m.spirv)
	}
	sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		return {uv.x, lod, f32(layer), f32(channel)}
	}
	// st steps 1/16 across and 1/8 down a pixel: 4 x 4 texels of the
	// 64x32 array (lod 2), 8 x 8 of the 128x64 2D one.
	W, H :: 8, 2
	targets: [5][W * H][4]f32
	slices := [5][][4]f32{targets[0][:], targets[1][:], targets[2][:], targets[3][:], targets[4][:]}
	varyings := []Varying{{dx = {1.0 / 16, 0, 0, 0}, dy = {0, 1.0 / 8, 0, 0}}}
	sizes := [][3]i32{{64, 32, 6}, {128, 64, 1}}
	block := make([]u8, m.block_size, context.temp_allocator)
	// 2.5 rounds to 2 and 3.5 to 4 (nearest even).
	testing.expect(t, set_uniform(m, block, "uLayer", {2.5}))
	run_targets(&m.program, block, W, H, slices[:], temp_scratch(), {sampler = sampler, varyings = varyings, channel_sizes = sizes})
	for i in 0 ..< W * H {
		x := (f32(i % W) + 0.5) / 16
		near(t, targets[0][i], {x, 2, 2, 0})
		near(t, targets[1][i], {x, 1, 4, 0})
		near(t, targets[2][i], {32, 16, 6, 0})
		near(t, targets[3][i], {0, 0, 0, 0})
		near(t, targets[4][i], {0, 0, 0, 0})
	}

	// An infinite layer saturates and NaN is 0, for the host to clamp.
	for c in ([][2]f32{{math.inf_f32(1), f32(max(i32))}, {math.nan_f32(), 0}}) {
		testing.expect(t, set_uniform(m, block, "uLayer", {c[0]}))
		run_targets(&m.program, block, W, H, slices[:], temp_scratch(), {sampler = sampler, varyings = varyings, channel_sizes = sizes})
		testing.expect_value(t, targets[0][0].z, c[1])
	}

	// The header profiles' channels are 2D.
	sm, serrs, sok := compile_shadertoy("uniform sampler2DArray a;\nvoid mainImage(out vec4 o, in vec2 p) { o = vec4(1); }")
	defer destroy(sm)
	testing.expect(t, !sok && strings.contains(format_errors(serrs, context.temp_allocator), "sampler2DArray is GLSL's"))
	// texelFetch takes a sampler2D.
	fm, _, fok := compile_glsl("uniform sampler2DArray a;\nout vec4 c;\nvoid main() { c = texelFetch(a, ivec2(0), 0); }")
	defer destroy(fm)
	testing.expect(t, !fok)
}

// GLSL binds 16 samplers, the header profiles four channels; sampling
// calls the host only for the running lanes.
@(test)
test_samplers :: proc(t: ^testing.T) {
	b := strings.builder_make(context.temp_allocator)
	for i in 0 ..< SAMPLER_COUNT {
		fmt.sbprintf(&b, "uniform sampler2D s%d;\n", i)
	}
	strings.write_string(&b, "out vec4 c;\nvoid main() {\n    c = vec4(0.0);\n    if (gl_FragCoord.x > 2.0) { return; }\n")
	for i in 0 ..< SAMPLER_COUNT {
		fmt.sbprintf(&b, "    c += textureLod(s%d, vec2(0.5), 0.0);\n", i)
	}
	strings.write_string(&b, "}\n")
	m, errs, ok := compile_glsl(strings.to_string(b))
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, m.num_samplers, SAMPLER_COUNT)
	testing.expect_value(t, m.channel_names[SAMPLER_COUNT - 1], "s15")
	if have_spirv_val() {
		expect_valid_spirv(t, "test_samplers", m.spirv)
	}
	// 4x2: x > 2 returns early in half the lanes, which sample nothing.
	count := 0
	block := make([]u8, m.block_size, context.temp_allocator)
	out := make([][4]f32, 8, context.temp_allocator)
	run_f32(&m.program, block, 4, 2, out, temp_scratch(), {user = &count, sampler = proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		(^int)(user)^ += 1
		return {f32(channel), 0, 0, 0}
	}})
	testing.expect_value(t, count, 4 * SAMPLER_COUNT)
	near(t, out[0], {120, 0, 0, 0})
	near(t, out[3], {0, 0, 0, 0})

	// A discarded lane still samples for its quad (as a GPU's helper), so
	// derivatives of the result stay the quad's.
	dm, derrs, dok := compile_glsl("uniform sampler2D tex;\nin vec2 uv;\nout vec4 c;\nvoid main() {\n    if (int(gl_FragCoord.x) % 2 == 1) { discard; }\n    vec4 s = textureLod(tex, uv, 0.0);\n    c = vec4(dFdx(s.x), dFdy(s.y), 0.0, 1.0);\n}")
	defer destroy(dm)
	if testing.expectf(t, dok, "%s", format_errors(derrs, context.temp_allocator)) {
		dblock := make([]u8, max(dm.block_size, 16), context.temp_allocator)
		dout := make([][4]f32, 8, context.temp_allocator)
		run_f32(&dm.program, dblock, 4, 2, dout, temp_scratch(), {sampler = proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			return {uv.x, uv.y, 0, 1}
		}})
		for x in ([]int{0, 2}) {
			near(t, dout[x], {0.25, 0.5, 0, 1})
		}
	}

	strings.write_string(&b, "uniform sampler2D s16;\n")
	em, eerrs, eok := compile_glsl(strings.to_string(b))
	defer destroy(em)
	testing.expect(t, !eok && strings.contains(format_errors(eerrs, context.temp_allocator), "at most 16 sampler2D uniforms"))
	five := "uniform shader a;\nuniform shader b;\nuniform shader c;\nuniform shader d;\nuniform shader e;\nhalf4 main(float2 p) { return a.eval(p); }"
	testing.expect(t, strings.contains(errors_of(five), "at most 4 shader uniforms"))
}

// Instructions of a SPIR-V module whose opcode is `op`.
@(private = "file")
spirv_count :: proc(words: []u32, op: u32) -> int {
	n := 0
	for i := 5; i < len(words); i += int(words[i] >> 16) {
		if words[i] & 0xffff == op {
			n += 1
		}
		if words[i] >> 16 == 0 {
			break
		}
	}
	return n
}

// What o3d's software rasterizer hands run_fragment and run_vertex beyond
// colours and floats: gl_SampleMask[0] out per lane, gl_FragCoord.z in,
// packed vertex formats; and fine derivatives in SPIR-V, as the
// interpreter computes them.
@(test)
test_fragment_extras :: proc(t: ^testing.T) {
	src := `#version 300 es
precision highp float;
layout(location = 0) out vec4 c;
layout(location = 2) out ivec2 cell;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    gl_SampleMask[0] = (p.x + 3 * p.y) & 15;
    c = vec4(gl_FragCoord.z, dFdx(gl_FragCoord.z), fwidth(gl_FragCoord.x), 1.0);
    cell = p - 1;
}`
	m, errs, ok := compile_glsl(src)
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	for want in ([]string{"uint out_gl_SampleMask [[sample_mask]];", "oksl_out.out_gl_SampleMask = uint(g.out_gl_SampleMask[0]);"}) {
		testing.expectf(t, strings.contains(m.msl, want), "MSL lacks %s", want)
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "test_fragment_extras", m.spirv)
	}
	// OpDPdxFine and OpFwidthFine, not OpDPdx / OpFwidth.
	testing.expect_value(t, spirv_count(m.spirv, 210), 1)
	testing.expect_value(t, spirv_count(m.spirv, 212), 1)
	testing.expect_value(t, spirv_count(m.spirv, 207) + spirv_count(m.spirv, 209), 0)

	Q :: LANES / 4 + 1
	origin := make([][2]i32, Q, context.temp_allocator)
	depth := make([]f32, Q * 4, context.temp_allocator)
	for q in 0 ..< Q {
		origin[q] = {i32(q * 2), 6}
		for i in 0 ..< 4 {
			depth[q * 4 + i] = 0.25 + 0.01 * f32(q * 2 + (i & 1))
		}
	}
	// Every location: 0, nothing at 1, 2 (an ivec2, as floats).
	out := make([][4]f32, Q * 4, context.temp_allocator)
	none := make([][4]f32, Q * 4, context.temp_allocator)
	cells := make([][4]f32, Q * 4, context.temp_allocator)
	targets := [3][][4]f32{out, none, cells}
	shaded := make([]Quad_Lanes, Q, context.temp_allocator)
	masks := make([]u32, Q * 4, context.temp_allocator)
	scratch: Scratch
	defer scratch_destroy(&scratch)
	block := make([]u8, max(m.block_size, 16), context.temp_allocator)
	run_fragment(&m.program, block, {count = Q, origin = origin, depth = depth}, targets[:], shaded, &scratch, {}, masks)
	for q in 0 ..< Q {
		for i in 0 ..< 4 {
			x, y := q * 2 + (i & 1), 6 + (i >> 1)
			testing.expect_value(t, masks[q * 4 + i], u32((x + 3 * y) & 15))
			near(t, out[q * 4 + i], {depth[q * 4 + i], 0.01, 1, 1})
			near(t, none[q * 4 + i], {0, 0, 0, 1})
			near(t, cells[q * 4 + i], {f32(x - 1), f32(y - 1), 0, 1})
		}
	}
	// Not written: every sample.
	plain, _, pok := compile_glsl("out vec4 c;\nvoid main() { c = vec4(1.0); }")
	defer destroy(plain)
	testing.expect(t, pok)
	run_fragment(&plain.program, block, {count = Q, origin = origin}, targets[:1], shaded, &scratch, {}, masks)
	testing.expect_value(t, masks[5], ~u32(0))
	// Fragment shaders only.
	vm, verrs, vok := compile_glsl("void main() { gl_SampleMask[0] = 1; gl_Position = vec4(0.0); }", {stage = .Vertex})
	defer destroy(vm)
	testing.expect(t, !vok && strings.contains(format_errors(verrs, context.temp_allocator), "gl_SampleMask"))

	// Packed attributes: colours as unorm bytes, joints as u16s.
	vs := `#version 300 es
in vec4 color;
in uvec4 joints;
out vec4 vColor;
flat out uvec4 vJoints;
void main() { vColor = color; vJoints = joints + 1u; gl_Position = vec4(0.0); }`
	pm, perrs, pok2 := compile_glsl(vs, {stage = .Vertex})
	defer destroy(pm)
	if !testing.expectf(t, pok2, "%s", format_errors(perrs, context.temp_allocator)) {
		return
	}
	Packed :: struct {
		color:  [4]u8,
		joints: [4]u16,
	}
	N :: 3
	verts := [N]Packed{{{0, 51, 255, 128}, {0, 1, 2, 65535}}, {{255, 0, 0, 0}, {7, 8, 9, 10}}, {{1, 2, 3, 4}, {300, 0, 0, 0}}}
	attrs: [2]Attribute
	for a in pm.stage_inputs {
		if a.name == "color" {attrs[a.location] = {&verts[0].color, size_of(Packed), .Ubyte4_Norm}}
		if a.name == "joints" {attrs[a.location] = {&verts[0].joints, size_of(Packed), .Ushort4}}
	}
	positions: [N][4]f32
	record := pm.varying_floats
	vary := make([]f32, N * record, context.temp_allocator)
	vblock := make([]u8, max(pm.block_size, 16), context.temp_allocator)
	run_vertex(&pm.program, vblock, attrs[:], N, positions[:], vary, &scratch)
	offsets: map[string]int
	defer delete(offsets)
	for o in pm.stage_outputs {
		offsets[o.name] = o.offset
	}
	for v, i in verts {
		r := vary[i * record:]
		c := offsets["vColor"]
		testing.expect_value(t, [4]f32{r[c], r[c + 1], r[c + 2], r[c + 3]}, [4]f32{f32(v.color[0]) / 255, f32(v.color[1]) / 255, f32(v.color[2]) / 255, f32(v.color[3]) / 255})
		j := offsets["vJoints"]
		got := transmute([4]u32)[4]f32{r[j], r[j + 1], r[j + 2], r[j + 3]}
		testing.expect_value(t, got, [4]u32{u32(v.joints[0]) + 1, u32(v.joints[1]) + 1, u32(v.joints[2]) + 1, u32(v.joints[3]) + 1})
	}
}

@(test)
test_if_and_ternary_cond_mutation :: proc(t: ^testing.T) {
	defer free_all(context.temp_allocator)
	src := `
uniform float u0;
out vec4 fragColor;

void main() {
    bool b = gl_FragCoord.x < 8.0;
    float r = 0.0;
    if (b) {
        b = false;
        r += 1.0;
    } else {
        r += 10.0;
    }
    bool c = gl_FragCoord.y < 4.0;
    float q = c ? ((c = false) ? 100.0 : 2.0) : 20.0;
    fragColor = vec4(r, q, b ? 1.0 : 0.0, c ? 1.0 : 0.0) + u0 * 0.0;
}
`
	m, errs, ok := compile_glsl(src, {no_spirv = true, no_msl = true})
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	block := make([]u8, max(m.block_size, 16), context.temp_allocator)
	out := make([][4]f32, 16 * 8, context.temp_allocator)
	run_f32(&m.program, block, 16, 8, out, temp_scratch())
	for y in 0 ..< 8 {
		for x in 0 ..< 16 {
			got := out[y * 16 + x]
			want_r := f32(1) if x < 8 else f32(10)
			want_q := f32(2) if y < 4 else f32(20)
			near(t, got, {want_r, want_q, 0, 0})
		}
	}
}

@(test)
test_nan_and_jit_parity :: proc(t: ^testing.T) {
	defer free_all(context.temp_allocator)
	src := `
layout(std430) readonly buffer Buf { uint words[]; } buf;
uniform float uZero;
uniform float uVal;
uniform uint uBig0;
uniform uint uBig1;
out vec4 fragColor;

void main() {
    float nan_v = uZero / uZero;
    float pos_inf = 1.0 / uZero;
    float neg_inf = -1.0 / uZero;

    float c_nan = clamp(nan_v, 0.25, 0.75);
    float ss_nan = smoothstep(0.0, 1.0, nan_v);
    float mn_nan_first = min(nan_v, uVal);
    float mn_nan_second = min(uVal, nan_v);
    float mx_nan_first = max(nan_v, uVal);
    float mx_nan_second = max(uVal, nan_v);

    int i_nan = int(nan_v);
    int i_hi = int(pos_inf);
    int i_lo = int(neg_inf);
    uint u_nan = uint(nan_v);
    uint u_neg = uint(-3.0 + uZero);
    uint u_hi = uint(pos_inf);
    float uf0 = float(uBig0);
    float uf1 = float(uBig1);

    uint lane = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4u;
    uint bw = buf.words[lane];
    float arr[4] = float[4](1.0, 2.0, 3.0, 4.0);
    float dyn_v = arr[int(lane & 3u)];

    float flags = float(
        (isnan(mn_nan_first) ? 1 : 0) +
        (isnan(mx_nan_first) ? 2 : 0) +
        (i_nan == 0 ? 4 : 0) +
        (i_hi == 2147483647 ? 8 : 0) +
        (i_lo == (-2147483647 - 1) ? 16 : 0) +
        (u_nan == 0u ? 32 : 0) +
        (u_neg == 0u ? 64 : 0) +
        (u_hi == 4294967295u ? 128 : 0)
    );
    fragColor = vec4(
        c_nan + ss_nan,
        mn_nan_second + mx_nan_second,
        flags + float(bw) + dyn_v,
        uf0 - uf1
    );
}
`
	m, errs, ok := compile_glsl(src, {no_spirv = true, no_msl = true})
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	block := make([]u8, max(m.block_size, 64), context.temp_allocator)
	set_uniform(m, block, "uZero", {0.0})
	set_uniform(m, block, "uVal", {1.5})
	if u0, found := find_uniform(m, "uBig0"); found {
		(^u32)(&block[u0.offset])^ = 0xffff_ffff
	}
	if u1, found := find_uniform(m, "uBig1"); found {
		(^u32)(&block[u1.offset])^ = 0x8000_0001
	}

	words := [6]u32{10, 20, 30, 40, 50, 60}
	bufs := [1][]u32{words[:]}
	opts := Run_Options{buffers = bufs[:]}

	out_jit := make([][4]f32, 6 * 3, context.temp_allocator)
	out_vm := make([][4]f32, 6 * 3, context.temp_allocator)
	run_f32(&m.program, block, 6, 3, out_jit, temp_scratch(), opts)

	p_vm := m.program
	p_vm.jit_body = nil
	p_vm.jit_init = nil
	run_f32(&p_vm, block, 6, 3, out_vm, temp_scratch(), opts)

	for i in 0 ..< len(out_jit) {
		x, y := i % 6, i / 6
		lane := u32(x + y * 4)
		want_bw := f32(words[lane]) if int(lane) < len(words) else 0
		want_dyn := f32((lane & 3) + 1)
		near(t, out_jit[i], {0.25, 3.0, 255.0 + want_bw + want_dyn, f32(u32(0xffff_ffff)) - f32(u32(0x8000_0001))})
		testing.expect_value(t, transmute([4]u32)out_jit[i], transmute([4]u32)out_vm[i])
	}

	// Clamp with NaN bounds and sampler host context preservation.
	{
		tex_src := `
uniform float uz;
uniform sampler2D tex;
out vec4 fragColor;
void main() {
	float n = uz / uz;
	float x = gl_FragCoord.x - 2.0;
	vec4 s = texture(tex, gl_FragCoord.xy * 0.25);
	fragColor = vec4(clamp(x, n, 1.0), clamp(x, -1.0, n), clamp(n, n, 2.0), s.x);
}
`
		mt, terrs, tok := compile_glsl(tex_src, {no_spirv = true, no_msl = true})
		defer destroy(mt)
		if !testing.expectf(t, tok, "%s", format_errors(terrs, context.temp_allocator)) {
			return
		}
		tblock := make([]u8, max(mt.block_size, 16), context.temp_allocator)
		set_uniform(mt, tblock, "uz", {0.0})

		Samp_State :: struct {
			calls: int,
			bad:   int,
		}
		sampler_cb :: proc(user: rawptr, ch: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			st := (^Samp_State)(user)
			st.calls += 1
			if context.user_index != 0x5eed_cafe {
				st.bad += 1
			}
			return {f32(context.user_index & 0xff), 0, 0, 1}
		}

		saved_idx := context.user_index
		context.user_index = 0x5eed_cafe
		defer context.user_index = saved_idx

		st_jit, st_vm: Samp_State
		tout_jit := make([][4]f32, 4 * 2, context.temp_allocator)
		tout_vm := make([][4]f32, 4 * 2, context.temp_allocator)
		run_f32(&mt.program, tblock, 4, 2, tout_jit, temp_scratch(), {sampler = sampler_cb, user = &st_jit})
		pm := mt.program
		pm.jit_body = nil
		pm.jit_init = nil
		run_f32(&pm, tblock, 4, 2, tout_vm, temp_scratch(), {sampler = sampler_cb, user = &st_vm})

		testing.expect(t, st_jit.calls == 8 && st_jit.bad == 0)
		testing.expect(t, st_vm.calls == 8 && st_vm.bad == 0)
		for i in 0 ..< len(tout_jit) {
			testing.expect_value(t, transmute([4]u32)tout_jit[i], transmute([4]u32)tout_vm[i])
		}
	}
}

// ---- compute shaders ----------------------------------------------------------------

@(private = "file")
compile_compute :: proc(src: string) -> (^Module, []Error, bool) {
	return compile_glsl(src, {profile = .Glsl, stage = .Compute})
}

// Bit-exact RGBA8 -> NV12 Q16/Q18 reference matching oicut/video/color.odin.
@(test)
test_compute_nv12 :: proc(t: ^testing.T) {
	src_bytes, rerr := os.read_entire_file("testdata/nv12.comp", context.temp_allocator)
	if !testing.expectf(t, rerr == nil, "read nv12.comp: %v", rerr) {
		return
	}
	m, errs, ok := compile_compute(string(src_bytes))
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	testing.expect_value(t, m.threadcount, [3]u32{8, 8, 1})
	testing.expect_value(t, m.num_samplers, 0)
	testing.expect_value(t, m.num_readonly_storage_textures, 1)
	testing.expect_value(t, m.num_readonly_storage_buffers, 0)
	testing.expect_value(t, m.num_readwrite_storage_textures, 0)
	testing.expect_value(t, m.num_readwrite_storage_buffers, 1)
	testing.expect_value(t, m.num_uniform_buffers, 1)
	testing.expect(t, m.program.jit_body == nil && m.program.jit_init == nil)

	// Non-multiple-of-8 frame (34x18) with padded row stride on both the
	// source RGBA8 image (148 bytes/row) and NV12 output (40 bytes/row).
	W, H :: 34, 18
	SRC_STRIDE :: W * 4 + 12
	Y_STRIDE :: 40
	CHROMA_OFFSET :: Y_STRIDE * H
	OUT_WORDS :: (CHROMA_OFFSET + Y_STRIDE * (H / 2)) / 4

	rgba := make([]u8, SRC_STRIDE * H, context.temp_allocator)
	for &b in rgba {
		b = 0xdd
	}
	corners := [][3]u8 {
		{0, 0, 0},
		{255, 255, 255},
		{255, 0, 0},
		{0, 255, 0},
		{0, 0, 255},
		{255, 255, 0},
		{0, 255, 255},
		{255, 0, 255},
	}
	for y in 0 ..< H {
		row := rgba[y * SRC_STRIDE:]
		for x in 0 ..< W {
			idx := y * W + x
			c: [3]u8
			if idx < len(corners) {
				c = corners[idx]
			} else {
				c = {u8((x * 37 + y * 13 + 7) & 0xff), u8((x * 19 + y * 53 + 101) & 0xff), u8((x * 71 + y * 29 + 211) & 0xff)}
			}
			row[x * 4 + 0] = c[0]
			row[x * 4 + 1] = c[1]
			row[x * 4 + 2] = c[2]
			row[x * 4 + 3] = u8((x + y) & 0xff)
		}
	}

	q16 :: proc(v: f64) -> i32 {
		return i32(math.round(v * 65536.0))
	}

	Config :: struct {
		kr, kb:     f64,
		full_range: bool,
	}
	configs := []Config {
		{0.2126, 0.0722, false}, // BT.709 limited
		{0.2126, 0.0722, true},  // BT.709 full
		{0.2990, 0.1140, false}, // BT.601 limited
		{0.2990, 0.1140, true},  // BT.601 full
	}

	block := make([]u8, m.block_size, context.temp_allocator)
	got_words := make([]u32, OUT_WORDS, context.temp_allocator)
	want_bytes := make([]u8, OUT_WORDS * 4, context.temp_allocator)

	for cfg in configs {
		ys := (255.0 if cfg.full_range else 219.0) / 255.0
		cs := (255.0 if cfg.full_range else 224.0) / 255.0
		yr := q16(cfg.kr * ys)
		yb := q16(cfg.kb * ys)
		yg := q16(ys) - yr - yb
		y_bias := ((i32(0) if cfg.full_range else i32(16 << 16)) + (1 << 15))

		cb := cs / (2.0 * (1.0 - cfg.kb))
		cr := cs / (2.0 * (1.0 - cfg.kr))
		u0 := q16(-cfg.kr * cb)
		u2 := q16((1.0 - cfg.kb) * cb)
		u1 := -u0 - u2
		v0 := q16((1.0 - cfg.kr) * cr)
		v2 := q16(-cfg.kb * cr)
		v1 := -v0 - v2
		c_bias := i32((128 << 18) + (1 << 17))
		c_max := i32(255)

		ky := [4]i32{yr, yg, yb, y_bias}
		ku := [4]i32{u0, u1, u2, c_bias}
		kv := [4]i32{v0, v1, v2, c_max}
		sz := [4]u32{W, H, Y_STRIDE, CHROMA_OFFSET}
		copy(block[0:16], slice.to_bytes(ky[:]))
		copy(block[16:32], slice.to_bytes(ku[:]))
		copy(block[32:48], slice.to_bytes(kv[:]))
		copy(block[48:64], slice.to_bytes(sz[:]))

		for &w in got_words {
			w = 0xa5a5_5a5a
		}
		want_words := slice.reinterpret([]u32, want_bytes)
		for &w in want_words {
			w = 0xa5a5_5a5a
		}

		// Reference: nv12.comp processes 8x2 pixel blocks across y_stride.
		fetch_rgb :: proc(rgba: []u8, x, y: int) -> [3]i32 {
			cx := min(x, W - 1)
			cy := min(y, H - 1)
			off := cy * SRC_STRIDE + cx * 4
			return {i32(rgba[off + 0]), i32(rgba[off + 1]), i32(rgba[off + 2])}
		}
		luma :: proc(c: [3]i32, ky: [4]i32) -> u8 {
			return u8(clamp((ky[0] * c[0] + ky[1] * c[1] + ky[2] * c[2] + ky[3]) >> 16, 0, 255))
		}
		for by := 0; by < H; by += 2 {
			for bx := 0; bx < Y_STRIDE; bx += 2 {
				a := fetch_rgb(rgba, bx, by)
				b := fetch_rgb(rgba, bx + 1, by)
				c := fetch_rgb(rgba, bx, by + 1)
				d := fetch_rgb(rgba, bx + 1, by + 1)
				want_bytes[by * Y_STRIDE + bx + 0] = luma(a, ky)
				want_bytes[by * Y_STRIDE + bx + 1] = luma(b, ky)
				want_bytes[(by + 1) * Y_STRIDE + bx + 0] = luma(c, ky)
				want_bytes[(by + 1) * Y_STRIDE + bx + 1] = luma(d, ky)
				s := a + b + c + d
				uv_off := CHROMA_OFFSET + (by / 2) * Y_STRIDE + bx
				want_bytes[uv_off + 0] = u8(clamp((ku[0] * s[0] + ku[1] * s[1] + ku[2] * s[2] + ku[3]) >> 18, 0, kv[3]))
				want_bytes[uv_off + 1] = u8(clamp((kv[0] * s[0] + kv[1] * s[1] + kv[2] * s[2] + ku[3]) >> 18, 0, kv[3]))
			}
		}

		groups := [3]int{((Y_STRIDE / 8) + 7) / 8, ((H / 2) + 7) / 8, 1}
		img := Storage_Image{data = rgba, width = W, height = H, stride = SRC_STRIDE, format = .Rgba8}
		run_compute(&m.program, block, groups, temp_scratch(), {images = {img}, buffers = {got_words}})

		mismatches := 0
		for i in 0 ..< OUT_WORDS {
			if got_words[i] != want_words[i] {
				mismatches += 1
			}
		}
		testing.expectf(t, mismatches == 0, "nv12 mismatch count %d (kr=%v full=%v)", mismatches, cfg.kr, cfg.full_range)
	}
}

// Disjoint workgroup ranges produce the exact same state whether run in one
// call, uneven slices, or across 8 concurrent threads with independent Scratch.
@(test)
test_compute_workgroup_ranges_and_threads :: proc(t: ^testing.T) {
	src_bytes, rerr := os.read_entire_file("testdata/particles.comp", context.temp_allocator)
	if !testing.expectf(t, rerr == nil, "read particles.comp: %v", rerr) {
		return
	}
	m, errs, ok := compile_compute(string(src_bytes))
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}

	GROUPS :: [3]int{5, 3, 2} // 30 workgroups * 32 lanes = 960 invocations
	N_PARTICLES :: 917
	STATE_WORDS :: N_PARTICLES * 8

	forces := make([]u32, 16 * 4, context.temp_allocator)
	for i in 0 ..< len(forces) {
		forces[i] = transmute(u32)(f32((i % 11) - 5) * 0.375)
	}

	init_state :: proc(buf: []u32) {
		for p in 0 ..< N_PARTICLES {
			base := p * 8
			buf[base + 0] = transmute(u32)(f32(p) * 0.25)
			buf[base + 1] = transmute(u32)(f32(p % 17) * -0.5)
			buf[base + 2] = transmute(u32)(f32(p % 9) * 1.125)
			buf[base + 3] = transmute(u32)(0.5 + f32(p % 5) * 0.5)
			buf[base + 4] = transmute(u32)(f32((p % 7) - 3) * 0.125)
			buf[base + 5] = transmute(u32)(f32((p % 5) - 2) * 0.25)
			buf[base + 6] = transmute(u32)(f32((p % 3) - 1) * 0.5)
			buf[base + 7] = u32(p * 3)
		}
	}

	block := make([]u8, m.block_size, context.temp_allocator)
	testing.expect(t, set_uniform(m, block, "uDt", {0.0625}))
	testing.expect(t, set_uniform(m, block, "uCount", {f32(N_PARTICLES)}))

	state_single := make([]u32, STATE_WORDS, context.temp_allocator)
	state_sliced := make([]u32, STATE_WORDS, context.temp_allocator)
	state_mt := make([]u32, STATE_WORDS, context.temp_allocator)
	init_state(state_single)
	init_state(state_sliced)
	init_state(state_mt)

	// 1. Full dispatch in one call.
	run_compute(&m.program, block, GROUPS, temp_scratch(), {buffers = {forces, state_single}})

	// 2. Uneven range splits + zero/clamped/out-of-range slices.
	opts_sliced := Run_Options{buffers = {forces, state_sliced}}
	run_compute(&m.program, block, GROUPS, temp_scratch(), opts_sliced, 0, 0)
	run_compute(&m.program, block, GROUPS, temp_scratch(), opts_sliced, -4, 0)
	run_compute(&m.program, block, GROUPS, temp_scratch(), opts_sliced, 0, 1)
	run_compute(&m.program, block, GROUPS, temp_scratch(), opts_sliced, 1, 7)
	run_compute(&m.program, block, GROUPS, temp_scratch(), opts_sliced, 8, 13)
	run_compute(&m.program, block, GROUPS, temp_scratch(), opts_sliced, 21, 99)
	run_compute(&m.program, block, GROUPS, temp_scratch(), opts_sliced, 30, 5)

	// 3. 8 OS threads concurrently running disjoint workgroup slices.
	Worker :: struct {
		prog:   ^Program,
		block:  []u8,
		forces: []u32,
		state:  []u32,
		first:  int,
		count:  int,
	}
	WORKERS :: 8
	workers: [WORKERS]Worker
	threads: [WORKERS]^thread.Thread
	for w in 0 ..< WORKERS {
		workers[w] = {
			prog   = &m.program,
			block  = block,
			forces = forces,
			state  = state_mt,
			first  = w * 4,
			count  = 4,
		}
		threads[w] = thread.create_and_start_with_poly_data(&workers[w], proc(wk: ^Worker) {
			sc: Scratch
			defer scratch_destroy(&sc)
			run_compute(wk.prog, wk.block, GROUPS, &sc, {buffers = {wk.forces, wk.state}}, wk.first, wk.count)
		})
	}
	for th in threads {
		thread.join(th)
		thread.destroy(th)
	}

	for i in 0 ..< STATE_WORDS {
		if state_sliced[i] != state_single[i] || state_mt[i] != state_single[i] {
			testing.expectf(t, false, "word %d mismatch: single=%08x sliced=%08x mt=%08x", i, state_single[i], state_sliced[i], state_mt[i])
			break
		}
	}
}

// All 6 storage image formats, IEEE-754 RNE f16 conversion, row stride padding,
// single-channel .gba = (0, 0, 1) substitution, and OOB load/store safety.
@(test)
test_compute_image_formats_and_bounds :: proc(t: ^testing.T) {
	// 1. rgba16f RNE rounding, subnormals, overflow, -0.0, NaN, and roundtrip load.
	src_f16 := `#version 310 es
layout(local_size_x = 16) in;
layout(rgba16f) writeonly uniform highp image2D dst16;
layout(rgba16f) readonly uniform highp image2D src16;
buffer IO { vec4 vals[]; } io;
uniform int uMode;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= 10u) return;
    if (uMode == 0) {
        imageStore(dst16, ivec2(int(i), 0), io.vals[i]);
    } else {
        io.vals[i] = imageLoad(src16, ivec2(int(i), 0));
    }
}`
	mf16, ef16, ok16 := compile_compute(src_f16)
	defer destroy(mf16)
	if !testing.expectf(t, ok16, "%s", format_errors(ef16, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "f16", mf16.spirv)
	}
	tie_even := math.ldexp_f32(1.0, 0) + math.ldexp_f32(1.0, -11)       // 1.0 + 0.5 ulp -> rounds to even (0x3c00)
	tie_odd := math.ldexp_f32(1.0, 0) + math.ldexp_f32(3.0, -11)        // 1.0 + 1.5 ulp -> rounds to even (0x3c02)
	min_sub := math.ldexp_f32(1.0, -24)                                 // 0x0001
	sub_tie_zero := math.ldexp_f32(1.0, -25)                            // 0.5 * min_sub -> 0x0000 (even)
	sub_tie_up := math.ldexp_f32(3.0, -25)                              // 1.5 * min_sub -> 0x0002 (even)
	neg_zero := transmute(f32)u32(0x8000_0000)
	qnan := transmute(f32)u32(0x7fc0_1234)

	vals := make([][4]f32, 10, context.temp_allocator)
	vals[0] = {1.0, -2.5, 0.0, neg_zero}
	vals[1] = {tie_even, tie_odd, min_sub, sub_tie_zero}
	vals[2] = {sub_tie_up, 65504.0, 65519.996, 65520.0} // 65519.996 < 65504 + 16 -> 65504 (even); 65520.0 == tie -> +Inf (even 0x7c00)
	vals[3] = {math.inf_f32(1), math.inf_f32(-1), qnan, -65504.0}
	for i in 4 ..< 10 {
		vals[i] = {f32(i), -f32(i), f32(i) * 0.5, 1.0}
	}

	f16_words := make([][4]u16, 10, context.temp_allocator)
	f16_img := Storage_Image{data = slice.to_bytes(f16_words), width = 10, height = 1, format = .Rgba16f}
	vals_u32 := slice.reinterpret([]u32, vals)
	b16 := make([]u8, mf16.block_size, context.temp_allocator)
	testing.expect(t, set_uniform(mf16, b16, "uMode", {0}))
	run_compute(&mf16.program, b16, {1, 1, 1}, temp_scratch(), {images = {f16_img, f16_img}, buffers = {vals_u32}})

	testing.expect_value(t, f16_words[0], [4]u16{0x3c00, 0xc100, 0x0000, 0x8000})
	testing.expect_value(t, f16_words[1], [4]u16{0x3c00, 0x3c02, 0x0001, 0x0000})
	testing.expect_value(t, f16_words[2], [4]u16{0x0002, 0x7bff, 0x7bff, 0x7c00})
	testing.expect_value(t, f16_words[3][0], u16(0x7c00))
	testing.expect_value(t, f16_words[3][1], u16(0xfc00))
	testing.expect(t, (f16_words[3][2] & 0x7c00) == 0x7c00 && (f16_words[3][2] & 0x03ff) != 0)
	testing.expect_value(t, f16_words[3][3], u16(0xfbff))

	// Load back and check IEEE values (+ -0.0 sign bit, Inf, NaN, subnormal).
	for &v in vals {
		v = {-99, -99, -99, -99}
	}
	testing.expect(t, set_uniform(mf16, b16, "uMode", {1}))
	run_compute(&mf16.program, b16, {1, 1, 1}, temp_scratch(), {images = {f16_img, f16_img}, buffers = {vals_u32}})
	testing.expect_value(t, vals[0][0], f32(1.0))
	testing.expect_value(t, vals[0][1], f32(-2.5))
	testing.expect_value(t, transmute(u32)vals[0][3], u32(0x8000_0000))
	testing.expect_value(t, vals[1][2], min_sub)
	testing.expect_value(t, vals[2][1], f32(65504.0))
	testing.expect(t, math.is_inf(vals[2][3], 1))
	testing.expect(t, math.is_inf(vals[3][0], 1) && math.is_inf(vals[3][1], -1) && math.is_nan(vals[3][2]))

	// 2. Single-channel RW formats (r32f, r32ui, r32i), rgba8 stride/clamp/NaN,
	//    rgba32f exact, imageSize, and OOB load/store behaviour.
	src_all := `#version 310 es
layout(local_size_x = 8) in;
layout(rgba8) writeonly uniform highp image2D out8;
layout(rgba32f) writeonly uniform highp image2D out32;
layout(r32f) uniform highp image2D rwf;
layout(r32ui) uniform highp uimage2D rwu;
layout(r32i) uniform highp iimage2D rwi;
buffer Probe { uvec4 info[]; } probe;
void main() {
    int i = int(gl_LocalInvocationIndex);
    vec4 lf = imageLoad(rwf, ivec2(i, 0));     // in-bounds for i < 4, OOB for i >= 4
    uvec4 lu = imageLoad(rwu, ivec2(i, 0));
    ivec4 li = imageLoad(rwi, ivec2(i, 0));
    ivec2 sz = imageSize(out8);
    probe.info[i] = uvec4(
        floatBitsToUint(lf.w) ^ (uint(lf.y == 0.0 && lf.z == 0.0) << 31u),
        lu.x + lu.w * 100u + lu.y + lu.z,
        uint(li.x + li.w * 1000 + li.y + li.z),
        uint(sz.x * 100 + sz.y)
    );
    imageStore(rwf, ivec2(i, 0), vec4(lf.x + 2.5, 9.0, 9.0, 9.0));
    imageStore(rwu, ivec2(i, 0), uvec4(lu.x + 7u, 9u, 9u, 9u));
    imageStore(rwi, ivec2(i, 0), ivec4(li.x - 5, 9, 9, 9));
    float nanv = uintBitsToFloat(0x7fc00000u);
    imageStore(out8, ivec2(i - 1, i & 1), vec4(-0.5, 1.5, 100.0 / 255.0, nanv));
    imageStore(out32, ivec2(i, 0), vec4(float(i), -float(i), 0.25, 0.75));
    // OOB buffer store is masked off cleanly.
    if (i == 7) {
        probe.info[99] = uvec4(1u);
    }
}`
	mall, eall, okall := compile_compute(src_all)
	defer destroy(mall)
	if !testing.expectf(t, okall, "%s", format_errors(eall, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "all_formats", mall.spirv)
	}

	// out8 is 3x2 with stride = 20 bytes (8 padding bytes per row filled with 0xaa).
	out8_bytes := make([]u8, 20 * 2, context.temp_allocator)
	for &b in out8_bytes {
		b = 0xaa
	}
	out32_px := make([][4]f32, 4, context.temp_allocator)
	rwf_px := [4]f32{1.25, -3.0, 0.0, 10.5}
	rwu_px := [4]u32{10, 20, 30, 40}
	rwi_px := [4]i32{-10, 0, 15, -99}
	probe_words := make([][4]u32, 8, context.temp_allocator)

	// Binding order for RW/WO storage images: declaration order since none are readonly!
	testing.expect_value(t, mall.num_readonly_storage_textures, 0)
	testing.expect_value(t, mall.num_readwrite_storage_textures, 5)
	imgs := []Storage_Image {
		{data = out8_bytes, width = 3, height = 2, stride = 20, format = .Rgba8},
		{data = slice.to_bytes(out32_px), width = 4, height = 1, format = .Rgba32f},
		{data = slice.to_bytes(rwf_px[:]), width = 4, height = 1, format = .R32f},
		{data = slice.to_bytes(rwu_px[:]), width = 4, height = 1, format = .R32ui},
		{data = slice.to_bytes(rwi_px[:]), width = 4, height = 1, format = .R32i},
	}
	run_compute(&mall.program, nil, {1, 1, 1}, temp_scratch(), {images = imgs, buffers = {slice.reinterpret([]u32, probe_words)}})

	for i in 0 ..< 8 {
		// In-bounds single-channel load sets .w = 1 (1.0f / 1u / 1), OOB returns (0, 0, 0, 0).
		want_w_bits := (transmute(u32)f32(1.0) if i < 4 else u32(0)) | (1 << 31)
		testing.expect_value(t, probe_words[i][0], want_w_bits)
		want_u := (u32(10 * (i + 1)) + 100) if i < 4 else 0
		testing.expect_value(t, probe_words[i][1], want_u)
		init_i := [4]i32{-10, 0, 15, -99}
		want_i := u32(init_i[i] + 1000) if i < 4 else 0
		testing.expect_value(t, probe_words[i][2], want_i)
		testing.expect_value(t, probe_words[i][3], u32(302))
	}
	testing.expect_value(t, rwf_px, [4]f32{3.75, -0.5, 2.5, 13.0})
	testing.expect_value(t, rwu_px, [4]u32{17, 27, 37, 47})
	testing.expect_value(t, rwi_px, [4]i32{-15, -5, 10, -104})
	for i in 0 ..< 4 {
		testing.expect_value(t, out32_px[i], [4]f32{f32(i), -f32(i), 0.25, 0.75})
	}
	// Check rgba8 written texels (i=1 -> (0,1), i=2 -> (1,0), i=3 -> (2,1); i=0 -> (-1,0) OOB; i>=4 -> x>=3 OOB)
	// and verify row padding bytes [12..20) remain untouched (0xaa).
	testing.expect_value(t, [4]u8{out8_bytes[4], out8_bytes[5], out8_bytes[6], out8_bytes[7]}, [4]u8{0, 255, 100, 0})
	testing.expect_value(t, [4]u8{out8_bytes[20], out8_bytes[21], out8_bytes[22], out8_bytes[23]}, [4]u8{0, 255, 100, 0})
	testing.expect_value(t, [4]u8{out8_bytes[28], out8_bytes[29], out8_bytes[30], out8_bytes[31]}, [4]u8{0, 255, 100, 0})
	for row in 0 ..< 2 {
		for b in 12 ..< 20 {
			testing.expect_value(t, out8_bytes[row * 20 + b], u8(0xaa))
		}
	}

	// Valid 0x0 empty Storage_Images drop stores and return (0, 0, 0, 0) on loads.
	empty_res := make([]u32, 32, context.temp_allocator)
	empty_imgs := []Storage_Image {
		{width = 0, height = 0, format = .Rgba8},
		{width = 0, height = 0, format = .Rgba32f},
		{width = 0, height = 0, format = .R32f},
		{width = 0, height = 0, format = .R32ui},
		{width = 0, height = 0, format = .R32i},
	}
	run_compute(&mall.program, nil, {1, 1, 1}, temp_scratch(), {images = empty_imgs, buffers = {empty_res}})
	for i in 0 ..< 8 {
		// Out-of-bounds loads on 0x0 return (0, 0, 0, 0); bit 31 = 1 (lf.y==0 && lf.z==0) ^ floatBitsToUint(0.0).
		testing.expect_value(t, empty_res[i * 4 + 0], u32(0x8000_0000))
		testing.expect_value(t, empty_res[i * 4 + 1], u32(0))
		testing.expect_value(t, empty_res[i * 4 + 2], u32(0))
	}
}

// SDL_GPU compute binding order: read-only resources precede read-write/writeonly
// in reflection, SPIR-V descriptor sets (set 0 RO, set 1 RW, set 2 UBO), MSL flat
// indices, and Run_Options slices.
@(test)
test_compute_binding_order_and_reflection :: proc(t: ^testing.T) {
	src := `#version 310 es
layout(local_size_x = 16, local_size_y = 2) in;
uniform float uScale;
uniform sampler2D sTex0;
uniform sampler2DArray sTex1;

buffer BufRW0 { uint rw0[]; };
readonly buffer BufRO0 { uint ro0[]; };
writeonly buffer BufWO1 { uint wo1[]; };
readonly buffer BufRO1 { uint ro1[]; };

layout(r32ui) uniform highp uimage2D imgRW0;
layout(rgba8) readonly uniform highp image2D imgRO0;
layout(rgba32f) writeonly uniform highp image2D imgWO1;
layout(r32i) readonly uniform highp iimage2D imgRO1;

void main() {
    uint idx = gl_LocalInvocationIndex;
    vec4 t0 = textureLod(sTex0, vec2(0.25, 0.75), 0.0);
    vec4 t1 = textureGrad(sTex1, vec3(0.5, 0.5, 1.0), vec2(0.1, 0.0), vec2(0.0, 0.1));
    ivec2 sz = textureSize(sTex0, 0);
    vec4 c0 = imageLoad(imgRO0, ivec2(0, 0));
    ivec4 c1 = imageLoad(imgRO1, ivec2(0, 0));
    uvec4 prev = imageLoad(imgRW0, ivec2(int(idx), 0));
    uint sum = ro0[0] + ro1[0] + prev.x + uint(c1.x) + uint(sz.x);
    rw0[idx] += sum;
    wo1[idx] = sum * 2u;
    imageStore(imgRW0, ivec2(int(idx), 0), uvec4(sum, 0u, 0u, 0u));
    imageStore(imgWO1, ivec2(int(idx), 0), (t0 + t1 + c0) * uScale);
}`
	m, errs, ok := compile_compute(src)
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "binding_order", m.spirv)
	}

	testing.expect_value(t, m.threadcount, [3]u32{16, 2, 1})
	testing.expect_value(t, m.num_samplers, 2)
	testing.expect_value(t, m.num_readonly_storage_textures, 2)
	testing.expect_value(t, m.num_readonly_storage_buffers, 2)
	testing.expect_value(t, m.num_readwrite_storage_textures, 2)
	testing.expect_value(t, m.num_readwrite_storage_buffers, 2)
	testing.expect_value(t, m.num_uniform_buffers, 1)

	testing.expect_value(t, len(m.storage_buffers), 4)
	testing.expect(t, m.storage_buffers[0].name == "BufRO0" && m.storage_buffers[0].readonly && !m.storage_buffers[0].writeonly)
	testing.expect(t, m.storage_buffers[1].name == "BufRO1" && m.storage_buffers[1].readonly && !m.storage_buffers[1].writeonly)
	testing.expect(t, m.storage_buffers[2].name == "BufRW0" && !m.storage_buffers[2].readonly && !m.storage_buffers[2].writeonly)
	testing.expect(t, m.storage_buffers[3].name == "BufWO1" && !m.storage_buffers[3].readonly && m.storage_buffers[3].writeonly)

	testing.expect_value(t, len(m.storage_images), 4)
	testing.expect(t, m.storage_images[0].name == "imgRO0" && m.storage_images[0].format == .Rgba8 && m.storage_images[0].readonly && !m.storage_images[0].writeonly)
	testing.expect(t, m.storage_images[1].name == "imgRO1" && m.storage_images[1].format == .R32i && m.storage_images[1].readonly && !m.storage_images[1].writeonly)
	testing.expect(t, m.storage_images[2].name == "imgRW0" && m.storage_images[2].format == .R32ui && !m.storage_images[2].readonly && !m.storage_images[2].writeonly)
	testing.expect(t, m.storage_images[3].name == "imgWO1" && m.storage_images[3].format == .Rgba32f && !m.storage_images[3].readonly && m.storage_images[3].writeonly)

	// MSL flat binding slots: textures 0..1 = samplers, 2..3 = RO images, 4..5 = RW/WO images;
	// buffer 0 = UBO, buffers 1..2 = RO buffers, buffers 3..4 = RW/WO buffers.
	for want in ([]string {
		"kernel void oksl_main(",
		"texture2d<float> ok_sTex0_tex [[texture(0)]], sampler ok_sTex0_smp [[sampler(0)]]",
		"texture2d_array<float> ok_sTex1_tex [[texture(1)]], sampler ok_sTex1_smp [[sampler(1)]]",
		"texture2d<float, access::read> ok_imgRO0_img [[texture(2)]]",
		"texture2d<int, access::read> ok_imgRO1_img [[texture(3)]]",
		"texture2d<uint, access::read_write> ok_imgRW0_img [[texture(4)]]",
		"texture2d<float, access::write> ok_imgWO1_img [[texture(5)]]",
		"const device uint* oksl_buf_ro0 [[buffer(1)]]",
		"const device uint* oksl_buf_ro1 [[buffer(2)]]",
		"device uint* oksl_buf_rw0 [[buffer(3)]]",
		"device uint* oksl_buf_wo1 [[buffer(4)]]",
		"constant Oksl_Uniforms& u [[buffer(0)]]",
	}) {
		testing.expectf(t, strings.contains(m.msl, want), "MSL lacks %q in:\n%s", want, m.msl)
	}

	// Execute via run_compute using the reflection order.
	ro0 := []u32{10}
	ro1 := []u32{20}
	rw0 := make([]u32, 32, context.temp_allocator)
	wo1 := make([]u32, 32, context.temp_allocator)
	for i in 0 ..< 32 {
		rw0[i] = u32(i)
	}

	img_ro0 := [4]u8{255, 0, 0, 255}
	img_ro1 := [1]i32{7}
	img_rw0 := make([]u32, 32, context.temp_allocator)
	for i in 0 ..< 32 {
		img_rw0[i] = 5
	}
	img_wo1 := make([][4]f32, 32, context.temp_allocator)

	block := make([]u8, m.block_size, context.temp_allocator)
	testing.expect(t, set_uniform(m, block, "uScale", {2.0}))
	sampler_cb :: proc(user: rawptr, ch: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		return {f32(ch + 1), f32(layer), 0.5, 1.0}
	}
	opts := Run_Options {
		sampler       = sampler_cb,
		channel_sizes = {{64, 32, 1}, {16, 16, 4}},
		buffers       = {ro0, ro1, rw0, wo1},
		images        = {
			{data = img_ro0[:], width = 1, height = 1, format = .Rgba8},
			{data = slice.to_bytes(img_ro1[:]), width = 1, height = 1, format = .R32i},
			{data = slice.to_bytes(img_rw0), width = 32, height = 1, format = .R32ui},
			{data = slice.to_bytes(img_wo1), width = 32, height = 1, format = .Rgba32f},
		},
	}
	run_compute(&m.program, block, {1, 1, 1}, temp_scratch(), opts)

	// sum = 10 + 20 + 5 + 7 + 64 = 106
	for i in 0 ..< 32 {
		testing.expect_value(t, rw0[i], u32(i) + 106)
		testing.expect_value(t, wo1[i], u32(212))
		testing.expect_value(t, img_rw0[i], u32(106))
		// (t0={1,0,0.5,1} + t1={2,1,0.5,1} + c0={1,0,0,1}) * 2.0 = {8, 2, 2, 6}
		near(t, img_wo1[i], {8, 2, 2, 6})
	}
}

// Buffer lvalues (non-contiguous/composed swizzles, dynamic vector component,
// pre/post ++/--, out/inout args, writeonly stores), >64-lane multi-chunk
// workgroups, gl_WorkGroupSize constant expressions, and RW-to-RO/WO image params.
@(test)
test_compute_buffer_lvalues_and_workgroup_size :: proc(t: ^testing.T) {
	src_lval := `#version 310 es
layout(local_size_x = 4) in;
struct Item {
    vec4 v;
    ivec4 iv;
};
buffer Data { Item items[]; } data;
writeonly buffer Out { uvec4 res[]; } out_buf;

void set_out(out vec2 p) { p = vec2(9.0, 10.0); }
void bump_inout(inout float x) { x += 3.5; }

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= 4u) return;
    // Non-contiguous and composed swizzle stores + compound assignment.
    data.items[i].v.zx = vec2(10.0, 20.0);
    data.items[i].v.zyx.xz += vec2(1.0, 2.0); // .z += 1 -> 11, .x += 2 -> 22
    // Dynamic component store and compound assignment.
    int c = int(i & 3u);
    data.items[i].v[c] += 100.0;
    // Pre and post increment/decrement on buffer elements.
    int pre = ++data.items[i].iv.x;
    int post = data.items[i].iv.y--;
    // out and inout parameters backed by buffer lvalues.
    set_out(data.items[i].v.yw);
    bump_inout(data.items[i].v.w); // 10.0 + 3.5 = 13.5
    out_buf.res[i] = uvec4(uint(pre), uint(post), uint(data.items[i].iv.y), i);
}`
	ml, el, okl := compile_compute(src_lval)
	defer destroy(ml)
	if !testing.expectf(t, okl, "%s", format_errors(el, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "buffer_lvalues", ml.spirv)
	}

	Item_CPU :: struct {
		v:  [4]f32,
		iv: [4]i32,
	}
	items := make([]Item_CPU, 4, context.temp_allocator)
	for i in 0 ..< 4 {
		items[i] = {v = {1, 2, 3, 4}, iv = {10 + i32(i), 20 + i32(i), 0, 0}}
	}
	res_out := make([][4]u32, 4, context.temp_allocator)
	run_compute(&ml.program, nil, {1, 1, 1}, temp_scratch(), {
		buffers = {slice.reinterpret([]u32, items), slice.reinterpret([]u32, res_out)},
	})

	for i in 0 ..< 4 {
		// Before dynamic [c] += 100 and set_out(.yw)/bump_inout(.w):
		// v starts {1,2,3,4} -> .zx={10,20} => {20,2,10,4} -> .zyx.xz += {1,2} => {22,2,11,4}
		// then v[i] += 100, then .yw = {9, 10}, then .w += 3.5 => 13.5!
		want_v := [4]f32{22, 9, 11, 13.5}
		if i == 0 {
			want_v[0] += 100
		} else if i == 2 {
			want_v[2] += 100
		}
		// Note: for i==1 or i==3, v[i] += 100 happened BEFORE set_out(v.yw) overwrote y and w!
		near(t, items[i].v, want_v)
		testing.expect_value(t, items[i].iv[0], 11 + i32(i))
		testing.expect_value(t, items[i].iv[1], 19 + i32(i))
		testing.expect_value(t, res_out[i], [4]u32{u32(11 + i), u32(20 + i), u32(19 + i), u32(i)})
	}

	// 256-lane workgroup (4 chunks of 64 lanes), gl_WorkGroupSize constant array,
	// and passing a RW uimage2D to both readonly and writeonly helper params.
	src_256 := `#version 310 es
layout(local_size_x = 256) in;
const uint CHUNKS = gl_WorkGroupSize.x / 64u;
layout(r32ui) uniform highp uimage2D hist;

uint read_slot(readonly highp uimage2D im, int x) {
    return imageLoad(im, ivec2(x, 0)).x;
}
void write_slot(writeonly highp uimage2D im, int x, uint v) {
    imageStore(im, ivec2(x, 0), uvec4(v, 0u, 0u, 0u));
}
void main() {
    uint bins[gl_WorkGroupSize.x / 64u];
    for (uint k = 0u; k < CHUNKS; ++k) {
        bins[k] = k + 1u;
    }
    int gx = int(gl_GlobalInvocationID.x);
    uint cur = read_slot(hist, gx);
    uint tag = bins[gl_LocalInvocationIndex / 64u] * 1000u + gl_WorkGroupID.x * 500u + gl_LocalInvocationID.x;
    write_slot(hist, gx, cur + tag);
}`
	m256, e256, ok256 := compile_compute(src_256)
	defer destroy(m256)
	if !testing.expectf(t, ok256, "%s", format_errors(e256, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "workgroup_256", m256.spirv)
	}
	testing.expect(t, strings.contains(m256.msl, "template<access oksl_a0>"))
	testing.expect_value(t, m256.threadcount, [3]u32{256, 1, 1})

	slots := make([]u32, 512, context.temp_allocator)
	for i in 0 ..< 512 {
		slots[i] = 7
	}
	run_compute(&m256.program, nil, {2, 1, 1}, temp_scratch(), {
		images = {{data = slice.to_bytes(slots), width = 512, height = 1, format = .R32ui}},
	})
	for wg in 0 ..< 2 {
		for lid in 0 ..< 256 {
			gx := wg * 256 + lid
			want := u32(7 + ((lid / 64) + 1) * 1000 + wg * 500 + lid)
			if slots[gx] != want {
				testing.expectf(t, false, "slot %d: got %d, want %d", gx, slots[gx], want)
				break
			}
		}
	}
}

// Phase 1 compute & graphics stage diagnostics.
@(test)
test_compute_diagnostics :: proc(t: ^testing.T) {
	expect_code :: proc(t: ^testing.T, src: string, want: Error_Code, opts := Options{profile = .Glsl, stage = .Compute}, loc := #caller_location) {
		m, errs, ok := compile(src, opts)
		defer destroy(m)
		found := false
		for e in errs {
			if e.code == want {
				found = true
				break
			}
		}
		testing.expectf(t, !ok && found, "expected %v for:\n%s\ngot:\n%s", want, src, format_errors(errs, context.temp_allocator), loc = loc)
	}

	// Dialect & options restrictions.
	expect_code(t, "half4 main(float2 p) { return half4(1); }", .Stage_Unsupported, {profile = .Sksl, stage = .Compute})
	expect_code(t, "void mainImage(out vec4 o, in vec2 p) { o = vec4(1); }", .Stage_Unsupported, {profile = .Shadertoy, stage = .Compute})
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() {}", .Tap_Stage, {profile = .Glsl, stage = .Compute, taps = {{position = "p"}}})

	// Workgroup local_size rules.
	expect_code(t, "void main() {}", .Missing_Local_Size)
	expect_code(t, "int n = 4;\nlayout(local_size_x = n) in;\nvoid main() {}", .Local_Size_Constant)
	expect_code(t, "const int N = 4;\nlayout(local_size_x = N) in;\nvoid main() {}", .Local_Size_Constant)
	expect_code(t, "layout(local_size_x = 4 * 2) in;\nvoid main() {}", .Local_Size_Constant)
	expect_code(t, "layout(local_size_x = (8)) in;\nvoid main() {}", .Local_Size_Constant)
	expect_code(t, "layout(local_size_x = ~0u) in;\nvoid main() {}", .Local_Size_Constant)
	expect_code(t, "layout(local_size_x = -4) in;\nvoid main() {}", .Local_Size_Constant)
	expect_code(t, "layout(local_size_x = 0) in;\nvoid main() {}", .Local_Size_Positive)
	expect_code(t, "layout(local_size_x = 32, local_size_y = 32, local_size_z = 2) in;\nvoid main() {}", .Local_Size_Max)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(local_size_x = 16) in;\nvoid main() {}", .Local_Size_Conflict)
	// Matching duplicate declaration and uint/hex literals are valid GLSL ES 3.1.
	{
		m, errs, ok := compile_compute("layout(local_size_x = 8u, local_size_y = 0x8) in;\nlayout(local_size_y = 8, local_size_x = 010) in;\nvoid main() {}")
		defer destroy(m)
		testing.expectf(t, ok && m.threadcount == {8, 8, 1}, "%s", format_errors(errs, context.temp_allocator))
	}

	// Stage boundaries: compute rejects graphics features; graphics rejects compute features.
	expect_code(t, "layout(local_size_x = 8) in;\nin vec2 v;\nvoid main() {}", .Compute_Stage_Var)
	expect_code(t, "layout(local_size_x = 8) in;\nout vec4 o;\nvoid main() {}", .Compute_Stage_Var)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() { discard; }", .Fragment_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nuniform float u;\nvoid main() { float d = dFdx(u); }", .Fragment_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nuniform float u;\nvoid main() { float d = fwidth(u); }", .Fragment_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() { vec4 c = gl_FragCoord; }", .Fragment_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() { bool f = gl_FrontFacing; }", .Fragment_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() { gl_SampleMask[0] = 1; }", .Fragment_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() { gl_Position = vec4(0); }", .Vertex_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() { int v = gl_VertexID; }", .Vertex_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid main() { gl_WorkGroupSize = uvec3(1u); }", .Is_Const)

	// Implicit-LOD sampling rejected in compute.
	expect_code(t, "layout(local_size_x = 8) in;\nuniform sampler2D s;\nvoid main() { vec4 c = texture(s, vec2(0.5)); }", .Compute_Implicit_Sample)
	expect_code(t, "layout(local_size_x = 8) in;\nuniform sampler2D s;\nvoid main() { vec4 c = texture(s, vec2(0.5), 1.0); }", .Compute_Implicit_Sample)
	expect_code(t, "layout(local_size_x = 8) in;\nuniform sampler2D s;\nvoid main() { vec4 c = textureProj(s, vec3(0.5, 0.5, 1.0)); }", .Compute_Implicit_Sample)

	// Compute-only features rejected in Fragment and Vertex.
	frag_opts := Options{profile = .Glsl, stage = .Fragment}
	vert_opts := Options{profile = .Glsl, stage = .Vertex}
	expect_code(t, "layout(local_size_x = 8) in;\nout vec4 o;\nvoid main() { o = vec4(1); }", .Compute_Only, frag_opts)
	expect_code(t, "out vec4 o;\nvoid main() { o = vec4(float(gl_GlobalInvocationID.x)); }", .Compute_Only, frag_opts)
	expect_code(t, "void main() { gl_Position = vec4(float(gl_LocalInvocationIndex)); }", .Compute_Only, vert_opts)
	expect_code(t, "out vec4 o;\nvoid main() { o = vec4(float(gl_WorkGroupSize.x)); }", .Compute_Only, frag_opts)
	expect_code(t, "layout(rgba8) readonly uniform image2D im;\nout vec4 o;\nvoid main() { o = imageLoad(im, ivec2(0)); }", .Compute_Only, frag_opts)
	expect_code(t, "layout(rgba8) readonly uniform image2D im;\nvoid main() { gl_Position = imageLoad(im, ivec2(0)); }", .Compute_Only, vert_opts)
	expect_code(t, "buffer B { float a[]; } b;\nout vec4 o;\nvoid main() { o = vec4(b.a[0]); }", .Buffer_Writable, frag_opts)
	expect_code(t, "writeonly buffer B { float a[]; } b;\nvoid main() { gl_Position = vec4(1); }", .Buffer_Writable, vert_opts)

	// Buffer & image access qualifiers.
	expect_code(t, "layout(local_size_x = 8) in;\nreadonly writeonly buffer B { uint a[]; } b;\nvoid main() {}", .Readonly_Writeonly)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba8) readonly writeonly uniform image2D im;\nvoid main() {}", .Readonly_Writeonly)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid f(readonly writeonly image2D im) {}\nvoid main() {}", .Readonly_Writeonly)

	expect_code(t, "layout(local_size_x = 8) in;\nreadonly buffer B { uint a[]; } b;\nvoid main() { b.a[0] = 1u; }", .Read_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nreadonly buffer B { uint a[]; } b;\nvoid main() { b.a[0] += 1u; }", .Read_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nreadonly buffer B { uint a[]; } b;\nvoid main() { b.a[0]++; }", .Read_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nreadonly buffer B { uint a[]; } b;\nvoid f(inout uint x) { x = 1u; }\nvoid main() { f(b.a[0]); }", .Read_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba8) readonly uniform image2D im;\nvoid main() { imageStore(im, ivec2(0), vec4(1)); }", .Read_Only)

	expect_code(t, "layout(local_size_x = 8) in;\nwriteonly buffer B { uint a[]; } b;\nbuffer Out { uint y[]; } o;\nvoid main() { o.y[0] = b.a[0]; }", .Write_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nwriteonly buffer B { uint a[]; } b;\nvoid main() { b.a[0] += 1u; }", .Write_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nwriteonly buffer B { uint a[]; } b;\nvoid main() { ++b.a[0]; }", .Write_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nwriteonly buffer B { uint a[]; } b;\nvoid f(inout uint x) { x += 1u; }\nvoid main() { f(b.a[0]); }", .Write_Only)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba8) writeonly uniform image2D im;\nbuffer Out { vec4 y[]; } o;\nvoid main() { o.y[0] = imageLoad(im, ivec2(0)); }", .Write_Only)

	// Image format and parameter rules.
	expect_code(t, "layout(local_size_x = 8) in;\nreadonly uniform image2D im;\nvoid main() {}", .Image_Format_Required)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid f(layout(rgba8) readonly image2D im) {}\nvoid main() {}", .Param_Layout)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba16ui) readonly uniform uimage2D im;\nvoid main() {}", .Image_Format_Unknown)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba8) readonly uniform uimage2D im;\nvoid main() {}", .Image_Format_Mismatch)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(r32ui) uniform image2D im;\nvoid main() {}", .Image_Format_Mismatch)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(r32i) uniform uimage2D im;\nvoid main() {}", .Image_Format_Mismatch)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba8) uniform image2D im;\nvoid main() {}", .Image_RW_Format)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba16f) uniform image2D im;\nvoid main() {}", .Image_RW_Format)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba32f) uniform image2D im;\nvoid main() {}", .Image_RW_Format)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid f(out uimage2D im) {}\nvoid main() {}", .Image_Out_Param)
	expect_code(t, "layout(local_size_x = 8) in;\nvoid f(inout uimage2D im) {}\nvoid main() {}", .Image_Out_Param)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba8) readonly uniform image2D im;\nvoid f(writeonly image2D p) {}\nvoid main() { f(im); }", .No_Overload)
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(rgba8) readonly uniform image2D a;\nlayout(rgba16f) readonly uniform image2D b;\nvec4 f(readonly image2D p) { return imageLoad(p, ivec2(0)); }\nvoid main() { vec4 x = f(a) + f(b); }", .Image_Param_Format)

	// > MAX_STORAGE_IMAGES (16) limit.
	{
		b := strings.builder_make(context.temp_allocator)
		strings.write_string(&b, "layout(local_size_x = 8) in;\n")
		for i in 0 ..< 17 {
			fmt.sbprintf(&b, "layout(rgba8) readonly uniform image2D im%d;\n", i)
		}
		strings.write_string(&b, "buffer Out { float v[]; } o;\nvoid main() { o.v[0] = ")
		for i in 0 ..< 17 {
			if i > 0 {
				strings.write_string(&b, " + ")
			}
			fmt.sbprintf(&b, "imageLoad(im%d, ivec2(0)).x", i)
		}
		strings.write_string(&b, "; }\n")
		expect_code(t, strings.to_string(b), .Too_Many_Images)
	}
}

// Regression tests for all Round 1 compute review items (evaluation order &
// aliasing, std430 vec3[] strides, local_size 64-bit overflow, opaque ?:,
// gl_WorkGroupSize lexical ordering, exact rgba8 imageLoad, MSL parameter
// shadowing & read_write fence, row_major/column_major buffer layouts, AST
// record sizes, portable layout-less image parameters, and diagnostic nits).
@(test)
test_compute_review_fixes :: proc(t: ^testing.T) {
	// 1. Evaluation order & aliasing (VM + MSL sequencing + oksl_fence).
	alias_src := `#version 310 es
layout(local_size_x = 1) in;
buffer B { uint w[]; } b;
buffer V { uvec2 v[]; } vb;
layout(r32ui) uniform highp uimage2D im;
uint g = 0u;

void setg(out uint dst) {
    g = 3u;
    dst = 77u;
}
uvec4 mutate_p(inout ivec2 p) {
    p = ivec2(5, 0);
    return uvec4(99u, 0u, 0u, 0u);
}
uint bump(inout uint x) {
    x += 10u;
    return 3u;
}

void main() {
    uint k = 5u;
    b.w[k] = k++;                      // b.w[5] = 5, k = 6
    b.w[0] = k;

    uint m = 7u;
    b.w[m] += (m = 20u);               // b.w[7] += 20 (was 1) -> 21, m = 20

    g = 1u;
    setg(b.w[g]);                      // b.w[1] = 77 (addressed before setg mutates g=3)

    uint j = 0u;
    vb.v[0] = uvec2(10u, 20u);
    vb.v[1] = uvec2(30u, 40u);
    vb.v[j].y += vb.v[j++].x;          // vb.v[0].y = 20 + 10 = 30, j = 1
    b.w[2] = j;

    ivec2 p = ivec2(2, 0);
    imageStore(im, p, mutate_p(p));    // stores 99 at x=2, not x=5
    b.w[8] = imageLoad(im, ivec2(2, 0)).x;
    b.w[9] = imageLoad(im, ivec2(5, 0)).x;

    uint q = 10u;
    b.w[10] = q - bump(q);             // 10 - 3 = 7 (LHS=10 snapshotted before bump mutates q=20)
    b.w[11] = q;

    uint r = 4u;
    b.w[12] = max(r, bump(r));         // max(4, 3) = 4 (arg0=4 snapshotted before bump mutates r=14)
    b.w[13] = r;

    b.w[3] = 5u;
    b.w[3] += (b.w[3] = 100u);         // RHS sets b.w[3]=100, then LHS loads 100 + 100 = 200

    uint a[4] = uint[4](1u, 2u, 3u, 4u);
    uint am = 1u;
    a[am] += (am = 2u);                // a[1] = 2 + 2 = 4, am = 2
    b.w[14] = a[1];
    b.w[15] = a[2];

    uint x = 5u;
    x += (x = 100u);                   // RHS sets x=100, then 100 + 100 = 200
    b.w[16] = x;
}`
	ma, ea, oka := compile_compute(alias_src)
	defer destroy(ma)
	if !testing.expectf(t, oka, "%s", format_errors(ea, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "alias_fixes", ma.spirv)
	}
	testing.expect(t, strings.contains(ma.msl, "oksl_fence(ok_im_img)"))

	bw := make([]u32, 32, context.temp_allocator)
	bw[7] = 1
	vw := make([]u32, 8, context.temp_allocator)
	imw := make([]u32, 8, context.temp_allocator)
	run_compute(&ma.program, nil, {1, 1, 1}, temp_scratch(), {
		buffers = {bw, vw},
		images  = {{data = slice.to_bytes(imw), width = 8, height = 1, format = .R32ui}},
	})
	testing.expect_value(t, bw[5], u32(5))
	testing.expect_value(t, bw[0], u32(6))
	testing.expect_value(t, bw[7], u32(21))
	testing.expect_value(t, bw[1], u32(77))
	testing.expect_value(t, bw[3], u32(200))
	testing.expect_value(t, vw[0], u32(10))
	testing.expect_value(t, vw[1], u32(30))
	testing.expect_value(t, bw[2], u32(1))
	testing.expect_value(t, imw[2], u32(99))
	testing.expect_value(t, imw[5], u32(0))
	testing.expect_value(t, bw[8], u32(99))
	testing.expect_value(t, bw[9], u32(0))
	testing.expect_value(t, bw[10], u32(7))
	testing.expect_value(t, bw[11], u32(20))
	testing.expect_value(t, bw[12], u32(4))
	testing.expect_value(t, bw[13], u32(14))
	testing.expect_value(t, bw[14], u32(4))
	testing.expect_value(t, bw[15], u32(3))
	testing.expect_value(t, bw[16], u32(200))

	// 2. std430 runtime-sized array stride on vec3/ivec3/uvec3 vs structs.
	stride_src := `#version 310 es
layout(local_size_x = 4) in;
struct PackA { vec3 v; float s; };
struct PackB { vec3 a; vec3 b; };
struct Single { vec3 v; };
readonly buffer B0 { vec3 f3[]; } b0;
readonly buffer B1 { ivec3 i3[]; } b1;
buffer B2 { uvec3 u3[]; } b2;
readonly buffer B3 { PackA pa[]; } b3;
readonly buffer B4 { PackB pb[]; } b4;
readonly buffer B5 { Single sg[]; } b5;
readonly buffer B6 { float f1[]; } b6;
void main() {
    uint i = gl_GlobalInvocationID.x;
    vec3 f = b0.f3[i] + b3.pa[i].v + b4.pb[i].b + b5.sg[i].v;
    ivec3 iv = b1.i3[i];
    b2.u3[i] += uvec3(uint(f.x + b6.f1[i]), uint(iv.y), uint(f.z));
}`
	ms, es, oks := compile_compute(stride_src)
	defer destroy(ms)
	if !testing.expectf(t, oks, "%s", format_errors(es, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "std430_strides", ms.spirv)
	}
	// Binding order: readonly B0, B1, B3, B4, B5, B6 first, then RW B2.
	want_strides := [7]int{16, 16, 16, 32, 16, 4, 16}
	for ws, i in want_strides {
		testing.expect_value(t, ms.storage_buffers[i].stride, ws)
	}
	b0w := make([]u32, 16, context.temp_allocator)
	b1w := make([]u32, 16, context.temp_allocator)
	b3w := make([]u32, 16, context.temp_allocator)
	b4w := make([]u32, 32, context.temp_allocator)
	b5w := make([]u32, 16, context.temp_allocator)
	b6w := make([]u32, 4, context.temp_allocator)
	b2w := make([]u32, 16, context.temp_allocator)
	for i in 0 ..< 4 {
		// Element i of vec3[] sits at word i * 4 (16-byte stride), NOT i * 3!
		b0w[i * 4 + 0] = transmute(u32)f32(i + 1)
		b0w[i * 4 + 2] = transmute(u32)f32(10 + i)
		b1w[i * 4 + 1] = u32(20 + i)
		b3w[i * 4 + 0] = transmute(u32)f32(2)
		b4w[i * 8 + 4] = transmute(u32)f32(3) // PackB.b is at offset 16 (word 4) inside 32-byte struct
		b5w[i * 4 + 0] = transmute(u32)f32(4)
		b6w[i] = transmute(u32)f32(5)
		b2w[i * 4 + 0] = 100
		b2w[i * 4 + 3] = 0xdead_beef // padding word 3 must remain untouched
	}
	run_compute(&ms.program, nil, {1, 1, 1}, temp_scratch(), {
		buffers = {b0w, b1w, b3w, b4w, b5w, b6w, b2w},
	})
	for i in 0 ..< 4 {
		// f.x = (i+1) + 2 + 3 + 4 = i + 10; + f1[i](5) = i + 15; + 100 = 115 + i
		testing.expect_value(t, b2w[i * 4 + 0], u32(115 + i))
		testing.expect_value(t, b2w[i * 4 + 1], u32(20 + i))
		testing.expect_value(t, b2w[i * 4 + 2], u32(10 + i))
		testing.expect_value(t, b2w[i * 4 + 3], u32(0xdead_beef))
	}

	// 6. Exact rgba8 imageLoad for all 256 byte values (0 ulp vs b / 255.0f).
	rgba8_src := `#version 310 es
layout(local_size_x = 64) in;
layout(rgba8) readonly uniform highp image2D src;
writeonly buffer Out { vec4 vals[]; } out_buf;
void main() {
    int x = int(gl_GlobalInvocationID.x);
    out_buf.vals[x] = imageLoad(src, ivec2(x, 0));
}`
	m8, e8, ok8 := compile_compute(rgba8_src)
	defer destroy(m8)
	if !testing.expectf(t, ok8, "%s", format_errors(e8, context.temp_allocator)) {
		return
	}
	in_bytes := make([]u8, 256, context.temp_allocator)
	for i in 0 ..< 256 {
		in_bytes[i] = u8(i)
	}
	out_f := make([][4]f32, 64, context.temp_allocator)
	run_compute(&m8.program, nil, {1, 1, 1}, temp_scratch(), {
		images  = {{data = in_bytes, width = 64, height = 1, format = .Rgba8}},
		buffers = {slice.reinterpret([]u32, out_f)},
	})
	for i in 0 ..< 256 {
		got := out_f[i / 4][i % 4]
		want := f32(i) / f32(255)
		if transmute(u32)got != transmute(u32)want {
			testing.expectf(t, false, "rgba8 byte %d: got 0x%08x, want 0x%08x", i, transmute(u32)got, transmute(u32)want)
			break
		}
	}

	// 7 & 13. MSL parameter shadowing (image + sampler) & chained layout-less image params.
	shadow_src := `#version 310 es
layout(local_size_x = 4) in;
uniform sampler2D s;
layout(rgba8) readonly uniform highp image2D src;
layout(r32ui) uniform highp uimage2D rw;
writeonly buffer Out { vec4 o[]; } out_buf;

vec4 leaf(readonly highp image2D src, sampler2D s, ivec2 p) {
    return imageLoad(src, p) + textureLod(s, vec2(0.5), 0.0);
}
vec4 mid(readonly highp image2D src, sampler2D s, ivec2 p) {
    return leaf(src, s, p);
}
void bump_rw(highp uimage2D rw, int x) {
    uint v = imageLoad(rw, ivec2(x, 0)).x;
    imageStore(rw, ivec2(x, 0), uvec4(v + 1u, 0u, 0u, 0u));
}
void main() {
    int x = int(gl_GlobalInvocationID.x);
    bump_rw(rw, x);
    out_buf.o[x] = mid(src, s, ivec2(x, 0));
}`
	msh, esh, oksh := compile_compute(shadow_src)
	defer destroy(msh)
	if !testing.expectf(t, oksh, "%s", format_errors(esh, context.temp_allocator)) {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "shadow_and_chain", msh.spirv)
	}
	testing.expect(t, strings.contains(msh.msl, "okp_src_img"))
	testing.expect(t, strings.contains(msh.msl, "okp_s_tex"))
	testing.expect(t, strings.contains(msh.msl, "okp_s_smp"))
	testing.expect(t, strings.contains(msh.msl, "oksl_fence(okp_rw_img)"))

	// 9. row_major and column_major accepted on storage buffers in fragment, vertex, and compute.
	for st in ([]Stage{.Fragment, .Vertex, .Compute}) {
		rm_src := `#version 310 es
precision highp float;
layout(local_size_x = 4) in;
layout(std430, row_major) readonly buffer A { vec4 x[]; } a;
layout(column_major, std430) readonly buffer B { vec4 y[]; } b;
void main() {}`
		if st == .Fragment {
			rm_src = `#version 310 es
precision highp float;
layout(std430, row_major) readonly buffer A { vec4 x[]; } a;
layout(column_major) readonly buffer B { vec4 y[]; } b;
out vec4 o;
void main() { o = a.x[0] + b.y[0]; }`
		} else if st == .Vertex {
			rm_src = `#version 310 es
layout(std430, row_major) readonly buffer A { vec4 x[]; } a;
layout(column_major) readonly buffer B { vec4 y[]; } b;
void main() { gl_Position = a.x[0] + b.y[0]; }`
		}
		mr, er, okr := compile(rm_src, {profile = .Glsl, stage = st})
		defer destroy(mr)
		testing.expectf(t, okr, "stage %v failed row_major/column_major: %s", st, format_errors(er, context.temp_allocator))
	}

	// 3, 4, 5, 13, 15. Diagnostics & single-error guarantees.
	expect_err :: proc(t: ^testing.T, src: string, want: Error_Code, opts := Options{profile = .Glsl, stage = .Compute}, loc := #caller_location) {
		m, errs, ok := compile(src, opts)
		defer destroy(m)
		found := false
		for e in errs {
			if e.code == want {
				found = true
				break
			}
		}
		testing.expectf(t, !ok && found, "expected %v for:\n%s\ngot:\n%s", want, src, format_errors(errs, context.temp_allocator), loc = loc)
	}

	// 3. 64-bit multiplication overflow in local_size.
	expect_err(t, "layout(local_size_x = 2097152, local_size_y = 2097152, local_size_z = 2097152) in;\nvoid main() {}", .Local_Size_Max)
	expect_err(t, "layout(local_size_x = 65536, local_size_y = 65536, local_size_z = 2) in;\nvoid main() {}", .Local_Size_Max)
	expect_err(t, "layout(local_size_x = 1025) in;\nvoid main() {}", .Local_Size_Max)

	// 4. Opaque types (images and samplers) in ?: and function return types rejected.
	expect_err(t, "layout(local_size_x = 4) in;\nuniform bool c;\nlayout(rgba8) readonly uniform image2D a;\nlayout(rgba8) readonly uniform image2D b;\nvoid main() { vec4 v = imageLoad(c ? a : b, ivec2(0)); }", .Opaque_Ternary)
	expect_err(t, "precision highp float;\nuniform bool c;\nuniform sampler2D a;\nuniform sampler2D b;\nout vec4 o;\nvoid main() { o = texture(c ? a : b, vec2(0.5)); }", .Opaque_Ternary, {profile = .Glsl, stage = .Fragment})
	expect_err(t, "layout(local_size_x = 4) in;\nlayout(r32f) uniform image2D a;\nimage2D get() { return a; }\nvoid main() { imageStore(get(), ivec2(0), vec4(1)); }", .Opaque_Return)
	expect_err(t, "precision highp float;\nuniform sampler2D s;\nsampler2D get() { return s; }\nout vec4 o;\nvoid main() { o = texture(get(), vec2(0.5)); }", .Opaque_Return, {profile = .Glsl, stage = .Fragment})

	// 5. gl_WorkGroupSize referenced before layout(local_size_*) in;.
	expect_err(t, "const uint N = gl_WorkGroupSize.x;\nlayout(local_size_x = 8) in;\nvoid main() {}", .Work_Group_Size_Before_Decl)
	expect_err(t, "uint f() { return gl_WorkGroupSize.x; }\nlayout(local_size_x = 8) in;\nvoid main() {}", .Work_Group_Size_Before_Decl)

	// 13. Missing format on uniform image fires once (not twice on parameter).
	{
		m, errs, ok := compile_compute("layout(local_size_x = 4) in;\nreadonly uniform image2D im;\nvoid main() {}")
		defer destroy(m)
		testing.expect(t, !ok && len(errs) == 1 && errs[0].code == .Image_Format_Required)
	}

	// 15. Misleading diagnostics & enum ordinal stability vs v0.0.9.
	expect_err(t, "layout(local_size_x = 1) in;\nlayout(local_size_x = 8) in;\nvoid main() {}", .Local_Size_Conflict)
	expect_err(t, "precision highp float;\nlayout(early_fragment_tests) in;\nout vec4 o;\nvoid main() { o = vec4(1); }", .Image_Format_Unknown, {profile = .Glsl, stage = .Fragment})
	{
		for tc in ([][2]string {
			{"image2D", "an image2D must be a uniform"},
			{"iimage2D", "an iimage2D must be a uniform"},
			{"uimage2D", "a uimage2D must be a uniform"},
		}) {
			m, errs, ok := compile_compute(fmt.tprintf("layout(local_size_x = 4) in;\nvoid main() {{ %s x; }}", tc[0]))
			defer destroy(m)
			msg := format_errors(errs, context.temp_allocator)
			testing.expectf(t, !ok && strings.contains(msg, tc[1]), "expected %q in %q", tc[1], msg)
		}
	}

	// AST record sizes and v0.0.9 enum ordinals.
	testing.expect(t, size_of(Var_Decl) <= 64)
	testing.expect(t, size_of(Param) <= 48)
	testing.expect(t, size_of(Buffer_Decl) <= 80)
	testing.expect(t, size_of(Top) <= 24)
	testing.expect_value(t, int(Error_Code.Pass_Missing), 159)
	testing.expect_value(t, int(Interface_Kind.Buffer), 9)
	testing.expect_value(t, int(Builtin.Uint_Bits_To_Float), 73)
	testing.expect_value(t, int(Storage.Stage_Out), 8)
	testing.expect_value(t, int(Feature.Gl_Sample_Mask), 15)
}

@(test)
test_compute_round2_fixes :: proc(t: ^testing.T) {
	expect_code :: proc(t: ^testing.T, body: string, code: Error_Code, opts := Options{profile = .Glsl, stage = .Compute}, loc := #caller_location) {
		m, errs, ok := compile(fmt.tprintf("#version 310 es\n%s", body), opts)
		defer destroy(m)
		testing.expect(t, !ok && len(errs) > 0, loc = loc)
		if len(errs) > 0 {
			testing.expect_value(t, errs[0].code, code, loc = loc)
		}
	}
	// Every declared MSL temporary `_tN = {};` in oksl_main must be assigned `(_tN = `
	// in the emitted body (regression test for dropped `pre` wrappers in constructors
	// and built-in calls), and C++14 unsequenced forms must be sequenced.
	src_bytes, rerr := os.read_entire_file("testdata/eval_order.comp", context.temp_allocator)
	testing.expect(t, rerr == nil)
	if rerr != nil {
		return
	}
	m, errs, ok := compile(string(src_bytes), {profile = .Glsl, stage = .Compute})
	defer destroy(m)
	testing.expectf(t, ok, "eval_order.comp failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	if have_spirv_val() {
		expect_valid_spirv(t, "eval_order.comp", m.spirv)
	}

	main_pos := strings.index(m.msl, "void ok_main(")
	testing.expect(t, main_pos >= 0)
	if main_pos >= 0 {
		rest_msl := m.msl[main_pos:]
		end_fn := strings.index(rest_msl, "\n}\n")
		fn_msl := rest_msl[:end_fn] if end_fn >= 0 else rest_msl
		temp_count := 0
		for line in strings.split_lines(fn_msl, context.temp_allocator) {
			trimmed := strings.trim_space(line)
			if ix := strings.index(trimmed, " _t"); ix >= 0 && strings.has_suffix(trimmed, " = {};") {
				rest := trimmed[ix + 1:]
				eq := strings.index(rest, " = {};")
				tname := rest[:eq]
				assign_pat := fmt.tprintf("(%s = ", tname)
				testing.expectf(t, strings.contains(fn_msl, assign_pat), "MSL temporary %s declared but never assigned in ok_main", tname)
				temp_count += 1
			}
		}
		testing.expect(t, temp_count >= 10)
		// C++14 sequencing checks:
		// - binary LHS write sequenced before RHS
		testing.expect(t, !strings.contains(fn_msl, "((ok_s = ") || !strings.contains(fn_msl, ") + uint2(ok_s"))
		// - nested subscript side effect sequenced
		testing.expect(t, !strings.contains(fn_msl, "ok_mat[(ok_mi++)][ok_mi]"))
		// - built-in and constructor sequencing when argument 0 writes (`max(bump(e1), e1)`, `uvec2(bump(e2), e2)`, `Pair(ce++, ce)`)
		testing.expect(t, !strings.contains(fn_msl, ", ok_e1)") && strings.contains(fn_msl, " = ok_e1)"))
		testing.expect(t, !strings.contains(fn_msl, ", ok_e2)") && strings.contains(fn_msl, " = ok_e2)"))
		testing.expect(t, !strings.contains(fn_msl, ", ok_ce}") && strings.contains(fn_msl, " = ok_ce)"))
		// - `=` sequencing (`eq_x = eq_x++`) and indexed `=` rvalue tail (`(ok_eq_v[_tX] = _tY), _tY)`)
		testing.expect(t, !strings.contains(fn_msl, "(ok_eq_x = (ok_eq_x++))"))
		eq_ix := strings.index(fn_msl, "(ok_eq_v[")
		testing.expect(t, eq_ix >= 0)
		if eq_ix >= 0 {
			after_eq := fn_msl[eq_ix:]
			close_ix := strings.index(after_eq, "))")
			testing.expect(t, close_ix > 0 && strings.contains(after_eq[:close_ix], "), _t"))
		}
		// - pre-write and post-write oksl_fence on imageStore, with arguments hoisted into temporaries before pre-write fence
		testing.expect(t, strings.contains(fn_msl, "(oksl_fence(ok_rw_img_img), ok_rw_img_img.write(_t"))
		testing.expect(t, strings.contains(fn_msl, "), oksl_fence(ok_rw_img_img))"))
		// - void `?:` casts both arms to `(void)` so an out/inout arm compiles opposite a void arm in Metal
		testing.expect(t, strings.contains(fn_msl, "? (void)((_t"))
		testing.expect(t, strings.contains(fn_msl, ") : (void)ok_nop_void("))
	}

	// Execute eval_order.comp on CPU and verify exact results across all 16 invocations.
	buf := make([]u32, 320, context.temp_allocator)
	img_words := make([]u32, 32 * 16, context.temp_allocator)
	img := Storage_Image{
		data   = slice.to_bytes(img_words),
		width  = 32,
		height = 16,
		format = .R32ui,
	}
	ublock := make([]u8, m.block_size, context.temp_allocator)
	set_uniform(m, ublock, "uMatArr", {1, 2, 3, 4}, 0)
	set_uniform(m, ublock, "uMatArr", {5, 6, 7, 8}, 1)
	run_compute(&m.program, ublock, {1, 1, 1}, temp_scratch(), {
		buffers = {buf},
		images  = {img},
	})
	for gy in u32(0) ..< 4 {
		for gx in u32(0) ..< 4 {
			id := gy * 4 + gx
			base := id * 18
			seed := id + 2
			testing.expect_value(t, buf[base + 0], max(seed, u32(1)) * 100 + (seed + 10))
			testing.expect_value(t, buf[base + 1], (seed + 3) * 100 + 1 + (seed + 13))
			testing.expect_value(t, buf[base + 2], (seed + 5) * 100 + (seed + 15))
			testing.expect_value(t, buf[base + 3], (seed + 1) * 10000 + (seed + 2) * 100 + 1)
			testing.expect_value(t, buf[base + 4], (seed + 9) * 2)
			testing.expect_value(t, buf[base + 5], (seed + 10) * 10000 + 1 * 100 + (seed + 11))
			testing.expect_value(t, buf[base + 6], (seed + 4) * 1000 + 1 * 100 + (seed + 6) * 10 + 1)
			// snap_inout(x, x = seed + 20u): old a is seed, b is seed + 20, final x is 2*seed + 20.
			want_sr := seed * 100 + (seed + 20)
			testing.expect_value(t, buf[base + 7], want_sr * 100 + (2 * seed + 20))
			// mat[mi++][mi]: mat[0][1] = seed + 2, final mi = 1.
			testing.expect_value(t, buf[base + 8], (seed + 2) * 10 + 1)
			tx, ty_c := int(gx) * 7, int(gy) * 4
			w1 := 1 + seed * 10
			w2 := u32(tx + 2) + 1 + seed
			testing.expect_value(t, buf[base + 9], w1 * 1000 + w2)
			testing.expect_value(t, img_words[ty_c * 32 + tx], w1)
			testing.expect_value(t, img_words[(ty_c + 2) * 32 + (tx + 2)], w2)
			ip_end_x, ip_end_y := tx + 1, ty_c + 1
			ip2_end_x, ip2_end_y := tx + 3, ty_c + 1
			testing.expect_value(t, buf[base + 10], u32(ip_end_x * 100 + ip_end_y * 10 + ip2_end_x + ip2_end_y))
			// u3 = (1, 1, 11) -> 121; carr = (1, 2, 3) -> 123; cc = 12; cp = (1, 2) -> 12.
			testing.expect_value(t, buf[base + 11], u32(121 * 1000 + 123 + 12 + 12))
			// inout order: two(3, 7) = 37 across swizzle, static/dynamic array, and buffer; two(c, two(c, 5)) = 45, c = 45.
			testing.expect_value(t, buf[base + 12], u32(3737))
			testing.expect_value(t, buf[base + 13], u32(3737))
			testing.expect_value(t, buf[base + 14], u32(4545))
			war_acc := u32(0)
			for wi in u32(0) ..< 4 {
				war_acc = war_acc * 10 + ((seed + wi * 13) % 10)
			}
			testing.expect_value(t, img_words[(ty_c + 1) * 32 + (tx + 3)], seed + 52)
			testing.expect_value(t, buf[base + 15], war_acc * 1000 + 115)
			// uMatArr[0][1].y = 4 -> 40; uMatArr[1][1].x = 7 -> 70; mn = 1, mr = 0 -> 4080.
			testing.expect_value(t, buf[base + 16], u32(4080))
			vt_want := (seed + 5) if (id & 1) == 0 else (seed + 6)
			vf_want := ((seed + 11) if (id & 2) == 0 else (seed + 22)) + (30 if (id & 1) == 0 else 0)
			testing.expect_value(t, img_words[ty_c * 32 + (tx + 4)], vt_want)
			testing.expect_value(t, buf[base + 17], vt_want * 100 + vf_want)
		}
	}

	// Fragment inout argument evaluation order (JIT and interpreter must both give 37, 37, 45, 45).
	{
		fsrc := `#version 300 es
precision highp float;
precision highp int;
out vec4 o;
uint two(inout uint x, uint y) { x = x * 10u + y; return x; }
void main() {
    uint q = 3u;
    uint r0 = two(q, (q = 7u));
    uint arr[2] = uint[2](0u, 3u);
    uint i = uint(gl_FragCoord.x) & 1u | 1u;
    uint r1 = two(arr[i], (arr[1] = 7u));
    uint c = 2u;
    uint r2 = two(c, two(c, 5u));
    o = vec4(float(r0), float(r1), float(r2), float(c));
}`
		fm, ferrs, fok := compile_glsl(fsrc)
		defer destroy(fm)
		testing.expectf(t, fok, "inout.frag failed:\n%s", format_errors(ferrs, context.temp_allocator))
		if fok {
			px := make([][4]f32, 16, context.temp_allocator)
			run_f32(&fm.program, nil, 4, 4, px, temp_scratch())
			testing.expect_value(t, px[0], [4]f32{37, 37, 45, 45})
		}
	}

	// GLSL ES 3.10 §4.4.1.1: multiple local_size_* declarations must specify the
	// exact same set of axes with the exact same values.
	{
		for tc in ([][2]string {
			{"layout(local_size_x = 8) in;\nlayout(local_size_y = 4) in;\nvoid main() {}", "without local_size_x"},
			{"layout(local_size_x = 8) in;\nlayout(local_size_x = 8, local_size_y = 4) in;\nvoid main() {}", "with local_size_y"},
			{"layout(local_size_x = 8, local_size_y = 4) in;\nlayout(local_size_x = 8) in;\nvoid main() {}", "without local_size_y"},
			{"layout(local_size_x = 8, local_size_y = 4) in;\nlayout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;\nvoid main() {}", "with local_size_z"},
		}) {
			rm, rerrs, rok := compile(fmt.tprintf("#version 310 es\n%s", tc[0]), {profile = .Glsl, stage = .Compute})
			defer destroy(rm)
			testing.expect(t, !rok && len(rerrs) > 0 && rerrs[0].code == .Local_Size_Redeclared)
			if len(rerrs) > 0 {
				msg := format_errors(rerrs, context.temp_allocator)
				testing.expectf(t, strings.contains(msg, tc[1]), "expected %q in %q", tc[1], msg)
			}
		}
		expect_code(t, "layout(local_size_x = 8) in;\nlayout(local_size_x = 16) in;\nvoid main() {}", .Local_Size_Conflict, {profile = .Glsl, stage = .Compute})
		expect_code(t, "layout(local_size_x = 8, local_size_y = 4) in;\nlayout(local_size_x = 8, local_size_y = 2) in;\nvoid main() {}", .Local_Size_Conflict, {profile = .Glsl, stage = .Compute})
		expect_code(t, "layout(local_size_x = 8) in;\nlayout(local_size_y = 2, local_size_x = 4) in;\nvoid main() {}", .Local_Size_Conflict, {profile = .Glsl, stage = .Compute})
		expect_code(t, "layout(local_size_x = 8, local_size_x = 4) in;\nvoid main() {}", .Local_Size_Conflict, {profile = .Glsl, stage = .Compute})
		// Non-literal expressions and invalid/out-of-range literals rejected in local_size_* (§4.4.1.1 / §4.1.3), while `8u` and `0x8` succeed.
		for bad_ls in ([]string {
			"const int N = 4;\nlayout(local_size_x = N) in;\nvoid main() {}",
			"layout(local_size_x = 4 * 2) in;\nvoid main() {}",
			"layout(local_size_x = (8)) in;\nvoid main() {}",
			"layout(local_size_x = ~0u) in;\nvoid main() {}",
		}) {
			lm, lerrs, lok := compile(fmt.tprintf("#version 310 es\n%s", bad_ls), {profile = .Glsl, stage = .Compute})
			defer destroy(lm)
			testing.expect(t, !lok && len(lerrs) > 0 && lerrs[0].code == .Local_Size_Constant)
		}
		for bad_lit in ([]string {
			"layout(local_size_x = 4294967304) in;\nvoid main() {}",
			"layout(local_size_x = 4294967296u) in;\nvoid main() {}",
			"layout(local_size_x = 0x100000008) in;\nvoid main() {}",
			"layout(local_size_x = 08) in;\nvoid main() {}",
			"layout(local_size_x = 0x) in;\nvoid main() {}",
		}) {
			lm, lerrs, lok := compile(fmt.tprintf("#version 310 es\n%s", bad_lit), {profile = .Glsl, stage = .Compute})
			defer destroy(lm)
			testing.expectf(t, !lok && len(lerrs) > 0 && lerrs[0].code == .Invalid_Int_Literal, "expected Invalid_Int_Literal on %q, got %v", bad_lit, lerrs)
		}
		// Offending axis positions on cross-declaration errors, and no cascading redeclaration error after a bad value.
		{
			pm1, perrs1, pok1 := compile("#version 310 es\nlayout(local_size_x = 8, local_size_y = 4) in;\nlayout(local_size_x = 8, local_size_y = 2) in;\nvoid main() {}", {profile = .Glsl, stage = .Compute})
			defer destroy(pm1)
			pmsg1 := format_errors(perrs1, context.temp_allocator)
			testing.expectf(t, !pok1 && strings.contains(pmsg1, "3:26: local_size_y is already declared as 4 (got 2)"), "got %q", pmsg1)

			pm2, perrs2, pok2 := compile("#version 310 es\nlayout(local_size_x = 8, local_size_y = 4) in;\nlayout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;\nvoid main() {}", {profile = .Glsl, stage = .Compute})
			defer destroy(pm2)
			pmsg2 := format_errors(perrs2, context.temp_allocator)
			testing.expectf(t, !pok2 && strings.contains(pmsg2, "3:44: workgroup size redeclared with local_size_z"), "got %q", pmsg2)

			dm1, derrs1, dok1 := compile("#version 310 es\nlayout(local_size_x = 8, local_size_y = 4) in;\nlayout(local_size_x = 8, local_size_y = 0) in;\nvoid main() { uint k = gl_WorkGroupSize.x; }", {profile = .Glsl, stage = .Compute})
			defer destroy(dm1)
			testing.expectf(t, !dok1 && len(derrs1) == 1 && derrs1[0].code == .Local_Size_Positive, "expected 1 Local_Size_Positive error, got %d: %s", len(derrs1), format_errors(derrs1, context.temp_allocator))

			dm2, derrs2, dok2 := compile("#version 310 es\nlayout(local_size_x = 2000) in;\nlayout(local_size_x = 8) in;\nvoid main() { uint k = gl_WorkGroupSize.x; }", {profile = .Glsl, stage = .Compute})
			defer destroy(dm2)
			testing.expectf(t, !dok2 && len(derrs2) == 1 && derrs2[0].code == .Local_Size_Max, "expected 1 Local_Size_Max error, got %d: %s", len(derrs2), format_errors(derrs2, context.temp_allocator))
		}
		{
			sm, serrs, sok := compile("#version 310 es\nlayout(local_size_x = 8u, local_size_y = 0x2) in;\nlayout(local_size_y = 2, local_size_x = 8) in;\nvoid main() {}", {profile = .Glsl, stage = .Compute})
			defer destroy(sm)
			testing.expectf(t, sok, "identical repeat local_size failed:\n%s", format_errors(serrs, context.temp_allocator))
			if sok {
				testing.expect_value(t, sm.threadcount, [3]u32{8, 2, 1})
			}
		}
	}

	// Default block layout qualifiers: accept only packing and matrix-order qualifiers, and require commas in layout(...).
	{
		dm, derrs, dok := compile(
			"#version 310 es\nlayout(std140) buffer;\nlayout(std430) buffer;\nlayout(column_major) buffer;\nlayout(row_major) uniform;\nlayout(std140) uniform;\nlayout(local_size_x = 8) in;\nbuffer B { vec3 w[]; } b;\nvoid main() { b.w[0] = vec3(1.0); }",
			{profile = .Glsl, stage = .Compute},
		)
		defer destroy(dm)
		testing.expectf(t, dok, "default buffer layout failed:\n%s", format_errors(derrs, context.temp_allocator))
		dfm, dferrs, dfok := compile(
			"#version 310 es\nprecision highp float;\nlayout(std430) buffer;\nlayout(std430) readonly buffer B { vec4 w[]; } b;\nout vec4 o;\nvoid main() { o = b.w[0]; }",
			{profile = .Glsl, stage = .Fragment},
		)
		defer destroy(dfm)
		testing.expectf(t, dfok, "fragment default buffer layout failed:\n%s", format_errors(dferrs, context.temp_allocator))
		expect_code(t, "layout(std140) buffer;\nlayout(local_size_x = 8) in;\nbuffer B { vec3 w[]; } b;\nvoid main() { b.w[0] = vec3(1.0); }", .Buffer_Layout, {profile = .Glsl, stage = .Compute})
		for bad_dl in ([]string {
			"layout(std430, binding = 2) buffer;",
			"layout(std430, set = 1) buffer;",
			"layout(std430) coherent buffer;",
			"layout(std430) volatile buffer;",
			"layout(std430) restrict buffer;",
			"layout(std430) readonly buffer;",
			"layout(std430) writeonly buffer;",
			"layout() buffer;",
			"layout(std430,) buffer;",
			"layout(,std430) buffer;",
			"layout(std430) highp buffer;",
			"layout(std430) const buffer;",
			"layout(scalar) buffer;",
			"layout(rgba8) buffer;",
			"layout(location = 2) buffer;",
			"layout(local_size_y = 2) buffer;",
			"layout(foo) buffer;",
			"layout(binding = 2) uniform;",
			"layout(std430) uniform;",
			"layout() uniform;",
			"layout(std430 row_major) buffer;",
			"layout(std430 set = 1 binding = 0) buffer B2 { uint w[]; } b2;",
		}) {
			bm, _, bok := compile(fmt.tprintf("#version 310 es\nlayout(local_size_x = 1) in;\n%s\nlayout(std430) buffer B {{ uint w[]; }} b;\nvoid main() {{ b.w[0] = 1u; }}", bad_dl), {profile = .Glsl, stage = .Compute})
			defer destroy(bm)
			testing.expectf(t, !bok, "expected invalid default layout %q to fail", bad_dl)
		}
		{
			ncm, ncerrs, ncok := compile("#version 310 es\nlayout(local_size_x = 8 local_size_y = 4) in;\nvoid main() {}", {profile = .Glsl, stage = .Compute})
			defer destroy(ncm)
			ncmsg := format_errors(ncerrs, context.temp_allocator)
			testing.expectf(t, !ncok && len(ncerrs) > 0 && ncerrs[0].code == .Expected && strings.contains(ncmsg, "expected ','"), "got %q", ncmsg)
		}
		for bad_frag_dl in ([]string {
			"layout(std430) flat buffer;",
			"layout(std430) in buffer;",
			"layout(std430) out buffer;",
		}) {
			bfm, _, bfok := compile(fmt.tprintf("#version 310 es\nprecision highp float;\n%s\nout vec4 o;\nvoid main() {{ o = vec4(1.0); }}", bad_frag_dl), {profile = .Glsl, stage = .Fragment})
			defer destroy(bfm)
			testing.expectf(t, !bfok, "expected invalid fragment default layout %q to fail", bad_frag_dl)
		}
	}

	// Item 10c: imageStore rejects mismatched texel base types (`ivec4`/`uvec4` into float image, etc.).
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(r32f) uniform image2D im;\nvoid main() { imageStore(im, ivec2(0), ivec4(7)); }", .No_Overload, {profile = .Glsl, stage = .Compute})
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(r32f) uniform image2D im;\nvoid main() { imageStore(im, ivec2(0), uvec4(7u)); }", .No_Overload, {profile = .Glsl, stage = .Compute})
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(r32ui) uniform uimage2D im;\nvoid main() { imageStore(im, ivec2(0), ivec4(7)); }", .No_Overload, {profile = .Glsl, stage = .Compute})
	expect_code(t, "layout(local_size_x = 8) in;\nlayout(r32i) uniform iimage2D im;\nvoid main() { imageStore(im, ivec2(0), vec4(7.0)); }", .No_Overload, {profile = .Glsl, stage = .Compute})

	// Non-std430 helper struct alongside a storage buffer must not divide by zero in std430_field_offset.
	{
		helper_src := `
#version 310 es
layout(local_size_x = 1) in;
struct LocalOnly {
    mat2 m;
    bool flag;
    float arr[2];
};
struct BufElem {
    vec3 a;
    uint b;
};
layout(std430) buffer B { BufElem items[]; } b;
void main() {
    LocalOnly loc = LocalOnly(mat2(1.0), true, float[2](2.0, 3.0));
    if (loc.flag) {
        b.items[0] = BufElem(vec3(loc.arr[0], loc.arr[1], loc.m[0][0]), 7u);
    }
}`
		hm, herrs, hok := compile(helper_src, {profile = .Glsl, stage = .Compute})
		defer destroy(hm)
		testing.expectf(t, hok, "helper struct shader failed:\n%s", format_errors(herrs, context.temp_allocator))
		if hok {
			testing.expect_value(t, len(hm.structs), 2)
			testing.expect_value(t, hm.structs[0].fields[0].offset, 0)
			testing.expect_value(t, hm.structs[0].fields[1].offset, 0)
			testing.expect_value(t, hm.structs[0].fields[2].offset, 0)
			testing.expect_value(t, hm.structs[1].fields[0].offset, 0)
			testing.expect_value(t, hm.structs[1].fields[1].offset, 12)
		}
	}

	// Single-struct buffer shader (0-based Type.strukt == 0) populates Module.structs[0].
	{
		psrc, perr := os.read_entire_file("testdata/particles.comp", context.temp_allocator)
		testing.expect(t, perr == nil)
		if perr == nil {
			pm, _, pok := compile(string(psrc), {profile = .Glsl, stage = .Compute})
			defer destroy(pm)
			testing.expect(t, pok && len(pm.structs) == 1 && len(pm.structs[0].fields) == 4)
			if pok && len(pm.structs) == 1 && len(pm.structs[0].fields) == 4 {
				testing.expect_value(t, pm.structs[0].fields[0].offset, 0)
				testing.expect_value(t, pm.structs[0].fields[1].offset, 12)
				testing.expect_value(t, pm.structs[0].fields[2].offset, 16)
				testing.expect_value(t, pm.structs[0].fields[3].offset, 28)
			}
		}
	}
}

@(test)
test_compute_ensure :: proc(t: ^testing.T) {
	// `ensure` in run_compute stays active even when built with `-disable-assert`.
	m, _, ok := compile(
		"#version 310 es\nlayout(local_size_x = 4) in;\nlayout(rgba8) writeonly uniform image2D dst;\nvoid main() { imageStore(dst, ivec2(0), vec4(1.0)); }",
		{profile = .Glsl, stage = .Compute},
	)
	testing.cleanup(t, proc(p: rawptr) { destroy((^Module)(p)) }, m)
	testing.expect(t, ok)
	if !ok {
		return
	}
	tiny := make([]u8, 4, context.temp_allocator)
	testing.expect_assert_message(t, "oksl: Storage_Image.data is too small for width, height and stride")
	run_compute(&m.program, nil, {1, 1, 1}, temp_scratch(), {
		images = {{data = tiny, width = 8, height = 8, format = .Rgba8}},
	})
}

@(test)
test_compute_round3_fixes :: proc(t: ^testing.T) {
	has_member_decorations :: proc(spv: []u32) -> (col_major, mat_stride: bool) {
		i := 5
		for i < len(spv) {
			wc := int(spv[i] >> 16)
			op := Op(spv[i] & 0xffff)
			if wc <= 0 || i + wc > len(spv) {
				break
			}
			if op == .Member_Decorate && wc >= 4 {
				if spv[i + 3] == DEC_COL_MAJOR {
					col_major = true
				} else if spv[i + 3] == DEC_MATRIX_STRIDE && wc >= 5 && spv[i + 4] == 16 {
					mat_stride = true
				}
			}
			i += wc
		}
		return
	}

	// 1. Uniform matrix arrays emit ColMajor and MatrixStride in SPIR-V across all stages.
	for tc in ([]struct {
		name:  string,
		stage: Stage,
		src:   string,
	} {
		{"um_comp", .Compute, "#version 310 es\nlayout(local_size_x = 1) in;\nbuffer B { vec4 w[]; } b;\nuniform mat4 bones[4];\nvoid main() { b.w[0] = bones[1] * vec4(1.0) + bones[2][3]; }"},
		{"um_frag", .Fragment, "#version 300 es\nprecision highp float;\nuniform mat2 um[2];\nuniform float k;\nout vec4 o;\nvoid main() { int n = int(k); o = vec4(um[n][n], um[1][1]); }"},
		{"um_vert", .Vertex, "#version 300 es\nlayout(location = 0) in vec4 pos;\nlayout(location = 1) in vec4 w;\nuniform mat4 bones[4];\nvoid main() { gl_Position = bones[int(w.x)] * pos + bones[1][2]; }"},
	}) {
		um, uerrs, uok := compile(tc.src, {profile = .Glsl, stage = tc.stage})
		defer destroy(um)
		testing.expectf(t, uok, "%s failed:\n%s", tc.name, format_errors(uerrs, context.temp_allocator))
		if uok {
			cm, ms := has_member_decorations(um.spirv)
			testing.expectf(t, cm && ms, "%s missing ColMajor (%v) or MatrixStride (%v)", tc.name, cm, ms)
			if have_spirv_val() {
				expect_valid_spirv(t, tc.name, um.spirv)
			}
		}
	}

	// 3. `?:` with void arms (imageStore in compute, and void user functions in fragment).
	{
		vt_comp := `#version 310 es
layout(local_size_x = 1) in;
layout(std430) buffer B { uint w[]; } b;
layout(r32ui) uniform highp uimage2D im;
void main() {
    uint k = 1u;
    k > 0u ? imageStore(im, ivec2(0, 1), uvec4(5u)) : imageStore(im, ivec2(0, 1), uvec4(6u));
    b.w[0] = imageLoad(im, ivec2(0, 1)).x;
}`
		vm, verrs, vok := compile(vt_comp, {profile = .Glsl, stage = .Compute})
		defer destroy(vm)
		testing.expectf(t, vok, "vt_comp failed:\n%s", format_errors(verrs, context.temp_allocator))
		if vok {
			void_id: u32
			has_void_phi := false
			for i := 5; i < len(vm.spirv); {
				wc := int(vm.spirv[i] >> 16)
				op := Op(vm.spirv[i] & 0xffff)
				if wc <= 0 || i + wc > len(vm.spirv) {
					break
				}
				if op == .Type_Void && wc >= 2 {
					void_id = vm.spirv[i + 1]
				} else if op == .Phi && wc >= 3 && vm.spirv[i + 1] == void_id {
					has_void_phi = true
				}
				i += wc
			}
			testing.expect(t, !has_void_phi)
			if have_spirv_val() {
				expect_valid_spirv(t, "vt_comp", vm.spirv)
			}
			bw := make([]u32, 4, context.temp_allocator)
			imw := make([]u32, 16, context.temp_allocator)
			run_compute(&vm.program, nil, {1, 1, 1}, temp_scratch(), {
				buffers = {bw},
				images  = {{data = slice.to_bytes(imw), width = 4, height = 4, format = .R32ui}},
			})
			testing.expect_value(t, bw[0], u32(5))
		}

		vt_frag := `#version 300 es
precision highp float;
out vec4 o;
float g;
void f() { g = 1.0; }
void h() { g = 2.0; }
void main() { g = 0.0; gl_FragCoord.x > 2.0 ? f() : h(); o = vec4(g); }`
		fm, ferrs, fok := compile_glsl(vt_frag)
		defer destroy(fm)
		testing.expectf(t, fok, "vt_frag failed:\n%s", format_errors(ferrs, context.temp_allocator))
		if fok {
			if have_spirv_val() {
				expect_valid_spirv(t, "vt_frag", fm.spirv)
			}
			px := make([][4]f32, 16, context.temp_allocator)
			run_f32(&fm.program, nil, 4, 4, px, temp_scratch())
			testing.expect_value(t, px[0][0], f32(2.0)) // x = 0.5 <= 2.0 -> h()
			testing.expect_value(t, px[1][0], f32(2.0)) // x = 1.5 <= 2.0 -> h()
			testing.expect_value(t, px[2][0], f32(1.0)) // x = 2.5 >  2.0 -> f()
			testing.expect_value(t, px[3][0], f32(1.0)) // x = 3.5 >  2.0 -> f()
		}
	}

	// 4. testdata/image_war.comp compiles cleanly, fences before and after write, and matches accumulator.
	{
		wsrc, werr := os.read_entire_file("testdata/image_war.comp", context.temp_allocator)
		testing.expect(t, werr == nil)
		if werr == nil {
			wm, werrs, wok := compile(string(wsrc), {profile = .Glsl, stage = .Compute})
			defer destroy(wm)
			testing.expectf(t, wok, "image_war.comp failed:\n%s", format_errors(werrs, context.temp_allocator))
			if wok {
				if have_spirv_val() {
					expect_valid_spirv(t, "image_war.comp", wm.spirv)
				}
				testing.expect(t, strings.contains(wm.msl, "(oksl_fence(ok_ui_img), ok_ui_img.write(_t"))
				testing.expect(t, strings.contains(wm.msl, "), oksl_fence(ok_ui_img))"))
				imw := make([]u32, 16 * 16, context.temp_allocator)
				bw := make([]u32, 16 * 16, context.temp_allocator)
				run_compute(&wm.program, nil, {2, 2, 1}, temp_scratch(), {
					buffers = {bw},
					images  = {{data = slice.to_bytes(imw), width = 16, height = 16, format = .R32ui}},
				})
				want_acc := u32(0)
				for k in u32(0) ..< 16 {
					want_acc = want_acc * 31 + (k * 7 + 1)
				}
				testing.expect_value(t, bw[0], want_acc)
				testing.expect_value(t, bw[255], want_acc)
				testing.expect_value(t, imw[255], u32(15 * 7 + 1))
			}
		}
	}

	// 7. Inline read-modify-write `imageStore(tex, p, imageLoad(tex, p) + ...)` with NO `may_write`
	// argument hoists `tex.read(...)` into a temporary BEFORE `(oksl_fence(tex), tex.write(_t...))`.
	{
		inc_src := `#version 310 es
layout(local_size_x = 1) in;
layout(r32f) uniform highp image2D im;
void incf(highp image2D tex, ivec2 p) {
    imageStore(tex, p, imageLoad(tex, p) + vec4(1.0));
}
void main() {
    incf(im, ivec2(0));
}`
		im_m, im_errs, im_ok := compile(inc_src, {profile = .Glsl, stage = .Compute})
		defer destroy(im_m)
		testing.expectf(t, im_ok, "incf failed:\n%s", format_errors(im_errs, context.temp_allocator))
		if im_ok {
			fn_pos := strings.index(im_m.msl, "void ok_incf(")
			testing.expect(t, fn_pos >= 0)
			if fn_pos >= 0 {
				rest := im_m.msl[fn_pos:]
				end := strings.index(rest, "\n}\n")
				body := rest[:end] if end >= 0 else rest
				read_pos := strings.index(body, "okp_tex_img.read(")
				fence_write_pos := strings.index(body, "(oksl_fence(okp_tex_img), okp_tex_img.write(_t")
				testing.expectf(t, read_pos >= 0 && fence_write_pos > read_pos, "expected read hoisted before pre-write fence in:\n%s", body)
			}
		}
	}
}

@(test)
test_int32_const_fold_and_literals :: proc(t: ^testing.T) {
	run_words :: proc(t: ^testing.T, src: string, n_words: int, loc := #caller_location) -> ([]u32, ^Module) {
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Compute})
		testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator), loc = loc)
		if !ok {
			destroy(m)
			return nil, nil
		}
		if have_spirv_val() {
			expect_valid_spirv(t, "int32_cf", m.spirv, loc = loc)
		}
		words := make([]u32, n_words, context.temp_allocator)
		run_compute(&m.program, nil, {1, 1, 1}, temp_scratch(), {buffers = {words}})
		return words, m
	}

	// 1. cf_index.comp: unsigned 32-bit folded array indices and const variables.
	{
		src := `#version 310 es
layout(local_size_x = 1) in;
layout(std430) buffer B { uint w[]; } b;
const uint K = (0u - 1u) % 4u;
void main() {
    uint arr[4] = uint[4](10u, 11u, 12u, 13u);
    b.w[0] = arr[(~10u) % 4u];
    b.w[1] = arr[K];
}`
		w, m := run_words(t, src, 4)
		defer destroy(m)
		if w != nil {
			testing.expect_value(t, w[0], u32(11))
			testing.expect_value(t, w[1], u32(13))
		}
	}

	// 2. cf_switch.comp: uint switch labels folded in 32-bit unsigned (matches glslang on RADV: 10, 20, 30, 40, 99).
	{
		src := `#version 310 es
layout(local_size_x = 1) in;
layout(std430) buffer B { uint w[]; } b;
uint pick(uint x) {
    switch (x) {
    case (~10u) % 4u: return 10u;
    case (0u - 1u) / 2u: return 20u;
    case (~0u) >> 4: return 30u;
    case (2147483648u + 2147483648u) + 5u: return 40u;
    default: return 99u;
    }
}
void main() {
    b.w[0] = pick(1u);
    b.w[1] = pick(0x7fffffffu);
    b.w[2] = pick(0x0fffffffu);
    b.w[3] = pick(5u);
    b.w[4] = pick(0xfffffffdu);
}`
		w, m := run_words(t, src, 8)
		defer destroy(m)
		if w != nil {
			testing.expect_value(t, w[0], u32(10))
			testing.expect_value(t, w[1], u32(20))
			testing.expect_value(t, w[2], u32(30))
			testing.expect_value(t, w[3], u32(40))
			testing.expect_value(t, w[4], u32(99))
			testing.expect(t, strings.contains(m.msl, "case 1u:"))
			testing.expect(t, strings.contains(m.msl, "case 2147483647u:"))
			testing.expect(t, strings.contains(m.msl, "case 268435455u:"))
			testing.expect(t, strings.contains(m.msl, "case 5u:"))
		}
	}

	// 3. cf_dup.comp and duplicate case labels after 32-bit int/uint folding.
	for tc in ([][2]string {
		{
			"layout(local_size_x = 1) in;\nlayout(std430) buffer B { int w[]; } b;\nvoid main() {\n    switch (b.w[0]) {\n    case -2147483647 - 1: b.w[1] = 1; break;\n    case 2147483647 + 1: b.w[1] = 2; break;\n    }\n}",
			"case -2147483648 is already handled",
		},
		{
			"layout(local_size_x = 1) in;\nvoid main() {\n    switch (0) {\n    case 2147483648: break;\n    case 2147483647 + 1: break;\n    }\n}",
			"case -2147483648 is already handled",
		},
		{
			"layout(local_size_x = 1) in;\nvoid main() {\n    switch (0u) {\n    case (2147483648u + 2147483648u) + 5u: break;\n    case 5u: break;\n    }\n}",
			"case 5 is already handled",
		},
		{
			"layout(local_size_x = 1) in;\nvoid main() {\n    switch (0) {\n    case -1: break;\n    case 0xffffffff: break;\n    }\n}",
			"case -1 is already handled",
		},
		{
			"layout(local_size_x = 1) in;\nvoid main() {\n    switch (0u) {\n    case 0u - 1u: break;\n    case ~0u: break;\n    }\n}",
			"case 4294967295 is already handled",
		},
	}) {
		m, errs, ok := compile(fmt.tprintf("#version 310 es\n%s", tc[0]), {profile = .Glsl, stage = .Compute})
		defer destroy(m)
		msg := format_errors(errs, context.temp_allocator)
		testing.expectf(t, !ok && len(errs) == 1 && errs[0].code == .Case_Duplicate && strings.contains(msg, tc[1]), "expected %q, got:\n%s", tc[1], msg)
	}

	// 3b. GLSL ES 3.10 §6.2: case label type must match switch selector type (no int/uint mixing).
	for tc in ([][2]string {
		{
			"layout(local_size_x = 1) in;\nlayout(std430) buffer B { uint w[]; } b;\nvoid main() { switch (b.w[0]) { case -2: b.w[1] = 1u; break; default: break; } }",
			"case label must be uint, not int",
		},
		{
			"layout(local_size_x = 1) in;\nlayout(std430) buffer B { uint w[]; } b;\nvoid main() { switch (int(b.w[0])) { case 0xffffffffu: b.w[1] = 1u; break; default: break; } }",
			"case label must be int, not uint",
		},
		{
			"layout(local_size_x = 1) in;\nconst uint K = 1u;\nvoid main() { switch (0) { case K: break; } }",
			"case label must be int, not uint",
		},
		{
			"layout(local_size_x = 1) in;\nconst int K = 1;\nvoid main() { switch (0u) { case K: break; } }",
			"case label must be uint, not int",
		},
	}) {
		m, errs, ok := compile(fmt.tprintf("#version 310 es\n%s", tc[0]), {profile = .Glsl, stage = .Compute})
		defer destroy(m)
		msg := format_errors(errs, context.temp_allocator)
		testing.expectf(t, !ok && len(errs) == 1 && errs[0].code == .Case_Type && strings.contains(msg, tc[1]), "expected %q, got:\n%s", tc[1], msg)
	}
	{
		m, errs, ok := compile("#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { switch (1.0) { case 1: break; } }", {profile = .Glsl, stage = .Compute})
		defer destroy(m)
		testing.expect(t, !ok && len(errs) == 1 && errs[0].code == .Switch_Type)
	}

	// 4. 32-bit wrap-around and signedness across ~, -, +, *, /, %, >>, <<, comparisons, bools and ?:.
	{
		src := `#version 310 es
layout(local_size_x = 1) in;
layout(std430) buffer B { uint w[]; } b;
const bool U_CMP = (~0u > 1u) && !(1u > ~0u) && (1u < ~0u) && !(~0u < 1u) && (~0u >= 1u) && !(1u >= ~0u) && (1u <= ~0u) && !(~0u <= 1u);
const bool S_CMP = ((2147483647 + 1) < 0) && (-1 < 1) && !(1 < -1) && (1 > -1) && !(-1 > 1) && (-1 <= 1) && !(1 <= -1) && (1 >= -1) && !(-1 >= 1) && ((~0) == -1) && ((2147483648) == (-2147483647 - 1));
const bool T_BOOL_NORM = bool(5u);
uint fold_u(uint x) {
    switch (x) {
    case ~10u: return 1u;                              // 0xfffffff5u (~)
    case -10u: return 2u;                              // 0xfffffff6u (unary -)
    case 0u - 3u: return 3u;                           // 0xfffffffdu (binary -)
    case 0x80000001u * 2u: return 4u;                  // 2u (*)
    case 0xfffffffeu / 3u: return 5u;                  // 0x55555554u (/)
    case 0xfffffffeu % 5u: return 6u;                  // 4u (%)
    case 0x80000000u >> 28: return 7u;                 // 8u (>> logical)
    case 0xf0000001u << 4: return 8u;                  // 16u (<< wrap)
    case U_CMP ? 100u : 200u: return 9u;               // unsigned comparisons (<, <=, >, >=)
    default: return 0u;
    }
}
int fold_i(int x) {
    switch (x) {
    case ~10: return 11;                               // -11 (~)
    case -(-2147483647 - 1): return 12;                // -2147483648 (unary - wrap)
    case (-2147483647 - 1) - 1: return 13;             // 2147483647 (binary - wrap)
    case 0x40000001 * 4: return 14;                    // 4 (* wrap)
    case (-2147483647 - 1) / -1 + 1: return 15;        // -2147483647 (INT_MIN / -1 wrap)
    case (-2147483647 - 1) % -1: return 16;            // 0 (INT_MIN % -1)
    case (-32) >> 3: return 17;                        // -4 (>> arithmetic)
    case 0x40000003 << 2: return 18;                   // 12 (<< wrap)
    case S_CMP ? 100 : 200: return 19;                 // signed comparisons (<, <=, >, >=)
    default: return 0;
    }
}
void main() {
    b.w[0] = fold_u(0xfffffff5u);
    b.w[1] = fold_u(0xfffffff6u);
    b.w[2] = fold_u(0xfffffffdu);
    b.w[3] = fold_u(2u);
    b.w[4] = fold_u(0x55555554u);
    b.w[5] = fold_u(4u);
    b.w[6] = fold_u(8u);
    b.w[7] = fold_u(16u);
    b.w[8] = fold_u(100u);
    b.w[9] = uint(fold_i(-11));
    b.w[10] = uint(fold_i(2147483648));
    b.w[11] = uint(fold_i(2147483647));
    b.w[12] = uint(fold_i(4));
    b.w[13] = uint(fold_i(-2147483647));
    b.w[14] = uint(fold_i(0));
    b.w[15] = uint(fold_i(-4));
    b.w[16] = uint(fold_i(12));
    b.w[17] = uint(fold_i(100));
    uint a_bool[int(bool(2)) + 1] = uint[2](7u, 9u);
    uint b_bool[uint(T_BOOL_NORM) + 1u] = uint[2](3u, 4u);
    b.w[18] = a_bool[1] * 10u + b_bool[1];
    uint a_mindiv[(((-2147483647 - 1) / -1) == 2147483648 && ((-2147483647 - 1) % -1) == 0 && (7 % -1) == 0) ? 2 : 1] = uint[2](5u, 6u);
    b.w[19] = a_mindiv[1];
}`
		w, m := run_words(t, src, 24)
		defer destroy(m)
		if w != nil {
			want := [?]u32{1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 94, 6}
			for v, i in want {
				testing.expect_value(t, w[i], v)
			}
			testing.expect(t, strings.contains(m.msl, "case (-2147483647 - 1):"))
		}
	}

	// 4b. Array sizes above max(i32) report Array_Size_Positive.
	for big_arr in ([]string {
		"layout(local_size_x = 1) in;\nvoid main() { uint a[0x80000000u]; }",
		"layout(local_size_x = 1) in;\nvoid main() { uint a[0xffffffffu]; }",
	}) {
		m, errs, ok := compile(fmt.tprintf("#version 310 es\n%s", big_arr), {profile = .Glsl, stage = .Compute})
		defer destroy(m)
		testing.expectf(t, !ok && len(errs) == 1 && errs[0].code == .Array_Size_Positive, "expected Array_Size_Positive on %q, got %v", big_arr, errs)
	}

	// 5. Undefined shifts (< 0 or >= 32) do not fold as constant expressions.
	for bad_sh in ([]string {
		"layout(local_size_x = 1) in;\nvoid main() { switch (0) { case 1 << 32: break; } }",
		"layout(local_size_x = 1) in;\nvoid main() { switch (0u) { case 1u >> 32u: break; } }",
		"layout(local_size_x = 1) in;\nvoid main() { switch (0) { case 1 << -1: break; } }",
		"layout(local_size_x = 1) in;\nvoid main() { switch (0) { case 1 >> -1: break; } }",
	}) {
		m, errs, ok := compile(fmt.tprintf("#version 310 es\n%s", bad_sh), {profile = .Glsl, stage = .Compute})
		defer destroy(m)
		testing.expectf(t, !ok && len(errs) > 0 && errs[0].code == .Case_Constant, "expected Case_Constant on %q, got %v", bad_sh, errs)
	}

	// 6. Integer literal validation across parse_primary, location, local_size, #if and #line,
	// plus GLSL ES 3.10 §4.1.3 handling of `2147483648` without `u` (signed -2147483648).
	for bad_lit in ([]string {
		// parse_primary
		"#version 310 es\nlayout(local_size_x = 1) in;\nlayout(std430) buffer B { uint w[]; } b;\nvoid main() { b.w[0] = 4294967304u; }",
		"#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { int x = 4294967296; }",
		"#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { uint x = 0x100000000u; }",
		"#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { int x = 040000000000; }",
		"#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { int x = 08; }",
		"#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { uint x = 09u; }",
		"#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { int x = 0x; }",
		"#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { uint x = 0xu; }",
		// layout(location = ...)
		"#version 310 es\nprecision highp float;\nlayout(location = 4294967296) out vec4 c;\nvoid main() { c = vec4(1.0); }",
		"#version 310 es\nprecision highp float;\nlayout(location = 4294967296u) out vec4 c;\nvoid main() { c = vec4(1.0); }",
		"#version 310 es\nprecision highp float;\nlayout(location = 08) out vec4 c;\nvoid main() { c = vec4(1.0); }",
		"#version 310 es\nprecision highp float;\nlayout(location = 0x) out vec4 c;\nvoid main() { c = vec4(1.0); }",
		// #if / #elif
		"#version 310 es\n#if 4294967296\n#endif\nlayout(local_size_x = 1) in;\nvoid main() {}",
		"#version 310 es\n#if 08\n#endif\nlayout(local_size_x = 1) in;\nvoid main() {}",
		"#version 310 es\n#if 0x\n#endif\nlayout(local_size_x = 1) in;\nvoid main() {}",
		"#version 310 es\n#if 0\n#elif 08\n#endif\nlayout(local_size_x = 1) in;\nvoid main() {}",
		// #line
		"#version 310 es\n#line 4294967296\nlayout(local_size_x = 1) in;\nvoid main() {}",
		"#version 310 es\n#line 4294967296u\nlayout(local_size_x = 1) in;\nvoid main() {}",
		"#version 310 es\n#line 08\nlayout(local_size_x = 1) in;\nvoid main() {}",
		"#version 310 es\n#line 0x\nlayout(local_size_x = 1) in;\nvoid main() {}",
		"#version 310 es\n#line 10 08\nlayout(local_size_x = 1) in;\nvoid main() {}",
	}) {
		stage := Stage.Fragment if strings.contains(bad_lit, "out vec4") else Stage.Compute
		m, errs, ok := compile(bad_lit, {profile = .Glsl, stage = stage})
		defer destroy(m)
		testing.expectf(t, !ok && len(errs) > 0 && errs[0].code == .Invalid_Int_Literal, "expected Invalid_Int_Literal on %q, got %v", bad_lit, errs)
	}

	// `2147483648` without `u` is signed -2147483648 (§4.1.3), while `2147483648u` is +2147483648.
	{
		m1, e1, ok1 := compile("#version 310 es\nlayout(local_size_x = 2147483648) in;\nvoid main() {}", {profile = .Glsl, stage = .Compute})
		defer destroy(m1)
		testing.expect(t, !ok1 && len(e1) == 1 && e1[0].code == .Local_Size_Positive && e1[0].n == -2147483648)

		m2, e2, ok2 := compile("#version 310 es\nlayout(local_size_x = 2147483648u) in;\nvoid main() {}", {profile = .Glsl, stage = .Compute})
		defer destroy(m2)
		testing.expect(t, !ok2 && len(e2) == 1 && e2[0].code == .Local_Size_Max && e2[0].n == 2147483648)

		m3, e3, ok3 := compile("#version 310 es\nprecision highp float;\nlayout(location = 0xffffffff) out vec4 c;\nvoid main() { c = vec4(1.0); }", {profile = .Glsl, stage = .Fragment})
		defer destroy(m3)
		testing.expect(t, !ok3 && len(e3) == 1 && e3[0].code == .Location_Range && e3[0].n == -1)

		m4, e4, ok4 := compile("#version 310 es\nprecision highp float;\nlayout(location = 2147483648) out vec4 c;\nvoid main() { c = vec4(1.0); }", {profile = .Glsl, stage = .Fragment})
		defer destroy(m4)
		testing.expect(t, !ok4 && len(e4) == 1 && e4[0].code == .Location_Range && e4[0].n == -2147483648)

		m5, e5, ok5 := compile("#version 310 es\nprecision highp float;\nlayout(location = 2147483648u) out vec4 c;\nvoid main() { c = vec4(1.0); }", {profile = .Glsl, stage = .Fragment})
		defer destroy(m5)
		testing.expect(t, !ok5 && len(e5) == 1 && e5[0].code == .Location_Range && e5[0].n == 2147483648)

		// #if treats `2147483648` and `0xffffffff` as signed 32-bit (-2147483648 and -1).
		pp_src := `#version 310 es
#if 2147483648 < 0 && 0xffffffff == -1 && 037777777777 == -1 && 00 == 0
layout(local_size_x = 1) in;
layout(std430) buffer B { int w[]; } b;
void main() {
    int k = 5;
    b.w[0] = (2147483648 < 0 && 4294967295 == -1) ? max(k, 2147483648) : -99;
}
#else
#error wrong_sign
#endif`
		w, m := run_words(t, pp_src, 4)
		defer destroy(m)
		if w != nil {
			testing.expect_value(t, i32(w[0]), i32(5))
			testing.expect(t, strings.contains(m.msl, "max(ok_k, (-2147483647 - 1))"))
		}
	}

	// 7. testdata/const_fold.comp compiles cleanly and passes spirv-val.
	{
		cf_bytes, rerr := os.read_entire_file("testdata/const_fold.comp", context.temp_allocator)
		testing.expect(t, rerr == nil)
		if rerr == nil {
			cm, cerrs, cok := compile(string(cf_bytes), {profile = .Glsl, stage = .Compute})
			defer destroy(cm)
			testing.expectf(t, cok, "const_fold.comp failed:\n%s", format_errors(cerrs, context.temp_allocator))
			if cok && have_spirv_val() {
				expect_valid_spirv(t, "const_fold.comp", cm.spirv)
			}
		}
	}
}
