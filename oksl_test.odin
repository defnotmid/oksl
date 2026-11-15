package oksl

import "core:fmt"
import "core:math"
import "core:mem"
import "core:mem/virtual"
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
	return new_clone(Scratch{make([dynamic]Reg, context.temp_allocator), make([dynamic]Lane_Mask, context.temp_allocator), make([dynamic]u32, context.temp_allocator)}, context.temp_allocator)
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

@(private = "file")
Ref_Chan_Spec :: struct {
	data:   []f32,
	w, h:   int,
	comps:  int,
	filter: Channel_Filter,
}

@(private = "file")
ref_chan_texel :: proc(s: ^Ref_Chan_Spec, x, y: int) -> [4]f32 {
	b := (y * s.w + x) * s.comps
	switch s.comps {
	case 1:
		return {s.data[b], 0, 0, 1}
	case 2:
		return {s.data[b], s.data[b + 1], 0, 1}
	case:
		return {s.data[b], s.data[b + 1], s.data[b + 2], s.data[b + 3]}
	}
}

@(private = "file")
ref_chan_sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
	s := (^[4]Ref_Chan_Spec)(user)[channel]
	// Channel 1 in test_bound_channel_equivalence is dedicated to texelFetch(sFetch, fp, 0).
	if channel == 1 {
		if uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1 {
			return {}
		}
		px := clamp(int(math.floor(uv.x * f32(s.w))), 0, s.w - 1)
		py := clamp(int(math.floor(uv.y * f32(s.h))), 0, s.h - 1)
		return ref_chan_texel(&s, px, py)
	}
	if s.filter == .Nearest {
		px := coord_nearest(uv.x, f32(s.w), i32(s.w - 1))
		py := coord_nearest(uv.y, f32(s.h), i32(s.h - 1))
		return ref_chan_texel(&s, int(px), int(py))
	}
	x0, x1, fx := coord_linear(uv.x, f32(s.w), i32(s.w - 1))
	y0, y1, fy := coord_linear(uv.y, f32(s.h), i32(s.h - 1))
	c00 := ref_chan_texel(&s, int(x0), int(y0))
	c10 := ref_chan_texel(&s, int(x1), int(y0))
	c01 := ref_chan_texel(&s, int(x0), int(y1))
	c11 := ref_chan_texel(&s, int(x1), int(y1))
	top := c00 + (c10 - c00) * fx
	bot := c01 + (c11 - c01) * fx
	return top + (bot - top) * fy
}

@(private = "file")
run_f32_interp :: proc(p: ^Program, uniforms: []u8, width, height: int, out: [][4]f32, scratch: ^Scratch, opts: Run_Options) {
	saved_body, saved_init := p.jit_body, p.jit_init
	p.jit_body, p.jit_init = nil, nil
	defer {
		p.jit_body, p.jit_init = saved_body, saved_init
	}
	run_f32(p, uniforms, width, height, out, scratch, opts)
}

@(test)
test_bound_channel_equivalence :: proc(t: ^testing.T) {
	TW, TH :: 7, 5
	r_tex: [TW * TH]f32
	rg_tex: [TW * TH][2]f32
	rgba_tex: [TW * TH][4]f32
	for y in 0 ..< TH {
		for x in 0 ..< TW {
			i := y * TW + x
			r_tex[i] = f32((i * 5) % 19 - 7) * 0.25
			rg_tex[i] = {f32((i * 3 + 1) % 21 - 8) * 0.25, f32((i * 7 + 2) % 17 - 5) * 0.125}
			rgba_tex[i] = {
				f32((i * 3 + 2) % 23 - 9) * 0.25,
				f32((i * 5 + 4) % 19 - 6) * 0.125,
				f32((i * 7 + 1) % 25 - 10) * 0.25,
				f32((i * 2 + 3) % 13 - 3) * 0.25,
			}
		}
	}

	src := `#version 310 es
precision highp float;
uniform sampler2D s;
uniform sampler2D sFetch;
uniform int uLod;
layout(location = 0) out vec4 o;
void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    ivec2 sz0 = textureSize(s, 0);
    ivec2 szDyn = textureSize(s, uLod);
    vec2 uv = (gl_FragCoord.xy / vec2(13.0, 9.0)) * 1.8 - 0.4;
    vec2 centre = (vec2(px.x % sz0.x, px.y % sz0.y) + 0.5) / vec2(sz0);
    vec4 a = texture(s, uv);
    vec4 b = texture(s, centre, 2.25);
    vec4 c = textureLod(s, uv, 3.75) + textureLod(s, centre, -1.5);
    vec4 d = textureGrad(s, uv * 0.9 + 0.05, vec2(0.25, 0.03), vec2(-0.02, 0.2));
    ivec2 fp = ivec2(px.x - 3, px.y - 2);
    vec4 f = texelFetch(sFetch, fp, 0);
    vec4 acc = a + b * 0.5 + c * 0.25 + d * 0.125 + f + vec4(vec2(sz0 + szDyn) * 0.01, 0.0, 0.0);
    if (((px.x + px.y) & 1) == 0) {
        vec4 q = textureLod(s, uv * 0.7, 0.0);
        acc += (dFdx(q) + dFdy(q)) * 0.25;
    }
    o = acc;
}`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	block := make([]u8, max(m.block_size, 16), context.temp_allocator)
	set_uniform(m, block, "uLod", {0})

	W, H :: 13, 9
	for comps in ([?]int{1, 2, 4}) {
		for filter in ([?]Channel_Filter{.Nearest, .Linear}) {
			cimg: Channel_Image
			switch comps {
			case 1:
				cimg = channel_image(r_tex[:], TW, TH, filter)
			case 2:
				cimg = channel_image(rg_tex[:], TW, TH, filter)
			case:
				cimg = channel_image(rgba_tex[:], TW, TH, filter)
			}
			specs: [4]Ref_Chan_Spec
			specs[0] = {data = cimg.data, w = TW, h = TH, comps = comps, filter = filter}
			specs[1] = specs[0]
			sizes := [2][3]i32{{TW, TH, 1}, {TW, TH, 1}}

			out_bound_jit := make([][4]f32, W * H, context.temp_allocator)
			out_bound_vm := make([][4]f32, W * H, context.temp_allocator)
			out_cb_jit := make([][4]f32, W * H, context.temp_allocator)
			out_cb_vm := make([][4]f32, W * H, context.temp_allocator)

			b_opts := Run_Options{channel_images = {cimg, cimg}, x = 1, y = 1, w = 11, h = 7}
			c_opts := Run_Options{sampler = ref_chan_sampler, user = &specs, channel_sizes = sizes[:], x = 1, y = 1, w = 11, h = 7}

			run_f32(&m.program, block, W, H, out_bound_jit, temp_scratch(), b_opts)
			run_f32_interp(&m.program, block, W, H, out_bound_vm, temp_scratch(), b_opts)
			run_f32(&m.program, block, W, H, out_cb_jit, temp_scratch(), c_opts)
			run_f32_interp(&m.program, block, W, H, out_cb_vm, temp_scratch(), c_opts)

			wj := slice.reinterpret([]u32, out_bound_jit)
			wv := slice.reinterpret([]u32, out_bound_vm)
			wcj := slice.reinterpret([]u32, out_cb_jit)
			wcv := slice.reinterpret([]u32, out_cb_vm)
			for i in 0 ..< len(wj) {
				if wj[i] != wv[i] || wj[i] != wcj[i] || wj[i] != wcv[i] {
					testing.expectf(
						t,
						false,
						"mismatch comps=%d filter=%v word=%d: bound_jit=%08x bound_vm=%08x cb_jit=%08x cb_vm=%08x",
						comps, filter, i, wj[i], wv[i], wcj[i], wcv[i],
					)
					break
				}
			}
		}
	}

	// Non-zero lod on textureSize still returns the single-level bound image's size,
	// plus NaN / +-Inf / huge UVs, INT_MIN / INT_MAX texelFetch, and side-effecting coordinates.
	edge_src := `#version 310 es
precision highp float;
uniform sampler2D sNear;
uniform sampler2D sLin;
uniform float uZero;
layout(location = 0) out vec4 o;
void main() {
    float nanv = uZero / uZero;
    float pinf = 1.0 / uZero;
    float ninf = -1.0 / uZero;
    vec4 a = textureLod(sNear, vec2(nanv, pinf), 5.0) +
             textureLod(sNear, vec2(pinf, nanv), 5.0) +
             textureLod(sNear, vec2(ninf, 1e35), 0.0) +
             textureLod(sNear, vec2(1e35, ninf), 0.0) +
             textureLod(sNear, vec2(-1e35, -1e35), 0.0);
    vec4 b = textureLod(sLin, vec2(nanv, pinf), 5.0) +
             textureLod(sLin, vec2(pinf, nanv), 5.0) +
             textureLod(sLin, vec2(ninf, 1e35), 0.0) +
             textureLod(sLin, vec2(1e35, ninf), 0.0) +
             textureLod(sLin, vec2(-1e35, -1e35), 0.0);
    int k = 1;
    vec4 f0 = texelFetch(sLin, ivec2(k++, 2), 0);
    ivec2 ip = ivec2(2, 1);
    vec4 f1 = texelFetch(sNear, ip, uZero == 0.0 ? (ip = ivec2(99, 99)).x : 0);
    vec4 oob = texelFetch(sNear, ivec2(-2147483647 - 1, 0), 0) +
               texelFetch(sNear, ivec2(0, -2147483647 - 1), 0) +
               texelFetch(sNear, ivec2(2147483647, 0), 0) +
               texelFetch(sNear, ivec2(0, 2147483647), 0) +
               texelFetch(sNear, ivec2(-1, 1), 0) +
               texelFetch(sNear, ivec2(1, -1), 0) +
               texelFetch(sNear, ivec2(7, 1), 0) +
               texelFetch(sNear, ivec2(1, 5), 0);
    ivec2 sz3 = textureSize(sLin, 3);
    o = a + b + f0 + f1 + oob * 100.0 + vec4(float(k), float(ip.x), vec2(sz3));
}`
	em, eerrs, eok := compile(edge_src, {profile = .Glsl, stage = .Fragment})
	defer destroy(em)
	testing.expectf(t, eok, "edge compile failed:\n%s", format_errors(eerrs, context.temp_allocator))
	if eok {
		eblock := make([]u8, max(em.block_size, 16), context.temp_allocator)
		set_uniform(em, eblock, "uZero", {0})
		eimgs := []Channel_Image{
			channel_image(rgba_tex[:], TW, TH, .Nearest),
			channel_image(rgba_tex[:], TW, TH, .Linear),
		}
		ej := make([][4]f32, 16, context.temp_allocator)
		ev := make([][4]f32, 16, context.temp_allocator)
		run_f32(&em.program, eblock, 4, 4, ej, temp_scratch(), {channel_images = eimgs})
		run_f32_interp(&em.program, eblock, 4, 4, ev, temp_scratch(), {channel_images = eimgs})
		testing.expect(t, slice.equal(slice.reinterpret([]u32, ej), slice.reinterpret([]u32, ev)))
		// Verify k incremented once (to 2), ip ended at 99 while f1 fetched (2, 1), oob was 0, and sz3 == (7, 5).
		want_f0 := rgba_tex[2 * TW + 1]
		want_f1 := rgba_tex[1 * TW + 2]
		c_bl := rgba_tex[(TH - 1) * TW + 0]
		c_tr := rgba_tex[0 * TW + (TW - 1)]
		c_tl := rgba_tex[0]
		expected := (c_bl * 2 + c_tr * 2 + c_tl) * 2 + want_f0 + want_f1 + [4]f32{2, 99, TW, TH}
		for c in 0 ..< 4 {
			testing.expect_value(t, ej[0][c], expected[c])
		}
		// Also verify side-effecting k++ only runs once on unbound callback path!
		ucount := 0
		uch_sizes := [2][3]i32{{TW, TH, 1}, {TW, TH, 1}}
		u_sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			(^int)(user)^ += 1
			return {}
		}
		uj := make([][4]f32, 16, context.temp_allocator)
		uv_out := make([][4]f32, 16, context.temp_allocator)
		run_f32(&em.program, eblock, 4, 4, uj, temp_scratch(), {sampler = u_sampler, user = &ucount, channel_sizes = uch_sizes[:]})
		run_f32_interp(&em.program, eblock, 4, 4, uv_out, temp_scratch(), {sampler = u_sampler, user = &ucount, channel_sizes = uch_sizes[:]})
		testing.expect_value(t, uj[0], [4]f32{2, 99, 1, 1})
		testing.expect(t, slice.equal(slice.reinterpret([]u32, uj), slice.reinterpret([]u32, uv_out)))
	}

	// Shadertoy profile (flip_y = true) with bound Channel_Image: texture() flips uv.y and texelFetch() reads max_y - p.y.
	st_src := `void mainImage(out vec4 col, in vec2 fragCoord) {
    ivec2 p = ivec2(fragCoord);
    vec4 f_in = texelFetch(iChannel0, ivec2(1, 0), 0);
    vec4 f_oob = texelFetch(iChannel0, ivec2(1, -1), 0) + texelFetch(iChannel0, ivec2(1, 5), 0);
    vec4 t_bot = texture(iChannel0, vec2(1.5 / 7.0, 0.5 / 5.0));
    col = vec4(f_in.x, t_bot.x, f_oob.w, float(p.x >= 0 ? 1 : 0));
}`
	sm, serrs, sok := compile(st_src, {profile = .Shadertoy})
	defer destroy(sm)
	testing.expectf(t, sok, "shadertoy compile failed:\n%s", format_errors(serrs, context.temp_allocator))
	if sok {
		sblock := make([]u8, sm.block_size, context.temp_allocator)
		write_builtins(sblock, {resolution = {4, 2, 1}})
		// Use compile_program Buffer A so alpha/HDR floats are unclamped.
		g, gerrs, gok := compile_program({
			sources = #partial {.Buffer_A = st_src, .Image = "void mainImage(out vec4 c, in vec2 fc) { c = texelFetch(iChannel0, ivec2(fc), 0); }"},
			channels = #partial {.Buffer_A = {0 = {.External, 0}}, .Image = {0 = {.Buffer, u8(Pass_Kind.Buffer_A)}}},
		})
		defer destroy_program(g)
		testing.expectf(t, gok, "graph compile failed:\n%s", format_errors(gerrs, context.temp_allocator))
		if gok {
			ba := &g.passes[0].module.program
			bj := make([][4]f32, 8, context.temp_allocator)
			bv := make([][4]f32, 8, context.temp_allocator)
			sopts := Run_Options{channel_images = {channel_image(r_tex[:], TW, TH, .Nearest)}}
			run_f32(ba, sblock, 4, 2, bj, temp_scratch(), sopts)
			run_f32_interp(ba, sblock, 4, 2, bv, temp_scratch(), sopts)
			testing.expect(t, slice.equal(slice.reinterpret([]u32, bj), slice.reinterpret([]u32, bv)))
			// Shadertoy y=0 flips to row TH-1 = 4, x=1.
			want_flip := r_tex[(TH - 1) * TW + 1]
			testing.expect_value(t, bj[0], [4]f32{want_flip, want_flip, 0, 1})
		}
	}
}

@(test)
test_bound_channel_mixing_and_runners :: proc(t: ^testing.T) {
	r_img := [4]f32{0.5, -1.25, 2.0, 0.75}
	rg_img := [4][2]f32{{0.25, 0.5}, {1.25, -0.5}, {-0.75, 1.5}, {2.25, 0.125}}
	rgba_img := [4][4]f32{
		{0.125, 0.25, 0.5, 0.75},
		{1.5, -0.25, 0.75, 0.5},
		{-0.5, 1.25, -0.75, 0.25},
		{2.0, 0.5, 1.0, 1.25},
	}

	// 1. Mixed bound (ch 0 Nearest r32f, ch 2 Linear rg32f) and unbound (ch 1 empty Channel_Image, ch 3 past slice),
	// keeping >8 live vec4 values across the unbound calls to stress BB register cache flush/restore,
	// plus cross-BB unbound texelFetch with dynamic lod.
	mix_src := `#version 310 es
precision highp float;
uniform sampler2D s0;
uniform sampler2D s1;
uniform sampler2D s2;
uniform sampler2D s3;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec2 uv = vec2((float(px.x & 1) + 0.5) * 0.5, (float(px.y & 1) + 0.5) * 0.5);
    vec4 v0 = vec4(float(px.x) + 1.0, 2.0, 3.0, 4.0);
    vec4 v1 = v0 * 1.5 + vec4(0.25);
    vec4 v2 = v1 - vec4(0.5, 0.25, 0.125, 0.0);
    vec4 b0 = textureLod(s0, uv, 0.0);
    vec4 u1 = textureLod(s1, uv, 2.0);
    vec4 f1 = texelFetch(s1, px & 1, px.y & 1);
    vec4 f2 = texelFetch(s1, px & 1, (px.x & 1) == 0 ? 1 : 3);
    vec4 b2 = textureLod(s2, vec2(0.5, 0.5), 0.0);
    vec4 u3 = textureLod(s3, uv, 1.0);
    o0 = (v0 + v1 + v2) * 0.01 + b0 + u1 + f1 + f2;
    o1 = b2 + u3;
}`
	mm, merrs, mok := compile(mix_src, {profile = .Glsl, stage = .Fragment})
	defer destroy(mm)
	testing.expectf(t, mok, "mix compile failed:\n%s", format_errors(merrs, context.temp_allocator))
	if !mok {
		return
	}

	Call_Tracker :: struct {
		calls: [SAMPLER_COUNT]int,
		sink:  [16][4]f32,
	}
	mix_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		tr := (^Call_Tracker)(user)
		if channel >= 0 && channel < SAMPLER_COUNT {
			tr.calls[channel] += 1
		}
		// Exercise caller-saved XMM/YMM registers across the host callback boundary.
		base := [4]f32{uv.x, uv.y, lod, f32(channel)}
		r0 := base * 1.1; r1 := base * 1.2; r2 := base * 1.3; r3 := base * 1.4
		r4 := base * 1.5; r5 := base * 1.6; r6 := base * 1.7; r7 := base * 1.8
		r8 := base * 1.9; r9 := base * 2.0; r10 := base * 2.1; r11 := base * 2.2
		r12 := base * 2.3; r13 := base * 2.4; r14 := base * 2.5; r15 := base * 2.6
		tr.sink = {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15}
		return base
	}

	mblock := make([]u8, max(mm.block_size, 16), context.temp_allocator)
	msizes := [4][3]i32{{2, 2, 1}, {2, 2, 1}, {2, 2, 1}, {2, 2, 1}}
	mcimgs := [3]Channel_Image {
		channel_image(r_img[:], 2, 2, .Nearest),
		{data = r_img[:0], width = 2, height = 2, components = 1}, // non-nil empty slice -> unbound ch 1
		channel_image(rg_img[:], 2, 2, .Linear),
		// ch 3 past slice length -> also unbound
	}

	// Test `run_targets` (JIT vs Interpreter) and verify callback call counts.
	tj0, tj1 := make([][4]f32, 16, context.temp_allocator), make([][4]f32, 16, context.temp_allocator)
	tv0, tv1 := make([][4]f32, 16, context.temp_allocator), make([][4]f32, 16, context.temp_allocator)
	tr_jit, tr_vm: Call_Tracker
	mopts_jit := Run_Options{sampler = mix_cb, user = &tr_jit, channel_sizes = msizes[:], channel_images = mcimgs[:]}
	mopts_vm := Run_Options{sampler = mix_cb, user = &tr_vm, channel_sizes = msizes[:], channel_images = mcimgs[:]}

	run_targets(&mm.program, mblock, 4, 4, {tj0, tj1}, temp_scratch(), mopts_jit)
	{
		sb, si := mm.program.jit_body, mm.program.jit_init
		mm.program.jit_body, mm.program.jit_init = nil, nil
		run_targets(&mm.program, mblock, 4, 4, {tv0, tv1}, temp_scratch(), mopts_vm)
		mm.program.jit_body, mm.program.jit_init = sb, si
	}
	testing.expect_value(t, tr_jit.calls[0], 0)
	testing.expect_value(t, tr_jit.calls[2], 0)
	testing.expect(t, tr_jit.calls[1] > 0 && tr_jit.calls[3] > 0)
	testing.expect_value(t, tr_vm.calls[0], 0)
	testing.expect_value(t, tr_vm.calls[2], 0)
	testing.expect_value(t, tr_jit.calls[1], tr_vm.calls[1])
	testing.expect_value(t, tr_jit.calls[3], tr_vm.calls[3])
	testing.expect(t, slice.equal(slice.reinterpret([]u32, tj0), slice.reinterpret([]u32, tv0)))
	testing.expect(t, slice.equal(slice.reinterpret([]u32, tj1), slice.reinterpret([]u32, tv1)))

	// Test `run` (RGBA8) and extra entries > SAMPLER_COUNT in channel_images.
	long_cimgs: [SAMPLER_COUNT + 2]Channel_Image
	long_cimgs[0] = mcimgs[0]
	long_cimgs[2] = mcimgs[2]
	long_cimgs[SAMPLER_COUNT] = channel_image(r_img[:], 2, 2, .Nearest)
	u8_jit := make([]u8, 4 * 4 * 4, context.temp_allocator)
	u8_vm := make([]u8, 4 * 4 * 4, context.temp_allocator)
	tr_u8: Call_Tracker
	u8_opts := Run_Options{sampler = mix_cb, user = &tr_u8, channel_sizes = msizes[:], channel_images = long_cimgs[:]}
	run(&mm.program, mblock, 4, 4, u8_jit, 16, temp_scratch(), u8_opts)
	{
		sb, si := mm.program.jit_body, mm.program.jit_init
		mm.program.jit_body, mm.program.jit_init = nil, nil
		run(&mm.program, mblock, 4, 4, u8_vm, 16, temp_scratch(), u8_opts)
		mm.program.jit_body, mm.program.jit_init = sb, si
	}
	testing.expect(t, slice.equal(u8_jit, u8_vm))

	// Test `run_fragment` with Fragment_Quads (odd count = 3 + helper lanes).
	origin := [3][2]i32{{0, 0}, {2, 0}, {0, 2}}
	helpers := [3]Quad_Lanes{{}, {3}, {1, 2}}
	fq := Fragment_Quads{count = 3, origin = origin[:], helpers = helpers[:]}
	fj0, fj1 := make([][4]f32, 12, context.temp_allocator), make([][4]f32, 12, context.temp_allocator)
	fv0, fv1 := make([][4]f32, 12, context.temp_allocator), make([][4]f32, 12, context.temp_allocator)
	sh_j, sh_v := make([]Quad_Lanes, 3, context.temp_allocator), make([]Quad_Lanes, 3, context.temp_allocator)
	run_fragment(&mm.program, mblock, fq, {fj0, fj1}, sh_j, temp_scratch(), u8_opts)
	{
		sb, si := mm.program.jit_body, mm.program.jit_init
		mm.program.jit_body, mm.program.jit_init = nil, nil
		run_fragment(&mm.program, mblock, fq, {fv0, fv1}, sh_v, temp_scratch(), u8_opts)
		mm.program.jit_body, mm.program.jit_init = sb, si
	}
	testing.expect(t, slice.equal(sh_j, sh_v))
	testing.expect(t, slice.equal(slice.reinterpret([]u32, fj0), slice.reinterpret([]u32, fv0)))
	testing.expect(t, slice.equal(slice.reinterpret([]u32, fj1), slice.reinterpret([]u32, fv1)))

	// Straight-line unbound texelFetch shaders so fallback uv (`b`) and lod (`c`) each live strictly in the BB cache.
	for uf_src in ([?]string{
		"#version 310 es\nprecision highp float;\nuniform sampler2D s;\nout vec4 o;\nvoid main() { ivec2 p = ivec2(gl_FragCoord.xy); o = texelFetch(s, p & 1, 0); }",
		"#version 310 es\nprecision highp float;\nuniform sampler2D s;\nout vec4 o;\nvoid main() { o = texelFetch(s, ivec2(1, 0), int(gl_FragCoord.x)); }",
	}) {
		uf_m, _, uf_ok := compile(uf_src, {profile = .Glsl, stage = .Fragment})
		defer destroy(uf_m)
		testing.expect(t, uf_ok)
		if uf_ok {
			uf_j := make([][4]f32, 4, context.temp_allocator)
			uf_v := make([][4]f32, 4, context.temp_allocator)
			uf_tr: Call_Tracker
			uf_opts := Run_Options{sampler = mix_cb, user = &uf_tr, channel_sizes = msizes[:1]}
			run_f32(&uf_m.program, nil, 2, 2, uf_j, temp_scratch(), uf_opts)
			run_f32_interp(&uf_m.program, nil, 2, 2, uf_v, temp_scratch(), uf_opts)
			testing.expect(t, slice.equal(slice.reinterpret([]u32, uf_j), slice.reinterpret([]u32, uf_v)))
		}
	}

	// 2. Unbound `sampler2DArray` alongside a bound `sampler2D` in one shader succeeds.
	arr_mix_src := `#version 310 es
precision highp float;
uniform sampler2D s2d;
uniform sampler2DArray sArr;
layout(location = 0) out vec4 o;
void main() {
    o = textureLod(s2d, vec2(0.25, 0.25), 0.0) + textureLod(sArr, vec3(0.5, 0.5, 1.0), 0.0);
}`
	am, aerrs, aok := compile(arr_mix_src, {profile = .Glsl, stage = .Fragment})
	defer destroy(am)
	testing.expectf(t, aok, "arr mix failed:\n%s", format_errors(aerrs, context.temp_allocator))
	if aok {
		aj := make([][4]f32, 4, context.temp_allocator)
		av := make([][4]f32, 4, context.temp_allocator)
		atr: Call_Tracker
		acimgs := [2]Channel_Image{channel_image(r_img[:], 2, 2, .Nearest), {}}
		aopts := Run_Options{sampler = mix_cb, user = &atr, channel_images = acimgs[:]}
		run_f32(&am.program, nil, 2, 2, aj, temp_scratch(), aopts)
		run_f32_interp(&am.program, nil, 2, 2, av, temp_scratch(), aopts)
		testing.expect_value(t, atr.calls[0], 0)
		testing.expect(t, atr.calls[1] > 0)
		testing.expect(t, slice.equal(slice.reinterpret([]u32, aj), slice.reinterpret([]u32, av)))
	}

	// 3. `run_vertex` (11 vertices, first_vertex = 2) with bound Nearest & Linear channels.
	vs_src := `#version 310 es
uniform sampler2D s0;
uniform sampler2D s1;
out vec4 vData;
void main() {
    int id = gl_VertexID;
    vec2 uv = vec2(float(id & 1) * 0.5 + 0.25, float((id >> 1) & 1) * 0.5 + 0.25);
    vec4 a = textureLod(s0, uv, 0.0);
    vec4 b = textureLod(s1, vec2(0.5, 0.5), 0.0);
    vec4 f = texelFetch(s0, ivec2(id % 3, 0), 0);
    ivec2 sz = textureSize(s1, 0);
    gl_Position = a + b;
    vData = f + vec4(vec2(sz), 0.0, 0.0);
}`
	vm, verrs, vok := compile(vs_src, {profile = .Glsl, stage = .Vertex})
	defer destroy(vm)
	testing.expectf(t, vok, "vs compile failed:\n%s", format_errors(verrs, context.temp_allocator))
	if vok {
		VN :: 11
		pj, pv_pos := make([][4]f32, VN, context.temp_allocator), make([][4]f32, VN, context.temp_allocator)
		vj, vv := make([]f32, VN * vm.varying_floats, context.temp_allocator), make([]f32, VN * vm.varying_floats, context.temp_allocator)
		vopts := Run_Options{
			first_vertex   = 2,
			channel_images = {channel_image(r_img[:], 2, 2, .Nearest), channel_image(rg_img[:], 2, 2, .Linear)},
		}
		run_vertex(&vm.program, nil, nil, VN, pj, vj, temp_scratch(), vopts)
		{
			sb, si := vm.program.jit_body, vm.program.jit_init
			vm.program.jit_body, vm.program.jit_init = nil, nil
			run_vertex(&vm.program, nil, nil, VN, pv_pos, vv, temp_scratch(), vopts)
			vm.program.jit_body, vm.program.jit_init = sb, si
		}
		testing.expect(t, slice.equal(slice.reinterpret([]u32, pj), slice.reinterpret([]u32, pv_pos)))
		testing.expect(t, slice.equal(slice.reinterpret([]u32, vj), slice.reinterpret([]u32, vv)))
		// id == 2 -> id % 3 == 2 -> out-of-bounds on 2x2 s0 -> f == (0,0,0,0), sz == (2,2).
		testing.expect_value(t, [4]f32{vj[0], vj[1], vj[2], vj[3]}, [4]f32{2, 2, 0, 0})
	}

	// 4. `run_compute` sampling bound r32f, rg32f, rgba32f and an unbound channel.
	cs_src := `#version 310 es
layout(local_size_x = 8, local_size_y = 2) in;
uniform sampler2D s0;
uniform sampler2D s1;
uniform sampler2D s2;
uniform sampler2D s3;
layout(std430) writeonly buffer Out { vec4 v[]; } outBuf;
void main() {
    uint i = gl_GlobalInvocationID.x;
    vec2 uv = vec2(float(i & 1u) * 0.5 + 0.25, 0.5);
    vec4 a = textureLod(s0, uv, 0.0);
    vec4 b = textureLod(s1, vec2(0.5, 0.5), 2.0);
    vec4 c = textureGrad(s2, vec2(0.5, 0.5), vec2(0.1, 0.0), vec2(0.0, 0.1));
    vec4 f = texelFetch(s2, ivec2(int(i) - 1, 1), 0);
    vec4 u = textureLod(s3, uv, 1.0);
    ivec2 sz = textureSize(s2, 0);
    outBuf.v[i] = a + b + c + f + u + vec4(vec2(sz), 0.0, 0.0);
}`
	cm, cerrs, cok := compile(cs_src, {profile = .Glsl, stage = .Compute})
	defer destroy(cm)
	testing.expectf(t, cok, "cs compile failed:\n%s", format_errors(cerrs, context.temp_allocator))
	if cok {
		words := make([]u32, 16 * 4, context.temp_allocator)
		ctr: Call_Tracker
		copts := Run_Options {
			sampler = mix_cb,
			user    = &ctr,
			buffers = {words},
			channel_images = {
				channel_image(r_img[:], 2, 2, .Nearest),
				channel_image(rg_img[:], 2, 2, .Linear),
				channel_image(rgba_img[:], 2, 2, .Linear),
			},
		}
		run_compute(&cm.program, nil, {2, 1, 1}, temp_scratch(), copts)
		testing.expect_value(t, ctr.calls[0], 0)
		testing.expect_value(t, ctr.calls[1], 0)
		testing.expect_value(t, ctr.calls[2], 0)
		testing.expect_value(t, ctr.calls[3], 32)
		v4 := slice.reinterpret([][4]f32, words)
		// At i = 0: uv = (0.25, 0.5):
		// s0 Nearest(0.25*2=0.5->0, 0.5*2=1.0->1) -> (r_img[2], 0, 0, 1) = (2.0, 0, 0, 1)
		// s1 Linear(0.5, 0.5) -> average of all 4 rg_img texels, b=0, a=1
		// s2 Linear(0.5, 0.5) -> average of all 4 rgba_img texels
		// f = texelFetch(s2, (-1, 1), 0) -> (0, 0, 0, 0)
		// u = (0.25, 0.5, 1.0, 3.0)
		avg_rg := (rg_img[0] + rg_img[1] + rg_img[2] + rg_img[3]) * 0.25
		avg_rgba := (rgba_img[0] + rgba_img[1] + rgba_img[2] + rgba_img[3]) * 0.25
		want0 := [4]f32{2.0, 0, 0, 1} + [4]f32{avg_rg.x, avg_rg.y, 0, 1} + avg_rgba + [4]f32{0.25, 0.5, 1.0, 3.0} + [4]f32{2, 2, 0, 0}
		for k in 0 ..< 4 {
			testing.expectf(t, abs(v4[0][k] - want0[k]) <= 1e-6, "compute lane 0[%d]: got %v, want %v", k, v4[0][k], want0[k])
		}
	}

	// 5. multipass Pass_Graph: Buffer A renders float values, Image pass samples Buffer A as a bound Linear Channel_Image.
	{
		g, gerrs, gok := compile_program({
			sources = #partial {
				.Buffer_A = "void mainImage(out vec4 c, in vec2 fc) { c = vec4(fc.x, -fc.y, 2.5, 0.5); }",
				.Image    = "void mainImage(out vec4 c, in vec2 fc) { c = texture(iChannel0, fc / iResolution.xy); }",
			},
			channels = #partial {.Image = {0 = {.Buffer, u8(Pass_Kind.Buffer_A)}}},
		})
		defer destroy_program(g)
		testing.expectf(t, gok, "pass graph failed:\n%s", format_errors(gerrs, context.temp_allocator))
		if gok {
			GW, GH :: 6, 4
			buf_a := make([][4]f32, GW * GH, context.temp_allocator)
			blk_a := make([]u8, g.passes[0].module.block_size, context.temp_allocator)
			write_builtins(blk_a, {resolution = {GW, GH, 1}})
			run_f32(&g.passes[0].module.program, blk_a, GW, GH, buf_a, temp_scratch())

			blk_i := make([]u8, g.passes[1].module.block_size, context.temp_allocator)
			write_builtins(blk_i, {resolution = {GW, GH, 1}})
			img_j := make([][4]f32, GW * GH, context.temp_allocator)
			img_v := make([][4]f32, GW * GH, context.temp_allocator)
			iopts := Run_Options{channel_images = {channel_image(buf_a, GW, GH, .Linear)}}
			run_f32(&g.passes[1].module.program, blk_i, GW, GH, img_j, temp_scratch(), iopts)
			run_f32_interp(&g.passes[1].module.program, blk_i, GW, GH, img_v, temp_scratch(), iopts)
			testing.expect(t, slice.equal(slice.reinterpret([]u32, img_j), slice.reinterpret([]u32, img_v)))
			testing.expect_value(t, img_j[0], [4]f32{0.5 / 1.0, 0, 1, 1})
		}
	}

	// 6. testdata/bound_channels.frag compiles, passes spirv-val, and matches bit-for-bit between JIT and Interpreter.
	{
		bc_bytes, rerr := os.read_entire_file("testdata/bound_channels.frag", context.temp_allocator)
		testing.expect(t, rerr == nil)
		if rerr == nil {
			bcm, bcerrs, bcok := compile(string(bc_bytes), {profile = .Glsl, stage = .Fragment})
			defer destroy(bcm)
			testing.expectf(t, bcok, "bound_channels.frag failed:\n%s", format_errors(bcerrs, context.temp_allocator))
			if bcok {
				if have_spirv_val() {
					expect_valid_spirv(t, "bound_channels.frag", bcm.spirv)
				}
				bc_cimgs := [6]Channel_Image {
					channel_image(r_img[:], 2, 2, .Nearest),
					channel_image(rg_img[:], 2, 2, .Nearest),
					channel_image(rgba_img[:], 2, 2, .Nearest),
					channel_image(r_img[:], 2, 2, .Linear),
					channel_image(rg_img[:], 2, 2, .Linear),
					channel_image(rgba_img[:], 2, 2, .Linear),
				}
				bc_j := make([][4]f32, 64 * 36, context.temp_allocator)
				bc_v := make([][4]f32, 64 * 36, context.temp_allocator)
				run_f32(&bcm.program, nil, 64, 36, bc_j, temp_scratch(), {channel_images = bc_cimgs[:]})
				run_f32_interp(&bcm.program, nil, 64, 36, bc_v, temp_scratch(), {channel_images = bc_cimgs[:]})
				testing.expect(t, slice.equal(slice.reinterpret([]u32, bc_j), slice.reinterpret([]u32, bc_v)))
			}
		}
	}
}

@(private = "file")
compile_simple_frag_for_ensure :: proc(t: ^testing.T, src := "#version 310 es\nprecision highp float;\nuniform sampler2D s;\nout vec4 o;\nvoid main() { o = textureLod(s, vec2(0.5), 0.0); }") -> ^Module {
	m, _, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	testing.cleanup(t, proc(p: rawptr) { destroy((^Module)(p)) }, m)
	testing.expect(t, ok)
	return m
}

@(test)
test_channel_image_ensure_array :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t, "#version 310 es\nprecision highp float;\nuniform sampler2DArray s;\nout vec4 o;\nvoid main() { o = textureLod(s, vec3(0.5), 0.0); }")
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image cannot bind to a sampler2DArray channel")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {channel_image(px[:], 1, 1)}})
}

@(test)
test_channel_image_ensure_comps_zero :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [4]f32{1, 2, 3, 4}
	testing.expect_assert_message(t, "oksl: Channel_Image.components must be 1, 2 or 4")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 1, height = 1, components = 0}}})
}

@(test)
test_channel_image_ensure_comps_three :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [4]f32{1, 2, 3, 4}
	testing.expect_assert_message(t, "oksl: Channel_Image.components must be 1, 2 or 4")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 1, height = 1, components = 3}}})
}

@(test)
test_channel_image_ensure_filter :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: invalid Channel_Image.filter")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 1, height = 1, components = 1, filter = Channel_Filter(5)}}})
}

@(test)
test_channel_image_ensure_width :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image dimensions out of range")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 0, height = 1, components = 1}}})
}

@(test)
test_channel_image_ensure_height :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image dimensions out of range")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 1, height = 0, components = 1}}})
}

@(test)
test_channel_image_ensure_px_overflow :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image dimensions out of range")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 1 << 32, height = 1 << 32, components = 1}}})
}

@(test)
test_channel_image_ensure_need_overflow :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image dimensions out of range")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 1 << 31, height = 1 << 31, components = 4}}})
}

@(test)
test_channel_image_ensure_max_i32 :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image dimensions out of range")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 65536, height = 32768, components = 1}}})
}

@(test)
test_channel_image_ensure_short :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [23]f32{}
	testing.expect_assert_message(t, "oksl: Channel_Image.data is shorter than width * height * components")
	run_f32(&m.program, nil, 0, 0, nil, temp_scratch(), {channel_images = {{data = px[:], width = 4, height = 3, components = 2}}})
}

@(test)
test_channel_image_ensure_vertex :: proc(t: ^testing.T) {
	m, _, ok := compile("#version 310 es\nvoid main() { gl_Position = vec4(0.0); }", {profile = .Glsl, stage = .Vertex})
	testing.cleanup(t, proc(p: rawptr) { destroy((^Module)(p)) }, m)
	testing.expect(t, ok)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image.data is shorter than width * height * components")
	run_vertex(&m.program, nil, nil, 0, nil, nil, temp_scratch(), {channel_images = {{data = px[:], width = 2, height = 2, components = 1}}})
}

@(test)
test_channel_image_ensure_fragment :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image.data is shorter than width * height * components")
	run_fragment(&m.program, nil, {}, nil, nil, temp_scratch(), {channel_images = {{data = px[:], width = 2, height = 2, components = 1}}})
}

@(test)
test_channel_image_ensure_compute :: proc(t: ^testing.T) {
	m, _, ok := compile("#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() {}", {profile = .Glsl, stage = .Compute})
	testing.cleanup(t, proc(p: rawptr) { destroy((^Module)(p)) }, m)
	testing.expect(t, ok)
	px := [1]f32{1}
	testing.expect_assert_message(t, "oksl: Channel_Image.data is shorter than width * height * components")
	run_compute(&m.program, nil, {0, 1, 1}, temp_scratch(), {channel_images = {{data = px[:], width = 2, height = 2, components = 1}}})
}

@(test)
test_texel_fetch_taps_and_eval :: proc(t: ^testing.T) {
	// 1. Vertex shader with a tap, a texelFetch helper called twice, and side-effecting p and lod.
	{
		PRELUDE :: "in vec3 open0;\nout vec4 tap0;\n"
		VS :: `#version 310 es
in vec3 position;
uniform sampler2D map;
vec4 fetch(int i) {
    return texelFetch(map, ivec2(i, 0), 0);
}
void main() {
    int idx_p = 0;
    int idx_l = 0;
    float a = fetch(1).x;
    float b = fetch(3).x;
    float c = texelFetch(map, ivec2(idx_p++, 0), idx_l++).x;
    gl_Position = vec4(a + position.x, b, float(idx_p * 100 + idx_l * 1000) + c, 1.0);
}`
		m, errs, ok := compile(VS, {
			profile = .Glsl,
			stage   = .Vertex,
			prelude = PRELUDE,
			taps    = {{position = "tap0", attributes = {{"position", "open0"}}}},
		})
		defer destroy(m)
		if testing.expectf(t, ok, "tap vs compile failed:\n%s", format_errors(errs, context.temp_allocator)) {
			if have_spirv_val() {
				expect_valid_spirv(t, "tap_texel_fetch", m.spirv)
			}
			testing.expect_value(t, strings.count(m.msl, "idx_p++"), 2)
			testing.expect_value(t, strings.count(m.msl, "idx_l++"), 2)
			testing.expect(t, strings.contains(m.msl, ".get_width(") && !strings.contains(m.msl, "float2(1.0f, -1.0f)"))

			tex := [4]f32{10, 20, 30, 40}
			Tap_Map :: struct {
				tex:   [4]f32,
				calls: int,
			}
			tap_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
				tm := (^Tap_Map)(user)
				tm.calls += 1
				x := clamp(int(math.floor(uv.x * 4.0)), 0, 3)
				return {tm.tex[x], 0, 0, 1}
			}
			tm := Tap_Map{tex = tex}
			sizes := [1][3]i32{{4, 1, 1}}
			tap_off := 0
			for o in m.stage_outputs {
				if o.name == "tap0" {
					tap_off = o.offset
				}
			}

			// Distinct attribute twin: position.x = 1.0, open0.x = 1.5.
			pos_attr := [1][3]f32{{1.0, 0, 0}}
			open_attr := [1][3]f32{{1.5, 0, 0}}
			attrs: [8]Attribute
			for a in m.stage_inputs {
				ptr := &open_attr[0] if a.name == "open0" else &pos_attr[0]
				attrs[a.location] = {ptr, size_of([3]f32), .Float3}
			}
			opts_list := [2]Run_Options {
				{channel_images = {channel_image(tex[:], 4, 1, .Nearest)}},
				{sampler = tap_cb, user = &tm, channel_sizes = sizes[:]},
			}
			for ropts, mode in opts_list {
				for interp in ([2]bool{false, true}) {
					sb, si := m.program.jit_body, m.program.jit_init
					if interp {
						m.program.jit_body, m.program.jit_init = nil, nil
					}
					tm.calls = 0
					gp: [1][4]f32
					vary := make([]f32, m.varying_floats, context.temp_allocator)
					run_vertex(&m.program, nil, attrs[:], 1, gp[:], vary, temp_scratch(), ropts)
					m.program.jit_body, m.program.jit_init = sb, si
					got_tap := [4]f32{vary[tap_off], vary[tap_off + 1], vary[tap_off + 2], vary[tap_off + 3]}
					testing.expectf(t, gp[0] == {21.0, 40.0, 1110.0, 1.0}, "mode=%d interp=%v gl_Position=%v", mode, interp, gp[0])
					testing.expectf(t, got_tap == {21.5, 40.0, 1110.0, 1.0}, "mode=%d interp=%v tap0=%v", mode, interp, got_tap)
					testing.expect_value(t, tm.calls, 0 if mode == 0 else 6)
				}
			}

			// Identical attribute twin across LANES + 3 vertices: tap0 == gl_Position bit-for-bit.
			VN :: LANES + 3
			vpos: [VN][3]f32
			for &v, i in vpos {
				v = {f32(i) * 0.5 - 2.0, 0, 0}
			}
			for a in m.stage_inputs {
				attrs[a.location] = {&vpos[0], size_of([3]f32), .Float3}
			}
			for ropts in opts_list {
				for interp in ([2]bool{false, true}) {
					sb, si := m.program.jit_body, m.program.jit_init
					if interp {
						m.program.jit_body, m.program.jit_init = nil, nil
					}
					gp: [VN][4]f32
					vary := make([]f32, VN * m.varying_floats, context.temp_allocator)
					run_vertex(&m.program, nil, attrs[:], VN, gp[:], vary, temp_scratch(), ropts)
					m.program.jit_body, m.program.jit_init = sb, si
					for i in 0 ..< VN {
						r := vary[i * m.varying_floats + tap_off:]
						testing.expect_value(t, [4]f32{r[0], r[1], r[2], r[3]}, gp[i])
					}
				}
			}
		}
	}

	// 2. Buffer-backed p and lod (trim_buffer_loads), unsigned vs signed lod, mip size, nil sampler/sizes,
	//    masked quads/lanes, channel 15, and ternary short-circuiting (can_select_expr).
	{
		Rec :: struct {
			calls: int,
			uv:    [2]f32,
			lods:  [2]f32,
		}
		rec_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			r := (^Rec)(user)
			idx := min(r.calls, 1)
			r.uv = uv
			r.lods[idx] = lod
			r.calls += 1
			return {uv.x, uv.y, f32(channel), lod * 0 + 1}
		}

		FS_BUF :: `#version 310 es
precision highp float;
layout(std430) readonly buffer B { ivec4 w[]; } b;
layout(std430) readonly buffer U { uvec4 w[]; } ub;
uniform sampler2D s;
out vec4 o;
void main() {
    vec4 skip = false ? texelFetch(s, ivec2(0), 0) : vec4(7.0);
    if (gl_FragCoord.x < 2.0 && gl_FragCoord.y < 2.0) {
        vec4 u = texelFetch(s, b.w[0].xy, ub.w[0].x);
        vec4 sgn = texelFetch(s, b.w[0].xy, b.w[1].x);
        o = u + sgn + skip;
    } else {
        o = skip;
    }
}`
		fm, ferrs, fok := compile(FS_BUF, {profile = .Glsl, stage = .Fragment})
		defer destroy(fm)
		if testing.expectf(t, fok, "fs_buf failed:\n%s", format_errors(ferrs, context.temp_allocator)) {
			if have_spirv_val() {
				expect_valid_spirv(t, "fs_buf_texel_fetch", fm.spirv)
			}
			// b.w[0].xy = (1, 2), ub.w[0].x = 0xffffffffu (4294967296.0), b.w[1].x = -1 (-1.0).
			b_words := [8]u32{1, 2, 99, 99, transmute(u32)i32(-1), 99, 99, 99}
			u_words := [4]u32{0xffffffff, 99, 99, 99}
			sizes := [1][3]i32{{8, 4, 1}}
			for interp in ([2]bool{false, true}) {
				rec: Rec
				out := make([][4]f32, 4 * 4, context.temp_allocator)
				opts := Run_Options{sampler = rec_cb, user = &rec, channel_sizes = sizes[:], buffers = {b_words[:], u_words[:]}}
				if interp {
					run_f32_interp(&fm.program, nil, 4, 4, out, temp_scratch(), opts)
				} else {
					run_f32(&fm.program, nil, 4, 4, out, temp_scratch(), opts)
				}
				// 4x4 target with x < 2 && y < 2: only top-left quad active (4 lanes * 2 fetches = 8 calls),
				// top-right quad idle within the same 8-lane batch (0 calls), bottom quads idle, and `?:` skipped.
				testing.expect_value(t, rec.calls, 8)
				testing.expect_value(t, rec.lods[0], f32(u32(0xffffffff)))
				testing.expect_value(t, rec.lods[1], f32(-1))
				testing.expect_value(t, rec.uv, [2]f32{1.5, 2.5})
				testing.expect_value(t, out[2], [4]f32{7, 7, 7, 7})

				// Nil sampler writes 0 on fetched lanes.
				out_nil := make([][4]f32, 4, context.temp_allocator)
				opts_nil := Run_Options{buffers = {b_words[:], u_words[:]}}
				if interp {
					run_f32_interp(&fm.program, nil, 2, 1, out_nil, temp_scratch(), opts_nil)
				} else {
					run_f32(&fm.program, nil, 2, 1, out_nil, temp_scratch(), opts_nil)
				}
				testing.expect_value(t, out_nil[0], [4]f32{7, 7, 7, 7})

				// Mip size shift (lod = 1 on 8x4 -> 4x2 -> uv = (1.5/4, 2.5/2) = (0.375, 1.25)),
				// mip floor of 1 on each axis past the last level (lod = 3 on 8x4 and 4x8 -> 1x1 -> uv = (1.5, 2.5)),
				// and nil channel_sizes (+Inf).
				mip_b := [8]u32{1, 2, 0, 0, 1, 0, 0, 0}
				mip_u := [4]u32{1, 0, 0, 0}
				rec_mip: Rec
				out_mip := make([][4]f32, 1, context.temp_allocator)
				opts_mip := Run_Options{sampler = rec_cb, user = &rec_mip, channel_sizes = sizes[:], buffers = {mip_b[:], mip_u[:]}}
				if interp {
					run_f32_interp(&fm.program, nil, 1, 1, out_mip, temp_scratch(), opts_mip)
				} else {
					run_f32(&fm.program, nil, 1, 1, out_mip, temp_scratch(), opts_mip)
				}
				testing.expect_value(t, rec_mip.uv, [2]f32{0.375, 1.25})

				floor_b := [8]u32{1, 2, 0, 0, 3, 0, 0, 0}
				floor_u := [4]u32{3, 0, 0, 0}
				for fsz in ([2][1][3]i32{{{8, 4, 1}}, {{4, 8, 1}}}) {
					sz_slice := fsz
					rec_fl: Rec
					opts_fl := Run_Options{sampler = rec_cb, user = &rec_fl, channel_sizes = sz_slice[:], buffers = {floor_b[:], floor_u[:]}}
					if interp {
						run_f32_interp(&fm.program, nil, 1, 1, out_mip, temp_scratch(), opts_fl)
					} else {
						run_f32(&fm.program, nil, 1, 1, out_mip, temp_scratch(), opts_fl)
					}
					testing.expect_value(t, rec_fl.uv, [2]f32{1.5, 2.5})
				}

				rec_nosz: Rec
				opts_nosz := Run_Options{sampler = rec_cb, user = &rec_nosz, buffers = {mip_b[:], mip_u[:]}}
				if interp {
					run_f32_interp(&fm.program, nil, 1, 1, out_mip, temp_scratch(), opts_nosz)
				} else {
					run_f32(&fm.program, nil, 1, 1, out_mip, temp_scratch(), opts_nosz)
				}
				testing.expect(t, math.is_inf(rec_nosz.uv.x, 1) && math.is_inf(rec_nosz.uv.y, 1))
			}
		}

		// Compute unbound texelFetch with masked lanes and unsigned/signed lod.
		CS_BUF :: `#version 310 es
layout(local_size_x = 8) in;
layout(std430) readonly buffer In { ivec4 w[]; } b;
layout(std430) readonly buffer UIn { uvec4 w[]; } ub;
layout(std430) writeonly buffer Out { vec4 v[]; } outBuf;
uniform sampler2D s;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < 3u) {
        outBuf.v[i] = texelFetch(s, b.w[0].xy, ub.w[0].x) + texelFetch(s, b.w[0].xy, b.w[1].x);
    }
}`
		cm, cerrs, cok := compile(CS_BUF, {profile = .Glsl, stage = .Compute})
		defer destroy(cm)
		if testing.expectf(t, cok, "cs_buf failed:\n%s", format_errors(cerrs, context.temp_allocator)) {
			b_words := [8]u32{1, 2, 0, 0, transmute(u32)i32(-1), 0, 0, 0}
			u_words := [4]u32{0xffffffff, 0, 0, 0}
			out_w := make([]u32, 8 * 4, context.temp_allocator)
			sizes := [1][3]i32{{8, 4, 1}}
			rec: Rec
			run_compute(&cm.program, nil, {1, 1, 1}, temp_scratch(), {
				sampler       = rec_cb,
				user          = &rec,
				channel_sizes = sizes[:],
				buffers       = {b_words[:], u_words[:], out_w},
			})
			testing.expect_value(t, rec.calls, 6)
			testing.expect_value(t, rec.lods[0], f32(u32(0xffffffff)))
			testing.expect_value(t, rec.lods[1], f32(-1))
			testing.expect_value(t, rec.uv, [2]f32{1.5, 2.5})
		}

		// Channel 15 (SAMPLER_COUNT - 1) boundary check.
		CH15 :: `#version 310 es
precision highp float;
uniform sampler2D s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
out vec4 o;
void main() {
    o = texelFetch(s0, ivec2(0), 0) + texelFetch(s1, ivec2(0), 0) + texelFetch(s2, ivec2(0), 0) + texelFetch(s3, ivec2(0), 0) +
        texelFetch(s4, ivec2(0), 0) + texelFetch(s5, ivec2(0), 0) + texelFetch(s6, ivec2(0), 0) + texelFetch(s7, ivec2(0), 0) +
        texelFetch(s8, ivec2(0), 0) + texelFetch(s9, ivec2(0), 0) + texelFetch(s10, ivec2(0), 0) + texelFetch(s11, ivec2(0), 0) +
        texelFetch(s12, ivec2(0), 0) + texelFetch(s13, ivec2(0), 0) + texelFetch(s14, ivec2(0), 0) + texelFetch(s15, ivec2(0), 0);
}`
		m15, e15, ok15 := compile(CH15, {profile = .Glsl, stage = .Fragment})
		defer destroy(m15)
		if testing.expectf(t, ok15, "ch15 failed:\n%s", format_errors(e15, context.temp_allocator)) {
			px := [1]f32{3.0}
			cimgs: [SAMPLER_COUNT]Channel_Image
			for &ci in cimgs {
				ci = channel_image(px[:], 1, 1)
			}
			out15 := make([][4]f32, 1, context.temp_allocator)
			run_f32(&m15.program, nil, 1, 1, out15, temp_scratch(), {channel_images = cimgs[:]})
			testing.expect_value(t, out15[0], [4]f32{48, 0, 0, 16})
		}
	}

	// 3. Shadertoy header texelFetch: dynamic lod, side-effecting p and lod, header size & flip_y.
	{
		ST :: `void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    int idx_p = 1;
    int idx_l = int(fragCoord.x) & 1;
    vec4 t = texelFetch(iChannel0, ivec2(idx_p++, 1), idx_l++);
    fragColor = vec4(t.xy, float(idx_p) * 0.25, float(idx_l) * 0.25);
}`
		sm, serrs, sok := compile(ST, {profile = .Shadertoy, unclamped = true})
		defer destroy(sm)
		if testing.expectf(t, sok, "st texelFetch failed:\n%s", format_errors(serrs, context.temp_allocator)) {
			if have_spirv_val() {
				expect_valid_spirv(t, "st_texel_fetch", sm.spirv)
			}
			testing.expect(
				t,
				strings.contains(sm.msl, "u.iChannelResolution") &&
				!strings.contains(sm.msl, ".get_width(") &&
				strings.contains(sm.msl, "float2(1.0f, -1.0f)") &&
				strings.count(sm.msl, "idx_p++") == 1 &&
				strings.count(sm.msl, "idx_l++") == 1,
			)
			blk := make([]u8, sm.block_size, context.temp_allocator)
			write_builtins(blk, {resolution = {2, 1, 1}, channel_resolution = {0 = {5, 4, 1}}})

			// Unbound callback receives uv = ((1 + 0.5)/5, ((1 + 0.5)/4)*-1 + 1) = (0.3, 0.625) and lod = 0, 1.
			st_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
				return {uv.x, uv.y + lod * 10.0, 0, 1}
			}
			oj := make([][4]f32, 2, context.temp_allocator)
			ov := make([][4]f32, 2, context.temp_allocator)
			run_f32(&sm.program, blk, 2, 1, oj, temp_scratch(), {sampler = st_cb})
			run_f32_interp(&sm.program, blk, 2, 1, ov, temp_scratch(), {sampler = st_cb})
			testing.expect(t, slice.equal(slice.reinterpret([]u32, oj), slice.reinterpret([]u32, ov)))
			near(t, oj[0], {0.3, 0.625, 0.5, 0.25})
			near(t, oj[1], {0.3, 10.625, 0.5, 0.5})

			// Bound 5x4 image: Shadertoy flip_y maps p=(1, 1) to row (4 - 1 - 1) = 2, col 1 -> index 11.
			st_img: [20]f32
			st_img[2 * 5 + 1] = 9.5
			bj := make([][4]f32, 2, context.temp_allocator)
			bv := make([][4]f32, 2, context.temp_allocator)
			bopts := Run_Options{channel_images = {channel_image(st_img[:], 5, 4)}}
			run_f32(&sm.program, blk, 2, 1, bj, temp_scratch(), bopts)
			run_f32_interp(&sm.program, blk, 2, 1, bv, temp_scratch(), bopts)
			testing.expect(t, slice.equal(slice.reinterpret([]u32, bj), slice.reinterpret([]u32, bv)))
			near(t, bj[0], {9.5, 0.0, 0.5, 0.25})
		}
	}
}

@(test)
test_scratch_reuse_clears_channel_images :: proc(t: ^testing.T) {
	scratch: Scratch
	defer scratch_destroy(&scratch)

	img_a := [2]f32{10, 10}
	img_b := [3]f32{7, 7, 7}
	empty_imgs := [1]Channel_Image{{}}
	sizes := [1][3]i32{{5, 1, 1}}
	cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		return {99, 0, 0, 1}
	}
	runs := [4]Run_Options {
		{channel_images = {channel_image(img_a[:], 2, 1)}},
		{sampler = cb, channel_sizes = sizes[:]},
		{sampler = cb, channel_sizes = sizes[:], channel_images = empty_imgs[:]},
		{channel_images = {channel_image(img_b[:], 3, 1)}},
	}
	wants := [4][4]f32 {
		{10, 2, 0, 1},
		{99, 5, 0, 1},
		{99, 5, 0, 1},
		{7, 3, 0, 1},
	}

	// 1. run_vertex reusing one Scratch across bound -> unbound -> empty slice -> bound.
	{
		VS :: "#version 310 es\nuniform sampler2D s;\nvoid main() { gl_Position = vec4(textureLod(s, vec2(0.25, 0.5), 0.0).x, float(textureSize(s, 0).x), 0.0, 1.0); }"
		vm, verrs, vok := compile(VS, {profile = .Glsl, stage = .Vertex})
		defer destroy(vm)
		if testing.expectf(t, vok, "vs failed:\n%s", format_errors(verrs, context.temp_allocator)) {
			for ropts, r in runs {
				gp: [1][4]f32
				run_vertex(&vm.program, nil, nil, 1, gp[:], nil, &scratch, ropts)
				testing.expectf(t, gp[0] == wants[r], "run_vertex[%d]: got %v, want %v", r, gp[0], wants[r])
			}
		}
	}

	// 2. run_fragment and run_f32 reusing the same Scratch across bound -> unbound -> empty slice -> bound.
	{
		FS :: "#version 310 es\nprecision highp float;\nuniform sampler2D s;\nlayout(location = 0) out vec4 o;\nvoid main() { o = vec4(textureLod(s, vec2(0.25, 0.5), 0.0).x, float(textureSize(s, 0).x), 0.0, 1.0); }"
		fm, ferrs, fok := compile(FS, {profile = .Glsl, stage = .Fragment})
		defer destroy(fm)
		if testing.expectf(t, fok, "fs failed:\n%s", format_errors(ferrs, context.temp_allocator)) {
			origins := [1][2]i32{{0, 0}}
			quads := Fragment_Quads{count = 1, origin = origins[:]}
			for ropts, r in runs {
				t0: [4][4]f32
				targets := [1][][4]f32{t0[:]}
				shaded: [1]Quad_Lanes
				run_fragment(&fm.program, nil, quads, targets[:], shaded[:], &scratch, ropts)
				testing.expectf(t, t0[0] == wants[r], "run_fragment[%d]: got %v, want %v", r, t0[0], wants[r])
			}
			for ropts, r in runs {
				px: [1][4]f32
				run_f32(&fm.program, nil, 1, 1, px[:], &scratch, ropts)
				testing.expectf(t, px[0] == wants[r], "run_f32[%d]: got %v, want %v", r, px[0], wants[r])
			}
		}
	}
}

@(test)
test_texel_fetch_header_size_reads :: proc(t: ^testing.T) {
	// 1. trim_buffer_loads keeps 2 components of c when d & 2 != 0, and drops c when d & 2 == 0.
	for hdr in ([2]bool{true, false}) {
		blocks := [5]Slots{{0, 1}, {1, 2}, {3, 1}, {4, 4}, {8, 4}}
		code := [2]Instr {
			{op = .Load_Buf, n = 4, dst = 4, a = 0, b = 4},
			{op = .Texel_Fetch, n = 2, dst = 8, a = 1, b = 3, c = 4, d = 2 if hdr else 0},
		}
		p := Program{code = code[:], num_slots = 12}
		trim_buffer_loads(&p, blocks[:])
		if hdr {
			testing.expect(t, len(p.code) == 2 && p.code[0].op == .Load_Buf && p.code[0].n == 2)
		} else {
			testing.expect(t, len(p.code) == 1 && p.code[0].op == .Texel_Fetch)
		}
	}

	// 2. instr_read_write marks {c, 2} across basic blocks so a computed size in BB 1 flushes to memory before Texel_Fetch in BB 2.
	consts := [4][2]u32 {
		{0, transmute(u32)f32(4.0)},
		{1, transmute(u32)f32(2.0)},
		{2, 1},
		{3, 1},
	}
	outputs := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 6, n = 4, base = .Float}}
	code := [3]Instr {
		{op = .Mov, n = 2, dst = 4, a = 0},
		{op = .Texel_Fetch, n = 2, k = 0, d = 2, dst = 10, a = 2, b = 2, c = 4},
		{op = .Store, n = 4, dst = 6, a = 10},
	}
	p := Program {
		code      = code[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 14,
		num_masks = 2,
		stage     = .Fragment,
	}
	jit_compile(&p)
	defer jit_destroy(&p)

	uv_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		return {uv.x, uv.y, 0, 1}
	}
	oj: [1][4]f32
	ov: [1][4]f32
	run_f32(&p, nil, 1, 1, oj[:], temp_scratch(), {sampler = uv_cb})
	run_f32_interp(&p, nil, 1, 1, ov[:], temp_scratch(), {sampler = uv_cb})
	testing.expect_value(t, oj[0], [4]f32{0.375, 0.75, 0, 1})
	testing.expect_value(t, ov[0], [4]f32{0.375, 0.75, 0, 1})
}

when !ODIN_DISABLE_ASSERT {
	@(private = "file")
	lower_with_channel_binding :: proc(t: ^testing.T, bad_binding: i32) {
		context.allocator = context.temp_allocator
		src :: "#version 310 es\nprecision highp float;\nuniform sampler2D s;\nout vec4 o;\nvoid main() { o = textureLod(s, vec2(0.5), 0.0); }"
		u, toks, front_ok := module_front(src, {profile = .Glsl, stage = .Fragment}, .Main, .Prelude)
		testing.expect(t, front_ok)
		checked, check_ok := module_check(u, toks)
		testing.expect(t, check_ok)
		for &row in checked.interface {
			if row.kind == .Channel {
				row.binding = bad_binding
			}
		}
		_ = build_program(u, &checked)
	}

	@(test)
	test_lower_sampler_channel_assert_neg :: proc(t: ^testing.T) {
		testing.expect_assert_message(t, "oksl: sampler channel out of range")
		lower_with_channel_binding(t, -1)
	}

	@(test)
	test_lower_sampler_channel_assert_hi :: proc(t: ^testing.T) {
		testing.expect_assert_message(t, "oksl: sampler channel out of range")
		lower_with_channel_binding(t, SAMPLER_COUNT)
	}
}

@(test)
test_compute_phase2_diagnostics :: proc(t: ^testing.T) {
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

	// 1. `shared` outside compute (Fragment, Vertex, Shadertoy, SkSL).
	expect_err(t, "#version 310 es\nprecision highp float;\nshared float s;\nout vec4 o;\nvoid main() { o = vec4(s); }", .Compute_Only, {profile = .Glsl, stage = .Fragment})
	expect_err(t, "#version 310 es\nshared float s;\nvoid main() { gl_Position = vec4(s); }", .Compute_Only, {profile = .Glsl, stage = .Vertex})
	expect_err(t, "shared float s;\nvoid mainImage(out vec4 c, in vec2 fc) { c = vec4(s); }", .Compute_Only, {profile = .Shadertoy})
	expect_err(t, "shared float s;\nhalf4 main(float2 p) { return half4(s); }", .Compute_Only, {profile = .Sksl})

	// 2. `shared` initializers.
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared int s = 1;\nvoid main() {}", .Shared_Initializer)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared vec2 a[2] = vec2[2](vec2(0.0), vec2(1.0));\nvoid main() {}", .Shared_Initializer)

	// 3. `shared` combined with invalid qualifiers, layouts, functions, or buffers.
	for bad_decl in ([?]string{
		"const shared int s;",
		"shared const int s;",
		"uniform shared int s;",
		"shared uniform int s;",
		"readonly shared int s;",
		"writeonly shared int s;",
		"coherent shared int s;",
		"volatile shared int s;",
		"restrict shared int s;",
		"shared shared int s;",
		"layout(location = 0) shared int s;",
		"shared buffer B { int a[]; };",
		"layout(std430) shared buffer B { int a[]; };",
		"shared void f() {}",
	}) {
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\n%s\nvoid main() {{}}", bad_decl), .Shared_Qualifier)
	}
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nin shared int s;\nvoid main() {}", .Compute_Stage_Var)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nout shared int s;\nvoid main() {}", .Compute_Stage_Var)

	// 4. `shared` in non-global scopes (local, for-init, param, struct field, buffer member).
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() { shared int x; }", .Shared_Global)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() { for (shared int i = 0; i < 2; i++) {} }", .Shared_Global)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nvoid f(shared int x) {}\nvoid main() {}", .Shared_Global)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nstruct S { shared int x; };\nvoid main() {}", .Shared_Global)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nbuffer B { shared int a[]; };\nvoid main() {}", .Shared_Global)

	// 5. `shared` invalid types (unsized array, opaque sampler/image, struct containing opaque).
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared int a[];\nvoid main() {}", .Shared_Type)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared sampler2D s;\nvoid main() {}", .Shared_Type)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared sampler2DArray s;\nvoid main() {}", .Shared_Type)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared image2D im;\nvoid main() {}", .Shared_Type)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nstruct Bad { sampler2D s; };\nshared Bad b;\nvoid main() {}", .Shared_Type)

	// 6. Total `shared` size limit (16384 bytes) and alignment padding.
	{
		// Exact 16384 bytes via 1024 vec4s succeeds.
		m_ok1, e_ok1, ok1 := compile("#version 310 es\nlayout(local_size_x = 4) in;\nshared vec4 a[1024];\nvoid main() { a[0] = vec4(1.0); }", {profile = .Glsl, stage = .Compute})
		defer destroy(m_ok1)
		testing.expectf(t, ok1, "16384B vec4[1024] failed: %s", format_errors(e_ok1, context.temp_allocator))

		// Alignment padding: int (4B -> padded to 16B) + vec4[1023] (16368B) = 16384B exact succeeds.
		m_ok2, e_ok2, ok2 := compile("#version 310 es\nlayout(local_size_x = 4) in;\nshared int pad;\nshared vec4 a[1023];\nvoid main() { pad = 1; a[0] = vec4(float(pad)); }", {profile = .Glsl, stage = .Compute})
		defer destroy(m_ok2)
		testing.expectf(t, ok2, "padded 16384B failed: %s", format_errors(e_ok2, context.temp_allocator))
	}
	// Adding 4 more bytes after padded 16384B -> 16388B fails (would be 16376B without 16B alignment padding).
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared int pad;\nshared vec4 a[1023];\nshared int over;\nvoid main() {}", .Shared_Size_Max)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared vec4 a[1025];\nvoid main() {}", .Shared_Size_Max)

	// 7. Barrier stage rules, argument counts, and GLSL ES 3.10 §8.14 control-flow legality.
	for fn_name in ([?]string{"barrier", "memoryBarrierShared", "groupMemoryBarrier"}) {
		expect_err(t, fmt.tprintf("#version 310 es\nprecision highp float;\nout vec4 o;\nvoid main() {{ %s(); o = vec4(1.0); }}", fn_name), .Compute_Only, {profile = .Glsl, stage = .Fragment})
		expect_err(t, fmt.tprintf("#version 310 es\nvoid main() {{ %s(); gl_Position = vec4(1.0); }}", fn_name), .Compute_Only, {profile = .Glsl, stage = .Vertex})
		expect_err(t, fmt.tprintf("void mainImage(out vec4 c, in vec2 fc) {{ %s(); c = vec4(1.0); }}", fn_name), .Compute_Only, {profile = .Shadertoy})
		expect_err(t, fmt.tprintf("half4 main(float2 p) {{ %s(); return half4(1.0); }}", fn_name), .Unknown_Function, {profile = .Sksl})
	}
	for fn_name in ([?]string{"memoryBarrier", "memoryBarrierBuffer", "memoryBarrierImage"}) {
		expect_err(t, fmt.tprintf("half4 main(float2 p) {{ %s(); return half4(1.0); }}", fn_name), .Unknown_Function, {profile = .Sksl})
		expect_err(t, fmt.tprintf("void mainImage(out vec4 c, in vec2 fc) {{ %s(); c = vec4(1.0); }}", fn_name), .Compute_Only, {profile = .Shadertoy})
		// Allowed in GLSL ES 3.10 fragment and vertex stages per §8.15 (SPIR-V OpMemoryBarrier, MSL no-op).
		mf, ef, okf := compile(fmt.tprintf("#version 310 es\nprecision highp float;\nout vec4 o;\nvoid main() {{ %s(); o = vec4(1.0); }}", fn_name), {profile = .Glsl, stage = .Fragment})
		defer destroy(mf)
		testing.expectf(t, okf, "%s in fragment failed: %s", fn_name, format_errors(ef, context.temp_allocator))
		if okf {
			if have_spirv_val() {
				expect_valid_spirv(t, fn_name, mf.spirv)
			}
			testing.expect(t, strings.contains(mf.msl, "((void)0);"))
		}
		mv, ev, okv := compile(fmt.tprintf("#version 310 es\nvoid main() {{ %s(); gl_Position = vec4(1.0); }}", fn_name), {profile = .Glsl, stage = .Vertex})
		defer destroy(mv)
		testing.expectf(t, okv, "%s in vertex failed: %s", fn_name, format_errors(ev, context.temp_allocator))
		if okv && have_spirv_val() {
			expect_valid_spirv(t, fn_name, mv.spirv)
		}
	}
	for fn_name in ([?]string{"barrier", "memoryBarrier", "memoryBarrierShared", "memoryBarrierBuffer", "memoryBarrierImage", "groupMemoryBarrier"}) {
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() {{ %s(1); }}", fn_name), .Arg_Count)
	}
	// ES 3.10 §8.14 permits barrier() inside uniform if/else/switch/for/while/do-while, helpers, and after return.
	{
		CF_SRC :: `#version 310 es
layout(local_size_x = 8) in;
uniform int uMode;
shared uint sh[8];
void sync_helper() { barrier(); }
void main() {
    uint tid = gl_LocalInvocationIndex;
    sh[tid] = tid;
    if (uMode < 0) { return; }
    barrier();
    if (uMode == 0) {
        barrier();
    } else {
        sync_helper();
    }
    switch (uMode) {
    case 0:
        barrier();
        break;
    default:
        barrier();
        break;
    }
    for (int i = 0; i < 2; i++) { barrier(); }
    int w = 0;
    while (w < 1) { barrier(); w++; }
    do { barrier(); } while (false);
}`
		mcf, ecf, okcf := compile(CF_SRC, {profile = .Glsl, stage = .Compute})
		defer destroy(mcf)
		testing.expectf(t, okcf, "uniform control-flow barrier failed: %s", format_errors(ecf, context.temp_allocator))
		if okcf && have_spirv_val() {
			expect_valid_spirv(t, "cf_barrier", mcf.spirv)
		}
	}

	// 8. Atomic operation diagnostics.
	for op in ([?]string{"atomicAdd", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExchange"}) {
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nshared int s;\nvoid main() {{ %s(s); }}", op), .Arg_Count)
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nshared int s;\nvoid main() {{ %s(s, 1, 2); }}", op), .Arg_Count)
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nshared float s;\nvoid main() {{ %s(s, 1.0); }}", op), .No_Overload)
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nshared bool s;\nvoid main() {{ %s(s, true); }}", op), .No_Overload)
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nshared ivec2 s;\nvoid main() {{ %s(s, ivec2(1)); }}", op), .No_Overload)
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nshared uint s;\nvoid main() {{ %s(s, 1); }}", op), .No_Overload)
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() {{ int loc = 0; %s(loc, 1); }}", op), .Atomic_Target)
	}
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared int s;\nvoid main() { atomicCompSwap(s, 0); }", .Arg_Count)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared int s;\nvoid main() { atomicCompSwap(s, 0, 1, 2); }", .Arg_Count)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared uint s;\nvoid main() { atomicCompSwap(s, 0, 1u); }", .No_Overload)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared uint s;\nvoid main() { atomicCompSwap(s, 0u, 1); }", .No_Overload)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared float s;\nvoid main() { atomicCompSwap(s, 0.0, 1.0); }", .No_Overload)

	// Invalid atomic targets: global non-shared, parameter, uniform, literal, expression, repeated swizzle, readonly & writeonly buffers.
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nint g;\nvoid main() { atomicAdd(g, 1); }", .Atomic_Target)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nvoid f(inout int p) { atomicAdd(p, 1); }\nvoid main() {}", .Atomic_Target)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nuniform int u;\nvoid main() { atomicAdd(u, 1); }", .Atomic_Target)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() { atomicAdd(1, 1); }", .Atomic_Target)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared int s;\nvoid main() { atomicAdd(s + 1, 1); }", .Atomic_Target)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nshared ivec2 s;\nvoid main() { atomicAdd(s.xx[0], 1); }", .Swizzle_Repeat)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nreadonly buffer B { int a[]; } b;\nvoid main() { atomicAdd(b.a[0], 1); }", .Read_Only)
	{
		m_wo, e_wo, ok_wo := compile("#version 310 es\nlayout(local_size_x = 4) in;\nwriteonly buffer B { int a[]; } b;\nvoid main() { atomicAdd(b.a[0], 1); }", {profile = .Glsl, stage = .Compute})
		defer destroy(m_wo)
		testing.expect(t, !ok_wo && len(e_wo) == 1 && e_wo[0].code == .Write_Only)
	}

	// 9. Outside ES 3.10 core: atomic_uint counters and imageAtomic* builtins.
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nlayout(binding = 0) uniform atomic_uint c;\nvoid main() {}", .Unsupported_Atomic_Counter)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() { atomic_uint c; }", .Unsupported_Atomic_Counter)
	expect_err(t, "#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() { uint x = atomic_uint(0); }", .Unsupported_Atomic_Counter)
	for fn_name in ([?]string{"atomicCounter", "atomicCounterIncrement", "atomicCounterDecrement", "memoryBarrierAtomicCounter"}) {
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nvoid main() {{ %s(); }}", fn_name), .Unsupported_Atomic_Counter)
	}
	for fn_name in ([?]string{"imageAtomicAdd", "imageAtomicMin", "imageAtomicMax", "imageAtomicAnd", "imageAtomicOr", "imageAtomicXor", "imageAtomicExchange", "imageAtomicCompSwap"}) {
		expect_err(t, fmt.tprintf("#version 310 es\nlayout(local_size_x = 4) in;\nlayout(r32ui) uniform uimage2D im;\nvoid main() {{ %s(im, ivec2(0), 1u); }}", fn_name), .Unsupported_Image_Atomic)
	}
}

@(test)
test_compute_shared_types_and_zeroing :: proc(t: ^testing.T) {
	// 1. All shared types (bool, int, uint, float, vectors, matrices, arrays, nested structs),
	//    swizzle writes/compositions/indexing, out/inout helpers, and ternary short-circuiting.
	TYPES_SRC :: `#version 310 es
layout(local_size_x = 8) in;

struct Inner {
    bool flag;
    ivec2 pair;
};

struct Outer {
    Inner in_s;
    mat2x3 m;
    uvec4 uv;
};

shared bool sh_b;
shared bvec3 sh_bv;
shared int sh_i;
shared uint sh_u;
shared float sh_f;
shared vec4 sh_v4;
shared mat2 sh_m2;
shared Outer sh_arr[8];
shared uint sh_guard_cnt;

layout(std430) writeonly buffer Out { uvec4 out_v[]; } dst;

void write_out_vec2(out vec2 dst_v, float seed) {
    dst_v = vec2(seed + 10.0, seed + 20.0);
}

void bump_inout(inout int x, int delta) {
    x += delta;
}

void main() {
    uint tid = gl_LocalInvocationIndex;
    int dyn = int(tid & 3u);

    if (tid == 0u) {
        sh_b = true;
        sh_bv = bvec3(false, true, true);
        sh_i = 12;
        bump_inout(sh_i, 5);
        sh_u = 7u;
        sh_u++;
        --sh_u;
        sh_f = 2.5;
        sh_v4 = vec4(1.0, 2.0, 3.0, 4.0);
        write_out_vec2(sh_v4.wzyx.yx, 100.0); // writes .z = 110, .w = 120
        sh_v4.wzyx[1] = 50.0;                 // writes .z = 50
        sh_m2 = mat2(1.0, 2.0, 3.0, 4.0);
        sh_m2[1][0] += 10.0;                  // col 1 row 0: 3 -> 13
    }
    sh_arr[tid].in_s.flag = (tid & 1u) == 0u;
    sh_arr[tid].in_s.pair = ivec2(int(tid) * 3, -int(tid));
    sh_arr[tid].m = mat2x3(
        float(tid) + 1.0, 2.0, 3.0,
        4.0, 5.0, float(tid) + 6.0
    );
    sh_arr[tid].uv = uvec4(10u, 20u, 30u, 40u);
    sh_arr[tid].uv.wzyx[dyn] += tid + 1u;

    // Untaken ternary arms must NOT touch shared memory or execute atomics.
    uint skip_sh = false ? sh_arr[tid].uv.x : 99u;
    uint skip_at = false ? atomicAdd(sh_guard_cnt, 1u) : 77u;

    barrier();

    uint peer = (tid + 3u) & 7u;
    Outer p = sh_arr[peer];
    uint flags = (sh_b ? 1u : 0u) | (sh_bv.y ? 2u : 0u) | (p.in_s.flag ? 4u : 0u);
    int sum_i = sh_i + p.in_s.pair.x + p.in_s.pair.y;
    float sum_f = sh_f + sh_v4.x + sh_v4.y + sh_v4.z + sh_v4.w + sh_m2[1][0] + p.m[0][0] + p.m[1][2];
    uint sum_u = sh_u + p.uv.x + p.uv.y + p.uv.z + p.uv.w + skip_sh + skip_at + sh_guard_cnt;
    dst.out_v[tid] = uvec4(flags, uint(sum_i), floatBitsToUint(sum_f), sum_u);
}`
	mt, et, okt := compile(TYPES_SRC, {profile = .Glsl, stage = .Compute})
	defer destroy(mt)
	testing.expectf(t, okt, "shared types compile failed:\n%s", format_errors(et, context.temp_allocator))
	if okt {
		if have_spirv_val() {
			expect_valid_spirv(t, "shared_types", mt.spirv)
		}
		testing.expect(t, strings.contains(mt.msl, "threadgroup uint oksl_sh["))
		out_words := make([]u32, 8 * 4, context.temp_allocator)
		run_compute(&mt.program, nil, {1, 1, 1}, temp_scratch(), {buffers = {out_words}})
		for tid in u32(0) ..< 8 {
			peer := (tid + 3) & 7
			want_flags := u32(1 | 2 | (4 if (peer & 1) == 0 else 0))
			want_i := i32(17) + i32(peer) * 3 - i32(peer)
			// sh_f (2.5) + sh_v4 (1 + 2 + 50 + 120 = 173) + sh_m2[1][0] (13) + p.m[0][0] (peer+1) + p.m[1][2] (peer+6)
			want_f := f32(2.5 + 173.0 + 13.0) + f32(peer + 1) + f32(peer + 6)
			// p.uv started (10,20,30,40)=100, plus (peer + 1) added to one component; sh_u=7, skip_sh=99, skip_at=77, sh_guard_cnt=0.
			want_u := u32(7 + 100 + (peer + 1) + 99 + 77 + 0)
			base := tid * 4
			testing.expect_value(t, out_words[base + 0], want_flags)
			testing.expect_value(t, i32(out_words[base + 1]), want_i)
			testing.expect_value(t, transmute(f32)out_words[base + 2], want_f)
			testing.expect_value(t, out_words[base + 3], want_u)
		}
	}

	// 2. Deterministic per-workgroup zeroing of uninitialized `shared` memory on both
	//    single-chunk (local_size_x = 32) and multi-chunk barrier (local_size_x = 128) paths.
	for lsx in ([2]int{32, 128}) {
		zsrc := fmt.tprintf(`#version 310 es
layout(local_size_x = %d) in;
shared uint sh_uninit[%d];
shared uint sh_sum;
layout(std430) writeonly buffer Out {{ uint sums[]; }} dst;
void main() {{
    uint tid = gl_LocalInvocationIndex;
    uint wg = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    sh_uninit[tid] += (wg + 1u);
    barrier();
    atomicAdd(sh_sum, sh_uninit[(tid + 7u) %% %du]);
    barrier();
    if (tid == 0u) {{
        dst.sums[wg] = sh_sum;
    }}
}}`, lsx, lsx, lsx)
		mz, ez, okz := compile(zsrc, {profile = .Glsl, stage = .Compute})
		defer destroy(mz)
		testing.expectf(t, okz, "zeroing lsx=%d failed:\n%s", lsx, format_errors(ez, context.temp_allocator))
		if okz {
			sc := temp_scratch()
			resize(&sc.shared, 256)
			slice.fill(sc.shared[:], 0xdeadbeef)
			sums := make([]u32, 6, context.temp_allocator)
			run_compute(&mz.program, nil, {3, 2, 1}, sc, {buffers = {sums}})
			for g in 0 ..< 6 {
				testing.expect_value(t, sums[g], u32((g + 1) * lsx))
			}
		}
	}
}

@(test)
test_compute_barriers_chunks_and_pool :: proc(t: ^testing.T) {
	// 1. Barrier chunk synchronization, live local/mask survival across barrier(),
	//    helper barriers, and uniform hoisting across workgroup sizes 64, 96, 256, and 1024.
	for lsx in ([4]int{64, 96, 256, 1024}) {
		bsrc := fmt.tprintf(`#version 310 es
layout(local_size_x = %d) in;
uniform float uScale;
uniform int uRounds;
shared float sh_ring[%d];
layout(std430) writeonly buffer Out {{ float v[]; }} dst;

void step_barrier(uint tid, int r, float gain) {{
    float peer = sh_ring[(tid + uint(r * 19 + 65)) %% %du];
    barrier();
    sh_ring[tid] = peer + gain;
    barrier();
}}

void main() {{
    uint tid = gl_LocalInvocationIndex;
    uint wg = gl_WorkGroupID.x;
    float hoisted = uScale * 2.0 + 1.0;
    float live_local = float(tid) * 0.5 + float(wg) * 10.0;
    sh_ring[tid] = float(tid + wg * 100u);
    barrier();
    if ((tid & 1u) == 0u) {{
        live_local += 3.0;
    }}
    for (int r = 0; r < uRounds; r++) {{
        step_barrier(tid, r, hoisted);
    }}
    dst.v[wg * %du + tid] = sh_ring[tid] + live_local;
}}`, lsx, lsx, lsx, lsx)
		mb, eb, okb := compile(bsrc, {profile = .Glsl, stage = .Compute})
		defer destroy(mb)
		testing.expectf(t, okb, "barrier lsx=%d compile failed:\n%s", lsx, format_errors(eb, context.temp_allocator))
		if !okb {
			continue
		}
		if have_spirv_val() {
			expect_valid_spirv(t, "barrier_chunks", mb.spirv)
		}
		testing.expect(t, mb.program.init_len > 0)
		ublock := make([]u8, mb.block_size, context.temp_allocator)
		set_uniform(mb, ublock, "uScale", {1.5})  // hoisted = 4.0
		set_uniform(mb, ublock, "uRounds", {3})
		NG :: 5
		out_single := make([]u32, NG * lsx, context.temp_allocator)
		run_compute(&mb.program, ublock, {NG, 1, 1}, temp_scratch(), {buffers = {out_single}})

		// Verify exact expected values across all chunks and lanes.
		// Over 3 rounds, shift mod lsx is (0*19+65) + (1*19+65) + (2*19+65) = 65 + 84 + 103 = 252.
		shift := (65 + 84 + 103) % lsx
		f_single := slice.reinterpret([]f32, out_single)
		for wg in 0 ..< NG {
			for tid in 0 ..< lsx {
				src_tid := (tid + shift) % lsx
				want_ring := f32(src_tid + wg * 100) + 3.0 * 4.0
				want_live := f32(tid) * 0.5 + f32(wg) * 10.0 + (3.0 if (tid & 1) == 0 else 0.0)
				testing.expect_value(t, f_single[wg * lsx + tid], want_ring + want_live)
			}
		}

		// Also dispatch over a 4-thread pool via first_group / group_count and verify bit-identical output.
		if lsx == 256 {
			Pool_Task :: struct {
				prog:        ^Program,
				ublock:      []u8,
				out_words:   []u32,
				first_group: int,
				group_count: int,
			}
			worker_proc :: proc(th: ^thread.Thread) {
				task := (^Pool_Task)(th.data)
				sc: Scratch
				defer scratch_destroy(&sc)
				run_compute(task.prog, task.ublock, {NG, 1, 1}, &sc, {buffers = {task.out_words}}, task.first_group, task.group_count)
			}
			out_pool := make([]u32, NG * lsx, context.temp_allocator)
			ranges := [4][2]int{{0, 1}, {1, 2}, {3, 1}, {4, 1}}
			tasks: [4]Pool_Task
			threads: [4]^thread.Thread
			for i in 0 ..< 4 {
				tasks[i] = {prog = &mb.program, ublock = ublock, out_words = out_pool, first_group = ranges[i][0], group_count = ranges[i][1]}
				threads[i] = thread.create(worker_proc)
				threads[i].data = &tasks[i]
				thread.start(threads[i])
			}
			for th in threads {
				thread.join(th)
				thread.destroy(th)
			}
			testing.expect(t, slice.equal(out_single, out_pool))
		}
	}

	// 2. trim_buffer_loads preserves Load_Buf operands feeding Load_Shared, Store_Shared, Atomic_Shared, and Atomic_Buf.
	{
		blocks := [6]Slots{{0, 1}, {1, 4}, {5, 4}, {9, 4}, {13, 4}, {17, 1}}
		code := [8]Instr {
			{op = .Load_Buf, n = 4, dst = 1, a = 0, b = 4},
			{op = .Load_Buf, n = 4, dst = 5, a = 0, b = 4},
			{op = .Load_Buf, n = 4, dst = 9, a = 0, b = 4},
			{op = .Load_Buf, n = 4, dst = 13, a = 0, b = 4},
			{op = .Load_Shared, n = 1, k = 1, dst = 17, a = 2, c = 0},                                   // k != 0 -> reads dyn slot a=2 -> block 1[1]
			{op = .Store_Shared, n = 1, k = 1, a = 6, b = 8, c = 0},                                     // reads value a=6 and dyn b=8 -> block 2[1..3]
			{op = .Atomic_Shared, n = 1, k = u8(Atomic_Op.Comp_Swap), dst = 17, a = 10, b = 11, c = 12}, // reads a,b,c = 10..12 -> block 3[1..3]
			{op = .Atomic_Buf, n = 1, k = u8(Atomic_Op.Comp_Swap), dst = 17, a = 13, b = 14, c = 16, d = 0}, // reads a,b,c = 13,14,16 -> block 4[0..3]
		}
		p := Program{code = code[:], num_slots = 18}
		trim_buffer_loads(&p, blocks[:])
		testing.expect_value(t, len(p.code), 8)
		testing.expect(t, p.code[0].op == .Load_Buf && p.code[0].n == 1)
		testing.expect(t, p.code[1].op == .Load_Buf && p.code[1].n == 3)
		testing.expect(t, p.code[2].op == .Load_Buf && p.code[2].n == 3)
		testing.expect(t, p.code[3].op == .Load_Buf && p.code[3].n == 4)
	}

	// 3. Early `return` before `barrier()` across multiple chunks (both workgroup-uniform
	//    early return per GLSL ES 3.10 §8.14 and CPU divergent chunk/lane early return):
	//    returned lanes neither stall the barrier scheduler nor resume after `barrier()`.
	{
		RET_SRC :: `#version 310 es
layout(local_size_x = 160) in;
shared uint sh_data[160];
layout(std430) writeonly buffer Out { uint out_v[]; } dst;
void main() {
    uint tid = gl_LocalInvocationIndex;
    uint wg = gl_WorkGroupID.x;
    uint out_idx = wg * 160u + tid;
    sh_data[tid] = tid + 10u;
    if (wg == 1u) {
        dst.out_v[out_idx] = 9000u + tid;
        return;
    }
    if (wg == 2u && (tid < 64u || (tid & 1u) == 1u)) {
        dst.out_v[out_idx] = 5000u + tid;
        return;
    }
    barrier();
    dst.out_v[out_idx] = sh_data[(tid + 64u) % 160u] + 1000u;
}`
		mr, er, okr := compile(RET_SRC, {profile = .Glsl, stage = .Compute})
		defer destroy(mr)
		testing.expectf(t, okr, "early return barrier failed:\n%s", format_errors(er, context.temp_allocator))
		if okr {
			if have_spirv_val() {
				expect_valid_spirv(t, "early_ret_barrier", mr.spirv)
			}
			out_r := make([]u32, 3 * 160, context.temp_allocator)
			run_compute(&mr.program, nil, {3, 1, 1}, temp_scratch(), {buffers = {out_r}})
			for tid in u32(0) ..< 160 {
				testing.expect_value(t, out_r[0 * 160 + tid], ((tid + 64) % 160) + 10 + 1000)
				testing.expect_value(t, out_r[1 * 160 + tid], 9000 + tid)
				if tid < 64 || (tid & 1) == 1 {
					testing.expect_value(t, out_r[2 * 160 + tid], 5000 + tid)
				} else {
					testing.expect_value(t, out_r[2 * 160 + tid], ((tid + 64) % 160) + 10 + 1000)
				}
			}
		}
	}
}

@(test)
test_compute_atomics_and_testdata :: proc(t: ^testing.T) {
	// 1. Signed vs unsigned Min/Max, Add/And/Or/Xor/Exchange/CompSwap, helper writes_state sequencing,
	//    masked-off lanes, out-of-bounds Atomic_Buf safety, and deterministic (chunk, then lane) CPU order across 160 threads.
	ORDER_SRC :: `#version 310 es
layout(local_size_x = 160) in;

shared uint sh_seq;
shared int sh_imin;
shared int sh_imax;
shared int sh_imin_flip;
shared int sh_imax_flip;
shared int sh_imax_neg;
shared int sh_cas_ord;
shared uint sh_umin;
shared uint sh_umax;
shared int sh_iand;
shared int sh_ior;
shared int sh_ixor;
shared int sh_iexch;
shared int sh_icas;
shared uint sh_uand;
shared uint sh_uor;
shared uint sh_uxor;
shared uint sh_uexch;
shared uint sh_ucas;
shared uint sh_masked;
shared uint sh_side_a;
shared uint sh_side_b;

layout(std430) buffer IBuf { int iw[]; } ib;
layout(std430) buffer UBuf { uint uw[]; } ub;

uint helper_write_sh() {
    sh_side_a = 40u;
    return 2u;
}

uint helper_atomic_sh() {
    return atomicAdd(sh_side_b, 30u);
}

void main() {
    uint tid = gl_LocalInvocationIndex;
    if (tid == 0u) {
        sh_imin = 5;
        sh_imax = -5;
        sh_imin_flip = -5;
        sh_imax_flip = 5;
        sh_imax_neg = -5;
        sh_cas_ord = 12;
        sh_umin = 0x80000010u;
        sh_umax = 5u;
        sh_iand = 0xff;
        sh_ior = 0x10;
        sh_ixor = 0x55;
        sh_iexch = -77;
        sh_icas = -12;
        sh_uand = 0xffu;
        sh_uor = 0x10u;
        sh_uxor = 0x55u;
        sh_uexch = 88u;
        sh_ucas = 34u;
        sh_side_a = 10u;
        sh_side_b = 5u;
    }
    barrier();

    ub.uw[16u + tid] = atomicAdd(sh_seq, 1u);
    ub.uw[176u + tid] = atomicAdd(ub.uw[0], 1u);

    if (tid < 3u) {
        atomicAdd(sh_masked, 1u);
    }
    uint unreached_ternary = (tid > 999u) ? atomicAdd(sh_masked, 100u) : 0u;
    if (unreached_ternary != 0u) {
        atomicAdd(sh_masked, 1000u);
    }
    if (tid == 1u) {
        atomicMin(sh_imin, -10);
        atomicMax(sh_imax, 3);
        atomicMin(sh_imin_flip, 3);
        atomicMax(sh_imax_flip, -10);
        atomicMax(sh_imax_neg, -2);
        atomicMin(ib.iw[0], -10);
        atomicMax(ib.iw[1], 3);
        atomicMin(ib.iw[12], 3);
        atomicMax(ib.iw[13], -10);
        atomicMax(ib.iw[14], -2);
        atomicMin(sh_umin, 5u);
        atomicMax(sh_umax, 0x80000020u);
        atomicMin(ub.uw[1], 5u);
        atomicMax(ub.uw[2], 0x80000020u);

        // atomicCompSwap evaluation order: compare is captured before data mutates it.
        int cmp_sh = 12;
        int cas_ord_sh = atomicCompSwap(sh_cas_ord, cmp_sh, (cmp_sh = 99) + 1);
        int cmp_buf = 34;
        int cas_ord_buf = atomicCompSwap(ib.iw[15], cmp_buf, (cmp_buf = 88) + 1);
        ib.iw[16] = cas_ord_sh * 1000 + cmp_sh;
        ib.iw[17] = cas_ord_buf * 1000 + cmp_buf;

        int prev_iand = atomicAnd(sh_iand, 0x3c);
        int prev_ior  = atomicOr(sh_ior, 0x05);
        int prev_ixor = atomicXor(sh_ixor, 0x0f);
        int prev_iex  = atomicExchange(sh_iexch, 42);
        int miss_icas = atomicCompSwap(sh_icas, 99, 100);
        int hit_icas  = atomicCompSwap(sh_icas, -12, 73);
        ib.iw[4] = prev_iand + prev_ior + prev_ixor + prev_iex + miss_icas + hit_icas;

        int bp_iand = atomicAnd(ib.iw[5], 0x3c);
        int bp_ior  = atomicOr(ib.iw[6], 0x05);
        int bp_ixor = atomicXor(ib.iw[7], 0x0f);
        int bp_iex  = atomicExchange(ib.iw[8], 42);
        int bm_icas = atomicCompSwap(ib.iw[9], 99, 100);
        int bh_icas = atomicCompSwap(ib.iw[9], -12, 73);
        ib.iw[10] = bp_iand + bp_ior + bp_ixor + bp_iex + bm_icas + bh_icas;

        uint prev_uand = atomicAnd(sh_uand, 0x3cu);
        uint prev_uor  = atomicOr(sh_uor, 0x05u);
        uint prev_uxor = atomicXor(sh_uxor, 0x0fu);
        uint prev_uex  = atomicExchange(sh_uexch, 91u);
        uint miss_ucas = atomicCompSwap(sh_ucas, 99u, 100u);
        uint hit_ucas  = atomicCompSwap(sh_ucas, 34u, 67u);
        ub.uw[7] = prev_uand + prev_uor + prev_uxor + prev_uex + miss_ucas + hit_ucas;

        uint bp_uand = atomicAnd(ub.uw[8], 0x3cu);
        uint bp_uor  = atomicOr(ub.uw[9], 0x05u);
        uint bp_uxor = atomicXor(ub.uw[10], 0x0fu);
        uint bp_uex  = atomicExchange(ub.uw[11], 91u);
        uint bm_ucas = atomicCompSwap(ub.uw[12], 99u, 100u);
        uint bh_ucas = atomicCompSwap(ub.uw[12], 34u, 67u);
        ub.uw[13] = bp_uand + bp_uor + bp_uxor + bp_uex + bm_ucas + bh_ucas;

        // Left-to-right sequencing when a helper or direct atomic updates shared.
        uint seq1 = sh_side_a * 100u + helper_write_sh() + sh_side_a;
        uint seq2 = sh_side_b * 100u + helper_atomic_sh() + sh_side_b + atomicAdd(sh_side_b, 4u);
        ub.uw[14] = seq1;
        ub.uw[15] = seq2 + sh_side_b;

        // Out-of-bounds buffer atomics return 0 and leave valid elements intact.
        int hi_idx = int(tid) + 99999;
        int neg_idx = int(tid) - 10;
        uint oob1 = atomicAdd(ub.uw[hi_idx], 7u);
        int oob2 = atomicCompSwap(ib.iw[neg_idx], 0, 99);
        ub.uw[3] = oob1 + uint(oob2);
    }
    barrier();
    if (tid == 0u) {
        ib.iw[2] = sh_imin;
        ib.iw[3] = sh_imax;
        ib.iw[18] = sh_imin_flip;
        ib.iw[19] = sh_imax_flip;
        ib.iw[20] = sh_imax_neg;
        ib.iw[21] = sh_cas_ord;
        ib.iw[11] = sh_iand + sh_ior * 100 + sh_ixor * 10000 + sh_iexch + sh_icas;
        ub.uw[4] = sh_umin;
        ub.uw[5] = sh_umax;
        ub.uw[6] = sh_masked;
        ub.uw[336] = sh_uand + sh_uor * 100u + sh_uxor * 10000u + sh_uexch + sh_ucas;
    }
}`
	mo, eo, oko := compile(ORDER_SRC, {profile = .Glsl, stage = .Compute})
	defer destroy(mo)
	testing.expectf(t, oko, "order_src failed:\n%s", format_errors(eo, context.temp_allocator))
	if oko {
		if have_spirv_val() {
			expect_valid_spirv(t, "order_atomics", mo.spirv)
		}
		// Helper calls with writes_state and direct atomics must sequence operands via temporaries in MSL
		// (note 4 added shared ints shifted sh_side_a/sh_side_b to indices 20u/21u).
		testing.expect(t, !strings.contains(mo.msl, "oksl_sh[20u] * 100u) + ok_helper_write_sh"))
		testing.expect(t, !strings.contains(mo.msl, "oksl_sh[21u] * 100u) + ok_helper_atomic_sh"))
		testing.expect(t, !strings.contains(mo.msl, "oksl_sh[21u]) + atomic_fetch_add_explicit"))
		// Used atomic results must route through volatile locals in MSL.
		testing.expect(t, strings.contains(mo.msl, "volatile int _t"))
		testing.expect(t, strings.contains(mo.msl, "volatile uint _t"))

		iw := [22]i32{5, -5, 0, 0, 0, 0xff, 0x10, 0x55, -77, -12, 0, 0, -5, 5, -5, 34, 0, 0, 0, 0, 0, 0}
		uw := make([]u32, 337, context.temp_allocator)
		uw[1] = 0x80000010
		uw[2] = 5
		uw[3] = 0xdead
		uw[8], uw[9], uw[10], uw[11], uw[12] = 0xff, 0x10, 0x55, 88, 34
		run_compute(&mo.program, nil, {1, 1, 1}, temp_scratch(), {
			buffers = {slice.reinterpret([]u32, iw[:]), uw},
		})
		testing.expect_value(t, uw[0], u32(160))
		testing.expect_value(t, iw[0], i32(-10))
		testing.expect_value(t, iw[1], i32(3))
		testing.expect_value(t, iw[2], i32(-10))
		testing.expect_value(t, iw[3], i32(3))
		testing.expect_value(t, iw[12], i32(-5))
		testing.expect_value(t, iw[13], i32(5))
		testing.expect_value(t, iw[14], i32(-2))
		testing.expect_value(t, iw[15], i32(89))
		testing.expect_value(t, iw[16], i32(12 * 1000 + 99))
		testing.expect_value(t, iw[17], i32(34 * 1000 + 88))
		testing.expect_value(t, iw[18], i32(-5))
		testing.expect_value(t, iw[19], i32(5))
		testing.expect_value(t, iw[20], i32(-2))
		testing.expect_value(t, iw[21], i32(100))
		want_prev_i := i32(0xff + 0x10 + 0x55 + (-77) + (-12) + (-12))
		want_post_i := i32((0xff & 0x3c) + (0x10 | 0x05) * 100 + (0x55 ~ 0x0f) * 10000 + 42 + 73)
		testing.expect_value(t, iw[4], want_prev_i)
		testing.expect_value(t, iw[5], i32(0xff & 0x3c))
		testing.expect_value(t, iw[6], i32(0x10 | 0x05))
		testing.expect_value(t, iw[7], i32(0x55 ~ 0x0f))
		testing.expect_value(t, iw[8], i32(42))
		testing.expect_value(t, iw[9], i32(73))
		testing.expect_value(t, iw[10], want_prev_i)
		testing.expect_value(t, iw[11], want_post_i)

		want_prev_u := u32(0xff + 0x10 + 0x55 + 88 + 34 + 34)
		want_post_u := u32((0xff & 0x3c) + (0x10 | 0x05) * 100 + (0x55 ~ 0x0f) * 10000 + 91 + 67)
		testing.expect_value(t, uw[1], u32(5))
		testing.expect_value(t, uw[2], u32(0x80000020))
		testing.expect_value(t, uw[3], u32(0))
		testing.expect_value(t, uw[4], u32(5))
		testing.expect_value(t, uw[5], u32(0x80000020))
		testing.expect_value(t, uw[6], u32(3))
		testing.expect_value(t, uw[7], want_prev_u)
		testing.expect_value(t, uw[8], u32(0xff & 0x3c))
		testing.expect_value(t, uw[9], u32(0x10 | 0x05))
		testing.expect_value(t, uw[10], u32(0x55 ~ 0x0f))
		testing.expect_value(t, uw[11], u32(91))
		testing.expect_value(t, uw[12], u32(67))
		testing.expect_value(t, uw[13], want_prev_u)
		testing.expect_value(t, uw[14], u32(10 * 100 + 2 + 40))
		testing.expect_value(t, uw[15], u32(5 * 100 + 5 + 35 + 35 + 39))
		testing.expect_value(t, uw[336], want_post_u)
		for tid in u32(0) ..< 160 {
			testing.expect_value(t, uw[16 + tid], tid)
			testing.expect_value(t, uw[176 + tid], tid)
		}
	}

	// 2. Compile, validate SPIR-V opcodes, verify MSL temporaries & constructs, and run all 11 Phase 2 testdata shaders.
	spv_has_op :: proc(words: []u32, want_op: u32) -> bool {
		i := 5
		for i < len(words) {
			op := words[i] & 0xffff
			wc := int(words[i] >> 16)
			if op == want_op {
				return true
			}
			if wc <= 0 {
				break
			}
			i += wc
		}
		return false
	}
	for name in ([11]string{
		"tile_max.comp", "tile_max_odd.comp", "histogram.comp", "grid_bin.comp", "atomics.comp",
		"probe_65.comp", "probe_eval.comp", "probe_inout.comp", "probe_loop.comp", "probe_old.comp", "probe_return.comp",
	}) {
		path := fmt.tprintf("testdata/%s", name)
		src_bytes, rerr := os.read_entire_file(path, context.temp_allocator)
		testing.expectf(t, rerr == nil, "failed to read %s", path)
		if rerr != nil {
			continue
		}
		m, errs, ok := compile(string(src_bytes), {profile = .Glsl, stage = .Compute})
		defer destroy(m)
		testing.expectf(t, ok, "%s failed:\n%s", name, format_errors(errs, context.temp_allocator))
		if !ok {
			continue
		}
		if have_spirv_val() {
			expect_valid_spirv(t, name, m.spirv)
		}
		testing.expectf(t, strings.contains(m.msl, "threadgroup uint oksl_sh["), "%s missing oksl_sh", name)
		testing.expectf(t, strings.contains(m.msl, "threadgroup_barrier("), "%s missing threadgroup_barrier", name)
		if name == "grid_bin.comp" {
			// Discarded atomic calls must not allocate volatile temporaries.
			testing.expectf(t, !strings.contains(m.msl, "volatile "), "%s emitted unexpected volatile for discarded atomics", name)
		}
		if name == "histogram.comp" {
			// 1 used atomic result + 4 discarded atomic statements -> exactly 1 volatile local.
			testing.expect_value(t, strings.count(m.msl, "volatile uint _t"), 1)
		}
		if name == "probe_eval.comp" || name == "probe_old.comp" {
			testing.expectf(t, strings.contains(m.msl, "volatile uint _t"), "%s missing volatile local workaround", name)
		}
		if name == "atomics.comp" {
			for spv_op in ([?]u32{
				224, // OpControlBarrier
				225, // OpMemoryBarrier
				229, // OpAtomicExchange
				230, // OpAtomicCompareExchange
				234, // OpAtomicIAdd
				236, // OpAtomicSMin
				237, // OpAtomicUMin
				238, // OpAtomicSMax
				239, // OpAtomicUMax
				240, // OpAtomicAnd
				241, // OpAtomicOr
				242, // OpAtomicXor
			}) {
				testing.expectf(t, spv_has_op(m.spirv, spv_op), "atomics.comp missing SPIR-V op %d", spv_op)
			}
			for msl_sub in ([?]string{
				"threadgroup_barrier(mem_flags::mem_threadgroup)",
				"threadgroup_barrier(mem_flags::mem_device)",
				"threadgroup_barrier(mem_flags::mem_texture)",
				"threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture)",
				"(threadgroup atomic_int*)&oksl_sh[",
				"(threadgroup atomic_uint*)&oksl_sh[",
				"(device atomic_int*)&oksl_buf_",
				"(device atomic_uint*)&oksl_buf_",
				"atomic_fetch_add_explicit(",
				"atomic_fetch_min_explicit(",
				"atomic_fetch_max_explicit(",
				"atomic_fetch_and_explicit(",
				"atomic_fetch_or_explicit(",
				"atomic_fetch_xor_explicit(",
				"atomic_exchange_explicit(",
				"oksl_cas(",
			}) {
				testing.expectf(t, strings.contains(m.msl, msl_sub), "atomics.comp missing MSL %q", msl_sub)
			}
		}
		// Verify every declared `_tN` temporary (plain or volatile) in `ok_main` is assigned.
		if main_pos := strings.index(m.msl, "void ok_main("); main_pos >= 0 {
			rest_msl := m.msl[main_pos:]
			end_fn := strings.index(rest_msl, "\n}\n")
			fn_msl := rest_msl[:end_fn] if end_fn >= 0 else rest_msl
			for line in strings.split_lines(fn_msl, context.temp_allocator) {
				trimmed := strings.trim_space(line)
				is_plain_decl := strings.has_suffix(trimmed, " = {};")
				is_vol_decl := strings.has_prefix(trimmed, "volatile ") && strings.has_suffix(trimmed, ";")
				if ix := strings.index(trimmed, " _t"); ix >= 0 && (is_plain_decl || is_vol_decl) {
					rest := trimmed[ix + 1 : len(trimmed) - 1]
					eq := strings.index(rest, " = {}")
					tname := rest[:eq] if eq >= 0 else rest
					testing.expectf(t, strings.contains(fn_msl, fmt.tprintf("(%s = ", tname)), "%s: MSL temp %s declared but unassigned", name, tname)
				}
			}
		}

		// Dispatch a small run on CPU to exercise every instruction path.
		bufs := make([][]u32, len(m.storage_buffers), context.temp_allocator)
		for sb, bi in m.storage_buffers {
			bufs[bi] = make([]u32, 8192, context.temp_allocator)
			if sb.readonly {
				for &w, wi in bufs[bi] {
					w = u32((wi * 73 + 19) & 255)
				}
			}
		}
		ublk := make([]u8, m.block_size, context.temp_allocator)
		set_uniform(m, ublk, "uSize", {32, 32})
		set_uniform(m, ublk, "uCount", {64})
		set_uniform(m, ublk, "uGridSize", {4, 4})
		run_compute(&m.program, ublk, {2, 2, 1}, temp_scratch(), {buffers = bufs})
	}
}

@(test)
test_run_fragment_targets_and_shapes :: proc(t: ^testing.T) {
	MAX_Q :: 17
	MAX_PX :: MAX_Q * 4
	SENTINEL :: [4]f32{-9, -11, -13, -17}
	MASK_SENTINEL :: u32(0xDEAD_BEEF)
	SHADED_SENTINEL :: transmute(Quad_Lanes)u8(0x0F)

	// 1. Multi-target vec4 shader with non-contiguous varyings (2 spans), gl_FragCoord, gl_FrontFacing, derivatives, and discard.
	{
		src := `#version 310 es
precision highp float;
layout(location = 0) in vec2 vA;
layout(location = 3) in vec3 vB;
uniform int uDiscardMask;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
void main() {
    float dx = dFdx(vA.x) + dFdy(vB.y);
    int laneId = int(vB.z);
    if (((uDiscardMask >> laneId) & 1) != 0) {
        discard;
    }
    float ff = gl_FrontFacing ? 1.0 : -1.0;
    o0 = vec4(vA, gl_FragCoord.xy * 0.125);
    o1 = vec4(vB.xy, gl_FragCoord.z, gl_FragCoord.w * ff);
    o2 = vec4(dx, vA.x + vB.x, vA.y - vB.y, ff);
    o3 = vec4(float(laneId), vB.z * 0.5, gl_FragCoord.x + gl_FragCoord.z, 2.5);
}`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
		p_int := m.program
		p_int.jit_body, p_int.jit_init = nil, nil

		// Determine varying column offsets from `m.program.inputs`.
		off_a, off_b: int
		for iv in m.program.inputs {
			if iv.kind == .Stage_In && iv.location == 0 {
				off_a = int(iv.offset)
			} else if iv.kind == .Stage_In && iv.location == 3 {
				off_b = int(iv.offset)
			}
		}
		vfloats := m.varying_floats

		origin: [MAX_Q][2]i32
		helpers_arr: [MAX_Q]Quad_Lanes
		ff_arr: [MAX_Q]bool
		col_a0, col_a1, col_b0, col_b1, col_b2, depth_col: [MAX_PX]f32
		for q in 0 ..< MAX_Q {
			origin[q] = {i32(q * 2 + 4), i32(10 + (q % 5) * 2)}
			// Mix: no helpers (0x0), partial (0xA, 0x5, 0x1), and full-helper quad (0xF at q == 2).
			h_pat := [5]u8{0x0, 0xA, 0xF, 0x1, 0x6}
			helpers_arr[q] = transmute(Quad_Lanes)h_pat[q % len(h_pat)]
			ff_arr[q] = (q % 3) != 1
			for c in 0 ..< 4 {
				idx := q * 4 + c
				col_a0[idx] = f32(idx) * 0.25 + 1.0
				col_a1[idx] = f32(idx) * -0.5 + 3.0
				col_b0[idx] = f32(q) * 1.5 + f32(c) * 0.125
				col_b1[idx] = f32(c * c) * 0.75 - f32(q) * 0.25
				col_b2[idx] = f32(idx % 20)
				depth_col[idx] = f32(idx + 1) * 0.03125
			}
		}

		ublk := make([]u8, m.block_size, context.temp_allocator)
		q_counts := [?]int{0, 1, 2, 3, 5, 17}
		discard_masks := [?]i32{
			0,
			0b0010_0101, // partial quads
			0x0000_000F, // full quad 0 discarded inside pair (0, 1)
			0x0000_00FF, // full 2-quad pair (0, 1) discarded
			0x0005_5555, // alternating lanes across many quads
		}
		// Target modes: 0=nil, 1..4=1..4 targets, 5=sparse {nil, t1, nil, t3}, 6=unbound 5th target, 7=9 targets (> FRAGMENT_OUTPUTS).
		for qc in q_counts {
			total := qc * 4
			vars := make([]f32, vfloats * total, context.temp_allocator)
			if total > 0 {
				copy(vars[(off_a + 0) * total:], col_a0[:total])
				copy(vars[(off_a + 1) * total:], col_a1[:total])
				copy(vars[(off_b + 0) * total:], col_b0[:total])
				copy(vars[(off_b + 1) * total:], col_b1[:total])
				copy(vars[(off_b + 2) * total:], col_b2[:total])
			}
			for use_helpers in ([2]bool{false, true}) {
				for use_depth_ff in ([2]bool{false, true}) {
					for use_smask in ([2]bool{false, true}) {
						for dmask in discard_masks {
							set_uniform(m, ublk, "uDiscardMask", {f32(dmask)})
							for mode in 0 ..< 8 {
								nt := mode
								if mode == 5 {
									nt = 4
								} else if mode == 6 {
									nt = 5
								} else if mode == 7 {
									nt = 9
								}
								buf_j := make([][MAX_PX][4]f32, nt, context.temp_allocator)
								buf_i := make([][MAX_PX][4]f32, nt, context.temp_allocator)
								tj := make([][][4]f32, nt, context.temp_allocator)
								ti := make([][][4]f32, nt, context.temp_allocator)
								for k in 0 ..< nt {
									for p in 0 ..< MAX_PX {
										buf_j[k][p] = SENTINEL
										buf_i[k][p] = SENTINEL
									}
									if mode == 5 && (k == 0 || k == 2) {
										continue
									}
									tj[k] = buf_j[k][:total]
									ti[k] = buf_i[k][:total]
								}
								quads := Fragment_Quads {
									count        = qc,
									origin       = origin[:qc],
									helpers      = helpers_arr[:qc] if use_helpers else nil,
									depth        = depth_col[:total] if use_depth_ff else nil,
									front_facing = ff_arr[:qc] if use_depth_ff else nil,
									varyings     = vars,
								}
								shaded_j, shaded_i: [MAX_Q]Quad_Lanes
								for q in 0 ..< MAX_Q {
									shaded_j[q] = SHADED_SENTINEL
									shaded_i[q] = SHADED_SENTINEL
								}
								sm_j, sm_i: [MAX_PX]u32
								for p in 0 ..< MAX_PX {
									sm_j[p] = MASK_SENTINEL
									sm_i[p] = MASK_SENTINEL
								}
								run_fragment(&m.program, ublk, quads, tj if nt > 0 else nil, shaded_j[:qc], temp_scratch(), {}, sm_j[:total] if use_smask else nil)
								run_fragment(&p_int, ublk, quads, ti if nt > 0 else nil, shaded_i[:qc], temp_scratch(), {}, sm_i[:total] if use_smask else nil)
								if !slice.equal(shaded_j[:], shaded_i[:]) {
									testing.expectf(t, false, "shaded mismatch qc=%d mode=%d h=%v df=%v sm=%v d=%x", qc, mode, use_helpers, use_depth_ff, use_smask, dmask)
									return
								}
								for k in 0 ..< nt {
									bj := slice.reinterpret([]u32, buf_j[k][:])
									bi := slice.reinterpret([]u32, buf_i[k][:])
									if !slice.equal(bj, bi) {
										testing.expectf(t, false, "target[%d] mismatch qc=%d mode=%d h=%v df=%v sm=%v d=%x", k, qc, mode, use_helpers, use_depth_ff, use_smask, dmask)
										return
									}
								}
								if mode == 6 && !use_smask {
									// Location 4 is unbound in shader -> shaded lanes receive {0, 0, 0, 1}, others stay SENTINEL.
									for q in 0 ..< qc {
										hm := helpers_arr[q] if use_helpers else {}
										for c in 0 ..< 4 {
											idx := q * 4 + c
											disc := ((u32(dmask) >> u32(idx % 20)) & 1) != 0
											want := SENTINEL if (c in hm || disc) else [4]f32{0, 0, 0, 1}
											testing.expect_value(t, buf_j[4][idx], want)
										}
									}
								}
								if !slice.equal(sm_j[:], sm_i[:]) {
									testing.expectf(t, false, "sample_masks mismatch qc=%d mode=%d", qc, mode)
									return
								}
							}
						}
					}
				}
			}
		}
	}

	// 2. Narrow float outputs: float -> {r, 0, 0, 1}, vec2 -> {r, g, 0, 1}, vec3 -> {r, g, b, 1}.
	{
		src := `#version 310 es
precision highp float;
layout(location = 0) in float vX;
layout(location = 0) out float o1;
layout(location = 1) out vec2 o2;
layout(location = 2) out vec3 o3;
void main() {
    o1 = vX + 1.5;
    o2 = vec2(vX * 2.0, vX - 3.0);
    o3 = vec3(vX, -vX, vX * 0.5 + 4.0);
}`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "narrow compile failed:\n%s", format_errors(errs, context.temp_allocator))
		p_int := m.program
		p_int.jit_body, p_int.jit_init = nil, nil

		origin := [3][2]i32{{0, 0}, {2, 0}, {4, 0}}
		helpers := [3]Quad_Lanes{{1}, {}, {0, 3}}
		vx: [12]f32
		for i in 0 ..< 12 {
			vx[i] = f32(i) * 1.25 - 2.0
		}
		for qc in ([2]int{2, 3}) {
			for use_h in ([2]bool{false, true}) {
				for nt in 1 ..= 3 {
					bj, bi: [3][12][4]f32
					for k in 0 ..< 3 {
						for p in 0 ..< 12 {
							bj[k][p] = SENTINEL
							bi[k][p] = SENTINEL
						}
					}
					tj := make([][][4]f32, nt, context.temp_allocator)
					ti := make([][][4]f32, nt, context.temp_allocator)
					for k in 0 ..< nt {
						tj[k] = bj[k][:qc * 4]
						ti[k] = bi[k][:qc * 4]
					}
					sh_j, sh_i: [3]Quad_Lanes
					quads := Fragment_Quads{count = qc, origin = origin[:qc], helpers = helpers[:qc] if use_h else nil, varyings = vx[:qc * 4]}
					run_fragment(&m.program, nil, quads, tj, sh_j[:qc], temp_scratch())
					run_fragment(&p_int, nil, quads, ti, sh_i[:qc], temp_scratch())
					testing.expect(t, slice.equal(sh_j[:], sh_i[:]))
					for k in 0 ..< nt {
						testing.expect(t, slice.equal(slice.reinterpret([]u32, bj[k][:]), slice.reinterpret([]u32, bi[k][:])))
					}
					for q in 0 ..< qc {
						hm := helpers[q] if use_h else {}
						for c in 0 ..< 4 {
							idx := q * 4 + c
							x := vx[idx]
							if c in hm {
								testing.expect_value(t, bj[0][idx], SENTINEL)
							} else {
								testing.expect_value(t, bj[0][idx], [4]f32{x + 1.5, 0, 0, 1})
								if nt >= 2 {
									testing.expect_value(t, bj[1][idx], [4]f32{x * 2.0, x - 3.0, 0, 1})
								}
								if nt >= 3 {
									testing.expect_value(t, bj[2][idx], [4]f32{x, -x, x * 0.5 + 4.0, 1})
								}
							}
						}
					}
				}
			}
		}
	}

	// 3. Integer fragment outputs (ivec2, uvec4) and explicit gl_SampleMask[0] write.
	{
		src := `#version 310 es
precision highp float;
precision highp int;
layout(location = 0) in float vX;
layout(location = 0) out ivec2 oI;
layout(location = 1) out uvec4 oU;
void main() {
    int k = int(vX);
    gl_SampleMask[0] = (1 << (k & 7)) | 0x100;
    oI = ivec2(-k * 3, k + 100);
    oU = uvec4(uint(k), uint(k * 17 + 5), 0x80000001u, 0xffffffffu);
}`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "int-out compile failed:\n%s", format_errors(errs, context.temp_allocator))
		p_int := m.program
		p_int.jit_body, p_int.jit_init = nil, nil

		origin := [3][2]i32{{0, 0}, {2, 0}, {4, 0}}
		helpers := [3]Quad_Lanes{{2}, {}, {0, 1}}
		vx: [12]f32
		for i in 0 ..< 12 {
			vx[i] = f32(i + 1)
		}
		for use_sm in ([2]bool{false, true}) {
			bj, bi: [2][12][4]f32
			sm_j, sm_i: [12]u32
			sh_j, sh_i: [3]Quad_Lanes
			for p in 0 ..< 12 {
				bj[0][p], bj[1][p] = SENTINEL, SENTINEL
				bi[0][p], bi[1][p] = SENTINEL, SENTINEL
				sm_j[p], sm_i[p] = MASK_SENTINEL, MASK_SENTINEL
			}
			tj := [2][][4]f32{bj[0][:], bj[1][:]}
			ti := [2][][4]f32{bi[0][:], bi[1][:]}
			quads := Fragment_Quads{count = 3, origin = origin[:], helpers = helpers[:], varyings = vx[:]}
			run_fragment(&m.program, nil, quads, tj[:], sh_j[:], temp_scratch(), {}, sm_j[:] if use_sm else nil)
			run_fragment(&p_int, nil, quads, ti[:], sh_i[:], temp_scratch(), {}, sm_i[:] if use_sm else nil)
			testing.expect(t, slice.equal(sh_j[:], sh_i[:]))
			testing.expect(t, slice.equal(slice.reinterpret([]u32, bj[0][:]), slice.reinterpret([]u32, bi[0][:])))
			testing.expect(t, slice.equal(slice.reinterpret([]u32, bj[1][:]), slice.reinterpret([]u32, bi[1][:])))
			testing.expect(t, slice.equal(sm_j[:], sm_i[:]))
		}
	}

	// 4. Odd quad counts (1 and 3 quads) execute only the 4 real lanes on the trailing quad for both texture() and texelFetch().
	{
		src := `#version 310 es
precision highp float;
in vec2 vUV;
uniform sampler2D s0;
uniform sampler2D s1;
layout(location = 0) out vec4 o;
void main() {
    vec4 a = texture(s0, vUV);
    vec4 b = texelFetch(s1, ivec2(gl_FragCoord.xy), 0);
    o = a + b;
}`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "odd-quad sampler compile failed:\n%s", format_errors(errs, context.temp_allocator))
		p_int := m.program
		p_int.jit_body, p_int.jit_init = nil, nil

		Odd_Sampler_State :: struct {
			n_pixels: int,
			calls:    [2]int,
			bad_uv:   int,
			expected: [2][12][2]f32,
			seen:     [2][12]int,
		}
		odd_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			st := (^Odd_Sampler_State)(user)
			if channel < 0 || channel >= 2 {
				st.bad_uv += 1
				return {}
			}
			st.calls[channel] += 1
			matched := false
			for p in 0 ..< st.n_pixels {
				if st.expected[channel][p] == uv {
					st.seen[channel][p] += 1
					matched = true
					break
				}
			}
			if !matched {
				st.bad_uv += 1
			}
			return {uv.x, uv.y, f32(channel), 1}
		}

		origins := [3][2]i32{{2, 0}, {4, 0}, {6, 2}}
		sizes := [2][3]i32{{16, 16, 1}, {16, 16, 1}}
		scratch := temp_scratch()
		for count in ([2]int{1, 3}) {
			n_px := count * 4
			vars := make([]f32, 2 * n_px, context.temp_allocator)
			st: Odd_Sampler_State
			st.n_pixels = n_px
			for q in 0 ..< count {
				for i in 0 ..< 4 {
					p := q * 4 + i
					u := (f32(p) + 1.5) / 16.0
					v := f32(0.375)
					vars[0 * n_px + p] = u
					vars[1 * n_px + p] = v
					st.expected[0][p] = {u, v}
					px := origins[q].x + i32(i & 1)
					py := origins[q].y + i32((i >> 1) & 1)
					st.expected[1][p] = {(f32(px) + 0.5) / 16.0, (f32(py) + 0.5) / 16.0}
				}
			}
			tj_buf, ti_buf: [12][4]f32
			sh_j, sh_i: [3]Quad_Lanes
			tj := [1][][4]f32{tj_buf[:n_px]}
			ti := [1][][4]f32{ti_buf[:n_px]}
			quads := Fragment_Quads{count = count, origin = origins[:count], varyings = vars}
			opts := Run_Options{sampler = odd_cb, user = &st, channel_sizes = sizes[:]}
			run_fragment(&m.program, nil, quads, tj[:], sh_j[:count], scratch, opts)
			testing.expect_value(t, st.calls[0], n_px)
			testing.expect_value(t, st.calls[1], n_px)
			testing.expect_value(t, st.bad_uv, 0)
			for p in 0 ..< n_px {
				testing.expect_value(t, st.seen[0][p], 1)
				testing.expect_value(t, st.seen[1][p], 1)
			}

			st_int := st
			st_int.calls = {}
			st_int.bad_uv = 0
			st_int.seen = {}
			opts_int := Run_Options{sampler = odd_cb, user = &st_int, channel_sizes = sizes[:]}
			run_fragment(&p_int, nil, quads, ti[:], sh_i[:count], scratch, opts_int)
			testing.expect_value(t, st_int.calls[0], n_px)
			testing.expect_value(t, st_int.calls[1], n_px)
			testing.expect_value(t, st_int.bad_uv, 0)
			testing.expect(t, slice.equal(sh_j[:count], sh_i[:count]))
			testing.expect(t, slice.equal(slice.reinterpret([]u32, tj_buf[:n_px]), slice.reinterpret([]u32, ti_buf[:n_px])))
		}
	}
}

@(test)
test_jit_init_const_and_mixed :: proc(t: ^testing.T) {
	scratch := temp_scratch()

	// 1. Pure-constant hoisted init matches interpreter.
	{
		src := `#version 310 es
precision highp float;
layout(location = 0) out vec4 o;
void main() {
    mat2 m = mat2(1.5, -2.0, 3.0, 0.5) * mat2(2.0, 1.0, -0.5, 4.0);
    vec2 p = m * (gl_FragCoord.xy * 0.25 + vec2(0.125, -0.375));
    float s = sin(0.5) + cos(1.25) + exp2(1.5);
    o = vec4(p, s, m[0][1] + m[1][0]);
}`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "pure-const init compile failed:\n%s", format_errors(errs, context.temp_allocator))
		oj, oi: [12][4]f32
		run_f32(&m.program, nil, 4, 3, oj[:], scratch)
		run_f32_interp(&m.program, nil, 4, 3, oi[:], scratch, {})
		testing.expect(t, slice.equal(slice.reinterpret([]u32, oj[:]), slice.reinterpret([]u32, oi[:])))
	}

	// 2. Mixed constant + uniform/buffer dynamic init reused across multiple draws on the same Scratch.
	{
		src := `#version 310 es
precision highp float;
uniform vec2 uScale;
layout(std430, binding = 0) readonly buffer B { float b[]; };
layout(location = 0) out vec4 o;
void main() {
    float c = sin(0.75) * 4.0 + 2.5;
    vec2 u = uScale * 3.0 + vec2(c, -c);
    float h = b[0] * uScale.x + b[1];
    o = vec4(u + gl_FragCoord.xy * 0.125, h, c);
}`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "mixed init compile failed:\n%s", format_errors(errs, context.temp_allocator))
		when JIT_ENABLED {
			testing.expect(t, m.program.jit_init != nil)
		}
		ublk := make([]u8, m.block_size, context.temp_allocator)
		for step in 0 ..< 4 {
			set_uniform(m, ublk, "uScale", {f32(step) * 1.5 - 1.0, f32(step) * -0.75 + 2.0})
			bdata := [2]f32{f32(step * 3 + 1) * 0.5, f32(step * 5 - 2) * 0.25}
			buf_u32 := slice.reinterpret([]u32, bdata[:])
			bufs := [1][]u32{buf_u32}
			oj, oi: [12][4]f32
			run_f32(&m.program, ublk, 4, 3, oj[:], scratch, {buffers = bufs[:]})
			run_f32_interp(&m.program, ublk, 4, 3, oi[:], scratch, {buffers = bufs[:]})
			testing.expectf(t, slice.equal(slice.reinterpret([]u32, oj[:]), slice.reinterpret([]u32, oi[:])), "mixed init mismatch at step %d", step)
		}
	}

	// 3. Reusing the same Scratch from bound channel_images -> unbound callback sampler -> partial bound channel_images.
	{
		src := `#version 310 es
precision highp float;
uniform sampler2D s0;
uniform sampler2D s1;
layout(location = 0) out vec4 o;
void main() {
    o = textureLod(s0, vec2(0.5), 0.0) + textureLod(s1, vec2(0.5), 0.0) * 2.0;
}`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "chan reuse compile failed:\n%s", format_errors(errs, context.temp_allocator))

		tex0 := [4]f32{1, 2, 3, 4}
		tex1 := [4]f32{10, 20, 30, 40}
		cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			f := f32(channel + 1) * 100.0
			return {f, f + 1, f + 2, 1}
		}
		both_imgs := [2]Channel_Image{
			{data = tex0[:], width = 1, height = 1, components = 4},
			{data = tex1[:], width = 1, height = 1, components = 4},
		}
		one_img := [1]Channel_Image{
			{data = tex0[:], width = 1, height = 1, components = 4},
		}
		for opts in ([3]Run_Options{
			{sampler = cb, channel_images = both_imgs[:]},
			{sampler = cb},
			{sampler = cb, channel_images = one_img[:]},
		}) {
			oj, oi: [1][4]f32
			run_f32(&m.program, nil, 1, 1, oj[:], scratch, opts)
			run_f32_interp(&m.program, nil, 1, 1, oi[:], scratch, opts)
			testing.expect_value(t, oj[0], oi[0])
		}
	}
}

@(test)
test_jit_idiv_irem :: proc(t: ^testing.T) {
	EDGES_N :: 33
	edges := [EDGES_N]u32 {
		0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 255, 65535, 65536, 65537, 1_000_003,
		0x7FFF_FFFD, 0x7FFF_FFFE, 0x7FFF_FFFF,
		0x8000_0000, 0x8000_0001, 0x8000_0002,
		0xAAAA_AAAA, 0x5555_5555, 0xC000_0000, 0xFFFF_0000,
		0xFFFF_FFF9, 0xFFFF_FFFD, 0xFFFF_FFFE, 0xFFFF_FFFF,
	}
	EDGE_PAIRS :: EDGES_N * EDGES_N
	RAND_PAIRS :: 262_144
	W :: 512
	H :: (EDGE_PAIRS + RAND_PAIRS + W - 1) / W
	TOTAL :: W * H

	in_ab := make([]u32, TOTAL * 2, context.temp_allocator)
	p := 0
	for a in edges {
		for b in edges {
			in_ab[p * 2 + 0] = a
			in_ab[p * 2 + 1] = b
			p += 1
		}
	}
	rng := u64(0x9E37_79B9_7F4A_7C15)
	next_u32 :: #force_inline proc(s: ^u64) -> u32 {
		x := s^
		x ~= x << 13
		x ~= x >> 7
		x ~= x << 17
		s^ = x
		return u32(x >> 32) ~ u32(x)
	}
	for ; p < TOTAL; p += 1 {
		a := next_u32(&rng)
		b := next_u32(&rng)
		// Bias part of the stream toward high-bit u32s, small divisors, and signed extremes.
		switch p & 7 {
		case 1:
			a |= 0x8000_0000
			b = b & 7 // includes 0, 1, 2..7
		case 2:
			b |= 0x8000_0000
		case 3:
			a = 0x8000_0000 + (a & 3) - 1
			b = 0xFFFF_FFFF - (b & 3) // includes -1, -2, -3, -4
		case 4:
			b = (b % 31)
		}
		in_ab[p * 2 + 0] = a
		in_ab[p * 2 + 1] = b
	}

	// Fragment shader (compiled by x86-64 AVX2 JIT): scalar + vector (ivec4/uvec4) + live BB cache + init-hoisted div/rem.
	src := `#version 310 es
precision highp float;
precision highp int;
uniform ivec2 uInit;
layout(std430, binding = 0) readonly buffer In { uint ab[]; };
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;

vec2 pack32(uint v) {
    return vec2(float(v & 0xFFFFu), float(v >> 16u));
}

void main() {
    uint idx = uint(int(gl_FragCoord.y) * 512 + int(gl_FragCoord.x));
    uint ua = ab[idx * 2u + 0u];
    uint ub = ab[idx * 2u + 1u];
    int ia = int(ua);
    int ib = int(ub);

    // Uniform-only divisions hoist into jit_init.
    uint init_mix = uint(uInit.x / uInit.y) ^ (uint(uInit.x % uInit.y) << 8u)
        ^ (uint(uInit.x) / uint(uInit.y)) ^ ((uint(uInit.x) % uint(uInit.y)) << 16u);

    // Keep live values around the divisions to stress the BB register cache (ymm8..15).
    uint keep0 = ua ^ 0x13579BDFu;
    int keep1 = ia + 12345;
    int iq = ia / ib;
    int ir = ia % ib;
    uint uq = ua / ub;
    uint ur = ua % ub;

    ivec4 iv_a = ivec4(ia, -ia, ia ^ 0x55555555, ib);
    ivec4 iv_b = ivec4(ib, ib, -ib, ia);
    ivec4 iv_q = iv_a / iv_b;
    ivec4 iv_r = iv_a % iv_b;

    uvec4 uv_a = uvec4(ua, ~ua, ua ^ 0x80000000u, ub);
    uvec4 uv_b = uvec4(ub, ub | 1u, ub >> 1u, ua);
    uvec4 uv_q = uv_a / uv_b;
    uvec4 uv_r = uv_a % uv_b;

    uint vec_mix = uint(iv_q.x ^ iv_q.y ^ iv_q.z ^ iv_q.w)
        ^ (uint(iv_r.x ^ iv_r.y ^ iv_r.z ^ iv_r.w) * 3u)
        ^ (uv_q.x ^ (uv_q.y << 1u) ^ (uv_q.z >> 1u) ^ uv_q.w)
        ^ ((uv_r.x ^ uv_r.y ^ uv_r.z ^ uv_r.w) * 7u)
        ^ (keep0 + uint(keep1));

    o0 = vec4(pack32(uint(iq)), pack32(uint(ir)));
    o1 = vec4(pack32(uq), pack32(ur));
    o2 = vec4(pack32(vec_mix), pack32(init_mix));
}`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "idiv fragment compile failed:\n%s", format_errors(errs, context.temp_allocator))
	when JIT_ENABLED {
		testing.expect(t, m.program.jit_body != nil)
	}

	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil

	ublk := make([]u8, m.block_size, context.temp_allocator)
	set_uniform(m, ublk, "uInit", {-100_003, 37})
	bufs := [1][]u32{in_ab}

	t0_j := make([][4]f32, TOTAL, context.temp_allocator)
	t1_j := make([][4]f32, TOTAL, context.temp_allocator)
	t2_j := make([][4]f32, TOTAL, context.temp_allocator)
	t0_i := make([][4]f32, TOTAL, context.temp_allocator)
	t1_i := make([][4]f32, TOTAL, context.temp_allocator)
	t2_i := make([][4]f32, TOTAL, context.temp_allocator)
	tj := [3][][4]f32{t0_j, t1_j, t2_j}
	ti := [3][][4]f32{t0_i, t1_i, t2_i}

	run_targets(&m.program, ublk, W, H, tj[:], temp_scratch(), {buffers = bufs[:]})
	run_targets(&p_int, ublk, W, H, ti[:], temp_scratch(), {buffers = bufs[:]})

	unpack32 :: #force_inline proc(lo, hi: f32) -> u32 {
		return u32(lo) | (u32(hi) << 16)
	}
	read_row :: #force_inline proc(t0, t1: [][4]f32, idx: int) -> [4]u32 {
		c0, c1 := t0[idx], t1[idx]
		return {unpack32(c0.x, c0.y), unpack32(c0.z, c0.w), unpack32(c1.x, c1.y), unpack32(c1.z, c1.w)}
	}

	for idx in 0 ..< TOTAL {
		ua := in_ab[idx * 2 + 0]
		ub := in_ab[idx * 2 + 1]
		rj := read_row(t0_j, t1_j, idx)
		ri := read_row(t0_i, t1_i, idx)
		vj := unpack32(t2_j[idx].x, t2_j[idx].y)
		vi := unpack32(t2_i[idx].x, t2_i[idx].y)
		hj := unpack32(t2_j[idx].z, t2_j[idx].w)
		hi := unpack32(t2_i[idx].z, t2_i[idx].w)
		if rj != ri || vj != vi || hj != hi {
			testing.expectf(
				t,
				false,
				"idiv/irem JIT != interp at idx=%d (a=0x%08x, b=0x%08x): jit=%v/%08x/%08x interp=%v/%08x/%08x",
				idx, ua, ub, rj, vj, hj, ri, vi, hi,
			)
			return
		}
		if rj[0] != bin_lane(ua, ub, .I_Div) ||
		   rj[1] != bin_lane(ua, ub, .I_Rem) ||
		   rj[2] != bin_lane(ua, ub, .U_Div) ||
		   rj[3] != bin_lane(ua, ub, .U_Rem) {
			testing.expectf(t, false, "bin_lane mismatch at idx=%d (a=0x%08x, b=0x%08x): got %v", idx, ua, ub, rj)
			return
		}
	}

	// Explicit marquee checks on the edge pairs at the front of `in_ab`.
	find_edge :: proc(edges: []u32, t0, t1: [][4]f32, a, b: u32) -> [4]u32 {
		for ea, ia in edges {
			if ea == a {
				for eb, ib in edges {
					if eb == b {
						return read_row(t0, t1, ia * len(edges) + ib)
					}
				}
			}
		}
		return {}
	}
	testing.expect_value(t, find_edge(edges[:], t0_j, t1_j, 0x8000_0000, 0), [4]u32{0, 0, 0, 0})
	testing.expect_value(t, find_edge(edges[:], t0_j, t1_j, 0xFFFF_FFFF, 0), [4]u32{0, 0, 0, 0})
	testing.expect_value(t, find_edge(edges[:], t0_j, t1_j, 0x8000_0000, 0xFFFF_FFFF), [4]u32{0x8000_0000, 0, 0, 0x8000_0000})
	testing.expect_value(t, find_edge(edges[:], t0_j, t1_j, 0xFFFF_FFFF, 1), [4]u32{0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0})
	testing.expect_value(t, find_edge(edges[:], t0_j, t1_j, 0x8000_0000, 1), [4]u32{0x8000_0000, 0, 0x8000_0000, 0})
	testing.expect_value(t, find_edge(edges[:], t0_j, t1_j, 0xFFFF_FFFF, 2), [4]u32{0, 0xFFFF_FFFF, 0x7FFF_FFFF, 1})
	testing.expect_value(t, find_edge(edges[:], t0_j, t1_j, 0xFFFF_FFFF, 0x8000_0000), [4]u32{0, 0xFFFF_FFFF, 1, 0x7FFF_FFFF})
}

@(test)
test_run_targets_region_sized :: proc(t: ^testing.T) {
	src := `#version 310 es
precision highp float;
layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vVar;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec2 o2;
void main() {
    int ix = int(gl_FragCoord.x);
    int iy = int(gl_FragCoord.y);
    if ((ix >= 16 && ix < 24 && ((ix + iy) & 3) == 1) || (ix == 2 && iy == 3)) {
        discard;
    }
    float curve = vUV.x * vUV.x * 13.0 + vUV.y * gl_FragCoord.x * 0.25 + vVar.x * vVar.y;
    float dx = dFdx(curve);
    float dy = dFdy(curve);
    float fw = fwidth(vUV.x * gl_FragCoord.y * 0.125 + vVar.z);
    o0 = vec4(gl_FragCoord.xy * 0.0625, dx, dy);
    o1 = vec4(vUV, fw, vVar.x + vVar.y + gl_FragCoord.w);
    o2 = vec2(dx - dy + vVar.z, gl_FragCoord.x + gl_FragCoord.y * 0.5);
}`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "region_targets shader compile failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil

	varyings := [2]Varying {
		{}, // populated per layer size below
		{a = {0.25, -0.5, 1.25, 0}, dx = {0.03125, 0.015625, -0.0078125, 0}, dy = {-0.015625, 0.0625, 0.03125, 0}},
	}
	SENTINEL :: [4]f32{-12345.0, 67890.0, -42.5, 999.25}
	scratch := temp_scratch()

	Region_Case :: struct {
		x, y, w, h: int,
	}

	for dims in ([2][2]int{{40, 70}, {37, 67}}) {
		W, H := dims[0], dims[1]
		varyings[0] = {
			a  = {0, 0, 0, 0},
			dx = {1.0 / f32(W), 0, 0, 0},
			dy = {0, 1.0 / f32(H), 0, 0},
		}
		for use_vars in ([2]bool{false, true}) {
			vars_slice := varyings[:] if use_vars else nil

			ref_j, ref_i: [3][][4]f32
			for k in 0 ..< 3 {
				ref_j[k] = make([][4]f32, W * H, context.temp_allocator)
				ref_i[k] = make([][4]f32, W * H, context.temp_allocator)
				for &px in ref_j[k] {px = SENTINEL}
				for &px in ref_i[k] {px = SENTINEL}
			}
			run_targets(&m.program, nil, W, H, ref_j[:], scratch, {varyings = vars_slice})
			run_targets(&p_int, nil, W, H, ref_i[:], scratch, {varyings = vars_slice})
			for k in 0 ..< 3 {
				testing.expect(t, slice.equal(slice.reinterpret([]u32, ref_j[k]), slice.reinterpret([]u32, ref_i[k])))
			}

			regions := make([dynamic]Region_Case, 0, 16, context.temp_allocator)
			// 32-row full-width bands including shorter last band (o3d's exact call pattern).
			for by := 0; by < H; by += 32 {
				append(&regions, Region_Case{0, by, W, min(32, H - by)})
			}
			// 1-row bands at top, odd row, interior, and bottom.
			append(&regions, Region_Case{0, 0, W, 1})
			append(&regions, Region_Case{0, 1, W, 1})
			append(&regions, Region_Case{0, 33, W, 1})
			append(&regions, Region_Case{0, H - 1, W, 1})
			// Aligned interior region with x > 0, y > 0 (hits full_tile fast path in both 8x8 interpreter and 4x2 JIT).
			append(&regions, Region_Case{8, 8, 24, 16})
			// Unaligned sub-regions with x > 0, y > 0, odd starts and odd sizes.
			append(&regions, Region_Case{5, 7, 23, 19})
			append(&regions, Region_Case{3, 12, W - 4, 21})

			crop_ref := make([][4]f32, W * H, context.temp_allocator)
			check_crop :: proc(t: ^testing.T, got, full: [][4]f32, W: int, rc: Region_Case, tmp: [][4]f32) {
				n := rc.w * rc.h
				for ry in 0 ..< rc.h {
					copy(tmp[ry * rc.w:(ry + 1) * rc.w], full[(rc.y + ry) * W + rc.x:][:rc.w])
				}
				if !slice.equal(slice.reinterpret([]u32, got[:n]), slice.reinterpret([]u32, tmp[:n])) {
					testing.expectf(t, false, "region crop mismatch for W=%d rc=%v", W, rc)
				}
			}

			for rc in regions {
				n := rc.w * rc.h
				ropts := Run_Options{
					x              = rc.x,
					y              = rc.y,
					w              = rc.w,
					h              = rc.h,
					region_targets = true,
					varyings       = vars_slice,
				}

				for prog in ([2]^Program{&m.program, &p_int}) {
					// run_f32 (with extra tail elements to verify nothing writes past w * h).
					f32_buf := make([][4]f32, n + 7, context.temp_allocator)
					for &px in f32_buf {px = SENTINEL}
					run_f32(prog, nil, W, H, f32_buf, scratch, ropts)
					check_crop(t, f32_buf[:n], ref_j[0], W, rc, crop_ref)
					for tail_px in f32_buf[n:] {
						testing.expect_value(t, tail_px, SENTINEL)
					}

					// 1, 2, and 3 targets with nil entries (including {nil, b1, b2} like o3d's far plane).
					Mask3 :: bit_set[0 ..< 3;u8]
					for mask in ([6]Mask3{{0}, {0, 1}, {1}, {1, 2}, {0, 2}, {0, 1, 2}}) {
						bs: [3][][4]f32
						tlen := 1
						if 2 in mask {
							tlen = 3
						} else if 1 in mask {
							tlen = 2
						}
						for k in 0 ..< tlen {
							if k in mask {
								bs[k] = make([][4]f32, n, context.temp_allocator)
								for &px in bs[k] {px = SENTINEL}
							}
						}
						run_targets(prog, nil, W, H, bs[:tlen], scratch, ropts)
						for k in 0 ..< tlen {
							if k in mask {
								check_crop(t, bs[k], ref_j[k], W, rc, crop_ref)
							}
						}
					}
				}
			}
		}
	}
}

@(test)
test_region_targets_ensure_no_region_w :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [16][4]f32
	targets := [1][][4]f32{buf[:]}
	testing.expect_assert_message(t, "oksl: region_targets requires opts.w > 0 and opts.h > 0")
	run_targets(&m.program, nil, 8, 8, targets[:], temp_scratch(), {w = 0, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_no_region_h :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil
	buf: [16][4]f32
	targets := [1][][4]f32{buf[:]}
	testing.expect_assert_message(t, "oksl: region_targets requires opts.w > 0 and opts.h > 0")
	run_targets(&p_int, nil, 8, 8, targets[:], temp_scratch(), {w = 4, h = 0, region_targets = true})
}

@(test)
test_region_targets_ensure_neg_x :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [16][4]f32
	targets := [1][][4]f32{buf[:]}
	testing.expect_assert_message(t, "oksl: region_targets region lies outside the layer")
	run_targets(&m.program, nil, 8, 8, targets[:], temp_scratch(), {x = -1, y = 0, w = 4, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_neg_y :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [16][4]f32
	targets := [1][][4]f32{buf[:]}
	testing.expect_assert_message(t, "oksl: region_targets region lies outside the layer")
	run_targets(&m.program, nil, 8, 8, targets[:], temp_scratch(), {x = 0, y = -1, w = 4, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_outside_x :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [16][4]f32
	targets := [1][][4]f32{buf[:]}
	testing.expect_assert_message(t, "oksl: region_targets region lies outside the layer")
	run_targets(&m.program, nil, 8, 8, targets[:], temp_scratch(), {x = 5, y = 0, w = 4, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_outside_y :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [16][4]f32
	targets := [1][][4]f32{buf[:]}
	testing.expect_assert_message(t, "oksl: region_targets region lies outside the layer")
	run_targets(&m.program, nil, 8, 8, targets[:], temp_scratch(), {x = 0, y = 5, w = 4, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_short_target :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [15][4]f32
	targets := [2][][4]f32{nil, buf[:]}
	testing.expect_assert_message(t, "oksl: target is shorter than opts.w * opts.h")
	run_targets(&m.program, nil, 8, 8, targets[:], temp_scratch(), {x = 2, y = 2, w = 4, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_short_run_f32 :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [15][4]f32
	testing.expect_assert_message(t, "oksl: target is shorter than opts.w * opts.h")
	run_f32(&m.program, nil, 8, 8, buf[:], temp_scratch(), {x = 0, y = 0, w = 4, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_nil_run_f32 :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	testing.expect_assert_message(t, "oksl: target is shorter than opts.w * opts.h")
	run_f32(&m.program, nil, 8, 8, nil, temp_scratch(), {x = 0, y = 0, w = 4, h = 4, region_targets = true})
}

@(test)
test_region_targets_ensure_run_rgba8 :: proc(t: ^testing.T) {
	m := compile_simple_frag_for_ensure(t)
	buf: [64]u8
	testing.expect_assert_message(t, "oksl: region_targets applies to float targets (run_f32 and run_targets), not run")
	run(&m.program, nil, 8, 8, buf[:], 16, temp_scratch(), {x = 0, y = 0, w = 4, h = 4, region_targets = true})
}

@(test)
test_runner_run_region_sampler_lanes :: proc(t: ^testing.T) {
	src := `#version 310 es
precision highp float;
layout(location = 0) in vec2 vUV;
uniform sampler2D s0;
uniform sampler2D s1;
layout(location = 0) out vec4 o;
void main() {
    vec4 a = texture(s0, vUV);
    vec4 b = texelFetch(s1, ivec2(gl_FragCoord.xy), 0);
    o = a + b;
}`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "region sampler compile failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil

	Region_Sampler_State :: struct {
		lw, lh:                  int,
		start_x, start_y:        int,
		end_x, end_y:            int,
		calls:                   [2]int,
		bad_uv:                  int,
	}
	region_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
		st := (^Region_Sampler_State)(user)
		if channel < 0 || channel >= 2 {
			st.bad_uv += 1
			return {}
		}
		st.calls[channel] += 1
		fx := uv.x * f32(st.lw) - 0.5
		fy := uv.y * f32(st.lh) - 0.5
		px := int(math.round(fx))
		py := int(math.round(fy))
		if abs(fx - f32(px)) > 1e-4 || abs(fy - f32(py)) > 1e-4 {
			st.bad_uv += 1
			return {}
		}
		qx, qy := px &~ 1, py &~ 1
		if !(qx + 1 >= st.start_x && qx < st.end_x && qy + 1 >= st.start_y && qy < st.end_y) {
			st.bad_uv += 1
		}
		return {uv.x, uv.y, f32(channel), 1}
	}

	SENTINEL :: [4]f32{-77.0, 88.0, -99.0, 11.0}
	scratch := temp_scratch()

	Rect :: struct {
		x, y, w, h: int,
	}
	// Powers-of-two denominators aren't needed because `region_cb` checks within 1e-4, so odd layer sizes work directly.
	for dims in ([4][2]int{{5, 5}, {9, 7}, {13, 11}, {12, 10}}) {
		LW, LH := dims[0], dims[1]
		sizes := [2][3]i32{{i32(LW), i32(LH), 1}, {i32(LW), i32(LH), 1}}
		rects := [?]Rect {
			{0, 0, LW, LH},       // full odd/even layer
			{1, 1, 4, 2},         // odd start (1,1), odd end (5,3) -> 3x2 quads (hits exec == 0x0F on right batch!)
			{1, 1, 1, 1},         // 1x1 at odd (1,1) -> 1x1 quad (hits exec == 0x0F!)
			{1, 1, 2, 3},         // odd start (1,1), odd end x=3, even end y=4 -> 2x2 quads
			{3, 1, LW - 4, LH - 2}, // odd start (3,1) to (LW-1, LH-1)
			{0, 1, 5, 3},         // 5 wide (3 quad cols -> exec == 0x0F!), odd y
		}
		for rc in rects {
			if rc.w <= 0 || rc.h <= 0 || rc.x + rc.w > LW || rc.y + rc.h > LH {
				continue
			}
			sx, sy := rc.x, rc.y
			ex, ey := rc.x + rc.w, rc.y + rc.h
			nq_x, nq_y := 0, 0
			for qx := sx &~ 1; qx < ex; qx += 2 {
				if qx + 1 >= sx {nq_x += 1}
			}
			for qy := sy &~ 1; qy < ey; qy += 2 {
				if qy + 1 >= sy {nq_y += 1}
			}
			want_calls := 4 * nq_x * nq_y

			full_j := make([][4]f32, LW * LH, context.temp_allocator)
			full_i := make([][4]f32, LW * LH, context.temp_allocator)
			for use_reg in ([2]bool{false, true}) {
				st_j := Region_Sampler_State{lw = LW, lh = LH, start_x = sx, start_y = sy, end_x = ex, end_y = ey}
				st_i := st_j

				n_out := rc.w * rc.h if use_reg else LW * LH
				bj := make([][4]f32, n_out, context.temp_allocator)
				bi := make([][4]f32, n_out, context.temp_allocator)
				for &px in bj {px = SENTINEL}
				for &px in bi {px = SENTINEL}
				tj := [1][][4]f32{bj}
				ti := [1][][4]f32{bi}

				opts_j := Run_Options{
					sampler        = region_cb,
					user           = &st_j,
					x              = rc.x,
					y              = rc.y,
					w              = rc.w,
					h              = rc.h,
					region_targets = use_reg,
					channel_sizes  = sizes[:],
				}
				opts_i := opts_j
				opts_i.user = &st_i

				run_targets(&m.program, nil, LW, LH, tj[:], scratch, opts_j)
				run_targets(&p_int, nil, LW, LH, ti[:], scratch, opts_i)

				testing.expect_value(t, st_j.calls[0], want_calls)
				testing.expect_value(t, st_j.calls[1], want_calls)
				testing.expect_value(t, st_j.bad_uv, 0)
				testing.expect_value(t, st_i.calls[0], want_calls)
				testing.expect_value(t, st_i.calls[1], want_calls)
				testing.expect_value(t, st_i.bad_uv, 0)
				testing.expect(t, slice.equal(slice.reinterpret([]u32, bj), slice.reinterpret([]u32, bi)))

				if !use_reg {
					copy(full_j, bj)
					copy(full_i, bi)
				} else {
					for ry in 0 ..< rc.h {
						row_got := bj[ry * rc.w:(ry + 1) * rc.w]
						row_ref := full_j[(rc.y + ry) * LW + rc.x:][:rc.w]
						testing.expect(t, slice.equal(slice.reinterpret([]u32, row_got), slice.reinterpret([]u32, row_ref)))
					}
				}
			}
		}
	}
}

@(test)
test_jit_dyn_slots_only :: proc(t: ^testing.T) {
	// Dynamic array indexing next to cached single-BB temporaries (both uniform
	// and lane-divergent indices, static write -> Load_Dyn in one BB, Store_Dyn ->
	// static/dynamic reads, and swizzled vec3 array reads).
	make_src :: proc(dynamic_idx: bool) -> string {
		idx_expr := "idx" if dynamic_idx else "1"
		return fmt.tprintf(`#version 310 es
precision highp float;
uniform int uBase;
out vec4 o;
void main() {{
    int px = int(gl_FragCoord.x);
    int py = int(gl_FragCoord.y);
    int idx = clamp((px + py + uBase) %% 3, 0, 2);
    int idx2 = clamp((px * 2 + uBase) %% 4, 0, 3);

    vec4 v = vec4(float(px) * 0.1 + 1.0, float(py) * 0.2 + 0.5, 0.75, 1.25);
    vec3 a[3];
    a[0] = v.xyz + vec3(1.0, 2.0, 3.0);
    a[1] = v.yzx * vec3(1.5, 0.5, 2.5);
    a[2] = v.zxy - vec3(0.25, 0.75, 1.25);

    // 24 vec4 ops (96 scalar writes) in the same BB as Load_Dyn: must stay in YMM cache,
    // while array a (9 slots) MUST flush to memory before Load_Dyn reads [rbx + (a+off)*32].
    vec4 t0 = v * 1.1 + 0.05;
    vec4 t1 = t0 * 0.9 + v;
    vec4 t2 = t1 * 1.05 - t0 * 0.25;
    vec4 t3 = t2 + t1 * 0.5;
    vec4 t4 = t3 * 0.8 + t0;
    vec4 t5 = t4 - t2 * 0.3;
    vec4 t6 = t5 * 1.2 + t3 * 0.1;
    vec4 t7 = t6 + t4 * 0.4;
    vec4 t8 = t7 * 0.75 + t5;
    vec4 t9 = t8 - t6 * 0.2 + t1;
    vec4 t10 = t9 * 1.15 + t7 * 0.3;
    vec4 t11 = t10 + t8 * 0.6 - t2 * 0.1;
    vec4 t12 = t11 * 0.95 + t9 * 0.2;
    vec4 t13 = t12 - t10 * 0.15 + t3 * 0.25;
    vec4 t14 = t13 * 1.05 + t11 * 0.35;
    vec4 t15 = t14 + t12 * 0.45 - t4 * 0.05;

    vec2 picked = a[%s].zy;

    float b[4];
    b[0] = 10.0;
    b[1] = 20.0;
    b[2] = 30.0;
    b[3] = 40.0;
    if (%s > 0) {{
        b[idx2] = t15.x + picked.x;
    }}
    float s = b[0] + b[2] + b[%s];
    o = t15 + vec4(picked, s, t0.w + t8.z);
}}`, idx_expr, "uBase + px" if dynamic_idx else "uBase", "idx2" if dynamic_idx else "1")
	}

	m_dyn, e1, ok1 := compile(make_src(true), {profile = .Glsl, stage = .Fragment})
	defer destroy(m_dyn)
	testing.expectf(t, ok1, "dyn compile failed:\n%s", format_errors(e1, context.temp_allocator))
	m_sta, e2, ok2 := compile(make_src(false), {profile = .Glsl, stage = .Fragment})
	defer destroy(m_sta)
	testing.expectf(t, ok2, "sta compile failed:\n%s", format_errors(e2, context.temp_allocator))
	if !ok1 || !ok2 {
		return
	}

	// Only `a` (9 slots) and `b` (4 slots) are dynamically indexed.
	dyn_total := i32(0)
	for ds in m_dyn.program.dyn_slots {
		dyn_total += ds.n
	}
	testing.expect_value(t, dyn_total, i32(13))

	when JIT_ENABLED {
		if jit_avx2_ready {
			// Reverting Item 1 marks all slots cross_bb and emits >600 B of extra [rbx] stores on t0..t15.
			testing.expectf(
				t,
				len(m_dyn.program.jit_mem) < len(m_sta.program.jit_mem) + 450,
				"expected cached temporaries alongside dynamic arrays: dyn=%d B vs static=%d B",
				len(m_dyn.program.jit_mem),
				len(m_sta.program.jit_mem),
			)
		}
	}

	p_int := m_dyn.program
	p_int.jit_body, p_int.jit_init = nil, nil
	scratch := temp_scratch()
	ublk := make([]u8, max(m_dyn.block_size, 16), context.temp_allocator)
	for base_val in ([3]f32{0, 1, 2}) {
		set_uniform(m_dyn, ublk, "uBase", {base_val})
		oj: [32][4]f32
		oi: [32][4]f32
		run_f32(&m_dyn.program, ublk, 8, 4, oj[:], scratch)
		run_f32(&p_int, ublk, 8, 4, oi[:], scratch)
		testing.expect(t, slice.equal(slice.reinterpret([]u32, oj[:]), slice.reinterpret([]u32, oi[:])))
	}
}

@(private = "file")
jit_vec4_bits_eq :: proc(a, b: [4]f32) -> bool {
	for c in 0 ..< 4 {
		if !(math.is_nan(a[c]) && math.is_nan(b[c])) && transmute(u32)a[c] != transmute(u32)b[c] {
			return false
		}
	}
	return true
}

@(test)
test_jit_dst_reg_direct :: proc(t: ^testing.T) {
	// 1. Synthetic Program exercising dst == src aliasing (in-place Splat, scalar F_Mul k=1,
	//    in-place Gather reversal, Cross, Normalize, Select), dead dst, and cross_bb dst.
	consts := [8][2]u32 {
		{0, transmute(u32)f32(2.0)},
		{1, transmute(u32)f32(-3.0)},
		{2, transmute(u32)f32(5.0)},
		{3, transmute(u32)f32(0.5)},
		{8, transmute(u32)f32(1.25)},
		{9, transmute(u32)f32(2.5)},
		{10, transmute(u32)f32(-1.5)},
		{11, transmute(u32)f32(4.0)},
	}
	outputs := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 12, n = 4, base = .Float}}
	code := [13]Instr {
		{op = .Mov, n = 1, dst = 4, a = 0},                       // slot 4 = 2.0 (cached)
		{op = .Splat, n = 4, dst = 4, a = 4},                     // in-place splat: slots 4..7 = (2,2,2,2)
		{op = .F_Add, n = 4, dst = 4, a = 4, b = 0},              // in-place vec add: (4, -1, 7, 2.5)
		{op = .F_Mul, n = 4, k = 1, dst = 4, a = 4, b = 8},       // scalar a aliases dst+0: 4.0 * (1.25, 2.5, -1.5, 4.0) = (5, 10, -6, 16)
		{op = .Gather, n = 4, dst = 4, a = 7, b = 6, c = 5, d = 4}, // in-place reverse swizzle: (16, -6, 10, 5)
		{op = .Cross, n = 3, dst = 4, a = 4, b = 8},              // in-place cross with (1.25, 2.5, -1.5)
		{op = .Normalize, n = 3, dst = 4, a = 4},                 // in-place normalize slots 4..6
		{op = .F_Add, n = 1, dst = 7, a = 7, b = 3},              // dead intra-BB write if overwritten next
		{op = .F_Mul, n = 1, dst = 7, a = 4, b = 2},              // overwrites dead slot 7, read below
		{op = .F_Lt, n = 1, dst = 16, a = 5, b = 0},              // bool in slot 16
		{op = .Select, n = 4, k = 1, dst = 4, a = 4, b = 8, c = 16}, // in-place scalar select
		{op = .F_Add, n = 4, dst = 12, a = 4, b = 8},             // cross_bb output dst dead after pc
		{op = .F_Sub, n = 1, dst = 17, a = 0, b = 1},             // dead non-cross_bb dst (unread)
	}
	p := Program {
		code      = code[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 18,
		num_masks = 2,
		stage     = .Fragment,
	}
	jit_compile(&p)
	defer jit_destroy(&p)

	oj1: [1][4]f32
	oi1: [1][4]f32
	run_f32(&p, nil, 1, 1, oj1[:], temp_scratch())
	run_f32_interp(&p, nil, 1, 1, oi1[:], temp_scratch(), {})
	testing.expectf(
		t,
		jit_vec4_bits_eq(oj1[0], oi1[0]),
		"bit mismatch: jit=%v (0x%08x) interp=%v (0x%08x)",
		oj1[0], transmute([4]u32)oj1[0],
		oi1[0], transmute([4]u32)oi1[0],
	)

	// 2. Relative two-program comparison: 32 alternating vec4 F_Add/F_Mul ops in cached
	//    BB-local slots 4..7 and 8..11 write directly into ymm8..15 (5 B per component,
	//    identical length to reg-to-reg Mov) without bouncing through scratch ymm0.
	outputs0 := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 0, n = 4, base = .Float}}
	consts2 := [8][2]u32 {
		{8, transmute(u32)f32(1.25)},
		{9, transmute(u32)f32(2.5)},
		{10, transmute(u32)f32(-1.5)},
		{11, transmute(u32)f32(4.0)},
		{12, transmute(u32)f32(2.0)},
		{13, transmute(u32)f32(-3.0)},
		{14, transmute(u32)f32(5.0)},
		{15, transmute(u32)f32(0.5)},
	}
	chain_alu: [35]Instr
	chain_mov: [35]Instr
	chain_alu[0] = {op = .Mov, n = 4, dst = 4, a = 12}
	chain_alu[1] = {op = .Mov, n = 4, dst = 8, a = 8}
	chain_mov[0] = {op = .Mov, n = 4, dst = 4, a = 12}
	chain_mov[1] = {op = .Mov, n = 4, dst = 8, a = 8}
	for i in 0 ..< 32 {
		if (i & 1) == 0 {
			chain_alu[2 + i] = {op = .F_Add, n = 4, dst = 4, a = 4, b = 8}
			chain_mov[2 + i] = {op = .Mov, n = 4, dst = 4, a = 8}
		} else {
			chain_alu[2 + i] = {op = .F_Mul, n = 4, dst = 8, a = 8, b = 4}
			chain_mov[2 + i] = {op = .Mov, n = 4, dst = 8, a = 4}
		}
	}
	chain_alu[34] = {op = .F_Add, n = 4, dst = 0, a = 4, b = 8}
	chain_mov[34] = {op = .F_Add, n = 4, dst = 0, a = 4, b = 8}

	p_cached := Program {
		code      = chain_alu[:],
		consts    = consts2[:],
		outputs   = outputs0[:],
		num_slots = 16,
		num_masks = 2,
		stage     = .Fragment,
	}
	jit_compile(&p_cached)
	defer jit_destroy(&p_cached)

	p_mov := Program {
		code      = chain_mov[:],
		consts    = consts2[:],
		outputs   = outputs0[:],
		num_slots = 16,
		num_masks = 2,
		stage     = .Fragment,
	}
	jit_compile(&p_mov)
	defer jit_destroy(&p_mov)

	when JIT_ENABLED {
		if jit_avx2_ready {
			// Direct-dst vaddps/vmulps (5 B) matches reg-to-reg vmovaps (5 B);
			// bouncing through ymm0 adds 128 vmovaps copies (+512 B).
			testing.expectf(
				t,
				len(p_cached.jit_mem) * 6 < len(p_mov.jit_mem) * 7,
				"expected direct-dst cached ALU chain to omit ymm0 bounces: alu=%d B vs mov=%d B",
				len(p_cached.jit_mem),
				len(p_mov.jit_mem),
			)
		}
	}
	oj2, oi2: [1][4]f32
	run_f32(&p_cached, nil, 1, 1, oj2[:], temp_scratch())
	run_f32_interp(&p_cached, nil, 1, 1, oi2[:], temp_scratch(), {})
	testing.expect(t, jit_vec4_bits_eq(oj2[0], oi2[0]))

	oj3, oi3: [1][4]f32
	run_f32(&p_mov, nil, 1, 1, oj3[:], temp_scratch())
	run_f32_interp(&p_mov, nil, 1, 1, oi3[:], temp_scratch(), {})
	testing.expect(t, jit_vec4_bits_eq(oj3[0], oi3[0]))
}

@(test)
test_jit_mask_restore_dead_kills :: proc(t: ^testing.T) {
	// 1. Relative two-program comparison: 20 Mask_Restore ops whose kill masks (2..4) have
	//    no writers prune to 0-byte no-ops vs the same sequence when 2..4 are written.
	consts := [1][2]u32{{0, transmute(u32)f32(1.0)}}
	outputs := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 1, n = 4, base = .Float}}
	code_zero: [25]Instr
	code_zero[0] = {op = .Mask_Copy, dst = 5, a = MASK_EXEC}
	code_zero[1] = {op = .Mask_Clear, dst = 2}
	code_zero[2] = {op = .Mask_Clear, dst = 3}
	code_zero[3] = {op = .Mask_Clear, dst = 4}
	for i in 0 ..< 20 {
		code_zero[4 + i] = {op = .Mask_Restore, dst = 5, a = 5, b = 2, c = 3, d = 4}
	}
	code_zero[24] = {op = .Splat, n = 4, dst = 1, a = 0}

	code_live: [28]Instr
	for i in 0 ..< 25 {
		code_live[i] = code_zero[i]
	}
	code_live[25] = {op = .Mask_Copy, dst = 2, a = 5}
	code_live[26] = {op = .Mask_Copy, dst = 3, a = 5}
	code_live[27] = {op = .Mask_Copy, dst = 4, a = 5}

	p_zero := Program {
		code      = code_zero[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 5,
		num_masks = 6,
		stage     = .Fragment,
	}
	jit_compile(&p_zero)
	defer jit_destroy(&p_zero)

	p_live := Program {
		code      = code_live[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 5,
		num_masks = 6,
		stage     = .Fragment,
	}
	jit_compile(&p_live)
	defer jit_destroy(&p_live)

	when JIT_ENABLED {
		if jit_avx2_ready {
			testing.expectf(
				t,
				len(p_zero.jit_mem) * 2 < len(p_live.jit_mem),
				"expected dead-kill Mask_Restore pruning: dead=%d B vs live=%d B",
				len(p_zero.jit_mem),
				len(p_live.jit_mem),
			)
		}
	}

	// 2. Adversarial static-proof violation:
	//    - break and continue appear AFTER an inner if/else inside a loop (reaching the earlier
	//      Mask_Restore on the next iteration via the loop back-edge);
	//    - early return in an inlined helper precedes a subsequent if/else;
	//    - divergent discard precedes a subsequent if/else.
	src := `#version 310 es
precision highp float;
out vec4 o;

float helper_early(float x) {
	float v = 1.0;
	if (x > 5.0) {
		return 42.0;
	}
	if (x > 2.5) {
		v += 10.0;
	} else {
		v += 3.0;
	}
	return v;
}

void main() {
	float x = floor(gl_FragCoord.x);
	float y = floor(gl_FragCoord.y);
	float acc = helper_early(x);

	for (int i = 0; i < 6; i++) {
		// Earlier Mask_Restore in loop body: reached on iteration i+1 after break/continue below!
		if (x > float(i)) {
			acc += float(i + 1) * 2.0;
		} else {
			acc += 0.5;
		}
		if (i == 1 && int(x) == 2) {
			continue;
		}
		acc += 100.0;
		if (i == 2 && int(x) == 4) {
			break;
		}
		acc += 10.0;
	}

	if (int(y) == 1 && int(x) == 6) {
		discard;
	}
	if (y > 1.5) {
		acc += 7.0;
	} else {
		acc -= 3.0;
	}
	o = vec4(acc, x, y, 1.0);
}
`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil

	W, H :: 8, 4
	oj: [W * H][4]f32
	oi: [W * H][4]f32
	for px in 0 ..< W * H {
		oj[px] = {-99, -99, -99, -99}
		oi[px] = {-99, -99, -99, -99}
	}
	run_f32(&m.program, nil, W, H, oj[:], temp_scratch())
	run_f32(&p_int, nil, W, H, oi[:], temp_scratch())
	for px in 0 ..< W * H {
		testing.expectf(t, oj[px] == oi[px], "px %d: jit=%v interp=%v", px, oj[px], oi[px])
	}
}

@(test)
test_jit_cmp_mask_cond_fusion :: proc(t: ^testing.T) {
	// 1. Fails if reverted: 16 comparisons fused into Mask_Cond (one per comparison VM_Op,
	//    testing both pc+1 and pc+2 with Mask_Copy, and both k=0 and k=1) vs unfused when
	//    the bool slot is also read later.
	cmp_ops := [16]VM_Op {
		.F_Lt, .F_Le, .F_Gt, .F_Ge, .F_Eq, .F_Ne,
		.I_Lt, .I_Le, .I_Gt, .I_Ge, .I_Eq, .I_Ne,
		.U_Lt, .U_Le, .U_Gt, .U_Ge,
	}
	consts := [4][2]u32 {
		{0, transmute(u32)f32(-2.0)}, // also 0xC0000000 (> 2^31 as u32, < 0 as i32)
		{1, transmute(u32)f32(3.0)},
		{2, 0x7FC0_0000},             // quiet NaN
		{3, transmute(u32)f32(1.0)},
	}
	outputs := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 4, n = 4, base = .Float}}

	code_fused: [dynamic]Instr
	defer delete(code_fused)
	code_unfused: [dynamic]Instr
	defer delete(code_unfused)

	append(&code_fused, Instr{op = .Splat, n = 4, dst = 4, a = 3})
	append(&code_unfused, Instr{op = .Splat, n = 4, dst = 4, a = 3})
	// Cache slot 8 first, then overwrite slot 8 in-place on op 0 to test cached-cond invalidation.
	append(&code_fused, Instr{op = .Mov, n = 1, dst = 8, a = 0})
	append(&code_unfused, Instr{op = .Mov, n = 1, dst = 8, a = 0})

	for op, idx in cmp_ops {
		sa := i32(8) if idx == 0 else (i32(2) if idx == 5 else i32(0))
		sb := i32(1)
		cond_s := i32(8 + idx)
		k_mc := u8(idx & 1)
		append(&code_fused, Instr{op = op, n = 1, dst = cond_s, a = sa, b = sb})
		append(&code_unfused, Instr{op = op, n = 1, dst = cond_s, a = sa, b = sb})
		if (idx & 1) == 0 {
			append(&code_fused, Instr{op = .Mask_Copy, dst = 2, a = MASK_EXEC})
			append(&code_unfused, Instr{op = .Mask_Copy, dst = 2, a = MASK_EXEC})
		}
		src_m := i32(2) if (idx & 1) == 0 else i32(MASK_EXEC)
		append(&code_fused, Instr{op = .Mask_Cond, k = k_mc, dst = 3, a = src_m, b = cond_s})
		append(&code_unfused, Instr{op = .Mask_Cond, k = k_mc, dst = 3, a = src_m, b = cond_s})
		// Keep cond_s alive after Mask_Cond in code_unfused so fusion is illegal there.
		append(&code_unfused, Instr{op = .Bit_Or, n = 1, dst = 25, a = cond_s, b = cond_s})
	}
	append(&code_unfused, Instr{op = .I_To_F, n = 1, dst = 26, a = 25})

	p_fused := Program {
		code      = code_fused[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 28,
		num_masks = 4,
		stage     = .Fragment,
	}
	jit_compile(&p_fused)
	defer jit_destroy(&p_fused)

	p_unfused := Program {
		code      = code_unfused[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 28,
		num_masks = 4,
		stage     = .Fragment,
	}
	jit_compile(&p_unfused)
	defer jit_destroy(&p_unfused)

	when JIT_ENABLED {
		if jit_avx2_ready {
			// 16 extra Bit_Or + 1 I_To_F in p_unfused add ~85 B, while unfusing 16 cmp+Mask_Cond pairs adds ~350 B (>430 B diff).
			// Reverting Item 3b collapses the difference to ~85 B.
			testing.expectf(
				t,
				len(p_fused.jit_mem) + 260 < len(p_unfused.jit_mem),
				"expected fused cmp+Mask_Cond to shrink code: fused=%d B vs unfused=%d B",
				len(p_fused.jit_mem),
				len(p_unfused.jit_mem),
			)
		}
	}

	// 2. End-to-end GLSL test covering fused comparisons (float/NaN, signed int, unsigned >= 2^31)
	//    AND adversarial cases where the comparison bool is reused in the same BB or across BBs.
	src := `#version 310 es
precision highp float;
uniform float uZero;
out vec4 o;

void main() {
	float x = floor(gl_FragCoord.x) - 3.5;
	float y = floor(gl_FragCoord.y) - 1.5;
	int ix = int(floor(gl_FragCoord.x)) - 4;
	int iy = int(floor(gl_FragCoord.y)) - 2;
	uint ux = uint(int(floor(gl_FragCoord.x))) * 0x30000000u;
	uint uy = 0x80000000u;
	float nan_v = uZero / uZero;

	float acc = 0.0;
	// Fused float, NaN, signed int, unsigned int, and loop conditions:
	if (x < y)  { acc += 1.0; }
	if (x <= y) { acc += 2.0; }
	if (x > y)  { acc += 4.0; }
	if (x >= y) { acc += 8.0; }
	if (x == -0.5) { acc += 16.0; }
	if (nan_v != 1.0) { acc += 32.0; }
	if (nan_v < 1.0)  { acc += 1000.0; }
	if (ix < iy)  { acc += 64.0; }
	if (ix <= iy) { acc += 128.0; }
	if (ix > iy)  { acc += 256.0; }
	if (ix >= iy) { acc += 512.0; }
	if (ix == -1) { acc += 1024.0; }
	if (ix != -2) { acc += 2048.0; }
	if (ux < uy)  { acc += 0.25; }
	if (ux <= uy) { acc += 0.5; }
	if (ux > uy)  { acc += 0.125; }
	if (ux >= uy) { acc += 0.0625; }

	for (int k = 0; k < 4; k++) {
		if (k <= ix + 4) {
			acc += 3.0;
		}
	}

	// Adversarial 1: bool used by Mask_Cond AND converted to float after the branch (cross-BB).
	bool b_cross = (x < 0.0);
	if (b_cross) {
		acc += 5.0;
	}
	acc += float(b_cross) * 11.0;

	// Adversarial 2: bool used by two separate Mask_Conds across BBs.
	bool b_twice = (ux >= uy);
	if (b_twice) {
		acc += 7.0;
	}
	if (b_twice) {
		acc += 13.0;
	}

	// Adversarial 3: bool read in the same BB before Mask_Cond via Select.
	bool b_same = (ix != 0);
	float sel = b_same ? 17.0 : 19.0;
	if (b_same) {
		acc += sel;
	}

	o = vec4(acc, sel, float(b_cross), float(b_twice));
}
`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil
	ublk := make([]u8, max(m.block_size, 16), context.temp_allocator)
	set_uniform(m, ublk, "uZero", {0.0})

	W, H :: 8, 4
	oj: [W * H][4]f32
	oi: [W * H][4]f32
	run_f32(&m.program, ublk, W, H, oj[:], temp_scratch())
	run_f32(&p_int, ublk, W, H, oi[:], temp_scratch())
	for px in 0 ..< W * H {
		testing.expectf(t, oj[px] == oi[px], "px %d: jit=%v interp=%v", px, oj[px], oi[px])
	}
}

@(test)
test_jit_mask_eq_exec :: proc(t: ^testing.T) {
	// 1. Relative two-program comparison: within a single BB, copying/testing a mask known
	//    equal to MASK_EXEC (r13d) avoids memory reloads vs copying from an unrelated mask.
	consts := [3][2]u32 {
		{0, 1},
		{1, 0},
		{2, transmute(u32)f32(3.0)},
	}
	outputs := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 4, n = 4, base = .Float}}
	code_opt: [dynamic]Instr
	defer delete(code_opt)
	code_noopt: [dynamic]Instr
	defer delete(code_noopt)

	append(&code_opt, Instr{op = .Mask_Copy, dst = 4, a = MASK_EXEC})
	append(&code_opt, Instr{op = .Jump, a = 2})
	append(&code_noopt, Instr{op = .Mask_Copy, dst = 4, a = MASK_EXEC})
	append(&code_noopt, Instr{op = .Jump, a = 2}) // BB boundary clears mask_eq_gen[4]
	for _ in 0 ..< 20 {
		append(&code_opt, Instr{op = .Mask_Copy, dst = 2, a = MASK_EXEC})
		append(&code_opt, Instr{op = .Mask_Copy, dst = MASK_EXEC, a = 2}) // 0-byte no-op when m2 == r13d
		append(&code_opt, Instr{op = .Mask_Cond, dst = 3, a = 2, b = 0})
		append(&code_opt, Instr{op = .Mask_Copy, dst = MASK_EXEC, a = 2}) // 0-byte no-op when m2 == r13d

		append(&code_noopt, Instr{op = .Mask_Copy, dst = 2, a = 4})       // m2 loaded from m4 != r13d
		append(&code_noopt, Instr{op = .Mask_Copy, dst = MASK_EXEC, a = 2})
		append(&code_noopt, Instr{op = .Mask_Cond, dst = 3, a = 2, b = 0})
		append(&code_noopt, Instr{op = .Mask_Copy, dst = MASK_EXEC, a = 2})
	}
	append(&code_opt, Instr{op = .Splat, n = 4, dst = 4, a = 2})
	append(&code_noopt, Instr{op = .Splat, n = 4, dst = 4, a = 2})

	p_opt := Program {
		code      = code_opt[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 8,
		num_masks = 5,
		stage     = .Fragment,
	}
	jit_compile(&p_opt)
	defer jit_destroy(&p_opt)

	p_noopt := Program {
		code      = code_noopt[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 8,
		num_masks = 5,
		stage     = .Fragment,
	}
	jit_compile(&p_noopt)
	defer jit_destroy(&p_noopt)

	when JIT_ENABLED {
		if jit_avx2_ready {
			testing.expectf(
				t,
				len(p_opt.jit_mem) * 6 < len(p_noopt.jit_mem) * 5,
				"expected r13d mask reuse to shrink emission: opt=%d B vs noopt=%d B",
				len(p_opt.jit_mem),
				len(p_noopt.jit_mem),
			)
		}
	}

	// 2. Adversarial intra-BB invalidation:
	//    a) Mask_Cond(MASK_EXEC, ...) narrows r13d -> saved m2 no longer equals r13d;
	//    b) Mask_Clear(3) clears m3 -> m3 no longer equals r13d;
	//    c) Mask_Or(dst = 2, ...) widens m2 after m2 == r13d -> MUST invalidate m2 == r13d;
	//    d) emit_copy_mask's else branch (Mask_Copy(dst = 3, a = 2) when m2 != r13d, even
	//       if m3 previously equalled r13d) MUST set mask_eq_gen[3] = 0, never exec_gen.
	adv_consts := [5][2]u32 {
		{0, 0},
		{1, transmute(u32)f32(7.0)},
		{2, transmute(u32)f32(99.0)},
		{3, transmute(u32)f32(42.0)},
		{4, transmute(u32)f32(77.0)},
	}
	outputs2 := [2]Program_Var {
		{kind = .Stage_Out, location = 0, slot = 4, n = 4, base = .Float},
		{kind = .Stage_Out, location = 1, slot = 16, n = 4, base = .Float},
	}
	adv_code_full := [?]Instr {
		{op = .Splat, n = 4, dst = 8, a = 2},                     // slots 8..11 = 99.0
		{op = .Splat, n = 4, dst = 12, a = 3},                    // slots 12..15 = 42.0
		{op = .Splat, n = 4, dst = 20, a = 4},                    // slots 20..23 = 77.0
		{op = .Splat, n = 4, dst = 4, a = 1},                     // o0 = 7.0
		{op = .Splat, n = 4, dst = 16, a = 1},                    // o1 = 7.0
		// (a) & (b): Mask_Cond narrowing MASK_EXEC + Mask_Clear invalidating dst
		{op = .Mask_Copy, dst = 2, a = MASK_EXEC},                // m2 = 0xFF (equals r13d)
		{op = .Mask_Cond, dst = MASK_EXEC, a = 2, b = 0},         // r13d = 0 (invalidates m2 == r13d)
		{op = .Store_Zero, n = 4, dst = 4},                       // masked out (r13d == 0)
		{op = .Mask_Copy, dst = MASK_EXEC, a = 2},                // MUST reload m2 (= 0xFF) into r13d
		{op = .Store, n = 4, dst = 4, a = 8},                     // o0 = 99.0
		{op = .Mask_Copy, dst = 3, a = MASK_EXEC},                // m3 = 0xFF (equals r13d)
		{op = .Mask_Clear, dst = 3},                              // m3 = 0 (invalidates m3 == r13d)
		{op = .Mask_Copy, dst = MASK_EXEC, a = 3},                // MUST reload m3 (= 0) into r13d
		{op = .Store_Zero, n = 4, dst = 4},                       // MUST be masked out (r13d == 0)
		// (c): Mask_Or MUST invalidate dst when dst previously equalled r13d!
		//      Right here r13d == 0 and m2 == 0xFF.
		{op = .Mask_Copy, dst = 5, a = MASK_EXEC},                // m5 = 0 (equals r13d == 0)
		{op = .Mask_Or, dst = 5, a = 5, b = 2},                   // m5 = 0 | 0xFF = 0xFF (MUST clear mask_eq_gen[5]!)
		{op = .Mask_Copy, dst = MASK_EXEC, a = 5},                // MUST reload m5 (= 0xFF) into r13d!
		{op = .Store, n = 4, dst = 4, a = 12},                    // o0 = 42.0 (stays 99.0 if Mask_Or kept mask_eq_gen[5])
		// (d): emit_copy_mask else-branch MUST set mask_eq_gen[dst] = 0 (NOT exec_gen)!
		{op = .Mask_Copy, dst = 2, a = MASK_EXEC},                // m2 = 0xFF
		{op = .Mask_Cond, dst = MASK_EXEC, a = 2, b = 0},         // r13d = 0, so m2 (0xFF) != r13d (0)
		{op = .Mask_Copy, dst = 3, a = MASK_EXEC},                // m3 = 0 (pre-equals r13d == 0)
		{op = .Mask_Copy, dst = 3, a = 2},                        // else-branch: m3 = m2 (0xFF); MUST set mask_eq_gen[3] = 0!
		{op = .Mask_Copy, dst = MASK_EXEC, a = 3},                // MUST reload m3 (= 0xFF) into r13d!
		{op = .Store, n = 4, dst = 16, a = 20},                   // o1 = 77.0 (stays 7.0 if else-branch claimed m3 == r13d)
	}
	p_adv := Program {
		code      = adv_code_full[:],
		consts    = adv_consts[:],
		outputs   = outputs2[:],
		num_slots = 24,
		num_masks = 6,
		stage     = .Fragment,
	}
	jit_compile(&p_adv)
	defer jit_destroy(&p_adv)

	tj0, tj1, ti0, ti1: [1][4]f32
	tjs := [2][][4]f32{tj0[:], tj1[:]}
	tis := [2][][4]f32{ti0[:], ti1[:]}
	run_targets(&p_adv, nil, 1, 1, tjs[:], temp_scratch())
	p_adv_int := p_adv
	p_adv_int.jit_body, p_adv_int.jit_init = nil, nil
	run_targets(&p_adv_int, nil, 1, 1, tis[:], temp_scratch())
	testing.expect_value(t, tj0[0], [4]f32{42, 42, 42, 42})
	testing.expect_value(t, tj1[0], [4]f32{77, 77, 77, 77})
	testing.expect(t, jit_vec4_bits_eq(tj0[0], ti0[0]))
	testing.expect(t, jit_vec4_bits_eq(tj1[0], ti1[0]))

	// 3. End-to-end GLSL nested if/else, short-circuit && / ||, and switch.
	src := `#version 310 es
precision highp float;
out vec4 o;
void main() {
	int x = int(floor(gl_FragCoord.x));
	int y = int(floor(gl_FragCoord.y));
	float v = 1.0;
	if ((x > 1 && y < 3) || x == 6) {
		if (x < 4) {
			v += 10.0;
		} else {
			v += 20.0;
		}
	} else {
		v += 5.0;
	}
	switch (x & 3) {
	case 0: v += 100.0; break;
	case 1: v += 200.0;
	case 2: v += 300.0; break;
	default: v += 400.0; break;
	}
	o = vec4(v, float(x), float(y), 1.0);
}
`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil
	oj: [32][4]f32
	oi: [32][4]f32
	run_f32(&m.program, nil, 8, 4, oj[:], temp_scratch())
	run_f32(&p_int, nil, 8, 4, oi[:], temp_scratch())
	for px in 0 ..< 32 {
		testing.expectf(t, jit_vec4_bits_eq(oj[px], oi[px]), "px %d: jit=%v interp=%v", px, oj[px], oi[px])
	}
}

@(test)
test_jit_store_exec_ymask_cache :: proc(t: ^testing.T) {
	// 1. Fails if reverted: after a narrowing Mask_Cond (so MASK_EXEC is not entry mask),
	//    24 Store / Store_Zero ops interleaved with F_Mul(k=2) in one BB share a single
	//    expanded exec YMM mask instead of emitting 24 vpmovsxbd loads (+138 B).
	consts := [3][2]u32 {
		{0, 1},
		{1, transmute(u32)f32(1.5)},
		{2, transmute(u32)f32(2.0)},
	}
	outputs := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 4, n = 4, base = .Float}}
	code_cached: [dynamic]Instr
	defer delete(code_cached)
	code_reexp: [dynamic]Instr
	defer delete(code_reexp)

	append(&code_cached, Instr{op = .Splat, n = 4, dst = 4, a = 1})
	append(&code_cached, Instr{op = .Mask_Cond, dst = MASK_EXEC, a = MASK_EXEC, b = 0})
	append(&code_reexp, Instr{op = .Splat, n = 4, dst = 4, a = 1})
	append(&code_reexp, Instr{op = .Mask_Cond, dst = MASK_EXEC, a = MASK_EXEC, b = 0})

	for i in 0 ..< 24 {
		append(&code_cached, Instr{op = .F_Mul, n = 4, k = 2, dst = 8, a = 4, b = 2})
		append(&code_reexp, Instr{op = .F_Mul, n = 4, k = 2, dst = 8, a = 4, b = 2})
		// In code_reexp, touch MASK_EXEC with Mask_Or(0, 0, 0) (7 B) to force re-expansion (+6 B).
		append(&code_reexp, Instr{op = .Mask_Or, dst = MASK_EXEC, a = MASK_EXEC, b = MASK_EXEC})
		if (i & 3) == 3 {
			append(&code_cached, Instr{op = .Store_Zero, n = 1, dst = 7})
			append(&code_reexp, Instr{op = .Store_Zero, n = 1, dst = 7})
		} else {
			append(&code_cached, Instr{op = .Store, n = 4, dst = 4, a = 8})
			append(&code_reexp, Instr{op = .Store, n = 4, dst = 4, a = 8})
		}
	}

	p_cached := Program {
		code      = code_cached[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 12,
		num_masks = 2,
		stage     = .Fragment,
	}
	jit_compile(&p_cached)
	defer jit_destroy(&p_cached)

	p_reexp := Program {
		code      = code_reexp[:],
		consts    = consts[:],
		outputs   = outputs[:],
		num_slots = 12,
		num_masks = 2,
		stage     = .Fragment,
	}
	jit_compile(&p_reexp)
	defer jit_destroy(&p_reexp)

	when JIT_ENABLED {
		if jit_avx2_ready {
			// 24 Mask_Or(0,0,0) alone is 24 * 10 = 240 B; with 23 saved vpmovsxbd (23 * 6 = 138 B),
			// p_reexp - p_cached is ~378 B. Reverting Item 4a shrinks the gap to 240 B (< 290 B).
			testing.expectf(
				t,
				len(p_cached.jit_mem) + 300 < len(p_reexp.jit_mem),
				"expected cached exec ymask across BB stores: cached=%d B vs reexp=%d B",
				len(p_cached.jit_mem),
				len(p_reexp.jit_mem),
			)
		}
	}

	// 2. Adversarial clobber & MASK_EXEC update test under divergent lanes:
	//    Interleave Store / Store_Zero inside a divergent branch with Cross, integer division,
	//    Load_Buf, dynamic array Load_Dyn, scalar-RHS F_Mul(k=2), and nested narrowing branches.
	src := `#version 310 es
precision highp float;
layout(std430, binding = 0) readonly buffer Buf { float data[]; } buf;
out vec4 o;
void main() {
	int ix = int(floor(gl_FragCoord.x));
	int iy = int(floor(gl_FragCoord.y));
	vec4 acc = vec4(1.0, 2.0, 3.0, 4.0);
	float arr[3];
	arr[0] = 0.5;
	arr[1] = 1.5;
	arr[2] = 2.5;
	if ((ix + iy) % 2 == 0) {
		acc += vec4(10.0);
		vec3 cr = cross(acc.xyz, vec3(float(ix), float(iy), 1.5));
		acc.xyz += cr * 0.25;                   // Store after Cross (clobbers ymm7) + F_Mul(k=2)
		int q = (ix * 7 + 3) / max(iy + 1, 1);  // I_Div (clobbers ymm7)
		acc.w += float(q);                      // Store after I_Div
		float bv = buf.data[uint(ix & 3)];      // Load_Buf (clobbers ymm7)
		acc.x += bv;                            // Store after Load_Buf
		float dv = arr[clamp(ix % 3, 0, 2)];    // Load_Dyn (clobbers ymm7)
		acc.y += dv;                            // Store after Load_Dyn
		if (ix > 3) {
			acc.z = 0.0;                        // Narrowed MASK_EXEC in nested branch
		}
		acc.w += 5.0;                           // Restored MASK_EXEC
	}
	o = acc;
}
`
	m2, errs2, ok2 := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m2)
	testing.expectf(t, ok2, "compile failed:\n%s", format_errors(errs2, context.temp_allocator))
	if !ok2 {
		return
	}
	p_int2 := m2.program
	p_int2.jit_body, p_int2.jit_init = nil, nil
	bwords := [4]u32 {
		transmute(u32)f32(11.0),
		transmute(u32)f32(22.0),
		transmute(u32)f32(33.0),
		transmute(u32)f32(44.0),
	}
	bview := [1][]u32{bwords[:]}
	opts := Run_Options{buffers = bview[:]}
	oj2: [32][4]f32
	oi2: [32][4]f32
	run_f32(&m2.program, nil, 8, 4, oj2[:], temp_scratch(), opts)
	run_f32(&p_int2, nil, 8, 4, oi2[:], temp_scratch(), opts)
	for px in 0 ..< 32 {
		testing.expectf(t, jit_vec4_bits_eq(oj2[px], oi2[px]), "px %d: jit=%v interp=%v", px, oj2[px], oi2[px])
	}

	// 3. Texture / Texture_Bias / Texel_Fetch MUST invalidate exec_ymask_valid:
	//    Inside a divergent branch, Store(n=9) populates ymm7 (exec_ymask_valid = true);
	//    bound texture gather stubs zero ymm7 and unbound callbacks clobber ymm7;
	//    a subsequent Store(n=9) in the same BB MUST re-expand ymm7.
	{
		src_tex := `#version 310 es
precision highp float;
uniform sampler2D s0;
uniform sampler2D s1;
uniform sampler2D s2;
out vec4 o;
void main() {
	int ix = int(floor(gl_FragCoord.x));
	int iy = int(floor(gl_FragCoord.y));
	vec2 uv = (gl_FragCoord.xy + vec2(0.25)) * 0.25;
	mat3 m0 = mat3(9.0);
	mat3 m1 = mat3(8.0);
	mat3 m2 = mat3(7.0);
	if (((ix + iy) & 1) == 0) {
		m0 = mat3(2.0);
		vec4 t0 = texture(s0, uv);
		m0 = mat3(t0.x + 1.0);
		vec4 t1 = textureLod(s1, uv, 0.0);
		m1 = mat3(t1.y + 2.0);
		vec4 t2 = texelFetch(s2, ivec2(ix & 3, iy & 3), 0);
		m2 = mat3(t2.z + 3.0);
	}
	o = vec4(m0[0][0], m1[1][1], m2[2][2], m0[2][2] + m1[0][0] + m2[0][0]);
}
`
		mt, et, okt := compile(src_tex, {profile = .Glsl, stage = .Fragment})
		defer destroy(mt)
		testing.expectf(t, okt, "src_tex compile failed:\n%s", format_errors(et, context.temp_allocator))
		if !okt {
			return
		}
		tex_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			// Actively overwrite SIMD registers (including ymm7) with all-zeros / non-mask values.
			z := #simd[8]f32{0, 0, 0, 0, 0, 0, 0, 0}
			arr := transmute([8]f32)z
			b := f32(channel + 1) + arr[0]
			return {uv.x + b, uv.y + b, lod + b * 0.5, 1.0}
		}
		tdata: [16 * 4]f32
		for i in 0 ..< 16 {
			tdata[i * 4 + 0] = f32(i + 1) * 0.5
			tdata[i * 4 + 1] = f32(i + 2) * 0.25
			tdata[i * 4 + 2] = f32(i + 3) * 0.75
			tdata[i * 4 + 3] = 1.0
		}
		sizes := [3][3]i32{{4, 4, 1}, {4, 4, 1}, {4, 4, 1}}
		imgs_bound := [3]Channel_Image {
			{data = tdata[:], width = 4, height = 4, components = 4, filter = .Linear},
			{data = tdata[:], width = 4, height = 4, components = 4, filter = .Nearest},
			{data = tdata[:], width = 4, height = 4, components = 4, filter = .Nearest},
		}
		for opts_t, ti in ([2]Run_Options {
			{sampler = tex_cb, channel_sizes = sizes[:], channel_images = imgs_bound[:]},
			{sampler = tex_cb, channel_sizes = sizes[:]},
		}) {
			ojt, oit: [32][4]f32
			run_f32(&mt.program, nil, 8, 4, ojt[:], temp_scratch(), opts_t)
			run_f32_interp(&mt.program, nil, 8, 4, oit[:], temp_scratch(), opts_t)
			for px in 0 ..< 32 {
				testing.expectf(t, jit_vec4_bits_eq(ojt[px], oit[px]), "tex case %d px %d: jit=%v interp=%v", ti, px, ojt[px], oit[px])
			}
		}
	}

	// 4. cache_reset MUST clear exec_ymask_valid across basic-block boundaries and helper splits:
	//    (a) 3-BB CFG where BB 0 narrows r13d, zeroes ymm7 via Cross, and jumps over BB 1
	//        (whose Store(n=8) sets exec_ymask_valid = true) directly to BB 2's Store(n=8);
	//    (b) GLSL divergent branch with helper `atan(y, x)` splitting the BB between two mat3 stores.
	{
		consts_bb := [5][2]u32 {
			{4, transmute(u32)f32(4.0)},
			{5, transmute(u32)f32(0.0)},
			{6, transmute(u32)f32(9.0)},
			{7, transmute(u32)f32(11.0)},
			{24, transmute(u32)f32(33.0)},
		}
		inputs_bb := [1]Program_Var{{kind = .Frag_Coord, slot = 0, n = 4, base = .Float}}
		outputs_bb := [2]Program_Var {
			{kind = .Stage_Out, location = 0, slot = 8, n = 4, base = .Float},
			{kind = .Stage_Out, location = 1, slot = 12, n = 4, base = .Float},
		}
		code_bb := [?]Instr {
			// BB 0: init dst 8..15 = 9.0, src_a 16..23 = 11.0, src_b 25..32 = 33.0, zero 33..35 = 0.0
			{op = .Splat, n = 8, dst = 8, a = 6},
			{op = .Splat, n = 8, dst = 16, a = 7},
			{op = .Splat, n = 8, dst = 25, a = 24},
			{op = .Splat, n = 3, dst = 33, a = 5},
			{op = .F_Lt, n = 1, dst = 36, a = 0, b = 4},              // Frag_Coord.x < 4.0 (lanes 0..3)
			{op = .Mask_Cond, dst = MASK_EXEC, a = MASK_EXEC, b = 36},
			{op = .Cross, n = 3, dst = 37, a = 33, b = 33},           // zeroes ymm7!
			{op = .Jump, a = 9},                                      // jump to BB 2 (pc 9)
			// BB 1 (pc 8): sets exec_ymask_valid = true at end of BB 1!
			{op = .Store, n = 8, dst = 8, a = 16},
			// BB 2 (pc 9): jump target -> cache_reset MUST clear exec_ymask_valid!
			{op = .Store, n = 8, dst = 8, a = 25},
		}
		p_bb := Program {
			stage     = .Fragment,
			num_slots = 40,
			num_masks = 2,
			consts    = consts_bb[:],
			inputs    = inputs_bb[:],
			outputs   = outputs_bb[:],
			code      = code_bb[:],
		}
		jit_compile(&p_bb)
		defer jit_destroy(&p_bb)
		p_bb_int := p_bb
		p_bb_int.jit_body, p_bb_int.jit_init = nil, nil
		bj0, bj1, bi0, bi1: [8][4]f32
		tjb := [2][][4]f32{bj0[:], bj1[:]}
		tib := [2][][4]f32{bi0[:], bi1[:]}
		run_targets(&p_bb, nil, 8, 1, tjb[:], temp_scratch())
		run_targets(&p_bb_int, nil, 8, 1, tib[:], temp_scratch())
		for px in 0 ..< 8 {
			testing.expectf(t, jit_vec4_bits_eq(bj0[px], bi0[px]), "bb-reset t0 px %d: jit=%v interp=%v", px, bj0[px], bi0[px])
			testing.expectf(t, jit_vec4_bits_eq(bj1[px], bi1[px]), "bb-reset t1 px %d: jit=%v interp=%v", px, bj1[px], bi1[px])
		}

		src_hlp := `#version 310 es
precision highp float;
out vec4 o;
void main() {
	int ix = int(floor(gl_FragCoord.x));
	mat3 m = mat3(9.0);
	if ((ix & 1) == 0) {
		m = mat3(2.0);
		float a = atan(float(ix) + 1.0, 2.0);
		m = mat3(a + 3.0);
	}
	o = vec4(m[0][0], m[1][1], m[2][2], 1.0);
}
`
		mh, eh, okh := compile(src_hlp, {profile = .Glsl, stage = .Fragment})
		defer destroy(mh)
		testing.expectf(t, okh, "src_hlp compile failed:\n%s", format_errors(eh, context.temp_allocator))
		if okh {
			ojh, oih: [32][4]f32
			run_f32(&mh.program, nil, 8, 4, ojh[:], temp_scratch())
			run_f32_interp(&mh.program, nil, 8, 4, oih[:], temp_scratch(), {})
			for px in 0 ..< 32 {
				testing.expectf(t, jit_vec4_bits_eq(ojh[px], oih[px]), "hlp-reset px %d: jit=%v interp=%v", px, ojh[px], oih[px])
			}
		}
	}
}

@(test)
test_jit_entry_unmasked_stores :: proc(t: ^testing.T) {
	// 1. Static codegen check: entry Store_Zero / Store before any jump target or
	//    MASK_EXEC write omit masked blends (vpandn / vpblendvb + old-value loads).
	{
		inputs := [2]Program_Var {
			{kind = .Stage_In, location = 0, slot = 0, n = 4, base = .Float},
			{kind = .Stage_In, location = 1, slot = 12, n = 1, base = .Float},
		}
		outputs := [2]Program_Var {
			{kind = .Stage_Out, location = 0, slot = 4, n = 4, base = .Float},
			{kind = .Stage_Out, location = 1, slot = 8, n = 4, base = .Float},
		}
		code_entry := [2]Instr {
			{op = .Store_Zero, n = 4, dst = 4},
			{op = .Store, n = 4, dst = 8, a = 0},
		}
		code_narrow := [3]Instr {
			{op = .Mask_Cond, k = 0, dst = MASK_EXEC, a = MASK_EXEC, b = 12},
			{op = .Store_Zero, n = 4, dst = 4},
			{op = .Store, n = 4, dst = 8, a = 0},
		}
		p_entry := Program {
			stage     = .Fragment,
			num_slots = 16,
			num_masks = 2,
			inputs    = inputs[:],
			outputs   = outputs[:],
			code      = code_entry[:],
		}
		jit_compile(&p_entry)
		defer jit_destroy(&p_entry)

		p_narrow := Program {
			stage     = .Fragment,
			num_slots = 16,
			num_masks = 2,
			inputs    = inputs[:],
			outputs   = outputs[:],
			code      = code_narrow[:],
		}
		jit_compile(&p_narrow)
		defer jit_destroy(&p_narrow)

		when JIT_ENABLED {
			if jit_avx2_ready {
				// Mask_Cond itself is <= 24 B; the 8 masked stores (4 vpandn + 4 vpblendvb + 8 old loads + ensure_exec_ymask)
				// add > 70 B on top of that. Fails if 4b is reverted!
				testing.expectf(
					t,
					len(p_entry.jit_mem) + 60 < len(p_narrow.jit_mem),
					"expected unmasked entry stores to be >60 B smaller: entry=%d B vs narrowed=%d B",
					len(p_entry.jit_mem),
					len(p_narrow.jit_mem),
				)
			}
		}
	}

	// 2. Adversarial proof-failure case A: pc == 0 is a backward jump target!
	//    Textually there is no write to MASK_EXEC before pc 0, yet iteration 2 enters pc 0
	//    with narrowed MASK_EXEC, so Store and Store_Zero at pc 0..1 MUST remain masked.
	{
		code := [?]Instr {
			// pc 0: backward loop header! Must NOT use unmasked Store/Store_Zero!
			{op = .Store, n = 1, dst = 2, a = 1},
			{op = .Store_Zero, n = 1, dst = 3},
			// pc 2: increment iter counter (slot 1 += slot 4 (1.0))
			{op = .F_Add, n = 1, dst = 1, a = 1, b = 4},
			// pc 3..4: keep only lanes where slot 1 < slot 0 (per-lane trip limit)
			{op = .F_Lt, n = 1, dst = 5, a = 1, b = 0},
			{op = .Mask_Cond, k = 0, dst = MASK_EXEC, a = MASK_EXEC, b = 5},
			// Restore slot 3 to slot 1 on still-active lanes so dropped lanes keep 0.0 from iter 1
			// while if pc 0..1 were unmasked on iter 2, slot 2/3 on dropped lanes would be clobbered!
			{op = .Store, n = 1, dst = 3, a = 1},
			{op = .Jump_If_None, dst = MASK_EXEC, a = 8},
			{op = .Jump, a = 0},
		}
		consts := [?][2]u32{
			{1, transmute(u32)f32(0.0)},
			{4, transmute(u32)f32(1.0)},
		}
		inputs := [?]Program_Var{{kind = .Frag_Coord, slot = 0, n = 4, base = .Float}}
		outputs := [?]Program_Var{{kind = .Stage_Out, location = 0, slot = 2, n = 2, base = .Float}}
		p := Program {
			stage     = .Fragment,
			num_slots = 8,
			num_masks = 2,
			consts    = consts[:],
			inputs    = inputs[:],
			outputs   = outputs[:],
			code      = code[:],
		}
		jit_compile(&p)
		defer jit_destroy(&p)
		p_int := p
		p_int.jit_body, p_int.jit_init = nil, nil
		oj, oi: [16][4]f32
		run_f32(&p, nil, 4, 4, oj[:], temp_scratch())
		run_f32(&p_int, nil, 4, 4, oi[:], temp_scratch())
		for px in 0 ..< 16 {
			testing.expectf(t, oj[px] == oi[px], "pc0-loop px %d: jit=%v interp=%v", px, oj[px], oi[px])
		}
	}

	// 3. Adversarial proof-failure case B + partial-lane entry batches (quads=3 with helpers,
	//    dFdx/dFdy, 2 targets, plus 5x3 run_f32): entry Store_Zero/Store followed by narrowing branch.
	{
		src := `#version 310 es
precision highp float;
in float v;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
	vec4 a = vec4(0.0);                       // entry Store_Zero
	vec4 b = vec4(v, v * 2.0, 3.0, 1.0);      // entry Store
	float dx = dFdx(b.x);
	float dy = dFdy(b.y);
	if (gl_FragCoord.x > 2.0) {               // narrows MASK_EXEC: subsequent stores MUST be masked
		a = vec4(10.0, 20.0, dx, dy);
		b.zw = vec2(0.0);                     // Store_Zero / Store under narrowed MASK_EXEC
	}
	o0 = a + vec4(dx, dy, 0.0, 1.0);
	o1 = b;
}
`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
		if !ok {
			return
		}
		p_int := m.program
		p_int.jit_body, p_int.jit_init = nil, nil

		origin := [3][2]i32{{0, 0}, {2, 0}, {4, 2}}
		helpers := [3]Quad_Lanes{{2}, {}, {1, 3}}
		vx: [12]f32
		for i in 0 ..< 12 {
			vx[i] = f32(i) * 0.75 + 1.0
		}
		SENT := [4]f32{-777.0, -888.0, -999.0, -111.0}
		bj, bi: [2][12][4]f32
		sh_j, sh_i: [3]Quad_Lanes
		for p in 0 ..< 12 {
			bj[0][p], bj[1][p] = SENT, SENT
			bi[0][p], bi[1][p] = SENT, SENT
		}
		tj := [2][][4]f32{bj[0][:], bj[1][:]}
		ti := [2][][4]f32{bi[0][:], bi[1][:]}
		quads := Fragment_Quads{count = 3, origin = origin[:], helpers = helpers[:], varyings = vx[:]}
		run_fragment(&m.program, nil, quads, tj[:], sh_j[:], temp_scratch())
		run_fragment(&p_int, nil, quads, ti[:], sh_i[:], temp_scratch())
		testing.expect(t, slice.equal(sh_j[:], sh_i[:]))
		for p in 0 ..< 12 {
			testing.expectf(t, bj[0][p] == bi[0][p], "t0 px %d: jit=%v interp=%v", p, bj[0][p], bi[0][p])
			testing.expectf(t, bj[1][p] == bi[1][p], "t1 px %d: jit=%v interp=%v", p, bj[1][p], bi[1][p])
		}
	}
}

@(test)
test_jit_store_runtime_all_lanes :: proc(t: ^testing.T) {
	// 1. All-lanes (r13d == 0xFF) vs partial-lanes (0x0F odd quad / divergent) Store & Store_Zero.
	src := `#version 310 es
precision highp float;
uniform int u_mode;
out vec4 o;
void main() {
	vec4 acc = vec4(1.0, 2.0, 3.0, 4.0);
	int ix = int(floor(gl_FragCoord.x));
	int iy = int(floor(gl_FragCoord.y));
	bool take = (u_mode == 0) ? (ix >= 0) : (((ix + iy) & 1) == 0);
	if (take) {
		acc = vec4(0.0); // non-entry Store_Zero
		for (int k = 0; k < 4; k++) {
			acc += vec4(float(k + 1), float(ix), float(iy), 0.5); // non-entry Store (cached)
			if (k == 2 && ix > 4) {
				acc.zw = vec2(0.0); // nested narrowed Store_Zero
				acc.xy += vec2(7.0, 9.0);
			}
		}
	}
	o = acc;
}
`
	m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m)
	testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
	if !ok {
		return
	}
	when JIT_ENABLED {
		if jit_avx2_ready {
			// Verify non-entry Store / Store_Zero emitted the r13d == 0xFF fast-path check (41 81 FD FF 00 00 00).
			sig := [7]u8{0x41, 0x81, 0xFD, 0xFF, 0x00, 0x00, 0x00}
			hits := 0
			mem_code := m.program.jit_mem
			for i := 0; i + len(sig) <= len(mem_code); i += 1 {
				if slice.equal(mem_code[i:i + len(sig)], sig[:]) {
					hits += 1
				}
			}
			testing.expectf(t, hits >= 3, "expected >=3 runtime r13d==0xFF store checks, found %d", hits)
		}
	}
	p_int := m.program
	p_int.jit_body, p_int.jit_init = nil, nil
	for mode in ([2]i32{0, 1}) {
		blk := [4]u8{}
		(^i32)(&blk[0])^ = mode
		oj, oi: [30][4]f32
		run_f32(&m.program, blk[:], 10, 3, oj[:], temp_scratch())
		run_f32(&p_int, blk[:], 10, 3, oi[:], temp_scratch())
		for px in 0 ..< 30 {
			testing.expectf(t, jit_vec4_bits_eq(oj[px], oi[px]), "mode=%d px=%d: jit=%v interp=%v", mode, px, oj[px], oi[px])
		}
	}

	// 2. Regression tests for the r13d == 0xFF fast-path leaving exec_ymask_valid stale:
	//    (a) Exact mat3 repro at 16x4 with u = 2 (I_Div clobbers ymm7 -> Store(n=1) takes
	//        r13d==0xFF fast path -> Store(n=9) mat3 MUST still expand ymm7);
	//    (b) 2-vec4 struct assignment (n = 8) after I_Div + Store(n=1);
	//    (c) Store_Zero(n > 4) after Cross + Store_Zero(n <= 4) on the r13d==0xFF fast path
	//        (tested both in GLSL and via explicit IR Store_Zero(n=1) -> Store_Zero(n=8)).
	{
		src_mat3 := `#version 310 es
precision highp float;
uniform int u;
out vec4 o;
void main() {
	mat3 m = mat3(5.0);
	float x = 0.0;
	int ix = int(gl_FragCoord.x);
	if (ix >= 0) {
		int q = ix / (u + 3);
		x = float(q);
		m = mat3(float(q) + 1.0);
	}
	o = vec4(x, m[0][0], m[1][1], m[2][2]);
}
`
		mm, em, okm := compile(src_mat3, {profile = .Glsl, stage = .Fragment})
		defer destroy(mm)
		testing.expectf(t, okm, "src_mat3 compile failed:\n%s", format_errors(em, context.temp_allocator))
		if okm {
			blk: [4]u8
			(^i32)(&blk[0])^ = 2
			oj, oi: [64][4]f32
			run_f32(&mm.program, blk[:], 16, 4, oj[:], temp_scratch())
			run_f32_interp(&mm.program, blk[:], 16, 4, oi[:], temp_scratch(), {})
			testing.expect_value(t, oj[0], [4]f32{0, 1, 1, 1})
			for px in 0 ..< 64 {
				testing.expectf(t, jit_vec4_bits_eq(oj[px], oi[px]), "mat3 repro px %d: jit=%v interp=%v", px, oj[px], oi[px])
			}
		}

		src_struct := `#version 310 es
precision highp float;
struct Pair {
	vec4 a;
	vec4 b;
};
uniform int u;
out vec4 o;
void main() {
	Pair p = Pair(vec4(5.0), vec4(6.0));
	Pair pz = Pair(vec4(7.0), vec4(8.0));
	float x = 0.0;
	float z = 9.0;
	int ix = int(gl_FragCoord.x);
	if (ix >= 0) {
		int q = ix / (u + 3);
		x = float(q);
		p = Pair(vec4(float(q) + 1.0), vec4(float(q) + 2.0));
	}
	if (ix >= 0) {
		int q2 = ix / (u + 3);
		z = 0.0;
		pz = Pair(vec4(0.0), vec4(float(q2) * 0.0));
	}
	o = vec4(x + z, p.a.x + pz.a.x, p.b.y + pz.b.y, p.b.w + pz.b.w);
}
`
		ms, es, oks := compile(src_struct, {profile = .Glsl, stage = .Fragment})
		defer destroy(ms)
		testing.expectf(t, oks, "src_struct compile failed:\n%s", format_errors(es, context.temp_allocator))
		if oks {
			blk: [4]u8
			(^i32)(&blk[0])^ = 2
			oj, oi: [64][4]f32
			run_f32(&ms.program, blk[:], 16, 4, oj[:], temp_scratch())
			run_f32_interp(&ms.program, blk[:], 16, 4, oi[:], temp_scratch(), {})
			for px in 0 ..< 64 {
				testing.expectf(t, jit_vec4_bits_eq(oj[px], oi[px]), "struct repro px %d: jit=%v interp=%v", px, oj[px], oi[px])
			}
		}

		// Explicit IR Store_Zero(n=1) -> Store_Zero(n=8) after Cross zeroes ymm7:
		consts_sz := [3][2]u32 {
			{0, 1},
			{1, transmute(u32)f32(0.0)},
			{2, transmute(u32)f32(7.0)},
		}
		outputs_sz := [2]Program_Var {
			{kind = .Stage_Out, location = 0, slot = 8, n = 4, base = .Float},
			{kind = .Stage_Out, location = 1, slot = 12, n = 4, base = .Float},
		}
		code_sz := [?]Instr {
			{op = .Splat, n = 1, dst = 4, a = 2},                      // slot 4 = 7.0
			{op = .Splat, n = 8, dst = 8, a = 2},                      // slots 8..15 = 7.0
			{op = .Splat, n = 3, dst = 16, a = 1},                     // slots 16..18 = 0.0
			{op = .Mask_Cond, dst = MASK_EXEC, a = MASK_EXEC, b = 0},  // ends entry_exec_end; r13d = 0xFF
			{op = .Cross, n = 3, dst = 19, a = 16, b = 16},            // zeroes ymm7; exec_ymask_valid = false
			{op = .Store_Zero, n = 1, dst = 4},                        // fast path (r13d == 0xFF) MUST NOT leave exec_ymask_valid = true!
			{op = .Store_Zero, n = 8, dst = 8},                        // n > 4 MUST expand ymm7 and zero 8..15!
			{op = .F_Add, n = 1, dst = 8, a = 8, b = 4},
		}
		p_sz := Program {
			stage     = .Fragment,
			num_slots = 24,
			num_masks = 2,
			consts    = consts_sz[:],
			outputs   = outputs_sz[:],
			code      = code_sz[:],
		}
		jit_compile(&p_sz)
		defer jit_destroy(&p_sz)
		p_sz_int := p_sz
		p_sz_int.jit_body, p_sz_int.jit_init = nil, nil
		zj0, zj1, zi0, zi1: [16][4]f32
		for &px in zj0 {px = {-1, -1, -1, -1}}
		for &px in zj1 {px = {-1, -1, -1, -1}}
		tjz := [2][][4]f32{zj0[:], zj1[:]}
		tiz := [2][][4]f32{zi0[:], zi1[:]}
		run_targets(&p_sz, nil, 16, 1, tjz[:], temp_scratch())
		run_targets(&p_sz_int, nil, 16, 1, tiz[:], temp_scratch())
		for px in 0 ..< 16 {
			testing.expect_value(t, zj0[px], [4]f32{0, 0, 0, 0})
			testing.expect_value(t, zj1[px], [4]f32{0, 0, 0, 0})
			testing.expect(t, jit_vec4_bits_eq(zj0[px], zi0[px]))
			testing.expect(t, jit_vec4_bits_eq(zj1[px], zi1[px]))
		}
	}
}

@(test)
test_jit_liveness_cross_bb :: proc(t: ^testing.T) {
	// 1. Fails if reverted: packed-slot reuse across two basic blocks (slots 4..11 dead at BB 0 exit,
	//    reused in BB 1). Coarse per-slot cross_bb marks 4..11 cross-BB and emits 16 extra [rbx] stores
	//    (+96..128 B), whereas CFG liveness keeps both blocks register-only.
	{
		inputs := [1]Program_Var{{kind = .Stage_In, location = 0, slot = 0, n = 4, base = .Float}}
		outputs := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 20, n = 4, base = .Float}}
		make_reuse_prog :: proc(tb1: i32, inp: []Program_Var, outp: []Program_Var) -> (Program, [dynamic]Instr) {
			c: [dynamic]Instr
			append(&c, Instr{op = .F_Add, n = 4, dst = 4, a = 0, b = 0})
			append(&c, Instr{op = .F_Mul, n = 4, dst = 8, a = 4, b = 0})
			append(&c, Instr{op = .F_Sub, n = 4, dst = 20, a = 8, b = 4})
			append(&c, Instr{op = .Jump, a = 4})
			append(&c, Instr{op = .F_Mul, n = 4, dst = tb1, a = 0, b = 0})
			append(&c, Instr{op = .F_Add, n = 4, dst = tb1 + 4, a = tb1, b = 0})
			append(&c, Instr{op = .F_Add, n = 4, dst = 20, a = 20, b = tb1 + 4})
			return Program {
				stage     = .Fragment,
				num_slots = 24,
				num_masks = 2,
				inputs    = inp,
				outputs   = outp,
				code      = c[:],
			}, c
		}
		p_reused, c1 := make_reuse_prog(4, inputs[:], outputs[:])
		defer {
			delete(c1)
			jit_destroy(&p_reused)
		}
		p_distinct, c2 := make_reuse_prog(12, inputs[:], outputs[:])
		defer {
			delete(c2)
			jit_destroy(&p_distinct)
		}
		jit_compile(&p_reused)
		jit_compile(&p_distinct)
		when JIT_ENABLED {
			if jit_avx2_ready {
				testing.expectf(
					t,
					len(p_reused.jit_mem) <= len(p_distinct.jit_mem) + 16,
					"expected packed-slot reuse across BBs to omit dead [rbx] stores: reused=%d B vs distinct=%d B",
					len(p_reused.jit_mem),
					len(p_distinct.jit_mem),
				)
			}
		}
	}

	// 2. Adversarial proof-negative cases where values ARE live across BB boundaries:
	//    - value live across an if/else diamond join;
	//    - loop-carried value live across a loop back-edge;
	//    - dynamic array (dyn_slots) next to BB-local temporaries;
	//    - whole >16-component writes (`float[20]` and 20-slot `Big` struct -> Store with n=20,
	//      exercising `idx >= 16` in `need_mem_write`) live out of an `if` block and read after it;
	//    - partial-lane discard after writing stage output.
	{
		src := `#version 310 es
precision highp float;
struct Big {
	mat4 m;
	vec4 tail;
};
out vec4 o;
void main() {
	int ix = int(floor(gl_FragCoord.x));
	int iy = int(floor(gl_FragCoord.y));
	vec4 pre_diamond = vec4(float(ix) * 0.5, float(iy) * 0.25, 1.5, 2.5);
	float arr[3];
	arr[0] = pre_diamond.x + 1.0;
	arr[1] = pre_diamond.y + 2.0;
	arr[2] = 3.5;

	float fa20[20] = float[20](
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
		11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
	);
	Big bg = Big(mat4(1.0), vec4(4.0, 5.0, 6.0, 7.0));
	if (((ix + iy) & 1) == 0) {
		fa20 = float[20](
			101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
			111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0
		);
		bg = Big(mat4(2.0), vec4(9.0, 8.0, 7.0, 6.0));
	}
	vec4 hi20 = vec4(fa20[16], fa20[17], fa20[18], fa20[19]) * 0.01;

	// Loop-carried state across back-edge + dynamic array indexing inside loop.
	vec4 carry = bg.m[3] * 0.1 + bg.tail + hi20;
	for (int k = 0; k < 4; k++) {
		vec4 tmp = vec4(float(k), arr[clamp((ix + k) % 3, 0, 2)], carry.z * 0.5, 1.0);
		if (k == 2 && ix == 1 && iy == 1) {
			o = vec4(-1.0);
			discard;
		}
		carry = carry * 0.75 + tmp;
	}
	o = carry + pre_diamond;
}
`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
		if !ok {
			return
		}
		has_wide_store := false
		for ins in m.program.code {
			if ins.op == .Store && ins.n >= 20 {
				has_wide_store = true
				break
			}
		}
		testing.expect(t, has_wide_store, "expected whole-aggregate Store with n >= 20")

		p_int := m.program
		p_int.jit_body, p_int.jit_init = nil, nil
		oj, oi: [36][4]f32
		for &px in oj {px = {-99, -99, -99, -99}}
		for &px in oi {px = {-99, -99, -99, -99}}
		run_f32(&m.program, nil, 6, 6, oj[:], temp_scratch())
		run_f32(&p_int, nil, 6, 6, oi[:], temp_scratch())
		for px in 0 ..< 36 {
			testing.expectf(t, jit_vec4_bits_eq(oj[px], oi[px]), "px %d: jit=%v interp=%v", px, oj[px], oi[px])
		}
	}
}

@(test)
test_jit_texture_same_bb :: proc(t: ^testing.T) {
	// 1. Relative two-program comparison: 6 dirty accumulators (10..15) + coords (8..9)
	//    stay in ymm8..15 across 6 Texel_Fetch ops in one BB vs the identical sequence with
	//    BB-splitting fallthrough Jumps after each Texel_Fetch (which spill & reload [rbx]).
	{
		consts := [1][2]u32{{6, 0}}
		outputs := [1]Program_Var{{slot = 20, n = 4}}
		code_same := make([dynamic]Instr, 0, 56, context.temp_allocator)
		code_split := make([dynamic]Instr, 0, 64, context.temp_allocator)
		for k in i32(0) ..< 8 {
			append(&code_same, Instr{op = .I_Add, n = 1, dst = 8 + k, a = k, b = 6})
			append(&code_split, Instr{op = .I_Add, n = 1, dst = 8 + k, a = k, b = 6})
		}
		for _ in i32(0) ..< 6 {
			append(&code_same, Instr{op = .Texel_Fetch, n = 2, k = 0, dst = 16, a = 8, b = 6})
			append(&code_same, Instr{op = .F_Add, n = 4, dst = 12, a = 12, b = 16})
			append(&code_same, Instr{op = .F_Add, n = 2, dst = 10, a = 10, b = 12})

			append(&code_split, Instr{op = .Jump, a = i32(len(code_split) + 1)})
			append(&code_split, Instr{op = .Texel_Fetch, n = 2, k = 0, dst = 16, a = 8, b = 6})
			append(&code_split, Instr{op = .Jump, a = i32(len(code_split) + 1)})
			append(&code_split, Instr{op = .F_Add, n = 4, dst = 12, a = 12, b = 16})
			append(&code_split, Instr{op = .F_Add, n = 2, dst = 10, a = 10, b = 12})
		}
		append(&code_same, Instr{op = .F_Add, n = 2, dst = 12, a = 12, b = 10})
		append(&code_same, Instr{op = .Mov, n = 4, dst = 20, a = 12})
		append(&code_split, Instr{op = .F_Add, n = 2, dst = 12, a = 12, b = 10})
		append(&code_split, Instr{op = .Mov, n = 4, dst = 20, a = 12})

		p_same := Program {
			code      = code_same[:],
			consts    = consts[:],
			outputs   = outputs[:],
			num_slots = 24,
			num_masks = 2,
			stage     = .Fragment,
		}
		jit_compile(&p_same)
		defer jit_destroy(&p_same)

		p_split := Program {
			code      = code_split[:],
			consts    = consts[:],
			outputs   = outputs[:],
			num_slots = 24,
			num_masks = 2,
			stage     = .Fragment,
		}
		jit_compile(&p_split)
		defer jit_destroy(&p_split)

		when JIT_ENABLED {
			if jit_avx2_ready {
				testing.expectf(
					t,
					len(p_same.jit_mem) * 8 < len(p_split.jit_mem) * 7,
					"expected single-BB texture caching to shrink emission: same_bb=%d B vs split_bb=%d B",
					len(p_same.jit_mem),
					len(p_split.jit_mem),
				)
			}
		}
	}

	// 2. Adversarial mixed bound + unbound callback execution in one basic block:
	//    - all 8 ymm8..15 occupied by BB-local dirty values across texture(), textureLod()
	//      on sampler2DArray, and texelFetch();
	//    - computed dirty coords, bias, lod, and array layer read by unbound callbacks;
	//    - sampler callback actively clobbers YMM registers via #simd[8]f32 arithmetic.
	{
		src := `#version 310 es
precision highp float;
uniform sampler2D s0;
uniform sampler2D s1;
uniform highp sampler2DArray s2;
out vec4 o;
void main() {
	float x = gl_FragCoord.x;
	float y = gl_FragCoord.y;
	float k0 = x * 0.25 + 1.0;
	float k1 = y * 0.50 + 2.0;
	float k2 = (x + y) * 0.125 + 3.0;
	float k3 = (x - y) * 0.375 + 4.0;
	float k4 = x * y * 0.0625 + 5.0;
	float k5 = x * 0.75 - y * 0.25 + 6.0;
	float k6 = y * 0.75 - x * 0.125 + 7.0;
	float k7 = (x * 2.0 + y) * 0.1 + 8.0;

	vec2 uv = vec2(fract(k0 * 0.31), fract(k1 * 0.47));
	float bias = k2 * 0.1;
	vec4 t0 = texture(s0, uv, bias);

	float layer = floor(mod(x + y, 3.0));
	float lod = k3 * 0.25;
	vec4 t2 = textureLod(s2, vec3(uv.yx, layer), lod);

	ivec2 ip = ivec2(int(x) & 3, int(y) & 3);
	int ilod = int(x) & 1;
	vec4 t1 = texelFetch(s1, ip, ilod);

	float keep = k0 + k1 * 1.1 + k2 * 1.2 + k3 * 1.3 + k4 * 1.4 + k5 * 1.5 + k6 * 1.6 + k7 * 1.7;
	o = t0 * 0.5 + t1 * 0.75 + t2 * 0.25 + vec4(keep, k4 - k5, k6 - k7, 1.0);
}
`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "adv compile failed:\n%s", format_errors(errs, context.temp_allocator))
		if !ok {
			return
		}

		clobber_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			// Clobber wide SIMD registers inside the callback to verify ymm8..15 save/restore.
			v0 := #simd[8]f32{uv.x, uv.y, lod, f32(layer), f32(channel), 1.25, 2.5, 3.75}
			v1 := v0 * #simd[8]f32{3, 5, 7, 11, 13, 17, 19, 23} + #simd[8]f32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
			v2 := v1 * v0 - v1
			arr := transmute([8]f32)v2
			base := f32(channel + 1) + arr[0] * 0.01 + lod * 0.5 + f32(layer) * 2.0
			return {base + uv.x, base + uv.y * 2.0, base - lod * 0.25, 1.0}
		}

		img0_data: [16 * 4]f32
		img1_data: [16 * 4]f32
		for i in 0 ..< 16 {
			img0_data[i * 4 + 0] = f32(i) * 0.25
			img0_data[i * 4 + 1] = f32(16 - i) * 0.125
			img0_data[i * 4 + 2] = f32(i % 5) * 0.5
			img0_data[i * 4 + 3] = 1.0
			img1_data[i * 4 + 0] = f32(i) * 1.5 + 3.0
			img1_data[i * 4 + 1] = f32(i) * 0.5 + 1.0
			img1_data[i * 4 + 2] = f32(i & 3) * 2.0
			img1_data[i * 4 + 3] = 0.75
		}
		sizes := [3][3]i32{{4, 4, 1}, {4, 4, 1}, {4, 4, 3}}
		imgs_both := [2]Channel_Image {
			{data = img0_data[:], width = 4, height = 4, components = 4, filter = .Linear},
			{data = img1_data[:], width = 4, height = 4, components = 4, filter = .Nearest},
		}
		imgs_only0 := [2]Channel_Image {
			{data = img0_data[:], width = 4, height = 4, components = 4, filter = .Nearest},
			{},
		}
		imgs_only1 := [2]Channel_Image {
			{},
			{data = img1_data[:], width = 4, height = 4, components = 4, filter = .Nearest},
		}

		for opts, case_idx in ([4]Run_Options {
			{sampler = clobber_cb, channel_sizes = sizes[:]},
			{sampler = clobber_cb, channel_sizes = sizes[:], channel_images = imgs_both[:]},
			{sampler = clobber_cb, channel_sizes = sizes[:], channel_images = imgs_only0[:]},
			{sampler = clobber_cb, channel_sizes = sizes[:], channel_images = imgs_only1[:]},
		}) {
			oj, oi: [32][4]f32
			run_f32(&m.program, nil, 8, 4, oj[:], temp_scratch(), opts)
			run_f32_interp(&m.program, nil, 8, 4, oi[:], temp_scratch(), opts)
			for px in 0 ..< 32 {
				for c in 0 ..< 4 {
					diff := abs(oj[px][c] - oi[px][c])
					testing.expectf(
						t,
						diff <= 1e-5,
						"case %d px %d comp %d: jit=%v interp=%v",
						case_idx, px, c, oj[px], oi[px],
					)
				}
			}
		}
	}
}

@(test)
test_jit_texture_dead_components :: proc(t: ^testing.T) {
	// 1. Fails if reverted: reading only .r from 2 texelFetch + 2 texture ops must emit
	//    substantially smaller bound-channel stubs than reading all .rgba (> 280 B smaller).
	{
		src_r := `#version 310 es
precision highp float;
uniform sampler2D s0;
uniform sampler2D s1;
uniform sampler2D s2;
out vec4 o;
void main() {
	ivec2 ip = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y));
	vec2 uv = gl_FragCoord.xy * 0.125;
	float r0 = texelFetch(s0, ip, 0).r;
	float r1 = texelFetch(s0, ip + ivec2(1, 0), 0).r;
	float r2 = texture(s1, uv).r;
	float r3 = texture(s1, uv + vec2(0.25)).r;
	vec4 unused_tap = texture(s2, uv);
	o = vec4(r0 + r1 + r2 + r3, 0.0, 0.0, 1.0);
}
`
		src_rgba := `#version 310 es
precision highp float;
uniform sampler2D s0;
uniform sampler2D s1;
uniform sampler2D s2;
out vec4 o;
void main() {
	ivec2 ip = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y));
	vec2 uv = gl_FragCoord.xy * 0.125;
	vec4 a0 = texelFetch(s0, ip, 0);
	vec4 a1 = texelFetch(s0, ip + ivec2(1, 0), 0);
	vec4 a2 = texture(s1, uv);
	vec4 a3 = texture(s1, uv + vec2(0.25));
	vec4 a4 = texture(s2, uv);
	o = a0 + a1 + a2 + a3 + a4;
}
`
		mr, er, ok_r := compile(src_r, {profile = .Glsl, stage = .Fragment})
		defer destroy(mr)
		m4, e4, ok_4 := compile(src_rgba, {profile = .Glsl, stage = .Fragment})
		defer destroy(m4)
		testing.expectf(t, ok_r, "src_r compile failed:\n%s", format_errors(er, context.temp_allocator))
		testing.expectf(t, ok_4, "src_rgba compile failed:\n%s", format_errors(e4, context.temp_allocator))
		if !ok_r || !ok_4 {
			return
		}
		when JIT_ENABLED {
			if jit_avx2_ready {
				testing.expectf(
					t,
					len(mr.program.jit_mem) + 350 < len(m4.program.jit_mem),
					"expected dead-component trimming to save >350 B: .r=%d B vs .rgba=%d B",
					len(mr.program.jit_mem),
					len(m4.program.jit_mem),
				)
			}
		}
	}

	// 2. Adversarial component masks (.r, .g, .rg, .b, .a, .rgb, .rgba, unused 0x0)
	//    across 1-, 2-, and 4-component Nearest and Linear bound images + unbound
	//    counting callback (including OOB texelFetch lanes).
	{
		src := `#version 310 es
precision highp float;
uniform sampler2D s0;
uniform sampler2D s1;
uniform sampler2D s2;
uniform sampler2D s3;
uniform sampler2D s4;
uniform sampler2D s5;
uniform sampler2D s6;
uniform sampler2D s7;
out vec4 o;
void main() {
	int ix = int(gl_FragCoord.x) - 1; // lane 0 is -1 (OOB for texelFetch)
	int iy = int(gl_FragCoord.y);
	ivec2 ip = ivec2(ix, iy);
	vec2 uv = (gl_FragCoord.xy + vec2(0.37, 0.61)) * 0.25;

	float c_r  = texelFetch(s0, ip, 0).r;
	float c_g  = texelFetch(s1, ip, 0).g;
	vec2  c_rg = texture(s2, uv).rg;
	float c_b  = texture(s3, uv).b;
	float c_a  = texture(s4, uv).a;
	vec3  c_3  = texelFetch(s5, ip, 0).rgb;
	vec4  c_4  = texture(s6, uv);
	vec4  dead0 = texture(s7, uv);
	vec4  dead1 = texelFetch(s7, ip, 0);

	o = vec4(
		c_r + c_rg.x + c_3.x + c_4.x,
		c_g + c_rg.y + c_3.y + c_4.y,
		c_b + c_3.z + c_4.z,
		c_a + c_4.w
	);
}
`
		m, errs, ok := compile(src, {profile = .Glsl, stage = .Fragment})
		defer destroy(m)
		testing.expectf(t, ok, "compile failed:\n%s", format_errors(errs, context.temp_allocator))
		if !ok {
			return
		}

		count_cb :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32, layer: int) -> [4]f32 {
			cnt := (^int)(user)
			cnt^ += 1
			b := f32(channel + 1) * 0.5
			return {b + uv.x, b + uv.y, b * 0.25 + lod, 0.875}
		}

		tex1: [16 * 1]f32
		tex2: [16 * 2]f32
		tex4: [16 * 4]f32
		for i in 0 ..< 16 {
			tex1[i] = f32(i + 1) * 0.25
			tex2[i * 2 + 0] = f32(i + 1) * 0.5
			tex2[i * 2 + 1] = f32(16 - i) * 0.125
			tex4[i * 4 + 0] = f32(i) * 0.2 + 0.1
			tex4[i * 4 + 1] = f32(i % 5) * 0.3 + 0.2
			tex4[i * 4 + 2] = f32(i & 3) * 0.4 + 0.3
			tex4[i * 4 + 3] = f32(i % 3) * 0.25 + 0.25
		}
		sizes: [8][3]i32
		for &sz in sizes {
			sz = {4, 4, 1}
		}

		for comps in ([3]int{1, 2, 4}) {
			data := tex1[:] if comps == 1 else (tex2[:] if comps == 2 else tex4[:])
			for filt in ([2]Channel_Filter{.Nearest, .Linear}) {
				for mode in 0 ..< 3 {
					imgs: [8]Channel_Image
					for ch in 0 ..< 8 {
						bind := mode == 0 || (mode == 1 && (ch & 1) == 0) || (mode == 2 && ch != 7)
						if bind {
							imgs[ch] = {data = data, width = 4, height = 4, components = comps, filter = filt}
						}
					}
					cnt_j, cnt_i := 0, 0
					oj, oi: [24][4]f32
					opts_j := Run_Options {
						sampler        = count_cb,
						user           = &cnt_j,
						channel_sizes  = sizes[:],
						channel_images = imgs[:],
					}
					opts_i := opts_j
					opts_i.user = &cnt_i
					run_f32(&m.program, nil, 6, 4, oj[:], temp_scratch(), opts_j)
					run_f32_interp(&m.program, nil, 6, 4, oi[:], temp_scratch(), opts_i)
					testing.expectf(
						t,
						cnt_j == cnt_i,
						"comps=%d filt=%v mode=%d: callback count jit=%d interp=%d",
						comps, filt, mode, cnt_j, cnt_i,
					)
					for px in 0 ..< 24 {
						for c in 0 ..< 4 {
							diff := abs(oj[px][c] - oi[px][c])
							testing.expectf(
								t,
								diff <= 1e-5,
								"comps=%d filt=%v mode=%d px=%d c=%d: jit=%v interp=%v",
								comps, filt, mode, px, c, oj[px], oi[px],
							)
						}
					}
				}
			}
		}
	}
}

@(test)
test_jit_load_buf_bounds_and_cold :: proc(t: ^testing.T) {
	// 1. Relative two-program comparison: 8 vec4 buffer loads (Load_Buf n=4, stride=1,
	//    loading 32 words) share one uniform bounds check and one cold OOB/big-buf stub per
	//    op, emitting smaller code than 20 scalar Load_Buf(n=1, stride=1) loads (20 words).
	{
		fills_sz := [1]Uniform_Fill{{slot = 0, offset = 0, base = .Uint}}
		outputs_sz := [1]Program_Var{{kind = .Stage_Out, location = 0, slot = 4, n = 4, base = .Float}}
		code_v4 := make([dynamic]Instr, 0, 20, context.temp_allocator)
		code_f1 := make([dynamic]Instr, 0, 44, context.temp_allocator)
		for i in i32(0) ..< 8 {
			dst4 := i32(4) if i == 0 else i32(8)
			append(&code_v4, Instr{op = .Load_Buf, n = 4, k = 0, dst = dst4, a = 0, b = 1, c = i * 4})
			if i > 0 {
				append(&code_v4, Instr{op = .F_Add, n = 4, dst = 4, a = 4, b = 8})
			}
		}
		for i in i32(0) ..< 20 {
			dst1 := i32(4) if i == 0 else i32(8)
			append(&code_f1, Instr{op = .Load_Buf, n = 1, k = 0, dst = dst1, a = 0, b = 1, c = i})
			if i > 0 {
				append(&code_f1, Instr{op = .F_Add, n = 1, dst = 4, a = 4, b = 8})
			}
		}
		append(&code_f1, Instr{op = .Splat, n = 4, dst = 4, a = 4})

		p_v4 := Program {
			code        = code_v4[:],
			fills       = fills_sz[:],
			outputs     = outputs_sz[:],
			num_slots   = 16,
			num_masks   = 2,
			num_buffers = 1,
			stage       = .Fragment,
		}
		jit_compile(&p_v4)
		defer jit_destroy(&p_v4)

		p_f1 := Program {
			code        = code_f1[:],
			fills       = fills_sz[:],
			outputs     = outputs_sz[:],
			num_slots   = 16,
			num_masks   = 2,
			num_buffers = 1,
			stage       = .Fragment,
		}
		jit_compile(&p_f1)
		defer jit_destroy(&p_f1)

		when JIT_ENABLED {
			if jit_avx2_ready {
				testing.expectf(
					t,
					len(p_v4.jit_mem) < len(p_f1.jit_mem),
					"expected shared bounds check & cold stubs on Load_Buf(n=4): 8x_v4=%d B vs 20x_f1=%d B",
					len(p_v4.jit_mem),
					len(p_f1.jit_mem),
				)
			}
		}
	}

	// 2. Adversarial uniform & divergent bounds (in-bounds, straddling buf_len, index == len,
	//    all-OOB, and u32(max) 32-bit wraparound) for n = 1, 2, 4 (and 3), passing every buffer
	//    as a sub-slice of a larger backing array whose words past the end hold non-zero sentinels
	//    (catches off-by-one `lea [rax + (n - 2)]` and `ja` vs `jae` overreads!).
	src := `#version 310 es
precision highp float;
layout(std430, binding = 0) readonly buffer B0 { float f[]; } b0;
layout(std430, binding = 1) readonly buffer B1 { vec2  v2[]; } b1;
layout(std430, binding = 2) readonly buffer B2 { vec4  v4[]; } b2;
uniform uint uMode;
uniform uint uIdx;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
	uint lane = uint(int(gl_FragCoord.y) * 4 + int(gl_FragCoord.x));
	uint w = uIdx;
	if (uMode == 1u) {
		if      (lane == 0u) w = 0u;
		else if (lane == 1u) w = uIdx;
		else if (lane == 2u) w = uIdx + 1u;
		else if (lane == 3u) w = uIdx + 2u;
		else if (lane == 4u) w = uIdx + 3u;
		else if (lane == 5u) w = 0x7fffffffu;
		else if (lane == 6u) w = 0xfffffffeu;
		else                 w = 0xffffffffu;
	}
	o0 = b2.v4[w];
	o1 = vec4(b1.v2[w], b0.f[w], b0.f[w + 3u]);
}
`
	// Direct stride=1, off=0 IR program emitting separate outputs for n = 4 (t0.xyzw),
	// n = 2 (t1.xy), n = 1 (t1.z), and n = 3 (t1.w = comp 2), driven in both uniform
	// (uMode == 0 -> all 8 lanes use uIdx) and divergent (uMode == 1 -> uIdx + lane_x) modes.
	fills := [2]Uniform_Fill {
		{slot = 0, offset = 0, base = .Uint}, // uIdx
		{slot = 1, offset = 4, base = .Uint}, // uMode (0 = uniform, 1 = divergent)
	}
	inputs := [1]Program_Var{{kind = .Frag_Coord, slot = 2, n = 2, base = .Float}}
	outputs := [2]Program_Var {
		{kind = .Stage_Out, location = 0, slot = 16, n = 4, base = .Float},
		{kind = .Stage_Out, location = 1, slot = 20, n = 4, base = .Float},
	}
	consts := [1][2]u32{{4, 0}}
	code_ir := [?]Instr {
		{op = .F_To_U, n = 1, dst = 5, a = 2},                        // lane_x 0..7
		{op = .I_Mul, n = 1, dst = 5, a = 5, b = 1},                  // 0 when uMode==0, lane_x when uMode==1
		{op = .I_Add, n = 1, dst = 6, a = 0, b = 5},                  // idx = uIdx (+ lane_x)
		{op = .Load_Buf, n = 4, k = 0, dst = 16, a = 6, b = 1, c = 0}, // n = 4 -> t0.xyzw
		{op = .Load_Buf, n = 2, k = 0, dst = 20, a = 6, b = 1, c = 0}, // n = 2 -> t1.xy
		{op = .Load_Buf, n = 1, k = 0, dst = 22, a = 6, b = 1, c = 0}, // n = 1 -> t1.z
		{op = .Load_Buf, n = 3, k = 0, dst = 8,  a = 6, b = 1, c = 0}, // n = 3 -> t1.w = slot 10
		{op = .Mov, n = 1, dst = 23, a = 10},
	}
	p_ir := Program {
		code        = code_ir[:],
		fills       = fills[:],
		inputs      = inputs[:],
		outputs     = outputs[:],
		consts      = consts[:],
		num_slots   = 28,
		num_masks   = 2,
		num_buffers = 1,
		stage       = .Fragment,
	}
	jit_compile(&p_ir)
	defer jit_destroy(&p_ir)
	p_ir_int := p_ir
	p_ir_int.jit_body, p_ir_int.jit_init = nil, nil

	m2, errs2, ok2 := compile(src, {profile = .Glsl, stage = .Fragment})
	defer destroy(m2)
	testing.expectf(t, ok2, "src compile failed:\n%s", format_errors(errs2, context.temp_allocator))
	if !ok2 {
		return
	}
	p_m2_int := m2.program
	p_m2_int.jit_body, p_m2_int.jit_init = nil, nil

	// Backing array with 32 words: sub-slices [:10] and [:12] leave non-zero float sentinels
	// immediately past `len(buf)` so any 1-word overread returns 930.0+ instead of 0.0!
	backing_words: [32]u32
	for i in 0 ..< 12 {
		backing_words[i] = transmute(u32)(f32(i + 1) * 1.25)
	}
	for i in 12 ..< len(backing_words) {
		backing_words[i] = transmute(u32)f32(800 + i * 13)
	}
	// Overwrite indices 10..11 with sentinels when testing len == 10, then restore for len == 12.
	test_indices := [?]u32 {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100,
		0x7FFF_FFFE, 0x8000_0000, 0xFFFF_FFFC, 0xFFFF_FFFD, 0xFFFF_FFFE, 0xFFFF_FFFF,
	}
	for sub_len in ([2]int{10, 12}) {
		for i in 0 ..< len(backing_words) {
			if i < sub_len {
				backing_words[i] = transmute(u32)(f32(i + 1) * 1.25)
			} else {
				backing_words[i] = transmute(u32)f32(900 + i * 17)
			}
		}
		sub := backing_words[:sub_len]
		for mode in u32(0) ..= 1 {
			for u_idx in test_indices {
				unif_ir: [8]u8
				(^u32)(&unif_ir[0])^ = u_idx
				(^u32)(&unif_ir[4])^ = mode
				bufs1 := [1][]u32{sub}
				tj0, tj1, ti0, ti1: [16][4]f32
				tjs := [2][][4]f32{tj0[:], tj1[:]}
				tis := [2][][4]f32{ti0[:], ti1[:]}
				run_targets(&p_ir, unif_ir[:], 8, 2, tjs[:], temp_scratch(), {buffers = bufs1[:]})
				run_targets(&p_ir_int, unif_ir[:], 8, 2, tis[:], temp_scratch(), {buffers = bufs1[:]})
				for px in 0 ..< 16 {
					testing.expectf(
						t,
						jit_vec4_bits_eq(tj0[px], ti0[px]),
						"IR len=%d mode=%d uIdx=0x%x t0 px=%d: jit=%v interp=%v",
						sub_len, mode, u_idx, px, tj0[px], ti0[px],
					)
					testing.expectf(
						t,
						jit_vec4_bits_eq(tj1[px], ti1[px]),
						"IR len=%d mode=%d uIdx=0x%x t1 px=%d: jit=%v interp=%v",
						sub_len, mode, u_idx, px, tj1[px], ti1[px],
					)
				}

				unif: [16]u8
				for u in m2.uniforms {
					if u.name == "uMode" {
						(^u32)(&unif[u.offset])^ = mode
					} else if u.name == "uIdx" {
						(^u32)(&unif[u.offset])^ = u_idx
					}
				}
				bufs3 := [3][]u32{sub, sub, sub}
				gj0, gj1, gi0, gi1: [8][4]f32
				gjs := [2][][4]f32{gj0[:], gj1[:]}
				gis := [2][][4]f32{gi0[:], gi1[:]}
				run_targets(&m2.program, unif[:], 4, 2, gjs[:], temp_scratch(), {buffers = bufs3[:]})
				run_targets(&p_m2_int, unif[:], 4, 2, gis[:], temp_scratch(), {buffers = bufs3[:]})
				for px in 0 ..< 8 {
					testing.expectf(
						t,
						jit_vec4_bits_eq(gj0[px], gi0[px]),
						"GLSL len=%d mode=%d uIdx=0x%x t0 px=%d: jit=%v interp=%v",
						sub_len, mode, u_idx, px, gj0[px], gi0[px],
					)
					testing.expectf(
						t,
						jit_vec4_bits_eq(gj1[px], gi1[px]),
						"GLSL len=%d mode=%d uIdx=0x%x t1 px=%d: jit=%v interp=%v",
						sub_len, mode, u_idx, px, gj1[px], gi1[px],
					)
				}
			}
		}
	}

	// Sparse > 2^31-word (> 8 GiB) buffer: commit page 0 and the page at 2^31 words,
	// with non-zero sentinels placed immediately past BIG_WORDS on the committed high page.
	BIG_WORDS :: (int(1) << 31) + 6
	BIG_BYTES :: uint(BIG_WORDS + 16) * size_of(u32)
	if reserved, rerr := virtual.reserve(BIG_BYTES); rerr == .None && len(reserved) >= int(BIG_BYTES) {
		defer virtual.release(raw_data(reserved), len(reserved))
		hi_byte_off := uint(1) << 33 // (1 << 31) * 4 bytes (page-aligned)
		c0 := virtual.commit(raw_data(reserved), 4096)
		c1 := virtual.commit(([^]byte)(raw_data(reserved))[hi_byte_off:], 4096)
		if c0 == .None && c1 == .None {
			big_backing := mem.slice_ptr((^u32)(raw_data(reserved)), BIG_WORDS + 10)
			for i in 0 ..< 8 {
				big_backing[i] = transmute(u32)(f32(i + 1) * 2.5)
			}
			for i in 0 ..< 6 {
				big_backing[(int(1) << 31) + i] = transmute(u32)(f32(i + 10) * 0.5)
			}
			for i in 6 ..< 16 {
				big_backing[(int(1) << 31) + i] = transmute(u32)f32(700 + i * 19)
			}
			big_slice := big_backing[:BIG_WORDS]
			big_indices := [?]u32 {
				0,
				2,
				0x8000_0000,
				0x8000_0002, // all 4 in bounds above 2^31
				0x8000_0003, // 3 in bounds + 1 OOB above 2^31 (straddling sentinel!)
				0x8000_0004, // 2 in bounds + 2 OOB above 2^31
				0x8000_0005, // 1 in bounds + 3 OOB above 2^31 (n=2 straddling sentinel!)
				0x8000_0006, // index == len above 2^31 (n=1 sentinel!)
				0xFFFF_FFFE, // wraps 0xFFFF_FFFE, 0xFFFF_FFFF (OOB), 0, 1 (in bounds at page 0!)
			}
			bufs_big := [1][]u32{big_slice}
			for mode in u32(0) ..= 1 {
				for u_idx in big_indices {
					unif_ir: [8]u8
					(^u32)(&unif_ir[0])^ = u_idx
					(^u32)(&unif_ir[4])^ = mode
					tj0, tj1, ti0, ti1: [16][4]f32
					tjs := [2][][4]f32{tj0[:], tj1[:]}
					tis := [2][][4]f32{ti0[:], ti1[:]}
					run_targets(&p_ir, unif_ir[:], 8, 2, tjs[:], temp_scratch(), {buffers = bufs_big[:]})
					run_targets(&p_ir_int, unif_ir[:], 8, 2, tis[:], temp_scratch(), {buffers = bufs_big[:]})
					for px in 0 ..< 16 {
						testing.expectf(t, jit_vec4_bits_eq(tj0[px], ti0[px]), "IR >2^31 mode=%d uIdx=0x%x t0 px=%d: jit=%v interp=%v", mode, u_idx, px, tj0[px], ti0[px])
						testing.expectf(t, jit_vec4_bits_eq(tj1[px], ti1[px]), "IR >2^31 mode=%d uIdx=0x%x t1 px=%d: jit=%v interp=%v", mode, u_idx, px, tj1[px], ti1[px])
					}
				}
			}
		}
	}
}


// ---- DXBC ---------------------------------------------------------------------------

// MD5 with RFC 1321's padding over md5_block, to check the compression
// function the DXBC checksum shares.
@(private = "file")
md5_rfc :: proc(msg: []u8) -> (digest: [16]u8) {
	state := [4]u32{0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476}
	padded := make([dynamic]u8, context.temp_allocator)
	append(&padded, ..msg)
	append(&padded, 0x80)
	for len(padded) % 64 != 56 {
		append(&padded, 0)
	}
	bits := u64(len(msg)) * 8
	for i in 0 ..< 8 {
		append(&padded, u8(bits >> uint(8 * i)))
	}
	for b := 0; b < len(padded); b += 64 {
		w: [16]u32
		for i in 0 ..< 16 {
			w[i] = u32(padded[b + 4 * i]) | u32(padded[b + 4 * i + 1]) << 8 | u32(padded[b + 4 * i + 2]) << 16 | u32(padded[b + 4 * i + 3]) << 24
		}
		md5_block(&state, w)
	}
	for v, i in state {
		for k in 0 ..< 4 {
			digest[4 * i + k] = u8(v >> uint(8 * k))
		}
	}
	return
}

// The container checksum: MD5's compression on RFC 1321's vectors, and
// containers of one chunk (both final-block shapes: fewer than 56 bytes
// left, and 56 or more) against the checksums vkd3d-dxbc --emit gave the
// same bytes.
@(test)
test_dxbc_checksum :: proc(t: ^testing.T) {
	vectors := [?][2]string {
		{"", "d41d8cd98f00b204e9800998ecf8427e"},
		{"abc", "900150983cd24fb0d6963f7d28e17f72"},
		{"message digest", "f96b697d7cb7938d525a2f31aaf161d0"},
		{"12345678901234567890123456789012345678901234567890123456789012345678901234567890", "57edf4a22be3c955ac49da2e2107b67a"},
	}
	for v in vectors {
		d := md5_rfc(transmute([]u8)v[0])
		testing.expectf(t, d == hex_digest(v[1]), "md5(%q) = %02x", v[0], d)
	}
	Want :: struct {
		len: int,
		sum: [4]u32,
	}
	wants := [?]Want {
		{0, {0x38f21f24, 0xeeb6071a, 0xe6974c5d, 0xc470fe8f}},
		{32, {0x651ed640, 0xcfbd4905, 0x0f03016f, 0xaceff30d}},
		{36, {0x65c6dcb6, 0x2b83c255, 0xe74da885, 0x916da9a6}},
		{40, {0x20795276, 0x2122faaa, 0xcc4e2eab, 0x530ce50b}},
		{100, {0xe11af0a1, 0xc1cbb5eb, 0xc81c8667, 0xf205a790}},
		{1000, {0x33f58a16, 0x958dadde, 0xed2d25f7, 0x67cd03cc}},
	}
	for w in wants {
		data := make([]u8, w.len, context.temp_allocator)
		for &b, i in data {
			b = u8(i * 7 + 3)
		}
		parts := [2][]u8{data[:w.len / 2], data[w.len / 2:]}
		chunks := [1]Dx_Chunk{{{'T', 'E', 'S', 'T'}, parts[:]}}
		blob := dxbc_container(chunks[:], context.temp_allocator)
		testing.expect_value(t, len(blob), 44 + w.len)
		sum := w.sum
		got := slice.reinterpret([]u32, blob[4:20])
		testing.expectf(t, slice.equal(got, sum[:]), "%d bytes: %x, want %x", w.len, got, sum)
	}
}

@(private = "file")
hex_digest :: proc(s: string) -> (out: [16]u8) {
	for i in 0 ..< 16 {
		hi, lo := s[2 * i], s[2 * i + 1]
		nib :: proc(c: u8) -> u8 {return c - '0' if c <= '9' else c - 'a' + 10}
		out[i] = nib(hi) << 4 | nib(lo)
	}
	return
}

// A DXBC container's chunks by tag.
@(private = "file")
dxbc_chunk :: proc(blob: []u8, tag: string) -> []u8 {
	u32_at :: proc(b: []u8, at: int) -> int {return int(b[at]) | int(b[at + 1]) << 8 | int(b[at + 2]) << 16 | int(b[at + 3]) << 24}
	for i in 0 ..< u32_at(blob, 28) {
		at := u32_at(blob, 32 + 4 * i)
		if string(blob[at:][:4]) == tag {
			return blob[at + 8:][:u32_at(blob, at + 4)]
		}
	}
	return nil
}


// A signature element as the test reads it back: the second mask byte is
// an input's components read, an output's components never written.
@(private = "file")
Sig_Row :: struct {
	name:                           string,
	index, sysval, ctype, register: u32,
	mask, rw:                       u8,
}

@(private = "file")
dxbc_sig :: proc(chunk: []u8) -> []Sig_Row {
	w := slice.reinterpret([]u32, chunk)
	rows := make([]Sig_Row, w[0], context.temp_allocator)
	for &r, i in rows {
		e := w[2 + 6 * i:]
		name := string(cstring(&chunk[e[0]]))
		r = {name, e[1], e[2], e[3], e[4], u8(e[5]), u8(e[5] >> 8)}
	}
	return rows
}

// A decoded operand: file, indices, a relative register (or -1).
@(private = "file")
Operand :: struct {
	file: u32,
	idx:  [3]u32,
	dims: int,
	rel:  i64,
	// An immediate's first word.
	imm:  u32,
	// The token's component count (0: none, 1: one, 2: four), selection
	// mode (0: mask, 1: swizzle, 2: one component) and its mask, swizzle
	// or component.
	ncomp, mode, sel: u32,
}

// One instruction: opcode, controls and operands (from the operand
// tokens to its end, for the ones the emitter writes without trailing
// words).
@(private = "file")
Ins :: struct {
	op, ctl: u32,
	words:   []u32,
}

// Decodes a module's DXBC program and checks everything a blob must hold
// without a driver: its operands against the declarations, its structure
// (dxbc_check_structure) and its signatures against the module's stage
// variables (dxbc_check_signatures).
@(private = "file")
dxbc_program :: proc(t: ^testing.T, m: ^Module, name: string, loc := #caller_location) -> (version: u32, ins: []Ins) {
	blob := m.dxbc
	if !testing.expectf(t, len(blob) > 32 && string(blob[:4]) == "DXBC", "%s: no DXBC", name, loc = loc) {
		return
	}
	sum := dxbc_checksum(blob[20:])
	testing.expectf(t, slice.equal(slice.reinterpret([]u32, blob[4:20]), sum[:]), "%s: checksum", name, loc = loc)
	testing.expectf(t, int(slice.reinterpret([]u32, blob[20:32])[1]) == len(blob), "%s: container size", name, loc = loc)
	shex := slice.reinterpret([]u32, dxbc_chunk(blob, "SHEX"))
	testing.expect_value(t, int(shex[1]), len(shex), loc = loc)
	out := make([dynamic]Ins, context.temp_allocator)
	for i := 2; i < len(shex); {
		n := int(shex[i] >> 24 & 0x7f)
		if !testing.expect(t, n > 0 && i + n <= len(shex), loc = loc) {
			break
		}
		append(&out, Ins{shex[i] & 0x7ff, shex[i] >> 11 & 0x1fff, shex[i + 1:i + n]})
		i += n
	}
	dxbc_check_operands(t, name, out[:], loc)
	dxbc_check_structure(t, name, out[:], loc)
	dxbc_check_signatures(t, m, name, out[:], loc)
	return shex[0], out[:]
}

// Decodes the operands of an instruction's words.
@(private = "file")
dxbc_operands :: proc(words: []u32) -> (ops: []Operand, ok: bool) {
	out := make([dynamic]Operand, context.temp_allocator)
	i := 0
	for i < len(words) {
		tok := words[i]
		i += 1
		if tok >> 31 != 0 {
			i += 1
		}
		o := Operand{file = tok >> 12 & 0xff, dims = int(tok >> 20 & 3), rel = -1, ncomp = tok & 3, mode = tok >> 2 & 3, sel = tok >> 4 & 0xff}
		for k in 0 ..< o.dims {
			rep := tok >> (22 + 3 * uint(k)) & 7
			if rep == 0 || rep == 3 {
				o.idx[k] = words[i]
				i += 1
			}
			if rep == 2 || rep == 3 {
				// The relative register: a temp's component.
				i += 1
				o.rel = i64(words[i])
				i += 1
			}
		}
		if o.file == 4 {
			o.imm = words[i] if i < len(words) else 0
			i += 1 if tok & 3 == 1 else 4
		}
		append(&out, o)
	}
	return out[:], i == len(words)
}

// Checks every operand of the program's instructions against the
// declarations: temps below dcl_temps, x# arrays declared and constant
// indices inside them, inputs and outputs declared, constant buffer
// elements inside the buffer, resource and sampler ids declared. Every
// destination (the first operand, the first two of imul, udiv and sincos)
// writes through a mask that is not empty (a null destination writes
// nothing) and names only components its register has (an output's
// declared ones).
@(private = "file")
dxbc_check_operands :: proc(t: ^testing.T, name: string, ins: []Ins, loc := #caller_location) {
	temps: u32
	x_sizes := make(map[u32]u32, context.temp_allocator)
	inputs := make(map[u32]bool, context.temp_allocator)
	outputs := make(map[u32]bool, context.temp_allocator)
	output_masks := make(map[u32]u32, context.temp_allocator)
	ids := make(map[[2]u32]bool, context.temp_allocator)
	cb_size: u32
	for in_ in ins {
		switch Dx_Op(in_.op) {
		case .Dcl_Temps:
			temps = in_.words[0]
		case .Dcl_Indexable_Temp:
			x_sizes[in_.words[0]] = in_.words[1]
		case .Dcl_Input, .Dcl_Input_Ps, .Dcl_Input_Sgv, .Dcl_Input_Ps_Sgv, .Dcl_Input_Ps_Siv:
			inputs[in_.words[1]] = true
		case .Dcl_Output, .Dcl_Output_Siv:
			if in_.words[0] >> 12 & 0xff == 2 {
				outputs[in_.words[1]] = true
				output_masks[in_.words[1]] = in_.words[0] >> 4 & 0xf
			}
		case .Dcl_Constant_Buffer:
			cb_size = in_.words[4]
		case .Dcl_Global_Flags:
		case .Dcl_Resource, .Dcl_Resource_Raw:
			ids[{7, in_.words[1]}] = true
		case .Dcl_Sampler:
			ids[{6, in_.words[1]}] = true
		case .Add, .And, .Break, .Breakc, .Case, .Continue, .Default, .Discard, .Div, .Dp2, .Dp3, .Dp4, .Else, .End_If,
		     .End_Loop, .End_Switch, .Eq, .Exp, .Frc, .Ftoi, .Ftou, .Ge, .Iadd, .If, .Ieq, .Ige, .Ilt, .Imax, .Imin, .Imul,
		     .Ine, .Ineg, .Ishl, .Ishr, .Itof, .Log, .Loop, .Lt, .Mad, .Min, .Max, .Mov, .Movc, .Mul, .Ne, .Not, .Or,
		     .Resinfo, .Ret, .Round_Ne, .Round_Ni, .Round_Pi, .Round_Z, .Rsq, .Sample, .Sample_L, .Sample_D, .Sample_B,
		     .Sqrt, .Switch, .Sincos, .Udiv, .Ult, .Uge, .Umax, .Umin, .Ushr, .Utof, .Xor, .Deriv_Rtx_Fine, .Deriv_Rty_Fine,
		     .Rcp, .Ld_Raw:
			ops, ok := dxbc_operands(in_.words)
			testing.expectf(t, ok, "%s: op %x: operands overrun", name, in_.op, loc = loc)
			dsts := 0
			#partial switch Dx_Op(in_.op) {
			case .Break, .Breakc, .Case, .Continue, .Default, .Discard, .Else, .End_If, .End_Loop, .End_Switch, .If, .Loop, .Ret, .Switch:
			case .Imul, .Udiv, .Sincos:
				dsts = 2
			case:
				dsts = 1
			}
			for o, k in ops[:min(dsts, len(ops))] {
				switch o.file {
				case 13, 15:
				// null, oMask
				case:
					mask := o.sel & 0xf
					testing.expectf(t, o.ncomp == 2 && o.mode == 0 && mask != 0, "%s: op %v destination %d writes mask %x", name, Dx_Op(in_.op), k, mask, loc = loc)
					if o.file == 2 {
						testing.expectf(t, mask & ~output_masks[o.idx[0]] == 0, "%s: op %v writes o%d.%x of %x", name, Dx_Op(in_.op), o.idx[0], mask, output_masks[o.idx[0]], loc = loc)
					}
					testing.expectf(t, o.file == 0 || o.file == 2 || o.file == 3, "%s: op %v writes file %d", name, Dx_Op(in_.op), o.file, loc = loc)
				}
			}
			for o in ops {
				switch o.file {
				case 0:
					testing.expectf(t, o.idx[0] < temps, "%s: r%d of %d", name, o.idx[0], temps, loc = loc)
				case 1:
					testing.expectf(t, inputs[o.idx[0]], "%s: v%d undeclared", name, o.idx[0], loc = loc)
				case 2:
					testing.expectf(t, outputs[o.idx[0]], "%s: o%d undeclared", name, o.idx[0], loc = loc)
				case 3:
					size, declared := x_sizes[o.idx[0]]
					testing.expectf(t, declared && (o.rel >= 0 || o.idx[1] < size), "%s: x%d[%d] of %d", name, o.idx[0], o.idx[1], size, loc = loc)
				case 6, 7:
					testing.expectf(t, ids[{o.file, o.idx[0]}], "%s: %v id %d undeclared", name, o.file, o.idx[0], loc = loc)
				case 8:
					testing.expectf(t, o.rel >= 0 || o.idx[2] < cb_size, "%s: cb element %d of %d", name, o.idx[2], cb_size, loc = loc)
				}
				if o.rel >= 0 {
					testing.expectf(t, u32(o.rel) < temps, "%s: relative r%d of %d", name, o.rel, temps, loc = loc)
				}
			}
		case .Ld_Raw + Dx_Op(1) ..= max(Dx_Op):
		}
	}
}

// The program's structure, by Shader Model 4/5's rules and what drivers
// assume: if / else / endif, loop / endloop and switch / endswitch nest;
// a breakc's innermost loop or switch is a loop (in a switch it would
// leave the switch, not the loop); break and continue sit in a construct
// they can leave; every case or default whose code is not empty ends in
// break or ret, and nothing precedes a switch's first label; every
// discard is followed at once by ret (discard ends the invocation); the
// program ends with ret.
@(private = "file")
dxbc_check_structure :: proc(t: ^testing.T, name: string, ins: []Ins, loc := #caller_location) {
	Frame :: struct {
		op:        Dx_Op,
		// A switch: a label seen, and code since the last label.
		labelled:  bool,
		has_code:  bool,
	}
	stack := make([dynamic]Frame, context.temp_allocator)
	breakable :: proc(stack: []Frame) -> (Dx_Op, bool) {
		#reverse for f in stack {
			if f.op == .Loop || f.op == .Switch {
				return f.op, true
			}
		}
		return .Mov, false
	}
	for in_, i in ins {
		op := Dx_Op(in_.op)
		if op >= .Dcl_Resource && op <= .Dcl_Global_Flags || op == .Dcl_Resource_Raw {
			continue
		}
		if len(stack) > 0 && stack[len(stack) - 1].op == .Switch && op != .Case && op != .Default && op != .End_Switch {
			top := &stack[len(stack) - 1]
			testing.expectf(t, top.labelled, "%s: %v before a switch's first label (instruction %d)", name, op, i, loc = loc)
			top.has_code = true
		}
		#partial switch op {
		case .If, .Loop, .Switch:
			append(&stack, Frame{op = op})
		case .Else, .End_If:
			ok := len(stack) > 0 && stack[len(stack) - 1].op == .If
			testing.expectf(t, ok, "%s: %v outside an if (instruction %d)", name, op, i, loc = loc)
			if ok && op == .End_If {
				pop(&stack)
			}
		case .End_Loop:
			ok := len(stack) > 0 && stack[len(stack) - 1].op == .Loop
			testing.expectf(t, ok, "%s: endloop outside a loop (instruction %d)", name, i, loc = loc)
			if ok {
				pop(&stack)
			}
		case .Case, .Default, .End_Switch:
			ok := len(stack) > 0 && stack[len(stack) - 1].op == .Switch
			testing.expectf(t, ok, "%s: %v outside a switch (instruction %d)", name, op, i, loc = loc)
			if !ok {
				continue
			}
			top := &stack[len(stack) - 1]
			if top.has_code {
				prev := Dx_Op(ins[i - 1].op)
				testing.expectf(t, prev == .Break || prev == .Ret, "%s: a case ends in %v, not break or ret (instruction %d)", name, prev, i, loc = loc)
			}
			top.labelled = true
			top.has_code = false
			if op == .End_Switch {
				pop(&stack)
			}
		case .Breakc:
			inner, ok := breakable(stack[:])
			testing.expectf(t, ok && inner == .Loop, "%s: breakc whose innermost construct is %v (instruction %d)", name, inner if ok else Dx_Op.Ret, i, loc = loc)
		case .Break:
			_, ok := breakable(stack[:])
			testing.expectf(t, ok, "%s: break outside a loop or switch (instruction %d)", name, i, loc = loc)
		case .Continue:
			in_loop := false
			for f in stack {
				in_loop ||= f.op == .Loop
			}
			testing.expectf(t, in_loop, "%s: continue outside a loop (instruction %d)", name, i, loc = loc)
		case .Discard:
			next_ret := i + 1 < len(ins) && Dx_Op(ins[i + 1].op) == .Ret
			testing.expectf(t, next_ret, "%s: discard not followed by ret (instruction %d)", name, i, loc = loc)
		}
	}
	testing.expectf(t, len(stack) == 0, "%s: %d constructs left open", name, len(stack), loc = loc)
	testing.expectf(t, len(ins) > 0 && Dx_Op(ins[len(ins) - 1].op) == .Ret, "%s: the program does not end with ret", name, loc = loc)
}

// The signatures against the module's stage variables and system values,
// derived here independently of the emitter: TEXCOORD<location> (vertex
// inputs and outputs, fragment inputs) and SV_Target<location> in register
// <location>, the component type of the variable's base (float 3, int 2,
// uint 1) and its components; an input's read mask its components, an
// output's never-written mask the others. Fragment inputs are
// interpolated linearly unless flat or integer (constant).
@(private = "file")
dxbc_check_signatures :: proc(t: ^testing.T, m: ^Module, name: string, prog: []Ins, loc := #caller_location) {
	ctype :: proc(b: Base) -> u32 {
		#partial switch b {
		case .Int:
			return 2
		case .Uint:
			return 1
		}
		return 3
	}
	fragment := m.stage == .Fragment
	ins := dxbc_sig(dxbc_chunk(m.dxbc, "ISGN"))
	outs := dxbc_sig(dxbc_chunk(m.dxbc, "OSGN"))
	varyings_in, varyings_out := 0, 0
	for r in ins {
		testing.expectf(t, r.rw == r.mask, "%s: input %s%d read mask %x of %x", name, r.name, r.index, r.rw, r.mask, loc = loc)
		switch r.name {
		case "TEXCOORD":
			varyings_in += 1
			found := false
			for v in m.stage_inputs {
				if u32(v.location) == r.index {
					found = true
					want := Sig_Row{"TEXCOORD", u32(v.location), 0, ctype(v.type.base), u32(v.location), u8(1 << uint(v.type.rows)) - 1, u8(1 << uint(v.type.rows)) - 1}
					testing.expectf(t, r == want, "%s: input %v, want %v", name, r, want, loc = loc)
					if fragment {
						interp: u32 = 1 if v.flat || v.type.base != .Float else 2
						for in_ in prog {
							if Dx_Op(in_.op) == .Dcl_Input_Ps && in_.words[1] == r.register {
								testing.expectf(t, in_.ctl & 0xf == interp, "%s: input %d interpolation %d, want %d", name, r.register, in_.ctl & 0xf, interp, loc = loc)
							}
						}
					}
				}
			}
			testing.expectf(t, found, "%s: input TEXCOORD%d is no stage input", name, r.index, loc = loc)
		case "SV_Position":
			testing.expectf(t, fragment && r == Sig_Row{"SV_Position", 0, 1, 3, 16, 0xf, 0xf}, "%s: %v", name, r, loc = loc)
		case "SV_IsFrontFace":
			testing.expectf(t, fragment && r == Sig_Row{"SV_IsFrontFace", 0, 9, 1, 17, 1, 1}, "%s: %v", name, r, loc = loc)
		case "SV_VertexID":
			testing.expectf(t, !fragment && r == Sig_Row{"SV_VertexID", 0, 6, 1, 16, 1, 1}, "%s: %v", name, r, loc = loc)
		case:
			testing.expectf(t, false, "%s: input %s", name, r.name, loc = loc)
		}
	}
	for r in outs {
		testing.expectf(t, r.rw == 0xf ~ r.mask, "%s: output %s%d never-written mask %x of %x", name, r.name, r.index, r.rw, r.mask, loc = loc)
		switch r.name {
		case "TEXCOORD", "SV_Target":
			testing.expectf(t, (r.name == "SV_Target") == fragment, "%s: output %s", name, r.name, loc = loc)
			varyings_out += 1
			found := false
			for v in m.stage_outputs {
				if u32(v.location) == r.index {
					found = true
					mask := u8(1 << uint(v.type.rows)) - 1
					want := Sig_Row{r.name, u32(v.location), 0, ctype(v.type.base), u32(v.location), mask, 0xf ~ mask}
					testing.expectf(t, r == want, "%s: output %v, want %v", name, r, want, loc = loc)
				}
			}
			testing.expectf(t, found, "%s: output %s%d is no stage output", name, r.name, r.index, loc = loc)
		case "SV_Position":
			testing.expectf(t, !fragment && r == Sig_Row{"SV_Position", 0, 1, 3, 16, 0xf, 0}, "%s: %v", name, r, loc = loc)
		case "SV_Coverage":
			testing.expectf(t, fragment && r == Sig_Row{"SV_Coverage", 0, 0, 1, 0xffffffff, 1, 0xe}, "%s: %v", name, r, loc = loc)
		case:
			testing.expectf(t, false, "%s: output %s", name, r.name, loc = loc)
		}
	}
	testing.expectf(t, varyings_in == len(m.stage_inputs) && varyings_out == len(m.stage_outputs), "%s: %d and %d stage rows for %d inputs and %d outputs", name, varyings_in, varyings_out, len(m.stage_inputs), len(m.stage_outputs), loc = loc)
}

// Containers, signatures and SDL_GPU's D3D12 registers: a fragment shader
// with flat and smooth inputs, three targets, gl_FragCoord,
// gl_FrontFacing, a sampler2D, a sampler2DArray, a storage buffer and
// uniforms; a vertex shader with attributes, gl_VertexID, a sampler and a
// storage buffer.
@(test)
test_dxbc_interface :: proc(t: ^testing.T) {
	fs := `#version 310 es
precision highp float;
in vec2 ramp;
layout(location = 0) flat in ivec2 id;
layout(location = 1) out vec4 second;
layout(location = 0) out vec4 first;
layout(location = 2) out uvec2 count;
uniform float uGain;
uniform sampler2D tex;
uniform sampler2DArray layers;
readonly buffer Words { uint words[]; };
void main() {
    vec4 c = texture(tex, ramp) + texture(layers, vec3(ramp, 1.0));
    first = c * uGain + gl_FragCoord;
    second = vec4(vec2(id), gl_FrontFacing ? 1.0 : 0.0, 1.0);
    count = uvec2(words[id.x], gl_FragCoord.x);
}`
	m, errs, ok := compile_glsl(fs, {dxbc = true})
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	blob := m.dxbc
	ins := dxbc_sig(dxbc_chunk(blob, "ISGN"))
	outs := dxbc_sig(dxbc_chunk(blob, "OSGN"))
	// Locations are registers; SV_Position after them.
	want_in := [?]Sig_Row{{"TEXCOORD", 0, 0, 2, 0, 3, 3}, {"TEXCOORD", 1, 0, 3, 1, 3, 3}, {"SV_Position", 0, 1, 3, 16, 15, 15}, {"SV_IsFrontFace", 0, 9, 1, 17, 1, 1}}
	want_out := [?]Sig_Row{{"SV_Target", 0, 0, 3, 0, 15, 0}, {"SV_Target", 1, 0, 3, 1, 15, 0}, {"SV_Target", 2, 0, 1, 2, 3, 12}}
	testing.expectf(t, slice.equal(ins, want_in[:]), "ISGN %v", ins)
	testing.expectf(t, slice.equal(outs, want_out[:]), "OSGN %v", outs)
	version, prog := dxbc_program(t, m, "fragment")
	testing.expect_value(t, version, 0x51) // ps_5_1
	// Pixel shader registers: t[n] and s[n] in space 2 (channel n, then
	// the storage buffer), b0 in space 3.
	spaces: [dynamic][3]u32
	spaces.allocator = context.temp_allocator
	rcp_w, flat_id := false, false
	for in_ in prog {
		#partial switch Dx_Op(in_.op) {
		case .Dcl_Constant_Buffer:
			append(&spaces, [3]u32{in_.op, in_.words[2], in_.words[5]})
		case .Dcl_Sampler, .Dcl_Resource_Raw:
			append(&spaces, [3]u32{in_.op, in_.words[2], in_.words[4]})
		case .Dcl_Resource:
			append(&spaces, [3]u32{in_.op | in_.ctl << 16, in_.words[2], in_.words[5]})
		case .Rcp:
			// gl_FragCoord.w is 1 / SV_Position.w.
			rcp_w = in_.words[2] >> 12 & 0xff == 1 && in_.words[3] == 16 && in_.words[2] >> 4 & 3 == 3
		case .Dcl_Input_Ps:
			if in_.words[1] == 0 {
				flat_id = in_.ctl == 1
			}
		}
	}
	sampler, resource, raw, cb := u32(Dx_Op.Dcl_Sampler), u32(Dx_Op.Dcl_Resource), u32(Dx_Op.Dcl_Resource_Raw), u32(Dx_Op.Dcl_Constant_Buffer)
	want := [?][3]u32{{cb, 0, 3}, {sampler, 0, 2}, {sampler, 1, 2}, {resource | 3 << 16, 0, 2}, {resource | 8 << 16, 1, 2}, {raw, 2, 2}}
	testing.expectf(t, slice.equal(spaces[:], want[:]), "%v", spaces)
	testing.expect(t, rcp_w)
	testing.expect(t, flat_id)

	vs := `#version 310 es
layout(location = 3) in vec4 position;
layout(location = 0) in ivec2 cell;
readonly buffer Offsets { vec4 offsets[]; };
uniform sampler2D heights;
uniform mat4 mvp;
out vec2 uv;
layout(location = 2) flat out int which;
layout(location = 4) flat out uvec3 bits;
void main() {
    uv = vec2(cell) / 8.0;
    which = gl_VertexID;
    bits = uvec3(cell.xyx) * 3u;
    gl_Position = mvp * (position + offsets[gl_VertexID] + textureLod(heights, uv, 0.0));
}`
	vm, verrs, vok := compile_glsl(vs, {stage = .Vertex, dxbc = true})
	defer destroy(vm)
	if !testing.expectf(t, vok, "%s", format_errors(verrs, context.temp_allocator)) {
		return
	}
	vins := dxbc_sig(dxbc_chunk(vm.dxbc, "ISGN"))
	vouts := dxbc_sig(dxbc_chunk(vm.dxbc, "OSGN"))
	want_vin := [?]Sig_Row{{"TEXCOORD", 0, 0, 2, 0, 3, 3}, {"TEXCOORD", 3, 0, 3, 3, 15, 15}, {"SV_VertexID", 0, 6, 1, 16, 1, 1}}
	want_vout := [?]Sig_Row{{"TEXCOORD", 0, 0, 3, 0, 3, 12}, {"TEXCOORD", 2, 0, 2, 2, 1, 14}, {"TEXCOORD", 4, 0, 1, 4, 7, 8}, {"SV_Position", 0, 1, 3, 16, 15, 0}}
	testing.expectf(t, slice.equal(vins, want_vin[:]), "vertex ISGN %v", vins)
	testing.expectf(t, slice.equal(vouts, want_vout[:]), "vertex OSGN %v", vouts)
	vversion, vprog := dxbc_program(t, vm, "vertex")
	testing.expect_value(t, vversion, 0x10051) // vs_5_1
	clear(&spaces)
	for in_ in vprog {
		#partial switch Dx_Op(in_.op) {
		case .Dcl_Constant_Buffer:
			append(&spaces, [3]u32{in_.op, in_.words[2], in_.words[5]})
		case .Dcl_Sampler, .Dcl_Resource_Raw:
			append(&spaces, [3]u32{in_.op, in_.words[2], in_.words[4]})
		case .Dcl_Resource:
			append(&spaces, [3]u32{in_.op | in_.ctl << 16, in_.words[2], in_.words[5]})
		}
	}
	// Vertex shader registers: t[n] and s[n] in space 0, b0 in space 1.
	want_v := [?][3]u32{{cb, 0, 1}, {sampler, 0, 0}, {resource | 3 << 16, 0, 0}, {raw, 1, 0}}
	testing.expectf(t, slice.equal(spaces[:], want_v[:]), "vertex %v", spaces)
}

// DXBC is opt-in: without Options.dxbc no stage has a blob (and the other
// outputs are as they were); with it, vertex and fragment shaders have
// one and a compute shader is an error (Dxbc_Compute).
@(test)
test_dxbc_opt_in :: proc(t: ^testing.T) {
	Case :: struct {
		src:   string,
		stage: Stage,
	}
	cases := [?]Case {
		{"#version 310 es\nprecision highp float;\nout vec4 c;\nvoid main() { c = vec4(gl_FragCoord.xy, 0.0, 1.0); }", .Fragment},
		{"#version 310 es\nin vec4 p;\nvoid main() { gl_Position = p; }", .Vertex},
		{"#version 310 es\nlayout(local_size_x = 8) in;\nbuffer B { uint w[]; };\nvoid main() { w[gl_GlobalInvocationID.x] = 1u; }", .Compute},
	}
	for cs in cases {
		m, errs, ok := compile_glsl(cs.src, {stage = cs.stage})
		testing.expectf(t, ok && m.dxbc == nil && len(m.spirv) > 0 && len(m.msl) > 0, "%v: %s", cs.stage, format_errors(errs, context.temp_allocator))
		spirv := slice.clone(m.spirv, context.temp_allocator)
		msl := strings.clone(m.msl, context.temp_allocator)
		destroy(m)
		dm, derrs, dok := compile_glsl(cs.src, {stage = cs.stage, dxbc = true})
		if cs.stage == .Compute {
			testing.expectf(t, !dok && len(derrs) == 1 && derrs[0].code == .Dxbc_Compute, "compute with dxbc: %v", derrs)
		} else {
			testing.expectf(t, dok && len(dm.dxbc) > 0, "%v: %s", cs.stage, format_errors(derrs, context.temp_allocator))
			testing.expectf(t, slice.equal(dm.spirv, spirv) && dm.msl == msl, "%v: DXBC changed the other outputs", cs.stage)
		}
		destroy(dm)
	}
}

// Compiles with DXBC and decodes the program.
@(private = "file")
dxbc_compile :: proc(t: ^testing.T, src: string, options: Options, loc := #caller_location) -> (m: ^Module, prog: []Ins, ok: bool) {
	o := options
	o.dxbc = true
	errs: []Error
	m, errs, ok = compile(src, o)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator), loc = loc) {
		return
	}
	_, prog = dxbc_program(t, m, src, loc)
	return
}

// The emitter's own lowering, by the instructions it leaves (every blob
// also goes through dxbc_program's structural checks): each `seq` must
// appear as consecutive instructions. A discard is followed by ret inside
// and outside loops; every loop has one shape, so a continue is a bare
// continue in a for (its step at the loop's top), a while and a do-while
// (its test at the top too, behind the first-pass flag), and a switch case
// that continues still ends with a break; integer division guards a zero
// divisor; an early return from a loop in a function breaks through a
// flag, from a switch in a switch through an if; run-time indexes are
// clamped; textureSize is resinfo_uint; texelFetch converts a uint lod as
// unsigned.
@(test)
test_dxbc_lowering :: proc(t: ^testing.T) {
	Case :: struct {
		src:     string,
		seqs:    [][]Dx_Op,
		profile: Profile,
	}
	cases := [?]Case {
		{"void mainImage(out vec4 c, in vec2 f) { for (int i = 0; i < 9; i++) { if (f.x > float(i)) discard; } c = vec4(1.0); }", {{.Discard, .Ret, .End_If}}, .Shadertoy},
		{"void mainImage(out vec4 c, in vec2 f) { if (f.x > 3.0) discard; c = vec4(1.0); }", {{.If, .Discard, .Ret, .End_If}}, .Shadertoy},
		{"void mainImage(out vec4 c, in vec2 f) { int a = 0; for (int i = 0; i < 4; i++) { switch (int(f.x) % 3) { case 0: continue; default: a++; } } c = vec4(float(a)); }", {{.Loop, .If}, {.End_If, .Mov, .Ilt, .Breakc}, {.Case, .Continue, .Break, .Default}}, .Shadertoy},
		{"void mainImage(out vec4 c, in vec2 f) { int a = 0, k = 0; do { k++; switch (int(f.x) % 3) { case 1: continue; } a++; } while (k < 4); c = vec4(float(a)); }", {{.Loop, .If, .Ilt, .Breakc, .End_If, .Mov}, {.Case, .Continue, .Break, .End_Switch}}, .Shadertoy},
		{"void mainImage(out vec4 c, in vec2 f) { int a = 0; while (a < int(f.x)) { a++; if (a == 3) continue; a++; } c = vec4(float(a)); }", {{.Loop, .Ftoi, .Ilt, .Breakc}, {.If, .Continue, .End_If}}, .Shadertoy},
		{"void mainImage(out vec4 c, in vec2 f) { int x = int(f.x) / int(f.y); c = vec4(float(x)); }", {{.Ieq, .Movc}}, .Shadertoy},
		{"void mainImage(out vec4 c, in vec2 f) { uint x = uint(f.x) % uint(f.y); c = vec4(float(x)); }", {{.Udiv, .Ieq, .Movc}}, .Shadertoy},
		{"int find(int n) { for (int i = 0; i < 8; i++) { for (int j = 0; j < 8; j++) { if (i * j == n) return i; } } return -1; }\nvoid mainImage(out vec4 c, in vec2 f) { c = vec4(float(find(int(f.x)))); }", {{.Mov, .Break, .End_If, .End_Loop, .Breakc, .End_Loop, .Breakc}}, .Shadertoy},
		{"int pick(int a, int b) { switch (a) { case 0: switch (b) { case 1: return 7; } return 3; } return 5; }\nvoid mainImage(out vec4 c, in vec2 f) { c = vec4(float(pick(int(f.x), int(f.y)))); }", {{.End_Switch, .If, .Break, .End_If}}, .Shadertoy},
		{"void mainImage(out vec4 c, in vec2 f) { float a[4] = float[4](1.0, 2.0, 3.0, 4.0); c = vec4(a[int(f.x)]); }", {{.Imax, .Imin, .Mov}}, .Shadertoy},
		{"uniform sampler2D s;\nout vec4 c;\nvoid main() { c = vec4(vec2(textureSize(s, 1)), 0.0, 1.0); }", {{.Resinfo}}, .Glsl},
		{"void mainImage(out vec4 c, in vec2 f) { c = texelFetch(iChannel0, ivec2(f), uint(f.x)); }", {{.Utof, .Sample_L}}, .Shadertoy},
	}
	for cs in cases {
		m, prog, ok := dxbc_compile(t, cs.src, {profile = cs.profile})
		defer destroy(m)
		if !ok {
			continue
		}
		for seq in cs.seqs {
			found := false
			for i in 0 ..= len(prog) - len(seq) {
				run := true
				for op, k in seq {
					if Dx_Op(prog[i + k].op) != op {
						run = false
						break
					}
				}
				if run {
					if seq[0] == .Resinfo {
						testing.expect_value(t, prog[i].ctl & 3, 2) // _uint
					}
					found = true
					break
				}
			}
			testing.expectf(t, found, "%s: %v not found", cs.src, seq)
		}
	}
}

// Run-time indexes are clamped to the last element, register or
// component, as on the CPU: an index one past it would read past an
// indexable temp (which drivers need not catch: RADV's selects return the
// last element anyway) or the wrong component.
@(test)
test_dxbc_index_clamps :: proc(t: ^testing.T) {
	src := "void mainImage(out vec4 c, in vec2 f) { int i = int(f.x); float a[5] = float[5](1.0, 2.0, 3.0, 4.0, 5.0); mat3 m = mat3(f.x); vec3 v = vec3(f, 1.0); vec2 w = f; a[i] = 0.5; v[i] = 2.0; c = vec4(a[i + 1], m[i].y, v[i + 2] + w[i], (w * 2.0)[i]); }"
	m, prog, ok := dxbc_compile(t, src, {profile = .Shadertoy})
	defer destroy(m)
	if !ok {
		return
	}
	// Every imax(i, 0) is followed by imin(i, last), one per index: a[i]
	// twice (last 4), m[i] (2), v[i] twice (2), w[i] and (w * 2.0)[i] (1).
	lasts := make([dynamic]u32, context.temp_allocator)
	for in_, i in prog {
		if Dx_Op(in_.op) != .Imin || i == 0 || Dx_Op(prog[i - 1].op) != .Imax {
			continue
		}
		lo, _ := dxbc_operands(prog[i - 1].words)
		hi, _ := dxbc_operands(in_.words)
		testing.expectf(t, len(lo) == 3 && lo[2].file == 4 && lo[2].imm == 0, "lower clamp %v", lo)
		if len(hi) == 3 && hi[2].file == 4 {
			append(&lasts, hi[2].imm)
		}
	}
	slice.sort(lasts[:])
	want := []u32{1, 1, 2, 2, 2, 4, 4}
	testing.expectf(t, slice.equal(lasts[:], want), "upper clamps %v, want %v", lasts, want)
}

// Storage buffer reads load only the words read, at word element * stride
// + offset (in words, before the clamp), clamped to 0x3ffffffc and then
// made a byte address: a field of a 6-vec4 struct is one ld_raw, a whole
// struct one per field (a struct field's own fields at their offsets), a
// component chosen at run time is clamped first.
@(test)
test_dxbc_buffer_reads :: proc(t: ^testing.T) {
	Case :: struct {
		src:     string,
		ld_raws: int,
		// Word offsets added before each clamp (0: none).
		offsets: []u32,
	}
	decls :: "#version 310 es\nprecision highp float;\nstruct S { vec4 a; vec4 b; vec4 c; vec4 d; vec4 e; vec4 clip; };\nreadonly buffer B { S rows[]; };\nreadonly buffer W { uvec3 v[]; };\nstruct Inner { vec2 x; float y; };\nstruct Outer { vec4 a; Inner b; uint c; float d; };\nreadonly buffer N { Outer ns[]; };\nuniform int uI;\nout vec4 o;\n"
	cases := [?]Case {
		{decls + "void main() { o = rows[uI].clip; }", 1, {20}},
		{decls + "void main() { o = vec4(rows[uI].b.zx, 0.0, 1.0); }", 1, {4}},
		{decls + "void main() { S s = rows[uI]; o = s.a + s.clip; }", 6, {0, 4, 8, 12, 16, 20}},
		{decls + "void main() { o = vec4(float(v[uI][uI + 1]), float(v[uI].y), 0.0, 1.0); }", 2, {0, 1}},
		{decls + "void main() { Outer n = ns[uI]; o = vec4(n.b.x, n.b.y + float(n.c), n.a.x + n.d); }", 5, {0, 4, 6, 8, 9}},
		{decls + "void main() { Inner b = ns[uI].b; o = vec4(b.x, b.y, ns[uI].b.y); }", 3, {4, 6, 6}},
	}
	for cs in cases {
		m, prog, ok := dxbc_compile(t, cs.src, {profile = .Glsl})
		defer destroy(m)
		if !ok {
			continue
		}
		n := 0
		for in_, i in prog {
			if Dx_Op(in_.op) != .Ld_Raw {
				continue
			}
			// ... [iadd a, base, off] umin a, w, 0x3ffffffc; ishl a, a, 2; ld_raw
			testing.expectf(t, i >= 2 && Dx_Op(prog[i - 2].op) == .Umin && Dx_Op(prog[i - 1].op) == .Ishl, "%s: ld_raw %d not after umin, ishl", cs.src, n)
			if i >= 2 {
				ops, _ := dxbc_operands(prog[i - 2].words)
				testing.expectf(t, len(ops) == 3 && ops[2].file == 4 && ops[2].imm == 0x3ffffffc, "%s: clamp %v", cs.src, ops)
				shl, _ := dxbc_operands(prog[i - 1].words)
				testing.expectf(t, len(shl) == 3 && shl[2].file == 4 && shl[2].imm == 2, "%s: shift %v", cs.src, shl)
			}
			off: u32
			if i >= 3 && Dx_Op(prog[i - 3].op) == .Iadd {
				ops, _ := dxbc_operands(prog[i - 3].words)
				if len(ops) == 3 && ops[2].file == 4 {
					off = ops[2].imm
				}
			}
			if n < len(cs.offsets) {
				testing.expectf(t, off == cs.offsets[n], "%s: ld_raw %d at word offset %d, want %d", cs.src, n, off, cs.offsets[n])
			}
			n += 1
		}
		testing.expectf(t, n == cs.ld_raws, "%s: %d ld_raws, want %d", cs.src, n, cs.ld_raws)
	}
}

// An allocator over the heap that fills every block it frees or moves
// with 0xa5 first, so code that keeps a pointer into a dynamic array
// across an append that grows it reads garbage instead of the old values.
@(private = "file")
poison_allocator_proc :: proc(data: rawptr, mode: mem.Allocator_Mode, size, alignment: int, old_memory: rawptr, old_size: int, loc := #caller_location) -> ([]byte, mem.Allocator_Error) {
	heap := os.heap_allocator()
	#partial switch mode {
	case .Free:
		if old_memory != nil && old_size > 0 {
			mem.set(old_memory, 0xa5, old_size)
		}
	case .Resize, .Resize_Non_Zeroed:
		out, err := mem.alloc_bytes(size, alignment, heap, loc)
		if err != nil || old_memory == nil {
			return out, err
		}
		mem.copy(raw_data(out), old_memory, min(size, old_size))
		mem.set(old_memory, 0xa5, old_size)
		mem.free_with_size(old_memory, old_size, heap, loc)
		return out, nil
	case .Free_All:
		// Its blocks are left to the process.
		return nil, nil
	}
	return os.heap_allocator_proc(data, mode, size, alignment, old_memory, old_size, loc)
}

// The emitter's output does not depend on how its scratch is reused: a
// chain of 12 functions each returning the next one's result (so the call
// stack grows past its first capacity while a return is being emitted)
// gives the same blob with a scratch allocator that poisons what it frees.
@(test)
test_dxbc_scratch_reuse :: proc(t: ^testing.T) {
	sb := strings.builder_make(context.temp_allocator)
	fmt.sbprint(&sb, "float f12(float x) { if (x > 2.0) { return x * 0.5; } return x + 1.0; }\n")
	for i := 11; i >= 1; i -= 1 {
		fmt.sbprintf(&sb, "float f%d(float x) {{ if (x < -1.0) {{ return -x; }} return f%d(x * 1.5) + %d.0; }}\n", i, i + 1, i)
	}
	fmt.sbprint(&sb, "void mainImage(out vec4 c, in vec2 p) { c = vec4(f1(p.x), f1(p.y), 0.0, 1.0); }\n")
	src := strings.to_string(sb)
	m, errs, ok := compile(src, {profile = .Shadertoy, dxbc = true})
	defer destroy(m)
	if !testing.expectf(t, ok, "%s", format_errors(errs, context.temp_allocator)) {
		return
	}
	want := slice.clone(m.dxbc, context.temp_allocator)
	_, _ = dxbc_program(t, m, "call chain")
	poisoned: []u8
	{
		context.temp_allocator = {procedure = poison_allocator_proc, data = nil}
		pm, _, pok := compile(src, {profile = .Shadertoy, dxbc = true})
		if pok {
			poisoned = slice.clone(pm.dxbc, context.allocator)
		}
		destroy(pm)
	}
	defer delete(poisoned)
	testing.expect(t, slice.equal(poisoned, want))
}

// The module's arena holds the DXBC container and nothing else of the
// emitter's: its scratch is temporary.
@(test)
test_dxbc_arena :: proc(t: ^testing.T) {
	src := #load("testdata/raymarch_sdf.glsl", string)
	options, _ := options_of_path("raymarch_sdf.glsl")
	a, _, aok := compile(src, options)
	defer destroy(a)
	options.dxbc = true
	b, _, bok := compile(src, options)
	defer destroy(b)
	if !testing.expect(t, aok && bok) {
		return
	}
	grown := int(b.arena.total_used) - int(a.arena.total_used)
	// Alignment padding (here and in what follows it) aside.
	testing.expectf(t, abs(grown - len(b.dxbc)) <= 64, "the arena grew %d bytes for a %d-byte container", grown, len(b.dxbc))
}

// vkd3d is a development tool: tests that use it skip without it.
@(private = "file")
have_vkd3d :: proc() -> bool {
	_, _, _, err := os.process_exec({command = {"vkd3d-compiler", "--version"}}, context.temp_allocator)
	_, _, _, err2 := os.process_exec({command = {"vkd3d-dxbc", "--version"}}, context.temp_allocator)
	return err == nil && err2 == nil && have_spirv_val()
}

@(private = "file")
dxbc_files: int

// vkd3d-dxbc reads the container (checking its checksum), vkd3d-compiler
// disassembles it and translates it to SPIR-V without a message, and
// spirv-val accepts the SPIR-V.
@(private = "file")
expect_valid_dxbc :: proc(t: ^testing.T, name: string, blob: []u8, loc := #caller_location) {
	dir := os.temp_directory(context.temp_allocator) or_else "/tmp"
	base := fmt.tprintf("%s/oksl_test_%d_%d", dir, os.get_pid(), sync.atomic_add(&dxbc_files, 1))
	in_path, spv := fmt.tprintf("%s.dxbc", base), fmt.tprintf("%s.spv", base)
	_ = os.write_entire_file(in_path, blob)
	defer os.remove(in_path)
	defer os.remove(spv)
	commands := [?][]string {
		{"vkd3d-dxbc", "-t", in_path},
		{"vkd3d-compiler", "-x", "dxbc-tpf", "-b", "d3d-asm", "-o", "/dev/null", in_path},
		{"vkd3d-compiler", "-x", "dxbc-tpf", "-b", "spirv-binary", "-o", spv, in_path},
		{"spirv-val", "--target-env", "vulkan1.0", spv},
	}
	for c, i in commands {
		state, stdout, stderr, err := os.process_exec({command = c}, context.temp_allocator)
		quiet := i == 0 || len(stderr) == 0
		if !testing.expectf(t, err == nil && state.exit_code == 0 && quiet, "%s: %s: %s%s", name, c[0], stdout, stderr, loc = loc) {
			return
		}
	}
}

// Every testdata vertex and fragment shader's DXBC through dxbc_program's
// checks, and through vkd3d and spirv-val.
@(test)
test_dxbc_vkd3d :: proc(t: ^testing.T) {
	check_vkd3d := have_vkd3d()
	if !check_vkd3d {
		fmt.println("vkd3d-compiler, vkd3d-dxbc or spirv-val not installed; checking the programs only")
	}
	for path in corpus() {
		src, _ := os.read_entire_file(path, context.temp_allocator)
		options, _ := options_of_path(path)
		if options.stage == .Compute {
			continue
		}
		options.dxbc = true
		m, errs, ok := compile(string(src), options)
		defer destroy(m)
		if !testing.expectf(t, ok, "%s: %s", path, format_errors(errs, context.temp_allocator)) {
			continue
		}
		dxbc_program(t, m, path)
		if check_vkd3d {
			expect_valid_dxbc(t, path, m.dxbc)
		}
	}
}
