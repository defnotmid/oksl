package oksl

import "core:fmt"
import "core:os"
import "core:strings"
import "core:sync"
import "core:testing"

@(private = "file")
oplat_spv_counter: int

@(private = "file")
has_error_code :: proc(errs: []Error, code: Error_Code) -> bool {
	for e in errs {
		if e.code == code {
			return true
		}
	}
	return false
}

@(private = "file")
expect_spirv_v13 :: proc(t: ^testing.T, name: string, words: []u32, loc := #caller_location) {
	if _, _, _, err := os.process_exec({command = {"spirv-val", "--version"}}, context.temp_allocator); err != nil {
		return
	}
	dir := os.temp_directory(context.temp_allocator) or_else "/tmp"
	out := fmt.tprintf("%s/oksl_oplat_test_%d_%d.spv", dir, os.get_pid(), sync.atomic_add(&oplat_spv_counter, 1))
	_ = os.write_entire_file(out, ([^]u8)(raw_data(words))[:len(words) * 4])
	state, stdout, stderr, err := os.process_exec(
		{command = {"spirv-val", "--target-env", "vulkan1.3", "--scalar-block-layout", out}},
		context.temp_allocator,
	)
	os.remove(out)
	testing.expectf(t, err == nil && state.exit_code == 0, "%s: %s%s", name, stdout, stderr, loc = loc)
}

@(private = "file")
spirv_has_op :: proc(words: []u32, target_op: u16) -> bool {
	i := 5
	for i < len(words) {
		wc := int(words[i] >> 16)
		op := u16(words[i] & 0xffff)
		if op == target_op {
			return true
		}
		if wc <= 0 {
			break
		}
		i += wc
	}
	return false
}

@(private = "file")
spirv_has_cap :: proc(words: []u32, target_cap: u32) -> bool {
	i := 5
	for i < len(words) {
		wc := int(words[i] >> 16)
		op := u16(words[i] & 0xffff)
		if op == 17 && wc >= 2 && words[i + 1] == target_cap {
			return true
		}
		if wc <= 0 {
			break
		}
		i += wc
	}
	return false
}

@(private = "file")
spirv_has_decor :: proc(words: []u32, target_decor: u32) -> bool {
	i := 5
	for i < len(words) {
		wc := int(words[i] >> 16)
		op := u16(words[i] & 0xffff)
		if (op == 71 && wc >= 3 && words[i + 2] == target_decor) ||
		   (op == 72 && wc >= 4 && words[i + 3] == target_decor) {
			return true
		}
		if wc <= 0 {
			break
		}
		i += wc
	}
	return false
}

@(private = "file")
find_push_field :: proc(m: ^Module, name: string) -> (Push_Field_Info, bool) {
	for f in m.push.fields {
		if f.name == name {
			return f, true
		}
	}
	return {}, false
}

@(private = "file")
find_struct :: proc(m: ^Module, name: string) -> (Buffer_Struct_Info, bool) {
	for s in m.structs {
		if s.name == name {
			return s, true
		}
	}
	return {}, false
}

@(test)
test_oplat_layout_oracle :: proc(t: ^testing.T) {
	// Case 1: All scalar & vector types + arrays in <= 128 B
	ScalarVecPush :: struct {
		f:  f32,
		i:  i32,
		u:  u32,
		v2: [2]f32,
		v3: [3]f32,
		v4: [4]f32,
		iv: [2]i32,
		uv: [3]u32,
		fa: [3]f32,
	}
	src1 := `#version 300 es
precision highp float;
push ScalarVecPush {
	float f;
	int i;
	uint u;
	vec2 v2;
	vec3 v3;
	vec4 v4;
	ivec2 iv;
	uvec3 uv;
	float fa[3];
} p;
out vec4 color;
void main() {
	color = vec4(p.f + float(p.i) + float(p.u), p.v2.x + p.v3.y, p.v4.z + float(p.iv.x), float(p.uv.z) + p.fa[2]);
}`
	m1, errs1, ok1 := compile(src1, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
	defer destroy(m1)
	testing.expectf(t, ok1, "ScalarVecPush: %s", format_errors(errs1, context.temp_allocator))
	if ok1 {
		testing.expect_value(t, m1.layout, Layout.Oplat)
		testing.expect_value(t, m1.push.name, "ScalarVecPush")
		testing.expect_value(t, m1.push.size, size_of(ScalarVecPush))
		testing.expect_value(t, m1.push.align, align_of(ScalarVecPush))
		want_offsets := [?]struct {
			name:   string,
			offset: int,
			size:   int,
			kind:   Push_Field_Kind,
		}{
			{"f", int(offset_of(ScalarVecPush, f)), size_of(f32), .Value},
			{"i", int(offset_of(ScalarVecPush, i)), size_of(i32), .Value},
			{"u", int(offset_of(ScalarVecPush, u)), size_of(u32), .Value},
			{"v2", int(offset_of(ScalarVecPush, v2)), size_of([2]f32), .Value},
			{"v3", int(offset_of(ScalarVecPush, v3)), size_of([3]f32), .Value},
			{"v4", int(offset_of(ScalarVecPush, v4)), size_of([4]f32), .Value},
			{"iv", int(offset_of(ScalarVecPush, iv)), size_of([2]i32), .Value},
			{"uv", int(offset_of(ScalarVecPush, uv)), size_of([3]u32), .Value},
			{"fa", int(offset_of(ScalarVecPush, fa)), size_of([3]f32), .Value},
		}
		testing.expect_value(t, len(m1.push.fields), len(want_offsets))
		for w in want_offsets {
			pf, found := find_push_field(m1, w.name)
			testing.expectf(t, found, "missing field %s", w.name)
			testing.expect_value(t, pf.offset, w.offset)
			testing.expect_value(t, pf.size, w.size)
			testing.expect_value(t, pf.kind, w.kind)
		}
		asserts := format_odin_asserts(m1, context.temp_allocator)
		testing.expectf(t, strings.contains(asserts, fmt.tprintf("#assert(size_of(ScalarVecPush) == %d)", size_of(ScalarVecPush))), "%s", asserts)
		testing.expectf(t, strings.contains(asserts, fmt.tprintf("#assert(align_of(ScalarVecPush) == %d)", align_of(ScalarVecPush))), "%s", asserts)
		testing.expectf(t, strings.contains(asserts, fmt.tprintf("#assert(offset_of(ScalarVecPush, uv) == %d)", int(offset_of(ScalarVecPush, uv)))), "%s", asserts)
		// MSL must use packed_ types for 2/3/4 vectors and static_asserts.
		testing.expectf(t, strings.contains(m1.msl, "constant ok_ScalarVecPush&"), "%s", m1.msl)
		testing.expectf(t, strings.contains(m1.msl, "[[buffer(0)]]"), "%s", m1.msl)
		testing.expectf(t, strings.contains(m1.msl, "packed_float3"), "%s", m1.msl)
		testing.expectf(t, strings.contains(m1.msl, "packed_int2"), "%s", m1.msl)
		testing.expectf(t, strings.contains(m1.msl, "packed_uint3"), "%s", m1.msl)
		testing.expectf(t, strings.contains(m1.msl, "static_assert(sizeof(ok_ScalarVecPush)"), "%s", m1.msl)
	}

	// Case 2: All 5 alignment/padding edge cases + exact Odin oracle checks.
	// 2a: vec3 + float -> 16 B, align 4 (no padding)
	Vec3Float :: struct {
		a: [3]f32,
		b: f32,
	}
	// 2b: float + mat4 -> 80 B, align 16 (12 bytes pad before mat4)
	FloatMat4 :: struct {
		a: f32,
		m: matrix[4, 4]f32,
	}
	// 2c: float + mat2 -> 24 B, align 8 (4 bytes pad before mat2!)
	FloatMat2 :: struct {
		a: f32,
		m: matrix[2, 2]f32,
	}
	// 2d: vec2 + mat3 + float -> 48 B, align 4 (mat3 size 36, col stride 12)
	Vec2Mat3Float :: struct {
		a: [2]f32,
		m: matrix[3, 3]f32,
		b: f32,
	}
	// 2e: mat4 + float -> 80 B, align 16 (12 bytes tail padding!)
	Mat4FloatTail :: struct {
		m: matrix[4, 4]f32,
		a: f32,
	}
	// 2f: Nested struct with tail rounding (`S2` must round up from 68 to 80 before `c`).
	S2 :: struct {
		m: matrix[4, 4]f32,
		a: f32,
	}
	NestedRound :: struct {
		s: S2,
		c: [4]f32,
	}
	// 2g: Nested struct array + packed mat3 in MSL
	Sub :: struct {
		v: [3]f32,
		m: matrix[2, 2]f32,
	}
	SubArrayPush :: struct {
		x:  f32,
		s:  [2]Sub,
		m3: matrix[3, 3]f32,
	}
	// 2h: Exact 128 B boundary
	Exact128 :: struct {
		m:  matrix[4, 4]f32,
		v:  [4]f32,
		m3: matrix[3, 3]f32,
		a:  [3]f32,
	}

	Edge_Case :: struct {
		name:     string,
		src:      string,
		want_sz:  int,
		want_al:  int,
		offs:     []int,
		want_msl: []string,
	}
	edges := [?]Edge_Case{
		{
			"Vec3Float",
			`#version 300 es
precision highp float;
push Vec3Float { vec3 a; float b; } u;
out vec4 c; void main() { c = vec4(u.a, u.b); }`,
			size_of(Vec3Float), align_of(Vec3Float),
			{int(offset_of(Vec3Float, a)), int(offset_of(Vec3Float, b))},
			{"packed_float3"},
		},
		{
			"FloatMat4",
			`#version 300 es
precision highp float;
push FloatMat4 { float a; mat4 m; } u;
out vec4 c; void main() { c = u.m * vec4(u.a); }`,
			size_of(FloatMat4), align_of(FloatMat4),
			{int(offset_of(FloatMat4, a)), int(offset_of(FloatMat4, m))},
			{"_pad0", "float4x4"},
		},
		{
			"FloatMat2",
			`#version 300 es
precision highp float;
push FloatMat2 { float a; mat2 m; } u;
out vec4 c; void main() { c = vec4(u.m * vec2(u.a), 0.0, 1.0); }`,
			size_of(FloatMat2), align_of(FloatMat2),
			{int(offset_of(FloatMat2, a)), int(offset_of(FloatMat2, m))},
			{"_pad0", "float2x2", "alignof(ok_FloatMat2) == 8"},
		},
		{
			"Vec2Mat3Float",
			`#version 300 es
precision highp float;
push Vec2Mat3Float { vec2 a; mat3 m; float b; } u;
out vec4 c; void main() { c = vec4(u.m * vec3(u.a, u.b), 1.0); }`,
			size_of(Vec2Mat3Float), align_of(Vec2Mat3Float),
			{int(offset_of(Vec2Mat3Float, a)), int(offset_of(Vec2Mat3Float, m)), int(offset_of(Vec2Mat3Float, b))},
			{"packed_float3 ok_m[3];", "oksl_unpack_mat3", "return float3x3(float3(m[0]), float3(m[1]), float3(m[2]));"},
		},
		{
			"Mat4FloatTail",
			`#version 300 es
precision highp float;
push Mat4FloatTail { mat4 m; float a; } u;
out vec4 c; void main() { c = u.m * vec4(u.a); }`,
			size_of(Mat4FloatTail), align_of(Mat4FloatTail),
			{int(offset_of(Mat4FloatTail, m)), int(offset_of(Mat4FloatTail, a))},
			{"sizeof(ok_Mat4FloatTail) == 80", "char _pad0[12];"},
		},
		{
			"NestedRound",
			`#version 300 es
precision highp float;
struct S2 { mat4 m; float a; };
push NestedRound { S2 s; vec4 c; } u;
out vec4 o; void main() { o = u.s.m * u.c + vec4(u.s.a); }`,
			size_of(NestedRound), align_of(NestedRound),
			{int(offset_of(NestedRound, s)), int(offset_of(NestedRound, c))},
			{"oksl_push_S2", "sizeof(oksl_push_S2) == 80", "__builtin_offsetof(ok_NestedRound, ok_c) == 80"},
		},
		{
			"SubArrayPush",
			`#version 300 es
precision highp float;
struct Sub { vec3 v; mat2 m; };
push SubArrayPush { float x; Sub s[2]; mat3 m3; } u;
out vec4 o; void main() { o = vec4(u.m3 * (u.s[1].v + vec3(u.s[0].m[0], u.x)), 1.0); }`,
			size_of(SubArrayPush), align_of(SubArrayPush),
			{int(offset_of(SubArrayPush, x)), int(offset_of(SubArrayPush, s)), int(offset_of(SubArrayPush, m3))},
			{"oksl_push_Sub", "oksl_unpack_mat3", "packed_float3"},
		},
		{
			"Exact128",
			`#version 300 es
precision highp float;
push Exact128 { mat4 m; vec4 v; mat3 m3; float a[3]; } u;
out vec4 o; void main() { o = u.m * u.v + vec4(u.m3 * vec3(u.a[0], u.a[1], u.a[2]), 1.0); }`,
			size_of(Exact128), align_of(Exact128),
			{int(offset_of(Exact128, m)), int(offset_of(Exact128, v)), int(offset_of(Exact128, m3)), int(offset_of(Exact128, a))},
			{"sizeof(ok_Exact128) == 128"},
		},
	}

	for ec in edges {
		m, errs, ok := compile(ec.src, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
		testing.expectf(t, ok, "%s: %s", ec.name, format_errors(errs, context.temp_allocator))
		if ok {
			testing.expect_value(t, m.push.size, ec.want_sz)
			testing.expect_value(t, m.push.align, ec.want_al)
			testing.expect_value(t, len(m.push.fields), len(ec.offs))
			for i in 0 ..< min(len(m.push.fields), len(ec.offs)) {
				testing.expectf(t, m.push.fields[i].offset == ec.offs[i], "%s field %d: got %d want %d", ec.name, i, m.push.fields[i].offset, ec.offs[i])
			}
			for sub in ec.want_msl {
				testing.expectf(t, strings.contains(m.msl, sub), "%s msl missing %q:\n%s", ec.name, sub, m.msl)
			}
			if ec.name == "NestedRound" {
				st, found := find_struct(m, "S2")
				testing.expect(t, found)
				testing.expect_value(t, st.size, size_of(S2))
				testing.expect_value(t, st.align, align_of(S2))
				testing.expect_value(t, st.fields[0].offset, int(offset_of(S2, m)))
				testing.expect_value(t, st.fields[1].offset, int(offset_of(S2, a)))
			}
			if ec.name == "SubArrayPush" {
				st, found := find_struct(m, "Sub")
				testing.expect(t, found)
				testing.expect_value(t, st.size, size_of(Sub))
				testing.expect_value(t, st.align, align_of(Sub))
				testing.expect_value(t, st.fields[0].offset, int(offset_of(Sub, v)))
				testing.expect_value(t, st.fields[1].offset, int(offset_of(Sub, m)))
			}
		}
		destroy(m)
	}

	// MSL push struct must carry the `ok_` prefix so names like `constant` or `half` never collide.
	kw_src := `#version 300 es
precision highp float;
push constant { float x; } u;
out vec4 c; void main() { c = vec4(u.x); }`
	mk, kerrs, kok := compile(kw_src, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
	testing.expectf(t, kok, "keyword block: %s", format_errors(kerrs, context.temp_allocator))
	if kok {
		testing.expect_value(t, mk.push.name, "constant")
		testing.expectf(t, strings.contains(mk.msl, "struct ok_constant {"), "%s", mk.msl)
		testing.expectf(t, strings.contains(mk.msl, "constant ok_constant&"), "%s", mk.msl)
		testing.expectf(t, !strings.contains(mk.msl, "struct constant {"), "%s", mk.msl)
	}
	destroy(mk)
}

@(test)
test_oplat_diagnostics :: proc(t: ^testing.T) {
	Err_Case :: struct {
		name:    string,
		src:     string,
		opts:    Options,
		want:    Error_Code,
		want_ok: bool,
	}
	cases := [?]Err_Case{
		{
			"duplicate push block",
			`#version 300 es
precision highp float;
push A { float a; } a;
push B { float b; } b;
out vec4 c; void main() { c = vec4(a.a + b.b); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Duplicate, false,
		},
		{
			"push under Sdl_Gpu",
			`#version 300 es
precision highp float;
push A { float a; } a;
out vec4 c; void main() { c = vec4(a.a); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Sdl_Gpu},
			.Push_Layout, false,
		},
		{
			"push under SkSL",
			`push A { float a; } a;
half4 main(float2 p) { return half4(a.a); }`,
			{profile = .Sksl, stage = .Fragment, layout = .Oplat},
			.Push_Profile, false,
		},
		{
			"push mixed with uniform",
			`#version 300 es
precision highp float;
push A { float a; } a;
uniform float b;
out vec4 c; void main() { c = vec4(a.a + b); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Mixed_Decl, false,
		},
		{
			"push mixed with buffer",
			`#version 310 es
precision highp float;
push A { float a; } a;
layout(std430, binding = 0) readonly buffer B { float b[]; };
out vec4 c; void main() { c = vec4(a.a + b[0]); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Mixed_Decl, false,
		},
		{
			"push mixed with sampler2D",
			`#version 300 es
precision highp float;
push A { vec2 uv; } a;
uniform sampler2D tex;
out vec4 c; void main() { c = texture(tex, a.uv); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Mixed_Decl, false,
		},
		{
			"push mixed with image2D",
			`#version 310 es
layout(local_size_x = 1) in;
push A { uint n; } a;
layout(rgba8, binding = 0) writeonly uniform image2D img;
void main() { imageStore(img, ivec2(0), vec4(float(a.n))); }`,
			{profile = .Glsl, stage = .Compute, layout = .Oplat},
			.Push_Mixed_Decl, false,
		},
		{
			"empty push block",
			`#version 300 es
precision highp float;
push Empty { };
out vec4 c; void main() { c = vec4(1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Empty, false,
		},
		{
			"contract 144 B > 128 B",
			`#version 300 es
precision highp float;
push Over128 {
	mat4 view_proj;
	vec4 tint;
	uint count;
	vec3 dir;
	float time;
	mat3 rot;
} u;
out vec4 c; void main() { c = u.view_proj * u.tint + vec4(u.rot * u.dir, u.time + float(u.count)); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Too_Large, false,
		},
		{
			"bool direct field",
			`#version 300 es
precision highp float;
push P { bool flag; } u;
out vec4 c; void main() { c = vec4(u.flag ? 1.0 : 0.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Bool, false,
		},
		{
			"bvec2 field",
			`#version 300 es
precision highp float;
push P { bvec2 flags; } u;
out vec4 c; void main() { c = vec4(u.flags.x ? 1.0 : 0.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Bool, false,
		},
		{
			"nested bool inside struct array",
			`#version 300 es
precision highp float;
struct S { bool bad[2]; };
push P { S s; } u;
out vec4 c; void main() { c = vec4(u.s.bad[0] ? 1.0 : 0.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Bool, false,
		},
		{
			"non-square matrix",
			`#version 300 es
precision highp float;
push P { mat2x3 m; } u;
out vec4 c; void main() { c = vec4(u.m * vec2(1.0), 1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Matrix_Shape, false,
		},
		{
			"matrix array",
			`#version 300 es
precision highp float;
push P { mat2 m[2]; } u;
out vec4 c; void main() { c = vec4(u.m[0][0], 0.0, 1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Matrix_Array, false,
		},
		{
			"write to push field",
			`#version 300 es
precision highp float;
push P { float x; } u;
out vec4 c; void main() { u.x = 1.0; c = vec4(u.x); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Read_Only, false,
		},
		{
			"out parameter push field",
			`#version 300 es
precision highp float;
push P { float x; } u;
void mut(out float v) { v = 1.0; }
out vec4 c; void main() { mut(u.x); c = vec4(u.x); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Read_Only, false,
		},
		{
			"bare push instance expr",
			`#version 300 es
precision highp float;
push P { float x; } u;
out vec4 c; void main() { float v = float(u); c = vec4(v); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Block_Use, false,
		},
		{
			"pointer field",
			`#version 300 es
precision highp float;
push P { float* ptr; } u;
out vec4 c; void main() { c = vec4(1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Unsupported_Field, false,
		},
		{
			"texture2D handle field",
			`#version 300 es
precision highp float;
push P { texture2D tex; } u;
out vec4 c; void main() { c = vec4(1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Unsupported_Field, false,
		},
		{
			"sampler handle field",
			`#version 300 es
precision highp float;
push P { sampler s; } u;
out vec4 c; void main() { c = vec4(1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Unsupported_Field, false,
		},
		{
			"image2D handle field",
			`#version 300 es
precision highp float;
push P { image2D im; } u;
out vec4 c; void main() { c = vec4(1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Push_Unsupported_Field, false,
		},
		{
			"loose uniform without push in GLSL",
			`#version 300 es
precision highp float;
uniform float a;
out vec4 c; void main() { c = vec4(a); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Oplat_No_Push, false,
		},
		{
			"unsupported sampler2D without push in GLSL",
			`#version 300 es
precision highp float;
uniform sampler2D tex;
out vec4 c; void main() { c = texture(tex, vec2(0.5)); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.Oplat_Unsupported_Resource, false,
		},
		{
			"SkSL with uniform under Oplat",
			`uniform float4 tint;
half4 main(float2 p) { return half4(tint); }`,
			{profile = .Sksl, stage = .Fragment, layout = .Oplat},
			.Oplat_No_Push, false,
		},
		{
			"Shadertoy with iTime/uniform under Oplat",
			`void mainImage(out vec4 c, in vec2 p) { c = vec4(sin(iTime), 0.0, 0.0, 1.0); }`,
			{profile = .Shadertoy, stage = .Fragment, layout = .Oplat},
			.Oplat_No_Push, false,
		},
		{
			"dxbc requested with Oplat",
			`#version 300 es
precision highp float;
push P { float x; } u;
out vec4 c; void main() { c = vec4(u.x); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat, dxbc = true},
			.Oplat_Dxbc, false,
		},
		{
			"vertex input under Oplat rejected",
			`#version 300 es
in vec3 pos;
void main() { gl_Position = vec4(pos, 1.0); }`,
			{profile = .Glsl, stage = .Vertex, layout = .Oplat},
			.Oplat_Vertex_Input, false,
		},
		{
			"vertex prelude used input under Oplat rejected",
			`#version 300 es
void main() { gl_Position = vec4(pre_pos, 1.0); }`,
			{profile = .Glsl, stage = .Vertex, layout = .Oplat, prelude = "in vec3 pre_pos;\n"},
			.Oplat_Vertex_Input, false,
		},
		{
			"vertex prelude unused input under Oplat succeeds",
			`#version 300 es
void main() { gl_Position = vec4(float(gl_VertexID), 0.0, 0.0, 1.0); }`,
			{profile = .Glsl, stage = .Vertex, layout = .Oplat, prelude = "in vec3 pre_pos;\n"},
			.None, true,
		},
		{
			"fragment input under Oplat succeeds",
			`#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 c; void main() { c = vec4(v_uv, 0.0, 1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.None, true,
		},
		{
			"taps under Oplat rejected",
			`#version 300 es
void main() { gl_Position = vec4(0.0); }`,
			{profile = .Glsl, stage = .Vertex, layout = .Oplat, prelude = "out vec4 openClip;\n", taps = []Tap{{position = "openClip"}}},
			.Oplat_Taps, false,
		},
		{
			"prelude used uniform mixed with push rejected",
			`#version 300 es
precision highp float;
push P { float x; } u;
out vec4 c; void main() { c = vec4(u.x + used_pre); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat, prelude = "uniform float used_pre;\n"},
			.Push_Mixed_Decl, false,
		},
		{
			"GLSL with no push and no uniforms succeeds",
			`#version 300 es
precision highp float;
out vec4 c; void main() { c = vec4(0.25, 0.5, 0.75, 1.0); }`,
			{profile = .Glsl, stage = .Fragment, layout = .Oplat},
			.None, true,
		},
	}

	for tc in cases {
		m, errs, ok := compile(tc.src, tc.opts)
		if tc.want_ok {
			testing.expectf(t, ok, "%s: expected ok, got %s", tc.name, format_errors(errs, context.temp_allocator))
			if ok {
				testing.expect_value(t, m.push.size, 0)
				testing.expect_value(t, len(m.push.fields), 0)
			}
		} else {
			testing.expectf(t, !ok && has_error_code(errs, tc.want), "%s: want %v, got ok=%v errs=%s", tc.name, tc.want, ok, format_errors(errs, context.temp_allocator))
		}
		destroy(m)
	}

	// Prelude with an unused uniform does not block a push block.
	m_pre, errs_pre, ok_pre := compile(
		`#version 300 es
precision highp float;
push P { float x; } u;
out vec4 c; void main() { c = vec4(u.x); }`,
		{profile = .Glsl, stage = .Fragment, layout = .Oplat, prelude = "uniform float unused_pre;\n"},
	)
	testing.expectf(t, ok_pre, "unused prelude uniform: %s", format_errors(errs_pre, context.temp_allocator))
	if ok_pre {
		testing.expect_value(t, m_pre.push.size, 4)
	}
	destroy(m_pre)
}

@(test)
test_oplat_pipeline :: proc(t: ^testing.T) {
	// Valid pipeline: matching push block with different instance names (`vp` vs `fp`)
	// and different unused helper structs.
	vs_ok := `#version 300 es
struct UnusedV { float z; };
struct Sub { vec3 v; mat2 m; };
push Frame { mat4 mvp; Sub sub; float t; } vp;
void main() {
	vec3 pos = vec3(float(gl_VertexID), 0.0, 1.0);
	gl_Position = vp.mvp * vec4(pos + vp.sub.v * vp.t + vec3(vp.sub.m[0], 0.0), 1.0);
}`
	fs_ok := `#version 300 es
precision highp float;
struct UnusedF { int k; };
struct Sub { vec3 v; mat2 m; };
push Frame { mat4 mvp; Sub sub; float t; } fp;
out vec4 col;
void main() { col = vec4(fp.sub.v, fp.t) + fp.mvp[3]; }`
	p_ok, errs_ok, ok := compile_pipeline({vertex = vs_ok, fragment = fs_ok}, {layout = .Oplat})
	testing.expectf(t, ok, "matching pipeline failed: %s", format_errors(errs_ok, context.temp_allocator))
	if ok {
		testing.expect_value(t, p_ok.vertex.push.size, p_ok.fragment.push.size)
		testing.expect_value(t, len(p_ok.vertex.push.fields), 3)
		testing.expect_value(t, p_ok.vertex.push.name, "Frame")
		expect_spirv_v13(t, "pipeline.vertex", p_ok.vertex.spirv)
		expect_spirv_v13(t, "pipeline.fragment", p_ok.fragment.spirv)
	}
	destroy_pipeline(p_ok)

	// Valid pipeline: neither stage declares push block or uniforms.
	vs_nopush := `#version 300 es
void main() { gl_Position = vec4(float(gl_VertexID), 0.0, 0.0, 1.0); }`
	fs_nopush := `#version 300 es
precision highp float;
out vec4 col; void main() { col = vec4(1.0); }`
	p_np, errs_np, ok_np := compile_pipeline({vertex = vs_nopush, fragment = fs_nopush}, {layout = .Oplat})
	testing.expectf(t, ok_np, "nopush pipeline failed: %s", format_errors(errs_np, context.temp_allocator))
	destroy_pipeline(p_np)

	// Taps in compile_pipeline are rejected under .Oplat.
	p_tap, errs_tap, ok_tap := compile_pipeline(
		{vertex = vs_nopush, fragment = fs_nopush, vertex_prelude = "out vec4 openClip;\n"},
		{layout = .Oplat, taps = []Tap{{position = "openClip"}}},
	)
	testing.expectf(t, !ok_tap && has_error_code(errs_tap, .Oplat_Taps), "pipeline taps: got ok=%v errs=%s", ok_tap, format_errors(errs_tap, context.temp_allocator))
	destroy_pipeline(p_tap)

	Mismatch :: struct {
		name: string,
		vs:   string,
		fs:   string,
		want: Error_Code,
	}
	vs_frame := `#version 300 es
push Frame { float a; vec2 b; } u;
void main() { gl_Position = vec4(float(gl_VertexID), u.b, u.a); }`
	fs_frame := `#version 300 es
precision highp float;
push Frame { float a; vec2 b; } u;
out vec4 col; void main() { col = vec4(u.b, u.a, 1.0); }`

	mismatches := [?]Mismatch{
		{"vs has push, fs missing", vs_frame, fs_nopush, .Push_Stage_Missing},
		{"vs missing, fs has push", vs_nopush, fs_frame, .Push_Stage_Missing},
		{
			"name mismatch",
			vs_frame,
			`#version 300 es
precision highp float;
push Other { float a; vec2 b; } u;
out vec4 col; void main() { col = vec4(u.b, u.a, 1.0); }`,
			.Push_Stage_Name,
		},
		{
			"count mismatch",
			vs_frame,
			`#version 300 es
precision highp float;
push Frame { float a; } u;
out vec4 col; void main() { col = vec4(u.a); }`,
			.Push_Stage_Count,
		},
		{
			"field name mismatch",
			vs_frame,
			`#version 300 es
precision highp float;
push Frame { float a; vec2 c; } u;
out vec4 col; void main() { col = vec4(u.c, u.a, 1.0); }`,
			.Push_Stage_Field_Name,
		},
		{
			"field type mismatch",
			vs_frame,
			`#version 300 es
precision highp float;
push Frame { float a; vec3 b; } u;
out vec4 col; void main() { col = vec4(u.b, u.a); }`,
			.Push_Stage_Field_Type,
		},
		{
			"field offset mismatch via struct internals",
			`#version 300 es
struct S { float a; float b; };
push Frame { S s; float tail; } u;
void main() { gl_Position = vec4(float(gl_VertexID), u.s.a, u.s.b, u.tail); }`,
			`#version 300 es
precision highp float;
struct S { float a; vec2 b; };
push Frame { S s; float tail; } u;
out vec4 col; void main() { col = vec4(u.s.a, u.s.b, u.tail); }`,
			.Push_Stage_Field_Offset,
		},
	}

	for mm in mismatches {
		p, errs, pok := compile_pipeline({vertex = mm.vs, fragment = mm.fs}, {layout = .Oplat})
		testing.expectf(t, !pok && has_error_code(errs, mm.want), "%s: want %v, got ok=%v errs=%s", mm.name, mm.want, pok, format_errors(errs, context.temp_allocator))
		destroy_pipeline(p)
	}
}

@(test)
test_oplat_spirv_val :: proc(t: ^testing.T) {
	// Fragment with discard (must emit SPIR-V 1.6 + OpTerminateInvocation, no OpDemote/OpKill/DescriptorSet).
	frag_discard := `#version 300 es
precision highp float;
push FragPush { mat4 mvp; vec3 light; mat2 rot; float cut; uint mode; } fp;
in vec3 v_pos;
out vec4 frag_col;
float helper_fn(float v) { return v * fp.cut + float(fp.mode); }
void main() {
	if (v_pos.x < fp.cut) discard;
	vec2 r = fp.rot * v_pos.xy;
	frag_col = fp.mvp * vec4(r, helper_fn(fp.light.x), 1.0);
}`
	// Fragment discarding from helpers inside loops and conditions (terminates in both SPIR-V and MSL).
	frag_discard_func_loop := `#version 300 es
precision highp float;
push LoopPush { int iters; float cut; } lp;
in vec2 v_uv;
out vec4 frag_col;
void maybe_discard(int i) {
	if (i == 0 && v_uv.x < lp.cut) {
		discard;
	}
}
bool check_discard(float v) {
	if (v < 0.0) {
		discard;
	}
	return v > 0.5;
}
void main() {
	float acc = 0.0;
	for (int i = 0; i < lp.iters; i++) {
		maybe_discard(i);
		if (check_discard(v_uv.y)) {
			acc += 0.25;
		}
	}
	frag_col = vec4(acc, v_uv, 1.0);
}`
	// Fragment with module-level private global and struct/array fields (without exceeding 128 B).
	frag_global := `#version 300 es
precision highp float;
struct Item { vec3 p; mat3 m; };
push P { Item items[2]; vec4 tint; } u;
float g_acc = 1.5;
out vec4 col;
void main() {
	g_acc += u.tint.x;
	col = vec4(u.items[1].m * u.items[0].p + g_acc, u.tint.w);
}`
	// Compute shader with shared memory + push block (no buffers/descriptors).
	comp_shared := `#version 310 es
layout(local_size_x = 4, local_size_y = 1) in;
push CompPush { uint base; ivec2 ofs; mat3 xform; } cp;
shared vec4 scratch[4];
void main() {
	uint i = gl_LocalInvocationID.x;
	scratch[i] = vec4(cp.xform * vec3(float(cp.base + i), float(cp.ofs.x), float(cp.ofs.y)), 1.0);
	barrier();
	vec4 r = scratch[i ^ 1u];
	if (r.x < -1000.0) { scratch[i] = r; }
}`

	Spec :: struct {
		name:           string,
		src:            string,
		stage:          Stage,
		check_term:     bool,
		check_msl_term: bool,
	}
	specs := [?]Spec{
		{"frag_discard", frag_discard, .Fragment, true, false},
		{"frag_discard_func_loop", frag_discard_func_loop, .Fragment, true, true},
		{"frag_global", frag_global, .Fragment, false, false},
		{"comp_shared", comp_shared, .Compute, false, false},
	}

	for sp in specs {
		m, errs, ok := compile(sp.src, {profile = .Glsl, stage = sp.stage, layout = .Oplat})
		testing.expectf(t, ok, "%s: %s", sp.name, format_errors(errs, context.temp_allocator))
		if ok {
			testing.expectf(t, len(m.spirv) >= 5 && m.spirv[1] == 0x00010600, "%s: spv version %08x", sp.name, m.spirv[1] if len(m.spirv) > 1 else 0)
			// DescriptorSet decoration == 34; oplat rung 1 emits no descriptor sets.
			testing.expectf(t, !spirv_has_decor(m.spirv, 34), "%s: unexpected DescriptorSet decoration", sp.name)
			if sp.check_term {
				testing.expectf(t, spirv_has_op(m.spirv, 4416), "%s: missing OpTerminateInvocation", sp.name)
				testing.expectf(t, !spirv_has_cap(m.spirv, 5379), "%s: unexpected DemoteToHelperInvocation capability", sp.name)
				testing.expectf(t, !spirv_has_op(m.spirv, 5380), "%s: emitted OpDemoteToHelperInvocation instead of OpTerminateInvocation", sp.name)
				testing.expectf(t, !spirv_has_op(m.spirv, 252), "%s: emitted OpKill instead of OpTerminateInvocation", sp.name)
			}
			if sp.check_msl_term {
				testing.expectf(t, strings.contains(m.msl, "bool _discarded;"), "%s msl missing _discarded:\n%s", sp.name, m.msl)
				testing.expectf(t, strings.contains(m.msl, "discard_fragment(); return;"), "%s msl missing void helper return:\n%s", sp.name, m.msl)
				testing.expectf(t, strings.contains(m.msl, "discard_fragment(); return {};"), "%s msl missing value helper return:\n%s", sp.name, m.msl)
				testing.expectf(t, strings.contains(m.msl, "if (g._discarded) return;"), "%s msl missing helper entry/caller guard:\n%s", sp.name, m.msl)
				testing.expectf(t, strings.contains(m.msl, "if (g._discarded) return Oksl_Out{};"), "%s msl missing entry caller guard:\n%s", sp.name, m.msl)
			}
			expect_spirv_v13(t, sp.name, m.spirv)
		}
		destroy(m)
	}
}

@(test)
test_oplat_cpu_vm :: proc(t: ^testing.T) {
	// 1) Fragment shader reading nested struct + padded mat2 + mat3 + float[3] from Odin struct bytes.
	Sub :: struct {
		m: matrix[4, 4]f32,
		a: f32,
	}
	VM_Frag_Push :: struct {
		sub:  Sub,       // offset 0, size 80 (padded from 68!)
		tag:  f32,       // offset 80
		rot2: matrix[2, 2]f32, // offset 88 (align 8!)
		arr:  [3]f32,    // offset 104
		ivec: [2]i32,    // offset 116
		uval: u32,       // offset 124 (total 128 B!)
	}
	#assert(size_of(VM_Frag_Push) == 128)

	frag_src := `#version 300 es
precision highp float;
struct Sub { mat4 m; float a; };
push VM_Frag_Push {
	Sub sub;
	float tag;
	mat2 rot2;
	float arr[3];
	ivec2 ivec;
	uint uval;
} u;
out vec4 out_col;
void main() {
	vec4 v0 = u.sub.m * vec4(1.0, 2.0, -1.0, 1.0);
	vec2 r2 = u.rot2 * vec2(u.tag, u.arr[2]);
	out_col = vec4(v0.x + u.sub.a, r2.x + float(u.ivec.y), r2.y + u.arr[1], float(u.uval));
}`
	mf, errs_f, ok_f := compile(frag_src, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
	testing.expectf(t, ok_f, "vm frag compile: %s", format_errors(errs_f, context.temp_allocator))
	defer destroy(mf)
	if ok_f {
		testing.expect_value(t, mf.push.size, size_of(VM_Frag_Push))
		push_val := VM_Frag_Push{
			sub = {
				m = matrix[4, 4]f32{
					2, 0, 0, 1,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1,
				},
				a = 0.5,
			},
			tag = 3.0,
			rot2 = matrix[2, 2]f32{
				1, 2,
				-3, 4,
			},
			arr = {10, 20, 5},
			ivec = {-7, 9},
			uval = 42,
		}
		bytes := ([^]u8)(&push_val)[:size_of(VM_Frag_Push)]
		W, H :: 6, 4
		jit_out: [W * H][4]f32
		int_out: [W * H][4]f32
		scratch: Scratch
		defer scratch_destroy(&scratch)
		run_f32(&mf.program, bytes, W, H, jit_out[:], &scratch)
		saved_body, saved_init := mf.program.jit_body, mf.program.jit_init
		mf.program.jit_body, mf.program.jit_init = nil, nil
		run_f32(&mf.program, bytes, W, H, int_out[:], &scratch)
		mf.program.jit_body, mf.program.jit_init = saved_body, saved_init

		// Expected values:
		// v0 = push_val.sub.m * [1, 2, -1, 1] -> v0.x = 2*1 + 1 = 3.0; + a(0.5) = 3.5
		// r2 = push_val.rot2 * [3, 5] -> [1*3 + 2*5, -3*3 + 4*5] = [13, 11]
		// out = {3.5, 13 + 9 = 22, 11 + 20 = 31, 42}
		want := [4]f32{3.5, 22.0, 31.0, 42.0}
		for px in 0 ..< W * H {
			testing.expect_value(t, transmute([4]u32)jit_out[px], transmute([4]u32)int_out[px])
			testing.expect_value(t, transmute([4]u32)jit_out[px], transmute([4]u32)want)
		}
	}

	// 2) Vertex + Compute under .Oplat reading mat3 & vectors from Odin struct bytes (JIT vs interpreter).
	Mat3Push :: struct {
		bias: [2]f32,
		m3:   matrix[3, 3]f32,
		tail: f32,
	}
	vs_src := `#version 300 es
push Mat3Push { vec2 bias; mat3 m3; float tail; } u;
out vec3 v_out;
void main() {
	vec3 pos = vec3(float(gl_VertexID), float(gl_VertexID) * 0.5, 1.0);
	vec3 t = u.m3 * pos;
	v_out = t + vec3(u.bias, u.tail);
	gl_Position = vec4(t, 1.0);
}`
	mv, errs_v, ok_v := compile(vs_src, {profile = .Glsl, stage = .Vertex, layout = .Oplat})
	testing.expectf(t, ok_v, "vm vert compile: %s", format_errors(errs_v, context.temp_allocator))
	defer destroy(mv)
	if ok_v {
		testing.expect_value(t, mv.push.size, size_of(Mat3Push))
		mp := Mat3Push{
			bias = {0.25, -0.5},
			m3 = matrix[3, 3]f32{
				2, 0, 1,
				0, 3, -1,
				1, 0, 4,
			},
			tail = 1.5,
		}
		bytes := ([^]u8)(&mp)[:size_of(Mat3Push)]
		N :: LANES + 2
		pos_j, pos_i: [N][4]f32
		rec := mv.varying_floats
		var_j := make([]f32, N * rec, context.temp_allocator)
		var_i := make([]f32, N * rec, context.temp_allocator)
		scratch: Scratch
		defer scratch_destroy(&scratch)
		run_vertex(&mv.program, bytes, nil, N, pos_j[:], var_j, &scratch)
		saved_body, saved_init := mv.program.jit_body, mv.program.jit_init
		mv.program.jit_body, mv.program.jit_init = nil, nil
		run_vertex(&mv.program, bytes, nil, N, pos_i[:], var_i, &scratch)
		mv.program.jit_body, mv.program.jit_init = saved_body, saved_init

		for i in 0 ..< N {
			testing.expect_value(t, transmute([4]u32)pos_j[i], transmute([4]u32)pos_i[i])
			pos_in := [3]f32{f32(i), f32(i) * 0.5, 1.0}
			exp_t := mp.m3 * pos_in
			testing.expect_value(t, pos_j[i], [4]f32{exp_t.x, exp_t.y, exp_t.z, 1.0})
			for k in 0 ..< rec {
				testing.expect_value(t, transmute(u32)var_j[i * rec + k], transmute(u32)var_i[i * rec + k])
			}
			testing.expect_value(t, [3]f32{var_j[i * rec], var_j[i * rec + 1], var_j[i * rec + 2]}, exp_t + {mp.bias.x, mp.bias.y, mp.tail})
		}
	}

	// 2b) Full 128 B All_Types fragment shader (dynamic struct array + packed vectors + mat2 + mat3).
	All_Types_Inner :: struct {
		a: [2]f32,
		b: f32,
	}
	All_Types_Push :: struct {
		v3:  [3]f32,
		m2:  matrix[2, 2]f32,
		iv:  [2]i32,
		uv:  [3]u32,
		v4:  [4]f32,
		m3:  matrix[3, 3]f32,
		arr: [2]All_Types_Inner,
	}
	#assert(size_of(All_Types_Push) == 128)
	all_types_src := `#version 300 es
precision highp float;
struct All_Types_Inner { vec2 a; float b; };
push All_Types_Push {
	vec3 v3;
	mat2 m2;
	ivec2 iv;
	uvec3 uv;
	vec4 v4;
	mat3 m3;
	All_Types_Inner arr[2];
} p;
out vec4 out_color;
void main() {
	int idx = int(gl_FragCoord.x) & 1;
	int band = int(gl_FragCoord.y) & 3;
	All_Types_Inner item = p.arr[idx];
	if (band == 0) {
		out_color = vec4(p.m3 * vec3(1.0, 0.0, 0.0) + vec3(item.a.x * 0.25, 0.0, 0.0), 1.0);
	} else if (band == 1) {
		out_color = vec4(p.m3 * vec3(0.0, 1.0, 0.0) + vec3(0.0, item.a.y * 0.25, 0.0), 1.0);
	} else if (band == 2) {
		out_color = vec4(p.m3 * vec3(0.0, 0.0, 1.0) + vec3(0.0, 0.0, item.b * 0.25), 1.0);
	} else {
		vec2 q = p.m2 * p.v3.xy;
		out_color = vec4(q.x + float(p.iv.x) * 0.05 + p.v4.w * 0.2, q.y + float(p.iv.y) * 0.05 + p.v3.z * 0.2 + float(p.uv.x) * 0.05, p.v4.x + p.v4.y * 0.25 + float(p.uv.y + p.uv.z) * 0.04 + item.b * 0.2, 1.0);
	}
}`
	ma, errs_a, ok_a := compile(all_types_src, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
	testing.expectf(t, ok_a, "all_types frag compile: %s", format_errors(errs_a, context.temp_allocator))
	defer destroy(ma)
	if ok_a {
		testing.expect_value(t, ma.push.size, size_of(All_Types_Push))
		at := All_Types_Push{
			v3  = {0.35, 0.25, 0.45},
			m2  = matrix[2, 2]f32{0.8, 0.4, 0.3, 0.9},
			iv  = {2, 3},
			uv  = {1, 2, 3},
			v4  = {0.15, 0.35, 0.2, 0.5},
			m3  = matrix[3, 3]f32{0.12, 0.45, 0.78, 0.23, 0.56, 0.89, 0.34, 0.67, 0.91},
			arr = {{a = {0.2, 0.6}, b = 0.4}, {a = {0.7, 0.1}, b = 0.8}},
		}
		bytes := ([^]u8)(&at)[:size_of(All_Types_Push)]
		AW, AH :: 4, 4
		aj, ai: [AW * AH][4]f32
		scratch: Scratch
		defer scratch_destroy(&scratch)
		run_f32(&ma.program, bytes, AW, AH, aj[:], &scratch)
		saved_body, saved_init := ma.program.jit_body, ma.program.jit_init
		ma.program.jit_body, ma.program.jit_init = nil, nil
		run_f32(&ma.program, bytes, AW, AH, ai[:], &scratch)
		ma.program.jit_body, ma.program.jit_init = saved_body, saved_init
		for py in 0 ..< AH {
			for px in 0 ..< AW {
				i := py * AW + px
				testing.expect_value(t, transmute([4]u32)aj[i], transmute([4]u32)ai[i])
			}
		}
		// Spot-check band 0, idx 1: col0 of m3 + [0.7*0.25, 0, 0]
		col0 := [4]f32{0.12 + 0.7 * 0.25, 0.23, 0.34, 1.0}
		for c in 0 ..< 4 {
			testing.expect(t, abs(aj[1][c] - col0[c]) < 1e-6)
		}
	}

	// 3) Compute shader reading Odin struct bytes.
	CompPush :: struct {
		gain: f32,
		m3:   matrix[3, 3]f32,
		k:    u32,
	}
	comp_ok_src := `#version 310 es
layout(local_size_x = 4) in;
push CompPush { float gain; mat3 m3; uint k; } u;
shared float sh[4];
void main() {
	uint id = gl_LocalInvocationID.x;
	vec3 r = u.m3 * vec3(float(id + u.k), 1.0, 2.0);
	sh[id] = r.x * u.gain;
	barrier();
}`
	mc, errs_c, ok_c := compile(comp_ok_src, {profile = .Glsl, stage = .Compute, layout = .Oplat})
	testing.expectf(t, ok_c, "vm comp compile: %s", format_errors(errs_c, context.temp_allocator))
	defer destroy(mc)
	if ok_c {
		testing.expect_value(t, mc.push.size, size_of(CompPush))
		cp := CompPush{gain = 2.0, m3 = 1, k = 3}
		bytes := ([^]u8)(&cp)[:size_of(CompPush)]
		scratch: Scratch
		defer scratch_destroy(&scratch)
		run_compute(&mc.program, bytes, {1, 1, 1}, &scratch)
		saved_body, saved_init := mc.program.jit_body, mc.program.jit_init
		mc.program.jit_body, mc.program.jit_init = nil, nil
		run_compute(&mc.program, bytes, {1, 1, 1}, &scratch)
		mc.program.jit_body, mc.program.jit_init = saved_body, saved_init
	}
}

@(test)
test_oplat_discard_derivatives :: proc(t: ^testing.T) {
	// Pins oksl's own CPU VM behavior (JIT == interpreter bit-for-bit): CPU keeps helper lanes
	// across partial-quad discard, whereas on GPUs under .Oplat `discard` terminates (OpTerminateInvocation
	// / early-return MSL) and derivatives after a non-uniform discard are undefined per GLSL.
	vs_src := `#version 300 es
push CutPush { float threshold; float scale; } u;
out float vx;
out float kill_flag;
void main() {
	vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
	vx = p.x;
	kill_flag = p.y;
	gl_Position = vec4(p, 0.0, 1.0);
}`
	fs_src := `#version 300 es
precision highp float;
push CutPush { float threshold; float scale; } u;
in float vx;
in float kill_flag;
out vec4 col;
void main() {
	if (kill_flag > u.threshold) discard;
	float dx = dFdx(vx);
	float dy = dFdy(vx);
	col = vec4(dx * u.scale, dy * u.scale, vx, 1.0);
}`
	p, errs, ok := compile_pipeline({vertex = vs_src, fragment = fs_src}, {layout = .Oplat})
	testing.expectf(t, ok, "discard deriv pipeline: %s", format_errors(errs, context.temp_allocator))
	defer destroy_pipeline(p)
	if !ok {
		return
	}
	fm := p.fragment
	vx_off, kill_off := -1, -1
	for fin in fm.stage_inputs {
		if fin.name == "vx" {vx_off = fin.offset}
		if fin.name == "kill_flag" {kill_off = fin.offset}
	}
	testing.expect(t, vx_off >= 0 && kill_off >= 0)

	CutPush :: struct {
		threshold: f32,
		scale:     f32,
	}
	cp := CutPush{threshold = 0.5, scale = 4.0}
	bytes := ([^]u8)(&cp)[:size_of(CutPush)]

	Q :: 2
	rec := fm.varying_floats
	origin := [Q][2]i32{{0, 0}, {2, 0}}
	facing := [Q]bool{true, true}
	helpers: [Q]Quad_Lanes
	varyings := make([]f32, rec * Q * 4, context.temp_allocator)
	// For each quad q, lanes (0,1,2,3) are at (x0,y0),(x0+1,y0),(x0,y0+1),(x0+1,y0+1).
	// Set vx = lane_x * 2 + lane_y * 3 -> dFdx = 2.0 everywhere, dFdy = 3.0 everywhere!
	// Discard lane 1 in quad 0 (top-right) and lane 2 in quad 1 (bottom-left).
	for q in 0 ..< Q {
		for l in 0 ..< 4 {
			j := q * 4 + l
			lx := f32(l & 1)
			ly := f32(l >> 1)
			varyings[vx_off * Q * 4 + j] = lx * 2.0 + ly * 3.0
			discard_this := (q == 0 && l == 1) || (q == 1 && l == 2)
			varyings[kill_off * Q * 4 + j] = 1.0 if discard_this else 0.0
		}
	}

	out_j, out_i: [Q * 4][4]f32
	shaded_j, shaded_i: [Q]Quad_Lanes
	tgts_j := [1][][4]f32{out_j[:]}
	tgts_i := [1][][4]f32{out_i[:]}
	scratch: Scratch
	defer scratch_destroy(&scratch)

	run_fragment(&fm.program, bytes, {count = Q, origin = origin[:], front_facing = facing[:], helpers = helpers[:], varyings = varyings}, tgts_j[:], shaded_j[:], &scratch)
	saved_body, saved_init := fm.program.jit_body, fm.program.jit_init
	fm.program.jit_body, fm.program.jit_init = nil, nil
	run_fragment(&fm.program, bytes, {count = Q, origin = origin[:], front_facing = facing[:], helpers = helpers[:], varyings = varyings}, tgts_i[:], shaded_i[:], &scratch)
	fm.program.jit_body, fm.program.jit_init = saved_body, saved_init

	testing.expect_value(t, shaded_j[0], Quad_Lanes{0, 2, 3})
	testing.expect_value(t, shaded_j[1], Quad_Lanes{0, 1, 3})
	testing.expect_value(t, shaded_j[0], shaded_i[0])
	testing.expect_value(t, shaded_j[1], shaded_i[1])

	for q in 0 ..< Q {
		for l in 0 ..< 4 {
			if l not_in shaded_j[q] {
				continue
			}
			j := q * 4 + l
			testing.expect_value(t, transmute([4]u32)out_j[j], transmute([4]u32)out_i[j])
			// dFdx = 2.0 * scale(4) = 8.0, dFdy = 3.0 * scale(4) = 12.0 even in lanes paired with discarded lane (CPU VM).
			testing.expect_value(t, out_j[j][0], f32(8.0))
			testing.expect_value(t, out_j[j][1], f32(12.0))
		}
	}

	// Terminating discard inside a helper called from a loop exits control flow in both CPU engines.
	loop_src := `#version 300 es
precision highp float;
push LoopCut { int iters; float cut; } u;
out vec4 col;
void maybe_kill(int i, float x) {
	if (i == 0 && x > u.cut) {
		discard;
	}
}
void main() {
	float acc = 1.0;
	for (int i = 0; i < u.iters; i++) {
		maybe_kill(i, gl_FragCoord.x);
		acc += 1.0;
	}
	col = vec4(acc, 0.0, 0.0, 1.0);
}`
	ml, lerrs, lok := compile(loop_src, {profile = .Glsl, stage = .Fragment, layout = .Oplat})
	testing.expectf(t, lok, "loop discard compile: %s", format_errors(lerrs, context.temp_allocator))
	defer destroy(ml)
	if lok {
		LoopCut :: struct {
			iters: i32,
			cut:   f32,
		}
		lc := LoopCut{iters = 4, cut = 1.0}
		lbytes := ([^]u8)(&lc)[:size_of(LoopCut)]
		lq := [1][4]f32{}
		// Render a 2x2 quad: x < 1 survives (acc = 5.0), x > 1 discards (unshaded).
		lo_j, lo_i: [4][4]f32
		ls_j, ls_i: [1]Quad_Lanes
		for k in 0 ..< 4 {
			lo_j[k] = {9, 9, 9, 9}
			lo_i[k] = {9, 9, 9, 9}
		}
		_ = lq
		orig := [1][2]i32{{0, 0}}
		fac := [1]bool{true}
		hlp := [1]Quad_Lanes{}
		t_j := [1][][4]f32{lo_j[:]}
		t_i := [1][][4]f32{lo_i[:]}
		run_fragment(&ml.program, lbytes, {count = 1, origin = orig[:], front_facing = fac[:], helpers = hlp[:]}, t_j[:], ls_j[:], &scratch)
		saved_b, saved_in := ml.program.jit_body, ml.program.jit_init
		ml.program.jit_body, ml.program.jit_init = nil, nil
		run_fragment(&ml.program, lbytes, {count = 1, origin = orig[:], front_facing = fac[:], helpers = hlp[:]}, t_i[:], ls_i[:], &scratch)
		ml.program.jit_body, ml.program.jit_init = saved_b, saved_in
		testing.expect_value(t, ls_j[0], Quad_Lanes{0, 2})
		testing.expect_value(t, ls_j[0], ls_i[0])
		testing.expect_value(t, lo_j[0], [4]f32{5, 0, 0, 1})
		testing.expect_value(t, lo_j[1], [4]f32{9, 9, 9, 9})
		testing.expect_value(t, lo_j, lo_i)
	}
}
